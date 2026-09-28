//! The caller's half of a request: resolve the voice and the seed,
//! tokenize, size the patch budget, queue the job, and block on the reply.
//!
//! Everything here runs on the caller's thread. No device work happens on
//! it, so a bad request — an unknown voice, a reference from another engine,
//! empty text, a speed the model cannot serve — fails before it ever reaches
//! a batch. The generation itself runs on the engine's owner thread
//! (`super::batch`, `super::run`), which groups whatever is queued at the
//! same time into ONE generation.
//!
//! A streaming caller keeps its sink here and drains decoded chunks off a
//! channel: the sink is not `Send`, so it must not cross to the owner
//! thread. A sink error sets the job's cancel flag, which retires that row
//! of the running batch.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;
use boostr::model::audio::voxcpm::vae::decoder::SAMPLE_RATE;

use crate::error::{Error, Result};
use crate::lang::Lang;
use crate::tts::{TtsEngine, Voice};
use crate::voxcpm::engine::batch::{Job, JobVoice, Work};
use crate::voxcpm::engine::reply::Reply;
use crate::voxcpm::engine::request::{SynthRequest, VoiceSource};
use crate::voxcpm::engine::types::{
    MAX_LEN_CAP, VoxCpm2Engine, VoxCpm2RenderClient, ZERO_SHOT_VOICE_ID,
};
use crate::voxcpm::tokenizer::{normalize_whitespace, tokenize};

impl<R> VoxCpm2Engine<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Everything before device work: resolve the voice and the seed,
    /// tokenize, size the budget. Runs on the caller's thread, so a bad
    /// request never joins a batch.
    ///
    /// The job owns what it reads: a table voice travels as its id, which
    /// the owner thread looks up when it builds the batch, and a request
    /// reference travels as an [`Arc`] of its tensor.
    fn plan(&self, request: &SynthRequest<'_, R>, reply: Reply) -> Result<Job<R>> {
        let (voice, ref_len) = self.resolve_voice(request.voice)?;
        let normalized = normalize_whitespace(request.text);
        if normalized.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "text",
                reason: "input text must not be empty".into(),
            });
        }
        let mut text_token_ids = tokenize(&self.inner.tokenizer, &normalized);
        let text_len = text_token_ids.len();
        text_token_ids.push(AUDIO_START_ID);

        // The clone pipeline's budget: six patches per text token plus ten,
        // capped.
        let max_len = (text_len * 6 + 10).min(MAX_LEN_CAP);
        let seq_len = seq_len_for(ref_len, text_token_ids.len());

        Ok(Job {
            voice,
            seed: request.seed.unwrap_or(self.inner.options.seed),
            text_token_ids,
            max_len,
            seq_len,
            reply,
        })
    }

    /// The job's voice and its reference length in patches, `None` for
    /// zero-shot. An id outside the voice table, or a reference another
    /// engine encoded, is refused here.
    fn resolve_voice(&self, voice: VoiceSource<'_, R>) -> Result<(JobVoice<R>, Option<usize>)> {
        match voice {
            VoiceSource::ZeroShot => Ok((JobVoice::ZeroShot, None)),
            VoiceSource::Named(id) if id == ZERO_SHOT_VOICE_ID => Ok((JobVoice::ZeroShot, None)),
            VoiceSource::Named(id) => {
                let voice = self
                    .inner
                    .voices
                    .get(id)
                    .ok_or_else(|| Error::InvalidArgument {
                        arg: "voice",
                        reason: format!(
                            "unknown voice {id:?}; available voices: [{}], or {:?} for \
                             zero-shot rendering",
                            self.inner
                                .voices
                                .keys()
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(", "),
                            ZERO_SHOT_VOICE_ID,
                        ),
                    })?;
                let patches = voice.ref_feat.shape().first().copied().unwrap_or(0);
                Ok((JobVoice::Named(id.to_string()), Some(patches)))
            }
            VoiceSource::Reference(reference) => {
                if reference.engine != self.inner.id {
                    return Err(Error::InvalidArgument {
                        arg: "voice",
                        reason: "this reference was encoded by a different VoxCPM2 engine; \
                                 encode it again with this engine's encode_reference"
                            .into(),
                    });
                }
                Ok((
                    JobVoice::Reference(Arc::clone(&reference.feat)),
                    Some(reference.patches()),
                ))
            }
        }
    }

    /// Queue work for the owner thread.
    pub(super) fn submit(&self, work: Work<R>) -> Result<()> {
        let Some(jobs) = self.jobs.as_ref() else {
            return Err(worker_gone());
        };
        jobs.send(work).map_err(|_| worker_gone())
    }

    /// Whole-utterance render of one request: queue it, wait for its
    /// waveform.
    ///
    /// Equal requests render equal samples on one backend, alone or batched
    /// with any others.
    pub fn render_request(&self, request: &SynthRequest<'_, R>) -> Result<Vec<f32>> {
        let (done, reply) = channel();
        self.submit(Work::Render(self.plan(request, Reply::Buffered(done))?))?;
        reply.recv().map_err(|_| worker_gone())?
    }

    /// Chunked render of one request: queue it, then drain the decoded
    /// chunks into `sink` as the owner thread produces them. The sender
    /// drops when the row ends, which is how the last chunk is recognised.
    ///
    /// A `sink` error cancels this row and returns at once; the batch runs
    /// on for the other rows.
    pub fn render_request_stream(
        &self,
        request: &SynthRequest<'_, R>,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        if self.inner.options.stream_chunk_patches == 0 {
            return Err(Error::InvalidArgument {
                arg: "stream_chunk_patches",
                reason: "expected at least 1 patch per chunk, got 0".into(),
            });
        }
        let (chunks, reply) = channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let job = self.plan(
            request,
            Reply::Streaming {
                chunks,
                cancelled: Arc::clone(&cancelled),
            },
        )?;
        self.submit(Work::Render(job))?;

        let mut delivered = false;
        while let Ok(message) = reply.recv() {
            let samples = message?;
            if let Err(err) = sink(&samples) {
                cancelled.store(true, Ordering::Relaxed);
                return Err(err);
            }
            delivered = true;
        }
        // Every render emits at least one patch, so an empty stream means
        // the owner thread died without reporting anything.
        if delivered {
            Ok(())
        } else {
            Err(worker_gone())
        }
    }

    /// Whole-utterance render in a voice-table voice at the load-time seed.
    fn render(&self, text: &str, voice_id: &str) -> Result<Vec<f32>> {
        self.render_request(&SynthRequest::new(text, VoiceSource::Named(voice_id)))
    }

    /// Chunked render in a voice-table voice at the load-time seed.
    fn render_stream(
        &self,
        text: &str,
        voice_id: &str,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        let request = SynthRequest::new(text, VoiceSource::Named(voice_id));
        self.render_request_stream(&request, sink)
    }
}

impl<R> TtsEngine for VoxCpm2Engine<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        check_speed(speed)?;
        self.render(text, voice)
    }

    fn synthesize_stream(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
        sink: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        check_speed(speed)?;
        self.render_stream(text, voice, sink)
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    /// [`ZERO_SHOT_VOICE_ID`] first, then one entry per reference recording.
    /// VoxCPM2 takes raw text in any language it was trained on, so every
    /// voice carries the product language tag rather than a per-voice one.
    fn voices(&self) -> Vec<Voice> {
        std::iter::once(Voice::new(ZERO_SHOT_VOICE_ID, Lang::Ms, ZERO_SHOT_VOICE_ID))
            .chain(
                self.inner
                    .voices
                    .keys()
                    .map(|id| Voice::new(id.clone(), Lang::Ms, id.clone())),
            )
            .collect()
    }
}

/// The owner thread is gone: it panicked, or the engine is shutting down.
/// Every waiting caller gets this instead of blocking forever.
pub(super) fn worker_gone() -> Error {
    Error::ModelError {
        reason: "VoxCPM2 render thread is not running; an earlier render panicked or the \
                 engine is shutting down"
            .into(),
    }
}

/// The model has no rate control; a silently ignored `speed` would return
/// audio the caller did not ask for.
fn check_speed(speed: f32) -> Result<()> {
    if speed != 1.0 {
        return Err(Error::InvalidArgument {
            arg: "speed",
            reason: format!("VoxCPM2 renders at its natural pace only; got {speed}"),
        });
    }
    Ok(())
}

/// Sequence length behind `prefill`'s cache sizing. Mirrors `clone.rs`'s
/// zero-shot/reference branch exactly (see its `SequenceLayout` docs): a
/// reference prefix adds `t_ref + 2` positions before the text; zero-shot
/// carries no reference prefix at all, not a reference of zero patches.
/// `text_len` is the token count AFTER `AUDIO_START_ID` is appended.
fn seq_len_for(ref_len: Option<usize>, text_len: usize) -> usize {
    match ref_len {
        Some(t_ref) => t_ref + 2 + text_len,
        None => text_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seq_len_for_with_reference_adds_t_ref_plus_two() {
        assert_eq!(seq_len_for(Some(40), 12), 40 + 2 + 12);
    }

    #[test]
    fn seq_len_for_zero_shot_is_text_len_only() {
        assert_eq!(seq_len_for(None, 12), 12);
    }

    #[test]
    fn speed_other_than_natural_is_refused() {
        assert!(check_speed(1.0).is_ok());
        let err = check_speed(1.5).unwrap_err();
        assert!(err.to_string().contains("1.5"), "{err}");
    }

    #[test]
    fn a_dead_worker_is_an_error_not_a_hang() {
        let err = worker_gone();
        assert!(err.to_string().contains("not running"), "{err}");
    }
}
