//! The caller's half of a request: plan it (`super::plan`), queue the job,
//! and block on the reply.
//!
//! Everything here runs on the caller's thread. The generation itself runs
//! on the engine's owner thread (`super::batch`, `super::run`), which groups
//! whatever is queued at the same time into ONE generation.
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
use boostr::model::audio::voxcpm::vae::decoder::SAMPLE_RATE;

use crate::error::{Error, Result};
use crate::lang::Lang;
use crate::tts::{TtsEngine, Voice};
use crate::voxcpm::engine::batch::Work;
use crate::voxcpm::engine::reply::Reply;
use crate::voxcpm::engine::request::{Rendered, SynthRequest, VoiceSource};
use crate::voxcpm::engine::types::{VoxCpm2Engine, VoxCpm2RenderClient, ZERO_SHOT_VOICE_ID};

impl<R> VoxCpm2Engine<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
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
    /// with any others. A continuation request's waveform holds only its new
    /// speech, never the prompt clip.
    pub fn render_request(&self, request: &SynthRequest<'_, R>) -> Result<Vec<f32>> {
        Ok(self.render_request_with_attempts(request)?.samples)
    }

    /// [`render_request`](Self::render_request), with the takes generated:
    /// a take that runs on past its words is generated again as the
    /// request's [`Badcase`](crate::voxcpm::Badcase) says.
    pub fn render_request_with_attempts(&self, request: &SynthRequest<'_, R>) -> Result<Rendered> {
        let (done, reply) = channel();
        self.submit(Work::Render(self.plan(request, Reply::Buffered(done))?))?;
        reply.recv().map_err(|_| worker_gone())?
    }

    /// Chunked render of one request: queue it, then drain the decoded
    /// chunks into `sink` as the owner thread produces them. The sender
    /// drops when the row ends, which is how the last chunk is recognised.
    ///
    /// A streamed take is never retried, as upstream streams one take. The
    /// chunks concatenate to the waveform
    /// [`render_request`](Self::render_request) returns for its first take,
    /// so a continuation request streams no prompt audio either.
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

    /// Whole-utterance render in a voice-table voice at the load-time seed,
    /// a bad take retried as upstream's non-streaming `generate` retries it.
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

/// `synthesize` retries a bad take ([`Badcase`](crate::voxcpm::Badcase)'s
/// defaults), as upstream's non-streaming `generate` does.
/// `synthesize_stream` streams one take and never retries it, as upstream
/// streams. The chunks therefore concatenate to the request's first take,
/// which is what `synthesize` returns when that take is not bad.
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

#[cfg(test)]
mod tests {
    use super::*;

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
