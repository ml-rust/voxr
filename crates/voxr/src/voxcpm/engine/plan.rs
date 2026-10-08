//! Planning one request on the caller's thread: resolve the voice, the
//! continuation prompt, the seed and the sampler, tokenize, and size the
//! patch budget and the prefill.
//!
//! No device work happens here, so a bad request — an unknown voice, a
//! reference or prompt from another engine, empty text, an empty prompt
//! transcript, zero solver steps — fails before it ever reaches a batch.
//!
//! A continuation row follows the reference implementation: its text is the
//! prompt transcript as given, followed with no separator by the
//! whitespace-normalized request text, and its patch budget counts the
//! request text alone.

use std::sync::Arc;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;

use crate::error::{Error, Result};
use crate::voxcpm::engine::batch::{Job, JobVoice, Sampler};
use crate::voxcpm::engine::reply::Reply;
use crate::voxcpm::engine::request::{Badcase, PromptSource, SynthRequest, VoiceSource};
use crate::voxcpm::engine::types::{
    MAX_LEN_CAP, VoxCpm2Engine, VoxCpm2RenderClient, ZERO_SHOT_VOICE_ID,
};
use crate::voxcpm::tokenizer::{is_blank, request_ids};

/// A request's continuation prompt, checked against this engine.
struct ResolvedPrompt<'a, R: Runtime> {
    /// The prompt tensor the job owns an [`Arc`] of.
    feat: Arc<Tensor<R>>,
    /// Prompt patches (`T_prompt`) the prefill appends after the text.
    patches: usize,
    /// The prompt clip's transcript.
    text: &'a str,
}

impl<R> VoxCpm2Engine<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Everything before device work: resolve the voice, the prompt and the
    /// seed, tokenize, size the budget. Runs on the caller's thread, so a
    /// bad request never joins a batch.
    ///
    /// The job owns what it reads: a table voice travels as its id, which
    /// the owner thread looks up when it builds the batch, and a request
    /// reference or prompt travels as an [`Arc`] of its tensor.
    pub(super) fn plan(&self, request: &SynthRequest<'_, R>, reply: Reply) -> Result<Job<R>> {
        let (voice, ref_len) = self.resolve_voice(request.voice)?;
        let prompt = request
            .prompt
            .map(|prompt| self.resolve_prompt(prompt))
            .transpose()?;
        if is_blank(request.text) {
            return Err(Error::InvalidArgument {
                arg: "text",
                reason: "input text must not be empty".into(),
            });
        }
        let (mut text_token_ids, target_len) = request_ids(
            &self.inner.tokenizer,
            request.text,
            prompt.as_ref().map(|prompt| prompt.text),
        );
        text_token_ids.push(AUDIO_START_ID);

        // Upstream's budget: `int(tokens * ratio + 10)` patches, capped. A
        // prompt transcript is already spoken in the prompt clip, so it adds
        // no patches.
        check_badcase(request.badcase)?;
        let budget = (target_len as f64 * request.badcase.ratio + 10.0) as usize;
        let max_len = budget.min(MAX_LEN_CAP);
        let prompt_len = prompt.as_ref().map_or(0, |prompt| prompt.patches);
        let seq_len = seq_len_for(ref_len, text_token_ids.len(), prompt_len);

        Ok(Job {
            voice,
            prompt: prompt.map(|prompt| prompt.feat),
            seed: request.seed.unwrap_or(self.inner.options.seed),
            sampler: self.sampler(request)?,
            text_token_ids,
            max_len,
            seq_len,
            target_len,
            badcase: request.badcase,
            attempt: 1,
            reply,
        })
    }

    /// The request's solver settings, each falling back to its load-time
    /// value. Zero steps and a guidance scale that is not finite are
    /// refused.
    fn sampler(&self, request: &SynthRequest<'_, R>) -> Result<Sampler> {
        let options = &self.inner.options;
        let sampler = Sampler {
            n_timesteps: request.n_timesteps.unwrap_or(options.n_timesteps),
            cfg_value: request.cfg_value.unwrap_or(options.cfg_value),
        };
        check_sampler(sampler)?;
        Ok(sampler)
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

    /// The request's prompt, checked: encoded by this engine, and naming a
    /// transcript that is not blank.
    fn resolve_prompt<'a>(&self, prompt: PromptSource<'a, R>) -> Result<ResolvedPrompt<'a, R>> {
        if prompt.prompt.engine != self.inner.id {
            return Err(Error::InvalidArgument {
                arg: "prompt",
                reason: "this prompt was encoded by a different VoxCPM2 engine; encode it \
                         again with this engine's encode_prompt"
                    .into(),
            });
        }
        if is_blank(prompt.text) {
            return Err(Error::InvalidArgument {
                arg: "prompt_text",
                reason: format!(
                    "prompt transcript {:?} holds no words; pass the words spoken in the \
                     prompt clip",
                    prompt.text
                ),
            });
        }
        Ok(ResolvedPrompt {
            feat: Arc::clone(&prompt.prompt.feat),
            patches: prompt.prompt.patches(),
            text: prompt.text,
        })
    }
}

/// Refuses zero attempts and a ratio that is not finite and positive.
fn check_badcase(badcase: Badcase) -> Result<()> {
    if badcase.max_attempts == 0 {
        return Err(Error::InvalidArgument {
            arg: "badcase.max_attempts",
            reason: "0 takes; pass at least 1, or Badcase::no_retry to retry none".into(),
        });
    }
    if !(badcase.ratio.is_finite() && badcase.ratio > 0.0) {
        return Err(Error::InvalidArgument {
            arg: "badcase.ratio",
            reason: format!(
                "ratio {} is not finite and positive; pass a ratio such as 6.0",
                badcase.ratio
            ),
        });
    }
    Ok(())
}

/// Refuses zero solver steps and a guidance scale that is not finite.
fn check_sampler(sampler: Sampler) -> Result<()> {
    if sampler.n_timesteps == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_timesteps",
            reason: "0 solver steps per patch; pass at least 1".into(),
        });
    }
    if !sampler.cfg_value.is_finite() {
        return Err(Error::InvalidArgument {
            arg: "cfg_value",
            reason: format!(
                "guidance scale {} is not finite; pass a finite scale such as 2.0",
                sampler.cfg_value
            ),
        });
    }
    Ok(())
}

/// Sequence length behind `prefill`'s cache sizing, matching boostr's
/// `SequenceLayout`: a reference prefix adds `t_ref + 2` positions before the
/// text, and zero-shot carries no reference prefix at all, not a reference
/// of zero patches. A continuation prompt adds its `prompt_len` patches after
/// the text. `text_len` is the token count AFTER `AUDIO_START_ID` is
/// appended.
fn seq_len_for(ref_len: Option<usize>, text_len: usize, prompt_len: usize) -> usize {
    let prefix = match ref_len {
        Some(t_ref) => t_ref + 2,
        None => 0,
    };
    prefix + text_len + prompt_len
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::model::audio::voxcpm::model::SequenceLayout;

    #[test]
    fn zero_attempts_and_a_bad_ratio_are_refused() {
        assert!(check_badcase(Badcase::default()).is_ok());
        assert!(check_badcase(Badcase::no_retry(6.0)).is_ok());
        let none = Badcase {
            max_attempts: 0,
            ..Badcase::default()
        };
        assert!(check_badcase(none).is_err_and(|e| e.to_string().contains("max_attempts")));
        for ratio in [0.0, -1.0, f64::NAN] {
            assert!(check_badcase(Badcase::no_retry(ratio)).is_err(), "{ratio}");
        }
    }

    #[test]
    fn zero_steps_and_a_scale_that_is_not_finite_are_refused() {
        let ok = Sampler {
            n_timesteps: 16,
            cfg_value: 1.6,
        };
        assert!(check_sampler(ok).is_ok());
        let no_steps = check_sampler(Sampler {
            n_timesteps: 0,
            ..ok
        });
        assert!(no_steps.is_err_and(|e| e.to_string().contains("n_timesteps")));
        let nan = check_sampler(Sampler {
            cfg_value: f32::NAN,
            ..ok
        });
        assert!(nan.is_err_and(|e| e.to_string().contains("cfg_value")));
    }

    #[test]
    fn seq_len_for_with_reference_adds_t_ref_plus_two() {
        assert_eq!(seq_len_for(Some(40), 12, 0), 40 + 2 + 12);
    }

    #[test]
    fn seq_len_for_zero_shot_is_text_len_only() {
        assert_eq!(seq_len_for(None, 12, 0), 12);
    }

    /// A prompt tail adds its patches after the text, with or without a
    /// reference prefix.
    #[test]
    fn seq_len_for_adds_the_prompt_tail() {
        assert_eq!(seq_len_for(Some(40), 12, 9), 40 + 2 + 12 + 9);
        assert_eq!(seq_len_for(None, 12, 9), 12 + 9);
    }

    /// Every mode agrees with the layout the prefill builds, which refuses
    /// a cache length that differs from it.
    #[test]
    fn seq_len_for_matches_the_prefill_layout() {
        let text = [11u32, 22, AUDIO_START_ID];
        for (ref_len, prompt_len) in [(None, 0), (Some(5), 0), (None, 3), (Some(5), 3)] {
            let layout =
                SequenceLayout::build(ref_len.unwrap_or(0), &text, prompt_len).expect("layout");
            assert_eq!(
                seq_len_for(ref_len, text.len(), prompt_len),
                layout.seq_len(),
                "ref {ref_len:?}, prompt {prompt_len}"
            );
        }
    }
}
