//! Planning one request on the caller's thread: resolve the voice, the
//! continuation prompt and the seed, tokenize, and size the patch budget and
//! the prefill.
//!
//! No device work happens here, so a bad request — an unknown voice, a
//! reference or prompt from another engine, empty text, an empty prompt
//! transcript — fails before it ever reaches a batch.
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
use crate::voxcpm::engine::batch::{Job, JobVoice};
use crate::voxcpm::engine::reply::Reply;
use crate::voxcpm::engine::request::{PromptSource, SynthRequest, VoiceSource};
use crate::voxcpm::engine::types::{
    MAX_LEN_CAP, VoxCpm2Engine, VoxCpm2RenderClient, ZERO_SHOT_VOICE_ID,
};
use crate::voxcpm::tokenizer::{normalize_whitespace, tokenize};

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
        let target = normalize_whitespace(request.text);
        if target.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "text",
                reason: "input text must not be empty".into(),
            });
        }
        let tokenizer = &self.inner.tokenizer;
        let (mut text_token_ids, target_len) = match &prompt {
            None => {
                let ids = tokenize(tokenizer, &target);
                let len = ids.len();
                (ids, len)
            }
            // The reference implementation normalizes the request text only;
            // the transcript joins it unchanged.
            Some(prompt) => (
                tokenizer.encode_raw(&format!("{}{target}", prompt.text)),
                tokenize(tokenizer, &target).len(),
            ),
        };
        text_token_ids.push(AUDIO_START_ID);

        // The clone pipeline's budget: six patches per request-text token
        // plus ten, capped. A prompt transcript is already spoken in the
        // prompt clip, so it adds no patches.
        let max_len = (target_len * 6 + 10).min(MAX_LEN_CAP);
        let prompt_len = prompt.as_ref().map_or(0, |prompt| prompt.patches);
        let seq_len = seq_len_for(ref_len, text_token_ids.len(), prompt_len);

        Ok(Job {
            voice,
            prompt: prompt.map(|prompt| prompt.feat),
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
        if normalize_whitespace(prompt.text).is_empty() {
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
