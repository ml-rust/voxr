//! The request-level surface: a [`SynthRequest`] names its text, its voice,
//! its optional continuation prompt and its seed. A [`Reference`] is a voice
//! and a [`Prompt`] is a continuation clip, each encoded once from samples
//! the caller holds and reused across requests. `super::encode` holds the
//! encode methods.

use std::sync::Arc;

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A reference voice encoded by one
/// [`VoxCpm2Engine`](crate::voxcpm::VoxCpm2Engine).
///
/// Holds the `[T_ref, patch_size, feat_dim]` tensor behind an [`Arc`], so a
/// clone is cheap and every request using it shares one tensor. A queued
/// request owns its own clone, so the reference lives until the last
/// request using it is done. Only the engine that encoded it accepts it.
pub struct Reference<R: Runtime> {
    pub(super) feat: Arc<Tensor<R>>,
    /// The id of the engine that encoded it.
    pub(super) engine: u64,
}

impl<R: Runtime> Reference<R> {
    /// Reference patches (`T_ref`) this reference adds to a prefill.
    pub fn patches(&self) -> usize {
        self.feat.shape().first().copied().unwrap_or(0)
    }
}

impl<R: Runtime> Clone for Reference<R> {
    fn clone(&self) -> Self {
        Self {
            feat: Arc::clone(&self.feat),
            engine: self.engine,
        }
    }
}

/// A continuation ("Hi-Fi") prompt clip encoded by one
/// [`VoxCpm2Engine`](crate::voxcpm::VoxCpm2Engine).
///
/// Holds the `[T_prompt, patch_size, feat_dim]` tensor behind an [`Arc`],
/// left-padded so the clip's real audio ends at its last patch. A request
/// that names it continues from that clip: the prompt patches sit after the
/// text in the prefill, and the output starts where the clip ends. Only the
/// engine that encoded it accepts it.
pub struct Prompt<R: Runtime> {
    pub(super) feat: Arc<Tensor<R>>,
    /// The id of the engine that encoded it.
    pub(super) engine: u64,
}

impl<R: Runtime> Prompt<R> {
    /// Prompt patches (`T_prompt`) this prompt adds to a prefill.
    pub fn patches(&self) -> usize {
        self.feat.shape().first().copied().unwrap_or(0)
    }
}

impl<R: Runtime> Clone for Prompt<R> {
    fn clone(&self) -> Self {
        Self {
            feat: Arc::clone(&self.feat),
            engine: self.engine,
        }
    }
}

/// A request's continuation prompt: the encoded clip and the words spoken
/// in it.
///
/// The model reads `text` + the request's text as one transcript, so `text`
/// is the clip's exact transcript, not a description of it.
pub struct PromptSource<'a, R: Runtime> {
    /// A prompt from
    /// [`VoxCpm2Engine::encode_prompt`](crate::voxcpm::VoxCpm2Engine::encode_prompt)
    /// on the same engine.
    pub prompt: &'a Prompt<R>,
    /// The transcript of the prompt clip.
    pub text: &'a str,
}

impl<'a, R: Runtime> PromptSource<'a, R> {
    /// `prompt`, whose clip speaks `text`.
    pub fn new(prompt: &'a Prompt<R>, text: &'a str) -> Self {
        Self { prompt, text }
    }
}

impl<R: Runtime> Clone for PromptSource<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for PromptSource<'_, R> {}

/// The voice one request renders with.
pub enum VoiceSource<'a, R: Runtime> {
    /// An id from [`TtsEngine::voices`](crate::tts::TtsEngine::voices).
    /// [`ZERO_SHOT_VOICE_ID`](crate::voxcpm::ZERO_SHOT_VOICE_ID) selects
    /// zero-shot, the same as [`VoiceSource::ZeroShot`].
    Named(&'a str),
    /// A reference from
    /// [`VoxCpm2Engine::encode_reference`](crate::voxcpm::VoxCpm2Engine::encode_reference)
    /// on the same engine.
    Reference(&'a Reference<R>),
    /// No reference recording: the model's own voice, or one a style prefix
    /// designs.
    ZeroShot,
}

impl<R: Runtime> Clone for VoiceSource<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for VoiceSource<'_, R> {}

/// One synthesis request.
///
/// A style or voice-design prefix is part of `text`: compose it with
/// [`paren`](crate::voxcpm::paren), as `paren(style) + text`.
pub struct SynthRequest<'a, R: Runtime> {
    /// The text to speak, any style prefix included.
    pub text: &'a str,
    /// The voice to render with.
    pub voice: VoiceSource<'a, R>,
    /// The clip this request continues from, `None` for no continuation.
    ///
    /// Hi-Fi cloning names the same recording twice: as the voice, through
    /// [`VoiceSource::Reference`], and here. The request's output holds
    /// only the new speech, never the prompt clip.
    pub prompt: Option<PromptSource<'a, R>>,
    /// This request's base seed. `None` takes
    /// [`VoxCpm2SynthOptions::seed`](crate::voxcpm::VoxCpm2SynthOptions::seed)
    /// from load time.
    pub seed: Option<u64>,
}

impl<'a, R: Runtime> SynthRequest<'a, R> {
    /// A request for `text` in `voice` at the load-time seed, with no
    /// continuation prompt.
    pub fn new(text: &'a str, voice: VoiceSource<'a, R>) -> Self {
        Self {
            text,
            voice,
            prompt: None,
            seed: None,
        }
    }

    /// The same request continuing from `prompt`, whose clip speaks
    /// `prompt_text`.
    pub fn with_prompt(self, prompt: &'a Prompt<R>, prompt_text: &'a str) -> Self {
        Self {
            prompt: Some(PromptSource::new(prompt, prompt_text)),
            ..self
        }
    }

    /// The same request at base seed `seed`.
    pub fn with_seed(self, seed: u64) -> Self {
        Self {
            seed: Some(seed),
            ..self
        }
    }
}

impl<R: Runtime> Clone for SynthRequest<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for SynthRequest<'_, R> {}
