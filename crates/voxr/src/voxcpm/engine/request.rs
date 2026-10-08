//! The request-level surface: a [`SynthRequest`] names its text, its voice,
//! its optional continuation prompt, its seed and its sampler settings. A [`Reference`] is a voice
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

/// How a buffered request retries a bad take, as VoxCPM's
/// `retry_badcase` does.
///
/// A take is bad when it generates at least `ratio` patches per token of
/// the request text: it ran on past its words. A bad take is generated again
/// with fresh noise, up to `max_attempts` takes in all, and the last take is
/// kept when every one is bad. `ratio` also sets the patch budget:
/// `ratio * tokens + 10` patches, capped. A streamed request is never
/// retried, as upstream never retries one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Badcase {
    /// Takes generated at most, the first included. `1` retries none.
    pub max_attempts: usize,
    /// Patches per request-text token at which a take is bad.
    pub ratio: f64,
}

impl Badcase {
    /// Upstream's attempts when it retries.
    pub const MAX_ATTEMPTS: usize = 3;
    /// Upstream's `retry_badcase_ratio_threshold`.
    pub const RATIO: f64 = 6.0;

    /// No retry: one take, the budget still read from `ratio`.
    pub fn no_retry(ratio: f64) -> Self {
        Self {
            max_attempts: 1,
            ratio,
        }
    }
}

/// Upstream `VoxCPM.generate`'s defaults: 3 takes at most, ratio 6.
impl Default for Badcase {
    fn default() -> Self {
        Self {
            max_attempts: Self::MAX_ATTEMPTS,
            ratio: Self::RATIO,
        }
    }
}

/// A rendered request: its waveform, and the takes generated to make it.
#[derive(Debug, Clone, PartialEq)]
pub struct Rendered {
    /// The kept take's samples.
    pub samples: Vec<f32>,
    /// Takes generated, the kept one included: 1 when the first was good.
    pub attempts: usize,
}

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
    /// This request's classifier-free guidance scale. `None` takes
    /// [`VoxCpm2SynthOptions::cfg_value`](crate::voxcpm::VoxCpm2SynthOptions::cfg_value)
    /// from load time.
    pub cfg_value: Option<f32>,
    /// This request's flow-matching solver steps per patch. `None` takes
    /// [`VoxCpm2SynthOptions::n_timesteps`](crate::voxcpm::VoxCpm2SynthOptions::n_timesteps)
    /// from load time.
    pub n_timesteps: Option<usize>,
    /// How a bad take is retried. Defaults to upstream's 3 takes at ratio 6.
    pub badcase: Badcase,
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
            cfg_value: None,
            n_timesteps: None,
            badcase: Badcase::default(),
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

    /// The same request at guidance scale `cfg_value`.
    pub fn with_cfg(self, cfg_value: f32) -> Self {
        Self {
            cfg_value: Some(cfg_value),
            ..self
        }
    }

    /// The same request retrying bad takes as `badcase` says.
    pub fn with_badcase(self, badcase: Badcase) -> Self {
        Self { badcase, ..self }
    }

    /// The same request at `n_timesteps` solver steps per patch.
    pub fn with_steps(self, n_timesteps: usize) -> Self {
        Self {
            n_timesteps: Some(n_timesteps),
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
