//! The request-level surface: a [`SynthRequest`] names its text, its voice
//! and its seed, and a [`Reference`] is a voice encoded once from samples
//! the caller holds and reused across requests.
//!
//! A reference encode runs on the engine's owner thread, like every other
//! use of the model. The caller resamples on its own thread, queues the
//! samples and blocks. The owner thread serves the encode between two
//! batches, or between two steps of a persistent batch, so it never lands
//! inside a running step.

use std::sync::Arc;
use std::sync::mpsc::channel;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::VoxCpm2Model;

use crate::error::{Error, Result};
use crate::resample::resample;
use crate::voxcpm::engine::batch::{EncodeJob, Work};
use crate::voxcpm::engine::render::worker_gone;
use crate::voxcpm::engine::types::{EngineInner, REF_RATE, VoxCpm2Engine, VoxCpm2RenderClient};

/// A reference voice encoded by one [`VoxCpm2Engine`].
///
/// Holds the `[T_ref, patch_size, feat_dim]` tensor behind an [`Arc`], so a
/// clone is cheap and every request using it shares one tensor. A queued
/// request owns its own clone, so the reference lives until the last
/// request using it is done. Only the engine that encoded it accepts it.
pub struct Reference<R: Runtime> {
    pub(super) feat: Arc<Tensor<R>>,
    /// [`EngineInner::id`] of the engine that encoded it.
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

/// The voice one request renders with.
pub enum VoiceSource<'a, R: Runtime> {
    /// An id from [`TtsEngine::voices`](crate::tts::TtsEngine::voices).
    /// [`ZERO_SHOT_VOICE_ID`](crate::voxcpm::ZERO_SHOT_VOICE_ID) selects
    /// zero-shot, the same as [`VoiceSource::ZeroShot`].
    Named(&'a str),
    /// A reference from [`VoxCpm2Engine::encode_reference`] on the same
    /// engine.
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
    /// This request's base seed. `None` takes
    /// [`VoxCpm2SynthOptions::seed`](crate::voxcpm::VoxCpm2SynthOptions::seed)
    /// from load time.
    pub seed: Option<u64>,
}

impl<'a, R: Runtime> SynthRequest<'a, R> {
    /// A request for `text` in `voice` at the load-time seed.
    pub fn new(text: &'a str, voice: VoiceSource<'a, R>) -> Self {
        Self {
            text,
            voice,
            seed: None,
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

impl<R> VoxCpm2Engine<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Encode a reference recording once, for any number of requests to
    /// name through [`VoiceSource::Reference`].
    ///
    /// `samples` is mono PCM at `sample_rate`. It is resampled to the
    /// encoder's rate here, on the caller's thread, with the resampler the
    /// voices directory goes through at load. The same samples therefore
    /// encode to the same tensor a voices-directory file does.
    ///
    /// The encode itself runs on the owner thread. This call blocks until
    /// the owner thread reaches it: after the running batch, or after the
    /// running step of a persistent batch.
    pub fn encode_reference(&self, samples: &[f32], sample_rate: u32) -> Result<Reference<R>> {
        if samples.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "samples",
                reason: format!(
                    "reference recording at {sample_rate} Hz has 0 samples; pass mono PCM \
                     holding the reference voice"
                ),
            });
        }
        let samples = resample(samples, sample_rate, REF_RATE)?;
        let (reply, encoded) = channel();
        self.submit(Work::Encode(EncodeJob { samples, reply }))?;
        let feat = encoded.recv().map_err(|_| worker_gone())??;
        Ok(Reference {
            feat: Arc::new(feat),
            engine: self.inner.id,
        })
    }
}

impl<R> EngineInner<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Run one queued reference encode and reply with its tensor. Runs on
    /// the owner thread only. A caller that already went away is not an
    /// error in turn, so the send result is dropped.
    pub(super) fn serve_encode(&self, job: EncodeJob<R>) {
        let encoded = self
            .model
            .encode_reference(self.client.as_ref(), &job.samples)
            .map_err(Error::from);
        let _ = job.reply.send(encoded);
    }
}
