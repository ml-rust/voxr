//! Encoding caller-held recordings: a [`Reference`] voice, a continuation
//! [`Prompt`], or both from one clip for Hi-Fi cloning.
//!
//! An encode runs on the engine's owner thread, like every other use of the
//! model. The caller resamples on its own thread, queues the samples with
//! the padding it wants and blocks. The owner thread serves the encode
//! between two batches, or between two steps of a persistent batch, so it
//! never lands inside a running step.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::{PadSide, VoxCpm2Model};

use crate::error::{Error, Result};
use crate::resample::resample_like_librosa;
use crate::voxcpm::engine::batch::{EncodeJob, Work};
use crate::voxcpm::engine::render::worker_gone;
use crate::voxcpm::engine::request::{Prompt, Reference};
use crate::voxcpm::engine::types::{EngineInner, REF_RATE, VoxCpm2Engine, VoxCpm2RenderClient};

impl<R> VoxCpm2Engine<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Encode a reference recording once, for any number of requests to
    /// name through [`VoiceSource::Reference`](crate::voxcpm::VoiceSource::Reference).
    ///
    /// `samples` is mono PCM at `sample_rate`. It is resampled to the
    /// encoder's rate here, on the caller's thread, with the resampler the
    /// voices directory goes through at load. The same samples therefore
    /// encode to the same tensor a voices-directory file does. A file read
    /// with [`decode_audio_like_ffmpeg`](crate::decode_audio_like_ffmpeg)
    /// gives the samples the voices directory and `librosa.load` read. The
    /// encode right-pads the clip to whole patches.
    ///
    /// The encode itself runs on the owner thread. This call blocks until
    /// the owner thread reaches it: after the running batch, or after the
    /// running step of a persistent batch.
    pub fn encode_reference(&self, samples: &[f32], sample_rate: u32) -> Result<Reference<R>> {
        let samples = to_encoder_rate(samples, sample_rate, "reference")?;
        let feat = wait_encoded(self.queue_encode(samples, PadSide::Right)?)?;
        Ok(Reference {
            feat: Arc::new(feat),
            engine: self.inner.id,
        })
    }

    /// Encode a continuation prompt clip once, for any number of requests
    /// to name through
    /// [`SynthRequest::prompt`](crate::voxcpm::SynthRequest::prompt).
    ///
    /// Resamples as [`encode_reference`](Self::encode_reference) does. The
    /// encode left-pads the clip to whole patches, so its real audio ends at
    /// its last patch. Blocks on the owner thread the same way.
    pub fn encode_prompt(&self, samples: &[f32], sample_rate: u32) -> Result<Prompt<R>> {
        let samples = to_encoder_rate(samples, sample_rate, "prompt")?;
        let feat = wait_encoded(self.queue_encode(samples, PadSide::Left)?)?;
        Ok(Prompt {
            feat: Arc::new(feat),
            engine: self.inner.id,
        })
    }

    /// Encode one clip as both a reference and a continuation prompt: the
    /// pair a Hi-Fi request names.
    ///
    /// Resamples once. The two encodes differ in padding, so they are two
    /// VAE encodes, queued together. The pair equals
    /// [`encode_reference`](Self::encode_reference) and
    /// [`encode_prompt`](Self::encode_prompt) on the same samples.
    pub fn encode_reference_and_prompt(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<(Reference<R>, Prompt<R>)> {
        let samples = to_encoder_rate(samples, sample_rate, "Hi-Fi")?;
        let reference = self.queue_encode(samples.clone(), PadSide::Right)?;
        let prompt = self.queue_encode(samples, PadSide::Left)?;
        let reference = Reference {
            feat: Arc::new(wait_encoded(reference)?),
            engine: self.inner.id,
        };
        let prompt = Prompt {
            feat: Arc::new(wait_encoded(prompt)?),
            engine: self.inner.id,
        };
        Ok((reference, prompt))
    }

    /// Queue one encode of encoder-rate `samples`, padded on `side`, and
    /// return the channel its tensor arrives on.
    fn queue_encode(
        &self,
        samples: Vec<f32>,
        side: PadSide,
    ) -> Result<Receiver<Result<Tensor<R>>>> {
        let (reply, encoded) = channel();
        self.submit(Work::Encode(EncodeJob {
            samples,
            side,
            reply,
        }))?;
        Ok(encoded)
    }
}

impl<R> EngineInner<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Run one queued encode and reply with its tensor: a right-padded
    /// reference encode or a left-padded prompt encode, as the job names.
    /// Runs on the owner thread only. A caller that already went away is not
    /// an error in turn, so the send result is dropped.
    pub(super) fn serve_encode(&self, job: EncodeJob<R>) {
        let client = self.client.as_ref();
        let encoded = match job.side {
            PadSide::Right => self.model.encode_reference(client, &job.samples),
            PadSide::Left => self.model.encode_prompt(client, &job.samples),
        };
        let _ = job.reply.send(encoded.map_err(Error::from));
    }
}

/// `samples` at `sample_rate`, resampled to the encoder's rate the way the
/// reference's `librosa.load(path, sr=16000)` resamples, by
/// [`resample_like_librosa`]. `what` names the recording in the error for an
/// empty clip.
fn to_encoder_rate(samples: &[f32], sample_rate: u32, what: &str) -> Result<Vec<f32>> {
    if samples.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: format!(
                "{what} recording at {sample_rate} Hz has 0 samples; pass mono PCM holding \
                 the {what} clip"
            ),
        });
    }
    resample_like_librosa(samples, sample_rate, REF_RATE)
}

/// Block for one queued encode's tensor. A closed channel means the owner
/// thread is gone.
fn wait_encoded<R: Runtime>(encoded: Receiver<Result<Tensor<R>>>) -> Result<Tensor<R>> {
    encoded.recv().map_err(|_| worker_gone())?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty clip is refused before any resample or queue, naming the
    /// clip and its rate.
    #[test]
    fn an_empty_clip_is_refused_with_its_name_and_rate() {
        let err = to_encoder_rate(&[], 44_100, "prompt").unwrap_err();
        let text = err.to_string();
        assert!(text.contains("prompt"), "{text}");
        assert!(text.contains("44100"), "{text}");
    }

    /// A clip already at the encoder's rate passes through at its length.
    #[test]
    fn a_clip_at_the_encoder_rate_keeps_its_length() {
        let clip = vec![0.25f32; 1600];
        let out = to_encoder_rate(&clip, REF_RATE, "reference").expect("resample");
        assert_eq!(out.len(), clip.len());
    }
}
