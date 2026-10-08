//! voxr's VoxCPM2 engine as a [`SpeechEngine`]: one take per request, its
//! seed, guidance and steps its own, its voice designed, named or cloned,
//! and a clone's reference encoded once and kept.

use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use sha1::{Digest, Sha1};
use voxr::TtsEngine;
use voxr::voxcpm::{
    Prompt, Reference, SynthRequest, VoiceSource as VoxVoice, VoxCpm2Engine, VoxCpm2RenderClient,
};

use super::seed::engine_seed;
use super::voxcpm_text::voxcpm_text;
use crate::error::{Error, Result};
use crate::take::{CloneReference, SpeakRequest, SpeechEngine, Take, VoiceSource};

/// Recordings a [`VoxcpmSpeech`] keeps encoded: a designed voice's takes
/// clone at most this many references in one run.
const KEPT: usize = 4;

/// A cloned voice as VoxCPM2 holds it: the reference every take clones,
/// and, for Hi-Fi, the same recording encoded as the prompt a take
/// continues from.
pub struct VoxcpmVoice<R: Runtime> {
    /// The encoded reference.
    pub reference: Reference<R>,
    /// The encoded Hi-Fi prompt. `None` when the voice clones without its
    /// transcript.
    pub prompt: Option<Prompt<R>>,
}

/// A recording's identity in the kept encodes: the sha1 of its rate and
/// sample bits.
type RecordingKey = [u8; 20];

/// A loaded [`VoxCpm2Engine`] as a [`SpeechEngine`]: the skill's VoxCPM2
/// runner for one take.
///
/// A take speaks the text [`voxcpm_text`] builds from the request, at the
/// request's seed, guidance and steps. The voice is:
///
/// - [`VoiceSource::Own`]: zero-shot, a design prefixing the text.
/// - [`VoiceSource::Named`]: a voice from the engine's voices directory.
/// - [`VoiceSource::Clone`] with [`CloneReference::Handle`]: a
///   [`VoxcpmVoice`] from [`encode_voice`](Self::encode_voice).
/// - [`VoiceSource::Clone`] with [`CloneReference::Samples`]: the
///   recording, encoded on this thread the first time it is cloned and kept
///   for later takes. A Hi-Fi take encodes it as its prompt too.
///
/// The take is at the engine's rate. VoxCPM2 takes its pace from the
/// reference and the delivery notes, so a speed other than 1 is refused.
pub struct VoxcpmSpeech<'a, R: Runtime<DType = DType>> {
    engine: &'a VoxCpm2Engine<R>,
    kept: Vec<(RecordingKey, VoxcpmVoice<R>)>,
}

impl<'a, R> VoxcpmSpeech<'a, R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Speaks with `engine`.
    pub fn new(engine: &'a VoxCpm2Engine<R>) -> Self {
        Self {
            engine,
            kept: Vec::new(),
        }
    }

    /// `samples`, mono at `rate`, encoded as a voice to clone. With `hifi`,
    /// it is encoded as the Hi-Fi prompt too.
    pub fn encode_voice(&self, samples: &[f32], rate: u32, hifi: bool) -> Result<VoxcpmVoice<R>> {
        let encoded = if hifi {
            self.engine
                .encode_reference_and_prompt(samples, rate)
                .map(|(reference, prompt)| VoxcpmVoice {
                    reference,
                    prompt: Some(prompt),
                })
        } else {
            self.engine
                .encode_reference(samples, rate)
                .map(|reference| VoxcpmVoice {
                    reference,
                    prompt: None,
                })
        };
        encoded.map_err(|e| Error::Engine {
            reason: format!(
                "VoxCPM2 encoding a {}-sample reference at {rate} Hz: {e}. Pass a mono recording of speech",
                samples.len()
            ),
        })
    }

    /// Where the kept voice of `samples` at `rate` sits, encoded now when no
    /// take has cloned it yet, or when a Hi-Fi take needs its prompt. The
    /// oldest kept voice makes room past [`KEPT`].
    fn kept_voice(&mut self, samples: &[f32], rate: u32, hifi: bool) -> Result<usize> {
        let key = recording_key(samples, rate);
        let at = match self.kept.iter().position(|(k, _)| *k == key) {
            Some(at) => {
                let entry = self.kept.remove(at);
                self.kept.push(entry);
                self.kept.len() - 1
            }
            None => {
                let voice = self.encode_voice(samples, rate, hifi)?;
                if self.kept.len() == KEPT {
                    self.kept.remove(0);
                }
                self.kept.push((key, voice));
                self.kept.len() - 1
            }
        };
        if let Some((_, voice)) = self.kept.get_mut(at).filter(|_| hifi)
            && voice.prompt.is_none()
        {
            let prompt = self
                .engine
                .encode_prompt(samples, rate)
                .map_err(|e| Error::Engine {
                    reason: format!(
                        "VoxCPM2 encoding a {}-sample Hi-Fi prompt at {rate} Hz: {e}. Pass a mono recording of speech",
                        samples.len()
                    ),
                })?;
            voice.prompt = Some(prompt);
        }
        Ok(at)
    }
}

/// The sha1 of `rate` and the bits of `samples`.
fn recording_key(samples: &[f32], rate: u32) -> RecordingKey {
    let mut hash = Sha1::new();
    hash.update(rate.to_le_bytes());
    for s in samples {
        hash.update(s.to_le_bytes());
    }
    hash.finalize().into()
}

/// Refuses a speed other than 1, guidance that is not finite, and zero
/// steps, naming the take's text.
fn check_request<H>(request: &SpeakRequest<'_, H>) -> Result<(f32, usize)> {
    if request.speed != 1.0 {
        return Err(Error::InvalidArgument {
            arg: "speed",
            reason: format!(
                "VoxCPM2 speaks at the pace of its reference and delivery notes, and \"{}\" asks for speed {}. Pass speed 1 and time-stretch the joined track",
                request.text, request.speed
            ),
        });
    }
    let cfg = request.cfg as f32;
    if !cfg.is_finite() {
        return Err(Error::InvalidArgument {
            arg: "cfg",
            reason: format!(
                "guidance {} is not a finite number. Pass a guidance such as 1.6",
                request.cfg
            ),
        });
    }
    if request.steps == 0 {
        return Err(Error::InvalidArgument {
            arg: "steps",
            reason: "0 diffusion steps. Pass at least 1, such as 16".to_string(),
        });
    }
    Ok((cfg, request.steps as usize))
}

impl<R> SpeechEngine for VoxcpmSpeech<'_, R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    type Reference = VoxcpmVoice<R>;

    fn speak(&mut self, request: &SpeakRequest<'_, VoxcpmVoice<R>>) -> Result<Take> {
        let (cfg, steps) = check_request(request)?;
        let seed = engine_seed(request.seed)?;
        let planned = voxcpm_text(request);
        let hifi = planned.hifi.is_some();
        let kept_at = match request.voice {
            VoiceSource::Clone {
                reference: CloneReference::Samples { samples, rate },
                ..
            } => Some(self.kept_voice(samples, rate, hifi)?),
            _ => None,
        };
        let voice: Option<&VoxcpmVoice<R>> = match request.voice {
            VoiceSource::Clone {
                reference: CloneReference::Handle(voice),
                ..
            } => Some(voice),
            VoiceSource::Clone { .. } => kept_at.and_then(|at| self.kept.get(at)).map(|(_, v)| v),
            _ => None,
        };
        let source = match (request.voice, voice) {
            (VoiceSource::Named { name }, _) => VoxVoice::Named(name),
            (VoiceSource::Own { .. }, _) => VoxVoice::ZeroShot,
            (VoiceSource::Clone { .. }, Some(voice)) => VoxVoice::Reference(&voice.reference),
            (VoiceSource::Clone { .. }, None) => {
                return Err(Error::Engine {
                    reason: "the cloned reference was not kept after its encode. Pass it again"
                        .to_string(),
                });
            }
        };
        let mut synth = SynthRequest::new(&planned.text, source)
            .with_seed(seed)
            .with_cfg(cfg)
            .with_steps(steps);
        if let (Some(transcript), Some(voice)) = (planned.hifi, voice) {
            let prompt = voice.prompt.as_ref().ok_or_else(|| Error::InvalidArgument {
                arg: "voice",
                reason: "a Hi-Fi take continues from its reference as a prompt, and this voice was encoded without one. Encode it with VoxcpmSpeech::encode_voice and hifi on".to_string(),
            })?;
            synth = synth.with_prompt(prompt, transcript);
        }
        let samples = self
            .engine
            .render_request(&synth)
            .map_err(|e| Error::Engine {
                reason: format!(
                    "VoxCPM2 speaking \"{}\" at seed {seed}: {e}. Check the voice, and that the GPU has free memory",
                    planned.text
                ),
            })?;
        Ok(Take {
            samples,
            rate: TtsEngine::sample_rate(self.engine),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take::VoiceSource;

    fn request(speed: f64, cfg: f64, steps: u32) -> SpeakRequest<'static, ()> {
        SpeakRequest {
            text: "Hello.",
            delivery: None,
            style: None,
            voice: VoiceSource::Own { design: None },
            seed: 7.0,
            cfg,
            steps,
            speed,
        }
    }

    #[test]
    fn speed_guidance_and_steps_are_checked() {
        assert_eq!(check_request(&request(1.0, 1.6, 16)).ok(), Some((1.6, 16)));
        let slow = check_request(&request(0.9, 1.6, 16)).err();
        assert!(slow.is_some_and(|e| e.to_string().contains("speed 0.9")));
        assert!(check_request(&request(1.0, f64::NAN, 16)).is_err());
        assert!(check_request(&request(1.0, 1.6, 0)).is_err());
    }

    #[test]
    fn a_recording_key_reads_rate_and_samples() {
        let a = recording_key(&[0.5, -0.25], 48_000);
        assert_eq!(a, recording_key(&[0.5, -0.25], 48_000));
        assert_ne!(a, recording_key(&[0.5, -0.25], 24_000));
        assert_ne!(a, recording_key(&[0.5, 0.25], 48_000));
    }
}
