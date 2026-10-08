//! voxr's VITS (MMS-TTS) engine as a [`SpeechEngine`]: one take per phrase
//! in the checkpoint's own voice, the skill's Hugging Face model run.

use voxr::TtsEngine;
use voxr::tts::VitsEngine;

use super::seed::engine_seed;
use crate::error::{Error, Result};
use crate::take::{SpeakRequest, SpeechEngine, Take, VoiceSource};
use crate::wav::pcm16_round_trip;

/// The rate the skill decodes a phrase of its Python runner at.
const TAKE_RATE: u32 = 48_000;

/// A loaded [`VitsEngine`] as a [`SpeechEngine`].
///
/// A take speaks the request's text in the checkpoint's one voice, at the
/// request's speed, with its noise drawn from the request's seed: the same
/// text, speed and seed give the same take, as the skill's
/// `torch.manual_seed` per phrase does. VITS holds no voice to clone or
/// design, and reads a delivery note as words, so a request with either is
/// refused.
///
/// The take comes back at 48 kHz, as the skill's Python runner hands it
/// on: the runner writes each phrase as a 16-bit WAV at the checkpoint's
/// rate, and the skill decodes it with `ffmpeg -ac 1 -ar 48000`. The take
/// is rounded to 16 bits, then resampled through
/// [`voxr::resample_like_ffmpeg`]. The skill runs a checkpoint with
/// PyTorch weights there, the only kind loaded here.
pub struct VitsSpeech<'a> {
    engine: &'a VitsEngine,
    voice: String,
}

impl<'a> VitsSpeech<'a> {
    /// Speaks with `engine`. Errors when the engine lists no voice.
    pub fn new(engine: &'a VitsEngine) -> Result<Self> {
        let voice = TtsEngine::voices(engine)
            .into_iter()
            .next()
            .map(|v| v.id)
            .ok_or_else(|| Error::Engine {
                reason: "the VITS checkpoint lists no voice. Load a checkpoint whose tokenizer_config.json names its language".to_string(),
            })?;
        Ok(Self { engine, voice })
    }

    /// The checkpoint's one voice: its language code, such as `zlm`.
    pub fn voice(&self) -> &str {
        &self.voice
    }
}

impl SpeechEngine for VitsSpeech<'_> {
    type Reference = ();

    fn speak(&mut self, request: &SpeakRequest<'_, ()>) -> Result<Take> {
        match request.voice {
            VoiceSource::Own { design: None } => {}
            VoiceSource::Named { name } if name == self.voice => {}
            other => {
                return Err(Error::InvalidArgument {
                    arg: "voice",
                    reason: format!(
                        "this VITS checkpoint speaks only its own voice {}, and the take asks for a {} voice. Ask for the model's own voice",
                        self.voice,
                        other.kind()
                    ),
                });
            }
        }
        if let Some(note) = request.prefix() {
            return Err(Error::InvalidArgument {
                arg: "style",
                reason: format!(
                    "VITS reads the delivery note {note} as words. Speak the text without it, or use VoxCPM2"
                ),
            });
        }
        let seed = engine_seed(request.seed)?;
        let speed = request.speed as f32;
        let samples = self
            .engine
            .synthesize_seeded(request.text, &self.voice, speed, Some(seed))
            .map_err(|e| Error::Engine {
                reason: format!(
                    "VITS speaking \"{}\" at speed {speed}, seed {seed}: {e}. Check the text uses characters the checkpoint's vocabulary holds",
                    request.text
                ),
            })?;
        let rate = self.engine.sample_rate();
        let samples = voxr::resample_like_ffmpeg(&pcm16_round_trip(&samples), rate, TAKE_RATE)
            .map_err(|e| Error::Engine {
                reason: format!(
                    "resampling the VITS take of \"{}\" from {rate} Hz to {TAKE_RATE} Hz: {e}. Load a checkpoint at a common rate such as 16 or 22.05 kHz",
                    request.text
                ),
            })?;
        Ok(Take {
            samples,
            rate: TAKE_RATE,
        })
    }
}
