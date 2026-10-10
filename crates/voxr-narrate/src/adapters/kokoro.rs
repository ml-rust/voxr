//! voxr's Kokoro-82M engine as a [`SpeechEngine`]: one take per phrase in
//! a named voice, at a native speed.

use boostr::model::audio::kokoro::KokoroClient;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use voxr::KOKORO_VOICES;
use voxr::kokoro::KokoroEngine;

use super::seed::engine_seed;
use crate::error::{Error, Result};
use crate::take::{SpeakRequest, SpeechEngine, Take, VoiceSource, check_kokoro_voice};

/// A loaded [`KokoroEngine`] as a [`SpeechEngine`].
///
/// A take speaks the request's text in its named voice. The speed divides
/// each phoneme's predicted duration, so the audio is never time-stretched.
/// Kokoro holds no voice to clone or design, and reads a delivery note as
/// words, so a request with either is refused.
///
/// The harmonic source draws from the request's seed. The same text, voice,
/// speed and seed give the same take on one device. CPU and GPU run
/// separate PRNGs, so one seed gives a different take on each, as it does
/// for VITS.
pub struct KokoroSpeech<'a, R: Runtime<DType = DType> = CpuRuntime> {
    engine: &'a KokoroEngine<R>,
    voices: Vec<&'static str>,
}

impl<'a, R: Runtime<DType = DType>> KokoroSpeech<'a, R> {
    /// Speaks with `engine`. Its voices are each [`KOKORO_VOICES`] id whose
    /// voice pack the engine's resolver finds, in that order.
    pub fn new(engine: &'a KokoroEngine<R>) -> Self {
        let voices = KOKORO_VOICES
            .iter()
            .copied()
            .filter(|id| engine.resolver.resolve_path(id).is_ok())
            .collect();
        Self { engine, voices }
    }

    /// The voices this engine speaks, in [`KOKORO_VOICES`] order.
    pub fn voices(&self) -> &[&'static str] {
        &self.voices
    }
}

impl<R: Runtime<DType = DType>> SpeechEngine for KokoroSpeech<'_, R>
where
    R::Client: KokoroClient<R>,
{
    type Reference = ();

    /// Refuses a voice not in [`voices`](Self::voices) with
    /// [`check_kokoro_voice`]'s error, before any audio is made.
    fn speak(&mut self, request: &SpeakRequest<'_, ()>) -> Result<Take> {
        let VoiceSource::Named { name } = request.voice else {
            return Err(Error::InvalidArgument {
                arg: "voice",
                reason: format!(
                    "Kokoro speaks only its named voices, and this take asks for a {} voice. Pass a voice such as af_heart",
                    request.voice.kind()
                ),
            });
        };
        if let Some(note) = request.prefix() {
            return Err(Error::InvalidArgument {
                arg: "style",
                reason: format!(
                    "Kokoro reads the delivery note {note} as words. Speak the text without it, or use VoxCPM2"
                ),
            });
        }
        check_kokoro_voice(name, &self.voices)?;
        let seed = engine_seed(request.seed)?;
        let speed = request.speed as f32;
        let samples = self
            .engine
            .synthesize_seeded(request.text, name, speed, Some(seed))
            .map_err(|e| Error::Engine {
                reason: format!(
                    "Kokoro speaking \"{}\" in {name} at speed {speed}: {e}. Check the voice pack and the G2P models of the loaded Kokoro",
                    request.text
                ),
            })?;
        Ok(Take {
            samples,
            rate: self.engine.sample_rate(),
        })
    }
}
