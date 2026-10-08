//! voxr's Kokoro-82M engine as a [`SpeechEngine`]: one take per phrase in
//! a named voice, at a native speed.

use voxr::KOKORO_VOICES;
use voxr::kokoro::KokoroEngine;

use crate::error::{Error, Result};
use crate::take::{SpeakRequest, SpeechEngine, Take, VoiceSource, check_kokoro_voice};

/// A loaded [`KokoroEngine`] as a [`SpeechEngine`].
///
/// A take speaks the request's text in its named voice. The speed divides
/// each phoneme's predicted duration, so the audio is never time-stretched.
/// Kokoro holds no voice to clone or design, and reads a delivery note as
/// words, so a request with either is refused. The seed is not read: the
/// same text, voice and speed give the same take.
pub struct KokoroSpeech<'a> {
    engine: &'a KokoroEngine,
    voices: Vec<&'static str>,
}

impl<'a> KokoroSpeech<'a> {
    /// Speaks with `engine`. Its voices are each [`KOKORO_VOICES`] id whose
    /// voice pack the engine's resolver finds, in that order.
    pub fn new(engine: &'a KokoroEngine) -> Self {
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

impl SpeechEngine for KokoroSpeech<'_> {
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
        let speed = request.speed as f32;
        let samples = self
            .engine
            .synthesize(request.text, name, speed)
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
