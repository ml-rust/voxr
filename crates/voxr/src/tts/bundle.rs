//! Text-to-speech bundle: neural acoustic model + vocoder + voice catalog.
//!
//! A bundle can run in two modes. Without an attached [`TtsEngine`] it
//! validates the surrounding pipeline — voice lookup and WAV encoding — and
//! returns [`TtsError::NotImplemented`] from the synthesis step, which lets
//! callers assert their plumbing is correct without loading model weights.
//! With an engine attached via [`TtsBundle::with_engine`], `synthesize` runs
//! the full `text → waveform` neural path, G2P included — each engine owns
//! its own G2P front end (Kokoro's is `KokoroG2p`, gated on the `kokoro`
//! feature), since different engines need different phoneme sets or none at
//! all. The engine is a trait object, so one bundle type serves Kokoro and
//! VoxCPM2 alike.

use std::sync::Arc;

use thiserror::Error;

use crate::lang::Lang;
use crate::tts::engine::TtsEngine;

/// Errors raised by the TTS pipeline.
#[derive(Debug, Error)]
pub enum TtsError {
    #[error("voice '{0}' not registered on this bundle")]
    UnknownVoice(String),
    #[error("neural synthesis requires an engine; this bundle was created without one")]
    NotImplemented,
    #[error("load error: {0}")]
    Load(String),
    #[error("engine error: {0}")]
    Engine(String),
    /// The engine refused the request itself — an option it has no control
    /// for, or a voice it does not carry — as opposed to failing while
    /// rendering. A server answers this with a client error.
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

/// A single voice the bundle supports.
///
/// Kokoro's canonical naming: two-letter prefix (`af` = American Female,
/// `am` = American Male, `bf` = British Female, `bm` = British Male, `jf/jm`
/// = Japanese, etc.) followed by a display name (`alloy`, `nova`, `adam`,
/// `alice`, …).
#[derive(Debug, Clone)]
pub struct Voice {
    pub id: String,
    pub lang: Lang,
    /// Display name (used in API error messages and catalogs).
    pub display_name: String,
}

impl Voice {
    pub fn new(id: impl Into<String>, lang: Lang, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            lang,
            display_name: display_name.into(),
        }
    }
}

/// Configuration controlling a synthesis call.
#[derive(Debug, Clone)]
pub struct SynthesizeOptions {
    /// Playback-rate multiplier applied to the duration predictor. Kokoro
    /// supports ~0.25×..4×. Enforced by the caller (blazr).
    pub speed: f32,
}

impl Default for SynthesizeOptions {
    fn default() -> Self {
        Self { speed: 1.0 }
    }
}

/// End-to-end TTS bundle.
///
/// Holds the voice catalog and, optionally, a neural synthesis engine. Without
/// an engine, only the G2P + voice-catalog portions are functional and
/// `synthesize` returns [`TtsError::NotImplemented`].
pub struct TtsBundle {
    voices: Vec<Voice>,
    /// Sample rate of generated waveforms (Kokoro is 24 kHz).
    pub sample_rate: u32,
    /// Neural synthesis engine. When `None`, `synthesize` returns
    /// [`TtsError::NotImplemented`] (scaffolding mode). When `Some`, the full
    /// `text → waveform` pipeline runs.
    engine: Option<Arc<dyn TtsEngine>>,
}

impl TtsBundle {
    /// Create an engine-less bundle with a fixed voice catalog. Useful for
    /// validating voice lookup without loading model weights; `synthesize`
    /// returns [`TtsError::NotImplemented`].
    pub fn scaffolding(voices: Vec<Voice>, sample_rate: u32) -> Self {
        Self {
            voices,
            sample_rate,
            engine: None,
        }
    }

    /// Attach a real synthesis engine. Once attached, `synthesize` runs the
    /// full neural path instead of returning `NotImplemented`.
    pub fn with_engine(mut self, engine: Arc<dyn TtsEngine>) -> Self {
        self.sample_rate = engine.sample_rate();
        self.engine = Some(engine);
        self
    }

    /// Whether this bundle has a live neural engine.
    pub fn has_engine(&self) -> bool {
        self.engine.is_some()
    }

    /// All voices this bundle exposes.
    pub fn voices(&self) -> &[Voice] {
        &self.voices
    }

    /// Look up a voice by id (e.g. `"af_alloy"`).
    pub fn voice(&self, id: &str) -> Option<&Voice> {
        self.voices.iter().find(|v| v.id == id)
    }

    /// Synthesize a waveform.
    ///
    /// If an engine is attached, runs the full neural pipeline end-to-end.
    /// Otherwise returns [`TtsError::NotImplemented`] after validating the
    /// voice — same behavior as before the engine landed, so the server
    /// surfaces configuration issues fast.
    pub fn synthesize(
        &self,
        text: &str,
        voice_id: &str,
        options: &SynthesizeOptions,
    ) -> Result<Vec<f32>, TtsError> {
        match &self.engine {
            Some(engine) => engine
                .synthesize(text, voice_id, options.speed)
                .map_err(engine_error),
            None => self.scaffolding_failure(text, voice_id),
        }
    }

    /// Synthesize a waveform chunk by chunk.
    ///
    /// Same request contract as [`TtsBundle::synthesize`]; the chunks handed
    /// to `sink`, concatenated, equal that call's output. A `sink` error
    /// aborts the render. Engine errors map exactly as in `synthesize`, so a
    /// server's status mapping is shared between the two paths.
    pub fn synthesize_stream(
        &self,
        text: &str,
        voice_id: &str,
        options: &SynthesizeOptions,
        sink: &mut dyn FnMut(&[f32]) -> crate::error::Result<()>,
    ) -> Result<(), TtsError> {
        match &self.engine {
            Some(engine) => engine
                .synthesize_stream(text, voice_id, options.speed, sink)
                .map_err(engine_error),
            None => self.scaffolding_failure(text, voice_id).map(|_| ()),
        }
    }

    /// Scaffolding path: validate the voice, then fail with
    /// `NotImplemented`. Exercises the voice-lookup plumbing so a request
    /// for an unregistered voice surfaces before synthesis is even
    /// attempted.
    fn scaffolding_failure(&self, text: &str, voice_id: &str) -> Result<Vec<f32>, TtsError> {
        let _ = text;
        self.voice(voice_id)
            .ok_or_else(|| TtsError::UnknownVoice(voice_id.to_string()))?;
        Err(TtsError::NotImplemented)
    }
}

/// An engine refuses a request with `InvalidArgument` whether it raises it
/// here or inside a boostr model call; everything else failed while
/// rendering.
fn engine_error(e: crate::error::Error) -> TtsError {
    match e {
        crate::error::Error::InvalidArgument { .. }
        | crate::error::Error::Boostr(boostr::error::Error::InvalidArgument { .. }) => {
            TtsError::InvalidRequest(e.to_string())
        }
        other => TtsError::Engine(other.to_string()),
    }
}

/// Kokoro-82M's English voices, in the order `kokoro-js` lists them: the
/// `a*` American and `b*` British voice packs the misaki G2P speaks.
pub const KOKORO_VOICES: [&str; 28] = [
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
    "bf_alice",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
];

/// The Kokoro voice catalog: every id in [`KOKORO_VOICES`], in its order.
/// An `a*` voice is American English and a `b*` voice British English, and
/// its name is the id after the underscore, capitalized.
pub fn default_kokoro_voices() -> Vec<Voice> {
    KOKORO_VOICES
        .iter()
        .map(|&id| {
            let lang = if id.starts_with('b') {
                Lang::EnGb
            } else {
                Lang::EnUs
            };
            let name = id.split_once('_').map_or(id, |(_, name)| name);
            let mut chars = name.chars();
            let name: String = chars
                .next()
                .map(|c| c.to_uppercase().chain(chars).collect())
                .unwrap_or_default();
            Voice::new(id, lang, name)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffolding_bundle_exposes_voices() {
        let bundle = TtsBundle::scaffolding(default_kokoro_voices(), 24_000);
        assert!(bundle.voice("af_alloy").is_some());
        assert!(bundle.voice("not-a-voice").is_none());
        assert_eq!(bundle.sample_rate, 24_000);
    }

    #[test]
    fn synthesize_returns_not_implemented() {
        let bundle = TtsBundle::scaffolding(default_kokoro_voices(), 24_000);
        let err = bundle
            .synthesize("hello", "af_alloy", &SynthesizeOptions::default())
            .unwrap_err();
        assert!(matches!(err, TtsError::NotImplemented));
    }

    #[test]
    fn synthesize_validates_voice_first() {
        let bundle = TtsBundle::scaffolding(default_kokoro_voices(), 24_000);
        let err = bundle
            .synthesize("hello", "ghost_voice", &SynthesizeOptions::default())
            .unwrap_err();
        assert!(matches!(err, TtsError::UnknownVoice(_)));
    }

    #[test]
    fn default_catalog_has_known_voices() {
        let voices = default_kokoro_voices();
        assert_eq!(voices.len(), KOKORO_VOICES.len());
        assert_eq!(voices[0].id, "af_heart");
        assert_eq!(voices[0].display_name, "Heart");
        assert!(voices.iter().any(|v| v.id == "af_nova"));
        let george = voices.iter().find(|v| v.id == "bm_george");
        assert!(george.is_some_and(|v| v.lang == Lang::EnGb && v.display_name == "George"));
    }

    /// A zero-shot voice (VoxCPM2's [`ZERO_SHOT_VOICE_ID`], mirrored here as
    /// a plain string so this test needs no `voxcpm` feature) behaves like
    /// any other catalog entry at the bundle layer: `voice()` finds it, and
    /// `synthesize`'s scaffolding path never reports it unknown.
    #[test]
    fn zero_shot_voice_is_found_and_not_reported_unknown() {
        let mut voices = default_kokoro_voices();
        voices.push(Voice::new("zero-shot", Lang::EnUs, "zero-shot"));
        let bundle = TtsBundle::scaffolding(voices, 24_000);
        assert!(bundle.voice("zero-shot").is_some());

        let err = bundle
            .synthesize("hello", "zero-shot", &SynthesizeOptions::default())
            .unwrap_err();
        assert!(matches!(err, TtsError::NotImplemented));
    }
}
