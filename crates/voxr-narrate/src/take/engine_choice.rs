//! The engine a run speaks with when none is named: the skill's
//! `defaultEngine`, ported as a pure function. Reading the GPU or `uv`
//! stays with the caller; readiness comes in already decided.

use crate::error::{Error, Result};
use crate::pattern::{Patterns, starts_with_en};

/// Kokoro's own voice, played with no cloning: `onnx-community/Kokoro-82M-v1.0-ONNX`.
pub const KOKORO_MODEL: &str = "onnx-community/Kokoro-82M-v1.0-ONNX";

/// Kokoro's voice when none is asked for.
pub const DEFAULT_VOICE: &str = "af_heart";

/// The engine `speak()` ends up asking for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// Kokoro-82M, through `kokoro-js`.
    Kokoro,
    /// VoxCPM2, cloning a voice.
    Voxcpm,
}

/// [`default_engine`]'s inputs: a run's own voice, reference and language,
/// before any config fallback.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultEngineOpts<'a> {
    /// The voice asked for, if any.
    pub voice: Option<&'a str>,
    /// A recording to clone, if any.
    pub reference: Option<&'a str>,
    /// The script's language, if any.
    pub language: Option<&'a str>,
}

/// A JavaScript-truthy string: set and not empty.
pub(super) fn truthy(s: Option<&str>) -> bool {
    s.is_some_and(|s| !s.is_empty())
}

/// The engine a run speaks with when none is named: the skill's
/// `defaultEngine`.
///
/// A voice shaped like Kokoro's own (`af_heart`, `bm_george`, …) picks
/// Kokoro. Else a ready VoxCPM2, or a reference to clone, picks VoxCPM2.
/// Else a language other than English has nowhere to go: Kokoro speaks
/// English only, and VoxCPM2 is not ready, so this errors with
/// [`Error::InvalidArgument`] (`arg: "language"`). Otherwise, Kokoro.
///
/// `voxcpm_ready` and `voxcpm_why` are the caller's own read of the
/// engine's readiness (an NVIDIA GPU with enough memory, and `uv`), as the
/// skill's `voxcpmReady()` reports it.
pub fn default_engine(
    opts: &DefaultEngineOpts<'_>,
    voxcpm_ready: bool,
    voxcpm_why: &str,
) -> Result<Engine> {
    let p = Patterns::get()?;
    if truthy(opts.voice) && p.kokoro_voice.is_match(opts.voice.unwrap_or_default()) {
        return Ok(Engine::Kokoro);
    }
    if voxcpm_ready || truthy(opts.reference) {
        return Ok(Engine::Voxcpm);
    }
    if truthy(opts.language) {
        let language = opts.language.unwrap_or_default();
        if !starts_with_en(language) {
            return Err(Error::InvalidArgument {
                arg: "language",
                reason: format!(
                    "Kokoro speaks English only, and VoxCPM2 {voxcpm_why}. Use the user's TTS (--command), a recording, or a Hugging Face model whose license allows the video's use (--model, references/tts.md)."
                ),
            });
        }
    }
    Ok(Engine::Kokoro)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kokoro_shaped_voice_picks_kokoro() -> Result<()> {
        let d = DefaultEngineOpts {
            voice: Some("af_heart"),
            reference: None,
            language: None,
        };
        assert_eq!(default_engine(&d, false, "not ready")?, Engine::Kokoro);
        Ok(())
    }

    #[test]
    fn a_ready_voxcpm_or_a_reference_picks_voxcpm() -> Result<()> {
        let no_voice = DefaultEngineOpts::default();
        assert_eq!(default_engine(&no_voice, true, "")?, Engine::Voxcpm);
        let with_ref = DefaultEngineOpts {
            reference: Some("clip.wav"),
            ..DefaultEngineOpts::default()
        };
        assert_eq!(
            default_engine(&with_ref, false, "not ready")?,
            Engine::Voxcpm
        );
        Ok(())
    }

    #[test]
    fn a_non_english_language_with_no_voxcpm_errors() {
        let d = DefaultEngineOpts {
            language: Some("ms"),
            ..DefaultEngineOpts::default()
        };
        let err = default_engine(
            &d,
            false,
            "needs an NVIDIA GPU with 8 GB of memory (none found)",
        )
        .err()
        .map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some(
                "invalid argument 'language': Kokoro speaks English only, and VoxCPM2 needs an NVIDIA GPU with 8 GB of memory (none found). Use the user's TTS (--command), a recording, or a Hugging Face model whose license allows the video's use (--model, references/tts.md)."
            )
        );
    }

    #[test]
    fn english_or_no_language_falls_back_to_kokoro() -> Result<()> {
        let en = DefaultEngineOpts {
            language: Some("en-US"),
            ..DefaultEngineOpts::default()
        };
        assert_eq!(default_engine(&en, false, "")?, Engine::Kokoro);
        assert_eq!(
            default_engine(&DefaultEngineOpts::default(), false, "")?,
            Engine::Kokoro
        );
        Ok(())
    }

    #[test]
    fn a_falsy_voice_reference_or_language_is_read_as_unset() -> Result<()> {
        // JavaScript's `""` is falsy: an empty voice, reference or
        // language never trips its own branch.
        let empty = DefaultEngineOpts {
            voice: Some(""),
            reference: Some(""),
            language: Some(""),
        };
        assert_eq!(default_engine(&empty, false, "not ready")?, Engine::Kokoro);
        Ok(())
    }
}
