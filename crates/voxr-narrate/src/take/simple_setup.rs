//! What a speak run of an engine other than VoxCPM2 speaks with, what it
//! returns, the label of each engine, and the check of a Kokoro voice.

use std::path::Path;

use serde::Serialize;

use super::speak_resolve::EngineChoice;
use crate::check::{HeardWord, StrayPause};
use crate::error::{Error, Result};

/// What a run of [`speak_phrases`](super::speak_phrases) speaks with.
#[derive(Debug, Clone, Copy)]
pub struct SimpleSetup<'a> {
    /// The voiceover's path. Its directory exists.
    pub out: &'a Path,
    /// The engine, as [`resolve_speak`](super::resolve_speak) picks it.
    /// VoxCPM2 is refused: it speaks through
    /// [`speak_voxcpm`](super::speak_voxcpm).
    pub engine: &'a EngineChoice,
    /// The pace: Kokoro speaks at it. Every other engine speaks at 1 and is
    /// time-stretched afterwards.
    pub speed: f64,
    /// The command speaks the whole script in one call, without the pause
    /// marks: for an engine that loads slowly on every call. Only the
    /// command engine reads it.
    pub one_call: bool,
    /// The language spoken, which picks the recognizer.
    pub language: Option<&'a str>,
    /// The user's config `asr.model`: the recognizer heard first in place
    /// of the language's own ([`asr_model`](super::hear::asr_model)).
    pub asr_model: Option<&'a str>,
}

/// The check of the whole voiceover, as the speak command prints it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WholeCheck {
    /// Where the check starts: always 0, the whole file.
    pub at: f64,
    /// The script as captions show it.
    pub text: String,
    /// The voiceover says its script.
    pub ok: bool,
    /// The words heard, clipped.
    pub heard: String,
    /// Why the voiceover fails; empty when it passes.
    pub why: String,
    /// Words both recognizers missed or misheard, marked for a listen.
    pub doubt: Vec<String>,
    /// Pauses the script does not ask for.
    pub pauses: Vec<StrayPause>,
}

/// One step of a run of [`speak_phrases`](super::speak_phrases), in order.
#[derive(Debug, Clone, PartialEq)]
pub enum SimpleStep {
    /// The command spoke phrase `done` of `total`.
    CommandProgress {
        /// Phrases spoken so far.
        done: usize,
        /// Phrases in the script.
        total: usize,
    },
}

impl SimpleStep {
    /// The line the skill prints for this step.
    pub fn message(&self) -> String {
        match self {
            Self::CommandProgress { done, total } => {
                format!("tts command: {done}/{total} phrases")
            }
        }
    }
}

/// A finished run of [`speak_phrases`](super::speak_phrases).
#[derive(Debug, Clone, PartialEq)]
pub struct SimpleRun {
    /// The voiceover's samples, as joined before the WAV rounds them.
    pub samples: Vec<f32>,
    /// Their sample rate: the last take's.
    pub rate: u32,
    /// The voiceover's length in seconds.
    pub duration: f64,
    /// Each phrase's span on the voiceover's clock, its text as captions
    /// show it. `None` when the command speaks the script in one call.
    pub phrases: Option<Vec<HeardWord>>,
    /// The engine's label: `Kokoro (<voice>)`, the model id, or
    /// `command: <template>`.
    pub engine: String,
    /// The one check of the whole voiceover.
    pub checks: Vec<WholeCheck>,
    /// One line when the check fails.
    pub problems: Vec<String>,
    /// Every step, in order.
    pub log: Vec<SimpleStep>,
}

impl EngineChoice {
    /// The engine's label in a run's result: `VoxCPM2` and its voice,
    /// `Kokoro (<voice>)`, the model id, or `command: <template>`.
    pub fn label(&self) -> String {
        match self {
            Self::Voxcpm { voice } if voice.is_empty() => "VoxCPM2".to_string(),
            Self::Voxcpm { voice } => format!("VoxCPM2 {voice}"),
            Self::Command { command, .. } => format!("command: {command}"),
            Self::Kokoro { voice, .. } => format!("Kokoro ({voice})"),
            Self::Model { model } => model.clone(),
        }
    }
}

/// Refuses a Kokoro voice the model does not hold. `voices` are the
/// model's voices, in its own order.
pub fn check_kokoro_voice(voice: &str, voices: &[&str]) -> Result<()> {
    if voices.contains(&voice) {
        return Ok(());
    }
    Err(Error::InvalidArgument {
        arg: "voice",
        reason: format!(
            "Kokoro has no voice \"{voice}\". Voices: {}.",
            voices.join(", ")
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        let voxcpm = |v: &str| EngineChoice::Voxcpm {
            voice: v.to_string(),
        };
        assert_eq!(voxcpm("").label(), "VoxCPM2");
        assert_eq!(voxcpm("(warm)").label(), "VoxCPM2 (warm)");
        let model = EngineChoice::Model {
            model: "facebook/mms-tts-eng".to_string(),
        };
        assert_eq!(model.label(), "facebook/mms-tts-eng");
    }

    #[test]
    fn an_unknown_kokoro_voice_lists_the_voices() {
        assert!(check_kokoro_voice("af_heart", &["af_heart", "bm_george"]).is_ok());
        let err = check_kokoro_voice("zz", &["af_heart", "bm_george"])
            .err()
            .map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some(
                "invalid argument 'voice': Kokoro has no voice \"zz\". Voices: af_heart, bm_george."
            )
        );
    }
}
