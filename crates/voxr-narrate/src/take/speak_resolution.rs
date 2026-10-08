//! What the speak resolution ends up with: the engine and its own
//! settings, the validated speed and the tag decision, and the questions a
//! run asks of them.

use super::place::CueUnits;

/// The engine and its own settings [`resolve_speak`](super::speak_resolve::resolve_speak) ends up with.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineChoice {
    /// VoxCPM2, cloning `voice` (empty for a designed voice).
    Voxcpm {
        /// The voice to clone, or a design description. Empty for none.
        voice: String,
    },
    /// The user's own TTS command template.
    Command {
        /// The command template, with its `{text}`/`{out}`/… placeholders.
        command: String,
        /// The command's `{voice}`: `opts.voice ?? tts.voice ?? ''`.
        voice: String,
        /// The command's `{reference}`: `opts.reference ?? tts.reference ?? ''`.
        reference: String,
    },
    /// Kokoro-82M, through `kokoro-js`.
    Kokoro {
        /// The Kokoro model id.
        model: String,
        /// The voice to speak in.
        voice: String,
    },
    /// Any other Hugging Face text-to-speech model.
    Model {
        /// The model id or path.
        model: String,
    },
}

/// What `speak()` decides before it makes any audio: the engine to speak
/// with, the validated speed, and whether the script's tags are stripped
/// (and a notice of that logged).
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakResolution {
    /// The engine and its settings.
    pub engine: EngineChoice,
    /// The validated speed: `opts.speed ?? tts.speed ?? 1`.
    pub speed: f64,
    /// The engine reads tags like `[laughing]` as words, so they are
    /// stripped from the script before it speaks.
    pub strip_tags: bool,
    /// [`strip_tags`](Self::strip_tags) is set, and this is not a
    /// recursive stretched or per-line call: the skill logs a notice.
    pub log_tag_notice: bool,
    /// The skill's `kokoroEngine`: Kokoro sets its own speed, so the track
    /// is never time-stretched. Set for `--engine kokoro` even when a
    /// `--command` speaks instead.
    pub kokoro_engine: bool,
}

impl SpeakResolution {
    /// The engine is VoxCPM2, which is never time-stretched and speaks a
    /// timed script's lines apart.
    pub fn voxcpm(&self) -> bool {
        matches!(self.engine, EngineChoice::Voxcpm { .. })
    }

    /// The run speaks at speed 1 and is time-stretched to
    /// [`speed`](Self::speed) after: the speed is not 1, the run is not
    /// already a stretched one, and the engine is neither VoxCPM2 nor
    /// Kokoro.
    pub fn stretches(&self, stretched: bool) -> bool {
        self.speed != 1.0 && !stretched && !self.voxcpm() && !self.kokoro_engine
    }

    /// How a timed script's line counts its spoken spans: one per line for
    /// VoxCPM2, one per phrase for every other engine.
    pub fn units(&self) -> CueUnits {
        if self.voxcpm() {
            CueUnits::Lines
        } else {
            CueUnits::Phrases
        }
    }
}
