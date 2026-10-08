//! What a whole speak run of [`speak_script`](super::speak_script) reads
//! and returns: where it writes and what the machine runs, the seam its
//! engine runs go through, each step it logs, and its result as the skill's
//! `speak()` returns it.

use std::path::{Path, PathBuf};

use serde::Serialize;

use super::join_checks::PartCheck;
use super::simple_setup::{SimpleRun, SimpleSetup, SimpleStep, WholeCheck};
use super::speak_setup::{VoxcpmRun, VoxcpmSetup};
use super::speak_step::SpeakStep;
use crate::check::HeardWord;
use crate::error::Result;

/// Where a speak run writes, and what the machine runs.
#[derive(Debug, Clone, Copy)]
pub struct SpeakTarget<'a> {
    /// The voiceover's path. Its directory exists.
    pub out: &'a Path,
    /// The VoxCPM2 take cache directory.
    pub cache_dir: &'a Path,
    /// The VoxCPM2 checkpoint on disk. `None` when the model downloads
    /// first.
    pub checkpoint: Option<&'a str>,
    /// VoxCPM2 runs here: an NVIDIA GPU with enough memory, and `uv`.
    pub voxcpm_ready: bool,
    /// Why VoxCPM2 is not ready, as the skill's `voxcpmReady().why` words
    /// it, such as `needs an NVIDIA GPU with 8 GB of memory (none found)`.
    pub voxcpm_why: &'a str,
}

/// The engine runs a speak run goes through. The caller loads the engines.
///
/// An implementation runs [`speak_voxcpm`](super::speak_voxcpm) and
/// [`speak_phrases`](super::speak_phrases) with them, or stubs either in a
/// test. Each run writes its voiceover to the setup's `out`.
pub trait SpeakRuns {
    /// A VoxCPM2 run of `script`.
    fn voxcpm(&mut self, script: &str, setup: &VoxcpmSetup<'_>) -> Result<VoxcpmRun>;

    /// A run of `script` with Kokoro, a Hugging Face model or the user's
    /// command, as `setup.engine` names it.
    fn phrases(&mut self, script: &str, setup: &SimpleSetup<'_>) -> Result<SimpleRun>;
}

/// The checks a run returns: one per VoxCPM2 part, or one of the whole
/// voiceover for every other engine. Either serializes as a JSON array.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ScriptChecks {
    /// VoxCPM2: each part's check, heat and marks.
    Parts(Vec<PartCheck>),
    /// Every other engine: the one check of the whole voiceover.
    Whole(Vec<WholeCheck>),
}

impl ScriptChecks {
    /// The number of checks.
    pub fn len(&self) -> usize {
        match self {
            Self::Parts(list) => list.len(),
            Self::Whole(list) => list.len(),
        }
    }

    /// No check at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One step of a speak run, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum ScriptStep {
    /// The engine reads tags like `[laughing]` as words, so the script
    /// speaks without them.
    TagsStripped,
    /// A step of the VoxCPM2 run.
    Voxcpm(SpeakStep),
    /// A step of the run of every other engine.
    Phrases(SimpleStep),
}

impl ScriptStep {
    /// The line the skill prints for this step, when it prints one.
    pub fn message(&self) -> Option<String> {
        match self {
            Self::TagsStripped => Some(
                "speak: only VoxCPM2 voices tags like [laughing] or [sigh]. This engine speaks the script without them."
                    .to_string(),
            ),
            Self::Voxcpm(step) => step.message(),
            Self::Phrases(step) => Some(step.message()),
        }
    }
}

/// A finished speak run: the skill's `speak()` result. It serializes to
/// the JSON object the skill returns, with the log left out.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpokenScript {
    /// The voiceover's length in seconds.
    pub duration: f64,
    /// Each phrase's span on the voiceover's clock. `None` when the command
    /// speaks the script in one call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phrases: Option<Vec<HeardWord>>,
    /// The engine's label: `VoxCPM2` and its voice, `Kokoro (<voice>)`, the
    /// model id, or `command: <template>`.
    pub engine: String,
    /// Each part's check, or the one check of the whole voiceover.
    pub checks: ScriptChecks,
    /// One line per problem the speech check or the placement found.
    pub problems: Vec<String>,
    /// A timed script's lines that start late or overrun their slot.
    /// `None` when no line was placed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<Vec<String>>,
    /// VoxCPM2's designed voice copied beside the output: `Some(None)` for
    /// a user's recording, which serializes as `null`. `None` for every
    /// other engine, which leaves the key out.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_file: Option<Option<PathBuf>>,
    /// Every step, in order.
    #[serde(skip)]
    pub log: Vec<ScriptStep>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tag_notice_reads_as_the_skill_prints_it() {
        assert_eq!(
            ScriptStep::TagsStripped.message().as_deref(),
            Some(
                "speak: only VoxCPM2 voices tags like [laughing] or [sigh]. This engine speaks the script without them."
            )
        );
        let step = ScriptStep::Phrases(SimpleStep::CommandProgress { done: 1, total: 2 });
        assert_eq!(step.message().as_deref(), Some("tts command: 1/2 phrases"));
    }

    #[test]
    fn json_leaves_out_what_the_skill_leaves_undefined() -> Result<()> {
        let mut spoken = SpokenScript {
            duration: 1.5,
            phrases: None,
            engine: "command: tts".to_string(),
            checks: ScriptChecks::Whole(Vec::new()),
            problems: Vec::new(),
            timing: None,
            voice_file: None,
            log: vec![ScriptStep::TagsStripped],
        };
        let json = |s: &SpokenScript| {
            serde_json::to_string(s).map_err(|e| crate::error::Error::DataError {
                reason: e.to_string(),
            })
        };
        assert_eq!(
            json(&spoken)?,
            r#"{"duration":1.5,"engine":"command: tts","checks":[],"problems":[]}"#
        );
        spoken.voice_file = Some(None);
        spoken.timing = Some(vec!["late".to_string()]);
        assert_eq!(
            json(&spoken)?,
            r#"{"duration":1.5,"engine":"command: tts","checks":[],"problems":[],"timing":["late"],"voiceFile":null}"#
        );
        Ok(())
    }
}
