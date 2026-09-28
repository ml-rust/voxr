//! What a VoxCPM2 run speaks with and where it keeps its files, the
//! user's recording it clones, and what a finished run holds.

use std::path::{Path, PathBuf};

use super::cache::{mtime_millis, ref_id};
use super::join::JoinedTrack;
use super::join_checks::PartCheck;
use super::speak_step::SpeakStep;
use crate::error::{Result, file_error};
use crate::text::Chunk;

/// A user's recording to clone, and its cache identity.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceReference<'a> {
    /// The recording.
    pub path: &'a Path,
    /// Its cache id: [`ref_id`] of its path, size and modified time.
    pub id: String,
}

impl<'a> VoiceReference<'a> {
    /// The recording at `path`, its id read from its metadata.
    pub fn of(path: &'a Path) -> Result<Self> {
        let meta = std::fs::metadata(path).map_err(|e| {
            file_error(
                "reading the reference",
                path,
                &e,
                "Pass a readable recording as the reference",
            )
        })?;
        let modified = meta.modified().map_err(|e| {
            file_error(
                "reading the modified time of the reference",
                path,
                &e,
                "Pass a recording on a filesystem that keeps modified times",
            )
        })?;
        Ok(Self {
            path,
            id: ref_id(&path.to_string_lossy(), meta.len(), mtime_millis(modified)),
        })
    }
}

/// What a VoxCPM2 run speaks with, and where it keeps its files.
#[derive(Debug, Clone, Copy)]
pub struct VoxcpmSetup<'a> {
    /// The VoxCPM2 take cache directory.
    pub cache_dir: &'a Path,
    /// The voiceover's path. Its directory exists.
    pub out: &'a Path,
    /// The voice description, such as `(young woman, warm, low)`; empty
    /// when there is none.
    pub voice: &'a str,
    /// The run's style, parentheses included; empty when there is none.
    pub style: &'a str,
    /// A user's recording to clone instead of a designed voice.
    pub reference: Option<&'a VoiceReference<'a>>,
    /// The seed asked for on the command line.
    pub seed: Option<f64>,
    /// The seed the config names.
    pub config_seed: Option<f64>,
    /// Classifier-free guidance.
    pub cfg: f64,
    /// Diffusion steps.
    pub steps: u32,
    /// Hi-Fi cloning: every part clones with the reference's transcript.
    pub hifi: bool,
    /// The language spoken.
    pub language: Option<&'a str>,
    /// Parts, from 1, to speak again with a fresh seed.
    pub rerolls: &'a [usize],
    /// A timed script: every line is its own part, to place it at its time.
    pub lines: bool,
    /// The VoxCPM2 checkpoint on disk. `None` when the model downloads
    /// first; cache keys then fall back to [`VOXCPM`](super::cache::VOXCPM).
    pub checkpoint: Option<&'a str>,
}

/// A finished VoxCPM2 run.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxcpmRun {
    /// The run's seed, as `<out>.takes.json` records it.
    pub seed: f64,
    /// The parts as finally spoken, split where a whole part failed.
    pub parts: Vec<Chunk>,
    /// The voiceover: its samples at
    /// [`JOIN_RATE`](super::join::JOIN_RATE), its duration and its phrase
    /// spans.
    pub track: JoinedTrack,
    /// Each part's check, heat and marks.
    pub checks: Vec<PartCheck>,
    /// One line per part whose every take failed.
    pub problems: Vec<String>,
    /// The designed voice copied beside the output; `None` for a user's
    /// recording.
    pub voice_file: Option<PathBuf>,
    /// The median pitch of the voice's reference, which each part's heat
    /// reads against.
    pub voice_median: f64,
    /// Every step, in order.
    pub log: Vec<SpeakStep>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn a_reference_id_joins_path_size_and_mtime() -> Result<()> {
        let dir = tempfile::tempdir().map_err(|e| Error::File {
            reason: format!("creating a temporary directory: {e}"),
        })?;
        let rec = dir.path().join("rec.wav");
        std::fs::write(&rec, b"12345").map_err(|e| Error::File {
            reason: e.to_string(),
        })?;
        let r = VoiceReference::of(&rec)?;
        let prefix = format!("{}:5:", rec.display());
        assert!(r.id.starts_with(&prefix), "{}", r.id);
        assert!(VoiceReference::of(&dir.path().join("none.wav")).is_err());
        Ok(())
    }
}
