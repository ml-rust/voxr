//! The voice every take clones, on disk: a designed voice kept in the cache
//! by its key and reused once made, a user's recording prepared once into
//! the cache, and the designed voice copied beside the output.
//!
//! A designed voice is `voice-{key}.wav`, with its transcript beside it
//! (`.txt`), its first design (`.design.wav`), and a marker under
//! `prepared/` named by its bytes' hash, so a copy passed back as a
//! reference is never prepared twice. The voice file is written last, so a
//! run that stops partway leaves no voice to reuse.

use std::path::{Path, PathBuf};

use super::anchor::{ANCHOR_RATE, AnchorStep, DesignSetup, DesignedVoice, design_voice};
use super::cache::{
    content_hash, prepared_marker, reference_file, voice_design_file, voice_file,
    voice_transcript_file,
};
use super::engine::{SpeechEngine, TakeStore};
use super::files::voice_output_file;
use crate::error::{Result, file_error};
use crate::pattern::trim;
use crate::polish::prepare_reference;
use crate::wav::pcm16_wav;

/// What a designed voice is looked up and designed with.
#[derive(Debug, Clone, Copy)]
pub struct AnchorSetup<'a> {
    /// The VoxCPM2 take cache directory.
    pub cache_dir: &'a Path,
    /// The voice's key: [`anchor_key`](super::cache::anchor_key).
    pub key: &'a str,
    /// The voice description. `None` lets the model pick its own voice.
    pub voice: Option<&'a str>,
    /// The run's style, parentheses included.
    pub style: Option<&'a str>,
    /// The run's seed.
    pub seed: f64,
    /// Classifier-free guidance.
    pub cfg: f64,
    /// Diffusion steps.
    pub steps: u32,
    /// Hi-Fi cloning: every part clones with the reference's transcript.
    pub hifi: bool,
}

/// A designed voice, ready to clone.
#[derive(Debug, Clone, PartialEq)]
pub struct Anchor {
    /// The voice file in the cache.
    pub file: PathBuf,
    /// The words the voice says, for Hi-Fi cloning. `None` without Hi-Fi,
    /// or when a voice from the cache has no saved transcript.
    pub transcript: Option<String>,
    /// The design, when this run made it. `None` when the cache held it.
    pub designed: Option<DesignedVoice>,
}

/// Writes `bytes` to the cache file `path` whole: [`write_whole_with`],
/// with the cache directory's next action.
pub(crate) fn write_whole(path: &Path, bytes: &[u8]) -> Result<()> {
    write_whole_with(
        path,
        bytes,
        "Check the cache directory is writable and has free space",
    )
}

/// Writes `bytes` to `path` whole: through a `.part` file beside it,
/// renamed over `path` once written. `next` is the action an error names.
pub(crate) fn write_whole_with(path: &Path, bytes: &[u8], next: &str) -> Result<()> {
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    std::fs::write(&part, bytes).map_err(|e| file_error("writing", &part, &e, next))?;
    std::fs::rename(&part, path).map_err(|e| file_error("renaming onto", path, &e, next))
}

pub(crate) fn create_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| {
        file_error(
            "creating",
            dir,
            &e,
            "Check the parent directory is writable",
        )
    })
}

/// The Hi-Fi transcript saved beside a designed voice, trimmed. `None`
/// when there is no file, or it holds only whitespace.
fn saved_transcript(file: &Path) -> Result<Option<String>> {
    if !file.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(file).map_err(|e| {
        file_error(
            "reading",
            file,
            &e,
            "Remove the file to design the voice's transcript again",
        )
    })?;
    let text = trim(&String::from_utf8_lossy(&bytes)).to_string();
    Ok((!text.is_empty()).then_some(text))
}

/// The designed voice `setup.key` names: the cache's copy, else one
/// designed now through `engine` on `script` and kept.
///
/// With Hi-Fi, the transcript saved beside the voice is read first. A voice
/// already in the cache with none logs [`AnchorStep::HifiUntranscribed`]. A
/// transcript read with a run style logs [`AnchorStep::HifiIgnoresStyle`].
/// A new design's slower takes clone with the transcript read, and the new
/// design's own transcript replaces it.
///
/// A new design's slower takes go to `store`. Then the transcript, the
/// first design, the prepared marker and last the voice itself are
/// written. Every step is appended to `log`.
pub fn designed_anchor<E: SpeechEngine, S: TakeStore>(
    script: &str,
    setup: &AnchorSetup<'_>,
    engine: &mut E,
    store: &mut S,
    log: &mut Vec<AnchorStep>,
) -> Result<Anchor> {
    let cache = setup.cache_dir;
    create_dir(cache)?;
    let file = voice_file(cache, setup.key);
    let said = voice_transcript_file(cache, setup.key);
    let mut transcript = None;
    if setup.hifi {
        transcript = saved_transcript(&said)?;
        if transcript.is_none() && file.exists() {
            log.push(AnchorStep::HifiUntranscribed);
        }
        if transcript.is_some() && setup.style.is_some_and(|s| !s.is_empty()) {
            log.push(AnchorStep::HifiIgnoresStyle);
        }
    }
    if file.exists() {
        return Ok(Anchor {
            file,
            transcript,
            designed: None,
        });
    }
    let design = DesignSetup {
        voice: setup.voice,
        style: setup.style,
        seed: setup.seed,
        cfg: setup.cfg,
        steps: setup.steps,
        transcript: transcript.as_deref(),
    };
    let voice = design_voice(script, &design, engine, log)?;
    for (spec, take) in &voice.retries {
        store.save(spec, take)?;
    }
    write_whole(&said, voice.transcript.as_bytes())?;
    write_whole(
        &voice_design_file(cache, setup.key),
        &pcm16_wav(&voice.design, ANCHOR_RATE)?,
    )?;
    let bytes = pcm16_wav(&voice.anchor, ANCHOR_RATE)?;
    let marker = prepared_marker(cache, &content_hash(&bytes));
    if let Some(dir) = marker.parent() {
        create_dir(dir)?;
    }
    write_whole(&marker, b"")?;
    write_whole(&file, &bytes)?;
    Ok(Anchor {
        file,
        transcript: setup.hifi.then(|| voice.transcript.clone()),
        designed: Some(voice),
    })
}

/// How a user's recording came to be ready to clone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferencePrep {
    /// The cache already held its prepared copy.
    Held,
    /// The recording was prepared already (its bytes carry a marker), so
    /// it was copied as it is.
    Copied,
    /// The recording was prepared now.
    Prepared,
}

/// A user's recording, ready to clone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloneFile {
    /// The prepared copy in the cache.
    pub file: PathBuf,
    /// How it came to be there.
    pub prep: ReferencePrep,
}

/// The prepared copy of the user's `reference` recording under `key`
/// ([`prepared_reference_key`](super::cache::prepared_reference_key)). The
/// recording itself never changes.
///
/// A copy the cache holds is used. A recording whose bytes carry a
/// prepared marker (a designed voice's `.voice.wav` passed back) is copied
/// as it is, so its compression does not double. Otherwise `decode` reads
/// the recording as mono samples at [`ANCHOR_RATE`], and the copy is
/// [`prepare_reference`] of them.
pub fn clone_file(
    cache_dir: &Path,
    key: &str,
    reference: &Path,
    decode: &mut dyn FnMut(&Path) -> Result<Vec<f32>>,
) -> Result<CloneFile> {
    create_dir(cache_dir)?;
    let file = reference_file(cache_dir, key);
    if file.exists() {
        return Ok(CloneFile {
            file,
            prep: ReferencePrep::Held,
        });
    }
    let bytes = std::fs::read(reference).map_err(|e| {
        file_error(
            "reading the reference",
            reference,
            &e,
            "Pass a readable recording as the reference",
        )
    })?;
    if prepared_marker(cache_dir, &content_hash(&bytes)).exists() {
        write_whole(&file, &bytes)?;
        return Ok(CloneFile {
            file,
            prep: ReferencePrep::Copied,
        });
    }
    let x = decode(reference)?;
    write_whole(
        &file,
        &pcm16_wav(&prepare_reference(&x, ANCHOR_RATE).0, ANCHOR_RATE)?,
    )?;
    Ok(CloneFile {
        file,
        prep: ReferencePrep::Prepared,
    })
}

/// Copies the designed voice `anchor` beside the output `out` as
/// [`voice_output_file`]: to listen to before judging the rest, or to pass
/// as a reference to give another video the same voice.
pub fn copy_voice_beside(anchor: &Path, out: &Path) -> Result<PathBuf> {
    let target = voice_output_file(out);
    std::fs::copy(anchor, &target).map_err(|e| {
        file_error(
            "copying the designed voice to",
            &target,
            &e,
            "Check the output directory is writable",
        )
    })?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::take::engine::tests::{MemoryStore, ScriptedEngine};

    fn setup<'a>(cache_dir: &'a Path, hifi: bool, style: Option<&'a str>) -> AnchorSetup<'a> {
        AnchorSetup {
            cache_dir,
            key: "k",
            voice: Some("(warm, low)"),
            style,
            seed: 7.0,
            cfg: 1.6,
            steps: 16,
            hifi,
        }
    }

    fn temp() -> Result<tempfile::TempDir> {
        tempfile::tempdir().map_err(|e| Error::File {
            reason: format!("creating a temporary directory: {e}"),
        })
    }

    fn io(e: std::io::Error) -> Error {
        Error::File {
            reason: e.to_string(),
        }
    }

    #[test]
    fn a_new_design_writes_every_file_then_is_reused() -> Result<()> {
        let dir = temp()?;
        let (mut engine, mut store) = (ScriptedEngine::default(), MemoryStore::default());
        let mut log = Vec::new();
        let made = designed_anchor(
            "Hello there.",
            &setup(dir.path(), true, None),
            &mut engine,
            &mut store,
            &mut log,
        )?;
        assert_eq!(made.file, dir.path().join("voice-k.wav"));
        assert_eq!(made.transcript.as_deref(), Some("Hello there."));
        assert_eq!(engine.calls.len(), 2);
        assert_eq!(store.held.len(), 1);
        let said = std::fs::read_to_string(dir.path().join("voice-k.txt")).map_err(io)?;
        assert_eq!(said, "Hello there.");
        assert!(dir.path().join("voice-k.design.wav").exists());
        let bytes = std::fs::read(&made.file).map_err(io)?;
        assert!(prepared_marker(dir.path(), &content_hash(&bytes)).exists());

        let mut again = ScriptedEngine::default();
        let mut log = Vec::new();
        let held = designed_anchor(
            "Other words.",
            &setup(dir.path(), true, Some("(calm)")),
            &mut again,
            &mut store,
            &mut log,
        )?;
        assert!(again.calls.is_empty());
        assert!(held.designed.is_none());
        assert_eq!(held.transcript.as_deref(), Some("Hello there."));
        assert_eq!(log, [AnchorStep::HifiIgnoresStyle]);
        Ok(())
    }

    #[test]
    fn a_held_voice_without_a_transcript_is_noted() -> Result<()> {
        let dir = temp()?;
        std::fs::write(dir.path().join("voice-k.wav"), b"").map_err(io)?;
        let mut log = Vec::new();
        let held = designed_anchor(
            "Hello there.",
            &setup(dir.path(), true, None),
            &mut ScriptedEngine::default(),
            &mut MemoryStore::default(),
            &mut log,
        )?;
        assert_eq!(held.transcript, None);
        assert_eq!(log, [AnchorStep::HifiUntranscribed]);
        Ok(())
    }

    #[test]
    fn a_reference_is_prepared_once_and_a_marked_one_copied() -> Result<()> {
        let dir = temp()?;
        let rec = dir.path().join("rec.wav");
        std::fs::write(&rec, pcm16_wav(&[0.1, -0.2, 0.3], ANCHOR_RATE)?).map_err(io)?;
        let mut decoded = 0;
        let mut decode = |_: &Path| -> Result<Vec<f32>> {
            decoded += 1;
            Ok(vec![0.1, -0.2, 0.3])
        };
        let first = clone_file(dir.path(), "a", &rec, &mut decode)?;
        assert_eq!(first.prep, ReferencePrep::Prepared);
        assert_eq!(
            clone_file(dir.path(), "a", &rec, &mut decode)?.prep,
            ReferencePrep::Held
        );
        let bytes = std::fs::read(&rec).map_err(io)?;
        let marker = prepared_marker(dir.path(), &content_hash(&bytes));
        std::fs::create_dir_all(marker.parent().unwrap_or(dir.path())).map_err(io)?;
        std::fs::write(&marker, b"").map_err(io)?;
        let copied = clone_file(dir.path(), "b", &rec, &mut decode)?;
        assert_eq!(copied.prep, ReferencePrep::Copied);
        assert_eq!(std::fs::read(&copied.file).map_err(io)?, bytes);
        assert_eq!(decoded, 1);
        Ok(())
    }

    #[test]
    fn the_voice_is_copied_beside_the_output() -> Result<()> {
        let dir = temp()?;
        let anchor = dir.path().join("voice-k.wav");
        std::fs::write(&anchor, b"voice").map_err(io)?;
        let copy = copy_voice_beside(&anchor, &dir.path().join("out.wav"))?;
        assert_eq!(copy, dir.path().join("out.voice.wav"));
        assert_eq!(std::fs::read(copy).map_err(io)?, b"voice");
        Ok(())
    }
}
