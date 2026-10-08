//! Which narrations a transcribe run reads: the files passed, or the
//! manifest's voiceover and scene audio, the one script `--script` names,
//! and the recognizer each is read with.

use std::path::{Path, PathBuf};

use super::manifest::{DEFAULT_MANIFEST, LoadedManifest, load_manifest};
use super::outcome::CommandError;
use super::paths::resolve;
use crate::error::Result;
use crate::take::{DEFAULT_ASR_MODEL, SMALL_ASR_MODEL};
use crate::transcript::Narration;

/// The label of a narration passed as a file, read on its own clock.
pub const FILE_LABEL: &str = "file";

/// Every flag `transcribe` takes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscribeFlags {
    /// Audio files to read. None: the manifest's voiceover and every scene
    /// audio.
    pub files: Vec<String>,
    /// `--model`: the recognizer.
    pub model: Option<String>,
    /// `--language`.
    pub language: Option<String>,
    /// `--script`: the one narration's script.
    pub script: Option<String>,
    /// `--manifest`: the video manifest. `None` reads `video.json`.
    pub manifest: Option<String>,
    /// `--manifest` came last, with no value: the skill's manifest path is
    /// then undefined. Files passed are read on their own, and a run with
    /// none fails.
    pub manifest_missing: bool,
}

/// What a transcribe run reads and where it writes.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscribePlan {
    /// Where the words files go: `<out>/voice/`.
    pub out_dir: PathBuf,
    /// The narrations to read, in order.
    pub narrations: Vec<Narration>,
    /// The manifest, when one was read.
    pub manifest: Option<LoadedManifest>,
    /// No file was passed: the run writes the video timeline too.
    pub timeline: bool,
}

/// The recognizer a run reads with: `--model`, else the config's
/// `asr.model`, else the small model for a language other than English
/// and the base model for every other: the skill's `opts.model ??
/// asrModel(opts.language)`.
pub fn transcribe_model(
    model: Option<&str>,
    language: Option<&str>,
    configured: Option<&str>,
) -> String {
    if let Some(m) = model.or(configured) {
        return m.to_string();
    }
    match language {
        Some(l) if !l.is_empty() && !l.get(..2).is_some_and(|p| p.eq_ignore_ascii_case("en")) => {
            SMALL_ASR_MODEL.to_string()
        }
        _ => DEFAULT_ASR_MODEL.to_string(),
    }
}

/// The narrations a run reads, as `transcribeCmd` and `main` pick them.
///
/// - Files passed and no manifest at `--manifest`: the files, each on its
///   own clock, words under `<cwd>/out/voice/`.
/// - Otherwise the manifest is read ([`load_manifest`]). Files passed are
///   read in place of its narrations, and its out directory holds the
///   words.
/// - `--script` needs exactly one narration, and an existing file.
/// - No narration at all is a bad input.
///
/// A narration's own file is checked when it is read
/// ([`check_narration`]), as the skill checks it.
pub fn plan_transcribe(
    flags: &TranscribeFlags,
    cwd: &Path,
    probe: &mut dyn FnMut(&Path) -> Result<f64>,
) -> std::result::Result<TranscribePlan, CommandError> {
    let manifest_arg = Path::new(flags.manifest.as_deref().unwrap_or(DEFAULT_MANIFEST));
    let loose = !flags.files.is_empty();
    if flags.manifest_missing && !loose {
        return Err(CommandError::Failed(
            "transcribe --manifest: no path follows it. Pass the manifest's path, or drop --manifest to read video.json".to_string(),
        ));
    }
    let manifest = if loose && (flags.manifest_missing || !resolve(cwd, manifest_arg).exists()) {
        None
    } else {
        Some(load_manifest(manifest_arg, cwd, probe)?)
    };
    let out_dir = match &manifest {
        Some(m) => m.narration.out_dir.clone(),
        None => resolve(cwd, Path::new("out")),
    };
    let mut list: Vec<Narration> = if loose {
        flags
            .files
            .iter()
            .map(|f| Narration {
                file: resolve(cwd, Path::new(f)),
                at: 0.0,
                label: FILE_LABEL.to_string(),
                script: None,
            })
            .collect()
    } else {
        manifest
            .as_ref()
            .map(|m| crate::transcript::narrations(&m.narration))
            .unwrap_or_default()
    };
    if let Some(script) = &flags.script {
        if list.len() != 1 {
            return Err(CommandError::Input(format!(
                "transcribe --script: {} narrations to read, and one script. Pass one FILE, or set \"script\" per narration in video.json.",
                list.len()
            )));
        }
        let path = resolve(cwd, Path::new(script));
        if !path.exists() {
            return Err(CommandError::Input(format!(
                "transcribe --script: {script} does not exist."
            )));
        }
        list[0].script = Some(path);
    }
    if list.is_empty() {
        return Err(CommandError::Input(
            "transcribe: no narration to read. Set \"voiceover\" or a scene \"audio\" in video.json, or pass a file."
                .to_string(),
        ));
    }
    Ok(TranscribePlan {
        out_dir,
        narrations: list,
        manifest,
        timeline: !loose,
    })
}

/// A narration's file must exist before it is read: a bad input.
pub fn check_narration(n: &Narration) -> std::result::Result<(), CommandError> {
    if n.file.exists() {
        Ok(())
    } else {
        Err(CommandError::Input(format!(
            "transcribe: {} does not exist.",
            n.file.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recognizer_follows_flag_config_then_language() {
        assert_eq!(transcribe_model(Some("m"), Some("ms"), Some("c")), "m");
        assert_eq!(transcribe_model(None, Some("ms"), Some("c")), "c");
        assert_eq!(transcribe_model(None, Some("ms"), None), SMALL_ASR_MODEL);
        assert_eq!(
            transcribe_model(None, Some("en-GB"), None),
            DEFAULT_ASR_MODEL
        );
        assert_eq!(transcribe_model(None, None, None), DEFAULT_ASR_MODEL);
    }

    #[test]
    fn loose_files_need_no_manifest_and_one_script() -> std::result::Result<(), CommandError> {
        let dir = tempfile::tempdir().map_err(|e| CommandError::Failed(e.to_string()))?;
        let mut probe = |_: &Path| -> Result<f64> { Ok(1.0) };
        let flags = TranscribeFlags {
            files: vec!["a.wav".to_string(), "b.wav".to_string()],
            script: Some("s.txt".to_string()),
            ..TranscribeFlags::default()
        };
        assert!(matches!(
            plan_transcribe(&flags, dir.path(), &mut probe),
            Err(CommandError::Input(m)) if m.starts_with("transcribe --script: 2 narrations to read")
        ));
        let one = TranscribeFlags {
            files: vec!["a.wav".to_string()],
            ..TranscribeFlags::default()
        };
        let plan = plan_transcribe(&one, dir.path(), &mut probe)?;
        assert!(plan.manifest.is_none() && !plan.timeline);
        assert_eq!(plan.out_dir, dir.path().join("out"));
        assert_eq!(plan.narrations[0].label, FILE_LABEL);
        assert!(matches!(
            check_narration(&plan.narrations[0]),
            Err(CommandError::Input(m)) if m.ends_with("a.wav does not exist.")
        ));
        Ok(())
    }

    #[test]
    fn a_manifest_flag_with_no_value_reads_files_alone() -> std::result::Result<(), CommandError> {
        let dir = tempfile::tempdir().map_err(|e| CommandError::Failed(e.to_string()))?;
        // A manifest beside the files is not read: the skill's path is
        // undefined, and no file exists there.
        std::fs::write(dir.path().join(DEFAULT_MANIFEST), "{}")
            .map_err(|e| CommandError::Failed(e.to_string()))?;
        let mut probe = |_: &Path| -> Result<f64> { Ok(1.0) };
        let flags = TranscribeFlags {
            files: vec!["a.wav".to_string()],
            manifest_missing: true,
            ..TranscribeFlags::default()
        };
        let plan = plan_transcribe(&flags, dir.path(), &mut probe)?;
        assert!(plan.manifest.is_none() && !plan.timeline);
        let none = TranscribeFlags {
            manifest_missing: true,
            ..TranscribeFlags::default()
        };
        assert!(matches!(
            plan_transcribe(&none, dir.path(), &mut probe),
            Err(CommandError::Failed(m)) if m.starts_with("transcribe --manifest: no path follows it.")
        ));
        Ok(())
    }
}
