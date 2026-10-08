//! What the speak command reads and decides before any audio: the script
//! from its file or the words given, a timed script's lines, the engine
//! the script is reviewed for, the script review, the VoxCPM2 speed
//! notice, the written script, and the voiceover's path.

use std::path::{Path, PathBuf};

use super::outcome::{CommandError, Line, parts};
use super::paths::resolve;
use super::speak_errors::speak_failure;
use super::speak_flags::SpeakFlags;
use super::speak_lines::{VOXCPM_SPEED_NOTICE, review_lines};
use crate::pattern::trim;
use crate::take::{DefaultEngineOpts, Engine, SpeakOpts, TtsConfig, default_engine};
use crate::text::{
    ReviewNote, ReviewOptions, TimedCue, WrittenScript, parse_timed_script, review, written_script,
};

/// The voiceover's path when `--out` is not given.
pub const DEFAULT_OUT: &str = "assets/voiceover.wav";

/// A speak run ready to make audio.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedSpeak {
    /// The script file, resolved, when the script argument names one.
    pub source: Option<PathBuf>,
    /// The script the voice speaks: a timed script's lines joined by line
    /// breaks, else the text as given, trimmed.
    pub script: String,
    /// A timed script's lines, each placed at its time.
    pub cues: Option<Vec<TimedCue>>,
    /// The engine the script was reviewed for: `None` when a command or a
    /// model speaks it.
    pub engine_for: Option<String>,
    /// The script review's notes.
    pub review: Vec<ReviewNote>,
    /// The written script for captions, checked against the ear script.
    pub written: Option<WrittenScript>,
    /// `--reference`, resolved against the working directory.
    pub reference: Option<String>,
    /// The voiceover's path, resolved. Its directory is not created here.
    pub out: PathBuf,
    /// `--review`: the run stops after the review, with exit code 0.
    pub review_only: bool,
}

impl PreparedSpeak {
    /// The run options `speak()` receives.
    pub fn opts<'a>(&'a self, flags: &'a SpeakFlags) -> SpeakOpts<'a> {
        flags.opts(self.reference.as_deref(), self.cues.as_deref())
    }
}

/// `opts.engine ?? (opts.command || opts.model ? null : tts.engine ??
/// (tts.command || tts.model ? null : defaultEngine(…)))`: the engine
/// `speak` reviews the script for, before `speak()` resolves its own.
fn engine_for(
    flags: &SpeakFlags,
    tts: &TtsConfig<'_>,
    ready: bool,
    why: &str,
) -> Result<Option<String>, CommandError> {
    let set = |s: &Option<String>| s.as_deref().is_some_and(|s| !s.is_empty());
    let set_cfg = |s: Option<&str>| s.is_some_and(|s| !s.is_empty());
    if let Some(engine) = &flags.engine {
        return Ok(Some(engine.clone()));
    }
    if set(&flags.command) || set(&flags.model) {
        return Ok(None);
    }
    if let Some(engine) = tts.engine {
        return Ok(Some(engine.to_string()));
    }
    if set_cfg(tts.command) || set_cfg(tts.model) {
        return Ok(None);
    }
    let picked = default_engine(
        &DefaultEngineOpts {
            voice: flags.voice.as_deref(),
            reference: flags.reference.as_deref(),
            language: flags.language.as_deref(),
        },
        ready,
        why,
    )
    .map_err(|e| speak_failure(&e, &flags.opts(None, None)))?;
    Ok(Some(
        match picked {
            Engine::Kokoro => "kokoro",
            Engine::Voxcpm => "voxcpm",
        }
        .to_string(),
    ))
}

/// A timed script's lines joined, else `raw` as it is.
fn script_of(raw: &str) -> Result<(String, Option<Vec<TimedCue>>), CommandError> {
    let cues =
        parse_timed_script(raw).map_err(|e| CommandError::Failed(parts(&e).1.to_string()))?;
    let script = match &cues {
        Some(cues) => cues
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        None => raw.to_string(),
    };
    Ok((script, cues))
}

/// Reads `path` as text, a failed run when it cannot be read.
fn read_text(path: &Path) -> Result<String, CommandError> {
    let bytes = std::fs::read(path).map_err(|e| {
        CommandError::Failed(format!(
            "speak: reading {}: {e}. Pass a readable text file",
            path.display()
        ))
    })?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Everything `speakCmd` does before `speak()`, in its order.
///
/// - The script argument, its parts joined by spaces and trimmed, names a
///   file under `cwd` or holds the words. An empty one is a bad input.
/// - A timed script (two lines or more with a time) speaks its lines.
/// - The engine the script is reviewed for: the flag, the config, else
///   the best engine here (`voxcpm_ready`, `voxcpm_why`).
/// - The review prints when it holds a note, or with `--review`, which
///   then stops the run.
/// - VoxCPM2 with a speed other than 1 prints [`VOXCPM_SPEED_NOTICE`].
/// - `--written` must exist and hold the same words, else a bad input.
///
/// The review can fail on a language with no default engine, a failed
/// run. Every line to print goes to `lines` as it comes, so the review
/// prints even when the written script is refused after it.
pub fn prepare_speak(
    flags: &SpeakFlags,
    tts: &TtsConfig<'_>,
    voxcpm_ready: bool,
    voxcpm_why: &str,
    cwd: &Path,
    lines: &mut Vec<Line>,
) -> Result<PreparedSpeak, CommandError> {
    let arg = trim(&flags.script.join(" ")).to_string();
    if arg.is_empty() {
        return Err(CommandError::Input(
            "speak needs the script: a text file, or the words in quotes.".to_string(),
        ));
    }
    let arg_path = resolve(cwd, Path::new(&arg));
    let source = arg_path.exists().then_some(arg_path);
    let raw = match &source {
        Some(path) => trim(&read_text(path)?).to_string(),
        None => arg,
    };
    let (script, cues) = script_of(&raw)?;
    let engine = engine_for(flags, tts, voxcpm_ready, voxcpm_why)?;
    let voxcpm = engine.as_deref() == Some("voxcpm");
    let notes = review(
        &script,
        &ReviewOptions {
            language: flags.language.as_deref(),
            notes: voxcpm,
        },
    )
    .map_err(|e| CommandError::Failed(parts(&e).1.to_string()))?;
    lines.extend(
        review_lines(&notes, flags.review)
            .into_iter()
            .map(Line::out),
    );
    let reference = flags
        .reference
        .as_deref()
        .filter(|r| !r.is_empty())
        .map(|r| resolve(cwd, Path::new(r)).to_string_lossy().into_owned());
    let out = resolve(cwd, Path::new(flags.out.as_deref().unwrap_or(DEFAULT_OUT)));
    let mut prepared = PreparedSpeak {
        source,
        script,
        cues,
        engine_for: engine,
        review: notes,
        written: None,
        reference,
        out,
        review_only: flags.review,
    };
    if flags.review {
        return Ok(prepared);
    }
    if voxcpm && flags.speed.or(tts.speed).unwrap_or(1.0) != 1.0 {
        lines.push(Line::err(VOXCPM_SPEED_NOTICE));
    }
    if let Some(file) = &flags.written {
        let path = resolve(cwd, Path::new(file));
        if !path.exists() {
            return Err(CommandError::Input(format!(
                "speak --written {file}: no such file. Pass the written script: the same words as the ear script, punctuated for reading."
            )));
        }
        let wraw = trim(&read_text(&path)?).to_string();
        let (written, _) = script_of(&wraw)?;
        let checked =
            written_script(&prepared.script, &written).map_err(|e| {
                match speak_failure(&e, &prepared.opts(flags)) {
                    CommandError::Failed(m)
                    | CommandError::Input(m)
                    | CommandError::Unsupported(m) => CommandError::Input(m),
                }
            })?;
        prepared.written = Some(checked);
    }
    Ok(prepared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::outcome::Stream;

    fn flags(script: &str) -> SpeakFlags {
        SpeakFlags {
            script: vec![script.to_string()],
            ..SpeakFlags::default()
        }
    }

    fn prepare(f: &SpeakFlags, ready: bool, cwd: &Path) -> Result<PreparedSpeak, CommandError> {
        prepare_lines(f, ready, cwd).map(|(p, _)| p)
    }

    fn prepare_lines(
        f: &SpeakFlags,
        ready: bool,
        cwd: &Path,
    ) -> Result<(PreparedSpeak, Vec<Line>), CommandError> {
        let mut lines = Vec::new();
        let p = prepare_speak(
            f,
            &TtsConfig::default(),
            ready,
            "needs an NVIDIA GPU",
            cwd,
            &mut lines,
        )?;
        Ok((p, lines))
    }

    #[test]
    fn an_empty_script_is_a_bad_input() {
        let err = prepare(&flags("  "), false, Path::new("/tmp")).err();
        assert_eq!(
            err,
            Some(CommandError::Input(
                "speak needs the script: a text file, or the words in quotes.".to_string()
            ))
        );
    }

    #[test]
    fn words_given_are_the_script_and_kokoro_reviews_them() -> Result<(), CommandError> {
        let (p, lines) = prepare_lines(&flags("One two three"), false, Path::new("/tmp/v"))?;
        assert_eq!(p.script, "One two three");
        assert!(p.source.is_none());
        assert_eq!(p.engine_for.as_deref(), Some("kokoro"));
        assert_eq!(p.out, Path::new("/tmp/v/assets/voiceover.wav"));
        assert_eq!(p.review.len(), 1, "no end punctuation");
        assert_eq!(lines[0].stream, Stream::Out);
        Ok(())
    }

    #[test]
    fn a_file_holds_the_script_and_a_timed_one_its_lines() -> Result<(), CommandError> {
        let dir = tempfile::tempdir().map_err(|e| CommandError::Failed(e.to_string()))?;
        std::fs::write(
            dir.path().join("vo.txt"),
            "0-2s: One two.\n2-4s: Three four.\n",
        )
        .map_err(|e| CommandError::Failed(e.to_string()))?;
        let p = prepare(&flags("vo.txt"), false, dir.path())?;
        assert_eq!(
            p.source.as_deref(),
            Some(dir.path().join("vo.txt").as_path())
        );
        assert_eq!(p.script, "One two.\nThree four.");
        assert_eq!(p.cues.as_ref().map(Vec::len), Some(2));
        Ok(())
    }

    #[test]
    fn voxcpm_with_a_speed_prints_the_notice() -> Result<(), CommandError> {
        let f = SpeakFlags {
            speed: Some(0.9),
            ..flags("One two.")
        };
        let (p, lines) = prepare_lines(&f, true, Path::new("/tmp"))?;
        assert_eq!(p.engine_for.as_deref(), Some("voxcpm"));
        assert_eq!(lines.last(), Some(&Line::err(VOXCPM_SPEED_NOTICE)));
        Ok(())
    }

    #[test]
    fn review_stops_before_the_written_check() -> Result<(), CommandError> {
        let f = SpeakFlags {
            review: true,
            written: Some("missing.txt".to_string()),
            ..flags("One two.")
        };
        let (p, lines) = prepare_lines(&f, false, Path::new("/tmp"))?;
        assert!(p.review_only);
        assert_eq!(
            lines[0].text,
            "Script review: nothing likely to be misread."
        );
        Ok(())
    }

    #[test]
    fn a_written_script_must_exist_and_match() -> Result<(), CommandError> {
        let dir = tempfile::tempdir().map_err(|e| CommandError::Failed(e.to_string()))?;
        let missing = SpeakFlags {
            written: Some("w.txt".to_string()),
            ..flags("One two.")
        };
        assert!(matches!(
            prepare(&missing, false, dir.path()),
            Err(CommandError::Input(m)) if m.starts_with("speak --written w.txt: no such file.")
        ));
        std::fs::write(dir.path().join("w.txt"), "One, three.")
            .map_err(|e| CommandError::Failed(e.to_string()))?;
        assert!(matches!(
            prepare(&missing, false, dir.path()),
            Err(CommandError::Input(m)) if m.starts_with("speak --written: the scripts' words differ")
        ));
        std::fs::write(dir.path().join("w.txt"), "One, two!")
            .map_err(|e| CommandError::Failed(e.to_string()))?;
        let p = prepare(&missing, false, dir.path())?;
        assert_eq!(p.written.map(|w| w.text).as_deref(), Some("One, two!"));
        Ok(())
    }

    #[test]
    fn a_language_with_no_engine_fails_the_run() {
        let f = SpeakFlags {
            language: Some("ms".to_string()),
            ..flags("Satu dua.")
        };
        let err = prepare(&f, false, Path::new("/tmp")).err();
        assert!(matches!(
            err,
            Some(CommandError::Failed(m)) if m.starts_with("speak --language ms: Kokoro speaks English only")
        ));
    }
}
