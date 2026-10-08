//! One narration's words file: what the recognizer heard, with a known
//! script's words placed on the recognized timings, phrase by phrase when
//! the phrase times are known.

use std::path::{Path, PathBuf};

use super::outcome::{CommandError, parts};
use crate::check::align_script;
use crate::pattern::{Patterns, collapse, trim};
use crate::take::{PhrasesFile, phrases_file};
use crate::text::{Cue, read_script, spoken};
use crate::transcript::{
    Narration, Transcribed, WordsFile, align_phrases, fit_file, words_file, write_words_file,
};

fn failed(e: &crate::error::Error) -> CommandError {
    CommandError::Failed(parts(e).1.to_string())
}

/// The phrase spans recorded beside `file`, when they belong to `script`.
fn recorded_phrases(
    file: &Path,
    script: &str,
) -> std::result::Result<Option<(Vec<Cue>, PathBuf)>, CommandError> {
    let pf = phrases_file(file);
    if !pf.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&pf).map_err(|e| {
        CommandError::Failed(format!(
            "reading {}: {e}. Check the file is readable, or remove it",
            pf.display()
        ))
    })?;
    let recorded: PhrasesFile = serde_json::from_slice(&bytes).map_err(|e| {
        CommandError::Failed(format!(
            "reading {}: {e}. Remove it, or speak the script again to rewrite it",
            pf.display()
        ))
    })?;
    let p = Patterns::get().map_err(|e| failed(&e))?;
    if trim(&collapse(p, &recorded.script)) != script {
        return Ok(None);
    }
    let cues = recorded
        .phrases
        .into_iter()
        .map(|w| Cue {
            start: w.start,
            end: w.end,
            text: w.text,
        })
        .collect();
    Ok(Some((cues, pf)))
}

/// One narration's words file from what the recognizer heard (`t`) in
/// `samples_16k`, the file's 16 kHz mono samples.
///
/// With a script, its words replace the recognized ones on the recognized
/// timings, and the recognized words stay in `heard`. The words align
/// phrase by phrase when the phrase times are known: from subtitles (an
/// `.srt` or `.vtt` script), or from the spans speak recorded beside the
/// file for this same script. Each word then settles on the sound.
pub fn narration_words(
    n: &Narration,
    model: &str,
    t: Transcribed,
    samples_16k: &[f32],
) -> std::result::Result<WordsFile, CommandError> {
    let mut words = WordsFile::new(n.file.to_string_lossy(), model, t);
    let Some(script_path) = &n.script else {
        return Ok(words);
    };
    let heard = std::mem::take(&mut words.words);
    let src = read_script(script_path).map_err(|e| failed(&e))?;
    let script = spoken(&src.text).map_err(|e| failed(&e))?;
    let phrases = match src.phrases {
        Some(cues) => Some((cues, script_path.clone())),
        None => recorded_phrases(&n.file, &script)?,
    };
    words.words = match &phrases {
        Some((cues, _)) => {
            let aligned = align_phrases(&heard, cues).map_err(|e| failed(&e))?;
            let mut placed = aligned.words.clone();
            fit_file(&mut placed, samples_16k, Some(aligned.groups(cues)))
                .map_err(|e| failed(&e))?;
            placed
        }
        None => {
            let mut placed = align_script(&heard, &script).map_err(|e| failed(&e))?;
            fit_file(&mut placed, samples_16k, None).map_err(|e| failed(&e))?;
            placed
        }
    };
    words.heard = Some(heard);
    words.phrases = phrases.map(|(_, file)| file.to_string_lossy().into_owned());
    words.script = Some(script_path.to_string_lossy().into_owned());
    Ok(words)
}

/// Writes a narration's words file under `out_dir` and returns its path.
pub fn write_narration(
    out_dir: &Path,
    n: &Narration,
    words: &WordsFile,
) -> std::result::Result<PathBuf, CommandError> {
    let out = words_file(out_dir, &n.file);
    write_words_file(&out, words).map_err(|e| failed(&e))?;
    Ok(out)
}
