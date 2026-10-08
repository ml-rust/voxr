//! What a transcribe run prints and writes besides each words file: each
//! narration's words at their times, the video timeline, the voiceover's
//! overrun, and the JSON object `--json` prints.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::outcome::{CommandError, parts};
use super::transcribe_plan::FILE_LABEL;
use crate::error::Result;
use crate::js::js_to_fixed_text;
use crate::transcript::{
    Narration, NarrationManifest, TimelineWords, WordsFile, spoken_words, write_timeline,
};

fn failed(e: &crate::error::Error) -> CommandError {
    CommandError::Failed(parts(e).1.to_string())
}

/// What the run prints for one narration: its file, label, length, word
/// count and script, the words file, then each word at its time. A file
/// passed alone is on its own clock; a manifest's narration is on the
/// video's.
pub fn narration_lines(n: &Narration, words: &WordsFile, out: &Path) -> Vec<String> {
    let from = match &n.script {
        Some(s) => format!(", words from {}", s.display()),
        None => String::new(),
    };
    let clock = if n.label == FILE_LABEL {
        "file"
    } else {
        "video"
    };
    let mut lines = vec![
        format!(
            "{} ({}, {}s, {} words{from}) -> {}",
            n.file.display(),
            n.label,
            js_to_fixed_text(words.duration, 2),
            words.words.len(),
            out.display()
        ),
        format!("  {clock} seconds, start-end, word:"),
    ];
    lines.extend(words.words.iter().map(|w| {
        format!(
            "  {}-{}  {}",
            js_to_fixed_text(n.at + w.start, 2),
            js_to_fixed_text(n.at + w.end, 2),
            w.text
        )
    }));
    lines
}

/// Writes every narration's words on one video timeline, the source of a
/// captions track, and returns its path and the line the run prints.
pub fn write_video_timeline(
    m: &NarrationManifest,
) -> std::result::Result<(PathBuf, String), CommandError> {
    let spoken = spoken_words(m).map_err(|e| failed(&e))?;
    let out =
        write_timeline(&m.out_dir, &TimelineWords::new(&spoken.words)).map_err(|e| failed(&e))?;
    let line = format!(
        "video timeline -> {}  (for <div data-captions=\"../out/voice/words.json\">)",
        out.display()
    );
    Ok((out, line))
}

/// The line a run prints when the voiceover runs past the video's end.
/// `None` when it fits.
pub fn overrun_line(overrun: f64) -> Option<String> {
    (overrun > 0.0).then(|| {
        format!(
            "The voiceover runs {}s past the video's end. Size the scenes to its words above, so the video holds all of it.",
            js_to_fixed_text(overrun, 2)
        )
    })
}

/// One narration as a run read it.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadNarration {
    /// The narration.
    pub narration: Narration,
    /// Its words file.
    pub out: PathBuf,
    /// What it holds.
    pub words: WordsFile,
}

/// The one JSON object `--json` prints: each narration's words file as the
/// skill's `transcribe()` result with its script fields (`file`, `model`,
/// `duration`, `words`, `looped`, `heard`, `phrases`, `script`), with
/// `where`, `at` and `wordsFile` beside it; then the video timeline's path
/// and the voiceover's overrun, when there is one.
pub fn transcribe_json(
    read: &[ReadNarration],
    timeline: Option<&Path>,
    overrun: f64,
) -> Result<Value> {
    let mut list = Vec::with_capacity(read.len());
    for r in read {
        let mut entry = Map::new();
        entry.insert("where".into(), json!(r.narration.label));
        entry.insert("at".into(), json!(r.narration.at));
        entry.insert("wordsFile".into(), json!(r.out.to_string_lossy()));
        let words = serde_json::to_value(&r.words).map_err(|e| crate::error::Error::DataError {
            reason: format!(
                "writing the words of {} as JSON: {e}. Report this as a bug",
                r.narration.file.display()
            ),
        })?;
        if let Value::Object(fields) = words {
            entry.extend(fields);
        }
        list.push(Value::Object(entry));
    }
    let mut object = Map::new();
    object.insert("narrations".into(), Value::Array(list));
    if let Some(t) = timeline {
        object.insert("timeline".into(), json!(t.to_string_lossy()));
    }
    if overrun > 0.0 {
        object.insert("voiceoverOverrun".into(), json!(overrun));
    }
    Ok(Value::Object(object))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::HeardWord;
    use crate::take::DEFAULT_ASR_MODEL;

    fn word(text: &str, start: f64, end: f64) -> HeardWord {
        HeardWord {
            text: text.to_string(),
            start,
            end,
        }
    }

    #[test]
    fn lines_show_each_word_on_its_clock() {
        let n = Narration {
            file: PathBuf::from("/v/vo.wav"),
            at: 1.5,
            label: "voiceover".to_string(),
            script: Some(PathBuf::from("/v/vo.txt")),
        };
        let words = WordsFile {
            file: "/v/vo.wav".to_string(),
            model: DEFAULT_ASR_MODEL.to_string(),
            duration: 2.5,
            words: vec![word("Hi.", 0.1, 0.4)],
            looped: false,
            heard: None,
            phrases: None,
            script: None,
        };
        assert_eq!(
            narration_lines(&n, &words, Path::new("/v/out/voice/vo.words.json")),
            [
                "/v/vo.wav (voiceover, 2.50s, 1 words, words from /v/vo.txt) -> /v/out/voice/vo.words.json",
                "  video seconds, start-end, word:",
                "  1.60-1.90  Hi.",
            ]
        );
        assert_eq!(overrun_line(0.0), None);
        assert_eq!(
            overrun_line(1.234).as_deref(),
            Some(
                "The voiceover runs 1.23s past the video's end. Size the scenes to its words above, so the video holds all of it."
            )
        );
    }

    #[test]
    fn json_lists_each_words_file_with_its_place() -> Result<()> {
        let n = Narration {
            file: PathBuf::from("/v/vo.wav"),
            at: 0.5,
            label: "voiceover".to_string(),
            script: None,
        };
        let words = WordsFile {
            file: "/v/vo.wav".to_string(),
            model: DEFAULT_ASR_MODEL.to_string(),
            duration: 1.0,
            words: vec![word("Hi.", 0.1, 0.4)],
            looped: false,
            heard: None,
            phrases: None,
            script: None,
        };
        let read = [ReadNarration {
            narration: n,
            out: PathBuf::from("/v/out/voice/vo.words.json"),
            words,
        }];
        let v = transcribe_json(&read, Some(Path::new("/v/out/voice/words.json")), 0.0)?;
        let first = &v["narrations"][0];
        assert_eq!(first["where"], "voiceover");
        assert_eq!(first["at"], 0.5);
        assert_eq!(first["wordsFile"], "/v/out/voice/vo.words.json");
        assert_eq!(first["words"][0]["text"], "Hi.");
        assert_eq!(v["timeline"], "/v/out/voice/words.json");
        assert!(v.get("voiceoverOverrun").is_none());
        Ok(())
    }
}
