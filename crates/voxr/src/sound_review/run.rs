//! The review end to end, as the skill's `sound-review.py` runs it: read
//! `sounds.json`, measure every WAV, rank the base variants with a CLAP
//! model unless `--no-model`, and print the table. A caller supplies the
//! model and prints the outcome.

use std::collections::HashMap;
use std::path::Path;

use super::index::SoundIndex;
use super::measure::RawMeasures;
use super::pyfmt::{py_path_join, py_str_repr};
use super::rank::{AudioTextScorer, Rank, model_clip, rank};
use super::report::{Row, header, summary};
use crate::wav::{WavF64, decode_wav_f64};

/// What the review prints and its exit code.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReviewOutcome {
    /// Standard output, each line ending in a newline.
    pub stdout: String,
    /// Standard error, each line ending in a newline.
    pub stderr: String,
    /// 0 when every sound passes or help is asked for, 1 when one fails a
    /// limit or the run stops on an error, 2 with no directory.
    pub code: i32,
}

impl ReviewOutcome {
    fn error(line: String) -> Self {
        Self {
            stderr: format!("{line}\n"),
            code: 1,
            ..Self::default()
        }
    }
}

/// The base variant's suffix: only these are ranked.
const BASE_SUFFIX: &str = "_v0";

/// Reads one WAV as `soundfile.read` reads it. The error is the line it prints.
fn read_wav(path: &str) -> Result<WavF64, String> {
    let opening = |why: &str| {
        format!(
            "soundfile.LibsndfileError: Error opening {}: {why}",
            py_str_repr(path)
        )
    };
    let bytes = std::fs::read(Path::new(path)).map_err(|_| opening("System error."))?;
    if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err(opening("Format not recognised."));
    }
    decode_wav_f64(&bytes).map_err(|e| format!("sound-review: {path}: {e}"))
}

/// Python's `open()` error line for `path`.
fn open_error(path: &str, e: &std::io::Error) -> String {
    let p = py_str_repr(path);
    match e.kind() {
        std::io::ErrorKind::NotFound => {
            format!("FileNotFoundError: [Errno 2] No such file or directory: {p}")
        }
        std::io::ErrorKind::PermissionDenied => {
            format!("PermissionError: [Errno 13] Permission denied: {p}")
        }
        std::io::ErrorKind::IsADirectory => {
            format!("IsADirectoryError: [Errno 21] Is a directory: {p}")
        }
        _ => format!("OSError: {e}: {p}"),
    }
}

/// Runs the review on `argv`, the arguments after the command. `usage` is
/// printed with exit code 2 when no directory is given, and with exit code 0
/// when any argument is `-h` or `--help`. `scorer` loads the model the first
/// time the review ranks; its error is the line printed.
pub fn run_review(
    argv: &[String],
    usage: &str,
    scorer: &mut dyn FnMut() -> Result<Box<dyn AudioTextScorer>, String>,
) -> ReviewOutcome {
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        return ReviewOutcome {
            stdout: format!("{usage}\n"),
            ..ReviewOutcome::default()
        };
    }
    let Some(root) = argv.first() else {
        return ReviewOutcome {
            stdout: format!("{usage}\n"),
            code: 2,
            ..ReviewOutcome::default()
        };
    };
    let no_model = argv.iter().any(|a| a == "--no-model");
    let index_path = py_path_join(root, "sounds.json");
    let text = match std::fs::read_to_string(Path::new(&index_path)) {
        Ok(t) => t,
        Err(e) => return ReviewOutcome::error(open_error(&index_path, &e)),
    };
    let index = match SoundIndex::parse(&text) {
        Ok(i) => i,
        Err(e) => return ReviewOutcome::error(format!("json.decoder.JSONDecodeError: {e}")),
    };

    let mut rows = Vec::with_capacity(index.0.len());
    let mut base: Vec<(String, WavF64, String)> = Vec::new();
    for entry in &index.0 {
        let path = py_path_join(root, &format!("{}.wav", entry.label));
        let wav = match read_wav(&path) {
            Ok(w) => w,
            Err(line) => return ReviewOutcome::error(line),
        };
        let measures = match RawMeasures::of(&wav) {
            Ok(m) => m.rounded(),
            Err(e) => return ReviewOutcome::error(format!("ValueError: {path}: {e}")),
        };
        rows.push(Row::judge(&entry.label, measures, entry.ends_on_cue()));
        if !no_model && entry.label.ends_with(BASE_SUFFIX) {
            let Some(says) = entry.says() else {
                return ReviewOutcome::error("KeyError: 'says'".to_string());
            };
            base.push((entry.label.clone(), wav, says.to_string()));
        }
    }

    let (ranks, texts) = if base.is_empty() {
        (Vec::new(), 0)
    } else {
        match rank_base(&base, scorer) {
            Ok(r) => r,
            Err(line) => return ReviewOutcome::error(line),
        }
    };

    let failing = rows.iter().filter(|r| !r.fails.is_empty()).count();
    let ranked: HashMap<&str, &Rank> = ranks.iter().map(|(l, r)| (l.as_str(), r)).collect();
    let mut stdout = format!("{}\n", header());
    for row in &rows {
        stdout.push_str(&row.line(ranked.get(row.label.as_str()).copied(), texts));
        stdout.push('\n');
    }
    stdout.push_str(&format!("\n{}\n", summary(rows.len(), failing)));
    ReviewOutcome {
        stdout,
        stderr: String::new(),
        code: i32::from(failing > 0),
    }
}

/// Ranks the base variants: each label's rank, and the number of texts.
fn rank_base(
    base: &[(String, WavF64, String)],
    scorer: &mut dyn FnMut() -> Result<Box<dyn AudioTextScorer>, String>,
) -> Result<(Vec<(String, Rank)>, usize), String> {
    let mut texts: Vec<String> = base.iter().map(|(_, _, s)| s.clone()).collect();
    texts.sort();
    texts.dedup();
    let own: Vec<usize> = base
        .iter()
        .map(|(_, _, s)| texts.binary_search(s).unwrap_or(0))
        .collect();
    let clips = base
        .iter()
        .map(|(_, wav, _)| model_clip(wav))
        .collect::<crate::Result<Vec<_>>>()
        .map_err(|e| format!("sound-review: {e}"))?;
    let mut model = scorer()?;
    let scores = model
        .scores(&clips, &texts)
        .map_err(|e| format!("sound-review: {e}"))?;
    let ranks = rank(&scores, &own, &texts).map_err(|e| format!("sound-review: {e}"))?;
    let labeled = base
        .iter()
        .map(|(label, _, _)| label.clone())
        .zip(ranks)
        .collect();
    Ok((labeled, texts.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::encode_wav_pcm16_values;

    struct Fixed;

    impl AudioTextScorer for Fixed {
        fn scores(&mut self, clips: &[Vec<f32>], texts: &[String]) -> crate::Result<Vec<f32>> {
            let n = texts.len();
            Ok((0..clips.len() * n)
                .map(|k| if k / n == k % n { 1.0 } else { 0.0 })
                .collect())
        }
    }

    fn write_catalog(dir: &Path) {
        let tone: Vec<i16> = (0..4800)
            .map(|i| {
                let fade = 1.0 - i as f64 / 4800.0;
                (8000.0 * fade * (i as f64 * 0.05).sin()) as i16
            })
            .collect();
        let wav = encode_wav_pcm16_values(&tone, 48_000).expect("wav");
        for label in ["a_v0", "a_v1", "b_v0"] {
            std::fs::write(dir.join(format!("{label}.wav")), &wav).expect("write");
        }
        std::fs::write(
            dir.join("sounds.json"),
            r#"{"a_v0": {"says": "an a"}, "a_v1": {"says": "an a"}, "b_v0": {"says": "a b"}}"#,
        )
        .expect("index");
    }

    #[test]
    fn no_directory_prints_usage_and_exits_2() {
        let out = run_review(&[], "USAGE", &mut || Err("unused".to_string()));
        assert_eq!((out.stdout.as_str(), out.code), ("USAGE\n", 2));
    }

    #[test]
    fn help_prints_usage_and_exits_0() {
        for flag in ["-h", "--help"] {
            let out = run_review(&[flag.to_string()], "USAGE", &mut || {
                Err("unused".to_string())
            });
            assert_eq!((out.stdout.as_str(), out.code), ("USAGE\n", 0), "{flag}");
            assert!(out.stderr.is_empty(), "{flag}: {}", out.stderr);
        }
        let after_dir = run_review(
            &["/nonexistent".to_string(), "--help".to_string()],
            "USAGE",
            &mut || Err("unused".to_string()),
        );
        assert_eq!(after_dir.code, 0);
    }

    #[test]
    fn a_missing_index_reads_as_python_reports_it() {
        let out = run_review(&["/nonexistent".to_string()], "", &mut || {
            Err("unused".to_string())
        });
        assert_eq!(out.code, 1);
        assert_eq!(
            out.stderr,
            "FileNotFoundError: [Errno 2] No such file or directory: '/nonexistent/sounds.json'\n"
        );
    }

    #[test]
    fn a_catalog_is_measured_and_ranked() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_catalog(dir.path());
        let root = dir.path().display().to_string();
        let out = run_review(std::slice::from_ref(&root), "", &mut || Ok(Box::new(Fixed)));
        let lines: Vec<&str> = out.stdout.lines().collect();
        assert_eq!(lines.len(), 6, "{}", out.stdout);
        assert!(lines[1].starts_with("a_v0 ") && lines[1].contains("/2 "));
        assert!(lines[2].starts_with("a_v1 ") && !lines[2].contains('/'));
        assert_eq!(lines[5], "3 sounds, 0 failing a numeric limit.");
        assert_eq!(out.code, 0);
        let skipped = run_review(&[root, "--no-model".to_string()], "", &mut || {
            Err("the model must not load".to_string())
        });
        assert_eq!(skipped.code, 0, "{}", skipped.stderr);
    }
}
