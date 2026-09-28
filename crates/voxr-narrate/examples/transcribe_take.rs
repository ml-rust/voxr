//! Reads narration takes into timed words with Whisper, and writes each as
//! a words file: the JSON the narration skill's transcribe command writes.
//!
//! ```text
//! cargo run --release --features whisper --example transcribe_take -- \
//!     --model WHISPER_DIR [--language xx] [--out DIR] AUDIO...
//! ```
//!
//! `WHISPER_DIR` is an HF-layout Whisper checkpoint whose
//! `generation_config.json` lists `alignment_heads`. It loads once, on the
//! CPU in f32, and reads every file. `--language` defaults to `en`.
//!
//! Each file is decoded to 16 kHz mono and read by `transcribe_narration`.
//! With `--out`, the words file of `take.wav` is `DIR/take.words.json`.
//! Without it, each words file goes to stdout. Progress and errors go to
//! stderr. A file that fails is reported and skipped, and the exit code is
//! 1 when any file failed.
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use voxr_narrate::{
    TranscribeOptions, WhisperTranscriber, WordsFile, js_json, read_take_16k, transcribe_narration,
    write_words_file,
};

const USAGE: &str =
    "usage: transcribe_take --model WHISPER_DIR [--language xx] [--out DIR] AUDIO...";

struct Args {
    model: PathBuf,
    language: Option<String>,
    out: Option<PathBuf>,
    audio: Vec<PathBuf>,
}

/// The value after `flag`, advancing `i` past it.
fn take_value(argv: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    argv.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value. {USAGE}"))
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut model = None;
    let mut language = None;
    let mut out = None;
    let mut audio = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        match arg {
            "--model" => model = Some(PathBuf::from(take_value(&argv, &mut i, arg)?)),
            "--language" => language = Some(take_value(&argv, &mut i, arg)?),
            "--out" => out = Some(PathBuf::from(take_value(&argv, &mut i, arg)?)),
            "-h" | "--help" => return Err(USAGE.to_string()),
            flag if flag.starts_with("--") => {
                return Err(format!("unknown flag {flag}. {USAGE}"));
            }
            file => audio.push(PathBuf::from(file)),
        }
        i += 1;
    }
    let model = model.ok_or_else(|| format!("--model is required. {USAGE}"))?;
    if audio.is_empty() {
        return Err(format!("no audio file given. {USAGE}"));
    }
    Ok(Args {
        model,
        language,
        out,
        audio,
    })
}

/// `take.wav` → `take.words.json`: the file name with its last extension
/// replaced.
fn words_name(file: &Path) -> String {
    let stem = file
        .file_stem()
        .map_or_else(|| "take".into(), |s| s.to_string_lossy());
    format!("{stem}.words.json")
}

/// Reads one take and writes its words file.
fn transcribe_one(
    transcriber: &mut WhisperTranscriber,
    args: &Args,
    file: &Path,
) -> voxr_narrate::Result<()> {
    let started = Instant::now();
    let pcm = read_take_16k(file)?;
    let opts = TranscribeOptions {
        language: args.language.clone(),
    };
    let t = transcribe_narration(&pcm, transcriber, &opts)?;
    let (count, duration, looped) = (t.words.len(), t.duration, t.looped);
    let path = std::path::absolute(file).unwrap_or_else(|_| file.to_path_buf());
    let words = WordsFile::new(
        path.display().to_string(),
        args.model.display().to_string(),
        t,
    );
    match &args.out {
        Some(dir) => {
            let dest = dir.join(words_name(file));
            write_words_file(&dest, &words)?;
            eprintln!(
                "{} ({duration:.2}s, {count} words{}) -> {} in {:.1}s",
                file.display(),
                if looped { ", looped" } else { "" },
                dest.display(),
                started.elapsed().as_secs_f64()
            );
        }
        None => {
            println!("{}", js_json(&words)?);
            eprintln!(
                "{} ({duration:.2}s, {count} words{}) in {:.1}s",
                file.display(),
                if looped { ", looped" } else { "" },
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("transcribe_take: {e}");
            return ExitCode::from(2);
        }
    };
    let loading = Instant::now();
    let mut transcriber = match WhisperTranscriber::from_dir(&args.model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("transcribe_take: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "loaded {} in {:.1}s",
        args.model.display(),
        loading.elapsed().as_secs_f64()
    );
    let mut failed = 0usize;
    for file in &args.audio {
        if let Err(e) = transcribe_one(&mut transcriber, &args, file) {
            eprintln!("transcribe_take: {}: {e}", file.display());
            failed += 1;
        }
    }
    if failed > 0 {
        eprintln!(
            "transcribe_take: {failed} of {} files failed",
            args.audio.len()
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
