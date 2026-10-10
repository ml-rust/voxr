//! What the speak command does after `speak()`: the voice polish, the pace
//! against its target, the script files beside the voiceover, every line
//! it prints, its exit code, and the one JSON object `--json` prints.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::outcome::{CommandError, Line, parts};
use super::paths::relative;
use super::speak_flags::SpeakFlags;
use super::speak_lines::{output_lines, pace_line, polish_line, timed_lines, verdict_lines};
use super::speak_prepare::PreparedSpeak;
use super::speak_rows::{doubtful_parts, heard_back_lines};
use crate::check::HeardWord;
use crate::error::Result;
use crate::polish::{PolishReport, Span, polish};
use crate::take::{
    DEFAULT_TARGET_PACE, PaceSpan, PaceVerdict, PhrasesFile, SpokenScript, TtsConfig,
    output_text_file, pace, pace_verdict, phrases_file, takes_file,
};
use crate::text::spoken;
use crate::wav::pcm16_wav;

/// The rate the polish and the pace read the voiceover at.
const FINISH_RATE: u32 = 48_000;

/// What the finish reads besides the run's result.
#[derive(Debug, Clone, Copy)]
pub struct FinishInputs<'a> {
    /// The command's flags.
    pub flags: &'a SpeakFlags,
    /// The run as prepared.
    pub prepared: &'a PreparedSpeak,
    /// The user's config `tts`: its speed scales the pace suggestion.
    pub tts: &'a TtsConfig<'a>,
    /// The user's config `tts.pace`: the target pace without `--pace`.
    pub tts_pace: Option<f64>,
    /// The working directory the printed paths are relative to.
    pub cwd: &'a Path,
}

/// The files a speak run leaves beside its voiceover.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakFiles {
    /// The voiceover.
    pub out: PathBuf,
    /// The spoken script: `<out>.txt`, or `<out>.spoken.txt` when the
    /// source script sits at `<out>.txt`.
    pub script: PathBuf,
    /// The phrase spans, `<out>.phrases.json`. `None` when the engine
    /// spoke in one call and the file was removed.
    pub phrases: Option<PathBuf>,
    /// VoxCPM2's chosen takes, `<out>.takes.json`.
    pub takes: Option<PathBuf>,
    /// VoxCPM2's designed voice, `<out>.voice.wav`.
    pub voice: Option<PathBuf>,
}

/// A finished speak command.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakOutcome {
    /// 0 when no problem remains, parts marked `ok ?` included: those
    /// need a listen. 1 when a problem remains.
    pub code: i32,
    /// The files written.
    pub files: SpeakFiles,
    /// What the voice polish did. `None` with `--raw`.
    pub polish: Option<PolishReport>,
    /// The pace against its target. `None` when the run has no phrase
    /// spans or no speech.
    pub pace: Option<PaceVerdict>,
    /// The phrase spans as `<out>.phrases.json` holds them: the written
    /// script's words with `--written`.
    pub phrases: Option<Vec<HeardWord>>,
}

/// A file error as a failed run, naming the file and the next action.
fn file_failed(what: &str, path: &Path, e: &std::io::Error) -> CommandError {
    CommandError::Failed(format!(
        "speak: {what} {}: {e}. Check the directory is writable and has free space",
        path.display()
    ))
}

fn failed(e: &crate::error::Error) -> CommandError {
    CommandError::Failed(format!("speak: {}", parts(e).1))
}

/// Polishes the voiceover in place and returns what it did.
fn polish_file(
    out: &Path,
    phrases: &[HeardWord],
    decode: &mut dyn FnMut(&Path) -> Result<Vec<f32>>,
) -> std::result::Result<PolishReport, CommandError> {
    let mut samples = decode(out).map_err(|e| failed(&e))?;
    let spans: Vec<Span> = phrases
        .iter()
        .map(|p| Span {
            start: p.start,
            end: p.end,
        })
        .collect();
    let report = polish(&mut samples, &spans, FINISH_RATE);
    let wav = pcm16_wav(&samples, FINISH_RATE).map_err(|e| failed(&e))?;
    std::fs::write(out, wav).map_err(|e| file_failed("writing", out, &e))?;
    Ok(report)
}

/// The pace over `phrases` on the voiceover as it is on disk.
fn measured_pace(
    out: &Path,
    phrases: &[HeardWord],
    decode: &mut dyn FnMut(&Path) -> Result<Vec<f32>>,
) -> std::result::Result<f64, CommandError> {
    let samples = decode(out).map_err(|e| failed(&e))?;
    let spans: Vec<PaceSpan> = phrases
        .iter()
        .map(|p| PaceSpan {
            text: p.text.clone(),
            start: p.start,
            end: p.end,
        })
        .collect();
    Ok(pace(&spans, Some((&samples, FINISH_RATE))))
}

/// The phrase spans with the written words each covers.
fn written_spans(
    prepared: &PreparedSpeak,
    phrases: &[HeardWord],
) -> std::result::Result<Vec<HeardWord>, CommandError> {
    let Some(written) = &prepared.written else {
        return Ok(phrases.to_vec());
    };
    let texts = written
        .span_texts(
            phrases
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .map_err(|e| failed(&e))?;
    Ok(phrases
        .iter()
        .zip(texts)
        .map(|(p, text)| HeardWord {
            text,
            start: p.start,
            end: p.end,
        })
        .collect())
}

/// Everything `speakCmd` does after `speak()` returns `spoken`, in its
/// order.
///
/// - Unless `--raw`, the voiceover is polished in place and the polish
///   line prints.
/// - The pace prints against `--pace`, else `tts.pace`, else 4.1, when the
///   run has phrase spans.
/// - A timed script prints its placement.
/// - The spoken script goes beside the voiceover (the written one with
///   `--written`), and the phrase spans with it. A run without spans
///   removes a stale spans file.
/// - The output files print, then each part as heard back, then the
///   verdict, which counts the parts marked `ok ?` and sets the exit code.
///   Only a problem exits 1.
///
/// `decode` reads an audio file as 48 kHz mono samples, as the skill's
/// ffmpeg decode does. Every line to print goes to `lines` as it comes.
/// The run's own steps printed while it ran ([`SpeakRuns::note`](crate::SpeakRuns::note)).
pub fn finish_speak(
    spoken_run: &SpokenScript,
    inputs: &FinishInputs<'_>,
    decode: &mut dyn FnMut(&Path) -> Result<Vec<f32>>,
    lines: &mut Vec<Line>,
) -> std::result::Result<SpeakOutcome, CommandError> {
    let FinishInputs {
        flags,
        prepared,
        tts,
        tts_pace,
        cwd,
    } = *inputs;
    let out = prepared.out.as_path();
    let target = flags.pace.or(tts_pace).unwrap_or(DEFAULT_TARGET_PACE);
    let phrases = spoken_run.phrases.as_deref();
    let polished = if flags.raw {
        None
    } else {
        let report = polish_file(out, phrases.unwrap_or(&[]), decode)?;
        lines.push(Line::out(polish_line(&report)));
        Some(report)
    };
    let verdict = match phrases {
        Some(list) => {
            let at = flags.speed.or(tts.speed).unwrap_or(1.0);
            Some(pace_verdict(measured_pace(out, list, decode)?, target, at))
        }
        None => None,
    };
    if let Some(line) = verdict
        .as_ref()
        .and_then(|v| pace_line(v, &spoken_run.engine))
    {
        lines.push(Line::out(line));
    }
    if let Some(cues) = &prepared.cues {
        let timing = spoken_run.timing.as_deref().unwrap_or(&[]);
        lines.extend(timed_lines(cues.len(), timing).into_iter().map(Line::out));
    }
    let txt = output_text_file(out, prepared.source.as_deref());
    let shown = match &prepared.written {
        Some(w) => w.text.clone(),
        None => spoken(&prepared.script).map_err(|e| failed(&e))?,
    };
    std::fs::write(&txt, format!("{shown}\n")).map_err(|e| file_failed("writing", &txt, &e))?;
    let spans_file = phrases_file(out);
    let written = match phrases {
        Some(list) => {
            let spans = written_spans(prepared, list)?;
            let file = PhrasesFile {
                script: shown.clone(),
                phrases: spans.clone(),
            };
            let text = file.write().map_err(|e| failed(&e))?;
            std::fs::write(&spans_file, text)
                .map_err(|e| file_failed("writing", &spans_file, &e))?;
            Some(spans)
        }
        None => {
            match std::fs::remove_file(&spans_file) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(file_failed("removing", &spans_file, &e)),
            }
            None
        }
    };
    let rel_out = relative(cwd, out);
    let rel_txt = relative(cwd, &txt);
    lines.extend(
        output_lines(&rel_out, spoken_run.duration, &spoken_run.engine, &rel_txt)
            .into_iter()
            .map(Line::out),
    );
    lines.extend(
        heard_back_lines(&spoken_run.checks, &spoken_run.engine)
            .into_iter()
            .map(Line::out),
    );
    let (verdict_text, code) =
        verdict_lines(&spoken_run.problems, &doubtful_parts(&spoken_run.checks));
    lines.extend(verdict_text.into_iter().map(Line::out));
    let takes = takes_file(out);
    let files = SpeakFiles {
        out: out.to_path_buf(),
        script: txt,
        phrases: written.as_ref().map(|_| spans_file),
        takes: takes.exists().then_some(takes),
        voice: spoken_run.voice_file.clone().flatten(),
    };
    Ok(SpeakOutcome {
        code,
        files,
        polish: polished,
        pace: verdict,
        phrases: written,
    })
}

/// `p` as JSON text: a path the skill reads back as a string.
fn path_json(p: &Path) -> Value {
    Value::String(p.to_string_lossy().into_owned())
}

impl SpeakOutcome {
    /// The one JSON object `--json` prints: the skill's `speak()` result
    /// (`duration`, `phrases`, `engine`, `checks`, `problems`, `timing`,
    /// `voiceFile`), then what the command adds to it: `exitCode`, `files`,
    /// `polish`, `pace` and the script `review`.
    pub fn json(
        &self,
        spoken_run: &SpokenScript,
        review: &[crate::text::ReviewNote],
    ) -> Result<Value> {
        let mut object = match to_json(spoken_run)? {
            Value::Object(map) => map,
            other => {
                return Err(crate::error::Error::DataError {
                    reason: format!(
                        "the speak result serialized to {other}, not an object. Report this as a bug"
                    ),
                });
            }
        };
        object.insert("exitCode".into(), json!(self.code));
        let mut files = serde_json::Map::new();
        files.insert("out".into(), path_json(&self.files.out));
        files.insert("script".into(), path_json(&self.files.script));
        if let Some(p) = &self.files.phrases {
            files.insert("phrases".into(), path_json(p));
        }
        if let Some(p) = &self.files.takes {
            files.insert("takes".into(), path_json(p));
        }
        if let Some(p) = &self.files.voice {
            files.insert("voice".into(), path_json(p));
        }
        object.insert("files".into(), Value::Object(files));
        if let Some(s) = &self.polish {
            object.insert(
                "polish".into(),
                json!({
                    "leveled": s.leveled,
                    "maxGain": s.max_gain_db,
                    "essShare": s.ess_share,
                    "essDeepest": s.ess_deepest_db,
                }),
            );
        }
        if let Some(v) = &self.pace {
            object.insert(
                "pace".into(),
                json!({
                    "rate": v.rate,
                    "target": v.target,
                    "low": v.low,
                    "high": v.high,
                    "suggestedSpeed": v.suggested_speed,
                }),
            );
        }
        object.insert("review".into(), to_json(review)?);
        Ok(Value::Object(object))
    }
}

/// `v` as a JSON value, a serialization error as a data error.
fn to_json<T: serde::Serialize + ?Sized>(v: &T) -> Result<Value> {
    serde_json::to_value(v).map_err(|e| crate::error::Error::DataError {
        reason: format!("writing the speak result as JSON: {e}. Report this as a bug"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take::{ScriptChecks, WholeCheck};

    #[test]
    fn json_is_the_speak_result_and_what_the_command_adds() -> Result<()> {
        let run = SpokenScript {
            duration: 1.5,
            phrases: Some(vec![HeardWord {
                text: "Hi.".to_string(),
                start: 0.0,
                end: 0.5,
            }]),
            engine: "Kokoro (af_heart)".to_string(),
            checks: ScriptChecks::Whole(vec![WholeCheck {
                at: 0.0,
                text: "Hi.".to_string(),
                ok: true,
                heard: "Hi.".to_string(),
                why: String::new(),
                doubt: Vec::new(),
                pauses: Vec::new(),
            }]),
            problems: Vec::new(),
            timing: None,
            voice_file: None,
        };
        let outcome = SpeakOutcome {
            code: 0,
            files: SpeakFiles {
                out: PathBuf::from("/v/a.wav"),
                script: PathBuf::from("/v/a.txt"),
                phrases: Some(PathBuf::from("/v/a.phrases.json")),
                takes: None,
                voice: None,
            },
            polish: None,
            pace: Some(pace_verdict(4.0, DEFAULT_TARGET_PACE, 1.0)),
            phrases: None,
        };
        let v = outcome.json(&run, &[])?;
        assert_eq!(v["engine"], "Kokoro (af_heart)");
        assert_eq!(v["exitCode"], 0);
        assert_eq!(v["checks"][0]["heard"], "Hi.");
        assert_eq!(v["files"]["phrases"], "/v/a.phrases.json");
        assert!(v["files"].get("takes").is_none() && v.get("polish").is_none());
        assert_eq!(v["pace"]["rate"], 4.0);
        assert_eq!(v["review"], json!([]));
        Ok(())
    }
}
