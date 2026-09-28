//! The user's own TTS command as a speech engine: the skill's `command()`.
//!
//! The command template takes five placeholders: `{text}` (the text,
//! shell-quoted), `{text_file}` (a file holding it), `{out}` (the audio
//! file the command writes, in any format the decoder reads), `{voice}` and
//! `{reference}` (a recording to clone). Each take runs the command once.
//! The caller runs the shell line and decodes its output: this crate never
//! spawns a process.

use std::path::Path;

use super::engine::{SpeakRequest, SpeechEngine, Take};
use super::join::JOIN_RATE;
use crate::error::{Error, Result, file_error};

/// The file in the work directory that holds the text of a take.
pub const TEXT_FILE: &str = "script.txt";

/// The file in the work directory the command writes its audio to.
pub const OUT_FILE: &str = "speech.wav";

/// The next action an error writing into the work directory names.
const DIR_NEXT: &str = "Check the work directory exists, is writable and has free space";

/// `String.prototype.replaceAll(pattern, replacement)` with a non-empty
/// string pattern: each occurrence, left to right, replaced by
/// `replacement` read as JavaScript reads it. `$$` is a `$`, `$&` the
/// match, `` $` `` the text before the match and `$'` the text after it.
/// Any other `$` stays as it is.
pub(crate) fn js_replace_all(haystack: &str, pattern: &str, replacement: &str) -> String {
    if pattern.is_empty() {
        return haystack.to_string();
    }
    let mut out = String::with_capacity(haystack.len());
    let mut last = 0;
    for (at, matched) in haystack.match_indices(pattern) {
        out.push_str(&haystack[last..at]);
        let after = &haystack[at + matched.len()..];
        substitute(&mut out, replacement, matched, &haystack[..at], after);
        last = at + matched.len();
    }
    out.push_str(&haystack[last..]);
    out
}

/// Appends `replacement` to `out`, its `$` patterns read as
/// [`js_replace_all`] reads them.
fn substitute(out: &mut String, replacement: &str, matched: &str, before: &str, after: &str) {
    let mut rest = replacement;
    while let Some(at) = rest.find('$') {
        out.push_str(&rest[..at]);
        let tail = &rest[at + 1..];
        let piece = match tail.as_bytes().first() {
            Some(b'$') => Some("$"),
            Some(b'&') => Some(matched),
            Some(b'`') => Some(before),
            Some(b'\'') => Some(after),
            _ => None,
        };
        match piece {
            Some(p) => {
                out.push_str(p);
                rest = &tail[1..];
            }
            None => {
                out.push('$');
                rest = tail;
            }
        }
    }
    out.push_str(rest);
}

/// `text` quoted for the shell: in double quotes with each `"` escaped on
/// Windows, else in single quotes with each `'` written as `'\''`.
pub fn shell_quote(text: &str, windows: bool) -> String {
    if windows {
        format!("\"{}\"", text.replace('"', "\\\""))
    } else {
        format!("'{}'", text.replace('\'', "'\\''"))
    }
}

/// The shell line of one take: `template` with `{text_file}`, `{out}`,
/// `{voice}`, `{reference}` and then `{text}` replaced, in that order, each
/// by its value quoted for the shell. A placeholder inside a value
/// substituted earlier is replaced too, as the skill replaces it.
pub fn command_line(
    template: &str,
    text: &str,
    text_file: &Path,
    out: &Path,
    voice: &str,
    reference: &str,
    windows: bool,
) -> String {
    let q = |t: &str| shell_quote(t, windows);
    let line = js_replace_all(template, "{text_file}", &q(&text_file.to_string_lossy()));
    let line = js_replace_all(&line, "{out}", &q(&out.to_string_lossy()));
    let line = js_replace_all(&line, "{voice}", &q(voice));
    let line = js_replace_all(&line, "{reference}", &q(reference));
    js_replace_all(&line, "{text}", &q(text))
}

/// The user's TTS command as a [`SpeechEngine`]: the skill's `command()`
/// for one take.
///
/// Each take writes its text and a line break to [`TEXT_FILE`] in `dir`,
/// removes any [`OUT_FILE`] left there, and runs [`command_line`] through
/// `run`. `run` returns the exit code, `None` when a signal ends the
/// command. `decode` reads the output as mono samples at
/// [`JOIN_RATE`], `None` when it holds no audio it reads. The caller owns
/// `dir` and removes it after the run. The engine reads only the request's
/// text: the voice and reference are its own, and the speed is ignored.
pub struct CommandEngine<'a, R, D> {
    /// The command template.
    pub template: &'a str,
    /// The voice `{voice}` names; empty for none.
    pub voice: &'a str,
    /// The recording `{reference}` names; empty for none.
    pub reference: &'a str,
    /// The work directory for [`TEXT_FILE`] and [`OUT_FILE`].
    pub dir: &'a Path,
    /// Runs one shell line and returns its exit code.
    pub run: R,
    /// Decodes the output file at [`JOIN_RATE`] mono.
    pub decode: D,
    /// Quotes for the Windows shell: `cfg!(windows)`.
    pub windows: bool,
}

impl<R, D> SpeechEngine for CommandEngine<'_, R, D>
where
    R: FnMut(&str) -> Result<Option<i32>>,
    D: FnMut(&Path) -> Option<Vec<f32>>,
{
    type Reference = ();

    fn speak(&mut self, request: &SpeakRequest<'_, ()>) -> Result<Take> {
        let text = request.text;
        let text_file = self.dir.join(TEXT_FILE);
        let raw = self.dir.join(OUT_FILE);
        match std::fs::remove_file(&raw) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(file_error("removing", &raw, &e, DIR_NEXT));
            }
            _ => {}
        }
        std::fs::write(&text_file, format!("{text}\n"))
            .map_err(|e| file_error("writing", &text_file, &e, DIR_NEXT))?;
        let line = command_line(
            self.template,
            text,
            &text_file,
            &raw,
            self.voice,
            self.reference,
            self.windows,
        );
        let status = (self.run)(&line)?;
        if status != Some(0) {
            let code = status.map_or_else(|| "null".to_string(), |c| c.to_string());
            return Err(Error::Engine {
                reason: format!(
                    "the TTS command exited {code} on \"{text}\": {}",
                    self.template
                ),
            });
        }
        match (self.decode)(&raw) {
            Some(samples) if !samples.is_empty() => Ok(Take {
                samples,
                rate: JOIN_RATE,
            }),
            _ => Err(Error::Engine {
                reason: "the TTS command wrote no audio ffmpeg can read at {out}. Check that the command writes its output to the {out} path.".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take::engine::VoiceSource;

    #[test]
    fn replacement_patterns_read_as_javascript_reads_them() {
        assert_eq!(js_replace_all("a-b-a", "a", "x"), "x-b-x");
        assert_eq!(js_replace_all("a-b", "a", "$$"), "$-b");
        assert_eq!(js_replace_all("a-b", "a", "[$&]"), "[a]-b");
        assert_eq!(js_replace_all("x-a-y", "a", "$`"), "x-x--y");
        assert_eq!(js_replace_all("x-a-y", "a", "$'"), "x--y-y");
        assert_eq!(js_replace_all("a", "a", "$1 $< $"), "$1 $< $");
        assert_eq!(js_replace_all("aaa", "aa", "b"), "ba");
        assert_eq!(js_replace_all("abc", "", "x"), "abc");
    }

    #[test]
    fn quotes_for_each_shell() {
        assert_eq!(shell_quote("it's", false), "'it'\\''s'");
        assert_eq!(shell_quote("say \"hi\"", true), "\"say \\\"hi\\\"\"");
    }

    #[test]
    fn placeholders_are_replaced_in_the_skills_order() {
        let line = command_line(
            "tts {text} -f {text_file} -o {out} -v {voice} -r {reference}",
            "Pay $$ now.",
            Path::new("/w/script.txt"),
            Path::new("/w/speech.wav"),
            "o'neil",
            "ref {text}.wav",
            false,
        );
        assert_eq!(
            line,
            "tts 'Pay $ now.' -f '/w/script.txt' -o '/w/speech.wav' -v 'o'\\''neil' -r 'ref 'Pay $ now.'.wav'"
        );
    }

    type Run<'a> = Box<dyn FnMut(&str) -> Result<Option<i32>> + 'a>;
    type Decode<'a> = Box<dyn FnMut(&Path) -> Option<Vec<f32>> + 'a>;

    fn engine<'a>(
        dir: &'a Path,
        status: Option<i32>,
        audio: Option<Vec<f32>>,
        lines: &'a mut Vec<String>,
    ) -> CommandEngine<'a, Run<'a>, Decode<'a>> {
        CommandEngine {
            template: "tts {text} {out}",
            voice: "",
            reference: "",
            dir,
            run: Box::new(move |line: &str| -> Result<Option<i32>> {
                lines.push(line.to_string());
                Ok(status)
            }),
            decode: Box::new(move |_: &Path| audio.clone()),
            windows: false,
        }
    }

    fn request(text: &str) -> SpeakRequest<'_, ()> {
        SpeakRequest {
            text,
            delivery: None,
            style: None,
            voice: VoiceSource::Own { design: None },
            seed: 7.0,
            cfg: 1.6,
            steps: 16,
            speed: 1.0,
        }
    }

    fn tempdir() -> Result<tempfile::TempDir> {
        tempfile::tempdir().map_err(|e| Error::File {
            reason: format!("creating a temporary directory: {e}"),
        })
    }

    #[test]
    fn a_take_writes_its_text_and_decodes_the_output() -> Result<()> {
        let dir = tempdir()?;
        let mut lines = Vec::new();
        let take = engine(dir.path(), Some(0), Some(vec![0.5; 4]), &mut lines)
            .speak(&request("Hello there."))?;
        assert_eq!(take.samples, [0.5; 4]);
        assert_eq!(take.rate, JOIN_RATE);
        let written = std::fs::read_to_string(dir.path().join(TEXT_FILE))
            .map_err(|e| file_error("reading", dir.path(), &e, "test"))?;
        assert_eq!(written, "Hello there.\n");
        let out = dir.path().join(OUT_FILE);
        assert_eq!(lines, [format!("tts 'Hello there.' '{}'", out.display())]);
        Ok(())
    }

    #[test]
    fn a_nonzero_exit_or_no_audio_is_refused() -> Result<()> {
        let dir = tempdir()?;
        let mut lines = Vec::new();
        let exited = engine(dir.path(), Some(3), None, &mut lines)
            .speak(&request("Hi."))
            .err()
            .map(|e| e.to_string());
        assert_eq!(
            exited.as_deref(),
            Some("speech engine error: the TTS command exited 3 on \"Hi.\": tts {text} {out}")
        );
        let killed = engine(dir.path(), None, None, &mut lines)
            .speak(&request("Hi."))
            .err()
            .map(|e| e.to_string());
        assert!(killed.is_some_and(|m| m.contains("exited null on")));
        for audio in [None, Some(Vec::new())] {
            let silent = engine(dir.path(), Some(0), audio, &mut lines)
                .speak(&request("Hi."))
                .err()
                .map(|e| e.to_string());
            assert!(silent.is_some_and(|m| m.contains("wrote no audio ffmpeg can read")));
        }
        Ok(())
    }
}
