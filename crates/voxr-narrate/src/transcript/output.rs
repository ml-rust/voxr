//! Transcript files: a narration's words file, every narration's words on
//! the video timeline, and the JSON layout both are written in.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::ser::{Formatter, PrettyFormatter};

use super::transcriber::Transcribed;
use crate::check::HeardWord;
use crate::error::{Error, Result};
use crate::js::{js_max, js_string, js_to_fixed};

/// Seconds between a scene's start and its narration's start.
pub const AUDIO_LEAD: f64 = 0.3;
/// Seconds a scene holds after its narration ends.
pub const AUDIO_TAIL: f64 = 0.5;
/// Decimals of the times on the video timeline.
const TIME_DECIMALS: usize = 3;

/// One narration's words file, as written beside the video's output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordsFile {
    /// The narration's audio file.
    pub file: String,
    /// The recognizer model that read it.
    pub model: String,
    /// The audio's length in seconds, rounded to milliseconds.
    pub duration: f64,
    /// The words: the script's when one is known, else the recognized ones.
    pub words: Vec<HeardWord>,
    /// The recognizer looped on a phrase and the loop was collapsed.
    pub looped: bool,
    /// With a script, the words as recognized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heard: Option<Vec<HeardWord>>,
    /// With phrase spans, the file they came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phrases: Option<String>,
    /// The script file whose words these are.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
}

impl WordsFile {
    /// The words file of a take read by `model`, with no script.
    pub fn new(file: impl Into<String>, model: impl Into<String>, t: Transcribed) -> Self {
        Self {
            file: file.into(),
            model: model.into(),
            duration: t.duration,
            words: t.words,
            looped: t.looped,
            heard: None,
            phrases: None,
            script: None,
        }
    }
}

/// Every narration's words on the video timeline: the source of a captions
/// track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineWords {
    /// The words in video seconds, rounded to milliseconds, by start.
    pub words: Vec<HeardWord>,
}

impl TimelineWords {
    /// Words with their times rounded to milliseconds.
    pub fn new(words: &[HeardWord]) -> Self {
        let words = words
            .iter()
            .map(|w| HeardWord {
                text: w.text.clone(),
                start: js_to_fixed(w.start, TIME_DECIMALS),
                end: js_to_fixed(w.end, TIME_DECIMALS),
            })
            .collect();
        Self { words }
    }
}

/// serde_json's pretty layout with a one-space indent, numbers written as
/// JavaScript writes them.
struct JsFormatter(PrettyFormatter<'static>);

impl Formatter for JsFormatter {
    fn write_f64<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(js_string(value).as_bytes())
    }

    fn write_f32<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        writer.write_all(js_string(f64::from(value)).as_bytes())
    }

    fn begin_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.begin_array(writer)
    }

    fn end_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_array(writer)
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.0.begin_array_value(writer, first)
    }

    fn end_array_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_array_value(writer)
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.begin_object(writer)
    }

    fn end_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_object(writer)
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.0.begin_object_key(writer, first)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.begin_object_value(writer)
    }

    fn end_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_object_value(writer)
    }
}

/// `JSON.stringify(value, null, 1)`: one-space indent, numbers as
/// JavaScript writes them (`2`, `1.5`, `0.30000000000000004`), NaN and
/// infinities as `null`, and no trailing newline.
pub fn js_json<T: Serialize>(value: &T) -> Result<String> {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(
        &mut out,
        JsFormatter(PrettyFormatter::with_indent(b" ")),
    );
    value.serialize(&mut ser).map_err(|e| Error::DataError {
        reason: format!("writing JSON: {e}"),
    })?;
    String::from_utf8(out).map_err(|e| Error::DataError {
        reason: format!("writing JSON: {e}"),
    })
}

/// Where a narration's words file goes: `<out_dir>/voice/<name>.words.json`,
/// the audio file's name with its last extension removed.
pub fn words_file(out_dir: &Path, file: &Path) -> PathBuf {
    let name = file
        .components()
        .next_back()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = match name.rfind('.') {
        Some(i) if i + 1 < name.len() => &name[..i],
        _ => name.as_str(),
    };
    out_dir.join("voice").join(format!("{stem}.words.json"))
}

/// Writes a words file, creating its directory.
pub fn write_words_file(path: &Path, words: &WordsFile) -> Result<()> {
    write_text(path, &js_json(words)?)
}

/// Writes every narration's words on the video timeline to
/// `<out_dir>/voice/words.json`, and returns that path.
pub fn write_timeline(out_dir: &Path, words: &TimelineWords) -> Result<PathBuf> {
    let path = out_dir.join("voice").join("words.json");
    write_text(&path, &js_json(words)?)?;
    Ok(path)
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    let io_error = |e: io::Error| Error::DataError {
        reason: format!("writing {}: {e}", path.display()),
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(io_error)?;
    }
    std::fs::write(path, text).map_err(io_error)
}

/// A voiceover: one narration over the whole video.
#[derive(Debug, Clone, PartialEq)]
pub struct Voiceover {
    /// The audio file.
    pub file: PathBuf,
    /// Where it starts, in video seconds. Negative skips its lead.
    pub at: f64,
    /// Its script file, if any.
    pub script: Option<PathBuf>,
}

/// A scene, and the narration it plays.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneNarration {
    /// The scene's name.
    pub name: String,
    /// The scene's start, in video seconds.
    pub start: f64,
    /// The narration it plays, if any.
    pub audio: Option<PathBuf>,
    /// The narration's script file, if any.
    pub script: Option<PathBuf>,
}

/// The parts of a video manifest that place narration.
#[derive(Debug, Clone, PartialEq)]
pub struct NarrationManifest {
    /// The video's output directory.
    pub out_dir: PathBuf,
    /// The voiceover, if any.
    pub voiceover: Option<Voiceover>,
    /// Every scene, in order.
    pub scenes: Vec<SceneNarration>,
}

/// A narration file and where it plays.
#[derive(Debug, Clone, PartialEq)]
pub struct Narration {
    /// The audio file.
    pub file: PathBuf,
    /// Where it starts, in video seconds.
    pub at: f64,
    /// `voiceover`, or `scene <name>`.
    pub label: String,
    /// Its script file, if any.
    pub script: Option<PathBuf>,
}

/// Every narration file in the manifest and where it plays: the voiceover
/// first, then each scene's audio, [`AUDIO_LEAD`] after the scene starts.
/// An empty audio path is no narration.
pub fn narrations(m: &NarrationManifest) -> Vec<Narration> {
    let mut list = Vec::new();
    if let Some(v) = &m.voiceover {
        list.push(Narration {
            file: v.file.clone(),
            at: v.at,
            label: "voiceover".to_string(),
            script: v.script.clone(),
        });
    }
    for s in &m.scenes {
        if let Some(audio) = s.audio.as_ref().filter(|a| !a.as_os_str().is_empty()) {
            list.push(Narration {
                file: audio.clone(),
                at: s.start + AUDIO_LEAD,
                label: format!("scene {}", s.name),
                script: s.script.clone(),
            });
        }
    }
    list
}

/// Transcribed words on the video timeline, and the narrations with no
/// words file yet.
#[derive(Debug, Clone, PartialEq)]
pub struct Spoken {
    /// Every narration's words in video seconds, by start.
    pub words: Vec<HeardWord>,
    /// Narrations whose words file does not exist.
    pub missing: Vec<Narration>,
}

/// The `words` of a words file; the rest is not read.
#[derive(Deserialize)]
struct WordsOnly {
    words: Vec<HeardWord>,
}

/// All transcribed words on the video timeline, read from each narration's
/// words file.
///
/// Each word moves by its narration's start. Words in a skipped lead (a
/// negative voiceover start) are never heard in the video: a word ending
/// at or before 0 is dropped, and one straddling 0 starts at 0. The words
/// are sorted by start, keeping file order on a tie.
pub fn spoken_words(m: &NarrationManifest) -> Result<Spoken> {
    let mut words = Vec::new();
    let mut missing = Vec::new();
    for n in narrations(m) {
        let f = words_file(&m.out_dir, &n.file);
        if !f.exists() {
            missing.push(n);
            continue;
        }
        let read_error = |reason: String| Error::DataError {
            reason: format!("reading {}: {reason}", f.display()),
        };
        let bytes = std::fs::read(&f).map_err(|e| read_error(e.to_string()))?;
        let file: WordsOnly =
            serde_json::from_slice(&bytes).map_err(|e| read_error(e.to_string()))?;
        for w in file.words {
            if n.at + w.end > 0.0 {
                words.push(HeardWord {
                    text: w.text,
                    start: js_max(0.0, n.at + w.start),
                    end: n.at + w.end,
                });
            }
        }
    }
    words.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(Spoken { words, missing })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64) -> HeardWord {
        HeardWord {
            text: text.to_string(),
            start,
            end,
        }
    }

    #[test]
    fn json_numbers_follow_javascript() -> Result<()> {
        let t = TimelineWords {
            words: vec![word("a\"b\n", 1.5, 2.0), word("c", 0.1 + 0.2, f64::NAN)],
        };
        assert_eq!(
            js_json(&t)?,
            "{\n \"words\": [\n  {\n   \"text\": \"a\\\"b\\n\",\n   \"start\": 1.5,\n   \"end\": 2\n  },\n  {\n   \"text\": \"c\",\n   \"start\": 0.30000000000000004,\n   \"end\": null\n  }\n ]\n}"
        );
        assert_eq!(
            js_json(&TimelineWords { words: Vec::new() })?,
            "{\n \"words\": []\n}"
        );
        Ok(())
    }

    #[test]
    fn words_file_drops_the_last_extension() {
        let out = Path::new("out");
        assert_eq!(
            words_file(out, Path::new("a/voiceover.wav")),
            Path::new("out/voice/voiceover.words.json")
        );
        assert_eq!(
            words_file(out, Path::new("take.v2.mp3")),
            Path::new("out/voice/take.v2.words.json")
        );
        assert_eq!(
            words_file(out, Path::new("noext")),
            Path::new("out/voice/noext.words.json")
        );
        assert_eq!(
            words_file(out, Path::new("file.")),
            Path::new("out/voice/file..words.json")
        );
        assert_eq!(
            words_file(out, Path::new(".hidden")),
            Path::new("out/voice/.words.json")
        );
    }

    #[test]
    fn words_file_omits_unset_fields() -> Result<()> {
        let t = Transcribed {
            words: vec![word("hi", 0.5, 1.0)],
            looped: false,
            duration: 2.0,
        };
        let text = js_json(&WordsFile::new("a.wav", "m", t))?;
        assert!(
            text.starts_with("{\n \"file\": \"a.wav\",\n \"model\": \"m\",\n \"duration\": 2,\n")
        );
        assert!(text.ends_with("\"looped\": false\n}"));
        Ok(())
    }

    #[test]
    fn scenes_play_after_their_lead() {
        let m = NarrationManifest {
            out_dir: PathBuf::from("out"),
            voiceover: None,
            scenes: vec![
                SceneNarration {
                    name: "intro".into(),
                    start: 2.0,
                    audio: Some(PathBuf::from("intro.wav")),
                    script: None,
                },
                SceneNarration {
                    name: "quiet".into(),
                    start: 5.0,
                    audio: None,
                    script: None,
                },
            ],
        };
        let list = narrations(&m);
        assert_eq!(list.len(), 1);
        assert_eq!((list[0].at, list[0].label.as_str()), (2.3, "scene intro"));
    }
}
