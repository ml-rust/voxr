//! The take-loop's files: a run's `.takes.json`, a take's `.phrases.json`,
//! and the script text files beside the audio.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use crate::check::HeardWord;
use crate::error::Result;
use crate::transcript::output::js_json;

/// The default seed, when nothing else names one.
pub const DEFAULT_SEED: f64 = 7.0;

/// `<out>.takes.json`, read back: the seed a run drew, and the seed that
/// made the take kept for each part, by the part's own text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TakesFile {
    /// The run's own seed.
    pub seed: f64,
    /// Text to the seed of its kept take, in the order takes were made.
    pub takes: Vec<(String, f64)>,
}

impl Serialize for TakesFile {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("seed", &self.seed)?;
        map.serialize_entry("takes", &TakesMap(&self.takes))?;
        map.end()
    }
}

struct TakesMap<'a>(&'a [(String, f64)]);

impl Serialize for TakesMap<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (text, seed) in self.0 {
            map.serialize_entry(text, seed)?;
        }
        map.end()
    }
}

/// A `.takes.json` read leniently: a `takes` object at the top level is the
/// current format, and its `seed` is the run's own. Anything else — an old
/// cache written before the `takes` wrapper existed, or a foreign file — is
/// read as its own flat text-to-seed map, with no run seed of its own, the
/// same fallback the skill's `current`/`chosen` pair falls to. A value that
/// is not a number is dropped: the skill keeps a foreign document verbatim,
/// but a seed is never anything else once it reaches the take loop.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReadTakes {
    /// The run's seed, when the file holds the current format.
    pub seed: Option<f64>,
    /// Each part's kept seed, by its text.
    pub takes: HashMap<String, f64>,
}

fn number_map(value: &Value) -> HashMap<String, f64> {
    value
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
                .collect()
        })
        .unwrap_or_default()
}

/// Reads a `.takes.json` document. Invalid JSON reads as no saved takes,
/// the same as the skill's `catch {}` around `JSON.parse`.
pub fn read_takes_file(text: &str) -> ReadTakes {
    let Ok(value) = serde_json::from_str::<Value>(text) else {
        return ReadTakes::default();
    };
    match value.get("takes").filter(|t| t.is_object()) {
        Some(takes) => ReadTakes {
            seed: value.get("seed").and_then(Value::as_f64),
            takes: number_map(takes),
        },
        None => ReadTakes {
            seed: None,
            takes: number_map(&value),
        },
    }
}

/// The seed a run resolves to, and whether it resets the chosen takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedSeed {
    /// `asked ?? before ?? configSeed ?? 7`.
    pub seed: f64,
    /// An asked seed that differs from `before`: the chosen takes reset to
    /// empty, and every part draws a fresh take at the new seed.
    pub reset: bool,
}

/// `asked ?? before ?? configSeed ?? 7`, where `before` is `read.seed` when
/// the file held the current format, else `configSeed ?? 7`; and whether a
/// new asked seed resets the chosen takes.
pub fn resolve_seed(
    asked: Option<f64>,
    read: &ReadTakes,
    config_seed: Option<f64>,
) -> ResolvedSeed {
    let before = read.seed.or(config_seed).unwrap_or(DEFAULT_SEED);
    match asked {
        Some(a) => ResolvedSeed {
            seed: a,
            reset: a != before,
        },
        None => ResolvedSeed {
            seed: before,
            reset: false,
        },
    }
}

/// The takes chosen for this run: empty on a reset, else `read.takes`.
pub fn chosen_takes(read: ReadTakes, resolved: &ResolvedSeed) -> HashMap<String, f64> {
    if resolved.reset {
        HashMap::new()
    } else {
        read.takes
    }
}

impl TakesFile {
    /// Writes this file in the skill's own layout: `JSON.stringify(v,
    /// null, 1)`.
    pub fn write(&self) -> Result<String> {
        js_json(self)
    }
}

/// A take's phrase spans, as `<out>.phrases.json` holds them: the script
/// text as shown, and each phrase's place in the audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhrasesFile {
    /// The script as captions show it.
    pub script: String,
    /// Each phrase's text and place in the audio, in order.
    pub phrases: Vec<HeardWord>,
}

impl PhrasesFile {
    /// Writes this file in the skill's own layout.
    pub fn write(&self) -> Result<String> {
        js_json(self)
    }
}

/// Drops a path's last extension, the same rule as `path.replace(/\.[^./]
/// +$/, '')`: the last dot in the final path component, with a non-empty,
/// slash-free tail. A path with no such dot is returned unchanged.
fn without_extension(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    let search_from = text.rfind('/').map_or(0, |i| i + 1);
    match text[search_from..].rfind('.') {
        Some(rel) if search_from + rel + 1 < text.len() => {
            PathBuf::from(&text[..search_from + rel])
        }
        _ => path.to_path_buf(),
    }
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// `audio.replace(/\.[^./]+$/, '') + '.phrases.json'`: the phrase spans a
/// take writes beside its audio.
pub fn phrases_file(audio: &Path) -> PathBuf {
    with_suffix(&without_extension(audio), ".phrases.json")
}

/// `out.replace(/\.[^./]+$/, '') + '.takes.json'`: the recorded takes a
/// run writes beside its audio.
pub fn takes_file(out: &Path) -> PathBuf {
    with_suffix(&without_extension(out), ".takes.json")
}

/// `out.replace(/\.[^./]+$/, '') + '.voice.wav'`: the designed voice a
/// run copies beside its audio, to listen to or to pass back as a
/// reference.
pub fn voice_output_file(out: &Path) -> PathBuf {
    with_suffix(&without_extension(out), ".voice.wav")
}

/// `out.replace(/\.[^./]+$/, '.txt')`: the plain script text a take writes
/// beside its audio.
pub fn script_text_file(out: &Path) -> PathBuf {
    with_suffix(&without_extension(out), ".txt")
}

/// [`script_text_file`], unless the script's own source file already sits
/// at that path, in which case `.spoken.txt` replaces it, so the spoken
/// transcript never overwrites the source script.
///
/// Both paths must already be resolved the same way (the skill compares
/// `resolve(arg) === plain`); this never resolves them itself.
pub fn output_text_file(out: &Path, script_source: Option<&Path>) -> PathBuf {
    let plain = script_text_file(out);
    if script_source.is_some_and(|s| s == plain) {
        plain.with_extension("spoken.txt")
    } else {
        plain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takes_file_serializes_in_insertion_order() -> Result<()> {
        let takes = TakesFile {
            seed: 7.0,
            takes: vec![("b line".to_string(), 12.0), ("a line".to_string(), 3.0)],
        };
        assert_eq!(
            takes.write()?,
            "{\n \"seed\": 7,\n \"takes\": {\n  \"b line\": 12,\n  \"a line\": 3\n }\n}"
        );
        Ok(())
    }

    #[test]
    fn reads_current_format() {
        let read = read_takes_file(r#"{"seed": 9, "takes": {"hi": 3}}"#);
        assert_eq!(read.seed, Some(9.0));
        assert_eq!(read.takes.get("hi"), Some(&3.0));
    }

    #[test]
    fn foreign_file_reads_as_its_own_flat_map() {
        let read = read_takes_file(r#"{"hi": 3, "there": 4}"#);
        assert_eq!(read.seed, None);
        assert_eq!(read.takes.get("hi"), Some(&3.0));
        assert_eq!(read.takes.get("there"), Some(&4.0));
    }

    #[test]
    fn invalid_json_reads_as_no_saved_takes() {
        let read = read_takes_file("not json");
        assert_eq!(read, ReadTakes::default());
    }

    #[test]
    fn seed_resolution_falls_through_the_chain() {
        let none = ReadTakes::default();
        let r = resolve_seed(None, &none, None);
        assert_eq!(
            r,
            ResolvedSeed {
                seed: DEFAULT_SEED,
                reset: false
            }
        );
        let r = resolve_seed(None, &none, Some(3.0));
        assert_eq!(
            r,
            ResolvedSeed {
                seed: 3.0,
                reset: false
            }
        );
        let current = ReadTakes {
            seed: Some(11.0),
            takes: HashMap::new(),
        };
        let r = resolve_seed(None, &current, Some(3.0));
        assert_eq!(
            r,
            ResolvedSeed {
                seed: 11.0,
                reset: false
            }
        );
        let r = resolve_seed(Some(11.0), &current, None);
        assert_eq!(
            r,
            ResolvedSeed {
                seed: 11.0,
                reset: false
            }
        );
        let r = resolve_seed(Some(99.0), &current, None);
        assert_eq!(
            r,
            ResolvedSeed {
                seed: 99.0,
                reset: true
            }
        );
    }

    #[test]
    fn phrases_and_takes_file_names() {
        assert_eq!(
            phrases_file(Path::new("out/voiceover.wav")),
            Path::new("out/voiceover.phrases.json")
        );
        assert_eq!(
            takes_file(Path::new("out/voiceover.wav")),
            Path::new("out/voiceover.takes.json")
        );
        assert_eq!(
            script_text_file(Path::new("out/voiceover.wav")),
            Path::new("out/voiceover.txt")
        );
        assert_eq!(
            voice_output_file(Path::new("out/voiceover.wav")),
            Path::new("out/voiceover.voice.wav")
        );
    }

    #[test]
    fn output_text_file_avoids_overwriting_the_source_script() {
        let out = Path::new("assets/voiceover.wav");
        let plain = Path::new("assets/voiceover.txt");
        assert_eq!(output_text_file(out, None), plain);
        assert_eq!(output_text_file(out, Some(Path::new("other.txt"))), plain);
        assert_eq!(
            output_text_file(out, Some(plain)),
            Path::new("assets/voiceover.spoken.txt")
        );
    }
}
