//! The transcript oracle's fixture as `tests/tools/narration_oracle/
//! transcript_oracle.mjs` writes it: each case's inputs and the JavaScript
//! outputs, times as `String(x)`.

use std::collections::BTreeMap;

use serde::Deserialize;
use voxr_narrate::{Cue, HeardWord};

use super::signals::{self, Op};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Fixture {
    pub constants: Constants,
    pub transcribe: Vec<TranscribeCase>,
    pub signals: Vec<SignalCase>,
    pub even_out: Vec<EvenCase>,
    pub collapse: Vec<CollapseCase>,
    pub align_phrases: Vec<AlignCase>,
    pub find_phrase: FindCases,
    pub words_file: Vec<WordsFileCase>,
    pub spoken: SpokenCase,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Constants {
    pub audio_lead: String,
    pub audio_tail: String,
}

/// A `signals.mjs` op as the fixture writes it.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum OpIn {
    Tone {
        hz: String,
        start: usize,
        len: usize,
        amp: f64,
        attack: usize,
        release: usize,
    },
    Noise {
        start: usize,
        len: usize,
        amp: f64,
        seed: u32,
    },
}

/// Every tone name `signals.mjs` knows.
const TONE_NAMES: [&str; 15] = [
    "60", "120", "125", "130", "135", "150", "180", "200", "220", "240", "260", "300", "1500",
    "70_44", "200_44",
];

/// The signal `ops` build over `n` samples, as `signals.mjs` renders it.
pub fn render(n: usize, ops: &[OpIn]) -> Vec<f32> {
    let ops: Vec<Op> = ops
        .iter()
        .map(|op| match op {
            OpIn::Tone {
                hz,
                start,
                len,
                amp,
                attack,
                release,
            } => {
                let name = TONE_NAMES
                    .into_iter()
                    .find(|t| t == hz)
                    .unwrap_or_else(|| panic!("no tone {hz}"));
                signals::tone(name, *start, *len, *amp, *attack, *release)
            }
            OpIn::Noise {
                start,
                len,
                amp,
                seed,
            } => signals::noise(*start, *len, *amp, *seed),
        })
        .collect();
    signals::render(n, &ops)
}

/// A word the JavaScript returned, times as `String(x)`.
#[derive(Debug, Deserialize)]
pub struct OutWord {
    pub text: String,
    pub start: String,
    pub end: String,
}

/// A phrase span as the fixture writes it.
#[derive(Debug, Deserialize)]
pub struct CueIn {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

pub fn cues(list: &[CueIn]) -> Vec<Cue> {
    list.iter()
        .map(|c| Cue {
            start: c.start,
            end: c.end,
            text: c.text.clone(),
        })
        .collect()
}

/// A take read with scripted recognizer words, one list per piece. A time
/// the recognizer did not place is `None`.
#[derive(Debug, Deserialize)]
pub struct TranscribeCase {
    pub name: String,
    pub n: usize,
    pub ops: Vec<OpIn>,
    pub language: Option<String>,
    pub script: Vec<Vec<(String, f64, Option<f64>)>>,
    pub file: String,
    pub input: String,
    pub floor: String,
    pub pieces: Vec<(usize, usize)>,
    pub calls: Vec<Call>,
    pub words: Vec<OutWord>,
    pub looped: bool,
    pub duration: String,
    pub json: String,
    pub compose: Option<Compose>,
}

/// One recognizer call: the piece's length, the language named, and the
/// piece's hash.
#[derive(Debug, Deserialize)]
pub struct Call {
    pub n: usize,
    pub language: String,
    pub input: String,
}

/// The transcribe command's script step: phrase spans when given, else the
/// plain script.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Compose {
    pub script: Option<String>,
    pub script_file: String,
    pub phrases: Option<Vec<CueIn>>,
    pub phrases_file: Option<String>,
    pub words: Vec<OutWord>,
    pub json: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignalCase {
    pub name: String,
    pub n: usize,
    pub ops: Vec<OpIn>,
    pub input: String,
    pub floor: String,
    pub snap_onsets: Vec<WordsCase>,
    pub snap_ends: Vec<WordsCase>,
    pub fit_to_sound: Vec<WordsCase>,
    pub last_run: Vec<LastRunCase>,
    pub fit_file: Vec<FitFileCase>,
}

#[derive(Debug, Deserialize)]
pub struct WordsCase {
    pub words: Vec<HeardWord>,
    pub out: Vec<OutWord>,
}

#[derive(Debug, Deserialize)]
pub struct LastRunCase {
    pub from: f64,
    pub to: f64,
    pub out: Option<(String, String)>,
}

#[derive(Debug, Deserialize)]
pub struct FitFileCase {
    pub words: Vec<HeardWord>,
    pub phrase: Option<Vec<usize>>,
    pub spans: Option<Vec<CueIn>>,
    pub out: Vec<OutWord>,
}

/// `limit` is `None` for no limit.
#[derive(Debug, Deserialize)]
pub struct EvenCase {
    pub words: Vec<HeardWord>,
    pub limit: Option<f64>,
    pub out: Vec<OutWord>,
}

#[derive(Debug, Deserialize)]
pub struct CollapseCase {
    pub words: Vec<HeardWord>,
    pub out: Vec<OutWord>,
    pub found: bool,
}

#[derive(Debug, Deserialize)]
pub struct AlignCase {
    pub words: Vec<HeardWord>,
    pub phrases: Vec<CueIn>,
    pub out: Vec<OutWord>,
    pub phrase: Vec<usize>,
}

/// Phrases looked for in one list of spoken words.
#[derive(Debug, Deserialize)]
pub struct FindCases {
    pub words: Vec<HeardWord>,
    pub cases: Vec<FindCase>,
}

#[derive(Debug, Deserialize)]
pub struct FindCase {
    pub phrase: String,
    pub near: f64,
    pub out: Option<FindOut>,
}

#[derive(Debug, Deserialize)]
pub struct FindOut {
    pub start: String,
    pub end: String,
    pub score: String,
}

#[derive(Debug, Deserialize)]
pub struct WordsFileCase {
    pub file: String,
    pub out: String,
}

/// A manifest, the words files beside it by name, and what the JavaScript
/// read from them.
#[derive(Debug, Deserialize)]
pub struct SpokenCase {
    pub manifest: ManifestIn,
    pub files: BTreeMap<String, String>,
    pub narrations: Vec<NarrationOut>,
    pub words: Vec<OutWord>,
    pub missing: Vec<NarrationOut>,
    pub timeline: String,
}

#[derive(Debug, Deserialize)]
pub struct ManifestIn {
    pub voiceover: VoiceoverIn,
    pub scenes: Vec<SceneIn>,
}

#[derive(Debug, Deserialize)]
pub struct VoiceoverIn {
    pub file: String,
    pub at: f64,
}

#[derive(Debug, Deserialize)]
pub struct SceneIn {
    pub name: String,
    pub start: f64,
    pub audio: Option<String>,
    pub script: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct NarrationOut {
    pub file: String,
    pub at: String,
    #[serde(rename = "where")]
    pub label: String,
    pub script: Option<String>,
}
