//! Output parity between the speaking-pace port in `voxr_narrate` and the
//! JavaScript it ports.
//!
//! The fixture (`tests/fixtures/narration_pace_oracle.json`) was generated
//! by `tests/tools/narration_oracle/pace_oracle.mjs` over the cases in
//! `pace_cases.mjs` beside it, which `common::pace_cases` lists in the same
//! order and `common::signals` builds bit for bit. A sample output is
//! `String(x)`, JavaScript's own text of the double: parsed back with
//! [`num`] and compared as `{:?}`, the two sides read the same exact double
//! or differ.

use std::path::Path;

use serde::Deserialize;
use voxr_narrate::{PaceSpan, REF_PACE, pace, pace_verdict, sounding, syllables};

mod common;
use common::pace_cases::{
    PhraseCase, pace_no_audio_cases, sounding_cases, text_cases, verdict_cases,
};
use common::signals;
use common::{assert_oracle_current, check, load_json, num};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/narration_pace_oracle.json"
);
const ORACLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/tools/narration_oracle/pace_oracle.mjs"
);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Fixture {
    ref_pace: String,
    texts: Vec<TextOut>,
    sounding: Vec<SoundingOut>,
    pace_no_audio: Vec<PaceNoAudioOut>,
    verdict: Vec<VerdictOut>,
}

#[derive(Debug, Deserialize)]
struct TextOut {
    name: String,
    text: String,
    syllables: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SoundingOut {
    name: String,
    rate: u32,
    n: usize,
    input: String,
    spans: Vec<SpanOut>,
    #[serde(default)]
    pace_value: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpanOut {
    start: f64,
    end: f64,
    value: String,
}

#[derive(Debug, Deserialize)]
struct PaceNoAudioOut {
    name: String,
    value: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VerdictOut {
    name: String,
    measured_pace: f64,
    target: f64,
    speed: f64,
    rate: String,
    low: String,
    high: String,
    suggested_speed: String,
}

/// A number as `{:?}`: the exact double, so a value read back from the
/// fixture's `String(x)` text compares against the ported result exactly.
fn exact(v: f64) -> String {
    format!("{v:?}")
}

/// `exact` on the fixture's `String(x)` text, read back to the double it
/// names.
fn exact_text(text: &str) -> String {
    exact(num(text))
}

fn spans(phrases: &[PhraseCase]) -> Vec<PaceSpan> {
    phrases
        .iter()
        .map(|p| PaceSpan {
            text: p.text.to_string(),
            start: p.start,
            end: p.end,
        })
        .collect()
}

#[test]
fn ref_pace_matches_javascript() {
    let fixture: Fixture = load_json(Path::new(FIXTURE));
    assert_eq!(num(&fixture.ref_pace), REF_PACE);
}

#[test]
fn syllables_matches_javascript_on_every_text() {
    let fixture: Fixture = load_json(Path::new(FIXTURE));
    let cases = text_cases();
    assert_eq!(
        cases.len(),
        fixture.texts.len(),
        "common::pace_cases::text_cases and TEXT_CASES list a different number of cases"
    );
    let mut diffs = Vec::new();
    for (case, want) in cases.iter().zip(&fixture.texts) {
        assert_eq!(case.name, want.name, "case order differs");
        assert_eq!(
            case.text, want.text,
            "{}: text differs from the fixture",
            case.name
        );
        check(
            &mut diffs,
            case.name,
            syllables(case.text).to_string(),
            want.syllables.to_string(),
        );
    }
    assert!(
        diffs.is_empty(),
        "{} differences:\n{}",
        diffs.len(),
        diffs.join("\n")
    );
}

#[test]
fn sounding_and_pace_match_javascript_on_every_signal() {
    let fixture: Fixture = load_json(Path::new(FIXTURE));
    let cases = sounding_cases();
    let ours: Vec<(&str, u32, usize)> = cases.iter().map(|c| (c.name, c.rate, c.n)).collect();
    let theirs: Vec<(&str, u32, usize)> = fixture
        .sounding
        .iter()
        .map(|c| (c.name.as_str(), c.rate, c.n))
        .collect();
    assert_eq!(
        ours, theirs,
        "common::pace_cases::sounding_cases and SOUNDING_CASES list different cases"
    );
    let mut diffs = Vec::new();
    for (case, want) in cases.iter().zip(&fixture.sounding) {
        assert_eq!(case.name, want.name, "case order differs");
        let x = signals::render(case.n, &case.ops);
        let input = signals::fnv1a(&x);
        if input != want.input {
            diffs.push(format!(
                "{}: input signal differs from the oracle's: hash {input}, want {}",
                case.name, want.input
            ));
            continue;
        }
        assert_eq!(
            case.spans.len(),
            want.spans.len(),
            "{}: spans listed on one side only",
            case.name
        );
        for (&(start, end), span) in case.spans.iter().zip(&want.spans) {
            assert_eq!(
                (start, end),
                (span.start, span.end),
                "{}: span differs",
                case.name
            );
            check(
                &mut diffs,
                &format!("{} sounding [{start},{end}]", case.name),
                exact(sounding(&x, start, end, case.rate)),
                exact_text(&span.value),
            );
        }
        if let Some(want_pace) = &want.pace_value {
            let phrase_spans = spans(&case.phrases);
            check(
                &mut diffs,
                &format!("{} pace", case.name),
                exact(pace(&phrase_spans, Some((&x, case.rate)))),
                exact_text(want_pace),
            );
        }
    }
    assert!(
        diffs.is_empty(),
        "{} differences:\n{}",
        diffs.len(),
        diffs.join("\n")
    );
}

#[test]
fn pace_without_audio_matches_javascript() {
    let fixture: Fixture = load_json(Path::new(FIXTURE));
    let cases = pace_no_audio_cases();
    assert_eq!(
        cases.len(),
        fixture.pace_no_audio.len(),
        "common::pace_cases::pace_no_audio_cases and PACE_NO_AUDIO_CASES list a different number of cases"
    );
    let mut diffs = Vec::new();
    for (case, want) in cases.iter().zip(&fixture.pace_no_audio) {
        assert_eq!(case.name, want.name, "case order differs");
        let phrase_spans = spans(&case.phrases);
        check(
            &mut diffs,
            case.name,
            exact(pace(&phrase_spans, None)),
            exact_text(&want.value),
        );
    }
    assert!(
        diffs.is_empty(),
        "{} differences:\n{}",
        diffs.len(),
        diffs.join("\n")
    );
}

#[test]
fn pace_verdict_matches_javascript() {
    let fixture: Fixture = load_json(Path::new(FIXTURE));
    let cases = verdict_cases();
    assert_eq!(
        cases.len(),
        fixture.verdict.len(),
        "common::pace_cases::verdict_cases and VERDICT_CASES list a different number of cases"
    );
    let mut diffs = Vec::new();
    for (case, want) in cases.iter().zip(&fixture.verdict) {
        assert_eq!(case.name, want.name, "case order differs");
        assert_eq!(
            case.measured_pace, want.measured_pace,
            "{}: measuredPace differs",
            case.name
        );
        assert_eq!(case.target, want.target, "{}: target differs", case.name);
        assert_eq!(case.speed, want.speed, "{}: speed differs", case.name);
        let v = pace_verdict(case.measured_pace, case.target, case.speed);
        check(
            &mut diffs,
            &format!("{}: rate", case.name),
            exact(v.rate),
            exact_text(&want.rate),
        );
        check(
            &mut diffs,
            &format!("{}: low", case.name),
            exact(v.low),
            exact_text(&want.low),
        );
        check(
            &mut diffs,
            &format!("{}: high", case.name),
            exact(v.high),
            exact_text(&want.high),
        );
        check(
            &mut diffs,
            &format!("{}: suggestedSpeed", case.name),
            exact(v.suggested_speed),
            exact_text(&want.suggested_speed),
        );
    }
    assert!(
        diffs.is_empty(),
        "{} differences:\n{}",
        diffs.len(),
        diffs.join("\n")
    );
}

#[test]
fn pace_oracle_fixture_is_current() {
    assert_oracle_current(ORACLE, FIXTURE);
}
