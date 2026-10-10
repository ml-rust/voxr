//! The lines the speak command prints, word for word as the skill's
//! `speakCmd` prints them: the script review, the VoxCPM2 speed notice, the
//! voice polish, the pace, a timed script's placement, the output files,
//! each part as it was heard back, and the verdict.

use super::speak_rows::Doubtful;
use crate::js::{js_string, js_to_fixed_text};
use crate::polish::PolishReport;
use crate::take::{PaceComparison, PaceVerdict};
use crate::text::ReviewNote;

/// The notice VoxCPM2 runs print when a speed is asked for: VoxCPM2 is
/// never time-stretched.
pub const VOXCPM_SPEED_NOTICE: &str = "speak --speed: VoxCPM2 is never time-stretched, so --speed and \"tts.speed\" are ignored for it. Its pace comes from the reference: a recording at the pace wanted, or a new --seed for another designed voice.";

/// The verdict when every part was heard back as written and none is
/// marked `ok ?`.
pub const ALL_HEARD: &str = "Speech check: every phrase was heard back as written.";

/// The heading of the parts heard back.
pub const HEARD_BACK: &str = "Heard back, phrase by phrase:";

/// The script review: a heading, one line per note, and a blank line.
/// Empty when there is no note and `review` (`--review`) is off.
pub fn review_lines(notes: &[ReviewNote], review: bool) -> Vec<String> {
    if notes.is_empty() && !review {
        return Vec::new();
    }
    let mut out = vec![if notes.is_empty() {
        "Script review: nothing likely to be misread.".to_string()
    } else {
        format!(
            "Script review, {} note(s) (narration.md, Prepare the script for speech):",
            notes.len()
        )
    }];
    out.extend(
        notes
            .iter()
            .map(|r| format!("  line {}  \"{}\": {}", r.line, r.text, r.note)),
    );
    out.push(String::new());
    out
}

/// What the voice polish did, as one line.
pub fn polish_line(s: &PolishReport) -> String {
    let gain = if s.leveled > 0 {
        format!(" (up to {} dB)", js_to_fixed_text(s.max_gain_db, 1))
    } else {
        String::new()
    };
    let ess = if s.ess_share != 0.0 && !s.ess_share.is_nan() {
        format!(" (up to {} dB)", js_to_fixed_text(-s.ess_deepest_db, 1))
    } else {
        String::new()
    };
    format!(
        "Voice polish: loudness evened in {} sentence(s){gain}, compressed gently, harsh \"s\" turned down in {}% of the audio{ess}, normalized to a -1 dB peak. --raw skips this.",
        s.leveled,
        js_to_fixed_text(s.ess_share * 100.0, 1)
    )
}

/// The engine label of a VoxCPM2 run: `VoxCPM2`, then its voice.
pub(super) fn is_voxcpm(engine: &str) -> bool {
    engine.starts_with("VoxCPM2")
}

/// The pace line: the measured pace against its range, with the fix when
/// it falls outside. `None` when no pace was measured (a rate of 0).
///
/// Outside the range, VoxCPM2 is told to change its reference, and every
/// other engine is given the `--speed` that reaches the target.
pub fn pace_line(v: &PaceVerdict, engine: &str) -> Option<String> {
    if v.rate == 0.0 || v.rate.is_nan() {
        return None;
    }
    let fix = if is_voxcpm(engine) {
        "VoxCPM2 takes its pace from the reference, so change the reference: a recording at the pace wanted, or a new --seed for another designed voice. Pace words in --style or a note skew the voice.".to_string()
    } else {
        format!(
            "--speed {} reaches {}.",
            js_string(v.suggested_speed),
            js_string(v.target)
        )
    };
    let (lo, hi) = (js_string(v.low), js_string(v.high));
    let verdict = match v.comparison() {
        PaceComparison::Faster => format!(": faster than the {lo} to {hi} range. {fix}"),
        PaceComparison::Slower => format!(": slower than the {lo} to {hi} range. {fix}"),
        PaceComparison::InRange => {
            format!(
                ": in the {lo} to {hi} range around {}.",
                js_string(v.target)
            )
        }
    };
    Some(format!(
        "Pace: {} syllables per second while speaking{verdict}",
        js_to_fixed_text(v.rate, 1)
    ))
}

/// A timed script's placement: the count of lines, then each line that
/// starts late or overruns its slot, indented.
pub fn timed_lines(cues: usize, timing: &[String]) -> Vec<String> {
    let mut out = vec![format!("Timed script: {cues} lines placed at their times.")];
    out.extend(timing.iter().map(|t| format!("  {t}")));
    out
}

/// The output files and where they go in the manifest. `out` and `txt`
/// are relative to the working directory.
pub fn output_lines(out: &str, duration: f64, engine: &str, txt: &str) -> Vec<String> {
    vec![
        format!(
            "{out} ({}s, {engine}) and its script {txt}",
            js_to_fixed_text(duration, 2)
        ),
        format!(
            "Next, in video.json: \"voiceover\": {{ \"file\": \"{out}\", \"script\": \"{txt}\" }} for narration across the whole video,"
        ),
        format!(
            "or \"audio\": \"{out}\" and \"script\": \"{txt}\" on the one scene it narrates. Then run transcribe."
        ),
    ]
}

/// `n` and `one`, with an `s` unless `n` is 1.
fn count(n: usize, one: &str) -> String {
    format!("{n} {one}{}", if n == 1 { "" } else { "s" })
}

/// The parts marked `ok ?`: how many, their numbers, and their notes
/// summed by kind, zero kinds left out. `None` when no part is marked.
fn marked_text(d: &Doubtful) -> Option<String> {
    if d.parts.is_empty() {
        return None;
    }
    let kinds: Vec<String> = [
        (d.words, "doubtful word"),
        (d.pauses, "stray pause"),
        (d.peaks, "pitch peak"),
        (d.clicks, "click"),
    ]
    .into_iter()
    .filter(|&(n, _)| n > 0)
    .map(|(n, one)| count(n, one))
    .collect();
    let numbers: Vec<String> = d.parts.iter().map(usize::to_string).collect();
    let single = d.parts.len() == 1;
    Some(format!(
        "{} of {} part{} {} marked ok ?: {} ({}). Listen to {} before using the take: each note above gives its fix.",
        d.parts.len(),
        d.total,
        if d.total == 1 { "" } else { "s" },
        if single { "is" } else { "are" },
        numbers.join(", "),
        kinds.join(", "),
        if single { "it" } else { "them" }
    ))
}

/// The verdict and its exit code.
///
/// - No problem and no part marked `ok ?`: [`ALL_HEARD`], exit 0.
/// - No problem, parts marked `ok ?`: one line naming them, exit 0.
/// - Problems: every problem the speech check found, then the `ok ?` line
///   when a part is marked, exit 1.
pub fn verdict_lines(problems: &[String], doubtful: &Doubtful) -> (Vec<String>, i32) {
    let marked = marked_text(doubtful);
    if problems.is_empty() {
        let line = match marked {
            Some(m) => format!("Speech check: no problem remains, but {m}"),
            None => ALL_HEARD.to_string(),
        };
        return (vec![line], 0);
    }
    let mut out = vec![
        String::new(),
        format!(
            "The speech check found {} problem(s), heard back with Whisper:",
            problems.len()
        ),
    ];
    out.extend(problems.iter().map(|p| format!("  {p}")));
    out.extend(marked.map(|m| format!("Speech check: {m}")));
    (out, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take::pace_verdict;

    #[test]
    fn review_prints_only_with_notes_or_when_asked() {
        assert!(review_lines(&[], false).is_empty());
        assert_eq!(
            review_lines(&[], true),
            ["Script review: nothing likely to be misread.", ""]
        );
        let note = ReviewNote {
            line: 2,
            text: "v2.1".to_string(),
            note: "letters and digits together".to_string(),
        };
        assert_eq!(
            review_lines(std::slice::from_ref(&note), false),
            [
                "Script review, 1 note(s) (narration.md, Prepare the script for speech):",
                "  line 2  \"v2.1\": letters and digits together",
                ""
            ]
        );
    }

    #[test]
    fn polish_reports_what_it_did() {
        let none = PolishReport {
            leveled: 0,
            max_gain_db: 0.0,
            ess_share: 0.0,
            ess_deepest_db: 0.0,
        };
        assert_eq!(
            polish_line(&none),
            "Voice polish: loudness evened in 0 sentence(s), compressed gently, harsh \"s\" turned down in 0.0% of the audio, normalized to a -1 dB peak. --raw skips this."
        );
        let some = PolishReport {
            leveled: 2,
            max_gain_db: 3.25,
            ess_share: 0.0125,
            ess_deepest_db: -4.04,
        };
        assert_eq!(
            polish_line(&some),
            "Voice polish: loudness evened in 2 sentence(s) (up to 3.3 dB), compressed gently, harsh \"s\" turned down in 1.3% of the audio (up to 4.0 dB), normalized to a -1 dB peak. --raw skips this."
        );
    }

    #[test]
    fn pace_names_the_fix_per_engine() {
        let fast = pace_verdict(4.84, 4.1, 1.0);
        assert_eq!(
            pace_line(&fast, "Kokoro (af_heart)").as_deref(),
            Some(
                "Pace: 4.8 syllables per second while speaking: faster than the 3.5 to 4.7 range. --speed 0.85 reaches 4.1."
            )
        );
        assert!(
            pace_line(&fast, "VoxCPM2")
                .is_some_and(|l| l.ends_with("Pace words in --style or a note skew the voice."))
        );
        let fine = pace_verdict(4.0, 4.1, 1.0);
        assert_eq!(
            pace_line(&fine, "x").as_deref(),
            Some(
                "Pace: 4.0 syllables per second while speaking: in the 3.5 to 4.7 range around 4.1."
            )
        );
        assert_eq!(pace_line(&pace_verdict(0.0, 4.1, 1.0), "x"), None);
    }

    fn none_marked(total: usize) -> Doubtful {
        Doubtful {
            total,
            ..Doubtful::default()
        }
    }

    #[test]
    fn a_clean_run_exits_0_with_every_phrase_heard() {
        assert_eq!(
            verdict_lines(&[], &none_marked(5)),
            (vec![ALL_HEARD.to_string()], 0)
        );
    }

    #[test]
    fn doubtful_parts_exit_0_and_are_counted() {
        let two = Doubtful {
            total: 5,
            parts: vec![2, 4],
            words: 0,
            pauses: 1,
            peaks: 1,
            clicks: 0,
        };
        assert_eq!(
            verdict_lines(&[], &two),
            (
                vec![
                    "Speech check: no problem remains, but 2 of 5 parts are marked ok ?: 2, 4 (1 stray pause, 1 pitch peak). Listen to them before using the take: each note above gives its fix."
                        .to_string()
                ],
                0
            )
        );
        let one = Doubtful {
            total: 5,
            parts: vec![3],
            words: 2,
            pauses: 2,
            peaks: 0,
            clicks: 1,
        };
        assert_eq!(
            verdict_lines(&[], &one).0,
            [
                "Speech check: no problem remains, but 1 of 5 parts is marked ok ?: 3 (2 doubtful words, 2 stray pauses, 1 click). Listen to it before using the take: each note above gives its fix."
            ]
        );
    }

    #[test]
    fn problems_exit_1_and_name_the_doubtful_parts_after() {
        let (lines, code) = verdict_lines(&["part 1: missing".to_string()], &none_marked(1));
        assert_eq!(code, 1);
        assert_eq!(
            lines,
            [
                "",
                "The speech check found 1 problem(s), heard back with Whisper:",
                "  part 1: missing"
            ]
        );
        let marked = Doubtful {
            total: 2,
            parts: vec![2],
            words: 1,
            pauses: 0,
            peaks: 0,
            clicks: 0,
        };
        let (lines, code) = verdict_lines(&["part 1: missing".to_string()], &marked);
        assert_eq!(code, 1);
        assert_eq!(
            lines,
            [
                "",
                "The speech check found 1 problem(s), heard back with Whisper:",
                "  part 1: missing",
                "Speech check: 1 of 2 parts is marked ok ?: 2 (1 doubtful word). Listen to it before using the take: each note above gives its fix."
            ]
        );
    }
}
