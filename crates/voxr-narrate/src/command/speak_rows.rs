//! The heard-back rows of the speak command: each part as the speech check
//! heard it, with its notes on indented lines.

use super::speak_lines::{HEARD_BACK, is_voxcpm};
use crate::check::StrayPause;
use crate::glitch::HEATED;
use crate::js::{js_string, js_to_fixed_text};
use crate::pattern::utf16_prefix;
use crate::take::ScriptChecks;

/// One part as the heard-back rows read it.
struct Row<'a> {
    at: f64,
    ok: bool,
    heard: &'a str,
    why: &'a str,
    doubt: &'a [String],
    heat: f64,
    hot_text: Option<&'a str>,
    pauses: &'a [StrayPause],
    clicks: &'a [f64],
}

fn rows(checks: &ScriptChecks) -> Vec<Row<'_>> {
    match checks {
        ScriptChecks::Parts(list) => list
            .iter()
            .map(|c| Row {
                at: c.at,
                ok: c.ok,
                heard: &c.heard,
                why: &c.why,
                doubt: &c.doubt,
                heat: c.heat,
                hot_text: c.hot_text.as_deref(),
                pauses: &c.pauses,
                clicks: &c.clicks,
            })
            .collect(),
        ScriptChecks::Whole(list) => list
            .iter()
            .map(|c| Row {
                at: c.at,
                ok: c.ok,
                heard: &c.heard,
                why: &c.why,
                doubt: &c.doubt,
                heat: f64::NAN,
                hot_text: None,
                pauses: &c.pauses,
                clicks: &[],
            })
            .collect(),
    }
}

/// A part that passed with something to listen to: a doubtful word, a
/// pitch peak, a stray pause or a click. Its row is marked `ok ?`.
fn doubtful(c: &Row<'_>) -> bool {
    c.ok && (!c.doubt.is_empty() || c.heat > HEATED || !c.pauses.is_empty() || !c.clicks.is_empty())
}

/// The parts marked `ok ?`, and what their notes hold, summed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Doubtful {
    /// Parts in the run, marked or not.
    pub total: usize,
    /// Each marked part's number, from 1.
    pub parts: Vec<usize>,
    /// Words both recognizers missed or misheard.
    pub words: usize,
    /// Pauses the script does not ask for.
    pub pauses: usize,
    /// Parts whose pitch peaks over [`HEATED`].
    pub peaks: usize,
    /// Clicks the repair left.
    pub clicks: usize,
}

/// The parts of `checks` marked `ok ?`: an `ok ?` part exits 0 and needs a
/// listen.
pub fn doubtful_parts(checks: &ScriptChecks) -> Doubtful {
    let all = rows(checks);
    let mut d = Doubtful {
        total: all.len(),
        ..Doubtful::default()
    };
    for (part, c) in all.iter().enumerate().filter(|(_, c)| doubtful(c)) {
        d.parts.push(part + 1);
        d.words += c.doubt.len();
        d.pauses += c.pauses.len();
        d.peaks += usize::from(c.heat > HEATED);
        d.clicks += c.clicks.len();
    }
    d
}

/// The fix a stray pause note gives. VoxCPM2 draws a whole part again on
/// any edit to its text, so a reworded part can pause somewhere else.
fn pause_fix(part: usize, engine: &str) -> String {
    if is_voxcpm(engine) {
        format!(
            "Listen to it, and --reroll {} if it is heard: an edit to the text draws the whole part again, and VoxCPM2 can pause somewhere else.",
            part + 1
        )
    } else {
        "Add a comma where the break belongs, or reword".to_string()
    }
}

/// The text of one heard-back row, its notes after line breaks.
fn row_text(c: &Row<'_>, part: usize, engine: &str) -> String {
    let hot = c.heat > HEATED;
    let status = if !c.ok {
        "FAIL"
    } else if doubtful(c) {
        "ok ?"
    } else {
        "ok  "
    };
    let mut s = format!("  {status} {:>6}s  {}", js_to_fixed_text(c.at, 2), c.heard);
    if !c.ok {
        s.push_str(&format!("\n        {}", c.why));
    }
    if hot {
        let quote = match c.hot_text.filter(|t| !t.is_empty()) {
            Some(t) => format!("\"{}\": ", utf16_prefix(t, 60)),
            None => String::new(),
        };
        s.push_str(&format!(
            "\n        {quote}its pitch peaks at {}x the voice's usual pitch: it can sound shouted or excited. Listen, then calm its delivery note or --reroll {}",
            js_string(c.heat),
            part + 1
        ));
    }
    for p in c.pauses {
        s.push_str(&format!(
            "\n        a {}s pause between \"{}\" and \"{}\", where the script has no break: the voice misread the grammar. {}",
            js_string(p.gap),
            p.after,
            p.before,
            pause_fix(part, engine)
        ));
    }
    if !c.clicks.is_empty() {
        let at: Vec<String> = c
            .clicks
            .iter()
            .map(|v| format!("{}s", js_string(*v)))
            .collect();
        s.push_str(&format!(
            "\n        a click the repair could not smooth, {} into this part: listen to it, and --reroll {} if it is heard",
            at.join(", "),
            part + 1
        ));
    }
    if !c.doubt.is_empty() {
        let words: Vec<String> = c.doubt.iter().map(|w| format!("\"{w}\"")).collect();
        let fix = if is_voxcpm(engine) {
            format!("--reroll {}", part + 1)
        } else {
            "reword the phrase".to_string()
        };
        s.push_str(&format!(
            "\n        both recognizers missed or misheard {}: listen to it, and {fix} if it is said wrong",
            words.join(", ")
        ));
    }
    s
}

/// Every part as it was heard back, so a wrong word shows even when the
/// check passes it: a blank line, the heading, then one row per part with
/// its notes on indented lines.
pub fn heard_back_lines(checks: &ScriptChecks, engine: &str) -> Vec<String> {
    let mut out = vec![String::new(), HEARD_BACK.to_string()];
    for (part, row) in rows(checks).iter().enumerate() {
        out.extend(row_text(row, part, engine).split('\n').map(str::to_string));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take::{PartCheck, WholeCheck};

    #[test]
    fn heard_back_rows_carry_their_notes() {
        let checks = ScriptChecks::Parts(vec![
            PartCheck {
                at: 0.0,
                text: "One.".to_string(),
                ok: true,
                heard: "One.".to_string(),
                why: String::new(),
                seed: 7.0,
                doubt: Vec::new(),
                heat: 1.2,
                hot_text: None,
                pauses: Vec::new(),
                clicks: Vec::new(),
            },
            PartCheck {
                at: 12.5,
                text: "Two three.".to_string(),
                ok: false,
                heard: "Two tree.".to_string(),
                why: "misheard: three".to_string(),
                seed: 8.0,
                doubt: vec!["three".to_string()],
                heat: 1.75,
                hot_text: Some("Two three.".to_string()),
                pauses: vec![StrayPause {
                    after: "Two".to_string(),
                    before: "three".to_string(),
                    gap: 0.5,
                }],
                clicks: vec![0.25],
            },
        ]);
        let got = heard_back_lines(&checks, "VoxCPM2");
        assert_eq!(
            got,
            [
                "",
                "Heard back, phrase by phrase:",
                "  ok     0.00s  One.",
                "  FAIL  12.50s  Two tree.",
                "        misheard: three",
                "        \"Two three.\": its pitch peaks at 1.75x the voice's usual pitch: it can sound shouted or excited. Listen, then calm its delivery note or --reroll 2",
                "        a 0.5s pause between \"Two\" and \"three\", where the script has no break: the voice misread the grammar. Listen to it, and --reroll 2 if it is heard: an edit to the text draws the whole part again, and VoxCPM2 can pause somewhere else.",
                "        a click the repair could not smooth, 0.25s into this part: listen to it, and --reroll 2 if it is heard",
                "        both recognizers missed or misheard \"three\": listen to it, and --reroll 2 if it is said wrong",
            ]
        );
        let whole = ScriptChecks::Whole(vec![WholeCheck {
            at: 0.0,
            text: "One.".to_string(),
            ok: true,
            heard: "One.".to_string(),
            why: String::new(),
            doubt: vec!["One".to_string()],
            pauses: Vec::new(),
        }]);
        assert_eq!(
            heard_back_lines(&whole, "Kokoro (af_heart)")[2..],
            [
                "  ok ?   0.00s  One.",
                "        both recognizers missed or misheard \"One\": listen to it, and reword the phrase if it is said wrong"
            ]
        );
    }

    fn pause(after: &str, before: &str) -> StrayPause {
        StrayPause {
            after: after.to_string(),
            before: before.to_string(),
            gap: 0.6,
        }
    }

    #[test]
    fn a_stray_pause_is_fixed_by_rewording_outside_voxcpm() {
        let whole = ScriptChecks::Whole(vec![WholeCheck {
            at: 0.0,
            text: "One two.".to_string(),
            ok: true,
            heard: "One two.".to_string(),
            why: String::new(),
            doubt: Vec::new(),
            pauses: vec![pause("One", "two")],
        }]);
        assert_eq!(
            heard_back_lines(&whole, "Kokoro (af_heart)")[2..],
            [
                "  ok ?   0.00s  One two.",
                "        a 0.6s pause between \"One\" and \"two\", where the script has no break: the voice misread the grammar. Add a comma where the break belongs, or reword"
            ]
        );
    }

    #[test]
    fn doubtful_parts_sum_the_notes_of_passing_parts_only() {
        let part =
            |ok: bool, doubt: &[&str], heat: f64, pauses: Vec<StrayPause>, clicks: Vec<f64>| {
                PartCheck {
                    at: 0.0,
                    text: "One two.".to_string(),
                    ok,
                    heard: "One two.".to_string(),
                    why: String::new(),
                    seed: 7.0,
                    doubt: doubt.iter().map(|w| w.to_string()).collect(),
                    heat,
                    hot_text: None,
                    pauses,
                    clicks,
                }
            };
        let checks = ScriptChecks::Parts(vec![
            part(true, &[], 1.0, Vec::new(), Vec::new()),
            part(true, &["two"], 1.75, vec![pause("One", "two")], Vec::new()),
            part(false, &["one"], 1.75, vec![pause("One", "two")], vec![0.2]),
            part(
                true,
                &[],
                f64::NAN,
                vec![pause("a", "b"), pause("c", "d")],
                vec![0.1],
            ),
        ]);
        assert_eq!(
            doubtful_parts(&checks),
            Doubtful {
                total: 4,
                parts: vec![2, 4],
                words: 1,
                pauses: 3,
                peaks: 1,
                clicks: 1,
            }
        );
    }
}
