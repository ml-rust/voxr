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

/// The text of one heard-back row, its notes after line breaks.
fn row_text(c: &Row<'_>, part: usize, engine: &str) -> String {
    let hot = c.heat > HEATED;
    let status = if !c.ok {
        "FAIL"
    } else if !c.doubt.is_empty() || hot || !c.pauses.is_empty() || !c.clicks.is_empty() {
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
            "\n        a {}s pause between \"{}\" and \"{}\", where the script has no break: the voice misread the grammar. Add a comma where the break belongs, or reword",
            js_string(p.gap),
            p.after,
            p.before
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
                "        a 0.5s pause between \"Two\" and \"three\", where the script has no break: the voice misread the grammar. Add a comma where the break belongs, or reword",
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
}
