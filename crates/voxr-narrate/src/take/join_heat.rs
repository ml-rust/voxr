//! A part's hottest sentence: how far each sentence's pitch rises above
//! the voice's usual pitch, questions aside.

use crate::glitch::heat;
use crate::js::{js_round, js_to_fixed};
use crate::pattern::trim;

use super::join::{JOIN_RATE, ShapedTake, subarray};

/// Decimals of a part's heat.
const HEAT_DECIMALS: usize = 2;
/// Characters that close a question after its `?`.
const QUESTION_CLOSERS: [char; 5] = ['"', '\'', '”', '’', ')'];

/// A question rises on purpose: its trimmed text ends in `?`, closing
/// quotes and brackets aside.
fn is_question(text: &str) -> bool {
    trim(text).trim_end_matches(QUESTION_CLOSERS).ends_with('?')
}

/// A part's hottest sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct PartHeat {
    /// The highest [`heat`] of the part's sentences, questions aside, to 2
    /// decimals. 0 when no sentence reads over 0.
    pub heat: f64,
    /// That sentence's script text. `None` when no sentence reads over 0.
    pub hot_text: Option<String>,
}

/// The heat of each sentence of `shaped` against `voice_median`, the
/// voice's usual pitch, and the hottest one kept. A question is left out.
pub fn part_heat(shaped: &ShapedTake, voice_median: f64) -> PartHeat {
    let rate = f64::from(JOIN_RATE);
    let mut top = (0.0f64, None::<&str>);
    for sp in shaped.spans.iter().filter(|sp| !is_question(&sp.text)) {
        let from = js_round(sp.start * rate) as i64;
        let to = js_round(sp.end * rate) as i64;
        let h = heat(subarray(&shaped.audio, from, to), voice_median);
        if h > top.0 {
            top = (h, Some(sp.text.as_str()));
        }
    }
    PartHeat {
        heat: js_to_fixed(top.0, HEAT_DECIMALS),
        hot_text: top.1.map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn questions_are_found_past_closing_quotes() {
        assert!(is_question("Where does it go?"));
        assert!(is_question("\"Where does it go?”) "));
        assert!(!is_question("It goes north."));
    }

    #[test]
    fn the_hottest_sentence_is_kept() {
        // 200 Hz over a 100 Hz voice: heat 2.
        let audio: Vec<f32> = (0..96000)
            .map(|k| (0.3 * (std::f64::consts::TAU * 200.0 * k as f64 / 48000.0).sin()) as f32)
            .collect();
        let span = |text: &str, start: f64, end: f64| crate::check::HeardWord {
            text: text.to_string(),
            start,
            end,
        };
        let shaped = ShapedTake {
            audio,
            spans: vec![span("Is it hot?", 0.0, 1.0), span("It is.", 1.0, 2.0)],
        };
        let got = part_heat(&shaped, 100.0);
        assert_eq!(got.heat, 2.0);
        assert_eq!(got.hot_text.as_deref(), Some("It is."));
    }
}
