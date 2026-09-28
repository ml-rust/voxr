//! Crushed words given back their time: recognizer times can overlap, one
//! word swallowing the time of the next few, which are left with none.

use crate::check::HeardWord;
use crate::check::speech_match::letters;
use crate::error::Result;
use crate::js::{js_max, js_min};
use crate::pattern::Patterns;

/// A word shorter than this, in seconds, is crushed.
const CRUSHED: f64 = 0.08;
/// A word is swollen when it lasts longer than this plus
/// [`SLOW_LETTER`] per letter.
const SWOLLEN_BASE: f64 = 0.25;
/// Seconds per letter at a slow pace.
const SLOW_LETTER: f64 = 0.12;
/// Seconds per letter a run needs at the least.
const MIN_LETTER: f64 = 0.05;
/// Seconds per word a run takes beyond its letters at a slow pace.
const SLOW_WORD: f64 = 0.1;
/// A run grows while it spans fewer than this many words past its first.
const MAX_SPAN: usize = 6;

/// Spreads each run of crushed words (under 80 ms) by letter count.
///
/// The swollen word before a run (much longer than its letters need) joins
/// it. When the next word starts too soon to leave the run room, that word
/// joins the run. With no word after it, the run's ceiling is `limit` (the
/// end of the audio, or of a phrase), and with no room there the run
/// borrows the words before it. The run takes the time up to its ceiling,
/// at most what its letters need at a slow pace, never ending before its
/// last word did.
pub fn even_out(words: &mut [HeardWord], limit: f64) -> Result<()> {
    let p = Patterns::get()?;
    let weight: Vec<usize> = words.iter().map(|w| letters(p, &w.text)).collect();
    let sum = |a: usize, b: usize| weight[a..=b].iter().sum::<usize>() as f64;
    let crushed = |w: &HeardWord| w.end - w.start < CRUSHED;
    let swollen = |w: &HeardWord, n: usize| w.end - w.start > SWOLLEN_BASE + SLOW_LETTER * n as f64;
    let len = words.len();
    let mut i = 0;
    while i < len {
        if !crushed(&words[i]) {
            i += 1;
            continue;
        }
        let mut a = i;
        while a > 0 && crushed(&words[a - 1]) {
            a -= 1;
        }
        if a > 0 && swollen(&words[a - 1], weight[a - 1]) {
            a -= 1;
        }
        let mut b = i;
        while b + 1 < len && crushed(&words[b + 1]) {
            b += 1;
        }
        let from = words[a].start;
        while b + 1 < len && words[b + 1].start - from < MIN_LETTER * sum(a, b) && b - a < MAX_SPAN
        {
            b += 1;
        }
        let ceiling = if b + 1 < len {
            words[b + 1].start
        } else {
            js_min(limit, words[b].end + SLOW_LETTER * weight[b] as f64)
        };
        while a > 0 && ceiling - words[a].start < MIN_LETTER * sum(a, b) && b - a < MAX_SPAN {
            a -= 1;
        }
        let total = sum(a, b);
        let start = words[a].start;
        let count = (b - a + 1) as f64;
        let to = js_max(
            words[b].end,
            js_min(ceiling, start + SLOW_LETTER * total + SLOW_WORD * count),
        );
        let mut t = start;
        for (w, &n) in words[a..=b].iter_mut().zip(&weight[a..=b]) {
            let d = ((to - start) * n as f64) / total;
            w.start = t;
            w.end = t + d;
            t += d;
        }
        i = b + 1;
    }
    Ok(())
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
    fn swollen_word_shares_with_the_crushed_run() -> Result<()> {
        let mut words = [
            word("swallowed", 0.0, 2.0),
            word("it", 2.0, 2.01),
            word("all", 2.01, 2.02),
            word("next", 3.0, 3.4),
        ];
        even_out(&mut words, f64::INFINITY)?;
        // 14 letters would take 1.98 s at a slow pace: the run keeps its
        // own 2.02 s, shared by letter count.
        assert_eq!(words[0].start, 0.0);
        assert!((words[2].end - 2.02).abs() < 1e-12);
        assert!((words[1].end - words[1].start - 2.02 * 2.0 / 14.0).abs() < 1e-12);
        assert_eq!(words[3].start, 3.0);
        Ok(())
    }

    #[test]
    fn run_at_the_end_stays_under_the_limit() -> Result<()> {
        let mut words = [
            word("long", 0.0, 0.5),
            word("a", 0.95, 0.96),
            word("b", 0.96, 0.97),
        ];
        even_out(&mut words, 1.0)?;
        assert!((words[2].end - 1.0).abs() < 1e-12);
        assert!(words.iter().all(|w| w.end >= w.start));
        Ok(())
    }

    #[test]
    fn nothing_crushed_changes_nothing() -> Result<()> {
        let mut words = [word("a", 0.0, 0.5), word("b", 0.5, 1.0)];
        let before = words.clone();
        even_out(&mut words, 1.0)?;
        assert_eq!(words, before);
        Ok(())
    }
}
