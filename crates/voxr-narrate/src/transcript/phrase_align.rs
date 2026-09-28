//! Script words placed phrase by phrase on recognized timings, and a phrase
//! found in the spoken words.

use serde::Serialize;

use super::fit::PhraseGroups;
use crate::check::speech_match::{alike, letters, norm, script_words};
use crate::check::{HeardWord, align_script};
use crate::error::Result;
use crate::js::js_max;
use crate::pattern::Patterns;
use crate::text::Cue;

/// A word within this many seconds outside its phrase's span is inside it.
const EDGE_SLACK: f64 = 0.02;
/// A word shorter than this, in seconds, has no time of its own.
const MIN_WORD: f64 = 0.04;
/// A run of spoken words matches a phrase when at least this share of its
/// words are alike.
pub const PHRASE_MATCH: f64 = 0.7;

/// Script words, each with the phrase it belongs to.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhraseWords {
    /// The words, phrase by phrase.
    pub words: Vec<HeardWord>,
    /// The index of each word's phrase.
    pub phrase: Vec<usize>,
}

impl PhraseWords {
    /// The grouping [`fit_file`](super::fit::fit_file) reads, over the
    /// phrase spans these words were aligned to.
    pub fn groups<'a>(&'a self, spans: &'a [Cue]) -> PhraseGroups<'a> {
        PhraseGroups {
            spans,
            phrase: &self.phrase,
        }
    }
}

/// Script words on recognized timings, phrase by phrase, when the audio's
/// phrase spans are known.
///
/// The words align with the whole transcript first, by text
/// ([`align_script()`] over the phrase texts joined by spaces), so a word the
/// recognizer timed early still finds its phrase. Each phrase then takes
/// its count of script words, and its words are placed inside its span:
/// words timed inside the span keep their times, and words that spill past
/// an edge (the recognizer runs early at phrase starts) share the room
/// between that edge and the nearest word inside, by letter count. A word
/// between the two that sits outside the span (a stray time) goes between
/// its neighbors. A phrase with no word inside shares its whole span. A
/// phrase that gets no words is skipped.
pub fn align_phrases(words: &[HeardWord], phrases: &[Cue]) -> Result<PhraseWords> {
    let p = Patterns::get()?;
    let script = phrases
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let all = align_script(words, &script)?;
    let mut out = PhraseWords::default();
    let mut k = 0;
    for (pi, span) in phrases.iter().enumerate() {
        let n = script_words(p, &span.text).len();
        let mut group = all[k.min(all.len())..(k + n).min(all.len())].to_vec();
        k += n;
        if group.is_empty() {
            continue;
        }
        let weight: Vec<usize> = group.iter().map(|w| letters(p, &w.text)).collect();
        let inside = |w: &HeardWord| {
            w.start >= span.start - EDGE_SLACK
                && w.end <= span.end + EDGE_SLACK
                && w.end - w.start >= MIN_WORD
        };
        match group.iter().position(inside) {
            None => spread(&mut group, &weight, span.start, span.end),
            Some(first) => {
                let mut last = group.len() - 1;
                while !inside(&group[last]) {
                    last -= 1;
                }
                if first > 0 {
                    let to = group[first].start;
                    spread(&mut group[..first], &weight[..first], span.start, to);
                }
                if last + 1 < group.len() {
                    let from = group[last].end;
                    spread(&mut group[last + 1..], &weight[last + 1..], from, span.end);
                }
                for i in first + 1..last {
                    if inside(&group[i]) {
                        continue;
                    }
                    group[i].start = group[i - 1].end;
                    group[i].end = js_max(group[i].start, group[i + 1].start);
                }
            }
        }
        out.phrase.extend(std::iter::repeat_n(pi, group.len()));
        out.words.extend(group);
    }
    Ok(out)
}

/// Shares `[from, to]` among words by letter count, in order.
fn spread(list: &mut [HeardWord], weight: &[usize], from: f64, to: f64) {
    let total = weight.iter().sum::<usize>() as f64;
    let mut t = from;
    for (w, &n) in list.iter_mut().zip(weight) {
        let d = (js_max(0.0, to - from) * n as f64) / total;
        w.start = t;
        w.end = t + d;
        t += d;
    }
}

/// Where a phrase was found in the spoken words.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PhraseHit {
    /// Start of the first matching word, in the words' seconds.
    pub start: f64,
    /// End of the last matching word.
    pub end: f64,
    /// The share of the phrase's words that are alike.
    pub score: f64,
}

/// Finds a phrase in the spoken words: the run of words that matches it
/// best, in order.
///
/// Speech recognition gets some words wrong, so a run matches when at
/// least [`PHRASE_MATCH`] of its words are alike. Words compare by their
/// normalized pieces. Ties go to the run nearest `near`, in the words'
/// seconds, the first on an equal distance. A phrase with no words is never
/// found.
pub fn find_phrase(words: &[HeardWord], phrase: &str, near: f64) -> Result<Option<PhraseHit>> {
    let p = Patterns::get()?;
    let want = norm(p, phrase);
    if want.is_empty() {
        return Ok(None);
    }
    let said: Vec<(String, usize)> = words
        .iter()
        .enumerate()
        .flat_map(|(i, w)| norm(p, &w.text).into_iter().map(move |t| (t, i)))
        .collect();
    let mut best: Option<PhraseHit> = None;
    for i in 0..(said.len() + 1).saturating_sub(want.len()) {
        let hits = want
            .iter()
            .enumerate()
            .filter(|(k, t)| alike(&said[i + k].0, t))
            .count();
        let score = hits as f64 / want.len() as f64;
        if score < PHRASE_MATCH {
            continue;
        }
        let hit = PhraseHit {
            start: words[said[i].1].start,
            end: words[said[i + want.len() - 1].1].end,
            score,
        };
        let better = best.is_none_or(|b| {
            score > b.score
                || (score == b.score && (hit.start - near).abs() < (b.start - near).abs())
        });
        if better {
            best = Some(hit);
        }
    }
    Ok(best)
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

    fn cue(text: &str, start: f64, end: f64) -> Cue {
        Cue {
            start,
            end,
            text: text.to_string(),
        }
    }

    #[test]
    fn spilled_word_shares_the_room_at_the_edge() -> Result<()> {
        let heard = [
            word("The", 0.40, 0.55),
            word("river", 0.6, 0.9),
            word("runs", 0.95, 1.5),
            word("north", 2.1, 2.6),
        ];
        let spans = [cue("The river runs", 0.5, 1.6), cue("north", 2.0, 2.8)];
        let out = align_phrases(&heard, &spans)?;
        assert_eq!(out.phrase, [0, 0, 0, 1]);
        assert_eq!(out.words[0].start, 0.5);
        assert!((out.words[0].end - 0.6).abs() < 1e-12);
        assert_eq!(out.words[3].start, 2.1);
        Ok(())
    }

    #[test]
    fn phrase_with_no_word_inside_shares_its_span() -> Result<()> {
        let heard = [word("and", 5.0, 5.2), word("back", 5.3, 5.6)];
        let out = align_phrases(&heard, &[cue("and back", 3.5, 4.5)])?;
        assert_eq!(out.words[0].start, 3.5);
        assert!((out.words[1].end - 4.5).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn phrase_found_with_a_misheard_word() -> Result<()> {
        let words = [
            word("bermula", 1.0, 1.4),
            word("pada", 1.4, 1.6),
            word("tahun", 1.6, 1.9),
            word("1857,", 1.9, 2.5),
            word("ketika", 2.6, 3.0),
        ];
        let hit = find_phrase(&words, "bermula pada tahun 1857 kemudian", 0.0)?;
        assert_eq!(
            hit,
            Some(PhraseHit {
                start: 1.0,
                end: 3.0,
                score: 0.8
            })
        );
        assert_eq!(
            find_phrase(&words, "bermula sini tahun 1985 kemudian", 0.0)?,
            None
        );
        assert_eq!(find_phrase(&words, "—", 0.0)?, None);
        Ok(())
    }
}
