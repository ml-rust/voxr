//! A part that fails every take whole, spoken again in pieces: its beats
//! first, then a beat's sentences. Shorter inputs keep the model stable,
//! at the cost of more draws of the voice.

use super::engine::{Graded, SpeechEngine, TakeJudge, TakeStore};
use super::files::TakesFile;
use super::pick::TakeLoop;
use super::step::{PartTake, TakeStep};
use crate::error::Result;
use crate::text::{Chunk, PhraseKind};

/// Rounds of splitting: beats, then sentences.
pub const SPLIT_ROUNDS: usize = 2;

/// A part broken into pieces: its beats, the phrases up to each blank line
/// or pause mark. A part with one beat breaks into its sentences.
///
/// Each piece keeps the part's delivery note, and the pause and kind of its
/// own last phrase. The last piece keeps the part's own pause.
pub fn break_up(part: &Chunk) -> Vec<Chunk> {
    let mut groups: Vec<Vec<_>> = Vec::new();
    let mut group = Vec::new();
    let last = part.sentences.len().saturating_sub(1);
    for (k, p) in part.sentences.iter().enumerate() {
        group.push(p.clone());
        if matches!(p.kind, PhraseKind::Beat | PhraseKind::Mark) || k == last {
            groups.push(std::mem::take(&mut group));
        }
    }
    let pieces = if groups.len() > 1 {
        groups
    } else {
        part.sentences.iter().map(|p| vec![p.clone()]).collect()
    };
    let count = pieces.len();
    pieces
        .into_iter()
        .enumerate()
        .filter_map(|(k, ps)| {
            let end = ps.last()?;
            let (pause, kind) = (end.pause, end.kind);
            let text: Vec<&str> = ps.iter().map(|p| p.text.as_str()).collect();
            Some(Chunk {
                text: text.join(" "),
                pause: if k + 1 == count { part.pause } else { pause },
                kind,
                delivery: part.delivery.clone(),
                sentences: ps,
            })
        })
        .collect()
}

/// A run's parts, as finally spoken, and the take kept for each.
#[derive(Debug, Clone, PartialEq)]
pub struct BestTakes<V> {
    /// The parts, split where a whole part failed every take.
    pub parts: Vec<Chunk>,
    /// The take kept for each part.
    pub best: Vec<PartTake<V>>,
    /// Every step, in order.
    pub log: Vec<TakeStep>,
}

/// A JavaScript array index: the canonical decimal text of an integer
/// under `2^32 - 1`. An object lists these keys first, ascending.
fn array_index(key: &str) -> Option<u32> {
    let n: u32 = key.parse().ok()?;
    (n != u32::MAX && n.to_string() == key).then_some(n)
}

impl<V> BestTakes<V> {
    /// The `.takes.json` a run at `seed` writes: each part's text to the
    /// seed of its kept take.
    ///
    /// The takes are listed as `Object.fromEntries` lists them: a text met
    /// twice keeps its first place and its last seed, and texts that are
    /// array indices come first, ascending.
    pub fn takes_file(&self, seed: f64) -> TakesFile {
        let mut takes: Vec<(String, f64)> = Vec::new();
        for (part, take) in self.parts.iter().zip(&self.best) {
            match takes.iter_mut().find(|(t, _)| *t == part.text) {
                Some(entry) => entry.1 = take.seed,
                None => takes.push((part.text.clone(), take.seed)),
            }
        }
        let (mut indices, named): (Vec<_>, Vec<_>) = takes
            .into_iter()
            .partition(|(t, _)| array_index(t).is_some());
        indices.sort_by_key(|(t, _)| array_index(t));
        indices.extend(named);
        TakesFile {
            seed,
            takes: indices,
        }
    }
}

impl<E, J, S> TakeLoop<'_, E, J, S>
where
    E: SpeechEngine,
    J: TakeJudge,
    S: TakeStore,
{
    /// The best take of each part, splitting what fails whole.
    ///
    /// [`pick`](TakeLoop::pick) picks every part, re-rolling `rolls`. Then,
    /// for up to [`SPLIT_ROUNDS`] rounds, each part whose kept take fails
    /// and that holds more than one phrase is replaced by its pieces
    /// ([`break_up`]), and every part is picked again with no re-rolls.
    pub fn best_takes(
        &mut self,
        mut parts: Vec<Chunk>,
        rolls: &[usize],
        random: &mut dyn FnMut() -> f64,
    ) -> Result<BestTakes<J::Verdict>> {
        let mut log = Vec::new();
        let mut best = self.pick(&parts, rolls, random, &mut log)?;
        for round in 0..SPLIT_ROUNDS {
            let split: Vec<bool> = parts
                .iter()
                .zip(&best)
                .map(|(u, b)| !b.verdict.ok() && u.sentences.len() > 1)
                .collect();
            if !split.iter().any(|&s| s) {
                break;
            }
            log.push(TakeStep::Split {
                round,
                parts: split
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &s)| s.then_some(i))
                    .collect(),
            });
            parts = parts
                .iter()
                .zip(&split)
                .flat_map(|(u, &s)| if s { break_up(u) } else { vec![u.clone()] })
                .collect();
            best = self.pick(&parts, &[], random, &mut log)?;
        }
        Ok(BestTakes { parts, best, log })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::take::engine::tests::{MemoryStore, ScriptedEngine, ScriptedJudge, sequence};
    use crate::take::pick::tests::{parts, setup};
    use crate::take::seed::text_seed;

    #[test]
    fn beats_break_first_then_sentences() -> Result<()> {
        let script = "One two three.\n\nFour five six. Seven eight nine.";
        let whole = parts(script)?;
        assert_eq!(whole.len(), 1);
        let beats = break_up(&whole[0]);
        let texts: Vec<&str> = beats.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            texts,
            ["One two three.", "Four five six. Seven eight nine."]
        );
        assert_eq!(beats[1].pause, whole[0].pause);
        let sentences = break_up(&beats[1]);
        let texts: Vec<&str> = sentences.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["Four five six.", "Seven eight nine."]);
        Ok(())
    }

    #[test]
    fn a_part_failing_whole_is_spoken_by_beats() -> Result<()> {
        let script = "One two three.\n\nFour five six.";
        let chosen = HashMap::new();
        let (mut e, mut j, mut s): (ScriptedEngine, ScriptedJudge, MemoryStore) =
            Default::default();
        let whole = parts(script)?;
        let seed = text_seed(&whole[0].text, 7.0);
        for k in 0..3 {
            j.score(&whole[0].text, seed + 1000.0 * k as f64, false, Some(1.0));
        }
        let mut tl = TakeLoop {
            setup: setup(&chosen),
            engine: &mut e,
            judge: &mut j,
            store: &mut s,
        };
        let got = tl.best_takes(whole, &[], &mut sequence(vec![]))?;
        let texts: Vec<&str> = got.parts.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["One two three.", "Four five six."]);
        assert!(got.best.iter().all(|b| b.verdict.ok));
        assert!(got.log.contains(&TakeStep::Split {
            round: 0,
            parts: vec![0]
        }));
        let file = got.takes_file(7.0);
        assert_eq!(file.takes[0].1, text_seed("One two three.", 7.0));
        Ok(())
    }

    #[test]
    fn array_index_texts_list_first() {
        let chunk = |text: &str| Chunk {
            text: text.to_string(),
            pause: 0.0,
            kind: PhraseKind::End,
            sentences: Vec::new(),
            delivery: None,
        };
        let take = |seed: f64| PartTake { seed, verdict: () };
        let got = BestTakes {
            parts: vec![chunk("b"), chunk("10"), chunk("2"), chunk("b"), chunk("01")],
            best: vec![take(1.0), take(2.0), take(3.0), take(4.0), take(5.0)],
            log: Vec::new(),
        };
        let file = got.takes_file(7.0);
        let keys: Vec<(&str, f64)> = file.takes.iter().map(|(t, s)| (t.as_str(), *s)).collect();
        assert_eq!(keys, [("2", 3.0), ("10", 2.0), ("b", 4.0), ("01", 5.0)]);
    }
}
