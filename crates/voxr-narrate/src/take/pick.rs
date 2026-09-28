//! The best take of each part: its recorded take, else the seed its own
//! text draws, retried with new seeds, and the best of the tries kept.

use std::collections::{HashMap, HashSet};

use super::engine::{
    Graded, SpeakRequest, SpeechEngine, Take, TakeJudge, TakeSpec, TakeStore, VoiceSource,
};
use super::seed::text_seed;
use super::step::{PartTake, TakeStep};
use crate::error::{Error, Result};
use crate::text::{Chunk, say};

/// Takes a part gets before the best one is kept.
pub const ATTEMPTS: usize = 3;
/// The seed a retry adds to the seed of the take that failed.
pub const RETRY_STEP: f64 = 1000.0;
/// The smallest seed a re-rolled part draws.
const REROLL_BASE: f64 = 100000.0;
/// The span re-rolled seeds are drawn from.
const REROLL_SPAN: f64 = 900000.0;

/// What a run speaks with.
#[derive(Debug)]
pub struct PickSetup<'a, H> {
    /// The run's own seed, which each part's text seed builds on.
    pub seed: f64,
    /// The seed of the take kept for each part, by its text.
    pub chosen: &'a HashMap<String, f64>,
    /// The run's style, parentheses included.
    pub style: Option<&'a str>,
    /// The voice every part speaks in.
    pub voice: VoiceSource<'a, H>,
    /// Classifier-free guidance.
    pub cfg: f64,
    /// Diffusion steps.
    pub steps: u32,
}

/// Speaks, stores and judges takes for a run.
pub struct TakeLoop<'a, E: SpeechEngine, J, S> {
    /// What the run speaks with.
    pub setup: PickSetup<'a, E::Reference>,
    /// Makes takes.
    pub engine: &'a mut E,
    /// Grades takes.
    pub judge: &'a mut J,
    /// Holds takes already made.
    pub store: &'a mut S,
}

/// `100000 + Math.floor(random * 900000)`: a re-rolled part's fresh seed.
fn reroll_seed(random: f64) -> f64 {
    REROLL_BASE + (random * REROLL_SPAN).floor()
}

/// `a < b`, or `a <= b` when `or_equal`, with `None` as JavaScript's
/// `undefined`: every comparison with it is false.
fn beats(a: Option<f64>, b: Option<f64>, or_equal: bool) -> bool {
    let (a, b) = (a.unwrap_or(f64::NAN), b.unwrap_or(f64::NAN));
    a < b || (or_equal && a <= b)
}

/// The request for `spec`'s take, spoken as `text`. An empty run style is
/// no style.
fn request<'r, H>(
    setup: &'r PickSetup<'_, H>,
    spec: &'r TakeSpec,
    text: &'r str,
) -> SpeakRequest<'r, H> {
    SpeakRequest {
        text,
        delivery: spec.delivery.as_deref(),
        style: setup.style.filter(|s| !s.is_empty()),
        voice: setup.voice,
        seed: spec.seed,
        cfg: setup.cfg,
        steps: setup.steps,
        // VoxCPM2 is never sped up: its pace comes from the reference.
        speed: 1.0,
    }
}

impl<E, J, S> TakeLoop<'_, E, J, S>
where
    E: SpeechEngine,
    J: TakeJudge,
    S: TakeStore,
{
    fn judge_take(
        &mut self,
        spec: &TakeSpec,
        take: &Take,
        log: &mut Vec<TakeStep>,
    ) -> Result<J::Verdict> {
        log.push(TakeStep::Judge(spec.clone()));
        self.judge.judge(spec, take)
    }

    /// The best take of each part.
    ///
    /// A part's first seed is a fresh one when it is re-rolled, else its
    /// seed in `chosen`, else [`text_seed`]. Each of up to [`ATTEMPTS`]
    /// rounds speaks the takes the store lacks, then judges every take of
    /// the round in order. A failed take's part tries again at its seed
    /// plus [`RETRY_STEP`]. The best take by badness is kept.
    ///
    /// `rolls` names parts from 1 to re-roll. Each draws a fresh seed from
    /// `random`, a value in `[0, 1)`, in order: a part named twice keeps
    /// its last draw. A re-rolled part's recorded take, when the store
    /// holds it, is judged first and competes: a fresh take replaces it
    /// only when as good or better, since a re-roll is asked for what the
    /// check cannot hear. Otherwise only a better take replaces the one
    /// kept.
    ///
    /// Every step is appended to `log`.
    pub fn pick(
        &mut self,
        parts: &[Chunk],
        rolls: &[usize],
        random: &mut dyn FnMut() -> f64,
        log: &mut Vec<TakeStep>,
    ) -> Result<Vec<PartTake<J::Verdict>>> {
        if let Some(n) = rolls.iter().find(|&&n| n < 1 || n > parts.len()) {
            return Err(Error::InvalidArgument {
                arg: "rolls",
                reason: format!(
                    "re-roll part {n}: the script has parts 1 to {}. Pass a part number in that range",
                    parts.len()
                ),
            });
        }
        let mut fresh: Vec<(usize, f64)> = Vec::new();
        for &n in rolls {
            let seed = reroll_seed(random());
            match fresh.iter_mut().find(|(i, _)| *i == n - 1) {
                Some(entry) => entry.1 = seed,
                None => fresh.push((n - 1, seed)),
            }
        }
        let fresh_seed = |i: usize| fresh.iter().find(|(k, _)| *k == i).map(|&(_, s)| s);
        let mut best: Vec<Option<PartTake<J::Verdict>>> = parts.iter().map(|_| None).collect();
        for &(i, _) in &fresh {
            let part = &parts[i];
            let Some(&prev) = self.setup.chosen.get(&part.text) else {
                continue;
            };
            let spec = TakeSpec::new(i, part, prev);
            if self.store.has(&spec) {
                let take = self.store.load(&spec)?;
                let verdict = self.judge_take(&spec, &take, log)?;
                best[i] = Some(PartTake {
                    seed: prev,
                    verdict,
                });
            }
        }
        let mut todo: Vec<TakeSpec> = parts
            .iter()
            .enumerate()
            .map(|(i, part)| {
                let seed = fresh_seed(i)
                    .or_else(|| self.setup.chosen.get(&part.text).copied())
                    .unwrap_or_else(|| text_seed(&part.text, self.setup.seed));
                TakeSpec::new(i, part, seed)
            })
            .collect();
        log.extend(
            fresh
                .iter()
                .map(|&(index, seed)| TakeStep::Rerolled { index, seed }),
        );
        for attempt in 0..ATTEMPTS {
            if todo.is_empty() {
                break;
            }
            let missing: Vec<usize> = (0..todo.len())
                .filter(|&k| !self.store.has(&todo[k]))
                .collect();
            let mut made: HashSet<usize> = HashSet::new();
            let mut spoken: HashMap<usize, Take> = HashMap::new();
            for k in missing {
                let spec = &todo[k];
                let text = say(&spec.text)?;
                log.push(TakeStep::Speak(spec.clone()));
                let take = self.engine.speak(&request(&self.setup, spec, &text))?;
                self.store.save(spec, &take)?;
                made.insert(spec.index);
                spoken.insert(spec.index, take);
            }
            let mut again = Vec::new();
            for spec in todo {
                let take = match spoken.remove(&spec.index) {
                    Some(take) => take,
                    None => self.store.load(&spec)?,
                };
                let verdict = self.judge_take(&spec, &take, log)?;
                let rerolled = fresh_seed(spec.index).is_some();
                let slot = best.get_mut(spec.index).ok_or_else(|| Error::DataError {
                    reason: format!(
                        "take of part {} is outside the {} parts picked; build each take from its own part",
                        spec.index + 1,
                        parts.len()
                    ),
                })?;
                let keep = match &*slot {
                    None => true,
                    Some(b) => beats(verdict.badness(), b.verdict.badness(), rerolled),
                };
                let failure = (!verdict.ok())
                    .then(|| (verdict.why().to_string(), verdict.heard().to_string()));
                if keep {
                    *slot = Some(PartTake {
                        seed: spec.seed,
                        verdict,
                    });
                }
                if let Some((why, heard)) = failure {
                    let (index, seed) = (spec.index, spec.seed);
                    let retry = TakeSpec {
                        seed: seed + RETRY_STEP,
                        ..spec
                    };
                    let regenerating = attempt < ATTEMPTS - 1;
                    if made.contains(&index) || (regenerating && !self.store.has(&retry)) {
                        log.push(TakeStep::Failed {
                            index,
                            seed,
                            why,
                            heard,
                            regenerating,
                        });
                    }
                    again.push(retry);
                }
            }
            todo = again;
        }
        best.into_iter()
            .enumerate()
            .map(|(i, b)| {
                b.ok_or_else(|| Error::DataError {
                    reason: format!(
                        "part {} of {} has no judged take after picking; check that the store holds every take it saves",
                        i + 1,
                        parts.len()
                    ),
                })
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::take::engine::CloneReference;
    use crate::take::engine::tests::{
        HANDLE, MemoryStore, ScriptedEngine, ScriptedJudge, sequence,
    };
    use crate::text::{ChunkOptions, chunks};

    pub(crate) fn setup(chosen: &HashMap<String, f64>) -> PickSetup<'_, u8> {
        PickSetup {
            seed: 7.0,
            chosen,
            style: None,
            voice: VoiceSource::Clone {
                reference: CloneReference::Handle(&HANDLE),
                transcript: None,
            },
            cfg: 1.6,
            steps: 16,
        }
    }

    pub(crate) fn parts(script: &str) -> Result<Vec<Chunk>> {
        chunks(
            script,
            ChunkOptions {
                lines: false,
                max: 700,
                whole: true,
            },
        )
    }

    #[test]
    fn first_try_success_speaks_and_judges_once() -> Result<()> {
        let chosen = HashMap::new();
        let (mut e, mut j, mut s): (ScriptedEngine, ScriptedJudge, MemoryStore) =
            Default::default();
        let mut tl = TakeLoop {
            setup: setup(&chosen),
            engine: &mut e,
            judge: &mut j,
            store: &mut s,
        };
        let units = parts("Hello there, friend.")?;
        let mut log = Vec::new();
        let best = tl.pick(&units, &[], &mut sequence(vec![]), &mut log)?;
        let seed = text_seed("Hello there, friend.", 7.0);
        assert_eq!(best[0].seed, seed);
        assert_eq!(
            e.calls,
            [("Hello there, friend.".to_string(), None, seed, "handle")]
        );
        assert_eq!(j.calls.len(), 1);
        Ok(())
    }

    #[test]
    fn a_failed_take_retries_and_the_best_is_kept() -> Result<()> {
        let chosen = HashMap::new();
        let (mut e, mut j, mut s): (ScriptedEngine, ScriptedJudge, MemoryStore) =
            Default::default();
        let text = "One two three four.";
        let seed = text_seed(text, 7.0);
        j.score(text, seed, false, Some(2.0));
        j.score(text, seed + 1000.0, false, Some(1.0));
        j.score(text, seed + 2000.0, false, Some(3.0));
        let mut tl = TakeLoop {
            setup: setup(&chosen),
            engine: &mut e,
            judge: &mut j,
            store: &mut s,
        };
        let mut log = Vec::new();
        let best = tl.pick(&parts(text)?, &[], &mut sequence(vec![]), &mut log)?;
        assert_eq!(best[0].seed, seed + 1000.0);
        assert!(!best[0].verdict.ok);
        let seeds: Vec<f64> = e.calls.iter().map(|c| c.2).collect();
        assert_eq!(seeds, [seed, seed + 1000.0, seed + 2000.0]);
        let messages: Vec<String> = log.iter().filter_map(TakeStep::message).collect();
        assert_eq!(messages.len(), 3);
        assert!(messages[2].ends_with("\"heard One two three four.\"."));
        Ok(())
    }

    #[test]
    fn a_reroll_tie_replaces_the_recorded_take() -> Result<()> {
        let text = "Hello there, friend.";
        let chosen = HashMap::from([(text.to_string(), 555.0)]);
        let (mut e, mut j, mut s): (ScriptedEngine, ScriptedJudge, MemoryStore) =
            Default::default();
        s.hold(text, 555.0, None);
        let fresh = reroll_seed(0.25);
        j.score(text, 555.0, true, Some(0.5));
        j.score(text, fresh, true, Some(0.5));
        let mut tl = TakeLoop {
            setup: setup(&chosen),
            engine: &mut e,
            judge: &mut j,
            store: &mut s,
        };
        let mut log = Vec::new();
        let best = tl.pick(&parts(text)?, &[1], &mut sequence(vec![0.25]), &mut log)?;
        assert_eq!(best[0].seed, fresh);
        assert_eq!(
            j.calls,
            [(text.to_string(), 555.0), (text.to_string(), fresh)]
        );
        Ok(())
    }

    #[test]
    fn a_part_out_of_range_is_refused() {
        let chosen = HashMap::new();
        let (mut e, mut j, mut s): (ScriptedEngine, ScriptedJudge, MemoryStore) =
            Default::default();
        let mut tl = TakeLoop {
            setup: setup(&chosen),
            engine: &mut e,
            judge: &mut j,
            store: &mut s,
        };
        let units = vec![Chunk {
            text: "Hi.".to_string(),
            pause: 0.0,
            kind: crate::text::PhraseKind::End,
            sentences: Vec::new(),
            delivery: None,
        }];
        let got = tl.pick(&units, &[2], &mut sequence(vec![0.5]), &mut Vec::new());
        assert!(matches!(
            got,
            Err(Error::InvalidArgument { arg: "rolls", .. })
        ));
    }

    #[test]
    fn undefined_badness_never_compares() {
        assert!(!beats(None, Some(1.0), true));
        assert!(!beats(Some(1.0), None, true));
        assert!(beats(Some(1.0), Some(1.0), true));
        assert!(!beats(Some(1.0), Some(1.0), false));
    }
}
