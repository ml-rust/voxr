//! A run's kept takes joined into its voiceover, and what the speak command
//! prints of each part: its check, its heat, its stray pauses and clicks,
//! and the parts that failed every take.

use serde::Serialize;

use crate::check::StrayPause;
use crate::error::{Error, Result};
use crate::text::spoken;

use super::engine::Graded;
use super::join::{JOIN_RATE, JoinPart, JoinedTrack, join_takes};
use super::judge::JudgedTake;
use super::split::BestTakes;

/// One part of the voiceover as the speak command reports it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PartCheck {
    /// The part's start in the voiceover, in seconds to 2 decimals.
    pub at: f64,
    /// The part's text as captions show it.
    pub text: String,
    /// The kept take passed its check.
    pub ok: bool,
    /// The words heard, clipped.
    pub heard: String,
    /// Why the take fails; empty when it passes.
    pub why: String,
    /// The seed that made the kept take.
    pub seed: f64,
    /// Words both recognizers missed or misheard, marked for a listen.
    pub doubt: Vec<String>,
    /// The part's hottest sentence's heat, questions aside.
    pub heat: f64,
    /// That sentence's script text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hot_text: Option<String>,
    /// Pauses the script does not ask for.
    pub pauses: Vec<StrayPause>,
    /// Clicks left after every repair, in seconds into the take.
    pub clicks: Vec<f64>,
}

impl BestTakes<JudgedTake> {
    /// Joins every part's kept take ([`join_takes`]), heat read against
    /// `voice_median`, the median pitch of the voice's reference.
    ///
    /// Each take is at [`JOIN_RATE`]; another rate is refused.
    pub fn join(&self, voice_median: f64) -> Result<JoinedTrack> {
        if self.parts.len() != self.best.len() {
            return Err(Error::DataError {
                reason: format!(
                    "{} parts hold {} kept takes; keep one take per part, as best_takes returns them",
                    self.parts.len(),
                    self.best.len()
                ),
            });
        }
        let mut parts = Vec::with_capacity(self.parts.len());
        for (i, (part, b)) in self.parts.iter().zip(&self.best).enumerate() {
            let take = &b.verdict.take;
            if take.rate != JOIN_RATE {
                return Err(Error::InvalidArgument {
                    arg: "best",
                    reason: format!(
                        "part {} \"{}\" holds a take at {} Hz; resample it to {JOIN_RATE} Hz before joining",
                        i + 1,
                        part.text,
                        take.rate
                    ),
                });
            }
            parts.push(JoinPart {
                part,
                audio: &take.samples,
                heard: &b.verdict.heard.words,
            });
        }
        join_takes(&parts, voice_median)
    }

    /// Each part's check, heat and marks, as the speak command prints them,
    /// with its start in `track`.
    pub fn checks(&self, track: &JoinedTrack) -> Result<Vec<PartCheck>> {
        self.parts
            .iter()
            .zip(&self.best)
            .enumerate()
            .map(|(i, (part, b))| {
                let (Some(at), Some(heat)) = (track.start_at(i), track.heat.get(i)) else {
                    return Err(Error::DataError {
                        reason: format!(
                            "part {} has no place in the joined track of {} parts; join this run's own takes",
                            i + 1,
                            track.starts.len()
                        ),
                    });
                };
                let v = &b.verdict;
                Ok(PartCheck {
                    at,
                    text: spoken(&part.text)?,
                    ok: v.ok(),
                    heard: v.heard().to_string(),
                    why: v.why().to_string(),
                    seed: b.seed,
                    doubt: v.heard.doubt.clone(),
                    heat: heat.heat,
                    hot_text: heat.hot_text.clone(),
                    pauses: v.heard.pauses.clone(),
                    clicks: v.clicks.clone(),
                })
            })
            .collect()
    }

    /// One line per part whose kept take still fails: every take failed,
    /// so it is reworded or re-rolled.
    pub fn problems(&self) -> Vec<String> {
        self.parts
            .iter()
            .zip(&self.best)
            .enumerate()
            .filter(|(_, (_, b))| !b.verdict.ok())
            .map(|(i, (part, b))| {
                format!(
                    "part {n} \"{text}\": {why} (heard \"{heard}\"). Three takes failed, so reword it first (a longer line, another word order). --reroll {n} draws new takes.",
                    n = i + 1,
                    text = part.text,
                    why = b.verdict.why(),
                    heard = b.verdict.heard(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::{HeardWord, Verdict};
    use crate::take::engine::Take;
    use crate::take::hear::HeardCheck;
    use crate::take::step::PartTake;
    use crate::text::{ChunkOptions, chunks};

    fn judged(ok: bool, rate: u32) -> JudgedTake {
        JudgedTake {
            heard: HeardCheck {
                check: Verdict {
                    ok,
                    extra: Vec::new(),
                    missing: Vec::new(),
                    unsure: Some(Vec::new()),
                    why: if ok {
                        String::new()
                    } else {
                        "missing: north".to_string()
                    },
                    badness: Some(if ok { 0.0 } else { 1.0 }),
                },
                doubt: vec!["river".to_string()],
                words: Vec::<HeardWord>::new(),
                pauses: Vec::new(),
                heard: "the river runs".to_string(),
            },
            take: Take {
                samples: vec![0.0; 4800],
                rate,
            },
            seed: 7.0,
            thumps: 0,
            fixed: 0,
            clicks: vec![0.25],
        }
    }

    fn best(ok: bool, rate: u32) -> Result<BestTakes<JudgedTake>> {
        let parts = chunks(
            "The river runs {N|north}.",
            ChunkOptions {
                lines: false,
                max: 700,
                whole: true,
            },
        )?;
        Ok(BestTakes {
            parts,
            best: vec![PartTake {
                seed: 7.0,
                verdict: judged(ok, rate),
            }],
            log: Vec::new(),
        })
    }

    #[test]
    fn checks_carry_each_parts_marks() -> Result<()> {
        let b = best(true, JOIN_RATE)?;
        let track = b.join(150.0)?;
        let checks = b.checks(&track)?;
        assert_eq!(checks.len(), 1);
        let c = &checks[0];
        assert_eq!(
            (c.at, c.text.as_str(), c.ok),
            (0.0, "The river runs N.", true)
        );
        assert_eq!(
            (c.doubt.as_slice(), c.clicks.as_slice()),
            (&["river".to_string()][..], &[0.25][..])
        );
        assert_eq!(c.hot_text, None);
        assert!(b.problems().is_empty());
        Ok(())
    }

    #[test]
    fn a_failed_part_is_a_problem() -> Result<()> {
        let b = best(false, JOIN_RATE)?;
        assert_eq!(
            b.problems(),
            [
                "part 1 \"The river runs {N|north}.\": missing: north (heard \"the river runs\"). Three takes failed, so reword it first (a longer line, another word order). --reroll 1 draws new takes."
            ]
        );
        Ok(())
    }

    #[test]
    fn another_rate_is_refused() -> Result<()> {
        let b = best(true, 24000)?;
        assert!(matches!(
            b.join(150.0),
            Err(Error::InvalidArgument { arg: "best", .. })
        ));
        Ok(())
    }
}
