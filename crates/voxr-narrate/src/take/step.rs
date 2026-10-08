//! The steps a take loop takes, as it logs them, and the take it keeps for
//! each part.

use super::engine::TakeSpec;
use crate::js::js_string;

/// One step of the take loop, in the order it happens.
#[derive(Debug, Clone, PartialEq)]
pub enum TakeStep {
    /// A re-rolled part drew a fresh seed.
    Rerolled {
        /// The part, from 0.
        index: usize,
        /// The seed drawn.
        seed: f64,
    },
    /// The engine is about to speak this many takes in a row: one batch
    /// to the model.
    Batch {
        /// Takes in the batch.
        parts: usize,
    },
    /// The engine speaks a take.
    Speak(TakeSpec),
    /// The engine spoke a take of its batch.
    Spoken {
        /// Takes of the batch spoken so far, this one included.
        done: usize,
        /// Takes in the batch.
        total: usize,
    },
    /// The judge grades a take.
    Judge(TakeSpec),
    /// A take failed and a new take follows, or this one was new.
    Failed {
        /// The part, from 0.
        index: usize,
        /// The failed take's seed.
        seed: f64,
        /// Why it failed.
        why: String,
        /// The words heard.
        heard: String,
        /// Another try follows.
        regenerating: bool,
    },
    /// Parts that failed every take whole are spoken again in pieces.
    Split {
        /// 0 for the first split, 1 for the second.
        round: usize,
        /// The parts split, from 0.
        parts: Vec<usize>,
    },
}

impl TakeStep {
    /// The line the skill prints for this step, when it prints one.
    pub fn message(&self) -> Option<String> {
        match self {
            Self::Rerolled { index, seed } => Some(format!(
                "speak: part {} re-rolled with seed {}.",
                index + 1,
                js_string(*seed)
            )),
            Self::Failed {
                index,
                seed,
                why,
                heard,
                regenerating,
            } => Some(format!(
                "speak: part {} (seed {}): {why}. Heard: \"{heard}\".{}",
                index + 1,
                js_string(*seed),
                if *regenerating { " Regenerating." } else { "" }
            )),
            Self::Split { round, parts } => {
                let list: Vec<String> = parts.iter().map(|i| (i + 1).to_string()).collect();
                Some(format!(
                    "speak: part(s) {} failed every take whole, so {} are spoken apart.",
                    list.join(", "),
                    if *round > 0 {
                        "their sentences"
                    } else {
                        "their beats"
                    }
                ))
            }
            Self::Batch { .. } | Self::Speak(_) | Self::Spoken { .. } | Self::Judge(_) => None,
        }
    }
}

/// The take kept for one part.
#[derive(Debug, Clone, PartialEq)]
pub struct PartTake<V> {
    /// The seed that made it.
    pub seed: f64,
    /// Its verdict.
    pub verdict: V,
}
