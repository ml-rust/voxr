//! One step of a voice's design or reuse: the message the skill's `speak`
//! command prints for each step, in the order the steps happen.

use crate::js::js_to_fixed_text;
use crate::take::engine::TakeSpec;

/// One step of a voice's design or reuse, in the order it happens.
#[derive(Debug, Clone, PartialEq)]
pub enum AnchorStep {
    /// Hi-Fi cloning is asked for, and the designed voice in the cache has
    /// no saved transcript.
    HifiUntranscribed,
    /// Hi-Fi cloning holds a transcript, and the run has a style VoxCPM2
    /// ignores with it.
    HifiIgnoresStyle,
    /// The voice is designed, speaking the opening.
    Designing {
        /// The opening, as the script writes it.
        opening: String,
    },
    /// The engine speaks a take.
    Speak(TakeSpec),
    /// The reference is spoken again, slower.
    Slower {
        /// The delivery asked for, parentheses included.
        ask: &'static str,
        /// The pace the reference spoke at, in syllables per second.
        pace: f64,
    },
    /// Breaths in the reference are turned down.
    Breaths {
        /// How many breath runs.
        runs: usize,
    },
}

impl AnchorStep {
    /// The line the skill prints for this step, when it prints one.
    pub fn message(&self) -> Option<String> {
        match self {
            Self::HifiUntranscribed => Some(
                "speak --hifi: this designed voice has no saved transcript (it was made before --hifi existed). Pass a new --seed to design it again with one, or drop --hifi."
                    .to_string(),
            ),
            Self::HifiIgnoresStyle => Some(
                "speak --hifi: VoxCPM2 ignores --style in Hi-Fi cloning. Delivery notes on a line still apply to that line."
                    .to_string(),
            ),
            Self::Designing { opening } => Some(format!(
                "speak: designing the voice from --voice, speaking the script's opening: \"{opening}\""
            )),
            Self::Slower { ask, pace } => Some(format!(
                "speak: speaking the designed voice's reference again, {} (it spoke at {} syllables per second): clones follow a reference's pace and manner.",
                ask.strip_prefix('(')
                    .and_then(|a| a.strip_suffix(')'))
                    .unwrap_or(*ask),
                js_to_fixed_text(*pace, 1)
            )),
            Self::Breaths { runs } => Some(format!(
                "speak: turned down {runs} breath(s) in the designed voice's reference."
            )),
            Self::Speak(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breaths_print_their_count() {
        assert_eq!(
            AnchorStep::Breaths { runs: 2 }.message().as_deref(),
            Some("speak: turned down 2 breath(s) in the designed voice's reference.")
        );
        assert_eq!(
            AnchorStep::Speak(TakeSpec {
                index: 0,
                text: String::new(),
                seed: 7.0,
                delivery: None,
            })
            .message(),
            None
        );
    }
}
