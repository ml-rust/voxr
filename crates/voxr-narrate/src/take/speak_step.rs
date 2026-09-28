//! One step of a VoxCPM2 run, in order: the notes before it, each call to
//! the model, the voice's design, the take loop, and the repairs made. The
//! speak command prints each step's message.

use super::anchor::{AnchorStep, js_word_count};
use super::step::TakeStep;
use crate::text::Chunk;

/// A part under this many words fails more often with VoxCPM2.
const SHORT_WORDS: usize = 3;

/// One step of a VoxCPM2 run, in the order it happens.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeakStep {
    /// Neither a voice description nor a reference is given: VoxCPM2 picks
    /// its own voice.
    NoVoice,
    /// Parts of one or two words, which fail more often: each part's number
    /// from 1 and its text.
    ShortParts(Vec<(usize, String)>),
    /// The model speaks one batch of takes.
    Generating {
        /// Takes in the batch.
        parts: usize,
        /// No checkpoint is on disk, so the model downloads first.
        download: bool,
    },
    /// A step of the voice's design or reuse.
    Anchor(AnchorStep),
    /// A step of the take loop.
    Take(TakeStep),
    /// Thumps and clicks the kept takes had repaired in place.
    Repaired {
        /// Thumps silenced at take edges.
        thumps: usize,
        /// Clicks smoothed away.
        fixed: usize,
    },
}

impl SpeakStep {
    /// The line the skill prints for this step, when it prints one.
    pub fn message(&self) -> Option<String> {
        match self {
            Self::NoVoice => Some(
                "speak: no --voice given, so VoxCPM2 picks its own voice. Design one with --voice \"(age, pitch, tone, accent)\" (references/tts/voxcpm2.md)."
                    .to_string(),
            ),
            Self::ShortParts(parts) => {
                let list: Vec<String> = parts.iter().map(|(n, t)| format!("{n} \"{t}\"")).collect();
                Some(format!(
                    "speak: short parts fail more often with VoxCPM2: {}. When one fails, join it to the line before or after.",
                    list.join(", ")
                ))
            }
            Self::Generating { parts, download } => Some(format!(
                "speak: VoxCPM2 is speaking {parts} part(s){}, about 3 s per sentence on a GPU.",
                if *download { " after the download" } else { "" }
            )),
            Self::Anchor(step) => step.message(),
            Self::Take(step) => step.message(),
            Self::Repaired { thumps, fixed } => {
                let mut done = Vec::new();
                if *thumps > 0 {
                    done.push(format!("{thumps} thump(s)"));
                }
                if *fixed > 0 {
                    done.push(format!("{fixed} click(s)"));
                }
                (!done.is_empty())
                    .then(|| format!("speak: repaired in place: {}.", done.join(", ")))
            }
        }
    }
}

/// The parts of fewer than three words, numbered from 1, as
/// `text.split(/\s+/)` counts them.
pub(super) fn short_parts(parts: &[Chunk]) -> Vec<(usize, String)> {
    parts
        .iter()
        .enumerate()
        .filter(|(_, p)| js_word_count(&p.text) < SHORT_WORDS)
        .map(|(i, p)| (i + 1, p.text.clone()))
        .collect()
}

/// Appends the design steps to `log`, each call to the model after a
/// [`SpeakStep::Generating`] of one take: a design speaks one take a call.
pub(super) fn push_anchor_steps(log: &mut Vec<SpeakStep>, steps: Vec<AnchorStep>, download: bool) {
    for step in steps {
        if matches!(step, AnchorStep::Speak(_)) {
            log.push(SpeakStep::Generating { parts: 1, download });
        }
        log.push(SpeakStep::Anchor(step));
    }
}

/// Appends the take loop's steps to `log`. Each run of takes spoken in a
/// row is one batch to the model, so a [`SpeakStep::Generating`] of the
/// run's length goes before it.
pub(super) fn push_take_steps(log: &mut Vec<SpeakStep>, steps: Vec<TakeStep>, download: bool) {
    let mut steps = steps.into_iter().peekable();
    while let Some(step) = steps.next() {
        if matches!(step, TakeStep::Speak(_)) {
            let mut batch = vec![step];
            while let Some(next) = steps.next_if(|s| matches!(s, TakeStep::Speak(_))) {
                batch.push(next);
            }
            log.push(SpeakStep::Generating {
                parts: batch.len(),
                download,
            });
            log.extend(batch.into_iter().map(SpeakStep::Take));
        } else {
            log.push(SpeakStep::Take(step));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take::engine::TakeSpec;

    fn spec(seed: f64) -> TakeSpec {
        TakeSpec {
            index: 0,
            text: "Hi.".to_string(),
            seed,
            delivery: None,
        }
    }

    #[test]
    fn each_run_of_takes_spoken_is_one_batch() {
        let mut log = Vec::new();
        let steps = vec![
            TakeStep::Speak(spec(1.0)),
            TakeStep::Speak(spec(2.0)),
            TakeStep::Judge(spec(1.0)),
            TakeStep::Speak(spec(3.0)),
        ];
        push_take_steps(&mut log, steps, false);
        let messages: Vec<String> = log.iter().filter_map(SpeakStep::message).collect();
        assert_eq!(
            messages,
            [
                "speak: VoxCPM2 is speaking 2 part(s), about 3 s per sentence on a GPU.",
                "speak: VoxCPM2 is speaking 1 part(s), about 3 s per sentence on a GPU.",
            ]
        );
        assert_eq!(log.len(), 6);
    }

    #[test]
    fn each_design_call_is_its_own_batch() {
        let mut log = Vec::new();
        let steps = vec![
            AnchorStep::Designing {
                opening: "Hi.".to_string(),
            },
            AnchorStep::Speak(spec(7.0)),
        ];
        push_anchor_steps(&mut log, steps, true);
        assert_eq!(
            log[1],
            SpeakStep::Generating {
                parts: 1,
                download: true
            }
        );
        assert_eq!(
            log[1].message().as_deref(),
            Some(
                "speak: VoxCPM2 is speaking 1 part(s) after the download, about 3 s per sentence on a GPU."
            )
        );
    }

    #[test]
    fn repairs_name_only_what_was_repaired() {
        let msg = |thumps, fixed| SpeakStep::Repaired { thumps, fixed }.message();
        assert_eq!(msg(0, 0), None);
        assert_eq!(
            msg(2, 0).as_deref(),
            Some("speak: repaired in place: 2 thump(s).")
        );
        assert_eq!(
            msg(1, 3).as_deref(),
            Some("speak: repaired in place: 1 thump(s), 3 click(s).")
        );
    }

    #[test]
    fn short_parts_count_words_as_javascript_splits_them() {
        let part = |text: &str| Chunk {
            text: text.to_string(),
            pause: 0.0,
            kind: crate::text::PhraseKind::End,
            sentences: Vec::new(),
            delivery: None,
        };
        let parts = [part("One two three."), part("Yes, sir."), part("Go.")];
        let short = short_parts(&parts);
        assert_eq!(
            short,
            [(2, "Yes, sir.".to_string()), (3, "Go.".to_string())]
        );
        assert_eq!(
            SpeakStep::ShortParts(short).message().as_deref(),
            Some(
                "speak: short parts fail more often with VoxCPM2: 2 \"Yes, sir.\", 3 \"Go.\". When one fails, join it to the line before or after."
            )
        );
    }
}
