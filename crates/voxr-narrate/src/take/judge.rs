//! The production judge of a take: its edges tidied, its clicks repaired,
//! then heard back and checked against its text, with a second recognizer
//! consulted on a loop and on doubtful words.

use crate::error::Result;
use crate::glitch::{repair_clicks, tidy};
use crate::text::sayable;
use crate::transcript::Transcribed;
use crate::wav::pcm16_round_trip;

use super::engine::{Graded, Take, TakeJudge, TakeSpec};
use super::hear::{HeardCheck, Hearing, Recognizer, heard_check};

/// The badness a stray pause adds to its take: as much as one extra word,
/// so a retry or a re-roll keeps the take with fewer.
pub const BAD_STRAY_PAUSE: f64 = 0.3;

/// A take judged by [`HeardJudge`].
#[derive(Debug, Clone, PartialEq)]
pub struct JudgedTake {
    /// The take heard back and checked. Each stray pause adds
    /// [`BAD_STRAY_PAUSE`] to its badness. A fault the edge repair cannot
    /// fix fails the check and adds one badness each.
    pub heard: HeardCheck,
    /// The take as used: edges tidied, clicks repaired, and read back from
    /// the 16-bit file the skill keeps it in.
    pub take: Take,
    /// The seed that made it.
    pub seed: f64,
    /// Thumps silenced at its edges.
    pub thumps: usize,
    /// Clicks smoothed away.
    pub fixed: usize,
    /// Clicks left after every repair, in seconds, marked for a listen.
    pub clicks: Vec<f64>,
}

impl Graded for JudgedTake {
    fn ok(&self) -> bool {
        self.heard.check.ok
    }

    fn badness(&self) -> Option<f64> {
        self.heard.check.badness
    }

    fn why(&self) -> &str {
        &self.heard.check.why
    }

    fn heard(&self) -> &str {
        &self.heard.heard
    }
}

/// The skill's judge: a take repaired, then heard. A repair is faster than
/// a new take, so only what cannot be repaired fails it: a wrong word, or
/// a cut-off end.
pub struct HeardJudge<H> {
    /// The recognizers.
    pub hearing: H,
    /// The recognizer heard first, from config or [`asr_model`](super::hear::asr_model).
    pub main: Recognizer,
    /// The language spoken.
    pub language: Option<String>,
}

impl<H: Hearing> TakeJudge for HeardJudge<H> {
    type Verdict = JudgedTake;

    /// The take's edges tidied ([`tidy`]), its clicks repaired
    /// ([`repair_clicks`]), rounded to 16 bits as its tidied file holds it
    /// ([`pcm16_round_trip`]), and heard against the part's [`sayable`]
    /// text ([`heard_check`]). Each stray pause adds [`BAD_STRAY_PAUSE`] to
    /// the badness. Each tidy fault then fails the take, joins its reason
    /// and adds one badness.
    fn judge(&mut self, spec: &TakeSpec, take: &Take) -> Result<JudgedTake> {
        let t = tidy(&take.samples, take.rate);
        let pops = repair_clicks(&t.samples, take.rate);
        let used = pcm16_round_trip(&pops.samples);
        let text = sayable(&spec.text)?;
        let mut heard = heard_check(
            &mut self.hearing,
            self.main,
            &used,
            take.rate,
            &text,
            self.language.as_deref(),
        )?;
        if !heard.pauses.is_empty() {
            let c = &mut heard.check;
            c.badness =
                Some(c.badness.unwrap_or(0.0) + BAD_STRAY_PAUSE * heard.pauses.len() as f64);
        }
        if !t.faults.is_empty() {
            let c = &mut heard.check;
            c.ok = false;
            let why: Vec<&str> = std::iter::once(c.why.as_str())
                .chain(t.faults.iter().map(String::as_str))
                .filter(|w| !w.is_empty())
                .collect();
            c.why = why.join(", ");
            c.badness = Some(c.badness.unwrap_or(0.0) + t.faults.len() as f64);
        }
        Ok(JudgedTake {
            heard,
            take: Take {
                samples: used,
                rate: take.rate,
            },
            seed: spec.seed,
            thumps: t.repairs.len(),
            fixed: pops.fixed.len(),
            clicks: pops.left,
        })
    }
}

/// A judge that also hears a user's recording, for the transcript Hi-Fi
/// cloning clones with.
pub trait ReferenceHearing {
    /// The words the run's main recognizer hears in `samples`, at `rate`
    /// samples per second.
    fn hear_reference(&mut self, samples: &[f32], rate: u32) -> Result<Transcribed>;
}

impl<H: Hearing> ReferenceHearing for HeardJudge<H> {
    /// Hears through [`HeardJudge::main`], in [`HeardJudge::language`].
    fn hear_reference(&mut self, samples: &[f32], rate: u32) -> Result<Transcribed> {
        self.hearing
            .hear(samples, rate, self.main, self.language.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::super::hear::tests::{ScriptedHearing, hearing};
    use super::*;
    use crate::check::HeardWord;

    #[test]
    fn a_cut_off_take_fails_and_adds_badness() -> Result<()> {
        // A steady tone to the last sample: the take stops mid-sound.
        let samples: Vec<f32> = (0..24000).map(|i| 0.5 * (i as f32 * 0.05).sin()).collect();
        let mut judge = HeardJudge {
            hearing: hearing(&[("hello there", false)], &[]),
            main: Recognizer::Default,
            language: None,
        };
        let spec = TakeSpec {
            index: 0,
            text: "Hello there.".to_string(),
            seed: 7.0,
            delivery: None,
        };
        let j = judge.judge(
            &spec,
            &Take {
                samples,
                rate: 48000,
            },
        )?;
        assert!(!j.ok());
        assert_eq!(j.why(), crate::glitch::CUT_OFF);
        assert_eq!(j.badness(), Some(1.0));
        assert_eq!(j.seed, 7.0);
        Ok(())
    }

    #[test]
    fn a_stray_pause_adds_badness_before_the_faults() -> Result<()> {
        // "Hello" then "there" after 0.6 s: a pause the script does not ask for.
        let word = |text: &str, start: f64| HeardWord {
            text: text.to_string(),
            start,
            end: start + 0.4,
        };
        let mut judge = HeardJudge {
            hearing: ScriptedHearing {
                default: vec![Transcribed {
                    words: vec![word("hello", 0.0), word("there", 1.0)],
                    looped: false,
                    duration: 5.0,
                }],
                small: Vec::new(),
                configured: Vec::new(),
                calls: Vec::new(),
            },
            main: Recognizer::Default,
            language: None,
        };
        let spec = TakeSpec {
            index: 0,
            text: "Hello there.".to_string(),
            seed: 7.0,
            delivery: None,
        };
        // A steady tone to the last sample: the take stops mid-sound.
        let samples: Vec<f32> = (0..24000).map(|i| 0.5 * (i as f32 * 0.05).sin()).collect();
        let j = judge.judge(
            &spec,
            &Take {
                samples,
                rate: 48000,
            },
        )?;
        assert_eq!(j.heard.pauses.len(), 1);
        assert!(!j.ok());
        assert_eq!(j.badness(), Some(BAD_STRAY_PAUSE + 1.0));
        Ok(())
    }

    #[test]
    fn a_reference_is_heard_by_the_main_recognizer() -> Result<()> {
        let mut judge = HeardJudge {
            hearing: hearing(&[], &[("hello there", false)]),
            main: Recognizer::Small,
            language: Some("ms".to_string()),
        };
        let heard = judge.hear_reference(&[0.0; 8], 48000)?;
        let words: Vec<&str> = heard.words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(words, ["hello", "there"]);
        assert_eq!(judge.hearing.calls, [Recognizer::Small]);
        Ok(())
    }
}
