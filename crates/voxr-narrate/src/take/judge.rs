//! The production judge of a take: its edges tidied, its clicks repaired,
//! then heard back and checked against its text, with a second recognizer
//! consulted on a loop and on doubtful words.

use crate::error::Result;
use crate::glitch::{repair_clicks, tidy};
use crate::text::sayable;

use super::engine::{Graded, Take, TakeJudge, TakeSpec};
use super::hear::{HeardCheck, Hearing, Recognizer, heard_check};

/// A take judged by [`HeardJudge`].
#[derive(Debug, Clone, PartialEq)]
pub struct JudgedTake {
    /// The take heard back and checked. A fault the edge repair cannot fix
    /// fails the check and adds one badness each.
    pub heard: HeardCheck,
    /// The take as used: edges tidied and clicks repaired.
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
    /// ([`repair_clicks`]), and the result heard against the part's
    /// [`sayable`] text ([`heard_check`]). Each tidy fault fails the take,
    /// joins its reason and adds one badness.
    fn judge(&mut self, spec: &TakeSpec, take: &Take) -> Result<JudgedTake> {
        let t = tidy(&take.samples, take.rate);
        let pops = repair_clicks(&t.samples, take.rate);
        let text = sayable(&spec.text)?;
        let mut heard = heard_check(
            &mut self.hearing,
            self.main,
            &pops.samples,
            take.rate,
            &text,
            self.language.as_deref(),
        )?;
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
                samples: pops.samples,
                rate: take.rate,
            },
            seed: spec.seed,
            thumps: t.repairs.len(),
            fixed: pops.fixed.len(),
            clicks: pops.left,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::hear::tests::hearing;
    use super::*;

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
}
