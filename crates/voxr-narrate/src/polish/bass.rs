//! A voice's bass against its presence, and a thin voice's bass lifted to
//! level with it by a low shelf.
//!
//! The band levels and the shelf run through the crate's biquad filter, so
//! a level or a lifted sample can differ from the JavaScript in the last
//! bits.

use super::filter::{Kind, filter};
use crate::js::js_min;
use crate::js_math::js_log10;

/// The bass band, in Hz.
const BASS_LO: f64 = 120.0;
const BASS_HI: f64 = 250.0;
/// The presence band, in Hz.
const PRESENCE_LO: f64 = 1000.0;
const PRESENCE_HI: f64 = 2000.0;
/// Added to a band's mean power before its level is taken.
const TINY: f64 = 1e-12;
/// The shelf's corner, in Hz.
const SHELF_HZ: f64 = 250.0;
/// Shelves at most, each after a fresh measure.
const MAX_PASSES: usize = 2;
/// A shortfall of this many dB or less is left as it is.
const MIN_SHORT_DB: f64 = 0.5;
/// The most the shelves lift in all, in dB.
const MAX_GAIN_DB: f64 = 9.0;
/// A shelf's gain per dB of shortfall: a shelf lifts the bass band by less
/// than its gain.
const STEP_PER_SHORT: f64 = 1.4;

/// What [`lift_bass`] measured and did.
#[derive(Debug, Clone, PartialEq)]
pub struct BassReport {
    /// The total shelf gain, in dB: 0 when no shelf ran. The JavaScript
    /// returns this.
    pub gain: f64,
    /// Each shortfall measured, in dB, in order: the bass band's level
    /// under the presence band's, the negated [`bass_balance`].
    pub shorts: Vec<f64>,
    /// Each shelf's gain, in dB, in order.
    pub steps: Vec<f64>,
}

/// The mean level, in dB, of `x` through two high-pass stages at `lo` Hz
/// and two low-pass stages at `hi` Hz: `10 * log10(power + 1e-12)`, the
/// power taken over at least 1 sample.
fn band_level(x: &[f32], lo: f64, hi: f64, rate: u32) -> f64 {
    let mut y = filter(x, Kind::HighPass, lo, rate);
    y = filter(&y, Kind::HighPass, lo, rate);
    y = filter(&y, Kind::LowPass, hi, rate);
    y = filter(&y, Kind::LowPass, hi, rate);
    let mut e = 0.0f64;
    for &v in &y {
        e += f64::from(v) * f64::from(v);
    }
    10.0 * js_log10(e / y.len().max(1) as f64 + TINY)
}

/// The 120-250 Hz band's level less the 1-2 kHz band's, in dB, of `x`
/// sampled at `rate` Hz. A full voice holds about +1 to +9 dB, a thin one,
/// hard on the ear, under 0.
pub fn bass_balance(x: &[f32], rate: u32) -> f64 {
    band_level(x, BASS_LO, BASS_HI, rate) - band_level(x, PRESENCE_LO, PRESENCE_HI, rate)
}

/// Lifts a thin voice's bass toward its presence, `x` sampled at `rate` Hz.
///
/// Up to 2 passes each measure the shortfall, `-bass_balance`. A pass stops
/// the loop when the shortfall is 0.5 dB or less or the gain so far has
/// reached 9 dB. Otherwise it runs a low shelf at 250 Hz with gain
/// `min(9 - gain, 1.4 * shortfall)` dB over the result so far. A voice
/// with bass at or above its presence passes unchanged.
pub fn lift_bass(x: &[f32], rate: u32) -> (Vec<f32>, BassReport) {
    let mut report = BassReport {
        gain: 0.0,
        shorts: Vec::new(),
        steps: Vec::new(),
    };
    let mut y = x.to_vec();
    for _ in 0..MAX_PASSES {
        let short = -bass_balance(&y, rate);
        report.shorts.push(short);
        if short <= MIN_SHORT_DB || report.gain >= MAX_GAIN_DB {
            break;
        }
        let step = js_min(MAX_GAIN_DB - report.gain, short * STEP_PER_SHORT);
        y = filter(&y, Kind::LowShelf(step), SHELF_HZ, rate);
        report.gain += step;
        report.steps.push(step);
    }
    (y, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A 240 Hz tone of amplitude `bass` plus a 1.5 kHz tone of amplitude
    /// `presence`, 0.25 s at 48 kHz.
    fn voice(bass: f64, presence: f64) -> Vec<f32> {
        (0..12000)
            .map(|i| {
                let t = i as f64 / 48000.0;
                (bass * (2.0 * PI * 240.0 * t).sin() + presence * (2.0 * PI * 1500.0 * t).sin())
                    as f32
            })
            .collect()
    }

    #[test]
    fn a_full_voice_is_left_as_it_is() {
        let x = voice(0.3, 0.1);
        assert!(bass_balance(&x, 48000) > 5.0);
        let (y, report) = lift_bass(&x, 48000);
        assert_eq!(y, x);
        assert_eq!(report.gain, 0.0);
        assert_eq!(report.shorts.len(), 1);
        assert!(report.steps.is_empty());
    }

    #[test]
    fn a_thin_voice_is_lifted_in_two_passes() {
        let x = voice(0.16, 0.2);
        let (y, report) = lift_bass(&x, 48000);
        assert_eq!(report.steps.len(), 2);
        assert_eq!(report.steps[0], report.shorts[0] * 1.4);
        assert!(report.gain > 5.0 && report.gain < 9.0);
        assert!(bass_balance(&y, 48000) > bass_balance(&x, 48000) + 3.0);
    }

    #[test]
    fn a_very_thin_voice_is_held_at_9_db() {
        let (_, report) = lift_bass(&voice(0.03, 0.3), 48000);
        assert_eq!(report.steps, vec![9.0]);
        assert_eq!(report.gain, 9.0);
        // The second measure finds the cap reached.
        assert_eq!(report.shorts.len(), 2);
    }

    #[test]
    fn silence_is_balanced() {
        assert_eq!(bass_balance(&[], 48000), 0.0);
        let (y, report) = lift_bass(&[0.0; 480], 48000);
        assert_eq!(y, vec![0.0; 480]);
        assert_eq!(report.gain, 0.0);
    }
}
