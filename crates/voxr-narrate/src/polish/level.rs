//! Sentence levels evened out: each sentence's speech is brought toward the
//! median level of the take's sentences, so a take that starts louder than
//! it goes on sounds even.
//!
//! Every sum runs in `f64` in sample order, and every stored sample is
//! rounded to `f32`, so the result matches the JavaScript bit for bit.

use crate::glitch::heat::percentile;
use crate::js::{js_max, js_min, js_round};
use crate::js_math::{js_log10, js_pow};

/// Frames per second the speech level is read in: 10 ms frames.
const FRAMES_PER_SECOND: f64 = 100.0;
/// A frame with an RMS level at or under this is silence.
const MIN_RMS: f64 = 0.01;
/// The median.
const MEDIAN: f64 = 0.5;
/// The largest change to a sentence's level, in dB.
const MAX_GAIN_DB: f64 = 6.0;
/// A sentence within this many dB of the target is left as it is.
const MIN_GAIN_DB: f64 = 1.0;
/// Length of the gain ramp at each end of a sentence, in seconds.
const RAMP: f64 = 0.04;
/// Sentences with a speech level needed before any is changed.
const MIN_LEVELS: usize = 2;

/// A stretch of a take, in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    /// Start, in seconds.
    pub start: f64,
    /// End, in seconds.
    pub end: f64,
}

/// What [`even_levels`] changed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelReport {
    /// Sentences whose level changed.
    pub leveled: usize,
    /// The largest change, in dB, as a magnitude; 0 when none changed.
    pub max_gain_db: f64,
}

/// Sample `k` of `x`, or NaN where JavaScript reads `undefined`: a negative,
/// fractional or past-the-end index.
fn sample(x: &[f32], k: f64) -> f64 {
    if k >= 0.0 && k.fract() == 0.0 && k < x.len() as f64 {
        f64::from(x[k as usize])
    } else {
        f64::NAN
    }
}

/// The loudness, in dB, of the sounding 10 ms frames of `x` from `start` to
/// `end` seconds; `None` when no frame sounds.
///
/// Frames of `rate / 100` samples start at `floor(start * rate)` and must
/// end by `ceil(end * rate)` and by the end of `x`. A frame sounds when its
/// RMS level is over 0.01. The level is the energy of the sounding frames
/// over their sample count. A rate that is not a multiple of 100 Hz gives a
/// fractional hop: a frame at a fractional sample index reads nothing and
/// never sounds. A rate of 0 Hz or an infinite start reads nothing.
pub fn speech_level(x: &[f32], start: f64, end: f64, rate: u32) -> Option<f64> {
    let rate = f64::from(rate);
    let hop = rate / FRAMES_PER_SECOND;
    let mut i = (start * rate).floor();
    if hop == 0.0 || !i.is_finite() {
        return None;
    }
    let stop = js_min(x.len() as f64, (end * rate).ceil());
    let mut e = 0.0f64;
    let mut n = 0.0f64;
    while i + hop <= stop {
        let mut f = 0.0f64;
        let mut k = i;
        while k < i + hop {
            let v = sample(x, k);
            f += v * v;
            k += 1.0;
        }
        if (f / hop).sqrt() > MIN_RMS {
            e += f;
            n += hop;
        }
        i += hop;
    }
    (n != 0.0).then(|| 10.0 * js_log10(e / n))
}

/// Brings each span's speech toward the median speech level of the spans.
///
/// The target is the median of the spans' [`speech_level`]s, the value at
/// `floor(count / 2)` of them sorted. Nothing changes unless 2 or more
/// spans have a level. A span with no level is left as it is. Each other
/// span's gain is the target minus its level, held within ±6 dB; a gain
/// under 1 dB is left out. The gain applies from `round(start * rate)` to
/// `round(end * rate)` or the end of `x`, rising and falling over 40 ms
/// linear ramps at each end: sample `k` is multiplied by
/// `1 + (lin - 1) * min(1, (k - a) / ramp, (b - k) / ramp)`. Spans are
/// applied in order, each reading the samples the previous one left.
pub fn even_levels(x: &mut [f32], spans: &[Span], rate: u32) -> LevelReport {
    let mut report = LevelReport {
        leveled: 0,
        max_gain_db: 0.0,
    };
    let levels: Vec<Option<f64>> = spans
        .iter()
        .map(|sp| speech_level(x, sp.start, sp.end, rate))
        .collect();
    let known: Vec<f64> = levels.iter().flatten().copied().collect();
    if known.len() < MIN_LEVELS {
        return report;
    }
    let target = percentile(&known, MEDIAN);
    let rate = f64::from(rate);
    let ramp = js_round(RAMP * rate);
    let len = x.len() as f64;
    for (sp, level) in spans.iter().zip(&levels) {
        let Some(level) = level else {
            continue;
        };
        let g = js_max(-MAX_GAIN_DB, js_min(MAX_GAIN_DB, target - level));
        if g.abs() < MIN_GAIN_DB {
            continue;
        }
        report.leveled += 1;
        report.max_gain_db = js_max(report.max_gain_db, g.abs());
        let lin = js_pow(10.0, g / 20.0);
        let a = js_round(sp.start * rate);
        let b = js_min(len, js_round(sp.end * rate));
        // A sample before the start of `x` is never written. A NaN bound
        // writes nothing.
        let mut k = if a < b { js_max(a, 0.0) } else { b };
        while k < b {
            let edge = js_min(1.0, js_min((k - a) / ramp, (b - k) / ramp));
            let s = &mut x[k as usize];
            *s = (f64::from(*s) * (1.0 + (lin - 1.0) * edge)) as f32;
            k += 1.0;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `amp` held over `len` samples.
    fn flat(len: usize, amp: f32) -> Vec<f32> {
        vec![amp; len]
    }

    #[test]
    fn level_reads_the_sounding_frames() {
        let mut x = flat(4800, 0.1);
        x.extend(flat(4800, 0.0));
        // 0.1 held: about -20 dB over the ten sounding frames only.
        assert_eq!(speech_level(&x, 0.0, 0.2, 48000), Some(-19.999999870570146));
        assert_eq!(speech_level(&x, 0.1, 0.2, 48000), None);
        assert_eq!(speech_level(&x, 0.0, 0.2, 0), None);
        assert_eq!(speech_level(&x, f64::NEG_INFINITY, 0.2, 48000), None);
        assert_eq!(speech_level(&x, f64::NAN, 0.2, 48000), None);
    }

    #[test]
    fn a_fractional_hop_reads_every_other_frame() {
        // At 22050 Hz a hop is 220.5 samples: 5 of the 10 frames start at a
        // whole index and read 221 samples each, counted as 220.5.
        let x = flat(2205, 0.1);
        assert_eq!(speech_level(&x, 0.0, 0.1, 22050), Some(-19.990163071757642));
    }

    #[test]
    fn fewer_than_two_levels_change_nothing() {
        let mut x = flat(9600, 0.0);
        x[..4800].fill(0.1);
        let before = x.clone();
        let spans = [
            Span {
                start: 0.0,
                end: 0.1,
            },
            Span {
                start: 0.1,
                end: 0.2,
            },
        ];
        let report = even_levels(&mut x, &spans, 48000);
        assert_eq!(report.leveled, 0);
        assert_eq!(x, before);
    }

    #[test]
    fn gains_are_held_within_six_db_with_ramps() {
        let mut x = flat(4800, 0.4);
        x.extend(flat(4800, 0.1));
        x.extend(flat(4800, 0.05));
        let spans = [
            Span {
                start: 0.0,
                end: 0.1,
            },
            Span {
                start: 0.1,
                end: 0.2,
            },
            Span {
                start: 0.2,
                end: 0.3,
            },
        ];
        let report = even_levels(&mut x, &spans, 48000);
        // 0.4 is 12 dB over the 0.1 median and 0.05 is 6.02 dB under it:
        // both are held to 6 dB, the median span is left as it is.
        assert_eq!(report.leveled, 2);
        assert_eq!(report.max_gain_db, 6.0);
        // The first sample sits at the ramp's start: no gain.
        assert_eq!(x[0], 0.4);
        // Past the 1920-sample ramp the full -6 dB applies.
        assert_eq!(f64::from(x[2400]), 0.20047490298748016);
        assert_eq!(x[7200], 0.1);
    }
}
