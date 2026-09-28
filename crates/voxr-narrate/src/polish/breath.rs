//! Breath in a voice: runs of 80 ms or more that are quiet, 20 to 45 dB
//! under the voice's loud level, and unpitched. A clone learns its manner
//! from its reference, so a reference's breath runs are turned down.
//!
//! Every sum runs in `f64` in the JavaScript's order and each output sample
//! is rounded to `f32`, so the result matches the JavaScript bit for bit.

use std::ops::Range;

use crate::glitch::heat::percentile;
use crate::js::{js_max, js_min, js_round};
use crate::js_math::{js_log10, js_pow};

/// Frames per second: 10 ms frames.
const FRAMES_PER_SECOND: f64 = 100.0;
/// The shortest and longest pitch periods tried, as rates: 2 to 10 ms, a
/// voice from 100 to 500 Hz.
const HIGHEST_PITCH_HZ: f64 = 500.0;
const LOWEST_PITCH_HZ: f64 = 100.0;
/// Samples between the pitch periods tried.
const LAG_STEP: usize = 4;
/// A frame whose best normalized autocorrelation reaches this is pitched.
const PITCHED: f64 = 0.5;
/// Added to energies before a level or a square root, so silence has one.
const TINY: f64 = 1e-12;
/// The percentile of the frame levels taken as the voice's loud level.
const LOUD: f64 = 0.95;
/// A breath frame is under the loud level by more than this, in dB.
const BREATH_TOP_DB: f64 = 20.0;
/// A breath frame is under the loud level by less than this, in dB.
const BREATH_FLOOR_DB: f64 = 45.0;
/// Breath frames a run needs: 80 ms.
const MIN_RUN: usize = 8;
/// The cut on a breath run, in dB.
const CUT_DB: f64 = -20.0;
/// Length of the cut's ramp at each end of a run, in seconds.
const RAMP: f64 = 0.01;

/// What [`quiet_breaths`] turned down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BreathReport {
    /// The breath runs, as sample ranges, in order. Their count is what the
    /// JavaScript returns.
    pub runs: Vec<Range<usize>>,
}

/// One 10 ms frame: its level in dB and whether it is pitched.
struct Frame {
    level: f64,
    pitched: bool,
}

/// The frame of `hop` samples from `k`: its level, and whether its best
/// normalized autocorrelation over the pitch periods tried reaches 0.5.
fn frame(x: &[f32], k: usize, hop: usize, lags: (usize, usize)) -> Frame {
    let mut e = 0.0f64;
    for &v in &x[k..k + hop] {
        e += f64::from(v) * f64::from(v);
    }
    let mut best = 0.0f64;
    let mut lag = lags.0;
    while lag <= lags.1 && k + hop + lag <= x.len() {
        let mut c = 0.0f64;
        let mut e2 = 0.0f64;
        for i in k..k + hop {
            let (a, b) = (f64::from(x[i]), f64::from(x[i + lag]));
            c += a * b;
            e2 += b * b;
        }
        best = js_max(best, c / (e * e2 + TINY).sqrt());
        lag += LAG_STEP;
    }
    Frame {
        level: 10.0 * js_log10(e / hop as f64 + TINY),
        pitched: best >= PITCHED,
    }
}

/// The breath runs of `x`, sampled at `rate` Hz, as sample ranges.
///
/// Frames of `round(rate / 100)` samples run while a whole frame fits. A
/// frame is pitched when its best autocorrelation, normalized by both
/// windows' energy, reaches 0.5 over lags from `round(rate / 500)` to
/// `round(rate / 100)` in steps of 4 samples; a lag whose window runs past
/// the end is not tried. A frame's level is `10 * log10(energy / hop +
/// 1e-12)` dB. The loud level is the frame level at `floor(frames * 0.95)`
/// sorted ascending. A breath frame is unpitched and more than 20 but less
/// than 45 dB under the loud level. A run of 8 breath frames or more is a
/// breath run. A rate under 50 Hz makes a frame of 0 samples and gives no
/// run.
pub fn breath_runs(x: &[f32], rate: u32) -> Vec<Range<usize>> {
    let rate = f64::from(rate);
    let hop = js_round(rate / FRAMES_PER_SECOND) as usize;
    if hop == 0 {
        return Vec::new();
    }
    let lags = (
        js_round(rate / HIGHEST_PITCH_HZ) as usize,
        js_round(rate / LOWEST_PITCH_HZ) as usize,
    );
    let frames: Vec<Frame> = (0..)
        .map(|f| f * hop)
        .take_while(|&k| k + hop <= x.len())
        .map(|k| frame(x, k, hop, lags))
        .collect();
    if frames.is_empty() {
        return Vec::new();
    }
    let levels: Vec<f64> = frames.iter().map(|f| f.level).collect();
    let loud = percentile(&levels, LOUD);
    let mut runs = Vec::new();
    let mut run = 0usize;
    // One frame past the last ends a run still open.
    for i in 0..=frames.len() {
        let breath = frames.get(i).is_some_and(|f| {
            f.level < loud - BREATH_TOP_DB && f.level > loud - BREATH_FLOOR_DB && !f.pitched
        });
        if breath {
            run += 1;
        } else {
            if run >= MIN_RUN {
                runs.push((i - run) * hop..i * hop);
            }
            run = 0;
        }
    }
    runs
}

/// Turns the breath runs of `x`, sampled at `rate` Hz, down by 20 dB and
/// leaves the rest as it is.
///
/// The runs are [`breath_runs`]. The cut ramps linearly over
/// `round(0.01 * rate)` samples at each end of a run: sample `k` of a run
/// from `a` to `b` is multiplied by
/// `1 - (1 - 0.1) * min(1, (k - a) / ramp, (b - k) / ramp)`.
pub fn quiet_breaths(x: &[f32], rate: u32) -> (Vec<f32>, BreathReport) {
    let runs = breath_runs(x, rate);
    let mut y = x.to_vec();
    let low = js_pow(10.0, CUT_DB / 20.0);
    let ramp = js_round(RAMP * f64::from(rate));
    for run in &runs {
        let (a, b) = (run.start as f64, run.end as f64);
        for k in run.clone() {
            let kf = k as f64;
            let edge = js_min(1.0, js_min((kf - a) / ramp, (b - kf) / ramp));
            y[k] = (f64::from(y[k]) * (1.0 - (1.0 - low) * edge)) as f32;
        }
    }
    (y, BreathReport { runs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A 200 Hz tone of amplitude 0.3 over `len` samples at 48 kHz.
    fn voice(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (0.3 * (2.0 * PI * 200.0 * i as f64 / 48000.0).sin()) as f32)
            .collect()
    }

    /// Uniform noise in `[-amp, amp)` over `len` samples, from xorshift32.
    fn breath(len: usize, amp: f64) -> Vec<f32> {
        let mut s: u32 = 7;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (amp * (f64::from(s) / 2147483648.0 - 1.0)) as f32
            })
            .collect()
    }

    /// A voice, a breath of `frames` 10 ms frames, then the voice again.
    fn take(frames: usize) -> Vec<f32> {
        let mut x = voice(14400);
        x.extend(breath(frames * 480, 0.009));
        x.extend(voice(14400));
        x
    }

    #[test]
    fn a_breath_between_phrases_is_turned_down_with_ramps() {
        let x = take(15);
        let (y, report) = quiet_breaths(&x, 48000);
        assert_eq!(report.runs, vec![14400..21600]);
        let low = js_pow(10.0, -1.0);
        // The ramp's first sample is untouched, the middle is cut in full.
        assert_eq!(y[14400], x[14400]);
        let mid = 18000;
        assert_eq!(y[mid], (f64::from(x[mid]) * (1.0 - (1.0 - low))) as f32);
        // Halfway down the 480-sample ramp: half the cut.
        let half = 14640;
        assert_eq!(
            y[half],
            (f64::from(x[half]) * (1.0 - (1.0 - low) * 0.5)) as f32
        );
        assert_eq!(y[..14400], x[..14400]);
        assert_eq!(y[21600..], x[21600..]);
    }

    #[test]
    fn a_breath_under_8_frames_is_kept() {
        let x = take(5);
        let (y, report) = quiet_breaths(&x, 48000);
        assert!(report.runs.is_empty());
        assert_eq!(y, x);
    }

    #[test]
    fn a_voice_alone_holds_no_breath() {
        assert!(breath_runs(&voice(48000), 48000).is_empty());
    }

    #[test]
    fn short_empty_and_slow_inputs_hold_no_breath() {
        assert!(breath_runs(&[], 48000).is_empty());
        assert!(breath_runs(&[0.1; 479], 48000).is_empty());
        // Under 50 Hz a frame holds no sample.
        assert!(breath_runs(&[0.1; 100], 40).is_empty());
        let (y, report) = quiet_breaths(&[0.1; 100], 40);
        assert_eq!(y, vec![0.1; 100]);
        assert!(report.runs.is_empty());
    }
}
