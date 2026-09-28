//! A de-esser for generated speech: an "s" louder than the voice's own
//! vowels sounds harsh, and a generated voice often makes it so. In each
//! frame where the 4.5-11 kHz band rises over 4 dB under the vowel level,
//! that band is turned down to that limit, by at most 15 dB, and eases back
//! over about 40 ms. Everything else passes untouched.
//!
//! Every sum runs in `f64` in the JavaScript's order and the output is
//! stored as `f32`, so the result matches the JavaScript bit for bit.

use super::fft::fft;
use crate::js::{js_max, js_min, js_round};
use crate::js_math::{js_log10, js_pow};
use std::f64::consts::PI;

/// Samples in one frame: about 21 ms at 48 kHz.
const FRAME: usize = 1024;
/// Samples between frames: half overlap.
const HOP: usize = FRAME / 2;
/// The vowel band, in Hz.
const VOWEL_LO: f64 = 100.0;
const VOWEL_HI: f64 = 3000.0;
/// The sibilant band, in Hz.
const SIBILANT_LO: f64 = 4500.0;
const SIBILANT_HI: f64 = 11000.0;
/// Where the cut starts easing in, in Hz: full from [`SIBILANT_LO`] up.
const EDGE: f64 = 3500.0;
/// The sibilant limit, in dB of power under the vowel level.
const LIMIT_DB: f64 = -4.0;
/// The deepest cut, in dB.
const MAX_CUT_DB: f64 = -15.0;
/// How fast a cut eases back, in dB per frame.
const RELEASE_DB: f64 = 1.5;
/// A frame cut deeper than this counts toward the share, in dB.
const COUNTED_DB: f64 = -0.5;
/// A frame cut deeper than this has its spectrum changed, in dB.
const APPLIED_DB: f64 = -0.05;

/// What [`de_ess`] did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeEssStats {
    /// The share of frames cut by more than 0.5 dB; 0 with no frame.
    pub share: f64,
    /// The deepest cut, in dB: 0 or below.
    pub deepest: f64,
}

/// The sqrt-Hann window, stored as `f32`: on analysis and synthesis at half
/// overlap it rebuilds the signal exactly where no gain applies.
fn window() -> Vec<f32> {
    (0..FRAME)
        .map(|i| (0.5 - 0.5 * ((2.0 * PI * i as f64) / FRAME as f64).cos()).sqrt() as f32)
        .collect()
}

/// The windowed frame `f` of `x`, transformed; samples past the end read 0.
fn spectrum(x: &[f32], win: &[f32], f: usize, re: &mut [f64], im: &mut [f64]) {
    for (i, (r, w)) in re.iter_mut().zip(win).enumerate() {
        let v = x.get(f * HOP + i).map_or(0.0, |&s| f64::from(s));
        *r = v * f64::from(*w);
    }
    im.fill(0.0);
    fft(re, im, false);
}

/// The power of bins `lo` to `hi - 1`, summed in bin order. A bin past the
/// frame reads `undefined` in JavaScript, which makes the sum NaN.
fn band(re: &[f64], im: &[f64], lo: usize, hi: usize) -> f64 {
    let mut s = 0.0f64;
    for k in lo..hi.min(re.len()) {
        s += re[k] * re[k] + im[k] * im[k];
    }
    if lo < hi && hi > re.len() {
        f64::NAN
    } else {
        s
    }
}

/// De-esses `x`, sampled at `rate` Hz.
///
/// Frames of 1024 samples, half overlapping, are windowed with a sqrt-Hann
/// window and transformed; the last frame is padded with zeros. A band runs
/// from bin `round(lo * 1024 / rate)` up to, not including,
/// `round(hi * 1024 / rate)`. The vowel level is the median of the upper
/// half of the frames' nonzero 100-3000 Hz powers, sorted ascending. With
/// no vowel power at all, `x` passes unchanged with zero stats.
///
/// A frame whose 4.5-11 kHz power is over the vowel level less 4 dB wants
/// the cut that brings it to that limit, held to 15 dB. The cut follows a
/// deeper want at once and eases back by 1.5 dB a frame. A frame cut by
/// more than 0.05 dB scales bins from 3.5 kHz up to half the frame and
/// their mirrors: fully from 4.5 kHz, linearly less below. Each frame is
/// transformed back, windowed again and added in place. A rate of 0 Hz
/// leaves `x` unchanged.
pub fn de_ess(x: &[f32], rate: u32) -> (Vec<f32>, DeEssStats) {
    let unchanged = (
        x.to_vec(),
        DeEssStats {
            share: 0.0,
            deepest: 0.0,
        },
    );
    if rate == 0 {
        return unchanged;
    }
    let rate = f64::from(rate);
    let bin = |hz: f64| js_round((hz * FRAME as f64) / rate) as usize;
    let (v0, v1, s0, s1, edge) = (
        bin(VOWEL_LO),
        bin(VOWEL_HI),
        bin(SIBILANT_LO),
        bin(SIBILANT_HI),
        bin(EDGE),
    );
    let frames = js_max(
        0.0,
        ((x.len() as f64 - FRAME as f64) / HOP as f64).ceil() + 1.0,
    ) as usize;
    let win = window();
    let mut re = vec![0.0f64; FRAME];
    let mut im = vec![0.0f64; FRAME];
    let mut vowel = Vec::with_capacity(frames);
    let mut sib = Vec::with_capacity(frames);
    for f in 0..frames {
        spectrum(x, &win, f, &mut re, &mut im);
        vowel.push(band(&re, &im, v0, v1));
        sib.push(band(&re, &im, s0, s1));
    }
    let mut loud: Vec<f64> = vowel.iter().copied().filter(|&v| v > 0.0).collect();
    loud.sort_by(f64::total_cmp);
    let upper = &loud[loud.len() / 2..];
    let reference = upper.get(upper.len() / 2).copied().unwrap_or(0.0);
    if reference == 0.0 {
        return unchanged;
    }
    let limit = reference * js_pow(10.0, LIMIT_DB / 10.0);
    let mut out = vec![0.0f32; x.len()];
    let mut g = 0.0f64;
    let mut cut = 0usize;
    let mut deepest = 0.0f64;
    let half = FRAME / 2;
    for (f, &sib) in sib.iter().enumerate() {
        let want = if sib > limit {
            js_max(MAX_CUT_DB, 10.0 * js_log10(limit / sib))
        } else {
            0.0
        };
        g = if want < g {
            want
        } else {
            js_min(want, g + RELEASE_DB)
        };
        // The spectrum is rebuilt: the same frame gives the same doubles.
        spectrum(x, &win, f, &mut re, &mut im);
        if g < COUNTED_DB {
            cut += 1;
        }
        deepest = js_min(deepest, g);
        if g < APPLIED_DB {
            let lin = js_pow(10.0, g / 20.0);
            for k in edge..=half {
                // A soft edge from 3.5 to 4.5 kHz, the full cut above.
                let m = if k >= s0 {
                    lin
                } else {
                    1.0 + (lin - 1.0) * ((k - edge) as f64 / (s0 - edge) as f64)
                };
                re[k] *= m;
                im[k] *= m;
                if k > 0 && k < half {
                    re[FRAME - k] *= m;
                    im[FRAME - k] *= m;
                }
            }
        }
        fft(&mut re, &mut im, true);
        for (i, (r, w)) in re.iter().zip(&win).enumerate() {
            if let Some(o) = out.get_mut(f * HOP + i) {
                *o = (f64::from(*o) + r * f64::from(*w)) as f32;
            }
        }
    }
    let share = if frames == 0 {
        0.0
    } else {
        cut as f64 / frames as f64
    };
    (out, DeEssStats { share, deepest })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a over each value's `f32` bits, low byte first.
    fn fnv1a(x: &[f32]) -> u32 {
        let mut h: u32 = 0x811c9dc5;
        for byte in x.iter().flat_map(|s| s.to_bits().to_le_bytes()) {
            h ^= u32::from(byte);
            h = h.wrapping_mul(0x01000193);
        }
        h
    }

    #[test]
    fn window_matches_javascript() {
        // The JavaScript window's hash; its `Math.cos` differs from the
        // platform `cos` on some angles, and the `f32` rounding hides it.
        assert_eq!(fnv1a(&window()), 0x3b275f68);
    }

    #[test]
    fn silence_passes_unchanged() {
        let (y, stats) = de_ess(&[0.0; 3000], 48000);
        assert_eq!(y, vec![0.0; 3000]);
        assert_eq!(
            stats,
            DeEssStats {
                share: 0.0,
                deepest: 0.0
            }
        );
        let (y, stats) = de_ess(&[], 48000);
        assert!(y.is_empty());
        assert_eq!(stats.share, 0.0);
    }

    #[test]
    fn a_bin_past_the_frame_makes_the_band_nan() {
        let re = [1.0; 4];
        let im = [0.0; 4];
        assert_eq!(band(&re, &im, 1, 3), 2.0);
        assert!(band(&re, &im, 2, 6).is_nan());
        assert_eq!(band(&re, &im, 6, 6), 0.0);
    }

    #[test]
    fn a_dull_tone_is_rebuilt_without_a_cut() {
        let x: Vec<f32> = (0..8192)
            .map(|i| (0.3 * (2.0 * PI * 200.0 * i as f64 / 48000.0).sin()) as f32)
            .collect();
        let (y, stats) = de_ess(&x, 48000);
        assert_eq!(
            stats,
            DeEssStats {
                share: 0.0,
                deepest: 0.0
            }
        );
        // Away from the first and last half frame, the windows sum to 1.
        for i in 512..7680 {
            assert!((y[i] - x[i]).abs() < 1e-6, "sample {i}");
        }
    }
}
