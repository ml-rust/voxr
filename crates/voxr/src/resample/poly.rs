//! `scipy.signal.resample_poly(x, up, down)` at its defaults: a Kaiser
//! (β 5) windowed-sinc low-pass of `20 · max(up, down) + 1` taps, run as a
//! polyphase filter with zeros past both ends.
//!
//! Provenance: a port of SciPy's `resample_poly`, `firwin` and `upfirdn`
//! (BSD-3-Clause). The filter is designed in `f64`, then cast to the
//! sample type, as SciPy casts it to the dtype of `x`. Each output sums its
//! taps in ascending input order from zero, as `upfirdn` does, in the
//! sample type.

use std::f64::consts::PI;
use std::ops::{Add, Mul};

use super::math::{bessel_i0, gcd};
use crate::error::{Error, Result};

/// The Kaiser β of SciPy's default `window=('kaiser', 5.0)`.
const KAISER_BETA: f64 = 5.0;

/// Taps either side of the centre, per unit of `max(up, down)`.
const HALF_LEN_PER_RATE: usize = 10;

/// A sample type [`resample_poly`] runs in: `f32` or `f64`.
pub trait PolySample: Copy + Add<Output = Self> + Mul<Output = Self> {
    /// Zero.
    const ZERO: Self;
    /// `v` rounded to this type.
    fn from_f64(v: f64) -> Self;
}

impl PolySample for f32 {
    const ZERO: Self = 0.0;
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl PolySample for f64 {
    const ZERO: Self = 0.0;
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// `np.sinc`: `sin(πx) / (πx)`, with `x = 0` read as `1e-20`.
fn np_sinc(x: f64) -> f64 {
    let y = PI * if x == 0.0 { 1.0e-20 } else { x };
    y.sin() / y
}

/// `firwin(2 · half_len + 1, 1 / max_rate, window=('kaiser', 5.0))`: the
/// low-pass taps, scaled to unit gain at DC.
fn design(half_len: usize, max_rate: usize) -> Vec<f64> {
    let numtaps = 2 * half_len + 1;
    let alpha = 0.5 * (numtaps - 1) as f64;
    let cutoff = 1.0 / max_rate as f64;
    let mid = (numtaps - 1) as f64 / 2.0;
    let i0_beta = bessel_i0(KAISER_BETA);
    let taps: Vec<f64> = (0..numtaps)
        .map(|i| {
            let m = i as f64 - alpha;
            let ratio = (i as f64 - mid) / mid;
            let window = bessel_i0(KAISER_BETA * (1.0 - ratio.powi(2)).sqrt()) / i0_beta;
            cutoff * np_sinc(cutoff * m) * window
        })
        .collect();
    let gain = crate::metrics::pairwise_sum(&taps);
    taps.into_iter().map(|t| t / gain).collect()
}

/// Resamples `x` by `up / down`, as `scipy.signal.resample_poly(x, up,
/// down)` does with every other argument at its default.
///
/// The ratio reduces first. At `1 / 1` the result is `x` unchanged. The
/// output holds `ceil(len · up / down)` samples.
///
/// Returns [`Error::InvalidArgument`] when `up` or `down` is 0.
pub fn resample_poly<T: PolySample>(x: &[T], up: usize, down: usize) -> Result<Vec<T>> {
    if up == 0 || down == 0 {
        return Err(Error::InvalidArgument {
            arg: "up/down",
            reason: format!("up ({up}) and down ({down}) must both be at least 1"),
        });
    }
    let g = gcd(up as u64, down as u64) as usize;
    let (up, down) = (up / g, down / g);
    if up == 1 && down == 1 {
        return Ok(x.to_vec());
    }
    let n_in = x.len();
    let n_out = (n_in * up).div_ceil(down);
    let max_rate = up.max(down);
    let half_len = HALF_LEN_PER_RATE * max_rate;
    let up_t = T::from_f64(up as f64);
    let taps: Vec<T> = design(half_len, max_rate)
        .into_iter()
        .map(|t| T::from_f64(t) * up_t)
        .collect();
    // SciPy prepends `down - half_len % down` zeros to the taps and drops
    // the first `(half_len + pre_pad) / down` outputs of the full filter.
    let pre_pad = down - half_len % down;
    let pre_remove = (half_len + pre_pad) / down;
    let padded_len = pre_pad + taps.len();
    let mut out = Vec::with_capacity(n_out);
    for j in 0..n_out {
        // Full-filter output `t` reads upsampled index `t · down - k` with
        // padded tap `k`; input `i` sits at upsampled index `i · up`.
        let at = (j + pre_remove) * down;
        let first = (at + 1).saturating_sub(padded_len).div_ceil(up);
        let last = (at / up).min(n_in.saturating_sub(1));
        let mut acc = T::ZERO;
        if n_in > 0 {
            for (i, &sample) in x.iter().enumerate().take(last + 1).skip(first) {
                let k = at - i * up;
                if k >= pre_pad {
                    acc = acc + sample * taps[k - pre_pad];
                }
            }
        }
        out.push(acc);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_ratio_returns_the_input() {
        let x = [0.25f64, -0.5, 1.0];
        assert_eq!(resample_poly(&x, 3, 3).expect("resample"), x.to_vec());
    }

    #[test]
    fn zero_rates_are_refused() {
        assert!(resample_poly(&[1.0f64], 0, 1).is_err());
        assert!(resample_poly(&[1.0f64], 1, 0).is_err());
    }

    #[test]
    fn output_length_rounds_up() {
        let x = vec![0.0f64; 7];
        assert_eq!(resample_poly(&x, 4, 1).expect("up").len(), 28);
        assert_eq!(resample_poly(&x, 1, 3).expect("down").len(), 3);
        assert_eq!(resample_poly(&x, 2, 3).expect("ratio").len(), 5);
    }

    #[test]
    fn taps_have_unit_gain_and_linear_phase() {
        let taps = design(40, 4);
        assert_eq!(taps.len(), 81);
        let sum: f64 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "gain {sum}");
        for i in 0..40 {
            assert_eq!(taps[i], taps[80 - i], "tap {i}");
        }
    }

    #[test]
    fn upsampling_keeps_the_input_samples_on_their_phase() {
        // A slow sine upsampled by 4 passes through its own samples: the
        // centre tap of phase 0 carries the gain, up to the filter's ripple.
        let x: Vec<f64> = (0..400).map(|i| (i as f64 * 0.01).sin()).collect();
        let y = resample_poly(&x, 4, 1).expect("resample");
        for i in 100..300 {
            assert!((y[4 * i] - x[i]).abs() < 1e-3, "sample {i}");
        }
    }
}
