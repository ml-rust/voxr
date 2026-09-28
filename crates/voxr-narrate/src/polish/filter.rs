//! Second-order filters from the RBJ audio EQ cookbook: a high-pass, a
//! low-pass and a low shelf, run in `f64` direct form 1 with each output
//! sample stored as `f32`.
//!
//! The coefficients are computed as the JavaScript computes them and then
//! divided by `a0`. The JavaScript divides each output sum by `a0` instead,
//! so an output sample can differ from it in the last bits.

use biquad::{Biquad, Coefficients, DirectForm1};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

use crate::js_math::js_pow;

/// The filter shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Kind {
    HighPass,
    LowPass,
    /// A low shelf with this gain in dB.
    LowShelf(f64),
}

/// The coefficients for a `kind` filter at `f` Hz with Q `1 / sqrt(2)`,
/// sampled at `rate` Hz.
fn coefficients(kind: Kind, f: f64, rate: f64) -> Coefficients<f64> {
    let w = (2.0 * PI * f) / rate;
    let cos = w.cos();
    let alpha = w.sin() / (2.0 * FRAC_1_SQRT_2);
    let (b, a) = match kind {
        Kind::LowShelf(gain_db) => {
            let a = js_pow(10.0, gain_db / 40.0);
            let r = 2.0 * a.sqrt() * alpha;
            (
                [
                    a * (a + 1.0 - (a - 1.0) * cos + r),
                    2.0 * a * (a - 1.0 - (a + 1.0) * cos),
                    a * (a + 1.0 - (a - 1.0) * cos - r),
                ],
                [
                    a + 1.0 + (a - 1.0) * cos + r,
                    -2.0 * (a - 1.0 + (a + 1.0) * cos),
                    a + 1.0 + (a - 1.0) * cos - r,
                ],
            )
        }
        Kind::HighPass | Kind::LowPass => {
            let (k, b1) = if kind == Kind::HighPass {
                let k = 1.0 + cos;
                (k, -k)
            } else {
                let k = 1.0 - cos;
                (k, k)
            };
            (
                [k / 2.0, b1, k / 2.0],
                [1.0 + alpha, -2.0 * cos, 1.0 - alpha],
            )
        }
    };
    Coefficients {
        a1: a[1] / a[0],
        a2: a[2] / a[0],
        b0: b[0] / a[0],
        b1: b[1] / a[0],
        b2: b[2] / a[0],
    }
}

/// `x` through a fresh `kind` filter at `f` Hz, sampled at `rate` Hz.
pub(super) fn filter(x: &[f32], kind: Kind, f: f64, rate: u32) -> Vec<f32> {
    let mut stage = DirectForm1::<f64>::new(coefficients(kind, f, f64::from(rate)));
    x.iter().map(|&v| stage.run(f64::from(v)) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sine of `hz` at 48 kHz, `len` samples long.
    fn sine(hz: f64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (0.5 * (2.0 * PI * hz * i as f64 / 48000.0).sin()) as f32)
            .collect()
    }

    /// The RMS level of the second half of `x`, past the filter's settling.
    fn tail_rms(x: &[f32]) -> f64 {
        let tail = &x[x.len() / 2..];
        (tail.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / tail.len() as f64).sqrt()
    }

    #[test]
    fn a_flat_shelf_passes_the_signal() {
        let x = sine(300.0, 4800);
        let y = filter(&x, Kind::LowShelf(0.0), 250.0, 48000);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn the_corner_is_3_db_down() {
        let x = sine(250.0, 48000);
        let rms = tail_rms(&x);
        for kind in [Kind::HighPass, Kind::LowPass] {
            let y = filter(&x, kind, 250.0, 48000);
            let db = 20.0 * (tail_rms(&y) / rms).log10();
            assert!((db + 3.0103).abs() < 0.01, "{kind:?}: {db} dB");
        }
    }

    #[test]
    fn a_shelf_lifts_the_lows_by_its_gain() {
        let x = sine(20.0, 96000);
        let y = filter(&x, Kind::LowShelf(9.0), 250.0, 48000);
        let db = 20.0 * (tail_rms(&y) / tail_rms(&x)).log10();
        assert!((db - 9.0).abs() < 0.05, "{db} dB");
    }

    #[test]
    fn each_call_starts_from_rest() {
        let x = sine(1000.0, 480);
        let a = filter(&x, Kind::HighPass, 120.0, 48000);
        let b = filter(&x, Kind::HighPass, 120.0, 48000);
        assert_eq!(a, b);
        assert!(filter(&[], Kind::LowPass, 250.0, 48000).is_empty());
    }
}
