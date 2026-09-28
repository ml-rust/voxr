//! A wave centered on zero with its peak set to a level.
//!
//! The mean sums in `f64` in sample order and each output sample is rounded
//! to `f32`, so the result matches the JavaScript bit for bit.

use crate::js::js_max;
use crate::js_math::js_pow;

/// Removes the DC offset of `x`, then scales it so its peak is `peak_db`
/// dB.
///
/// The offset is the mean of `x`, and the peak is the largest magnitude of
/// `x` less that mean. A silent or empty `x` is only centered. A NaN sample
/// makes the whole output NaN.
pub fn normalize_peak(x: &[f32], peak_db: f64) -> Vec<f32> {
    let mut mean = 0.0f64;
    for &v in x {
        mean += f64::from(v);
    }
    mean /= if x.is_empty() { 1.0 } else { x.len() as f64 };
    let mut peak = 0.0f64;
    for &v in x {
        peak = js_max(peak, (f64::from(v) - mean).abs());
    }
    // A peak of 0 or NaN leaves the gain at 1.
    let g = if peak != 0.0 && !peak.is_nan() {
        js_pow(10.0, peak_db / 20.0) / peak
    } else {
        1.0
    };
    x.iter()
        .map(|&v| ((f64::from(v) - mean) * g) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_offset_goes_and_the_peak_lands_on_the_level() {
        let y = normalize_peak(&[0.5, 0.7, 0.3, 0.5], -1.0);
        // Mean 0.5, peak 0.2 either way.
        let top = (0.2f64 * (js_pow(10.0, -1.0 / 20.0) / 0.2)) as f32;
        assert!(y[0].abs() < 1e-6);
        assert!((y[1] - top).abs() < 1e-6);
        assert!((y[2] + top).abs() < 1e-6);
    }

    #[test]
    fn silence_and_empty_pass() {
        assert_eq!(normalize_peak(&[0.25; 4], -1.0), vec![0.0; 4]);
        assert!(normalize_peak(&[], -1.0).is_empty());
    }

    #[test]
    fn a_nan_sample_spreads() {
        let y = normalize_peak(&[0.1, f32::NAN, -0.1], -1.0);
        assert!(y.iter().all(|v| v.is_nan()));
    }
}
