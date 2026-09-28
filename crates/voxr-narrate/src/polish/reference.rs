//! A reference voice prepared for cloning: breaths turned down, a thin
//! voice's bass lifted, gently compressed, then centered with its peak at
//! -1 dB.

use super::bass::lift_bass;
use super::breath::{BreathReport, quiet_breaths};
use super::compressor::{CompressorOptions, compress_voice};
use super::normalize::normalize_peak;

/// The peak a prepared reference is normalized to, in dB.
pub const REFERENCE_PEAK_DB: f64 = -1.0;

/// Prepares `x`, sampled at `rate` Hz, as a reference to clone a voice
/// from: [`quiet_breaths`], then [`lift_bass`], then [`compress_voice`]
/// with the default settings, then [`normalize_peak`] to
/// [`REFERENCE_PEAK_DB`]. The length stays. The breath report is
/// [`quiet_breaths`]'s, from the same pass that quiets the breaths: a
/// caller that also needs the breath run count reads it here instead of
/// calling [`quiet_breaths`] again.
pub fn prepare_reference(x: &[f32], rate: u32) -> (Vec<f32>, BreathReport) {
    let (quiet, report) = quiet_breaths(x, rate);
    let (lifted, _) = lift_bass(&quiet, rate);
    let compressed = compress_voice(&lifted, rate, CompressorOptions::default());
    (normalize_peak(&compressed, REFERENCE_PEAK_DB), report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::js_math::js_pow;
    use std::f64::consts::PI;

    #[test]
    fn a_reference_keeps_its_length_and_peaks_at_minus_one_db() {
        let x: Vec<f32> = (0..24000)
            .map(|i| (0.2 * (2.0 * PI * 200.0 * i as f64 / 48000.0).sin() + 0.05) as f32)
            .collect();
        let (y, _) = prepare_reference(&x, 48000);
        assert_eq!(y.len(), x.len());
        let peak = y.iter().fold(0.0f32, |p, v| p.max(v.abs()));
        assert!((f64::from(peak) - js_pow(10.0, -1.0 / 20.0)).abs() < 1e-6);
        assert!(prepare_reference(&[], 48000).0.is_empty());
    }
}
