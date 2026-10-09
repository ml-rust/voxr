//! Sharpness in acum, DIN 45692 in shape: the loudest 100 ms of a signal,
//! its Welch spectrum grouped into 24 critical bands, each band's power
//! raised to 0.23 as a loudness proxy, and the loudness-weighted mean band
//! scaled by 0.11. Bands above 15.8 Bark weigh more, as DIN 45692's
//! weighting function does.

use super::pairwise::pairwise_sum;
use super::welch::welch;
use super::window_sum::{first_argmax, window_sums};
use crate::error::Result;

/// The excerpt judged, in seconds.
const EXCERPT_SECS: f64 = 0.1;

/// Largest Welch segment, in samples.
const MAX_SEGMENT: usize = 2048;

/// Critical bands counted.
const BANDS: usize = 24;

/// Exponent turning band power into a loudness proxy.
const LOUDNESS_EXPONENT: f64 = 0.23;

/// Band centre, in Bark, above which the weighting rises.
const WEIGHT_KNEE: f64 = 15.8;

/// Scale from weighted mean Bark to acum.
const ACUM_SCALE: f64 = 0.11;

/// Zwicker and Terhardt's critical-band rate of `f` Hz, in Bark.
pub fn bark(f: f64) -> f64 {
    13.0 * (0.00076 * f).atan() + 3.5 * (f / 7500.0).powi(2).atan()
}

/// DIN 45692's weighting at band centre `z` Bark.
fn weight(z: f64) -> f64 {
    if z <= WEIGHT_KNEE {
        1.0
    } else {
        0.15 * (0.42 * (z - WEIGHT_KNEE)).exp() + 0.85
    }
}

/// Sharpness of mono `m` at `rate` Hz, in acum: of its loudest 100 ms when
/// it is longer, else of all of it.
///
/// Returns [`crate::Error::InvalidArgument`] when `m` is empty or `rate`
/// is 0.
pub fn sharpness(m: &[f64], rate: u32) -> Result<f64> {
    let n = (EXCERPT_SECS * f64::from(rate)) as usize;
    let excerpt = if m.len() > n {
        let energy: Vec<f64> = m.iter().map(|v| v * v).collect();
        let start = first_argmax(&window_sums(&energy, n)).unwrap_or(0);
        &m[start..start + n]
    } else {
        m
    };
    let psd = welch(excerpt, f64::from(rate), excerpt.len().min(MAX_SEGMENT))?;
    let bark_of: Vec<f64> = psd.freqs.iter().map(|&f| bark(f)).collect();
    let mut loudness = [0.0f64; BANDS];
    let mut band = Vec::new();
    for (k, slot) in loudness.iter_mut().enumerate() {
        let (lo, hi) = (k as f64, k as f64 + 1.0);
        band.clear();
        band.extend(
            bark_of
                .iter()
                .zip(&psd.power)
                .filter(|(z, _)| **z >= lo && **z < hi)
                .map(|(_, p)| *p),
        );
        *slot = pairwise_sum(&band).powf(LOUDNESS_EXPONENT);
    }
    let weighted: Vec<f64> = loudness
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let z = k as f64 + 0.5;
            s * weight(z) * z
        })
        .collect();
    Ok(ACUM_SCALE * pairwise_sum(&weighted) / (pairwise_sum(&loudness) + 1e-12))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn tone(freq: f64, rate: u32, secs: f64) -> Vec<f64> {
        let n = (secs * f64::from(rate)) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / f64::from(rate)).sin())
            .collect()
    }

    #[test]
    fn bark_scale_landmarks() {
        assert_eq!(bark(0.0), 0.0);
        assert!((bark(1000.0) - 8.51).abs() < 0.01);
        assert!(bark(16_000.0) > 23.0 && bark(16_000.0) < 25.0);
    }

    #[test]
    fn a_high_tone_is_sharper_than_a_low_one() {
        let low = sharpness(&tone(200.0, 48_000, 0.5), 48_000).expect("low");
        let high = sharpness(&tone(6_000.0, 48_000, 0.5), 48_000).expect("high");
        assert!(high > 2.0 * low, "low {low}, high {high}");
    }

    #[test]
    fn silence_has_no_sharpness() {
        assert_eq!(sharpness(&[0.0; 9600], 48_000).expect("silence"), 0.0);
    }

    #[test]
    fn empty_input_is_refused() {
        assert!(sharpness(&[], 48_000).is_err());
    }
}
