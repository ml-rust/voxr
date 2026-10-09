//! Level measures of a finished sound: its share of energy above 8 kHz,
//! how far its last 5 ms sits under its loudest 50 ms, and its 4x
//! oversampled true peak.

use super::pairwise::{pairwise_mean, pairwise_sum};
use super::welch::welch;
use super::window_sum::{exact_sum, window_sums};
use crate::error::{Error, Result};
use crate::resample::resample_poly;

/// Frequency above which energy counts as hiss, in Hz.
const HISS_FLOOR_HZ: f64 = 8000.0;

/// Largest Welch segment for the hiss share, in samples.
const HISS_SEGMENT: usize = 4096;

/// The loudest run the tail is compared with, in seconds.
const LOUD_SECS: f64 = 0.05;

/// The end of the sound judged as its tail, in seconds.
const TAIL_SECS: f64 = 0.005;

/// Oversampling factor of the true peak.
const OVERSAMPLE: usize = 4;

/// Energy above 8 kHz against the whole, in dB, from mono `m`'s Welch
/// spectrum (segments of up to 4096 samples).
///
/// Returns [`Error::InvalidArgument`] when `m` is empty or `rate` is 0.
pub fn hiss_db(m: &[f64], rate: u32) -> Result<f64> {
    let psd = welch(m, f64::from(rate), m.len().min(HISS_SEGMENT))?;
    let high: Vec<f64> = psd
        .freqs
        .iter()
        .zip(&psd.power)
        .filter(|(f, _)| **f > HISS_FLOOR_HZ)
        .map(|(_, p)| *p)
        .collect();
    let share = pairwise_sum(&high) / (pairwise_sum(&psd.power) + 1e-20);
    Ok(10.0 * (share + 1e-20).log10())
}

/// RMS of mono `m`'s loudest run of `w` samples, `w` at least 1. A signal
/// shorter than `w` counts its whole energy against the span, as a
/// valid-mode convolution with the longer kernel does.
fn loudest_rms(m: &[f64], w: usize) -> f64 {
    let scale = 1.0 / w as f64;
    let energy: Vec<f64> = m.iter().map(|v| v * v * scale).collect();
    let peak = if m.len() >= w {
        window_sums(&energy, w)
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max)
    } else {
        exact_sum(&energy)
    };
    (peak + 1e-20).sqrt()
}

/// The last 5 ms of mono `m` against its loudest 50 ms, as an RMS ratio
/// in dB. A rate under 200 Hz has no whole sample in 5 ms; the tail is
/// then all of `m`, as NumPy reads `m[-0:]`.
///
/// Returns [`Error::InvalidArgument`] when `m` is empty or `rate` is under
/// 20 Hz, which leaves no whole sample in 50 ms.
pub fn tail_db(m: &[f64], rate: u32) -> Result<f64> {
    let w = (LOUD_SECS * f64::from(rate)) as usize;
    if m.is_empty() || w == 0 {
        return Err(Error::InvalidArgument {
            arg: "tail_db",
            reason: format!(
                "needs samples ({}) and a rate of at least 20 Hz ({rate})",
                m.len()
            ),
        });
    }
    let loud = loudest_rms(m, w);
    let k = (TAIL_SECS * f64::from(rate)) as usize;
    let tail = if k == 0 {
        m
    } else {
        &m[m.len().saturating_sub(k)..]
    };
    let squares: Vec<f64> = tail.iter().map(|v| v * v).collect();
    let rms = pairwise_mean(&squares).sqrt();
    Ok(20.0 * (rms / loud + 1e-12).log10())
}

/// The 4x oversampled true peak over every channel of `channels`, in dBTP.
/// Each channel resamples as `scipy.signal.resample_poly(x, 4, 1)` does.
///
/// Returns [`Error::InvalidArgument`] when no channel holds a sample.
pub fn true_peak_dbtp(channels: &[Vec<f64>]) -> Result<f64> {
    if channels.iter().all(Vec::is_empty) {
        return Err(Error::InvalidArgument {
            arg: "true_peak_dbtp",
            reason: "needs at least one sample".to_string(),
        });
    }
    let mut peak = 0.0f64;
    for channel in channels {
        for v in resample_poly(channel, OVERSAMPLE, 1)? {
            peak = peak.max(v.abs());
        }
    }
    Ok(20.0 * (peak + 1e-12).log10())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn tone(freq: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / 48_000.0).sin())
            .collect()
    }

    #[test]
    fn a_low_tone_has_no_hiss_and_a_high_one_is_all_hiss() {
        assert!(hiss_db(&tone(500.0, 48_000), 48_000).expect("low") < -60.0);
        assert!(hiss_db(&tone(12_000.0, 48_000), 48_000).expect("high") > -0.1);
    }

    #[test]
    fn a_steady_tone_ends_at_its_own_level() {
        let t = tail_db(&tone(1000.0, 48_000), 48_000).expect("tail");
        assert!(t.abs() < 0.1, "tail {t}");
    }

    #[test]
    fn a_fade_out_ends_far_below() {
        let mut x = tone(1000.0, 48_000);
        let n = x.len() as f64;
        for (i, v) in x.iter_mut().enumerate() {
            *v *= 1.0 - i as f64 / n;
        }
        assert!(tail_db(&x, 48_000).expect("tail") < -40.0);
    }

    #[test]
    fn true_peak_finds_the_peak_between_samples() {
        // A tone at a quarter of the rate, sampled 45 degrees off its
        // crests: every sample sits at 0.707, the waveform peaks at 1.
        let x: Vec<f64> = (0..4800)
            .map(|i| (PI / 2.0 * i as f64 + PI / 4.0).sin())
            .collect();
        let tp = true_peak_dbtp(&[x]).expect("tp");
        assert!(tp > -0.5 && tp < 0.5, "true peak {tp}");
    }

    #[test]
    fn empty_input_is_refused() {
        assert!(tail_db(&[], 48_000).is_err());
        assert!(true_peak_dbtp(&[Vec::new()]).is_err());
        assert!(hiss_db(&[], 48_000).is_err());
    }
}
