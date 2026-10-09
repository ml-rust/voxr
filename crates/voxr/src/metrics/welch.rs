//! Welch's power spectral density, as `scipy.signal.welch(x, fs,
//! nperseg=n)` computes it with every other argument at its default.
//!
//! Provenance: a port of SciPy's `welch`, `csd` and `ShortTimeFFT`
//! (BSD-3-Clause). The defaults: a periodic Hann window, half-segment
//! overlap, each segment's mean removed, one-sided density scaling, and the
//! mean over segments.

use std::f64::consts::PI;

use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::Complex128;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use super::pairwise::{pairwise_mean, pairwise_sum};
use crate::error::{Error, Result};

/// A one-sided power spectral density.
#[derive(Debug, Clone, PartialEq)]
pub struct Psd {
    /// Bin frequencies in Hz: `k · fs / nperseg`.
    pub freqs: Vec<f64>,
    /// Power per Hz at each bin.
    pub power: Vec<f64>,
}

/// `scipy.signal.get_window('hann', n)`: the periodic Hann window, built as
/// SciPy's general cosine builds it over `np.linspace(-π, π, n + 1)`.
fn periodic_hann(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let step = (PI - -PI) / n as f64;
    (0..n)
        .map(|i| 0.5 + 0.5 * (i as f64 * step + -PI).cos())
        .collect()
}

/// `scipy.signal.welch(x, fs, nperseg=nperseg)`.
///
/// A signal shorter than `nperseg` takes its own length as the segment, as
/// SciPy does after its warning.
///
/// Returns [`Error::InvalidArgument`] when `x` is empty, `nperseg` is 0, or
/// `fs` is not positive.
pub fn welch(x: &[f64], fs: f64, nperseg: usize) -> Result<Psd> {
    if x.is_empty() || nperseg == 0 || fs.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(Error::InvalidArgument {
            arg: "welch",
            reason: format!(
                "needs samples ({}), a segment length ({nperseg}) and a positive rate ({fs})",
                x.len()
            ),
        });
    }
    let nperseg = nperseg.min(x.len());
    let noverlap = nperseg / 2;
    let hop = nperseg - noverlap;
    let segments = (x.len() - noverlap) / hop;

    // `ShortTimeFFT.scale_to('psd')`: the window times
    // `1 / sqrt(sum(w²) / T)`, the sum a left fold (Python's `sum`).
    let window = periodic_hann(nperseg);
    let energy = window.iter().fold(0.0f64, |acc, w| acc + w * w);
    let factor = 1.0 / (energy / (1.0 / fs)).sqrt();
    let window: Vec<f64> = window.iter().map(|w| w * factor).collect();

    let mut frames = vec![0.0f64; segments * nperseg];
    for (s, frame) in frames.chunks_exact_mut(nperseg).enumerate() {
        let seg = &x[s * hop..s * hop + nperseg];
        let mean = pairwise_mean(seg);
        for ((dst, &v), &w) in frame.iter_mut().zip(seg).zip(&window) {
            *dst = (v - mean) * w;
        }
    }

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(&frames, &[segments, nperseg], &device)?;
    let spectrum: Vec<Complex128> = client
        .rfft(&input, FftNormalization::None)?
        .contiguous()?
        .try_to_vec()?;
    let bins = nperseg / 2 + 1;
    let paired_end = if nperseg.is_multiple_of(2) {
        bins - 1
    } else {
        bins
    };

    let mut column = vec![0.0f64; segments];
    let power: Vec<f64> = (0..bins)
        .map(|k| {
            for (s, slot) in column.iter_mut().enumerate() {
                let c = spectrum[s * bins + k];
                let p = c.re * c.re + c.im * c.im;
                *slot = if k >= 1 && k < paired_end { p * 2.0 } else { p };
            }
            if segments > 1 {
                pairwise_sum(&column) / segments as f64
            } else {
                column[0]
            }
        })
        .collect();
    let freqs = rfft_freqs(nperseg, fs);
    Ok(Psd { freqs, power })
}

/// `scipy.fft.rfftfreq(n, 1 / fs)`: `k · (1 / (n · (1 / fs)))`.
fn rfft_freqs(n: usize, fs: f64) -> Vec<f64> {
    let val = 1.0 / (n as f64 * (1.0 / fs));
    (0..n / 2 + 1).map(|k| k as f64 * val).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_is_periodic() {
        let w = periodic_hann(8);
        assert_eq!(w[0], 0.0);
        assert!((w[4] - 1.0).abs() < 1e-15);
        assert!((w[2] - 0.5).abs() < 1e-15 && (w[6] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn a_sine_peaks_at_its_bin_with_its_power() {
        let fs = 48_000.0;
        let n = 4096;
        let f0 = 64.0 * fs / n as f64;
        let x: Vec<f64> = (0..n * 4)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).sin())
            .collect();
        let psd = welch(&x, fs, n).expect("welch");
        assert_eq!(psd.power.len(), n / 2 + 1);
        let peak = psd
            .power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(k, _)| k);
        assert_eq!(peak, Some(64));
        // Density times bin width sums to the mean square, 0.5.
        let total: f64 = psd.power.iter().sum::<f64>() * fs / n as f64;
        assert!((total - 0.5).abs() < 1e-6, "power {total}");
    }

    #[test]
    fn a_short_signal_is_one_segment() {
        let x = [1.0, -1.0, 1.0, -1.0, 1.0];
        let psd = welch(&x, 10.0, 4096).expect("welch");
        assert_eq!(psd.freqs, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn empty_input_is_refused() {
        assert!(welch(&[], 48_000.0, 16).is_err());
        assert!(welch(&[1.0], 0.0, 16).is_err());
    }
}
