//! Cross-correlation of two segments through numr's real FFT.
//!
//! Each segment's alignment signal is zero-padded to twice the segment length
//! and transformed once. The correlation of two segments is the inverse
//! transform, scaled by one over its length, of one spectrum times the
//! conjugate of the other, so lag `i` holds `sum_n earlier[n + i] * later[n]` for every `i` below the
//! segment length. The transforms run in `f64`; the result is `f32`.

use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::Complex128;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use crate::error::Result;

/// FFT client and padding buffer for one segment length.
pub(super) struct Correlator {
    client: CpuClient,
    device: CpuDevice,
    /// Transform length: twice the segment length.
    padded: usize,
    /// Zero-padded real input, reused across calls.
    signal: Vec<f64>,
    /// Spectrum product, reused across `correlate` calls.
    product: Vec<Complex128>,
    /// Correlation at every lag, reused across `correlate` calls.
    lags: Vec<f32>,
}

impl Correlator {
    /// A correlator for segments of `segment_len` frames.
    pub(super) fn new(segment_len: usize) -> Self {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let padded = 2 * segment_len;
        Self {
            client,
            device,
            padded,
            signal: vec![0.0; padded],
            product: Vec::new(),
            lags: Vec::new(),
        }
    }

    /// Spectrum of `head` zero-padded to the transform length: `padded / 2 + 1`
    /// bins. `head` holds at most the segment length.
    pub(super) fn spectrum(&mut self, head: &[f32]) -> Result<Vec<Complex128>> {
        self.signal.fill(0.0);
        for (dst, &src) in self.signal.iter_mut().zip(head) {
            *dst = f64::from(src);
        }
        let input = Tensor::<CpuRuntime>::from_slice(&self.signal, &[self.padded], &self.device)?;
        let spectrum = self.client.rfft(&input, FftNormalization::None)?;
        Ok(spectrum.try_to_vec::<Complex128>()?)
    }

    /// Correlation at every lag of the transform: entry `i` holds
    /// `sum_n earlier[n + i] * later[n]`, circular over the padded length.
    pub(super) fn correlate(
        &mut self,
        earlier: &[Complex128],
        later: &[Complex128],
    ) -> Result<&[f32]> {
        self.product.clear();
        self.product
            .extend(earlier.iter().zip(later).map(|(a, b)| Complex128 {
                re: a.re * b.re + a.im * b.im,
                im: a.im * b.re - a.re * b.im,
            }));
        let input =
            Tensor::<CpuRuntime>::from_slice(&self.product, &[self.product.len()], &self.device)?;
        let lags = self
            .client
            .irfft(&input, Some(self.padded), FftNormalization::Backward)?;
        self.lags.clear();
        self.lags
            .extend(lags.try_to_vec::<f64>()?.into_iter().map(|v| v as f32));
        Ok(&self.lags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Catches a sign-convention mix-up: a flipped transform direction turns
    /// lag `i` into lag `-i` and sends every alignment the wrong way.
    #[test]
    fn matches_direct_sum() -> Result<()> {
        let len = 16;
        let earlier: Vec<f32> = (0..len).map(|n| ((n * 7) % 5) as f32 - 2.0).collect();
        let later: Vec<f32> = (0..len).map(|n| ((n * 3) % 4) as f32 - 1.5).collect();
        let mut correlator = Correlator::new(len);
        let a = correlator.spectrum(&earlier)?;
        let b = correlator.spectrum(&later)?;
        let got = correlator.correlate(&a, &b)?;
        assert_eq!(got.len(), 2 * len);
        for (lag, &value) in got.iter().enumerate().take(len) {
            let want: f32 = (0..len - lag).map(|n| earlier[n + lag] * later[n]).sum();
            assert!(
                (value - want).abs() < 1e-3,
                "lag {lag}: got {value}, want {want}"
            );
        }
        Ok(())
    }
}
