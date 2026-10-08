//! The conversion loop and librosa's output length.
//!
//! # Length
//!
//! librosa sizes its result as `ceil(len · to / from)`, the ratio taken in
//! `f64`. The filtered part is `round(len · to / from)` samples, halves
//! rounding up. librosa pads the gap between the two with zeros.
//!
//! # Edges
//!
//! Input outside `0..len` reads as zero. The filter has linear phase and is
//! centred on each output time, so output `m` aligns with input time
//! `m · from / to` with no delay to remove.

use super::filter::{Filter, Rows};
use crate::error::{Error, Result};

/// The length of [`resample_like_librosa`]'s result: `ceil(len · to /
/// from)` with the ratio in `f64`, as `librosa.resample` sizes it.
pub fn librosa_output_len(len: usize, from: u32, to: u32) -> usize {
    if from == to || from == 0 {
        return len;
    }
    let ratio = f64::from(to) / f64::from(from);
    (len as f64 * ratio).ceil() as usize
}

/// `librosa.resample(samples, orig_sr=from, target_sr=to)` at librosa's
/// default quality: a linear-phase Kaiser-windowed sinc low-pass at the
/// lower rate's Nyquist limit.
///
/// Returns [`librosa_output_len`] samples. Equal rates return the input
/// unchanged, as librosa does. An empty input gives an empty output.
///
/// Returns [`Error::InvalidArgument`] when either rate is 0 or the rate
/// ratio passes 338.
pub fn resample_like_librosa(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == 0 {
        return Err(zero_rate("from"));
    }
    if to == 0 {
        return Err(zero_rate("to"));
    }
    if from == to {
        return Ok(samples.to_vec());
    }
    let filter = Filter::new(from, to)?;
    let mut out = filtered(samples, &filter, filtered_len(samples.len(), from, to));
    out.resize(librosa_output_len(samples.len(), from, to), 0.0);
    Ok(out)
}

fn zero_rate(arg: &'static str) -> Error {
    Error::InvalidArgument {
        arg,
        reason: format!("{arg} sample rate is 0 Hz; pass a positive rate"),
    }
}

/// `round(len · to / from)`, halves rounding up, in exact integers.
fn filtered_len(len: usize, from: u32, to: u32) -> usize {
    let num = 2 * len as u128 * u128::from(to) + u128::from(from);
    let n = num / (2 * u128::from(from));
    usize::try_from(n).unwrap_or(usize::MAX)
}

/// The first `n_out` outputs of `filter` over `samples`.
fn filtered(samples: &[f32], filter: &Filter, n_out: usize) -> Vec<f32> {
    let mut rows = Rows::new(filter);
    let (up, down) = (u128::from(filter.up), u128::from(filter.down));
    let reach = filter.reach as i128;
    let len = samples.len() as i128;
    let mut out = Vec::with_capacity(n_out);
    for m in 0..n_out {
        let time = m as u128 * down;
        let first = (time / up) as i128 - reach;
        let row = rows.row(filter, (time % up) as u64);
        let lo = (-first).clamp(0, row.len() as i128) as usize;
        let hi = (len - first).clamp(lo as i128, row.len() as i128) as usize;
        let start = (first + lo as i128) as usize;
        let acc: f64 = row[lo..hi]
            .iter()
            .zip(&samples[start..start + (hi - lo)])
            .map(|(&c, &x)| c * f64::from(x))
            .sum();
        out.push(acc as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_follow_librosa() {
        assert_eq!(librosa_output_len(48_000, 48_000, 16_000), 16_000);
        assert_eq!(librosa_output_len(100, 44_100, 16_000), 37);
        assert_eq!(librosa_output_len(7, 16_000, 16_000), 7);
        assert_eq!(filtered_len(100, 44_100, 16_000), 36);
        assert_eq!(filtered_len(3, 2, 1), 2);
        assert_eq!(filtered_len(1, 22_050, 16_000), 1);
    }

    #[test]
    fn the_gap_past_the_filtered_part_is_zero() -> Result<()> {
        let x = vec![0.5f32; 100];
        let y = resample_like_librosa(&x, 44_100, 16_000)?;
        assert_eq!(y.len(), 37);
        assert_eq!(y[36], 0.0);
        assert!(y[35] != 0.0);
        Ok(())
    }

    #[test]
    fn equal_rates_return_the_input() -> Result<()> {
        let x = [0.1f32, -0.2, 0.3];
        assert_eq!(resample_like_librosa(&x, 16_000, 16_000)?, x);
        Ok(())
    }

    #[test]
    fn empty_input_gives_empty_output() -> Result<()> {
        assert!(resample_like_librosa(&[], 48_000, 16_000)?.is_empty());
        Ok(())
    }

    #[test]
    fn zero_rates_are_refused() {
        let from = resample_like_librosa(&[0.0], 0, 16_000);
        assert!(matches!(
            from,
            Err(Error::InvalidArgument { arg: "from", .. })
        ));
        let to = resample_like_librosa(&[0.0], 16_000, 0);
        assert!(matches!(to, Err(Error::InvalidArgument { arg: "to", .. })));
    }

    #[test]
    fn a_tone_in_the_passband_keeps_its_level() -> Result<()> {
        let from = 48_000u32;
        let x: Vec<f32> = (0..from)
            .map(|n| {
                (2.0 * std::f64::consts::PI * 1_000.0 * f64::from(n) / f64::from(from)).sin() as f32
            })
            .collect();
        let y = resample_like_librosa(&x, from, 16_000)?;
        for (m, &v) in y.iter().enumerate().skip(400).take(15_000) {
            let want = (2.0 * std::f64::consts::PI * 1_000.0 * m as f64 / 16_000.0).sin();
            assert!(
                (f64::from(v) - want).abs() < 1e-5,
                "sample {m}: {v} vs {want}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_tone_past_the_lower_nyquist_limit_is_stopped() -> Result<()> {
        let from = 48_000u32;
        let x: Vec<f32> = (0..from)
            .map(|n| {
                (2.0 * std::f64::consts::PI * 9_000.0 * f64::from(n) / f64::from(from)).sin() as f32
            })
            .collect();
        let y = resample_like_librosa(&x, from, 16_000)?;
        let peak = y[400..15_600].iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(peak < 1e-5, "peak {peak}");
        Ok(())
    }
}
