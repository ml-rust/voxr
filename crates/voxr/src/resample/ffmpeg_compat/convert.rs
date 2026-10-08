//! The conversion: each output's input window, phase and dot product.
//!
//! # Alignment and edges
//!
//! Output `m` sits at input time `t(m) = m * down / up`, so output 0 lines
//! up with input sample 0 and the conversion adds no delay. Its window
//! covers input samples `floor(t(m)) - (taps / 2 - 1)` through
//! `floor(t(m)) + taps / 2`. Before the start, input `-k` reads input `k`: a
//! mirror about sample 0. Past the end, input `n - 1 + k` reads input
//! `n - k`: a mirror that repeats the last sample. A constant input
//! therefore stays constant up to both edges.
//!
//! # Arithmetic
//!
//! An output on a phase is one dot product (see `kernel`). An output between
//! two phases takes the dot product with both, then moves from the first
//! towards the second by the fractional phase, as `f32`.

use super::bank::{Bank, Plan};
use super::kernel::{dot, dot_between};
use super::length::{output_len, plan_for};
use crate::error::{Error, Result};

/// Resample mono `f32` samples from `from` Hz to `to` Hz as ffmpeg's default
/// resampler does (`ffmpeg -ar <to>`, float processing).
///
/// The result equals ffmpeg's output sample for sample when the reduced
/// ratio's upsampling term is at most 1024, as for any pair of 8, 11.025,
/// 16, 22.05, 24, 32, 44.1, 48 and 96 kHz. A larger term interpolates
/// between phases, and the result differs from ffmpeg's by float rounding.
/// The bits are those of ffmpeg's x86-64 AVX2 and FMA code path. On a CPU
/// without both, ffmpeg sums in another order, and the two differ by float
/// rounding.
/// [`ffmpeg_resampled_len`](super::ffmpeg_resampled_len) gives the length.
/// Equal rates return the samples unchanged, as ffmpeg passes them through.
///
/// ffmpeg converts 16-bit input to float as `v / 32768` before resampling,
/// so `samples` scaled that way reproduce a 16-bit WAV's conversion too.
///
/// Returns [`Error::InvalidArgument`] when either rate is 0 or the ratio
/// needs a filter over the tap limit.
pub fn resample_like_ffmpeg(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    let Some(plan) = plan_for(from, to)? else {
        return Ok(samples.to_vec());
    };
    let out_len = output_len(samples.len(), &plan)?;
    convert(samples, &plan, &Bank::new(&plan), out_len)
}

/// [`resample_like_ffmpeg`] for interleaved frames of `channels` samples.
///
/// ffmpeg resamples each channel on its own, so each channel of the result
/// equals [`resample_like_ffmpeg`] on that channel. Returns
/// [`Error::InvalidArgument`] when `channels` is 0, when `samples` is not a
/// whole number of frames, or for anything [`resample_like_ffmpeg`] rejects.
pub fn resample_interleaved_like_ffmpeg(
    samples: &[f32],
    channels: usize,
    from: u32,
    to: u32,
) -> Result<Vec<f32>> {
    if channels == 0 || !samples.len().is_multiple_of(channels) {
        return Err(Error::InvalidArgument {
            arg: "channels",
            reason: format!(
                "{} samples do not split into frames of {channels} channels; pass whole \
                 interleaved frames and a channel count of at least 1",
                samples.len()
            ),
        });
    }
    let Some(plan) = plan_for(from, to)? else {
        return Ok(samples.to_vec());
    };
    let out_len = output_len(samples.len() / channels, &plan)?;
    let bank = Bank::new(&plan);
    let mut out = vec![0.0f32; out_len * channels];
    for ch in 0..channels {
        let mono: Vec<f32> = samples.iter().skip(ch).step_by(channels).copied().collect();
        let converted = convert(&mono, &plan, &bank, out_len)?;
        for (slot, v) in out.iter_mut().skip(ch).step_by(channels).zip(converted) {
            *slot = v;
        }
    }
    Ok(out)
}

/// Where an output sits: input index, phase, and the remainder of the phase
/// position in units of `1 / up` phase.
#[derive(Debug, Clone, Copy)]
struct Position {
    index: u64,
    phase: u64,
    rem: u64,
}

/// `out_len` outputs of `samples` under `plan`.
fn convert(samples: &[f32], plan: &Plan, bank: &Bank, out_len: usize) -> Result<Vec<f32>> {
    if out_len == 0 {
        return Ok(Vec::new());
    }
    let n = samples.len() as u64;
    let lead = (plan.taps / 2 - 1) as u64;
    let last_index = (out_len as u128 - 1) * u128::from(plan.down) / u128::from(plan.up);
    let tail = (last_index as u64 + plan.taps as u64 / 2 + 1).saturating_sub(n);
    let padded = mirror_pad(samples, lead, tail)?;

    let step = plan.down * plan.phases;
    let (step_phases, step_rem) = (step / plan.up, step % plan.up);
    let mut at = Position {
        index: 0,
        phase: 0,
        rem: 0,
    };
    let mut out = Vec::with_capacity(out_len);
    for _ in 0..out_len {
        let start = at.index as usize;
        let window = padded
            .get(start..start + plan.taps)
            .ok_or_else(|| internal(plan, n, "an output window runs past the padded input"))?;
        let row = |k: u64| {
            bank.row(k)
                .ok_or_else(|| internal(plan, n, "a phase index runs past the filter bank"))
        };
        let value = if at.rem == 0 {
            dot(window, row(at.phase)?)
        } else {
            let frac = (at.rem as f64 / plan.up as f64) as f32;
            dot_between(window, row(at.phase)?, row(at.phase + 1)?, frac)
        };
        out.push(value);

        let mut phase = at.phase + step_phases;
        at.rem += step_rem;
        if at.rem >= plan.up {
            at.rem -= plan.up;
            phase += 1;
        }
        at.index += phase / plan.phases;
        at.phase = phase % plan.phases;
    }
    Ok(out)
}

/// `samples` with `lead` mirrored samples before them and `tail` after, as
/// the module docs describe.
fn mirror_pad(samples: &[f32], lead: u64, tail: u64) -> Result<Vec<f32>> {
    let n = samples.len() as u64;
    if lead >= n || tail > n {
        return Err(Error::DataError {
            reason: format!(
                "mirroring {lead} samples before and {tail} after a {n}-sample input reads \
                 outside it; the output length rule rules this out, so report this as a \
                 voxr bug"
            ),
        });
    }
    let before = samples.iter().skip(1).take(lead as usize).rev();
    let after = samples.iter().rev().take(tail as usize);
    Ok(before.chain(samples).chain(after).copied().collect())
}

fn internal(plan: &Plan, n: u64, what: &str) -> Error {
    Error::DataError {
        reason: format!(
            "resampling {n} samples by {}/{} with {} taps: {what}; the inputs passed every \
             check, so report this as a voxr bug",
            plan.up, plan.down, plan.taps
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_pad_reflects_both_edges() -> Result<()> {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(
            mirror_pad(&x, 2, 3)?,
            [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0]
        );
        assert!(mirror_pad(&x, 4, 0).is_err());
        Ok(())
    }

    #[test]
    fn a_constant_stays_constant_to_the_edges() -> Result<()> {
        let x = vec![0.25f32; 3000];
        for (from, to) in [(48_000, 16_000), (44_100, 48_000), (48_000, 44_100)] {
            let y = resample_like_ffmpeg(&x, from, to)?;
            for (i, &v) in y.iter().enumerate() {
                assert!((v - 0.25).abs() < 1e-5, "{from}->{to} sample {i}: {v}");
            }
        }
        Ok(())
    }

    #[test]
    fn upsampling_by_two_keeps_every_input_sample() -> Result<()> {
        let x: Vec<f32> = (0..500)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
            .collect();
        let y = resample_like_ffmpeg(&x, 24_000, 48_000)?;
        assert_eq!(y.len(), 1000);
        for (i, &v) in x.iter().enumerate() {
            assert_eq!(y[2 * i], v, "input {i}");
        }
        Ok(())
    }

    #[test]
    fn equal_rates_pass_through() -> Result<()> {
        let x = [0.1f32, -0.2, 0.3];
        assert_eq!(resample_like_ffmpeg(&x, 22_050, 22_050)?, x);
        Ok(())
    }

    #[test]
    fn an_interpolating_ratio_has_the_ruled_length() -> Result<()> {
        let x = vec![0.5f32; 4000];
        let y = resample_like_ffmpeg(&x, 48_000, 44_101)?;
        assert_eq!(
            y.len(),
            super::super::ffmpeg_resampled_len(4000, 48_000, 44_101)?
        );
        assert!(y.iter().all(|v| (v - 0.5).abs() < 1e-4));
        Ok(())
    }

    #[test]
    fn interleaved_channels_resample_independently() -> Result<()> {
        let left: Vec<f32> = (0..800).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let right: Vec<f32> = (0..800).map(|i| (i as f32 * 0.11).cos() * 0.3).collect();
        let both: Vec<f32> = left
            .iter()
            .zip(&right)
            .flat_map(|(&l, &r)| [l, r])
            .collect();
        let y = resample_interleaved_like_ffmpeg(&both, 2, 16_000, 48_000)?;
        let l = resample_like_ffmpeg(&left, 16_000, 48_000)?;
        let r = resample_like_ffmpeg(&right, 16_000, 48_000)?;
        let joined: Vec<f32> = l.iter().zip(&r).flat_map(|(&a, &b)| [a, b]).collect();
        assert_eq!(y, joined);
        Ok(())
    }

    #[test]
    fn a_partial_frame_is_refused() {
        let r = resample_interleaved_like_ffmpeg(&[0.0; 5], 2, 16_000, 48_000);
        assert!(matches!(
            r,
            Err(Error::InvalidArgument {
                arg: "channels",
                ..
            })
        ));
    }
}
