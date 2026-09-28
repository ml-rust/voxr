//! Sample-rate conversion for mono `f32` audio.
//!
//! Recorded corpora arrive at 44.1 or 48 kHz; [`boostr::model::audio::neucodec`] and the mel
//! front-end demand 16 kHz. [`resample`] is that missing link: a windowed-sinc
//! converter between any two rates, run by rubato's asynchronous sinc resampler.
//!
//! **CPU-only.** Resampling runs once per file at load time, on a host-side
//! `Vec<f32>` that has just come out of the WAV decoder and has never touched a
//! device. Uploading it to run a strided, phase-dependent gather would cost more
//! than the filter itself. Stays here (not in numr) because it is an audio
//! specific composition, not a core numerical op.
//!
//! # Filter
//!
//! rubato evaluates a Blackman² windowed sinc ([`WINDOW`]) at [`OVERSAMPLING`]
//! points per input sample and interpolates cubically between them. The cutoff
//! is `min(from_rate, to_rate) / 2`, the lower of the two Nyquist limits,
//! scaled by [`ROLLOFF`] to leave a transition band. That cutoff is the
//! anti-aliasing: content the output rate cannot represent is stopped before it
//! folds back into the audible band.
//!
//! `taps_per_phase` sets the filter's half-width in samples of the LOWER of the
//! two rates. rubato measures its sinc length in input samples, so the length
//! passed to it is `2 * taps_per_phase * max(1, from_rate / to_rate)`, rounded
//! up to a multiple of 8 by rubato: the same span of the signal whichever
//! direction the conversion goes.
//!
//! # Edges
//!
//! Input outside `0..samples.len()` reads as zero, and rubato's group delay is
//! trimmed so output sample `m` sits at input time `m * from_rate / to_rate`.
//! The first and last few dozen output samples therefore ramp in and out over
//! the filter's half-width instead of holding steady amplitude. For speech
//! corpora that region is leading and trailing silence; callers that need exact
//! edge amplitude must pad the signal themselves before calling.

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction, calculate_cutoff,
};

use crate::error::{Error, Result};

use crate::wav::decode::{WavData, to_mono};

/// Filter half-width in samples of the lower rate. At this width the
/// transition band fits between [`ROLLOFF`] and just past the lower Nyquist
/// limit.
pub const DEFAULT_TAPS_PER_PHASE: usize = 64;

/// Sinc window. Squared Blackman: a faster rolloff than Blackman-Harris² for
/// the same length, with a stopband still far below 16-bit resolution.
const WINDOW: WindowFunction = WindowFunction::Blackman2;

/// Cutoff as a fraction of the lower Nyquist limit, leaving a transition band
/// below it rather than aliasing right at the edge. A filter too short for this
/// cutoff gets rubato's own, lower one instead, which keeps aliasing under the
/// window's sidelobe level.
const ROLLOFF: f32 = 0.95;

/// Sinc evaluations per input sample in rubato's filter table.
const OVERSAMPLING: usize = 128;

/// Input frames rubato processes per call.
const CHUNK_FRAMES: usize = 1024;

/// Largest filter table [`resample`] builds, in taps: rubato's table holds
/// `sinc_len * OVERSAMPLING` of them, so this bounds `taps_per_phase`.
pub const MAX_FILTER_TAPS: usize = 1 << 20;

/// Largest term of the reduced rate ratio `to_rate / from_rate` accepted.
///
/// Real rates share large common factors: 44100 to 16000 reduces to 160/441.
/// A pair that barely reduces, such as 44100 to 16001 (already coprime), is a
/// mistyped rate, not a workload, so it is rejected rather than converted.
const MAX_RATIO_TERM: u64 = 16_383;

fn bad(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

fn gcd(a: u32, b: u32) -> u32 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Number of output samples for `in_len` input samples: `ceil(in_len * to / from)`.
///
/// Computed in `u64` so a long recording at a high ratio cannot wrap.
fn output_len(in_len: usize, from_rate: u32, to_rate: u32) -> Result<usize> {
    let scaled = (in_len as u64).checked_mul(to_rate as u64).ok_or_else(|| {
        bad(
            "samples",
            format!("{in_len} samples at {to_rate} Hz overflows a u64 output length"),
        )
    })?;
    let from = from_rate as u64;
    let out = scaled.div_ceil(from);
    usize::try_from(out).map_err(|_| {
        bad(
            "samples",
            format!("output length {out} does not fit in a usize"),
        )
    })
}

/// Reject a rate pair whose reduced ratio has a term above [`MAX_RATIO_TERM`].
fn check_ratio(from_rate: u32, to_rate: u32) -> Result<()> {
    let divisor = gcd(from_rate, to_rate).max(1);
    let up = (to_rate / divisor) as u64;
    let down = (from_rate / divisor) as u64;
    if up.max(down) > MAX_RATIO_TERM {
        return Err(bad(
            "from_rate",
            format!(
                "resampling {from_rate} Hz to {to_rate} Hz reduces to {up}/{down}, a ratio \
                 with a term above {MAX_RATIO_TERM}; real rates share a larger common \
                 factor, so check both rates for a typo"
            ),
        ));
    }
    Ok(())
}

/// rubato's sinc length for `taps_per_phase`: the filter's full width in input
/// samples. Errors when the filter table would pass [`MAX_FILTER_TAPS`].
fn sinc_len(taps_per_phase: usize, from_rate: u32, to_rate: u32) -> Result<usize> {
    let decimation = (from_rate as f64 / to_rate as f64).max(1.0);
    let len = (2.0 * taps_per_phase as f64 * decimation).ceil();
    if len * OVERSAMPLING as f64 > MAX_FILTER_TAPS as f64 {
        return Err(bad(
            "taps_per_phase",
            format!(
                "{taps_per_phase} taps per phase at {from_rate} Hz to {to_rate} Hz needs a \
                 {len}-sample sinc, over the {MAX_FILTER_TAPS}-tap table limit at \
                 {OVERSAMPLING}x oversampling; lower taps_per_phase"
            ),
        ));
    }
    Ok(len as usize)
}

/// Resample mono f32 samples from `from_rate` to `to_rate`.
///
/// Uses [`DEFAULT_TAPS_PER_PHASE`]; see [`resample_with_taps`] for the rest of
/// the contract.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    resample_with_taps(samples, from_rate, to_rate, DEFAULT_TAPS_PER_PHASE)
}

/// Resample mono f32 samples, choosing the filter's half-width.
///
/// `taps_per_phase` is the half-width in samples of the lower rate; rubato gets
/// a sinc of `2 * taps_per_phase * max(1, from_rate / to_rate)` input samples
/// (see the module docs). Wider filters move the cutoff closer to Nyquist and
/// cost proportionally more arithmetic per output sample.
///
/// Returns exactly `ceil(samples.len() * to_rate / from_rate)` samples. An empty
/// input gives an empty output. `from_rate == to_rate` returns the input
/// unchanged, bit for bit, with no filtering. Out-of-range input reads as zero,
/// so the edges ramp; see the module docs.
///
/// Returns [`Error::InvalidArgument`] when either rate is 0, when
/// `taps_per_phase` is 0, when the rate pair reduces to a ratio with a term
/// above 16383, when the filter table exceeds [`MAX_FILTER_TAPS`], or when the
/// output length overflows.
pub fn resample_with_taps(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
    taps_per_phase: usize,
) -> Result<Vec<f32>> {
    if from_rate == 0 {
        return Err(bad("from_rate", "source sample rate is 0 Hz".to_string()));
    }
    if to_rate == 0 {
        return Err(bad("to_rate", "target sample rate is 0 Hz".to_string()));
    }
    if taps_per_phase == 0 {
        return Err(bad(
            "taps_per_phase",
            "filter length is 0 taps per phase".to_string(),
        ));
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    let out_len = output_len(samples.len(), from_rate, to_rate)?;
    if samples.is_empty() || out_len == 0 {
        return Ok(Vec::new());
    }
    check_ratio(from_rate, to_rate)?;
    let sinc_len = sinc_len(taps_per_phase, from_rate, to_rate)?;

    let failed = |what: &str, err: &dyn std::fmt::Display| Error::DataError {
        reason: format!(
            "rubato failed to {what} for {} samples at {from_rate} Hz to {to_rate} Hz \
             with a {sinc_len}-sample sinc: {err}; the inputs passed every check, so \
             report this as a voxr bug",
            samples.len()
        ),
    };
    let params = SincInterpolationParameters::new(sinc_len, WINDOW)
        .f_cutoff(ROLLOFF.min(calculate_cutoff(sinc_len, WINDOW)))
        .oversampling_factor(OVERSAMPLING)
        .interpolation(SincInterpolationType::Cubic);
    let ratio = to_rate as f64 / from_rate as f64;
    let mut resampler =
        Async::<f32>::new_sinc(ratio, 1.0, &params, CHUNK_FRAMES, 1, FixedAsync::Input)
            .map_err(|e| failed("build the resampler", &e))?;
    let input = InterleavedSlice::new(samples, 1, samples.len())
        .map_err(|e| failed("wrap the input", &e))?;
    let mut out = resampler
        .process_all(&input, samples.len(), None)
        .map_err(|e| failed("resample", &e))?
        .take_data();
    // rubato sizes the output as `ceil(ratio * len)` in f64, which can land one
    // past the exact integer ceiling when the product is a whole number.
    out.resize(out_len, 0.0);
    Ok(out)
}

/// Downmix to mono and resample to `target_rate` in one step.
///
/// Averages the channels with [`to_mono`], then applies [`resample`]. Returns
/// [`Error::InvalidArgument`] for anything either of those rejects.
pub fn to_mono_at_rate(data: &WavData, target_rate: u32) -> Result<Vec<f32>> {
    let mono = to_mono(&data.samples, data.channels)?;
    if data.sample_rate == target_rate {
        return Ok(mono);
    }
    resample(&mono, data.sample_rate, target_rate)
}
