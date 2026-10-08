//! Parity of `voxr::resample_like_ffmpeg` against the ffmpeg binary's default
//! resampler.
//!
//! Each case writes a mono WAV, runs `ffmpeg -i in.wav -ac 1 -ar <rate> -f
//! f32le -` on it, and requires the same length and the same bits. 16-bit
//! inputs reach the resampler as `v / 32768`, as ffmpeg converts them. A
//! ratio whose upsampling term is over 1024 interpolates between phases and
//! is held to an SNR floor instead.
//!
//! The bits match ffmpeg's AVX2 and FMA code path. On a CPU without both,
//! ffmpeg sums in another order, so every case there is held to the SNR
//! floor.
//!
//! The binary comes from `$FFMPEG`, else `ffmpeg` on `PATH`. Without either,
//! every test here prints a notice and passes without checking.

mod common;

use std::f64::consts::TAU;
use std::path::{Path, PathBuf};
use std::process::Command;

use voxr::{encode_wav_f32, encode_wav_pcm16, ffmpeg_resampled_len, resample_like_ffmpeg, snr_db};

/// The rate pairs common speech and video rates.
const PAIRS: [(u32, u32); 11] = [
    (16_000, 48_000),
    (22_050, 48_000),
    (24_000, 48_000),
    (44_100, 48_000),
    (48_000, 16_000),
    (8_000, 48_000),
    (32_000, 48_000),
    (48_000, 44_100),
    (44_100, 16_000),
    (48_000, 24_000),
    (96_000, 48_000),
];

/// The lowest SNR, in dB, a result that differs from ffmpeg's by float
/// rounding keeps: an interpolating ratio, or any ratio on a CPU where
/// ffmpeg takes a different SIMD path.
const MIN_ROUNDING_SNR_DB: f64 = 138.0;

/// Seconds of signal per long case, plus a partial packet.
const SECONDS: f64 = 1.5163;

fn ffmpeg() -> Option<PathBuf> {
    let bin = std::env::var_os("FFMPEG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffmpeg"));
    let ok = Command::new(&bin)
        .arg("-version")
        .output()
        .is_ok_and(|out| out.status.success());
    ok.then_some(bin)
}

fn skip_notice() {
    eprintln!("skipping: ffmpeg unavailable; set FFMPEG or put ffmpeg on PATH");
}

fn frames(rate: u32) -> usize {
    (SECONDS * f64::from(rate)) as usize
}

/// A logarithmic sweep from 50 Hz to just under the Nyquist limit.
fn sweep(rate: u32, n: usize) -> Vec<f64> {
    let span = n as f64 / f64::from(rate);
    let top = 0.49 * f64::from(rate);
    let k = (top / 50.0).ln() / span;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            0.7 * (TAU * 50.0 * ((k * t).exp() - 1.0) / k).sin()
        })
        .collect()
}

/// Speech-like bursts: 30 harmonics over a gliding fundamental, a
/// syllable-rate envelope, pauses of digital silence, and a hiss floor.
fn speech(rate: u32, n: usize) -> Vec<f64> {
    let floor = common::hiss(n, 0.003, 11);
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            phase += TAU * (130.0 + 40.0 * (TAU * 1.7 * t).sin()) / f64::from(rate);
            let voiced: f64 = (1..=30)
                .map(|h| {
                    let h = f64::from(h);
                    (1.0 + 0.6 * (TAU * (0.5 + 0.11 * h) * t).sin()) / h * (h * phase).sin()
                })
                .sum();
            let envelope = (TAU * 2.9 * t).sin().max(0.0).powf(0.6);
            if t % 1.0 < 0.8 {
                0.15 * voiced * envelope + f64::from(floor[i])
            } else {
                0.0
            }
        })
        .collect()
}

/// Full-scale white noise.
fn noise(n: usize, seed: u64) -> Vec<f64> {
    common::hiss(n, 0.9, seed)
        .into_iter()
        .map(f64::from)
        .collect()
}

fn to_pcm16(x: &[f64]) -> Vec<i16> {
    x.iter()
        .map(|&v| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16)
        .collect()
}

/// ffmpeg's decode of `wav` at `to` Hz, mono, as raw floats.
fn run_ffmpeg(bin: &Path, wav: &[u8], to: u32) -> Vec<f32> {
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
    let input = dir.path().join("in.wav");
    std::fs::write(&input, wav).unwrap_or_else(|e| panic!("writing {}: {e}", input.display()));
    let out = Command::new(bin)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&input)
        .args(["-ac", "1", "-ar", &to.to_string(), "-f", "f32le", "-"])
        .output()
        .unwrap_or_else(|e| panic!("running {}: {e}", bin.display()));
    assert!(
        out.status.success(),
        "ffmpeg -ar {to} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}

/// The SNR of `ours` against `reference`, infinite when they are equal.
fn snr(reference: &[f32], ours: &[f32]) -> f64 {
    if reference.is_empty() && ours.is_empty() {
        return f64::INFINITY;
    }
    snr_db(reference, ours).unwrap_or(f64::NAN)
}

/// Runs one case and returns `(reference, ours)` after checking the length
/// against ffmpeg and against `ffmpeg_resampled_len`.
fn case(
    bin: &Path,
    label: &str,
    x: &[f64],
    from: u32,
    to: u32,
    pcm16: bool,
) -> (Vec<f32>, Vec<f32>) {
    let (wav, input) = if pcm16 {
        let pcm = to_pcm16(x);
        let scaled: Vec<f32> = pcm.iter().map(|&v| f32::from(v) / 32767.0).collect();
        let wav = encode_wav_pcm16(&scaled, from).unwrap_or_else(|e| panic!("{label}: {e}"));
        (wav, pcm.iter().map(|&v| f32::from(v) / 32768.0).collect())
    } else {
        let input: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let wav = encode_wav_f32(&input, from).unwrap_or_else(|e| panic!("{label}: {e}"));
        (wav, input)
    };
    let reference = run_ffmpeg(bin, &wav, to);
    let ours = resample_like_ffmpeg(&input, from, to).unwrap_or_else(|e| panic!("{label}: {e}"));
    let ruled =
        ffmpeg_resampled_len(input.len(), from, to).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_eq!(
        ours.len(),
        reference.len(),
        "{label}: length against ffmpeg"
    );
    assert_eq!(ruled, reference.len(), "{label}: ffmpeg_resampled_len");
    (reference, ours)
}

/// Whether this CPU gives ffmpeg the AVX2 and FMA path whose bits
/// `resample_like_ffmpeg` reproduces.
fn ffmpeg_takes_fma_path() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Requires the same bits at every sample, reporting the first difference.
/// Off the FMA path it requires [`MIN_ROUNDING_SNR_DB`] instead.
fn assert_bit_exact(label: &str, reference: &[f32], ours: &[f32]) {
    if !ffmpeg_takes_fma_path() {
        let snr = snr(reference, ours);
        eprintln!("{label}: no AVX2 and FMA, SNR {snr:.1} dB");
        assert!(
            snr >= MIN_ROUNDING_SNR_DB,
            "{label}: SNR {snr:.1} dB is under {MIN_ROUNDING_SNR_DB} dB"
        );
        return;
    }
    let differ = reference
        .iter()
        .zip(ours)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let first = reference
        .iter()
        .zip(ours)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    eprintln!(
        "{label}: {} samples, SNR {:.1} dB, {differ} differ",
        reference.len(),
        snr(reference, ours)
    );
    assert!(
        differ == 0,
        "{label}: {differ} of {} samples differ, first at {first:?}; SNR {:.1} dB",
        reference.len(),
        snr(reference, ours)
    );
}

#[test]
fn pcm16_wav_matches_ffmpeg_bit_for_bit() {
    let Some(bin) = ffmpeg() else {
        return skip_notice();
    };
    for (from, to) in PAIRS {
        let n = frames(from);
        for (name, x) in [("speech", speech(from, n)), ("sweep", sweep(from, n))] {
            let label = format!("s16 {name} {from} -> {to} Hz");
            let (reference, ours) = case(&bin, &label, &x, from, to, true);
            assert_bit_exact(&label, &reference, &ours);
        }
    }
}

#[test]
fn f32_wav_matches_ffmpeg_bit_for_bit() {
    let Some(bin) = ffmpeg() else {
        return skip_notice();
    };
    for (from, to) in PAIRS {
        let x = noise(frames(from), u64::from(from ^ to));
        let label = format!("f32 noise {from} -> {to} Hz");
        let (reference, ours) = case(&bin, &label, &x, from, to, false);
        assert_bit_exact(&label, &reference, &ours);
    }
}

/// Lengths around the filter length cover the flush-only start, the first
/// streamed output and inputs too short for any output.
#[test]
fn short_inputs_match_ffmpeg_bit_for_bit() {
    let Some(bin) = ffmpeg() else {
        return skip_notice();
    };
    for (from, to) in [
        (24_000, 48_000),
        (44_100, 48_000),
        (48_000, 16_000),
        (48_000, 44_100),
    ] {
        for n in [
            1, 2, 20, 21, 22, 23, 33, 40, 66, 67, 70, 99, 100, 101, 102, 150, 211,
        ] {
            let x = noise(n, n as u64);
            let label = format!("f32 noise {n} samples {from} -> {to} Hz");
            let (reference, ours) = case(&bin, &label, &x, from, to, false);
            assert_bit_exact(&label, &reference, &ours);
        }
    }
}

#[test]
fn interpolated_ratios_track_ffmpeg() {
    let Some(bin) = ffmpeg() else {
        return skip_notice();
    };
    for (from, to) in [(48_000, 44_101), (16_001, 48_000), (44_100, 15_999)] {
        let n = frames(from);
        for (name, x) in [("speech", speech(from, n)), ("noise", noise(n, 5))] {
            let label = format!("{name} {from} -> {to} Hz");
            let (reference, ours) = case(&bin, &label, &x, from, to, name == "speech");
            let snr = snr(&reference, &ours);
            eprintln!("{label}: SNR {snr:.1} dB");
            assert!(
                snr >= MIN_ROUNDING_SNR_DB,
                "{label}: SNR {snr:.1} dB is under {MIN_ROUNDING_SNR_DB} dB"
            );
        }
    }
}
