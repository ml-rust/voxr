//! Parity of `voxr::atempo` against the ffmpeg binary's `atempo` filter.
//!
//! Each case writes a WAV, runs `ffmpeg -i in.wav -af atempo=<tempo>` on it,
//! and requires the same output length and the same samples. Both 16-bit PCM
//! and 32-bit float inputs run, since ffmpeg filters each in its own format.
//!
//! The binary comes from `$FFMPEG`, else `ffmpeg` on `PATH`. Without either,
//! every test here prints a notice and passes without checking.

mod common;

use std::f64::consts::TAU;
use std::path::{Path, PathBuf};
use std::process::Command;

use voxr::{atempo, atempo_pcm16, encode_wav_f32, encode_wav_pcm16, snr_db};

const TEMPOS: [f64; 6] = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0];

/// Two seconds plus a partial input packet, so the last packet is short.
const SECONDS_PLUS: f64 = 2.0162;

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
    (SECONDS_PLUS * f64::from(rate)) as usize
}

/// A 440 Hz tone at half scale.
fn tone(rate: u32) -> Vec<f64> {
    (0..frames(rate))
        .map(|n| 0.5 * (TAU * 440.0 * n as f64 / f64::from(rate)).sin())
        .collect()
}

/// A logarithmic sweep from 100 Hz to 8 kHz at half scale.
fn sweep(rate: u32) -> Vec<f64> {
    let n = frames(rate);
    let span = n as f64 / f64::from(rate);
    let k = (8000.0f64 / 100.0).ln() / span;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            0.5 * (TAU * 100.0 * ((k * t).exp() - 1.0) / k).sin()
        })
        .collect()
}

/// Speech-like bursts: 24 harmonics over a gliding 90-150 Hz fundamental,
/// syllable-rate envelope, pauses, and a hiss floor as a recording has.
fn speech(rate: u32) -> Vec<f64> {
    let n = frames(rate);
    let floor = common::hiss(n, 0.002, 7);
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            phase += TAU * (120.0 + 30.0 * (TAU * 1.3 * t).sin()) / f64::from(rate);
            let voiced: f64 = (1..=24)
                .map(|h| {
                    let h = f64::from(h);
                    let swell = 1.0 + 0.8 * (TAU * (0.7 + 0.13 * h) * t).sin();
                    swell / h * (h * phase).sin()
                })
                .sum();
            let envelope = (TAU * 3.1 * t).sin().max(0.0).powf(0.7);
            let gate = if t % 1.2 < 0.9 { 1.0 } else { 0.0 };
            0.12 * voiced * envelope * gate + f64::from(floor[i])
        })
        .collect()
}

fn to_pcm16(x: &[f64]) -> Vec<i16> {
    x.iter()
        .map(|&v| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16)
        .collect()
}

/// Runs ffmpeg's atempo over `wav` and returns the raw output in `format`
/// (`s16le` or `f32le`).
fn run_ffmpeg(bin: &Path, dir: &Path, wav: &[u8], tempo: f64, format: &str) -> Vec<u8> {
    let input = dir.join("in.wav");
    let output = dir.join("out.raw");
    std::fs::write(&input, wav).unwrap_or_else(|e| panic!("writing {}: {e}", input.display()));
    let status = Command::new(bin)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&input)
        .args(["-af", &format!("atempo={tempo}"), "-f", format])
        .arg(&output)
        .status()
        .unwrap_or_else(|e| panic!("running {}: {e}", bin.display()));
    assert!(status.success(), "ffmpeg atempo={tempo} -f {format} failed");
    std::fs::read(&output).unwrap_or_else(|e| panic!("reading {}: {e}", output.display()))
}

/// Asserts equal length and equal samples, reporting SNR and the first
/// differing index on failure.
fn assert_same(label: &str, reference: &[f32], ours: &[f32], exact: impl Fn(usize) -> bool) {
    let mismatched: Vec<usize> = (0..reference.len().min(ours.len()))
        .filter(|&i| !exact(i))
        .collect();
    let snr = if reference.is_empty() || ours.is_empty() {
        f64::NAN
    } else {
        snr_db(reference, ours).unwrap_or(f64::NAN)
    };
    assert!(
        reference.len() == ours.len() && mismatched.is_empty(),
        "{label}: ffmpeg {} frames, ours {}; {} samples differ (first at {:?}); SNR {snr:.1} dB",
        reference.len(),
        ours.len(),
        mismatched.len(),
        mismatched.first(),
    );
}

fn check_pcm16(bin: &Path, name: &str, signal: &[f64], rate: u32) {
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
    let pcm = to_pcm16(signal);
    let scaled: Vec<f32> = pcm.iter().map(|&v| f32::from(v) / 32767.0).collect();
    let wav = encode_wav_pcm16(&scaled, rate).unwrap_or_else(|e| panic!("encode {name}: {e}"));
    for tempo in TEMPOS {
        let raw = run_ffmpeg(bin, dir.path(), &wav, tempo, "s16le");
        let reference: Vec<i16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| i16::from_le_bytes(b))
            .collect();
        let ours = atempo_pcm16(&pcm, rate, tempo)
            .unwrap_or_else(|e| panic!("atempo_pcm16 {name} {tempo}: {e}"));
        let as_f32 = |x: &[i16]| x.iter().map(|&v| f32::from(v)).collect::<Vec<f32>>();
        assert_same(
            &format!("s16 {name} {rate} Hz tempo {tempo}"),
            &as_f32(&reference),
            &as_f32(&ours),
            |i| reference[i] == ours[i],
        );
    }
}

fn check_f32(bin: &Path, name: &str, signal: &[f64], rate: u32) {
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
    let x: Vec<f32> = signal.iter().map(|&v| v as f32).collect();
    let wav = encode_wav_f32(&x, rate).unwrap_or_else(|e| panic!("encode {name}: {e}"));
    for tempo in TEMPOS {
        let raw = run_ffmpeg(bin, dir.path(), &wav, tempo, "f32le");
        let reference: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        let ours = atempo(&x, rate, tempo).unwrap_or_else(|e| panic!("atempo {name} {tempo}: {e}"));
        assert_same(
            &format!("f32 {name} {rate} Hz tempo {tempo}"),
            &reference,
            &ours,
            |i| reference[i].to_bits() == ours[i].to_bits(),
        );
    }
}

#[test]
fn pcm16_matches_ffmpeg_at_48k() {
    let Some(bin) = ffmpeg() else {
        return skip_notice();
    };
    let rate = 48_000;
    check_pcm16(&bin, "tone", &tone(rate), rate);
    check_pcm16(&bin, "sweep", &sweep(rate), rate);
    check_pcm16(&bin, "speech", &speech(rate), rate);
}

#[test]
fn f32_matches_ffmpeg_at_48k() {
    let Some(bin) = ffmpeg() else {
        return skip_notice();
    };
    let rate = 48_000;
    check_f32(&bin, "tone", &tone(rate), rate);
    check_f32(&bin, "sweep", &sweep(rate), rate);
    check_f32(&bin, "speech", &speech(rate), rate);
}

/// 24 kHz halves both the segment and the input packet.
#[test]
fn pcm16_speech_matches_ffmpeg_at_24k() {
    let Some(bin) = ffmpeg() else {
        return skip_notice();
    };
    check_pcm16(&bin, "speech", &speech(24_000), 24_000);
}
