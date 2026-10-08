//! The ffmpeg binary run as the skill runs it, and the check of a
//! resampled track against it: bit for bit on the AVX2 and FMA path that
//! `voxr::resample_like_ffmpeg` reproduces, else within float rounding.

use std::path::{Path, PathBuf};
use std::process::Command;

use voxr_narrate::read_wav_file;

/// The lowest SNR, in dB, a resampled track keeps against ffmpeg's where
/// ffmpeg sums in another order than its AVX2 and FMA path. A 16-bit track
/// then differs by at most one step in a few samples.
pub const MIN_ROUNDING_SNR_DB: f64 = 90.0;

/// `$FFMPEG`, else `ffmpeg` on `PATH`, when it runs.
pub fn bin() -> Option<PathBuf> {
    let bin = std::env::var_os("FFMPEG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffmpeg"));
    let ok = Command::new(&bin)
        .arg("-version")
        .output()
        .is_ok_and(|out| out.status.success());
    ok.then_some(bin)
}

pub fn skip_notice() {
    eprintln!("skipping the SNR checks: ffmpeg unavailable; set FFMPEG or put ffmpeg on PATH");
}

fn write_input(wav: &[u8]) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("creating a temp dir: {e}"));
    let input = dir.path().join("in.wav");
    std::fs::write(&input, wav).unwrap_or_else(|e| panic!("writing {}: {e}", input.display()));
    (dir, input)
}

fn run(bin: &Path, args: &[&str]) -> Vec<u8> {
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("running {}: {e}", bin.display()));
    assert!(
        out.status.success(),
        "ffmpeg {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// The skill's `stretch` command over `wav`, read back as ffmpeg
/// decodes the result.
pub fn stretch(bin: &Path, wav: &[u8], speed: f64) -> Vec<f32> {
    let (dir, input) = write_input(wav);
    let output = dir.path().join("out.wav");
    let tempo = format!("atempo={speed}");
    let (i, o) = (input.to_string_lossy(), output.to_string_lossy());
    run(
        bin,
        &[
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            &i,
            "-af",
            &tempo,
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            &o,
        ],
    );
    read_wav_file(&output)
        .unwrap_or_else(|e| panic!("reading {o}: {e}"))
        .samples
}

/// `wav` decoded as the skill's `samples` decodes it: mono 32-bit
/// float at 48 kHz.
pub fn decode_48k(bin: &Path, wav: &[u8]) -> Vec<f32> {
    let (_dir, input) = write_input(wav);
    let i = input.to_string_lossy();
    let raw = run(
        bin,
        &[
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &i,
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "f32le",
            "-",
        ],
    );
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}

/// Prints the SNR of `ours` against `reference`. Fails on a length
/// mismatch, on any differing sample on the FMA path, and under
/// [`MIN_ROUNDING_SNR_DB`] off it.
pub fn check_snr(name: &str, reference: &[f32], ours: &[f32]) {
    assert_eq!(reference.len(), ours.len(), "resampled length, {name}");
    let snr = voxr::snr_db(reference, ours).unwrap_or(f64::NAN);
    eprintln!("case {name}: SNR against ffmpeg's resampler {snr:.1} dB");
    if takes_fma_path() {
        let differ = reference
            .iter()
            .zip(ours)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(differ, 0, "case {name}: samples differ; SNR {snr:.1} dB");
    } else {
        assert!(
            snr >= MIN_ROUNDING_SNR_DB,
            "case {name}: SNR {snr:.1} dB is under {MIN_ROUNDING_SNR_DB} dB"
        );
    }
}

/// Whether this CPU gives ffmpeg the AVX2 and FMA path whose bits
/// `voxr::resample_like_ffmpeg` reproduces.
pub fn takes_fma_path() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
