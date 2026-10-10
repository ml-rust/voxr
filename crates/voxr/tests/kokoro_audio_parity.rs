//! Kokoro's audio path against the reference `kokoro` package, every stage
//! that draws no random number: durations and waveform length (exact), the
//! F0 and energy curves, the decoder trunk, the STFT of an excitation, the
//! generator over a fixed excitation spectrum, and the sine source over a
//! voiced contour (each to f32 rounding).
//!
//! The fixture (`tests/fixtures/kokoro_audio_parity.json`) is written by
//! `tests/tools/kokoro/kokoro_audio_parity.py`; its docstring names every
//! field and both synthetic signals.
//!
//! Skips when the Kokoro model directory is unavailable (`KOKORO_DIR`, else
//! `$BOOSTR_MODELS_DIR/Kokoro-82M`).

use std::f64::consts::PI;

use boostr::Runtime;
use boostr::format::load_voice_pt;
use boostr::model::audio::kokoro::{
    KokoroModelV2, KokoroSynthOptions, SineGen, decode_prosody_durations, length_regulator,
    load_kokoro_v2, select_voice_style, split_voice_style,
};
use boostr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use boostr::tensor::Tensor;
use serde::Deserialize;

mod common;
use common::{load_json, model_fixture, skip_notice};

/// Each value stays within this fraction of the
/// largest reference value it is checked with: f32 rounding over a deep
/// network, far under any error a wrong layer gives.
const REL_TOL: f32 = 1e-4;

/// A spectrum phase is compared where its magnitude reaches this fraction
/// of the largest magnitude: below it, f32 rounding moves the phase freely.
const PHASE_FLOOR: f32 = 1e-2;

/// The largest gap, in radians modulo 2π, a well-defined phase keeps from the reference.
const PHASE_TOL: f32 = 1e-3;

#[derive(Deserialize)]
struct Fixture {
    phonemes: String,
    voice: String,
    ids: Vec<i64>,
    speeds: Vec<Speed>,
    f0: Vec<f32>,
    n: Vec<f32>,
    trunk_channels: Vec<usize>,
    trunk: Vec<Vec<f32>>,
    spectrum: Spectrum,
    generator: Generator,
    sine: Sine,
}

#[derive(Deserialize)]
struct Spectrum {
    source_len: usize,
    frames: usize,
    frame_step: usize,
    mag: Vec<Vec<f32>>,
    phase: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct Speed {
    speed: f32,
    durations: Vec<u32>,
    samples: usize,
}

#[derive(Deserialize)]
struct Generator {
    frames: usize,
    frame_step: usize,
    mag: Vec<Vec<f32>>,
    phase: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct Sine {
    len: usize,
    step: usize,
    values: Vec<Vec<f32>>,
}

/// The script's `hiss`: a 32-bit integer hash of `t`, in `[-0.5, 0.5)`.
fn hiss(t: u64) -> f64 {
    let mut x = t.wrapping_mul(2_654_435_761) & 0xFFFF_FFFF;
    x ^= x >> 16;
    x = x.wrapping_mul(0x45D9F3B) & 0xFFFF_FFFF;
    x ^= x >> 16;
    x as f64 / (1u64 << 32) as f64 - 0.5
}

/// The script's excitation spectrum `[22, frames]`: smooth magnitudes, then
/// phases inside (-2.5, 2.5).
fn spectrum(frames: usize) -> Vec<f32> {
    let row = |b: usize, phase: bool| {
        (0..frames).map(move |f| {
            let f = f as f64;
            let b = b as f64;
            let v = if phase {
                2.5 * (0.003 * f + 0.7 * b).sin()
            } else {
                0.5 + 0.4 * (0.01 * f + b).sin()
            };
            v as f32
        })
    };
    (0..11)
        .flat_map(|b| row(b, false))
        .chain((0..11).flat_map(|b| row(b, true)))
        .collect()
}

/// `rows` `[bins, frames]` read at every `step`th frame.
fn every(rows: &[f32], bins: usize, frames: usize, step: usize) -> Vec<Vec<f32>> {
    (0..bins)
        .map(|b| {
            (0..frames)
                .step_by(step)
                .map(|f| rows[b * frames + f])
                .collect()
        })
        .collect()
}

/// The script's excitation: two sines at 150 and 450 Hz plus `hiss`, in
/// f64, as f32.
fn source(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            let tones = 0.3 * (2.0 * PI * 150.0 * t / 24000.0).sin()
                + 0.1 * (2.0 * PI * 450.0 * t / 24000.0).sin();
            (tones + 0.2 * hiss(i as u64)) as f32
        })
        .collect()
}

/// The script's voiced contour: 120 Hz plus 40 Hz of slow swing.
fn sine_f0(n: usize) -> Vec<f32> {
    (0..n)
        .map(|t| (120.0 + 40.0 * (2.0 * PI * t as f64 / n as f64).sin()) as f32)
        .collect()
}

/// Panics unless `got` and `want` have one length and differ by under
/// [`REL_TOL`] of `want`'s largest magnitude anywhere.
fn assert_close(got: &[f32], want: &[f32], what: &str) {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert_within(got, want, REL_TOL * scale, what);
}

/// Panics unless `got` and `want` have one length and differ by `tol` at
/// most anywhere.
fn assert_within(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let (at, diff) = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .enumerate()
        .fold(
            (0, 0f32),
            |best, (i, d)| if d > best.1 { (i, d) } else { best },
        );
    assert!(
        diff <= tol,
        "{what}: value {at} is {} against the reference {}, off by {diff} (allowed {tol})",
        got[at],
        want[at],
    );
}

fn values(t: &Tensor<CpuRuntime>) -> Vec<f32> {
    t.contiguous().expect("contiguous").to_vec()
}

#[test]
fn kokoro_matches_the_reference_on_every_deterministic_stage() {
    let Some(dir) = model_fixture("KOKORO_DIR", "Kokoro-82M") else {
        skip_notice("Kokoro-82M model directory", "KOKORO_DIR");
        return;
    };
    let fx: Fixture = load_json(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/kokoro_audio_parity.json"),
    );
    let device = CpuDevice::new();
    let client: CpuClient = CpuRuntime::default_client(&device);
    let model: KokoroModelV2<CpuRuntime> =
        load_kokoro_v2::<CpuRuntime, _>(&client, &dir, &device).expect("loading Kokoro");
    let pack = load_voice_pt::<CpuRuntime>(dir.join(format!("voices/{}.pt", fx.voice)), &device)
        .expect("loading the voice pack");
    let row = select_voice_style(&pack, fx.phonemes.chars().count()).expect("voice row");
    let ids = Tensor::<CpuRuntime>::from_slice(&fx.ids, &[1, fx.ids.len()], &device).expect("ids");

    for s in &fx.speeds {
        // Durations and sample count draw no random number, so the seed is
        // irrelevant here and stays unset.
        let opts = KokoroSynthOptions::with_speed(s.speed);
        let (_, _, durations) = model
            .forward_to_spectrogram(&client, &ids, &row, &opts)
            .expect("spectrogram");
        assert_eq!(durations, s.durations, "durations at speed {}", s.speed);
        let wave = model
            .synthesize(&client, &ids, &row, &opts)
            .expect("synthesis");
        assert_eq!(
            wave.shape().iter().product::<usize>(),
            s.samples,
            "samples at speed {}",
            s.speed
        );
    }

    let (dec_style, pred_style) = split_voice_style(&row, model.config.style_dim).expect("style");
    let d_en = model.bert.forward(&client, &ids).expect("bert");
    let (logits, features) = model
        .predictor
        .predict_duration_features(&client, &d_en, &pred_style)
        .expect("durations");
    let shape = logits.shape().to_vec();
    let durations =
        decode_prosody_durations(&values(&logits), shape[1], shape[2], 1.0).expect("decode");
    let frames = length_regulator(&client, &features, &durations).expect("regulate");
    let (f0, n) = model
        .predictor
        .predict_f0_n(&client, &frames, &pred_style)
        .expect("F0 and energy");
    assert_close(&values(&f0), &fx.f0, "F0");
    assert_close(&values(&n), &fx.n, "energy");

    let t_en = model
        .text_encoder
        .forward(&client, &ids)
        .expect("text encoder");
    let asr = length_regulator(&client, &t_en, &durations)
        .expect("regulate")
        .transpose(1, 2)
        .expect("transpose")
        .contiguous()
        .expect("contiguous");
    let trunk = model
        .decoder
        .trunk(&client, &asr, &f0, &n, &dec_style)
        .expect("decoder trunk");
    let width = trunk.shape()[2];
    let trunk_values = values(&trunk);
    for (&c, want) in fx.trunk_channels.iter().zip(&fx.trunk) {
        assert_close(
            &trunk_values[c * width..(c + 1) * width],
            want,
            &format!("trunk channel {c}"),
        );
    }

    let sp = &fx.spectrum;
    assert_eq!(width * 300, sp.source_len, "excitation length");
    let src =
        Tensor::<CpuRuntime>::from_slice(&source(sp.source_len), &[1, sp.source_len], &device)
            .expect("source");
    let har = model
        .decoder
        .generator
        .source_spectrum(&client, &src)
        .expect("spectrum");
    let bins = sp.mag.len();
    assert_eq!(har.shape(), [1, 2 * bins, sp.frames], "spectrum shape");
    let har_values = values(&har);
    let (mag, phase) = har_values.split_at(bins * sp.frames);
    let mag = every(mag, bins, sp.frames, sp.frame_step);
    let phase = every(phase, bins, sp.frames, sp.frame_step);
    let peak = sp.mag.iter().flatten().fold(0f32, |m, v| m.max(*v));
    for b in 0..bins {
        assert_close(&mag[b], &sp.mag[b], &format!("spectrum magnitude bin {b}"));
        for (f, ((got, want), m)) in phase[b]
            .iter()
            .zip(&sp.phase[b])
            .zip(&sp.mag[b])
            .enumerate()
        {
            if *m < PHASE_FLOOR * peak {
                continue;
            }
            let d = (got - want).rem_euclid(std::f32::consts::TAU);
            let d = d.min(std::f32::consts::TAU - d);
            assert!(
                d <= PHASE_TOL,
                "spectrum phase bin {b} frame {}: {got} against the reference {want}",
                f * sp.frame_step
            );
        }
    }

    let g = &fx.generator;
    let frames = width * 60 + 1;
    let given = Tensor::<CpuRuntime>::from_slice(&spectrum(frames), &[1, 22, frames], &device)
        .expect("given spectrum");
    let (mag, phase) = model
        .decoder
        .generator
        .forward_with_spectrum(&client, &trunk, &dec_style, &given)
        .expect("generator");
    assert_eq!(mag.shape()[2], g.frames, "generator frames");
    for (name, got, want) in [("magnitude", &mag, &g.mag), ("phase", &phase, &g.phase)] {
        let rows = every(&values(got), want.len(), g.frames, g.frame_step);
        for (bin, (got_row, want_row)) in rows.iter().zip(want).enumerate() {
            assert_close(got_row, want_row, &format!("generator {name} bin {bin}"));
        }
    }

    let s = &fx.sine;
    let mut sine = SineGen::new(24000.0, 8).with_upsample_scale(300);
    sine.noise_std = 0.0;
    sine.voiced_threshold = 10.0;
    let contour = Tensor::<CpuRuntime>::from_slice(&sine_f0(s.len), &[1, s.len, 1], &device)
        .expect("contour");
    let sines = values(&sine.forward(&client, &contour, None).expect("sine"));
    let harmonics = s.values.first().map_or(0, Vec::len);
    let got: Vec<f32> = (0..s.len)
        .step_by(s.step)
        .flat_map(|t| sines[t * harmonics..(t + 1) * harmonics].to_vec())
        .collect();
    let want: Vec<f32> = s.values.iter().flatten().copied().collect();
    // A sine's f32 argument is exact to an ulp of the phase it reaches, so
    // the tolerance is the amplitude times two ulps of the largest phase:
    // the top harmonic's whole cycle count, in radians.
    let cycles: f64 = sine_f0(s.len)
        .iter()
        .map(|&f| f64::from(f) * harmonics as f64 / 24000.0)
        .sum();
    let tol = sine.sine_amp * 2.0 * f32::EPSILON * (2.0 * PI * cycles) as f32;
    assert_within(&got, &want, tol, "sine source");
}
