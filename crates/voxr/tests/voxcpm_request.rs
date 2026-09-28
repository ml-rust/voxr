//! The request-level path of [`VoxCpm2Engine`]: a seed per request, a
//! reference encoded once from caller-held samples, and zero-shot, each
//! checked against the audio the id-based path renders.
//!
//! Gated on a real model: set `VOXCPM2_GGUF` to a compressr GGUF that embeds
//! the VAE and the tokenizer (the test skips when it is unset or the path is
//! absent). `VOXCPM2_VOICES_DIR`, optional, points at a directory of
//! reference recordings; the request-reference test needs it and skips
//! without it. With the `cuda` feature the engine runs on device 0 with an
//! F16 decoder, the configuration blazr serves; without it, on the CPU at
//! F32.

#![cfg(feature = "voxcpm")]

mod common;

use std::path::{Path, PathBuf};

use common::voxcpm::{Rt, WINDOW, assert_bit_identical, gguf_fixture, voices_fixture};
use voxr::TtsEngine;
use voxr::voxcpm::{
    SynthRequest, VoiceSource, VoxCpm2Engine, VoxCpm2SynthOptions, ZERO_SHOT_VOICE_ID, paren,
};

const TEXT: &str = "Selamat pagi, today we test one request at a time.";

/// A second prompt, so the batched request shares its batch with a row of a
/// different length as well as a different seed.
const OTHER_TEXT: &str = "Satu dua tiga.";

/// The load-time seed. Not the default, so a path that ignores it shows.
const LOAD_SEED: u64 = 7;

/// Two request seeds, the second a retry-style offset of the first.
const SEED_A: u64 = 11;
const SEED_B: u64 = SEED_A + 1000;

/// Containers the engine reads from a voices directory.
const VOICE_EXTENSIONS: [&str; 4] = ["wav", "flac", "mp3", "ogg"];

fn load_engine(gguf: &Path, voices: Option<&Path>) -> VoxCpm2Engine<Rt> {
    common::voxcpm::load_engine(
        gguf,
        voices,
        VoxCpm2SynthOptions {
            seed: LOAD_SEED,
            batch_window: WINDOW,
            max_batch: 8,
            ..VoxCpm2SynthOptions::default()
        },
    )
}

fn render(engine: &VoxCpm2Engine<Rt>, request: &SynthRequest<'_, Rt>) -> Vec<f32> {
    let samples = engine.render_request(request).expect("render_request");
    assert!(!samples.is_empty(), "a render returned no samples");
    assert!(
        samples.iter().all(|v| v.is_finite()),
        "a render has non-finite samples"
    );
    samples
}

/// The file behind voice `id`, found the way the engine lists voices: an
/// audio file in `dir` whose stem is `id`.
fn voice_file(dir: &Path, id: &str) -> PathBuf {
    std::fs::read_dir(dir)
        .expect("read voices directory")
        .map(|entry| entry.expect("voices directory entry").path())
        .find(|path| {
            path.is_file()
                && path.file_stem().and_then(|s| s.to_str()) == Some(id)
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|ext| VOICE_EXTENSIONS.iter().any(|a| ext.eq_ignore_ascii_case(a)))
        })
        .unwrap_or_else(|| panic!("no audio file for voice {id:?} in {}", dir.display()))
}

/// The same text at two seeds renders two different waveforms.
#[test]
fn different_seeds_render_different_audio() {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("VOXCPM2_GGUF unset or absent; skipping");
        return;
    };
    let engine = load_engine(&gguf, None);
    let a = render(
        &engine,
        &SynthRequest::new(TEXT, VoiceSource::ZeroShot).with_seed(SEED_A),
    );
    let b = render(
        &engine,
        &SynthRequest::new(TEXT, VoiceSource::ZeroShot).with_seed(SEED_B),
    );
    let same = a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(
        !same,
        "seeds {SEED_A} and {SEED_B} rendered identical audio"
    );
}

/// One (text, seed) renders the same samples twice alone, and again when it
/// shares a batch with a request at another seed.
#[test]
fn a_seeded_request_repeats_alone_and_batched() {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("VOXCPM2_GGUF unset or absent; skipping");
        return;
    };
    let engine = load_engine(&gguf, None);
    let request = SynthRequest::new(TEXT, VoiceSource::ZeroShot).with_seed(SEED_A);
    let neighbour = SynthRequest::new(OTHER_TEXT, VoiceSource::ZeroShot).with_seed(SEED_B);

    let first = render(&engine, &request);
    let second = render(&engine, &request);
    assert_bit_identical(&second, &first, "the same request rendered twice");

    let (batched, neighbour_batched) = std::thread::scope(|scope| {
        let target = scope.spawn(|| render(&engine, &request));
        let other = scope.spawn(|| render(&engine, &neighbour));
        (
            target.join().expect("render thread"),
            other.join().expect("render thread"),
        )
    });
    assert_bit_identical(&batched, &first, "the request batched beside another seed");

    let neighbour_alone = render(&engine, &neighbour);
    assert_bit_identical(
        &neighbour_batched,
        &neighbour_alone,
        "the neighbour batched beside the request",
    );
}

/// A reference encoded from a voices-directory file's samples renders what
/// naming that file's voice id renders, at the same seed.
#[test]
fn a_request_reference_matches_the_voices_directory_voice() {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("VOXCPM2_GGUF unset or absent; skipping");
        return;
    };
    let Some(voices) = voices_fixture() else {
        eprintln!("VOXCPM2_VOICES_DIR unset or absent; skipping");
        return;
    };
    let engine = load_engine(&gguf, Some(&voices));
    let Some(voice) = engine
        .voices()
        .into_iter()
        .find(|voice| voice.id != ZERO_SHOT_VOICE_ID)
    else {
        eprintln!("{} holds no voice; skipping", voices.display());
        return;
    };

    let path = voice_file(&voices, &voice.id);
    let bytes = std::fs::read(&path).expect("read voice file");
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(voxr::extension_hint);
    let data = voxr::decode_audio(&bytes, hint).expect("decode voice file");
    let mono = voxr::resample::to_mono_at_rate(&data, data.sample_rate).expect("downmix");
    let reference = engine
        .encode_reference(&mono, data.sample_rate)
        .expect("encode_reference");
    assert!(reference.patches() > 0, "the reference holds no patches");

    let by_reference = render(
        &engine,
        &SynthRequest::new(TEXT, VoiceSource::Reference(&reference)).with_seed(SEED_A),
    );
    let by_id = render(
        &engine,
        &SynthRequest::new(TEXT, VoiceSource::Named(&voice.id)).with_seed(SEED_A),
    );
    assert_bit_identical(&by_reference, &by_id, &format!("voice {}", voice.id));

    // The style prefix is text: the same composed text through either voice
    // source still matches.
    let styled = format!("{}{TEXT}", paren("relaxed, explaining"));
    let styled_reference = render(
        &engine,
        &SynthRequest::new(&styled, VoiceSource::Reference(&reference)).with_seed(SEED_B),
    );
    let styled_id = render(
        &engine,
        &SynthRequest::new(&styled, VoiceSource::Named(&voice.id)).with_seed(SEED_B),
    );
    assert_bit_identical(&styled_reference, &styled_id, "styled request");
}

/// Zero-shot through the request path, at the load-time seed, renders what
/// `synthesize` with [`ZERO_SHOT_VOICE_ID`] renders.
#[test]
fn a_zero_shot_request_matches_the_id_path() {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("VOXCPM2_GGUF unset or absent; skipping");
        return;
    };
    let engine = load_engine(&gguf, None);
    let old = engine
        .synthesize(TEXT, ZERO_SHOT_VOICE_ID, 1.0)
        .expect("synthesize");

    let default_seed = render(&engine, &SynthRequest::new(TEXT, VoiceSource::ZeroShot));
    assert_bit_identical(&default_seed, &old, "zero-shot at the default seed");

    let named_seed = render(
        &engine,
        &SynthRequest::new(TEXT, VoiceSource::ZeroShot).with_seed(LOAD_SEED),
    );
    assert_bit_identical(&named_seed, &old, "zero-shot at the load seed named");

    let by_id = render(
        &engine,
        &SynthRequest::new(TEXT, VoiceSource::Named(ZERO_SHOT_VOICE_ID)),
    );
    assert_bit_identical(&by_id, &old, "the zero-shot id through the request path");
}
