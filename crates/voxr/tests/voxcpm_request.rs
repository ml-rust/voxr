//! The request-level path of [`VoxCpm2Engine`]: a seed per request, a
//! reference encoded once from caller-held samples, and zero-shot, each
//! checked against the audio the id-based path renders. Hi-Fi cloning, a
//! reference plus a continuation prompt from one clip, is checked for its
//! context trim, its determinism, its stream parity and its engine check.
//!
//! Gated on a real model: set `VOXCPM2_GGUF` to a compressr GGUF that embeds
//! the VAE and the tokenizer (the test skips when it is unset or the path is
//! absent). `VOXCPM2_VOICES_DIR`, optional, points at a directory of
//! reference recordings; the request-reference and Hi-Fi tests need it and
//! skip without it. With the `cuda` feature the engine runs on device 0 with an
//! F16 decoder, the configuration blazr serves; without it, on the CPU at
//! F32.

#![cfg(feature = "voxcpm")]

mod common;

use std::path::Path;

use common::voxcpm::{
    Rt, WINDOW, assert_bit_identical, gguf_fixture, voice_samples, voices_fixture,
};
use voxr::TtsEngine;
use voxr::voxcpm::{
    Prompt, Reference, SynthRequest, VoiceSource, VoxCpm2Engine, VoxCpm2SynthOptions,
    ZERO_SHOT_VOICE_ID, paren,
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

/// Patches per streamed chunk.
const CHUNK_PATCHES: usize = 4;

/// The Hi-Fi prompt transcript. A voices directory carries no transcripts,
/// so every Hi-Fi test names this one. The checks here are structural —
/// trim, determinism, stream parity, the engine check — and hold for any
/// transcript.
const PROMPT_TEXT: &str = "This is the voice we continue from.";

fn load_engine(gguf: &Path, voices: Option<&Path>) -> VoxCpm2Engine<Rt> {
    common::voxcpm::load_engine(
        gguf,
        voices,
        VoxCpm2SynthOptions {
            seed: LOAD_SEED,
            batch_window: WINDOW,
            max_batch: 8,
            stream_chunk_patches: CHUNK_PATCHES,
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

/// Every chunk of a streamed render of `request`, in order.
fn stream(engine: &VoxCpm2Engine<Rt>, request: &SynthRequest<'_, Rt>) -> Vec<Vec<f32>> {
    let mut chunks = Vec::new();
    engine
        .render_request_stream(request, &mut |samples: &[f32]| {
            chunks.push(samples.to_vec());
            Ok(())
        })
        .expect("render_request_stream");
    chunks
}

/// An engine over the voices directory, and the first voice's clip encoded
/// as a Hi-Fi pair. `None`, after a skip notice, when a fixture is missing.
fn hifi_fixture() -> Option<(VoxCpm2Engine<Rt>, Reference<Rt>, Prompt<Rt>)> {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("VOXCPM2_GGUF unset or absent; skipping");
        return None;
    };
    let Some(voices) = voices_fixture() else {
        eprintln!("VOXCPM2_VOICES_DIR unset or absent; skipping");
        return None;
    };
    let engine = load_engine(&gguf, Some(&voices));
    let Some(voice) = engine
        .voices()
        .into_iter()
        .find(|voice| voice.id != ZERO_SHOT_VOICE_ID)
    else {
        eprintln!("{} holds no voice; skipping", voices.display());
        return None;
    };
    let (clip, rate) = voice_samples(&voices, &voice.id);
    let (reference, prompt) = engine
        .encode_reference_and_prompt(&clip, rate)
        .expect("encode_reference_and_prompt");
    assert!(reference.patches() > 0, "the reference holds no patches");
    assert_eq!(
        prompt.patches(),
        reference.patches(),
        "one clip pads to the same patch count on either side"
    );
    Some((engine, reference, prompt))
}

/// The Hi-Fi request for `text`: the clip as the voice and as the prompt.
fn hifi<'a>(
    text: &'a str,
    reference: &'a Reference<Rt>,
    prompt: &'a Prompt<Rt>,
) -> SynthRequest<'a, Rt> {
    SynthRequest::new(text, VoiceSource::Reference(reference)).with_prompt(prompt, PROMPT_TEXT)
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

    let (mono, rate) = voice_samples(&voices, &voice.id);
    let reference = engine
        .encode_reference(&mono, rate)
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

/// A Hi-Fi render holds only new speech. The prompt context the decode runs
/// behind spans up to three patches. Every streamed chunk but the last is
/// exactly [`CHUNK_PATCHES`] patches and the last is at most that, so no
/// chunk carries context. The buffered render is a whole number of patches.
#[test]
fn a_hifi_render_holds_no_prompt_audio() {
    let Some((engine, reference, prompt)) = hifi_fixture() else {
        return;
    };
    let per_patch = engine.samples_per_patch();
    let chunk = CHUNK_PATCHES * per_patch;
    let request = hifi(TEXT, &reference, &prompt).with_seed(SEED_A);

    let buffered = render(&engine, &request);
    assert_eq!(
        buffered.len() % per_patch,
        0,
        "{} samples is not a whole number of {per_patch}-sample patches",
        buffered.len()
    );

    let chunks = stream(&engine, &request);
    let Some((last, full)) = chunks.split_last() else {
        panic!("a Hi-Fi stream emitted no chunk");
    };
    for (i, samples) in full.iter().enumerate() {
        assert_eq!(
            samples.len(),
            chunk,
            "chunk {i} is not {CHUNK_PATCHES} patches"
        );
    }
    assert!(
        !last.is_empty() && last.len() <= chunk && last.len() % per_patch == 0,
        "the last chunk holds {} samples, expected 1 to {CHUNK_PATCHES} whole patches",
        last.len()
    );
}

/// One Hi-Fi request renders the same samples twice alone, and again when it
/// shares a batch with a plain reference request.
#[test]
fn a_hifi_request_repeats_alone_and_batched() {
    let Some((engine, reference, prompt)) = hifi_fixture() else {
        return;
    };
    let request = hifi(TEXT, &reference, &prompt).with_seed(SEED_A);
    let neighbour =
        SynthRequest::new(OTHER_TEXT, VoiceSource::Reference(&reference)).with_seed(SEED_B);

    let first = render(&engine, &request);
    let second = render(&engine, &request);
    assert_bit_identical(&second, &first, "the Hi-Fi request rendered twice");

    let (batched, neighbour_batched) = std::thread::scope(|scope| {
        let target = scope.spawn(|| render(&engine, &request));
        let other = scope.spawn(|| render(&engine, &neighbour));
        (
            target.join().expect("render thread"),
            other.join().expect("render thread"),
        )
    });
    assert_bit_identical(
        &batched,
        &first,
        "the Hi-Fi request batched beside a plain one",
    );

    let neighbour_alone = render(&engine, &neighbour);
    assert_bit_identical(
        &neighbour_batched,
        &neighbour_alone,
        "the plain request batched beside a Hi-Fi one",
    );
}

/// A streamed Hi-Fi render concatenates to the buffered one.
#[test]
fn a_streamed_hifi_render_matches_the_buffered_one() {
    let Some((engine, reference, prompt)) = hifi_fixture() else {
        return;
    };
    let request = hifi(TEXT, &reference, &prompt).with_seed(SEED_B);
    let buffered = render(&engine, &request);
    let streamed = stream(&engine, &request).concat();
    assert_bit_identical(&streamed, &buffered, "the streamed Hi-Fi render");
}

/// A prompt encoded by another engine is refused before it reaches a batch.
/// The first engine drops before the second loads, so only one model is
/// resident at a time.
#[test]
fn a_prompt_from_another_engine_is_refused() {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("VOXCPM2_GGUF unset or absent; skipping");
        return;
    };
    // One second of a quiet tone at the encoder's rate: any clip encodes.
    let clip: Vec<f32> = (0..16_000).map(|i| 0.1 * (i as f32 * 0.05).sin()).collect();
    let foreign = {
        let engine = load_engine(&gguf, None);
        engine.encode_prompt(&clip, 16_000).expect("encode_prompt")
    };
    let engine = load_engine(&gguf, None);
    let request = SynthRequest::new(TEXT, VoiceSource::ZeroShot).with_prompt(&foreign, PROMPT_TEXT);
    let err = engine
        .render_request(&request)
        .expect_err("a foreign prompt rendered");
    assert!(
        err.to_string().contains("different VoxCPM2 engine"),
        "{err}"
    );
}
