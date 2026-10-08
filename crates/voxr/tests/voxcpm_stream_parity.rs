//! Streamed and buffered VoxCPM2 renders of one request are the same
//! waveform: the concatenation of every `synthesize_stream` chunk equals
//! the request's first take bit for bit, which is `synthesize`'s output
//! when that take is not retried.
//!
//! Gated on a real model: set `VOXCPM2_GGUF` to a compressr GGUF that embeds
//! the VAE and the tokenizer (the test skips when it is unset or the path is
//! absent). `VOXCPM2_VOICES_DIR`, optional, points at a directory of
//! reference recordings; when set, one prompt renders with the first voice
//! it lists as well as zero-shot. With the `cuda` feature the engine runs on
//! device 0 with an F16 decoder, the configuration blazr serves; without it,
//! on the CPU at F32.

#![cfg(feature = "voxcpm")]

mod common;

use std::path::Path;
use std::sync::Arc;

use common::voxcpm::{Rt, first_take, gguf_fixture, voices_fixture};
use voxr::TtsEngine;
use voxr::voxcpm::{VoxCpm2Engine, VoxCpm2SynthOptions, ZERO_SHOT_VOICE_ID};

const PROMPTS: [&str; 2] = [
    "Selamat pagi, today we test streaming synthesis.",
    "The quick brown fox jumps over the lazy dog, dan kemudian ia tidur di bawah pokok.",
];

fn load_engine(gguf: &Path, voices: Option<&Path>, chunk: usize) -> Arc<VoxCpm2Engine<Rt>> {
    Arc::new(common::voxcpm::load_engine(
        gguf,
        voices,
        VoxCpm2SynthOptions {
            stream_chunk_patches: chunk,
            ..VoxCpm2SynthOptions::default()
        },
    ))
}

/// Render `text` both ways and return `(first take, streamed chunks)`. A
/// stream is one take, so the buffered side keeps its first take too.
fn render_both(engine: &VoxCpm2Engine<Rt>, text: &str, voice: &str) -> (Vec<f32>, Vec<Vec<f32>>) {
    let buffered = first_take(engine, text, voice);
    let mut chunks: Vec<Vec<f32>> = Vec::new();
    engine
        .synthesize_stream(text, voice, 1.0, &mut |samples: &[f32]| {
            chunks.push(samples.to_vec());
            Ok(())
        })
        .expect("synthesize_stream");
    (buffered, chunks)
}

fn assert_bit_identical(buffered: &[f32], chunks: &[Vec<f32>], label: &str) {
    let streamed: Vec<f32> = chunks.concat();
    assert_eq!(
        streamed.len(),
        buffered.len(),
        "{label}: streamed {} samples, buffered {}",
        streamed.len(),
        buffered.len()
    );
    let first_mismatch = streamed
        .iter()
        .zip(buffered)
        .position(|(s, b)| s.to_bits() != b.to_bits());
    if let Some(i) = first_mismatch {
        let n = streamed
            .iter()
            .zip(buffered)
            .filter(|(s, b)| s.to_bits() != b.to_bits())
            .count();
        let max = streamed
            .iter()
            .zip(buffered)
            .fold(0.0f32, |m, (s, b)| m.max((s - b).abs()));
        panic!(
            "{label}: {n} of {} samples differ, first at {i} \
             (streamed {} vs buffered {}), max abs diff {max}",
            buffered.len(),
            streamed[i],
            buffered[i]
        );
    }
}

#[test]
fn streamed_chunks_concatenate_to_the_buffered_render() {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("skipping: set VOXCPM2_GGUF to a compressr VoxCPM2 GGUF");
        return;
    };
    let voices = voices_fixture();
    let engine = load_engine(&gguf, voices.as_deref(), 4);

    for text in PROMPTS {
        let (buffered, chunks) = render_both(engine.as_ref(), text, ZERO_SHOT_VOICE_ID);
        assert!(
            chunks.len() > 1,
            "{text:?}: expected several chunks, got {}",
            chunks.len()
        );
        assert_bit_identical(&buffered, &chunks, text);
    }

    let voice = engine
        .voices()
        .into_iter()
        .map(|v| v.id)
        .find(|id| id != ZERO_SHOT_VOICE_ID);
    if let Some(voice) = voice {
        let (buffered, chunks) = render_both(engine.as_ref(), PROMPTS[0], &voice);
        assert_bit_identical(&buffered, &chunks, &format!("voice {voice}"));
    } else {
        eprintln!("no reference voice loaded; set VOXCPM2_VOICES_DIR to cover the cloned path");
    }
}

#[test]
fn a_sink_error_aborts_the_stream() {
    let Some(gguf) = gguf_fixture() else {
        eprintln!("skipping: set VOXCPM2_GGUF to a compressr VoxCPM2 GGUF");
        return;
    };
    let engine = load_engine(&gguf, None, 1);
    let mut calls = 0usize;
    let err = engine
        .synthesize_stream(PROMPTS[0], ZERO_SHOT_VOICE_ID, 1.0, &mut |_: &[f32]| {
            calls += 1;
            Err(voxr::Error::ModelError {
                reason: "client gone".into(),
            })
        })
        .expect_err("a sink error must abort the render");
    assert_eq!(calls, 1, "generation must stop at the first sink error");
    assert!(err.to_string().contains("client gone"), "{err}");
}
