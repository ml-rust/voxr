//! Parity of voxr's CLAP scoring with transformers' `ClapProcessor` and
//! `ClapModel` (`laion/larger_clap_general`): token ids, log-mel input
//! features, audio and text embeddings, the score matrix, and the sound
//! review's ranking of the catalog's base variants.
//!
//! The fixture comes from `tests/tools/clap/make_clap_fixture.py`. The test
//! reads the checkpoint from `$CLAP_DIR` and the fixture from
//! `$CLAP_FIXTURE_DIR`, and prints a notice and passes without checking when
//! either is unset. The CUDA case runs with the `cuda` feature.
//!
//! Token ids, input features and ranks must match exactly. Embeddings and
//! scores match to the bounds below, on both runtimes: numr and PyTorch
//! order the sums of every matmul, softmax and norm differently, and the
//! differences compound through the towers' layers.

#![cfg(feature = "clap")]

use std::path::{Path, PathBuf};

use boostr::format::SafeTensors;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use serde::Deserialize;
use voxr::clap::ClapBundle;
use voxr::sound_review::{Rank, model_clip, rank};
use voxr::wav::decode_wav_f64;

/// Largest difference of an embedding component.
const EMBED_ABS: f32 = 2e-6;

/// Largest difference of a score.
const SCORE_ABS: f32 = 5e-6;

#[derive(Deserialize)]
struct RankRef {
    label: String,
    /// Index in the texts of the clip's own `says` text.
    own: usize,
    rank: usize,
    heard: String,
}

#[derive(Deserialize)]
struct Meta {
    clips: Vec<PathBuf>,
    base: Vec<String>,
    texts: Vec<String>,
    says_texts: usize,
    token_ids: Vec<Vec<u32>>,
    ranks: Vec<RankRef>,
}

struct Fixture {
    meta: Meta,
    features: Vec<f32>,
    audio: Vec<f32>,
    text: Vec<f32>,
    scores: Vec<f32>,
}

fn dirs() -> Option<(PathBuf, PathBuf)> {
    let model = std::env::var_os("CLAP_DIR").map(PathBuf::from);
    let fixture = std::env::var_os("CLAP_FIXTURE_DIR").map(PathBuf::from);
    match (model, fixture) {
        (Some(m), Some(f)) if m.exists() && f.exists() => Some((m, f)),
        _ => {
            eprintln!(
                "skipping: CLAP checkpoint or fixture unavailable; set CLAP_DIR (a \
                 laion/larger_clap_general snapshot) and CLAP_FIXTURE_DIR \
                 (tests/tools/clap/make_clap_fixture.py writes it)"
            );
            None
        }
    }
}

fn load_fixture(dir: &Path) -> Fixture {
    let meta: Meta = serde_json::from_str(
        &std::fs::read_to_string(dir.join("clap_fixture.json")).expect("clap_fixture.json"),
    )
    .expect("parse clap_fixture.json");
    let mut st = SafeTensors::open(dir.join("clap_fixture.safetensors")).expect("safetensors");
    let device = CpuDevice::new();
    let mut take = |name: &str| -> Vec<f32> {
        st.load_tensor::<CpuRuntime>(name, &device)
            .expect(name)
            .to_vec()
    };
    Fixture {
        features: take("input_features"),
        audio: take("audio_embeds"),
        text: take("text_embeds"),
        scores: take("scores"),
        meta,
    }
}

fn max_abs(a: &[f32], b: &[f32], what: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{what}: length");
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn check<R: Runtime<DType = DType>>(device: &R::Device, label: &str)
where
    R::Client: boostr::model::audio::clap::ClapClient<R> + boostr::quant::traits::DequantOps<R>,
{
    let Some((model_dir, fixture_dir)) = dirs() else {
        return;
    };
    let fx = load_fixture(&fixture_dir);
    let bundle = ClapBundle::<R>::from_dir_on(&model_dir, device).expect("load CLAP");
    let m = &fx.meta;

    for (text, want) in m.texts.iter().zip(&m.token_ids) {
        assert_eq!(&bundle.token_ids(text), want, "token ids of {text:?}");
    }

    let clips: Vec<Vec<f32>> = m
        .clips
        .iter()
        .map(|p| {
            model_clip(&decode_wav_f64(&std::fs::read(p).expect("wav")).expect("decode"))
                .expect("clip")
        })
        .collect();
    let features: Vec<f32> = bundle
        .features
        .features::<CpuRuntime>(&clips, &CpuDevice::new())
        .expect("features")
        .to_vec();
    let feat_diff = max_abs(&features, &fx.features, "features");
    let feat_exact = features
        .iter()
        .zip(&fx.features)
        .filter(|(a, b)| a == b)
        .count();

    let audio: Vec<f32> = bundle.audio_embeddings(&clips).expect("audio").to_vec();
    let text: Vec<f32> = bundle.text_embeddings(&m.texts).expect("text").to_vec();
    let scores = bundle.scores(&clips, &m.texts).expect("scores");
    let (audio_diff, text_diff) = (
        max_abs(&audio, &fx.audio, "audio"),
        max_abs(&text, &fx.text, "text"),
    );
    let score_diff = max_abs(&scores, &fx.scores, "scores");
    eprintln!(
        "{label}: {} clips, {} texts; features {feat_exact}/{} exact, worst {feat_diff:.3e} dB; \
         worst audio {audio_diff:.3e}, text {text_diff:.3e}, score {score_diff:.3e}",
        clips.len(),
        m.texts.len(),
        features.len()
    );

    // The review's ranking: base variants against the says texts.
    let n_texts = m.texts.len();
    let says = &m.texts[..m.says_texts];
    let sub: Vec<f32> = (0..m.base.len())
        .flat_map(|i| scores[i * n_texts..i * n_texts + m.says_texts].to_vec())
        .collect();
    let own: Vec<usize> = m
        .ranks
        .iter()
        .zip(&m.base)
        .map(|(r, label)| {
            assert_eq!(&r.label, label, "fixture order");
            r.own
        })
        .collect();
    let got = rank(&sub, &own, says).expect("rank");
    for (r, want) in got.iter().zip(&m.ranks) {
        assert_eq!(
            r,
            &Rank {
                rank: want.rank,
                heard: want.heard.clone()
            },
            "{label}: rank of {}",
            want.label
        );
    }

    assert_eq!(
        feat_exact,
        features.len(),
        "{label}: features differ, by up to {feat_diff} dB"
    );
    assert!(
        audio_diff <= EMBED_ABS,
        "{label}: audio embeddings differ by {audio_diff}"
    );
    assert!(
        text_diff <= EMBED_ABS,
        "{label}: text embeddings differ by {text_diff}"
    );
    assert!(
        score_diff <= SCORE_ABS,
        "{label}: scores differ by {score_diff}"
    );
}

#[test]
fn cpu_matches_transformers() {
    check::<CpuRuntime>(&CpuDevice::new(), "cpu");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_transformers() {
    use numr::runtime::cuda::{CudaDevice, CudaRuntime};
    check::<CudaRuntime>(&CudaDevice::new(0), "cuda");
}
