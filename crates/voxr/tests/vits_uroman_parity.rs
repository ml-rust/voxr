//! Parity of the uroman step of [`VitsTokenizer`] with the MMS-TTS recipe:
//! Python `uroman` 1.3.1.1 `romanize_string(text)`, then transformers'
//! `VitsTokenizer`, then `VitsModel`.
//!
//! Fixtures come from `tests/tools/uroman/uroman_fixture.py` and are not
//! checked in. Every test skips unless `UROMAN_FIXTURE_DIR` points at the
//! generator's output. The id and waveform tests also need the
//! checkpoints: `MMS_TTS_KOR_DIR` and `MMS_TTS_AMH_DIR`, else
//! `$BOOSTR_MODELS_DIR/facebook/mms-tts-kor` and `.../mms-tts-amh`.
//!
//! * Romanization: [`romanize`] equals Python uroman's string exactly.
//! * Ids: [`VitsTokenizer::encode`] equals the recipe's ids exactly.
//! * Waveform: the ids, replayed with transformers' captured noise draws
//!   through [`VitsModel::synthesize_with_noise`], give transformers'
//!   waveform above the signal-to-error floor of boostr's
//!   `vits_synthesis_parity.rs`, and its `sequence_lengths` exactly.

#![cfg(feature = "vits")]

use std::path::{Path, PathBuf};

use boostr::model::audio::vits::{VitsModel, VitsSynthOptions};
use boostr::model::config::VitsConfig;
use boostr::nn::{VarBuilder, VarMap};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use serde::Deserialize;
use voxr::tts::vits::romanize;
use voxr::tts::{VitsEngine, VitsTokenizer};

mod common;
use common::{load_json, model_fixture, skip_notice};

/// The floor of boostr's `vits_synthesis_parity.rs`: f32 drift lands above
/// it, and a wrong formula lands far below it.
const WAVEFORM_MIN_SNR_DB: f64 = 80.0;

/// The `is_uroman` checkpoints the fixture covers: fixture subdirectory,
/// hub id, env var.
const CHECKPOINTS: [(&str, &str, &str); 2] = [
    ("mms-tts-kor", "facebook/mms-tts-kor", "MMS_TTS_KOR_DIR"),
    ("mms-tts-amh", "facebook/mms-tts-amh", "MMS_TTS_AMH_DIR"),
];

#[derive(Deserialize)]
struct Romanized {
    text: String,
    romanized: String,
}

#[derive(Deserialize)]
struct Tokens {
    text: String,
    input_ids: Vec<u32>,
}

fn fixture_dir() -> Option<PathBuf> {
    let dir = std::env::var("UROMAN_FIXTURE_DIR")
        .ok()
        .filter(|d| !d.is_empty())
        .map(PathBuf::from);
    if dir.is_none() {
        eprintln!("skipping: uroman fixtures unavailable; set UROMAN_FIXTURE_DIR");
    }
    dir
}

fn checkpoint(hub_id: &str, var: &str) -> Option<PathBuf> {
    let dir = model_fixture(var, hub_id);
    if dir.is_none() {
        skip_notice(&format!("{hub_id} checkpoint"), var);
    }
    dir
}

fn assert_no_failures(what: &str, failures: &[String], total: usize) {
    assert!(
        failures.is_empty(),
        "{what}: {} of {total} cases mismatched:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn romanization_matches_python_uroman() {
    let Some(dir) = fixture_dir() else {
        return;
    };
    let cases: Vec<Romanized> = load_json(&dir.join("romanization.json"));
    assert!(!cases.is_empty(), "romanization.json holds no cases");
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|case| {
            let got = romanize(&case.text);
            (got != case.romanized).then(|| {
                format!(
                    "text {:?}: got {got:?}, want {:?}",
                    case.text, case.romanized
                )
            })
        })
        .collect();
    assert_no_failures("romanization", &failures, cases.len());
}

#[test]
fn token_ids_match_the_mms_recipe() {
    let Some(dir) = fixture_dir() else {
        return;
    };
    for (name, hub_id, var) in CHECKPOINTS {
        let Some(ckpt) = checkpoint(hub_id, var) else {
            continue;
        };
        let tokenizer = VitsTokenizer::from_dir(&ckpt).expect("load VitsTokenizer");
        assert!(tokenizer.is_uroman(), "{hub_id} sets is_uroman: true");
        let cases: Vec<Tokens> = load_json(&dir.join(name).join("tokens.json"));
        assert!(!cases.is_empty(), "{name}/tokens.json holds no cases");
        let failures: Vec<String> = cases
            .iter()
            .filter_map(|case| {
                let got = tokenizer
                    .encode(&case.text)
                    .unwrap_or_else(|e| panic!("{hub_id}: encoding {:?}: {e}", case.text));
                (got != case.input_ids).then(|| {
                    format!(
                        "text {:?}: got {got:?}, want {:?}",
                        case.text, case.input_ids
                    )
                })
            })
            .collect();
        assert_no_failures(hub_id, &failures, cases.len());
    }
}

fn snr_db(ours: &[f32], reference: &[f32]) -> f64 {
    let signal: f64 = reference.iter().map(|y| f64::from(*y).powi(2)).sum();
    let error: f64 = ours
        .iter()
        .zip(reference)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum();
    10.0 * (signal / error).log10()
}

fn check_waveforms(hub_id: &str, ckpt: &Path, waveform_dir: &Path) {
    // The engine loads the checkpoint: an `is_uroman` one is no longer
    // refused.
    VitsEngine::from_dir(ckpt).unwrap_or_else(|e| panic!("{hub_id}: VitsEngine::from_dir: {e}"));

    let tokenizer = VitsTokenizer::from_dir(ckpt).expect("load VitsTokenizer");
    let config = VitsConfig::from_json_file(ckpt.join("config.json"))
        .unwrap_or_else(|e| panic!("{hub_id}: config.json: {e}"));
    config
        .validate()
        .expect("checkpoint config passes validation");
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let mut weights =
        VarMap::<CpuRuntime>::from_safetensors(ckpt.join("model.safetensors"), &device)
            .unwrap_or_else(|e| panic!("{hub_id}: model.safetensors: {e}"));
    let mut vb = VarBuilder::new(&mut weights, &device);
    let model = VitsModel::<CpuRuntime>::from_varbuilder(&client, &mut vb, &config)
        .unwrap_or_else(|e| panic!("{hub_id}: VitsModel: {e}"));
    let opts = VitsSynthOptions::from_config(&config);

    let index: serde_json::Value = load_json(&waveform_dir.join("index.json"));
    let sentences = index["sentences"]
        .as_array()
        .expect("index.json `sentences` is an array");
    assert!(
        !sentences.is_empty(),
        "{hub_id}: index.json lists no sentences"
    );

    for sentence in sentences {
        let text = sentence["text"].as_str().expect("sentence has `text`");
        let file = sentence["file"].as_str().expect("sentence has `file`");
        let path = waveform_dir.join(file);
        let mut fx = VarMap::<CpuRuntime>::from_safetensors(&path, &device)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut take = |name: &str| {
            fx.take_tensor(name)
                .unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()))
        };
        let input_ids = take("input_ids");
        let attention_mask = take("attention_mask");
        let duration_noise = take("noise_randn_0");
        let prior_noise = take("noise_randn_like_1");
        let ref_waveform = take("waveform");
        let ref_lengths = take("sequence_lengths");

        let want_ids: Vec<i64> = input_ids.contiguous().unwrap().to_vec();
        let got_ids: Vec<i64> = tokenizer
            .encode(text)
            .unwrap_or_else(|e| panic!("{hub_id}: encoding {text:?}: {e}"))
            .into_iter()
            .map(i64::from)
            .collect();
        assert_eq!(got_ids, want_ids, "{hub_id} {text:?}: input ids");

        let out = model
            .synthesize_with_noise(
                &client,
                &input_ids,
                &attention_mask,
                &opts,
                &duration_noise,
                &prior_noise,
            )
            .unwrap_or_else(|e| panic!("{hub_id} {text:?}: synthesize_with_noise: {e}"));
        assert_eq!(
            out.waveform.shape(),
            ref_waveform.shape(),
            "{hub_id} {text:?}: waveform shape"
        );
        let ours: Vec<f32> = out.waveform.contiguous().unwrap().to_vec();
        let reference: Vec<f32> = ref_waveform.contiguous().unwrap().to_vec();
        let snr = snr_db(&ours, &reference);
        eprintln!("{hub_id} {text:?}: waveform snr {snr:.1} dB");
        assert!(
            snr >= WAVEFORM_MIN_SNR_DB,
            "{hub_id} {text:?}: waveform snr {snr:.1} dB < {WAVEFORM_MIN_SNR_DB} dB"
        );
        let got_lengths: Vec<i64> = out.sequence_lengths.contiguous().unwrap().to_vec();
        let want_lengths: Vec<i64> = ref_lengths.contiguous().unwrap().to_vec();
        assert_eq!(
            got_lengths, want_lengths,
            "{hub_id} {text:?}: sequence_lengths"
        );
    }
}

#[test]
fn waveform_matches_transformers() {
    let Some(dir) = fixture_dir() else {
        return;
    };
    for (name, hub_id, var) in CHECKPOINTS {
        let Some(ckpt) = checkpoint(hub_id, var) else {
            continue;
        };
        check_waveforms(hub_id, &ckpt, &dir.join(name).join("waveform"));
    }
}
