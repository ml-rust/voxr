//! VoxCPM2 through `speak_script`: a short script spoken in a voice
//! designed from a description, the same script cloned from a recording,
//! and one Hi-Fi take cloned with the recording's transcript. Each part is
//! checked by Whisper in the take loop, and the voiceover on disk is heard
//! back.
//!
//! Needs `VOXCPM2_GGUF`, a compressr GGUF that embeds the VAE and the
//! tokenizer, `VOXCPM2_REF_WAV`, a speech recording to clone, and
//! `WHISPER_BASE_DIR` (or `whisper-base` under `BOOSTR_MODELS_DIR`).
//! `VOXCPM2_VOICES_DIR`, optional, is the engine's voices directory. With
//! the `cuda` feature the engine runs on device 0 with an F16 decoder, as
//! voxr's own VoxCPM2 tests run it; without it, on the CPU at F32. Skips
//! with a notice when a model is missing.

mod common;

use std::path::{Path, PathBuf};

use boostr::model::audio::voxcpm::VoxCpm2Weights;
use common::models::{
    Rt, RunDir, assert_heard_back, assert_spoken, device, hearing, model_path, skip, speak,
    whisper_base,
};
use numr::runtime::Runtime;
use voxr::voxcpm::{VoxCpm2Engine, VoxCpm2LoadOptions, VoxCpm2SynthOptions};
use voxr_narrate::{NarrationRuns, SpeakOpts, VoxcpmSpeech};

const SCRIPT: &str = "The river runs north past the old mill, and the town wakes slowly.";
const LANGUAGE: Option<&str> = Some("en");

fn load_engine(gguf: &Path, voices: Option<&Path>) -> VoxCpm2Engine<Rt> {
    let weights = VoxCpm2Weights::Gguf {
        path: gguf.to_path_buf(),
        config: None,
    };
    let device = device();
    let client = std::sync::Arc::new(Rt::default_client(&device));
    let vae_decoder_dtype = cfg!(feature = "cuda").then_some(numr::dtype::DType::F16);
    let options = VoxCpm2LoadOptions {
        dtype: None,
        vae_decoder_dtype,
        synth: VoxCpm2SynthOptions::default(),
        adapter: None,
    };
    VoxCpm2Engine::<Rt>::load(&weights, None, voices, &device, client, options)
        .unwrap_or_else(|e| panic!("loading VoxCPM2 from {}: {e}", gguf.display()))
}

/// The model, the recording and the Whisper checkpoint, or a skip notice.
fn fixtures() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let Some(gguf) = model_path("VOXCPM2_GGUF", "") else {
        skip("VoxCPM2 GGUF", "VOXCPM2_GGUF");
        return None;
    };
    let Some(recording) = model_path("VOXCPM2_REF_WAV", "") else {
        skip("reference recording", "VOXCPM2_REF_WAV");
        return None;
    };
    let base = whisper_base()?;
    Some((gguf, recording, base))
}

#[test]
fn voxcpm_designs_clones_and_clones_hifi() {
    let Some((gguf, recording, base)) = fixtures() else {
        return;
    };
    let voices = model_path("VOXCPM2_VOICES_DIR", "");
    let engine = load_engine(&gguf, voices.as_deref());
    let checkpoint = gguf.to_string_lossy().to_string();
    let reference = recording.to_string_lossy().to_string();
    let mut runs = NarrationRuns::new(hearing(&base), |_: &str| Ok(Some(0)))
        .with_voxcpm(VoxcpmSpeech::new(&engine));
    let base_opts = SpeakOpts {
        engine: Some("voxcpm"),
        language: LANGUAGE,
        seed: Some(7.0),
        ..SpeakOpts::default()
    };

    let designed = RunDir::new();
    let opts = SpeakOpts {
        voice: Some("(a calm adult narrator, warm, clear)"),
        ..base_opts
    };
    let spoken = speak(SCRIPT, &opts, &designed, Some(&checkpoint), &mut runs);
    assert_spoken(&spoken);
    assert!(
        matches!(&spoken.voice_file, Some(Some(file)) if file.is_file()),
        "no designed voice beside the voiceover: {:?}",
        spoken.voice_file
    );
    assert_heard_back(runs.hearing(), &designed, SCRIPT, LANGUAGE);

    let cloned = RunDir::new();
    let opts = SpeakOpts {
        reference: Some(&reference),
        ..base_opts
    };
    let spoken = speak(SCRIPT, &opts, &cloned, Some(&checkpoint), &mut runs);
    assert_spoken(&spoken);
    assert_eq!(spoken.voice_file, Some(None));
    assert_heard_back(runs.hearing(), &cloned, SCRIPT, LANGUAGE);

    let hifi = RunDir::new();
    let opts = SpeakOpts {
        reference: Some(&reference),
        hifi: Some(true),
        ..base_opts
    };
    let spoken = speak(SCRIPT, &opts, &hifi, Some(&checkpoint), &mut runs);
    assert_spoken(&spoken);
    assert_heard_back(runs.hearing(), &hifi, SCRIPT, LANGUAGE);
}
