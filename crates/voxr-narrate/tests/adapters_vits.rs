//! A VITS (MMS-TTS) model through `speak_script`: a short Malay script
//! spoken by voxr's VITS engine as the Hugging Face model it is loaded
//! under, checked by Whisper, then heard back from the voiceover on disk.
//!
//! Needs `MMS_TTS_ZLM_DIR` (or `facebook/mms-tts-zlm` under
//! `BOOSTR_MODELS_DIR`) and `WHISPER_BASE_DIR` (or `whisper-base`). The
//! script is Malay, so the small recognizer hears it first: set
//! `WHISPER_SMALL_DIR` to a multilingual checkpoint. Skips with a notice
//! when a model is missing.

mod common;

use common::models::{RunDir, assert_heard_back, assert_spoken, hearing, model_path, skip, speak};
use voxr::tts::VitsEngine;
use voxr_narrate::{NarrationRuns, SpeakOpts, SpeakRequest, SpeechEngine, VitsSpeech, VoiceSource};

const MODEL: &str = "facebook/mms-tts-zlm";
const SCRIPT: &str = "Selamat pagi semua. Sungai itu mengalir ke utara.";
const LANGUAGE: Option<&str> = Some("ms");

fn vits() -> Option<VitsEngine> {
    let Some(dir) = model_path("MMS_TTS_ZLM_DIR", MODEL) else {
        skip("facebook/mms-tts-zlm checkpoint", "MMS_TTS_ZLM_DIR");
        return None;
    };
    Some(
        VitsEngine::from_dir(&dir)
            .unwrap_or_else(|e| panic!("loading VITS from {}: {e}", dir.display())),
    )
}

#[test]
fn a_vits_model_speaks_a_script_whisper_hears() {
    let (Some(engine), Some(base)) = (vits(), common::models::whisper_base()) else {
        return;
    };
    let speech = VitsSpeech::new(&engine).expect("the checkpoint's voice");
    let dir = RunDir::new();
    let mut runs =
        NarrationRuns::new(hearing(&base), |_: &str| Ok(Some(0))).with_model(MODEL, speech);
    let opts = SpeakOpts {
        model: Some(MODEL),
        language: LANGUAGE,
        ..SpeakOpts::default()
    };
    let spoken = speak(SCRIPT, &opts, &dir, None, &mut runs);
    assert_eq!(spoken.engine, MODEL);
    assert_spoken(&spoken);
    assert_heard_back(runs.hearing(), &dir, SCRIPT, LANGUAGE);
}

#[test]
fn a_take_follows_its_seed() {
    let Some(engine) = vits() else {
        return;
    };
    let mut speech = VitsSpeech::new(&engine).expect("the checkpoint's voice");
    let request = |seed| SpeakRequest {
        text: "Selamat pagi semua.",
        delivery: None,
        style: None,
        voice: VoiceSource::<()>::Own { design: None },
        seed,
        cfg: 1.6,
        steps: 16,
        speed: 1.0,
    };
    let a = speech.speak(&request(7.0)).expect("seed 7");
    let again = speech.speak(&request(7.0)).expect("seed 7 again");
    let b = speech.speak(&request(8.0)).expect("seed 8");
    assert_eq!(a, again, "one seed gave two takes");
    assert_ne!(a.samples, b.samples, "two seeds gave one take");
    assert_eq!(a.rate, 16_000);
}
