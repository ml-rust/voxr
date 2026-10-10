//! Kokoro through `speak_script`: a short script spoken by voxr's Kokoro
//! engine at a native speed, checked by Whisper, then heard back from the
//! voiceover on disk. Also the voice check against the real voice list,
//! and a take that follows its seed.
//!
//! Needs `KOKORO_DIR` (or `Kokoro-82M` under `BOOSTR_MODELS_DIR`) with its
//! `g2p/tagger`, and `WHISPER_BASE_DIR` (or `whisper-base`). Skips with a
//! notice when either is missing.

mod common;

use common::models::{RunDir, assert_heard_back, assert_spoken, hearing, model_path, skip, speak};
use voxr::kokoro::KokoroEngine;
use voxr_narrate::{
    KokoroSpeech, NarrationRuns, SpeakOpts, SpeakRequest, SpeechEngine, VoiceSource,
};

const SCRIPT: &str = "The river runs north past the old mill. The town wakes slowly.";

fn kokoro() -> Option<KokoroEngine> {
    let Some(dir) = model_path("KOKORO_DIR", "Kokoro-82M") else {
        skip("Kokoro-82M model directory", "KOKORO_DIR");
        return None;
    };
    Some(
        KokoroEngine::from_dir(&dir, None)
            .unwrap_or_else(|e| panic!("loading Kokoro from {}: {e}", dir.display())),
    )
}

#[test]
fn kokoro_speaks_a_script_whisper_hears() {
    let (Some(engine), Some(base)) = (kokoro(), common::models::whisper_base()) else {
        return;
    };
    let speech = KokoroSpeech::new(&engine);
    assert_eq!(speech.voices().first(), Some(&"af_heart"));
    let dir = RunDir::new();
    let mut runs = NarrationRuns::new(hearing(&base), |_: &str| Ok(Some(0))).with_kokoro(speech);
    let opts = SpeakOpts {
        voice: Some("af_heart"),
        speed: Some(1.1),
        ..SpeakOpts::default()
    };
    let spoken = speak(SCRIPT, &opts, &dir, None, &mut runs);
    assert_eq!(spoken.engine, "Kokoro (af_heart)");
    assert_eq!(spoken.phrases.as_ref().map(Vec::len), Some(2));
    assert_spoken(&spoken);
    assert_heard_back(runs.hearing(), &dir, SCRIPT, None);
}

#[test]
fn speed_is_native_and_an_unknown_voice_lists_the_voices() {
    let Some(engine) = kokoro() else {
        return;
    };
    let mut speech = KokoroSpeech::new(&engine);
    let request = |speed, name| SpeakRequest {
        text: "The river runs north past the old mill.",
        delivery: None,
        style: None,
        voice: VoiceSource::<()>::Named { name },
        seed: 7.0,
        cfg: 1.6,
        steps: 16,
        speed,
    };
    let natural = speech.speak(&request(1.0, "af_heart")).expect("speed 1");
    let fast = speech.speak(&request(1.5, "af_heart")).expect("speed 1.5");
    assert_eq!(natural.rate, 24_000);
    let ratio = natural.samples.len() as f64 / fast.samples.len() as f64;
    assert!(
        (1.3..1.7).contains(&ratio),
        "speed 1.5 made the take {ratio} times shorter"
    );
    let err = speech
        .speak(&request(1.0, "zz_nobody"))
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("Kokoro has no voice \"zz_nobody\". Voices: af_heart, af_alloy"),
        "{err}"
    );
}

#[test]
fn a_take_follows_its_seed() {
    let Some(engine) = kokoro() else {
        return;
    };
    let mut speech = KokoroSpeech::new(&engine);
    let request = |seed| SpeakRequest {
        text: "The river runs north past the old mill.",
        delivery: None,
        style: None,
        voice: VoiceSource::<()>::Named { name: "af_heart" },
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
    assert_eq!(
        a.samples.len(),
        b.samples.len(),
        "the seed moved a duration"
    );
}
