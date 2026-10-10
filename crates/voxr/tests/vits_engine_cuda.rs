//! [`VitsEngine`] on CUDA: a seeded take repeats sample for sample, holds
//! only finite samples, and lands near the CPU engine's take of the same
//! input and seed in length. The bundle [`load_vits_from_dir_on`] builds
//! renders what the engine renders with the model's default seed.
//!
//! MMS-TTS predicts durations stochastically, and each device draws that
//! noise from its own PRNG. One seed therefore gives each device its own
//! durations, so the lengths agree within [`LENGTH_RATIO`] and not exactly.
//!
//! Gated on CUDA and the checkpoint: skips unless `MMS_TTS_ZLM_DIR`, else
//! `$BOOSTR_MODELS_DIR/facebook/mms-tts-zlm`, is present on disk.

#![cfg(all(feature = "vits", feature = "cuda"))]

mod common;

use common::{model_fixture, skip_notice};
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
use voxr::tts::{SynthesizeOptions, TtsEngine, VitsEngine, load_vits_from_dir_on};

const TEXT: &str = "Selamat pagi, apa khabar?";
const SEED: u64 = 7;
/// The range the CUDA take's length over the CPU take's length stays in.
/// Two noise draws move single durations, never the whole sentence's pace.
const LENGTH_RATIO: std::ops::RangeInclusive<f64> = 0.8..=1.25;

#[test]
fn a_seeded_cuda_take_repeats_and_matches_the_cpu_length() {
    let Some(dir) = model_fixture("MMS_TTS_ZLM_DIR", "facebook/mms-tts-zlm") else {
        skip_notice("facebook/mms-tts-zlm checkpoint", "MMS_TTS_ZLM_DIR");
        return;
    };
    let device = CudaDevice::new(0);

    let gpu = VitsEngine::<CudaRuntime>::from_dir_on(&dir, &device)
        .unwrap_or_else(|e| panic!("loading VITS on CUDA from {}: {e}", dir.display()));
    let voice = TtsEngine::voices(&gpu)
        .into_iter()
        .next()
        .map(|v| v.id)
        .expect("the checkpoint's one voice");
    let take = gpu
        .synthesize_seeded(TEXT, &voice, 1.0, Some(SEED))
        .expect("CUDA take");
    let again = gpu
        .synthesize_seeded(TEXT, &voice, 1.0, Some(SEED))
        .expect("CUDA take again");
    assert!(!take.is_empty(), "the CUDA take holds no samples");
    assert!(
        take.iter().all(|s| s.is_finite()),
        "the CUDA take holds a non-finite sample"
    );
    assert_eq!(take, again, "one seed gave two CUDA takes");

    let cpu = VitsEngine::from_dir(&dir)
        .unwrap_or_else(|e| panic!("loading VITS on the CPU from {}: {e}", dir.display()));
    let cpu_take = cpu
        .synthesize_seeded(TEXT, &voice, 1.0, Some(SEED))
        .expect("CPU take");
    let ratio = take.len() as f64 / cpu_take.len() as f64;
    assert!(
        LENGTH_RATIO.contains(&ratio),
        "the CUDA take is {} samples and the CPU take {}: ratio {ratio} is outside {LENGTH_RATIO:?}",
        take.len(),
        cpu_take.len()
    );

    let bundle = load_vits_from_dir_on::<CudaRuntime>(&dir, &device).unwrap_or_else(|e| {
        panic!(
            "loading the VITS bundle on CUDA from {}: {e}",
            dir.display()
        )
    });
    let from_bundle = bundle
        .synthesize(TEXT, &voice, &SynthesizeOptions::default())
        .expect("bundle take");
    let unseeded = gpu
        .synthesize_seeded(TEXT, &voice, 1.0, None)
        .expect("CUDA take at the default seed");
    assert_eq!(
        from_bundle, unseeded,
        "the bundle and the engine rendered different takes at the default seed"
    );
}
