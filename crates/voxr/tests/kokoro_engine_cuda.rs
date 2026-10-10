//! [`KokoroEngine`] on CUDA: a seeded take repeats sample for sample, holds
//! only finite samples, and is as long as the CPU engine's take of the same
//! input. The durations draw no random number, so both devices agree on the
//! length. The harmonic source draws from per-device PRNGs, so the samples
//! themselves differ between devices.
//!
//! Gated on CUDA and the model directory: skips unless `KOKORO_DIR`, else
//! `$BOOSTR_MODELS_DIR/Kokoro-82M`, is present with its `g2p/tagger`.

#![cfg(all(feature = "kokoro", feature = "cuda"))]

mod common;

use common::{model_fixture, skip_notice};
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
use voxr::kokoro::KokoroEngine;

const TEXT: &str = "The river runs north past the old mill.";
const VOICE: &str = "af_heart";
const SEED: u64 = 7;

#[test]
fn a_seeded_cuda_take_repeats_and_matches_the_cpu_length() {
    let Some(dir) = model_fixture("KOKORO_DIR", "Kokoro-82M") else {
        skip_notice("Kokoro-82M model directory", "KOKORO_DIR");
        return;
    };

    let gpu = KokoroEngine::<CudaRuntime>::from_dir_on(&dir, None, &CudaDevice::new(0))
        .unwrap_or_else(|e| panic!("loading Kokoro on CUDA from {}: {e}", dir.display()));
    let take = gpu
        .synthesize_seeded(TEXT, VOICE, 1.0, Some(SEED))
        .expect("CUDA take");
    let again = gpu
        .synthesize_seeded(TEXT, VOICE, 1.0, Some(SEED))
        .expect("CUDA take again");
    assert!(!take.is_empty(), "the CUDA take holds no samples");
    assert!(
        take.iter().all(|s| s.is_finite()),
        "the CUDA take holds a non-finite sample"
    );
    assert_eq!(take, again, "one seed gave two CUDA takes");

    let cpu = KokoroEngine::from_dir(&dir, None)
        .unwrap_or_else(|e| panic!("loading Kokoro on the CPU from {}: {e}", dir.display()));
    let cpu_take = cpu
        .synthesize_seeded(TEXT, VOICE, 1.0, Some(SEED))
        .expect("CPU take");
    assert_eq!(
        take.len(),
        cpu_take.len(),
        "CUDA and CPU predicted different durations for one input"
    );
}
