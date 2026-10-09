//! Parity of voxr's sound-review measures with the motion-video skill's
//! `sound-review.py`, run on generated clips (tones, noise, a chirp, short
//! clips, silence, a hot tone, bursts in silence, 8 to 48 kHz, 16-bit,
//! 24-bit and float WAVs) and on speech recordings.
//!
//! The fixture comes from `tests/tools/sound_review/make_metrics_fixture.py`:
//! the script's own `measure()`, unrounded and rounded, and the float32
//! clip its model ranking hears. The test reads
//! `$SOUND_REVIEW_METRICS_DIR/metrics.json` and the WAVs it names, and
//! prints a notice and passes without checking when the variable is unset.
//!
//! Every rounded measure must match exactly. The unrounded ones match to
//! the bounds below. They are not bit-exact because the two FFTs (numr
//! against pocketfft) round differently, which leaves a spectrum floor
//! that sharpness lifts in a near-empty band: it raises each band's power
//! to 0.23.

use std::path::PathBuf;

use serde::Deserialize;
use voxr::sound_review::{RawMeasures, model_clip};
use voxr::wav::decode_wav_f64;

/// Largest relative difference of an unrounded sharpness.
const SHARP_REL: f64 = 1e-7;

/// Largest absolute difference of an unrounded hiss, tail or true peak, dB.
const DB_ABS: f64 = 1e-7;

/// Largest absolute difference of a model-clip sample.
const CLIP_ABS: f32 = 1e-6;

#[derive(Deserialize)]
struct Measures {
    sharp: f64,
    hiss: f64,
    tail: f64,
    tp: f64,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    file: PathBuf,
    raw: Measures,
    rounded: Measures,
    model_clip: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

fn rel(a: f64, b: f64) -> f64 {
    if a == b {
        0.0
    } else {
        (a - b).abs() / a.abs().max(b.abs())
    }
}

#[test]
fn measures_match_the_skill_script() {
    let Some(dir) = std::env::var_os("SOUND_REVIEW_METRICS_DIR").map(PathBuf::from) else {
        eprintln!(
            "skipping: sound-review metrics fixture unavailable; set SOUND_REVIEW_METRICS_DIR \
             (tests/tools/sound_review/make_metrics_fixture.py writes it)"
        );
        return;
    };
    let text = std::fs::read_to_string(dir.join("metrics.json")).expect("metrics.json");
    let fixture: Fixture = serde_json::from_str(&text).expect("parse metrics.json");
    assert!(!fixture.cases.is_empty());
    let mut worst = [0.0f64; 4];
    let mut worst_clip = 0.0f32;
    for case in &fixture.cases {
        let bytes = std::fs::read(&case.file).expect("wav");
        let wav = decode_wav_f64(&bytes).expect("decode");
        let raw = RawMeasures::of(&wav).expect("measure");
        let got = [raw.sharp, raw.hiss, raw.tail, raw.tp];
        let want = [case.raw.sharp, case.raw.hiss, case.raw.tail, case.raw.tp];
        worst[0] = worst[0].max(rel(got[0], want[0]));
        for k in 1..4 {
            worst[k] = worst[k].max((got[k] - want[k]).abs());
        }
        let r = raw.rounded();
        let rounded = [r.sharp, r.hiss, r.tail, r.tp];
        let want_rounded = [
            case.rounded.sharp,
            case.rounded.hiss,
            case.rounded.tail,
            case.rounded.tp,
        ];
        assert_eq!(rounded, want_rounded, "{}: rounded measures", case.name);
        if let Some(want_clip) = &case.model_clip {
            let clip = model_clip(&wav).expect("model clip");
            assert_eq!(clip.len(), want_clip.len(), "{}: clip length", case.name);
            for (a, b) in clip.iter().zip(want_clip) {
                worst_clip = worst_clip.max((a - b).abs());
            }
        }
    }
    eprintln!(
        "{} cases; worst difference: sharp {:.3e} relative; hiss {:.3e}, tail {:.3e}, tp {:.3e} \
         dB; model-clip sample {:.3e}",
        fixture.cases.len(),
        worst[0],
        worst[1],
        worst[2],
        worst[3],
        worst_clip
    );
    assert!(
        worst[0] <= SHARP_REL,
        "sharp: relative difference {}",
        worst[0]
    );
    for (k, name) in ["hiss", "tail", "tp"].iter().enumerate() {
        assert!(
            worst[k + 1] <= DB_ABS,
            "{name}: difference {} dB",
            worst[k + 1]
        );
    }
    assert!(worst_clip <= CLIP_ABS, "model clip: {worst_clip}");
}
