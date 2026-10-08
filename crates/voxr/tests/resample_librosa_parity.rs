//! Parity of `voxr::resample::resample_like_librosa` against
//! `librosa.resample` at its default quality.
//!
//! The fixture comes from `tests/tools/resample/make_soxr_fixture.py`. It
//! holds, per rate pair and signal, the `float32` input and librosa's
//! `resample` of it. Each case must match librosa's length exactly.
//!
//! voxr's filter is an independent design, so its samples differ from
//! librosa's by the gap between two filters, not by float rounding. Each
//! rate pair has two SNR floors, set from measurement with a margin:
//!
//! - steady: the chirps and the speech, where the passband and transition
//!   band dominate.
//! - every case: impulses and short clips as well, where the first and last
//!   outputs dominate.
//!
//! The test reads `$VOXR_SOXR_FIXTURE_DIR/resample_soxr.safetensors`. It
//! prints a notice and passes without checking when the variable is unset.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;
use voxr::resample::{librosa_output_len, resample_like_librosa};

/// Per rate pair: `(from, to, steady floor dB, every-case floor dB)`.
///
/// On 1:3, 2:3 and 3:1 voxr's filter matches librosa's response to near
/// 20-bit precision. On the other pairs librosa's own response differs
/// slightly from pair to pair, and its first outputs differ more, which
/// bounds the agreement lower.
const FLOORS: &[(u32, u32, f64, f64)] = &[
    (4_000, 96_000, 63.0, 19.0),
    (8_000, 16_000, 83.0, 72.0),
    (8_000, 48_000, 68.0, 19.0),
    (11_025, 16_000, 68.0, 23.0),
    (12_345, 16_000, 68.0, 23.0),
    (16_000, 16_001, 68.0, 23.0),
    (16_000, 44_100, 68.0, 20.0),
    (16_000, 48_000, 127.0, 117.0),
    (16_001, 16_000, 67.0, 23.0),
    (22_050, 16_000, 64.0, 22.0),
    (24_000, 16_000, 124.0, 117.0),
    (30_001, 16_000, 63.0, 22.0),
    (32_000, 16_000, 79.0, 71.0),
    (44_100, 16_000, 64.0, 24.0),
    (44_100, 48_000, 68.0, 23.0),
    (44_101, 16_000, 64.0, 38.0),
    (48_000, 16_000, 124.0, 117.0),
    (48_000, 44_100, 63.0, 23.0),
    (96_000, 16_000, 65.0, 25.0),
    (192_000, 16_000, 60.0, 25.0),
];

/// Signals whose SNR the steady floor gates.
const STEADY: &[&str] = &["chirp", "long", "speech"];

#[derive(Deserialize)]
struct Case {
    name: String,
    from: u32,
    to: u32,
}

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// A safetensors file of 1-D `F32` tensors, and its metadata.
struct Fixture {
    tensors: BTreeMap<String, Vec<f32>>,
    metadata: BTreeMap<String, String>,
}

fn read_fixture(path: &PathBuf) -> Fixture {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let head: [u8; 8] = bytes[..8]
        .try_into()
        .unwrap_or_else(|e| panic!("header: {e}"));
    let head_len = u64::from_le_bytes(head) as usize;
    let header: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + head_len]).unwrap_or_else(|e| panic!("header: {e}"));
    let data = &bytes[8 + head_len..];
    let mut tensors = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    for (key, value) in header {
        if key == "__metadata__" {
            metadata = serde_json::from_value(value).unwrap_or_else(|e| panic!("metadata: {e}"));
            continue;
        }
        let info: TensorInfo =
            serde_json::from_value(value).unwrap_or_else(|e| panic!("tensor {key}: {e}"));
        assert_eq!(info.dtype, "F32", "tensor {key}");
        assert_eq!(info.shape.len(), 1, "tensor {key}");
        let raw = &data[info.data_offsets[0]..info.data_offsets[1]];
        let (words, rest) = raw.as_chunks::<4>();
        assert!(
            rest.is_empty(),
            "tensor {key}: {} trailing bytes",
            rest.len()
        );
        let values = words
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect::<Vec<_>>();
        assert_eq!(values.len(), info.shape[0], "tensor {key}");
        tensors.insert(key, values);
    }
    Fixture { tensors, metadata }
}

/// SNR in dB of `test` against `reference`, and the largest absolute error.
fn agreement(reference: &[f32], test: &[f32]) -> (f64, f64) {
    let (mut signal, mut noise, mut max_err) = (0.0f64, 0.0f64, 0.0f64);
    for (&r, &t) in reference.iter().zip(test) {
        let e = f64::from(t) - f64::from(r);
        signal += f64::from(r) * f64::from(r);
        noise += e * e;
        max_err = max_err.max(e.abs());
    }
    let snr = if noise == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (signal / noise).log10()
    };
    (snr, max_err)
}

#[test]
fn resample_like_librosa_matches_librosa() {
    let Some(dir) = std::env::var_os("VOXR_SOXR_FIXTURE_DIR") else {
        eprintln!(
            "skipping: resample fixture unavailable; set VOXR_SOXR_FIXTURE_DIR to the output \
             directory of tests/tools/resample/make_soxr_fixture.py"
        );
        return;
    };
    let path = PathBuf::from(dir).join("resample_soxr.safetensors");
    let fixture = read_fixture(&path);
    let cases: Vec<Case> =
        serde_json::from_str(&fixture.metadata["cases"]).unwrap_or_else(|e| panic!("cases: {e}"));
    let tensor = |name: &str, part: &str| -> &[f32] {
        let key = format!("{name}/{part}");
        fixture
            .tensors
            .get(&key)
            .unwrap_or_else(|| panic!("fixture has no tensor {key}"))
    };

    // Per rate pair: the lowest steady SNR and the lowest over every case.
    let mut worst: BTreeMap<(u32, u32), (f64, f64)> = BTreeMap::new();
    let mut failures = Vec::new();
    for case in &cases {
        let input = tensor(&case.name, "input");
        let reference = tensor(&case.name, "librosa");
        let ours = resample_like_librosa(input, case.from, case.to)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let want_len = librosa_output_len(input.len(), case.from, case.to);
        if ours.len() != reference.len() || want_len != reference.len() {
            failures.push(format!(
                "{}: length {} (librosa_output_len {want_len}), librosa {}",
                case.name,
                ours.len(),
                reference.len()
            ));
            continue;
        }
        let (snr, max_err) = agreement(reference, &ours);
        println!(
            "{:<26} n={:<7} snr={snr:>7.1} dB max_err={max_err:.2e}",
            case.name,
            ours.len()
        );
        if case.from == case.to {
            if ours != reference {
                failures.push(format!("{}: equal rates must return the input", case.name));
            }
            continue;
        }
        if reference.iter().all(|&v| v == 0.0) {
            continue;
        }
        let signal = case.name.rsplit('/').next().unwrap_or_default();
        let w = worst
            .entry((case.from, case.to))
            .or_insert((f64::INFINITY, f64::INFINITY));
        if STEADY.contains(&signal) {
            w.0 = w.0.min(snr);
        }
        w.1 = w.1.min(snr);
    }
    for (&(from, to), &(steady, every)) in &worst {
        println!(
            "{from:>6} -> {to:<6} lowest snr: steady {steady:>6.1} dB, every case {every:>6.1} dB"
        );
        let Some(&(_, _, steady_floor, every_floor)) =
            FLOORS.iter().find(|f| (f.0, f.1) == (from, to))
        else {
            failures.push(format!("{from} -> {to}: no floor for this rate pair"));
            continue;
        };
        if steady < steady_floor {
            failures.push(format!(
                "{from} -> {to}: steady snr {steady:.1} dB under its {steady_floor} dB floor"
            ));
        }
        if every < every_floor {
            failures.push(format!(
                "{from} -> {to}: snr {every:.1} dB under its {every_floor} dB floor"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} cases fail:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
