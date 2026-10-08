//! Parity between [`voxcpm_text`] and the skill's VoxCPM2 runner,
//! `voxcpm_speak.py`: for every invocation shape the runner takes, the text,
//! the Hi-Fi transcript and the recordings the adapter hands VoxCPM2 equal
//! the keyword arguments the runner passes `VoxCPM.generate`.
//!
//! The fixture (`tests/fixtures/voxcpm_runner_calls.json`) is checked in.
//! `tests/tools/voxcpm_runner/runner_calls.py` writes it by running the
//! runner unmodified with the model stubbed. With `VOXCPM_RUNNER` naming
//! the runner and `python3` on PATH, the test reruns the tool and checks
//! the fixture is current. Otherwise it skips that check with a notice.

use std::path::Path;
use std::process::Command;

use serde::Deserialize;
use serde_json::Value;
use voxr_narrate::{CloneReference, SpeakRequest, VoiceSource, voxcpm_text};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/voxcpm_runner_calls.json"
);
const TOOL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/tools/voxcpm_runner/runner_calls.py"
);

#[derive(Debug, Deserialize)]
struct Invocation {
    name: String,
    calls: Vec<Call>,
}

#[derive(Debug, Deserialize)]
struct Call {
    request: Request,
    seed: u64,
    generate: Generate,
}

/// What the runner was asked for one sentence.
#[derive(Debug, Deserialize)]
struct Request {
    text: String,
    seed: u64,
    delivery: Option<String>,
    style: Option<String>,
    voice: Option<String>,
    prompt_text: Option<String>,
    /// Whether a recording to clone existed when the call ran.
    anchored: bool,
}

/// The keyword arguments the runner passed `VoxCPM.generate`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Generate {
    text: String,
    cfg_value: f64,
    inference_timesteps: u32,
    prompt_text: Option<String>,
    prompt_wav_path: Option<String>,
    reference_wav_path: Option<String>,
}

fn fixture() -> Vec<Invocation> {
    let bytes = std::fs::read(FIXTURE).unwrap_or_else(|e| panic!("reading {FIXTURE}: {e}"));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parsing {FIXTURE}: {e}"))
}

#[test]
fn the_adapter_passes_what_the_runner_passes() {
    let samples = [0.0f32; 4];
    let mut diffs = Vec::new();
    for invocation in fixture() {
        for (k, call) in invocation.calls.iter().enumerate() {
            let r = &call.request;
            let voice = if r.anchored {
                VoiceSource::Clone {
                    reference: CloneReference::Samples {
                        samples: &samples,
                        rate: 48_000,
                    },
                    transcript: r.prompt_text.as_deref(),
                }
            } else {
                VoiceSource::Own {
                    design: r.voice.as_deref(),
                }
            };
            let request: SpeakRequest<'_, ()> = SpeakRequest {
                text: &r.text,
                delivery: r.delivery.as_deref(),
                style: r.style.as_deref(),
                voice,
                seed: r.seed as f64,
                cfg: call.generate.cfg_value,
                steps: call.generate.inference_timesteps,
                speed: 1.0,
            };
            let planned = voxcpm_text(&request);
            let g = &call.generate;
            let at = format!("{} call {k}", invocation.name);
            if planned.text != g.text {
                diffs.push(format!(
                    "{at}: text {:?}, runner {:?}",
                    planned.text, g.text
                ));
            }
            if planned.hifi != g.prompt_text.as_deref() {
                diffs.push(format!(
                    "{at}: Hi-Fi transcript {:?}, runner {:?}",
                    planned.hifi, g.prompt_text
                ));
            }
            // A clone reads its reference, and a Hi-Fi take the same
            // recording as its prompt too: `encode_reference_and_prompt`.
            if r.anchored != g.reference_wav_path.is_some() {
                diffs.push(format!("{at}: reference {:?}", g.reference_wav_path));
            }
            if g.prompt_wav_path.is_some() && g.prompt_wav_path != g.reference_wav_path {
                diffs.push(format!(
                    "{at}: prompt {:?} is not the reference {:?}",
                    g.prompt_wav_path, g.reference_wav_path
                ));
            }
            if call.seed != r.seed {
                diffs.push(format!("{at}: seed {}, runner {}", r.seed, call.seed));
            }
        }
    }
    assert!(diffs.is_empty(), "{}", diffs.join("\n"));
}

#[test]
fn the_fixture_is_current() {
    let Some(runner) = std::env::var_os("VOXCPM_RUNNER").filter(|r| !r.is_empty()) else {
        eprintln!("skipping: runner rerun unavailable; set VOXCPM_RUNNER to voxcpm_speak.py");
        return;
    };
    if !Path::new(&runner).is_file() {
        eprintln!("skipping: VOXCPM_RUNNER {runner:?} is not a file");
        return;
    }
    let Ok(output) = Command::new("python3")
        .arg("-I")
        .arg(TOOL)
        .arg(&runner)
        .output()
    else {
        eprintln!("skipping: runner rerun unavailable; put python3 on PATH");
        return;
    };
    assert!(
        output.status.success(),
        "{TOOL} failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let fresh: Value =
        serde_json::from_slice(&output.stdout).unwrap_or_else(|e| panic!("parsing {TOOL}: {e}"));
    let committed: Value = serde_json::from_slice(
        &std::fs::read(FIXTURE).unwrap_or_else(|e| panic!("reading {FIXTURE}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parsing {FIXTURE}: {e}"));
    assert!(
        fresh == committed,
        "{FIXTURE} is stale: regenerate it with `python3 -I {TOOL} <runner> > {FIXTURE}`"
    );
}
