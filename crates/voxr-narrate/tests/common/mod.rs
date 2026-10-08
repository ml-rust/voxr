//! Shared fixtures for the narration oracle tests: JSON fixture loading, the
//! fixture freshness check, the deterministic test signals, the polish
//! cases, the pace cases, the transcript fixture and the take oracles'
//! files.

// Each file in `tests/` compiles as its own crate and includes this module
// whole, so a helper one test crate does not call reads as dead code there.
#![allow(dead_code)]

use std::path::Path;
use std::process::Command;

use serde::de::DeserializeOwned;
use serde_json::Value;

#[cfg(feature = "stretch")]
pub mod ffmpeg;
pub mod pace_cases;
pub mod polish_cases;
pub mod signals;
pub mod take_files;
pub mod transcript_cases;

/// Read and parse `path` as JSON, panicking with the path on either failure.
pub fn load_json<T: DeserializeOwned>(path: &Path) -> T {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}

/// `String(x)` read back: the exact double.
pub fn num(text: &str) -> f64 {
    text.parse()
        .unwrap_or_else(|e| panic!("fixture number {text:?}: {e}"))
}

/// Records a mismatch between the port and the fixture as a difference.
pub fn check(diffs: &mut Vec<String>, what: &str, got: String, expected: String) {
    if got != expected {
        diffs.push(format!("{what}: got {got}, want {expected}"));
    }
}

/// Reruns `oracle` and checks the committed `fixture` is what it prints.
/// Needs `NARRATION_JS_DIR` (the directory holding the JavaScript modules)
/// and `node` on PATH.
pub fn assert_oracle_current(oracle: &str, fixture: &str) {
    let Some(js_dir) = std::env::var_os("NARRATION_JS_DIR").filter(|d| !d.is_empty()) else {
        eprintln!("skipping: narration oracle rerun unavailable; set NARRATION_JS_DIR");
        return;
    };
    let Ok(output) = Command::new("node")
        .arg(oracle)
        .env("NARRATION_JS_DIR", &js_dir)
        .output()
    else {
        eprintln!("skipping: narration oracle rerun unavailable; put node on PATH");
        return;
    };
    assert!(
        output.status.success(),
        "{oracle} failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let fresh: Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("parsing {oracle} output: {e}"));
    let committed: Value = load_json(Path::new(fixture));
    assert!(
        fresh == committed,
        "{fixture} is stale: regenerate it with `NARRATION_JS_DIR=<dir> node {oracle} > {fixture}`"
    );
}
