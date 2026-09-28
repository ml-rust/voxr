//! Shared fixtures for the narration oracle test: JSON fixture loading and
//! the deterministic glitch-check test signals.

use std::path::Path;

use serde::de::DeserializeOwned;

pub mod signals;

/// Read and parse `path` as JSON, panicking with the path on either failure.
pub fn load_json<T: DeserializeOwned>(path: &Path) -> T {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}
