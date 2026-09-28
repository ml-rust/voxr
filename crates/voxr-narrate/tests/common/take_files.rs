//! What the take oracles share: signal ops read from a fixture, the
//! synthetic take `takeSignal` builds, a model call as the stub model
//! records it, and every file under a directory with the hash of its
//! bytes.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use voxr_narrate::{CloneReference, Error, Result, SpeakRequest, VoiceSource};

use super::signals::{self, Op, hiss, noise, tone};

/// One signal op, as signals.mjs writes it.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum OpJson {
    Tone {
        hz: String,
        start: usize,
        len: usize,
        amp: f64,
        attack: usize,
        release: usize,
    },
    Noise {
        start: usize,
        len: usize,
        amp: f64,
        seed: u32,
    },
    Hiss {
        start: usize,
        len: usize,
        amp: f64,
        seed: u32,
        attack: usize,
        release: usize,
    },
}

impl OpJson {
    /// The op `common::signals` builds.
    pub fn op(&self) -> Op {
        match self {
            Self::Tone {
                hz,
                start,
                len,
                amp,
                attack,
                release,
            } => {
                let name = signals::TONES
                    .iter()
                    .map(|t| t.0)
                    .find(|t| *t == hz.as_str())
                    .unwrap_or_else(|| panic!("no tone {hz} in signals::TONES"));
                tone(name, *start, *len, *amp, *attack, *release)
            }
            Self::Noise {
                start,
                len,
                amp,
                seed,
            } => noise(*start, *len, *amp, *seed),
            Self::Hiss {
                start,
                len,
                amp,
                seed,
                attack,
                release,
            } => hiss(*start, *len, *amp, *seed, *attack, *release),
        }
    }
}

/// `render(n, ops)` over fixture ops.
pub fn render_json(n: usize, ops: &[OpJson]) -> Vec<f32> {
    let ops: Vec<Op> = ops.iter().map(OpJson::op).collect();
    signals::render(n, &ops)
}

/// `takeSignal` in signals.mjs: 0.1 s of floor noise, a 200 Hz tone of
/// `len` samples, split by a breath when asked, then 0.1 s of floor noise.
pub fn take_signal(len: usize, breath: bool, noise_seed: u32) -> Vec<f32> {
    let lead = 4800;
    let gap = if breath { 960 + 7200 + 960 } else { 0 };
    let n = lead + len + gap + lead;
    let mut ops: Vec<Op> = vec![noise(0, n, 0.00002, noise_seed)];
    if breath {
        let half = len / 2;
        ops.push(tone("200", lead, half, 0.3, 480, 480));
        ops.push(hiss(lead + half + 960, 7200, 0.009, 41, 480, 480));
        ops.push(tone("200", lead + half + gap, len - half, 0.3, 480, 480));
    } else {
        ops.push(tone("200", lead, len, 0.3, 480, 480));
    }
    signals::render(n, &ops)
}

/// An [`Error::File`] naming `what` failed on `path`.
pub fn io(what: &str, path: &Path) -> impl FnOnce(std::io::Error) -> Error {
    let reason = format!("{what} {}", path.display());
    move |e| Error::File {
        reason: format!("{reason}: {e}"),
    }
}

/// One file and the FNV-1a hash of its bytes.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Listed {
    pub path: String,
    pub bytes: String,
}

/// Every file under `root`, by its `/`-joined path from `root`, with the
/// FNV-1a hash of its bytes, sorted by path.
pub fn listing(root: &Path) -> Result<Vec<Listed>> {
    let mut out = Vec::new();
    let mut dirs: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).map_err(io("listing", &dir))? {
            let path = entry.map_err(io("listing", &dir))?.path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            let rel = path.strip_prefix(root).map_err(|e| Error::DataError {
                reason: format!("{} is outside {}: {e}", path.display(), root.display()),
            })?;
            let parts: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            let bytes = std::fs::read(&path).map_err(io("reading", &path))?;
            out.push(Listed {
                path: parts.join("/"),
                bytes: signals::fnv1a_bytes(&bytes),
            });
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// One model call: the text as spoken, the prefix, the seed, the voice
/// kind, the design description, the Hi-Fi transcript, and the hash of the
/// samples a clone reads from its reference.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Call {
    pub text: String,
    pub prefix: Option<String>,
    pub seed: f64,
    pub voice: String,
    pub design: Option<String>,
    pub transcript: Option<String>,
    pub reference: Option<String>,
}

impl Call {
    /// `request` as the stub model records it. A clone from a handle and a
    /// named voice are refused: VoxCPM2 designs a voice or clones one from
    /// samples.
    pub fn of(request: &SpeakRequest<'_, ()>) -> Result<Self> {
        let (design, transcript, reference) = match request.voice {
            VoiceSource::Own { design } => (design, None, None),
            VoiceSource::Named { name } => {
                return Err(Error::DataError {
                    reason: format!("VoxCPM2 has no named voice {name:?}"),
                });
            }
            VoiceSource::Clone {
                reference: CloneReference::Samples { samples, .. },
                transcript,
            } => (None, transcript, Some(signals::fnv1a(samples))),
            VoiceSource::Clone { .. } => {
                return Err(Error::DataError {
                    reason: "a take clones from samples, never a handle".to_string(),
                });
            }
        };
        Ok(Self {
            text: request.text.to_string(),
            prefix: request.prefix().map(str::to_string),
            seed: request.seed,
            voice: request.voice.kind().to_string(),
            design: design.map(str::to_string),
            transcript: transcript.map(str::to_string),
            reference,
        })
    }
}
