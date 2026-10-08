//! The skill-parity fixtures `tests/tools/voxcpm/make_skill_fixture.py`
//! writes: one `skill_<mode>.{safetensors,json}` pair per runner branch and
//! `skill_tokens.json`, and the agreement measures the parity test reports.

use std::path::{Path, PathBuf};

use boostr::format::safetensors_loader::SafeTensorsLoader;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use serde::Deserialize;

/// The runner branches the fixture covers, in the order the test runs them.
pub const MODES: [&str; 4] = ["design", "zeroshot", "clone", "hifi"];

/// Resolves `VOXCPM2_CHECKPOINT`: the `openbmb/VoxCPM2` snapshot directory
/// (`config.json`, `model.safetensors`, `audiovae.pth`, `tokenizer.json`).
pub fn checkpoint() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("VOXCPM2_CHECKPOINT").ok()?);
    path.join("model.safetensors").is_file().then_some(path)
}

/// Resolves `VOXCPM2_SKILL_FIXTURE_DIR`: the directory the fixture tool
/// wrote.
pub fn fixture_dir() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("VOXCPM2_SKILL_FIXTURE_DIR").ok()?);
    path.join("skill_tokens.json").is_file().then_some(path)
}

/// One take the reference generated.
#[derive(Debug, Deserialize)]
pub struct TakeMeta {
    pub patches: usize,
    pub draws: usize,
    pub context_len: usize,
    pub max_len: usize,
    pub min_len: usize,
}

/// `skill_<mode>.json`.
#[derive(Debug, Deserialize)]
pub struct ModeMeta {
    pub mode: String,
    pub generate_text: String,
    pub target_text: String,
    pub prompt_text: Option<String>,
    pub reference_wav: Option<String>,
    pub full_ids: Vec<u32>,
    pub target_ids: Vec<u32>,
    pub cfg_value: f32,
    pub inference_timesteps: usize,
    pub retry_badcase_max_times: usize,
    pub retry_badcase_ratio_threshold: f64,
    pub sample_rate: u32,
    pub encode_sample_rate: u32,
    pub decode_patch_samples: usize,
    pub takes: Vec<TakeMeta>,
    pub wav_samples: usize,
}

/// One entry of `skill_tokens.json`.
#[derive(Debug, Deserialize)]
pub struct TokenCase {
    pub text: String,
    pub collapsed: String,
    pub ids: Vec<u32>,
}

/// One mode's fixture: its metadata and its open tensor file.
pub struct ModeFixture {
    pub meta: ModeMeta,
    tensors: SafeTensorsLoader,
}

impl ModeFixture {
    pub fn open(dir: &Path, mode: &str) -> Self {
        let meta = super::load_json(&dir.join(format!("skill_{mode}.json")));
        let path = dir.join(format!("skill_{mode}.safetensors"));
        let tensors = SafeTensorsLoader::open(&path)
            .unwrap_or_else(|e| panic!("opening {}: {e}", path.display()));
        Self { meta, tensors }
    }

    /// Tensor `name`, or `None` when this mode has none by that name.
    pub fn tensor(&mut self, name: &str, device: &CpuDevice) -> Option<Tensor<CpuRuntime>> {
        if self.tensors.tensor_info(name).is_err() {
            return None;
        }
        Some(
            self.tensors
                .load_tensor::<CpuRuntime>(name, device)
                .unwrap_or_else(|e| panic!("{}: loading {name}: {e}", self.meta.mode)),
        )
    }

    /// Tensor `name` as host floats.
    pub fn floats(&mut self, name: &str, device: &CpuDevice) -> Option<Vec<f32>> {
        self.tensor(name, device).map(|t| contiguous(&t).to_vec())
    }

    /// Tensor `name` as host integers.
    pub fn ints(&mut self, name: &str, device: &CpuDevice) -> Option<Vec<i64>> {
        self.tensor(name, device).map(|t| contiguous(&t).to_vec())
    }
}

fn contiguous(t: &Tensor<CpuRuntime>) -> Tensor<CpuRuntime> {
    t.contiguous().expect("contiguous")
}

/// `||got - want|| / ||want||`, over equal lengths.
pub fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "rel_l2 over unequal lengths");
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        err += (f64::from(*g) - f64::from(*w)).powi(2);
        norm += f64::from(*w).powi(2);
    }
    (err / norm.max(f64::MIN_POSITIVE)).sqrt()
}

/// Signal-to-error ratio of `got` against `want`, in dB.
pub fn snr_db(got: &[f32], want: &[f32]) -> f64 {
    -20.0 * rel_l2(got, want).max(1e-12).log10()
}

/// Largest absolute difference, over equal lengths.
pub fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "max_abs over unequal lengths");
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()))
}
