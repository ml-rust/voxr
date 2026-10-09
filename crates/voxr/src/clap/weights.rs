//! A CLAP checkpoint's weights: `model.safetensors`, else the
//! `pytorch_model.bin` state dict some repos ship alone.

use std::path::Path;

use boostr::format::TorchStateDict;
use boostr::nn::VarMap;
use numr::dtype::DType;
use numr::runtime::Runtime;

use crate::error::{Error, Result};

/// The SafeTensors weights file.
pub const SAFETENSORS: &str = "model.safetensors";

/// The PyTorch weights file.
pub const PYTORCH_BIN: &str = "pytorch_model.bin";

/// Buffers the state dict carries that no CLAP forward reads.
const UNREAD_BUFFERS: [&str; 4] = [
    "num_batches_tracked",
    "position_ids",
    "token_type_ids",
    "relative_position_index",
];

/// Every weight of the checkpoint in `dir`, on `device`.
///
/// Returns [`Error::ModelError`] when `dir` holds neither weights file, or
/// the file does not parse.
pub fn load_weights<R: Runtime<DType = DType>>(
    dir: &Path,
    device: &R::Device,
) -> Result<VarMap<R>> {
    let st = dir.join(SAFETENSORS);
    if st.is_file() {
        return Ok(VarMap::<R>::from_safetensors(&st, device)?);
    }
    let bin = dir.join(PYTORCH_BIN);
    if !bin.is_file() {
        return Err(Error::ModelError {
            reason: format!(
                "{} holds neither {SAFETENSORS} nor {PYTORCH_BIN}; point it at a complete CLAP \
                 checkpoint such as laion/larger_clap_general",
                dir.display()
            ),
        });
    }
    let dict = TorchStateDict::open(&bin)?;
    let mut map = VarMap::<R>::new();
    let mut names: Vec<&str> = dict
        .keys()
        .filter(|k| !UNREAD_BUFFERS.iter().any(|b| k.ends_with(b)))
        .collect();
    names.sort_unstable();
    for name in names {
        map.insert(name.to_string(), dict.load_tensor::<R>(name, device)?);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn a_directory_without_weights_names_both_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = load_weights::<CpuRuntime>(dir.path(), &CpuDevice::new())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains(SAFETENSORS) && err.contains(PYTORCH_BIN),
            "{err}"
        );
    }
}
