//! [`ClapBundle`]: a CLAP checkpoint directory loaded for scoring. The
//! model and its log-mel front end from boostr, the tokenizer from the
//! checkpoint's `tokenizer.json` through splintr.
//!
//! Expected layout (what `laion/larger_clap_general` ships):
//! ```text
//! <dir>/
//!   config.json                 # architecture
//!   preprocessor_config.json    # log-mel front end
//!   tokenizer.json              # RoBERTa byte-level BPE
//!   model.safetensors           # weights, or pytorch_model.bin
//! ```

use std::path::{Path, PathBuf};

use boostr::model::audio::clap::{ClapClient, ClapConfig, ClapFeatureExtractor, ClapModel};
use boostr::nn::VarBuilder;
use boostr::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{MatmulOps, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use splintr::{AnyTokenizer, from_json_path};

use super::weights::load_weights;
use crate::error::{Error, Result};

/// Clips embedded per forward pass, which bounds activation memory.
const AUDIO_BATCH: usize = 8;

/// A CLAP checkpoint loaded on one device.
pub struct ClapBundle<R: Runtime> {
    pub model: ClapModel<R>,
    pub features: ClapFeatureExtractor,
    pub tokenizer: AnyTokenizer,
    client: R::Client,
    device: R::Device,
}

fn read(path: PathBuf) -> Result<String> {
    std::fs::read_to_string(&path).map_err(|e| Error::ModelError {
        reason: format!("reading {}: {e}", path.display()),
    })
}

impl<R: Runtime<DType = DType>> ClapBundle<R>
where
    R::Client: ClapClient<R> + DequantOps<R>,
{
    /// Loads the checkpoint in `dir` onto `device`.
    ///
    /// Returns [`Error::ModelError`] when a file is missing or malformed, or
    /// the checkpoint is fused.
    pub fn from_dir_on(dir: &Path, device: &R::Device) -> Result<Self> {
        let config = ClapConfig::from_json(&read(dir.join("config.json"))?)?;
        let features =
            ClapFeatureExtractor::from_json(&read(dir.join("preprocessor_config.json"))?)?;
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = from_json_path(&tokenizer_path).map_err(|e| Error::ModelError {
            reason: format!("loading {}: {e}", tokenizer_path.display()),
        })?;
        let client = R::default_client(device);
        let mut weights = load_weights::<R>(dir, device)?;
        let mut vb = VarBuilder::new(&mut weights, device);
        let model = ClapModel::from_varbuilder(&client, &mut vb, &config)?;
        Ok(Self {
            model,
            features,
            tokenizer,
            client,
            device: device.clone(),
        })
    }

    /// The rate clips must be at, in Hz.
    pub fn sampling_rate(&self) -> u32 {
        self.features.sampling_rate()
    }

    /// The token ids the checkpoint's tokenizer gives `text`, with `<s>`
    /// and `</s>` around them.
    pub fn token_ids(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Unit-length embeddings of `clips` (mono, at [`Self::sampling_rate`]),
    /// `[clips, projection_dim]` on the device.
    pub fn audio_embeddings(&self, clips: &[Vec<f32>]) -> Result<Tensor<R>> {
        let mut parts = Vec::with_capacity(clips.len().div_ceil(AUDIO_BATCH));
        for chunk in clips.chunks(AUDIO_BATCH) {
            let feats = self.features.features::<R>(chunk, &self.device)?;
            parts.push(self.model.audio_embeddings(&self.client, &feats)?);
        }
        self.join(parts)
    }

    /// Unit-length embeddings of `texts`, `[texts, projection_dim]` on the
    /// device.
    pub fn text_embeddings(&self, texts: &[String]) -> Result<Tensor<R>> {
        let ids: Vec<Vec<u32>> = texts.iter().map(|t| self.token_ids(t)).collect();
        Ok(self
            .model
            .text_embeddings(&self.client, &ids, &self.device)?)
    }

    fn join(&self, parts: Vec<Tensor<R>>) -> Result<Tensor<R>> {
        let refs: Vec<&Tensor<R>> = parts.iter().collect();
        match refs.as_slice() {
            [] => Err(Error::InvalidArgument {
                arg: "clips",
                reason: "no clips to embed".to_string(),
            }),
            [one] => Ok((*one).clone()),
            many => Ok(self.client.cat(many, 0)?),
        }
    }

    /// The cosine score of every clip against every text, row-major
    /// `[clips × texts]`: `audio · textᵀ` of the unit-length embeddings.
    pub fn scores(&self, clips: &[Vec<f32>], texts: &[String]) -> Result<Vec<f32>> {
        let audio = self.audio_embeddings(clips)?;
        let text = self.text_embeddings(texts)?;
        let text_t = text.transpose(0, 1)?.contiguous()?;
        let scores = self.client.matmul(&audio, &text_t)?;
        Ok(scores.contiguous()?.to_vec())
    }
}

impl<R: Runtime<DType = DType>> crate::sound_review::AudioTextScorer for ClapBundle<R>
where
    R::Client: ClapClient<R> + DequantOps<R>,
{
    fn scores(&mut self, clips: &[Vec<f32>], texts: &[String]) -> Result<Vec<f32>> {
        ClapBundle::scores(self, clips, texts)
    }
}
