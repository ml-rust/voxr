//! `KokoroEngine` — the ready-to-call synthesis bundle.
//!
//! Ties together:
//!
//! * [`KokoroModelV2`] — the neural model
//! * [`KokoroG2p`] — misaki G2P + reference Kokoro's phoneme-budget chunking
//! * [`KokoroPhonemeVocab`] — IPA → token-id mapping
//! * [`VoiceResolver`] — where to find voice packs on disk
//! * The config (sample rate, min frames/phoneme, …) used at synthesis
//!
//! The engine is the piece that [`crate::tts::TtsBundle`] owns when
//! the neural path is live. Keeping it separate keeps `TtsBundle` scaffolding
//! callable unchanged — if no engine is attached, `synthesize` returns
//! `NotImplemented` exactly as it always did.

use crate::error::{Error, Result};
use crate::kokoro::g2p::{Dialect, KokoroG2p, KokoroG2pModels, phonemes_to_input_ids};
use crate::kokoro::voice::VoiceResolver;
use boostr::model::audio::kokoro::{
    KokoroClient, KokoroModelV2, KokoroPhonemeVocab, KokoroSynthOptions, load_kokoro_v2,
    select_voice_style,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::Path;
use std::sync::Arc;

/// A loaded Kokoro model with its G2P, vocab and voice lookup, on runtime
/// `R`. The model, the token ids and the voice pack live on `R`'s device.
/// G2P, chunking and voice-file decode run on the host.
pub struct KokoroEngine<R: Runtime<DType = DType> = CpuRuntime> {
    pub model: KokoroModelV2<R>,
    pub vocab: KokoroPhonemeVocab,
    pub resolver: VoiceResolver,
    pub client: Arc<R::Client>,
    pub device: Arc<R::Device>,
    pub g2p: KokoroG2p,
    /// The seed every [`KokoroEngine::synthesize`] call passes to the
    /// harmonic source. `None` draws from the runtime's unseeded RNG.
    seed: Option<u64>,
}

impl KokoroEngine<CpuRuntime> {
    /// Loads a Kokoro model directory on the CPU:
    ///
    /// ```text
    /// DIR/
    ///   config.json     KokoroConfig, its "vocab" the phoneme → id map
    ///   vocab.json      optional phoneme → id map, read before config.json's
    ///   weights         .safetensors (one file or shards) or .pth
    ///   voices/         optional voice packs, searched before the defaults
    ///   g2p/tagger/     spaCy tagger, required (KokoroG2pModels::resolve)
    ///   g2p/oov/        optional out-of-vocabulary fallback
    /// ```
    ///
    /// `g2p` overrides the G2P models; `None` resolves them from `dir`.
    /// Errors name the file that failed to load.
    pub fn from_dir(dir: &Path, g2p: Option<KokoroG2pModels>) -> Result<Self> {
        Self::from_dir_on(dir, g2p, &CpuDevice::new())
    }
}

impl<R: Runtime<DType = DType>> KokoroEngine<R>
where
    R::Client: KokoroClient<R>,
{
    /// Loads the Kokoro model directory `dir` onto `device`. The layout and
    /// `g2p` read as [`KokoroEngine::from_dir`] reads them. The G2P models
    /// stay on the host.
    pub fn from_dir_on(
        dir: &Path,
        g2p: Option<KokoroG2pModels>,
        device: &R::Device,
    ) -> Result<Self> {
        let device = Arc::new(device.clone());
        let client = Arc::new(R::default_client(&device));
        let model = load_kokoro_v2::<R, _>(client.as_ref(), dir, device.as_ref()).map_err(|e| {
            Error::ModelError {
                reason: format!("loading Kokoro weights from {}: {e}", dir.display()),
            }
        })?;
        let vocab_json = dir.join("vocab.json");
        let vocab_source = if vocab_json.exists() {
            vocab_json
        } else {
            dir.join("config.json")
        };
        let vocab =
            KokoroPhonemeVocab::from_json_file(&vocab_source).map_err(|e| Error::ModelError {
                reason: format!(
                    "loading the phoneme vocab from {}: {e}",
                    vocab_source.display()
                ),
            })?;
        let voices_dir = dir.join("voices");
        let resolver = if voices_dir.is_dir() {
            VoiceResolver::new().with_asset_dir(&voices_dir)
        } else {
            VoiceResolver::new()
        };
        let g2p = match g2p {
            Some(models) => models,
            None => KokoroG2pModels::resolve(dir, None, None)?,
        }
        .load()?;
        Ok(Self {
            model,
            vocab,
            resolver,
            client,
            device,
            g2p,
            seed: None,
        })
    }

    /// Fixes the harmonic-source seed every [`KokoroEngine::synthesize`]
    /// call passes, instead of the runtime's unseeded RNG. The same text,
    /// voice, speed and seed then give the same samples on one device. CPU
    /// and GPU run separate PRNGs, so one seed gives different samples on
    /// each.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run the full text-to-waveform pipeline.
    ///
    /// * `text` — UTF-8 input text. Split into reference-Kokoro-compatible
    ///   chunks internally; a script over the 510-phoneme-character budget
    ///   runs as several model calls, concatenated in order.
    /// * `voice_spec` — voice id (bundled or present in `resolver`'s asset
    ///   directory) or absolute path. See [`VoiceResolver::resolve_path`].
    ///   Its id's first character selects the G2P dialect: `a` = American
    ///   English, `b` = British English; any other prefix errs.
    /// * `speed` — pace; `1.0` is natural pace, >1 faster, <1 slower. It
    ///   divides each phoneme's predicted duration, as reference Kokoro's
    ///   `speed` does, so the audio is never time-stretched. Errs unless it
    ///   is finite and positive.
    ///
    /// The harmonic source draws from the seed
    /// [`with_seed`](Self::with_seed) fixed, else from the runtime's
    /// unseeded RNG.
    ///
    /// Returns f32 samples at the model's sample rate.
    pub fn synthesize(&self, text: &str, voice_spec: &str, speed: f32) -> Result<Vec<f32>> {
        self.synthesize_seeded(text, voice_spec, speed, self.seed)
    }

    /// [`synthesize`](Self::synthesize) with the harmonic source drawn from
    /// `seed` for this call alone. `None` draws from the runtime's unseeded
    /// RNG. The same text, voice, speed and seed give the same samples on
    /// one device. CPU and GPU run separate PRNGs, so one seed gives
    /// different samples on each. The first chunk of a long text draws
    /// from `seed` itself. Each later chunk draws from `seed` mixed with
    /// its index.
    pub fn synthesize_seeded(
        &self,
        text: &str,
        voice_spec: &str,
        speed: f32,
        seed: Option<u64>,
    ) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "text",
                reason: "input text must not be empty".into(),
            });
        }
        if !(speed.is_finite() && speed > 0.0) {
            return Err(Error::InvalidArgument {
                arg: "speed",
                reason: format!("expected a finite positive pace such as 1.0, got {speed}"),
            });
        }

        // 1. Dialect from the voice id, G2P + chunk the text.
        let dialect = Dialect::from_voice_id(voice_id(voice_spec))?;
        let chunks = self.g2p.chunks(text, dialect)?;

        // 2. Resolve + load the voice pack once; each chunk picks its own row.
        let voice_pack = self.resolver.load::<R>(voice_spec, &self.device)?;

        let mut waveform = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            // Phonemes → ids → [1, T_phon] tensor, reference Kokoro's
            // `KModel.forward`: one id per known phoneme character,
            // bracketed by the reserved BOS/EOS id 0.
            let ids = phonemes_to_input_ids(&self.vocab, &chunk.phonemes);
            if ids.len() <= 2 {
                // Every character in this chunk fell outside the vocab.
                continue;
            }
            let token_ids = Tensor::<R>::from_slice(&ids, &[1, ids.len()], &self.device)?;

            // Voice row selection reads the phoneme-character count, not
            // the id count — reference Kokoro's `pack[len(ps) - 1]`.
            let voice_row = select_voice_style(&voice_pack, chunk.phonemes.chars().count())?;

            let opts = KokoroSynthOptions {
                speed,
                seed: seed.map(|s| chunk_seed(s, index)),
            };
            let chunk_waveform =
                self.model
                    .synthesize(self.client.as_ref(), &token_ids, &voice_row, &opts)?;
            waveform.extend(chunk_waveform.contiguous()?.to_vec::<f32>());
        }

        if waveform.is_empty() {
            return Err(Error::DataError {
                reason: "no phonemes mapped to vocab — check that G2P and vocab agree".into(),
            });
        }
        Ok(waveform)
    }

    pub fn sample_rate(&self) -> u32 {
        self.model.config.sample_rate
    }
}

/// The seed chunk `index` of one synthesis passes to the harmonic source.
/// Chunk 0 keeps `seed`. Later chunks mix their index in with an odd
/// constant, so neighbouring chunks never draw from adjacent seeds. Boostr
/// already spends `seed` and `seed + 1` on one chunk's two draws.
fn chunk_seed(seed: u64, index: usize) -> u64 {
    seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// The voice id a spec names, for dialect selection: a bare id is used
/// as-is, a path's file stem is used instead (so `./voices/af_heart.pt`
/// still reads as `af_heart`).
fn voice_id(voice_spec: &str) -> &str {
    Path::new(voice_spec)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(voice_spec)
}

impl<R: Runtime<DType = DType>> crate::tts::TtsEngine for KokoroEngine<R>
where
    R::Client: KokoroClient<R>,
    KokoroModelV2<R>: Send + Sync,
{
    fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        KokoroEngine::synthesize(self, text, voice, speed)
    }

    fn sample_rate(&self) -> u32 {
        KokoroEngine::sample_rate(self)
    }

    /// The canonical catalog; voice packs beyond it resolve through
    /// `VoiceResolver` at synthesis time.
    fn voices(&self) -> Vec<crate::tts::Voice> {
        crate::tts::default_kokoro_voices()
    }
}

#[cfg(test)]
mod tests {
    use super::{chunk_seed, voice_id};

    // A unit test cannot build a full KokoroEngine without a real checkpoint:
    // every primitive needs non-empty weight tensors of matching shapes. The
    // engine only wires parts together. These tests cover the small helpers.
    // The `kokoro_pipeline_oracle` and `kokoro_audio_parity` integration tests
    // cover the whole engine when a model directory is set.

    #[test]
    fn voice_id_reads_bare_ids_and_path_stems() {
        assert_eq!(voice_id("af_heart"), "af_heart");
        assert_eq!(voice_id("./voices/af_heart.pt"), "af_heart");
        assert_eq!(voice_id("/abs/path/bf_emma.safetensors"), "bf_emma");
    }

    #[test]
    fn chunk_seed_keeps_the_first_chunk_and_spreads_the_rest() {
        assert_eq!(chunk_seed(7, 0), 7);
        let seeds: Vec<u64> = (0..4).map(|i| chunk_seed(7, i)).collect();
        for (i, a) in seeds.iter().enumerate() {
            for b in &seeds[i + 1..] {
                assert!(
                    a.abs_diff(*b) > 1,
                    "chunk seeds {a} and {b} collide or touch"
                );
            }
        }
    }
}
