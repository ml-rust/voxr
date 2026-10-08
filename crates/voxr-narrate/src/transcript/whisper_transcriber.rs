//! Whisper as the recognizer of a narration take: one checkpoint loaded
//! once and read piece by piece, the same reading over a checkpoint the
//! caller already holds, and a take decoded to 16 kHz mono.

use std::path::Path;

use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, ConvOps, IndexingOps, MatmulOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, SortingOps, StatisticalOps, TensorOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use voxr::WhisperBundle;

use super::envelope::TRANSCRIPT_RATE;
use super::transcriber::{DEFAULT_LANGUAGE, Transcriber};
use crate::check::HeardWord;
use crate::error::{Error, Result};

/// The files a Whisper checkpoint directory holds, named in load errors.
const CHECKPOINT_LAYOUT: &str = "an HF-layout Whisper directory (config.json, tokenizer.json, model.safetensors, \
     generation_config.json)";

/// A Whisper checkpoint that times each word it hears.
///
/// The checkpoint loads once, with every weight cast to f32, and reads
/// every piece of every take. Word times come from the checkpoint's
/// cross-attention alignment heads, so `generation_config.json` must list
/// `alignment_heads`. A piece longer than Whisper's 30 s window is an
/// error; [`transcribe_narration`](super::transcribe_narration) cuts pieces
/// of at most 20 s.
pub struct WhisperTranscriber<R: Runtime<DType = DType> = CpuRuntime> {
    bundle: WhisperBundle<R>,
    client: R::Client,
}

impl WhisperTranscriber<CpuRuntime> {
    /// Loads the checkpoint in `dir` on the CPU.
    ///
    /// # Errors
    ///
    /// [`Error::Recognizer`] naming `dir` when it is not a directory, does
    /// not load as a Whisper checkpoint, or lists no `alignment_heads`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_dir_on(dir, &CpuDevice::new())
    }
}

impl<R: Runtime<DType = DType>> WhisperTranscriber<R>
where
    R::Client: TypeConversionOps<R>,
{
    /// Loads the checkpoint in `dir` on `device`.
    ///
    /// # Errors
    ///
    /// [`Error::Recognizer`] naming `dir` when it is not a directory, does
    /// not load as a Whisper checkpoint, or lists no `alignment_heads`.
    pub fn from_dir_on(dir: &Path, device: &R::Device) -> Result<Self> {
        if !dir.is_dir() {
            return Err(Error::Recognizer {
                reason: format!(
                    "Whisper checkpoint {} is not a directory: pass {CHECKPOINT_LAYOUT}",
                    dir.display()
                ),
            });
        }
        let client = R::default_client(device);
        let bundle = WhisperBundle::<R>::from_dir_with_dtype(dir, device, &client, DType::F32)
            .map_err(|e| Error::Recognizer {
                reason: format!(
                    "loading Whisper checkpoint {}: {e}. Check it is {CHECKPOINT_LAYOUT}",
                    dir.display()
                ),
            })?;
        if bundle.generation.alignment_heads.is_none() {
            return Err(Error::Recognizer {
                reason: format!(
                    "Whisper checkpoint {} lists no `alignment_heads` in \
                     generation_config.json, and word times need them. Use a checkpoint \
                     whose generation_config.json lists them, as every openai/whisper-* \
                     release does",
                    dir.display()
                ),
            });
        }
        Ok(Self { bundle, client })
    }
}

impl<R: Runtime<DType = DType>> WhisperTranscriber<R> {
    /// The loaded checkpoint.
    pub fn bundle(&self) -> &WhisperBundle<R> {
        &self.bundle
    }

    /// The checkpoint as a [`WhisperWords`], borrowed.
    pub fn words(&self) -> WhisperWords<'_, R> {
        WhisperWords::new(&self.bundle, &self.client)
    }
}

/// A Whisper checkpoint the caller holds, read as a [`Transcriber`]: what
/// [`WhisperTranscriber`] does, over a borrowed bundle and client.
///
/// The bundle's `generation_config.json` must list `alignment_heads`, as
/// [`WhisperTranscriber::from_dir_on`] checks at load. A bundle without
/// them fails each read, naming its checkpoint.
pub struct WhisperWords<'a, R: Runtime<DType = DType>> {
    bundle: &'a WhisperBundle<R>,
    client: &'a R::Client,
}

impl<'a, R: Runtime<DType = DType>> WhisperWords<'a, R> {
    /// Reads with `bundle` on `client`.
    pub fn new(bundle: &'a WhisperBundle<R>, client: &'a R::Client) -> Self {
        Self { bundle, client }
    }
}

impl<R: Runtime<DType = DType>> Transcriber for WhisperWords<'_, R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + MatmulOps<R>
        + BinaryOps<R>
        + ActivationOps<R>
        + NormalizationOps<R>
        + ConvOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + StatisticalOps<R>
        + SortingOps<R>,
{
    /// The words Whisper hears in `samples`, read in `language`, else in
    /// [`DEFAULT_LANGUAGE`]. An english-only checkpoint ignores the
    /// language.
    fn transcribe_words(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<Vec<HeardWord>> {
        let opts = voxr::TranscribeOptions {
            language: Some(language.unwrap_or(DEFAULT_LANGUAGE)),
            translate: false,
            max_new_tokens: None,
        };
        let (_, words) = self
            .bundle
            .transcribe_words(self.client, samples, sample_rate as usize, &opts)
            .map_err(|e| Error::Recognizer {
                reason: format!(
                    "Whisper checkpoint {} reading {} samples at {sample_rate} Hz: {e}",
                    self.bundle.checkpoint_dir.display(),
                    samples.len()
                ),
            })?;
        Ok(words.iter().map(HeardWord::from).collect())
    }
}

impl<R: Runtime<DType = DType>> Transcriber for WhisperTranscriber<R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + MatmulOps<R>
        + BinaryOps<R>
        + ActivationOps<R>
        + NormalizationOps<R>
        + ConvOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + StatisticalOps<R>
        + SortingOps<R>,
{
    /// The words Whisper hears in `samples`: [`WhisperWords`] over this
    /// checkpoint.
    fn transcribe_words(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<Vec<HeardWord>> {
        self.words()
            .transcribe_words(samples, sample_rate, language)
    }
}

/// Reads an audio file as 16 kHz mono samples, the input of
/// [`transcribe_narration`](super::transcribe_narration).
///
/// Any container voxr decodes (WAV, FLAC, MP3, OGG/Vorbis) is downmixed and
/// resampled to 16 kHz.
///
/// # Errors
///
/// [`Error::DataError`] naming `path` when it cannot be opened or decoded.
pub fn read_take_16k(path: &Path) -> Result<Vec<f32>> {
    voxr::decode_audio_file_mono_at(path, TRANSCRIPT_RATE).map_err(|e| Error::DataError {
        reason: format!(
            "decoding {}: {e}. Pass a WAV, FLAC, MP3 or OGG/Vorbis file",
            path.display()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_error(dir: &Path) -> String {
        match WhisperTranscriber::from_dir(dir) {
            Ok(_) => panic!("{} loaded as a Whisper checkpoint", dir.display()),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn missing_checkpoint_names_its_path() -> std::io::Result<()> {
        let root = tempfile::tempdir()?;
        let dir = root.path().join("no-such-whisper");
        let msg = load_error(&dir);
        assert!(msg.starts_with("recognizer error: "), "{msg}");
        assert!(msg.contains(&dir.display().to_string()), "{msg}");
        assert!(msg.contains("is not a directory"), "{msg}");
        Ok(())
    }

    #[test]
    fn empty_checkpoint_names_its_path() -> std::io::Result<()> {
        let root = tempfile::tempdir()?;
        let msg = load_error(root.path());
        assert!(msg.contains(&root.path().display().to_string()), "{msg}");
        assert!(msg.contains("config.json"), "{msg}");
        Ok(())
    }

    #[test]
    fn unreadable_take_names_its_path() -> std::io::Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("take.wav");
        std::fs::write(&path, b"not audio")?;
        let msg = match read_take_16k(&path) {
            Ok(pcm) => panic!("decoded {} samples from junk", pcm.len()),
            Err(e) => e.to_string(),
        };
        assert!(msg.contains(&path.display().to_string()), "{msg}");
        Ok(())
    }
}
