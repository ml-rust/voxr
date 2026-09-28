//! The VoxCPM2 take cache on disk: one WAV per take, named by the take's
//! key ([`take_key`]), so a run speaks only the parts it has no take for.
//!
//! A take is kept as a 32-bit float WAV, so a take read back from the cache
//! is judged on the same samples as the take the engine just made. A take
//! file the skill wrote as 16-bit PCM reads back too.

use std::path::{Path, PathBuf};

use super::anchor_files::{create_dir, write_whole};
use super::cache::{TakeKeyInput, take_file, take_key};
use super::engine::{Take, TakeSpec, TakeStore};
use crate::error::Result;
use crate::wav::{f32_wav, read_wav_file};

/// What keys every take of a run, besides the take's own text, seed and
/// delivery note.
#[derive(Debug, Clone, Copy)]
pub struct TakeKeys<'a> {
    /// The VoxCPM2 checkpoint on disk. `None` falls back to
    /// [`VOXCPM`](super::cache::VOXCPM).
    pub checkpoint: Option<&'a str>,
    /// The voice description; empty when there is none.
    pub voice: &'a str,
    /// The reference recording's cache id ([`ref_id`](super::cache::ref_id));
    /// empty for a designed voice.
    pub ref_id: &'a str,
    /// The designed voice's key; `None` when a reference recording is
    /// cloned instead.
    pub anchor_key: Option<&'a str>,
    /// The run's style, parentheses included; empty when there is none. A
    /// part's own delivery note replaces it.
    pub style: &'a str,
    /// Classifier-free guidance.
    pub cfg: f64,
    /// Diffusion steps.
    pub steps: u32,
    /// Hi-Fi cloning is on.
    pub hifi: bool,
}

/// A [`TakeStore`] over the cache directory: each take at
/// `<cache>/<take key>.wav`.
#[derive(Debug, Clone, Copy)]
pub struct FileStore<'a> {
    /// The VoxCPM2 take cache directory.
    pub cache_dir: &'a Path,
    /// What keys every take of the run.
    pub keys: TakeKeys<'a>,
}

impl FileStore<'_> {
    /// The file `spec`'s take is kept in: the skill's `takeFile(s)`.
    pub fn file(&self, spec: &TakeSpec) -> PathBuf {
        let k = &self.keys;
        let key = take_key(
            k.checkpoint,
            k.voice,
            k.ref_id,
            TakeKeyInput {
                anchor_key: k.anchor_key,
                style: spec.delivery.as_deref().unwrap_or(k.style),
                text: &spec.text,
                seed: spec.seed,
                cfg: k.cfg,
                steps: f64::from(k.steps),
                hifi: k.hifi,
            },
        );
        take_file(self.cache_dir, &key)
    }
}

impl TakeStore for FileStore<'_> {
    fn has(&mut self, spec: &TakeSpec) -> bool {
        self.file(spec).exists()
    }

    fn load(&mut self, spec: &TakeSpec) -> Result<Take> {
        read_wav_file(&self.file(spec))
    }

    fn save(&mut self, spec: &TakeSpec, take: &Take) -> Result<()> {
        create_dir(self.cache_dir)?;
        write_whole(&self.file(spec), &f32_wav(&take.samples, take.rate)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::take::cache::{LEGACY_CFG, LEGACY_STEPS};

    fn keys(style: &str) -> TakeKeys<'_> {
        TakeKeys {
            checkpoint: None,
            voice: "(warm, low)",
            ref_id: "",
            anchor_key: Some("abc"),
            style,
            cfg: 1.6,
            steps: 16,
            hifi: false,
        }
    }

    fn spec(delivery: Option<&str>) -> TakeSpec {
        TakeSpec {
            index: 0,
            text: "Hello there.".to_string(),
            seed: 7.0,
            delivery: delivery.map(str::to_string),
        }
    }

    #[test]
    fn a_delivery_note_replaces_the_run_style_in_the_key() {
        let dir = Path::new("/cache");
        let store = FileStore {
            cache_dir: dir,
            keys: keys("(calm)"),
        };
        let own = TakeKeyInput {
            anchor_key: Some("abc"),
            style: "(asking)",
            text: "Hello there.",
            seed: 7.0,
            cfg: 1.6,
            steps: 16.0,
            hifi: false,
        };
        let want = take_file(dir, &take_key(None, "(warm, low)", "", own));
        assert_eq!(store.file(&spec(Some("(asking)"))), want);
        assert_ne!(store.file(&spec(None)), want);
    }

    #[test]
    fn legacy_settings_keep_the_older_key() {
        let mut k = keys("");
        k.cfg = LEGACY_CFG;
        k.steps = LEGACY_STEPS as u32;
        let store = FileStore {
            cache_dir: Path::new("/cache"),
            keys: k,
        };
        let input = TakeKeyInput {
            anchor_key: Some("abc"),
            style: "",
            text: "Hello there.",
            seed: 7.0,
            cfg: LEGACY_CFG,
            steps: LEGACY_STEPS,
            hifi: false,
        };
        assert_eq!(
            store.file(&spec(None)),
            take_file(
                Path::new("/cache"),
                &take_key(None, "(warm, low)", "", input)
            )
        );
    }

    #[test]
    fn a_saved_take_reads_back_exactly() -> Result<()> {
        let dir = tempfile::tempdir().map_err(|e| Error::File {
            reason: format!("creating a temporary directory: {e}"),
        })?;
        let cache = dir.path().join("cache");
        let mut store = FileStore {
            cache_dir: &cache,
            keys: keys(""),
        };
        let s = spec(None);
        assert!(!store.has(&s));
        let take = Take {
            samples: vec![0.1, -0.25, 0.3333],
            rate: 48000,
        };
        store.save(&s, &take)?;
        assert!(store.has(&s));
        assert_eq!(store.load(&s)?, take);
        Ok(())
    }
}
