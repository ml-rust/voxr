//! Real-model fixtures for the adapter tests: model paths from env vars,
//! the runtime the models run on, the Whisper hearing, and one speak run
//! into a temporary directory.
//!
//! Each model path is its own env var, else a directory under
//! `BOOSTR_MODELS_DIR`, as voxr's own tests resolve them:
//!
//! - `WHISPER_BASE_DIR`, else `whisper-base`: the default recognizer.
//! - `WHISPER_SMALL_DIR`, else `whisper-small`: the small recognizer. When
//!   neither is on disk, the base checkpoint serves both, with a notice.
//! - `KOKORO_DIR`, else `Kokoro-82M`.
//! - `MMS_TTS_ZLM_DIR`, else `facebook/mms-tts-zlm`.
//! - `VOXCPM2_GGUF`, `VOXCPM2_REF_WAV` and `VOXCPM2_VOICES_DIR`, with no
//!   fallback.
//!
//! A test whose model is missing prints a skip notice and passes.

use std::path::{Path, PathBuf};

use voxr_narrate::{
    RecognizerSource, ScriptChecks, SpeakOpts, SpeakRuns, SpeakTarget, SpokenScript, TtsConfig,
    WhisperHearing, WhisperTranscriber, read_wav_file, speak_script,
};

/// The runtime the models run on: CUDA device 0 under the `cuda` feature,
/// else the CPU.
#[cfg(feature = "cuda")]
pub type Rt = numr::runtime::cuda::CudaRuntime;
#[cfg(not(feature = "cuda"))]
pub type Rt = numr::runtime::cpu::CpuRuntime;

/// The device [`Rt`] runs on.
#[cfg(feature = "cuda")]
pub fn device() -> <Rt as numr::runtime::Runtime>::Device {
    numr::runtime::cuda::CudaDevice::new(0)
}
#[cfg(not(feature = "cuda"))]
pub fn device() -> <Rt as numr::runtime::Runtime>::Device {
    numr::runtime::cpu::CpuDevice::new()
}

/// `$var`, else `$BOOSTR_MODELS_DIR/<relative>` when `relative` is not
/// empty. `None` when neither is set or the path is absent.
pub fn model_path(var: &str, relative: &str) -> Option<PathBuf> {
    let path = match std::env::var(var) {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ if relative.is_empty() => return None,
        _ => {
            let root = std::env::var("BOOSTR_MODELS_DIR")
                .ok()
                .filter(|r| !r.is_empty())?;
            PathBuf::from(root).join(relative)
        }
    };
    path.exists().then_some(path)
}

/// Prints the skip notice for a missing `what`, naming `var`.
pub fn skip(what: &str, var: &str) {
    eprintln!("skipping: {what} unavailable; set {var} or BOOSTR_MODELS_DIR");
}

/// The Whisper base checkpoint, or a skip notice.
pub fn whisper_base() -> Option<PathBuf> {
    let found = model_path("WHISPER_BASE_DIR", "whisper-base");
    if found.is_none() {
        skip("Whisper base checkpoint", "WHISPER_BASE_DIR");
    }
    found
}

/// The Whisper small checkpoint, else the base one with a notice.
pub fn whisper_small(base: &Path) -> PathBuf {
    model_path("WHISPER_SMALL_DIR", "whisper-small").unwrap_or_else(|| {
        eprintln!(
            "WHISPER_SMALL_DIR unset and no whisper-small under BOOSTR_MODELS_DIR: the base checkpoint serves as the small recognizer"
        );
        base.to_path_buf()
    })
}

/// Loads a Whisper checkpoint on [`device`].
pub fn load_whisper(dir: &Path) -> WhisperTranscriber<Rt> {
    WhisperTranscriber::<Rt>::from_dir_on(dir, &device())
        .unwrap_or_else(|e| panic!("loading Whisper from {}: {e}", dir.display()))
}

/// The hearing of every adapter test: the base checkpoint loaded now, the
/// small one loaded the first time a take needs it.
pub fn hearing(base: &Path) -> WhisperHearing<'static> {
    let small = whisper_small(base);
    WhisperHearing::new(
        RecognizerSource::loaded(load_whisper(base)),
        RecognizerSource::lazy(move || WhisperTranscriber::<Rt>::from_dir_on(&small, &device())),
    )
}

/// A speak run's output files, removed when the test ends.
pub struct RunDir {
    dir: tempfile::TempDir,
}

impl RunDir {
    /// A new empty directory.
    pub fn new() -> Self {
        Self {
            dir: tempfile::tempdir().expect("creating a temporary directory"),
        }
    }

    /// The voiceover's path.
    pub fn out(&self) -> PathBuf {
        self.dir.path().join("voiceover.wav")
    }

    /// The take cache.
    pub fn cache(&self) -> PathBuf {
        self.dir.path().join("cache")
    }
}

/// `script` spoken through `runs` into `dir`, with VoxCPM2 ready.
/// `checkpoint` keys VoxCPM2's cache.
pub fn speak(
    script: &str,
    opts: &SpeakOpts<'_>,
    dir: &RunDir,
    checkpoint: Option<&str>,
    runs: &mut impl SpeakRuns,
) -> SpokenScript {
    let (out, cache) = (dir.out(), dir.cache());
    let target = SpeakTarget {
        asr_model: None,
        out: &out,
        cache_dir: &cache,
        checkpoint,
        voxcpm_ready: true,
        voxcpm_why: "",
    };
    speak_script(script, opts, &TtsConfig::default(), &target, runs)
        .unwrap_or_else(|e| panic!("speaking {script:?}: {e}"))
}

/// Panics unless every check of `spoken` passed and it reports no problem.
pub fn assert_spoken(spoken: &SpokenScript) {
    let failed: Vec<String> = match &spoken.checks {
        ScriptChecks::Parts(list) => list
            .iter()
            .filter(|c| !c.ok)
            .map(|c| format!("{:?}: {} (heard {:?})", c.text, c.why, c.heard))
            .collect(),
        ScriptChecks::Whole(list) => list
            .iter()
            .filter(|c| !c.ok)
            .map(|c| format!("{:?}: {} (heard {:?})", c.text, c.why, c.heard))
            .collect(),
    };
    assert!(failed.is_empty(), "failed checks: {failed:?}");
    assert!(!spoken.checks.is_empty(), "no check ran");
    assert!(
        spoken.problems.is_empty(),
        "problems: {:?}",
        spoken.problems
    );
    assert!(spoken.duration > 0.5, "{} s of audio", spoken.duration);
}

/// The voiceover on disk, heard back through `hearing` against `script`,
/// and checked.
pub fn assert_heard_back(
    hearing: &mut WhisperHearing<'_>,
    dir: &RunDir,
    script: &str,
    language: Option<&str>,
) {
    let take = read_wav_file(&dir.out()).expect("reading the voiceover");
    let main = voxr_narrate::asr_model(language, None);
    let text = voxr_narrate::sayable(script).expect("the script's sayable text");
    let heard = voxr_narrate::heard_check(hearing, main, &take.samples, take.rate, &text, language)
        .expect("hearing the voiceover");
    assert!(
        heard.check.ok,
        "the voiceover fails its check: {} (heard {:?})",
        heard.check.why, heard.heard
    );
}
