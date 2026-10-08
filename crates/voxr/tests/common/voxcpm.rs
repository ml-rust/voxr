//! Shared VoxCPM2 fixtures: the model/voices/reference-wav paths the
//! checkpoint-gated engine tests skip without, a runtime alias matching the
//! engine blazr serves, the canonical engine loader, and a bit-exact sample
//! comparison.

// Each file in `tests/` compiles as its own crate and pulls this module in
// wholesale, so an item a given test crate does not call reads as dead code
// there. The items ARE used — just not all of them by every crate.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use boostr::model::audio::voxcpm::VoxCpm2Weights;
use voxr::voxcpm::{VOICE_EXTENSIONS, VoxCpm2Engine, VoxCpm2LoadOptions, VoxCpm2SynthOptions};

/// The runtime the engine runs on: CUDA under the `cuda` feature, else the
/// CPU.
#[cfg(feature = "cuda")]
pub type Rt = boostr::runtime::cuda::CudaRuntime;
#[cfg(not(feature = "cuda"))]
pub type Rt = numr::runtime::cpu::CpuRuntime;

/// Long enough that every concurrent request in a test lands in the same
/// drain window, whatever the machine is doing.
pub const WINDOW: Duration = Duration::from_millis(500);

/// Resolves `VOXCPM2_GGUF`: a compressr GGUF that embeds the VAE and the
/// tokenizer. Returns `None` when it is unset or the path is not a file, so
/// callers skip.
pub fn gguf_fixture() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("VOXCPM2_GGUF").ok()?);
    path.is_file().then_some(path)
}

/// Resolves `VOXCPM2_VOICES_DIR`: a directory of reference recordings.
/// Returns `None` when it is unset or the path is not a directory, so
/// callers skip or fall back to zero-shot.
pub fn voices_fixture() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("VOXCPM2_VOICES_DIR").ok()?);
    path.is_dir().then_some(path)
}

/// Resolves `VOXCPM2_REF_WAV`: a reference recording for the clone path.
/// Returns `None` when it is unset or the path is absent, so callers skip.
pub fn ref_wav_fixture() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("VOXCPM2_REF_WAV").ok()?);
    path.exists().then_some(path)
}

/// Loads a [`VoxCpm2Engine`] from `gguf`, with `voices` as its optional
/// voices directory and `synth` as its synthesis options. Runs on CUDA
/// device 0 with an F16 decoder under the `cuda` feature, the configuration
/// blazr serves; otherwise on the CPU at F32.
pub fn load_engine(
    gguf: &Path,
    voices: Option<&Path>,
    synth: VoxCpm2SynthOptions,
) -> VoxCpm2Engine<Rt> {
    let weights = VoxCpm2Weights::Gguf {
        path: gguf.to_path_buf(),
        config: None,
    };
    #[cfg(feature = "cuda")]
    {
        use boostr::runtime::cuda::CudaDevice;
        use numr::runtime::Runtime;
        let device = CudaDevice::new(0);
        let client = std::sync::Arc::new(Rt::default_client(&device));
        let options = VoxCpm2LoadOptions {
            dtype: None,
            vae_decoder_dtype: Some(numr::dtype::DType::F16),
            synth,
            adapter: None,
        };
        VoxCpm2Engine::<Rt>::load(&weights, None, voices, &device, client, options)
            .expect("load VoxCPM2 on CUDA")
    }
    #[cfg(not(feature = "cuda"))]
    {
        use numr::runtime::Runtime;
        use numr::runtime::cpu::CpuDevice;
        let device = CpuDevice::new();
        let client = std::sync::Arc::new(Rt::default_client(&device));
        let options = VoxCpm2LoadOptions {
            dtype: None,
            vae_decoder_dtype: None,
            synth,
            adapter: None,
        };
        VoxCpm2Engine::<Rt>::load(&weights, None, voices, &device, client, options)
            .expect("load VoxCPM2 on CPU")
    }
}

/// `text` in voice `voice` at the load-time seed, one take: what a stream
/// of the same request concatenates to. `synthesize` may retry the take, a
/// stream never does, so a stream is compared against this.
pub fn first_take(engine: &VoxCpm2Engine<Rt>, text: &str, voice: &str) -> Vec<f32> {
    use voxr::voxcpm::{Badcase, SynthRequest, VoiceSource};
    let request = SynthRequest::new(text, VoiceSource::Named(voice))
        .with_badcase(Badcase::no_retry(Badcase::RATIO));
    engine.render_request(&request).expect("first take")
}

/// The file behind voice `id`, found the way the engine lists voices: an
/// audio file in `dir` whose stem is `id`.
pub fn voice_file(dir: &Path, id: &str) -> PathBuf {
    std::fs::read_dir(dir)
        .expect("read voices directory")
        .map(|entry| entry.expect("voices directory entry").path())
        .find(|path| {
            path.is_file()
                && path.file_stem().and_then(|s| s.to_str()) == Some(id)
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|ext| VOICE_EXTENSIONS.iter().any(|a| ext.eq_ignore_ascii_case(a)))
        })
        .unwrap_or_else(|| panic!("no audio file for voice {id:?} in {}", dir.display()))
}

/// Voice `id`'s recording in `dir` as mono PCM, with its sample rate, read
/// as the engine reads a voices-directory file: integer PCM over
/// `2^(bits - 1)`, then the channel mean.
pub fn voice_samples(dir: &Path, id: &str) -> (Vec<f32>, u32) {
    let path = voice_file(dir, id);
    let bytes = std::fs::read(&path).expect("read voice file");
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(voxr::extension_hint);
    let data = voxr::decode_audio_like_ffmpeg(&bytes, hint).expect("decode voice file");
    let mono = voxr::to_mono(&data.samples, data.channels).expect("downmix");
    (mono, data.sample_rate)
}

/// Panics unless `got` equals `want` bit for bit, naming the first differing
/// sample.
pub fn assert_bit_identical(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: {} samples, expected {}",
        got.len(),
        want.len()
    );
    let mismatch = got
        .iter()
        .zip(want)
        .position(|(g, w)| g.to_bits() != w.to_bits());
    if let Some(i) = mismatch {
        panic!(
            "{label}: sample {i} is {:e} ({:#010x}), expected {:e} ({:#010x})",
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        );
    }
}
