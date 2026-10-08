//! The engine struct, the state its owner thread shares with every caller,
//! the client-capability bundle the request path needs, and the constants
//! both the loader and the request path use.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::thread::JoinHandle;

use crate::voxcpm::tokenizer::VoxCpm2Tokenizer;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, HostCopyOps, IndexingOps, RandomOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::VoxCpmClient;
use boostr::model::audio::voxcpm::model::{LoraAdapterReport, VoxCpm2Model};
use boostr::model::audio::voxcpm::vae::decoder::HOP_LENGTH;
use boostr::quant::traits::DequantOps;

use crate::voxcpm::engine::batch::Work;
use crate::voxcpm::options::VoxCpm2SynthOptions;

/// Sample rate the reference and prompt encoder expects.
pub(super) const REF_RATE: u32 = 16_000;

/// Hard cap on generated patches for one request, whatever the text length.
pub(super) const MAX_LEN_CAP: usize = 4096;

/// Containers `decode_audio` probes; a file with any other extension in the
/// voices directory is not a voice.
pub const VOICE_EXTENSIONS: [&str; 4] = ["wav", "flac", "mp3", "ogg"];

/// Reserved voice id selecting zero-shot rendering: no reference recording,
/// no reference encode, `ref_feat = None` straight through to `prefill`.
///
/// A fixed literal, not an `Option<String>` layered on top of the render
/// path: an omitted voice must be an explicit, listable choice — present in
/// [`VoxCpm2Engine::voices`](crate::tts::TtsEngine::voices) like any other
/// id — never a silent fallback a caller has to infer. A literal also can't
/// collide with a voice file by accident: [`VoxCpm2Engine::load`] refuses
/// any file stemmed `zero-shot`.
pub const ZERO_SHOT_VOICE_ID: &str = "zero-shot";

/// Every client capability the VoxCPM2 request path needs, as one bound.
///
/// A supertrait bundle with a blanket impl, so a client implementing the
/// parts implements this automatically and no caller names it.
pub trait VoxCpm2RenderClient<R: Runtime<DType = DType>>:
    VoxCpmClient<R>
    + TensorOps<R>
    + ScalarOps<R>
    + ReduceOps<R>
    + IndexingOps<R>
    + ShapeOps<R>
    + ActivationOps<R>
    + BinaryOps<R>
    + UnaryOps<R>
    + CompareOps<R>
    + ConditionalOps<R>
    + TypeConversionOps<R>
    + RandomOps<R>
    + HostCopyOps<R>
    + DequantOps<R>
    + Send
    + Sync
    + 'static
{
}

impl<R, C> VoxCpm2RenderClient<R> for C
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + TypeConversionOps<R>
        + RandomOps<R>
        + HostCopyOps<R>
        + DequantOps<R>
        + Send
        + Sync
        + 'static,
{
}

/// A reference voice, encoded once.
pub(super) struct EncodedVoice<R: Runtime> {
    /// `[T_ref, patch_size, feat_dim]` reference patches from
    /// `encode_reference`.
    pub(super) ref_feat: Tensor<R>,
}

/// Everything a render reads, shared by every caller thread and the owner
/// thread behind an [`Arc`]. Immutable after [`VoxCpm2Engine::load`]: a
/// caller reads the voice table and the tokenizer to plan its request, the
/// owner thread reads the model and the client to run it. Only the owner
/// thread runs the model, reference and prompt encodes included.
pub(super) struct EngineInner<R: Runtime<DType = DType>> {
    pub(super) model: VoxCpm2Model<R>,
    pub(super) client: Arc<R::Client>,
    pub(super) tokenizer: VoxCpm2Tokenizer,
    pub(super) voices: BTreeMap<String, EncodedVoice<R>>,
    pub(super) options: VoxCpm2SynthOptions,
    pub(super) adapter: Option<LoraAdapterReport>,
    /// This engine's identity, unique within the process. A
    /// [`Reference`](crate::voxcpm::Reference) and a
    /// [`Prompt`](crate::voxcpm::Prompt) record it, so one encoded by another
    /// engine is refused.
    pub(super) id: u64,
}

/// A fresh [`EngineInner::id`]: every call returns a value no earlier call
/// in this process returned.
pub(super) fn next_engine_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// VoxCPM2 voice-cloning engine on runtime `R`.
pub struct VoxCpm2Engine<R: Runtime<DType = DType>> {
    pub(super) inner: Arc<EngineInner<R>>,
    /// Queue into the owner thread. `None` only while [`Drop`] closes it.
    pub(super) jobs: Option<Sender<Work<R>>>,
    /// The owner thread. `None` only while [`Drop`] joins it.
    pub(super) worker: Option<JoinHandle<()>>,
}

impl<R: Runtime<DType = DType>> VoxCpm2Engine<R> {
    /// The adapter this engine was loaded with, if
    /// [`VoxCpm2LoadOptions::adapter`](crate::voxcpm::VoxCpm2LoadOptions::adapter)
    /// was `Some` — rank, alpha, targets and tensor counts, for a caller to
    /// log without reaching into the model. `None` when the engine is
    /// serving the base (unadapted) model.
    pub fn adapter(&self) -> Option<&LoraAdapterReport> {
        self.inner.adapter.as_ref()
    }

    /// Output samples one generated patch decodes to: the model's patch size
    /// times the decoder's hop, at
    /// [`TtsEngine::sample_rate`](crate::tts::TtsEngine::sample_rate).
    ///
    /// Every waveform a render returns is a whole number of patches, and
    /// every streamed chunk but the last is exactly
    /// [`VoxCpm2SynthOptions::stream_chunk_patches`](crate::voxcpm::VoxCpm2SynthOptions::stream_chunk_patches)
    /// patches.
    pub fn samples_per_patch(&self) -> usize {
        self.inner.model.config.patch_size * HOP_LENGTH
    }
}

/// Close the queue, then join the owner thread: no request outlives the
/// engine and no thread is left detached. A thread that already died of a
/// panic joins with an error, which is ignored here — unwinding out of
/// `drop` would abort the process.
impl<R: Runtime<DType = DType>> Drop for VoxCpm2Engine<R> {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
