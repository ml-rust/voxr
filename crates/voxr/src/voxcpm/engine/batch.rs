//! The queue between the callers and the engine's owner thread: what a
//! request hands over, the per-row options a batch runs with, and the owner
//! thread itself. `super::queue` takes work off the queue.
//!
//! A caller resolves its own request (voice, seed, tokens, budget), sends
//! the result as a [`Job`] and blocks on its reply channel. The owner thread
//! blocks on the queue, takes the first job, keeps draining until
//! [`VoxCpm2SynthOptions::batch_window`] expires or the batch is full, and
//! runs the whole group through one batched generation (`super::run`).
//!
//! A job owns everything it reads and borrows nothing. A voice from the
//! engine's table travels as its id, and the owner thread looks the tensor
//! up in [`EngineInner`] when it builds the batch. A reference the caller
//! encoded travels as an [`Arc`] of its tensor, and so does a continuation
//! prompt, so each tensor lives until the last job holding it is done.
//!
//! [`VoxCpm2SynthOptions::batch_window`]: crate::voxcpm::VoxCpm2SynthOptions::batch_window

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::{PadSide, RowOptions, VoxCpm2Model};

use crate::error::Result;
use crate::voxcpm::engine::queue::{collect_batch, recv_job};
use crate::voxcpm::engine::reply::Reply;
use crate::voxcpm::engine::types::{EngineInner, VoxCpm2RenderClient};

// The allowed row counts and the two roundings over them live in boostr,
// beside the generation state whose stepped width reads the same list. The
// engine and the loop must agree on which shapes the DiT's graph cache holds,
// so the engine imports the one definition.
pub(super) use boostr::model::audio::voxcpm::model::{batch_width, clamp_max_batch};

/// Options for a slot that holds no request. It counts no patch, and its
/// noise draw reaches no row's output, so its seed is arbitrary.
pub(super) const FREE_ROW: RowOptions = RowOptions {
    max_len: 1,
    seed: 0,
};

/// The voice one queued job renders with.
pub(super) enum JobVoice<R: Runtime> {
    /// An id in the engine's voice table, looked up on the owner thread.
    Named(String),
    /// A reference the caller encoded for this request. The job owns an
    /// [`Arc`] of the tensor.
    Reference(Arc<Tensor<R>>),
    /// No reference: `ref_feat = None` straight through to `prefill`.
    ZeroShot,
}

/// One queued request. Holds no borrow, so it can sit in the queue while the
/// caller blocks elsewhere.
pub(super) struct Job<R: Runtime> {
    /// The voice this row renders with.
    pub(super) voice: JobVoice<R>,
    /// The `[T_prompt, patch_size, feat_dim]` continuation prompt this row
    /// continues from, `None` for no continuation.
    pub(super) prompt: Option<Arc<Tensor<R>>>,
    /// This row's base seed. Patch `i` draws with `seed + i`.
    pub(super) seed: u64,
    /// Text tokens with `AUDIO_START_ID` already appended: the prompt
    /// transcript and the request text as one string, for a continuation
    /// row.
    pub(super) text_token_ids: Vec<u32>,
    /// This row's patch cap.
    pub(super) max_len: usize,
    /// This row's prefill sequence length, reference prefix and prompt tail
    /// included.
    pub(super) seq_len: usize,
    /// Where this row's output goes.
    pub(super) reply: Reply,
}

impl<R: Runtime> Job<R> {
    /// This row's cap and seed, as the generation loop reads them.
    pub(super) fn row_options(&self) -> RowOptions {
        RowOptions {
            max_len: self.max_len,
            seed: self.seed,
        }
    }
}

/// A request to encode one recording on the owner thread.
pub(super) struct EncodeJob<R: Runtime> {
    /// Mono PCM at the encoder's rate.
    pub(super) samples: Vec<f32>,
    /// Where the zero padding to whole patches goes: [`PadSide::Right`] for
    /// a reference, [`PadSide::Left`] for a continuation prompt.
    pub(super) side: PadSide,
    /// Where the encoded `[T, patch_size, feat_dim]` tensor goes.
    pub(super) reply: Sender<Result<Tensor<R>>>,
}

/// One message on the owner thread's queue.
pub(super) enum Work<R: Runtime> {
    /// A render, batched with whatever else is queued.
    Render(Job<R>),
    /// A reference or prompt encode, run at once between two steps or
    /// batches.
    Encode(EncodeJob<R>),
}

/// Per-row caps and seeds for a batch of `width` rows whose first
/// `real.len()` rows are real, in queue order.
///
/// Every real row draws from its OWN request's seed. Row `b` of a step draws
/// `randn_seeded(seed + i)` over its own `[1, feat_dim, patch_size]` and the
/// rows are stacked, so a row's noise depends on neither the batch width nor
/// its neighbours. A seed derived from arrival order or slot index breaks
/// that.
///
/// A slot past the last request gets [`FREE_ROW`]. The loop needs one entry
/// per slot, and a free slot counts no patch.
pub(super) fn row_options(real: &[RowOptions], width: usize) -> Vec<RowOptions> {
    (0..width)
        .map(|b| real.get(b).copied().unwrap_or(FREE_ROW))
        .collect()
}

/// Start the owner thread and return the queue into it.
///
/// The thread holds an [`Arc`] of the shared state and nothing else, so it
/// ends exactly when the returned [`Sender`] drops.
pub(super) fn spawn_worker<R>(
    inner: Arc<EngineInner<R>>,
) -> Result<(Sender<Work<R>>, JoinHandle<()>)>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    let (sender, receiver) = channel::<Work<R>>();
    let handle = std::thread::Builder::new()
        .name("voxcpm2-render".to_string())
        .spawn(move || worker_loop(&inner, &receiver))?;
    Ok((sender, handle))
}

/// Block for a job, batch it with whatever else is queued, run the batch,
/// repeat. Returns when the queue closes, which is what [`Drop`] does.
///
/// With
/// [`VoxCpm2SynthOptions::continuous_batching`](crate::voxcpm::VoxCpm2SynthOptions::continuous_batching)
/// the thread runs `super::continuous` instead: one batch stays open and
/// queued requests join its free slots between steps.
fn worker_loop<R>(inner: &Arc<EngineInner<R>>, jobs: &Receiver<Work<R>>)
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    if inner.options.continuous_batching {
        super::continuous::run_continuous(inner, jobs);
        return;
    }
    let window = inner.options.batch_window;
    let max_batch = clamp_max_batch(inner.options.max_batch);
    let mut serve = |encode: EncodeJob<R>| inner.serve_encode(encode);
    while let Some(first) = recv_job(jobs, &mut serve) {
        let batch = collect_batch(jobs, first, window, max_batch, &mut serve);
        inner.run_batch(batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    type TestJob = Job<CpuRuntime>;

    /// The persistent batch is the default route: the owner thread keeps one
    /// batch open and admits arrivals into its free slots. Opting out builds
    /// a batch, runs it to completion and tears it down.
    #[test]
    fn the_default_options_take_the_persistent_batch_route() {
        use crate::voxcpm::options::VoxCpm2SynthOptions;
        assert!(VoxCpm2SynthOptions::default().continuous_batching);
        assert!(
            !VoxCpm2SynthOptions {
                continuous_batching: false,
                ..VoxCpm2SynthOptions::default()
            }
            .continuous_batching
        );
    }

    /// The cap the owner thread opens a batch at is one the loop can step
    /// at. The roundings themselves are covered where they are defined.
    #[test]
    fn the_engine_cap_is_a_width_the_loop_can_run() {
        for cap in [0usize, 1, 3, 7, 8, 64] {
            let clamped = clamp_max_batch(cap);
            assert_eq!(batch_width(clamped), clamped, "cap {cap}");
        }
    }

    /// Each real row keeps its own request's seed; a free slot caps at one
    /// patch.
    #[test]
    fn row_options_carry_each_rows_own_seed_and_pad_free_slots() {
        let real = [
            RowOptions {
                max_len: 30,
                seed: 7,
            },
            RowOptions {
                max_len: 12,
                seed: 1007,
            },
        ];
        let opts = row_options(&real, 4);
        assert_eq!(opts, [real[0], real[1], FREE_ROW, FREE_ROW]);
        assert_eq!(FREE_ROW.max_len, 1);
    }

    /// Row `b` of the options is job `b` of the batch, in queue order.
    #[test]
    fn row_options_keep_job_order() {
        let real: Vec<RowOptions> = [(5, 3), (6, 1), (7, 2)]
            .into_iter()
            .map(|(max_len, seed)| RowOptions { max_len, seed })
            .collect();
        let opts = row_options(&real, 4);
        let caps: Vec<usize> = opts.iter().map(|r| r.max_len).collect();
        let seeds: Vec<u64> = opts.iter().take(3).map(|r| r.seed).collect();
        assert_eq!(caps, [5, 6, 7, 1]);
        assert_eq!(seeds, [3, 1, 2]);
    }

    /// A job reports its own cap and seed to the loop.
    #[test]
    fn a_job_hands_the_loop_its_own_cap_and_seed() {
        let (tx, _rx) = channel();
        let job = TestJob {
            voice: JobVoice::ZeroShot,
            prompt: None,
            seed: 42,
            text_token_ids: vec![1, 2],
            max_len: 22,
            seq_len: 2,
            reply: Reply::Buffered(tx),
        };
        assert_eq!(
            job.row_options(),
            RowOptions {
                max_len: 22,
                seed: 42
            }
        );
    }
}
