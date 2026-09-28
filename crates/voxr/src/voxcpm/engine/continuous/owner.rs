//! The owner thread's continuous-batching loop: one persistent
//! [`GenerateState`], its slot table, and the admit/step/release cycle.
//! A reference encode queued meanwhile runs between two steps.

use std::sync::mpsc::Receiver;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::{
    GenerateOptions, GenerateState, PrefillRow, VoxCpm2Model,
};

use crate::error::{Error, Result};
use crate::voxcpm::engine::batch::{EncodeJob, FREE_ROW, Job, Work, clamp_max_batch};
use crate::voxcpm::engine::continuous::refuse::{refuse_queue, too_large};
use crate::voxcpm::engine::continuous::slots::{Slots, drain};
use crate::voxcpm::engine::queue::{collect_batch, recv_job};
use crate::voxcpm::engine::run::Row;
use crate::voxcpm::engine::types::{EngineInner, MAX_LEN_CAP, VoxCpm2RenderClient};

/// Run one persistent batch until the queue closes.
///
/// The cycle: take whatever is queued into free slots, step the batch once,
/// then reply for and release every slot that finished. Only the cold start
/// blocks, because an empty batch has nothing to step. A reference encode
/// taken off the queue runs at once, between two steps.
///
/// A failure to open the batch, or a failed step, is not survivable. Every
/// row reads the state behind it. Every waiting caller is failed and every
/// later request refused, so no caller blocks on a stopped thread.
pub(in crate::voxcpm::engine) fn run_continuous<R>(inner: &EngineInner<R>, jobs: &Receiver<Work<R>>)
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    let width = clamp_max_batch(inner.options.max_batch);
    let capacity = match slot_capacity(inner) {
        Ok(capacity) => capacity,
        Err(err) => return refuse_queue(jobs, &err),
    };
    let mut batch = match Batch::open(inner, width, capacity) {
        Ok(batch) => batch,
        Err(err) => return refuse_queue(jobs, &err),
    };

    let mut serve = |encode: EncodeJob<R>| inner.serve_encode(encode);
    let mut closed = false;
    loop {
        if !closed {
            closed = batch.admit_queued(inner, jobs, &mut serve);
        }
        if !batch.slots.any_occupied() {
            if closed {
                return;
            }
            match recv_job(jobs, &mut serve) {
                Some(first) => batch.cold_start(inner, jobs, first, &mut serve),
                None => return,
            }
            // Every job of the cold start can have been refused. Go round
            // again to block for the next one rather than step an empty
            // batch.
            continue;
        }
        if let Err(err) = batch.step(inner) {
            batch.fail_rows(&err);
            drop(batch);
            return refuse_queue(jobs, &err);
        }
    }
}

/// Cache positions one slot is opened with.
///
/// The budget is
/// [`VoxCpm2SynthOptions::continuous_capacity`](crate::voxcpm::VoxCpm2SynthOptions::continuous_capacity),
/// which a caller sizes to the requests it serves. The model's rope table
/// caps it: a position the table does not hold cannot be rotated, so a longer
/// cache can be filled and never read.
///
/// Deliberately NOT derived from [`MAX_LEN_CAP`]. That is the ceiling ONE
/// request can reach, and a persistent batch holds `max_batch` slots of this
/// size in two caches for its whole life.
fn slot_capacity<R: Runtime<DType = DType>>(inner: &EngineInner<R>) -> Result<usize> {
    let want = inner.options.continuous_capacity;
    Ok(match inner.model.lane_rope_positions()? {
        Some(positions) => want.min(positions),
        None => want,
    })
}

/// One persistent batch: the state every slot steps in, the rows seated in
/// it, and the per-row options the loop reads.
struct Batch<R: Runtime<DType = DType>> {
    state: GenerateState<R>,
    slots: Slots<Row<R>>,
    options: GenerateOptions,
    /// Cache positions one slot holds — see [`slot_capacity`].
    capacity: usize,
}

impl<R> Batch<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Open an empty batch of `width` slots.
    ///
    /// `width` is the CEILING, fixed for the batch's whole life. Nothing is
    /// seated above it. The device tensors start at one row and
    /// [`GenerateState`] widens them as slots above the allocation are
    /// seated, so an idle batch holds one row.
    ///
    /// One step runs over the prefix of slots holding a request, which
    /// [`GenerateState`] recomputes on every seat and release. That stepped
    /// width is the DiT Euler graph cache's shape key and takes one of a
    /// short list of values, so the captures are shared.
    fn open(inner: &EngineInner<R>, width: usize, capacity: usize) -> Result<Self> {
        // `rows` names every slot's seed, so the load-time seed here is only
        // the loop's fallback and reaches no row.
        let mut options = GenerateOptions::new(MAX_LEN_CAP, inner.options.seed);
        options.cfm.n_timesteps = inner.options.n_timesteps;
        options.cfm.cfg_value = inner.options.cfg_value;
        options.min_len = inner.options.min_len;
        let mut batch = Self {
            state: GenerateState::open(&inner.model, width, capacity)?,
            slots: Slots::new(width),
            options,
            capacity,
        };
        batch.refresh_options();
        Ok(batch)
    }

    /// Take whatever is queued into the free slots, serving any reference
    /// encode met on the way. Returns `true` once the queue has closed.
    fn admit_queued(
        &mut self,
        inner: &EngineInner<R>,
        jobs: &Receiver<Work<R>>,
        serve: &mut dyn FnMut(EncodeJob<R>),
    ) -> bool {
        let (taken, closed) = drain(jobs, self.slots.free_count(), serve);
        for job in taken {
            self.seat_next(inner, job);
        }
        closed
    }

    /// Fill an empty batch, starting from the job the thread just blocked
    /// for.
    ///
    /// This is the ONE place the drain window applies. In the steady state a
    /// joiner is taken between two steps and waits for neither the window nor
    /// the batch. A request arriving alone pays the window once, here.
    fn cold_start(
        &mut self,
        inner: &EngineInner<R>,
        jobs: &Receiver<Work<R>>,
        first: Job<R>,
        serve: &mut dyn FnMut(EncodeJob<R>),
    ) {
        let free = self.slots.free_count();
        for job in collect_batch(jobs, first, inner.options.batch_window, free, serve) {
            self.seat_next(inner, job);
        }
    }

    /// Seat `job` in the lowest free slot.
    fn seat_next(&mut self, inner: &EngineInner<R>, job: Job<R>) {
        match self.slots.free() {
            Some(slot) => self.seat(inner, slot, job),
            None => job.reply.fail(Error::ModelError {
                reason: format!(
                    "VoxCPM2 continuous batch took a request with no free slot among its {}; \
                     this is an engine bug, report it with the batch width",
                    self.slots.width()
                ),
            }),
        }
    }

    /// Prefill `job` on its own and seat it in slot `slot`.
    ///
    /// The prefill runs here, on the owner thread, between two steps. It is a
    /// different sequence length from a decode step, and a lane is installed
    /// from a batch of one whose keys start at cache position 0.
    ///
    /// Anything that refuses the request — an unknown voice, a request too
    /// long for a slot, a prefill error — is reported on the job's own reply
    /// and leaves the slot free and the batch untouched.
    fn seat(&mut self, inner: &EngineInner<R>, slot: usize, job: Job<R>) {
        // One past the last patch is where the lane table advances to on the
        // final step, so the budget is strictly below the slot's capacity.
        if job.seq_len + job.max_len >= self.capacity {
            job.reply.fail(too_large(&job, self.capacity));
            return;
        }
        let ref_feat = match inner.reference(&job.voice) {
            Ok(ref_feat) => ref_feat,
            Err(err) => return job.reply.fail(err),
        };
        let client = inner.client.as_ref();
        // The handoff cache holds this row's prefix and nothing else. Its
        // keys are copied into the slot's lane from position 0, so sizing it
        // to the slot's whole capacity reserves the batch's budget a second
        // time on every admission. The prefill itself refuses a length that
        // does not match the sequence the layout builds.
        let prefilled = match inner.model.prefill_batch(
            client,
            &[PrefillRow {
                ref_feat,
                text_token_ids: &job.text_token_ids,
            }],
            job.seq_len,
        ) {
            Ok(prefilled) => prefilled,
            Err(err) => return job.reply.fail(Error::from(err)),
        };
        if let Err(err) = self
            .state
            .admit(client, slot, &prefilled, job.row_options())
        {
            return job.reply.fail(Error::from(err));
        }
        self.slots.seat(
            slot,
            Row {
                job,
                live: true,
                emitted: 0,
            },
        );
        self.refresh_options();
    }

    /// Step the seated slots once, then reply for and release the slots that
    /// finished.
    ///
    /// The step covers the prefix of slots holding a request, not the whole
    /// allocation. The retire walk below covers every slot, because the slot
    /// table and the per-row records are sized by the allocation.
    ///
    /// A released slot's row is DROPPED here, which drops its reply sender.
    /// That ends a streaming caller's drain loop, whether the row ran to its
    /// stop token or was retired mid-stream.
    fn step(&mut self, inner: &EngineInner<R>) -> Result<()> {
        let client = inner.client.as_ref();
        let chunk = inner.options.stream_chunk_patches.max(1);
        inner
            .model
            .patch_generator()
            .step(client, &mut self.state, &self.options)?;

        let mut released = false;
        for slot in 0..self.slots.width() {
            let retire = {
                let Some(row) = self.slots.get_mut(slot) else {
                    continue;
                };
                inner.pump_row(&mut self.state, slot, row, chunk);
                let done = !row.live || self.state.finished(slot);
                if done {
                    inner.deliver_row(&self.state, slot, row);
                }
                done
            };
            if retire {
                self.slots.take(slot);
                self.state.release(client, slot)?;
                released = true;
            }
        }
        if released {
            self.refresh_options();
        }
        Ok(())
    }

    /// Report `err` on every row still seated and empty the batch.
    fn fail_rows(&mut self, err: &Error) {
        for (_, row) in self.slots.drain_all() {
            if row.live {
                row.job.reply.fail(Error::ModelError {
                    reason: err.to_string(),
                });
            }
        }
    }

    /// Rebuild the per-row caps and seeds from the state's own slot record.
    ///
    /// `admit` and `release` write [`GenerateState::slot_options`], so the
    /// loop's caps and seeds cannot disagree with the slots they belong to.
    /// The options carry one entry per slot, and a free slot takes
    /// [`FREE_ROW`].
    fn refresh_options(&mut self) {
        self.options.rows = self
            .state
            .slot_options
            .iter()
            .map(|seated| seated.unwrap_or(FREE_ROW))
            .collect();
    }
}
