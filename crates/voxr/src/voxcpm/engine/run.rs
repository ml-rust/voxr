//! What the owner thread does with one batch: resolve each row's reference
//! voice, prefill each request on its own and seat it into a slot, drive the
//! per-patch loop, and hand each row its audio.
//!
//! A row's reference is either a voice-table tensor in [`EngineInner`] or
//! the tensor its job holds an [`Arc`](std::sync::Arc) of. Both are looked
//! up where the prefill reads them, so a row stores no borrow. A
//! continuation prompt is always the job's own tensor.
//!
//! Every decode goes through boostr's `decode_row` and `decode_row_from`.
//! Both decode a continuation row behind its prompt context and trim the
//! context's samples, so neither a buffered reply nor a streamed chunk holds
//! prompt audio.
//!
//! Every row of a batch steps on every iteration; a finished row's patches
//! are computed and its bookkeeping stands still. That is what lets a row be
//! retired mid-batch — a cancelled streaming row is marked finished and runs
//! dead to the end, exactly as a row that stopped early does.
//!
//! A row's noise draw depends on neither the batch width nor its neighbours,
//! so a batched row is bit for bit the render it would have had alone.

use std::sync::atomic::Ordering;

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use boostr::model::audio::voxcpm::model::{
    GenerateOptions, GenerateOutcome, GenerateState, PrefillRow, StepOutcome, VoxCpm2Model,
};

use crate::error::{Error, Result};
use crate::voxcpm::engine::batch::{Job, JobVoice, batch_width, row_options};
use crate::voxcpm::engine::types::{EngineInner, VoxCpm2RenderClient};

/// One row of a running batch: the request, and whether the caller is still
/// taking output.
pub(super) struct Row<R: Runtime> {
    pub(super) job: Job<R>,
    /// Cleared once the row's caller went away or its own decode failed.
    /// A dead row keeps stepping; nothing is decoded or sent for it again.
    pub(super) live: bool,
    /// Patches already handed to a streaming caller.
    pub(super) emitted: usize,
}

impl<R> EngineInner<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: VoxCpm2RenderClient<R>,
    VoxCpm2Model<R>: Send + Sync,
    Tensor<R>: Send + Sync,
{
    /// Generate every job of one batch together and reply to each.
    ///
    /// Never propagates: this is the owner thread's top level, so every
    /// failure leaves through the rows' reply channels.
    pub(super) fn run_batch(&self, jobs: Vec<Job<R>>) {
        let mut rows = Vec::with_capacity(jobs.len());
        for job in jobs {
            let known = self.reference(&job.voice).map(|_| ());
            match known {
                Ok(()) => rows.push(Row {
                    job,
                    live: true,
                    emitted: 0,
                }),
                Err(err) => job.reply.fail(err),
            }
        }
        if rows.is_empty() {
            return;
        }
        if let Err(err) = self.generate_batch(&mut rows) {
            fail_all(&rows, err);
        }
    }

    /// A job voice's encoded reference: the voice-table tensor for an id,
    /// the job's own tensor for a request reference, `None` for zero-shot.
    /// The voice table is fixed at load, so an unknown id here means the
    /// caller's own check and this one disagree.
    pub(super) fn reference<'a>(&'a self, voice: &'a JobVoice<R>) -> Result<Option<&'a Tensor<R>>> {
        match voice {
            JobVoice::ZeroShot => Ok(None),
            JobVoice::Reference(feat) => Ok(Some(feat.as_ref())),
            JobVoice::Named(voice_id) => {
                let voice = self
                    .voices
                    .get(voice_id)
                    .ok_or_else(|| Error::InvalidArgument {
                        arg: "voice",
                        reason: format!(
                            "voice {voice_id:?} passed the request check but is not in the \
                             engine's voice table; this is an engine bug, report it with \
                             the voice id"
                        ),
                    })?;
                Ok(Some(&voice.ref_feat))
            }
        }
    }

    /// Prefill, generate and decode the whole batch. An error here belongs
    /// to every row; a per-row failure is reported on that row and does not
    /// return.
    fn generate_batch(&self, rows: &mut [Row<R>]) -> Result<()> {
        let width = batch_width(rows.len());
        let real: Vec<_> = rows.iter().map(|row| row.job.row_options()).collect();
        let max_len = real.iter().map(|row| row.max_len).max().unwrap_or(1);
        let s_max = rows.iter().map(|row| row.job.seq_len).max().unwrap_or(1);
        // One capacity for every slot: the longest prefill in this batch plus
        // the largest patch budget any of its rows asked for.
        let capacity = s_max + max_len;

        // `rows` names every slot's seed, so the load-time seed here is only
        // the loop's fallback and reaches no row.
        let mut options = GenerateOptions::new(max_len, self.options.seed);
        options.cfm.n_timesteps = self.options.n_timesteps;
        options.cfm.cfg_value = self.options.cfg_value;
        options.min_len = self.options.min_len;
        options.rows = row_options(&real, width);

        // Each request prefills on its own and is seated into its own slot.
        // The slots past the last request stay free: they still run every
        // sub-step, on an empty key range, and count no patch.
        let client = self.client.as_ref();
        let mut state = GenerateState::open(&self.model, width, capacity)?;
        for (b, row) in rows.iter().enumerate() {
            let prefilled = self.model.prefill_batch(
                client,
                &[PrefillRow {
                    ref_feat: self.reference(&row.job.voice)?,
                    text_token_ids: &row.job.text_token_ids,
                    prompt_feat: row.job.prompt.as_deref(),
                }],
                capacity,
            )?;
            state.admit(client, b, &prefilled, options.row(b))?;
        }

        self.drive(&mut state, &options, rows)?;
        self.deliver(&state, rows);
        Ok(())
    }

    /// Step the batch to the end, emitting streaming chunks as they become
    /// decodable and retiring rows whose callers went away.
    fn drive(
        &self,
        state: &mut GenerateState<R>,
        options: &GenerateOptions,
        rows: &mut [Row<R>],
    ) -> Result<()> {
        let chunk = self.options.stream_chunk_patches.max(1);
        let client = self.client.as_ref();
        let generator = self.model.patch_generator();
        while !state.all_finished() {
            let outcome = generator.step(client, state, options)?;
            for (b, row) in rows.iter_mut().enumerate() {
                self.pump_row(state, b, row, chunk);
            }
            if outcome == StepOutcome::Stopped {
                break;
            }
        }
        Ok(())
    }

    /// Hand a streaming row whatever it has ready, and retire it once its
    /// caller has gone away. A buffered row emits nothing before the end, so
    /// this leaves it alone.
    ///
    /// Called once per row per step by both the whole-batch driver and the
    /// continuous one, so a row's chunk boundaries do not depend on which
    /// loop is running it.
    pub(super) fn pump_row(
        &self,
        state: &mut GenerateState<R>,
        b: usize,
        row: &mut Row<R>,
        chunk: usize,
    ) {
        if !row.live {
            return;
        }
        let Some((_, cancelled)) = row.job.reply.stream() else {
            return;
        };
        if cancelled.load(Ordering::Relaxed) {
            retire(state, b);
            row.live = false;
            return;
        }
        if state.patch_len[b] - row.emitted < chunk {
            return;
        }
        let client = self.client.as_ref();
        match host_samples(self.model.decode_row_from(client, state, b, row.emitted)) {
            Ok(samples) => {
                row.emitted = state.patch_len[b];
                if !row.job.reply.send_chunk(samples) {
                    retire(state, b);
                    row.live = false;
                }
            }
            Err(err) => fail_row(state, b, row, err),
        }
    }

    /// Final reply per row: the whole waveform for a buffered caller, the
    /// tail chunk for a streaming one. A streaming caller sees the end when
    /// its sender drops with the batch.
    fn deliver(&self, state: &GenerateState<R>, rows: &mut [Row<R>]) {
        for (b, row) in rows.iter_mut().enumerate() {
            self.deliver_row(state, b, row);
        }
    }

    /// Row `b`'s last message: the whole waveform for a buffered caller, the
    /// patches a streaming one has not seen yet. The row is dead afterwards
    /// either way; its caller sees the end when the row's sender drops.
    pub(super) fn deliver_row(&self, state: &GenerateState<R>, b: usize, row: &mut Row<R>) {
        if !row.live {
            return;
        }
        row.live = false;
        let client = self.client.as_ref();
        let streaming = row.job.reply.stream().is_some();
        let decoded = if streaming {
            if row.emitted >= state.patch_len[b] {
                return;
            }
            self.model.decode_row_from(client, state, b, row.emitted)
        } else {
            self.model.decode_row(client, state, b)
        };
        match host_samples(decoded) {
            Ok(samples) if streaming => {
                row.emitted = state.patch_len[b];
                row.job.reply.send_chunk(samples);
            }
            Ok(samples) => row.job.reply.send_samples(samples),
            Err(err) => row.job.reply.fail(err),
        }
    }
}

/// A decoded `[1, 1, samples]` tensor as host floats.
pub(super) fn host_samples<R: Runtime<DType = DType>>(
    decoded: std::result::Result<Tensor<R>, boostr::error::Error>,
) -> Result<Vec<f32>> {
    let decoded = decoded?;
    Ok(decoded.contiguous()?.to_vec())
}

/// Mark row `b` finished so the loop stops advancing its bookkeeping. The
/// row's patches are still computed on every remaining step — nothing skips
/// a row — but nothing reads them again.
pub(super) fn retire<R: Runtime>(state: &mut GenerateState<R>, b: usize) {
    if let Some(outcome) = state.outcomes.get_mut(b) {
        outcome.get_or_insert(GenerateOutcome::MaxLen);
    }
}

/// Report a per-row failure and retire the row; the rest of the batch runs on.
fn fail_row<R: Runtime>(state: &mut GenerateState<R>, b: usize, row: &mut Row<R>, err: Error) {
    row.job.reply.fail(err);
    retire(state, b);
    row.live = false;
}

/// Report one batch-wide failure to every row still waiting.
///
/// The error moves to the first live row unchanged, so a batch of one keeps
/// the exact error a single render returns today; the rest carry its text.
pub(super) fn fail_all<R: Runtime>(rows: &[Row<R>], err: Error) {
    let text = err.to_string();
    let mut original = Some(err);
    for row in rows.iter().filter(|row| row.live) {
        match original.take() {
            Some(err) => row.job.reply.fail(err),
            None => row.job.reply.fail(Error::ModelError {
                reason: text.clone(),
            }),
        }
    }
}
