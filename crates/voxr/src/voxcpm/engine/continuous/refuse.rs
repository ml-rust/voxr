//! What a persistent batch refuses: a request no slot can hold, and every
//! queued message once the batch has stopped.

use std::sync::mpsc::Receiver;

use numr::runtime::Runtime;

use crate::error::Error;
use crate::voxcpm::engine::batch::{Job, Work};

/// Fail every queued and every later request, reference and prompt encodes
/// included, until the queue closes.
///
/// A caller blocks on its reply channel, so a stopped thread must keep
/// answering rather than stop reading the queue.
pub(super) fn refuse_queue<R: Runtime>(jobs: &Receiver<Work<R>>, err: &Error) {
    let reason = format!("VoxCPM2 continuous batch is not running: {err}");
    while let Ok(work) = jobs.recv() {
        let refusal = Error::ModelError {
            reason: reason.clone(),
        };
        match work {
            Work::Render(job) => job.reply.fail(refusal),
            Work::Encode(encode) => {
                let _ = encode.reply.send(Err(refusal));
            }
        }
    }
}

/// A request no slot of a persistent batch can hold.
pub(super) fn too_large<R: Runtime>(job: &Job<R>, capacity: usize) -> Error {
    Error::InvalidArgument {
        arg: "text",
        reason: format!(
            "this request needs {} cache positions ({} for its prefill, {} for its patches, \
             1 past the last) but a continuous batch's slot holds {capacity}; shorten the \
             text, or use a shorter reference or prompt recording",
            job.seq_len + job.max_len + 1,
            job.seq_len,
            job.max_len,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::voxcpm::engine::batch::{EncodeJob, JobVoice, test_sampler};
    use crate::voxcpm::engine::reply::Reply;
    use crate::voxcpm::engine::request::{Badcase, Rendered};
    use boostr::model::audio::voxcpm::model::PadSide;
    use numr::runtime::cpu::CpuRuntime;
    use std::sync::mpsc::channel;

    fn job(
        seq_len: usize,
        max_len: usize,
    ) -> (Job<CpuRuntime>, std::sync::mpsc::Receiver<Result<Rendered>>) {
        let (tx, rx) = channel();
        (
            Job {
                voice: JobVoice::ZeroShot,
                prompt: None,
                seed: 0,
                sampler: test_sampler(10),
                text_token_ids: vec![1, 2],
                max_len,
                seq_len,
                target_len: 2,
                badcase: Badcase::default(),
                attempt: 1,
                reply: Reply::Buffered(tx),
            },
            rx,
        )
    }

    /// The default slot size holds an ordinary clone request: a reference
    /// recording of a few hundred patches, a sentence of around a hundred
    /// tokens, and the patches that sentence generates.
    ///
    /// A default that stops holding an ordinary request refuses one at the
    /// engine's door.
    #[test]
    fn the_default_slot_holds_an_ordinary_request() {
        use crate::voxcpm::options::VoxCpm2SynthOptions;
        let (t_ref, text_len) = (256usize, 96usize);
        let prefill = t_ref + 2 + text_len;
        let patches = text_len * 6 + 10;
        let capacity = VoxCpm2SynthOptions::default().continuous_capacity;
        assert!(
            prefill + patches < capacity,
            "prefill {prefill} + patches {patches} must leave the {capacity}-position slot \
             the one position the lane table advances past the last patch"
        );
    }

    /// A request whose prefill plus patches fills the slot exactly is one
    /// position too long. The lane table advances once past the last patch.
    #[test]
    fn a_request_that_cannot_fit_a_slot_is_refused_with_its_own_numbers() {
        let (job, rx) = job(100, 28);
        job.reply.fail(too_large(&job, 128));
        let err = rx.recv().expect("reply").expect_err("refused");
        let text = err.to_string();
        assert!(text.contains("129"), "{text}");
        assert!(text.contains("100"), "{text}");
        assert!(text.contains("128"), "{text}");
    }

    /// The refusal reaches the caller rather than leaving it queued.
    #[test]
    fn a_refused_queue_answers_every_waiting_caller() {
        let (tx, rx) = channel();
        let (first, first_reply) = job(4, 4);
        let (second, second_reply) = job(4, 4);
        let (encode_reply, encode_rx) = channel();
        tx.send(Work::Render(first)).expect("queue");
        tx.send(Work::Encode(EncodeJob {
            samples: vec![0.0; 4],
            side: PadSide::Right,
            reply: encode_reply,
        }))
        .expect("queue");
        tx.send(Work::Render(second)).expect("queue");
        drop(tx);

        refuse_queue(
            &rx,
            &Error::ModelError {
                reason: "boom".into(),
            },
        );
        for reply in [first_reply, second_reply] {
            let err = reply.recv().expect("reply").expect_err("refused");
            assert!(err.to_string().contains("boom"), "{err}");
        }
        let err = encode_rx.recv().expect("reply").expect_err("refused");
        assert!(err.to_string().contains("boom"), "{err}");
    }
}
