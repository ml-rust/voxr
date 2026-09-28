//! Taking work off the owner thread's queue: block for the next render,
//! gather a batch within the drain window, and serve every reference or
//! prompt encode met on the way at once, without spending a batch slot on it.

use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use numr::runtime::Runtime;

use crate::voxcpm::engine::batch::{EncodeJob, Job, Work};

/// Block for the next render job, serving every encode request that
/// arrives before it. `None` once the queue closes.
pub(super) fn recv_job<R: Runtime>(
    jobs: &Receiver<Work<R>>,
    serve: &mut dyn FnMut(EncodeJob<R>),
) -> Option<Job<R>> {
    loop {
        match jobs.recv() {
            Ok(Work::Render(job)) => return Some(job),
            Ok(Work::Encode(encode)) => serve(encode),
            Err(_) => return None,
        }
    }
}

/// Take `first`, then keep taking queued jobs until the batch is full, the
/// window expires, or the queue closes. An encode request met on the way is
/// served at once and takes no slot.
///
/// The owner thread already blocked on an empty queue to get `first`, so this
/// is the only wait a request pays beyond its own generation. It ends early
/// on a full batch and on a closed queue, so a lone request on an engine
/// being dropped is not held for the window.
pub(super) fn collect_batch<R: Runtime>(
    jobs: &Receiver<Work<R>>,
    first: Job<R>,
    window: Duration,
    max_batch: usize,
    serve: &mut dyn FnMut(EncodeJob<R>),
) -> Vec<Job<R>> {
    let mut batch = Vec::with_capacity(max_batch.max(1));
    batch.push(first);
    let deadline = Instant::now() + window;
    while batch.len() < max_batch {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match jobs.recv_timeout(deadline - now) {
            Ok(Work::Render(job)) => batch.push(job),
            Ok(Work::Encode(encode)) => serve(encode),
            Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
        }
    }
    batch
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::voxcpm::engine::batch::JobVoice;
    use crate::voxcpm::engine::reply::Reply;
    use boostr::model::audio::voxcpm::model::PadSide;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;
    use std::sync::mpsc::channel;

    type TestJob = Job<CpuRuntime>;
    type TestWork = Work<CpuRuntime>;

    fn render() -> TestWork {
        // In the engine the job's owner keeps the receiver alive. Here
        // nothing reads it, so it is dropped.
        let (tx, _rx) = channel();
        Work::Render(TestJob {
            voice: JobVoice::ZeroShot,
            prompt: None,
            seed: 0,
            text_token_ids: vec![1, 2],
            max_len: 22,
            seq_len: 2,
            reply: Reply::Buffered(tx),
        })
    }

    fn encode() -> (TestWork, Receiver<Result<Tensor<CpuRuntime>>>) {
        let (tx, rx) = channel();
        (
            Work::Encode(EncodeJob {
                samples: vec![0.0; 4],
                side: PadSide::Right,
                reply: tx,
            }),
            rx,
        )
    }

    /// Take the first render off the queue, counting the encodes served
    /// before it.
    fn first_job(rx: &Receiver<TestWork>, served: &mut usize) -> TestJob {
        recv_job(rx, &mut |_| *served += 1).expect("a render is queued")
    }

    /// Encodes queued ahead of a render are served, in order, before the
    /// render comes back.
    #[test]
    fn encodes_ahead_of_a_render_are_served_first() {
        let (tx, rx) = channel();
        let (first, _first_rx) = encode();
        let (second, _second_rx) = encode();
        tx.send(first).expect("queue");
        tx.send(second).expect("queue");
        tx.send(render()).expect("queue");
        let mut served = 0;
        first_job(&rx, &mut served);
        assert_eq!(served, 2);
    }

    /// A closed queue with only encodes left serves them and ends.
    #[test]
    fn a_closed_queue_serves_what_is_left_and_ends() {
        let (tx, rx) = channel();
        let (only, _only_rx) = encode();
        tx.send(only).expect("queue");
        drop(tx);
        let mut served = 0;
        assert!(recv_job(&rx, &mut |_| served += 1).is_none());
        assert_eq!(served, 1);
    }

    /// Everything queued inside the window rides in one batch. An empty queue
    /// does not end the drain early, because the window exists so a request
    /// arriving a moment later still joins. An encode in between takes no
    /// slot.
    #[test]
    fn queued_jobs_batch_together() {
        let (tx, rx) = channel();
        for _ in 0..3 {
            tx.send(render()).expect("queue");
        }
        let (between, _between_rx) = encode();
        tx.send(between).expect("queue");
        let mut served = 0;
        let first = first_job(&rx, &mut served);
        let window = Duration::from_millis(50);
        let started = Instant::now();
        let batch = collect_batch(&rx, first, window, 8, &mut |_| served += 1);
        assert_eq!(batch.len(), 3);
        assert_eq!(served, 1, "the encode was served, not batched");
        assert!(started.elapsed() >= window, "ended the window early");
    }

    /// A full batch starts at once, whatever is left in the window.
    #[test]
    fn a_full_batch_starts_without_waiting() {
        let (tx, rx) = channel();
        for _ in 0..4 {
            tx.send(render()).expect("queue");
        }
        let mut served = 0;
        let first = first_job(&rx, &mut served);
        let started = Instant::now();
        let batch = collect_batch(&rx, first, Duration::from_secs(30), 2, &mut |_| {});
        assert_eq!(batch.len(), 2, "the cap holds");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "waited the window out"
        );
    }

    /// Nothing else can arrive on a closed queue, so a lone job does not
    /// wait the window out.
    #[test]
    fn a_lone_job_on_a_closed_queue_starts_at_once() {
        let (tx, rx) = channel();
        tx.send(render()).expect("queue");
        drop(tx);
        let mut served = 0;
        let first = first_job(&rx, &mut served);
        let started = Instant::now();
        let batch = collect_batch(&rx, first, Duration::from_secs(30), 8, &mut |_| {});
        assert_eq!(batch.len(), 1);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "waited the window out"
        );
    }

    /// A cap of one turns batching off and skips the window with it.
    #[test]
    fn a_cap_of_one_never_waits() {
        let (tx, rx) = channel();
        for _ in 0..2 {
            tx.send(render()).expect("queue");
        }
        let mut served = 0;
        let first = first_job(&rx, &mut served);
        let started = Instant::now();
        let batch = collect_batch(&rx, first, Duration::from_secs(30), 1, &mut |_| {});
        assert_eq!(batch.len(), 1);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "waited with batching off"
        );
    }

    /// A lone job on an open queue costs the window and no more.
    #[test]
    fn a_lone_job_costs_the_window_and_no_more() {
        let (tx, rx) = channel();
        tx.send(render()).expect("queue");
        let mut served = 0;
        let first = first_job(&rx, &mut served);
        let window = Duration::from_millis(20);
        let started = Instant::now();
        let batch = collect_batch(&rx, first, window, 8, &mut |_| {});
        let waited = started.elapsed();
        assert_eq!(batch.len(), 1);
        assert!(waited >= window, "returned before the window: {waited:?}");
        assert!(
            waited < Duration::from_secs(5),
            "waited far past the window"
        );
    }
}
