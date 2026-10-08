//! Taking work off the owner thread's queue: block for the next render,
//! gather a batch within the drain window, and serve every reference or
//! prompt encode met on the way at once, without spending a batch slot on it.
//! A render whose sampler differs from its batch's goes to a held list.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use numr::runtime::Runtime;

use crate::voxcpm::engine::batch::{EncodeJob, Job, Work};

/// Takes from `held`, in order, every job whose sampler is `first`'s, until
/// `batch` holds `max_batch` jobs. The rest stay held in their order.
fn take_held<R: Runtime>(batch: &mut Vec<Job<R>>, held: &mut VecDeque<Job<R>>, max_batch: usize) {
    let Some(sampler) = batch.first().map(|job| job.sampler) else {
        return;
    };
    let mut at = 0;
    while at < held.len() && batch.len() < max_batch {
        if held[at].sampler == sampler {
            if let Some(job) = held.remove(at) {
                batch.push(job);
            }
        } else {
            at += 1;
        }
    }
}

/// Puts `job` in `batch` when the batch runs its sampler and no job with
/// another sampler waits in `held`. Otherwise it waits in `held`, behind
/// every job held before it, so no job waits behind later arrivals forever.
pub(super) fn route<R: Runtime>(job: Job<R>, batch: &mut Vec<Job<R>>, held: &mut VecDeque<Job<R>>) {
    let joins = held.is_empty()
        && batch
            .first()
            .is_none_or(|first| first.sampler == job.sampler);
    if joins {
        batch.push(job);
    } else {
        held.push_back(job);
    }
}

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

/// Take `first`, then every held job with its sampler, then keep taking
/// queued jobs until the batch is full, the window expires, or the queue
/// closes. An encode request met on the way is served at once and takes no
/// slot. A queued job goes through [`route`]: one with another sampler, or
/// any job once one is held, waits in `held` for a later batch.
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
    held: &mut VecDeque<Job<R>>,
) -> Vec<Job<R>> {
    let mut batch = Vec::with_capacity(max_batch.max(1));
    batch.push(first);
    take_held(&mut batch, held, max_batch);
    let deadline = Instant::now() + window;
    while batch.len() < max_batch {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match jobs.recv_timeout(deadline - now) {
            Ok(Work::Render(job)) => route(job, &mut batch, held),
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
    use crate::voxcpm::engine::batch::{JobVoice, test_sampler};
    use crate::voxcpm::engine::reply::Reply;
    use crate::voxcpm::engine::request::Badcase;
    use boostr::model::audio::voxcpm::model::PadSide;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;
    use std::sync::mpsc::channel;

    type TestJob = Job<CpuRuntime>;
    type TestWork = Work<CpuRuntime>;

    fn render() -> TestWork {
        Work::Render(job_at(0, 10))
    }

    /// A job at `seed` whose sampler runs `steps` steps.
    fn job_at(seed: u64, steps: usize) -> TestJob {
        // In the engine the job's owner keeps the receiver alive. Here
        // nothing reads it, so it is dropped.
        let (tx, _rx) = channel();
        TestJob {
            voice: JobVoice::ZeroShot,
            prompt: None,
            seed,
            sampler: test_sampler(steps),
            text_token_ids: vec![1, 2],
            max_len: 22,
            seq_len: 2,
            target_len: 2,
            badcase: Badcase::default(),
            attempt: 1,
            reply: Reply::Buffered(tx),
        }
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

    fn held() -> VecDeque<TestJob> {
        VecDeque::new()
    }

    fn seeds(batch: &[TestJob]) -> Vec<u64> {
        batch.iter().map(|job| job.seed).collect()
    }

    /// A job with another sampler waits in the held list, and so does every
    /// job queued after it. The next batch opens with the held job and takes
    /// the held jobs that share its sampler.
    #[test]
    fn a_job_with_another_sampler_waits_for_its_own_batch() {
        let (tx, rx) = channel();
        tx.send(Work::Render(job_at(1, 10))).expect("queue");
        tx.send(Work::Render(job_at(2, 16))).expect("queue");
        tx.send(Work::Render(job_at(3, 10))).expect("queue");
        tx.send(Work::Render(job_at(4, 16))).expect("queue");
        drop(tx);
        let mut served = 0;
        let mut held = held();
        let first = first_job(&rx, &mut served);
        let window = Duration::from_secs(30);
        let batch = collect_batch(&rx, first, window, 8, &mut |_| {}, &mut held);
        assert_eq!(seeds(&batch), [1]);
        assert_eq!(
            held.iter().map(|job| job.seed).collect::<Vec<_>>(),
            [2, 3, 4]
        );

        let first = held.pop_front().expect("a held job");
        let batch = collect_batch(&rx, first, window, 8, &mut |_| {}, &mut held);
        assert_eq!(seeds(&batch), [2, 4]);
        assert_eq!(held.iter().map(|job| job.seed).collect::<Vec<_>>(), [3]);
    }

    /// Held jobs with the batch's sampler join it up to the cap, in order.
    #[test]
    fn held_jobs_join_up_to_the_cap() {
        let (tx, rx) = channel::<TestWork>();
        drop(tx);
        let mut held: VecDeque<TestJob> =
            [job_at(2, 10), job_at(3, 16), job_at(4, 10), job_at(5, 10)].into();
        let batch = collect_batch(
            &rx,
            job_at(1, 10),
            Duration::ZERO,
            3,
            &mut |_| {},
            &mut held,
        );
        assert_eq!(seeds(&batch), [1, 2, 4]);
        assert_eq!(held.iter().map(|job| job.seed).collect::<Vec<_>>(), [3, 5]);
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
        let batch = collect_batch(&rx, first, window, 8, &mut |_| served += 1, &mut held());
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
        let batch = collect_batch(
            &rx,
            first,
            Duration::from_secs(30),
            2,
            &mut |_| {},
            &mut held(),
        );
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
        let batch = collect_batch(
            &rx,
            first,
            Duration::from_secs(30),
            8,
            &mut |_| {},
            &mut held(),
        );
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
        let batch = collect_batch(
            &rx,
            first,
            Duration::from_secs(30),
            1,
            &mut |_| {},
            &mut held(),
        );
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
        let batch = collect_batch(&rx, first, window, 8, &mut |_| {}, &mut held());
        let waited = started.elapsed();
        assert_eq!(batch.len(), 1);
        assert!(waited >= window, "returned before the window: {waited:?}");
        assert!(
            waited < Duration::from_secs(5),
            "waited far past the window"
        );
    }
}
