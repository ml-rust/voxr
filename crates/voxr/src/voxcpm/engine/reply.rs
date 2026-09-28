//! Where one queued job's output goes: a whole waveform for a buffered
//! caller, or decoded chunks for a streaming one.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use crate::error::{Error, Result};

/// Where one job's output goes.
pub(super) enum Reply {
    /// A whole-utterance caller: one message, the samples or the error.
    Buffered(Sender<Result<Vec<f32>>>),
    /// A streaming caller: one message per decoded chunk, then the sender
    /// drops, which is how the caller sees the end of the row. The caller's
    /// sink is not `Send`, so it stays on the caller's thread and the chunks
    /// cross the channel instead.
    Streaming {
        chunks: Sender<Result<Vec<f32>>>,
        /// Set by the caller when its sink failed. The owner thread stops
        /// decoding for this row and retires it.
        cancelled: Arc<AtomicBool>,
    },
}

impl Reply {
    /// The streaming channel and its cancel flag, or `None` for a buffered
    /// reply.
    pub(super) fn stream(&self) -> Option<(&Sender<Result<Vec<f32>>>, &AtomicBool)> {
        match self {
            Self::Buffered(_) => None,
            Self::Streaming { chunks, cancelled } => Some((chunks, cancelled)),
        }
    }

    /// Hand one decoded chunk to a streaming caller. `false` once the caller
    /// has cancelled or dropped its receiver, after which the row is dead
    /// and gets no further chunks.
    pub(super) fn send_chunk(&self, samples: Vec<f32>) -> bool {
        let Some((chunks, cancelled)) = self.stream() else {
            return false;
        };
        !cancelled.load(Ordering::Relaxed) && chunks.send(Ok(samples)).is_ok()
    }

    /// Hand the whole waveform to a buffered caller.
    pub(super) fn send_samples(&self, samples: Vec<f32>) {
        if let Self::Buffered(done) = self {
            let _ = done.send(Ok(samples));
        }
    }

    /// Report a failure on this row. A caller that already went away is not
    /// an error in turn, so the send result is dropped.
    pub(super) fn fail(&self, err: Error) {
        let sender = match self {
            Self::Buffered(done) => done,
            Self::Streaming { chunks, .. } => chunks,
        };
        let _ = sender.send(Err(err));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{Receiver, channel};

    fn buffered() -> (Reply, Receiver<Result<Vec<f32>>>) {
        let (tx, rx) = channel();
        (Reply::Buffered(tx), rx)
    }

    fn streaming() -> (Reply, Receiver<Result<Vec<f32>>>, Arc<AtomicBool>) {
        let (tx, rx) = channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        (
            Reply::Streaming {
                chunks: tx,
                cancelled: Arc::clone(&cancelled),
            },
            rx,
            cancelled,
        )
    }

    #[test]
    fn a_cancelled_streaming_row_receives_no_further_chunks() {
        let (reply, rx, cancelled) = streaming();
        assert!(reply.send_chunk(vec![0.5, 0.25]));
        cancelled.store(true, Ordering::Relaxed);
        assert!(!reply.send_chunk(vec![1.0]));
        let first = rx.recv().expect("first chunk").expect("ok");
        assert_eq!(first, [0.5, 0.25]);
        drop(reply);
        assert!(rx.recv().is_err(), "a cancelled row sent a second chunk");
    }

    #[test]
    fn a_disconnected_streaming_row_stops_receiving_chunks() {
        let (reply, rx, _cancelled) = streaming();
        drop(rx);
        assert!(!reply.send_chunk(vec![1.0]));
    }

    #[test]
    fn a_buffered_reply_takes_samples_not_chunks() {
        let (reply, rx) = buffered();
        assert!(!reply.send_chunk(vec![1.0]));
        reply.send_samples(vec![0.5]);
        let got = rx.recv().expect("reply").expect("ok");
        assert_eq!(got, [0.5]);
    }

    #[test]
    fn a_failure_reaches_either_kind_of_caller() {
        let (reply, rx) = buffered();
        reply.fail(Error::ModelError {
            reason: "boom".into(),
        });
        let err = rx.recv().expect("reply").expect_err("error");
        assert!(err.to_string().contains("boom"), "{err}");

        let (reply, rx, _cancelled) = streaming();
        reply.fail(Error::ModelError {
            reason: "bang".into(),
        });
        let err = rx.recv().expect("reply").expect_err("error");
        assert!(err.to_string().contains("bang"), "{err}");
    }
}
