//! VoxCPM2 as a [`TtsEngine`](crate::tts::TtsEngine): the clone pipeline
//! behind one `synthesize`, and the same pipeline emitting audio chunk by
//! chunk behind `synthesize_stream`.
//!
//! A voice is a reference recording. The engine encodes every recording in a
//! voices directory once at load, so a request pays only its own prefill,
//! generation and decode. A caller encodes a recording it holds with
//! `encode_reference` and names the returned [`Reference`] in any number of
//! [`SynthRequest`]s. Text reaches the model as raw tokens, so a Malay and
//! English code-switched sentence needs no language switch.
//!
//! Every [`SynthRequest`] carries its own seed, or takes the load-time seed.
//!
//! Concurrent requests share one generation. Both entry points stay `&self`
//! and blocking: they resolve their request, queue it and wait, while the
//! engine's own thread runs whatever is queued as ONE batched generation. A
//! row's noise draw is independent of the batch width and its neighbours, so
//! a batched render is bit for bit the render that row has alone.
//!
//! [`types`] holds the engine struct, the shared state and the constants,
//! [`load`] the loader and voice-directory handling, [`request`] the request
//! and reference types, [`render`] the caller's half of a request, [`batch`]
//! the queued messages and the owner thread, [`queue`] taking work off the
//! queue, [`reply`] where a row's output goes,
//! [`run`] the batched generation itself, [`continuous`] the persistent batch
//! behind
//! [`VoxCpm2SynthOptions::continuous_batching`](crate::voxcpm::VoxCpm2SynthOptions::continuous_batching).

mod batch;
mod continuous;
mod load;
mod queue;
mod render;
mod reply;
mod request;
mod run;
mod types;

pub use request::{Reference, SynthRequest, VoiceSource};
pub use types::{VoxCpm2Engine, VoxCpm2RenderClient, ZERO_SHOT_VOICE_ID};
