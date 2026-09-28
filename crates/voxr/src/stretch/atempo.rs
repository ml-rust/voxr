//! `atempo`-compatible tempo change for interleaved PCM.
//!
//! The method is WSOLA, waveform-similarity overlap-add (W. Verhelst and
//! M. Roelands, "An overlap-add technique based on waveform similarity
//! (WSOLA) for high quality time-scale modification of speech", ICASSP
//! 1993). The implementation follows Pavel Koshevoy's MIT-licensed yae audio
//! tempo filter: a Hann-windowed segment of the smallest power of two not
//! below `rate / 24` frames, a half-segment output hop, and FFT
//! cross-correlation alignment against the previous segment. The output
//! matches the output of ffmpeg's `atempo` filter.
//!
//! # Framing
//!
//! How the stream ends depends on where processing stopped when the input
//! ran out. That depends on the packet sizes the input arrives in and on the
//! output buffer sizes. This module replays the sizes an ffmpeg run shows
//! for a WAV input:
//!
//! - Input packets hold the largest power of two not above `rate / 10`
//!   frames. The last packet holds the rest.
//! - Output buffers hold the packet's frame count over the tempo, rounded.
//! - End-of-stream passes fill the last partly filled buffer first. Each
//!   later pass gets a fresh buffer of three segments, as yae's audio track
//!   allocates. One pass writes at most one segment, so a fresh buffer never
//!   runs out of room.
//!
//! With these sizes, the output equals `ffmpeg -i in.wav -af atempo=<tempo>`
//! for a WAV input in the same sample format.
//!
//! # Known deviation
//!
//! Where one segment is exact digital silence and its neighbour is not, the
//! true correlation is zero at every lag. The lag then comes from FFT
//! rounding noise. That noise differs between this module's transform and
//! ffmpeg's, and the two outputs take different, equally valid alignments
//! from there on. Signals with a noise floor, as recorded or synthesized
//! speech has, do not hit this.
//!
//! # Licence notice
//!
//! The segment engine in [`super::wsola`] follows the yae audio tempo filter
//! (`yae/ffmpeg/yae_audio_tempo_filter.h` and
//! `yae/ffmpeg/yae_audio_fragment.h`, <https://github.com/pkoshevoy/aeyae>).
//!
//! Copyright (c) Pavel Koshevoy
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the
//! "Software"), to deal in the Software without restriction, including
//! without limitation the rights to use, copy, modify, merge, publish,
//! distribute, sublicense, and/or sell copies of the Software, and to permit
//! persons to whom the Software is furnished to do so, subject to the
//! following conditions:
//!
//! The above copyright notice and this permission notice shall be included
//! in all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//! OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//! MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
//! NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//! DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//! OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//! USE OR OTHER DEALINGS IN THE SOFTWARE.

use super::sample::PcmSample;
use super::wsola::{Stop, Wsola};
use crate::error::{Error, Result};

/// Smallest tempo accepted: half speed.
pub const MIN_TEMPO: f64 = 0.5;

/// Largest tempo accepted: double speed.
pub const MAX_TEMPO: f64 = 2.0;

/// Lowest sample rate accepted, in Hz: the segment needs at least two frames.
const MIN_RATE: u32 = 48;

/// Highest sample rate accepted, in Hz.
const MAX_RATE: u32 = 768_000;

/// Changes the tempo of mono `f32` audio without changing its pitch.
///
/// `tempo` above 1 shortens the audio, below 1 lengthens it. The output
/// equals ffmpeg's `atempo=<tempo>` over a 32-bit float WAV at `rate`.
/// Input shorter than half a segment (1024 frames at 48 kHz) yields no
/// output, as in ffmpeg.
pub fn atempo(samples: &[f32], rate: u32, tempo: f64) -> Result<Vec<f32>> {
    atempo_interleaved(samples, 1, rate, tempo)
}

/// Changes the tempo of mono signed 16-bit PCM without changing its pitch.
///
/// The output equals ffmpeg's `atempo=<tempo>` over a 16-bit PCM WAV at
/// `rate`, sample for sample.
pub fn atempo_pcm16(samples: &[i16], rate: u32, tempo: f64) -> Result<Vec<i16>> {
    atempo_interleaved(samples, 1, rate, tempo)
}

/// Changes the tempo of interleaved audio with `channels` channels.
///
/// Every channel takes the same segment positions. Alignment follows, frame
/// by frame, the channel with the largest magnitude.
pub fn atempo_interleaved<S: PcmSample>(
    samples: &[S],
    channels: u16,
    rate: u32,
    tempo: f64,
) -> Result<Vec<S>> {
    validate(samples.len(), channels, rate, tempo)?;
    let ch = usize::from(channels);
    let frames = samples.len() / ch;
    let mut engine = Wsola::new(samples, ch, rate, tempo);
    let packet = packet_frames(rate);
    let mut buffer: Option<OutputBuffer> = None;

    let mut offered = 0usize;
    while offered < frames {
        let chunk = packet.min(frames - offered);
        offered += chunk;
        engine.offer(offered);
        let capacity = (0.5 + chunk as f64 / tempo) as usize;
        while engine.hungry() {
            let buf = buffer.get_or_insert(OutputBuffer::new(capacity));
            let before = engine.written();
            let stop = engine.run(&mut buf.left)?;
            buf.filled += (engine.written() - before) as usize;
            if buf.left == 0 {
                buffer = None;
            }
            if stop == Stop::NeedInput {
                break;
            }
        }
    }

    let drain_capacity = 3 * engine.segment_len();
    let pass_limit = 4 * (frames / engine.segment_len()) + 64;
    for _ in 0..pass_limit {
        let buf = buffer.get_or_insert(OutputBuffer::new(drain_capacity));
        let before = engine.written();
        let done = engine.drain(&mut buf.left)?;
        buf.filled += (engine.written() - before) as usize;
        if buf.filled > 0 {
            buffer = None;
        }
        if done {
            return Ok(engine.into_output());
        }
    }
    Err(Error::DataError {
        reason: format!(
            "atempo end-of-stream pass did not finish after {pass_limit} passes \
             ({frames} frames, {channels} channels, {rate} Hz, tempo {tempo}); \
             report this input as a bug"
        ),
    })
}

/// Frames per input packet: the largest power of two not above `rate / 10`.
fn packet_frames(rate: u32) -> usize {
    let ceiling = ((rate / 10) as usize).max(1);
    let mut frames = 1usize;
    while frames * 2 <= ceiling {
        frames *= 2;
    }
    frames
}

fn validate(len: usize, channels: u16, rate: u32, tempo: f64) -> Result<()> {
    if !(MIN_TEMPO..=MAX_TEMPO).contains(&tempo) {
        return Err(Error::InvalidArgument {
            arg: "tempo",
            reason: format!(
                "tempo {tempo} is outside [{MIN_TEMPO}, {MAX_TEMPO}]; \
                 for a larger change, apply atempo more than once"
            ),
        });
    }
    if !(MIN_RATE..=MAX_RATE).contains(&rate) {
        return Err(Error::InvalidArgument {
            arg: "rate",
            reason: format!("sample rate {rate} Hz is outside [{MIN_RATE}, {MAX_RATE}] Hz"),
        });
    }
    if channels == 0 {
        return Err(Error::InvalidArgument {
            arg: "channels",
            reason: "channel count is 0; pass at least 1".into(),
        });
    }
    if !len.is_multiple_of(usize::from(channels)) {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: format!(
                "{len} samples do not split into whole frames of {channels} channels; \
                 trim the partial frame"
            ),
        });
    }
    Ok(())
}

/// One output buffer of the replayed framing: room left and frames written.
/// A buffer is handed on once full, or during the end-of-stream passes once
/// it holds any frame.
struct OutputBuffer {
    left: usize,
    filled: usize,
}

impl OutputBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            left: capacity,
            filled: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stretch::wsola::segment_frames;

    #[test]
    fn packet_is_largest_power_of_two_under_a_tenth_second() {
        assert_eq!(packet_frames(48_000), 4096);
        assert_eq!(packet_frames(44_100), 4096);
        assert_eq!(packet_frames(24_000), 2048);
        assert_eq!(packet_frames(16_000), 1024);
        assert_eq!(packet_frames(11_025), 1024);
        assert_eq!(packet_frames(96_000), 8192);
    }

    #[test]
    fn rejects_out_of_range_arguments() {
        let x = vec![0.0f32; 4800];
        for bad in [0.49, 2.01, f64::NAN, f64::INFINITY] {
            assert!(atempo(&x, 48_000, bad).is_err(), "tempo {bad}");
        }
        assert!(atempo(&x, 0, 1.0).is_err());
        assert!(atempo_interleaved(&x[..3], 2, 48_000, 1.0).is_err());
        assert!(atempo_interleaved(&x, 0, 48_000, 1.0).is_err());
    }

    #[test]
    fn empty_and_short_inputs_yield_nothing() -> Result<()> {
        assert!(atempo(&[], 48_000, 1.5)?.is_empty());
        // Under half a segment never completes the first segment.
        assert!(atempo(&vec![0.1f32; 1000], 48_000, 1.5)?.is_empty());
        Ok(())
    }

    /// Output length tracks the tempo: longer below 1, shorter above.
    #[test]
    fn length_scales_with_tempo() -> Result<()> {
        let rate = 48_000u32;
        let x: Vec<f32> = (0..rate as usize)
            .map(|n| (n as f32 * 0.0577).sin() * 0.5)
            .collect();
        for tempo in [0.5, 0.8, 1.0, 1.25, 1.5, 2.0] {
            let got = atempo(&x, rate, tempo)?.len() as f64;
            let want = x.len() as f64 / tempo;
            assert!(
                (got - want).abs() < 2.0 * segment_frames(rate) as f64,
                "tempo {tempo}: {got} frames, about {want} expected"
            );
        }
        Ok(())
    }

    #[test]
    fn stereo_keeps_channels_apart() -> Result<()> {
        let x: Vec<i16> = (0..24_000).flat_map(|n| [(n % 200) as i16, 0]).collect();
        let y = atempo_interleaved(&x, 2, 48_000, 1.25)?;
        assert_eq!(y.len() % 2, 0);
        assert!(y.chunks(2).all(|f| f[1] == 0));
        Ok(())
    }
}
