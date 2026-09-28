//! Streaming WSOLA engine: segment placement, alignment and crossfade.
//!
//! The engine follows Pavel Koshevoy's MIT-licensed yae audio tempo filter:
//! `AudioTempoFilter` and `AudioFragment` in the aeyae project. The licence
//! notice is in [`super::atempo`] and in `LICENSE-aeyae` beside this file.
//!
//! Output advances by half a segment per step. The input read point
//! advances by `tempo` times that, truncated to whole frames. Each new
//! segment then shifts to the lag where it best matches the segment before
//! it. The two overlap by half a segment and crossfade under a Hann window.
//!
//! The engine runs against two limits the caller controls: how much input
//! it has been offered ([`Wsola::offer`]) and how much output room it has.
//! It stops at whichever runs out first. Its end-of-stream pass
//! ([`Wsola::drain`]) behaves differently depending on where it stopped, and
//! that dependence shows in the output. [`super::atempo`] therefore replays
//! the input packet and output buffer sizes of an ffmpeg run.

use numr::dtype::Complex128;

use super::correlate::Correlator;
use super::sample::PcmSample;
use crate::error::Result;

/// One analysis segment.
struct Segment<S> {
    /// First input frame the segment reads; negative before the signal.
    input_at: i64,
    /// First output frame the segment covers.
    output_at: i64,
    /// Frames of the segment that hold input; the rest lies past the input
    /// available when it was read.
    len: usize,
    /// Interleaved samples, one segment length of frames.
    samples: Vec<S>,
    /// Spectrum of the segment's alignment signal.
    spectrum: Vec<Complex128>,
}

impl<S: PcmSample> Segment<S> {
    fn new(input_at: i64, output_at: i64, frames: usize, channels: usize) -> Self {
        Self {
            input_at,
            output_at,
            len: 0,
            samples: vec![S::default(); frames * channels],
            spectrum: Vec::new(),
        }
    }
}

/// What the engine does next when it resumes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Next {
    /// Read the current segment at its nominal position, then align it.
    Read,
    /// Re-read the current segment at its aligned position.
    Reread,
    /// Crossfade the previous segment into the current one.
    Crossfade,
}

/// Why [`Wsola::run`] returned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Stop {
    /// The next segment reaches past the input offered so far.
    NeedInput,
    /// The output room ran out mid-crossfade.
    OutputFull,
}

/// WSOLA state over one interleaved input buffer.
pub(super) struct Wsola<'a, S> {
    input: &'a [S],
    channels: usize,
    /// Frames the caller has offered.
    offered: usize,
    /// Frames the engine has taken in; never above `offered`.
    taken: usize,
    /// Segment length in frames, a power of two.
    seg: usize,
    half: i64,
    tempo: f64,
    /// Nominal input advance per segment.
    stride: i64,
    /// Hann window over one segment.
    taper: Vec<f32>,
    prev: Segment<S>,
    cur: Segment<S>,
    /// Segments placed before `cur`.
    placed: u64,
    next: Next,
    /// Output frames written.
    written: i64,
    out: Vec<S>,
    correlator: Correlator,
    /// Alignment signal of the segment being read, reused across reads.
    mono: Vec<f32>,
}

/// Segment length for `rate`: the smallest power of two not below
/// `rate / 24`, about 42 ms of audio.
pub(super) fn segment_frames(rate: u32) -> usize {
    let floor = (rate / 24) as usize;
    let mut frames = 1usize;
    while frames < floor {
        frames *= 2;
    }
    frames
}

impl<'a, S: PcmSample> Wsola<'a, S> {
    /// An engine over `input`, `channels` interleaved, at `rate` Hz.
    pub(super) fn new(input: &'a [S], channels: usize, rate: u32, tempo: f64) -> Self {
        let seg = segment_frames(rate);
        let half = (seg / 2) as i64;
        let span = (seg - 1) as f64;
        let taper = (0..seg)
            .map(|i| (0.5 * (1.0 - (std::f64::consts::TAU * i as f64 / span).cos())) as f32)
            .collect();
        Self {
            input,
            channels,
            offered: 0,
            taken: 0,
            seg,
            half,
            tempo,
            stride: (tempo * half as f64) as i64,
            taper,
            // The first segment starts half a segment early, so the first
            // output half needs no fade-in partner.
            prev: Segment::new(0, 0, seg, channels),
            cur: Segment::new(-half, -half, seg, channels),
            placed: 0,
            next: Next::Read,
            written: 0,
            out: Vec::new(),
            correlator: Correlator::new(seg),
            mono: vec![0.0; seg],
        }
    }

    /// Segment length in frames.
    pub(super) fn segment_len(&self) -> usize {
        self.seg
    }

    /// Output frames written so far.
    pub(super) fn written(&self) -> i64 {
        self.written
    }

    /// Makes the first `frames` input frames available.
    pub(super) fn offer(&mut self, frames: usize) {
        self.offered = frames.min(self.input.len() / self.channels);
    }

    /// True while offered input remains untaken.
    pub(super) fn hungry(&self) -> bool {
        self.taken < self.offered
    }

    /// The interleaved output.
    pub(super) fn into_output(self) -> Vec<S> {
        self.out
    }

    /// Reads `cur` from the taken input and, when `with_spectrum`, refreshes
    /// its alignment spectrum.
    fn read(&mut self, with_spectrum: bool) -> Result<()> {
        let seg = self.seg as i64;
        let short = (self.cur.input_at + seg - self.taken as i64).max(0);
        self.cur.len = (seg - short).max(0) as usize;
        let ch = self.channels;
        for frame in 0..self.seg {
            let at = self.cur.input_at + frame as i64;
            let dst = &mut self.cur.samples[frame * ch..(frame + 1) * ch];
            if frame < self.cur.len && at >= 0 {
                let src = at as usize * ch;
                dst.copy_from_slice(&self.input[src..src + ch]);
            } else {
                dst.fill(S::default());
            }
        }
        if !with_spectrum {
            return Ok(());
        }
        for (frame, value) in self.mono.iter_mut().enumerate() {
            *value = if frame < self.cur.len {
                loudest(&self.cur.samples[frame * ch..(frame + 1) * ch])
            } else {
                0.0
            };
        }
        self.cur.spectrum = self.correlator.spectrum(&self.mono)?;
        Ok(())
    }

    /// Shifts `cur` to the lag that best matches `prev` and returns the shift
    /// in frames, positive toward earlier input. A nonzero shift leaves `cur`
    /// unread.
    fn align(&mut self) -> Result<i64> {
        let seg = self.seg as i64;
        let half = self.half;
        // How far the input read point runs ahead of where the tempo puts
        // it, measured at the centre of `prev`. Both timelines start at frame
        // 0, the centre of the first segment. The search window and the
        // weighting lean against the lead.
        let lead = ((self.prev.output_at + half) as f64 * self.tempo
            - (self.prev.input_at + half) as f64) as i64;
        let lags = self
            .correlator
            .correlate(&self.prev.spectrum, &self.cur.spectrum)?;
        let lo = (-lead).clamp(0, seg);
        let hi = (seg - lead).min(seg - seg / 16).max(0);
        let mut shift = -lead;
        let mut top = f32::MIN;
        for lag in lo..hi {
            let weight = (lead + lag) as f32 * (lag - lo) as f32 * (hi - lag) as f32;
            let score = lags[lag as usize] * weight;
            if score > top {
                top = score;
                shift = lag - half;
            }
        }
        if shift != 0 {
            self.cur.input_at -= shift;
            self.cur.len = 0;
        }
        Ok(shift)
    }

    /// Makes `cur` the previous segment and places a new one after it.
    fn step(&mut self) {
        self.placed += 1;
        std::mem::swap(&mut self.prev, &mut self.cur);
        self.cur.input_at = self.prev.input_at + self.stride;
        self.cur.output_at = self.prev.output_at + self.half;
        self.cur.len = 0;
    }

    /// Writes the overlap of `prev` and `cur` from the current output
    /// position, at most `room` frames. True once the overlap is complete.
    fn crossfade(&mut self, room: &mut usize) -> bool {
        let start = self.written.max(self.cur.output_at);
        let stop = (self.prev.output_at + self.prev.len as i64)
            .min(self.cur.output_at + self.cur.len as i64);
        let ch = self.channels;
        let mut k = 0i64;
        while self.written < stop && *room > 0 {
            let ia = (start + k - self.prev.output_at) as usize;
            let ib = (start + k - self.cur.output_at) as usize;
            let (wa, wb) = (self.taper[ia], self.taper[ib]);
            let a = &self.prev.samples[ia * ch..(ia + 1) * ch];
            let b = &self.cur.samples[ib * ch..(ib + 1) * ch];
            if self.cur.input_at + k < 0 {
                self.out.extend_from_slice(a);
            } else {
                self.out.extend(
                    a.iter()
                        .zip(b)
                        .map(|(&x, &y)| S::from_f32(x.to_f32() * wa + y.to_f32() * wb)),
                );
            }
            self.written += 1;
            *room -= 1;
            k += 1;
        }
        self.written == stop
    }

    /// Advances until input or output room runs out.
    pub(super) fn run(&mut self, room: &mut usize) -> Result<Stop> {
        loop {
            if self.next != Next::Crossfade {
                let need = self.cur.input_at + self.seg as i64;
                let reach = (need.max(0) as usize).min(self.offered);
                self.taken = self.taken.max(reach);
                if (self.taken as i64) < need {
                    return Ok(Stop::NeedInput);
                }
                self.read(true)?;
                if self.next == Next::Read {
                    if self.placed == 0 {
                        self.step();
                        continue;
                    }
                    self.next = if self.align()? != 0 {
                        Next::Reread
                    } else {
                        Next::Crossfade
                    };
                    continue;
                }
                self.next = Next::Crossfade;
            }
            if !self.crossfade(room) {
                return Ok(Stop::OutputFull);
            }
            self.step();
            self.next = Next::Read;
        }
    }

    /// One end-of-stream pass over all offered input. True once the last
    /// frame is written. False when it needs more room or placed another
    /// segment, and then the caller calls again.
    ///
    /// A pass reads `cur` from the input that exists and aligns it once. A
    /// shifted segment is read again, but its spectrum stays that of the
    /// first read. The pass then writes the crossfade with `prev`. When `cur`
    /// ends before the last input frame, the pass places the next segment by
    /// the steady-state rule. Segments keep coming until one covers the last
    /// input frame. That segment writes its remaining frames as they are.
    pub(super) fn drain(&mut self, room: &mut usize) -> Result<bool> {
        self.taken = self.offered;
        let total = self.taken as i64;
        if self.placed == 0 {
            return Ok(true);
        }
        let len = self.cur.len as i64;
        if self.cur.input_at + len == total && self.written == self.cur.output_at + len {
            return Ok(true);
        }
        if self.cur.input_at + len < total {
            self.read(true)?;
            if self.align()? != 0 {
                // The spectrum stays that of the pre-shift read.
                self.read(false)?;
            }
        }
        let len = self.cur.len as i64;
        let head_end = self.cur.output_at + len.min(self.half);
        if self.written < head_end && !self.crossfade(room) {
            return Ok(false);
        }
        if self.cur.input_at + len < total {
            self.step();
            return Ok(false);
        }
        let from = self.written.max(head_end);
        let stop = self.cur.output_at + len;
        let take = ((stop - from).max(0) as usize).min(*room);
        let ch = self.channels;
        let first = (from - self.cur.output_at) as usize;
        self.out
            .extend_from_slice(&self.cur.samples[first * ch..(first + take) * ch]);
        self.written += take as i64;
        *room -= take;
        Ok(self.written == stop)
    }
}

/// The alignment value of one interleaved frame: the sample of the loudest
/// channel, loudness capped at [`PcmSample::CEILING`], first channel on ties.
fn loudest<S: PcmSample>(frame: &[S]) -> f32 {
    let mut pick = frame.first().map_or(0.0, |s| s.to_f32());
    let mut level = pick.abs().min(S::CEILING);
    for s in frame.iter().skip(1) {
        let v = s.to_f32();
        let l = v.abs().min(S::CEILING);
        if level < l {
            level = l;
            pick = v;
        }
    }
    pick
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_is_power_of_two_covering_rate_over_24() {
        assert_eq!(segment_frames(48_000), 2048);
        assert_eq!(segment_frames(44_100), 2048);
        assert_eq!(segment_frames(24_000), 1024);
        assert_eq!(segment_frames(16_000), 1024);
        assert_eq!(segment_frames(8_000), 512);
        // 49152 / 24 is exactly 2048, which already is a power of two.
        assert_eq!(segment_frames(49_152), 2048);
    }

    #[test]
    fn loudest_picks_capped_magnitude_first_on_ties() {
        assert_eq!(loudest(&[0.2f32, -0.7, 0.5]), -0.7);
        // Both exceed the ceiling, so both count as 1.0 and the first wins.
        assert_eq!(loudest(&[1.5f32, -3.0]), 1.5);
        assert_eq!(loudest(&[-9i16, 9]), -9.0);
    }

    #[test]
    fn taper_is_symmetric_hann() {
        let engine = Wsola::<f32>::new(&[], 1, 48_000, 1.0);
        let n = engine.taper.len();
        assert_eq!(engine.taper[0], 0.0);
        assert_eq!(engine.taper[n - 1], engine.taper[0]);
        assert!((engine.taper[n / 2] - 1.0).abs() < 1e-5);
    }
}
