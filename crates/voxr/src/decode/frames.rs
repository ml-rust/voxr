//! Interleaving decoded symphonia buffers into `f32` samples.

use symphonia::core::audio::{AudioBufferRef, Signal};

use crate::wav::decode::scale;

/// How integer PCM becomes float.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IntScale {
    /// Divided by the positive full scale and clamped to `[-1, 1]`, as
    /// [`decode_wav`](crate::decode_wav) does.
    FullScale,
    /// Divided by `2^(bits - 1)`, as ffmpeg does.
    Ffmpeg,
}

/// Append frames `from..to` of one decoded audio buffer to `out`, interleaved
/// frame-major with channel `k` read from plane `order[k]`, converting every
/// symphonia sample format to `f32` in `[-1, 1]`.
pub(super) fn push_frame(
    audio: &AudioBufferRef<'_>,
    order: &[usize],
    int_scale: IntScale,
    from: usize,
    to: usize,
    out: &mut Vec<f32>,
) {
    macro_rules! interleave {
        ($buf:expr, $convert:expr) => {{
            let to = to.min($buf.frames());
            if let [plane] = order {
                out.reserve(to.saturating_sub(from));
                for i in from..to {
                    out.push($convert($buf.chan(*plane)[i]));
                }
            } else {
                out.reserve(to.saturating_sub(from) * order.len());
                for i in from..to {
                    for &c in order {
                        out.push($convert($buf.chan(c)[i]));
                    }
                }
            }
        }};
    }
    let ffmpeg = int_scale == IntScale::Ffmpeg;
    match audio {
        // ffmpeg's integer-to-float conversions: a power-of-two divisor,
        // exact after the integer rounds to `f32`, and no clamp.
        AudioBufferRef::S8(b) if ffmpeg => interleave!(b, |x: i8| f32::from(x) / 128.0),
        AudioBufferRef::S16(b) if ffmpeg => interleave!(b, |x: i16| f32::from(x) / 32_768.0),
        AudioBufferRef::S24(b) if ffmpeg => {
            interleave!(b, |x: symphonia::core::sample::i24| x.inner() as f32
                / 8_388_608.0)
        }
        AudioBufferRef::S32(b) if ffmpeg => interleave!(b, |x: i32| x as f32 / 2_147_483_648.0),
        AudioBufferRef::U16(b) if ffmpeg => {
            interleave!(b, |x: u16| (f32::from(x) - 32_768.0) / 32_768.0)
        }
        AudioBufferRef::U24(b) if ffmpeg => interleave!(b, |x: symphonia::core::sample::u24| {
            (x.inner() as f32 - 8_388_608.0) / 8_388_608.0
        }),
        AudioBufferRef::U32(b) if ffmpeg => {
            interleave!(
                b,
                |x: u32| ((f64::from(x) - 2_147_483_648.0) / 2_147_483_648.0) as f32
            )
        }
        AudioBufferRef::F32(b) => interleave!(b, |x: f32| x),
        AudioBufferRef::F64(b) => interleave!(b, |x: f64| x as f32),
        // Signed formats go through `wav::decode::scale`, the same helper the
        // native WAV parser uses: divide by positive full scale and clamp, so
        // both decoders return identical values for identical bytes. That scale
        // also matches what `wav::encode` writes, keeping the round trip exact.
        AudioBufferRef::S16(b) => interleave!(b, |x: i16| scale(x as f64, i16::MAX as f64)),
        AudioBufferRef::S32(b) => interleave!(b, |x: i32| scale(x as f64, i32::MAX as f64)),
        AudioBufferRef::U8(b) => interleave!(b, |x: u8| (x as f32 - 128.0) / 128.0),
        // Same exact-bounds scale as U32/U24 below: `(x / MAX) * 2 - 1` maps
        // 0 to -1.0 and MAX to +1.0 with no overshoot.
        // The `(x - MAX/2) / (MAX/2)` form overshoots +1.0 by one step at the
        // top rail because `u16::MAX` is odd, so `MAX/2.0` is not its own
        // true midpoint divisor.
        AudioBufferRef::U16(b) => {
            interleave!(b, |x: u16| (x as f32 / u16::MAX as f32) * 2.0 - 1.0)
        }
        AudioBufferRef::U32(b) => interleave!(b, |x: u32| (x as f32 / u32::MAX as f32) * 2.0 - 1.0),
        AudioBufferRef::S8(b) => interleave!(b, |x: i8| scale(x as f64, i8::MAX as f64)),
        AudioBufferRef::S24(b) => interleave!(b, |x: symphonia::core::sample::i24| scale(
            x.inner() as f64,
            8_388_607.0
        )),
        AudioBufferRef::U24(b) => interleave!(b, |x: symphonia::core::sample::u24| (x.inner()
            as f32
            / 16_777_215.0)
            * 2.0
            - 1.0),
    }
}
