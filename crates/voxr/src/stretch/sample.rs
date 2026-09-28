//! PCM sample formats the tempo stretcher reads and writes.

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i16 {}
}

/// A PCM sample format [`atempo_interleaved`](super::atempo_interleaved)
/// accepts: `f32` for float samples, `i16` for signed 16-bit PCM.
///
/// The format decides two things. The crossfade result rounds to it: `i16`
/// truncates toward zero and saturates at the type's range, `f32` keeps the
/// float. And when channels compete for the alignment signal, each sample's
/// magnitude counts up to [`PcmSample::CEILING`] only.
pub trait PcmSample: Copy + Default + sealed::Sealed {
    /// Largest magnitude a sample counts with when the loudest channel is
    /// picked for alignment: `1.0` for `f32`, `32767.0` for `i16`.
    const CEILING: f32;

    /// The sample as `f32`, unscaled: an `i16` of 1000 reads as `1000.0`.
    fn to_f32(self) -> f32;

    /// The format's value nearest `value` by truncation, saturating.
    fn from_f32(value: f32) -> Self;
}

impl PcmSample for f32 {
    const CEILING: f32 = 1.0;

    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(value: f32) -> Self {
        value
    }
}

impl PcmSample for i16 {
    const CEILING: f32 = i16::MAX as f32;

    fn to_f32(self) -> f32 {
        f32::from(self)
    }

    fn from_f32(value: f32) -> Self {
        // `as` truncates toward zero and saturates at the i16 range.
        value as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i16_truncates_toward_zero_and_saturates() {
        assert_eq!(i16::from_f32(12.9), 12);
        assert_eq!(i16::from_f32(-12.9), -12);
        assert_eq!(i16::from_f32(40_000.0), i16::MAX);
        assert_eq!(i16::from_f32(-40_000.0), i16::MIN);
        assert_eq!((-7i16).to_f32(), -7.0);
    }

    #[test]
    fn f32_passes_through() {
        assert_eq!(f32::from_f32(0.25), 0.25);
        assert_eq!((-0.5f32).to_f32(), -0.5);
    }
}
