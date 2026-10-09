//! RIFF/WAVE to `f64` samples, read as libsndfile reads them (Python's
//! `soundfile.read(path, dtype='float64')`): an integer sample is divided
//! by `2^(bits - 1)` with no clamp, and a float sample widens exactly.
//!
//! [`super::decode_wav`] instead divides by the positive full scale and
//! clamps, the convention its own writer round-trips. The two differ by
//! one part in `2^(bits - 1)`.

use crate::error::Result;

use super::decode::{SUPPORTED, bad, locate, open_reader, rejection};

/// A decoded WAV file in `f64`: interleaved samples plus their format.
#[derive(Debug, Clone, PartialEq)]
pub struct WavF64 {
    /// Interleaved samples, frame-major.
    pub samples: Vec<f64>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count, at least 1.
    pub channels: u16,
}

impl WavF64 {
    /// Number of frames, i.e. samples per channel.
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels.max(1))
    }

    /// Each channel's samples, de-interleaved.
    pub fn channel_vecs(&self) -> Vec<Vec<f64>> {
        let c = usize::from(self.channels.max(1));
        (0..c)
            .map(|ch| self.samples.iter().skip(ch).step_by(c).copied().collect())
            .collect()
    }

    /// The mean over channels of each frame: the channels summed left to
    /// right, then divided by the count, as NumPy's `x.mean(axis=1)` does.
    pub fn mono(&self) -> Vec<f64> {
        let c = usize::from(self.channels.max(1));
        let n = c as f64;
        self.samples
            .chunks_exact(c)
            .map(|frame| frame.iter().fold(0.0f64, |acc, &v| acc + v) / n)
            .collect()
    }
}

/// Decode a WAV byte buffer to `f64`, as libsndfile does.
///
/// Supports the formats [`super::decode_wav`] supports and returns its
/// errors.
pub fn decode_wav_f64(bytes: &[u8]) -> Result<WavF64> {
    let (format, data) = locate(bytes)?;
    let reader = open_reader(&format, data)?;
    let rejected = rejection(&format);
    let spec = reader.spec();
    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, _) => reader
            .into_samples::<f32>()
            .map(|s| s.map(f64::from))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(rejected)?,
        (hound::SampleFormat::Int, bits @ (16 | 24 | 32)) => {
            let unit = (1i64 << (bits - 1)) as f64;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| f64::from(v) / unit))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(rejected)?
        }
        (hound::SampleFormat::Int, bits) => {
            return Err(bad(format!(
                "integer PCM with {bits} bits per sample is unsupported; supported are \
                 {SUPPORTED}"
            )));
        }
    };
    Ok(WavF64 {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::encode_wav_pcm16_values;

    #[test]
    fn integers_divide_by_a_power_of_two() {
        let bytes =
            encode_wav_pcm16_values(&[i16::MIN, -1, 0, 1, i16::MAX], 48_000).expect("encode");
        let wav = decode_wav_f64(&bytes).expect("decode");
        assert_eq!(
            wav.samples,
            vec![-1.0, -1.0 / 32768.0, 0.0, 1.0 / 32768.0, 32767.0 / 32768.0]
        );
        assert_eq!(
            (wav.sample_rate, wav.channels, wav.frames()),
            (48_000, 1, 5)
        );
    }

    #[test]
    fn mono_averages_each_frame() {
        let wav = WavF64 {
            samples: vec![0.5, 0.25, -1.0, 1.0],
            sample_rate: 8000,
            channels: 2,
        };
        assert_eq!(wav.mono(), vec![0.375, 0.0]);
        assert_eq!(wav.channel_vecs(), vec![vec![0.5, -1.0], vec![0.25, 1.0]]);
    }

    #[test]
    fn garbage_is_refused() {
        assert!(decode_wav_f64(b"not a wav file at all").is_err());
    }
}
