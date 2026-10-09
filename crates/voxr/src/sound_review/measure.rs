//! The four numbers the review prints for each sound, measured as the
//! skill's `measure()` measures them and rounded as it rounds them.

use crate::error::Result;
use crate::metrics::{hiss_db, sharpness, tail_db, true_peak_dbtp};
use crate::wav::WavF64;

use super::pyfmt::{np_round, py_round};

/// A sound's measures, unrounded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawMeasures {
    /// Sharpness of the loudest 100 ms, acum.
    pub sharp: f64,
    /// Energy above 8 kHz against the whole, dB.
    pub hiss: f64,
    /// Last 5 ms against the loudest 50 ms, dB.
    pub tail: f64,
    /// 4x oversampled true peak, dBTP.
    pub tp: f64,
}

impl RawMeasures {
    /// Measures `wav`: the mono mean of its channels for sharpness, hiss
    /// and tail, every channel for the true peak.
    pub fn of(wav: &WavF64) -> Result<Self> {
        let m = wav.mono();
        let rate = wav.sample_rate;
        Ok(Self {
            sharp: sharpness(&m, rate)?,
            hiss: hiss_db(&m, rate)?,
            tail: tail_db(&m, rate)?,
            tp: true_peak_dbtp(&wav.channel_vecs())?,
        })
    }

    /// The printed values: sharpness to 2 places by Python's `round`, the
    /// others to 1 place by NumPy's (they are NumPy floats there).
    pub fn rounded(&self) -> Self {
        Self {
            sharp: py_round(self.sharp, 2),
            hiss: np_round(self.hiss, 1),
            tail: np_round(self.tail, 1),
            tp: np_round(self.tp, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tone_measures_in_range() {
        let rate = 48_000u32;
        let samples: Vec<f64> = (0..rate as usize)
            .flat_map(|i| {
                let v = 0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 48_000.0).sin();
                [v, v]
            })
            .collect();
        let wav = WavF64 {
            samples,
            sample_rate: rate,
            channels: 2,
        };
        let r = RawMeasures::of(&wav).expect("measures").rounded();
        assert!(r.sharp > 0.3 && r.sharp < 0.6, "{r:?}");
        assert!(r.hiss < -60.0, "{r:?}");
        assert!(r.tail.abs() < 0.5, "{r:?}");
        assert!((r.tp - -6.0).abs() < 0.1, "{r:?}");
    }
}
