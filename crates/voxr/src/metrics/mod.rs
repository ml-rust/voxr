//! Single-signal measures of a finished sound, each a port of the NumPy
//! and SciPy computation the motion-video skill's sound review runs:
//! sharpness, the share of energy above 8 kHz, the tail against the
//! loudest 50 ms, and the 4x oversampled true peak.

mod level;
mod pairwise;
mod sharpness;
mod welch;
mod window_sum;

pub use level::{hiss_db, tail_db, true_peak_dbtp};
pub use pairwise::{pairwise_mean, pairwise_sum};
pub use sharpness::{bark, sharpness};
pub use welch::{Psd, welch};
