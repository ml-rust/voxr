//! Recognizability: where each sound's own `says` text lands when an
//! audio-text model scores the sound against every text. Each text's
//! scores are standardized over the sounds first, so a text the model
//! favours for every sound gains no edge.

use crate::error::{Error, Result};
use crate::resample::resample_poly;
use crate::wav::WavF64;

/// The rate the model hears at, in Hz.
pub const MODEL_RATE: u32 = 48_000;

/// Added to each text's standard deviation before dividing.
const STD_EPS: f32 = 1e-9;

/// Scores every clip against every text: what a CLAP model gives.
pub trait AudioTextScorer {
    /// `clips`: mono at [`MODEL_RATE`]. Returns the scores row-major,
    /// `[clips × texts]`.
    fn scores(&mut self, clips: &[Vec<f32>], texts: &[String]) -> Result<Vec<f32>>;
}

/// A sound as the model hears it: the `float32` mean of its channels,
/// resampled to 48 kHz as `scipy.signal.resample_poly` does, and padded
/// with zeros to at least one second.
pub fn model_clip(wav: &WavF64) -> Result<Vec<f32>> {
    let c = usize::from(wav.channels.max(1));
    let n = c as f32;
    let mono: Vec<f32> = wav
        .samples
        .chunks_exact(c)
        .map(|frame| frame.iter().fold(0.0f32, |acc, &v| acc + v as f32) / n)
        .collect();
    let mut clip = resample_poly(&mono, MODEL_RATE as usize, wav.sample_rate as usize)?;
    if clip.len() < MODEL_RATE as usize {
        clip.resize(MODEL_RATE as usize, 0.0);
    }
    Ok(clip)
}

/// Where each clip's own text lands, and the text heard first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rank {
    /// 1-based place of the clip's own text.
    pub rank: usize,
    /// The text that placed first.
    pub heard: String,
}

/// Ranks `scores` (`[clips × texts]`, row-major). `own[i]` is the index in
/// `texts` of clip `i`'s own text.
///
/// Each column is standardized over the clips in `float32`, summing down
/// the column as NumPy reduces axis 0. Each row then sorts by descending
/// score; equal scores keep text order.
///
/// Returns [`Error::InvalidArgument`] when the sizes disagree.
pub fn rank(scores: &[f32], own: &[usize], texts: &[String]) -> Result<Vec<Rank>> {
    let (rows, cols) = (own.len(), texts.len());
    if scores.len() != rows * cols || own.iter().any(|&o| o >= cols) {
        return Err(Error::InvalidArgument {
            arg: "scores",
            reason: format!("{} scores for {rows} clips and {cols} texts", scores.len()),
        });
    }
    let n = rows as f32;
    let mut z = scores.to_vec();
    for j in 0..cols {
        let mean = (0..rows).fold(0.0f32, |acc, i| acc + scores[i * cols + j]) / n;
        let var = (0..rows).fold(0.0f32, |acc, i| {
            let d = scores[i * cols + j] - mean;
            acc + d * d
        }) / n;
        let std = var.sqrt() + STD_EPS;
        for i in 0..rows {
            z[i * cols + j] = (scores[i * cols + j] - mean) / std;
        }
    }
    let mut out = Vec::with_capacity(rows);
    for (i, &own_text) in own.iter().enumerate() {
        let row = &z[i * cols..(i + 1) * cols];
        let mut order: Vec<usize> = (0..cols).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        let place = order.iter().position(|&t| t == own_text).unwrap_or(cols);
        out.push(Rank {
            rank: place + 1,
            heard: order.first().map(|&t| texts[t].clone()).unwrap_or_default(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("t{i}")).collect()
    }

    #[test]
    fn a_text_liked_for_everything_gains_no_edge() {
        // Text 1 scores high for both clips; standardized, each clip's own
        // text wins.
        let scores = [0.5f32, 0.9, 0.1, 0.89];
        let r = rank(&scores, &[0, 1], &texts(2)).expect("rank");
        assert_eq!(
            r[0],
            Rank {
                rank: 1,
                heard: "t0".into()
            }
        );
        assert_eq!(
            r[1],
            Rank {
                rank: 1,
                heard: "t1".into()
            }
        );
    }

    #[test]
    fn a_misheard_clip_names_the_winner() {
        let scores = [0.9f32, 0.1, 0.0, 0.2, 0.8, 0.0, 0.1, 0.9, 0.05];
        let r = rank(&scores, &[0, 2, 2], &texts(3)).expect("rank");
        assert_eq!(
            r[1],
            Rank {
                rank: 3,
                heard: "t1".into()
            }
        );
    }

    #[test]
    fn mismatched_sizes_are_refused() {
        assert!(rank(&[0.0; 3], &[0, 1], &texts(2)).is_err());
        assert!(rank(&[0.0; 4], &[0, 2], &texts(2)).is_err());
    }

    #[test]
    fn a_short_clip_pads_to_one_second() {
        let wav = WavF64 {
            samples: vec![0.5, 0.25, -0.5, -0.25],
            sample_rate: MODEL_RATE,
            channels: 2,
        };
        let clip = model_clip(&wav).expect("clip");
        assert_eq!(clip.len(), MODEL_RATE as usize);
        assert_eq!(&clip[..3], &[0.375, -0.375, 0.0]);
    }
}
