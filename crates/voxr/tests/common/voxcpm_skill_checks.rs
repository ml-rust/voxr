//! The per-recording and decoder checks of `voxcpm_skill_parity`, with
//! their gates, and the agreement report they fill.

use std::path::Path;

use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use super::voxcpm_skill::{ModeFixture, rel_l2, snr_db};

type Rt = CpuRuntime;

/// Encoder output of the same 16 kHz samples, relative L2.
const ENCODE_REL: f64 = 1e-4;
/// Encoder output of voxr's own resampled samples, relative L2.
const PREPROCESS_REL: f64 = 1e-4;
/// voxr's 16 kHz samples against `librosa.load`'s, in dB.
const RESAMPLE_SNR_DB: f64 = 120.0;
/// Decoder output of the reference's own patches, in dB.
const DECODE_SNR_DB: f64 = 100.0;

pub fn patches_of(feat: Option<Tensor<Rt>>) -> usize {
    feat.map_or(0, |f| f.shape()[0])
}

/// Mono samples of `path` at their own rate, read as `librosa.load` reads
/// them: integer PCM over `2^(bits - 1)`, then the channel mean.
pub fn recording(path: &Path) -> (Vec<f32>, u32) {
    let bytes = std::fs::read(path).expect("read the reference wav");
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(voxr::extension_hint);
    let data =
        voxr::decode::decode_audio_like_ffmpeg(&bytes, hint).expect("decode the reference wav");
    let mono = voxr::to_mono(&data.samples, data.channels).expect("downmix");
    (mono, data.sample_rate)
}

/// Agreement of one mode, gathered for the report and the gates.
#[derive(Default)]
pub struct Agreement {
    pub lines: Vec<String>,
    pub failures: Vec<String>,
}

impl Agreement {
    pub fn measure(&mut self, label: String, value: f64, ok: bool) {
        self.lines.push(format!("{label}: {value:.3e}"));
        if !ok {
            self.failures
                .push(format!("{label}: {value:.3e} is outside its gate"));
        }
    }

    pub fn exact(&mut self, label: String, got: usize, want: usize) {
        self.lines
            .push(format!("{label}: {got} (reference {want})"));
        if got != want {
            self.failures
                .push(format!("{label}: {got}, reference {want}"));
        }
    }
}

pub fn host(t: &Tensor<Rt>) -> Vec<f32> {
    t.contiguous().expect("contiguous").to_vec()
}

/// Gate 2: one recording, `side` naming its encode. Returns nothing: the
/// prefill reads the reference's own patches, so this gate stands alone.
pub fn check_recording(
    model: &VoxCpm2Model<Rt>,
    client: &CpuClient,
    fx: &mut ModeFixture,
    name: &str,
    ours_16k: &[f32],
    out: &mut Agreement,
) {
    let device = CpuDevice::new();
    let mode = fx.meta.mode.clone();
    let (Some(want_16k), Some(want_feat)) = (
        fx.floats(&format!("{name}_wav16k"), &device),
        fx.floats(&format!("{name}_feat"), &device),
    ) else {
        return;
    };
    let encode = |samples: &[f32]| {
        let feat = if name == "ref" {
            model.encode_reference(client, samples)
        } else {
            model.encode_prompt(client, samples)
        };
        host(&feat.expect("encode"))
    };
    out.exact(
        format!("{mode} {name} 16 kHz samples"),
        ours_16k.len(),
        want_16k.len(),
    );
    if ours_16k.len() == want_16k.len() {
        let snr = snr_db(ours_16k, &want_16k);
        out.measure(
            format!("{mode} {name} resample SNR dB"),
            snr,
            snr >= RESAMPLE_SNR_DB,
        );
    }
    let same = encode(&want_16k);
    let rel = rel_l2(&same, &want_feat);
    out.measure(
        format!("{mode} {name} encode rel L2 (same samples)"),
        rel,
        rel <= ENCODE_REL,
    );
    let ours = encode(ours_16k);
    out.exact(
        format!("{mode} {name} patches"),
        ours.len(),
        want_feat.len(),
    );
    if ours.len() == want_feat.len() {
        let rel = rel_l2(&ours, &want_feat);
        out.measure(
            format!("{mode} {name} encode rel L2 (voxr samples)"),
            rel,
            rel <= PREPROCESS_REL,
        );
    }
}

/// Gate 4's isolation: the reference's own last take, decoded behind its
/// own context as `_generate_with_prompt_cache` decodes it.
pub fn check_decoder(
    model: &VoxCpm2Model<Rt>,
    client: &CpuClient,
    fx: &mut ModeFixture,
    want_wav: &[f32],
    out: &mut Agreement,
) {
    let device = CpuDevice::new();
    let mode = fx.meta.mode.clone();
    let last = fx.meta.takes.len() - 1;
    let context = fx.meta.takes[last].context_len;
    let (p, d) = (model.config.patch_size, model.config.feat_dim);
    let to_vars = |flat: &[f32]| -> Vec<Var<Rt>> {
        flat.chunks(p * d)
            .map(|c| {
                Var::new(
                    Tensor::<Rt>::from_slice(c, &[1, p, d], &device).expect("patch"),
                    false,
                )
            })
            .collect()
    };
    let mut patches = Vec::new();
    if context > 0 {
        let prompt = fx.floats("prompt_feat", &device).expect("prompt_feat");
        patches = to_vars(&prompt[prompt.len() - context * p * d..]);
    }
    patches.extend(to_vars(
        &fx.floats(&format!("take{last}_patches"), &device)
            .expect("patches"),
    ));
    let decoded = if context > 0 {
        model.decode_patches_from(client, &patches, context)
    } else {
        model.decode_patches(client, &patches)
    };
    let wav = host(&decoded.expect("decode"));
    out.exact(format!("{mode} decoder samples"), wav.len(), want_wav.len());
    if wav.len() == want_wav.len() {
        let snr = snr_db(&wav, want_wav);
        out.measure(format!("{mode} decoder SNR dB"), snr, snr >= DECODE_SNR_DB);
    }
}
