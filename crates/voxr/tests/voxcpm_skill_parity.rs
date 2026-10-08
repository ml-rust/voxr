//! Parity between voxr's VoxCPM2 path and Python `voxcpm` as the skill's
//! runner, `voxcpm_speak.py`, calls it: one fixture per runner branch
//! (design, zero-shot, clone, Hi-Fi clone), written by
//! `tests/tools/voxcpm/make_skill_fixture.py` from `VoxCPM.generate` itself.
//!
//! Each mode is checked at four points, in pipeline order:
//!
//! 1. Text: the ids [`request_ids`] gives the runner's text, and the prefill
//!    layout around them, equal the reference's exactly. The patch budget
//!    equals the reference's `max_len`.
//! 2. Recordings: the reference wav, decoded and resampled to 16 kHz the
//!    way the engine does it, against the samples `librosa.load` gives. The
//!    VAE encode of the reference's own 16 kHz samples against its latent
//!    patches isolates the encoder. The encode of voxr's samples measures
//!    the whole preprocessing path.
//! 3. Generation: the prefill from the reference's own patches, stepped with
//!    the reference's own noise draws (torch's generator cannot be matched,
//!    so its draws are injected). The stop step and the badcase decision
//!    must equal the reference's, and the first patches must agree closely.
//!    Later patches are reported, not gated: one FSQ level flip moves a
//!    patch, and the autoregression carries the move forward.
//! 4. Decode: the row's waveform against the reference's, sample count
//!    exact, its agreement reported. Decoding the reference's own patches
//!    isolates the decoder, and that agreement is gated.
//!
//! The model runs on the CPU at F32, as the fixture's reference does, under
//! every feature set. Set `VOXCPM2_CHECKPOINT` (the `openbmb/VoxCPM2`
//! snapshot), `VOXCPM2_SKILL_FIXTURE_DIR` (the tool's output) and, for the
//! recordings, `VOXCPM2_REF_WAV` (the recording the fixture cloned). Each
//! test skips with a notice when what it needs is absent.

#![cfg(feature = "voxcpm")]

mod common;

use std::path::Path;

use boostr::model::audio::voxcpm::TokenizerSource;
use boostr::model::audio::voxcpm::model::{
    AUDIO_START_ID, GenerateOptions, GenerateState, PrefillRow, SequenceLayout, StepOutcome,
    VoxCpm2Model,
};
use common::voxcpm::ref_wav_fixture;
use common::voxcpm_skill::{
    MODES, ModeFixture, TokenCase, checkpoint, fixture_dir, max_abs, rel_l2, snr_db,
};
use common::voxcpm_skill_checks::{
    Agreement, check_decoder, check_recording, host, patches_of, recording,
};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use voxr::voxcpm::{VoxCpm2Tokenizer, load_tokenizer, normalize_whitespace, request_ids, tokenize};

/// The engine's patch cap, upstream's `max_len` default.
const MAX_LEN_CAP: usize = 4096;

/// Patch 0, relative L2: the prefill and the first step, before any FSQ
/// level can flip between the two sides.
const FIRST_PATCH_REL: f64 = 1e-4;
/// Each of the first [`EARLY_PATCHES`] patches, relative L2. One FSQ level
/// flip moves a patch by more: past these, a flip compounds and the
/// patches are reported, not gated.
const EARLY_PATCH_REL: f64 = 1e-3;
const EARLY_PATCHES: usize = 3;

type Rt = CpuRuntime;

fn skip(what: &str) {
    eprintln!(
        "skipping: {what} unavailable; set VOXCPM2_CHECKPOINT, VOXCPM2_SKILL_FIXTURE_DIR and \
         VOXCPM2_REF_WAV"
    );
}

fn tokenizer(ckpt: &Path) -> VoxCpm2Tokenizer {
    load_tokenizer(&TokenizerSource::File(ckpt.join("tokenizer.json"))).expect("tokenizer")
}

#[test]
fn tokens_match_the_reference_text_tokenizer() {
    let (Some(ckpt), Some(dir)) = (checkpoint(), fixture_dir()) else {
        return skip("the checkpoint or the skill fixture");
    };
    let tok = tokenizer(&ckpt);
    let cases: Vec<TokenCase> = common::load_json(&dir.join("skill_tokens.json"));
    let mut diffs = Vec::new();
    for case in &cases {
        if normalize_whitespace(&case.text) != case.collapsed {
            diffs.push(format!(
                "{:?}: collapses to {:?}",
                case.text,
                normalize_whitespace(&case.text)
            ));
        }
        let got = tokenize(&tok, &case.text);
        if got != case.ids {
            diffs.push(format!(
                "{:?}: ids {got:?}, reference {:?}",
                case.text, case.ids
            ));
        }
    }
    assert!(diffs.is_empty(), "{}", diffs.join("\n"));
}

#[test]
fn requests_lay_out_as_the_reference() {
    let (Some(ckpt), Some(dir)) = (checkpoint(), fixture_dir()) else {
        return skip("the checkpoint or the skill fixture");
    };
    let tok = tokenizer(&ckpt);
    let device = CpuDevice::new();
    let mut diffs = Vec::new();
    for mode in MODES {
        let mut fx = ModeFixture::open(&dir, mode);
        let (mut ids, target_len) =
            request_ids(&tok, &fx.meta.generate_text, fx.meta.prompt_text.as_deref());
        if ids != fx.meta.full_ids || target_len != fx.meta.target_ids.len() {
            diffs.push(format!(
                "{mode}: ids {ids:?} (target {target_len}), reference {:?} (target {})",
                fx.meta.full_ids,
                fx.meta.target_ids.len()
            ));
        }
        ids.push(AUDIO_START_ID);
        let t_ref = patches_of(fx.tensor("ref_feat", &device));
        let t_prompt = patches_of(fx.tensor("prompt_feat", &device));
        let layout = SequenceLayout::build(t_ref, &ids, t_prompt).expect("layout");
        let mask = |m: &[f32]| m.iter().map(|&v| v as i64).collect::<Vec<_>>();
        let want = |fx: &mut ModeFixture, name| fx.ints(name, &device).expect(name);
        for (name, got) in [
            ("text_token", layout.token_ids.clone()),
            ("text_mask", mask(&layout.text_mask)),
            ("audio_mask", mask(&layout.audio_mask)),
        ] {
            if got != want(&mut fx, name) {
                diffs.push(format!("{mode}: {name} differs from the reference prefill"));
            }
        }
        let ratio = fx.meta.retry_badcase_ratio_threshold;
        let budget = ((target_len as f64 * ratio + 10.0) as usize).min(MAX_LEN_CAP);
        for (k, take) in fx.meta.takes.iter().enumerate() {
            if budget != take.max_len || take.min_len != 2 {
                diffs.push(format!(
                    "{mode} take {k}: budget {budget}, min_len 2; reference {} and {}",
                    take.max_len, take.min_len
                ));
            }
        }
    }
    assert!(diffs.is_empty(), "{}", diffs.join("\n"));
}

/// Gates 3 and 4 for one mode.
fn check_generation(
    model: &VoxCpm2Model<Rt>,
    client: &CpuClient,
    tok: &VoxCpm2Tokenizer,
    fx: &mut ModeFixture,
    out: &mut Agreement,
) {
    let device = CpuDevice::new();
    let mode = fx.meta.mode.clone();
    let (mut ids, target_len) =
        request_ids(tok, &fx.meta.generate_text, fx.meta.prompt_text.as_deref());
    ids.push(AUDIO_START_ID);
    let ref_feat = fx.tensor("ref_feat", &device);
    let prompt_feat = fx.tensor("prompt_feat", &device);
    let budget = ((target_len as f64 * fx.meta.retry_badcase_ratio_threshold + 10.0) as usize)
        .min(MAX_LEN_CAP);
    let seq_len = SequenceLayout::build(
        patches_of(ref_feat.clone()),
        &ids,
        patches_of(prompt_feat.clone()),
    )
    .expect("layout")
    .seq_len();
    let mut options = GenerateOptions::new(budget, 0);
    options.cfm.n_timesteps = fx.meta.inference_timesteps;
    options.cfm.cfg_value = fx.meta.cfg_value;
    options.min_len = 2;
    let generator = model.patch_generator();

    let mut kept = None;
    let mut takes = 0;
    while takes < fx.meta.retry_badcase_max_times {
        let Some(z) = fx.floats(&format!("take{takes}_z"), &device) else {
            out.failures.push(format!(
                "{mode}: voxr takes a take {takes} the reference never drew"
            ));
            break;
        };
        let patch = model.config.patch_size * model.config.feat_dim;
        let row = PrefillRow {
            ref_feat: ref_feat.as_ref(),
            text_token_ids: &ids,
            prompt_feat: prompt_feat.as_ref(),
        };
        let prefill = model
            .prefill_batch(client, &[row], seq_len + budget + 1)
            .expect("prefill");
        let mut state = GenerateState::start(prefill, model.config).expect("start");
        let mut drawn = 0;
        while !state.all_finished() {
            let Some(draw) = z.get(drawn * patch..(drawn + 1) * patch) else {
                out.failures.push(format!(
                    "{mode} take {takes}: voxr runs past the reference's stop"
                ));
                break;
            };
            let draw = Tensor::<Rt>::from_slice(
                draw,
                &[1, model.config.feat_dim, model.config.patch_size],
                &device,
            )
            .expect("noise");
            let draw = Var::new(
                draw.transpose(1, 2)
                    .expect("transpose")
                    .contiguous()
                    .expect("contiguous"),
                false,
            );
            drawn += 1;
            if generator
                .step_with_noise(client, &mut state, &draw, &options)
                .expect("step")
                == StepOutcome::Stopped
            {
                break;
            }
        }
        let made = state.patch_len[0];
        out.exact(
            format!("{mode} take {takes} patches (stop step)"),
            made,
            fx.meta.takes.get(takes).map_or(0, |t| t.patches),
        );
        if let (Some(want), Ok(rows)) = (
            fx.floats(&format!("take{takes}_patches"), &device),
            state.row_patches(0),
        ) {
            let got: Vec<f32> = rows.iter().flat_map(|p| host(p.tensor())).collect();
            let n = got.len().min(want.len());
            let per_patch: Vec<f64> = got[..n]
                .chunks(patch)
                .zip(want[..n].chunks(patch))
                .map(|(g, w)| rel_l2(g, w))
                .collect();
            for (k, &rel) in per_patch.iter().take(EARLY_PATCHES).enumerate() {
                let gate = if k == 0 {
                    FIRST_PATCH_REL
                } else {
                    EARLY_PATCH_REL
                };
                out.measure(
                    format!("{mode} take {takes} patch {k} rel L2"),
                    rel,
                    rel <= gate,
                );
            }
            let first_flip = per_patch.iter().position(|&rel| rel > EARLY_PATCH_REL);
            out.lines.push(format!(
                "{mode} take {takes} patches rel L2 {:.3e}, first patch past {EARLY_PATCH_REL:.0e}: {first_flip:?}",
                rel_l2(&got[..n], &want[..n])
            ));
            let profile: Vec<String> = per_patch.iter().map(|r| format!("{r:.1e}")).collect();
            out.lines.push(format!(
                "{mode} take {takes} per-patch rel L2: {}",
                profile.join(" ")
            ));
        }
        takes += 1;
        let bad = made as f64 >= target_len as f64 * fx.meta.retry_badcase_ratio_threshold;
        kept = Some(state);
        if !bad {
            break;
        }
    }
    out.exact(format!("{mode} takes"), takes, fx.meta.takes.len());
    let Some(state) = kept else { return };
    let want_wav = fx.floats("wav", &device).expect("wav");
    let wav = host(&model.decode_row(client, &state, 0).expect("decode"));
    out.exact(format!("{mode} wav samples"), wav.len(), want_wav.len());
    if wav.len() == want_wav.len() {
        out.lines.push(format!(
            "{mode} wav SNR dB {:.1}, max abs {:.3e}",
            snr_db(&wav, &want_wav),
            max_abs(&wav, &want_wav)
        ));
    }
    check_decoder(model, client, fx, &want_wav, out);
}

#[test]
fn seeded_generation_matches_the_reference() {
    let (Some(ckpt), Some(dir), Some(ref_wav)) = (checkpoint(), fixture_dir(), ref_wav_fixture())
    else {
        return skip("the checkpoint, the skill fixture or the reference wav");
    };
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let model = VoxCpm2Model::<Rt>::from_checkpoint(
        &ckpt,
        Some(&ckpt.join("audiovae.pth")),
        &device,
        Some(DType::F32),
        None,
    )
    .expect("load the checkpoint at F32");
    let tok = tokenizer(&ckpt);
    let (samples, rate) = recording(&ref_wav);
    let mut out = Agreement::default();
    for mode in MODES {
        let mut fx = ModeFixture::open(&dir, mode);
        if fx.meta.reference_wav.is_some() {
            let ours_16k =
                voxr::resample::resample_like_librosa(&samples, rate, fx.meta.encode_sample_rate)
                    .expect("resample");
            check_recording(&model, &client, &mut fx, "ref", &ours_16k, &mut out);
            check_recording(&model, &client, &mut fx, "prompt", &ours_16k, &mut out);
        }
        check_generation(&model, &client, &tok, &mut fx, &mut out);
    }
    eprintln!("{}", out.lines.join("\n"));
    assert!(out.failures.is_empty(), "{}", out.failures.join("\n"));
}
