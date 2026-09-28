//! Checks the ported VoxCPM2 PREFILL (`encode_reference`, `encode_prompt`
//! and `prefill_batch_capturing`) against the Python model's own output, in
//! every prefix mode: reference, zero-shot, ref+continuation and
//! continuation-only.
//!
//! ```text
//! cargo run --release -p voxr --features voxcpm --example voxcpm_prefill_check -- CKPT_DIR FIXTURE_DIR [FIXTURE_NAME]
//! ```
//!
//! `CKPT_DIR` is the VoxCPM2 checkpoint (`config.json` + `model.safetensors`).
//! `FIXTURE_DIR` holds the fixture written by
//! `crates/voxr/tests/tools/voxcpm/make_prefill_fixture.py` from the
//! reference implementation, and `audiovae.safetensors` (produced by
//! `crates/voxr/tests/tools/voxcpm/convert_audiovae.py`), same layout as
//! `voxcpm_vae_check`. `FIXTURE_NAME` selects the fixture file and defaults
//! to `prefill_fixture.safetensors`; pass `prefill_zeroshot_fixture.safetensors`
//! for the no-reference one, or a fixture written by
//! `crates/voxr/tests/tools/voxcpm/make_continuation_fixture.py` for a
//! continuation mode.
//!
//! This is Unit A of the end-to-end orchestrator: reference-audio encoding
//! and the two-LM prefill, up to (not including) the per-patch CFM sampling
//! loop. It is a real numerical gate, not a smoke test — running and
//! producing plausible numbers proves nothing, reproducing the reference
//! output does. Exits non-zero on mismatch.
//!
//! # One gate, every mode
//!
//! The fixture's `__metadata__` carries `T_ref`. `T_ref > 0` is the
//! reference voice-clone prefix; `T_ref == 0` is the zero-shot prefix, which
//! carries no reference audio at all. The gate reads that field and branches
//! on it: with a reference it encodes `ref_wav` and expects the `103` /
//! filler / `104` layout, without one it expects no delimiter anywhere.
//! Everything after the layout — the prefill body and all six float checks —
//! is identical in every mode.
//!
//! # Continuation tail
//!
//! `__metadata__["T_prompt"] > 0` marks a continuation fixture. The gate
//! then encodes the fixture's `prompt_wav` with `encode_prompt` (left pad),
//! checks it against `prompt_feat`, expects `T_prompt` filler ids after the
//! text, and prefills with the prompt as the row's tail. A fixture without
//! the key predates continuation support and carries no tail, so a missing
//! `T_prompt` reads as 0.
//!
//! Zero-shot is worth its own fixture because it drives code shapes the
//! reference fixture never reaches: `audio_mask` is zero at every position,
//! so `masked_feat` and the masked `feat_embed` are exactly zero, and
//! `fusion_concat_proj` sees a zero right half across the whole sequence.
//! Those are shape-valid under either fixture, so only numbers separate a
//! correct port from a plausible one.
//!
//! # `ref_wav` comes from the fixture, not the original 48 kHz recording
//!
//! The reference fixture's `ref_wav` `[1, 87040]` is the reference's OWN
//! `librosa.load(..., sr=16000)` + right-pad output — i.e. exactly what its
//! `_encode_wav` fed to the AudioVAE. This example feeds that tensor to
//! `encode_reference` directly and never touches the original 48 kHz
//! recording. boostr's resampler is not bit-identical to librosa's
//! `soxr_hq`; decoding the original file here would conflate a resampler
//! difference with a genuine port bug in the VAE / patch fold / prefill
//! path, which is what this gate exists to isolate. The zero-shot fixture
//! has no `ref_wav`, and the gate skips this stage for it.
//!
//! # The layout check is exact-integer on purpose
//!
//! Token ids and the two masks are checked with `==`, not a tolerance, in
//! BOTH modes. A wrong delimiter id, an off-by-one boundary, a masks-swapped
//! bug, or a stray reference prefix left behind in zero-shot is shape-valid
//! and silently produces a different (but still plausible-looking) float
//! sequence; a 2e-3 float tolerance would hide it completely. Integer
//! equality cannot.
use boostr::format::safetensors_loader::SafeTensorsLoader;
use boostr::model::audio::voxcpm::model::config::{
    AUDIO_START_ID, REF_AUDIO_END_ID, REF_AUDIO_FILLER_ID, REF_AUDIO_START_ID,
};
use boostr::model::audio::voxcpm::model::sequence::SequenceLayout;
use boostr::model::audio::voxcpm::model::{PrefillRow, VoxCpm2Model};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::path::{Path, PathBuf};

/// Sizes both KV caches; matches the reference checkpoint's `config.json`
/// `max_length`. Only the fixture's own `S` positions are actually prefilled.
const MAX_LENGTH: usize = 8192;

/// Default fixture file, the reference voice-clone one.
const DEFAULT_FIXTURE: &str = "prefill_fixture.safetensors";

/// Read `__metadata__["T_ref"]` out of an already-open fixture loader.
///
/// `T_ref` is what selects the gate's mode, so a missing or unparsable field
/// is an error rather than a silent default.
fn fixture_t_ref(fx: &SafeTensorsLoader, path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let raw = fx
        .first()
        .and_then(|st| st.metadata().get("T_ref"))
        .ok_or_else(|| format!("{}: no __metadata__.T_ref", path.display()))?;
    Ok(raw.parse::<usize>()?)
}

/// Read `__metadata__["T_prompt"]`, the continuation tail's patch count. A
/// fixture written before continuation support has no such key and no tail,
/// so a missing key is 0; an unparsable one is an error.
fn fixture_t_prompt(fx: &SafeTensorsLoader) -> Result<usize, Box<dyn std::error::Error>> {
    match fx.first().and_then(|st| st.metadata().get("T_prompt")) {
        Some(raw) => Ok(raw.parse::<usize>()?),
        None => Ok(0),
    }
}

/// Load `<name>`, run nothing (caller already has `got`), and report shapes /
/// max abs error / span / relative error / OK-or-MISMATCH in the sibling
/// gates' line format. Returns whether this case passed.
fn check(
    label: &str,
    got: &Tensor<CpuRuntime>,
    want: &Tensor<CpuRuntime>,
) -> Result<bool, Box<dyn std::error::Error>> {
    println!("{label}: got {:?} want {:?}", got.shape(), want.shape());
    assert_eq!(got.shape(), want.shape());
    let g: Vec<f32> = got.contiguous()?.to_vec();
    let w: Vec<f32> = want.contiguous()?.to_vec();
    let max = g
        .iter()
        .zip(&w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let span =
        w.iter().cloned().fold(f32::MIN, f32::max) - w.iter().cloned().fold(f32::MAX, f32::min);
    // Scale-aware tolerance. A fixed absolute bound is meaningless across this
    // gate's tensors: their spans range from ~9.5 (`ref_feat`) to ~4309
    // (`residual_enc_inputs`, a 4096-wide Linear producing values up to +-2200).
    // For a 4096-term f32 dot product, accumulated error grows as roughly
    // sqrt(n) * eps * |value| ~= 64 * 1.19e-7 * 2200 ~= 1.7e-2, so a 2e-3 bound
    // rejects correct output purely for being large. Gate on RELATIVE error
    // instead, with an absolute floor so a near-zero-span tensor cannot pass on
    // a vacuous ratio. The floor is the same 2e-3 the sibling gates use.
    let tol = 2e-3f32.max(span * 1e-5);
    let pass = max <= tol;
    println!(
        "  max abs err {max:.3e}  span {span:.4}  rel {:.2e}  tol {tol:.3e}  {}",
        max / span.max(1e-9),
        if pass { "OK" } else { "MISMATCH" }
    );
    if !pass {
        // Where the error sits: the count over the bound and the sequence
        // positions they fall in (the second-to-last axis).
        let width = got.shape().last().copied().unwrap_or(1).max(1);
        let over: Vec<usize> = g
            .iter()
            .zip(&w)
            .enumerate()
            .filter(|(_, (a, b))| (*a - *b).abs() > tol)
            .map(|(i, _)| i)
            .collect();
        let mut rows: Vec<usize> = over.iter().map(|i| i / width).collect();
        rows.dedup();
        println!(
            "  {} of {} elements over tol, at positions {:?}",
            over.len(),
            g.len(),
            &rows[..rows.len().min(16)]
        );
    }
    Ok(pass)
}

/// Audio rows over the bound that one FSQ level flip explains: at most one
/// per `FSQ_FLIP_ROWS` audio rows, and never a text row.
const FSQ_FLIP_ROWS: usize = 32;

/// [`check`] for a tensor after the FSQ blend, `[1, S, H]`. FSQ rounds each
/// audio row's latent to a level, so f32 drift at a level boundary moves
/// the whole row by one level step. Such a row passes when it is an audio
/// row (`audio_mask` 1) and the flipped rows stay within the
/// [`FSQ_FLIP_ROWS`] budget. Every other row gates as [`check`] does.
fn check_blended(
    label: &str,
    got: &Tensor<CpuRuntime>,
    want: &Tensor<CpuRuntime>,
    audio_mask: &[f32],
) -> Result<bool, Box<dyn std::error::Error>> {
    if check(label, got, want)? {
        return Ok(true);
    }
    let width = got.shape().last().copied().unwrap_or(1).max(1);
    let g: Vec<f32> = got.contiguous()?.to_vec();
    let w: Vec<f32> = want.contiguous()?.to_vec();
    let span =
        w.iter().cloned().fold(f32::MIN, f32::max) - w.iter().cloned().fold(f32::MAX, f32::min);
    let tol = 2e-3f32.max(span * 1e-5);
    let mut flipped: Vec<usize> = Vec::new();
    for (row, (gr, wr)) in g.chunks(width).zip(w.chunks(width)).enumerate() {
        if gr.iter().zip(wr).any(|(a, b)| (a - b).abs() > tol) {
            flipped.push(row);
        }
    }
    let audio_rows = audio_mask.iter().filter(|&&m| m == 1.0).count();
    let budget = (audio_rows / FSQ_FLIP_ROWS).max(1);
    let text_row = flipped.iter().find(|&&r| audio_mask.get(r) != Some(&1.0));
    let pass = text_row.is_none() && flipped.len() <= budget;
    println!(
        "  FSQ level flips: {} audio row(s) {:?}, budget {budget}, text rows off: {:?}  {}",
        flipped.len(),
        flipped,
        text_row,
        if pass { "OK" } else { "MISMATCH" }
    );
    Ok(pass)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ck = PathBuf::from(std::env::args().nth(1).expect("checkpoint dir"));
    let fx_dir = PathBuf::from(std::env::args().nth(2).expect("fixture dir"));
    let fixture_name = std::env::args()
        .nth(3)
        .unwrap_or_else(|| DEFAULT_FIXTURE.to_string());
    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());

    // audiovae.safetensors lives inside FIXTURE_DIR, same layout as
    // voxcpm_vae_check.
    let vae = fx_dir.join("audiovae.safetensors");
    // f32 ground truth, same rationale as the other VoxCPM2 gates: the
    // checkpoint is BF16 and internal ops upcast anyway.
    let model = VoxCpm2Model::<CpuRuntime>::from_checkpoint(
        &ck,
        Some(&vae),
        &device,
        Some(DType::F32),
        None,
    )?;

    let fx_path = fx_dir.join(&fixture_name);
    let mut fx = SafeTensorsLoader::open(&fx_path)?;
    let t_ref_meta = fixture_t_ref(&fx, &fx_path)?;
    let t_prompt = fixture_t_prompt(&fx)?;
    let has_reference = t_ref_meta > 0;
    println!(
        "fixture: {fixture_name}  T_ref={t_ref_meta}  T_prompt={t_prompt}  mode={}",
        match (has_reference, t_prompt > 0) {
            (true, false) => "reference voice clone",
            (false, false) => "zero-shot (no reference)",
            (true, true) => "ref+continuation (Hi-Fi)",
            (false, true) => "continuation-only",
        }
    );

    let mut ok = true;

    // --- 1. Reference encode: ref_wav -> ref_feat ---------------------------
    // Reference mode only. The zero-shot fixture carries no wav and no feat,
    // and `prefill` gets `None` below.
    let ref_feat_got = if has_reference {
        let ref_wav = fx.load_tensor::<CpuRuntime>("ref_wav", &device)?;
        let ref_feat_want = fx.load_tensor::<CpuRuntime>("ref_feat", &device)?;
        let ref_wav_samples: Vec<f32> = ref_wav.contiguous()?.to_vec();
        let got = model.encode_reference(&client, &ref_wav_samples)?;
        ok &= check("encode_reference", &got, &ref_feat_want)?;
        Some(got)
    } else {
        println!("encode_reference: skipped (no reference in this fixture)");
        None
    };

    let t_ref = ref_feat_got.as_ref().map_or(0, |f| f.shape()[0]);
    assert_eq!(
        t_ref, t_ref_meta,
        "encoded T_ref must match the fixture's own T_ref"
    );

    // --- 1b. Prompt encode: prompt_wav -> prompt_feat (left pad) -------------
    // `prompt_wav` is the reference's own left-padded 16 kHz samples, so the
    // pad here is a no-op and only the VAE and the fold are gated.
    let prompt_feat_got = if t_prompt > 0 {
        let prompt_wav = fx.load_tensor::<CpuRuntime>("prompt_wav", &device)?;
        let prompt_feat_want = fx.load_tensor::<CpuRuntime>("prompt_feat", &device)?;
        let prompt_wav_samples: Vec<f32> = prompt_wav.contiguous()?.to_vec();
        let got = model.encode_prompt(&client, &prompt_wav_samples)?;
        ok &= check("encode_prompt", &got, &prompt_feat_want)?;
        assert_eq!(
            got.shape()[0],
            t_prompt,
            "encoded T_prompt must match the fixture's own T_prompt"
        );
        Some(got)
    } else {
        println!("encode_prompt: skipped (no continuation tail in this fixture)");
        None
    };

    // --- 2. Sequence layout, EXACT integer -----------------------------------
    let text_token_tensor = fx.load_tensor::<CpuRuntime>("text_token", &device)?;
    let full_ids: Vec<i64> = text_token_tensor.contiguous()?.to_vec();
    let seq_len_want = full_ids.len();

    // `text_token` is the FULL S-length sequence: the reference prefix
    // ([103] ++ [0; T_ref] ++ [104], reference modes only) ++ text_token_ids
    // ++ [0; T_prompt] (continuation modes only). Slice out the text ids and
    // feed only those to the prefill — it builds the prefix and the tail
    // itself.
    let prefix_len = if has_reference { t_ref + 2 } else { 0 };
    let text_end = seq_len_want
        .checked_sub(t_prompt)
        .expect("S must be at least T_prompt");
    let text_length = text_end
        .checked_sub(prefix_len)
        .expect("S must be at least the prefix length plus T_prompt");
    let text_token_ids: Vec<u32> = full_ids[prefix_len..text_end]
        .iter()
        .map(|&id| id as u32)
        .collect();
    assert!(
        full_ids[text_end..]
            .iter()
            .all(|&id| id == i64::from(REF_AUDIO_FILLER_ID)),
        "the continuation tail must carry filler ids ({REF_AUDIO_FILLER_ID}) only"
    );

    // Verify the slice boundary before trusting it.
    if has_reference {
        // Position t_ref+1 (the last ref-prefix id) must be the 104 delimiter.
        assert_eq!(
            full_ids[t_ref + 1],
            i64::from(REF_AUDIO_END_ID),
            "expected the ref-audio-end delimiter ({REF_AUDIO_END_ID}) at index T_ref+1={}, got {}",
            t_ref + 1,
            full_ids[t_ref + 1]
        );
    } else {
        // No reference: S is the prompt plus any tail, and NEITHER delimiter
        // may appear.
        assert_eq!(
            seq_len_want,
            text_length + t_prompt,
            "no-reference S must equal text_length + T_prompt"
        );
        for (i, &id) in full_ids.iter().enumerate() {
            assert!(
                id != i64::from(REF_AUDIO_START_ID) && id != i64::from(REF_AUDIO_END_ID),
                "zero-shot sequence must carry no reference delimiter, found {id} at index {i}"
            );
        }
    }
    assert_eq!(
        *text_token_ids.last().expect("non-empty text slice"),
        AUDIO_START_ID,
        "text slice must end with AUDIO_START_ID ({AUDIO_START_ID})"
    );

    let layout = SequenceLayout::build(t_ref, &text_token_ids, t_prompt)?;
    println!(
        "layout: t_ref={t_ref} text_length={} t_prompt={} S={} (want S={seq_len_want})",
        layout.text_length,
        layout.prompt_len,
        layout.seq_len()
    );
    assert_eq!(layout.text_length, text_length);

    let mut layout_ok = layout.seq_len() == seq_len_want;
    if !layout_ok {
        println!(
            "  MISMATCH: seq_len got {} want {seq_len_want}",
            layout.seq_len()
        );
    }
    if let Some(i) = (0..seq_len_want).find(|&i| layout.token_ids[i] != full_ids[i]) {
        layout_ok = false;
        println!(
            "  MISMATCH: token_ids first differ at index {i}: got {} want {}",
            layout.token_ids[i], full_ids[i]
        );
    }
    let text_mask_want = fx.load_tensor::<CpuRuntime>("text_mask", &device)?;
    let text_mask_want: Vec<f32> = text_mask_want.contiguous()?.to_vec();
    if let Some(i) = (0..seq_len_want).find(|&i| layout.text_mask[i] != text_mask_want[i]) {
        layout_ok = false;
        println!(
            "  MISMATCH: text_mask first differs at index {i}: got {} want {}",
            layout.text_mask[i], text_mask_want[i]
        );
    }
    let audio_mask_want = fx.load_tensor::<CpuRuntime>("audio_mask", &device)?;
    let audio_mask_want: Vec<f32> = audio_mask_want.contiguous()?.to_vec();
    if let Some(i) = (0..seq_len_want).find(|&i| layout.audio_mask[i] != audio_mask_want[i]) {
        layout_ok = false;
        println!(
            "  MISMATCH: audio_mask first differs at index {i}: got {} want {}",
            layout.audio_mask[i], audio_mask_want[i]
        );
    }
    println!(
        "layout (exact-integer): {}",
        if layout_ok { "OK" } else { "MISMATCH" }
    );
    ok &= layout_ok;

    // --- 3-6. Intermediate anchors, then 7-8. final hidden states -----------
    let row = PrefillRow {
        ref_feat: ref_feat_got.as_ref(),
        text_token_ids: &text_token_ids,
        prompt_feat: prompt_feat_got.as_ref(),
    };
    let state = model.prefill_batch_capturing(&client, &[row], MAX_LENGTH)?;
    let intermediates = state
        .intermediates
        .expect("prefill_capturing always returns Some(intermediates)");

    let feat_embed_want = fx.load_tensor::<CpuRuntime>("feat_embed", &device)?;
    ok &= check(
        "feat_embed",
        intermediates.feat_embed.tensor(),
        &feat_embed_want,
    )?;

    let combined_embed_want = fx.load_tensor::<CpuRuntime>("combined_embed", &device)?;
    ok &= check(
        "combined_embed",
        intermediates.combined_embed.tensor(),
        &combined_embed_want,
    )?;

    let enc_outputs_want = fx.load_tensor::<CpuRuntime>("enc_outputs_post_blend", &device)?;
    ok &= check_blended(
        "enc_outputs_post_blend",
        intermediates.enc_outputs.tensor(),
        &enc_outputs_want,
        &audio_mask_want,
    )?;

    let residual_enc_inputs_want = fx.load_tensor::<CpuRuntime>("residual_enc_inputs", &device)?;
    ok &= check_blended(
        "residual_enc_inputs",
        intermediates.residual_enc_inputs.tensor(),
        &residual_enc_inputs_want,
        &audio_mask_want,
    )?;

    let lm_hidden_want = fx.load_tensor::<CpuRuntime>("lm_hidden", &device)?;
    ok &= check("lm_hidden", state.lm_hidden.tensor(), &lm_hidden_want)?;

    let residual_hidden_want = fx.load_tensor::<CpuRuntime>("residual_hidden", &device)?;
    ok &= check(
        "residual_hidden",
        state.residual_hidden.tensor(),
        &residual_hidden_want,
    )?;

    // --- 8b. first-patch condition --------------------------------------------
    // The reference's `prefix_feat_cond = feat[:, -1]`: the prompt's last
    // patch with a continuation tail, the zero text-pad patch without one.
    let cond_got = state
        .prefix_feat_cond
        .as_ref()
        .expect("every prefill output carries its condition");
    let cond_want = match &prompt_feat_got {
        Some(_) => {
            let prompt_feat_want = fx.load_tensor::<CpuRuntime>("prompt_feat", &device)?;
            prompt_feat_want.narrow(0, t_prompt - 1, 1)?.contiguous()?
        }
        None => Tensor::<CpuRuntime>::zeros(cond_got.shape(), DType::F32, &device)?,
    };
    ok &= check("prefix_feat_cond", cond_got.tensor(), &cond_want)?;

    // --- 9. position ---------------------------------------------------------
    println!(
        "position: {} (want {seq_len_want})",
        state.lanes.position(0)
    );
    ok &= state.lanes.position(0) == seq_len_want;

    println!("\n{}", if ok { "VERIFIED" } else { "FAILED" });
    std::process::exit(if ok { 0 } else { 1 });
}
