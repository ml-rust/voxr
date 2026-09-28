#!/usr/bin/env python3
"""Reference ground truth for VoxCPM2's continuation modes: the prefill, the
first per-patch steps, and the decode context trim.

Two modes, picked by whether --ref-wav is given:
    ref+continuation ("Hi-Fi"): `_generate`'s combined branch
        (`reference_wav_path and prompt_wav_path`, voxcpm2.py:484-529).
    continuation-only: `_generate`'s last branch (voxcpm2.py:592-626).
Both run the identical prefill body and write the identical tensor names.
`T_ref` in the metadata tells them apart; `T_prompt > 0` marks both as
continuation fixtures for the Rust gates.

Sequence (patch_size=4, latent_dim=64), text = prompt_text + target_text:
    text_token = prefix_tokens ++ text_token(ends 101) ++ zeros(T_prompt)
    audio_feat = prefix_feats  ++ zeros(text_length)   ++ prompt_feat
    text_mask  = prefix_t_mask ++ ones(text_length)    ++ zeros(T_prompt)
    audio_mask = prefix_a_mask ++ zeros(text_length)   ++ ones(T_prompt)
`prefix_*` is `_make_ref_prefix(ref_feat)` with a reference, empty without.

Encoding (`_encode_wav`, voxcpm2.py:396-422): 16 kHz mono, padded to a
multiple of patch_size * chunk_size (2560) samples BEFORE the VAE encode.
The reference pads on the RIGHT, the prompt on the LEFT. The encode is
inlined so the padded samples fed to the VAE are saved (`ref_wav`,
`prompt_wav`); each inline encode is checked against `_encode_wav` itself.

`_inference` (voxcpm2.py:1011-1101):
    prefix_feat_cond = feat[:, -1]           -- the prompt's last patch here
    has_continuation_audio = feat_mask[0, -1] == 1
    audio_indices = feat_mask.nonzero()      -- EVERY audio position,
                                                the reference span included
    context_len   = min(streaming_prefix_len - 1, len(audio_indices))
    pred_feat_seq = feat[:, audio_indices[-context_len:]]   (pre-seeded)
Non-streaming decode (voxcpm2.py:671-677): decode `pred_feat_seq` (context
plus generated patches) in one VAE call, then drop
`patch_size * decode_chunk_size * context_len` samples off the front.

Step noise is INJECTED, as in make_step_fixture.py: per iteration a `z_i`
is drawn under torch.manual_seed(7000 + i), saved, and `solve_euler` is
replayed with it and asserted equal to the library's own `solve_euler`.

Outputs, both under --out-dir:
    prefill fixture: text_token, text_mask, audio_mask, ref_wav/ref_feat
        (reference only), prompt_wav, prompt_feat, combined_embed,
        feat_embed, enc_outputs_post_blend, residual_enc_inputs,
        lm_hidden, residual_hidden. Metadata: T_ref, T_prompt,
        text_length, S, prompt_text, target_text.
    step fixture: init_lm_hidden, init_residual_hidden,
        init_prefix_feat_cond, init_position, t_span, step{i}_* (as
        make_step_fixture.py), context_patches, trim_patches, trim_wav.
        Metadata: context_len, trim_samples, decode_patch_len,
        streaming_prefix_len, n_iterations, n_timesteps, ...

Built in f32, CPU, no_grad, full VoxCPM2Model loaded strict=True.

Paths come from CLI args, with env fallbacks for the directories:
    --ckpt-dir  (VOXCPM_CKPT_DIR)    config.json, model.safetensors,
                                     audiovae.pth, tokenizer files
    --out-dir   (VOXCPM_FIXTURE_DIR) where both fixtures are written
    --pkg-dir   (VOXCPM_PKG_DIR)     optional, prepended to sys.path when
                                     the voxcpm package is not installed
The wavs are CLI args only: --prompt-wav (required), --ref-wav (optional).
"""
import argparse
import os
import sys
import wave
from pathlib import Path

N_ITERATIONS = 3
N_TIMESTEPS = 32
CFG_VALUE = 2.0
MIN_LEN = 2
TEMPERATURE = 1.0
SWAY_SAMPLING_COEF = 1.0
USE_CFG_ZERO_STAR = True
STREAMING_PREFIX_LEN = 4  # `_generate` / `_inference` default
DEFAULT_TARGET_TEXT = "Saya dah cuba macam-macam, tapi honestly it doesn't make sense."


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default=os.environ.get("VOXCPM_CKPT_DIR"))
    p.add_argument("--out-dir", default=os.environ.get("VOXCPM_FIXTURE_DIR"))
    p.add_argument("--pkg-dir", default=os.environ.get("VOXCPM_PKG_DIR"))
    p.add_argument("--prompt-wav", required=True, help="continuation prompt audio")
    p.add_argument("--prompt-text", required=True, help="transcript of --prompt-wav")
    p.add_argument("--ref-wav", default=None, help="reference audio; omit for continuation-only")
    p.add_argument("--target-text", default=DEFAULT_TARGET_TEXT)
    p.add_argument("--prefill-name", default=None, help="prefill fixture file name")
    p.add_argument("--step-name", default=None, help="step fixture file name")
    args = p.parse_args()
    for flag, value, env in [("--ckpt-dir", args.ckpt_dir, "VOXCPM_CKPT_DIR"),
                             ("--out-dir", args.out_dir, "VOXCPM_FIXTURE_DIR")]:
        if not value:
            p.error(f"{flag} is required: pass it or set {env}")
    for flag, value in [("--prompt-wav", args.prompt_wav), ("--ref-wav", args.ref_wav)]:
        if value is not None and not Path(value).is_file():
            p.error(f"{flag} {value!r} is not a file: pass an existing wav")
    mode = "hifi" if args.ref_wav else "continuation"
    args.prefill_name = args.prefill_name or f"prefill_{mode}_fixture.safetensors"
    args.step_name = args.step_name or f"step_{mode}_fixture.safetensors"
    return args


args = parse_args()
if args.pkg_dir:
    sys.path.insert(0, str(Path(args.pkg_dir)))

import torch  # noqa: E402
import librosa  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import LlamaTokenizerFast  # noqa: E402
from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig  # noqa: E402
from voxcpm.modules.audiovae import AudioVAEV2  # noqa: E402

ckpt_dir, out_dir = Path(args.ckpt_dir), Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
HAS_REF = args.ref_wav is not None
print(f"mode: {'ref+continuation (Hi-Fi)' if HAS_REF else 'continuation-only'}", file=sys.stderr)


def rng(t):
    return f"[{t.min().item():.4f}, {t.max().item():.4f}]"


def describe_wav(label, path):
    try:
        with wave.open(str(path), "rb") as w:
            rate, secs, ch = w.getframerate(), w.getnframes() / w.getframerate(), w.getnchannels()
        print(f"{label}: {path}  {secs:.3f}s  {rate} Hz  {ch} channel(s) on disk", file=sys.stderr)
    except (wave.Error, EOFError):
        print(f"{label}: {path}  (not PCM wav; librosa decodes it)", file=sys.stderr)


# --- Build the FULL VoxCPM2Model, f32, CPU, strict=True ----------------------
cfg = VoxCPMConfig.model_validate_json((ckpt_dir / "config.json").read_text())
cfg.device = "cpu"
cfg.dtype = "float32"
tokenizer = LlamaTokenizerFast.from_pretrained(str(ckpt_dir))
audio_vae = AudioVAEV2(config=cfg.audio_vae_config)
model = VoxCPM2Model(cfg, tokenizer, audio_vae, None, device="cpu").to(torch.float32)
sd = load_file(ckpt_dir / "model.safetensors")
vae_ckpt = torch.load(ckpt_dir / "audiovae.pth", map_location="cpu", weights_only=True)
for k, v in vae_ckpt.get("state_dict", vae_ckpt).items():
    sd[f"audio_vae.{k}"] = v
model.load_state_dict(sd, strict=True)
model = model.eval()
print(f"loaded {len(sd)} tensors (model.safetensors + audiovae.pth), strict=True", file=sys.stderr)

P, D = model.patch_size, model.audio_vae.latent_dim
SR = model._encode_sample_rate
PATCH_LEN = P * model.chunk_size


def encode_inline(path, side):
    """`_encode_wav` inlined; returns (padded 16 kHz samples, [T, P, D] feat)."""
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    audio = torch.from_numpy(audio).unsqueeze(0)
    pad_amount = 0
    if audio.size(1) % PATCH_LEN != 0:
        pad_amount = PATCH_LEN - audio.size(1) % PATCH_LEN
        pad = (pad_amount, 0) if side == "left" else (0, pad_amount)
        audio = torch.nn.functional.pad(audio, pad)
    feat = model.audio_vae.encode(audio.to(model.device), SR).cpu()
    feat = feat.view(D, -1, P).permute(1, 2, 0).contiguous()
    library = model._encode_wav(str(path), padding_mode=side)
    assert torch.equal(feat, library), f"inline encode of {path} diverged from _encode_wav"
    print(f"  {side}-padded by {pad_amount} to {audio.size(1)} samples -> {tuple(feat.shape)} "
          f"range {rng(feat)}", file=sys.stderr)
    return audio.contiguous(), feat


prefill_tensors, step_tensors = {}, {}

with torch.no_grad():
    # --- Encode ---------------------------------------------------------------
    if HAS_REF:
        describe_wav("ref wav", args.ref_wav)
        ref_wav, ref_feat = encode_inline(args.ref_wav, "right")
        prefill_tensors["ref_wav"], prefill_tensors["ref_feat"] = ref_wav, ref_feat
    describe_wav("prompt wav", args.prompt_wav)
    prompt_wav, prompt_feat = encode_inline(args.prompt_wav, "left")
    prefill_tensors["prompt_wav"], prefill_tensors["prompt_feat"] = prompt_wav, prompt_feat
    T_ref = ref_feat.size(0) if HAS_REF else 0
    T_prompt = prompt_feat.size(0)

    # --- Sequence, exactly as `_generate` builds it ---------------------------
    text = args.prompt_text + args.target_text
    print(f"text: {text!r}", file=sys.stderr)
    text_token = torch.LongTensor(model.text_tokenizer(text))
    text_token = torch.cat([text_token, torch.tensor([model.audio_start_token], dtype=torch.int32)], dim=-1)
    text_length = text_token.shape[0]
    prompt_pad_token = torch.zeros(T_prompt, dtype=torch.int32)
    text_pad_feat = torch.zeros((text_length, P, D), dtype=torch.float32)
    tokens = [text_token, prompt_pad_token]
    feats = [text_pad_feat, prompt_feat]
    t_masks = [torch.ones(text_length, dtype=torch.int32), torch.zeros(T_prompt, dtype=torch.int32)]
    a_masks = [torch.zeros(text_length, dtype=torch.int32), torch.ones(T_prompt, dtype=torch.int32)]
    if HAS_REF:
        ref_tokens, ref_feats, ref_t_mask, ref_a_mask = model._make_ref_prefix(ref_feat, text_token.device)
        tokens, feats = [ref_tokens] + tokens, [ref_feats] + feats
        t_masks, a_masks = [ref_t_mask] + t_masks, [ref_a_mask] + a_masks
    full_text_token = torch.cat(tokens)
    full_audio_feat = torch.cat(feats, dim=0)
    full_text_mask = torch.cat(t_masks)
    full_audio_mask = torch.cat(a_masks)
    S = full_text_token.shape[0]
    assert S == text_length + T_prompt + (T_ref + 2 if HAS_REF else 0)
    print(f"T_ref={T_ref} T_prompt={T_prompt} text_length={text_length} S={S}", file=sys.stderr)

    text_b = full_text_token.unsqueeze(0)
    text_mask_b = full_text_mask.unsqueeze(0)
    feat_b = full_audio_feat.unsqueeze(0).to(torch.float32)
    feat_mask_b = full_audio_mask.unsqueeze(0)

    # --- Prefill body (`_inference`, voxcpm2.py:1011-1058) --------------------
    feat_embed = model.enc_to_lm_proj(model.feat_encoder(feat_b))
    scale_emb = model.config.lm_config.scale_emb if model.config.lm_config.use_mup else 1.0
    text_embed = model.base_lm.embed_tokens(text_b) * scale_emb
    combined_embed = text_mask_b.unsqueeze(-1) * text_embed + feat_mask_b.unsqueeze(-1) * feat_embed

    prefix_feat_cond = feat_b[:, -1, ...].clone()
    has_continuation_audio = feat_mask_b[0, -1].item() == 1
    assert has_continuation_audio, "a continuation sequence must end on an audio position"
    audio_indices = feat_mask_b.squeeze(0).nonzero(as_tuple=True)[0]
    context_len = min(STREAMING_PREFIX_LEN - 1, len(audio_indices))
    context_patches = feat_b[:, audio_indices[-context_len:], :, :].clone()  # [1, ctx, P, D]
    pred_feat_seq = list(context_patches.split(1, dim=1))
    print(f"context_len={context_len} (audio positions {len(audio_indices)}, "
          f"streaming_prefix_len={STREAMING_PREFIX_LEN})", file=sys.stderr)

    enc_outputs, kv_cache_tuple = model.base_lm(inputs_embeds=combined_embed, is_causal=True)
    model.base_lm.kv_cache.fill_caches(kv_cache_tuple)
    enc_outputs_post_blend = (
        model.fsq_layer(enc_outputs) * feat_mask_b.unsqueeze(-1)
        + enc_outputs * text_mask_b.unsqueeze(-1)
    )
    lm_hidden = enc_outputs_post_blend[:, -1, :].clone()
    residual_enc_inputs = model.fusion_concat_proj(
        torch.cat((enc_outputs_post_blend, feat_mask_b.unsqueeze(-1) * feat_embed), dim=-1)
    )
    residual_enc_outputs, residual_kv_cache_tuple = model.residual_lm(
        inputs_embeds=residual_enc_inputs, is_causal=True
    )
    model.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
    residual_hidden = residual_enc_outputs[:, -1, :].clone()

    prefill_tensors["text_token"] = full_text_token.contiguous()
    prefill_tensors["text_mask"] = full_text_mask.to(torch.float32).contiguous()
    prefill_tensors["audio_mask"] = full_audio_mask.to(torch.float32).contiguous()
    prefill_tensors["combined_embed"] = combined_embed.contiguous()
    prefill_tensors["feat_embed"] = feat_embed.contiguous()
    prefill_tensors["enc_outputs_post_blend"] = enc_outputs_post_blend.contiguous()
    prefill_tensors["residual_enc_inputs"] = residual_enc_inputs.contiguous()
    prefill_tensors["lm_hidden"] = lm_hidden.contiguous()
    prefill_tensors["residual_hidden"] = residual_hidden.contiguous()

    step_tensors["init_lm_hidden"] = lm_hidden.contiguous()
    step_tensors["init_residual_hidden"] = residual_hidden.contiguous()
    step_tensors["init_prefix_feat_cond"] = prefix_feat_cond.contiguous()
    step_tensors["init_position"] = torch.tensor([S], dtype=torch.int64)
    step_tensors["context_patches"] = context_patches.squeeze(0).contiguous()

    # --- Per-patch loop, first N_ITERATIONS, noise injected -------------------
    cfm = model.feat_decoder
    t_span = torch.linspace(1, 0, N_TIMESTEPS + 1, dtype=torch.float32)
    t_span = t_span + SWAY_SAMPLING_COEF * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    zero_init_steps = max(1, int((N_TIMESTEPS + 1) * 0.04))
    step_tensors["t_span"] = t_span.contiguous()

    for i in range(N_ITERATIONS):
        dit_hidden = torch.cat((model.lm_to_dit_proj(lm_hidden), model.res_to_dit_proj(residual_hidden)), dim=-1)
        step_tensors[f"step{i}_mu"] = dit_hidden.contiguous()
        mu, cond = dit_hidden, prefix_feat_cond.transpose(1, 2).contiguous()
        torch.manual_seed(7000 + i)
        z = torch.randn((1, D, P), dtype=torch.float32) * TEMPERATURE
        step_tensors[f"step{i}_z"] = z.contiguous()

        x = z.clone()
        t, dt = t_span[0], t_span[0] - t_span[1]
        for step in range(1, len(t_span)):
            if USE_CFG_ZERO_STAR and step <= zero_init_steps:
                dphi_dt = torch.zeros_like(x)
            else:
                # Lockstep copy of make_step_fixture.py's replay of
                # unified_cfm.py `solve_euler`, batch 1.
                b = x.size(0)
                x_in = torch.zeros([2 * b, D, x.size(2)], dtype=x.dtype)
                mu_in = torch.zeros([2 * b, mu.size(1)], dtype=x.dtype)
                t_in = torch.zeros([2 * b], dtype=x.dtype)
                dt_in = torch.zeros([2 * b], dtype=x.dtype)
                cond_in = torch.zeros([2 * b, D, cond.size(2)], dtype=x.dtype)
                x_in[:b], x_in[b:] = x, x
                mu_in[:b] = mu
                t_in[:b], t_in[b:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:b], dt_in[b:] = dt.unsqueeze(0), dt.unsqueeze(0)
                dt_in = torch.zeros_like(dt_in)  # mean_mode is False
                cond_in[:b], cond_in[b:] = cond, cond
                dphi_dt = cfm.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [b, b], dim=0)
                st_star = cfm.optimized_scale(dphi_dt.view(b, -1), cfg_dphi_dt.view(b, -1))
                st_star = st_star.view(b, *([1] * (len(dphi_dt.shape) - 1)))
                dphi_dt = cfg_dphi_dt * st_star + CFG_VALUE * (dphi_dt - cfg_dphi_dt * st_star)
            x = x - dt * dphi_dt
            t = t - dt
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]
        out_y = cfm.solve_euler(x=z, t_span=t_span, mu=mu, cond=cond, cfg_value=CFG_VALUE,
                                use_cfg_zero_star=USE_CFG_ZERO_STAR)
        assert torch.equal(out_y, x), f"iteration {i}: replayed trajectory diverged from solve_euler"

        pred_feat = x.transpose(1, 2)  # [1, P, D]
        step_tensors[f"step{i}_pred_feat"] = pred_feat.contiguous()
        curr_embed = model.enc_to_lm_proj(model.feat_encoder(pred_feat.unsqueeze(1)))[:, 0, :]
        step_tensors[f"step{i}_curr_embed"] = curr_embed.contiguous()
        pred_feat_seq.append(pred_feat.unsqueeze(1))
        prefix_feat_cond = pred_feat

        stop_logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
        step_tensors[f"step{i}_stop_logits"] = stop_logits.contiguous()
        stop_argmax = stop_logits.argmax(dim=-1)[0].item()
        assert not ((i > MIN_LEN) and stop_argmax == 1), f"iteration {i}: stop guard fired"

        lm_hidden_pre = model.base_lm.forward_step(
            curr_embed, torch.tensor([model.base_lm.kv_cache.step()], device=curr_embed.device)
        ).clone()
        step_tensors[f"step{i}_lm_hidden_pre_fsq"] = lm_hidden_pre.contiguous()
        lm_hidden = model.fsq_layer(lm_hidden_pre)
        step_tensors[f"step{i}_lm_hidden_post"] = lm_hidden.contiguous()
        residual_hidden = model.residual_lm.forward_step(
            model.fusion_concat_proj(torch.cat((lm_hidden, curr_embed), dim=-1)),
            torch.tensor([model.residual_lm.kv_cache.step()], device=curr_embed.device),
        ).clone()
        step_tensors[f"step{i}_residual_hidden_post"] = residual_hidden.contiguous()
        print(f"iteration {i + 1}/{N_ITERATIONS}: pred_feat range {rng(pred_feat)} "
              f"stop argmax {stop_argmax}", file=sys.stderr)

    # --- Decode and trim, exactly as `_generate`'s non-streaming tail ---------
    seq = torch.cat(pred_feat_seq, dim=1)  # [1, ctx + N, P, D]
    latent_pred = seq.permute(0, 3, 1, 2).reshape(1, D, -1)  # "b t p d -> b d (t p)"
    decoded = model.audio_vae.decode(latent_pred.to(torch.float32))
    decode_patch_len = P * model._decode_chunk_size
    trim_wav = decoded[..., decode_patch_len * context_len:].squeeze(1).cpu()
    trim_samples = trim_wav.shape[-1]
    if trim_samples != N_ITERATIONS * decode_patch_len:
        print(f"WARNING: trimmed {trim_samples} samples, not {N_ITERATIONS} x {decode_patch_len}: "
              "the VAE decode is not length-exact here", file=sys.stderr)
    step_tensors["trim_patches"] = seq.squeeze(0).contiguous()
    step_tensors["trim_wav"] = trim_wav.contiguous()
    print(f"decode: {tuple(decoded.shape)} -> trimmed {decode_patch_len} x {context_len} samples "
          f"-> {trim_samples} samples", file=sys.stderr)

common_meta = {
    "dtype": "f32",
    "mode": "hifi" if HAS_REF else "continuation",
    "prompt_text": args.prompt_text,
    "target_text": args.target_text,
    "T_ref": str(T_ref),
    "T_prompt": str(T_prompt),
    "text_length": str(text_length),
    "S": str(S),
    "patch_size": str(P),
    "latent_dim": str(D),
}
prefill_path, step_path = out_dir / args.prefill_name, out_dir / args.step_name
save_file(prefill_tensors, str(prefill_path), metadata={**common_meta, "sample_rate": str(SR)})
save_file(step_tensors, str(step_path), metadata={
    **common_meta,
    "n_iterations": str(N_ITERATIONS),
    "n_timesteps": str(N_TIMESTEPS),
    "cfg_value": str(CFG_VALUE),
    "min_len": str(MIN_LEN),
    "temperature": str(TEMPERATURE),
    "sway_sampling_coef": str(SWAY_SAMPLING_COEF),
    "use_cfg_zero_star": str(USE_CFG_ZERO_STAR),
    "zero_init_steps": str(zero_init_steps),
    "streaming_prefix_len": str(STREAMING_PREFIX_LEN),
    "context_len": str(context_len),
    "decode_patch_len": str(decode_patch_len),
    "trim_samples": str(trim_samples),
})
for name, tensors, path in [("prefill", prefill_tensors, prefill_path), ("step", step_tensors, step_path)]:
    print(f"--- {name}: {path} ---", file=sys.stderr)
    for key, t in tensors.items():
        print(f"  {key}: {tuple(t.shape)} {t.dtype}  range {rng(t)}", file=sys.stderr)
