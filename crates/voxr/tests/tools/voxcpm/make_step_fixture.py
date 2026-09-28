#!/usr/bin/env python3
"""Reference ground truth for VoxCPM2's per-patch generation loop -- FIRST 3
ITERATIONS ONLY (`_inference`, model/voxcpm2.py:1064-1101).

Direct successor of make_prefill_fixture.py: this script re-runs that exact
same prefill (same ref wav, same target text, same strict=True full-model
load) so a Rust gate can run ITS OWN already-verified prefill and then drive
the per-patch loop forward from the resulting `lm_hidden` / `residual_hidden`
/ `prefix_feat_cond` / kv-cache position -- it does not need this script's
prefill numbers, only the per-step ones that follow from them.

Per-iteration order (voxcpm2.py:1064-1101), verified against the source and
not re-derived here:
    1. dit_hidden = cat(lm_to_dit_proj(lm_hidden), res_to_dit_proj(residual_hidden), -1)   [1,2048]
    2. pred_feat  = feat_decoder(mu=dit_hidden, patch_size, cond=prefix_feat_cond.transpose(1,2),
                                  n_timesteps, cfg_value).transpose(1,2)                     [1,4,64]
    3. curr_embed = enc_to_lm_proj(feat_encoder(pred_feat.unsqueeze(1)))
    4. pred_feat_seq.append(pred_feat); prefix_feat_cond = pred_feat
    5. stop_flag  = argmax(stop_head(stop_actn(stop_proj(lm_hidden))))   -- on the CURRENT
       (pre-step) lm_hidden, guard is `if i > min_len and stop_flag == 1: break`
    6. lm_hidden  = base_lm.forward_step(curr_embed[:,0,:], step);  lm_hidden = fsq_layer(lm_hidden)
    7. residual_hidden = residual_lm.forward_step(
           fusion_concat_proj(cat(lm_hidden_POST_fsq, curr_embed[:,0,:], -1)), step)

Initial state, from the SAME prefill as make_prefill_fixture.py:
    prefix_feat_cond = feat[:, -1, ...]   -- the zero text-pad patch in reference mode
    lm_hidden        = enc_outputs_post_blend[:, -1, :]   (post-fsq-blend, matches prefill fixture)
    residual_hidden  = residual_enc_outputs[:, -1, :]
    kv caches         filled from the prefill's own (kv_cache_tuple, residual_kv_cache_tuple)

Settings (overriding the library's inference_timesteps=10/cfg_value=2.0
defaults at voxcpm2.py:983-984): n_timesteps=32,
cfg_value=2.0, min_len=2, temperature=1.0, sway_sampling_coef=1.0,
use_cfg_zero_star=True -- these are UnifiedCFM.forward's own defaults
(unified_cfm.py:57-60) since voxcpm2.py:1065-1071 never overrides them.

With min_len=2 the break guard is `i > min_len` (strict), so it can only
fire at i=3 or later -- across i=0,1,2 (the 3 iterations captured here) the
stop flag structurally CANNOT break the loop, regardless of its value. Each
iteration's stop_logits/argmax are still computed and saved.

Determinism: UnifiedCFM.forward draws `z = torch.randn(...) * temperature`
internally (unified_cfm.py:64) and torch's RNG cannot be matched in Rust.
Following make_cfm_fixture.py's pattern, `feat_decoder(...)`/`forward` is
never called; per iteration a `z_i` is drawn under an explicit
`torch.manual_seed`, saved, and `cfm.solve_euler` is called directly with it
and the realized `t_span` (identical every iteration, saved once). The loop
body below is a lockstep re-implementation of `solve_euler`
(unified_cfm.py:84-136), and each iteration's final `x` is asserted equal to
calling `cfm.solve_euler` itself -- so the recorded state is proven to be
what the real sampler would have produced for that injected z.

Built in f32, CPU, torch.no_grad()/inference_mode(), full VoxCPM2Model loaded
strict=True (the loop crosses base_lm, residual_lm, feat_decoder, feat_encoder,
fsq_layer, enc_to_lm_proj, fusion_concat_proj, lm_to_dit_proj, res_to_dit_proj,
stop_proj/stop_actn/stop_head together, so no submodule can be isolated the
way the sibling fixtures do).

Usage:
    make_step_fixture.py PKG_DIR CKPT_DIR OUT.safetensors --ref-wav REF.wav

`--ref-wav` must name the same recorded voice sample passed to
`make_prefill_fixture.py`. It can also come from the `STEP_REF_WAV` env var.
"""
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(sys.argv[1])))
from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig  # noqa: E402
from transformers import LlamaTokenizerFast  # noqa: E402
from voxcpm.modules.audiovae import AudioVAEV2  # noqa: E402
import librosa  # noqa: E402

ckpt_dir, out = Path(sys.argv[2]), Path(sys.argv[3])
rest = sys.argv[4:]

# --- Same reference wav and target text as make_prefill_fixture.py, EXACTLY,
# so a Rust gate can run its own already-verified prefill and step from there.
REF_WAV_PATH = None
if "--ref-wav" in rest:
    REF_WAV_PATH = Path(rest[rest.index("--ref-wav") + 1])
elif os.environ.get("STEP_REF_WAV"):
    REF_WAV_PATH = Path(os.environ["STEP_REF_WAV"])
if REF_WAV_PATH is None:
    raise SystemExit(
        "needs --ref-wav REF.wav or STEP_REF_WAV set, the same wav passed "
        "to make_prefill_fixture.py"
    )
if not REF_WAV_PATH.exists():
    raise FileNotFoundError(f"reference wav not found: {REF_WAV_PATH}")

import wave  # noqa: E402
with wave.open(str(REF_WAV_PATH), "rb") as w:
    on_disk_rate = w.getframerate()
    on_disk_secs = w.getnframes() / on_disk_rate
    on_disk_ch = w.getnchannels()
print(f"ref wav: {REF_WAV_PATH}  {on_disk_secs:.3f}s  {on_disk_rate} Hz  "
      f"{on_disk_ch} channel(s) on disk", file=sys.stderr)

TARGET_TEXT = "Saya dah cuba macam-macam, tapi honestly it doesn't make sense."
print(f"target text: {TARGET_TEXT!r}", file=sys.stderr)

# --- Per-patch loop settings ------------------------------------------------
N_ITERATIONS = 3
N_TIMESTEPS = 32
CFG_VALUE = 2.0
MIN_LEN = 2
TEMPERATURE = 1.0
SWAY_SAMPLING_COEF = 1.0
USE_CFG_ZERO_STAR = True
print(f"n_iterations={N_ITERATIONS} n_timesteps={N_TIMESTEPS} cfg_value={CFG_VALUE} "
      f"min_len={MIN_LEN} temperature={TEMPERATURE} sway_sampling_coef={SWAY_SAMPLING_COEF} "
      f"use_cfg_zero_star={USE_CFG_ZERO_STAR}", file=sys.stderr)

# --- Build the FULL VoxCPM2Model, f32, CPU, strict=True ----------------------
cfg = VoxCPMConfig.model_validate_json((ckpt_dir / "config.json").read_text())
cfg.device = "cpu"
cfg.dtype = "float32"

tokenizer = LlamaTokenizerFast.from_pretrained(str(ckpt_dir))
audio_vae = AudioVAEV2(config=cfg.audio_vae_config)
model = VoxCPM2Model(cfg, tokenizer, audio_vae, None, device="cpu")
model = model.to(torch.float32)

sd = load_file(ckpt_dir / "model.safetensors")
vae_ckpt = torch.load(ckpt_dir / "audiovae.pth", map_location="cpu", weights_only=True)
vae_sd = vae_ckpt.get("state_dict", vae_ckpt)
for k, v in vae_sd.items():
    sd[f"audio_vae.{k}"] = v

model.load_state_dict(sd, strict=True)
model = model.eval()
print(f"loaded {len(sd)} tensors (model.safetensors + audiovae.pth), strict=True",
      file=sys.stderr)

tensors = {}


def rng(t):
    return f"[{t.min().item():.4f}, {t.max().item():.4f}]"


with torch.no_grad():
    # --- Prefill: byte-identical to make_prefill_fixture.py --------------------
    audio, _ = librosa.load(str(REF_WAV_PATH), sr=model._encode_sample_rate, mono=True)
    audio = torch.from_numpy(audio).unsqueeze(0)  # [1, N]
    patch_len = model.patch_size * model.chunk_size
    pad_amount = 0
    if audio.size(1) % patch_len != 0:
        pad_amount = patch_len - audio.size(1) % patch_len
        audio = torch.nn.functional.pad(audio, (0, pad_amount))
    print(f"ref wav resampled to {model._encode_sample_rate} Hz, {audio.size(1)} samples, "
          f"right-padded by {pad_amount} samples to a multiple of {patch_len}", file=sys.stderr)

    vae_out = model.audio_vae.encode(audio.to(model.device), model._encode_sample_rate).cpu()
    ref_feat = vae_out.view(model.audio_vae.latent_dim, -1, model.patch_size).permute(1, 2, 0).contiguous()
    T_ref = ref_feat.size(0)

    text_token = torch.LongTensor(model.text_tokenizer(TARGET_TEXT))
    text_token = torch.cat(
        [text_token, torch.tensor([model.audio_start_token], dtype=torch.int32, device=text_token.device)],
        dim=-1,
    )
    text_length = text_token.shape[0]

    ref_tokens, ref_feats, ref_t_mask, ref_a_mask = model._make_ref_prefix(ref_feat, text_token.device)

    text_pad_feat = torch.zeros((text_length, model.patch_size, model.audio_vae.latent_dim), dtype=torch.float32)
    full_text_token = torch.cat([ref_tokens, text_token])
    full_audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
    full_text_mask = torch.cat([ref_t_mask, torch.ones(text_length, dtype=torch.int32)])
    full_audio_mask = torch.cat([ref_a_mask, torch.zeros(text_length, dtype=torch.int32)])

    S = full_text_token.shape[0]
    assert S == T_ref + 2 + text_length
    print(f"T_ref={T_ref}  text_length={text_length}  S={S}", file=sys.stderr)

    text_token_b = full_text_token.unsqueeze(0)
    text_mask_b = full_text_mask.unsqueeze(0)
    audio_feat_b = full_audio_feat.unsqueeze(0).to(torch.float32)
    audio_mask_b = full_audio_mask.unsqueeze(0)

    feat_embed = model.feat_encoder(audio_feat_b)  # [1, S, h_enc]
    feat_embed = model.enc_to_lm_proj(feat_embed)  # [1, S, 2048]

    scale_emb = model.config.lm_config.scale_emb if model.config.lm_config.use_mup else 1.0
    text_embed = model.base_lm.embed_tokens(text_token_b) * scale_emb
    combined_embed = text_mask_b.unsqueeze(-1) * text_embed + audio_mask_b.unsqueeze(-1) * feat_embed

    enc_outputs, kv_cache_tuple = model.base_lm(inputs_embeds=combined_embed, is_causal=True)
    model.base_lm.kv_cache.fill_caches(kv_cache_tuple)

    enc_outputs_post_blend = (
        model.fsq_layer(enc_outputs) * audio_mask_b.unsqueeze(-1)
        + enc_outputs * text_mask_b.unsqueeze(-1)
    )
    lm_hidden = enc_outputs_post_blend[:, -1, :].clone()

    residual_enc_inputs = model.fusion_concat_proj(
        torch.cat((enc_outputs_post_blend, audio_mask_b.unsqueeze(-1) * feat_embed), dim=-1)
    )
    residual_enc_outputs, residual_kv_cache_tuple = model.residual_lm(
        inputs_embeds=residual_enc_inputs, is_causal=True
    )
    model.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
    residual_hidden = residual_enc_outputs[:, -1, :].clone()

    # prefix_feat_cond, per voxcpm2.py:1023 -- set from the ORIGINAL feat input
    # (audio_feat_b here), i.e. the last patch of the full sequence, which in
    # reference-only mode is the zero text-pad patch.
    prefix_feat_cond = audio_feat_b[:, -1, ...].clone()  # [1, 4, 64]

    print(f"init_lm_hidden: {tuple(lm_hidden.shape)} range {rng(lm_hidden)}", file=sys.stderr)
    print(f"init_residual_hidden: {tuple(residual_hidden.shape)} range {rng(residual_hidden)}", file=sys.stderr)
    print(f"init_prefix_feat_cond: {tuple(prefix_feat_cond.shape)} range {rng(prefix_feat_cond)}", file=sys.stderr)
    print(f"init_position (S): {S}", file=sys.stderr)

    tensors["init_lm_hidden"] = lm_hidden.contiguous()
    tensors["init_residual_hidden"] = residual_hidden.contiguous()
    tensors["init_prefix_feat_cond"] = prefix_feat_cond.contiguous()
    tensors["init_position"] = torch.tensor([S], dtype=torch.int64)

    # --- solve_euler settings shared by every iteration ------------------------
    cfm = model.feat_decoder
    feat_dim = model.audio_vae.latent_dim  # 64
    patch_size = model.patch_size  # 4

    t_span = torch.linspace(1, 0, N_TIMESTEPS + 1, dtype=torch.float32)
    t_span = t_span + SWAY_SAMPLING_COEF * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    zero_init_steps = max(1, int((N_TIMESTEPS + 1) * 0.04))
    print(f"t_span: {tuple(t_span.shape)} range {rng(t_span)}  zero_init_steps={zero_init_steps}",
          file=sys.stderr)
    tensors["t_span"] = t_span.contiguous()

    # --- Per-patch loop, first N_ITERATIONS iterations of voxcpm2.py:1064-1101 -
    for i in range(N_ITERATIONS):
        print(f"iteration {i + 1}/{N_ITERATIONS} ...", file=sys.stderr)

        # 1. dit_hidden
        dit_hidden_1 = model.lm_to_dit_proj(lm_hidden)
        dit_hidden_2 = model.res_to_dit_proj(residual_hidden)
        dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)  # [1, 2048]
        print(f"  step{i}_mu: {tuple(dit_hidden.shape)} range {rng(dit_hidden)}", file=sys.stderr)
        tensors[f"step{i}_mu"] = dit_hidden.contiguous()

        # 2. pred_feat, via solve_euler driven with SAVED, injected noise --
        # mirrors make_cfm_fixture.py's pattern: forward()/feat_decoder() is
        # never called since it would draw its own unsaved z internally.
        mu = dit_hidden
        cond = prefix_feat_cond.transpose(1, 2).contiguous()  # [1, 64, 4]

        torch.manual_seed(7000 + i)  # distinct from make_cfm_fixture.py's 6000
        z = torch.randn((1, feat_dim, patch_size), dtype=torch.float32) * TEMPERATURE
        tensors[f"step{i}_z"] = z.contiguous()

        x = z.clone()
        t, dt = t_span[0], t_span[0] - t_span[1]
        for step in range(1, len(t_span)):
            if USE_CFG_ZERO_STAR and step <= zero_init_steps:
                dphi_dt = torch.zeros_like(x)
            else:
                b = x.size(0)
                x_in = torch.zeros([2 * b, feat_dim, x.size(2)], dtype=x.dtype)
                mu_in = torch.zeros([2 * b, mu.size(1)], dtype=x.dtype)
                t_in = torch.zeros([2 * b], dtype=x.dtype)
                dt_in = torch.zeros([2 * b], dtype=x.dtype)
                cond_in = torch.zeros([2 * b, feat_dim, cond.size(2)], dtype=x.dtype)
                x_in[:b], x_in[b:] = x, x
                mu_in[:b] = mu
                t_in[:b], t_in[b:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:b], dt_in[b:] = dt.unsqueeze(0), dt.unsqueeze(0)
                dt_in = torch.zeros_like(dt_in)  # mean_mode is False, unified_cfm.py:113-114
                cond_in[:b], cond_in[b:] = cond, cond

                dphi_dt = cfm.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [b, b], dim=0)

                positive_flat = dphi_dt.view(b, -1)
                negative_flat = cfg_dphi_dt.view(b, -1)
                st_star = cfm.optimized_scale(positive_flat, negative_flat)
                st_star = st_star.view(b, *([1] * (len(dphi_dt.shape) - 1)))

                dphi_dt = cfg_dphi_dt * st_star + CFG_VALUE * (dphi_dt - cfg_dphi_dt * st_star)

            x = x - dt * dphi_dt
            t = t - dt
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]

        # Ground truth from the library itself, every argument explicit --
        # proves the hand-rolled loop above matches solve_euler exactly.
        out_y = cfm.solve_euler(
            x=z,
            t_span=t_span,
            mu=mu,
            cond=cond,
            cfg_value=CFG_VALUE,
            use_cfg_zero_star=USE_CFG_ZERO_STAR,
        )
        assert torch.equal(out_y, x), f"iteration {i}: replayed trajectory diverged from solve_euler"

        pred_feat = x.transpose(1, 2)  # [1, 4, 64]
        print(f"  step{i}_pred_feat: {tuple(pred_feat.shape)} range {rng(pred_feat)}", file=sys.stderr)
        tensors[f"step{i}_pred_feat"] = pred_feat.contiguous()

        # 3. curr_embed
        curr_embed_seq = model.feat_encoder(pred_feat.unsqueeze(1))  # [1, 1, h_enc]
        curr_embed_seq = model.enc_to_lm_proj(curr_embed_seq)  # [1, 1, 2048]
        curr_embed = curr_embed_seq[:, 0, :]  # [1, 2048]
        print(f"  step{i}_curr_embed: {tuple(curr_embed.shape)} range {rng(curr_embed)}", file=sys.stderr)
        tensors[f"step{i}_curr_embed"] = curr_embed.contiguous()

        # 4. append / update prefix_feat_cond
        prefix_feat_cond = pred_feat

        # 5. stop check, on the CURRENT (pre-step) lm_hidden
        stop_logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
        stop_argmax = stop_logits.argmax(dim=-1)[0].cpu().item()
        would_break = (i > MIN_LEN) and (stop_argmax == 1)
        print(f"  step{i}_stop_logits: {tuple(stop_logits.shape)} range {rng(stop_logits)}  "
              f"argmax={stop_argmax}  i={i} min_len={MIN_LEN}  would_break={would_break}",
              file=sys.stderr)
        tensors[f"step{i}_stop_logits"] = stop_logits.contiguous()
        assert not would_break, (
            f"iteration {i}: stop flag fired within the first {N_ITERATIONS} iterations -- "
            "brief assumed this guard cannot break here"
        )

        # 6. base_lm step, then fsq
        lm_hidden = model.base_lm.forward_step(
            curr_embed, torch.tensor([model.base_lm.kv_cache.step()], device=curr_embed.device)
        ).clone()
        lm_hidden_pre_fsq = lm_hidden.clone()
        print(f"  step{i}_lm_hidden_pre_fsq: {tuple(lm_hidden_pre_fsq.shape)} range {rng(lm_hidden_pre_fsq)}",
              file=sys.stderr)
        tensors[f"step{i}_lm_hidden_pre_fsq"] = lm_hidden_pre_fsq.contiguous()
        lm_hidden = model.fsq_layer(lm_hidden)
        print(f"  step{i}_lm_hidden_post: {tuple(lm_hidden.shape)} range {rng(lm_hidden)}", file=sys.stderr)
        tensors[f"step{i}_lm_hidden_post"] = lm_hidden.contiguous()

        # 7. residual_lm step, fed lm_hidden POST-fsq
        curr_residual_input = model.fusion_concat_proj(torch.cat((lm_hidden, curr_embed), dim=-1))
        residual_hidden = model.residual_lm.forward_step(
            curr_residual_input, torch.tensor([model.residual_lm.kv_cache.step()], device=curr_embed.device)
        ).clone()
        print(f"  step{i}_residual_hidden_post: {tuple(residual_hidden.shape)} range {rng(residual_hidden)}",
              file=sys.stderr)
        tensors[f"step{i}_residual_hidden_post"] = residual_hidden.contiguous()

print("--- tensor shapes and ranges ---", file=sys.stderr)
for name, t in tensors.items():
    print(f"  {name}: {tuple(t.shape)} {t.dtype}  range {rng(t)}", file=sys.stderr)

save_file(tensors, str(out), metadata={
    "dtype": "f32",
    "ref_wav_path": str(REF_WAV_PATH),
    "target_text": TARGET_TEXT,
    "n_iterations": str(N_ITERATIONS),
    "n_timesteps": str(N_TIMESTEPS),
    "cfg_value": str(CFG_VALUE),
    "min_len": str(MIN_LEN),
    "temperature": str(TEMPERATURE),
    "sway_sampling_coef": str(SWAY_SAMPLING_COEF),
    "use_cfg_zero_star": str(USE_CFG_ZERO_STAR),
    "zero_init_steps": str(zero_init_steps),
    "feat_dim": str(feat_dim),
    "patch_size": str(patch_size),
    "S": str(S),
})
print(f"wrote {out}  ({len(tensors)} tensors)", file=sys.stderr)
