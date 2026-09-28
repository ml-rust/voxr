#!/usr/bin/env python3
"""Reference ground truth for VoxCPM2's PREFILL stage, in both prefix modes.

Default mode is reference (isolated) voice cloning, i.e.
`reference_wav_path` in voxcpm_clone.py / voxcpm2.py:531-569 -- NOT
`prompt_wav_path` continuation mode. The reference transcript is never used;
only `target_text` is tokenized (voxcpm2.py:533 `text = target_text`).

Pass `--zero-shot` for the NO-REFERENCE prefix instead (voxcpm2.py:571-590,
the `elif len(prompt_wav_path) == 0` branch). Both modes run the identical
prefill body and dump the identical tensor names, so one Rust gate covers
both; `T_ref` in the metadata is what tells them apart.

Prefix, reference mode (voxcpm2.py `_make_ref_prefix`, :424-454;
patch_size=4, latent_dim=64):
    z1     = zeros((1, 4, 64))
    tokens = cat([[103],       zeros(T_ref),  [104]])
    feats  = cat([ z1,          ref_feat,      z1  ])
    t_mask = cat([[1],         zeros(T_ref),  [1] ])
    a_mask = cat([[0],         ones(T_ref),   [0] ])

Full sequence, reference mode (voxcpm2.py :556-569), length
S = T_ref + 2 + text_length:
    text_token = cat([ref_tokens, text_token])   # text_token already ends
                                                  # with audio_start=101
    audio_feat = cat([ref_feats,  zeros(text_length, 4, 64)])
    text_mask  = cat([ref_t_mask, ones(text_length)])
    audio_mask = cat([ref_a_mask, zeros(text_length)])

Full sequence, zero-shot mode (voxcpm2.py :571-590), length S = text_length.
There is NO prefix at all: no 103, no filler rows, no 104, and neither
zero-patch bookend.
    text_token = text_token                      # target text ++ [101]
    audio_feat = zeros(text_length, 4, 64)
    text_mask  = ones(text_length)
    audio_mask = zeros(text_length)

Prefill body (voxcpm2.py `_inference`, :1011-1058), run unchanged in both
modes:
    feat_embed     = enc_to_lm_proj(feat_encoder(feat))
    text_embed     = base_lm.embed_tokens(text) * scale_emb        # scale_emb == 1.0
    combined_embed = text_mask[...,None]*text_embed + feat_mask[...,None]*feat_embed
    enc_outputs, kv = base_lm(inputs_embeds=combined_embed, is_causal=True)
    enc_outputs = fsq_layer(enc_outputs)*feat_mask[...,None] + enc_outputs*text_mask[...,None]
    lm_hidden   = enc_outputs[:, -1, :]
    residual_enc_inputs  = fusion_concat_proj(cat((enc_outputs, feat_mask[...,None]*feat_embed), -1))
    residual_enc_outputs, rkv = residual_lm(inputs_embeds=residual_enc_inputs, is_causal=True)
    residual_hidden = residual_enc_outputs[:, -1, :]

Zero-shot exercises code shapes the reference fixture never reaches:
`audio_mask` is zero everywhere, so `masked_feat` and the masked
`feat_embed` are exactly zero at every position, `combined_embed` is
token-embedding only, and `fusion_concat_proj` sees a zero right half across
the whole sequence. Those are shape-valid under the reference fixture too,
which is exactly why they need their own numerical gate.

The prefill is fully deterministic -- no sampling happens before the first
CFM call -- so this is a real numerical gate, not a seeded-noise fixture.

The reference wav is right-padded to a multiple of patch_size*chunk_size
(2560, NOT the VAE's own 640) at 16 kHz mono BEFORE VAE encoding
(voxcpm2.py `_encode_wav`, :396-422); the patch fold is
`feat.view(64,-1,4).permute(1,2,0)` -> `[T,4,64]`. This fixture replicates
`_encode_wav` inline (rather than calling the private method) purely so the
padded 16 kHz samples that are actually fed to the VAE can be saved
alongside the resulting `ref_feat`. Zero-shot mode reads no wav and emits
neither `ref_wav` nor `ref_feat`.

Built in f32, CPU, torch.no_grad(), full VoxCPM2Model loaded strict=True (all
of base_lm, residual_lm, feat_encoder, fsq_layer, enc_to_lm_proj,
fusion_concat_proj together -- this stage crosses all of them, so no
submodule can be loaded in isolation the way the sibling fixtures do).

Usage:
    make_prefill_fixture.py PKG_DIR CKPT_DIR OUT.safetensors --ref-wav REF.wav [--zero-shot]

`--ref-wav` names a real recorded voice sample. It is required unless
`--zero-shot` is given, and it can also come from the `PREFILL_REF_WAV`
env var.
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
ZERO_SHOT = "--zero-shot" in rest

REF_WAV_PATH = None
if "--ref-wav" in rest:
    REF_WAV_PATH = Path(rest[rest.index("--ref-wav") + 1])
elif os.environ.get("PREFILL_REF_WAV"):
    REF_WAV_PATH = Path(os.environ["PREFILL_REF_WAV"])

print(f"mode: {'zero-shot (no reference)' if ZERO_SHOT else 'reference voice clone'}",
      file=sys.stderr)

if not ZERO_SHOT:
    if REF_WAV_PATH is None:
        raise SystemExit(
            "reference mode needs --ref-wav REF.wav or PREFILL_REF_WAV set "
            "(pass --zero-shot to run without a reference)"
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

print(f"encode_sample_rate={model._encode_sample_rate} chunk_size={model.chunk_size} "
      f"patch_size={model.patch_size} latent_dim={model.audio_vae.latent_dim} "
      f"ref_audio_start_token={model.ref_audio_start_token} "
      f"ref_audio_end_token={model.ref_audio_end_token} "
      f"audio_start_token={model.audio_start_token}", file=sys.stderr)

tensors = {}

with torch.no_grad():
    if not ZERO_SHOT:
        # --- _encode_wav, inlined so the padded samples fed to the VAE survive ---
        audio, _ = librosa.load(str(REF_WAV_PATH), sr=model._encode_sample_rate, mono=True)
        audio = torch.from_numpy(audio).unsqueeze(0)  # [1, N]
        patch_len = model.patch_size * model.chunk_size
        pad_amount = 0
        if audio.size(1) % patch_len != 0:
            pad_amount = patch_len - audio.size(1) % patch_len
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
        print(f"ref wav resampled to {model._encode_sample_rate} Hz, {audio.size(1)} samples "
              f"({audio.size(1) / model._encode_sample_rate:.3f}s), right-padded by {pad_amount} "
              f"samples to a multiple of patch_size*chunk_size={patch_len}", file=sys.stderr)

        vae_out = model.audio_vae.encode(audio.to(model.device), model._encode_sample_rate).cpu()
        ref_feat = vae_out.view(model.audio_vae.latent_dim, -1, model.patch_size).permute(1, 2, 0).contiguous()
        T_ref = ref_feat.size(0)
        print(f"ref_feat: {tuple(ref_feat.shape)}  range [{ref_feat.min():.4f}, {ref_feat.max():.4f}]  "
              f"T_ref={T_ref}", file=sys.stderr)
    else:
        T_ref = 0

    # --- Text tokens: both modes tokenize target_text ONLY (voxcpm2.py:533, :573) ---
    text_token = torch.LongTensor(model.text_tokenizer(TARGET_TEXT))
    text_token = torch.cat(
        [text_token, torch.tensor([model.audio_start_token], dtype=torch.int32, device=text_token.device)],
        dim=-1,
    )
    text_length = text_token.shape[0]

    text_pad_feat = torch.zeros((text_length, model.patch_size, model.audio_vae.latent_dim), dtype=torch.float32)

    if not ZERO_SHOT:
        # --- Ref prefix (voxcpm2.py `_make_ref_prefix`) ---
        ref_tokens, ref_feats, ref_t_mask, ref_a_mask = model._make_ref_prefix(ref_feat, text_token.device)

        full_text_token = torch.cat([ref_tokens, text_token])
        full_audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
        full_text_mask = torch.cat([ref_t_mask, torch.ones(text_length, dtype=torch.int32)])
        full_audio_mask = torch.cat([ref_a_mask, torch.zeros(text_length, dtype=torch.int32)])
    else:
        # --- Zero-shot: no prefix at all (voxcpm2.py:571-590) ---
        full_text_token = text_token
        full_audio_feat = text_pad_feat
        full_text_mask = torch.ones(text_length, dtype=torch.int32)
        full_audio_mask = torch.zeros(text_length, dtype=torch.int32)

    S = full_text_token.shape[0]
    expected_S = text_length + (T_ref + 2 if T_ref > 0 else 0)
    print(f"T_ref={T_ref}  text_length={text_length}  S={S}  (expected S = {expected_S})",
          file=sys.stderr)
    assert S == expected_S

    ids = full_text_token.tolist()
    print(f"first 12 token ids: {ids[:12]}", file=sys.stderr)
    print(f"last 6 token ids: {ids[-6:]}", file=sys.stderr)
    print(f"full_text_token dtype: {full_text_token.dtype}", file=sys.stderr)

    text_token_b = full_text_token.unsqueeze(0)
    text_mask_b = full_text_mask.unsqueeze(0)
    audio_feat_b = full_audio_feat.unsqueeze(0).to(torch.float32)
    audio_mask_b = full_audio_mask.unsqueeze(0)

    # --- Prefill body (voxcpm2.py `_inference`, :1011-1058) ---
    feat_embed = model.feat_encoder(audio_feat_b)  # [1, S, h_enc]
    feat_embed = model.enc_to_lm_proj(feat_embed)  # [1, S, 2048]

    scale_emb = model.config.lm_config.scale_emb if model.config.lm_config.use_mup else 1.0
    print(f"use_mup={model.config.lm_config.use_mup} scale_emb={scale_emb}", file=sys.stderr)

    text_embed = model.base_lm.embed_tokens(text_token_b) * scale_emb
    combined_embed = text_mask_b.unsqueeze(-1) * text_embed + audio_mask_b.unsqueeze(-1) * feat_embed

    enc_outputs, _kv = model.base_lm(inputs_embeds=combined_embed, is_causal=True)

    enc_outputs_post_blend = (
        model.fsq_layer(enc_outputs) * audio_mask_b.unsqueeze(-1)
        + enc_outputs * text_mask_b.unsqueeze(-1)
    )
    lm_hidden = enc_outputs_post_blend[:, -1, :].clone()

    residual_enc_inputs = model.fusion_concat_proj(
        torch.cat((enc_outputs_post_blend, audio_mask_b.unsqueeze(-1) * feat_embed), dim=-1)
    )
    residual_enc_outputs, _rkv = model.residual_lm(inputs_embeds=residual_enc_inputs, is_causal=True)
    residual_hidden = residual_enc_outputs[:, -1, :].clone()

def rng(t):
    return f"[{t.min().item():.4f}, {t.max().item():.4f}]"

if not ZERO_SHOT:
    tensors["ref_wav"] = audio.contiguous()
    tensors["ref_feat"] = ref_feat
tensors["text_token"] = full_text_token.contiguous()
tensors["text_mask"] = full_text_mask.to(torch.float32).contiguous()
tensors["audio_mask"] = full_audio_mask.to(torch.float32).contiguous()
tensors["combined_embed"] = combined_embed.contiguous()
tensors["feat_embed"] = feat_embed.contiguous()
tensors["enc_outputs_post_blend"] = enc_outputs_post_blend.contiguous()
tensors["residual_enc_inputs"] = residual_enc_inputs.contiguous()
tensors["lm_hidden"] = lm_hidden.contiguous()
tensors["residual_hidden"] = residual_hidden.contiguous()

print("--- tensor shapes and ranges ---", file=sys.stderr)
for name, t in tensors.items():
    print(f"  {name}: {tuple(t.shape)} {t.dtype}  range {rng(t)}", file=sys.stderr)

save_file(tensors, str(out), metadata={
    "dtype": "f32",
    "mode": "zero_shot" if ZERO_SHOT else "reference",
    "ref_wav_path": "" if ZERO_SHOT else str(REF_WAV_PATH),
    "target_text": TARGET_TEXT,
    "sample_rate": str(model._encode_sample_rate),
    "T_ref": str(T_ref),
    "text_length": str(text_length),
    "S": str(S),
    "patch_size": str(model.patch_size),
    "latent_dim": str(model.audio_vae.latent_dim),
    "ref_audio_start_token": str(model.ref_audio_start_token),
    "ref_audio_end_token": str(model.ref_audio_end_token),
    "audio_start_token": str(model.audio_start_token),
})
print(f"wrote {out}  ({len(tensors)} tensors)", file=sys.stderr)
