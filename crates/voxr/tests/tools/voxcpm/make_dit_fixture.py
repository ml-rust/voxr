#!/usr/bin/env python3
"""Reference input -> output pairs for VoxCPM2's DiT estimator (feat_decoder.estimator).

Two DiT implementations exist in the package: local_dit.py (v1) concatenates
[(mu + t).unsqueeze(1), cond, x]; local_dit_v2.py (v2, class VoxCPMLocDiT,
imported as VoxCPMLocDiTV2) concatenates [mu (2 tokens), t.unsqueeze(1), cond, x].
voxcpm2.py:45 imports VoxCPMLocDiTV2 from local_dit_v2 and voxcpm2.py:217
instantiates it as feat_decoder.estimator -- v2 is the live path, and this
fixture captures it.

Targets the estimator directly, not the CFM sampler: the estimator is a pure
deterministic function of (x, mu, t, cond, dt), so no noise handling is needed.

Built in f32: RoPE and RMSNorm upcast internally regardless of storage dtype,
so an f32 reference is the cleaner ground truth and the Rust port can be held
to it without bf16 rounding muddying the comparison.
"""
import json, sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(sys.argv[1])))
from voxcpm.modules.minicpm4 import MiniCPM4Config  # noqa: E402
from voxcpm.modules.locdit import VoxCPMLocDiTV2     # noqa: E402

ckpt_dir, out = Path(sys.argv[2]), Path(sys.argv[3])
cfg = json.loads((ckpt_dir / "config.json").read_text())

# Mirrors VoxCPM2Model.__init__ (model/voxcpm2.py:207-213): decoder_config
# STARTS from lm_config, so num_key_value_heads / rope_theta / rope_scaling /
# rms_norm_eps / use_mup all come from there unchanged. Only the five fields
# below are overridden from dit_config.
dec = MiniCPM4Config(**cfg["lm_config"])
d = cfg["dit_config"]
dec.hidden_size = d["hidden_dim"]
dec.intermediate_size = d["ffn_dim"]
dec.num_attention_heads = d["num_heads"]
dec.num_hidden_layers = d["num_layers"]
dec.kv_channels = d["kv_channels"]
dec.vocab_size = 0

feat_dim = cfg["feat_dim"]
patch_size = cfg["patch_size"]

print(f"hidden_size={dec.hidden_size} intermediate_size={dec.intermediate_size} "
      f"num_heads={dec.num_attention_heads} num_kv_heads={dec.num_key_value_heads} "
      f"num_layers={dec.num_hidden_layers} kv_channels={dec.kv_channels} "
      f"rms_norm_eps={dec.rms_norm_eps} rope_theta={dec.rope_theta} "
      f"use_mup={dec.use_mup} scale_emb={dec.scale_emb} "
      f"feat_dim={feat_dim} patch_size={patch_size}", file=sys.stderr)

# VoxCPMLocDiTV2(config, in_channels=feat_dim) -- model/voxcpm2.py:217. Note
# in_channels is feat_dim (the channel axis of x/cond), NOT patch_size.
model = VoxCPMLocDiTV2(dec, in_channels=feat_dim)
sd = load_file(ckpt_dir / "model.safetensors")
sub = {k[len("feat_decoder.estimator."):]: v for k, v in sd.items() if k.startswith("feat_decoder.estimator.")}
model.load_state_dict(sub, strict=True)
model = model.to(torch.float32).eval()
print(f"loaded {len(sub)} tensors", file=sys.stderr)

# --- Facts for the Rust port -------------------------------------------------
print(
    "concat order (local_dit_v2.py:109-110): "
    "mu = mu.view(x.size(0), -1, x.size(-1)); "
    "x = torch.cat([mu, (t).unsqueeze(1), cond, x], dim=1)",
    file=sys.stderr,
)
print(
    "output slice (local_dit_v2.py:113): "
    "hidden = hidden[:, prefix + mu.size(1) + 1 :, :]  "
    "where prefix = cond.size(1) (after cond_proj, i.e. patch_size)",
    file=sys.stderr,
)
print(
    "backbone call (local_dit_v2.py:112): "
    "hidden, _ = self.decoder(x, is_causal=False)  -- bidirectional, not causal",
    file=sys.stderr,
)
print(
    "sinusoidal embedding (local_dit_v2.py:7-22): dim=config.hidden_size "
    f"({dec.hidden_size}), half_dim=dim//2, emb=exp(arange(half_dim)*-(log(10000)/(half_dim-1))), "
    "scale=1000 (default arg to SinusoidalPosEmb.forward, unchanged at every call site)",
    file=sys.stderr,
)

tensors = {}
with torch.no_grad():
    # Distinct seeds from make_baselm_fixture.py (4000+i/4100) and
    # make_locenc_fixture.py (3000+i/3100): use 5000+i for cases here.
    # t values span near-0, mid, and near-1; dt=0 in every case (the
    # inference-time value per voxcpm2.py's sampling loop). Case 3 uses
    # batch > 1.
    cases = [
        # (batch, t_value)
        (1, 0.001),
        (1, 0.5),
        (1, 0.999),
        (3, 0.25),
    ]
    for i, (b, t_val) in enumerate(cases):
        torch.manual_seed(5000 + i)
        x = torch.randn(b, feat_dim, patch_size, dtype=torch.float32)
        mu = torch.randn(b, 2 * dec.hidden_size, dtype=torch.float32)
        cond = torch.randn(b, feat_dim, patch_size, dtype=torch.float32)
        t = torch.full((b,), t_val, dtype=torch.float32)
        dt = torch.zeros(b, dtype=torch.float32)

        out_y = model(x, mu, t, cond, dt)

        tensors[f"case{i}_x"] = x.contiguous()
        tensors[f"case{i}_mu"] = mu.contiguous()
        tensors[f"case{i}_cond"] = cond.contiguous()
        tensors[f"case{i}_t"] = t.contiguous()
        tensors[f"case{i}_dt"] = dt.contiguous()
        tensors[f"case{i}_out"] = out_y.contiguous()
        print(f"case{i}: x{tuple(x.shape)} mu{tuple(mu.shape)} cond{tuple(cond.shape)} "
              f"t={t_val} dt=0 -> out{tuple(out_y.shape)}  "
              f"range [{out_y.min():.4f}, {out_y.max():.4f}]", file=sys.stderr)

# f32 so the Rust side is compared against full precision, not bf16 rounding.
save_file(tensors, str(out), metadata={
    "dtype": "f32",
    "feat_dim": str(feat_dim),
    "patch_size": str(patch_size),
    "hidden_size": str(dec.hidden_size),
})
print(f"wrote {out}", file=sys.stderr)
