#!/usr/bin/env python3
"""Reference input -> output pairs for VoxCPM2's feat_encoder (locenc).

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
from voxcpm.modules.locenc import VoxCPMLocEnc      # noqa: E402

ckpt_dir, out = Path(sys.argv[2]), Path(sys.argv[3])
cfg = json.loads((ckpt_dir / "config.json").read_text())

# Mirrors VoxCPM2Model.__init__: the encoder config STARTS from lm_config, so
# num_key_value_heads / rope_theta / rope_scaling / rms_norm_eps / use_mup all
# come from there unchanged. Only the five fields below are overridden.
enc = MiniCPM4Config(**cfg["lm_config"])
e = cfg["encoder_config"]
enc.hidden_size = e["hidden_dim"]
enc.intermediate_size = e["ffn_dim"]
enc.num_attention_heads = e["num_heads"]
enc.num_hidden_layers = e["num_layers"]
enc.kv_channels = e["kv_channels"]
enc.vocab_size = 0

print(f"kv_heads={enc.num_key_value_heads} head_dim={enc.kv_channels} "
      f"use_mup={enc.use_mup} eps={enc.rms_norm_eps} theta={enc.rope_theta} "
      f"max_pos={enc.max_position_embeddings}", file=sys.stderr)

model = VoxCPMLocEnc(enc, input_dim=cfg["feat_dim"])
sd = load_file(ckpt_dir / "model.safetensors")
sub = {k[len("feat_encoder."):]: v for k, v in sd.items() if k.startswith("feat_encoder.")}
model.load_state_dict(sub, strict=True)
model = model.to(torch.float32).eval()
print(f"loaded {len(sub)} tensors", file=sys.stderr)

tensors = {}
with torch.no_grad():
    for i, (b, t) in enumerate(((1, 2), (1, 5))):
        torch.manual_seed(3000 + i)
        x = torch.randn(b, t, cfg["patch_size"], cfg["feat_dim"], dtype=torch.float32)
        y = model(x)
        tensors[f"case{i}_in"] = x.contiguous()
        tensors[f"case{i}_out"] = y.contiguous()
        print(f"case{i}: {tuple(x.shape)} -> {tuple(y.shape)}  "
              f"range [{y.min():.4f}, {y.max():.4f}]", file=sys.stderr)

# f32 so the Rust side is compared against full precision, not bf16 rounding.
save_file(tensors, str(out), metadata={"dtype": "f32", "patch_size": str(cfg["patch_size"])})
print(f"wrote {out}", file=sys.stderr)
