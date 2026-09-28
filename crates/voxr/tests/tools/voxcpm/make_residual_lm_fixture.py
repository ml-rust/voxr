#!/usr/bin/env python3
"""Reference inputs_embeds -> hidden_states pairs for VoxCPM2's residual_lm.

residual_lm is a MiniCPM4 decoder-only transformer (MiniCPMModel), built from
lm_config with exactly three overrides (model/voxcpm2.py:189-193):
    residual_lm_config = config.lm_config.model_copy(deep=True)
    residual_lm_config.num_hidden_layers = config.residual_lm_num_layers  # 8
    residual_lm_config.vocab_size = 0
    residual_lm_config.no_rope = config.residual_lm_no_rope              # True
There is no `residual_lm_config` section in config.json -- it is derived here
the same way voxcpm2.py derives it, from lm_config plus the two top-level
residual_lm_* keys. no_rope=True means rope_emb is None (model.py:348-351)
and apply_rotary_pos_emb is never called; attention stays causal.

VoxCPM2 always drives residual_lm with pre-computed inputs_embeds (never
token ids) and is_causal=True (model/voxcpm2.py:337, :1053), so this fixture
reproduces exactly that call. Built in f32: RoPE (when active) and RMSNorm
upcast internally regardless of storage dtype, so an f32 reference is the
cleaner ground truth and the Rust port can be held to it without bf16
rounding muddying the comparison.
"""
import json, sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(sys.argv[1])))
from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel  # noqa: E402

ckpt_dir, out = Path(sys.argv[2]), Path(sys.argv[3])
cfg = json.loads((ckpt_dir / "config.json").read_text())
lm = cfg["lm_config"]

config = MiniCPM4Config(**lm)
config.num_hidden_layers = cfg["residual_lm_num_layers"]
config.vocab_size = 0
config.no_rope = cfg["residual_lm_no_rope"]

rs = config.rope_scaling
print(f"num_layers={config.num_hidden_layers} hidden_size={config.hidden_size} "
      f"num_heads={config.num_attention_heads} num_kv_heads={config.num_key_value_heads} "
      f"kv_channels={config.kv_channels} intermediate_size={config.intermediate_size} "
      f"rms_norm_eps={config.rms_norm_eps} rope_theta={config.rope_theta} "
      f"no_rope={config.no_rope} "
      f"rope_scaling_type={rs.type} short_factor_len={len(rs.short_factor)} "
      f"long_factor_len={len(rs.long_factor)} "
      f"original_max_position_embeddings={rs.original_max_position_embeddings} "
      f"use_mup={config.use_mup} scale_emb={config.scale_emb} "
      f"vocab_size={config.vocab_size} "
      f"max_position_embeddings={config.max_position_embeddings}", file=sys.stderr)

model = MiniCPMModel(config)
sd = load_file(ckpt_dir / "model.safetensors")
sub = {k[len("residual_lm."):]: v for k, v in sd.items() if k.startswith("residual_lm.")}
model.load_state_dict(sub, strict=True)
model = model.to(torch.float32).eval()
print(f"loaded {len(sub)} tensors", file=sys.stderr)
assert len(sub) == 73, f"expected 73 residual_lm.* tensors, got {len(sub)}"

tensors = {}
with torch.no_grad():
    # seq_len=1 exercises the degenerate causal-mask case; seq_len=5 and 17
    # exercise real causal masking across positions.
    for i, (b, t) in enumerate(((1, 1), (1, 5), (1, 17))):
        torch.manual_seed(5000 + i)
        x = torch.randn(b, t, config.hidden_size, dtype=torch.float32)
        y, _ = model(inputs_embeds=x, is_causal=True)
        tensors[f"case{i}_in"] = x.contiguous()
        tensors[f"case{i}_out"] = y.contiguous()
        print(f"case{i}: {tuple(x.shape)} -> {tuple(y.shape)}  "
              f"range [{y.min():.4f}, {y.max():.4f}]", file=sys.stderr)

# --- Decode-mode fixture: full-sequence forward vs step-wise KV-cached
# decode (voxcpm2.py:1053 fill_caches + :1100 forward_step loop). Proves the
# Rust port's step path can be checked against the existing full-sequence
# path, and pins the reference's actual step-wise values.
DECODE_T = 6
with torch.no_grad():
    torch.manual_seed(5100)
    decode_in = torch.randn(1, DECODE_T, config.hidden_size, dtype=torch.float32)

    # Path A: full-sequence causal forward (same call as the case{i} loop above).
    y_full, _ = model(inputs_embeds=decode_in, is_causal=True)

    # Path B: incremental KV-cached decode, one position at a time via
    # forward_step (model.py:388-416), mirroring voxcpm2.py's generate loop.
    max_length = 32
    model.setup_cache(batch_size=1, max_length=max_length, device=decode_in.device, dtype=torch.float32)
    step_outputs = []
    for t in range(DECODE_T):
        pos = torch.tensor([model.kv_cache.step()], device=decode_in.device)
        h = model.forward_step(decode_in[:, t, :], pos)
        step_outputs.append(h)
    y_step = torch.stack(step_outputs, dim=1)  # (1, T, hidden)

max_diff = (y_full - y_step).abs().max().item()
print(f"decode: full-seq vs step-wise max diff = {max_diff:.8e}", file=sys.stderr)
DECODE_TOL = 1e-4
if max_diff > DECODE_TOL:
    print(f"FATAL: decode paths disagree beyond tolerance {DECODE_TOL} (max diff {max_diff:.8e}) "
          f"-- the reference's own full-sequence and step-wise paths disagree; "
          f"the premise that step-wise decode == full-sequence forward is false. Aborting.",
          file=sys.stderr)
    sys.exit(1)

tensors["decode_in"] = decode_in.contiguous()
tensors["decode_out"] = y_step.contiguous()
print(f"decode_in: {tuple(decode_in.shape)} -> decode_out: {tuple(y_step.shape)}  "
      f"range [{y_step.min():.4f}, {y_step.max():.4f}]", file=sys.stderr)

# Cache/mask facts for the Rust port.
kv = model.kv_cache
print(f"kv_cache.max_length={kv.max_length}  "
      f"kv_cache tensor shape (2, layers, batch, kv_heads, max_length, head_dim) = "
      f"{tuple(kv.kv_cache.shape)}  (StaticKVCache.__init__, minicpm4/cache.py:14-28)", file=sys.stderr)
print("step-mode attn mask (minicpm4/model.py:201): "
      "attn_mask = (torch.arange(key_cache.size(2), device=key_cache.device) <= position_id).view(1, 1, 1, -1)",
      file=sys.stderr)

# f32 so the Rust side is compared against full precision, not bf16 rounding.
save_file(tensors, str(out), metadata={"dtype": "f32"})
print(f"wrote {out}", file=sys.stderr)
