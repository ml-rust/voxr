#!/usr/bin/env python3
"""Reference input -> output pairs for VoxCPM2's fsq_layer and the seven
remaining top-level projections, for verifying the Rust port.

fsq_layer is a ScalarQuantizationLayer (model/voxcpm2.py:222-227):
    fsq_layer = ScalarQuantizationLayer(lm_config.hidden_size, lm_config.hidden_size,
                                         scalar_quantization_latent_dim, scalar_quantization_scale)
Its EVAL-mode forward (modules/layers/scalar_quantization_layer.py) is:
    hidden = in_proj(hidden)                          # 2048 -> 512
    hidden = tanh(hidden)
    hidden = round(hidden * scale) / scale             # scale = 9, straight quantize (no STE)
    return out_proj(hidden)                            # 512 -> 2048
The straight-through branch (training-only) is never taken here because the
model runs in eval().

The other seven tensors are plain nn.Linear layers used directly by
VoxCPM2Model.forward / forward_step (voxcpm2.py:228-236, :321-368):
    enc_to_lm_proj    Linear(encoder_config.hidden_dim=1024 -> lm_config.hidden_size=2048)
    lm_to_dit_proj    Linear(lm_config.hidden_size=2048 -> dit_config.hidden_dim=1024)
    res_to_dit_proj   Linear(lm_config.hidden_size=2048 -> dit_config.hidden_dim=1024)
    fusion_concat_proj Linear(lm_config.hidden_size*2=4096 -> lm_config.hidden_size=2048)
    stop_proj         Linear(lm_config.hidden_size=2048 -> lm_config.hidden_size=2048)
    stop_head         Linear(lm_config.hidden_size=2048 -> 2, bias=False)
The stop chain (voxcpm2.py:368): stop_head(SiLU(stop_proj(lm_hidden))).

Built in f32, CPU, torch.no_grad(). Modules are constructed directly and
loaded strict=True from the 15 relevant model.safetensors keys -- the full
VoxCPM2Model is never instantiated, since none of these eight modules depend
on the rest of the model's state.
"""
import json, sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(sys.argv[1])))
from voxcpm.modules.layers.scalar_quantization_layer import ScalarQuantizationLayer  # noqa: E402

ckpt_dir, out = Path(sys.argv[2]), Path(sys.argv[3])
cfg = json.loads((ckpt_dir / "config.json").read_text())

lm_hidden = cfg["lm_config"]["hidden_size"]
enc_hidden = cfg["encoder_config"]["hidden_dim"]
dit_hidden = cfg["dit_config"]["hidden_dim"]
latent_dim = cfg["scalar_quantization_latent_dim"]
scale = cfg["scalar_quantization_scale"]

print(f"lm_hidden={lm_hidden} enc_hidden={enc_hidden} dit_hidden={dit_hidden} "
      f"latent_dim={latent_dim} scale={scale}", file=sys.stderr)

fsq_layer = ScalarQuantizationLayer(lm_hidden, lm_hidden, latent_dim, scale)
enc_to_lm_proj = nn.Linear(enc_hidden, lm_hidden)
lm_to_dit_proj = nn.Linear(lm_hidden, dit_hidden)
res_to_dit_proj = nn.Linear(lm_hidden, dit_hidden)
fusion_concat_proj = nn.Linear(lm_hidden * 2, lm_hidden)
stop_proj = nn.Linear(lm_hidden, lm_hidden)
stop_actn = nn.SiLU()
stop_head = nn.Linear(lm_hidden, 2, bias=False)

modules = {
    "fsq_layer": fsq_layer,
    "enc_to_lm_proj": enc_to_lm_proj,
    "lm_to_dit_proj": lm_to_dit_proj,
    "res_to_dit_proj": res_to_dit_proj,
    "fusion_concat_proj": fusion_concat_proj,
    "stop_proj": stop_proj,
    "stop_head": stop_head,
}

sd = load_file(ckpt_dir / "model.safetensors")
loaded_count = 0
for name, mod in modules.items():
    prefix = f"{name}."
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    mod.load_state_dict(sub, strict=True)
    mod.to(torch.float32).eval()
    loaded_count += len(sub)
    print(f"loaded {name}: {len(sub)} tensors", file=sys.stderr)

print(f"loaded {loaded_count} tensors total", file=sys.stderr)

tensors = {}

with torch.no_grad():
    # --- 1. FSQ, ordinary inputs ---------------------------------------
    # Distinct seeds from the sibling fixtures (4000+, 5000+, 3000+): use
    # 6000+i here. Case 2 is rank-2 [b, 2048] -- the per-step decode path
    # (voxcpm2.py:1098) calls fsq_layer(lm_hidden) where lm_hidden is [b, h].
    fsq_cases = [
        (2, 3),   # [b, T, 2048]
        (1, 7),   # [b, T, 2048]
        None,     # rank-2 [b, 2048]
    ]
    for i, shape in enumerate(fsq_cases):
        torch.manual_seed(6000 + i)
        if shape is None:
            x = torch.randn(4, lm_hidden, dtype=torch.float32)
        else:
            b, t = shape
            x = torch.randn(b, t, lm_hidden, dtype=torch.float32)
        y = fsq_layer(x)
        tensors[f"fsq{i}_in"] = x.contiguous()
        tensors[f"fsq{i}_out"] = y.contiguous()
        print(f"fsq{i}: {tuple(x.shape)} -> {tuple(y.shape)}  "
              f"range [{y.min():.4f}, {y.max():.4f}]", file=sys.stderr)

    # --- 2. FSQ, TIE inputs ---------------------------------------------
    # Bypass in_proj/tanh entirely: build post-tanh values h = (k+0.5)/scale
    # for integer k spanning the representable range (|h| < 1), then apply
    # ONLY the quantization step round(h*scale)/scale. A random draw over
    # in_proj+tanh essentially never lands exactly on a tie, so this is the
    # only way to exercise the rounding rule directly.
    ks = list(range(-scale, scale))  # (k+0.5)/9 in (-1, 1) for k in [-9, 8]
    tie_in = torch.tensor([(k + 0.5) / scale for k in ks], dtype=torch.float32)
    tie_out = torch.round(tie_in * scale) / scale
    tensors["tie_in"] = tie_in.contiguous()
    tensors["tie_out"] = tie_out.contiguous()
    print(f"tie: {tuple(tie_in.shape)} -> {tuple(tie_out.shape)}  "
          f"range [{tie_out.min():.4f}, {tie_out.max():.4f}]", file=sys.stderr)
    print("tie pairs (k, h_in, h_in*scale, round(h_in*scale), h_out):", file=sys.stderr)
    for k, hv, ov in zip(ks, tie_in.tolist(), tie_out.tolist()):
        print(f"  k={k:+3d}  in={hv:+.10f}  in*scale={hv*scale:+.4f}  out={ov:+.10f}", file=sys.stderr)
    # Rounding rule observed from the printed numbers: e.g. k=-1 gives
    # in*scale=-0.5 -> out=-0.0 (rounds to even 0, not away from zero to
    # -1/9); k=1 gives in*scale=1.5 -> out=2/9 (rounds to even 2, not down
    # to 1). Every .5 case above rounds to the nearest EVEN integer, i.e.
    # torch.round implements round-half-to-even (banker's rounding), NOT
    # round-half-away-from-zero.

    # --- 3. Each remaining projection ------------------------------------
    torch.manual_seed(6100)
    enc_in = torch.randn(2, 5, enc_hidden, dtype=torch.float32)
    enc_out = enc_to_lm_proj(enc_in)
    tensors["enc_to_lm_proj_in"] = enc_in.contiguous()
    tensors["enc_to_lm_proj_out"] = enc_out.contiguous()
    print(f"enc_to_lm_proj: {tuple(enc_in.shape)} -> {tuple(enc_out.shape)}  "
          f"range [{enc_out.min():.4f}, {enc_out.max():.4f}]", file=sys.stderr)

    torch.manual_seed(6101)
    lm_to_dit_in = torch.randn(2, 5, lm_hidden, dtype=torch.float32)
    lm_to_dit_out = lm_to_dit_proj(lm_to_dit_in)
    tensors["lm_to_dit_proj_in"] = lm_to_dit_in.contiguous()
    tensors["lm_to_dit_proj_out"] = lm_to_dit_out.contiguous()
    print(f"lm_to_dit_proj: {tuple(lm_to_dit_in.shape)} -> {tuple(lm_to_dit_out.shape)}  "
          f"range [{lm_to_dit_out.min():.4f}, {lm_to_dit_out.max():.4f}]", file=sys.stderr)

    torch.manual_seed(6102)
    res_to_dit_in = torch.randn(2, 5, lm_hidden, dtype=torch.float32)
    res_to_dit_out = res_to_dit_proj(res_to_dit_in)
    tensors["res_to_dit_proj_in"] = res_to_dit_in.contiguous()
    tensors["res_to_dit_proj_out"] = res_to_dit_out.contiguous()
    print(f"res_to_dit_proj: {tuple(res_to_dit_in.shape)} -> {tuple(res_to_dit_out.shape)}  "
          f"range [{res_to_dit_out.min():.4f}, {res_to_dit_out.max():.4f}]", file=sys.stderr)

    torch.manual_seed(6103)
    fusion_in = torch.randn(2, 5, lm_hidden * 2, dtype=torch.float32)
    fusion_out = fusion_concat_proj(fusion_in)
    tensors["fusion_concat_proj_in"] = fusion_in.contiguous()
    tensors["fusion_concat_proj_out"] = fusion_out.contiguous()
    print(f"fusion_concat_proj: {tuple(fusion_in.shape)} -> {tuple(fusion_out.shape)}  "
          f"range [{fusion_out.min():.4f}, {fusion_out.max():.4f}]", file=sys.stderr)

    torch.manual_seed(6104)
    stop_proj_in = torch.randn(2, 5, lm_hidden, dtype=torch.float32)
    stop_proj_out = stop_proj(stop_proj_in)
    tensors["stop_proj_in"] = stop_proj_in.contiguous()
    tensors["stop_proj_out"] = stop_proj_out.contiguous()
    print(f"stop_proj: {tuple(stop_proj_in.shape)} -> {tuple(stop_proj_out.shape)}  "
          f"range [{stop_proj_out.min():.4f}, {stop_proj_out.max():.4f}]", file=sys.stderr)

    torch.manual_seed(6105)
    stop_head_in = torch.randn(2, 5, lm_hidden, dtype=torch.float32)
    stop_head_out = stop_head(stop_head_in)
    tensors["stop_head_in"] = stop_head_in.contiguous()
    tensors["stop_head_out"] = stop_head_out.contiguous()
    print(f"stop_head: {tuple(stop_head_in.shape)} -> {tuple(stop_head_out.shape)}  "
          f"range [{stop_head_out.min():.4f}, {stop_head_out.max():.4f}]", file=sys.stderr)

    # stop chain: stop_head(SiLU(stop_proj(x))) -- voxcpm2.py:368
    torch.manual_seed(6106)
    stop_chain_in = torch.randn(2, 5, lm_hidden, dtype=torch.float32)
    stop_chain_out = stop_head(stop_actn(stop_proj(stop_chain_in)))
    tensors["stop_chain_in"] = stop_chain_in.contiguous()
    tensors["stop_chain_out"] = stop_chain_out.contiguous()
    print(f"stop_chain: {tuple(stop_chain_in.shape)} -> {tuple(stop_chain_out.shape)}  "
          f"range [{stop_chain_out.min():.4f}, {stop_chain_out.max():.4f}]", file=sys.stderr)

# f32 so the Rust side is compared against full precision, not bf16 rounding.
save_file(tensors, str(out), metadata={
    "dtype": "f32",
    "lm_hidden": str(lm_hidden),
    "enc_hidden": str(enc_hidden),
    "dit_hidden": str(dit_hidden),
    "latent_dim": str(latent_dim),
    "scale": str(scale),
})
print(f"wrote {out}  ({len(tensors)} tensors)", file=sys.stderr)
