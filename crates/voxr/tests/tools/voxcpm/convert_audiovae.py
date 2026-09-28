#!/usr/bin/env python3
"""Convert VoxCPM2's `audiovae.pth` to safetensors for the Rust loader.

Two things are absorbed here so the runtime never has to know about them:

1. **Pickle.** `audiovae.pth` is a torch pickle. boostr reads safetensors, and
   teaching a Rust runtime to unpickle arbitrary Python objects is a security
   and maintenance cost with no upside for a file that never changes.

2. **Weight normalization.** The checkpoint stores `weight_g` / `weight_v`
   pairs from `torch.nn.utils.weight_norm`, where the effective weight is
   `g * v / ||v||` with the norm taken over every dimension except output
   channels. That reparameterization exists to conditioning gradients during
   TRAINING; at inference it is a fixed rescale that would otherwise have to run on
   every load. Folding it here turns each pair into the single `weight` tensor
   the forward pass actually uses.

    VENV/bin/python convert_audiovae.py IN.pth OUT.safetensors
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def fold_weight_norm(state):
    """Replace every `weight_g`/`weight_v` pair with the weight it encodes.

    Returns the new state dict and the number of pairs folded.
    """
    out, folded = {}, 0
    g_suffix, v_suffix = ".weight_g", ".weight_v"
    for key, value in state.items():
        if key.endswith(v_suffix):
            continue  # handled with its `_g` partner
        if not key.endswith(g_suffix):
            out[key] = value
            continue
        stem = key[: -len(g_suffix)]
        v = state.get(stem + v_suffix)
        if v is None:
            raise KeyError(f"{key} has no matching {stem}{v_suffix}")
        g = value
        # Norm over every axis but the output-channel axis, matching
        # weight_norm's default `dim=0`.
        dims = tuple(range(1, v.dim()))
        norm = v.norm(dim=dims, keepdim=True)
        out[stem + ".weight"] = (g / norm) * v
        folded += 1
    return out, folded


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} IN.pth OUT.safetensors")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    blob = torch.load(src, map_location="cpu", weights_only=True)
    state = blob["state_dict"] if "state_dict" in blob else blob
    meta = blob.get("metadata", {}) if isinstance(blob, dict) else {}
    print(f"read {len(state)} tensors from {src.name}; metadata {meta}")

    folded_state, folded = fold_weight_norm(state)
    print(f"folded {folded} weight_norm pair(s) -> {len(folded_state)} tensors")

    # safetensors refuses shared storage, which `torch.load` hands out freely.
    folded_state = {k: v.contiguous().clone() for k, v in folded_state.items()}

    save_file(
        folded_state,
        str(dst),
        metadata={"source": src.name, "sample_rate": str(meta.get("kwargs", {}).get("sample_rate", ""))},
    )
    print(f"wrote {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
