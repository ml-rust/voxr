#!/usr/bin/env python3
"""Reference waveform -> latent pairs for the Rust AudioVAE encoder.

Goes through `encode` exactly as VoxCPM2 calls it, so whatever that path does
(sampling, mu-only, scaling) is baked into the fixture rather than guessed at.
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(sys.argv[1])))
from voxcpm.modules.audiovae import AudioVAEV2, AudioVAEConfigV2  # noqa: E402

ckpt, out = Path(sys.argv[2]), Path(sys.argv[3])
cfg = AudioVAEConfigV2()
vae = AudioVAEV2(config=cfg)
vae.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"],
                    strict=False)
vae.eval()

import inspect  # noqa: E402
print("--- preprocess ---", file=sys.stderr)
print(inspect.getsource(vae.preprocess), file=sys.stderr)

tensors = {}
with torch.no_grad():
    for i, samples in enumerate((16000, 40960)):
        torch.manual_seed(2000 + i)
        x = torch.randn(1, 1, samples) * 0.1
        # Through `encode`, exactly as VoxCPM2 calls it: preprocess, then the
        # encoder's `mu`. Deterministic - no reparameterization sampling.
        mu = vae.encode(x, 16000)
        pre = vae.preprocess(x, 16000)
        tensors[f"case{i}_wave"] = x.contiguous()
        tensors[f"case{i}_preprocessed"] = pre.contiguous()
        tensors[f"case{i}_mu"] = mu.contiguous()
        print(f"case{i}: wave {tuple(x.shape)} -> pre {tuple(pre.shape)} -> mu {tuple(mu.shape)}",
              file=sys.stderr)
save_file(tensors, str(out), metadata={"note": "AudioVAE.encode(x, 16000) -> mu; deterministic", "in_rate": "16000"})
print(f"wrote {out}", file=sys.stderr)
