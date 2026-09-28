#!/usr/bin/env python3
"""Reference latent -> waveform pairs for the Rust AudioVAE decoder.

A port is only correct if it reproduces the original numerically. This writes
the decoder's own input and output for fixed, seeded latents so the Rust side
can be checked against the model that produced them rather than against a
plausible-sounding waveform.
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
state = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
missing, unexpected = vae.load_state_dict(state, strict=False)
print(f"missing {len(missing)} unexpected {len(unexpected)}", file=sys.stderr)
if missing:
    print("  first missing:", missing[:5], file=sys.stderr)
vae.eval()

tensors = {}
with torch.no_grad():
    for i, frames in enumerate((8, 23)):
        torch.manual_seed(1000 + i)
        # [B, latent_dim, T]
        z = torch.randn(1, cfg.latent_dim, frames)
        # `decode` with no sr_cond, which is exactly what VoxCPM2 calls. It
        # conditions on out_sample_rate (48000 -> bucket 3), NOT the 16 kHz
        # input rate. Conditioning on 16000 picks bucket 0 and a different
        # affine per block, so a fixture built that way would verify a decoder
        # nothing ever runs.
        y = vae.decode(z)
        tensors[f"case{i}_latent"] = z.contiguous()
        tensors[f"case{i}_waveform"] = y.contiguous()
        print(f"case{i}: latent {tuple(z.shape)} -> waveform {tuple(y.shape)}  "
              f"range [{y.min():.4f}, {y.max():.4f}]", file=sys.stderr)

save_file(tensors, str(out), metadata={"out_sample_rate": str(cfg.out_sample_rate), "sr_bucket": "3"})
print(f"wrote {out}", file=sys.stderr)
