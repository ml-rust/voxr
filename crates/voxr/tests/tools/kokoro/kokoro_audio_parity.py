#!/usr/bin/env python3
"""Oracle fixture for Kokoro's audio path: the reference `kokoro` package's
`KModel` on one phoneme string, every stage that draws no random number.

Usage:
    KOKORO_DIR=<Kokoro-82M dir> python3 -I kokoro_audio_parity.py OUT.json

`KOKORO_DIR` holds `config.json`, `kokoro-v1_0.pth` and `voices/af_heart.pt`.
Nothing is downloaded.

The fixture holds, for the phonemes of "hello world" in `af_heart`:

* `ids`: the input ids, `[0, *ids, 0]`.
* `speeds`: per speed, the predicted durations and the waveform's length.
  The waveform's samples hold random draws, so only its length is kept.
* `f0`, `n`: the F0 and energy curves at speed 1.
* `trunk`: the decoder's output before the generator, channels
  `trunk_channels`.
* `spectrum`: the generator's STFT (`gen.stft.transform`) of the
  excitation `source(n)` below: every bin's magnitude and phase at frames
  `0, frame_step, ...`. A phase is only well defined where its magnitude
  is not near zero, so the test compares phases modulo 2π there.
* `generator`: `mag` and `phase` from the generator over the trunk, driven
  by the spectrum `spectrum(frames)` below in place of the random source
  module and its STFT: every bin at frames `0, frame_step, ...`.
* `sine`: `SineGen(24000, 300, 8, noise_std=0, voiced_threshold=10)` over
  the contour `sine_f0(t)` below, every harmonic at samples
  `0, sine_step, ...`. The contour is voiced throughout, so no noise is
  drawn, and the random initial phase falls outside every downsampled
  frame.

Both synthetic signals are computed in f64 and cast to f32, as the Rust
test computes them.
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from kokoro.istftnet import SineGen
from kokoro.model import KModel

PHONEMES = "həlˈoʊ wˈɜɹld"
VOICE = "af_heart"
SPEEDS = [1.0, 0.8]
TRUNK_CHANNELS = [0, 128, 256, 384]
FRAME_STEP = 100
SINE_LEN = 3000
SINE_STEP = 30


def hiss(t):
    """A deterministic broadband value in [-0.5, 0.5): a 32-bit integer hash
    of `t`."""
    x = (t * 2654435761) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x45D9F3B) & 0xFFFFFFFF
    x ^= x >> 16
    return x / float(1 << 32) - 0.5


def source(n):
    """Two sines at 150 and 450 Hz plus `hiss`, so every STFT bin holds
    energy and its phase is well defined."""
    t = np.arange(n, dtype=np.float64)
    s = 0.3 * np.sin(2 * math.pi * 150 * t / 24000) + 0.1 * np.sin(2 * math.pi * 450 * t / 24000)
    s = s + 0.2 * np.array([hiss(i) for i in range(n)], dtype=np.float64)
    return s.astype(np.float32)


def spectrum(frames):
    """A smooth excitation spectrum `[22, frames]`: magnitudes in [0.1, 0.9]
    and phases inside (-2.5, 2.5), away from the ±π wrap."""
    f = np.arange(frames, dtype=np.float64)
    mag = [0.5 + 0.4 * np.sin(0.01 * f + b) for b in range(11)]
    phase = [2.5 * np.sin(0.003 * f + 0.7 * b) for b in range(11)]
    return np.stack(mag + phase).astype(np.float32)


def sine_f0(n):
    t = np.arange(n, dtype=np.float64)
    return (120 + 40 * np.sin(2 * math.pi * t / SINE_LEN)).astype(np.float32)


def floats(t):
    return [float(v) for v in t.reshape(-1).tolist()]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: KOKORO_DIR=<dir> kokoro_audio_parity.py OUT.json")
    d = os.environ.get("KOKORO_DIR")
    if not d:
        sys.exit("set KOKORO_DIR to the Kokoro-82M directory")
    torch.manual_seed(0)
    m = KModel(repo_id="hexgrad/Kokoro-82M", config=f"{d}/config.json", model=f"{d}/kokoro-v1_0.pth").eval()
    pack = torch.load(f"{d}/voices/{VOICE}.pt", weights_only=True)
    ids = [0] + [m.vocab[c] for c in PHONEMES if c in m.vocab] + [0]
    ref_s = pack[len(PHONEMES) - 1]
    out = {"phonemes": PHONEMES, "voice": VOICE, "ids": ids, "speeds": []}
    with torch.no_grad():
        for speed in SPEEDS:
            audio, dur = m.forward_with_tokens(torch.LongTensor([ids]), ref_s, speed)
            out["speeds"].append({"speed": speed, "durations": dur.tolist(), "samples": int(audio.shape[-1])})

        n = len(ids)
        input_ids = torch.LongTensor([ids])
        lengths = torch.full((1,), n, dtype=torch.long)
        mask = torch.zeros((1, n), dtype=torch.bool)
        d_en = m.bert_encoder(m.bert(input_ids, attention_mask=(~mask).int())).transpose(-1, -2)
        s = ref_s[:, 128:]
        dd = m.predictor.text_encoder(d_en, s, lengths, mask)
        x, _ = m.predictor.lstm(dd)
        dur = torch.round(torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)).clamp(min=1).long().squeeze()
        idx = torch.repeat_interleave(torch.arange(n), dur)
        aln = torch.zeros((n, idx.shape[0]))
        aln[idx, torch.arange(idx.shape[0])] = 1
        en = dd.transpose(-1, -2) @ aln.unsqueeze(0)
        f0, nn_ = m.predictor.F0Ntrain(en, s)
        asr = m.text_encoder(input_ids, lengths, mask) @ aln.unsqueeze(0)
        dec = m.decoder
        sty = ref_s[:, :128]
        f0c = dec.F0_conv(f0.unsqueeze(1))
        nc = dec.N_conv(nn_.unsqueeze(1))
        trunk = dec.encode(torch.cat([asr, f0c, nc], 1), sty)
        ar = dec.asr_res(asr)
        res = True
        for b in dec.decode:
            if res:
                trunk = torch.cat([trunk, ar, f0c, nc], 1)
            trunk = b(trunk, sty)
            if b.upsample_type != "none":
                res = False
        out["f0"] = floats(f0[0])
        out["n"] = floats(nn_[0])
        out["trunk_channels"] = TRUNK_CHANNELS
        out["trunk"] = [floats(trunk[0, c]) for c in TRUNK_CHANNELS]

        gen = dec.generator
        src = torch.from_numpy(source(trunk.shape[-1] * 300)).unsqueeze(0)
        hs, hp = gen.stft.transform(src)
        out["spectrum"] = {
            "source_len": int(src.shape[-1]),
            "frames": int(hs.shape[-1]),
            "frame_step": FRAME_STEP,
            "mag": [floats(row) for row in hs[0, :, ::FRAME_STEP]],
            "phase": [floats(row) for row in hp[0, :, ::FRAME_STEP]],
        }
        har = torch.from_numpy(spectrum(trunk.shape[-1] * 60 + 1)).unsqueeze(0)
        g = trunk
        for i in range(gen.num_upsamples):
            g = F.leaky_relu(g, negative_slope=0.1)
            xs = gen.noise_res[i](gen.noise_convs[i](har), sty)
            g = gen.ups[i](g)
            if i == gen.num_upsamples - 1:
                g = gen.reflection_pad(g)
            g = g + xs
            acc = None
            for j in range(gen.num_kernels):
                r = gen.resblocks[i * gen.num_kernels + j](g, sty)
                acc = r if acc is None else acc + r
            g = acc / gen.num_kernels
        g = gen.conv_post(F.leaky_relu(g))
        half = gen.post_n_fft // 2 + 1
        mag = torch.exp(g[0, :half, ::FRAME_STEP])
        phase = torch.sin(g[0, half:, ::FRAME_STEP])
        out["generator"] = {
            "frames": int(g.shape[-1]),
            "frame_step": FRAME_STEP,
            "mag": [floats(row) for row in mag],
            "phase": [floats(row) for row in phase],
        }

        sg = SineGen(24000, 300, 8, sine_amp=0.1, noise_std=0.0, voiced_threshold=10)
        f0s = torch.from_numpy(sine_f0(SINE_LEN)).reshape(1, SINE_LEN, 1)
        sines, _, _ = sg(f0s)
        out["sine"] = {"len": SINE_LEN, "step": SINE_STEP, "values": [floats(row) for row in sines[0, ::SINE_STEP]]}

    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(out, f)
        f.write("\n")


if __name__ == "__main__":
    main()
