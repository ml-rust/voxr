#!/usr/bin/env python3
"""Generate log-mel ground truth for boostr's Whisper preprocessing.

Stores the exact 16 kHz mono samples fed to HuggingFace's own
`WhisperFeatureExtractor` alongside the log-mel it produced, so the Rust port is
compared against the reference implementation rather than against a
reimplementation of it.

Whisper's preprocessing is NOT generic mel: it is `n_fft=400`, `hop=160`,
`center=True` (reflect), the SLANEY mel scale with Slaney-normalized filters,
`log10` compression, a dynamic-range clamp to `max - 8.0`, and a final
`(x + 4) / 4` affine. Every one of those is load-bearing.

Usage:
    python make_whisper_mel_fixture.py <audio-file> <models-dir> [offset-seconds]

Writes `whisper_mel_fixture.safetensors` into `<models-dir>`, holding the
80-bin (tiny/base/small/medium) and 128-bin (large-v3) mels for the same audio.
"""

import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from safetensors.numpy import save_file
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor

RATE = 16_000
SECONDS = 30.0  # Whisper's fixed window; the extractor pads or trims to exactly this


def load_mono_16k(path):
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != RATE:
        g = gcd(rate, RATE)
        mono = resample_poly(mono, RATE // g, rate // g).astype(np.float32)
    return mono


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    audio_path, root = Path(sys.argv[1]), Path(sys.argv[2])
    offset = float(sys.argv[3]) if len(sys.argv) == 4 else 0.0

    samples = load_mono_16k(audio_path)
    start = int(offset * RATE)
    n = int(SECONDS * RATE)
    if len(samples) < start + n:
        sys.exit(f"{audio_path} is shorter than {offset + SECONDS}s at {RATE} Hz")
    samples = np.ascontiguousarray(samples[start : start + n])

    tensors = {"input": samples}
    for bins in (80, 128):
        fe = WhisperFeatureExtractor(feature_size=bins, sampling_rate=RATE)
        mel = fe(samples, sampling_rate=RATE, return_tensors="np").input_features[0]
        tensors[f"mel_{bins}"] = np.ascontiguousarray(mel.astype(np.float32))
        print(f"mel_{bins}: {mel.shape} min {mel.min():.6f} max {mel.max():.6f}")

    # A short slice too: it pins the reflect-padding at the START of the signal,
    # which a port that frames from sample 0 gets wrong only in the first frames.
    short = np.ascontiguousarray(samples[: RATE // 2])
    fe = WhisperFeatureExtractor(feature_size=80, sampling_rate=RATE)
    tensors["input_short"] = short
    tensors["mel_80_short"] = np.ascontiguousarray(
        fe(short, sampling_rate=RATE, return_tensors="np")
        .input_features[0]
        .astype(np.float32)
    )

    path = root / "whisper_mel_fixture.safetensors"
    save_file(
        tensors,
        path,
        metadata={
            "sample_rate": str(RATE),
            "n_fft": "400",
            "hop_length": "160",
            "source": audio_path.name,
            "offset_seconds": str(offset),
        },
    )
    print(f"{path.name}: {', '.join(sorted(tensors))}")


if __name__ == "__main__":
    main()
