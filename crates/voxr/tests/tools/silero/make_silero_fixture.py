#!/usr/bin/env python3
"""Generate a golden parity fixture for the Rust Silero VAD port.

Runs the upstream ONNX model over a fixed slice of real speech and stores BOTH
the exact input samples and the resulting per-chunk speech probabilities. The
Rust test reads the same samples, so the comparison never depends on matching
Python's decoder or resampler.

The model input is 576 samples, NOT 512: a 64-sample context carried from the
previous chunk, concatenated in front of the current 512-sample chunk. Feeding a
bare 512 returns a near-zero probability for every chunk instead of erroring, so
a port that gets this wrong reports continuous silence and looks merely
conservative.

Usage:
    python make_silero_fixture.py <audio-file> <models-dir>/silero-vad [offset-seconds]

Writes `silero_vad_fixture.safetensors` beside the ONNX model.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from math import gcd
from safetensors.numpy import save_file
from scipy.signal import resample_poly

RATE = 16_000
CHUNK = 512  # Silero's fixed 16 kHz window
CONTEXT = 64  # carried from the previous chunk, prepended to every input
SECONDS = 60.0


def load_mono_16k(path):
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != RATE:
        # Band-limited polyphase, matching what boostr's own resampler does.
        # The fixture stores the samples it actually fed the model, so Rust
        # never has to reproduce this step bit-for-bit.
        g = gcd(rate, RATE)
        mono = resample_poly(mono, RATE // g, rate // g).astype(np.float32)
    return mono


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    audio_path, root = Path(sys.argv[1]), Path(sys.argv[2])
    offset = float(sys.argv[3]) if len(sys.argv) == 4 else 0.0

    samples = load_mono_16k(audio_path)
    start = (int(offset * RATE) // CHUNK) * CHUNK
    n = (int(SECONDS * RATE) // CHUNK) * CHUNK
    if len(samples) < start + n:
        sys.exit(f"{audio_path} is shorter than {offset + SECONDS}s at {RATE} Hz")
    samples = np.ascontiguousarray(samples[start : start + n])

    session = ort.InferenceSession(
        str(root / "silero_vad.onnx"), providers=["CPUExecutionProvider"]
    )
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT, dtype=np.float32)
    probs = []
    for pos in range(0, n, CHUNK):
        chunk = samples[pos : pos + CHUNK]
        window = np.concatenate([context, chunk]).reshape(1, CONTEXT + CHUNK)
        out, state = session.run(
            None,
            {
                "input": window,
                "state": state,
                "sr": np.array(RATE, dtype=np.int64),
            },
        )
        context = chunk[-CONTEXT:]
        probs.append(float(out[0, 0]))

    probs = np.asarray(probs, dtype=np.float32)
    path = root / "silero_vad_fixture.safetensors"
    save_file(
        {"input": samples, "prob": probs, "state_final": np.ascontiguousarray(state)},
        path,
        metadata={
            "sample_rate": str(RATE),
            "chunk": str(CHUNK),
            "context": str(CONTEXT),
            "offset_seconds": str(offset),
            "source": audio_path.name,
        },
    )
    voiced = float((probs > 0.5).mean())
    print(f"{path.name}: {len(samples)} samples, {len(probs)} chunks")
    print(f"    prob min {probs.min():.6f} max {probs.max():.6f} voiced {voiced:.3f}")


if __name__ == "__main__":
    main()
