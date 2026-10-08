#!/usr/bin/env python3
"""Write the ground truth for `tests/resample_librosa_parity.rs`.

For each rate pair and test signal the fixture holds the `float32` input,
python-soxr's `soxr.resample(x, from, to, 'soxr_hq')` and librosa's
`librosa.resample(x, orig_sr=from, target_sr=to)`. An optional speech WAV
adds a case through `librosa.load(path, sr=16000, mono=True)`.

Signals are deterministic: a log chirp under noise, an impulse train, short
random clips (shorter than every filter), and a long chirp of 32 s on some
pairs.

Usage:
    python -I make_soxr_fixture.py --out DIR [--speech WAV]

`--out` defaults to `$VOXR_SOXR_FIXTURE_DIR` and `--speech` to
`$VOXR_SOXR_SPEECH_WAV`. The script writes `DIR/resample_soxr.safetensors`.
Its metadata key `cases` is a JSON list of `{"name", "from", "to"}`. Each
case `name` has tensors `name/input`, `name/soxr` and `name/librosa`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import soxr
from safetensors.numpy import save_file

TARGET = 16_000

# The pairs the VoxCPM2 reference path meets, then pairs that reach the
# planner's other branches: filter-only pre stage with exact and
# interpolating polyphase stages, half-band stages, post DFT stages and
# frequency-domain decimation.
PAIRS = [
    (48_000, 16_000),
    (24_000, 16_000),
    (44_100, 16_000),
    (22_050, 16_000),
    (32_000, 16_000),
    (96_000, 16_000),
    (8_000, 16_000),
    (16_000, 16_000),
    (11_025, 16_000),
    (16_000, 48_000),
    (44_100, 48_000),
    (48_000, 44_100),
    (16_000, 44_100),
    (12_345, 16_000),
    (44_101, 16_000),
    (16_000, 16_001),
    (16_001, 16_000),
    (30_001, 16_000),
    (192_000, 16_000),
    (8_000, 48_000),
    (4_000, 96_000),
]

# Pairs that also get the 32 s signal.
LONG_PAIRS = {(48_000, 16_000), (44_100, 16_000), (24_000, 16_000), (16_000, 48_000)}


def chirp(rate: int, seconds: float, seed: int) -> np.ndarray:
    """A log sweep from 20 Hz to 0.95 of Nyquist at 0.5, plus noise at 0.05."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    f0, f1 = 20.0, 0.95 * rate / 2
    k = np.log(f1 / f0) / seconds
    phase = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    noise = np.random.default_rng(seed).standard_normal(n) * 0.05
    return (0.5 * np.sin(phase) + noise).astype(np.float32)


def impulses(rate: int) -> np.ndarray:
    """One second of impulses every 997 samples, with one at each end."""
    x = np.zeros(rate, dtype=np.float32)
    x[::997] = 0.9
    x[-1] = -0.9
    return x


def noise(n: int, seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).uniform(-0.8, 0.8, n)).astype(np.float32)


def signals(rate: int, long: bool) -> dict:
    out = {
        "chirp": chirp(rate, 2.0, rate),
        "impulses": impulses(rate),
        "short64": noise(64, rate + 1),
        "short7": noise(7, rate + 2),
        "one": noise(1, rate + 3),
        "empty": np.zeros(0, dtype=np.float32),
    }
    if long:
        out["long"] = chirp(rate, 32.0, rate + 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.environ.get("VOXR_SOXR_FIXTURE_DIR"))
    ap.add_argument("--speech", default=os.environ.get("VOXR_SOXR_SPEECH_WAV"))
    args = ap.parse_args()
    if not args.out:
        print("error: pass --out DIR or set VOXR_SOXR_FIXTURE_DIR", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    cases = []

    def add(name: str, src: int, dst: int, x: np.ndarray, librosa_y: np.ndarray):
        y_soxr = soxr.resample(x, src, dst, "soxr_hq") if src != dst else x
        assert y_soxr.dtype == np.float32 and librosa_y.dtype == np.float32, name
        tensors[f"{name}/input"] = np.ascontiguousarray(x)
        tensors[f"{name}/soxr"] = np.ascontiguousarray(y_soxr)
        tensors[f"{name}/librosa"] = np.ascontiguousarray(librosa_y)
        cases.append({"name": name, "from": src, "to": dst})

    for src, dst in PAIRS:
        for sig, x in signals(src, (src, dst) in LONG_PAIRS).items():
            y = librosa.resample(x, orig_sr=src, target_sr=dst)
            add(f"{src}_{dst}/{sig}", src, dst, x, y)

    if args.speech:
        info = sf.info(args.speech)
        x, rate = sf.read(args.speech, dtype="float32", always_2d=True)
        x = librosa.to_mono(x.T)
        y, sr = librosa.load(args.speech, sr=TARGET, mono=True)
        assert sr == TARGET
        assert np.array_equal(y, librosa.resample(x, orig_sr=rate, target_sr=TARGET))
        add(f"{info.samplerate}_{TARGET}/speech", rate, TARGET, x, y)

    path = out_dir / "resample_soxr.safetensors"
    meta = {
        "cases": json.dumps(cases),
        "librosa": librosa.__version__,
        "soxr": soxr.__version__,
    }
    save_file(tensors, str(path), metadata=meta)
    print(f"wrote {len(cases)} cases to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
