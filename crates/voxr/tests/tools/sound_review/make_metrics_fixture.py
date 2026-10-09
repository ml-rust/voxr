"""Fixture for `sound_review_metrics_oracle.rs`: the motion-video skill's
sound-review metrics, run by the skill's own script on generated clips and
on speech recordings.

Writes into OUT:
  clips/<name>.wav   every clip, as the review reads it
  metrics.json       per clip: the unrounded and the rounded measures, and the
                     float32 model clip (mono, 48 kHz, padded) for short clips

The unrounded measures come from the script's own `measure()` with Python's
`round` replaced by the identity, so every line of the script runs as-is.

Usage:
  python make_metrics_fixture.py --script PATH/sound-review.py \
      --speech-dir PATH/Kokoro-82M/samples --out ~/.cache/voxr-test/sound-review-metrics
"""

import argparse
import builtins
import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_script(path):
    spec = importlib.util.spec_from_file_location("sound_review", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tone(freq, secs, sr, amp=0.5, phase=0.0):
    t = np.arange(int(secs * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


def clips(rng):
    """(name, samples [frames, channels] float64, rate, subtype)."""
    out = []
    sr = 48000
    s = tone(440, 1.0, sr)
    out.append(("sine440_stereo", np.stack([s, s], 1), sr, "PCM_24"))
    out.append(("sine9k", tone(9000, 0.5, sr, 0.3)[:, None], sr, "PCM_24"))
    out.append(("noise", (rng.uniform(-0.3, 0.3, int(0.3 * sr)))[:, None], sr, "PCM_24"))
    burst = np.zeros(sr)
    n = int(0.03 * sr)
    burst[20000:20000 + n] = rng.uniform(-0.5, 0.5, n) * np.exp(-np.arange(n) / (0.005 * sr))
    out.append(("click_plateau", np.stack([burst, 0.5 * burst], 1), sr, "PCM_24"))
    loud = np.zeros(sr)
    loud[30000:30000 + 2400] = np.sign(np.sin(np.arange(2400) * 0.07)) * 0.9
    out.append(("loud_plateau", loud[:, None], sr, "PCM_24"))
    t = np.arange(2 * sr) / sr
    chirp = 0.4 * np.sin(2 * np.pi * (100 * t + (16000 - 100) / (2 * 2.0) * t * t))
    out.append(("chirp", chirp[:, None], sr, "PCM_24"))
    out.append(("short40ms", tone(1200, 0.04, sr)[:, None], sr, "PCM_24"))
    out.append(("short80ms", (tone(300, 0.08, sr) * np.linspace(1, 0, int(0.08 * sr)))[:, None], sr, "PCM_24"))
    out.append(("silence", np.zeros((sr // 2, 2)), sr, "PCM_24"))
    out.append(("hot11k", tone(11000, 0.4, sr, 0.999, 0.3)[:, None], sr, "PCM_24"))
    a = tone(523.25, 0.7, 44100, 0.4) + rng.normal(0, 0.01, int(0.7 * 44100))
    b = tone(659.25, 0.7, 44100, 0.3)
    out.append(("mix44k", np.stack([a, b], 1), 44100, "PCM_24"))
    out.append(("tone22k_pcm16", tone(2000, 0.6, 22050, 0.6)[:, None], 22050, "PCM_16"))
    f = tone(3000, 0.25, sr, 0.7) * np.exp(-np.arange(int(0.25 * sr)) / 3000)
    out.append(("decay_float", np.stack([f, -f], 1), sr, "FLOAT"))
    out.append(("tone8k_pcm16", tone(700, 0.3, 8000, 0.5)[:, None], 8000, "PCM_16"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--speech-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mod = load_script(args.script)
    out = Path(args.out).expanduser()
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    cases = []
    for name, x, sr, subtype in clips(rng):
        path = out / "clips" / f"{name}.wav"
        sf.write(path, x, sr, subtype=subtype)
        cases.append((name, path))
    for p in sorted(Path(args.speech_dir).glob("*.wav")):
        cases.append((f"speech_{p.stem}", p))

    records = []
    for name, path in cases:
        rounded = mod.measure(path)
        mod.round = lambda v, n=None: v
        raw = mod.measure(path)
        mod.round = builtins.round
        rec = {
            "name": name,
            "file": str(path),
            "raw": {k: float(v) for k, v in raw.items()},
            "rounded": {k: float(v) for k, v in rounded.items()},
        }
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        if len(x) <= sr:
            a = resample_poly(x.mean(axis=1), 48000, sr).astype(np.float32)
            a = np.pad(a, (0, max(0, 48000 - len(a))))
            rec["model_clip"] = [float(v) for v in a]
        records.append(rec)
    (out / "metrics.json").write_text(json.dumps({"cases": records}))
    print(f"{len(records)} cases in {out}")


if __name__ == "__main__":
    main()
