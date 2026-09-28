#!/usr/bin/env python3
"""Capture upstream Silero speech timestamps as ground truth for the Rust segmenter.

Runs the reference `get_speech_timestamps` from silero-vad's own `utils_vad.py`
over the same 60 s audio stored in `silero_vad_fixture.safetensors`, under
several parameter sets, and records the resulting sample-accurate segment
boundaries.

The parameter sets deliberately include a finite `max_speech_duration_s`: with
the default of infinity the whole max-speech branch of the algorithm — the one
with `possible_ends`, `prev_end` and `next_start` — never executes, so a port
that omits it entirely still matches. Long tutorial recordings are exactly the
case that needs it.

`utils_vad.py` is NOT vendored. Fetch it next to this script first:
    curl -sSLO https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/utils_vad.py

Usage:
    python make_vad_timestamps_fixture.py <models-dir>/silero-vad <utils_vad-dir>
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

RATE = 16_000

# Each entry becomes one case in the fixture. Names are the keys the Rust test
# looks up, so do not rename them without updating the test.
CASES = {
    "defaults": {},
    # The longest natural segment in this audio is 4.16 s, so the cap must sit
    # BELOW that to make the max-speech branch actually run. A cap above the
    # longest segment reproduces the defaults exactly and tests nothing.
    "max_speech_2s": {"max_speech_duration_s": 2.0},
    "max_speech_2s_legacy": {
        "max_speech_duration_s": 2.0,
        "use_max_poss_sil_at_max_speech": False,
    },
    "strict": {
        "threshold": 0.7,
        "min_speech_duration_ms": 500,
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 100,
    },
    "permissive": {
        "threshold": 0.3,
        "min_speech_duration_ms": 100,
        "min_silence_duration_ms": 50,
        "speech_pad_ms": 0,
    },
}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    root, utils_dir = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(utils_dir))
    # `utils_vad` imports torchaudio at module scope but only uses it to read
    # audio files, which this script does not do. Stub it so a torchaudio built
    # against a different torch ABI cannot break the import.
    sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
    from utils_vad import OnnxWrapper, get_speech_timestamps

    audio = load_file(root / "silero_vad_fixture.safetensors")["input"].astype(np.float32)
    tensor = torch.from_numpy(audio)

    model = OnnxWrapper(str(root / "silero_vad.onnx"), force_onnx_cpu=True)

    out = {"sample_rate": RATE, "num_samples": int(len(audio)), "cases": {}}
    for name, params in CASES.items():
        model.reset_states()
        stamps = get_speech_timestamps(tensor, model, sampling_rate=RATE, **params)
        out["cases"][name] = {
            "params": params,
            "segments": [[int(s["start"]), int(s["end"])] for s in stamps],
        }
        total = sum(s["end"] - s["start"] for s in stamps)
        print(f"{name:14s} {len(stamps):3d} segments, {total / RATE:6.2f}s speech")

    path = root / "silero_vad_timestamps.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
