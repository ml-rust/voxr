#!/usr/bin/env python3
"""Every `VoxCPM.generate` call the skill's runner, `voxcpm_speak.py`, makes
for a matrix of invocations, printed as JSON for `voxcpm_runner_oracle.rs`.

The runner runs unmodified, as `__main__`, with two modules stubbed ahead of
it on `sys.path`: `voxcpm` (a `VoxCPM` whose `generate` records its keyword
arguments and returns silence) and `torch` (a `manual_seed` that records the
seed). `numpy` and `soundfile` are the real ones: the runner writes its
anchor and takes with them.

Each invocation names the runner's flags and its sentence list. Each
recorded call carries the request it answers: the sentence's text, seed and
`style`, the run's `--style`, `--voice` and `--prompt-text`, and whether an
anchor existed when the call ran. Paths print as `<reference>` (the
`--reference` given) and `<anchor>` (the runner's own `anchor.wav`).

Usage: runner_calls.py RUNNER > voxcpm_runner_calls.json
       RUNNER is the path of voxcpm_speak.py, or set VOXCPM_RUNNER.
"""
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

INVOCATIONS = [
    ("design with a description", ["--voice", "A calm male narrator"],
     [{"index": 0, "text": "The opening words of the script.", "seed": 7}]),
    ("design with a wrapped description, then clones", ["--voice", " (warm, low) ", "--style", "relaxed"],
     [{"index": 0, "text": "The opening words.", "seed": 7},
      {"index": 1, "text": "A second part.", "seed": 8},
      {"index": 2, "text": "A question?", "seed": 9, "style": "(asking)"}]),
    ("design without a description", [],
     [{"index": 0, "text": "The model picks its own voice.", "seed": 11}]),
    ("design without a description, with a style", ["--style", "relaxed"],
     [{"index": 0, "text": "The style waits for the clones.", "seed": 12, "style": "(asking)"},
      {"index": 1, "text": "Now it applies.", "seed": 13}]),
    ("reference spoken again slower", ["--reference", "REF"],
     [{"index": 0, "text": "The opening words.", "seed": 8, "style": "(calm, natural tone, speaking slowly)"}]),
    ("clone with a style", ["--reference", "REF", "--style", "relaxed, explaining", "--cfg", "1.6", "--steps", "16"],
     [{"index": 0, "text": "Plain part.", "seed": 100},
      {"index": 1, "text": "Noted part.", "seed": 101, "style": "(asking)"},
      {"index": 2, "text": "Bare note.", "seed": 102, "style": " asking "},
      {"index": 3, "text": "Empty note.", "seed": 103, "style": ""}]),
    ("clone with a wrapped style", ["--reference", "REF", "--style", "  (calm)  "],
     [{"index": 0, "text": "Calm part.", "seed": 5}]),
    ("clone without a style", ["--reference", "REF"],
     [{"index": 0, "text": "No style at all.", "seed": 6}]),
    ("hifi with a style", ["--reference", "REF", "--prompt-text", "The words the reference says.", "--style", "(relaxed)"],
     [{"index": 0, "text": "Hi-Fi part.", "seed": 1007},
      {"index": 1, "text": "Noted Hi-Fi part.", "seed": 1008, "style": "(asking)"},
      {"index": 2, "text": "Empty-noted Hi-Fi part.", "seed": 1009, "style": ""}]),
    ("hifi without a style", ["--reference", "REF", "--prompt-text", "Exact transcript."],
     [{"index": 0, "text": "Only Hi-Fi.", "seed": 3}]),
]

STUB_VOXCPM = '''
import numpy as np
CALLS = []
class _Tts:
    sample_rate = 48000
class VoxCPM:
    tts_model = _Tts()
    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()
    def generate(self, **kw):
        CALLS.append(kw)
        return np.zeros(480, dtype=np.float32)
'''

STUB_TORCH = '''
SEEDS = []
def manual_seed(seed):
    SEEDS.append(seed)
'''


def flag(argv, name, default=""):
    return argv[argv.index(name) + 1] if name in argv else default


def main():
    runner = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VOXCPM_RUNNER")
    if not runner or not Path(runner).is_file():
        sys.exit(f"runner_calls: {runner!r} is not a file. Pass the path of voxcpm_speak.py or set VOXCPM_RUNNER.")
    out = []
    with tempfile.TemporaryDirectory() as stub_dir:
        (Path(stub_dir) / "voxcpm.py").write_text(STUB_VOXCPM)
        (Path(stub_dir) / "torch.py").write_text(STUB_TORCH)
        sys.path.insert(0, stub_dir)
        import torch
        import voxcpm
        for name, flags, sentences in INVOCATIONS:
            with tempfile.TemporaryDirectory() as work:
                ref = os.path.join(work, "reference.wav")
                Path(ref).write_bytes(b"")
                argv = [ref if f == "REF" else f for f in flags]
                listing = os.path.join(work, "sentences.json")
                Path(listing).write_text(json.dumps(sentences))
                voxcpm.CALLS.clear()
                torch.SEEDS.clear()
                anchor = os.path.join(work, "anchor.wav")
                had_anchor = []
                generate = voxcpm.VoxCPM.generate

                def watched(self, **kw):
                    had_anchor.append("REF" in flags or os.path.exists(anchor))
                    return generate(self, **kw)

                voxcpm.VoxCPM.generate = watched
                sys.argv = [runner, "--sentences", listing, "--dir", work, *argv]
                runpy.run_path(runner, run_name="__main__")
                voxcpm.VoxCPM.generate = generate
                show = lambda v: "<reference>" if v == ref else "<anchor>" if v == anchor else v
                calls = []
                for s, kw, seed, anchored in zip(sentences, voxcpm.CALLS, torch.SEEDS, had_anchor):
                    calls.append({
                        "request": {
                            "text": s["text"], "seed": s["seed"], "delivery": s.get("style"),
                            "style": flag(argv, "--style") or None, "voice": flag(argv, "--voice") or None,
                            "prompt_text": flag(argv, "--prompt-text") or None, "anchored": anchored,
                        },
                        "seed": seed,
                        "generate": {k: show(v) for k, v in sorted(kw.items())},
                    })
                out.append({"name": name, "calls": calls})
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
