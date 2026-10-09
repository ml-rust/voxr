"""Fixture for `clap_parity.rs`: transformers' ClapProcessor and ClapModel on
the clips and texts the motion-video skill's sound review ranks, plus speech
and generated clips.

Every clip is prepared as the review prepares it: the float32 channel mean,
`resample_poly` to 48 kHz, zeros to one second. A clip over the model's
10 s span is cut at offset 0: `np.random.randint` is pinned to its low bound
while the processor runs, so the cut repeats.

Writes into OUT:
  clips/<name>.wav             the generated clips
  clap_fixture.json            clip paths, texts, token ids, and the review's
                               ranks of the catalog's base variants
  clap_fixture.safetensors     input_features [clips, 1, 1001, 64],
                               audio_embeds [clips, 512], text_embeds
                               [texts, 512], scores [clips, texts]

Usage:
  python make_clap_fixture.py --model PATH_OR_ID --catalog DIR_WITH_SOUNDS_JSON \
      --speech-dir PATH/Kokoro-82M/samples --out ~/.cache/voxr-test/clap [--device cpu]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.numpy import save_file
from scipy.signal import resample_poly
from transformers import ClapModel, ClapProcessor


def model_clip(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    a = resample_poly(x.mean(axis=1), 48000, sr).astype(np.float32)
    return np.pad(a, (0, max(0, 48000 - len(a))))


def generated(out):
    sr = 48000
    rng = np.random.default_rng(11)
    t = lambda s: np.arange(int(s * sr)) / sr
    clips = {
        "sine440": 0.5 * np.sin(2 * np.pi * 440 * t(1.0)),
        "noise": rng.uniform(-0.3, 0.3, 2 * sr),
        "chirp": 0.4 * np.sin(2 * np.pi * (200 * t(3.0) + 1500 * t(3.0) ** 2)),
        "long_tone": 0.3 * np.sin(2 * np.pi * 330 * t(12.5)),
        "click": np.exp(-t(0.05) * 400) * rng.uniform(-0.8, 0.8, int(0.05 * sr)),
    }
    (out / "clips").mkdir(parents=True, exist_ok=True)
    paths = []
    for name, x in clips.items():
        p = out / "clips" / f"{name}.wav"
        sf.write(p, x, sr, subtype="PCM_24")
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--speech-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    catalog = Path(args.catalog).expanduser()
    index = json.loads((catalog / "sounds.json").read_text())
    base = [label for label in index if label.endswith("_v0")]
    paths = [catalog / f"{b}.wav" for b in base]
    paths += sorted(Path(args.speech_dir).expanduser().glob("*.wav"))
    paths += generated(out)
    texts = sorted({index[b]["says"] for b in base})
    texts += ["a person speaking", "a dog barking", "silence", "a pure sine tone"]

    model = ClapModel.from_pretrained(args.model).to(args.device).eval()
    proc = ClapProcessor.from_pretrained(args.model)
    clips = [model_clip(p) for p in paths]
    real_randint = np.random.randint
    np.random.randint = lambda low, high=None, *a, **k: 0 if high is not None else low * 0
    try:
        feats = proc(audio=clips, sampling_rate=48000, return_tensors="pt")
    finally:
        np.random.randint = real_randint
    tok = proc(text=texts, return_tensors="pt", padding=True)
    ids = [[int(i) for i, m in zip(row, mask) if m] for row, mask in zip(tok["input_ids"], tok["attention_mask"])]
    with torch.inference_mode():
        te = model.get_text_features(**tok.to(args.device))
        te = torch.nn.functional.normalize(getattr(te, "pooler_output", te), dim=-1)
        ae = model.get_audio_features(**feats.to(args.device))
        ae = torch.nn.functional.normalize(getattr(ae, "pooler_output", ae), dim=-1)
        scores = (ae @ te.T).float().cpu().numpy()

    # The review's ranking: base variants against the says texts only.
    n_base, n_says = len(base), len({index[b]["says"] for b in base})
    s = scores[:n_base, :n_says].copy()
    s = (s - s.mean(axis=0, keepdims=True)) / (s.std(axis=0, keepdims=True) + 1e-9)
    ranks = []
    for i, label in enumerate(base):
        own = texts.index(index[label]["says"])
        order = np.argsort(-s[i])
        ranks.append(
            {"label": label, "own": own, "rank": int(np.where(order == own)[0][0]) + 1, "heard": texts[order[0]]}
        )

    save_file(
        {
            "input_features": feats["input_features"].numpy().astype(np.float32),
            "audio_embeds": ae.float().cpu().numpy(),
            "text_embeds": te.float().cpu().numpy(),
            "scores": scores,
        },
        str(out / "clap_fixture.safetensors"),
    )
    meta = {
        "device": args.device,
        "clips": [str(p) for p in paths],
        "base": base,
        "texts": texts,
        "says_texts": n_says,
        "token_ids": ids,
        "ranks": ranks,
    }
    (out / "clap_fixture.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(paths)} clips, {len(texts)} texts in {out}")


if __name__ == "__main__":
    main()
