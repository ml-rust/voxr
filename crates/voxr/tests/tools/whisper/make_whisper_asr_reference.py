#!/usr/bin/env python3
"""Capture a reference Whisper transcription for boostr's integration test.

Runs HuggingFace's own `WhisperForConditionalGeneration` with GREEDY decoding
over the exact samples stored in `whisper_mel_fixture.safetensors`, so the Rust
test compares against the reference implementation on identical input rather
than against a transcript typed by hand.

Greedy decoding is deterministic, so the token ids are a legitimate assertion
target. They are still sensitive to float ordering, which is why the fixture
records the full id sequence AND the text: a port that drifts on one token can
be told apart from one that is transcribing something else entirely.

The checkpoint's own `generation_config` carries a deprecated
`forced_decoder_ids`, which silently fights the `language`/`task` flags and
makes generation stop after a single token. It is cleared before generating.

Usage:
    python make_whisper_asr_reference.py <checkpoint-dir> <models-dir> [language]

Writes `<models-dir>/<checkpoint-name>_reference.json`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor

RATE = 16_000


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    ckpt, root = Path(sys.argv[1]), Path(sys.argv[2])
    language = sys.argv[3] if len(sys.argv) == 4 else "ms"

    fixture = load_file(root / "whisper_mel_fixture.safetensors")
    samples = fixture["input"].astype(np.float32)

    processor = WhisperProcessor.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(ckpt, dtype=dtype)
    model = model.to(device).eval()
    # Deprecated and actively harmful: it overrides the language/task flags and
    # truncates generation to one token.
    model.generation_config.forced_decoder_ids = None

    features = processor(
        samples, sampling_rate=RATE, return_tensors="pt"
    ).input_features.to(device, dtype)

    with torch.no_grad():
        # Must be passed by KEYWORD: positionally, `generate` treats the first
        # argument as decoder input ids and stops after one token.
        ids = model.generate(
            input_features=features,
            do_sample=False,
            num_beams=1,
            language=language,
            task="transcribe",
            max_new_tokens=200,
        )[0]

    ids = [int(i) for i in ids]
    text = processor.tokenizer.decode(ids, skip_special_tokens=True)

    # Fixed sub-ranges, so `transcribe_segments` can be pinned to a real
    # reference instead of a "non-empty" smoke check. These MUST stay in sync
    # with the ranges AND the token budget in boostr's
    # `whisper_transcribe_segments_covers_each_range`.
    #
    # 48 rather than 200: whisper-tiny falls into a repetition loop on the
    # second range and would otherwise run to the full budget, costing ~80 s of
    # CPU decode on every gate run for no extra coverage. 48 still truncates
    # mid-loop, so the cap itself stays exercised.
    SEGMENT_MAX_NEW_TOKENS = 48
    segments = []
    for start_s, end_s in ((0, 10), (10, 20)):
        window = np.ascontiguousarray(samples[start_s * RATE : end_s * RATE])
        feats = processor(
            window, sampling_rate=RATE, return_tensors="pt"
        ).input_features.to(device, dtype)
        with torch.no_grad():
            seg_ids = model.generate(
                input_features=feats,
                do_sample=False,
                num_beams=1,
                language=language,
                task="transcribe",
                max_new_tokens=SEGMENT_MAX_NEW_TOKENS,
            )[0]
        seg_ids = [int(i) for i in seg_ids]
        segments.append(
            {
                "start": start_s * RATE,
                "end": end_s * RATE,
                "token_ids": seg_ids,
                "text": processor.tokenizer.decode(seg_ids, skip_special_tokens=True),
            }
        )
        print(f"  [{start_s:2d}s,{end_s:2d}s) {len(seg_ids):3d} tokens: {segments[-1]['text'][:80]}")

    out = {
        "checkpoint": ckpt.name,
        "language": language,
        "segments": segments,
        "sample_rate": RATE,
        "num_samples": int(len(samples)),
        "token_ids": ids,
        "text": text,
    }
    path = root / f"{ckpt.name}_reference.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{path.name}: {len(ids)} tokens")
    print(f"  text: {text[:300]}")


if __name__ == "__main__":
    main()
