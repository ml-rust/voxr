#!/usr/bin/env python3
"""Does the base model's `Audio quality:` description actually control output noise?

The question this settles: a home corpus can never be noise-free, so the noise
has to be handled either by DENOISING the training waveforms or by LABELLING it
and letting the model condition on it. This checkpoint documents an
`Audio quality:` field that names background noise, which is the second
mechanism. If that field genuinely moves the noise floor of generated audio,
labelling wins and no denoiser is needed. If it does not, the field is
decorative and the corpus must be denoised.

Costs no recording and no training. Generates audio TOKEN IDS only; decoding to
a waveform and measuring the floor is boostr's job (`examples/noise_probe_decode.rs`),
so the measurement uses the same `measure_quality` the corpus pipeline uses and
no `neucodec` Python package is needed.

    EXPRESSIVE_TTS_MODEL=CHECKPOINT_DIR python3 noise_conditioning_probe.py --out probe.json
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Required via EXPRESSIVE_TTS_MODEL, or --model.
MODEL = os.environ.get("EXPRESSIVE_TTS_MODEL")

# One speaker, one text, one description. Exactly ONE line differs between the
# two described arms, so any difference in the decoded noise floor is
# attributable to that line and nothing else.
SPEAKER = "genshin-voice_audio_Rahman"

TEXT = (
    "Okay, so sebelum kita mula, saya nak terangkan sikit pasal setup ni. "
    "The first thing you need to do is check your environment. "
    "Kalau semua okay, kita boleh teruskan ke bahagian seterusnya."
)

DESCRIPTION_TEMPLATE = """
Vocal qualities: Medium pitch, clear and steady, with a warm and approachable demeanor.
Speaking style: Conversational and explanatory, suitable for instructional content.
Pace: Moderate and measured, with natural pauses between clauses.
Fluency: Consistently fluent throughout, with no stutters.
Acoustic environment: The recording has a confined and enclosed sound, suggesting a small room.
Audio quality: {quality}
Content style: Educational or instructional.
""".strip()

CLEAN = ("The audio is pristine and studio-clean, with absolutely no background "
         "noise, hiss, or ambient sound of any kind.")
NOISY = ("The audio is noticeably noisy, with persistent background hiss and "
         "constant room tone audible throughout the entire recording.")

# A larger seed count gives Welch's t-test enough power to separate the arms.
SEEDS = list(range(1, 21))


def build_prompt(quality):
    """`None` quality means the no-description arm — the speaker's own default."""
    if quality is None:
        return f"<|im_start|>{SPEAKER}: {TEXT}<|speech_start|>"
    description = DESCRIPTION_TEMPLATE.format(quality=quality)
    return f"<|im_start|>{SPEAKER}: {TEXT}<|description|>{description}<|speech_start|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="where to write the token JSON")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("needs --model CHECKPOINT_DIR or EXPRESSIVE_TTS_MODEL set")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    # forced_decoder_ids on a generation config silently overrides sampling and
    # truncates output; this checkpoint is not Whisper but the check is cheap.
    gc = model.generation_config
    if getattr(gc, "forced_decoder_ids", None):
        print("  clearing forced_decoder_ids", file=sys.stderr)
        gc.forced_decoder_ids = None

    arms = [("no_description", None), ("clean", CLEAN), ("noisy", NOISY)]
    results = []

    for arm, quality in arms:
        prompt = build_prompt(quality)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        for seed in SEEDS:
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.15,
                )
            new = out[0][prompt_len:]
            text = tok.decode(new, skip_special_tokens=False)
            codes = [int(m) for m in re.findall(r"<\|s_(\d+)\|>", text)]
            # Ran out of budget rather than emitting <|im_end|> means the take is
            # truncated mid-utterance; its trailing frames are not comparable.
            complete = "<|im_end|>" in text
            results.append({
                "arm": arm,
                "seed": seed,
                "codes": codes,
                "complete": complete,
                "n_codes": len(codes),
            })
            print(f"  {arm:15s} seed {seed:3d}  {len(codes):5d} codes  "
                  f"{'complete' if complete else 'TRUNCATED'}", file=sys.stderr)

    Path(args.out).write_text(json.dumps({
        "model": args.model,
        "speaker": SPEAKER,
        "text": TEXT,
        "clean_quality": CLEAN,
        "noisy_quality": NOISY,
        "sample_rate": 24000,
        "results": results,
    }))
    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
