#!/usr/bin/env python3
"""VoxCPM2 zero-shot cloning from a recorded reference voice.

Uses `reference_wav_path`, NOT `prompt_wav_path`.

`prompt_wav_path` is CONTINUATION mode: it concatenates prompt_text with the
target text and asks the model to carry on speaking, so it keeps going past
the end of the target sentence. `reference_wav_path` isolates the voice
through ref_audio tokens with no continuation semantics, and does not have
this failure mode.

Do NOT reach for `retry_badcase_ratio_threshold` to fix over-long output. In
continuation mode `target_text_length` excludes the prompt text, so the cap is
computed against the wrong length: tightening it from 6.0 to 3.0 made the
babbling worse, not better.

    VENV/bin/python voxcpm_clone.py --prompt-wav W --prompt-text T --out-dir DIR
"""

import argparse
import sys
from pathlib import Path

TEXTS = {
    "standard": (
        "Saya duduk Shah Alam, kerja kat KL. "
        "Petang tadi hujan lebat, jalan depan rumah banjir sikit."
    ),
    "kelantan": "Kawe nak gi kedai meta, demo nak kirim gapo-gapo dok?",
    "codeswitch": "Saya dah cuba macam-macam, tapi honestly it doesn't make sense.",
}


def median_f0(wav, rate, fmin=70.0, fmax=320.0):
    """Median F0 over voiced frames, by autocorrelation. Hz, or None if unvoiced.

    Used to score candidates against the reference. Absolute accuracy matters
    less than consistency: every candidate is measured the same way, so the
    ranking holds even if each estimate is a hertz or two off.
    """
    import numpy as np
    a = np.asarray(wav, dtype=np.float64)
    if a.ndim > 1:
        a = a.mean(axis=1)
    win = int(0.04 * rate)
    hop = int(0.01 * rate)
    lo, hi = int(rate / fmax), int(rate / fmin)
    if len(a) < win or hi <= lo:
        return None
    out = []
    for start in range(0, len(a) - win, hop):
        fr = a[start:start + win]
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, mode="full")[win - 1:]
        if ac[0] <= 1e-9:
            continue
        seg = ac[lo:hi]
        if seg.size == 0:
            continue
        k = int(np.argmax(seg))
        # Unvoiced frames have no dominant period; 0.3 of zero-lag energy is
        # the usual cut and keeps breathy voiced frames without admitting hiss.
        if seg[k] < 0.3 * ac[0]:
            continue
        out.append(rate / (lo + k))
    if not out:
        return None
    return float(np.median(out))


def load_wav_mono(path):
    """(samples, rate) as float32 mono, without pulling in a decoder library."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    a = a.astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-wav", required=True,
                    help="reference audio carrying the voice to clone")
    ap.add_argument("--prompt-text", required=True,
                    help="transcript of --prompt-wav. Unused in reference mode, "
                         "but required by --continuation.")
    ap.add_argument("--best-of", type=int, default=4, metavar="N",
                    help="generate N candidates per line and keep the one whose "
                         "median F0 is closest to the reference (default 4). "
                         "Pitch varies between draws, so N > 1 picks the closest "
                         "match instead of the first draw. Set 1 to take the "
                         "first draw.")
    ap.add_argument("--continuation", action="store_true",
                    help="use prompt_wav_path continuation mode instead. It "
                         "speaks past the end of the target text.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="openbmb/VoxCPM2")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    from voxcpm import VoxCPM

    print(f"loading {args.model} ...", file=sys.stderr)
    model = VoxCPM.from_pretrained(args.model)
    rate = model.tts_model.sample_rate

    mode = (
        dict(prompt_wav_path=args.prompt_wav, prompt_text=args.prompt_text)
        if args.continuation
        else dict(reference_wav_path=args.prompt_wav)
    )
    print(f"mode: {'continuation' if args.continuation else 'reference'}", file=sys.stderr)

    ref_f0 = None
    if args.best_of > 1:
        ref_samples, ref_rate = load_wav_mono(args.prompt_wav)
        ref_f0 = median_f0(ref_samples, ref_rate)
        if ref_f0 is None:
            print("reference has no measurable pitch; taking the first draw",
                  file=sys.stderr)
        else:
            print(f"reference F0: {ref_f0:.1f} Hz", file=sys.stderr)

    for label, text in TEXTS.items():
        best, best_err, tried = None, None, []
        for _ in range(max(1, args.best_of)):
            wav = model.generate(
                text=text,
                cfg_value=2.0,
                inference_timesteps=32,
                **mode,
            )
            if ref_f0 is None:
                best = wav
                break
            f0 = median_f0(wav, rate)
            # A candidate with no measurable pitch is unusable, not merely a
            # poor match, so it never wins regardless of what else is on offer.
            err = abs(f0 - ref_f0) if f0 is not None else float("inf")
            tried.append(f0)
            if best_err is None or err < best_err:
                best, best_err = wav, err

        path = out / f"clone_{label}.wav"
        sf.write(path, best, rate)
        detail = ""
        if ref_f0 is not None and tried:
            shown = ", ".join("-" if f is None else f"{f:.0f}" for f in tried)
            detail = f"  F0 [{shown}] -> off by {best_err:.0f} Hz"
        print(f"  {label:11s} {len(best) / rate:5.1f}s -> {path}{detail}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
