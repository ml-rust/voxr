# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each release must have a non-empty section here before it can be tagged — `.github/workflows/release-validate.yml` refuses a tag whose version has no entry, and the GitHub Release body is this file's section for that version. voxr and voxr-narrate release together at one version.

## [0.3.0] - 2026-10-11

### Added

#### voxr-narrate

- `AnchorStep::ShortOpening`: a voice designed from an opening of fewer than `OPENING_WORDS` words says that its reference is short.
- `Doubtful` and `doubtful_parts`: the parts a speech check marks `ok ?`, counted by kind (doubtful word, stray pause, pitch peak, click).

### Changed

#### voxr-narrate

- `verdict_lines` takes the `Doubtful` count beside the problems. The summary names the parts marked `ok ?`, with their part numbers and kinds. The exit code stays 0.
- `AnchorStep` has a new variant, so a `match` over it needs an arm for `ShortOpening`.
- A VoxCPM2 stray pause advises `--reroll` for its part. Each stray pause adds `BAD_STRAY_PAUSE` to a take's badness, so retries and re-rolls keep the take with fewer.

## [0.2.0] - 2026-10-10

### Changed

- **Dependencies** — voxr and voxr-narrate require boostr 0.5 and numr 0.10. The `Tensor` and runtime types they pass through come from these versions, so a caller that hands voxr a tensor must use the same ones.

## [0.1.1] - 2026-10-10

### Fixed

#### voxr-narrate

- Compiling the narration patterns no longer overflows a 1 MiB stack, the size of a Windows main thread. A debug build crashed on the first script it read.

## [0.1.0] - 2026-10-09

First release of `voxr` and `voxr-narrate`.

### Added

#### voxr

- RIFF/WAVE read and write, including stream headers of unknown length.
- Compressed-audio decode behind `decode`: FLAC, MP3, AAC-LC and ALAC in MP4/M4A/MOV, and Opus and Vorbis in Ogg/WebM/MKV, with ffmpeg-matched trims and downmix. Multichannel AAC, HE-AAC and Opus past two channels are refused.
- Resamplers: a general polyphase resampler, and resamplers that match ffmpeg's default output, SciPy's `resample_poly` and librosa's default output.
- A WSOLA tempo engine that matches ffmpeg's `atempo` output.
- The reference-take enhancement chain: denoise, filters, loudness normalization and a limiter.
- Pitch tracking, take quality, spectral distance, and word and character error rates.
- Single-sound metrics and the sound review of the motion-video skill, with CLAP ranking behind `clap`.
- English G2P: the misaki lexicon pipeline behind `misaki`, the spaCy `en_core_web_sm` tokenizer behind `spacy-tokenizer`, its tagger behind `spacy-tagger`, and a neural out-of-vocabulary fallback.
- `TtsEngine`, `TtsBundle` and `Voice` behind `tts`.
- Kokoro synthesis behind `kokoro`, loaded from a model directory.
- VITS (MMS-TTS) synthesis behind `vits`, with uroman romanization for `is_uroman` checkpoints.
- VoxCPM2 voice cloning behind `voxcpm`: reusable reference voices, continuation prompts, per-request guidance and steps, bad-take retry and batched requests.
- Whisper transcription with word timings behind `whisper`.
- Silero VAD segmentation behind `vad`.
- Speech-corpus preparation behind `corpus`.
- The TTS quality gate: signal statistics, pace, intelligibility and timbre.

#### voxr-narrate

- Narration script, timed script and subtitle parsing, tags, and numbers read back from words and digits.
- The speech check of a take against its script, and glitch detection and repair.
- The voice polish chain, and reference-voice preparation for cloning.
- The take loop: cache keys, seeds, take records, the pick of each part's best take, speaking pace and its verdict, anchor voices, timed placement and joins.
- A whole VoxCPM2 run from script to voiceover, and a per-phrase run for every other engine.
- Engine adapters for Kokoro, VITS and VoxCPM2, and Whisper as the hearing, behind features.
- Time-stretch of the spoken track to a pace behind `stretch`.
- The speak and transcribe commands of the motion-video skill: argument parsing, every printed line, file and exit code, and the video manifest's narration.
