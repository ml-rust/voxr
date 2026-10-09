# voxr-narrate

Script-to-speech narration over [voxr](https://crates.io/crates/voxr).

voxr-narrate implements the speak and transcribe contract of the motion-video skill (<https://github.com/farhan-syah/motion-video-skill>). Its outputs match the skill's own scripts: printed lines, files and exit codes.

## What it holds

- `text`: narration scripts, timed scripts, subtitles, tags, and numbers read back from words and digits.
- `check`: the check of a spoken take against its script.
- `glitch`: glitch detection and repair in a take.
- `polish`: the voice polish of a take.
- `take`: the take loop, with cache keys, seeds, the pick of each part's best take, pace, anchor voices, placement and joins.
- `transcript`: a spoken take read into timed words.
- `command`: the speak and transcribe commands and the video manifest's narration.
- `wav`: the WAV files the skill writes.

Behind features:

| Feature | Adds |
| --- | --- |
| `whisper` | Whisper word timings, `WhisperHearing`, and `NarrationRuns` over loaded engines |
| `stretch` | `stretch_track` and `resample_48k` over voxr's ffmpeg-matched atempo and resampler |
| `kokoro` | `KokoroSpeech` |
| `vits` | `VitsSpeech` |
| `voxcpm` | `VoxcpmSpeech` |
| `cuda` | The CUDA runtime for Whisper and VoxCPM2. Needs CUDA 12.8 or newer, or 13.x |

## Models

No weights are compiled in. The engines load their checkpoints at run time from directories.

## License

Apache-2.0. The fdlibm notices for the ported math are in `LICENSE-OTHERS`.
