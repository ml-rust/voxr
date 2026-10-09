# voxr

Audio and speech on top of [boostr](https://crates.io/crates/boostr) models and [numr](https://crates.io/crates/numr) tensors.

boostr holds the model networks and their tensor kernels. voxr holds everything around them that works on samples, files, text and voices.

## What it holds

Always built:

- `wav`: RIFF/WAVE read and write, and stream headers of unknown length.
- `resample`: a general polyphase resampler, plus resamplers that match ffmpeg's default output, SciPy's `resample_poly` and librosa's default output.
- `stretch`: a WSOLA tempo engine that matches ffmpeg's `atempo` output.
- `enhance`: the reference-take chain of denoise, filters, loudness and limiter.
- `metrics` and `sound_review`: single-sound measures and the sound review of the motion-video skill (<https://github.com/farhan-syah/motion-video-skill>).
- `pitch`, `quality`, `spectral`: pitch tracking, take quality and two-signal spectral distance.
- `eval`: word and character error rates for ASR.
- `tts_eval`: the signal and summary halves of the TTS quality gate.

Behind features:

| Feature | Adds |
| --- | --- |
| `decode` | Compressed-audio decode through symphonia, with ffmpeg-matched trims and downmix |
| `spacy-tokenizer` | The spaCy `en_core_web_sm` 3.8.0 tokenizer |
| `spacy-tagger` | The `en_core_web_sm` tagger |
| `misaki` | misaki English G2P with embedded US and GB lexicons |
| `tts` | `TtsEngine`, `TtsBundle` and `Voice` |
| `kokoro` | Kokoro synthesis |
| `vits` | VITS (MMS-TTS) synthesis |
| `voxcpm` | VoxCPM2 voice cloning |
| `whisper` | Whisper transcription with word timings |
| `vad` | Silero VAD segmentation |
| `clap` | CLAP audio-text scoring |
| `neucodec` | The NeuCodec model gate |
| `corpus` | Speech-corpus preparation |
| `f16` | Half-precision tensors |
| `cuda` | The CUDA backend. Needs CUDA 12.8 or newer, or 13.x |

The repository README lists what each feature implies.

## Models

No weights are compiled in. Each engine loads its checkpoint at run time from a directory, and voxr fetches nothing over the network.

## License

Apache-2.0. Ported code and redistributed data are listed with their licences in `LICENSE-OTHERS`.
