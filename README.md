# voxr

Audio and speech in Rust, on top of [boostr](https://github.com/ml-rust/boostr) models and [numr](https://github.com/ml-rust/numr) tensors.

voxr holds the sample-level side: files, codecs, DSP, G2P, voices, and the TTS, ASR and VAD pipelines. The model networks and their kernels live in boostr. voxr-narrate turns a narration script into a checked voiceover.

## Crates

| Crate | What it holds |
| --- | --- |
| [`voxr`](crates/voxr) | WAV and compressed-audio decode, resamplers, tempo stretch, loudness and enhancement, single-sound metrics, misaki and spaCy G2P, voices, and the Kokoro, VITS, VoxCPM2, Whisper, Silero VAD, CLAP and NeuCodec pipelines |
| [`voxr-narrate`](crates/voxr-narrate) | Script parsing, the speech check of a take against its script, glitch repair, voice polish, the take loop, engine adapters, and the speak and transcribe commands |

voxr-narrate implements the speak and transcribe contract of the motion-video skill (<https://github.com/farhan-syah/motion-video-skill>). voxr implements its sound review.

## Features

Every feature is off by default. A build with none has WAV, the resamplers, tempo stretch, enhancement, metrics, the sound review, pitch, ASR error rates, and the signal half of the TTS quality gate.

### voxr

| Feature | Adds |
| --- | --- |
| `decode` | WAV, FLAC, MP3, AAC-LC and ALAC in MP4/M4A/MOV, and Opus and Vorbis in Ogg/WebM/MKV, with ffmpeg-matched trims and downmix. Multichannel AAC, HE-AAC and Opus past two channels are refused |
| `spacy-tokenizer` | The spaCy `en_core_web_sm` 3.8.0 tokenizer, with its patterns and tables embedded |
| `spacy-tagger` | The `en_core_web_sm` tagger. Implies `spacy-tokenizer` |
| `misaki` | misaki English G2P with the US and GB lexicons embedded. Implies `spacy-tagger` and `f16` |
| `tts` | `TtsEngine`, `TtsBundle` and `Voice`, the engine-agnostic synthesis surface |
| `kokoro` | Kokoro synthesis. Implies `tts` and `misaki` |
| `vits` | VITS (MMS-TTS) synthesis, with uroman romanization. Implies `tts` |
| `voxcpm` | VoxCPM2 voice cloning. Implies `tts` and `decode` |
| `whisper` | Whisper transcription with word timings. Implies `decode` |
| `vad` | Silero VAD segmentation |
| `clap` | CLAP audio-text scoring, which ranks sounds in the sound review |
| `neucodec` | The NeuCodec model gate |
| `corpus` | Speech-corpus preparation. Implies `whisper`, `vad`, `neucodec` and `decode` |
| `f16` | Half-precision tensors |
| `cuda` | The CUDA backend |

### voxr-narrate

| Feature | Adds |
| --- | --- |
| `whisper` | Whisper word timings, `WhisperHearing`, and `NarrationRuns` over loaded engines |
| `stretch` | `stretch_track` and `resample_48k` over voxr's ffmpeg-matched atempo and resampler |
| `kokoro` | `KokoroSpeech`, Kokoro as a speech engine |
| `vits` | `VitsSpeech`, VITS as a speech engine |
| `voxcpm` | `VoxcpmSpeech`, VoxCPM2 as a speech engine |
| `cuda` | The CUDA runtime for Whisper and VoxCPM2 |

## Models

No model weights are compiled in or shipped. Every engine loads its checkpoint at run time from a model directory. voxr fetches nothing over the network: download a checkpoint, for example from Hugging Face, into a directory first. The spaCy tagger and the out-of-vocabulary G2P model for Kokoro are in the Hugging Face repo [`fs90/kokoro-g2p`](https://huggingface.co/fs90/kokoro-g2p). Place them under a Kokoro model directory as `g2p/tagger/` and `g2p/oov/`.

The embedded data is text-processing data only: the misaki lexicons behind `misaki`, and the spaCy tokenizer tables behind `spacy-tokenizer`.

## CUDA

The `cuda` feature builds numr's CUDA kernels. It needs CUDA 12.8 or newer, and 13.x also works.

`NUMR_CUDA_ARCH` selects the GPU architectures the build compiles native code for. Unset, the build targets the GPUs on the build host. Set `NUMR_CUDA_ARCH=all` for any binary that runs on another machine. numr's README lists every value.

## Minimum Rust version

1.89.

## License

Apache-2.0. See [LICENSE](LICENSE).

Code ported from other projects and data voxr redistributes are listed with their licences in [crates/voxr/LICENSE-OTHERS](crates/voxr/LICENSE-OTHERS) and [crates/voxr-narrate/LICENSE-OTHERS](crates/voxr-narrate/LICENSE-OTHERS). Each embedded data set carries its own LICENSE and NOTICE beside it under `crates/voxr/assets/`.
