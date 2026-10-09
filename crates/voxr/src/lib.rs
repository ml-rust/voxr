//! Sample-level audio DSP, file codecs, voices, and product pipelines.
//!
//! Placement rule: code that operates on `Tensor<R>` with backend kernels
//! belongs in `boostr::model::audio`. Code that operates on `Vec<f32>`
//! samples, files, directories, voices, or text pipelines belongs here.
//!
//! Always compiled: RIFF/WAVE in and out ([`wav`]), the polyphase
//! [`resample`](mod@resample)rs (general, ffmpeg-exact and SciPy-exact), the single-sound
//! [`metrics`] and the [`sound_review`] built on them, the reference-take
//! [`enhance`] chain, [`pitch`], [`quality`], two-signal [`spectral`]
//! distance, the `atempo`-compatible tempo [`stretch`], ASR [`eval`]
//! metrics, the signal and summary halves of the [`tts_eval`] quality gate,
//! and the [`Lang`] tag a voice carries. Feature gates add the
//! compressed-audio [`decode`]rs, the [`clap`] audio-text scorer, the
//! misaki G2P backend ([`g2p`]), the spaCy `en_core_web_sm` tokenizer and
//! tagger, the [`tts`] surface, the [`kokoro`] and [`voxcpm`]
//! engines, the [`whisper`] bundle, the [`vad`] segmenter, [`corpus`]
//! preparation, and the pace, intelligibility and timbre legs of
//! [`tts_eval`].
//!
//! [`sound_review`] implements the sound review of the motion-video skill
//! (<https://github.com/farhan-syah/motion-video-skill>). Docs here call it
//! "the motion-video skill" or "the skill".

pub mod enhance;
pub mod error;
pub mod eval;
pub mod ffmpeg;
pub mod g2p;
pub mod lang;
pub mod metrics;
pub mod pitch;
pub mod quality;
pub mod resample;
pub mod sound_review;
pub mod spectral;
pub mod stretch;
pub mod tts_eval;
pub mod wav;

#[cfg(feature = "clap")]
pub mod clap;
#[cfg(feature = "corpus")]
pub mod corpus;
#[cfg(feature = "decode")]
pub mod decode;
#[cfg(feature = "kokoro")]
pub mod kokoro;
#[cfg(feature = "tts")]
pub mod tts;
#[cfg(feature = "vad")]
pub mod vad;
#[cfg(feature = "voxcpm")]
pub mod voxcpm;
#[cfg(feature = "whisper")]
pub mod whisper;

pub use error::{Error, Result};

#[cfg(feature = "corpus")]
pub use corpus::{
    CorpusOptions, MAX_UTTERANCE_SECS, PRETRAINED_TOKENIZER_NAMES, SpeechCorpusBuilder,
    TextTokenizer, Utterance, check_max_speech_duration, pack_utterances,
    pack_utterances_with_layout,
};
#[cfg(feature = "decode")]
pub use decode::{
    decode_audio, decode_audio_file_mono_at, decode_audio_like_ffmpeg, decode_audio_mono_at,
    extension_hint,
};
pub use eval::{
    ErrorRate, align, by_group, canonical_character_error_rate, canonical_word_error_rate,
    canonicalize, character_error_rate, grand_total, normalize, total, word_error_rate,
};
pub use ffmpeg::{ffmpeg_program, ffprobe_program};
pub use lang::{Lang, LangTag};
pub use pitch::{PitchOptions, PitchTrack, estimate_pitch};
pub use quality::{TakeQuality, measure_quality};
pub use resample::{
    DEFAULT_TAPS_PER_PHASE, MAX_FILTER_TAPS, PolySample, ffmpeg_default_layout,
    ffmpeg_resampled_len, librosa_output_len, resample, resample_interleaved_like_ffmpeg,
    resample_like_ffmpeg, resample_like_librosa, resample_poly, resample_with_taps,
    to_mono_at_rate, to_mono_like_ffmpeg,
};
pub use spectral::{SpectralDistance, multi_resolution_stft_distance, snr_db};
pub use stretch::{MAX_TEMPO, MIN_TEMPO, PcmSample, atempo, atempo_interleaved, atempo_pcm16};
#[cfg(feature = "tts")]
pub use tts::{
    KOKORO_VOICES, SynthesizeOptions, TtsBundle, TtsEngine, TtsError, Voice, default_kokoro_voices,
};
#[cfg(feature = "voxcpm")]
pub use tts_eval::{F0Stats, TimbreProxy, TimbreScorer, timbre_proxy};
#[cfg(feature = "whisper")]
pub use tts_eval::{Intelligibility, IntelligibilityClient, IntelligibilityScorer};
pub use tts_eval::{
    MetricStat, RowScore, SignalStats, Summary, Thresholds, UNLABELLED_AXIS, signal_stats,
    violations,
};
#[cfg(feature = "vad")]
pub use tts_eval::{Pace, pace, pace_from_segments};
#[cfg(feature = "vad")]
pub use vad::{SpeechSegment, VadSegmentOptions, segments_from_probabilities, speech_timestamps};
pub use wav::{
    WavData, WavF64, decode_wav, decode_wav_f64, encode_f32_raw, encode_pcm16_raw, encode_wav_f32,
    encode_wav_pcm16, encode_wav_pcm16_values, to_mono, wav_stream_header_f32,
    wav_stream_header_pcm16,
};
#[cfg(feature = "whisper")]
pub use whisper::{
    TranscribeOptions, Transcription, WhisperBundle, WhisperGenerationConfig, WordTiming,
};

#[cfg(all(test, feature = "decode"))]
pub(crate) mod test_utils {
    use std::path::PathBuf;

    /// Resolve a real-audio fixture: `$AUDIO_CORPUS_FLAC`, else the first
    /// `.flac` in `$AUDIO_CORPUS_DIR` by sorted name. `None` when neither is
    /// set or the resolved path is absent, so callers skip.
    pub(crate) fn corpus_flac() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("AUDIO_CORPUS_FLAC") {
            let path = PathBuf::from(p);
            return path.exists().then_some(path);
        }
        let dir = PathBuf::from(std::env::var("AUDIO_CORPUS_DIR").ok()?);
        let mut flacs: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "flac"))
            .collect();
        flacs.sort();
        flacs.into_iter().next()
    }
}
