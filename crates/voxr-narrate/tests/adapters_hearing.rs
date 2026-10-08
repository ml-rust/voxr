//! `WhisperHearing` against `TranscriberHearing`: over the same Whisper
//! checkpoints, with `to_16k` as the resampler, both hear the same timed
//! words in a recording, for each recognizer. The default recognizer is
//! Whisper base, the small one Whisper small, and the configured one Whisper
//! small again under the config's slot.
//!
//! Needs `WHISPER_BASE_DIR` (or `whisper-base` under `BOOSTR_MODELS_DIR`),
//! `WHISPER_SMALL_DIR` (or `whisper-small`; else base serves, with a notice)
//! and `VOXCPM2_REF_WAV`, a speech recording. Skips with a notice when base
//! or the recording is missing.

mod common;

use common::models::{load_whisper, model_path, skip, whisper_base, whisper_small};
use voxr_narrate::{
    Hearing, Recognizer, RecognizerSource, TranscriberHearing, WhisperHearing, decode_48k, to_16k,
};

#[test]
fn whisper_hearing_hears_what_transcriber_hearing_hears() {
    let Some(base) = whisper_base() else {
        return;
    };
    let Some(recording) = model_path("VOXCPM2_REF_WAV", "") else {
        skip("speech recording", "VOXCPM2_REF_WAV");
        return;
    };
    let take = decode_48k(&recording).expect("decoding the recording");
    let whisper = load_whisper(&base);
    let small_whisper = load_whisper(&whisper_small(&base));
    let mut hearing = WhisperHearing::new(
        RecognizerSource::loaded(whisper.words()),
        RecognizerSource::loaded(small_whisper.words()),
    )
    .with_configured(RecognizerSource::loaded(small_whisper.words()));
    let (mut default, mut small, mut configured) = (
        whisper.words(),
        small_whisper.words(),
        small_whisper.words(),
    );
    let mut reference = TranscriberHearing {
        default: &mut default,
        small: &mut small,
        configured: Some(&mut configured),
        to_16k,
    };
    for (recognizer, language) in [
        (Recognizer::Default, None),
        (Recognizer::Small, Some("en")),
        (Recognizer::Configured, Some("en")),
    ] {
        let ours = hearing
            .hear(&take, 48_000, recognizer, language)
            .expect("WhisperHearing");
        let theirs = reference
            .hear(&take, 48_000, recognizer, language)
            .expect("TranscriberHearing");
        assert!(!ours.words.is_empty(), "nothing heard in the recording");
        assert_eq!(ours, theirs, "{recognizer:?} heard differently");
    }
}
