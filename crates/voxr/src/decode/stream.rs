//! Compressed-audio decoding via symphonia, with Opus registered beside its
//! own codecs, and every file trimmed as ffmpeg trims it: Opus pre-skip,
//! an MP4 audio track's edit list, and the gapless trims a demuxer reads
//! (an MP3's LAME header). A video file decodes its first audio track.
//!
//! Complements [`crate::wav::decode`], which parses uncompressed RIFF/WAVE bytes
//! directly. This module hands the same job — and any container symphonia can
//! probe — to symphonia's decoders, and returns the same [`WavData`] shape at
//! the file's native rate. Downmix and rate conversion are NOT done here: the
//! caller reaches for [`crate::resample::to_mono_at_rate`], so there is exactly
//! one downmix implementation and one resampler in the crate.
//!
//! **CPU-only, one decode pass.** Every sample format symphonia can hand back
//! (float, signed/unsigned integer, 8 through 32 bits) is converted to `f32` in
//! [`push_frame`](super::frames) as each packet arrives, so the whole file is visited once.

use std::sync::OnceLock;

use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::well_known::{CODEC_ID_AAC, CODEC_ID_OPUS, CODEC_ID_VORBIS};
use symphonia::core::codecs::audio::{AudioCodecParameters, AudioDecoderOptions};
use symphonia::core::codecs::registry::CodecRegistry;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, Track, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::error::{Error, Result};

use super::frames::{IntScale, push_frame};
use super::layout::ffmpeg_layout;
use super::opus::{OpusDecoder, pre_skip};
use super::trims::Trims;
use crate::wav::decode::WavData;

/// symphonia's codecs, with Opus.
fn codecs() -> &'static CodecRegistry {
    static CODECS: OnceLock<CodecRegistry> = OnceLock::new();
    CODECS.get_or_init(|| {
        let mut registry = CodecRegistry::new();
        symphonia::default::register_enabled_codecs(&mut registry);
        registry.register_audio_decoder::<OpusDecoder>();
        registry
    })
}

/// The highest AAC core rate SBR doubles: an HE-AAC stream codes its
/// low band at half its output rate, at most 24 kHz.
const SBR_CORE_MAX: u32 = 24_000;

/// A refusal of a well-formed file voxr does not decode, from symphonia's
/// "unsupported" errors and voxr's own checks.
fn unsupported(reason: String) -> Error {
    Error::Unsupported { reason }
}

/// `e` as voxr's error for `what`: [`Error::Unsupported`] when symphonia
/// does not support the stream, else a bad input.
fn symphonia_error(what: &str, e: SymphoniaError) -> Error {
    match e {
        SymphoniaError::Unsupported(_) => unsupported(format!("{what}: {e}")),
        e => bad(format!("{what}: {e}")),
    }
}

pub(super) fn bad(reason: String) -> Error {
    Error::InvalidArgument {
        arg: "audio bytes",
        reason,
    }
}

/// `track` and its audio codec parameters, when it is audio a registered
/// decoder decodes.
fn decodable(track: &Track) -> Option<(&Track, AudioCodecParameters)> {
    track
        .codec_params
        .as_ref()
        .and_then(CodecParameters::audio)
        .filter(|p| codecs().get_audio_decoder(p.codec).is_some())
        .map(|p| (track, p.clone()))
}

/// The first audio track of `mss` decoded to interleaved `f32` at its own
/// rate, trimmed by `trims` as ffmpeg trims it.
pub(super) fn decode_stream(
    mss: MediaSourceStream<'_>,
    hint: Option<&str>,
    trims: Trims,
    int_scale: IntScale,
) -> Result<WavData> {
    let mut probe_hint = Hint::new();
    if let Some(ext) = hint {
        probe_hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &probe_hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| symphonia_error("probing audio format", e))?;

    // The default audio track when it decodes, else the first audio track
    // that does: a video file's default track is its picture.
    let (track, codec_params) = format
        .default_track(TrackType::Audio)
        .and_then(decodable)
        .or_else(|| format.tracks().iter().find_map(decodable))
        .ok_or_else(|| {
            unsupported("audio file has no audio track symphonia decodes".to_string())
        })?;
    let track_id = track.id;
    let (track_delay, track_padding) = (track.delay, track.padding);

    // Gapless off: the decoder yields every frame, priming and padding
    // included, and the trims below cut them as ffmpeg does.
    let mut decoder = codecs()
        .make_audio_decoder(
            &codec_params,
            &AudioDecoderOptions::default().gapless(false),
        )
        .map_err(|e| {
            let what = format!(
                "creating the {} decoder",
                codecs()
                    .get_audio_decoder(codec_params.codec)
                    .map_or("audio", |d| d.codec.info.short_name)
            );
            symphonia_error(&what, e)
        })?;
    let opus = codec_params.codec == CODEC_ID_OPUS;
    let vorbis = codec_params.codec == CODEC_ID_VORBIS;
    let aac = codec_params.codec == CODEC_ID_AAC;
    // What ffmpeg drops: the encoder's priming at the start (Opus pre-skip,
    // an MP3's LAME delay, an MP4 edit's media start, a Matroska codec delay), and its padding at
    // the end (the demuxer's padding, a WebM's discard padding), or all
    // past an MP4 edit's length. The demuxer's own per-packet trims are not
    // read: they disagree across demuxers.
    let delay = if opus {
        pre_skip(&codec_params, track_delay)
    } else {
        track_delay.map_or(0, |d| d as usize)
    };
    let padding = track_padding.map_or(0, |p| p as usize);

    // Interleaved samples at the source rate; the caller downmixes/resamples.
    let mut samples: Vec<f32> = Vec::new();
    // The rate and channels the decoded audio carries.
    let mut spec: Option<(u32, usize)> = None;
    // The ffmpeg layout, and the plane each interleaved channel reads.
    let mut layout: Option<(Option<u64>, Vec<usize>)> = None;
    // The frames the first decoded packet yields.
    let mut first_packet: Option<usize> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) | Err(SymphoniaError::ResetRequired) => break,
            // A file cut short mid-packet ends at its last whole packet.
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(symphonia_error("reading packet", e)),
        };
        if packet.track_id != track_id {
            continue;
        }
        let audio = match decoder.decode(&packet) {
            Ok(a) => a,
            Err(SymphoniaError::DecodeError(_)) => continue, // skip corrupt frames
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(symphonia_error("decoding packet", e)),
        };
        let here = (audio.spec().rate(), audio.spec().channels().count());
        match spec {
            None => {
                if aac && here.0 <= SBR_CORE_MAX && trims.media_scale != Some(here.0) {
                    return Err(unsupported(format!(
                        "AAC at a core rate of {} Hz can carry SBR (HE-AAC), which symphonia does not decode: it would play at half the rate. Decode it with another decoder",
                        here.0
                    )));
                }
                spec = Some(here);
            }
            Some(s) if s != here => {
                return Err(unsupported(format!(
                    "the audio changes from {} Hz, {} channel(s) to {} Hz, {} channel(s) mid-stream",
                    s.0, s.1, here.0, here.1
                )));
            }
            Some(_) => {}
        }
        let (_, order) = layout.get_or_insert_with(|| match trims.wav {
            // A WAV file's planes are its channels in file order.
            Some(header) => (header.layout, (0..here.1).collect()),
            None => ffmpeg_layout(audio.spec().channels(), here.1),
        });
        first_packet.get_or_insert(audio.frames());
        push_frame(&audio, order, int_scale, 0, audio.frames(), &mut samples);
    }
    let Some((sample_rate, channels)) = spec.filter(|&(rate, ch)| rate > 0 && ch > 0) else {
        return Err(bad("decoded audio is empty".to_string()));
    };
    let channels_u16 = u16::try_from(channels)
        .map_err(|_| bad(format!("audio file declares {channels} channels, too many")))?;
    let frames = samples.len() / channels;
    let at_rate = |ns: u64| {
        usize::try_from((u128::from(ns) * u128::from(sample_rate) + 500_000_000) / 1_000_000_000)
            .unwrap_or(usize::MAX)
    };
    let mkv = trims.matroska.unwrap_or_default();
    let padding = padding.saturating_add(at_rate(mkv.discard_ns));
    // Opus primes by its pre-skip, which a Matroska codec delay repeats.
    // Vorbis's first packet yields its first window, which ffmpeg's
    // decoder drops. A Matroska codec delay is not applied on top.
    let delay = if opus {
        delay
    } else if vorbis {
        delay.max(first_packet.unwrap_or(0))
    } else {
        delay.max(at_rate(mkv.codec_delay_ns))
    };
    let (skip, keep) = match trims.edit {
        Some(edit) if !opus => (
            edit.skip_frames(sample_rate),
            edit.keep_frames(sample_rate).unwrap_or(frames),
        ),
        _ => (delay, frames.saturating_sub(delay + padding)),
    };
    let from = skip.min(frames);
    let to = from.saturating_add(keep).min(frames);
    samples.truncate(to * channels);
    samples.drain(..from * channels);

    if samples.is_empty() {
        return Err(bad("decoded audio is empty".to_string()));
    }

    Ok(WavData {
        samples,
        sample_rate,
        channels: channels_u16,
        layout: layout.and_then(|(mask, _)| mask),
    })
}
