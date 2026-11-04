//! What a container says apart from symphonia: the trims ffmpeg applies
//! and a WAV file's channel layout.

use containr::{EditList, TrackKind, mkv, mp4};

use crate::wav::layout::wav_layout;

/// File extensions of the ISO base media family, whose edit lists trim.
const MP4_EXTENSIONS: [&str; 6] = ["mp4", "m4a", "m4v", "mov", "3gp", "m4b"];

/// File extensions of the Matroska family, whose blocks mark padding.
const MKV_EXTENSIONS: [&str; 3] = ["mkv", "webm", "mka"];

/// The first audio track's first edit that plays media.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Edit {
    /// Media time the edit starts at, in the track's timescale: the
    /// priming to skip.
    skip: u64,
    /// The track's timescale (`mdhd`).
    media_scale: u32,
    /// How long the edit plays, in the movie's timescale. 0 plays to the
    /// end.
    keep: u64,
    /// The movie's timescale (`mvhd`).
    movie_scale: u32,
}

impl Edit {
    /// The first edit of `list` that plays media, in a track of `media_scale`.
    fn first_of(list: &EditList, media_scale: u32) -> Option<Self> {
        list.entries.iter().find_map(|entry| {
            u64::try_from(entry.media_time).ok().map(|skip| Self {
                skip,
                media_scale,
                keep: entry.segment_duration,
                movie_scale: list.movie_timescale,
            })
        })
    }

    /// Frames to skip at the start, at `rate`.
    pub(super) fn skip_frames(&self, rate: u32) -> usize {
        scale(self.skip, rate, self.media_scale)
    }

    /// Frames the edit plays, at `rate`. `None` when it plays to the end.
    pub(super) fn keep_frames(&self, rate: u32) -> Option<usize> {
        (self.keep > 0).then(|| scale(self.keep, rate, self.movie_scale))
    }
}

/// `value` in units of `1 / from` seconds, as frames at `rate`, rounded.
fn scale(value: u64, rate: u32, from: u32) -> usize {
    if from == 0 {
        return 0;
    }
    let frames = (u128::from(value) * u128::from(rate) + u128::from(from) / 2) / u128::from(from);
    usize::try_from(frames).unwrap_or(usize::MAX)
}

/// The trims of a Matroska audio track, in nanoseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Matroska {
    /// The codec delay: priming to drop at the start.
    pub(super) codec_delay_ns: u64,
    /// The discard padding of every block: padding to drop at the end.
    pub(super) discard_ns: u64,
}

/// The trim facts containr read. An error gives none: symphonia decides on
/// its own whether the file decodes, and the decode then applies no trim.
fn or_none<T>(read: containr::Result<Option<T>>) -> Option<T> {
    read.unwrap_or_else(|error| {
        tracing::warn!("reading the container's trims failed, so the decode applies none: {error}");
        None
    })
}

/// What the container says, read apart from symphonia: what to trim, and
/// a WAV file's channel layout.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Trims {
    /// An MP4 audio track's edit.
    pub(super) edit: Option<Edit>,
    /// An MP4 audio track's timescale. An AAC track that plays at its core
    /// rate carries no SBR: HE-AAC's track counts at the doubled rate.
    pub(super) media_scale: Option<u32>,
    /// A Matroska audio track's codec delay and discard padding.
    pub(super) matroska: Option<Matroska>,
    /// A WAV file's `fmt` chunk. symphonia fills in a layout the file
    /// does not name, where ffmpeg takes its default one.
    pub(super) wav: Option<WavHeader>,
}

/// What a WAV file's `fmt` chunk says of its channels.
#[derive(Debug, Clone, Copy)]
pub(super) struct WavHeader {
    /// The layout `dwChannelMask` names, if any.
    pub(super) layout: Option<u64>,
}

/// The longest WAV prefix read for its `fmt` chunk.
const WAV_PREFIX: u64 = 1 << 20;

impl Trims {
    /// The trims of the file `r` holds, read when its bytes or extension
    /// name a container that carries them.
    pub(super) fn of<R: std::io::Read + std::io::Seek>(
        r: &mut R,
        head: &[u8],
        hint: Option<&str>,
    ) -> Self {
        let named =
            |list: &[&str]| hint.is_some_and(|h| list.iter().any(|e| e.eq_ignore_ascii_case(h)));
        if head.get(4..8) == Some(b"ftyp") || named(&MP4_EXTENSIONS) {
            let track = or_none(mp4::first_track_timing(&mut *r, TrackKind::Audio));
            Self {
                edit: track
                    .as_ref()
                    .and_then(|t| Edit::first_of(&t.edit_list, t.timescale)),
                media_scale: track.map(|t| t.timescale),
                ..Self::default()
            }
        } else if head.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) || named(&MKV_EXTENSIONS) {
            Self {
                matroska: or_none(mkv::first_track_timing(&mut *r, TrackKind::Audio)).map(|t| {
                    Matroska {
                        codec_delay_ns: t.codec_delay,
                        discard_ns: t.discard_padding,
                    }
                }),
                ..Self::default()
            }
        } else if head.starts_with(b"RIFF") {
            let mut prefix = Vec::new();
            let read = r.seek(std::io::SeekFrom::Start(0)).is_ok()
                && std::io::Read::read_to_end(
                    &mut std::io::Read::take(&mut *r, WAV_PREFIX),
                    &mut prefix,
                )
                .is_ok();
            Self {
                wav: (read && prefix.get(8..12) == Some(b"WAVE")).then(|| WavHeader {
                    layout: wav_layout(&prefix),
                }),
                ..Self::default()
            }
        } else {
            Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    //! Trims read through containr. containr's own tests cover its parse.

    use std::io::Cursor;

    use super::*;

    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    fn header(scale: u32) -> Vec<u8> {
        let mut b = vec![0u8; 12];
        b.extend_from_slice(&scale.to_be_bytes());
        b.extend_from_slice(&[0, 0, 0, 0]);
        b
    }

    /// An MP4 of one track of `handler`, with `moov_extra` at the end of its
    /// `moov`.
    fn mp4_file(handler: &[u8; 4], moov_extra: &[u8]) -> Vec<u8> {
        let mut hdlr = vec![0u8; 8];
        hdlr.extend_from_slice(handler);
        hdlr.extend_from_slice(&[0u8; 12]);
        let mdia = [boxed(b"mdhd", &header(48_000)), boxed(b"hdlr", &hdlr)].concat();
        let mut elst = vec![0, 0, 0, 0, 0, 0, 0, 2];
        // An empty edit, then one that skips 1024 and plays 2 s at 1000/s.
        elst.extend_from_slice(&5u32.to_be_bytes());
        elst.extend_from_slice(&(-1i32).to_be_bytes());
        elst.extend_from_slice(&[0, 1, 0, 0]);
        elst.extend_from_slice(&2000u32.to_be_bytes());
        elst.extend_from_slice(&1024u32.to_be_bytes());
        elst.extend_from_slice(&[0, 1, 0, 0]);
        let trak = [
            boxed(b"tkhd", &[0u8; 84]),
            boxed(b"edts", &boxed(b"elst", &elst)),
            boxed(b"mdia", &mdia),
        ]
        .concat();
        let moov = [
            boxed(b"mvhd", &header(1000)),
            boxed(b"trak", &trak),
            moov_extra.to_vec(),
        ]
        .concat();
        [
            boxed(b"ftyp", b"M4A \0\0\0\0"),
            boxed(b"moov", &moov),
            boxed(b"mdat", &[0u8; 64]),
        ]
        .concat()
    }

    fn trims_of(file: &[u8], hint: Option<&str>) -> Trims {
        Trims::of(&mut Cursor::new(file), file, hint)
    }

    fn assert_skips_its_priming(trims: Trims) {
        let Some(edit) = trims.edit else {
            panic!("no edit read");
        };
        assert_eq!(edit.skip_frames(48_000), 1024);
        assert_eq!(edit.keep_frames(48_000), Some(96_000));
        assert_eq!(trims.media_scale, Some(48_000));
    }

    #[test]
    fn the_audio_tracks_edit_skips_its_priming() {
        assert_skips_its_priming(trims_of(&mp4_file(b"soun", &[]), None));
    }

    #[test]
    fn a_fragmented_file_keeps_its_edit_and_timescale() {
        let mut file = mp4_file(b"soun", &boxed(b"mvex", &[]));
        file.extend_from_slice(&boxed(b"moof", &[]));
        assert_skips_its_priming(trims_of(&file, None));
    }

    #[test]
    fn a_file_with_no_audio_track_gives_no_edit() {
        let trims = trims_of(&mp4_file(b"vide", &[]), None);
        assert_eq!(trims.edit, None);
        assert_eq!(trims.media_scale, None);
        let trims = trims_of(b"not an mp4", Some("m4a"));
        assert_eq!(trims.edit, None);
        assert_eq!(trims.media_scale, None);
    }

    #[test]
    fn an_edit_of_zero_length_plays_to_the_end() {
        let edit = Edit {
            skip: 1024,
            media_scale: 48_000,
            keep: 0,
            movie_scale: 1000,
        };
        assert_eq!(edit.skip_frames(48_000), 1024);
        assert_eq!(edit.keep_frames(48_000), None);
    }

    /// An element of a one-byte-size body.
    fn el(id: &[u8], body: &[u8]) -> Vec<u8> {
        let mut out = id.to_vec();
        out.push(0x80 | body.len() as u8);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn the_last_blocks_padding_is_read() {
        let entry = [
            el(&[0xD7], &[2]),
            el(&[0x83], &[2]),
            el(&[0x56, 0xAA], &[0x0F, 0x42, 0x40]),
        ]
        .concat();
        let tracks = el(&[0x16, 0x54, 0xAE, 0x6B], &el(&[0xAE], &entry));
        // A block of track 2 that marks 13.5 ms as padding, then one of
        // track 1 whose padding is not the audio's.
        let ours = [
            el(&[0xA1], &[0x82, 0, 0, 0]),
            el(&[0x75, 0xA2], &[0x00, 0xCD, 0xFE, 0x60]),
        ]
        .concat();
        let theirs = [el(&[0xA1], &[0x81, 0, 0, 0]), el(&[0x75, 0xA2], &[0x10])].concat();
        let cluster = el(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[el(&[0xE7], &[0]), el(&[0xA0], &ours), el(&[0xA0], &theirs)].concat(),
        );
        let segment = el(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        let file = [el(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();
        assert_eq!(
            trims_of(&file, None).matroska,
            Some(Matroska {
                codec_delay_ns: 1_000_000,
                discard_ns: 13_500_000
            })
        );
        assert_eq!(trims_of(b"RIFF....", Some("mkv")).matroska, None);
    }
}
