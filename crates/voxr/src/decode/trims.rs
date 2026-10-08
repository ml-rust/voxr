//! What a container says apart from symphonia: the trims ffmpeg applies
//! and a WAV file's channel layout.

use super::mkv_discard::{Matroska, matroska_trims};
use super::mp4_edit::{Edit, audio_track};
use crate::wav::layout::wav_layout;

/// File extensions of the ISO base media family, whose edit lists trim.
const MP4_EXTENSIONS: [&str; 6] = ["mp4", "m4a", "m4v", "mov", "3gp", "m4b"];

/// File extensions of the Matroska family, whose blocks mark padding.
const MKV_EXTENSIONS: [&str; 3] = ["mkv", "webm", "mka"];

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
            let track = audio_track(r).unwrap_or_default();
            Self {
                edit: track.edit,
                media_scale: track.media_scale,
                ..Self::default()
            }
        } else if head.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) || named(&MKV_EXTENSIONS) {
            Self {
                matroska: matroska_trims(r),
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
