//! The trims of a Matroska/WebM file's first audio track, which ffmpeg
//! applies and symphonia does not: its codec delay, the encoder priming at
//! the start, and the discard padding its blocks mark past the audio's
//! end.

use std::io::{Read, Seek, SeekFrom};

const SEGMENT: u32 = 0x1853_8067;
const TRACKS: u32 = 0x1654_AE6B;
const TRACK_ENTRY: u32 = 0xAE;
const TRACK_NUMBER: u32 = 0xD7;
const TRACK_TYPE: u32 = 0x83;
const CODEC_DELAY: u32 = 0x56AA;
const CLUSTER: u32 = 0x1F43_B675;
const BLOCK_GROUP: u32 = 0xA0;
const BLOCK: u32 = 0xA1;
const DISCARD_PADDING: u32 = 0x75A2;
/// `TrackType` of an audio track.
const AUDIO: u64 = 2;

/// One element: its id, and where its body starts and ends.
struct Element {
    id: u32,
    body: u64,
    end: u64,
}

/// A variable-length integer: its length from the first byte's leading
/// zeros. With `keep_marker` (an element id) the length marker stays.
fn vint<R: Read>(r: &mut R, keep_marker: bool) -> Option<(u64, usize, bool)> {
    let mut first = [0u8; 1];
    r.read_exact(&mut first).ok()?;
    let len = first[0].leading_zeros() as usize + 1;
    if len > 8 {
        return None;
    }
    // The bits after the length marker; 8 bits shift through `u16`, as a
    // `u8` shift by 8 overflows.
    let low = (0xFFu16 >> len) as u8;
    let mut value = if keep_marker {
        u64::from(first[0])
    } else {
        u64::from(first[0] & low)
    };
    let mut all_ones = first[0] & low == low;
    let mut rest = [0u8; 7];
    r.read_exact(&mut rest[..len - 1]).ok()?;
    for &b in &rest[..len - 1] {
        value = (value << 8) | u64::from(b);
        all_ones &= b == 0xFF;
    }
    Some((value, len, all_ones))
}

/// The element at the reader's position, ending no later than `limit`. An
/// unknown size runs to `limit`.
fn element<R: Read + Seek>(r: &mut R, limit: u64) -> Option<Element> {
    let start = r.stream_position().ok()?;
    if start >= limit {
        return None;
    }
    let (id, id_len, _) = vint(r, true)?;
    if id_len > 4 {
        return None;
    }
    let (size, size_len, unknown) = vint(r, false)?;
    let body = start + (id_len + size_len) as u64;
    let end = if unknown {
        limit
    } else {
        body.checked_add(size)?.min(limit)
    };
    Some(Element {
        id: u32::try_from(id).ok()?,
        body,
        end,
    })
}

/// The body of `e` as an integer, big-endian, `signed` as two's complement.
fn integer<R: Read + Seek>(r: &mut R, e: &Element, signed: bool) -> Option<i64> {
    let len = usize::try_from(e.end - e.body)
        .ok()
        .filter(|&n| (1..=8).contains(&n))?;
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf[..len]).ok()?;
    let mut v: u64 = 0;
    for &b in &buf[..len] {
        v = (v << 8) | u64::from(b);
    }
    if signed && buf[0] & 0x80 != 0 {
        let shift = 64 - 8 * len as u32;
        Some(((v << shift) as i64) >> shift)
    } else {
        i64::try_from(v).ok()
    }
}

/// Each child of a body ending at `end`, passed to `f` with the reader at
/// its body; the reader moves past each after.
fn children<R: Read + Seek>(
    r: &mut R,
    end: u64,
    f: &mut dyn FnMut(&mut R, &Element) -> Option<()>,
) -> Option<()> {
    while let Some(e) = element(r, end) {
        f(r, &e)?;
        r.seek(SeekFrom::Start(e.end)).ok()?;
    }
    Some(())
}

/// The trims of a Matroska audio track, in nanoseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Matroska {
    /// The codec delay: priming to drop at the start.
    pub(super) codec_delay_ns: u64,
    /// The discard padding of every block: padding to drop at the end.
    pub(super) discard_ns: u64,
}

/// The number and codec delay of the first audio track in a `Tracks`
/// body.
fn audio_track<R: Read + Seek>(r: &mut R, tracks: &Element) -> Option<(u64, u64)> {
    let mut found = None;
    children(r, tracks.end, &mut |r, entry| {
        if entry.id != TRACK_ENTRY || found.is_some() {
            return Some(());
        }
        let (mut number, mut kind, mut delay) = (None, None, 0);
        children(r, entry.end, &mut |r, e| {
            match e.id {
                TRACK_NUMBER => number = integer(r, e, false),
                TRACK_TYPE => kind = integer(r, e, false),
                CODEC_DELAY => delay = integer(r, e, false).unwrap_or(0),
                _ => {}
            }
            Some(())
        })?;
        if kind == Some(AUDIO as i64) {
            found = number
                .and_then(|n| u64::try_from(n).ok())
                .map(|n| (n, u64::try_from(delay).unwrap_or(0)));
        }
        Some(())
    })?;
    found
}

/// The discard padding of one `BlockGroup` of `track`, in nanoseconds.
fn group_padding<R: Read + Seek>(r: &mut R, group: &Element, track: u64) -> Option<i64> {
    let (mut number, mut padding) = (None, 0i64);
    children(r, group.end, &mut |r, e| {
        match e.id {
            BLOCK => number = vint(r, false).map(|(n, _, _)| n),
            DISCARD_PADDING => padding = integer(r, e, true).unwrap_or(0),
            _ => {}
        }
        Some(())
    })?;
    Some(if number == Some(track) {
        padding.max(0)
    } else {
        0
    })
}

/// The trims of the first audio track of the Matroska file at `r`. `None`
/// when it is no Matroska file, has no audio track, or an element does not
/// read.
pub(super) fn matroska_trims<R: Read + Seek>(r: &mut R) -> Option<Matroska> {
    let len = r.seek(SeekFrom::End(0)).ok()?;
    r.seek(SeekFrom::Start(0)).ok()?;
    let mut total = None;
    children(r, len, &mut |r, top| {
        if top.id != SEGMENT {
            return Some(());
        }
        let mut track = None;
        let mut sum = 0i64;
        children(r, top.end, &mut |r, e| {
            match e.id {
                TRACKS => track = audio_track(r, e),
                CLUSTER => {
                    if let Some((t, _)) = track {
                        children(r, e.end, &mut |r, g| {
                            if g.id == BLOCK_GROUP {
                                sum += group_padding(r, g, t)?;
                            }
                            Some(())
                        })?;
                    }
                }
                _ => {}
            }
            Some(())
        })?;
        if let Some((_, codec_delay_ns)) = track {
            total = u64::try_from(sum).ok().map(|discard_ns| Matroska {
                codec_delay_ns,
                discard_ns,
            });
        }
        Some(())
    })?;
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

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
            &[el(&[0xA0], &ours), el(&[0xA0], &theirs)].concat(),
        );
        let segment = el(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
        let file = [el(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat();
        assert_eq!(
            matroska_trims(&mut Cursor::new(file)),
            Some(Matroska {
                codec_delay_ns: 1_000_000,
                discard_ns: 13_500_000
            })
        );
        assert_eq!(matroska_trims(&mut Cursor::new(b"RIFF....".to_vec())), None);
    }

    #[test]
    fn an_eight_byte_size_reads() {
        // A Void element ffmpeg writes: its size in eight bytes.
        let mut r = Cursor::new(vec![0xEC, 0x01, 0, 0, 0, 0, 0, 0, 0x04, 1, 2, 3, 4]);
        let e = element(&mut r, 13);
        assert!(e.is_some_and(|e| e.id == 0xEC && e.body == 9 && e.end == 13));
    }
}
