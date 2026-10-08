//! The edit list of an MP4/M4A/MOV file's first audio track: the encoder
//! priming its first edit skips and the length it plays for, as ffmpeg
//! applies them. symphonia reads the edit list but plays the whole track,
//! priming included, so the decode trims by it.

use std::io::{Read, Seek, SeekFrom};

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

/// One box: its type, and where its body starts and ends.
struct Atom {
    kind: [u8; 4],
    body: u64,
    end: u64,
}

/// The box at the reader's position, ending no later than `limit`.
fn atom<R: Read + Seek>(r: &mut R, limit: u64) -> Option<Atom> {
    let start = r.stream_position().ok()?;
    if start + 8 > limit {
        return None;
    }
    let mut head = [0u8; 8];
    r.read_exact(&mut head).ok()?;
    let size = u64::from(u32::from_be_bytes([head[0], head[1], head[2], head[3]]));
    let kind = [head[4], head[5], head[6], head[7]];
    let (body, end) = match size {
        0 => (start + 8, limit),
        1 => {
            let mut large = [0u8; 8];
            r.read_exact(&mut large).ok()?;
            (start + 16, start.checked_add(u64::from_be_bytes(large))?)
        }
        n => (start + 8, start.checked_add(n)?),
    };
    (end >= body && end <= limit).then_some(Atom { kind, body, end })
}

/// Every box between the reader's position and `end`, each passed to `f`
/// with the reader at its body; the reader moves past each after.
fn each<R: Read + Seek>(
    r: &mut R,
    end: u64,
    f: &mut dyn FnMut(&mut R, &Atom) -> Option<()>,
) -> Option<()> {
    while let Some(a) = atom(r, end) {
        f(r, &a)?;
        r.seek(SeekFrom::Start(a.end)).ok()?;
    }
    Some(())
}

/// The body of `a`, read whole: only small boxes are read this way.
fn body<R: Read + Seek>(r: &mut R, a: &Atom) -> Option<Vec<u8>> {
    let len = usize::try_from(a.end - a.body)
        .ok()
        .filter(|&n| n <= 1 << 20)?;
    r.seek(SeekFrom::Start(a.body)).ok()?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).ok()?;
    Some(buf)
}

fn u32_at(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(i..i + 4)?.try_into().ok()?))
}

fn u64_at(b: &[u8], i: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(i..i + 8)?.try_into().ok()?))
}

/// The timescale of an `mvhd` or `mdhd` body.
fn timescale(b: &[u8]) -> Option<u32> {
    match b.first()? {
        1 => u32_at(b, 20),
        _ => u32_at(b, 12),
    }
}

/// The first edit of an `elst` body that plays media: its length and its
/// media start.
fn first_edit(b: &[u8]) -> Option<(u64, u64)> {
    let v1 = *b.first()? == 1;
    let count = u32_at(b, 4)? as usize;
    let size = if v1 { 20 } else { 12 };
    (0..count).find_map(|k| {
        let at = 8 + k * size;
        let (keep, media) = if v1 {
            (u64_at(b, at)?, u64_at(b, at + 8)? as i64)
        } else {
            (
                u64::from(u32_at(b, at)?),
                i64::from(u32_at(b, at + 4)? as i32),
            )
        };
        u64::try_from(media).ok().map(|m| (keep, m))
    })
}

/// What one `trak` holds: its handler, its timescale and its first edit.
#[derive(Default)]
struct Track {
    handler: Option<[u8; 4]>,
    scale: Option<u32>,
    edit: Option<(u64, u64)>,
}

fn track<R: Read + Seek>(r: &mut R, end: u64) -> Option<Track> {
    let mut t = Track::default();
    each(r, end, &mut |r, a| {
        match &a.kind {
            b"edts" => {
                each(r, a.end, &mut |r, e| {
                    if &e.kind == b"elst" {
                        t.edit = body(r, e).as_deref().and_then(first_edit);
                    }
                    Some(())
                })?;
            }
            b"mdia" => {
                each(r, a.end, &mut |r, m| {
                    match &m.kind {
                        b"mdhd" => t.scale = body(r, m).as_deref().and_then(timescale),
                        b"hdlr" => {
                            t.handler = body(r, m)
                                .and_then(|b| b.get(8..12).and_then(|h| h.try_into().ok()))
                        }
                        _ => {}
                    }
                    Some(())
                })?;
            }
            _ => {}
        }
        Some(())
    })?;
    Some(t)
}

/// What the first audio track of an MP4 says: its edit, and its media
/// timescale.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct AudioTrack {
    /// The first edit that plays media, if any.
    pub(super) edit: Option<Edit>,
    /// The track's timescale (`mdhd`): the rate its sample durations count
    /// in, which is the rate the track plays at.
    pub(super) media_scale: Option<u32>,
}

/// The first audio track in the MP4 at `r`, which is left at an unknown
/// position. `None` when the file is no MP4, has no audio track, or a box
/// does not read.
pub(super) fn audio_track<R: Read + Seek>(r: &mut R) -> Option<AudioTrack> {
    let len = r.seek(SeekFrom::End(0)).ok()?;
    r.seek(SeekFrom::Start(0)).ok()?;
    let mut found = None;
    each(r, len, &mut |r, top| {
        if &top.kind != b"moov" || found.is_some() {
            return Some(());
        }
        let mut movie_scale = None;
        each(r, top.end, &mut |r, a| {
            match &a.kind {
                b"mvhd" => movie_scale = body(r, a).as_deref().and_then(timescale),
                b"trak" if found.is_none() => {
                    let t = track(r, a.end)?;
                    if t.handler == Some(*b"soun") {
                        let edit = match (t.edit, t.scale, movie_scale) {
                            (Some((keep, skip)), Some(media_scale), Some(movie)) => Some(Edit {
                                skip,
                                media_scale,
                                keep,
                                movie_scale: movie,
                            }),
                            _ => None,
                        };
                        found = Some(AudioTrack {
                            edit,
                            media_scale: t.scale,
                        });
                    }
                }
                _ => {}
            }
            Some(())
        })
    })?;
    found
}

/// The edit of the first audio track in the MP4 at `r`. `None` when
/// [`audio_track`] finds none, or the track has no edit that plays media.
#[cfg(test)]
fn audio_edit<R: Read + Seek>(r: &mut R) -> Option<Edit> {
    audio_track(r).and_then(|t| t.edit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

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

    fn mp4(handler: &[u8; 4]) -> Vec<u8> {
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
        let moov = [boxed(b"mvhd", &header(1000)), boxed(b"trak", &trak)].concat();
        [
            boxed(b"ftyp", b"M4A \0\0\0\0"),
            boxed(b"moov", &moov),
            boxed(b"mdat", &[0u8; 64]),
        ]
        .concat()
    }

    #[test]
    fn the_audio_tracks_edit_skips_its_priming() {
        let edit = audio_edit(&mut Cursor::new(mp4(b"soun")));
        let Some(edit) = edit else {
            panic!("no edit read");
        };
        assert_eq!(edit.skip_frames(48_000), 1024);
        assert_eq!(edit.keep_frames(48_000), Some(96_000));
        let track = audio_track(&mut Cursor::new(mp4(b"soun")));
        assert_eq!(track.and_then(|t| t.media_scale), Some(48_000));
        assert_eq!(audio_edit(&mut Cursor::new(mp4(b"vide"))), None);
        assert_eq!(audio_edit(&mut Cursor::new(b"not an mp4".to_vec())), None);
    }
}
