//! `decode_audio_like_ffmpeg` and `to_mono_like_ffmpeg` against ffmpeg on
//! clips ffmpeg encodes here:
//!
//! - AAC in M4A, MP4, MOV (with a video track) and MKV, and ALAC in M4A
//! - Opus in WebM and Ogg, Vorbis in WebM, and MP3
//! - quad, 5.0 and 5.1 in FLAC and in `WAVE_FORMAT_EXTENSIBLE` WAV, and
//!   4- and 6-channel WAVs that name no layout
//!
//! Each clip decodes to the same length as `ffmpeg -ac 1 -f f32le` gives.
//! Its mono mix and each of its channels stay above a per-codec SNR floor
//! against ffmpeg's.
//!
//! The source is a speech-like signal: tones under an envelope, a pause,
//! and a noise floor. ffmpeg is `$FFMPEG`, else `ffmpeg` on `PATH`. The
//! test skips with a notice when it does not run, or lacks an encoder.

use std::path::{Path, PathBuf};
use std::process::Command;

use voxr::{decode_audio_like_ffmpeg, encode_wav_pcm16, to_mono_like_ffmpeg};

const RATE: u32 = 48_000;

/// 2.5 s of a speech-like signal: two voiced runs with a pause, and noise.
/// The second channel is the first at a lower level. Each further channel
/// speaks higher and quieter, so a channel mixed at the wrong weight or
/// read from the wrong place shows.
fn source(channels: usize) -> Vec<f32> {
    let n = (RATE as usize) * 5 / 2;
    let mut seed = 0x2545_f491u32;
    let mut out = Vec::with_capacity(n * channels);
    for i in 0..n {
        let t = i as f64 / f64::from(RATE);
        let voiced = (0.1..1.1).contains(&t) || (1.4..2.3).contains(&t);
        let env = if voiced {
            (std::f64::consts::PI * ((t * 4.0) % 1.0)).sin().abs()
        } else {
            0.0
        };
        let f0 = 140.0 + 30.0 * (t * 3.0).sin();
        let tone = |rise: f64| {
            (0..6)
                .map(|h| {
                    let k = f64::from(h + 1);
                    (std::f64::consts::TAU * f0 * rise * k * t).sin() / k
                })
                .sum::<f64>()
        };
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let noise = (f64::from(seed) / f64::from(u32::MAX) - 0.5) * 0.004;
        for c in 0..channels {
            let (pan, rise) = match c {
                0 => (1.0, 1.0),
                1 => (0.7, 1.0),
                _ => (0.9 - 0.1 * c as f64, 1.0 + 0.15 * (c - 1) as f64),
            };
            out.push((0.25 * env * tone(rise) * pan + noise) as f32);
        }
    }
    out
}

/// A 16-bit WAV of `x`, `channels` interleaved.
fn wav(x: &[f32], channels: u16) -> Vec<u8> {
    if channels == 1 {
        return encode_wav_pcm16(x, RATE).unwrap_or_else(|e| panic!("encoding: {e}"));
    }
    let data = x.len() * 2;
    let mut b = Vec::with_capacity(44 + data);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&((36 + data) as u32).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&channels.to_le_bytes());
    b.extend_from_slice(&RATE.to_le_bytes());
    b.extend_from_slice(&(RATE * 2 * u32::from(channels)).to_le_bytes());
    b.extend_from_slice(&(2 * channels).to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(data as u32).to_le_bytes());
    for v in x {
        b.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    b
}

/// Runs ffmpeg with `args`; `None` when it fails, with its error shown.
fn run(args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new(voxr::ffmpeg_program())
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .output()
        .ok()?;
    if out.status.success() {
        Some(out.stdout)
    } else {
        eprintln!("ffmpeg {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        None
    }
}

/// `file` as ffmpeg decodes it: mono, f32, at its own rate.
fn ffmpeg_mono(file: &Path) -> Vec<f32> {
    ffmpeg_samples(file, true, None)
}

/// `file` as ffmpeg decodes it, f32 at its own rate: mono, or every
/// channel interleaved in ffmpeg's order. `decoder` names the decoder,
/// else ffmpeg picks its default one.
fn ffmpeg_samples(file: &Path, mono: bool, decoder: Option<&str>) -> Vec<f32> {
    let path = file.to_string_lossy();
    let mut args = Vec::new();
    if let Some(d) = decoder {
        args.extend(["-c:a", d]);
    }
    args.extend(["-i", path.as_ref()]);
    if mono {
        args.extend(["-ac", "1"]);
    }
    args.extend(["-f", "f32le", "-"]);
    let bytes = run(&args).unwrap_or_else(|| panic!("ffmpeg could not decode {path}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

/// The lag of `got` behind `want`, in frames, that lines them up best.
fn best_lag(want: &[f32], got: &[f32]) -> i64 {
    let n = want.len().min(got.len()).min(48_000);
    (-3000i64..=3000)
        .max_by(|&a, &b| {
            let score = |lag: i64| -> f64 {
                (0..n)
                    .filter_map(|i| {
                        let j = i as i64 + lag;
                        (j >= 0 && (j as usize) < got.len())
                            .then(|| f64::from(want[i]) * f64::from(got[j as usize]))
                    })
                    .sum()
            };
            score(a).total_cmp(&score(b))
        })
        .unwrap_or(0)
}

/// The SNR of `got` against `want`, in dB.
fn snr(want: &[f32], got: &[f32]) -> f64 {
    let (mut signal, mut error) = (0.0f64, 0.0f64);
    for (w, g) in want.iter().zip(got) {
        signal += f64::from(*w) * f64::from(*w);
        error += (f64::from(*w) - f64::from(*g)).powi(2);
    }
    if error == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (signal / error).log10()
    }
}

/// One clip: its file name, the ffmpeg encoding flags, its source's
/// channels and layout, and the SNR floor its decode must keep against
/// ffmpeg's. Empty flags take the source WAV itself, which names no
/// layout. A lossy floor sits well under what the decode reaches and
/// well over audible error.
///
/// Stereo Opus sits lowest. ffmpeg's own Opus decoder rounds stereo apart
/// from libopus, the reference decoder, by as much as voxr's decode does.
/// Against libopus that decode keeps `libopus_floor`.
struct Clip {
    name: &'static str,
    flags: &'static [&'static str],
    video: bool,
    channels: u16,
    layout: Option<&'static str>,
    floor: f64,
    libopus_floor: Option<f64>,
}

const CLIPS: &[Clip] = &[
    Clip {
        name: "aac.m4a",
        flags: &["-c:a", "aac", "-b:a", "128k"],
        video: false,
        channels: 1,
        layout: None,
        floor: 110.0,
        libopus_floor: None,
    },
    Clip {
        name: "aac-stereo.m4a",
        flags: &["-c:a", "aac", "-b:a", "160k"],
        video: false,
        channels: 2,
        layout: None,
        floor: 110.0,
        libopus_floor: None,
    },
    Clip {
        name: "aac-video.mp4",
        flags: &["-c:v", "mpeg4", "-c:a", "aac", "-shortest"],
        video: true,
        channels: 1,
        layout: None,
        floor: 110.0,
        libopus_floor: None,
    },
    Clip {
        name: "aac-video.mov",
        flags: &["-c:v", "mpeg4", "-c:a", "aac", "-shortest"],
        video: true,
        channels: 1,
        layout: None,
        floor: 110.0,
        libopus_floor: None,
    },
    Clip {
        name: "aac.mkv",
        flags: &["-c:a", "aac"],
        video: false,
        channels: 1,
        layout: None,
        floor: 110.0,
        libopus_floor: None,
    },
    Clip {
        name: "alac.m4a",
        flags: &["-c:a", "alac"],
        video: false,
        channels: 1,
        layout: None,
        floor: 120.0,
        libopus_floor: None,
    },
    Clip {
        name: "opus.webm",
        flags: &["-c:a", "libopus", "-b:a", "96k"],
        video: false,
        channels: 1,
        layout: None,
        floor: 90.0,
        libopus_floor: None,
    },
    Clip {
        name: "opus-stereo.webm",
        flags: &["-c:a", "libopus", "-b:a", "96k"],
        video: false,
        channels: 2,
        layout: None,
        floor: 55.0,
        libopus_floor: Some(75.0),
    },
    Clip {
        name: "opus.opus",
        flags: &["-c:a", "libopus", "-b:a", "64k"],
        video: false,
        channels: 1,
        layout: None,
        floor: 90.0,
        libopus_floor: None,
    },
    Clip {
        name: "vorbis.webm",
        flags: &["-c:a", "libvorbis", "-q:a", "5"],
        video: false,
        channels: 1,
        layout: None,
        floor: 110.0,
        libopus_floor: None,
    },
    Clip {
        name: "mp3.mp3",
        flags: &["-c:a", "libmp3lame", "-b:a", "128k"],
        video: false,
        channels: 1,
        layout: None,
        floor: 110.0,
        libopus_floor: None,
    },
    Clip {
        name: "quad.flac",
        flags: &["-c:a", "flac"],
        video: false,
        channels: 4,
        layout: Some("quad"),
        floor: 120.0,
        libopus_floor: None,
    },
    Clip {
        name: "5.0.flac",
        flags: &["-c:a", "flac"],
        video: false,
        channels: 5,
        layout: Some("5.0"),
        floor: 120.0,
        libopus_floor: None,
    },
    Clip {
        name: "5.1.flac",
        flags: &["-c:a", "flac"],
        video: false,
        channels: 6,
        layout: Some("5.1"),
        floor: 120.0,
        libopus_floor: None,
    },
    Clip {
        name: "quad.wav",
        flags: &["-c:a", "pcm_s16le"],
        video: false,
        channels: 4,
        layout: Some("quad"),
        floor: 120.0,
        libopus_floor: None,
    },
    Clip {
        name: "5.0.wav",
        flags: &["-c:a", "pcm_s16le"],
        video: false,
        channels: 5,
        layout: Some("5.0"),
        floor: 120.0,
        libopus_floor: None,
    },
    Clip {
        name: "5.1.wav",
        flags: &["-c:a", "pcm_s16le"],
        video: false,
        channels: 6,
        layout: Some("5.1"),
        floor: 120.0,
        libopus_floor: None,
    },
    Clip {
        name: "4ch-plain.wav",
        flags: &[],
        video: false,
        channels: 4,
        layout: None,
        floor: 120.0,
        libopus_floor: None,
    },
    Clip {
        name: "6ch-plain.wav",
        flags: &[],
        video: false,
        channels: 6,
        layout: None,
        floor: 120.0,
        libopus_floor: None,
    },
];

/// The clip encoded from `src`, or `None` when this ffmpeg cannot.
fn encode(dir: &Path, src: &Path, clip: &Clip) -> Option<PathBuf> {
    let out = dir.join(clip.name);
    let src = src.to_string_lossy();
    let out_text = out.to_string_lossy();
    if clip.flags.is_empty() {
        return std::fs::copy(src.as_ref(), &out).ok().map(|_| out.clone());
    }
    let mut args: Vec<&str> = Vec::new();
    if clip.video {
        args.extend(["-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=3"]);
    }
    if let Some(layout) = clip.layout {
        args.extend(["-channel_layout", layout]);
    }
    args.extend(["-i", &src]);
    if clip.video {
        args.extend(["-map", "0:v", "-map", "1:a"]);
    }
    args.extend(clip.flags);
    args.push(&out_text);
    run(&args).map(|_| out)
}

#[test]
fn every_container_decodes_as_ffmpeg_decodes_it() {
    if run(&["-version"]).is_none() {
        eprintln!("skipping: ffmpeg unavailable; set FFMPEG or put ffmpeg on PATH");
        return;
    }
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("temporary directory: {e}"));
    let mut sources: std::collections::BTreeMap<u16, PathBuf> = Default::default();
    let mut failures = Vec::new();
    for clip in CLIPS {
        let src = sources.entry(clip.channels).or_insert_with(|| {
            let path = dir.path().join(format!("source-{}.wav", clip.channels));
            let pcm = source(usize::from(clip.channels));
            std::fs::write(&path, wav(&pcm, clip.channels)).unwrap_or_else(|e| panic!("{e}"));
            path
        });
        let Some(file) = encode(dir.path(), src, clip) else {
            eprintln!("skipping {}: this ffmpeg cannot encode it", clip.name);
            continue;
        };
        let want = ffmpeg_mono(&file);
        let bytes = std::fs::read(&file).unwrap_or_else(|e| panic!("{e}"));
        let ext = clip.name.rsplit('.').next();
        let decoded = match decode_audio_like_ffmpeg(&bytes, ext) {
            Ok(d) => d,
            Err(e) => {
                failures.push(format!("{}: {e}", clip.name));
                continue;
            }
        };
        if decoded.channels != clip.channels {
            failures.push(format!(
                "{}: {} channels, the source has {}",
                clip.name, decoded.channels, clip.channels
            ));
            continue;
        }
        let got = match to_mono_like_ffmpeg(&decoded.samples, decoded.channels, decoded.layout) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{}: {e}", clip.name));
                continue;
            }
        };
        let db = snr(&want, &got);
        eprintln!(
            "{}: {} Hz, layout {:?}, ffmpeg {} frames, voxr {} frames, SNR {db:.1} dB",
            clip.name,
            decoded.sample_rate,
            decoded.layout,
            want.len(),
            got.len(),
        );
        if got.len() != want.len() || db < clip.floor {
            // The lag names a start trim that differs from ffmpeg's.
            failures.push(format!(
                "{}: {} frames, ffmpeg {}, SNR {db:.1} dB (floor {} dB), lag {}",
                clip.name,
                got.len(),
                want.len(),
                clip.floor,
                best_lag(&want, &got)
            ));
        }
        if clip.channels > 1 {
            let n = usize::from(clip.channels);
            let every = ffmpeg_samples(&file, false, None);
            let channel = |x: &[f32], c: usize| -> Vec<f32> {
                x.iter().skip(c).step_by(n).copied().collect()
            };
            for c in 0..n {
                let db = snr(&channel(&every, c), &channel(&decoded.samples, c));
                eprintln!("{} channel {c}: SNR {db:.1} dB", clip.name);
                if db < clip.floor {
                    failures.push(format!(
                        "{} channel {c}: SNR {db:.1} dB under {} dB",
                        clip.name, clip.floor
                    ));
                }
            }
            if let Some(floor) = clip.libopus_floor {
                match run(&["-hide_banner", "-decoders"])
                    .filter(|d| String::from_utf8_lossy(d).contains(" libopus "))
                {
                    None => eprintln!(
                        "{}: no libopus decoder, its reference check is skipped",
                        clip.name
                    ),
                    Some(_) => {
                        let reference = ffmpeg_samples(&file, false, Some("libopus"));
                        for c in 0..n {
                            let db = snr(&channel(&reference, c), &channel(&decoded.samples, c));
                            eprintln!("{} channel {c} against libopus: SNR {db:.1} dB", clip.name);
                            if db < floor {
                                failures.push(format!(
                                    "{} channel {c}: SNR {db:.1} dB against libopus, under {floor} dB",
                                    clip.name
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
