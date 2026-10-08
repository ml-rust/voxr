//! The fallback for a file voxr does not decode: the skill's own ffmpeg
//! and ffprobe commands, so the result matches the skill by construction.
//! It runs only on voxr's "unsupported" refusal, never on a corrupt or
//! truncated file.

use std::ffi::OsStr;
use std::path::Path;

use super::tools::{Tools, run};
use crate::error::{Error, Result};

/// The file at `path` as mono samples at `rate`, through the skill's
/// decode: `ffmpeg -hide_banner -loglevel error -i <file> -ac 1 -ar <rate>
/// -f f32le -`.
pub(super) fn decode(tools: &Tools, path: &Path, rate: u32, refusal: &str) -> Result<Vec<f32>> {
    let rate = rate.to_string();
    let args: [&OsStr; 12] = [
        "-hide_banner".as_ref(),
        "-loglevel".as_ref(),
        "error".as_ref(),
        "-i".as_ref(),
        path.as_os_str(),
        "-ac".as_ref(),
        "1".as_ref(),
        "-ar".as_ref(),
        rate.as_ref(),
        "-f".as_ref(),
        "f32le".as_ref(),
        "-".as_ref(),
    ];
    let bytes = run(&tools.ffmpeg, &args, path, refusal)?;
    if bytes.is_empty() {
        return Err(Error::DataError {
            reason: format!(
                "decoding {}: voxr refuses it ({refusal}), and ffmpeg decoded no audio from it. Pass a file with an audio track",
                path.display()
            ),
        });
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect())
}

/// The duration of the file at `path`, through the skill's probe:
/// `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1
/// <file>`.
pub(super) fn duration(tools: &Tools, path: &Path, refusal: &str) -> Result<f64> {
    let args: [&OsStr; 7] = [
        "-v".as_ref(),
        "error".as_ref(),
        "-show_entries".as_ref(),
        "format=duration".as_ref(),
        "-of".as_ref(),
        "default=nw=1:nk=1".as_ref(),
        path.as_os_str(),
    ];
    let out = run(&tools.ffprobe, &args, path, refusal)?;
    let text = String::from_utf8_lossy(&out);
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|d| d.is_finite() && *d > 0.0)
        .ok_or_else(|| Error::DataError {
            reason: format!(
                "probing {}: voxr refuses it ({refusal}), and ffprobe reports no duration (\"{}\"). Re-export the file as WAV or MP3",
                path.display(),
                text.trim()
            ),
        })
}

#[cfg(test)]
mod tests {
    //! The fallback end to end: files voxr refuses, made by ffmpeg (and by
    //! GStreamer's FDK encoder for HE-AAC, which ffmpeg's own AAC encoder
    //! does not write) in a temporary directory, then decoded through
    //! [`decode_at`](super::super::decode::decode_at). A test whose tool is missing
    //! prints a skip notice and passes.

    use std::process::Command;

    use super::super::decode::{Native, decode_at, decode_mono, probe_with};
    use super::*;

    /// Runs `program` with `args`. `None`, with the reason shown, when it
    /// does not run or fails.
    fn tool(program: &OsStr, args: &[&str]) -> Option<Vec<u8>> {
        let out = Command::new(program).args(args).output().ok()?;
        if out.status.success() {
            Some(out.stdout)
        } else {
            eprintln!(
                "{} {args:?}: {}",
                program.to_string_lossy(),
                String::from_utf8_lossy(&out.stderr)
            );
            None
        }
    }

    /// The tools, or `None` with a skip notice when ffmpeg does not run.
    fn tools() -> Option<Tools> {
        let tools = Tools::from_env();
        if tool(&tools.ffmpeg, &["-version"]).is_none() {
            eprintln!("skipping: ffmpeg unavailable. Set FFMPEG or put ffmpeg on PATH");
            return None;
        }
        Some(tools)
    }

    /// `path` as the skill's command decodes it, run here directly.
    fn skill_decode(tools: &Tools, path: &Path, rate: u32) -> Vec<f32> {
        let path = path.to_string_lossy();
        let rate = rate.to_string();
        let args = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            &path,
            "-ac",
            "1",
            "-ar",
            &rate,
            "-f",
            "f32le",
            "-",
        ];
        let bytes = tool(&tools.ffmpeg, &args).unwrap_or_else(|| panic!("ffmpeg decoding {path}"));
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect()
    }

    /// Checks voxr refuses `path` as unsupported, then that the fallback
    /// decodes it at 48 and 16 kHz exactly as the skill's command does.
    fn falls_back(tools: &Tools, path: &Path) -> Result<()> {
        let refused = matches!(decode_mono(path)?, Native::Unsupported(_));
        assert!(refused, "voxr decodes {} itself", path.display());
        for rate in [48_000, 16_000] {
            let got = decode_at(path, rate, tools)?;
            assert!(!got.is_empty());
            assert_eq!(
                got,
                skill_decode(tools, path, rate),
                "{} at {rate} Hz",
                path.display()
            );
        }
        assert!(probe_with(path, tools)? > 1.0);
        Ok(())
    }

    fn temp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap_or_else(|e| panic!("temporary directory: {e}"))
    }

    #[test]
    fn surround_aac_decodes_as_the_skill_decodes_it() -> Result<()> {
        let Some(tools) = tools() else { return Ok(()) };
        let dir = temp();
        let file = dir.path().join("surround.m4a");
        let tones = "sin(2*PI*220*t)/4|sin(2*PI*330*t)/4|sin(2*PI*440*t)/4|sin(2*PI*55*t)/4|sin(2*PI*550*t)/4|sin(2*PI*660*t)/4";
        let source = format!("aevalsrc={tones}:c=5.1:s=48000:d=1.5");
        let out = file.to_string_lossy();
        let made = tool(
            &tools.ffmpeg,
            &[
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &source,
                "-c:a",
                "aac",
                "-b:a",
                "384k",
                &out,
            ],
        );
        assert!(made.is_some(), "ffmpeg could not write 5.1 AAC");
        falls_back(&tools, &file)
    }

    #[test]
    fn he_aac_decodes_as_the_skill_decodes_it() -> Result<()> {
        let Some(tools) = tools() else { return Ok(()) };
        let gst = OsStr::new("gst-launch-1.0");
        let inspect = Command::new("gst-inspect-1.0").arg("fdkaacenc").output();
        if !inspect.is_ok_and(|o| o.status.success()) {
            eprintln!("skipping: GStreamer's fdkaacenc, the HE-AAC encoder, is unavailable");
            return Ok(());
        }
        let dir = temp();
        let wav = dir.path().join("source.wav");
        let adts = dir.path().join("he.aac");
        let source = "aevalsrc=sin(2*PI*220*t)/4|sin(2*PI*330*t)/4:c=stereo:s=48000:d=2";
        let wav_text = wav.to_string_lossy();
        let made = tool(
            &tools.ffmpeg,
            &[
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                source,
                &wav_text,
            ],
        );
        assert!(made.is_some(), "ffmpeg could not write the source");
        let location = format!("location={wav_text}");
        let sink = format!("location={}", adts.to_string_lossy());
        let encoded = tool(
            gst,
            &[
                "-q",
                "filesrc",
                &location,
                "!",
                "wavparse",
                "!",
                "audioconvert",
                "!",
                "fdkaacenc",
                "bitrate=48000",
                "!",
                "audio/mpeg,profile=he-aac-v1,stream-format=adts",
                "!",
                "filesink",
                &sink,
            ],
        );
        assert!(encoded.is_some(), "fdkaacenc could not write HE-AAC");
        // ffmpeg's MP4 copy keeps the implicit SBR signal: the AAC
        // configuration names the 24 kHz core, the track counts at 48 kHz.
        for name in ["he.m4a", "he.aac"] {
            let file = dir.path().join(name);
            if name.ends_with(".m4a") {
                let out = file.to_string_lossy();
                let input = adts.to_string_lossy();
                let remuxed = tool(
                    &tools.ffmpeg,
                    &[
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        &input,
                        "-c",
                        "copy",
                        &out,
                    ],
                );
                assert!(remuxed.is_some(), "ffmpeg could not remux HE-AAC");
            }
            falls_back(&tools, &file)?;
        }
        Ok(())
    }

    /// A WAV of WMA, a format voxr refuses as unsupported.
    fn wma_wav() -> Vec<u8> {
        let mut fmt = Vec::new();
        for v in [0x0161u16, 2] {
            fmt.extend_from_slice(&v.to_le_bytes());
        }
        fmt.extend_from_slice(&48_000u32.to_le_bytes());
        fmt.extend_from_slice(&16_000u32.to_le_bytes());
        for v in [4u16, 16] {
            fmt.extend_from_slice(&v.to_le_bytes());
        }
        let data = [0u8; 64];
        let mut b = b"RIFF".to_vec();
        b.extend_from_slice(&((4 + 8 + fmt.len() + 8 + data.len()) as u32).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        b.extend_from_slice(&fmt);
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(&data);
        b
    }

    #[test]
    fn without_ffmpeg_the_error_names_the_file_the_refusal_and_the_fix() -> Result<()> {
        let dir = temp();
        let file = dir.path().join("speech.wav");
        std::fs::write(&file, wma_wav()).map_err(|e| Error::File {
            reason: e.to_string(),
        })?;
        let absent = dir.path().join("no-ffmpeg");
        let tools = Tools {
            ffmpeg: absent.clone().into_os_string(),
            ffprobe: absent.clone().into_os_string(),
        };
        for err in [
            decode_at(&file, 48_000, &tools).err(),
            probe_with(&file, &tools).err(),
        ] {
            let text = err.map(|e| e.to_string()).unwrap_or_default();
            assert!(
                text.contains(&format!("decoding {}: voxr refuses it (", file.display())),
                "{text}"
            );
            assert!(text.contains("unsupported"), "{text}");
            assert!(
                text.contains(&format!(
                    "{}, which reads what voxr does not, did not run: ",
                    absent.display()
                )),
                "{text}"
            );
            assert!(
                text.ends_with("Install ffmpeg, or set FFMPEG and FFPROBE to its programs"),
                "{text}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_truncated_file_keeps_its_own_error() -> Result<()> {
        let dir = temp();
        let file = dir.path().join("cut.wav");
        // A WAV header whose `fmt ` chunk stops after 4 of its 16 bytes.
        let mut cut = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00".to_vec();
        cut.extend_from_slice(&[1, 0, 1, 0]);
        std::fs::write(&file, cut).map_err(|e| Error::File {
            reason: e.to_string(),
        })?;
        let absent = dir.path().join("no-ffmpeg");
        let tools = Tools {
            ffmpeg: absent.clone().into_os_string(),
            ffprobe: absent.into_os_string(),
        };
        let text = decode_at(&file, 48_000, &tools)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            text.contains(&format!("decoding {}: ", file.display())),
            "{text}"
        );
        assert!(!text.contains("voxr refuses it"), "{text}");
        Ok(())
    }
}
