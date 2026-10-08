//! A video manifest (`video.json`) read for its narration: where each
//! scene starts on the video clock, the voiceover and its start, each
//! narration's script, and how far the voiceover runs past the video's
//! end. The checks and messages are the skill's `loadManifest`, for the
//! fields that place narration.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::js_value::{is_integer, js_display, number};
use super::manifest_fields::{
    Probe, duration, existing, fail, field, read_json, scene_name, voiceover,
};
use super::manifest_sound::{music, scene_sound, sound};
use super::outcome::CommandError;
use super::paths::resolve;
use crate::js::{js_round, js_string, js_to_fixed_text};
use crate::transcript::{AUDIO_LEAD, AUDIO_TAIL, NarrationManifest, SceneNarration};

/// The manifest's path when `--manifest` is not given.
pub const DEFAULT_MANIFEST: &str = "video.json";

/// A manifest read for its narration.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedManifest {
    /// The manifest file, resolved.
    pub path: PathBuf,
    /// The narration it places.
    pub narration: NarrationManifest,
    /// Seconds the voiceover runs past the video's end; 0 when it fits.
    pub voiceover_overrun: f64,
}

/// One scene as the placement reads it.
struct Scene {
    name: String,
    duration: f64,
    audio: Option<PathBuf>,
    script: Option<PathBuf>,
}

/// What every scene is read against.
struct Context<'a> {
    path: &'a Path,
    root: &'a Path,
    fps: f64,
    /// Seconds in a bar, from `out/beats.json`: NaN when no scene uses bars.
    bar: f64,
}

/// The frame rate, once the frame size and rate check out.
fn frame_rate(raw: &Value, path: &Path) -> Result<f64, CommandError> {
    let width = field(raw, "width").cloned().unwrap_or(Value::from(1920));
    let height = field(raw, "height").cloned().unwrap_or(Value::from(1080));
    let fps = field(raw, "fps").cloned().unwrap_or(Value::from(30));
    for (k, v) in [("width", &width), ("height", &height)] {
        let even = number(v).is_some_and(|x| x >= 2.0 && x % 2.0 == 0.0);
        if !is_integer(v) || !even {
            return Err(fail(
                path,
                format!(
                    "\"{k}\" is {}. It must be a positive even integer, because H.264 yuv420p rejects odd sizes.",
                    js_display(v)
                ),
            ));
        }
    }
    let rate = number(&fps).unwrap_or(f64::NAN);
    if !is_integer(&fps) || !(12.0..=120.0).contains(&rate) {
        return Err(fail(
            path,
            format!(
                "\"fps\" is {}. Use an integer from 12 to 120.",
                js_display(&fps)
            ),
        ));
    }
    Ok(rate)
}

/// Seconds in a bar when a scene uses bars: four beats of `out/beats.json`.
fn bar_length(list: &[Value], out_dir: &Path, path: &Path) -> Result<f64, CommandError> {
    if !list.iter().any(|s| field(s, "bars").is_some()) {
        return Ok(f64::NAN);
    }
    let beats = out_dir.join("beats.json");
    if !beats.exists() {
        return Err(fail(
            path,
            format!(
                "a scene uses \"bars\" but {} does not exist. Run \"beats\" first.",
                beats.display()
            ),
        ));
    }
    let parsed = read_json(&beats)?;
    Ok(parsed
        .get("beat")
        .and_then(Value::as_f64)
        .unwrap_or(f64::NAN)
        * 4.0)
}

/// Scene `i`: its name, its length (given, from bars, or from its audio),
/// its audio and its script.
fn scene(
    i: usize,
    s: &Value,
    cx: &Context<'_>,
    names: &mut Vec<String>,
    probe: &mut Probe<'_>,
) -> Result<Scene, CommandError> {
    let path = cx.path;
    let at = format!("scenes[{i}]");
    let Some(file) = s.get("file").and_then(Value::as_str) else {
        return Err(fail(
            path,
            format!("{at}.file must be a path to an .html file."),
        ));
    };
    let name = scene_name(&resolve(cx.root, Path::new(file)));
    if names.contains(&name) {
        return Err(fail(
            path,
            format!("{at}: scene name \"{name}\" repeats. Scene file basenames must be unique."),
        ));
    }
    names.push(name.clone());
    let audio = field(s, "audio")
        .map(|v| existing(cx.root, path, v, &format!("{at}.audio")))
        .transpose()?;
    let audio_duration = match &audio {
        Some(a) => duration(probe, a)?,
        None => 0.0,
    };
    let script = field(s, "script")
        .map(|v| existing(cx.root, path, v, &format!("{at}.script")))
        .transpose()?;
    let bars = field(s, "bars");
    if field(s, "duration").is_some() && bars.is_some() {
        return Err(fail(
            path,
            format!("{at} sets both \"duration\" and \"bars\". Keep one."),
        ));
    }
    let given = match bars {
        Some(b) => Some(Value::from(number(b).unwrap_or(f64::NAN) * cx.bar)),
        None => field(s, "duration").cloned(),
    };
    if bars.is_some_and(|b| !number(b).is_some_and(|x| x > 0.0)) {
        return Err(fail(path, format!("{at}.bars must be a positive number.")));
    }
    let needed = if audio.is_some() {
        AUDIO_LEAD + audio_duration + AUDIO_TAIL
    } else {
        0.0
    };
    let length = match given {
        None if audio.is_none() => {
            return Err(fail(
                path,
                format!(
                    "{at} needs \"duration\" in seconds, \"bars\", or \"audio\" to derive it from."
                ),
            ));
        }
        None => (needed * cx.fps).ceil() / cx.fps,
        // `typeof duration !== 'number' || duration <= 0`: NaN passes.
        Some(v) => match v.as_f64() {
            Some(d) if d > 0.0 || d.is_nan() => d,
            _ => {
                return Err(fail(
                    path,
                    format!("{at}.duration must be a positive number of seconds."),
                ));
            }
        },
    };
    if audio.is_some() && length < needed {
        return Err(fail(
            path,
            format!(
                "{at} lasts {}s but the narration needs {}s ({}s lead + {}s audio + {}s tail). Remove \"duration\" to derive it, or raise it.",
                js_to_fixed_text(length, 2),
                js_to_fixed_text(needed, 2),
                js_string(AUDIO_LEAD),
                js_to_fixed_text(audio_duration, 2),
                js_string(AUDIO_TAIL)
            ),
        ));
    }
    scene_sound(i, s, &at, length, path)?;
    Ok(Scene {
        name,
        duration: length,
        audio,
        script,
    })
}

/// The skill's `loadManifest(path, { voiceoverFits: false })`, for the
/// fields that place narration: the frame size and rate, every scene's
/// file, name, audio, script, duration or bars, and the voiceover.
///
/// `probe` reads an audio file's duration in seconds, as ffprobe reports
/// it. A voiceover running past the video's end is reported in
/// [`LoadedManifest::voiceover_overrun`], not refused: transcribe reads the
/// words that size the scenes.
///
/// Every refusal is a bad input naming the manifest and the field. The
/// sound fields (`energy`, `in`, `inVol`, `inDur`, `music`, `sound`) are
/// checked as the skill checks them, though transcribe reads none of
/// them.
pub fn load_manifest(
    manifest: &Path,
    cwd: &Path,
    probe: &mut dyn FnMut(&Path) -> crate::error::Result<f64>,
) -> Result<LoadedManifest, CommandError> {
    let path = resolve(cwd, manifest);
    if !path.exists() {
        return Err(CommandError::Input(format!(
            "{} not found. Run \"init <dir>\" or pass the manifest path.",
            path.display()
        )));
    }
    let raw = read_json(&path)?;
    let root = path
        .parent()
        .map_or_else(|| cwd.to_path_buf(), Path::to_path_buf);
    let fps = frame_rate(&raw, &path)?;
    let Some(list) = raw
        .get("scenes")
        .and_then(Value::as_array)
        .filter(|l| !l.is_empty())
    else {
        return Err(fail(&path, "\"scenes\" must be a non-empty array."));
    };
    let out_dir = resolve(&root, Path::new("out"));
    let cx = Context {
        path: &path,
        root: &root,
        fps,
        bar: bar_length(list, &out_dir, &path)?,
    };
    let mut names = Vec::new();
    let scenes = list
        .iter()
        .enumerate()
        .map(|(i, s)| scene(i, s, &cx, &mut names, probe))
        .collect::<Result<Vec<_>, _>>()?;
    // Frame boundaries round the running total, so rounding never drifts it.
    let mut t = 0.0;
    let mut placed = Vec::with_capacity(scenes.len());
    for s in scenes {
        let a = js_round(t * fps);
        t += s.duration;
        placed.push(SceneNarration {
            name: s.name,
            start: a / fps,
            audio: s.audio,
            script: s.script,
        });
    }
    let total = js_round(t * fps) / fps;
    music(&raw, &root, &path)?;
    let (voiceover, voiceover_overrun) = match field(&raw, "voiceover") {
        Some(spec) => {
            let (v, overrun) = voiceover(spec, &root, &path, total, probe)?;
            (Some(v), overrun)
        }
        None => (None, 0.0),
    };
    sound(&raw, &path)?;
    Ok(LoadedManifest {
        path,
        narration: NarrationManifest {
            out_dir,
            voiceover,
            scenes: placed,
        },
        voiceover_overrun,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dir(tempfile::TempDir);

    impl Dir {
        fn new() -> Self {
            Self(tempfile::tempdir().unwrap_or_else(|e| panic!("temporary directory: {e}")))
        }
        fn write(&self, name: &str, text: &str) -> PathBuf {
            let p = self.0.path().join(name);
            if let Some(d) = p.parent() {
                std::fs::create_dir_all(d).unwrap_or_else(|e| panic!("{e}"));
            }
            std::fs::write(&p, text).unwrap_or_else(|e| panic!("{e}"));
            p
        }
        fn load(&self) -> Result<LoadedManifest, CommandError> {
            let mut probe = |p: &Path| -> crate::error::Result<f64> {
                Ok(std::fs::metadata(p).map(|m| m.len() as f64).unwrap_or(0.0))
            };
            load_manifest(Path::new("video.json"), self.0.path(), &mut probe)
        }
    }

    #[test]
    fn scenes_start_on_frame_boundaries_and_narration_follows() -> Result<(), CommandError> {
        let d = Dir::new();
        // Each audio file's probed duration is its byte length.
        d.write("a.wav", "12");
        d.write("vo.wav", "123456");
        d.write("vo.txt", "Hi.");
        d.write(
            "video.json",
            r#"{"fps": 30, "scenes": [{"file": "scenes/01-hook.html", "duration": 1.01},
                {"file": "scenes/02-b.html", "audio": "a.wav"}],
                "voiceover": {"file": "vo.wav", "start": 0.5, "script": "vo.txt"}}"#,
        );
        let m = d.load()?;
        let s = &m.narration.scenes;
        assert_eq!((s[0].name.as_str(), s[0].start), ("01-hook", 0.0));
        assert_eq!(s[1].start, 1.0);
        assert!(s[1].audio.as_deref().is_some_and(|a| a.ends_with("a.wav")));
        let v = m.narration.voiceover.as_ref().map(|v| v.at);
        assert_eq!(v, Some(0.5));
        // 1.01 rounds to 30 frames, then 2.8 s of scene: the video ends at
        // 3.8 s, and the voiceover at 6.5 s.
        assert!(
            (m.voiceover_overrun - 2.7).abs() < 1e-9,
            "{}",
            m.voiceover_overrun
        );
        Ok(())
    }

    #[test]
    fn refusals_name_the_manifest_and_the_field() {
        let d = Dir::new();
        assert!(
            matches!(d.load(), Err(CommandError::Input(m)) if m.ends_with("video.json not found. Run \"init <dir>\" or pass the manifest path."))
        );
        d.write("video.json", r#"{"width": 1921, "scenes": []}"#);
        assert!(
            matches!(d.load(), Err(CommandError::Input(m)) if m.ends_with("\"width\" is 1921. It must be a positive even integer, because H.264 yuv420p rejects odd sizes."))
        );
        d.write("video.json", r#"{"scenes": []}"#);
        assert!(
            matches!(d.load(), Err(CommandError::Input(m)) if m.ends_with("\"scenes\" must be a non-empty array."))
        );
        d.write("video.json", r#"{"scenes": [{"file": "a.html"}]}"#);
        assert!(
            matches!(d.load(), Err(CommandError::Input(m)) if m.ends_with("scenes[0] needs \"duration\" in seconds, \"bars\", or \"audio\" to derive it from."))
        );
        d.write("video.json", r#"{"scenes": [{"file": "a.html", "duration": 1}], "voiceover": {"file": "vo.wav", "start": "x"}}"#);
        assert!(
            matches!(d.load(), Err(CommandError::Input(m)) if m.contains("\"voiceover\" file") && m.ends_with("vo.wav does not exist."))
        );
        d.write("vo.wav", "1");
        assert!(
            matches!(d.load(), Err(CommandError::Input(m)) if m.ends_with("\"voiceover.start\" is \"x\". Use video seconds: 0.5 starts it half a second in, -2 skips its first 2 seconds."))
        );
        d.write("a.wav", "123");
        d.write(
            "video.json",
            r#"{"scenes": [{"file": "a.html", "audio": "a.wav", "duration": 1}]}"#,
        );
        assert!(
            matches!(d.load(), Err(CommandError::Input(m)) if m.ends_with("scenes[0] lasts 1.00s but the narration needs 3.80s (0.3s lead + 3.00s audio + 0.5s tail). Remove \"duration\" to derive it, or raise it."))
        );
    }

    #[test]
    fn names_drop_their_last_extension() {
        assert_eq!(scene_name(Path::new("/a/01-hook.html")), "01-hook");
        assert_eq!(scene_name(Path::new("/a/.hidden")), ".hidden");
        assert_eq!(scene_name(Path::new("/a/x.y.html")), "x.y");
    }
}
