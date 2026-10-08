//! The field readers a manifest is read with, as the skill's
//! `loadManifest` reads its fields, and the voiceover.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::js_value::{js_display, js_stringify};
use super::outcome::CommandError;
use super::paths::resolve;
use crate::transcript::Voiceover;

/// Seconds a voiceover may run past the video's end before it overruns.
const OVERRUN_SLACK: f64 = 0.01;

/// A duration probe: an audio file's length in seconds, as ffprobe
/// reports it.
pub(super) type Probe<'a> = dyn FnMut(&Path) -> crate::error::Result<f64> + 'a;

/// `path.basename(file, path.extname(file))`: the file name without its
/// last extension. A leading dot starts no extension.
pub(super) fn scene_name(file: &Path) -> String {
    let base = file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    match base.rfind('.') {
        Some(i) if i > 0 => base[..i].to_string(),
        _ => base,
    }
}

/// A bad manifest, as the skill's `fail(path, msg)` words it.
pub(super) fn fail(path: &Path, msg: impl std::fmt::Display) -> CommandError {
    CommandError::Input(format!("{}: {msg}", path.display()))
}

/// A field's value, `undefined` and `null` alike as `None`: the skill's
/// `?? default` and `!= null`.
pub(super) fn field<'v>(v: &'v Value, key: &str) -> Option<&'v Value> {
    v.get(key).filter(|x| !x.is_null())
}

/// Reads `path` as JSON, the skill's error when it is not.
pub(super) fn read_json(path: &Path) -> Result<Value, CommandError> {
    let bytes = std::fs::read(path).map_err(|e| {
        CommandError::Failed(format!(
            "reading {}: {e}. Check the file is readable",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|e| fail(path, format!("invalid JSON: {e}")))
}

/// The path field `name` holds, resolved against the manifest's directory
/// `root`, which must exist. A value that is not a string is refused.
pub(super) fn existing(
    root: &Path,
    manifest: &Path,
    v: &Value,
    name: &str,
) -> Result<PathBuf, CommandError> {
    let path = match v {
        Value::String(s) => resolve(root, Path::new(s)),
        other => {
            return Err(fail(
                manifest,
                format!(
                    "{name} is {}. It must be a path, as a string.",
                    js_stringify(other)
                ),
            ));
        }
    };
    if path.exists() {
        Ok(path)
    } else {
        Err(fail(
            manifest,
            format!("{name} {} does not exist.", path.display()),
        ))
    }
}

/// The duration of the audio file at `file`, a bad input when it cannot
/// be read.
pub(super) fn duration(probe: &mut Probe<'_>, file: &Path) -> Result<f64, CommandError> {
    probe(file).map_err(|e| {
        CommandError::Input(format!(
            "cannot read {}: {}. Check the file is a valid audio file.",
            file.display(),
            e.reason()
        ))
    })
}

/// The voiceover: `"vo.wav"`, or `{ "file", "start", "script" }`, and how
/// far it runs past `total`, the video's end.
pub(super) fn voiceover(
    spec: &Value,
    root: &Path,
    path: &Path,
    total: f64,
    probe: &mut Probe<'_>,
) -> Result<(Voiceover, f64), CommandError> {
    let (file_value, start, script) = match spec {
        Value::String(_) => (Some(spec), None, None),
        other => (
            field(other, "file"),
            field(other, "start"),
            field(other, "script"),
        ),
    };
    let file_text = file_value.map(js_display).unwrap_or_default();
    let file = resolve(root, Path::new(&file_text));
    let named = file_value.is_some_and(|v| !matches!(v, Value::String(s) if s.is_empty()));
    if !named || !file.exists() {
        return Err(fail(
            path,
            format!("\"voiceover\" file {} does not exist.", file.display()),
        ));
    }
    let at = match start {
        None => 0.0,
        Some(v) => v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
            fail(
                path,
                format!(
                    "\"voiceover.start\" is {}. Use video seconds: 0.5 starts it half a second in, -2 skips its first 2 seconds.",
                    js_stringify(v)
                ),
            )
        })?,
    };
    let length = duration(probe, &file)?;
    let overrun = if at + length > total + OVERRUN_SLACK {
        at + length - total
    } else {
        0.0
    };
    let script = script
        .map(|v| existing(root, path, v, "\"voiceover.script\""))
        .transpose()?;
    Ok((Voiceover { file, at, script }, overrun))
}
