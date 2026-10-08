//! The sound direction a manifest carries, checked as the skill's
//! `loadManifest` checks it: each scene's energy and transition, the
//! music, and the `sound` block with its bed. Transcribe reads none of it,
//! and still refuses a manifest the skill refuses.

use std::path::Path;

use serde_json::Value;

use super::js_value::{js_display, js_stringify, number};
use super::manifest_fields::{fail, field};
use super::outcome::CommandError;
use super::paths::resolve;
use crate::js::{js_string, js_to_fixed_text};
use crate::pattern::trim;

/// The transitions a scene's `in` names.
const TRANSITIONS: [&str; 4] = ["whoosh", "swoosh", "reverse", "riser"];

/// The sound palettes: `synth`, then each struck material, as the skill's
/// `PALETTES` lists them.
const PALETTES: [&str; 8] = [
    "synth", "wood", "glass", "metal", "plastic", "stone", "rubber", "ceramic",
];

/// The spaces the effects sit in.
const SPACES: [&str; 3] = ["tight", "room", "hall"];

/// The roots a key names: the skill's `PITCH_CLASS`.
const ROOTS: [&str; 17] = [
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
];

/// `typeof v === 'number' && lo <= v && v <= hi`.
fn within(v: &Value, lo: f64, hi: f64) -> bool {
    number(v).is_some_and(|x| x >= lo && x <= hi)
}

/// Scene `at`'s energy (one level, or `[seconds, energy]` points in time
/// order inside its `duration`), its transition `in` and the transition's
/// `inVol` and `inDur`.
pub(super) fn scene_sound(
    i: usize,
    s: &Value,
    at: &str,
    duration: f64,
    path: &Path,
) -> Result<(), CommandError> {
    let energy = field(s, "energy").cloned().unwrap_or(Value::from(0.6));
    let unit = |e: &Value| within(e, 0.0, 1.0);
    match &energy {
        Value::Number(_) => {
            if !unit(&energy) {
                return Err(fail(
                    path,
                    format!(
                        "{at}.energy is {}. Use a number from 0 to 1, or a list of [seconds, energy] points.",
                        js_display(&energy)
                    ),
                ));
            }
        }
        Value::Array(points) if !points.is_empty() => {
            for (k, p) in points.iter().enumerate() {
                let pair = p.as_array().filter(|a| a.len() == 2);
                let ok = pair.is_some_and(|a| a[0].is_number() && unit(&a[1]));
                let Some(pair) = pair.filter(|_| ok) else {
                    return Err(fail(
                        path,
                        format!(
                            "{at}.energy[{k}] is {}. Use [seconds into the scene, energy 0-1].",
                            js_stringify(p)
                        ),
                    ));
                };
                let t = number(&pair[0]).unwrap_or(f64::NAN);
                if t < 0.0 || t > duration {
                    return Err(fail(
                        path,
                        format!(
                            "{at}.energy[{k}] sits at {}s, outside the scene's 0-{}s.",
                            js_string(t),
                            js_to_fixed_text(duration, 2)
                        ),
                    ));
                }
                if k > 0 {
                    let before = points[k - 1]
                        .as_array()
                        .and_then(|a| a.first())
                        .and_then(Value::as_f64)
                        .unwrap_or(f64::NAN);
                    if t < before {
                        return Err(fail(
                            path,
                            format!(
                                "{at}.energy[{k}] at {}s comes before the point at {}s. List points in time order.",
                                js_string(t),
                                js_string(before)
                            ),
                        ));
                    }
                }
            }
        }
        other => {
            return Err(fail(
                path,
                format!(
                    "{at}.energy is {}. Use a number from 0 to 1, or a list of [seconds, energy] points.",
                    js_stringify(other)
                ),
            ));
        }
    }
    if let Some(t) = field(s, "in") {
        if !t.as_str().is_some_and(|t| TRANSITIONS.contains(&t)) {
            return Err(fail(
                path,
                format!(
                    "{at}.in is \"{}\". Use one of: {}.",
                    js_display(t),
                    TRANSITIONS.join(", ")
                ),
            ));
        }
        if i == 0 {
            return Err(fail(
                path,
                format!("{at}.in: the first scene has no cut to transition across. Remove it."),
            ));
        }
    }
    for (k, lo, hi) in [("inVol", 0.0, 4.0), ("inDur", 0.2, 4.0)] {
        if let Some(v) = field(s, k)
            && !within(v, lo, hi)
        {
            return Err(fail(
                path,
                format!(
                    "{at}.{k} is {}. Use a number from {} to {}.",
                    js_display(v),
                    js_string(lo),
                    js_string(hi)
                ),
            ));
        }
    }
    Ok(())
}

/// The music: an existing `file`, a `volume` above 0 and at most 1, and a
/// `start` of 0 seconds or more into the track.
pub(super) fn music(raw: &Value, root: &Path, path: &Path) -> Result<(), CommandError> {
    let Some(m) = field(raw, "music") else {
        return Ok(());
    };
    let named = field(m, "file");
    let file = resolve(root, Path::new(&named.map(js_display).unwrap_or_default()));
    let truthy = named.is_some_and(|v| match v {
        Value::String(s) => !s.is_empty(),
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        _ => true,
    });
    if !truthy || !file.exists() {
        return Err(fail(
            path,
            format!("\"music.file\" {} does not exist.", file.display()),
        ));
    }
    let volume = field(m, "volume").cloned().unwrap_or(Value::from(0.25));
    if !number(&volume).is_some_and(|v| v > 0.0 && v <= 1.0) {
        return Err(fail(
            path,
            format!(
                "\"music.volume\" is {}. Use a number above 0 and at most 1.",
                js_display(&volume)
            ),
        ));
    }
    let start = field(m, "start").cloned().unwrap_or(Value::from(0));
    if !number(&start).is_some_and(|v| v >= 0.0) {
        return Err(fail(
            path,
            format!(
                "\"music.start\" is {}. Use seconds into the track, 0 or more.",
                js_display(&start)
            ),
        ));
    }
    Ok(())
}

/// The skill's `parseKey(spec)` accepts `spec`: a root (`C`, `F#`, `Eb`)
/// and an optional mode (`m`, `min`, `minor`, `maj`, `major`), any case.
fn is_key(spec: &Value) -> bool {
    let Some(text) = spec.as_str() else {
        return false;
    };
    let t = trim(text);
    let mut chars = t.chars();
    let Some(letter) = chars
        .next()
        .filter(|c| matches!(c.to_ascii_uppercase(), 'A'..='G'))
    else {
        return false;
    };
    let rest = chars.as_str();
    let (accidental, rest) = match rest.chars().next() {
        Some(c @ ('#' | 'b' | 'B')) => (Some(c), &rest[1..]),
        _ => (None, rest),
    };
    let mode = rest.trim_start_matches(crate::pattern::is_space);
    let modes = ["m", "min", "minor", "maj", "major"];
    if !mode.is_empty() && !modes.iter().any(|m| m.eq_ignore_ascii_case(mode)) {
        return false;
    }
    let root = format!(
        "{}{}",
        letter.to_ascii_uppercase(),
        accidental.map(String::from).unwrap_or_default()
    );
    ROOTS.contains(&root.as_str())
}

/// The bed under the effects: absent, `false`, `true`, or an object of
/// `fragments` (1-8), `level` (-48 to -12) and `seed` (a whole number).
fn bed(raw: Option<&Value>, path: &Path) -> Result<(), CommandError> {
    let raw = match raw {
        None | Some(Value::Bool(false)) => return Ok(()),
        Some(v) => v,
    };
    let given = match raw {
        Value::Bool(true) => serde_json::Map::new(),
        Value::Object(map) => map.clone(),
        other => {
            return Err(fail(
                path,
                format!(
                    "\"sound.bed\" is {}. Use true, false, or an object with fragments, level and seed.",
                    js_stringify(other)
                ),
            ));
        }
    };
    let get = |k: &str, default: Value| given.get(k).cloned().unwrap_or(default);
    if given.contains_key("air") {
        return Err(fail(
            path,
            "\"sound.bed.air\" was removed: the bed has no noise layer. Delete the field.",
        ));
    }
    let known = ["fragments", "level", "seed"];
    let unknown: Vec<&str> = given
        .keys()
        .map(String::as_str)
        .filter(|k| !known.contains(k))
        .collect();
    if !unknown.is_empty() {
        return Err(fail(
            path,
            format!(
                "\"sound.bed\" has unknown field(s) {}. Use fragments, level and seed.",
                unknown.join(", ")
            ),
        ));
    }
    for (k, default, lo, hi) in [("fragments", 2.0, 1.0, 8.0), ("level", -32.0, -48.0, -12.0)] {
        let v = get(k, Value::from(default));
        if !within(&v, lo, hi) {
            return Err(fail(
                path,
                format!(
                    "\"sound.bed.{k}\" is {}. Use a number from {} to {}.",
                    js_stringify(&v),
                    js_string(lo),
                    js_string(hi)
                ),
            ));
        }
    }
    let seed = get("seed", Value::from(1));
    if !number(&seed).is_some_and(|s| s.fract() == 0.0 && s >= 0.0) {
        return Err(fail(
            path,
            format!(
                "\"sound.bed.seed\" is {}. Use a whole number of 0 or more.",
                js_stringify(&seed)
            ),
        ));
    }
    Ok(())
}

/// The `sound` block: its bed, `space`, `fadeIn` and `fadeOut` (0-5 s),
/// `palette` and `key`.
pub(super) fn sound(raw: &Value, path: &Path) -> Result<(), CommandError> {
    let empty = Value::Object(serde_json::Map::new());
    let s = field(raw, "sound").unwrap_or(&empty);
    bed(field(s, "bed"), path)?;
    let space = field(s, "space").cloned().unwrap_or(Value::from("room"));
    if !space.as_str().is_some_and(|v| SPACES.contains(&v)) {
        return Err(fail(
            path,
            format!(
                "\"sound.space\" is \"{}\". Use \"tight\", \"room\" or \"hall\".",
                js_display(&space)
            ),
        ));
    }
    for (k, default) in [("fadeIn", 0.3), ("fadeOut", 0.8)] {
        let v = field(s, k).cloned().unwrap_or(Value::from(default));
        if !within(&v, 0.0, 5.0) {
            return Err(fail(
                path,
                format!(
                    "\"sound.{k}\" is {}. Use seconds from 0 to 5.",
                    js_display(&v)
                ),
            ));
        }
    }
    let palette = field(s, "palette").cloned().unwrap_or(Value::from("synth"));
    if !palette.as_str().is_some_and(|p| PALETTES.contains(&p)) {
        return Err(fail(
            path,
            format!(
                "\"sound.palette\" is {}. Use one of: {}.",
                js_stringify(&palette),
                PALETTES.join(", ")
            ),
        ));
    }
    let key = field(s, "key").cloned().unwrap_or(Value::from("auto"));
    if key.as_str() != Some("auto") && !is_key(&key) {
        return Err(fail(
            path,
            format!(
                "\"sound.key\" is {}. Use \"auto\" or a key such as \"C\", \"F# minor\" or \"Ebm\".",
                js_stringify(&key)
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keys_parse_as_the_skill_parses_them() {
        for good in ["C", "f# minor", "Ebm", "A major", " Bb maj ", "g"] {
            assert!(is_key(&json!(good)), "{good}");
        }
        for bad in ["H", "Cb", "E#", "C minorish", "", "#"] {
            assert!(!is_key(&json!(bad)), "{bad}");
        }
        assert!(!is_key(&json!(3)));
    }
}
