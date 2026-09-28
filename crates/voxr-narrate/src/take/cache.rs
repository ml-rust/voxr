//! VoxCPM2 take cache: the keys a designed voice, a prepared reference and
//! a take are kept under, and the file names those keys carry.
//!
//! A designed voice is kept by its description, seed and language, so
//! every speak call with the same voice clones the same one. A take is
//! kept by the model, the voice, the reference, the style, the text and
//! the seed, so a run speaks only the parts it has no take for.

use std::path::{Path, PathBuf};

use sha1::{Digest, Sha1};

use crate::js::js_string;

/// The VoxCPM2 checkpoint a key falls back to when none is on disk.
pub const VOXCPM: &str = "openbmb/VoxCPM2";

/// A part's own delivery cfg and steps that keep the runner's older keys:
/// takes made before `--cfg`/`--steps` existed are kept unchanged.
pub const LEGACY_CFG: f64 = 2.0;
/// See [`LEGACY_CFG`].
pub const LEGACY_STEPS: f64 = 10.0;

/// One value `JSON.stringify` sees inside a cache key's part array.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyPart<'a> {
    /// A string, written as `JSON.stringify` escapes it.
    Str(&'a str),
    /// A number, written as JavaScript writes it, or `null` when it is not
    /// finite (`JSON.stringify` turns NaN and the infinities into `null`).
    Num(f64),
    /// `null`, or an array element left `undefined`.
    Null,
}

fn json_str(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_default()
}

/// `bytes.toString('hex')`: lowercase hex, two digits per byte.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn json_part(part: &KeyPart<'_>) -> String {
    match part {
        KeyPart::Str(s) => json_str(s),
        KeyPart::Num(x) if x.is_finite() => js_string(*x),
        KeyPart::Num(_) | KeyPart::Null => "null".to_string(),
    }
}

/// `createHash('sha1').update(JSON.stringify([checkpoint ?? VOXCPM, voice,
/// refId, ...parts])).digest('hex').slice(0, 20)`.
///
/// `checkpoint` is the caller's resolved VoxCPM2 checkpoint path, already
/// looked up; this never reads the filesystem itself.
pub fn cache_key(
    checkpoint: Option<&str>,
    voice: &str,
    ref_id: &str,
    parts: &[KeyPart<'_>],
) -> String {
    let mut items = vec![
        json_str(checkpoint.unwrap_or(VOXCPM)),
        json_str(voice),
        json_str(ref_id),
    ];
    items.extend(parts.iter().map(json_part));
    let json = format!("[{}]", items.join(","));
    let digest = Sha1::digest(json.as_bytes());
    hex(digest.as_slice())[..20].to_string()
}

/// A designed voice's key: `hash('reference', seed, language ?? '')`. Used
/// with an empty reference id, since a designed voice never comes from a
/// user's recording.
pub fn anchor_key(
    checkpoint: Option<&str>,
    voice: &str,
    seed: f64,
    language: Option<&str>,
) -> String {
    cache_key(
        checkpoint,
        voice,
        "",
        &[
            KeyPart::Str("reference"),
            KeyPart::Num(seed),
            KeyPart::Str(language.unwrap_or("")),
        ],
    )
}

/// A user's recording, cloned from its prepared copy: `hash('breaths',
/// 'bass', 'compressed', 'normalized')`. The recording itself never
/// changes; only its cloned-from copy is kept under this key.
pub fn prepared_reference_key(checkpoint: Option<&str>, voice: &str, ref_id: &str) -> String {
    cache_key(
        checkpoint,
        voice,
        ref_id,
        &[
            KeyPart::Str("breaths"),
            KeyPart::Str("bass"),
            KeyPart::Str("compressed"),
            KeyPart::Str("normalized"),
        ],
    )
}

/// One take's cache key inputs.
#[derive(Debug, Clone, Copy)]
pub struct TakeKeyInput<'a> {
    /// The designed voice's key, or `None` when a reference recording is
    /// used instead (the skill's `anchorKey`, `null` in that case).
    pub anchor_key: Option<&'a str>,
    /// The part's own delivery note, or the run's `--style`.
    pub style: &'a str,
    /// The part's spoken text.
    pub text: &'a str,
    /// The seed this take was made with.
    pub seed: f64,
    /// VoxCPM2's guidance scale.
    pub cfg: f64,
    /// VoxCPM2's diffusion steps.
    pub steps: f64,
    /// Hi-Fi cloning was on.
    pub hifi: bool,
}

/// A take's key: `hash(anchorKey, style, text, seed, [cfg, steps when not
/// the legacy defaults], [hifi])`.
///
/// `cfg` and `steps` are left out of the key when they are exactly
/// [`LEGACY_CFG`] and [`LEGACY_STEPS`], so takes made at the runner's own
/// settings before `--cfg`/`--steps` existed keep their older keys.
pub fn take_key(
    checkpoint: Option<&str>,
    voice: &str,
    ref_id: &str,
    input: TakeKeyInput<'_>,
) -> String {
    let mut parts = vec![
        input.anchor_key.map_or(KeyPart::Null, KeyPart::Str),
        KeyPart::Str(input.style),
        KeyPart::Str(input.text),
        KeyPart::Num(input.seed),
    ];
    if input.cfg != LEGACY_CFG || input.steps != LEGACY_STEPS {
        parts.push(KeyPart::Num(input.cfg));
        parts.push(KeyPart::Num(input.steps));
    }
    if input.hifi {
        parts.push(KeyPart::Str("hifi"));
    }
    cache_key(checkpoint, voice, ref_id, &parts)
}

/// `path:size:mtimeMs`: a reference recording's cache identity, built from
/// its path and the size and modified time its metadata carries.
///
/// `path` is written verbatim, as the skill interpolates the path it was
/// given. `mtime_ms` is milliseconds since the epoch, fractional exactly as
/// `Stats.mtimeMs` is: seconds times 1000 plus the sub-second nanoseconds
/// divided by 1e6.
pub fn ref_id(path: &str, size: u64, mtime_ms: f64) -> String {
    format!("{path}:{size}:{}", js_string(mtime_ms))
}

/// `mtime_ms` for [`ref_id`], from a file's modified time.
pub fn mtime_millis(modified: std::time::SystemTime) -> f64 {
    match modified.duration_since(std::time::UNIX_EPOCH) {
        Ok(dur) => dur.as_secs() as f64 * 1000.0 + f64::from(dur.subsec_nanos()) / 1_000_000.0,
        Err(e) => {
            -(e.duration().as_secs() as f64 * 1000.0
                + f64::from(e.duration().subsec_nanos()) / 1_000_000.0)
        }
    }
}

/// `join(cache, 'voice-${key}.wav')`: a designed voice's reference.
pub fn voice_file(cache_dir: &Path, key: &str) -> PathBuf {
    cache_dir.join(format!("voice-{key}.wav"))
}

/// `join(cache, 'ref-${key}.wav')`: a user's recording, cloned from its
/// prepared copy.
pub fn reference_file(cache_dir: &Path, key: &str) -> PathBuf {
    cache_dir.join(format!("ref-{key}.wav"))
}

/// `join(cache, '${key}.wav')`: one take.
pub fn take_file(cache_dir: &Path, key: &str) -> PathBuf {
    cache_dir.join(format!("{key}.wav"))
}

/// `createHash('sha1').update(readFileSync(file)).digest('hex')`: the full
/// hex digest of a prepared reference's bytes, the name its marker file
/// carries under `prepared/`.
pub fn content_hash(bytes: &[u8]) -> String {
    hex(Sha1::digest(bytes).as_slice())
}

/// `join(cache, 'prepared', contentHash)`: a prepared reference's marker,
/// left beside the cache once its compression is done, so it is never
/// redone on a copy already prepared.
pub fn prepared_marker(cache_dir: &Path, content_hash: &str) -> PathBuf {
    cache_dir.join("prepared").join(content_hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_json_escapes_like_javascript() {
        let key = cache_key(
            None,
            "voice \"a\"",
            "",
            &[
                KeyPart::Str("line\nbreak"),
                KeyPart::Num(1.5),
                KeyPart::Null,
            ],
        );
        assert_eq!(key.len(), 20);
        // The digest is stable and the default checkpoint falls back to VOXCPM.
        assert_eq!(
            key,
            cache_key(
                Some(VOXCPM),
                "voice \"a\"",
                "",
                &[
                    KeyPart::Str("line\nbreak"),
                    KeyPart::Num(1.5),
                    KeyPart::Null
                ]
            )
        );
    }

    #[test]
    fn non_finite_numbers_serialize_as_null() {
        assert_eq!(
            cache_key(None, "v", "", &[KeyPart::Num(f64::NAN)]),
            cache_key(None, "v", "", &[KeyPart::Null]),
        );
    }

    #[test]
    fn take_key_omits_cfg_and_steps_at_legacy_defaults() {
        let base = TakeKeyInput {
            anchor_key: Some("abc"),
            style: "",
            text: "hello",
            seed: 7.0,
            cfg: LEGACY_CFG,
            steps: LEGACY_STEPS,
            hifi: false,
        };
        let legacy = take_key(None, "v", "", base);
        let mut changed = base;
        changed.cfg = 1.6;
        let with_cfg = take_key(None, "v", "", changed);
        assert_ne!(legacy, with_cfg);
        // Passing the legacy numbers explicitly reproduces the same key as omitting them.
        assert_eq!(
            legacy,
            cache_key(
                None,
                "v",
                "",
                &[
                    KeyPart::Str("abc"),
                    KeyPart::Str(""),
                    KeyPart::Str("hello"),
                    KeyPart::Num(7.0)
                ]
            ),
        );
    }

    #[test]
    fn take_key_null_anchor_for_a_reference_voice() {
        let input = TakeKeyInput {
            anchor_key: None,
            style: "calm",
            text: "hi",
            seed: 1.0,
            cfg: LEGACY_CFG,
            steps: LEGACY_STEPS,
            hifi: true,
        };
        assert_eq!(
            take_key(None, "v", "ref", input),
            cache_key(
                None,
                "v",
                "ref",
                &[
                    KeyPart::Null,
                    KeyPart::Str("calm"),
                    KeyPart::Str("hi"),
                    KeyPart::Num(1.0),
                    KeyPart::Str("hifi")
                ]
            ),
        );
    }

    #[test]
    fn file_names_match_the_skill() {
        let dir = Path::new("/cache");
        assert_eq!(voice_file(dir, "abc"), Path::new("/cache/voice-abc.wav"));
        assert_eq!(reference_file(dir, "abc"), Path::new("/cache/ref-abc.wav"));
        assert_eq!(take_file(dir, "abc"), Path::new("/cache/abc.wav"));
        assert_eq!(
            prepared_marker(dir, "deadbeef"),
            Path::new("/cache/prepared/deadbeef")
        );
    }

    #[test]
    fn ref_id_joins_path_size_and_mtime() {
        assert_eq!(
            ref_id("/rec.wav", 12345, 1_699_999_999_999.25),
            "/rec.wav:12345:1699999999999.25"
        );
    }
}
