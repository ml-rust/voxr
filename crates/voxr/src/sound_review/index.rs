//! The review's `sounds.json`: one entry per rendered sound, in file
//! order, as Python's `json.loads` keeps a dict's keys.

use std::fmt;

use serde::Deserialize;
use serde::de::{Deserializer, MapAccess, Visitor};
use serde_json::Value;

/// One rendered sound: its label (the WAV's stem) and its metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct SoundEntry {
    pub label: String,
    pub meta: Value,
}

impl SoundEntry {
    /// The `says` text, when the metadata holds a string there.
    pub fn says(&self) -> Option<&str> {
        self.meta.get("says").and_then(Value::as_str)
    }

    /// Python's `bool(meta.get("endsOnCue"))`: a build that stops on its
    /// cue, whose tail is not judged.
    pub fn ends_on_cue(&self) -> bool {
        self.meta.get("endsOnCue").is_some_and(truthy)
    }
}

/// Python truthiness of a JSON value.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The index, in file order. A repeated key keeps its first place and its
/// last value, as a Python dict does.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SoundIndex(pub Vec<SoundEntry>);

struct IndexVisitor;

impl<'de> Visitor<'de> for IndexVisitor {
    type Value = SoundIndex;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("an object of sounds by label")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<SoundIndex, A::Error> {
        let mut entries: Vec<SoundEntry> = Vec::new();
        while let Some((label, meta)) = map.next_entry::<String, Value>()? {
            match entries.iter_mut().find(|e| e.label == label) {
                Some(e) => e.meta = meta,
                None => entries.push(SoundEntry { label, meta }),
            }
        }
        Ok(SoundIndex(entries))
    }
}

impl<'de> Deserialize<'de> for SoundIndex {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_map(IndexVisitor)
    }
}

impl SoundIndex {
    /// Parses `sounds.json` text. The error is serde's reason.
    pub fn parse(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_keep_file_order() {
        let idx = SoundIndex::parse(
            r#"{"zip_v0": {"says": "a zipper", "endsOnCue": false},
                "boom_v0": {"says": "a boom", "endsOnCue": true},
                "zip_v0": {"says": "zip again"}}"#,
        )
        .expect("index");
        let labels: Vec<&str> = idx.0.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(labels, ["zip_v0", "boom_v0"]);
        assert_eq!(idx.0[0].says(), Some("zip again"));
        assert!(!idx.0[0].ends_on_cue());
        assert!(idx.0[1].ends_on_cue());
    }

    #[test]
    fn a_list_is_refused() {
        assert!(SoundIndex::parse("[1, 2]").is_err());
        assert!(SoundIndex::parse("{").is_err());
    }
}
