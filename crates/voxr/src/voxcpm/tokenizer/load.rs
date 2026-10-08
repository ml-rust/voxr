//! [`VoxCpm2Tokenizer`]: VoxCPM2's `tokenizer.json` through splintr, giving
//! the ids the reference `text_tokenizer` gives.
//!
//! The reference wraps a `LlamaTokenizerFast` in
//! `mask_multichar_chinese_tokens` and calls `tokenize` then
//! `convert_tokens_to_ids`. Three things follow, and this type does each:
//!
//! - **No BOS.** `tokenize` never runs the post-processor that prepends
//!   `<s>`, so every encode here is `encode_raw`.
//! - **A leading space is the word boundary.** transformers loads the file
//!   with a `Metaspace` pre-tokenizer that turns each space into `▁` and
//!   prepends `▁` only when the text does not already start with one. The
//!   file's declared normalizer, which splintr runs, prepends `▁` always.
//!   The two agree once one leading space or `▁` is dropped before the
//!   encode, which is what [`VoxCpm2Tokenizer::ids`] does.
//! - **Multi-character Han tokens split.** A token whose text without `▁`
//!   is a vocabulary entry of two or more characters, all in U+4E00 to
//!   U+9FFF, becomes the ids of its characters, each looked up alone (the
//!   unknown id for a character outside the vocabulary). The `▁` of the
//!   token is dropped with it.

use std::collections::HashMap;

use boostr::model::audio::voxcpm::TokenizerSource;
use serde_json::Value;
use splintr::AnyTokenizer;

use crate::error::{Error, Result};
use crate::voxcpm::tokenizer::text::normalize_whitespace;

/// The metaspace marker the vocabulary spells a word boundary with.
const METASPACE: char = '\u{2581}';

/// VoxCPM2's tokenizer, producing the reference model's ids.
pub struct VoxCpm2Tokenizer {
    inner: AnyTokenizer,
    /// Per token id that splits, the ids of its characters.
    split: HashMap<u32, Vec<u32>>,
}

impl VoxCpm2Tokenizer {
    /// The ids the reference gives `text`, exactly as given: no whitespace
    /// collapse, no BOS. See the module docs.
    pub fn ids(&self, text: &str) -> Vec<u32> {
        let mut chars = text.chars();
        let body = match chars.next() {
            Some(' ') | Some(METASPACE) => chars.as_str(),
            _ => text,
        };
        let raw = self.inner.encode_raw(body);
        if self.split.is_empty() {
            return raw;
        }
        let mut out = Vec::with_capacity(raw.len());
        for id in raw {
            match self.split.get(&id) {
                Some(parts) => out.extend_from_slice(parts),
                None => out.push(id),
            }
        }
        out
    }

    /// The splintr tokenizer underneath, with no split applied.
    pub fn splintr(&self) -> &AnyTokenizer {
        &self.inner
    }
}

/// The ids the reference gives `text` after `generate`'s whitespace
/// collapse ([`normalize_whitespace`]): what a request's text reaches the
/// model as.
pub fn tokenize(tokenizer: &VoxCpm2Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.ids(&normalize_whitespace(text))
}

/// A request's text ids and its target length, as the reference builds
/// them for `text` and an optional continuation transcript.
///
/// The reference collapses the whitespace of `text` only. With a
/// transcript, the ids are those of `transcript + text`, joined with no
/// separator. The target length counts the ids of the collapsed `text`
/// alone: the patch budget reads it.
pub fn request_ids(
    tokenizer: &VoxCpm2Tokenizer,
    text: &str,
    transcript: Option<&str>,
) -> (Vec<u32>, usize) {
    let target = normalize_whitespace(text);
    let target_ids = tokenizer.ids(&target);
    match transcript {
        None => {
            let len = target_ids.len();
            (target_ids, len)
        }
        Some(transcript) => (
            tokenizer.ids(&format!("{transcript}{target}")),
            target_ids.len(),
        ),
    }
}

/// Load VoxCPM2's `tokenizer.json` from `source`: the text a GGUF embeds,
/// or a file on disk.
pub fn load_tokenizer(source: &TokenizerSource) -> Result<VoxCpm2Tokenizer> {
    let text = match source {
        TokenizerSource::Embedded(text) => text.clone(),
        TokenizerSource::File(path) => {
            std::fs::read_to_string(path).map_err(|e| Error::ModelError {
                reason: format!("VoxCPM2 tokenizer.json {}: {e}", path.display()),
            })?
        }
    };
    from_json(&text)
}

/// [`load_tokenizer`] on the `tokenizer.json` text itself.
pub fn from_json(text: &str) -> Result<VoxCpm2Tokenizer> {
    let inner = splintr::from_json_bytes(text.as_bytes()).map_err(|e| Error::ModelError {
        reason: format!("VoxCPM2 tokenizer.json: {e}"),
    })?;
    let json: Value = serde_json::from_str(text).map_err(|e| Error::ModelError {
        reason: format!("VoxCPM2 tokenizer.json is not JSON: {e}"),
    })?;
    let vocab = vocab_of(&json)?;
    let unk = json
        .pointer("/model/unk_token")
        .and_then(Value::as_str)
        .and_then(|t| vocab.get(t).copied());
    Ok(VoxCpm2Tokenizer {
        inner,
        split: split_table(&vocab, unk)?,
    })
}

/// `model.vocab` with every `added_tokens` entry, as transformers'
/// `tokenizer.vocab` holds them.
fn vocab_of(json: &Value) -> Result<HashMap<String, u32>> {
    let model = json
        .pointer("/model/vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| Error::ModelError {
            reason: "VoxCPM2 tokenizer.json has no model.vocab object".into(),
        })?;
    let mut vocab = HashMap::with_capacity(model.len());
    for (token, id) in model {
        vocab.insert(token.clone(), id_of(id, token)?);
    }
    if let Some(added) = json.get("added_tokens").and_then(Value::as_array) {
        for entry in added {
            if let (Some(token), Some(id)) = (
                entry.get("content").and_then(Value::as_str),
                entry.get("id"),
            ) {
                vocab.insert(token.to_string(), id_of(id, token)?);
            }
        }
    }
    Ok(vocab)
}

fn id_of(id: &Value, token: &str) -> Result<u32> {
    id.as_u64()
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| Error::ModelError {
            reason: format!("VoxCPM2 tokenizer.json gives token {token:?} the id {id}, not a u32"),
        })
}

/// Whether `ch` is in the CJK Unified Ideographs block the reference
/// splits.
fn is_han(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
}

/// The reference's split, per token id: every token whose text without
/// `▁` is a multi-character, all-Han vocabulary entry maps to its
/// characters' ids. A character outside the vocabulary takes `unk`, and
/// with no `unk` the table cannot be built.
fn split_table(vocab: &HashMap<String, u32>, unk: Option<u32>) -> Result<HashMap<u32, Vec<u32>>> {
    let multichar = |t: &str| t.chars().count() >= 2 && t.chars().all(is_han);
    let mut split = HashMap::new();
    for (token, &id) in vocab {
        let clean: String = token.chars().filter(|&c| c != METASPACE).collect();
        if !multichar(&clean) || !vocab.contains_key(&clean) {
            continue;
        }
        let mut parts = Vec::with_capacity(clean.chars().count());
        for ch in clean.chars() {
            let part = vocab
                .get(ch.encode_utf8(&mut [0; 4]) as &str)
                .copied()
                .or(unk)
                .ok_or_else(|| Error::ModelError {
                    reason: format!(
                        "VoxCPM2 tokenizer.json splits {token:?} into characters, but {ch:?} \
                         is not in its vocabulary and it names no unk_token"
                    ),
                })?;
            parts.push(part);
        }
        split.insert(id, parts);
    }
    Ok(split)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Miniature Llama-2-shaped tokenizer.json: metaspace normalizer
    /// (`Prepend "▁"` then `Replace " " -> "▁"`), `pre_tokenizer: null`,
    /// the same declared shape as VoxCPM2's real `tokenizer.json`. `你好`
    /// and `▁你好` are whole tokens, so the split shows.
    const MINI_TOKENIZER_JSON: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": {"type": "Sequence", "normalizers": [
        {"type": "Prepend", "prepend": "▁"},
        {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
      ]},
      "pre_tokenizer": null,
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "BPE",
        "dropout": null,
        "unk_token": "<unk>",
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "byte_fallback": false,
        "vocab": {"<unk>": 0, "a": 1, "b": 2, "e": 5, "h": 8, "t": 20, "▁": 27, "▁t": 28, "▁th": 31, "▁the": 35, "你": 40, "好": 41, "▁你": 42, "你好": 43, "▁你好": 44, "世": 45},
        "merges": ["▁ t", "▁t h", "▁th e", "你 好", "▁ 你", "▁你 好"]
      }
    }"#;

    fn mini() -> VoxCpm2Tokenizer {
        from_json(MINI_TOKENIZER_JSON).expect("mini tokenizer.json loads")
    }

    #[test]
    fn a_leading_space_is_the_word_boundary_not_a_second_one() {
        let tok = mini();
        assert_eq!(tok.ids("the"), vec![35]);
        assert_eq!(tok.ids(" the"), vec![35]);
        assert_eq!(tok.ids("\u{2581}the"), vec![35]);
        assert_eq!(tok.ids("the "), vec![35, 27]);
    }

    #[test]
    fn multi_character_han_tokens_split_into_characters() {
        let tok = mini();
        assert_eq!(tok.splintr().encode_raw("你好"), vec![44]);
        assert_eq!(tok.ids("你好"), vec![40, 41]);
    }

    #[test]
    fn a_character_outside_the_vocabulary_takes_the_unknown_id() {
        let vocab: HashMap<String, u32> = [("世界".to_string(), 9), ("世".to_string(), 3)]
            .into_iter()
            .collect();
        let split = split_table(&vocab, Some(0)).expect("split");
        assert_eq!(split.get(&9), Some(&vec![3, 0]));
        assert!(split_table(&vocab, None).is_err());
    }

    #[test]
    fn a_token_whose_clean_text_is_not_an_entry_does_not_split() {
        let vocab: HashMap<String, u32> = [("▁世界".to_string(), 9), ("世".to_string(), 3)]
            .into_iter()
            .collect();
        assert!(split_table(&vocab, Some(0)).expect("split").is_empty());
    }

    #[test]
    fn embedded_and_file_sources_load_the_same_tokenizer() {
        let embedded = load_tokenizer(&TokenizerSource::Embedded(MINI_TOKENIZER_JSON.to_string()))
            .expect("embedded source loads");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, MINI_TOKENIZER_JSON).expect("write");
        let from_file = load_tokenizer(&TokenizerSource::File(path)).expect("file loads");
        assert_eq!(embedded.ids("the"), vec![35]);
        assert_eq!(from_file.ids("the"), vec![35]);
        assert!(load_tokenizer(&TokenizerSource::Embedded("not json".to_string())).is_err());
    }
}
