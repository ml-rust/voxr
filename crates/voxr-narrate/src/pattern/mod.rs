//! The module's regular expressions, compiled once, and the JavaScript string
//! semantics they run under: the `\s` set, `trim`, UTF-16 lengths, `split`
//! and `Number(string)`.
//!
//! The patterns spell out what JavaScript means where regexr differs: `\s` is
//! [`WS`], `\d` is `[0-9]`, `^`/`$` are `\A`/`\z`, and a case-insensitive
//! word is an ASCII class per letter ([`ci`]), since JavaScript's `/i`
//! without `u` does not fold `ſ` to `s` the way Unicode folding does.

mod compiled;
mod js;

pub(crate) use compiled::{Patterns, collapse};
pub(crate) use js::{
    group, is_space, number, replace_all, replace_all_with, split, starts_with_en, trim, utf16_len,
    utf16_prefix,
};
