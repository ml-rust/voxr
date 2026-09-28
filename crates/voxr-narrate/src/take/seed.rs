//! The seed a part draws from its own text, when no take is recorded for
//! it yet.

use sha1::{Digest, Sha1};

use super::cache::hex;

/// `seed + (parseInt(sha1(text).slice(0, 6), 16) % 1000)`: the seed a
/// part's own text draws, on top of the run's base seed.
///
/// A part's seed comes from its own text, not its place in the script:
/// adding or splitting a line leaves every other part's seed, and so its
/// cached take, as it was. The sha1 input is the text's UTF-8 bytes, the
/// same bytes `sha1(text)` hashes in the skill.
pub fn text_seed(text: &str, base: f64) -> f64 {
    let digest = Sha1::digest(text.as_bytes());
    let text_hex = hex(digest.as_slice());
    let n = i64::from_str_radix(&text_hex[..6], 16).unwrap_or(0);
    base + (n % 1000) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_a_bounded_offset_to_the_base_seed() {
        let a = text_seed("hello", 7.0);
        assert!((7.0..1007.0).contains(&a));
        // The same text always draws the same offset.
        assert_eq!(a, text_seed("hello", 7.0));
        // A different base moves the seed by exactly the base's difference.
        assert_eq!(text_seed("hello", 107.0), a + 100.0);
    }

    #[test]
    fn differs_by_text() {
        assert_ne!(text_seed("hello", 7.0), text_seed("world", 7.0));
    }
}
