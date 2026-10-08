//! The skill's own reading of a flag's value: a number with its lower
//! bound, and the part numbers `--reroll` takes.

use super::outcome::CommandError;
use crate::js::js_string;
use crate::pattern::{number, trim};

/// `num(flag, value, min)`: `value` read as JavaScript's `Number()` reads
/// it, refused unless it is finite and at least `min`.
///
/// The error is the skill's: `--pace needs a number of at least 2, got
/// "1".`, a bad input.
pub fn flag_number(flag: &str, value: &str, min: f64) -> Result<f64, CommandError> {
    let n = number(value);
    if !n.is_finite() || n < min {
        return Err(CommandError::Input(format!(
            "{flag} needs a number of at least {}, got \"{value}\".",
            js_string(min)
        )));
    }
    Ok(n)
}

/// `--reroll 3,7`: each comma-separated value, trimmed, as a part number
/// from 1.
///
/// The skill reads each value with `Number()` and lets VoxCPM2 refuse one
/// out of range. A value that is not a whole number from 1 never names a
/// part, so it is refused here, a bad input.
pub fn reroll_parts(value: &str) -> Result<Vec<usize>, CommandError> {
    value
        .split(',')
        .map(|v| {
            let n = number(trim(v));
            if n.is_finite() && n >= 1.0 && n.fract() == 0.0 && n <= usize::MAX as f64 {
                Ok(n as usize)
            } else {
                Err(CommandError::Input(format!(
                    "--reroll needs part numbers from 1, such as 3,7, got \"{value}\"."
                )))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_read_as_javascript_reads_them() -> Result<(), CommandError> {
        assert_eq!(flag_number("--speed", " 0.9 ", 0.5)?, 0.9);
        assert_eq!(flag_number("--seed", "0x10", 0.0)?, 16.0);
        assert_eq!(flag_number("--seed", "", 0.0)?, 0.0);
        assert_eq!(
            flag_number("--pace", "1", 2.0),
            Err(CommandError::Input(
                "--pace needs a number of at least 2, got \"1\".".to_string()
            ))
        );
        assert!(flag_number("--cfg", "abc", 0.5).is_err());
        assert!(flag_number("--cfg", "Infinity", 0.5).is_err());
        Ok(())
    }

    #[test]
    fn reroll_reads_whole_part_numbers() -> Result<(), CommandError> {
        assert_eq!(reroll_parts("3, 7")?, [3, 7]);
        assert_eq!(reroll_parts("1")?, [1]);
        assert!(reroll_parts("0").is_err());
        assert!(reroll_parts("2.5").is_err());
        assert!(reroll_parts("x,2").is_err());
        Ok(())
    }
}
