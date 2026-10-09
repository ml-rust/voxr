//! The review's limits, verdicts and printed table, character for
//! character as the skill's script prints them.

use super::measure::RawMeasures;
use super::pyfmt::py_float_repr;
use super::rank::Rank;

/// Sharpness above this fails, acum: the sound reads as harsh.
pub const SHARP_FAIL: f64 = 2.6;
/// Hiss above this fails, dB: audible hiss or fizz.
pub const HISS_FAIL: f64 = -10.0;
/// A tail above this fails, dB: the sound is cut off, not finished.
pub const TAIL_FAIL: f64 = -20.0;
/// A true peak above this fails, dBTP.
pub const TP_FAIL: f64 = -1.0;
/// A recognizability rank above this warns.
pub const RANK_WARN: usize = 10;

/// One sound's line of the table.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub label: String,
    /// The rounded measures.
    pub measures: RawMeasures,
    /// Names of the limits failed, in the order sharp, hiss, tail, tp.
    pub fails: Vec<&'static str>,
}

impl Row {
    /// The row of `label` for its rounded `measures`. A build that stops on
    /// its cue (`ends_on_cue`) is not judged on its tail.
    pub fn judge(label: &str, measures: RawMeasures, ends_on_cue: bool) -> Self {
        let m = measures;
        let fails = [
            ("sharp", m.sharp > SHARP_FAIL),
            ("hiss", m.hiss > HISS_FAIL),
            ("tail", m.tail > TAIL_FAIL && !ends_on_cue),
            ("tp", m.tp > TP_FAIL),
        ]
        .into_iter()
        .filter_map(|(name, bad)| bad.then_some(name))
        .collect();
        Self {
            label: label.to_string(),
            measures,
            fails,
        }
    }

    /// The printed line. `rank`: the sound's recognizability over `texts`
    /// texts, when it was ranked.
    pub fn line(&self, rank: Option<&Rank>, texts: usize) -> String {
        let m = &self.measures;
        let rk = rank.map_or_else(String::new, |r| format!("{}/{texts}", r.rank));
        let mut note = if self.fails.is_empty() {
            "ok".to_string()
        } else {
            format!("FAIL {}", self.fails.join(","))
        };
        if let Some(r) = rank.filter(|r| r.rank > RANK_WARN) {
            note.push_str(&format!("  warn: heard as '{}'", r.heard));
        }
        format!(
            "{:<22}{:>6}{:>7}{:>7}{:>6}  {rk:<8}{note}",
            self.label,
            py_float_repr(m.sharp),
            py_float_repr(m.hiss),
            py_float_repr(m.tail),
            py_float_repr(m.tp),
        )
    }
}

/// The table's header line.
pub fn header() -> String {
    format!(
        "{:<22}{:>6}{:>7}{:>7}{:>6}  {:<8}verdict",
        "sound", "sharp", "hiss", "tail", "tp", "rank"
    )
}

/// The closing line, after a blank one.
pub fn summary(rows: usize, failing: usize) -> String {
    format!("{rows} sounds, {failing} failing a numeric limit.")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(sharp: f64, hiss: f64, tail: f64, tp: f64) -> RawMeasures {
        RawMeasures {
            sharp,
            hiss,
            tail,
            tp,
        }
    }

    #[test]
    fn lines_match_the_script() {
        assert_eq!(
            header(),
            "sound                  sharp   hiss   tail    tp  rank    verdict"
        );
        let row = Row::judge("pop_v0", m(0.87, -76.4, -50.1, -16.4), false);
        let rank = Rank {
            rank: 2,
            heard: "x".into(),
        };
        assert_eq!(
            row.line(Some(&rank), 41),
            "pop_v0                  0.87  -76.4  -50.1 -16.4  2/41    ok"
        );
        let row = Row::judge("blip_v1", m(0.7, -116.1, -41.4, -20.5), false);
        assert_eq!(
            row.line(None, 41),
            "blip_v1                  0.7 -116.1  -41.4 -20.5          ok"
        );
    }

    #[test]
    fn failures_and_warnings_read_as_the_script_writes_them() {
        let row = Row::judge("riser_v0", m(2.7, -9.0, -5.0, -0.5), true);
        assert_eq!(row.fails, vec!["sharp", "hiss", "tp"]);
        let rank = Rank {
            rank: 11,
            heard: "a boom".into(),
        };
        assert!(
            row.line(Some(&rank), 41)
                .ends_with("11/41   FAIL sharp,hiss,tp  warn: heard as 'a boom'")
        );
        assert_eq!(summary(265, 0), "265 sounds, 0 failing a numeric limit.");
    }
}
