//! A whole VoxCPM2 run: the script split into parts, the voice every part
//! clones (designed from a description, or prepared from a user's
//! recording), the best take of each part, and the takes joined into the
//! voiceover.
//!
//! A run keeps the take chosen for each part in `<out>.takes.json`, with
//! the seed that made the voice, so a later run keeps both. Only an asked
//! seed other than the recorded one starts over. A run writes `<out>`,
//! `<out>.takes.json` and, for a designed voice, `<out>.voice.wav`, besides
//! the cache.

use std::path::{Path, PathBuf};

use super::anchor::{ANCHOR_RATE, AnchorStep};
use super::anchor_files::{
    AnchorSetup, OUT_NEXT, clone_file, copy_voice_beside, designed_anchor, write_whole_with,
};
use super::cache::{anchor_key, prepared_reference_key};
use super::engine::{CloneReference, SpeechEngine, TakeJudge, VoiceSource};
use super::files::{ReadTakes, chosen_takes, read_takes_file, resolve_seed, takes_file};
use super::join::JOIN_RATE;
use super::judge::{JudgedTake, ReferenceHearing};
use super::pick::{PickSetup, TakeLoop};
use super::progress::{Sink, Tee};
use super::speak_setup::{VoiceReference, VoxcpmRun, VoxcpmSetup};
use super::speak_step::{SpeakStep, anchor_step, short_parts, take_step};
use super::step::TakeStep;
use super::store::{FileStore, TakeKeys};
use crate::error::{Error, Result};
use crate::glitch::median_pitch;
use crate::text::{Chunk, ChunkOptions, chunks};
use crate::wav::{pcm16_wav, read_wav_file};

/// The most words one take of a whole script holds before it splits at a
/// sentence end.
pub const WHOLE_MAX: usize = 700;

/// The voice every part clones.
struct Voice {
    /// The reference in the cache: the designed voice or the prepared
    /// recording.
    file: PathBuf,
    /// The reference's words, for Hi-Fi cloning.
    transcript: Option<String>,
    /// The designed voice's copy beside the output.
    voice_file: Option<PathBuf>,
}

fn non_empty(text: &str) -> Option<&str> {
    (!text.is_empty()).then_some(text)
}

/// The script's parts: one take of the whole script up to [`WHOLE_MAX`]
/// words, or, for a timed script, one part per line. An empty script and a
/// re-roll outside the parts are refused.
fn parts_of(script: &str, setup: &VoxcpmSetup<'_>) -> Result<Vec<Chunk>> {
    let options = if setup.lines {
        ChunkOptions {
            lines: true,
            ..ChunkOptions::default()
        }
    } else {
        ChunkOptions {
            lines: false,
            max: WHOLE_MAX,
            whole: true,
        }
    };
    let units = chunks(script, options)?;
    if units.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "script",
            reason: "the script is empty. Pass a script with at least one word to speak"
                .to_string(),
        });
    }
    if let Some(n) = setup.rerolls.iter().find(|&&n| n < 1 || n > units.len()) {
        return Err(Error::InvalidArgument {
            arg: "rerolls",
            reason: format!("--reroll {n}: the script has parts 1 to {}.", units.len()),
        });
    }
    Ok(units)
}

/// `<out>.takes.json` read back. A missing or unreadable file reads as no
/// saved takes, as the skill's `catch {}` reads it.
fn read_record(file: &Path) -> ReadTakes {
    std::fs::read(file)
        .map(|bytes| read_takes_file(&String::from_utf8_lossy(&bytes)))
        .unwrap_or_default()
}

/// The user's recording, prepared into the cache, and with Hi-Fi its
/// words as the run's recognizer hears them.
fn cloned_voice<J: ReferenceHearing>(
    setup: &VoxcpmSetup<'_>,
    reference: &VoiceReference<'_>,
    judge: &mut J,
    decode: &mut dyn FnMut(&Path) -> Result<Vec<f32>>,
    log: &mut dyn Sink<SpeakStep>,
) -> Result<Voice> {
    let key = prepared_reference_key(setup.checkpoint, setup.voice, &reference.id);
    let clone = clone_file(setup.cache_dir, &key, reference.path, decode)?;
    let mut transcript = None;
    if setup.hifi {
        let recording = decode(reference.path)?;
        let heard = judge.hear_reference(&recording, ANCHOR_RATE)?;
        let words: Vec<&str> = heard.words.iter().map(|w| w.text.as_str()).collect();
        transcript = non_empty(&words.join(" ")).map(str::to_string);
        if transcript.is_some() && !setup.style.is_empty() {
            log.step(SpeakStep::Anchor(AnchorStep::HifiIgnoresStyle));
        }
    }
    Ok(Voice {
        file: clone.file,
        transcript,
        voice_file: None,
    })
}

/// The designed voice `key` names, from the cache or designed now, and
/// copied beside the output.
fn designed_voice<E: SpeechEngine>(
    script: &str,
    setup: &VoxcpmSetup<'_>,
    seed: f64,
    key: &str,
    engine: &mut E,
    store: &mut FileStore<'_>,
    log: &mut dyn Sink<SpeakStep>,
) -> Result<Voice> {
    let anchor_setup = AnchorSetup {
        cache_dir: setup.cache_dir,
        key,
        voice: non_empty(setup.voice),
        style: non_empty(setup.style),
        seed,
        cfg: setup.cfg,
        steps: setup.steps,
        hifi: setup.hifi,
    };
    let download = setup.checkpoint.is_none();
    let mut steps = |step: AnchorStep| anchor_step(log, step, download);
    let anchor = designed_anchor(script, &anchor_setup, engine, store, &mut steps)?;
    let voice_file = copy_voice_beside(&anchor.file, setup.out)?;
    Ok(Voice {
        file: anchor.file,
        transcript: anchor.transcript,
        voice_file: Some(voice_file),
    })
}

/// Speaks `script` with VoxCPM2 into the voiceover at `setup.out`: the
/// skill's `voxcpm()`.
///
/// The parts are the whole script as one take ([`WHOLE_MAX`] words at
/// most), or one per line for a timed script. The seed and the take chosen
/// for each part come from `<out>.takes.json` ([`resolve_seed`]). The voice
/// is a user's recording, prepared once into the cache ([`clone_file`]),
/// or a voice designed from `setup.voice` and kept in the cache
/// ([`designed_anchor`]), then copied beside the output. With Hi-Fi, a
/// recording's transcript is heard through `judge`.
///
/// Each part's best take comes from [`TakeLoop::best_takes`], with takes
/// kept in the cache ([`FileStore`]) and `random` drawing each re-rolled
/// seed. The kept takes are joined, heat read against the reference's
/// median pitch, and written to `setup.out` with `<out>.takes.json`.
///
/// `decode` reads a recording as mono samples at [`ANCHOR_RATE`]. `engine`
/// speaks at [`JOIN_RATE`]. Every step goes to `progress` as it happens,
/// and the run keeps a copy.
pub fn speak_voxcpm<E, J>(
    script: &str,
    setup: &VoxcpmSetup<'_>,
    engine: &mut E,
    judge: &mut J,
    decode: &mut dyn FnMut(&Path) -> Result<Vec<f32>>,
    random: &mut dyn FnMut() -> f64,
    progress: &mut dyn Sink<SpeakStep>,
) -> Result<VoxcpmRun>
where
    E: SpeechEngine,
    J: TakeJudge<Verdict = JudgedTake> + ReferenceHearing,
{
    let mut log = Tee::new(progress);
    if setup.voice.is_empty() && setup.reference.is_none() {
        log.step(SpeakStep::NoVoice);
    }
    let units = parts_of(script, setup)?;
    let short = short_parts(&units);
    if !short.is_empty() {
        log.step(SpeakStep::ShortParts(short));
    }
    let record = takes_file(setup.out);
    let read = read_record(&record);
    let resolved = resolve_seed(setup.seed, &read, setup.config_seed);
    let seed = resolved.seed;
    let chosen = chosen_takes(read, &resolved);
    let designed_key = anchor_key(setup.checkpoint, setup.voice, seed, setup.language);
    let mut store = FileStore {
        cache_dir: setup.cache_dir,
        keys: TakeKeys {
            checkpoint: setup.checkpoint,
            voice: setup.voice,
            ref_id: setup.reference.map_or("", |r| r.id.as_str()),
            anchor_key: setup.reference.is_none().then_some(designed_key.as_str()),
            style: setup.style,
            cfg: setup.cfg,
            steps: setup.steps,
            hifi: setup.hifi,
        },
    };
    let voice = match setup.reference {
        Some(r) => cloned_voice(setup, r, judge, decode, &mut log)?,
        None => designed_voice(
            script,
            setup,
            seed,
            &designed_key,
            engine,
            &mut store,
            &mut log,
        )?,
    };
    let reference = read_wav_file(&voice.file)?;
    let mut take_loop = TakeLoop {
        setup: PickSetup {
            seed,
            chosen: &chosen,
            style: non_empty(setup.style),
            voice: VoiceSource::Clone {
                reference: CloneReference::Samples {
                    samples: &reference.samples,
                    rate: reference.rate,
                },
                transcript: voice.transcript.as_deref(),
            },
            cfg: setup.cfg,
            steps: setup.steps,
        },
        engine,
        judge,
        store: &mut store,
    };
    let download = setup.checkpoint.is_none();
    let best = {
        let mut steps = |step: TakeStep| take_step(&mut log, step, download);
        take_loop.best_takes(units, setup.rerolls, random, &mut steps)?
    };
    let voice_median = match setup.reference {
        Some(r) => median_pitch(&decode(r.path)?),
        None => median_pitch(&reference.samples),
    };
    let track = best.join(voice_median)?;
    write_whole_with(setup.out, &pcm16_wav(&track.samples, JOIN_RATE)?, OUT_NEXT)?;
    let takes = best.takes_file(seed).write()?;
    write_whole_with(&record, takes.as_bytes(), OUT_NEXT)?;
    let thumps: usize = best.best.iter().map(|b| b.verdict.thumps).sum();
    let fixed: usize = best.best.iter().map(|b| b.verdict.fixed).sum();
    if thumps > 0 || fixed > 0 {
        log.step(SpeakStep::Repaired { thumps, fixed });
    }
    let checks = best.checks(&track)?;
    let problems = best.problems();
    Ok(VoxcpmRun {
        seed,
        parts: best.parts,
        track,
        checks,
        problems,
        voice_file: voice.voice_file,
        voice_median,
        log: log.kept,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup<'a>(rerolls: &'a [usize], lines: bool) -> VoxcpmSetup<'a> {
        VoxcpmSetup {
            asr_model: None,
            cache_dir: Path::new("/cache"),
            out: Path::new("/out/voiceover.wav"),
            voice: "",
            style: "",
            reference: None,
            seed: None,
            config_seed: None,
            cfg: 1.6,
            steps: 16,
            hifi: false,
            language: None,
            rerolls,
            lines,
            checkpoint: None,
            device: None,
        }
    }

    #[test]
    fn a_whole_script_is_one_part_and_a_timed_one_a_part_a_line() -> Result<()> {
        let script = "One two three.\nFour five six.";
        assert_eq!(parts_of(script, &setup(&[], false))?.len(), 1);
        assert_eq!(parts_of(script, &setup(&[], true))?.len(), 2);
        Ok(())
    }

    #[test]
    fn an_empty_script_and_a_reroll_out_of_range_are_refused() {
        assert!(matches!(
            parts_of("  ", &setup(&[], false)),
            Err(Error::InvalidArgument { arg: "script", .. })
        ));
        // The skill's own words: `speak --reroll 2: the script has parts 1 to 1.`
        assert!(matches!(
            parts_of("One two three.", &setup(&[2], false)),
            Err(Error::InvalidArgument { arg: "rerolls", ref reason }) if reason == "--reroll 2: the script has parts 1 to 1."
        ));
        assert!(matches!(
            parts_of("One two three.", &setup(&[0], false)),
            Err(Error::InvalidArgument { arg: "rerolls", .. })
        ));
    }

    #[test]
    fn a_missing_record_reads_as_no_saved_takes() {
        assert_eq!(
            read_record(Path::new("/no/such/voiceover.takes.json")),
            ReadTakes::default()
        );
    }
}
