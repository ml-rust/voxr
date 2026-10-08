//! The work directory a TTS command run writes its text and audio in: a
//! fresh directory under the system's temporary directory, removed with
//! everything in it once the run ends.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Result, file_error};

/// Names tried before a work directory is refused.
const TRIES: u32 = 16;

/// A work directory that removes itself when dropped.
pub(crate) struct WorkDir {
    path: PathBuf,
}

impl WorkDir {
    /// A new, empty `motion-video-tts-*` directory under
    /// [`std::env::temp_dir`], named by the process, the time and a count,
    /// as the skill's `mkdtemp` makes one.
    pub(crate) fn create() -> Result<Self> {
        static COUNT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut last = None;
        for _ in 0..TRIES {
            let n = COUNT.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "motion-video-tts-{}-{nanos:x}-{n}",
                std::process::id()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some((path, e)),
                Err(e) => {
                    return Err(file_error(
                        "creating the work directory",
                        &path,
                        &e,
                        "Check the temporary directory is writable",
                    ));
                }
            }
        }
        let (path, e) = last.unwrap_or_else(|| {
            (
                root.clone(),
                std::io::Error::from(std::io::ErrorKind::AlreadyExists),
            )
        });
        Err(file_error(
            "creating the work directory",
            &path,
            &e,
            "Remove stale motion-video-tts-* directories from the temporary directory",
        ))
    }

    /// The directory.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for WorkDir {
    /// Removes the directory and what the run left in it. A directory that
    /// cannot be removed stays: dropping has no caller to report to.
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_directory_is_new_and_removed_on_drop() -> Result<()> {
        let a = WorkDir::create()?;
        let b = WorkDir::create()?;
        assert_ne!(a.path(), b.path());
        assert!(a.path().is_dir());
        let file = a.path().join("speech.wav");
        std::fs::write(&file, b"x").map_err(|e| file_error("writing", &file, &e, "test"))?;
        let gone = a.path().to_path_buf();
        drop(a);
        assert!(!gone.exists());
        Ok(())
    }
}
