//! Paths as the skill's Node `path` module reads them: `resolve` against a
//! working directory and `relative` between two resolved paths, both
//! lexical, with no symbolic link followed.

use std::path::{Component, Path, PathBuf};

/// `path.resolve(cwd, p)`: `p` made absolute against `cwd` and normalized
/// lexically. `.` is dropped and `..` drops the component before it, never
/// past the root. `cwd` is absolute.
pub fn resolve(cwd: &Path, p: &Path) -> PathBuf {
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    };
    let mut out = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::Prefix(_) | Component::RootDir => out.push(c.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if out.parent().is_some() {
                    out.pop();
                }
            }
            Component::Normal(name) => out.push(name),
        }
    }
    out
}

/// `path.relative(from, to)`: the path from `from` to `to`, both already
/// resolved. Empty when they are the same path.
pub fn relative(from: &Path, to: &Path) -> String {
    let a: Vec<Component<'_>> = from.components().collect();
    let b: Vec<Component<'_>> = to.components().collect();
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let ups = a.len() - common;
    let mut parts: Vec<String> = std::iter::repeat_n("..".to_string(), ups).collect();
    parts.extend(
        b[common..]
            .iter()
            .map(|c| c.as_os_str().to_string_lossy().into_owned()),
    );
    parts.join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_normalizes_lexically() {
        let cwd = Path::new("/home/me/video");
        assert_eq!(
            resolve(cwd, Path::new("assets/voiceover.wav")),
            Path::new("/home/me/video/assets/voiceover.wav")
        );
        assert_eq!(
            resolve(cwd, Path::new("./a/../b/./c.wav")),
            Path::new("/home/me/video/b/c.wav")
        );
        assert_eq!(resolve(cwd, Path::new("/x/../../y")), Path::new("/y"));
        assert_eq!(resolve(cwd, Path::new("")), Path::new("/home/me/video"));
    }

    #[test]
    fn relative_walks_up_then_down() {
        let cwd = Path::new("/home/me/video");
        assert_eq!(
            relative(cwd, Path::new("/home/me/video/assets/v.wav")),
            "assets/v.wav"
        );
        assert_eq!(
            relative(cwd, Path::new("/home/me/other/v.wav")),
            "../other/v.wav"
        );
        assert_eq!(relative(cwd, cwd), "");
        assert_eq!(relative(cwd, Path::new("/home")), "../..");
    }
}
