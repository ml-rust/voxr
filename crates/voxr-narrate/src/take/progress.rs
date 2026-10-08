//! Where a run's steps go as each one happens: a [`Sink`] hands each step
//! on at once, so a caller prints a long run's progress live, and a
//! [`Tee`] keeps a copy of every step it passes on.

/// Takes each step of a run as it happens.
pub trait Sink<T> {
    /// One step, in the order the run takes them.
    fn step(&mut self, step: T);
}

/// A list keeps every step.
impl<T> Sink<T> for Vec<T> {
    fn step(&mut self, step: T) {
        self.push(step);
    }
}

/// A closure takes every step.
impl<T, F: FnMut(T)> Sink<T> for F {
    fn step(&mut self, step: T) {
        self(step);
    }
}

/// Passes each step on to `out`, and keeps a copy.
pub struct Tee<'a, T> {
    /// Every step passed on, in order.
    pub kept: Vec<T>,
    out: &'a mut dyn Sink<T>,
}

impl<'a, T> Tee<'a, T> {
    /// A tee in front of `out`, with nothing kept yet.
    pub fn new(out: &'a mut dyn Sink<T>) -> Self {
        Self {
            kept: Vec::new(),
            out,
        }
    }
}

impl<T: Clone> Sink<T> for Tee<'_, T> {
    fn step(&mut self, step: T) {
        self.out.step(step.clone());
        self.kept.push(step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tee_passes_each_step_on_at_once_and_keeps_it() {
        let mut seen = Vec::new();
        let mut live = |s: u32| seen.push(s * 10);
        let mut tee = Tee::new(&mut live);
        tee.step(1);
        tee.step(2);
        assert_eq!(tee.kept, [1, 2]);
        drop(tee);
        assert_eq!(seen, [10, 20]);
    }
}
