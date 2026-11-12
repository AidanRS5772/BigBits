//! The shapes a fit samples (docs/tuning_system.md, "How knobs are tuned"). An equation over two
//! inputs (or three: the series' frame, terms and x) is fitted on shapes scattered over the area
//! where its algorithm competes, not along a few fixed ratios, so a misfit anywhere in the area
//! shows; and every other shape is held out, for the fit to be judged on shapes it never saw
//! (`fit::fit_split`).

/// `n` points over [a.0, a.1] x [b.0, b.1], spread evenly in the logarithm of each coordinate: the
/// R2 sequence (an additive recurrence on the plastic number's powers), whose points fill the
/// area without lining up, so n points give n distinct values of each coordinate where a grid
/// gives √n. Every other point of it is again such a sequence, which is what the held-out half
/// relies on. The points are the sequence's `from + 1` to `from + n`: a report that judges
/// equations on shapes no tune fitted them to takes its points far along (`UNSEEN`).
pub fn scatter(from: usize, n: usize, a: (f64, f64), b: (f64, f64)) -> Vec<(f64, f64)> {
    const STEP: (f64, f64) = (0.754_877_666_246_692_7, 0.569_840_290_998_053_2);
    (from + 1..=from + n)
        .map(|i| {
            let (u, v) = ((0.5 + STEP.0 * i as f64).fract(), (0.5 + STEP.1 * i as f64).fract());
            (a.0 * (a.1 / a.0).powf(u), b.0 * (b.1 / b.0).powf(v))
        })
        .collect()
}

/// `scatter` over three coordinates: the R3 sequence (the same recurrence on the powers of the
/// root of x^4 = x + 1), for a fit whose shapes have three dials.
pub fn scatter3(from: usize, n: usize, a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> Vec<(f64, f64, f64)> {
    const STEP: (f64, f64, f64) = (0.819_172_513_396_164_4, 0.671_043_606_703_789_2, 0.549_700_477_901_970_2);
    let at = |r: (f64, f64), step: f64, i: usize| r.0 * (r.1 / r.0).powf((0.5 + step * i as f64).fract());
    (from + 1..=from + n).map(|i| (at(a, STEP.0, i), at(b, STEP.1, i), at(c, STEP.2, i))).collect()
}

/// Where a report's points start in the sequence: past every point any level's tune takes.
pub const UNSEEN: usize = 1_000_000;

/// A fit's shapes, and which of them are held out.
pub struct Design<S> {
    pub shapes: Vec<S>,
    pub held: Vec<bool>,
}

impl<S: Copy + PartialEq> Design<S> {
    /// The design of the points `scatter` gives, each made a shape by `shape` (None: outside the
    /// algorithm's own shapes); a shape that repeats an earlier one is left out, and every other
    /// one left is held out.
    pub fn of(from: usize, n: usize, a: (f64, f64), b: (f64, f64), shape: impl Fn(f64, f64) -> Option<S>) -> Self {
        let mut shapes: Vec<S> = Vec::new();
        for (x, y) in scatter(from, n, a, b) {
            if let Some(s) = shape(x, y).filter(|s| !shapes.contains(s)) {
                shapes.push(s);
            }
        }
        let held = (0..shapes.len()).map(|i| i % 2 == 1).collect();
        Design { shapes, held }
    }

    /// The same shapes held out two by two (two fitted, two held) in place of every other one.
    /// A coordinate whose step is near a half (the R3 sequence's third, 0.55, and the R2's
    /// second, 0.57) puts alternate points in opposite halves of its range, and each half-sequence
    /// then crosses the range only once in about ten points: with a dozen points to a set, one
    /// half of a split by parity holds the low values and the other the high ones (a series
    /// fit's first trial held out 15 of rect's shapes and fitted 4). A pair spans both halves.
    pub fn in_pairs(mut self) -> Self {
        self.held = (0..self.shapes.len()).map(|i| i / 2 % 2 == 1).collect();
        self
    }

    /// `of` over three coordinates (`scatter3`).
    pub fn of3(from: usize, n: usize, a: (f64, f64), b: (f64, f64), c: (f64, f64), shape: impl Fn(f64, f64, f64) -> Option<S>) -> Self {
        let mut shapes: Vec<S> = Vec::new();
        for (x, y, z) in scatter3(from, n, a, b, c) {
            if let Some(s) = shape(x, y, z).filter(|s| !shapes.contains(s)) {
                shapes.push(s);
            }
        }
        let held = (0..shapes.len()).map(|i| i % 2 == 1).collect();
        Design { shapes, held }
    }
}

/// Products l x s with s over `sizes` and l / s over `ratios`, kept where `keep(l, s)`.
pub fn products(n: usize, sizes: (usize, usize), ratios: (f64, f64), keep: impl Fn(usize, usize) -> bool) -> Design<(usize, usize)> {
    Design::of(0, n, (sizes.0 as f64, sizes.1 as f64), ratios, |s, r| {
        let s = s.round() as usize;
        let l = ((s as f64 * r).round() as usize).max(s);
        keep(l, s).then_some((l, s))
    })
}
