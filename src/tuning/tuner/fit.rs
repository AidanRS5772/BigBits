//! Fitting runtime equations to measurements: each algorithm's time in its theoretical form,
//! linear in its constants, fitted to its own timings. Every fit is found one way: on relative
//! error (prediction / time - 1), by least squares (`solve`: one SVD, which also gives each
//! constant's standard error and the design's condition number), judged on shapes it never saw
//! (`fit_split`). A constant is a time, so it can't be negative: a term that solves negative
//! isn't needed, is left out at 0, and the rest are solved again (`solve_nonneg`).
//! Which terms a form needs is the equation's author's question, not a tune's: `analysis`
//! (`cargo tune tune --analyze`) answers it from a fit's timings.

use nalgebra::{DMatrix, DVector};
use std::sync::atomic::{AtomicBool, Ordering};

/// A fitted equation's constants and how well it fits.
#[derive(Clone, Debug)]
pub struct Fit {
    pub beta: Vec<f64>,
    /// Each constant's standard error (0 for one no shape exercises, or left out).
    pub se: Vec<f64>,
    /// The design's condition number, its columns scaled to unit length.
    pub cond: f64,
    /// Root mean square of the relative residuals over every shape.
    pub rel_rms: f64,
    /// The largest |relative residual|.
    pub max_rel: f64,
    /// The row it is at and that residual, signed.
    pub worst: Option<(usize, f64)>,
    pub n: usize,
    /// (shapes, relative rms) of the held-out shapes under constants solved without them.
    pub held: Option<(usize, f64)>,
    /// The timings' own noise, where they were measured twice.
    pub noise: Option<f64>,
    /// The terms that solved negative and were left out: (index, the value solved, its standard error).
    pub dropped: Vec<(usize, f64, f64)>,
}

/// A fit is accepted when the relative rms it is judged by is at most this. The worst single
/// residual is reported and never judged.
pub const ACCEPT_RMS: f64 = 0.05;
/// In a rejected fit's evidence, which the tuner counts: reported with a proposed form, to wait for the user.
pub const REJECTED: &str = "REJECTED";
/// In the evidence of a fit above the line whose model was already revised once: accepted as it
/// is, and marked (docs/tuning_implementation.md, "Marked problems").
pub const NOISY: &str = "NOISY";

impl Fit {
    /// The relative rms the acceptance line reads: the held-out shapes', where some were held out.
    pub fn judged(&self) -> f64 {
        self.held.map_or(self.rel_rms, |h| h.1)
    }
    pub fn accepted(&self) -> bool {
        self.judged() <= ACCEPT_RMS
    }
    /// The evidence for a tuning file, for a fit on its model's first form.
    pub fn evidence(&self) -> String {
        self.evidence_with(false)
    }
    /// For a fit whose model was already revised once: above the line it reads NOISY, not REJECTED.
    pub fn evidence_revised(&self) -> String {
        self.evidence_with(true)
    }
    pub fn evidence_with(&self, revised: bool) -> String {
        let mut e = format!("fit: {} shapes; relative rms {:.1}%, worst {:.1}%", self.n, 100.0 * self.rel_rms, 100.0 * self.max_rel);
        if let Some((n, rms)) = self.held {
            e += &format!("; held out {:.1}% over {n}", 100.0 * rms);
        }
        if let Some(noise) = self.noise {
            e += &format!("; passes differ {:.1}%", 100.0 * noise);
        }
        e += &format!("; condition {:.0e}", self.cond);
        if !self.accepted() && revised {
            e += &format!("; {NOISY}: rms above {:.0}% after one revision of its model, accepted as it is (a marked problem)", 100.0 * ACCEPT_RMS);
        } else if !self.accepted() {
            e += &format!("; {REJECTED}: rms above {:.0}%", 100.0 * ACCEPT_RMS);
        }
        e
    }
    /// The terms left out for solving negative, by the names `keys` gives the constants (empty when none).
    pub fn dropped_text(&self, keys: &[&str]) -> String {
        if self.dropped.is_empty() {
            return String::new();
        }
        let terms: Vec<String> = self.dropped.iter().map(|&(j, v, se)| format!("{} solved {v:.3e} ± {se:.1e}", keys[j])).collect();
        format!(" [not needed, left out at 0: {}]", terms.join(", "))
    }
}

/// The root mean square of relative residuals (0 for none).
pub fn rel_rms(rel: &[f64]) -> f64 {
    (rel.iter().map(|r| r * r).sum::<f64>() / rel.len().max(1) as f64).sqrt()
}

/// A least-squares solution with what the solve says about it.
pub struct Solved {
    pub beta: Vec<f64>,
    pub se: Vec<f64>,
    pub cond: f64,
}

/// argmin |A x - b| by one SVD. Each column is scaled to unit length first, so the condition
/// number reads the design and not its units; a column of zeros reads 0. Standard errors are
/// from the residual variance and the diagonal of (A'A)^-1.
pub fn solve(a: &[Vec<f64>], b: &[f64]) -> Solved {
    let (m, n) = (a.len(), a[0].len());
    let norm: Vec<f64> = (0..n).map(|j| a.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt()).collect();
    let cols: Vec<usize> = (0..n).filter(|&j| norm[j] > 0.0).collect();
    let (mut beta, mut se) = (vec![0.0; n], vec![0.0; n]);
    if cols.is_empty() {
        return Solved { beta, se, cond: 1.0 };
    }
    let k = cols.len();
    let mat = DMatrix::from_fn(m, k, |i, j| a[i][cols[j]] / norm[cols[j]]);
    let rhs = DVector::from_column_slice(b);
    let svd = mat.clone().svd(true, true);
    let sol = svd.solve(&rhs, 1e-12).expect("an SVD with both factors solves");
    let sigma = &svd.singular_values;
    let (top, low) = (sigma.iter().cloned().fold(0.0, f64::max), sigma.iter().cloned().fold(f64::INFINITY, f64::min));
    let resid = (&mat * &sol - &rhs).norm_squared();
    let var = if m > k { resid / (m - k) as f64 } else { 0.0 };
    let v_t = svd.v_t.as_ref().expect("computed");
    for (j, &col) in cols.iter().enumerate() {
        // (A'A)^-1's diagonal, from V S^-2 V' over the directions the solve kept.
        let diag: f64 = (0..sigma.len()).filter(|&l| sigma[l] > 1e-12 * top).map(|l| (v_t[(l, j)] / sigma[l]).powi(2)).sum();
        beta[col] = sol[j] / norm[col];
        se[col] = (var * diag).sqrt() / norm[col];
    }
    Solved { beta, se, cond: if low > 0.0 { top / low } else { f64::INFINITY } }
}

/// `solve` with no constant below zero: while one solves negative, the term that takes the most
/// from the fit that way is left out and the rest solved again. Returns the solution and the
/// terms left out as (index, the value solved, its standard error).
pub fn solve_nonneg(a: &[Vec<f64>], b: &[f64]) -> (Solved, Vec<(usize, f64, f64)>) {
    let (mut a, mut dropped) = (a.to_vec(), Vec::new());
    loop {
        let s = solve(&a, b);
        let weight = |j: usize| s.beta[j] * a.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt();
        let Some(j) = (0..s.beta.len()).filter(|&j| s.beta[j] < 0.0).min_by(|&i, &j| weight(i).partial_cmp(&weight(j)).unwrap()) else {
            return (s, dropped);
        };
        dropped.push((j, s.beta[j], s.se[j]));
        for row in &mut a {
            row[j] = 0.0;
        }
    }
}

// A relative-error fit's rows: each regressor row over its time, against 1 less the part of the
// time already priced.
fn relative(x: &[Vec<f64>], t: &[f64], fixed: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let a = x.iter().zip(t).map(|(r, &ti)| r.iter().map(|v| v / ti).collect()).collect();
    let b = t.iter().zip(fixed).map(|(ti, f)| 1.0 - f / ti).collect();
    (a, b)
}

// Each row's relative residual under `beta`.
fn residuals(x: &[Vec<f64>], t: &[f64], fixed: &[f64], beta: &[f64]) -> Vec<f64> {
    x.iter().zip(t.iter().zip(fixed)).map(|(r, (&ti, f))| (r.iter().zip(beta).map(|(v, c)| v * c).sum::<f64>() + f) / ti - 1.0).collect()
}

/// beta >= 0 minimizing the sum of ((x_i . beta - t_i) / t_i)^2, with its report.
pub fn fit_rel(x: &[Vec<f64>], t: &[f64]) -> Fit {
    fit_rel_over(x, t, &vec![0.0; t.len()])
}

/// `fit_rel` over a part of each time that is already priced (`fixed`).
pub fn fit_rel_over(x: &[Vec<f64>], t: &[f64], fixed: &[f64]) -> Fit {
    assert!(x.len() == t.len() && fixed.len() == t.len());
    assert!(!t.is_empty() && t.iter().all(|&v| v > 0.0), "times must be positive");
    let (a, b) = relative(x, t, fixed);
    let (Solved { beta, se, cond }, dropped) = solve_nonneg(&a, &b);
    let rel = residuals(x, t, fixed, &beta);
    let worst = (0..rel.len()).max_by(|&a, &b| rel[a].abs().total_cmp(&rel[b].abs())).map(|i| (i, rel[i]));
    Fit { rel_rms: rel_rms(&rel), max_rel: rel.iter().fold(0.0f64, |m, r| m.max(r.abs())), worst, n: rel.len(), beta, se, cond, held: None, noise: None, dropped }
}

// The rows `held` marks (`want` true) or the others, of each of a fit's inputs.
fn rows_where(x: &[Vec<f64>], t: &[f64], fixed: &[f64], held: &[bool], want: bool) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let rows = (0..t.len()).filter(|&i| held[i] == want);
    (rows.clone().map(|i| x[i].clone()).collect(), rows.clone().map(|i| t[i]).collect(), rows.map(|i| fixed[i]).collect())
}

/// `fit_rel_over` judged on shapes it didn't fit: the constants solved without the rows `held`
/// marks give the held-out rms, and those solved on every row the rest of the report. With no
/// row held out, or none left to fit, it is `fit_rel_over`.
pub fn fit_split(x: &[Vec<f64>], t: &[f64], fixed: &[f64], held: &[bool]) -> Fit {
    assert!(held.len() == t.len());
    let mut fit = fit_rel_over(x, t, fixed);
    let ((xa, ta, fa), (xh, th, fh)) = (rows_where(x, t, fixed, held, false), rows_where(x, t, fixed, held, true));
    if !ta.is_empty() && !th.is_empty() {
        let (a, b) = relative(&xa, &ta, &fa);
        fit.held = Some((th.len(), rel_rms(&residuals(&xh, &th, &fh, &solve_nonneg(&a, &b).0.beta))));
    }
    fit
}

// Whether fits print their form's analysis (`--analyze`).
static ANALYZE: AtomicBool = AtomicBool::new(false);

pub fn set_analyze(on: bool) {
    ANALYZE.store(on, Ordering::Relaxed);
}

/// One equation's fit as every section takes it: `fit_split`, the form `name` analysed first
/// when the tune asked for it (`keys` name the constants), the timings' `noise` in the record.
pub fn fit_named(name: &str, keys: &[&str], x: &[Vec<f64>], t: &[f64], fixed: &[f64], held: &[bool], noise: Option<f64>) -> Fit {
    if ANALYZE.load(Ordering::Relaxed) {
        eprint!("{}", analysis(name, keys, x, t, fixed, held));
    }
    Fit { noise, ..fit_split(x, t, fixed, held) }
}

/// The developer's analysis of a form, from one fit's inputs: which of its terms the timings
/// need. First the form solved with no constraint: each constant against its standard error
/// (a ratio under about 2: the timings can't tell it from 0) and the term's share of the
/// predicted time. Then every form short of some terms that solves with no negative constant,
/// the best of each size by its held-out rms, and the smallest within 0.2 points of the best.
pub fn analysis(name: &str, keys: &[&str], x: &[Vec<f64>], t: &[f64], fixed: &[f64], held: &[bool]) -> String {
    let k = keys.len();
    assert!(k == x[0].len() && k <= 14, "one name a term, and few enough terms to try every form");
    let (a, b) = relative(x, t, fixed);
    let full = solve(&a, &b);
    let width = keys.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut out = format!("  form analysis: {name} ({} shapes, {} held out; condition {:.0e})\n", t.len(), held.iter().filter(|h| **h).count(), full.cond);
    out += &format!("    {:<width$}  {:>11} {:>9} {:>6}   share of the time: mean, largest\n", "term, all solved freely", "constant", "± se", "t");
    for j in 0..k {
        let shares: Vec<f64> = x.iter().zip(fixed).map(|(r, f)| full.beta[j] * r[j] / (r.iter().zip(&full.beta).map(|(v, c)| v * c).sum::<f64>() + f)).collect();
        let mean = shares.iter().sum::<f64>() / shares.len() as f64;
        let most = shares.iter().cloned().fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });
        let tr = if full.se[j] > 0.0 { full.beta[j] / full.se[j] } else { f64::INFINITY * full.beta[j].signum() };
        out += &format!("    {:<width$}  {:>11.4e} {:>9.1e} {:>6.1}   {:>5.1}%, {:>5.1}%\n", keys[j], full.beta[j], full.se[j], tr, 100.0 * mean, 100.0 * most);
    }
    // Every form: the terms `mask` keeps.
    let mut best: Vec<Option<(u32, Fit)>> = vec![None; k + 1];
    for mask in 1u32..1 << k {
        let rows: Vec<Vec<f64>> = x.iter().map(|r| (0..k).filter(|j| mask >> j & 1 == 1).map(|j| r[j]).collect()).collect();
        let f = fit_split(&rows, t, fixed, held);
        let size = mask.count_ones() as usize;
        if f.dropped.is_empty() && best[size].as_ref().is_none_or(|(_, g)| f.judged() < g.judged()) {
            best[size] = Some((mask, f));
        }
    }
    let names = |mask: u32| (0..k).filter(|j| mask >> j & 1 == 1).map(|j| keys[j]).collect::<Vec<_>>().join(" + ");
    out += "    forms with no negative constant, the best of each size:   held out    fitted     worst\n";
    for size in (1..=k).rev() {
        match &best[size] {
            Some((mask, f)) => out += &format!("      {size} terms: {:<52} {:>6.1}%   {:>6.1}%   {:>6.1}%\n", names(*mask), 100.0 * f.judged(), 100.0 * f.rel_rms, 100.0 * f.max_rel),
            None => out += &format!("      {size} terms: none (every form of this size solves a constant negative)\n"),
        }
    }
    let least = best.iter().flatten().map(|(_, f)| f.judged()).fold(f64::INFINITY, f64::min);
    if let Some((mask, f)) = best.iter().flatten().find(|(_, f)| f.judged() <= least + 0.002) {
        out += &format!("    the smallest within 0.2 points of the best ({:.1}%): {} at {:.1}%\n", 100.0 * least, names(*mask), 100.0 * f.judged());
    }
    out
}
