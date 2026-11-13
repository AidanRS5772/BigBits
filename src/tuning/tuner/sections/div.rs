//! Section `div`. Sweeps with the algorithm forced through the prepared drivers: the dynamic
//! reciprocal cutoff (Knuth against NR) and BZ's leaf. The quotient dispatchers have two inputs
//! (q, d): `fit_equations` fits Knuth's, BZ's and NR's runtime equations to `division_timings`,
//! `small_table` measures the tables for divisions too small to read them, `tune_late` fits
//! again on the shapes the tables leave (`Rows`), and `division` is `cargo tune division`.

use crate::tuning::resolve::Value;
use crate::tuning::tuner::design::{Design, UNSEEN};
use crate::tuning::tuner::fit::{fit_named, Fit};
use crate::tuning::tuner::measure::{dump, less_harness, pad, time_fit, verbose, Level, PadBuf, Probe};
use crate::tuning::tuner::out::{push_fit as push, Tuned};
use crate::tuning::tuner::sweep::{cutoff_sweep, geo_to, keep, lost};
use crate::tuning::tuner::tables::{self, Plan, Row, Spec};
use crate::tuning::{install, prepare};
use crate::utils::div::{
    div_dyn, div_prepared_dyn, div_rem_dyn, div_rem_prepared_dyn, division_preflight, dyn_div_alg_at, dyn_div_can, dyn_div_has_choice,
    dyn_div_small, rcp_prepared_dyn, use_bz_for_top_block, use_bz_for_top_block_at, DivAlg, DivEq, RcpAlg, BZ_TOP_FORCE, DIV_SMALL_PICKS,
};
use crate::utils::mul::MulEq;
use crate::tuning::table::Table2;
use std::cell::RefCell;
use std::hint::black_box;
use std::sync::atomic::Ordering;

// A divisor of n limbs with its top bit set.
fn divisor(n: usize, pass: usize) -> Vec<u64> {
    let mut d = pad(n, n as u64 * 41 + 3, pass).to_vec();
    d[n - 1] |= 1 << 63;
    d
}

/// The numerator of a probe with a logical quotient of q limbs by a divisor of d: (q - 1) + d limbs.
pub(crate) fn numerator(q: usize, d: usize, pass: usize) -> PadBuf {
    pad(q - 1 + d, (q * 31 + d) as u64, pass)
}

/// Its divisor, top bit set, 13 limbs further into its allocation in the second pass than the
/// numerator: Knuth's loads from the divisor can alias its stores to the numerator.
pub(crate) fn placed_divisor(d: usize, pass: usize) -> PadBuf {
    let mut dv = pad(d, d as u64 * 41 + 3, 2 * pass);
    dv[d - 1] |= 1 << 63;
    dv
}

/// A probe with a remainder restores its numerator before each call: that copy alone, timed on
/// its own and taken off the probe's time (`less_harness`).
pub(crate) fn copy_probe(q: usize, d: usize, pass: usize) -> Probe<'static> {
    let (src, mut dst) = (numerator(q, d, pass), numerator(q, d, pass));
    Probe::new("copy", move || {
        dst.copy_from_slice(&src);
        black_box(&dst);
    })
}

pub fn tune(level: Level) -> Vec<Tuned> {
    let saved = crate::tuning::snapshot();
    let mut out = Vec::new();

    // Reciprocals of R limbs from an R-limb divisor: Knuth while R <= knob. Only the dynamic one:
    // the static cutoff also depends on N.
    let rcp: fn(&[u64], &mut [u64], RcpAlg) = rcp_prepared_dyn;
    let sizes: Vec<usize> = (6..=160).collect();
    out.push(cutoff_sweep(level, "div.dyn_rcp_knuth_nr", &sizes, None, 0, &|r, p| {
        let at = |alg: RcpAlg| {
            let d = divisor(r, p);
            let mut x = pad(r, 7, p);
            Probe::new("rcp", move || {
                rcp(black_box(&d), &mut x, alg);
                black_box(&x);
            })
        };
        [at(RcpAlg::Knuth), at(RcpAlg::NR)]
    }));

    // BZ's leaf at 2d / d: Knuth against one BZ level over Knuth leaves (the knob at half of d
    // inside the probe), each probe running both entries. The static entries share the knob and
    // cross later (approximate).
    let coarse = geo_to(16.0, 8.0, 600);
    let t = cutoff_sweep(level, "div.bz", &coarse, Some(600), 0, &|d_len, p| {
        let at = |alg: DivAlg, leaf: u64| {
            let (n, d) = (pad(2 * d_len, d_len as u64 * 43, p), divisor(d_len, p));
            let (mut q, mut work) = (pad(d_len, 11, p), n.to_vec());
            let hd = prepare("div.bz", &Value::Int(leaf)).expect("a valid div.bz");
            Probe::new("div", move || {
                install(hd);
                let r = division_preflight(&n, &d, &mut q).expect("a nonempty quotient");
                black_box(div_prepared_dyn(black_box(&n), black_box(&d), &mut q, r, alg));
                work.copy_from_slice(&n);
                let r = division_preflight(&work, &d, &mut q).expect("a nonempty quotient");
                black_box(div_rem_prepared_dyn(&mut work, black_box(&d), &mut q, r, alg));
            })
        };
        [at(DivAlg::Knuth, usize::MAX as u64), at(DivAlg::BZ, (d_len.div_ceil(2)).max(3) as u64)]
    });
    // The leaf is live for what follows: BZ's timings and its equation read it.
    crate::tuning::restore(&saved);
    keep("div", &mut out, t);

    // The first fit, on every shape: the tables are measured under it, then `tune_late` fits again.
    let (timings, noise) = division_timings(level, false);
    let (tuned, _) = fit_equations(&timings, noise, Rows::First);
    out.extend(tuned);
    out.extend(top_block_scale(&timings, level.eps()));
    KEPT.with(|k| *k.borrow_mut() = Some((timings, noise)));
    crate::tuning::restore(&saved);
    out
}

thread_local! {
    // A tune's timings and their noise, from its first fit to its second.
    static KEPT: RefCell<Option<(Vec<DivTiming>, f64)>> = const { RefCell::new(None) };
}

/// Which timings the equations' fits take: an equation is fitted on the shapes it is read for.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Rows {
    /// Every shape: a tune's first fit, whose evidence carries no verdict.
    First,
    /// The shapes each equation is read for under the tables in force (`read_for`).
    Read,
}

/// Whether the dispatch reads its equations for a division: not where the entry's table names an
/// algorithm that can take the shape. NR's and BZ's fits take only these; Knuth's takes every
/// shape, as its equation is also a price inside the others.
pub fn read_for(rem: bool, q: usize, d: usize, tables: [&Table2; 2], leaf: usize) -> bool {
    dyn_div_small(tables[rem as usize], q, d, leaf).is_none()
}

/// Second phase, once the tables are in force: the equations fitted again from the first
/// phase's timings. Nothing without a first phase.
pub fn tune_late(_: Level) -> Vec<Tuned> {
    let Some((timings, noise)) = KEPT.with(|k| k.borrow_mut().take()) else { return Vec::new() };
    fit_equations(&timings, noise, Rows::Read).0
}

/// Static BZ's top-block rule (`use_bz_for_top_block`) keeps one constant, swept here on the
/// dynamic timings with the top block forced each way: the scale that loses least. `tie` is the
/// tie band.
fn top_block_scale(t: &[DivTiming], tie: f64) -> Option<Tuned> {
    let both: Vec<&DivTiming> = t.iter().filter(|x| x.bz_top.iter().all(|v| v.is_finite())).collect();
    if both.is_empty() {
        return None;
    }
    let leaf = crate::tuning::k::div::bz();
    // (padded ns, Knuth ns) of each shape.
    let tops: Vec<(f64, f64)> = both.iter().map(|x| (x.bz_top[0], x.bz_top[1])).collect();
    let loss = |scale: f64| lost(&tops, |i| use_bz_for_top_block_at(both[i].d, (both[i].q - 1) % both[i].d + 1, leaf, scale));
    // 1/4 to 16, eight a doubling.
    let cands: Vec<f64> = (-16..=32).map(|i| 2f64.powf(i as f64 / 8.0)).collect();
    let losses: Vec<(f64, f64)> = cands.iter().map(|&c| loss(c)).collect();
    let b = (0..cands.len()).min_by(|&a, &b| losses[a].0.partial_cmp(&losses[b].0).unwrap()).expect("candidates");
    let previous = crate::tuning::k::div::bz_top_padded_cost_scale();
    let (best, lb, lp) = (cands[b], losses[b], loss(previous));
    let range = format!("sweep {:.2}-{:.0}, {} values over {} shapes of forced dynamic BZ", cands[0], cands[cands.len() - 1], cands.len(), both.len());
    let (value, evidence) = if lp.0 - lb.0 <= tie {
        (previous, format!("{range}: kept, measured time lost mean {:.2}%, worst {:.1}%, within the tie band of the best ({best:.3}: {:.2}%)", 100.0 * lp.0, 100.0 * lp.1, 100.0 * lb.0))
    } else {
        (best, format!("{range}: measured time lost mean {:.2}%, worst {:.1}%; the previous value {previous} lost {:.2}%, worst {:.1}%", 100.0 * lb.0, 100.0 * lb.1, 100.0 * lp.0, 100.0 * lp.1))
    };
    Some(Tuned { key: "bz_top_padded_cost_scale", value: Value::Float(value), evidence })
}

/// One shape's timings: with or without remainder, logical quotient width q and divisor width
/// d, whether a fit holds it out, each forced algorithm's ns (infinite where not timed), BZ's
/// with its top block forced padded and Knuth, and the public entry's.
#[derive(Clone, Copy, Debug)]
pub struct DivTiming {
    pub rem: bool,
    pub held: bool,
    pub q: usize,
    pub d: usize,
    pub ns: [f64; 3],
    pub bz_top: [f64; 2],
    pub auto: f64,
}

pub const DIV_ALGS: [DivAlg; 3] = [DivAlg::Knuth, DivAlg::BZ, DivAlg::NR];

/// Knuth, BZ and NR forced over (q, d) scattered from 4 to 4096 limbs each, for both entries,
/// every other shape held out; returns the timings and their noise. Knuth is skipped past 2^22
/// limb products. A tune times only what the fits read; `report` adds the entry and BZ under its
/// own top-block choice, on shapes no tune takes.
pub fn division_timings(level: Level, report: bool) -> (Vec<DivTiming>, f64) {
    let design = Design::of(if report { UNSEEN } else { 0 }, level.fit_shapes(200), (4.0, 4096.0), (4.0, 4096.0), |q, d| {
        Some((q.round() as usize, d.round() as usize))
    });
    let shapes: Vec<(bool, usize, usize, bool)> =
        [false, true].into_iter().flat_map(|rem| design.shapes.iter().zip(&design.held).map(move |(&(q, d), &held)| (rem, q, d, held))).collect();
    let knuth_ok = |q: usize, d: usize| q * d <= 1 << 22;
    let leaf = crate::tuning::k::div::bz();
    // A shape's probes, in order (force 1: top block padded, 2: Knuth).
    let runs = [(Some(DivAlg::Knuth), 0u8), (Some(DivAlg::BZ), 0), (Some(DivAlg::NR), 0), (None, 0), (Some(DivAlg::BZ), 1), (Some(DivAlg::BZ), 2)];
    let timed = |alg: Option<DivAlg>, force: u8, q: usize, d: usize| {
        let can = !(alg == Some(DivAlg::Knuth) && !knuth_ok(q, d)) && !(force != 0 && d <= leaf);
        // The fits read Knuth, NR and BZ's two forced top blocks.
        can && (report || force != 0 || matches!(alg, Some(DivAlg::Knuth | DivAlg::NR)))
    };
    let build = |pass: usize| -> Vec<Probe<'static>> {
        let mut probes = Vec::new();
        for &(rem, q, d, _) in &shapes {
            // The numerator's copy is the harness's work: timed first, taken off the shape's timings.
            if rem {
                probes.push(copy_probe(q, d, pass));
            }
            for (alg, force) in runs {
                if !timed(alg, force, q, d) {
                    continue;
                }
                let dv = placed_divisor(d, pass);
                let (n0, mut n, mut quo) = (numerator(q, d, pass), numerator(q, d, pass), pad(q - 1, 7, pass));
                probes.push(Probe::new("div", move || {
                    if rem {
                        n.copy_from_slice(&n0);
                    }
                    BZ_TOP_FORCE.store(force, Ordering::Relaxed);
                    let top = match (rem, alg) {
                        (false, Some(a)) => {
                            let r = division_preflight(&n, &dv, &mut quo).expect("a nonempty request");
                            div_prepared_dyn(&n, &dv, &mut quo, r, a)
                        }
                        (true, Some(a)) => {
                            let r = division_preflight(&n, &dv, &mut quo).expect("a nonempty request");
                            div_rem_prepared_dyn(&mut n, &dv, &mut quo, r, a)
                        }
                        (false, None) => div_dyn(&n, &dv, &mut quo),
                        (true, None) => div_rem_dyn(&mut n, &dv, &mut quo),
                    };
                    BZ_TOP_FORCE.store(0, Ordering::Relaxed);
                    black_box(top);
                }));
            }
        }
        probes
    };
    let measured = time_fit(level, build);
    let t = &measured.ns;
    // One mistimed copy would bend its shape's timings: each is held to three times the median
    // rate a limb over all copies, plus 5 ns for the call.
    let probes_of = |rem: bool, q: usize, d: usize| rem as usize + runs.iter().filter(|&&(alg, force)| timed(alg, force, q, d)).count();
    let rate = {
        let mut at = 0;
        let mut r = Vec::new();
        for &(rem, q, d, _) in &shapes {
            if rem {
                r.push(t[at] / (q - 1 + d) as f64);
            }
            at += probes_of(rem, q, d);
        }
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        r.get(r.len() / 2).copied().unwrap_or(0.0)
    };
    let mut bounded = 0;
    let mut at = 0;
    let timings = shapes
        .iter()
        .map(|&(rem, q, d, held)| {
            let bound = 3.0 * rate * (q - 1 + d) as f64 + 5.0;
            let copy = if rem { t[at].min(bound) } else { 0.0 };
            bounded += (rem && t[at] > bound) as usize;
            at += rem as usize;
            let (mut ns, mut bz_top, mut auto) = ([f64::INFINITY; 3], [f64::INFINITY; 2], f64::INFINITY);
            for (alg, force) in runs {
                if !timed(alg, force, q, d) {
                    continue;
                }
                let own = less_harness(t[at], copy);
                at += 1;
                match (alg, force) {
                    (Some(a), 0) => ns[alg_index(a)] = own,
                    (None, _) => auto = own,
                    (_, f) => bz_top[f as usize - 1] = own,
                }
            }
            DivTiming { rem, held, q, d, ns, bz_top, auto }
        })
        .collect();
    if bounded > 0 {
        eprintln!("  division timings: {bounded} numerator copies timed past 3x the median rate ({rate:.3} ns a limb) and held to it");
    }
    (timings, measured.noise)
}

fn alg_index(a: DivAlg) -> usize {
    DIV_ALGS.iter().position(|x| *x == a).expect("one of the three")
}

// The knobs each equation's fit sets, in its constants' order.
const KNUTH: [&str; 4] = ["knuth_ns_qd", "knuth_ns_q", "knuth_ns_d", "knuth_ns_shift"];
const NR: [&str; 5] = ["nr_ns_mid_school", "nr_ns_mid_fft", "nr_ns_hi_school", "nr_ns_hi_guarded", "nr_ns_q"];
const NR_REM: [&str; 7] =
    ["nr_rem_ns_mid_school", "nr_rem_ns_mid_fft", "nr_rem_ns_hi_school", "nr_rem_ns_hi_guarded", "nr_rem_ns_prod", "nr_rem_ns_q", "nr_rem_ns_d"];
const BZ: [&str; 2] = ["bz_ns_block", "bz_ns_top"];

/// The four equations fitted to a set of timings, as tuned values and as the equations: Knuth
/// and BZ on both entries' timings, NR on each entry's own (where it can run: q >= 8), BZ from
/// its two forced top-block modes. Under `Rows::Read` NR's and BZ's fits leave out the shapes
/// the tables decide.
pub(crate) fn fit_equations(t: &[DivTiming], noise: f64, rows: Rows) -> (Vec<Tuned>, DivEq) {
    let (mul, kara) = (MulEq::read(), crate::tuning::k::mul::kara());
    let mut eq = DivEq::read();
    let mut out = Vec::new();
    let (small, leaf) = ([crate::tuning::k::div::dyn_small(), crate::tuning::k::div::dyn_rem_small()], eq.bz_leaf);
    let read = |x: &DivTiming| rows == Rows::First || read_for(x.rem, x.q, x.d, small, leaf);
    // A first fit's evidence carries no verdict.
    let said = |f: &Fit, revised: bool| match rows {
        Rows::First => format!("a first fit, which the tables are measured under (fitted again after them, on the shapes they leave to the equations): {} shapes, held out {:.1}%", f.n, 100.0 * f.judged()),
        Rows::Read => f.evidence_with(revised),
    };
    // What a fit left to the tables, for its evidence.
    let left = |all: usize, kept: usize, table: &str| match rows {
        Rows::First => String::new(),
        Rows::Read => format!(", not the {} of {all} shapes {table} an algorithm for", all - kept),
    };
    let m = |l: usize, s: usize| mul.mul_ns(l, s, kara);
    // Where a fit's largest residual is; `rows` are the timings its rows came from.
    fn worst_at(f: &Fit, rows: &[&DivTiming]) -> String {
        f.worst.map_or(String::new(), |(i, _)| format!("; worst at q {} d {}{}", rows[i].q, rows[i].d, if rows[i].rem { " with remainder" } else { "" }))
    }
    let fit = |name: &str, keys: &[&'static str], x: &[Vec<f64>], y: &[f64], rows: &[&DivTiming]| -> Fit {
        let held: Vec<bool> = rows.iter().map(|r| r.held).collect();
        fit_named(name, keys, x, y, &vec![0.0; y.len()], &held, Some(noise))
    };

    let rows: Vec<&DivTiming> = t.iter().filter(|x| x.ns[0].is_finite()).collect();
    let (x, y): (Vec<Vec<f64>>, Vec<f64>) = rows.iter().map(|x| (vec![(x.q * x.d) as f64, x.q as f64, x.d as f64, 1.0], x.ns[0])).unzip();
    let f = fit("div Knuth (forced, both entries): a q d + b q + c d + e", &KNUTH, &x, &y, &rows);
    eq.knuth.copy_from_slice(&f.beta);
    let at = worst_at(&f, &rows);
    push(&mut out, &KNUTH, &f, said(&f, false) + &at + " (Knuth forced, both entries, every shape: its equation is also a price in BZ's and in a series' finalize; a q d + b q + c d + e)");

    // NR: its products by tier (`DivEq::nr_parts`), one multiple a tier. A tier no shape reaches
    // keeps the multiple in force.
    let start = eq;
    let tiers = |q: usize| start.nr_parts(q, &mul, kara);
    let keep_unseen = |x: &[Vec<f64>], beta: &mut [f64], before: &[f64]| {
        for i in 0..4 {
            if x.iter().all(|row| row[i] == 0.0) {
                beta[i] = before[i];
            }
        }
    };
    let can = t.iter().filter(|x| !x.rem && x.q >= 8).count();
    let rows: Vec<&DivTiming> = t.iter().filter(|x| !x.rem && x.q >= 8 && read(x)).collect();
    let x: Vec<Vec<f64>> = rows.iter().map(|r| tiers(r.q).into_iter().chain([r.q as f64]).collect()).collect();
    let y: Vec<f64> = rows.iter().map(|r| r.ns[2]).collect();
    let mut f = fit("div NR (forced, quotient only, q >= 8): its products by tier + b q", &NR, &x, &y, &rows);
    keep_unseen(&x, &mut f.beta, &start.nr);
    eq.nr.copy_from_slice(&f.beta);
    let at = worst_at(&f, &rows);
    push(
        &mut out,
        &NR,
        &f,
        said(&f, true)
            + &at
            + &format!(
                " (NR forced, quotient only, q >= 8{}; its middle and high products by tier, each a multiple of mul's price at its own shape: school middle, FFT middle, school high, guarded high, + b q)",
                left(can, rows.len(), "div.dyn_small names")
            ),
    );

    let can = t.iter().filter(|x| x.rem && x.q >= 8).count();
    let rows: Vec<&DivTiming> = t.iter().filter(|x| x.rem && x.q >= 8 && read(x)).collect();
    let x: Vec<Vec<f64>> =
        rows.iter().map(|r| tiers(r.q).into_iter().chain([m(r.d, (r.q - 1).min(r.d + 1)), r.q as f64, r.d as f64]).collect()).collect();
    let y: Vec<f64> = rows.iter().map(|r| r.ns[2]).collect();
    let mut f = fit("div NR with remainder (forced, q >= 8): the four tiers + r Mul(d, min(q - 1, d + 1)) + b q + c d", &NR_REM, &x, &y, &rows);
    keep_unseen(&x, &mut f.beta, &start.nr_rem);
    eq.nr_rem.copy_from_slice(&f.beta);
    let at = worst_at(&f, &rows);
    push(
        &mut out,
        &NR_REM,
        &f,
        said(&f, true)
            + &at
            + &format!(
                " (NR forced, with remainder, q >= 8{}; the same four tiers + r Mul(d, min(q - 1, d + 1)) + b q + c d)",
                left(can, rows.len(), "div.dyn_rem_small names")
            ),
    );

    // BZ over Knuth's equation just fitted, with the top block forced each way so the fit doesn't
    // depend on the rule it feeds: a (blocks (P + L) + padded top) + b Knuth(top, d). A Knuth top
    // over no whole block is Knuth alone and is left out.
    let unit = DivEq { bz: [1.0, 1.0], ..eq };
    let (mut x, mut y, mut shapes) = (Vec::new(), Vec::new(), Vec::new());
    let can = t.iter().filter(|r| r.d > eq.bz_leaf).count();
    let mut kept = 0;
    for r in t.iter().filter(|r| r.d > eq.bz_leaf && read(r)) {
        kept += 1;
        let (blocks, top) = ((r.q - 1) / r.d, (r.q - 1) % r.d + 1);
        let (products, leaves) = unit.bz_block_parts(r.d, &mul, kara);
        let whole = blocks as f64 * (products + leaves);
        if r.bz_top[0].is_finite() {
            x.push(vec![whole + unit.bz_padded_top_ns(top, r.d, &mul, kara), 0.0]);
            y.push(r.bz_top[0]);
            shapes.push((r, "padded", unit.bz_leaves_ns(r.d)));
        }
        if r.bz_top[1].is_finite() && blocks > 0 {
            x.push(vec![whole, unit.knuth_ns(top, r.d)]);
            y.push(r.bz_top[1]);
            shapes.push((r, "knuth", 0.0));
        }
    }
    let of: Vec<&DivTiming> = shapes.iter().map(|s| s.0).collect();
    let f = fit("div BZ (forced, top block padded and Knuth, both entries): a (blocks + padded top) + b Knuth top", &BZ, &x, &y, &of);
    eq.bz.copy_from_slice(&f.beta);
    let at = worst_at(&f, &of);
    if verbose() {
        // BIGBITS_TUNE_VERBOSE: every BZ row, measured and fitted. knuth / knuth_eq is how fast the
        // machine ran when the row was timed.
        let start = DivEq { bz: [1.0, 1.0], ..DivEq::read() };
        let mut text = String::from("rem\tq\td\ttop\tmeasured\tfitted\tblocks_and_top\tknuth_top\ttop_leaves\tknuth\tknuth_eq\tstart_blocks\tstart_knuth_top\n");
        for (((r, mode, leaves), row), ns) in shapes.iter().zip(&x).zip(&y) {
            let fitted: f64 = row.iter().zip(&f.beta).map(|(a, b)| a * b).sum();
            let (blocks, top) = ((r.q - 1) / r.d, (r.q - 1) % r.d + 1);
            let (products, block_leaves) = start.bz_block_parts(r.d, &mul, kara);
            let whole = blocks as f64 * (products + block_leaves);
            let (s0, s1) = if *mode == "padded" { (whole + start.bz_padded_top_ns(top, r.d, &mul, kara), 0.0) } else { (whole, start.knuth_ns(top, r.d)) };
            text += &format!("{}\t{}\t{}\t{mode}\t{ns:.1}\t{fitted:.1}\t{:.1}\t{:.1}\t{leaves:.1}\t{:.1}\t{:.1}\t{s0:.1}\t{s1:.1}\n", r.rem as u8, r.q, r.d, row[0], row[1], r.ns[0], start.knuth_ns(r.q, r.d));
        }
        dump("bz_fit.tsv", &text);
    }
    push(
        &mut out,
        &BZ,
        &f,
        said(&f, true)
            + &at
            + &format!(
                " (BZ forced with its top block padded and Knuth, both entries, divisors past the leaf{}; a (blocks (P + L) + [padded] padded top by its quotient's significant pieces) + b [Knuth top] Knuth(top, d))",
                left(can, kept, "the entries' tables name")
            ),
    );
    (out, eq)
}

// A small-division table's rows: eight divisor widths at a time, over quotient widths. It ends
// with the first eight rows that are the equations' throughout.
struct SmallSpec {
    rem: bool,
    next: usize,
    done: bool,
    rows: Vec<Row>,
}

impl Spec for SmallSpec {
    fn equations(&self) -> Option<u32> {
        Some((DIV_SMALL_PICKS.len() - 1) as u32)
    }

    fn group(&mut self) -> Option<Vec<(u32, Vec<u32>)>> {
        // Divisor widths: every one to 64 limbs, then eight to the octave.
        let widths: Vec<u32> = (1..=64).chain(tables::octave_steps(64, 4096, 8.0).into_iter().skip(1)).collect();
        if self.done || self.next >= widths.len() {
            return None;
        }
        let leaf = crate::tuning::k::div::bz();
        // A row's quotient widths, where the dispatch has a choice: every one to 24 limbs, then
        // eight to the octave.
        let quotients = |d: u32| -> Vec<u32> {
            (2..=24).chain(tables::octave_steps(24, 4096, 8.0).into_iter().skip(1)).filter(|&q| dyn_div_has_choice(q as usize, d as usize, leaf)).collect()
        };
        let rows = widths[self.next..].iter().take(8).map(|&d| (d, quotients(d))).collect();
        self.next += 8;
        Some(rows)
    }

    fn probes(&self, key: u32, size: u32, pass: usize, probes: &mut Vec<Probe<'static>>) -> Vec<Plan> {
        let (d, q, rem) = (key as usize, size as usize, self.rem);
        let leaf = crate::tuning::k::div::bz();
        if rem {
            probes.push(copy_probe(q, d, pass));
        }
        let mut picks = Vec::new();
        for (i, alg) in DIV_ALGS.into_iter().enumerate() {
            debug_assert_eq!(DIV_SMALL_PICKS[i], Some(alg));
            // What can take the shape, as the dispatch asks.
            if !dyn_div_can(alg, q, d, leaf) {
                continue;
            }
            picks.push(i as u32);
            let dv = placed_divisor(d, pass);
            let (n0, mut n, mut quo) = (numerator(q, d, pass), numerator(q, d, pass), pad(q - 1, 7, pass));
            probes.push(Probe::new("div", move || {
                if rem {
                    n.copy_from_slice(&n0);
                }
                let r = division_preflight(&n, &dv, &mut quo).expect("a nonempty request");
                let top = if rem { div_rem_prepared_dyn(&mut n, &dv, &mut quo, r, alg) } else { div_prepared_dyn(&n, &dv, &mut quo, r, alg) };
                black_box(top);
            }));
        }
        let (eq, mul, kara) = (DivEq::read(), MulEq::read(), crate::tuning::k::mul::kara());
        probes.push(Probe::new("decide", move || {
            black_box(dyn_div_alg_at(black_box(q), black_box(d), rem, &eq, &mul, kara));
        }));
        vec![Plan { harness: rem, picks, decide: true, ..Plan::default() }]
    }

    fn settled(&mut self, rows: Vec<Row>) {
        let equations = self.equations();
        self.done = rows.iter().all(|r| r.all_equations(equations));
        self.rows.extend(rows);
    }
}

/// `div.dyn_small` (with `rem`, `div.dyn_rem_small`) and its evidence: at each (d, q) the
/// dispatch has a choice at, the algorithms that can take it forced, with the decision beside
/// them (`tables::measure`).
pub fn small_table(level: Level, rem: bool) -> (Vec<u32>, String) {
    let mut spec = SmallSpec { rem, next: 0, done: false, rows: Vec::new() };
    let eps = level.eps().max(0.03);
    let tally = tables::measure(level, &mut spec, eps);
    let equations = spec.equations();
    let t = tables::assemble(&spec.rows, |_, _| true);
    let others = tables::sizes_above(&spec.rows, equations, eps, |_, _| true);
    // The widest divisor with an index of its own.
    let widest = spec.rows.iter().filter(|r| !r.all_equations(equations)).map(|r| r.key).max().unwrap_or(0);
    let ev = format!(
        "measured (table, {}): {} change points ({others}), the table's own indices at divisors to {widest} limbs; {}",
        if rem { "with remainder" } else { "quotient only" },
        t.len() / 3,
        tally.text()
    );
    (if t.is_empty() { vec![1, 1, equations.unwrap()] } else { t }, ev)
}

/// `cargo tune division`: every forced algorithm and the entries on shapes no tune fitted, written
/// to `out`; for each entry the time lost against the best forced algorithm by the equations'
/// picks (on the shapes the tables leave them) and by the entry, on the half held out. With
/// `refit` the equations are fitted now to the other half.
pub fn division(level: Level, out: &std::path::Path, refit: bool) -> std::io::Result<()> {
    let (all, noise) = division_timings(level, true);
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut text = String::from("rem\tq\td\tknuth\tbz\tnr\tauto\tbz_padded\tbz_knuth_top\theld\n");
    for x in &all {
        text += &format!(
            "{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{}\n",
            x.rem as u8, x.q, x.d, x.ns[0], x.ns[1], x.ns[2], x.auto, x.bz_top[0], x.bz_top[1], x.held as u8
        );
    }
    std::fs::write(out, text)?;
    println!("dynamic division ({}), {} shapes written to {}; their two passes differ {:.1}%", level.name(), all.len(), out.display(), 100.0 * noise);
    let (mul, kara) = (MulEq::read(), crate::tuning::k::mul::kara());
    let (small, leaf) = ([crate::tuning::k::div::dyn_small(), crate::tuning::k::div::dyn_rem_small()], crate::tuning::k::div::bz());
    let eq = if refit {
        let fitted: Vec<DivTiming> = all.iter().filter(|x| !x.held).map(|x| DivTiming { held: false, ..*x }).collect();
        let (tuned, eq) = fit_equations(&fitted, noise, Rows::Read);
        // Each equation's evidence once, and BZ's constants.
        for x in tuned.iter().filter(|x| [KNUTH[0], NR[0], NR_REM[0]].contains(&x.key) || BZ.contains(&x.key)) {
            println!("  div.{} = {}   # {}", x.key, x.value.text(), x.evidence);
        }
        eq
    } else {
        DivEq::read()
    };
    let t: Vec<DivTiming> = all.into_iter().filter(|x| x.held).collect();
    for rem in [false, true] {
        let rows: Vec<&DivTiming> = t.iter().filter(|x| x.rem == rem).collect();
        let best = |x: &DivTiming| x.ns.iter().cloned().fold(f64::INFINITY, f64::min);
        let v: Vec<f64> = rows.iter().map(|x| x.auto / best(x) - 1.0).collect();
        let auto = (v.iter().sum::<f64>() / v.len() as f64, v.iter().cloned().fold(f64::MIN, f64::max));
        let wins: Vec<usize> = (0..3).map(|i| rows.iter().filter(|x| x.ns[i] == best(x)).count()).collect();
        println!("  {} ({} shapes; fastest: Knuth {}, BZ {}, NR {}):", if rem { "div_rem_dyn" } else { "div_dyn" }, rows.len(), wins[0], wins[1], wins[2]);
        // The equations' picks where they are read (not where the entry's table decides), with
        // what each loses.
        let mut worst: Vec<(f64, &DivTiming, DivAlg)> = rows
            .iter()
            .filter(|x| read_for(x.rem, x.q, x.d, small, leaf))
            .map(|x| {
                let pick = dyn_div_alg_at(x.q, x.d, x.rem, &eq, &mul, kara);
                (x.ns[alg_index(pick)] / best(x) - 1.0, *x, pick)
            })
            .collect();
        let eqs = (worst.iter().map(|w| w.0).sum::<f64>() / worst.len().max(1) as f64, worst.iter().map(|w| w.0).fold(0.0, f64::max));
        println!("    the fitted equations: mean {:.2}%, worst {:.1}%   (on the {} shapes the table leaves them)", 100.0 * eqs.0, 100.0 * eqs.1, worst.len());
        println!("    the entry itself:     mean {:.2}%, worst {:.1}%   (its dispatch included)", 100.0 * auto.0, 100.0 * auto.1);
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (loss, x, pick) in worst.iter().take(4).filter(|w| w.0 > 0.02) {
            println!(
                "      q {} d {}: picks {:?}, {:.0}% over; ns Knuth {:.0} BZ {:.0} NR {:.0}; predicted {:.0} / {:.0} / {:.0}",
                x.q, x.d, pick, 100.0 * loss, x.ns[0], x.ns[1], x.ns[2],
                eq.knuth_ns(x.q, x.d),
                eq.bz_plan(x.q, x.d, &mul, kara).unwrap_or(f64::NAN),
                if x.q >= 8 { eq.nr_ns(x.q, x.d, x.rem, &mul, kara) } else { f64::NAN },
            );
        }
    }
    // BZ's own top-block choice where both were timed: the equations' and static BZ's cutoff rule.
    let both: Vec<&DivTiming> = t.iter().filter(|x| x.bz_top.iter().all(|v| v.is_finite())).collect();
    let stat = |padded: &dyn Fn(&DivTiming) -> bool| {
        let v: Vec<f64> = both.iter().map(|x| x.bz_top[if padded(x) { 0 } else { 1 }] / x.bz_top[0].min(x.bz_top[1]) - 1.0).collect();
        (v.iter().sum::<f64>() / v.len().max(1) as f64, v.iter().cloned().fold(0.0, f64::max))
    };
    let top = |x: &DivTiming| (x.q - 1) % x.d + 1;
    let (new, old) = (stat(&|x| eq.bz_top_padded(x.d, top(x), &mul, kara)), stat(&|x| use_bz_for_top_block(x.d, top(x))));
    println!("  BZ's top block, padded or Knuth ({} shapes with both timed):", both.len());
    println!("    the fitted equations: mean {:.2}%, worst {:.1}%", 100.0 * new.0, 100.0 * new.1);
    println!("    the cutoff rule:      mean {:.2}%, worst {:.1}%", 100.0 * old.0, 100.0 * old.1);
    Ok(())
}

