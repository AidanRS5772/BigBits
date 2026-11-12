//! Section `div`'s one-input decisions, as sweeps through the prepared drivers with the algorithm
//! forced: the two reciprocal cutoffs (Knuth against Newton-Raphson, by precision) and BZ's leaf
//! (Knuth against one BZ level over Knuth leaves); and the quotient dispatchers' choice, which has
//! two inputs (q, d): the fitted runtime equations of Knuth, BZ and NR (`fit_equations`, from
//! `division_timings`' shapes scattered over the (q, d) plane) that `div_dyn` and `div_rem_dyn`
//! compare per call, the Knuth guards computed from them, and `cargo tune division`, which reports
//! the measured time the equations' picks lose on shapes no tune fitted them to.

use crate::tuning::resolve::Value;
use crate::tuning::tuner::design::{Design, UNSEEN};
use crate::tuning::tuner::fit::{analyze, fit_split, Fit};
use crate::tuning::tuner::measure::{pad, time_fit, verbose, Level, Probe};
use crate::tuning::tuner::out::{push_fit as push, Tuned};
use crate::tuning::tuner::sweep::{cutoff_sweep, geo, keep};
use crate::tuning::{install, prepare};
use crate::utils::div::{
    div_dyn, div_prepared_dyn, div_rem_dyn, div_rem_prepared_dyn, division_preflight, dyn_div_alg_at, rcp_prepared_dyn,
    use_bz_for_top_block, use_bz_for_top_block_at, DivAlg, DivEq, RcpAlg, BZ_TOP_FORCE,
};
use crate::utils::mul::MulEq;
use std::hint::black_box;
use std::sync::atomic::Ordering;

// A divisor of n limbs with its top bit set.
fn divisor(n: usize, pass: usize) -> Vec<u64> {
    let mut d = pad(n, n as u64 * 41 + 3, pass).to_vec();
    d[n - 1] |= 1 << 63;
    d
}

pub fn tune(level: Level) -> Vec<Tuned> {
    let saved = crate::tuning::snapshot();
    let mut out = Vec::new();

    // Reciprocals of R limbs from an R-limb divisor: Knuth while R <= knob.
    // Only the dynamic one: the static cutoff also depends on N (static NR zero-fills [u64; N]
    // buffers: swept at N = 192 it read 28, and rcp_static::<1024> at 30 limbs then ran 1.50x
    // slower), so it has two inputs and waits for fitted equations in (R, N).
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

    // BZ's leaf, at 2d / d: Knuth against one BZ level whose halves are Knuth leaves (the knob at
    // half of d inside the probe). Knuth while d <= knob. Each probe runs both entries, the
    // quotient alone and with the remainder, as the leaf is theirs in common. It reads 99 on the
    // M1 either way. The static entries share the knob as their Knuth-to-BZ switch and cross
    // later: div_rem_static<2048> at 200/100 runs 10% slower at 99 than at the default 140,
    // which the tune's gate lists on every run (static values are approximate).
    let coarse: Vec<usize> = geo(16.0, 8.0).take_while(|&n| n <= 600).collect();
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
    // The leaf is live for what follows: BZ's timings and its equation both read it.
    crate::tuning::restore(&saved);
    keep("div", &mut out, t);

    // The quotient dispatchers have two inputs (q, d), so they compare fitted runtime equations:
    // Knuth, BZ and NR forced over shapes scattered across (q, d), each fitted in its own
    // structure, and the Knuth guards computed from the fits.
    let (timings, noise) = division_timings(level, false);
    let (tuned, _) = fit_equations(&timings, Some(noise));
    out.extend(tuned);
    out.extend(top_block_scale(&timings, level.eps()));
    crate::tuning::restore(&saved);
    out
}

/// Static BZ's top-block rule (`use_bz_for_top_block`: the padded 2/1 block when the scale times
/// `bz_2_1_cost` is at most the Knuth top block's work) keeps one constant. It is swept on the
/// dynamic timings with the top block forced each way, which the equations' fit takes anyway:
/// the scale with the least measured time lost. Static values are approximate, so no static
/// division is timed for it. Called with `div.bz` live at its swept value; `tie` is the tie band.
fn top_block_scale(t: &[DivTiming], tie: f64) -> Option<Tuned> {
    let both: Vec<&DivTiming> = t.iter().filter(|x| x.bz_top.iter().all(|v| v.is_finite())).collect();
    if both.is_empty() {
        return None;
    }
    let leaf = crate::tuning::k::div::bz();
    let loss = |scale: f64| {
        let v: Vec<f64> = both
            .iter()
            .map(|x| x.bz_top[if use_bz_for_top_block_at(x.d, (x.q - 1) % x.d + 1, leaf, scale) { 0 } else { 1 }] / x.bz_top[0].min(x.bz_top[1]) - 1.0)
            .collect();
        (v.iter().sum::<f64>() / v.len() as f64, v.iter().cloned().fold(0.0, f64::max))
    };
    // 1/4 to 16, eight a doubling.
    let cands: Vec<f64> = (-16..=32).map(|i| 2f64.powf(i as f64 / 8.0)).collect();
    let best = cands.iter().cloned().min_by(|a, b| loss(*a).0.partial_cmp(&loss(*b).0).unwrap()).expect("candidates");
    let previous = crate::tuning::k::div::bz_top_padded_cost_scale();
    let (lb, lp) = (loss(best), loss(previous));
    let range = format!("sweep {:.2}-{:.0}, {} values over {} shapes of forced dynamic BZ", cands[0], cands[cands.len() - 1], cands.len(), both.len());
    let (value, evidence) = if lp.0 - lb.0 <= tie {
        (previous, format!("{range}: kept, measured time lost mean {:.2}%, worst {:.1}%, within the tie band of the best ({best:.3}: {:.2}%)", 100.0 * lp.0, 100.0 * lp.1, 100.0 * lb.0))
    } else {
        (best, format!("{range}: measured time lost mean {:.2}%, worst {:.1}%; the previous value {previous} lost {:.2}%, worst {:.1}%", 100.0 * lb.0, 100.0 * lb.1, 100.0 * lp.0, 100.0 * lp.1))
    };
    Some(Tuned { key: "bz_top_padded_cost_scale", value: Value::Float(value), evidence })
}

/// One shape's timings: quotient-only (`rem` false) or with remainder, the logical quotient width
/// q (the quotient body and its structural limb) and the divisor width d, whether a fit holds the
/// shape out, each forced algorithm's ns (infinite where it wasn't timed), BZ's with its top
/// block forced padded and forced Knuth (at divisors past the leaf), and the public entry's.
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

/// Times Knuth, BZ and NR forced through the prepared drivers over (q, d) scattered from 4 to
/// 4096 limbs each (`design::scatter`; numerators to 8191 limbs), for `div_dyn` and `div_rem_dyn`,
/// every other shape held out; returns the timings and their noise (`measure::time_fit`). Knuth is
/// skipped past 2^22 limb products, where it is far behind. For a tune (`report` false) only what
/// the fits read is timed. For `cargo tune division` (`report` true) the public entry and BZ under
/// its own top-block choice are timed as well, on shapes no tune takes (`design::UNSEEN`).
pub fn division_timings(level: Level, report: bool) -> (Vec<DivTiming>, f64) {
    let design = Design::of(if report { UNSEEN } else { 0 }, level.fit_shapes(200), (4.0, 4096.0), (4.0, 4096.0), |q, d| {
        Some((q.round() as usize, d.round() as usize))
    });
    let shapes: Vec<(bool, usize, usize, bool)> =
        [false, true].into_iter().flat_map(|rem| design.shapes.iter().zip(&design.held).map(move |(&(q, d), &held)| (rem, q, d, held))).collect();
    let knuth_ok = |q: usize, d: usize| q * d <= 1 << 22;
    let leaf = crate::tuning::k::div::bz();
    // A shape's probes, in order: each algorithm forced, the entry, and BZ with its top block
    // forced each way (1: padded, 2: Knuth) where the divisor is past the leaf.
    let runs = [(Some(DivAlg::Knuth), 0u8), (Some(DivAlg::BZ), 0), (Some(DivAlg::NR), 0), (None, 0), (Some(DivAlg::BZ), 1), (Some(DivAlg::BZ), 2)];
    let timed = |alg: Option<DivAlg>, force: u8, q: usize, d: usize| {
        let can = !(alg == Some(DivAlg::Knuth) && !knuth_ok(q, d)) && !(force != 0 && d <= leaf);
        // The fits read Knuth, NR and BZ's two forced top blocks: the entry and BZ under its own
        // choice are the report's.
        can && (report || force != 0 || matches!(alg, Some(DivAlg::Knuth | DivAlg::NR)))
    };
    let build = |pass: usize| -> Vec<Probe<'static>> {
        let mut probes = Vec::new();
        for &(rem, q, d, _) in &shapes {
            // n of (q - 1) + d limbs: a quotient body of q - 1 limbs and its structural limb.
            let n_len = q - 1 + d;
            let numerator = || pad(n_len, (q * 31 + d) as u64, pass);
            // With a remainder the numerator is overwritten, so each call restores it first.
            // That copy is the harness's work, not the division's: it is timed on its own (the
            // shape's first probe) and taken off the shape's timings. Quotient-only runs leave
            // the numerator alone and copy nothing. (Until 2026-11-10 every run copied it and
            // nothing was taken off: at a short quotient over a long divisor the copy was more
            // than half of NR's quotient-only time.)
            if rem {
                let (src, mut dst) = (numerator(), numerator());
                probes.push(Probe::new("copy", move || {
                    dst.copy_from_slice(&src);
                    black_box(&dst);
                }));
            }
            for (alg, force) in runs {
                if !timed(alg, force, q, d) {
                    continue;
                }
                // The divisor sits 13 limbs further into its allocation in the second pass than
                // the numerator does, so the two passes differ in where the one lies against the
                // other (Knuth's loads from the divisor can alias its stores to the numerator).
                let mut dv = pad(d, d as u64 * 41 + 3, 2 * pass);
                dv[d - 1] |= 1 << 63;
                let (n0, mut n, mut quo) = (numerator(), numerator(), pad(q - 1, 7, pass));
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
    // A copy's time is taken off its shape's timings, so one mistimed copy probe would bend
    // them all (a tune on 2026-11-10 fitted Knuth and NR with remainder with one row 47% and 59%
    // off, which a rerun didn't repeat). Copies run at one rate per limb, so each is held to
    // three times the median rate over all the shapes' copies (the rate rises 2.5x where the
    // buffers leave the L1 cache), plus 5 ns for the call.
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
            // The division alone: less the shape's numerator copy. A copy is a few percent of
            // the cheapest division with a remainder; the floor only keeps a mistimed pair finite.
            let bound = 3.0 * rate * (q - 1 + d) as f64 + 5.0;
            let copy = if rem { t[at].min(bound) } else { 0.0 };
            bounded += (rem && t[at] > bound) as usize;
            at += rem as usize;
            let (mut ns, mut bz_top, mut auto) = ([f64::INFINITY; 3], [f64::INFINITY; 2], f64::INFINITY);
            for (alg, force) in runs {
                if !timed(alg, force, q, d) {
                    continue;
                }
                let own = (t[at] - copy).max(0.25 * t[at]);
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

/// The widest divisor at which the equations pick Knuth for every quotient width (to 2^17 limbs:
/// every width to 512, then sixteen an octave), for `div.dyn_knuth_d` and `div.dyn_rem_knuth_d`:
/// computation on the fitted equations, timing nothing.
pub(crate) fn knuth_guard(eq: &DivEq, mul: &MulEq, rem: bool) -> usize {
    let kara = crate::tuning::k::mul::kara();
    let mut qs: Vec<usize> = (2..=512).collect();
    qs.extend(geo(512.0, 16.0).skip(1).take(128));
    let knuth_everywhere = |d: usize| {
        qs.iter().all(|&q| dyn_div_alg_at(q, d, rem, eq, mul, kara) == DivAlg::Knuth)
    };
    (1..4096).take_while(|&d| knuth_everywhere(d)).last().unwrap_or(0)
}

// The equations' pick for a shape, at explicit equations.
fn eq_pick(x: &DivTiming, eq: &DivEq, mul: &MulEq) -> DivAlg {
    dyn_div_alg_at(x.q, x.d, x.rem, eq, mul, crate::tuning::k::mul::kara())
}

fn alg_index(a: DivAlg) -> usize {
    DIV_ALGS.iter().position(|x| *x == a).expect("one of the three")
}

/// The four equations fitted to a set of timings (on relative error, each judged on the shapes
/// the set holds out: `fit_split`), as the knobs of section `div` and the equations themselves:
/// Knuth and BZ on both entries' timings (each does the same work with or without the remainder),
/// NR on each entry's own. NR is fitted where it can run (nr_split: q >= 8) and BZ from its two
/// forced top-block modes at divisors past the leaf. `noise` is the timings' own, for the evidence.
pub(crate) fn fit_equations(t: &[DivTiming], noise: Option<f64>) -> (Vec<Tuned>, DivEq) {
    let (mul, kara) = (MulEq::read(), crate::tuning::k::mul::kara());
    let mut eq = DivEq::read();
    let mut out = Vec::new();
    let m = |l: usize, s: usize| mul.mul_ns(l, s, kara);
    // Where a fit's largest residual is, for its evidence: the first place to look when one misses.
    fn worst_at(shapes: &[(usize, usize, bool)], x: &[Vec<f64>], y: &[f64], beta: &[f64]) -> String {
        let rel = |i: usize| (x[i].iter().zip(beta).map(|(a, b)| a * b).sum::<f64>() / y[i] - 1.0).abs();
        match (0..y.len()).max_by(|&a, &b| rel(a).partial_cmp(&rel(b)).unwrap()) {
            Some(i) => format!("; worst at q {} d {}{}", shapes[i].0, shapes[i].1, if shapes[i].2 { " with remainder" } else { "" }),
            None => String::new(),
        }
    }
    let shape = |r: &&DivTiming| (r.q, r.d, r.rem);
    // One equation's fit over `rows`: judged on the rows held out, and its form's analysis when
    // the tune was asked for it.
    let fit = |name: &str, keys: &[&'static str], x: &[Vec<f64>], y: &[f64], held: Vec<bool>| -> Fit {
        let fixed = vec![0.0; y.len()];
        analyze(name, keys, x, y, &fixed, &held);
        Fit { noise, ..fit_split(x, y, &fixed, &held) }
    };
    let held_of = |rows: &[&DivTiming]| -> Vec<bool> { rows.iter().map(|r| r.held).collect() };

    let rows: Vec<&DivTiming> = t.iter().filter(|x| x.ns[0].is_finite()).collect();
    let (x, y): (Vec<Vec<f64>>, Vec<f64>) = rows.iter().map(|x| (vec![(x.q * x.d) as f64, x.q as f64, x.d as f64, 1.0], x.ns[0])).unzip();
    const KNUTH: [&str; 4] = ["knuth_ns_qd", "knuth_ns_q", "knuth_ns_d", "knuth_ns_shift"];
    let f = fit("div Knuth (forced, both entries): a q d + b q + c d + e", &KNUTH, &x, &y, held_of(&rows));
    eq.knuth.copy_from_slice(&f.beta);
    let at = worst_at(&rows.iter().map(shape).collect::<Vec<_>>(), &x, &y, &f.beta);
    push(&mut out, &KNUTH, "knuth_ns_err", &f, f.evidence() + &at + " (Knuth forced, both entries; a q d + b q + c d + e)");

    // NR: its own products by tier (DivEq::nr_parts, which reads only the two tier boundaries),
    // one multiple a tier. A tier no shape of the grid reaches has no evidence, so its multiple
    // stays the one in force instead of fitting to zero.
    let start = eq;
    let tiers = |q: usize| start.nr_parts(q, &mul, kara);
    let keep_unseen = |x: &[Vec<f64>], beta: &mut [f64], before: &[f64]| {
        for i in 0..4 {
            if x.iter().all(|row| row[i] == 0.0) {
                beta[i] = before[i];
            }
        }
    };
    let rows: Vec<&DivTiming> = t.iter().filter(|x| !x.rem && x.q >= 8).collect();
    let x: Vec<Vec<f64>> = rows.iter().map(|r| tiers(r.q).into_iter().chain([r.q as f64]).collect()).collect();
    let y: Vec<f64> = rows.iter().map(|r| r.ns[2]).collect();
    const NR: [&str; 5] = ["nr_ns_mid_school", "nr_ns_mid_fft", "nr_ns_hi_school", "nr_ns_hi_guarded", "nr_ns_q"];
    let mut f = fit("div NR (forced, quotient only, q >= 8): its products by tier + b q", &NR, &x, &y, held_of(&rows));
    keep_unseen(&x, &mut f.beta, &start.nr);
    eq.nr.copy_from_slice(&f.beta);
    let at = worst_at(&rows.iter().map(shape).collect::<Vec<_>>(), &x, &y, &f.beta);
    push(
        &mut out,
        &NR,
        "nr_ns_err",
        &f,
        f.evidence_revised() + &at + " (NR forced, quotient only, q >= 8; its middle and high products by tier, each a multiple of mul's price at its own shape: school middle, FFT middle, school high, guarded high, + b q)",
    );

    let rows: Vec<&DivTiming> = t.iter().filter(|x| x.rem && x.q >= 8).collect();
    let x: Vec<Vec<f64>> =
        rows.iter().map(|r| tiers(r.q).into_iter().chain([m(r.d, (r.q - 1).min(r.d + 1)), r.q as f64, r.d as f64]).collect()).collect();
    let y: Vec<f64> = rows.iter().map(|r| r.ns[2]).collect();
    const NR_REM: [&str; 7] = [
        "nr_rem_ns_mid_school",
        "nr_rem_ns_mid_fft",
        "nr_rem_ns_hi_school",
        "nr_rem_ns_hi_guarded",
        "nr_rem_ns_prod",
        "nr_rem_ns_q",
        "nr_rem_ns_d",
    ];
    let mut f = fit("div NR with remainder (forced, q >= 8): the four tiers + r Mul(d, min(q - 1, d + 1)) + b q + c d", &NR_REM, &x, &y, held_of(&rows));
    keep_unseen(&x, &mut f.beta, &start.nr_rem);
    eq.nr_rem.copy_from_slice(&f.beta);
    let at = worst_at(&rows.iter().map(shape).collect::<Vec<_>>(), &x, &y, &f.beta);
    push(
        &mut out,
        &NR_REM,
        "nr_rem_ns_err",
        &f,
        f.evidence_revised() + &at + " (NR forced, with remainder, q >= 8; the same four tiers + r Mul(d, min(q - 1, d + 1)) + b q + c d)",
    );

    // BZ on Knuth's equation just fitted, from its timings with the top block forced each way,
    // so the fit doesn't depend on the rule it feeds: a (blocks (P + L) + [padded] padded top) +
    // b [Knuth top] Knuth(top, d), with (P, L) one 2/1 block's products and leaves and the
    // padded top priced by its quotient's significant pieces (bz_padded_top_ns). A Knuth top over
    // no whole block is Knuth alone and is left out.
    let unit = DivEq { bz: [1.0, 1.0], ..eq };
    let (mut x, mut y, mut shapes) = (Vec::new(), Vec::new(), Vec::new());
    for r in t.iter().filter(|r| r.d > eq.bz_leaf) {
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
    const BZ: [&str; 2] = ["bz_ns_block", "bz_ns_top"];
    let f = fit(
        "div BZ (forced, top block padded and Knuth, both entries): a (blocks + padded top) + b Knuth top",
        &BZ,
        &x,
        &y,
        shapes.iter().map(|(r, _, _)| r.held).collect(),
    );
    eq.bz.copy_from_slice(&f.beta);
    let at = worst_at(&shapes.iter().map(|(r, _, _)| (r.q, r.d, r.rem)).collect::<Vec<_>>(), &x, &y, &f.beta);
    // BIGBITS_TUNE_VERBOSE: every BZ row's measured and fitted time, for reading the residuals.
    if verbose() {
        // knuth and knuth_eq are the same shape's Knuth run (timed in the same rounds) and its
        // prediction under the equations the build started with: their ratio is how fast the
        // machine was running when the row was timed. start_blocks and start_knuth_top are the
        // row's two regressors under those starting equations.
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
        let _ = std::fs::write("target/tuning/bz_fit.tsv", text);
    }
    push(
        &mut out,
        &BZ,
        "bz_ns_err",
        &f,
        f.evidence_revised() + &at + " (BZ forced with its top block padded and Knuth, both entries, divisors past the leaf; a (blocks (P + L) + [padded] padded top by its quotient's significant pieces) + b [Knuth top] Knuth(top, d))",
    );
    out.extend(guards_of(&eq, &mul));
    (out, eq)
}

fn guards_of(eq: &DivEq, mul: &MulEq) -> Vec<Tuned> {
    [("dyn_knuth_d", false), ("dyn_rem_knuth_d", true)]
        .into_iter()
        .map(|(key, rem)| Tuned {
            key,
            value: Value::Int(knuth_guard(eq, mul, rem) as u64),
            evidence: format!("derived: the widest divisor at which the fitted equations pick Knuth for every quotient to 2^17 limbs ({})", if rem { "with remainder" } else { "quotient only" }),
        })
        .collect()
}

/// `div.dyn_knuth_d` and `div.dyn_rem_knuth_d` under the equations in force, for `cargo tune
/// derive` and the end of a tune: they follow from div's and mul's equations and time nothing.
pub fn guards() -> Vec<Tuned> {
    guards_of(&DivEq::read(), &MulEq::read())
}

// `division`'s timings as its tab-separated file.
fn timings_tsv(t: &[DivTiming]) -> String {
    let mut text = String::from("rem\tq\td\tknuth\tbz\tnr\tauto\tbz_padded\tbz_knuth_top\theld\n");
    for x in t {
        text += &format!(
            "{}\t{}\t{}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{}\n",
            x.rem as u8, x.q, x.d, x.ns[0], x.ns[1], x.ns[2], x.auto, x.bz_top[0], x.bz_top[1], x.held as u8
        );
    }
    text
}

/// `cargo tune division`: every forced algorithm and the entries timed on shapes no tune fitted
/// the equations to, written to `out` (tab-separated), and for each entry the measured time lost
/// against the best forced algorithm by the equations' picks and by the entry itself, on the half
/// of those shapes held out. The equations are the live knobs', or with `refit` ones fitted now
/// to the other half. (Until 2026-11-11 it also reported the cutoff rule the equations replaced:
/// 0.8% mean and 37% worst for div_dyn, 3.4% and 59% for div_rem_dyn, on the M1.)
pub fn division(level: Level, out: &std::path::Path, refit: bool) -> std::io::Result<()> {
    let (all, noise) = division_timings(level, true);
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, timings_tsv(&all))?;
    println!("dynamic division ({}), {} shapes written to {}; their two passes differ {:.1}%", level.name(), all.len(), out.display(), 100.0 * noise);
    let mul = MulEq::read();
    let eq = if refit {
        let fitted: Vec<DivTiming> = all.iter().filter(|x| !x.held).map(|x| DivTiming { held: false, ..*x }).collect();
        let (tuned, eq) = fit_equations(&fitted, Some(noise));
        for x in tuned.iter().filter(|x| x.key.ends_with("_err") || x.key.ends_with("knuth_d") || x.key.starts_with("bz_ns_")) {
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
        let stat = |f: &dyn Fn(&DivTiming) -> f64| {
            let v: Vec<f64> = rows.iter().map(|x| f(x) / best(x) - 1.0).collect();
            (v.iter().sum::<f64>() / v.len() as f64, v.iter().cloned().fold(f64::MIN, f64::max))
        };
        let eqs = stat(&|x| x.ns[alg_index(eq_pick(x, &eq, &mul))]);
        let auto = stat(&|x| x.auto);
        let wins: Vec<usize> = (0..3).map(|i| rows.iter().filter(|x| x.ns[i] == best(x)).count()).collect();
        println!("  {} ({} shapes; fastest: Knuth {}, BZ {}, NR {}):", if rem { "div_rem_dyn" } else { "div_dyn" }, rows.len(), wins[0], wins[1], wins[2]);
        println!("    the fitted equations: mean {:.2}%, worst {:.1}%", 100.0 * eqs.0, 100.0 * eqs.1);
        println!("    the entry itself:     mean {:.2}%, worst {:.1}%   (its dispatch included)", 100.0 * auto.0, 100.0 * auto.1);
        // Where the equations' pick costs most.
        let mut worst: Vec<(f64, &DivTiming)> = rows.iter().map(|x| (x.ns[alg_index(eq_pick(x, &eq, &mul))] / best(x) - 1.0, *x)).collect();
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (loss, x) in worst.iter().take(4).filter(|w| w.0 > 0.02) {
            let pick = eq_pick(x, &eq, &mul);
            let kara = crate::tuning::k::mul::kara();
            println!(
                "      q {} d {}: picks {:?}, {:.0}% over; ns Knuth {:.0} BZ {:.0} NR {:.0}; predicted {:.0} / {:.0} / {:.0}",
                x.q, x.d, pick, 100.0 * loss, x.ns[0], x.ns[1], x.ns[2],
                eq.knuth_ns(x.q, x.d),
                eq.bz_plan(x.q, x.d, &mul, kara).unwrap_or(f64::NAN),
                if x.q >= 8 { eq.nr_ns(x.q, x.d, x.rem, &mul, kara) } else { f64::NAN },
            );
        }
    }
    // BZ's own choice for its top block, padded or Knuth, on the shapes where both were timed:
    // the equations' choice (what bz_div_core_dyn does) and the cutoff rule it replaced there
    // (use_bz_for_top_block, which static BZ keeps), against the faster of the two.
    let kara = crate::tuning::k::mul::kara();
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

