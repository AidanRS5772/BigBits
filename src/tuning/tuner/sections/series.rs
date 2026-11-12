//! Section `series`: the dynamic nodes' dispatch among Horner, rectangular splitting and binary
//! splitting compares fitted runtime equations, one an algorithm (`SeriesEq` over each
//! algorithm's `Parts`: products priced by mul's equations, finalize's division by division's,
//! and a price for each kind of linear pass, call, term and allocation), behind two guards on
//! the shape's ratios. `timings` measures the dynamic entries with each algorithm forced;
//! `tune` fits the equations to them, each judged on the workloads held out, and sweeps the
//! guards; `report` (`cargo tune series`) gives the measured time the picks lose, under the live
//! equations and under ones fitted now, on workloads no tune fitted them to.
//!
//! **The workloads are no caller's** (the user's direction, 2026-11-11: the evaluators are
//! general, and their fitting stands on its own). They are the synthetic family of
//! `tuner/synth.rs`, each member through its typed context, scattered over the dials that
//! decide an evaluation's work: the frame (4 to 1024 limbs), the terms (2 to a term a bit, with
//! the shift set for them), and for a Hyper series with an x, x's width (2 bits to the whole
//! frame); the members set the coefficients' size and whether they converge on their own. Two
//! ratios of a shape say which algorithms are in the running: rho, x's bits over the bits a term
//! fixes, and beta, the exact fraction's width over the frame (`Shape::rho`, `Shape::beta`). An
//! algorithm is timed, and so fitted, where it competes and some way past (`RECT_FROM`,
//! `HORNER_TO`, `BIN_SPLIT_TO`), not where it is hopeless.
//!
//! Each forced time has the entry's own work taken off (the context, the term count from the
//! tail bound, the eligibility pass: a probe of its own, as division's numerator copy is), since
//! that part is the same whichever algorithm runs and reaches half the time of a small call.

use crate::tuning::Value;
use crate::tuning::tuner::derive::SMALL_EXTENT_NS;
use crate::tuning::tuner::design::{Design, UNSEEN};
use crate::tuning::tuner::fit::{analyze, fit_split, rel_rms, Fit};
use crate::tuning::tuner::measure::{time_fit, verbose, Level, Probe};
use crate::tuning::tuner::out::Tuned;
use crate::tuning::tuner::sweep::{keep, param_sweep, with_knob};
use crate::tuning::tuner::synth::{x_of, Big, Den, Fact, Geo, Odd, Synth};
use crate::utils::series::{
    bin_split_tree_at, series_eval_dyn, series_eval_dyn_at, BBPSeries, Bbp, Costs, Engel, EngelSeries, Guarded, Hyper, HyperSeries, Parts,
    Series, SeriesAlg, SeriesEnv, SeriesEq, SeriesNode, SplitCutoffs,
};
use std::hint::black_box;
use std::rc::Rc;

pub const ALGS: [SeriesAlg; 3] = [SeriesAlg::Horner, SeriesAlg::Rect, SeriesAlg::BinSplit];

/// Which side of the small-series table's extent a workload is on: whether its best algorithm
/// is predicted past `SMALL_EXTENT_NS` by the short prices in force, which is the test the
/// table's derivation makes. Each has its own fit (the user's go-ahead, 2026-11-11): the
/// dispatch reads the equations for the long series alone, and one fit over every size read
/// 5-6% on them, pulled by the short ones that no form predicts within 9%; the short series'
/// fit feeds the table's derivation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    Long,
    Short,
}

impl Regime {
    // The live prices of the regime.
    fn eq(self) -> SeriesEq {
        match self {
            Regime::Long => SeriesEq::read(),
            Regime::Short => SeriesEq::read_short(),
        }
    }

    // The family's count is taken this many times over for the regime to keep about that many
    // shapes: a quarter of the family's shapes are long.
    fn walk(self) -> usize {
        match self {
            Regime::Long => 3,
            Regime::Short => 1,
        }
    }

    fn what(self) -> &'static str {
        match self {
            Regime::Long => "past the small-series table's extent",
            Regime::Short => "under the small-series table's extent",
        }
    }
}
// The parallel gains tried for a tree that splits across the pool: 0, 0.05, ..., 1.
const GAINS: usize = 21;

// Where each algorithm is timed, by the shape's ratios: its own area and a margin past it, so an
// equation is good up to where a guard or a tenfold loss takes over. Rect from a rho of 1/4
// (below it its powers save nothing: it reads 1.1-1.6x Horner). Horner to a rho of 64 where x
// has 16 limbs or more and rect can run (past it rect is 3-4x faster; a narrower x leaves Horner
// in the running at any rho). Binary splitting to a beta of 24 where another can run (it loses
// 2-3x at 3 to 9 and 5-8x to 64).
const RECT_FROM: f64 = 0.25;
const HORNER_TO: (f64, usize) = (64.0, 16);
const BIN_SPLIT_TO: f64 = 24.0;
// A forced algorithm predicted past this many ns, or 50 times the best, isn't timed.
const HOPELESS_NS: f64 = 1e8;

/// One workload's measurements: each forced algorithm's ns (infinite where it can't run or was
/// left out of its area), the dispatcher's own, each algorithm's parts, and what settles the
/// range ahead of the equations (`horner_first`, the guards).
pub struct SeriesTiming {
    pub name: String,
    /// Whether a fit holds the workload out.
    pub held: bool,
    pub ns: [f64; 3],
    pub auto: f64,
    /// Binary splitting's tree alone, without its finalize (the division by Q).
    pub tree: f64,
    /// The entry's own work, common to every algorithm and inside each forced time: the context,
    /// the term count from the tail bound, and the eligibility pass.
    pub entry: f64,
    /// Each algorithm's parts under the live knobs, rect's at the block length that was timed.
    pub parts: [Option<Parts>; 3],
    /// Binary splitting's parts at each parallel gain of `GAINS`.
    pub gained: Vec<Parts>,
    pub horner_first: bool,
    /// What the small-series table says to run under the live knobs: read ahead of everything.
    pub small: Option<SeriesAlg>,
    /// Whether each algorithm is in its area here: a fit reads only those. (A tune times only
    /// those too; `report` times every one that isn't hopeless, to judge a pick wherever it
    /// lands.)
    pub area: [bool; 3],
    /// The guards' verdict under the live knobs, and the two ratios they read.
    pub guard: Guarded,
    pub rho: f64,
    pub beta: f64,
}

impl SeriesTiming {
    // Algorithm i's own time, where it was timed.
    fn own(&self, i: usize) -> Option<f64> {
        Some(self.ns[i] - self.entry).filter(|v| v.is_finite() && *v > 0.0)
    }

    // The fastest forced algorithm's time.
    fn best(&self) -> f64 {
        self.ns.iter().cloned().fold(f64::INFINITY, f64::min)
    }
}

/// What walks the family: called once a workload, with its typed context, the precision asked
/// of it in limbs and whether a fit holds it out.
trait Visit {
    fn series<C: Series + 'static>(&mut self, name: String, ctx: C, w: usize, held: bool);
}

const FRAMES: (f64, f64) = (4.0, 1024.0);

// The most terms a shape of w limbs takes: a term a bit, and a bound on what a probe costs (a
// frame's limbs times its terms).
fn most_terms(w: usize) -> u64 {
    (64 * w as u64).min((1 << 19) / w as u64).max(2)
}

// The point a fraction of the way from lo to hi in log scale; the fraction is log2 of y, which
// a scatter draws over 1 to 2.
fn along(lo: f64, hi: f64, y: f64) -> u64 {
    (lo * (hi / lo).powf(y.log2())).round() as u64
}

// A member's Hyper series with an x of its own, at n points of (frame, terms, x's bits).
fn hyper_x<S: HyperSeries + Synth + 'static>(from: usize, n: usize, v: &mut impl Visit) {
    let d = Design::of3(from, n, FRAMES, (1.0, 2.0), (1.0, 2.0), |w, y, z| {
        let w = w.round() as usize;
        Some((w, along(2.0, most_terms(w) as f64, y), along(2.0, 64.0 * w as f64, z)))
    })
    .in_pairs();
    for (&(w, t, xb), &held) in d.shapes.iter().zip(&d.held) {
        // Leaked: the context borrows x for the probes' lifetime.
        let x: &'static [u64] = Box::leak(x_of(xb, 31 * xb + w as u64).into_boxed_slice());
        let ctx = Hyper::<S>::new(x, xb + S::decay(64 * w as u64, t)).expect("x 2^-shift <= 1/2");
        v.series(format!("hyper {} w{w} t{t} x{xb}", S::name()), ctx, w, held);
    }
}

// The (frame, terms) points of a series without an x.
fn frames_terms(from: usize, n: usize) -> Design<(usize, u64)> {
    Design::of(from, n, FRAMES, (1.0, 2.0), |w, y| {
        let w = w.round() as usize;
        Some((w, along(2.0, most_terms(w) as f64, y)))
    })
    .in_pairs()
}

// A member's Hyper series at x = 1, its Engel series and its BBP series.
fn hyper_one<S: HyperSeries + Synth + 'static>(from: usize, n: usize, v: &mut impl Visit) {
    let d = frames_terms(from, n);
    for (&(w, t), &held) in d.shapes.iter().zip(&d.held) {
        v.series(format!("hyper {} w{w} t{t} x0", S::name()), Hyper::<S>::new(&[1], S::decay(64 * w as u64, t)).expect("a shift of 1 or more"), w, held);
    }
}

fn engel<S: EngelSeries + Synth + 'static>(from: usize, n: usize, v: &mut impl Visit) {
    let d = frames_terms(from, n);
    for (&(w, t), &held) in d.shapes.iter().zip(&d.held) {
        v.series(format!("engel {} w{w} t{t}", S::name()), Engel::<S>::new(S::decay(64 * w as u64, t)), w, held);
    }
}

fn bbp<S: BBPSeries + Synth + 'static>(from: usize, n: usize, v: &mut impl Visit) {
    let d = frames_terms(from, n);
    for (&(w, t), &held) in d.shapes.iter().zip(&d.held) {
        v.series(format!("bbp {} w{w} t{t}", S::name()), Bbp::<S>::new(S::decay(64 * w as u64, t)), w, held);
    }
}

/// The family: every member's shapes, `n(quick)` of each set given its count at the quick
/// level, each set from its own stretch of its sequence past `from`. 80 shapes at the quick
/// level: 44 Hyper with an x (coefficients of 8 and 56 bits that add no convergence, 32-bit
/// denominators and a factorial's), 12 Hyper at x = 1, 12 Engel and 12 BBP.
fn family(from: usize, n: &dyn Fn(usize) -> usize, v: &mut impl Visit) {
    // Sets apart in their sequences, so no two share a point.
    let at = |set: usize| from + 20_000 * set;
    hyper_x::<Geo<8>>(at(0), n(12), v);
    hyper_x::<Geo<56>>(at(1), n(12), v);
    hyper_x::<Den<32>>(at(2), n(8), v);
    hyper_x::<Fact>(at(3), n(12), v);
    hyper_one::<Geo<8>>(at(4), n(4), v);
    hyper_one::<Den<32>>(at(5), n(4), v);
    hyper_one::<Fact>(at(6), n(4), v);
    engel::<Fact>(at(7), n(6), v);
    engel::<Den<32>>(at(8), n(6), v);
    bbp::<Odd>(at(9), n(6), v);
    bbp::<Big<40>>(at(10), n(6), v);
}

// The fit's walk: each workload's parts, and a probe for each algorithm in its area, for the
// dispatcher's own pick and binary splitting's tree alone (`extras`, for `report`: no fit reads
// them), and for the entry's own work.
struct Timer<'p> {
    regime: Regime,
    extras: bool,
    probes: &'p mut Vec<Probe<'static>>,
    rows: Vec<SeriesTiming>,
}

impl Visit for Timer<'_> {
    // The context's series to w limbs (below its zero limbs), as a caller of the typed entry
    // gets it. Nothing when the frame leaves it no limbs.
    fn series<C: Series + 'static>(&mut self, name: String, ctx: C, w: usize, held: bool) {
        let zeros = ctx.zeros();
        let Some(len) = w.checked_sub(zeros).filter(|&l| l > 0) else { return };
        let (a, b) = ctx.range(64 * (len + zeros) as u64);
        let node = ctx.dyn_ctx();
        let ok = <C::Dyn as SeriesNode>::eligible(&node, a, b, len, zeros);
        let shape = <C::Dyn as SeriesNode>::shape(&node, a, b, len, zeros);
        let least = |c: &Costs| ALGS.iter().filter_map(|&alg| c.of(alg)).fold(f64::INFINITY, f64::min);
        if (least(&shape.predictions_at(ok, &SeriesEq::read_short())) > SMALL_EXTENT_NS) != (self.regime == Regime::Long) {
            return;
        }
        let costs: Costs = shape.predictions_at(ok, &self.regime.eq());
        let mut env = SeriesEnv::read();
        let parts = <C::Dyn as SeriesNode>::parts(&node, a, b, len, zeros, None, &env);
        let gained = (0..GAINS)
            .map(|i| {
                env.par_gain = i as f64 / (GAINS - 1) as f64;
                <C::Dyn as SeriesNode>::parts(&node, a, b, len, zeros, Some(1), &env)[2].expect("bin_split always runs")
            })
            .collect();
        let (rho, beta) = (shape.rho(), shape.beta());
        let others = ok.horner || ok.rect;
        let in_area = |alg: SeriesAlg| match alg {
            SeriesAlg::Horner => !(ok.rect && rho > HORNER_TO.0 && shape.x_limbs() >= HORNER_TO.1),
            SeriesAlg::Rect => rho >= RECT_FROM,
            SeriesAlg::BinSplit => !(others && beta > BIN_SPLIT_TO),
        };
        let area = ALGS.map(|alg| costs.of(alg).is_some() && in_area(alg));
        let best = least(&costs);
        let ctx = Rc::new(ctx);
        let mut ns = [f64::INFINITY; 3];
        for (i, alg) in ALGS.into_iter().enumerate() {
            let Some(pred) = costs.of(alg) else { continue };
            if !(area[i] || self.extras) || pred > HOPELESS_NS || pred > 50.0 * best {
                continue;
            }
            ns[i] = 0.0; // timed: filled in from the probes' order
            let (ctx, mut out) = (ctx.clone(), vec![0u64; len]);
            self.probes.push(Probe::new("forced", move || {
                black_box(series_eval_dyn_at(&*ctx, &mut out, alg));
            }));
        }
        if self.extras {
            let mut out = vec![0u64; len];
            let auto_ctx = ctx.clone();
            self.probes.push(Probe::new("auto", move || {
                black_box(series_eval_dyn(&*auto_ctx, &mut out));
            }));
            let tree_ctx = ctx.clone();
            self.probes.push(Probe::new("tree", move || {
                black_box(bin_split_tree_at::<C::Dyn>(&tree_ctx.dyn_ctx(), a, b, true, SplitCutoffs::of::<C::Dyn>()));
            }));
        }
        self.probes.push(Probe::new("entry", move || {
            let node = ctx.dyn_ctx();
            let zeros = ctx.zeros();
            let (a, b) = ctx.range(64 * (len + zeros) as u64);
            black_box(<C::Dyn as SeriesNode>::eligible(&node, a, b, len, zeros));
        }));
        self.rows.push(SeriesTiming {
            name,
            held,
            ns,
            auto: f64::INFINITY,
            tree: f64::INFINITY,
            entry: 0.0,
            parts,
            gained,
            horner_first: <C::Dyn as SeriesNode>::horner_first(&node, a, b, len, zeros),
            small: <C::Dyn as SeriesNode>::small(&node, a, b, len, zeros).map(|s| s.0),
            area,
            guard: <C::Dyn as SeriesNode>::guards(&node, a, b, len, zeros),
            rho,
            beta,
        });
    }
}

/// The family's shapes of one regime, from a walk of `Regime::walk` times the level's count
/// (each set's points held out two by two), each
/// algorithm forced in its area, by `time_fit`; returns the timings and their noise. For
/// `report` the dispatcher's pick and the tree alone are timed as well, on points no tune takes
/// (`design::UNSEEN`); with `cache`, the times are read from that file when it holds this set
/// (the noise is then unknown), and written to it otherwise.
pub fn timings(level: Level, report: bool, cache: Option<&std::path::Path>, regime: Regime) -> (Vec<SeriesTiming>, Option<f64>) {
    let build = |probes: &mut Vec<Probe<'static>>| -> Vec<SeriesTiming> {
        let mut timer = Timer { regime, extras: report, probes, rows: Vec::new() };
        family(if report { UNSEEN } else { 0 }, &|quick| regime.walk() * level.fit_shapes(quick), &mut timer);
        timer.rows
    };
    let mut rows = build(&mut Vec::new());
    if let Some(path) = cache {
        if load(path, &mut rows) {
            return (rows, None);
        }
    }
    let timed = time_fit(level, |_| {
        let mut probes = Vec::new();
        build(&mut probes);
        probes
    });
    let mut at = 0;
    for r in &mut rows {
        // The probes of a workload, in the order the walk pushed them.
        for v in r.ns.iter_mut().filter(|v| v.is_finite()) {
            *v = timed.ns[at];
            at += 1;
        }
        if report {
            (r.auto, r.tree) = (timed.ns[at], timed.ns[at + 1]);
            at += 2;
        }
        r.entry = timed.ns[at];
        at += 1;
    }
    if let Some(path) = cache {
        save(path, &rows);
    }
    (rows, Some(timed.noise))
}

// `--timings FILE`: the report's timings kept between runs, one line a workload (the name, the
// three forced times, the dispatcher's, the tree's, the entry's own work; `inf` where a forced
// algorithm wasn't timed), so a change to the equations can be read against the same
// measurements. The dispatcher's time is the dispatch of the build that measured it.
fn load(path: &std::path::Path, rows: &mut [SeriesTiming]) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else { return false };
    let lines: std::collections::HashMap<&str, Vec<f64>> = text
        .lines()
        .filter_map(|l| {
            let mut f = l.split('\t');
            Some((f.next()?, f.filter_map(|v| v.parse().ok()).collect()))
        })
        .collect();
    if !rows.iter().all(|r| lines.get(r.name.as_str()).is_some_and(|v| v.len() == 6)) {
        return false;
    }
    for r in rows {
        let v = &lines[r.name.as_str()];
        (r.ns, r.auto, r.tree, r.entry) = ([v[0], v[1], v[2]], v[3], v[4], v[5]);
    }
    true
}

fn save(path: &std::path::Path, rows: &[SeriesTiming]) {
    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) else { return };
    for r in rows {
        let _ = writeln!(f, "{}\t{}\t{}\t{}\t{}\t{}\t{}", r.name, r.ns[0], r.ns[1], r.ns[2], r.auto, r.tree, r.entry);
    }
}

// Algorithm i's (`alg`'s) prices over the columns `cols`, fitted to its own times in its area
// and judged on the workloads held out; the other columns keep `base`'s prices, whose part of
// each time is taken as given. A tree that splits across the pool isn't fitted: its gain is
// found afterwards.
fn fit_cols(t: &[SeriesTiming], alg: &str, i: usize, cols: &[usize], base: [f64; 12], noise: Option<f64>) -> ([f64; 12], Fit, String) {
    let (mut x, mut y, mut fixed, mut held, mut names) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in t {
        if let (Some(p), Some(own)) = (r.parts[i].filter(|p| r.area[i] && p.wake == 0.0), r.own(i)) {
            let f = p.features();
            x.push(cols.iter().map(|&c| f[c]).collect::<Vec<f64>>());
            y.push(own);
            fixed.push((0..12).filter(|c| !cols.contains(c)).map(|c| base[c] * f[c]).sum::<f64>());
            held.push(r.held);
            names.push(r.name.as_str());
        }
    }
    let keys: Vec<&str> = cols.iter().map(|&c| Parts::NAMES[c]).collect();
    analyze(&format!("series {alg} (forced, the entry's own work taken off): a price for each kind of work"), &keys, &x, &y, &fixed, &held);
    let fit = Fit { noise, ..fit_split(&x, &y, &fixed, &held) };
    let mut prices = base;
    for (&c, &v) in cols.iter().zip(&fit.beta) {
        prices[c] = v;
    }
    // The workload the fit misses most, for its evidence.
    let miss = |k: usize| (x[k].iter().zip(&fit.beta).map(|(a, b)| a * b).sum::<f64>() + fixed[k]) / y[k] - 1.0;
    let worst = (0..y.len()).max_by(|&a, &b| miss(a).abs().partial_cmp(&miss(b).abs()).unwrap()).map_or(String::new(), |k| format!("; worst at {} ({:+.0}%, {:.0} ns)", names[k], 100.0 * miss(k), y[k]));
    (prices, fit, worst)
}

/// The equations fitted to a set of timings.
pub struct Fitted {
    pub eq: SeriesEq,
    /// What a parallel split gains, and the trees that split across the pool: how many, and the
    /// relative rms of their predictions under it.
    pub gain: f64,
    pub par: (usize, f64),
    /// Horner's, rect's and binary splitting's fits, and the workload each misses most.
    pub fits: [Fit; 3],
    pub worst: [String; 3],
}

// The relative rms of the predictions of the trees that split across the pool, at gain g of
// `GAINS`, under binary splitting's prices; and how many they are.
fn par_rms(t: &[SeriesTiming], g: usize, prices: &[f64; 12], wake: f64) -> (usize, f64) {
    let e: Vec<f64> = t.iter().filter(|r| r.gained[0].wake != 0.0).filter_map(|r| r.own(2).map(|own| r.gained[g].ns_with(prices, wake) / own - 1.0)).collect();
    (e.len(), rel_rms(&e))
}

/// Fits the three equations to `t`: binary splitting on its serial trees, then the gain under
/// which those prices predict the trees that split across the pool best (kept in each timing's
/// parts); Horner; and rect on Horner's prices for the linear kernels it shares, with its own
/// product multiple, call price and fixed cost.
pub fn fit_equations(t: &mut [SeriesTiming], noise: Option<f64>) -> Fitted {
    let live = SeriesEq::read();
    let (bin_split, fb, wb) = fit_cols(t, "bin_split", 2, &SeriesEq::BIN_SPLIT, [0.0; 12], noise);
    let gi = (0..GAINS).min_by(|&a, &b| par_rms(t, a, &bin_split, live.wake).1.partial_cmp(&par_rms(t, b, &bin_split, live.wake).1).unwrap()).unwrap();
    let par = par_rms(t, gi, &bin_split, live.wake);
    for r in t.iter_mut() {
        r.parts[2] = Some(r.gained[gi]);
    }
    let (horner, fh, wh) = fit_cols(t, "horner", 0, &SeriesEq::HORNER, [0.0; 12], noise);
    let mut base = horner;
    base[8] = 0.0;
    let (rect, fr, wr) = fit_cols(t, "rect", 1, &SeriesEq::RECT, base, noise);
    Fitted { eq: SeriesEq { horner, rect, bin_split, wake: live.wake }, gain: gi as f64 / (GAINS - 1) as f64, par, fits: [fh, fr, fb], worst: [wh, wr, wb] }
}

// Each algorithm's predicted ns under `eq`.
fn predict(r: &SeriesTiming, eq: &SeriesEq) -> [Option<f64>; 3] {
    let prices = [&eq.horner, &eq.rect, &eq.bin_split];
    [0, 1, 2].map(|i| r.parts[i].map(|p| p.ns_with(prices[i], eq.wake)))
}

// The dispatch's pick under `eq`, the live table and the live guards: the small-series table's
// where it has one, Horner where horner_first says so, rect where its guard does, else the
// least prediction among the algorithms that can run and the beta guard leaves (bin_split on a
// tie, as Costs::pick). (Rect's time here is at the model's block length; its guard runs the
// balance, which the entry's own time has. The table is the live one whatever `eq` is.)
fn pick(r: &SeriesTiming, eq: &SeriesEq) -> usize {
    if let Some(alg) = r.small {
        return ALGS.iter().position(|&a| a == alg).expect("one of the three");
    }
    if r.horner_first {
        return 0;
    }
    if r.guard.rect_first {
        return 1;
    }
    let ns = predict(r, eq);
    let mut best = (2, if r.guard.no_bin_split { f64::INFINITY } else { ns[2].expect("bin_split always runs") });
    for i in [1, 0] {
        if let Some(v) = ns[i].filter(|&v| v <= best.1) {
            best = (i, v);
        }
    }
    best.0
}

// (mean, worst, of the total) time lost by picks against each workload's fastest timed
// algorithm: the mean and worst of the workloads' own relative losses, and the loss of the
// set's summed time, which the large workloads carry. A pick that wasn't timed (skipped as
// hopeless) counts as 50 times the best, the skip's own bound.
fn lost(t: &[SeriesTiming], pick: impl Fn(&SeriesTiming) -> f64) -> (f64, f64, f64) {
    let got: Vec<f64> = t.iter().map(|r| Some(pick(r)).filter(|g| g.is_finite()).unwrap_or(50.0 * r.best())).collect();
    let v: Vec<f64> = t.iter().zip(&got).map(|(r, g)| g / r.best() - 1.0).collect();
    let total = got.iter().sum::<f64>() / t.iter().map(|r| r.best()).sum::<f64>() - 1.0;
    (v.iter().sum::<f64>() / v.len() as f64, v.iter().cloned().fold(f64::MIN, f64::max), total)
}

// The mean of the entry's own relative loss over the workloads whose fastest algorithm takes
// from `lo` to `hi` ns, and how many they are: where the dispatch's own cost shows.
fn entry_lost(t: &[SeriesTiming], lo: f64, hi: f64) -> (usize, f64) {
    let v: Vec<f64> = t.iter().filter(|r| (lo..hi).contains(&r.best())).map(|r| r.auto / r.best() - 1.0).collect();
    (v.len(), v.iter().sum::<f64>() / v.len().max(1) as f64)
}

// The equations' relative rms against each algorithm's own times.
fn rms(t: &[SeriesTiming], eq: &SeriesEq) -> [f64; 3] {
    [0, 1, 2].map(|i| {
        let e: Vec<f64> = t.iter().filter_map(|r| Some(predict(r, eq)[i]? / r.own(i)? - 1.0)).collect();
        rel_rms(&e)
    })
}

// The knob of algorithm `alg`'s price for column c of `Parts`: `series.<alg>_ns_<column>` for
// the long series, `series.short_<alg>_ns_<column>` for the short.
fn price_key(regime: Regime, alg: &str, c: usize) -> &'static str {
    let short = if regime == Regime::Short { "short_" } else { "" };
    crate::tuning::knob(&format!("series.{short}{alg}_ns_{}", Parts::NAMES[c])).expect("every fitted column has its knob").name
}

// A guard's sweep times the dispatcher's own entry on shapes the guard can change: the family
// again, on points of its sequences neither a tune's fit nor a report takes.
const GUARD_FROM: usize = UNSEEN / 2;
// The value that turns a guard off: no shape's ratio reaches it.
const OFF: f64 = 1e9;

// What a guard's sweep asks of a shape.
struct Seen {
    rect: bool,
    others: bool,
    rho: f64,
    beta: f64,
}

type Load = Rc<dyn Fn() -> Probe<'static>>;

// A guard's loads run from 50 us to 20 ms by their best algorithm's prediction. The floor is
// what keeps a guard to its purpose, which is to settle ranges where the algorithms are far
// apart: under it the dispatch's own cost (0.1-0.2 us) is a share of the call that a sweep can
// see, and the first sweep without it took rect from a rho of 2 at any x to save that much on
// calls under a microsecond, overriding equations that were right on long ones (a 100 us
// series 35% slower, Horner's by its prediction and by the clock). What the dispatch costs a
// short series is a question for the model's own cost, not for a guard on a ratio.
const GUARD_LOADS_NS: (f64, f64) = (5e4, 2e7);

// The walk that collects a sweep's loads: the typed entry on every shape `want` takes, bar
// those horner_first settles and those outside `GUARD_LOADS_NS`.
struct Loads<'f> {
    want: &'f dyn Fn(&Seen) -> bool,
    found: Vec<Load>,
}

impl Visit for Loads<'_> {
    fn series<C: Series + 'static>(&mut self, _: String, ctx: C, w: usize, _: bool) {
        let zeros = ctx.zeros();
        let Some(len) = w.checked_sub(zeros).filter(|&l| l > 0) else { return };
        let (a, b) = ctx.range(64 * (len + zeros) as u64);
        let node = ctx.dyn_ctx();
        let ok = <C::Dyn as SeriesNode>::eligible(&node, a, b, len, zeros);
        let shape = <C::Dyn as SeriesNode>::shape(&node, a, b, len, zeros);
        let costs: Costs = <C::Dyn as SeriesNode>::costs(&node, a, b, len, zeros, None);
        let best = ALGS.iter().filter_map(|&alg| costs.of(alg)).fold(f64::INFINITY, f64::min);
        let seen = Seen { rect: ok.rect, others: ok.horner || ok.rect, rho: shape.rho(), beta: shape.beta() };
        if <C::Dyn as SeriesNode>::horner_first(&node, a, b, len, zeros) || !(GUARD_LOADS_NS.0..=GUARD_LOADS_NS.1).contains(&best) || !(self.want)(&seen) {
            return;
        }
        let ctx = Rc::new(ctx);
        self.found.push(Rc::new(move || {
            let (ctx, mut out) = (ctx.clone(), vec![0u64; len]);
            Probe::new("auto", move || {
                black_box(series_eval_dyn(&*ctx, &mut out));
            })
        }));
    }
}

// Up to `most` of the family's shapes that `want` takes, spread evenly through them.
fn loads(want: &dyn Fn(&Seen) -> bool, most: usize) -> Vec<Load> {
    let mut l = Loads { want, found: Vec::new() };
    family(GUARD_FROM, &|quick| 8 * quick, &mut l);
    let (n, k) = (l.found.len(), most.min(l.found.len()));
    (0..k).map(|i| l.found[i * n / k].clone()).collect()
}

// One guard's knob as a sweep over `loads`: the dispatcher's entry under each candidate, which
// charges a candidate the picks it gets wrong.
fn sweep_guard(level: Level, path: &'static str, cands: &[Value], loads: &[Load], out: &mut Vec<Tuned>) {
    if loads.is_empty() {
        return;
    }
    let all: Vec<usize> = (0..loads.len()).collect();
    keep("series", out, param_sweep(level, path, &all, cands, &|w, c, _| with_knob(path, c, loads[w]())));
}

/// The guards, each a sweep under the equations in force (`tune` has just made its fit live):
/// the beta past which binary splitting isn't predicted, on shapes of beta 1.5 to 64 that
/// another algorithm can sum; then rect's guard, its rho and its narrowest x, on shapes rect can
/// sum at a rho of 1 or more.
fn sweep_guards(level: Level, out: &mut Vec<Tuned>) {
    let floats = |v: &[f64]| -> Vec<Value> { v.iter().map(|&x| Value::Float(x)).collect() };
    let wide = loads(&|s| s.others && (1.5..=64.0).contains(&s.beta), 20);
    sweep_guard(level, "series.bin_split_beta", &floats(&[1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0, OFF]), &wide, out);
    let by_x = loads(&|s| s.rect && s.rho >= 1.0, 20);
    sweep_guard(level, "series.rect_rho", &floats(&[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, OFF]), &by_x, out);
    let limbs: Vec<Value> = [2u64, 4, 8, 16, 32, 64].iter().map(|&x| Value::Int(x)).collect();
    sweep_guard(level, "series.rect_x_limbs", &limbs, &by_x, out);
}

/// The series section: for the long series and for the short ones, the three equations fitted
/// to the synthetic family's shapes of that regime (rect on Horner's prices for the kernels they
/// share, but for the one-limb divisions, which Horner's long series hardly make); the parallel
/// gain, from the long trees; and the guards swept under the long equations. The equations
/// replaced the shared prices over counted work on 2026-11-10 (that model's revision), so a fit
/// above the line reads NOISY and its constants are written.
pub fn tune(level: Level) -> Vec<Tuned> {
    let saved = crate::tuning::snapshot();
    let mut out = Vec::new();
    for regime in [Regime::Long, Regime::Short] {
        let (mut t, noise) = timings(level, false, None, regime);
        let f = fit_equations(&mut t, noise);
        let sets: [(&str, &[usize], &[f64; 12], &Fit, &str, &str); 3] = [
            ("horner", &SeriesEq::HORNER, &f.eq.horner, &f.fits[0], "Horner forced", &f.worst[0]),
            ("rect", &SeriesEq::RECT, &f.eq.rect, &f.fits[1], "rect forced, on Horner's prices for the other linear kernels", &f.worst[1]),
            ("bin_split", &SeriesEq::BIN_SPLIT, &f.eq.bin_split, &f.fits[2], "binary splitting forced, serial trees", &f.worst[2]),
        ];
        for (alg, cols, prices, fit, what, worst) in sets {
            let keys: Vec<&str> = cols.iter().map(|&c| price_key(regime, alg, c)).collect();
            let evidence = format!("{} ({what}, where it competes; synthetic series {}, the entry's own work taken off{worst}){}", fit.evidence_revised(), regime.what(), fit.dropped_text(&keys));
            for &c in cols {
                keep("series", &mut out, Tuned { key: price_key(regime, alg, c), value: Value::Float(prices[c]), evidence: evidence.clone() });
            }
        }
        let evidence = format!("the gain under which the serial trees' prices predict the {} trees that split across the pool best: relative rms {:.1}%", f.par.0, 100.0 * f.par.1);
        // With no such tree in the set there is nothing to read the gain from: it stays.
        if regime == Regime::Long && f.par.0 != 0 {
            keep("series", &mut out, Tuned { key: "bin_split_par_gain", value: Value::Float(f.gain), evidence });
        }
    }
    sweep_guards(level, &mut out);
    crate::tuning::restore(&saved);
    out
}

// One set's lines: the picks' loss and each equation's rms under `eq`, and the picks that cost
// most.
fn show(label: &str, t: &[SeriesTiming], eq: &SeriesEq) {
    let (l, e) = (lost(t, |r| r.ns[pick(r, eq)]), rms(t, eq));
    println!(
        "    {label:<16} picks lose mean {:.2}%, worst {:.1}%, {:.2}% of the total time; rms Horner {:.1}%, rect {:.1}%, bin_split {:.1}%",
        100.0 * l.0, 100.0 * l.1, 100.0 * l.2, 100.0 * e[0], 100.0 * e[1], 100.0 * e[2]
    );
    let mut worst: Vec<(f64, &SeriesTiming)> = t.iter().map(|r| (r.ns[pick(r, eq)] / r.best() - 1.0, r)).collect();
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (loss, r) in worst.iter().take(4).filter(|w| w.0 > 0.05) {
        let pred: Vec<String> = predict(r, eq).iter().map(|p| p.map_or("-".into(), |v| format!("{v:.0}"))).collect();
        println!(
            "      {}: picks {:?}{}, {:.0}% over; ns H {:.0} R {:.0} B {:.0} (entry {:.0}); predicted {} / {} / {}",
            r.name, ALGS[pick(r, eq)], if r.horner_first { " (horner_first)" } else { "" }, 100.0 * loss, r.ns[0], r.ns[1], r.ns[2], r.entry, pred[0], pred[1], pred[2]
        );
    }
}

/// `cargo tune series`: on synthetic workloads no tune fitted the equations to, for the long
/// series and then the short ones: the equations fitted now on the half not held out, and on
/// each half the measured time lost by the dispatch's picks under the live equations and under
/// the fitted ones, each equation's error, and the entry's own loss. (The live table decides
/// the short series whatever the equations: their "fitted now" lines say how the short prices
/// fit, which is what the table is derived from.)
pub fn report(level: Level, cache: Option<&std::path::Path>) {
    println!("the series equations ({}), on the synthetic family", level.name());
    let mut tsv: Vec<String> = Vec::new();
    for regime in [Regime::Long, Regime::Short] {
        report_regime(level, cache, regime, &mut tsv);
    }
    if verbose() {
        let _ = std::fs::create_dir_all("target/tuning");
        let head = format!("regime\tset\tname\talg\tns\ttree\tpred\thf\t{}\twake\trho\tbeta\tarea", Parts::NAMES.join("\t"));
        let _ = std::fs::write("target/tuning/series_parts.tsv", format!("{head}\n{}\n", tsv.join("\n")));
    }
}

fn report_regime(level: Level, cache: Option<&std::path::Path>, regime: Regime, tsv: &mut Vec<String>) {
    let (all, noise) = timings(level, true, cache, regime);
    let (mut held, mut fitted): (Vec<SeriesTiming>, Vec<SeriesTiming>) = all.into_iter().partition(|r| r.held);
    let live = regime.eq();
    println!("series {}: {} workloads fitted, {} held out", regime.what(), fitted.len(), held.len());
    let settled = |f: &dyn Fn(&SeriesTiming) -> bool| fitted.iter().chain(held.iter()).filter(|r| f(r)).count();
    println!(
        "  ahead of the equations: the small-series table takes {}, horner_first {}, the rect guard {}, and the beta guard leaves binary splitting out of {}",
        settled(&|r| r.small.is_some()),
        settled(&|r| r.small.is_none() && r.horner_first),
        settled(&|r| r.small.is_none() && !r.horner_first && r.guard.rect_first),
        settled(&|r| r.small.is_none() && !r.horner_first && !r.guard.rect_first && r.guard.no_bin_split)
    );
    let live_parts: Vec<Option<Parts>> = fitted.iter().chain(held.iter()).map(|r| r.parts[2]).collect();
    let f = fit_equations(&mut fitted, noise);
    // The held-out trees at the fitted gain too.
    let gi = (f.gain * (GAINS - 1) as f64).round() as usize;
    for r in held.iter_mut() {
        r.parts[2] = Some(r.gained[gi]);
    }
    let named = |cols: &[usize], prices: &[f64; 12]| -> String {
        cols.iter().map(|&c| format!("{} {:.4}", Parts::NAMES[c], prices[c])).collect::<Vec<_>>().join(", ")
    };
    for (i, (cols, prices)) in [(&SeriesEq::HORNER[..], &f.eq.horner), (&SeriesEq::RECT[..], &f.eq.rect), (&SeriesEq::BIN_SPLIT[..], &f.eq.bin_split)].into_iter().enumerate() {
        println!("  fitted now, {:?}: {}\n    {}", ALGS[i], f.fits[i].evidence_revised(), named(cols, prices));
    }
    let held_par = par_rms(&held, gi, &f.eq.bin_split, live.wake);
    if f.par.0 + held_par.0 != 0 {
        println!(
            "  trees that split across the pool: gain {:.2} a thread; rms {:.1}% over {} fitted, {:.1}% over {} held out",
            f.gain, 100.0 * f.par.1, f.par.0, 100.0 * held_par.1, held_par.0
        );
    }
    if verbose() {
        for (set, t) in [("fit", &fitted), ("held", &held)] {
            for r in t.iter() {
                let pred = predict(r, &f.eq);
                for i in 0..3 {
                    if let (Some(p), Some(own)) = (r.parts[i], r.own(i)) {
                        let cols: Vec<String> = p.features().iter().map(|v| format!("{v}")).collect();
                        tsv.push(format!("{regime:?}\t{set}\t{}\t{:?}\t{own}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", r.name, ALGS[i], r.tree, pred[i].unwrap_or(f64::NAN), r.horner_first as u8, cols.join("\t"), p.wake, r.rho, r.beta, r.area[i] as u8));
                    }
                }
            }
        }
    }
    for (name, t) in [("fitted set", &fitted), ("held-out set", &held)] {
        let wins: Vec<usize> = (0..3).map(|i| t.iter().filter(|r| r.ns[i] == r.best()).count()).collect();
        println!("  {name} ({} workloads; fastest: Horner {}, rect {}, bin_split {}):", t.len(), wins[0], wins[1], wins[2]);
        show("fitted now", t, &f.eq);
    }
    // Under the live equations the trees take the live gain's parts again.
    for (r, p) in fitted.iter_mut().chain(held.iter_mut()).zip(live_parts) {
        r.parts[2] = p;
    }
    for (name, t) in [("fitted set", &fitted), ("held-out set", &held)] {
        println!("  {name}:");
        show("live equations", t, &live);
        let l = lost(t, |r| r.auto);
        println!("    the entry itself: mean {:.2}%, worst {:.1}%, {:.2}% of the total time   (its dispatch included)", 100.0 * l.0, 100.0 * l.1, 100.0 * l.2);
        let bands: Vec<String> = [(0.0, 1e3, "under 1 us"), (1e3, 1e4, "1-10 us"), (1e4, 1e5, "10-100 us"), (1e5, f64::INFINITY, "past 100 us")]
            .iter()
            .filter(|b| entry_lost(t, b.0, b.1).0 != 0)
            .map(|&(lo, hi, what)| {
                let (n, mean) = entry_lost(t, lo, hi);
                format!("{what} {:.1}% ({n})", 100.0 * mean)
            })
            .collect();
        println!("      by the fastest algorithm's time: {}", bands.join(", "));
        let mut worst: Vec<&SeriesTiming> = t.iter().collect();
        worst.sort_by(|a, b| (b.auto / b.best()).partial_cmp(&(a.auto / a.best())).unwrap());
        for r in worst.iter().take(2).filter(|r| r.auto / r.best() > 1.1) {
            let ran = r.small.map_or("the equations".to_string(), |alg| format!("the table: {alg:?}"));
            println!("      {}: the entry {:.0} ns against {:.0} ({ran}); ns H {:.0} R {:.0} B {:.0}", r.name, r.auto, r.best(), r.ns[0], r.ns[1], r.ns[2]);
        }
    }
}
