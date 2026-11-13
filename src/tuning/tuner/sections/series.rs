//! Section `series`: the dynamic nodes' dispatch among Horner, rect and binary splitting.
//! `timings` forces each algorithm on the synthetic family (`tuner/synth.rs`: no caller's series)
//! and times the decision; `tune` fits the three equations (`SeriesEq` over each algorithm's
//! `Parts`) to the series whose decision is within the share of the call; `small_table` measures
//! the table under that fit; `tune_late` fits again on what the table leaves (`Rows`) and sweeps
//! rect's guard; `report` is `cargo tune series`. Each forced time has the entry's own work
//! taken off (context, term count, eligibility: the same whichever algorithm runs).

use crate::tuning::Value;
use crate::tuning::tuner::design::{Design, UNSEEN};
use crate::tuning::tuner::fit::{fit_named, fit_rel, rel_rms, Fit};
use crate::tuning::tuner::measure::{dump, pad, time_fit, verbose, Level, Probe};
use crate::tuning::tuner::out::Tuned;
use crate::tuning::tuner::sweep::{ints, keep, param_sweep, with_knob};
use crate::tuning::tuner::synth::{x_static, Big, Den, Fact, Geo, Odd, Synth};
use crate::tuning::tuner::tables::{self, Plan, Row, Spec};
use crate::utils::series::{
    bin_split_tree_at, series_eval_at, series_eval_dyn, series_eval_dyn_at, small_frame_span, BBPSeries, Bbp, Costs, Engel, EngelSeries, Guarded, Hyper,
    HyperSeries, Parts, Series, SeriesAlg, SeriesEnv, SeriesEq, SeriesGuards, SeriesNode, Shape, SplitCutoffs, SMALL_KINDS, SMALL_PICKS, SMALL_ROWS,
};
use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;

pub const ALGS: [SeriesAlg; 3] = [SeriesAlg::Horner, SeriesAlg::Rect, SeriesAlg::BinSplit];
// Each one's equation: its knobs' name and the columns its fit prices.
const EQS: [(&str, &[usize]); 3] = [("horner", &SeriesEq::HORNER), ("rect", &SeriesEq::RECT), ("bin_split", &SeriesEq::BIN_SPLIT)];

// An equation set's prices, in `ALGS`' order.
fn prices(eq: &SeriesEq) -> [&[f64; Parts::N]; 3] {
    [&eq.horner, &eq.rect, &eq.bin_split]
}

// The column of `Parts` a kind of work is counted in.
fn col(name: &str) -> usize {
    Parts::NAMES.iter().position(|n| *n == name).expect("a column of Parts")
}

// The least of a range's predictions, over the algorithms that can sum it.
fn least(c: &Costs) -> f64 {
    ALGS.iter().filter_map(|&alg| c.of(alg)).fold(f64::INFINITY, f64::min)
}

/// The frame a context's series is summed in at `w` limbs, as the typed entry's caller gets it:
/// (zero limbs, output limbs below them, range of terms). None when no limb is left.
pub(crate) fn framed<C: Series>(ctx: &C, w: usize) -> Option<(usize, usize, (u64, u64))> {
    let zeros = ctx.zeros();
    let len = w.checked_sub(zeros).filter(|&l| l > 0)?;
    Some((zeros, len, ctx.range(64 * (len + zeros) as u64)))
}

// The family is walked this many times over: a fit takes only the part its equations are read for.
const WALK: usize = 3;
// The parallel gains tried for a tree split across the pool: 0, 0.05, ..., 1.
const GAINS: usize = 21;

// Where each algorithm is timed, and so fitted: rect from a rho of 1/4 (below it its powers save
// nothing), Horner to a rho of 64 where x has 16 limbs or more and rect can run (rect's guard
// takes those), binary splitting at every beta.
const RECT_FROM: f64 = 0.25;
const HORNER_TO: (f64, usize) = (64.0, 16);
// A forced algorithm predicted past this many ns, or 50 times the best, isn't timed.
const HOPELESS_NS: f64 = 1e8;

/// One workload's measurements: each forced algorithm's ns (infinite where not timed), the
/// dispatcher's, the decision's, each algorithm's parts, and what settles the range ahead of them.
#[derive(Clone)]
pub struct SeriesTiming {
    pub name: String,
    /// Held out of a fit.
    pub held: bool,
    pub ns: [f64; 3],
    pub auto: f64,
    /// Binary splitting's tree without its finalize.
    pub tree: f64,
    /// The entry's own work, inside every forced time.
    pub entry: f64,
    /// The decision at explicit knobs: shape, guards, equations.
    pub decide: f64,
    /// Each algorithm's parts under the live knobs (rect's at the block length timed).
    pub parts: [Option<Parts>; 3],
    /// Binary splitting's parts at each gain of `GAINS`.
    pub gained: Vec<Parts>,
    /// The small-series table's pick under the live knobs.
    pub small: Option<SeriesAlg>,
    /// Whether each algorithm is in its area: a fit reads only those.
    pub area: [bool; 3],
    /// The guard's verdict under the live knobs, and the shape's two ratios.
    pub guard: Guarded,
    pub rho: f64,
    pub beta: f64,
    /// `Shape::dials` and the table's class, for the dump.
    pub dials: [f64; 13],
    pub class: Option<usize>,
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

    /// Whether the decision is within `share` of the call (else the workload is the table's).
    pub fn read(&self, share: f64) -> bool {
        self.decide <= share * self.best()
    }

    /// Whether a fit takes the workload.
    pub fn fitted(&self, rows: Rows) -> bool {
        match rows {
            Rows::Share(share) => self.read(share),
            Rows::Table => self.small.is_none(),
        }
    }
}

/// Which workloads a fit takes: the calls its equations are read for.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Rows {
    /// Those whose own decision is within this share of the call (a first fit; `report`).
    Share(f64),
    /// Those the table in force names no algorithm for. Not `Share`: a row stands for a kind of
    /// series, so it takes some within the share and hands on some past it.
    Table,
}

/// Walks the family: a workload's typed context, its precision in limbs, whether a fit holds it out.
trait Visit {
    fn series<C: Series + 'static>(&mut self, name: String, ctx: C, w: usize, held: bool);
}

const FRAMES: (f64, f64) = (4.0, 1024.0);

// The most terms at w limbs: a term a bit, and a bound on a probe's cost.
fn most_terms(w: usize) -> u64 {
    (64 * w as u64).min((1 << 19) / w as u64).max(2)
}

// A point between lo and hi in log scale, from a scatter's y in [1, 2).
fn along(lo: f64, hi: f64, y: f64) -> u64 {
    (lo * (hi / lo).powf(y.log2())).round() as u64
}

// A member's Hyper series with an x, at n points of (frame, terms, x's bits).
fn hyper_x<S: HyperSeries + Synth + 'static>(from: usize, n: usize, v: &mut impl Visit) {
    let d = Design::of3(from, n, FRAMES, (1.0, 2.0), (1.0, 2.0), |w, y, z| {
        let w = w.round() as usize;
        Some((w, along(2.0, most_terms(w) as f64, y), along(2.0, 64.0 * w as f64, z)))
    })
    .in_pairs();
    for (&(w, t, xb), &held) in d.shapes.iter().zip(&d.held) {
        let ctx = Hyper::<S>::new(x_static(xb, 31 * xb + w as u64), xb + S::decay(64 * w as u64, t)).expect("x 2^-shift <= 1/2");
        v.series(format!("hyper {} w{w} t{t} x{xb}", S::name()), ctx, w, held);
    }
}

// A member's series without an x of its own (`make` gives its context from the shift), at n
// points of (frame, terms).
fn plain<S: Synth, C: Series + 'static>(kind: &str, tail: &str, make: impl Fn(u64) -> C, from: usize, n: usize, v: &mut impl Visit) {
    let d = Design::of(from, n, FRAMES, (1.0, 2.0), |w, y| {
        let w = w.round() as usize;
        Some((w, along(2.0, most_terms(w) as f64, y)))
    })
    .in_pairs();
    for (&(w, t), &held) in d.shapes.iter().zip(&d.held) {
        v.series(format!("{kind} {} w{w} t{t}{tail}", S::name()), make(S::decay(64 * w as u64, t)), w, held);
    }
}

/// The family, `n(quick)` shapes of each set (80 at the quick level), each set from its own
/// stretch of its sequence past `from`.
fn family(from: usize, n: &dyn Fn(usize) -> usize, v: &mut impl Visit) {
    // Sets apart in their sequences: no two share a point.
    let at = |set: usize| from + 20_000 * set;
    hyper_x::<Geo<8>>(at(0), n(12), v);
    hyper_x::<Geo<56>>(at(1), n(12), v);
    hyper_x::<Den<32>>(at(2), n(8), v);
    hyper_x::<Fact>(at(3), n(12), v);
    let one = "a shift of 1 or more";
    plain::<Geo<8>, _>("hyper", " x0", |s| Hyper::<Geo<8>>::new(&[1], s).expect(one), at(4), n(4), v);
    plain::<Den<32>, _>("hyper", " x0", |s| Hyper::<Den<32>>::new(&[1], s).expect(one), at(5), n(4), v);
    plain::<Fact, _>("hyper", " x0", |s| Hyper::<Fact>::new(&[1], s).expect(one), at(6), n(4), v);
    plain::<Fact, _>("engel", "", Engel::<Fact>::new, at(7), n(6), v);
    plain::<Den<32>, _>("engel", "", Engel::<Den<32>>::new, at(8), n(6), v);
    plain::<Odd, _>("bbp", "", Bbp::<Odd>::new, at(9), n(6), v);
    plain::<Big<40>, _>("bbp", "", Bbp::<Big<40>>::new, at(10), n(6), v);
}

// The fit's walk: each workload's parts and probes (each algorithm in its area, then with
// `extras` the dispatcher's pick and the tree alone, then the entry's own work and the decision).
struct Timer<'p> {
    extras: bool,
    probes: &'p mut Vec<Probe<'static>>,
    rows: Vec<SeriesTiming>,
}

impl Visit for Timer<'_> {
    fn series<C: Series + 'static>(&mut self, name: String, ctx: C, w: usize, held: bool) {
        let Some((zeros, len, (a, b))) = framed(&ctx, w) else { return };
        let node = ctx.dyn_ctx();
        let ok = <C::Dyn as SeriesNode>::eligible(&node, a, b, len, zeros);
        let shape = <C::Dyn as SeriesNode>::shape(&node, a, b, len, zeros);
        let costs: Costs = shape.predictions_at(ok, &SeriesEq::read());
        let mut env = SeriesEnv::read();
        let parts = <C::Dyn as SeriesNode>::parts(&node, a, b, len, zeros, None, &env);
        let gained = (0..GAINS)
            .map(|i| {
                env.par_gain = i as f64 / (GAINS - 1) as f64;
                <C::Dyn as SeriesNode>::parts(&node, a, b, len, zeros, Some(1), &env)[2].expect("bin_split always runs")
            })
            .collect();
        let (rho, beta) = (shape.rho(), shape.beta());
        let in_area = |alg: SeriesAlg| match alg {
            SeriesAlg::Horner => !(ok.rect && rho > HORNER_TO.0 && shape.x_limbs() >= HORNER_TO.1),
            SeriesAlg::Rect => rho >= RECT_FROM,
            SeriesAlg::BinSplit => true,
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
        let entry_ctx = ctx.clone();
        self.probes.push(Probe::new("entry", move || {
            let node = entry_ctx.dyn_ctx();
            let zeros = entry_ctx.zeros();
            let (a, b) = entry_ctx.range(64 * (len + zeros) as u64);
            black_box(<C::Dyn as SeriesNode>::eligible(&node, a, b, len, zeros));
        }));
        self.probes.push(decide_probe::<C>(Rc::new(ctx.dyn_ctx()), (a, b), (len, zeros)));
        self.rows.push(SeriesTiming {
            name,
            held,
            ns,
            auto: f64::INFINITY,
            tree: f64::INFINITY,
            entry: 0.0,
            decide: 0.0,
            parts,
            gained,
            small: <C::Dyn as SeriesNode>::small(&node, a, b, len, zeros).map(|s| s.0),
            area,
            guard: <C::Dyn as SeriesNode>::guards(&node, a, b, len, zeros),
            rho,
            beta,
            dials: shape.dials(),
            class: <C::Dyn as SeriesNode>::small_class(&node, a, b),
        });
    }
}

// A probe of the decision for a range: shape, guards and equations, knobs read beforehand.
pub(crate) fn decide_probe<C: Series + 'static>(node: Rc<<C::Dyn as crate::utils::series::Node>::Ctx>, (a, b): (u64, u64), (len, zeros): (usize, usize)) -> Probe<'static> {
    let ok = <C::Dyn as SeriesNode>::eligible(&node, a, b, len, zeros);
    let (env, eq, guards) = (SeriesEnv::read(), SeriesEq::read(), SeriesGuards::read());
    Probe::new("decide", move || {
        let shape = <C::Dyn as SeriesNode>::shape(black_box(&node), a, b, len, zeros);
        let g = shape.guards(ok, &guards);
        black_box(shape.decide_with(ok, g, &env, &eq));
    })
}

/// The family's timings and their noise. `report` also times the dispatcher and the tree, on
/// points no tune takes; `cache` reads the times from a file that holds this set, else writes it.
pub fn timings(level: Level, report: bool, cache: Option<&std::path::Path>) -> (Vec<SeriesTiming>, Option<f64>) {
    let build = |probes: &mut Vec<Probe<'static>>| -> Vec<SeriesTiming> {
        let mut timer = Timer { extras: report, probes, rows: Vec::new() };
        family(if report { UNSEEN } else { 0 }, &|quick| WALK * level.fit_shapes(quick), &mut timer);
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
        // A workload's probes, in the order the walk pushed them.
        for v in r.ns.iter_mut().filter(|v| v.is_finite()) {
            *v = timed.ns[at];
            at += 1;
        }
        if report {
            (r.auto, r.tree) = (timed.ns[at], timed.ns[at + 1]);
            at += 2;
        }
        (r.entry, r.decide) = (timed.ns[at], timed.ns[at + 1]);
        at += 2;
    }
    if let Some(path) = cache {
        save(path, &rows);
    }
    (rows, Some(timed.noise))
}

// `--timings FILE`: a line a workload (name, three forced times, dispatcher, tree, entry,
// decision), so a change to the equations is read against the same measurements.
fn load(path: &std::path::Path, rows: &mut [SeriesTiming]) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else { return false };
    let lines: std::collections::HashMap<&str, Vec<f64>> = text
        .lines()
        .filter_map(|l| {
            let mut f = l.split('\t');
            Some((f.next()?, f.filter_map(|v| v.parse().ok()).collect()))
        })
        .collect();
    if !rows.iter().all(|r| lines.get(r.name.as_str()).is_some_and(|v| v.len() == 7)) {
        return false;
    }
    for r in rows {
        let v = &lines[r.name.as_str()];
        (r.ns, r.auto, r.tree, r.entry, r.decide) = ([v[0], v[1], v[2]], v[3], v[4], v[5], v[6]);
    }
    true
}

fn save(path: &std::path::Path, rows: &[SeriesTiming]) {
    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) else { return };
    for r in rows {
        let _ = writeln!(f, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", r.name, r.ns[0], r.ns[1], r.ns[2], r.auto, r.tree, r.entry, r.decide);
    }
}

// Algorithm i's prices over its columns (`EQS`), fitted to its own times in its area on the
// workloads `rows` takes; the other columns keep `base`'s prices. `tied` (into, from, weight)
// prices `from` at `weight` times `into`. Trees split across the pool aren't fitted. Returns the
// prices, the fit and the workload it misses most.
fn fit_cols(t: &[SeriesTiming], i: usize, base: [f64; Parts::N], tied: Option<(usize, usize, f64)>, noise: Option<f64>, rows: Rows) -> ([f64; Parts::N], Fit, String) {
    let (alg, cols) = EQS[i];
    let (mut x, mut y, mut fixed, mut held, mut names) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in t {
        if let (Some(p), Some(own)) = (r.parts[i].filter(|p| r.area[i] && p.wake == 0.0 && r.fitted(rows)), r.own(i)) {
            let mut f = p.features();
            // A tied column is that much of the other's work.
            if let Some((into, from, weight)) = tied {
                f[into] += weight * f[from];
                f[from] = 0.0;
            }
            x.push(cols.iter().map(|&c| f[c]).collect::<Vec<f64>>());
            y.push(own);
            fixed.push((0..Parts::N).filter(|c| !cols.contains(c)).map(|c| base[c] * f[c]).sum::<f64>());
            held.push(r.held);
            names.push(r.name.as_str());
        }
    }
    let keys: Vec<&str> = cols.iter().map(|&c| Parts::NAMES[c]).collect();
    let fit = fit_named(&format!("series {alg} (forced, the entry's own work taken off): a price for each kind of work"), &keys, &x, &y, &fixed, &held, noise);
    let mut prices = base;
    for (&c, &v) in cols.iter().zip(&fit.beta) {
        prices[c] = v;
    }
    if let Some((into, from, weight)) = tied {
        prices[from] = weight * prices[into];
    }
    let worst = fit.worst.map_or(String::new(), |(k, miss)| format!("; worst at {} ({:+.0}%, {:.0} ns)", names[k], 100.0 * miss, y[k]));
    (prices, fit, worst)
}

/// The equations fitted to a set of timings.
pub struct Fitted {
    pub eq: SeriesEq,
    /// The parallel gain (and its index in `GAINS`); the trees split across the pool: (count, rms).
    pub gain: f64,
    pub gi: usize,
    pub par: (usize, f64),
    /// Horner's, rect's and binary splitting's fits, and each one's worst workload.
    pub fits: [Fit; 3],
    pub worst: [String; 3],
}

// (count, relative rms) of the pool-split trees' predictions at gain g.
fn par_rms<'a>(t: impl IntoIterator<Item = &'a SeriesTiming>, g: usize, prices: &[f64; Parts::N], wake: f64) -> (usize, f64) {
    let e: Vec<f64> = t.into_iter().filter(|r| r.gained[0].wake != 0.0).filter_map(|r| r.own(2).map(|own| r.gained[g].ns_with(prices, wake) / own - 1.0)).collect();
    (e.len(), rel_rms(&e))
}

/// Fits the three equations to the workloads `rows` takes: binary splitting on serial trees, then
/// the gain that predicts the pool-split ones best (kept in each timing's parts), Horner, and
/// rect on Horner's prices for the kernels they share.
pub fn fit_equations(t: &mut [SeriesTiming], noise: Option<f64>, rows: Rows) -> Fitted {
    let live = SeriesEq::read();
    let (bin_split, fb, wb) = fit_cols(t, 2, [0.0; Parts::N], None, noise, rows);
    let rms: Vec<(usize, f64)> = (0..GAINS).map(|g| par_rms(t.iter(), g, &bin_split, live.wake)).collect();
    let gi = (0..GAINS).min_by(|&a, &b| rms[a].1.partial_cmp(&rms[b].1).unwrap()).unwrap();
    for r in t.iter_mut() {
        r.parts[2] = Some(r.gained[gi]);
    }
    // A two-limb pass is priced at the kernels' ratio times a one-limb pass.
    let pass2 = (col("mul1"), col("mul2"), crate::tuning::k::series::horner_pass2());
    let (horner, fh, wh) = fit_cols(t, 0, [0.0; Parts::N], Some(pass2), noise, rows);
    // Rect has no price a call or a group.
    let mut base = horner;
    (base[col("call")], base[col("step")]) = (0.0, 0.0);
    let (rect, fr, wr) = fit_cols(t, 1, base, None, noise, rows);
    Fitted { eq: SeriesEq { horner, rect, bin_split, wake: live.wake }, gain: gi as f64 / (GAINS - 1) as f64, gi, par: rms[gi], fits: [fh, fr, fb], worst: [wh, wr, wb] }
}

// Each algorithm's predicted ns under `eq`.
fn predict(r: &SeriesTiming, eq: &SeriesEq) -> [Option<f64>; 3] {
    let prices = prices(eq);
    [0, 1, 2].map(|i| r.parts[i].map(|p| p.ns_with(prices[i], eq.wake)))
}

// The dispatch's pick under `eq`: the live table's, else rect's guard, else the least prediction
// (bin_split on a tie, as `Costs::pick`).
fn pick(r: &SeriesTiming, eq: &SeriesEq) -> usize {
    if let Some(alg) = r.small {
        return ALGS.iter().position(|&a| a == alg).expect("one of the three");
    }
    if r.guard.rect_first {
        return 1;
    }
    let ns = predict(r, eq);
    let mut best = (2, ns[2].expect("bin_split always runs"));
    for i in [1, 0] {
        if let Some(v) = ns[i].filter(|&v| v <= best.1) {
            best = (i, v);
        }
    }
    best.0
}

// (mean, worst, of the summed time) lost by picks against each workload's fastest algorithm. A
// pick that wasn't timed counts as 50 times the best.
fn lost(t: &[&SeriesTiming], pick: impl Fn(&SeriesTiming) -> f64) -> (f64, f64, f64) {
    if t.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let got: Vec<f64> = t.iter().map(|r| Some(pick(r)).filter(|g| g.is_finite()).unwrap_or(50.0 * r.best())).collect();
    let v: Vec<f64> = t.iter().zip(&got).map(|(r, g)| g / r.best() - 1.0).collect();
    let total = got.iter().sum::<f64>() / t.iter().map(|r| r.best()).sum::<f64>() - 1.0;
    (v.iter().sum::<f64>() / v.len() as f64, v.iter().cloned().fold(f64::MIN, f64::max), total)
}

// The guard sweep's loads: the family on points no fit or report takes.
const GUARD_FROM: usize = UNSEEN / 2;
// Turns a guard off: no ratio reaches it.
const OFF: f64 = 1e9;

type Load = Rc<dyn Fn() -> Probe<'static>>;

// Loads run 50 us to 20 ms by their best prediction. The floor keeps the dispatch's own cost
// out of the sweep: a short series is the table's to answer for, not a guard's.
const GUARD_LOADS_NS: (f64, f64) = (5e4, 2e7);

const GUARD_LOADS: usize = 20;

// Collects the guard sweep's loads: the typed entry on every shape rect can sum at a rho of 1 or
// more (where its guard can change the pick), bar those outside `GUARD_LOADS_NS`.
impl Visit for Vec<Load> {
    fn series<C: Series + 'static>(&mut self, _: String, ctx: C, w: usize, _: bool) {
        let Some((zeros, len, (a, b))) = framed(&ctx, w) else { return };
        let node = ctx.dyn_ctx();
        let ok = <C::Dyn as SeriesNode>::eligible(&node, a, b, len, zeros);
        let shape = <C::Dyn as SeriesNode>::shape(&node, a, b, len, zeros);
        let best = least(&<C::Dyn as SeriesNode>::costs(&node, a, b, len, zeros, None));
        let timed = (GUARD_LOADS_NS.0..=GUARD_LOADS_NS.1).contains(&best);
        if !(timed && ok.rect && shape.rho() >= 1.0) {
            return;
        }
        let ctx = Rc::new(ctx);
        self.push(Rc::new(move || {
            let (ctx, mut out) = (ctx.clone(), vec![0u64; len]);
            Probe::new("auto", move || {
                black_box(series_eval_dyn(&*ctx, &mut out));
            })
        }));
    }
}

/// Rect's guard: its rho and its narrowest x, each swept on the dispatcher's entry over up to
/// `GUARD_LOADS` of the shapes found, under the equations in force.
fn sweep_guards(level: Level, out: &mut Vec<Tuned>) {
    let mut found: Vec<Load> = Vec::new();
    family(GUARD_FROM, &|quick| 8 * quick, &mut found);
    // The loads timed, spread evenly through those found.
    let (n, k) = (found.len(), GUARD_LOADS.min(found.len()));
    let all: Vec<usize> = (0..k).collect();
    let rhos: Vec<Value> = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, OFF].into_iter().map(Value::Float).collect();
    for (path, cands) in [("series.rect_rho", rhos), ("series.rect_x_limbs", ints(&[2, 4, 8, 16, 32, 64]))] {
        if k > 0 {
            keep("series", out, param_sweep(level, path, &all, &cands, &|w, c, _| with_knob(path, c, found[w * n / k]())));
        }
    }
}

// A tune's timings, their noise and its first fits' held-out rms: first phase to second.
type Kept = (Vec<SeriesTiming>, Option<f64>, [f64; 3]);

thread_local! {
    static KEPT: RefCell<Option<Kept>> = const { RefCell::new(None) };
}

// The equations' prices and the parallel gain as tuned values, set live. `said` gives a fit's evidence.
fn keep_fitted(f: &Fitted, said: &dyn Fn(usize, &Fit, &str) -> String, out: &mut Vec<Tuned>) {
    const WHAT: [&str; 3] = ["Horner forced", "rect forced, on Horner's prices for the kernels they share", "binary splitting forced, serial trees"];
    for (i, prices) in prices(&f.eq).into_iter().enumerate() {
        let ((alg, cols), what) = (EQS[i], WHAT[i]);
        // A column's knob: `series.<alg>_ns_<column>`.
        let key = |c: &usize| crate::tuning::knob(&format!("series.{alg}_ns_{}", Parts::NAMES[*c])).expect("every fitted column has its knob").name;
        let keys: Vec<&'static str> = cols.iter().map(key).collect();
        let evidence = said(i, &f.fits[i], &format!("{what}, where it competes")) + &f.fits[i].dropped_text(&keys);
        for (&c, &key) in cols.iter().zip(&keys) {
            keep("series", out, Tuned { key, value: Value::Float(prices[c]), evidence: evidence.clone() });
        }
    }
    let evidence = format!("the gain under which the serial trees' prices predict the {} trees that split across the pool best: relative rms {:.1}%", f.par.0, 100.0 * f.par.1);
    // No pool-split tree: nothing to read the gain from.
    if f.par.0 != 0 {
        keep("series", out, Tuned { key: "bin_split_par_gain", value: Value::Float(f.gain), evidence });
    }
}

// `series.horner_pass2`: `mul_prim2`'s ns a limb over `mul_prim`'s, each timed in place over
// 64-1024 limbs and fitted as a L + c. None when a fit gives no slope.
fn pass2_ratio(level: Level) -> Option<Tuned> {
    use crate::utils::mul::{mul_prim, mul_prim2};
    let n = level.fit_shapes(16);
    let lens: Vec<usize> = (0..n).map(|i| (64.0 * 16f64.powf((i as f64 + 0.5) / n as f64)).round() as usize).collect();
    let timed = time_fit(level, |pass| {
        let mut probes = Vec::new();
        for (i, &l) in lens.iter().enumerate() {
            // Odd multipliers never run a buffer to 0.
            let (mut a, mut b, m) = (pad(l, 2 * i as u64 + 3, pass), pad(l, 2 * i as u64 + 5, pass), 0x9e37_79b9_7f4a_7c15u64);
            probes.push(Probe::new("mul_prim", move || {
                black_box(mul_prim(black_box(&mut a), m));
            }));
            probes.push(Probe::new("mul_prim2", move || {
                black_box(mul_prim2(black_box(&mut b), ((m as u128) << 64) | m as u128));
            }));
        }
        probes
    });
    let x: Vec<Vec<f64>> = lens.iter().map(|&l| vec![l as f64, 1.0]).collect();
    let of = |from: usize| -> Vec<f64> { timed.ns.iter().skip(from).step_by(2).cloned().collect() };
    let (one, two) = (fit_rel(&x, &of(0)), fit_rel(&x, &of(1)));
    if !(one.beta[0] > 0.0 && two.beta[0] > 0.0) {
        return None;
    }
    let evidence = format!(
        "mul_prim2 at {:.3} ns a limb over mul_prim at {:.3}, each timed in place at {n} lengths of 64-1024 limbs and fitted as a L + c (relative rms {:.1}% and {:.1}%; passes differ {:.1}%)",
        two.beta[0],
        one.beta[0],
        100.0 * two.rel_rms,
        100.0 * one.rel_rms,
        100.0 * timed.noise
    );
    Some(Tuned { key: "horner_pass2", value: Value::Float((two.beta[0] / one.beta[0]).clamp(1.0, 16.0)), evidence })
}

/// First phase: the equations fitted to the family, on the series the rule on tables would leave
/// them one by one (`Rows::Share`). The table is measured under this fit, which carries no verdict.
pub fn tune(level: Level) -> Vec<Tuned> {
    let saved = crate::tuning::snapshot();
    let mut out = Vec::new();
    let share = tables::share();
    let (mut t, noise) = timings(level, false, None);
    // After the timings: as a tune's first measurement both kernels read 1.85x slow.
    if let Some(ratio) = pass2_ratio(level) {
        keep("series", &mut out, ratio);
    }
    let f = fit_equations(&mut t, noise, Rows::Share(share));
    let (read, all) = (t.iter().filter(|r| r.read(share)).count(), t.len());
    let said = |_: usize, fit: &Fit, what: &str| {
        format!(
            "a first fit, which the table is measured under (fitted again after it, on the series it leaves to the equations): {} shapes, held out {:.1}% ({what}; of the {read} of {all} synthetic series whose decision is within {:.0}% of the call)",
            fit.n,
            100.0 * fit.judged(),
            100.0 * share
        )
    };
    keep_fitted(&f, &said, &mut out);
    let first = [0, 1, 2].map(|i| f.fits[i].judged());
    KEPT.with(|k| *k.borrow_mut() = Some((t, noise, first)));
    crate::tuning::restore(&saved);
    out
}

// The table's pick for every workload of the walk, by name.
impl Visit for std::collections::HashMap<String, Option<SeriesAlg>> {
    fn series<C: Series + 'static>(&mut self, name: String, ctx: C, w: usize, _: bool) {
        let Some((zeros, len, (a, b))) = framed(&ctx, w) else { return };
        self.insert(name, <C::Dyn as SeriesNode>::small(&ctx.dyn_ctx(), a, b, len, zeros).map(|s| s.0));
    }
}

/// Second phase, once `series.small` is in force: the equations fitted again from the first
/// phase's timings on the series the table leaves them (`Rows::Table`), then rect's guard. A fit
/// above the line reads NOISY (the model was revised once). Nothing without a first phase.
pub fn tune_late(level: Level) -> Vec<Tuned> {
    let Some((mut t, noise, first)) = KEPT.with(|k| k.borrow_mut().take()) else { return Vec::new() };
    let saved = crate::tuning::snapshot();
    let mut out = Vec::new();
    // The table just measured decides each timing's pick now.
    let mut named = std::collections::HashMap::new();
    family(0, &|quick| WALK * level.fit_shapes(quick), &mut named);
    for r in t.iter_mut() {
        r.small = named[&r.name];
    }
    let f = fit_equations(&mut t, noise, Rows::Table);
    let (read, all, share) = (t.iter().filter(|r| r.fitted(Rows::Table)).count(), t.len(), tables::share());
    // What the two rules differ by.
    let taken = t.iter().filter(|r| r.read(share) && !r.fitted(Rows::Table)).count();
    let handed = t.iter().filter(|r| !r.read(share) && r.fitted(Rows::Table)).count();
    let said = |i: usize, fit: &Fit, what: &str| {
        format!(
            "{} ({what}; the {read} of {all} synthetic series series.small names no algorithm for, the calls the equations are read for: the table takes {taken} whose own decision is within {:.0}% of the call and hands on {handed} past it; the first fit, on the series within it, read {:.1}% held out; the entry's own work taken off{})",
            fit.evidence_revised(),
            100.0 * share,
            100.0 * first[i],
            f.worst[i]
        )
    };
    keep_fitted(&f, &said, &mut out);
    sweep_guards(level, &mut out);
    crate::tuning::restore(&saved);
    out
}

// The small-series table's rows: every kind, an octave of frames at a time. A kind closes once a
// frame's first index is the equations'.
struct SmallSpec {
    terms: Vec<u32>,
    bucket: usize,
    open: Vec<usize>,
    rows: Vec<Row>,
}

// A kind's members, the series its cells are timed on: a width of x (ten), q's bits
// (`MEMBER_Q`) and an r past 1 or not, numbered (width * 4 + q) * 2 + r.
pub(crate) const MEMBERS: usize = MEMBER_KIND.len() * MEMBER_Q.len() * 2;
const MEMBER_Q: [u64; 4] = [8, 16, 32, 62];
// The table's kind of each width.
const MEMBER_KIND: [usize; 10] = [0, 1, 2, 3, 3, 4, 5, 5, 6, 7];

pub(crate) fn member_kind(member: usize) -> usize {
    MEMBER_KIND[member / 2 / MEMBER_Q.len()]
}

// A member's series at `len` limbs and `n` terms, as probes (each algorithm that can sum it,
// forced, then the decision); None where the member has no series.
pub(crate) fn member_cell(member: usize, len: usize, n: u64, probes: &mut Vec<Probe<'static>>) -> Option<Plan> {
    // Each Hyper width's x in bits.
    const X_BITS: [u64; 8] = [0, 12, 60, 2 * 64 - 8, 7 * 64, 14 * 64, 28 * 64, 48 * 64];
    let (rc, qc, width) = (member % 2, member / 2 % MEMBER_Q.len(), member / 2 / MEMBER_Q.len());
    if width >= MEMBER_KIND.len() || (width < 2 && rc == 1) {
        return None;
    }
    let class = MEMBER_KIND[width];
    // The shift's inputs, by the member's own rule: the frame's bits over the terms after the first.
    let (bits, after) = (64 * len as u64, n.max(2) - 1);
    fn cell<C: Series + 'static>(ctx: C, class: usize, len: usize, n: u64, probes: &mut Vec<Probe<'static>>) -> Option<Plan> {
        let (zeros, a) = (ctx.zeros(), ctx.start());
        let node = Rc::new(ctx.dyn_ctx());
        // Not filed under the member's kind: it would be timed for the wrong row.
        if <C::Dyn as SeriesNode>::small_class(&node, a, a + n) != Some(class) {
            return None;
        }
        let ok = <C::Dyn as SeriesNode>::eligible(&node, a, a + n, len, zeros);
        let mut picks = Vec::new();
        for (i, alg) in ALGS.into_iter().enumerate() {
            if !ok.allows(alg) || (alg == SeriesAlg::Rect && !<C::Dyn as SeriesNode>::RECT) {
                continue;
            }
            debug_assert_eq!(SMALL_PICKS[i], Some(alg));
            picks.push(i as u32);
            let (node, mut out) = (node.clone(), vec![0u64; len]);
            probes.push(Probe::new("cell", move || {
                black_box(series_eval_at::<C::Dyn>(&node, a, a + n, zeros, &mut out, alg));
            }));
        }
        // For the dump: the series' numbers and its two ratios.
        let shape = <C::Dyn as SeriesNode>::shape(&node, a, a + n, len, zeros);
        let note: Vec<f64> = shape.dials().into_iter().chain([shape.rho(), shape.beta()]).collect();
        probes.push(decide_probe::<C>(node, (a, a + n), (len, zeros)));
        Some(Plan { harness: false, picks, decide: true, least: 1.0, note })
    }
    // The cell of the context `$ctx` makes of the member `$S` with q of the member's bits.
    macro_rules! by_q {
        ($member:ident, |$S:ident| $ctx:expr) => {
            match qc {
                0 => by_q!(@ $member<8>, $S, $ctx),
                1 => by_q!(@ $member<16>, $S, $ctx),
                2 => by_q!(@ $member<32>, $S, $ctx),
                _ => by_q!(@ $member<62>, $S, $ctx),
            }
        };
        (@ $ty:ty, $S:ident, $ctx:expr) => {{
            type $S = $ty;
            cell($ctx, class, len, n, probes)
        }};
    }
    match (width, rc) {
        (0, _) => by_q!(Big, |S| Bbp::<S>::new(S::decay(bits, after))),
        (1, _) => by_q!(Den, |S| Engel::<S>::new(S::decay(bits, after))),
        (width, rc) => {
            let xb = X_BITS[width - 2];
            let x: &'static [u64] = if xb == 0 { &[1] } else { x_static(xb, 31 * xb + len as u64) };
            if rc == 1 {
                by_q!(Geo, |S| Hyper::<S>::new(x, xb + S::decay(bits, after)).ok()?)
            } else {
                by_q!(Den, |S| Hyper::<S>::new(x, xb + S::decay(bits, after)).ok()?)
            }
        }
    }
}

impl Spec for SmallSpec {
    fn equations(&self) -> Option<u32> {
        Some((SMALL_PICKS.len() - 1) as u32)
    }

    fn group(&mut self) -> Option<Vec<(u32, Vec<u32>)>> {
        if self.bucket >= SMALL_ROWS || self.open.is_empty() {
            return None;
        }
        Some(self.open.iter().map(|&kind| ((kind * SMALL_ROWS + self.bucket) as u32, self.terms.clone())).collect())
    }

    fn probes(&self, key: u32, size: u32, _: usize, probes: &mut Vec<Probe<'static>>) -> Vec<Plan> {
        let (kind, bucket) = (key as usize / SMALL_ROWS, key as usize % SMALL_ROWS);
        // The octave's smallest frame: its shortest series.
        let frame = small_frame_span(bucket).0;
        // A grid step is timed at its last term count (binary splitting steps up where its Q passes
        // a limb); the rule is read at its first, that fraction of the terms timed.
        let i = self.terms.iter().position(|&t| t == size).expect("a size of the grid");
        let count = self.terms.get(i + 1).map_or(size, |&next| next - 1) as u64;
        let least = size as f64 / count as f64;
        (0..MEMBERS).filter(|&m| member_kind(m) == kind).filter_map(|m| member_cell(m, frame, count, probes)).map(|p| Plan { least, ..p }).collect()
    }

    fn settled(&mut self, rows: Vec<Row>) {
        let equations = self.equations();
        self.open.retain(|&kind| !rows.iter().any(|r| r.key as usize / SMALL_ROWS == kind && r.all_equations(equations)));
        self.rows.extend(rows);
        self.bucket += 1;
    }
}

/// The `series.small` table and its evidence: for every kind and octave of frames, over the terms
/// by thirds of an octave, the kind's members each summed by every algorithm that can, with the
/// decision beside them (`tables::measure`).
pub fn small_table(level: Level) -> (Vec<u32>, String) {
    let mut spec = SmallSpec { terms: tables::octave_steps(1, 1 << 15, 3.0), bucket: 0, open: (0..SMALL_KINDS).collect(), rows: Vec::new() };
    let eps = level.eps().max(0.03);
    // Timed at the level below: tens of thousands of probes.
    let tally = tables::measure(level.lighter(), &mut spec, eps);
    spec.rows.sort_by_key(|r| r.key);
    if verbose() {
        // Every series at every index: a layout can be scored on these without timing again.
        let mut tsv = vec![format!("kind\tbucket\tstep\tleast\t{}\trho\tbeta\thorner\trect\tbin_split\tdecide", Shape::DIALS.join("\t"))];
        for r in &spec.rows {
            for c in &r.cells {
                for m in &c.of {
                    let of = |pick: u32| m.ns.iter().find(|x| x.0 == pick).map_or(f64::NAN, |x| x.1);
                    let note: Vec<String> = m.note.iter().map(|v| format!("{v}")).collect();
                    tsv.push(format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        r.key as usize / SMALL_ROWS,
                        r.key as usize % SMALL_ROWS,
                        c.size,
                        c.least,
                        note.join("\t"),
                        of(0),
                        of(1),
                        of(2),
                        m.decide.unwrap_or(f64::NAN)
                    ));
                }
            }
        }
        dump("series_cells.tsv", &(tsv.join("\n") + "\n"));
    }
    let same = |a: u32, b: u32| a as usize / SMALL_ROWS == b as usize / SMALL_ROWS;
    let t = tables::assemble(&spec.rows, same);
    let equations = spec.equations();
    let others = tables::sizes_above(&spec.rows, equations, eps, same);
    // The table's cells by pick.
    let mut cells = [0usize; 3];
    for r in &spec.rows {
        for c in &r.cells {
            let pick = tables::pick_at(&r.points, c.size).unwrap_or(u32::MAX) as usize;
            if pick < 3 {
                cells[pick] += 1;
            }
        }
    }
    let all = cells.iter().sum::<usize>().max(1) as f64;
    let ev = format!(
        "measured (table): {} change points over {} kinds, each cell on its kind's members ({others}); {}; inside it Horner has {:.0}% of the cells, rect {:.0}%, binary splitting {:.0}%",
        t.len() / 3,
        SMALL_KINDS,
        tally.text(),
        100.0 * cells[0] as f64 / all,
        100.0 * cells[1] as f64 / all,
        100.0 * cells[2] as f64 / all
    );
    (t, ev)
}

// One set's lines: the picks' loss, each equation's rms under `eq`, and the costliest picks.
fn show(label: &str, t: &[&SeriesTiming], eq: &SeriesEq) {
    let l = lost(t, |r| r.ns[pick(r, eq)]);
    // Each equation's relative rms against its algorithm's own times.
    let e = [0, 1, 2].map(|i| rel_rms(&t.iter().filter_map(|r| Some(predict(r, eq)[i]? / r.own(i)? - 1.0)).collect::<Vec<f64>>()));
    println!(
        "    {label:<16} picks lose mean {:.2}%, worst {:.1}%, {:.2}% of the total time; rms Horner {:.1}%, rect {:.1}%, bin_split {:.1}%",
        100.0 * l.0, 100.0 * l.1, 100.0 * l.2, 100.0 * e[0], 100.0 * e[1], 100.0 * e[2]
    );
    let mut worst: Vec<(f64, &SeriesTiming)> = t.iter().map(|r| (r.ns[pick(r, eq)] / r.best() - 1.0, *r)).collect();
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (loss, r) in worst.iter().take(4).filter(|w| w.0 > 0.05) {
        let pred: Vec<String> = predict(r, eq).iter().map(|p| p.map_or("-".into(), |v| format!("{v:.0}"))).collect();
        println!(
            "      {}: picks {:?}{}, {:.0}% over; ns H {:.0} R {:.0} B {:.0} (entry {:.0}, decision {:.0}); predicted {} / {} / {}",
            r.name, ALGS[pick(r, eq)], if r.small.is_some() { " (the table)" } else { "" }, 100.0 * loss, r.ns[0], r.ns[1], r.ns[2], r.entry, r.decide, pred[0], pred[1], pred[2]
        );
    }
}

// The entry's own loss on a set, by the fastest algorithm's time, and its two worst workloads.
fn show_entry(t: &[&SeriesTiming]) {
    let l = lost(t, |r| r.auto);
    println!("    the entry itself: mean {:.2}%, worst {:.1}%, {:.2}% of the total time   (its dispatch included)", 100.0 * l.0, 100.0 * l.1, 100.0 * l.2);
    let bands: Vec<String> = [(0.0, 1e3, "under 1 us"), (1e3, 1e4, "1-10 us"), (1e4, 1e5, "10-100 us"), (1e5, f64::INFINITY, "past 100 us")]
        .iter()
        .filter_map(|&(lo, hi, what)| {
            let v: Vec<f64> = t.iter().filter(|r| (lo..hi).contains(&r.best())).map(|r| r.auto / r.best() - 1.0).collect();
            (!v.is_empty()).then(|| format!("{what} {:.1}% ({})", 100.0 * (v.iter().sum::<f64>() / v.len() as f64), v.len()))
        })
        .collect();
    println!("      by the fastest algorithm's time: {}", bands.join(", "));
    let mut worst: Vec<&SeriesTiming> = t.to_vec();
    worst.sort_by(|a, b| (b.auto / b.best()).partial_cmp(&(a.auto / a.best())).unwrap());
    for r in worst.iter().take(2).filter(|r| r.auto / r.best() > 1.1) {
        let ran = r.small.map_or("the equations".to_string(), |alg| format!("the table: {alg:?}"));
        println!("      {}: the entry {:.0} ns against {:.0} ({ran}); ns H {:.0} R {:.0} B {:.0}", r.name, r.auto, r.best(), r.ns[0], r.ns[1], r.ns[2]);
    }
}

/// `cargo tune series`, on workloads no tune fitted: the equations fitted now on the half not
/// held out; on each half the time lost by the picks under the live and the fitted equations and
/// by the entry; the same where the table decides; and what other shares would leave the equations.
pub fn report(level: Level, cache: Option<&std::path::Path>) {
    let here = tables::share();
    // The table in force was measured at the rule's share: under another, a series is the equations' by its own decision.
    let rows = if here == tables::SHARE { Rows::Table } else { Rows::Share(here) };
    println!("the series equations ({}), on the synthetic family", level.name());
    let (mut all, noise) = timings(level, true, cache);
    // The workloads the equations are read for, held out or fitted.
    fn of(t: &[SeriesTiming], rows: Rows, held: bool) -> Vec<&SeriesTiming> {
        t.iter().filter(|r| r.fitted(rows) && r.held == held).collect()
    }
    // `fit_equations` sets the trees' parts to its gain's: the live equations read this copy.
    let (live, as_live) = (SeriesEq::read(), all.clone());
    let f = fit_equations(&mut all, noise, rows);
    let read: Vec<&SeriesTiming> = all.iter().filter(|r| r.fitted(rows)).collect();
    let tabled: Vec<&SeriesTiming> = all.iter().filter(|r| !r.fitted(rows)).collect();
    match rows {
        Rows::Table => println!(
            "{} workloads: the equations are read for {} (series.small names no algorithm for them), and the table has the other {}; by its own decision alone ({:.0}% of the call) the table takes {} that are within it and hands on {} past it",
            all.len(),
            read.len(),
            tabled.len(),
            100.0 * here,
            tabled.iter().filter(|r| r.read(here)).count(),
            read.iter().filter(|r| !r.read(here)).count()
        ),
        Rows::Share(_) => println!(
            "{} workloads: the equations are read for {} (the decision within {:.0}% of the call), and a table at that share has the other {}",
            all.len(),
            read.len(),
            100.0 * here,
            tabled.len()
        ),
    }
    let settled = |f: &dyn Fn(&SeriesTiming) -> bool| all.iter().filter(|r| f(r)).count();
    println!(
        "  as the dispatch stands: the small-series table takes {} and the rect guard {}",
        settled(&|r| r.small.is_some()),
        settled(&|r| r.small.is_none() && r.guard.rect_first)
    );
    let named = |cols: &[usize], prices: &[f64; Parts::N]| -> String {
        cols.iter().map(|&c| format!("{} {:.4}", Parts::NAMES[c], prices[c])).collect::<Vec<_>>().join(", ")
    };
    println!("where the equations are read:");
    for (i, prices) in prices(&f.eq).into_iter().enumerate() {
        println!("  fitted now, {:?}: {}\n    {}", ALGS[i], f.fits[i].evidence_revised(), named(EQS[i].1, prices));
    }
    let held_par = par_rms(of(&all, rows, true), f.gi, &f.eq.bin_split, live.wake);
    if f.par.0 + held_par.0 != 0 {
        println!(
            "  trees that split across the pool: gain {:.2} a thread; rms {:.1}% over {} fitted, {:.1}% over {} held out",
            f.gain, 100.0 * f.par.1, f.par.0, 100.0 * held_par.1, held_par.0
        );
    }
    for (name, held) in [("fitted set", false), ("held-out set", true)] {
        let t = of(&all, rows, held);
        let wins: Vec<usize> = (0..3).map(|i| t.iter().filter(|r| r.ns[i] == r.best()).count()).collect();
        println!("  {name} ({} workloads; fastest: Horner {}, rect {}, bin_split {}):", t.len(), wins[0], wins[1], wins[2]);
        show("fitted now", &t, &f.eq);
    }
    // Other shares on the same timings: the equations' own picks, no table.
    println!("  the rule's number, on these timings (each share's equations fitted to the workloads whose own decision is within it; a table is taken to pick the fastest):");
    let mut again: Vec<SeriesTiming> = Vec::new();
    for share in tables::OTHER_SHARES.into_iter().chain([f64::INFINITY]) {
        again.clear();
        again.extend(all.iter().map(|r| SeriesTiming { small: None, ..r.clone() }));
        let g = fit_equations(&mut again, noise, Rows::Share(share));
        let read: Vec<&SeriesTiming> = again.iter().filter(|r| r.read(share) && r.held).collect();
        let lost_ns: f64 = read.iter().map(|r| Some(r.ns[pick(r, &g.eq)]).filter(|v| v.is_finite()).unwrap_or(50.0 * r.best()) - r.best()).sum();
        let held_ns: f64 = again.iter().filter(|r| r.held).map(|r| r.best()).sum();
        let l = lost(&read, |r| r.ns[pick(r, &g.eq)]);
        println!(
            "    {:>4}: read for {:>3} of {} workloads; held out Horner {:.1}%, rect {:.1}%, bin_split {:.1}%; their picks lose mean {:.2}%, worst {:.1}% ({:.2}% of all the workloads' time)",
            if share.is_finite() { format!("{:.0}%", 100.0 * share) } else { "all".into() },
            again.iter().filter(|r| r.read(share)).count(),
            again.len(),
            100.0 * g.fits[0].judged(),
            100.0 * g.fits[1].judged(),
            100.0 * g.fits[2].judged(),
            100.0 * l.0,
            100.0 * l.1,
            100.0 * lost_ns / held_ns
        );
    }
    if verbose() {
        let mut tsv: Vec<String> = Vec::new();
        for r in all.iter() {
            let pred = predict(r, &f.eq);
            for i in 0..3 {
                if let (Some(p), Some(own)) = (r.parts[i], r.own(i)) {
                    let cols: Vec<String> = p.features().iter().map(|v| format!("{v}")).collect();
                    tsv.push(format!(
                        "{}\t{}\t{}\t{:?}\t{own}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        if r.fitted(rows) { "read" } else { "table" },
                        if r.held { "held" } else { "fit" },
                        r.name,
                        ALGS[i],
                        r.tree,
                        pred[i].unwrap_or(f64::NAN),
                        r.decide,
                        cols.join("\t"),
                        p.wake,
                        r.rho,
                        r.beta,
                        r.area[i] as u8
                    ));
                }
            }
        }
        let head = format!("region\tset\tname\talg\tns\ttree\tpred\tdecide\t{}\twake\trho\tbeta\tarea", Parts::NAMES.join("\t"));
        dump("series_parts.tsv", &format!("{head}\n{}\n", tsv.join("\n")));
    }
    if verbose() {
        // A row a workload: its numbers, times, and the table's and the equations' picks.
        let mut tsv = vec![format!("name\theld\tclass\t{}\trho\tbeta\thorner\trect\tbin_split\tentry\tdecide\tauto\ttable\tequations\tcan", Shape::DIALS.join("\t"))];
        for r in &as_live {
            let dials: Vec<String> = r.dials.iter().map(|v| format!("{v}")).collect();
            let bare = SeriesTiming { small: None, ..r.clone() };
            tsv.push(format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                r.name,
                r.held as u8,
                r.class.map_or(-1, |c| c as i64),
                dials.join("\t"),
                r.rho,
                r.beta,
                r.ns[0],
                r.ns[1],
                r.ns[2],
                r.entry,
                r.decide,
                r.auto,
                r.small.map_or(-1, |alg| ALGS.iter().position(|&a| a == alg).expect("one of the three") as i64),
                pick(&bare, &live),
                // Which algorithms can sum the range.
                (0..3).map(|i| (r.parts[i].is_some() as u8) << i).sum::<u8>()
            ));
        }
        dump("series_workloads.tsv", &(tsv.join("\n") + "\n"));
    }
    for (name, held) in [("fitted set", false), ("held-out set", true)] {
        println!("  {name}:");
        let t = of(&as_live, rows, held);
        show("live equations", &t, &live);
        show_entry(&t);
    }
    let t: Vec<&SeriesTiming> = as_live.iter().filter(|r| !r.fitted(rows)).collect();
    println!("where the table decides ({} workloads):", t.len());
    show("the live table", &t, &live);
    show_entry(&t);
}
