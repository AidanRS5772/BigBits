//! Sweeps: a decision with one input is measured directly, and its value is the one with the
//! least measured time lost. `cutoff_sweep` is a two-sided crossover; `param_sweep` an algorithm
//! parameter with no sides (the candidate that loses least against the best at each size).
//! Sweeps of whole algorithms take a fixed number of samples twice and are judged by whether
//! both measurements support the choice, not by their samples' spread.
//! The tie band: a sweep keeps the knob's previous value unless another loses less by more than
//! the level's eps (`keep_or_take`): where a knob hardly matters, its best value follows noise.

use crate::tuning::resolve::Value;
use crate::tuning::tuner::measure::{time_fixed, time_min, Level, Probe};
use crate::tuning::tuner::out::Tuned;
use crate::tuning::{install, prepare};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A knob's name within its section, from its path.
pub fn key_of(path: &'static str) -> &'static str {
    path.split_once('.').expect("a knob path").1
}

/// The integer of an integer knob's value.
pub fn int(v: &Value) -> u64 {
    match v {
        Value::Int(c) => *c,
        _ => unreachable!("an integer knob's value"),
    }
}

/// Integer candidates as knob values.
pub fn ints(v: &[u64]) -> Vec<Value> {
    v.iter().map(|&x| Value::Int(x)).collect()
}

/// Timings taken as (lower side, upper side) probes in turn, as pairs.
pub fn pairs(t: &[f64]) -> Vec<(f64, f64)> {
    t.chunks(2).map(|c| (c[0], c[1])).collect()
}

/// Sizes `per_octave` to a doubling from `lo`, without end.
pub fn geo(lo: f64, per_octave: f64) -> impl Iterator<Item = usize> {
    (0u32..).map(move |i| (lo * 2f64.powf(i as f64 / per_octave)).round() as usize)
}

/// `geo`'s sizes up to `hi`.
pub fn geo_to(lo: f64, per_octave: f64, hi: usize) -> Vec<usize> {
    geo(lo, per_octave).take_while(|&n| n <= hi).collect()
}

/// Makes a swept value of `section` live for the sweeps after it, and records it (replacing an
/// earlier value of the same knob).
pub fn keep(section: &str, out: &mut Vec<Tuned>, t: Tuned) {
    t.set_live(section).expect("a swept value is valid");
    out.retain(|o| o.key != t.key);
    out.push(t);
}

// Whole-algorithm sweeps run, and those whose two measurements disagreed.
static WHOLE: AtomicUsize = AtomicUsize::new(0);
static UNSTABLE: AtomicUsize = AtomicUsize::new(0);

/// (whole-algorithm sweeps, unstable ones) since the last call, which resets both.
pub fn stability_counts() -> (usize, usize) {
    (WHOLE.swap(0, Ordering::Relaxed), UNSTABLE.swap(0, Ordering::Relaxed))
}

// Cutoff sweeps whose value sits at an end of the sizes they timed: one side never won, so the
// real crossover may lie outside the grid.
static EDGES: AtomicUsize = AtomicUsize::new(0);

/// Those since the last call, which resets it.
pub fn edge_count() -> usize {
    EDGES.swap(0, Ordering::Relaxed)
}

// Two fixed-sample measurements of the probes `build(pass)` makes, and their element-wise least.
// A whole-algorithm sweep chooses from the least, and is stable when that choice is within the
// tie band of the best in each measurement alone.
fn measure_twice(level: Level, build: &dyn Fn(usize) -> Vec<Probe<'static>>) -> ([Vec<f64>; 2], Vec<f64>) {
    let saved = crate::tuning::snapshot();
    let runs = [0, 1].map(|pass| {
        let t = time_fixed(&mut build(pass), level);
        crate::tuning::restore(&saved);
        t
    });
    let least = runs[0].iter().zip(&runs[1]).map(|(a, b)| a.min(*b)).collect();
    WHOLE.fetch_add(1, Ordering::Relaxed);
    (runs, least)
}

// A whole-algorithm sweep's outcome: `picked` when stable, else the previous value (or `picked`
// with none to fall back on), the evidence saying so.
fn settle(picked: Tuned, stable: bool, previous: Option<Value>) -> Tuned {
    if stable {
        return Tuned { evidence: picked.evidence + "; two measurements agree", ..picked };
    }
    // A choice that is the standing value changes nothing.
    if previous.as_ref() == Some(&picked.value) {
        return picked;
    }
    UNSTABLE.fetch_add(1, Ordering::Relaxed);
    match previous {
        Some(p) => Tuned {
            key: picked.key,
            evidence: format!("UNSTABLE: two measurements disagreed (one read {}); the previous value is kept", picked.value.text()),
            value: p,
        },
        None => Tuned { evidence: picked.evidence + "; UNSTABLE: two measurements disagreed", ..picked },
    }
}

// (mean, worst) time lost over shapes: `lower(i)` says whether shape i takes the first of t[i]'s two times.
pub fn lost(t: &[(f64, f64)], lower: impl Fn(usize) -> bool) -> (f64, f64) {
    let (mut sum, mut worst) = (0.0f64, 0.0f64);
    for (i, &(lo, hi)) in t.iter().enumerate() {
        let l = (if lower(i) { lo } else { hi }) / lo.min(hi) - 1.0;
        sum += l;
        worst = worst.max(l);
    }
    (sum / t.len() as f64, worst)
}

// A cutoff sweep's second pass: sizes about 1.5% apart from c / 1.5 to 1.5 c (at most `cap`). A
// transform's time jumps at each size step, which a coarse pass steps over.
pub fn dense_around(c: usize, cap: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (-28..=28).map(|i| (c as f64 * 2f64.powf(i as f64 / 48.0)).round() as usize).filter(|&n| n >= 8 && n <= cap).collect();
    v.dedup();
    v
}

// The tie band's rule and a sweep's evidence. `best` is the candidate with the least mean loss
// and `previous` the value in force, each with its (mean, worst) loss: the previous stays when
// it isn't the best and loses within `tie` of it. Returns the value, whether it was kept that
// way, and the evidence (`what`, then the loss, then `edge`).
pub fn keep_or_take<T: Copy + PartialEq>(
    what: &str,
    edge: &str,
    best: (T, (f64, f64)),
    previous: Option<(T, (f64, f64))>,
    tie: f64,
    show: impl Fn(T) -> String,
) -> (T, bool, String) {
    if let Some((p, l)) = previous.filter(|(p, l)| *p != best.0 && l.0 - best.1 .0 <= tie) {
        let evidence = format!(
            "{what}: kept, measured time lost mean {:.2}%, worst {:.1}%, within the tie band of the best ({}: {:.2}%)",
            100.0 * l.0, 100.0 * l.1, show(best.0), 100.0 * best.1 .0
        );
        return (p, true, evidence);
    }
    let before = previous
        .map(|(p, l)| format!("; the previous value {} lost {:.2}%, worst {:.1}%", show(p), 100.0 * l.0, 100.0 * l.1))
        .unwrap_or_default();
    (best.0, false, format!("{what}: measured time lost mean {:.2}%, worst {:.1}%{edge}{before}", 100.0 * best.1 .0, 100.0 * best.1 .1))
}

// The cutoff with the least mean loss, among below every size and each size, and its loss.
fn best_cutoff(sizes: &[usize], t: &[(f64, f64)]) -> (usize, (f64, f64)) {
    std::iter::once(sizes[0] - 1)
        .chain(sizes.iter().cloned())
        .map(|c| (c, lost(t, |i| sizes[i] <= c)))
        .min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
        .unwrap()
}

// The cutoff knob at `path` as a sweep over `sizes` (the lower side while n + plus <= knob; plus
// = 1: "below the knob"): the knob, and the cut among `sizes`.
pub fn sweep_cutoff(path: &'static str, sizes: &[usize], t: &[(f64, f64)], plus: usize, tie: f64) -> (Tuned, usize) {
    let best = best_cutoff(sizes, t);
    let edge = if best.0 == *sizes.last().unwrap() {
        " (at the sweep's top: the upper side never won)"
    } else if best.0 == sizes[0] - 1 {
        " (at the sweep's bottom: the lower side never won)"
    } else {
        ""
    };
    // The value in force, on the same timings.
    let previous = match crate::tuning::get(path) {
        Some(Value::Int(c)) => Some((c, lost(t, |i| (sizes[i] + plus) as u64 <= c))),
        _ => None,
    };
    let range = format!("sweep {}-{}, {} sizes", sizes[0], sizes[sizes.len() - 1], sizes.len());
    let (value, kept, evidence) = keep_or_take(&range, edge, ((best.0 + plus) as u64, best.1), previous, tie, |c| c.to_string());
    if !kept && !edge.is_empty() {
        EDGES.fetch_add(1, Ordering::Relaxed);
    }
    let cut = if kept { (value as usize).saturating_sub(plus) } else { best.0 };
    (Tuned { key: key_of(path), value: Value::Int(value), evidence }, cut)
}

// A one-input cutoff as a sweep: `probes(n, pass)` gives the lower and upper side at size n. One
// pass over `coarse`, then, with `dense` (the most a size may be), sizes 1.5% apart around its
// crossing.
pub fn cutoff_sweep(
    level: Level,
    path: &'static str,
    coarse: &[usize],
    dense: Option<usize>,
    plus: usize,
    probes: &dyn Fn(usize, usize) -> [Probe<'static>; 2],
) -> Tuned {
    // Probes may force knobs while they run: the values from before come back after each measurement.
    let saved = crate::tuning::snapshot();
    let run = |sizes: &[usize]| -> Vec<(f64, f64)> {
        let t = time_min(level, |pass| sizes.iter().flat_map(|&n| probes(n, pass)).collect());
        crate::tuning::restore(&saved);
        pairs(&t)
    };
    let t = run(coarse);
    let edges = EDGES.load(Ordering::Relaxed);
    let (tuned, c) = sweep_cutoff(path, coarse, &t, plus, level.eps());
    let Some(cap) = dense else { return tuned };
    // Only the dense pass's result is the sweep's: the coarse grid's edge isn't counted.
    EDGES.store(edges, Ordering::Relaxed);
    let sizes: Vec<usize> = dense_around(c, cap).into_iter().filter(|&n| n >= coarse[0]).collect();
    sweep_cutoff(path, &sizes, &run(&sizes), plus, level.eps()).0
}

/// `probe` with the knob `path` at `value`, installed before each call.
pub fn with_knob<'a>(path: &str, value: &Value, mut probe: Probe<'a>) -> Probe<'a> {
    let hd = prepare(path, value).expect("a valid value for the swept knob");
    Probe::new(probe.name.clone(), move || {
        install(hd);
        (probe.run)();
    })
}

// The least mean loss of any of the k candidates over n sizes, from t[size * k + candidate].
fn best_loss(n: usize, k: usize, t: &[f64]) -> f64 {
    (0..k).map(|c| param_loss(n, k, t, c).0).fold(f64::INFINITY, f64::min)
}

// A parameter sweep but for its choice: `choose(timings, previous)` gives the candidate's index
// and the evidence, from the least of two measurements.
fn param_sweep_by(
    level: Level,
    path: &'static str,
    sizes: &[usize],
    cands: &[Value],
    probe: &dyn Fn(usize, &Value, usize) -> Probe<'static>,
    choose: impl FnOnce(&[f64], Option<&Value>) -> (usize, String),
) -> Tuned {
    let (runs, least) = measure_twice(level, &|pass| sizes.iter().flat_map(|&n| cands.iter().map(move |c| probe(n, c, pass))).collect());
    let previous = crate::tuning::get(path);
    let (chosen, evidence) = choose(&least, previous.as_ref());
    let (n, k) = (sizes.len(), cands.len());
    let stable = runs.iter().all(|t| param_loss(n, k, t, chosen).0 - best_loss(n, k, t) <= level.eps());
    settle(Tuned { key: key_of(path), value: cands[chosen].clone(), evidence }, stable, previous.filter(|p| cands.contains(p)))
}

/// An algorithm parameter as a sweep: `probe(size, candidate, pass)` is one call at `size` under
/// `candidate`. The value loses least on the mean against the best candidate at each size.
pub fn param_sweep(
    level: Level,
    path: &'static str,
    sizes: &[usize],
    cands: &[Value],
    probe: &dyn Fn(usize, &Value, usize) -> Probe<'static>,
) -> Tuned {
    param_sweep_by(level, path, sizes, cands, probe, |least, previous| pick_param(sizes.len(), cands, least, previous, level.eps()))
}

/// `param_sweep` for a gate that should stand as low as it can: the first candidate within the
/// tie band of the best. Past where a gate does its job its loss is flat, and one set higher
/// than it needs to be shuts out shapes no workload covers.
pub fn param_sweep_lowest(
    level: Level,
    path: &'static str,
    sizes: &[usize],
    cands: &[Value],
    probe: &dyn Fn(usize, &Value, usize) -> Probe<'static>,
) -> Tuned {
    let (n, k) = (sizes.len(), cands.len());
    param_sweep_by(level, path, sizes, cands, probe, |least, _| {
        let best = best_loss(n, k, least);
        let chosen = (0..k).find(|&c| param_loss(n, k, least, c).0 - best <= level.eps()).expect("the best is within its own band");
        let (mean, worst) = param_loss(n, k, least, chosen);
        let evidence = format!(
            "sweep of {k} values over {n} sizes: the lowest within the tie band of the best ({:.2}%), measured time lost mean {:.2}%, worst {:.1}%",
            100.0 * best, 100.0 * mean, 100.0 * worst
        );
        (chosen, evidence)
    })
}

/// A cutoff between two whole-algorithm code paths over `sizes` (no dense pass), judged by stability.
pub fn cutoff_sweep_whole(
    level: Level,
    path: &'static str,
    sizes: &[usize],
    plus: usize,
    probes: &dyn Fn(usize, usize) -> [Probe<'static>; 2],
) -> Tuned {
    let (runs, least) = measure_twice(level, &|pass| sizes.iter().flat_map(|&n| probes(n, pass)).collect());
    let previous = crate::tuning::get(path);
    let (picked, _) = sweep_cutoff(path, sizes, &pairs(&least), plus, level.eps());
    let value = int(&picked.value);
    let stable = runs.iter().all(|t| {
        let t = pairs(t);
        lost(&t, |i| (sizes[i] + plus) as u64 <= value).0 - best_cutoff(sizes, &t).1 .0 <= level.eps()
    });
    settle(picked, stable, previous)
}

/// Candidate c's (mean, worst) loss against the best candidate at each size, from t[size * k + candidate].
pub fn param_loss(sizes: usize, k: usize, t: &[f64], c: usize) -> (f64, f64) {
    let rel: Vec<f64> = (0..sizes).map(|s| t[s * k + c] / t[s * k..(s + 1) * k].iter().cloned().fold(f64::INFINITY, f64::min) - 1.0).collect();
    (rel.iter().sum::<f64>() / sizes as f64, rel.iter().cloned().fold(0.0, f64::max))
}

/// `param_sweep`'s choice: the candidate's index and the evidence. `previous` is kept when its
/// mean loss is within `tie` of the best's.
pub fn pick_param(sizes: usize, cands: &[Value], t: &[f64], previous: Option<&Value>, tie: f64) -> (usize, String) {
    let k = cands.len();
    let loss: Vec<(f64, f64)> = (0..k).map(|c| param_loss(sizes, k, t, c)).collect();
    let best = (0..k).min_by(|&a, &b| loss[a].0.partial_cmp(&loss[b].0).unwrap()).unwrap();
    let prev = previous.and_then(|p| cands.iter().position(|c| c == p));
    let what = format!("sweep of {k} values over {sizes} sizes");
    let (chosen, _, evidence) = keep_or_take(&what, "", (best, loss[best]), prev.map(|i| (i, loss[i])), tie, |i| cands[i].text());
    (chosen, evidence)
}
