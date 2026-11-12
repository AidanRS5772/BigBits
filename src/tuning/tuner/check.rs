//! The end-to-end check: the same public calls timed under two sets of knob values, in one
//! process. `cargo tune check` reports it for a tuning file against the layers below it, and
//! every `cargo tune tune` ends each section with it as a gate (the user's decision of
//! 2026-11-10): a section whose values make its own workloads, and those of every section built
//! on it, slower on average than the values before it, by more than the level's allowance
//! (`Level::gate`: 3%, 2%, 1%), is dropped, so a tune never leaves a machine more than that
//! slower than it found it. The workloads are `check_items`' (shared with `benches/tuning_bias.rs`).
//!
//! Each round times one batch of every item under one set, then under the other, alternating
//! which goes first; an item's result is the ratio of the pair, so the machine's speed at that
//! moment cancels. A section's verdict is the mean log ratio over its items, with a 95% interval
//! from the rounds.

use super::check_items::{items, Item, T95};
use super::measure::Level;
use crate::tuning::schema_types::Section;
use crate::tuning::{restore, Snapshot};
use std::time::Instant;

/// The schema sections whose knobs an item section's calls read directly.
pub fn owners(item_section: &str) -> &'static [&'static str] {
    match item_section {
        "mul" => &["mul"],
        "div" | "divgrid" => &["div"],
        "sqrt" => &["sqrt"],
        "series" => &["bin_split", "series"],
        "consts" => &["bin_split", "consts"],
        "trans" => &["theta", "agm", "exp", "ln"],
        _ => &[],
    }
}

// Whether `section` is `into` or upstream of it, through the schema's `after` lists.
fn feeds(schema: &[Section], section: &str, into: &str) -> bool {
    section == into || schema.iter().find(|s| s.name == into).is_some_and(|s| s.after.iter().any(|a| feeds(schema, section, a)))
}

/// Whether an item section belongs to the gate of the schema section `tuned`: its calls read
/// `tuned`'s knobs, directly or through a section built on it.
pub fn in_gate(schema: &[Section], tuned: &str, item_section: &str) -> bool {
    owners(item_section).iter().any(|o| feeds(schema, tuned, o))
}

/// Whether an item's outputs are the same under any valid knob values: full products, squares,
/// residues and divisions. High and middle products, reciprocals, roots and the transcendental
/// entries are specified only to within their error, which the algorithm chosen moves.
pub fn exact(name: &str) -> bool {
    ["mul_dyn", "mul_static", "sqr_dyn", "sqr_static", "sqr_mod_bnm1_dyn", "div_dyn", "div_rem_dyn", "div_static", "div_rem_static"]
        .iter()
        .any(|p| name.starts_with(p) && name[p.len()..].starts_with([' ', '<']))
}

/// One item's paired rounds: ln(time after / time before) per round, and whether its outputs
/// agreed under the two sets.
#[derive(Clone, Debug)]
pub struct Rounds {
    pub section: &'static str,
    pub name: String,
    pub logs: Vec<f64>,
    pub same: bool,
}

/// Times every item `pick` selects under `before` and under `after`. Leaves `after` live.
pub fn compare(before: &Snapshot, after: &Snapshot, level: Level, pick: &dyn Fn(&Item) -> bool) -> Vec<Rounds> {
    let mut its: Vec<Item> = items().into_iter().filter(|i| pick(i)).collect();
    // One warm call under each set gives the outputs' hashes and a first time; a batch then lasts
    // about the level's batch time under the slower of the two.
    let once = |its: &mut Vec<Item>| -> Vec<(u64, f64)> {
        its.iter_mut()
            .map(|it| {
                let hash = (it.run)(true);
                let t = Instant::now();
                (it.run)(false);
                (hash, t.elapsed().as_nanos().max(1) as f64)
            })
            .collect()
    };
    restore(before);
    let first = once(&mut its);
    restore(after);
    let second = once(&mut its);
    let batch: Vec<u32> = first.iter().zip(&second).map(|(a, b)| (level.batch_ns() / a.1.max(b.1)).clamp(1.0, 1e6) as u32).collect();
    let mut out: Vec<Rounds> = its
        .iter()
        .zip(first.iter().zip(&second))
        .map(|(it, (a, b))| Rounds { section: it.section, name: it.name.clone(), logs: Vec::new(), same: a.0 == b.0 })
        .collect();
    let mut ns = [vec![0.0f64; its.len()], vec![0.0f64; its.len()]];
    for round in 0..level.rounds() {
        // Before first on even rounds, after first on odd ones.
        for side in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            restore(if side == 0 { before } else { after });
            for (i, it) in its.iter_mut().enumerate() {
                let t = Instant::now();
                for _ in 0..batch[i] {
                    (it.run)(false);
                }
                ns[side][i] = t.elapsed().as_nanos().max(1) as f64 / batch[i] as f64;
            }
        }
        for (i, r) in out.iter_mut().enumerate() {
            r.logs.push((ns[1][i] / ns[0][i]).ln());
        }
    }
    restore(after);
    out
}

// The mean of the samples and the half-width of its 95% interval.
fn mean_interval(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    if x.is_empty() {
        return (0.0, 0.0);
    }
    let mean = x.iter().sum::<f64>() / n;
    if x.len() < 2 {
        return (mean, f64::INFINITY);
    }
    let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    (mean, T95[(x.len() - 2).min(29)] * sd / n.sqrt())
}

/// A ratio after / before with its 95% interval.
#[derive(Clone, Debug)]
pub struct Stat {
    pub name: String,
    pub ratio: f64,
    pub lo: f64,
    pub hi: f64,
}

/// What a set of items says about the values after against the values before.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub items: usize,
    /// The geometric mean of after / before over the items, with its interval over the rounds.
    pub all: Stat,
    /// Items more than 2% slower, with an interval that excludes 1: listed, never a reason to drop.
    pub slower: Vec<Stat>,
    /// Exact items whose outputs differ between the two sets: a correctness failure.
    pub differ: Vec<String>,
    /// Whether the values after lose: the mean is past 1 + the allowance with an interval that
    /// excludes 1, or an exact output differs.
    pub dropped: bool,
}

/// The verdict of `rounds` at allowance `eps` (`Level::gate`, for the tuner).
pub fn verdict(rounds: &[Rounds], eps: f64) -> Verdict {
    let stat = |name: &str, logs: &[f64]| {
        let (mean, half) = mean_interval(logs);
        Stat { name: name.to_string(), ratio: mean.exp(), lo: (mean - half).exp(), hi: (mean + half).exp() }
    };
    let n = rounds.first().map_or(0, |r| r.logs.len());
    let per_round: Vec<f64> = (0..n).map(|k| rounds.iter().map(|r| r.logs[k]).sum::<f64>() / rounds.len() as f64).collect();
    let all = stat("all", &per_round);
    let slower: Vec<Stat> = rounds.iter().map(|r| stat(&r.name, &r.logs)).filter(|s| s.ratio > 1.02 && s.lo > 1.0).collect();
    let differ: Vec<String> = rounds.iter().filter(|r| !r.same && exact(&r.name)).map(|r| r.name.clone()).collect();
    let dropped = !differ.is_empty() || (all.ratio > 1.0 + eps && all.lo > 1.0);
    Verdict { items: rounds.len(), all, slower, differ, dropped }
}

/// Prints a verdict: its mean and interval, the items listed as slower, and any exact output
/// that differs.
pub fn report(what: &str, v: &Verdict) {
    if v.items == 0 {
        eprintln!("  check {what}: no workloads");
        return;
    }
    eprintln!("  check {what}: after / before {:.4} [{:.4}, {:.4}] over {} workloads", v.all.ratio, v.all.lo, v.all.hi, v.items);
    for s in &v.slower {
        eprintln!("    slower: {:<34} {:.3} [{:.3}, {:.3}]", s.name, s.ratio, s.lo, s.hi);
    }
    for name in &v.differ {
        eprintln!("    OUTPUT DIFFERS: {name}");
    }
}
