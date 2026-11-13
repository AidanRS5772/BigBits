//! The rule every dispatcher is held to, checked (`cargo tune dispatch`, and the end of a tune):
//! where a dispatcher reads its fitted equations, that decision is within `tables::SHARE` of the
//! call; an index where it would cost more belongs in the dispatcher's measured table.
//! A table whose rows stand for kinds of call (`series.small`) may hand an index past the share
//! to the equations as its own pick, where no one algorithm serves the kind: the check reports
//! those and doesn't count them against the rule.
//! `TABLES` is every dispatcher in the library that reads fitted equations at run time, with its
//! table; `test_every_table_is_a_dispatchers` holds the schema to it.

use crate::tuning::k;
use crate::tuning::tuner::measure::{less_harness, pad, time_fit, Level, Probe};
use crate::tuning::tuner::sections::div::{copy_probe, numerator, placed_divisor};
use crate::tuning::tuner::sections::series::{decide_probe, framed};
use crate::tuning::tuner::synth::{x_static, Den, Fact, Geo, Odd, Synth};
use crate::tuning::tuner::tables;
use crate::utils::div::{div_dyn, div_rem_dyn, division_preflight, dyn_div_alg_at, dyn_div_has_choice, dyn_div_small, DivEq};
use crate::utils::mul::{dyn_dispatch_at, mul_dyn, school_at, MulEq};
use crate::utils::series::{series_eval_dyn, Bbp, Engel, Hyper, Series, SeriesNode};
use std::hint::black_box;
use std::rc::Rc;

/// (dispatcher, its table, whether the table's rows stand for kinds of call).
pub const TABLES: [(&str, &str, bool); 4] = [
    ("mul_dyn", "mul.kara", false),
    ("div_dyn", "div.dyn_small", false),
    ("div_rem_dyn", "div.dyn_rem_small", false),
    ("series_eval_dyn", "series.small", true),
];

/// A share this far past the rule's is a breach; up to it, an index at the line.
const BREACH: f64 = 1.25;

// A dispatcher's place in `TABLES`.
fn who(name: &str) -> usize {
    TABLES.iter().position(|t| t.0 == name).expect("a dispatcher of TABLES")
}

// One index where a dispatcher reads its equations: probes for the entry, for the harness's own
// work inside it (if any), and for the decision at explicit knobs, in that order.
struct Index {
    who: usize,
    name: String,
    harness: bool,
}

// A series index, where the live dispatch reads the equations for it (the table has no pick).
fn series_index<C: Series + 'static>(name: String, ctx: C, w: usize, rows: &mut Vec<Index>, probes: &mut Vec<Probe<'static>>) {
    let Some((zeros, len, (a, b))) = framed(&ctx, w) else { return };
    let node = Rc::new(ctx.dyn_ctx());
    if <C::Dyn as SeriesNode>::small(&node, a, b, len, zeros).is_some() {
        return;
    }
    let (ctx, mut out) = (Rc::new(ctx), vec![0u64; len]);
    probes.push(Probe::new("entry", move || {
        black_box(series_eval_dyn(&*ctx, &mut out));
    }));
    probes.push(decide_probe::<C>(node, (a, b), (len, zeros)));
    rows.push(Index { who: who("series_eval_dyn"), name: format!("{name} ({} terms)", b - a), harness: false });
}

// The indices tried, over each dispatcher's small sizes, kept where the live dispatch reads its equations.
fn indices(pass: usize, rows: &mut Vec<Index>, probes: &mut Vec<Probe<'static>>) {
    let (mul, div, kara, leaf) = (MulEq::read(), DivEq::read(), k::mul::kara(), k::div::bz());
    // Products past the schoolbook table: Karatsuba's equation against the FFT's.
    for s in [8usize, 12, 16, 24, 32, 40, 46, 52, 60, 70, 82, 96, 120, 160, 240, 400] {
        for m in [1.0f64, 1.5, 2.5, 4.0] {
            let l = (s as f64 * m) as usize;
            if school_at(kara, l, s) {
                continue;
            }
            let (a, b, mut o) = (pad(l, 3, pass), pad(s, 5, pass), pad(l + s - 1, 7, pass));
            probes.push(Probe::new("entry", move || {
                black_box(mul_dyn(black_box(&a), black_box(&b), &mut o));
            }));
            probes.push(Probe::new("decide", move || {
                black_box(dyn_dispatch_at(black_box(l), black_box(s), kara, &mul));
            }));
            rows.push(Index { who: who("mul_dyn"), name: format!("{l} x {s}"), harness: false });
        }
    }
    // Quotients, alone and with the remainder.
    for (entry, rem, table) in [("div_dyn", false, k::div::dyn_small()), ("div_rem_dyn", true, k::div::dyn_rem_small())] {
        for d in [1usize, 2, 4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 200, 400, 1000] {
            for q in [2usize, 4, 8, 9, 12, 16, 24, 40, 100, 300] {
                if !dyn_div_has_choice(q, d, leaf) || dyn_div_small(table, q, d, leaf).is_some() {
                    continue;
                }
                let (n0, mut n, dv, mut quo) = (numerator(q, d, pass), numerator(q, d, pass), placed_divisor(d, pass), pad(q - 1, 7, pass));
                debug_assert!(division_preflight(&n, &dv, &mut quo).is_some());
                probes.push(Probe::new("entry", move || {
                    if rem {
                        n.copy_from_slice(&n0);
                        black_box(div_rem_dyn(&mut n, black_box(&dv), &mut quo));
                    } else {
                        black_box(div_dyn(black_box(&n), black_box(&dv), &mut quo));
                    }
                }));
                if rem {
                    probes.push(copy_probe(q, d, pass));
                }
                probes.push(Probe::new("decide", move || {
                    black_box(dyn_div_alg_at(black_box(q), black_box(d), rem, &div, &mul, kara));
                }));
                rows.push(Index { who: who(entry), name: format!("q {q} d {d}"), harness: rem });
            }
        }
    }
    // Series of every kind, short to long.
    for w in [2usize, 4, 16, 64, 200, 600] {
        for t in [2u64, 4, 16, 64, 300, 1500] {
            let bits = 64 * w as u64;
            series_index(format!("bbp odd w{w} t{t}"), Bbp::<Odd>::new(Odd::decay(bits, t)), w, rows, probes);
            series_index(format!("engel fact w{w} t{t}"), Engel::<Fact>::new(Fact::decay(bits, t)), w, rows, probes);
            series_index(format!("hyper den32 w{w} t{t} x0"), Hyper::<Den<32>>::new(&[1], <Den<32> as Synth>::decay(bits, t)).expect("a shift of 1 or more"), w, rows, probes);
            for xb in [40u64, 100, 500, 3000] {
                let x = x_static(xb, 7 * xb + w as u64);
                series_index(format!("hyper geo8 w{w} t{t} x{xb}"), Hyper::<Geo<8>>::new(x, xb + <Geo<8> as Synth>::decay(bits, t)).expect("x 2^-shift <= 1/2"), w, rows, probes);
            }
        }
    }
}

/// A dispatcher's verdict: the indices where it reads its equations among those tried, the
/// decision's mean share of the call there, and the indices past the rule's share, worst first.
pub struct Verdict {
    pub who: &'static str,
    pub table: &'static str,
    pub read: usize,
    pub mean: f64,
    pub past: Vec<(String, f64)>,
    /// The indices past the share are the table's own choice (its rows stand for kinds, and it is measured).
    pub chosen: bool,
}

impl Verdict {
    /// Whether an index is past the rule by more than its measurement allows.
    pub fn breached(&self) -> bool {
        !self.chosen && self.past.first().is_some_and(|p| p.1 > BREACH * tables::share())
    }
}

/// Times the check: on the indices where each dispatcher reads its equations under the values
/// in force, the decision against the entry's call.
pub fn check(level: Level) -> Vec<Verdict> {
    let mut rows: Vec<Index> = Vec::new();
    let timed = time_fit(level, |pass| {
        rows.clear();
        let mut probes = Vec::new();
        indices(pass, &mut rows, &mut probes);
        probes
    });
    let mut shares: Vec<Vec<(String, f64)>> = vec![Vec::new(); TABLES.len()];
    let mut at = 0;
    for r in &rows {
        let harness = if r.harness { timed.ns[at + 1] } else { 0.0 };
        let (entry, decide) = (timed.ns[at], timed.ns[at + 1 + r.harness as usize]);
        at += 2 + r.harness as usize;
        // The call the decision is a share of: the entry's, less the harness's own work.
        shares[r.who].push((r.name.clone(), decide / less_harness(entry, harness)));
    }
    // A table that names something: more than the default's one change point.
    let measured = |table: &str| matches!(crate::tuning::get(table), Some(crate::tuning::Value::Table(t)) if t.len() > 3);
    TABLES
        .iter()
        .zip(shares)
        .map(|(&(who, table, kinds), mut s)| {
            s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mean = s.iter().map(|x| x.1).sum::<f64>() / s.len().max(1) as f64;
            Verdict { who, table, read: s.len(), mean, past: s.into_iter().filter(|x| x.1 > tables::share()).collect(), chosen: kinds && measured(table) }
        })
        .collect()
}

/// A verdict's line.
pub fn line(v: &Verdict) -> String {
    let worst = v.past.iter().take(3).map(|(name, share)| format!("{name}: {:.1}%", 100.0 * share)).collect::<Vec<_>>().join(", ");
    let verdict = if v.past.is_empty() {
        format!("within the rule's {:.0}% at every one", 100.0 * tables::share())
    } else if v.chosen {
        format!(
            "past the rule's {:.0}% at {} ({worst}), by {}'s own choice: its rows stand for kinds of series, and where no one algorithm serves a kind the table hands the index to the equations",
            100.0 * tables::share(),
            v.past.len(),
            v.table
        )
    } else if v.breached() {
        format!("PAST the rule's {:.0}% at {} ({worst}): those indices belong in {}", 100.0 * tables::share(), v.past.len(), v.table)
    } else {
        format!("at the rule's {:.0}% line at {} ({worst})", 100.0 * tables::share(), v.past.len())
    };
    format!("{}: reads its equations at {} of the indices tried, the decision {:.1}% of the call on average; {verdict}", v.who, v.read, 100.0 * v.mean)
}

/// `cargo tune dispatch`: the check, a line a dispatcher; false if any is past the rule.
pub fn report(level: Level) -> bool {
    println!("what a dispatcher's decision costs against its call, where it reads its equations ({}):", level.name());
    let verdicts = check(level);
    for v in &verdicts {
        println!("  {}", line(v));
    }
    !verdicts.iter().any(Verdict::breached)
}
