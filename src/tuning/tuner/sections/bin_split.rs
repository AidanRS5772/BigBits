//! Section `bin_split`: the leaf cutoffs and the two parallel gates. Each acts at every level of a
//! tree, so each is a parameter sweep over workloads (not a two-sided crossover): the trees of
//! docs/bin_split_cutoff_tuning.md's workloads, built through `bin_split_tree_at` with explicit
//! cutoffs (the two gates end to end through exp_dyn and ln_dyn_at instead: see below). The dynamic set is swept twice, as its knobs are coupled. Static trees at N = 1100 only
//! (approximate, like every static value).

use crate::tuning::resolve::Value;
use crate::tuning::tuner::check_items::EXP_FRACTION;
use crate::tuning::tuner::measure::{pad, time_fixed, Level, Probe};
use crate::tuning::tuner::out::Tuned;
use crate::tuning::tuner::sweep::{int, ints, keep, param_sweep, param_sweep_lowest, with_knob};
use crate::utils::consts::consts::Ln2S;
use crate::utils::series::{
    bin_split_tree_at, DynHyperCtx, DynNodeBBP, DynNodeEngel, DynNodeHyper, Node, Shift, SplitCutoffs, StaticHyperCtx,
    StaticNodeBBP, StaticNodeEngel, StaticNodeHyper,
};
use crate::utils::trans::{exp_dyn, ln_dyn_at, ATanh, Exp};
use std::hint::black_box;

const N: usize = 1100;

// The fewest terms n with n * bits_per_term + (fact ? log2 n! : 0) >= 64 * limbs.
fn terms(limbs: usize, bits_per_term: f64, fact: bool) -> u64 {
    let (want, mut got, mut n) = (64.0 * limbs as f64, 0.0, 0u64);
    while got < want {
        n += 1;
        got += bits_per_term + if fact { (n as f64).log2() } else { 0.0 };
    }
    n.max(2)
}

// A tree of `Nd` over [a, a + t) under `cut`.
fn tree<Nd>(ctx: Nd::Ctx, a: u64, t: u64, cut: impl Fn() -> SplitCutoffs + 'static) -> Probe<'static>
where
    Nd: Node + Send + 'static,
    Nd::Ctx: Sync + 'static,
{
    Probe::new("tree", move || {
        black_box(bin_split_tree_at::<Nd>(&ctx, a, a + t, true, cut()));
    })
}

// An x of `limbs` limbs with its top bit set.
fn x_of(limbs: usize) -> Vec<u64> {
    let mut x: Vec<u64> = (0..limbs as u64).map(|i| 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(i + 1) | 1).collect();
    x[limbs - 1] |= 1 << 63;
    x
}

pub fn tune(level: Level) -> Vec<Tuned> {
    let saved = crate::tuning::snapshot();
    let mut out: Vec<Tuned> = Vec::new();
    // Each swept value is kept live for the sweeps after it (`keep`), and a later round's value
    // replaces an earlier one's.
    // Fewer candidates and workloads than the old tuning doc's tables (the user's decision,
    // 2026-11-11); every default is among the candidates, so the tie band can keep it.
    let leaves = ints(&[32, 48, 64, 96, 128, 192, 256]);
    let wide_leaves = ints(&[64, 128, 192, 256, 512]);
    // Every tree's cutoffs are the live ones with one field replaced.

    // The dynamic knobs are coupled: with the split gate low, large leaves win only by avoiding
    // parallel splits that cost more than they save, and once the gate is raised small leaves are
    // right (a single pass from the defaults chose leaves of 384-512 losing 8-10%; the second
    // finds 48-96 losing under 3%). So the set is swept twice, each sweep under the values of the
    // ones before it.
    type Ln2 = Ln2S<6, 1>;
    let floors = ints(&[4, 8, 16, 32, 64]);
    for _round in 0..2 {
        // The callers: whole exp_dyn (0-2) and ln_dyn_at (3-5) calls on the tune's pool. Every
        // dynamic knob here is swept on them as well as on its own trees, or through them alone
        // (the gates): a tree's best cutoff depends on the series, and the trees below are
        // exp's and ln 2's. Swept on those alone, the Hyper leaf read 192 limbs on the M1 where
        // ln's atanh chunks want about 48, and ln_dyn at 300 and 500 limbs ran 7% and 11%
        // slower for it (found by the tune's own end-to-end gate, 2026-11-10).
        let caller = |w: usize| -> Probe<'static> {
            let n = [256usize, 1024, 4096, 256, 512, 1024][w];
            let (x, mut o) = (pad(n, n as u64 * 7 + 1, 0), pad(n, 3, 0));
            // ln's series as ln_dyn runs it: under the live square-root scale.
            let a = crate::tuning::k::ln::dyn_sqrt_a();
            if w < 3 {
                Probe::new("exp_dyn", move || {
                    black_box(exp_dyn(black_box(&x), EXP_FRACTION, &mut o).expect("X < 1"));
                })
            } else {
                Probe::new("ln_dyn_at", move || {
                    black_box(ln_dyn_at(black_box(&x), 1 - n as i64, &mut o, a).expect("X > 0"));
                })
            }
        };
        // A leaf knob's sweep: its trees, then the callers that build such trees.
        let through = |path: &'static str, trees: usize, callers: &[usize], cands: &[Value], tree_at: &dyn Fn(usize, u64) -> Probe<'static>| -> Tuned {
            let loads: Vec<usize> = (0..trees + callers.len()).collect();
            param_sweep(level, path, &loads, cands, &|w, c, _| if w < trees { tree_at(w, int(c)) } else { with_knob(path, c, caller(callers[w - trees])) })
        };

        // Dynamic BBP: ln 2's first series (6 bits a term) at 256-4096 limbs, and atanh's at 64
        // bits a term at 1024 limbs; and ln's calls.
        let bbp = |w: usize, cut: Box<dyn Fn() -> SplitCutoffs>| -> Probe<'static> {
            match w {
                0..=2 => tree::<DynNodeBBP<Ln2>>(Shift(6), 0, terms([256, 1024, 4096][w], 6.0, false), cut),
                _ => tree::<DynNodeBBP<ATanh>>(Shift(64), 1, terms(1024, 64.0, false), cut),
            }
        };
        let t = through("bin_split.dyn_bbp_leaf_limbs", 4, &[3, 4, 5], &leaves, &|w, c| bbp(w, Box::new(move || SplitCutoffs { leaf_limbs: c, ..SplitCutoffs::of::<DynNodeBBP<Ln2>>() })));
        keep("bin_split", &mut out, t);

        // Dynamic Engel: 1/n! at shifts 1, 8 and 64, at 256 and 1024 limbs; and exp's calls
        // (its top bit's series).
        let es: Vec<(u64, usize)> = vec![(1, 1024), (8, 1024), (64, 256)];
        let t = through("bin_split.dyn_engel_leaf_limbs", es.len(), &[0, 1, 2], &leaves, &|w, c| {
            let (k, limbs) = es[w];
            tree::<DynNodeEngel<Exp>>(Shift(k), 1, terms(limbs, k as f64, true), move || SplitCutoffs { leaf_limbs: c, ..SplitCutoffs::of::<DynNodeEngel<Exp>>() })
        });
        keep("bin_split", &mut out, t);

        // Dynamic Hyper: exp's chunk series x^n / n! with an x of 1, 4 and 16 limbs at shift
        // 2 bits(x), at 256 and 1024 limbs. Leaf limbs first, then the term floor under it.
        let hyper_shapes: Vec<(usize, usize)> = vec![(1, 1024), (4, 1024), (16, 256)];
        let hyper = |(xl, limbs): (usize, usize), cut: Box<dyn Fn() -> SplitCutoffs>| {
            let bits = 64 * xl as u64;
            tree::<DynNodeHyper<Exp>>(DynHyperCtx::new(&x_of(xl), 2 * bits), 1, terms(limbs, bits as f64, true), cut)
        };
        // Both exp's and ln's chunks are Hyper series, so every caller is in these two.
        let hs = &hyper_shapes;
        let t = through("bin_split.dyn_hyper_leaf_limbs", hs.len(), &[0, 1, 2, 3, 4, 5], &leaves, &|w, c| {
            hyper(hs[w], Box::new(move || SplitCutoffs { leaf_limbs: c, ..SplitCutoffs::of::<DynNodeHyper<Exp>>() }))
        });
        keep("bin_split", &mut out, t);
        let t = through("bin_split.dyn_hyper_leaf_terms", hs.len(), &[0, 1, 2, 3, 4, 5], &floors, &|w, c| {
            hyper(hs[w], Box::new(move || SplitCutoffs { leaf_terms: c, ..SplitCutoffs::of::<DynNodeHyper<Exp>>() }))
        });
        keep("bin_split", &mut out, t);

        // The gates, under the leaves just found, swept end to end through their callers: whole
        // exp_dyn and ln_dyn_at calls on the tune's pool, with the knob installed. Isolated trees
        // mislead here: started from the main thread a parallel split has to wake the pool, and
        // on them the split gate read 1024, which made exp_dyn and ln_dyn 3-5% slower at 300 and
        // 1040 limbs. Inside exp and ln the trees are built by chunks already on pool workers.
        let gates = ints(&[64, 128, 256, 512, 1024, 1 << 40]);
        for path in ["bin_split.dyn_par", "bin_split.dyn_merge_par"] {
            let t = param_sweep(level, path, &[0, 1, 2, 3, 4, 5], &gates, &|w, c, _| with_knob(path, c, caller(w)));
            keep("bin_split", &mut out, t);
        }

        // The split's work gate applies only off the pool, where a split has to wake it. Its
        // workloads are the trees a caller's own thread starts, with a clear loss on each side
        // of the right value. Few wide terms, where the wake costs several times the tree:
        // Engel trees at 1024 limbs with about 2, 8, 32 and 125 terms, and 2 and 64 at 4096
        // (2 thousand to 260 thousand limbs times terms). Many narrow terms, which gain most
        // of a factor of two from the split: atanh's BBP trees as ln's first chunk builds them,
        // 0.7 and 2 million at 256 limbs and 8 million at 1024. And whole exp_dyn and
        // ln_dyn_at calls, where the pool is already busy with chunks while such a tree runs.
        // Its history: swept on Engel trees alone it read 2^24, the top candidate (a serial
        // 7000-term Engel tree loses only 2%), and with the callers added 2^22 or 2^24 by
        // noise; in the tunes that kept those, ln_dyn at 300 to 1040 limbs was listed 6-9% slower.
        let work = ints(&[0, 1 << 14, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 24]);
        let few: Vec<(u64, usize)> = vec![(32768, 1024), (8192, 1024), (2048, 1024), (512, 1024), (131072, 4096), (4096, 4096)];
        let many: Vec<(u64, usize)> = vec![(6, 256), (2, 256), (8, 1024)];
        let loads: Vec<usize> = (0..few.len() + many.len() + 6).collect();
        // The lowest value that does the job: past it the loss is flat, and a gate set higher
        // than it has to be would keep off-pool trees no workload here covers from splitting.
        let t = param_sweep_lowest(level, "bin_split.dyn_par_work", &loads, &work, &|w, c, _| {
            let par_work = int(c);
            if w < few.len() {
                let (k, limbs) = few[w];
                tree::<DynNodeEngel<Exp>>(Shift(k), 1, terms(limbs, k as f64, true), move || SplitCutoffs { par_work, ..SplitCutoffs::of::<DynNodeEngel<Exp>>() })
            } else if w < few.len() + many.len() {
                let (k, limbs) = many[w - few.len()];
                tree::<DynNodeBBP<ATanh>>(Shift(k), 1, terms(limbs, k as f64, false), move || SplitCutoffs { par_work, ..SplitCutoffs::of::<DynNodeBBP<ATanh>>() })
            } else {
                with_knob("bin_split.dyn_par_work", c, caller(w - few.len() - many.len()))
            }
        });
        keep("bin_split", &mut out, t);
    }

    // One pool wake, which the series cost model charges a tree that splits off the pool: a
    // 2-term tree at 1024 limbs started here, with the split and without.
    if rayon::current_num_threads() > 1 {
        let two = |par_limbs: u64| {
            tree::<DynNodeEngel<Exp>>(Shift(32768), 1, 2, move || SplitCutoffs { par_limbs, par_work: 0, ..SplitCutoffs::of::<DynNodeEngel<Exp>>() })
        };
        // The least of two measurements of each. A wake's time moves from one to the next by
        // more than a level's spread bound, so these two probes don't go to the "samples" line.
        let mut probes = [two(0), two(u64::MAX)];
        let (a, b) = (time_fixed(&mut probes, level), time_fixed(&mut probes, level));
        let t = [a[0].min(b[0]), a[1].min(b[1])];
        out.push(Tuned {
            key: "dyn_wake_ns",
            value: Value::Float((t[0] - t[1]).max(0.0)),
            evidence: format!("measured: a 2-term Engel tree at 1024 limbs started off the pool takes {:.0} ns split and {:.0} ns serial", t[0], t[1]),
        });
    }

    // Static trees at N = 1100, about 1000 limbs of terms: approximate values.
    let t = param_sweep(level, "bin_split.static_bbp_leaf_limbs", &[1000], &wide_leaves, &|limbs, c, _| {
        let c = int(c);
        tree::<StaticNodeBBP<N, Ln2>>(Shift(6), 0, terms(limbs, 6.0, false), move || SplitCutoffs { leaf_limbs: c, ..SplitCutoffs::of::<StaticNodeBBP<N, Ln2>>() })
    });
    keep("bin_split", &mut out, t);
    let t = param_sweep(level, "bin_split.static_engel_leaf_limbs", &[1000], &wide_leaves, &|limbs, c, _| {
        let c = int(c);
        tree::<StaticNodeEngel<N, Exp>>(Shift(8), 1, terms(limbs, 8.0, true), move || SplitCutoffs { leaf_limbs: c, ..SplitCutoffs::of::<StaticNodeEngel<N, Exp>>() })
    });
    keep("bin_split", &mut out, t);
    let static_hyper = |limbs: usize, cut: Box<dyn Fn() -> SplitCutoffs>| {
        tree::<StaticNodeHyper<N, Exp>>(StaticHyperCtx::<N>::new(&x_of(4), 512), 1, terms(limbs, 256.0, true), cut)
    };
    // bin_split.static_hyper_leaf_limbs is not swept: on this one workload it read 768 (from 64),
    // which made pi_static (a wide-coefficient series) 1.30x slower. It keeps its default.
    let t = param_sweep(level, "bin_split.static_hyper_leaf_terms", &[1000], &floors, &|limbs, c, _| {
        let c = int(c);
        static_hyper(limbs, Box::new(move || SplitCutoffs { leaf_terms: c, ..SplitCutoffs::of::<StaticNodeHyper<N, Exp>>() }))
    });
    keep("bin_split", &mut out, t);
    crate::tuning::restore(&saved);
    out
}
