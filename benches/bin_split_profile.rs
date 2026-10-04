use big_bits::utils::{
    bin_split::{
        bin_split, BBPSeries, DynHyperCtx, DynNodeBBP, DynNodeEngel, DynNodeHyper, EngelSeries,
        HyperSeries, Node, Shift,
    },
    LN2_TERM_CUTOFF,
};
use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

const CUTOFF: u64 = LN2_TERM_CUTOFF;

// The four ln(2) series, sum 1 / (q(n) 16^n).
struct Ln2A;
impl BBPSeries for Ln2A {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
}
struct Ln2B;
impl BBPSeries for Ln2B {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        2 * n + 1
    }
}
struct Ln2C;
impl BBPSeries for Ln2C {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        4 * n + 1
    }
}
struct Ln2D;
impl BBPSeries for Ln2D {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        4 * n + 3
    }
}

// e - 2 = sum_{n >= 2} 1 / n!
struct EMinus2;
impl EngelSeries for EMinus2 {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
}

// Ratio (2n + 1) / (2n + 4) * x / 2^shift per term.
struct Hyp;
impl HyperSeries for Hyp {
    fn p(n: u64) -> u64 {
        n + 1
    }
    fn q(n: u64) -> u64 {
        2 * n + 2
    }
    fn r(n: u64) -> u64 {
        2 * n + 1
    }
}

#[derive(Default, Clone, Copy)]
struct Times {
    total: f64,
    leaves: f64,
    finalize: f64,
}

// Time only the leaves of the same tree bin_split builds.
fn leaf_time<N: Node>(ctx: &N::Ctx, a: u64, b: u64, last: bool) -> Duration {
    if b - a < CUTOFF {
        let s = Instant::now();
        black_box(N::leaf(ctx, a, b, last));
        return s.elapsed();
    }
    let m = (a + b) / 2;
    leaf_time::<N>(ctx, a, m, false) + leaf_time::<N>(ctx, m, b, last)
}

fn profile<N: Node + Send>(ctx: &N::Ctx, a: u64, b: u64, limbs: usize, reps: usize) -> Times
where
    N::Ctx: Sync,
{
    let mut out = vec![0; limbs];
    let mut t = Times {
        total: f64::MAX,
        leaves: f64::MAX,
        finalize: f64::MAX,
    };
    for _ in 0..reps {
        let s = Instant::now();
        let node = bin_split::<N, CUTOFF>(ctx, a, b);
        let built = s.elapsed().as_secs_f64();
        let f = Instant::now();
        node.finalize(ctx, &mut out);
        let fin = f.elapsed().as_secs_f64();
        black_box(&out);
        t.total = t.total.min(built + fin);
        t.finalize = t.finalize.min(fin);
        t.leaves = t.leaves.min(leaf_time::<N>(ctx, a, b, true).as_secs_f64());
    }
    t
}

fn add(a: Times, b: Times) -> Times {
    Times {
        total: a.total + b.total,
        leaves: a.leaves + b.leaves,
        finalize: a.finalize + b.finalize,
    }
}

// Leaves are timed serially, so the merge share is only meaningful on one thread.
fn report(name: &str, limbs: usize, t: Times) {
    let merges = if rayon::current_num_threads() == 1 {
        let m = t.total - t.leaves - t.finalize;
        format!("{:>10.3} ms ({:>4.1}%)", m * 1e3, 100.0 * m / t.total)
    } else {
        "n/a (parallel)".to_string()
    };
    println!(
        "{name:<12} {limbs:>6} limbs  total {:>10.3} ms  leaves(serial) {:>9.3} ms  merges {merges}  finalize {:>8.3} ms",
        t.total * 1e3,
        t.leaves * 1e3,
        t.finalize * 1e3,
    );
}

// Terms of the Engel series until sum log2(n) covers bits.
fn engel_terms(bits: f64) -> u64 {
    let (mut n, mut acc) = (2u64, 0.0);
    while acc < bits {
        acc += (n as f64).log2();
        n += 1;
    }
    n
}

fn main() {
    let sizes: Vec<usize> = env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let sizes = if sizes.is_empty() {
        vec![256, 1024, 4096, 16384]
    } else {
        sizes
    };
    println!(
        "rayon threads: {}, leaf cutoff: {CUTOFF}",
        rayon::current_num_threads()
    );

    for &limbs in &sizes {
        let reps = (20_000 / limbs).clamp(1, 20);
        let bits = 64.0 * (limbs + 1) as f64;

        // ln(2): four BBP trees at 4 bits per term.
        let ctx = Shift(4);
        let n = 1 + 16 * (limbs as u64 + 1) + 4;
        let ln2 = [
            profile::<DynNodeBBP<Ln2A>>(&ctx, 1, n, limbs, reps),
            profile::<DynNodeBBP<Ln2B>>(&ctx, 1, n, limbs, reps),
            profile::<DynNodeBBP<Ln2C>>(&ctx, 1, n, limbs, reps),
            profile::<DynNodeBBP<Ln2D>>(&ctx, 1, n, limbs, reps),
        ]
        .into_iter()
        .fold(Times::default(), add);
        report("ln2 (4xBBP)", limbs, ln2);

        let ctx = Shift(0);
        let n = engel_terms(bits);
        report(
            "engel e-2",
            limbs,
            profile::<DynNodeEngel<EMinus2>>(&ctx, 2, n, limbs, reps),
        );

        // About 8 bits per term in both hypergeometric cases.
        let terms = (bits / 8.0) as u64 + 4;
        let ctx = DynHyperCtx::new(&[1], 8);
        report(
            "hyper x=1",
            limbs,
            profile::<DynNodeHyper<Hyp>>(&ctx, 0, terms, limbs, reps),
        );
        let x = [0x9e37_79b9_7f4a_7c15, 0x0000_00ff_ffff_ffff];
        let ctx = DynHyperCtx::new(&x, 8 + 104);
        report(
            "hyper x=2L",
            limbs,
            profile::<DynNodeHyper<Hyp>>(&ctx, 0, terms, limbs, reps),
        );
    }
}
