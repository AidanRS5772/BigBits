// Tunes binary splitting's leaf and parallel cutoffs.
//
//   cargo bench --bench bin_split_cutoffs -- <family> [--limbs 8,16,max] [--terms 1,4]
//       [--par 256,max] [--cands 0/31/max,64/1/max] [--rounds 9] [--sizes 256,1024]
//       [--a 0,0.5,1] [--j 0,16,24] [--tail 0,1,3,7,15] [--t 0,1,3,6]
//
// Families: bbp, engel, hyper, sbbp, sengel, shyper (leaf sweeps, serial), par (parallel
// cutoff, all threads, default leaves), e2e (old term cutoffs against the defaults), ln
// (ln_dyn end to end against its square-root scale A, all threads) and exp (exp_dyn end to
// end against its square count j, with ln_dyn on the same X for scale, all threads) and agm
// (agm_dyn_at and agm_static_at on ln_agm's inputs against the tail's term cap, with the roots
// each cap took and ln_dyn's AGM on the same X for scale) and lnagm (ln_agm_dyn_at::<T>
// and ln_agm_static_at::<N, T> end to end against theta's term count T, with the series ln_dyn_at
// on the same X for scale).
// Each candidate times a tree build only: finalize doesn't depend on the tree's shape.
// Candidates alternate within each round and each one reports its median over rounds,
// relative to the workload's best; the summary is the geometric mean over workloads.

use big_bits::utils::{
    mul::{mul_dyn, sqr_dyn},
    series::{
        bin_split_tree_at, BBPSeries, DynHyperCtx, DynNodeBBP, DynNodeEngel, DynNodeHyper, Engel,
        HyperSeries, Node, Series, Shift, SplitCutoffs, StaticHyperCtx, StaticNodeBBP,
        StaticNodeEngel, StaticNodeHyper,
    },
    trans::{
        agm_dyn_at, agm_static_at, exp_dyn_at, exp_dyn_entry, ln_agm_dyn_at, ln_agm_static_at,
        ln_dyn, ln_dyn_at, theta_dyn, Exp,
    },
};
use big_bits::utils::{DYN_LN_AGM_TERMS, DYN_LN_SQRT_A};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cell::RefCell;
use std::{env, hint::black_box, time::Instant};

// e^y - 1's term count for y <= 2^-s at w limbs, at least one.
fn exp_terms(s: u64, w: usize) -> u64 {
    (Engel::<Exp>::new(s).range(64 * w as u64).1 - 1).max(1)
}

// START, and a tail bound of 0: these series are only summed over explicit ranges.
macro_rules! no_tail {
    ($start:expr) => {
        const START: u64 = $start;
        fn tail_bits(_: u64, _: i64) -> u64 {
            0
        }
    };
}

// The four ln(2) series, sum 1 / (q(n) 16^n).
struct Ln2A;
impl BBPSeries for Ln2A {
    no_tail!(1);
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
}
struct Ln2B;
impl BBPSeries for Ln2B {
    no_tail!(0);
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        2 * n + 1
    }
}
struct Ln2C;
impl BBPSeries for Ln2C {
    no_tail!(0);
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        4 * n + 1
    }
}
struct Ln2D;
impl BBPSeries for Ln2D {
    no_tail!(0);
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        4 * n + 3
    }
}

// Ratio (2n + 1) / (2n + 4) per term.
struct Hyp;
impl HyperSeries for Hyp {
    no_tail!(0);
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

fn ln2_terms(limbs: usize) -> u64 {
    1 + 16 * (limbs as u64 + 1) + 4
}

// A random number of exactly bits bits.
fn rand_x(bits: u64, rng: &mut StdRng) -> Vec<u64> {
    let len = bits.div_ceil(64) as usize;
    let mut x: Vec<u64> = (0..len).map(|_| rng.gen()).collect();
    let top = bits - 64 * (len as u64 - 1);
    x[len - 1] = (x[len - 1] >> (64 - top)) | 1 << (top - 1);
    x
}

// exp_dyn_entry's chunk schedule for a normalized w-limb x: chunk i covers bits
// [s, s + width) with width = 2^(i + 1), so its series has shift s + width and its
// terms shrink by about s bits each. Returns (x bits, shift, per-term bits).
fn exp_chunks(w: usize) -> Vec<(u64, u64, u64)> {
    let end = 64 * w as u64;
    let (mut s, mut sz, mut out) = (1u64, 1u64, Vec::new());
    while s < end {
        let width = (2 * sz).min(end - s);
        out.push((width, s + width, s));
        sz *= 2;
        s += sz;
    }
    out
}

type Run = Box<dyn Fn(SplitCutoffs)>;

struct Workload {
    name: String,
    run: Run,
}

fn workload(name: String, run: impl Fn(SplitCutoffs) + 'static) -> Workload {
    Workload {
        name,
        run: Box::new(run),
    }
}

fn tree<N>(ctx: &N::Ctx, a: u64, b: u64, cut: SplitCutoffs)
where
    N: Node + Send,
    N::Ctx: Sync,
{
    black_box(bin_split_tree_at::<N>(ctx, a, b, true, cut));
}

fn ln2_dyn_tree(n: u64, cut: SplitCutoffs) {
    let ctx = Shift(4);
    tree::<DynNodeBBP<Ln2A>>(&ctx, 1, n, cut);
    tree::<DynNodeBBP<Ln2B>>(&ctx, 1, n, cut);
    tree::<DynNodeBBP<Ln2C>>(&ctx, 1, n, cut);
    tree::<DynNodeBBP<Ln2D>>(&ctx, 1, n, cut);
}

fn ln2_static_tree<const N: usize>(n: u64, cut: SplitCutoffs) {
    let ctx = Shift(4);
    tree::<StaticNodeBBP<N, Ln2A>>(&ctx, 1, n, cut);
    tree::<StaticNodeBBP<N, Ln2B>>(&ctx, 1, n, cut);
    tree::<StaticNodeBBP<N, Ln2C>>(&ctx, 1, n, cut);
    tree::<StaticNodeBBP<N, Ln2D>>(&ctx, 1, n, cut);
}

fn sizes_or(args: &Args, default: &[usize]) -> Vec<usize> {
    args.sizes.clone().unwrap_or_else(|| default.to_vec())
}

fn bbp_workloads(args: &Args) -> Vec<Workload> {
    let mut v = Vec::new();
    for l in sizes_or(args, &[64, 256, 1024, 4096, 16384]) {
        let n = ln2_terms(l);
        v.push(workload(format!("ln2 {l}"), move |c| ln2_dyn_tree(n, c)));
    }
    // Sparse terms: 1 / ((4n + 3) 2^(k n)).
    for k in [64u64, 256] {
        for l in sizes_or(args, &[256, 1024, 4096]) {
            let n = 64 * (l as u64 + 1) / k + 4;
            v.push(workload(format!("bbp k={k} {l}"), move |c| {
                tree::<DynNodeBBP<Ln2D>>(&Shift(k), 1, n, c)
            }));
        }
    }
    v
}

fn engel_workloads(args: &Args) -> Vec<Workload> {
    let mut v = Vec::new();
    for k in [0u64, 1, 8, 64, 512] {
        for l in sizes_or(args, &[256, 1024, 4096, 16384]) {
            // k = 0 is e - 2; k > 0 is exp's first chunk at shift k.
            let n = exp_terms(k.max(1), l) + 1;
            if n < 8 {
                continue;
            }
            v.push(workload(format!("engel k={k} {l}"), move |c| {
                tree::<DynNodeEngel<Exp>>(&Shift(k), 1, n, c)
            }));
        }
    }
    v
}

fn hyper_workloads(args: &Args) -> Vec<Workload> {
    let mut v = Vec::new();
    let mut rng = StdRng::seed_from_u64(7);
    for l in sizes_or(args, &[256, 1024, 4096]) {
        let n = 64 * (l as u64 + 1) / 8 + 4;
        v.push(workload(format!("hyper x=1 {l}"), move |c| {
            tree::<DynNodeHyper<Hyp>>(&DynHyperCtx::new(&[1], 8), 0, n, c)
        }));
    }
    for xb in [8u64, 32, 128, 512, 2048, 8192] {
        for l in sizes_or(args, &[256, 1024, 4096]) {
            let n = exp_terms(xb, l) + 1;
            if n < 8 {
                continue;
            }
            let ctx = DynHyperCtx::new(&rand_x(xb, &mut rng), 2 * xb);
            v.push(workload(format!("exp xb={xb} {l}"), move |c| {
                tree::<DynNodeHyper<Exp>>(&ctx, 1, n, c)
            }));
        }
    }
    v
}

fn sbbp_at<const N: usize>(v: &mut Vec<Workload>) {
    let n = ln2_terms(N - 4);
    v.push(workload(format!("ln2 N={N}"), move |c| ln2_static_tree::<N>(n, c)));
}

fn sengel_at<const N: usize>(v: &mut Vec<Workload>) {
    for k in [0u64, 1, 64] {
        let n = exp_terms(k.max(1), N - 3) + 1;
        v.push(workload(format!("engel k={k} N={N}"), move |c| {
            tree::<StaticNodeEngel<N, Exp>>(&Shift(k), 1, n, c)
        }));
    }
}

fn shyper_at<const N: usize>(v: &mut Vec<Workload>) {
    let mut rng = StdRng::seed_from_u64(11);
    let n = 64 * (N as u64 - 2) / 8 + 4;
    v.push(workload(format!("hyper x=1 N={N}"), move |c| {
        tree::<StaticNodeHyper<N, Hyp>>(&StaticHyperCtx::<N>::new(&[1], 8), 0, n, c)
    }));
    for xb in [32u64, 512] {
        let n = exp_terms(xb, N - 3) + 1;
        if n < 8 {
            continue;
        }
        let ctx = StaticHyperCtx::<N>::new(&rand_x(xb, &mut rng), 2 * xb);
        v.push(workload(format!("exp xb={xb} N={N}"), move |c| {
            tree::<StaticNodeHyper<N, Exp>>(&ctx, 1, n, c)
        }));
    }
}

fn static_workloads(family: &str) -> Vec<Workload> {
    let mut v = Vec::new();
    match family {
        "sbbp" => {
            sbbp_at::<64>(&mut v);
            sbbp_at::<256>(&mut v);
            sbbp_at::<1024>(&mut v);
            sbbp_at::<2048>(&mut v);
            sbbp_at::<4096>(&mut v);
        }
        "sengel" => {
            sengel_at::<64>(&mut v);
            sengel_at::<256>(&mut v);
            sengel_at::<1024>(&mut v);
        }
        _ => {
            shyper_at::<64>(&mut v);
            shyper_at::<256>(&mut v);
            shyper_at::<1024>(&mut v);
        }
    }
    v
}

// The parallel cutoff, on the workloads large enough to split in parallel.
fn par_workloads(args: &Args) -> Vec<Workload> {
    let mut v = Vec::new();
    let mut rng = StdRng::seed_from_u64(13);
    for l in sizes_or(args, &[1024, 4096, 16384]) {
        let n = ln2_terms(l);
        v.push(workload(format!("ln2 {l}"), move |c| {
            ln2_dyn_tree(n, with_leaves::<DynNodeBBP<Ln2A>>(c))
        }));
        let n = exp_terms(1, l) + 1;
        v.push(workload(format!("engel k=1 {l}"), move |c| {
            tree::<DynNodeEngel<Exp>>(&Shift(1), 1, n, with_leaves::<DynNodeEngel<Exp>>(c))
        }));
        for xb in [32u64, 512] {
            let n = exp_terms(xb, l) + 1;
            let ctx = DynHyperCtx::new(&rand_x(xb, &mut rng), 2 * xb);
            v.push(workload(format!("exp xb={xb} {l}"), move |c| {
                tree::<DynNodeHyper<Exp>>(&ctx, 1, n, with_leaves::<DynNodeHyper<Exp>>(c))
            }));
        }
    }
    v
}

// A parallel-cutoff candidate with N's default leaves.
fn with_leaves<N: Node>(c: SplitCutoffs) -> SplitCutoffs {
    SplitCutoffs {
        par_limbs: c.par_limbs,
        ..SplitCutoffs::of::<N>()
    }
}

struct Args {
    family: String,
    limbs: Vec<u64>,
    terms: Vec<u64>,
    par: Vec<u64>,
    // Explicit candidates, instead of every limbs x terms x par combination.
    cands: Option<Vec<SplitCutoffs>>,
    rounds: usize,
    sizes: Option<Vec<usize>>,
    // ln's square-root scales.
    a: Option<Vec<f64>>,
    // exp's square counts.
    j: Option<Vec<u64>>,
    // The AGM tail's term caps.
    tail: Option<Vec<u64>>,
    // ln_agm's theta term counts.
    t: Option<Vec<u64>>,
}

// "limbs/terms/par" candidates, comma separated.
fn parse_cands(s: &str) -> Vec<SplitCutoffs> {
    s.split(',')
        .map(|c| match parse_list(&c.replace('/', ","))[..] {
            [leaf_limbs, leaf_terms, par_limbs] => SplitCutoffs {
                leaf_limbs,
                leaf_terms,
                par_limbs,
            },
            _ => panic!("candidates are limbs/terms/par"),
        })
        .collect()
}

fn parse_list(s: &str) -> Vec<u64> {
    s.split(',')
        .map(|t| match t {
            "max" => u64::MAX,
            t => t.parse().expect("list of integers or max"),
        })
        .collect()
}

fn args() -> Args {
    let mut it = env::args().skip(1).filter(|a| a != "--bench");
    let mut a = Args {
        family: it.next().expect("family"),
        limbs: (1..=11).map(|i| 1 << i).collect(),
        terms: vec![1],
        par: vec![u64::MAX],
        cands: None,
        rounds: 9,
        sizes: None,
        a: None,
        j: None,
        tail: None,
        t: None,
    };
    while let Some(flag) = it.next() {
        let val = it.next().expect("flag value");
        match flag.as_str() {
            "--limbs" => a.limbs = parse_list(&val),
            "--terms" => a.terms = parse_list(&val),
            "--par" => a.par = parse_list(&val),
            "--cands" => a.cands = Some(parse_cands(&val)),
            "--rounds" => a.rounds = val.parse().unwrap(),
            "--sizes" => a.sizes = Some(parse_list(&val).iter().map(|&s| s as usize).collect()),
            "--a" => a.a = Some(val.split(',').map(|t| t.parse().expect("numbers")).collect()),
            "--j" => a.j = Some(parse_list(&val)),
            "--tail" => a.tail = Some(parse_list(&val)),
            "--t" => a.t = Some(parse_list(&val)),
            f => panic!("unknown flag {f}"),
        }
    }
    a
}

fn show(v: u64) -> String {
    if v == u64::MAX {
        "max".into()
    } else {
        v.to_string()
    }
}

fn label(c: &SplitCutoffs) -> String {
    format!("{}/{}/{}", show(c.leaf_limbs), show(c.leaf_terms), show(c.par_limbs))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// Median seconds per call of each candidate, alternating candidates each round.
fn measure<C: Copy>(run: &dyn Fn(C), cands: &[C], rounds: usize) -> Vec<f64> {
    // Calibrate on the middle candidate so each sample takes about 20 ms.
    let mid = cands[cands.len() / 2];
    run(mid);
    let s = Instant::now();
    run(mid);
    let reps = ((0.02 / s.elapsed().as_secs_f64()).ceil() as usize).clamp(1, 10_000);

    let mut samples = vec![Vec::with_capacity(rounds); cands.len()];
    for r in 0..rounds {
        for j in 0..cands.len() {
            let i = if r % 2 == 0 {
                (j + r) % cands.len()
            } else {
                (cands.len() - 1 - j + r) % cands.len()
            };
            let s = Instant::now();
            for _ in 0..reps {
                run(cands[i]);
            }
            samples[i].push(s.elapsed().as_secs_f64() / reps as f64);
        }
    }
    samples.iter_mut().map(|s| median(s)).collect()
}

fn sweep(ws: &[Workload], cands: &[SplitCutoffs], rounds: usize) {
    let mut log_sum = vec![0.0; cands.len()];
    let header: Vec<String> = cands.iter().map(label).collect();
    println!("{:<22} {:>10}  {}", "workload", "best ms", header.join(" "));
    for w in ws {
        let t = measure(&*w.run, cands, rounds);
        let best = t.iter().cloned().fold(f64::MAX, f64::min);
        let cells: Vec<String> = t
            .iter()
            .zip(&header)
            .map(|(&x, h)| format!("{:>w$.3}", x / best, w = h.len()))
            .collect();
        println!("{:<22} {:>10.3}  {}", w.name, best * 1e3, cells.join(" "));
        for (s, &x) in log_sum.iter_mut().zip(&t) {
            *s += (x / best).ln();
        }
    }
    let cells: Vec<String> = log_sum
        .iter()
        .zip(&header)
        .map(|(&s, h)| format!("{:>w$.3}", (s / ws.len() as f64).exp(), w = h.len()))
        .collect();
    println!("{:<22} {:>10}  {}", "geomean", "", cells.join(" "));
}

// The term cutoffs bin_split used before limbs, as the cutoffs that reproduce
// them: leaves of fewer than terms terms, whatever their size.
fn old(terms: u64) -> SplitCutoffs {
    SplitCutoffs {
        leaf_limbs: 0,
        leaf_terms: terms - 1,
        par_limbs: u64::MAX,
    }
}

fn serial<N: Node>() -> SplitCutoffs {
    SplitCutoffs {
        par_limbs: u64::MAX,
        ..SplitCutoffs::of::<N>()
    }
}

// Old term cutoffs (ln2 640, exp 32) against the defaults, serial, plus exp_dyn_entry
// itself with the defaults for scale.
fn e2e(args: &Args) {
    let mut rng = StdRng::seed_from_u64(17);
    println!("{:<22} {:>10} {:>10} {:>9}", "workload", "old ms", "new ms", "new/old");
    let row = |name: String, old_t: f64, new_t: f64| {
        println!(
            "{name:<22} {:>10.3} {:>10.3} {:>9.3}",
            old_t * 1e3,
            new_t * 1e3,
            new_t / old_t
        )
    };
    for l in sizes_or(args, &[64, 256, 1024, 4096, 16384]) {
        let n = ln2_terms(l);
        let w = workload(format!("ln2 {l}"), move |c| ln2_dyn_tree(n, c));
        let t = measure(&*w.run, &[old(640), serial::<DynNodeBBP<Ln2A>>()], args.rounds);
        row(w.name, t[0], t[1]);
    }
    for l in sizes_or(args, &[256, 1024, 4096]) {
        // exp_dyn_entry's series: the Engel first chunk, then every Hyper chunk.
        let first = exp_terms(1, l) + 1;
        let chunks: Vec<(DynHyperCtx, u64)> = exp_chunks(l)
            .into_iter()
            .map(|(xb, shift, s)| (DynHyperCtx::new(&rand_x(xb, &mut rng), shift), exp_terms(s, l) + 1))
            .collect();
        let new = (serial::<DynNodeEngel<Exp>>(), serial::<DynNodeHyper<Exp>>());
        let w = workload(format!("exp chunks {l}"), move |c| {
            let (ce, ch) = if c.leaf_limbs == 0 { (c, c) } else { new };
            tree::<DynNodeEngel<Exp>>(&Shift(1), 1, first, ce);
            for (ctx, n) in &chunks {
                tree::<DynNodeHyper<Exp>>(ctx, 1, *n, ch);
            }
        });
        let t = measure(&*w.run, &[old(32), new.1], args.rounds);
        row(w.name, t[0], t[1]);

        let x = rand_x(64 * l as u64, &mut rng);
        let mut out = vec![0; l];
        let s = Instant::now();
        let reps = 5;
        for _ in 0..reps {
            black_box(exp_dyn_entry(&x, 0, &mut out));
        }
        println!(
            "{:<22} {:>10} {:>10.3}",
            format!("exp_dyn_entry {l}"),
            "",
            s.elapsed().as_secs_f64() / reps as f64 * 1e3
        );
    }

    // ln2_static past its table, with N = out.len() + 4.
    fn static_row<const N: usize>(rounds: usize) -> (String, f64, f64) {
        let n = ln2_terms(N - 4);
        let w = workload(format!("ln2 static N={N}"), move |c| ln2_static_tree::<N>(n, c));
        let t = measure(&*w.run, &[old(640), SplitCutoffs::of::<StaticNodeBBP<N, Ln2A>>()], rounds);
        (w.name, t[0], t[1])
    }
    let rounds = args.rounds;
    let rows = std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || {
            prefer_fast_cores();
            [
                static_row::<1028>(rounds),
                static_row::<2048>(rounds),
                static_row::<4096>(rounds),
            ]
        })
        .unwrap()
        .join()
        .unwrap();
    for (name, o, n) in rows {
        row(name, o, n);
    }
}

// f for each candidate (A for ln, j for exp) on X in [1, 2) and X in [2^63, 2^64), both with
// m = 0, relative to each workload's best. other, when given, times its own function on the
// same X.
fn scale_e2e<C: Copy + std::fmt::Display>(
    args: &Args,
    name: &str,
    (knob, a): (&str, &[C]),
    f: &dyn Fn(&[u64], &mut [u64], C),
    other: Option<(&str, &dyn Fn(&[u64], &mut [u64]))>,
) {
    let mut rng = StdRng::seed_from_u64(23);
    let header: Vec<String> = a.iter().map(|a| format!("{knob}={a}")).collect();
    let other_h = other.map_or(String::new(), |(n, _)| format!(" {:>10}", format!("{n} ms")));
    println!("{:<22} {:>10}{other_h}  {}", "workload", "best ms", header.join(" "));
    let mut log_sum = vec![0.0; a.len()];
    let mut rows = 0;
    for l in sizes_or(args, &[4, 16, 64, 256, 1024]) {
        let near = rand_x(64 * l as u64 + 1, &mut rng);
        let far = rand_x(64 * (l as u64 + 1), &mut rng);
        for (row, x) in [("[1,2)", near), ("2^63x", far)] {
            let t = measure(&|a: C| f(&x, &mut vec![0; l], a), a, args.rounds);
            let best = t.iter().cloned().fold(f64::MAX, f64::min);
            let cells: Vec<String> = t
                .iter()
                .zip(&header)
                .map(|(&x, h)| format!("{:>w$.3}", x / best, w = h.len()))
                .collect();
            let other_c = other.map_or(String::new(), |(_, g)| {
                let t = measure(&|_: ()| g(&x, &mut vec![0; l]), &[()], args.rounds);
                format!(" {:>10.3}", t[0] * 1e3)
            });
            let row = format!("{name} {row} {l}");
            println!("{row:<22} {:>10.3}{other_c}  {}", best * 1e3, cells.join(" "));
            for (s, &x) in log_sum.iter_mut().zip(&t) {
                *s += (x / best).ln();
            }
            rows += 1;
        }
    }
    let cells: Vec<String> = log_sum
        .iter()
        .zip(&header)
        .map(|(&s, h)| format!("{:>w$.3}", (s / rows as f64).exp(), w = h.len()))
        .collect();
    let pad = other.map_or(String::new(), |_| format!(" {:>10}", ""));
    println!("{:<22} {:>10}{pad}  {}", "geomean", "", cells.join(" "));
}

// a b exactly.
fn full_mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = vec![0; a.len() + b.len() - 1];
    let top = mul_dyn(a, b, &mut out);
    out.push(top);
    out
}

// x's limbs from lo, n of them, zero past its end.
fn limbs_from(x: &[u64], lo: usize, n: usize) -> Vec<u64> {
    (0..n).map(|i| x.get(lo + i).copied().unwrap_or(0)).collect()
}

// ln_dyn's AGM inputs (ln_agm_dyn_at::<DYN_LN_AGM_TERMS>) at o output limbs for an X of
// 64 (o + 1) random bits (no cancellation), with agm_frame's f, l and g: θ from theta_dyn at
// q = x 2^-(bits(x) + l),
// b1 = θ2 θ3 and a1 = (θ2 + θ3)² / 2 - b1, from exact products floored to g fraction limbs
// (ln_agm_core's high products can differ in the last limbs).
fn agm_inputs(x: &[u64], o: usize) -> (Vec<u64>, Vec<u64>) {
    let f = o as u64 + 1;
    let n1 = DYN_LN_AGM_TERMS as u64 + 1;
    let l = (64 * (f + 1) + 3).div_ceil(4 * n1 * n1).max(2);
    let g = (f + 2 + l.div_ceil(64)) as usize;
    let z = ((l - 2) / 64) as usize;
    let (mut th2, mut th3) = (vec![0; g - z], vec![0; g + 1]);
    theta_dyn::<DYN_LN_AGM_TERMS>(x, 64 * x.len() as u64 + l, &mut th2, &mut th3);
    let th2 = &th2[..th2.iter().rposition(|&v| v != 0).unwrap() + 1];
    let b1 = limbs_from(&full_mul(th2, &th3), g, g);
    let mut sum = th3.clone();
    let mut c = 0u128;
    for (i, s) in sum.iter_mut().enumerate() {
        c += *s as u128 + th2.get(i).copied().unwrap_or(0) as u128;
        *s = c as u64;
        c >>= 64;
    }
    let mut sq = vec![0; 2 * sum.len() - 1];
    let top = sqr_dyn(&sum, &mut sq);
    sq.push(top);
    let mut a1 = limbs_from(&sq, g, g + 1);
    // a1 = sq / 2 - b1, in g limbs.
    for i in 0..g {
        a1[i] = a1[i] >> 1 | a1[i + 1] << 63;
    }
    a1.truncate(g);
    let mut borrow = 0i128;
    for (v, &w) in a1.iter_mut().zip(&b1) {
        let d = *v as i128 - w as i128 - borrow;
        *v = d as u64;
        borrow = (d < 0) as i128;
    }
    assert_eq!(borrow, 0);
    (a1, b1)
}

// agm_dyn_at against its tail's term cap at each size, relative to the best, each cap's
// roots/terms, j = cap 0's roots - its own, then agm_static_at at a few N sized to the frame.
fn agm_family(args: &Args) {
    let caps: Vec<usize> = args
        .tail
        .as_deref()
        .unwrap_or(&[0, 1, 2, 3, 5, 7, 10, 15, 19])
        .iter()
        .map(|&c| c as usize)
        .collect();
    let header: Vec<String> = caps.iter().map(|c| format!("I<={c}")).collect();
    let w = header.iter().map(|h| h.len()).max().unwrap().max(7);
    println!(
        "{:<22} {:>10} {:>10}  {}",
        "workload",
        "best ms",
        "ln ms",
        header.iter().map(|h| format!("{h:>w$}")).collect::<Vec<_>>().join(" ")
    );
    let mut rng = StdRng::seed_from_u64(29);
    let report = |name: String, t: &[f64], steps: &[(usize, usize)], ln: Option<f64>| {
        let best = t.iter().cloned().fold(f64::MAX, f64::min);
        let cells: Vec<String> = t.iter().map(|&x| format!("{:>w$.3}", x / best)).collect();
        let ln = ln.map_or(String::new(), |v| format!("{:.3}", v * 1e3));
        println!("{name:<22} {:>10.3} {ln:>10}  {}", best * 1e3, cells.join(" "));
        let r0 = steps[0].0;
        let cells: Vec<String> = steps
            .iter()
            .map(|&(r, i)| format!("{:>w$}", format!("{r}/{i} j{}", r0 - r)))
            .collect();
        println!("{:<22} {:>10} {:>10}  {}", "  roots/terms", "", "", cells.join(" "));
    };
    for o in sizes_or(args, &[64, 256, 1024, 4096, 16384]) {
        let x = rand_x(64 * (o as u64 + 1), &mut rng);
        let (a1, b1) = agm_inputs(&x, o);
        let out = RefCell::new(vec![0; a1.len()]);
        let steps: Vec<_> = caps
            .iter()
            .map(|&c| agm_dyn_at(&a1, &b1, c, &mut out.borrow_mut()))
            .collect();
        let run = |c: usize| {
            black_box(agm_dyn_at(&a1, &b1, c, &mut out.borrow_mut()));
        };
        let t = measure(&run, &caps, args.rounds);
        let lo = RefCell::new(vec![0; o]);
        let ln = measure(
            &|_: ()| {
                black_box(ln_agm_dyn_at::<DYN_LN_AGM_TERMS>(&x, 0, &mut lo.borrow_mut()).unwrap());
            },
            &[()],
            args.rounds,
        );
        report(format!("agm_dyn {o} (g={})", a1.len()), &t, &steps, Some(ln[0]));
    }

    // Static, with N two limbs past the frame g.
    fn static_row<const N: usize>(o: usize, caps: &[usize], rounds: usize) -> (String, Vec<f64>, Vec<(usize, usize)>) {
        let mut rng = StdRng::seed_from_u64(31);
        let (a1, b1) = agm_inputs(&rand_x(64 * (o as u64 + 1), &mut rng), o);
        assert!(a1.len() + 2 <= N);
        let out = RefCell::new(vec![0; a1.len()]);
        let steps: Vec<_> = caps
            .iter()
            .map(|&c| agm_static_at::<N>(&a1, &b1, c, &mut out.borrow_mut()).unwrap())
            .collect();
        let run = |c: usize| {
            black_box(agm_static_at::<N>(&a1, &b1, c, &mut out.borrow_mut()).unwrap());
        };
        (format!("agm_static N={N}"), measure(&run, caps, rounds), steps)
    }
    let (caps2, rounds) = (caps.clone(), args.rounds);
    let rows = std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || {
            prefer_fast_cores();
            [
                static_row::<72>(64, &caps2, rounds),
                static_row::<272>(256, &caps2, rounds),
                static_row::<1048>(1024, &caps2, rounds),
                static_row::<4168>(4096, &caps2, rounds),
            ]
        })
        .unwrap()
        .join()
        .unwrap();
    for (name, t, steps) in rows {
        report(name, &t, &steps, None);
    }
}

// An ln entry for the T sweep: X = x 2^(64 m) into out, unwrapped.
type LnAt = fn(&[u64], i64, &mut [u64]);

// ln_agm_dyn_at::<t>, for t up to theta's 16.
fn ln_agm_dyn_t(t: usize) -> LnAt {
    macro_rules! by_t {
        ($($n:literal)*) => {
            match t {
                $($n => |x, m, o| {
                    black_box(ln_agm_dyn_at::<$n>(x, m, o).unwrap());
                },)*
                _ => panic!("theta takes at most 16 terms"),
            }
        };
    }
    by_t!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
}

// ln_agm_static_at::<N, t>, for t up to static theta's 9.
fn ln_agm_static_t<const N: usize>(t: usize) -> LnAt {
    macro_rules! by_t {
        ($($n:literal)*) => {
            match t {
                $($n => |x, m, o| {
                    black_box(ln_agm_static_at::<N, $n>(x, m, o).expect("N fits the frame"));
                },)*
                _ => panic!("static theta takes at most 9 terms"),
            }
        };
    }
    by_t!(0 1 2 3 4 5 6 7 8 9)
}

// ln_agm end to end against theta's T at each size, on X = x 2^(-64 o) in [2^63, 2^64) with o + 1
// random limbs (no cancellation), relative to the best T, with the series ln_dyn_at on the same X
// for scale. Then the static entry at N with room for T = 0's frame.
fn lnagm_family(args: &Args) {
    let list = |default: &[u64]| -> Vec<usize> {
        args.t.as_deref().unwrap_or(default).iter().map(|&t| t as usize).collect()
    };
    let ts = list(&[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16]);
    let header: Vec<String> = ts.iter().map(|t| format!("T={t}")).collect();
    let w = header.iter().map(|h| h.len()).max().unwrap().max(6);
    let head = |ts: &[usize]| {
        let cells: Vec<String> = ts.iter().map(|t| format!("{:>w$}", format!("T={t}"))).collect();
        println!("{:<22} {:>10} {:>10}  {}", "workload", "best ms", "series ms", cells.join(" "));
    };
    let row = |name: String, t: &[f64], series: Option<f64>| {
        let best = t.iter().cloned().fold(f64::MAX, f64::min);
        let cells: Vec<String> = t.iter().map(|&x| format!("{:>w$.3}", x / best)).collect();
        let s = series.map_or(String::new(), |v| format!("{:.3}", v * 1e3));
        println!("{name:<22} {:>10.3} {s:>10}  {}", best * 1e3, cells.join(" "));
    };
    let input = |o: usize, rng: &mut StdRng| {
        let mut x = rand_x(64 * (o as u64 + 1), rng);
        x[o] |= 1 << 63;
        x
    };
    head(&ts);
    let mut rng = StdRng::seed_from_u64(37);
    for o in sizes_or(args, &[256, 512, 1024, 2048, 4096, 8192, 16384, 32768]) {
        let x = input(o, &mut rng);
        let m = -(o as i64);
        let out = RefCell::new(vec![0; o]);
        let fs: Vec<LnAt> = ts.iter().map(|&t| ln_agm_dyn_t(t)).collect();
        let t = measure(&|i: usize| fs[i](&x, m, &mut out.borrow_mut()), &(0..ts.len()).collect::<Vec<_>>(), args.rounds);
        let series = measure(
            &|_: ()| {
                black_box(ln_dyn_at(&x, m, &mut out.borrow_mut(), DYN_LN_SQRT_A).unwrap());
            },
            &[()],
            args.rounds,
        );
        row(format!("ln_agm_dyn {o}"), &t, Some(series[0]));
    }

    // Static T stops at 9, where theta's least-work plan still fits its slots.
    let sts: Vec<usize> = list(&[0, 1, 2, 3, 4, 5, 6, 8, 9]).into_iter().filter(|&t| t <= 9).collect();
    fn static_row<const N: usize>(o: usize, ts: &[usize], rounds: usize) -> (String, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(41);
        let mut x = rand_x(64 * (o as u64 + 1), &mut rng);
        x[o] |= 1 << 63;
        let m = -(o as i64);
        let out = RefCell::new(vec![0; o]);
        let fs: Vec<LnAt> = ts.iter().map(|&t| ln_agm_static_t::<N>(t)).collect();
        let idx: Vec<usize> = (0..ts.len()).collect();
        let t = measure(&|i: usize| fs[i](&x, m, &mut out.borrow_mut()), &idx, rounds);
        (format!("ln_agm_static {o} N={N}"), t)
    }
    head(&sts);
    let rounds = args.rounds;
    let rows = std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || {
            prefer_fast_cores();
            // T = 0's frame: g = o + 3 + ceil(l / 64), l = ceil((64 (o + 2) + 3) / 4).
            [
                static_row::<330>(256, &sts, rounds),
                static_row::<1300>(1024, &sts, rounds),
                static_row::<5160>(4096, &sts, rounds),
            ]
        })
        .unwrap()
        .join()
        .unwrap();
    for (name, t) in rows {
        row(name, &t, None);
    }
}

// Asks macOS to keep this thread on a performance core.
fn prefer_fast_cores() {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
        }
        const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
        unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
    }
}

fn main() {
    prefer_fast_cores();
    let a = args();
    let mut cands = a.cands.clone().unwrap_or_default();
    if cands.is_empty() {
        for &leaf_limbs in &a.limbs {
            for &leaf_terms in &a.terms {
                for &par_limbs in &a.par {
                    cands.push(SplitCutoffs {
                        leaf_limbs,
                        leaf_terms,
                        par_limbs,
                    });
                }
            }
        }
    }
    println!(
        "family {}, {} rounds, rayon threads {}, cutoffs as leaf limbs/leaf terms/par limbs",
        a.family,
        a.rounds,
        rayon::current_num_threads()
    );
    match a.family.as_str() {
        "bbp" => sweep(&bbp_workloads(&a), &cands, a.rounds),
        "engel" => sweep(&engel_workloads(&a), &cands, a.rounds),
        "hyper" => sweep(&hyper_workloads(&a), &cands, a.rounds),
        "par" => sweep(&par_workloads(&a), &cands, a.rounds),
        "e2e" => e2e(&a),
        "ln" => scale_e2e(
            &a,
            "ln",
            ("A", a.a.as_deref().unwrap_or(&[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])),
            &|x, out, s| {
                black_box(ln_dyn_at(x, 0, out, s).unwrap());
            },
            None,
        ),
        "exp" => scale_e2e(
            &a,
            "exp",
            ("j", a.j.as_deref().unwrap_or(&[0, 8, 16, 24, 32, 40])),
            &|x, out, s| {
                black_box(exp_dyn_at(x, 0, out, s).unwrap());
            },
            Some(("ln", &|x, out| {
                black_box(ln_dyn(x, 0, out).unwrap());
            })),
        ),
        "agm" => agm_family(&a),
        "lnagm" => lnagm_family(&a),
        f @ ("sbbp" | "sengel" | "shyper") => {
            // Static trees need far more stack than the main thread has.
            let f = f.to_string();
            let rounds = a.rounds;
            std::thread::Builder::new()
                .stack_size(1 << 30)
                .spawn(move || {
                    prefer_fast_cores();
                    sweep(&static_workloads(&f), &cands, rounds)
                })
                .unwrap()
                .join()
                .unwrap();
        }
        f => panic!("unknown family {f}"),
    }
}
