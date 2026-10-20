// Tunes binary splitting's leaf and parallel cutoffs.
//
//   cargo bench --bench bin_split_cutoffs -- <family> [--limbs 8,16,max] [--terms 1,4]
//       [--par 256,max] [--cands 0/31/max,64/1/max] [--rounds 9] [--sizes 256,1024]
//
// Families: bbp, engel, hyper, sbbp, sengel, shyper (leaf sweeps, serial), par (parallel
// cutoff, all threads, default leaves), exp (exp_dyn's first-chunk width from --limbs and
// chunk-parallel gate from --par, defaulting to the current constants) and e2e (old term
// cutoffs against the defaults).
// Each candidate times a tree build only: finalize doesn't depend on the tree's shape.
// Candidates alternate within each round and each one reports its median over rounds,
// relative to the workload's best; the summary is the geometric mean over workloads.

use big_bits::utils::{
    bin_split::{
        bin_split_tree_at, BBPSeries, DynHyperCtx, DynNodeBBP, DynNodeEngel, DynNodeHyper,
        HyperSeries, Node, Shift, SplitCutoffs, StaticHyperCtx, StaticNodeBBP, StaticNodeEngel,
        StaticNodeHyper,
    },
    trans::{exp_dyn, exp_dyn_at, exp_terms, Exp, ExpCutoffs},
    DYN_EXP_CHUNK_MIN_BITS, DYN_EXP_CHUNK_PAR_WORK,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{env, hint::black_box, time::Instant};

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

// Ratio (2n + 1) / (2n + 4) per term.
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

// exp_dyn's chunk schedule for a normalized w-limb x: chunk i covers bits
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

// exp_dyn's schedule cutoffs: whole exp_dyn calls over precision l and leading zero
// limbs m, which together set the chunk count. A candidate's leaf_limbs is the first
// chunk's minimum width in bits, and its par_limbs the parallel gate's chunks * v.
fn exp_workloads(args: &Args) -> Vec<Workload> {
    let mut v = Vec::new();
    let mut rng = StdRng::seed_from_u64(19);
    let mut shapes = Vec::new();
    for l in sizes_or(args, &[4, 8, 12, 16, 24, 32, 48, 64, 128, 256, 1024, 4096]) {
        let lm = l as u64;
        let mut ms = vec![0, 1, 2, 4, lm / 8, lm / 4, lm / 2, lm];
        ms.sort();
        ms.dedup();
        shapes.extend(ms.into_iter().map(|m| (l, m)));
    }
    if args.sizes.is_none() {
        // Where the narrow chunks near x's top cost the most.
        shapes.push((16384, 0));
    }
    for (l, m) in shapes {
        let x = rand_x(64 * l as u64 - 1, &mut rng);
        let out = std::cell::RefCell::new(vec![0; l]);
        v.push(workload(format!("l={l} m={m}"), move |c| {
            let cut = ExpCutoffs {
                chunk_min_bits: c.leaf_limbs,
                par_work: c.par_limbs as usize,
            };
            black_box(exp_dyn_at(&x, m, &mut out.borrow_mut(), cut));
        }));
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
    let family = it.next().expect("family");
    let exp = family == "exp";
    let mut a = Args {
        limbs: if exp { vec![DYN_EXP_CHUNK_MIN_BITS] } else { (1..=11).map(|i| 1 << i).collect() },
        terms: vec![1],
        par: vec![if exp { DYN_EXP_CHUNK_PAR_WORK as u64 } else { u64::MAX }],
        family,
        cands: None,
        rounds: 9,
        sizes: None,
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
fn measure(w: &Workload, cands: &[SplitCutoffs], rounds: usize) -> Vec<f64> {
    // Calibrate on the middle candidate so each sample takes about 20 ms.
    let mid = cands[cands.len() / 2];
    (w.run)(mid);
    let s = Instant::now();
    (w.run)(mid);
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
                (w.run)(cands[i]);
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
        let t = measure(w, cands, rounds);
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

// Old term cutoffs (ln2 640, exp 32) against the defaults, serial, plus exp_dyn
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
        let t = measure(&w, &[old(640), serial::<DynNodeBBP<Ln2A>>()], args.rounds);
        row(w.name, t[0], t[1]);
    }
    for l in sizes_or(args, &[256, 1024, 4096]) {
        // exp_dyn's series: the Engel first chunk, then every Hyper chunk.
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
        let t = measure(&w, &[old(32), new.1], args.rounds);
        row(w.name, t[0], t[1]);

        let x = rand_x(64 * l as u64, &mut rng);
        let mut out = vec![0; l];
        let s = Instant::now();
        let reps = 5;
        for _ in 0..reps {
            black_box(exp_dyn(&x, 0, &mut out));
        }
        println!(
            "{:<22} {:>10} {:>10.3}",
            format!("exp_dyn {l}"),
            "",
            s.elapsed().as_secs_f64() / reps as f64 * 1e3
        );
    }

    // ln2_static past its table, with N = out.len() + 4.
    fn static_row<const N: usize>(rounds: usize) -> (String, f64, f64) {
        let n = ln2_terms(N - 4);
        let w = workload(format!("ln2 static N={N}"), move |c| ln2_static_tree::<N>(n, c));
        let t = measure(&w, &[old(640), SplitCutoffs::of::<StaticNodeBBP<N, Ln2A>>()], rounds);
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
        "exp" => {
            println!("exp: leaf limbs = first-chunk min bits, par limbs = parallel chunks * v");
            sweep(&exp_workloads(&a), &cands, a.rounds)
        }
        "e2e" => e2e(&a),
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
