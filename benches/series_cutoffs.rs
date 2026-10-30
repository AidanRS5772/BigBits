//! Complete series evaluation, including validation and finalize. See docs/series_cutoff_tuning.md.
use big_bits::utils::{
    series::*,
    trans::{ATanh, Exp},
    utils::{cmp_buf, sub_buf},
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::{cmp::Ordering, env, hint::black_box, time::Instant};

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

struct Linear;
struct Quadratic;
struct Signed;
macro_rules! narrow {
    ($tr:ident $(, $r:ident)?) => {
        impl $tr for Linear {
            no_tail!(0);
            fn p(_: u64) -> u64 { 1 }
            fn q(n: u64) -> u64 { 2*n+3 }
            $(fn $r(n: u64) -> u64 { 2*n+1 })?
        }
        impl $tr for Quadratic {
            no_tail!(0);
            fn p(n: u64) -> u64 { n+1 }
            fn q(n: u64) -> u64 { (n+1)*(2*n+3) }
            $(fn $r(n: u64) -> u64 { (n+1)*(2*n+1) })?
        }
        impl $tr for Signed {
            no_tail!(0);
            const SIGNED: bool = true;
            fn p(n: u64) -> u64 { n%5+1 }
            fn q(n: u64) -> u64 { 2*n+7 }
            fn neg(n: u64) -> bool { n%3 == 1 }
            $(fn $r(n: u64) -> u64 { 2*n+1 })?
        }
    }
}
narrow!(BBPSeries);
narrow!(EngelSeries);
narrow!(HyperSeries, r);
struct Wide;
macro_rules! wide {
    ($tr:ident $(, $r:ident)?) => {
        impl $tr for Wide {
            no_tail!(0);
            fn p(n: u128) -> u128 { (1<<68)+n+1 }
            fn q(n: u128) -> u128 { (1<<70)+2*n+3 }
            $(fn $r(n: u128) -> u128 { (1<<70)+2*n+1 })?
        }
    }
}
wide!(WideBBPSeries);
wide!(WideEngelSeries);
wide!(WideHyperSeries, r);

#[derive(Clone)]
struct Case {
    name: &'static str,
    family: &'static str,
    len: usize,
    a: u64,
    t: u64,
    k: u64,
    zeros: usize,
    x: Vec<u64>,
    wide: bool,
}
struct Args {
    sizes: Vec<usize>,
    rounds: usize,
    ms: u64,
    seed: u64,
    pool: bool,
    blocks: bool,
    check: bool,
    holdout: bool,
    oversize: bool,
    work_only: bool,
    shapes: String,
    modes: String,
    algs: String,
}
fn arg(name: &str, default: &str) -> String {
    let v: Vec<_> = env::args().collect();
    v.windows(2)
        .find(|v| v[0] == name)
        .map_or(default.into(), |v| v[1].clone())
}
fn flag(name: &str) -> bool {
    env::args().any(|s| s == name)
}
fn cases(l: usize, args: &Args) -> Vec<Case> {
    let mut rng = StdRng::seed_from_u64(args.seed + l as u64);
    let mut v = Vec::new();
    let mut add = |name, family, t, k, zeros, xl: usize, wide| {
        let mut x: Vec<u64> = (0..xl).map(|_| rng.gen()).collect();
        if xl == 0 {
            x.push(1);
        } else {
            *x.last_mut().unwrap() |= 1 << 63;
        }
        v.push(Case {
            name,
            family,
            len: l,
            a: if args.holdout { 3 } else { 1 },
            t,
            k: k + 64 * xl as u64,
            zeros,
            x,
            wide,
        });
    };
    let n = 8 * l as u64 + 7;
    add("linear", "bbp", n, 8, 0, 0, false);
    add("sparse", "bbp", l as u64 + 3, 130, 2, 0, false);
    add("short", "bbp", 7, 3, 0, 0, false);
    add("signed", "bbp", n / 2 + 1, 4, 0, 0, false);
    add("wide", "bbp", l as u64 + 3, 8, 0, 0, true);
    add("factorial", "engel", exp_terms(1, l), 1, 0, 0, false);
    add("linear", "engel", n / 2 + 1, 3, 0, 0, false);
    add("wide", "engel", l as u64 / 2 + 3, 3, 0, 0, true);
    add("x1", "hyper", n, 8, 0, 0, false);
    add("word", "hyper", exp_terms(3, l), 3, 0, 1, false);
    let xl = (l / 4).clamp(2, 256);
    let gap = 64 * xl as u64;
    add("chunk", "hyper", exp_terms(gap, l), gap, 0, xl, false);
    add(
        "boundary",
        "hyper",
        (2 * l as u64 + 7).min(1024),
        0,
        0,
        2,
        false,
    );
    add(
        "deep",
        "hyper",
        l as u64 / 2 + 3,
        130,
        2,
        (l / 4).clamp(2, 256),
        false,
    );
    add("wide", "hyper", l as u64 + 3, 8, 0, 2, true);
    if args.holdout {
        // Unseen polynomial/alternating/atanh shapes, not just new random values.
        for c in &mut v {
            if c.name == "linear" {
                c.name = "quadratic";
            }
            if c.family == "hyper" && c.name == "boundary" {
                c.name = "atanh";
            }
            if c.family == "hyper" && c.name == "chunk" {
                c.name = "signed";
            }
            c.t = (c.t * 3 / 4).max(1);
        }
    }
    v.into_iter()
        .filter(|c| args.shapes == "all" || args.shapes.split(',').any(|s| s == c.name))
        .collect()
}
fn close(out: &[u64], sign: bool, want: &(Vec<u64>, bool), c: &Case, label: &str) {
    let (mut hi, lo) = if cmp_buf(out, &want.0) == Ordering::Less {
        (want.0.clone(), out)
    } else {
        (out.to_vec(), &want.0[..])
    };
    sub_buf(&mut hi, lo);
    assert!(
        hi.first().copied().unwrap_or(0) <= 1 && hi.iter().skip(1).all(|&x| x == 0),
        "{} {} len={} {label}: accuracy",
        c.family,
        c.name,
        c.len
    );
    assert!(
        want.0.iter().skip(1).all(|&x| x == 0) && want.0[0] <= 1 || sign == want.1,
        "sign"
    );
}
fn reference<N: Node + Send>(ctx: &N::Ctx, c: &Case) -> (Vec<u64>, bool)
where
    N::Ctx: Sync,
{
    // Dynamic nodes retain exact P/Q; their finalize performs the exact shifted division.
    let mut out = vec![0; c.len];
    let sign = bin_split::<N>(ctx, c.a, c.a + c.t).finalize(ctx, c.zeros, &mut out);
    (out, sign)
}
fn measure<N: SeriesNode>(
    ctx: &N::Ctx,
    c: &Case,
    capacity: usize,
    raw_e: u64,
    q_e: u64,
    want: &(Vec<u64>, bool),
    args: &Args,
    rect: Option<&dyn Fn(&mut [u64], usize) -> bool>,
) where
    N::Ctx: Sync,
{
    let b = c.a + c.t;
    let ok = N::eligible(ctx, c.a, b, c.len, c.zeros);
    let picked = N::dispatch(ctx, c.a, b, c.len, c.zeros);
    // The cost model's work for each row: rect's at the row's block length.
    let costs = N::costs(ctx, c.a, b, c.len, c.zeros, None);
    let work = |alg: Option<SeriesAlg>, m: usize| match (alg, m) {
        (_, 0) => costs.of(alg.unwrap_or(picked)).unwrap(),
        _ => {
            N::costs(ctx, c.a, b, c.len, c.zeros, Some(m))
                .rect
                .unwrap()
                .1
        }
    };
    let xl = if c.x == [1] { 0 } else { c.x.len() };
    let emit = |label: &str, m: usize, round: usize, ns: f64, w: Work| {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:?},{},{},{:.3},{},{},{},{},{},{},{}",
            if capacity == 0 { "dyn" } else { "static" },
            c.family,
            c.name,
            c.wide,
            c.len,
            capacity,
            c.a,
            c.t,
            c.k,
            c.zeros,
            xl,
            N::limbs(ctx, c.a, b),
            raw_e,
            q_e,
            N::LEAF_LIMBS,
            N::LEAF_TERMS,
            label,
            picked,
            m,
            round,
            ns,
            w.div,
            w.scalar,
            w.product,
            w.term,
            w.call,
            w.alloc,
            w.fill
        );
    };
    if args.work_only {
        // Work vectors alone, rect's at every block length a measurement may have used, so
        // a changed model can be refitted to earlier timings (fit --work).
        for alg in [SeriesAlg::Horner, SeriesAlg::Rect, SeriesAlg::BinSplit] {
            if let Some(w) = costs.of(alg) {
                let m = if alg == SeriesAlg::Rect {
                    costs.rect.unwrap().0
                } else {
                    0
                };
                emit(&format!("{alg:?}"), m, 0, 0.0, w);
            }
        }
        if ok.rect {
            let max = (c.t as usize).min(if capacity == 0 { 2048 } else { 16 });
            for m in 1..=max {
                emit(&format!("W{m}"), m, 0, 0.0, work(Some(SeriesAlg::Rect), m));
            }
        }
        return;
    }
    let mut candidates: Vec<(String, Option<SeriesAlg>, usize)> =
        [SeriesAlg::Horner, SeriesAlg::Rect, SeriesAlg::BinSplit]
            .into_iter()
            .filter(|&a| ok.allows(a))
            .map(|a| (format!("{a:?}"), Some(a), 0))
            .collect();
    candidates.push(("Dispatch".into(), None, 0));
    if args.blocks && ok.rect {
        for scale in [0.25, 0.5, 0.75, 1., 1.5, 2., 3., 4.] {
            let m = ((c.t as f64).sqrt() * scale).round() as usize;
            let m = m.clamp(
                1,
                (c.t as usize).min(if capacity == 0 { usize::MAX } else { 16 }),
            );
            for m in [
                m.saturating_sub(1).max(1),
                m,
                (m + 1).min((c.t as usize).min(if capacity == 0 { usize::MAX } else { 16 })),
            ] {
                if !candidates.iter().any(|r| r.2 == m) {
                    candidates.push((format!("M{m}"), Some(SeriesAlg::Rect), m));
                }
            }
        }
    }
    let checked_rect = |out: &mut [u64], m| {
        assert!(N::eligible(ctx, c.a, b, c.len, c.zeros).rect);
        rect.unwrap()(out, m)
    };
    let mut out = vec![0; c.len];
    let mut run = |alg: Option<SeriesAlg>, m: usize| {
        let sign = if m != 0 {
            checked_rect(&mut out, m)
        } else if let Some(a) = alg {
            series_eval_at::<N>(ctx, c.a, b, c.zeros, &mut out, a)
        } else {
            series_eval_range::<N>(ctx, c.a, b, c.zeros, &mut out)
        };
        black_box((&out, sign));
        sign
    };
    // Check all candidates before any timing, and calibrate independently.
    let mut counts = Vec::new();
    for (label, alg, m) in &candidates {
        let sign = run(*alg, *m);
        // End the mutable borrow of out through run for the correctness check below.
        let _ = sign;
        let start = Instant::now();
        run(*alg, *m);
        let ns = start.elapsed().as_nanos().max(1);
        counts.push(((args.ms as u128 * 1_000_000 / ns).clamp(1, 1_000_000)) as usize);
        black_box(label);
    }
    drop(run);
    for (label, alg, m) in &candidates {
        let sign = if *m != 0 {
            checked_rect(&mut out, *m)
        } else if let Some(a) = alg {
            series_eval_at::<N>(ctx, c.a, b, c.zeros, &mut out, *a)
        } else {
            series_eval_range::<N>(ctx, c.a, b, c.zeros, &mut out)
        };
        close(&out, sign, want, c, label);
    }
    if args.check {
        return;
    }
    let mut order: Vec<_> = (0..candidates.len()).collect();
    let mut rng = StdRng::seed_from_u64(args.seed + c.len as u64 + c.t);
    for round in 0..args.rounds {
        order.shuffle(&mut rng);
        for &i in &order {
            let (label, alg, m) = &candidates[i];
            if args.algs != "all" && !args.algs.split(',').any(|a| a == label) {
                continue;
            }
            let start = Instant::now();
            for _ in 0..counts[i] {
                let sign = if *m != 0 {
                    checked_rect(&mut out, *m)
                } else if let Some(a) = alg {
                    series_eval_at::<N>(ctx, c.a, b, c.zeros, &mut out, *a)
                } else {
                    series_eval_range::<N>(ctx, c.a, b, c.zeros, &mut out)
                };
                black_box((&out, sign));
            }
            let elapsed = start.elapsed();
            let ns = elapsed.as_secs_f64() * 1e9 / counts[i] as f64;
            if elapsed.as_millis() < args.ms as u128 / 2 {
                counts[i] = (counts[i] * 2).min(1_000_000);
            }
            // A forced rect row runs the model's block length.
            let block = match (alg, m) {
                (Some(SeriesAlg::Rect), 0) => costs.rect.unwrap().0,
                _ => *m,
            };
            emit(label, block, round, ns, work(*alg, *m));
        }
    }
}
fn coefficient_limbs(c: &Case, q: impl Fn(u128) -> u128) -> u64 {
    c.t.saturating_mul((128 - q((c.a + c.t / 2) as u128).leading_zeros()) as u64)
        .div_ceil(64)
}
fn bbp<S: WideBBPSeries, const N: usize>(c: &Case, args: &Args) {
    let ctx = Shift(c.k);
    let want = if args.work_only {
        (Vec::new(), false)
    } else {
        reference::<DynNodeBBP<S>>(&ctx, c)
    };
    let raw_e = DynNodeBBP::<S>::limbs(&ctx, c.a, c.a + c.t);
    if N == 2 {
        measure::<DynNodeBBP<S>>(
            &ctx,
            c,
            0,
            raw_e,
            coefficient_limbs(c, S::q),
            &want,
            args,
            None,
        )
    } else {
        measure::<StaticNodeBBP<N, S>>(
            &ctx,
            c,
            N,
            raw_e,
            coefficient_limbs(c, S::q),
            &want,
            args,
            None,
        )
    }
}
fn engel<S: WideEngelSeries, const N: usize>(c: &Case, args: &Args) {
    let ctx = Shift(c.k);
    let want = if args.work_only {
        (Vec::new(), false)
    } else {
        reference::<DynNodeEngel<S>>(&ctx, c)
    };
    let raw_e = DynNodeEngel::<S>::limbs(&ctx, c.a, c.a + c.t);
    if N == 2 {
        measure::<DynNodeEngel<S>>(
            &ctx,
            c,
            0,
            raw_e,
            coefficient_limbs(c, S::q),
            &want,
            args,
            None,
        )
    } else {
        measure::<StaticNodeEngel<N, S>>(
            &ctx,
            c,
            N,
            raw_e,
            coefficient_limbs(c, S::q),
            &want,
            args,
            None,
        )
    }
}
fn hyper<S: WideHyperSeries, const N: usize>(c: &Case, args: &Args) {
    let ctx = DynHyperCtx::new(&c.x, c.k);
    let want = if args.work_only {
        (Vec::new(), false)
    } else {
        reference::<DynNodeHyper<S>>(&ctx, c)
    };
    let raw_e = DynNodeHyper::<S>::limbs(&ctx, c.a, c.a + c.t);
    if N == 2 {
        let rect = |out: &mut [u64], m| rect_dyn_m::<S>(&ctx, c.a, c.a + c.t, c.zeros, out, m);
        measure::<DynNodeHyper<S>>(
            &ctx,
            c,
            0,
            raw_e,
            coefficient_limbs(c, S::q),
            &want,
            args,
            Some(&rect),
        );
    } else {
        let ctx = StaticHyperCtx::<N>::new(&c.x, c.k);
        let rect =
            |out: &mut [u64], m| rect_static_m::<N, S>(&ctx, c.a, c.a + c.t, c.zeros, out, m);
        measure::<StaticNodeHyper<N, S>>(
            &ctx,
            c,
            N,
            raw_e,
            coefficient_limbs(c, S::q),
            &want,
            args,
            Some(&rect),
        );
    }
}
fn execute<const N: usize>(c: &Case, args: &Args) {
    match (c.family, c.name) {
        ("bbp", "wide") => bbp::<Wide, N>(c, args),
        ("bbp", "signed") => bbp::<Signed, N>(c, args),
        ("bbp", "quadratic") => bbp::<Quadratic, N>(c, args),
        ("bbp", _) => bbp::<Linear, N>(c, args),
        ("engel", "wide") => engel::<Wide, N>(c, args),
        ("engel", "factorial") => engel::<Exp, N>(c, args),
        ("engel", "quadratic") => engel::<Quadratic, N>(c, args),
        ("engel", _) => engel::<Linear, N>(c, args),
        (_, "wide") => hyper::<Wide, N>(c, args),
        (_, "word" | "chunk" | "deep") => hyper::<Exp, N>(c, args),
        (_, "atanh") => hyper::<ATanh, N>(c, args),
        (_, "signed") => hyper::<Signed, N>(c, args),
        _ => hyper::<Linear, N>(c, args),
    }
}
fn main() {
    let args = Args {
        sizes: arg("--sizes", "1,2,4,8,16,32,64,128,256,512,1024,2048,4096")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect(),
        rounds: arg("--rounds", "15").parse().unwrap(),
        ms: arg("--ms", "10").parse().unwrap(),
        seed: arg("--seed", "71").parse().unwrap(),
        pool: arg("--execution", "caller") == "pool",
        blocks: flag("--blocks"),
        check: flag("--check"),
        holdout: flag("--holdout"),
        oversize: flag("--oversize"),
        work_only: flag("--work-only"),
        shapes: arg("--shapes", "all"),
        modes: arg("--modes", "both"),
        algs: arg("--algs", "all"),
    };
    let threads: usize = arg("--threads", "0").parse().unwrap();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .unwrap();
    eprintln!(
        "threads={} pool={} seed={} holdout={} rounds={} ms={}",
        rayon::current_num_threads(),
        args.pool,
        args.seed,
        args.holdout,
        args.rounds,
        args.ms
    );
    println!("mode,family,shape,wide,len,capacity,a,t,k,zeros,xl,e,raw_e,q_e,leaf_limbs,leaf_terms,alg,picked,m,round,ns,w_div,w_scalar,w_product,w_term,w_call,w_alloc,w_fill");
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let work = || {
                assert_eq!(rayon::current_thread_index().is_some(), args.pool);
                for &l in &args.sizes {
                    for c in cases(l, &args) {
                        eprintln!("{} {} len={l}", c.family, c.name);
                        if args.modes != "static" {
                            execute::<2>(&c, &args);
                        }
                        if args.modes != "dyn" && args.oversize {
                            // Static storage far past the output, as N = 8192 for every size.
                            if l + 4 <= 8192 {
                                execute::<8192>(&c, &args);
                            }
                        } else if args.modes != "dyn" {
                            match l + 4 {
                                0..=8 => execute::<8>(&c, &args),
                                9..=32 => execute::<32>(&c, &args),
                                33..=128 => execute::<128>(&c, &args),
                                129..=512 => execute::<512>(&c, &args),
                                513..=2048 => execute::<2048>(&c, &args),
                                2049..=8192 => execute::<8192>(&c, &args),
                                _ => (),
                            }
                        }
                    }
                }
            };
            if args.pool {
                rayon::in_place_scope(|scope| scope.spawn(|_| work()));
            } else {
                work();
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
