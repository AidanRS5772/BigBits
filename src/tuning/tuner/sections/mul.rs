//! Section `mul`'s measurements (the pilot, docs/tuning_implementation.md P2.3): the schoolbook
//! kernels' cost models and the Karatsuba level's fitted form, each fitted where it competes, on
//! shapes scattered over that area (`design::products`) and judged on the half held out.

use crate::tuning::resolve::Value;
use crate::tuning::tuner::derive::{chunk_products, level_products, school_with};
use crate::tuning::tuner::design::{products, Design};
use crate::tuning::tuner::fit::{analyze, fit_split, Fit};
use crate::tuning::tuner::measure::{pad, time_fit, time_min, Level, Probe};
use crate::tuning::tuner::out::{push_fit, Tuned};
use crate::tuning::tuner::sweep::{cutoff_sweep, dense_around, geo, ints, keep, keep_or_take, lost, pairs, sweep_cutoff};
use crate::utils::mul::{
    chunking_karatsuba, chunking_karatsuba_cost, karatsuba_entry_static, karatsuba_sqr_entry_static, ntt_entry_static,
    ntt_sqr_entry_static, fft_entry, fft_mul_size, fft_sqr_entry, fft_transform_work, karatsuba_core,
    karatsuba_cost, karatsuba_entry_dyn, karatsuba_sqr_core, karatsuba_sqr_entry_dyn, mul_buf, sqr_buf,
};
use std::hint::black_box;

type Mul = fn(&[u64], &[u64], &mut [u64]) -> u64;
type Sqr = fn(&[u64], &mut [u64]) -> u64;

// `mul.kara` with every shape on schoolbook, so a Karatsuba level's sub-products are leaves.
pub(crate) fn all_school() -> Value {
    Value::Table(vec![1, 1, 0])
}

pub fn tune(level: Level) -> Vec<Tuned> {
    let mut out = Vec::new();
    let n = level.fit_shapes(60);
    // mul_buf, W = (l s, 1), fitted where it competes with Karatsuba: as a level's
    // sub-products (from about 10 limbs) and against a level (to about 100), at every ratio a row
    // of mul.kara's table covers (l to 4s). A wider range of sizes fits the crossover worse, as
    // the row kernels' cost per limb isn't constant across it.
    let fit = fit_lin(
        level,
        &mut out,
        &["school_scale", "school_shift"],
        "school_err",
        "mul_buf, s 10-96, l from s to 4s; W = l s, 1",
        &products(n, (10, 96), (1.0, 4.0), |_, _| true),
        |(l, s)| vec![(l * s) as f64, 1.0],
        |(l, s), pass| {
            let (a, b) = (pad(l, l as u64 * 31, pass), pad(s, s as u64 * 17, pass));
            let mut o = pad(l + s - 1, 3, pass);
            Probe::new(format!("mul_buf {l}x{s}"), move || {
                black_box(mul_buf(black_box(&a), black_box(&b), &mut o));
            })
        },
    );
    let school = school_with(fit.beta[0], fit.beta[1]);

    // The next two fits run with mul.kara all schoolbook, so their sub-products are leaves.
    let saved = crate::tuning::snapshot();
    crate::tuning::set("mul.kara", all_school()).unwrap();
    // One karatsuba_core level over schoolbook leaves, in its theoretical form with fitted
    // constants: scale * (three sub-products) + shift, at every shape a level takes (s above
    // half of l).
    fit_lin(
        level,
        &mut out,
        &["kara_scale", "kara_shift"],
        "kara_err",
        "karatsuba_core over schoolbook leaves, s 24-96, l from s to 2s - 2",
        &products(n, (24, 96), (1.0, 2.0), |l, s| s > (l + 1) / 2),
        |(l, s)| vec![level_products(l, s, &school), 1.0],
        |(l, s), pass| {
            let h = (l + 1) / 2;
            let (a, b) = (pad(l, l as u64 * 37, pass), pad(s, s as u64 * 41, pass));
            let (mut o, mut cross, mut scratch) = (pad(l + s - 1, 5, pass), pad(2 * h + 3, 7, pass), pad(4 * l + 8, 9, pass));
            Probe::new(format!("kara level {l}x{s}"), move || {
                black_box(karatsuba_core(black_box(&a), black_box(&b), h, &mut o, &mut cross, &mut scratch));
            })
        },
    );
    // chunking_karatsuba over schoolbook products: scale * (its chunk products' fitted
    // schoolbook time) + pass * (l + s). This is the table's chunking
    // term, fitted on chunking itself rather than assembled from pass units (the user's
    // decision, 2026-11-10).
    fit_lin(
        level,
        &mut out,
        &["chunk_scale", "chunk_pass"],
        "chunk_err",
        "chunking_karatsuba over schoolbook products, s 24-96, l from 2s to 4s",
        &products(n, (24, 96), (2.0, 4.0), |l, s| s <= (l + 1) / 2),
        |(l, s)| vec![chunk_products(l, s, |a, b| school(a.max(b) as f64, a.min(b) as f64)), (l + s) as f64],
        |(l, s), pass| {
            let (a, b) = (pad(l, l as u64 * 67, pass), pad(s, s as u64 * 71, pass));
            let (mut o, mut scratch) = (pad(l + s - 1, 33, pass), pad(8 * l + 64, 35, pass));
            Probe::new(format!("chunking {l}x{s}"), move || {
                black_box(chunking_karatsuba(black_box(&a), black_box(&b), &mut o, &mut scratch));
            })
        },
    );
    crate::tuning::restore(&saved);

    // mul.karatsuba_sqr as a sweep: sqr_buf against one level over sqr_buf leaves at every size
    // from 40 to 140. The two are within a few percent over a wide range there, where the fitted
    // models' margin put the cutoff anywhere from 75 to 79 between runs.
    let sizes: Vec<usize> = (40..=140).collect();
    let saved = crate::tuning::snapshot();
    crate::tuning::set("mul.karatsuba_sqr", Value::Int(usize::MAX as u64)).unwrap();
    let t = time_min(level, |pass| {
        sizes
            .iter()
            .flat_map(|&n| {
                let h = (n + 1) / 2;
                let (a, a2) = (pad(n, n as u64 * 47, pass), pad(n, n as u64 * 47, pass));
                let (mut o, mut o2) = (pad(2 * n - 1, 25, pass), pad(2 * n - 1, 27, pass));
                let (mut cross, mut scratch) = (pad(2 * h + 1, 29, pass), pad(8, 31, pass));
                [
                    Probe::new("sqr_buf", move || {
                        black_box(sqr_buf(black_box(&a), &mut o));
                    }),
                    Probe::new("level", move || {
                        black_box(karatsuba_sqr_core(black_box(&a2), h, &mut o2, &mut cross, &mut scratch));
                    }),
                ]
            })
            .collect()
    });
    crate::tuning::restore(&saved);
    out.push(sweep_cutoff("mul.karatsuba_sqr", &sizes, &pairs(&t), level.eps()).0);
    out
}

// A product's probe through `f`, and a square's.
fn product(f: Mul) -> impl Fn((usize, usize), usize) -> Probe<'static> {
    move |(l, s), pass| {
        let (a, b) = (pad(l, l as u64 * 53, pass), pad(s, s as u64 * 59, pass));
        let mut o = pad(l + s - 1, 19, pass);
        Probe::new("product", move || {
            black_box(f(black_box(&a), black_box(&b), &mut o));
        })
    }
}

fn square(f: Sqr) -> impl Fn(usize, usize) -> Probe<'static> {
    move |n, pass| {
        let a = pad(n, n as u64 * 61, pass);
        let mut o = pad(2 * n - 1, 23, pass);
        Probe::new("square", move || {
            black_box(f(black_box(&a), &mut o));
        })
    }
}

// `steps` sizes from `lo`, `per_octave` to a doubling.
fn grid(lo: f64, steps: usize, per_octave: f64) -> Vec<usize> {
    geo(lo, per_octave).take(steps).collect()
}

// Each size s as the shapes (m s, s) for the multiples `mults`, with at most `cap` limbs in all.
fn shapes_of(sizes: &[usize], mults: &[f64], cap: usize) -> Vec<(usize, usize)> {
    sizes.iter().flat_map(|&s| mults.iter().map(move |m| ((s as f64 * m).round() as usize, s))).filter(|&(l, s)| l + s <= cap).collect()
}

// The static Karatsuba-to-NTT ratio sweeps' shapes: l the near-square multiples of s
// (`is_karatsuba`'s Karatsuba ratio) or the unbalanced ones (its chunking ratio).
const NEAR_M: [f64; 4] = [1.0, 1.2, 1.5, 1.9];
const FAR_M: [f64; 4] = [2.0, 3.0, 4.0, 8.0];

// Fits t = b0 w0 + b1 w1 + ... to the probes of `design`'s shapes (`feats` gives a shape's
// regressors w, a 1 among them where the form has a shift), judged on the shapes it holds out.
// Records the knobs `keys` names (the constants, in order; one that solves negative isn't needed
// and reads 0) and `err` (the rms it is judged by) with `what` in their evidence, and returns the
// fit.
fn fit_lin(
    level: Level,
    out: &mut Vec<Tuned>,
    keys: &[&'static str],
    err: &'static str,
    what: &str,
    design: &Design<(usize, usize)>,
    feats: impl Fn((usize, usize)) -> Vec<f64>,
    probe: impl Fn((usize, usize), usize) -> Probe<'static>,
) -> Fit {
    let timed = time_fit(level, |pass| design.shapes.iter().map(|&s| probe(s, pass)).collect());
    let rows: Vec<Vec<f64>> = design.shapes.iter().map(|&s| feats(s)).collect();
    let fixed = vec![0.0; rows.len()];
    let mut fit = fit_split(&rows, &timed.ns, &fixed, &design.held);
    fit.noise = Some(timed.noise);
    push_fit(out, keys, err, &fit, fit.evidence() + " (" + what + ")");
    analyze(&format!("mul.{}: {what}", err.trim_end_matches("_err")), keys, &rows, &timed.ns, &fixed, &design.held);
    fit
}

/// mul's second phase, measured once `mul.kara` and `mul.karatsuba_sqr` are derived and live:
/// recursive Karatsuba through its real entries, and the FFT entries, each in its fitted form.
pub fn tune_late(level: Level) -> Vec<Tuned> {
    let mut out = Vec::new();
    // Recursive Karatsuba in each regime: its theoretical work at the exponent log2 3, plus
    // schoolbook's l s, which its leaves still do. The schoolbook weight stands where a fitted
    // exponent stood (the user's decision, 2026-11-11), so these are linear fits like every
    // other. (A term a limb and a shift were in the form at first, and the timings didn't need
    // them.)
    let n = level.fit_shapes(60);
    fit_lin(
        level,
        &mut out,
        &["kara_rec_scale", "kara_rec_school"],
        "kara_rec_err",
        "karatsuba_entry_dyn, s 96-1300, l from s to 2s - 2; W = karatsuba_cost, l s",
        &products(n, (96, 1300), (1.0, 2.0), |l, s| s > (l + 1) / 2),
        |(l, s)| vec![karatsuba_cost(l, s), (l * s) as f64],
        product(karatsuba_entry_dyn),
    );
    fit_lin(
        level,
        &mut out,
        &["chunk_rec_scale", "chunk_rec_school"],
        "chunk_rec_err",
        "karatsuba_entry_dyn, s 96-650, l from 2s to 8s; W = chunking_karatsuba_cost, l s",
        &products(n, (96, 650), (2.0, 8.0), |l, s| s <= (l + 1) / 2),
        |(l, s)| vec![chunking_karatsuba_cost(l, s), (l * s) as f64],
        product(karatsuba_entry_dyn),
    );
    fit_lin(
        level,
        &mut out,
        &["fft_scale"],
        "fft_err",
        "fft_entry, s 96-2200, l from s to 8s; W = its transform's work",
        &products(n, (96, 2200), (1.0, 8.0), |_, _| true),
        |(l, s)| vec![fft_transform_work(fft_mul_size(l, s))],
        product(fft_entry),
    );

    // The static entries below are swept at N = 8192 only: approximate values, as a static
    // decision's best value also depends on N (the user's decision, 2026-11-11: approximate
    // static values until static memory handling is reworked).
    const N: usize = 8192;

    // The one-input and static Karatsuba-to-transform decisions, as sweeps: both entries timed at
    // each shape, and the value (a cutoff, or is_karatsuba's ratio against the NTT) with the least
    // measured time lost. (Dynamic products compare the equations fitted above per call.)
    let pair = |shapes: &[(usize, usize)], a: Mul, b: Mul| {
        pairs(&time_min(level, |pass| shapes.iter().flat_map(|&sh| [product(a)(sh, pass), product(b)(sh, pass)]).collect()))
    };
    let pair_sq = |sizes: &[usize], a: Sqr, b: Sqr| {
        pairs(&time_min(level, |pass| sizes.iter().flat_map(|&n| [square(a)(n, pass), square(b)(n, pass)]).collect()))
    };
    let tie = level.eps();

    let sq_sizes = grid(150.0, 26, 8.0); // 150 to about 1300
    let t = pair_sq(&sq_sizes, karatsuba_sqr_entry_dyn, fft_sqr_entry);
    let dense = dense_around(sweep_cutoff("mul.fft_sqr", &sq_sizes, &t, tie).1, usize::MAX);
    let t = pair_sq(&dense, karatsuba_sqr_entry_dyn, fft_sqr_entry);
    out.push(sweep_cutoff("mul.fft_sqr", &dense, &t, tie).0);
    let sizes = grid(256.0, 22, 6.0); // 256 to about 2900
    let sq_sizes: Vec<usize> = grid(500.0, 26, 8.0).into_iter().filter(|&n| n <= 3600).collect();
    let t = pair_sq(&sq_sizes, karatsuba_sqr_entry_static::<N>, ntt_sqr_entry_static::<N>);
    let dense = dense_around(sweep_cutoff("mul.static_ntt_sqr", &sq_sizes, &t, tie).1, 3600);
    let t = pair_sq(&dense, karatsuba_sqr_entry_static::<N>, ntt_sqr_entry_static::<N>);
    out.push(sweep_cutoff("mul.static_ntt_sqr", &dense, &t, tie).0);
    for (path, mults, chunking) in [("mul.ntt_karatsuba", NEAR_M, false), ("mul.ntt_chunking_karatsuba", FAR_M, true)] {
        let shapes = shapes_of(&sizes, &mults, 6000);
        let t = pair(&shapes, karatsuba_entry_static::<N>, ntt_entry_static::<N>);
        out.push(sweep_ratio(path, &shapes, &t, chunking, tie));
    }
    late_cutoffs(level, &mut out);
    ntt_gates(level, &mut out);
    out
}

/// `cargo tune equations`: Karatsuba against the FFT for dynamic products. Both entries and
/// `mul_dyn` are timed over two grids of shapes the equations weren't fitted on, and the report
/// gives the measured time lost by comparing the fitted equations (`MulEq`) and by `mul_dyn`
/// itself, with each equation's error against these timings. (Until 2026-11-11 it also reported
/// the ratio rule the equations replaced, whose two knobs are purged.)
pub fn equations(level: Level) {
    use crate::utils::mul::{mul_dyn, MulEq};
    let eq = MulEq::read();
    let held = grid(110.0, 22, 5.0);
    let sets: [(&str, Vec<(usize, usize)>); 2] = [
        ("near-square", shapes_of(&held, &[1.1, 1.35, 1.7], usize::MAX)),
        ("unbalanced", shapes_of(&held, &[2.5, 3.5, 6.0], usize::MAX)),
    ];
    println!("Karatsuba against the FFT for dynamic products ({}), measured time lost:", level.name());
    for (name, shapes) in sets {
        let t = time_min(level, |pass| {
            shapes.iter().flat_map(|&sh| [karatsuba_entry_dyn as Mul, fft_entry, mul_dyn].map(|f| product(f)(sh, pass))).collect()
        });
        // mul_dyn itself, dispatch and all, against the better forced entry at each shape.
        let whole: Vec<f64> = t.chunks(3).map(|c| c[2] / c[0].min(c[1]) - 1.0).collect();
        let whole = (whole.iter().sum::<f64>() / whole.len() as f64, whole.iter().cloned().fold(f64::MIN, f64::max));
        let t: Vec<(f64, f64)> = t.chunks(3).map(|c| (c[0], c[1])).collect();
        let eqs = lost(&t, |i| eq.kara_ns(shapes[i].0, shapes[i].1) < eq.fft_ns(shapes[i].0, shapes[i].1));
        let rms = |pred: &dyn Fn(usize, usize) -> f64, pick: &dyn Fn(&(f64, f64)) -> f64| {
            (shapes.iter().zip(&t).map(|(&(l, s), m)| (pred(l, s) / pick(m) - 1.0).powi(2)).sum::<f64>() / shapes.len() as f64).sqrt()
        };
        println!("  {name} ({} shapes):", shapes.len());
        println!("    fitted equations:         mean {:.2}%, worst {:.1}%", 100.0 * eqs.0, 100.0 * eqs.1);
        println!("    mul_dyn itself:           mean {:.2}%, worst {:.1}%   (its dispatch included)", 100.0 * whole.0, 100.0 * whole.1);
        println!(
            "    equation error here:      Karatsuba rms {:.1}%, FFT rms {:.1}%",
            100.0 * rms(&|l, s| eq.kara_ns(l, s), &|m| m.0),
            100.0 * rms(&|l, s| eq.fft_ns(l, s), &|m| m.1)
        );
    }
}

// The NTT's four parallel gates. Each acts at every level of a transform, so each is a parameter
// swept through the NTT entries on the tune's pool: the dynamic entries where mul_dyn takes them
// (past the FFT's limits, 70000 limbs here) and the static ones at a few thousand limbs.
fn ntt_gates(level: Level, out: &mut Vec<Tuned>) {
    use crate::tuning::tuner::sweep::{param_sweep, with_knob};
    use crate::utils::mul::{ntt_entry_dyn, ntt_sqr_entry_dyn};
    const N: usize = 8192;
    let entry = |w: usize| -> Probe<'static> {
        let n = [70_000usize, 70_000, 3000, 3600][w];
        let (a, b) = (pad(n, n as u64 * 3 + 1, 0), pad(n, n as u64 * 5 + 2, 0));
        let mut o = pad(2 * n - 1, 7, 0);
        match w {
            0 => Probe::new("ntt_dyn", move || {
                black_box(ntt_entry_dyn(black_box(&a), black_box(&b), &mut o));
            }),
            1 => Probe::new("ntt_sqr_dyn", move || {
                black_box(ntt_sqr_entry_dyn(black_box(&a), &mut o));
            }),
            2 => Probe::new("ntt_static", move || {
                black_box(ntt_entry_static::<N>(black_box(&a), black_box(&b), &mut o));
            }),
            _ => Probe::new("ntt_sqr_static", move || {
                black_box(ntt_sqr_entry_static::<N>(black_box(&a), &mut o));
            }),
        }
    };
    let never = 1u64 << 40;
    let gates: [(&'static str, [u64; 6]); 4] = [
        ("mul.ntt_par_conv", [128, 256, 512, 1024, 2048, never]),
        ("mul.ntt_par", [900, 1800, 3600, 7200, 14400, never]),
        ("mul.ntt_par_radix3", [3456, 6912, 13824, 27648, 55296, never]),
        ("mul.ntt_par_radix5", [3200, 6400, 12800, 25600, 51200, never]),
    ];
    for (path, values) in gates {
        let t = param_sweep(level, path, &[0, 1, 2, 3], &ints(&values), &|w, c, _| with_knob(path, c, entry(w)));
        keep("mul", out, t);
    }
}

// mul's remaining one-input decisions, each a sweep of its two code paths: the high products
// (hi_*_buf against the guarded product; columns against rows inside hi_*_buf), the middle
// products (mid_mul_buf against the transform) and the cyclic square.
fn late_cutoffs(level: Level, out: &mut Vec<Tuned>) {
    use crate::utils::mul::{
        bnm1_size, fft_mid_mul, fft_sqr_mod_bnm1, hi_mul_buf, hi_mul_buf_by_rows, hi_mul_buf_cols, hi_mul_guarded_dyn,
        hi_mul_guarded_static, hi_sqr_buf, hi_sqr_buf_by_rows, hi_sqr_buf_cols, hi_sqr_guarded_dyn, hi_sqr_guarded_static,
        mid_mul_buf, ntt_mid_mul_static, sqr_mod_bnm1_dyn,
    };
    const N: usize = 4096;
    // Sizes eight to a doubling from `lo` to `hi`.
    let grid = |lo: f64, hi: usize| -> Vec<usize> { geo(lo, 8.0).take_while(|&n| n <= hi).collect() };
    // A window of w limbs from operands of 2w and w limbs (a square's operand: w limbs).
    let hi_mul = |f: Mul, wide: usize| {
        move |w: usize, pass: usize| {
            let (a, b) = (pad(wide * w, w as u64 * 3, pass), pad(w, w as u64 * 5, pass));
            let mut o = pad(w, 7, pass);
            Probe::new("hi_mul", move || {
                black_box(f(black_box(&a), black_box(&b), &mut o));
            })
        }
    };
    let hi_sqr = |f: Sqr| {
        move |w: usize, pass: usize| {
            let a = pad(w, w as u64 * 11, pass);
            let mut o = pad(w, 13, pass);
            Probe::new("hi_sqr", move || {
                black_box(f(black_box(&a), &mut o));
            })
        }
    };
    let mid = |f: fn(&[u64], &[u64], &mut [u64]) -> (u64, u64)| {
        move |n: usize, pass: usize| {
            let (l, s) = (pad(2 * n - 1, n as u64 * 17, pass), pad(n, n as u64 * 19, pass));
            let mut o = pad(n, 23, pass);
            Probe::new("mid", move || {
                black_box(f(black_box(&l), black_box(&s), &mut o));
            })
        }
    };
    let every: Vec<usize> = (6..=200).collect();
    out.push(cutoff_sweep(level, "mul.hi_mul_row", &every, None, 0, &|w, p| [hi_mul(hi_mul_buf_cols, 1)(w, p), hi_mul(hi_mul_buf_by_rows, 1)(w, p)]));
    out.push(cutoff_sweep(level, "mul.hi_sqr_row", &every, None, 0, &|w, p| {
        [hi_sqr(hi_sqr_buf_cols)(w, p), hi_sqr(hi_sqr_buf_by_rows)(w, p)]
    }));
    // The guarded products dispatch through mul_*/sqr_*, so they run under the values tuned above.
    let (g, gs) = (grid(96.0, 1500), grid(96.0, 1300));
    out.push(cutoff_sweep(level, "mul.dyn_partial_mul", &g, Some(1500), 0, &|w, p| [hi_mul(hi_mul_buf, 2)(w, p), hi_mul(hi_mul_guarded_dyn, 2)(w, p)]));
    out.push(cutoff_sweep(level, "mul.static_partial_mul", &gs, Some(1300), 0, &|w, p| {
        [hi_mul(hi_mul_buf, 2)(w, p), hi_mul(hi_mul_guarded_static::<N>, 2)(w, p)]
    }));
    out.push(cutoff_sweep(level, "mul.dyn_partial_sqr", &g, Some(1500), 0, &|w, p| [hi_sqr(hi_sqr_buf)(w, p), hi_sqr(hi_sqr_guarded_dyn)(w, p)]));
    out.push(cutoff_sweep(level, "mul.static_partial_sqr", &gs, Some(1300), 0, &|w, p| {
        [hi_sqr(hi_sqr_buf)(w, p), hi_sqr(hi_sqr_guarded_static::<N>)(w, p)]
    }));
    // The middle products switch at n >= knob, so the knob is one past the last schoolbook size.
    out.push(cutoff_sweep(level, "mul.fft_mid", &grid(32.0, 700), Some(700), 1, &|n, p| [mid(mid_mul_buf)(n, p), mid(fft_mid_mul)(n, p)]));
    out.push(cutoff_sweep(level, "mul.ntt_mid", &grid(128.0, 2000), Some(2000), 1, &|n, p| {
        [mid(mid_mul_buf)(n, p), mid(ntt_mid_mul_static::<N>)(n, p)]
    }));
    // The cyclic square at its own sizes m = bnm1_size, a of m limbs: sqr_dyn and a fold (the
    // knob at its maximum inside the probe's entry) against the cyclic FFT.
    let mut ms: Vec<usize> = (48..=1200).map(bnm1_size).collect();
    ms.dedup();
    let saved = crate::tuning::get("mul.sqr_bnm1_fft").expect("declared");
    let (hold, back) = (
        crate::tuning::prepare("mul.sqr_bnm1_fft", &Value::Int(usize::MAX as u64)).unwrap(),
        crate::tuning::prepare("mul.sqr_bnm1_fft", &saved).unwrap(),
    );
    out.push(cutoff_sweep(level, "mul.sqr_bnm1_fft", &ms, None, 0, &|m, p| {
        let (a, a2) = (pad(m, m as u64 * 29, p), pad(m, m as u64 * 29, p));
        let (mut o, mut o2) = (pad(m, 31, p), pad(m, 37, p));
        [
            Probe::new("fold", move || {
                crate::tuning::install(hold);
                sqr_mod_bnm1_dyn(black_box(&a), &mut o);
                crate::tuning::install(back);
            }),
            Probe::new("cyclic", move || {
                fft_sqr_mod_bnm1(black_box(&a2), &mut o2);
                black_box(&o2);
            }),
        ]
    }));
}

// One of is_karatsuba's ratios (the knob at `path`) as a sweep over `shapes`: Karatsuba where its
// cost form is below ratio * fft_cost.
fn sweep_ratio(path: &'static str, shapes: &[(usize, usize)], t: &[(f64, f64)], chunking: bool, tie: f64) -> Tuned {
    use crate::utils::mul::fft_cost;
    // Each shape's (Karatsuba cost, FFT cost): neither depends on the ratio.
    let costs: Vec<(f64, f64)> =
        shapes.iter().map(|&(l, s)| (if chunking { chunking_karatsuba_cost(l, s) } else { karatsuba_cost(l, s) }, fft_cost(l, s))).collect();
    let loss = |r: f64| lost(t, |i| costs[i].0 < r * costs[i].1);
    let best = (25..=4000).map(|i| i as f64 * 0.01).map(|r| (r, loss(r))).min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap()).unwrap();
    let previous = match crate::tuning::get(path) {
        Some(Value::Float(r)) => Some((r, loss(r))),
        _ => None,
    };
    let (value, _, evidence) = keep_or_take(&format!("sweep, {} shapes", shapes.len()), "", best, previous, tie, |r| r.to_string());
    Tuned { key: crate::tuning::tuner::sweep::key_of(path), value: Value::Float(value), evidence }
}
