//! Section `mul`: the schoolbook kernels' equations (the full product's, the high product's
//! two), fitted where each competes; `mul.kara`, the schoolbook-against-Karatsuba table,
//! measured (`kara_table`); and under it recursive Karatsuba's and the FFT's equations, which
//! `mul_dyn` compares past the table, with the one-input cutoffs as sweeps (`tune_late`).

use crate::tuning::resolve::Value;
use crate::tuning::tuner::design::{products, Design};
use crate::tuning::tuner::fit::fit_named;
use crate::tuning::tuner::measure::{pad, time_fit, time_min, Level, PadBuf, Probe};
use crate::tuning::tuner::out::{push_fit, Tuned};
use crate::tuning::tuner::sweep::{cutoff_sweep, dense_around, geo, geo_to, ints, keep, keep_or_take, lost, pairs, sweep_cutoff};
use crate::tuning::tuner::tables::{self, Plan, Row, Spec};
use crate::utils::mul::{
    chunking_karatsuba_cost, karatsuba_entry_static, karatsuba_sqr_entry_static, ntt_entry_static, ntt_sqr_entry_static, fft_entry,
    fft_mul_size, fft_sqr_entry, fft_transform_work, hi_mul_buf_by_rows, hi_mul_buf_cols, hi_window_products, karatsuba_cost,
    karatsuba_entry_dyn, karatsuba_sqr_core, karatsuba_sqr_entry_dyn, mul_buf, school_at, sqr_buf,
};
use std::hint::black_box;

type Mul = fn(&[u64], &[u64], &mut [u64]) -> u64;
type Sqr = fn(&[u64], &mut [u64]) -> u64;
// A middle product: long[2n - 1] by short[n] into out[n].
type Mid = fn(&[u64], &[u64], &mut [u64]) -> (u64, u64);

// Rows of `mul.kara` below this are schoolbook throughout.
const KARA_FROM: usize = 8;
// The table ends after this many rows in a row that are Karatsuba throughout (the last stands
// for every wider short side), and at `KARA_ROWS` whatever it finds.
const KARA_RUN: usize = 6;
const KARA_ROWS: usize = 512;

// `mul.kara`'s rows, a short side s at a time. A row's Karatsuba runs on the rows before it, so
// a row is measured in three steps, each under what is settled so far: the square, the
// near-square shapes (to l = 2s - 2), the chunked ones (to 4s). The entry asks the table about
// its own shape first, so a step's indices are Karatsuba's in the live table while timed.
struct KaraSpec {
    s: usize,
    step: u8,
    // The row in hand's cells so far; the rows done; how many of the last are Karatsuba throughout.
    cells: Vec<tables::Cell>,
    rows: Vec<Row>,
    run: usize,
    eps: f64,
}

impl KaraSpec {
    // A row's grid: every l from s to 4s while that is 40 or fewer, else sixteen to the octave, with
    // the two either side of where a level gives way to chunking.
    fn grid(s: usize) -> Vec<u32> {
        let (s, top) = (s as u32, 4 * s as u32);
        let mut g: Vec<u32> = if top - s < 40 { (s..=top).collect() } else { tables::octave_steps(s, top, 16.0) };
        g.extend([2 * s - 2, 2 * s - 1, top]);
        g.sort_unstable();
        g.dedup();
        g
    }

    // The table of the rows done and the row in hand, which with `forced` is Karatsuba's from that l.
    fn table(&self, forced: Option<u32>) -> Vec<u32> {
        let mut t: Vec<u32> = (1..KARA_FROM as u32).flat_map(|s| [s, s, 0]).collect();
        t.extend(tables::assemble(&self.rows, |_, _| false));
        let mut points = tables::row(&self.cells, None, self.eps, tables::share());
        if let Some(l) = forced.filter(|_| points.last().is_none_or(|p| p.1 != 1)) {
            points.push((l, 1));
        }
        for (l, pick) in points {
            t.extend([self.s as u32, l, pick]);
        }
        t
    }
}

impl Spec for KaraSpec {
    fn equations(&self) -> Option<u32> {
        None
    }

    fn group(&mut self) -> Option<Vec<(u32, Vec<u32>)>> {
        if self.run >= KARA_RUN || self.s > KARA_ROWS {
            return None;
        }
        let s = self.s as u32;
        let sizes: Vec<u32> = Self::grid(self.s)
            .into_iter()
            .filter(|&l| match self.step {
                0 => l == s,
                1 => l > s && l <= 2 * s - 2,
                _ => l >= 2 * s - 1,
            })
            .collect();
        crate::tuning::set("mul.kara", Value::Table(self.table(sizes.first().copied()))).expect("a valid table");
        Some(vec![(s, sizes)])
    }

    fn probes(&self, key: u32, size: u32, pass: usize, probes: &mut Vec<Probe<'static>>) -> Vec<Plan> {
        for f in [mul_buf as Mul, karatsuba_entry_dyn] {
            probes.push(product(f)((size as usize, key as usize), pass));
        }
        vec![Plan { picks: vec![0, 1], ..Plan::default() }]
    }

    fn settled(&mut self, rows: Vec<Row>) {
        self.cells.extend(rows.into_iter().flat_map(|r| r.cells));
        if self.step == 2 {
            let (s, cells) = (self.s as u32, std::mem::take(&mut self.cells));
            let points = tables::row(&cells, None, self.eps, tables::share());
            self.run = if points == [(s, 1)] { self.run + 1 } else { 0 };
            self.rows.push(Row { key: s, sizes: Self::grid(self.s), points, cells });
            self.s += 1;
        }
        self.step = (self.step + 1) % 3;
    }
}

/// The `mul.kara` table and its evidence: at each index (s, l), `mul_buf` against
/// `karatsuba_entry_dyn`. The table decides at every size, so each row is measured over its
/// whole grid; a pick stays while within the tie band of the other.
pub fn kara_table(level: Level) -> (Vec<u32>, String) {
    let saved = crate::tuning::snapshot();
    let mut spec = KaraSpec { s: KARA_FROM, step: 0, cells: Vec::new(), rows: Vec::new(), run: 0, eps: level.eps() };
    let tally = tables::measure(level, &mut spec, level.eps());
    crate::tuning::restore(&saved);
    // The all-Karatsuba run the table ends on: its first row stands for the rest.
    let keep = spec.rows.len() - spec.run.saturating_sub(1).min(spec.rows.len());
    spec.rows.truncate(keep);
    let t = spec.table(None);
    let first = |m: u32| spec.rows.iter().find(|r| tables::pick_at(&r.points, m * r.key) == Some(1)).map_or(0, |r| r.key);
    let ev = format!(
        "measured (table): Karatsuba from {} for squares and {} at l = 4s; every row from s = {} is Karatsuba throughout; {} change points; {}",
        first(1),
        first(4),
        spec.rows.last().map_or(0, |r| r.key),
        t.len() / 3,
        tally.text()
    );
    (t, ev)
}

pub fn tune(level: Level) -> Vec<Tuned> {
    let mut out = Vec::new();
    let n = level.fit_shapes(60);
    // mul_buf, fitted where it competes with Karatsuba (10 to about 100 limbs, l to 4s): a wider
    // range fits the crossover worse.
    fit_lin(
        level,
        &mut out,
        &["school_scale", "school_shift"],
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
    // hi_mul_buf's two kernels, each forced on windows of l limbs by an operand of 2 limbs to the
    // window: the window's limb products, a term a row (rows kernel) or a column, and a shift. The
    // columns kernel to a third past `mul.hi_mul_row`: a wider range fits it worse where it runs.
    let cols_to = (4 * crate::tuning::k::mul::hi_mul_row() / 3).clamp(16, 256);
    let windows = |from: f64, to: f64| {
        Design::of(0, n, (from, to), (1.0, 2.0), |l, y| {
            let l = l.round() as usize;
            Some((l, ((2.0 * (l as f64 / 2.0).powf(y.log2())).round() as usize).clamp(2, l)))
        })
    };
    let window = |f: Mul| {
        move |(l, s): (usize, usize), pass: usize| binary(format!("hi window {l}x{s}"), f, pad(l, l as u64 * 73, pass), pad(s, s as u64 * 79, pass), pad(l, 37, pass))
    };
    fit_lin(
        level,
        &mut out,
        &["hi_rows_scale", "hi_rows_row", "hi_rows_shift"],
        "hi_mul_buf_by_rows, windows of 12-342 limbs by 2 limbs to the window; W = its limb products, s, 1",
        &windows(12.0, 342.0),
        |(l, s)| vec![hi_window_products(l as f64, s as f64), s as f64, 1.0],
        window(hi_mul_buf_by_rows),
    );
    fit_lin(
        level,
        &mut out,
        &["hi_cols_scale", "hi_cols_col", "hi_cols_shift"],
        &format!("hi_mul_buf_cols, windows of 2-{cols_to} limbs by 2 limbs to the window; W = its limb products, l, 1"),
        &windows(2.0, cols_to as f64),
        |(l, s)| vec![hi_window_products(l as f64, s as f64), l as f64, 1.0],
        window(hi_mul_buf_cols),
    );

    // mul.karatsuba_sqr, a sweep: sqr_buf against one level over sqr_buf leaves, 40 to 140 limbs.
    let sizes: Vec<usize> = (40..=140).collect();
    let saved = crate::tuning::snapshot();
    crate::tuning::set("mul.karatsuba_sqr", Value::Int(usize::MAX as u64)).unwrap();
    let t = time_min(level, |pass| {
        sizes
            .iter()
            .flat_map(|&n| {
                let h = n.div_ceil(2);
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
    out.push(sweep_cutoff("mul.karatsuba_sqr", &sizes, &pairs(&t), 0, level.eps()).0);
    out
}

// A product's probe through `f`; a square's.
fn product(f: Mul) -> impl Fn((usize, usize), usize) -> Probe<'static> {
    move |(l, s), pass| binary("product", f, pad(l, l as u64 * 53, pass), pad(s, s as u64 * 59, pass), pad(l + s - 1, 19, pass))
}

fn square(f: Sqr) -> impl Fn(usize, usize) -> Probe<'static> {
    move |n, pass| unary("square", f, pad(n, n as u64 * 61, pass), pad(2 * n - 1, 23, pass))
}

// `f` on two operands into `o`, and on one, as probes.
fn binary<R: 'static>(name: impl Into<String>, f: fn(&[u64], &[u64], &mut [u64]) -> R, a: PadBuf, b: PadBuf, mut o: PadBuf) -> Probe<'static> {
    Probe::new(name, move || {
        black_box(f(black_box(&a), black_box(&b), &mut o));
    })
}

fn unary(name: &str, f: Sqr, a: PadBuf, mut o: PadBuf) -> Probe<'static> {
    Probe::new(name, move || {
        black_box(f(black_box(&a), &mut o));
    })
}

// `steps` sizes from `lo`, `per_octave` to a doubling.
fn grid(lo: f64, steps: usize, per_octave: f64) -> Vec<usize> {
    geo(lo, per_octave).take(steps).collect()
}

// Each size s as the shapes (m s, s) for the multiples `mults`, with at most `cap` limbs in all.
fn shapes_of(sizes: &[usize], mults: &[f64], cap: usize) -> Vec<(usize, usize)> {
    sizes.iter().flat_map(|&s| mults.iter().map(move |m| ((s as f64 * m).round() as usize, s))).filter(|&(l, s)| l + s <= cap).collect()
}

// The static Karatsuba-to-NTT ratio sweeps' multiples of s: near-square, and unbalanced (chunking).
const NEAR_M: [f64; 4] = [1.0, 1.2, 1.5, 1.9];
const FAR_M: [f64; 4] = [2.0, 3.0, 4.0, 8.0];

// Fits t = b . w to the probes of `design`'s shapes (`feats` gives a shape's regressors),
// judged on the shapes held out, and records the knobs `keys` (the scale first) with `what`
// in their evidence.
fn fit_lin(
    level: Level,
    out: &mut Vec<Tuned>,
    keys: &[&'static str],
    what: &str,
    design: &Design<(usize, usize)>,
    feats: impl Fn((usize, usize)) -> Vec<f64>,
    probe: impl Fn((usize, usize), usize) -> Probe<'static>,
) {
    let timed = time_fit(level, |pass| design.shapes.iter().map(|&s| probe(s, pass)).collect());
    let rows: Vec<Vec<f64>> = design.shapes.iter().map(|&s| feats(s)).collect();
    let name = keys[0].strip_suffix("_scale").unwrap_or(keys[0]);
    let fit = fit_named(&format!("mul.{name}: {what}"), keys, &rows, &timed.ns, &vec![0.0; rows.len()], &design.held, Some(timed.noise));
    push_fit(out, keys, &fit, fit.evidence() + " (" + what + ")");
}

/// Second phase, once `mul.kara` is in force: recursive Karatsuba's and the FFT's equations
/// through their real entries, then the cutoffs.
pub fn tune_late(level: Level) -> Vec<Tuned> {
    let mut out = Vec::new();
    // Recursive Karatsuba in each regime: its work at the exponent log2 3, plus schoolbook's l s
    // for its leaves.
    let n = level.fit_shapes(60);
    // Fitted on the shapes they are read for: not where `mul.kara` says schoolbook.
    let kara = crate::tuning::k::mul::kara();
    let read = move |l: usize, s: usize| !school_at(kara, l, s);
    fit_lin(
        level,
        &mut out,
        &["kara_rec_scale", "kara_rec_school"],
        "karatsuba_entry_dyn, s 96-1300, l from s to 2s - 2, where mul.kara doesn't say schoolbook; W = karatsuba_cost, l s",
        &products(n, (96, 1300), (1.0, 2.0), |l, s| s > l.div_ceil(2) && read(l, s)),
        |(l, s)| vec![karatsuba_cost(l, s), (l * s) as f64],
        product(karatsuba_entry_dyn),
    );
    fit_lin(
        level,
        &mut out,
        &["chunk_rec_scale", "chunk_rec_school"],
        "karatsuba_entry_dyn, s 96-650, l from 2s to 8s, where mul.kara doesn't say schoolbook; W = chunking_karatsuba_cost, l s",
        &products(n, (96, 650), (2.0, 8.0), |l, s| s <= l.div_ceil(2) && read(l, s)),
        |(l, s)| vec![chunking_karatsuba_cost(l, s), (l * s) as f64],
        product(karatsuba_entry_dyn),
    );
    fit_lin(
        level,
        &mut out,
        &["fft_scale"],
        "fft_entry, s 96-2200, l from s to 8s, where mul.kara doesn't say schoolbook; W = its transform's work",
        &products(n, (96, 2200), (1.0, 8.0), read),
        |(l, s)| vec![fft_transform_work(fft_mul_size(l, s))],
        product(fft_entry),
    );

    // Live before the sweeps below, whose entries dispatch through mul_dyn's comparison of them.
    for t in &out {
        t.set_live("mul").expect("a fitted value is valid");
    }

    // Static entries are swept at one N: approximate values.
    const N: usize = 8192;

    // The one-input and static Karatsuba-to-transform decisions: both entries timed at each shape,
    // the value with the least time lost.
    let pair = |shapes: &[(usize, usize)], a: Mul, b: Mul| {
        pairs(&time_min(level, |pass| shapes.iter().flat_map(|&sh| [product(a)(sh, pass), product(b)(sh, pass)]).collect()))
    };
    let pair_sq = |sizes: &[usize], a: Sqr, b: Sqr| {
        pairs(&time_min(level, |pass| sizes.iter().flat_map(|&n| [square(a)(n, pass), square(b)(n, pass)]).collect()))
    };
    let tie = level.eps();

    // A square cutoff: a coarse pass over `sizes`, then a dense one around its choice (to `cap`).
    let mut sqr_cutoff = |path: &'static str, sizes: &[usize], cap: usize, a: Sqr, b: Sqr| {
        let dense = dense_around(sweep_cutoff(path, sizes, &pair_sq(sizes, a, b), 0, tie).1, cap);
        out.push(sweep_cutoff(path, &dense, &pair_sq(&dense, a, b), 0, tie).0);
    };
    sqr_cutoff("mul.fft_sqr", &grid(150.0, 26, 8.0), usize::MAX, karatsuba_sqr_entry_dyn, fft_sqr_entry); // 150 to about 1300
    let sq_sizes: Vec<usize> = grid(500.0, 26, 8.0).into_iter().filter(|&n| n <= 3600).collect();
    sqr_cutoff("mul.static_ntt_sqr", &sq_sizes, 3600, karatsuba_sqr_entry_static::<N>, ntt_sqr_entry_static::<N>);
    let sizes = grid(256.0, 22, 6.0); // 256 to about 2900
    for (path, mults, chunking) in [("mul.ntt_karatsuba", NEAR_M, false), ("mul.ntt_chunking_karatsuba", FAR_M, true)] {
        let shapes = shapes_of(&sizes, &mults, 6000);
        let t = pair(&shapes, karatsuba_entry_static::<N>, ntt_entry_static::<N>);
        out.push(sweep_ratio(path, &shapes, &t, chunking, tie));
    }
    late_cutoffs(level, &mut out);
    ntt_gates(level, &mut out);
    out
}

/// `cargo tune equations`: Karatsuba, the FFT and `mul_dyn` timed on shapes the equations weren't
/// fitted on; the time lost by the equations' comparison and by `mul_dyn`, and each equation's error.
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
        // mul_dyn, dispatch and all, against the better forced entry.
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

// The NTT's four parallel gates, each swept through the NTT entries on the tune's pool: the
// dynamic ones at 70000 limbs (past the FFT's limits), the static at a few thousand.
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

// The remaining one-input decisions, each a sweep of its two code paths: high products, middle
// products, the cyclic square.
fn late_cutoffs(level: Level, out: &mut Vec<Tuned>) {
    use crate::utils::mul::{
        bnm1_size, fft_mid_mul, fft_sqr_mod_bnm1, hi_mul_buf, hi_mul_buf_by_rows, hi_mul_buf_cols, hi_mul_guarded_dyn,
        hi_mul_guarded_static, hi_sqr_buf, hi_sqr_buf_by_rows, hi_sqr_buf_cols, hi_sqr_guarded_dyn, hi_sqr_guarded_static,
        mid_mul_buf, ntt_mid_mul_static, sqr_mod_bnm1_dyn,
    };
    const N: usize = 4096;
    let grid = |lo: f64, hi: usize| geo_to(lo, 8.0, hi); // eight sizes to a doubling
    // A window of w limbs from operands of `wide` w and w limbs; a square's from w limbs.
    let hi_mul = |f: Mul, wide: usize| move |w: usize, pass: usize| binary("hi_mul", f, pad(wide * w, w as u64 * 3, pass), pad(w, w as u64 * 5, pass), pad(w, 7, pass));
    let hi_sqr = |f: Sqr| move |w: usize, pass: usize| unary("hi_sqr", f, pad(w, w as u64 * 11, pass), pad(w, 13, pass));
    let mid = |f: Mid| move |n: usize, pass: usize| binary("mid", f, pad(2 * n - 1, n as u64 * 17, pass), pad(n, n as u64 * 19, pass), pad(n, 23, pass));
    let every: Vec<usize> = (6..=200).collect();
    out.push(cutoff_sweep(level, "mul.hi_mul_row", &every, None, 0, &|w, p| [hi_mul(hi_mul_buf_cols, 1)(w, p), hi_mul(hi_mul_buf_by_rows, 1)(w, p)]));
    out.push(cutoff_sweep(level, "mul.hi_sqr_row", &every, None, 0, &|w, p| {
        [hi_sqr(hi_sqr_buf_cols)(w, p), hi_sqr(hi_sqr_buf_by_rows)(w, p)]
    }));
    // The guarded products dispatch through mul_*/sqr_*: they run under the values tuned above.
    let (g, gs) = (grid(96.0, 1500), grid(96.0, 1300));
    out.push(cutoff_sweep(level, "mul.dyn_partial_mul", &g, Some(1500), 0, &|w, p| [hi_mul(hi_mul_buf, 2)(w, p), hi_mul(hi_mul_guarded_dyn, 2)(w, p)]));
    out.push(cutoff_sweep(level, "mul.static_partial_mul", &gs, Some(1300), 0, &|w, p| {
        [hi_mul(hi_mul_buf, 2)(w, p), hi_mul(hi_mul_guarded_static::<N>, 2)(w, p)]
    }));
    out.push(cutoff_sweep(level, "mul.dyn_partial_sqr", &g, Some(1500), 0, &|w, p| [hi_sqr(hi_sqr_buf)(w, p), hi_sqr(hi_sqr_guarded_dyn)(w, p)]));
    out.push(cutoff_sweep(level, "mul.static_partial_sqr", &gs, Some(1300), 0, &|w, p| {
        [hi_sqr(hi_sqr_buf)(w, p), hi_sqr(hi_sqr_guarded_static::<N>)(w, p)]
    }));
    // The middle products switch at n >= knob: one past the last schoolbook size.
    out.push(cutoff_sweep(level, "mul.fft_mid", &grid(32.0, 700), Some(700), 1, &|n, p| [mid(mid_mul_buf)(n, p), mid(fft_mid_mul)(n, p)]));
    out.push(cutoff_sweep(level, "mul.ntt_mid", &grid(128.0, 2000), Some(2000), 1, &|n, p| {
        [mid(mid_mul_buf)(n, p), mid(ntt_mid_mul_static::<N>)(n, p)]
    }));
    // The cyclic square at m = bnm1_size: sqr_dyn and a fold (the knob at its maximum inside the
    // probe) against the cyclic FFT.
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

// One of is_karatsuba's ratios as a sweep: Karatsuba where its cost is below ratio * fft_cost.
fn sweep_ratio(path: &'static str, shapes: &[(usize, usize)], t: &[(f64, f64)], chunking: bool, tie: f64) -> Tuned {
    use crate::utils::mul::fft_cost;
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
