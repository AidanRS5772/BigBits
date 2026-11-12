//! Sections `exp` and `ln`: their algorithm parameters as parameter sweeps through the public
//! entries (the knob installed per probe) and the `_at` entries (which take the parameter). Each
//! later sweep runs under the values found before it. Static at N = 1100 only (approximate).
//! Also `theta`'s plan gate, and ln's series-against-AGM cutoff and theta term count, both on one
//! thread, where the library defines them, with its parallel bound on the pool. The static chunk
//! gates (`*.static_chunk_par_work`) keep their defaults.

use crate::tuning::tuner::check_items::EXP_FRACTION;
use crate::tuning::tuner::measure::{on_one_thread, pad, Level, Probe};
use crate::tuning::tuner::out::Tuned;
use crate::tuning::tuner::sweep::{cutoff_sweep_whole, geo, int, ints, keep, param_sweep, with_knob};
use crate::utils::trans::{
    exp_dyn, exp_dyn_at, exp_static, exp_static_at, ln_agm_dyn_at, ln_dyn_at, theta_dyn_at, theta_static_at,
    ThetaMode,
};
use std::hint::black_box;

const N: usize = 1100;

fn sizes(hi: usize) -> Vec<usize> {
    // Sizes four apart (fewer measurements: the user's decision, 2026-11-11).
    (0..20).map(|i| 4usize << (2 * i)).take_while(|&n| n <= hi).collect()
}

pub fn tune_exp(level: Level) -> Vec<Tuned> {
    let saved = crate::tuning::snapshot();
    let mut out = Vec::new();
    // e^X for a full-precision X in [0, 1) of n limbs (`EXP_FRACTION`), to n limbs.
    type Exp = fn(&[u64], i64, &mut [u64]);
    let run = |f: Exp, n: usize, p: usize| {
        let (x, mut o) = (pad(n, n as u64 * 7 + 1, p), pad(n, 3, p));
        Probe::new("exp", move || f(black_box(&x), EXP_FRACTION, &mut o))
    };
    let dynamic: Exp = |x, m, o| {
        black_box(exp_dyn(x, m, o).expect("X < 1"));
    };
    let stat: Exp = |x, m, o| {
        black_box(exp_static::<N>(x, m, o).expect("N holds the frame"));
    };
    let widths = ints(&[4, 8, 16, 32, 64]);
    for (path, f, hi) in [("exp.dyn_chunk_min_bits", dynamic, 4096), ("exp.static_chunk_min_bits", stat, 1024)] {
        let t = param_sweep(level, path, &sizes(hi), &widths, &|n, c, p| with_knob(path, c, run(f, n, p)));
        keep("exp", &mut out, t);
    }
    let squares = ints(&[0, 12, 24, 36, 48]);
    let t = param_sweep(level, "exp.dyn_sqr_bits", &sizes(4096), &squares, &|n, c, p| {
        let (x, mut o, b) = (pad(n, n as u64 * 7 + 1, p), pad(n, 3, p), int(c));
        Probe::new("exp_dyn_at", move || {
            black_box(exp_dyn_at(black_box(&x), EXP_FRACTION, &mut o, b).expect("X < 1"));
        })
    });
    keep("exp", &mut out, t);
    let t = param_sweep(level, "exp.static_sqr_bits", &sizes(1024), &squares, &|n, c, p| {
        let (x, mut o, b) = (pad(n, n as u64 * 7 + 1, p), pad(n, 3, p), int(c));
        Probe::new("exp_static_at", move || {
            black_box(exp_static_at::<N>(black_box(&x), EXP_FRACTION, &mut o, b).expect("N holds the frame"));
        })
    });
    keep("exp", &mut out, t);
    // The chunk gate (chunks x working limbs >= knob runs the chunks on the pool), through whole
    // exp_dyn calls across the sizes where it opens. Dynamic only: the static gate starts
    // scoped threads and stays at its default (approximate).
    let work = ints(&[256, 512, 768, 1280, 2048, 4096, 1 << 40]);
    let around: Vec<usize> = vec![32, 64, 96, 128, 192, 256, 512, 1024];
    let t = param_sweep(level, "exp.dyn_chunk_par_work", &around, &work, &|n, c, p| with_knob("exp.dyn_chunk_par_work", c, run(dynamic, n, p)));
    keep("exp", &mut out, t);
    crate::tuning::restore(&saved);
    out
}

pub fn tune_ln(level: Level) -> Vec<Tuned> {
    let saved = crate::tuning::snapshot();
    let mut out = Vec::new();
    // ln X by the series, for a generic X of n limbs with one integer limb, to n limbs.
    // Under the live square-root scale, which is not swept (see below).
    let a_dyn = crate::tuning::k::ln::dyn_sqrt_a();
    // X and the output, n limbs each.
    let xo = |n: usize, p: usize| (pad(n, n as u64 * 11 + 5, p), pad(n, 3, p));
    let dynamic = |n: usize, a: f64, p: usize| {
        let (x, mut o) = xo(n, p);
        Probe::new("ln_dyn_at", move || {
            black_box(ln_dyn_at(black_box(&x), 1 - n as i64, &mut o, a).expect("X > 0"));
        })
    };
    // The chunk gate, as exp's: whole series logarithms across the sizes where it opens.
    let work = ints(&[128, 256, 512, 768, 1280, 2048, 1 << 40]);
    let around: Vec<usize> = vec![32, 64, 96, 128, 192, 256, 512, 1024];
    let t = param_sweep(level, "ln.dyn_chunk_par_work", &around, &work, &|n, c, p| with_knob("ln.dyn_chunk_par_work", c, dynamic(n, a_dyn, p)));
    keep("ln", &mut out, t);
    // ln.dyn_sqrt_a and ln.static_sqrt_a are not swept: like the chunk widths, their best value
    // differs by size, and the user's choice for such knobs is the standing value. The dynamic
    // sweep's 0.5 made ln_dyn 13-26% faster at 30-64 limbs and 8.6% slower at 1040 on 8 threads,
    // and it moved the one-thread AGM cutoff below from 32 limbs to 1024, so whether that sweep
    // settled or stayed unstable flipped the whole section from run to run. The static sweep's
    // 1.0 made ln_static 12-20% faster at most sizes and 1.11x and 1.20x slower at 120 and 1040.
    // The theta-AGM logarithm against the series, for a generic X on one thread (the library
    // scales the comparison for more threads). First the AGM's theta term count, at sizes where
    // it runs; then the cutoff under that count: the series while o < knob.
    fn agm(t: usize, x: &[u64], m: i64, o: &mut [u64]) {
        macro_rules! at {
            ($($n:literal)*) => {
                match t {
                    $($n => { black_box(ln_agm_dyn_at::<$n>(x, m, o).expect("X > 0")); })*
                    _ => unreachable!("a theta term count past 16"),
                }
            };
        }
        at!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
    }
    let counts = ints(&[3, 4, 5, 6, 8, 10]);
    let t = param_sweep(level, "ln.dyn_agm_terms", &[1024, 4096], &counts, &|n, c, p| {
        let ((x, mut o), terms) = (xo(n, p), int(c) as usize);
        Probe::new("ln_agm", move || on_one_thread(|| agm(terms, &x, 1 - n as i64, &mut o)))
    });
    keep("ln", &mut out, t);
    let terms = crate::tuning::k::ln::dyn_agm_terms();
    let coarse: Vec<usize> = geo(32.0, 3.0).take_while(|&n| n <= 2048).collect();
    let t = cutoff_sweep_whole(level, "ln.dyn_agm_limbs", &coarse, 1, &|n, p| {
        let ((x, mut o), (x2, mut o2)) = (xo(n, p), xo(n, p));
        [
            Probe::new("series", move || {
                on_one_thread(|| {
                    black_box(ln_dyn_at(&x, 1 - n as i64, &mut o, a_dyn).expect("X > 0"));
                })
            }),
            Probe::new("agm", move || on_one_thread(|| agm(terms, &x2, 1 - n as i64, &mut o2))),
        ]
    });
    keep("ln", &mut out, t);
    // The pair's parallel half, on the tune's pool: the AGM (the lower side here) while
    // o < knob, then the series, whose chunks split across threads from some size.
    let coarse: Vec<usize> = geo(32.0, 3.0).take_while(|&n| n <= 4096).collect();
    let t = cutoff_sweep_whole(level, "ln.dyn_agm_par_until", &coarse, 1, &|n, p| {
        let ((x, mut o), (x2, mut o2)) = (xo(n, p), xo(n, p));
        [
            Probe::new("agm", move || agm(terms, &x2, 1 - n as i64, &mut o2)),
            Probe::new("series", move || {
                black_box(ln_dyn_at(&x, 1 - n as i64, &mut o, a_dyn).expect("X > 0"));
            }),
        ]
    });
    keep("ln", &mut out, t);
    crate::tuning::restore(&saved);
    out
}

/// Section `theta`: the plan gate, the serial plan against the threaded parallel plan through
/// `theta_*_at`, in the frames ln's AGM gives it (T = 5 dynamic, 6 static; q's lead from the
/// precision). The parallel plan runs from frame limbs >= knob, and the frame is the output and a
/// guard limb, so the knob is two past the last serial size.
pub fn tune_theta(level: Level) -> Vec<Tuned> {
    // x of n limbs with its top bit set, and the shift that gives q the lead the AGM would.
    let input = |n: usize, t: u64, p: usize| {
        let mut x = pad(n, n as u64 * 13 + 7, p).to_vec();
        x[n - 1] |= 1 << 63;
        let lead = (64 * n as u64 / (4 * (t + 1) * (t + 1))).max(2);
        (x, 64 * n as u64 + lead)
    };
    let grid = |hi: usize| -> Vec<usize> { geo(32.0, 3.0).take_while(|&n| n <= hi).collect() };
    let dynamic = cutoff_sweep_whole(level, "theta.dyn_par_limbs", &grid(4096), 2, &|n, p| {
        let at = |mode: ThetaMode| {
            let ((x, shift), mut th2, mut th3) = (input(n, 5, p), pad(n, 3, p), pad(n + 1, 5, p));
            Probe::new("theta_dyn", move || {
                black_box(theta_dyn_at::<5>(black_box(&x), shift, &mut th2, &mut th3, mode));
            })
        };
        [at(ThetaMode::Serial), at(ThetaMode::Parallel { threads: true })]
    });
    let stat = cutoff_sweep_whole(level, "theta.static_par_limbs", &grid(1000), 2, &|n, p| {
        let at = |mode: ThetaMode| {
            let ((x, shift), mut th2, mut th3) = (input(n, 6, p), pad(n, 3, p), pad(n + 1, 5, p));
            Probe::new("theta_static", move || {
                black_box(theta_static_at::<1100, 6>(black_box(&x), shift, &mut th2, &mut th3, mode).expect("N holds the frame"));
            })
        };
        [at(ThetaMode::Serial), at(ThetaMode::Parallel { threads: true })]
    });
    vec![dynamic, stat]
}
