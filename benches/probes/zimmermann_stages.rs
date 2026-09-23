//! Zimmermann stage timing and forced division experiments, outside production.
//! Includes the actual sqrt source so its private core need not be copied or
//! exposed solely for measurement. The measured outer wrapper follows its
//! dynamic wrapper, adding optional timers and a zero-shift denormalization skip.
//! Build after `cargo build --profile prof` (substitute the rlib's actual hash):
//! rustc --edition=2021 -O -g -C lto=thin -C codegen-units=1 \
//!   benches/probes/zimmermann_stages.rs \
//!   --extern big_bits=target/prof/deps/libbig_bits-HASH.rlib \
//!   -L dependency=target/prof/deps -o target/zimmermann_sqrt_profiles/stages_probe
pub use big_bits::utils;
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

#[derive(Default)]
struct Stats {
    division_ns: u128,
    square_ns: u128,
    divisions: u64,
    squares: u64,
}

#[allow(dead_code, unused_imports)]
mod measured {
    include!("../../src/utils/sqrt.rs");
    use super::Stats;
    use crate::utils::div::{div_prepared_dyn, div_rem_prepared_dyn, division_preflight, DivAlg};
    use std::time::Instant;

    // BACKEND overrides only divisors wider than the production Knuth cutoff.
    // 0 = production dispatch, 1 = Knuth, 2 = BZ, 3 = NR,
    // 4 = direct all-ones square, 5 = Knuth for remainder divisions up to 257 limbs.
    pub fn run<const BACKEND: u8, const SKIP_NOOP: bool, const TIMED: bool>(
        x: &mut [u64],
        s: &mut [u64],
        mode: u8,
        stats: &mut Stats,
    ) {
        let output = match mode {
            0 => SqrtOutput::RootRem,
            1 => SqrtOutput::Root,
            2 => SqrtOutput::ApproxRoot,
            _ => unreachable!(),
        };
        assert!(s.len() >= output.dyn_cutoff());
        let x_len = x.len();
        let sh = x[x_len - 1].leading_zeros() as u8 & !1_u8;
        shl_buf(x, sh);
        let mut div = |n: &mut [u64], d: &[u64], q: &mut [u64], remainder: bool| {
            let start = if TIMED { Some(Instant::now()) } else { None };
            let backend = if BACKEND == 5 {
                if remainder && d.len() <= 257 {
                    1
                } else {
                    0
                }
            } else {
                BACKEND
            };
            let result = if backend == 0 || backend == 4 || d.len() <= crate::utils::BZ_CUTOFF {
                if remainder {
                    div_rem_dyn(n, d, q)
                } else {
                    div_dyn(n, d, q)
                }
            } else if let Some(request) = division_preflight(n, d, q) {
                match (backend, remainder) {
                    (1, false) => div_prepared_dyn(n, d, q, request, DivAlg::Knuth),
                    (1, true) => div_rem_prepared_dyn(n, d, q, request, DivAlg::Knuth),
                    (2, false) => div_prepared_dyn(n, d, q, request, DivAlg::BZ),
                    (2, true) => div_rem_prepared_dyn(n, d, q, request, DivAlg::BZ),
                    (3, false) => div_prepared_dyn(n, d, q, request, DivAlg::NR),
                    (3, true) => div_rem_prepared_dyn(n, d, q, request, DivAlg::NR),
                    _ => unreachable!(),
                }
            } else {
                0
            };
            if let Some(start) = start {
                stats.division_ns += start.elapsed().as_nanos();
                stats.divisions += 1;
            }
            result
        };
        let mut sqr = |value: &[u64], out: &mut [u64]| {
            let start = if TIMED { Some(Instant::now()) } else { None };
            let result = if BACKEND == 4 && value.iter().all(|&limb| limb == u64::MAX) {
                // (B^k - 1)^2 = B^(2k) - 2*B^k + 1. Production already
                // knows this case from its saturated flag; the probe scans
                // the callback input to avoid modifying the included core.
                let k = value.len();
                out[..k].fill(0);
                out[0] = 1;
                out[k] = u64::MAX - 1;
                out[k + 1..2 * k].fill(u64::MAX);
                out[2 * k..].fill(0);
                0
            } else {
                sqr_dyn(value, out)
            };
            if let Some(start) = start {
                stats.square_ns += start.elapsed().as_nanos();
                stats.squares += 1;
            }
            result
        };
        zimmerman_sqrt_entry(
            x,
            s,
            output,
            &mut div,
            &mut sqr,
            &mut |square_len, finish| {
                let mut guard = ScratchGuard::acquire();
                finish(guard.get(square_len));
            },
        );
        if output == SqrtOutput::RootRem {
            if !SKIP_NOOP || sh != 0 {
                sqrt_denormalization(x, s, sh);
            }
        } else {
            shr_buf(s, sh / 2);
        }
    }
}

type Candidate = fn(&mut [u64], &mut [u64], u8, &mut Stats);
fn production(x: &mut [u64], s: &mut [u64], mode: u8, _: &mut Stats) {
    match mode {
        0 => utils::sqrt::sqrt_dyn(x, s),
        1 => utils::sqrt::sqrt_only_dyn(x, s),
        2 => utils::sqrt::sqrt_approx_dyn(x, s),
        _ => unreachable!(),
    }
}
const CANDIDATES: [(&str, Candidate); 8] = [
    ("production", production),
    ("mirror", measured::run::<0, false, false>),
    ("skip_noop", measured::run::<0, true, false>),
    ("knuth", measured::run::<1, false, false>),
    ("bz", measured::run::<2, false, false>),
    ("nr", measured::run::<3, false, false>),
    ("max_square", measured::run::<4, false, false>),
    ("hybrid", measured::run::<5, false, false>),
];

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn inputs(n: usize, width: usize, pattern: &str, count: usize) -> Vec<Vec<u64>> {
    let mut state = 0x5351_5254_5052_4f46u64;
    assert!(n < width && width <= 2 * n);
    (0..count)
        .map(|_| {
            let mut x: Vec<_> = (0..width).map(|_| random(&mut state)).collect();
            match pattern {
                "random" => x[width - 1] |= 1 << 63,
                "shifted" => x[width - 1] = 1,
                "ones" => x.fill(u64::MAX),
                "power" => {
                    x.fill(0);
                    x[width - 1] = 1 << 62;
                }
                _ => unreachable!(),
            }
            x
        })
        .collect()
}

fn validate() {
    let mut count = 0;
    for n in [17, 32, 64, 128, 256, 512, 1024, 2048] {
        for width in [n + 1, 2 * n - 1, 2 * n] {
            for pattern in ["random", "shifted", "ones", "power"] {
                for x in inputs(n, width, pattern, 4) {
                    let mut exact_x = x.clone();
                    let mut exact_s = vec![0; n];
                    utils::sqrt::sqrt_dyn(&mut exact_x, &mut exact_s);
                    let mut upper = exact_s.clone();
                    let upper_overflow = utils::utils::inc_buf(&mut upper);
                    for mode in 0..3 {
                        for &(name, f) in &CANDIDATES[1..] {
                            let mut work = x.clone();
                            let mut s = vec![u64::MAX; n];
                            f(&mut work, &mut s, mode, &mut Stats::default());
                            if mode == 2 {
                                assert!(
                                    s == exact_s || (!upper_overflow && s == upper),
                                    "{name}: n={n}, width={width}, {pattern}, approximate root"
                                );
                            } else {
                                assert_eq!(
                                    s, exact_s,
                                    "{name}: n={n}, width={width}, {pattern}, mode={mode}"
                                );
                                if mode == 0 {
                                    assert_eq!(work, exact_x, "{name}: remainder");
                                }
                            }
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("PASS: {count} candidate root/remainder comparisons with production");
}

fn time(f: Candidate, values: &[Vec<u64>], n: usize, mode: u8, duration: Duration) -> f64 {
    let mut x = vec![0; values[0].len()];
    let mut s = vec![0; n];
    let mut stats = Stats::default();
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed() < duration {
        for value in values {
            x.copy_from_slice(black_box(value));
            f(black_box(&mut x), black_box(&mut s), mode, &mut stats);
            black_box((&x, &s));
        }
        count += values.len();
    }
    start.elapsed().as_nanos() as f64 / count as f64
}

fn main() {
    let mut pattern = String::from("random");
    let mut sizes = vec![32, 64, 128, 256, 512, 1024, 2048];
    let mut modes = vec![0, 1, 2];
    let mut check_only = false;
    let mut variants: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pattern" => pattern = args.next().unwrap(),
            "--sizes" => {
                sizes = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|n| n.parse().unwrap())
                    .collect()
            }
            "--mode" => modes = vec![args.next().unwrap().parse().unwrap()],
            "--check-only" => check_only = true,
            "--variants" => variants = Some(args.next().unwrap()),
            _ => panic!("unknown argument {arg}"),
        }
    }
    validate();
    if check_only {
        return;
    }
    let candidates: Vec<_> = CANDIDATES
        .iter()
        .copied()
        .filter(|(name, _)| {
            variants
                .as_ref()
                .is_none_or(|v| v.split(',').any(|s| s == *name))
        })
        .collect();
    assert!(!candidates.is_empty());
    eprintln!("pattern={pattern}, sizes={sizes:?}, modes={modes:?}");
    println!("kind,limbs,mode,variant,median_ns,min_ns,max_ns,division_pct,square_pct,divisions_per_call,squares_per_call");
    for n in sizes {
        let values = inputs(n, 2 * n, &pattern, 64);
        for &mode in &modes {
            let mut times = vec![[0.0_f64; 5]; candidates.len()];
            for &(_, f) in &candidates {
                time(f, &values, n, mode, Duration::from_millis(30));
            }
            for round in 0..5 {
                for step in 0..candidates.len() {
                    let i = (round + step) % candidates.len();
                    times[i][round] =
                        time(candidates[i].1, &values, n, mode, Duration::from_millis(70));
                }
            }
            for (i, &(name, _)) in candidates.iter().enumerate() {
                times[i].sort_by(f64::total_cmp);
                println!(
                    "timing,{n},{mode},{name},{:.3},{:.3},{:.3},,,,",
                    times[i][2], times[i][0], times[i][4]
                );
            }
            let mut x = vec![0; 2 * n];
            let mut s = vec![0; n];
            let mut stats = Stats::default();
            let mut count = 0;
            let start = Instant::now();
            while start.elapsed() < Duration::from_millis(150) {
                for value in &values {
                    x.copy_from_slice(value);
                    measured::run::<0, false, true>(&mut x, &mut s, mode, &mut stats);
                    black_box((&x, &s));
                }
                count += values.len();
            }
            let elapsed = start.elapsed().as_nanos() as f64;
            println!(
                "stages,{n},{mode},auto,{:.3},,,{:.3},{:.3},{:.3},{:.3}",
                elapsed / count as f64,
                100.0 * stats.division_ns as f64 / elapsed,
                100.0 * stats.square_ns as f64 / elapsed,
                stats.divisions as f64 / count as f64,
                stats.squares as f64 / count as f64
            );
        }
    }
}
