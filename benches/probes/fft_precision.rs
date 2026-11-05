//! Measures the 16-bit FFT entries' rounding against their round-off check (FFT_ROUNDOFF_LIMIT and
//! FFT_COEFF_LIMIT) and dispatch limits (FFT_16BIT_CUTOFF, FFT_SQR_16BIT_CUTOFF,
//! FFT_MID_16BIT_CUTOFF, FFT_BNM1_16BIT_CUTOFF). For every FFT transform size in a range it runs
//! `count` operands at that size's widest shape through the unchecked (raw) entry and reports, per
//! size:
//! - trips: calls the round-off check redoes on the NTT (FftRounding::trips: the rounding error
//!   passes FFT_ROUNDOFF_LIMIT or the largest coefficient reaches FFT_COEFF_LIMIT), over: calls
//!   whose largest coefficient reaches FFT_COEFF_LIMIT, and t30/t35, calls whose error passes 0.30
//!   and 0.35, for the tail's shape;
//! - wrong: calls whose raw result is wrong, against the residue verifier (products, squares) or
//!   the NTT (middle products, cyclic squares; skipped by `no-oracle`), and under: the wrong calls
//!   below FFT_COEFF_LIMIT, which only the rounding error can catch;
//! - missed: wrong calls that did not trip, which the check must never allow;
//! - err: the largest rounding error, peak: log2 of the largest coefficient, and min_wrong: the
//!   smallest error among the wrong calls under the size gate.
//! Modes: `worst` cycles the worst-case kinds (all ones, a step, random digits all at least
//! 0xF000/0xFF00/0xE000; KIND=k forces one, and KIND=6/7 takes digits at least 0x8000/0xC000);
//! `rate` takes uniformly random limbs (the dispatch limits' trip rates). Rounding differs between
//! rustfft's NEON and AVX paths, so measure on each architecture; see docs/fft_precision.md.
//!
//! Build after `cargo build --release` (substitute the rlib's actual hash):
//! rustc --edition=2021 -O -C lto=thin -C panic=abort -C codegen-units=1 \
//!   benches/probes/fft_precision.rs \
//!   --extern big_bits=target/release/deps/libbig_bits-HASH.rlib \
//!   -L dependency=target/release/deps -o target/fft_precision
//! target/fft_precision <worst|rate> <sqr|mul|mid|cyclic> <lo N> <hi N> <count> [seed] [no-oracle]
//! STOP=k ends a size after k + 1 trips (count then shows the calls made), for the rate limits'
//! "more than k trips" rule.
//! The size column is operand limbs for sqr, output limbs for mul, short limbs n for mid and
//! modulus limbs m for cyclic.
use big_bits::utils::mul::{
    bnm1_size, fft_entry_raw, fft_mid_mul_raw, fft_sqr_entry_raw, fft_sqr_mod_bnm1_raw,
    find_fft_size, ntt_mid_mul_dyn, ntt_sqr_entry_dyn, FftRounding,
};
use big_bits::utils::utils::fold_mod_bnm1;
use big_bits::utils::verify::{verify_mul, verify_sqr};
use big_bits::utils::FFT_COEFF_LIMIT;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed ^ (*seed >> 29)
}

// kind < 5: src/tests/mod.rs's fft_worst_operand kinds; kind 5: uniformly random limbs; kinds 6
// and 7: random digits all at least 0x8000 and 0xC000, between the two. The top limb is nonzero,
// so every operand has its full length.
fn operand(kind: usize, len: usize, seed: &mut u64) -> Vec<u64> {
    let mut v: Vec<u64> = match kind {
        0 => vec![u64::MAX; len],
        1 => (0..len)
            .map(|i| if i >= len / 2 { u64::MAX } else { 0 })
            .collect(),
        2..=4 | 6 | 7 => {
            let digits = [
                0xF000_F000_F000_F000u64,
                0xFF00_FF00_FF00_FF00,
                0xE000_E000_E000_E000,
                0,
                0x8000_8000_8000_8000,
                0xC000_C000_C000_C000,
            ];
            (0..len).map(|_| lcg(seed) | digits[kind - 2]).collect()
        }
        _ => (0..len).map(|_| lcg(seed)).collect(),
    };
    if kind == 5 {
        if v[len - 1] == 0 {
            v[len - 1] = 1;
        }
    } else {
        v[len - 1] |= 1 << 63;
    }
    v
}

fn full(f: impl FnOnce(&mut [u64]) -> u64, len: usize) -> Vec<u64> {
    let mut out = vec![0; len];
    out[len - 1] = f(&mut out[..len - 1]);
    out
}

// One call at transform size n: (shape size, rounding, wrong if checked).
fn run(
    op: &str,
    n: usize,
    kind: usize,
    oracle: bool,
    seed: &mut u64,
) -> (usize, FftRounding, Option<bool>) {
    match op {
        "sqr" => {
            let l = (n + 1) / 8; // 8l - 1 digits of square fit n
            let a = operand(kind, l, seed);
            let mut err = FftRounding::default();
            let got = full(
                |o| {
                    let (top, e) = fft_sqr_entry_raw(&a, o);
                    err = e;
                    top
                },
                2 * l,
            );
            (l, err, Some(!verify_sqr(&a, &got)))
        }
        "mul" => {
            let l = (n + 1) / 8;
            let (a, b) = (operand(kind, l, seed), operand(kind, l, seed));
            let mut err = FftRounding::default();
            let got = full(
                |o| {
                    let (top, e) = fft_entry_raw(&a, &b, o);
                    err = e;
                    top
                },
                2 * l,
            );
            (2 * l - 1, err, Some(!verify_mul(&a, &b, &got)))
        }
        "mid" => {
            let m = (n - 4) / 8; // the middle product's transform covers 8m + 4 digits
            let (long, short) = (operand(kind, 2 * m - 1, seed), operand(kind, m, seed));
            let mut got = vec![0; m];
            let (gc, err) = fft_mid_mul_raw(&long, &short, &mut got);
            let wrong = oracle.then(|| {
                let mut exact = vec![0; m];
                let ec = ntt_mid_mul_dyn(&long, &short, &mut exact);
                // The two middle products differ by design in their lowest two limbs.
                got[2..] != exact[2..] || gc != ec
            });
            (m, err, wrong)
        }
        "cyclic" => {
            let m = n / 4;
            let a = operand(kind, m, seed);
            let mut got = vec![0; m];
            let err = fft_sqr_mod_bnm1_raw(&a, &mut got);
            let wrong = oracle.then(|| {
                let mut exact = vec![0; m];
                fold_mod_bnm1(&full(|o| ntt_sqr_entry_dyn(&a, o), 2 * m), &mut exact);
                got != exact
            });
            (m, err, wrong)
        }
        _ => panic!("op is sqr, mul, mid or cyclic"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mode, op) = (args[1].as_str(), args[2].as_str());
    let (lo, hi, count): (usize, usize, usize) = (
        args[3].parse().unwrap(),
        args[4].parse().unwrap(),
        args[5].parse().unwrap(),
    );
    let mut seed: u64 = args.get(6).map_or(1, |s| s.parse().unwrap());
    let oracle = args.get(7).map_or(true, |s| s != "no-oracle");
    let forced_kind: Option<usize> = std::env::var("KIND").ok().map(|k| k.parse().unwrap());
    let stop: Option<usize> = std::env::var("STOP").ok().map(|k| k.parse().unwrap());
    assert!(mode == "worst" || mode == "rate", "mode is worst or rate");
    let mut t = lo;
    while t <= hi {
        let n = find_fft_size(t);
        t = n + 1;
        if n > hi {
            break;
        }
        if op == "cyclic" && bnm1_size(n / 4) != n / 4 {
            continue;
        }
        let (mut size, mut max_err, mut max_peak, mut min_wrong) =
            (0, 0.0f64, 0.0f64, f64::INFINITY);
        let (mut trips, mut over, mut t30, mut t35) = (0, 0, 0, 0);
        let (mut wrong, mut under, mut missed) = (0, 0, 0);
        let mut calls = 0;
        for i in 0..count {
            let kind = match (mode, forced_kind) {
                ("rate", _) => 5,
                (_, Some(k)) => k,
                _ => i % 5,
            };
            let (sz, r, bad) = run(op, n, kind, oracle, &mut seed);
            size = sz;
            max_err = max_err.max(r.err);
            max_peak = max_peak.max(r.peak);
            t30 += (r.err > 0.30) as usize;
            t35 += (r.err > 0.35) as usize;
            let gated = !(r.peak < FFT_COEFF_LIMIT);
            over += gated as usize;
            let tripped = r.trips();
            trips += tripped as usize;
            if bad == Some(true) {
                wrong += 1;
                missed += !tripped as usize;
                if !gated {
                    under += 1;
                    min_wrong = min_wrong.min(r.err);
                }
            }
            calls += 1;
            if stop.is_some_and(|k| trips > k) {
                break;
            }
        }
        let checked = oracle || op == "sqr" || op == "mul";
        let col = |v: usize| {
            if checked {
                v.to_string()
            } else {
                "-".to_string()
            }
        };
        println!(
            "{mode} {op} N={n} size={size} count={calls} trips={trips} over={over} t30={t30} \
             t35={t35} wrong={} under={} missed={missed} err={max_err:.4} peak={:.3} \
             min_wrong={min_wrong:.4}",
            col(wrong),
            col(under),
            max_peak.log2()
        );
    }
}
