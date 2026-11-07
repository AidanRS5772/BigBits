//! Measures what `find_fft_size` and the 16-bit FFT's dispatch limits rest on (see
//! docs/fft_precision.md):
//! - `sizes [seconds]`: rustfft's ns per point for each size class b 2^a, at sizes near 2^10 to
//!   2^24, for choosing the size policy. Run it on each architecture: rustfft's NEON and AVX paths
//!   charge differently for odd factors.
//! - `entry <mul|sqr|mid|cyclic> <limbs> <fft|ntt|both> [rounds]`: one entry at one size, the raw
//!   FFT against the NTT, with the FFT's rounding and whether its result is right. limbs is the
//!   operands' length (mid: short; cyclic: the modulus m). Run each size in a fresh process: past
//!   a few million limbs a process holds gigabytes of plans, twiddles and pooled scratch, so with
//!   `both` the second side can run under memory pressure the first caused. To compare sizes near
//!   the limit, run `fft` and `ntt` separately and read each one's peak with `/usr/bin/time -l`.
//!
//! Build after `cargo build --release` (substitute the rlibs' actual hashes):
//! rustc --edition=2021 -O -C lto=thin -C panic=abort -C codegen-units=1 \
//!   benches/probes/fft_sizes.rs \
//!   --extern big_bits=target/release/deps/libbig_bits-HASH.rlib \
//!   --extern rustfft=target/release/deps/librustfft-HASH.rlib \
//!   -L dependency=target/release/deps -o target/fft_sizes
use big_bits::utils::mul::{
    find_fft_size, fft_entry_raw, fft_mid_mul_raw, fft_sqr_entry_raw, fft_sqr_mod_bnm1_raw,
    ntt_entry_dyn, ntt_mid_mul_dyn, ntt_sqr_entry_dyn, FftRounding,
};
use big_bits::utils::utils::{fold_mod_bnm1, sub_buf};
use rustfft::{num_complex::Complex, FftPlanner};
use std::time::Instant;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed ^ (*seed >> 29)
}

fn operand(len: usize, seed: &mut u64) -> Vec<u64> {
    let mut v: Vec<u64> = (0..len).map(|_| lcg(seed)).collect();
    v[len - 1] |= 1 << 63;
    v
}

fn fastest(rounds: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn point_cost(planner: &mut FftPlanner<f64>, n: usize, seconds: f64) -> f64 {
    let fft = planner.plan_fft_forward(n);
    let source: Vec<_> = (0..n)
        .map(|i| {
            Complex::new(
                ((i * 2654435761) % 65536) as f64 - 32768.0,
                ((i * 40503) % 65536) as f64 - 32768.0,
            )
        })
        .collect();
    let mut x = source.clone();
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    let start = Instant::now();
    let (mut best, mut rounds) = (f64::MAX, 0);
    while rounds < 3 || start.elapsed().as_secs_f64() < seconds {
        x.copy_from_slice(&source);
        let t = Instant::now();
        fft.process_with_scratch(&mut x, &mut scratch);
        best = best.min(t.elapsed().as_secs_f64());
        rounds += 1;
    }
    std::hint::black_box(&x);
    best * 1e9 / n as f64
}

fn sizes(seconds: f64) {
    const ODD: [usize; 14] = [1, 3, 5, 7, 9, 15, 21, 25, 27, 35, 45, 63, 125, 243];
    let mut planner = FftPlanner::new();
    print!("{:>6}", "odd");
    for e in (10..=24).step_by(2) {
        print!(" {:>7}", format!("~2^{e}"));
    }
    println!();
    for odd in ODD {
        print!("{odd:>6}");
        for e in (10..=24).step_by(2) {
            let mut n = odd * 4;
            while n < 1 << e {
                n *= 2;
            }
            print!(" {:>7.2}", point_cost(&mut planner, n, seconds));
        }
        println!();
    }
}

// Whether two values, as integers of the same length, differ by more than their two low limbs.
fn differ(a: &[u64], b: &[u64]) -> bool {
    let sub = |x: &[u64], y: &[u64]| {
        let mut d = x.to_vec();
        sub_buf(&mut d, y);
        d.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1)
    };
    sub(a, b).min(sub(b, a)) > 2
}

fn entry(op: &str, limbs: usize, side: &str, rounds: usize) {
    let mut seed = 1;
    let (a, b) = match op {
        "mid" => (operand(2 * limbs - 1, &mut seed), operand(limbs, &mut seed)),
        _ => (operand(limbs, &mut seed), operand(limbs, &mut seed)),
    };
    let width = match op {
        "mid" | "cyclic" => limbs,
        _ => 2 * limbs - 1,
    };
    let (mut fft_out, mut ntt_out) = (vec![0; width + 2], vec![0; width + 2]);
    let mut rounding = FftRounding::default();
    let (mut fft_time, mut ntt_time) = (0.0, 0.0);
    if side != "ntt" {
        fft_time = fastest(rounds, || {
            let out = &mut fft_out;
            rounding = match op {
                "mul" => {
                    let (top, r) = fft_entry_raw(&a, &b, &mut out[..width]);
                    out[width] = top;
                    r
                }
                "sqr" => {
                    let (top, r) = fft_sqr_entry_raw(&a, &mut out[..width]);
                    out[width] = top;
                    r
                }
                "mid" => {
                    let (of, r) = fft_mid_mul_raw(&a, &b, &mut out[..width]);
                    (out[width], out[width + 1]) = of;
                    r
                }
                "cyclic" => fft_sqr_mod_bnm1_raw(&a, &mut out[..width]),
                _ => panic!("op is mul, sqr, mid or cyclic"),
            };
        });
    }
    if side != "fft" {
        ntt_time = fastest(rounds, || {
            let out = &mut ntt_out;
            match op {
                "mul" => out[width] = ntt_entry_dyn(&a, &b, &mut out[..width]),
                "sqr" => out[width] = ntt_sqr_entry_dyn(&a, &mut out[..width]),
                "mid" => (out[width], out[width + 1]) = ntt_mid_mul_dyn(&a, &b, &mut out[..width]),
                "cyclic" => {
                    let mut full = vec![0; 2 * limbs];
                    full[2 * limbs - 1] = ntt_sqr_entry_dyn(&a, &mut full[..2 * limbs - 1]);
                    fold_mod_bnm1(&full, &mut out[..width]);
                }
                _ => panic!("op is mul, sqr, mid or cyclic"),
            }
        });
    }
    let n = if op == "cyclic" { 4 * limbs } else { find_fft_size(8 * limbs) };
    let wrong = side == "both" && differ(&fft_out, &ntt_out);
    println!(
        "{op} limbs={limbs} n={n} fft={:.1}ms ntt={:.1}ms err={:.4}{}",
        fft_time * 1e3,
        ntt_time * 1e3,
        rounding.err,
        if wrong { " WRONG" } else { "" }
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("sizes") => sizes(args.get(2).map_or(0.04, |s| s.parse().unwrap())),
        Some("entry") => entry(
            &args[2],
            args[3].parse().unwrap(),
            &args[4],
            args.get(5).map_or(3, |s| s.parse().unwrap()),
        ),
        _ => panic!("fft_sizes sizes [seconds] | entry <op> <limbs> <fft|ntt|both> [rounds]"),
    }
}
