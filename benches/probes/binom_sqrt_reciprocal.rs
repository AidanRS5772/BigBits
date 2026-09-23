//! Standalone x86-64 division microbenchmark, including one reciprocal setup
//! per root and a remainder dependency between consecutive divisions.
//! This measures only the candidate division primitive, not a complete sqrt.
//! rustc --edition=2021 -O benches/probes/binom_sqrt_reciprocal.rs -o /tmp/recip_probe
//! taskset -c 2 /tmp/recip_probe

use std::{
    arch::asm,
    hint::black_box,
    time::{Duration, Instant},
};

#[inline(always)]
fn hardware(lo: u64, hi: u64, d: u64) -> (u64, u64) {
    let mut q = lo;
    let mut r = hi;
    unsafe {
        asm!("div {d}", d=in(reg)d, inout("rax")q, inout("rdx")r,
             options(pure, nomem, nostack));
    }
    (q, r)
}

#[inline(always)]
fn inverse(d: u64) -> u64 {
    // floor((B^2 - 1)/d) - B, with B=2^64 and B/2 <= d < B.
    hardware(u64::MAX, !d, d).0
}

#[inline(always)]
fn reciprocal(lo: u64, hi: u64, d: u64, inv: u64) -> (u64, u64) {
    let p = hi as u128 * inv as u128;
    let carry = ((p as u64 as u128) + lo as u128 + ((lo as u128 * inv as u128) >> 64)) >> 64;
    let q = hi + (p >> 64) as u64 + carry as u64;
    let n = ((hi as u128) << 64) | lo as u128;
    let r = n - q as u128 * d as u128;
    if r >= d as u128 {
        (q + 1, (r - d as u128) as u64)
    } else {
        (q, r as u64)
    }
}

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

struct Case {
    d: u64,
    numerators: Vec<(u64, u64)>,
}

fn run<const RECIP: bool>(cases: &[Case], duration: Duration) -> f64 {
    let start = Instant::now();
    let mut count = 0u64;
    let mut checksum = 0u64;
    loop {
        for c in cases {
            let c = black_box(c);
            let inv = if RECIP { inverse(c.d) } else { 0 };
            let mut hi = c.numerators[0].1;
            for &(lo, _) in &c.numerators {
                let (q, r) = if RECIP {
                    reciprocal(lo, hi, c.d, inv)
                } else {
                    hardware(lo, hi, c.d)
                };
                checksum = checksum.wrapping_add(q ^ r);
                hi = r;
            }
            count += 1;
        }
        if start.elapsed() >= duration {
            break;
        }
    }
    black_box(checksum);
    start.elapsed().as_nanos() as f64 / count as f64
}

fn main() {
    let mut state = 934653563546u64;
    let mut checked = 0;
    for d in [1 << 63, (1 << 63) + 1, u64::MAX - 1, u64::MAX] {
        for hi in [0, 1, d / 2, d - 1] {
            for lo in [0, 1, u64::MAX] {
                assert_eq!(hardware(lo, hi, d), reciprocal(lo, hi, d, inverse(d)));
                checked += 1;
            }
        }
    }
    for _ in 0..200_000 {
        let d = rng(&mut state) | (1 << 63);
        let hi = ((rng(&mut state) as u128 * d as u128) >> 64) as u64;
        let lo = rng(&mut state);
        let n = ((hi as u128) << 64) | lo as u128;
        let expected = ((n / d as u128) as u64, (n % d as u128) as u64);
        assert_eq!(reciprocal(lo, hi, d, inverse(d)), expected);
        checked += 1;
    }
    println!("checked={checked}");
    for limbs in [4, 8, 16, 32, 63] {
        let cases: Vec<_> = (0..256)
            .map(|_| {
                let d = rng(&mut state) | (1 << 63);
                let numerators = (0..limbs - 2)
                    .map(|_| {
                        let hi = ((rng(&mut state) as u128 * d as u128) >> 64) as u64;
                        (rng(&mut state), hi)
                    })
                    .collect();
                Case { d, numerators }
            })
            .collect();
        run::<false>(&cases, Duration::from_millis(50));
        run::<true>(&cases, Duration::from_millis(50));
        let mut hw = Vec::new();
        let mut rc = Vec::new();
        for trial in 0..5 {
            let duration = Duration::from_millis(150);
            if trial % 2 == 0 {
                hw.push(run::<false>(&cases, duration));
                rc.push(run::<true>(&cases, duration));
            } else {
                rc.push(run::<true>(&cases, duration));
                hw.push(run::<false>(&cases, duration));
            }
        }
        hw.sort_by(f64::total_cmp);
        rc.sort_by(f64::total_cmp);
        println!("root_limbs={limbs} divisions={} hardware_ns={:.2} reciprocal_ns={:.2} reduction_pct={:.1}",limbs-2,hw[2],rc[2],100.*(1.-rc[2]/hw[2]));
    }
}
