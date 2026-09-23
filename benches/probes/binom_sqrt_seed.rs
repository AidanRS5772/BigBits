//! Isolated sqrt_4x2 experiments. No production algorithms are changed.
//! Build after `cargo build --profile prof` (substitute the rlib's actual hash):
//! rustc --edition=2021 -O -g benches/probes/binom_sqrt_seed.rs \
//!   --extern big_bits=target/prof/deps/libbig_bits-HASH.rlib \
//!   -L dependency=target/prof/deps -o /tmp/binom_sqrt_seed
//! taskset -c 2 /tmp/binom_sqrt_seed
use big_bits::utils::sqrt::correct_sqrt;
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

type Seed = fn(&mut [u64; 4]) -> [u64; 2];

// MODE 0: original; 1: comparisons for rem / s_hi; 2: divide half the numerator;
// 3: comparisons with the remaining remainder explicitly computed as y - t*s_hi.
// FIXED: replace slice-based correction with a fixed-width arithmetic chain.
// NARROW: communicate the proven one-limb root bounds to the compiler.
#[inline(never)]
fn seed<const MODE: u8, const FIXED: bool, const NARROW: bool>(x: &mut [u64; 4]) -> [u64; 2] {
    let x_hi = x[2] as u128 | ((x[3] as u128) << 64);
    let s_hi = x_hi.isqrt();
    let s_hi = if NARROW { s_hi as u64 as u128 } else { s_hi };
    let rem = x_hi - s_hi * s_hi;
    let (mut s_lo, u) = if MODE == 2 {
        if rem == 2 * s_hi {
            (u64::MAX as u128, 2 * s_hi + x[1] as u128)
        } else {
            // rem < 2*s_hi <= 2*(B-1), so half fits in 128 bits.
            let half = (rem << 63) | ((x[1] as u128) >> 1);
            let q = half / s_hi;
            let q = if NARROW { q as u64 as u128 } else { q };
            let v = half % s_hi;
            (q, (v << 1) | ((x[1] & 1) as u128))
        }
    } else {
        let (c, r0) = if MODE == 1 || MODE == 3 {
            if rem >= s_hi {
                let r0 = rem - s_hi;
                if r0 >= s_hi {
                    (2, r0 - s_hi)
                } else {
                    (1, r0)
                }
            } else {
                (0, rem)
            }
        } else {
            (rem / s_hi, rem % s_hi)
        };
        let y = (r0 << 64) | x[1] as u128;
        let t = y / s_hi;
        let v = if MODE == 3 { y - t * s_hi } else { y % s_hi };
        let (q, d) = match c {
            0 => (t >> 1, t & 1),
            1 => ((1 << 63) | (t >> 1), t & 1),
            2 => (u64::MAX as u128, t + 2),
            _ => unreachable!(),
        };
        (q, v + d * s_hi)
    };
    if NARROW {
        s_lo = s_lo as u64 as u128;
    }
    let s_lo_sqr = s_lo * s_lo;
    if FIXED {
        let low = x[0] as u128 | ((u as u64 as u128) << 64);
        let (mut low, borrow) = low.overflowing_sub(s_lo_sqr);
        let (mut high, negative) = ((u >> 64) as u64).overflowing_sub(borrow as u64);
        if negative {
            s_lo -= 1;
            let root = (s_hi << 64) | s_lo;
            // Add 2*new_root + 1 = 2*old_root - 1 in 129 bits.
            let correction = (root << 1) | 1;
            let (next, carry) = low.overflowing_add(correction);
            low = next;
            high = high
                .wrapping_add((root >> 127) as u64)
                .wrapping_add(carry as u64);
        }
        *x = [low as u64, (low >> 64) as u64, high, 0];
    } else {
        let mut buf = [x[0], u as u64, (u >> 64) as u64];
        if correct_sqrt(
            &mut buf,
            &[s_lo as u64, s_hi as u64],
            &[s_lo_sqr as u64, (s_lo_sqr >> 64) as u64],
        ) {
            s_lo -= 1;
        }
        *x = [buf[0], buf[1], buf[2], 0];
    }
    [s_lo as u64, s_hi as u64]
}

const VARIANTS: [(&str, Seed); 9] = [
    ("baseline", seed::<0, false, false>),
    ("compare", seed::<1, false, false>),
    ("half", seed::<2, false, false>),
    ("baseline_fixed", seed::<0, true, false>),
    ("half_fixed", seed::<2, true, false>),
    ("compare_manual", seed::<3, false, false>),
    ("baseline_narrow", seed::<0, false, true>),
    ("half_narrow", seed::<2, false, true>),
    ("half_fixed_narrow", seed::<2, true, true>),
];

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn square(root: [u64; 2]) -> [u64; 4] {
    let mut out = [0_u64; 4];
    for i in 0..2 {
        let mut carry = 0_u128;
        for j in 0..2 {
            let v = root[i] as u128 * root[j] as u128 + out[i + j] as u128 + carry;
            out[i + j] = v as u64;
            carry = v >> 64;
        }
        out[i + 2] = carry as u64;
    }
    out
}

fn add(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], bool) {
    let mut out = [0; 4];
    let mut carry = 0_u128;
    for i in 0..4 {
        let t = a[i] as u128 + b[i] as u128 + carry;
        out[i] = t as u64;
        carry = t >> 64;
    }
    (out, carry != 0)
}

fn check(input: [u64; 4]) {
    if input[3] < 1 << 62 {
        return;
    }
    let mut expected = input;
    let root = seed::<0, false, false>(&mut expected);
    // Independent characterization: root^2 + remainder == input and remainder < 2*root+1.
    assert_eq!(
        add(square(root), expected),
        (input, false),
        "input={input:x?}"
    );
    let bound = add([root[0], root[1], 0, 0], [root[0], root[1], 0, 0]).0;
    let bound = add(bound, [1, 0, 0, 0]).0;
    assert!(expected.iter().rev().cmp(bound.iter().rev()).is_lt());
    for &(name, candidate) in &VARIANTS[1..] {
        let mut actual = input;
        assert_eq!(candidate(&mut actual), root, "{name}: input={input:x?}");
        assert_eq!(actual, expected, "{name}: input={input:x?}");
    }
}

fn validate() {
    let mut state = 0xabc123789deadbee;
    let mut cases = 0;
    for _ in 0..500_000 {
        check([
            random(&mut state),
            random(&mut state),
            random(&mut state),
            random(&mut state) | (1 << 62),
        ]);
        cases += 1;
    }
    // Near exact squares, including downward correction of the low root digit.
    for _ in 0..20_000 {
        let s = [random(&mut state), random(&mut state) | (1 << 63)];
        let sqr = square(s);
        for input in [sqr, add(sqr, [1, 0, 0, 0]).0, add(sqr, [u64::MAX; 4]).0] {
            check(input);
            cases += 1;
        }
    }
    // Boundary remainders exercise all original c arms and the saturated branch.
    for h in [1 << 63, (1 << 63) + 1, u64::MAX - 1, u64::MAX] {
        let h = h as u128;
        for r in [0, 1, h - 1, h, h + 1, 2 * h - 1, 2 * h] {
            let hi = h * h + r;
            for a in [0, 1, 1 << 63, u64::MAX - 1, u64::MAX] {
                for b in [0, 1, u64::MAX] {
                    check([b, a, hi as u64, (hi >> 64) as u64]);
                    cases += 1;
                }
            }
        }
    }
    eprintln!("checked {cases} normalized random/square/boundary inputs");
}

fn time(f: Seed, inputs: &[[u64; 4]], duration: Duration) -> f64 {
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed() < duration {
        for input in inputs {
            let mut x = black_box(*input);
            black_box(f(black_box(&mut x)));
            black_box(x);
        }
        count += inputs.len();
    }
    start.elapsed().as_nanos() as f64 / count as f64
}

fn main() {
    validate();
    if std::env::args().any(|arg| arg == "--check-only") {
        return;
    }
    let mut state = 0xb807564a32763014;
    println!("pattern,variant,median_ns,min_ns,max_ns");
    for pattern in ["random", "squares", "saturated"] {
        let inputs: Vec<_> = (0..4096)
            .map(|_| match pattern {
                "squares" => square([random(&mut state), random(&mut state) | (1 << 63)]),
                "saturated" => {
                    let h = (random(&mut state) | (1 << 63)) as u128;
                    let hi = h * h + 2 * h;
                    [
                        random(&mut state),
                        random(&mut state),
                        hi as u64,
                        (hi >> 64) as u64,
                    ]
                }
                _ => [
                    random(&mut state),
                    random(&mut state),
                    random(&mut state),
                    random(&mut state) | (1 << 62),
                ],
            })
            .collect();
        let mut timings = [[0.0_f64; 5]; VARIANTS.len()];
        for &(_, f) in &VARIANTS {
            time(f, &inputs, Duration::from_millis(50));
        }
        // Rotate the order to limit drift/thermal bias.
        for round in 0..5 {
            for step in 0..VARIANTS.len() {
                let i = (round + step) % VARIANTS.len();
                timings[i][round] = time(VARIANTS[i].1, &inputs, Duration::from_millis(100));
            }
        }
        for (i, &(name, _)) in VARIANTS.iter().enumerate() {
            timings[i].sort_by(f64::total_cmp);
            println!(
                "{pattern},{name},{:.3},{:.3},{:.3}",
                timings[i][2], timings[i][0], timings[i][4]
            );
        }
    }
}
