use super::{mul_ref, rand_nonzero_vec};
use crate::utils::trans::{theta_dyn, theta_static};
use crate::utils::utils::{add_buf, buf_len, cmp_buf, sub_buf, trim_lz};
use crate::utils::CapacityError;
use std::cmp::Ordering;
use std::f64::consts::{LN_2, PI};

fn bits(x: &[u64]) -> u64 {
    x.last()
        .map_or(0, |&top| 64 * x.len() as u64 - top.leading_zeros() as u64)
}

// floor(x 2^-lo) mod 2^64.
fn window(x: &[u64], lo: i64) -> u64 {
    let get = |i: i64| {
        usize::try_from(i)
            .ok()
            .and_then(|i| x.get(i))
            .map_or(0, |&l| l as u128)
    };
    let l = lo.div_euclid(64);
    ((get(l + 1) << 64 | get(l)) >> lo.rem_euclid(64)) as u64
}

// floor(a b 2^-64w) for w-limb fractions a and b, by schoolbook.
fn fmul(a: &[u64], b: &[u64], w: usize) -> Vec<u64> {
    let p = mul_ref(&a[..buf_len(a)], &b[..buf_len(b)]);
    (0..w).map(|i| p.get(w + i).copied().unwrap_or(0)).collect()
}

// q^e for a w-limb fraction q by binary powering, each product cut to w limbs.
fn fpow(q: &[u64], e: u64, w: usize) -> Vec<u64> {
    let mut p = q.to_vec();
    for i in (0..63 - e.leading_zeros()).rev() {
        p = fmul(&p, &p, w);
        if e >> i & 1 == 1 {
            p = fmul(&p, q, w);
        }
    }
    p
}

// (th2, th3, z) as theta_dyn defines them, from sums in one absolute frame of w fraction limbs,
// two or more past both outputs. Each power comes from binary powering, not theta's recurrence.
fn theta_oracle(
    x: &[u64],
    shift: u64,
    terms: usize,
    w2: usize,
    w3: usize,
) -> (Vec<u64>, Vec<u64>, usize) {
    let z = ((shift - bits(x) - 2) / 64) as usize;
    let w = (w3 + 2).max(z + w2 + 3);
    let q: Vec<u64> = (0..w)
        .map(|i| window(x, 64 * i as i64 + shift as i64 - 64 * w as i64))
        .collect();
    let (mut s2, mut s3) = (vec![0; w + 1], vec![0; w + 1]);
    for n in 1..=2 * terms as u64 + 1 {
        let p = fpow(&q, n * n, w);
        if buf_len(&p) == 0 {
            break;
        }
        let s = if n % 2 == 1 { &mut s2 } else { &mut s3 };
        add_buf(s, &p);
        add_buf(s, &p);
    }
    s3[w] += 1;
    (
        s2[w - z - w2..w - z].to_vec(),
        s3[w + 1 - w3..].to_vec(),
        z,
    )
}

fn assert_within_unit(got: &[u64], want: &[u64], what: &str) {
    let (hi, lo) = match cmp_buf(got, want) {
        Ordering::Less => (want, got),
        _ => (got, want),
    };
    let mut diff = hi.to_vec();
    sub_buf(&mut diff, lo);
    trim_lz(&mut diff);
    assert!(
        diff.len() <= 1 && diff.first().map_or(true, |&d| d <= 1),
        "{what}: diff={diff:?}"
    );
}

type ThetaFn<'a> = &'a dyn Fn(&[u64], u64, usize, &mut [u64], &mut [u64]) -> usize;

// theta at q = x 2^-(bits(x) + l) against the oracle; exact demands bit-identical outputs.
fn check(theta: ThetaFn, x: &[u64], l: u64, terms: usize, w2: usize, w3: usize, exact: bool) {
    let shift = bits(x) + l;
    let (want2, want3, wz) = theta_oracle(x, shift, terms, w2, w3);
    let (mut th2, mut th3) = (vec![0; w2], vec![0; w3]);
    let z = theta(x, shift, terms, &mut th2, &mut th3);
    let what = format!("x.len()={} l={l} terms={terms} w2={w2} w3={w3}", x.len());
    assert_eq!(z, wz, "z {what}");
    if exact {
        assert_eq!(th2, want2, "th2 {what}");
        assert_eq!(th3, want3, "th3 {what}");
    } else {
        assert_within_unit(&th2, &want2, &format!("th2 {what}"));
        assert_within_unit(&th3, &want3, &format!("th3 {what}"));
    }
}

const LS: [u64; 11] = [2, 3, 63, 64, 65, 66, 127, 128, 129, 130, 1000];

// x shorter and longer than both frames, at every l in LS, with terms both capped by N and by
// the frames (50), and outputs of very different lengths, so the step width follows the wider.
fn theta_random(theta: ThetaFn, ws: &[(usize, usize)], mut seed: u64) {
    for &l in &LS {
        for terms in [0, 1, 2, 3, 7, 50] {
            for &(w2, w3) in ws {
                for xl in [1, w2.max(w3) + 3] {
                    let x = rand_nonzero_vec(xl, seed);
                    seed += 1;
                    check(theta, &x, l, terms, w2, w3, false);
                }
            }
        }
    }
}

#[test]
fn test_theta_dyn_random() {
    let ws = [(1, 1), (2, 5), (5, 2), (17, 17), (40, 3), (3, 40)];
    theta_random(&theta_dyn, &ws, 1);
}

// q a power of two makes every power and product exact (u = 1/2, so each product
// renormalizes), and the terms' single bits never carry.
#[test]
fn test_theta_dyn_exact_powers_of_two() {
    for k in [0, 1, 37, 63] {
        for &l in &LS {
            for x in [vec![1u64 << k], vec![0, 0, 1u64 << k]] {
                for (w2, w3) in [(1, 1), (3, 7), (9, 2)] {
                    check(&theta_dyn, &x, l, 20, w2, w3, true);
                }
            }
        }
    }
}

// u just below 1 (all ones) and sparse u just above 1/2.
#[test]
fn test_theta_dyn_sparse_and_all_ones() {
    let xs = [
        vec![u64::MAX; 4],
        vec![1, 0, 0, 1 << 63],
        vec![1, 1 << 63],
        vec![u64::MAX, 0, 0, 1],
    ];
    for x in &xs {
        for &l in &LS {
            for (w2, w3) in [(1, 1), (4, 4), (12, 3)] {
                check(&theta_dyn, x, l, 20, w2, w3, false);
            }
        }
    }
}

// Past DYN_PARTIAL_MUL_CUTOFF and STATIC_PARTIAL_SQR_CUTOFF, so the guarded high products and
// the transforms run: l for N = 3 (p / (4 (N + 1)²)), and a small l capped by the frame.
#[test]
fn test_theta_dyn_large() {
    let x = rand_nonzero_vec(702, 7000);
    check(&theta_dyn, &x, 2800, 3, 700, 700, false);
    check(&theta_dyn, &x, 700, 50, 700, 650, false);
}

fn theta_static_cases<const N: usize>(seed: u64) {
    let theta = |x: &[u64], s, t, a: &mut [u64], b: &mut [u64]| {
        theta_static::<N>(x, s, t, a, b).expect("within N")
    };
    let ws = [(N - 2, N - 1), (1, N - 1), (N - 2, 1), (1, 1)];
    let mut seed = seed;
    for l in [2, 65, 130, 16 * N as u64] {
        for terms in [1, 3, 50] {
            for &(w2, w3) in &ws {
                for xl in [1, N + 3] {
                    let x = rand_nonzero_vec(xl, seed);
                    seed += 1;
                    check(&theta, &x, l, terms, w2, w3, false);
                }
            }
        }
    }
    // One limb past the bound through either output.
    let x = rand_nonzero_vec(3, seed);
    for (w2, w3) in [(N - 1, 1), (1, N)] {
        let (mut th2, mut th3) = (vec![0xAA; w2], vec![0xAA; w3]);
        let r = theta_static::<N>(&x, bits(&x) + 70, 3, &mut th2, &mut th3);
        assert_eq!(r, Err(CapacityError { min_n: N + 1 }));
        assert!(th2.iter().chain(&th3).all(|&l| l == 0xAA));
    }
}

#[test]
fn test_theta_static() {
    theta_static_cases::<4>(100);
    theta_static_cases::<8>(200);
    theta_static_cases::<16>(300);
    theta_static_cases::<64>(400);
    theta_static_cases::<400>(500);
}

#[test]
fn test_theta_static_large() {
    let theta = |x: &[u64], s, t, a: &mut [u64], b: &mut [u64]| {
        theta_static::<702>(x, s, t, a, b).expect("within N")
    };
    let x = rand_nonzero_vec(702, 7001);
    check(&theta, &x, 2800, 3, 700, 700, false);
}

// ln(1/q) = π / (4 AGM(θ2², θ3²)) in f64, independent of the oracle's formula: checks the
// identity and the outputs' scaling (θ2's factor 2 and z, θ3's integer limb).
#[test]
fn test_theta_agm_identity() {
    let mut seed = 900;
    for l in [2, 3, 5, 10, 31, 64, 65, 100, 200, 400] {
        for xl in 1..=3 {
            let x = rand_nonzero_vec(xl, seed);
            seed += 1;
            let shift = bits(&x) + l;
            let (mut th2, mut th3) = ([0; 3], [0; 2]);
            let z = theta_dyn(&x, shift, 10, &mut th2, &mut th3);
            let b64 = 2f64.powi(-64);
            let t2 = (th2[2] as f64 + th2[1] as f64 * b64) * 2f64.powi(-64 * (z as i32 + 1));
            let t3 = th3[1] as f64 + th3[0] as f64 * b64;
            let (mut a, mut b) = (t2 * t2, t3 * t3);
            for _ in 0..64 {
                (a, b) = ((a + b) / 2.0, (a * b).sqrt());
            }
            let xm = x[xl - 1] as f64 + if xl > 1 { x[xl - 2] as f64 * b64 } else { 0.0 };
            let want = (shift as f64 - 64.0 * (xl - 1) as f64) * LN_2 - xm.ln();
            let got = PI / (4.0 * a);
            assert!(
                (got - want).abs() <= 1e-12 * want,
                "l={l} xl={xl}: {got} vs {want}"
            );
        }
    }
}

#[test]
#[should_panic]
fn test_theta_q_at_least_quarter_panics() {
    theta_dyn(&[1], 2, 3, &mut [0; 2], &mut [0; 2]);
}

#[test]
#[should_panic]
fn test_theta_untrimmed_x_panics() {
    theta_dyn(&[1, 0], 200, 3, &mut [0; 2], &mut [0; 2]);
}
