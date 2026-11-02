use super::{mul_ref, rand_nonzero_vec, rand_vec};
use crate::utils::consts::consts::{ln2_dyn, pi_dyn};
use crate::utils::sqrt::sqrt_only_dyn;
use crate::utils::trans::{agm_dyn, agm_static, theta_dyn};
use crate::utils::utils::{add_buf, buf_len, cmp_buf, inc_buf, shr_buf, sub_buf};
use crate::utils::CapacityError;
use std::cmp::Ordering;
use std::f64::consts::{LN_2, PI};

fn bits(x: &[u64]) -> u64 {
    let n = buf_len(x);
    if n == 0 {
        0
    } else {
        64 * n as u64 - x[n - 1].leading_zeros() as u64
    }
}

// The n-limb window of v from limb lo, zero past its end.
fn window(v: &[u64], lo: usize, n: usize) -> Vec<u64> {
    (0..n).map(|i| v.get(lo + i).copied().unwrap_or(0)).collect()
}

// x + 2^e, x long enough.
fn add_pow2(x: &mut [u64], e: u64) {
    let carry = add_buf(&mut x[(e / 64) as usize..], &[1 << (e % 64)]);
    assert!(!carry);
}

// x >> n bits.
fn shr_bits(x: &[u64], n: usize) -> Vec<u64> {
    let mut v = window(x, n / 64, x.len());
    shr_buf(&mut v, (n % 64) as u8);
    v
}

// ⌊√x⌋ through sqrt_only_dyn, an odd length by its virtual low limb (a root 32 bits too high),
// checked by schoolbook: s² <= x < (s + 1)².
fn isqrt(x: &[u64]) -> Vec<u64> {
    let t = buf_len(x);
    let mut s = match t {
        0 => vec![0],
        1 => vec![x[0].isqrt()],
        _ => {
            let mut s = vec![0; t.div_ceil(2)];
            sqrt_only_dyn(&mut x[..t].to_vec(), &mut s);
            if t % 2 == 1 {
                shr_buf(&mut s, 32);
            }
            s
        }
    };
    let n = buf_len(&s);
    assert_ne!(cmp_buf(&mul_ref(&s[..n], &s[..n]), x), Ordering::Greater);
    s.push(0);
    inc_buf(&mut s);
    let n = buf_len(&s);
    assert_eq!(cmp_buf(&mul_ref(&s[..n], &s[..n]), x), Ordering::Greater);
    s
}

// log2 x for x > 0, from its top 64 bits.
fn log2(x: &[u64]) -> f64 {
    let lo = bits(x).saturating_sub(64);
    let top = shr_bits(x, lo as usize)[0];
    (top as f64).log2() + lo as f64
}

// log2(2^x + 2^y).
fn log2_sum(x: f64, y: f64) -> f64 {
    let (hi, lo) = (x.max(y), x.min(y));
    hi + (lo - hi).exp2().ln_1p() / LN_2
}

// The AGM's a_n with exact floor means, G limbs past an l-limb frame and floored back to it,
// and the most a run in that frame may differ from it, for nonzero a and b. Each computed pair
// (a_k, b_k), k >= 1, is up to half a unit off in a (a floor) and a unit in b (a root), which
// moves a_n by α_k / 2 + β_k for (α_k, β_k) = ∂a_n / ∂(a_k, b_k). These come back from
// (α, β) = (1/2, 1/2) at k = n - 1 through each step's Jacobian, in log2 so that a / b far past
// f64's range is fine: α_k = (α + β √(b_k / a_k)) / 2 and β_k = (α + β √(a_k / b_k)) / 2 from
// k + 1's. The bound is twice their sum plus the last mean's half unit, for the f64 and
// second-order slack, plus 2 for the oracle's own floor and error.
fn agm_oracle(a: &[u64], b: &[u64], n: usize, l: usize) -> (Vec<u64>, Vec<u64>) {
    const G: usize = 2;
    let w = l + G + 1;
    let up = |x: &[u64]| {
        let mut v = vec![0; w];
        v[G..G + x.len()].copy_from_slice(x);
        v
    };
    let (mut a, mut b) = (up(a), up(b));
    // log2(b_k / a_k) for k = 1..n - 1.
    let mut ratios = Vec::new();
    for k in 0..n {
        if k > 0 {
            ratios.push(log2(&b) - log2(&a));
        }
        let next = (k + 1 < n).then(|| isqrt(&mul_ref(&a[..buf_len(&a)], &b[..buf_len(&b)])));
        add_buf(&mut a, &b);
        shr_buf(&mut a, 1);
        if let Some(g) = next {
            b = window(&g, 0, w);
        }
    }
    assert_eq!(buf_len(&a[G + l..]), 0);

    let (mut la, mut lb, mut total) = (-1.0, -1.0, -1.0_f64);
    // ratios[i] is pair k = i + 1, and the last, k = n - 1, starts from (1/2, 1/2).
    for (i, &r) in ratios.iter().enumerate().rev() {
        if i + 2 < n {
            (la, lb) = (
                log2_sum(la, lb + r / 2.0) - 1.0,
                log2_sum(la, lb - r / 2.0) - 1.0,
            );
        }
        total = log2_sum(total, log2_sum(la - 1.0, lb));
    }
    let mut tol = vec![0; w + 2];
    add_pow2(&mut tol, (total + 1.0).ceil() as u64);
    add_pow2(&mut tol, 1);
    (a[G..G + l].to_vec(), tol)
}

type AgmFn<'a> = &'a dyn Fn(&[u64], &[u64], usize, &mut [u64]);

fn agm_static_fn<const N: usize>(a: &[u64], b: &[u64], n: usize, out: &mut [u64]) {
    agm_static::<N>(a, b, n, out).unwrap();
}

// agm into l limbs within the oracle's bound, exactly for n <= 1 (no root).
fn check(agm: AgmFn, a: &[u64], b: &[u64], n: usize, l: usize) {
    let (want, tol) = agm_oracle(a, b, n, l);
    let mut got = vec![u64::MAX; l];
    agm(a, b, n, &mut got);
    let what = format!("l={l} n={n} a.len()={} b.len()={}", a.len(), b.len());
    if n <= 1 {
        assert_eq!(got, want, "{what}");
        return;
    }
    let (mut diff, lo) = match cmp_buf(&got, &want) {
        Ordering::Less => (want.clone(), &got),
        _ => (got.clone(), &want),
    };
    sub_buf(&mut diff, lo);
    assert_ne!(
        cmp_buf(&diff, &tol),
        Ordering::Greater,
        "{what}: off by {:?}, bound {:?}",
        &diff[..buf_len(&diff)],
        &tol[..buf_len(&tol)]
    );
}

const ITERS: [usize; 6] = [0, 1, 2, 3, 10, 30];

// Across the approximate root's binomial/Zimmermann switch (11-limb roots) and the square
// shapes' School/Karatsuba (about 55 limbs) and Karatsuba/FFT (about 230) switches.
#[test]
fn test_agm_random_against_oracle() {
    let mut seed = 1;
    for l in [1, 2, 3, 4, 5, 10, 11, 12, 13, 40, 56, 100, 240, 300] {
        for n in ITERS {
            let (a, b) = (rand_nonzero_vec(l, seed), rand_nonzero_vec(l, seed + 1));
            seed += 2;
            check(&agm_dyn, &a, &b, n, l);
        }
    }
}

// b far below a, with every count of zero top limbs: odd and even product lengths, and errors in
// b that cost about a / b.
#[test]
fn test_agm_small_b() {
    let mut seed = 1000;
    for l in [2, 3, 4, 5, 12, 13, 40] {
        for k in 1..l {
            for n in [2, 3, 8, 20] {
                let a = rand_nonzero_vec(l, seed);
                let mut b = rand_nonzero_vec(l - k, seed + 1);
                seed += 2;
                check(&agm_dyn, &a, &b, n, l);
                b.resize(l, 0);
                check(&agm_dyn, &b, &a, n, l);
            }
        }
    }
    check(&agm_dyn, &rand_nonzero_vec(20, 7), &[1], 30, 20);
}

// a = B^(l-1) makes the first product b B^(l-1), so b's length sets its parity (odd ones take
// the padded root and its 32-bit shift) and b's top limb its top limb, from 63 leading zeros
// to none.
#[test]
fn test_agm_product_parity() {
    let mut seed = 2000;
    for l in [2, 3, 4, 5, 12, 13, 24, 25] {
        let mut a = vec![0; l];
        a[l - 1] = 1;
        for lb in 1..=l {
            for top in [1, 2, 3, 4, 7, 1 << 32, 1 << 63, u64::MAX] {
                let mut b = rand_vec(lb, seed);
                seed += 1;
                b[lb - 1] = top;
                for n in [2, 3, 5] {
                    check(&agm_dyn, &a, &b, n, l);
                }
            }
        }
    }
}

// Products of one limb, which the root takes as x B like any odd length, in one-limb and wider
// frames.
#[test]
fn test_agm_one_limb_product() {
    for (a, b) in [(3, 5), (1, 1), (1, 2), (7, 9), (1 << 20, 1 << 30), (u32::MAX as u64, 3)] {
        for l in [1, 2, 3] {
            for n in [2, 3, 5] {
                check(&agm_dyn, &[a], &[b], n, l);
                check(&agm_static_fn::<6>, &[a], &[b], n, l);
            }
        }
    }
}

#[test]
fn test_agm_exact_cases() {
    let mut seed = 3000;
    for l in [1, 2, 5, 13, 40] {
        let (a, b) = (rand_nonzero_vec(l, seed), rand_nonzero_vec(l, seed + 1));
        seed += 2;
        let mut out = vec![0; l];
        let zero = vec![0; l];
        for n in [0, 1, 2, 3, 7, 64, 65, 64 * l + 3] {
            // A zero b stays zero, so each step halves a.
            agm_dyn(&a, &zero, n, &mut out);
            assert_eq!(out, shr_bits(&a, n), "b = 0, l={l} n={n}");
            if n > 0 {
                agm_dyn(&zero, &b, n, &mut out);
                assert_eq!(out, shr_bits(&b, n), "a = 0, l={l} n={n}");
            }
        }
        agm_dyn(&a, &b, 0, &mut out);
        assert_eq!(out, a);
        for n in [1, 2, 3, 10] {
            agm_dyn(&a, &b, n, &mut out);
            let mut swapped = vec![0; l];
            agm_dyn(&b, &a, n, &mut swapped);
            assert_eq!(out, swapped, "symmetry, l={l} n={n}");
            // Shorter inputs are zero-extended.
            let mut wide = vec![0; l + 3];
            agm_dyn(&a, &b, n, &mut wide);
            assert_eq!(wide[..l], out[..], "wider out, l={l} n={n}");
            assert_eq!(buf_len(&wide[l..]), 0);
            check(&agm_dyn, &a, &a, n, l);
            check(&agm_dyn, &vec![u64::MAX; l], &vec![u64::MAX; l], n, l);
        }
    }
}

#[test]
#[should_panic]
fn test_agm_input_longer_than_out_panics() {
    agm_dyn(&[1, 2, 3], &[1], 3, &mut [0; 2]);
}

fn check_static<const N: usize>(seed: &mut u64) {
    let f: AgmFn = &agm_static_fn::<N>;
    for l in [1, 2, N / 4, N / 2, N - 3, N - 2] {
        if l == 0 || l + 2 > N {
            continue;
        }
        for n in ITERS {
            let (a, b) = (rand_nonzero_vec(l, *seed), rand_nonzero_vec(l, *seed + 1));
            let small = rand_nonzero_vec(l.div_ceil(3), *seed + 2);
            *seed += 3;
            check(f, &a, &b, n, l);
            check(f, &a, &small, n, l);
        }
    }
}

#[test]
fn test_agm_static_against_oracle() {
    let mut seed = 4000;
    check_static::<3>(&mut seed);
    check_static::<8>(&mut seed);
    check_static::<26>(&mut seed);
    check_static::<128>(&mut seed);
    check_static::<600>(&mut seed);
}

#[test]
fn test_agm_static_capacity() {
    let (a, b) = (rand_nonzero_vec(4, 5000), rand_nonzero_vec(4, 5001));
    let mut out = [7; 4];
    assert_eq!(
        agm_static::<5>(&a, &b, 3, &mut out),
        Err(CapacityError { min_n: 6 })
    );
    assert_eq!(out, [7; 4]);
    assert_eq!(agm_static::<6>(&a, &b, 3, &mut out), Ok(()));
}

// ln(1/q) = π / (4 AGM(θ2², θ3²)) at q = 2^-s. theta_dyn's outputs are squared into one frame
// of f' fraction limbs, with θ2²'s 2s leading zero bits and a guard limb past p = 64 f bits, and
// the AGM runs for n steps from the condition (A) of docs/agm-ln-theta.md, plus one. Its result
// M must satisfy 4 s M ln 2 = π to within 2^(8 - p) relative.
fn check_ln_identity(agm: AgmFn, f: usize, s: u64) {
    let fp = f + (2 * s).div_ceil(64) as usize + 1;
    let l = fp + 1;
    // θ2 2^(64 (w2 + z)) at about the bits θ2² keeps in the frame, two limbs over.
    let z = ((s - 3) / 64) as usize;
    let w2 = fp - 2 * z + 2;
    let (mut th2, mut th3) = (vec![0; w2], vec![0; l]);
    assert_eq!(theta_dyn::<16>(&[1], s, &mut th2, &mut th3), z);
    let a = window(&mul_ref(&th3, &th3), fp, l);
    let b = window(&mul_ref(&th2, &th2), 2 * (w2 + z) - fp, l);

    let p = 64.0 * f as f64;
    let n = ((p + 16.0) * 4.0 * s as f64 * LN_2 * LN_2 / (PI * PI))
        .log2()
        .ceil()
        .max(0.0) as usize
        + 1;
    let mut m = vec![0; l];
    agm(&b, &a, n, &mut m);

    let (mut ln2, mut pi) = (vec![0; fp], vec![0; fp + 1]);
    ln2_dyn(&mut ln2);
    pi_dyn(&mut pi);
    let mut lhs = mul_ref(&mul_ref(&m[..buf_len(&m)], &ln2), &[4 * s]);
    let mut rhs = vec![0; fp];
    rhs.extend_from_slice(&pi);
    let w = lhs.len().max(rhs.len());
    lhs.resize(w, 0);
    rhs.resize(w, 0);
    let (mut diff, lo) = match cmp_buf(&lhs, &rhs) {
        Ordering::Less => (rhs.clone(), &lhs),
        _ => (lhs.clone(), &rhs),
    };
    sub_buf(&mut diff, lo);
    // π B^(2 f') < 2^(128 f' + 2).
    let bound = 128 * fp as u64 + 2 + 8 - 64 * f as u64;
    assert!(
        bits(&diff) <= bound,
        "f={f} s={s} n={n}: error 2^{} past 2^{bound}",
        bits(&diff)
    );
}

#[test]
fn test_agm_theta_ln_identity() {
    for f in [1, 2, 8, 32] {
        for s in [3, 4, 64, 67, 200, 1000] {
            check_ln_identity(&agm_dyn, f, s);
        }
    }
    check_ln_identity(&agm_static_fn::<200>, 32, 1000);
    check_ln_identity(&agm_static_fn::<200>, 8, 67);
}
