use super::rand_vec;
use crate::utils::consts::consts::ln2_dyn;
use crate::utils::div::div_prim;
use crate::utils::mul::mul_buf;
use crate::utils::trans::exp_dyn;
use crate::utils::utils::{cmp_buf, sub_buf, trim_lz};

// e^x - 1 as a p-limb fraction by summing x^n / n! directly, with no binary splitting.
fn exp_oracle(x: &[u64], m: usize, p: usize) -> Vec<u64> {
    let mut xf = vec![0; p];
    for (i, &l) in x.iter().enumerate() {
        if let Some(j) = (p + i).checked_sub(m + x.len()) {
            xf[j] = l;
        }
    }
    let (mut sum, mut t, mut n) = (vec![0; p], xf.clone(), 1);
    let mut prod = vec![0; 2 * p];
    while t.iter().any(|&l| l != 0) {
        let carry = crate::utils::utils::add_buf(&mut sum, &t);
        assert!(!carry);
        // schoolbook, so the oracle shares no fast multiplication path with exp_dyn
        prod[2 * p - 1] = mul_buf(&t, &xf, &mut prod[..2 * p - 1]);
        t.copy_from_slice(&prod[p..]);
        n += 1;
        div_prim(&mut t, n);
    }
    sum
}

fn check(x: &[u64], m: usize, l: usize) {
    // Trimming x's top zero limbs moves them into m, keeping x's value.
    let mut x = x.to_vec();
    let m = m + x.len();
    trim_lz(&mut x);
    let m = m - x.len();
    if x.is_empty() {
        return;
    }
    let mut got = vec![0; l];
    let z = exp_dyn(&x, m as u64, &mut got) as usize;
    let what = format!("x.len()={} m={m} l={l} z={z}", x.len());
    assert_ne!(got[l - 1], 0, "{what}: out's top limb must be nonzero");

    // e^x - 1 at p limbs: its top z limbs must be zero, and the l below them are out.
    let p = l + z + 3;
    let want = exp_oracle(&x, m, p);
    assert!(
        want[p - z..].iter().all(|&d| d == 0),
        "{what}: omitted limbs aren't zero"
    );
    let want = &want[p - z - l..p - z];
    let (hi, lo) = match cmp_buf(&got, want) {
        std::cmp::Ordering::Less => (want.to_vec(), got.clone()),
        _ => (got.clone(), want.to_vec()),
    };
    let mut diff = hi;
    sub_buf(&mut diff, &lo);
    trim_lz(&mut diff);
    assert!(
        diff.len() <= 1 && diff.first().map_or(true, |&d| d <= 1),
        "{what}: diff={diff:?}"
    );
}

#[test]
fn test_exp_dyn_random() {
    let mut seed = 0;
    for l in 1..=40 {
        for m in 0..3 {
            for xl in [1, 2, l / 2 + 1, l, l + 3] {
                seed += 1;
                let mut x = rand_vec(xl, seed);
                if m == 0 {
                    // keep x < 1/2 < ln(2)
                    *x.last_mut().unwrap() >>= 1 + seed % 5;
                }
                check(&x, m, l);
            }
        }
    }
}

#[test]
fn test_exp_dyn_single_bit_and_sparse() {
    for l in [1, 2, 5, 17] {
        for m in 0..3 {
            for b in [1u64, 2, 63] {
                let mut x = vec![0; 4];
                x[3] = 1 << (63 - b);
                check(&x, m, l);
                // set bits far apart, so most chunks are zero
                x[0] = 1;
                x[1] = 1 << 40;
                check(&x, m, l);
            }
        }
    }
}

#[test]
fn test_exp_dyn_near_ln2() {
    for l in [1, 2, 3, 8, 30] {
        for xl in [1, l, l + 2] {
            let mut x = vec![0; xl];
            ln2_dyn(&mut x);
            check(&x, 0, l);
        }
    }
}

// Tiny x, as argument reduction can produce: out must keep every limb significant
// however many leading zero limbs e^x - 1 has.
#[test]
fn test_exp_dyn_large_m() {
    let mut seed = 1000;
    for l in 1..=8 {
        for m in [0, 1, 2, 5, l, l + 1, 2 * l, 40] {
            for xl in [1, 2, l + 2] {
                seed += 1;
                let mut x = rand_vec(xl, seed);
                if m == 0 {
                    // keep x < 1/2 < ln(2)
                    *x.last_mut().unwrap() >>= 1;
                }
                check(&x, m, l);
            }
            // Top bit at a limb boundary (64 divides t0), the only case with an extra zero limb.
            let mut x = rand_vec(l + 1, seed);
            *x.last_mut().unwrap() |= 1 << 63;
            check(&x, m.max(1), l);
            // x just under 2^-t0, the top of e^x - 1's range.
            check(&vec![u64::MAX; l + 1], m.max(1), l);
            check(&[1 << 63], m.max(1), l);
        }
    }
}

#[test]
fn test_exp_dyn_large() {
    for (l, m) in [(64, 0), (150, 1), (300, 0)] {
        let mut x = rand_vec(l, l as u64);
        if m == 0 {
            *x.last_mut().unwrap() >>= 2;
        }
        check(&x, m, l);
    }
}
