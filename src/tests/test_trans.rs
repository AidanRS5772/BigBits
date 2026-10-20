use super::rand_vec;
use crate::utils::consts::consts::ln2_dyn;
use crate::utils::div::div_prim;
use crate::utils::mul::mul_buf;
use crate::utils::trans::{exp_dyn, exp_dyn_at, ExpCutoffs};
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

fn check(x: &[u64], m: usize, l: usize, exp: impl Fn(&[u64], u64, &mut [u64]) -> u64) {
    // Trimming x's top zero limbs moves them into m, keeping x's value.
    let mut x = x.to_vec();
    let m = m + x.len();
    trim_lz(&mut x);
    let m = m - x.len();
    if x.is_empty() {
        return;
    }
    let mut got = vec![0; l];
    let z = exp(&x, m as u64, &mut got) as usize;
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
                check(&x, m, l, exp_dyn);
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
                check(&x, m, l, exp_dyn);
                // set bits far apart, so most chunks are zero
                x[0] = 1;
                x[1] = 1 << 40;
                check(&x, m, l, exp_dyn);
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
            check(&x, 0, l, exp_dyn);
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
                check(&x, m, l, exp_dyn);
            }
            // Top bit at a limb boundary (64 divides t0), the only case with an extra zero limb.
            let mut x = rand_vec(l + 1, seed);
            *x.last_mut().unwrap() |= 1 << 63;
            check(&x, m.max(1), l, exp_dyn);
            // x just under 2^-t0, the top of e^x - 1's range.
            check(&vec![u64::MAX; l + 1], m.max(1), l, exp_dyn);
            check(&[1 << 63], m.max(1), l, exp_dyn);
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
        check(&x, m, l, exp_dyn);
    }
}

// Parallel chunks combine in the same order as the serial loop, so the two paths must
// agree bit for bit.
fn check_par_matches_serial(x: &[u64], m: u64, l: usize) {
    for chunk_min_bits in [0, 64, 200] {
        let at = |par_work| ExpCutoffs { chunk_min_bits, par_work };
        let (mut par, mut ser) = (vec![0; l], vec![0; l]);
        let zp = exp_dyn_at(x, m, &mut par, at(0));
        let zs = exp_dyn_at(x, m, &mut ser, at(usize::MAX));
        let what = format!("x.len()={} m={m} l={l} min={chunk_min_bits}", x.len());
        assert_eq!(zp, zs, "{what}: z");
        assert_eq!(par, ser, "{what}: out");
    }
}

#[test]
fn test_exp_dyn_parallel_matches_serial() {
    let mut seed = 2000;
    for l in (1..=64).chain([150, 300, 1000]) {
        for m in [0, 1, l as u64, 40] {
            seed += 1;
            let mut x = rand_vec(l, seed);
            if m == 0 {
                *x.last_mut().unwrap() >>= 1;
            }
            check_par_matches_serial(&x, m, l);
            // Sparse: most chunks are zero, so few run.
            let mut sparse = vec![0; l + 2];
            sparse[l + 1] = 1 << 40;
            sparse[0] = 1;
            check_par_matches_serial(&sparse, m, l);
        }
    }
    // A single-bit x has no chunks, only the first series.
    check_par_matches_serial(&[1 << 62], 0, 8);

    // Forcing the parallel path must also work on a one-thread pool.
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let mut x = rand_vec(200, 77);
        *x.last_mut().unwrap() >>= 1;
        check_par_matches_serial(&x, 0, 200);
    });
}

// Any first-chunk width is valid, from none (0) to one chunk covering all of x
// (u64::MAX).
#[test]
fn test_exp_dyn_chunk_min_widths() {
    for chunk_min_bits in [0, 1, 2, 16, 64, 200, u64::MAX] {
        let exp = |x: &[u64], m: u64, out: &mut [u64]| {
            exp_dyn_at(x, m, out, ExpCutoffs { chunk_min_bits, ..ExpCutoffs::DEFAULT })
        };
        let mut seed = 3000;
        for l in 1..=24 {
            for m in [0, 1, 3] {
                seed += 1;
                let mut x = rand_vec(l + 1, seed);
                if m == 0 {
                    *x.last_mut().unwrap() >>= 1;
                }
                check(&x, m, l, exp);
            }
        }
        // Sparse x, a lone top bit, and a top bit on a limb boundary.
        for l in [1, 5, 17] {
            let mut x = vec![0; 4];
            x[3] = 1 << 61;
            check(&x, 0, l, exp);
            x[0] = 1;
            x[1] = 1 << 40;
            check(&x, 0, l, exp);
            let mut x = rand_vec(l + 1, l as u64);
            *x.last_mut().unwrap() |= 1 << 63;
            check(&x, 1, l, exp);
        }
    }
}
