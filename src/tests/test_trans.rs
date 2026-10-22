use super::rand_vec;
use crate::utils::consts::consts::ln2_dyn;
use crate::utils::div::div_prim;
use crate::utils::mul::{mul_buf, mul_prim};
use crate::utils::trans::{exp_dyn, exp_dyn_entry, exp_static, exp_static_entry};
use crate::utils::utils::{buf_len, cmp_buf, shl_buf, shr_buf, split_sh, sub_buf, trim_lz};
use std::cmp::Ordering;

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
        // schoolbook, so the oracle shares no fast multiplication path with exp_dyn_entry
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
    assert_within_ulp(&got, &want[p - z - l..p - z], &what);
}

fn assert_within_ulp(got: &[u64], want: &[u64], what: &str) {
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

// e^X for X = 2^(64m) x, x's last limb worth 1, as (o, k): e^X = 2^(64k) o, with o an
// l-limb number in [1, 2^64) whose top limb is its integer part. Squares e^(X / 2^j)
// j times, so it shares neither ln(2) nor division with exp_dyn.
fn exp_full_oracle(x: &[u64], m: i64, l: usize) -> (Vec<u64>, u64) {
    let mut x = x.to_vec();
    let m = m as i128 - x.len() as i128;
    trim_lz(&mut x);
    // x's top limb is worth 2^(64m), so X < 2^(64m + bits) and y = X / 2^j < 2^-8.
    let m = m + x.len() as i128;
    let bits = 64 - x[x.len() - 1].leading_zeros() as i128;
    let j = (64 * m + bits + 8).max(0);
    // y = x 2^-fb, padded out to whole limbs for exp_oracle.
    let fb = j - 64 * (m - x.len() as i128 + 1);
    let pad = (64 - fb % 64) % 64;
    let mut y = x;
    y.push(0);
    shl_buf(&mut y, pad as u8);
    trim_lz(&mut y);
    let ym = ((fb + pad) / 64) as usize - y.len();

    let w = l + 4;
    let mut a = exp_oracle(&y, ym, w);
    a.push(1);
    // e^X = a 2^e throughout; every square keeps a's top w + 1 limbs.
    let mut e = -64 * w as i128;
    let mut prod = vec![0; 2 * w + 2];
    for _ in 0..j {
        prod[2 * w + 1] = mul_buf(&a, &a, &mut prod[..2 * w + 1]);
        let drop = buf_len(&prod) - w - 1;
        a.copy_from_slice(&prod[drop..drop + w + 1]);
        e = 2 * e + 64 * drop as i128;
    }
    let top = e + 64 * (w + 1) as i128 - 1 - a[w].leading_zeros() as i128;
    let k = top.div_euclid(64);
    let (sl, sb) = split_sh((64 * (k - l as i128 + 1) - e) as u64);
    let mut o = a[sl..].to_vec();
    shr_buf(&mut o, sb);
    assert!(o[l..].iter().all(|&d| d == 0));
    o.truncate(l);
    (o, k as u64)
}

type ExpFn = fn(&[u64], i64, &mut [u64]) -> Option<u64>;

// exp against exp_full_oracle within 1 ulp. Within an ulp of a power of 2^64 the
// two can pick neighboring exponents; the finer result then drops its low limb and
// both compare at the coarser scale.
fn check_full(x: &[u64], m: i64, l: usize, exp: ExpFn) {
    let what = format!("x.len()={} m={m} l={l}", x.len());
    let mut got = vec![0; l];
    let k = exp(x, m, &mut got).unwrap_or_else(|| panic!("{what}: overflow"));
    assert_ne!(got[l - 1], 0, "{what}: out's top limb must be nonzero");
    let (want, kw) = exp_full_oracle(x, m, l);
    let what = format!("{what} k={k} kw={kw}");
    let coarse = |v: &[u64]| [&v[1..], &[0]].concat();
    match k.cmp(&kw) {
        Ordering::Equal => assert_within_ulp(&got, &want, &what),
        Ordering::Less if k + 1 == kw => assert_within_ulp(&coarse(&got), &want, &what),
        Ordering::Greater if kw + 1 == k => assert_within_ulp(&got, &coarse(&want), &what),
        _ => panic!("{what}: exponents too far apart"),
    }
}

fn one(l: usize) -> Vec<u64> {
    let mut v = vec![0; l];
    v[l - 1] = 1;
    v
}

fn exp_random(exp: ExpFn, ls: &[usize], mut seed: u64) {
    for &l in ls {
        for m in -3..=1 {
            for xl in [1, l, l + 2] {
                seed += 1;
                let mut x = rand_vec(xl, seed);
                if m == 1 {
                    // 2^64 <= X < 22 * 2^64 < 2^69 ln(2), so n / 64 fits i64
                    x[xl - 1] = x[xl - 1] % 21 + 1;
                }
                check_full(&x, m, l, exp);
            }
        }
    }
}

// X = c ln(2) leaves r near 0 or ln(2); when 64 divides c, e^X is near 2^c, where
// either neighboring exponent is right.
fn exp_near_ln2_multiples(exp: ExpFn, ls: &[usize]) {
    for &l in ls {
        for c in [1, 5, 63, 64, 65, 192, 64_000, 64_001] {
            let mut x = vec![0; l + 6];
            ln2_dyn(&mut x[..l + 5]);
            let top = mul_prim(&mut x[..l + 5], c);
            x[l + 5] = top;
            check_full(&x, 0, l, exp);
        }
    }
}

// Top zero limbs move into m; X = 0 and X below out's precision give exactly 1.
fn exp_zero_limbs_and_tiny(exp: ExpFn, ls: &[usize]) {
    let x = rand_vec(3, 7500);
    for &l in ls {
        let (mut a, mut b) = (vec![0; l], vec![0; l]);
        let ka = exp(&[x[0], x[1], 0, 0], 0, &mut a);
        let kb = exp(&x[..2], -2, &mut b);
        assert_eq!((ka, &a), (kb, &b));
        let ka = exp(&[x[0], x[1], x[2] % 22, 0], 2, &mut a);
        let kb = exp(&[x[0], x[1], x[2] % 22], 1, &mut b);
        assert_eq!((ka, &a), (kb, &b));

        for (x, m) in [
            (&[][..], 0),
            (&[0, 0], 5),
            (&[7], i64::MIN),
            (&[3, 0, 0], i64::MIN + 1),
            (&[u64::MAX], -(l as i64) - 3),
        ] {
            let mut out = vec![u64::MAX; l];
            assert_eq!(exp(x, m, &mut out), Some(0), "x={x:?} m={m}");
            assert_eq!(out, one(l), "x={x:?} m={m}");
        }
    }
}

// n / 64 must fit i64: n < 2^69, i.e. X < 2^69 ln(2) ~ 22.18 * 2^64.
fn exp_overflow(exp: ExpFn, ls: &[usize]) {
    for &l in ls {
        let mut out = vec![0; l];
        for (x, m) in [
            (&[1][..], 2),
            (&[1], i64::MAX),
            (&[0, 23], 1),
            (&[u64::MAX, u64::MAX, 22], 1),
            (&[5, 1], 2),
        ] {
            assert_eq!(exp(x, m, &mut out), None, "x={x:?} m={m}");
        }
        // 2^69 times ln(2) truncated to two limbs, just under the bound: n = 2^69 - 1.
        let mut ln2 = [0; 2];
        ln2_dyn(&mut ln2);
        let x = [ln2[0] << 5, ln2[1] << 5 | ln2[0] >> 59, ln2[1] >> 59];
        assert_eq!(exp(&x, 1, &mut out), Some(i64::MAX as u64));
        assert!(out[l - 1] >= 1 << 63);
        check_full(&x, 1, l, exp);
        check_full(&[0, 22], 1, l, exp);
    }
}

#[test]
fn test_exp_dyn_random() {
    exp_random(exp_dyn, &[1, 2, 3, 8, 20], 7000);
}

#[test]
fn test_exp_dyn_large() {
    for (l, m) in [(64, 0), (64, 1), (150, -1), (150, 0)] {
        let mut x = rand_vec(l, l as u64 + 100);
        if m == 1 {
            x[l - 1] = x[l - 1] % 21 + 1;
        }
        check_full(&x, m, l, exp_dyn);
    }
}

#[test]
fn test_exp_dyn_near_ln2_multiples() {
    exp_near_ln2_multiples(exp_dyn, &[1, 2, 5, 12]);
}

#[test]
fn test_exp_dyn_zero_limbs_and_tiny() {
    exp_zero_limbs_and_tiny(exp_dyn, &[1, 4]);
}

#[test]
fn test_exp_dyn_overflow() {
    exp_overflow(exp_dyn, &[1, 3]);
}

#[test]
fn test_exp_dyn_entry_random() {
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
                check(&x, m, l, exp_dyn_entry);
            }
        }
    }
}

#[test]
fn test_exp_dyn_entry_single_bit_and_sparse() {
    for l in [1, 2, 5, 17] {
        for m in 0..3 {
            for b in [1u64, 2, 63] {
                let mut x = vec![0; 4];
                x[3] = 1 << (63 - b);
                check(&x, m, l, exp_dyn_entry);
                // set bits far apart, so most chunks are zero
                x[0] = 1;
                x[1] = 1 << 40;
                check(&x, m, l, exp_dyn_entry);
            }
        }
    }
}

#[test]
fn test_exp_dyn_entry_near_ln2() {
    for l in [1, 2, 3, 8, 30] {
        for xl in [1, l, l + 2] {
            let mut x = vec![0; xl];
            ln2_dyn(&mut x);
            check(&x, 0, l, exp_dyn_entry);
        }
    }
}

// Tiny x, as argument reduction can produce: out must keep every limb significant
// however many leading zero limbs e^x - 1 has.
#[test]
fn test_exp_dyn_entry_large_m() {
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
                check(&x, m, l, exp_dyn_entry);
            }
            // Top bit at a limb boundary (64 divides t0), the only case with an extra zero limb.
            let mut x = rand_vec(l + 1, seed);
            *x.last_mut().unwrap() |= 1 << 63;
            check(&x, m.max(1), l, exp_dyn_entry);
            // x just under 2^-t0, the top of e^x - 1's range.
            check(&vec![u64::MAX; l + 1], m.max(1), l, exp_dyn_entry);
            check(&[1 << 63], m.max(1), l, exp_dyn_entry);
        }
    }
}

#[test]
fn test_exp_dyn_entry_large() {
    for (l, m) in [(64, 0), (150, 1), (300, 0)] {
        let mut x = rand_vec(l, l as u64);
        if m == 0 {
            *x.last_mut().unwrap() >>= 2;
        }
        check(&x, m, l, exp_dyn_entry);
    }
}

// Parallel chunks combine in the same order as the serial loop, so the two paths must
// agree bit for bit. A one-thread pool always takes the serial path.
#[test]
fn test_exp_dyn_entry_parallel_matches_serial() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let check_par_matches_serial = |x: &[u64], m: u64, l: usize| {
        let (mut par, mut ser) = (vec![0; l], vec![0; l]);
        let zp = exp_dyn_entry(x, m, &mut par);
        let zs = pool.install(|| exp_dyn_entry(x, m, &mut ser));
        let what = format!("x.len()={} m={m} l={l}", x.len());
        assert_eq!(zp, zs, "{what}: z");
        assert_eq!(par, ser, "{what}: out");
    };
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
}

// exp_static_entry against the oracle at N, from one limb to the most N allows,
// over the shapes the dynamic tests use.
fn static_entry_at<const N: usize>() {
    let exp = exp_static_entry::<N>;
    let mut seed = 5000;
    let mut ls = vec![1, N / 2, N - 3];
    ls.retain(|&l| l + 3 <= N);
    ls.sort();
    ls.dedup();
    for &l in &ls {
        for m in [0, 1, 3, l + 1, 40] {
            for xl in [1, l, l + 3] {
                seed += 1;
                let mut x = rand_vec(xl, seed);
                if m == 0 {
                    *x.last_mut().unwrap() >>= 1;
                }
                check(&x, m, l, exp);
            }
            let mut sparse = vec![0; l + 2];
            sparse[0] = 1;
            sparse[l / 2] = 1 << 40;
            sparse[l + 1] = 1 << 61;
            check(&sparse, m, l, exp);
            check(&[1 << 60], m, l, exp);
            if m > 0 {
                let mut x = rand_vec(l + 1, seed);
                *x.last_mut().unwrap() |= 1 << 63;
                check(&x, m, l, exp);
                check(&vec![u64::MAX; l + 1], m, l, exp);
            }
        }
        for xl in [1, l, l + 2] {
            let mut x = vec![0; xl];
            ln2_dyn(&mut x);
            check(&x, 0, l, exp);
        }
    }
}

#[test]
fn test_exp_static_entry() {
    static_entry_at::<4>();
    static_entry_at::<5>();
    static_entry_at::<6>();
    static_entry_at::<8>();
    static_entry_at::<16>();
    static_entry_at::<64>();
}

// Wide enough that exp_static_entry gives its chunks their own threads.
#[test]
fn test_exp_static_entry_threaded() {
    for (m, seed) in [(0, 6000), (1, 6001)] {
        let mut x = rand_vec(297, seed);
        if m == 0 {
            *x.last_mut().unwrap() >>= 1;
        }
        check(&x, m, 297, exp_static_entry::<300>);
    }
}

#[test]
#[should_panic(expected = "out.len() + 3 <= N")]
fn test_exp_static_entry_small_n() {
    exp_static_entry::<8>(&[1 << 62], 0, &mut [0; 6]);
}

// exp_static through the exp_dyn cases at N, from one limb to the most N allows.
fn exp_static_at<const N: usize>() {
    let mut ls = vec![1, 3, N / 2, N - 4];
    ls.retain(|&l| l + 4 <= N);
    ls.sort();
    ls.dedup();
    exp_random(exp_static::<N>, &ls, 8000 + N as u64);
    exp_near_ln2_multiples(exp_static::<N>, &ls);
    exp_zero_limbs_and_tiny(exp_static::<N>, &ls);
    exp_overflow(exp_static::<N>, &ls);
}

#[test]
fn test_exp_static() {
    exp_static_at::<5>();
    exp_static_at::<6>();
    exp_static_at::<7>();
    exp_static_at::<9>();
    exp_static_at::<16>();
    exp_static_at::<64>();
}

// Wide enough that exp_static_entry gives r's chunks their own threads.
#[test]
fn test_exp_static_threaded() {
    for (m, seed) in [(0, 8500), (1, 8501)] {
        let mut x = rand_vec(300, seed);
        if m == 1 {
            x[299] = x[299] % 21 + 1;
        }
        check_full(&x, m, 296, exp_static::<300>);
    }
}

#[test]
#[should_panic(expected = "out.len() + 4 <= N")]
fn test_exp_static_small_n() {
    exp_static::<8>(&[1], 0, &mut [0; 5]);
}

// ln(2) past its 1024-limb table is summed at out.len() + 2 limbs.
#[test]
#[should_panic(expected = "out.len() + 5 <= N")]
fn test_exp_static_past_ln2_table_small_n() {
    exp_static::<1028>(&[1], 0, &mut [0; 1024]);
}
