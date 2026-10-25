use super::rand_vec;
use crate::utils::consts::consts::ln2_dyn;
use crate::utils::div::div_prim;
use crate::utils::mul::{mul_buf, mul_prim};
use crate::utils::trans::{
    exp_dyn, exp_dyn_at, exp_dyn_entry, exp_static, exp_static_at, exp_static_entry, ln_dyn,
    ln_dyn_at, ln_dyn_entry, ln_static, ln_static_at, ln_static_entry, ExpError, LnError,
};
use crate::utils::CapacityError;
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

type ExpFn<'a> = &'a dyn Fn(&[u64], i64, &mut [u64]) -> Result<u64, ExpError>;

// exp against exp_full_oracle within 1 ulp. Within an ulp of a power of 2^64 the
// two can pick neighboring exponents; the finer result then drops its low limb and
// both compare at the coarser scale.
fn check_full(x: &[u64], m: i64, l: usize, exp: ExpFn) {
    let what = format!("x.len()={} m={m} l={l}", x.len());
    let mut got = vec![0; l];
    let k = exp(x, m, &mut got).unwrap_or_else(|e| panic!("{what}: {e}"));
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
            assert_eq!(exp(x, m, &mut out), Ok(0), "x={x:?} m={m}");
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
            assert_eq!(exp(x, m, &mut out), Err(ExpError::Overflow), "x={x:?} m={m}");
        }
        // 2^69 times ln(2) truncated to two limbs, just under the bound: n = 2^69 - 1.
        let mut ln2 = [0; 2];
        ln2_dyn(&mut ln2);
        let x = [ln2[0] << 5, ln2[1] << 5 | ln2[0] >> 59, ln2[1] >> 59];
        assert_eq!(exp(&x, 1, &mut out), Ok(i64::MAX as u64));
        assert!(out[l - 1] >= 1 << 63);
        check_full(&x, 1, l, exp);
        check_full(&[0, 22], 1, l, exp);
    }
}

#[test]
fn test_exp_dyn_random() {
    exp_random(&exp_dyn, &[1, 2, 3, 8, 20], 7000);
}

#[test]
fn test_exp_dyn_large() {
    for (l, m) in [(64, 0), (64, 1), (150, -1), (150, 0)] {
        let mut x = rand_vec(l, l as u64 + 100);
        if m == 1 {
            x[l - 1] = x[l - 1] % 21 + 1;
        }
        check_full(&x, m, l, &exp_dyn);
    }
}

#[test]
fn test_exp_dyn_near_ln2_multiples() {
    exp_near_ln2_multiples(&exp_dyn, &[1, 2, 5, 12]);
}

#[test]
fn test_exp_dyn_zero_limbs_and_tiny() {
    exp_zero_limbs_and_tiny(&exp_dyn, &[1, 4]);
}

#[test]
fn test_exp_dyn_overflow() {
    exp_overflow(&exp_dyn, &[1, 3]);
}

// X in [2^-(t + 1), 2^-t) for every t from 0 past the square count, which r's t leading
// zeros cut down to none.
fn exp_lead_sweep(exp: ExpFn, ls: &[usize], bits: u64) {
    for &l in ls {
        for t in 0..=bits + 2 {
            let mut x = rand_vec(l + 2, 9500 + t);
            x[l + 1] = x[l + 1] >> (t % 64) | 1 << (63 - t % 64);
            check_full(&x, -1 - (t / 64) as i64, l, exp);
        }
    }
}

// Every square count must give the same answer within 1 ulp; 0 takes none. From 61 squares
// their frame has a second extra limb, and from 125 a third.
#[test]
fn test_exp_dyn_sqr_counts() {
    for bits in [0, 1, 8, 24, 60, 61, 130] {
        let exp: ExpFn = &|x, m, out| exp_dyn_at(x, m, out, bits);
        let ls = [1, 4, 16];
        exp_random(exp, &ls, 9000);
        exp_near_ln2_multiples(exp, &ls);
        exp_zero_limbs_and_tiny(exp, &ls);
        exp_overflow(exp, &ls);
        exp_lead_sweep(exp, &ls, bits);
    }
}

// On two threads a 150-limb core sums its chunks in parallel, so exp takes no squares; on
// one it sums them serially, after the squares. Both must be right. The choice itself only
// changes the time: in 64 tries at n mod 64 = 63 the two never differed by a bit.
#[test]
fn test_exp_dyn_sqr_threads() {
    let l = 150;
    for threads in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        for seed in [9900, 9901] {
            let x = rand_vec(l + 1, seed);
            check_full(&x, 0, l, &|x, m, out| pool.install(|| exp_dyn_at(x, m, out, 24)));
        }
    }
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
    let exp = |x: &[u64], m, out: &mut [u64]| exp_static_entry::<N>(x, m, out).unwrap();
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
        check(&x, m, 297, |x, m, out| exp_static_entry::<300>(x, m, out).unwrap());
    }
}

// Too small an N is the caller's to handle, so it's an error, not a panic.
#[test]
fn test_exp_static_entry_small_n() {
    let mut out = [u64::MAX; 6];
    let err = exp_static_entry::<8>(&[1 << 62], 0, &mut out);
    assert_eq!(err, Err(CapacityError { min_n: 9 }));
    assert_eq!(out, [u64::MAX; 6], "a capacity error leaves out alone");
}

// exp_static through the exp_dyn cases at N, from one limb to the most N allows.
fn exp_static_cases<const N: usize>() {
    let mut ls = vec![1, 3, N / 2, N - 3];
    ls.retain(|&l| l + 3 <= N);
    ls.sort();
    ls.dedup();
    exp_random(&exp_static::<N>, &ls, 8000 + N as u64);
    exp_near_ln2_multiples(&exp_static::<N>, &ls);
    exp_zero_limbs_and_tiny(&exp_static::<N>, &ls);
    exp_overflow(&exp_static::<N>, &ls);
}

#[test]
fn test_exp_static() {
    exp_static_cases::<4>();
    exp_static_cases::<5>();
    exp_static_cases::<6>();
    exp_static_cases::<7>();
    exp_static_cases::<9>();
    exp_static_cases::<16>();
    exp_static_cases::<64>();
}

// Wide enough that exp_static_entry gives r's chunks their own threads.
#[test]
fn test_exp_static_threaded() {
    for (m, seed) in [(0, 8500), (1, 8501)] {
        let mut x = rand_vec(300, seed);
        if m == 1 {
            x[299] = x[299] % 21 + 1;
        }
        check_full(&x, m, 297, &exp_static::<300>);
    }
}

#[test]
fn test_exp_static_small_n() {
    let small = Err(ExpError::Capacity(CapacityError { min_n: 9 }));
    let mut out = [u64::MAX; 6];
    assert_eq!(exp_static::<8>(&[1], 0, &mut out), small);
    // N is checked before any input, even one that needs no work.
    assert_eq!(exp_static::<8>(&[], 0, &mut out), small);
    assert_eq!(exp_static::<8>(&[1], 5, &mut out), small);
    assert_eq!(out, [u64::MAX; 6], "a capacity error leaves out alone");
}

// exp_static takes the squares once N has room for them (to l = N - 4 up to 60 squares,
// N - 5 up to 124 and N - 6 past that), where it agrees with exp_dyn_at; at N - 3 it takes
// none.
#[test]
fn test_exp_static_sqr_counts() {
    for bits in [8, 24, 61, 130] {
        let exp: ExpFn = &|x, m, out| exp_static_at::<40>(x, m, out, bits);
        let ls = [1, 4, 34, 35, 36, 37];
        exp_random(exp, &ls, 9700);
        exp_near_ln2_multiples(exp, &ls);
        exp_lead_sweep(exp, &ls, bits);
        for &l in &ls {
            let x = rand_vec(l + 1, 9800 + l as u64);
            let (mut got, mut want) = (vec![0; l], vec![0; l]);
            let k = exp(&x, 0, &mut got).unwrap();
            assert_eq!(exp_dyn_at(&x, 0, &mut want, bits), Ok(k));
            assert_within_ulp(&got, &want, &format!("bits={bits} l={l}"));
        }
    }
}

// From 1023 output limbs, ln(2)'s 1025 limbs pass its table and are summed, which
// needs one more limb of N. At exactly the bound, exp_static matches exp_dyn.
#[test]
fn test_exp_static_past_ln2_table() {
    fn check<const N: usize>(o: usize) {
        let mut x = rand_vec(o, o as u64);
        x[o - 1] >>= 1;
        let (mut got, mut want) = (vec![0; o], vec![0; o]);
        let k = exp_static::<N>(&x, 0, &mut got).unwrap();
        assert_eq!(exp_dyn(&x, 0, &mut want), Ok(k));
        assert_within_ulp(&got, &want, &format!("o={o} N={N}"));
    }
    check::<1025>(1022);
    check::<1027>(1023);
    let small = Err(ExpError::Capacity(CapacityError { min_n: 1027 }));
    assert_eq!(exp_static::<1026>(&[1], 0, &mut [0; 1023]), small);
}

// floor(a B^p / b) for a < b, one bit at a time, so the ln oracles share no division
// with ln_dyn.
fn frac_div(a: &[u64], b: &[u64], p: usize) -> Vec<u64> {
    let n = b.len() + 1;
    let (mut r, mut q) = (vec![0; n], vec![0; p]);
    r[..a.len()].copy_from_slice(a);
    let b = [b, &[0]].concat();
    for i in (0..64 * p).rev() {
        shl_buf(&mut r, 1);
        if cmp_buf(&r, &b) != Ordering::Less {
            sub_buf(&mut r, &b);
            q[i / 64] |= 1 << (i % 64);
        }
    }
    q
}

// 2 atanh(z) for z = a / b < 1/2 as a p-limb fraction, summing z^(2n + 1) / (2n + 1)
// directly with schoolbook products.
fn atanh2_oracle(a: &[u64], b: &[u64], p: usize) -> Vec<u64> {
    let z = frac_div(a, b, p);
    let mut prod = vec![0; 2 * p];
    prod[2 * p - 1] = mul_buf(&z, &z, &mut prod[..2 * p - 1]);
    let z2 = prod[p..].to_vec();
    let (mut sum, mut t, mut n) = (vec![0; p + 1], z, 0);
    while t.iter().any(|&l| l != 0) {
        let mut term = t.clone();
        div_prim(&mut term, 2 * n + 1);
        crate::utils::utils::add_buf(&mut sum, &term);
        prod[2 * p - 1] = mul_buf(&t, &z2, &mut prod[..2 * p - 1]);
        t.copy_from_slice(&prod[p..]);
        n += 1;
    }
    shl_buf(&mut sum, 1);
    sum.truncate(p);
    sum
}

// |ln(1 ± δ)| for δ = x 2^-64(m + x.len()) as a p-limb fraction: 2 atanh(δ / (2 ± δ)).
fn ln_oracle(x: &[u64], m: usize, neg: bool, p: usize) -> Vec<u64> {
    let mut d = vec![0; p];
    for (i, &l) in x.iter().enumerate() {
        if let Some(j) = (p + i).checked_sub(m + x.len()) {
            d[j] = l;
        }
    }
    let mut den = [&d[..], &[0]].concat();
    let two = [vec![0; p], vec![2]].concat();
    if neg {
        let mut t = two;
        sub_buf(&mut t, &den);
        den = t;
    } else {
        crate::utils::utils::add_buf(&mut den, &two);
    }
    atanh2_oracle(&d, &den, p)
}

type LnEntryFn<'a> = &'a dyn Fn(&[u64], u64, bool, &mut [u64]) -> u64;

fn check_ln_entry(x: &[u64], m: usize, neg: bool, l: usize, ln: LnEntryFn) {
    let mut x = x.to_vec();
    let m = m + x.len();
    trim_lz(&mut x);
    let m = m - x.len();
    if x.is_empty() {
        return;
    }
    let mut got = vec![0; l];
    let z = ln(&x, m as u64, neg, &mut got) as usize;
    let what = format!("x.len()={} m={m} neg={neg} l={l} z={z}", x.len());
    assert_ne!(got[l - 1], 0, "{what}: out's top limb must be nonzero");
    let p = l + z + 3;
    let want = ln_oracle(&x, m, neg, p);
    assert!(
        want[p - z..].iter().all(|&d| d == 0),
        "{what}: omitted limbs aren't zero"
    );
    assert_within_ulp(&got, &want[p - z - l..p - z], &what);
}

#[test]
fn test_ln_dyn_entry_random() {
    let ln: LnEntryFn = &|x, m, neg, out| ln_dyn_entry(x, m, neg, out);
    let mut seed = 9000;
    for l in (1..=20).chain([27, 40]) {
        for m in 0..3 {
            for xl in [1, 2, l / 2 + 1, l, l + 3] {
                for neg in [false, true] {
                    seed += 1;
                    let mut x = rand_vec(xl, seed);
                    if neg && m == 0 {
                        // keep δ < 1/2
                        *x.last_mut().unwrap() >>= 1;
                    }
                    check_ln_entry(&x, m, neg, l, ln);
                }
            }
        }
    }
}

// ln 2 = 2 atanh(1/3) at p limbs, cached as the oracles ask for the same p again and again.
fn ln2_oracle(p: usize) -> Vec<u64> {
    thread_local! {
        static LN2: std::cell::RefCell<std::collections::HashMap<usize, Vec<u64>>> =
            Default::default();
    }
    LN2.with(|c| {
        c.borrow_mut()
            .entry(p)
            .or_insert_with(|| atanh2_oracle(&[1], &[3], p))
            .clone()
    })
}

// |ln X| for X = 2^(64m) x, x's last limb worth 1, as (o, k, neg): |ln X| = 2^(64k) o, with
// o an l-limb number in [1, 2^64) whose top limb is its integer part (all zero for X = 1).
// Sums e ln 2 + ln u for u = X 2^-e in [1, 2), each 2 atanh at P fraction limbs, with
// ln 2 = 2 atanh(1/3), growing P until 3 guard limbs sit below o however much they cancel.
fn ln_full_oracle(x: &[u64], m: i64, l: usize) -> (Vec<u64>, i64, bool) {
    let mut x = x.to_vec();
    let base = m as i128 - x.len() as i128 + 1;
    trim_lz(&mut x);
    let bl = 64 * x.len() as i128 - x[x.len() - 1].leading_zeros() as i128;
    let e = 64 * base + bl - 1;
    // Enough limbs to keep u - 1's top l + 6, so only X = 1 sums to 0.
    let mut y = x.clone();
    *y.last_mut().unwrap() &= !(1 << ((bl - 1) % 64));
    trim_lz(&mut y);
    let ybits = y.last().map_or(0, |&t| 64 * y.len() as i128 - t.leading_zeros() as i128);
    let mut p = l + 7 + ((bl - 1 - ybits) / 64) as usize;
    loop {
        // u B^p and its neighbors u B^p -+ B^p
        let mut u = vec![0; p + 1];
        let sh = 64 * p as i128 - bl + 1;
        let (sl, sb) = split_sh(sh.unsigned_abs() as u64);
        if sh >= 0 {
            u[sl..sl + x.len()].copy_from_slice(&x);
            shl_buf(&mut u[sl..], sb);
        } else {
            let t = &x[sl..];
            let n = t.len().min(p + 1);
            u[..n].copy_from_slice(&t[..n]);
            shr_buf(&mut u, sb);
            if let Some(&hi) = t.get(n) {
                u[n - 1] |= hi << (64 - sb);
            }
        }
        let mut a = u.clone();
        a[p] = 0;
        let mut b = u;
        b[p] = 2;
        let lnu = atanh2_oracle(&a, &b, p);
        let ln2 = ln2_oracle(p);
        let ea = e.unsigned_abs();
        let mut s = vec![0; p + 3];
        s[p + 2] = mul_buf(&ln2, &[ea as u64, (ea >> 64) as u64], &mut s[..p + 2]);
        if e >= 0 {
            crate::utils::utils::add_buf(&mut s, &lnu);
        } else {
            sub_buf(&mut s, &lnu);
        }
        if y.is_empty() && e == 0 {
            return (vec![0; l], 0, false);
        }
        // They cancelled to below P, or to fewer than 3 guard limbs below o.
        let t = s.iter().rposition(|&d| d != 0).unwrap_or(0);
        if t < l + 2 {
            p += (l + 3 - t).max(p);
            continue;
        }
        return (s[t + 1 - l..t + 1].to_vec(), t as i64 - p as i64, e < 0);
    }
}

type LnFn<'a> = &'a dyn Fn(&[u64], i64, &mut [u64]) -> Result<(i64, bool), LnError>;

// ln against ln_full_oracle within 1 ulp, allowing either exponent next to a power of 2^64
// as check_full does.
fn check_ln_full(x: &[u64], m: i64, l: usize, ln: LnFn) {
    let what = format!("x.len()={} m={m} l={l}", x.len());
    let mut got = vec![0; l];
    let (k, neg) = ln(x, m, &mut got).unwrap_or_else(|e| panic!("{what}: {e}"));
    let (want, kw, nw) = ln_full_oracle(x, m, l);
    let what = format!("{what} k={k} kw={kw} neg={neg}");
    if want.iter().all(|&d| d == 0) {
        assert_eq!((k, neg, &got), (0, false, &want), "{what}: ln 1 = 0");
        return;
    }
    assert_eq!(neg, nw, "{what}: sign");
    assert_ne!(got[l - 1], 0, "{what}: out's top limb must be nonzero");
    let coarse = |v: &[u64]| [&v[1..], &[0]].concat();
    match k.cmp(&kw) {
        Ordering::Equal => assert_within_ulp(&got, &want, &what),
        Ordering::Less if k + 1 == kw => assert_within_ulp(&coarse(&got), &want, &what),
        Ordering::Greater if kw + 1 == k => assert_within_ulp(&got, &coarse(&want), &what),
        _ => panic!("{what}: exponents too far apart"),
    }
}

fn ln_random(ln: LnFn, ls: &[usize], mut seed: u64) {
    for &l in ls {
        for m in -3..=2 {
            for xl in [1, l, l + 2] {
                seed += 1;
                check_ln_full(&rand_vec(xl, seed), m, l, ln);
            }
        }
    }
}

#[test]
fn test_ln_dyn_random() {
    ln_random(&|x, m, out| ln_dyn(x, m, out), &[1, 2, 3, 8, 20], 9500);
}

// X = 2^e exactly (ln X = e ln 2), X = 1, X within 2^-64t of 1 from either side, X in
// [1/2, 1) (no -ln 2 + ln 2X cancellation), and m far from 0.
fn ln_special(ln: LnFn, ls: &[usize]) {
    for &l in ls {
        for m in [-3, -1, 0, 1, 4] {
            for b in [0, 1, 62, 63] {
                check_ln_full(&[1 << b], m, l, ln);
                check_ln_full(&[0, 0, 1 << b], m, l, ln);
            }
        }
        for t in [1, 2, l, l + 1, 2 * l + 3] {
            let mut x = vec![0; t + 1];
            (x[0], x[t]) = (1, 1);
            check_ln_full(&x, 0, l, ln);
            check_ln_full(&vec![u64::MAX; t], -1, l, ln);
            // the top bit of 1 - X on a limb boundary
            let mut x = vec![u64::MAX; t + 1];
            x[t - 1] = 1 << 63;
            check_ln_full(&x, -1, l, ln);
        }
        for seed in 0..4 {
            let mut x = rand_vec(l + 1, 9700 + seed);
            x[l] |= 1 << 63;
            check_ln_full(&x, -1, l, ln);
            check_ln_full(&rand_vec(l, 9800 + seed), 1 << 60, l, ln);
            check_ln_full(&rand_vec(l, 9900 + seed), -(1 << 60), l, ln);
        }
    }
}

fn ln_zero(ln: LnFn, l: usize) {
    let mut out = vec![u64::MAX; l];
    assert_eq!(ln(&[], 0, &mut out), Err(LnError::Zero));
    assert_eq!(ln(&[0, 0], 5, &mut out), Err(LnError::Zero));
    assert_eq!(ln(&[1], 0, &mut out), Ok((0, false)));
    assert_eq!(out, vec![0; l]);
    assert_eq!(ln(&[0, 1], 1, &mut out), Ok((0, false)));
}

#[test]
fn test_ln_dyn_special() {
    ln_special(&|x, m, out| ln_dyn(x, m, out), &[1, 2, 5]);
    ln_zero(&|x, m, out| ln_dyn(x, m, out), 3);
}

#[test]
fn test_ln_dyn_large() {
    for (l, m) in [(40, 0), (40, -1), (64, 3)] {
        check_ln_full(&rand_vec(l + 1, l as u64 + 300), m, l, &|x, m, out| ln_dyn(x, m, out));
    }
}

// Every sqrt count must give the same answer within 1 ulp; 0 takes none.
#[test]
fn test_ln_dyn_sqrt_counts() {
    for a in [0.0, 0.5, 1.0, 2.0, 3.0] {
        let ln: LnFn = &|x, m, out| ln_dyn_at(x, m, out, a);
        for l in [1, 4, 16] {
            for (m, seed) in [(0, 1), (-1, 2), (2, 3)] {
                check_ln_full(&rand_vec(l + 1, 10_000 + seed), m, l, ln);
            }
            // u just above 1 and just below 2
            let mut x = vec![0; l + 2];
            (x[l], x[l + 1]) = (1 << 40, 1);
            check_ln_full(&x, 0, l, ln);
            check_ln_full(&vec![u64::MAX; l + 2], 0, l, ln);
        }
    }
}

#[test]
fn test_ln_dyn_entry_single_bit_and_sparse() {
    let ln: LnEntryFn = &|x, m, neg, out| ln_dyn_entry(x, m, neg, out);
    for l in [1, 2, 5, 17] {
        for m in 0..3 {
            for b in [1u64, 2, 63] {
                for neg in [false, true] {
                    let mut x = vec![0; 4];
                    x[3] = 1 << (63 - b);
                    check_ln_entry(&x, m, neg, l, ln);
                    // set bits far apart, so the residual skips long runs
                    (x[0], x[1]) = (1, 1 << 40);
                    check_ln_entry(&x, m, neg, l, ln);
                }
            }
        }
    }
}

// δ near 1 for ln(1 + δ), and δ = 1/2 or just under for ln(1 - δ).
#[test]
fn test_ln_dyn_entry_extremes() {
    let ln: LnEntryFn = &|x, m, neg, out| ln_dyn_entry(x, m, neg, out);
    for l in [1, 2, 3, 8, 30] {
        for xl in [1, l, l + 2] {
            check_ln_entry(&vec![u64::MAX; xl], 0, false, l, ln);
            let mut half = vec![0; xl];
            half[xl - 1] = 1 << 63;
            check_ln_entry(&half, 0, true, l, ln);
            check_ln_entry(&vec![u64::MAX >> 1; xl], 0, true, l, ln);
        }
    }
}

// Tiny δ, as argument reduction can produce: out keeps every limb significant however
// many leading zero limbs L has, including when 2 a S and then every chunk drop below the
// frame and only the tail is left.
#[test]
fn test_ln_dyn_entry_large_m() {
    let ln: LnEntryFn = &|x, m, neg, out| ln_dyn_entry(x, m, neg, out);
    let mut seed = 11_000;
    for l in 1..=8 {
        for m in [0, 1, 2, 5, l, l + 1, 2 * l, 40] {
            for neg in [false, true] {
                for xl in [1, 2, l + 2] {
                    seed += 1;
                    let mut x = rand_vec(xl, seed);
                    if m == 0 {
                        *x.last_mut().unwrap() >>= 1;
                    }
                    check_ln_entry(&x, m, neg, l, ln);
                }
                // L's top bit next to a limb boundary, the only case with an extra limb.
                let mut x = rand_vec(l + 1, seed);
                *x.last_mut().unwrap() |= 1 << 63;
                check_ln_entry(&x, m.max(1), neg, l, ln);
                check_ln_entry(&vec![u64::MAX; l + 1], m.max(1), neg, l, ln);
                check_ln_entry(&[1 << 63], m.max(1), neg, l, ln);
                check_ln_entry(&[1 << 62], m.max(1), neg, l, ln);
            }
        }
    }
}

#[test]
fn test_ln_dyn_entry_large() {
    let ln: LnEntryFn = &|x, m, neg, out| ln_dyn_entry(x, m, neg, out);
    for (l, m, neg) in [(64, 0, false), (100, 1, true), (150, 0, false)] {
        let mut x = rand_vec(l, l as u64);
        *x.last_mut().unwrap() >>= 2;
        check_ln_entry(&x, m, neg, l, ln);
    }
}

// The parallel series combine like the serial ones, so the paths agree bit for bit. A
// one-thread pool always takes the serial path.
#[test]
fn test_ln_dyn_entry_parallel_matches_serial() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let check = |x: &[u64], m: u64, neg: bool, l: usize| {
        let (mut par, mut ser) = (vec![0; l], vec![0; l]);
        let zp = ln_dyn_entry(x, m, neg, &mut par);
        let zs = pool.install(|| ln_dyn_entry(x, m, neg, &mut ser));
        let what = format!("x.len()={} m={m} neg={neg} l={l}", x.len());
        assert_eq!(zp, zs, "{what}: z");
        assert_eq!(par, ser, "{what}: out");
    };
    let mut seed = 12_000;
    for l in (1..=64).step_by(7).chain([150, 300, 1000]) {
        for m in [0, 1, 40] {
            for neg in [false, true] {
                seed += 1;
                let mut x = rand_vec(l, seed);
                *x.last_mut().unwrap() >>= 1;
                check(&x, m, neg, l);
            }
        }
    }
}

// ln_static_entry against the oracle at N, from one limb to the most N allows.
fn ln_static_entry_at<const N: usize>() {
    let ln: LnEntryFn = &|x, m, neg, out| ln_static_entry::<N>(x, m, neg, out).unwrap();
    let mut seed = 13_000 + N as u64;
    let mut ls = vec![1, N / 2, N - 3];
    ls.retain(|&l| l + 3 <= N);
    ls.sort();
    ls.dedup();
    for &l in &ls {
        for m in [0, 1, 3, l + 1, 40] {
            for neg in [false, true] {
                for xl in [1, l, l + 3] {
                    seed += 1;
                    let mut x = rand_vec(xl, seed);
                    if m == 0 {
                        *x.last_mut().unwrap() >>= 1;
                    }
                    check_ln_entry(&x, m, neg, l, ln);
                }
                let mut sparse = vec![0; l + 2];
                (sparse[0], sparse[l / 2], sparse[l + 1]) = (1, 1 << 40, 1 << 61);
                check_ln_entry(&sparse, m, neg, l, ln);
                check_ln_entry(&[1 << 60], m, neg, l, ln);
                if m > 0 {
                    let mut x = rand_vec(l + 1, seed);
                    *x.last_mut().unwrap() |= 1 << 63;
                    check_ln_entry(&x, m, neg, l, ln);
                    check_ln_entry(&vec![u64::MAX; l + 1], m, neg, l, ln);
                }
            }
        }
        check_ln_entry(&vec![u64::MAX; l + 2], 0, false, l, ln);
        check_ln_entry(&[1 << 63], 0, true, l, ln);
    }
}

#[test]
fn test_ln_static_entry() {
    ln_static_entry_at::<4>();
    ln_static_entry_at::<5>();
    ln_static_entry_at::<6>();
    ln_static_entry_at::<8>();
    ln_static_entry_at::<16>();
    ln_static_entry_at::<64>();
}

// Wide enough that ln_static_entry gives its chunks their own threads, which must match
// the dynamic entry.
#[test]
fn test_ln_static_entry_threaded() {
    for (m, neg, seed) in [(0, false, 14_000), (0, true, 14_001), (1, false, 14_002)] {
        let mut x = rand_vec(297, seed);
        *x.last_mut().unwrap() >>= 1;
        if m == 1 {
            // The oracle converges fast once δ is below 2^-64.
            let ln: LnEntryFn = &|x, m, neg, out| ln_static_entry::<300>(x, m, neg, out).unwrap();
            check_ln_entry(&x, m, neg, 297, ln);
        }
        let (mut got, mut want) = (vec![0; 297], vec![0; 297]);
        let z = ln_static_entry::<300>(&x, m as u64, neg, &mut got).unwrap();
        assert_eq!(z, ln_dyn_entry(&x, m as u64, neg, &mut want));
        assert_within_ulp(&got, &want, &format!("m={m} neg={neg}"));
    }
}

#[test]
fn test_ln_static_entry_small_n() {
    let mut out = [u64::MAX; 6];
    let err = ln_static_entry::<8>(&[1 << 62], 0, false, &mut out);
    assert_eq!(err, Err(CapacityError { min_n: 9 }));
    assert_eq!(out, [u64::MAX; 6], "a capacity error leaves out alone");
}

// ln_static through the ln_dyn cases at N, from one limb to the most N allows. Only the
// wider N leave the square roots room.
fn ln_static_cases<const N: usize>() {
    let mut ls = vec![1, 3, N / 2, N - 4];
    ls.retain(|&l| l + 4 <= N);
    ls.sort();
    ls.dedup();
    ln_random(&|x, m, out| ln_static::<N>(x, m, out), &ls, 15_000 + N as u64);
    // The oracle's cancellation near 1 grows with l, so wide l stop at the random cases.
    ls.retain(|&l| l <= 12);
    ln_special(&|x, m, out| ln_static::<N>(x, m, out), &ls);
    ln_zero(&|x, m, out| ln_static::<N>(x, m, out), N - 4);
}

#[test]
fn test_ln_static() {
    ln_static_cases::<5>();
    ln_static_cases::<6>();
    ln_static_cases::<7>();
    ln_static_cases::<9>();
    ln_static_cases::<16>();
    ln_static_cases::<64>();
}

#[test]
fn test_ln_static_sqrt_counts() {
    for a in [0.0, 1.0, 3.0] {
        let ln: LnFn = &|x, m, out| ln_static_at::<40>(x, m, out, a);
        for l in [1, 4, 16] {
            for (m, seed) in [(0, 1), (-1, 2), (2, 3)] {
                check_ln_full(&rand_vec(l + 1, 16_000 + seed), m, l, ln);
            }
        }
    }
}

#[test]
fn test_ln_static_small_n() {
    let small = Err(LnError::Capacity(CapacityError { min_n: 10 }));
    let mut out = [u64::MAX; 6];
    assert_eq!(ln_static::<9>(&[3], 0, &mut out), small);
    // N is checked before any input, even one that needs no work.
    assert_eq!(ln_static::<9>(&[], 0, &mut out), small);
    assert_eq!(ln_static::<9>(&[1], 0, &mut out), small);
    assert_eq!(out, [u64::MAX; 6], "a capacity error leaves out alone");
}

// Past 1022 output limbs, ln(2)'s o + 2 limbs pass its table and are summed. At exactly the
// bound, ln_static matches ln_dyn.
#[test]
fn test_ln_static_past_ln2_table() {
    fn check<const N: usize>(o: usize) {
        let x = rand_vec(o, o as u64);
        let (mut got, mut want) = (vec![0; o], vec![0; o]);
        let k = ln_static::<N>(&x, 3, &mut got).unwrap();
        assert_eq!(ln_dyn(&x, 3, &mut want), Ok(k));
        assert_within_ulp(&got, &want, &format!("o={o} N={N}"));
    }
    check::<1026>(1022);
    check::<1027>(1023);
}
