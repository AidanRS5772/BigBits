use super::{rand_nonzero_vec, rand_vec};
use crate::utils::mul::*;
use crate::utils::utils::{buf_len, trim_lz};
use crate::utils::{
    CHUNKING_KARATSUBA_CUTOFF, FFT_SQR_CUTOFF, KARATSUBA_CUTOFF, KARATSUBA_SQR_CUTOFF,
};

// ─── boundary finders ────────────────────────────────────────────────────────
//
// These use the pub(crate) dispatch predicates from mul.rs directly, so they
// stay correct whenever a constant changes.

/// A long length that keeps `s = CHUNKING_KARATSUBA_CUTOFF` well inside the
/// `s ≤ half` branch of `is_school`, regardless of the constant's value.
fn flat_branch_l() -> usize {
    4 * CHUNKING_KARATSUBA_CUTOFF
}

/// Walk the parabola peak (`l ≈ 1.5 · KARATSUBA_CUTOFF`) and return
/// `(l, s_school, s_karatsuba)` that straddle the curved boundary.
fn curved_boundary() -> (usize, usize, usize) {
    let c = KARATSUBA_CUTOFF as f64;
    let l = (1.5 * c).round() as usize;
    // Start from half+1 (s > half branch) and walk up until is_school flips.
    let half = (l + 1) / 2;
    let s_school = (half + 1..)
        .take_while(|&s| is_school(l, s))
        .last()
        .expect("no school size found in curved branch");
    (l, s_school, s_school + 1)
}

/// Smallest balanced size n where `is_school(n, n)` is false (first karatsuba).
fn first_balanced_karatsuba() -> usize {
    (3..).find(|&n| !is_school(n, n)).unwrap()
}

/// For a given `l`, find s values that straddle the Chunking / Recurse
/// boundary inside karatsuba (`(l+1)/2`), both past the school region.
fn chunking_recurse_boundary(l: usize) -> (usize, usize) {
    let half = (l + 1) / 2;
    let s_chunk = (1..half)
        .rev()
        .find(|&s| !is_school(l, s))
        .expect("no chunking size");
    let s_rec = (half..=l)
        .find(|&s| !is_school(l, s))
        .expect("no recurse size");
    (s_chunk, s_rec)
}

// ─── misc helpers ────────────────────────────────────────────────────────────

/// Schoolbook reference result (normalized).
fn mul_ref(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; a.len() + b.len() - 1];
    let c = mul_buf(a, b, &mut out);
    if c > 0 {
        out.push(c);
    }
    trim_lz(&mut out);
    out
}

fn norm(mut v: Vec<u64>, c: u64) -> Vec<u64> {
    if c > 0 {
        v.push(c);
    }
    v
}

fn norm_mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (v, c) = mul_vec(a, b);
    norm(v, c)
}

fn norm_sqr(a: &[u64]) -> Vec<u64> {
    let (v, c) = sqr_vec(a);
    norm(v, c)
}

// ─── mul_prim ───────────────────────────────────────────────────────────────

#[test]
fn test_mul_prim_by_zero() {
    let mut v = vec![42u64, 7];
    let c = mul_prim(&mut v, 0);
    assert_eq!(v, vec![0, 0]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim_by_one() {
    let mut v = vec![u64::MAX, 1];
    let c = mul_prim(&mut v, 1);
    assert_eq!(v, vec![u64::MAX, 1]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim_overflow_carry() {
    // u64::MAX * 2 = 2^65 - 2 → [u64::MAX-1], carry=1
    let mut v = vec![u64::MAX];
    let c = mul_prim(&mut v, 2);
    assert_eq!(v, vec![u64::MAX - 1]);
    assert_eq!(c, 1);
}

#[test]
fn test_mul_prim_two_limbs() {
    let mut v = vec![1u64, 1];
    let c = mul_prim(&mut v, 2);
    assert_eq!(v, vec![2, 2]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim_max_times_max() {
    let mut v = vec![u64::MAX];
    let c = mul_prim(&mut v, u64::MAX);
    assert_eq!(v, vec![1]);
    assert_eq!(c, u64::MAX - 1);
}

#[test]
fn test_mul_prim_multi_limb_carry() {
    let mut v = vec![u64::MAX, u64::MAX];
    let c = mul_prim(&mut v, 2);
    assert_eq!(v, vec![u64::MAX - 1, u64::MAX]);
    assert_eq!(c, 1);
}

#[test]
fn test_mul_prim_correctness_single_limb() {
    for &(a, p) in &[(1u64, 5u64), (7, 11), (u64::MAX, 3), (0xDEAD_BEEF, 0xCAFE)] {
        let expected = (a as u128) * (p as u128);
        let mut v = vec![a];
        let c = mul_prim(&mut v, p);
        let got = v[0] as u128 | ((c as u128) << 64);
        assert_eq!(got, expected, "mul_prim({a:#x}, {p:#x}) mismatch");
    }
}

// ─── mul_prim2 ──────────────────────────────────────────────────────────────

#[test]
fn test_mul_prim2_by_one() {
    let mut v = vec![5u64];
    let c = mul_prim2(&mut v, 1);
    assert_eq!(v, vec![5]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim2_by_small() {
    let mut v = vec![3u64];
    let c = mul_prim2(&mut v, 5);
    assert_eq!(v, vec![15]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim2_overflow_into_high64() {
    let mut v = vec![1u64];
    let c = mul_prim2(&mut v, 1u128 << 64);
    assert_eq!(v, vec![0]);
    assert_eq!(c, 1);
}

#[test]
fn test_mul_prim2_two_limbs() {
    let mut v = vec![1u64, 0];
    let c = mul_prim2(&mut v, 5);
    assert_eq!(v, vec![5, 0]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim2_by_zero() {
    let mut v = vec![u64::MAX, u64::MAX];
    let c = mul_prim2(&mut v, 0);
    assert_eq!(v, vec![0, 0]);
    assert_eq!(c, 0);
}

// ─── mul_buf ────────────────────────────────────────────────────────────────

#[test]
fn test_mul_buf_empty_a() {
    let mut out = vec![0u64];
    let c = mul_buf(&[], &[3], &mut out);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_buf_basic() {
    let mut out = vec![0u64];
    let c = mul_buf(&[5], &[4], &mut out);
    assert_eq!(out, vec![20]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_buf_overflow() {
    let mut out = vec![0u64];
    let c = mul_buf(&[u64::MAX], &[u64::MAX], &mut out);
    assert_eq!(out, vec![1]);
    assert_eq!(c, u64::MAX - 1);
}

#[test]
fn test_mul_buf_multi_limb() {
    let mut out = vec![0u64; 3];
    let c = mul_buf(&[1, 1], &[1, 1], &mut out);
    assert_eq!(out, vec![1, 2, 1]);
    assert_eq!(c, 0);
}

// ─── mul_vec ────────────────────────────────────────────────────────────────

#[test]
fn test_mul_vec_basic() {
    let (v, c) = mul_vec(&[2], &[3]);
    assert_eq!(v, vec![6]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_vec_identity() {
    let a = vec![u64::MAX, 1, 42];
    let (v, c) = mul_vec(&a, &[1]);
    assert_eq!(v, a);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_vec_zero() {
    let a = vec![u64::MAX, 1, 42];
    let (mut v, c) = mul_vec(&a, &[0]);
    trim_lz(&mut v);
    assert!(v.is_empty() || v.iter().all(|&x| x == 0));
    assert_eq!(c, 0);
}

#[test]
fn test_mul_vec_max_times_max() {
    let (v, c) = mul_vec(&[u64::MAX], &[u64::MAX]);
    assert_eq!(v, vec![1]);
    assert_eq!(c, u64::MAX - 1);
}

#[test]
fn test_mul_vec_multi_limb() {
    let (v, c) = mul_vec(&[1, 1], &[2]);
    assert_eq!(v, vec![2, 2]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_vec_commutativity() {
    for seed in 0u64..20 {
        let a = rand_nonzero_vec(3, seed);
        let b = rand_nonzero_vec(4, seed + 100);
        assert_eq!(
            norm_mul(&a, &b),
            norm_mul(&b, &a),
            "commutativity failed seed={seed}"
        );
    }
}

// ─── is_school flat boundary (s ≤ half branch) ──────────────────────────────
//
// When s ≤ (l+1)/2, school requires s ≤ CHUNKING_KARATSUBA_CUTOFF.
// flat_branch_l() gives a long length where the cutoff is safely in the flat
// branch; the boundary is the cutoff itself.

#[test]
fn test_dispatch_school_flat_at_cutoff() {
    // s = CHUNKING_KARATSUBA_CUTOFF → last school size in flat branch
    let l = flat_branch_l();
    let s = CHUNKING_KARATSUBA_CUTOFF;
    assert!(is_school(l, s), "l={l}, s={s} should be school");
    let a = rand_nonzero_vec(l, 1001);
    let b = rand_nonzero_vec(s, 1002);
    assert_eq!(norm_mul(&a, &b), mul_ref(&a, &b));
}

#[test]
fn test_dispatch_school_flat_one_past_cutoff() {
    // s = CHUNKING_KARATSUBA_CUTOFF + 1 → first karatsuba chunking size
    let l = flat_branch_l();
    let s = CHUNKING_KARATSUBA_CUTOFF + 1;
    assert!(!is_school(l, s), "l={l}, s={s} should not be school");
    let a = rand_nonzero_vec(l, 1003);
    let b = rand_nonzero_vec(s, 1004);
    assert_eq!(norm_mul(&a, &b), mul_ref(&a, &b));
}

// ─── is_school curved boundary (s > half branch) ────────────────────────────
//
// curved_boundary() locates the peak of the parabolic boundary and returns
// (l, s_school, s_karatsuba).  The tests verify correctness on both sides.

#[test]
fn test_dispatch_school_curved_at_boundary() {
    let (l, s_school, _) = curved_boundary();
    assert!(
        is_school(l, s_school),
        "l={l}, s={s_school} should be school"
    );
    let a = rand_nonzero_vec(l, 2001);
    let b = rand_nonzero_vec(s_school, 2002);
    assert_eq!(norm_mul(&a, &b), mul_ref(&a, &b));
}

#[test]
fn test_dispatch_school_curved_one_past_boundary() {
    let (l, _, s_kara) = curved_boundary();
    assert!(
        !is_school(l, s_kara),
        "l={l}, s={s_kara} should not be school"
    );
    let a = rand_nonzero_vec(l, 2003);
    let b = rand_nonzero_vec(s_kara, 2004);
    assert_eq!(norm_mul(&a, &b), mul_ref(&a, &b));
}

// ─── balanced school → karatsuba transition ──────────────────────────────────

#[test]
fn test_dispatch_balanced_last_school() {
    let n = first_balanced_karatsuba();
    assert!(
        is_school(n - 1, n - 1),
        "size {} should still be school",
        n - 1
    );
    let a = rand_nonzero_vec(n - 1, 3001);
    assert_eq!(norm_mul(&a, &a), mul_ref(&a, &a));
}

#[test]
fn test_dispatch_balanced_first_karatsuba() {
    let n = first_balanced_karatsuba();
    assert!(!is_school(n, n), "size {n} should be karatsuba");
    let a = rand_nonzero_vec(n, 3002);
    assert_eq!(norm_mul(&a, &a), mul_ref(&a, &a));
}

// ─── karatsuba chunking vs recurse ───────────────────────────────────────────
//
// Within karatsuba, kdispatch picks Chunking when s < (l+1)/2 and Recurse
// otherwise (both post-school).  chunking_recurse_boundary() finds real sizes
// on each side for an l large enough to have both regions.

#[test]
fn test_karatsuba_chunking_path() {
    let l = 4 * first_balanced_karatsuba();
    let (s, _) = chunking_recurse_boundary(l);
    assert!(
        s < (l + 1) / 2,
        "s={s} must be in chunking region for l={l}"
    );
    for seed in 0u64..5 {
        let a = rand_nonzero_vec(l, seed + 4000);
        let b = rand_nonzero_vec(s, seed + 4100);
        assert_eq!(
            norm_mul(&a, &b),
            mul_ref(&a, &b),
            "chunking path seed={seed}"
        );
    }
}

#[test]
fn test_karatsuba_recurse_path() {
    let l = 4 * first_balanced_karatsuba();
    let (_, s) = chunking_recurse_boundary(l);
    assert!(
        s >= (l + 1) / 2,
        "s={s} must be in recurse region for l={l}"
    );
    for seed in 0u64..5 {
        let a = rand_nonzero_vec(l, seed + 5000);
        let b = rand_nonzero_vec(s, seed + 5100);
        assert_eq!(
            norm_mul(&a, &b),
            mul_ref(&a, &b),
            "recurse path seed={seed}"
        );
    }
}

#[test]
fn test_karatsuba_chunking_recurse_boundary() {
    // The two boundary sizes right next to (l+1)/2 should both be correct.
    let l = 4 * first_balanced_karatsuba();
    let (s_chunk, s_rec) = chunking_recurse_boundary(l);
    let a = rand_nonzero_vec(l, 6000);
    let b_c = rand_nonzero_vec(s_chunk, 6001);
    let b_r = rand_nonzero_vec(s_rec, 6001);
    assert_eq!(norm_mul(&a, &b_c), mul_ref(&a, &b_c), "chunking side");
    assert_eq!(norm_mul(&a, &b_r), mul_ref(&a, &b_r), "recurse side");
}

// ─── mul_arr ────────────────────────────────────────────────────────────────

#[test]
fn test_mul_arr_basic() {
    let (arr, c) = mul_arr::<2>(&[2], &[3]).expect("mul_arr should succeed");
    assert_eq!(arr[0], 6);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_arr_too_small_returns_err() {
    let r = mul_arr::<2>(&[u64::MAX, 1], &[u64::MAX, 1]);
    assert!(r.is_err());
}

#[test]
fn test_mul_arr_matches_mul_vec() {
    let a = rand_nonzero_vec(4, 42);
    let b = rand_nonzero_vec(4, 43);
    let expected = norm_mul(&a, &b);
    let (arr, c) = mul_arr::<16>(&a, &b).expect("mul_arr failed");
    // arr has N=16 slots; trim trailing zeros left by unused slots since norm
    // no longer calls trim_lz.
    let mut got = norm(arr.to_vec(), c);
    trim_lz(&mut got);
    assert_eq!(got, expected);
}

#[test]
fn test_mul_arr_karatsuba_path() {
    // Use first_balanced_karatsuba() to ensure this hits the karatsuba static path.
    let n = first_balanced_karatsuba();
    let a = rand_nonzero_vec(n, 7000);
    let expected = norm_mul(&a, &a);
    // N must fit 2n-1 output limbs; 512 is a generous upper bound for typical constants.
    const N: usize = 512;
    let (arr, c) = mul_arr::<N>(&a, &a).expect("mul_arr karatsuba failed");
    assert_eq!(norm(arr[..2 * n - 1].to_vec(), c), expected);
}

// ─── sqr_buf / sqr_vec ──────────────────────────────────────────────────────

#[test]
fn test_sqr_buf_single_zero() {
    let mut out = vec![0u64];
    let c = sqr_buf(&[0], &mut out);
    assert_eq!(out, vec![0]);
    assert_eq!(c, 0);
}

#[test]
fn test_sqr_buf_one() {
    let mut out = vec![0u64];
    let c = sqr_buf(&[1], &mut out);
    assert_eq!(out, vec![1]);
    assert_eq!(c, 0);
}

#[test]
fn test_sqr_buf_two() {
    let mut out = vec![0u64];
    let c = sqr_buf(&[2], &mut out);
    assert_eq!(out, vec![4]);
    assert_eq!(c, 0);
}

#[test]
fn test_sqr_vec_zero() {
    let (v, c) = sqr_vec(&[0]);
    assert_eq!(v, vec![0]);
    assert_eq!(c, 0);
}

#[test]
fn test_sqr_vec_one() {
    let (v, c) = sqr_vec(&[1]);
    assert_eq!(v, vec![1]);
    assert_eq!(c, 0);
}

#[test]
fn test_sqr_vec_two() {
    let (v, c) = sqr_vec(&[2]);
    assert_eq!(v, vec![4]);
    assert_eq!(c, 0);
}

#[test]
fn test_sqr_vec_max_u64() {
    let (v, c) = sqr_vec(&[u64::MAX]);
    assert_eq!(v, vec![1]);
    assert_eq!(c, u64::MAX - 1);
}

#[test]
fn test_sqr_equals_mul_small() {
    // val=0 is excluded: sqr_vec returns [0] while mul_ref trims to [] — a
    // normalization-only difference now that norm no longer calls trim_lz.
    for &val in &[1u64, 2, 100, u64::MAX / 2, u64::MAX] {
        assert_eq!(norm_sqr(&[val]), mul_ref(&[val], &[val]), "val={val:#x}");
    }
}

#[test]
fn test_sqr_equals_mul_random() {
    for seed in 0u64..30 {
        let len = (seed % 8 + 1) as usize;
        let v = rand_nonzero_vec(len, seed);
        assert_eq!(norm_sqr(&v), mul_ref(&v, &v), "seed={seed}");
    }
}

// ─── sqr dispatch boundaries ─────────────────────────────────────────────────
//
// dyn_sqr_dispatch:
//   n ≤ KARATSUBA_SQR_CUTOFF  →  School
//   n ≤ FFT_SQR_CUTOFF        →  Karatsuba
//   n  > FFT_SQR_CUTOFF       →  FFT
//
// All three paths must satisfy sqr_vec(x) == mul_vec(x, x).

#[test]
fn test_sqr_school_karatsuba_boundary() {
    for len in [
        KARATSUBA_SQR_CUTOFF - 1,
        KARATSUBA_SQR_CUTOFF,
        KARATSUBA_SQR_CUTOFF + 1,
        KARATSUBA_SQR_CUTOFF + 2,
    ] {
        let v = rand_nonzero_vec(len, len as u64 + 500);
        assert_eq!(
            norm_sqr(&v),
            mul_ref(&v, &v),
            "sqr school/karatsuba boundary len={len}"
        );
    }
}

#[test]
fn test_sqr_karatsuba_fft_boundary() {
    for len in [FFT_SQR_CUTOFF - 1, FFT_SQR_CUTOFF, FFT_SQR_CUTOFF + 1] {
        let v = rand_nonzero_vec(len, len as u64 + 9000);
        assert_eq!(
            norm_sqr(&v),
            mul_ref(&v, &v),
            "sqr karatsuba/fft boundary len={len}"
        );
    }
}

// ─── sqr_arr ────────────────────────────────────────────────────────────────

#[test]
fn test_sqr_arr_basic() {
    let (arr, c) = sqr_arr::<4>(&[3]).expect("sqr_arr failed");
    assert_eq!(arr[0], 9);
    assert_eq!(c, 0);
}

#[test]
fn test_sqr_arr_too_small_returns_err() {
    let r = sqr_arr::<4>(&[u64::MAX, u64::MAX, u64::MAX]);
    assert!(r.is_err());
}

#[test]
fn test_sqr_arr_matches_sqr_vec() {
    let v = rand_nonzero_vec(4, 77);
    let expected = norm_sqr(&v);
    let (arr, ac) = sqr_arr::<16>(&v).expect("sqr_arr failed");
    assert_eq!(ac, 0);
    assert_eq!(norm(arr[..2 * v.len()].to_vec(), 0), expected);
}

// ─── powi_vec ───────────────────────────────────────────────────────────────

#[test]
fn test_powi_vec_pow_zero() {
    for &base in &[1u64, 2, 7, u64::MAX] {
        assert_eq!(powi_vec(&[base], 0), vec![1], "base={base}");
    }
}

#[test]
fn test_powi_vec_pow_one() {
    assert_eq!(powi_vec(&[5], 1), vec![5]);
}

#[test]
fn test_powi_vec_pow_two() {
    assert_eq!(powi_vec(&[3], 2), vec![9]);
}

#[test]
fn test_powi_vec_pow_ten() {
    assert_eq!(powi_vec(&[2], 10), vec![1024]);
}

#[test]
fn test_powi_vec_pow_64_gives_2_to_64() {
    let mut r = powi_vec(&[2], 64);
    trim_lz(&mut r);
    assert_eq!(r, vec![0, 1]);
}

#[test]
fn test_powi_vec_cube() {
    assert_eq!(powi_vec(&[3], 3), vec![27]);
}

#[test]
fn test_powi_vec_power_of_two_exponents() {
    assert_eq!(powi_vec(&[5], 2), vec![25]);
    assert_eq!(powi_vec(&[5], 4), vec![625]);
    assert_eq!(powi_vec(&[5], 8), vec![390625]);
}

#[test]
fn test_powi_vec_recursive_property() {
    for &(base, max_pow) in &[(2u64, 20usize), (3, 10), (7, 8)] {
        let mut prev = powi_vec(&[base], 1);
        for pow in 2..=max_pow {
            let curr = powi_vec(&[base], pow);
            let by_mul = norm_mul(&prev, &[base]);
            let mut curr2 = curr.clone();
            trim_lz(&mut curr2);
            assert_eq!(curr2, by_mul, "base={base}, pow={pow}");
            prev = curr;
        }
    }
}

// ─── powi_arr ───────────────────────────────────────────────────────────────

#[test]
fn test_powi_arr_basic() {
    let r = powi_arr::<4>(&[2], 3).expect("powi_arr failed");
    assert_eq!(r[0], 8);
}

#[test]
fn test_powi_arr_overflow_returns_err() {
    let r = powi_arr::<2>(&[2], 128);
    assert!(r.is_err());
}

#[test]
fn test_powi_arr_matches_powi_vec() {
    let mut expected = powi_vec(&[3u64], 5);
    trim_lz(&mut expected);
    let arr = powi_arr::<8>(&[3u64], 5).expect("powi_arr failed");
    let mut got: Vec<u64> = arr.into_iter().collect();
    trim_lz(&mut got);
    assert_eq!(got, expected);
}

// ─── Karatsuba / FFT boundary helpers ────────────────────────────────────────

/// First balanced n where is_karatsuba(n, n) is false (first FFT region input).
fn first_fft_balanced() -> usize {
    (3..).find(|&n| !is_karatsuba(n, n)).unwrap()
}

/// For a given l, walk the s > half region and return (s_karatsuba, s_fft)
/// straddling the Karatsuba/FFT boundary. Both are non-school.
/// Returns None when no such pair exists for this l.
fn karatsuba_fft_recurse_boundary(l: usize) -> Option<(usize, usize)> {
    let half = (l + 1) / 2;
    let mut last_kara: Option<usize> = None;
    for s in half..=l {
        if is_school(l, s) {
            continue;
        }
        if is_karatsuba(l, s) {
            last_kara = Some(s);
        } else if let Some(sk) = last_kara {
            return Some((sk, s));
        }
    }
    None
}

/// Search for (l, s_karatsuba, s_fft) satisfying:
///   s <= (l+1)/2, !is_school, and the Karatsuba/FFT boundary is crossed.
/// Returns None if no such triple exists in the search range.
fn find_chunking_fft_boundary() -> Option<(usize, usize, usize)> {
    for l in (50..3000).step_by(50) {
        let half = (l + 1) / 2;
        let mut last_kara: Option<usize> = None;
        for s in (CHUNKING_KARATSUBA_CUTOFF + 1)..half {
            if is_school(l, s) {
                continue;
            }
            if is_karatsuba(l, s) {
                last_kara = Some(s);
            } else if let Some(sk) = last_kara {
                return Some((l, sk, s));
            }
        }
    }
    None
}

// ─── Karatsuba / FFT balanced boundary ───────────────────────────────────────

#[test]
fn test_dispatch_karatsuba_fft_balanced_last_karatsuba() {
    let n = first_fft_balanced();
    assert!(
        is_karatsuba(n - 1, n - 1),
        "size {} should still be karatsuba",
        n - 1
    );
    for seed in 0u64..5 {
        let a = rand_nonzero_vec(n - 1, seed + 20000);
        let b = rand_nonzero_vec(n - 1, seed + 20100);
        assert_eq!(
            norm_mul(&a, &b),
            mul_ref(&a, &b),
            "last balanced karatsuba seed={seed}"
        );
    }
}

#[test]
fn test_dispatch_karatsuba_fft_balanced_first_fft() {
    let n = first_fft_balanced();
    assert!(!is_karatsuba(n, n), "size {n} should be FFT");
    for seed in 0u64..5 {
        let a = rand_nonzero_vec(n, seed + 21000);
        let b = rand_nonzero_vec(n, seed + 21100);
        assert_eq!(
            norm_mul(&a, &b),
            mul_ref(&a, &b),
            "first balanced FFT seed={seed}"
        );
    }
}

// ─── Karatsuba / FFT unbalanced boundary (s > half, recurse sub-case) ────────

#[test]
fn test_dispatch_karatsuba_fft_recurse_boundary() {
    // Use first_fft_balanced as l: the balanced FFT threshold is exactly where
    // the recurse-sub-case boundary sits for most constant configurations.
    let n = first_fft_balanced();
    if let Some((s_kara, s_fft)) = karatsuba_fft_recurse_boundary(n) {
        assert!(
            is_karatsuba(n, s_kara),
            "l={n}, s={s_kara} should be karatsuba"
        );
        assert!(!is_karatsuba(n, s_fft), "l={n}, s={s_fft} should be FFT");
        let a = rand_nonzero_vec(n, 22000);
        let b_k = rand_nonzero_vec(s_kara, 22001);
        let b_f = rand_nonzero_vec(s_fft, 22002);
        assert_eq!(
            norm_mul(&a, &b_k),
            mul_ref(&a, &b_k),
            "recurse karatsuba side"
        );
        assert_eq!(norm_mul(&a, &b_f), mul_ref(&a, &b_f), "recurse FFT side");
    }
}

// ─── Karatsuba / FFT unbalanced boundary (s <= half, chunking sub-case) ──────

#[test]
fn test_dispatch_karatsuba_fft_chunking_boundary() {
    if let Some((l, s_kara, s_fft)) = find_chunking_fft_boundary() {
        let half = (l + 1) / 2;
        assert!(s_kara < half && s_fft < half);
        assert!(
            !is_school(l, s_kara) && is_karatsuba(l, s_kara),
            "l={l}, s={s_kara} should be chunking karatsuba"
        );
        assert!(
            !is_school(l, s_fft) && !is_karatsuba(l, s_fft),
            "l={l}, s={s_fft} should be chunking FFT"
        );
        for seed in 0u64..5 {
            let a = rand_nonzero_vec(l, seed + 23000);
            let b_k = rand_nonzero_vec(s_kara, seed + 23100);
            let b_f = rand_nonzero_vec(s_fft, seed + 23200);
            assert_eq!(
                norm_mul(&a, &b_k),
                mul_ref(&a, &b_k),
                "chunking karatsuba side seed={seed}"
            );
            assert_eq!(
                norm_mul(&a, &b_f),
                mul_ref(&a, &b_f),
                "chunking FFT side seed={seed}"
            );
        }
    }
}

// ─── FFT mul correctness ──────────────────────────────────────────────────────

#[test]
fn test_fft_mul_random_correctness() {
    let n = first_fft_balanced();
    for seed in 0u64..10 {
        let l = n + seed as usize % 30;
        let s = n + seed as usize % 20;
        let a = rand_nonzero_vec(l, seed + 24000);
        let b = rand_nonzero_vec(s, seed + 24100);
        assert_eq!(norm_mul(&a, &b), mul_ref(&a, &b), "FFT mul seed={seed}");
    }
}

#[test]
fn test_fft_mul_all_max_limbs() {
    // All-MAX inputs maximize coefficient magnitude; verifies that optimal_b()
    // is conservative enough to avoid f64 mantissa overflow in accumulate().
    let n = first_fft_balanced();
    let a = vec![u64::MAX; n];
    let b = vec![u64::MAX; n];
    assert_eq!(norm_mul(&a, &b), mul_ref(&a, &b), "FFT mul all-MAX");
}

#[test]
fn test_fft_mul_sparse_high_limb() {
    // Last limb = 1 (minimal significant bits in the top word).
    // Tests that bit_chunk_length() uses actual bit width, not just limb count.
    let n = first_fft_balanced();
    let mut a = vec![u64::MAX; n];
    *a.last_mut().unwrap() = 1;
    let b = rand_nonzero_vec(n, 25000);
    assert_eq!(
        norm_mul(&a, &b),
        mul_ref(&a, &b),
        "FFT mul sparse high limb"
    );
}

#[test]
fn test_fft_mul_highly_unbalanced() {
    // l >> s, both well past the school threshold.
    // Exercises the chunking-sub-case cost branch of is_karatsuba at large scale.
    let n = first_fft_balanced();
    let l = n * 4;
    for seed in 0u64..5 {
        let a = rand_nonzero_vec(l, seed + 26000);
        let b = rand_nonzero_vec(n, seed + 26100);
        assert_eq!(
            norm_mul(&a, &b),
            mul_ref(&a, &b),
            "FFT highly unbalanced seed={seed}"
        );
    }
}

// ─── FFT sqr correctness ──────────────────────────────────────────────────────

#[test]
fn test_fft_sqr_random_correctness() {
    for &(len, seed_base) in &[
        (FFT_SQR_CUTOFF + 10, 27000u64),
        (FFT_SQR_CUTOFF + 50, 27100),
        (FFT_SQR_CUTOFF + 100, 27200),
    ] {
        for seed in 0u64..5 {
            let v = rand_nonzero_vec(len, seed + seed_base);
            assert_eq!(
                norm_sqr(&v),
                mul_ref(&v, &v),
                "FFT sqr len={len} seed={seed}"
            );
        }
    }
}

#[test]
fn test_fft_sqr_all_max_limbs() {
    for &len in &[
        FFT_SQR_CUTOFF + 1,
        FFT_SQR_CUTOFF + 50,
        FFT_SQR_CUTOFF + 100,
    ] {
        let v = vec![u64::MAX; len];
        assert_eq!(norm_sqr(&v), mul_ref(&v, &v), "FFT sqr all-MAX len={len}");
    }
}

#[test]
fn test_fft_sqr_sparse_high_limb() {
    // Last limb = 1; tests optimal_b_sqr() accuracy with sparse top word.
    for &len in &[FFT_SQR_CUTOFF + 1, FFT_SQR_CUTOFF + 50] {
        let mut v = vec![u64::MAX; len];
        *v.last_mut().unwrap() = 1;
        assert_eq!(
            norm_sqr(&v),
            mul_ref(&v, &v),
            "FFT sqr sparse high limb len={len}"
        );
    }
}

// ─── FFT_BIT_BUFFER precision robustness ──────────────────────────────────────
//
// These tests probe the mantissa budget enforced by FFT_BIT_BUFFER.
// optimal_b() picks the largest b satisfying 2b + log_n + log_log_n + BIT_BUFFER ≤ 54.
// Larger b → larger coefficient magnitudes → less f64 headroom.
//
// Three axes are covered for both mul and sqr:
//  1. Extended random seeds at the smallest FFT inputs (where b is largest).
//  2. Dense (all-MAX) inputs across a contiguous size range — exercises every
//     distinct (b, n) pair that optimal_b can select.
//  3. Sparse (last limb = 1) inputs across the same range — bit_chunk_length
//     is slightly smaller, so optimal_b picks an even larger b, sitting as
//     close as possible to the BIT_BUFFER ceiling.

// ── mul: extended random seeds at the balanced FFT boundary ──────────────────

#[test]
fn test_fft_mul_precision_random_extended() {
    // The original bug manifested at first_fft_balanced() with a specific
    // seed. 50 seeds give broad coverage of random bit patterns at the
    // tightest-budget size.
    let n = first_fft_balanced();
    for seed in 0u64..50 {
        let a = rand_nonzero_vec(n, seed + 35000);
        let b = rand_nonzero_vec(n, seed + 35100);
        assert_eq!(norm_mul(&a, &b), mul_ref(&a, &b),
            "FFT mul precision random seed={seed}");
    }
}

// ── mul: dense sweep — every size from n to n+50 ─────────────────────────────

#[test]
fn test_fft_mul_precision_dense_sweep() {
    // All-MAX limbs maximise the value of every b-bit chunk (each chunk = 2^b - 1).
    // Testing every size in the range exercises each (b, n) pair that
    // optimal_b can return, confirming BIT_BUFFER is sufficient at each level.
    let n = first_fft_balanced();
    for len in n..=(n + 50) {
        let a = vec![u64::MAX; len];
        let b = vec![u64::MAX; len];
        assert_eq!(norm_mul(&a, &b), mul_ref(&a, &b),
            "FFT mul precision dense len={len}");
    }
}

// ── mul: sparse sweep — last limb = 1, every size from n to n+50 ─────────────

#[test]
fn test_fft_mul_precision_sparse_sweep() {
    // last = 1 → bit_chunk_length is (64(n-1)+1)/b instead of 64n/b,
    // so optimal_b selects a larger b than it would for all-MAX inputs of
    // the same limb count. This puts the computation closest to the
    // BIT_BUFFER ceiling and is therefore the hardest precision case.
    let n = first_fft_balanced();
    for len in n..=(n + 50) {
        let mut a = vec![u64::MAX; len];
        let mut b = vec![u64::MAX; len];
        *a.last_mut().unwrap() = 1;
        *b.last_mut().unwrap() = 1;
        assert_eq!(norm_mul(&a, &b), mul_ref(&a, &b),
            "FFT mul precision sparse len={len}");
    }
}

// ── sqr: extended random seeds just past the sqr FFT boundary ────────────────

#[test]
fn test_fft_sqr_precision_random_extended() {
    for seed in 0u64..50 {
        let len = FFT_SQR_CUTOFF + 1 + seed as usize % 30;
        let v = rand_nonzero_vec(len, seed + 36000);
        assert_eq!(norm_sqr(&v), mul_ref(&v, &v),
            "FFT sqr precision random seed={seed} len={len}");
    }
}

// ── sqr: dense sweep ──────────────────────────────────────────────────────────

#[test]
fn test_fft_sqr_precision_dense_sweep() {
    for len in (FFT_SQR_CUTOFF + 1)..=(FFT_SQR_CUTOFF + 50) {
        let v = vec![u64::MAX; len];
        assert_eq!(norm_sqr(&v), mul_ref(&v, &v),
            "FFT sqr precision dense len={len}");
    }
}

// ── sqr: sparse sweep — last limb = 1, every size in range ───────────────────

#[test]
fn test_fft_sqr_precision_sparse_sweep() {
    for len in (FFT_SQR_CUTOFF + 1)..=(FFT_SQR_CUTOFF + 50) {
        let mut v = vec![u64::MAX; len];
        *v.last_mut().unwrap() = 1;
        assert_eq!(norm_sqr(&v), mul_ref(&v, &v),
            "FFT sqr precision sparse len={len}");
    }
}

// ─── empty-buffer early-return paths ─────────────────────────────────────────

#[test]
fn test_mul_prim_empty_buf() {
    let c = mul_prim(&mut [], 99);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim2_empty_buf() {
    let c = mul_prim2(&mut [], 99u128);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_dyn_empty_inputs() {
    // The debug_assert fires before the empty check, so out must satisfy
    // out.len() >= a.len() + b.len() - 1 even for the empty-input case.
    // a empty: 0 + 3 - 1 = 2 minimum slots required by the assert.
    let mut out = vec![0u64; 2];
    let c = mul_dyn(&[], &[1, 2, 3], &mut out);
    assert_eq!(c, 0);
    // b empty: 2 + 0 - 1 = 1 minimum slot.
    let mut out2 = vec![0u64; 1];
    let c2 = mul_dyn(&[1, 2], &[], &mut out2);
    assert_eq!(c2, 0);
}

#[test]
fn test_sqr_buf_empty() {
    let mut out: Vec<u64> = vec![];
    let c = sqr_buf(&[], &mut out);
    assert_eq!(c, 0);
}

// ─── mul_static / mul_arr Prim2 path (s == 2) ────────────────────────────────

#[test]
fn test_mul_arr_prim2_path() {
    // short = 2 limbs → static_dispatch → Prim2.
    // mul_prim2 is called on the full N-element out buffer, so it writes the
    // 7th limb into arr[a.len()+b.len()-1] rather than returning it as carry.
    // We therefore read a.len()+b.len() slots from arr (not a.len()+b.len()-1)
    // and expect c == 0.
    let a = rand_nonzero_vec(5, 40001);
    let b = [0xDEAD_BEEF_u64, 0xCAFE_BABE_u64];
    let expected = norm_mul(&a, &b);
    let (arr, c) = mul_arr::<16>(&a, &b).expect("mul_arr prim2 failed");
    assert_eq!(c, 0, "carry should be 0: extra limb lives inside arr");
    let mut got = arr[..a.len() + b.len()].to_vec();
    trim_lz(&mut got);
    let mut exp = expected.clone();
    trim_lz(&mut exp);
    assert_eq!(got, exp);
}

// ─── sqr_arr / sqr_static Karatsuba path ─────────────────────────────────────
// BUG: karatsuba_sqr_core receives `out` un-sliced from sqr_static/sqr_arr,
// so `z2 = out[2*half..]` is the full tail of the N-element array instead of
// the correct `2*(n-half)-1` elements.  sub_buf(cross, z2) then fires the
// `lhs must be longer than rhs` debug_assert.  See production-bug report.
#[test]
#[ignore = "production bug: karatsuba_sqr_entry_static passes un-sliced out to karatsuba_sqr_core"]
fn test_sqr_arr_karatsuba_static_path() {
    // buf.len() > KARATSUBA_SQR_CUTOFF → static_sqr_dispatch → Karatsubaa
    let n = KARATSUBA_SQR_CUTOFF + 2;
    let v = rand_nonzero_vec(n, 50000);
    let expected = norm_sqr(&v);
    // N must fit 2n-1 output limbs
    const N: usize = 512;
    let (arr, c) = sqr_arr::<N>(&v).expect("sqr_arr karatsuba failed");
    let got = norm(arr[..2 * n - 1].to_vec(), c);
    assert_eq!(got, expected);
}

// ─── karatsuba_entry_static else branch (scratch_sz >= N) ─────────────────────
// BUG: in the else branch, `cross = [0; N]` is N=100 elements but
// karatsuba_core does `add_buf(&mut out[half..], &cross)` where out[half..] is
// only 48 elements.  The cross slice must be trimmed to 2*half+1 before the
// call.  See production-bug report.
#[test]
#[ignore = "production bug: karatsuba_entry_static else branch passes oversized cross to karatsuba_core"]
fn test_karatsuba_entry_static_large_scratch_path() {
    let l = 49;
    let s = 25;
    let long_v = rand_nonzero_vec(l, 60000);
    let short_v = rand_nonzero_vec(s, 60001);
    let expected = norm_mul(&long_v, &short_v);
    const N: usize = 100;
    let mut out = [0u64; N];
    let c = karatsuba_entry_static::<N>(&long_v, &short_v, &mut out);
    let got = norm(out[..l + s - 1].to_vec(), c);
    assert_eq!(got, expected, "karatsuba_entry_static else branch");
}

// ─── karatsuba_sqr_entry_static else branch (scratch_sz > N) ─────────────────
// BUG: same family as the mul case — cross = [0; N] is passed un-sliced to
// karatsuba_sqr_core, which does sub_buf(cross, z2) where z2 is larger.
// See production-bug report.
#[test]
#[ignore = "production bug: karatsuba_sqr_entry_static else branch passes oversized cross/out"]
fn test_karatsuba_sqr_entry_static_large_scratch_path() {
    let n = 80;
    let v = rand_nonzero_vec(n, 70000);
    let expected = norm_sqr(&v);
    const N: usize = 100;
    let mut out = vec![0u64; 2 * n - 1];
    let c = karatsuba_sqr_entry_static::<N>(&v, &mut out);
    let got = norm(out, c);
    assert_eq!(got, expected, "karatsuba_sqr_entry_static else branch");
}

// ─── optimal_b with last limb == 0 ───────────────────────────────────────────
//
// bit_chunk_length has an early-return branch for last == 0 (lines 615-616).

#[test]
fn test_optimal_b_last_limb_zero() {
    // last_l = 0 triggers the `if last == 0` branch of bit_chunk_length.
    // Any result is acceptable; we just verify it doesn't panic and returns Ok.
    let result = optimal_b(4, 0, 3, 1);
    assert!(result.is_ok(), "optimal_b should succeed with last_l=0");
}

// ─── powi_vec mul-carry path ──────────────────────────────────────────────────
//
// powi_vec([u64::MAX], 3) exercises the path where mul_dyn returns a
// non-zero carry during the multiply step (lines 1255-1256).

#[test]
fn test_powi_vec_mul_carry() {
    // u64::MAX^3 = (2^64-1)^3, which is large enough that the mul step
    // inside powi_vec overflows the pre-allocated mul_len window.
    let base = vec![u64::MAX];
    let result = powi_vec(&base, 3);
    // Cross-check: result should equal norm_mul(&norm_mul(&base, &base), &base)
    let sq = norm_mul(&base, &base);
    let expected = norm_mul(&sq, &base);
    assert_eq!(result, expected, "powi_vec mul-carry path mismatch");
}

// ─── powi_arr carry paths ─────────────────────────────────────────────────────

#[test]
fn test_powi_arr_mul_carry() {
    // Same as powi_vec_mul_carry but through powi_arr.
    const N: usize = 8;
    let base = vec![u64::MAX];
    let result = powi_arr::<N>(&base, 3).expect("powi_arr failed");
    let mut got: Vec<u64> = result.to_vec();
    trim_lz(&mut got);
    let sq = norm_mul(&base, &base);
    let mut expected = norm_mul(&sq, &base);
    trim_lz(&mut expected);
    assert_eq!(got, expected);
}
