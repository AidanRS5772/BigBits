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

