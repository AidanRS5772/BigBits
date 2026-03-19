use super::{rand_nonzero_vec, rand_vec};
use crate::utils::mul::*;
use crate::utils::utils::{buf_len, trim_lz};

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
    // u64::MAX * 2 = 2^65 - 2 = [u64::MAX-1], carry=1
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
    // u64::MAX * u64::MAX = 2^128 - 2^65 + 1 → low=1, carry=u64::MAX-1
    let mut v = vec![u64::MAX];
    let c = mul_prim(&mut v, u64::MAX);
    assert_eq!(v, vec![1]);
    assert_eq!(c, u64::MAX - 1);
}

#[test]
fn test_mul_prim_multi_limb_carry() {
    // [u64::MAX, u64::MAX] * 2:
    //   limb0: u64::MAX*2 = [u64::MAX-1], carry=1
    //   limb1: u64::MAX*2+1 = 2^65-1 = [u64::MAX], carry=1
    let mut v = vec![u64::MAX, u64::MAX];
    let c = mul_prim(&mut v, 2);
    assert_eq!(v, vec![u64::MAX - 1, u64::MAX]);
    assert_eq!(c, 1);
}

/// Verify mul_prim reconstructed result matches original * prim (for 1-limb inputs).
#[test]
fn test_mul_prim_correctness_single_limb() {
    for &(a, p) in &[
        (1u64, 5u64),
        (7, 11),
        (u64::MAX, 3),
        (0xDEAD_BEEF, 0xCAFE),
    ] {
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
    // [1] * 2^64 = 0 in low limb, carry = 1
    let mut v = vec![1u64];
    let c = mul_prim2(&mut v, 1u128 << 64);
    assert_eq!(v, vec![0]);
    assert_eq!(c, 1);
}

#[test]
fn test_mul_prim2_two_limbs() {
    // [1, 0] * 5 = [5, 0], carry = 0
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
    let mut out = vec![0u64]; // 1+1-1 = 1 limb
    let c = mul_buf(&[5], &[4], &mut out);
    assert_eq!(out, vec![20]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_buf_overflow() {
    let mut out = vec![0u64]; // result overflows into carry
    let c = mul_buf(&[u64::MAX], &[u64::MAX], &mut out);
    assert_eq!(out, vec![1]); // low limb
    assert_eq!(c, u64::MAX - 1); // high limb
}

#[test]
fn test_mul_buf_multi_limb() {
    // [1, 1] * [1, 1] = 1 + 2^64 + 2^64 + 2^128 = [1, 2, 1]
    // out must be 2+2-1 = 3 limbs
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
    // u64::MAX * u64::MAX = [1, u64::MAX-1], carry=0 (fits in 2 limbs via out)
    let (v, c) = mul_vec(&[u64::MAX], &[u64::MAX]);
    assert_eq!(v, vec![1]);
    assert_eq!(c, u64::MAX - 1);
}

#[test]
fn test_mul_vec_multi_limb() {
    // [1, 1] * [2] = [2, 2]
    let (v, c) = mul_vec(&[1, 1], &[2]);
    assert_eq!(v, vec![2, 2]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_vec_commutativity() {
    // mul_vec(a, b) == mul_vec(b, a) for random inputs.
    for seed in 0u64..20 {
        let a = rand_nonzero_vec(3, seed);
        let b = rand_nonzero_vec(4, seed + 100);
        let (mut ab, c_ab) = mul_vec(&a, &b);
        let (mut ba, c_ba) = mul_vec(&b, &a);
        if c_ab > 0 {
            ab.push(c_ab);
        }
        if c_ba > 0 {
            ba.push(c_ba);
        }
        trim_lz(&mut ab);
        trim_lz(&mut ba);
        assert_eq!(ab, ba, "commutativity failed for seed {seed}");
    }
}

/// At the Karatsuba boundary: mul_vec and mul_buf should agree.
#[test]
fn test_mul_vec_karatsuba_boundary() {
    for len in [30usize, 31, 32, 33, 34] {
        let a = rand_nonzero_vec(len, len as u64);
        let b = rand_nonzero_vec(len, len as u64 + 1000);
        let (mut vec_result, c_vec) = mul_vec(&a, &b);
        if c_vec > 0 {
            vec_result.push(c_vec);
        }

        let mut buf_result = vec![0u64; a.len() + b.len() - 1];
        let c_buf = mul_buf(&a, &b, &mut buf_result);
        if c_buf > 0 {
            buf_result.push(c_buf);
        }

        trim_lz(&mut vec_result);
        trim_lz(&mut buf_result);
        assert_eq!(vec_result, buf_result, "Karatsuba boundary failed at len={len}");
    }
}

// ─── mul_arr ────────────────────────────────────────────────────────────────

#[test]
fn test_mul_arr_basic() {
    let r = mul_arr::<2>(&[2], &[3]);
    let (arr, c) = r.expect("mul_arr should succeed");
    assert_eq!(arr[0], 6);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_arr_too_small_returns_err() {
    // [u64::MAX, 1] * [u64::MAX, 1] needs at least 3 limbs; N=2 → Err
    let r = mul_arr::<2>(&[u64::MAX, 1], &[u64::MAX, 1]);
    assert!(r.is_err());
}

#[test]
fn test_mul_arr_matches_mul_vec() {
    let a = rand_nonzero_vec(4, 42);
    let b = rand_nonzero_vec(4, 43);
    let (mut expected, c) = mul_vec(&a, &b);
    if c > 0 {
        expected.push(c);
    }
    trim_lz(&mut expected);

    let (arr, c_arr) = mul_arr::<16>(&a, &b).expect("mul_arr failed");
    let mut got: Vec<u64> = arr.to_vec();
    if c_arr > 0 {
        got.push(c_arr);
    }
    trim_lz(&mut got);
    assert_eq!(got, expected);
}

// ─── sqr_buf / sqr_vec ──────────────────────────────────────────────────────

#[test]
fn test_sqr_buf_single_zero() {
    let mut out = vec![0u64]; // 2*1-1=1
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
    // (2^64-1)^2 = 2^128 - 2^65 + 1 → low limb = 1, carry = u64::MAX - 1
    let (v, c) = sqr_vec(&[u64::MAX]);
    assert_eq!(v, vec![1]);
    assert_eq!(c, u64::MAX - 1);
}

/// sqr_vec(x) must equal mul_vec(x, x) for all inputs.
#[test]
fn test_sqr_equals_mul_small() {
    for &val in &[0u64, 1, 2, 100, u64::MAX / 2, u64::MAX] {
        let (mut sv, sc) = sqr_vec(&[val]);
        let (mut mv, mc) = mul_vec(&[val], &[val]);
        if sc > 0 {
            sv.push(sc);
        }
        if mc > 0 {
            mv.push(mc);
        }
        trim_lz(&mut sv);
        trim_lz(&mut mv);
        assert_eq!(sv, mv, "sqr != mul for val={val:#x}");
    }
}

#[test]
fn test_sqr_equals_mul_random() {
    for seed in 0u64..30 {
        let len = (seed % 8 + 1) as usize;
        let v = rand_nonzero_vec(len, seed);
        let (mut sv, sc) = sqr_vec(&v);
        let (mut mv, mc) = mul_vec(&v, &v);
        if sc > 0 {
            sv.push(sc);
        }
        if mc > 0 {
            mv.push(mc);
        }
        trim_lz(&mut sv);
        trim_lz(&mut mv);
        assert_eq!(sv, mv, "sqr != mul for seed={seed}");
    }
}

/// At the Karatsuba squaring boundary (32 limbs).
#[test]
fn test_sqr_vec_karatsuba_boundary() {
    for len in [31usize, 32, 33, 34] {
        let v = rand_nonzero_vec(len, len as u64 + 500);
        let (mut sv, sc) = sqr_vec(&v);
        let (mut mv, mc) = mul_vec(&v, &v);
        if sc > 0 {
            sv.push(sc);
        }
        if mc > 0 {
            mv.push(mc);
        }
        trim_lz(&mut sv);
        trim_lz(&mut mv);
        assert_eq!(sv, mv, "Karatsuba sqr boundary failed at len={len}");
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
    // sqr of a 3-limb number needs 5 limbs; N=4 → Err
    let r = sqr_arr::<4>(&[u64::MAX, u64::MAX, u64::MAX]);
    assert!(r.is_err());
}

#[test]
fn test_sqr_arr_matches_sqr_vec() {
    let v = rand_nonzero_vec(4, 77);
    let (mut expected, ec) = sqr_vec(&v);
    if ec > 0 {
        expected.push(ec);
    }
    trim_lz(&mut expected);

    let (arr, ac) = sqr_arr::<16>(&v).expect("sqr_arr failed");
    let mut got: Vec<u64> = arr[..2 * v.len() - 1].to_vec();
    if ac > 0 {
        got.push(ac);
    }
    trim_lz(&mut got);
    assert_eq!(got, expected);
}

// ─── powi_vec ───────────────────────────────────────────────────────────────

#[test]
fn test_powi_vec_pow_zero() {
    for &base in &[1u64, 2, 7, u64::MAX] {
        let r = powi_vec(&[base], 0);
        assert_eq!(r, vec![1], "base={base}");
    }
}

#[test]
fn test_powi_vec_pow_one() {
    let r = powi_vec(&[5], 1);
    assert_eq!(r, vec![5]);
}

#[test]
fn test_powi_vec_pow_two() {
    let r = powi_vec(&[3], 2);
    assert_eq!(r, vec![9]);
}

#[test]
fn test_powi_vec_pow_ten() {
    let r = powi_vec(&[2], 10);
    assert_eq!(r, vec![1024]);
}

#[test]
fn test_powi_vec_pow_64_gives_2_to_64() {
    // 2^64 = [0, 1]
    let mut r = powi_vec(&[2], 64);
    trim_lz(&mut r);
    assert_eq!(r, vec![0, 1]);
}

#[test]
fn test_powi_vec_cube() {
    let r = powi_vec(&[3], 3);
    assert_eq!(r, vec![27]);
}

#[test]
fn test_powi_vec_power_of_two_exponents() {
    // 5^2 = 25, 5^4 = 625, 5^8 = 390625
    assert_eq!(powi_vec(&[5], 2), vec![25]);
    assert_eq!(powi_vec(&[5], 4), vec![625]);
    assert_eq!(powi_vec(&[5], 8), vec![390625]);
}

/// Recursive check: powi_vec(v, n) == mul_vec(powi_vec(v, n-1), v)
#[test]
fn test_powi_vec_recursive_property() {
    for &(base, max_pow) in &[(2u64, 20usize), (3, 10), (7, 8)] {
        let mut prev = powi_vec(&[base], 1);
        for pow in 2..=max_pow {
            let curr = powi_vec(&[base], pow);
            let (mut by_mul, c) = mul_vec(&prev, &[base]);
            if c > 0 {
                by_mul.push(c);
            }
            trim_lz(&mut by_mul);
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
    // 2^128 requires 3 limbs; N=2 → Err
    let r = powi_arr::<2>(&[2], 128);
    assert!(r.is_err());
}

#[test]
fn test_powi_arr_matches_powi_vec() {
    let base = [3u64];
    let pow = 5;
    let mut expected = powi_vec(&base, pow);
    trim_lz(&mut expected);

    let arr = powi_arr::<8>(&base, pow).expect("powi_arr failed");
    let mut got: Vec<u64> = arr.into_iter().collect();
    trim_lz(&mut got);
    assert_eq!(got, expected);
}

// ─── short_mul_vec ──────────────────────────────────────────────────────────
//
// short_mul_vec(a, b, p) returns the TOP p limbs of a*b.
// Invariant: short_mul_vec(a, b, p) == mul_vec(a, b).0[full_len - p ..]
// where full_len = a.len() + b.len() - 1.

#[test]
fn test_short_mul_vec_full_precision() {
    // When p >= full product length, result matches mul_vec exactly.
    let a = vec![3u64, 2];
    let b = vec![5u64, 1];
    let (full, _) = mul_vec(&a, &b);
    let full_len = a.len() + b.len() - 1;
    let (short, _) = short_mul_vec(&a, &b, full_len + 10);
    assert_eq!(short, full);
}

#[test]
fn test_short_mul_vec_top_one_limb() {
    let a = vec![2u64, 1]; // value = 2 + 2^64
    let b = vec![3u64];    // value = 3
    // full product = [6, 3], full_len=2, top 1 limb = [3]
    let (short, _) = short_mul_vec(&a, &b, 1);
    assert_eq!(short, vec![3]);
}

#[test]
fn test_short_mul_vec_matches_full_top() {
    // Only test p > SHORT_CUTOFF (32): that path uses mul_vec and gives exact top-p limbs.
    // The short_mul_buf path (p <= 32) is an intentional approximation — it omits carry
    // propagation from below the computed window and may not match the exact top limb.
    for seed in 0u64..10 {
        let a = rand_nonzero_vec(20, seed);
        let b = rand_nonzero_vec(20, seed + 50);
        let full_len = a.len() + b.len() - 1;
        let (mut full, _) = mul_vec(&a, &b);
        full.resize(full_len, 0);

        for p in 33..=full_len {
            let (short, _) = short_mul_vec(&a, &b, p);
            assert_eq!(short, full[full_len - p..], "seed={seed}, p={p}");
        }
    }
}

/// Boundary: test at SHORT_CUTOFF (32) — only the p > 32 path gives exact top-p limbs.
#[test]
fn test_short_mul_vec_cutoff_boundary() {
    let a = rand_nonzero_vec(20, 999);
    let b = rand_nonzero_vec(20, 888);
    let full_len = a.len() + b.len() - 1;
    let (mut full, _) = mul_vec(&a, &b);
    full.resize(full_len, 0);

    // p=33 uses the mul_vec path (p > SHORT_CUTOFF=32), giving exact top-p limbs.
    let (short, _) = short_mul_vec(&a, &b, 33);
    assert_eq!(short, full[full_len - 33..], "p=33");
}

// ─── short_sqr_vec ──────────────────────────────────────────────────────────

#[test]
fn test_short_sqr_vec_matches_sqr_top() {
    // Only test p > SHORT_SQR_CUTOFF (32): that path uses sqr_vec and gives exact top-p limbs.
    for seed in 0u64..8 {
        let v = rand_nonzero_vec(20, seed + 200);
        let full_len = 2 * v.len() - 1;
        let (mut full, _) = sqr_vec(&v);
        full.resize(full_len, 0);

        for p in 33..=full_len {
            let (short, _) = short_sqr_vec(&v, p);
            assert_eq!(short, full[full_len - p..], "seed={seed}, p={p}");
        }
    }
}

#[test]
fn test_short_sqr_vec_equals_short_mul_self() {
    // short_sqr_vec(v, p) should equal short_mul_vec(v, v, p).
    for seed in 0u64..10 {
        let v = rand_nonzero_vec(6, seed + 300);
        let full_len = 2 * v.len() - 1;
        for p in [1usize, full_len / 2, full_len] {
            let (sq, _) = short_sqr_vec(&v, p);
            let (ml, _) = short_mul_vec(&v, &v, p);
            assert_eq!(sq, ml, "seed={seed}, p={p}");
        }
    }
}

/// Boundary at SHORT_SQR_CUTOFF (32) — only p > 32 gives exact top-p limbs.
#[test]
fn test_short_sqr_vec_cutoff_boundary() {
    let v = rand_nonzero_vec(20, 777);
    let full_len = 2 * v.len() - 1;
    let (mut full, _) = sqr_vec(&v);
    full.resize(full_len, 0);

    // p=33 uses the sqr_vec path (p > SHORT_SQR_CUTOFF=32), giving exact top-p limbs.
    let (short, _) = short_sqr_vec(&v, 33);
    assert_eq!(short, full[full_len - 33..], "p=33");
}
