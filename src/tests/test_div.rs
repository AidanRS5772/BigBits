use super::{rand_nonzero_vec, to_u128, verify_divmod};
use crate::utils::div::*;
use crate::utils::utils::{buf_len, cmp_buf, trim_lz};

// ─── div_prim ───────────────────────────────────────────────────────────────

#[test]
fn test_div_prim_basic() {
    let mut v = vec![6u64];
    let rem = div_prim(&mut v, 3);
    assert_eq!(v, vec![2]);
    assert_eq!(rem, 0);
}

#[test]
fn test_div_prim_with_remainder() {
    let mut v = vec![7u64];
    let rem = div_prim(&mut v, 3);
    assert_eq!(v, vec![2]);
    assert_eq!(rem, 1);
}

#[test]
fn test_div_prim_divide_zero() {
    let mut v = vec![0u64];
    let rem = div_prim(&mut v, 5);
    assert_eq!(v, vec![0]);
    assert_eq!(rem, 0);
}

#[test]
fn test_div_prim_by_one() {
    let mut v = vec![u64::MAX, 42];
    let orig = v.clone();
    let rem = div_prim(&mut v, 1);
    assert_eq!(v, orig);
    assert_eq!(rem, 0);
}

#[test]
fn test_div_prim_max_by_two() {
    let mut v = vec![u64::MAX];
    let rem = div_prim(&mut v, 2);
    assert_eq!(v, vec![u64::MAX / 2]);
    assert_eq!(rem, 1);
}

#[test]
fn test_div_prim_two_limbs() {
    // [0, 1] = 2^64. 2^64 / 2 = 2^63 = [2^63, 0].
    let mut v = vec![0u64, 1];
    let rem = div_prim(&mut v, 2);
    assert_eq!(v, vec![1u64 << 63, 0]);
    assert_eq!(rem, 0);
}

#[test]
fn test_div_prim_exact_two_limbs() {
    // (2^128 - 1) / (2^64 - 1) = 2^64 + 1 = [1, 1]
    let mut v = vec![u64::MAX, u64::MAX];
    let rem = div_prim(&mut v, u64::MAX);
    assert_eq!(v, vec![1, 1]);
    assert_eq!(rem, 0);
}

/// Verify the invariant: quotient * divisor + remainder == original, for small single-limb cases.
#[test]
fn test_div_prim_invariant() {
    let cases: &[(u64, u64)] = &[
        (0, 1),
        (1, 1),
        (100, 7),
        (u64::MAX, 3),
        (u64::MAX, u64::MAX),
        (1_000_000_000_000_000_007, 1_000_000_007),
    ];

    for &(a, d) in cases {
        let orig = to_u128(&[a]);
        let mut v = vec![a];
        let rem = div_prim(&mut v, d);
        let reconstructed = to_u128(&v) * (d as u128) + (rem as u128);
        assert_eq!(reconstructed, orig, "invariant failed for a={a}, d={d}");
    }
}

// ─── div_vec ────────────────────────────────────────────────────────────────

/// Helper: run div_vec and check that n becomes the remainder.
fn run_div(n: &[u64], d: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut n_work = n.to_vec();
    let mut d_work = d.to_vec();
    let q = div_vec(&mut n_work, &mut d_work);
    let r = n_work; // remainder now in n
    (q, r)
}

#[test]
fn test_div_vec_basic() {
    let (q, r) = run_div(&[6], &[3]);
    let mut q2 = q.clone();
    trim_lz(&mut q2);
    assert_eq!(q2, vec![2]);
    let mut r2 = r.clone();
    trim_lz(&mut r2);
    assert!(r2.is_empty() || r2.iter().all(|&x| x == 0));
}

#[test]
fn test_div_vec_with_remainder() {
    let (q, r) = run_div(&[7], &[3]);
    let mut q2 = q;
    trim_lz(&mut q2);
    assert_eq!(q2, vec![2]);
    let mut r2 = r;
    trim_lz(&mut r2);
    assert_eq!(r2, vec![1]);
}

#[test]
fn test_div_vec_n_less_than_d_returns_zero_quotient() {
    let (q, r) = run_div(&[1], &[2]);
    let mut q2 = q;
    trim_lz(&mut q2);
    assert!(q2.is_empty());
    // remainder == original n
    let mut r2 = r;
    trim_lz(&mut r2);
    assert_eq!(r2, vec![1]);
}

#[test]
fn test_div_vec_equal_n_and_d() {
    let (q, r) = run_div(&[u64::MAX], &[u64::MAX]);
    let mut q2 = q;
    trim_lz(&mut q2);
    assert_eq!(q2, vec![1]);
    let mut r2 = r;
    trim_lz(&mut r2);
    assert!(r2.is_empty() || r2.iter().all(|&x| x == 0));
}

#[test]
fn test_div_vec_two_limb_numerator() {
    // [0, 1] = 2^64, divided by [2] = 2 → quotient = [2^63], remainder = 0
    let (q, r) = run_div(&[0, 1], &[2]);
    let mut q2 = q;
    trim_lz(&mut q2);
    assert_eq!(q2, vec![1u64 << 63]);
    let mut r2 = r;
    trim_lz(&mut r2);
    assert!(r2.is_empty() || r2.iter().all(|&x| x == 0));
}

#[test]
fn test_div_vec_multi_limb_exact() {
    // (2^128 - 1) / (2^64 - 1) = 2^64 + 1 = [1, 1]
    let (q, r) = run_div(&[u64::MAX, u64::MAX], &[u64::MAX]);
    let mut q2 = q;
    trim_lz(&mut q2);
    assert_eq!(q2, vec![1, 1]);
    let mut r2 = r;
    trim_lz(&mut r2);
    assert!(r2.is_empty() || r2.iter().all(|&x| x == 0));
}

#[test]
fn test_div_vec_multi_by_multi() {
    // [u64::MAX, u64::MAX] / [u64::MAX, u64::MAX] = 1, rem = 0
    let (q, r) = run_div(&[u64::MAX, u64::MAX], &[u64::MAX, u64::MAX]);
    let mut q2 = q;
    trim_lz(&mut q2);
    assert_eq!(q2, vec![1]);
    let mut r2 = r;
    trim_lz(&mut r2);
    assert!(r2.is_empty() || r2.iter().all(|&x| x == 0));
}

/// The fundamental invariant: q * d + r == n and r < d.
/// Tested with random inputs at several sizes including the BZ_CUTOFF boundary.
#[test]
fn test_div_vec_invariant_small_random() {
    for seed in 0u64..40 {
        let n_len = (seed % 4 + 1) as usize;
        let d_len = ((seed / 4) as usize % n_len) + 1;
        let n = rand_nonzero_vec(n_len, seed);
        let d = rand_nonzero_vec(d_len, seed + 1000);

        let n_orig = n.clone();
        let mut n_work = n.clone();
        let mut d_work = d.clone();
        let q = div_vec(&mut n_work, &mut d_work);
        let r = &n_work;

        assert!(
            verify_divmod(&n_orig, &d, &q, r),
            "q*d+r != n: seed={seed}, n_len={n_len}, d_len={d_len}"
        );

        // r < d (unless q == 0 and n < d, in which case r == n < d already)
        let mut r_trimmed = r.clone();
        trim_lz(&mut r_trimmed);
        let mut d_trimmed = d.clone();
        trim_lz(&mut d_trimmed);
        assert!(
            cmp_buf(&r_trimmed, &d_trimmed) == std::cmp::Ordering::Less
                || r_trimmed.is_empty()
                || r_trimmed.iter().all(|&x| x == 0),
            "remainder r >= d: seed={seed}"
        );
    }
}

/// Test near the BZ_CUTOFF to exercise the recursive Burnikel-Ziegler path.
#[test]
fn test_div_vec_invariant_bz_cutoff_even() {
    for seed in 0u64..5 {
        let d_len = BZ_CUTOFF + (seed % 2 + 1) as usize;
        let n_len = d_len * 2 + (seed as usize % 10);
        let n = rand_nonzero_vec(n_len, seed + 5000);
        let d = rand_nonzero_vec(d_len, seed + 6000);

        let n_orig = n.clone();
        let mut n_work = n.clone();
        let mut d_work = d.clone();
        let q = div_vec(&mut n_work, &mut d_work);
        let r = n_work.clone();

        assert!(
            verify_divmod(&n_orig, &d, &q, &r),
            "BZ invariant failed for seed={seed}"
        );
    }
}

/// Deep BZ recursion: divisor large enough that div_2_1 recurses into itself.
/// Tests both even (2*BZ_CUTOFF+2) and odd (2*BZ_CUTOFF+1) at this depth.
#[test]
fn test_div_vec_invariant_bz_deep_recursive() {
    for (d_len, seed_base) in [
        (BZ_CUTOFF * 2 + 2, 10000u64),
        (BZ_CUTOFF * 2 + 1, 11000u64),
    ] {
        for seed in 0u64..3 {
            let n_len = d_len * 2 + 4;
            let n = rand_nonzero_vec(n_len, seed + seed_base);
            let d = rand_nonzero_vec(d_len, seed + seed_base + 100);

            let n_orig = n.clone();
            let mut n_work = n.clone();
            let mut d_work = d.clone();
            let q = div_vec(&mut n_work, &mut d_work);
            let r = n_work.clone();

            assert!(
                verify_divmod(&n_orig, &d, &q, &r),
                "BZ deep invariant failed: d_len={d_len}, seed={seed}"
            );
        }
    }
}

/// Test when divisor is much smaller than dividend.
#[test]
fn test_div_vec_invariant_large_quotient() {
    for seed in 0u64..10 {
        let n = rand_nonzero_vec(8, seed + 2000);
        let d = rand_nonzero_vec(1, seed + 3000);

        let n_orig = n.clone();
        let mut n_work = n.clone();
        let mut d_work = d.clone();
        let q = div_vec(&mut n_work, &mut d_work);
        let r = n_work.clone();

        assert!(
            verify_divmod(&n_orig, &d, &q, &r),
            "large quotient invariant failed for seed={seed}"
        );
    }
}

/// Test when divisor is close in size to dividend (quotient ≈ 1).
#[test]
fn test_div_vec_invariant_near_equal_sizes() {
    for seed in 0u64..10 {
        let n = rand_nonzero_vec(4, seed + 4000);
        let d = rand_nonzero_vec(3, seed + 4100);

        let n_orig = n.clone();
        let mut n_work = n.clone();
        let mut d_work = d.clone();
        let q = div_vec(&mut n_work, &mut d_work);
        let r = n_work.clone();

        assert!(
            verify_divmod(&n_orig, &d, &q, &r),
            "near-equal size invariant failed for seed={seed}"
        );
    }
}

// ─── div_arr ────────────────────────────────────────────────────────────────

#[test]
fn test_div_arr_basic() {
    let mut n = [6u64, 0, 0, 0];
    let mut d = [3u64, 0, 0, 0];
    let n_len = buf_len(&n);
    let d_len = buf_len(&d);
    let q = div_arr::<4>(&mut n[..n_len], &mut d[..d_len]);
    assert_eq!(q[0], 2);
    assert_eq!(q[1], 0);
}

#[test]
fn test_div_arr_matches_div_vec() {
    for seed in 0u64..8 {
        let n_vec = rand_nonzero_vec(4, seed + 7000);
        let d_vec = rand_nonzero_vec(2, seed + 7100);

        // div_vec path
        let n_orig = n_vec.clone();
        let mut n_work_v = n_vec.clone();
        let mut d_work_v = d_vec.clone();
        let q_vec = div_vec(&mut n_work_v, &mut d_work_v);
        let mut q_vec_trimmed = q_vec;
        trim_lz(&mut q_vec_trimmed);

        // div_arr path
        let mut n_arr = [0u64; 8];
        let mut d_arr = [0u64; 8];
        n_arr[..n_vec.len()].copy_from_slice(&n_vec);
        d_arr[..d_vec.len()].copy_from_slice(&d_vec);
        let n_a_len = buf_len(&n_arr);
        let d_a_len = buf_len(&d_arr);
        let q_arr = div_arr::<8>(&mut n_arr[..n_a_len], &mut d_arr[..d_a_len]);
        let mut q_arr_trimmed: Vec<u64> = q_arr.to_vec();
        trim_lz(&mut q_arr_trimmed);

        assert_eq!(
            q_vec_trimmed, q_arr_trimmed,
            "div_arr vs div_vec mismatch for seed={seed}"
        );

        // Also verify the invariant
        let r_arr: Vec<u64> = n_arr[..n_a_len].to_vec();
        assert!(
            verify_divmod(&n_orig, &d_vec, &q_arr_trimmed, &r_arr),
            "div_arr invariant failed for seed={seed}"
        );
    }
}

// ─── div_buf_of (tested indirectly via div_vec) ─────────────────────────────
// div_buf_of is used internally by bz_div_init and div_2_1. All div_vec tests
// that pass through the Knuth path (d.len() <= 64) exercise it indirectly.

/// Specifically trigger the Knuth path by using d.len() = 2 (just above 1 limb).
#[test]
fn test_div_buf_of_path_via_div_vec() {
    for seed in 0u64..15 {
        let n = rand_nonzero_vec(4, seed + 8000);
        let d = rand_nonzero_vec(2, seed + 8100);
        let n_orig = n.clone();
        let mut n_work = n.clone();
        let mut d_work = d.clone();
        let q = div_vec(&mut n_work, &mut d_work);
        let r = n_work.clone();
        assert!(
            verify_divmod(&n_orig, &d, &q, &r),
            "Knuth (div_buf_of) invariant failed for seed={seed}"
        );
    }
}

// ─── Panic cases (documented / expected behaviors) ──────────────────────────

#[test]
#[should_panic]
fn test_div_vec_panic_on_zero_divisor() {
    let mut n = vec![1u64];
    let mut d: Vec<u64> = vec![];
    div_vec(&mut n, &mut d);
}
