use super::{rand_nonzero_vec, to_u128, verify_divmod};
use crate::utils::div::*;
use crate::utils::utils::{cmp_buf, shl_buf, shr_buf, sub_buf, trim_lz};
use crate::utils::BZ_CUTOFF;

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

// ─── direct algorithm helpers ───────────────────────────────────────────────

fn assert_divmod_algorithm(name: &str, n: &[u64], d: &[u64], q: &[u64], r: &[u64]) {
    let mut q_trimmed = q.to_vec();
    trim_lz(&mut q_trimmed);
    let mut r_trimmed = r.to_vec();
    trim_lz(&mut r_trimmed);
    let mut d_trimmed = d.to_vec();
    trim_lz(&mut d_trimmed);

    assert!(
        verify_divmod(n, d, &q_trimmed, &r_trimmed),
        "{name}: q*d+r != n; q={q_trimmed:?}, r={r_trimmed:?}"
    );
    assert!(
        r_trimmed.is_empty() || cmp_buf(&r_trimmed, &d_trimmed).is_lt(),
        "{name}: remainder is not reduced; r={r_trimmed:?}, d={d_trimmed:?}"
    );
}

fn run_knuth_div_buf_of(n: &[u64], d: &[u64]) -> (Vec<u64>, Vec<u64>) {
    assert!(d.len() >= 2, "div_buf_of requires a multi-limb divisor");
    assert!(n.len() >= d.len(), "div_buf_of requires n.len() >= d.len()");

    let mut n_work = n.to_vec();
    let mut d_work = d.to_vec();
    let sh = d_work[d_work.len() - 1].leading_zeros() as u8;
    shl_buf(&mut d_work, sh);
    let mut of = shl_buf(&mut n_work, sh);

    let mut q = vec![0u64; n_work.len() - d_work.len() + 1];
    div_buf_of(&mut n_work, &mut of, &d_work, &mut q);
    assert_eq!(of, 0, "Knuth remainder overflow limb should be zero");
    shr_buf(&mut n_work, sh);

    (q, n_work)
}

fn assert_quotient_algorithm(name: &str, n: &[u64], d: &[u64], q: &[u64]) {
    let mut n_trimmed = n.to_vec();
    trim_lz(&mut n_trimmed);
    let mut d_trimmed = d.to_vec();
    trim_lz(&mut d_trimmed);
    let mut q_trimmed = q.to_vec();
    trim_lz(&mut q_trimmed);

    let qd = super::mul_ref(&q_trimmed, &d_trimmed);
    assert!(
        cmp_buf(&qd, &n_trimmed).is_le(),
        "{name}: quotient is too large; q={q_trimmed:?}"
    );

    let mut r = n_trimmed.clone();
    r.resize(r.len().max(qd.len()), 0);
    assert!(!sub_buf(&mut r, &qd), "{name}: q*d > n");
    trim_lz(&mut r);

    assert!(
        verify_divmod(n, d, &q_trimmed, &r),
        "{name}: q*d+r != n; q={q_trimmed:?}, r={r:?}"
    );
    assert!(
        r.is_empty() || cmp_buf(&r, &d_trimmed).is_lt(),
        "{name}: quotient is too small; r={r:?}, d={d_trimmed:?}"
    );
}

// ─── direct algorithm entry points ──────────────────────────────────────────

#[test]
fn test_knuth_div_buf_of_direct_invariant() {
    let cases = [
        (vec![u64::MAX, u64::MAX], vec![u64::MAX, 1]),
        (vec![0, 1, 2], vec![3, 1]),
        (vec![5, 0, 7, 9], vec![u64::MAX - 3, 8]),
    ];

    for (idx, (n, d)) in cases.into_iter().enumerate() {
        let (q, r) = run_knuth_div_buf_of(&n, &d);
        assert_divmod_algorithm(&format!("knuth edge case {idx}"), &n, &d, &q, &r);
    }

    for seed in 0u64..64 {
        let d_len = (seed % 4 + 2) as usize;
        let n_len = d_len + (seed % 5) as usize;
        let n = rand_nonzero_vec(n_len, seed + 7200);
        let d = rand_nonzero_vec(d_len, seed + 7300);

        let (q, r) = run_knuth_div_buf_of(&n, &d);
        assert_divmod_algorithm(&format!("knuth random seed={seed}"), &n, &d, &q, &r);
    }
}

#[test]
fn test_knuth_div_buf_of_varied_sizes() {
    let cases = [
        (2usize, 0usize),
        (3, 1),
        (8, 3),
        (31, 7),
        (96, 17),
        (257, 33),
    ];

    for (idx, &(d_len, extra_q)) in cases.iter().enumerate() {
        for seed in 0u64..3 {
            let n_len = d_len + extra_q;
            let n = rand_nonzero_vec(n_len, 8100 + seed + 17 * idx as u64);
            let d = rand_nonzero_vec(d_len, 8200 + seed + 17 * idx as u64);

            let (q, r) = run_knuth_div_buf_of(&n, &d);
            assert_divmod_algorithm(
                &format!("knuth varied d_len={d_len} n_len={n_len} seed={seed}"),
                &n,
                &d,
                &q,
                &r,
            );
        }
    }
}

#[test]
fn test_burnikel_ziegler_direct_invariant() {
    for seed in 0u64..4 {
        let d_len = BZ_CUTOFF + 3 + seed as usize;
        let n_len = 2 * d_len + 5;
        let n = rand_nonzero_vec(n_len, seed + 7400);
        let d = rand_nonzero_vec(d_len, seed + 7500);

        let mut n_dyn = n.clone();
        let mut d_dyn = d.clone();
        let mut q_dyn = vec![0u64; n_len - d_len + 1];
        bz_div_dyn(&mut n_dyn, &mut d_dyn, &mut q_dyn);
        assert_divmod_algorithm(
            &format!("burnikel-ziegler dyn seed={seed}"),
            &n,
            &d,
            &q_dyn,
            &n_dyn,
        );

        const N: usize = 512;
        let mut n_static = [0u64; N];
        let mut d_static = [0u64; N];
        let mut q_static = [0u64; N];
        n_static[..n_len].copy_from_slice(&n);
        d_static[..d_len].copy_from_slice(&d);
        bz_div_static::<N>(
            &mut n_static[..n_len],
            &mut d_static[..d_len],
            &mut q_static[..n_len - d_len + 1],
        );
        assert_divmod_algorithm(
            &format!("burnikel-ziegler static seed={seed}"),
            &n,
            &d,
            &q_static[..n_len - d_len + 1],
            &n_static[..n_len],
        );
    }
}

#[test]
fn test_burnikel_ziegler_varied_mul_sizes() {
    const STATIC_N: usize = 2048;

    let cases = [(BZ_CUTOFF + 1, 5usize), (192, 11), (384, 13), (640, 17)];

    for (idx, &(d_len, extra_q)) in cases.iter().enumerate() {
        for seed in 0u64..2 {
            let n_len = 2 * d_len + extra_q;
            let q_len = n_len - d_len + 1;
            assert!(n_len <= STATIC_N);
            assert!(q_len <= STATIC_N);

            let n = rand_nonzero_vec(n_len, 8300 + seed + 19 * idx as u64);
            let d = rand_nonzero_vec(d_len, 8400 + seed + 19 * idx as u64);

            let mut n_dyn = n.clone();
            let mut d_dyn = d.clone();
            let mut q_dyn = vec![0u64; q_len];
            bz_div_dyn(&mut n_dyn, &mut d_dyn, &mut q_dyn);
            assert_divmod_algorithm(
                &format!("burnikel-ziegler dyn d_len={d_len} n_len={n_len} seed={seed}"),
                &n,
                &d,
                &q_dyn,
                &n_dyn,
            );

            let mut n_static = [0u64; STATIC_N];
            let mut d_static = [0u64; STATIC_N];
            let mut q_static = [0u64; STATIC_N];
            n_static[..n_len].copy_from_slice(&n);
            d_static[..d_len].copy_from_slice(&d);
            bz_div_static::<STATIC_N>(
                &mut n_static[..n_len],
                &mut d_static[..d_len],
                &mut q_static[..q_len],
            );
            assert_divmod_algorithm(
                &format!("burnikel-ziegler static d_len={d_len} n_len={n_len} seed={seed}"),
                &n,
                &d,
                &q_static[..q_len],
                &n_static[..n_len],
            );
        }
    }
}

#[test]
fn test_newton_raphson_div_dyn_varied_mul_sizes() {
    let cases = [(6usize, 4usize), (24, 16), (96, 72), (180, 140), (360, 320)];

    for (idx, &(d_len, q_len)) in cases.iter().enumerate() {
        for seed in 0u64..2 {
            let n_len = d_len + q_len - 1;
            let n = rand_nonzero_vec(n_len, 8500 + seed + 23 * idx as u64);
            let mut d = rand_nonzero_vec(d_len, 8600 + seed + 23 * idx as u64);
            let mut q = vec![0u64; q_len];

            nr_div_dyn(&n, &mut d, &mut q);
            assert_quotient_algorithm(
                &format!("newton-raphson dyn d_len={d_len} q_len={q_len} seed={seed}"),
                &n,
                &d,
                &q,
            );
        }
    }
}
