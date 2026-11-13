use super::{rand_nonzero_vec, to_u128, verify_divmod};
use crate::utils::div::*;
use crate::utils::mul::mul_dyn;
use crate::utils::utils::{add_buf, cmp_buf, dec_buf, inc_buf, shl_buf, shr_buf, sub_buf, trim_lz};
use crate::tuning::{k, resolved};
use crate::utils::{ScratchGuard};

fn knuth_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_prepared_dyn(n, d, q, request, DivAlg::Knuth);
    }
}

fn knuth_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_rem_prepared_dyn(n, d, q, request, DivAlg::Knuth);
    }
}

fn knuth_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_prepared_static::<N>(n, d, q, request, DivAlg::Knuth);
    }
}

fn knuth_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_rem_prepared_static::<N>(n, d, q, request, DivAlg::Knuth);
    }
}

fn bz_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_prepared_dyn(n, d, q, request, DivAlg::BZ);
    }
}

fn bz_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_rem_prepared_dyn(n, d, q, request, DivAlg::BZ);
    }
}

fn bz_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_prepared_static::<N>(n, d, q, request, DivAlg::BZ);
    }
}

fn bz_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_rem_prepared_static::<N>(n, d, q, request, DivAlg::BZ);
    }
}

fn nr_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_prepared_dyn(n, d, q, request, DivAlg::NR);
    }
}

fn nr_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_rem_prepared_dyn(n, d, q, request, DivAlg::NR);
    }
}

fn nr_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_prepared_static::<N>(n, d, q, request, DivAlg::NR);
    }
}

fn nr_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        div_rem_prepared_static::<N>(n, d, q, request, DivAlg::NR);
    }
}

fn knuth_rcp_dyn(d: &[u64], rcp: &mut [u64]) {
    if reciprocal_preflight(d, rcp) {
        rcp_prepared_dyn(d, rcp, RcpAlg::Knuth);
    }
}

fn knuth_rcp_static<const N: usize>(d: &[u64], rcp: &mut [u64]) {
    if reciprocal_preflight(d, rcp) {
        rcp_prepared_static::<N>(d, rcp, RcpAlg::Knuth);
    }
}

fn nr_rcp_dyn(d: &[u64], rcp: &mut [u64]) {
    if reciprocal_preflight(d, rcp) {
        rcp_prepared_dyn(d, rcp, RcpAlg::NR);
    }
}

fn nr_rcp_static<const N: usize>(d: &[u64], rcp: &mut [u64]) {
    if reciprocal_preflight(d, rcp) {
        rcp_prepared_static::<N>(d, rcp, RcpAlg::NR);
    }
}

// ─── div_prim ───────────────────────────────────────────────────────────────

// Bit-at-a-time division, independent of the reciprocal and assembly kernels.
// Keep the 129th remainder bit separately so two-limb divisors near u128::MAX work.
fn div_prim_bitwise_ref(n: &[u64], d: u128) -> (Vec<u64>, u128) {
    assert_ne!(d, 0);
    let mut q = vec![0; n.len()];
    let mut r = 0u128;
    for i in (0..n.len()).rev() {
        for bit in (0..64).rev() {
            let high = r >> 127 != 0;
            r = (r << 1) | ((n[i] >> bit) & 1) as u128;
            if high || r >= d {
                r = r.wrapping_sub(d);
                q[i] |= 1 << bit;
            }
        }
    }
    (q, r)
}

#[test]
fn test_div_prim_asm_normalization_sweep() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x6469_7670);
    // Every normalization shift, odd/even loop counts, empty inputs and long loops.
    for shift in 0..64 {
        let leading = 1u64 << (63 - shift);
        for d in [
            leading,
            leading | (leading - 1),
            leading | (rng.gen::<u64>() & (leading - 1)),
        ] {
            for len in (0..10).chain([31, 32, 33, 255, 256, 257]) {
                for pattern in 0..5 {
                    let n: Vec<u64> = (0..len)
                        .map(|i| match pattern {
                            0 => 0,
                            1 => u64::MAX,
                            2 => {
                                if i % 2 == 0 {
                                    u64::MAX
                                } else {
                                    0
                                }
                            }
                            3 => {
                                if i + 1 == len {
                                    1
                                } else {
                                    0
                                }
                            }
                            _ => rng.gen(),
                        })
                        .collect();
                    let (want, rem) = div_prim_bitwise_ref(&n, d as u128);
                    let mut storage = vec![0xa55a_a55a_a55a_a55a; len + 2];
                    storage[1..len + 1].copy_from_slice(&n);
                    let got_rem = div_prim(&mut storage[1..len + 1], d);
                    assert_eq!(
                        (&storage[1..len + 1], got_rem as u128),
                        (want.as_slice(), rem),
                        "len={len} d={d:#x} pattern={pattern}"
                    );
                    assert_eq!(storage[0], 0xa55a_a55a_a55a_a55a);
                    assert_eq!(storage[len + 1], storage[0]);
                }
            }
        }
    }
}

#[test]
fn test_div_prim2_asm_normalization_sweep() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x6469_7632);
    // Include the one-limb dispatch and every shift of a genuine two-limb divisor.
    let mut divisors = vec![1, 2, u64::MAX as u128];
    for shift in 0..64 {
        let leading = 1u128 << (127 - shift);
        divisors.extend([
            leading,
            leading | (leading - 1),
            leading | (rng.gen::<u128>() & (leading - 1)),
        ]);
    }
    for d in divisors {
        for len in (0..10).chain([31, 32, 33, 255, 256, 257]) {
            for pattern in 0..5 {
                let n: Vec<u64> = (0..len)
                    .map(|i| match pattern {
                        0 => 0,
                        1 => u64::MAX,
                        2 => {
                            if i % 2 == 0 {
                                u64::MAX
                            } else {
                                0
                            }
                        }
                        3 => {
                            if i + 1 == len {
                                1
                            } else {
                                0
                            }
                        }
                        _ => rng.gen(),
                    })
                    .collect();
                let (want, rem) = div_prim_bitwise_ref(&n, d);
                if len <= 2 {
                    let value = to_u128(&n);
                    assert_eq!((to_u128(&want), rem), (value / d, value % d));
                }
                let mut storage = vec![0xa55a_a55a_a55a_a55a; len + 2];
                storage[1..len + 1].copy_from_slice(&n);
                let got_rem = div_prim2(&mut storage[1..len + 1], d);
                assert_eq!(
                    (&storage[1..len + 1], got_rem),
                    (want.as_slice(), rem),
                    "len={len} d={d:#x} pattern={pattern}"
                );
                assert_eq!(storage[0], 0xa55a_a55a_a55a_a55a);
                assert_eq!(storage[len + 1], storage[0]);
            }
        }
    }
}

#[test]
fn test_knuth_est_asm_two_limb_boundaries() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x332f_3264);
    let edges = [0, 1, 2, (1 << 63) - 1, 1 << 63, u64::MAX - 1, u64::MAX];
    for case in 0..20_000 {
        let d1 = if case < edges.len() * edges.len() {
            edges[case / edges.len()]
        } else {
            rng.gen()
        } | (1 << 63);
        let d0 = if case < edges.len() * edges.len() {
            edges[case % edges.len()]
        } else {
            rng.gen()
        };
        let d = (d1 as u128) << 64 | d0 as u128;
        let top = match case % 4 {
            0 => 0,
            1 => d - 1,
            2 => d / 2,
            _ => rng.gen::<u128>() % d,
        };
        let u0 = edges.get(case % 10).copied().unwrap_or_else(|| rng.gen());
        let n = if case % 2 == 0 {
            // Construct q*d+r to exercise exact division and remainders next to d,
            // including q=MAX where the estimate's increment can wrap.
            let q = edges[case % edges.len()];
            let r = match case % 3 {
                0 => 0,
                1 => 1,
                _ => d - 1,
            };
            let lo = q as u128 * d0 as u128;
            let hi = q as u128 * d1 as u128 + (lo >> 64);
            let (n0, carry) = (lo as u64).overflowing_add(r as u64);
            let mid = (hi as u64) as u128 + (r >> 64) + carry as u128;
            [n0, mid as u64, ((hi >> 64) + (mid >> 64)) as u64]
        } else {
            [u0, top as u64, (top >> 64) as u64]
        };
        let (want, rem) = div_prim_bitwise_ref(&n, d);
        let mut win = [n[0], n[1]];
        let mut of = n[2];
        let q = knuth_est(&mut win, &mut of, &[d0, d1], d1, d0, reciprocal_3_2(d1, d0));
        assert_eq!(
            (q, to_u128(&win), of),
            (want[0], rem, 0),
            "n={n:x?} d={d:#x}"
        );
    }
}

#[test]
fn test_div_reciprocal_seeds_against_bitwise_division() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x7263_7032);
    let edges = [0, 1, 2, (1 << 63) - 1, 1 << 63, u64::MAX - 1, u64::MAX];
    for case in 0..10_000 {
        let d1 = if case < 49 {
            edges[case / 7]
        } else {
            rng.gen()
        } | (1 << 63);
        let d0 = if case < 49 {
            edges[case % 7]
        } else {
            rng.gen()
        };
        let want = u128::MAX / d1 as u128 - (1u128 << 64);
        assert_eq!(reciprocal_2_1(d1), want as u64, "d1={d1:#x}");
        let d = (d1 as u128) << 64 | d0 as u128;
        let (want, _) = div_prim_bitwise_ref(&[u64::MAX; 3], d);
        assert_eq!(reciprocal_3_2(d1, d0), want[0], "d={d:#x}");
    }
}

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

#[test]
fn test_dynamic_and_static_public_division_dispatch() {
    const N: usize = 32;
    let cases = [
        (vec![5u64], vec![7u64]),
        (vec![0, 1], vec![3]),
        (vec![u64::MAX, u64::MAX, 7], vec![11, 3]),
        (vec![0, 0, 1, 9, 2], vec![u64::MAX, 1, 1]),
        (vec![7, 8, 9, 10], vec![7, 8, 9, 10]),
    ];

    for (idx, (original_n, original_d)) in cases.into_iter().enumerate() {
        let body_len = div_quotient_len(original_n.len(), original_d.len());
        let q_len = body_len + usize::from(original_n.len() >= original_d.len());

        let mut q_dyn = vec![0u64; q_len];
        div_dyn(&original_n, &original_d, &mut q_dyn);

        let mut n_dyn = original_n.clone();
        let mut q_rem_dyn = vec![0u64; q_len];
        div_rem_dyn(&mut n_dyn, &original_d, &mut q_rem_dyn);
        assert_eq!(q_rem_dyn, q_dyn, "dynamic quotient mismatch case={idx}");
        assert!(verify_divmod(&original_n, &original_d, &q_rem_dyn, &n_dyn));
        assert!(cmp_buf(&n_dyn, &original_d).is_lt());

        let mut q_static = [0u64; N];
        div_static::<N>(&original_n, &original_d, &mut q_static[..q_len]);
        let mut n_static = original_n.clone();
        let mut q_rem_static = [0u64; N];
        div_rem_static::<N>(&mut n_static, &original_d, &mut q_rem_static[..q_len]);
        assert_eq!(
            &q_static[..q_len],
            &q_dyn,
            "static quotient mismatch case={idx}"
        );
        assert_eq!(
            &q_rem_static[..q_len],
            &q_dyn,
            "static div-rem quotient mismatch case={idx}"
        );
        assert!(verify_divmod(
            &original_n,
            &original_d,
            &q_rem_static[..q_len],
            &n_static
        ));
        assert_eq!(n_static, n_dyn, "remainder mismatch case={idx}");
    }
}

#[test]
fn test_public_division_dispatch_rejects_zero_divisor() {
    let dyn_result = std::panic::catch_unwind(|| {
        let mut q = [];
        div_dyn(&[1], &[], &mut q);
    });
    assert!(dyn_result.is_err());

    let static_result = std::panic::catch_unwind(|| {
        let mut q = [];
        div_static::<1>(&[1], &[], &mut q);
    });
    assert!(static_result.is_err());
}

#[test]
fn test_bz_single_limb_div_rem() {
    let original = [u64::MAX, 7];
    let mut n = original;
    let d = [3u64];
    let mut q = [0u64; 2];
    bz_div_rem_dyn(&mut n, &d, &mut q);

    let expected = (original[0] as u128) | ((original[1] as u128) << 64);
    let quotient = (q[0] as u128) | ((q[1] as u128) << 64);
    assert_eq!(quotient, expected / 3);
    assert_eq!(n, [expected.wrapping_rem(3) as u64, 0]);
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

fn forced_knuth_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_prepared_dyn(n, d, q, request, DivAlg::Knuth)
}

fn forced_bz_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_prepared_dyn(n, d, q, request, DivAlg::BZ)
}

fn forced_nr_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_prepared_dyn(n, d, q, request, DivAlg::NR)
}

fn forced_knuth_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_rem_prepared_dyn(n, d, q, request, DivAlg::Knuth)
}

fn forced_bz_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_rem_prepared_dyn(n, d, q, request, DivAlg::BZ)
}

fn forced_nr_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_rem_prepared_dyn(n, d, q, request, DivAlg::NR)
}

fn forced_knuth_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_prepared_static::<N>(n, d, q, request, DivAlg::Knuth)
}

fn forced_bz_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_prepared_static::<N>(n, d, q, request, DivAlg::BZ)
}

fn forced_nr_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_prepared_static::<N>(n, d, q, request, DivAlg::NR)
}

fn forced_knuth_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_rem_prepared_static::<N>(n, d, q, request, DivAlg::Knuth)
}

fn forced_bz_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_rem_prepared_static::<N>(n, d, q, request, DivAlg::BZ)
}

fn forced_nr_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) -> u64 {
    let request = division_preflight(n, d, q).unwrap();
    div_rem_prepared_static::<N>(n, d, q, request, DivAlg::NR)
}

fn exercise_body_overflow_contract<const N: usize>(case: &str, n: &[u64], d: &[u64]) {
    let body_len = div_quotient_len(n.len(), d.len());
    let mut expected_body = vec![0; body_len];
    let expected_overflow = forced_knuth_div_dyn(n, d, &mut expected_body);

    let dyn_divs: [(&str, fn(&[u64], &[u64], &mut [u64]) -> u64); 4] = [
        ("Knuth dyn", forced_knuth_div_dyn),
        ("Burnikel-Ziegler dyn", forced_bz_div_dyn),
        ("Newton-Raphson dyn", forced_nr_div_dyn),
        ("dispatched dyn", div_dyn),
    ];
    for (name, divide) in dyn_divs {
        let mut q = vec![u64::MAX; body_len];
        let overflow = divide(n, d, &mut q);
        assert_eq!(q, expected_body, "{case}, {name}: quotient body");
        assert_eq!(overflow, expected_overflow, "{case}, {name}: overflow");

        let mut q = vec![u64::MAX; body_len + 3];
        assert_eq!(divide(n, d, &mut q), 0, "{case}, {name}: absorbed return");
        assert_eq!(
            &q[..body_len],
            &expected_body,
            "{case}, {name}: absorbed body"
        );
        assert_eq!(
            q[body_len], expected_overflow,
            "{case}, {name}: absorbed limb"
        );
        assert!(q[body_len + 1..].iter().all(|&limb| limb == 0));
    }

    let static_divs: [(&str, fn(&[u64], &[u64], &mut [u64]) -> u64); 4] = [
        ("Knuth static", forced_knuth_div_static::<N>),
        ("Burnikel-Ziegler static", forced_bz_div_static::<N>),
        ("Newton-Raphson static", forced_nr_div_static::<N>),
        ("dispatched static", div_static::<N>),
    ];
    for (name, divide) in static_divs {
        let mut q = vec![u64::MAX; body_len];
        let overflow = divide(n, d, &mut q);
        assert_eq!(q, expected_body, "{case}, {name}: quotient body");
        assert_eq!(overflow, expected_overflow, "{case}, {name}: overflow");

        let mut q = vec![u64::MAX; body_len + 3];
        assert_eq!(divide(n, d, &mut q), 0, "{case}, {name}: absorbed return");
        assert_eq!(
            &q[..body_len],
            &expected_body,
            "{case}, {name}: absorbed body"
        );
        assert_eq!(
            q[body_len], expected_overflow,
            "{case}, {name}: absorbed limb"
        );
        assert!(q[body_len + 1..].iter().all(|&limb| limb == 0));
    }

    let mut expected_rem = n.to_vec();
    let mut expected_rem_q = vec![0; body_len];
    let expected_rem_overflow = forced_knuth_div_rem_dyn(&mut expected_rem, d, &mut expected_rem_q);
    assert_eq!(expected_rem_q, expected_body);
    assert_eq!(expected_rem_overflow, expected_overflow);

    let dyn_div_rems: [(&str, fn(&mut [u64], &[u64], &mut [u64]) -> u64); 4] = [
        ("Knuth div-rem dyn", forced_knuth_div_rem_dyn),
        ("Burnikel-Ziegler div-rem dyn", forced_bz_div_rem_dyn),
        ("Newton-Raphson div-rem dyn", forced_nr_div_rem_dyn),
        ("dispatched div-rem dyn", div_rem_dyn),
    ];
    for (name, divide) in dyn_div_rems {
        let mut rem = n.to_vec();
        let mut q = vec![u64::MAX; body_len];
        let overflow = divide(&mut rem, d, &mut q);
        assert_eq!(q, expected_body, "{case}, {name}: quotient body");
        assert_eq!(overflow, expected_overflow, "{case}, {name}: overflow");
        assert_eq!(rem, expected_rem, "{case}, {name}: remainder");

        let mut rem = n.to_vec();
        let mut q = vec![u64::MAX; body_len + 3];
        assert_eq!(
            divide(&mut rem, d, &mut q),
            0,
            "{case}, {name}: absorbed return"
        );
        assert_eq!(&q[..body_len], &expected_body);
        assert_eq!(q[body_len], expected_overflow);
        assert!(q[body_len + 1..].iter().all(|&limb| limb == 0));
        assert_eq!(rem, expected_rem, "{case}, {name}: absorbed remainder");
    }

    let static_div_rems: [(&str, fn(&mut [u64], &[u64], &mut [u64]) -> u64); 4] = [
        ("Knuth div-rem static", forced_knuth_div_rem_static::<N>),
        (
            "Burnikel-Ziegler div-rem static",
            forced_bz_div_rem_static::<N>,
        ),
        (
            "Newton-Raphson div-rem static",
            forced_nr_div_rem_static::<N>,
        ),
        ("dispatched div-rem static", div_rem_static::<N>),
    ];
    for (name, divide) in static_div_rems {
        let mut rem = n.to_vec();
        let mut q = vec![u64::MAX; body_len];
        let overflow = divide(&mut rem, d, &mut q);
        assert_eq!(q, expected_body, "{case}, {name}: quotient body");
        assert_eq!(overflow, expected_overflow, "{case}, {name}: overflow");
        assert_eq!(rem, expected_rem, "{case}, {name}: remainder");

        let mut rem = n.to_vec();
        let mut q = vec![u64::MAX; body_len + 3];
        assert_eq!(
            divide(&mut rem, d, &mut q),
            0,
            "{case}, {name}: absorbed return"
        );
        assert_eq!(&q[..body_len], &expected_body);
        assert_eq!(q[body_len], expected_overflow);
        assert!(q[body_len + 1..].iter().all(|&limb| limb == 0));
        assert_eq!(rem, expected_rem, "{case}, {name}: absorbed remainder");
    }

    for k in 0..body_len {
        let mut q = vec![u64::MAX; k];
        let overflow = hi_div_dyn(n, d, &mut q);
        assert_eq!(q, expected_body[body_len - k..], "{case}: hi dyn k={k}");
        assert_eq!(overflow, expected_overflow, "{case}: hi dyn overflow k={k}");

        let mut q_static = vec![u64::MAX; k];
        let static_overflow = hi_div_static::<N>(n, d, &mut q_static);
        assert_eq!(q_static, q, "{case}: hi static k={k}");
        assert_eq!(
            static_overflow, overflow,
            "{case}: hi static overflow k={k}"
        );
    }

    let mut exact = vec![u64::MAX; body_len];
    assert_eq!(hi_div_dyn(n, d, &mut exact), expected_overflow);
    assert_eq!(exact, expected_body);

    let mut oversized = vec![u64::MAX; body_len + 2];
    assert_eq!(hi_div_dyn(n, d, &mut oversized), 0);
    assert_eq!(&oversized[..body_len], &expected_body);
    assert_eq!(oversized[body_len], expected_overflow);
    assert_eq!(oversized[body_len + 1], 0);

    let mut exact_static = vec![u64::MAX; body_len];
    assert_eq!(
        hi_div_static::<N>(n, d, &mut exact_static),
        expected_overflow
    );
    assert_eq!(exact_static, expected_body);

    let mut oversized_static = vec![u64::MAX; body_len + 2];
    assert_eq!(hi_div_static::<N>(n, d, &mut oversized_static), 0);
    assert_eq!(&oversized_static[..body_len], &expected_body);
    assert_eq!(oversized_static[body_len], expected_overflow);
    assert_eq!(oversized_static[body_len + 1], 0);
}

#[test]
fn test_division_body_overflow_contract_matrix() {
    const N: usize = resolved::div::BZ + 32;
    let mut d = rand_nonzero_vec(k::div::bz() + 5, 13_300);
    d[0] = 0;
    d[1] = 0;
    *d.last_mut().unwrap() |= 1 << 63;
    let mut n = rand_nonzero_vec(d.len() + 20, 13_301);
    *n.last_mut().unwrap() = u64::MAX;

    exercise_body_overflow_contract::<N>("large factored divisor", &n, &d);
    exercise_body_overflow_contract::<8>("one-limb divisor", &[u64::MAX, 2, 7, 11, 3], &[19]);
    exercise_body_overflow_contract::<4>("nonzero low divisor limb", &[0, 0, 2], &[1, 1]);
}

#[test]
fn test_equal_and_shorter_division_shapes() {
    let equal_n = [7, 8, 9, 10];
    let equal_d = equal_n;
    let mut q = [];
    assert_eq!(div_dyn(&equal_n, &equal_d, &mut q), 1);
    let mut rem = equal_n;
    assert_eq!(div_rem_dyn(&mut rem, &equal_d, &mut q), 1);
    assert!(rem.iter().all(|&limb| limb == 0));

    let wide_n = [u64::MAX, u64::MAX];
    let wide_d = [1, 1];
    assert_eq!(div_dyn(&wide_n, &wide_d, &mut []), u64::MAX);
    assert_eq!(div_static::<2>(&wide_n, &wide_d, &mut []), u64::MAX);
    let mut wide_rem = wide_n;
    assert_eq!(div_rem_dyn(&mut wide_rem, &wide_d, &mut []), u64::MAX);
    assert_eq!(wide_rem, [0, 0]);

    let mut absorbed = [u64::MAX; 2];
    assert_eq!(div_dyn(&wide_n, &wide_d, &mut absorbed), 0);
    assert_eq!(absorbed, [u64::MAX, 0]);

    let n = [11, 22, 33, 4];
    let d = [0, 44, 55, 5];
    let mut oversized = [u64::MAX; 3];
    assert_eq!(div_dyn(&n, &d, &mut oversized), 0);
    assert_eq!(oversized, [0; 3]);
    let mut rem = n;
    assert_eq!(div_rem_dyn(&mut rem, &d, &mut oversized), 0);
    assert_eq!(rem, n);

    let shorter = [u64::MAX];
    let longer = [0, 1];
    assert_eq!(hi_div_dyn(&shorter, &longer, &mut []), 0);
}

#[test]
#[cfg(debug_assertions)]
fn test_standard_division_rejects_undersized_body() {
    let n = [1, 2, 3];
    let d = [7];
    let result = std::panic::catch_unwind(|| {
        let mut q = [0; 1];
        div_dyn(&n, &d, &mut q);
    });
    assert!(result.is_err());

    let result = std::panic::catch_unwind(|| {
        let mut n = n;
        let mut q = [0; 1];
        div_rem_static::<4>(&mut n, &d, &mut q);
    });
    assert!(result.is_err());
}

#[test]
fn test_hi_div_static_capacity_uses_prepared_window() {
    const N: usize = 17;
    let d = rand_nonzero_vec(12, 13_400);
    let n = rand_nonzero_vec(80, 13_401);
    let body_len = div_quotient_len(n.len(), d.len());
    let mut full_body = vec![0; body_len];
    let expected_overflow = div_dyn(&n, &d, &mut full_body);

    let mut q = [0; 5];
    let overflow = hi_div_static::<N>(&n, &d, &mut q);
    assert_eq!(&q, &full_body[body_len - q.len()..]);
    assert_eq!(overflow, expected_overflow);
}

#[test]
fn test_forced_division_entry_matrix_dyn_and_static() {
    const N: usize = resolved::div::BZ + 80;
    let d = rand_nonzero_vec(k::div::bz() + 5, 13_000);
    let n = rand_nonzero_vec(d.len() + 71, 13_001);
    let q_len = div_quotient_len(n.len(), d.len()) + 1;

    let mut expected_r = n.clone();
    let mut expected_q = vec![0u64; q_len];
    knuth_div_rem_dyn(&mut expected_r, &d, &mut expected_q);
    assert_divmod_algorithm("forced Knuth reference", &n, &d, &expected_q, &expected_r);

    let dyn_divs: [(&str, fn(&[u64], &[u64], &mut [u64])); 3] = [
        ("Knuth dyn", knuth_div_dyn),
        ("Burnikel-Ziegler dyn", bz_div_dyn),
        ("Newton-Raphson dyn", nr_div_dyn),
    ];
    for (name, algorithm) in dyn_divs {
        let n_before = n.clone();
        let d_before = d.clone();
        let mut q = vec![0u64; q_len];
        algorithm(&n, &d, &mut q);
        assert_eq!(q, expected_q, "{name}: quotient mismatch");
        assert_eq!(n, n_before, "{name}: numerator changed");
        assert_eq!(d, d_before, "{name}: divisor changed");
    }

    let dyn_div_rems: [(&str, fn(&mut [u64], &[u64], &mut [u64])); 3] = [
        ("Knuth div-rem dyn", knuth_div_rem_dyn),
        ("Burnikel-Ziegler div-rem dyn", bz_div_rem_dyn),
        ("Newton-Raphson div-rem dyn", nr_div_rem_dyn),
    ];
    for (name, algorithm) in dyn_div_rems {
        let mut r = n.clone();
        let d_before = d.clone();
        let mut q = vec![0u64; q_len];
        algorithm(&mut r, &d, &mut q);
        assert_eq!(q, expected_q, "{name}: quotient mismatch");
        assert_eq!(r, expected_r, "{name}: remainder mismatch");
        assert_eq!(d, d_before, "{name}: divisor changed");
    }

    let static_divs: [(&str, fn(&[u64], &[u64], &mut [u64])); 3] = [
        ("Knuth static", knuth_div_static::<N>),
        ("Burnikel-Ziegler static", bz_div_static::<N>),
        ("Newton-Raphson static", nr_div_static::<N>),
    ];
    for (name, algorithm) in static_divs {
        let n_before = n.clone();
        let d_before = d.clone();
        let mut q = vec![0u64; q_len];
        algorithm(&n, &d, &mut q);
        assert_eq!(q, expected_q, "{name}: quotient mismatch");
        assert_eq!(n, n_before, "{name}: numerator changed");
        assert_eq!(d, d_before, "{name}: divisor changed");
    }

    let static_div_rems: [(&str, fn(&mut [u64], &[u64], &mut [u64])); 3] = [
        ("Knuth div-rem static", knuth_div_rem_static::<N>),
        ("Burnikel-Ziegler div-rem static", bz_div_rem_static::<N>),
        ("Newton-Raphson div-rem static", nr_div_rem_static::<N>),
    ];
    for (name, algorithm) in static_div_rems {
        let mut r = n.clone();
        let d_before = d.clone();
        let mut q = vec![0u64; q_len];
        algorithm(&mut r, &d, &mut q);
        assert_eq!(q, expected_q, "{name}: quotient mismatch");
        assert_eq!(r, expected_r, "{name}: remainder mismatch");
        assert_eq!(d, d_before, "{name}: divisor changed");
    }
}

#[test]
fn test_public_division_dispatch_split_cutoffs() {
    const N: usize = 17 * (resolved::div::BZ + 1);
    let cases = [
        (
            "mandatory Knuth boundary",
            k::div::knuth(),
            32usize,
            13_110u64,
        ),
        (
            "NR eligible below BZ recursion",
            k::div::knuth() + 1,
            32,
            13_120,
        ),
        (
            "BZ choice becomes direct Knuth",
            k::div::knuth() + 8,
            160,
            13_100,
        ),
        ("last Knuth leaf width", k::div::bz(), 16 * k::div::bz(), 13_200),
        (
            "first recursive BZ width",
            k::div::bz() + 1,
            16 * (k::div::bz() + 1),
            13_210,
        ),
        ("NR fallback below BZ recursion", k::div::bz(), 7, 13_220),
        ("NR fallback above BZ recursion", k::div::bz() + 1, 7, 13_230),
    ];

    for (name, d_len, q_len, seed) in cases {
        let n_len = d_len + q_len - 1;
        assert!(n_len <= N);
        let d = rand_nonzero_vec(d_len, seed);
        let n = rand_nonzero_vec(n_len, seed + 1);

        let mut expected_r = n.clone();
        let mut expected_q = vec![0u64; q_len];
        knuth_div_rem_dyn(&mut expected_r, &d, &mut expected_q);

        let mut q_dyn = vec![0u64; q_len];
        div_dyn(&n, &d, &mut q_dyn);
        assert_eq!(q_dyn, expected_q, "{name}: dynamic quotient mismatch");

        let mut r_dyn = n.clone();
        let mut q_rem_dyn = vec![0u64; q_len];
        div_rem_dyn(&mut r_dyn, &d, &mut q_rem_dyn);
        assert_eq!(q_rem_dyn, expected_q, "{name}: dynamic div-rem quotient");
        assert_eq!(r_dyn, expected_r, "{name}: dynamic remainder mismatch");

        let mut q_static = vec![0u64; q_len];
        div_static::<N>(&n, &d, &mut q_static);
        assert_eq!(q_static, expected_q, "{name}: static quotient mismatch");

        let mut r_static = n.clone();
        let mut q_rem_static = vec![0u64; q_len];
        div_rem_static::<N>(&mut r_static, &d, &mut q_rem_static);
        assert_eq!(q_rem_static, expected_q, "{name}: static div-rem quotient");
        assert_eq!(r_static, expected_r, "{name}: static remainder mismatch");
    }
}

fn run_knuth_div_buf_of(n: &[u64], d: &[u64]) -> (Vec<u64>, Vec<u64>) {
    assert!(d.len() >= 2, "div_buf_of requires a multi-limb divisor");
    assert!(n.len() >= d.len(), "div_buf_of requires n.len() >= d.len()");

    let mut n_work = n.to_vec();
    let mut d_work = d.to_vec();
    let sh = d_work[d_work.len() - 1].leading_zeros() as u8;
    shl_buf(&mut d_work, sh);
    let mut of = shl_buf(&mut n_work, sh);

    let mut q = vec![0u64; n_work.len() - d_work.len()];
    let quotient_overflow = div_buf_of(&mut n_work, &mut of, &d_work, &mut q);
    q.push(quotient_overflow);
    assert_eq!(of, 0, "Knuth remainder overflow limb should be zero");
    shr_buf(&mut n_work, sh);

    (q, n_work)
}

fn run_bz_top_block_knuth(n: &mut [u64], d: &[u64], out: &mut [u64]) {
    let mut of = 0;
    div_buf_of(n, &mut of, d, out);
    assert_eq!(of, 0, "top-block Knuth overflow limb should be zero");
}

fn run_bz_top_block_padded(n: &mut [u64], d: &[u64], out: &mut [u64]) {
    let dlen = d.len();
    let qlen = out.len();
    let mut scratch = ScratchGuard::acquire();
    let [top_n, q_tmp, div_scratch] = scratch.get_splits([2 * dlen, dlen, dlen]);

    top_n[..n.len()].copy_from_slice(n);
    top_n[n.len()..].fill(0);

    div_2_1(top_n, d, q_tmp, div_scratch, &mut |n, d, q| {
        mul_dyn(n, d, q);
    });

    out.copy_from_slice(&q_tmp[..qlen]);
    n.fill(0);
    n[..dlen].copy_from_slice(&top_n[..dlen]);
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
        let d_len = k::div::bz() + 3 + seed as usize;
        let n_len = 2 * d_len + 5;
        let n = rand_nonzero_vec(n_len, seed + 7400);
        let d = rand_nonzero_vec(d_len, seed + 7500);

        let mut n_dyn = n.clone();
        let d_dyn = d.clone();
        let mut q_dyn = vec![0u64; n_len - d_len + 1];
        bz_div_rem_dyn(&mut n_dyn, &d_dyn, &mut q_dyn);
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
        bz_div_rem_static::<N>(
            &mut n_static[..n_len],
            &d_static[..d_len],
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
fn test_burnikel_ziegler_dynamic_top_block_shapes() {
    let d_len = k::div::bz() + 8;
    let recursive_top_q = (d_len + 1) / 2;
    let cases = [
        ("t0_knuth", recursive_top_q - 1),
        ("t0_scratch_q", recursive_top_q),
        ("t0_out_q", d_len),
        ("t1_reused_out_q", d_len + recursive_top_q),
    ];

    for (idx, &(name, q_len)) in cases.iter().enumerate() {
        let n_len = d_len + q_len - 1;
        let n = rand_nonzero_vec(n_len, 8700 + idx as u64);
        let d = rand_nonzero_vec(d_len, 8800 + idx as u64);
        let mut n_work = n.clone();
        let d_work = d.clone();
        let mut q = vec![0u64; q_len];

        bz_div_rem_dyn(&mut n_work, &d_work, &mut q);
        assert_divmod_algorithm(name, &n, &d, &q, &n_work);
    }
}

#[test]
fn test_burnikel_ziegler_static_top_block_shapes() {
    fn run<const N: usize>(name: &str, d_len: usize, q_len: usize, seed: u64) {
        let n_len = d_len + q_len - 1;
        assert!(n_len <= N);

        let n = rand_nonzero_vec(n_len, seed);
        let d = rand_nonzero_vec(d_len, seed + 100);
        let mut n_work = [0u64; N];
        let mut d_work = [0u64; N];
        let mut q = [0u64; N];
        n_work[..n_len].copy_from_slice(&n);
        d_work[..d_len].copy_from_slice(&d);

        bz_div_rem_static::<N>(&mut n_work[..n_len], &d_work[..d_len], &mut q[..q_len]);
        assert_divmod_algorithm(name, &n, &d, &q[..q_len], &n_work[..n_len]);
    }

    const FIT_N: usize = 3 * (resolved::div::BZ + 8);
    let d_len = k::div::bz() + 8;
    let recursive_top_q = (d_len + 1) / 2;
    run::<FIT_N>("static t0 knuth", d_len, recursive_top_q - 1, 8900);
    run::<FIT_N>("static t0 scratch q", d_len, recursive_top_q, 8901);
    run::<FIT_N>("static t0 out q", d_len, d_len, 8902);
    run::<FIT_N>(
        "static t1 reused out q",
        d_len,
        d_len + recursive_top_q,
        8903,
    );

    // Fit the operands, but not the padded 2*d_len BZ top block.
    const LIMITED_N: usize = 2 * (resolved::div::BZ + 8) - 1;
    run::<LIMITED_N>("static capacity fallback", d_len, recursive_top_q, 8904);
}

#[test]
fn test_burnikel_ziegler_top_block_cost_model_is_monotone() {
    for d_len in [k::div::bz(), k::div::bz() + 8, 192, 512, 1024] {
        assert!(!use_bz_for_top_block(d_len, 0));
        if d_len <= k::div::bz() {
            assert!(!use_bz_for_top_block(d_len, d_len));
            continue;
        }

        let mut seen_bz = false;
        for q_len in 1..=d_len {
            let use_bz = use_bz_for_top_block(d_len, q_len);
            assert_eq!(
                use_bz,
                k::div::bz_top_padded_cost_scale() * bz_2_1_cost(d_len)
                    <= bz_top_block_knuth_work(d_len, q_len)
            );
            assert!(
                !seen_bz || use_bz,
                "BZ top-block dispatch should stay true after d_len={d_len}, q_len={q_len}"
            );
            seen_bz |= use_bz;
        }
    }
}

#[test]
fn test_burnikel_ziegler_forced_top_block_paths_match() {
    let cases = [
        (k::div::bz() + 8, 1usize),
        (k::div::bz() + 8, (k::div::bz() + 8) / 2),
        (k::div::bz() + 8, k::div::bz() + 8),
        (192, 17),
        (384, 192),
    ];

    for (idx, &(d_len, q_len)) in cases.iter().enumerate() {
        let n_len = d_len + q_len - 1;
        let n = rand_nonzero_vec(n_len, 9000 + idx as u64);
        let mut d = rand_nonzero_vec(d_len, 9100 + idx as u64);
        d[d_len - 1] |= 1 << 63;

        let mut n_knuth = n.clone();
        let mut n_bz = n.clone();
        let mut q_knuth = vec![0u64; q_len];
        let mut q_bz = vec![0u64; q_len];

        run_bz_top_block_knuth(&mut n_knuth, &d, &mut q_knuth);
        run_bz_top_block_padded(&mut n_bz, &d, &mut q_bz);

        assert_eq!(q_bz, q_knuth, "forced top quotient mismatch");
        assert_eq!(n_bz, n_knuth, "forced top remainder mismatch");
        assert_divmod_algorithm(
            &format!("forced top block d_len={d_len} q_len={q_len}"),
            &n,
            &d,
            &q_bz,
            &n_bz,
        );
    }
}

#[test]
fn test_burnikel_ziegler_varied_mul_sizes() {
    const STATIC_N: usize = 2048;

    let cases = [(k::div::bz() + 1, 5usize), (192, 11), (384, 13), (640, 17)];

    for (idx, &(d_len, extra_q)) in cases.iter().enumerate() {
        for seed in 0u64..2 {
            let n_len = 2 * d_len + extra_q;
            let q_len = n_len - d_len + 1;
            assert!(n_len <= STATIC_N);
            assert!(q_len <= STATIC_N);

            let n = rand_nonzero_vec(n_len, 8300 + seed + 19 * idx as u64);
            let d = rand_nonzero_vec(d_len, 8400 + seed + 19 * idx as u64);

            let mut n_dyn = n.clone();
            let d_dyn = d.clone();
            let mut q_dyn = vec![0u64; q_len];
            bz_div_rem_dyn(&mut n_dyn, &d_dyn, &mut q_dyn);
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
            bz_div_rem_static::<STATIC_N>(
                &mut n_static[..n_len],
                &d_static[..d_len],
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
fn test_newton_raphson_schedule_lands_exactly_on_target() {
    let mut sizes = [0usize; 64];

    let steps = nr_rcp_schedule(12, &mut sizes);
    assert_eq!(&sizes[..steps], &[12, 6, 3, 2]);

    let steps = nr_rcp_schedule(1027, &mut sizes);
    assert_eq!(
        &sizes[..steps],
        &[1027, 514, 257, 129, 65, 33, 17, 9, 5, 3, 2]
    );

    assert_eq!(nr_rcp_schedule(0, &mut sizes), 0);
    assert_eq!(nr_rcp_schedule(1, &mut sizes), 0);

    let steps = nr_rcp_schedule(usize::MAX, &mut sizes);
    assert_eq!(steps, usize::BITS as usize);
    assert_eq!(sizes[steps - 1], 2);

    // Forward from precision 1, every step must double or double-minus-one-limb
    // and the chain must land exactly on the target.
    for p_target in 2usize..=600 {
        let steps = nr_rcp_schedule(p_target, &mut sizes);
        assert_eq!(sizes[0], p_target);
        let mut p = 1;
        for &q in sizes[..steps].iter().rev() {
            assert!(
                q == 2 * p || q == 2 * p - 1,
                "p_target={p_target} p={p} q={q}"
            );
            p = q;
        }
        assert_eq!(p, p_target);
    }
}

#[test]
fn test_newton_raphson_div_dyn_trim_schedule_sweep() {
    // Contiguous q_len sweep (from the q_len >= 8 contract floor) hits varied
    // trim patterns in the reciprocal schedule, with both the padded and
    // sliced d_work paths.
    for q_len in 8usize..=34 {
        for (case, d_len) in [(0u64, q_len + 3), (1, 2 * q_len + 8)] {
            let n_len = d_len + q_len - 1;
            let n = rand_nonzero_vec(n_len, 9300 + 7 * q_len as u64 + case);
            let mut d = rand_nonzero_vec(d_len, 9400 + 7 * q_len as u64 + case);
            let mut q = vec![0u64; q_len];

            nr_div_dyn(&n, &mut d, &mut q);
            assert_quotient_algorithm(
                &format!("newton-raphson trim sweep d_len={d_len} q_len={q_len}"),
                &n,
                &d,
                &q,
            );
        }
    }

    // Reciprocal targets on and just above powers of two, where the old blind
    // doubling wasted the most work and the new schedule is trim-heavy.
    for q_len in [59usize, 60, 61, 62, 124, 125, 126, 127, 251, 252, 253] {
        let d_len = q_len + 3;
        let n_len = d_len + q_len - 1;
        let n = rand_nonzero_vec(n_len, 9500 + 7 * q_len as u64);
        let mut d = rand_nonzero_vec(d_len, 9600 + 7 * q_len as u64);
        let mut q = vec![0u64; q_len];

        nr_div_dyn(&n, &mut d, &mut q);
        assert_quotient_algorithm(
            &format!("newton-raphson trim sweep large d_len={d_len} q_len={q_len}"),
            &n,
            &d,
            &q,
        );
    }
}

#[test]
fn test_newton_raphson_div_dyn_varied_mul_sizes() {
    let cases = [
        (10usize, 8usize),
        (24, 16),
        (96, 72),
        (180, 140),
        (360, 320),
    ];

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

#[test]
fn test_newton_raphson_div_dyn_bench_1024_regression() {
    let d_len = 1024;
    let n_len = 2 * d_len;
    let q_len = d_len + 1;
    let n = rand_nonzero_vec(n_len, 9000);
    let mut d = rand_nonzero_vec(d_len, 9100);
    d[d_len - 1] |= 1 << 63;
    let mut q = vec![0u64; q_len];

    nr_div_dyn(&n, &mut d, &mut q);
    assert_quotient_algorithm("newton-raphson dyn bench size 1024", &n, &d, &q);
}

#[test]
fn test_newton_raphson_div_dyn_exact_multiple_boundaries() {
    // n = d·k has remainder 0, so the correction guards are ambiguous and the
    // exact-correction path must run deterministically. n ± offsets probe both
    // sides of the integer boundary. Sizes cover both padded and sliced d_top.
    let cases = [(96usize, 96usize), (200, 150), (150, 200), (40, 40)];

    for (idx, &(d_len, k_len)) in cases.iter().enumerate() {
        let mut d = rand_nonzero_vec(d_len, 9700 + 13 * idx as u64);
        let mut k = rand_nonzero_vec(k_len, 9800 + 13 * idx as u64);
        d[d_len - 1] |= 1 << 63;
        k[k_len - 1] |= 1 << 63;

        let n_len = d_len + k_len;
        let q_len = n_len - d_len + 1;
        let mut n = vec![0u64; n_len];
        n[n_len - 1] = mul_dyn(&d, &k, &mut n[..n_len - 1]);
        assert_ne!(n[n_len - 1], 0);

        let mut q = vec![0u64; q_len];
        nr_div_dyn(&n, &mut d, &mut q);
        assert_eq!(
            &q[..k_len],
            &k[..],
            "exact multiple d_len={d_len} k_len={k_len}"
        );
        assert_eq!(
            q[k_len], 0,
            "exact multiple top d_len={d_len} k_len={k_len}"
        );

        // n + (d-1): still q = k, with the maximum remainder.
        let mut d_m1 = d.clone();
        dec_buf(&mut d_m1);
        let mut n_hi = n.clone();
        assert!(!add_buf(&mut n_hi, &d_m1));
        let mut q = vec![0u64; q_len];
        nr_div_dyn(&n_hi, &mut d, &mut q);
        assert_eq!(
            &q[..k_len],
            &k[..],
            "max remainder d_len={d_len} k_len={k_len}"
        );
        assert_eq!(q[k_len], 0, "max remainder top d_len={d_len} k_len={k_len}");

        // n - 1 = d·(k-1) + (d-1): boundary from below.
        let mut n_lo = n.clone();
        dec_buf(&mut n_lo);
        let mut k_m1 = k.clone();
        dec_buf(&mut k_m1);
        let mut q = vec![0u64; q_len];
        nr_div_dyn(&n_lo, &mut d, &mut q);
        assert_eq!(
            &q[..k_len],
            &k_m1[..],
            "boundary below d_len={d_len} k_len={k_len}"
        );
        assert_eq!(
            q[k_len], 0,
            "boundary below top d_len={d_len} k_len={k_len}"
        );
    }
}

#[test]
fn test_newton_raphson_limb_power_and_near_power_divisors() {
    const N: usize = 64;
    for &d_len in &[6usize, 20] {
        for &q_len in &[8usize, 9, 20] {
            for &low in &[0u64, 1, 2, 19, u64::MAX] {
                let n_len = d_len + q_len - 1;
                let n = vec![u64::MAX; n_len];
                let mut d = vec![0u64; d_len];
                d[0] = low;
                d[d_len - 1] = 1;

                let mut n_bz = n.clone();
                let d_bz = d.clone();
                let mut q_bz = vec![0u64; q_len];
                bz_div_rem_dyn(&mut n_bz, &d_bz, &mut q_bz);

                let mut d_dyn = d.clone();
                let mut q_dyn = vec![0u64; q_len];
                nr_div_dyn(&n, &mut d_dyn, &mut q_dyn);
                assert_eq!(q_dyn, q_bz, "dyn q d={d_len} q={q_len} low={low}");
                assert_eq!(d_dyn, d);

                let mut n_rem_dyn = n.clone();
                let mut d_rem_dyn = d.clone();
                let mut q_rem_dyn = vec![0u64; q_len];
                nr_div_rem_dyn(&mut n_rem_dyn, &mut d_rem_dyn, &mut q_rem_dyn);
                assert_eq!(q_rem_dyn, q_bz, "dyn rem q d={d_len} q={q_len} low={low}");
                assert_eq!(n_rem_dyn, n_bz, "dyn r d={d_len} q={q_len} low={low}");

                let mut d_static = d.clone();
                let mut q_static = vec![0u64; q_len];
                nr_div_static::<N>(&n, &mut d_static, &mut q_static);
                assert_eq!(q_static, q_bz, "static q d={d_len} q={q_len} low={low}");

                let mut n_rem_static = n.clone();
                let mut d_rem_static = d.clone();
                let mut q_rem_static = vec![0u64; q_len];
                nr_div_rem_static::<N>(&mut n_rem_static, &mut d_rem_static, &mut q_rem_static);
                assert_eq!(
                    q_rem_static, q_bz,
                    "static rem q d={d_len} q={q_len} low={low}"
                );
                assert_eq!(n_rem_static, n_bz, "static r d={d_len} q={q_len} low={low}");
            }
        }
    }
}

#[test]
fn test_newton_raphson_unsupported_small_shape_falls_back() {
    let n = vec![u64::MAX; 9];
    let d = vec![7, 3];
    let mut q_bz = vec![0u64; 8];
    bz_div_dyn(&n, &d, &mut q_bz);

    let mut d_dyn = d.clone();
    let mut q_dyn = vec![0u64; 8];
    nr_div_dyn(&n, &mut d_dyn, &mut q_dyn);
    assert_eq!(q_dyn, q_bz);

    let mut d_static = d;
    let mut q_static = vec![0u64; 8];
    nr_div_static::<16>(&n, &mut d_static, &mut q_static);
    assert_eq!(q_static, q_bz);
}

#[test]
fn test_newton_raphson_div_rem_dyn_varied_sizes() {
    let cases = [
        (10usize, 8usize),
        (24, 16),
        (96, 72),
        (180, 140),
        (360, 320),
    ];

    for (idx, &(d_len, q_len)) in cases.iter().enumerate() {
        for seed in 0u64..2 {
            let n_len = d_len + q_len - 1;
            let n = rand_nonzero_vec(n_len, 8500 + seed + 23 * idx as u64);
            let d = rand_nonzero_vec(d_len, 8600 + seed + 23 * idx as u64);

            let mut n_work = n.clone();
            let mut d_work = d.clone();
            let mut q = vec![0u64; q_len];
            nr_div_rem_dyn(&mut n_work, &mut d_work, &mut q);

            assert_eq!(d, d_work, "d must be preserved d_len={d_len} q_len={q_len}");
            assert!(
                n_work[d_len + 1..].iter().all(|&r| r == 0),
                "remainder window must be zero above d.len()+1"
            );
            assert_divmod_algorithm(
                &format!("newton-raphson div_rem d_len={d_len} q_len={q_len} seed={seed}"),
                &n,
                &d,
                &q,
                &n_work,
            );
        }
    }
}

#[test]
fn test_newton_raphson_div_rem_dyn_boundaries() {
    // Exact multiples and off-by-one numerators pin the exact q and r values
    // and force the +-1 fixup window onto integer boundaries.
    let cases = [(96usize, 96usize), (200, 150), (150, 200), (40, 40)];

    for (idx, &(d_len, k_len)) in cases.iter().enumerate() {
        let mut d = rand_nonzero_vec(d_len, 9700 + 13 * idx as u64);
        let mut k = rand_nonzero_vec(k_len, 9800 + 13 * idx as u64);
        d[d_len - 1] |= 1 << 63;
        k[k_len - 1] |= 1 << 63;

        let n_len = d_len + k_len;
        let q_len = n_len - d_len + 1;
        let mut n = vec![0u64; n_len];
        n[n_len - 1] = mul_dyn(&d, &k, &mut n[..n_len - 1]);
        let mut d_m1 = d.clone();
        dec_buf(&mut d_m1);

        // n = d*k: q = k, r = 0.
        let mut n_work = n.clone();
        let mut q = vec![0u64; q_len];
        nr_div_rem_dyn(&mut n_work, &mut d, &mut q);
        assert_eq!(&q[..k_len], &k[..], "exact multiple q d_len={d_len}");
        assert_eq!(q[k_len], 0, "exact multiple q top d_len={d_len}");
        assert!(
            n_work.iter().all(|&r| r == 0),
            "exact multiple r d_len={d_len}"
        );

        // n + (d-1): q = k, r = d-1.
        let mut n_work = n.clone();
        assert!(!add_buf(&mut n_work, &d_m1));
        let mut q = vec![0u64; q_len];
        nr_div_rem_dyn(&mut n_work, &mut d, &mut q);
        assert_eq!(&q[..k_len], &k[..], "max remainder q d_len={d_len}");
        assert_eq!(&n_work[..d_len], &d_m1[..], "max remainder r d_len={d_len}");
        assert!(n_work[d_len..].iter().all(|&r| r == 0));

        // n - 1 = d*(k-1) + (d-1): q = k-1, r = d-1.
        let mut n_work = n.clone();
        dec_buf(&mut n_work);
        let mut k_m1 = k.clone();
        dec_buf(&mut k_m1);
        let mut q = vec![0u64; q_len];
        nr_div_rem_dyn(&mut n_work, &mut d, &mut q);
        assert_eq!(&q[..k_len], &k_m1[..], "boundary below q d_len={d_len}");
        assert_eq!(
            &n_work[..d_len],
            &d_m1[..],
            "boundary below r d_len={d_len}"
        );
        assert!(n_work[d_len..].iter().all(|&r| r == 0));
    }
}

#[test]
fn test_newton_raphson_div_rem_dyn_shape_sweep() {
    // Small sizes from the contract floor plus skewed operands; q longer than
    // d exercises the q mod B^(d.len()+1) truncation in the remainder window.
    let mut cases: Vec<(usize, usize)> = Vec::new();
    for q_len in 8..=24 {
        cases.push((q_len + 3, q_len));
        cases.push((2 * q_len + 8, q_len));
    }
    cases.push((8, 200));
    cases.push((12, 129));
    cases.push((600, 80));
    cases.push((400, 33));

    for (idx, &(d_len, q_len)) in cases.iter().enumerate() {
        let n_len = d_len + q_len - 1;
        let n = rand_nonzero_vec(n_len, 10_400 + 7 * idx as u64);
        let d = rand_nonzero_vec(d_len, 10_500 + 7 * idx as u64);

        let mut n_work = n.clone();
        let mut d_work = d.clone();
        let mut q = vec![0u64; q_len];
        nr_div_rem_dyn(&mut n_work, &mut d_work, &mut q);

        assert_divmod_algorithm(
            &format!("newton-raphson div_rem skewed d_len={d_len} q_len={q_len}"),
            &n,
            &d,
            &q,
            &n_work,
        );
    }
}

#[test]
fn test_newton_raphson_div_static_matches_dyn() {
    const STATIC_N: usize = 2048;
    // Varied and skewed shapes; the exact quotient and remainder must agree
    // with the dyn pipeline limb for limb.
    let cases = [
        (10usize, 8usize),
        (24, 16),
        (96, 72),
        (180, 140),
        (360, 320),
        (8, 200),
        (12, 129),
        (600, 80),
        (400, 33),
    ];

    for (idx, &(d_len, q_len)) in cases.iter().enumerate() {
        let n_len = d_len + q_len - 1;
        assert!(n_len <= STATIC_N);
        let n = rand_nonzero_vec(n_len, 11_000 + 7 * idx as u64);
        let d = rand_nonzero_vec(d_len, 11_100 + 7 * idx as u64);

        let mut d_dyn = d.clone();
        let mut q_dyn = vec![0u64; q_len];
        nr_div_dyn(&n, &mut d_dyn, &mut q_dyn);

        let mut d_static = d.clone();
        let mut q_static = vec![0u64; q_len];
        nr_div_static::<STATIC_N>(&n, &mut d_static, &mut q_static);
        assert_eq!(
            q_dyn, q_static,
            "static quotient mismatch d_len={d_len} q_len={q_len}"
        );
        assert_eq!(d, d_static, "d must be preserved d_len={d_len}");

        let mut n_rem_dyn = n.clone();
        let mut d_rem_dyn = d.clone();
        let mut q_rem_dyn = vec![0u64; q_len];
        nr_div_rem_dyn(&mut n_rem_dyn, &mut d_rem_dyn, &mut q_rem_dyn);

        let mut n_rem_static = n.clone();
        let mut d_rem_static = d.clone();
        let mut q_rem_static = vec![0u64; q_len];
        nr_div_rem_static::<STATIC_N>(&mut n_rem_static, &mut d_rem_static, &mut q_rem_static);
        assert_eq!(
            q_rem_dyn, q_rem_static,
            "static div_rem quotient mismatch d_len={d_len} q_len={q_len}"
        );
        assert_eq!(
            n_rem_dyn, n_rem_static,
            "static remainder mismatch d_len={d_len} q_len={q_len}"
        );
        assert_divmod_algorithm(
            &format!("newton-raphson static div_rem d_len={d_len} q_len={q_len}"),
            &n,
            &d,
            &q_rem_static,
            &n_rem_static,
        );
    }
}

#[test]
fn test_newton_raphson_div_rem_static_boundaries() {
    const STATIC_N: usize = 512;
    // Exact multiples and off-by-one numerators pin exact q and r values on
    // the stack-only pipeline.
    for (idx, &(d_len, k_len)) in [(96usize, 96usize), (150, 200)].iter().enumerate() {
        let mut d = rand_nonzero_vec(d_len, 11_200 + 13 * idx as u64);
        let mut k = rand_nonzero_vec(k_len, 11_300 + 13 * idx as u64);
        d[d_len - 1] |= 1 << 63;
        k[k_len - 1] |= 1 << 63;

        let n_len = d_len + k_len;
        let q_len = n_len - d_len + 1;
        assert!(n_len <= STATIC_N);
        let mut n = vec![0u64; n_len];
        n[n_len - 1] = mul_dyn(&d, &k, &mut n[..n_len - 1]);
        let mut d_m1 = d.clone();
        dec_buf(&mut d_m1);

        // n = d*k: q = k, r = 0.
        let mut n_work = n.clone();
        let mut q = vec![0u64; q_len];
        nr_div_rem_static::<STATIC_N>(&mut n_work, &mut d, &mut q);
        assert_eq!(&q[..k_len], &k[..], "static exact multiple q d_len={d_len}");
        assert_eq!(q[k_len], 0);
        assert!(n_work.iter().all(|&r| r == 0), "static exact multiple r");

        // n + (d-1): q = k, r = d-1.
        let mut n_work = n.clone();
        assert!(!add_buf(&mut n_work, &d_m1));
        let mut q = vec![0u64; q_len];
        nr_div_rem_static::<STATIC_N>(&mut n_work, &mut d, &mut q);
        assert_eq!(&q[..k_len], &k[..], "static max remainder q d_len={d_len}");
        assert_eq!(&n_work[..d_len], &d_m1[..], "static max remainder r");
        assert!(n_work[d_len..].iter().all(|&r| r == 0));

        // n - 1 = d*(k-1) + (d-1): q = k-1, r = d-1.
        let mut n_work = n.clone();
        dec_buf(&mut n_work);
        let mut k_m1 = k.clone();
        dec_buf(&mut k_m1);
        let mut q = vec![0u64; q_len];
        nr_div_rem_static::<STATIC_N>(&mut n_work, &mut d, &mut q);
        assert_eq!(
            &q[..k_len],
            &k_m1[..],
            "static boundary below q d_len={d_len}"
        );
        assert_eq!(&n_work[..d_len], &d_m1[..], "static boundary below r");
        assert!(n_work[d_len..].iter().all(|&r| r == 0));
    }
}

#[test]
fn test_newton_raphson_div_static_tight_fit() {
    // n_len == N exactly, with a 5-smooth and a non-5-smooth N; the latter's
    // near-N remainder product drives the chunked static NTT split.
    fn run<const N: usize>(d_len: usize, seed: u64) {
        let q_len = N - d_len + 1;
        let n = rand_nonzero_vec(N, seed);
        let d = rand_nonzero_vec(d_len, seed + 50);

        let mut n_rem_dyn = n.clone();
        let mut d_rem_dyn = d.clone();
        let mut q_rem_dyn = vec![0u64; q_len];
        nr_div_rem_dyn(&mut n_rem_dyn, &mut d_rem_dyn, &mut q_rem_dyn);

        let mut n_rem_static = n.clone();
        let mut d_rem_static = d.clone();
        let mut q_rem_static = vec![0u64; q_len];
        nr_div_rem_static::<N>(&mut n_rem_static, &mut d_rem_static, &mut q_rem_static);

        assert_eq!(q_rem_dyn, q_rem_static, "tight fit quotient N={N}");
        assert_eq!(n_rem_dyn, n_rem_static, "tight fit remainder N={N}");
    }

    run::<2048>(1024, 11_400);
    run::<2043>(1022, 11_500);
}

#[test]
fn test_newton_raphson_div_dyn_skewed_shapes() {
    // d much shorter than q (padded d_top) and much longer (sliced d_top),
    // covering both h-l parities of the high/low quotient split.
    let cases = [(8usize, 200usize), (12, 129), (600, 80), (400, 33)];

    for (idx, &(d_len, q_len)) in cases.iter().enumerate() {
        let n_len = d_len + q_len - 1;
        let n = rand_nonzero_vec(n_len, 10_000 + 7 * idx as u64);
        let mut d = rand_nonzero_vec(d_len, 10_100 + 7 * idx as u64);
        let mut q = vec![0u64; q_len];

        nr_div_dyn(&n, &mut d, &mut q);
        assert_quotient_algorithm(
            &format!("newton-raphson skewed d_len={d_len} q_len={q_len}"),
            &n,
            &d,
            &q,
        );
    }
}

// ─── reciprocal stack ────────────────────────────────────────────────────────

pub(crate) fn exact_rcp_reference(d: &[u64], r_len: usize) -> Vec<u64> {
    if r_len == 0 {
        return Vec::new();
    }
    let n_len = d
        .len()
        .checked_add(r_len)
        .expect("reciprocal reference size overflow");
    let mut n = vec![0u64; n_len];
    n[n_len - 1] = 1;
    let mut q = vec![0u64; r_len + 1];
    knuth_div_dyn(&n, d, &mut q);
    if q[r_len] != 0 {
        vec![u64::MAX; r_len]
    } else {
        q.truncate(r_len);
        q
    }
}

pub(crate) fn assert_rcp_precision(name: &str, actual: &[u64], exact: &[u64]) {
    assert_eq!(actual.len(), exact.len(), "{name}: output length changed");
    let mut error = match cmp_buf(actual, exact) {
        std::cmp::Ordering::Less => {
            let mut error = exact.to_vec();
            assert!(!sub_buf(&mut error, actual));
            error
        }
        std::cmp::Ordering::Equal => vec![0; actual.len()],
        std::cmp::Ordering::Greater => {
            let mut error = actual.to_vec();
            assert!(!sub_buf(&mut error, exact));
            error
        }
    };
    trim_lz(&mut error);
    assert!(
        error.len() <= 1,
        "{name}: reciprocal error exceeds one limb; error={error:?}"
    );
}

fn run_all_precision_rcps(d: &[u64], r_len: usize) -> Vec<Vec<u64>> {
    let original = d.to_vec();
    let exact = exact_rcp_reference(d, r_len);
    let algorithms: [(&str, fn(&[u64], &mut [u64])); 3] = [
        ("knuth", knuth_rcp_dyn),
        ("newton-raphson", nr_rcp_dyn),
        ("dispatcher", rcp_dyn),
    ];
    let mut results = Vec::new();

    for (name, algorithm) in algorithms {
        let mut rcp = vec![0xa5a5_a5a5_a5a5_a5a5; r_len];
        algorithm(d, &mut rcp);
        assert_eq!(d, original, "{name} modified the divisor");
        assert_rcp_precision(name, &rcp, &exact);
        results.push(rcp);
    }
    results
}

#[test]
fn test_knuth_div_rcp_seed_is_upper_biased() {
    let cases = [(2usize, 1usize), (2, 4), (3, 3), (4, 7), (6, 2), (5, 12)];
    for (idx, &(d_len, r_len)) in cases.iter().enumerate() {
        let mut d = rand_nonzero_vec(d_len, 12_000 + 17 * idx as u64);
        d[d_len - 1] |= 1 << 63;

        let mut exact = exact_rcp_reference(&d, r_len);
        assert!(!inc_buf(&mut exact));

        let mut seed = vec![0u64; r_len];
        let mut win = vec![u64::MAX; d_len];
        knuth_rcp_normalized(&d, &mut seed, &mut win, 1);
        assert!(!inc_buf(&mut seed));
        assert_eq!(
            exact, seed,
            "exact reciprocal + 1 vs Knuth division seed d_len={d_len} r_len={r_len}"
        );
    }
}

#[test]
fn test_explicit_reciprocal_entries_precision_sweep() {
    for r in 1usize..=40 {
        let mut normalized = rand_nonzero_vec(r, 12_100 + 3 * r as u64);
        normalized[r - 1] |= 1 << 63;
        run_all_precision_rcps(&normalized, r);

        let mut unnormalized = rand_nonzero_vec(r, 12_500 + 3 * r as u64);
        unnormalized[r - 1] = (unnormalized[r - 1] & 0xff) | 1;
        run_all_precision_rcps(&unnormalized, r);
    }
}

#[test]
fn test_explicit_reciprocal_entries_skewed_and_schedule_edges() {
    let cases = [
        (200usize, 8usize),
        (8, 200),
        (1, 10),
        (3, 129),
        (96, 97),
        (7, 8),
        (8, 9),
        (15, 17),
        (31, 33),
    ];
    for (idx, &(d_len, r)) in cases.iter().enumerate() {
        let d = rand_nonzero_vec(d_len, 12_200 + 11 * idx as u64);
        run_all_precision_rcps(&d, r);
    }
}

#[test]
fn test_explicit_reciprocal_entries_literals_and_boundaries() {
    run_all_precision_rcps(&[1], 3);
    run_all_precision_rcps(&[2], 3);
    run_all_precision_rcps(&[3], 3);
    run_all_precision_rcps(&[u64::MAX], 3);
    run_all_precision_rcps(&[1, 1], 2);
    run_all_precision_rcps(&[1, 1], 3);
    run_all_precision_rcps(&[0, 1 << 37], 3);

    for r in [1usize, 2, 8] {
        run_all_precision_rcps(&[0, 0, 1], r);
    }
    run_all_precision_rcps(&[1, 0, 1], 2);

    // Regression around the normalization/error-accumulation boundary. The
    // Newton entry may fall back, but must retain the precision contract.
    for r in [20usize, 21] {
        run_all_precision_rcps(&[1, 0, 0, 0, 0, 1], r);
    }
}

#[test]
fn test_reciprocal_dispatch_precision_hierarchy() {
    for cutoff in [k::div::dyn_rcp_knuth_nr(), k::div::static_rcp_knuth_nr()] {
        assert_eq!(rcp_alg_dispatch(cutoff, cutoff), RcpAlg::Knuth);
        assert_eq!(rcp_alg_dispatch(cutoff + 1, cutoff), RcpAlg::NR);
    }
}

#[test]
fn test_explicit_reciprocal_entry_contract_errors() {
    let mut empty = [];
    knuth_rcp_dyn(&[], &mut empty);
    nr_rcp_dyn(&[], &mut empty);
    rcp_dyn(&[], &mut empty);
    knuth_rcp_static::<1>(&[], &mut empty);
    nr_rcp_static::<1>(&[], &mut empty);
    rcp_static::<1>(&[], &mut empty);

    let algorithms: [fn(&[u64], &mut [u64]); 6] = [
        knuth_rcp_dyn,
        nr_rcp_dyn,
        rcp_dyn,
        knuth_rcp_static::<1>,
        nr_rcp_static::<1>,
        rcp_static::<1>,
    ];
    for d in [&[][..], &[0][..], &[1, 0][..]] {
        for algorithm in algorithms {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut out = [0u64; 1];
                algorithm(d, &mut out);
            }));
            assert!(result.is_err(), "invalid divisor {d:?} was accepted");
        }
    }

    let static_algorithms: [fn(&[u64], &mut [u64]); 3] =
        [knuth_rcp_static::<1>, nr_rcp_static::<1>, rcp_static::<1>];
    for algorithm in static_algorithms {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut oversized = [0u64; 2];
            algorithm(&[3], &mut oversized);
        }));
        assert!(result.is_err(), "static reciprocal exceeded its capacity");
        // Static entries take divisors of at most N limbs, even when only
        // their top window would be read.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut out = [0u64; 1];
            algorithm(&[3, 3], &mut out);
        }));
        assert!(result.is_err(), "static reciprocal accepted a divisor past N");
    }
}

#[test]
fn test_nr_rcp_two_guard_limb_regression() {
    let r = 512usize;
    let mut state = (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 2;
    let mut d = Vec::with_capacity(r + 1);
    for _ in 0..=r {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        d.push(state);
    }
    d[r] = 1;

    let exact = exact_rcp_reference(&d, r);
    let mut dynamic = vec![0u64; r];
    nr_rcp_dyn(&d, &mut dynamic);
    assert_rcp_precision("dynamic NR guard limbs", &dynamic, &exact);

    let mut static_rcp = vec![0u64; r];
    nr_rcp_static::<520>(&d, &mut static_rcp);
    assert_rcp_precision("static NR guard limbs", &static_rcp, &exact);
}

#[test]
fn test_nr_rcp_static_tight_fit() {
    // With r == N the guard limbs and the last divisor window pass N; NR still
    // completes, with the window's zero limbs past N dropped by mid_mul_static.
    fn run<const N: usize>(seed: u64) {
        let d = rand_nonzero_vec(N, seed);
        let mut rcp = vec![0u64; N];
        nr_rcp_static::<N>(&d, &mut rcp);
        let exact = exact_rcp_reference(&d, N);
        assert_rcp_precision(&format!("nr_rcp_static tight N={N}"), &rcp, &exact);
    }
    run::<2048>(12_300);
    run::<2043>(12_400);
}

#[test]
fn test_static_reciprocal_entry_matrix_precision() {
    fn run<const N: usize>(name: &str, d_len: usize, r_len: usize, seed: u64) {
        assert!(r_len <= N);
        let d = rand_nonzero_vec(d_len, seed);
        let original_d = d.clone();
        let algorithms: [(&str, fn(&[u64], &mut [u64])); 3] = [
            ("Knuth", knuth_rcp_static::<N>),
            ("Newton-Raphson", nr_rcp_static::<N>),
            ("static dispatcher", rcp_static::<N>),
        ];

        let exact = exact_rcp_reference(&d, r_len);
        for (algorithm_name, algorithm) in algorithms {
            let mut rcp = vec![0xa5a5_a5a5_a5a5_a5a5; r_len];
            algorithm(&d, &mut rcp);
            assert_eq!(d, original_d, "{name}: {algorithm_name} changed d");
            assert_rcp_precision(&format!("{name}: {algorithm_name}"), &rcp, &exact);
        }
    }

    // The refinement bands fit N, so static NR can complete without its
    // capacity fallback.
    run::<256>("actual NR", 128, 96, 12_500);
    // Strongly skewed precision exercises the caller-materialized divisor padding.
    run::<256>("skewed", 12, 129, 12_600);
    // d + r exceeds N, but reciprocal wrappers only need the precision window.
    run::<128>("precision window capacity", 96, 120, 12_700);
    // Only the divisor's top R + 1 limbs reach the result.
    run::<16>("divisor past the precision window", 16, 8, 12_800);
}

#[test]
fn test_reciprocal_entries_ignore_discarded_low_divisor_limbs() {
    const R: usize = 8;
    let mut top = rand_nonzero_vec(R + 1, 12_900);
    top[R] |= 1 << 63;
    let mut long = rand_nonzero_vec(37, 12_901);
    long.extend_from_slice(&top);

    let algorithms: [(&str, fn(&[u64], &mut [u64])); 6] = [
        ("dynamic Knuth", knuth_rcp_dyn),
        ("dynamic Newton-Raphson", nr_rcp_dyn),
        ("dynamic dispatcher", rcp_dyn),
        ("static Knuth", knuth_rcp_static::<64>),
        ("static Newton-Raphson", nr_rcp_static::<64>),
        ("static dispatcher", rcp_static::<64>),
    ];

    for (name, algorithm) in algorithms {
        let mut short_result = [0u64; R];
        let mut long_result = [0u64; R];
        algorithm(&top, &mut short_result);
        algorithm(&long, &mut long_result);
        assert_eq!(
            short_result, long_result,
            "{name} used discarded divisor limbs"
        );
    }
}

#[test]
fn test_static_reciprocal_tiny_capacities() {
    let mut one = [0u64; 1];
    nr_rcp_static::<1>(&[3], &mut one);
    assert_rcp_precision("static N=1", &one, &exact_rcp_reference(&[3], 1));

    let mut two = [0u64; 2];
    nr_rcp_static::<2>(&[3], &mut two);
    let exact = exact_rcp_reference(&[3], 2);
    assert_rcp_precision("static N=2 NR", &two, &exact);

    let mut knuth = [0u64; 2];
    knuth_rcp_static::<2>(&[3], &mut knuth);
    assert_rcp_precision("static N=2 Knuth", &knuth, &exact);
}

#[test]
fn test_sliding_reciprocal_seed_matches_power_numerator() {
    for d_len in [2, 3, 7, 16, 65] {
        for r_len in [1, 2, 3, 8, 17, 64] {
            for pattern in 0..4 {
                let mut d = rand_nonzero_vec(d_len, 19_000 + pattern);
                d[d_len - 1] |= 1 << 63;
                if pattern == 0 {
                    d.fill(0);
                    d[d_len - 1] = 1 << 63;
                } else if pattern == 1 {
                    d.fill(u64::MAX);
                }
                let mut power = vec![0; d_len + r_len];
                *power.last_mut().unwrap() = 1;
                let mut exact = vec![0; r_len + 1];
                assert_eq!(div_buf_of(&mut power, &mut 0, &d, &mut exact), 0);
                assert_eq!(exact.pop(), Some(0));
                assert!(!inc_buf(&mut exact));

                let mut sliding = vec![u64::MAX; r_len];
                let mut win = vec![u64::MAX; d_len];
                knuth_rcp_normalized(&d, &mut sliding, &mut win, 1);
                assert!(!inc_buf(&mut sliding));
                assert_eq!(sliding, exact, "d={d_len} r={r_len} pattern={pattern}");
            }
        }
    }
}

#[test]
fn test_static_division_capacity_uses_factored_operands() {
    // Physical inputs exceed N; removing the exact B^64 divisor factor leaves
    // a 32-limb numerator and a 16-limb divisor, including supported NR shapes.
    let mut d = vec![0; 64];
    d.extend(rand_nonzero_vec(16, 20_200));
    let n = rand_nonzero_vec(96, 20_201);
    exercise_body_overflow_contract::<32>("tight factored operands", &n, &d);
}

// rcp_dyn/rcp_static used to break the error contract from R = 727 upward
// (R % 8 == 7 first, then runs past ~1450): hi_mul_dyn/hi_mul_static cut
// operands to exactly out.len() limbs, injecting ~B ulps into an untruncated
// Newton step, and each later step squared it. The old sweeps stopped at R = 200.
#[test]
fn test_nr_reciprocal_high_precision_regressions() {
    let precisions = (720usize..=760).chain([1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 2047]);
    for r in precisions {
        let mut normalized = rand_nonzero_vec(r, 13_000 + r as u64);
        normalized[r - 1] |= 1 << 63;
        let exact = exact_rcp_reference(&normalized, r);
        let mut rcp = vec![0u64; r];
        nr_rcp_dyn(&normalized, &mut rcp);
        assert_rcp_precision(&format!("newton-raphson r={r}"), &rcp, &exact);
        rcp_dyn(&normalized, &mut rcp);
        assert_rcp_precision(&format!("dispatcher r={r}"), &rcp, &exact);
    }

    // Unnormalized and longer-than-precision divisors take the shifted path
    // and the high-window trim.
    for &(d_len, r) in &[(727usize, 727usize), (1600, 1455), (3000, 743), (5, 1455)] {
        let mut d = rand_nonzero_vec(d_len, 13_500 + (d_len * 7 + r) as u64);
        d[d_len - 1] = (d[d_len - 1] >> 37) | 1;
        let exact = exact_rcp_reference(&d, r);
        let mut rcp = vec![0u64; r];
        nr_rcp_dyn(&d, &mut rcp);
        assert_rcp_precision(&format!("newton-raphson d={d_len} r={r}"), &rcp, &exact);
    }
}

#[test]
fn test_nr_reciprocal_static_high_precision_regressions() {
    for r in [727usize, 735, 743, 1455] {
        let mut d = rand_nonzero_vec(r, 14_000 + r as u64);
        d[r - 1] |= 1 << 63;
        let exact = exact_rcp_reference(&d, r);
        let mut rcp = vec![0u64; r];
        nr_rcp_static::<2048>(&d, &mut rcp);
        assert_rcp_precision(&format!("static newton-raphson r={r}"), &rcp, &exact);
        rcp_static::<2048>(&d, &mut rcp);
        assert_rcp_precision(&format!("static dispatcher r={r}"), &rcp, &exact);
    }
}

// Karp-Markstein's y can round up across its cut when the true quotient has a run of all-ones limbs
// just below it (or d's top limbs are all ones). Then err > 0, and the floor of y - c is y - ceil(c).
#[test]
fn test_nr_quotient_floor_when_estimate_rounds_up() {
    use crate::utils::mul::mul_buf;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x4e52_464c_4f4f52);
    for &(dl, ql) in &[(128usize, 128usize), (200, 200), (256, 256), (400, 400), (512, 512), (1024, 512), (2000, 300)] {
        let qlog = ql + 1;
        let l = qlog - (qlog / 2 + 1);
        for _ in 0..16 {
            let mut d: Vec<u64> = (0..dl).map(|_| rng.gen()).collect();
            d[dl - 1] |= 1 << 63;
            let mut q: Vec<u64> = (0..ql).map(|_| rng.gen()).collect();
            q[ql - 1] = (q[ql - 1] >> 1) | (1 << 62);
            q[l - 4..l].fill(u64::MAX);
            let mut r: Vec<u64> = (0..dl).map(|_| rng.gen()).collect();
            r[dl - 1] = rng.gen::<u64>() >> 2;
            let mut n = vec![0u64; dl + ql];
            mul_buf(&q, &d, &mut n);
            add_buf(&mut n, &r);
            while n.last() == Some(&0) {
                n.pop();
            }
            let mut got = vec![0u64; n.len() - dl + 1];
            div_dyn(&n, &d, &mut got);
            assert_eq!(&got[..ql], &q[..], "div_dyn {dl}x{ql}");
            let req = division_preflight(&n, &d, &mut got).unwrap();
            div_prepared_dyn(&n, &d, &mut got, req, DivAlg::NR);
            assert_eq!(&got[..ql], &q[..], "forced NR {dl}x{ql}");
            if dl + ql <= 1000 {
                div_static::<1100>(&n, &d, &mut got);
                assert_eq!(&got[..ql], &q[..], "div_static {dl}x{ql}");
            }
        }
    }
    // An all-ones divisor with a run of ones on top of n: x = 1.000..., so y is n's own limbs.
    for &(dl, nl) in &[(128usize, 256usize), (300, 620), (884, 1729)] {
        for k in (dl..nl - nl / 8).step_by(7) {
            let d = vec![u64::MAX; dl];
            let mut n: Vec<u64> = (0..nl).map(|_| rng.gen()).collect();
            n[k..].fill(u64::MAX);
            let qlen = nl - dl + 1;
            let (mut qk, mut rk, mut qn) = (vec![0u64; qlen], n.clone(), vec![0u64; qlen]);
            let req = division_preflight(&n, &d, &mut qk).unwrap();
            div_rem_prepared_dyn(&mut rk, &d, &mut qk, req, DivAlg::Knuth);
            let req = division_preflight(&n, &d, &mut qn).unwrap();
            div_prepared_dyn(&n, &d, &mut qn, req, DivAlg::NR);
            assert_eq!(qn, qk, "all-ones d={dl} n={nl} ones from {k}");
        }
    }
}

// n = (d m) B^k + small, with the quotient's low part all zero, leaves |c| below x_norm's truncation:
// the subtraction comes out positive, and negating it put q off by about B^(k - 2). div_rem's window
// accepts that when d has a zero limb in the right place.
#[test]
fn test_nr_rem_scaled_multiple_of_sparse_divisor() {
    use crate::utils::{mul::mul_buf, utils::add_prim};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x4e52_5245_4d);
    for &(dl, ql) in &[(136usize, 140usize), (308, 94), (400, 200), (1283, 41), (800, 800)] {
        let qlog = ql + 1;
        let l = qlog - (qlog / 2 + 1);
        for t in 0..64u64 {
            let mut d: Vec<u64> = (0..dl).map(|_| if rng.gen::<u64>() % 8 == 0 { rng.gen() } else { 0 }).collect();
            d[dl - 1] |= 1 << 63;
            let mut q: Vec<u64> = (0..ql).map(|_| rng.gen()).collect();
            q[ql - 1] |= 1;
            let k = (l + (t % 6) as usize).saturating_sub(2).min(ql - 1);
            q[..k].fill(0);
            let mut n = vec![0u64; dl + ql + 1];
            mul_buf(&q, &d, &mut n[..dl + ql]);
            let delta = t % 4;
            add_prim(&mut n, delta);
            while n.last() == Some(&0) {
                n.pop();
            }
            let qlen = n.len() - dl + 1;
            let (mut got, mut r) = (vec![0u64; qlen], n.clone());
            let req = division_preflight(&n, &d, &mut got).unwrap();
            div_rem_prepared_dyn(&mut r, &d, &mut got, req, DivAlg::NR);
            assert_eq!(&got[..ql], &q[..], "forced NR q {dl}x{ql} t={t}");
            assert_eq!(r[0], delta, "forced NR r {dl}x{ql} t={t}");
            assert!(r[1..].iter().all(|&x| x == 0), "forced NR r {dl}x{ql} t={t}");
            let (mut got, mut r) = (vec![0u64; qlen], n.clone());
            div_rem_dyn(&mut r, &d, &mut got);
            assert_eq!(&got[..ql], &q[..], "div_rem_dyn q {dl}x{ql} t={t}");
            assert_eq!(r[0], delta, "div_rem_dyn r {dl}x{ql} t={t}");
            let mut got = vec![0u64; qlen];
            div_dyn(&n, &d, &mut got);
            assert_eq!(&got[..ql], &q[..], "div_dyn {dl}x{ql} t={t}");
        }
    }
}

// div_3_2 multiplies each quotient piece at its significant width and skips a zero piece. A 2/1
// block whose quotient has zero upper limbs (BZ's top block on a padded numerator) must still
// give the exact quotient and remainder, at every count of significant quotient limbs: none,
// one, a piece boundary, one either side of it, and all of them.
#[test]
fn test_div_2_1_short_quotients() {
    use crate::utils::mul::mul_buf;
    let leaf = k::div::bz();
    for d_len in [leaf + 1, leaf + 2, 2 * leaf + 1, 4 * leaf + 3, 601.max(leaf + 5)] {
        let mut d = rand_nonzero_vec(d_len, 71_000 + d_len as u64);
        d[d_len - 1] |= 1 << 63;
        let half = d_len / 2;
        for top in [0, 1, 2, half / 2, half - 1, half, half + 1, d_len - 1, d_len] {
            for small_rem in [false, true] {
                // n = q0 d + r0 with q0 of `top` limbs and r0 < d: built by schoolbook.
                let q0: Vec<u64> = if top == 0 { Vec::new() } else { rand_nonzero_vec(top, 72_000 + (d_len * 31 + top) as u64) };
                let mut r0 = if small_rem { vec![0u64; d_len] } else { rand_nonzero_vec(d_len, 73_000 + (d_len * 17 + top) as u64) };
                if small_rem {
                    r0[0] = 5;
                } else {
                    r0[d_len - 1] &= (1 << 63) - 1; // below d, whose top bit is set
                }
                let mut n = vec![0u64; 2 * d_len];
                if top > 0 {
                    let carry = mul_buf(&d, &q0, &mut n[..d_len + top - 1]);
                    n[d_len + top - 1] = carry;
                }
                assert!(!add_buf(&mut n, &r0), "n stays within 2 d_len limbs");
                let (mut q, mut scratch) = (vec![u64::MAX; d_len], vec![0u64; 2 * d_len]);
                let mut mul = |a: &mut [u64], b: &[u64], o: &mut [u64]| {
                    mul_dyn(a, b, o);
                };
                div_2_1(&mut n, &d, &mut q, &mut scratch, &mut mul);
                assert_eq!(&q[..top], &q0[..], "quotient, d {d_len} top {top} small_rem {small_rem}");
                assert!(q[top..].iter().all(|&x| x == 0), "zero upper quotient, d {d_len} top {top}");
                assert_eq!(&n[..d_len], &r0[..], "remainder, d {d_len} top {top} small_rem {small_rem}");
            }
        }
    }
}

// div_dyn's and div_rem_dyn's dispatch compares fitted equations (DivEq), at explicit values here:
// each equation by hand, which shapes each algorithm can take, and the least prediction winning.
#[test]
fn test_dyn_div_alg_compares_equations() {
    use crate::utils::mul::MulEq;
    let kara = k::mul::kara();
    let mul = MulEq { school: [0.35, 14.0], kara: [1.5, 0.02], chunk: [1.6, 0.03], fft: 0.6 };
    let eq = DivEq {
        knuth: [0.4, 9.5, 0.4, 20.0],
        nr: [1.8, 1.2, 1.0, 0.9, 4.0],
        nr_rem: [1.7, 1.1, 0.95, 0.85, 1.0, 4.0, 1.5],
        bz: [1.04, 0.98],
        bz_leaf: 100,
        fft_mid: 109,
        partial_mul: 312,
    };
    let close = |a: f64, b: f64| (a / b - 1.0).abs() < 1e-12;
    let m = |l: usize, s: usize| mul.mul_ns(l, s, kara);

    // The equations by hand.
    assert!(close(eq.knuth_ns(30, 50), 0.4 * 1500.0 + 9.5 * 30.0 + 0.4 * 50.0 + 20.0));

    // NR's products by tier. A schoolbook middle product with an n-limb short side is
    // schoolbook's n x n time, an FFT one the FFT's time at its own transform size; a schoolbook
    // high product into `out` limbs is half a schoolbook product of that window, and a guarded
    // one the full product of its operands' tops.
    let mid_school = |n: usize| mul.school_ns(n, n);
    let mid_fft = |n: usize| mul.fft_ns_at(crate::utils::mul::fft_mid_size(n));
    let hi_school = |out: usize| 0.5 * 0.35 * (out * out) as f64 + 14.0;
    let same = |got: [f64; 4], want: [f64; 4]| got.iter().zip(&want).all(|(g, w)| (*w == 0.0 && *g == 0.0) || close(*g, *w));
    // q = 8: h = 5, l = 3. The chain to h + 2 = 7 limbs steps 1 -> 2 -> 4 -> 7, each from p with a
    // middle product of short side p + 1 and a high product into p limbs; then the numerator's
    // top into h + 2 = 7 limbs, the error band (short side h + 1 = 6) and the correction into
    // l + 2 = 5 limbs. All schoolbook at these sizes.
    let want = [
        mid_school(2) + mid_school(3) + mid_school(5) + mid_school(6),
        0.0,
        hi_school(1) + hi_school(2) + hi_school(4) + hi_school(7) + hi_school(5),
        0.0,
    ];
    assert!(same(eq.nr_parts(8, &mul, kara), want), "{:?} != {want:?}", eq.nr_parts(8, &mul, kara));
    // q = 700: h = 351, l = 349. The chain to 353 steps 1, 2, 3, 6, 12, 23, 45, 89, 177 -> 353.
    // Middle products from 109 limbs take the FFT (178 and the band's 352); the two top high
    // products pass 312 limbs and are guarded, at their operands' tops of `out` limbs.
    let want = [
        [2, 3, 4, 7, 13, 24, 46, 90].iter().map(|&n| mid_school(n)).sum(),
        mid_fft(178) + mid_fft(352),
        [1, 2, 3, 6, 12, 23, 45, 89, 177].iter().map(|&o| hi_school(o)).sum(),
        m(353, 353) + m(351, 351),
    ];
    assert!(same(eq.nr_parts(700, &mul, kara), want), "{:?} != {want:?}", eq.nr_parts(700, &mul, kara));
    // The tier boundaries are the dispatchers' own: with the FFT middle product from 2 limbs and
    // no guarded tier, q = 8's middle products all move to the FFT tier.
    let all_fft = DivEq { fft_mid: 2, ..eq };
    let p = all_fft.nr_parts(8, &mul, kara);
    assert!(p[0] == 0.0 && close(p[1], mid_fft(2) + mid_fft(3) + mid_fft(5) + mid_fft(6)));

    let dot = |w: &[f64], q: usize| -> f64 { eq.nr_parts(q, &mul, kara).iter().zip(w).map(|(a, b)| a * b).sum() };
    // Quotient only, NR's time has no term in d: it reads the divisor's top window.
    assert!(close(eq.nr_ns(300, 500, false, &mul, kara), dot(&eq.nr[..4], 300) + 4.0 * 300.0));
    assert_eq!(eq.nr_ns(300, 500, false, &mul, kara), eq.nr_ns(300, 5000, false, &mul, kara));
    // With the remainder: nr_rem_finish's product of d by the quotient's low min(q - 1, d + 1) limbs.
    assert!(close(eq.nr_ns(300, 500, true, &mul, kara), dot(&eq.nr_rem[..4], 300) + m(500, 299) + 4.0 * 300.0 + 1.5 * 500.0));
    assert!(close(eq.nr_ns(900, 500, true, &mul, kara), dot(&eq.nr_rem[..4], 900) + m(500, 501) + 4.0 * 900.0 + 1.5 * 500.0));
    assert!(eq.nr_base_ns(300, 500, false) < eq.nr_ns(300, 500, false, &mul, kara));

    // A 2/1 block, as (products, leaves): at or under the leaf it is Knuth alone; past it each
    // level halves the divisor and doubles the count of products and leaves (300 -> 150 -> 75).
    let (p, l) = eq.bz_block_parts(100, &mul, kara);
    assert!(p == 0.0 && close(l, eq.knuth_ns(100, 100)));
    let (p, l) = eq.bz_block_parts(300, &mul, kara);
    assert!(close(p, 2.0 * m(150, 150) + 4.0 * m(75, 75)) && close(l, 4.0 * eq.knuth_ns(75, 75)));
    assert!(close(eq.bz_leaves_ns(300), l));
    // A top block on a padded numerator: its leaves in full, and each level's products by the
    // quotient's significant pieces. A full quotient is a whole block. With 75 limbs under a
    // 300-limb divisor: at the 150-limb level one piece of 75 limbs (150 x 75), none whole; at
    // the 75-limb level one whole piece (75 x 75) and three zero ones. With 200 limbs: one whole
    // 150-limb piece and a 150 x 50; then two whole 75-limb pieces and a 75 x 50.
    assert!(close(eq.bz_padded_top_ns(300, 300, &mul, kara), p + l));
    assert!(close(eq.bz_padded_top_ns(75, 300, &mul, kara), m(150, 75) + m(75, 75) + l));
    assert!(close(eq.bz_padded_top_ns(200, 300, &mul, kara), m(150, 150) + m(150, 50) + 2.0 * m(75, 75) + m(75, 50) + l));
    assert!(close(eq.bz_padded_top_ns(1, 300, &mul, kara), m(150, 1) + m(75, 1) + l));
    // BZ pads its top block where that is predicted cheaper than Knuth, never at the leaf.
    for (d, top) in [(300, 300), (300, 150), (300, 20), (1000, 1000), (1000, 400), (1000, 30)] {
        assert_eq!(eq.bz_top_padded(d, top, &mul, kara), 1.04 * eq.bz_padded_top_ns(top, d, &mul, kara) < 0.98 * eq.knuth_ns(top, d), "d {d} top {top}");
    }
    assert!(eq.bz_top_padded(1000, 1000, &mul, kara) && !eq.bz_top_padded(1000, 30, &mul, kara));
    assert!(!eq.bz_top_padded(100, 100, &mul, kara));
    // The plan: whole blocks and the top block either way, each at its multiple; None where BZ would be Knuth alone
    // (at the leaf, or a Knuth top over no whole block).
    for (q, d) in [(1000, 300), (601, 300), (300, 300), (250, 300), (20, 300), (5000, 1000), (700, 1000), (50, 100)] {
        let (blocks, top) = ((q - 1) / d, (q - 1) % d + 1);
        let (p, l) = eq.bz_block_parts(d, &mul, kara);
        let padded = eq.bz_top_padded(d, top, &mul, kara);
        let top_ns = if padded { 1.04 * eq.bz_padded_top_ns(top, d, &mul, kara) } else { 0.98 * eq.knuth_ns(top, d) };
        let want = (d > 100 && (blocks > 0 || padded)).then(|| 1.04 * blocks as f64 * (p + l) + top_ns);
        match (eq.bz_plan(q, d, &mul, kara), want) {
            (Some(got), Some(want)) => assert!(close(got, want), "q {q} d {d}: {got} != {want}"),
            (None, None) => {}
            other => panic!("q {q} d {d}: {other:?}"),
        }
    }
    assert!(eq.bz_plan(20, 300, &mul, kara).is_none() && eq.bz_plan(300, 300, &mul, kara).is_some());

    // The choice: Knuth where nothing else can run (a quotient under NR's 8 limbs at the leaf).
    for (q, d) in [(1, 5000), (7, 100), (7, 3), (5, 100)] {
        assert_eq!(dyn_div_alg_at(q, d, false, &eq, &mul, kara), DivAlg::Knuth, "q {q} d {d}");
    }
    // Otherwise the least prediction among those that can run, at every shape.
    for rem in [false, true] {
        for (q, d) in [(8, 8), (9, 40), (40, 64), (7, 5000), (30, 2000), (200, 150), (150, 150), (120, 150), (2000, 300), (1500, 1400), (4000, 90)] {
            let mut want = (DivAlg::Knuth, eq.knuth_ns(q, d));
            if q >= 8 && eq.nr_ns(q, d, rem, &mul, kara) < want.1 {
                want = (DivAlg::NR, eq.nr_ns(q, d, rem, &mul, kara));
            }
            if eq.bz_plan(q, d, &mul, kara).is_some_and(|t| t < want.1) {
                want.0 = DivAlg::BZ;
            }
            assert_eq!(dyn_div_alg_at(q, d, rem, &eq, &mul, kara), want.0, "q {q} d {d} rem {rem}");
        }
    }
    // A free NR takes every shape it can enter, and costly ones leave Knuth.
    let free_nr = DivEq { nr: [0.0; 5], nr_rem: [0.0; 7], ..eq };
    let dear = DivEq { nr: [0.0, 0.0, 0.0, 0.0, 1e18], nr_rem: [0.0, 0.0, 0.0, 0.0, 0.0, 1e18, 0.0], bz: [1e18, 1e18], ..eq };
    for (q, d) in [(8, 4), (50, 50), (3000, 200), (200, 3000)] {
        assert_eq!(dyn_div_alg_at(q, d, false, &free_nr, &mul, kara), DivAlg::NR);
        assert_eq!(dyn_div_alg_at(q, d, true, &free_nr, &mul, kara), DivAlg::NR);
        assert_eq!(dyn_div_alg_at(q, d, false, &dear, &mul, kara), DivAlg::Knuth);
    }
    // Free blocks (a zero multiple, with Knuth tops at full price) pad every top block, so BZ
    // takes every shape past the leaf, a quotient shorter than the divisor included; at or
    // under the leaf it is never chosen.
    let free_bz = DivEq { bz: [0.0, 1.0], ..eq };
    for (q, d) in [(3000, 200), (200, 200), (150, 200), (9, 3000)] {
        assert_eq!(dyn_div_alg_at(q, d, false, &free_bz, &mul, kara), DivAlg::BZ, "q {q} d {d}");
    }
    assert_ne!(dyn_div_alg_at(3000, 90, false, &free_bz, &mul, kara), DivAlg::BZ);
}

// A small division reads a measured table in place of the equations (`div.dyn_small`,
// `div.dyn_rem_small`, change points over (d, q)): a lookup at a table made by hand, a pick that
// can't take the shape handed on to the equations, and the default tables, which keep Knuth at
// narrow divisors as the guards before them did.
#[test]
fn test_dyn_div_small_table() {
    use crate::tuning::table::{row_index, Table2};
    // Divisors of 8 to 39 limbs: Knuth for quotients to 19 limbs, NR to 99, the equations from
    // 100. From 40: BZ, which a divisor under the leaf can't take. From 60: the equations.
    let t: &'static [u32] = Box::leak(Box::new([8, 2, 0, 8, 20, 2, 8, 100, 3, 40, 2, 1, 60, 1, 3]));
    let table = Table2 { t, rows: Box::leak(row_index(t).into_boxed_slice()) };
    let leaf = 99;
    assert_eq!(dyn_div_small(&table, 5, 8, leaf), Some(DivAlg::Knuth));
    assert_eq!(dyn_div_small(&table, 19, 39, leaf), Some(DivAlg::Knuth));
    assert_eq!(dyn_div_small(&table, 20, 8, leaf), Some(DivAlg::NR));
    assert_eq!(dyn_div_small(&table, 99, 30, leaf), Some(DivAlg::NR));
    assert_eq!(dyn_div_small(&table, 100, 8, leaf), None, "past the row's extent");
    assert_eq!(dyn_div_small(&table, 30, 45, leaf), None, "BZ under its leaf is handed on");
    assert_eq!(dyn_div_small(&table, 30, 45, 40), Some(DivAlg::BZ));
    assert_eq!(dyn_div_small(&table, 30, 70, leaf), None);
    assert_eq!(dyn_div_small(&table, 30, 5000, leaf), None, "a divisor past the last row reads it");
    assert_eq!(dyn_div_small(&table, 9, 3, leaf), Some(DivAlg::Knuth), "below the first row: its first pick");
    // What can take a shape: NR from a quotient of 8 limbs, BZ past its leaf, Knuth always.
    assert!(dyn_div_can(DivAlg::Knuth, 1, 1, leaf) && !dyn_div_can(DivAlg::NR, 7, 50, leaf) && dyn_div_can(DivAlg::NR, 8, 50, leaf));
    assert!(!dyn_div_can(DivAlg::BZ, 50, 99, leaf) && dyn_div_can(DivAlg::BZ, 50, 100, leaf) && !dyn_div_can(DivAlg::BZ, 1, 100, leaf));
    assert!(!dyn_div_has_choice(7, 99, leaf) && dyn_div_has_choice(8, 4, leaf) && dyn_div_has_choice(2, 100, leaf) && !dyn_div_has_choice(1, 5000, leaf));
    // The defaults: Knuth at divisors to 32 limbs (64 with the remainder), the equations past.
    use crate::tuning::schema_types::DefaultValue;
    for (path, guard) in [("div.dyn_small", 32usize), ("div.dyn_rem_small", 64)] {
        let DefaultValue::Table(t) = crate::tuning::knob(path).unwrap().default else { panic!("{path} is a table") };
        let table = Table2 { t, rows: Box::leak(row_index(t).into_boxed_slice()) };
        for q in [8usize, 100, 20000] {
            assert_eq!(dyn_div_small(&table, q, guard, leaf), Some(DivAlg::Knuth), "{path}");
            assert_eq!(dyn_div_small(&table, q, guard + 1, leaf), None, "{path}");
        }
    }
}

// The live dispatcher: Knuth where nothing else can run, the table's algorithm where it names
// one, and the public entries exact at divisors across the table and wherever the equations
// send a shape.
#[test]
fn test_dyn_div_table_and_entries() {
    let leaf = k::div::bz();
    for (rem, table) in [(false, k::div::dyn_small()), (true, k::div::dyn_rem_small())] {
        for d_len in [1usize, 2, 20, 33, 40, 70, 110] {
            for q_len in [1usize, 7, 8, 40, 300] {
                let (n, d) = (rand_nonzero_vec(q_len + d_len, 61_000 + (q_len * 7 + d_len) as u64), rand_nonzero_vec(d_len, 62_000 + d_len as u64));
                let mut q = vec![0u64; q_len];
                let r = division_preflight(&n, &d, &mut q).expect("a nonempty request");
                let alg = dyn_div_alg(&r, rem);
                if !dyn_div_has_choice(q_len + 1, d_len, leaf) {
                    assert_eq!(alg, DivAlg::Knuth, "d {d_len} q {q_len} rem {rem}");
                } else if let Some(want) = dyn_div_small(table, q_len + 1, d_len, leaf) {
                    assert_eq!(alg, want, "d {d_len} q {q_len} rem {rem}");
                }
                // Whatever it picks, the entry agrees with forced Knuth.
                let (mut want_q, mut want_n) = (vec![0u64; q_len], n.clone());
                let want_top = div_rem_prepared_dyn(&mut want_n, &d, &mut want_q, r, DivAlg::Knuth);
                let mut got_n = n.clone();
                let top = if rem { div_rem_dyn(&mut got_n, &d, &mut q) } else { div_dyn(&n, &d, &mut q) };
                assert_eq!((top, &q), (want_top, &want_q), "d {d_len} q {q_len} rem {rem}");
                if rem {
                    assert_eq!(got_n, want_n, "remainder, d {d_len} q {q_len}");
                }
            }
        }
    }
}

// The div section's random-knob check (test_tuning.rs): every public division entry under the live
// knobs (so Knuth, BZ and NR at whatever widths they send them), against q·d + r = n with r < d;
// high quotients against the full quotient's top digits; reciprocals within their error bound.
#[cfg(feature = "tune")]
pub(super) fn random_knob_check(rng: &mut rand::rngs::StdRng) {
    use rand::Rng;
    for _ in 0..3 {
        let dl = rng.gen_range(1..=300);
        let nl = dl + rng.gen_range(0..=300);
        let (n, d) = (rand_nonzero_vec(nl, rng.gen()), rand_nonzero_vec(dl, rng.gen()));
        let m = nl - dl;
        let what = format!("{nl}/{dl}");
        let mut q = vec![0; m + 1];
        div_dyn(&n, &d, &mut q);
        let (mut r, mut q2) = (n.clone(), vec![0; m + 1]);
        div_rem_dyn(&mut r, &d, &mut q2);
        assert_eq!(q, q2, "div_dyn and div_rem_dyn quotients differ at {what}");
        assert_divmod_algorithm(&format!("div_rem_dyn {what}"), &n, &d, &q2, &r);
        let mut qs = vec![0; m + 1];
        div_static::<1024>(&n, &d, &mut qs);
        assert_eq!(qs, q, "div_static at {what}");
        let (mut r, mut q2) = (n.clone(), vec![0; m + 1]);
        div_rem_static::<1024>(&mut r, &d, &mut q2);
        assert_divmod_algorithm(&format!("div_rem_static {what}"), &n, &d, &q2, &r);
        if m >= 2 {
            let kk = rng.gen_range(1..m);
            let mut hq = vec![0; kk];
            let of = hi_div_dyn(&n, &d, &mut hq);
            assert_eq!((&hq[..], of), (&q[m - kk..m], q[m]), "hi_div_dyn {kk} limbs at {what}");
            hq.fill(0);
            let of = hi_div_static::<1024>(&n, &d, &mut hq);
            assert_eq!((&hq[..], of), (&q[m - kk..m], q[m]), "hi_div_static {kk} limbs at {what}");
        }
    }
    let (dl, rl) = (rng.gen_range(1..=200), rng.gen_range(1..=300));
    let d = rand_nonzero_vec(dl, rng.gen());
    let exact = exact_rcp_reference(&d, rl);
    let mut x = vec![0; rl];
    rcp_dyn(&d, &mut x);
    assert_rcp_precision(&format!("rcp_dyn {dl} -> {rl}"), &x, &exact);
    x.fill(0);
    rcp_static::<1024>(&d, &mut x);
    assert_rcp_precision(&format!("rcp_static {dl} -> {rl}"), &x, &exact);
}
