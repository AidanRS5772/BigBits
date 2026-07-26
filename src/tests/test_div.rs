use super::{rand_nonzero_vec, to_u128, verify_divmod};
use crate::utils::div::*;
use crate::utils::mul::mul_dyn;
use crate::utils::utils::{add_buf, cmp_buf, dec_buf, inc_buf, shl_buf, shr_buf, sub_buf, trim_lz};
use crate::utils::{
    ScratchGuard, BZ_CUTOFF, BZ_TOP_PADDED_COST_SCALE, DYN_RCP_KNUTH_NR_CUTOFF,
    STATIC_RCP_KNUTH_NR_CUTOFF,
};

fn knuth_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        knuth_div_wrapper_dyn(n, d, q, request);
    }
}

fn knuth_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        knuth_div_rem_wrapper_dyn(n, d, q, request);
    }
}

fn knuth_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        knuth_div_wrapper_static::<N>(n, d, q, request);
    }
}

fn knuth_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        knuth_div_rem_wrapper_static::<N>(n, d, q, request);
    }
}

fn bz_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        bz_div_wrapper_dyn(n, d, q, request);
    }
}

fn bz_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        bz_div_rem_wrapper_dyn(n, d, q, request);
    }
}

fn bz_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        bz_div_wrapper_static::<N>(n, d, q, request);
    }
}

fn bz_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        bz_div_rem_wrapper_static::<N>(n, d, q, request);
    }
}

fn nr_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        nr_div_wrapper_dyn(n, d, q, request);
    }
}

fn nr_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        nr_div_rem_wrapper_dyn(n, d, q, request);
    }
}

fn nr_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        nr_div_wrapper_static::<N>(n, d, q, request);
    }
}

fn nr_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        nr_div_rem_wrapper_static::<N>(n, d, q, request);
    }
}

fn knuth_rcp_dyn(d: &[u64], rcp: &mut [u64]) {
    if reciprocal_preflight(d, rcp) {
        knuth_rcp_wrapper_dyn(d, rcp);
    }
}

fn knuth_rcp_static<const N: usize>(d: &[u64], rcp: &mut [u64]) {
    if reciprocal_preflight(d, rcp) {
        knuth_rcp_wrapper_static::<N>(d, rcp);
    }
}

fn nr_rcp_dyn(d: &[u64], rcp: &mut [u64]) {
    if reciprocal_preflight(d, rcp) {
        nr_rcp_wrapper_dyn(d, rcp);
    }
}

fn nr_rcp_static<const N: usize>(d: &[u64], rcp: &mut [u64]) {
    if reciprocal_preflight(d, rcp) {
        nr_rcp_wrapper_static::<N>(d, rcp);
    }
}

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
        let q_len = div_quotient_len(original_n.len(), original_d.len());

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

struct HighDivisionReference {
    q: Vec<u64>,
    residual: Vec<u64>,
    quotient_skip: usize,
    unchanged_prefix: usize,
}

/// Reference the high-quotient contract from a full Knuth quotient, then form
/// the partial residual independently from
///
///     n = (q * d) * B^quotient_skip + residual.
///
/// `q_len` may exceed the full structural quotient width; those excess limbs
/// are high zero padding in the public output.
fn high_division_reference(n: &[u64], d: &[u64], q_len: usize) -> HighDivisionReference {
    let full_q_len = div_quotient_len(n.len(), d.len());
    let used_q_len = q_len.min(full_q_len);
    let quotient_skip = full_q_len - used_q_len;

    let mut full_q = vec![u64::MAX; full_q_len];
    knuth_div_dyn(n, d, &mut full_q);

    let mut q = vec![0; q_len];
    if used_q_len != 0 {
        q[..used_q_len].copy_from_slice(&full_q[quotient_skip..]);
    }

    let mut residual = n.to_vec();
    if used_q_len != 0 {
        let product = super::mul_ref(&q[..used_q_len], d);
        if !product.is_empty() {
            assert!(product.len() <= residual.len() - quotient_skip);
            assert!(
                !sub_buf(&mut residual[quotient_skip..], &product),
                "reference high quotient exceeds the numerator"
            );

            let mut reconstructed = residual.clone();
            assert!(
                !add_buf(&mut reconstructed[quotient_skip..], &product),
                "reference reconstruction overflowed"
            );
            assert_eq!(reconstructed, n, "reference partial-residual identity");
        }
    }

    let divisor_zeros = d.iter().position(|&limb| limb != 0).unwrap();
    HighDivisionReference {
        q,
        residual,
        quotient_skip,
        unchanged_prefix: quotient_skip + divisor_zeros,
    }
}

fn assert_high_division_result(
    name: &str,
    n: &[u64],
    d: &[u64],
    q: &[u64],
    residual: &[u64],
    expected: &HighDivisionReference,
) {
    assert_eq!(q, expected.q, "{name}: high quotient mismatch");
    assert_eq!(
        residual, expected.residual,
        "{name}: partial residual mismatch"
    );
    assert_eq!(
        &residual[..expected.unchanged_prefix],
        &n[..expected.unchanged_prefix],
        "{name}: omitted numerator prefix changed"
    );
    assert!(
        cmp_buf(&residual[expected.quotient_skip..], d).is_lt(),
        "{name}: shifted residual is not below the divisor"
    );

    let divisor_zeros = d.iter().position(|&limb| limb != 0).unwrap();
    assert!(
        cmp_buf(&residual[expected.unchanged_prefix..], &d[divisor_zeros..]).is_lt(),
        "{name}: factored residual window is not reduced"
    );
}

fn exercise_high_division_entry_matrix<const N: usize>(
    case: &str,
    n: &[u64],
    d: &[u64],
    q_lengths: &[usize],
) {
    let dyn_divs: [(&str, fn(&[u64], &[u64], &mut [u64])); 4] = [
        ("Knuth dyn", knuth_div_dyn),
        ("Burnikel-Ziegler dyn", bz_div_dyn),
        ("Newton-Raphson dyn", nr_div_dyn),
        ("dispatched dyn", div_dyn),
    ];
    let dyn_div_rems: [(&str, fn(&mut [u64], &[u64], &mut [u64])); 4] = [
        ("Knuth div-rem dyn", knuth_div_rem_dyn),
        ("Burnikel-Ziegler div-rem dyn", bz_div_rem_dyn),
        ("Newton-Raphson div-rem dyn", nr_div_rem_dyn),
        ("dispatched div-rem dyn", div_rem_dyn),
    ];
    let static_divs: [(&str, fn(&[u64], &[u64], &mut [u64])); 4] = [
        ("Knuth static", knuth_div_static::<N>),
        ("Burnikel-Ziegler static", bz_div_static::<N>),
        ("Newton-Raphson static", nr_div_static::<N>),
        ("dispatched static", div_static::<N>),
    ];
    let static_div_rems: [(&str, fn(&mut [u64], &[u64], &mut [u64])); 4] = [
        ("Knuth div-rem static", knuth_div_rem_static::<N>),
        ("Burnikel-Ziegler div-rem static", bz_div_rem_static::<N>),
        ("Newton-Raphson div-rem static", nr_div_rem_static::<N>),
        ("dispatched div-rem static", div_rem_static::<N>),
    ];

    for &q_len in q_lengths {
        let expected = high_division_reference(n, d, q_len);

        for (algorithm, divide) in dyn_divs {
            let n_input = n.to_vec();
            let d_input = d.to_vec();
            let mut q = vec![0xa5a5_a5a5_a5a5_a5a5; q_len];
            divide(&n_input, &d_input, &mut q);
            assert_eq!(q, expected.q, "{case}, q_len={q_len}, {algorithm}");
            assert_eq!(n_input, n, "{case}, q_len={q_len}, {algorithm}: n changed");
            assert_eq!(d_input, d, "{case}, q_len={q_len}, {algorithm}: d changed");
        }

        for (algorithm, divide) in dyn_div_rems {
            let mut residual = n.to_vec();
            let d_input = d.to_vec();
            let mut q = vec![0xa5a5_a5a5_a5a5_a5a5; q_len];
            divide(&mut residual, &d_input, &mut q);
            assert_high_division_result(
                &format!("{case}, q_len={q_len}, {algorithm}"),
                n,
                d,
                &q,
                &residual,
                &expected,
            );
            assert_eq!(d_input, d, "{case}, q_len={q_len}, {algorithm}: d changed");
        }

        for (algorithm, divide) in static_divs {
            let n_input = n.to_vec();
            let d_input = d.to_vec();
            let mut q = vec![0xa5a5_a5a5_a5a5_a5a5; q_len];
            divide(&n_input, &d_input, &mut q);
            assert_eq!(q, expected.q, "{case}, q_len={q_len}, {algorithm}");
            assert_eq!(n_input, n, "{case}, q_len={q_len}, {algorithm}: n changed");
            assert_eq!(d_input, d, "{case}, q_len={q_len}, {algorithm}: d changed");
        }

        for (algorithm, divide) in static_div_rems {
            let mut residual = n.to_vec();
            let d_input = d.to_vec();
            let mut q = vec![0xa5a5_a5a5_a5a5_a5a5; q_len];
            divide(&mut residual, &d_input, &mut q);
            assert_high_division_result(
                &format!("{case}, q_len={q_len}, {algorithm}"),
                n,
                d,
                &q,
                &residual,
                &expected,
            );
            assert_eq!(d_input, d, "{case}, q_len={q_len}, {algorithm}: d changed");
        }
    }
}

#[test]
fn test_high_quotient_entry_contract_matrix() {
    const N: usize = 128;
    let mut d = rand_nonzero_vec(BZ_CUTOFF + 5, 13_300);
    d[0] = 0;
    d[1] = 0;
    *d.last_mut().unwrap() |= 1 << 63;

    let mut n = rand_nonzero_vec(d.len() + 20, 13_301);
    *n.last_mut().unwrap() = u64::MAX;
    let full_q_len = div_quotient_len(n.len(), d.len());
    assert_eq!(full_q_len, 21);

    let q_lengths = [
        0,
        1,
        8,
        full_q_len / 2,
        full_q_len - 1,
        full_q_len,
        full_q_len + 2,
    ];
    exercise_high_division_entry_matrix::<N>("large factored divisor", &n, &d, &q_lengths);
}

#[test]
fn test_high_quotient_equal_length_smaller_numerator() {
    const N: usize = 8;
    let n = [11, 22, 33, 4];
    let d = [0, 44, 55, 5];
    assert!(cmp_buf(&n, &d).is_lt());
    assert_eq!(div_quotient_len(n.len(), d.len()), 1);

    exercise_high_division_entry_matrix::<N>("equal-length n < d", &n, &d, &[0, 1, 3]);
}

#[test]
fn test_high_quotient_cropped_window_below_divisor() {
    let n = [7, 1, 9];
    let d = [5, 10];
    assert!(cmp_buf(&n, &d).is_gt());
    assert!(cmp_buf(&n[1..], &d).is_lt());

    // The original division is nondegenerate, but its requested top
    // structural quotient limb is zero. The wrapper must pass that cropped
    // window to the rigid algorithm without changing the partial residual.
    exercise_high_division_entry_matrix::<4>("cropped window below divisor", &n, &d, &[1]);

    // Factoring the divisor's exact B^2 term makes both rigid operands fit N,
    // even though the original numerator and divisor exceed it.
    exercise_high_division_entry_matrix::<2>(
        "factored cropped window fits static capacity",
        &[7, 8, 9, 10, 11],
        &[0, 0, 5],
        &[1],
    );
}

#[test]
fn test_high_quotient_boundary_shapes() {
    // Dropping a nonzero low divisor limb would incorrectly turn this top
    // quotient digit from 1 into 2.
    exercise_high_division_entry_matrix::<4>("nonzero low divisor limb", &[0, 0, 2], &[1, 1], &[1]);

    exercise_high_division_entry_matrix::<8>(
        "one-limb divisor",
        &[u64::MAX, 2, 0, 7, 11, 3],
        &[19],
        &[0, 1, 3, 6, 8],
    );

    exercise_high_division_entry_matrix::<4>("shorter numerator", &[u64::MAX], &[0, 1], &[0, 1, 3]);
}

#[test]
fn test_high_quotient_static_capacity_uses_prepared_window() {
    const N: usize = 16;
    let d = rand_nonzero_vec(12, 13_400);
    let n = rand_nonzero_vec(80, 13_401);
    assert!(n.len() > N);

    // The original numerator exceeds N, but each rigid high-quotient window
    // has only d.len()+q.len()-1 limbs and fits the static scratch arrays.
    exercise_high_division_entry_matrix::<N>("trimmed static capacity", &n, &d, &[1, 5]);
}

#[test]
fn test_forced_division_entry_matrix_dyn_and_static() {
    const N: usize = 256;
    let d = rand_nonzero_vec(BZ_CUTOFF + 5, 13_000);
    let n = rand_nonzero_vec(d.len() + 71, 13_001);
    let q_len = div_quotient_len(n.len(), d.len());

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
fn test_public_division_dispatch_above_knuth_cutoff() {
    const N: usize = 256;
    let cases = [
        ("BZ side", BZ_CUTOFF + 8, 160usize, 13_100u64),
        ("NR side", 160usize, 96usize, 13_200u64),
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

    let mut q = vec![0u64; n_work.len() - d_work.len() + 1];
    div_buf_of(&mut n_work, &mut of, &d_work, &mut q);
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
        let d_len = BZ_CUTOFF + 3 + seed as usize;
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
    let d_len = BZ_CUTOFF + 8;
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

    const FIT_N: usize = 512;
    let d_len = BZ_CUTOFF + 8;
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

    const LIMITED_N: usize = 250;
    run::<LIMITED_N>("static capacity fallback", d_len, recursive_top_q, 8904);
}

#[test]
fn test_burnikel_ziegler_top_block_cost_model_is_monotone() {
    for d_len in [BZ_CUTOFF, BZ_CUTOFF + 8, 192, 512, 1024] {
        assert!(!use_bz_for_top_block(d_len, 0));
        if d_len <= BZ_CUTOFF {
            assert!(!use_bz_for_top_block(d_len, d_len));
            continue;
        }

        let mut seen_bz = false;
        for q_len in 1..=d_len {
            let use_bz = use_bz_for_top_block(d_len, q_len);
            assert_eq!(
                use_bz,
                BZ_TOP_PADDED_COST_SCALE * bz_2_1_cost(d_len)
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
        (BZ_CUTOFF + 8, 1usize),
        (BZ_CUTOFF + 8, (BZ_CUTOFF + 8) / 2),
        (BZ_CUTOFF + 8, BZ_CUTOFF + 8),
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

#[test]
fn test_shl_top_copy_alignment_and_discarded_limb_carry() {
    let mut short = [u64::MAX; 3];
    shl_top_copy(&[0x8000_0000_0000_0001], &mut short, 1);
    assert_eq!(short, [0, 0, 2]);

    let mut equal = [0; 2];
    shl_top_copy(&[u64::MAX, 1], &mut equal, 1);
    assert_eq!(equal, [u64::MAX - 1, 3]);

    let mut long = [0; 2];
    shl_top_copy(&[99, 1 << 63, 5, 6], &mut long, 1);
    assert_eq!(long, [11, 12]);

    shl_top_copy(&[99, 2, 3, 4], &mut long, 63);
    assert_eq!(long, [(1 << 63) | 1, 1]);

    shl_top_copy(&[99, 1 << 63, 5, 6], &mut long, 0);
    assert_eq!(long, [5, 6]);
}

fn exact_rcp_reference(d: &[u64], r_len: usize) -> Vec<u64> {
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

fn assert_rcp_precision(name: &str, actual: &[u64], exact: &[u64]) {
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
        knuth_div_rcp_seed_dyn(&d, &mut seed);
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
    for cutoff in [DYN_RCP_KNUTH_NR_CUTOFF, STATIC_RCP_KNUTH_NR_CUTOFF] {
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
    // With r == N there is no spare guard capacity. NR may use its Knuth
    // capacity fallback when a refinement band needs one extra limb.
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
    // Strongly skewed precision exercises top-aligned divisor padding.
    run::<256>("skewed", 12, 129, 12_600);
    // d + r exceeds N, but reciprocal wrappers only need the precision window.
    run::<128>("precision window capacity", 96, 120, 12_700);
    // The original divisor exceeds N, but its precision window fits the
    // static reciprocal wrappers.
    run::<16>("long divisor precision window", 80, 8, 12_800);
}

#[test]
fn test_static_reciprocal_capacity_tight_precision_window() {
    const N: usize = 8;
    let mut d = vec![u64::MAX; 3 * N];
    d[2 * N..].fill(0);
    d[3 * N - 1] = 1;
    let exact = exact_rcp_reference(&d, N);
    let algorithms: [(&str, fn(&[u64], &mut [u64])); 3] = [
        ("Knuth", knuth_rcp_static::<N>),
        ("Newton-Raphson", nr_rcp_static::<N>),
        ("dispatcher", rcp_static::<N>),
    ];

    for (name, algorithm) in algorithms {
        let mut actual = [0u64; N];
        algorithm(&d, &mut actual);
        assert_rcp_precision(name, &actual, &exact);
    }
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
        ("static Knuth", knuth_rcp_static::<16>),
        ("static Newton-Raphson", nr_rcp_static::<16>),
        ("static dispatcher", rcp_static::<16>),
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
