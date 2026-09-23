//! Tests of private NR outcomes and transactional remainder finishing.
use super::*;
use crate::tests::rand_nonzero_vec;

#[test]
fn test_nr_reciprocal_guard_capacity_retries_knuth() {
    let d = rand_nonzero_vec(32, 20_300);
    let mut actual = [u64::MAX; 16];
    let window = reciprocal_divisor_window(&d, actual.len(), 16);
    assert!(!nr_rcp_attempt_static::<16>(window, &mut actual));
    assert_eq!(actual, [u64::MAX; 16]);

    let mut expected = [0; 16];
    rcp_prepared_static::<16>(&d, &mut expected, RcpAlg::Knuth);
    rcp_prepared_static::<16>(&d, &mut actual, RcpAlg::NR);
    assert_eq!(actual, expected);
}

#[test]
fn test_nr_guard_outcomes_and_finishing() {
    const N: usize = 64;
    let mut d = rand_nonzero_vec(20, 20_100);
    d[19] |= 1 << 63;
    let mut k = rand_nonzero_vec(16, 20_101);
    k[15] |= 1 << 63;
    let mut multiple = vec![0; 36];
    mul_dyn(&d, &k, &mut multiple);
    let mut interior = multiple.clone();
    let mut half = d.clone();
    shr_buf(&mut half, 1);
    assert!(!add_buf(&mut interior, &half));

    for (n, outcome) in [
        (&multiple, NrEstimate::NeedsCorrection),
        (&interior, NrEstimate::Exact),
    ] {
        let mut q = vec![u64::MAX; 17];
        assert_eq!(nr_quo_est_dyn(n, &d, &mut q), outcome);
        assert_eq!(nr_quo_est_static::<N>(n, &d, &mut q), outcome);
        for divide in [nr_div_attempt_dyn, nr_div_attempt_static::<N>] {
            assert!(divide(n, &d, &mut q));
            assert_eq!(&q[..16], k);
            assert_eq!(q[16], 0);
        }
        for divide in [nr_div_rem_attempt_dyn, nr_div_rem_attempt_static::<N>] {
            let mut rem = n.clone();
            assert!(divide(&mut rem, &d, &mut q));
            assert_eq!(&q[..16], k);
            let expected = if outcome == NrEstimate::Exact {
                half.clone()
            } else {
                vec![0; 20]
            };
            assert_eq!(&rem[..20], expected);
            assert!(rem[20..].iter().all(|&v| v == 0));
        }
    }
}

#[test]
fn test_nr_failed_attempt_preserves_numerator_for_retry() {
    // Unsupported shape and a supported near-power shape rejected by estimation.
    for (n, d) in [
        (vec![u64::MAX; 9], vec![7, 3]),
        (vec![u64::MAX; 27], {
            let mut d = vec![0; 20];
            d[0] = 1;
            d[19] = 1;
            d
        }),
    ] {
        let len = n.len() - d.len() + 1;
        let mut q = vec![0xabab_abab_abab_abab; len];
        assert_eq!(nr_quo_est_dyn(&n, &d, &mut q), NrEstimate::Failed);
        assert_eq!(nr_quo_est_static::<64>(&n, &d, &mut q), NrEstimate::Failed);
        for attempt in [nr_div_rem_attempt_dyn, nr_div_rem_attempt_static::<64>] {
            let mut rem = n.clone();
            assert!(!attempt(&mut rem, &d, &mut q));
            assert_eq!(rem, n);
        }
        let mut expected_rem = n.clone();
        let mut expected_q = vec![0; len];
        let request = division_preflight(&n, &d, &mut expected_q).unwrap();
        div_rem_prepared_dyn(
            &mut expected_rem,
            &d,
            &mut expected_q,
            request,
            DivAlg::Knuth,
        );
        for spare in [false, true] {
            for divide in [div_rem_prepared_dyn, div_rem_prepared_static::<64>] {
                let mut rem = n.clone();
                let body = len - 1;
                let mut actual = vec![u64::MAX; body + if spare { 3 } else { 0 }];
                let request = division_preflight(&rem, &d, &mut actual).unwrap();
                let overflow = divide(&mut rem, &d, &mut actual, request, DivAlg::NR);
                assert_eq!(&actual[..body], &expected_q[..body]);
                if spare {
                    assert_eq!(overflow, 0);
                    assert_eq!(actual[body], expected_q[body]);
                    assert_eq!(&actual[body + 1..], &[0, 0]);
                } else {
                    assert_eq!(overflow, expected_q[body]);
                }
                assert_eq!(rem, expected_rem);
            }
        }
    }
}

#[test]
fn test_nr_correction_rejects_distant_estimate_without_committing() {
    let d = [3, 5, 7, 1 << 63];
    let mut n = [0u64; 12];
    n[11] = 1;
    let original = n;
    let mut exact = [0u64; 9];
    let request = division_preflight(&n, &d, &mut exact).unwrap();
    div_prepared_dyn(&n, &d, &mut exact, request, DivAlg::Knuth);
    for excessive in [false, true] {
        let mut q = exact;
        if excessive {
            add_prim(&mut q, 100);
        } else {
            sub_prim(&mut q, 100);
        }
        let mut candidate = n;
        let mut full_product = [0; 12];
        assert!(!nr_exact_correction(
            &mut candidate,
            &d,
            &mut q,
            &mut full_product,
            &mut |a, b, o| {
                mul_dyn(a, b, o);
            }
        ));
        let mut q = exact;
        if excessive {
            add_prim(&mut q, 100);
        } else {
            sub_prim(&mut q, 100);
        }
        let mut rem = [0; 5];
        let mut product = [0; 8];
        assert!(!nr_rem_finish(
            &n,
            &d,
            &mut q,
            &mut rem,
            &mut product,
            &mut |a, b, o| {
                mul_dyn(a, b, o);
            }
        ));
        assert_eq!(n, original);
    }
}
