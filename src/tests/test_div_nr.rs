//! Tests of private NR outcomes and transactional remainder finishing.
use super::*;
use crate::tests::rand_nonzero_vec;
use crate::tests::test_div::{assert_rcp_precision, exact_rcp_reference};

// At N = 3 an error band has no room to grow past p + 1, so NR declines without
// touching the output and the prepared driver retries through Knuth.
#[test]
fn test_nr_reciprocal_tiny_capacity_retries_knuth() {
    let d = rand_nonzero_vec(3, 20_300);
    let mut actual = [u64::MAX; 3];
    let window = reciprocal_divisor_window(&d, actual.len());
    assert!(!nr_rcp_attempt_static::<3>(window, &mut actual));
    assert_eq!(actual, [u64::MAX; 3]);

    let mut expected = [0; 3];
    rcp_prepared_static::<3>(&d, &mut expected, RcpAlg::Knuth);
    rcp_prepared_static::<3>(&d, &mut actual, RcpAlg::NR);
    assert_eq!(actual, expected);
}

// Static NR used to give up once its two guard limbs or its last divisor window
// passed N (R >= N - 3 or N - 2, by parity), leaving Knuth's O(R * D) path. Every
// R <= N must now complete through NR, both with the school middle product and
// with the NTT one, whose window then has up to 3 zero limbs past N.
#[test]
fn test_nr_reciprocal_static_completes_up_to_capacity() {
    fn run<const N: usize>(rs: &[usize], tops: &[u64], d_lens: &[usize], seed: u64) {
        for &r in rs {
            for (t, &top) in tops.iter().enumerate() {
                for &d_len in d_lens {
                    let mut d = rand_nonzero_vec(d_len, seed + (r * 31 + t * 7 + d_len) as u64);
                    *d.last_mut().unwrap() = top;
                    let window = reciprocal_divisor_window(&d, r);
                    let mut actual = vec![0; r];
                    let what = format!("N={N} R={r} top={top:#x} d_len={d_len}");
                    assert!(
                        nr_rcp_attempt_static::<N>(window, &mut actual),
                        "{what}: NR declined"
                    );
                    assert_rcp_precision(&what, &actual, &exact_rcp_reference(&d, r));
                }
            }
        }
    }
    let tops = [u64::MAX, 1 << 63 | 5, 3, 1];
    for n in [16usize, 64] {
        let rs: Vec<usize> = (n - 4..=n).collect();
        let d_lens = [n - 4, n];
        match n {
            16 => run::<16>(&rs, &tops, &d_lens, 20_400),
            _ => run::<64>(&rs, &tops, &d_lens, 20_500),
        }
    }
    // The NTT middle product starts at a 200-limb short operand: p + 1 >= 200.
    run::<419>(&[415, 416, 417, 418, 419], &tops, &[419], 20_600);
    run::<2048>(&[2045, 2047, 2048], &tops[1..3], &[2048], 20_700);
    // 2043 isn't 5-smooth, so the padded convolution is chunked.
    run::<2043>(&[2042, 2043], &tops[1..3], &[2043], 20_800);
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

// NR's error band is a middle product, which drops the carries from the columns below
// it. With an odd quotient width, the band's base lines up with the estimate's top
// guard limb, and a divisor with a small top limb scaled that dropped carry up to
// several units there. An exact division then came out one short while its guard
// limb, just under all-ones, passed as proof of the floor.
#[test]
fn test_nr_exact_division_small_top_divisor() {
    fn off_limb(q: &[u64], want: &[u64]) -> Option<usize> {
        (0..q.len().max(want.len())).find(|&i| q.get(i) != want.get(i))
    }
    fn run<const N: usize>(
        q_lens: &[usize],
        d_scale: usize,
        tops: &[u64],
        seeds: u64,
        static_too: bool,
    ) {
        for &q_len in q_lens {
            for &top in tops {
                for s in 0..seeds {
                    let seed = 30_000 + (q_len as u64) * 97 + top % 1000 * 13 + s;
                    let mut d = rand_nonzero_vec(d_scale * q_len, seed);
                    *d.last_mut().unwrap() = top;
                    let mut want = rand_nonzero_vec(q_len, seed + 1);
                    // A small top limb keeps n at q_len + d.len() - 1 limbs, so the
                    // quotient width is q_len, which is odd.
                    *want.last_mut().unwrap() = 1 << 40;
                    let mut n = vec![0; q_len + d.len()];
                    mul_dyn(&want, &d, &mut n);
                    trim_lz(&mut n);
                    assert_eq!(n.len() - d.len() + 1, q_len);
                    let what = format!("q_len={q_len} d_len={} top={top:#x} seed={seed}", d.len());

                    let mut q = vec![0; q_len];
                    let req = division_preflight(&n, &d, &mut q).unwrap();
                    div_prepared_dyn(&n, &d, &mut q, req, DivAlg::NR);
                    trim_lz(&mut q);
                    assert!(
                        q == want,
                        "dyn NR {what} off from limb {:?}",
                        off_limb(&q, &want)
                    );
                    if static_too {
                        let mut q = vec![0; q_len];
                        let req = division_preflight(&n, &d, &mut q).unwrap();
                        div_prepared_static::<N>(&n, &d, &mut q, req, DivAlg::NR);
                        trim_lz(&mut q);
                        assert!(
                            q == want,
                            "static NR {what} off from limb {:?}",
                            off_limb(&q, &want)
                        );
                    }
                }
            }
        }
    }
    // Below 200 limbs static NR takes the school middle product, which adds back the
    // column below the band and loses under an ulp; the dyn FFT and NTT ones and the
    // static NTT one (from h + 1 = 200) drop every lower column.
    let tops = [1, 0xb, 0xff, 1 << 20];
    run::<512>(&[9, 21, 63, 101], 3, &tops, 6, true);
    run::<1024>(&[201, 255], 3, &tops, 4, true);
    run::<2048>(&[401], 3, &tops, 3, true);
    run::<1>(&[739], 4, &tops, 3, false);
}
