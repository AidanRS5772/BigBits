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
        assert_eq!(nr_quo_est_dyn(n, &d, &mut q, None), outcome);
        assert_eq!(nr_quo_est_static::<N>(n, &d, &mut q), outcome);
        for divide in [div_prepared_dyn, div_prepared_static::<N>] {
            let request = division_preflight(n, &d, &mut q).unwrap();
            assert_eq!(divide(n, &d, &mut q, request, DivAlg::NR), 0);
            assert_eq!(&q[..16], k);
            assert_eq!(q[16], 0);
        }
        for divide in [div_rem_prepared_dyn, div_rem_prepared_static::<N>] {
            let mut rem = n.clone();
            let request = division_preflight(&rem, &d, &mut q).unwrap();
            assert_eq!(divide(&mut rem, &d, &mut q, request, DivAlg::NR), 0);
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
        assert_eq!(nr_quo_est_dyn(&n, &d, &mut q, None), NrEstimate::Failed);
        assert_eq!(nr_quo_est_static::<64>(&n, &d, &mut q), NrEstimate::Failed);
        // The driver retries through BZ, which needs n intact.
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

// ─── carried reciprocals ────────────────────────────────────────────────────

fn carry_d(len: usize, seed: u64) -> Vec<u64> {
    let mut d = rand_nonzero_vec(len, seed);
    d[len - 1] |= 1 << 63;
    d
}

// The chain's buffers for target precision t: its sizes, step count and divisor window.
fn chain_shape(d: &[u64], t: usize) -> ([usize; 64], usize, Vec<u64>) {
    let mut sizes = [0; 64];
    let steps = nr_rcp_schedule(t, &mut sizes);
    let mut d_work = vec![0; 2 * if steps >= 2 { sizes[1] } else { 1 } + 1];
    shl_top_copy(d, &mut d_work, 0);
    (sizes, steps, d_work)
}

// The reciprocal a chain from the one-limb seed gives at precision t.
fn chain_rcp(d: &[u64], t: usize) -> Vec<u64> {
    let (sizes, steps, d_work) = chain_shape(d, t);
    let mut x = vec![0; t + 1];
    let (mut err, mut cor) = (vec![0; d_work.len() + 1], vec![0; d_work.len() + 1]);
    assert!(nr_rcp_chain(
        &d_work,
        &mut x,
        &mut err,
        &mut cor,
        &mut [0; 4],
        &sizes[..steps],
        &mut |a, b, o| mid_mul_dyn(a, b, o),
        &mut |a, b, o| hi_mul_dyn(a, b, o),
    ));
    x
}

// nr_rcp_resume from `seed`, with the result in x, which starts zeroed as the driver's does.
fn resume_rcp(seed: &[u64], d: &[u64], t: usize, x: &mut Vec<u64>) -> bool {
    let (sizes, steps, d_work) = chain_shape(d, t);
    let mut buf = vec![0; seed.len()];
    let mut carry = RcpCarry::new(&mut buf);
    if seed.len() > 1 {
        carry.keep(seed, true);
    }
    *x = vec![0; t + 1];
    let (mut err, mut cor) = (vec![0; d_work.len() + 1], vec![0; d_work.len() + 1]);
    nr_rcp_resume(
        &carry,
        &d_work,
        x,
        &mut err,
        &mut cor,
        &sizes[..steps],
        &mut |a, b, o| mid_mul_dyn(a, b, o),
        &mut |a, b, o| hi_mul_dyn(a, b, o),
    )
}

// The distance |a - b| fits the low limb: both have the precision their last limb allows.
fn assert_rcp_close(what: &str, a: &[u64], b: &[u64]) {
    let mut diff = a.to_vec();
    if sub_buf(&mut diff, b) {
        twos_comp(&mut diff);
    }
    assert!(
        diff[1..].iter().all(|&limb| limb == 0),
        "{what}: resumed reciprocal off by {:x?}",
        &diff[..diff.len().min(4)]
    );
}

// Every way a chain can resume: one step (a seed reaching the second-to-last precision, by the 1 or
// 2 limbs the stages' schedules leave), several steps (a stage that didn't divide, or kept nothing),
// and none (a seed past the target). Each lands as close as a chain from the one-limb seed.
#[test]
fn test_nr_rcp_resume_reaches_chain_precision() {
    for t in [9usize, 24, 65, 133, 264, 524, 1044, 2084] {
        let d = carry_d(2 * t + 8, 20_900 + t as u64);
        let chain = chain_rcp(&d, t);
        let (sizes, steps, _) = chain_shape(&d, t);
        let mut seeds = vec![sizes[1] + 1, sizes[1] + 2, t, t + 3];
        seeds.extend(sizes[2..steps].iter().map(|&p| p + 1));
        for have in seeds {
            let what = format!("t={t} seed p={have}");
            let mut x = vec![];
            assert!(
                resume_rcp(&chain_rcp(&d, have), &d, t, &mut x),
                "{what}: rejected"
            );
            assert_rcp_close(&what, &x, &chain);
        }
    }
}

// With no limb to spare, a seed's last-limb error can spill into a second band limb. Stages always
// have a spare limb, so this only costs a resume from further back a restart: either it's rejected
// and x is clear, or it lands as close as the chain.
#[test]
fn test_nr_rcp_resume_seed_without_a_spare_limb() {
    for t in [65usize, 264, 524, 1044] {
        let d = carry_d(2 * t + 8, 20_930 + t as u64);
        let chain = chain_rcp(&d, t);
        let (sizes, _, _) = chain_shape(&d, t);
        let mut x = vec![];
        if resume_rcp(&chain_rcp(&d, sizes[1]), &d, t, &mut x) {
            assert_rcp_close(&format!("t={t}"), &x, &chain);
        } else {
            assert!(x.iter().all(|&limb| limb == 0), "t={t}: x not cleared");
        }
    }
}

// No seed yet, or one below every step, leaves the chain to the caller with x untouched.
#[test]
fn test_nr_rcp_resume_without_a_usable_seed() {
    let d = carry_d(70, 20_950);
    let mut x = vec![];
    assert!(!resume_rcp(&[0], &d, 30, &mut x));
    assert!(x.iter().all(|&limb| limb == 0));
}

// A seed that isn't this divisor's fails its first band, however it's wrong, and resume leaves x
// zeroed for the chain that follows. (A zero seed is the one its band can't see: see below.)
#[test]
fn test_nr_rcp_resume_rejects_bad_seeds() {
    for t in [24usize, 133, 524] {
        let d = carry_d(2 * t + 8, 21_000 + t as u64);
        let (sizes, _, _) = chain_shape(&d, t);
        let p = sizes[1];
        let good = chain_rcp(&d, p);
        let mut bad: Vec<(String, Vec<u64>)> = vec![(
            "another divisor's".to_string(),
            chain_rcp(&carry_d(2 * t + 8, 21_100 + t as u64), p),
        )];
        for limb in [p, p / 2, 2, 1, 0] {
            let mut seed = good.clone();
            seed[limb] ^= 1 << 63;
            bad.push((format!("limb {limb} flipped"), seed));
        }
        for (name, seed) in bad {
            let mut x = vec![];
            assert!(
                !resume_rcp(&seed, &d, t, &mut x),
                "t={t} {name}: seed accepted"
            );
            assert!(
                x.iter().all(|&limb| limb == 0),
                "t={t} {name}: x not cleared"
            );
        }
    }
}

// What a chain of divisions by growing prefixes of one divisor (a root's stages) must give: exact
// quotients and remainders, whatever the carry holds.
fn assert_carried_division(what: &str, n: &[u64], d: &[u64], carry: &mut RcpCarry) {
    let mut want = vec![0; n.len() - d.len()];
    let mut want_rem = n.to_vec();
    let request = division_preflight(n, d, &mut want).unwrap();
    let want_top = div_rem_prepared_dyn(&mut want_rem, d, &mut want, request, DivAlg::Knuth);

    let mut q = vec![u64::MAX; want.len()];
    let top = div_carry_dyn(n, d, &mut q, carry);
    assert_eq!((top, &q), (want_top, &want), "{what}: quotient");

    let mut rem = n.to_vec();
    let top = div_rem_carry_dyn(&mut rem, d, &mut q, carry);
    assert_eq!((top, &q), (want_top, &want), "{what}: div_rem quotient");
    assert_eq!(rem, want_rem, "{what}: remainder");
}

// The stages' own shape: each divisor is the last one's limbs with more below them.
#[test]
fn test_nr_carry_across_growing_prefixes() {
    let root = carry_d(1600, 21_200);
    let mut buf = vec![0; 400];
    let mut carry = RcpCarry::new(&mut buf);
    let mut kept = 0;
    for w in [100usize, 200, 400, 800, 1600] {
        let d = &root[root.len() - w..];
        let mut n = rand_nonzero_vec(2 * w - 1, 21_300 + w as u64);
        n[2 * w - 2] = d[w - 1] - 1;
        assert_carried_division(&format!("w={w}"), &n, d, &mut carry);
        // Each NR division leaves a wider reciprocal, until one passes the buffer.
        assert!(carry.p >= kept, "w={w}: lost the reciprocal");
        kept = carry.p;
    }
    assert!(kept > 0, "no stage kept a reciprocal");
}

// A stage's final -1 can borrow through its zero low limbs into the limbs a lower stage divided by,
// and a long enough run reaches the limbs the reciprocal read. The divisor's bits differ from then
// on, but its value is a unit lower, far below the reciprocal's precision, so the seed still holds.
#[test]
fn test_nr_carry_across_a_borrowed_prefix() {
    let t = 133;
    let mut prefix = carry_d(300, 21_400);
    // The seed reads the top 135 limbs; the zeros run 35 limbs into them.
    prefix[..200].fill(0);
    let seed = chain_rcp(&prefix, t);

    let mut d = vec![0; 300];
    d.extend_from_slice(&prefix);
    assert!(!dec_buf(&mut d));
    assert_ne!(&d[300..], &prefix[..]);
    let target = 2 * t - 1;
    let mut x = vec![];
    assert!(resume_rcp(&seed, &d, target, &mut x), "seed rejected");
    assert_rcp_close("borrowed", &x, &chain_rcp(&d, target));

    let mut buf = vec![0; 200];
    let mut carry = RcpCarry::new(&mut buf);
    carry.keep(&seed, true);
    let mut n = rand_nonzero_vec(1199, 21_402);
    n[1198] = d[599] - 1;
    assert_carried_division("the prefix less one", &n, &d, &mut carry);
}

// A carry from another divisor costs a restart, not a wrong result, and the restart's reciprocal
// replaces it.
#[test]
fn test_nr_carry_from_another_divisor() {
    let (other, d) = (carry_d(400, 21_500), carry_d(800, 21_501));
    let mut buf = vec![0; 500];
    let mut carry = RcpCarry::new(&mut buf);
    carry.keep(&chain_rcp(&other, 133), true);
    let mut n = rand_nonzero_vec(1599, 21_502);
    n[1598] = d[799] - 1;
    assert_carried_division("foreign seed", &n, &d, &mut carry);
    let want = chain_rcp(&d, carry.p);
    assert_rcp_close("restarted", &carry.x[..=carry.p], &want);
}

// A zero reciprocal passes its own band, since d 0 has no error limbs to read. The quotient's band
// catches it, the driver retries through BZ, and the carry forgets it for the next stage.
#[test]
fn test_nr_carry_zero_seed() {
    let d = carry_d(800, 21_600);
    let mut buf = vec![0; 300];
    let mut carry = RcpCarry::new(&mut buf);
    carry.keep(&vec![0; 134], true);
    let mut n = rand_nonzero_vec(1599, 21_601);
    n[1598] = d[799] - 1;

    let mut want = vec![0; 799];
    let request = division_preflight(&n, &d, &mut want).unwrap();
    let want_top = div_prepared_dyn(&n, &d, &mut want, request, DivAlg::Knuth);
    let mut q = vec![u64::MAX; 799];
    let top = div_carry_dyn(&n, &d, &mut q, &mut carry);
    assert_eq!((top, &q), (want_top, &want));
    assert_eq!(carry.p, 0, "the carry kept a failed reciprocal");
}

// Both carried entries read and leave the carry: the wrong one wouldn't change a single output.
#[test]
fn test_nr_carry_entries_use_the_carry() {
    let d = carry_d(1000, 21_700);
    let mut n = rand_nonzero_vec(1999, 21_701);
    n[1998] = d[999] - 1;
    let mut q = vec![0; 999];

    let mut buf = vec![0; 600];
    let mut carry = RcpCarry::new(&mut buf);
    div_carry_dyn(&n, &d, &mut q, &mut carry);
    assert!(carry.p > 0, "div_carry_dyn left no reciprocal");

    let mut buf = vec![0; 600];
    let mut carry = RcpCarry::new(&mut buf);
    let mut rem = n.clone();
    div_rem_carry_dyn(&mut rem, &d, &mut q, &mut carry);
    assert!(carry.p > 0, "div_rem_carry_dyn left no reciprocal");

    // Each reads it, too: a zero seed is only there to be read, and failing clears it.
    for rem in [false, true] {
        let mut buf = vec![0; 600];
        let mut carry = RcpCarry::new(&mut buf);
        carry.keep(&vec![0; 134], true);
        if rem {
            div_rem_carry_dyn(&mut n.clone(), &d, &mut q, &mut carry);
        } else {
            div_carry_dyn(&n, &d, &mut q, &mut carry);
        }
        assert_eq!(carry.p, 0, "seed ignored (remainder: {rem})");
    }
}

// keep: a reciprocal past the buffer leaves the last one, and a failed one clears it.
#[test]
fn test_nr_carry_keep() {
    let mut buf = vec![0; 4];
    let mut carry = RcpCarry::new(&mut buf);
    carry.keep(&[1, 2, 3], true);
    assert_eq!(carry.p, 2);
    carry.keep(&[4, 5, 6, 7, 8], true);
    assert_eq!((carry.p, &carry.x[..=carry.p]), (2, &[1, 2, 3][..]));
    carry.keep(&[9, 9, 9], false);
    assert_eq!(carry.p, 0);
}
