use super::{mul_ref, rand_vec};
use crate::utils::sqrt::{
    binom_sqrt, sqrt_approx_dyn, sqrt_approx_dyn_at, sqrt_approx_static, sqrt_dyn, sqrt_dyn_at,
    sqrt_only_dyn, sqrt_only_dyn_at, sqrt_only_static, sqrt_static, zimmerman_sqrt_approx_dyn,
    zimmerman_sqrt_approx_static, zimmerman_sqrt_dyn, zimmerman_sqrt_only_dyn,
    zimmerman_sqrt_only_static, zimmerman_sqrt_static,
};
use crate::utils::utils::{add_buf, cmp_buf, dec_buf, eq_buf, inc_buf, sub_buf};
use crate::tuning::{k, resolved};
#[cfg(feature = "tune")]
use rand::{rngs::StdRng, Rng};

/// Verify both outputs for
/// `shifted_x = x * B^(2 * root.len() - x.len()) = root^2 + remainder`.
fn assert_sqrt_contract_with(x: &[u64], root_len: usize, sqrt_alg: fn(&mut [u64], &mut [u64])) {
    assert!(root_len < x.len());
    assert!(x.len() <= 2 * root_len);
    assert_ne!(x.last().copied().unwrap(), 0);

    let shift = 2 * root_len - x.len();
    let mut shifted_x = vec![0; shift];
    shifted_x.extend_from_slice(x);

    let mut work = x.to_vec();
    let mut root = vec![u64::MAX; root_len];
    sqrt_alg(&mut work, &mut root);

    let root_squared = mul_ref(&root, &root);
    let mut expected_remainder = shifted_x.clone();
    assert!(
        !sub_buf(&mut expected_remainder, &root_squared),
        "root is too large: x={x:#x?}, root={root:#x?}, \
         root_squared={root_squared:#x?}, shifted_x={shifted_x:#x?}"
    );
    assert!(
        cmp_buf(&root_squared, &shifted_x).is_le(),
        "root is too large: x={x:#x?}, root={root:#x?}, \
         root_squared={root_squared:#x?}, shifted_x={shifted_x:#x?}"
    );

    let mut next_root = root.clone();
    if inc_buf(&mut next_root) {
        next_root.push(1);
    }
    let next_root_squared = mul_ref(&next_root, &next_root);
    assert!(
        cmp_buf(&shifted_x, &next_root_squared).is_lt(),
        "root is too small: x={x:#x?}, root={root:#x?}, \
         next_root_squared={next_root_squared:#x?}, shifted_x={shifted_x:#x?}"
    );
    assert!(
        eq_buf(&work, &expected_remainder),
        "remainder mismatch: x={x:#x?}, root={root:#x?}, \
         got_remainder={work:#x?}, expected_remainder={expected_remainder:#x?}"
    );
}

fn assert_sqrt_contract(x: &[u64], root_len: usize) {
    assert_sqrt_contract_with(x, root_len, binom_sqrt);
}

fn assert_zimmerman_contract(x: &[u64], root_len: usize) {
    assert_sqrt_contract_with(x, root_len, zimmerman_sqrt_dyn);
    assert_sqrt_contract_with(x, root_len, zimmerman_sqrt_static::<256>);

    let mut dyn_work = x.to_vec();
    let mut static_work = x.to_vec();
    let mut dyn_root = vec![u64::MAX; root_len];
    let mut static_root = vec![u64::MAX; root_len];
    zimmerman_sqrt_dyn(&mut dyn_work, &mut dyn_root);
    zimmerman_sqrt_static::<256>(&mut static_work, &mut static_root);
    assert_eq!(dyn_root, static_root, "root mismatch for x={x:#x?}");
    assert_eq!(dyn_work, static_work, "remainder mismatch for x={x:#x?}");
}

#[test]
fn test_binom_sqrt_single_limb_root() {
    for x in [
        vec![0, 1 << 62],
        vec![u64::MAX, 1 << 62],
        vec![u64::MAX, u64::MAX],
    ] {
        assert_sqrt_contract(&x, 1);
    }
}

#[test]
fn test_binom_sqrt_two_limb_root() {
    for x_len in 3..=4 {
        for case in 0..16 {
            let seed = 5_000 + 32 * x_len as u64 + case;
            let mut x = rand_vec(x_len, seed);
            x[x_len - 1] |= 1 << 62;
            assert_sqrt_contract(&x, 2);
        }
    }
}

#[test]
fn test_binom_sqrt_seed_remainder_boundaries() {
    // Exercise both sides of the old c=0/1 branches and the rem=2*s_hi
    // saturation branch, including the three-limb input's virtual low zero.
    for h in [1 << 63, (1 << 63) + 1, u64::MAX - 1, u64::MAX] {
        let h = h as u128;
        for rem in [0, 1, h - 1, h, h + 1, 2 * h - 1, 2 * h] {
            let hi = h * h + rem;
            for x1 in [0, 1, 1 << 63, u64::MAX - 1, u64::MAX] {
                for x0 in [0, 1, u64::MAX] {
                    let x = [x0, x1, hi as u64, (hi >> 64) as u64];
                    assert_sqrt_contract(&x, 2);
                    if x0 == 0 {
                        assert_sqrt_contract(&x[1..], 2);
                    }
                }
            }
        }
    }
}

#[test]
fn test_binom_sqrt_sub64_all_shapes_and_normalization() {
    // Directly call binom_sqrt regardless of the dispatcher cutoff.
    for root_len in 1..64 {
        for x_len in root_len + 1..=2 * root_len {
            for case in 0..4 {
                let seed = 130_000 + 10_000 * root_len as u64 + 64 * x_len as u64 + case;
                let mut x = rand_vec(x_len, seed);
                x[x_len - 1] = match case {
                    0 => x[x_len - 1] | (1 << 63),
                    1 => 1,
                    2 => 1 << ((root_len + x_len) % 64),
                    _ => {
                        x.fill(u64::MAX);
                        u64::MAX
                    }
                };
                assert_sqrt_contract(&x, root_len);
            }
        }
    }
}

#[test]
fn test_binom_sqrt_sub64_square_boundaries() {
    for root_len in [1, 2, 3, 4, 8, 16, 32, 63] {
        for case in 0..8 {
            let mut root = rand_vec(root_len, 1_000_000 + 32 * root_len as u64 + case);
            root[root_len - 1] = match case {
                0 => 1 << 63,
                1 => {
                    root.fill(u64::MAX);
                    u64::MAX
                }
                _ => root[root_len - 1] | (1 << 63),
            };
            let square = mul_ref(&root, &root);
            let mut below = square.clone();
            assert!(!dec_buf(&mut below));
            let mut above = square.clone();
            assert!(!inc_buf(&mut above));
            let mut below_next = square.clone();
            assert!(!add_buf(&mut below_next, &root));
            assert!(!add_buf(&mut below_next, &root));
            for x in [&below, &square, &above, &below_next] {
                assert_sqrt_contract(x, root_len);
            }
        }
    }
}

#[test]
fn test_binom_sqrt_perfect_squares_main_loop_all_supported_shapes() {
    for root_len in 3usize..=8 {
        for x_len in root_len + 1..=2 * root_len {
            let shift = 2 * root_len - x_len;
            let zero_root_limbs = shift.div_ceil(2);
            let mut expected = rand_vec(root_len, 10_000 + 31 * root_len as u64 + x_len as u64);
            expected[..zero_root_limbs].fill(0);
            expected[root_len - 1] |= 3 << 62;

            let square = mul_ref(&expected, &expected);
            assert_eq!(square.len(), 2 * root_len);
            assert!(square[..shift].iter().all(|&limb| limb == 0));

            let x = square[shift..].to_vec();
            assert_eq!(x.len(), x_len);
            assert!(x[x_len - 1] >= (1 << 62));

            let mut work = x.clone();
            let mut actual = vec![u64::MAX; root_len];
            binom_sqrt(&mut work, &mut actual);
            assert_eq!(
                actual, expected,
                "perfect-square mismatch for root_len={root_len}, x_len={x_len}, x={x:#x?}"
            );
            assert!(
                work.iter().all(|&limb| limb == 0),
                "perfect square left a nonzero remainder for root_len={root_len}, \
                 x_len={x_len}, remainder={work:#x?}"
            );
        }
    }
}

#[test]
fn test_binom_sqrt_normalized_randomized_contract() {
    for root_len in 3..=12 {
        for x_len in root_len + 1..=2 * root_len {
            for case in 0..16 {
                let seed = 20_000 + 1_000 * root_len as u64 + 32 * x_len as u64 + case as u64;
                let mut x = rand_vec(x_len, seed);
                x[x_len - 1] |= 1 << 62;
                assert_sqrt_contract(&x, root_len);
            }
        }
    }
}

#[test]
fn test_zimmerman_sqrt_dyn_full_and_virtually_padded_inputs() {
    for root_len in [50usize, 51, 64] {
        for x_len in [root_len + 1, 3 * root_len / 2, 2 * root_len] {
            for case in 0..3 {
                let seed = 30_000 + 1_000 * root_len as u64 + 32 * x_len as u64 + case;
                let mut x = rand_vec(x_len, seed);
                x[x_len - 1] = match case {
                    0 => 1,
                    1 => x[x_len - 1] | (1 << 61),
                    _ => x[x_len - 1] | (1 << 63),
                };
                assert_sqrt_contract_with(&x, root_len, zimmerman_sqrt_dyn);
            }
        }
    }
}

#[test]
fn test_zimmerman_sqrt_static_full_and_virtually_padded_inputs() {
    for root_len in [50usize, 51, 64] {
        for x_len in [root_len + 1, 3 * root_len / 2, 2 * root_len] {
            for case in 0..3 {
                let seed = 40_000 + 1_000 * root_len as u64 + 32 * x_len as u64 + case;
                let mut x = rand_vec(x_len, seed);
                x[x_len - 1] = match case {
                    0 => 1,
                    1 => x[x_len - 1] | (1 << 61),
                    _ => x[x_len - 1] | (1 << 63),
                };
                assert_sqrt_contract_with(&x, root_len, zimmerman_sqrt_static::<128>);
            }
        }
    }
}

#[test]
fn test_zimmerman_sqrt_all_input_shapes_around_cutoff() {
    for root_len in [49usize, 50, 51] {
        for x_len in root_len + 1..=2 * root_len {
            let seed = 50_000 + 1_000 * root_len as u64 + x_len as u64;
            let mut x = rand_vec(x_len, seed);
            x[x_len - 1] = match (x_len - root_len) % 3 {
                0 => 1,
                1 => x[x_len - 1] | (1 << 61),
                _ => x[x_len - 1] | (1 << 63),
            };
            assert_zimmerman_contract(&x, root_len);
        }
    }
}

#[test]
fn test_zimmerman_sqrt_deeper_recursion_boundaries() {
    for root_len in [99usize, 100, 101] {
        let mut x_lens = vec![
            root_len + 1,
            root_len + 2,
            3 * root_len / 2,
            2 * root_len - 1,
            2 * root_len,
        ];
        x_lens.sort_unstable();
        x_lens.dedup();

        for x_len in x_lens {
            let seed = 60_000 + 1_000 * root_len as u64 + x_len as u64;
            let mut x = rand_vec(x_len, seed);
            x[x_len - 1] |= 1 << 63;
            assert_zimmerman_contract(&x, root_len);
        }
    }
}

#[test]
fn test_zimmerman_sqrt_perfect_squares_with_virtual_padding() {
    for root_len in [50usize, 51, 100] {
        for x_len in [
            root_len + 1,
            root_len + 2,
            3 * root_len / 2,
            2 * root_len - 1,
            2 * root_len,
        ] {
            let shift = 2 * root_len - x_len;
            let mut root = rand_vec(root_len, 70_000 + 1_000 * root_len as u64 + x_len as u64);
            root[..shift.div_ceil(2)].fill(0);
            root[root_len - 1] |= 1 << 63;

            let square = mul_ref(&root, &root);
            assert_eq!(square.len(), 2 * root_len);
            assert!(square[..shift].iter().all(|&limb| limb == 0));
            assert_zimmerman_contract(&square[shift..], root_len);
        }
    }
}

#[test]
fn test_zimmerman_sqrt_values_around_square_boundaries() {
    for root_len in [50usize, 51, 100] {
        let mut root = rand_vec(root_len, 80_000 + root_len as u64);
        root[root_len - 1] = 1 << 63;
        let square = mul_ref(&root, &root);
        assert_eq!(square.len(), 2 * root_len);

        let mut below_square = square.clone();
        assert!(!dec_buf(&mut below_square));

        let mut below_next_square = square.clone();
        assert!(!add_buf(&mut below_next_square, &root));
        assert!(!add_buf(&mut below_next_square, &root));

        let mut next_square = below_next_square.clone();
        assert!(!inc_buf(&mut next_square));

        for x in [&below_square, &square, &below_next_square, &next_square] {
            assert_zimmerman_contract(x, root_len);
        }
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "Zimmermann sqrt operands exceed static capacity")]
fn test_zimmerman_sqrt_static_rejects_insufficient_capacity() {
    let root_len = 50;
    let mut x = vec![0; 2 * root_len];
    x[2 * root_len - 1] = 1;
    let mut root = vec![0; root_len];

    zimmerman_sqrt_static::<99>(&mut x, &mut root);
}

fn assert_sqrt_only_contract<const N: usize>(x: &[u64], root_len: usize) {
    // The independent digit-by-digit path supplies the expected root. Also
    // verify it against schoolbook squares, without relying on sqrt remainders.
    let mut reference_work = x.to_vec();
    let mut expected = vec![u64::MAX; root_len];
    binom_sqrt(&mut reference_work, &mut expected);
    let mut shifted_x = vec![0; 2 * root_len - x.len()];
    shifted_x.extend_from_slice(x);
    let mut next_root = expected.clone();
    if inc_buf(&mut next_root) {
        next_root.push(1);
    }
    assert!(cmp_buf(&mul_ref(&expected, &expected), &shifted_x).is_le());
    assert!(cmp_buf(&shifted_x, &mul_ref(&next_root, &next_root)).is_lt());

    let entries: [(&str, fn(&mut [u64], &mut [u64]), bool); 8] = [
        ("exact dyn", zimmerman_sqrt_only_dyn, false),
        ("exact static", zimmerman_sqrt_only_static::<N>, false),
        ("approx dyn", zimmerman_sqrt_approx_dyn, true),
        ("approx static", zimmerman_sqrt_approx_static::<N>, true),
        ("dispatched exact dyn", sqrt_only_dyn, false),
        ("dispatched exact static", sqrt_only_static::<N>, false),
        ("dispatched approx dyn", sqrt_approx_dyn, true),
        ("dispatched approx static", sqrt_approx_static::<N>, true),
    ];
    for (name, entry, approximate) in entries {
        let mut work = x.to_vec();
        // Poisoned outputs catch unwritten quotient limbs and scratch leakage.
        let mut root = vec![u64::MAX; root_len];
        entry(&mut work, &mut root);
        assert!(
            root == expected || (approximate && eq_buf(&root, &next_root)),
            "{name} violated root contract: root_len={root_len}, x={x:#x?}, \
             root={root:#x?}, expected={expected:#x?}"
        );
    }
}

#[test]
fn test_sqrt_only_all_shapes_and_normalization_around_cutoff() {
    let leaf = k::sqrt::leaf();
    let mut root_lens = vec![1, 2, 3, 2 * leaf - 1, 2 * leaf + 1];
    for cutoff in [
        leaf,
        k::sqrt::dyn_only(),
        k::sqrt::dyn_approx(),
        k::sqrt::static_only(),
        k::sqrt::static_approx(),
    ] {
        root_lens.extend([cutoff.saturating_sub(1), cutoff, cutoff + 1]);
    }
    root_lens.retain(|&r| (1..=64).contains(&r));
    root_lens.sort_unstable();
    root_lens.dedup();
    for root_len in root_lens {
        for x_len in root_len + 1..=2 * root_len {
            for case in 0..5 {
                let mut x = rand_vec(
                    x_len,
                    90_000 + 1000 * root_len as u64 + 10 * x_len as u64 + case,
                );
                x[x_len - 1] = match case {
                    0 => 1,
                    1 => 1 << 2,
                    2 => 1 << 61,
                    3 => 1 << 62,
                    _ => x[x_len - 1] | (1 << 63),
                };
                assert_sqrt_only_contract::<128>(&x, root_len);
            }
        }
    }
}

#[test]
fn test_sqrt_only_square_boundaries_and_saturation() {
    for root_len in [16, 17, 18, 33, 34, 65, 100] {
        let mut root = rand_vec(root_len, 100_000 + root_len as u64);
        root[root_len - 1] |= 1 << 63;
        let square = mul_ref(&root, &root);
        let mut below = square.clone();
        dec_buf(&mut below);
        let mut above = square.clone();
        inc_buf(&mut above);
        for x in [&below, &square, &above] {
            assert_sqrt_only_contract::<256>(x, root_len);
        }

        for x_len in root_len + 1..=2 * root_len {
            // An all-ones high radicand has remainder 2*s_hi, exercising
            // quotient saturation in recursive and virtually padded stages.
            let x = vec![u64::MAX; x_len];
            assert_sqrt_only_contract::<256>(&x, root_len);
            assert_zimmerman_contract(&x, root_len);
        }
    }
}

#[test]
fn test_sqrt_only_correction_and_one_unit_overestimate() {
    for root_len in [17, 18, 33, 34, 65] {
        let lo = (root_len - 1) / 2;
        let mut candidate = vec![0; root_len];
        candidate[root_len - 1] = 1 << 63;
        candidate[lo - 1] = 1;
        let mut x = mul_ref(&candidate, &candidate);
        assert!(!dec_buf(&mut x));
        let mut expected = candidate.clone();
        assert!(!dec_buf(&mut expected));

        let mut root = vec![0; root_len];
        zimmerman_sqrt_only_dyn(&mut x.clone(), &mut root);
        assert_eq!(root, expected);
        zimmerman_sqrt_only_static::<256>(&mut x.clone(), &mut root);
        assert_eq!(root, expected);
        sqrt_only_dyn(&mut x.clone(), &mut root);
        assert_eq!(root, expected);
        sqrt_only_static::<256>(&mut x.clone(), &mut root);
        assert_eq!(root, expected);
        // The uncorrected outer stage must actually return the candidate,
        // including when the exact correction borrows across many limbs.
        zimmerman_sqrt_approx_dyn(&mut x.clone(), &mut root);
        assert_eq!(root, candidate);
        zimmerman_sqrt_approx_static::<256>(&mut x.clone(), &mut root);
        assert_eq!(root, candidate);
        sqrt_approx_dyn(&mut x.clone(), &mut root);
        assert_eq!(root, candidate);
        sqrt_approx_static::<256>(&mut x.clone(), &mut root);
        assert_eq!(root, candidate);
    }
}

#[test]
fn test_sqrt_only_virtually_padded_perfect_squares() {
    for root_len in [17usize, 18, 33, 64] {
        for x_len in root_len + 1..2 * root_len {
            let shift = 2 * root_len - x_len;
            let mut root = rand_vec(root_len, 110_000 + 100 * root_len as u64 + x_len as u64);
            root[..shift.div_ceil(2)].fill(0);
            root[root_len - 1] |= 1 << 63;
            let square = mul_ref(&root, &root);
            assert_sqrt_only_contract::<128>(&square[shift..], root_len);
        }
    }
}

#[test]
fn test_sqrt_zero_numerator_and_power_of_two_squares() {
    for root_len in [17, 18, 33, 34, 65] {
        for x_len in root_len + 1..=2 * root_len {
            let mut x = vec![0; x_len];
            x[x_len - 1] = 1 << 62;
            assert_sqrt_only_contract::<256>(&x, root_len);
            assert_zimmerman_contract(&x, root_len);

            // Nonzero data below the recursive division's numerator window
            // must not affect how its empty numerator is trimmed.
            x[0] = 1;
            assert_sqrt_only_contract::<256>(&x, root_len);
            assert_zimmerman_contract(&x, root_len);
        }
    }
}

#[test]
fn test_sqrt_only_larger_division_backends() {
    for root_len in [255, 256, 257, 512] {
        for x_len in [
            root_len + 1,
            root_len + 2,
            3 * root_len / 2,
            2 * root_len - 1,
            2 * root_len,
        ] {
            let mut x = rand_vec(x_len, 120_000 + 1000 * root_len as u64 + x_len as u64);
            x[x_len - 1] |= 1 << 63;
            assert_sqrt_only_contract::<1024>(&x, root_len);
        }
    }
}

#[test]
fn test_sqrt_static_exact_capacity() {
    let mut full = rand_vec(34, 130_000);
    full[33] |= 1 << 63;
    assert_sqrt_only_contract::<34>(&full, 17);
    assert_sqrt_contract_with(&full, 17, sqrt_static::<34>);
    let mut padded = rand_vec(18, 130_001);
    padded[17] |= 1 << 63;
    assert_sqrt_only_contract::<18>(&padded, 17);
    assert_sqrt_contract_with(&padded, 17, sqrt_static::<18>);
}

#[test]
fn test_sqrt_dispatch_root_remainder_all_shapes_around_cutoff() {
    const N: usize = 2 * (resolved::sqrt::DYN_REM + resolved::sqrt::STATIC_REM) + 2;
    let mut root_lens = vec![1, 2];
    for cutoff in [k::sqrt::leaf(), k::sqrt::dyn_rem(), k::sqrt::static_rem()] {
        root_lens.extend([cutoff.saturating_sub(1), cutoff, cutoff + 1]);
    }
    root_lens.retain(|&r| r >= 1 && 2 * r <= N);
    root_lens.sort_unstable();
    root_lens.dedup();
    for root_len in root_lens {
        for x_len in root_len + 1..=2 * root_len {
            for top in [1, 1 << 61, 1 << 62, u64::MAX] {
                let mut x = rand_vec(x_len, 140_000 + 100 * root_len as u64 + x_len as u64);
                x[x_len - 1] = top;
                assert_sqrt_contract_with(&x, root_len, sqrt_dyn);
                assert_sqrt_contract_with(&x, root_len, sqrt_static::<N>);
            }
        }
    }
}

#[test]
#[cfg(debug_assertions)]
fn test_sqrt_dispatch_static_capacity_applies_to_both_algorithms() {
    // With N = 4 every root of 3 or more limbs exceeds capacity, so roots on both sides of each
    // entry's cutoff (whatever its value) must be rejected.
    const N: usize = 4;
    let entries: [(fn(&mut [u64], &mut [u64]), usize); 3] = [
        (sqrt_static::<N>, k::sqrt::static_rem()),
        (sqrt_only_static::<N>, k::sqrt::static_only()),
        (sqrt_approx_static::<N>, k::sqrt::static_approx()),
    ];
    for (entry, cutoff) in entries {
        for root_len in [3, cutoff.saturating_sub(1).max(3), cutoff.max(3)] {
            let result = std::panic::catch_unwind(|| {
                entry(&mut vec![u64::MAX; 2 * root_len], &mut vec![0; root_len]);
            });
            assert!(
                result.is_err(),
                "accepted operands exceeding static capacity"
            );
        }
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "Zimmermann sqrt operands exceed static capacity")]
fn test_sqrt_only_static_rejects_insufficient_capacity() {
    zimmerman_sqrt_only_static::<33>(&mut [1; 34], &mut [0; 17]);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "Zimmermann sqrt operands exceed static capacity")]
fn test_sqrt_approx_static_rejects_insufficient_capacity() {
    zimmerman_sqrt_approx_static::<17>(&mut [1; 18], &mut [0; 17]);
}

// The root outputs take a one-limb x into a one-limb root, X = x B, checked against u128 squares.
fn assert_one_limb_roots<const N: usize>() {
    let mut xs = vec![1, 2, 3, 4, 5, 1 << 32, (1 << 62) - 1, 1 << 62, 1 << 63];
    xs.extend([u64::MAX - 1, u64::MAX]);
    xs.extend(rand_vec(64, 77).into_iter().filter(|&x| x != 0));
    let entries: [(&str, fn(&mut [u64], &mut [u64]), bool); 8] = [
        ("exact dyn", zimmerman_sqrt_only_dyn, false),
        ("exact static", zimmerman_sqrt_only_static::<N>, false),
        ("approx dyn", zimmerman_sqrt_approx_dyn, true),
        ("approx static", zimmerman_sqrt_approx_static::<N>, true),
        ("dispatched exact dyn", sqrt_only_dyn, false),
        ("dispatched exact static", sqrt_only_static::<N>, false),
        ("dispatched approx dyn", sqrt_approx_dyn, true),
        ("dispatched approx static", sqrt_approx_static::<N>, true),
    ];
    for x in xs {
        let big_x = (x as u128) << 64;
        let want = big_x.isqrt();
        assert!(want * want <= big_x);
        assert!((want + 1).checked_mul(want + 1).map_or(true, |sq| sq > big_x));
        for (name, entry, approximate) in entries {
            let mut root = [u64::MAX];
            entry(&mut [x], &mut root);
            let got = root[0] as u128;
            assert!(
                got == want || (approximate && got == want + 1),
                "{name}: x={x:#x}, root={got:#x}, want={want:#x}"
            );
        }
    }
}

#[test]
fn test_sqrt_root_outputs_one_limb() {
    assert_one_limb_roots::<1>();
    assert_one_limb_roots::<4>();
}

// The remainder of a one-limb x can take two limbs, so the remainder entries keep x.len() > s.len().
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "x.len() > s.len()")]
fn test_sqrt_rem_rejects_one_limb() {
    sqrt_dyn(&mut [5], &mut [0]);
}

// ─── wrapped remainder ──────────────────────────────────────────────────────

// Every dynamic stage from the recursive leaf up forms its remainder from a wrapped square.
fn sqrt_dyn_wrapped_all(x: &mut [u64], s: &mut [u64]) {
    sqrt_dyn_at(x, s, k::sqrt::leaf());
}

// The wrapped stages change how a remainder is formed, not the root or remainder, so every output
// mode must match the classic stages (cutoff usize::MAX) bit for bit: the root everywhere, and
// the remainder for sqrt_dyn. The root-and-remainder output is also checked against schoolbook.
fn assert_wrapped_matches_classic(x: &[u64], root_len: usize) {
    type At = fn(&mut [u64], &mut [u64], usize);
    let mut cutoffs = vec![
        k::sqrt::leaf(),
        (root_len / 2).max(k::sqrt::leaf()),
        k::sqrt::dyn_wrapped_rem(),
    ];
    cutoffs.dedup();
    let modes: [(&str, At, bool); 3] = [
        ("rem", sqrt_dyn_at, true),
        ("only", sqrt_only_dyn_at, false),
        ("approx", sqrt_approx_dyn_at, false),
    ];
    for (name, at, remainder) in modes {
        let mut classic_x = x.to_vec();
        let mut classic_s = vec![u64::MAX; root_len];
        at(&mut classic_x, &mut classic_s, usize::MAX);
        for &cutoff in &cutoffs {
            let mut wrapped_x = x.to_vec();
            let mut wrapped_s = vec![0xdead_beef; root_len];
            at(&mut wrapped_x, &mut wrapped_s, cutoff);
            let shape = format!("root_len={root_len} x.len()={} cutoff={cutoff}", x.len());
            assert_eq!(wrapped_s, classic_s, "{name} root differs: {shape}");
            if remainder {
                assert_eq!(wrapped_x, classic_x, "{name} remainder differs: {shape}");
            }
        }
    }
    assert_sqrt_contract_with(x, root_len, sqrt_dyn_wrapped_all);
}

// Random inputs of every top-level shape (full, one short, three quarters, and the most padded),
// with and without a normalization shift, at sizes whose stages run the folded fallback (small)
// and the cyclic FFT (from `mul.sqr_bnm1_fft` limbs).
#[test]
fn test_sqrt_wrapped_remainder_matches_classic() {
    for root_len in [24usize, 50, 101, 400, 700, 1500] {
        let mut x_lens = vec![
            2 * root_len,
            2 * root_len - 1,
            3 * root_len / 2 + 1,
            root_len + 1,
        ];
        x_lens.dedup();
        for (i, x_len) in x_lens.into_iter().enumerate() {
            let mut x = rand_vec(x_len, 0x5_0000 + 100 * root_len as u64 + x_len as u64);
            x[x_len - 1] = match i % 3 {
                0 => x[x_len - 1] | (1 << 63),
                1 => (x[x_len - 1] >> 5) | 1,
                _ => 1,
            };
            assert_wrapped_matches_classic(&x, root_len);
        }
    }
}

// y^2 + delta (delta in {-1, 0, 2y}) at `len` limbs.
fn square_plus(y: &[u64], len: usize, delta: i8) -> Vec<u64> {
    let mut v = mul_ref(y, y);
    v.resize(len, 0);
    match delta {
        -1 => assert!(!sub_buf(&mut v, &[1])),
        0 => {}
        _ => {
            add_buf(&mut v, y);
            add_buf(&mut v, y);
        }
    }
    v
}

// Remainder edge cases at the top stage and at the core's top stage, where the wrapped residue
// is zero (perfect squares), negative (s one too big: Y^2 - 1), at its 2s maximum, or where a
// stage saturates (its child's remainder is twice its root).
#[test]
fn test_sqrt_wrapped_remainder_edges() {
    for root_len in [50usize, 400, 700] {
        let x_len = 2 * root_len;
        let mut y = rand_vec(root_len, 0x5_1000 + root_len as u64);
        y[root_len - 1] |= 1 << 63;
        let mut sparse_y = vec![0; root_len];
        sparse_y[root_len - 1] = 1 << 63;
        sparse_y[0] = 1;

        let mut cases = vec![
            ("Y^2", square_plus(&y, x_len, 0)),
            ("Y^2 - 1", square_plus(&y, x_len, -1)),
            ("(Y + 1)^2 - 1", square_plus(&y, x_len, 2)),
            ("sparse Y^2", square_plus(&sparse_y, x_len, 0)),
            ("sparse Y^2 - 1", square_plus(&sparse_y, x_len, -1)),
            ("all ones", vec![u64::MAX; x_len]),
        ];
        let mut min = vec![0; x_len];
        min[x_len - 1] = 1 << 62;
        cases.push(("smallest normalized", min));

        // Saturate the top stage (its child has hi_e limbs) and the core's top stage (its child
        // has h limbs): (t + 1)^2 - 1 in the top 2w limbs makes the child's remainder 2t.
        let hi_e = root_len - (root_len - 1) / 2;
        let h = hi_e - (hi_e - 1) / 2;
        for (name, w) in [("saturated top stage", hi_e), ("saturated core stage", h)] {
            let mut t = rand_vec(w, 0x5_2000 + (root_len * 10 + w) as u64);
            t[w - 1] |= 1 << 63;
            let mut x = rand_vec(x_len, 0x5_3000 + (root_len * 10 + w) as u64);
            x[x_len - 2 * w..].copy_from_slice(&square_plus(&t, 2 * w, 2));
            cases.push((name, x));
        }

        // Padded tops (X = x B^p): a perfect square with p low zero limbs, and one less than it
        // by B^p (a negative tentative remainder at the top stage).
        for p in [2usize, 2 * ((root_len - 1) / 2)] {
            let mut yp = vec![0; root_len];
            yp[p / 2..].copy_from_slice(&y[p / 2..]);
            let full = square_plus(&yp, x_len, 0);
            assert!(full[..p].iter().all(|&limb| limb == 0));
            cases.push(("padded Y^2", full[p..].to_vec()));
            let mut less = full[p..].to_vec();
            assert!(!sub_buf(&mut less, &[1]));
            cases.push(("padded Y^2 - B^p", less));
        }

        for (name, x) in cases {
            assert_ne!(x[x.len() - 1], 0, "{name}: untrimmed input");
            println!("{name}, root_len={root_len}");
            assert_wrapped_matches_classic(&x, root_len);
        }
    }
}

// ─── carried reciprocals ────────────────────────────────────────────────────

// From `sqrt.dyn_rcp_carry` root limbs each NR stage resumes from the last one's reciprocal,
// which changes how a quotient is reached, not its value. Every output mode against schoolbook.
fn assert_carried_roots(x: &[u64], root_len: usize) {
    assert_sqrt_contract_with(x, root_len, sqrt_dyn);
    let mut work = x.to_vec();
    let mut floor = vec![u64::MAX; root_len];
    sqrt_dyn(&mut work, &mut floor);
    let mut above = floor.clone();
    let wrapped = inc_buf(&mut above);

    let shape = format!("root_len={root_len} x.len()={}", x.len());
    let mut work = x.to_vec();
    let mut root = vec![u64::MAX; root_len];
    sqrt_only_dyn(&mut work, &mut root);
    assert_eq!(root, floor, "sqrt_only_dyn: {shape}");

    let mut work = x.to_vec();
    let mut root = vec![u64::MAX; root_len];
    sqrt_approx_dyn(&mut work, &mut root);
    assert!(
        root == floor || (!wrapped && root == above),
        "sqrt_approx_dyn: {shape}"
    );
}

// Across the carry's cutoff and past the first stage that reads it, at every top-level shape. The
// padded ones leave the top stage a short quotient: its reciprocal can be no wider than the one
// kept below it, which needs no refinement at all.
#[test]
fn test_sqrt_carried_reciprocal_all_shapes() {
    for root_len in [
        k::sqrt::dyn_rcp_carry().max(2) - 1,
        k::sqrt::dyn_rcp_carry().max(1),
        680,
        1044,
        2084,
    ] {
        let k = root_len;
        for (i, x_len) in [
            2 * k,
            2 * k - 1,
            2 * k - k / 8,
            2 * k - k / 4,
            k + k / 2 + 1,
            k + 2,
            k + 1,
        ]
        .into_iter()
        .enumerate()
        {
            let mut x = rand_vec(x_len, 0x5_6000 + (k * 10 + i) as u64);
            x[x_len - 1] |= 1 << 63;
            assert_carried_roots(&x, k);
        }
    }
}

// Inputs that test what a resume assumes: a root whose low limbs are zero, so that the final -1
// borrows up through the limbs a lower stage divided by and its divisor is no longer a prefix;
// perfect squares, whose exact quotients take the correction path; and a saturated stage, which
// doesn't divide, so that the stage above it resumes from further back.
#[test]
fn test_sqrt_carried_reciprocal_structured_roots() {
    for k in [700usize, 1500] {
        let mut y = rand_vec(k, 0x5_7000 + k as u64);
        y[k - 1] |= 1 << 63;
        let mut cases = vec![square_plus(&y, 2 * k, 0), square_plus(&y, 2 * k, -1)];
        for zeros in [k / 2, 3 * k / 4, 7 * k / 8, k - 1] {
            let mut sparse = y.clone();
            sparse[..zeros].fill(0);
            cases.push(square_plus(&sparse, 2 * k, 0));
            cases.push(square_plus(&sparse, 2 * k, -1));
        }
        // A run of all-ones root limbs, for all-ones quotient limbs.
        let mut run = y.clone();
        run[k / 3..2 * k / 3].fill(u64::MAX);
        cases.push(square_plus(&run, 2 * k, 0));
        cases.push(square_plus(&run, 2 * k, -1));
        // (t + 1)^2 - 1 in the top 2w limbs saturates the stage above the w-limb root.
        let mut w = k - (k - 1) / 2;
        while w >= 170 {
            let mut t = rand_vec(w, 0x5_8000 + (k * 10 + w) as u64);
            t[w - 1] |= 1 << 63;
            let mut x = rand_vec(2 * k, 0x5_9000 + (k * 10 + w) as u64);
            x[2 * k - 2 * w..].copy_from_slice(&square_plus(&t, 2 * w, 2));
            cases.push(x);
            w -= (w - 1) / 2;
        }
        for x in cases {
            assert_carried_roots(&x, k);
        }
    }
}

// The sqrt section's random-knob check (test_tuning.rs): every output mode, dyn and static, at
// roots from 1 to about 300 limbs and every top-level shape, against the schoolbook contracts.
// The sizes reach the leaf, the top-level cutoffs, the wrapped remainder and the carried
// reciprocal at the values the random vectors draw (up to the caps in test_tuning.rs).
#[cfg(feature = "tune")]
pub(super) fn random_knob_check(rng: &mut StdRng) {
    for root_len in [1usize, 2, 3, 4, 5, 7, 11, 16, 23, 40, 64, 130, 300] {
        let mut x_lens = vec![2 * root_len, root_len + 1, (3 * root_len).div_ceil(2) + 1];
        x_lens.retain(|&l| l > root_len && l <= 2 * root_len);
        x_lens.dedup();
        for x_len in x_lens {
            let mut x = rand_vec(x_len, rng.gen());
            x[x_len - 1] = match rng.gen_range(0..4) {
                0 => 1,
                1 => u64::MAX,
                2 => (x[x_len - 1] >> rng.gen_range(0..64)).max(1),
                _ => x[x_len - 1] | 1,
            };
            if rng.gen_range(0..8) == 0 {
                x.fill(u64::MAX); // every stage saturates
            }
            if root_len > 1 || x_len > 1 {
                assert_sqrt_contract_with(&x, root_len, sqrt_dyn);
                assert_sqrt_contract_with(&x, root_len, sqrt_static::<600>);
            }
            assert_sqrt_only_contract::<600>(&x, root_len);
        }
    }
}
