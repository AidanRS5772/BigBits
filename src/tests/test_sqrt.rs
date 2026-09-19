use super::{mul_ref, rand_vec};
use crate::utils::sqrt::{
    binom_sqrt, sqrt_approx_dyn, sqrt_approx_static, sqrt_dyn, sqrt_only_dyn, sqrt_only_static,
    sqrt_static, zimmerman_sqrt_approx_dyn, zimmerman_sqrt_approx_static, zimmerman_sqrt_dyn,
    zimmerman_sqrt_only_dyn, zimmerman_sqrt_only_static, zimmerman_sqrt_static,
};
use crate::utils::utils::{add_buf, cmp_buf, dec_buf, eq_buf, inc_buf, sub_buf};
use crate::utils::ZIMMERMAN_SQRT_CUTOFF;

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
    let cutoff = ZIMMERMAN_SQRT_CUTOFF;
    for root_len in [
        1,
        2,
        3,
        cutoff - 1,
        cutoff,
        cutoff + 1,
        2 * cutoff - 1,
        2 * cutoff + 1,
    ] {
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
            assert_sqrt_only_contract::<256>(&vec![u64::MAX; x_len], root_len);
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
    const N: usize = 2 * (ZIMMERMAN_SQRT_CUTOFF + 1);
    for root_len in [
        1,
        2,
        ZIMMERMAN_SQRT_CUTOFF - 1,
        ZIMMERMAN_SQRT_CUTOFF,
        ZIMMERMAN_SQRT_CUTOFF + 1,
    ] {
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
    const N: usize = ZIMMERMAN_SQRT_CUTOFF;
    let entries: [fn(&mut [u64], &mut [u64]); 3] = [
        sqrt_static::<N>,
        sqrt_only_static::<N>,
        sqrt_approx_static::<N>,
    ];
    for root_len in [N - 1, N] {
        for entry in entries {
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
