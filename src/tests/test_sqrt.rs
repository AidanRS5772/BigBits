use super::{mul_ref, rand_vec};
use crate::utils::sqrt::binom_sqrt;
use crate::utils::utils::{cmp_buf, eq_buf, inc_buf, sub_buf};

/// Verify both outputs for
/// `shifted_x = x * B^(2 * root.len() - x.len()) = root^2 + remainder`.
fn assert_sqrt_contract(x: &[u64], root_len: usize) {
    assert!(root_len < x.len());
    assert!(x.len() <= 2 * root_len);
    assert!(x.last().copied().unwrap() >= (1 << 62));

    let shift = 2 * root_len - x.len();
    let mut shifted_x = vec![0; shift];
    shifted_x.extend_from_slice(x);

    let mut work = x.to_vec();
    let mut root = vec![u64::MAX; root_len];
    binom_sqrt(&mut work, &mut root);

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
