use super::{rand_vec, to_u128};
use crate::utils::utils::*;
use std::cmp::Ordering::*;

// ─── trim_lz ────────────────────────────────────────────────────────────────

#[test]
fn test_trim_lz_empty() {
    let mut v: Vec<u64> = vec![];
    trim_lz(&mut v);
    assert!(v.is_empty());
}

#[test]
fn test_trim_lz_single_zero() {
    let mut v = vec![0u64];
    trim_lz(&mut v);
    assert!(v.is_empty());
}

#[test]
fn test_trim_lz_all_zeros() {
    let mut v = vec![0u64, 0, 0];
    trim_lz(&mut v);
    assert!(v.is_empty());
}

#[test]
fn test_trim_lz_no_change() {
    let mut v = vec![1u64, 2, 3];
    trim_lz(&mut v);
    assert_eq!(v, vec![1, 2, 3]);
}

#[test]
fn test_trim_lz_trailing_zeros() {
    let mut v = vec![1u64, 2, 0];
    trim_lz(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn test_trim_lz_interior_zero_stays() {
    let mut v = vec![0u64, 1, 0];
    trim_lz(&mut v);
    assert_eq!(v, vec![0, 1]);
}

#[test]
fn test_trim_lz_single_nonzero() {
    let mut v = vec![42u64];
    trim_lz(&mut v);
    assert_eq!(v, vec![42]);
}

// ─── buf_len ────────────────────────────────────────────────────────────────

#[test]
fn test_buf_len_empty() {
    assert_eq!(buf_len(&[]), 0);
}

#[test]
fn test_buf_len_single_zero() {
    assert_eq!(buf_len(&[0]), 0);
}

#[test]
fn test_buf_len_all_zeros() {
    assert_eq!(buf_len(&[0, 0, 0]), 0);
}

#[test]
fn test_buf_len_single_nonzero() {
    assert_eq!(buf_len(&[1]), 1);
}

#[test]
fn test_buf_len_trailing_zero() {
    assert_eq!(buf_len(&[1, 0]), 1);
}

#[test]
fn test_buf_len_leading_nonzero() {
    assert_eq!(buf_len(&[0, 1]), 2);
}

#[test]
fn test_buf_len_multi() {
    assert_eq!(buf_len(&[0, 0, 1]), 3);
}

// ─── scmp ───────────────────────────────────────────────────────────────────

#[test]
fn test_scmp_positive_orders() {
    assert_eq!(scmp(false, Less), Less);
    assert_eq!(scmp(false, Equal), Equal);
    assert_eq!(scmp(false, Greater), Greater);
}

#[test]
fn test_scmp_negative_reverses() {
    assert_eq!(scmp(true, Less), Greater);
    assert_eq!(scmp(true, Equal), Equal);
    assert_eq!(scmp(true, Greater), Less);
}

// ─── push_prim2 ─────────────────────────────────────────────────────────────

#[test]
fn test_push_prim2_zero() {
    let mut v: Vec<u64> = vec![];
    push_prim2(&mut v, 0);
    assert!(v.is_empty());
}

#[test]
fn test_push_prim2_one_limb() {
    let mut v: Vec<u64> = vec![];
    push_prim2(&mut v, 1);
    assert_eq!(v, vec![1]);
}

#[test]
fn test_push_prim2_max_u64() {
    let mut v: Vec<u64> = vec![];
    push_prim2(&mut v, u64::MAX as u128);
    assert_eq!(v, vec![u64::MAX]);
}

#[test]
fn test_push_prim2_two_limbs() {
    let mut v: Vec<u64> = vec![];
    push_prim2(&mut v, u64::MAX as u128 + 1); // = 2^64
    assert_eq!(v, vec![0, 1]);
}

#[test]
fn test_push_prim2_max_u128() {
    let mut v: Vec<u64> = vec![];
    push_prim2(&mut v, u128::MAX);
    assert_eq!(v, vec![u64::MAX, u64::MAX]);
}

// ─── combine_u64 ────────────────────────────────────────────────────────────

#[test]
fn test_combine_u64_zeros() {
    assert_eq!(combine_u64(0, 0), 0u128);
}

#[test]
fn test_combine_u64_low_only() {
    assert_eq!(combine_u64(u64::MAX, 0), u64::MAX as u128);
}

#[test]
fn test_combine_u64_high_only() {
    assert_eq!(combine_u64(0, 1), 1u128 << 64);
}

#[test]
fn test_combine_u64_both_max() {
    assert_eq!(combine_u64(u64::MAX, u64::MAX), u128::MAX);
}

#[test]
fn test_combine_u64_roundtrip() {
    for val in [0u128, 1, u64::MAX as u128, 1u128 << 64, u128::MAX] {
        let lo = val as u64;
        let hi = (val >> 64) as u64;
        assert_eq!(combine_u64(lo, hi), val, "roundtrip failed for {val}");
    }
}

// ─── eq_buf ─────────────────────────────────────────────────────────────────

#[test]
fn test_eq_buf_both_empty() {
    assert!(eq_buf(&[], &[]));
}

#[test]
fn test_eq_buf_zero_vs_empty() {
    assert!(eq_buf(&[0], &[]));
    assert!(eq_buf(&[], &[0]));
}

#[test]
fn test_eq_buf_multi_zeros_vs_empty() {
    assert!(eq_buf(&[0, 0], &[]));
}

#[test]
fn test_eq_buf_same_single() {
    assert!(eq_buf(&[1], &[1]));
}

#[test]
fn test_eq_buf_leading_zero_equal() {
    // [1, 0] represents the same value as [1]
    assert!(eq_buf(&[1, 0], &[1]));
    assert!(eq_buf(&[1], &[1, 0]));
}

#[test]
fn test_eq_buf_trailing_zero_in_longer_equal() {
    assert!(eq_buf(&[0, 1, 0], &[0, 1]));
}

#[test]
fn test_eq_buf_different_values() {
    assert!(!eq_buf(&[1], &[2]));
}

#[test]
fn test_eq_buf_different_limbs() {
    // [1, 0] != [0, 1]: different actual values
    assert!(!eq_buf(&[1, 0], &[0, 1]));
}

#[test]
fn test_eq_buf_nonzero_extra_limb() {
    // [0, 1] != [1]: second limb is nonzero
    assert!(!eq_buf(&[0, 1], &[1]));
}

// ─── cmp_buf ────────────────────────────────────────────────────────────────

#[test]
fn test_cmp_buf_both_empty() {
    assert_eq!(cmp_buf(&[], &[]), Equal);
}

#[test]
fn test_cmp_buf_zero_vs_empty() {
    assert_eq!(cmp_buf(&[0], &[]), Equal);
    assert_eq!(cmp_buf(&[], &[0]), Equal);
}

#[test]
fn test_cmp_buf_nonzero_vs_empty() {
    assert_eq!(cmp_buf(&[1], &[]), Greater);
    assert_eq!(cmp_buf(&[], &[1]), Less);
}

#[test]
fn test_cmp_buf_trailing_zero_equal() {
    assert_eq!(cmp_buf(&[1, 0], &[1]), Equal);
    assert_eq!(cmp_buf(&[1], &[1, 0]), Equal);
}

#[test]
fn test_cmp_buf_greater() {
    assert_eq!(cmp_buf(&[2], &[1]), Greater);
}

#[test]
fn test_cmp_buf_less() {
    assert_eq!(cmp_buf(&[1], &[2]), Less);
}

#[test]
fn test_cmp_buf_high_limb_dominates() {
    // [0, 1] = 2^64 > u64::MAX = [u64::MAX]
    assert_eq!(cmp_buf(&[0, 1], &[u64::MAX]), Greater);
}

#[test]
fn test_cmp_buf_equal_multi() {
    assert_eq!(cmp_buf(&[1, 1], &[1, 1]), Equal);
}

#[test]
fn test_cmp_buf_multi_differs_in_low_limb() {
    assert_eq!(cmp_buf(&[2, 1], &[1, 1]), Greater);
}

// ─── signed_shl / signed_shr ────────────────────────────────────────────────

#[test]
fn test_signed_shl_zero_shift() {
    assert_eq!(signed_shl(42, 0), 42);
}

#[test]
fn test_signed_shl_positive() {
    assert_eq!(signed_shl(1, 3), 8);
}

#[test]
fn test_signed_shl_negative_becomes_shr() {
    assert_eq!(signed_shl(8, -3), 1);
}

#[test]
fn test_signed_shr_zero_shift() {
    assert_eq!(signed_shr(42, 0), 42);
}

#[test]
fn test_signed_shr_positive() {
    assert_eq!(signed_shr(8, 3), 1);
}

#[test]
fn test_signed_shr_negative_becomes_shl() {
    assert_eq!(signed_shr(1, -3), 8);
}

#[test]
fn test_signed_shl_shr_inverse() {
    // signed_shl and signed_shr are inverses for equal magnitude shifts.
    let val = 0x0F0F0F0Fu64;
    for sh in 1..=16i32 {
        assert_eq!(signed_shr(signed_shl(val, sh), sh), val, "sh={sh}");
    }
}

// ─── lsb ────────────────────────────────────────────────────────────────────

#[test]
fn test_lsb_zero_shift() {
    assert_eq!(lsb(u64::MAX, 0), 0);
}

#[test]
fn test_lsb_negative_shift() {
    assert_eq!(lsb(u64::MAX, -1), 0);
}

#[test]
fn test_lsb_one_bit() {
    assert_eq!(lsb(0xFF, 1), 1);
}

#[test]
fn test_lsb_four_bits() {
    assert_eq!(lsb(0xFF, 4), 0xF);
}

#[test]
fn test_lsb_thirty_two_bits() {
    assert_eq!(lsb(0xDEAD_BEEF_CAFE_BABEu64, 32), 0xCAFE_BABE);
}

#[test]
fn test_lsb_sixty_three_bits() {
    assert_eq!(lsb(u64::MAX, 63), 0x7FFF_FFFF_FFFF_FFFF);
}

#[test]
fn test_lsb_zero_value() {
    assert_eq!(lsb(0, 32), 0);
}

// ─── acc ────────────────────────────────────────────────────────────────────

// Addition (comp = 0)
#[test]
fn test_add_buf_basic() {
    let mut lhs = vec![5u64];
    assert!(!add_buf(&mut lhs, &[3]));
    assert_eq!(lhs, vec![8]);
}

#[test]
fn test_add_buf_carry_out() {
    let mut lhs = vec![u64::MAX];
    assert!(add_buf(&mut lhs, &[1]));
    assert_eq!(lhs, vec![0]);
}

#[test]
fn test_add_buf_carry_propagates() {
    let mut lhs = vec![u64::MAX, u64::MAX];
    assert!(add_buf(&mut lhs, &[1]));
    assert_eq!(lhs, vec![0, 0]);
}

#[test]
fn test_add_buf_carry_into_next_limb() {
    // [u64::MAX, 0] + [1] → [0, 1], no final carry
    let mut lhs = vec![u64::MAX, 0];
    assert!(!add_buf(&mut lhs, &[1]));
    assert_eq!(lhs, vec![0, 1]);
}

#[test]
fn test_add_buf_zeros() {
    let mut lhs = vec![0u64, 0];
    assert!(!add_buf(&mut lhs, &[0]));
    assert_eq!(lhs, vec![0, 0]);
}

#[test]
fn test_acc_add_empty_rhs() {
    let mut lhs = vec![u64::MAX];
    assert!(!add_buf(&mut lhs, &[]));
    assert_eq!(lhs, vec![u64::MAX]);
}

// Subtraction (comp = 1): lhs ≥ rhs
#[test]
fn test_sub_buf_basic() {
    let mut lhs = vec![5u64];
    assert!(!sub_buf(&mut lhs, &[3]));
    assert_eq!(lhs, vec![2]);
}

#[test]
fn test_sub_buf_equal() {
    let mut lhs = vec![u64::MAX];
    assert!(!sub_buf(&mut lhs, &[u64::MAX]));
    assert_eq!(lhs, vec![0]);
}

#[test]
fn test_sub_buf_borrow_propagation() {
    // [0, 1] - [1] = [u64::MAX, 0], no underflow
    let mut lhs = vec![0u64, 1];
    assert!(!sub_buf(&mut lhs, &[1]));
    assert_eq!(lhs, vec![u64::MAX, 0]);
}

// Subtraction (comp = 1): lhs < rhs — returns true (caller should twos_comp)
#[test]
fn test_sub_buf_underflow() {
    let mut lhs = vec![3u64];
    assert!(sub_buf(&mut lhs, &[5]));
    // result is in two's-complement form; twos_comp([u64::MAX - 1]) = [2]
    twos_comp(&mut lhs);
    assert_eq!(lhs, vec![2]);
}

// ─── inc / dec ──────────────────────────────────────────────────────────────

#[test]
fn test_inc_buf_from_zero() {
    let mut v = vec![0u64];
    assert!(!inc_buf(&mut v));
    assert_eq!(v, vec![1]);
}

#[test]
fn test_inc_buf_overflow_single_limb() {
    let mut v = vec![u64::MAX];
    assert!(inc_buf(&mut v));
    assert_eq!(v, vec![0]);
}

#[test]
fn test_inc_buf_carry_into_second_limb() {
    let mut v = vec![u64::MAX, 0];
    assert!(!inc_buf(&mut v));
    assert_eq!(v, vec![0, 1]);
}

#[test]
fn test_inc_buf_overflow_all_max() {
    let mut v = vec![u64::MAX, u64::MAX];
    assert!(inc_buf(&mut v));
    assert_eq!(v, vec![0, 0]);
}

#[test]
fn test_inc_buf_no_carry_needed() {
    let mut v = vec![0u64, 1];
    assert!(!inc_buf(&mut v));
    assert_eq!(v, vec![1, 1]);
}

#[test]
fn test_dec_buf_from_one() {
    let mut v = vec![1u64];
    assert!(!dec_buf(&mut v));
    assert_eq!(v, vec![0]);
}

#[test]
fn test_dec_buf_underflow_zero() {
    let mut v = vec![0u64];
    assert!(dec_buf(&mut v));
    assert_eq!(v, vec![u64::MAX]);
}

#[test]
fn test_dec_buf_borrow_propagation() {
    // [0, 1] - 1 = [u64::MAX, 0]
    let mut v = vec![0u64, 1];
    assert!(!dec_buf(&mut v));
    assert_eq!(v, vec![u64::MAX, 0]);
}

#[test]
fn test_dec_buf_no_borrow() {
    let mut v = vec![u64::MAX];
    assert!(!dec_buf(&mut v));
    assert_eq!(v, vec![u64::MAX - 1]);
}

#[test]
fn test_inc_buf_dec_buf_roundtrip() {
    for seed in [0u64, 1, 42, 999] {
        let orig = rand_vec(4, seed);
        let mut v = orig.clone();
        inc_buf(&mut v);
        dec_buf(&mut v);
        assert_eq!(v, orig, "roundtrip failed for seed {seed}");
    }
}

// ─── twos_comp ──────────────────────────────────────────────────────────────

#[test]
fn test_twos_comp_one() {
    let mut v = vec![1u64];
    twos_comp(&mut v);
    assert_eq!(v, vec![u64::MAX]);
}

#[test]
fn test_twos_comp_max() {
    let mut v = vec![u64::MAX];
    twos_comp(&mut v);
    assert_eq!(v, vec![1]);
}

#[test]
fn test_twos_comp_two_limbs() {
    // [0, 1] → [0, u64::MAX]: -(2^64) mod 2^128 = 2^128 - 2^64 = [0, u64::MAX]
    let mut v = vec![0u64, 1];
    twos_comp(&mut v);
    assert_eq!(v, vec![0, u64::MAX]);
}

#[test]
fn test_twos_comp_multi_limb() {
    // [1, 1]: value = 1 + 2^64.  negated = [u64::MAX, u64::MAX - 1]
    let mut v = vec![1u64, 1];
    twos_comp(&mut v);
    assert_eq!(v, vec![u64::MAX, u64::MAX - 1]);
}

#[test]
fn test_twos_comp_double_negation() {
    // twos_comp is its own inverse for nonzero values.
    for seed in [1u64, 7, 13, 100] {
        let orig = rand_vec(4, seed);
        // Ensure at least one nonzero limb.
        if orig.iter().all(|&x| x == 0) {
            continue;
        }
        let mut v = orig.clone();
        twos_comp(&mut v);
        twos_comp(&mut v);
        assert_eq!(v, orig, "double twos_comp failed for seed {seed}");
    }
}

// ─── shl_buf ────────────────────────────────────────────────────────────────

#[test]
fn test_shl_buf_zero_shift() {
    let mut v = vec![1u64];
    let c = shl_buf(&mut v, 0);
    assert_eq!(v, vec![1]);
    assert_eq!(c, 0);
}

#[test]
fn test_shl_buf_by_one() {
    let mut v = vec![1u64];
    let c = shl_buf(&mut v, 1);
    assert_eq!(v, vec![2]);
    assert_eq!(c, 0);
}

#[test]
fn test_shl_buf_by_63() {
    let mut v = vec![1u64];
    let c = shl_buf(&mut v, 63);
    assert_eq!(v, vec![1u64 << 63]);
    assert_eq!(c, 0);
}

#[test]
fn test_shl_buf_overflow_one_limb() {
    let mut v = vec![u64::MAX];
    let c = shl_buf(&mut v, 1);
    assert_eq!(v, vec![u64::MAX - 1]); // = 0xFFFFFFFFFFFFFFFE
    assert_eq!(c, 1);
}

#[test]
fn test_shl_buf_overflow_63() {
    let mut v = vec![u64::MAX];
    let c = shl_buf(&mut v, 63);
    assert_eq!(v, vec![1u64 << 63]);
    assert_eq!(c, u64::MAX >> 1);
}

#[test]
fn test_shl_buf_two_limbs_no_carry() {
    let mut v = vec![1u64, 0];
    let c = shl_buf(&mut v, 1);
    assert_eq!(v, vec![2, 0]);
    assert_eq!(c, 0);
}

#[test]
fn test_shl_buf_carry_between_limbs() {
    // [u64::MAX, 1] << 1 = [u64::MAX-1, 3], no overflow
    let mut v = vec![u64::MAX, 1];
    let c = shl_buf(&mut v, 1);
    assert_eq!(v, vec![u64::MAX - 1, 3]);
    assert_eq!(c, 0);
}

#[test]
fn test_shl_buf_cross_limb_32bit_shift() {
    // [1, 0] in little-endian = value 1; shifted left 32 = value 2^32 = [1<<32, 0]
    let mut v = vec![1u64, 0];
    let c = shl_buf(&mut v, 32);
    assert_eq!(v, vec![1u64 << 32, 0]);
    assert_eq!(c, 0);
}

// ─── shr_buf ────────────────────────────────────────────────────────────────

#[test]
fn test_shr_buf_zero_shift() {
    let mut v = vec![2u64];
    let c = shr_buf(&mut v, 0);
    assert_eq!(v, vec![2]);
    assert_eq!(c, 0);
}

#[test]
fn test_shr_buf_by_one_even() {
    let mut v = vec![2u64];
    let c = shr_buf(&mut v, 1);
    assert_eq!(v, vec![1]);
    assert_eq!(c, 0);
}

#[test]
fn test_shr_buf_by_one_odd_produces_carry() {
    // 1 >> 1 = 0 with carry = 1<<63 (the bit that fell off the bottom)
    let mut v = vec![1u64];
    let c = shr_buf(&mut v, 1);
    assert_eq!(v, vec![0]);
    assert_eq!(c, 1u64 << 63);
}

#[test]
fn test_shr_buf_max_by_one() {
    let mut v = vec![u64::MAX];
    let c = shr_buf(&mut v, 1);
    assert_eq!(v, vec![u64::MAX >> 1]);
    assert_eq!(c, 1u64 << 63);
}

#[test]
fn test_shr_buf_two_limbs_carry_propagates() {
    // [0, 1] >> 1 = [1<<63, 0], no carry out
    let mut v = vec![0u64, 1];
    let c = shr_buf(&mut v, 1);
    assert_eq!(v, vec![1u64 << 63, 0]);
    assert_eq!(c, 0);
}

#[test]
fn test_shr_buf_two_max_limbs() {
    // [u64::MAX, u64::MAX] >> 1 = [u64::MAX, u64::MAX >> 1], carry = 1<<63
    let mut v = vec![u64::MAX, u64::MAX];
    let c = shr_buf(&mut v, 1);
    assert_eq!(v, vec![u64::MAX, u64::MAX >> 1]);
    assert_eq!(c, 1u64 << 63);
}

// Roundtrip: shl then shr (when no overflow from shl) restores original.
#[test]
fn test_shl_shr_roundtrip() {
    // Use values that won't overflow on shl.
    for &(val, sh) in &[(1u64, 1u8), (1, 32), (3, 4), (0xFF, 8)] {
        let mut v = vec![val];
        let overflow = shl_buf(&mut v, sh);
        if overflow == 0 {
            shr_buf(&mut v, sh);
            assert_eq!(v, vec![val], "roundtrip failed: val={val}, sh={sh}");
        }
    }
}

// For random multi-limb buffers: shl then shr (no overflow) roundtrips.
#[test]
fn test_shl_shr_roundtrip_multi_limb() {
    for seed in [10u64, 20, 30] {
        // Use a value with top limb having upper bits clear so shl by 1 won't overflow.
        let mut v: Vec<u64> = rand_vec(4, seed)
            .into_iter()
            .enumerate()
            .map(|(i, x)| if i == 3 { x >> 1 } else { x })
            .collect();
        let orig = v.clone();
        let c = shl_buf(&mut v, 1);
        assert_eq!(c, 0, "unexpected overflow for seed {seed}");
        shr_buf(&mut v, 1);
        assert_eq!(v, orig, "roundtrip failed for seed {seed}");
    }
}

// ─── to_u128 helper (self-test) ─────────────────────────────────────────────

#[test]
fn test_to_u128_values() {
    assert_eq!(to_u128(&[]), 0);
    assert_eq!(to_u128(&[1]), 1);
    assert_eq!(to_u128(&[u64::MAX]), u64::MAX as u128);
    assert_eq!(to_u128(&[0, 1]), 1u128 << 64);
    assert_eq!(to_u128(&[u64::MAX, u64::MAX]), u128::MAX);
}
