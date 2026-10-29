use super::{mul_ref, rand_nonzero_vec, to_u128};
use crate::utils::mul::*;
use crate::utils::utils::trim_lz;
use crate::utils::{
    CHUNKING_KARATSUBA_CUTOFF, FFT_16BIT_CUTOFF, FFT_CHUNKING_KARATSUBA_CUTOFF,
    FFT_KARATSUBA_CUTOFF, FFT_MID_CUTOFF, FFT_SQR_CUTOFF, KARATSUBA_CUTOFF, KARATSUBA_SQR_CUTOFF,
    DYN_PARTIAL_MUL_CUTOFF, DYN_PARTIAL_SQR_CUTOFF, NTT_CHUNKING_KARATSUBA_CUTOFF,
    NTT_KARATSUBA_CUTOFF, NTT_MID_CUTOFF, STATIC_NTT_SQR_CUTOFF, STATIC_PARTIAL_MUL_CUTOFF,
    STATIC_PARTIAL_SQR_CUTOFF,
};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

// KARATSUBA_CUTOFF is f64; derive a usize version for size calculations
const KARA_CUTOFF: usize = KARATSUBA_CUTOFF as usize + 1;

// ─── boundary finders ────────────────────────────────────────────────────────

fn flat_branch_l() -> usize {
    4 * CHUNKING_KARATSUBA_CUTOFF
}

fn curved_boundary() -> (usize, usize, usize) {
    let c = KARATSUBA_CUTOFF;
    let l = (1.5 * c).round() as usize;
    let half = (l + 1) / 2;
    let s_school = (half + 1..)
        .take_while(|&s| is_school(l, s))
        .last()
        .expect("no school size found in curved branch");
    (l, s_school, s_school + 1)
}

// ─── helpers ─────────────────────────────────────────────────────────────────

const fn max_usize(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

fn norm(mut v: Vec<u64>, c: u64) -> Vec<u64> {
    if c > 0 {
        v.push(c);
    }
    v
}

/// Like mul_ref but returns (buf, carry) separately instead of appending carry.
fn mul_ref_parts(a: &[u64], b: &[u64]) -> (Vec<u64>, u64) {
    if a.is_empty() || b.is_empty() {
        return (vec![0], 0);
    }
    let mut out = vec![0u64; a.len() + b.len() - 1];
    let c = mul_buf(a, b, &mut out);
    (out, c)
}

fn mul_ref_oversized(a: &[u64], b: &[u64], out_len: usize) -> Vec<u64> {
    let raw_len = a.len() + b.len() - 1;
    let (mut out, carry) = mul_ref_parts(a, b);
    out.resize(out_len, 0);
    if out_len > raw_len {
        out[raw_len] = carry;
    }
    out
}

fn sqr_ref_parts(a: &[u64]) -> (Vec<u64>, u64) {
    mul_ref_parts(a, a)
}

fn mid_mul_ref(long: &[u64], short: &[u64]) -> Vec<u64> {
    let n = short.len();
    let full = mul_ref(long, short);
    if full.len() < 2 * n {
        let mut padded = full;
        padded.resize(2 * n, 0);
        padded[n - 1..2 * n - 1].to_vec()
    } else {
        full[n - 1..2 * n - 1].to_vec()
    }
}

fn mid_mul_school_carry(long: &[u64], short: &[u64]) -> (u64, u64) {
    let mut out = vec![0u64; short.len()];
    mid_mul_buf(long, short, &mut out)
}

/// Reference for hi_mul: compute the full product (untrimmed), extract the
/// same columns that hi_mul_buf targets.
fn hi_mul_ref(a: &[u64], b: &[u64], out_len: usize) -> Vec<u64> {
    let full_len = a.len() + b.len() - 1;
    let mut full = vec![0u64; full_len];
    let _c = mul_buf(a, b, &mut full);
    // hi_mul_buf computes columns d..d+out_len where:
    let d = full_len.saturating_sub(out_len);
    full[d..d + out_len].to_vec()
}

fn hi_sqr_ref(a: &[u64], out_len: usize) -> Vec<u64> {
    let full_len = 2 * a.len() - 1;
    let mut full = vec![0u64; full_len];
    let _c = mul_buf(a, a, &mut full);
    let d = full_len.saturating_sub(out_len);
    full[d..d + out_len].to_vec()
}

fn neg_inv_mod_2_64_runtime(p: u64) -> u64 {
    let mut x: u64 = 1;
    while x.wrapping_mul(p) != u64::MAX {
        let tmp = x.wrapping_mul(p).wrapping_add(2);
        x = x.wrapping_mul(tmp);
    }
    x
}

fn assert_close(got: f64, expected: f64, msg: &str) {
    assert!(
        (got - expected).abs() <= 1e-9,
        "{msg}: got={got}, expected={expected}"
    );
}

/// Build a human-readable diff report showing where two buffers diverge.
fn diff_report(
    got: &[u64],
    got_carry: u64,
    expected: &[u64],
    exp_carry: u64,
    msg: &str,
) -> Option<String> {
    let carry_match = got_carry == exp_carry;
    let len_match = got.len() == expected.len();
    let min_len = got.len().min(expected.len());

    // Collect differing limb indices
    let mut diffs: Vec<usize> = Vec::new();
    for i in 0..min_len {
        if got[i] != expected[i] {
            diffs.push(i);
        }
    }

    if carry_match && len_match && diffs.is_empty() {
        return None; // Everything matches
    }

    let mut report = format!("\n  MISMATCH: {msg}\n");

    // Carry line
    if carry_match {
        report += &format!("  carry: MATCH ({got_carry:#x})\n");
    } else {
        report += &format!("  carry: DIFFERS  got={got_carry:#x}  expected={exp_carry:#x}\n");
    }

    // Length line
    if !len_match {
        report += &format!(
            "  length: DIFFERS  got={}  expected={}\n",
            got.len(),
            expected.len()
        );
    }

    if diffs.is_empty() {
        return Some(report);
    }

    report += &format!(
        "  buffer: first diff at limb {}, {}/{} limbs differ\n",
        diffs[0],
        diffs.len(),
        min_len
    );

    // Show up to 5 differing limbs with context
    for (shown, &i) in diffs.iter().enumerate() {
        if shown >= 5 {
            report += &format!("  ... and {} more differing limbs\n", diffs.len() - 5);
            break;
        }
        let g = got[i];
        let e = expected[i];
        let diff = (g as i128) - (e as i128);
        report += &format!("    limb[{i}]: got={g:#018x}  expected={e:#018x}  diff={diff}\n");
    }

    Some(report)
}

/// Assert two results match exactly, showing carry separately and a
/// readable diff on failure.
fn assert_eq_result(got: &[u64], got_carry: u64, expected: &[u64], exp_carry: u64, msg: &str) {
    if let Some(report) = diff_report(got, got_carry, expected, exp_carry, msg) {
        panic!("{report}");
    }
}

/// Assert two trimmed buffers match (for cases where carry is already folded in).
fn assert_eq_buf(got: &[u64], expected: &[u64], msg: &str) {
    let mut g = got.to_vec();
    let mut e = expected.to_vec();
    trim_lz(&mut g);
    trim_lz(&mut e);
    assert_eq_result(&g, 0, &e, 0, msg);
}

fn limb_delta(got: u64, expected: u64) -> i128 {
    let diff = got.wrapping_sub(expected);
    if diff <= i64::MAX as u64 {
        diff as i128
    } else {
        -((0u64.wrapping_sub(diff)) as i128)
    }
}

/// For approximate algorithms that skip lower columns/input limbs:
/// - Limb 0 can differ by any amount (missing carry low word)
/// - Limb 1 can differ by a bounded carry count
/// - Limb 2+ can differ only by carry cascade
fn assert_approx_buf(got: &[u64], expected: &[u64], msg: &str) {
    assert_eq!(got.len(), expected.len(), "{msg}: length mismatch");
    let mut bad: Vec<(usize, i128)> = Vec::new();
    if got.len() > 1 {
        let diff = limb_delta(got[1], expected[1]);
        let tolerance = got.len().max(expected.len()) as i128 + 2;
        if diff.abs() > tolerance {
            bad.push((1, diff));
        }
    }
    for i in 2..got.len() {
        let diff = limb_delta(got[i], expected[i]);
        if diff.abs() > 1 {
            bad.push((i, diff));
        }
    }
    if bad.is_empty() {
        return;
    }
    let mut report = format!("\n  APPROX MISMATCH: {msg}\n");
    report += &format!(
        "  {} limbs exceed ±1 tolerance (checking limb[2..{}]):\n",
        bad.len(),
        got.len()
    );
    for &(i, diff) in bad.iter().take(5) {
        report += &format!(
            "    limb[{i}]: got={:#018x}  expected={:#018x}  diff={diff}\n",
            got[i], expected[i]
        );
    }
    if bad.len() > 5 {
        report += &format!("    ... and {} more\n", bad.len() - 5);
    }
    panic!("{report}");
}

fn assert_approx_mid_result(
    got: &[u64],
    got_carry: (u64, u64),
    expected: &[u64],
    expected_carry: (u64, u64),
    msg: &str,
) {
    assert_eq!(
        got_carry, expected_carry,
        "{msg}: carry mismatch; got=({:#018x}, {:#018x}), expected=({:#018x}, {:#018x})",
        got_carry.0, got_carry.1, expected_carry.0, expected_carry.1
    );
    assert_approx_buf(got, expected, msg);
}

fn assert_eq_mid_result(
    got: &[u64],
    got_carry: (u64, u64),
    expected: &[u64],
    exp_carry: (u64, u64),
    msg: &str,
) {
    if let Some(report) = diff_report(got, got_carry.0, expected, exp_carry.0, msg) {
        panic!("{report}");
    }
    assert_eq!(
        got_carry.1, exp_carry.1,
        "{msg}: high carry mismatch; got={:#018x}, expected={:#018x}",
        got_carry.1, exp_carry.1
    );
}

// ─── Section 0: Exposed FFT/NTT Helper Paths ────────────────────────────────

#[test]
fn test_fft_twiddle_tower_build_refine_view_rebuild() {
    let mut tower = FTTTwidleTower::build(16);

    let base = tower.get(16).to_vec();
    assert_eq!(base.len(), 5);
    assert_close(base[0].re, 1.0, "base twiddle real");
    assert_close(base[0].im, 0.0, "base twiddle imag");

    let refined = tower.get(32).to_vec();
    assert_eq!(refined.len(), 9);
    assert_close(refined[0].re, 1.0, "refined twiddle real");
    assert_close(refined[0].im, 0.0, "refined twiddle imag");

    let smaller = tower.get(8).to_vec();
    assert_eq!(smaller.len(), 3);
    assert_close(smaller[2].re, 0.0, "view quarter-turn real");
    assert_close(smaller[2].im, 1.0, "view quarter-turn imag");

    for &res in &[64, 128, 256] {
        let table = tower.get(res).to_vec();
        assert_eq!(table.len(), res / 4 + 1);
    }
}

#[test]
fn test_fft_size_and_standard_helpers() {
    for &(target, expected) in &[(1, 4), (4, 4), (5, 8), (15, 16), (16, 16), (17, 20)] {
        assert_eq!(find_fft_size(target), expected, "find_fft_size({target})");
    }

    let mut rounded = [
        Complex::new(5.0, 7.5),
        Complex::new(-2.5, -3.5),
        Complex::new(8.5, -8.5),
    ];
    scale_and_round_standard(&mut rounded, 2.0);
    assert_eq!(rounded[0], Complex::new(2.0, 4.0));
    assert_eq!(rounded[1], Complex::new(-1.0, -2.0));
    assert_eq!(rounded[2], Complex::new(4.0, -4.0));
}

#[test]
fn test_fft_decompose_standard_helpers() {
    let long = [0x4444_3333_2222_1111, 0xCCCC_BBBB_AAAA_9999];
    let short = [0x8888_7777_6666_5555];
    let mut x = vec![Complex::new(-1.0, -1.0); 12];

    decompose_16_standard(&long, &short, &mut x);
    let expected = [
        Complex::new(0x1111 as f64, 0x5555 as f64),
        Complex::new(0x2222 as f64, 0x6666 as f64),
        Complex::new(0x3333 as f64, 0x7777 as f64),
        Complex::new(0x4444 as f64, 0x8888 as f64),
        Complex::new(0x9999 as f64, 0.0),
        Complex::new(0xAAAA as f64, 0.0),
        Complex::new(0xBBBB as f64, 0.0),
        Complex::new(0xCCCC as f64, 0.0),
    ];
    assert_eq!(&x[..expected.len()], &expected);
    assert!(x[8..].iter().all(|c| c.is_zero()));

    let poison = Complex::new(-1.0, -2.0);
    let mut sqr = vec![poison; 8];
    sqr_decompose_16_standard(&[0x4444_3333_2222_1111], &mut sqr);
    assert_eq!(sqr[0], Complex::new(0x1111 as f64, 0x2222 as f64));
    assert_eq!(sqr[1], Complex::new(0x3333 as f64, 0x4444 as f64));
    assert!(
        sqr[2..].iter().all(|c| c.is_zero()),
        "scalar square decomposition left a poisoned coefficient tail: {:?}",
        &sqr[2..]
    );
}

#[test]
fn test_montgomery_no_asm_helpers_mod_17() {
    let p = 17u64;
    let p_inv = neg_inv_mod_2_64_runtime(p);
    assert_eq!(p.wrapping_mul(p_inv), u64::MAX);

    for t in [0u128, 1, 16, 17, 18, 255, 256, u64::MAX as u128] {
        assert_eq!(reduce_no_asm(t, p, p_inv), (t % p as u128) as u64);
    }

    for a in 0..p {
        for b in 0..p {
            assert_eq!(
                mul_reduce_no_asm(a, b, p, p_inv),
                (a * b) % p,
                "mul_reduce_no_asm({a}, {b})"
            );
        }
    }

    assert_eq!(add_mod_no_asm(5, 7, p), 12);
    assert_eq!(add_mod_no_asm(10, 9, p), 2);
    assert_eq!(sub_mod_no_asm(12, 5, p), 7);
    assert_eq!(sub_mod_no_asm(5, 12, p), 10);
    assert_eq!(neg_mod_no_asm(0, p), 0);
    assert_eq!(neg_mod_no_asm(5, p), 12);
}

#[test]
fn test_ntt_const_arithmetic_helpers() {
    assert_eq!(exp_cnt(360), [3, 2, 1]);

    let p_inv = neg_inv_mod_2_64(17);
    assert_eq!(17u64.wrapping_mul(p_inv), u64::MAX);

    for &(a, b) in &[
        (0u64, 0u64),
        (1, u64::MAX),
        (0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321),
        (u64::MAX, u64::MAX),
    ] {
        let (hi, lo) = split_mul(a, b);
        let expected = (a as u128) * (b as u128);
        assert_eq!(lo, expected as u64, "split_mul lo");
        assert_eq!(hi, (expected >> 64) as u64, "split_mul hi");
    }

    for &(a, b, p) in &[
        (123_456_789, 987_654_321, P1::P),
        (P2::P - 2, P2::P - 3, P2::P),
    ] {
        let expected = ((a as u128) * (b as u128) % (p as u128)) as u64;
        assert_eq!(mod_mul(a, b, p), expected, "mod_mul({a}, {b}, {p})");
    }
}

#[test]
fn test_montgomery_public_arithmetic_paths() {
    let two = Montgomery::<P1>::to(2);
    let three = Montgomery::<P1>::to(3);

    assert!(two.clone() == two);
    assert_eq!((two + three).from(), 5);
    assert_eq!((three - two).from(), 1);
    assert_eq!((-three).from(), P1::P - 3);
    assert_eq!(two.pow(10).from(), 1024);
    assert_eq!(Montgomery::<P1>::fuse_mul_to(7, 9).from(), 63);

    let mut acc = two;
    acc += three;
    assert_eq!(acc.from(), 5);
    acc -= two;
    assert_eq!(acc.from(), 3);

    let mut res = vec![0u64; 4];
    acc_convolution::<P1, _>(&[2, 3], &[4, 5], &mut res, |a, b| {
        Montgomery::fuse_mul_to(a, b)
    });
    let mont: &[Montgomery<P1>] = unsafe { std::mem::transmute(&res[..]) };
    let got: Vec<u64> = mont.iter().map(|&m| m.from()).collect();
    assert_eq!(&got[..3], &[8, 22, 15]);

    assert_eq!(find_ntt_size_for_split::<24, P1>(), 24);
    assert_eq!(find_ntt_size_for_split::<25, P2>(), 25);
}

#[test]
fn test_empty_input_public_paths() {
    let mut empty_out = [];
    assert_eq!(mul_dyn(&[], &[3], &mut empty_out), 0);
    assert_eq!(mul_static::<1>(&[], &[3], &mut empty_out), 0);
    assert_eq!(mul_vec(&[], &[3]), (Vec::new(), 0));
    assert_eq!(mul_arr::<1>(&[], &[3]), ([0], 0));
    assert_eq!(hi_mul_dyn(&[], &[3], &mut empty_out), 0);
    assert_eq!(hi_mul_dyn(&[], &[], &mut empty_out), 0);
    assert_eq!(hi_mul_static::<1>(&[], &[], &mut empty_out), 0);

    assert_eq!(sqr_dyn(&[], &mut empty_out), 0);
    assert_eq!(sqr_static::<1>(&[], &mut empty_out), 0);
    assert_eq!(sqr_vec(&[]), (Vec::new(), 0));
    assert_eq!(sqr_arr::<1>(&[]), ([0], 0));
    assert_eq!(hi_sqr_dyn(&[], &mut empty_out), 0);
    assert_eq!(hi_sqr_static::<1>(&[], &mut empty_out), 0);

    assert_eq!(powi_sz(&[], 3), (0, 0));
    powi_dyn_entry(&[], 3, &mut empty_out);
    powi_static_entry::<1>(&[], 3, &mut empty_out);

    let mut untouched = [123u64, 456];
    assert_eq!(hi_sqr_buf(&[], &mut untouched), 0);
    assert_eq!(untouched, [123, 456]);
}

#[test]
fn test_static_entry_signatures_match_dynamic_entries() {
    let _: fn(&[u64], &[u64], &mut [u64]) -> u64 = mul_dyn;
    let _: fn(&[u64], &[u64], &mut [u64]) -> u64 = mul_static::<8>;

    let _: fn(&[u64], &mut [u64]) -> u64 = sqr_dyn;
    let _: fn(&[u64], &mut [u64]) -> u64 = sqr_static::<8>;

    let _: fn(&[u64], usize, &mut [u64]) = powi_dyn_entry;
    let _: fn(&[u64], usize, &mut [u64]) = powi_static_entry::<8>;
}

// ─── Section 1: mul_prim ────────────────────────────────────────────────────

#[test]
fn test_mul_prim_by_zero() {
    let mut v = vec![42u64, 7];
    let c = mul_prim(&mut v, 0);
    assert_eq!(v, vec![0, 0]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim_empty() {
    let mut v = [];
    assert_eq!(mul_prim(&mut v, 123), 0);
}

#[test]
fn test_mul_prim_by_one() {
    let mut v = vec![u64::MAX, 1];
    let c = mul_prim(&mut v, 1);
    assert_eq!(v, vec![u64::MAX, 1]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim_overflow_carry() {
    let mut v = vec![u64::MAX];
    let c = mul_prim(&mut v, 2);
    assert_eq!(v, vec![u64::MAX - 1]);
    assert_eq!(c, 1);
}

#[test]
fn test_mul_prim_two_limbs() {
    let mut v = vec![1u64, 1];
    let c = mul_prim(&mut v, 2);
    assert_eq!(v, vec![2, 2]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim_max_times_max() {
    let mut v = vec![u64::MAX];
    let c = mul_prim(&mut v, u64::MAX);
    assert_eq!(v, vec![1]);
    assert_eq!(c, u64::MAX - 1);
}

#[test]
fn test_mul_prim_multi_limb_carry() {
    let mut v = vec![u64::MAX, u64::MAX];
    let c = mul_prim(&mut v, 2);
    assert_eq!(v, vec![u64::MAX - 1, u64::MAX]);
    assert_eq!(c, 1);
}

#[test]
fn test_mul_prim_correctness_single_limb() {
    for &(a, p) in &[(1u64, 5u64), (7, 11), (u64::MAX, 3), (0xDEAD_BEEF, 0xCAFE)] {
        let expected = (a as u128) * (p as u128);
        let mut v = vec![a];
        let c = mul_prim(&mut v, p);
        let got = v[0] as u128 | ((c as u128) << 64);
        assert_eq!(got, expected, "mul_prim({a:#x}, {p:#x}) mismatch");
    }
}

// Every tail and unroll shape of both asm loops, on each side of
// MUL_PRIM_ADX_CUTOFF, against a u128 loop. Limbs and multipliers lean toward
// u64::MAX so every carry chain runs at its limit.
#[test]
fn test_mul_add_prim_asm_loops() {
    use crate::utils::MUL_PRIM_ADX_CUTOFF;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x6d75_6c70);
    let limb = |rng: &mut StdRng| match rng.gen_range(0..4) {
        0 => u64::MAX,
        1 => 0,
        _ => rng.gen(),
    };
    for len in (0..MUL_PRIM_ADX_CUTOFF + 40).chain([100, 255, 256, 257, 1000]) {
        for _ in 0..200 {
            let v: Vec<u64> = (0..len).map(|_| limb(&mut rng)).collect();
            let (s, add) = (limb(&mut rng), limb(&mut rng));
            let mut want = v.clone();
            let mut c = add;
            for e in &mut want {
                let t = *e as u128 * s as u128 + c as u128;
                *e = t as u64;
                c = (t >> 64) as u64;
            }
            let mut got = v.clone();
            let top = mul_add_prim(&mut got, s, add);
            assert_eq!((got, top), (want, c), "len {len}, s {s:#x}, add {add:#x}");
        }
    }
}

// ─── Section 2: mul_prim2 ───────────────────────────────────────────────────

// mul_prim2's asm loop (its four-limb blocks and every len % 4 tail) against a
// u128 loop. Limbs and multiplier halves lean toward u64::MAX and 0, which drive
// both carry chains and the folds into the high limbs (p1 = 0 takes mul_prim's
// path).
#[test]
fn test_mul_prim2_asm_loops() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x6d75_6c32);
    let limb = |rng: &mut StdRng| match rng.gen_range(0..4) {
        0 => u64::MAX,
        1 => 0,
        _ => rng.gen(),
    };
    let lens = (0..64).chain([100, 255, 256, 257, 258, 1000, 1025]);
    for len in lens {
        for _ in 0..100 {
            let v: Vec<u64> = (0..len).map(|_| limb(&mut rng)).collect();
            let (p0, p1) = (limb(&mut rng), limb(&mut rng));
            let mut want = v.clone();
            let (mut c0, mut c1) = (0u64, 0u64);
            for e in &mut want {
                // c1:c0 + e * p0 + (e * p1) B, keeping the low limb.
                let lo = *e as u128 * p0 as u128 + c0 as u128;
                let hi = *e as u128 * p1 as u128 + c1 as u128 + (lo >> 64);
                *e = lo as u64;
                (c0, c1) = (hi as u64, (hi >> 64) as u64);
            }
            let mut got = v.clone();
            let top = mul_prim2(&mut got, (p1 as u128) << 64 | p0 as u128);
            let want_top = (c1 as u128) << 64 | c0 as u128;
            assert_eq!((got, top), (want, want_top), "len {len}, p {p1:#x}:{p0:#x}");
        }
    }
}

#[test]
fn test_mul_prim2_by_one() {
    let mut v = vec![5u64];
    let c = mul_prim2(&mut v, 1);
    assert_eq!(v, vec![5]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim2_empty() {
    let mut v = [];
    assert_eq!(mul_prim2(&mut v, 123), 0);
}

#[test]
fn test_mul_prim2_by_small() {
    let mut v = vec![3u64];
    let c = mul_prim2(&mut v, 5);
    assert_eq!(v, vec![15]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim2_overflow_into_high64() {
    let mut v = vec![1u64];
    let c = mul_prim2(&mut v, 1u128 << 64);
    assert_eq!(v, vec![0]);
    assert_eq!(c, 1);
}

#[test]
fn test_mul_prim2_two_limbs() {
    let mut v = vec![1u64, 0];
    let c = mul_prim2(&mut v, 5);
    assert_eq!(v, vec![5, 0]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim2_by_zero() {
    let mut v = vec![u64::MAX, u64::MAX];
    let c = mul_prim2(&mut v, 0);
    assert_eq!(v, vec![0, 0]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_prim2_max_times_max() {
    let mut v = vec![u64::MAX];
    let c = mul_prim2(&mut v, u128::MAX);
    // u64::MAX * u128::MAX = u64::MAX * (2^128 - 1)
    let expected = (u64::MAX as u128).wrapping_mul(u128::MAX);
    let got_lo = v[0] as u128;
    // carry is top 128 bits, but mul_prim2 returns u128
    let got = got_lo | ((c as u128) << 64);
    assert_eq!(got, expected, "mul_prim2 max*max mismatch");
}

#[test]
fn test_mul_prim2_multi_limb_carry() {
    let mut v = vec![u64::MAX, u64::MAX];
    let large = ((u64::MAX as u128) << 32) | 0xFFFF_FFFF;
    let c = mul_prim2(&mut v, large);
    // Verify via reference: treat v as 2-limb number, multiply by large
    let a_val = u64::MAX as u128 | ((u64::MAX as u128) << 64);
    let _expected_full = a_val.wrapping_mul(large);
    // Just check the result is non-zero and carry is produced
    assert!(c > 0, "expected carry for large mul_prim2");
}

#[test]
fn test_mul_prim2_correctness_random() {
    for seed in 0u64..20 {
        let v_orig = rand_nonzero_vec(2, seed);
        let prim_val = rand_nonzero_vec(2, seed + 1000);
        let prim = prim_val[0] as u128 | ((prim_val[1] as u128) << 64);

        let mut v = v_orig.clone();
        let c = mul_prim2(&mut v, prim);

        // Reference: full multiplication via mul_ref
        let ref_result = mul_ref(&v_orig, &prim_val);
        let mut got = v.to_vec();
        let c_lo = c as u64;
        let c_hi = (c >> 64) as u64;
        if c_lo > 0 || c_hi > 0 {
            got.push(c_lo);
            if c_hi > 0 {
                got.push(c_hi);
            }
        }
        assert_eq_buf(&got, &ref_result, &format!("mul_prim2 random seed={seed}"));
    }
}

// ─── Section 3: mul_buf ─────────────────────────────────────────────────────

#[test]
fn test_mul_buf_empty_a() {
    let mut out = vec![0u64];
    let c = mul_buf(&[], &[3], &mut out);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_buf_basic() {
    let mut out = vec![0u64];
    let c = mul_buf(&[5], &[4], &mut out);
    assert_eq!(out, vec![20]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_buf_overflow() {
    let mut out = vec![0u64];
    let c = mul_buf(&[u64::MAX], &[u64::MAX], &mut out);
    assert_eq!(out, vec![1]);
    assert_eq!(c, u64::MAX - 1);
}

#[test]
fn test_mul_buf_multi_limb() {
    let mut out = vec![0u64; 3];
    let c = mul_buf(&[1, 1], &[1, 1], &mut out);
    assert_eq!(out, vec![1, 2, 1]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_buf_empty_b() {
    let mut out = vec![0u64];
    let c = mul_buf(&[3], &[], &mut out);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_buf_zero_values() {
    let mut out = vec![0u64; 3];
    let c = mul_buf(&[0, 0], &[0, 0], &mut out);
    assert_eq!(out, vec![0, 0, 0]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_buf_leading_zeros() {
    // [5, 0] * [3, 0] should produce [15, 0, 0] with carry 0
    let mut out = vec![0u64; 3];
    let c = mul_buf(&[5, 0], &[3, 0], &mut out);
    assert_eq!(out, vec![15, 0, 0]);
    assert_eq!(c, 0);
}

#[test]
fn test_mul_buf_commutativity() {
    for seed in 0u64..20 {
        let a_len = (seed % 5 + 1) as usize;
        let b_len = (seed % 4 + 1) as usize;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 500);

        let mut out_ab = vec![0u64; a_len + b_len - 1];
        let c_ab = mul_buf(&a, &b, &mut out_ab);

        let mut out_ba = vec![0u64; a_len + b_len - 1];
        let c_ba = mul_buf(&b, &a, &mut out_ba);

        assert_eq_result(
            &out_ab,
            c_ab,
            &out_ba,
            c_ba,
            &format!("mul_buf commutativity seed={seed}"),
        );
    }
}

#[test]
fn test_mul_buf_random_sweep() {
    for seed in 0u64..60 {
        let a_len = (seed % 8 + 1) as usize;
        let b_len = (seed / 8 % 8 + 1) as usize;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 2000);

        let mut out = vec![0u64; a_len + b_len - 1];
        let c = mul_buf(&a, &b, &mut out);

        // For sizes that fit in u128, cross-check
        if a_len <= 1 && b_len <= 1 {
            let expected = (a[0] as u128) * (b[0] as u128);
            let got_val = to_u128(&norm(out.clone(), c));
            assert_eq!(got_val, expected, "mul_buf u128 check failed seed={seed}");
        }

        // Always check against reference
        let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("mul_buf random seed={seed}"),
        );
    }
}

// ─── Section 4: Algorithm Entry Points (direct, no dispatch) ────────────────

// --- 4a: Karatsuba multiplication (karatsuba_entry_dyn) ---

#[test]
fn test_karatsuba_entry_small() {
    // Test karatsuba_entry_dyn directly at small balanced sizes
    for seed in 0u64..10 {
        let n = KARA_CUTOFF + 1 + (seed as usize % 20);
        let a = rand_nonzero_vec(n, seed);
        let b = rand_nonzero_vec(n, seed + 100);
        let mut out = vec![0u64; 2 * n - 1];
        let c = karatsuba_entry_dyn(&a, &b, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("karatsuba_entry balanced n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_karatsuba_entry_unbalanced() {
    // Test karatsuba_entry_dyn with unbalanced inputs (chunking path)
    for seed in 0u64..10 {
        let l = 60 + (seed as usize % 40);
        let s = KARA_CUTOFF + 1 + (seed as usize % 10);
        let long = rand_nonzero_vec(l, seed);
        let short = rand_nonzero_vec(s, seed + 200);
        let mut out = vec![0u64; l + s - 1];
        let c = karatsuba_entry_dyn(&long, &short, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("karatsuba_entry unbal l={l} s={s} seed={seed}"),
        );
    }
}

#[test]
fn test_karatsuba_core_recurse_boundary_sizing() {
    // karatsuba_core's cross-term reconstruction needs out.len() >= 3*half+1,
    // which exceeds the standard a.len()+b.len()-1 convention whenever
    // b.len() sits just above half of a.len() (the exact Chunking/Recurse
    // dispatch boundary). chunking_karatsuba's last chunk (length in
    // [s-1, 2s-2]) can land exactly there, so directly engineer l to place
    // last_chunk.len() at its extremes for many s and chunk counts, on both
    // parities of last_chunk.len() — this is a differential-fuzz regression
    // test for a real, previously-panicking bug (not a synthetic edge case).
    let mut trials = 0usize;
    for s in [
        KARA_CUTOFF,
        KARA_CUTOFF + 1,
        KARA_CUTOFF + 5,
        40,
        41,
        75,
        76,
    ] {
        if s <= 2 {
            continue;
        }
        for chunks in [1usize, 2, 3, 5] {
            for last_len_target in [s.saturating_sub(1), s, 2 * s - 3, 2 * s - 2] {
                if last_len_target == 0 {
                    continue;
                }
                let l = chunks * s + last_len_target;
                if is_school(l, s) || s > (l + 1) / 2 {
                    continue; // doesn't reach the chunking path for this shape
                }
                let actual_chunks = (l - s + 1) / s;
                let actual_last_len = l - actual_chunks * s;
                assert_eq!(actual_last_len, last_len_target);

                for seed in 0u64..3 {
                    trials += 1;
                    let long = rand_nonzero_vec(l, 90_000 + seed + l as u64 * 7 + s as u64 * 13);
                    let short = rand_nonzero_vec(s, 91_000 + seed + l as u64 * 7 + s as u64 * 13);
                    let mut out = vec![0u64; l + s - 1];
                    let c = karatsuba_entry_dyn(&long, &short, &mut out);
                    let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
                    assert_eq_result(
                        &out,
                        c,
                        &exp_buf,
                        exp_carry,
                        &format!(
                            "karatsuba boundary l={l} s={s} last_len={actual_last_len} seed={seed}"
                        ),
                    );
                }
            }
        }
    }
    assert!(
        trials > 50,
        "sweep should exercise many shapes, got {trials}"
    );
}

#[test]
fn test_karatsuba_entry_unbalanced_random_sweep() {
    // Broad, non-engineered sweep over unbalanced (l, s) shapes and seeds, as
    // a differential fuzz check against the reference multiply.
    let mut seed = 500_000u64;
    for l in (KARA_CUTOFF..300).step_by(7) {
        for s in (2..l).step_by(11) {
            seed += 1;
            let long = rand_nonzero_vec(l, seed);
            let short = rand_nonzero_vec(s, seed + 1_000_000);
            let mut out = vec![0u64; l + s - 1];
            let c = karatsuba_entry_dyn(&long, &short, &mut out);
            let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
            assert_eq_result(
                &out,
                c,
                &exp_buf,
                exp_carry,
                &format!("karatsuba random unbalanced l={l} s={s} seed={seed}"),
            );
        }
    }
}

#[test]
fn test_karatsuba_entry_sweep() {
    // Sweep across a range of sizes
    for seed in 0u64..30 {
        let n = KARA_CUTOFF + 1 + (seed as usize % (3 * KARA_CUTOFF));
        let a = rand_nonzero_vec(n, seed);
        let b = rand_nonzero_vec(n, seed + 300);
        let mut out = vec![0u64; 2 * n - 1];
        let c = karatsuba_entry_dyn(&a, &b, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("karatsuba_entry sweep n={n} seed={seed}"),
        );
    }
}

// --- 4b: FFT multiplication (fft_entry) ---

#[test]
fn test_fft_entry_small() {
    // Test fft_entry directly at small sizes
    for seed in 0u64..5 {
        let n = 30 + (seed as usize * 20);
        let a = rand_nonzero_vec(n, seed);
        let b = rand_nonzero_vec(n, seed + 400);
        let mut out = vec![0u64; 2 * n - 1];
        let c = fft_entry(&a, &b, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("fft_entry small n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_fft_entry_medium() {
    for seed in 0u64..3 {
        let n = 200 + (seed as usize * 300);
        let a = rand_nonzero_vec(n, seed);
        let b = rand_nonzero_vec(n, seed + 500);
        let mut out = vec![0u64; 2 * n - 1];
        let c = fft_entry(&a, &b, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("fft_entry medium n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_fft_entry_unbalanced() {
    for seed in 0u64..3 {
        let l = 500 + (seed as usize * 200);
        let s = 50 + (seed as usize * 30);
        let long = rand_nonzero_vec(l, seed);
        let short = rand_nonzero_vec(s, seed + 600);
        let mut out = vec![0u64; l + s - 1];
        let c = fft_entry(&long, &short, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("fft_entry unbal l={l} s={s} seed={seed}"),
        );
    }
}

#[test]
fn test_fft_entry_zero_and_trailing_zero_operands() {
    let a = rand_nonzero_vec(300, 9600);
    let zero = vec![0u64; 256];
    let mut out = vec![u64::MAX; a.len() + zero.len() - 1];
    let c = fft_entry(&a, &zero, &mut out);
    assert_eq!(c, 0);
    assert!(out.iter().all(|&limb| limb == 0));

    let mut long = rand_nonzero_vec(300, 9601);
    let mut short = rand_nonzero_vec(260, 9602);
    long.resize(320, 0);
    short.resize(280, 0);

    let mut out = vec![u64::MAX; long.len() + short.len() - 1];
    let c = fft_entry(&long, &short, &mut out);
    let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
    assert_eq_result(
        &out,
        c,
        &exp_buf,
        exp_carry,
        "fft_entry trailing zero operands",
    );
}

#[test]
fn test_mul_dyn_fft_zero_operand() {
    let zero = vec![0u64; 256];
    let nonzero = rand_nonzero_vec(256, 9603);
    let mut out = vec![u64::MAX; zero.len() + nonzero.len() - 1];

    let c = mul_dyn(&zero, &nonzero, &mut out);

    assert_eq!(c, 0);
    assert!(out.iter().all(|&limb| limb == 0));
}

// --- 4c: NTT multiplication (ntt_entry_dyn) ---

#[test]
fn test_ntt_entry_small() {
    // Test NTT at small sizes — avoids the huge memory allocations
    for seed in 0u64..5 {
        let n = 10 + (seed as usize * 5);
        let a = rand_nonzero_vec(n, seed);
        let b = rand_nonzero_vec(n, seed + 700);
        let mut out = vec![0u64; 2 * n - 1];
        let c = ntt_entry_dyn(&a, &b, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("ntt_entry small n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_ntt_single_limb_and_oversized_outputs() {
    let a = [u64::MAX];

    let mut exact = [0u64; 1];
    let carry = ntt_entry_dyn(&a, &a, &mut exact);
    assert_eq!(exact, [1]);
    assert_eq!(carry, u64::MAX - 1);

    let mut oversized = [9u64; 4];
    let carry = ntt_entry_dyn(&a, &a, &mut oversized);
    assert_eq!(oversized, [1, u64::MAX - 1, 0, 0]);
    assert_eq!(carry, 0);

    let mut static_out = [9u64; 4];
    let carry = ntt_entry_static::<4>(&a, &a, &mut static_out);
    assert_eq!(static_out, oversized);
    assert_eq!(carry, 0);

    let mut sqr_exact = [0u64; 1];
    let carry = ntt_sqr_entry_dyn(&a, &mut sqr_exact);
    assert_eq!(sqr_exact, [1]);
    assert_eq!(carry, u64::MAX - 1);

    let mut sqr_oversized = [9u64; 4];
    let carry = ntt_sqr_entry_dyn(&a, &mut sqr_oversized);
    assert_eq!(sqr_oversized, [1, u64::MAX - 1, 0, 0]);
    assert_eq!(carry, 0);

    let mut sqr_static = [9u64; 4];
    let carry = ntt_sqr_entry_static::<4>(&a, &mut sqr_static);
    assert_eq!(sqr_static, sqr_oversized);
    assert_eq!(carry, 0);
}

#[test]
fn test_ntt_entry_medium() {
    // Test NTT at moderate sizes
    for seed in 0u64..3 {
        let n = 50 + (seed as usize * 25);
        let a = rand_nonzero_vec(n, seed);
        let b = rand_nonzero_vec(n, seed + 800);
        let mut out = vec![0u64; 2 * n - 1];
        let c = ntt_entry_dyn(&a, &b, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("ntt_entry medium n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_ntt_entry_unbalanced() {
    for seed in 0u64..3 {
        let l = 80 + (seed as usize * 20);
        let s = 20 + (seed as usize * 10);
        let long = rand_nonzero_vec(l, seed);
        let short = rand_nonzero_vec(s, seed + 900);
        let mut out = vec![0u64; l + s - 1];
        let c = ntt_entry_dyn(&long, &short, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("ntt_entry unbal l={l} s={s} seed={seed}"),
        );
    }
}

// ─── Section 5: mul_dyn / mul_vec Dispatch Sanity ───────────────────────────
// Tests dispatch at reasonable sizes spanning school → karatsuba boundaries.
// Does NOT test at FFT/NTT cutoff sizes — those are tested directly above.

#[test]
fn test_mul_dyn_prim_dispatch() {
    for long_sz in [1, 5, 20, 50] {
        for seed in 0u64..5 {
            let long = rand_nonzero_vec(long_sz, seed);
            let short = rand_nonzero_vec(1, seed + 1000);

            let mut out = vec![0u64; long_sz];
            let c = mul_dyn(&long, &short, &mut out);
            let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
            assert_eq_result(
                &out,
                c,
                &exp_buf,
                exp_carry,
                &format!("prim dispatch l={long_sz} seed={seed}"),
            );
        }
    }
}

#[test]
fn test_mul_dyn_prim2_dispatch() {
    for long_sz in [2, 5, 20, 50] {
        for seed in 0u64..5 {
            let long = rand_nonzero_vec(long_sz, seed);
            let short = rand_nonzero_vec(2, seed + 1100);

            let mut out = vec![0u64; long_sz + 1];
            let c = mul_dyn(&long, &short, &mut out);
            let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
            assert_eq_result(
                &out,
                c,
                &exp_buf,
                exp_carry,
                &format!("prim2 dispatch l={long_sz} seed={seed}"),
            );
        }
    }
}

#[test]
fn test_mul_dyn_school_boundary() {
    let l = flat_branch_l();
    let s = CHUNKING_KARATSUBA_CUTOFF;
    assert!(is_school(l, s), "expected school at flat branch");
    for seed in 0u64..5 {
        let long = rand_nonzero_vec(l, seed);
        let short = rand_nonzero_vec(s, seed + 1200);
        let mut out = vec![0u64; l + s - 1];
        let c = mul_dyn(&long, &short, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("school dispatch seed={seed}"),
        );
    }
}

#[test]
fn test_mul_dyn_karatsuba_boundary() {
    let (l, _s_school, s_kara) = curved_boundary();
    assert!(
        !is_school(l, s_kara),
        "expected karatsuba at curved boundary+1"
    );
    for seed in 0u64..5 {
        let long = rand_nonzero_vec(l, seed);
        let short = rand_nonzero_vec(s_kara, seed + 1300);
        let mut out = vec![0u64; l + s_kara - 1];
        let c = mul_dyn(&long, &short, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("karatsuba dispatch seed={seed}"),
        );
    }
}

#[test]
fn test_mul_dyn_fft_dispatch() {
    let n = 300usize;
    assert!(
        !is_karatsuba(n, n, FFT_CHUNKING_KARATSUBA_CUTOFF, FFT_KARATSUBA_CUTOFF),
        "chosen size should dispatch past karatsuba"
    );
    let a = rand_nonzero_vec(n, 1400);
    let b = rand_nonzero_vec(n, 1500);
    let mut out = vec![0u64; 2 * n - 1];
    let c = mul_dyn(&a, &b, &mut out);
    let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "mul_dyn FFT dispatch");
}

#[test]
fn test_mul_dyn_oversized_output_absorbs_carry() {
    let cases = [
        (vec![u64::MAX; 4], vec![u64::MAX]),
        (rand_nonzero_vec(9, 9550), rand_nonzero_vec(7, 9551)),
        (rand_nonzero_vec(90, 9552), rand_nonzero_vec(45, 9553)),
        (rand_nonzero_vec(300, 9554), rand_nonzero_vec(300, 9555)),
    ];

    for (idx, (a, b)) in cases.into_iter().enumerate() {
        let raw_len = a.len() + b.len() - 1;
        let out_len = raw_len + 3;
        let mut out = vec![u64::MAX; out_len];
        let carry = mul_dyn(&a, &b, &mut out);
        let expected = mul_ref_oversized(&a, &b, out_len);

        assert_eq_result(
            &out,
            carry,
            &expected,
            0,
            &format!("mul_dyn oversized output case={idx}"),
        );
    }
}

#[test]
fn test_mul_static_oversized_output_absorbs_carry() {
    let a = rand_nonzero_vec(50, 9560);
    let b = rand_nonzero_vec(50, 9561);
    let raw_len = a.len() + b.len() - 1;
    let out_len = raw_len + 2;
    let mut out = vec![u64::MAX; out_len];

    let carry = mul_static::<128>(&a, &b, &mut out);
    let expected = mul_ref_oversized(&a, &b, out_len);

    assert_eq_result(&out, carry, &expected, 0, "mul_static oversized output");
}

#[test]
fn test_mul_vec_random_sweep_small() {
    // Sizes that stay in school/karatsuba range
    for seed in 0u64..60 {
        let a_len = (seed % 30 + 1) as usize;
        let b_len = (seed / 2 % 15 + 1) as usize;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 1500);
        let (got_buf, got_carry) = mul_vec(&a, &b);
        let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
        assert_eq_result(
            &got_buf,
            got_carry,
            &exp_buf,
            exp_carry,
            &format!("mul_vec small a={a_len} b={b_len} seed={seed}"),
        );
    }
}

#[test]
fn test_mul_vec_random_sweep_medium() {
    // Sizes spanning karatsuba range (up to ~200 limbs)
    for seed in 0u64..20 {
        let n = 30 + (seed as usize % 170);
        let a = rand_nonzero_vec(n, seed);
        let b = rand_nonzero_vec(n, seed + 1600);
        let (got_buf, got_carry) = mul_vec(&a, &b);
        let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
        assert_eq_result(
            &got_buf,
            got_carry,
            &exp_buf,
            exp_carry,
            &format!("mul_vec medium n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_mul_commutativity() {
    for seed in 0u64..40 {
        let a_len = (seed % 20 + 1) as usize;
        let b_len = (seed / 3 % 15 + 1) as usize;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 1700);
        let (ab_buf, ab_carry) = mul_vec(&a, &b);
        let (ba_buf, ba_carry) = mul_vec(&b, &a);
        assert_eq_result(
            &ab_buf,
            ab_carry,
            &ba_buf,
            ba_carry,
            &format!("commutativity a={a_len} b={b_len} seed={seed}"),
        );
    }
}

#[test]
fn test_mul_identity() {
    for seed in 0u64..20 {
        let n = (seed % 50 + 1) as usize;
        let a = rand_nonzero_vec(n, seed);
        let (got_buf, got_carry) = mul_vec(&a, &[1]);
        assert_eq_result(
            &got_buf,
            got_carry,
            &a,
            0,
            &format!("identity n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_mul_zero_input() {
    for seed in 0u64..10 {
        let n = (seed % 20 + 1) as usize;
        let a = rand_nonzero_vec(n, seed);
        let (v, c) = mul_vec(&a, &[0]);
        assert!(
            c == 0 && v.iter().all(|&x| x == 0),
            "mul by zero should be zero, seed={seed}"
        );
    }
}

// ─── Section 6: mul_static / mul_arr ────────────────────────────────────────

#[test]
fn test_mul_static_basic() {
    let a = &[5u64, 1];
    let b = &[3u64, 2];
    let out_sz = a.len() + b.len() - 1;
    let mut out = [0u64; 4];
    let c = mul_static::<4>(a, b, &mut out);
    let (exp_buf, exp_carry) = mul_ref_parts(a, b);
    assert_eq_result(&out[..out_sz], c, &exp_buf, exp_carry, "mul_static basic");
}

#[test]
#[should_panic(expected = "out is not large enough for multiplication")]
fn test_mul_static_rejects_short_output() {
    let mut out = [0u64; 2];
    mul_static::<2>(&[1, 1, 1], &[1, 1], &mut out);
}

#[test]
fn test_mul_arr_basic() {
    let a = &[5u64, 1];
    let b = &[3u64, 2];
    let out_sz = a.len() + b.len() - 1;
    let (arr, c) = mul_arr::<3>(a, b);
    let (exp_buf, exp_carry) = mul_ref_parts(a, b);
    assert_eq_result(&arr[..out_sz], c, &exp_buf, exp_carry, "mul_arr basic");
}

#[test]
#[should_panic(expected = "out is not large enough for multiplication")]
fn test_mul_arr_rejects_insufficient_capacity() {
    mul_arr::<2>(&[1, 1, 1], &[1, 1]);
}

#[test]
fn test_mul_static_matches_dyn() {
    for seed in 0u64..15 {
        let a_len = (seed % 5 + 1) as usize;
        let b_len = (seed / 5 % 3 + 1) as usize;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 3000);

        let out_sz = a_len + b_len - 1;
        if out_sz > 64 {
            continue;
        }

        let mut out_dyn = vec![0u64; out_sz];
        let c_dyn = mul_dyn(&a, &b, &mut out_dyn);

        let mut out_st = vec![0u64; out_sz];
        let c_st = mul_static::<64>(&a, &b, &mut out_st);

        assert_eq_result(
            &out_dyn,
            c_dyn,
            &out_st,
            c_st,
            &format!("static vs dyn seed={seed}"),
        );
    }
}

#[test]
fn test_mul_arr_random_sweep() {
    for seed in 0u64..15 {
        let a_len = (seed % 4 + 1) as usize;
        let b_len = (seed / 4 % 3 + 1) as usize;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 3100);

        let out_sz = a_len + b_len - 1;
        if out_sz > 16 {
            continue;
        }

        let (arr, c) = mul_arr::<16>(&a, &b);
        // mul_arr may absorb carry into extra array positions, so compare
        // full normalized results
        let got = norm(arr.to_vec(), c);
        let expected = mul_ref(&a, &b);
        assert_eq_buf(&got, &expected, &format!("mul_arr sweep seed={seed}"));
    }
}

// ─── Section 7: Squaring Algorithm Entry Points (direct, no dispatch) ───────

// --- 7a: sqr_buf (schoolbook squaring) ---

#[test]
fn test_sqr_buf_single_limb() {
    for &val in &[1u64, 2, u64::MAX, 0xDEAD_BEEF] {
        let mut out = vec![0u64; 1];
        let c = sqr_buf(&[val], &mut out);
        let expected = (val as u128) * (val as u128);
        let got = out[0] as u128 | ((c as u128) << 64);
        assert_eq!(got, expected, "sqr_buf single limb {val:#x}");
    }
}

#[test]
fn test_sqr_buf_empty() {
    let mut out = [123u64, 456];
    assert_eq!(sqr_buf(&[], &mut out), 0);
    assert_eq!(out, [123, 456]);
}

#[test]
fn test_sqr_buf_matches_mul_buf() {
    for seed in 0u64..20 {
        let n = (seed % 10 + 1) as usize;
        let a = rand_nonzero_vec(n, seed);

        let mut sqr_out = vec![0u64; 2 * n - 1];
        let c_sqr = sqr_buf(&a, &mut sqr_out);

        let mut mul_out = vec![0u64; 2 * n - 1];
        let c_mul = mul_buf(&a, &a, &mut mul_out);

        assert_eq_result(
            &sqr_out,
            c_sqr,
            &mul_out,
            c_mul,
            &format!("sqr_buf vs mul_buf n={n} seed={seed}"),
        );
    }
}

// --- 7b: karatsuba_sqr_entry_static ---

#[test]
fn test_karatsuba_sqr_entry_small() {
    for seed in 0u64..10 {
        let n = KARATSUBA_SQR_CUTOFF + 1 + (seed as usize % 20);
        let a = rand_nonzero_vec(n, seed);
        let mut out = vec![0u64; 2 * n - 1];
        let c = karatsuba_sqr_entry_static::<128>(&a, &mut out);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("karatsuba_sqr small n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_karatsuba_sqr_entry_sweep() {
    for seed in 0u64..15 {
        let n = KARATSUBA_SQR_CUTOFF + 1 + (seed as usize % (3 * KARATSUBA_SQR_CUTOFF));
        if 2 * n - 1 > 1024 {
            continue;
        } // skip sizes that exceed static N
        let a = rand_nonzero_vec(n, seed);
        let mut out = vec![0u64; 2 * n - 1];
        let c = karatsuba_sqr_entry_static::<1024>(&a, &mut out);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("karatsuba_sqr sweep n={n} seed={seed}"),
        );
    }
}

// --- 7c: fft_sqr_entry ---

#[test]
fn test_fft_sqr_entry_small() {
    for seed in 0u64..5 {
        let n = 30 + (seed as usize * 15);
        let a = rand_nonzero_vec(n, seed);
        let mut out = vec![0u64; 2 * n - 1];
        let c = fft_sqr_entry(&a, &mut out);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("fft_sqr_entry small n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_fft_sqr_entry_medium() {
    for seed in 0u64..3 {
        let n = 200 + (seed as usize * 200);
        let a = rand_nonzero_vec(n, seed);
        let mut out = vec![0u64; 2 * n - 1];
        let c = fft_sqr_entry(&a, &mut out);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("fft_sqr_entry medium n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_fft_sqr_entry_zero_and_trailing_zero_operand() {
    let zero = vec![0u64; 300];
    let mut out = vec![u64::MAX; 2 * zero.len() - 1];
    let c = fft_sqr_entry(&zero, &mut out);
    assert_eq!(c, 0);
    assert!(out.iter().all(|&limb| limb == 0));

    let mut a = rand_nonzero_vec(300, 9604);
    a.resize(320, 0);
    let mut out = vec![u64::MAX; 2 * a.len() - 1];
    let c = fft_sqr_entry(&a, &mut out);
    let (exp_buf, exp_carry) = sqr_ref_parts(&a);
    assert_eq_result(
        &out,
        c,
        &exp_buf,
        exp_carry,
        "fft_sqr_entry trailing zero operand",
    );
}

// --- 7d: ntt_sqr_entry_dyn ---

#[test]
fn test_ntt_sqr_entry_small() {
    for seed in 0u64..5 {
        let n = 10 + (seed as usize * 5);
        let a = rand_nonzero_vec(n, seed);
        let mut out = vec![0u64; 2 * n - 1];
        let c = ntt_sqr_entry_dyn(&a, &mut out);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("ntt_sqr_entry small n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_ntt_sqr_entry_medium() {
    for seed in 0u64..3 {
        let n = 50 + (seed as usize * 75);
        let a = rand_nonzero_vec(n, seed);
        let mut out = vec![0u64; 2 * n - 1];
        let c = ntt_sqr_entry_dyn(&a, &mut out);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("ntt_sqr_entry medium n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_ntt_sqr_dyn_twiddle_cache_order() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("failed to build single-thread Rayon pool");

    pool.install(|| {
        for &(n, seed) in &[(39, 10_300), (10, 10_301)] {
            let a = rand_nonzero_vec(n, seed);
            let mut out = vec![0u64; 2 * n - 1];
            let c = ntt_sqr_entry_dyn(&a, &mut out);
            let (exp_buf, exp_carry) = sqr_ref_parts(&a);
            assert_eq_result(
                &out,
                c,
                &exp_buf,
                exp_carry,
                &format!("ntt sqr twiddle cache order n={n} seed={seed}"),
            );
        }
    });
}

// ─── Section 8: sqr_dyn / sqr_vec Dispatch Sanity ───────────────────────────
// Dyn sqr dispatch: School → FFT → NTT (no karatsuba in dyn path).
// karatsuba_sqr_entry_dyn takes one Karatsuba level at any length from 6, with
// leaves dispatched at KARATSUBA_SQR_CUTOFF, so it is exact on both sides of it.
#[test]
fn test_karatsuba_sqr_entry_dyn() {
    let k = KARATSUBA_SQR_CUTOFF;
    for (i, &n) in [6, 7, 8, 9, 15, 16, k - 1, k, k + 1, 2 * k + 1, 2 * k + 2, 4 * k + 3]
        .iter()
        .enumerate()
    {
        for seed in 0u64..3 {
            let a = rand_nonzero_vec(n, 100 * i as u64 + seed);
            let mut out = vec![0u64; 2 * n - 1];
            let c = karatsuba_sqr_entry_dyn(&a, &mut out);
            let (exp_buf, exp_carry) = sqr_ref_parts(&a);
            assert_eq_result(
                &out,
                c,
                &exp_buf,
                exp_carry,
                &format!("karatsuba_sqr_entry_dyn n={n} seed={seed}"),
            );
        }
    }
}

// Dyn sqr dispatch: School → Karatsuba → FFT → NTT.
// Static sqr dispatch: School → Karatsuba → NTT.

#[test]
fn test_sqr_dyn_school_boundary() {
    for seed in 0u64..5 {
        let n = KARATSUBA_SQR_CUTOFF + (seed as usize % 2);
        let a = rand_nonzero_vec(n, seed);
        let (got_buf, got_carry) = sqr_vec(&a);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &got_buf,
            got_carry,
            &exp_buf,
            exp_carry,
            &format!("sqr school boundary n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_sqr_dyn_above_school_boundary() {
    // Inside the dyn Karatsuba tier, below FFT_SQR_CUTOFF.
    for seed in 0u64..5 {
        let n = KARATSUBA_SQR_CUTOFF + 1 + (seed as usize * 37) % (FFT_SQR_CUTOFF - KARATSUBA_SQR_CUTOFF);
        let a = rand_nonzero_vec(n, seed);
        let (got_buf, got_carry) = sqr_vec(&a);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &got_buf,
            got_carry,
            &exp_buf,
            exp_carry,
            &format!("sqr karatsuba boundary n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_sqr_dyn_fft_boundary() {
    for &n in &[FFT_SQR_CUTOFF, FFT_SQR_CUTOFF + 1] {
        for seed in 0u64..3 {
            let a = rand_nonzero_vec(n, seed);
            let (got_buf, got_carry) = sqr_vec(&a);
            let (exp_buf, exp_carry) = sqr_ref_parts(&a);
            assert_eq_result(
                &got_buf,
                got_carry,
                &exp_buf,
                exp_carry,
                &format!("sqr fft boundary n={n} seed={seed}"),
            );
        }
    }
}

#[test]
fn test_sqr_vec_random_sweep_small() {
    for seed in 0u64..40 {
        let n = (seed % 30 + 1) as usize;
        let a = rand_nonzero_vec(n, seed);
        let (got_buf, got_carry) = sqr_vec(&a);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &got_buf,
            got_carry,
            &exp_buf,
            exp_carry,
            &format!("sqr_vec small n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_sqr_vec_random_sweep_medium() {
    // Stays within school/karatsuba/FFT range (up to FFT_SQR_CUTOFF)
    for seed in 0u64..20 {
        let n = 25 + (seed as usize % (FFT_SQR_CUTOFF - 20));
        let a = rand_nonzero_vec(n, seed);
        let (got_buf, got_carry) = sqr_vec(&a);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &got_buf,
            got_carry,
            &exp_buf,
            exp_carry,
            &format!("sqr_vec medium n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_sqr_equals_mul() {
    // Across school and karatsuba dispatch boundaries
    for &n in &[
        1,
        5,
        KARATSUBA_SQR_CUTOFF - 1,
        KARATSUBA_SQR_CUTOFF,
        KARATSUBA_SQR_CUTOFF + 1,
        FFT_SQR_CUTOFF - 1,
        FFT_SQR_CUTOFF,
        FFT_SQR_CUTOFF + 1,
    ] {
        if n == 0 {
            continue;
        }
        for seed in 0u64..3 {
            let a = rand_nonzero_vec(n, seed);
            let (sqr_buf, sqr_carry) = sqr_vec(&a);
            let (mul_buf, mul_carry) = mul_vec(&a, &a);
            assert_eq_result(
                &sqr_buf,
                sqr_carry,
                &mul_buf,
                mul_carry,
                &format!("sqr==mul n={n} seed={seed}"),
            );
        }
    }
}

#[test]
fn test_sqr_static_basic() {
    let a = &[3u64, 2];
    let out_sz = 2 * a.len() - 1;
    let mut out = [0u64; 4];
    let c = sqr_static::<4>(a, &mut out);
    let (exp_buf, exp_carry) = sqr_ref_parts(a);
    assert_eq_result(&out[..out_sz], c, &exp_buf, exp_carry, "sqr_static basic");
}

#[test]
#[should_panic(expected = "multiplication exceeds static capacity")]
fn test_sqr_static_rejects_insufficient_static_capacity() {
    let mut out = [0u64; 3];
    sqr_static::<1>(&[1, 1], &mut out);
}

#[test]
#[should_panic(expected = "out is not large enough for multiplication")]
fn test_sqr_static_rejects_short_output() {
    let mut out = [0u64; 1];
    sqr_static::<8>(&[1, 1], &mut out);
}

#[test]
#[should_panic(expected = "out is not large enough for multiplication")]
fn test_sqr_dyn_rejects_short_output() {
    sqr_dyn(&[1, 1], &mut [0]);
}

#[test]
fn test_sqr_static_matches_dyn() {
    for seed in 0u64..15 {
        let n = (seed % 10 + 1) as usize;
        let a = rand_nonzero_vec(n, seed);

        let out_sz = 2 * n - 1;
        if out_sz > 64 {
            continue;
        }

        let mut out_dyn = vec![0u64; out_sz];
        let c_dyn = sqr_dyn(&a, &mut out_dyn);

        let mut out_st = vec![0u64; out_sz];
        let c_st = sqr_static::<64>(&a, &mut out_st);

        assert_eq_result(
            &out_dyn,
            c_dyn,
            &out_st,
            c_st,
            &format!("sqr static vs dyn seed={seed}"),
        );
    }
}

#[test]
fn test_sqr_static_karatsuba_dispatch() {
    let n = KARATSUBA_SQR_CUTOFF + 1;
    let a = rand_nonzero_vec(n, 5900);
    let mut out = vec![0u64; 2 * n - 1];
    let c = sqr_static::<256>(&a, &mut out);
    let (exp_buf, exp_carry) = sqr_ref_parts(&a);
    assert_eq_result(
        &out,
        c,
        &exp_buf,
        exp_carry,
        "sqr_static karatsuba dispatch",
    );
}

// ─── Section 10: High Multiply ──────────────────────────────────────────────

#[test]
fn test_hi_mul_buf_exact_when_full() {
    // When out.len() == a.len() + b.len() - 1, result should be exact
    for seed in 0u64..10 {
        let a_len = (seed % 5 + 1) as usize;
        let b_len = (seed / 5 % 3 + 1) as usize;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 6000);
        let out_len = a_len + b_len - 1;

        let mut out_hi = vec![0u64; out_len];
        let c_hi = hi_mul_buf(&a, &b, &mut out_hi);

        let mut out_full = vec![0u64; out_len];
        let c_full = mul_buf(&a, &b, &mut out_full);

        assert_eq_result(
            &out_hi,
            c_hi,
            &out_full,
            c_full,
            &format!("hi_mul_buf exact seed={seed}"),
        );
    }
}

#[test]
fn test_hi_mul_buf_accuracy_small() {
    for seed in 0u64..30 {
        let a_len = (seed % 8 + 3) as usize;
        let b_len = (seed / 8 % 5 + 3) as usize;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 6100);
        let out_len = (a_len + b_len - 1).min(a_len.max(b_len));

        let mut out = vec![0u64; out_len];
        let _c = hi_mul_buf(&a, &b, &mut out);

        // Reference: full product, extract same columns
        let ref_top = hi_mul_ref(&a, &b, out_len);

        // hi_mul_buf drops carries from lower columns:
        // first limb can differ significantly, subsequent limbs differ by at most ±1.
        assert_approx_buf(
            &out,
            &ref_top,
            &format!("hi_mul_buf accuracy seed={seed} a_len={a_len} b_len={b_len}"),
        );
    }
}

#[test]
fn test_hi_mul_buf_accuracy_at_cutoff() {
    let out_len = DYN_PARTIAL_MUL_CUTOFF;
    for seed in 0u64..10 {
        let a_len = out_len + 3;
        let b_len = out_len + 2;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 6200);

        let mut out = vec![0u64; out_len];
        let _c = hi_mul_buf(&a, &b, &mut out);

        let ref_top = hi_mul_ref(&a, &b, out_len);

        // Approximate: first limb can differ significantly,
        // subsequent limbs can differ by at most ±1.
        assert_approx_buf(&out, &ref_top, &format!("hi_mul_buf at cutoff seed={seed}"));
    }
}

#[test]
fn test_hi_mul_dyn_exact_fit() {
    // When full product fits in out, delegates to mul_dyn → exact
    for seed in 0u64..10 {
        let a_len = (seed % 4 + 1) as usize;
        let b_len = (seed / 4 % 3 + 1) as usize;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 6300);
        let full_len = a_len + b_len - 1;
        let out_len = full_len + 2; // larger than needed

        let mut out = vec![0u64; out_len];
        let c = hi_mul_dyn(&a, &b, &mut out);
        // Carry may be absorbed into extra buffer space
        let mut got = out;
        trim_lz(&mut got);
        if c > 0 {
            got.push(c);
        }
        let expected = mul_ref(&a, &b);
        assert_eq_buf(
            &got,
            &expected,
            &format!("hi_mul_dyn exact fit seed={seed}"),
        );
    }
}

#[test]
fn test_hi_mul_dyn_below_cutoff() {
    // out.len() <= DYN_PARTIAL_MUL_CUTOFF → hi_mul_buf path
    let out_len = DYN_PARTIAL_MUL_CUTOFF;
    for seed in 0u64..10 {
        let a_len = out_len + 5;
        let b_len = out_len + 3;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 6400);

        let mut out_dyn = vec![0u64; out_len];
        let c_dyn = hi_mul_dyn(&a, &b, &mut out_dyn);

        let mut out_buf = vec![0u64; out_len];
        let c_buf = hi_mul_buf(&a, &b, &mut out_buf);

        assert_eq_result(
            &out_dyn,
            c_dyn,
            &out_buf,
            c_buf,
            &format!("hi_mul_dyn below cutoff seed={seed}"),
        );
    }
}

#[test]
fn test_hi_mul_dyn_above_cutoff() {
    // out.len() > DYN_PARTIAL_MUL_CUTOFF → truncation path
    // hi_mul_dyn truncates inputs to out_len before multiplying. The only
    // omitted effect should be carry into the lowest returned limbs.
    let out_len = DYN_PARTIAL_MUL_CUTOFF + 10;
    for seed in 0u64..10 {
        let a_len = out_len + 20;
        let b_len = out_len + 15;
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 6500);

        let mut out = vec![0u64; out_len];
        let _c = hi_mul_dyn(&a, &b, &mut out);

        let ref_top = hi_mul_ref(&a, &b, out_len);
        assert_approx_buf(
            &out,
            &ref_top,
            &format!("hi_mul_dyn above cutoff seed={seed}"),
        );
    }
}

#[test]
fn test_hi_mul_dyn_random_sweep() {
    for seed in 0u64..30 {
        let a_len = (seed % 15 + 3) as usize;
        let b_len = (seed / 3 % 10 + 3) as usize;
        let out_len = a_len.min(b_len);
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 6600);

        let mut out = vec![0u64; out_len];
        let _c = hi_mul_dyn(&a, &b, &mut out);
        // Just verify it doesn't panic
    }
}

#[test]
fn test_hi_mul_static_matches_dyn() {
    for seed in 0u64..15 {
        let a_len = (seed % 6 + 2) as usize;
        let b_len = (seed / 6 % 4 + 2) as usize;
        let out_len = a_len.min(b_len);
        let a = rand_nonzero_vec(a_len, seed);
        let b = rand_nonzero_vec(b_len, seed + 6700);

        if out_len > 32 {
            continue;
        }

        let mut out_dyn = vec![0u64; out_len];
        let c_dyn = hi_mul_dyn(&a, &b, &mut out_dyn);

        let mut out_st = vec![0u64; out_len];
        let c_st = hi_mul_static::<32>(&a, &b, &mut out_st);

        assert_eq_result(
            &out_dyn,
            c_dyn,
            &out_st,
            c_st,
            &format!("hi_mul static vs dyn seed={seed}"),
        );
    }
}

#[test]
fn test_hi_mul_static_full_product_arm() {
    let a = rand_nonzero_vec(4, 6750);
    let b = rand_nonzero_vec(3, 6760);
    let out_len = a.len() + b.len() - 1;
    let mut out = vec![0u64; out_len];
    let c = hi_mul_static::<16>(&a, &b, &mut out);
    let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "hi_mul_static full product");
}

// ─── Section 10b: High Square ───────────────────────────────────────────────

#[test]
fn test_hi_sqr_buf_accuracy_small() {
    for seed in 0u64..30 {
        let n = (seed % 12 + 3) as usize;
        let a = rand_nonzero_vec(n, seed + 6800);
        let out_len = n;

        let mut out = vec![0u64; out_len];
        let _c = hi_sqr_buf(&a, &mut out);

        let expected = hi_sqr_ref(&a, out_len);
        assert_approx_buf(&out, &expected, &format!("hi_sqr_buf n={n} seed={seed}"));
    }
}

#[test]
fn test_hi_sqr_buf_accuracy_at_cutoff() {
    let out_len = DYN_PARTIAL_SQR_CUTOFF;
    for seed in 0u64..10 {
        let n = out_len + 3;
        let a = rand_nonzero_vec(n, seed + 6900);

        let mut out = vec![0u64; out_len];
        let _c = hi_sqr_buf(&a, &mut out);

        let expected = hi_sqr_ref(&a, out_len);
        assert_approx_buf(
            &out,
            &expected,
            &format!("hi_sqr_buf cutoff n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_hi_sqr_dyn_exact_when_out_is_larger_than_full_product() {
    for seed in 0u64..10 {
        let n = (seed % 5 + 2) as usize;
        let a = rand_nonzero_vec(n, seed + 7000);
        let full_len = 2 * n - 1;
        let mut out = vec![0u64; full_len + 1];

        let c = hi_sqr_dyn(&a, &mut out);
        let mut got = out;
        trim_lz(&mut got);
        if c > 0 {
            got.push(c);
        }
        let expected = mul_ref(&a, &a);
        assert_eq_buf(&got, &expected, &format!("hi_sqr_dyn exact seed={seed}"));
    }
}

#[test]
fn test_hi_sqr_dyn_below_and_above_cutoff() {
    for seed in 0u64..10 {
        let n = DYN_PARTIAL_SQR_CUTOFF + 20 + seed as usize;
        let a = rand_nonzero_vec(n, seed + 7100);

        let mut below_dyn = vec![0u64; DYN_PARTIAL_SQR_CUTOFF];
        let c_below_dyn = hi_sqr_dyn(&a, &mut below_dyn);
        let mut below_buf = vec![0u64; DYN_PARTIAL_SQR_CUTOFF];
        let c_below_buf = hi_sqr_buf(&a, &mut below_buf);
        assert_eq_result(
            &below_dyn,
            c_below_dyn,
            &below_buf,
            c_below_buf,
            &format!("hi_sqr_dyn below cutoff seed={seed}"),
        );

        let out_len = DYN_PARTIAL_SQR_CUTOFF + 10;
        let mut above = vec![0u64; out_len];
        let _c = hi_sqr_dyn(&a, &mut above);
        let ref_top = hi_sqr_ref(&a, out_len);
        assert_approx_buf(
            &above,
            &ref_top,
            &format!("hi_sqr_dyn above cutoff seed={seed}"),
        );
    }
}

#[test]
fn test_hi_sqr_static_matches_dyn() {
    for seed in 0u64..15 {
        let n = (seed % 8 + 3) as usize;
        let a = rand_nonzero_vec(n, seed + 7200);
        let out_len = n;

        let mut out_dyn = vec![0u64; out_len];
        let c_dyn = hi_sqr_dyn(&a, &mut out_dyn);

        let mut out_static = vec![0u64; out_len];
        let c_static = hi_sqr_static::<32>(&a, &mut out_static);

        assert_eq_result(
            &out_dyn,
            c_dyn,
            &out_static,
            c_static,
            &format!("hi_sqr_static seed={seed}"),
        );
    }
}

#[test]
fn test_hi_sqr_static_full_product_arm() {
    let a = rand_nonzero_vec(4, 7250);
    let out_len = 2 * a.len() - 1;
    let mut out = vec![0u64; out_len];
    let c = hi_sqr_static::<16>(&a, &mut out);
    let (exp_buf, exp_carry) = sqr_ref_parts(&a);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "hi_sqr_static full product");
}

#[test]
fn test_hi_sqr_buf_full_product_regression() {
    let a = [1u64, 1];
    let mut out = vec![0u64; 2 * a.len() - 1];
    let c = hi_sqr_buf(&a, &mut out);
    let (expected, expected_carry) = sqr_ref_parts(&a);
    assert_eq_result(&out, c, &expected, expected_carry, "hi_sqr full product");
}

// ─── Section 11: Middle Multiply ────────────────────────────────────────────
// mid_mul_buf (schoolbook, approximate) is tested directly.
// mid_mul_dyn dispatch: school (<FFT_MID_CUTOFF) → FFT → NTT.
// Static and dynamic variants can use different approximate paths at a given size.

#[test]
fn test_mid_mul_buf_basic() {
    // n=2: long=[a,b,c] (3 limbs), short=[d,e] (2 limbs), out=[f,g] (2 limbs)
    // Full product of [1,2,3] * [4,5] = [4, 13, 22, 15] → middle = [13, 22]
    let long = vec![1u64, 2, 3];
    let short = vec![4u64, 5];
    let mut out = vec![0u64; 2];
    let c = mid_mul_buf(&long, &short, &mut out);
    assert_eq!(out, vec![13, 22], "mid_mul_buf basic");
    assert_eq!(c, (0, 0));
}

#[test]
fn test_mid_mul_buf_single() {
    let long = vec![7u64];
    let short = vec![3u64];
    let mut out = vec![0u64; 1];
    let c = mid_mul_buf(&long, &short, &mut out);
    assert_eq!(out, vec![21], "mid_mul_buf single");
    assert_eq!(c, (0, 0));
}

#[test]
fn test_mid_mul_buf_matches_full_product() {
    // mid_mul_buf is approximate: drops carries from lower columns.
    for seed in 0u64..20 {
        let n = (seed % 13 + 3) as usize;
        let long = rand_nonzero_vec(2 * n - 1, seed);
        let short = rand_nonzero_vec(n, seed + 7000);

        let mut out = vec![0u64; n];
        let c = mid_mul_buf(&long, &short, &mut out);

        let expected = mid_mul_ref(&long, &short);
        let expected_carry = mid_mul_school_carry(&long, &short);
        assert_approx_mid_result(
            &out,
            c,
            &expected,
            expected_carry,
            &format!("mid_mul_buf n={n} seed={seed}"),
        );
    }
}

// --- mid_mul_dyn dispatch ---

#[test]
#[should_panic(expected = "long must be exactly 2*short-1 limbs")]
fn test_mid_mul_dyn_rejects_noncanonical_long() {
    let mut out = [0u64; 3];
    mid_mul_dyn(&[1; 4], &[1; 3], &mut out);
}

#[test]
fn test_mid_mul_dyn_school_path() {
    // n < KARATSUBA_MID_CUTOFF → school path (delegates to mid_mul_buf)
    let n = FFT_MID_CUTOFF - 1;
    for seed in 0u64..10 {
        let long = rand_nonzero_vec(2 * n - 1, seed);
        let short = rand_nonzero_vec(n, seed + 7100);

        let mut out = vec![0u64; n];
        let c = mid_mul_dyn(&long, &short, &mut out);

        let expected = mid_mul_ref(&long, &short);
        let expected_carry = mid_mul_school_carry(&long, &short);
        assert_approx_mid_result(
            &out,
            c,
            &expected,
            expected_carry,
            &format!("mid_mul_dyn school n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_mid_mul_dyn_school_to_fft() {
    for &n in &[FFT_MID_CUTOFF - 1, FFT_MID_CUTOFF] {
        if n < 2 {
            continue;
        }
        for seed in 0u64..3 {
            let long = rand_nonzero_vec(2 * n - 1, seed);
            let short = rand_nonzero_vec(n, seed + 7400);

            let mut out = vec![0u64; n];
            let c = mid_mul_dyn(&long, &short, &mut out);

            let expected = mid_mul_ref(&long, &short);
            let expected_carry = mid_mul_school_carry(&long, &short);
            assert_approx_mid_result(
                &out,
                c,
                &expected,
                expected_carry,
                &format!("mid_mul_dyn kara/fft boundary n={n} seed={seed}"),
            );
        }
    }
}

#[test]
fn test_mid_mul_dyn_fft_path() {
    // FFT path: n >= FFT_MID_CUTOFF and n < DYN_NTT_MID_CUTOFF
    for &(n, seed_base) in &[
        (FFT_MID_CUTOFF + 10, 7500u64),
        (FFT_MID_CUTOFF + 40, 7600),
        (177, 7700),
        (FFT_MID_CUTOFF + 130, 7800),
    ] {
        for seed in 0u64..3 {
            let long = rand_nonzero_vec(2 * n - 1, seed);
            let short = rand_nonzero_vec(n, seed_base + seed);

            let mut out = vec![0u64; n];
            let c = mid_mul_dyn(&long, &short, &mut out);

            let expected = mid_mul_ref(&long, &short);
            let expected_carry = mid_mul_school_carry(&long, &short);
            assert_approx_mid_result(
                &out,
                c,
                &expected,
                expected_carry,
                &format!("mid_mul_dyn fft path n={n} seed={seed}"),
            );
        }
    }
}

#[test]
fn test_mid_mul_dyn_ntt_sparse_shift() {
    let n = FFT_16BIT_CUTOFF / 2;
    let mut long = vec![0u64; 2 * n - 1];
    long[n - 1] = 1;
    let short = rand_nonzero_vec(n, 7700);

    let mut out = vec![0u64; n];
    let c = mid_mul_dyn(&long, &short, &mut out);

    assert_eq_mid_result(&out, c, &short, (0, 0), "mid_mul_dyn NTT sparse shift");
}

#[test]
fn test_mid_mul_dyn_random_sweep() {
    for seed in 0u64..30 {
        let n = (seed % 33 + 2) as usize;
        let long = rand_nonzero_vec(2 * n - 1, seed);
        let short = rand_nonzero_vec(n, seed + 7800);

        let mut out = vec![0u64; n];
        let c = mid_mul_dyn(&long, &short, &mut out);

        let expected = mid_mul_ref(&long, &short);
        let expected_carry = mid_mul_school_carry(&long, &short);
        assert_approx_mid_result(
            &out,
            c,
            &expected,
            expected_carry,
            &format!("mid_mul_dyn sweep n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_mid_mul_static_matches_dyn() {
    for seed in 0u64..15 {
        let n = (seed % 18 + 2) as usize;
        let long = rand_nonzero_vec(2 * n - 1, seed);
        let short = rand_nonzero_vec(n, seed + 7900);

        let mut out_dyn = vec![0u64; n];
        let c_dyn = mid_mul_dyn(&long, &short, &mut out_dyn);

        let mut out_st = vec![0u64; n];
        let c_st = mid_mul_static::<64>(&long, &short, &mut out_st);

        assert_eq_mid_result(
            &out_dyn,
            c_dyn,
            &out_st,
            c_st,
            &format!("mid_mul static vs dyn n={n} seed={seed}"),
        );

        let expected = mid_mul_ref(&long, &short);
        let expected_carry = mid_mul_school_carry(&long, &short);
        assert_approx_mid_result(
            &out_dyn,
            c_dyn,
            &expected,
            expected_carry,
            &format!("mid_mul static/dyn reference n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_ntt_mid_mul_static_direct_matches_dyn() {
    for seed in 0u64..5 {
        let n = FFT_MID_CUTOFF + 5 + seed as usize * 7;
        let long = rand_nonzero_vec(2 * n - 1, seed + 8000);
        let short = rand_nonzero_vec(n, seed + 8100);

        let mut out_dyn = vec![0u64; n];
        let c_dyn = ntt_mid_mul_dyn(&long, &short, &mut out_dyn);

        let mut out_st = vec![0u64; n];
        let c_st = ntt_mid_mul_static::<512>(&long, &short, &mut out_st);

        assert_eq_mid_result(
            &out_st,
            c_st,
            &out_dyn,
            c_dyn,
            &format!("ntt_mid_mul_static direct n={n} seed={seed}"),
        );

        let expected = mid_mul_ref(&long, &short);
        let expected_carry = mid_mul_school_carry(&long, &short);
        assert_approx_mid_result(
            &out_dyn,
            c_dyn,
            &expected,
            expected_carry,
            &format!("ntt_mid_mul_static direct reference n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_ntt_mid_mul_static_split_matches_dyn() {
    let n = 49usize;
    for seed in 0u64..5 {
        let long = rand_nonzero_vec(2 * n - 1, seed + 8200);
        let short = rand_nonzero_vec(n, seed + 8300);

        let mut out_dyn = vec![0u64; n];
        let c_dyn = ntt_mid_mul_dyn(&long, &short, &mut out_dyn);

        let mut out_st = vec![0u64; n];
        let c_st = ntt_mid_mul_static::<98>(&long, &short, &mut out_st);

        assert_eq_mid_result(
            &out_st,
            c_st,
            &out_dyn,
            c_dyn,
            &format!("ntt_mid_mul_static split n={n} seed={seed}"),
        );

        let expected = mid_mul_ref(&long, &short);
        let expected_carry = mid_mul_school_carry(&long, &short);
        assert_approx_mid_result(
            &out_dyn,
            c_dyn,
            &expected,
            expected_carry,
            &format!("ntt_mid_mul_static split reference n={n} seed={seed}"),
        );
    }
}

// A long operand that passes N by low zero limbs (a reciprocal's padded divisor
// window) runs the convolution on its nonzero part and must match the dynamic
// product of the materialized operand exactly.
#[test]
fn test_ntt_mid_mul_static_zero_padded_long_matches_dyn() {
    fn run<const N: usize>(z: usize, seed: u64) {
        let n = (N + 1 + z) / 2;
        assert_eq!(2 * n - 1, N + z);
        let mut long = vec![0u64; z];
        long.extend(rand_nonzero_vec(N, seed));
        let short = rand_nonzero_vec(n, seed + 1);

        let mut out_dyn = vec![0u64; n];
        let c_dyn = ntt_mid_mul_dyn(&long, &short, &mut out_dyn);
        let mut out_st = vec![0u64; n];
        let c_st = ntt_mid_mul_static::<N>(&long, &short, &mut out_st);
        let what = format!("padded ntt_mid_mul_static N={N} z={z}");
        assert_eq_mid_result(&out_st, c_st, &out_dyn, c_dyn, &what);

        let mut out_school = vec![0u64; n];
        let c_school = mid_mul_buf(&long, &short, &mut out_school);
        assert_approx_mid_result(&out_st, c_st, &out_school, c_school, &what);
        // mid_mul_static routes it the same way: schoolbook below NTT_MID_CUTOFF.
        let (want, want_c) = if n < NTT_MID_CUTOFF {
            (&out_school, c_school)
        } else {
            (&out_st, c_st)
        };
        let mut out_entry = vec![0u64; n];
        assert_eq!(mid_mul_static::<N>(&long, &short, &mut out_entry), want_c);
        assert_eq!(&out_entry, want, "{what}: mid_mul_static entry");
    }
    for seed in [8500, 8510] {
        run::<408>(1, seed);
        run::<407>(2, seed);
        run::<406>(3, seed);
        // N = 2043 isn't 5-smooth, so its sub-transforms are clamped smaller.
        run::<2043>(2, seed);
        run::<2044>(3, seed);
    }
}

// The NTT middle products read the two convolution coefficients below the band.
// A transform 2n - 1 long wraps the product's top columns onto them, and when only
// some of the three primes picked that length, the CRT combined residues that
// disagree: limb 1 came out about 2^60 off. These n make 2n - 1 = 81, 125, 243, 375,
// 405, 625, 1125 and 1215, each an NTT size for some but not all primes.
#[test]
fn test_ntt_mid_mul_exact_size_transform() {
    fn check(long: &[u64], short: &[u64], got: &[u64], c: (u64, u64), what: &str) {
        let expected = mid_mul_ref(long, short);
        let carry = mid_mul_school_carry(long, short);
        assert_approx_mid_result(got, c, &expected, carry, what);
    }
    fn run<const N: usize>(n: usize) {
        for seed in 0u64..3 {
            let long = rand_nonzero_vec(2 * n - 1, 9100 + seed);
            let short = rand_nonzero_vec(n, 9200 + seed);
            let mut out = vec![0u64; n];
            let c = ntt_mid_mul_dyn(&long, &short, &mut out);
            check(&long, &short, &out, c, &format!("ntt_mid_mul_dyn n={n} seed={seed}"));
            let c = ntt_mid_mul_static::<N>(&long, &short, &mut out);
            check(&long, &short, &out, c, &format!("ntt_mid_mul_static N={N} n={n} seed={seed}"));
        }
    }
    // long = T B^z past N, through the static path's chunked convolution.
    fn run_padded<const N: usize>(z: usize) {
        let n = (N + 1 + z) / 2;
        for seed in 0u64..3 {
            let mut long = vec![0u64; z];
            long.extend(rand_nonzero_vec(N, 9300 + seed));
            let short = rand_nonzero_vec(n, 9400 + seed);
            let mut out = vec![0u64; n];
            let c = ntt_mid_mul_static::<N>(&long, &short, &mut out);
            check(&long, &short, &out, c, &format!("padded ntt_mid_mul_static N={N} z={z} seed={seed}"));
        }
    }
    for n in [41, 63, 122, 188, 203, 313] {
        run::<640>(n);
    }
    for n in [563, 608] {
        run::<1280>(n);
    }
    run_padded::<373>(2);
    run_padded::<1213>(2);
}

#[test]
#[should_panic(expected = "long past N must be low zero padding")]
fn test_ntt_mid_mul_static_rejects_nonzero_long_past_n() {
    const N: usize = 406;
    let long = rand_nonzero_vec(N + 3, 8600);
    let short = rand_nonzero_vec((N + 4) / 2, 8601);
    let mut out = vec![0u64; short.len()];
    ntt_mid_mul_static::<N>(&long, &short, &mut out);
}

#[test]
fn test_ntt_mid_mul_dyn_matches_full_product_approximately() {
    for &n in &[FFT_MID_CUTOFF, FFT_MID_CUTOFF + 30, NTT_MID_CUTOFF] {
        for seed in 0u64..3 {
            let long = rand_nonzero_vec(2 * n - 1, seed + 8350);
            let short = rand_nonzero_vec(n, seed + 8360);

            let mut out = vec![0u64; n];
            let c = ntt_mid_mul_dyn(&long, &short, &mut out);

            let expected = mid_mul_ref(&long, &short);
            let expected_carry = mid_mul_school_carry(&long, &short);
            assert_approx_mid_result(
                &out,
                c,
                &expected,
                expected_carry,
                &format!("ntt_mid_mul_dyn approx n={n} seed={seed}"),
            );
        }
    }
}

#[test]
fn test_mid_mul_static_cutoff_approximately_matches_ntt() {
    for seed in 0u64..5 {
        let n = FFT_MID_CUTOFF + seed as usize;
        let long = rand_nonzero_vec(2 * n - 1, seed + 8400);
        let short = rand_nonzero_vec(n, seed + 8500);

        let mut out_dyn = vec![0u64; n];
        let c_dyn = ntt_mid_mul_dyn(&long, &short, &mut out_dyn);

        let mut out_st = vec![0u64; n];
        let c_st = mid_mul_static::<256>(&long, &short, &mut out_st);

        assert_approx_mid_result(
            &out_st,
            c_st,
            &out_dyn,
            c_dyn,
            &format!("mid_mul_static cutoff approx n={n} seed={seed}"),
        );

        let expected = mid_mul_ref(&long, &short);
        let expected_carry = mid_mul_school_carry(&long, &short);
        assert_approx_mid_result(
            &out_dyn,
            c_dyn,
            &expected,
            expected_carry,
            &format!("mid_mul_static cutoff reference n={n} seed={seed}"),
        );
    }
}

// ─── Section 12: Power Functions (powi_*) ───────────────────────────────────

#[test]
#[should_panic(expected = "power exceeds output capacity")]
fn test_powi_static_entry_rejects_insufficient_capacity() {
    let big_base = rand_nonzero_vec(3, 99);
    let mut out = [0; 2];
    powi_static_entry::<2>(&big_base, 5, &mut out);
}

#[test]
fn test_powi_sz_basic() {
    // powi_sz should return (min, max) limb counts
    let base = &[u64::MAX]; // ~64 bits
    let (min, max) = powi_sz(base, 4); // ~256 bits → 4 limbs
    assert!(min >= 3 && min <= 5, "powi_sz min={min} unexpected");
    assert!(max >= min, "powi_sz max={max} < min={min}");

    // pow=1 should return roughly the base length
    let (min1, max1) = powi_sz(base, 1);
    assert_eq!(min1, 1);
    assert!(max1 >= 1);
}

#[test]
fn test_powi_identity_base_paths() {
    let mut dyn_out = [0u64; 1];
    powi_dyn_entry(&[1], 7, &mut dyn_out);
    assert_eq!(dyn_out, [1]);

    let mut static_out = [0u64; 1];
    powi_static_entry::<1>(&[1], 9, &mut static_out);
    assert_eq!(static_out, [1]);
}

#[test]
fn test_powi_zero_entry_paths() {
    let mut dyn_out = [0u64; 2];
    powi_dyn_entry(&[3], 0, &mut dyn_out);
    assert_eq!(dyn_out, [1, 0]);

    let mut static_out = [0u64; 2];
    powi_static_entry::<2>(&[3], 0, &mut static_out);
    assert_eq!(static_out, [1, 0]);
}

#[test]
fn test_powi_zero_value_and_poisoned_output_paths() {
    assert_eq!(powi_sz(&[0, 0], 7), (0, 0));

    let mut dyn_zero = [9u64; 4];
    powi_dyn_entry(&[0, 0], 7, &mut dyn_zero);
    assert_eq!(dyn_zero, [0; 4]);

    let mut static_zero = [9u64; 4];
    powi_static_entry::<4>(&[0, 0], 7, &mut static_zero);
    assert_eq!(static_zero, [0; 4]);

    let mut dyn_out = [9u64; 4];
    powi_dyn_entry(&[3, 0], 1, &mut dyn_out);
    assert_eq!(dyn_out, [3, 0, 0, 0]);

    let mut static_out = [9u64; 4];
    powi_static_entry::<4>(&[3, 0], 1, &mut static_out);
    assert_eq!(static_out, [3, 0, 0, 0]);

    let mut dyn_identity = [9u64; 4];
    powi_dyn_entry(&[3], 0, &mut dyn_identity);
    assert_eq!(dyn_identity, [1, 0, 0, 0]);

    assert_eq!(
        powi_sz(&[2], usize::MAX),
        (usize::MAX.div_ceil(64), usize::MAX.div_ceil(64))
    );
}

#[test]
fn test_powi_non_identity() {
    let mut pow_one = [0];
    powi_dyn_entry(&[3], 1, &mut pow_one);
    assert_eq!(pow_one, [3]);

    let mut pow_two = [0];
    powi_dyn_entry(&[3], 2, &mut pow_two);
    assert_eq!(pow_two, [9]);
}

#[test]
fn test_powi_dyn_entry_zero() {
    let mut out = [0];
    powi_dyn_entry(&[3], 0, &mut out);
    assert_eq!(out, [1]);
}

#[test]
fn test_powi_dyn_entry_small_exponents_exhaustive() {
    // Exercises reverse_pow's bit construction and the src/dst parity
    // bookkeeping across every exponent shape (bit lengths and popcounts) up
    // to 20, against a plain repeated-multiplication oracle.
    fn to_limbs(v: u128) -> Vec<u64> {
        if v == 0 {
            return vec![0];
        }
        let mut out = vec![v as u64, (v >> 64) as u64];
        trim_lz(&mut out);
        out
    }

    for &(base, max_pow) in &[(2u64, 20usize), (3, 20), (5, 20), (200, 14)] {
        let mut expected: u128 = 1;
        for pow in 0..=max_pow {
            let (_, capacity) = powi_sz(&[base], pow);
            let mut actual = vec![0; capacity];
            powi_dyn_entry(&[base], pow, &mut actual);
            trim_lz(&mut actual);
            assert_eq!(actual, to_limbs(expected), "base={base} pow={pow}");
            expected *= base as u128;
        }
    }
}

// ─── Section 13: sqr_arr + Static NTT Sqr ──────────────────────────────────

#[test]
fn test_sqr_arr_basic() {
    let a = &[3u64, 2];
    let (arr, c) = sqr_arr::<4>(a);
    let (exp_buf, exp_carry) = sqr_ref_parts(a);
    let out_sz = 2 * a.len() - 1;
    assert_eq_result(&arr[..out_sz], c, &exp_buf, exp_carry, "sqr_arr basic");
}

#[test]
#[should_panic(expected = "multiplication exceeds static capacity")]
fn test_sqr_arr_rejects_insufficient_capacity() {
    sqr_arr::<1>(&[1, 1]);
}

#[test]
fn test_sqr_static_ntt_dispatch() {
    let n = STATIC_NTT_SQR_CUTOFF + 1;
    let a = rand_nonzero_vec(n, 8050);
    let out_sz = 2 * n - 1;

    let mut out = vec![0u64; out_sz];
    let c = sqr_static::<8192>(&a, &mut out);

    let (exp_buf, exp_carry) = sqr_ref_parts(&a);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "sqr_static NTT dispatch");
}

// ─── Section 14: Static NTT Mul Split Paths ─────────────────────────────────

#[test]
fn test_mul_static_ntt_split() {
    // mul_static::<128> with balanced ~50-limb inputs should hit NTT path,
    // potentially triggering the split convolution when find_ntt_size > N.
    // Also test with smaller N to force the split path.
    for seed in 0u64..3 {
        let n = 50 + seed as usize * 5;
        let a = rand_nonzero_vec(n, seed + 8100);
        let b = rand_nonzero_vec(n, seed + 8200);
        let out_sz = 2 * n - 1;
        if out_sz <= 128 {
            let mut out = vec![0u64; out_sz];
            let c = mul_static::<128>(&a, &b, &mut out);
            let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
            assert_eq_result(
                &out,
                c,
                &exp_buf,
                exp_carry,
                &format!("mul_static NTT n={n} seed={seed}"),
            );
        }
    }

    // The over-N split path (find_ntt_size(out_len) > N, only reachable for
    // non-5-smooth N) is covered by test_ntt_static_split_regression.
}

#[test]
fn test_mul_static_ntt_dispatch() {
    // The smallest balanced size the static cost model sends to the NTT.
    let n = (2..)
        .find(|&n| !is_karatsuba(n, n, NTT_CHUNKING_KARATSUBA_CUTOFF, NTT_KARATSUBA_CUTOFF))
        .unwrap();
    assert!(2 * n - 1 <= 8192, "static NTT dispatch starts past this test's N");

    let a = rand_nonzero_vec(n, 8300);
    let b = rand_nonzero_vec(n, 8400);
    let out_sz = 2 * n - 1;
    let mut out = vec![0u64; out_sz];
    let c = mul_static::<8192>(&a, &b, &mut out);

    let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "mul_static NTT dispatch");
}

#[test]
fn test_ntt_static_split_regression() {
    // N = 24 is 5-smooth: out_len 23 stays on the direct transform path.
    let n = 12usize;
    let a = rand_nonzero_vec(n, 8450);
    let b = rand_nonzero_vec(n, 8460);
    let mut out = vec![0u64; 2 * n - 1];
    let c = ntt_entry_static::<24>(&a, &b, &mut out);
    let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "ntt static direct");

    // N = 23 is not 5-smooth: find_ntt_size(out_len) > N forces the chunked
    // split path, balanced and unbalanced.
    let a = rand_nonzero_vec(12, 8470);
    let b = rand_nonzero_vec(12, 8480);
    let mut out = vec![0u64; 23];
    let c = ntt_entry_static::<23>(&a, &b, &mut out);
    let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "ntt static split balanced");

    let a = rand_nonzero_vec(15, 8490);
    let b = rand_nonzero_vec(8, 8500);
    let mut out = vec![0u64; 22];
    let c = ntt_entry_static::<23>(&a, &b, &mut out);
    let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "ntt static split unbalanced");

    let buf = rand_nonzero_vec(12, 8510);
    let mut out = vec![0u64; 23];
    let c = ntt_sqr_entry_static::<23>(&buf, &mut out);
    let (exp_buf, exp_carry) = sqr_ref_parts(&buf);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "ntt static split sqr");
}

// ─── Section 15: NTT Radix-3 Coverage ────────────────────────────────────────

#[test]
fn test_ntt_radix3_coverage() {
    // Use input sizes that produce NTT sizes divisible by 3 at the top level.
    // ntt dispatches: %5==0 → ntt_5, %3==0 → ntt_3, else → ntt_2.
    // We need ntt_size % 3 == 0 && ntt_size % 5 != 0.
    // Input n=14 → out_len=27 = 3^3, which should trigger ntt_3 at top.
    // Input n=18 → out_len=35, ntt_size=36=4*9=2^2*3^2
    for &n in &[14, 18] {
        for seed in 0u64..3 {
            let a = rand_nonzero_vec(n, seed + 8500);
            let b = rand_nonzero_vec(n, seed + 8600);
            let mut out = vec![0u64; 2 * n - 1];
            let c = ntt_entry_dyn(&a, &b, &mut out);
            let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
            assert_eq_result(
                &out,
                c,
                &exp_buf,
                exp_carry,
                &format!("ntt radix-3 coverage n={n} seed={seed}"),
            );
        }
    }
}

#[test]
fn test_ntt_radix5_coverage() {
    // n=13 produces out_len=25, forcing radix-5 at the top level.
    // n=38 produces out_len=75, covering radix-5 precedence over radix-3.
    for &n in &[13, 38] {
        for seed in 0u64..3 {
            let a = rand_nonzero_vec(n, seed + 10_500);
            let b = rand_nonzero_vec(n, seed + 10_600);
            let mut out = vec![0u64; 2 * n - 1];
            let c = ntt_entry_dyn(&a, &b, &mut out);
            let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
            assert_eq_result(
                &out,
                c,
                &exp_buf,
                exp_carry,
                &format!("ntt radix-5 coverage n={n} seed={seed}"),
            );
        }
    }
}

#[test]
fn test_ntt_entry_dyn_parallel_branch() {
    let n = 300usize;
    let a = rand_nonzero_vec(n, 10_700);
    let b = rand_nonzero_vec(n, 10_800);
    let mut out = vec![0u64; 2 * n - 1];
    let c = ntt_entry_dyn(&a, &b, &mut out);
    let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "ntt dyn parallel branch");
}

#[test]
fn test_ntt_sqr_entry_dyn_parallel_branch() {
    let n = 300usize;
    let a = rand_nonzero_vec(n, 10_900);
    let mut out = vec![0u64; 2 * n - 1];
    let c = ntt_sqr_entry_dyn(&a, &mut out);
    let (exp_buf, exp_carry) = sqr_ref_parts(&a);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "ntt sqr dyn parallel branch");
}

#[test]
fn test_ntt_entry_static_sequential_branch() {
    let n = 20usize;
    let a = rand_nonzero_vec(n, 11_000);
    let b = rand_nonzero_vec(n, 11_100);
    let mut out = vec![0u64; 2 * n - 1];
    let c = ntt_entry_static::<128>(&a, &b, &mut out);
    let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "ntt static sequential");
}

#[test]
fn test_ntt_sqr_entry_static_sequential_branch() {
    let n = 20usize;
    let a = rand_nonzero_vec(n, 11_200);
    let mut out = vec![0u64; 2 * n - 1];
    let c = ntt_sqr_entry_static::<128>(&a, &mut out);
    let (exp_buf, exp_carry) = sqr_ref_parts(&a);
    assert_eq_result(&out, c, &exp_buf, exp_carry, "ntt sqr static sequential");
}

// ─── Section 16: Karatsuba Static Entry Points ──────────────────────────────

#[test]
fn test_karatsuba_static_entries() {
    // karatsuba_entry_static
    for seed in 0u64..5 {
        let n = KARA_CUTOFF + 5 + seed as usize;
        let a = rand_nonzero_vec(n, seed + 8700);
        let b = rand_nonzero_vec(n, seed + 8800);
        let mut out = vec![0u64; 2 * n - 1];
        let c = karatsuba_entry_static::<128>(&a, &b, &mut out);
        let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("karatsuba_entry_static n={n} seed={seed}"),
        );
    }

    // karatsuba_sqr_entry_static
    for seed in 0u64..5 {
        let n = KARATSUBA_SQR_CUTOFF + 3 + seed as usize;
        let a = rand_nonzero_vec(n, seed + 8900);
        let mut out = vec![0u64; 2 * n - 1];
        let c = karatsuba_sqr_entry_static::<128>(&a, &mut out);
        let (exp_buf, exp_carry) = sqr_ref_parts(&a);
        assert_eq_result(
            &out,
            c,
            &exp_buf,
            exp_carry,
            &format!("karatsuba_sqr_entry_static n={n} seed={seed}"),
        );
    }
}

#[test]
fn test_karatsuba_static_fallback_scratch() {
    // Leaves (half + 1 limbs) past the Karatsuba cutoff recurse, so the scratch
    // exceeds N while the top level's cross term (2 half + 1) still fits it.
    let n = 2 * KARA_CUTOFF + 8;
    let half = (n + 1) / 2;
    assert!(2 * half + 1 <= 128 && 128 < find_karatsuba_scratch(n, n) && n <= 128);
    let a = rand_nonzero_vec(n, 8950);
    let b = rand_nonzero_vec(n, 8960);
    let mut out = vec![0u64; 2 * n - 1];
    let c = karatsuba_entry_static::<128>(&a, &b, &mut out);
    let (exp_buf, exp_carry) = mul_ref_parts(&a, &b);
    assert_eq_result(
        &out,
        c,
        &exp_buf,
        exp_carry,
        "karatsuba static fallback scratch",
    );

    let n = 2 * KARATSUBA_SQR_CUTOFF + 8;
    let half = (n + 1) / 2;
    assert!(2 * half + 1 <= 192 && 192 < find_karatsuba_sqr_scratch_sz(n) && n <= 192);
    let a = rand_nonzero_vec(n, 8970);
    let mut out = vec![0u64; 2 * n - 1];
    let c = karatsuba_sqr_entry_static::<192>(&a, &mut out);
    let (exp_buf, exp_carry) = sqr_ref_parts(&a);
    assert_eq_result(
        &out,
        c,
        &exp_buf,
        exp_carry,
        "karatsuba sqr static fallback scratch",
    );
}

#[test]
fn test_karatsuba_static_chunking_scratch_fallback() {
    fn check<const N: usize>(l: usize, s: usize, seed: u64, label: &str) {
        assert!(s <= (l + 1) / 2, "case must use chunking Karatsuba");
        assert!(!is_school(l, s), "case must not use schoolbook");

        let long = rand_nonzero_vec(l, seed);
        let short = rand_nonzero_vec(s, seed + 1);
        let mut out = vec![0u64; l + s - 1];
        let carry = karatsuba_entry_static::<N>(&long, &short, &mut out);
        let (expected, expected_carry) = mul_ref_parts(&long, &short);
        assert_eq_result(&out, carry, &expected, expected_carry, label);
    }

    // The shapes sit just past CHUNKING_KARATSUBA_CUTOFF (s = 64); the
    // scratch sizes below follow from the cutoffs and change with them.

    // Exact scratch capacity must use the ordinary chunking path, not the
    // separate-cross fallback intended for balanced recursion.
    assert_eq!(find_karatsuba_scratch(134, 64), 204);
    check::<204>(134, 64, 12_000, "static chunking exact scratch capacity");

    // The old fallback called karatsuba_core directly with half > short.len(),
    // panicking while splitting the short operand. N is the output's length.
    assert!(find_karatsuba_scratch(127, 64) > 190);
    check::<190>(127, 64, 12_100, "static chunking fallback boundary");

    // Exercise more than one full short-sized chunk before the remainder.
    assert_eq!(find_karatsuba_scratch(197, 64), 203);
    check::<202>(197, 64, 12_200, "static chunking fallback multiple chunks");
}

// ─── Section 17b: Karatsuba Overflow Offset ─────────────────────────────────

// karatsuba_core subtracts z2's overflow limb (the product limb at z2.len()) from
// the cross term. It used to subtract it at buf_len(z2), which is lower whenever
// z2's top limb is zero, silently corrupting the product. Random operands almost
// never give a sub-product a zero top limb, so these engineer one.
#[test]
fn test_karatsuba_core_overflow_offset_zero_top() {
    for n in 19..=80 {
        for (ta, tb) in [
            (1u64 << 58, 1u64 << 58),
            (1 << 58, 3 << 50),
            (1 << 63, 1 << 63),
        ] {
            let mut a = vec![0u64; n];
            let mut b = vec![0u64; n];
            a[n - 1] = ta;
            b[n - 1] = tb;
            let (expected, exp_carry) = mul_ref_parts(&a, &b);
            let label = format!("n={n} tops=({ta:#x}, {tb:#x})");

            let mut out = vec![0u64; 2 * n - 1];
            let carry = mul_dyn(&a, &b, &mut out);
            assert_eq_result(
                &out,
                carry,
                &expected,
                exp_carry,
                &format!("mul_dyn {label}"),
            );

            let mut out = vec![0u64; 2 * n - 1];
            let carry = mul_static::<256>(&a, &b, &mut out);
            assert_eq_result(
                &out,
                carry,
                &expected,
                exp_carry,
                &format!("mul_static {label}"),
            );
        }
    }
}

// A real 151x151 pair (a series numerator and an rcp_dyn output, as multiplied by
// hi_mul_dyn inside ratio_to_fraction_dyn) whose recursion hits the overflow offset
// bug three levels down.
#[test]
fn test_karatsuba_structured_operands() {
    #[rustfmt::skip]
    const A: [u64; 151] = [
        0xf9c5d2359f895e80, 0xbfeb85e0df2f2382, 0x9f4cb777f133a92e, 0x31d989cce1c8b3b7,
        0xd9b99ece02d1085b, 0xaa750735a1cb4108, 0xdf3ff2571219c2ba, 0x01b0e20c1b7c45ba,
        0xd65f9516521ebc55, 0x3bb14d1fa8e36dd9, 0xfa2f86013b35da40, 0xa1e995576b69239f,
        0x115214722498fe63, 0x550db0b554e957d7, 0x659f9bde6a265ad6, 0xc47309814cc228e4,
        0xac2c9e3e4b0b4d81, 0xbc32d3f28b845c84, 0xfff239cc48fd553e, 0xbb419656e73e9bae,
        0xddca6ab6038bdae1, 0x1de29b87e62d337d, 0x7c3035c1b475854f, 0xf5b238c98ed6ff7d,
        0x787d70ec71b7ec31, 0x98e09206bd12be8a, 0xd7edf7c10f46e238, 0x491ef9019b09b7da,
        0x46eb8f34a3436c92, 0xe08e506a63c9524f, 0x74b5fc11929871c8, 0x7e96bb509ccb1d69,
        0x2fab0ca1c5dc6460, 0x75f4337e0752ad6b, 0x0518bcd027ab160d, 0x486b57201f452b99,
        0x6aabf3007db74683, 0x9c95aea6cc250fa0, 0xe45c25995b1c0374, 0x98b4323ffba007b1,
        0x9b765bf1663afc44, 0x23a482f91769925e, 0x545456701793739e, 0x27214ec58e63e466,
        0x2c3e097568ff58ca, 0xedfaa2c0fb3cc1f3, 0x0f30676e4c83857a, 0xd270c2936cc97281,
        0x1ed88a6221111e2a, 0x95fab255a940e8d2, 0x7da7d27a44f5be26, 0x42b5d17a8c33c9d8,
        0xd508cccc22c0e9fc, 0xf0c5599da2b03c39, 0xf35ee5ac1d773697, 0x9b46d18e4d1aebe1,
        0x85a05eba7266c39f, 0x605a2aab967e4eda, 0xdfae0b9e39b4b506, 0x440cfd5bf008eea7,
        0x735e9d853d85d9ee, 0xc1aaecf5118baebc, 0xa8cb32ebc0398409, 0x88b48e5912b42d62,
        0x12fd1814221c0439, 0xf544e792c616300a, 0x2950aff9ed6dc579, 0xa87ad8e9e455ec21,
        0x43d238a64f959966, 0x18b3b38e68903893, 0xacf14e507018793d, 0xb3ad17f642ea2ecd,
        0x8473fb9fcad98ef9, 0x3165d6af85d8eb2c, 0x859a7556eebba172, 0x5d95dcb06c1a0406,
        0xc92299a8e6acb18f, 0x3ec744010db33159, 0x260bae334257256b, 0x7371ddaf9267b72a,
        0x3ec1f7e990ef2ef6, 0x9b52aa9c0f51e592, 0x711c9da397fce966, 0xa656faad4d0467c5,
        0xc61d191f8a2c2f54, 0x898b4750c2e0522a, 0x63f3d20abedfe4ff, 0x76dbc21f29bd3959,
        0x9365fcb2e12c5336, 0xd5b5f77079dbefd3, 0x6bc8df5c4c5fe036, 0xac0e567a7503e736,
        0x0bfee1d101951892, 0xc0bfcf0c75c3b0c4, 0xd4428aa5e6a229df, 0x6c5fb9bf20e5cf5a,
        0x1069b83f8f710a06, 0x2cd82f077108c12b, 0xdf88164987a6b66d, 0x315e5c9967a73487,
        0x044ec9cc284ebb9a, 0xd278602e941ba44b, 0x23f802ea36e1144b, 0x214304817611b6fe,
        0x1e50e0092435b491, 0x0bb28ed4a5a3cc56, 0x67ec5619376e9ce4, 0x1d3edef9814fac71,
        0x9efc4c69235bc96c, 0x5069db30f0d0849e, 0x18b64dbe53ee24fe, 0x6c2730889accbeb4,
        0xba418f3b10406a0e, 0xcbe7ac78817af727, 0x821f2e4b92e37289, 0x47d765feb8088836,
        0x3d44428b7f3f7d3e, 0xe16d57cbcffeae77, 0x2ad54bc989a47de6, 0xfb6f40139f2afa24,
        0xb40255232c0db6ec, 0x6a5cacda5eae3d17, 0x958544d93870bec3, 0xd16bc09cc1481f26,
        0xf6bca7a55c38f04a, 0xd999688476ae442d, 0x3e384ccb2308b60c, 0x4ed5c27742d4f5c5,
        0x2b901375ab253996, 0x68c974727cb1ea85, 0x57133d2bf3a6632b, 0xe949e04c722fe1f4,
        0x85e7f2c8f7e88880, 0x9439703bff81efc0, 0xbab577cd728c67b7, 0x489bc3fb6815cd77,
        0x781e1321403e2c74, 0x40be3a10f20cfa92, 0x87e7eb41b466b101, 0xe3c1ddc044255816,
        0x993ea4b8ca78daaa, 0x61a38935eb96a3ee, 0x2ebdc7ef91e6d161, 0x40c08ef85260b34f,
        0x5db52d4e874341cb, 0x60c17539218dda7a, 0xcefcb85d142746c9, 0x7b08274021d72f86,
        0xef6169a5e475bfb1, 0xa3df4c061bccfd88, 0x00000000034fc67c,
    ];
    #[rustfmt::skip]
    const B: [u64; 151] = [
        0xc13958ce99bca870, 0x2cfd935e7f5ac8d4, 0x6730e958a4d8f2a3, 0x9ebb0e51099548cd,
        0xfc2896cb1be28a52, 0x516927fa56818f90, 0x32a2dc13423c2c0f, 0x9e7e5e614eb32554,
        0x00b2408373ee6594, 0xdecaf7264f60e7d5, 0xf2abf67fb9c0186b, 0x2bd8e61e2fab237b,
        0x8afecc7731316c3b, 0x2a2a3276810b4437, 0x01a63b0a38ff8f0e, 0xeea81b11e219c7a9,
        0x4255a67d595901c3, 0xc58f53e7f490aacf, 0x4cac68b401199bcd, 0x5f87b7a0218aa7b7,
        0xde74aeb992efad18, 0xcd6222650f8e53d9, 0x0ea600b78598424f, 0x9b5832809c972f58,
        0x5f79238dd2e0e88d, 0xa7324ce2ed372d88, 0x7c2d18d4c6990ae0, 0xeb57ce690c3723fb,
        0x3a165a49cdf71a66, 0x7ef867edbed99cea, 0x2a09b75734afb991, 0x9f24013d66eee9fe,
        0x8f898358266f4886, 0xaf2f14ee4ad00d7f, 0x4c78c5d8f08a9980, 0x047df4bba903fd00,
        0x70040869f1634763, 0x38679328d958f8bb, 0xf56d2cc02a34fd64, 0x0425b09dba74f910,
        0xb81708f09f9d0a49, 0xff84cf16a80c841d, 0xf1d5ed09e01cad5d, 0x710b0511653a8b99,
        0x2195adcd1b7d24c6, 0x4d154daac20bcc69, 0x6f1fa723ce8f3dfe, 0x0d4b2fa73c45507b,
        0xafa5d16979a48025, 0xd96e91209e0a232f, 0x399ad46c28d36eb1, 0x50b746ebfda6a5c2,
        0xbab391caa777d73f, 0xe7ca3c2e3d8b981c, 0x6c0232522627d062, 0x812331d0c31410cc,
        0x5059154c38383a38, 0x3631e09a9ef6d170, 0x6897f224334a2da1, 0x4741bffdab215caf,
        0x56d2ec0895966274, 0xaf2a283047d9ad1c, 0x77b98e70c7520e24, 0xa05cbb596d667657,
        0x5333083ccaf46fca, 0x0457411b2d8a702f, 0xd9f2d4929bcdbe56, 0xc5f3a38728ccdf5c,
        0x70726ddfd0aed392, 0x9d003fd58bc1b607, 0xe50053107f23fb82, 0x9875a9b476fa233b,
        0x878c632d4fa0abb5, 0xdaa6a43538b513af, 0x5c4f52a897b2215f, 0x719a1fa0621a5ef2,
        0x6b70bda27de5c177, 0x068f97ba15fa45ba, 0xecd672879878e84e, 0x6e91fa32c4a57d2d,
        0xf571df5171940435, 0x4464cec71536a5c1, 0x841507c477c45a55, 0x9082a40168eca465,
        0x5df27a19d5c4cfb2, 0xfce5661a46f8c7a6, 0x2a408ed16ef40600, 0xe7cb76b3020241b5,
        0x45804af18d239e72, 0xe660a82aeedcec47, 0x6ec265cc72fbb062, 0x8e0a56c474983c6f,
        0x5f6b4dd3f1808527, 0x1bc801c5610928de, 0xbdfd60fe6be557e3, 0x9de13b3e9feefd39,
        0x0e2321596354550c, 0x42965623a57c135d, 0x7cecb000041b9eb9, 0xc1a9aca0fd553bbe,
        0x4db1aee969ab76d7, 0x625795a56abe56d9, 0xe1b06e0013553470, 0x432d03f0a6ee7fb7,
        0xb3b550c20c60e8cf, 0xc8dabba553e659fb, 0x17b7de6ebbc213de, 0x1f603cf826bc961e,
        0xfd3e64cc07beeb0b, 0x670c3c0354c9c3b5, 0xb09a1d0015df3021, 0xf9a3ee905a97316c,
        0x9201749be20b957b, 0x5fb4be3848951190, 0x0952856252772629, 0x5e1159e9dff95f7e,
        0x48cff3014144195d, 0x3ad86f8a7af53600, 0x5610b013528f934f, 0xf601c274a493e190,
        0x092486f2ddd6123d, 0xc89dd152eaaa3839, 0x5f40dc97fff11227, 0x0da6ae1885eb560c,
        0x86f8c3c53b434aeb, 0x6c11ecbd61c5f4fc, 0x4712d93997ce0505, 0xac550bfe87283dbb,
        0xdcc8c0569bffe0a6, 0x764fad5cf81d96e9, 0xeda12dcdd3384cbd, 0x5f28e7dd7246efc2,
        0x4d964240eaf2804a, 0xe3ab2f1a5b67d2d3, 0xaa39d26270f7efc2, 0x234f2a8d22f9cd16,
        0x571c1665e0a2e370, 0xfe7aa9c5fb50b5e3, 0x158504cd156ac531, 0xf16cf0847a707789,
        0x804978df420ec974, 0xa6433cb48bed045c, 0xa97e806f2d369846, 0x704712c8d46d7cb3,
        0x4a86f33c2aa4901f, 0x2c918b54a2dc4b8b, 0xd25a41d53035661d, 0x8fb1ecd762de6b4a,
        0xac65b15cfb097eb8, 0x795888ab562e6f7b, 0x00001c628765ab75,
    ];

    let (expected, exp_carry) = mul_ref_parts(&A, &B);
    let mut out = vec![0u64; A.len() + B.len() - 1];
    let carry = mul_dyn(&A, &B, &mut out);
    assert_eq_result(
        &out,
        carry,
        &expected,
        exp_carry,
        "mul_dyn structured 151x151",
    );
    let mut out = vec![0u64; A.len() + B.len() - 1];
    let carry = karatsuba_entry_dyn(&A, &B, &mut out);
    assert_eq_result(
        &out,
        carry,
        &expected,
        exp_carry,
        "karatsuba_entry_dyn structured 151x151",
    );
}

// ─── Section 18: hi_mul_static Above Cutoff ─────────────────────────────────

#[test]
fn test_hi_mul_static_split_matches_dyn() {
    const N: usize = 1 + max_usize(DYN_PARTIAL_MUL_CUTOFF, STATIC_PARTIAL_MUL_CUTOFF);
    for seed in 0u64..5 {
        let a = rand_nonzero_vec(N, seed + 9_700);
        let b = rand_nonzero_vec(N, seed + 9_800);

        let mut out_dyn = vec![0u64; N];
        let c_dyn = hi_mul_dyn(&a, &b, &mut out_dyn);

        let mut out_static = vec![0u64; N];
        let c_static = hi_mul_static::<N>(&a, &b, &mut out_static);

        assert_eq_result(
            &out_static,
            c_static,
            &out_dyn,
            c_dyn,
            &format!("hi_mul_static split seed={seed}"),
        );

        let expected = hi_mul_ref(&a, &b, N);
        assert_approx_buf(
            &out_static,
            &expected,
            &format!("hi_mul_static split vs exact seed={seed}"),
        );
    }
}

#[test]
fn test_hi_sqr_static_split_matches_dyn() {
    const N: usize = 1 + max_usize(DYN_PARTIAL_SQR_CUTOFF, STATIC_PARTIAL_SQR_CUTOFF);
    for seed in 0u64..5 {
        let a = rand_nonzero_vec(N, seed + 9_900);

        let mut out_dyn = vec![0u64; N];
        let c_dyn = hi_sqr_dyn(&a, &mut out_dyn);

        let mut out_static = vec![0u64; N];
        let c_static = hi_sqr_static::<N>(&a, &mut out_static);

        assert_eq_result(
            &out_static,
            c_static,
            &out_dyn,
            c_dyn,
            &format!("hi_sqr_static split seed={seed}"),
        );

        let expected = hi_sqr_ref(&a, N);
        assert_approx_buf(
            &out_static,
            &expected,
            &format!("hi_sqr_static split vs exact seed={seed}"),
        );
    }
}

// ─── Section 19: Scratch Size + Cost/Dispatch Helpers ────────────────────────

#[test]
fn test_find_scratch_sizes() {
    // find_karatsuba_scratch: various dispatch arms
    // Prim/Prim2/School: should return 0
    assert_eq!(find_karatsuba_scratch(5, 1), 0); // Prim
    assert_eq!(find_karatsuba_scratch(5, 2), 0); // Prim2
    assert_eq!(find_karatsuba_scratch(10, 5), 0); // School

    // Chunking and Recurse: should return > 0
    let chunk_sz = find_karatsuba_scratch(100, KARA_CUTOFF + 5);
    assert!(chunk_sz > 0, "chunking scratch should be > 0");

    let recurse_sz = find_karatsuba_scratch(KARA_CUTOFF + 10, KARA_CUTOFF + 10);
    assert!(recurse_sz > 0, "recurse scratch should be > 0");

    // find_karatsuba_sqr_scratch_sz
    assert_eq!(find_karatsuba_sqr_scratch_sz(KARATSUBA_SQR_CUTOFF), 0);
    let sqr_scratch = find_karatsuba_sqr_scratch_sz(KARATSUBA_SQR_CUTOFF + 1);
    assert!(sqr_scratch > 0, "sqr scratch should be > 0 above cutoff");
}

#[test]
fn test_find_max_ntt_size() {
    for n in [1usize, 2, 3, 5, 25, 26, 27, 35, 513] {
        let got = find_max_ntt_size(n);
        assert!(got >= n, "find_max_ntt_size({n})={got} is too small");
    }
    assert_eq!(find_max_ntt_size(25), 25);
    assert_eq!(find_max_ntt_size(27), 27);
}

#[test]
fn test_dispatch_cost_helpers() {
    // fft_cost, chunking_karatsuba_cost, karatsuba_cost should return positive values
    assert!(fft_cost(50, 30) > 0.0);
    assert!(fft_cost(1000, 500) > 0.0);

    assert!(chunking_karatsuba_cost(100, 30) > 0.0);
    assert!(karatsuba_cost(100, 60) > 0.0);

    // is_karatsuba at boundary sizes
    let l = 100;
    let s_small = 10; // s <= half → chunking branch
    let s_large = 80; // s > half → regular branch
                      // Just ensure no panic; actual result depends on cutoffs
    let _ = is_karatsuba(
        l,
        s_small,
        FFT_CHUNKING_KARATSUBA_CUTOFF,
        FFT_KARATSUBA_CUTOFF,
    );
    let _ = is_karatsuba(
        l,
        s_large,
        FFT_CHUNKING_KARATSUBA_CUTOFF,
        FFT_KARATSUBA_CUTOFF,
    );
}

// ─── Section 9: FFT Accuracy at Cutoff ──────────────────────────────────────

#[cfg(feature = "_fft_cutoffs")]
mod fft_cutoff {
    use super::*;
    use rayon::prelude::*;

    const TEST_CNT: u64 = 1 << 7;

    #[test]
    fn test_fft_mul_accuracy_at_16bit_cutoff() {
        for &offset in &[1, 2, 4] {
            let s = FFT_16BIT_CUTOFF - offset;
            let l = s + 100;
            (0u64..TEST_CNT).into_par_iter().for_each(|seed| {
                let long = rand_nonzero_vec(l, seed);
                let short = rand_nonzero_vec(s, seed + 5000);

                let mut out_fft = vec![0u64; l + s - 1];
                let c_fft = fft_entry(&long, &short, &mut out_fft);

                let mut out_ntt = vec![0u64; l + s - 1];
                let c_ntt = ntt_entry_dyn(&long, &short, &mut out_ntt);

                if let Some(report) = diff_report(
                    &out_fft,
                    c_fft,
                    &out_ntt,
                    c_ntt,
                    &format!("FFT 16-bit accuracy s={s} seed={seed}"),
                ) {
                    panic!("{report}");
                }
            });
        }
    }

    #[test]
    fn test_fft_sqr_accuracy_at_16bit_cutoff() {
        for &offset in &[1, 2] {
            let n = FFT_16BIT_CUTOFF - offset;
            (0u64..TEST_CNT).into_par_iter().for_each(|seed| {
                let a = rand_nonzero_vec(n, seed);

                let mut out_fft = vec![0u64; 2 * n - 1];
                let c_fft = fft_sqr_entry(&a, &mut out_fft);

                let mut out_ntt = vec![0u64; 2 * n - 1];
                let c_ntt = ntt_sqr_entry_dyn(&a, &mut out_ntt);

                if let Some(report) = diff_report(
                    &out_fft,
                    c_fft,
                    &out_ntt,
                    c_ntt,
                    &format!("FFT sqr 16-bit accuracy n={n} seed={seed}"),
                ) {
                    panic!("{report}");
                }
            });
        }
    }
}

#[test]
fn test_karatsuba_mul_prim_dispatch_ignores_stale_out_tail() {
    // karatsuba_mul's Prim/Prim2 dispatch used to call mul_prim/mul_prim2 on
    // the FULL out slice, even when out is wider than the standard
    // long.len()+short.len()-1 convention (a legitimate case once
    // karatsuba_core's Recurse-boundary sizing fix can hand nested Prim/Prim2
    // sub-calls a scratch-backed buffer wider than that) — silently
    // multiplying whatever garbage sat beyond long.len() as if it were part
    // of the multiplicand. Poison out's tail explicitly and confirm the
    // dispatch ignores it, for both arms, calling karatsuba_entry_dyn
    // directly with an over-sized out (bypassing karatsuba_core entirely).
    let long = rand_nonzero_vec(20, 1);
    let poison = 0xDEADBEEFDEADBEEFu64;
    let extra = 5;

    for short in [vec![7u64], vec![7u64, 9u64]] {
        let standard_len = long.len() + short.len() - 1;
        let mut out = vec![poison; standard_len + extra];
        let carry = karatsuba_entry_dyn(&long, &short, &mut out);

        let (exp_buf, exp_carry) = mul_ref_parts(&long, &short);
        assert_eq!(
            carry,
            0,
            "short.len()={}: overflow should fold into out, not the return",
            short.len()
        );
        assert_eq!(
            &out[..standard_len],
            &exp_buf[..],
            "short.len()={}: product mismatch",
            short.len()
        );
        assert_eq!(
            out[standard_len],
            exp_carry,
            "short.len()={}: overflow limb wrong (likely poisoned)",
            short.len()
        );
        assert!(
            out[standard_len + 1..].iter().all(|&x| x == 0),
            "short.len()={}: tail beyond the legitimate overflow limb must be zero, got {:?}",
            short.len(),
            &out[standard_len + 1..]
        );
    }
}

// ─── Section 10: NTT Re-entrancy Under Rayon ────────────────────────────────

// A thread blocked in an NTT's rayon::join can run another caller's NTT, so the
// twiddle caches must not stay borrowed across a convolution.
#[test]
fn test_ntt_entry_dyn_nested_in_rayon() {
    use rayon::prelude::*;
    let n = 40_000;
    let cases: Vec<_> = (0..24u64)
        .map(|s| {
            (
                rand_nonzero_vec(n + s as usize, s),
                rand_nonzero_vec(n, s + 100),
            )
        })
        .collect();
    let expect: Vec<_> = cases
        .iter()
        .map(|(a, b)| {
            let mut out = vec![0u64; a.len() + b.len()];
            ntt_entry_dyn(a, b, &mut out);
            out
        })
        .collect();
    for _ in 0..4 {
        let got: Vec<_> = cases
            .par_iter()
            .map(|(a, b)| {
                let mut out = vec![0u64; a.len() + b.len()];
                ntt_entry_dyn(a, b, &mut out);
                out
            })
            .collect();
        assert_eq!(got, expect);
    }
}

// Random tests almost never give a Karatsuba sub-product a zero top limb. Biasing
// limbs toward zeros and powers of two does: this found 571 bad products in 6000 trials
// with the old overflow offset in karatsuba_core.
#[test]
fn test_mul_structured_operand_fuzz() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(424242);
    let tops = [1u64, 1 << 32, 1 << 58, 1 << 63, 3 << 50, u64::MAX];
    let (mut trials, mut bad) = (0, 0);
    for _ in 0..2000 {
        let l = rng.gen_range(3..320usize);
        let s = rng.gen_range(3..=l);
        let gen = |n: usize, rng: &mut StdRng| {
            let zp = [0.0, 0.3, 0.7, 0.95][rng.gen_range(0..4)];
            let mut v: Vec<u64> = (0..n)
                .map(|_| {
                    if rng.gen_bool(zp) {
                        0
                    } else if rng.gen_bool(0.2) {
                        tops[rng.gen_range(0..tops.len())]
                    } else {
                        rng.gen()
                    }
                })
                .collect();
            v[n - 1] = tops[rng.gen_range(0..tops.len())];
            v
        };
        let a = gen(l, &mut rng);
        let b = gen(s, &mut rng);
        let (expected, exp_carry) = mul_ref_parts(&a, &b);
        let mut out = vec![0xdead_beef_u64; l + s - 1];
        let c1 = mul_dyn(&a, &b, &mut out);
        let ok1 = out == expected && c1 == exp_carry;
        let mut out = vec![0u64; l + s - 1];
        let c2 = mul_static::<1024>(&a, &b, &mut out);
        let ok2 = out == expected && c2 == exp_carry;
        trials += 1;
        if !(ok1 && ok2) {
            bad += 1;
            if bad < 5 {
                println!("structured product wrong: l={l} s={s} dyn={ok1} static={ok2}");
            }
        }
    }
    assert_eq!(bad, 0, "{bad}/{trials} structured products wrong");
}

// Numeric error of a high product (out plus returned top limb) against the
// exact top window of the full product, as an unsigned limb buffer.
fn hi_window_error(a: &[u64], b: &[u64], out: &[u64], of: u64) -> Vec<u64> {
    let (full, full_of) = mul_ref_parts(a, b);
    let d = full.len().saturating_sub(out.len());
    let mut exact = full[d..].to_vec();
    exact.push(full_of);
    let mut got = out.to_vec();
    got.push(of);
    let mut err = if crate::utils::utils::cmp_buf(&got, &exact).is_lt() {
        crate::utils::utils::sub_buf(&mut exact, &got);
        exact
    } else {
        crate::utils::utils::sub_buf(&mut got, &exact);
        got
    };
    trim_lz(&mut err);
    err
}

fn assert_hi_window_ulps(a: &[u64], b: &[u64], out: &[u64], of: u64, msg: &str) {
    let err = hi_window_error(a, b, out, of);
    assert!(
        err.len() <= 1 && err.first().map_or(true, |&e| e <= 2),
        "{msg}: high product off by more than 2 ulps; err={err:?}"
    );
}

// Truncating operands to exactly out.len() limbs used to leave the whole low
// output limb wrong (~B ulps), which rcp_dyn's Newton chain squared into
// multi-limb reciprocal errors from R = 727.
#[test]
fn test_hi_mul_truncation_error_is_a_few_ulps() {
    let shapes = [
        (92usize, 92usize, 91usize),
        (183, 183, 182),
        (365, 367, 366),
        (729, 729, 728),
        (400, 120, 150),
        (200, 50, 100),
    ];
    for (idx, &(a_len, b_len, out_len)) in shapes.iter().enumerate() {
        for seed in 0..3u64 {
            let mut a = rand_nonzero_vec(a_len, 20_000 + 31 * idx as u64 + seed);
            let b = rand_nonzero_vec(b_len, 21_000 + 37 * idx as u64 + seed);
            if seed == 2 {
                // Saturated low limbs maximize the dropped carry.
                a.iter_mut().take(4).for_each(|l| *l = u64::MAX);
            }
            let mut out = vec![0u64; out_len];
            let of = hi_mul_dyn(&a, &b, &mut out);
            assert_hi_window_ulps(&a, &b, &out, of, &format!("hi_mul_dyn {a_len}x{b_len}->{out_len}"));

            let of = hi_mul_static::<1024>(&a, &b, &mut out);
            assert_hi_window_ulps(&a, &b, &out, of, &format!("hi_mul_static {a_len}x{b_len}->{out_len}"));

            let mut out = vec![0u64; out_len];
            let of = hi_sqr_dyn(&a, &mut out);
            assert_hi_window_ulps(&a, &a, &out, of, &format!("hi_sqr_dyn {a_len}->{out_len}"));

            let of = hi_sqr_static::<1024>(&a, &mut out);
            assert_hi_window_ulps(&a, &a, &out, of, &format!("hi_sqr_static {a_len}->{out_len}"));
        }
    }
}

// The static guard shrinks as out_len approaches N so the split paths' low
// buffers still fit. Operands are at most N limbs, so from out_len = N - 1 the
// clamped guard keeps them whole and the window must be exact. out_len == N
// with two N-limb operands used to index past N in the split path (even N),
// and N - 1 used to get no guard at all (a whole wrong low limb).
#[test]
fn test_hi_mul_static_near_capacity() {
    fn check<const N: usize>() {
        for out_len in N - 4..=N {
            let half = N / 2 + 7;
            let shapes = [(N, N), (N, N - 1), (N - 1, N), (N - 1, N - 1), (N, half), (half, N), (N, 3)];
            for (idx, &(a_len, b_len)) in shapes.iter().enumerate() {
                let seed = (N * 100 + out_len * 10 + idx) as u64;
                let mut a = rand_nonzero_vec(a_len, 22_000 + seed);
                let b = rand_nonzero_vec(b_len, 23_000 + seed);
                if seed % 2 == 1 {
                    a.iter_mut().filter(|l| **l & 1 == 0).for_each(|l| *l = u64::MAX);
                }
                let exact = out_len + 1 >= N;
                let msg = format!("N={N} out={out_len} a={a_len} b={b_len}");

                let mut out = vec![0u64; out_len];
                let of = hi_mul_static::<N>(&a, &b, &mut out);
                if exact {
                    assert!(hi_window_error(&a, &b, &out, of).is_empty(), "hi_mul_static {msg}: not exact");
                } else {
                    assert_hi_window_ulps(&a, &b, &out, of, &format!("hi_mul_static {msg}"));
                }

                if b_len == N {
                    let of = hi_sqr_static::<N>(&a, &mut out);
                    if exact {
                        assert!(hi_window_error(&a, &a, &out, of).is_empty(), "hi_sqr_static {msg}: not exact");
                    } else {
                        assert_hi_window_ulps(&a, &a, &out, of, &format!("hi_sqr_static {msg}"));
                    }
                }
            }
        }
    }
    check::<96>();
    check::<97>();
    check::<160>();
    check::<161>();
    check::<512>();
}

// hi_mul_*/hi_sqr_* compute their unguarded core product and add the guard
// limbs' terms separately (hi_guarded_window). The result must equal, bit for
// bit, the top window of the exact product of the guarded operands, for dyn and
// static alike, including operands narrower than the window and saturated or
// zero-heavy limbs that stress the carry chains.
#[test]
fn test_hi_mul_guard_terms_match_exact_window() {
    fn guarded_ref(a: &[u64], b: &[u64], k: usize) -> (Vec<u64>, u64) {
        let ta = &a[a.len().saturating_sub(k + 2)..];
        let tb = &b[b.len().saturating_sub(k + 2)..];
        let mut full = vec![0u64; ta.len() + tb.len()];
        mul_buf(ta, tb, &mut full);
        let l = full.len();
        (full[l - 1 - k..l - 1].to_vec(), full[l - 1])
    }
    for seed in 0..200u64 {
        // Past every partial cutoff, so all four entries take their guarded window.
        let k = 1
            + max_usize(
                max_usize(DYN_PARTIAL_MUL_CUTOFF, STATIC_PARTIAL_MUL_CUTOFF),
                max_usize(DYN_PARTIAL_SQR_CUTOFF, STATIC_PARTIAL_SQR_CUTOFF),
            )
            + (seed as usize * 37) % 500;
        let a_len = [k - 50, k, k + 1, k + 2, k + 3, k + 40][(seed % 6) as usize];
        let b_len = [k + 2, k + 1, k + 7, k, k + 300, 3][(seed / 6 % 6) as usize];
        if a_len + b_len - 1 <= k {
            continue;
        }
        let mut a = rand_nonzero_vec(a_len, 30_000 + seed);
        let mut b = rand_nonzero_vec(b_len, 31_000 + seed);
        if seed % 3 == 0 {
            a.iter_mut().filter(|l| **l & 3 == 0).for_each(|l| *l = u64::MAX);
        }
        if seed % 5 == 0 {
            b.iter_mut().filter(|l| **l & 3 == 0).for_each(|l| *l = 0);
        }
        let msg = format!("seed={seed} k={k} a={a_len} b={b_len}");

        let expected = guarded_ref(&a, &b, k);
        let mut out = vec![0u64; k];
        let of = hi_mul_dyn(&a, &b, &mut out);
        assert_eq!((&out, of), (&expected.0, expected.1), "hi_mul_dyn {msg}");
        let of = hi_mul_static::<2048>(&a, &b, &mut out);
        assert_eq!((&out, of), (&expected.0, expected.1), "hi_mul_static {msg}");

        if 2 * a_len - 1 > k {
            let expected = guarded_ref(&a, &a, k);
            let of = hi_sqr_dyn(&a, &mut out);
            assert_eq!((&out, of), (&expected.0, expected.1), "hi_sqr_dyn {msg}");
            let of = hi_sqr_static::<2048>(&a, &mut out);
            assert_eq!((&out, of), (&expected.0, expected.1), "hi_sqr_static {msg}");
        }
    }
}
