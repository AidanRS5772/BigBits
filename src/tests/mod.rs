// START, and a tail bound of 0, for series that tests only sum over explicit ranges.
macro_rules! no_tail {
    ($start:expr) => {
        const START: u64 = $start;
        fn tail_bits(_: u64, _: i64) -> u64 {
            0
        }
    };
}

mod test_bin_split;
mod test_series;
pub(crate) mod test_div;
mod test_mul;
mod test_agm;
mod test_sqrt;
mod test_theta;
mod test_theta_plan;
mod test_trans;
mod test_tuning;
mod test_utils;
mod test_verify;

use crate::utils::div::div_dyn;
use crate::utils::mul::{mul_buf, mul_vec};
use crate::utils::utils::{add_buf, eq_buf, sub_buf, trim_lz};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Decode a small buffer (≤2 limbs) to u128 for easy expected-value arithmetic.
pub(super) fn to_u128(buf: &[u64]) -> u128 {
    match buf.len() {
        0 => 0,
        1 => buf[0] as u128,
        2 => buf[0] as u128 | ((buf[1] as u128) << 64),
        _ => panic!("buf too large for u128 (len={})", buf.len()),
    }
}

// Generate a random Vec<u64> of `len` limbs with a given seed offset.
pub(super) fn rand_vec(len: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen::<u64>()).collect()
}

// Generate a random non-zero Vec<u64> of `len` limbs.
pub(super) fn rand_nonzero_vec(len: usize, seed: u64) -> Vec<u64> {
    let mut v = rand_vec(len, seed);
    if v[len - 1] == 0 {
        v[len - 1] = 1;
    }
    v
}

// Operand kinds that round worst in the 16-bit FFT. Balanced digits cancel in a coefficient's sum
// unless they keep one sign or a pattern, so these are written in balanced digits at full size: all
// -2^15, all 2^15 - 1, random signs, and periods 3 and 12. Raw 0x7FFF or 0x8000 limbs wouldn't do,
// as a carry that persists through them balances them to one sign whatever their digits.
pub(super) const FFT_WORST_KINDS: usize = 5;

pub(super) fn fft_worst_operand(kind: usize, len: usize, rng: &mut StdRng) -> Vec<u64> {
    const HI: i64 = (1 << 15) - 1;
    const LO: i64 = -(1 << 15);
    let digit = |k: usize, rng: &mut StdRng| match kind % FFT_WORST_KINDS {
        0 => LO,
        1 => HI,
        2 => {
            if rng.gen::<bool>() {
                HI
            } else {
                LO
            }
        }
        3 => {
            if k % 3 == 2 {
                LO
            } else {
                HI
            }
        }
        _ => {
            if k % 12 < 6 {
                HI
            } else {
                LO
            }
        }
    };
    let mut v = vec![0u64; len];
    let mut carry = 0i64;
    for k in 0..4 * len {
        let t = digit(k, rng) + carry;
        v[k / 4] |= (t.rem_euclid(1 << 16) as u64) << (16 * (k % 4));
        carry = t.div_euclid(1 << 16);
    }
    v[len - 1] |= 1 << 63;
    v
}

// A trimmed `n`-limb operand biased toward zero limbs and powers of two, which random limbs
// almost never produce: structured operands catch carry and offset bugs that random ones miss.
pub(super) fn structured_vec(n: usize, rng: &mut StdRng) -> Vec<u64> {
    const TOPS: [u64; 6] = [1, 1 << 32, 1 << 58, 1 << 63, 3 << 50, u64::MAX];
    let zp = [0.0, 0.3, 0.7, 0.95][rng.gen_range(0..4)];
    let mut v: Vec<u64> = (0..n)
        .map(|_| {
            if rng.gen_bool(zp) {
                0
            } else if rng.gen_bool(0.2) {
                TOPS[rng.gen_range(0..TOPS.len())]
            } else {
                rng.gen()
            }
        })
        .collect();
    v[n - 1] = TOPS[rng.gen_range(0..TOPS.len())];
    v
}

/// Schoolbook reference multiplication (normalized: carry appended, leading zeros trimmed).
pub(super) fn mul_ref(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return vec![0];
    }
    let mut out = vec![0u64; a.len() + b.len() - 1];
    let c = mul_buf(a, b, &mut out);
    if c > 0 {
        out.push(c);
    }
    trim_lz(&mut out);
    out
}

/// Verify the division invariant: q*d + r == n.
pub(super) fn verify_divmod(n: &[u64], d: &[u64], q: &[u64], r: &[u64]) -> bool {
    let qd = mul_ref(q, d);
    let mut sum = vec![0u64; qd.len().max(r.len())];
    sum[..qd.len()].copy_from_slice(&qd);
    let c = add_buf(&mut sum, r);
    if c {
        sum.push(1);
    }
    trim_lz(&mut sum);
    let mut n_trimmed = n.to_vec();
    trim_lz(&mut n_trimmed);
    eq_buf(&sum, &n_trimmed)
}

// floor(p 2^(64 len - shift) / q) by one exact division of the full operands. A negative
// exponent shifts p right first, which floors to the same quotient.
pub(super) fn exact_fraction(p: &[u64], q: &[u64], shift: i64, len: usize) -> Vec<u64> {
    let e = 64 * len as i64 - shift;
    if p.is_empty() {
        return vec![0; len];
    }
    if e < 0 {
        let (w, v) = ((-e / 64) as usize, (-e % 64) as u32);
        let mut n = p.get(w..).unwrap_or(&[]).to_vec();
        if v != 0 {
            for i in 0..n.len() {
                n[i] = (n[i] >> v) | n.get(i + 1).map_or(0, |&h| h << (64 - v));
            }
        }
        trim_lz(&mut n);
        return exact_fraction(&n, q, 64 * len as i64, len);
    }
    let (w, v) = ((e / 64) as usize, (e % 64) as u32);
    let mut n = vec![0; w + p.len() + 1];
    n[w..w + p.len()].copy_from_slice(p);
    if v != 0 {
        for i in (w..n.len()).rev() {
            n[i] = (n[i] << v) | if i > w { n[i - 1] >> (64 - v) } else { 0 };
        }
    }
    trim_lz(&mut n);
    let mut quo = vec![0; n.len().saturating_sub(q.len()) + 1];
    div_dyn(&n, q, &mut quo);
    assert!(
        quo[len.min(quo.len())..].iter().all(|&l| l == 0),
        "S must be below 1"
    );
    quo.resize(len, 0);
    quo
}
