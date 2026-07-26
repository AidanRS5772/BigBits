mod test_div;
mod test_mul;
mod test_utils;

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
