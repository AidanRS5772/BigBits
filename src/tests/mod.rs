mod test_utils;
mod test_mul;
mod test_div;

use crate::utils::mul::mul_vec;
use crate::utils::utils::{acc, eq_buf, trim_lz};
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

/// Verify the division invariant: q * d + r == n_orig, using big-integer arithmetic.
pub(super) fn verify_divmod(n_orig: &[u64], d: &[u64], q: &[u64], r: &[u64]) -> bool {
    // quotient zero means n < d, so remainder should equal n
    if q.is_empty() || q.iter().all(|&x| x == 0) {
        let mut n = n_orig.to_vec();
        let mut r2 = r.to_vec();
        trim_lz(&mut n);
        trim_lz(&mut r2);
        return eq_buf(&n, &r2);
    }

    if d.is_empty() {
        return false;
    }

    let (mut result, c) = mul_vec(q, d);
    if c > 0 {
        result.push(c);
    }

    if result.len() < r.len() {
        result.resize(r.len(), 0);
    }
    // Ignore the overflow bool — if q*d+r > n, the equality check below will catch it.
    acc(&mut result, r, 0);

    trim_lz(&mut result);
    let mut n = n_orig.to_vec();
    trim_lz(&mut n);
    eq_buf(&result, &n)
}

/// Generate a random Vec<u64> of `len` limbs with a given seed offset.
pub(super) fn rand_vec(len: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen::<u64>()).collect()
}

/// Generate a random non-zero Vec<u64> of `len` limbs.
pub(super) fn rand_nonzero_vec(len: usize, seed: u64) -> Vec<u64> {
    let mut v = rand_vec(len, seed);
    // Ensure the most significant limb is non-zero so buf_len == len.
    if v[len - 1] == 0 {
        v[len - 1] = 1;
    }
    v
}
