use crate::utils::{
    div::{knuth_est, mul_u64_asm},
    utils::{add_buf, cmp_buf, combine_u64, dec_buf, sub_buf},
};

fn correct_sqrt(x: &mut [u64], s: &[u64], q_sqr: &[u64]) -> bool {
    let c = sub_buf(x, q_sqr);
    if c {
        add_buf(x, s);
        add_buf(x, s);
        dec_buf(x);
    }
    return c;
}

fn sqrt_4x2(x0: &mut u64, x1: &mut u64, x2: &mut u64, x3: &mut u64) -> (u64, u64) {
    let x_hi = combine_u64(*x2, *x3);
    let s_hi = x_hi.isqrt();
    let rem = x_hi - s_hi * s_hi;
    let c = rem / s_hi;
    let r0 = rem % s_hi;
    let y = (r0 << 64) | *x1 as u128;
    let t = y / s_hi;
    let v = y % s_hi;
    let (mut s_lo, d) = match c {
        0 => (t >> 1, t & 1),
        1 => ((1 << 63) | (t >> 1), t & 1),
        2 => (u64::MAX as u128, t + 2),
        _ => unreachable!(),
    };

    let u = v + (d as u128) * s_hi;
    let s_lo_sqr = s_lo * s_lo;
    let mut buf = [*x0, u as u64, (u >> 64) as u64];
    if correct_sqrt(
        &mut buf,
        &[s_lo as u64, s_hi as u64],
        &[s_lo_sqr as u64, (s_lo_sqr >> 64) as u64],
    ) {
        s_lo -= 1;
    }
    *x0 = buf[0];
    *x1 = buf[1];
    *x2 = buf[2];
    *x3 = 0;
    (s_lo as u64, s_hi as u64)
}

fn binom_sqrt_est_reduced(x: &mut [u64], s: &[u64], s0: u64, s1: u64, c: u64) -> u64 {
    let t = {
        let (win, of) = x.split_at_mut(s.len());
        knuth_est(win, &mut of[0], s, s1, s0)
    };
    let (q, d) = match c {
        0 => (t >> 1, t & 1),
        1 => ((1 << 63) | (t >> 1), t & 1),
        2 => (u64::MAX, t + 2),
        _ => unreachable!(),
    };
    for _ in 0..d {
        add_buf(x, s);
    }
    return q;
}

/// Computes `floor(sqrt(x * B^(2 * s.len() - x.len())))`, where `B = 2^64`.
/// The input must be normalized, and its length must satisfy
/// `s.len() < x.len() <= 2 * s.len()`. On return, `x` contains the remainder.
pub fn binom_sqrt(x: &mut [u64], s: &mut [u64]) {
    if s.is_empty() {
        return;
    }
    // assume correct size bounds
    debug_assert!(x.len() > s.len());
    debug_assert!(2 * s.len() >= x.len());
    // assume normalized x
    debug_assert!(x.last().copied().unwrap() >= (1 << 62));

    let x_len = x.len();
    let s_len = s.len();
    s.fill(0);

    // x.len == 1 or 2
    if s_len == 1 {
        let term = combine_u64(x[0], x[1]);
        let sqrt = term.isqrt();
        let rem = term - sqrt * sqrt;
        s[0] = sqrt as u64;
        x[0] = rem as u64;
        x[1] = (rem >> 64) as u64;
        return;
    }

    let (mut x0, mut x1, mut x2, mut x3) = (
        x_len.checked_sub(4).map_or(0, |i| x[i]),
        x_len.checked_sub(3).map_or(0, |i| x[i]),
        x[x_len - 2],
        x[x_len - 1],
    );
    let (s0, s1) = sqrt_4x2(&mut x0, &mut x1, &mut x2, &mut x3);
    s[s_len - 2] = s0;
    s[s_len - 1] = s1;

    let mut x_start = if x_len == 3 {
        x[0] = x0;
        x[1] = x1;
        x[2] = x2;
        0
    } else {
        x[x_len - 4] = x0;
        x[x_len - 3] = x1;
        x[x_len - 2] = x2;
        x[x_len - 1] = x3;
        x_len - 4
    };

    if s_len == 2 {
        return;
    }

    let mut x_end = x_start + 3;
    let mut j = s_len - 2;
    while j > 0 {
        let mut c = 0;
        while cmp_buf(&x[x_start..x_end], &s[j..]).is_ge() {
            sub_buf(&mut x[x_start..x_end], &s[j..]);
            c += 1;
        }
        x_end -= 1;
        if x_start < 2 {
            let missing = 2 * j - x_start;
            let available = x_len - x_end;
            let shift = missing.min(available);
            x.copy_within(..x_end, shift);
            x[..shift].fill(0);
            x_start += shift;
            x_end += shift;
        }
        j -= 1;
        x_start -= 2;
        s[j] = binom_sqrt_est_reduced(&mut x[x_start + 1..x_end], &s[j + 1..], s0, s1, c);
        let s_sqr = unsafe { mul_u64_asm(s[j], s[j]) };
        if correct_sqrt(&mut x[x_start..x_end], &s[j..], &[s_sqr.1, s_sqr.0]) {
            s[j] -= 1;
        }
    }
}
