use crate::utils::{
    div::{div_rem_dyn, div_rem_static, knuth_est, mul_u64_asm},
    mul::{sqr_dyn, sqr_static},
    utils::{add_buf, add_prim, buf_len, cmp_buf, combine_u64, dec_buf, shl_buf, shr_buf, sub_buf},
    ScratchGuard,
};

#[inline]
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
pub fn binom_sqrt_core(x: &mut [u64], s: &mut [u64]) {
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
            let shift = (2 * j - x_start).min(x_len - x_end);
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

fn add_mul(x: &mut [u64], s: &[u64], d: u64) -> bool {
    assert!(x.len() > s.len(), "x needs at least one extra limb");
    let mut carry = 0_u128;
    for (x0, &s0) in x.iter_mut().zip(s) {
        let acc = s0 as u128 * d as u128 + *x0 as u128 + carry;
        *x0 = acc as u64;
        carry = acc >> 64;
    }
    add_prim(&mut x[s.len()..], carry as u64)
}

fn sqrt_denormalization(x: &mut [u64], s: &mut [u64], sh: u8) {
    let d = s[0] & ((1 << sh / 2) - 1);
    add_mul(x, s, 2 * d);
    let d_sqr = unsafe { mul_u64_asm(d, d) };
    sub_buf(x, &[d_sqr.1, d_sqr.0]);
    shr_buf(x, sh);
    shr_buf(s, sh / 2);
}

pub fn binom_sqrt(x: &mut [u64], s: &mut [u64]) {
    // assume correct size bounds
    debug_assert!(x.len() > s.len());
    debug_assert!(x.len() <= 2 * s.len());

    let x_len = x.len();
    let sh = x[x_len - 1].leading_zeros() as u8 & !1_u8;
    shl_buf(x, sh);
    binom_sqrt_core(x, s);
    sqrt_denormalization(x, s, sh);
}

const ZIMMERMAN_SQRT_CUTOFF: usize = 50;

fn reduce_sqrt_rem(sqrt_r: &mut [u64], s_hi: &[u64]) -> (bool, bool) {
    let reduced = cmp_buf(sqrt_r, s_hi).is_ge();
    let saturated = if reduced {
        sub_buf(sqrt_r, s_hi);
        cmp_buf(sqrt_r, s_hi).is_eq()
    } else {
        false
    };
    (reduced, saturated)
}

fn zimmermann_sqrt_core(
    x: &mut [u64],
    s: &mut [u64],
    div_rem_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64]),
    sqr_alg: &mut dyn FnMut(&[u64], &mut [u64]) -> u64,
) {
    debug_assert_eq!(x.len(), 2 * s.len());
    if s.len() < ZIMMERMAN_SQRT_CUTOFF {
        binom_sqrt_core(x, s);
        return;
    }
    let s_len = s.len();
    let lo = (s_len - 1) / 2;
    let (s_lo, s_hi) = s.split_at_mut(lo);
    zimmermann_sqrt_core(&mut x[2 * lo..], s_hi, div_rem_alg, sqr_alg);
    let (reduced, saturated) = reduce_sqrt_rem(&mut x[2 * lo..s_len + lo + 1], s_hi);

    if saturated {
        let rem = &mut x[..s_len + 1];
        rem[2 * lo..].fill(0);

        add_buf(&mut rem[lo..], s_hi);
        add_buf(&mut rem[lo..], s_hi);
        s_lo.fill(u64::MAX);
    } else {
        let (d, q_space) = x[lo..].split_at_mut(s_len);
        let q = &mut q_space[..lo + 1];
        let d_len = buf_len(d);
        div_rem_alg(&mut d[..d_len], s_hi, q);
        if shr_buf(q, 1) != 0 {
            add_buf(d, s_hi);
        }
        if reduced {
            q[lo - 1] |= 1 << 63;
        }
        s_lo.copy_from_slice(&q[..lo]);
    }

    let (rem, square_space) = x.split_at_mut(s_len + 1);
    let s_lo_sqr = &mut square_space[..2 * lo];
    sqr_alg(s_lo, s_lo_sqr);
    if correct_sqrt(rem, s, s_lo_sqr) {
        dec_buf(s);
    }
}

pub fn zimmerman_sqrt_dyn(x: &mut [u64], s: &mut [u64]) {
    // assume correct size bounds
    debug_assert!(x.len() > s.len());
    debug_assert!(2 * s.len() >= x.len());

    if s.len() < ZIMMERMAN_SQRT_CUTOFF {
        binom_sqrt(x, s);
        return;
    }

    let x_len = x.len();
    let s_len = s.len();
    let full_len = 2 * s_len;
    let sh = x[x_len - 1].leading_zeros() as u8 & !1_u8;
    shl_buf(x, sh);

    let mut div_rem = |n: &mut [u64], d: &[u64], q: &mut [u64]| div_rem_dyn(n, d, q);
    let mut sqr = |value: &[u64], out: &mut [u64]| sqr_dyn(value, out);

    if x_len == full_len {
        zimmermann_sqrt_core(x, s, &mut div_rem, &mut sqr);
        x[s_len + 1..].fill(0);
    } else {
        let hi = x_len / 2;
        let lo = s_len - hi;
        let (s_lo, s_hi) = s.split_at_mut(lo);
        let x_lo = x_len % 2;
        zimmermann_sqrt_core(&mut x[x_lo..], s_hi, &mut div_rem, &mut sqr);
        let (reduced, saturated) = reduce_sqrt_rem(&mut x[x_lo..hi + x_lo + 1], s_hi);

        let mut gaurd = ScratchGuard::acquire();

        if saturated {
        } else {
            x.copy_within(..hi + x_lo + 1, lo - x_lo);
            x[..lo - x_lo].fill(0);
            let u_len = buf_len(&x);
            let q = gaurd.get(lo + 1);
            div_rem_dyn(&mut x[..u_len], s_hi, q);
            if shr_buf(q, 1) != 0{
                add_buf(x, s_hi);
            }
            if reduced {
                
            }
        }
    }

    sqrt_denormalization(x, s, sh);
}
