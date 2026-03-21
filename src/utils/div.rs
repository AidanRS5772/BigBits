use crate::utils::{mul::*, utils::*};

pub fn div_prim(buf: &mut [u64], prim: u64) -> u64 {
    let prim_u128 = prim as u128;
    let mut rem: u128 = 0;
    for e in buf.iter_mut().rev() {
        let val = (rem << 64) | (*e as u128);
        rem = val % prim_u128;
        *e = (val / prim_u128) as u64;
    }

    return rem as u64;
}

fn sub_mul_of(win: &mut [u64], of: &mut u64, d: &[u64], q: u64) -> bool {
    let q_u128 = q as u128;
    let mut mul_carry: u64 = 0;
    let mut borrow: u64 = 0;

    for (w, &d) in win.iter_mut().zip(d) {
        let prod = q_u128 * (d as u128) + mul_carry as u128;
        mul_carry = (prod >> 64) as u64;
        let prod_lo = prod as u64;

        let (s1, b1) = w.overflowing_sub(prod_lo);
        let (s2, b2) = s1.overflowing_sub(borrow);
        *w = s2;
        borrow = b1 as u64 + b2 as u64;
    }

    let (s1, b1) = of.overflowing_sub(mul_carry);
    let (s2, b2) = s1.overflowing_sub(borrow);
    *of = s2;

    (b1 as u64 + b2 as u64) != 0
}

fn knuth_est(win: &mut [u64], of: &mut u64, d: &[u64], d1: u128, d0: u128, dfull: u128) -> u64 {
    let win_last = *win.last().unwrap() as u128;
    let of_u128 = *of as u128;
    let val = (of_u128 << 64) | win_last;
    let (mut qhat, rhat) = if of_u128 >= d1 {
        (u64::MAX, val - (u64::MAX as u128) * d1)
    } else {
        ((val / d1) as u64, val % d1)
    };

    if rhat < (1u128 << 64) {
        let a = (qhat as u128) * d0;
        let b = (rhat << 64) | (win[win.len() - 2] as u128);
        if a > b {
            qhat -= if (a - b) > dfull { 2 } else { 1 };
        }
    }

    if sub_mul_of(win, of, d, qhat) {
        qhat -= 1;
        if add_buf(win, d) {
            *of = of.wrapping_add(1);
        }
    }
    return qhat;
}

//assumes normalized d and handles overflow of normalized n
pub fn div_buf_of(n: &mut [u64], of: &mut u64, d: &[u64], out: &mut [u64]) {
    let d_len = d.len();
    let n_len = n.len();

    let d1 = d[d_len - 1] as u128;
    let d0 = d[d_len - 2] as u128;
    let dfull = ((d1 as u128) << 64) | (d0 as u128);

    let q_len = n_len - d_len;
    if out.len() > q_len {
        out[q_len] = knuth_est(&mut n[q_len..], of, d, d1, d0, dfull);
    }
    for i in (0..q_len).rev() {
        let (win, of) = n[i..].split_at_mut(d_len);
        out[i] = knuth_est(win, &mut of[0], d, d1, d0, dfull)
    }
}

fn div_3_2(n: &mut [u64], d: &[u64], d_lo_len: usize, q: &mut [u64], scratch: &mut [u64]) {
    let q_len = q.len();
    let (d_lo, d_hi) = d.split_at(d_lo_len);

    if cmp_buf(&n[d.len()..], d_hi) == std::cmp::Ordering::Less {
        div_2_1(&mut n[d_lo_len..], d_hi, q, scratch);
    } else {
        q.fill(u64::MAX);
        sub_buf(&mut n[d.len()..], d_hi);
        sub_buf(&mut n[d_lo_len..], d_hi);
    }

    let prod_len = q_len + d_lo_len;
    let (m, k_scratch) = scratch.split_at_mut(prod_len);
    if q_len >= d_lo_len {
        karatsuba_alg(q, d_lo, m, k_scratch);
    } else {
        karatsuba_alg(d_lo, q, m, k_scratch);
    }

    if sub_buf(n, &m[..prod_len]) {
        dec_buf(q);
        if !add_buf(n, d) {
            dec_buf(q);
            add_buf(n, d);
        }
    }
}

pub(crate) const BZ_CUTOFF: usize = 64;

fn div_2_1(n: &mut [u64], d: &[u64], q: &mut [u64], scratch: &mut [u64]) {
    let dlen = d.len();
    if dlen <= BZ_CUTOFF {
        div_buf_of(n, &mut 0, d, q);
        return;
    }

    let lo = dlen / 2;
    let hi = dlen - lo;
    let (q_lo, q_hi) = q.split_at_mut(lo);

    div_3_2(&mut n[lo..], d, lo, q_hi, scratch);
    div_3_2(&mut n[..dlen + lo], d, hi, q_lo, scratch);
}

fn bz_div_alg(
    n: &mut [u64],
    d: &mut [u64],
    out: &mut [u64],
    t: usize,
    mut div_fn: impl FnMut(&mut [u64], &[u64], &mut [u64]),
) {
    let dlen = d.len();
    for i in (0..t).rev() {
        let idx = dlen * i;
        div_fn(&mut n[idx..idx + 2 * dlen], d, &mut out[idx..idx + dlen]);
    }
}

fn find_bz_scratch_size(d: usize) -> usize {
    if d <= BZ_CUTOFF {
        return 0;
    }
    let half = (d + 1) / 2;
    2 * half + find_karatsuba_scratch_size(half, half)
}

fn bz_div_init(n: &mut [u64], d: &mut [u64], out: &mut [u64]) -> Option<(usize, u8)> {
    let dlen = d.len();
    let nlen = n.len();

    if dlen == 1 {
        out[..nlen].copy_from_slice(n);
        let rem = div_prim(out, d[0]);
        n.fill(0);
        n[0] = rem;
        return None;
    }

    let last_lz = d[dlen - 1].leading_zeros() as u8;
    shl_buf(d, last_lz);
    let mut last_n = shl_buf(n, last_lz);

    let t = (nlen - dlen) / dlen;
    let init_idx = dlen * t;
    div_buf_of(&mut n[init_idx..], &mut last_n, d, &mut out[init_idx..]);
    return Some((t, last_lz));
}

pub fn div_vec(n: &mut [u64], d: &mut [u64]) -> Vec<u64> {
    if d.len() == 0 {
        panic!("division by zero");
    }
    if n.len() < d.len() {
        return vec![];
    }
    let mut out = vec![0_u64; n.len() + 1 - d.len()];
    if let Some((t, sh)) = bz_div_init(n, d, &mut out) {
        if t > 0 {
            let size = find_bz_scratch_size(d.len());
            let mut scratch = vec![0_u64; size];
            bz_div_alg(n, d, &mut out, t, |n, d, q| div_2_1(n, d, q, &mut scratch));
        }
        shr_buf(n, sh);
        shr_buf(d, sh);
    }
    return out;
}

fn div_3_2_static<const N: usize>(
    n: &mut [u64],
    d: &[u64],
    half_len: usize,
    q: &mut [u64],
    m: &mut [u64],
    scratch: &mut [u64],
) {
    let (d_lo, d_hi) = d.split_at(half_len);
    if cmp_buf(&n[2 * half_len..], d_hi) == std::cmp::Ordering::Less {
        div_2_1(&mut n[half_len..], d_hi, q, scratch);
    } else {
        q.fill(u64::MAX);
        sub_buf(&mut n[2 * half_len..], d_hi);
        sub_buf(&mut n[half_len..], d_hi);
    }

    karatsuba_alg(q, d_lo, m, scratch);

    if add_buf(n, &m) {
        dec_buf(q);
        if !add_buf(n, d) {
            dec_buf(q);
            add_buf(n, d);
        }
    }
}

fn div_2_1_static<const N: usize>(
    n: &mut [u64],
    d: &[u64],
    q: &mut [u64],
    m: &mut [u64],
    scratch: &mut [u64],
) {
    let dlen = d.len();
    if dlen <= BZ_CUTOFF {
        div_buf_of(n, &mut 0, d, q);
        return;
    }

    let half_len = (dlen + 1) / 2;
    let (q_lo, q_hi) = q.split_at_mut(half_len);
    div_3_2_static::<N>(&mut n[half_len..], d, half_len, q_hi, m, scratch);
    div_3_2_static::<N>(&mut n[0..3 * half_len], d, half_len, q_lo, m, scratch);
}

pub fn div_arr<const N: usize>(n: &mut [u64], d: &mut [u64]) -> [u64; N] {
    let mut out = [0_u64; N];
    if let Some((t, sh)) = bz_div_init(n, d, &mut out) {
        if t > 0 {
            let size = find_bz_scratch_size(d.len());
            let mut scratch = [0_u64; N];
            if size > N {
                let mut m = [0_u64; N];
                bz_div_alg(n, d, &mut out, t, |n, d, q| {
                    div_2_1_static::<N>(n, d, q, &mut m, &mut scratch)
                });
            } else {
                bz_div_alg(n, d, &mut out, t, |n, d, q| div_2_1(n, d, q, &mut scratch));
            }
        }
        shr_buf(n, sh);
        shr_buf(d, sh);
    }
    return out;
}
