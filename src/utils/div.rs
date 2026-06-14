use core::error;
use std::{arch::asm, cmp};

use rustfft::num_traits::zero;

use crate::utils::{
    mul::*, utils::*, ScratchGuard, BZ_CUTOFF, FFT_16BIT_CUTOFF, FFT_CHUNKING_KARATSUBA_CUTOFF,
    FFT_KARATSUBA_CUTOFF, FFT_MID_CUTOFF, SHORT_MUL_CUTOFF,
};

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn div_rem_2_1_x86(q: &mut u64, r: &mut u64, d: u64) {
    asm!(
        "div rcx",
        inout("rax") *q,
        inout("rdx") *r,
        in("rcx") d,
        options(pure, nomem, nostack),
    );
}

#[inline(always)]
unsafe fn div_rem_2_1_asm(q: &mut u64, r: &mut u64, d: u64) {
    #[cfg(target_arch = "aarch64")]
    {
        let val = ((*r as u128) << 64) | (*q as u128);
        let d_u128 = d as u128;
        *r = (val % d_u128) as u64;
        *q = (val / d_u128) as u64;
    }

    #[cfg(target_arch = "x86_64")]
    {
        div_rem_2_1_x86(q, r, d);
    }
}

pub fn div_prim(buf: &mut [u64], prim: u64) -> u64 {
    if prim == 0 {
        panic!("Division by zero error")
    }
    if prim == 1 {
        return 0;
    }

    let mut r = 0;
    for q in buf.iter_mut().rev() {
        unsafe {
            div_rem_2_1_asm(q, &mut r, prim);
        }
    }

    return r;
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sub_mul_of_aarch(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    let overflow: u64;
    asm!(
        "mov {b}, xzr",
        "mov {mc}, xzr",
        "2:",
        "ldr {w}, [{win}]",
        "ldr {dv}, [{den}], #8",
        "mul   {lo}, {dv}, {q}",
        "umulh {hi}, {dv}, {q}",
        "adds {lo}, {lo}, {mc}",
        "adc  {mc}, {hi}, xzr",
        "cmp xzr, {b}",
        "sbcs {w}, {w}, {lo}",
        "cset {b}, cc",
        "str {w}, [{win}], #8",
        "subs {len}, {len}, #1",
        "cbnz {len}, 2b",
        "ldr {w}, [{ofp}]",
        "cmp xzr, {b}",
        "sbcs {w}, {w}, {mc}",
        "cset {overflow}, cc",
        "str {w}, [{ofp}]",
        win = inout(reg) win => _,
        den = inout(reg) d => _,
        ofp = in(reg) of,
        q = in(reg) q,
        len = inout(reg) len => _,
        overflow = out(reg) overflow,
        mc = out(reg) _,
        b = out(reg) _,
        w = out(reg) _,
        dv = out(reg) _,
        lo = out(reg) _,
        hi = out(reg) _,
        options(nostack),
    );
    overflow != 0
}

#[inline(always)]
unsafe fn sub_mul_of_asm(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        sub_mul_of_aarch(win, of, d, q, len)
    }

    #[cfg(target_arch = "x86_64")]
    {
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
}

fn sub_mul_of(win: &mut [u64], of: &mut u64, d: &[u64], q: u64) -> bool {
    unsafe { sub_mul_of_asm(win.as_mut_ptr(), of as *mut u64, d.as_ptr(), q, d.len()) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_u64_x86(a: u64, b: u64) -> (u64, u64) {
    let lo: u64;
    let hi: u64;
    asm!(
        "mul {tmp}",
        tmp = in(reg) b,
        inout("rax") a => lo,
        out("rdx") hi,
        options(nostack, nomem),
    );
    (hi, lo)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_u64_aarch(a: u64, b: u64) -> (u64, u64) {
    let lo: u64;
    let hi: u64;
    asm!(
        "mul {lo}, {a}, {b}",
        "umulh {hi}, {a}, {b}",
        a = in(reg) a,
        b = in(reg) b,
        lo = out(reg) lo,
        hi = out(reg) hi,
        options(nostack, nomem),
    );
    (hi, lo)
}

#[inline(always)]
unsafe fn mul_u64_asm(a: u64, b: u64) -> (u64, u64) {
    #[cfg(target_arch = "aarch64")]
    {
        mul_u64_aarch(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        mul_u64_x86(a, b)
    }
}

fn knuth_est(win: &mut [u64], of: &mut u64, d: &[u64], d1: u64, d0: u64) -> u64 {
    let (mut qhat, rhat_hi, rhat_lo) = if *of >= d1 {
        let (rhat_lo, c) = win.last().unwrap().overflowing_add(d1);
        (u64::MAX, *of - d1 + c as u64, rhat_lo)
    } else {
        let mut r = *of;
        let mut q = *win.last().unwrap();
        unsafe { div_rem_2_1_asm(&mut q, &mut r, d1) };
        (q, 0, r)
    };

    if rhat_hi == 0 {
        let u0 = win[win.len() - 2];
        let (a_hi, a_lo) = unsafe { mul_u64_asm(qhat, d0) };
        if a_hi > rhat_lo || (a_hi == rhat_lo && a_lo > u0) {
            qhat -= 1;
            let (r, carry) = rhat_lo.overflowing_add(d1);
            if !carry {
                let (a_lo, borrow) = a_lo.overflowing_sub(d0);
                let a_hi = a_hi.wrapping_sub(borrow as u64);
                if a_hi > r || (a_hi == r && a_lo > u0) {
                    qhat -= 1;
                }
            }
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

    let d1 = d[d_len - 1];
    let d0 = d[d_len - 2];

    let q_len = n_len - d_len;
    if out.len() > q_len {
        out[q_len] = knuth_est(&mut n[q_len..], of, d, d1, d0);
    }
    for i in (0..q_len).rev() {
        let (win, of) = n[i..].split_at_mut(d_len);
        out[i] = knuth_est(win, &mut of[0], d, d1, d0)
    }
}

fn div_3_2(
    n: &mut [u64],
    d: &[u64],
    d_lo_len: usize,
    q: &mut [u64],
    scratch: &mut [u64],
    mul_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64]),
) {
    let q_len = q.len();
    let (d_lo, d_hi) = d.split_at(d_lo_len);
    if cmp_buf(&n[d.len()..], d_hi) == std::cmp::Ordering::Less {
        div_2_1(&mut n[d_lo_len..], d_hi, q, scratch, mul_alg);
    } else {
        q.fill(u64::MAX);
        sub_buf(&mut n[d.len()..], d_hi);
        add_buf(&mut n[d_lo_len..], d_hi);
    }

    let m = &mut scratch[..q_len + d_lo_len];
    *m.last_mut().unwrap() = 0;
    mul_alg(q, d_lo, m);
    if sub_buf(n, m) {
        dec_buf(q);
        if !add_buf(n, d) {
            dec_buf(q);
            add_buf(n, d);
        }
    }
}

fn div_2_1(
    n: &mut [u64],
    d: &[u64],
    q: &mut [u64],
    scratch: &mut [u64],
    mul_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64]),
) {
    let dlen = d.len();
    if dlen <= BZ_CUTOFF {
        div_buf_of(n, &mut 0, d, q);
        return;
    }

    let lo = dlen / 2;
    let hi = dlen - lo;
    let (q_lo, q_hi) = q.split_at_mut(lo);

    div_3_2(&mut n[lo..], d, lo, q_hi, scratch, mul_alg);
    div_3_2(&mut n[..dlen + lo], d, hi, q_lo, scratch, mul_alg);
}

fn bz_div_alg(
    n: &mut [u64],
    d: &mut [u64],
    out: &mut [u64],
    scratch: &mut [u64],
    t: usize,
    mut mul_alg: impl FnMut(&mut [u64], &[u64], &mut [u64]),
) {
    let dlen = d.len();
    for i in (0..t).rev() {
        let idx = dlen * i;
        div_2_1(
            &mut n[idx..idx + 2 * dlen],
            d,
            &mut out[idx..idx + dlen],
            scratch,
            &mut mul_alg,
        );
    }
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

    let last_lz = d[d.len() - 1].leading_zeros() as u8;
    shl_buf(d, last_lz);
    let mut last_n = shl_buf(n, last_lz);

    let t = (nlen - dlen) / dlen;
    let init_idx = dlen * t;
    div_buf_of(&mut n[init_idx..], &mut last_n, d, &mut out[init_idx..]);
    return Some((t, last_lz));
}

pub fn bz_div_dyn(n: &mut [u64], d: &mut [u64], out: &mut [u64]) {
    if let Some((t, sh)) = bz_div_init(n, d, out) {
        if t > 0 {
            let mut scratch_gaurd = ScratchGuard::acquire();
            bz_div_alg(n, d, out, scratch_gaurd.get(d.len()), t, |n, d, q| {
                mul_dyn(n, d, q);
            });
        }
        shr_buf(n, sh);
        shr_buf(d, sh);
    }
}

pub fn bz_div_static<const N: usize>(n: &mut [u64], d: &mut [u64], out: &mut [u64]) {
    if d.len() == 0 {
        panic!("division by zero");
    }
    if n.len() < d.len() {
        return;
    }
    if let Some((t, sh)) = bz_div_init(n, d, out) {
        if t > 0 {
            let mut scratch = [0; N];
            bz_div_alg(n, d, out, &mut scratch, t, |n, d, q| {
                mul_static::<N>(n, d, q).unwrap();
            });
        }
        shr_buf(n, sh);
        shr_buf(d, sh);
    }
}

#[inline(always)]
fn end_ref(buf: &[u64], idx: usize) -> &[u64] {
    &buf[buf.len().saturating_sub(idx)..]
}

#[inline(always)]
fn end_mut(buf: &mut [u64], idx: usize) -> &mut [u64] {
    let len = buf.len();
    &mut buf[len.saturating_sub(idx)..]
}

fn find_start_p(rcp_len: usize) -> usize {
    const MAX: usize = 15;
    let target = rcp_len / 2;
    if target <= 1 {
        return 1;
    }

    let mut final_p = target;
    while final_p >> final_p.trailing_zeros() > MAX {
        final_p += 1;
    }

    (final_p >> final_p.trailing_zeros()).max(1)
}

pub fn nr_rcp_dyn(denom: &mut [u64], rcp: &mut [u64]) {
    rcp.fill(0);
    let sh = denom[denom.len() - 1].leading_zeros() as u8;
    shl_buf(denom, sh);

    let start_p = find_start_p(rcp.len());
    let final_p = {
        let t = rcp.len() / 2;
        let scale = (t + start_p - 1) / start_p;
        let log2 = (scale - 1).ilog2() + 1;
        start_p << log2
    };
    let err_len = final_p + 1;
    let cor_len = (final_p + 2).max(rcp.len().saturating_sub(final_p));
    let d_len = 2 * final_p + 1;
    let zeros_len = 3 * start_p + 1;

    let mut scratch = ScratchGuard::acquire();
    let (err, cor, zeros, d_work) = if d_len < denom.len() {
        let [err, cor, zeros] = scratch.get_splits([err_len, cor_len, zeros_len]);
        zeros.fill(0);
        (err, cor, zeros, end_mut(denom, d_len))
    } else {
        let [err, cor, zeros, d_work] = scratch.get_splits([err_len, cor_len, zeros_len, d_len]);
        zeros.fill(0);
        let d_idx = d_len - denom.len();
        d_work[..d_idx].fill(0);
        d_work[d_idx..].copy_from_slice(denom);
        (err, cor, zeros, d_work)
    };

    div_buf_of(
        zeros,
        &mut 1,
        end_ref(d_work, 2 * start_p + 1),
        end_mut(rcp, start_p + 1),
    );
    inc_buf(end_mut(rcp, start_p + 1));
    let mut p = start_p;

    while 2 * p + 1 < rcp.len() {
        let (x, d, e, c) = (
            end_ref(rcp, p + 1),
            end_ref(d_work, 2 * p + 1),
            &mut err[..p + 1],
            &mut cor[..2 * p + 2],
        );
        let of = mid_mul_dyn(d, x, e);
        let neg = {
            let (mut acc0, mut acc1, mut acc2) = (of, 0, 0);
            let conv = mul_elem(d, x, 2 * p + 1, &mut acc0, &mut acc1, &mut acc2);
            match conv {
                0 => false,
                u64::MAX => true,
                _ => panic!("middle product is not sign-extended: p={p}, conv={conv}, acc0={acc0}, acc1={acc1}, acc2={acc2}"),
            }
        };
        mul_dyn(x, e, c);
        let hi_c = end_mut(c, p + 1);
        if neg {
            sub_buf(hi_c, x);
            twos_comp(hi_c);
            add_buf(end_mut(rcp, 2 * p + 1), hi_c);
        } else {
            sub_buf(end_mut(rcp, 2 * p + 1), hi_c);
        }
        p *= 2;
    }

    let (x, d, e) = (
        end_ref(rcp, p + 1),
        end_ref(d_work, 2 * p + 1),
        &mut err[..p + 1],
    );
    let of = mid_mul_dyn(d, x, e);
    let neg = {
        let (mut acc0, mut acc1, mut acc2) = (of, 0, 0);
        let conv = mul_elem(d, x, 2 * p + 1, &mut acc0, &mut acc1, &mut acc2);
        match conv {
            0 => false,
            u64::MAX => true,
            _ => {
                let mut exact_e = vec![0u64; p + 1];
                let exact_of = mid_mul_buf(d, x, &mut exact_e);
                let (mut eacc0, mut eacc1, mut eacc2) = (exact_of, 0, 0);
                let exact_conv = mul_elem(d, x, 2 * p + 1, &mut eacc0, &mut eacc1, &mut eacc2);
                panic!(
                    "middle product is not sign-extended: final p={p}, conv={conv}, acc0={acc0}, acc1={acc1}, acc2={acc2}; exact_conv={exact_conv}, exact_next=[{eacc0},{eacc1},{eacc2}], exact_of={exact_of}"
                )
            }
        }
    };

    let final_c_len = rcp.len() - p;
    let hi_c = &mut cor[..final_c_len];
    hi_c[final_c_len - 1] = short_mul_dyn(x, e, &mut hi_c[..final_c_len - 1]);

    if neg {
        let hi_x = end_ref(x, final_c_len);
        sub_buf(hi_c, hi_x);
        twos_comp(hi_c);
        add_buf(rcp, hi_c);
    } else {
        sub_buf(rcp, hi_c);
    }

    shr_buf(denom, sh);
    shl_buf(rcp, sh);
}

fn nr_rem_correction(n: &mut [u64], d: &mut [u64], q: &mut [u64], prod: &mut [u64]) {
    mul_dyn(d, q, prod);
    if sub_buf(n, prod) {
        dec_buf(q);
        return;
    }
    if cmp_buf(n, d).is_ge() {
        inc_buf(q);
        return;
    }
}

pub fn nr_div_dyn(n: &[u64], d: &mut [u64], q: &mut [u64]) {
    const GAURD: usize = 3;
    let mut scratch = ScratchGuard::acquire();
    let [rcp, gaurded_q] = scratch.get_splits([q.len() + GAURD + 1, q.len() + GAURD]);
    nr_rcp_dyn(d, rcp);
    gaurded_q[q.len() + GAURD - 1] =
        short_mul_dyn(n, &rcp[1..], &mut gaurded_q[..q.len() + GAURD - 1]);
    let gaurd = &gaurded_q[..GAURD];
    q.copy_from_slice(&gaurded_q[GAURD..]);
    if gaurd.iter().all(|&r| r == 0) || gaurd.iter().all(|&r| r == u64::MAX) {
        let [prod, num] = scratch.get_splits([n.len(), n.len()]);
        num.copy_from_slice(n);
        nr_rem_correction(num, d, q, prod);
    }
}

// Dummy Functions for now as I change stuff
pub fn div_vec(_: &mut [u64], _: &mut [u64]) -> Vec<u64> {
    Vec::new()
}

pub fn div_arr<const N: usize>(_: &mut [u64], _: &mut [u64]) -> [u64; N] {
    [0; N]
}
