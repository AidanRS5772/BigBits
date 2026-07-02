use core::error;
use std::{arch::asm, cmp};

use rustfft::num_traits::zero;

use crate::utils::{mul::*, utils::*, ScratchGuard, BZ_CUTOFF, BZ_TOP_PADDED_COST_SCALE};

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
// TODO: Need to impliment assembly version for x86
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

pub fn div_3_2(
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

pub fn div_2_1(
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

pub fn bz_top_block_knuth_work(dlen: usize, qlen: usize) -> f64 {
    (dlen as f64) * (qlen as f64)
}

pub fn bz_top_block_padded_work(d: usize) -> f64 {
    const SCHOOL_TO_KARATSUBA: f64 = 3.394147384;
    const KARATSUBA_TO_FFT: f64 = 2.733850808;
    match dyn_dispatch(d, d) {
        DynDispatch::Prim | DynDispatch::Prim2 | DynDispatch::School => d as f64 * d as f64,
        DynDispatch::Karatsuba => SCHOOL_TO_KARATSUBA * (d as f64).powf(1.5849625007),
        DynDispatch::FFT | DynDispatch::NTT => {
            KARATSUBA_TO_FFT * (d as f64) * (d as f64).log2() * (d as f64).log2()
        }
    }
}

pub fn use_bz_for_top_block_with_scale(dlen: usize, qlen: usize, padded_cost_scale: f64) -> bool {
    if dlen <= BZ_CUTOFF {
        return false;
    }
    if qlen == 0 {
        return false;
    }

    padded_cost_scale * bz_top_block_padded_work(dlen) <= bz_top_block_knuth_work(dlen, qlen)
}

pub fn use_bz_for_top_block(dlen: usize, qlen: usize) -> bool {
    use_bz_for_top_block_with_scale(dlen, qlen, BZ_TOP_PADDED_COST_SCALE)
}

fn bz_div_init_static<const N: usize>(
    n: &mut [u64],
    d: &mut [u64],
    out: &mut [u64],
    scratch: &mut [u64; N],
) -> Option<(usize, u8)> {
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
    let init_len = nlen - init_idx;
    let init_qlen = init_len - dlen + 1;
    if !use_bz_for_top_block(dlen, init_qlen) || dlen > N / 2 {
        div_buf_of(&mut n[init_idx..], &mut last_n, d, &mut out[init_idx..]);
    } else {
        let mut top_n_storage = [0; N];
        let top_n = &mut top_n_storage[..2 * dlen];
        top_n[..init_len].copy_from_slice(&n[init_idx..]);
        top_n[init_len] = last_n;

        if out.len() >= dlen {
            let q_tmp = &mut out[..dlen];
            div_2_1(top_n, d, q_tmp, &mut scratch[..dlen], &mut |n, d, q| {
                mul_static::<N>(n, d, q).unwrap();
            });

            out.copy_within(0..init_qlen, init_idx);
            n[init_idx..].fill(0);
            n[init_idx..init_idx + dlen].copy_from_slice(&top_n[..dlen]);
        } else {
            let (q_tmp, div_scratch) = scratch[..2 * dlen].split_at_mut(dlen);
            div_2_1(top_n, d, q_tmp, div_scratch, &mut |n, d, q| {
                mul_static::<N>(n, d, q).unwrap();
            });

            out.copy_from_slice(&q_tmp[..init_qlen]);
            n.fill(0);
            n[..dlen].copy_from_slice(&top_n[..dlen]);
        }
    }
    if t == 0 {
        shr_buf(n, last_lz);
        shr_buf(d, last_lz);
        None
    } else {
        Some((t, last_lz))
    }
}

fn bz_div_init_dyn(n: &mut [u64], d: &mut [u64], out: &mut [u64]) -> Option<(usize, u8)> {
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
    let init_len = nlen - init_idx;
    let init_qlen = init_len - dlen + 1;
    if !use_bz_for_top_block(dlen, init_qlen) {
        div_buf_of(&mut n[init_idx..], &mut last_n, d, &mut out[init_idx..]);
    } else {
        let mut scratch = ScratchGuard::acquire();
        if out.len() >= dlen {
            let [top_n, div_scratch] = scratch.get_splits([2 * dlen, dlen]);
            top_n[..init_len].copy_from_slice(&n[init_idx..]);
            top_n[init_len] = last_n;
            top_n[init_len + 1..].fill(0);

            let q_tmp = &mut out[..dlen];
            div_2_1(top_n, d, q_tmp, div_scratch, &mut |n, d, q| {
                mul_dyn(n, d, q);
            });

            out.copy_within(0..init_qlen, init_idx);
            n[init_idx..].fill(0);
            n[init_idx..init_idx + dlen].copy_from_slice(&top_n[..dlen]);
        } else {
            let [top_n, q_tmp, div_scratch] = scratch.get_splits([2 * dlen, dlen, dlen]);
            top_n[..init_len].copy_from_slice(&n[init_idx..]);
            top_n[init_len] = last_n;
            top_n[init_len + 1..].fill(0);

            div_2_1(top_n, d, q_tmp, div_scratch, &mut |n, d, q| {
                mul_dyn(n, d, q);
            });

            out.copy_from_slice(&q_tmp[..init_qlen]);
            n[..].fill(0);
            n[..dlen].copy_from_slice(&top_n[..dlen]);
        }
    }
    if t == 0 {
        shr_buf(n, last_lz);
        shr_buf(d, last_lz);
        None
    } else {
        Some((t, last_lz))
    }
}

pub fn bz_div_dyn(n: &mut [u64], d: &mut [u64], out: &mut [u64]) {
    if let Some((t, sh)) = bz_div_init_dyn(n, d, out) {
        let mut scratch_gaurd = ScratchGuard::acquire();
        bz_div_alg(n, d, out, scratch_gaurd.get(d.len()), t, |n, d, q| {
            mul_dyn(n, d, q);
        });
        shr_buf(n, sh);
        shr_buf(d, sh);
    }
}

pub fn bz_div_static<const N: usize>(n: &mut [u64], d: &mut [u64], out: &mut [u64]) {
    let mut scratch = [0; N];
    if let Some((t, sh)) = bz_div_init_static::<N>(n, d, out, &mut scratch) {
        bz_div_alg(n, d, out, &mut scratch, t, |n, d, q| {
            mul_static::<N>(n, d, q).unwrap();
        });
        shr_buf(n, sh);
        shr_buf(d, sh);
    }
}

#[inline(always)]
pub fn end_ref(buf: &[u64], idx: usize) -> &[u64] {
    &buf[buf.len().saturating_sub(idx)..]
}

#[inline(always)]
pub fn end_mut(buf: &mut [u64], idx: usize) -> &mut [u64] {
    let len = buf.len();
    &mut buf[len.saturating_sub(idx)..]
}

// Backward ceiling-halving schedule: sizes[0] = p_target, sizes[i+1] = ceil(sizes[i]/2),
// stopping above precision 1. Walked in reverse every step is p -> 2p (even target)
// or p -> 2p-1 (odd target, one truncated limb), so the chain lands exactly on
// p_target with no oversized final iteration.
pub fn nr_rcp_schedule(p_target: usize, sizes: &mut [usize; 64]) -> usize {
    let mut steps = 0;
    let mut q = p_target;
    while q > 1 {
        sizes[steps] = q;
        steps += 1;
        q = (q + 1) >> 1;
    }
    steps
}

pub fn mid_mul_sign_ext(d: &[u64], x: &[u64], e: &mut [u64], p: usize) -> Option<(bool, usize)> {
    let mut e_idx = p + 1;
    let (mut acc0, mut acc1) = mid_mul_dyn(d, x, &mut e[..e_idx]);
    let mut acc2 = 0;
    let mut val = mul_elem(d, x, p + e_idx, &mut acc0, &mut acc1, &mut acc2);
    while val != 0 && val != u64::MAX {
        e[e_idx] = val;
        e_idx += 1;
        if e_idx > 2 * p + 1 {
            return None;
        }
        val = mul_elem(d, x, p + e_idx, &mut acc0, &mut acc1, &mut acc2);
    }
    Some((val == u64::MAX, e_idx))
}

pub fn bz_rcp_seed_dyn(d: &[u64], rcp: &mut [u64]) {
    if rcp.is_empty() {
        return;
    }

    let n_len = d.len() + rcp.len();
    let q_len = rcp.len() + 1;
    let mut scratch = ScratchGuard::acquire();
    let [n, d_work, q] = scratch.get_splits([n_len, d.len(), q_len]);

    n.fill(0);
    n[n_len - 1] = 1;
    d_work.copy_from_slice(d);
    q.fill(0);

    bz_div_dyn(n, d_work, q);
    debug_assert_eq!(q[rcp.len()], 0);
    rcp.copy_from_slice(&q[..rcp.len()]);
    inc_buf(rcp);
}

// Refines rcp from precision p to 2p (trunc = false) or 2p - 1 (trunc = true,
// dropping the lowest limb of the correction).
pub fn nr_rcp_refine_step_dyn(
    d: &[u64],
    rcp: &mut [u64],
    err: &mut [u64],
    cor: &mut [u64],
    p: usize,
    trunc: bool,
) -> bool {
    let skip = trunc as usize;
    debug_assert!(rcp.len() + skip == 2 * p + 1);

    let x = end_ref(rcp, p + 1);
    let Some((neg, e_len)) = mid_mul_sign_ext(d, x, err, p) else {
        return false;
    };

    let extra = e_len - p - 1;
    let c = &mut cor[..e_len];
    c[e_len - 1] = short_mul_dyn(x, &err[..e_len], &mut c[..e_len - 1]);
    if neg {
        sub_buf(&mut c[extra..], x);
        twos_comp(c);
        add_buf(rcp, &c[skip..]);
    } else {
        sub_buf(rcp, &c[skip..]);
    }

    true
}

pub fn nr_rcp_dyn(denom: &mut [u64], rcp: &mut [u64]) {
    if rcp.is_empty() {
        return;
    }

    rcp.fill(0);
    let sh = denom[denom.len() - 1].leading_zeros() as u8;
    shl_buf(denom, sh);

    let mut sizes = [0usize; 64];
    let steps = nr_rcp_schedule(rcp.len() - 1, &mut sizes);

    // The widest step reads d at 2p+1 limbs for its input precision p, and
    // mid_mul_sign_ext can write one limb past 2p+1 before bailing out. The seed
    // numerator holds B^(seed_r + 2) for a seed_r limb quotient.
    let pen_p = if steps >= 2 { sizes[1] } else { 1 };
    let d_len = 2 * pen_p + 1;
    let err_len = 2 * pen_p + 2;
    let seed_r = rcp.len().min(2);

    let mut scratch = ScratchGuard::acquire();
    let (err, cor, seed_n, d_work) = if d_len < denom.len() {
        let [err, cor, seed_n] = scratch.get_splits([err_len, err_len, seed_r + 2]);
        (err, cor, seed_n, end_mut(denom, d_len))
    } else {
        let [err, cor, seed_n, d_work] = scratch.get_splits([err_len, err_len, seed_r + 2, d_len]);
        let d_idx = d_len - denom.len();
        d_work[..d_idx].fill(0);
        d_work[d_idx..].copy_from_slice(denom);
        (err, cor, seed_n, d_work)
    };

    // Precision-1 seed: floor(B^(seed_r + 2) / top 3 limbs of d) + 1 by knuth division.
    seed_n.fill(0);
    let mut seed_of = 1;
    div_buf_of(
        seed_n,
        &mut seed_of,
        end_ref(d_work, 3),
        end_mut(rcp, seed_r),
    );
    inc_buf(end_mut(rcp, seed_r));

    let mut p = 1;
    for &q in sizes[..steps].iter().rev() {
        if !nr_rcp_refine_step_dyn(
            end_ref(d_work, 2 * p + 1),
            end_mut(rcp, q + 1),
            err,
            cor,
            p,
            q & 1 == 1,
        ) {
            bz_rcp_seed_dyn(end_ref(d_work, (2 * q + 1).min(d_len)), end_mut(rcp, q + 1));
        }
        p = q;
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
