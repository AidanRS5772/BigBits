use crate::utils::{mul::*, utils::*, ScratchGuard, BZ_CUTOFF, SCRATCH_POOL};

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

const NR_GUARD_LIMBS: usize = 2;

fn end_slice_mut(buf: &mut [u64], len: usize) -> &mut [u64] {
    let buf_len = buf.len();
    &mut buf[buf_len.saturating_sub(len)..]
}

fn end_slice_ref(buf: &[u64], len: usize) -> &[u64] {
    let buf_len = buf.len();
    &buf[buf_len.saturating_sub(len)..]
}

fn nr_rcp_required_d_len(pf: usize) -> usize {
    let mut p = 1;
    while 2 * p < pf {
        p *= 2;
    }
    2 * p + 1
}

fn nr_rcp_core(
    d: &mut [u64],
    out: &mut [u64],
    guard: usize,
    rcp: &mut [u64],
    e: &mut [u64],
    c: &mut [u64],
    init_win: &mut [u64],
    mut mid_mul: impl FnMut(&[u64], &[u64], &mut [u64]) -> u64,
    mut short_mul: impl FnMut(&[u64], &[u64], &mut [u64]) -> u64,
) {
    let dlen = d.len();
    let pf = out.len() + guard;
    debug_assert!(
        out.len() >= 2,
        "reciprocal output must have at least 2 limbs"
    );
    debug_assert!(
        rcp.len() >= pf && e.len() >= pf && c.len() >= pf,
        "reciprocal scratch buffers are too small"
    );
    debug_assert!(
        init_win.len() > dlen,
        "initial reciprocal window is too small"
    );
    debug_assert!(
        dlen >= nr_rcp_required_d_len(pf),
        "divisor is too short for this reciprocal precision"
    );

    let sh = d[dlen - 1].leading_zeros() as u8;
    shl_buf(d, sh);

    let rcp = &mut rcp[..pf];
    let e = &mut e[..pf];
    let c = &mut c[..pf];
    let init_win = &mut init_win[..=dlen];
    init_win.fill(0);
    rcp.fill(0);
    let mut of = 1;
    div_buf_of(init_win, &mut of, d, end_slice_mut(rcp, 2));
    if of != 0 || init_win.iter().any(|&x| x != 0) {
        inc_buf(end_slice_mut(rcp, 2));
    }
    let mut p = 1;
    while 2 * p < pf {
        mid_mul(
            end_slice_ref(d, 2 * p + 1),
            end_slice_ref(rcp, p + 1),
            &mut e[..p + 1],
        );
        mul_dyn(end_slice_ref(rcp, p + 1), &e[..p + 1], &mut c[..2*p+1]);
        sub_buf(end_slice_mut(rcp, 2 * p + 1), &c[p..2*p+1]);
        inc_buf(end_slice_mut(rcp, 2 * p + 1));
        p *= 2
    }

    //final iteration
    mid_mul(
        end_slice_ref(d, 2 * p + 1),
        end_slice_ref(rcp, p + 1),
        &mut e[..p + 1],
    );
    c[pf - p] = short_mul(end_slice_ref(rcp, p + 1), &e[..p + 1], &mut c[..pf - p]);
    let rcp_out = &mut rcp[guard..];
    sub_buf(rcp_out, &c[guard..pf - p + 1]);
    inc_buf(rcp_out);

    shl_buf(rcp_out, sh);
    out.copy_from_slice(&rcp[guard..]);
    shr_buf(d, sh);
}

fn nr_rcp_dyn(d: &mut [u64], out: &mut [u64]) {
    let pf = out.len() + NR_GUARD_LIMBS;
    let mut scratch = ScratchGuard::acquire();
    let [work, e, c, init_win] = scratch.get_splits([pf, pf, pf, d.len() + 1]);
    nr_rcp_core(
        d,
        out,
        NR_GUARD_LIMBS,
        work,
        e,
        c,
        init_win,
        mid_mul_dyn,
        short_mul_dyn,
    );
}

fn nr_rcp_static<const N: usize>(d: &mut [u64], out: &mut [u64]) -> Result<(), ()> {
    let pf = out.len();
    if out.len() < 2 || out.len() > N || pf > N || d.len() + 1 > N {
        return Err(());
    }
    if d.len() < nr_rcp_required_d_len(pf) {
        return Err(());
    }

    let mut work = [0; N];
    let mut e = [0; N];
    let mut c = [0; N];
    let mut init_win = [0; N];
    nr_rcp_core(
        d,
        out,
        0,
        &mut work,
        &mut e,
        &mut c,
        &mut init_win,
        mid_mul_static::<N>,
        short_mul_static::<N>,
    );
    Ok(())
}

fn rem_adj_core(
    n: &mut [u64],
    d: &[u64],
    q: &mut [u64],
    prod: &mut [u64],
    mut mul: impl FnMut(&[u64], &[u64], &mut [u64]) -> u64,
) {
    debug_assert_eq!(prod.len(), n.len(), "product buffer has wrong length");
    prod.fill(0);
    let of = mul(d, q, prod);
    if of != 0 || sub_buf(n, prod) {
        dec_buf(q);
        add_buf(n, d);
    }
    if cmp_buf(n, d).is_ge() {
        inc_buf(q);
        sub_buf(n, d);
    }
}

fn rem_adj(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    let mut scratch = ScratchGuard::acquire();
    let prod = scratch.get(n.len());
    rem_adj_core(n, d, q, prod, mul_dyn);
}

fn rem_adj_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    let mut prod = [0; N];
    rem_adj_core(n, d, q, &mut prod[..n.len()], |d, q, prod| {
        mul_static::<N>(d, q, prod).unwrap()
    });
}

pub fn nr_div_dyn(n: &[u64], d: &mut [u64], out: &mut [u64]) {
    let guard = d.len().ilog10() as usize;
    let qlen = n.len() - d.len() + 1;
    debug_assert_eq!(
        qlen,
        out.len(),
        "out buffer is not large enough for division"
    );

    let p = qlen + guard;
    let mut scratch = ScratchGuard::acquire();
    let [rcp, hi] = scratch.get_splits([p, p]);
    nr_rcp_dyn(d, rcp);
    hi[p - 1] = short_mul_dyn(n, rcp, &mut hi[..p - 1]);
    out.copy_from_slice(&hi[guard..]);
    if hi[..guard].iter().all(|&x| x == 0) {
        let tmp_n = scratch.get(n.len());
        tmp_n.copy_from_slice(n);
        rem_adj(tmp_n, d, out);
    }
}

pub fn nr_div_rem_dyn(n: &mut [u64], d: &mut [u64], out: &mut [u64]) {
    let qlen = n.len() - d.len() + 1;
    debug_assert_eq!(
        qlen,
        out.len(),
        "out buffer is not large enough for division"
    );
    let mut scratch = ScratchGuard::acquire();
    let rcp = scratch.get(qlen);
    nr_rcp_dyn(d, rcp);
    out[qlen - 1] = short_mul_dyn(n, rcp, &mut out[..qlen - 1]);
    rem_adj(n, d, out);
}

pub fn nr_div_rem_static<const N: usize>(
    n: &mut [u64],
    d: &mut [u64],
    out: &mut [u64],
) -> Result<(), ()> {
    if d.len() == 0 {
        panic!("division by zero");
    }
    if n.len() < d.len() {
        return Ok(());
    }

    let qlen = n.len() - d.len() + 1;
    debug_assert_eq!(
        qlen,
        out.len(),
        "out buffer is not large enough for division"
    );
    if n.len() > N || d.len() > N || qlen > N || out.len() != qlen {
        return Err(());
    }

    if d.len() == 1 {
        out.copy_from_slice(n);
        let rem = div_prim(out, d[0]);
        n.fill(0);
        n[0] = rem;
        return Ok(());
    }
    let p = qlen.checked_add(NR_GUARD_LIMBS).ok_or(())?;
    if p > N {
        return Err(());
    }

    let mut rcp = [0; N];
    let mut hi = [0; N];
    nr_rcp_static::<N>(d, &mut rcp[..p])?;
    hi[p - 1] = short_mul_static::<N>(n, &rcp[..p], &mut hi[..p - 1]);
    out.copy_from_slice(&hi[NR_GUARD_LIMBS..p]);
    rem_adj_static::<N>(n, d, out);
    Ok(())
}

pub fn div_vec(n: &mut [u64], d: &mut [u64]) -> Vec<u64> {
    if d.len() == 0 {
        panic!("division by zero");
    }
    if n.len() < d.len() {
        return vec![];
    }
    let mut out = vec![0_u64; n.len() + 1 - d.len()];
    bz_div_dyn(n, d, &mut out);
    return out;
}

pub fn div_arr<const N: usize>(n: &mut [u64], d: &mut [u64]) -> [u64; N] {
    if d.len() == 0 {
        panic!("division by zero");
    }

    let mut out = [0_u64; N];
    if n.len() < d.len() {
        return out;
    }

    let qlen = n.len() + 1 - d.len();
    if nr_div_rem_static::<N>(n, d, &mut out[..qlen]).is_err() {
        bz_div_static::<N>(n, d, &mut out);
    }
    return out;
}
