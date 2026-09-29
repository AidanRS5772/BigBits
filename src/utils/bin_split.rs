use std::marker::PhantomData;

use crate::utils::{
    div::{rcp_dyn, rcp_static},
    mul::{hi_mul_dyn, hi_mul_static, mul_add_prim, mul_dyn, mul_prim},
    utils::{add_buf, add_mul, add_prim, buf_len, shl_buf, shr_buf, trim_lz},
    ScratchGuard, PARTIAL_MUL_CUTOFF,
};

fn split_bits(bits: u64) -> (usize, u8) {
    ((bits / 64) as usize, (bits % 64) as u8)
}

fn push(buf: &mut [u64], len: &mut usize, c: u64) {
    if c != 0 {
        buf[*len] = c;
        *len += 1;
    }
}

fn prepend_shl(x: &mut [u64], x_len: &mut usize, sh: u64) -> u64 {
    if *x_len == 0 {
        return 0;
    }
    let (sl, sb) = split_bits(sh);
    if sl != 0 {
        x[..*x_len + sl].copy_within(..*x_len, sl);
        x[..sl].fill(0);
        *x_len += sl;
    }
    shl_buf(&mut x[sl..*x_len], sb)
}

// Exact P and Q over the terms [a, b); p needs b - a + R(b - a) / 64 + 1 limbs
// and q needs b - a.
fn bbp_leaf_terms<S: BBPSeries>(a: u64, b: u64, p: &mut [u64], q: &mut [u64]) -> (usize, usize) {
    q[0] = 1;
    let (mut p_len, mut q_len) = (0, 1);
    let mut n = a;
    while n < b {
        let (mut u, mut v, mut j) = (S::q(n), S::p(n), 1);
        n += 1;
        while S::R < 64 && n < b {
            let w = S::q(n) as u128;
            let (uw, vw) = (u as u128 * w, v as u128 * w);
            if (uw >> 64 | vw >> (64 - S::R)) != 0 {
                break;
            }
            let vw = (vw << S::R) + u as u128 * S::p(n) as u128;
            if vw >> 64 != 0 {
                break;
            }
            (u, v, j) = (uw as u64, vw as u64, j + 1);
            n += 1;
        }
        // P = (P * u) << Rj + Q * v, shifting u instead of P when the sub-limb
        // shift fits.
        let (sh, sb) = (S::R * j, (S::R * j % 64) as u32);
        let f = if u.leading_zeros() >= sb { sb } else { 0 };
        let c = mul_prim(&mut p[..p_len], u << f);
        push(p, &mut p_len, c);
        let c = prepend_shl(p, &mut p_len, sh - f as u64);
        push(p, &mut p_len, c);
        p_len = p_len.max(q_len);
        let c = add_mul(&mut p[..p_len], &q[..q_len], v);
        push(p, &mut p_len, c);
        let c = mul_prim(&mut q[..q_len], u);
        push(q, &mut q_len, c);
    }
    (buf_len(&p[..p_len]), q_len)
}

// Like push, but once buf is full the lowest limb is dropped into e.
fn push_or_drop(buf: &mut [u64], len: &mut usize, e: &mut usize, c: u64) {
    if c == 0 {
        return;
    }
    if *len == buf.len() {
        buf.copy_within(1.., 0);
        *len -= 1;
        *e += 1;
    }
    buf[*len] = c;
    *len += 1;
}

// buf[..len] <<= bits, moving whole limbs into e once buf is full.
fn shl_or_drop(buf: &mut [u64], len: &mut usize, e: &mut usize, bits: u64) {
    if *len == 0 {
        return;
    }
    let (sl, sb) = split_bits(bits);
    let m = sl.min(buf.len() - *len);
    if m != 0 {
        buf.copy_within(..*len, m);
        buf[..m].fill(0);
        *len += m;
    }
    *e += sl - m;
    let c = shl_buf(&mut buf[..*len], sb);
    push_or_drop(buf, len, e, c);
}

// P / B^e and exact Q over the terms [a, b), returning (p_len, q_len, e). q
// needs b - a limbs, and p keeps P exact (e = 0) with b - a + R(b - a) / 64 + 1.
fn engel_leaf_terms<S: EngelSeries>(
    a: u64,
    b: u64,
    p: &mut [u64],
    q: &mut [u64],
) -> (usize, usize, usize) {
    q[0] = 1;
    let (mut p_len, mut q_len, mut e) = (0, 1, 0);
    let mut n = a;
    while n < b {
        let (mut u, mut v, mut j) = (S::q(n), S::p(n), 1);
        n += 1;
        while S::R < 64 && n < b {
            let w = S::q(n) as u128;
            let (uw, vw) = (u as u128 * w, v as u128 * w);
            if (uw >> 64 | vw >> (64 - S::R)) != 0 {
                break;
            }
            let vw = (vw << S::R) + S::p(n) as u128;
            if vw >> 64 != 0 {
                break;
            }
            (u, v, j) = (uw as u64, vw as u64, j + 1);
            n += 1;
        }
        // P = (P * u) << Rj + v, shifting u instead of P when the sub-limb shift
        // fits; v falls below P's precision once e > 0.
        let (sh, sb) = (S::R * j, (S::R * j % 64) as u32);
        let f = if u.leading_zeros() >= sb { sb } else { 0 };
        // With no shift left, v rides in as the multiply's carry.
        let fused = sh == f as u64 || p_len == 0;
        let add = if fused && e == 0 { v } else { 0 };
        let c = mul_add_prim(&mut p[..p_len], u << f, add);
        push_or_drop(p, &mut p_len, &mut e, c);
        if !fused {
            shl_or_drop(p, &mut p_len, &mut e, sh - f as u64);
            if e == 0 && add_prim(&mut p[..p_len], v) {
                push_or_drop(p, &mut p_len, &mut e, 1);
            }
        }
        let c = mul_prim(&mut q[..q_len], u);
        push(q, &mut q_len, c);
    }
    (p_len, q_len, e)
}

// P / B^e and exact Q and R over the terms [a, b), returning (p_len, q_len,
// r_len, e). q and r need b - a limbs, and p keeps P exact (e = 0) with
// b - a + R(b - a) / 64 + 1.
fn hyper_leaf_terms<S: HyperSeries>(
    a: u64,
    b: u64,
    p: &mut [u64],
    q: &mut [u64],
    r: &mut [u64],
) -> (usize, usize, usize, usize) {
    (q[0], r[0]) = (1, 1);
    let (mut p_len, mut q_len, mut r_len, mut e) = (0, 1, 1, 0);
    let mut n = a;
    while n < b {
        let (mut u, mut v, mut w, mut j) = (S::q(n), S::r(n), S::p(n), 1);
        n += 1;
        while S::R < 64 && n < b {
            let (x, y) = (S::q(n) as u128, S::r(n) as u128);
            let (ux, vy, wx) = (u as u128 * x, v as u128 * y, w as u128 * x);
            if (ux >> 64 | vy >> 64 | wx >> (64 - S::R)) != 0 {
                break;
            }
            let wx = (wx << S::R) + v as u128 * S::p(n) as u128;
            if wx >> 64 != 0 {
                break;
            }
            (u, v, w, j) = (ux as u64, vy as u64, wx as u64, j + 1);
            n += 1;
        }
        // P = (P * u) << Rj + R * w, shifting u instead of P when the sub-limb
        // shift fits.
        let (sh, sb) = (S::R * j, (S::R * j % 64) as u32);
        let f = if u.leading_zeros() >= sb { sb } else { 0 };
        let c = mul_prim(&mut p[..p_len], u << f);
        push_or_drop(p, &mut p_len, &mut e, c);
        shl_or_drop(p, &mut p_len, &mut e, sh - f as u64);
        // R * w joins P at B^e; dropping R's low e limbs first errs by under
        // w < B, so only P's lowest limb is inexact.
        let rs = &r[e.min(r_len)..r_len];
        p_len = p_len.max(rs.len());
        let c = add_mul(&mut p[..p_len], rs, w);
        push_or_drop(p, &mut p_len, &mut e, c);
        let c = mul_prim(&mut r[..r_len], v);
        push(r, &mut r_len, c);
        let c = mul_prim(&mut q[..q_len], u);
        push(q, &mut q_len, c);
    }
    (buf_len(&p[..p_len]), q_len, r_len, e)
}

fn fit(mut v: Vec<u64>, n: usize) -> Vec<u64> {
    if v.capacity() < n {
        return vec![0u64; n];
    }
    v.clear();
    v.resize(n, 0);
    v
}

fn mul_len(a: &[u64], b: &[u64]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (x, y) = (a[a.len() - 1] as u128, b[b.len() - 1] as u128);
    a.len() + b.len() - (x * y + x + y < 1 << 64) as usize
}

fn hi_mul_drop<const N: usize>(a: &[u64], b: &[u64], drop: usize, out: &mut [u64]) -> usize {
    if a.is_empty() || b.is_empty() || drop >= a.len() + b.len() {
        return 0;
    }
    let core = a.len() + b.len() - 1 - drop;
    if core < out.len() {
        out[core] = hi_mul_static::<N>(a, b, &mut out[..core]);
        return buf_len(&out[..=core]);
    }
    let (lowest, rest) = out.split_at_mut((core > PARTIAL_MUL_CUTOFF) as usize);
    lowest.fill(0);
    let top = hi_mul_static::<N>(a, b, rest);
    debug_assert_eq!(top, 0);
    buf_len(out)
}

fn ratio_to_fraction(
    p: &[u64],
    q: &[u64],
    shift: i64,
    out: &mut [u64],
    x: &mut [u64],
    y: &mut [u64],
    rcp: impl FnOnce(&[u64], &mut [u64]),
    hi_mul: impl FnOnce(&[u64], &[u64], &mut [u64]) -> u64,
) {
    let len = out.len();
    debug_assert!(x.len() == len + 2 && y.len() >= len + 3);
    out.fill(0);
    if p.is_empty() || len == 0 {
        return
    }

    // x ≈ B^(q.len() + x.len() - 1) / q, so p * x / 2^bits = S * B^len, with
    // w >= 2 whole limbs in bits.
    rcp(q, x);
    let bits = 64 * (q.len() + x.len() - 1 - len) as i64 + shift;
    debug_assert!(bits >= 128, "finalize needs w >= 2 whole limbs");
    let (w, v) = split_bits(bits as u64);
    let full = p.len() + x.len();
    if full < w {
        return;
    }

    let y = &mut y[..full - w + 1];
    let core = y.len() - 1;
    y[core] = hi_mul(p, x, &mut y[..core]);
    shr_buf(y, v);
    let y = &y[1..];
    let n = y.len().min(len);
    out[..n].copy_from_slice(&y[..n]);
    debug_assert!(y[n..].iter().all(|&l| l == 0), "finalize expects S < 1");
}

fn ratio_to_fraction_dyn(p: &[u64], q: &[u64], shift: i64, out: &mut [u64]) {
    let len = out.len();
    let mut scratch = ScratchGuard::acquire();
    let [x, y] = scratch.get_splits([len + 2, len + 3]);
    ratio_to_fraction(p, q, shift, out, x, y, rcp_dyn, hi_mul_dyn);
}

fn ratio_to_fraction_static<const N: usize>(p: &[u64], q: &[u64], shift: i64, out: &mut [u64]) {
    let len = out.len();
    assert!(len + 3 <= N, "finalize output needs out.len() + 3 <= N");
    let (mut x, mut y) = ([0; N], [0; N]);
    let (x, y) = (&mut x[..len + 2], &mut y[..len + 3]);
    ratio_to_fraction(p, q, shift, out, x, y, rcp_static::<N>, hi_mul_static::<N>);
}

pub trait Node {
    fn merge(l: Self, r: Self) -> Self;
    fn leaf(a: u64, b: u64) -> Self;
    fn finalize(&self, out: &mut [u64]);
}

pub fn bin_split<N, const CUTOFF: u64>(a: u64, b: u64) -> N
where
    N: Node,
{
    const { assert!(CUTOFF != 0) };
    debug_assert!(a < b);
    if b - a < CUTOFF {
        return N::leaf(a, b);
    }
    let m = (a + b) / 2;
    let l = bin_split::<N, CUTOFF>(a, m);
    let r = bin_split::<N, CUTOFF>(m, b);
    N::merge(l, r)
}

pub trait BBPSeries {
    const R: u64;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
}

#[derive(Debug, Clone)]
pub struct DynNodeBBP<S: BBPSeries> {
    a: u64,
    b: u64,
    p: Vec<u64>,
    q: Vec<u64>,
    _s: PhantomData<S>,
}

impl<S: BBPSeries> Node for DynNodeBBP<S> {
    fn leaf(a: u64, b: u64) -> Self {
        let terms = (b - a) as usize;
        let words = split_bits(S::R * (b - a)).0 + 1;
        let (mut p, mut q) = (vec![0; terms + words], vec![0; terms]);
        let (p_len, q_len) = bbp_leaf_terms::<S>(a, b, &mut p, &mut q);
        p.truncate(p_len);
        q.truncate(q_len);
        DynNodeBBP {
            a,
            b,
            p,
            q,
            _s: PhantomData,
        }
    }

    fn merge(l: Self, r: Self) -> Self {
        let (sl, sb) = split_bits(S::R * (r.b - r.a)); // r(b - m)
        let pl = l.p.len() + r.q.len();
        let ql = l.q.len() + r.q.len();

        let mut scratch = ScratchGuard::acquire();
        let t = scratch.get(l.q.len() + r.p.len());
        mul_dyn(&l.q, &r.p, t);

        let mut p = fit(r.p, (sl + pl + 1).max(t.len()));
        mul_dyn(&l.p, &r.q, &mut p[sl..sl + pl]);
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);
        if add_buf(&mut p, t) {
            p.push(1);
        }
        trim_lz(&mut p);

        let mut q = fit(l.p, ql);
        mul_dyn(&l.q, &r.q, &mut q);
        trim_lz(&mut q);

        DynNodeBBP {
            a: l.a,
            b: r.b,
            p,
            q,
            _s: PhantomData,
        }
    }

    fn finalize(&self, out: &mut [u64]) {
        ratio_to_fraction_dyn(&self.p, &self.q, (S::R * (self.b - self.a)) as i64, out);
    }
}

#[derive(Debug, Clone)]
pub struct StaticNodeBBP<const N: usize, S: BBPSeries> {
    a: u64,
    b: u64,
    p: [u64; N],
    p_len: usize,
    q: [u64; N],
    q_len: usize,
    _s: PhantomData<S>,
}

impl<const N: usize, S: BBPSeries> Node for StaticNodeBBP<N, S> {
    fn leaf(a: u64, b: u64) -> Self {
        let (mut p, mut q) = ([0; N], [0; N]);
        let (p_len, mut q_len) = bbp_leaf_terms::<S>(a, b, &mut p, &mut q);
        let c = prepend_shl(&mut q, &mut q_len, S::R * (b - a));
        push(&mut q, &mut q_len, c);

        StaticNodeBBP {
            a,
            b,
            p,
            p_len,
            q,
            q_len,
            _s: PhantomData,
        }
    }

    fn merge(mut l: Self, mut r: Self) -> Self {
        const { assert!(N >= 2) };
        let (sl, sb) = split_bits(S::R * (l.b - l.a)); // r(m - a)
        let (lp, lq) = (&l.p[..l.p_len], &l.q[..l.q_len]);
        let (rp, rq) = (&r.p[..r.p_len], &r.q[..r.q_len]);

        let k = mul_len(lp, rq)
            .max(mul_len(lq, rp).saturating_sub(sl))
            .max(mul_len(lq, rq))
            .saturating_sub(N);

        let mut t = [0; N];
        let t_len = hi_mul_drop::<N>(lp, rq, k, &mut t);
        let t2_len = hi_mul_drop::<N>(lq, rp, k + sl, &mut l.p);
        shr_buf(&mut l.p[..t2_len], sb);
        let t2_len = buf_len(&l.p[..t2_len]);

        let mut p_len = t_len.max(t2_len);
        l.p[t2_len..p_len].fill(0);
        let carry = add_buf(&mut l.p[..p_len], &t[..t_len]);

        let mut q_len = hi_mul_drop::<N>(lq, rq, k, &mut r.p);

        if carry && p_len < N {
            l.p[p_len] = 1;
            p_len += 1;
        } else if carry {
            l.p.copy_within(1.., 0);
            l.p[N - 1] = 1;
            r.p.copy_within(1..q_len, 0);
            q_len -= 1;
        }

        StaticNodeBBP {
            a: l.a,
            b: r.b,
            p: l.p,
            p_len,
            q: r.p,
            q_len,
            _s: PhantomData,
        }
    }

    fn finalize(&self, out: &mut [u64]) {
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], 0, out);
    }
}

pub trait EngelSeries {
    const R: u64;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
}

#[derive(Debug, Clone)]
pub struct DynNodeEngel<S: EngelSeries> {
    a: u64,
    b: u64,
    p: Vec<u64>,
    q: Vec<u64>,
    _s: PhantomData<S>,
}

impl<S: EngelSeries> Node for DynNodeEngel<S> {
    fn leaf(a: u64, b: u64) -> Self {
        let terms = (b - a) as usize;
        let words = split_bits(S::R * (b - a)).0 + 1;
        let (mut p, mut q) = (vec![0; terms + words], vec![0; terms]);
        let (p_len, q_len, e) = engel_leaf_terms::<S>(a, b, &mut p, &mut q);
        debug_assert_eq!(e, 0);
        p.truncate(p_len);
        q.truncate(q_len);
        DynNodeEngel {
            a,
            b,
            p,
            q,
            _s: PhantomData,
        }
    }

    fn merge(l: Self, r: Self) -> Self {
        let (sl, sb) = split_bits(S::R * (r.b - r.a));
        let pl = l.p.len() + r.q.len();

        let mut p = vec![0; (sl + pl + 1).max(r.p.len()) + 1];
        mul_dyn(&l.p, &r.q, &mut p[sl..sl + pl]);
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);
        let carry = add_buf(&mut p, &r.p);
        debug_assert!(!carry);
        trim_lz(&mut p);

        let mut q = fit(l.p, l.q.len() + r.q.len());
        mul_dyn(&l.q, &r.q, &mut q);
        trim_lz(&mut q);

        DynNodeEngel {
            a: l.a,
            b: r.b,
            p,
            q,
            _s: PhantomData,
        }
    }

    fn finalize(&self, out: &mut [u64]) {
        ratio_to_fraction_dyn(&self.p, &self.q, (S::R * (self.b - self.a)) as i64, out);
    }
}

#[derive(Debug, Clone)]
pub struct StaticNodeEngel<const N: usize, S: EngelSeries> {
    a: u64,
    b: u64,
    ep: usize,
    eq: usize,
    p: [u64; N],
    p_len: usize,
    q: [u64; N],
    q_len: usize,
    _s: PhantomData<S>,
}

impl<const N: usize, S: EngelSeries> Node for StaticNodeEngel<N, S> {
    fn leaf(a: u64, b: u64) -> Self {
        let (mut p, mut q) = ([0; N], [0; N]);
        let (p_len, q_len, ep) = engel_leaf_terms::<S>(a, b, &mut p, &mut q);
        StaticNodeEngel {
            a,
            b,
            ep,
            eq: 0,
            p,
            p_len,
            q,
            q_len,
            _s: PhantomData,
        }
    }

    fn merge(mut l: Self, r: Self) -> Self {
        const { assert!(N >= 2) };
        let (sl, sb) = split_bits(S::R * (r.b - r.a));
        let (lp, lq) = (&l.p[..l.p_len], &l.q[..l.q_len]);
        let (rp, rq) = (&r.p[..r.p_len], &r.q[..r.q_len]);

        let t_len = mul_len(lp, rq);
        let t_e = l.ep + r.eq + sl;
        let t_top = if t_len == 0 {
            0
        } else {
            t_e + t_len + (sb != 0) as usize
        };
        let r_top = if rp.is_empty() { 0 } else { r.ep + rp.len() };
        let ep = (t_top.max(r_top) + 1).saturating_sub(N);

        let mut p = [0; N];
        let mut p_len = 0;
        if t_len != 0 {
            let off = t_e.saturating_sub(ep);
            p_len = off + hi_mul_drop::<N>(lp, rq, ep.saturating_sub(t_e), &mut p[off..]);
            let c = shl_buf(&mut p[off..p_len], sb);
            push(&mut p, &mut p_len, c);
        }
        if !rp.is_empty() {
            let off = r.ep.saturating_sub(ep);
            let rp = &rp[ep.saturating_sub(r.ep).min(rp.len())..];
            p_len = p_len.max(off + rp.len());
            if add_buf(&mut p[off..p_len], rp) {
                p[p_len] = 1;
                p_len += 1;
            }
        }

        let kq = mul_len(lq, rq).saturating_sub(N);
        let q_len = hi_mul_drop::<N>(lq, rq, kq, &mut l.p);

        StaticNodeEngel {
            a: l.a,
            b: r.b,
            ep,
            eq: l.eq + r.eq + kq,
            p,
            p_len,
            q: l.p,
            q_len,
            _s: PhantomData,
        }
    }

    fn finalize(&self, out: &mut [u64]) {
        let shift = (S::R * (self.b - self.a)) as i64 + 64 * (self.eq as i64 - self.ep as i64);
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], shift, out);
    }
}

pub trait HyperSeries {
    const R: u64;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
    fn r(n: u64) -> u64;
}

#[derive(Debug, Clone)]
pub struct DynNodeHyper<S: HyperSeries> {
    a: u64,
    b: u64,
    p: Vec<u64>,
    q: Vec<u64>,
    r: Vec<u64>,
    _s: PhantomData<S>,
}

impl<S: HyperSeries> Node for DynNodeHyper<S> {
    fn leaf(a: u64, b: u64) -> Self {
        let terms = (b - a) as usize;
        let words = split_bits(S::R * (b - a)).0 + 1;
        let (mut p, mut q, mut r) = (vec![0; terms + words], vec![0; terms], vec![0; terms]);
        let (p_len, q_len, r_len, e) = hyper_leaf_terms::<S>(a, b, &mut p, &mut q, &mut r);
        debug_assert_eq!(e, 0);
        p.truncate(p_len);
        q.truncate(q_len);
        r.truncate(r_len);
        DynNodeHyper {
            a,
            b,
            p,
            q,
            r,
            _s: PhantomData,
        }
    }

    fn merge(left: Self, right: Self) -> Self {
        let (sl, sb) = split_bits(S::R * (right.b - right.a));
        let tl = right.p.len() + left.r.len();
        let pl = left.p.len() + right.q.len();
        let ql = left.q.len() + right.q.len();
        let rl = left.r.len() + right.r.len();

        let mut p = vec![0; (sl + pl + 1).max(tl) + 1];
        mul_dyn(&left.p, &right.q, &mut p[sl..sl + pl]);
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);

        let mut t = fit(left.p, tl);
        mul_dyn(&right.p, &left.r, &mut t);
        let carry = add_buf(&mut p, &t);
        debug_assert!(!carry);
        trim_lz(&mut p);

        let mut q = fit(t, ql);
        mul_dyn(&left.q, &right.q, &mut q);
        trim_lz(&mut q);

        let mut r = fit(right.p, rl);
        mul_dyn(&left.r, &right.r, &mut r);
        trim_lz(&mut r);

        DynNodeHyper {
            a: left.a,
            b: right.b,
            p,
            q,
            r,
            _s: PhantomData,
        }
    }

    // out = floor(S * B^out.len()) for S = P / (Q * 2^(R(b - a))).
    fn finalize(&self, out: &mut [u64]) {
        ratio_to_fraction_dyn(&self.p, &self.q, (S::R * (self.b - self.a)) as i64, out);
    }
}

#[derive(Debug, Clone)]
pub struct StaticNodeHyper<const N: usize, S: HyperSeries> {
    a: u64,
    b: u64,
    ep: usize,
    eq: usize,
    er: usize,
    p: [u64; N],
    p_len: usize,
    q: [u64; N],
    q_len: usize,
    r: [u64; N],
    r_len: usize,
    _s: PhantomData<S>,
}

impl<const N: usize, S: HyperSeries> Node for StaticNodeHyper<N, S> {
    fn leaf(a: u64, b: u64) -> Self {
        let (mut p, mut q, mut r) = ([0; N], [0; N], [0; N]);
        let (p_len, q_len, r_len, ep) = hyper_leaf_terms::<S>(a, b, &mut p, &mut q, &mut r);
        StaticNodeHyper {
            a,
            b,
            ep,
            eq: 0,
            er: 0,
            p,
            p_len,
            q,
            q_len,
            r,
            r_len,
            _s: PhantomData,
        }
    }

    fn merge(mut left: Self, mut right: Self) -> Self {
        const { assert!(N >= 2) };
        let (sl, sb) = split_bits(S::R * (right.b - right.a));
        let (lp, lq, lr) = (
            &left.p[..left.p_len],
            &left.q[..left.q_len],
            &left.r[..left.r_len],
        );
        let (rp, rq, rr) = (
            &right.p[..right.p_len],
            &right.q[..right.q_len],
            &right.r[..right.r_len],
        );

        let (t_len, u_len) = (mul_len(lp, rq), mul_len(rp, lr));
        let (t_e, u_e) = (left.ep + right.eq + sl, right.ep + left.er);
        let t_top = if t_len == 0 {
            0
        } else {
            t_e + t_len + (sb != 0) as usize
        };
        let u_top = if u_len == 0 { 0 } else { u_e + u_len };
        let ep = (t_top.max(u_top) + 1).saturating_sub(N);

        let mut p = [0; N];
        let mut p_len = 0;
        if t_len != 0 {
            let off = t_e.saturating_sub(ep);
            p_len = off + hi_mul_drop::<N>(lp, rq, ep.saturating_sub(t_e), &mut p[off..]);
            let c = shl_buf(&mut p[off..p_len], sb);
            push(&mut p, &mut p_len, c);
        }

        if u_len != 0 {
            let off = u_e.saturating_sub(ep);
            let u_len = hi_mul_drop::<N>(rp, lr, ep.saturating_sub(u_e), &mut left.p);
            if u_len != 0 {
                p_len = p_len.max(off + u_len);
                if add_buf(&mut p[off..p_len], &left.p[..u_len]) {
                    p[p_len] = 1;
                    p_len += 1;
                }
            }
        }

        let kq = mul_len(lq, rq).saturating_sub(N);
        let q_len = hi_mul_drop::<N>(lq, rq, kq, &mut right.p);
        let kr = mul_len(lr, rr).saturating_sub(N);
        let r_len = hi_mul_drop::<N>(lr, rr, kr, &mut left.p);

        StaticNodeHyper {
            a: left.a,
            b: right.b,
            ep,
            eq: left.eq + right.eq + kq,
            er: left.er + right.er + kr,
            p,
            p_len,
            q: right.p,
            q_len,
            r: left.p,
            r_len,
            _s: PhantomData,
        }
    }

    fn finalize(&self, out: &mut [u64]) {
        let shift = (S::R * (self.b - self.a)) as i64 + 64 * (self.eq as i64 - self.ep as i64);
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], shift, out);
    }
}
