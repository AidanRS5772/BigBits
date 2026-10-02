use std::marker::PhantomData;

use crate::utils::{
    div::{rcp_dyn, rcp_static},
    mul::{hi_mul_dyn, hi_mul_static, mul_add_prim, mul_dyn, mul_prim, sqr_dyn},
    utils::{add_buf, add_mul, add_prim, buf_len, shl_buf, shr_buf, split_sh, trim_lz},
    ScratchGuard, PARTIAL_MUL_CUTOFF,
};

fn push(buf: &mut [u64], len: &mut usize, c: u64) {
    if c != 0 {
        buf[*len] = c;
        *len += 1;
    }
}

fn prepend_shl(x: &mut [u64], x_len: &mut usize, sh: u64) -> u64 {
    if *x_len == 0 || sh == 0 {
        return 0;
    }
    let (sl, sb) = split_sh(sh);
    if sl != 0 {
        x[..*x_len + sl].copy_within(..*x_len, sl);
        x[..sl].fill(0);
        *x_len += sl;
    }
    shl_buf(&mut x[sl..*x_len], sb)
}

fn bbp_leaf_terms<S: BBPSeries>(
    a: u64,
    b: u64,
    shift: u64,
    p: &mut [u64],
    q: &mut [u64],
) -> (usize, usize) {
    q[0] = 1;
    let (mut p_len, mut q_len) = (0, 1);
    let mut n = a;
    while n < b {
        let (mut u, mut v, mut j) = (S::q(n), S::p(n), 1);
        n += 1;
        while shift < 64 && n < b {
            let w = S::q(n) as u128;
            let (uw, vw) = (u as u128 * w, v as u128 * w);
            // v * w << shift must stay one limb; checked on u64 halves so the
            // shift by a runtime amount stays a single instruction.
            if (uw >> 64 | vw >> 64) != 0 || (vw as u64).leading_zeros() < shift as u32 {
                break;
            }
            let vw = (((vw as u64) << shift) as u128) + u as u128 * S::p(n) as u128;
            if vw >> 64 != 0 {
                break;
            }
            (u, v, j) = (uw as u64, vw as u64, j + 1);
            n += 1;
        }
        // P = (P * u) << shift * j + Q * v, shifting u instead of P when the
        // sub-limb shift fits.
        let (sh, sb) = (shift * j, (shift * j % 64) as u32);
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

fn shl_or_drop(buf: &mut [u64], len: &mut usize, e: &mut usize, bits: u64) {
    if *len == 0 || bits == 0 {
        return;
    }
    let (sl, sb) = split_sh(bits);
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

fn engel_leaf_terms<S: EngelSeries>(
    a: u64,
    b: u64,
    shift: u64,
    p: &mut [u64],
    q: &mut [u64],
) -> (usize, usize, usize) {
    q[0] = 1;
    let (mut p_len, mut q_len, mut e) = (0, 1, 0);
    let mut n = a;
    while n < b {
        let (mut u, mut v, mut j) = (S::q(n), S::p(n), 1);
        n += 1;
        while shift < 64 && n < b {
            let w = S::q(n) as u128;
            let (uw, vw) = (u as u128 * w, v as u128 * w);
            // v * w << shift must stay one limb; checked on u64 halves so the
            // shift by a runtime amount stays a single instruction.
            if (uw >> 64 | vw >> 64) != 0 || (vw as u64).leading_zeros() < shift as u32 {
                break;
            }
            let vw = (((vw as u64) << shift) as u128) + S::p(n) as u128;
            if vw >> 64 != 0 {
                break;
            }
            (u, v, j) = (uw as u64, vw as u64, j + 1);
            n += 1;
        }
        // P = (P * u) << shift * j + v, shifting u instead of P when the sub-limb
        // shift fits; v falls below P's precision once e > 0.
        let (sh, sb) = (shift * j, (shift * j % 64) as u32);
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

fn hyper_leaf_terms<S: HyperSeries>(
    a: u64,
    b: u64,
    shift: u64,
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
        while shift < 64 && n < b {
            let (x, y) = (S::q(n) as u128, S::r(n) as u128);
            let (ux, vy, wx) = (u as u128 * x, v as u128 * y, w as u128 * x);
            // w * x << shift must stay one limb; checked on u64 halves so the
            // shift by a runtime amount stays a single instruction.
            if (ux >> 64 | vy >> 64 | wx >> 64) != 0 || (wx as u64).leading_zeros() < shift as u32 {
                break;
            }
            let wx = (((wx as u64) << shift) as u128) + v as u128 * S::p(n) as u128;
            if wx >> 64 != 0 {
                break;
            }
            (u, v, w, j) = (ux as u64, vy as u64, wx as u64, j + 1);
            n += 1;
        }
        // P = (P * u) << shift * j + R * w, shifting u instead of P when the
        // sub-limb shift fits.
        let (sh, sb) = (shift * j, (shift * j % 64) as u32);
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

fn add_mul_at(p: &mut [u64], p_len: &mut usize, e: &mut usize, t: &[u64], et: usize, w: u64) {
    if t.is_empty() {
        return;
    }
    let top = (et + t.len() + 1).saturating_sub(*e);
    if top > p.len() {
        let d = top - p.len();
        let m = d.min(*p_len);
        p.copy_within(m..*p_len, 0);
        p[*p_len - m..*p_len].fill(0);
        *p_len -= m;
        *e += d;
    }
    let (off, t) = match et.checked_sub(*e) {
        Some(off) => (off, t),
        None => (0, &t[(*e - et).min(t.len())..]),
    };
    *p_len = (*p_len).max(off + t.len());
    let c = add_mul(&mut p[off..*p_len], t, w);
    push_or_drop(p, p_len, e, c);
}

fn hyper_x_leaf_terms<'t, S: HyperSeries>(
    a: u64,
    b: u64,
    shift: u64,
    x: &[u64],
    p: &mut [u64],
    mut t: &'t mut [u64],
    mut tmp: &'t mut [u64],
    hi_mul: impl Fn(&[u64], &[u64], usize, &mut [u64]) -> usize,
) -> (usize, usize) {
    t[0] = 1;
    let (mut p_len, mut e, mut t_len, mut et) = (0, 0, 1, 0);
    for n in a..b {
        // P = (P * q(n)) << shift + T * p(n), shifting q(n) instead of P when
        // the sub-limb shift fits.
        let (v, sb) = (S::q(n), (shift % 64) as u32);
        let f = if v.leading_zeros() >= sb { sb } else { 0 };
        let c = mul_prim(&mut p[..p_len], v << f);
        push_or_drop(p, &mut p_len, &mut e, c);
        shl_or_drop(p, &mut p_len, &mut e, shift - f as u64);
        add_mul_at(p, &mut p_len, &mut e, &t[..t_len], et, S::p(n));
        if n + 1 == b {
            break;
        }
        // T = T * r(n) * x.
        let c = mul_prim(&mut t[..t_len], S::r(n));
        push_or_drop(t, &mut t_len, &mut et, c);
        let drop = mul_len(&t[..t_len], x).saturating_sub(tmp.len());
        t_len = hi_mul(&t[..t_len], x, drop, tmp);
        std::mem::swap(&mut t, &mut tmp);
        et += drop;
    }
    (buf_len(&p[..p_len]), e)
}

fn mul_exact(a: &[u64], b: &[u64], drop: usize, out: &mut [u64]) -> usize {
    debug_assert_eq!(drop, 0);
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let n = a.len() + b.len();
    mul_dyn(a, b, &mut out[..n]);
    buf_len(&out[..n])
}

fn prod_terms(a: u64, b: u64, f: impl Fn(u64) -> u64, buf: &mut [u64]) -> usize {
    buf[0] = 1;
    let (mut len, mut n) = (1, a);
    while n < b {
        let mut u = f(n);
        n += 1;
        while n < b {
            let w = u as u128 * f(n) as u128;
            if w >> 64 != 0 {
                break;
            }
            u = w as u64;
            n += 1;
        }
        let c = mul_prim(&mut buf[..len], u);
        push(buf, &mut len, c);
    }
    len
}

fn pow_levels(terms: u64) -> usize {
    (64 - terms.saturating_sub(1).leading_zeros()).max(1) as usize
}

fn pow2_split(a: u64, b: u64) -> u64 {
    a + (1 << (63 - (b - a - 1).leading_zeros()))
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
        return;
    }

    // x ≈ B^(q.len() + x.len() - 1) / q, so p * x / 2^bits = S * B^len, with
    // w >= 2 whole limbs in bits.
    rcp(q, x);
    let bits = 64 * (q.len() + x.len() - 1 - len) as i64 + shift;
    debug_assert!(bits >= 128, "finalize needs w >= 2 whole limbs");
    let (w, v) = split_sh(bits as u64);
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

pub fn ratio_to_fraction_dyn(p: &[u64], q: &[u64], shift: i64, out: &mut [u64]) {
    let len = out.len();
    let mut scratch = ScratchGuard::acquire();
    let [x, y] = scratch.get_splits([len + 2, len + 3]);
    ratio_to_fraction(p, q, shift, out, x, y, rcp_dyn, hi_mul_dyn);
}

pub fn ratio_to_fraction_static<const N: usize>(p: &[u64], q: &[u64], shift: i64, out: &mut [u64]) {
    let len = out.len();
    assert!(len + 3 <= N, "finalize output needs out.len() + 3 <= N");
    let (mut x, mut y) = ([0; N], [0; N]);
    let (x, y) = (&mut x[..len + 2], &mut y[..len + 3]);
    ratio_to_fraction(p, q, shift, out, x, y, rcp_static::<N>, hi_mul_static::<N>);
}

pub trait Node: Sized {
    // Runtime state shared by every leaf and merge of a split.
    type Ctx;
    fn merge(ctx: &Self::Ctx, l: Self, r: Self) -> Self;
    fn leaf(ctx: &Self::Ctx, a: u64, b: u64, last: bool) -> Self;
    fn finalize(&self, ctx: &Self::Ctx, out: &mut [u64]);
    // Where [a, b) splits: its midpoint, unless a node needs another shape.
    fn split(_: &Self::Ctx, a: u64, b: u64) -> u64 {
        (a + b) / 2
    }
}

pub fn bin_split_tree<N, const CUTOFF: u64>(ctx: &N::Ctx, a: u64, b: u64, last: bool) -> N
where
    N: Node,
{
    const { assert!(CUTOFF >= 2) };
    debug_assert!(a < b);
    if b - a < CUTOFF {
        return N::leaf(ctx, a, b, last);
    }
    let m = N::split(ctx, a, b);
    let l = bin_split_tree::<N, CUTOFF>(ctx, a, m, false);
    let r = bin_split_tree::<N, CUTOFF>(ctx, m, b, last);
    N::merge(ctx, l, r)
}

pub fn bin_split<N, const CUTOFF: u64>(ctx: &N::Ctx, a: u64, b: u64) -> N
where
    N: Node,
{
    bin_split_tree::<N, CUTOFF>(ctx, a, b, true)
}

#[derive(Debug, Clone, Copy)]
pub struct Shift(pub u64);

pub trait BBPSeries {
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
}

#[derive(Debug, Clone)]
pub struct DynNodeBBP<S: BBPSeries> {
    pub a: u64,
    pub b: u64,
    pub p: Vec<u64>,
    pub q: Vec<u64>,
    _s: PhantomData<S>,
}

impl<S: BBPSeries> Node for DynNodeBBP<S> {
    type Ctx = Shift;

    fn leaf(ctx: &Shift, a: u64, b: u64, _: bool) -> Self {
        let terms = (b - a) as usize;
        let words = split_sh(ctx.0 * (b - a)).0 + 1;
        let (mut p, mut q) = (vec![0; terms + words], vec![0; terms]);
        let (p_len, q_len) = bbp_leaf_terms::<S>(a, b, ctx.0, &mut p, &mut q);
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

    fn merge(ctx: &Shift, l: Self, r: Self) -> Self {
        let (sl, sb) = split_sh(ctx.0 * (r.b - r.a)); // r(b - m)
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

    fn finalize(&self, ctx: &Shift, out: &mut [u64]) {
        ratio_to_fraction_dyn(&self.p, &self.q, (ctx.0 * (self.b - self.a)) as i64, out);
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
    type Ctx = Shift;

    fn leaf(ctx: &Shift, a: u64, b: u64, _: bool) -> Self {
        let (mut p, mut q) = ([0; N], [0; N]);
        let (p_len, mut q_len) = bbp_leaf_terms::<S>(a, b, ctx.0, &mut p, &mut q);
        let c = prepend_shl(&mut q, &mut q_len, ctx.0 * (b - a));
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

    fn merge(ctx: &Shift, mut l: Self, mut r: Self) -> Self {
        const { assert!(N >= 2) };
        let (sl, sb) = split_sh(ctx.0 * (l.b - l.a)); // r(m - a)
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

    fn finalize(&self, _: &Shift, out: &mut [u64]) {
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], 0, out);
    }
}

pub trait EngelSeries {
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
    type Ctx = Shift;

    fn leaf(ctx: &Shift, a: u64, b: u64, _: bool) -> Self {
        let terms = (b - a) as usize;
        let words = split_sh(ctx.0 * (b - a)).0 + 1;
        let (mut p, mut q) = (vec![0; terms + words], vec![0; terms]);
        let (p_len, q_len, e) = engel_leaf_terms::<S>(a, b, ctx.0, &mut p, &mut q);
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

    fn merge(ctx: &Shift, l: Self, r: Self) -> Self {
        let (sl, sb) = split_sh(ctx.0 * (r.b - r.a));
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

    fn finalize(&self, ctx: &Shift, out: &mut [u64]) {
        ratio_to_fraction_dyn(&self.p, &self.q, (ctx.0 * (self.b - self.a)) as i64, out);
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
    type Ctx = Shift;

    fn leaf(ctx: &Shift, a: u64, b: u64, _: bool) -> Self {
        let (mut p, mut q) = ([0; N], [0; N]);
        let (p_len, q_len, ep) = engel_leaf_terms::<S>(a, b, ctx.0, &mut p, &mut q);
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

    fn merge(ctx: &Shift, mut l: Self, r: Self) -> Self {
        const { assert!(N >= 2) };
        let (sl, sb) = split_sh(ctx.0 * (r.b - r.a));
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

    fn finalize(&self, ctx: &Shift, out: &mut [u64]) {
        let shift = (ctx.0 * (self.b - self.a)) as i64 + 64 * (self.eq as i64 - self.ep as i64);
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], shift, out);
    }
}

pub trait HyperSeries {
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
    fn r(n: u64) -> u64;
}

#[derive(Debug, Clone)]
pub struct DynHyperCtx {
    shift: u64,
    // x^(2^j), empty when x = 1.
    pow: Vec<Vec<u64>>,
}

impl DynHyperCtx {
    // x must be trimmed and nonzero, and splits may cover up to terms terms.
    pub fn new(x: &[u64], shift: u64, terms: u64) -> Self {
        debug_assert!(
            x.last().is_some_and(|&l| l != 0),
            "x must be trimmed and nonzero"
        );
        let mut pow: Vec<Vec<u64>> = Vec::new();
        if x != [1] {
            pow.push(x.to_vec());
            for _ in 1..pow_levels(terms) {
                let prev = pow.last().unwrap();
                let mut sq = vec![0; 2 * prev.len()];
                sqr_dyn(prev, &mut sq);
                trim_lz(&mut sq);
                pow.push(sq);
            }
        }
        DynHyperCtx { shift, pow }
    }

    fn x(&self) -> Option<&[u64]> {
        self.pow.first().map(Vec::as_slice)
    }

    // x^len for a power-of-two len, or None when x = 1.
    fn x_pow(&self, len: u64) -> Option<&[u64]> {
        if self.pow.is_empty() {
            return None;
        }
        debug_assert!(len.is_power_of_two());
        Some(&self.pow[len.trailing_zeros() as usize])
    }
}

#[derive(Debug, Clone)]
pub struct StaticHyperCtx<const N: usize, const L: usize = 32> {
    shift: u64,
    // Filled entries, 0 when x = 1.
    levels: usize,
    pow: [[u64; N]; L],
    pow_len: [usize; L],
    pow_e: [usize; L],
}

impl<const N: usize, const L: usize> StaticHyperCtx<N, L> {
    // x must be trimmed, nonzero and within N limbs, and splits may cover up to
    // terms terms.
    pub fn new(x: &[u64], shift: u64, terms: u64) -> Self {
        debug_assert!(
            x.last().is_some_and(|&l| l != 0),
            "x must be trimmed and nonzero"
        );
        let mut ctx = StaticHyperCtx {
            shift,
            levels: 0,
            pow: [[0; N]; L],
            pow_len: [0; L],
            pow_e: [0; L],
        };
        if x != [1] {
            assert!(x.len() <= N, "x must fit N limbs");
            ctx.levels = pow_levels(terms);
            assert!(
                ctx.levels <= L,
                "splits of {terms} terms need more than L powers"
            );
            ctx.pow[0][..x.len()].copy_from_slice(x);
            ctx.pow_len[0] = x.len();
            for j in 1..ctx.levels {
                let (lo, hi) = ctx.pow.split_at_mut(j);
                let prev = &lo[j - 1][..ctx.pow_len[j - 1]];
                let drop = mul_len(prev, prev).saturating_sub(N);
                ctx.pow_len[j] = hi_mul_drop::<N>(prev, prev, drop, &mut hi[0]);
                ctx.pow_e[j] = 2 * ctx.pow_e[j - 1] + drop;
            }
        }
        ctx
    }

    fn x(&self) -> Option<&[u64]> {
        (self.levels != 0).then(|| &self.pow[0][..self.pow_len[0]])
    }

    // x^len for a power-of-two len as its top limbs and the limbs dropped below
    // them, or None when x = 1.
    fn x_pow(&self, len: u64) -> Option<(&[u64], usize)> {
        if self.levels == 0 {
            return None;
        }
        debug_assert!(len.is_power_of_two());
        let j = len.trailing_zeros() as usize;
        assert!(j < self.levels, "split is longer than the context's terms");
        Some((&self.pow[j][..self.pow_len[j]], self.pow_e[j]))
    }
}

#[derive(Debug, Clone)]
pub struct DynNodeHyper<S: HyperSeries> {
    a: u64,
    b: u64,
    p: Vec<u64>,
    q: Vec<u64>,
    r: Option<Vec<u64>>,
    _s: PhantomData<S>,
}

impl<S: HyperSeries> Node for DynNodeHyper<S> {
    type Ctx = DynHyperCtx;

    fn leaf(ctx: &DynHyperCtx, a: u64, b: u64, last: bool) -> Self {
        let terms = (b - a) as usize;
        let words = split_sh(ctx.shift * (b - a)).0 + 1;
        let mut q = vec![0; terms];
        let (p, q_len, r) = match ctx.x() {
            None => {
                let mut p = vec![0; terms + words];
                // P needs R's running product, so a leaf that ends the sum still
                // forms R, but in scratch rather than a buffer it keeps.
                let (p_len, q_len, r, e) = if last {
                    let mut scratch = ScratchGuard::acquire();
                    let r = scratch.get(terms);
                    let (p_len, q_len, _, e) =
                        hyper_leaf_terms::<S>(a, b, ctx.shift, &mut p, &mut q, r);
                    (p_len, q_len, None, e)
                } else {
                    let mut r = vec![0; terms];
                    let (p_len, q_len, r_len, e) =
                        hyper_leaf_terms::<S>(a, b, ctx.shift, &mut p, &mut q, &mut r);
                    r.truncate(r_len);
                    (p_len, q_len, Some(r), e)
                };
                debug_assert_eq!(e, 0);
                p.truncate(p_len);
                (p, q_len, r)
            }
            // T = R * x^j only lives in scratch, and since it carries R into P, a
            // leaf that ends the sum never forms R.
            Some(x) => {
                let cap = terms * (x.len() + 1) + 1;
                let mut p = vec![0; cap + words];
                let mut scratch = ScratchGuard::acquire();
                let [t, tmp] = scratch.get_splits([cap + x.len(); 2]);
                let (p_len, e) =
                    hyper_x_leaf_terms::<S>(a, b, ctx.shift, x, &mut p, t, tmp, mul_exact);
                debug_assert_eq!(e, 0);
                p.truncate(p_len);
                let q_len = prod_terms(a, b, S::q, &mut q);
                let r = (!last).then(|| {
                    let mut r = vec![0; terms];
                    let r_len = prod_terms(a, b, S::r, &mut r);
                    r.truncate(r_len);
                    r
                });
                (p, q_len, r)
            }
        };
        q.truncate(q_len);
        DynNodeHyper {
            a,
            b,
            p,
            q,
            r,
            _s: PhantomData,
        }
    }

    fn merge(ctx: &DynHyperCtx, left: Self, right: Self) -> Self {
        let lr = left
            .r
            .expect("a node that ends the sum can't merge as a left child");
        let (sl, sb) = split_sh(ctx.shift * (right.b - right.a));
        // The left child's whole ratio product, its r's times x^len(left).
        let lt_buf;
        let lt = match ctx.x_pow(left.b - left.a) {
            None => &lr,
            Some(xp) => {
                let mut t = vec![0; lr.len() + xp.len()];
                mul_dyn(&lr, xp, &mut t);
                trim_lz(&mut t);
                lt_buf = t;
                &lt_buf
            }
        };
        let tl = right.p.len() + lt.len();
        let pl = left.p.len() + right.q.len();
        let ql = left.q.len() + right.q.len();

        let mut p = vec![0; (sl + pl + 1).max(tl) + 1];
        mul_dyn(&left.p, &right.q, &mut p[sl..sl + pl]);
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);

        let mut t = fit(left.p, tl);
        mul_dyn(&right.p, lt, &mut t);
        let carry = add_buf(&mut p, &t);
        debug_assert!(!carry);
        trim_lz(&mut p);

        let mut q = fit(t, ql);
        mul_dyn(&left.q, &right.q, &mut q);
        trim_lz(&mut q);

        // A right child without R ends the sum, and so does the merged node.
        let r = right.r.map(|rr| {
            let mut r = fit(right.p, lr.len() + rr.len());
            mul_dyn(&lr, &rr, &mut r);
            trim_lz(&mut r);
            r
        });

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
    fn finalize(&self, ctx: &DynHyperCtx, out: &mut [u64]) {
        ratio_to_fraction_dyn(
            &self.p,
            &self.q,
            (ctx.shift * (self.b - self.a)) as i64,
            out,
        );
    }

    fn split(ctx: &DynHyperCtx, a: u64, b: u64) -> u64 {
        match ctx.x() {
            Some(_) => pow2_split(a, b),
            None => (a + b) / 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StaticNodeHyper<const N: usize, S: HyperSeries, const L: usize = 32> {
    a: u64,
    b: u64,
    p: [u64; N],
    p_len: usize,
    ep: usize,
    q: [u64; N],
    q_len: usize,
    eq: usize,
    r: Option<[u64; N]>,
    r_len: Option<usize>,
    er: Option<usize>,
    _s: PhantomData<S>,
}

impl<const N: usize, S: HyperSeries, const L: usize> StaticNodeHyper<N, S, L> {
    // The leaf for multi-limb x, returning (p_len, ep, q_len, r_len). T = R * x^j
    // carries R into P, so a leaf that ends the sum never forms R. Kept out of
    // leaf so its scratch arrays don't enlarge the x = 1 leaf's frame.
    #[inline(never)]
    fn x_leaf(
        a: u64,
        b: u64,
        shift: u64,
        x: &[u64],
        last: bool,
        p: &mut [u64; N],
        q: &mut [u64; N],
        r: &mut [u64; N],
    ) -> (usize, usize, usize, usize) {
        let (mut t, mut tmp) = ([0; N], [0; N]);
        let (p_len, ep) =
            hyper_x_leaf_terms::<S>(a, b, shift, x, p, &mut t, &mut tmp, hi_mul_drop::<N>);
        let q_len = prod_terms(a, b, S::q, q);
        let r_len = if last { 0 } else { prod_terms(a, b, S::r, r) };
        (p_len, ep, q_len, r_len)
    }
}

impl<const N: usize, S: HyperSeries, const L: usize> Node for StaticNodeHyper<N, S, L> {
    type Ctx = StaticHyperCtx<N, L>;

    fn leaf(ctx: &StaticHyperCtx<N, L>, a: u64, b: u64, last: bool) -> Self {
        let (mut p, mut q, mut r) = ([0; N], [0; N], [0; N]);
        let (p_len, ep, q_len, r_len) = match ctx.x() {
            // P needs R's running product, so a leaf that ends the sum still
            // forms R but doesn't keep it.
            None => {
                let (p_len, q_len, r_len, ep) =
                    hyper_leaf_terms::<S>(a, b, ctx.shift, &mut p, &mut q, &mut r);
                (p_len, ep, q_len, r_len)
            }
            Some(x) => Self::x_leaf(a, b, ctx.shift, x, last, &mut p, &mut q, &mut r),
        };
        let keep = !last;
        StaticNodeHyper {
            a,
            b,
            p,
            p_len,
            ep,
            q,
            q_len,
            eq: 0,
            r: keep.then_some(r),
            r_len: keep.then_some(r_len),
            er: keep.then_some(0),
            _s: PhantomData,
        }
    }

    fn merge(ctx: &StaticHyperCtx<N, L>, mut left: Self, mut right: Self) -> Self {
        const { assert!(N >= 2) };
        let (sl, sb) = split_sh(ctx.shift * (right.b - right.a));
        let (lp, lq) = (&left.p[..left.p_len], &left.q[..left.q_len]);
        let (rp, rq) = (&right.p[..right.p_len], &right.q[..right.q_len]);
        let (lr, ler) = match (&left.r, left.r_len, left.er) {
            (Some(r), Some(len), Some(e)) => (&r[..len], e),
            _ => panic!("a node that ends the sum can't merge as a left child"),
        };
        // The left child's whole ratio product, its r's times x^len(left).
        let mut lt_buf;
        let (lt, lte) = match ctx.x_pow(left.b - left.a) {
            None => (lr, ler),
            Some((xp, xe)) => {
                let kt = mul_len(lr, xp).saturating_sub(N);
                lt_buf = [0; N];
                let lt_len = hi_mul_drop::<N>(lr, xp, kt, &mut lt_buf);
                (&lt_buf[..lt_len], ler + xe + kt)
            }
        };

        let (t_len, u_len) = (mul_len(lp, rq), mul_len(rp, lt));
        let (t_e, u_e) = (left.ep + right.eq + sl, right.ep + lte);
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
            let u_len = hi_mul_drop::<N>(rp, lt, ep.saturating_sub(u_e), &mut left.p);
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
        // A right child without R ends the sum, and so does the merged node.
        let (r_len, er) = match (&right.r, right.r_len, right.er) {
            (Some(r), Some(len), Some(e)) => {
                let rr = &r[..len];
                let kr = mul_len(lr, rr).saturating_sub(N);
                let r_len = hi_mul_drop::<N>(lr, rr, kr, &mut left.p);
                (Some(r_len), Some(ler + e + kr))
            }
            _ => (None, None),
        };

        StaticNodeHyper {
            a: left.a,
            b: right.b,
            p,
            p_len,
            ep,
            q: right.p,
            q_len,
            eq: left.eq + right.eq + kq,
            r: r_len.map(|_| left.p),
            r_len,
            er,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &StaticHyperCtx<N, L>, out: &mut [u64]) {
        let shift = (ctx.shift * (self.b - self.a)) as i64 + 64 * (self.eq as i64 - self.ep as i64);
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], shift, out);
    }

    fn split(ctx: &StaticHyperCtx<N, L>, a: u64, b: u64) -> u64 {
        match ctx.x() {
            Some(_) => pow2_split(a, b),
            None => (a + b) / 2,
        }
    }
}
