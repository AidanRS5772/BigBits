use std::marker::PhantomData;

use crate::utils::{
    div::{rcp_dyn, rcp_static},
    mul::{hi_mul_dyn, hi_mul_static, mul_add_prim, mul_dyn, mul_prim},
    utils::{
        add_buf, add_mul, add_prim, buf_len, inc_buf, shl_buf, shr_buf, split_sh, sub_mul, trim_lz,
    },
    ScratchGuard, DYN_BIN_SPLIT_MERGE_PAR_CUTOFF, DYN_BIN_SPLIT_PAR_CUTOFF, PARTIAL_MUL_CUTOFF,
};

fn merge_par(q_len: usize) -> bool {
    q_len >= DYN_BIN_SPLIT_MERGE_PAR_CUTOFF && rayon::current_thread_index().is_some()
}

fn join2<A: Send, B: Send>(
    par: bool,
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    if par {
        rayon::join(a, b)
    } else {
        (a(), b())
    }
}

fn push(buf: &mut [u64], len: &mut usize, c: u64) {
    if c != 0 {
        buf[*len] = c;
        *len += 1;
    }
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

fn sub_mul_flip(
    p: &mut [u64],
    len: &mut usize,
    off: usize,
    s: &[u64],
    d: u64,
    neg: &mut bool,
) -> u64 {
    let x = &mut p[..*len];
    let borrow = sub_mul(&mut x[off..], s, d);
    let c = if borrow == 0 {
        0
    } else {
        for l in x.iter_mut() {
            *l = !*l;
        }
        *neg = !*neg;
        borrow - 1 + inc_buf(x) as u64
    };
    if c == 0 {
        *len = buf_len(&p[..*len]);
    }
    c
}

#[inline(always)]
fn add_signed(
    p: &mut [u64],
    len: &mut usize,
    off: usize,
    t: &[u64],
    neg: &mut bool,
    t_neg: bool,
    signed: bool,
) -> u64 {
    if signed && *neg != t_neg {
        sub_mul_flip(p, len, off, t, 1, neg)
    } else {
        add_buf(&mut p[off..*len], t) as u64
    }
}

#[inline(always)]
fn add_or_sub(acc: u128, t: u128, sub: bool) -> (u128, bool) {
    if sub {
        (acc.abs_diff(t), t > acc)
    } else {
        (acc + t, false)
    }
}

fn mul_shl(buf: &mut [u64], len: &mut usize, e: &mut usize, u: u64, sh: u64, add: u64) {
    let sb = (sh % 64) as u32;
    let f = if u.leading_zeros() >= sb { sb } else { 0 };
    let bits = sh - f as u64;
    let shift = *len != 0 && bits != 0;
    let c = mul_add_prim(&mut buf[..*len], u << f, add);
    push_or_drop(buf, len, e, c);
    if !shift {
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

fn bbp_leaf_terms<S: BBPSeries>(
    a: u64,
    b: u64,
    shift: u64,
    p: &mut [u64],
    q: &mut [u64],
) -> (usize, usize, usize, bool) {
    q[0] = 1;
    let (mut p_len, mut q_len, mut e, mut pn) = (0, 1, 0, false);
    let mut n = a;
    while n < b {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        let mut vn = S::SIGNED && S::neg(n);
        let (mut u, mut v, mut j) = (S::q(n), S::p(n), 1);
        n += 1;
        while shift < 64 && n < b {
            debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
            let w = S::q(n) as u128;
            let (uw, vw) = (u as u128 * w, v as u128 * w);
            if (uw >> 64 | vw >> 64) != 0 || (vw as u64).leading_zeros() < shift as u32 {
                break;
            }
            let (vw, flip) = add_or_sub(
                ((vw as u64) << shift) as u128,
                u as u128 * S::p(n) as u128,
                S::SIGNED && S::neg(n) != vn,
            );
            if vw >> 64 != 0 {
                break;
            }
            (u, v, j) = (uw as u64, vw as u64, j + 1);
            vn ^= flip;
            n += 1;
        }
        mul_shl(p, &mut p_len, &mut e, u, shift * j, 0);
        p_len = p_len.max(q_len);
        let c = if S::SIGNED && vn != pn {
            sub_mul_flip(p, &mut p_len, 0, &q[..q_len], v, &mut pn)
        } else {
            add_mul(&mut p[..p_len], &q[..q_len], v)
        };
        push_or_drop(p, &mut p_len, &mut e, c);
        let c = mul_prim(&mut q[..q_len], u);
        push(q, &mut q_len, c);
    }
    (buf_len(&p[..p_len]), q_len, e, pn)
}

fn engel_leaf_terms<S: EngelSeries>(
    a: u64,
    b: u64,
    shift: u64,
    p: &mut [u64],
    q: &mut [u64],
) -> (usize, usize, usize, bool) {
    q[0] = 1;
    let (mut p_len, mut q_len, mut e, mut pn) = (0, 1, 0, false);
    let mut n = a;
    while n < b {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        let mut vn = S::SIGNED && S::neg(n);
        let (mut u, mut v, mut j) = (S::q(n), S::p(n), 1);
        n += 1;
        while shift < 64 && n < b {
            debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
            let w = S::q(n) as u128;
            let (uw, vw) = (u as u128 * w, v as u128 * w);
            if (uw >> 64 | vw >> 64) != 0 || (vw as u64).leading_zeros() < shift as u32 {
                break;
            }
            let (vw, flip) = add_or_sub(
                ((vw as u64) << shift) as u128,
                S::p(n) as u128,
                S::SIGNED && S::neg(n) != vn,
            );
            if vw >> 64 != 0 {
                break;
            }
            (u, v, j) = (uw as u64, vw as u64, j + 1);
            vn ^= flip;
            n += 1;
        }
        let sh = shift * j;
        if S::SIGNED && p_len == 0 {
            pn = vn;
        }
        let opposite = S::SIGNED && vn != pn;
        let fused = !opposite && (p_len == 0 || (sh < 64 && u.leading_zeros() as u64 >= sh));
        let add = if fused && e == 0 { v } else { 0 };
        mul_shl(p, &mut p_len, &mut e, u, sh, add);
        if !fused && e == 0 {
            let c = if opposite {
                sub_mul_flip(p, &mut p_len, 0, &[v], 1, &mut pn)
            } else {
                add_prim(&mut p[..p_len], v) as u64
            };
            push_or_drop(p, &mut p_len, &mut e, c);
        }
        let c = mul_prim(&mut q[..q_len], u);
        push(q, &mut q_len, c);
    }
    (p_len, q_len, e, pn)
}

fn hyper_leaf_terms<S: HyperSeries>(
    a: u64,
    b: u64,
    shift: u64,
    p: &mut [u64],
    q: &mut [u64],
    r: &mut [u64],
) -> (usize, usize, usize, usize, bool) {
    (q[0], r[0]) = (1, 1);
    let (mut p_len, mut q_len, mut r_len, mut e, mut pn) = (0, 1, 1, 0, false);
    let mut n = a;
    while n < b {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        let mut wn = S::SIGNED && S::neg(n);
        let (mut u, mut v, mut w, mut j) = (S::q(n), S::r(n), S::p(n), 1);
        n += 1;
        while shift < 64 && n < b {
            debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
            let (x, y) = (S::q(n) as u128, S::r(n) as u128);
            let (ux, vy, wx) = (u as u128 * x, v as u128 * y, w as u128 * x);
            if (ux >> 64 | vy >> 64 | wx >> 64) != 0 || (wx as u64).leading_zeros() < shift as u32 {
                break;
            }
            let (wx, flip) = add_or_sub(
                ((wx as u64) << shift) as u128,
                v as u128 * S::p(n) as u128,
                S::SIGNED && S::neg(n) != wn,
            );
            if wx >> 64 != 0 {
                break;
            }
            (u, v, w, j) = (ux as u64, vy as u64, wx as u64, j + 1);
            wn ^= flip;
            n += 1;
        }
        mul_shl(p, &mut p_len, &mut e, u, shift * j, 0);
        let rs = &r[e.min(r_len)..r_len];
        p_len = p_len.max(rs.len());
        let c = if S::SIGNED && wn != pn {
            sub_mul_flip(p, &mut p_len, 0, rs, w, &mut pn)
        } else {
            add_mul(&mut p[..p_len], rs, w)
        };
        push_or_drop(p, &mut p_len, &mut e, c);
        let c = mul_prim(&mut r[..r_len], v);
        push(r, &mut r_len, c);
        let c = mul_prim(&mut q[..q_len], u);
        push(q, &mut q_len, c);
    }
    (buf_len(&p[..p_len]), q_len, r_len, e, pn)
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
    full: bool,
) -> (usize, usize, &'t [u64], usize, bool) {
    t[0] = 1;
    let (mut p_len, mut e, mut t_len, mut et, mut pn) = (0, 0, 1, 0usize, false);
    for n in a..b {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        mul_shl(p, &mut p_len, &mut e, S::q(n), shift, 0);
        if t_len != 0 {
            let top = (et + t_len + 1).saturating_sub(e);
            if top > p.len() {
                let d = top - p.len();
                let m = d.min(p_len);
                p.copy_within(m..p_len, 0);
                p[p_len - m..p_len].fill(0);
                p_len -= m;
                e += d;
            }
            let (off, ts) = match et.checked_sub(e) {
                Some(off) => (off, &t[..t_len]),
                None => (0, &t[(e - et).min(t_len)..t_len]),
            };
            p_len = p_len.max(off + ts.len());
            let c = if S::SIGNED && S::neg(n) != pn {
                sub_mul_flip(p, &mut p_len, off, ts, S::p(n), &mut pn)
            } else {
                add_mul(&mut p[off..p_len], ts, S::p(n))
            };
            push_or_drop(p, &mut p_len, &mut e, c);
        }
        if n + 1 == b && !full {
            break;
        }
        let c = mul_prim(&mut t[..t_len], S::r(n));
        push_or_drop(t, &mut t_len, &mut et, c);
        let drop = mul_len(&t[..t_len], x).saturating_sub(tmp.len());
        t_len = hi_mul(&t[..t_len], x, drop, tmp);
        std::mem::swap(&mut t, &mut tmp);
        et += drop;
    }
    let t: &'t [u64] = t;
    (buf_len(&p[..p_len]), e, &t[..t_len], et, pn)
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

fn product(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut v = vec![0; a.len() + b.len()];
    mul_dyn(a, b, &mut v);
    trim_lz(&mut v);
    v
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

fn bit_len(x: u64) -> u64 {
    (64 - x.leading_zeros()) as u64
}

// Whether prod f(n) over [a, b) fits in cap bits, bounding each product's
// length by the sum of its factors' lengths.
fn prod_fits(a: u64, b: u64, f: impl Fn(u64) -> u64, cap: u64) -> bool {
    let mut bits = 0;
    for n in a..b {
        bits += bit_len(f(n));
        if bits > cap {
            return false;
        }
    }
    true
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
    type Ctx;
    fn merge(ctx: &Self::Ctx, l: Self, r: Self) -> Self;
    fn leaf(ctx: &Self::Ctx, a: u64, b: u64, last: bool) -> Self;
    fn finalize(&self, ctx: &Self::Ctx, out: &mut [u64]) -> bool;
    const PARALLEL: bool = false;
}

pub fn bin_split_tree<N, const CUTOFF: u64>(ctx: &N::Ctx, a: u64, b: u64, last: bool) -> N
where
    N: Node + Send,
    N::Ctx: Sync,
{
    const { assert!(CUTOFF >= 2) };
    debug_assert!(a < b);
    if b - a < CUTOFF {
        return N::leaf(ctx, a, b, last);
    }
    let m = (a + b) / 2;
    let (l, r) = join2(
        N::PARALLEL && b - a >= DYN_BIN_SPLIT_PAR_CUTOFF,
        || bin_split_tree::<N, CUTOFF>(ctx, a, m, false),
        || bin_split_tree::<N, CUTOFF>(ctx, m, b, last),
    );
    N::merge(ctx, l, r)
}

pub fn bin_split<N, const CUTOFF: u64>(ctx: &N::Ctx, a: u64, b: u64) -> N
where
    N: Node + Send,
    N::Ctx: Sync,
{
    bin_split_tree::<N, CUTOFF>(ctx, a, b, true)
}

#[derive(Debug, Clone, Copy)]
pub struct Shift(pub u64);

//$\sum^{N}_{n = 1} (-1)^{\sigma(n)} \; \frac{P(n)}{Q(n)} \; 2^{-kn}$
pub trait BBPSeries {
    const SIGNED: bool = false;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
    fn neg(_: u64) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub struct DynNodeBBP<S: BBPSeries> {
    pub a: u64,
    pub b: u64,
    pub p: Vec<u64>,
    pub q: Vec<u64>,
    // P's sign.
    pub neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<S: BBPSeries> Node for DynNodeBBP<S> {
    type Ctx = Shift;
    const PARALLEL: bool = true;

    fn leaf(ctx: &Shift, a: u64, b: u64, _: bool) -> Self {
        let terms = (b - a) as usize;
        let words = split_sh(ctx.0 * (b - a)).0 + 1;
        let (mut p, mut q) = (vec![0; terms + words], vec![0; terms]);
        let (p_len, q_len, e, neg) = bbp_leaf_terms::<S>(a, b, ctx.0, &mut p, &mut q);
        debug_assert_eq!(e, 0);
        p.truncate(p_len);
        q.truncate(q_len);
        DynNodeBBP {
            a,
            b,
            p,
            q,
            neg,
            _s: PhantomData,
        }
    }

    fn merge(ctx: &Shift, l: Self, r: Self) -> Self {
        let (sl, sb) = split_sh(ctx.0 * (r.b - r.a)); // r(b - m)
        let pl = l.p.len() + r.q.len();
        let tl = l.q.len() + r.p.len();

        let mut p = vec![0; (sl + pl + 1).max(tl) + 1];
        let mut scratch = ScratchGuard::acquire();
        let t = scratch.get(tl);
        let par = merge_par(l.q.len() + r.q.len());
        let (_, (t_len, q)) = join2(
            par,
            || mul_dyn(&l.p, &r.q, &mut p[sl..sl + pl]),
            || join2(par, || mul_exact(&l.q, &r.p, 0, t), || product(&l.q, &r.q)),
        );
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);
        let (mut len, mut neg) = (p.len(), S::SIGNED && l.neg);
        let c = add_signed(&mut p, &mut len, 0, &t[..t_len], &mut neg, r.neg, S::SIGNED);
        debug_assert_eq!(c, 0);
        trim_lz(&mut p);

        DynNodeBBP {
            a: l.a,
            b: r.b,
            p,
            q,
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &Shift, out: &mut [u64]) -> bool {
        ratio_to_fraction_dyn(&self.p, &self.q, (ctx.0 * (self.b - self.a)) as i64, out);
        S::SIGNED && self.neg && !self.p.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct StaticNodeBBP<const N: usize, S: BBPSeries> {
    pub(crate) a: u64,
    pub(crate) b: u64,
    pub(crate) p: [u64; N],
    pub(crate) p_len: usize,
    pub(crate) q: [u64; N],
    pub(crate) q_len: usize,
    pub(crate) neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<const N: usize, S: BBPSeries> StaticNodeBBP<N, S> {
    // Whether [a, b)'s exact P and Q * 2^(shift t) fit N limbs, where
    // |P| <= prod q * 2^(shift (t - 1)) * t * max p.
    fn leaf_fits(shift: u64, a: u64, b: u64) -> bool {
        let t = b - a;
        let cap = 64 * N as u64;
        let (mut bits, mut p_bits) = (shift * t + bit_len(t), 0);
        for n in a..b {
            bits += bit_len(S::q(n));
            p_bits = p_bits.max(bit_len(S::p(n)));
            if bits + p_bits > cap {
                return false;
            }
        }
        true
    }

    #[inline(never)]
    fn exact_leaf(shift: u64, a: u64, b: u64) -> Self {
        let (mut p, mut q) = ([0; N], [0; N]);
        let (p_len, mut q_len, mut e, neg) = bbp_leaf_terms::<S>(a, b, shift, &mut p, &mut q);
        mul_shl(&mut q, &mut q_len, &mut e, 1, shift * (b - a), 0);
        assert_eq!(e, 0, "one term's P and Q * 2^shift must fit N limbs");

        StaticNodeBBP {
            a,
            b,
            p,
            p_len,
            q,
            q_len,
            neg,
            _s: PhantomData,
        }
    }
}

impl<const N: usize, S: BBPSeries> Node for StaticNodeBBP<N, S> {
    type Ctx = Shift;

    // Only merges drop limbs, so a leaf too big for N limbs splits itself.
    fn leaf(ctx: &Shift, a: u64, b: u64, last: bool) -> Self {
        if b - a > 1 && !Self::leaf_fits(ctx.0, a, b) {
            let m = a + (b - a) / 2;
            return Self::merge(
                ctx,
                Self::leaf(ctx, a, m, false),
                Self::leaf(ctx, m, b, last),
            );
        }
        Self::exact_leaf(ctx.0, a, b)
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

        let (mut p_len, mut neg) = (t_len.max(t2_len), S::SIGNED && r.neg);
        l.p[t2_len..p_len].fill(0);
        let carry = add_signed(
            &mut l.p,
            &mut p_len,
            0,
            &t[..t_len],
            &mut neg,
            l.neg,
            S::SIGNED,
        );

        let mut q_len = hi_mul_drop::<N>(lq, rq, k, &mut r.p);

        let mut d = 0;
        push_or_drop(&mut l.p, &mut p_len, &mut d, carry);
        if d != 0 {
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
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, _: &Shift, out: &mut [u64]) -> bool {
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], 0, out);
        S::SIGNED && self.neg && self.p_len != 0
    }
}

//$\sum^{N}_{n=1} (-1)^{\sigma(n)} \; P(n) \prod^{n}_{r=1} \frac{2^{-k}}{Q(r)}$
pub trait EngelSeries {
    const SIGNED: bool = false;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
    // Whether term n is negative.
    fn neg(_: u64) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub struct DynNodeEngel<S: EngelSeries> {
    pub(crate) a: u64,
    pub(crate) b: u64,
    pub(crate) p: Vec<u64>,
    pub(crate) q: Vec<u64>,
    // P's sign.
    pub(crate) neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<S: EngelSeries> Node for DynNodeEngel<S> {
    type Ctx = Shift;
    const PARALLEL: bool = true;

    fn leaf(ctx: &Shift, a: u64, b: u64, _: bool) -> Self {
        let terms = (b - a) as usize;
        let words = split_sh(ctx.0 * (b - a)).0 + 1;
        let (mut p, mut q) = (vec![0; terms + words], vec![0; terms]);
        let (p_len, q_len, e, neg) = engel_leaf_terms::<S>(a, b, ctx.0, &mut p, &mut q);
        debug_assert_eq!(e, 0);
        p.truncate(p_len);
        q.truncate(q_len);
        DynNodeEngel {
            a,
            b,
            p,
            q,
            neg,
            _s: PhantomData,
        }
    }

    fn merge(ctx: &Shift, l: Self, r: Self) -> Self {
        let (sl, sb) = split_sh(ctx.0 * (r.b - r.a));
        let pl = l.p.len() + r.q.len();

        let mut p = vec![0; (sl + pl + 1).max(r.p.len()) + 1];
        let (_, q) = join2(
            merge_par(l.q.len() + r.q.len()),
            || mul_dyn(&l.p, &r.q, &mut p[sl..sl + pl]),
            || product(&l.q, &r.q),
        );
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);
        let (mut len, mut neg) = (p.len(), S::SIGNED && l.neg);
        let c = add_signed(&mut p, &mut len, 0, &r.p, &mut neg, r.neg, S::SIGNED);
        debug_assert_eq!(c, 0);
        trim_lz(&mut p);

        DynNodeEngel {
            a: l.a,
            b: r.b,
            p,
            q,
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &Shift, out: &mut [u64]) -> bool {
        ratio_to_fraction_dyn(&self.p, &self.q, (ctx.0 * (self.b - self.a)) as i64, out);
        S::SIGNED && self.neg && !self.p.is_empty()
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
    neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<const N: usize, S: EngelSeries> StaticNodeEngel<N, S> {
    // P drops limbs on its own, but [a, b)'s exact Q must fit N limbs.
    fn leaf_fits(a: u64, b: u64) -> bool {
        prod_fits(a, b, S::q, 64 * N as u64)
    }

    #[inline(never)]
    fn exact_leaf(shift: u64, a: u64, b: u64) -> Self {
        let (mut p, mut q) = ([0; N], [0; N]);
        let (p_len, q_len, ep, neg) = engel_leaf_terms::<S>(a, b, shift, &mut p, &mut q);
        StaticNodeEngel {
            a,
            b,
            ep,
            eq: 0,
            p,
            p_len,
            q,
            q_len,
            neg,
            _s: PhantomData,
        }
    }
}

impl<const N: usize, S: EngelSeries> Node for StaticNodeEngel<N, S> {
    type Ctx = Shift;

    // Only merges drop Q's limbs, so a leaf too big for N limbs splits itself.
    fn leaf(ctx: &Shift, a: u64, b: u64, last: bool) -> Self {
        if b - a > 1 && !Self::leaf_fits(a, b) {
            let m = a + (b - a) / 2;
            return Self::merge(
                ctx,
                Self::leaf(ctx, a, m, false),
                Self::leaf(ctx, m, b, last),
            );
        }
        Self::exact_leaf(ctx.0, a, b)
    }

    fn merge(ctx: &Shift, mut l: Self, r: Self) -> Self {
        const { assert!(N >= 2) };
        let (sl, sb) = split_sh(ctx.0 * (r.b - r.a));
        let (lp, lq) = (&l.p[..l.p_len], &l.q[..l.q_len]);
        let (rp, rq) = (&r.p[..r.p_len], &r.q[..r.q_len]);

        let t_len = mul_len(lp, rq);
        let t_e = l.ep + r.eq + sl;
        let t_top = (t_len != 0) as usize * (t_e + t_len + (sb != 0) as usize);
        let r_top = (!rp.is_empty()) as usize * (r.ep + rp.len());
        let ep = (t_top.max(r_top) + 1).saturating_sub(N);

        let (mut p, mut p_len, mut neg) = ([0; N], 0, S::SIGNED && l.neg);
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
            let c = add_signed(&mut p, &mut p_len, off, rp, &mut neg, r.neg, S::SIGNED);
            push(&mut p, &mut p_len, c);
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
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &Shift, out: &mut [u64]) -> bool {
        let shift = (ctx.0 * (self.b - self.a)) as i64 + 64 * (self.eq as i64 - self.ep as i64);
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], shift, out);
        S::SIGNED && self.neg && self.p_len != 0
    }
}

//$\sum^{N}_{n=1} (-1)^{\sigma(n)} \; x^{n} \; \frac{P(n)}{R(n)} \; \prod^{n}_{r=1} \frac{R(r)}{Q(r)} \; 2^{-k}$
pub trait HyperSeries {
    const SIGNED: bool = false;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
    fn r(n: u64) -> u64;
    // Whether term n is negative.
    fn neg(_: u64) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub struct DynHyperCtx {
    shift: u64,
    // None when x = 1.
    x: Option<Vec<u64>>,
}

impl DynHyperCtx {
    pub fn new(x: &[u64], shift: u64) -> Self {
        debug_assert!(
            x.last().is_some_and(|&l| l != 0),
            "x must be trimmed and nonzero"
        );
        DynHyperCtx {
            shift,
            x: (x != [1]).then(|| x.to_vec()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StaticHyperCtx<const N: usize> {
    shift: u64,
    x: Option<([u64; N], usize)>,
}

impl<const N: usize> StaticHyperCtx<N> {
    pub fn new(x: &[u64], shift: u64) -> Self {
        debug_assert!(
            x.last().is_some_and(|&l| l != 0),
            "x must be trimmed and nonzero"
        );
        assert!(x.len() <= N, "x must fit N limbs");
        let mut limbs = [0; N];
        limbs[..x.len()].copy_from_slice(x);
        StaticHyperCtx {
            shift,
            x: (x != [1]).then_some((limbs, x.len())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynNodeHyper<S: HyperSeries> {
    pub(crate) a: u64,
    pub(crate) b: u64,
    pub(crate) p: Vec<u64>,
    pub(crate) q: Vec<u64>,
    pub(crate) r: Option<Vec<u64>>,
    pub(crate) neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<S: HyperSeries> Node for DynNodeHyper<S> {
    type Ctx = DynHyperCtx;
    const PARALLEL: bool = true;

    fn leaf(ctx: &DynHyperCtx, a: u64, b: u64, last: bool) -> Self {
        let terms = (b - a) as usize;
        let words = split_sh(ctx.shift * (b - a)).0 + 1;
        let mut q = vec![0; terms];
        let (p, q_len, r, neg) = match ctx.x.as_deref() {
            None => {
                let (mut p, mut r) = (vec![0; terms + words], vec![0; terms]);
                let (p_len, q_len, r_len, e, neg) =
                    hyper_leaf_terms::<S>(a, b, ctx.shift, &mut p, &mut q, &mut r);
                debug_assert_eq!(e, 0);
                p.truncate(p_len);
                r.truncate(r_len);
                (p, q_len, (!last).then_some(r), neg)
            }
            Some(x) => {
                let cap = terms * (x.len() + 1) + 1;
                let mut p = vec![0; cap + words];
                let mut scratch = ScratchGuard::acquire();
                let [t, tmp] = scratch.get_splits([cap + x.len(); 2]);
                let (p_len, e, t, et, neg) =
                    hyper_x_leaf_terms::<S>(a, b, ctx.shift, x, &mut p, t, tmp, mul_exact, !last);
                debug_assert_eq!(e + et, 0);
                p.truncate(p_len);
                let q_len = prod_terms(a, b, S::q, &mut q);
                (p, q_len, (!last).then(|| t.to_vec()), neg)
            }
        };
        q.truncate(q_len);
        DynNodeHyper {
            a,
            b,
            p,
            q,
            r,
            neg,
            _s: PhantomData,
        }
    }

    fn merge(ctx: &DynHyperCtx, left: Self, right: Self) -> Self {
        let lr = left
            .r
            .expect("a node that ends the sum can't merge as a left child");
        let (sl, sb) = split_sh(ctx.shift * (right.b - right.a));
        let pl = left.p.len() + right.q.len();
        let tl = right.p.len() + lr.len();

        let par = merge_par(left.q.len() + right.q.len());
        let mut p = vec![0; (sl + pl + 1).max(tl) + 1];
        let mut scratch = ScratchGuard::acquire();
        let t = scratch.get(tl);
        let ((_, t_len), (q, r)) = join2(
            par,
            || {
                join2(
                    par,
                    || mul_dyn(&left.p, &right.q, &mut p[sl..sl + pl]),
                    || mul_exact(&right.p, &lr, 0, t),
                )
            },
            || {
                join2(
                    par,
                    || product(&left.q, &right.q),
                    || right.r.as_deref().map(|rr| product(&lr, rr)),
                )
            },
        );
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);
        let (mut len, mut neg) = (p.len(), S::SIGNED && left.neg);
        let c = add_signed(
            &mut p,
            &mut len,
            0,
            &t[..t_len],
            &mut neg,
            right.neg,
            S::SIGNED,
        );
        debug_assert_eq!(c, 0);
        trim_lz(&mut p);

        DynNodeHyper {
            a: left.a,
            b: right.b,
            p,
            q,
            r,
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &DynHyperCtx, out: &mut [u64]) -> bool {
        ratio_to_fraction_dyn(
            &self.p,
            &self.q,
            (ctx.shift * (self.b - self.a)) as i64,
            out,
        );
        S::SIGNED && self.neg && !self.p.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct StaticNodeHyper<const N: usize, S: HyperSeries> {
    a: u64,
    b: u64,
    p: [u64; N],
    p_len: usize,
    ep: usize,
    q: [u64; N],
    q_len: usize,
    eq: usize,
    r: Option<([u64; N], usize, usize)>,
    neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<const N: usize, S: HyperSeries> StaticNodeHyper<N, S> {
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
    ) -> (usize, usize, usize, usize, usize, bool) {
        let (mut t, mut tmp) = ([0; N], [0; N]);
        let (p_len, ep, t, er, neg) =
            hyper_x_leaf_terms::<S>(a, b, shift, x, p, &mut t, &mut tmp, hi_mul_drop::<N>, !last);
        r[..t.len()].copy_from_slice(t);
        (p_len, prod_terms(a, b, S::q, q), t.len(), ep, er, neg)
    }

    // P drops limbs on its own, and so does R when x != 1, but [a, b)'s exact
    // Q, and R when x = 1, must fit N limbs.
    fn leaf_fits(ctx: &StaticHyperCtx<N>, a: u64, b: u64) -> bool {
        let cap = 64 * N as u64;
        prod_fits(a, b, S::q, cap) && (ctx.x.is_some() || prod_fits(a, b, S::r, cap))
    }

    #[inline(never)]
    fn exact_leaf(ctx: &StaticHyperCtx<N>, a: u64, b: u64, last: bool) -> Self {
        let (mut p, mut q, mut r) = ([0; N], [0; N], [0; N]);
        let (p_len, q_len, r_len, ep, er, neg) = match &ctx.x {
            None => {
                let (p_len, q_len, r_len, ep, neg) =
                    hyper_leaf_terms::<S>(a, b, ctx.shift, &mut p, &mut q, &mut r);
                (p_len, q_len, r_len, ep, 0, neg)
            }
            Some((x, len)) => {
                Self::x_leaf(a, b, ctx.shift, &x[..*len], last, &mut p, &mut q, &mut r)
            }
        };
        StaticNodeHyper {
            a,
            b,
            p,
            p_len,
            ep,
            q,
            q_len,
            eq: 0,
            r: (!last).then_some((r, r_len, er)),
            neg,
            _s: PhantomData,
        }
    }
}

impl<const N: usize, S: HyperSeries> Node for StaticNodeHyper<N, S> {
    type Ctx = StaticHyperCtx<N>;

    // Only merges drop Q's limbs, so a leaf too big for N limbs splits itself.
    fn leaf(ctx: &StaticHyperCtx<N>, a: u64, b: u64, last: bool) -> Self {
        if b - a > 1 && !Self::leaf_fits(ctx, a, b) {
            let m = a + (b - a) / 2;
            return Self::merge(
                ctx,
                Self::leaf(ctx, a, m, false),
                Self::leaf(ctx, m, b, last),
            );
        }
        Self::exact_leaf(ctx, a, b, last)
    }

    fn merge(ctx: &StaticHyperCtx<N>, mut left: Self, mut right: Self) -> Self {
        const { assert!(N >= 2) };
        let (sl, sb) = split_sh(ctx.shift * (right.b - right.a));
        let (lp, lq) = (&left.p[..left.p_len], &left.q[..left.q_len]);
        let (rp, rq) = (&right.p[..right.p_len], &right.q[..right.q_len]);
        let (lr, ler) = left
            .r
            .as_ref()
            .map(|(r, len, e)| (&r[..*len], *e))
            .expect("a node that ends the sum can't merge as a left child");

        let (t_len, u_len) = (mul_len(lp, rq), mul_len(rp, lr));
        let (t_e, u_e) = (left.ep + right.eq + sl, right.ep + ler);
        let t_top = (t_len != 0) as usize * (t_e + t_len + (sb != 0) as usize);
        let u_top = (u_len != 0) as usize * (u_e + u_len);
        let ep = (t_top.max(u_top) + 1).saturating_sub(N);

        let (mut p, mut p_len, mut neg) = ([0; N], 0, S::SIGNED && left.neg);
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
                let u = &left.p[..u_len];
                let c = add_signed(&mut p, &mut p_len, off, u, &mut neg, right.neg, S::SIGNED);
                push(&mut p, &mut p_len, c);
            }
        }

        let kq = mul_len(lq, rq).saturating_sub(N);
        let q_len = hi_mul_drop::<N>(lq, rq, kq, &mut right.p);

        let r = right.r.as_ref().map(|(rr, len, e)| {
            let rr = &rr[..*len];
            let kr = mul_len(lr, rr).saturating_sub(N);
            let r_len = hi_mul_drop::<N>(lr, rr, kr, &mut left.p);
            (left.p, r_len, ler + e + kr)
        });

        StaticNodeHyper {
            a: left.a,
            b: right.b,
            p,
            p_len,
            ep,
            q: right.p,
            q_len,
            eq: left.eq + right.eq + kq,
            r,
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &StaticHyperCtx<N>, out: &mut [u64]) -> bool {
        let shift = (ctx.shift * (self.b - self.a)) as i64 + 64 * (self.eq as i64 - self.ep as i64);
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], shift, out);
        S::SIGNED && self.neg && self.p_len != 0
    }
}
