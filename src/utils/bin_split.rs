use std::marker::PhantomData;

use crate::utils::{
    div::{rcp_dyn, rcp_static},
    mul::{hi_mul_dyn, hi_mul_static, mul_add_prim, mul_dyn, mul_prim},
    utils::{add_buf, add_mul, buf_len, shl_buf, shr_buf, trim_lz},
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

// Exact P and Q over the terms [a, b); p needs 2(b - a) limbs and q needs b - a.
fn bbp_leaf_terms<S: BBPSeries>(a: u64, b: u64, p: &mut [u64], q: &mut [u64]) -> (usize, usize) {
    p[0] = S::p(a);
    q[0] = S::q(a);
    let (mut p_len, mut q_len) = (1, 1);
    for n in (a + 1)..b {
        let u = S::q(n);
        let v = S::p(n);
        let c = mul_prim(&mut p[..p_len], u << S::R);
        push(p, &mut p_len, c);
        let c = add_mul(&mut p[..p_len], &q[..q_len], v);
        push(p, &mut p_len, c);
        let c = mul_prim(&mut q[..q_len], u);
        push(q, &mut q_len, c);
    }
    (p_len, q_len)
}

// Exact P and Q over the terms [a, b); p and q each need b - a limbs. Runs of
// terms whose own P and Q fit in one limb are folded in with a single pass.
fn hyper_leaf_terms<S: HyperSeries>(
    a: u64,
    b: u64,
    p: &mut [u64],
    q: &mut [u64],
) -> (usize, usize) {
    q[0] = 1;
    let (mut p_len, mut q_len) = (0, 1);
    let mut n = a;
    while n < b {
        let (mut u, mut v) = (S::q(n), S::p(n));
        n += 1;
        while n < b {
            let w = S::q(n) as u128;
            let (uw, vw) = (u as u128 * w, v as u128 * w + S::p(n) as u128);
            if (uw | vw) >> 64 != 0 {
                break;
            }
            (u, v) = (uw as u64, vw as u64);
            n += 1;
        }
        let c = mul_add_prim(&mut p[..p_len], u, v);
        push(p, &mut p_len, c);
        let c = mul_prim(&mut q[..q_len], u);
        push(q, &mut q_len, c);
    }
    (p_len, q_len)
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
    shift: u64,
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
    let (w, v) = split_bits(64 * (q.len() + x.len() - 1 - len) as u64 + shift);
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

fn ratio_to_fraction_dyn(p: &[u64], q: &[u64], shift: u64, out: &mut [u64]) {
    let len = out.len();
    let mut scratch = ScratchGuard::acquire();
    let [x, y] = scratch.get_splits([len + 2, len + 3]);
    ratio_to_fraction(p, q, shift, out, x, y, rcp_dyn, hi_mul_dyn);
}

fn ratio_to_fraction_static<const N: usize>(p: &[u64], q: &[u64], out: &mut [u64]) {
    let len = out.len();
    assert!(len + 3 <= N, "finalize output needs out.len() + 3 <= N");
    let (mut x, mut y) = ([0; N], [0; N]);
    let (x, y) = (&mut x[..len + 2], &mut y[..len + 3]);
    ratio_to_fraction(p, q, 0, out, x, y, rcp_static::<N>, hi_mul_static::<N>);
}

pub trait Node {
    fn merge(l: Self, r: Self) -> Self;
    fn leaf(a: u64, b: u64) -> Self;
    fn finalize(&self, out: &mut [u64]);
}

fn bin_split<N, const CUTOFF: u64>(a: u64, b: u64) -> N
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
        let (mut p, mut q) = (vec![0; 2 * terms], vec![0; terms]);
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
        let t2 = scratch.get(l.q.len() + r.p.len());
        mul_dyn(&l.q, &r.p, t2);

        let mut p = fit(r.p, (sl + pl + 1).max(t2.len()));
        mul_dyn(&l.p, &r.q, &mut p[sl..sl + pl]);
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);
        if add_buf(&mut p, t2) {
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

    // out = floor(S * B^out.len()) for S = P / (Q * 2^(R(b - a))).
    fn finalize(&self, out: &mut [u64]) {
        ratio_to_fraction_dyn(&self.p, &self.q, S::R * (self.b - self.a), out);
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

        let (sl, sb) = split_bits(S::R * (b - a));
        q.copy_within(..q_len, sl);
        q[..sl].fill(0);
        q_len += sl;
        let c = shl_buf(&mut q[sl..q_len], sb);
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

        // lp and rp are dead once read, so t2 is built in l.p and Q in r.p.
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
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], out);
    }
}

pub trait HyperSeries {
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
}

#[derive(Debug, Clone)]
pub struct DynNodeHyper<S: HyperSeries> {
    p: Vec<u64>,
    q: Vec<u64>,
    _s: PhantomData<S>,
}

impl<S: HyperSeries> Node for DynNodeHyper<S> {
    fn leaf(a: u64, b: u64) -> Self {
        let terms = (b - a) as usize;
        let (mut p, mut q) = (vec![0; terms], vec![0; terms]);
        let (p_len, q_len) = hyper_leaf_terms::<S>(a, b, &mut p, &mut q);
        p.truncate(p_len);
        q.truncate(q_len);
        DynNodeHyper {
            p,
            q,
            _s: PhantomData,
        }
    }

    // P = Pl * Qr + Pr, where Pr can outgrow Pl * Qr when Pl is small; the
    // spare top limb absorbs the add's carry.
    fn merge(l: Self, r: Self) -> Self {
        let mut p = vec![0; (l.p.len() + r.q.len()).max(r.p.len()) + 1];
        mul_dyn(&l.p, &r.q, &mut p);
        let carry = add_buf(&mut p, &r.p);
        debug_assert!(!carry);
        trim_lz(&mut p);

        let mut q = fit(l.p, l.q.len() + r.q.len());
        mul_dyn(&l.q, &r.q, &mut q);
        trim_lz(&mut q);

        DynNodeHyper {
            p,
            q,
            _s: PhantomData,
        }
    }

    // out = floor(S * B^out.len()) for S = P / Q.
    fn finalize(&self, out: &mut [u64]) {
        ratio_to_fraction_dyn(&self.p, &self.q, 0, out);
    }
}

// Stores P / B^e and Q / B^e, where e counts low limbs dropped to fit N.
#[derive(Debug, Clone)]
pub struct StaticNodeHyper<const N: usize, S: HyperSeries> {
    e: usize,
    p: [u64; N],
    p_len: usize,
    q: [u64; N],
    q_len: usize,
    _s: PhantomData<S>,
}

impl<const N: usize, S: HyperSeries> Node for StaticNodeHyper<N, S> {
    fn leaf(a: u64, b: u64) -> Self {
        let (mut p, mut q) = ([0; N], [0; N]);
        let (p_len, q_len) = hyper_leaf_terms::<S>(a, b, &mut p, &mut q);
        StaticNodeHyper {
            e: 0,
            p,
            p_len,
            q,
            q_len,
            _s: PhantomData,
        }
    }

    // Pl * Qr and Ql * Qr carry B^(el + er) but Pr only carries B^er, so Pr
    // drops el more limbs than the products to share their scale. k keeps a
    // spare limb above P so the add's carry always fits in N.
    fn merge(mut l: Self, r: Self) -> Self {
        const { assert!(N >= 2) };
        let (lp, lq) = (&l.p[..l.p_len], &l.q[..l.q_len]);
        let (rp, rq) = (&r.p[l.e.min(r.p_len)..r.p_len], &r.q[..r.q_len]);

        let k = (mul_len(lp, rq).max(rp.len()) + 1)
            .max(mul_len(lq, rq))
            .saturating_sub(N);
        let rp = &rp[k.min(rp.len())..];

        // lp is dead once read, so Q is built in l.p.
        let mut p = [0; N];
        let t_len = hi_mul_drop::<N>(lp, rq, k, &mut p);
        let q_len = hi_mul_drop::<N>(lq, rq, k, &mut l.p);

        let mut p_len = t_len.max(rp.len());
        if add_buf(&mut p[..p_len], rp) {
            p[p_len] = 1;
            p_len += 1;
        }

        StaticNodeHyper {
            e: l.e + r.e + k,
            p,
            p_len,
            q: l.p,
            q_len,
            _s: PhantomData,
        }
    }

    // out = floor(S * B^out.len()) for S = P / Q.
    fn finalize(&self, out: &mut [u64]) {
        ratio_to_fraction_static::<N>(&self.p[..self.p_len], &self.q[..self.q_len], out);
    }
}
