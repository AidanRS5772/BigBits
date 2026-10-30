use std::{
    mem, panic,
    sync::{PoisonError, RwLock},
    thread,
};

use crate::utils::{
    series::{
        bin_split, bin_split_tree, join2, ratio_to_fraction_dyn, ratio_to_fraction_static,
        static_split_stack, BBPSeries, Bbp, DynHyperCtx, DynNodeBBP, DynNodeHyper, Hyper, Node,
        Series, Shift, StaticHyperCtx, StaticNodeBBP, StaticNodeHyper, WideHyperSeries,
    },
    div::div_dyn,
    mul::{hi_mul_static, mul_dyn, mul_prim},
    sqrt::{sqrt_approx_dyn, sqrt_approx_static},
    utils::{add_buf, add_mul, buf_len, shl_buf, shr_buf, sub_buf},
    CapacityError, ScratchGuard, DYN_LN2_PAR_CUTOFF, DYN_LN2_PAR_HALF, DYN_PI_PAR_CUTOFF,
    DYN_PI_PAR_HALF, STATIC_LN2_PAR_CUTOFF, STATIC_PI_PAR_CUTOFF,
};

fn cache_par(new: u64, cached: u64, cutoff: u64, half: u64) -> bool {
    new.saturating_mul(cached.saturating_mul(cached) + half * half) >= cutoff * half * half
}

pub fn bin_split_extend<'a, N>(ctx: &N::Ctx, node: &'a mut Option<N>, start: u64, end: u64) -> &'a N
where
    N: Node + Send,
    N::Ctx: Sync,
{
    let s = match node.take() {
        Some(old) => {
            let new = bin_split_tree::<N>(ctx, old.range().1, end, false);
            N::merge(ctx, old, new)
        }
        None => bin_split_tree::<N>(ctx, start, end, false),
    };
    node.insert(s)
}

const fn limbs<const L: usize>(bytes: &[u8]) -> [u64; L] {
    let mut t = [0; L];
    let mut i = 0;
    while i < 8 * L {
        t[i / 8] |= (bytes[i] as u64) << (8 * (i % 8));
        i += 1;
    }
    t
}

fn from_table(table: &[u64], out: &mut [u64]) -> bool {
    let hit = out.len() <= table.len();
    if hit {
        out.copy_from_slice(&table[table.len() - out.len()..]);
    }
    hit
}

struct DynCache<T> {
    nodes: T,
    buf: Vec<u64>,
}

fn cached<T: Default>(
    table: &[u64],
    cache: &RwLock<DynCache<T>>,
    out: &mut [u64],
    build: fn(&mut T, &mut Vec<u64>, usize),
) {
    let get = |c: &DynCache<T>, out: &mut [u64]| {
        let hit = out.len() < c.buf.len();
        if hit {
            out.copy_from_slice(&c.buf[c.buf.len() - out.len()..]);
        }
        hit
    };
    if from_table(table, out) || get(&cache.read().unwrap_or_else(PoisonError::into_inner), out) {
        return;
    }
    let mut work = {
        let mut c = cache.write().unwrap_or_else(PoisonError::into_inner);
        if get(&c, out) {
            return;
        }
        DynCache {
            nodes: mem::take(&mut c.nodes),
            buf: Vec::new(),
        }
    };
    build(&mut work.nodes, &mut work.buf, out.len() + 1);
    out.copy_from_slice(&work.buf[1..]);
    let mut c = cache.write().unwrap_or_else(PoisonError::into_inner);
    if work.buf.len() > c.buf.len() {
        *c = work;
    }
}

fn static_jobs<const N: usize, const K: usize>(
    par: bool,
    jobs: [&(dyn Fn() -> [u64; N] + Sync); K],
) -> [[u64; N]; K] {
    if !par {
        return jobs.map(|j| j());
    }
    thread::scope(|sc| {
        jobs.map(|j| {
            (
                j,
                thread::Builder::new()
                    .stack_size(static_split_stack(N))
                    .spawn_scoped(sc, j)
                    .ok(),
            )
        })
        .map(|(j, h)| h.map_or_else(j, |h| h.join().unwrap_or_else(|e| panic::resume_unwind(e))))
    })
}

// $ln(2) = \frac{1}{192} \sum^{\infty}_{n=0}\left(\frac{128}{6n+1} + \frac{8}{6n+3} + \frac{8}{6n+5} + \frac{1}{2n+2}\right) 64^{-n}$
// $ln(2) = 2/3 \sum^{\infty}_{n=0} \frac{1}{6n+1} \; 64^{-n} + 1/24 \sum^{\infty}_{n=0} \frac{1}{6n+5} \; 64^{-n} + 1/72 \sum^{\infty}_{n=0} \frac{1}{2n+1} \; 64^{-n} + 1/384 \sum^{\infty}_{n=0} \frac{1}{n+1} \; 64^{-n} $
const LN2_CTX: Shift = Shift(6);

// 1 / ((A n + B) 64^n), summed from n = 0.
pub(crate) struct Ln2S<const A: u64, const B: u64>;
impl<const A: u64, const B: u64> BBPSeries for Ln2S<A, B> {
    const START: u64 = 0;
    // Term n is 2^(-s (n + 1)) / (A n + B), so for s >= 1 the tail from n is at most twice it.
    fn tail_bits(n: u64, s: i64) -> u64 {
        let s = s.max(0) as u64;
        (n + 1)
            .saturating_mul(s)
            .saturating_add(A.saturating_mul(n).saturating_add(B).ilog2() as u64)
            .saturating_sub(1)
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        A * n + B
    }
}

// Each series' weight 2^-k / d, as (d, k).
const LN2_W: [(u64, i64); 4] = [(3, -1), (3, 3), (9, 3), (3, 7)];

// The four series' terms [0, n) for len limbs: the most any of their tail bounds needs to
// reach a limb past len, which covers the shares' weights (at most 2^7 / 3 on a node's S).
pub(crate) fn ln2_end(len: usize) -> u64 {
    fn end<const A: u64, const B: u64>(bits: u64) -> u64 {
        Bbp::<Ln2S<A, B>>::new(LN2_CTX.0).range(bits).1
    }
    let bits = 64 * (len as u64 + 1);
    let ends = [end::<6, 1>, end::<6, 5>, end::<2, 1>, end::<1, 1>];
    ends.into_iter().map(|end| end(bits)).max().unwrap()
}

// t <- w S for a series' S = P / (Q 2^sh) and weight w = 2^-k / d: d scales Q's top
// dq.len() - 1 limbs into dq (finalize reads Q's top t.len() + 2), and 2^-k joins the
// shift. A tree from 0 sums its terms 64^-(n + 1), which the shift's -6 undoes.
fn ln2_share(
    (p, q, sh): (&[u64], &[u64], i64),
    (d, k): (u64, i64),
    dq: &mut [u64],
    t: &mut [u64],
    finalize: fn(&[u64], &[u64], i64, &mut [u64]),
) {
    let q = &q[..buf_len(q)];
    let cut = q.len().saturating_sub(dq.len() - 1);
    let top = q.len() - cut;
    dq[..top].copy_from_slice(&q[cut..]);
    dq[top] = mul_prim(&mut dq[..top], d);
    let dq = &dq[..=top];
    finalize(p, &dq[..buf_len(dq)], sh + k - 6 + 64 * cut as i64, t);
}

pub(crate) type Ln2Nodes = (
    Option<DynNodeBBP<Ln2S<6, 1>>>,
    Option<DynNodeBBP<Ln2S<6, 5>>>,
    Option<DynNodeBBP<Ln2S<2, 1>>>,
    Option<DynNodeBBP<Ln2S<1, 1>>>,
);

// ln 2 at len limbs into buf: the sum of the series' shares, each extending its cached
// node and finalizing in its own task.
pub(crate) fn ln2_build(nodes: &mut Ln2Nodes, buf: &mut Vec<u64>, len: usize) {
    fn share<S: BBPSeries>(node: &mut Option<DynNodeBBP<S>>, n: u64, w: (u64, i64), t: &mut [u64]) {
        let s = bin_split_extend(&LN2_CTX, node, 0, n);
        let sh = (LN2_CTX.0 * (s.b - s.a)) as i64;
        let mut scratch = ScratchGuard::acquire();
        let dq = scratch.get(t.len() + 3);
        ln2_share((&s.p, &s.q, sh), w, dq, t, ratio_to_fraction_dyn);
    }

    let n = ln2_end(len);
    let have = nodes.0.as_ref().map_or(0, |s| s.range().1);
    // Rounded up, so a one-limb extension counts.
    let limbs = |terms: u64| (3 * terms).div_ceil(32);
    let par = cache_par(
        limbs(n - have),
        limbs(have),
        DYN_LN2_PAR_CUTOFF,
        DYN_LN2_PAR_HALF,
    );
    let mut scratch = ScratchGuard::acquire();
    let [t1, t2, t3, t4] = scratch.get_splits([len; 4]);
    let (s1, s2, s3, s4) = nodes;
    let [w1, w2, w3, w4] = LN2_W;
    join2(
        par,
        || join2(par, || share(s1, n, w1, t1), || share(s2, n, w2, t2)),
        || join2(par, || share(s3, n, w3, t3), || share(s4, n, w4, t4)),
    );
    buf.resize(len, 0);
    for t in [t1, t2, t3, t4] {
        let carry = add_buf(buf, t);
        debug_assert!(!carry, "ln(2) < 1");
    }
}

const LN2_BIN: &[u8] = include_bytes!("ln2.bin");
pub(crate) const LN2_LIMBS: usize = LN2_BIN.len() / 8;
const _: () = assert!(LN2_BIN.len() % 8 == 0, "ln2.bin must hold whole u64 limbs");
static LN2_TABLE: [u64; LN2_LIMBS] = limbs(LN2_BIN);

pub fn ln2_dyn(out: &mut [u64]) {
    static LN2: RwLock<DynCache<Ln2Nodes>> = RwLock::new(DynCache {
        nodes: (None, None, None, None),
        buf: Vec::new(),
    });
    cached(&LN2_TABLE, &LN2, out, ln2_build);
}

#[inline(never)]
fn ln2_share_static<const N: usize, S: BBPSeries>(n: u64, w: (u64, i64), len: usize) -> [u64; N] {
    let s = bin_split::<StaticNodeBBP<N, S>>(&LN2_CTX, 0, n);
    let (mut dq, mut t) = ([0; N], [0; N]);
    let dq = &mut dq[..(len + 3).min(N)];
    ln2_share(
        s.ratio(&LN2_CTX),
        w,
        dq,
        &mut t[..len],
        ratio_to_fraction_static::<N>,
    );
    t
}

// Past the table, sums the series at out.len() + guard limbs.
fn ln2_sum<const N: usize>(out: &mut [u64], guard: usize) -> Result<(), CapacityError> {
    if from_table(&LN2_TABLE, out) {
        return Ok(());
    }
    ln2_static_sum::<N>(out, guard)
}

// The series at out.len() + guard limbs, which tests call directly to skip the table.
pub(crate) fn ln2_static_sum<const N: usize>(
    out: &mut [u64],
    guard: usize,
) -> Result<(), CapacityError> {
    let len = out.len() + guard;
    // Static finalize needs 2 limbs past its output.
    if len + 2 > N {
        return Err(CapacityError { min_n: len + 2 });
    }

    let n = ln2_end(len);
    let [w1, w2, w3, w4] = LN2_W;
    let shares = static_jobs::<N, 4>(
        len >= STATIC_LN2_PAR_CUTOFF,
        [
            &|| ln2_share_static::<N, Ln2S<6, 1>>(n, w1, len),
            &|| ln2_share_static::<N, Ln2S<6, 5>>(n, w2, len),
            &|| ln2_share_static::<N, Ln2S<2, 1>>(n, w3, len),
            &|| ln2_share_static::<N, Ln2S<1, 1>>(n, w4, len),
        ],
    );
    let mut buf = [0; N];
    for t in &shares {
        let carry = add_buf(&mut buf[..len], &t[..len]);
        debug_assert!(!carry, "ln(2) < 1");
    }
    out.copy_from_slice(&buf[guard..len]);
    Ok(())
}

pub fn ln2_static<const N: usize>(out: &mut [u64]) -> Result<(), CapacityError> {
    ln2_sum::<N>(out, 1)
}

pub(crate) fn ln2_static_unguarded<const N: usize>(out: &mut [u64]) -> Result<(), CapacityError> {
    ln2_sum::<N>(out, 0)
}

//$1/\pi = \frac{\sqrt{10005}}{4270934400} \; \sum^{\infty}_{n=0} (-1)^{n} \frac{(6n)!}{(3n)!(n!)^3 640320^{3n}} (545140134 \cdot n + 13591409)$
//$P / Q = \sum^{\infty}_{n=1} (-1)^{n} (545140134 \cdot n + 13591409) \prod^{n}_{r=1} \frac{(6r-5)(6r-1)(2r-1)}{r^3 \cdot 333833583375 \cdot 2^{15}}$
//$\pi = \frac{426880 \sqrt{10005} \cdot Q}{13591409 \cdot Q + P}$

const PI_POW: u128 = 333_833_583_375;
const PI_SHIFT: u64 = 15;
const PI_CTX: DynHyperCtx = DynHyperCtx::new_shift(PI_SHIFT);

pub(crate) struct Pi;
impl WideHyperSeries for Pi {
    const SIGNED: bool = true;
    const START: u64 = 1;
    // |term n| = (13591409 + 545140134 n) prod r(i) / (q(i) 2^s), and each r(i) / q(i) <
    // 72 / PI_POW < 2^-32.11, so the terms fall far faster than by half and the tail from n
    // is under twice term n.
    fn tail_bits(n: u64, s: i64) -> u64 {
        let s = s.max(0) as u64;
        let lead = 128 - (13591409 + 545140134 * n as u128).leading_zeros() as u64;
        n.saturating_mul(s + 32)
            .saturating_add(n / 10)
            .saturating_sub(lead + 1)
    }
    fn p(n: u128) -> u128 {
        (13591409 + 545140134 * n) * Self::r(n)
    }
    fn q(n: u128) -> u128 {
        n * n * n * PI_POW
    }
    fn r(n: u128) -> u128 {
        (6 * n - 5) * (6 * n - 1) * (2 * n - 1)
    }
    fn neg(n: u128) -> bool {
        (n & 1) == 1
    }
}

// out <- floor(x * 2^sh), which must fit out.
pub(crate) fn shifted(x: &[u64], sh: i64, out: &mut [u64]) {
    let x = &x[..buf_len(x)];
    let (w, v) = (
        (sh.unsigned_abs() / 64) as usize,
        (sh.unsigned_abs() % 64) as u8,
    );
    out.fill(0);
    if sh >= 0 {
        out[w..w + x.len()].copy_from_slice(x);
        let c = shl_buf(&mut out[w..], v);
        debug_assert_eq!(c, 0);
    } else if w < x.len() {
        let x = &x[w..];
        let n = x.len().min(out.len());
        debug_assert!(x.len() <= n + 1);
        out[..n].copy_from_slice(&x[..n]);
        shr_buf(&mut out[..n], v);
        if let Some(&hi) = x.get(n) {
            debug_assert!(v > 0 && hi >> v == 0);
            out[n - 1] |= hi << (64 - v);
        }
    }
}

// The terms [1, n) for f fraction limbs: an error e in S moves pi by about e pi / 13591409,
// so a tail bound a limb past f leaves room to spare.
pub(crate) fn pi_end(f: usize) -> u64 {
    let ctx = Hyper::<Pi>::new(&[1], PI_SHIFT).expect("x = 1 is below 2^PI_SHIFT");
    ctx.range(64 * (f as u64 + 1)).1
}

fn pi_denominator<'a>(
    (p, q, sh): (&[u64], &[u64], i64),
    qt: &'a mut [u64],
    pt: &mut [u64],
    t: &'a mut [u64],
) -> (&'a [u64], &'a [u64]) {
    let q = &q[..buf_len(q)];
    let a = 64 * (qt.len() as i64 - q.len() as i64) + q[q.len() - 1].leading_zeros() as i64;
    shifted(q, a, qt);
    shifted(p, a - sh, pt);
    t.fill(0);
    let c = add_mul(t, qt, 13591409);
    let borrow = sub_buf(t, pt);
    debug_assert!(c == 0 && !borrow, "T is positive and fits its limbs");
    (&qt[..buf_len(qt)], &t[..buf_len(t)])
}

fn pi_root(x: &mut [u64], root: &mut [u64], sqrt: fn(&mut [u64], &mut [u64])) {
    x.fill(0);
    x[root.len()] = 10005;
    sqrt(x, root);
    shr_buf(root, 32);
    let c = mul_prim(root, 426880);
    debug_assert_eq!(c, 0);
}

pub(crate) fn pi_build(node: &mut Option<DynNodeHyper<Pi>>, buf: &mut Vec<u64>, len: usize) {
    let f = len - 1;
    let n = pi_end(f);
    let have = node.as_ref().map_or(1, |s| s.range().1);
    let limbs = |terms: u64| terms * 4711 / 6400;
    let par = cache_par(
        limbs(n - have),
        limbs(have - 1),
        DYN_PI_PAR_CUTOFF,
        DYN_PI_PAR_HALF,
    );
    let mut scratch = ScratchGuard::acquire();
    let [qt, pt, t, x, root, num] = scratch.get_splits([len, len, len + 1, len + 1, len, 2 * len]);
    let mut den = None;
    rayon::in_place_scope(|sc| {
        let den = &mut den;
        let series = move || {
            let (node, qt, pt, t) = (node, qt, pt, t);
            let s = bin_split_extend(&PI_CTX, node, 1, n);
            debug_assert!(s.neg && !s.p.is_empty(), "S < 0");
            *den = Some(pi_denominator(
                (&s.p, &s.q, (PI_CTX.shift * (s.b - s.a)) as i64),
                qt,
                pt,
                t,
            ));
        };
        if par {
            sc.spawn(|_| series());
        } else {
            series();
        }
        pi_root(x, root, sqrt_approx_dyn);
    });
    let (qt, t) = den.unwrap();
    let num = &mut num[..root.len() + qt.len()];
    let c = mul_dyn(root, qt, num);
    debug_assert_eq!(c, 0);
    buf.resize(len + 1, 0);
    let c = div_dyn(&num[..buf_len(num)], t, buf);
    debug_assert!(c == 0 && buf[f] == 3 && buf[len] == 0);
    buf.truncate(len);
}

const PI_BIN: &[u8] = include_bytes!("pi.bin");
pub(crate) const PI_LIMBS: usize = PI_BIN.len() / 8;
const _: () = assert!(PI_BIN.len() % 8 == 0, "pi.bin must hold whole u64 limbs");
static PI_TABLE: [u64; PI_LIMBS] = limbs(PI_BIN);

pub fn pi_dyn(out: &mut [u64]) {
    static PI: RwLock<DynCache<Option<DynNodeHyper<Pi>>>> = RwLock::new(DynCache {
        nodes: None,
        buf: Vec::new(),
    });
    cached(&PI_TABLE, &PI, out, pi_build);
}

#[inline(never)]
fn pi_ratio_static<const N: usize>(f: usize) -> [u64; N] {
    let ctx = StaticHyperCtx::<N>::new(&[1], PI_SHIFT);
    let s = bin_split::<StaticNodeHyper<N, Pi>>(&ctx, 1, pi_end(f));
    debug_assert!(s.is_neg(), "S < 0");
    let (mut qt, mut t, mut v) = ([0; N], [0; N], [0; N]);
    let (qt, t) = pi_denominator(
        s.ratio(&ctx),
        &mut qt[..f + 1],
        &mut v[..f + 1],
        &mut t[..f + 2],
    );
    ratio_to_fraction_static::<N>(qt, t, 0, &mut v[..f + 1]);
    v
}

pub(crate) fn pi_static_sum<const N: usize>(out: &mut [u64]) -> Result<(), CapacityError> {
    let f = out.len();
    if f + 3 > N {
        return Err(CapacityError { min_n: f + 3 });
    }
    let sqrt = || {
        let (mut x, mut root) = ([0; N], [0; N]);
        pi_root(&mut x[..f + 2], &mut root[..f + 1], sqrt_approx_static::<N>);
        root
    };
    let [v, root] = static_jobs::<N, 2>(
        f >= STATIC_PI_PAR_CUTOFF,
        [&|| pi_ratio_static::<N>(f), &sqrt],
    );

    let mut buf = [0; N];
    let top = hi_mul_static::<N>(&root[..f + 1], &v[..f + 1], &mut buf[..f]);
    debug_assert_eq!(top, 3);
    buf[f] = top;
    out.copy_from_slice(&buf[1..f + 1]);
    Ok(())
}

pub fn pi_static<const N: usize>(out: &mut [u64]) -> Result<(), CapacityError> {
    if from_table(&PI_TABLE, out) {
        return Ok(());
    }
    pi_static_sum::<N>(out)
}
