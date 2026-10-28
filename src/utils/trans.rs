use crate::utils::series::{
    bin_split, join2, ratio_to_fraction_dyn, ratio_to_fraction_static, static_split_stack,
    BBPSeries, DynHyperCtx, DynNodeBBP, DynNodeEngel, DynNodeHyper, EngelSeries, HyperSeries, Node,
    Shift, StaticHyperCtx, StaticNodeBBP, StaticNodeEngel, StaticNodeHyper,
};
use crate::utils::consts::consts::{ln2_dyn, ln2_static_unguarded, shifted, LN2_LIMBS};
use crate::utils::div::{div_dyn, div_rem_dyn, div_rem_static, div_static};
use crate::utils::mul::{
    hi_mul_dyn, hi_mul_static, hi_sqr_dyn, hi_sqr_static, mul_dyn, mul_prim2, mul_static,
};
use crate::utils::sqrt::{sqrt_approx_dyn, sqrt_approx_static};
use crate::utils::utils::{add_buf, buf_len, inc_buf, shl_buf, shr_buf, split_sh, sub_buf};
use crate::utils::{
    CapacityError, ScratchGuard, DYN_EXP_CHUNK_MIN_BITS, DYN_EXP_CHUNK_PAR_WORK, DYN_EXP_SQR_BITS,
    DYN_LN_CHUNK_MIN_BITS, DYN_LN_CHUNK_PAR_WORK, DYN_LN_SQRT_A, STATIC_EXP_CHUNK_MIN_BITS,
    STATIC_EXP_CHUNK_PAR_WORK, STATIC_EXP_SQR_BITS, STATIC_LN_CHUNK_MIN_BITS,
    STATIC_LN_CHUNK_PAR_WORK, STATIC_LN_SQRT_A,
};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::f64::consts::{LOG2_E, PI};
use std::{fmt, panic, thread};

const GUARD_LIMBS: usize = 1;
const EXP_RED_GUARD_BITS: u8 = 58;
const MAX_PAR_CHUNKS: usize = 64;

// hi_mul, mul and div.
type BinOp = fn(&[u64], &[u64], &mut [u64]) -> u64;
// An exp entry's minimum first-chunk width, parallel work threshold and whether it's dynamic.
type ChunkPar = (u64, usize, bool);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpError {
    Overflow,
    Capacity(CapacityError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LnError {
    Zero,
    Capacity(CapacityError),
}

// From<CapacityError>, Display and Error for an error with one other variant.
macro_rules! capacity_error {
    ($t:ident, $v:ident, $msg:literal) => {
        impl From<CapacityError> for $t {
            fn from(e: CapacityError) -> Self {
                $t::Capacity(e)
            }
        }

        impl fmt::Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $t::$v => write!(f, $msg),
                    $t::Capacity(e) => e.fmt(f),
                }
            }
        }

        impl std::error::Error for $t {}
    };
}

capacity_error!(ExpError, Overflow, "e^X's exponent doesn't fit i64");
capacity_error!(LnError, Zero, "ln(0) is undefined");

fn capacity<const N: usize>(min_n: usize) -> Result<(), CapacityError> {
    (N >= min_n).then_some(()).ok_or(CapacityError { min_n })
}

pub struct Exp;
impl EngelSeries for Exp {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
}
impl HyperSeries for Exp {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
    fn r(_: u64) -> u64 {
        1
    }
}

// atanh(x) = x (1 + ATanh(x^2)), summed from n = 1.
pub struct ATanh;
impl BBPSeries for ATanh {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        2 * n + 1
    }
}
impl HyperSeries for ATanh {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        2 * n + 1
    }
    fn r(n: u64) -> u64 {
        2 * n + 1
    }
}

// The fewest n with fits(n), for a monotone fits that holds from ceil(p / s).
fn min_terms(p: u64, s: u64, fits: impl Fn(u64) -> bool) -> u64 {
    if fits(0) {
        return 0;
    }
    let (mut a, mut b) = (0, p.div_ceil(s));
    while b - a > 1 {
        let n = (a + b) / 2;
        if fits(n) {
            b = n;
        } else {
            a = n;
        }
    }
    b
}

// Terms of e^x, x <= 2^-s, past which n s + log2(n!) (Stirling) reaches 64w bits.
pub fn exp_terms(s: u64, w: usize) -> u64 {
    let c = (LOG2_E - PI.log2() - 1.0) * 0.5;
    let p = 64 * w as u64;
    min_terms(p, s, |n| {
        let nf = n as f64;
        n * s + ((nf + 0.5) * nf.log2() - nf * LOG2_E - c) as u64 >= p
    })
}

// Terms of ATanh(x), x < 2^-s, whose tail x^(n + 1) / ((2n + 3)(1 - x)) is below 2^-64w.
pub fn ln_terms(s: u64, w: usize) -> u64 {
    // 1 / (1 - x) <= 2
    let p = 64 * w as u64 + 1;
    min_terms(p, s, |n| (n + 1) * s + (2 * n + 3).ilog2() as u64 >= p)
}

fn exp_zero_limbs(t: u64) -> usize {
    (t.saturating_sub(1) / 64) as usize
}

// Bits of x, whose top limb is nonzero.
fn bits(x: &[u64]) -> u64 {
    x.last()
        .map_or(0, |&top| 64 * x.len() as u64 - top.leading_zeros() as u64)
}

// An entry's result is below 2^-lo with at most one more leading zero bit: z zero limbs,
// then v working limbs, one more when that bit can cross a limb boundary.
struct Frame {
    z: usize,
    v: usize,
    w: usize,
    extra: usize,
}

impl Frame {
    fn new(lo: u64, o: usize) -> Self {
        let extra = ((lo + 1) % 64 == 0) as usize;
        let (z, v) = ((lo / 64) as usize, o + GUARD_LIMBS + extra);
        Frame {
            z,
            v,
            w: z + v,
            extra,
        }
    }

    // out <- acc's top limbs, all ones after a carry only rounding causes. Returns the zero
    // limbs above them.
    fn out(&self, acc: &mut [u64], carry: bool, out: &mut [u64]) -> u64 {
        if carry {
            acc.fill(u64::MAX);
        }
        let d = (self.extra == 1 && acc[self.v - 1] == 0) as usize;
        out.copy_from_slice(&acc[self.v - d - out.len()..self.v - d]);
        (self.z + d) as u64
    }
}

// A chunk from bit s spans [s, ~2s), at least min bits the first time, and stops at end.
fn chunk_width(s: u64, end: u64, min: &mut u64) -> u64 {
    (s & !1)
        .max(2)
        .max(std::mem::take(min))
        .min(end.saturating_sub(s))
}

fn exp_chunks(mut s: u64, end: u64, mut min: u64) -> impl Iterator<Item = (u64, u64)> {
    std::iter::from_fn(move || {
        let width = chunk_width(s, end, &mut min);
        (width != 0).then(|| {
            s += width;
            (s - width, width)
        })
    })
}

// An entry's frame for x 2^-64(m + x.len()) at o limbs: x's last bit e, its top bit's offset
// s, and where its chunks end.
fn exp_frame(x: &[u64], m: u64, o: usize) -> (u64, u64, Frame, u64) {
    let e = 64 * (m + x.len() as u64);
    let s = 64 * m + x[x.len() - 1].leading_zeros() as u64;
    let f = Frame::new(s.saturating_sub(1), o);
    let end = (e - x[0].trailing_zeros() as u64).min(64 * f.w as u64);
    (e, s, f, end)
}

// Whether n chunks of v working limbs, n v >= work, go parallel: on rayon for the dynamic
// entries, on a scoped thread each for the static ones.
fn chunks_par(n: usize, v: usize, work: usize, rayon: bool) -> bool {
    n >= 2
        && n * v >= work
        && if rayon {
            rayon::current_num_threads() > 1
        } else {
            n <= MAX_PAR_CHUNKS && thread::available_parallelism().is_ok_and(|p| p.get() > 1)
        }
}

fn chunk_bits<'r>(
    x: &[u64],
    e: u64,
    s: u64,
    width: u64,
    out: &'r mut [u64],
) -> Option<(&'r [u64], u64, u64)> {
    let (l, sh) = split_sh(e - s - width);
    let n = width.div_ceil(64) as usize;
    let c = (n + 1).min(x.len() - l);
    out[..c].copy_from_slice(&x[l..l + c]);
    shr_buf(&mut out[..c], sh);
    out[n - 1] &= u64::MAX >> (64 * n as u64 - width);
    let skip = out[..n].iter().position(|&l| l != 0)?;
    let tz = out[skip].trailing_zeros();
    let r = &mut out[skip..n];
    shr_buf(r, tz as u8);
    let r = &r[..buf_len(r)];
    let k = s + width - 64 * skip as u64 - tz as u64;
    Some((r, k - bits(r), k))
}

// n terms of the series in r 2^-k into out, the low limbs of a w-limb frame: an Engel or BBP
// tree (E) when r = 1, otherwise a Hyper tree.
fn series<E, H>(
    r: &[u64],
    k: u64,
    n: u64,
    w: usize,
    out: &mut [u64],
    ctx: fn(&[u64], u64) -> H::Ctx,
) where
    E: Node<Ctx = Shift> + Send,
    H: Node + Send,
    H::Ctx: Sync,
{
    let zeros = w - out.len();
    if r == [1] {
        bin_split::<E>(&Shift(k), 1, n + 1).finalize(&Shift(k), zeros, out);
    } else {
        let ctx = ctx(r, k);
        bin_split::<H>(&ctx, 1, n + 1).finalize(&ctx, zeros, out);
    }
}

// f over items and their slots, on rayon when par.
fn for_each_slot<T: Sync>(
    par: bool,
    items: &[T],
    slots: &mut [&mut [u64]],
    f: impl Fn(&T, &mut [u64]) + Sync + Send,
) {
    if par {
        items.par_iter().zip(slots).for_each(|(t, s)| f(t, s));
    } else {
        items.iter().zip(slots).for_each(|(t, s)| f(t, s));
    }
}

fn exp_combine(a: &mut [u64], b: &[u64], z: usize, c: &mut [u64], hi_mul: BinOp) {
    let o = b.len().saturating_sub(z);
    let carry = o != 0 && {
        c[o] = hi_mul(a, b, &mut c[..o]);
        add_buf(a, &c[1..=o])
    };
    if add_buf(a, b) | carry {
        // Only rounding can reach 2^-64z.
        a.fill(u64::MAX);
    }
}

pub fn exp_dyn_entry(x: &[u64], m: u64, out: &mut [u64]) -> u64 {
    assert!(x.last().is_some_and(|&l| l != 0));

    let (e, s, f, end) = exp_frame(x, m, out.len());
    let (z, v, w) = (f.z, f.v, f.w);
    let slot = |width: u64| width.div_ceil(64) as usize + 1;
    let chunks = || exp_chunks(s + 1, end, DYN_EXP_CHUNK_MIN_BITS);
    let mut scratch = ScratchGuard::acquire();
    let [a, c, mut rs] = scratch.get_splits([v, v + 1, chunks().map(|(_, wd)| slot(wd)).sum()]);
    let mut parts = Vec::new();
    for (s, width) in chunks() {
        let (r, rest) = std::mem::take(&mut rs).split_at_mut(slot(width));
        rs = rest;
        parts.extend(chunk_bits(x, e, s, width, r));
    }

    let exp = |r: &[u64], t, k, out: &mut [u64]| {
        let n = exp_terms(t, w);
        series::<DynNodeEngel<Exp>, DynNodeHyper<Exp>>(r, k, n, w, out, DynHyperCtx::new)
    };
    let top = w - exp_zero_limbs(s + 1);
    a[top..].fill(0);
    exp(&[1], s + 1, s + 1, &mut a[..top]);
    let par = chunks_par(parts.len(), v, DYN_EXP_CHUNK_PAR_WORK, true);
    let mut results = ScratchGuard::acquire();
    let mut bs = results.get_split_vec(parts.iter().map(|&(_, t, _)| w - exp_zero_limbs(t)));
    for_each_slot(par, &parts, &mut bs, |&(r, t, k), b| exp(r, t, k, b));
    for b in &bs {
        exp_combine(a, b, z, c, hi_mul_dyn);
    }
    f.out(a, false, out)
}

// Fraction limbs for j squares. Each at most doubles the error below it and adds about a
// unit, so j + 4 bits past o keep their 2^(j + 2) units under a quarter of o's last limb.
fn exp_sqr(o: usize, j: u64) -> usize {
    o + (j > 0) as usize * (j + 4).div_ceil(64) as usize
}

// xr holds o + 3 limbs, l2 and t at least o + 2 and o + 1. The squares run only when l2 also
// holds fs + 1 and t fs + 3 (the static entry's bound), and when the entry would sum r's
// chunks serially: in parallel the series they shorten overlap anyway.
fn exp_reduced(
    x: &[u64],
    m: i64,
    out: &mut [u64],
    [xr, l2, t]: [&mut [u64]; 3],
    sqr_bits: u64,
    ln2: fn(&mut [u64]),
    div_rem: fn(&mut [u64], &[u64], &mut [u64]) -> u64,
    exp_entry: fn(&[u64], u64, &mut [u64]) -> u64,
    (min, work, rayon): ChunkPar,
    hi_sqr: fn(&[u64], &mut [u64]) -> u64,
) -> Result<u64, ExpError> {
    let (len, o) = (buf_len(x), out.len());
    if len == 0 {
        out.fill(0);
        if let Some(top) = out.last_mut() {
            *top = 1;
        }
        return Ok(0);
    }
    let me = m as i128 - (x.len() - len) as i128;
    if me > 1 || (me == 1 && x[len - 1] > 22) {
        return Err(ExpError::Overflow);
    }
    let (f, g) = (o + 1, EXP_RED_GUARD_BITS);
    xr.fill(0);
    let lo = f as i128 + me + 1 - len as i128;
    let skip = (-lo).clamp(0, len as i128) as usize;
    let start = lo.max(0) as usize;
    xr[start..start + len - skip].copy_from_slice(&x[skip..len]);
    let carry = shl_buf(xr, g);
    debug_assert_eq!(carry, 0, "X < 2^69");
    let ln2b = &mut l2[..o + 2];
    ln2(ln2b);
    shr_buf(ln2b, 64 - g);
    let mut q = [0; 3];
    let xl = buf_len(xr);
    div_rem(&mut xr[..xl], ln2b, &mut q);
    if q[2] != 0 || q[1] >= 32 {
        // n >= 2^69, so n / 64 > i64::MAX
        return Err(ExpError::Overflow);
    }
    let (k, b) = ((q[1] << 58) | (q[0] >> 6), q[0] & 63);
    shr_buf(&mut xr[..=f], g);

    let rl = buf_len(&xr[..f]);
    t[..=o].fill(0);
    if rl != 0 {
        let rz = f - rl;
        let lead = 64 * rz as u64 + xr[rl - 1].leading_zeros() as u64;
        let mut j = sqr_bits.saturating_sub(lead);
        let mut fs = exp_sqr(o, j);
        // Whether the entry would sum r's chunks in parallel, counting any without bits.
        let parallel = || {
            let (_, s, fr, end) = exp_frame(&xr[..rl], rz as u64, o);
            chunks_par(exp_chunks(s + 1, end, min).count(), fr.v, work, rayon)
        };
        if j > 0 && (fs + 1 > l2.len() || fs + 3 > t.len() || parallel()) {
            (j, fs) = (0, o);
        }
        // e^r = (1 + u)^(2^j), u = e^(r 2^-j) - 1 at fs fraction limbs. r 2^-j is exact: r
        // moves up 64 - jb bits into xr[f], under jl more zero limbs.
        let (jl, jb) = split_sh(j);
        let top = f + (jb != 0) as usize;
        let c = shl_buf(&mut xr[..top], (64 - jb) % 64);
        debug_assert_eq!(c, 0);
        let xl = buf_len(&xr[..top]);
        let z = exp_zero_limbs(lead + j);
        if let Some(p) = fs.checked_sub(z).filter(|&p| p != 0) {
            let off = exp_entry(&xr[..xl], (top - xl + jl) as u64, &mut t[..p]) as usize - z;
            debug_assert!(off <= 1);
            t.copy_within(off..p, 0);
            t[p - off..fs].fill(0);
            let (u, c) = (&mut t[..fs], &mut l2[..fs + 1]);
            for _ in 0..j {
                // u <- (1 + u)^2 - 1 = 2u + u^2, with u^2's limbs from fs up in c[1..=sl].
                // u < 1/2 before the last square, and only rounding carries that past 1.
                let ul = buf_len(u);
                let sl = (2 * ul).saturating_sub(fs);
                if sl != 0 {
                    c[sl] = hi_sqr(&u[..ul], &mut c[..sl]);
                }
                let carry = shl_buf(u, 1);
                debug_assert_eq!(carry, 0);
                if sl != 0 && add_buf(u, &c[1..=sl]) {
                    u.fill(u64::MAX);
                }
            }
            t.copy_within(fs - o..fs, 0);
        }
    }
    let t = &mut t[..=o];
    t[o] = 1;
    let carry = shl_buf(t, b as u8);
    debug_assert_eq!(carry, 0, "2^b e^r < 2^64");
    out.copy_from_slice(&t[1..]);
    Ok(k)
}

pub fn exp_dyn(x: &[u64], m: i64, out: &mut [u64]) -> Result<u64, ExpError> {
    exp_dyn_at(x, m, out, DYN_EXP_SQR_BITS)
}

// exp_dyn with an explicit sqr_bits, for tests and tuning.
#[doc(hidden)]
pub fn exp_dyn_at(x: &[u64], m: i64, out: &mut [u64], sqr_bits: u64) -> Result<u64, ExpError> {
    let o = out.len();
    let fs = exp_sqr(o, sqr_bits);
    let mut scratch = ScratchGuard::acquire();
    let bufs = scratch.get_splits([o + 3, (o + 2).max(fs + 1), (o + 1).max(fs + 3)]);
    exp_reduced(
        x,
        m,
        out,
        bufs,
        sqr_bits,
        ln2_dyn,
        div_rem_dyn,
        exp_dyn_entry,
        (DYN_EXP_CHUNK_MIN_BITS, DYN_EXP_CHUNK_PAR_WORK, true),
        hi_sqr_dyn,
    )
}

// e^(r 2^-k) - 1, r 2^-k <= 2^-t, into out, the low limbs of a w-limb frame.
fn exp_series_static<const N: usize>(r: &[u64], t: u64, k: u64, w: usize, out: &mut [u64]) {
    let n = exp_terms(t, w);
    series::<StaticNodeEngel<N, Exp>, StaticNodeHyper<N, Exp>>(r, k, n, w, out, StaticHyperCtx::new)
}

#[inline(never)]
fn exp_chunk_static<const N: usize>(
    x: &[u64],
    e: u64,
    s: u64,
    width: u64,
    w: usize,
) -> Option<([u64; N], usize)> {
    let mut r = [0; N];
    let (r, t, k) = chunk_bits(x, e, s, width, &mut r)?;
    let (mut b, bl) = ([0; N], w - exp_zero_limbs(t));
    exp_series_static::<N>(r, t, k, w, &mut b[..bl]);
    Some((b, bl))
}

// Each series needs one guard limb past out, and static finalize 2 more.
pub fn exp_static_entry<const N: usize>(
    x: &[u64],
    m: u64,
    out: &mut [u64],
) -> Result<u64, CapacityError> {
    capacity::<N>(out.len() + 3)?;
    Ok(exp_static_core::<N>(x, m, out))
}

fn exp_static_core<const N: usize>(x: &[u64], m: u64, out: &mut [u64]) -> u64 {
    assert!(x.last().is_some_and(|&l| l != 0));

    let (e, s, f, end) = exp_frame(x, m, out.len());
    let (z, v, w) = (f.z, f.v, f.w);
    let (mut a, mut c) = ([0; N], [0; N]);
    let a = &mut a[..v];
    let top = w - exp_zero_limbs(s + 1);
    let top_series = |a: &mut [u64]| exp_series_static::<N>(&[1], s + 1, s + 1, w, &mut a[..top]);
    let chunks = || exp_chunks(s + 1, end, STATIC_EXP_CHUNK_MIN_BITS);
    let chunk = |(s, width)| exp_chunk_static::<N>(x, e, s, width, w);
    if chunks_par(chunks().count(), v, STATIC_EXP_CHUNK_PAR_WORK, false) {
        thread::scope(|sc| {
            let mut hs: [_; MAX_PAR_CHUNKS] = std::array::from_fn(|_| None);
            for (h, ch) in hs.iter_mut().zip(chunks()) {
                let builder = thread::Builder::new().stack_size(static_split_stack(N));
                *h = Some((ch, builder.spawn_scoped(sc, move || chunk(ch)).ok()));
            }
            top_series(a);
            for (ch, h) in hs.into_iter().flatten() {
                let b = match h {
                    Some(h) => h.join().unwrap_or_else(|e| panic::resume_unwind(e)),
                    None => chunk(ch),
                };
                if let Some((b, bl)) = b {
                    exp_combine(a, &b[..bl], z, &mut c, hi_mul_static::<N>);
                }
            }
        });
    } else {
        top_series(a);
        for (b, bl) in chunks().filter_map(chunk) {
            exp_combine(a, &b[..bl], z, &mut c, hi_mul_static::<N>);
        }
    }
    f.out(a, false, out)
}

// The squares run only when N leaves them room.
pub fn exp_static<const N: usize>(x: &[u64], m: i64, out: &mut [u64]) -> Result<u64, ExpError> {
    exp_static_at::<N>(x, m, out, STATIC_EXP_SQR_BITS)
}

#[doc(hidden)]
pub fn exp_static_at<const N: usize>(
    x: &[u64],
    m: i64,
    out: &mut [u64],
    sqr_bits: u64,
) -> Result<u64, ExpError> {
    let o = out.len();
    capacity::<N>(o + 3 + (o + 2 > LN2_LIMBS) as usize)?;
    let (mut xr, mut l2, mut t) = ([0; N], [0; N], [0; N]);
    exp_reduced(
        x,
        m,
        out,
        [&mut xr[..o + 3], &mut l2, &mut t],
        sqr_bits,
        |l2| ln2_static_unguarded::<N>(l2).expect("exp_static checks N"),
        div_rem_static::<N>,
        exp_static_core::<N>,
        (STATIC_EXP_CHUNK_MIN_BITS, STATIC_EXP_CHUNK_PAR_WORK, false),
        hi_sqr_static::<N>,
    )
}

// Whether x's bits from its top one down are at least y's.
fn mant_ge(x: &[u64], y: &[u64]) -> bool {
    // floor(v 2^-lo) mod 2^64
    let at = |v: &[u64], lo: i64| {
        let get = |i: usize| v.get(i).copied().unwrap_or(0);
        match split_sh(lo.max(0) as u64) {
            _ if lo <= -64 => 0,
            _ if lo < 0 => get(0) << -lo,
            (l, 0) => get(l),
            (l, b) => get(l) >> b | get(l + 1) << (64 - b),
        }
    };
    let (bx, by) = (bits(x) as i64, bits(y) as i64);
    for j in 1..=x.len().max(y.len()) as i64 + 1 {
        let (a, b) = (at(x, bx - 64 * j), at(y, by - 64 * j));
        if a != b {
            return a > b;
        }
    }
    true
}

// out <- a b 2^-sh, its bits below 1 dropped, for a result below B^out.len(). hi_mul only
// drops partial products, so this never rounds up. Returns the result's length.
fn mul_shr(a: &[u64], b: &[u64], sh: u64, out: &mut [u64], hi_mul: BinOp) -> usize {
    out.fill(0);
    let (sl, sb) = split_sh(sh);
    let full = a.len() + b.len();
    if a.is_empty() || b.is_empty() || full <= sl {
        return 0;
    }
    // The product's limbs from sl up; one past out holds only bits the shift moves down.
    let (r, o) = (full - sl, out.len());
    debug_assert!(r <= o + 1, "a b 2^-sh must fit out");
    let top = hi_mul(a, b, &mut out[..r - 1]);
    if r <= o {
        out[r - 1] = top;
        shr_buf(&mut out[..r], sb);
    } else {
        debug_assert_eq!(top.checked_shr(sb as u32).unwrap_or(0), 0);
        shr_buf(out, sb);
        if sb != 0 {
            out[o - 1] |= top << (64 - sb);
        }
    }
    buf_len(&out[..r.min(o)])
}

// δ = |u - 1| as fraction limbs below f.len(). For u >= 1 these are f's bits under u's
// leading one (bit 64 - sh of its top limb) moved up by sh, or f itself when sh = 0. For
// u < 1, δ = !f + 1, whose carry stops at f's lowest nonzero limb l0.
struct Delta<'a> {
    f: &'a [u64],
    sh: u32,
    l0: Option<usize>,
}

impl<'a> Delta<'a> {
    fn new(f: &'a [u64], sh: u32, below: bool) -> Self {
        let l0 = below.then(|| f.iter().position(|&l| l != 0).unwrap_or(f.len()));
        Delta { f, sh, l0 }
    }

    fn limb(&self, i: i64) -> u64 {
        let raw = |i: i64| match usize::try_from(i).ok().and_then(|i| self.f.get(i)) {
            Some(&l) if self.sh != 0 && i as usize == self.f.len() - 1 => {
                l & !(1 << (64 - self.sh))
            }
            Some(&l) => l,
            None => 0,
        };
        if let Some(l0) = self.l0 {
            return match usize::try_from(i).map(|i| i.cmp(&l0)) {
                Ok(Ordering::Greater) => !raw(i),
                Ok(Ordering::Equal) => raw(i).wrapping_neg(),
                _ => 0,
            };
        }
        match self.sh {
            0 => raw(i),
            64 => raw(i - 1),
            s => raw(i) << s | raw(i - 1) >> (64 - s),
        }
    }

    // δ's top nonzero limb and leading zero bits, or None when δ = 0.
    fn top(&self) -> Option<(usize, u64)> {
        let len = self.f.len();
        let it = (0..len).rev().find(|&i| self.limb(i as i64) != 0)?;
        let lead = 64 * (len - 1 - it) as u64 + self.limb(it as i64).leading_zeros() as u64;
        Some((it, lead))
    }
}

// |ln(1 ± δ)| < 2^-lo for δ = x 2^-64(m + x.len()): ln(1 + δ) is in (δ / 2, δ) and
// -ln(1 - δ) in (δ, 2δ) for δ <= 1/2. Chunks stop at end, where 2 atanh(a) - 2a < a^3 is
// below the frame.
fn ln_frame(x: &[u64], m: u64, neg: bool, o: usize) -> (Frame, u64, u64) {
    let sd = 64 * m + x[x.len() - 1].leading_zeros() as u64;
    let lo = if neg { sd.saturating_sub(1) } else { sd };
    let f = Frame::new(lo, o);
    let end = (64 * f.w as u64).div_ceil(3) + 2;
    (f, lo, end)
}

// n = δ / 4 at 2^-64w and d = (2 ± δ) / 4 at 2^-64v, so n / d = δ / (2 ± δ). d rounds up,
// which keeps n / d at most that and its top bit at lo + 2 or below.
fn ln_init(x: &[u64], m: u64, neg: bool, w: usize, nb: &mut [u64], db: &mut [u64], t: &mut [u64]) {
    let v = db.len();
    let e = (64 * (m + x.len() as u64) + 2) as i64;
    shifted(x, 64 * w as i64 - e, nb);
    shifted(x, 64 * v as i64 - e, t);
    db.fill(0);
    db[v - 1] = 1 << 63;
    let c = if neg {
        sub_buf(db, t)
    } else {
        add_buf(db, t) | inc_buf(db)
    };
    debug_assert!(!c);
}

// A <= z 2^k for z = n / d in [2^-s, 2^(1 - s)) and k = s + wd - 1: floor(z 2^k) less a unit
// or so, from n's top limbs over d's rounded up, 64 bits past k when the buffers allow, and
// never below z's top bit. Returns A's length in a.
fn chunk_quotient(
    (n, d): (&[u64], &[u64]),
    zb: u64,
    (k, wd): (u64, u64),
    [qn, qd, qq]: [&mut [u64]; 3],
    a: &mut [u64],
    div: BinOp,
) -> usize {
    if wd == 1 {
        a[0] = 1;
        return 1;
    }
    let cap = qn.len();
    let ql = (wd + 64).div_ceil(64) as usize + 1;
    let t = (ql + 1).min(cap.saturating_sub(ql + 1)).max(1);
    let (cn, cd) = (n.len().saturating_sub(t), d.len().saturating_sub(t));
    let mut dl = d.len() - cd;
    qd[..dl].copy_from_slice(&d[cd..]);
    if cd > 0 && inc_buf(&mut qd[..dl]) {
        qd[dl] = 1;
        dl += 1;
    }
    // q = floor(n' 2^c / d') <= z 2^kq, n's limbs counting from 2^-64w and d's from 2^-64v.
    let mut kq = k + 64;
    let mut c = kq as i64 - zb as i64 + 64 * (cn as i64 - cd as i64);
    let nt = &n[cn..];
    let limbs = ((bits(nt) as i64 + c).max(0) as u64).div_ceil(64) as usize;
    if limbs > cap {
        let cut = 64 * (limbs - cap) as u64;
        (c, kq) = (c - cut as i64, kq - cut);
    }
    let qn = &mut qn[..limbs.min(cap)];
    shifted(nt, c, qn);
    let nl = buf_len(qn);
    let aw = wd.div_ceil(64) as usize;
    let a = &mut a[..aw];
    if nl >= dl {
        let q = &mut qq[..nl - dl + 1];
        div(&qn[..nl], &qd[..dl], q);
        shifted(q, k as i64 - kq as i64, a);
    } else {
        a.fill(0);
    }
    if bits(&a[..buf_len(a)]) < wd {
        a.fill(0);
        a[((wd - 1) / 64) as usize] = 1 << ((wd - 1) % 64);
    }
    aw
}

// Removes chunks a = A 2^-k <= z, A odd, from z = n / d (n 2^-64w in nb, d 2^-64v in db),
// passing each to chunk with 2a in the w-limb frame, until z's top bit is past end and the
// first chunk's. atanh(z) = atanh(a) + atanh((z - a) / (1 - z a)), and that remainder is
// (n - a d) / (d - a n), so only each chunk's own bits need a (short) division. Returns the
// remainder's top bit, or u64::MAX once n is 0.
fn ln_chain(
    (nb, db): (&mut [u64], &mut [u64]),
    z: usize,
    end: u64,
    mut min: u64,
    [t1, t2, qn, qd, qq]: [&mut [u64]; 5],
    ab: &mut [u64],
    (hi_mul, div): (BinOp, BinOp),
    mut chunk: impl FnMut(&[u64], u64, &[u64]),
) -> u64 {
    let (zb, w) = (64 * z as u64, z + db.len());
    let mut s0 = None;
    loop {
        let (l1, l2) = {
            let (n, d) = (&nb[..buf_len(nb)], &db[..buf_len(db)]);
            if n.is_empty() {
                return u64::MAX;
            }
            // z in [2^-s, 2^(1 - s))
            let s = bits(d) + zb + !mant_ge(n, d) as u64 - bits(n);
            let wd = match s0 {
                None => {
                    s0 = Some(s);
                    1
                }
                Some(t) if s >= end.max(t + 1) => return s,
                Some(_) => chunk_width(s, end, &mut min).max(1),
            };
            let mut k = s + wd - 1;
            let al = chunk_quotient((n, d), zb, (k, wd), [&mut *qn, &mut *qd, &mut *qq], ab, div);
            let tl = ab[..al].iter().position(|&l| l != 0).unwrap();
            let tb = ab[tl].trailing_zeros() as u8;
            shr_buf(&mut ab[tl..al], tb);
            let a = &ab[tl..tl + buf_len(&ab[tl..al])];
            k -= 64 * tl as u64 + tb as u64;
            shifted(a, 64 * w as i64 + 1 - k as i64, t1);
            chunk(a, k, &t1[..buf_len(t1)]);
            // a n in d's units and a d in n's, both from the old n and d.
            let l2 = mul_shr(a, n, k + zb, t2, hi_mul);
            let l1 = mul_shr(a, d, k - zb, t1, hi_mul);
            (l1, l2)
        };
        let (bn, bd) = (sub_buf(nb, &t1[..l1]), sub_buf(db, &t2[..l2]));
        debug_assert!(!bn && !bd, "a <= n / d keeps n and d positive");
    }
}

// Where chunk a = A 2^-k's 2 a S, S = ATanh(a^2) < 2^-(2 lead + 1) for a < 2^-lead, lands in
// the w-limb frame: S's n terms at wi limbs below its top `zeros`, then A S shifted down by
// sh >= bits(A), so S's last-limb error costs under a unit. None when it's below the frame.
#[derive(Debug, Clone, Copy)]
struct LnChunk {
    n: u64,
    wi: usize,
    zeros: usize,
    sh: u64,
}

fn ln_chunk(a: &[u64], k: u64, w: usize) -> Option<LnChunk> {
    let lead = k - bits(a);
    let wi = w - ((lead - 1) / 64) as usize;
    let zeros = ((2 * lead + 1) / 64) as usize;
    let n = ln_terms(2 * lead, wi);
    (zeros < wi && n != 0).then(|| LnChunk {
        n,
        wi,
        zeros,
        sh: 64 * wi as u64 + k - 1 - 64 * w as u64,
    })
}

// 2 a S into t (v + 1 limbs), through sq and s. Returns its length.
fn ln_chunk_sum<B, H>(
    a: &[u64],
    k: u64,
    c: LnChunk,
    [sq, s, t]: [&mut [u64]; 3],
    ctx: fn(&[u64], u64) -> H::Ctx,
    (mul, hi_mul): (BinOp, BinOp),
) -> usize
where
    B: Node<Ctx = Shift> + Send,
    H: Node + Send,
    H::Ctx: Sync,
{
    let la = a.len();
    sq[2 * la - 1] = mul(a, a, &mut sq[..2 * la - 1]);
    let a2 = &sq[..buf_len(&sq[..2 * la])];
    let s = &mut s[..c.wi - c.zeros];
    series::<B, H>(a2, 2 * k, c.n, c.wi, s, ctx);
    mul_shr(a, &s[..buf_len(s)], c.sh, t, hi_mul)
}

// The tail 2 n / d < 2^-(sl - 2) into out, skipping its top (sl - 2) / 64 >= z limbs of the
// frame but keeping 2, as static finalize expects. Returns its length.
fn ln_tail(
    n: &[u64],
    d: &[u64],
    sl: u64,
    (w, v): (usize, usize),
    out: &mut [u64],
    ratio: fn(&[u64], &[u64], i64, &mut [u64]),
) -> usize {
    if sl == u64::MAX {
        return 0;
    }
    let tl = match w.saturating_sub(((sl - 2) / 64) as usize) {
        0 => return 0,
        l => l.max(2).min(w),
    };
    let (n, d) = (&n[..buf_len(n)], &d[..buf_len(d)]);
    ratio(n, d, 64 * (tl as i64 - v as i64) - 1, &mut out[..tl]);
    buf_len(&out[..tl])
}

pub fn ln_dyn_entry(x: &[u64], m: u64, neg: bool, out: &mut [u64]) -> u64 {
    assert!(x.last().is_some_and(|&l| l != 0));

    let (f, _, end) = ln_frame(x, m, neg, out.len());
    let (z, v, w) = (f.z, f.v, f.w);
    let cq = 2 * ((end + 64).div_ceil(64) as usize + 1) + 3;
    let al = end.div_ceil(64) as usize + 1;
    let mut scratch = ScratchGuard::acquire();
    let [nb, db, acc, t1, t2, qn, qd, qq, ab] =
        scratch.get_splits([v, v, v, v + 1, v + 1, cq, cq, cq, al]);
    ln_init(x, m, neg, w, nb, db, &mut t1[..v]);
    acc.fill(0);

    let (mut pool, mut chunks, mut carry) = (Vec::new(), Vec::new(), false);
    let bufs = [&mut *t1, t2, qn, qd, qq];
    let ops = (hi_mul_dyn as BinOp, div_dyn as BinOp);
    let min = DYN_LN_CHUNK_MIN_BITS;
    let sl = ln_chain((nb, db), z, end, min, bufs, ab, ops, |a, k, a2| {
        carry |= add_buf(acc, a2);
        if let Some(c) = ln_chunk(a, k, w) {
            chunks.push((pool.len(), a.len(), k, c));
            pool.extend_from_slice(a);
        }
    });

    let par = chunks_par(chunks.len(), v, DYN_LN_CHUNK_PAR_WORK, true);
    let mut results = ScratchGuard::acquire();
    let mut bs = results.get_split_vec(chunks.iter().map(|_| v + 1));
    let (nb, db) = (&*nb, &*db);
    // The tail reuses t1, which the chain is done with.
    let (_, tl) = join2(
        par,
        || {
            for_each_slot(par, &chunks, &mut bs, |&(i, l, k, c), b| {
                let mut sc = ScratchGuard::acquire();
                let [sq, s] = sc.get_splits([2 * l, c.wi - c.zeros]);
                ln_chunk_sum::<DynNodeBBP<ATanh>, DynNodeHyper<ATanh>>(
                    &pool[i..i + l],
                    k,
                    c,
                    [sq, s, b],
                    DynHyperCtx::new,
                    (mul_dyn, hi_mul_dyn),
                );
            })
        },
        || ln_tail(nb, db, sl, (w, v), &mut *t1, ratio_to_fraction_dyn),
    );
    for b in &bs {
        carry |= add_buf(acc, &b[..buf_len(b)]);
    }
    carry |= add_buf(acc, &t1[..tl]);
    f.out(acc, carry, out)
}

#[inline(never)]
fn ln_chunk_static<const N: usize>(a: &[u64], k: u64, c: LnChunk, v: usize) -> ([u64; N], usize) {
    let (mut sq, mut s, mut t) = ([0; N], [0; N], [0; N]);
    let l = ln_chunk_sum::<StaticNodeBBP<N, ATanh>, StaticNodeHyper<N, ATanh>>(
        a,
        k,
        c,
        [&mut sq, &mut s, &mut t[..v + 1]],
        StaticHyperCtx::new,
        (mul_static::<N>, hi_mul_static::<N>),
    );
    (t, l)
}

// Each series needs one guard limb past out, and static finalize 2 more.
pub fn ln_static_entry<const N: usize>(
    x: &[u64],
    m: u64,
    neg: bool,
    out: &mut [u64],
) -> Result<u64, CapacityError> {
    capacity::<N>(out.len() + 3)?;
    Ok(ln_static_core::<N>(x, m, neg, out))
}

fn ln_static_core<const N: usize>(x: &[u64], m: u64, neg: bool, out: &mut [u64]) -> u64 {
    assert!(x.last().is_some_and(|&l| l != 0));

    let (f, lo, end) = ln_frame(x, m, neg, out.len());
    let (z, v, w) = (f.z, f.v, f.w);
    let (mut nb, mut db, mut acc, mut t1, mut t2) = ([0; N], [0; N], [0; N], [0; N], [0; N]);
    let (mut qn, mut qd, mut qq, mut ab) = ([0; N], [0; N], [0; N], [0; N]);
    let (nb, db, acc) = (&mut nb[..v], &mut db[..v], &mut acc[..v]);
    let (t1, t2) = (&mut t1[..v + 1], &mut t2[..v + 1]);
    ln_init(x, m, neg, w, nb, db, &mut t1[..v]);

    // Chunks start at lo + 3 at the earliest, after the top one.
    let chunks = exp_chunks(lo + 3, end, STATIC_LN_CHUNK_MIN_BITS).count() + 1;
    let mut carry = false;
    let bufs = [t1, t2, &mut qn, &mut qd, &mut qq];
    let ops = (hi_mul_static::<N> as BinOp, div_static::<N> as BinOp);
    let min = STATIC_LN_CHUNK_MIN_BITS;
    // The tail reuses ab, which the chain is done with.
    let tail = |nb: &[u64], db: &[u64], sl, ab: &mut [u64]| {
        ln_tail(nb, db, sl, (w, v), ab, ratio_to_fraction_static::<N>)
    };
    let tl = if chunks_par(chunks, v, STATIC_LN_CHUNK_PAR_WORK, false) {
        thread::scope(|sc| {
            let mut hs: [_; MAX_PAR_CHUNKS] = std::array::from_fn(|_| None);
            let (mut nh, mut top) = (0, None);
            let sl = ln_chain((nb, db), z, end, min, bufs, &mut ab, ops, |a, k, a2| {
                carry |= add_buf(acc, a2);
                let Some(c) = ln_chunk(a, k, w) else {
                    return;
                };
                let (mut ac, al) = ([0; N], a.len());
                ac[..al].copy_from_slice(a);
                let job = move || ln_chunk_static::<N>(&ac[..al], k, c, v);
                // The caller's thread sums the top chunk once the chain is done.
                if top.is_none() {
                    top = Some(job);
                    return;
                }
                if nh < MAX_PAR_CHUNKS {
                    let builder = thread::Builder::new().stack_size(static_split_stack(N));
                    if let Ok(h) = builder.spawn_scoped(sc, job) {
                        hs[nh] = Some(h);
                        nh += 1;
                        return;
                    }
                }
                let (b, bl) = job();
                carry |= add_buf(acc, &b[..bl]);
            });
            if let Some(job) = top {
                let (b, bl) = job();
                carry |= add_buf(acc, &b[..bl]);
            }
            let tl = tail(nb, db, sl, &mut ab);
            for h in hs.into_iter().flatten() {
                let (b, bl) = h.join().unwrap_or_else(|e| panic::resume_unwind(e));
                carry |= add_buf(acc, &b[..bl]);
            }
            tl
        })
    } else {
        let sl = ln_chain((nb, db), z, end, min, bufs, &mut ab, ops, |a, k, a2| {
            carry |= add_buf(acc, a2);
            if let Some(c) = ln_chunk(a, k, w) {
                let (b, bl) = ln_chunk_static::<N>(a, k, c, v);
                carry |= add_buf(acc, &b[..bl]);
            }
        });
        tail(nb, db, sl, &mut ab)
    };
    carry |= add_buf(acc, &ab[..tl]);
    f.out(acc, carry, out)
}

// j = floor(A log2(64 o)) square roots, less the lead bits u already sits near 1, and the
// fraction limbs they keep: the entry's window below δ_y's top limb when e = 0, or the
// absolute frame and j's shift otherwise.
fn ln_sqrt(o: usize, sqrt_a: f64, e0: bool, lead: u64) -> (u64, usize) {
    let j = ((sqrt_a * (64.0 * o as f64).log2()).max(0.0) as u64).saturating_sub(lead);
    let fl = if e0 {
        ((lead + j) / 64) as usize + o + 5
    } else {
        o + 3 + j.div_ceil(64) as usize
    };
    (j, fl)
}

// ys, sq, ent and r hold at least o + 4, o + 3, o + 2 and o + 4 limbs. The square roots run
// only when ys and sq also hold ln_sqrt's fl + 1 and + 2, and ent o + 2 + ceil(j / 64).
fn ln_reduced(
    x: &[u64],
    m: i64,
    out: &mut [u64],
    [ys, sq, ent, r]: [&mut [u64]; 4],
    sqrt_a: f64,
    ln2: fn(&mut [u64]),
    sqrt: fn(&mut [u64], &mut [u64]),
    entry: fn(&[u64], u64, bool, &mut [u64]) -> u64,
) -> Result<(i64, bool), LnError> {
    let (len, o) = (buf_len(x), out.len());
    if len == 0 {
        return Err(LnError::Zero);
    }
    if o == 0 {
        return Ok((0, false));
    }
    let me = m as i128 - (x.len() - len) as i128;
    let x = &x[..len];
    let lz = x[len - 1].leading_zeros();
    // u = X 2^-e in [1, 2), except that X in [1/2, 1) stays u = X: -ln 2 + ln 2X would
    // cancel |ln X|'s leading bits.
    let e = 64 * me + 63 - lz as i128;
    let below = e == -1;
    let e = if below { 0 } else { e };
    let f = o + 1;
    let mut j = 0;
    // L = |ln y| = ent 2^-64(zl + pe), y = u^(1 / 2^j), unless it's 0 or below the frame.
    let sh = if below { 0 } else { lz + 1 };
    let l = 'l: {
        let src = Delta::new(x, sh, below);
        let Some((it, lead)) = src.top() else {
            break 'l None;
        };
        let fl;
        (j, fl) = ln_sqrt(o, sqrt_a, e == 0, lead);
        if fl + 1 > ys.len() || fl + 2 > sq.len() || o + 2 + j.div_ceil(64) as usize > ent.len() {
            j = 0;
        }
        let (src, (it, lead), win) = if j > 0 {
            let yb = &mut ys[..fl + 1];
            shifted(x, 64 * (fl as i64 - len as i64) + sh as i64, yb);
            for _ in 0..j {
                // sqrt(4y B^(2fl + 1)) = sqrt(y) B^fl 2^33, and 4y's top limb is nonzero.
                let rb = &mut sq[..fl + 2];
                rb[0] = 0;
                rb[1..].copy_from_slice(yb);
                let c = shl_buf(&mut rb[1..], 2);
                debug_assert_eq!(c, 0);
                sqrt(rb, yb);
                shr_buf(yb, 33);
            }
            let src = Delta::new(&ys[..fl], 0, below);
            let Some(top) = src.top() else {
                break 'l None;
            };
            (src, top, &mut sq[..])
        } else {
            (src, (it, lead), &mut ys[..])
        };
        // L < 2^-lead
        let pe = if e == 0 {
            o + 1
        } else {
            (f + j.div_ceil(64) as usize).saturating_sub((lead / 64) as usize)
        };
        if pe == 0 {
            break 'l None;
        }
        let win = &mut win[..pe + 3];
        let base = it as i64 + 1 - win.len() as i64;
        for (i, l) in win.iter_mut().enumerate() {
            *l = src.limb(base + i as i64);
        }
        let my = (src.f.len() - 1 - it) as u64;
        Some((pe, entry(win, my, below, &mut ent[..pe]) as usize))
    };

    if e == 0 {
        let Some((pe, zl)) = l else {
            // X = 1
            out.fill(0);
            return Ok((0, false));
        };
        // |ln X| = 2^j L
        let (jw, jb) = split_sh(j);
        ent[pe] = shl_buf(&mut ent[..pe], jb);
        let t = if ent[pe] != 0 { pe } else { pe - 1 };
        out.copy_from_slice(&ent[t + 1 - o..t + 1]);
        return Ok(((t + jw) as i64 - (zl + pe) as i64, below));
    }
    // |ln X| = |e ln 2 ± 2^j L| at f fraction limbs. ln 2 at f + 1 limbs keeps |e| < 2^70
    // times its truncation within r's last limb, and e < 0 is at most -2, so |e| ln 2 >= 2 ln 2
    // covers 2^j L < ln 2.
    let r = &mut r[..f + 3];
    ln2(&mut r[..f + 1]);
    let c = mul_prim2(&mut r[..f + 1], e.unsigned_abs());
    (r[f + 1], r[f + 2]) = (c as u64, (c >> 64) as u64);
    let rr = &mut r[1..];
    if let Some((pe, zl)) = l {
        let t = &mut sq[..f + 2];
        shifted(&ent[..pe], j as i64 + 64 * (f as i64 - (zl + pe) as i64), t);
        let t = &t[..buf_len(t)];
        let c = if e > 0 {
            add_buf(rr, t)
        } else {
            sub_buf(rr, t)
        };
        debug_assert!(!c);
    }
    let t = buf_len(rr) - 1;
    out.copy_from_slice(&rr[t + 1 - o..t + 1]);
    Ok((t as i64 - f as i64, e < 0))
}

pub fn ln_dyn(x: &[u64], m: i64, out: &mut [u64]) -> Result<(i64, bool), LnError> {
    ln_dyn_at(x, m, out, DYN_LN_SQRT_A)
}

// ln_dyn with an explicit sqrt_a, for tests and tuning.
#[doc(hidden)]
pub fn ln_dyn_at(x: &[u64], m: i64, out: &mut [u64], sqrt_a: f64) -> Result<(i64, bool), LnError> {
    let o = out.len();
    let (t, f0) = ln_sqrt(o, sqrt_a, true, 0);
    let fl = f0.max(ln_sqrt(o, sqrt_a, false, 0).1);
    let mut scratch = ScratchGuard::acquire();
    let bufs = scratch.get_splits([
        (o + 4).max(fl + 1),
        (o + 3).max(fl + 2),
        o + 2 + t.div_ceil(64) as usize,
        o + 4,
    ]);
    ln_reduced(
        x,
        m,
        out,
        bufs,
        sqrt_a,
        ln2_dyn,
        sqrt_approx_dyn,
        ln_dyn_entry,
    )
}

// The entry at o + 1 limbs needs o + 4, as does ln(2) at o + 2 past its table. The square
// roots run only when N leaves them room.
pub fn ln_static<const N: usize>(
    x: &[u64],
    m: i64,
    out: &mut [u64],
) -> Result<(i64, bool), LnError> {
    ln_static_at::<N>(x, m, out, STATIC_LN_SQRT_A)
}

#[doc(hidden)]
pub fn ln_static_at<const N: usize>(
    x: &[u64],
    m: i64,
    out: &mut [u64],
    sqrt_a: f64,
) -> Result<(i64, bool), LnError> {
    capacity::<N>(out.len() + 4)?;
    let (mut ys, mut sq, mut ent, mut r) = ([0; N], [0; N], [0; N], [0; N]);
    ln_reduced(
        x,
        m,
        out,
        [&mut ys, &mut sq, &mut ent, &mut r],
        sqrt_a,
        |l2| ln2_static_unguarded::<N>(l2).expect("ln_static checks N"),
        sqrt_approx_static::<N>,
        ln_static_core::<N>,
    )
}
