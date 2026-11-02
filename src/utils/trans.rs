use crate::utils::consts::consts::{
    ln2_dyn, ln2_static_unguarded, shifted, static_jobs, LN2_LIMBS,
};
use crate::utils::div::{div_dyn, div_rem_dyn, div_rem_static, div_static};
use crate::utils::mul::{
    hi_mul_dyn, hi_mul_static, hi_sqr_dyn, hi_sqr_static, mul_dyn, mul_prim2, mul_static, sqr_dyn,
    sqr_static,
};
use crate::utils::series::{
    bits, join2, ratio_to_fraction_dyn, ratio_to_fraction_static, series_eval_dyn,
    series_eval_static, static_split_stack, BBPSeries, Bbp, Engel, EngelSeries, Hyper, HyperSeries,
    Series,
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
use std::f64::consts::{LOG2_E, PI};
use std::{fmt, panic, thread};

const GUARD_LIMBS: usize = 1;
const EXP_RED_GUARD_BITS: u8 = 58;
const MAX_PAR_CHUNKS: usize = 64;

// mul, hi_mul and div; sqr and hi_sqr; sqrt (radicand as scratch, root).
type BinOp = fn(&[u64], &[u64], &mut [u64]) -> u64;
type UnOp = fn(&[u64], &mut [u64]) -> u64;
type SqrtOp = fn(&mut [u64], &mut [u64]);

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
    const START: u64 = 1;
    // e^y - 1 < 2^(1 - s) for y <= 2^-s; from n >= 2 the tail is at most y^(n - 1) / (n - 1)!,
    // with Stirling's log2 less half a bit.
    fn tail_bits(n: u64, s: i64) -> u64 {
        let s = s.max(0) as u64;
        match n.checked_sub(1) {
            None | Some(0) => s.saturating_sub(1),
            Some(m) => {
                let c = (LOG2_E - PI.log2() - 1.0) * 0.5;
                let mf = m as f64;
                let log2_fact = ((mf + 0.5) * mf.log2() - mf * LOG2_E - c) as u64;
                m.saturating_mul(s).saturating_add(log2_fact)
            }
        }
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
}
impl HyperSeries for Exp {
    const START: u64 = 1;
    fn tail_bits(n: u64, s: i64) -> u64 {
        <Exp as EngelSeries>::tail_bits(n, s)
    }
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
    const START: u64 = 1;
    // Tail y^n / ((2n + 1)(1 - y)) <= 2 y^n / (2n + 1) for y <= 2^-s <= 1/2.
    fn tail_bits(n: u64, s: i64) -> u64 {
        let s = s.max(0) as u64;
        n.saturating_mul(s)
            .saturating_add((2 * n.min(u64::MAX / 4) + 1).ilog2() as u64)
            .saturating_sub(1)
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        2 * n + 1
    }
}
impl HyperSeries for ATanh {
    const START: u64 = 1;
    fn tail_bits(n: u64, s: i64) -> u64 {
        <ATanh as BBPSeries>::tail_bits(n, s)
    }
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

// floor(x 2^-lo) mod 2^64.
fn window(x: &[u64], lo: i64) -> u64 {
    let get = |i: i64| {
        usize::try_from(i)
            .ok()
            .and_then(|i| x.get(i))
            .map_or(0, |&l| l as u128)
    };
    let l = lo.div_euclid(64);
    ((get(l + 1) << 64 | get(l)) >> lo.rem_euclid(64)) as u64
}

// An entry's result below 2^-lo, with at most one more leading zero bit: z zero limbs, then v
// working limbs, one extra when that bit can cross a limb boundary.
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

    // out <- acc's top limbs, all ones after a rounding carry. Returns the zero limbs above.
    fn out(&self, acc: &[u64], carry: bool, out: &mut [u64]) -> u64 {
        if carry {
            out.fill(u64::MAX);
            return self.z as u64;
        }
        let d = (self.extra == 1 && acc[self.v - 1] == 0) as usize;
        out.copy_from_slice(&acc[self.v - d - out.len()..self.v - d]);
        (self.z + d) as u64
    }
}

// From bit s, a chunk spans [s, ~2s), at least min bits the first time, and stops at end.
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

// For x 2^-64(m + x.len()) at o limbs: x's last bit e, its top bit's offset s, the frame, and
// where its chunks end.
fn exp_frame(x: &[u64], m: u64, o: usize) -> (u64, u64, Frame, u64) {
    assert!(x.last().is_some_and(|&l| l != 0));
    let e = 64 * (m + x.len() as u64);
    let s = 64 * m + x[x.len() - 1].leading_zeros() as u64;
    let f = Frame::new(s.saturating_sub(1), o);
    let end = (e - x[0].trailing_zeros() as u64).min(64 * f.w as u64);
    (e, s, f, end)
}

// Whether n chunks of v limbs go parallel: on rayon, or on a scoped thread each.
fn chunks_par(n: usize, v: usize, work: usize, rayon: bool) -> bool {
    n >= 2
        && n * v >= work
        && if rayon {
            rayon::current_num_threads() > 1
        } else {
            n <= MAX_PAR_CHUNKS && thread::available_parallelism().is_ok_and(|p| p.get() > 1)
        }
}

// x's chunk [s, s + width) into out as an odd r with its bit offset k, or None when it's 0.
fn chunk_bits<'r>(
    x: &[u64],
    e: u64,
    s: u64,
    width: u64,
    out: &'r mut [u64],
) -> Option<(&'r [u64], u64)> {
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
    Some((r, s + width - 64 * skip as u64 - tz as u64))
}

// series_eval_dyn or series_eval_static.
trait Eval {
    fn eval<C: Series>(ctx: &C, out: &mut [u64]);

    // ctx's series in a w-limb frame, below its zero limbs. Returns its limbs.
    fn frame<C: Series>(ctx: &C, w: usize, out: &mut [u64]) -> usize {
        let l = w - ctx.zeros();
        Self::eval(ctx, &mut out[..l]);
        l
    }
}

struct DynEval;

impl Eval for DynEval {
    fn eval<C: Series>(ctx: &C, out: &mut [u64]) {
        series_eval_dyn(ctx, out);
    }
}

struct StaticEval<const N: usize>;

impl<const N: usize> Eval for StaticEval<N> {
    fn eval<C: Series>(ctx: &C, out: &mut [u64]) {
        series_eval_static::<N, C>(ctx, out)
            .expect("the static entries' bounds leave every series room");
    }
}

const EXP_CHUNK: &str = "a chunk's bits lie below 2^-1";

// e^(r 2^-k) - 1 for an odd r, r 2^-k < 1/2, in a w-limb frame: Engel when r = 1, else Hyper.
fn exp_series<V: Eval>(r: &[u64], k: u64, w: usize, out: &mut [u64]) -> usize {
    match r {
        [1] => V::frame(&Engel::<Exp>::new(k), w, out),
        _ => V::frame(&Hyper::<Exp>::new(r, k).expect(EXP_CHUNK), w, out),
    }
}

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

// a <- (1 + a)(1 + b) - 1, b below z zero limbs of a's frame.
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

    let par = chunks_par(parts.len(), v, DYN_EXP_CHUNK_PAR_WORK, true);
    let zeros = |r: &[u64], k| match r {
        [1] => Engel::<Exp>::new(k).zeros(),
        _ => Hyper::<Exp>::new(r, k).expect(EXP_CHUNK).zeros(),
    };
    let mut results = ScratchGuard::acquire();
    let mut bs = results.get_split_vec(parts.iter().map(|&(r, k)| w - zeros(r, k)));
    // The top bit's series beside the chunks'.
    let (top, _) = join2(
        par,
        || exp_series::<DynEval>(&[1], s + 1, w, a),
        || {
            for_each_slot(par, &parts, &mut bs, |&(r, k), b| {
                exp_series::<DynEval>(r, k, w, b);
            })
        },
    );
    a[top..].fill(0);
    for b in &bs {
        exp_combine(a, b, z, c, hi_mul_dyn);
    }
    f.out(a, false, out)
}

// Fraction limbs for j squares: each at most doubles the error below it and adds about a unit,
// so j + 4 bits past o keep the total under a quarter of o's last limb.
fn exp_sqr(o: usize, j: u64) -> usize {
    o + (j > 0) as usize * (j + 4).div_ceil(64) as usize
}

// X = n ln 2 + r by one division, then e^r = (1 + u)^(2^j) with u = e^(r 2^-j) - 1 from the
// entry. xr holds o + 3 limbs, l2 and t at least o + 2 and o + 1. The squares run only when l2
// also holds fs + 1 and t fs + 3, and the entry would sum r's chunks serially.
fn exp_reduced(
    x: &[u64],
    m: i64,
    out: &mut [u64],
    [xr, l2, t]: [&mut [u64]; 3],
    sqr_bits: u64,
    ln2: fn(&mut [u64]),
    div_rem: fn(&mut [u64], &[u64], &mut [u64]) -> u64,
    exp_entry: fn(&[u64], u64, &mut [u64]) -> u64,
    (min, work, rayon): (u64, usize, bool),
    hi_sqr: UnOp,
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
    // floor(X B^f) 2^g over floor(ln 2 B^(o + 2) 2^(g - 64)): guard bits, not a guard limb,
    // cover n < 2^69 times ln 2's truncation.
    let (f, g) = (o + 1, EXP_RED_GUARD_BITS);
    let lo = (f as i128 + me + 1 - len as i128).max(-(len as i128));
    shifted(x, 64 * lo as i64, xr);
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
        let parallel = || {
            let (_, s, fr, end) = exp_frame(&xr[..rl], rz as u64, o);
            chunks_par(exp_chunks(s + 1, end, min).count(), fr.v, work, rayon)
        };
        if j > 0 && (fs + 1 > l2.len() || fs + 3 > t.len() || parallel()) {
            (j, fs) = (0, o);
        }
        // r 2^-j exactly: r moves up 64 - jb bits into xr[f], under jl more zero limbs.
        let (jl, jb) = split_sh(j);
        let top = f + (jb != 0) as usize;
        let c = shl_buf(&mut xr[..top], (64 - jb) % 64);
        debug_assert_eq!(c, 0);
        let xl = buf_len(&xr[..top]);
        let z = ((lead + j).saturating_sub(1) / 64) as usize;
        if let Some(p) = fs.checked_sub(z).filter(|&p| p != 0) {
            let off = exp_entry(&xr[..xl], (top - xl + jl) as u64, &mut t[..p]) as usize - z;
            debug_assert!(off <= 1);
            t.copy_within(off..p, 0);
            t[p - off..fs].fill(0);
            let (u, c) = (&mut t[..fs], &mut l2[..fs + 1]);
            for _ in 0..j {
                // u <- 2u + u^2, u^2's limbs from fs up in c[1..=sl]. Only rounding carries
                // the last square past 1.
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

#[inline(never)]
fn exp_chunk_static<const N: usize>(
    x: &[u64],
    e: u64,
    s: u64,
    width: u64,
    w: usize,
) -> Option<([u64; N], usize)> {
    let mut r = [0; N];
    let (r, k) = chunk_bits(x, e, s, width, &mut r)?;
    let mut b = [0; N];
    let bl = exp_series::<StaticEval<N>>(r, k, w, &mut b);
    Some((b, bl))
}

// Each series needs a guard limb past out, and static finalize 2 more.
pub fn exp_static_entry<const N: usize>(
    x: &[u64],
    m: u64,
    out: &mut [u64],
) -> Result<u64, CapacityError> {
    capacity::<N>(out.len() + 3)?;
    let (e, s, f, end) = exp_frame(x, m, out.len());
    let (z, v, w) = (f.z, f.v, f.w);
    let (mut a, mut c) = ([0; N], [0; N]);
    let a = &mut a[..v];
    let chunks = || exp_chunks(s + 1, end, STATIC_EXP_CHUNK_MIN_BITS);
    let chunk = |(s, width)| exp_chunk_static::<N>(x, e, s, width, w);
    let par = chunks_par(chunks().count(), v, STATIC_EXP_CHUNK_PAR_WORK, false);
    // The caller sums the top bit's series while the chunks run.
    static_jobs::<N, MAX_PAR_CHUNKS, _, _, _>(par, chunks().map(|ch| move || chunk(ch)), |bs| {
        exp_series::<StaticEval<N>>(&[1], s + 1, w, a);
        for (b, bl) in bs.flatten() {
            exp_combine(a, &b[..bl], z, &mut c, hi_mul_static::<N>);
        }
    });
    Ok(f.out(a, false, out))
}

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
        |x, m, t| exp_static_entry::<N>(x, m, t).expect("exp_static checks N"),
        (STATIC_EXP_CHUNK_MIN_BITS, STATIC_EXP_CHUNK_PAR_WORK, false),
        hi_sqr_static::<N>,
    )
}

// out <- floor(a b 2^-sh) < B^out.len(), never rounded up. Returns its length.
fn mul_shr(a: &[u64], b: &[u64], sh: u64, out: &mut [u64], hi_mul: BinOp) -> usize {
    let (sl, sb) = split_sh(sh);
    let (full, o) = (a.len() + b.len(), out.len());
    if a.is_empty() || b.is_empty() || full <= sl {
        out.fill(0);
        return 0;
    }
    // The product's limbs from sl up; one past out holds only bits the shift moves down.
    let r = full - sl;
    debug_assert!(r <= o + 1, "a b 2^-sh must fit out");
    let top = hi_mul(a, b, &mut out[..r - 1]);
    if r <= o {
        out[r - 1] = top;
        out[r..].fill(0);
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

// δ = |u - 1| as fraction limbs below f.len(): f's bits under u's leading one (moved up sh) for
// u >= 1, or !f + 1 for u < 1, whose carry stops at f's lowest nonzero limb l0.
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

    // Limb i < f.len(). Below l0, f's limbs (and so their negation) are 0.
    fn limb(&self, i: i64) -> u64 {
        let l = window(self.f, 64 * i - self.sh as i64);
        match self.l0 {
            Some(l0) if i > l0 as i64 => !l,
            Some(_) => l.wrapping_neg(),
            None => l,
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

// |ln(1 ± δ)| < 2^-lo for δ = x 2^-64(m + x.len()), as ln(1 + δ) is in (δ / 2, δ) and
// -ln(1 - δ) in (δ, 2δ). Chunks end where atanh's cubic term is below the frame.
fn ln_frame(x: &[u64], m: u64, neg: bool, o: usize) -> (Frame, u64, u64) {
    assert!(x.last().is_some_and(|&l| l != 0));
    let sd = 64 * m + x[x.len() - 1].leading_zeros() as u64;
    let lo = if neg { sd.saturating_sub(1) } else { sd };
    let f = Frame::new(lo, o);
    let end = (64 * f.w as u64).div_ceil(3) + 2;
    (f, lo, end)
}

// n = δ / 4 at 2^-64w and d = (2 ± δ) / 4 at 2^-64v, d rounded up, so n / d <= δ / (2 ± δ).
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

// A <= z 2^k for z = n / d in [2^-s, 2^(1 - s)), k = s + wd - 1: floor(z 2^k) less about a
// unit, from n's top limbs over d's rounded up, and never below z's top bit. Returns A, in qn.
fn chunk_quotient<'q>(
    (n, d): (&[u64], &[u64]),
    zb: u64,
    (k, wd): (u64, u64),
    qn: &'q mut [u64],
    [qd, qq]: [&mut [u64]; 2],
    div: BinOp,
) -> &'q mut [u64] {
    if wd == 1 {
        qn[0] = 1;
        return &mut qn[..1];
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
    // q = floor(n' 2^c / d') <= z 2^kq.
    let mut kq = k + 64;
    let mut c = kq as i64 - zb as i64 + 64 * (cn as i64 - cd as i64);
    let nt = &n[cn..];
    let limbs = ((bits(nt) as i64 + c).max(0) as u64).div_ceil(64) as usize;
    if limbs > cap {
        let cut = 64 * (limbs - cap) as u64;
        (c, kq) = (c - cut as i64, kq - cut);
    }
    let nq = &mut qn[..limbs.min(cap)];
    shifted(nt, c, nq);
    let nl = buf_len(nq);
    let aw = wd.div_ceil(64) as usize;
    if nl >= dl {
        let q = &mut qq[..nl - dl + 1];
        div(&qn[..nl], &qd[..dl], q);
        shifted(q, k as i64 - kq as i64, &mut qn[..aw]);
    } else {
        qn[..aw].fill(0);
    }
    let a = &mut qn[..aw];
    if bits(&a[..buf_len(a)]) < wd {
        a.fill(0);
        a[((wd - 1) / 64) as usize] = 1 << ((wd - 1) % 64);
    }
    a
}

// Takes chunks a = A 2^-k <= z, A odd, off z = n / d (n 2^-64w in nb, d 2^-64v in db), passing
// each to chunk with 2a in the w-limb frame, until z's top bit passes end and the first chunk's.
// atanh(z) = atanh(a) + atanh((n - a d) / (d - a n)), so only each chunk's bits need a (short)
// division. qd and qq hold at least v + 1 limbs. Returns the remainder's top bit, or u64::MAX
// once n is 0.
fn ln_chain(
    (nb, db): (&mut [u64], &mut [u64]),
    z: usize,
    end: u64,
    mut min: u64,
    [qn, qd, qq]: [&mut [u64]; 3],
    (hi_mul, div): (BinOp, BinOp),
    mut chunk: impl FnMut(&[u64], u64, &[u64]),
) -> u64 {
    let (zb, v) = (64 * z as u64, db.len());
    let w = z + v;
    let mut s0 = None;
    loop {
        let (l1, l2) = {
            let (n, d) = (&nb[..buf_len(nb)], &db[..buf_len(db)]);
            if n.is_empty() {
                return u64::MAX;
            }
            // z in [2^-s, 2^(1 - s)), from whether n's bits from its top one down reach d's.
            let (bn, bd) = (bits(n), bits(d));
            let ge = (1..=n.len().max(d.len()) as i64 + 1)
                .map(|j| (window(n, bn as i64 - 64 * j), window(d, bd as i64 - 64 * j)))
                .find(|(a, b)| a != b)
                .map_or(true, |(a, b)| a > b);
            let s = bd + zb + !ge as u64 - bn;
            let wd = match s0 {
                None => {
                    s0 = Some(s);
                    1
                }
                Some(t) if s >= end.max(t + 1) => return s,
                Some(_) => chunk_width(s, end, &mut min).max(1),
            };
            let mut k = s + wd - 1;
            let ab = chunk_quotient((n, d), zb, (k, wd), qn, [qd, qq], div);
            let tl = ab.iter().position(|&l| l != 0).unwrap();
            let tb = ab[tl].trailing_zeros() as u8;
            shr_buf(&mut ab[tl..], tb);
            let a = &ab[tl..tl + buf_len(&ab[tl..])];
            k -= 64 * tl as u64 + tb as u64;
            // A is in qn, so qq and qd are free.
            let (t1, t2) = (&mut qq[..v + 1], &mut qd[..v + 1]);
            shifted(a, 64 * w as i64 + 1 - k as i64, t1);
            chunk(a, k, &t1[..buf_len(t1)]);
            // a n in d's units and a d in n's, both from the old n and d.
            let l2 = mul_shr(a, n, k + zb, t2, hi_mul);
            let l1 = mul_shr(a, d, k - zb, t1, hi_mul);
            (l1, l2)
        };
        let (bn, bd) = (sub_buf(nb, &qq[..l1]), sub_buf(db, &qd[..l2]));
        debug_assert!(!bn && !bd, "a <= n / d keeps n and d positive");
    }
}

// Where chunk a = A 2^-k's 2 a ATanh(a^2) lands in the w-limb frame, as (wi, zeros, sh): the
// series at wi limbs below at least `zeros` zero limbs, then A times it shifted down sh >= bits(A),
// so its last-limb error costs under a unit. None when it's below the frame.
type LnChunk = (usize, usize, u64);

fn ln_chunk(a: &[u64], k: u64, w: usize) -> Option<LnChunk> {
    let lead = k - bits(a);
    let wi = w - ((lead - 1) / 64) as usize;
    let zeros = ((2 * lead + 1) / 64) as usize;
    (zeros < wi).then(|| (wi, zeros, 64 * wi as u64 + k - 1 - 64 * w as u64))
}

// 2 a ATanh(a^2) into t (v + 1 limbs): BBP when A = 1, else Hyper in A^2. Returns its length.
fn ln_chunk_sum<V: Eval>(
    a: &[u64],
    k: u64,
    (wi, _, sh): LnChunk,
    [sq, s, t]: [&mut [u64]; 3],
    (sqr, hi_mul): (UnOp, BinOp),
) -> usize {
    let la = a.len();
    sq[2 * la - 1] = sqr(a, &mut sq[..2 * la - 1]);
    let a2 = &sq[..buf_len(&sq[..2 * la])];
    let sl = if a2 == [1] {
        V::frame(&Bbp::<ATanh>::new(2 * k), wi, s)
    } else {
        let ctx = Hyper::<ATanh>::new(a2, 2 * k).expect("a chunk is below 1/2");
        V::frame(&ctx, wi, s)
    };
    mul_shr(a, &s[..buf_len(&s[..sl])], sh, t, hi_mul)
}

// The tail 2 n / d < 2^-(sl - 2) into out (0 once n is 0, sl = u64::MAX), skipping its top
// (sl - 2) / 64 >= z limbs of the frame but keeping 2, as static finalize expects. Returns its
// length.
fn ln_tail(
    n: &[u64],
    d: &[u64],
    sl: u64,
    (w, v): (usize, usize),
    out: &mut [u64],
    ratio: fn(&[u64], &[u64], i64, &mut [u64]),
) -> usize {
    let tl = match w.saturating_sub(((sl - 2) / 64) as usize) {
        0 => return 0,
        l => l.max(2).min(w),
    };
    let (n, d) = (&n[..buf_len(n)], &d[..buf_len(d)]);
    ratio(n, d, 64 * (tl as i64 - v as i64) - 1, &mut out[..tl]);
    buf_len(&out[..tl])
}

pub fn ln_dyn_entry(x: &[u64], m: u64, neg: bool, out: &mut [u64]) -> u64 {
    let (f, _, end) = ln_frame(x, m, neg, out.len());
    let (z, v, w) = (f.z, f.v, f.w);
    let cq = 2 * ((end + 64).div_ceil(64) as usize + 1) + 3;
    let mut scratch = ScratchGuard::acquire();
    let [nb, db, acc, qn, qd, qq] = scratch.get_splits([v, v, v, cq, cq.max(v + 1), cq.max(v + 1)]);
    ln_init(x, m, neg, w, nb, db, &mut qq[..v]);
    acc.fill(0);

    let (mut pool, mut chunks, mut carry) = (Vec::new(), Vec::new(), false);
    let bufs = [qn, qd, &mut *qq];
    let ops = (hi_mul_dyn as BinOp, div_dyn as BinOp);
    let min = DYN_LN_CHUNK_MIN_BITS;
    let sl = ln_chain((nb, db), z, end, min, bufs, ops, |a, k, a2| {
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
    // The tail reuses qq, which the chain is done with.
    let (_, tl) = join2(
        par,
        || {
            for_each_slot(par, &chunks, &mut bs, |&(i, l, k, c), b| {
                let mut sc = ScratchGuard::acquire();
                let [sq, s] = sc.get_splits([2 * l, c.0 - c.1]);
                ln_chunk_sum::<DynEval>(&pool[i..i + l], k, c, [sq, s, b], (sqr_dyn, hi_mul_dyn));
            })
        },
        || ln_tail(nb, db, sl, (w, v), &mut *qq, ratio_to_fraction_dyn),
    );
    for b in &bs {
        carry |= add_buf(acc, &b[..buf_len(b)]);
    }
    carry |= add_buf(acc, &qq[..tl]);
    f.out(acc, carry, out)
}

#[inline(never)]
fn ln_chunk_static<const N: usize>(a: &[u64], k: u64, c: LnChunk, v: usize) -> ([u64; N], usize) {
    let (mut sq, mut s, mut t) = ([0; N], [0; N], [0; N]);
    let l = ln_chunk_sum::<StaticEval<N>>(
        a,
        k,
        c,
        [&mut sq, &mut s, &mut t[..v + 1]],
        (sqr_static::<N>, hi_mul_static::<N>),
    );
    (t, l)
}

// Each series needs a guard limb past out, and static finalize 2 more.
pub fn ln_static_entry<const N: usize>(
    x: &[u64],
    m: u64,
    neg: bool,
    out: &mut [u64],
) -> Result<u64, CapacityError> {
    capacity::<N>(out.len() + 3)?;
    let (f, lo, end) = ln_frame(x, m, neg, out.len());
    let (z, v, w) = (f.z, f.v, f.w);
    let (mut nb, mut db, mut acc) = ([0; N], [0; N], [0; N]);
    let (mut qn, mut qd, mut qq) = ([0; N], [0; N], [0; N]);
    let (nb, db, acc) = (&mut nb[..v], &mut db[..v], &mut acc[..v]);
    ln_init(x, m, neg, w, nb, db, &mut qq[..v]);

    // Chunks start at lo + 3 at the earliest, after the top one.
    let chunks = exp_chunks(lo + 3, end, STATIC_LN_CHUNK_MIN_BITS).count() + 1;
    let ops = (hi_mul_static::<N> as BinOp, div_static::<N> as BinOp);
    // The chain, then the tail into qq, which the chain is done with.
    let mut chain = |chunk: &mut dyn FnMut(&[u64], u64, &[u64])| {
        let bufs = [&mut qn[..], &mut qd[..], &mut qq[..]];
        let min = STATIC_LN_CHUNK_MIN_BITS;
        let sl = ln_chain((&mut *nb, &mut *db), z, end, min, bufs, ops, chunk);
        ln_tail(nb, db, sl, (w, v), &mut qq, ratio_to_fraction_static::<N>)
    };
    let mut carry = false;
    let tl = if chunks_par(chunks, v, STATIC_LN_CHUNK_PAR_WORK, false) {
        thread::scope(|sc| {
            let mut hs: [_; MAX_PAR_CHUNKS] = std::array::from_fn(|_| None);
            let (mut nh, mut top) = (0, None);
            let tl = chain(&mut |a, k, a2| {
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
            for h in hs.into_iter().flatten() {
                let (b, bl) = h.join().unwrap_or_else(|e| panic::resume_unwind(e));
                carry |= add_buf(acc, &b[..bl]);
            }
            tl
        })
    } else {
        chain(&mut |a, k, a2| {
            carry |= add_buf(acc, a2);
            if let Some(c) = ln_chunk(a, k, w) {
                let (b, bl) = ln_chunk_static::<N>(a, k, c, v);
                carry |= add_buf(acc, &b[..bl]);
            }
        })
    };
    carry |= add_buf(acc, &qq[..tl]);
    Ok(f.out(acc, carry, out))
}

// j = floor(A log2(64 o)) square roots, less u's lead bits near 1, and the fraction limbs they
// keep: the entry's window below δ_y's top limb when e = 0, or the absolute frame and j's shift.
fn ln_sqrt(o: usize, sqrt_a: f64, e0: bool, lead: u64) -> (u64, usize) {
    let j = ((sqrt_a * (64.0 * o as f64).log2()).max(0.0) as u64).saturating_sub(lead);
    let fl = if e0 {
        ((lead + j) / 64) as usize + o + 5
    } else {
        o + 3 + j.div_ceil(64) as usize
    };
    (j, fl)
}

// |ln X| = |e ln 2 ± 2^j L|, L = |ln y| from the entry, y = (X 2^-e)^(1 / 2^j). ys, sq and ent
// hold at least o + 4, o + 3 and o + 2 limbs; the square roots run only when sq also holds fl + 2,
// and then ys and ent must hold fl + 1 and o + 2 + ceil(j / 64).
fn ln_reduced(
    x: &[u64],
    m: i64,
    out: &mut [u64],
    [ys, sq, ent]: [&mut [u64]; 3],
    sqrt_a: f64,
    ln2: fn(&mut [u64]),
    sqrt: SqrtOp,
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
    // u = X 2^-e in [1, 2), except X in [1/2, 1) stays u = X, as -ln 2 + ln 2X would cancel.
    let e = 64 * me + 63 - lz as i128;
    let below = e == -1;
    let e = if below { 0 } else { e };
    let f = o + 1;
    let mut j = 0;
    // (pe, zl) with L = ent 2^-64(zl + pe), or None when L is 0 or below the frame.
    let sh = if below { 0 } else { lz + 1 };
    let l = 'l: {
        let src = Delta::new(x, sh, below);
        let Some((it, lead)) = src.top() else {
            break 'l None;
        };
        let fl;
        (j, fl) = ln_sqrt(o, sqrt_a, e == 0, lead);
        if fl + 2 > sq.len() {
            j = 0;
        }
        let (src, (it, lead), win) = if j > 0 {
            let yb = &mut ys[..fl + 1];
            shifted(x, 64 * (fl as i64 - len as i64) + sh as i64, yb);
            for _ in 0..j {
                // sqrt(4y B^(2fl + 1)) = sqrt(y) B^fl 2^33, with 4y's top limb nonzero.
                let rb = &mut sq[..fl + 2];
                shifted(yb, 66, rb);
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
    // At f fraction limbs, ln 2 at f + 1 keeps |e| < 2^70 times its truncation within the last
    // limb, and e < 0 is at most -2, so |e| ln 2 covers 2^j L < ln 2.
    let r = &mut ys[..f + 3];
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

#[doc(hidden)]
pub fn ln_dyn_at(x: &[u64], m: i64, out: &mut [u64], sqrt_a: f64) -> Result<(i64, bool), LnError> {
    let o = out.len();
    // ln_sqrt's widest frame.
    let (t, fl) = ln_sqrt(o, sqrt_a, true, 0);
    let mut scratch = ScratchGuard::acquire();
    let bufs = scratch.get_splits([
        (o + 4).max(fl + 1),
        (o + 3).max(fl + 2),
        o + 2 + t.div_ceil(64) as usize,
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

// The entry at o + 1 limbs needs o + 4, as does ln 2 at o + 2 past its table.
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
    let (mut ys, mut sq, mut ent) = ([0; N], [0; N], [0; N]);
    ln_reduced(
        x,
        m,
        out,
        [&mut ys, &mut sq, &mut ent],
        sqrt_a,
        |l2| ln2_static_unguarded::<N>(l2).expect("ln_static checks N"),
        sqrt_approx_static::<N>,
        |x, m, neg, out| ln_static_entry::<N>(x, m, neg, out).expect("ln_static checks N"),
    )
}

// θ2(q⁴) = 2 Σ_{k>=0} q^((2k+1)²) and θ3(q⁴) = 1 + 2 Σ_{k>=1} q^((2k)²), for
// ln(1/q) = π / (4 AGM(θ2², θ3²)) (docs/agm-ln-theta.md). A_n = q^(n²) = A_(n-1) B_n with
// B_n = B_(n-1) q², odd n to θ2 and even n to θ3, each step as wide as later terms need.

// For q = u 2^-l, u in [1/2, 1): odd terms go to θ2 2^(64 z) with g2 limbs below 2^-64z (t = z),
// even ones to θ3 - 1 with g3 limbs below 1 (t = 0), each with a guard limb. last is the last
// power summed.
struct ThetaSchedule {
    l: u64,
    z: usize,
    g2: usize,
    g3: usize,
    last: u64,
}

impl ThetaSchedule {
    fn new(x: &[u64], shift: u64, terms: usize, w2: usize, w3: usize) -> Self {
        assert!(x.last().is_some_and(|&l| l != 0));
        assert!(w2 > 0 && w3 > 0, "theta's outputs must be non-empty");
        let l = shift
            .checked_sub(bits(x))
            .filter(|&l| l >= 2)
            .expect("theta needs q = x 2^-shift < 1/4");
        let mut s = ThetaSchedule {
            l,
            z: ((l - 2) / 64) as usize,
            g2: w2 + 1,
            g3: w3,
            last: 1,
        };
        // A parity's terms only shrink, so once neither n nor n + 1 lands, nothing later does.
        let max = (terms as u64).saturating_mul(2).saturating_add(1);
        for n in 2..=max {
            if s.ell(n).is_some() {
                s.last = n;
            } else if s.ell(n + 1).is_none() {
                break;
            }
        }
        s
    }

    // Term n's frame (t, g).
    fn frame(&self, n: u64) -> (usize, usize) {
        if n % 2 == 1 {
            (self.z, self.g2)
        } else {
            (0, self.g3)
        }
    }

    // Term n's window, from its top limb in the frame to a guard limb past it, or None when it's
    // below the frame.
    fn ell(&self, n: u64) -> Option<usize> {
        let (t, g) = self.frame(n);
        let zn = (self.l as u128 * (n as u128 * n as u128) - 1 - 64 * t as u128) / 64;
        (zn < g as u128).then(|| g - zn as usize + 1)
    }

    // Step n's width, the wider of n's and n + 1's windows (a parity's only narrow).
    fn width(&self, n: u64) -> usize {
        let w = |n| self.ell(n).filter(|_| n <= self.last).unwrap_or(0);
        w(n).max(w(n + 1))
    }

    // acc += floor(2 q^(n²) 2^(64 (t + g))), for q^(n²) = m 2^(-64 m.len() - e) 2^(-l n²).
    fn place(&self, n: u64, m: &[u64], e: u64, acc: &mut [u64], tmp: &mut [u64]) {
        let (t, g) = self.frame(n);
        let sh = 64 * m.len() as u128 + e as u128 + self.l as u128 * (n as u128 * n as u128)
            - 1
            - 64 * (t + g) as u128;
        debug_assert!(sh >= 64);
        let tmp = &mut tmp[..m.len()];
        shifted(m, -(sh as i64), tmp);
        // θ3 - 1 < 2^-7 and θ2 2^(64 z) <= 1.
        let carry = add_buf(acc, &tmp[..buf_len(tmp)]);
        debug_assert!(!carry);
    }
}

// m <- m 2^s after setting its top limb to top, s that limb's leading zeros. Returns s.
fn theta_norm(m: &mut [u64], top: u64) -> u64 {
    let r = m.len();
    m[r - 1] = top;
    let s = top.leading_zeros();
    debug_assert!(s < 64);
    shl_buf(m, s as u8);
    s as u64
}

// a, b and f hold width(1) limbs, q2 width(2), acc2 g2 and acc3 g3. Mantissas stay normalized,
// m 2^(-64 m.len()) in [1/2, 1), with a bit exponent beside them.
fn theta_core(
    s: &ThetaSchedule,
    x: &[u64],
    th2: &mut [u64],
    th3: &mut [u64],
    [mut a, mut b, mut f, q2, acc2, acc3]: [&mut [u64]; 6],
    hi_mul: BinOp,
    hi_sqr: UnOp,
) -> usize {
    let (acc2, acc3) = (&mut acc2[..s.g2], &mut acc3[..s.g3]);
    acc2.fill(0);
    acc3.fill(0);
    let r1 = s.width(1);
    shifted(x, 64 * r1 as i64 - bits(x) as i64, &mut a[..r1]);
    s.place(1, &a[..r1], 0, acc2, f);
    if s.last >= 2 {
        let r2 = s.width(2);
        let q2 = &mut q2[..r2];
        let top = hi_sqr(&a[r1 - r2..r1], &mut q2[..r2 - 1]);
        let e2 = theta_norm(q2, top);
        // A_1 = B_1 = u; each product goes to the free buffer, which then swaps in.
        b[..r1].copy_from_slice(&a[..r1]);
        let (mut ea, mut eb, mut r) = (0, 0, r1);
        for n in 2..=s.last {
            let rn = s.width(n);
            let top = hi_mul(&b[r - rn..r], &q2[r2 - rn..], &mut f[..rn - 1]);
            eb += e2 + theta_norm(&mut f[..rn], top);
            std::mem::swap(&mut b, &mut f);
            let top = hi_mul(&a[r - rn..r], &b[..rn], &mut f[..rn - 1]);
            ea += eb + theta_norm(&mut f[..rn], top);
            std::mem::swap(&mut a, &mut f);
            r = rn;
            if s.ell(n).is_some() {
                let acc = if n % 2 == 1 { &mut *acc2 } else { &mut *acc3 };
                s.place(n, &a[..r], ea, acc, f);
            }
        }
    }
    let w3 = th3.len();
    th3[..w3 - 1].copy_from_slice(&acc3[1..]);
    th3[w3 - 1] = 1;
    th2.copy_from_slice(&acc2[1..]);
    s.z
}

// θ2(q⁴) and θ3(q⁴) for q = x 2^-shift < 1/4 (x trimmed and nonzero), θ3 over k = 1..terms and
// θ2 over k = 0..terms, less terms below their output. Writes th3 = θ3 2^(64 (th3.len() - 1))
// and th2 = θ2 2^(64 (th2.len() + z)), returning z = (l - 2) / 64, l = shift - bits(x). Each is at
// most a unit below the truncated sum.
pub fn theta_dyn(x: &[u64], shift: u64, terms: usize, th2: &mut [u64], th3: &mut [u64]) -> usize {
    let s = ThetaSchedule::new(x, shift, terms, th2.len(), th3.len());
    let (r1, r2) = (s.width(1), s.width(2));
    let mut scratch = ScratchGuard::acquire();
    let bufs = scratch.get_splits([r1, r1, r1, r2, s.g2, s.g3]);
    theta_core(&s, x, th2, th3, bufs, hi_mul_dyn, hi_sqr_dyn)
}

// Needs N >= max(th2.len() + 2, th3.len() + 1), checked before any input.
pub fn theta_static<const N: usize>(
    x: &[u64],
    shift: u64,
    terms: usize,
    th2: &mut [u64],
    th3: &mut [u64],
) -> Result<usize, CapacityError> {
    capacity::<N>((th2.len() + 2).max(th3.len() + 1))?;
    let s = ThetaSchedule::new(x, shift, terms, th2.len(), th3.len());
    let mut bufs = [[0; N]; 6];
    let bufs = bufs.each_mut().map(|b| &mut b[..]);
    Ok(theta_core(
        &s,
        x,
        th2,
        th3,
        bufs,
        hi_mul_static::<N>,
        hi_sqr_static::<N>,
    ))
}

// The running a in out, b in g (out.len() limbs) and a b in prod (2 out.len()): each step's full
// product over the trimmed pair, rooted back to out.len() limbs.
fn agm_core(
    a: &[u64],
    b: &[u64],
    iters: usize,
    out: &mut [u64],
    [g, prod]: [&mut [u64]; 2],
    mul: BinOp,
    sqrt: SqrtOp,
) {
    let l = out.len();
    assert!(
        a.len() <= l && b.len() <= l,
        "the AGM's inputs must fit its output"
    );
    let (g, prod) = (&mut g[..l], &mut prod[..2 * l]);
    out[..a.len()].copy_from_slice(a);
    out[a.len()..].fill(0);
    g[..b.len()].copy_from_slice(b);
    g[b.len()..].fill(0);
    for i in 1..=iters {
        // The last geometric mean would feed no later mean.
        let t = (i < iters).then(|| {
            let (la, lg) = (buf_len(out), buf_len(g));
            if la == 0 || lg == 0 {
                return 0;
            }
            let p = &mut prod[..la + lg];
            p[la + lg - 1] = mul(&out[..la], &g[..lg], &mut p[..la + lg - 1]);
            buf_len(p)
        });
        let carry = add_buf(out, g);
        shr_buf(out, 1);
        if carry {
            out[l - 1] |= 1 << 63;
        }
        if let Some(t) = t {
            // An odd t reads as p B, a root 2^32 too high: ⌊⌊√(p B)⌋ / 2^32⌋ = ⌊√p⌋.
            let sl = t.div_ceil(2);
            if t > 0 {
                sqrt(&mut prod[..t], &mut g[..sl]);
                if t % 2 == 1 {
                    shr_buf(&mut g[..sl], 32);
                }
            }
            g[sl..].fill(0);
        }
    }
}

// a_iters of the AGM sequence from integers a and b at one scale (a.len(), b.len() <= out.len(),
// zero-extended): iters arithmetic means and iters - 1 geometric ones. iters = 0 copies a. Each
// mean is within a unit of its pair's exact mean, but a unit in b_k < a_k costs up to about
// a_k / b_k units of the result, so the frame must give the smaller input full precision.
pub fn agm_dyn(a: &[u64], b: &[u64], iters: usize, out: &mut [u64]) {
    let l = out.len();
    let mut scratch = ScratchGuard::acquire();
    let bufs = scratch.get_splits([l, 2 * l]);
    agm_core(a, b, iters, out, bufs, mul_dyn, sqrt_approx_dyn);
}

// Needs N >= 2 out.len(), checked before any input.
pub fn agm_static<const N: usize>(
    a: &[u64],
    b: &[u64],
    iters: usize,
    out: &mut [u64],
) -> Result<(), CapacityError> {
    capacity::<N>(2 * out.len())?;
    let (mut g, mut prod) = ([0; N], [0; N]);
    agm_core(
        a,
        b,
        iters,
        out,
        [&mut g, &mut prod],
        mul_static::<N>,
        sqrt_approx_static::<N>,
    );
    Ok(())
}
