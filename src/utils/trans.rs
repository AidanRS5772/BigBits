use crate::utils::consts::consts::{
    ln2_dyn, ln2_static_unguarded, pi_dyn, pi_static, shifted, static_jobs, LN2_LIMBS,
};
use crate::utils::div::{div_dyn, div_rem_dyn, div_rem_static, div_static};
use crate::utils::mul::{
    hi_mul_dyn, hi_mul_static, hi_sqr_dyn, hi_sqr_static, mul_prim2, sqr_dyn, sqr_static,
};
use crate::utils::series::{
    bit_len, bits, join2, ratio_to_fraction_dyn, ratio_to_fraction_static, series_eval_dyn,
    series_eval_static, static_split_stack, BBPSeries, Bbp, Engel, EngelSeries, Hyper, HyperSeries,
    Series,
};
use crate::utils::sqrt::{sqrt_approx_dyn, sqrt_approx_static};
use crate::utils::theta_plan::{ThetaPlan, ThetaPlans, MAX_VALUES};
use crate::utils::utils::{
    add_buf, add_prim, buf_len, cmp_buf, inc_buf, shl_buf, shr_buf, split_sh, sub_buf, twos_comp,
};
use crate::tuning::k;
use crate::utils::{CapacityError, ScratchGuard, THETA_STATIC_SLOTS};

// `$f` called with `$T`, a const generic, set to a usize knob's value `$v`: a const argument in a
// normal build (where accessors are `const fn`), and in a tune build a match over the knob's valid
// values, each arm binding `const $T`. The listed values must be the knob's whole valid range.
macro_rules! knob_generic {
    ($v:expr; [$($n:literal)*]; $T:ident => $e:expr) => {{
        #[cfg(not(feature = "tune"))]
        let r = {
            const $T: usize = $v;
            $e
        };
        #[cfg(feature = "tune")]
        let r = match $v {
            $($n => {
                const $T: usize = $n;
                $e
            })*
            v => unreachable!("knob value {v} outside its valid range"),
        };
        r
    }};
}
use rayon::prelude::*;
use std::cmp::Ordering;
use std::f64::consts::{LOG2_E, PI};
use std::sync::{Condvar, Mutex, PoisonError};
use std::{fmt, panic, thread};

const GUARD_LIMBS: usize = 1;
const EXP_RED_GUARD_BITS: u8 = 58;
const MAX_PAR_CHUNKS: usize = 64;

// mul, hi_mul and div; sqr and hi_sqr; sqrt (radicand as scratch, root).
type BinOp = fn(&[u64], &[u64], &mut [u64]) -> u64;
type UnOp = fn(&[u64], &mut [u64]) -> u64;
type SqrtOp = fn(&mut [u64], &mut [u64]);
type Ratio = fn(&[u64], &[u64], i64, &mut [u64]);

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
    let chunks = || exp_chunks(s + 1, end, k::exp::dyn_chunk_min_bits());
    let mut scratch = ScratchGuard::acquire();
    let [a, c, mut rs] = scratch.get_splits([v, v + 1, chunks().map(|(_, wd)| slot(wd)).sum()]);
    let mut parts = Vec::new();
    for (s, width) in chunks() {
        let (r, rest) = std::mem::take(&mut rs).split_at_mut(slot(width));
        rs = rest;
        parts.extend(chunk_bits(x, e, s, width, r));
    }

    let par = chunks_par(parts.len(), v, k::exp::dyn_chunk_par_work(), true);
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

// e^X for X = 2^(64m) x, where x is a fixed-point number whose top limb is its integer part: the
// top limb is worth 2^(64m) and x[0] is worth 2^(64(m + 1 - x.len())). So m = 0 is X in
// [0, 2^64), and m = -1 a full-precision X in [0, 1); m = -(x.len()) puts X a limb below an
// out of x's length, where e^X is 1 to every limb.
pub fn exp_dyn(x: &[u64], m: i64, out: &mut [u64]) -> Result<u64, ExpError> {
    exp_dyn_at(x, m, out, k::exp::dyn_sqr_bits())
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
        (k::exp::dyn_chunk_min_bits(), k::exp::dyn_chunk_par_work(), true),
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
    let chunks = || exp_chunks(s + 1, end, k::exp::static_chunk_min_bits());
    let chunk = |(s, width)| exp_chunk_static::<N>(x, e, s, width, w);
    let par = chunks_par(chunks().count(), v, k::exp::static_chunk_par_work(), false);
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
    exp_static_at::<N>(x, m, out, k::exp::static_sqr_bits())
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
        (k::exp::static_chunk_min_bits(), k::exp::static_chunk_par_work(), false),
        hi_sqr_static::<N>,
    )
}

// out <- floor(a b 2^-sh) < B^out.len(), never rounded up. Returns its length.
fn mul_shr(a: &[u64], b: &[u64], sh: u64, out: &mut [u64], hi_mul: BinOp) -> usize {
    if a.is_empty() || b.is_empty() {
        out.fill(0);
        return 0;
    }
    shr_prod(a.len() + b.len(), sh, out, |w| hi_mul(a, b, w))
}

// mul_shr(a, a, ..) through a high square.
fn sqr_shr(a: &[u64], sh: u64, out: &mut [u64], hi_sqr: UnOp) -> usize {
    shr_prod(2 * a.len(), sh, out, |w| hi_sqr(a, w))
}

// mul_shr for a product of full limbs whose window below its top limb hi writes, as hi_mul does.
fn shr_prod(full: usize, sh: u64, out: &mut [u64], hi: impl FnOnce(&mut [u64]) -> u64) -> usize {
    let (sl, sb) = split_sh(sh);
    let o = out.len();
    if full <= sl {
        out.fill(0);
        return 0;
    }
    // The product's limbs from sl up; one past out holds only bits the shift moves down.
    let r = full - sl;
    debug_assert!(r <= o + 1, "a b 2^-sh must fit out");
    let top = hi(&mut out[..r - 1]);
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
        (0..len).rev().find_map(|i| {
            let l = self.limb(i as i64);
            (l != 0).then(|| (i, 64 * (len - 1 - i) as u64 + l.leading_zeros() as u64))
        })
    }
}

// X = 2^(64m) x, x a fixed-point number whose top limb is its integer part: x trimmed, e = ⌊log2 X⌋, and δ = |u - 1| for u = X 2^-e,
// or u = X for X in [1/2, 1) (below), where -ln 2 + ln 2X would cancel.
struct LnNorm<'a> {
    x: &'a [u64],
    e: i128,
    below: bool,
    sh: u32,
}

impl<'a> LnNorm<'a> {
    // None for X = 0.
    fn new(x: &'a [u64], m: i64) -> Option<Self> {
        let len = buf_len(x);
        if len == 0 {
            return None;
        }
        let me = m as i128 - (x.len() - len) as i128;
        let x = &x[..len];
        let lz = x[len - 1].leading_zeros();
        let e = 64 * me + 63 - lz as i128;
        let below = e == -1;
        let sh = if below { 0 } else { lz + 1 };
        Some(LnNorm { x, e, below, sh })
    }

    // δ's leading zero bits, or None when δ = 0 (X = 2^e).
    fn lead(&self) -> Option<u64> {
        Delta::new(self.x, self.sh, self.below)
            .top()
            .map(|(_, lead)| lead)
    }

    // e ln 2 and ln u cancel only for X in [1/2, 2).
    fn cancels(&self) -> bool {
        self.e == 0 || self.below
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

// Chunks a = A 2^-k <= z (A odd) off z = n / d (n 2^-64w in nb, d 2^-64v in db), each passed to
// chunk with 2a in the w-limb frame, until z's top bit passes end and the first's.
// atanh(z) = atanh(a) + atanh((n - a d) / (d - a n)), so only a chunk's bits take a (short)
// division. qd and qq hold v + 1 limbs. Returns the remainder's top bit, or u64::MAX once n is 0.
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
    ratio: Ratio,
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
    let min = k::ln::dyn_chunk_min_bits();
    let sl = ln_chain((nb, db), z, end, min, bufs, ops, |a, k, a2| {
        carry |= add_buf(acc, a2);
        if let Some(c) = ln_chunk(a, k, w) {
            chunks.push((pool.len(), a.len(), k, c));
            pool.extend_from_slice(a);
        }
    });

    let par = chunks_par(chunks.len(), v, k::ln::dyn_chunk_par_work(), true);
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
    let chunks = exp_chunks(lo + 3, end, k::ln::static_chunk_min_bits()).count() + 1;
    let ops = (hi_mul_static::<N> as BinOp, div_static::<N> as BinOp);
    // The chain, adding each 2a and any series sum returns, then the tail into qq.
    let mut chain =
        |acc: &mut [u64],
         carry: &mut bool,
         sum: &mut dyn FnMut(&[u64], u64, LnChunk) -> Option<([u64; N], usize)>| {
            let bufs = [&mut qn[..], &mut qd[..], &mut qq[..]];
            let min = k::ln::static_chunk_min_bits();
            let sl = ln_chain((&mut *nb, &mut *db), z, end, min, bufs, ops, |a, k, a2| {
                *carry |= add_buf(acc, a2);
                if let Some((b, bl)) = ln_chunk(a, k, w).and_then(|c| sum(a, k, c)) {
                    *carry |= add_buf(acc, &b[..bl]);
                }
            });
            ln_tail(nb, db, sl, (w, v), &mut qq, ratio_to_fraction_static::<N>)
        };
    let mut carry = false;
    let tl = if chunks_par(chunks, v, k::ln::static_chunk_par_work(), false) {
        thread::scope(|sc| {
            let mut hs: [_; MAX_PAR_CHUNKS] = std::array::from_fn(|_| None);
            let (mut nh, mut top) = (0, None);
            let tl = chain(acc, &mut carry, &mut |a, k, c| {
                let (mut ac, al) = ([0; N], a.len());
                ac[..al].copy_from_slice(a);
                let job = move || ln_chunk_static::<N>(&ac[..al], k, c, v);
                // The caller's thread sums the top chunk once the chain is done.
                if top.is_none() {
                    top = Some(job);
                    return None;
                }
                if nh < MAX_PAR_CHUNKS {
                    let builder = thread::Builder::new().stack_size(static_split_stack(N));
                    if let Ok(h) = builder.spawn_scoped(sc, job) {
                        hs[nh] = Some(h);
                        nh += 1;
                        return None;
                    }
                }
                Some(job())
            });
            let joined = hs.into_iter().flatten();
            let rest = joined.map(|h| h.join().unwrap_or_else(|e| panic::resume_unwind(e)));
            for (b, bl) in top.map(|job| job()).into_iter().chain(rest) {
                carry |= add_buf(acc, &b[..bl]);
            }
            tl
        })
    } else {
        chain(acc, &mut carry, &mut |a, k, c| {
            Some(ln_chunk_static::<N>(a, k, c, v))
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

// |ln X| = |e ln 2 + 2^j L|, L = |ln y| from the entry, y = (X 2^-e)^(1 / 2^j). ys, sq and ent
// hold o + 4, o + 3 and o + 2 limbs; the roots run only when sq holds fl + 2, ys fl + 1 and ent
// o + 2 + ⌈j / 64⌉.
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
    let nx = LnNorm::new(x, m).ok_or(LnError::Zero)?;
    let o = out.len();
    if o == 0 {
        return Ok((0, false));
    }
    let (x, below, sh) = (nx.x, nx.below, nx.sh);
    let len = x.len();
    let e = if below { 0 } else { nx.e };
    let f = o + 1;
    let mut j = 0;
    // (pe, zl) with L = ent 2^-64(zl + pe), or None when L is 0 or below the frame.
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
    // e < 0 is at most -2, so |e| ln 2 covers 2^j L < ln 2.
    let t = &mut sq[..f + 2];
    match l {
        Some((pe, zl)) => shifted(&ent[..pe], j as i64 + 64 * (f as i64 - (zl + pe) as i64), t),
        None => t.fill(0),
    }
    Ok(ln2_combine(e, true, t, ys, ln2, out))
}

// |c ln 2 ± T| into out, + when add, for T at f = t.len() - 2 fraction limbs; r holds f + 3. ln 2
// at f + 1 limbs keeps |c| < 2^71 times its truncation within the last limb. Returns (k, neg).
fn ln2_combine(
    c: i128,
    add: bool,
    t: &mut [u64],
    r: &mut [u64],
    ln2: fn(&mut [u64]),
    out: &mut [u64],
) -> (i64, bool) {
    let f = t.len() - 2;
    let r = &mut r[..f + 3];
    ln2(&mut r[..f + 1]);
    let carry = mul_prim2(&mut r[..f + 1], c.unsigned_abs());
    (r[f + 1], r[f + 2]) = (carry as u64, (carry >> 64) as u64);
    let rr = &mut r[1..];
    let (res, neg): (&[u64], bool) = if c == 0 || (c > 0) == add {
        add_buf(rr, t);
        (rr, !add)
    } else if cmp_buf(rr, t) != Ordering::Less {
        sub_buf(rr, t);
        (rr, c < 0)
    } else {
        sub_buf(t, rr);
        (t, !add)
    };
    let (top, o) = (buf_len(res), out.len());
    out.copy_from_slice(&res[top - o..top]);
    (top as i64 - 1 - f as i64, neg)
}

// Whether ln_dyn (or ln_static, with its CPUs and knobs) takes the AGM for X at o limbs on
// `threads` threads: agm_wins from the cutoff `ln.<dyn|static>_agm_limbs`, measured on one
// thread, and on two or more threads only below `ln.<dyn|static>_agm_par_until`, measured on
// the pool. The series' chunks split across threads and the AGM's steps don't, so there the
// series takes over again once it is large enough to split: the serial/parallel pair.
pub(crate) fn ln_takes_agm(
    x: &[u64],
    m: i64,
    o: usize,
    (sqrt_a, cut, par_until): (f64, usize, usize),
    threads: usize,
) -> bool {
    o >= cut
        && (threads <= 1 || o < par_until)
        && LnNorm::new(x, m).is_some_and(|n| agm_wins(o, n.lead(), n.cancels(), sqrt_a, cut))
}

// The theta–AGM logarithm where ln_takes_agm picks it on the current rayon pool, else the series.
pub fn ln_dyn(x: &[u64], m: i64, out: &mut [u64]) -> Result<(i64, bool), LnError> {
    let o = out.len();
    let knobs = (k::ln::dyn_sqrt_a(), k::ln::dyn_agm_limbs(), k::ln::dyn_agm_par_until());
    if ln_takes_agm(x, m, o, knobs, rayon::current_num_threads()) {
        // T over ln.dyn_agm_terms' valid range, 0..=THETA_MAX_TERMS.
        return knob_generic!(k::ln::dyn_agm_terms(); [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16];
            T => ln_agm_dyn_at::<T>(x, m, out));
    }
    ln_dyn_at(x, m, out, k::ln::dyn_sqrt_a())
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

// The series needs N >= o + 4; the AGM runs where ln_takes_agm picks it on the machine's CPUs
// (which the static series' threads use, like theta's) and N holds its frame.
pub fn ln_static<const N: usize>(
    x: &[u64],
    m: i64,
    out: &mut [u64],
) -> Result<(i64, bool), LnError> {
    capacity::<N>(out.len() + 4)?;
    let o = out.len();
    let knobs = (
        k::ln::static_sqrt_a(),
        k::ln::static_agm_limbs(),
        k::ln::static_agm_par_until(),
    );
    let cpus = thread::available_parallelism().map_or(1, |c| c.get());
    if ln_takes_agm(x, m, o, knobs, cpus) {
        // T over ln.static_agm_terms' valid range: the plans that fit THETA_STATIC_SLOTS.
        let r = knob_generic!(k::ln::static_agm_terms(); [0 1 2 3 4 5 6 7 8 9];
            T => ln_agm_static_at::<N, T>(x, m, out));
        match r {
            Err(LnError::Capacity(_)) => {}
            r => return r,
        }
    }
    ln_static_at::<N>(x, m, out, k::ln::static_sqrt_a())
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
// ln(1/q) = π / (4 AGM(θ2², θ3²)) (docs/agm-ln-theta.md). The powers q^(n²) follow a compile-time
// plan of squares and products (theta_plan.rs), each value as wide as the terms it feeds.

// For q = u 2^-l, u in [1/2, 1): odd terms go to θ2 2^(64 z) with g2 limbs below 2^-64z (t = z),
// even ones to θ3 - 1 with g3 limbs below 1 (t = 0), each with a guard limb.
struct ThetaSchedule {
    l: u64,
    z: usize,
    g2: usize,
    g3: usize,
}

impl ThetaSchedule {
    fn new(x: &[u64], shift: u64, w2: usize, w3: usize) -> Self {
        assert!(x.last().is_some_and(|&l| l != 0));
        assert!(w2 > 0 && w3 > 0, "theta's outputs must be non-empty");
        let l = shift
            .checked_sub(bits(x))
            .filter(|&l| l >= 2)
            .expect("theta needs q = x 2^-shift < 1/4");
        ThetaSchedule {
            l,
            z: ((l - 2) / 64) as usize,
            g2: w2 + 1,
            g3: w3,
        }
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

// Which plan theta runs and how: the gate's (Auto), or forced for tests. Parallel runs the
// shortest-path plan, threaded or in order; both give the same bits.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThetaMode {
    Auto,
    Serial,
    Parallel { threads: bool },
}

// T's plan for g-limb frames, and its workers when threaded (at least 2 when forced): the gate
// takes the parallel plan from cutoff limbs if it fits, threaded on two or more workers.
fn theta_plan<const T: usize>(
    mode: ThetaMode,
    g: usize,
    cutoff: usize,
    fits: bool,
    cpus: usize,
) -> (&'static ThetaPlan, Option<usize>) {
    let par = match mode {
        ThetaMode::Auto => fits && g >= cutoff,
        ThetaMode::Serial => false,
        ThetaMode::Parallel { .. } => {
            assert!(fits, "theta's parallel plan doesn't fit its slots");
            true
        }
    };
    if !par {
        return (const { &ThetaPlans::<T>::SERIAL }, None);
    }
    let p = const { &ThetaPlans::<T>::PARALLEL };
    let w = cpus.min(p.width);
    let workers = match mode {
        ThetaMode::Parallel { threads } => threads.then_some(w.max(2)),
        _ => (w > 1).then_some(w),
    };
    (p, workers)
}

type Acc<'a> = [&'a mut [u64]; 3];

// A plan with its schedule, input and products. Values are normalized mantissas,
// m 2^(-64 m.len()) in [1/2, 1), each with a bit exponent beside it.
struct Theta<'a> {
    s: ThetaSchedule,
    p: &'a ThetaPlan,
    x: &'a [u64],
    // Each value's width: its own term's window, widened to the widest op reading it, or 0 when
    // nothing it feeds lands in the frame (and then it never runs). q's is the widest.
    w: [usize; MAX_VALUES],
    hi_mul: BinOp,
    hi_sqr: UnOp,
}

impl<'a> Theta<'a> {
    fn new(
        s: ThetaSchedule,
        p: &'a ThetaPlan,
        x: &'a [u64],
        (hi_mul, hi_sqr): (BinOp, UnOp),
    ) -> Self {
        let mut w = [0; MAX_VALUES];
        for v in (0..p.values).rev() {
            let o = p.op[v];
            if o.term != 0 {
                w[v] = w[v].max(s.ell(o.term as u64).unwrap_or(0));
            }
            if v > 0 && w[v] > 0 {
                for i in [o.a, o.b] {
                    w[i as usize] = w[i as usize].max(w[v]);
                }
            }
        }
        Theta {
            s,
            p,
            x,
            w,
            hi_mul,
            hi_sqr,
        }
    }

    // Value v into out[..w[v]] from its operands (as stored, with their exponents), by the top
    // w[v] limbs of each. Returns its exponent.
    fn value(
        &self,
        v: usize,
        (a, ea): (&[u64], u64),
        (b, eb): (&[u64], u64),
        out: &mut [u64],
    ) -> u64 {
        let w = self.w[v];
        let out = &mut out[..w];
        if v == 0 {
            shifted(self.x, 64 * w as i64 - bits(self.x) as i64, out);
            return 0;
        }
        let o = self.p.op[v];
        let (a, b) = (&a[a.len() - w..], &b[b.len() - w..]);
        out[w - 1] = if o.a == o.b {
            (self.hi_sqr)(a, &mut out[..w - 1])
        } else {
            (self.hi_mul)(a, b, &mut out[..w - 1])
        };
        let z = out[w - 1].leading_zeros();
        debug_assert!(z < 64);
        shl_buf(out, z as u8);
        ea + eb + z as u64
    }

    // Value v's term, if it is one, into θ2 (odd n) or θ3.
    fn place(&self, v: usize, m: &[u64], e: u64, [acc2, acc3, tmp]: &mut Acc) {
        let n = self.p.op[v].term as u64;
        if n != 0 {
            let acc = if n % 2 == 1 { acc2 } else { acc3 };
            self.s.place(n, &m[..self.w[v]], e, acc, tmp);
        }
    }

    fn out(&self, th2: &mut [u64], th3: &mut [u64], [acc2, acc3, _]: &Acc) -> usize {
        let w3 = th3.len();
        th3[..w3 - 1].copy_from_slice(&acc3[1..self.s.g3]);
        th3[w3 - 1] = 1;
        th2.copy_from_slice(&acc2[1..self.s.g2]);
        self.s.z
    }

    // The ops in order, each value in its plan slot.
    fn serial(&self, slots: &mut [&mut [u64]], acc: &mut Acc) {
        let (p, w) = (self.p, &self.w);
        let mut e = [0u64; MAX_VALUES];
        for v in (0..p.values).filter(|&v| w[v] > 0) {
            let (o, d) = (p.op[v], p.slot[v] as usize);
            let out = std::mem::take(&mut slots[d]);
            if v == 0 {
                self.value(0, (&[], 0), (&[], 0), out);
            } else {
                let (a, b) = (o.a as usize, o.b as usize);
                let sa = &slots[p.slot[a] as usize][..w[a]];
                let sb = &slots[p.slot[b] as usize][..w[b]];
                e[v] = self.value(v, (sa, e[a]), (sb, e[b]), out);
            }
            slots[d] = out;
            self.place(v, &slots[d][..w[v]], e[v], acc);
        }
    }
}

// The dataflow's equal slots. SAFETY: each running op owns its output slot, starts only after its
// operands completed under the scheduler's lock, and a slot frees only after its last reader.
struct SlotPool {
    base: *mut u64,
    len: usize,
}

unsafe impl Sync for SlotPool {}

impl SlotPool {
    // SAFETY: see SlotPool; slot i must be one the pool was built with room for.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self, i: u8) -> &mut [u64] {
        std::slice::from_raw_parts_mut(self.base.add(i as usize * self.len), self.len)
    }
}

// An op ready to run: value v into slot out, from its operands' slots and exponents.
#[derive(Clone, Copy)]
struct Job {
    v: u8,
    out: u8,
    a: (u8, u64),
    b: (u8, u64),
}

impl Job {
    fn run(self, t: &Theta, pool: &SlotPool) -> u64 {
        let w = |v: u8| t.w[v as usize];
        let o = t.p.op[self.v as usize];
        // SAFETY: the operands are complete and read-only until this op completes; out is ours.
        unsafe {
            let a = &pool.get(self.a.0)[..w(o.a)];
            let b = &pool.get(self.b.0)[..w(o.b)];
            t.value(
                self.v as usize,
                (a, self.a.1),
                (b, self.b.1),
                pool.get(self.out),
            )
        }
    }
}

// Ops dispatch in plan order once their operands are done and a slot is free: with none running,
// the live values are the in-order ones, so a pool of the plan's slots never stalls. [head, next)
// is dispatched, not yet taken; a value's readers include its own sum.
struct Flow<'a> {
    next: usize,
    head: usize,
    left: usize,
    done: [bool; MAX_VALUES],
    readers: [u8; MAX_VALUES],
    slot: [u8; MAX_VALUES],
    exp: [u64; MAX_VALUES],
    free: [u8; MAX_VALUES],
    nfree: usize,
    failed: bool,
    acc: Acc<'a>,
}

impl<'a> Flow<'a> {
    fn new(t: &Theta, slots: usize, acc: Acc<'a>) -> Self {
        let mut f = Flow {
            next: 0,
            head: 0,
            left: 0,
            done: [false; MAX_VALUES],
            readers: [0; MAX_VALUES],
            slot: [0; MAX_VALUES],
            exp: [0; MAX_VALUES],
            free: std::array::from_fn(|i| i as u8),
            nfree: slots,
            failed: false,
            acc,
        };
        for v in (0..t.p.values).filter(|&v| t.w[v] > 0) {
            f.left += 1;
            f.readers[v] += 1;
            let o = t.p.op[v];
            if v > 0 {
                f.readers[o.a as usize] += 1;
                if o.b != o.a {
                    f.readers[o.b as usize] += 1;
                }
            }
        }
        f
    }

    fn dispatch(&mut self, t: &Theta) {
        while self.next < t.p.values {
            let v = self.next;
            let o = t.p.op[v];
            if t.w[v] > 0 {
                if v > 0 && !(self.done[o.a as usize] && self.done[o.b as usize]) || self.nfree == 0
                {
                    return;
                }
                self.nfree -= 1;
                self.slot[v] = self.free[self.nfree];
            }
            self.next += 1;
        }
    }

    fn pop(&mut self, t: &Theta) -> Option<Job> {
        while self.head < self.next && t.w[self.head] == 0 {
            self.head += 1;
        }
        if self.head == self.next {
            return None;
        }
        let v = self.head;
        self.head += 1;
        let (a, b) = (t.p.op[v].a as usize, t.p.op[v].b as usize);
        Some(Job {
            v: v as u8,
            out: self.slot[v],
            a: (self.slot[a], self.exp[a]),
            b: (self.slot[b], self.exp[b]),
        })
    }

    fn release(&mut self, v: usize) {
        self.readers[v] -= 1;
        if self.readers[v] == 0 {
            self.free[self.nfree] = self.slot[v];
            self.nfree += 1;
        }
    }

    // Job's value is in its slot with exponent e: sum its term, release it and its operands,
    // and dispatch what that allows.
    fn complete(&mut self, t: &Theta, pool: &SlotPool, j: Job, e: u64) {
        let v = j.v as usize;
        (self.exp[v], self.done[v]) = (e, true);
        self.left -= 1;
        // SAFETY: v is complete and nothing writes its slot until its readers finish.
        let m = unsafe { &pool.get(j.out)[..t.w[v]] };
        t.place(v, m, e, &mut self.acc);
        if v > 0 {
            let o = t.p.op[v];
            self.release(o.a as usize);
            if o.b != o.a {
                self.release(o.b as usize);
            }
        }
        self.release(v);
        self.dispatch(t);
    }
}

fn lock<'m, T>(m: &'m Mutex<T>) -> std::sync::MutexGuard<'m, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// The dataflow on rayon: each finished op dispatches and spawns its successors in one scope.
fn theta_flow_dyn(t: &Theta, pool: &SlotPool, flow: &Mutex<Flow>) {
    fn spawn<'s>(sc: &rayon::Scope<'s>, t: &'s Theta, pool: &'s SlotPool, flow: &'s Mutex<Flow>) {
        let mut f = lock(flow);
        while let Some(j) = f.pop(t) {
            sc.spawn(move |sc| {
                let e = j.run(t, pool);
                lock(flow).complete(t, pool, j, e);
                spawn(sc, t, pool, flow);
            });
        }
    }
    lock(flow).dispatch(t);
    rayon::scope(|sc| spawn(sc, t, pool, flow));
}

// The dataflow on scoped threads (the caller and workers - 1 more), waiting on a condvar for ops.
fn theta_flow_static<const N: usize>(
    t: &Theta,
    pool: &SlotPool,
    flow: &Mutex<Flow>,
    workers: usize,
) {
    let cv = Condvar::new();
    let work = || {
        let mut f = lock(flow);
        while f.left > 0 && !f.failed {
            let Some(j) = f.pop(t) else {
                f = cv.wait(f).unwrap_or_else(PoisonError::into_inner);
                continue;
            };
            drop(f);
            let r = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let e = j.run(t, pool);
                lock(flow).complete(t, pool, j, e);
            }));
            f = lock(flow);
            if let Err(e) = r {
                f.failed = true;
                cv.notify_all();
                drop(f);
                panic::resume_unwind(e);
            }
            cv.notify_all();
        }
    };
    lock(flow).dispatch(t);
    thread::scope(|sc| {
        for _ in 1..workers {
            let b = thread::Builder::new().stack_size(static_split_stack(N));
            if b.spawn_scoped(sc, &work).is_err() {
                break;
            }
        }
        work();
    });
}

// θ2(q⁴) and θ3(q⁴) for q = x 2^-shift < 1/4 (x trimmed, nonzero), θ3 over k = 1..T and θ2 over
// k = 0..T: th3 = θ3 2^(64 (th3.len() - 1)), th2 = θ2 2^(64 (th2.len() + z)), each at most a unit
// low. Returns z = (l - 2) / 64, l = shift - bits(x).
pub fn theta_dyn<const T: usize>(x: &[u64], shift: u64, th2: &mut [u64], th3: &mut [u64]) -> usize {
    theta_dyn_at::<T>(x, shift, th2, th3, ThetaMode::Auto)
}

#[doc(hidden)]
pub fn theta_dyn_at<const T: usize>(
    x: &[u64],
    shift: u64,
    th2: &mut [u64],
    th3: &mut [u64],
    mode: ThetaMode,
) -> usize {
    let s = ThetaSchedule::new(x, shift, th2.len(), th3.len());
    let (g2, g3) = (s.g2, s.g3);
    let cpus = rayon::current_num_threads();
    let (p, workers) = theta_plan::<T>(mode, g2.max(g3), k::theta::dyn_par_limbs(), true, cpus);
    let t = Theta::new(s, p, x, (hi_mul_dyn, hi_sqr_dyn));
    let wq = t.w[0];
    // θ2, θ3 and a temp, then the pool's slots of q's width or the plan's slots at their widest.
    let mut lens = [0; MAX_VALUES];
    let slots = match workers {
        Some(k) => {
            lens[0] = (p.slots + k - 1) * wq;
            1
        }
        None => {
            for v in 0..p.values {
                let s = p.slot[v] as usize;
                lens[s] = lens[s].max(t.w[v]);
            }
            p.slots
        }
    };
    let mut scratch = ScratchGuard::acquire();
    let sizes = [g2, g3, wq]
        .into_iter()
        .chain(lens[..slots].iter().copied());
    let mut bufs = scratch.get_split_vec(sizes);
    let (acc, rest) = bufs.split_at_mut(3);
    let acc: &mut Acc = acc.try_into().expect("three buffers");
    acc[0].fill(0);
    acc[1].fill(0);
    match workers {
        Some(k) => {
            let pool = SlotPool {
                base: rest[0].as_mut_ptr(),
                len: wq,
            };
            let flow = Mutex::new(Flow::new(
                &t,
                p.slots + k - 1,
                acc.each_mut().map(|a| &mut **a),
            ));
            theta_flow_dyn(&t, &pool, &flow);
        }
        None => t.serial(rest, acc),
    }
    t.out(th2, th3, acc)
}

// Needs N >= max(th2.len() + 2, th3.len() + 1), checked before any input, and T's least-work plan
// to fit THETA_STATIC_SLOTS (a compile-time check).
pub fn theta_static<const N: usize, const T: usize>(
    x: &[u64],
    shift: u64,
    th2: &mut [u64],
    th3: &mut [u64],
) -> Result<usize, CapacityError> {
    theta_static_at::<N, T>(x, shift, th2, th3, ThetaMode::Auto)
}

#[doc(hidden)]
pub fn theta_static_at<const N: usize, const T: usize>(
    x: &[u64],
    shift: u64,
    th2: &mut [u64],
    th3: &mut [u64],
    mode: ThetaMode,
) -> Result<usize, CapacityError> {
    const {
        assert!(
            ThetaPlans::<T>::SERIAL.slots <= THETA_STATIC_SLOTS,
            "theta's least-work plan for T needs more than THETA_STATIC_SLOTS slots"
        )
    };
    capacity::<N>((th2.len() + 2).max(th3.len() + 1))?;
    let fits = const { ThetaPlans::<T>::PARALLEL.slots <= THETA_STATIC_SLOTS };
    let s = ThetaSchedule::new(x, shift, th2.len(), th3.len());
    let cpus = thread::available_parallelism().map_or(1, |c| c.get());
    let g = s.g2.max(s.g3);
    let (p, workers) = theta_plan::<T>(mode, g, k::theta::static_par_limbs(), fits, cpus);
    let t = Theta::new(s, p, x, (hi_mul_static::<N>, hi_sqr_static::<N>));
    let (mut arena, mut acc2, mut acc3, mut tmp) =
        ([[0; N]; THETA_STATIC_SLOTS], [0; N], [0; N], [0; N]);
    let mut acc: Acc = [&mut acc2, &mut acc3, &mut tmp];
    match workers {
        Some(k) => {
            // Slots of q's width, as many as the arena holds past the plan's.
            let wq = t.w[0];
            let k = k.min(THETA_STATIC_SLOTS * N / wq - p.slots + 1);
            let pool = SlotPool {
                base: arena.as_flattened_mut().as_mut_ptr(),
                len: wq,
            };
            let flow = Mutex::new(Flow::new(
                &t,
                p.slots + k - 1,
                acc.each_mut().map(|a| &mut **a),
            ));
            theta_flow_static::<N>(&t, &pool, &flow, k);
        }
        None => t.serial(&mut arena.each_mut().map(|a| &mut a[..]), &mut acc),
    }
    Ok(t.out(th2, th3, &acc))
}

// AGM(m + D, m - D) = m (1 - Σ g_i u^i) with u = (D / m)² and g_i = N_i 2^-e_i, the inverse of
// K's series Σ (C(2i, i) / 4^i)² u^i, since AGM(1 + ε, 1 - ε) = π / (2 K(ε)). Every g_i is
// positive and they fall, so the next term bounds the rest. N_20 passes u64.
#[doc(hidden)]
pub const AGM_TAIL: [(u64, u32); 19] = [
    (1, 2),
    (5, 6),
    (11, 8),
    (469, 14),
    (1379, 16),
    (17223, 20),
    (56001, 22),
    (11998869, 30),
    (41064827, 32),
    (571915951, 36),
    (2018982161, 38),
    (115338112823, 44),
    (415720532641, 46),
    (6041874952949, 50),
    (22103950817043, 52),
    (20825721430968213, 62),
    (77047750289886219, 64),
    (1145470055108455527, 68),
    (4274935497276922857, 70),
];

// Bits the tail's quotient ε keeps past those its correction C needs.
const AGM_TAIL_GUARD: i64 = 24;

// The AGM's kernels: high product, high square, approximate root, and the tail's ratio.
type AgmOps = (BinOp, UnOp, SqrtOp, Ratio);

// The fewest terms I whose remainder m Σ_(i>I) g_i u^i is at most half a unit, for m < 2^bm and
// u < 2^(-2t): under 2^(bm + 1) g_(I+1) u^(I+1), as u < 1/4 needs t >= 1. None past the table.
fn agm_tail_terms(bm: u64, t: u64) -> Option<usize> {
    if t == 0 {
        return None;
    }
    (0..=AGM_TAIL.len()).find(|&i| {
        let (n, e) = AGM_TAIL[i.min(AGM_TAIL.len() - 1)];
        2 * t * (i as u64 + 1) + e as u64 >= bm + 2 + bit_len(n)
    })
}

// ε's limbs, for a correction C < 2^cb.
fn agm_ratio_limbs(cb: i64) -> usize {
    ((cb + AGM_TAIL_GUARD).max(1) as u64).div_ceil(64) as usize
}

// Whether the tail can finish from (a, b) in at most cap terms with ε in at most ratio_max limbs,
// by bounds the tail's exact m = (a + b) / 2 and D = |a - b| / 2 meet: bits(m) in [top - 1, top]
// and bits(D) <= diff - 1, from the top limb where a and b differ.
fn agm_tail_ready(a: &[u64], b: &[u64], cap: usize, ratio_max: usize) -> bool {
    let top = bits(&a[..buf_len(a)]).max(bits(&b[..buf_len(b)]));
    let diff = (0..a.len())
        .rev()
        .find(|&i| a[i] != b[i])
        .map_or(0, |i| 64 * i as u64 + bit_len(a[i].abs_diff(b[i])));
    // |a - b| <= 1 leaves D = 0.
    if diff <= 1 {
        return true;
    }
    match agm_tail_terms(top, top.saturating_sub(diff + 1)) {
        Some(0) => true,
        Some(i) if i <= cap => agm_ratio_limbs(2 * diff as i64 - top as i64 - 1) <= ratio_max,
        _ => false,
    }
}

// x += floor(n 2^sh), which fits x.
fn add_shifted(x: &mut [u64], n: u64, sh: i64) {
    let c = if sh < 0 {
        add_prim(x, n.checked_shr(sh.unsigned_abs() as u32).unwrap_or(0))
    } else {
        let (w, v) = split_sh(sh as u64);
        let wide = (n as u128) << v;
        let parts = [wide as u64, (wide >> 64) as u64];
        add_buf(&mut x[w..], &parts[..buf_len(&parts)])
    };
    debug_assert!(!c);
}

// AGM(a, b) into out, from a in out and b in g (both nonzero), as m - C with
// C = m u H(u) = D ε H(u), ε = D / m and H = Σ g_i u^(i - 1) in [1/4, 0.26). Returns its terms.
// Each value is a floor at a frame one guard past what C needs, which shrinks by 2t bits a level
// (u < 2^-2t): ε at f1 = 64 we fraction bits past 2^-t, u at f1, Horner's h_i at f1 - 2t (i - 1),
// so only ε's division and the last product D (ε H) run at C's width. out is within 2 units.
fn agm_tail(
    out: &mut [u64],
    [g, d, e, u, h]: [&mut [u64]; 5],
    (hi_mul, hi_sqr, ratio): (BinOp, UnOp, Ratio),
) -> usize {
    let l = out.len();
    // D = |a - b| / 2 and m = lo + D = (a + b) / 2, floored alike (a ± b share a parity).
    let (b, d) = (&g[..l], &mut d[..l]);
    d.copy_from_slice(out);
    let out_lo = sub_buf(d, b);
    if out_lo {
        twos_comp(d);
    }
    shr_buf(d, 1);
    if !out_lo {
        out.copy_from_slice(b);
    }
    let c = add_buf(out, d);
    debug_assert!(!c, "m < B^l");
    let (ml, dl) = (buf_len(out), buf_len(d));
    if dl == 0 {
        return 0;
    }
    let (bm, bd) = (bits(&out[..ml]), bits(&d[..dl]));
    let t = bm - bd - 1;
    let terms = agm_tail_terms(bm, t).expect("agm_tail_ready bounds the terms");
    if terms == 0 {
        return 0;
    }
    let dv = &d[..dl];
    // ε 2^(64 we + t) < B^we, at least 64 we - 2 bits.
    let we = agm_ratio_limbs(2 * bd as i64 - bm as i64);
    let e = &mut e[..we];
    ratio(dv, &out[..ml], -(t as i64), e);
    let f1 = 64 * we as u64;
    // ε from the limbs u and ε W read: each one below costs under a unit.
    let ke = ((2 * t).saturating_sub(1) / 64).min(we as u64 - 1) as usize;
    let ev = &e[ke..buf_len(e)];
    let (mut cur, mut nxt) = (g, h);
    // W = u h_2 at f1 in cur[..cl], for I >= 2.
    let mut cl = 0;
    if terms >= 2 {
        // U = u 2^f1 < 2^(f1 - 2t).
        sqr_shr(ev, f1 + 2 * t - 128 * ke as u64, &mut u[..we], hi_sqr);
        let ul = buf_len(&u[..we]);
        // h_I = g_I, then h_i = g_i + u h_(i+1) at f_i = f1 - 2t (i - 1), ending at W.
        let level = |i: usize| f1 - 2 * t * (i as u64 - 1);
        let place = |x: &mut [u64], i: usize| {
            let (n, ei) = AGM_TAIL[i - 1];
            add_shifted(x, n, level(i) as i64 - ei as i64);
        };
        let top = level(terms).div_ceil(64) as usize;
        cur[..top].fill(0);
        place(&mut cur[..top], terms);
        cl = buf_len(&cur[..top]);
        for i in (1..terms).rev() {
            // u's limbs below level i's frame cost under a unit there.
            let ku = ((2 * t * (i as u64 - 1)) / 64) as usize;
            let ol = level(i).div_ceil(64) as usize;
            let sh = f1 - 2 * t - 64 * ku as u64;
            mul_shr(&u[ku..ul], &cur[..cl], sh, &mut nxt[..ol], hi_mul);
            if i > 1 {
                place(&mut nxt[..ol], i);
            }
            cl = buf_len(&nxt[..ol]);
            std::mem::swap(&mut cur, &mut nxt);
        }
    }
    // V = ε H 2^(64 we + t) = ε / 4 + ε W, in u.
    let v = &mut u[..we];
    mul_shr(ev, &cur[..cl], f1 - 64 * ke as u64, v, hi_mul);
    shr_buf(e, 2);
    let c = add_buf(v, e);
    debug_assert!(!c);
    let vl = buf_len(v);
    // Y = C 2^64 from D's limbs at and above C's (the dropped ones cost under 2^-10 units).
    let kd = (t.saturating_sub(9) / 64) as usize;
    let y = &mut cur[..l + 2];
    mul_shr(&dv[kd..], &v[..vl], f1 + t - 64 - 64 * kd as u64, y, hi_mul);
    let y = &y[1..];
    let b = sub_buf(out, &y[..buf_len(y)]);
    debug_assert!(!b, "C < m");
    terms
}

// The limit of a and b's AGM at out's scale (zero-extended), within a few units: means until the
// closed-form tail needs at most cap terms (or ε would pass ratio_max limbs), then the tail. Each
// mean is within a unit, but a unit in b_k < a_k costs about a_k / b_k units of the result. Each
// root reads its t-limb product's top ⌈t/2⌉ + 2 limbs (in prod) as padded by zeros, which moves it
// by under 2^-60 units. Returns (roots, tail terms).
fn agm_core(
    a: &[u64],
    b: &[u64],
    cap: usize,
    out: &mut [u64],
    bufs: [&mut [u64]; 5],
    ops: AgmOps,
    ratio_max: usize,
) -> (usize, usize) {
    let l = out.len();
    assert!(
        a.len() <= l && b.len() <= l,
        "the AGM's inputs must fit its output"
    );
    let [g, prod, e, u, h] = bufs.map(|x| &mut x[..l + 2]);
    out[..a.len()].copy_from_slice(a);
    out[a.len()..].fill(0);
    g[..b.len()].copy_from_slice(b);
    g[b.len()..].fill(0);
    // AGM(x, 0) = 0, which halving would take 64 l steps to reach.
    if buf_len(out) == 0 || buf_len(&g[..l]) == 0 {
        out.fill(0);
        return (0, 0);
    }
    let (hi_mul, hi_sqr, sqrt, ratio) = ops;
    // Halving log2(a / b), then doubling the gap.
    let most = 2 * bit_len(64 * l as u64) as usize + 16;
    let mut roots = 0;
    while !agm_tail_ready(out, &g[..l], cap, ratio_max) {
        debug_assert!(roots < most, "the AGM takes O(log P) steps");
        roots += 1;
        // (t, hl): the product's length and its top limbs in prod. Neither operand is ever 0.
        let (la, lg) = (buf_len(out), buf_len(&g[..l]));
        // The larger operand first, so swapped inputs make the same products.
        let (x, y) = (&out[..la], &g[..lg]);
        let (x, y) = if cmp_buf(x, y) == Ordering::Less {
            (y, x)
        } else {
            (x, y)
        };
        let t0 = la + lg;
        let h0 = t0.min(t0.div_ceil(2) + 2);
        prod[h0 - 1] = hi_mul(x, y, &mut prod[..h0 - 1]);
        let hl = buf_len(&prod[..h0]);
        let t = t0 - h0 + hl;
        let carry = add_buf(out, &g[..l]);
        shr_buf(out, 1);
        if carry {
            out[l - 1] |= 1 << 63;
        }
        // x B^(2 sl - hl): the dropped limbs, and for an odd t one more, whose 2^32 the shift
        // removes exactly, as ⌊⌊√(p B)⌋ / 2^32⌋ = ⌊√p⌋.
        let sl = t.div_ceil(2);
        sqrt(&mut prod[..hl], &mut g[..sl]);
        if t % 2 == 1 {
            shr_buf(&mut g[..sl], 32);
        }
        g[sl..l].fill(0);
    }
    (
        roots,
        agm_tail(out, [g, prod, e, u, h], (hi_mul, hi_sqr, ratio)),
    )
}

// AGM(a, b) at out's scale: the tail from agm.dyn_tail_terms terms.
pub fn agm_dyn(a: &[u64], b: &[u64], out: &mut [u64]) {
    agm_dyn_at(a, b, k::agm::dyn_tail_terms(), out);
}

// agm_dyn with the tail from at most max_terms terms (any cap past AGM_TAIL's 19 acts as 19),
// returning (roots, tail terms).
#[doc(hidden)]
pub fn agm_dyn_at(a: &[u64], b: &[u64], max_terms: usize, out: &mut [u64]) -> (usize, usize) {
    let l = out.len();
    let mut scratch = ScratchGuard::acquire();
    let bufs = scratch.get_splits([l + 2; 5]);
    agm_core(
        a,
        b,
        max_terms,
        out,
        bufs,
        (
            hi_mul_dyn,
            hi_sqr_dyn,
            sqrt_approx_dyn,
            ratio_to_fraction_dyn,
        ),
        usize::MAX,
    )
}

// Needs N >= out.len() + 2, checked before any input.
pub fn agm_static<const N: usize>(
    a: &[u64],
    b: &[u64],
    out: &mut [u64],
) -> Result<(), CapacityError> {
    agm_static_at::<N>(a, b, k::agm::static_tail_terms(), out).map(|_| ())
}

// agm_static's agm_dyn_at. The tail waits a step while its quotient wouldn't fit N, which only
// frames of a few limbs reach.
#[doc(hidden)]
pub fn agm_static_at<const N: usize>(
    a: &[u64],
    b: &[u64],
    max_terms: usize,
    out: &mut [u64],
) -> Result<(usize, usize), CapacityError> {
    capacity::<N>(out.len() + 2)?;
    let mut bufs = [[0; N]; 5];
    Ok(agm_core(
        a,
        b,
        max_terms,
        out,
        bufs.each_mut().map(|x| &mut x[..]),
        (
            hi_mul_static::<N>,
            hi_sqr_static::<N>,
            sqrt_approx_static::<N>,
            ratio_to_fraction_static::<N>,
        ),
        // ratio_to_fraction_static's out.len() + 2 <= N.
        N - 2,
    ))
}

// The theta–AGM logarithm (docs/agm-ln-exact-chain.md), with q = X 2^-shift, so no reciprocal:
// ln X = c ln 2 - π / (4 AGM(θ2(q⁴)², θ3(q⁴)²)), c = shift + 64b for X = x 2^(64b).

// Whether the AGM beats the series at P = 64 o bits. Per M(P) the series costs about
// log2(P / max(λ, j0))², λ = δ's lead and j0 = A log2 P, and the AGM about
// κ (1 + w / P) log2(P + w), w = λ when it widens. κ ties them at the cutoff for a generic X,
// on one thread; below it the AGM never wins.
pub(crate) fn agm_wins(
    o: usize,
    lead: Option<u64>,
    cancels: bool,
    sqrt_a: f64,
    cutoff: usize,
) -> bool {
    let cutoff = cutoff.max(1);
    let Some(lead) = lead.filter(|_| o >= cutoff) else {
        return false;
    };
    let series = |p: f64, lam: f64| {
        let j0 = (sqrt_a * p.log2()).max(1.0);
        (p / lam.max(j0)).log2().max(0.0).powi(2)
    };
    let ps = 64.0 * cutoff as f64;
    let (p, lam) = (64.0 * o as f64, lead as f64);
    let w = if cancels { lam } else { 0.0 };
    // As ratios, so the cutoff compares equal terms.
    series(p, lam) / ((1.0 + w / p) * (p + w).log2()) >= series(ps, 0.0) / ps.log2()
}

// (f, k, (l, g)): T's and ln X's f fraction limbs (more past the cancellation for X in [1/2, 2)),
// each buffer's k limbs, q's lead l from θ3's tail at p = 64 (f + 1) bits, and the AGM's g
// fraction limbs, l bits past p for b1 ~ 2q. No (l, g) for X = 2^e.
fn agm_frame(nx: &LnNorm, o: usize, terms: usize) -> (usize, usize, Option<(u64, usize)>) {
    let Some(lead) = nx.lead() else {
        return (o + 1, o + 4, None);
    };
    let f = o
        + 1
        + if nx.cancels() {
            (lead + 2).div_ceil(64) as usize
        } else {
            0
        };
    let n1 = (terms as u64).saturating_add(1);
    let l = (64 * (f as u64 + 1) + 3)
        .div_ceil(n1.saturating_mul(n1).saturating_mul(4))
        .max(2);
    let g = f + 2 + l.div_ceil(64) as usize;
    (f, g + 2, Some((l, g)))
}

// ln X = c ln 2 - T into out, each buffer k limbs.
fn ln_agm_core(
    nx: &LnNorm,
    (f, frame): (usize, Option<(u64, usize)>),
    out: &mut [u64],
    [a, b, c, d]: [&mut [u64]; 4],
    theta: fn(&[u64], u64, &mut [u64], &mut [u64]) -> usize,
    (hi_mul, agm): (BinOp, fn(&[u64], &[u64], &mut [u64])),
    (pi, ln2, ratio): (fn(&mut [u64]), fn(&mut [u64]), Ratio),
) -> (i64, bool) {
    let x = nx.x;
    let cl = match frame {
        // X = 1
        None if nx.e == 0 => {
            out.fill(0);
            return (0, false);
        }
        None => {
            c[..f + 2].fill(0);
            nx.e
        }
        Some((l, g)) => {
            // θ2 and θ3, both times 2^(64 g).
            let z = ((l - 2) / 64) as usize;
            let (th2, th3) = (&mut a[..g - z], &mut b[..g + 1]);
            let zt = theta(x, bits(x) + l, th2, th3);
            debug_assert_eq!(zt, z);
            let th2 = &th2[..buf_len(th2)];
            // The AGM's first step: b1 = θ2 θ3 and a1 = (θ2 + θ3)² / 2 - b1.
            let (b1, s) = (&mut c[..g], &mut d[..g + 1]);
            mul_shr(th2, th3, 64 * g as u64, b1, hi_mul);
            add_buf(th3, th2);
            mul_shr(th3, th3, 64 * g as u64, s, hi_mul);
            shr_buf(s, 1);
            sub_buf(s, b1);
            // M = AGM(a1, b1), which stops on its own gap.
            let mm = &mut a[..g];
            agm(&s[..g], b1, mm);
            // T = π / (4M), its integer part on top.
            let pb = &mut b[..f + 2];
            pi(pb);
            let t = &mut c[..f + 2];
            t[f + 1] = 0;
            ratio(
                pb,
                &mm[..buf_len(mm)],
                64 * (f as i64 + 2 - g as i64) + 2,
                &mut t[..f + 1],
            );
            nx.e + 1 + l as i128
        }
    };
    ln2_combine(cl, false, &mut c[..f + 2], d, ln2, out)
}

// ln_dyn through the theta–AGM logarithm with T theta terms.
#[doc(hidden)]
pub fn ln_agm_dyn_at<const T: usize>(
    x: &[u64],
    m: i64,
    out: &mut [u64],
) -> Result<(i64, bool), LnError> {
    let nx = LnNorm::new(x, m).ok_or(LnError::Zero)?;
    if out.is_empty() {
        return Ok((0, false));
    }
    let (f, k, frame) = agm_frame(&nx, out.len(), T);
    let mut scratch = ScratchGuard::acquire();
    Ok(ln_agm_core(
        &nx,
        (f, frame),
        out,
        scratch.get_splits([k; 4]),
        theta_dyn::<T>,
        (hi_mul_dyn, agm_dyn),
        (pi_dyn, ln2_dyn, ratio_to_fraction_dyn),
    ))
}

// N is checked once X sets the frame, before out is touched.
#[doc(hidden)]
#[inline(never)]
pub fn ln_agm_static_at<const N: usize, const T: usize>(
    x: &[u64],
    m: i64,
    out: &mut [u64],
) -> Result<(i64, bool), LnError> {
    let nx = LnNorm::new(x, m).ok_or(LnError::Zero)?;
    if out.is_empty() {
        return Ok((0, false));
    }
    let (f, k, frame) = agm_frame(&nx, out.len(), T);
    capacity::<N>(k)?;
    let mut bufs = [[0; N]; 4];
    Ok(ln_agm_core(
        &nx,
        (f, frame),
        out,
        bufs.each_mut().map(|b| &mut b[..k]),
        |x, s, a, b| theta_static::<N, T>(x, s, a, b).expect("ln_agm checks N"),
        (hi_mul_static::<N>, |a, b, o| {
            agm_static::<N>(a, b, o).expect("ln_agm checks N")
        }),
        (
            |o| pi_static::<N>(o).expect("ln_agm checks N"),
            |o| ln2_static_unguarded::<N>(o).expect("ln_agm checks N"),
            ratio_to_fraction_static::<N>,
        ),
    ))
}
