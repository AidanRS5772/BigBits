use crate::utils::bin_split::{
    bin_split, static_split_stack, DynHyperCtx, DynNodeEngel, DynNodeHyper, EngelSeries,
    HyperSeries, Node, Shift, StaticHyperCtx, StaticNodeEngel, StaticNodeHyper,
};
use crate::utils::consts::consts::{ln2_dyn, ln2_static, LN2_LIMBS};
use crate::utils::div::{div_rem_dyn, div_rem_static};
use crate::utils::mul::{hi_mul_dyn, hi_mul_static};
use crate::utils::utils::{add_buf, buf_len, shl_buf, shr_buf, split_sh};
use crate::utils::{
    ScratchGuard, DYN_EXP_CHUNK_MIN_BITS, DYN_EXP_CHUNK_PAR_WORK, STATIC_EXP_CHUNK_MIN_BITS,
    STATIC_EXP_CHUNK_PAR_WORK,
};
use rayon::prelude::*;
use std::f64::consts::{LOG2_E, PI};
use std::{panic, thread};

const EXP_GUARD_LIMBS: usize = 1;
// r = X - n ln(2) scales ln(2)'s truncation error by n < 2^69, so two limbs
// below out keep r's error far under out's last bit.
const EXP_RED_GUARD_LIMBS: usize = 2;
const EXP_MAX_PAR_CHUNKS: usize = 64;

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

pub fn exp_terms(s: u64, w: usize) -> u64 {
    let c = (LOG2_E - PI.log2() - 1.0) * 0.5;
    let p = 64 * w as u64;
    // E(a) < p <= E(b): E(0) = 0 and E(n) >= n s.
    let (mut a, mut b) = (0, p.div_ceil(s));
    while b - a > 1 {
        let n = (a + b) / 2;
        let nf = n as f64;
        if n * s + ((nf + 0.5) * nf.log2() - nf * LOG2_E - c) as u64 >= p {
            b = n;
        } else {
            a = n;
        }
    }
    b
}

fn exp_zero_limbs(t: u64) -> usize {
    (t.saturating_sub(1) / 64) as usize
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
    Some((
        r,
        k - (64 * r.len() as u64 - r[r.len() - 1].leading_zeros() as u64),
        k,
    ))
}

fn exp_chunks(mut s: u64, end: u64, mut min: u64) -> impl Iterator<Item = (u64, u64)> {
    std::iter::from_fn(move || {
        let width = (s & !1)
            .max(2)
            .max(std::mem::take(&mut min))
            .min(end.saturating_sub(s));
        (width != 0).then(|| {
            s += width;
            (s - width, width)
        })
    })
}

fn exp_series<E, H>(
    r: &[u64],
    t: u64,
    k: u64,
    w: usize,
    out: &mut [u64],
    ctx: fn(&[u64], u64) -> H::Ctx,
) where
    E: Node<Ctx = Shift> + Send,
    H: Node + Send,
    H::Ctx: Sync,
{
    let (n, zeros) = (exp_terms(t, w), w - out.len());
    if r == [1] {
        bin_split::<E>(&Shift(k), 1, n + 1).finalize(&Shift(k), zeros, out);
    } else {
        let ctx = ctx(r, k);
        bin_split::<H>(&ctx, 1, n + 1).finalize(&ctx, zeros, out);
    }
}

fn exp_combine(
    a: &mut [u64],
    b: &[u64],
    z: usize,
    c: &mut [u64],
    hi_mul: fn(&[u64], &[u64], &mut [u64]) -> u64,
) {
    let o = b.len().saturating_sub(z);
    let mut carry = false;
    if o != 0 {
        c[o] = hi_mul(a, b, &mut c[..o]);
        carry = add_buf(a, &c[1..=o]);
    }
    if add_buf(a, b) | carry {
        // Only rounding can reach 2^-64z.
        a.fill(u64::MAX);
    }
}

pub fn exp_dyn_entry(x: &[u64], m: u64, out: &mut [u64]) -> u64 {
    assert!(x.last().is_some_and(|&l| l != 0));

    let e = 64 * (m + x.len() as u64);
    let s = 64 * m + x[x.len() - 1].leading_zeros() as u64;
    let z = exp_zero_limbs(s);
    let extra = (s != 0 && s % 64 == 0) as usize;
    let v = out.len() + EXP_GUARD_LIMBS + extra;
    let w = v + z;
    let end = (e - x[0].trailing_zeros() as u64).min(64 * w as u64);

    let slot = |width: u64| width.div_ceil(64) as usize + 1;
    let chunks = || exp_chunks(s + 1, end, DYN_EXP_CHUNK_MIN_BITS);
    let mut scratch = ScratchGuard::acquire();
    let [a, c, mut rs] = scratch.get_splits([v, v + 1, chunks().map(|(_, wd)| slot(wd)).sum()]);
    let mut series = Vec::new();
    for (s, width) in chunks() {
        let (r, rest) = std::mem::take(&mut rs).split_at_mut(slot(width));
        rs = rest;
        series.extend(chunk_bits(x, e, s, width, r));
    }

    let exp = |r: &[u64], t, k, out: &mut [u64]| {
        exp_series::<DynNodeEngel<Exp>, DynNodeHyper<Exp>>(r, t, k, w, out, DynHyperCtx::new)
    };
    let top = w - exp_zero_limbs(s + 1);
    a[top..].fill(0);
    exp(&[1], s + 1, s + 1, &mut a[..top]);
    let n = series.len();
    let par = n >= 2 && n * v >= DYN_EXP_CHUNK_PAR_WORK && rayon::current_num_threads() > 1;
    let mut results = ScratchGuard::acquire();
    let mut bs = results.get_split_vec(series.iter().map(|&(_, t, _)| w - exp_zero_limbs(t)));
    if par {
        series
            .par_iter()
            .zip(&mut bs)
            .for_each(|(&(r, t, k), b)| exp(r, t, k, b));
    } else {
        series
            .iter()
            .zip(&mut bs)
            .for_each(|(&(r, t, k), b)| exp(r, t, k, b));
    }
    for b in &bs {
        exp_combine(a, b, z, c, hi_mul_dyn);
    }

    let d = (extra == 1 && a[v - 1] == 0) as usize;
    out.copy_from_slice(&a[v - d - out.len()..v - d]);
    (z + d) as u64
}

fn exp_reduced(
    x: &[u64],
    m: i64,
    out: &mut [u64],
    [xr, l2, t]: [&mut [u64]; 3],
    ln2: fn(&mut [u64]),
    div_rem: fn(&mut [u64], &[u64], &mut [u64]) -> u64,
    exp_entry: fn(&[u64], u64, &mut [u64]) -> u64,
) -> Option<u64> {
    let (len, o) = (buf_len(x), out.len());
    if len == 0 {
        out.fill(0);
        if let Some(top) = out.last_mut() {
            *top = 1;
        }
        return Some(0);
    }
    let me = m as i128 - (x.len() - len) as i128;
    if me > 1 {
        // X >= 2^128 > 2^69 ln(2)
        return None;
    }
    let f = o + EXP_RED_GUARD_LIMBS;
    // xr = floor(X 2^(64f)), at most f + 2 limbs as me <= 1: x[i] lands at lo + i.
    xr.fill(0);
    let lo = f as i128 + me + 1 - len as i128;
    let skip = (-lo).clamp(0, len as i128) as usize;
    let start = lo.max(0) as usize;
    xr[start..start + len - skip].copy_from_slice(&x[skip..len]);
    ln2(l2);
    let mut q = [0; 3];
    let xl = buf_len(xr);
    // q = n, xr = r 2^(64f) with 0 <= r < ln(2).
    div_rem(&mut xr[..xl], l2, &mut q);
    if q[2] != 0 || q[1] >= 32 {
        // n >= 2^69, so n / 64 > i64::MAX
        return None;
    }
    let (k, b) = ((q[1] << 58) | (q[0] >> 6), q[0] & 63);

    // t = e^r 2^(64(o + 1)). e^r - 1's limbs past o + 1 never reach out.
    let r = &xr[..f];
    let rl = buf_len(r);
    t.fill(0);
    if rl != 0 {
        let rz = f - rl;
        let z = exp_zero_limbs(64 * rz as u64 + r[rl - 1].leading_zeros() as u64);
        if let Some(p) = (o + 1).checked_sub(z).filter(|&p| p != 0) {
            let off = exp_entry(&r[..rl], rz as u64, &mut t[..p]) as usize - z;
            debug_assert!(off <= 1);
            t.copy_within(off..p, 0);
            t[p - off..p].fill(0);
        }
    }
    t[o + 1] = 1;
    let carry = shl_buf(t, b as u8);
    debug_assert_eq!(carry, 0, "2^b e^r < 2^64");
    out.copy_from_slice(&t[2..]);
    Some(k)
}

pub fn exp_dyn(x: &[u64], m: i64, out: &mut [u64]) -> Option<u64> {
    let o = out.len();
    let mut scratch = ScratchGuard::acquire();
    let bufs = scratch.get_splits([o + 4, o + 2, o + 2]);
    exp_reduced(x, m, out, bufs, ln2_dyn, div_rem_dyn, exp_dyn_entry)
}

fn exp_series_static<const N: usize>(r: &[u64], t: u64, k: u64, w: usize, out: &mut [u64]) {
    exp_series::<StaticNodeEngel<N, Exp>, StaticNodeHyper<N, Exp>>(
        r,
        t,
        k,
        w,
        out,
        StaticHyperCtx::new,
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
    let (r, t, k) = chunk_bits(x, e, s, width, &mut r)?;
    let (mut b, bl) = ([0; N], w - exp_zero_limbs(t));
    exp_series_static::<N>(r, t, k, w, &mut b[..bl]);
    Some((b, bl))
}

pub fn exp_static_entry<const N: usize>(x: &[u64], m: u64, out: &mut [u64]) -> u64 {
    assert!(x.last().is_some_and(|&l| l != 0));
    assert!(out.len() + 3 <= N, "exp_static_entry needs out.len() + 3 <= N");

    let e = 64 * (m + x.len() as u64);
    let s = 64 * m + x[x.len() - 1].leading_zeros() as u64;
    let z = exp_zero_limbs(s);
    let extra = (s != 0 && s % 64 == 0) as usize;
    let v = out.len() + EXP_GUARD_LIMBS + extra;
    let w = v + z;
    let end = (e - x[0].trailing_zeros() as u64).min(64 * w as u64);

    let (mut a, mut c) = ([0; N], [0; N]);
    let a = &mut a[..v];
    let top = w - exp_zero_limbs(s + 1);
    let chunks = || exp_chunks(s + 1, end, STATIC_EXP_CHUNK_MIN_BITS);
    let chunk = |(s, width)| exp_chunk_static::<N>(x, e, s, width, w);
    let n = chunks().count();
    let par = n >= 2
        && n <= EXP_MAX_PAR_CHUNKS
        && n * v >= STATIC_EXP_CHUNK_PAR_WORK
        && thread::available_parallelism().is_ok_and(|p| p.get() > 1);
    if par {
        thread::scope(|sc| {
            let mut hs: [_; EXP_MAX_PAR_CHUNKS] = std::array::from_fn(|_| None);
            for (h, ch) in hs.iter_mut().zip(chunks()) {
                let builder = thread::Builder::new().stack_size(static_split_stack(N));
                *h = Some((ch, builder.spawn_scoped(sc, move || chunk(ch)).ok()));
            }
            exp_series_static::<N>(&[1], s + 1, s + 1, w, &mut a[..top]);
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
        exp_series_static::<N>(&[1], s + 1, s + 1, w, &mut a[..top]);
        for (b, bl) in chunks().filter_map(chunk) {
            exp_combine(a, &b[..bl], z, &mut c, hi_mul_static::<N>);
        }
    }

    let d = (extra == 1 && a[v - 1] == 0) as usize;
    out.copy_from_slice(&a[v - d - out.len()..v - d]);
    (z + d) as u64
}

pub fn exp_static<const N: usize>(x: &[u64], m: i64, out: &mut [u64]) -> Option<u64> {
    let o = out.len();
    // exp_static_entry writes up to o + 1 limbs, and ln(2) past its table is summed
    // at o + 2 limbs, each needing 3 spare limbs.
    assert!(o + 4 <= N, "exp_static needs out.len() + 4 <= N");
    assert!(
        o + EXP_RED_GUARD_LIMBS <= LN2_LIMBS || o + 5 <= N,
        "exp_static past ln(2)'s table needs out.len() + 5 <= N"
    );
    let (mut xr, mut l2, mut t) = ([0; N], [0; N], [0; N]);
    exp_reduced(
        x,
        m,
        out,
        [&mut xr[..o + 4], &mut l2[..o + 2], &mut t[..o + 2]],
        ln2_static::<N>,
        div_rem_static::<N>,
        exp_static_entry::<N>,
    )
}
