use rayon::prelude::*;
use std::f64::consts::{LOG2_E, PI};
use crate::utils::{ScratchGuard, DYN_EXP_CHUNK_MIN_BITS, DYN_EXP_CHUNK_PAR_WORK};
use crate::utils::bin_split::{
    bin_split, ratio_to_fraction_dyn, DynHyperCtx, DynNodeEngel, DynNodeHyper,
    EngelSeries, HyperSeries, Shift,
};
use crate::utils::mul::{hi_mul_dyn, mul_dyn};
use crate::utils::utils::{add_buf, buf_len, shr_buf, split_sh};

const EXP_GUARD_LIMBS: usize = 1;

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
    Some((r, k - (64 * r.len() as u64 - r[r.len() - 1].leading_zeros() as u64), k))
}

fn exp_chunks(mut s: u64, end: u64, mut min: u64) -> impl Iterator<Item = (u64, u64)> {
    std::iter::from_fn(move || {
        let width = (s & !1).max(2).max(std::mem::take(&mut min)).min(end.saturating_sub(s));
        (width != 0).then(|| {
            s += width;
            (s - width, width)
        })
    })
}

fn exp_series(r: &[u64], t: u64, k: u64, w: usize, out: &mut [u64]) {
    let n = exp_terms(t, w);
    let shift = (k * n) as i64 - 64 * (w - out.len()) as i64;
    if r == [1] {
        let node = bin_split::<DynNodeEngel<Exp>>(&Shift(k), 1, n + 1);
        ratio_to_fraction_dyn(&node.p, &node.q, shift, out);
    } else {
        let node = bin_split::<DynNodeHyper<Exp>>(&DynHyperCtx::new(r, k), 1, n + 1);
        let mut scratch = ScratchGuard::acquire();
        let pr = scratch.get(node.p.len() + r.len());
        mul_dyn(&node.p, r, pr);
        ratio_to_fraction_dyn(&pr[..buf_len(pr)], &node.q, shift, out);
    }
}

fn exp_combine(a: &mut [u64], b: &[u64], z: usize, c: &mut [u64]) {
    let o = b.len().saturating_sub(z);
    let mut carry = false;
    if o != 0 {
        c[o] = hi_mul_dyn(a, b, &mut c[..o]);
        carry = add_buf(a, &c[1..=o]);
    }
    if add_buf(a, b) | carry {
        // Only rounding can reach 2^-64z.
        a.fill(u64::MAX);
    }
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct ExpCutoffs {
    pub chunk_min_bits: u64,
    pub par_work: usize,
}

impl ExpCutoffs {
    pub const DEFAULT: Self = ExpCutoffs {
        chunk_min_bits: DYN_EXP_CHUNK_MIN_BITS,
        par_work: DYN_EXP_CHUNK_PAR_WORK,
    };
}

pub fn exp_dyn(x: &[u64], m: u64, out: &mut [u64]) -> u64 {
    exp_dyn_at(x, m, out, ExpCutoffs::DEFAULT)
}

#[doc(hidden)]
pub fn exp_dyn_at(x: &[u64], m: u64, out: &mut [u64], cut: ExpCutoffs) -> u64 {
    assert!(x.last().is_some_and(|&l| l != 0));

    let e = 64 * (m + x.len() as u64);
    let s = 64 * m + x[x.len() - 1].leading_zeros() as u64;
    let z = exp_zero_limbs(s);
    let extra = (s != 0 && s % 64 == 0) as usize;
    let v = out.len() + EXP_GUARD_LIMBS + extra;
    let w = v + z;
    let end = (e - x[0].trailing_zeros() as u64).min(64 * w as u64);

    let slot = |width: u64| width.div_ceil(64) as usize + 1;
    let chunks = || exp_chunks(s + 1, end, cut.chunk_min_bits);
    let mut scratch = ScratchGuard::acquire();
    let [a, c, mut rs] = scratch.get_splits([v, v + 1, chunks().map(|(_, wd)| slot(wd)).sum()]);
    let mut series = Vec::new();
    for (s, width) in chunks() {
        let (r, rest) = std::mem::take(&mut rs).split_at_mut(slot(width));
        rs = rest;
        series.extend(chunk_bits(x, e, s, width, r));
    }

    exp_series(&[1], s + 1, s + 1, w, a);
    let n = series.len();
    let par = cut.par_work == 0
        || (n >= 2 && n * v >= cut.par_work && rayon::current_num_threads() > 1);
    let mut results = ScratchGuard::acquire();
    let mut bs = results.get_split_vec(series.iter().map(|&(_, t, _)| w - exp_zero_limbs(t)));
    if par {
        series.par_iter().zip(&mut bs).for_each(|(&(r, t, k), b)| exp_series(r, t, k, w, b));
    } else {
        series.iter().zip(&mut bs).for_each(|(&(r, t, k), b)| exp_series(r, t, k, w, b));
    }
    for b in &bs {
        exp_combine(a, b, z, c);
    }

    // With the slack limb, e^x - 1 may have one more leading zero limb.
    let d = (extra == 1 && a[v - 1] == 0) as usize;
    out.copy_from_slice(&a[v - d - out.len()..v - d]);
    (z + d) as u64
}
