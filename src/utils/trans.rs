use std::f64::consts::{E, PI};
use std::sync::OnceLock;
use crate::utils::ScratchGuard;
use crate::utils::bin_split::{
    bin_split, ratio_to_fraction_dyn, DynHyperCtx, DynNodeEngel, DynNodeHyper, EngelSeries,
    HyperSeries, Shift,
};
use crate::utils::mul::{hi_mul_dyn, mul_dyn};
use crate::utils::utils::{add_buf, buf_len, shr_buf, split_sh};

// R ~ remainder series function
// N ~ number of terms
// $E(N, s) = -log_2(R_{N}(2^{-s}))$
fn minimize_terms(e: fn(u64, u64) -> u64, s: u64, p: u64) -> u64 {
    let p = 64 * p;
    let mut n = 1;
    while e(n, s) < p {
        n <<= 1;
    }
    let (mut a, mut b) = (n / 2, n);
    while b - a > 1 {
        let m = (a + b) / 2;
        if e(m, s) >= p {
            b = m;
        } else {
            a = m;
        }
    }
    return b;
}

// out <- bits [idx, min(idx + sz, end)) of buf as an integer, with bits counted
// down from buf's top bit and bits at or past end treated as zero.
// returns length of effective buffer
fn bit_chunks(idx: u64, end: u64, sz: u64, buf: &[u64], out: &mut [u64]) -> usize {
    debug_assert!(sz <= 64 * (out.len() as u64));
    let total = 64 * buf.len() as u64;
    let hi = (idx + sz).min(end).min(total);
    if idx >= hi {
        return 0;
    }

    // The chunk's lowest bit sits at little-endian bit total - hi.
    let (l, sh) = split_sh(total - hi);
    let (n, top) = split_sh(hi - idx);
    let len = n + (top != 0) as usize;
    for i in 0..len {
        let next = buf.get(l + i + 1).copied().unwrap_or(0);
        out[i] = if sh == 0 {
            buf[l + i]
        } else {
            buf[l + i] >> sh | next << (64 - sh)
        };
    }
    if top != 0 {
        out[len - 1] &= (1 << top) - 1;
    }
    buf_len(&out[..len])
}

struct FirstExp;
impl EngelSeries for FirstExp {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
}

struct Exp;
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


fn exp_err(n: u64, s: u64) -> u64 {
    static CONSTS: OnceLock<(f64, f64)> = OnceLock::new();
    let (exp_err1, exp_err2) = *CONSTS.get_or_init(|| {
        let e1 = E.log2();
        let e2 = (e1 - PI.log2() - 1.0) * 0.5;
        (e1, e2)
    });
    let nf = n as f64;
    n * s
        + ((nf + 0.5) * nf.log2() - nf * exp_err1 - exp_err2)
            .max(0.0)
            .floor() as u64
}

const GAURD_TERMS: u64 = 1;
const EXP_GUARD_LIMBS: usize = 1;

fn exp_zero_limbs(t: u64) -> usize {
    (t.saturating_sub(1) / 64) as usize
}

fn first_exp_series_dyn(k: u64, w: usize, out: &mut [u64]) {
    let ctx = Shift(k);
    let n = minimize_terms(exp_err, k, w as u64);
    let node = bin_split::<DynNodeEngel<FirstExp>>(&ctx, 1, n + GAURD_TERMS);
    let z = (w - out.len()) as i64;
    ratio_to_fraction_dyn(&node.p, &node.q, (k * (node.b - node.a)) as i64 - 64 * z, out);
}

fn exp_series_dyn(t: u64, k: u64, r: &[u64], w: usize, out: &mut [u64]) {
    let ctx = DynHyperCtx::new(r, k);
    let n = minimize_terms(exp_err, t, w as u64);
    let node = bin_split::<DynNodeHyper<Exp>>(&ctx, 1, n + GAURD_TERMS);
    let mut scratch = ScratchGuard::acquire();
    let pr = scratch.get(node.p.len() + r.len());
    mul_dyn(&node.p, r, pr);
    let pr = &pr[..buf_len(pr)];
    let z = (w - out.len()) as i64;
    ratio_to_fraction_dyn(pr, &node.q, (k * (node.b - node.a)) as i64 - 64 * z, out);
}

// (1 + a)(1 + b) = 1 + (a + b + ab). a and b hold A and B at scale 2^-64w with their top
// limbs left off: z of them for a, more for b. c needs b.len() - z + 1 limbs.
fn exp_combine(a: &mut [u64], b: &[u64], z: usize, c: &mut [u64]) {
    let bl = b.len();
    let mut carry = false;
    if bl > z {
        let o = bl - z;
        c[o] = hi_mul_dyn(a, b, &mut c[..o]);
        carry = add_buf(a, &c[1..=o]);
    }
    carry |= add_buf(a, b);
    if carry {
        // Only rounding can reach 2^-64z.
        a.fill(u64::MAX);
    }
}

// input x : 0 <= x < ln(2), as x * 2^(64 (x.len() + m)) with the m leading zero limbs omitted.
// out <- e^x - 1 with its z leading zero limbs omitted, returning z:
// e^x = 1 + out * 2^(-64 (out.len() + z)), and out's top limb is nonzero.
pub fn exp_dyn(x: &[u64], m: u64, out: &mut [u64]) -> u64 {
    assert!(x.last() != Some(&0));

    let base = 64 * m;
    let bit_start = x.last().copied().unwrap().leading_zeros() as u64;
    let bit_end = 64 * x.len() as u64 - x.first().copied().unwrap().trailing_zeros() as u64;
    let mut s = base + bit_start;
    let z = exp_zero_limbs(s);
    let extra = (s != 0 && s % 64 == 0) as usize;
    let v = out.len() + EXP_GUARD_LIMBS + extra;
    let w = v + z;
    let end = (base + bit_end).min(64 * w as u64);

    let mut scratch = ScratchGuard::acquire();
    let [a, b, c, chunk] = scratch.get_splits([v, v, v + 1, x.len().min(v)]);
    first_exp_series_dyn(s + 1, w, a);
    s += 1;

    // A chunk's term count depends on its position, not its width, so start the doubling at
    // the chunk's own offset: [s, ~2s), [~2s, ~4s), ...
    let mut sz = (s / 2).max(1);
    while s < end {
        let width = (2 * sz).min(end - s);
        let len = bit_chunks(s - base, end - base, width, x, chunk);
        if len != 0 {
            let skip = chunk[..len].iter().take_while(|&&l| l == 0).count();
            let r = &mut chunk[skip..len];
            let tz = r[0].trailing_zeros();
            shr_buf(r, tz as u8);
            let r = &r[..buf_len(r)];
            let k = s + width - 64 * skip as u64 - tz as u64;
            let t = k - (64 * r.len() as u64 - r[r.len() - 1].leading_zeros() as u64);
            let bl = w - exp_zero_limbs(t);
            exp_series_dyn(t, k, r, w, &mut b[..bl]);
            exp_combine(a, &b[..bl], z, c);
        }
        sz *= 2;
        s += sz;
    }

    let (top, z) = if extra == 1 && a[v - 1] == 0 {
        (v - 1, z + 1)
    } else {
        (v, z)
    };
    out.copy_from_slice(&a[top - out.len()..top]);
    z as u64
}

