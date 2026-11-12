use std::{fmt, marker::PhantomData};

use crate::tuning::k;
use crate::utils::{
    div::{div_dyn, div_prim, div_prim2, dyn_div_best, rcp_static, DivEq},
    mul::{
        hi_mul_dyn, hi_mul_static, mul_add_prim, mul_dyn, mul_prim, mul_prim2, mul_static, mul_work,
        school_at, MulEq,
    },
    utils::{add_buf, add_mul, add_prim, buf_len, inc_buf, shl_buf, shr_buf, split_sh, sub_mul},
    CapacityError, ScratchGuard,
};

pub(crate) fn join2<A: Send, B: Send>(
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

// Pushes carry c, or once buf is full keeps its top 64 buf.len() - 1 bits (a spare bit for
// a term) and raises the bit exponent e by what it drops.
fn push_or_drop(buf: &mut [u64], len: &mut usize, e: &mut u64, c: u64) {
    if c == 0 {
        return;
    }
    if *len < buf.len() {
        buf[*len] = c;
        *len += 1;
    } else {
        let k = bit_len(c) + 1;
        *len = shr_bits(buf, *len, c, k);
        *e += k;
    }
}

// (ov : buf[..len]) >> k, returning its length.
fn shr_bits(buf: &mut [u64], mut len: usize, ov: u64, k: u64) -> usize {
    if k == 0 {
        push(buf, &mut len, ov);
        return len;
    }
    let (kl, kb) = split_sh(k);
    if kl > len {
        buf[..len].fill(0);
        return 0;
    }
    if kl == 0 {
        if len != 0 {
            shr_buf(&mut buf[..len], kb);
            buf[len - 1] |= ov << (64 - kb);
        }
        if ov >> kb != 0 {
            buf[len] = ov >> kb;
            return len + 1;
        }
    } else {
        buf.copy_within(kl..len, 0);
        buf[len - kl] = ov;
        buf[len - kl + 1..len].fill(0);
        shr_buf(&mut buf[..=len - kl], kb);
    }
    buf_len(&buf[..len])
}

fn shl_bits(buf: &mut [u64], mut len: usize, k: u8) -> usize {
    let c = shl_buf(&mut buf[..len], k);
    push(buf, &mut len, c);
    len
}

#[inline(always)]
fn add_pass(
    p: &mut [u64],
    len: &mut usize,
    e: &mut u64,
    pn: &mut bool,
    (off, x, m): (usize, &[u64], u64),
    neg: bool,
    signed: bool,
) -> bool {
    if x.is_empty() || m == 0 {
        return false;
    }
    *len = (*len).max(off + x.len());
    let c = if signed && neg != *pn {
        sub_mul_flip(p, len, off, x, m, pn)
    } else {
        add_mul(&mut p[off..*len], x, m)
    };
    let e0 = *e;
    push_or_drop(p, len, e, c);
    *e != e0
}

// p += w x 2^d, rounding the bits below p's bit 0.
fn add_term(
    p: &mut [u64],
    len: &mut usize,
    e: &mut u64,
    pn: &mut bool,
    (x, d): (&[u64], i64),
    w: u64,
    neg: bool,
    signed: bool,
) {
    let mut pass = |off, x, m| add_pass(p, len, e, pn, (off, x, m), neg, signed);
    let mulhi = |a: u64, b: u64| ((a as u128 * b as u128) >> 64) as u64;
    let (l, s) = split_sh(d.unsigned_abs());
    if d >= 0 {
        let m = (w as u128) << s;
        pass(l, x, m as u64);
        pass(l + 1, x, (m >> 64) as u64);
        return;
    }
    // g, the limb below x's part, rounds the bits it drops.
    let g = x.get(l.wrapping_sub(1)).copied().unwrap_or(0);
    let x = &x[l.min(x.len())..];
    if s == 0 {
        if !pass(0, x, w) {
            pass(0, &[mulhi(g, w)], 1);
        }
        return;
    }
    let m = (w as u128) << (64 - s);
    let (hi, lo) = ((m >> 64) as u64, m as u64);
    let x0 = x.first().copied().unwrap_or(0);
    let low = mulhi(x0, lo) as u128 + mulhi(g, hi) as u128;
    pass(0, x, hi);
    pass(0, x.get(1..).unwrap_or(&[]), lo);
    pass(
        0,
        &[low as u64, (low >> 64) as u64][..1 + (low >> 64 != 0) as usize],
        1,
    );
}

// Drops P's low bits until P plus a term below 2^top fits p, as add_term needs.
fn make_room(p: &mut [u64], len: &mut usize, e: &mut u64, top: u64) {
    let cap = *e + 64 * p.len() as u64;
    if top.max(*e + 64 * *len as u64) < cap {
        return;
    }
    let need = top.max(*e + bits(&p[..*len])) + 1;
    if need > cap {
        *len = shr_bits(p, *len, 0, need - cap);
        *e += need - cap;
    }
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
fn add_or_sub(acc: u128, t: u128, sub: bool) -> (u128, bool) {
    if sub {
        (acc.abs_diff(t), t > acc)
    } else {
        (acc + t, false)
    }
}

// buf = (buf u + add) 2^sh, dropping low bits like push_or_drop.
fn mul_shl(buf: &mut [u64], len: &mut usize, e: &mut u64, u: u64, sh: u64, add: u64) {
    let sb = (sh % 64) as u32;
    let f = if u.leading_zeros() >= sb { sb } else { 0 };
    let bits = sh - f as u64;
    let shift = *len != 0 && bits != 0;
    let c = mul_add_prim(&mut buf[..*len], u << f, add);
    push_or_drop(buf, len, e, c);
    if shift {
        shift_up(buf, len, e, bits);
    }
}

// buf = buf 2^bits for a nonzero buf, dropping low bits like push_or_drop.
#[inline(always)]
fn shift_up(buf: &mut [u64], len: &mut usize, e: &mut u64, bits: u64) {
    let free = 64 * (buf.len() - *len) as u64;
    let s = if bits <= free {
        bits
    } else {
        let top = buf[*len - 1];
        let room = free
            + if top == 0 {
                0
            } else {
                top.leading_zeros() as u64
            };
        if bits <= room {
            bits
        } else {
            room.saturating_sub(1)
        }
    };
    *e += bits - s;
    let (sl, sb) = split_sh(s);
    if sl != 0 {
        buf.copy_within(..*len, sl);
        buf[..sl].fill(0);
        *len += sl;
    }
    let c = shl_buf(&mut buf[..*len], sb);
    push(buf, len, c);
}

fn push2(buf: &mut [u64], len: &mut usize, c: u128) {
    if c != 0 {
        buf[*len] = c as u64;
        *len += 1;
        push(buf, len, (c >> 64) as u64);
    }
}

fn push_or_drop2(buf: &mut [u64], len: &mut usize, e: &mut u64, c: u128) {
    let (lo, hi) = (c as u64, (c >> 64) as u64);
    if hi == 0 {
        return push_or_drop(buf, len, e, lo);
    }
    if *len < buf.len() {
        buf[*len] = lo;
        *len += 1;
    } else {
        // Drop a whole limb so that hi is the limb right above.
        shr_bits(buf, *len, lo, 64);
        *e += 64;
    }
    push_or_drop(buf, len, e, hi);
}

fn mul_shl_wide(buf: &mut [u64], len: &mut usize, e: &mut u64, u: u128, sh: u64) {
    let shift = *len != 0 && sh != 0;
    let c = mul_prim2(&mut buf[..*len], u);
    push_or_drop2(buf, len, e, c);
    if shift {
        shift_up(buf, len, e, sh);
    }
}

// add_term for a wide w and x 2^ex, one limb of w at a time. Callers make room first.
fn add_term_wide(
    p: &mut [u64],
    len: &mut usize,
    e: &mut u64,
    pn: &mut bool,
    (x, ex): (&[u64], u64),
    w: u128,
    neg: bool,
    signed: bool,
) {
    let lo = (x, ex as i64 - *e as i64);
    add_term(p, len, e, pn, lo, w as u64, neg, signed);
    if w >> 64 != 0 {
        let hi = (x, (ex + 64) as i64 - *e as i64);
        add_term(p, len, e, pn, hi, (w >> 64) as u64, neg, signed);
    }
}

fn bbp_leaf_terms<S: BBPSeries>(
    a: u64,
    b: u64,
    shift: u64,
    p: &mut [u64],
    q: &mut [u64],
) -> (usize, usize, u64, bool) {
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
        let qv = (0, &q[..q_len], v);
        add_pass(p, &mut p_len, &mut e, &mut pn, qv, vn, S::SIGNED);
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
) -> (usize, usize, u64, bool) {
    let shr = |v: u64, e: u64| if e < 64 { v >> e } else { 0 };
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
        if p_len == 0 {
            e = 0;
            pn = S::SIGNED && vn;
        }
        let opposite = S::SIGNED && vn != pn;
        let fused = !opposite && (p_len == 0 || (sh < 64 && u.leading_zeros() as u64 >= sh));
        // v lands at P's exponent, before the shift when fused and after it otherwise.
        let add = if fused { shr(v, e) } else { 0 };
        mul_shl(p, &mut p_len, &mut e, u, sh, add);
        let v = shr(v, e);
        if !fused && v != 0 {
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
    (buf_len(&p[..p_len]), q_len, e, pn)
}

fn hyper_leaf_terms<S: HyperSeries>(
    a: u64,
    b: u64,
    shift: u64,
    p: &mut [u64],
    q: &mut [u64],
    r: &mut [u64],
) -> (usize, usize, usize, u64, bool) {
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
        if p_len == 0 {
            e = 0;
        }
        mul_shl(p, &mut p_len, &mut e, u, shift * j, 0);
        if e % 64 != 0 {
            make_room(p, &mut p_len, &mut e, bits(&r[..r_len]) + bit_len(w));
        }
        let rs = (&r[..r_len], -(e as i64));
        add_term(p, &mut p_len, &mut e, &mut pn, rs, w, wn, S::SIGNED);
        let c = mul_prim(&mut r[..r_len], v);
        push(r, &mut r_len, c);
        let c = mul_prim(&mut q[..q_len], u);
        push(q, &mut q_len, c);
    }
    (buf_len(&p[..p_len]), q_len, r_len, e, pn)
}

// The x != 1 leaf, one term per step, with Q into q. t_mul(t, x, et, e, out) puts
// t x 2^et in out as (len, exponent), keeping at least its bits from P's exponent e up.
// A WIDE leaf's q may need 128 bits.
fn hyper_x_leaf_terms<'t, S: WideHyperSeries, const WIDE: bool>(
    a: u64,
    b: u64,
    (shift, x): (u64, &[u64]),
    p: &mut [u64],
    q: &mut [u64],
    mut t: &'t mut [u64],
    mut tmp: &'t mut [u64],
    t_mul: impl Fn(&[u64], &[u64], u64, u64, &mut [u64]) -> (usize, u64),
) -> (usize, usize, u64, &'t [u64], u64, bool) {
    t[0] = 1;
    let (mut p_len, mut e, mut t_len, mut et, mut pn) = (0, 0, 1, 0, false);
    for n in (a..b).map(u128::from) {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        if p_len == 0 {
            e = 0;
        }
        if WIDE {
            mul_shl_wide(p, &mut p_len, &mut e, S::q(n), shift);
        } else {
            mul_shl(p, &mut p_len, &mut e, S::q(n) as u64, shift, 0);
        }
        let w = S::p(n);
        if t_len != 0 {
            let top = et + bits(&t[..t_len]) + bits(x) + bit_len128(w);
            make_room(p, &mut p_len, &mut e, top);
        }
        (t_len, et) = t_mul(&t[..t_len], x, et, e, tmp);
        std::mem::swap(&mut t, &mut tmp);
        if t_len != 0 {
            let ts = (&t[..t_len], et);
            add_term_wide(p, &mut p_len, &mut e, &mut pn, ts, w, S::neg(n), S::SIGNED);
        }
        let c = mul_prim2(&mut t[..t_len], S::r(n));
        push_or_drop2(t, &mut t_len, &mut et, c);
    }
    let mut q_len = 1;
    q[0] = 1;
    for n in a as u128..b as u128 {
        let c = mul_prim2(&mut q[..q_len], S::q(n));
        push2(q, &mut q_len, c);
    }
    let t: &'t [u64] = t;
    (buf_len(&p[..p_len]), q_len, e, &t[..t_len], et, pn)
}

// The wide kernels take one term per step: wide values seldom leave room to pack two.
fn bbp_leaf_terms_wide<S: WideBBPSeries>(
    a: u64,
    b: u64,
    shift: u64,
    p: &mut [u64],
    q: &mut [u64],
) -> (usize, usize, u64, bool) {
    q[0] = 1;
    let (mut p_len, mut q_len, mut e, mut pn) = (0, 1, 0, false);
    for n in a as u128..b as u128 {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        let (u, v) = (S::q(n), S::p(n));
        mul_shl_wide(p, &mut p_len, &mut e, u, shift);
        let qs = (&q[..q_len], 0);
        add_term_wide(p, &mut p_len, &mut e, &mut pn, qs, v, S::neg(n), S::SIGNED);
        let c = mul_prim2(&mut q[..q_len], u);
        push2(q, &mut q_len, c);
    }
    (buf_len(&p[..p_len]), q_len, e, pn)
}

fn engel_leaf_terms_wide<S: WideEngelSeries>(
    a: u64,
    b: u64,
    shift: u64,
    p: &mut [u64],
    q: &mut [u64],
) -> (usize, usize, u64, bool) {
    q[0] = 1;
    let (mut p_len, mut q_len, mut e, mut pn) = (0, 1, 0, false);
    for n in a as u128..b as u128 {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        let (u, v, vn) = (S::q(n), S::p(n), S::SIGNED && S::neg(n));
        if p_len == 0 {
            (e, pn) = (0, vn);
        }
        mul_shl_wide(p, &mut p_len, &mut e, u, shift);
        make_room(p, &mut p_len, &mut e, bit_len128(v));
        add_term_wide(p, &mut p_len, &mut e, &mut pn, (&[1], 0), v, vn, S::SIGNED);
        let c = mul_prim2(&mut q[..q_len], u);
        push2(q, &mut q_len, c);
    }
    (buf_len(&p[..p_len]), q_len, e, pn)
}

fn hyper_leaf_terms_wide<S: WideHyperSeries>(
    a: u64,
    b: u64,
    shift: u64,
    p: &mut [u64],
    q: &mut [u64],
    r: &mut [u64],
) -> (usize, usize, usize, u64, bool) {
    (q[0], r[0]) = (1, 1);
    let (mut p_len, mut q_len, mut r_len, mut e, mut pn) = (0, 1, 1, 0, false);
    for n in a as u128..b as u128 {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        let (u, v, w) = (S::q(n), S::r(n), S::p(n));
        if p_len == 0 {
            e = 0;
        }
        mul_shl_wide(p, &mut p_len, &mut e, u, shift);
        make_room(p, &mut p_len, &mut e, bits(&r[..r_len]) + bit_len128(w));
        let rs = (&r[..r_len], 0);
        add_term_wide(p, &mut p_len, &mut e, &mut pn, rs, w, S::neg(n), S::SIGNED);
        let c = mul_prim2(&mut r[..r_len], v);
        push2(r, &mut r_len, c);
        let c = mul_prim2(&mut q[..q_len], u);
        push2(q, &mut q_len, c);
    }
    (buf_len(&p[..p_len]), q_len, r_len, e, pn)
}

fn fits_u64(a: u64, b: u64, f: impl Fn(u128) -> u128) -> bool {
    (a..b).all(|n| f(n as u128) >> 64 == 0)
}

fn leaf_lens(narrow: bool, terms: usize, words: usize) -> (usize, usize) {
    if narrow {
        (terms + words, terms)
    } else {
        (2 * terms + words + 3, 2 * terms + 1)
    }
}

fn mul_exact(a: &[u64], b: &[u64], out: &mut [u64]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let n = a.len() + b.len();
    mul_dyn(a, b, &mut out[..n]);
    buf_len(&out[..n])
}

fn product(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut v = vec![0; a.len() + b.len()];
    let n = mul_exact(a, b, &mut v);
    v.truncate(n);
    v
}

pub(crate) fn bit_len(x: u64) -> u64 {
    (64 - x.leading_zeros()) as u64
}

fn bit_len128(x: u128) -> u64 {
    (128 - x.leading_zeros()) as u64
}

// Bits of x, whose top limb is nonzero.
pub(crate) fn bits(x: &[u64]) -> u64 {
    x.last()
        .map_or(0, |&top| 64 * x.len() as u64 - top.leading_zeros() as u64)
}

fn prod_fits(a: u64, b: u64, f: impl Fn(u128) -> u128, cap: u64) -> bool {
    let mut bits = 0;
    for n in a..b {
        bits += bit_len128(f(n as u128));
        if bits > cap {
            return false;
        }
    }
    true
}

// A static split thread's stack: about 1 KiB per limb of N plus 1 MiB, sized for optimized code at
// its tree's depth. Unoptimized frames are several times larger, which overflowed debug builds'
// threads on the deepest static trees (one-term Hyper leaves, found by the random-knob test; the
// release build of the same call fits), so debug builds reserve four times as much. Stacks are
// reserved lazily, so the unused part costs address space, not memory.
pub(crate) const fn static_split_stack(n: usize) -> usize {
    let s = 1024 * n + (1 << 20);
    if cfg!(debug_assertions) {
        4 * s
    } else {
        s
    }
}

// Static nodes keep each part as m 2^e, m in N limbs and e in bits: exact while it fits N
// limbs, and afterwards its top 64 N bits (P a bit or two fewer, for carries).

// a b's limbs from its bit e up, for trimmed a and b, as (len, ov, e') for
// (ov : out[..len]) 2^e' with e' <= e. Its bits from e up must fit N + 1 limbs.
fn hi_mul_from<const N: usize>(a: &[u64], b: &[u64], e: u64, out: &mut [u64]) -> (usize, u64, u64) {
    let n = a.len() + b.len();
    let d = ((e / 64) as usize).min(n - 2);
    if n - d <= N + 1 {
        let ov = hi_mul_static::<N>(a, b, &mut out[..n - d - 1]);
        return (n - d - 1, ov, 64 * d as u64);
    }
    // Then a b's top limb is zero, so normalizing the operand with more leading zeros
    // moves the window down a limb.
    let lz = |x: &[u64]| x[x.len() - 1].leading_zeros() as u64;
    let (x, y) = if lz(a) >= lz(b) { (a, b) } else { (b, a) };
    let z = lz(x);
    let d = (((e + z) / 64) as usize).min(n - 2);
    debug_assert!(n - d <= N + 1 && 64 * d as u64 >= z);
    let ov = hi_mul_shifted::<N>(x, y, &mut out[..n - d - 1]);
    (n - d - 1, ov, 64 * d as u64 - z)
}

// hi_mul_static of x << lz(x) and y, with x's copy out of the caller's frame.
#[cold]
#[inline(never)]
fn hi_mul_shifted<const N: usize>(x: &[u64], y: &[u64], out: &mut [u64]) -> u64 {
    let mut s = [0; N];
    let s = &mut s[..x.len()];
    s.copy_from_slice(x);
    shl_buf(s, x[x.len() - 1].leading_zeros() as u8);
    hi_mul_static::<N>(s, y, out)
}

// a b as (len, e), with its top bit at 64 N - 1 once it doesn't fit N limbs.
fn hi_mul_norm<const N: usize>(a: &[u64], b: &[u64], out: &mut [u64]) -> (usize, u64) {
    if a.is_empty() || b.is_empty() {
        return (0, 0);
    }
    let (top, cap) = (bits(a) + bits(b), 64 * N as u64);
    if top <= cap {
        let m = (a.len() + b.len()).min(N);
        let c = mul_static::<N>(a, b, &mut out[..m]);
        debug_assert_eq!(c, 0);
        return (buf_len(&out[..m]), 0);
    }
    // a b has top or top - 1 bits, so this window holds 64 N of them or more.
    let (len, ov, e) = hi_mul_from::<N>(a, b, top - 1 - cap, out);
    let win = if ov != 0 {
        64 * len as u64 + bit_len(ov)
    } else {
        bits(&out[..len])
    };
    (shr_bits(out, len, ov, win - cap), e + win - cap)
}

// floor(a b 2^(ea - e)) into out[off..off + len] above off zero limbs, returning
// (off, len). It must fit N limbs.
fn product_at<const N: usize>(
    a: &[u64],
    b: &[u64],
    ea: u64,
    e: u64,
    out: &mut [u64],
) -> (usize, usize) {
    if a.is_empty() || b.is_empty() {
        return (0, 0);
    }
    if ea >= e {
        let (off, s) = split_sh(ea - e);
        out[..off].fill(0);
        let out = &mut out[off..];
        let m = (a.len() + b.len()).min(out.len());
        let c = mul_static::<N>(a, b, &mut out[..m]);
        debug_assert_eq!(c, 0);
        (off, shl_bits(out, buf_len(&out[..m]), s))
    } else {
        let (len, ov, er) = hi_mul_from::<N>(a, b, e - ea, out);
        (0, shr_bits(out, len, ov, e - ea - er))
    }
}

// One past the top bit a b 2^e can have, or 0 when it's zero.
fn term_top(a: &[u64], b: &[u64], e: u64) -> u64 {
    if a.is_empty() || b.is_empty() {
        0
    } else {
        e + bits(a) + bits(b)
    }
}

// A merge's P = T + U exponent, from its terms' exponents and top bits (0 for a zero term),
// before either product. P is exact while it fits N limbs, and otherwise keeps its top 64 N
// bits less up to two: a carry, and a product's top bit known only within one.
fn p_exponent<const N: usize>(ta: u64, ea: u64, tu: u64, eu: u64, same: bool) -> u64 {
    let low = |top: u64, e: u64| if top == 0 { u64::MAX } else { e };
    let low = low(ta, ea).min(low(tu, eu));
    if low == u64::MAX {
        return 0;
    }
    low.max((ta.max(tu) + same as u64).saturating_sub(64 * N as u64))
}

// P = T + U, with T in p[..len] (zero when t_zero) and U at limb offset off, each with
// its sign. Returns P's length and sign.
fn p_sum(
    p: &mut [u64],
    len: usize,
    (off, us): (usize, &[u64]),
    (tn, un): (bool, bool),
    t_zero: bool,
    signed: bool,
) -> (usize, bool) {
    let mut neg = if t_zero { un } else { tn };
    let mut len = len.max(off + us.len());
    let c = if signed && neg != un {
        sub_mul_flip(p, &mut len, off, us, 1, &mut neg)
    } else {
        add_buf(&mut p[off..len], us) as u64
    };
    push(p, &mut len, c);
    (buf_len(&p[..len]), neg)
}

// out = floor(p * 2^(64 out.len() - shift) / q) up to truncation error. Only the top
// out.len() + 2 limbs of p and q reach out, so the division is sized by out.
pub fn ratio_to_fraction_dyn(p: &[u64], q: &[u64], shift: i64, out: &mut [u64]) {
    let len = out.len();
    out.fill(0);
    if p.is_empty() || len == 0 {
        return;
    }

    let keep = len + 2;
    let (kp, kq) = (p.len().saturating_sub(keep), q.len().saturating_sub(keep));
    let (p, q) = (&p[kp..], &q[kq..]);
    let e = 64 * (len + kp) as i64 - 64 * kq as i64 - shift;

    let mut scratch = ScratchGuard::acquire();
    let n = if e >= 0 {
        let (w, v) = split_sh(e as u64);
        let n = scratch.get(w + p.len() + 1);
        n[..w].fill(0);
        n[w..w + p.len()].copy_from_slice(p);
        n[w + p.len()] = shl_buf(&mut n[w..w + p.len()], v);
        n
    } else {
        let (w, v) = split_sh(e.unsigned_abs());
        if w >= p.len() {
            return;
        }
        let n = scratch.get(p.len() - w);
        n.copy_from_slice(&p[w..]);
        shr_buf(n, v);
        n
    };
    let n = &n[..buf_len(n)];
    debug_assert!(n.len() <= q.len() + len, "finalize expects S < 1");
    let top = div_dyn(n, q, out);
    debug_assert_eq!(top, 0, "finalize expects S < 1");
}

// ratio_to_fraction_dyn without its 2 out.len()-limb numerator, which wouldn't fit N:
// p times a (len + 2)-limb reciprocal of q, whose error costs out under one ulp.
pub fn ratio_to_fraction_static<const N: usize>(p: &[u64], q: &[u64], shift: i64, out: &mut [u64]) {
    let len = out.len();
    assert!(len + 2 <= N, "finalize output needs out.len() + 2 <= N");
    out.fill(0);
    if p.is_empty() || len == 0 {
        return;
    }

    let (mut x, mut y) = ([0; N], [0; N]);
    let x = &mut x[..len + 2];
    rcp_static::<N>(q, x);
    let bits = 64 * (q.len() + x.len() - 1 - len) as i64 + shift;
    // At w = 1 the window below is the whole product, exactly.
    debug_assert!(bits >= 64, "finalize needs w >= 1 whole limb");
    let (w, v) = split_sh(bits as u64);
    let full = p.len() + x.len();
    if full < w {
        return;
    }

    // y = p x / 2^(64 (w - 1)), with hi_mul's overflow on top, zero once y has len + 2
    // limbs below it.
    let core = full - w;
    debug_assert!(core <= len + 2, "finalize expects S < 1");
    let top = hi_mul_static::<N>(p, x, &mut y[..core]);
    let y = if core < len + 2 {
        y[core] = top;
        &mut y[..=core]
    } else {
        debug_assert_eq!(top, 0, "finalize expects S < 1");
        &mut y[..core]
    };
    shr_buf(y, v);
    let y = &y[1..];
    let n = y.len().min(len);
    out[..n].copy_from_slice(&y[..n]);
    debug_assert!(y[n..].iter().all(|&l| l == 0), "finalize expects S < 1");
}

// Limbs of b - a terms that each add about bits(f(midpoint)) + shift bits.
fn est_limbs(a: u64, b: u64, f: impl Fn(u128) -> u128, shift: u64) -> u64 {
    let bits = bit_len128(f((a + (b - a) / 2) as u128)) + shift;
    (b - a).saturating_mul(bits).div_ceil(64)
}

// A BBP or Engel leaf kernel: (a, b, shift, p, q) -> (P's length, Q's, P's exponent, sign).
type LeafKernel = fn(u64, u64, u64, &mut [u64], &mut [u64]) -> (usize, usize, u64, bool);

// A dynamic BBP or Engel leaf's exact P and Q, and P's sign.
fn dyn_leaf(a: u64, b: u64, shift: u64, narrow: bool, k: LeafKernel) -> (Vec<u64>, Vec<u64>, bool) {
    let words = split_sh(shift * (b - a)).0 + 1;
    let (pl, ql) = leaf_lens(narrow, (b - a) as usize, words);
    let (mut p, mut q) = (vec![0; pl], vec![0; ql]);
    let (p_len, q_len, e, neg) = k(a, b, shift, &mut p, &mut q);
    debug_assert_eq!(e, 0);
    p.truncate(p_len);
    q.truncate(q_len);
    (p, q, neg)
}

// A dynamic merge's exact P = lp rq 2^sh + u and Q = lq rq, and P's sign.
fn dyn_merge(
    (lp, lq, ln): (&[u64], &[u64], bool),
    (u, rq, un): (&[u64], &[u64], bool),
    sh: u64,
    signed: bool,
) -> (Vec<u64>, Vec<u64>, bool) {
    let (sl, sb) = split_sh(sh);
    let pl = lp.len() + rq.len();
    let mut p = vec![0; (sl + pl + 1).max(u.len()) + 1];
    let par = lq.len() + rq.len() >= k::bin_split::dyn_merge_par()
        && rayon::current_thread_index().is_some();
    let (_, q) = join2(
        par,
        || mul_dyn(lp, rq, &mut p[sl..sl + pl]),
        || product(lq, rq),
    );
    p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);
    let n = p.len();
    let (len, neg) = p_sum(&mut p, n, (0, u), (ln, un), false, signed);
    p.truncate(len);
    (p, q, neg)
}

// A dynamic node's S = P / (Q 2^bits) into out, and whether it's negative.
fn dyn_finalize(p: &[u64], q: &[u64], neg: bool, bits: u64, zeros: usize, out: &mut [u64]) -> bool {
    ratio_to_fraction_dyn(p, q, bits as i64 - 64 * zeros as i64, out);
    neg && !p.is_empty()
}

pub trait Node: Sized {
    type Ctx;
    // A tree splits [a, b) while it has more than leaf_terms() terms and an estimated
    // leaf_limbs() limbs or more: the knobs `bin_split.<dyn|static>_<kind>_leaf_*`.
    fn leaf_limbs() -> u64;
    fn leaf_terms() -> u64 {
        1
    }
    const PARALLEL: bool = false;
    // Estimated limbs of the largest part of the node [a, b). It only shapes the tree.
    fn limbs(ctx: &Self::Ctx, a: u64, b: u64) -> u64;
    // The terms [a, b) this node sums.
    fn range(&self) -> (u64, u64);
    fn merge(ctx: &Self::Ctx, l: Self, r: Self) -> Self;
    fn leaf(ctx: &Self::Ctx, a: u64, b: u64, last: bool) -> Self;
    // out <- S 2^(64 (out.len() + zeros)), skipping S's top zeros limbs, which the
    // caller knows are zero. Returns whether S is negative.
    fn finalize(&self, ctx: &Self::Ctx, zeros: usize, out: &mut [u64]) -> bool;
}

// Where a tree stops splitting, and from what size a PARALLEL tree builds a split's halves in
// parallel (on a pool of two or more threads): a tree's knobs, read once and passed down it.
// A split started off the pool (the caller's own thread) has to wake it, about 14 us on the M1,
// so there it also needs `par_work` of limbs times terms: a 2-term tree at 1024 limbs took
// 15.8 us split against 2.1 us serial. On a pool worker the limb gate alone decides, as it was
// tuned through exp and ln, whose chunks build their trees on workers.
#[derive(Debug, Clone, Copy)]
pub struct SplitCutoffs {
    pub leaf_limbs: u64,
    pub leaf_terms: u64,
    pub par_limbs: u64,
    pub par_work: u64,
}

impl SplitCutoffs {
    pub fn of<N: Node>() -> Self {
        SplitCutoffs {
            leaf_limbs: N::leaf_limbs(),
            leaf_terms: N::leaf_terms(),
            par_limbs: k::bin_split::dyn_par(),
            par_work: k::bin_split::dyn_par_work(),
        }
    }

    // Whether a node of `limbs` estimated limbs over `terms` terms builds its halves in parallel.
    fn splits(&self, limbs: u64, terms: u64) -> bool {
        limbs >= self.par_limbs
            && rayon::current_num_threads() > 1
            && (limbs.saturating_mul(terms) >= self.par_work || rayon::current_thread_index().is_some())
    }
}

// bin_split_tree with explicit cutoffs, for tests and tuning.
#[doc(hidden)]
pub fn bin_split_tree_at<N>(ctx: &N::Ctx, a: u64, b: u64, last: bool, cut: SplitCutoffs) -> N
where
    N: Node + Send,
    N::Ctx: Sync,
{
    debug_assert!(a < b);
    if b - a <= cut.leaf_terms.max(1) {
        return N::leaf(ctx, a, b, last);
    }
    let limbs = N::limbs(ctx, a, b);
    if limbs < cut.leaf_limbs {
        return N::leaf(ctx, a, b, last);
    }
    let m = (a + b) / 2;
    let (l, r) = join2(
        N::PARALLEL && cut.splits(limbs, b - a),
        || bin_split_tree_at::<N>(ctx, a, m, false, cut),
        || bin_split_tree_at::<N>(ctx, m, b, last, cut),
    );
    N::merge(ctx, l, r)
}

pub fn bin_split_tree<N>(ctx: &N::Ctx, a: u64, b: u64, last: bool) -> N
where
    N: Node + Send,
    N::Ctx: Sync,
{
    bin_split_tree_at::<N>(ctx, a, b, last, SplitCutoffs::of::<N>())
}

pub fn bin_split<N>(ctx: &N::Ctx, a: u64, b: u64) -> N
where
    N: Node + Send,
    N::Ctx: Sync,
{
    bin_split_tree::<N>(ctx, a, b, true)
}

// Only merges drop bits, so a static leaf too big for N limbs splits itself.
fn split_leaf<T: Node>(ctx: &T::Ctx, a: u64, b: u64, last: bool) -> T {
    let m = a + (b - a) / 2;
    T::merge(ctx, T::leaf(ctx, a, m, false), T::leaf(ctx, m, b, last))
}

// A narrow series trait (u64 coefficients) and its wide twin (u128). Every narrow series is
// a wide one, and Narrow<S> views a wide series whose values fit u64 as a narrow one, for the
// narrow kernels. A type implements one of the two, and calling p or q on a concrete narrow
// series with both in scope needs <T as BBPSeries>::q.
macro_rules! series_traits {
    ($narrow:ident, $wide:ident, $($f:ident),+) => {
        pub trait $narrow {
            const SIGNED: bool = false;
            // The first term's index.
            const START: u64;
            // Some b with |R_n| < 2^-b for the tail R_n from term n, given a per-term decay (2^-k,
            // or |x 2^-k| for Hyper) of at most 2^-s. Any lower bound on -log2 |R_n| is valid (0
            // always is); a tighter one takes fewer terms.
            fn tail_bits(n: u64, s: i64) -> u64;
            $(fn $f(n: u64) -> u64;)+
            fn neg(_: u64) -> bool {
                false
            }
        }

        pub trait $wide {
            const SIGNED: bool = false;
            const START: u64;
            fn tail_bits(n: u64, s: i64) -> u64;
            $(fn $f(n: u128) -> u128;)+
            fn neg(_: u128) -> bool {
                false
            }
        }

        impl<S: $narrow> $wide for S {
            const SIGNED: bool = <S as $narrow>::SIGNED;
            const START: u64 = <S as $narrow>::START;
            fn tail_bits(n: u64, s: i64) -> u64 {
                <S as $narrow>::tail_bits(n, s)
            }
            $(fn $f(n: u128) -> u128 {
                <S as $narrow>::$f(n as u64) as u128
            })+
            fn neg(n: u128) -> bool {
                <S as $narrow>::neg(n as u64)
            }
        }

        impl<S: $wide> $narrow for Narrow<S> {
            const SIGNED: bool = <S as $wide>::SIGNED;
            const START: u64 = <S as $wide>::START;
            fn tail_bits(n: u64, s: i64) -> u64 {
                <S as $wide>::tail_bits(n, s)
            }
            $(fn $f(n: u64) -> u64 {
                <S as $wide>::$f(n as u128) as u64
            })+
            fn neg(n: u64) -> bool {
                <S as $wide>::neg(n as u128)
            }
        }
    };
}

struct Narrow<S>(PhantomData<fn() -> S>);

#[derive(Debug, Clone, Copy)]
pub struct Shift(pub u64);

//$\sum^{N}_{n = 1} (-1)^{\sigma(n)} \; \frac{P(n)}{Q(n)} \; 2^{-kn}$
// Summed from n = START, with term n weighted 2^(-k (n - START + 1)).
series_traits!(BBPSeries, WideBBPSeries, p, q);

#[derive(Debug, Clone)]
pub struct DynNodeBBP<S: WideBBPSeries> {
    pub a: u64,
    pub b: u64,
    pub p: Vec<u64>,
    pub q: Vec<u64>,
    // P's sign.
    pub neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<S: WideBBPSeries> Node for DynNodeBBP<S> {
    type Ctx = Shift;
    fn leaf_limbs() -> u64 {
        k::bin_split::dyn_bbp_leaf_limbs()
    }
    const PARALLEL: bool = true;

    fn range(&self) -> (u64, u64) {
        (self.a, self.b)
    }

    // P, about Q 2^(shift t).
    fn limbs(ctx: &Shift, a: u64, b: u64) -> u64 {
        est_limbs(a, b, S::q, ctx.0)
    }

    fn leaf(ctx: &Shift, a: u64, b: u64, _: bool) -> Self {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let k: LeafKernel = if narrow {
            bbp_leaf_terms::<Narrow<S>>
        } else {
            bbp_leaf_terms_wide::<S>
        };
        let (p, q, neg) = dyn_leaf(a, b, ctx.0, narrow, k);
        DynNodeBBP {
            a,
            b,
            p,
            q,
            neg,
            _s: PhantomData,
        }
    }

    // P = lp rq 2^(shift (b - m)) + lq rp, Q = lq rq.
    fn merge(ctx: &Shift, l: Self, r: Self) -> Self {
        let u = product(&l.q, &r.p);
        let sh = ctx.0 * (r.b - r.a);
        let (p, q, neg) = dyn_merge((&l.p, &l.q, l.neg), (&u, &r.q, r.neg), sh, S::SIGNED);
        DynNodeBBP {
            a: l.a,
            b: r.b,
            p,
            q,
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &Shift, zeros: usize, out: &mut [u64]) -> bool {
        let bits = ctx.0 * (self.b - self.a);
        dyn_finalize(&self.p, &self.q, S::SIGNED && self.neg, bits, zeros, out)
    }
}

#[derive(Debug, Clone)]
pub struct StaticNodeBBP<const N: usize, S: WideBBPSeries> {
    a: u64,
    b: u64,
    // P 2^ep and Q 2^eq, with exponents in bits.
    p: [u64; N],
    p_len: usize,
    ep: u64,
    q: [u64; N],
    q_len: usize,
    eq: u64,
    neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<const N: usize, S: WideBBPSeries> StaticNodeBBP<N, S> {
    #[inline(never)]
    fn exact_leaf(shift: u64, a: u64, b: u64) -> Self {
        let (mut p, mut q) = ([0; N], [0; N]);
        let k: LeafKernel = if fits_u64(a, b, |n| S::p(n) | S::q(n)) {
            bbp_leaf_terms::<Narrow<S>>
        } else {
            bbp_leaf_terms_wide::<S>
        };
        let (p_len, q_len, e, neg) = k(a, b, shift, &mut p, &mut q);
        debug_assert_eq!(e, 0, "leaf checks that P stays exact");

        StaticNodeBBP {
            a,
            b,
            p,
            p_len,
            ep: 0,
            q,
            q_len,
            eq: 0,
            neg,
            _s: PhantomData,
        }
    }

    // (P, Q, shift) with S = P / (Q 2^shift).
    pub(crate) fn ratio(&self, ctx: &Shift) -> (&[u64], &[u64], i64) {
        let shift = (ctx.0 * (self.b - self.a)) as i64 + self.eq as i64 - self.ep as i64;
        (&self.p[..self.p_len], &self.q[..self.q_len], shift)
    }
}

impl<const N: usize, S: WideBBPSeries> Node for StaticNodeBBP<N, S> {
    type Ctx = Shift;
    fn leaf_limbs() -> u64 {
        k::bin_split::static_bbp_leaf_limbs()
    }

    fn range(&self) -> (u64, u64) {
        (self.a, self.b)
    }

    fn limbs(ctx: &Shift, a: u64, b: u64) -> u64 {
        est_limbs(a, b, S::q, ctx.0).min(N as u64)
    }

    // A leaf's exact P, at most prod q 2^(shift (t - 1)) t max p, must fit N limbs.
    fn leaf(ctx: &Shift, a: u64, b: u64, last: bool) -> Self {
        let t = b - a;
        let (mut bits, mut p_bits) = (ctx.0 * (t - 1) + bit_len(t), 0);
        let mut fits = |n| {
            bits += bit_len128(S::q(n as u128));
            p_bits = p_bits.max(bit_len128(S::p(n as u128)));
            bits + p_bits <= 64 * N as u64
        };
        if t > 1 && !(a..b).all(&mut fits) {
            return split_leaf(ctx, a, b, last);
        }
        Self::exact_leaf(ctx.0, a, b)
    }

    // P = lp rq 2^(shift (b - m)) + lq rp from lp rq in p and lq rp in l.p, then
    // Q = lq rq in r.p.
    fn merge(ctx: &Shift, mut l: Self, mut r: Self) -> Self {
        const { assert!(N >= 2) };
        let sh = ctx.0 * (r.b - r.a);
        let (lq, rq) = (&l.q[..l.q_len], &r.q[..r.q_len]);

        let (lp, rp) = (&l.p[..l.p_len], &r.p[..r.p_len]);
        let (ea, eu, signs) = (l.ep + r.eq + sh, l.eq + r.ep, (l.neg, r.neg));
        let (ta, tu) = (term_top(lp, rq, ea), term_top(lq, rp, eu));
        let ep = p_exponent::<N>(ta, ea, tu, eu, !S::SIGNED || l.neg == r.neg);

        let mut p = [0; N];
        let (off, n) = product_at::<N>(lp, rq, ea, ep, &mut p);
        let (uo, un) = product_at::<N>(lq, rp, eu, ep, &mut l.p);
        let u = (uo, &l.p[uo..uo + un]);
        let (p_len, neg) = p_sum(&mut p, off + n, u, signs, ta == 0, S::SIGNED);
        let (q_len, e) = hi_mul_norm::<N>(lq, rq, &mut r.p);

        StaticNodeBBP {
            a: l.a,
            b: r.b,
            p,
            p_len,
            ep,
            q: r.p,
            q_len,
            eq: l.eq + r.eq + e,
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &Shift, zeros: usize, out: &mut [u64]) -> bool {
        let (p, q, shift) = self.ratio(ctx);
        ratio_to_fraction_static::<N>(p, q, shift - 64 * zeros as i64, out);
        S::SIGNED && self.neg && self.p_len != 0
    }
}

//$\sum^{N}_{n=1} (-1)^{\sigma(n)} \; P(n) \prod^{n}_{r=1} \frac{2^{-k}}{Q(r)}$
// Summed from n = START, with the product from r = START.
series_traits!(EngelSeries, WideEngelSeries, p, q);

#[derive(Debug, Clone)]
pub struct DynNodeEngel<S: WideEngelSeries> {
    pub(crate) a: u64,
    pub(crate) b: u64,
    pub(crate) p: Vec<u64>,
    pub(crate) q: Vec<u64>,
    // P's sign.
    pub(crate) neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<S: WideEngelSeries> Node for DynNodeEngel<S> {
    type Ctx = Shift;
    fn leaf_limbs() -> u64 {
        k::bin_split::dyn_engel_leaf_limbs()
    }
    const PARALLEL: bool = true;

    fn range(&self) -> (u64, u64) {
        (self.a, self.b)
    }

    // P, about Q 2^(shift t).
    fn limbs(ctx: &Shift, a: u64, b: u64) -> u64 {
        est_limbs(a, b, S::q, ctx.0)
    }

    fn leaf(ctx: &Shift, a: u64, b: u64, _: bool) -> Self {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let k: LeafKernel = if narrow {
            engel_leaf_terms::<Narrow<S>>
        } else {
            engel_leaf_terms_wide::<S>
        };
        let (p, q, neg) = dyn_leaf(a, b, ctx.0, narrow, k);
        DynNodeEngel {
            a,
            b,
            p,
            q,
            neg,
            _s: PhantomData,
        }
    }

    // P = lp rq 2^(shift (b - m)) + rp, Q = lq rq.
    fn merge(ctx: &Shift, l: Self, r: Self) -> Self {
        let sh = ctx.0 * (r.b - r.a);
        let (p, q, neg) = dyn_merge((&l.p, &l.q, l.neg), (&r.p, &r.q, r.neg), sh, S::SIGNED);
        DynNodeEngel {
            a: l.a,
            b: r.b,
            p,
            q,
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &Shift, zeros: usize, out: &mut [u64]) -> bool {
        let bits = ctx.0 * (self.b - self.a);
        dyn_finalize(&self.p, &self.q, S::SIGNED && self.neg, bits, zeros, out)
    }
}

#[derive(Debug, Clone)]
pub struct StaticNodeEngel<const N: usize, S: WideEngelSeries> {
    a: u64,
    b: u64,
    // P 2^ep and Q 2^eq, with exponents in bits.
    p: [u64; N],
    p_len: usize,
    ep: u64,
    q: [u64; N],
    q_len: usize,
    eq: u64,
    neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<const N: usize, S: WideEngelSeries> StaticNodeEngel<N, S> {
    #[inline(never)]
    fn exact_leaf(shift: u64, a: u64, b: u64) -> Self {
        let (mut p, mut q) = ([0; N], [0; N]);
        let k: LeafKernel = if fits_u64(a, b, |n| S::p(n) | S::q(n)) {
            engel_leaf_terms::<Narrow<S>>
        } else {
            engel_leaf_terms_wide::<S>
        };
        let (p_len, q_len, ep, neg) = k(a, b, shift, &mut p, &mut q);
        StaticNodeEngel {
            a,
            b,
            p,
            p_len,
            ep,
            q,
            q_len,
            eq: 0,
            neg,
            _s: PhantomData,
        }
    }

    // (P, Q, shift) with S = P / (Q 2^shift).
    pub(crate) fn ratio(&self, ctx: &Shift) -> (&[u64], &[u64], i64) {
        let shift = (ctx.0 * (self.b - self.a)) as i64 + self.eq as i64 - self.ep as i64;
        (&self.p[..self.p_len], &self.q[..self.q_len], shift)
    }
}

impl<const N: usize, S: WideEngelSeries> Node for StaticNodeEngel<N, S> {
    type Ctx = Shift;
    fn leaf_limbs() -> u64 {
        k::bin_split::static_engel_leaf_limbs()
    }

    fn range(&self) -> (u64, u64) {
        (self.a, self.b)
    }

    fn limbs(ctx: &Shift, a: u64, b: u64) -> u64 {
        est_limbs(a, b, S::q, ctx.0).min(N as u64)
    }

    // P drops bits on its own, but a leaf's exact Q must fit N limbs.
    fn leaf(ctx: &Shift, a: u64, b: u64, last: bool) -> Self {
        if b - a > 1 && !prod_fits(a, b, S::q, 64 * N as u64) {
            return split_leaf(ctx, a, b, last);
        }
        Self::exact_leaf(ctx.0, a, b)
    }

    // P = lp rq 2^(shift (b - m)) + rp from lp rq in p and rp in place, then
    // Q = lq rq in l.p.
    fn merge(ctx: &Shift, mut l: Self, mut r: Self) -> Self {
        const { assert!(N >= 2) };
        let sh = ctx.0 * (r.b - r.a);
        let (lq, rq) = (&l.q[..l.q_len], &r.q[..r.q_len]);

        let lp = &l.p[..l.p_len];
        let ea = l.ep + r.eq + sh;
        let ta = term_top(lp, rq, ea);
        let tu = if r.p_len == 0 {
            0
        } else {
            r.ep + bits(&r.p[..r.p_len])
        };
        let ep = p_exponent::<N>(ta, ea, tu, r.ep, !S::SIGNED || l.neg == r.neg);

        let mut p = [0; N];
        let (off, n) = product_at::<N>(lp, rq, ea, ep, &mut p);
        // rp moves to P's exponent in place, dropping its bits below it.
        let (uo, un) = if r.ep >= ep {
            let (off, s) = split_sh(r.ep - ep);
            (off, shl_bits(&mut r.p, r.p_len, s))
        } else {
            (0, shr_bits(&mut r.p, r.p_len, 0, ep - r.ep))
        };
        let u = (uo, &r.p[..un]);
        let (p_len, neg) = p_sum(&mut p, off + n, u, (l.neg, r.neg), ta == 0, S::SIGNED);
        let (q_len, e) = hi_mul_norm::<N>(lq, rq, &mut l.p);

        StaticNodeEngel {
            a: l.a,
            b: r.b,
            p,
            p_len,
            ep,
            q: l.p,
            q_len,
            eq: l.eq + r.eq + e,
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &Shift, zeros: usize, out: &mut [u64]) -> bool {
        let (p, q, shift) = self.ratio(ctx);
        ratio_to_fraction_static::<N>(p, q, shift - 64 * zeros as i64, out);
        S::SIGNED && self.neg && self.p_len != 0
    }
}

//$\sum^{N}_{n=1} (-1)^{\sigma(n)} \; x^{n} \; \frac{P(n)}{R(n)} \; \prod^{n}_{r=1} \frac{R(r)}{Q(r)} \; 2^{-k}$
// Summed from n = START, with x^(n - START + 1) and the product from r = START.
series_traits!(HyperSeries, WideHyperSeries, p, q, r);

#[derive(Debug, Clone)]
pub struct DynHyperCtx {
    pub shift: u64,
    // None when x = 1, and x_bits are the bits x^n adds per term.
    x: Option<Vec<u64>>,
    x_bits: u64,
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
            x_bits: if x == [1] { 0 } else { bits(x) },
        }
    }

    pub const fn new_shift(shift: u64) -> Self {
        DynHyperCtx {
            shift,
            x: None,
            x_bits: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StaticHyperCtx<const N: usize> {
    shift: u64,
    x: Option<([u64; N], usize)>,
    x_bits: u64,
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
            x_bits: if x == [1] { 0 } else { bits(x) },
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynNodeHyper<S: WideHyperSeries> {
    pub(crate) a: u64,
    pub(crate) b: u64,
    pub(crate) p: Vec<u64>,
    pub(crate) q: Vec<u64>,
    pub(crate) r: Option<Vec<u64>>,
    pub(crate) neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<S: WideHyperSeries> Node for DynNodeHyper<S> {
    type Ctx = DynHyperCtx;
    fn leaf_limbs() -> u64 {
        k::bin_split::dyn_hyper_leaf_limbs()
    }
    fn leaf_terms() -> u64 {
        k::bin_split::dyn_hyper_leaf_terms()
    }
    const PARALLEL: bool = true;

    fn range(&self) -> (u64, u64) {
        (self.a, self.b)
    }

    // P, about the larger of Q 2^(shift t) and R x^t.
    fn limbs(ctx: &DynHyperCtx, a: u64, b: u64) -> u64 {
        est_limbs(a, b, S::q, ctx.shift).max(est_limbs(a, b, S::r, ctx.x_bits))
    }

    fn leaf(ctx: &DynHyperCtx, a: u64, b: u64, last: bool) -> Self {
        let terms = (b - a) as usize;
        let words = split_sh(ctx.shift * (b - a)).0 + 1;
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n) | S::r(n));
        let (pl, ql) = leaf_lens(narrow, terms, words);
        let mut q = vec![0; ql];
        let (p, q_len, r, neg) = match ctx.x.as_deref() {
            None => {
                let (mut p, mut r) = (vec![0; pl], vec![0; ql]);
                let (p_len, q_len, r_len, e, neg) = if narrow {
                    hyper_leaf_terms::<Narrow<S>>(a, b, ctx.shift, &mut p, &mut q, &mut r)
                } else {
                    hyper_leaf_terms_wide::<S>(a, b, ctx.shift, &mut p, &mut q, &mut r)
                };
                debug_assert_eq!(e, 0);
                p.truncate(p_len);
                r.truncate(r_len);
                (p, q_len, (!last).then_some(r), neg)
            }
            Some(x) => {
                // A spare limb past P keeps make_room from dropping bits. A wide term's q
                // and r can add two limbs each.
                let cap = (terms + 1) * (x.len() + if narrow { 1 } else { 4 }) + 1;
                let mut p = vec![0; cap + words + 1];
                let mut scratch = ScratchGuard::acquire();
                let [t, tmp] = scratch.get_splits([cap + x.len(); 2]);
                let exact =
                    |a: &[u64], b: &[u64], ea, _, out: &mut [u64]| (mul_exact(a, b, out), ea);
                let k = (ctx.shift, x);
                let (p_len, q_len, e, t, et, neg) = if narrow {
                    hyper_x_leaf_terms::<S, false>(a, b, k, &mut p, &mut q, t, tmp, exact)
                } else {
                    hyper_x_leaf_terms::<S, true>(a, b, k, &mut p, &mut q, t, tmp, exact)
                };
                debug_assert_eq!(e + et, 0);
                p.truncate(p_len);
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

    // P = lp rq 2^(shift (b - m)) + rp lr, Q = lq rq, R = lr rr.
    fn merge(ctx: &DynHyperCtx, left: Self, right: Self) -> Self {
        let lr = left
            .r
            .expect("a node that ends the sum can't merge as a left child");
        let (u, r) = (
            product(&right.p, &lr),
            right.r.as_deref().map(|rr| product(&lr, rr)),
        );
        let (l, rt) = (
            (&left.p[..], &left.q[..], left.neg),
            (&u[..], &right.q[..], right.neg),
        );
        let (p, q, neg) = dyn_merge(l, rt, ctx.shift * (right.b - right.a), S::SIGNED);
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

    fn finalize(&self, ctx: &DynHyperCtx, zeros: usize, out: &mut [u64]) -> bool {
        let bits = ctx.shift * (self.b - self.a);
        dyn_finalize(&self.p, &self.q, S::SIGNED && self.neg, bits, zeros, out)
    }
}

#[derive(Debug, Clone)]
pub struct StaticNodeHyper<const N: usize, S: WideHyperSeries> {
    a: u64,
    b: u64,
    // P 2^ep, Q 2^eq and R 2^er, with exponents in bits.
    p: [u64; N],
    p_len: usize,
    ep: u64,
    q: [u64; N],
    q_len: usize,
    eq: u64,
    r: Option<([u64; N], usize, u64)>,
    neg: bool,
    _s: PhantomData<fn() -> S>,
}

impl<const N: usize, S: WideHyperSeries> StaticNodeHyper<N, S> {
    // t keeps its top 64 N bits as R for a later merge, or, in a leaf that ends the sum,
    // only its bits from P's exponent up.
    #[inline(never)]
    fn x_leaf(
        a: u64,
        b: u64,
        (shift, x, last): (u64, &[u64], bool),
        p: &mut [u64; N],
        q: &mut [u64; N],
        r: &mut [u64; N],
    ) -> (usize, usize, usize, u64, u64, bool) {
        let (mut t, mut tmp) = ([0; N], [0; N]);
        let t_mul = |t: &[u64], x: &[u64], et: u64, e: u64, out: &mut [u64]| {
            if last {
                let (off, len) = product_at::<N>(t, x, et, e, out);
                (off + len, e)
            } else {
                let (len, d) = hi_mul_norm::<N>(t, x, out);
                (len, et + d)
            }
        };
        let k = (shift, x);
        let (p_len, q_len, ep, t, er, neg) = if fits_u64(a, b, |n| S::p(n) | S::q(n) | S::r(n)) {
            hyper_x_leaf_terms::<S, false>(a, b, k, p, q, &mut t, &mut tmp, t_mul)
        } else {
            hyper_x_leaf_terms::<S, true>(a, b, k, p, q, &mut t, &mut tmp, t_mul)
        };
        if !last {
            r[..t.len()].copy_from_slice(t);
        }
        (p_len, q_len, t.len(), ep, er, neg)
    }

    #[inline(never)]
    fn exact_leaf(ctx: &StaticHyperCtx<N>, a: u64, b: u64, last: bool) -> Self {
        let (mut p, mut q, mut r) = ([0; N], [0; N], [0; N]);
        let (p_len, q_len, r_len, ep, er, neg) = match &ctx.x {
            None => {
                let (p, q, r) = (&mut p, &mut q, &mut r);
                let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n) | S::r(n));
                let (p_len, q_len, r_len, ep, neg) = if narrow {
                    hyper_leaf_terms::<Narrow<S>>(a, b, ctx.shift, p, q, r)
                } else {
                    hyper_leaf_terms_wide::<S>(a, b, ctx.shift, p, q, r)
                };
                (p_len, q_len, r_len, ep, 0, neg)
            }
            Some((x, len)) => {
                let x = (ctx.shift, &x[..*len], last);
                Self::x_leaf(a, b, x, &mut p, &mut q, &mut r)
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

    // (P, Q, shift) with S = P / (Q 2^shift).
    pub(crate) fn ratio(&self, ctx: &StaticHyperCtx<N>) -> (&[u64], &[u64], i64) {
        let shift = (ctx.shift * (self.b - self.a)) as i64 + self.eq as i64 - self.ep as i64;
        (&self.p[..self.p_len], &self.q[..self.q_len], shift)
    }

    // Whether S is negative.
    pub(crate) fn is_neg(&self) -> bool {
        S::SIGNED && self.neg && self.p_len != 0
    }
}

impl<const N: usize, S: WideHyperSeries> Node for StaticNodeHyper<N, S> {
    type Ctx = StaticHyperCtx<N>;
    fn leaf_limbs() -> u64 {
        k::bin_split::static_hyper_leaf_limbs()
    }
    fn leaf_terms() -> u64 {
        k::bin_split::static_hyper_leaf_terms()
    }

    fn range(&self) -> (u64, u64) {
        (self.a, self.b)
    }

    fn limbs(ctx: &StaticHyperCtx<N>, a: u64, b: u64) -> u64 {
        let est = est_limbs(a, b, S::q, ctx.shift).max(est_limbs(a, b, S::r, ctx.x_bits));
        est.min(N as u64)
    }

    // P, and R when x != 1, drop bits on their own, but a leaf's exact Q, and R when
    // x = 1, must fit N limbs.
    fn leaf(ctx: &StaticHyperCtx<N>, a: u64, b: u64, last: bool) -> Self {
        let cap = 64 * N as u64;
        let fits = || prod_fits(a, b, S::q, cap) && (ctx.x.is_some() || prod_fits(a, b, S::r, cap));
        if b - a > 1 && !fits() {
            return split_leaf(ctx, a, b, last);
        }
        Self::exact_leaf(ctx, a, b, last)
    }

    // P = lp rq 2^(shift (b - m)) + rp lr from lp rq in p and rp lr in left.p, then
    // Q = lq rq in right.p and R = lr rr in left.p.
    fn merge(ctx: &StaticHyperCtx<N>, mut left: Self, mut right: Self) -> Self {
        const { assert!(N >= 2) };
        let sh = ctx.shift * (right.b - right.a);
        let (lq, rq) = (&left.q[..left.q_len], &right.q[..right.q_len]);
        let (lr, ler) = left
            .r
            .as_ref()
            .map(|(r, len, e)| (&r[..*len], *e))
            .expect("a node that ends the sum can't merge as a left child");

        let (lp, rp) = (&left.p[..left.p_len], &right.p[..right.p_len]);
        let (ea, eu) = (left.ep + right.eq + sh, right.ep + ler);
        let (ta, tu) = (term_top(lp, rq, ea), term_top(rp, lr, eu));
        let same = !S::SIGNED || left.neg == right.neg;
        let ep = p_exponent::<N>(ta, ea, tu, eu, same);

        let mut p = [0; N];
        let (off, n) = product_at::<N>(lp, rq, ea, ep, &mut p);
        let (uo, un) = product_at::<N>(rp, lr, eu, ep, &mut left.p);
        let u = (uo, &left.p[uo..uo + un]);
        let signs = (left.neg, right.neg);
        let (p_len, neg) = p_sum(&mut p, off + n, u, signs, ta == 0, S::SIGNED);
        let (q_len, e) = hi_mul_norm::<N>(lq, rq, &mut right.p);

        let r = right.r.as_ref().map(|(rr, len, er)| {
            let (r_len, k) = hi_mul_norm::<N>(lr, &rr[..*len], &mut left.p);
            (left.p, r_len, ler + er + k)
        });

        StaticNodeHyper {
            a: left.a,
            b: right.b,
            p,
            p_len,
            ep,
            q: right.p,
            q_len,
            eq: left.eq + right.eq + e,
            r,
            neg,
            _s: PhantomData,
        }
    }

    fn finalize(&self, ctx: &StaticHyperCtx<N>, zeros: usize, out: &mut [u64]) -> bool {
        let (p, q, shift) = self.ratio(ctx);
        ratio_to_fraction_static::<N>(p, q, shift - 64 * zeros as i64, out);
        self.is_neg()
    }
}

// The series_eval functions sum by reverse Horner, rectangular splitting (Hyper with x of
// two or more limbs) or binary splitting, whichever their cost model picks, and write S as
// Node::finalize does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesAlg {
    Horner,
    Rect,
    BinSplit,
}

// Which algorithms can sum a range. Binary splitting always can.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eligible {
    pub horner: bool,
    pub rect: bool,
}

impl Eligible {
    pub fn allows(self, alg: SeriesAlg) -> bool {
        match alg {
            SeriesAlg::Horner => self.horner,
            SeriesAlg::Rect => self.rect,
            SeriesAlg::BinSplit => true,
        }
    }
}

// The node types the series_eval functions sum with. Only they and the tests call these.
pub trait SeriesNode: Node + Send {
    // Whether the kind has a rect kernel: only Hyper's, whose steps multiply by x.
    #[doc(hidden)]
    const RECT: bool = false;
    // Horner accepts u128 coefficients; rect still requires u64 coefficients. Hyper needs
    // r(n) <= q(n) and k >= bits(x), and static kernels need room for their accumulators.
    #[doc(hidden)]
    fn eligible(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize) -> Eligible;
    // What the cost model sees of the range.
    #[doc(hidden)]
    fn shape(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize) -> Shape;
    // Each eligible algorithm's predicted work, rect's at block length m (None: the model's).
    #[doc(hidden)]
    fn costs(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize, m: Option<usize>) -> Costs {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        Self::shape(ctx, a, b, len, zeros).costs(ok, m)
    }
    // The rebuilt equations' parts for Horner, rect at block length m, and binary splitting
    // (dynamic nodes): None where the algorithm can't sum the range.
    #[doc(hidden)]
    fn parts(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize, m: Option<usize>, env: &SeriesEnv) -> [Option<Parts>; 3] {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        Self::shape(ctx, a, b, len, zeros).parts(ok, m, env)
    }
    // Whether the range goes to Horner without the cost model (horner_first): for BBP and
    // Engel, a range of at most `series.horner_terms` terms.
    #[doc(hidden)]
    fn horner_first(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize) -> bool {
        horner_first(b - a <= k::series::horner_terms(), len, || {
            Self::eligible(ctx, a, b, len, zeros)
        })
    }
    // The range's class in the small-series table (`small_class`): None for a static node and
    // for coefficients past 64 bits, which the table doesn't hold.
    #[doc(hidden)]
    fn small_class(_: &Self::Ctx, _: u64, _: u64) -> Option<usize> {
        None
    }
    // What the small-series table (`series.small`) says to run, and rect's block length: None
    // where it leaves the range to the equations, or names an algorithm that can't sum it.
    // Read ahead of everything else: a lookup costs a coefficient or two and a few compares,
    // where the equations cost as much as a short series does.
    #[doc(hidden)]
    fn small(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize) -> Option<(SeriesAlg, Option<usize>)> {
        let alg = small_pick(k::series::small(), Self::small_class(ctx, a, b)?, len, b - a)?;
        if (alg == SeriesAlg::Rect && !Self::RECT) || !Self::eligible(ctx, a, b, len, zeros).allows(alg) {
            return None;
        }
        // Rect's block length is still the model's: it is what the table's pick was derived at.
        let m = (alg == SeriesAlg::Rect).then(|| Self::shape(ctx, a, b, len, zeros).rect_m());
        Some((alg, m))
    }
    // What the guards settle for the range before any equation is read (`Shape::guards`).
    #[doc(hidden)]
    fn guards(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize) -> Guarded {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        Self::shape(ctx, a, b, len, zeros).live_guards(ok)
    }
    // The algorithm the range runs once horner_first has passed it on, and rect's block length:
    // what the guards settle, else the eligible algorithm with the least predicted time.
    #[doc(hidden)]
    fn decide(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize) -> (SeriesAlg, Option<usize>) {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        Self::shape(ctx, a, b, len, zeros).decide(ok)
    }
    // The small-series table's algorithm where it has one, Horner when horner_first, otherwise
    // `decide`'s.
    #[doc(hidden)]
    fn dispatch(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize) -> SeriesAlg {
        if let Some((alg, _)) = Self::small(ctx, a, b, len, zeros) {
            return alg;
        }
        if Self::horner_first(ctx, a, b, len, zeros) {
            return SeriesAlg::Horner;
        }
        Self::decide(ctx, a, b, len, zeros).0
    }
    #[doc(hidden)]
    fn horner(ctx: &Self::Ctx, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool;
    // Rect at block length m, or the cost model's.
    #[doc(hidden)]
    fn rect(_: &Self::Ctx, _: u64, _: u64, _: usize, _: &mut [u64], _: Option<usize>) -> bool {
        unreachable!("only a RECT kind has a rect kernel")
    }
}

// A small range at a small precision goes to Horner without the cost model, which costs
// about as much as the sum there: a Hyper range with a one-limb x, where the model picks Horner
// anyway, or a BBP or Engel range of a few terms, where Horner beats the model and its pick.
fn horner_first(small: bool, len: usize, eligible: impl FnOnce() -> Eligible) -> bool {
    small && len <= k::series::horner_limbs() && eligible().horner
}

// Node::finalize's output for the terms [a, b) by dispatch's pick, for tests and tuning.
// Horner and rect are within one unit of out's last limb, not bit-identical to bin_split.
#[doc(hidden)]
pub fn series_eval_range<N>(ctx: &N::Ctx, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool
where
    N: SeriesNode,
    N::Ctx: Sync,
{
    debug_assert!(a < b);
    // A short series by the table, which costs next to nothing to read; then the ranges
    // horner_first takes; then the guards and the equations.
    let small = N::small(ctx, a, b, out.len(), zeros);
    if small.is_none() && N::horner_first(ctx, a, b, out.len(), zeros) {
        return N::horner(ctx, a, b, zeros, out);
    }
    // Either decision has checked eligibility and found rect's block length.
    match small.unwrap_or_else(|| N::decide(ctx, a, b, out.len(), zeros)) {
        (SeriesAlg::Horner, _) => N::horner(ctx, a, b, zeros, out),
        (SeriesAlg::Rect, m) => N::rect(ctx, a, b, zeros, out, m),
        (SeriesAlg::BinSplit, _) => bin_split::<N>(ctx, a, b).finalize(ctx, zeros, out),
    }
}

// series_eval_range with the algorithm forced, for tests and tuning.
#[doc(hidden)]
pub fn series_eval_at<N>(
    ctx: &N::Ctx,
    a: u64,
    b: u64,
    zeros: usize,
    out: &mut [u64],
    alg: SeriesAlg,
) -> bool
where
    N: SeriesNode,
    N::Ctx: Sync,
{
    debug_assert!(a < b);
    let ok = N::eligible(ctx, a, b, out.len(), zeros).allows(alg);
    assert!(ok, "{alg:?} can't sum this range");
    match alg {
        SeriesAlg::Horner => N::horner(ctx, a, b, zeros, out),
        SeriesAlg::Rect => {
            assert!(N::RECT, "only a RECT kind has a rect kernel");
            N::rect(ctx, a, b, zeros, out, None)
        }
        SeriesAlg::BinSplit => bin_split::<N>(ctx, a, b).finalize(ctx, zeros, out),
    }
}

// A series with its shift, and x for Hyper. The context's type picks the series kind, so its
// constructor only accepts a series of that kind; series_eval_dyn and series_eval_static sum
// it to out's precision.
pub struct Hyper<'a, S> {
    x: &'a [u64],
    shift: u64,
    // |x 2^-shift| <= 2^-s.
    s: i64,
    _s: PhantomData<fn() -> S>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperCtxError {
    // x is empty or its top limb is zero.
    NotTrimmed,
    // x 2^-shift > 1/2: shift must be at least min_shift = ceil(log2 x) + 1.
    ShiftTooSmall { min_shift: u64 },
}

impl fmt::Display for HyperCtxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HyperCtxError::NotTrimmed => write!(f, "x must be trimmed and nonzero"),
            HyperCtxError::ShiftTooSmall { min_shift } => {
                write!(
                    f,
                    "x 2^-shift must be at most 1/2: shift must be >= {min_shift}"
                )
            }
        }
    }
}

impl std::error::Error for HyperCtxError {}

fn decay(shift: u64) -> i64 {
    shift.min(i64::MAX as u64) as i64
}

// ceil(log2 x) for a trimmed x != 0.
fn ceil_log2(x: &[u64]) -> u64 {
    let (top, low) = x.split_last().unwrap();
    bits(x) - (top.is_power_of_two() && low.iter().all(|&l| l == 0)) as u64
}

impl<'a, S: WideHyperSeries> Hyper<'a, S> {
    /// Fails unless x is trimmed and nonzero, with x 2^-shift <= 1/2. Only a Hyper series
    /// has a Hyper context:
    ///
    /// ```compile_fail
    /// use big_bits::utils::series::{EngelSeries, Hyper};
    /// struct EngelOnly;
    /// impl EngelSeries for EngelOnly {
    ///     const START: u64 = 1;
    ///     fn tail_bits(_: u64, _: i64) -> u64 { 0 }
    ///     fn p(_: u64) -> u64 { 1 }
    ///     fn q(n: u64) -> u64 { n }
    /// }
    /// let _ = Hyper::<EngelOnly>::new(&[3], 8);
    /// ```
    pub fn new(x: &'a [u64], shift: u64) -> Result<Self, HyperCtxError> {
        if x.last().map_or(true, |&l| l == 0) {
            return Err(HyperCtxError::NotTrimmed);
        }
        let lg = ceil_log2(x);
        if shift <= lg {
            return Err(HyperCtxError::ShiftTooSmall { min_shift: lg + 1 });
        }
        Ok(Hyper {
            x,
            shift,
            s: decay(shift - lg),
            _s: PhantomData,
        })
    }
}

mod sealed {
    pub trait Sealed {}
}

impl<S> sealed::Sealed for Hyper<'_, S> {}

// Past this many terms, a tail bound is taken never to reach its target.
const MAX_TERMS: u64 = 1 << 40;

// The least end b > a, by doubling and then bisection, whose tail(b) reaches bits, given that
// tail(a) doesn't (else a). A monotone tail gives the fewest terms; any other still stops
// where its own bound holds.
fn min_end(a: u64, bits: u64, tail: impl Fn(u64) -> u64) -> u64 {
    if tail(a) >= bits {
        return a;
    }
    let (mut lo, mut hi) = (0, 1);
    while tail(a.saturating_add(hi)) < bits {
        assert!(
            hi < MAX_TERMS,
            "the series' tail bound doesn't reach {bits} bits within 2^40 terms"
        );
        (lo, hi) = (hi, 2 * hi);
    }
    while hi - lo > 1 {
        let m = lo + (hi - lo) / 2;
        if tail(a + m) >= bits {
            hi = m;
        } else {
            lo = m;
        }
    }
    a + hi
}

// A context's nodes and its series' tail. The node items are for series_eval_dyn and
// series_eval_static.
pub trait Series: sealed::Sealed {
    #[doc(hidden)]
    type Dyn: SeriesNode<Ctx: Sync>;
    #[doc(hidden)]
    type Static<const N: usize>: SeriesNode<Ctx: Sync>;
    #[doc(hidden)]
    fn dyn_ctx(&self) -> <Self::Dyn as Node>::Ctx;
    // Err when x doesn't fit N limbs.
    #[doc(hidden)]
    fn static_ctx<const N: usize>(&self) -> Result<<Self::Static<N> as Node>::Ctx, CapacityError>;
    #[doc(hidden)]
    fn start(&self) -> u64;
    // The series' tail_bits at this context's s.
    #[doc(hidden)]
    fn tail_bits(&self, n: u64) -> u64;

    // S's leading zero limbs, from its tail bound at START: S < 2^(-64 zeros).
    fn zeros(&self) -> usize {
        (self.tail_bits(self.start()) / 64) as usize
    }

    // [START, end), the fewest terms whose tail bound reaches bits. Panics when the bound
    // doesn't within 2^40 terms.
    fn range(&self, bits: u64) -> (u64, u64) {
        let a = self.start();
        (a, min_end(a, bits, |n| self.tail_bits(n)))
    }
}

// Bbp and Engel, whose contexts are a shift.
macro_rules! shift_ctx {
    ($ctx:ident, $series:ident, $dyn:ident, $stat:ident) => {
        pub struct $ctx<S> {
            shift: u64,
            _s: PhantomData<fn() -> S>,
        }

        impl<S: $series> $ctx<S> {
            pub const fn new(shift: u64) -> Self {
                $ctx {
                    shift,
                    _s: PhantomData,
                }
            }
        }

        impl<S> sealed::Sealed for $ctx<S> {}

        impl<S: $series> Series for $ctx<S> {
            type Dyn = $dyn<S>;
            type Static<const N: usize> = $stat<N, S>;

            fn dyn_ctx(&self) -> Shift {
                Shift(self.shift)
            }

            fn static_ctx<const N: usize>(&self) -> Result<Shift, CapacityError> {
                Ok(Shift(self.shift))
            }

            fn start(&self) -> u64 {
                S::START
            }

            fn tail_bits(&self, n: u64) -> u64 {
                S::tail_bits(n, decay(self.shift))
            }
        }
    };
}

shift_ctx!(Bbp, WideBBPSeries, DynNodeBBP, StaticNodeBBP);
shift_ctx!(Engel, WideEngelSeries, DynNodeEngel, StaticNodeEngel);

impl<S: WideHyperSeries> Series for Hyper<'_, S> {
    type Dyn = DynNodeHyper<S>;
    type Static<const N: usize> = StaticNodeHyper<N, S>;

    fn dyn_ctx(&self) -> DynHyperCtx {
        DynHyperCtx::new(self.x, self.shift)
    }

    fn static_ctx<const N: usize>(&self) -> Result<StaticHyperCtx<N>, CapacityError> {
        if self.x.len() > N {
            return Err(CapacityError {
                min_n: self.x.len(),
            });
        }
        Ok(StaticHyperCtx::new(self.x, self.shift))
    }

    fn start(&self) -> u64 {
        S::START
    }

    fn tail_bits(&self, n: u64) -> u64 {
        S::tail_bits(n, self.s)
    }
}

// out <- S 2^(64 (out.len() + zeros)) for zeros = ctx.zeros(), the limbs S's tail bound puts
// above it. The sum stops once that bound reaches out's last limb, so out is within 2 units
// of S. Returns (S < 0, zeros).
pub fn series_eval_dyn<C: Series>(ctx: &C, out: &mut [u64]) -> (bool, usize) {
    eval::<C, C::Dyn>(ctx, &ctx.dyn_ctx(), out, None)
}

// series_eval_dyn with the algorithm forced, for tests and tuning.
#[doc(hidden)]
pub fn series_eval_dyn_at<C: Series>(ctx: &C, out: &mut [u64], alg: SeriesAlg) -> (bool, usize) {
    eval::<C, C::Dyn>(ctx, &ctx.dyn_ctx(), out, Some(alg))
}

// series_eval_dyn with [u64; N] storage. Without touching out, returns Err unless out.len() +
// 2 <= N, as static finalize needs, and a Hyper x fits N limbs.
pub fn series_eval_static<const N: usize, C: Series>(
    ctx: &C,
    out: &mut [u64],
) -> Result<(bool, usize), CapacityError> {
    eval_static::<N, C>(ctx, out, None)
}

// series_eval_static with the algorithm forced, for tests and tuning.
#[doc(hidden)]
pub fn series_eval_static_at<const N: usize, C: Series>(
    ctx: &C,
    out: &mut [u64],
    alg: SeriesAlg,
) -> Result<(bool, usize), CapacityError> {
    eval_static::<N, C>(ctx, out, Some(alg))
}

fn eval_static<const N: usize, C: Series>(
    ctx: &C,
    out: &mut [u64],
    alg: Option<SeriesAlg>,
) -> Result<(bool, usize), CapacityError> {
    let need = out.len() + 2;
    match ctx.static_ctx::<N>() {
        Ok(node) if need <= N => Ok(eval::<C, C::Static<N>>(ctx, &node, out, alg)),
        r => Err(CapacityError {
            min_n: need.max(r.err().map_or(0, |e| e.min_n)),
        }),
    }
}

// Sums ctx's range to out's precision on T's nodes, by the cost model or alg.
fn eval<C: Series, T: SeriesNode<Ctx: Sync>>(
    ctx: &C,
    node: &T::Ctx,
    out: &mut [u64],
    alg: Option<SeriesAlg>,
) -> (bool, usize) {
    let zeros = ctx.zeros();
    if out.is_empty() {
        return (false, zeros);
    }
    let (a, b) = ctx.range(64 * (out.len() + zeros) as u64);
    let neg = match alg {
        None => series_eval_range::<T>(node, a, b, zeros, out),
        Some(alg) => series_eval_at::<T>(node, a, b, zeros, out, alg),
    };
    (neg, zeros)
}

// hi_mul_dyn or hi_mul_static.
type HiMul = fn(&[u64], &[u64], &mut [u64]) -> u64;
// Term n's bit in Horner's frame.
type Bit<'a> = &'a dyn Fn(u64) -> i64;

// Horner's guard limb below out: each division truncates by a few units at most and scales
// the error below it by at most 1.
const HORNER_GUARD: usize = 1;
// At most this many powers of x on a static rect's stack, 8 N bytes each.
const RECT_MAX_M: usize = 16;
// At most this many terms share one division.
const GROUP_MAX: usize = 32;
// A term's own numerator, p(n) 2^d at the term's bit.
const ONE: &[u64] = &[1];

// Horner's accumulator: out, its guard limb and 2 limbs before division by q(n), plus one for
// wide coefficients and carries.
fn horner_acc_len(len: usize, narrow: bool) -> usize {
    len + HORNER_GUARD + 2 + usize::from(!narrow)
}

// The accumulator Horner runs in within cap limbs: the wide one when it fits, for its 128-bit
// group budget, and at least horner_acc_len, which eligibility checks.
fn horner_room(len: usize, narrow: bool, cap: usize) -> usize {
    horner_acc_len(len, false)
        .min(cap)
        .max(horner_acc_len(len, narrow))
}

// The bits a group's divisor Q = ∏q(n) may take: 64 per accumulator limb past the single-term
// layout, at most 128, as the un-divided sum is Q times one step's.
fn group_budget(acc: usize, out: usize) -> u64 {
    64 * (acc - out - HORNER_GUARD - 1).min(2) as u64
}

// Runs a Horner or rect kernel in the frame F = S 2^(64 w), w = len + zeros + HORNER_GUARD:
// term n's numerator lands at bit 64 w - k (n - a + 1), so 2^-k is never applied and acc is F
// throughout. Writes acc above its guard limb to out; returns S < 0.
fn horner_run(
    (k, a, zeros): (u64, u64, usize),
    out: &mut [u64],
    acc: &mut [u64],
    signed: bool,
    kernel: impl FnOnce(Bit, u64, &mut [u64]) -> (usize, bool),
) -> bool {
    let w = 64 * (out.len() + zeros + HORNER_GUARD) as i128;
    let bit = |n: u64| (w - k as i128 * (n - a + 1) as i128).max((i64::MIN / 2) as i128) as i64;
    acc.fill(0);
    let budget = group_budget(acc.len(), out.len());
    let (len, neg) = kernel(&bit, budget, acc);
    let g = HORNER_GUARD;
    out.copy_from_slice(&acc[g..g + out.len()]);
    debug_assert!(
        acc[g + out.len()..].iter().all(|&l| l == 0),
        "series_eval_range expects S < 2^(-64 zeros)"
    );
    signed && neg && len != 0
}

// acc[..len] <- about acc[..len] x / 2^(64 x.len()): hi_mul's window one limb lower in tmp,
// and its top limb. Returns the new length.
fn mul_frac(acc: &mut [u64], len: usize, x: &[u64], tmp: &mut [u64], hi_mul: HiMul) -> usize {
    if len == 0 {
        return 0;
    }
    let top = hi_mul(&acc[..len], x, &mut tmp[..len]);
    acc[..len - 1].copy_from_slice(&tmp[1..len]);
    acc[len - 1] = top;
    buf_len(&acc[..len])
}

// x shifted so its top bit is set, in out: x' = xn 2^(-64 x.len()) = x 2^(-bits(x)).
fn normalize<'x>(x: &[u64], out: &'x mut [u64]) -> &'x [u64] {
    let out = &mut out[..x.len()];
    out.copy_from_slice(x);
    shl_buf(out, x[x.len() - 1].leading_zeros() as u8);
    out
}

// x's zero low limbs. A power of x' padded to its frame keeps them, and products and adds
// skip them: x = x[z..] B^z.
fn low_zeros(x: &[u64]) -> usize {
    x.iter().position(|&l| l != 0).unwrap_or(0)
}

// The terms n - 1 down to n - m that share one division: while Q = ∏q(j) fits budget bits,
// the largest p(j) times Q (a bound on every weight, as r(j) <= q(j)) fits u128, and
// room(j, its bits) allows, but always one. rs[k] gets term n - 1 - k's r; returns (m, Q).
fn group(
    (a, n): (u64, u64),
    budget: u64,
    coef: impl Fn(u128) -> (u128, u128, u128),
    room: impl Fn(u64, u64) -> bool,
    rs: &mut [u128; GROUP_MAX],
) -> (usize, u128) {
    let (mut m, mut q_all, mut p_max) = (0, 1u128, 0u128);
    for j in (a..n).rev() {
        let (p, q, r) = coef(j as u128);
        if m != 0 {
            let fits = |nq: u128| {
                let w = p_max.max(p).checked_mul(nq);
                bit_len128(nq) <= budget && w.is_some_and(|w| room(j, bit_len128(w)))
            };
            if m == GROUP_MAX || !q_all.checked_mul(q).is_some_and(fits) {
                break;
            }
        }
        q_all *= q;
        p_max = p_max.max(p);
        rs[m] = r;
        m += 1;
    }
    (m, q_all)
}

// rs[k] <- ∏_{i>k} rs[i] over the group's m terms; returns ∏ rs (at most the group's Q).
fn suffix(rs: &mut [u128], m: usize) -> u128 {
    let mut all = 1u128;
    for r in rs[..m].iter_mut().rev() {
        (*r, all) = (all, all * *r);
    }
    all
}

// acc[..len] *= u, pushing the carry; 1 leaves it.
fn mul_acc(acc: &mut [u64], len: &mut usize, u: u128) {
    if u == 1 {
        return;
    }
    if u >> 64 == 0 {
        let c = mul_prim(&mut acc[..*len], u as u64);
        push(acc, len, c);
    } else {
        let c = mul_prim2(&mut acc[..*len], u);
        push2(acc, len, c);
    }
}

// acc[..len] /= q, truncating.
fn div_acc(acc: &mut [u64], len: &mut usize, q: u128) {
    if q >> 64 == 0 {
        div_prim(&mut acc[..*len], q as u64);
    } else {
        div_prim2(&mut acc[..*len], q);
    }
    *len = buf_len(&acc[..*len]);
}

// p += w x 2^d for a two-limb w, rounding the bits below p's bit 0. A term's own numerator
// (x = ONE) goes in as one two-limb vector, a power of x' a limb of w at a time.
fn add_weighted(
    p: &mut [u64],
    (len, e, pn): (&mut usize, &mut u64, &mut bool),
    (x, d): (&[u64], i64),
    w: u128,
    neg: bool,
    signed: bool,
) {
    if x == ONE {
        let limbs = [w as u64, (w >> 64) as u64];
        let x = (&limbs[..buf_len(&limbs)], d);
        return add_term(p, len, e, pn, x, 1, neg, signed);
    }
    add_term(p, len, e, pn, (x, d), w as u64, neg, signed);
    if w >> 64 != 0 {
        add_term(p, len, e, pn, (x, d + 64), (w >> 64) as u64, neg, signed);
    }
}

// One group's steps acc = (r acc + p x 2^d) / q, terms j = n - 1 - k for k < m, as one
// product and one division: acc = (R acc + Σ_k W_k p_k x_k 2^(d_k)) / Q with W_k =
// ∏_{i<k} q_i ∏_{i>k} r_i. term(j) is term j's (x, d); rs holds the group's r from group.
fn fold_group<'x, S: WideHyperSeries>(
    acc: &mut [u64],
    st: (&mut usize, &mut u64, &mut bool),
    (n, m, q_all): (u64, usize, u128),
    rs: &mut [u128; GROUP_MAX],
    term: impl Fn(u64) -> (&'x [u64], i64),
) {
    let (len, e, neg) = st;
    let r_all = suffix(rs, m);
    mul_acc(acc, len, r_all);
    let mut pre = 1u128;
    for (k, &suf) in rs[..m].iter().enumerate() {
        let j = n - 1 - k as u64;
        let i = j as u128;
        debug_assert!(S::SIGNED || !S::neg(i), "negative terms need SIGNED");
        let sub = S::SIGNED && S::neg(i);
        let w = S::p(i) * pre * suf;
        add_weighted(acc, (len, e, neg), term(j), w, sub, S::SIGNED);
        pre *= S::q(i);
    }
    div_acc(acc, len, q_all);
}

// Engel's acc = (acc + p(n) 2^d) / q(n) is Hyper's step with r(n) = 1.
struct EngelHyper<S>(PhantomData<S>);

impl<S: WideEngelSeries> WideHyperSeries for EngelHyper<S> {
    const SIGNED: bool = S::SIGNED;
    const START: u64 = S::START;
    fn tail_bits(n: u64, s: i64) -> u64 {
        S::tail_bits(n, s)
    }
    fn p(n: u128) -> u128 {
        S::p(n)
    }
    fn q(n: u128) -> u128 {
        S::q(n)
    }
    fn r(_: u128) -> u128 {
        1
    }
    fn neg(n: u128) -> bool {
        S::neg(n)
    }
}

// acc += floor(Σ_k p_k 2^(d_k) / q_k) per group of terms, from n = b - 1 down: the group's
// numerator Σ_k p_k (Q / q_k) 2^(d_k) in t (signed, as its terms may differ in sign), one
// division by Q, and one pass into acc.
fn horner_bbp<S: WideBBPSeries>(
    a: u64,
    b: u64,
    bit: Bit,
    budget: u64,
    acc: &mut [u64],
    t: &mut [u64],
) -> (usize, bool) {
    let (mut len, mut e, mut neg) = (0, 0, false);
    let mut ws = [0u128; GROUP_MAX];
    // The group's numerator stays below GROUP_MAX times its largest weighted term.
    let cap = 64 * t.len() as u64;
    let room = |j: u64, w: u64| bit(j).max(0) as u64 + w + 8 <= cap;
    let mut n = b;
    while n > a {
        let coef = |i: u128| (S::p(i), S::q(i), S::q(i));
        let (m, q_all) = group((a, n), budget, coef, room, &mut ws);
        suffix(&mut ws, m);
        let (mut pre, mut top) = (1u128, 0);
        for (k, w) in ws[..m].iter_mut().enumerate() {
            let j = n - 1 - k as u64;
            let i = j as u128;
            *w *= S::p(i) * pre;
            pre *= S::q(i);
            top = top.max(bit(j).max(0) as u64 + bit_len128(*w));
        }
        let top = ((top + 8) / 64 + 1).min(t.len() as u64) as usize;
        t[..top].fill(0);
        let (mut tl, mut te, mut tn) = (0, 0, false);
        for (k, &w) in ws[..m].iter().enumerate() {
            let j = n - 1 - k as u64;
            let i = j as u128;
            debug_assert!(S::SIGNED || !S::neg(i), "negative terms need SIGNED");
            let sub = S::SIGNED && S::neg(i);
            let st = (&mut tl, &mut te, &mut tn);
            add_weighted(t, st, (ONE, bit(j)), w, sub, S::SIGNED);
        }
        debug_assert_eq!(te, 0, "Horner's term overflowed");
        div_acc(t, &mut tl, q_all);
        let x = (0, &t[..tl], 1);
        add_pass(acc, &mut len, &mut e, &mut neg, x, tn, S::SIGNED);
        n -= m as u64;
    }
    debug_assert_eq!(e, 0, "Horner's accumulator overflowed");
    (len, neg)
}

// acc = x' (r(n) acc + p(n) 2^d) / q(n), from n = b - 1 down, x' = xn 2^(-64 xn.len()) through
// tmp, or 1 without x (the frame's k excludes bits(x)). A group of terms shares one division;
// without x it is also one product, and with x each step keeps its products by r and x'.
fn horner_hyper<S: WideHyperSeries>(
    a: u64,
    b: u64,
    bit: Bit,
    budget: u64,
    mut x: Option<(&[u64], &mut [u64], HiMul)>,
    acc: &mut [u64],
) -> (usize, bool) {
    let (mut len, mut e, mut neg) = (0, 0, false);
    let mut rs = [0u128; GROUP_MAX];
    let mut n = b;
    while n > a {
        let coef = |i: u128| (S::p(i), S::q(i), S::r(i));
        let (m, q_all) = group((a, n), budget, coef, |_, _| true, &mut rs);
        let st = (&mut len, &mut e, &mut neg);
        match x.as_mut() {
            None => fold_group::<S>(acc, st, (n, m, q_all), &mut rs, |j| (ONE, bit(j))),
            Some((xn, tmp, hi_mul)) => {
                let mut pre = 1u128;
                for (k, &r) in rs[..m].iter().enumerate() {
                    let j = n - 1 - k as u64;
                    let i = j as u128;
                    debug_assert!(S::SIGNED || !S::neg(i), "negative terms need SIGNED");
                    let sub = S::SIGNED && S::neg(i);
                    mul_acc(acc, st.0, r);
                    let w = S::p(i) * pre;
                    add_weighted(acc, (st.0, st.1, st.2), (ONE, bit(j)), w, sub, S::SIGNED);
                    *st.0 = mul_frac(acc, *st.0, xn, tmp, *hi_mul);
                    pre *= S::q(i);
                }
                div_acc(acc, st.0, q_all);
            }
        }
        n -= m as u64;
    }
    debug_assert_eq!(e, 0, "Horner's accumulator overflowed");
    (len, neg)
}

// pw[j] <- x'^(j + 1) 2^(64 P), P = pw[0].len(), for x' = xn 2^(-64 xn.len()), through tmp,
// and zs[j] <- pw[j]'s zero low limbs. Each step multiplies by x' without its padding.
fn rect_powers(xn: &[u64], pw: &mut [&mut [u64]], zs: &mut [u64], tmp: &mut [u64], hi_mul: HiMul) {
    let (first, rest) = pw.split_first_mut().expect("rect needs a power");
    let (p, c) = (first.len(), xn.len().min(first.len()));
    first[..p - c].fill(0);
    first[p - c..].copy_from_slice(&xn[xn.len() - c..]);
    let x1 = &first[low_zeros(first)..];
    let mut prev: &[u64] = first;
    for x in rest.iter_mut() {
        x.copy_from_slice(prev);
        mul_frac(x, p, x1, tmp, hi_mul);
        prev = x;
    }
    for (z, x) in zs.iter_mut().zip(pw.iter()) {
        *z = low_zeros(x) as u64;
    }
}

// Rectangular splitting: blocks of m = pw.len() terms from n = b - 1 down, each one high
// product by x'^s for its s terms and then Horner's x = 1 steps with p(n) x'^(i + 1) in
// place of p(n), so x' never multiplies in a step, grouped as fold_group does. Powers skip
// their zs[i] zero low limbs.
fn rect_hyper<S: WideHyperSeries>(
    a: u64,
    b: u64,
    bit: Bit,
    budget: u64,
    (pw, zs): (&[&mut [u64]], &[u64]),
    acc: &mut [u64],
    tmp: &mut [u64],
    hi_mul: HiMul,
) -> (usize, bool) {
    let (m, pb) = (pw.len() as u64, 64 * pw[0].len() as i64);
    let pow = |i: usize| (&pw[i][zs[i] as usize..], 64 * zs[i] as i64 - pb);
    let (mut len, mut e, mut neg) = (0, 0, false);
    let mut rs = [0u128; GROUP_MAX];
    let mut n = b;
    while n > a {
        let lo = n.saturating_sub(m).max(a);
        len = mul_frac(acc, len, pow((n - lo - 1) as usize).0, tmp, hi_mul);
        let mut j = n;
        while j > lo {
            let coef = |i: u128| (S::p(i), S::q(i), S::r(i));
            let (g, q_all) = group((lo, j), budget, coef, |_, _| true, &mut rs);
            let term = |k: u64| {
                let (x, d) = pow((k - lo) as usize);
                (x, bit(k) + d)
            };
            let st = (&mut len, &mut e, &mut neg);
            fold_group::<S>(acc, st, (j, g, q_all), &mut rs, term);
            j -= g as u64;
        }
        n = lo;
    }
    debug_assert_eq!(e, 0, "rect's accumulator overflowed");
    (len, neg)
}

// Horner needs k >= bits(x) and r(n) <= q(n), and rect also narrow coefficients. One pass.
fn hyper_eligible<S: WideHyperSeries>(a: u64, b: u64, k: u64, x_bits: u64) -> Eligible {
    let mut narrow = true;
    let horner = k >= x_bits
        && (a as u128..b as u128).all(|n| {
            let (p, q, r) = (S::p(n), S::q(n), S::r(n));
            narrow &= (p | q | r) >> 64 == 0;
            r <= q
        });
    Eligible {
        horner,
        rect: horner && narrow,
    }
}

// acc and tmp hold horner_room limbs, and xs x.len().
fn hyper_horner<S: WideHyperSeries>(
    (k, x): (u64, Option<&[u64]>),
    a: u64,
    b: u64,
    zeros: usize,
    out: &mut [u64],
    [acc, tmp, xs]: [&mut [u64]; 3],
    hi_mul: HiMul,
) -> bool {
    let (k, x) = match x {
        None => (k, None),
        Some(x) => (k - bits(x), Some((normalize(x, xs), tmp, hi_mul))),
    };
    let kernel = |bit: Bit, budget, acc: &mut [u64]| horner_hyper::<S>(a, b, bit, budget, x, acc);
    horner_run((k, a, zeros), out, acc, S::SIGNED, kernel)
}

// hyper_horner's buffers, plus m powers of narrow Horner's accumulator length in pw and their
// zero low limb counts in zs.
fn hyper_rect<S: WideHyperSeries>(
    (k, x): (u64, &[u64]),
    a: u64,
    b: u64,
    zeros: usize,
    out: &mut [u64],
    [acc, tmp, xs]: [&mut [u64]; 3],
    (pw, zs): (&mut [&mut [u64]], &mut [u64]),
    hi_mul: HiMul,
) -> bool {
    rect_powers(normalize(x, xs), pw, zs, tmp, hi_mul);
    let kernel = |bit: Bit, budget, acc: &mut [u64]| {
        let p = (&*pw, &*zs);
        rect_hyper::<Narrow<S>>(a, b, bit, budget, p, acc, tmp, hi_mul)
    };
    horner_run((k - bits(x), a, zeros), out, acc, S::SIGNED, kernel)
}

// Rectangular evaluation at an explicit block length. Used by the default policy too.
#[doc(hidden)]
pub fn rect_dyn_m<S: WideHyperSeries>(
    ctx: &DynHyperCtx,
    a: u64,
    b: u64,
    zeros: usize,
    out: &mut [u64],
    m: usize,
) -> bool {
    assert!(
        m >= 1 && m as u64 <= b - a,
        "invalid rectangular block length"
    );
    let x = ctx.x.as_deref().expect("rect needs x != 1");
    let n = horner_acc_len(out.len(), true);
    let mut scratch = ScratchGuard::acquire();
    let [acc, tmp, xs, zs] = scratch.get_splits([n, n, x.len(), m]);
    let mut powers = ScratchGuard::acquire();
    let mut pw = powers.get_split_vec(std::iter::repeat(n).take(m));
    let (bufs, p) = ([acc, tmp, xs], (&mut pw[..], zs));
    hyper_rect::<S>((ctx.shift, x), a, b, zeros, out, bufs, p, hi_mul_dyn)
}

// series_eval's dispatch predicts each eligible algorithm's time and takes the fastest. Dynamic
// nodes compare fitted runtime equations, one an algorithm (`SeriesEq` over `Parts`, further
// down). Static nodes keep the older model, approximate like every static decision: each
// algorithm's kernel primitive counts (`Work`), read off the kernels, times one ns constant per
// primitive shared by the three (`series.ns_*`), with products as mul_work's schoolbook limb
// products along mul_static's tiers. See docs/series_cutoff_tuning.md.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    BBP,
    Engel,
    Hyper,
}

// A node part: P, Q or Hyper's R.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    P,
    Q,
    R,
}

impl Family {
    // A merge's products, as pairs of its children's parts: BBP's lq rp, lp rq and lq rq;
    // Engel's last two; Hyper's rp lr, lr rr, lp rq and lq rq. A leaf's term multiplies
    // each pair's first part by a coefficient.
    fn products(self) -> &'static [(Part, Part)] {
        use Part::*;
        match self {
            Family::BBP => &[(Q, P), (P, Q), (Q, Q)],
            Family::Engel => &[(P, Q), (Q, Q)],
            Family::Hyper => &[(R, P), (R, R), (P, Q), (Q, Q)],
        }
    }
}

// A static algorithm's predicted work, by primitive.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Work {
    // Limbs of div_prim and div_prim2 passes, and Knuth quotient limbs.
    pub div: f64,
    // Limbs times scalar limbs of linear passes: products and multiply-adds by a one- or
    // two-limb scalar, additions, negations, clears and copies.
    pub scalar: f64,
    // Products of multi-limb operands, in schoolbook limb products (mul_work).
    pub product: f64,
    // Horner's and rect's terms, for their group and weight bookkeeping.
    pub term: f64,
    // Kernel calls (passes, products, divisions), for their fixed cost.
    pub call: f64,
    // Limbs of static [u64; N] arrays zero-filled or moved.
    pub fill: f64,
}

// The static model's ns per unit of each kind of work (the `series.ns_*` knobs), read once per
// decision (invariant 7).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeriesModel {
    pub div: f64,
    pub scalar: f64,
    pub product: f64,
    pub term: f64,
    pub call: f64,
    pub fill: f64,
}

impl SeriesModel {
    pub fn read() -> Self {
        SeriesModel {
            div: k::series::ns_div(),
            scalar: k::series::ns_scalar(),
            product: k::series::ns_product(),
            term: k::series::ns_term(),
            call: k::series::ns_call(),
            fill: k::series::ns_fill(),
        }
    }
}

impl Work {
    // The predicted time in ns under the model `m`.
    pub fn ns_with(&self, m: &SeriesModel) -> f64 {
        m.div * self.div + m.scalar * self.scalar + m.product * self.product + m.term * self.term + m.call * self.call + m.fill * self.fill
    }
}

// Each eligible algorithm's predicted ns, and rect's block length.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Costs {
    pub horner: Option<f64>,
    pub rect: Option<(usize, f64)>,
    pub bin_split: f64,
}

impl Costs {
    pub fn of(&self, alg: SeriesAlg) -> Option<f64> {
        match alg {
            SeriesAlg::Horner => self.horner,
            SeriesAlg::Rect => self.rect.map(|r| r.1),
            SeriesAlg::BinSplit => Some(self.bin_split),
        }
    }

    // The eligible algorithm with the least predicted time (binary splitting on a tie, and when
    // a prediction isn't a number).
    pub fn pick(&self) -> SeriesAlg {
        let mut best = (SeriesAlg::BinSplit, self.bin_split);
        for alg in [SeriesAlg::Rect, SeriesAlg::Horner] {
            if let Some(ns) = self.of(alg).filter(|&ns| ns <= best.1) {
                best = (alg, ns);
            }
        }
        best.0
    }
}

/// The small-series table, `series.small` (a derived knob, as `mul.kara` is): what to run for a
/// short series without reading an equation. It is a two-input table whose row key is
/// `class * SMALL_ROWS + the frame's bucket` (`small_frame`: the octave of its limbs, to 65535)
/// and whose other input is the terms; a cell is an index into `SMALL_PICKS`, the last of which
/// hands the range to the equations (every cell past the table's extent). A class is what the
/// pick turns on besides the frame and the terms, in buckets: the kind and x's width
/// (`SMALL_KINDS`: BBP, Engel, and Hyper with no x, an x of up to 16 bits, of up to a limb, of 2
/// limbs, 3-8, 9-16, 17-32 or more), q's bits at the range's midpoint (`SMALL_Q_BITS`: to 8, 16,
/// 32 and 64), and whether r is past 1 (a series whose r is 1 converges by its q as well as by
/// its shift). A bucket stands for its top: binary splitting's numbers grow by x's and q's bits
/// a term, and its division is a pass or a division by whether Q fits a limb, so a class that
/// read low would send it ranges it loses. The tuner derives it from the equations at each
/// class's representative shape (`Shape::of_class`, `tuner/derive.rs`), so it times nothing.
///
/// The layout is the table's size (2026-11-12, the user: "definitely shrink the size of the
/// series table"): with frames by half octaves to 255 limbs, six q buckets (25 and 42 bits
/// among them) and x of 3-4 and 5-8 limbs apart, the M1's table was 6563 change points; with
/// this one 2602, and its picks lose the same measured time on unseen short series (0.34% and
/// 1.60% on the report's halves, from 0.35% and 1.58%). The 8-bit and the 16-bit q buckets both
/// have to stay: without the first the picks lose 0.54% and 1.95%, without the second one
/// series runs 98% slow. A changed layout needs the series section's `derive` number raised
/// (`tuning/schema.rs`), so a table of the old layout is not read as one of the new.
#[doc(hidden)]
pub const SMALL_PICKS: [Option<SeriesAlg>; 4] = [Some(SeriesAlg::Horner), Some(SeriesAlg::Rect), Some(SeriesAlg::BinSplit), None];
#[doc(hidden)]
pub const SMALL_ROWS: usize = 16;
#[doc(hidden)]
pub const SMALL_KINDS: usize = 10;
#[doc(hidden)]
pub const SMALL_Q_BITS: [u64; 4] = [8, 16, 32, 64];
#[doc(hidden)]
pub const SMALL_CLASSES: usize = SMALL_KINDS * SMALL_Q_BITS.len() * 2;
// Each Hyper kind's representative x, in bits, from kind 2 up.
const SMALL_X_REP: [u64; 8] = [0, 12, 60, 2 * 64 - 8, 7 * 64, 14 * 64, 28 * 64, 48 * 64];

/// A Hyper range's kind in the table, by x's bits (0: x = 1).
#[doc(hidden)]
pub fn small_kind_hyper(x_bits: u64) -> usize {
    match x_bits.div_ceil(64) {
        0 => 2,
        1 if x_bits <= 16 => 3,
        1 => 4,
        2 => 5,
        3..=8 => 6,
        9..=16 => 7,
        17..=32 => 8,
        _ => 9,
    }
}

/// The class of a kind (0: BBP, 1: Engel, `small_kind_hyper`) with q and r of these bits at the
/// range's midpoint (r's 0 without one): None past 64 bits.
#[doc(hidden)]
pub fn small_class(kind: usize, q_bits: u64, r_bits: u64) -> Option<usize> {
    let q = SMALL_Q_BITS.iter().position(|&top| q_bits <= top)?;
    (r_bits <= 64).then_some((kind * SMALL_Q_BITS.len() + q) * 2 + usize::from(r_bits > 1))
}

/// A frame's bucket in the table, the octave of its limbs: 1, 2-3, 4-7, ... to 32768-65535
/// (`SMALL_ROWS` buckets). None from 65536 limbs. Every series the short prices put under the
/// table's extent has to have a bucket: the equations read at run time are fitted to the long
/// ones, and are far off on a short one (a 3-term series in 648 limbs, when the buckets stopped
/// at 255, ran 66% slow by their pick).
#[doc(hidden)]
pub fn small_frame(len: usize) -> Option<usize> {
    match len {
        0 | 65536.. => None,
        _ => Some(len.ilog2() as usize),
    }
}

/// The frames a bucket of `small_frame` holds, first and last.
#[doc(hidden)]
pub fn small_frame_span(bucket: usize) -> (usize, usize) {
    (1 << bucket, (2 << bucket) - 1)
}

/// The table's pick for a class, a frame and a term count, at an explicit table: None where it
/// hands the range to the equations (and for a frame past the table's buckets).
#[doc(hidden)]
pub fn small_pick(table: &crate::tuning::table::Table2, class: usize, len: usize, terms: u64) -> Option<SeriesAlg> {
    let pick = table.pick(class * SMALL_ROWS + small_frame(len)?, terms.min(u32::MAX as u64) as usize);
    SMALL_PICKS.get(pick).copied().flatten()
}

/// The dispatch's guards (knobs `series.bin_split_beta`, `series.rect_rho`,
/// `series.rect_x_limbs`): two ratios of the shape that settle a range before any equation is
/// read, where the algorithms are far apart. Read once per decision.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeriesGuards {
    /// Binary splitting isn't predicted, or taken, where beta passes this and another algorithm
    /// can sum the range.
    pub bin_split_beta: f64,
    /// Rect is taken outright where rho reaches `rect_rho` and x has at least `rect_x` limbs.
    pub rect_rho: f64,
    pub rect_x: usize,
}

impl SeriesGuards {
    pub fn read() -> Self {
        use k::series as s;
        SeriesGuards { bin_split_beta: s::bin_split_beta(), rect_rho: s::rect_rho(), rect_x: s::rect_x_limbs() }
    }
}

/// What the guards settle for a range.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Guarded {
    pub rect_first: bool,
    pub no_bin_split: bool,
}

// What the cost model sees of a call, O(1) from the series at its midpoint.
#[doc(hidden)]
pub struct Shape {
    family: Family,
    t: u64,
    len: usize,
    zeros: usize,
    // Horner's frame shift per term, x's limbs (0 for x = 1), and the leaves' shift per term.
    k: u64,
    xl: usize,
    shift: u64,
    signed: bool,
    // q(n)'s and p(n)'s bits at the midpoint.
    qb: u64,
    pb: u64,
    // r(n)'s bits there, x's bits, and log2 of q(n) and of r(n): the bits a term adds to Q and
    // (with x's) to R.
    rb: u64,
    xb: u64,
    lq: f64,
    lr: f64,
    // log2 of q(n) at the range's two Gauss points, whose mean is lq: what a group of Horner's
    // holds is a whole number of terms at each.
    gq: [f64; 2],
    // Whether the first term's q is 1: a group that holds that term alone divides by nothing.
    q1: bool,
    // The range's P (Node::limbs before N's cap), Q and R, in limbs.
    p: u64,
    q: u64,
    r: u64,
    // N for static nodes, 0 for dynamic ones.
    cap: usize,
    // Node::leaf_limbs() and leaf_terms(), and whether halves build in parallel.
    leaf: (u64, u64),
    par: bool,
}

// A pass by a `limbs`-limb scalar, per limb: mul_prim2 costs about four mul_prim passes.
fn scalar_units(limbs: usize) -> f64 {
    (limbs * limbs) as f64
}

struct RectFixed {
    p: usize,
    g: u64,
    limbs: (usize, usize),
    steps: f64,
    top: f64,
    power: f64,
}

// Limbs of a coefficient or weight of `bits` bits: one or two.
fn coef_limbs(bits: u64) -> usize {
    1 + usize::from(bits > 64)
}

impl Shape {
    // A range's Shape: the frame shifts k = shift - bits(x) a term, and P (the tree's width) is
    // the larger of Q 2^(shift t) and R x^t, as Node::limbs estimates it. BBP and Engel have
    // no x and r = 0.
    fn new<N: Node>(
        family: Family,
        (a, b): (u64, u64),
        (len, zeros): (usize, usize),
        (shift, x_bits, xl, signed): (u64, u64, usize, bool),
        (p, q, r): (
            impl Fn(u128) -> u128,
            impl Fn(u128) -> u128,
            impl Fn(u128) -> u128,
        ),
        cap: usize,
    ) -> Shape {
        let mid = (a + (b - a) / 2) as u128;
        let rm = r(mid);
        // The mean of log2 f over the range, from the two Gauss points (the midpoint alone is
        // half a bit high for a coefficient that grows like n).
        // log2 to within 0.01, without libm: the bit length and a parabola across the top bits.
        let log2 = |v: u128| {
            if v <= 1 {
                return 0.0;
            }
            let b = 127 - v.leading_zeros();
            let f = (v << (127 - b) >> 75) as f64 / (1u64 << 52) as f64 - 1.0;
            b as f64 + f + 0.344 * f * (1.0 - f)
        };
        let at = |x: f64| (a as f64 + (b - a) as f64 * x) as u128;
        let (lo, hi) = (at(0.211_324_865), at(0.788_675_135));
        let gq = [log2(q(lo)), log2(q(hi))];
        let (lq, lr) = ((gq[0] + gq[1]) / 2.0, (log2(r(lo)) + log2(r(hi))) / 2.0);
        let rl = est_limbs(a, b, r, x_bits);
        Shape {
            rb: bit_len128(rm),
            xb: x_bits,
            lq,
            lr,
            gq,
            q1: q(a as u128) == 1,
            family,
            t: b - a,
            len,
            zeros,
            k: shift.saturating_sub(x_bits),
            xl,
            shift,
            signed,
            qb: bit_len128(q(mid)),
            pb: bit_len128(p(mid)),
            p: est_limbs(a, b, &q, shift).max(rl),
            q: est_limbs(a, b, &q, 0),
            r: rl,
            cap,
            leaf: (N::leaf_limbs(), N::leaf_terms()),
            par: N::PARALLEL,
        }
    }

    /// The shape that stands for a class of the small-series table at a frame of `len` limbs
    /// and `t` terms, and what can sum it: a series that has converged to the frame, as the
    /// typed entries' ranges have. Its first term sits a term's worth of bits below the top
    /// (the zero limbs the entry skips), so the bits a term fixes are the frame's over t - 1;
    /// what the coefficients converge by on their own (q's bits, for Engel and for a Hyper r of
    /// 1) comes off that, and the rest is the shift's. None for a class no series has.
    pub fn of_class(class: usize, len: usize, t: u64) -> Option<(Shape, Eligible)> {
        let (rc, qc, kind) = (class % 2, class / 2 % SMALL_Q_BITS.len(), class / 2 / SMALL_Q_BITS.len());
        if kind >= SMALL_KINDS || (kind < 2 && rc == 1) {
            return None;
        }
        let (family, xb, leaf) = match kind {
            0 => (Family::BBP, 0, (k::bin_split::dyn_bbp_leaf_limbs(), 1)),
            1 => (Family::Engel, 0, (k::bin_split::dyn_engel_leaf_limbs(), 1)),
            _ => (Family::Hyper, SMALL_X_REP[kind - 2], (k::bin_split::dyn_hyper_leaf_limbs(), k::bin_split::dyn_hyper_leaf_terms())),
        };
        let xl = xb.div_ceil(64) as usize;
        let qb = SMALL_Q_BITS[qc];
        let lq = qb as f64 - 0.5;
        let (rb, lr) = match (family, rc) {
            (Family::Hyper, 1) => (qb, lq),
            (Family::Hyper, _) => (1, 0.0),
            _ => (0, 0.0),
        };
        let own = if family == Family::BBP || rc == 1 { 0.0 } else { lq };
        let decay = (64.0 * len as f64 / (t.max(2) - 1) as f64 - own).max(1.0) as u64;
        let shift = decay + xb;
        let of = |bits: u64| t.saturating_mul(bits).div_ceil(64);
        let r = if family == Family::Hyper { of(rb + xb) } else { 0 };
        let shape = Shape {
            family,
            t,
            len,
            zeros: ((decay - 1) / 64) as usize,
            k: decay,
            xl,
            shift,
            signed: false,
            qb,
            pb: 1,
            rb,
            xb,
            lq,
            lr,
            gq: [lq, lq],
            q1: false,
            p: of(qb + shift).max(r),
            q: of(qb),
            r,
            cap: 0,
            leaf,
            par: true,
        };
        Some((shape, Eligible { horner: true, rect: xl >= 2 }))
    }

    /// Every eligible algorithm's predicted ns for the shape (`costs`, for the tuner's
    /// derivations).
    pub fn predictions(&self, ok: Eligible) -> Costs {
        self.costs(ok, None)
    }

    /// rho: x's bits over the bits a term fixes (the frame's bits over the terms). It is what
    /// rect against Horner turns on: once a block's power fills the frame, rect's products by x
    /// come to about 2 / sqrt(rho) of Horner's. 0 without an x.
    pub fn rho(&self) -> f64 {
        self.xb as f64 * self.t as f64 / (64.0 * self.len as f64)
    }

    /// x's limbs (0 without an x).
    pub fn x_limbs(&self) -> usize {
        self.xl
    }

    /// beta: the exact fraction's width, the tree's P, over the frame. Binary splitting's
    /// numbers are that fraction's, so its work follows beta where Horner's and rect's follow
    /// the frame. (P carries the shift as well as x and the coefficients, so this is at most 1
    /// above (x's bits + the coefficients') over the bits a term fixes.)
    pub fn beta(&self) -> f64 {
        self.p as f64 / self.len as f64
    }

    /// The guards at explicit knobs: rect outright for a wide x at a high rho, and binary
    /// splitting left out at a high beta while another algorithm can sum the range.
    pub fn guards(&self, ok: Eligible, g: &SeriesGuards) -> Guarded {
        Guarded {
            rect_first: ok.rect && self.xl >= g.rect_x && self.rho() >= g.rect_rho,
            no_bin_split: (ok.horner || ok.rect) && self.beta() > g.bin_split_beta,
        }
    }

    // The guards under the live knobs. Static nodes have none: their model is approximate, and
    // their trees are cut at N.
    fn live_guards(&self, ok: Eligible) -> Guarded {
        if self.stat() {
            return Guarded::default();
        }
        self.guards(ok, &SeriesGuards::read())
    }

    // The algorithm to run and rect's block length under the live guards.
    fn decide(&self, ok: Eligible) -> (SeriesAlg, Option<usize>) {
        self.decide_at(ok, self.live_guards(ok))
    }

    /// The algorithm to run and rect's block length, given the guards' verdict: what they
    /// settle, else the least prediction among what they leave.
    pub fn decide_at(&self, ok: Eligible, g: Guarded) -> (SeriesAlg, Option<usize>) {
        if g.rect_first {
            // The balance of blocks against powers and adds, unpredicted: at a high rho a
            // block's power fills the frame well before the balance, which is then the best of
            // rect_len's candidates.
            return (SeriesAlg::Rect, Some(self.rect_cands(&SeriesEnv::read(), &SeriesEq::read())[0]));
        }
        let costs = self.costs_without(ok, None, g.no_bin_split);
        (costs.pick(), costs.rect.map(|r| r.0))
    }

    /// Every eligible algorithm's predicted ns under `eq` (a dynamic node's shape), rect's at
    /// the block length the run takes, which is the live equations' whatever `eq` is: what the
    /// table's derivation and the tuner's reports read for prices other than the live ones.
    pub fn predictions_at(&self, ok: Eligible, eq: &SeriesEq) -> Costs {
        let env = SeriesEnv::read();
        Costs {
            horner: ok.horner.then(|| self.horner_parts(&env).ns_with(&eq.horner, 0.0)),
            rect: ok.rect.then(|| {
                let m = self.rect_m();
                (m, self.rect_parts(m, &env).ns_with(&eq.rect, 0.0))
            }),
            bin_split: self.bin_split_parts(&env).ns_with(&eq.bin_split, eq.wake),
        }
    }

    // Every eligible algorithm's prediction: what the tests and the tuner read. The dispatch
    // itself goes through `decide`, which the guards may spare some of these.
    fn costs(&self, ok: Eligible, m: Option<usize>) -> Costs {
        self.costs_without(ok, m, false)
    }

    // Dynamic nodes: the fitted equations of each algorithm (`SeriesEq` over its `Parts`).
    // Static nodes keep the shared prices over `Work` (approximate, like every static decision).
    // With `no_bin_split` the tree isn't walked and reads as infinite (the beta guard).
    fn costs_without(&self, ok: Eligible, m: Option<usize>, no_bin_split: bool) -> Costs {
        if self.stat() {
            let model = SeriesModel::read();
            let rect = |m| match m {
                Some(m) => (m, self.rect(&self.rect_fixed(), m)),
                None => self.rect_best(),
            };
            return Costs {
                horner: ok.horner.then(|| self.horner().ns_with(&model)),
                rect: ok.rect.then(|| rect(m)).map(|(m, w)| (m, w.ns_with(&model))),
                bin_split: self.bin_split().ns_with(&model),
            };
        }
        let (env, eq) = (SeriesEnv::read(), SeriesEq::read());
        let horner = ok.horner.then(|| self.horner_parts(&env));
        // Rect can only save Horner's products by x (it divides in a narrower accumulator, so
        // more often), and pays its fixed cost for it: where those products cost less than
        // that, its block lengths aren't worth predicting, and it reads as Horner plus the
        // fixed cost, at block length 1.
        let by_x = horner.map_or(f64::INFINITY, |w| w.product * eq.horner[0] + w.mul2 * eq.horner[5]);
        let horner = horner.map(|w| w.ns_with(&eq.horner, 0.0));
        Costs {
            horner,
            rect: ok.rect.then(|| match (m, horner) {
                (Some(m), _) => (m, self.rect_parts(m, &env).ns_with(&eq.rect, 0.0)),
                (None, Some(h)) if by_x <= eq.rect[11] => (1, h + eq.rect[11]),
                (None, _) => self.rect_len(&env, &eq),
            }),
            bin_split: if no_bin_split { f64::INFINITY } else { self.bin_split_parts(&env).ns_with(&eq.bin_split, eq.wake) },
        }
    }

    // Rect's block length, as the run takes it when none is given.
    fn rect_m(&self) -> usize {
        if self.stat() {
            self.rect_best().0
        } else {
            self.rect_len(&SeriesEnv::read(), &SeriesEq::read()).0
        }
    }

    fn stat(&self) -> bool {
        self.cap != 0
    }

    // Limbs of `arrays` static [u64; N] arrays.
    fn fill(&self, arrays: f64) -> f64 {
        arrays * self.cap as f64
    }

    // Terms per division in an acc-limb accumulator, as `group` stops: Q = ∏q within the
    // budget, and p Q within 128 bits.
    fn group(&self, acc: usize) -> u64 {
        let q = self.qb.max(1);
        let g = (group_budget(acc, self.len) / q).min(128u64.saturating_sub(self.pb) / q);
        g.clamp(1, GROUP_MAX as u64)
    }

    // Limbs of a group's R (at most its Q) and of its terms' weights p Q / q.
    fn group_limbs(&self, g: u64) -> (usize, usize) {
        let q = (g * self.qb).min(128);
        (coef_limbs(q), coef_limbs((q + self.pb).min(128)))
    }

    // The frame's limbs summed over terms 1, 1 + stride, ... up to t, in an acc-limb
    // accumulator: term i spans about w + 1 - k i / 64 limbs, capped at acc.
    fn frame(&self, t: u64, acc: usize, stride: u64) -> f64 {
        let (g, top, cap) = (
            stride.max(1),
            (self.len + self.zeros + HORNER_GUARD + 1) as u64,
            acc as u64,
        );
        // How many of the sampled terms lie at or before term i.
        let count = |i: u64| if i == 0 { 0 } else { (i.min(t) - 1) / g + 1 };
        if self.k == 0 {
            return (cap.min(top) * count(t)) as f64;
        }
        // Terms up to i = (top - cap) / s span the whole accumulator, and those below top / s
        // part of it, with s = k / 64 limbs a term.
        let full = count(64 * top.saturating_sub(cap) / self.k);
        let n = (count((64 * top - 1) / self.k) - full) as f64;
        let (full, s, g) = (full as f64, self.k as f64 / 64.0, g as f64);
        let linear = n * (top as f64 - s) - s * g * (2.0 * full + n - 1.0) * n / 2.0;
        full * cap as f64 + linear.max(0.0)
    }

    // The static nodes' counts from here to `bin_split`. Horner: per group a division, BBP's
    // temp clear and add, and Hyper's product by R; per term its weight's add and bookkeeping,
    // and with x a product by r and a high product by x. A signed term usually flips the sum's
    // sign, negating the active limbs.
    fn horner(&self) -> Work {
        // Eligibility already requires a narrow series where only narrow Horner fits.
        let acc = horner_room(self.len, true, self.cap);
        let (g, t) = (self.group(acc), self.t as f64);
        let (steps, groups) = (self.frame(self.t, acc, 1), self.frame(self.t, acc, g));
        let n_groups = self.t.div_ceil(g) as f64;
        let mut work = Work {
            div: groups,
            term: t,
            call: n_groups + 2.0 * t,
            // A static Horner's [u64; N] arrays: acc and t; acc; acc, tmp and xs.
            fill: self.fill(match self.family {
                Family::BBP => 2.0,
                Family::Engel => 1.0,
                Family::Hyper => 3.0,
            }),
            ..Work::default()
        };
        match (self.family, self.xl) {
            (Family::BBP, _) => {
                work.scalar = 2.0 * groups;
                work.call += 2.0 * n_groups;
            }
            (Family::Engel, _) => (),
            (Family::Hyper, 0) => {
                work.scalar = groups * scalar_units(self.group_limbs(g).0);
                work.call += n_groups;
            }
            (Family::Hyper, xl) => {
                let r = scalar_units(coef_limbs(self.qb));
                work.scalar = steps * (r + 1.0);
                work.product = steps * mul_work(acc, xl.min(acc)) / acc as f64;
                work.call += 3.0 * t;
            }
        }
        if self.signed {
            work.scalar += steps;
        }
        work
    }

    // Rect's work that doesn't depend on its block length.
    fn rect_fixed(&self) -> RectFixed {
        let p = horner_acc_len(self.len, true);
        let g = self.group(p);
        RectFixed {
            p,
            g,
            limbs: self.group_limbs(g),
            steps: self.frame(self.t, p, 1),
            top: self.frame(1, p, 1),
            power: mul_work(p, self.xl.max(1)),
        }
    }

    // Powers x'^2..x'^m, a high product by x'^m per block before its terms, and grouped
    // scalar steps adding weights times the powers' significant limbs.
    fn rect(&self, f: &RectFixed, m: usize) -> Work {
        let (p, xl, (rl, wl)) = (f.p, self.xl.max(1), f.limbs);
        let (mf, t) = (m as f64, self.t as f64);
        // Groups don't cross a block: ⌈m / g⌉ divisions a block.
        let per_block = (m as u64).div_ceil(f.g) as f64;
        let tops = self.frame(self.t, p, m as u64);
        let n_blocks = self.t.div_ceil(m as u64) as f64;
        let groups = tops * per_block;
        // A block's product comes before its terms, at the deeper block's top limbs.
        let blocks = tops - f.top;
        // The mean of min(p, j xl) over j = 1..=m.
        let short = (p / xl).min(m);
        let adds = (xl * short * (short + 1) / 2 + (m - short) * p) as f64 / mf;
        let block = mul_work(p, (m * xl).min(p)) / p as f64;
        let flips = if self.signed { f.steps } else { 0.0 };
        Work {
            div: groups,
            scalar: groups * scalar_units(rl) + t * adds * scalar_units(wl) + blocks + flips,
            product: (mf - 1.0) * f.power + blocks * block,
            term: t,
            call: mf + n_blocks * (2.0 + 2.0 * per_block) + 2.0 * wl as f64 * t,
            fill: self.fill((RECT_MAX_M + 3) as f64),
        }
    }

    // Rect's block length and work: the fastest of the balance of block products against
    // powers and term adds, 1, and the length whose power first fills the accumulator.
    fn rect_best(&self) -> (usize, Work) {
        let hi = RECT_MAX_M.min(usize::try_from(self.t).unwrap_or(usize::MAX)).max(1);
        let (f, xl) = (self.rect_fixed(), self.xl.max(1));
        let model = SeriesModel::read();
        // Time ~ a / m + b m: full blocks and their divisions against the powers and the
        // growth of the term adds while powers are short.
        let a = model.product * f.steps / f.p as f64 * mul_work(f.p, f.p)
            + model.div * f.steps;
        let b = model.product * f.power + model.scalar * self.t as f64 * xl as f64 / 2.0;
        let ms = [(a / b).sqrt().round() as usize, 1, f.p.div_ceil(xl)].map(|m| m.clamp(1, hi));
        let mut best = (0, Work::default(), f64::INFINITY);
        for (i, &m) in ms.iter().enumerate() {
            if ms[..i].contains(&m) {
                continue;
            }
            let w = self.rect(&f, m);
            let ns = w.ns_with(&model);
            // The first candidate always stands, so an infinite or NaN prediction (a model
            // coefficient at its maximum) still gives a valid block length.
            if best.0 == 0 || ns < best.2 {
                best = (m, w, ns);
            }
        }
        (best.0, best.1)
    }

    // A static tree: leaves (each term a scalar pass per merge product), merges level by level,
    // and finalize's reciprocal and high product.
    fn bin_split(&self) -> Work {
        let (fam, t, n) = (self.family, self.t as f64, self.cap as f64);
        let parts = if fam == Family::Hyper { 3.0 } else { 2.0 };
        // Levels of splits: a node at level d has t / 2^d terms and E / 2^d limbs, capped at N.
        let (e, cap) = (self.p.max(1) as u128, self.cap as u128);
        let (leaf_limbs, leaf_terms) = (self.leaf.0 as u128, self.leaf.1.max(1) as u128);
        let mut d = 0;
        while d < 60 {
            let small = e < leaf_limbs << d || cap < leaf_limbs;
            if self.t as u128 <= leaf_terms << d || small && e <= cap << d {
                break;
            }
            d += 1;
        }
        let leaves = (1u64 << d) as f64;
        // A part's limbs at level j, and its product with another's.
        let part = |x: Part, j: i32| {
            let l = match x {
                Part::P => self.p,
                Part::Q => self.q,
                Part::R => self.r,
            };
            (l.max(1) as f64 / (1u64 << j) as f64).min(n)
        };
        let product = |a: f64, b: f64| mul_work(a.ceil().max(1.0) as usize, b.ceil().max(1.0) as usize);
        let pairs = fam.products();
        let (mut merges, mut passes) = (0.0, 0.0);
        for j in 0..d as i32 {
            let level: f64 = pairs.iter().map(|&(a, b)| product(part(a, j + 1), part(b, j + 1))).sum();
            let nodes = (1u64 << j) as f64;
            merges += nodes * level;
            // The merged P is shifted into place and summed: two passes.
            passes += nodes * 2.0 * part(Part::P, j);
        }
        // A leaf term multiplies each part, about half built, by a coefficient, and an x-leaf
        // term also its running term, about xl limbs a term, by x.
        let (d, cl) = (d as i32, coef_limbs(self.qb.max(self.pb)));
        // P's 2^shift moves P by whole limbs, and bit-shifts it unless the shift's low bits
        // fit beside q(n) in the multiplier (mul_shl).
        let (whole, bits) = (
            self.shift >= 64,
            self.shift % 64 > 64u64.saturating_sub(self.qb),
        );
        let units = |x: Part| match x {
            Part::R if self.xl != 0 => scalar_units(2),
            Part::P => scalar_units(cl) + f64::from(u8::from(whole) + u8::from(bits)),
            _ => scalar_units(cl),
        };
        let leaf_term: f64 = pairs
            .iter()
            .map(|&(a, _)| part(a, d) / 2.0 * units(a))
            .sum();
        let (mut leaf_product, mut leaf_calls, mut leaf_terms) = (0.0, pairs.len() as f64, 0.0);
        let mut leaf_arrays = 2.0 * parts;
        if self.xl != 0 {
            // An x-leaf's term multiplies its running term by x, adds it to P and multiplies
            // it by r(n) in two-limb passes.
            leaf_product = t * product(t / leaves * self.xl as f64 / 2.0, self.xl as f64);
            leaf_calls += 1.0;
            leaf_arrays += 2.0;
            leaf_terms = t;
        }
        // Finalize: a reciprocal of f limbs and a high product.
        let f = self.len + 2;
        let ql = (self.q.max(1) as usize).min(f);
        let finalize = ((f * ql) as f64).min(2.0 * mul_work(f, f)) + mul_work(f, f);
        let merge_n = leaves - 1.0;
        Work {
            div: f as f64,
            scalar: t * leaf_term + passes,
            product: merges + leaf_product + finalize,
            term: leaf_terms,
            call: t * leaf_calls + merge_n * (pairs.len() + 1) as f64 + 4.0,
            // Leaves' and merges' parts, and finalize's reciprocal and product.
            fill: self.fill(leaves * leaf_arrays + merge_n * (parts + 1.0) + 4.0),
        }
    }
}

// The dynamic nodes' runtime equations (rebuilt 2026-11-10, the user's decision): each
// algorithm's time as its own parts, read off its kernel, with multi-limb products priced by
// mul's fitted equations and finalize's division by division's, and one set of fitted prices an
// algorithm (`SeriesEq`). On exp's and ln's own series their picks lose 0.1-0.2% of measured
// time where the shared prices over `Work` lost 3.8-4.3% (`cargo tune series`).

/// An algorithm's predicted work by kind of operation: the regressors of its runtime equation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Parts {
    /// Products of operands of three limbs or more, in ns by mul's equations (`MulEq`).
    pub product: f64,
    /// Finalize's division, in ns by division's equations (`DivEq`).
    pub division: f64,
    /// Limbs of div_prim passes, and of div_prim2 passes.
    pub div1: f64,
    pub div2: f64,
    /// Limbs of passes by a one-limb multiplier (mul_prim, add_mul, and a product with a
    /// one-limb side), and by a two-limb one (mul_prim2).
    pub mul1: f64,
    pub mul2: f64,
    /// Limbs of additions and bit shifts.
    pub add: f64,
    /// Limbs zero-filled or copied.
    pub light: f64,
    /// Terms: their coefficients and bookkeeping.
    pub term: f64,
    /// Calls of the linear kernels above, for their fixed cost.
    pub call: f64,
    /// Heap allocations and scratch acquisitions.
    pub alloc: f64,
    /// The call's fixed cost.
    pub one: f64,
    /// Pool wakes (priced by `bin_split.dyn_wake_ns`, not fitted).
    pub wake: f64,
}

impl Parts {
    pub const NAMES: [&'static str; 12] =
        ["product", "division", "div1", "div2", "mul1", "mul2", "add", "light", "term", "call", "alloc", "one"];

    /// The fitted columns, in `NAMES`' order.
    pub fn features(&self) -> [f64; 12] {
        [
            self.product,
            self.division,
            self.div1,
            self.div2,
            self.mul1,
            self.mul2,
            self.add,
            self.light,
            self.term,
            self.call,
            self.alloc,
            self.one,
        ]
    }

    /// The predicted ns under `prices` and a wake's price.
    pub fn ns_with(&self, prices: &[f64; 12], wake: f64) -> f64 {
        let f = self.features();
        f.iter().zip(prices).map(|(x, p)| x * p).sum::<f64>() + wake * self.wake
    }

    // c passes over `limbs` limbs each by a multiplier of `by` limbs (one or two).
    #[inline(always)]
    fn pass(&mut self, by: usize, limbs: f64, c: f64) {
        self.passes(by, c * limbs, c);
    }

    // `calls` passes over `limbs` limbs in all by a multiplier of `by` limbs.
    #[inline(always)]
    fn passes(&mut self, by: usize, limbs: f64, calls: f64) {
        if by >= 2 {
            self.mul2 += limbs;
        } else {
            self.mul1 += limbs;
        }
        self.call += calls;
    }
}

/// The series algorithms' fitted runtime equations (the user's rule for a decision with several
/// inputs): each algorithm's prices over its `Parts`, in `Parts::NAMES`' order, with 0 where it
/// has no such work (knobs `series.horner_ns_*`, `series.rect_ns_*`, `series.bin_split_ns_*`),
/// and a pool wake's (`bin_split.dyn_wake_ns`). Rect runs Horner's kernels, so it takes
/// Horner's prices for their passes and has its own for its products, calls and fixed cost,
/// and for divisions by a one-limb Q: its groups end at its blocks, so it makes those in every
/// range, where Horner's groups fill two limbs and its long series hardly make one (1 of 67
/// on the M1), which leaves Horner's fit no price for them. Read once per decision.
///
/// There are two sets (since 2026-11-11): `read`, fitted on series past the small-series
/// table's extent, which is where the dispatch reads the equations; and `read_short`
/// (`series.short_*`), fitted on the series under it, which only the table's derivation reads.
/// One fit over every size read 5-6% on the long series, pulled by the short ones, which no
/// form tried predicts within 9%.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeriesEq {
    pub horner: [f64; 12],
    pub rect: [f64; 12],
    pub bin_split: [f64; 12],
    pub wake: f64,
}

impl SeriesEq {
    /// The columns each algorithm's fit has a price for; rect's own are `RECT`, the rest of its
    /// prices Horner's.
    /// (Horner's fixed cost and binary splitting's passes by a one-limb multiplier didn't lower
    /// the error on workloads held out, in the form analysis of 2026-11-11 on the M1, and have
    /// no price.)
    pub const HORNER: [usize; 8] = [0, 2, 3, 4, 5, 7, 8, 9];
    pub const RECT: [usize; 4] = [0, 2, 9, 11];
    pub const BIN_SPLIT: [usize; 9] = [0, 1, 2, 5, 6, 7, 8, 9, 10];

    pub fn read() -> Self {
        use k::series as s;
        let horner = [
            s::horner_ns_product(),
            0.0,
            s::horner_ns_div1(),
            s::horner_ns_div2(),
            s::horner_ns_mul1(),
            s::horner_ns_mul2(),
            0.0,
            s::horner_ns_light(),
            s::horner_ns_term(),
            s::horner_ns_call(),
            0.0,
            0.0,
        ];
        let mut rect = horner;
        (rect[0], rect[2], rect[8], rect[9], rect[11]) = (s::rect_ns_product(), s::rect_ns_div1(), 0.0, s::rect_ns_call(), s::rect_ns_one());
        let bin_split = [
            s::bin_split_ns_product(),
            s::bin_split_ns_division(),
            s::bin_split_ns_div1(),
            0.0,
            0.0,
            s::bin_split_ns_mul2(),
            s::bin_split_ns_add(),
            s::bin_split_ns_light(),
            s::bin_split_ns_term(),
            s::bin_split_ns_call(),
            s::bin_split_ns_alloc(),
            0.0,
        ];
        SeriesEq { horner, rect, bin_split, wake: k::bin_split::dyn_wake_ns() }
    }

    /// The prices fitted on the series under the small-series table's extent (`series.short_*`),
    /// for the table's derivation.
    pub fn read_short() -> Self {
        use k::series as s;
        let horner = [
            s::short_horner_ns_product(),
            0.0,
            s::short_horner_ns_div1(),
            s::short_horner_ns_div2(),
            s::short_horner_ns_mul1(),
            s::short_horner_ns_mul2(),
            0.0,
            s::short_horner_ns_light(),
            s::short_horner_ns_term(),
            s::short_horner_ns_call(),
            0.0,
            0.0,
        ];
        let mut rect = horner;
        (rect[0], rect[2], rect[8], rect[9], rect[11]) =
            (s::short_rect_ns_product(), s::short_rect_ns_div1(), 0.0, s::short_rect_ns_call(), s::short_rect_ns_one());
        let bin_split = [
            s::short_bin_split_ns_product(),
            s::short_bin_split_ns_division(),
            s::short_bin_split_ns_div1(),
            0.0,
            0.0,
            s::short_bin_split_ns_mul2(),
            s::short_bin_split_ns_add(),
            s::short_bin_split_ns_light(),
            s::short_bin_split_ns_term(),
            s::short_bin_split_ns_call(),
            s::short_bin_split_ns_alloc(),
            0.0,
        ];
        SeriesEq { horner, rect, bin_split, wake: k::bin_split::dyn_wake_ns() }
    }
}

/// What the equations read besides the shape, once per decision: mul's and division's
/// equations and the tree's parallel gates. (The pool itself is only looked at by a tree wide
/// enough to split.)
#[doc(hidden)]
pub struct SeriesEnv {
    mul: MulEq,
    div: DivEq,
    kara: &'static crate::tuning::table::Table2,
    knuth_d: usize,
    par_limbs: u64,
    par_work: u64,
    /// What a parallel split gains: a level run on n threads takes 1 / (1 + gain (n - 1)) of its
    /// serial time (1: all of it).
    pub par_gain: f64,
}

impl SeriesEnv {
    pub fn read() -> Self {
        SeriesEnv {
            mul: MulEq::read(),
            div: DivEq::read(),
            kara: k::mul::kara(),
            knuth_d: k::div::dyn_knuth_d(),
            par_limbs: k::bin_split::dyn_par(),
            par_work: k::bin_split::dyn_par_work(),
            par_gain: k::series::bin_split_par_gain(),
        }
    }

    // c of mul_dyn's products of operands of about l and s limbs: a mul_prim or mul_prim2 pass
    // when a side is one or two limbs (mul's schoolbook equation was fitted from 10 limbs up),
    // else its predicted ns.
    #[inline(always)]
    fn mul(&self, l: f64, s: f64, c: f64, w: &mut Parts) {
        let (l, s) = (l.ceil().max(1.0) as usize, s.ceil().max(1.0) as usize);
        let (l, s) = (l.max(s), l.min(s));
        if s <= 2 {
            w.pass(s, l as f64, c);
        } else {
            w.product += c * self.mul.mul_ns(l, s, self.kara);
        }
    }

    // hi_mul_dyn's predicted ns for an l-limb operand by an s-limb one (2 or more) into l limbs:
    // schoolbook's rows without the triangle below the window up to `mul.dyn_partial_mul`
    // (hi_mul_buf), the guarded product of the tops above.
    fn hi_ns(&self, l: f64, s: f64) -> f64 {
        let l = l.ceil().max(1.0) as usize;
        let s = (s.ceil().max(1.0) as usize).min(l);
        if l <= self.div.partial_mul {
            let [a, c] = self.mul.school;
            a * (l * s - s * (s - 1) / 2) as f64 + c
        } else {
            self.mul.mul_ns(l, s, self.kara)
        }
    }

    // c high products of an l-limb accumulator by an s-limb operand, as hi_mul_dyn takes them: a
    // mul_prim pass for one limb (the whole product fits); hi_mul_buf up to
    // `mul.dyn_partial_mul` limbs; above it the guarded product, a mul_prim2 pass for two limbs.
    #[inline(always)]
    fn hi(&self, l: f64, s: f64, c: f64, w: &mut Parts) {
        if s <= 1.0 {
            w.pass(1, l, c);
        } else if s <= 2.0 && l > self.div.partial_mul as f64 {
            w.pass(2, l, c);
        } else {
            w.product += c * self.hi_ns(l, s);
        }
    }

    // div_dyn's predicted ns at a logical quotient of q limbs over d: Knuth's without the other
    // equations at a divisor within its guard, as the dispatcher itself decides.
    fn div_ns(&self, q: usize, d: usize) -> f64 {
        if d <= self.knuth_d {
            return self.div.knuth_ns(q, d);
        }
        dyn_div_best(q, d, false, &self.div, &self.mul, self.kara).1
    }
}

// The sum of f(i) over i = 1..=n: exactly up to 8 terms, from 8 midpoints past that (exact
// where f is linear, as a schoolbook product is in one side).
fn sum_sampled(n: u64, mut f: impl FnMut(f64, f64)) {
    if n <= 8 {
        return (1..=n).for_each(|i| f(i as f64, 1.0));
    }
    let step = n as f64 / 8.0;
    (0..8).for_each(|j| f((j as f64 + 0.5) * step + 0.5, step))
}

impl Shape {
    fn parts(&self, ok: Eligible, m: Option<usize>, env: &SeriesEnv) -> [Option<Parts>; 3] {
        [
            ok.horner.then(|| self.horner_parts(env)),
            ok.rect.then(|| self.rect_parts(m.unwrap_or_else(|| self.rect_m()), env)),
            Some(self.bin_split_parts(env)),
        ]
    }

    // Node::limbs for a node of n terms: its P, in whole limbs of whole bits a term.
    #[inline(always)]
    fn est(&self, n: u64) -> u64 {
        let of = |bits: u64| n.saturating_mul(bits).div_ceil(64);
        of(self.qb + self.shift).max(of(self.rb + self.xb))
    }

    // A part's limbs in a node of n terms. x^n has about bits(x) - 1/2 bits a term.
    #[inline(always)]
    fn part(&self, x: Part, n: f64) -> f64 {
        let r = self.lr + if self.xb == 0 { 0.0 } else { self.xb as f64 - 0.5 };
        let bits = match x {
            Part::Q => self.lq,
            Part::R => r,
            Part::P => (self.lq + self.shift as f64).max(r),
        };
        (n * bits / 64.0).ceil().max(1.0)
    }

    // The accumulator's active limbs at term i of the range (from 1), as `frame` sums them.
    fn active(&self, i: f64, acc: usize) -> f64 {
        let top = (self.len + self.zeros + HORNER_GUARD + 1) as f64;
        (top - self.k as f64 * i / 64.0).clamp(1.0, acc as f64)
    }

    // `frame` over the terms first, first + stride, ... up to t, for a real first and stride
    // (groups hold a whole number of terms each, but not the same number).
    fn frame_from(&self, first: f64, acc: usize, stride: f64) -> f64 {
        let (t, top, cap) = (self.t as f64, (self.len + self.zeros + HORNER_GUARD + 1) as f64, acc as f64);
        if first > t {
            return 0.0;
        }
        let per = 1.0 / stride;
        let count = |i: f64| if i < first { 0.0 } else { ((i.min(t) - first) * per).floor() + 1.0 };
        if self.k == 0 {
            return cap.min(top) * count(t);
        }
        let s = self.k as f64 / 64.0;
        let full = count((top - cap) / s);
        let n = count((top - 1.0 / 64.0) / s) - full;
        let linear = n * (top - s * first) - s * stride * (2.0 * full + n - 1.0) * n / 2.0;
        full * cap + linear.max(0.0)
    }

    // The frames at the shallowest term of each group when the terms are taken g at a time (on
    // average) from the deep end, as the kernels group them: the shallowest group holds what is
    // left over, so the second starts after that.
    fn tops(&self, acc: usize, g: f64) -> f64 {
        let (t, top) = (self.t as f64, self.active(1.0, acc));
        // One group: its frame is the first term's.
        if g >= t {
            return top;
        }
        top + self.frame_from(1.0 + Self::top_group(t, g), acc, g)
    }

    // The terms in the shallowest group when t terms are taken g at a time from the deep end:
    // what the others leave, or a whole group.
    fn top_group(t: f64, g: f64) -> f64 {
        let left = t - g * (t / g).floor();
        if left < 0.5 {
            g
        } else {
            left
        }
    }

    // Terms per division in an acc-limb accumulator: as many as keep Q = ∏q within the group's
    // budget and p Q within 128 bits, at most GROUP_MAX. For a q whose size is the same across
    // the range (the same whole number of terms fit at both of its Gauss points), that whole
    // number: the kernel's groups are whole, and a 32-bit q fills a 128-bit group with exactly
    // 4 terms. For a q that grows, groups of several sizes: the budget over q's mean bits, less
    // the half term a group leaves unfilled on average. (Until 2026-11-11 the second form stood
    // for every q, and read 3.63 terms a group for that 32-bit q: three divisions for a range of
    // 8 terms that makes two, 39% high on a few terms in a wide frame. Whole numbers at each
    // Gauss point, averaged, were tried for every q and are wrong the other way for a growing
    // one: 32 terms of q(n) = n, 118 bits in all and one group, read as two, 70% high.)
    fn group_of(&self, acc: usize) -> f64 {
        let budget = group_budget(acc, self.len) as f64;
        let room = |bits: f64| (budget / bits).min((128.0 - self.pb as f64) / bits).floor();
        let (lo, hi) = (room(self.gq[0].max(0.5)), room(self.gq[1].max(0.5)));
        let q = self.lq.max(0.5);
        let g = if lo == hi { lo } else { ((budget - q / 2.0) / q).min((128.0 - self.pb as f64) / q) };
        g.clamp(1.0, GROUP_MAX as f64).min(self.t.max(1) as f64)
    }

    // Limbs of a product of g coefficients of `bits` bits each: one or two.
    fn coefs(g: f64, bits: f64) -> usize {
        1 + usize::from(g * bits > 64.0)
    }

    // Horner in the wide accumulator: per group one division by Q (div_prim2 once Q passes 64
    // bits) and, by family, BBP's numerator cleared and added, or Hyper's product by R; with
    // x, per term a product by r(n), a high product by x' and its copy back.
    fn horner_parts(&self, env: &SeriesEnv) -> Parts {
        let acc = horner_room(self.len, true, usize::MAX);
        let (g, t) = (self.group_of(acc), self.t as f64);
        let (steps, groups) = (self.frame(self.t, acc, 1), self.tops(acc, g));
        let n_groups = if g >= t { 1.0 } else { (t / g).ceil() };
        let mut w = Parts {
            term: t,
            call: n_groups + t,
            alloc: 1.0,
            one: 1.0,
            light: (acc + self.len) as f64,
            ..Parts::default()
        };
        // The shallowest group holds what the others leave: the first term alone, with q = 1,
        // divides by nothing.
        let alone = Self::top_group(t, g).round() == 1.0;
        let free = if self.q1 && alone { self.active(1.0, acc) } else { 0.0 };
        if Self::coefs(g, self.lq) == 2 {
            w.div2 = groups - free;
        } else {
            w.div1 = groups - free;
        }
        match (self.family, self.xl) {
            (Family::BBP, _) => {
                // The numerator is cleared, divided and added; its weights take a second pass
                // over the group's terms.
                w.light += groups;
                w.passes(1, groups, n_groups);
                w.call += n_groups + t;
            }
            (Family::Engel, _) => (),
            (Family::Hyper, 0) if self.rb > 1 => w.passes(Self::coefs(g, self.lr), groups, n_groups),
            (Family::Hyper, 0) => (),
            (Family::Hyper, xl) => {
                if self.rb > 1 {
                    w.passes(Self::coefs(1.0, self.lr), steps, t);
                }
                if xl == 1 {
                    w.passes(1, steps, t);
                } else if acc <= env.div.partial_mul {
                    w.product = self.school_products(acc, xl, env);
                } else {
                    sum_sampled(self.t, |i, c| env.hi(self.active(i, acc), xl as f64, c, &mut w));
                }
                w.light += steps;
                w.call += t;
            }
        }
        // A signed term usually flips the sum's sign, a pass over the active limbs.
        if self.signed {
            w.mul1 += if self.family == Family::BBP { groups } else { steps };
        }
        w
    }

    // Horner's high products by x' where every frame is hi_mul_buf's (an accumulator within
    // `mul.dyn_partial_mul` limbs), summed over the range in closed form, as the sum is taken
    // on every dispatch. A frame of L limbs by an x of xl costs schoolbook's rows without the
    // triangle below the window, with x clipped to the frame: a (L m - m (m - 1) / 2) + c
    // for m = min(xl, L). The frames run through four stretches: the whole accumulator; then
    // top - s i limbs (s = k / 64) while that holds x whole; then while it is a limb or more,
    // where m = L; then one limb.
    fn school_products(&self, acc: usize, xl: usize, env: &SeriesEnv) -> f64 {
        let [a, c] = env.mul.school;
        let (t, cap) = (self.t as f64, acc as f64);
        let top = (self.len + self.zeros + HORNER_GUARD + 1) as f64;
        let x = (xl as f64).min(cap);
        // n frames of `frames` limbs in all that hold x whole.
        let whole = |frames: f64, n: f64| a * (frames * x - n * x * (x - 1.0) / 2.0) + n * c;
        if self.k == 0 {
            // Every frame is the same.
            let (l, m) = (cap.min(top), x.min(top));
            return t * (a * (l * m - m * (m - 1.0) / 2.0) + c);
        }
        let s = self.k as f64 / 64.0;
        // The last term whose frame is `frame` limbs or more, at or after `from`.
        let until = |frame: f64, from: f64| ((top - frame) / s).floor().clamp(from, t);
        let i0 = until(cap, 0.0);
        let i1 = until(x, i0);
        let i2 = until(1.0, i1);
        // The sums of i and of i^2 over lo < i <= hi.
        let sum = |lo: f64, hi: f64| (hi * (hi + 1.0) - lo * (lo + 1.0)) / 2.0;
        let sum2 = |lo: f64, hi: f64| (hi * (hi + 1.0) * (2.0 * hi + 1.0) - lo * (lo + 1.0) * (2.0 * lo + 1.0)) / 6.0;
        let (n1, n2) = (i1 - i0, i2 - i1);
        // The frames' sum over the second stretch, and their sum and sum of squares over the third.
        let f1 = n1 * top - s * sum(i0, i1);
        let f2 = n2 * top - s * sum(i1, i2);
        let q2 = n2 * top * top - 2.0 * top * s * sum(i1, i2) + s * s * sum2(i1, i2);
        whole(i0 * cap, i0) + whole(f1, n1) + a * (q2 + f2) / 2.0 + c * n2 + (t - i2) * (a + c)
    }

    // Rect's candidate block lengths: the balance of block products against powers and term
    // adds (time ~ a / m + b m), 1, and the length whose power first fills the accumulator.
    fn rect_cands(&self, env: &SeriesEnv, eq: &SeriesEq) -> [usize; 3] {
        let hi = usize::try_from(self.t).unwrap_or(usize::MAX).max(1);
        let (p, xl) = (horner_acc_len(self.len, true), self.xl.max(1));
        let (pf, steps) = (p as f64, self.frame(self.t, p, 1));
        let a = eq.rect[0] * steps / pf * env.hi_ns(pf, pf) + eq.rect[2] * steps;
        let b = eq.rect[0] * env.hi_ns(pf, xl as f64) + eq.rect[4] * self.t as f64 * xl as f64;
        [(a / b).sqrt().round() as usize, 1, p.div_ceil(xl)].map(|m| m.clamp(1, hi))
    }

    // Rect's block length and predicted ns: the fastest of `rect_cands`.
    fn rect_len(&self, env: &SeriesEnv, eq: &SeriesEq) -> (usize, f64) {
        let ms = self.rect_cands(env, eq);
        let mut best = (0, f64::INFINITY);
        for (i, &m) in ms.iter().enumerate() {
            if ms[..i].contains(&m) {
                continue;
            }
            let ns = self.rect_parts(m, env).ns_with(&eq.rect, 0.0);
            // The first candidate always stands, so a prediction that isn't a number (a price
            // at its maximum) still gives a valid block length.
            if best.0 == 0 || ns < best.1 {
                best = (m, ns);
            }
        }
        best
    }

    // Rect at block length m: m - 1 powers (a high product and two copies each), per block
    // but the deepest a high product by its power, per group a product by R and a division,
    // and per term two passes of its weight over its power's limbs inside the frame.
    fn rect_parts(&self, m: usize, env: &SeriesEnv) -> Parts {
        let p = horner_acc_len(self.len, true);
        let (xl, mf, t) = (self.xl.max(1), m as f64, self.t as f64);
        let g = self.group_of(p).min(mf);
        let wl = Self::coefs(1.0, g * self.lq + self.pb as f64);
        let steps = self.frame(self.t, p, 1);
        let tops = self.tops(p, mf);
        let per_block = (mf / g).ceil();
        let n_blocks = self.t.div_ceil(m as u64);
        let (groups, n_groups) = (tops * per_block, n_blocks as f64 * per_block);
        let mut w = Parts {
            term: t,
            call: n_groups + 2.0 * wl as f64 * t + 2.0 * mf,
            alloc: 3.0,
            one: 1.0,
            light: (2.0 * mf - 1.0) * p as f64 + (p + self.len) as f64,
            ..Parts::default()
        };
        env.hi(p as f64, xl as f64, mf - 1.0, &mut w);
        // A block's product by its power comes before its terms, on the sum of the blocks
        // below it, whose frame is the next block's top; the deepest block has none.
        let pow = (m * xl).min(p) as f64;
        let first = 1.0 + if self.t % m as u64 == 0 { mf } else { (self.t % m as u64) as f64 };
        if n_blocks > 1 {
            sum_sampled(n_blocks - 1, |j, c| {
                let l = self.active(first + (j - 1.0) * mf, p);
                env.hi(l, pow.min(l), c, &mut w);
            });
            w.light += tops - self.active(1.0, p);
        }
        // The shallowest block's shallowest group is the first term alone when one term is left
        // over; with q = 1 it divides by nothing.
        let top_block = if self.t % m as u64 == 0 { mf } else { (self.t % m as u64) as f64 };
        let alone = Self::top_group(top_block, g).round() == 1.0;
        let free = if self.q1 && alone { self.active(1.0, p) } else { 0.0 };
        if Self::coefs(g, self.lq) == 2 {
            w.div2 = groups - free;
        } else {
            w.div1 = groups - free;
        }
        if self.rb > 1 {
            w.passes(Self::coefs(g, self.lr), groups, n_groups);
        }
        // The mean of min(p, j xl) over j = 1..=m, inside the mean active frame.
        let short = (p / xl).min(m);
        let adds = (xl * short * (short + 1) / 2 + (m - short) * p) as f64 / mf;
        w.mul1 += 2.0 * wl as f64 * t * adds.min(steps / t);
        if self.signed {
            w.mul1 += steps;
        }
        w
    }

    // The tree as bin_split_tree_at builds it, level by level: a node of n terms is a leaf at
    // leaf_terms terms or under leaf_limbs estimated limbs and otherwise splits into n / 2 and
    // n - n / 2 terms, so a level holds at most two sizes. Each leaf's and merge's parts are
    // divided by what its level's threads gain; then finalize's.
    fn bin_split_parts(&self, env: &SeriesEnv) -> Parts {
        let mut w = Parts::default();
        let is_leaf = |n: u64| n <= self.leaf.1.max(1) || self.est(n) < self.leaf.0;
        // The pool matters only to a tree whose root is wide enough to split.
        let wide = self.par && self.est(self.t) >= env.par_limbs;
        let threads = if wide { rayon::current_num_threads() as f64 } else { 1.0 };
        let off_pool = wide && rayon::current_thread_index().is_none();
        // A tree that is one leaf needs no levels.
        let mut level = [(self.t, 1.0f64), (0, 0.0)];
        if is_leaf(self.t) {
            self.leaf_parts(self.t, 1.0, env, &mut w);
            level[0].1 = 0.0;
        }
        let (mut on, mut par_open, mut depth) = (1.0f64, threads > 1.0, 0);
        while level[0].1 + level[1].1 > 0.0 {
            // The larger size first: it holds the level's rightmost node, whose merge has no R
            // product (its right child ends the sum).
            if level[1].1 > 0.0 && level[1].0 > level[0].0 {
                level.swap(0, 1);
            }
            let spread = 1.0 + env.par_gain * (on - 1.0);
            let (mut next, mut last, mut widest) = ([(0u64, 0.0f64); 2], 1.0f64, 0);
            for &(n, c) in level.iter().filter(|l| l.1 > 0.0) {
                if is_leaf(n) {
                    self.leaf_parts(n, c / spread, env, &mut w);
                    continue;
                }
                let (l, r) = (n / 2, n - n / 2);
                self.merge_parts((l, r), (c / spread, (c - last).max(0.0) / spread), env, &mut w);
                (last, widest) = (0.0, widest.max(n));
                for child in [l, r] {
                    let slot = next.iter_mut().find(|s| s.1 == 0.0 || s.0 == child);
                    let slot = slot.expect("a level holds two sizes");
                    *slot = (child, slot.1 + c);
                }
            }
            // This level's splits run their halves in parallel while every level above did:
            // from `bin_split.dyn_par` limbs, and off the pool the root also needs
            // `bin_split.dyn_par_work` of limbs times terms and pays one wake.
            if widest != 0 && par_open {
                let limbs = self.est(widest);
                let worth = depth > 0 || !off_pool || limbs.saturating_mul(widest) >= env.par_work;
                if limbs >= env.par_limbs && worth {
                    if depth == 0 && off_pool {
                        w.wake = 1.0;
                    }
                    on = (2.0 * on).min(threads);
                } else {
                    par_open = false;
                }
            }
            (level, depth) = (next, depth + 1);
        }
        // Finalize: the numerator from P's top limbs (a copy and a shift), and one division of
        // out's width by Q's top limbs, a div_prim pass when Q is one limb.
        let (f, t) = ((self.len + 2) as f64, self.t as f64);
        let (pt, qt) = (self.part(Part::P, t).min(f), self.part(Part::Q, t).min(f));
        w.light += qt + self.len as f64;
        w.add += pt;
        w.call += 3.0;
        if qt <= 1.0 {
            w.div1 += (self.len + 1) as f64;
        } else {
            w.division += env.div_ns(self.len + 1, qt as usize);
        }
        w.alloc += 1.0;
        w.one = 1.0;
        w
    }

    // c leaves of n terms each. Without x a step packs the terms whose coefficients fit one
    // limb together: P by u (and its shift), Q by u, and by family BBP's Q v added to P, or
    // Hyper's R w added and R by v; each part grows to its length over the steps. An x-leaf
    // takes a term a step: P by q(n), t by x (a product), t p(n) added to P, t by r(n) in
    // two-limb passes; then Q by each q(n) in two-limb passes.
    #[inline(always)]
    fn leaf_parts(&self, n: u64, c: f64, env: &SeriesEnv, w: &mut Parts) {
        let (nf, hyper) = (n as f64, self.family == Family::Hyper);
        let (p, q, r) = (self.part(Part::P, nf), self.part(Part::Q, nf), self.part(Part::R, nf));
        let words = (self.shift.saturating_mul(n) / 64 + 1) as f64;
        w.term += c * nf;
        // A step's shift of P: fused into the multiplier when its low bits fit beside u, and
        // by whole limbs (a move) from 64 bits.
        let shifts = |sh: u64, u_bits: u64| {
            let fused = sh % 64 == 0 || 64u64.saturating_sub(u_bits) >= sh % 64;
            (fused, sh - if fused { sh % 64 } else { 0 } >= 64)
        };
        if self.xl == 0 {
            let wide = 1 + usize::from(self.qb.max(self.pb).max(self.rb) > 64);
            let pack = if self.shift >= 64 || wide == 2 {
                1
            } else {
                let by_q = 64 / self.qb.max(self.rb).max(1);
                let by_p = 1 + 64u64.saturating_sub(self.pb) / (self.qb + self.shift).max(1);
                by_q.min(by_p).clamp(1, n)
            };
            let steps = n.div_ceil(pack) as f64;
            // A part that grows to x over the steps is x (steps + 1) / 2 limbs in all.
            let (cs, grown) = (c * steps, c * (steps + 1.0) / 2.0);
            let (fused, whole) = shifts(self.shift.saturating_mul(pack), pack * self.qb);
            w.passes(wide, grown * p, cs);
            let q_passes = if self.family == Family::BBP { 2.0 } else { 1.0 };
            w.passes(wide, grown * q * q_passes, cs * q_passes);
            if hyper {
                w.passes(wide, grown * r * 2.0, 2.0 * cs);
            }
            if !fused {
                w.add += grown * p;
                w.call += cs;
            }
            if whole {
                w.light += grown * p;
                w.call += cs;
            }
            let bufs = nf + words + nf * if hyper { 2.0 } else { 1.0 };
            w.light += c * bufs * wide as f64;
            w.alloc += c * if hyper { 3.0 } else { 2.0 };
        } else {
            // A part that grows to x over the terms is x (n - 1) / 2 limbs in all before each
            // term's step, and x (n + 1) / 2 after it.
            let (cn, before, after) = (c * nf, c * (nf - 1.0) / 2.0, c * (nf + 1.0) / 2.0);
            let (fused, whole) = shifts(self.shift, self.qb);
            w.passes(1, before * p + after * r, 2.0 * cn);
            w.passes(2, after * r + before * q, 2.0 * cn);
            if !fused {
                w.add += before * p;
                w.call += cn;
            }
            if whole {
                w.light += before * p;
                w.call += cn;
            }
            // t, of about r (i - 1) / n limbs before term i, times x: a pass over t for an x of
            // one or two limbs; schoolbook's time at t's mean length where the leaf's largest
            // such product is schoolbook's (its time is linear in the longer side); else sampled.
            let xf = self.xl as f64;
            if self.xl <= 2 {
                w.passes(self.xl, (before * r).max(cn * xf), cn);
            } else if school_at(env.kara, r.max(xf) as usize, r.min(xf) as usize) {
                let ([a, shift], mean) = (env.mul.school, r * (nf - 1.0) / (2.0 * nf) + 0.5);
                w.product += cn * (a * xf * mean + shift);
            } else {
                sum_sampled(n, |i, k| env.mul(r * (i - 1.0) / nf, xf, c * k, w));
            }
            w.light += c * ((nf + 1.0) * (self.xl as f64 + 1.0) + words + nf + r);
            w.alloc += c * 4.0;
        }
    }

    // c merges of children of l and r terms (c_r of them with an R product): P = lp rq 2^sh + u
    // and Q = lq rq into new zeroed buffers, with u = lq rp for BBP, rp for Engel and rp lr for
    // Hyper, which also forms R = lr rr; P's product is shifted when the shift has bits, and u
    // added.
    #[inline(always)]
    fn merge_parts(&self, (l, r): (u64, u64), (c, c_r): (f64, f64), env: &SeriesEnv, w: &mut Parts) {
        use Part::{P, Q, R};
        let (lf, rf) = (l as f64, r as f64);
        let (lp, lq, rp, rq) = (self.part(P, lf), self.part(Q, lf), self.part(P, rf), self.part(Q, rf));
        env.mul(lp, rq, c, w);
        env.mul(lq, rq, c, w);
        w.light += c * (self.part(P, lf + rf) + 2.0 + lq + rq);
        w.alloc += 2.0 * c;
        let u = match self.family {
            Family::BBP => {
                env.mul(lq, rp, c, w);
                w.light += c * (lq + rp);
                w.alloc += c;
                lq + rp
            }
            Family::Engel => rp,
            Family::Hyper => {
                let (lr, rr) = (self.part(R, lf), self.part(R, rf));
                env.mul(rp, lr, c, w);
                env.mul(lr, rr, c_r, w);
                w.light += c * (rp + lr) + c_r * (lr + rr);
                w.alloc += c + c_r;
                rp + lr
            }
        };
        if self.shift.wrapping_mul(r) % 64 != 0 {
            w.add += c * (lp + rq);
            w.call += c;
        }
        w.add += c * u;
        w.call += c;
    }
}

impl<S: WideBBPSeries> SeriesNode for DynNodeBBP<S> {
    fn eligible(_: &Shift, _: u64, _: u64, _: usize, _: usize) -> Eligible {
        Eligible {
            horner: true,
            rect: false,
        }
    }

    fn small_class(_: &Shift, a: u64, b: u64) -> Option<usize> {
        small_class(0, bit_len128(S::q((a + (b - a) / 2) as u128)), 0)
    }

    fn shape(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize) -> Shape {
        let coefs = (S::p, S::q, |_| 0);
        let k = (ctx.0, 0, 0, S::SIGNED);
        Shape::new::<Self>(Family::BBP, (a, b), (len, zeros), k, coefs, 0)
    }

    fn horner(ctx: &Shift, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let n = horner_room(out.len(), narrow, usize::MAX);
        let mut scratch = ScratchGuard::acquire();
        let [acc, t] = scratch.get_splits([n, n]);
        let kernel = |bit: Bit, budget, acc: &mut [u64]| horner_bbp::<S>(a, b, bit, budget, acc, t);
        horner_run((ctx.0, a, zeros), out, acc, S::SIGNED, kernel)
    }
}

impl<const N: usize, S: WideBBPSeries> SeriesNode for StaticNodeBBP<N, S> {
    fn eligible(_: &Shift, a: u64, b: u64, len: usize, _: usize) -> Eligible {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let horner = horner_acc_len(len, narrow) <= N;
        Eligible {
            horner,
            rect: false,
        }
    }

    fn shape(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize) -> Shape {
        let coefs = (S::p, S::q, |_| 0);
        let k = (ctx.0, 0, 0, S::SIGNED);
        Shape::new::<Self>(Family::BBP, (a, b), (len, zeros), k, coefs, N)
    }

    fn horner(ctx: &Shift, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let n = horner_room(out.len(), narrow, N);
        let (mut acc, mut t) = ([0; N], [0; N]);
        let kernel = |bit: Bit, budget, acc: &mut [u64]| {
            horner_bbp::<S>(a, b, bit, budget, acc, &mut t[..n])
        };
        horner_run((ctx.0, a, zeros), out, &mut acc[..n], S::SIGNED, kernel)
    }
}

impl<S: WideEngelSeries> SeriesNode for DynNodeEngel<S> {
    fn eligible(_: &Shift, _: u64, _: u64, _: usize, _: usize) -> Eligible {
        Eligible {
            horner: true,
            rect: false,
        }
    }

    fn small_class(_: &Shift, a: u64, b: u64) -> Option<usize> {
        small_class(1, bit_len128(S::q((a + (b - a) / 2) as u128)), 0)
    }

    fn shape(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize) -> Shape {
        let coefs = (S::p, S::q, |_| 0);
        let k = (ctx.0, 0, 0, S::SIGNED);
        Shape::new::<Self>(Family::Engel, (a, b), (len, zeros), k, coefs, 0)
    }

    fn horner(ctx: &Shift, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let mut scratch = ScratchGuard::acquire();
        let acc = scratch.get(horner_room(out.len(), narrow, usize::MAX));
        let kernel = |bit: Bit, budget, acc: &mut [u64]| {
            horner_hyper::<EngelHyper<S>>(a, b, bit, budget, None, acc)
        };
        horner_run((ctx.0, a, zeros), out, acc, S::SIGNED, kernel)
    }
}

impl<const N: usize, S: WideEngelSeries> SeriesNode for StaticNodeEngel<N, S> {
    fn eligible(_: &Shift, a: u64, b: u64, len: usize, _: usize) -> Eligible {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let horner = horner_acc_len(len, narrow) <= N;
        Eligible {
            horner,
            rect: false,
        }
    }

    fn shape(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize) -> Shape {
        let coefs = (S::p, S::q, |_| 0);
        let k = (ctx.0, 0, 0, S::SIGNED);
        Shape::new::<Self>(Family::Engel, (a, b), (len, zeros), k, coefs, N)
    }

    fn horner(ctx: &Shift, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let (n, mut acc) = (horner_room(out.len(), narrow, N), [0; N]);
        let kernel = |bit: Bit, budget, acc: &mut [u64]| {
            horner_hyper::<EngelHyper<S>>(a, b, bit, budget, None, acc)
        };
        horner_run((ctx.0, a, zeros), out, &mut acc[..n], S::SIGNED, kernel)
    }
}

impl<S: WideHyperSeries> DynNodeHyper<S> {
    fn shape_of(ctx: &DynHyperCtx, ab: (u64, u64), lz: (usize, usize)) -> Shape {
        let k = (
            ctx.shift,
            ctx.x_bits,
            ctx.x.as_ref().map_or(0, Vec::len),
            S::SIGNED,
        );
        Shape::new::<Self>(Family::Hyper, ab, lz, k, (S::p, S::q, S::r), 0)
    }
}

impl<S: WideHyperSeries> SeriesNode for DynNodeHyper<S> {
    const RECT: bool = true;

    fn eligible(ctx: &DynHyperCtx, a: u64, b: u64, _: usize, _: usize) -> Eligible {
        let mut ok = hyper_eligible::<S>(a, b, ctx.shift, ctx.x_bits);
        ok.rect &= ctx.x.as_ref().is_some_and(|x| x.len() >= 2);
        ok
    }

    fn horner_first(ctx: &DynHyperCtx, a: u64, b: u64, len: usize, zeros: usize) -> bool {
        let xl = ctx.x.as_ref().map_or(0, Vec::len);
        horner_first(xl == 1, len, || Self::eligible(ctx, a, b, len, zeros))
    }

    fn small_class(ctx: &DynHyperCtx, a: u64, b: u64) -> Option<usize> {
        let mid = (a + (b - a) / 2) as u128;
        small_class(small_kind_hyper(ctx.x_bits), bit_len128(S::q(mid)), bit_len128(S::r(mid)))
    }

    fn shape(ctx: &DynHyperCtx, a: u64, b: u64, len: usize, zeros: usize) -> Shape {
        Self::shape_of(ctx, (a, b), (len, zeros))
    }

    fn horner(ctx: &DynHyperCtx, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n) | S::r(n));
        let (n, x) = (horner_room(out.len(), narrow, usize::MAX), ctx.x.as_deref());
        let mut scratch = ScratchGuard::acquire();
        let bufs = scratch.get_splits([n, n, x.map_or(0, <[u64]>::len)]);
        hyper_horner::<S>((ctx.shift, x), a, b, zeros, out, bufs, hi_mul_dyn)
    }

    fn rect(
        ctx: &DynHyperCtx,
        a: u64,
        b: u64,
        zeros: usize,
        out: &mut [u64],
        m: Option<usize>,
    ) -> bool {
        let m = m.unwrap_or_else(|| Self::shape_of(ctx, (a, b), (out.len(), zeros)).rect_m());
        rect_dyn_m::<S>(ctx, a, b, zeros, out, m)
    }
}

impl<const N: usize, S: WideHyperSeries> StaticNodeHyper<N, S> {
    fn shape_of(ctx: &StaticHyperCtx<N>, ab: (u64, u64), lz: (usize, usize)) -> Shape {
        let k = (
            ctx.shift,
            ctx.x_bits,
            ctx.x.as_ref().map_or(0, |x| x.1),
            S::SIGNED,
        );
        Shape::new::<Self>(Family::Hyper, ab, lz, k, (S::p, S::q, S::r), N)
    }
}

impl<const N: usize, S: WideHyperSeries> SeriesNode for StaticNodeHyper<N, S> {
    const RECT: bool = true;

    fn eligible(ctx: &StaticHyperCtx<N>, a: u64, b: u64, len: usize, _: usize) -> Eligible {
        let mut ok = hyper_eligible::<S>(a, b, ctx.shift, ctx.x_bits);
        ok.horner &= horner_acc_len(len, ok.rect) <= N;
        ok.rect &= ok.horner && ctx.x.as_ref().is_some_and(|&(_, l)| l >= 2);
        ok
    }

    fn horner_first(ctx: &StaticHyperCtx<N>, a: u64, b: u64, len: usize, zeros: usize) -> bool {
        let xl = ctx.x.as_ref().map_or(0, |x| x.1);
        horner_first(xl == 1, len, || Self::eligible(ctx, a, b, len, zeros))
    }

    fn shape(ctx: &StaticHyperCtx<N>, a: u64, b: u64, len: usize, zeros: usize) -> Shape {
        Self::shape_of(ctx, (a, b), (len, zeros))
    }

    fn horner(ctx: &StaticHyperCtx<N>, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n) | S::r(n));
        let n = horner_room(out.len(), narrow, N);
        let x = ctx.x.as_ref().map(|(x, l)| &x[..*l]);
        let (mut acc, mut tmp, mut xs) = ([0; N], [0; N], [0; N]);
        let bufs = [&mut acc[..n], &mut tmp[..n], &mut xs[..]];
        hyper_horner::<S>((ctx.shift, x), a, b, zeros, out, bufs, hi_mul_static::<N>)
    }

    #[inline(never)]
    fn rect(
        ctx: &StaticHyperCtx<N>,
        a: u64,
        b: u64,
        zeros: usize,
        out: &mut [u64],
        m: Option<usize>,
    ) -> bool {
        let m = m.unwrap_or_else(|| Self::shape_of(ctx, (a, b), (out.len(), zeros)).rect_m());
        rect_static_m::<N, S>(ctx, a, b, zeros, out, m)
    }
}

// The static evaluator shares its explicit block implementation with tuning and tests.
#[doc(hidden)]
#[inline(never)]
pub fn rect_static_m<const N: usize, S: WideHyperSeries>(
    ctx: &StaticHyperCtx<N>,
    a: u64,
    b: u64,
    zeros: usize,
    out: &mut [u64],
    m: usize,
) -> bool {
    assert!(
        m >= 1 && m <= RECT_MAX_M && m as u64 <= b - a,
        "invalid rectangular block length"
    );
    let (x, l) = ctx.x.as_ref().expect("rect needs x != 1");
    let n = horner_acc_len(out.len(), true);
    let (mut acc, mut tmp, mut xs) = ([0; N], [0; N], [0; N]);
    let bufs = [&mut acc[..n], &mut tmp[..n], &mut xs[..]];
    let mut store = [[0; N]; RECT_MAX_M];
    let mut pw = store.each_mut().map(|p| &mut p[..n]);
    let mut zs = [0; RECT_MAX_M];
    let x = (ctx.shift, &x[..*l]);
    let p = (&mut pw[..m], &mut zs[..m]);
    hyper_rect::<S>(x, a, b, zeros, out, bufs, p, hi_mul_static::<N>)
}
