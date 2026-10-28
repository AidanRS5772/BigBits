use std::marker::PhantomData;

use crate::utils::{
    div::{div_dyn, div_prim, rcp_static},
    mul::{hi_mul_dyn, hi_mul_static, mul_add_prim, mul_dyn, mul_prim, mul_prim2, mul_static},
    utils::{
        add_buf, add_mul, add_prim, buf_len, inc_buf, shl_buf, shr_buf, split_sh, sub_mul,
    },
    ScratchGuard, DYN_BBP_LEAF_LIMBS, DYN_BIN_SPLIT_MERGE_PAR_CUTOFF, DYN_BIN_SPLIT_PAR_CUTOFF,
    DYN_ENGEL_LEAF_LIMBS, DYN_HYPER_LEAF_LIMBS, DYN_HYPER_LEAF_TERMS, DYN_SERIES_BIN_SPLIT_BBP,
    DYN_SERIES_BIN_SPLIT_ENGEL, DYN_SERIES_BIN_SPLIT_FIXED, DYN_SERIES_BIN_SPLIT_HYPER,
    DYN_SERIES_HORNER_BBP, DYN_SERIES_HORNER_ENGEL, DYN_SERIES_HORNER_HYPER,
    DYN_SERIES_RECT_HYPER, SERIES_RECT_M_SCALE, STATIC_BBP_LEAF_LIMBS, STATIC_ENGEL_LEAF_LIMBS,
    STATIC_HYPER_LEAF_LIMBS, STATIC_HYPER_LEAF_TERMS, STATIC_SERIES_BIN_SPLIT_BBP,
    STATIC_SERIES_BIN_SPLIT_ENGEL, STATIC_SERIES_BIN_SPLIT_FIXED, STATIC_SERIES_BIN_SPLIT_HYPER,
    STATIC_SERIES_HORNER_BBP, STATIC_SERIES_HORNER_ENGEL, STATIC_SERIES_HORNER_HYPER,
    STATIC_SERIES_RECT_HYPER,
};

fn merge_par(q_len: usize) -> bool {
    q_len >= DYN_BIN_SPLIT_MERGE_PAR_CUTOFF && rayon::current_thread_index().is_some()
}

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

// Pushes carry c, or once buf is full keeps its top 64 buf.len() - 1 bits (the spare bit
// leaves room for a term) and raises the bit exponent e by what it drops.
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

fn shr_bits(buf: &mut [u64], len: usize, ov: u64, k: u64) -> usize {
    if k == 0 {
        if ov == 0 {
            return len;
        }
        buf[len] = ov;
        return len + 1;
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
    let q_len = prod_terms(a, b, S::q, q);
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

fn prod_terms(a: u64, b: u64, f: impl Fn(u128) -> u128, buf: &mut [u64]) -> usize {
    buf[0] = 1;
    let mut len = 1;
    for n in a as u128..b as u128 {
        let c = mul_prim2(&mut buf[..len], f(n));
        push2(buf, &mut len, c);
    }
    len
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

fn bit_len(x: u64) -> u64 {
    (64 - x.leading_zeros()) as u64
}

fn bit_len128(x: u128) -> u64 {
    (128 - x.leading_zeros()) as u64
}

fn bits(x: &[u64]) -> u64 {
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

pub(crate) const fn static_split_stack(n: usize) -> usize {
    1024 * n + (1 << 20)
}

// Static nodes keep each part as m 2^e, m in N limbs and e in bits. A part stays exact
// while it fits N limbs, and from then on keeps its top 64 N bits (P a bit or two fewer,
// for carries), so it loses no more than rounding.

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

// A merge's P = T + U exponent, from its terms' exponents and top bits (0 for a zero
// term) before either product is computed. P keeps every bit while it fits N limbs, and
// otherwise its top 64 N bits less up to two: a carry, and a product's top bit being
// known only within one.
fn p_exponent<const N: usize>(ta: u64, ea: u64, tu: u64, eu: u64, same: bool) -> u64 {
    let low = |top: u64, e: u64| if top == 0 { u64::MAX } else { e };
    let low = low(ta, ea).min(low(tu, eu));
    if low == u64::MAX {
        return 0;
    }
    let top = ta.max(tu) + same as u64;
    let cap = 64 * N as u64;
    if top - low <= cap {
        low
    } else {
        top - cap
    }
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
    debug_assert!(bits >= 128, "finalize needs w >= 2 whole limbs");
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
    let par = merge_par(lq.len() + rq.len());
    let (_, q) = join2(par, || mul_dyn(lp, rq, &mut p[sl..sl + pl]), || product(lq, rq));
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
    // A tree splits [a, b) while it has more than LEAF_TERMS terms and an estimated
    // LEAF_LIMBS limbs or more.
    const LEAF_LIMBS: u64;
    const LEAF_TERMS: u64 = 1;
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

// Where a tree stops splitting, and from what size a Node::PARALLEL tree builds a
// split's halves in parallel (on a pool of two or more threads: on one, rayon::join
// would only move the work to the pool's thread).
#[derive(Debug, Clone, Copy)]
pub struct SplitCutoffs {
    pub leaf_limbs: u64,
    pub leaf_terms: u64,
    pub par_limbs: u64,
}

impl SplitCutoffs {
    pub const fn of<N: Node>() -> Self {
        SplitCutoffs {
            leaf_limbs: N::LEAF_LIMBS,
            leaf_terms: N::LEAF_TERMS,
            par_limbs: DYN_BIN_SPLIT_PAR_CUTOFF,
        }
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
        N::PARALLEL && limbs >= cut.par_limbs && rayon::current_num_threads() > 1,
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

// Every narrow series is a wide one, and Narrow<S> views a wide series whose values fit
// u64 as a narrow one.
macro_rules! narrow_wide {
    ($narrow:ident, $wide:ident, $($f:ident),+) => {
        impl<S: $narrow> $wide for S {
            const SIGNED: bool = <S as $narrow>::SIGNED;
            $(fn $f(n: u128) -> u128 {
                <S as $narrow>::$f(n as u64) as u128
            })+
            fn neg(n: u128) -> bool {
                <S as $narrow>::neg(n as u64)
            }
        }

        impl<S: $wide> $narrow for Narrow<S> {
            const SIGNED: bool = <S as $wide>::SIGNED;
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
pub trait BBPSeries {
    const SIGNED: bool = false;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
    fn neg(_: u64) -> bool {
        false
    }
}

// A BBPSeries whose p and q may need 128 bits; every BBPSeries is one. A leaf whose values
// all fit u64 runs the narrow kernel. A type implements one of the two traits, and
// calling p or q on a concrete narrow series with both in scope needs <T as BBPSeries>::q.
pub trait WideBBPSeries {
    const SIGNED: bool = false;
    fn p(n: u128) -> u128;
    fn q(n: u128) -> u128;
    fn neg(_: u128) -> bool {
        false
    }
}

narrow_wide!(BBPSeries, WideBBPSeries, p, q);

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
    const LEAF_LIMBS: u64 = DYN_BBP_LEAF_LIMBS;
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
    const LEAF_LIMBS: u64 = STATIC_BBP_LEAF_LIMBS;

    fn range(&self) -> (u64, u64) {
        (self.a, self.b)
    }

    fn limbs(ctx: &Shift, a: u64, b: u64) -> u64 {
        est_limbs(a, b, S::q, ctx.0).min(N as u64)
    }

    // A leaf's exact P, at most prod q 2^(shift (t - 1)) t max p, must fit N limbs.
    fn leaf(ctx: &Shift, a: u64, b: u64, last: bool) -> Self {
        let (t, cap) = (b - a, 64 * N as u64);
        let (mut bits, mut p_bits) = (ctx.0 * (t - 1) + bit_len(t), 0);
        let fits = (a..b).all(|n| {
            bits += bit_len128(S::q(n as u128));
            p_bits = p_bits.max(bit_len128(S::p(n as u128)));
            bits + p_bits <= cap
        });
        if t > 1 && !fits {
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
pub trait EngelSeries {
    const SIGNED: bool = false;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
    fn neg(_: u64) -> bool {
        false
    }
}

// An EngelSeries whose p and q may need 128 bits, as WideBBPSeries is for BBP.
pub trait WideEngelSeries {
    const SIGNED: bool = false;
    fn p(n: u128) -> u128;
    fn q(n: u128) -> u128;
    fn neg(_: u128) -> bool {
        false
    }
}

narrow_wide!(EngelSeries, WideEngelSeries, p, q);

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
    const LEAF_LIMBS: u64 = DYN_ENGEL_LEAF_LIMBS;
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
    const LEAF_LIMBS: u64 = STATIC_ENGEL_LEAF_LIMBS;

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
pub trait HyperSeries {
    const SIGNED: bool = false;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
    fn r(n: u64) -> u64;
    fn neg(_: u64) -> bool {
        false
    }
}

// A HyperSeries whose p, q and r may need 128 bits, as WideBBPSeries is for BBP.
pub trait WideHyperSeries {
    const SIGNED: bool = false;
    fn p(n: u128) -> u128;
    fn q(n: u128) -> u128;
    fn r(n: u128) -> u128;
    fn neg(_: u128) -> bool {
        false
    }
}

narrow_wide!(HyperSeries, WideHyperSeries, p, q, r);

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
    const LEAF_LIMBS: u64 = DYN_HYPER_LEAF_LIMBS;
    const LEAF_TERMS: u64 = DYN_HYPER_LEAF_TERMS;
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
        let (u, r) = (product(&right.p, &lr), right.r.as_deref().map(|rr| product(&lr, rr)));
        let (l, rt) = ((&left.p[..], &left.q[..], left.neg), (&u[..], &right.q[..], right.neg));
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
    const LEAF_LIMBS: u64 = STATIC_HYPER_LEAF_LIMBS;
    const LEAF_TERMS: u64 = STATIC_HYPER_LEAF_TERMS;

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

// series_eval sums [a, b) by reverse Horner, rectangular splitting (Hyper with x of two or
// more limbs) or binary splitting, whichever its cost model picks, and writes S as
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

// The node types series_eval sums with. Only series_eval, series_eval_at and the tests call
// these.
pub trait SeriesNode: Node + Send {
    // Whether the kind has a rect kernel: only Hyper's, whose steps multiply by x.
    #[doc(hidden)]
    const RECT: bool = false;
    // Horner and rect take ranges whose values fit u64 (wide ones are left to bin_split for
    // now), Hyper's with r(n) <= q(n) and k >= bits(x), and static ones the room they need.
    #[doc(hidden)]
    fn eligible(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize) -> Eligible;
    #[doc(hidden)]
    fn dispatch(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize) -> SeriesAlg;
    #[doc(hidden)]
    fn horner(ctx: &Self::Ctx, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool;
    #[doc(hidden)]
    fn rect(_: &Self::Ctx, _: u64, _: u64, _: usize, _: &mut [u64]) -> bool {
        unreachable!("only a RECT kind has a rect kernel")
    }
}

// out <- S 2^(64 (out.len() + zeros)), skipping S's top zeros limbs, which the caller knows
// are zero, as Node::finalize does. Returns whether S is negative. Horner and rect agree
// with the exact sum within one unit of out's last limb, not bit for bit with bin_split.
pub fn series_eval<N>(ctx: &N::Ctx, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool
where
    N: SeriesNode,
    N::Ctx: Sync,
{
    let alg = N::dispatch(ctx, a, b, out.len(), zeros);
    series_eval_at::<N>(ctx, a, b, zeros, out, alg)
}

// series_eval with the algorithm forced, for tests and tuning.
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
            N::rect(ctx, a, b, zeros, out)
        }
        SeriesAlg::BinSplit => bin_split::<N>(ctx, a, b).finalize(ctx, zeros, out),
    }
}

// hi_mul_dyn or hi_mul_static.
type HiMul = fn(&[u64], &[u64], &mut [u64]) -> u64;
// Term n's bit in Horner's frame.
type Bit<'a> = &'a dyn Fn(u64) -> i64;

// Horner's guard limb below out: each step truncates by a few units at most and scales the
// error below it by at most 1.
const HORNER_GUARD: usize = 1;
// At most this many powers of x on a static rect's stack, 8 N bytes each.
const RECT_MAX_M: usize = 16;

// Horner's accumulator: out, its guard limb, and 2 limbs for a value before its division
// by q(n).
const fn acc_len(len: usize) -> usize {
    len + HORNER_GUARD + 2
}

// Runs a Horner or rect kernel in the frame F = S 2^(64 w), w = len + zeros + HORNER_GUARD.
// The per-term 2^-k is never applied: term n's numerator lands at bit 64 w - k (n - a + 1),
// so acc is F throughout and needs no final shift. Writes acc above its guard limb to out;
// returns S < 0.
fn horner_run(
    (k, a, zeros): (u64, u64, usize),
    out: &mut [u64],
    acc: &mut [u64],
    signed: bool,
    kernel: impl FnOnce(Bit, &mut [u64]) -> (usize, bool),
) -> bool {
    let w = 64 * (out.len() + zeros + HORNER_GUARD) as i128;
    let bit = |n: u64| (w - k as i128 * (n - a + 1) as i128).max((i64::MIN / 2) as i128) as i64;
    acc.fill(0);
    let (len, neg) = kernel(&bit, acc);
    let g = HORNER_GUARD;
    out.copy_from_slice(&acc[g..g + out.len()]);
    debug_assert!(
        acc[g + out.len()..].iter().all(|&l| l == 0),
        "series_eval expects S < 2^(-64 zeros)"
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

// acc += floor(p(n) 2^d) / q(n), from n = b - 1 down, each term through t.
fn horner_bbp<S: BBPSeries>(a: u64, b: u64, bit: Bit, acc: &mut [u64], t: &mut [u64]) -> (usize, bool) {
    let (mut len, mut e, mut neg) = (0, 0, false);
    for n in (a..b).rev() {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        // floor(p(n) 2^d) reaches limb d / 64 + 1 at most.
        let (d, mut tl, mut te) = (bit(n), 0, 0);
        let top = ((d.max(0) / 64) as usize + 2).min(t.len());
        t[..top].fill(0);
        add_term(t, &mut tl, &mut te, &mut false, (&[1], d), S::p(n), false, false);
        div_prim(&mut t[..tl], S::q(n));
        let tl = buf_len(&t[..tl]);
        let sub = S::SIGNED && S::neg(n);
        add_pass(acc, &mut len, &mut e, &mut neg, (0, &t[..tl], 1), sub, S::SIGNED);
    }
    debug_assert_eq!(e, 0, "Horner's accumulator overflowed");
    (len, neg)
}

// acc = (acc + p(n) 2^d) / q(n), from n = b - 1 down.
fn horner_engel<S: EngelSeries>(a: u64, b: u64, bit: Bit, acc: &mut [u64]) -> (usize, bool) {
    let (mut len, mut e, mut neg) = (0, 0, false);
    for n in (a..b).rev() {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        let (t, sub) = ((&[1][..], bit(n)), S::SIGNED && S::neg(n));
        add_term(acc, &mut len, &mut e, &mut neg, t, S::p(n), sub, S::SIGNED);
        div_prim(&mut acc[..len], S::q(n));
        len = buf_len(&acc[..len]);
    }
    debug_assert_eq!(e, 0, "Horner's accumulator overflowed");
    (len, neg)
}

// acc = x' (r(n) acc + p(n) 2^d) / q(n), from n = b - 1 down, x' = xn 2^(-64 xn.len()) through
// tmp, or 1 without x. The frame's k is the series' less bits(x), which x' doesn't carry.
fn horner_hyper<S: HyperSeries>(
    a: u64,
    b: u64,
    bit: Bit,
    mut x: Option<(&[u64], &mut [u64], HiMul)>,
    acc: &mut [u64],
) -> (usize, bool) {
    let (mut len, mut e, mut neg) = (0, 0, false);
    for n in (a..b).rev() {
        debug_assert!(S::SIGNED || !S::neg(n), "negative terms need SIGNED");
        let c = mul_prim(&mut acc[..len], S::r(n));
        push(acc, &mut len, c);
        let (t, sub) = ((&[1][..], bit(n)), S::SIGNED && S::neg(n));
        add_term(acc, &mut len, &mut e, &mut neg, t, S::p(n), sub, S::SIGNED);
        if let Some((xn, tmp, hi_mul)) = x.as_mut() {
            len = mul_frac(acc, len, xn, tmp, *hi_mul);
        }
        div_prim(&mut acc[..len], S::q(n));
        len = buf_len(&acc[..len]);
    }
    debug_assert_eq!(e, 0, "Horner's accumulator overflowed");
    (len, neg)
}

// pw[j] <- x'^(j + 1) 2^(64 P), P = pw[0].len(), for x' = xn 2^(-64 xn.len()), through tmp.
fn rect_powers(xn: &[u64], pw: &mut [&mut [u64]], tmp: &mut [u64], hi_mul: HiMul) {
    let (first, rest) = pw.split_first_mut().expect("rect needs a power");
    let (p, c) = (first.len(), xn.len().min(first.len()));
    first[..p - c].fill(0);
    first[p - c..].copy_from_slice(&xn[xn.len() - c..]);
    let mut prev: &[u64] = first;
    for x in rest.iter_mut() {
        x.copy_from_slice(prev);
        mul_frac(x, p, first, tmp, hi_mul);
        prev = x;
    }
}

// Rectangular splitting: blocks of m = pw.len() terms from n = b - 1 down, each one high
// product by x'^s for its s terms and then Horner's scalar steps with p(n) x'^(i + 1) in
// place of p(n), so x' never multiplies in a step.
fn rect_hyper<S: HyperSeries>(
    a: u64,
    b: u64,
    bit: Bit,
    pw: &[&mut [u64]],
    acc: &mut [u64],
    tmp: &mut [u64],
    hi_mul: HiMul,
) -> (usize, bool) {
    let (m, pb) = (pw.len() as u64, 64 * pw[0].len() as i64);
    let (mut len, mut e, mut neg) = (0, 0, false);
    let mut n = b;
    while n > a {
        let lo = n.saturating_sub(m).max(a);
        len = mul_frac(acc, len, &pw[(n - lo - 1) as usize], tmp, hi_mul);
        for k in (lo..n).rev() {
            debug_assert!(S::SIGNED || !S::neg(k), "negative terms need SIGNED");
            let c = mul_prim(&mut acc[..len], S::r(k));
            push(acc, &mut len, c);
            let x = (&pw[(k - lo) as usize][..], bit(k) - pb);
            add_term(acc, &mut len, &mut e, &mut neg, x, S::p(k), S::SIGNED && S::neg(k), S::SIGNED);
            div_prim(&mut acc[..len], S::q(k));
            len = buf_len(&acc[..len]);
        }
        n = lo;
    }
    debug_assert_eq!(e, 0, "rect's accumulator overflowed");
    (len, neg)
}

// Narrow values, r(n) <= q(n) so no step scales the error below it up, and k >= bits(x).
fn hyper_narrow<S: WideHyperSeries>(a: u64, b: u64, k: u64, x_bits: u64) -> bool {
    k >= x_bits
        && (a..b).all(|n| {
            let (n, q, r) = (n as u128, S::q(n as u128), S::r(n as u128));
            (S::p(n) | q | r) >> 64 == 0 && r <= q
        })
}

// acc and tmp hold acc_len(out.len()) limbs, and xs x.len().
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
    let kernel = |bit: Bit, acc: &mut [u64]| horner_hyper::<Narrow<S>>(a, b, bit, x, acc);
    horner_run((k, a, zeros), out, acc, S::SIGNED, kernel)
}

// hyper_horner's buffers, plus m powers of acc_len(out.len()) limbs each in pw.
fn hyper_rect<S: WideHyperSeries>(
    (k, x): (u64, &[u64]),
    a: u64,
    b: u64,
    zeros: usize,
    out: &mut [u64],
    [acc, tmp, xs]: [&mut [u64]; 3],
    pw: &mut [&mut [u64]],
    hi_mul: HiMul,
) -> bool {
    rect_powers(normalize(x, xs), pw, tmp, hi_mul);
    let kernel = |bit: Bit, acc: &mut [u64]| rect_hyper::<Narrow<S>>(a, b, bit, pw, acc, tmp, hi_mul);
    horner_run((k - bits(x), a, zeros), out, acc, S::SIGNED, kernel)
}

// The dynamic rect at an explicit block length m, for tests.
pub(crate) fn rect_dyn_m<S: WideHyperSeries>(
    ctx: &DynHyperCtx,
    a: u64,
    b: u64,
    zeros: usize,
    out: &mut [u64],
    m: usize,
) -> bool {
    let x = ctx.x.as_deref().expect("rect needs x != 1");
    let n = acc_len(out.len());
    let mut scratch = ScratchGuard::acquire();
    let bufs = scratch.get_splits([n, n, x.len()]);
    let mut powers = ScratchGuard::acquire();
    let mut pw = powers.get_split_vec(std::iter::repeat(n).take(m));
    hyper_rect::<S>((ctx.shift, x), a, b, zeros, out, bufs, &mut pw, hi_mul_dyn)
}

// Reverse Horner's limb-steps: term i of t (from 1) spans about w + 1 - k i / 64 limbs of
// the frame, capped at the accumulator's, and none once the terms fall below it.
fn horner_work(len: usize, zeros: usize, k: u64, t: u64) -> f64 {
    let cap = acc_len(len) as f64;
    let top = (len + zeros + HORNER_GUARD + 1) as f64;
    let (t, s) = (t as f64, k as f64 / 64.0);
    if s == 0.0 {
        return cap * t;
    }
    let full = ((top - cap) / s).clamp(0.0, t);
    let end = (top / s).clamp(full, t);
    let at = |i: f64| (top - s * i).min(cap);
    cap * full + (end - full) * (at(full) + at(end)) / 2.0
}

// Rect's block length for t terms, at most max.
fn rect_m(t: u64, max: usize) -> usize {
    let m = (SERIES_RECT_M_SCALE * (t as f64).sqrt()).round() as usize;
    m.clamp(1, max.min(usize::try_from(t).unwrap_or(usize::MAX)))
}

// The eligible algorithm with the least weighted work estimate, for k the frame's shift and
// xl x's limbs (0 for x = 1). w: Horner, rect, bin_split and fixed weights.
fn cheapest<N: Node>(
    ctx: &N::Ctx,
    a: u64,
    b: u64,
    (len, zeros): (usize, usize),
    ok: Eligible,
    (k, xl, max_m): (u64, usize, usize),
    [wh, wr, wb, wf]: [f64; 4],
) -> SeriesAlg {
    let t = b - a;
    let steps = horner_work(len, zeros, k, t);
    let rect = || {
        let (m, p) = (rect_m(t, max_m) as f64, acc_len(len) as f64);
        wr * ((m - 1.0 + (t as f64 / m).ceil()) * p * p) + wh * steps
    };
    let e = N::limbs(ctx, a, b);
    let build = if t <= N::LEAF_TERMS || e < N::LEAF_LIMBS {
        t as f64 * e as f64 / 2.0
    } else {
        e as f64 * e as f64
    };
    let f = (len + 2) as f64;
    let costs = [
        ok.horner.then(|| wh * steps + wr * steps * xl as f64),
        ok.rect.then(rect),
        Some(wb * (build + f * f.min(e as f64)) + wf),
    ];
    let algs = [SeriesAlg::Horner, SeriesAlg::Rect, SeriesAlg::BinSplit];
    let mut best = (SeriesAlg::BinSplit, f64::INFINITY);
    for (alg, cost) in algs.into_iter().zip(costs) {
        if let Some(c) = cost.filter(|&c| c < best.1) {
            best = (alg, c);
        }
    }
    best.0
}

impl<S: WideBBPSeries> SeriesNode for DynNodeBBP<S> {
    fn eligible(_: &Shift, a: u64, b: u64, _: usize, _: usize) -> Eligible {
        let horner = fits_u64(a, b, |n| S::p(n) | S::q(n));
        Eligible { horner, rect: false }
    }

    fn dispatch(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize) -> SeriesAlg {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        let w = [DYN_SERIES_HORNER_BBP, 0.0, DYN_SERIES_BIN_SPLIT_BBP, DYN_SERIES_BIN_SPLIT_FIXED];
        cheapest::<Self>(ctx, a, b, (len, zeros), ok, (ctx.0, 0, 1), w)
    }

    fn horner(ctx: &Shift, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let n = acc_len(out.len());
        let mut scratch = ScratchGuard::acquire();
        let [acc, t] = scratch.get_splits([n, n]);
        let kernel = |bit: Bit, acc: &mut [u64]| horner_bbp::<Narrow<S>>(a, b, bit, acc, t);
        horner_run((ctx.0, a, zeros), out, acc, S::SIGNED, kernel)
    }
}

impl<const N: usize, S: WideBBPSeries> SeriesNode for StaticNodeBBP<N, S> {
    fn eligible(_: &Shift, a: u64, b: u64, len: usize, _: usize) -> Eligible {
        let horner = acc_len(len) <= N && fits_u64(a, b, |n| S::p(n) | S::q(n));
        Eligible { horner, rect: false }
    }

    fn dispatch(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize) -> SeriesAlg {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        let w = [STATIC_SERIES_HORNER_BBP, 0.0, STATIC_SERIES_BIN_SPLIT_BBP, STATIC_SERIES_BIN_SPLIT_FIXED];
        cheapest::<Self>(ctx, a, b, (len, zeros), ok, (ctx.0, 0, 1), w)
    }

    fn horner(ctx: &Shift, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let n = acc_len(out.len());
        let (mut acc, mut t) = ([0; N], [0; N]);
        let kernel = |bit: Bit, acc: &mut [u64]| horner_bbp::<Narrow<S>>(a, b, bit, acc, &mut t[..n]);
        horner_run((ctx.0, a, zeros), out, &mut acc[..n], S::SIGNED, kernel)
    }
}

impl<S: WideEngelSeries> SeriesNode for DynNodeEngel<S> {
    fn eligible(_: &Shift, a: u64, b: u64, _: usize, _: usize) -> Eligible {
        let horner = fits_u64(a, b, |n| S::p(n) | S::q(n));
        Eligible { horner, rect: false }
    }

    fn dispatch(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize) -> SeriesAlg {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        let w = [DYN_SERIES_HORNER_ENGEL, 0.0, DYN_SERIES_BIN_SPLIT_ENGEL, DYN_SERIES_BIN_SPLIT_FIXED];
        cheapest::<Self>(ctx, a, b, (len, zeros), ok, (ctx.0, 0, 1), w)
    }

    fn horner(ctx: &Shift, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let mut scratch = ScratchGuard::acquire();
        let acc = scratch.get(acc_len(out.len()));
        let kernel = |bit: Bit, acc: &mut [u64]| horner_engel::<Narrow<S>>(a, b, bit, acc);
        horner_run((ctx.0, a, zeros), out, acc, S::SIGNED, kernel)
    }
}

impl<const N: usize, S: WideEngelSeries> SeriesNode for StaticNodeEngel<N, S> {
    fn eligible(_: &Shift, a: u64, b: u64, len: usize, _: usize) -> Eligible {
        let horner = acc_len(len) <= N && fits_u64(a, b, |n| S::p(n) | S::q(n));
        Eligible { horner, rect: false }
    }

    fn dispatch(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize) -> SeriesAlg {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        let w = [STATIC_SERIES_HORNER_ENGEL, 0.0, STATIC_SERIES_BIN_SPLIT_ENGEL, STATIC_SERIES_BIN_SPLIT_FIXED];
        cheapest::<Self>(ctx, a, b, (len, zeros), ok, (ctx.0, 0, 1), w)
    }

    fn horner(ctx: &Shift, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let mut acc = [0; N];
        let kernel = |bit: Bit, acc: &mut [u64]| horner_engel::<Narrow<S>>(a, b, bit, acc);
        horner_run((ctx.0, a, zeros), out, &mut acc[..acc_len(out.len())], S::SIGNED, kernel)
    }
}

impl<S: WideHyperSeries> SeriesNode for DynNodeHyper<S> {
    const RECT: bool = true;

    fn eligible(ctx: &DynHyperCtx, a: u64, b: u64, _: usize, _: usize) -> Eligible {
        let horner = hyper_narrow::<S>(a, b, ctx.shift, ctx.x_bits);
        let rect = horner && ctx.x.as_ref().is_some_and(|x| x.len() >= 2);
        Eligible { horner, rect }
    }

    fn dispatch(ctx: &DynHyperCtx, a: u64, b: u64, len: usize, zeros: usize) -> SeriesAlg {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        let (k, xl) = (ctx.shift.saturating_sub(ctx.x_bits), ctx.x.as_ref().map_or(0, Vec::len));
        let w = [
            DYN_SERIES_HORNER_HYPER,
            DYN_SERIES_RECT_HYPER,
            DYN_SERIES_BIN_SPLIT_HYPER,
            DYN_SERIES_BIN_SPLIT_FIXED,
        ];
        cheapest::<Self>(ctx, a, b, (len, zeros), ok, (k, xl, usize::MAX), w)
    }

    fn horner(ctx: &DynHyperCtx, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let (n, x) = (acc_len(out.len()), ctx.x.as_deref());
        let mut scratch = ScratchGuard::acquire();
        let bufs = scratch.get_splits([n, n, x.map_or(0, <[u64]>::len)]);
        hyper_horner::<S>((ctx.shift, x), a, b, zeros, out, bufs, hi_mul_dyn)
    }

    fn rect(ctx: &DynHyperCtx, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        rect_dyn_m::<S>(ctx, a, b, zeros, out, rect_m(b - a, usize::MAX))
    }
}

impl<const N: usize, S: WideHyperSeries> SeriesNode for StaticNodeHyper<N, S> {
    const RECT: bool = true;

    fn eligible(ctx: &StaticHyperCtx<N>, a: u64, b: u64, len: usize, _: usize) -> Eligible {
        let horner = acc_len(len) <= N && hyper_narrow::<S>(a, b, ctx.shift, ctx.x_bits);
        let rect = horner && ctx.x.as_ref().is_some_and(|&(_, l)| l >= 2);
        Eligible { horner, rect }
    }

    fn dispatch(ctx: &StaticHyperCtx<N>, a: u64, b: u64, len: usize, zeros: usize) -> SeriesAlg {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        let (k, xl) = (ctx.shift.saturating_sub(ctx.x_bits), ctx.x.as_ref().map_or(0, |x| x.1));
        let w = [
            STATIC_SERIES_HORNER_HYPER,
            STATIC_SERIES_RECT_HYPER,
            STATIC_SERIES_BIN_SPLIT_HYPER,
            STATIC_SERIES_BIN_SPLIT_FIXED,
        ];
        cheapest::<Self>(ctx, a, b, (len, zeros), ok, (k, xl, RECT_MAX_M), w)
    }

    fn horner(ctx: &StaticHyperCtx<N>, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let n = acc_len(out.len());
        let x = ctx.x.as_ref().map(|(x, l)| &x[..*l]);
        let (mut acc, mut tmp, mut xs) = ([0; N], [0; N], [0; N]);
        let bufs = [&mut acc[..n], &mut tmp[..n], &mut xs[..]];
        hyper_horner::<S>((ctx.shift, x), a, b, zeros, out, bufs, hi_mul_static::<N>)
    }

    #[inline(never)]
    fn rect(ctx: &StaticHyperCtx<N>, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let (x, l) = ctx.x.as_ref().expect("rect needs x != 1");
        let (n, m) = (acc_len(out.len()), rect_m(b - a, RECT_MAX_M));
        let (mut acc, mut tmp, mut xs) = ([0; N], [0; N], [0; N]);
        let bufs = [&mut acc[..n], &mut tmp[..n], &mut xs[..]];
        let mut store = [[0; N]; RECT_MAX_M];
        let mut pw = store.each_mut().map(|p| &mut p[..n]);
        let x = (ctx.shift, &x[..*l]);
        hyper_rect::<S>(x, a, b, zeros, out, bufs, &mut pw[..m], hi_mul_static::<N>)
    }
}
