use std::{fmt, marker::PhantomData};

use crate::utils::{
    div::{div_dyn, div_prim, div_prim2, rcp_static},
    mul::{
        hi_mul_dyn, hi_mul_static, mul_add_prim, mul_dyn, mul_prim, mul_prim2, mul_static, mul_work,
    },
    utils::{add_buf, add_mul, add_prim, buf_len, inc_buf, shl_buf, shr_buf, split_sh, sub_mul},
    CapacityError, ScratchGuard, DYN_BBP_LEAF_LIMBS, DYN_BIN_SPLIT_MERGE_PAR_CUTOFF,
    DYN_BIN_SPLIT_PAR_CUTOFF, DYN_ENGEL_LEAF_LIMBS, DYN_HYPER_LEAF_LIMBS, DYN_HYPER_LEAF_TERMS,
    SERIES_NS_ALLOC, SERIES_NS_CALL, SERIES_NS_DIV, SERIES_NS_FILL, SERIES_NS_PRODUCT,
    SERIES_NS_SCALAR, SERIES_NS_TERM, STATIC_BBP_LEAF_LIMBS, STATIC_ENGEL_LEAF_LIMBS,
    STATIC_HYPER_LEAF_LIMBS, STATIC_HYPER_LEAF_TERMS, STATIC_RCP_KNUTH_NR_CUTOFF,
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
pub trait BBPSeries {
    const SIGNED: bool = false;
    // The first term's index.
    const START: u64;
    // Some b with |R_n| < 2^-b, where R_n sums the terms from n on, given 2^-k <= 2^-s. Any
    // lower bound on -log2 |R_n| is valid, and a tighter one means fewer terms. 0 always is.
    fn tail_bits(n: u64, s: i64) -> u64;
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
    const START: u64;
    fn tail_bits(n: u64, s: i64) -> u64;
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
// Summed from n = START, with the product from r = START. START and tail_bits are as in
// BBPSeries.
pub trait EngelSeries {
    const SIGNED: bool = false;
    const START: u64;
    fn tail_bits(n: u64, s: i64) -> u64;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
    fn neg(_: u64) -> bool {
        false
    }
}

// An EngelSeries whose p and q may need 128 bits, as WideBBPSeries is for BBP.
pub trait WideEngelSeries {
    const SIGNED: bool = false;
    const START: u64;
    fn tail_bits(n: u64, s: i64) -> u64;
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
// Summed from n = START, with x^(n - START + 1) and the product from r = START. START and
// tail_bits are as in BBPSeries, given |x 2^-k| <= 2^-s.
pub trait HyperSeries {
    const SIGNED: bool = false;
    const START: u64;
    fn tail_bits(n: u64, s: i64) -> u64;
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
    const START: u64;
    fn tail_bits(n: u64, s: i64) -> u64;
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
    // Each eligible algorithm's predicted work, rect's at block length m (None: the model's).
    #[doc(hidden)]
    fn costs(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize, m: Option<usize>) -> Costs;
    // The eligible algorithm with the least predicted time.
    #[doc(hidden)]
    fn dispatch(ctx: &Self::Ctx, a: u64, b: u64, len: usize, zeros: usize) -> SeriesAlg {
        Self::costs(ctx, a, b, len, zeros, None).pick()
    }
    #[doc(hidden)]
    fn horner(ctx: &Self::Ctx, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool;
    // Rect at block length m, or the cost model's.
    #[doc(hidden)]
    fn rect(_: &Self::Ctx, _: u64, _: u64, _: usize, _: &mut [u64], _: Option<usize>) -> bool {
        unreachable!("only a RECT kind has a rect kernel")
    }
}

// out <- S 2^(64 (out.len() + zeros)) for the terms [a, b), skipping S's top zeros limbs,
// which the caller knows are zero, as Node::finalize does. Returns whether S is negative.
// Horner and rect agree with the exact sum within one unit of out's last limb, not bit for
// bit with bin_split. series_eval_dyn and series_eval_static find the range and zeros
// themselves; this is for tests and tuning.
#[doc(hidden)]
pub fn series_eval_range<N>(ctx: &N::Ctx, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool
where
    N: SeriesNode,
    N::Ctx: Sync,
{
    debug_assert!(a < b);
    // The costs already checked eligibility and found rect's block length.
    let costs = N::costs(ctx, a, b, out.len(), zeros, None);
    match costs.pick() {
        SeriesAlg::Horner => N::horner(ctx, a, b, zeros, out),
        SeriesAlg::Rect => N::rect(ctx, a, b, zeros, out, costs.rect.map(|r| r.0)),
        SeriesAlg::BinSplit => bin_split::<N>(ctx, a, b).finalize(ctx, zeros, out),
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
// constructor only accepts a series of that kind, and series_eval_dyn and series_eval_static
// sum it to out's precision.
pub struct Bbp<S> {
    shift: u64,
    _s: PhantomData<fn() -> S>,
}

pub struct Engel<S> {
    shift: u64,
    _s: PhantomData<fn() -> S>,
}

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

const fn decay(shift: u64) -> i64 {
    if shift > i64::MAX as u64 {
        i64::MAX
    } else {
        shift as i64
    }
}

// ceil(log2 x) for a trimmed x != 0.
fn ceil_log2(x: &[u64]) -> u64 {
    let (top, low) = x.split_last().unwrap();
    bits(x) - (top.is_power_of_two() && low.iter().all(|&l| l == 0)) as u64
}

impl<S: WideBBPSeries> Bbp<S> {
    pub const fn new(shift: u64) -> Self {
        Bbp {
            shift,
            _s: PhantomData,
        }
    }
}

impl<S: WideEngelSeries> Engel<S> {
    pub const fn new(shift: u64) -> Self {
        Engel {
            shift,
            _s: PhantomData,
        }
    }
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

impl<S> sealed::Sealed for Bbp<S> {}
impl<S> sealed::Sealed for Engel<S> {}
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

impl<S: WideBBPSeries> Series for Bbp<S> {
    type Dyn = DynNodeBBP<S>;
    type Static<const N: usize> = StaticNodeBBP<N, S>;

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

impl<S: WideEngelSeries> Series for Engel<S> {
    type Dyn = DynNodeEngel<S>;
    type Static<const N: usize> = StaticNodeEngel<N, S>;

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
    eval_dyn(ctx, out, None)
}

// series_eval_dyn with the algorithm forced, for tests and tuning.
#[doc(hidden)]
pub fn series_eval_dyn_at<C: Series>(ctx: &C, out: &mut [u64], alg: SeriesAlg) -> (bool, usize) {
    eval_dyn(ctx, out, Some(alg))
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

fn eval_dyn<C: Series>(ctx: &C, out: &mut [u64], alg: Option<SeriesAlg>) -> (bool, usize) {
    let zeros = ctx.zeros();
    if out.is_empty() {
        return (false, zeros);
    }
    let (a, b) = ctx.range(64 * (out.len() + zeros) as u64);
    let node = ctx.dyn_ctx();
    let neg = match alg {
        None => series_eval_range::<C::Dyn>(&node, a, b, zeros, out),
        Some(alg) => series_eval_at::<C::Dyn>(&node, a, b, zeros, out, alg),
    };
    (neg, zeros)
}

fn eval_static<const N: usize, C: Series>(
    ctx: &C,
    out: &mut [u64],
    alg: Option<SeriesAlg>,
) -> Result<(bool, usize), CapacityError> {
    let need = out.len() + 2;
    let node = match ctx.static_ctx::<N>() {
        Ok(node) if need <= N => node,
        r => {
            let x = r.err().map_or(0, |e| e.min_n);
            return Err(CapacityError { min_n: need.max(x) });
        }
    };
    let zeros = ctx.zeros();
    if out.is_empty() {
        return Ok((false, zeros));
    }
    let (a, b) = ctx.range(64 * (out.len() + zeros) as u64);
    let neg = match alg {
        None => series_eval_range::<C::Static<N>>(&node, a, b, zeros, out),
        Some(alg) => series_eval_at::<C::Static<N>>(&node, a, b, zeros, out, alg),
    };
    Ok((neg, zeros))
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

// Narrow Horner's accumulator: out, its guard limb, and 2 limbs before division by q(n).
// Wide Horner reserves one more limb for its two-limb coefficients and carries.
const fn acc_len(len: usize) -> usize {
    len + HORNER_GUARD + 2
}

fn horner_acc_len(len: usize, narrow: bool) -> usize {
    acc_len(len) + usize::from(!narrow)
}

// The accumulator Horner runs in, within cap limbs: the wide one when it fits, whose 128-bit
// group budget (group_budget) takes up to twice the terms per division of the narrow one's
// 64, and at least horner_acc_len, which eligibility checks.
fn horner_room(len: usize, narrow: bool, cap: usize) -> usize {
    horner_acc_len(len, false)
        .min(cap)
        .max(horner_acc_len(len, narrow))
}

// The bits a group's divisor Q = ∏q(n) may take in an acc-limb accumulator for out limbs.
// Before its division the accumulator holds Q times what a single step would divide by
// q(n), so acc_len's spare limb allows 64 bits, as it allowed one q(n), and the wide
// accumulator's two allow 128.
fn group_budget(acc: usize, out: usize) -> u64 {
    64 * (acc - out - HORNER_GUARD - 1).min(2) as u64
}

// Runs a Horner or rect kernel in the frame F = S 2^(64 w), w = len + zeros + HORNER_GUARD.
// The per-term 2^-k is never applied: term n's numerator lands at bit 64 w - k (n - a + 1),
// so acc is F throughout and needs no final shift. The kernel gets the frame, its group
// budget and acc. Writes acc above its guard limb to out; returns S < 0.
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

// The terms n - 1 down to n - m that share one division: while Q = ∏q(j) stays within
// budget bits, the largest p(j) times Q (a bound on every weight, as r(j) <= q(j)) within
// u128, and room(j, that bound's bits) allows. A group always takes one term, so a q(n)
// past the budget sums alone as before. rs[k] gets the r of term n - 1 - k; returns (m, Q).
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

// acc = (acc + p(n) 2^d) / q(n), from n = b - 1 down, one division per group of terms.
fn horner_engel<S: WideEngelSeries>(
    a: u64,
    b: u64,
    bit: Bit,
    budget: u64,
    acc: &mut [u64],
) -> (usize, bool) {
    horner_hyper::<EngelHyper<S>>(a, b, bit, budget, None, acc)
}

// acc = x' (r(n) acc + p(n) 2^d) / q(n), from n = b - 1 down, x' = xn 2^(-64 xn.len()) through
// tmp, or 1 without x. The frame's k is the series' less bits(x), which x' doesn't carry.
// Each group of terms shares one division: without x its steps fold into one product too,
// and with x each step keeps its product by r and by x' and defers its q, U = x' (r U +
// ∏_{i<k} q_i p 2^d), as x' differs between the terms' weights.
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

// One pass checks contraction and whether every coefficient fits the narrow kernel.
fn hyper_eligible<S: WideHyperSeries>(a: u64, b: u64, k: u64, x_bits: u64) -> Eligible {
    if k < x_bits {
        return Eligible {
            horner: false,
            rect: false,
        };
    }
    let mut narrow = true;
    for n in a as u128..b as u128 {
        let (p, q, r) = (S::p(n), S::q(n), S::r(n));
        if r > q {
            return Eligible {
                horner: false,
                rect: false,
            };
        }
        narrow &= (p | q | r) >> 64 == 0;
    }
    Eligible {
        horner: true,
        rect: narrow,
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

// hyper_horner's buffers, plus m powers of acc_len(out.len()) limbs each in pw and their
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
    let n = acc_len(out.len());
    let mut scratch = ScratchGuard::acquire();
    let [acc, tmp, xs, zs] = scratch.get_splits([n, n, x.len(), m]);
    let mut powers = ScratchGuard::acquire();
    let mut pw = powers.get_split_vec(std::iter::repeat(n).take(m));
    let (bufs, p) = ([acc, tmp, xs], (&mut pw[..], zs));
    hyper_rect::<S>((ctx.shift, x), a, b, zeros, out, bufs, p, hi_mul_dyn)
}

// The frame's limbs at terms 1, 1 + stride, 1 + 2 stride, ... of t from the top: term i
// spans about w + 1 - k i / 64 limbs, capped at the accumulator's acc, and none once it
// falls below the frame. A group's division spans its shallowest term's limbs.
fn frame_sum(len: usize, zeros: usize, k: u64, t: u64, acc: usize, stride: u64) -> f64 {
    let (g, top, cap) = (
        stride.max(1),
        (len + zeros + HORNER_GUARD + 1) as u64,
        acc as u64,
    );
    // How many of the sampled terms lie at or before term i.
    let count = |i: u64| if i == 0 { 0 } else { (i.min(t) - 1) / g + 1 };
    if k == 0 {
        return (cap.min(top) * count(t)) as f64;
    }
    // Terms up to i = (top - cap) / s span the whole accumulator, and those below top / s
    // part of it, with s = k / 64 limbs a term.
    let full = count(64 * top.saturating_sub(cap) / k);
    let n = (count((64 * top - 1) / k) - full) as f64;
    let (full, s, g) = (full as f64, k as f64 / 64.0, g as f64);
    let linear = n * (top as f64 - s) - s * g * (2.0 * full + n - 1.0) * n / 2.0;
    full * cap as f64 + linear.max(0.0)
}

// series_eval's cost model. Each algorithm's time is predicted as the counts of its
// kernels' primitive operations (Work), Big-O in the call's parameters, times one measured
// constant per primitive (SERIES_NS_*), shared by every family and both allocation modes.
// Pass, array and product counts are structural, read off the kernels. Products are
// mul_work's schoolbook limb products, so a product's tier comes from mul.rs's own
// crossovers. See docs/series_cutoff_tuning.md.

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

    // The [u64; N] arrays a static Horner zero-fills: acc and t; acc; acc, tmp and xs.
    fn horner_arrays(self) -> f64 {
        match self {
            Family::BBP => 2.0,
            Family::Engel => 1.0,
            Family::Hyper => 3.0,
        }
    }

    fn parts(self) -> f64 {
        match self {
            Family::Hyper => 3.0,
            _ => 2.0,
        }
    }
}

// An algorithm's predicted work, by primitive.
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
    // Heap allocations and scratch acquisitions.
    pub alloc: f64,
    // Limbs of static [u64; N] arrays zero-filled or moved.
    pub fill: f64,
}

impl Work {
    // The predicted time in ns.
    pub fn ns(&self) -> f64 {
        SERIES_NS_DIV * self.div
            + SERIES_NS_SCALAR * self.scalar
            + SERIES_NS_PRODUCT * self.product
            + SERIES_NS_TERM * self.term
            + SERIES_NS_CALL * self.call
            + SERIES_NS_ALLOC * self.alloc
            + SERIES_NS_FILL * self.fill
    }
}

// Each eligible algorithm's predicted work, and rect's block length.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Costs {
    pub horner: Option<Work>,
    pub rect: Option<(usize, Work)>,
    pub bin_split: Work,
}

impl Costs {
    pub fn of(&self, alg: SeriesAlg) -> Option<Work> {
        match alg {
            SeriesAlg::Horner => self.horner,
            SeriesAlg::Rect => self.rect.map(|r| r.1),
            SeriesAlg::BinSplit => Some(self.bin_split),
        }
    }

    // The eligible algorithm with the least predicted time.
    pub fn pick(&self) -> SeriesAlg {
        let mut best = (SeriesAlg::BinSplit, self.bin_split.ns());
        for alg in [SeriesAlg::Rect, SeriesAlg::Horner] {
            if let Some(ns) = self.of(alg).map(|w| w.ns()).filter(|&ns| ns <= best.1) {
                best = (alg, ns);
            }
        }
        best.0
    }
}

// What the cost model sees of a call, all O(1) from the series at its midpoint.
struct Shape {
    family: Family,
    t: u64,
    len: usize,
    zeros: usize,
    // Horner's frame shift per term, x's limbs (0 for x = 1), and the leaves' shift per
    // term (k without x's bits taken out).
    k: u64,
    xl: usize,
    shift: u64,
    signed: bool,
    // q(n)'s and p(n)'s bits at the midpoint.
    qb: u64,
    pb: u64,
    // The whole range's P (the tree's width, Node::limbs before N's cap), Q and R, in
    // limbs: P about Q 2^(shift t) (and R x^t), Q without the shift, R with x^t.
    p: u64,
    q: u64,
    r: u64,
    // N for static nodes, 0 for dynamic ones.
    cap: usize,
    // Node::LEAF_LIMBS and LEAF_TERMS, and whether halves build in parallel.
    leaf: (u64, u64),
    par: bool,
}

// A pass by a scalar of `limbs` limbs, per limb: a two-limb scalar's four partial products
// and longer carries cost about four one-limb passes (mul_prim2 against mul_prim).
fn scalar_units(limbs: usize) -> f64 {
    (limbs * limbs) as f64
}

// The parts of rect's work that don't depend on its block length.
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
    if bits > 64 {
        2
    } else {
        1
    }
}

impl Shape {
    fn new<N: Node>(
        family: Family,
        (a, b): (u64, u64),
        (len, zeros): (usize, usize),
        (k, xl, shift, signed): (u64, usize, u64, bool),
        (p, q): (u128, u128),
        (pl, ql, rl): (u64, u64, u64),
        cap: usize,
    ) -> Shape {
        Shape {
            family,
            t: b - a,
            len,
            zeros,
            k,
            xl,
            shift,
            signed,
            qb: bit_len128(q),
            pb: bit_len128(p),
            p: pl,
            q: ql,
            r: rl,
            cap,
            leaf: (N::LEAF_LIMBS, N::LEAF_TERMS),
            par: N::PARALLEL,
        }
    }

    fn costs(&self, ok: Eligible, m: Option<usize>) -> Costs {
        Costs {
            horner: ok.horner.then(|| self.horner()),
            rect: ok.rect.then(|| match m {
                Some(m) => (m, self.rect(&self.rect_fixed(), m)),
                None => self.rect_best(),
            }),
            bin_split: self.bin_split(),
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

    // The frame's limbs at every stride-th term from the top, in an acc-limb accumulator.
    fn frame(&self, acc: usize, stride: u64) -> f64 {
        frame_sum(self.len, self.zeros, self.k, self.t, acc, stride)
    }

    // The frame's limbs at its first term.
    fn frame_top(&self, acc: usize) -> f64 {
        frame_sum(self.len, self.zeros, self.k, 1, acc, 1)
    }

    // Per group a division, BBP's temp clear and add, and Hyper's product by R; per term
    // its weight's add (two passes) and bookkeeping; with x, per term also a product by r
    // and a high product by x, copied back. A signed term whose sign differs from the sum's
    // usually flips it (each term outweighs the sum so far), negating the active limbs.
    fn horner(&self) -> Work {
        let cap = if self.stat() { self.cap } else { usize::MAX };
        // Eligibility already requires a narrow series where only narrow Horner fits.
        let acc = horner_room(self.len, true, cap);
        let (stat, g, t) = (self.stat(), self.group(acc), self.t as f64);
        let (steps, groups) = (self.frame(acc, 1), self.frame(acc, g));
        let n_groups = self.t.div_ceil(g) as f64;
        let mut work = Work {
            div: groups,
            term: t,
            call: n_groups + 2.0 * t,
            alloc: if stat { 0.0 } else { 1.0 },
            fill: self.fill(self.family.horner_arrays()),
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
                work.product = steps * mul_work(acc, xl.min(acc), stat) / acc as f64;
                work.call += 3.0 * t;
            }
        }
        if self.signed {
            work.scalar += steps;
        }
        work
    }

    // What rect's work shares across block lengths: its accumulator, group size, R and
    // weight limbs, the frame's limbs and its top's, and a power's product by x.
    fn rect_fixed(&self) -> RectFixed {
        let p = acc_len(self.len);
        let g = self.group(p);
        RectFixed {
            p,
            g,
            limbs: self.group_limbs(g),
            steps: self.frame(p, 1),
            top: self.frame_top(p),
            power: mul_work(p, self.xl.max(1), self.stat()),
        }
    }

    // Powers x'^2..x'^m, a high product by x'^m per block of m terms before its terms
    // (copied back), and grouped scalar steps whose terms add their weight times the power's
    // significant limbs, and flip signs as Horner's do.
    fn rect(&self, f: &RectFixed, m: usize) -> Work {
        let (p, xl, (rl, wl)) = (f.p, self.xl.max(1), f.limbs);
        let (mf, t) = (m as f64, self.t as f64);
        // Groups don't cross a block: ⌈m / g⌉ divisions a block.
        let per_block = (m as u64).div_ceil(f.g) as f64;
        let tops = self.frame(p, m as u64);
        let n_blocks = self.t.div_ceil(m as u64) as f64;
        let groups = tops * per_block;
        // A block's product comes before its terms, at the deeper block's top limbs.
        let blocks = tops - f.top;
        // The mean of min(p, j xl) over j = 1..=m.
        let short = (p / xl).min(m);
        let adds = (xl * short * (short + 1) / 2 + (m - short) * p) as f64 / mf;
        let block = mul_work(p, (m * xl).min(p), self.stat()) / p as f64;
        let flips = if self.signed { f.steps } else { 0.0 };
        Work {
            div: groups,
            scalar: groups * scalar_units(rl) + t * adds * scalar_units(wl) + blocks + flips,
            product: (mf - 1.0) * f.power + blocks * block,
            term: t,
            call: mf + n_blocks * (2.0 + 2.0 * per_block) + 2.0 * wl as f64 * t,
            alloc: if self.stat() { 0.0 } else { 3.0 },
            fill: self.fill((RECT_MAX_M + 3) as f64),
        }
    }

    // Rect's block length and work at it: the least predicted time at the balance of block
    // products against powers and longer term adds, at 1 (few terms, where powers don't pay),
    // and at the length whose power first fills the accumulator. The model's time is flat
    // near the balance, so finer steps don't pay for themselves on small calls.
    fn rect_best(&self) -> (usize, Work) {
        let max = if self.stat() { RECT_MAX_M } else { usize::MAX };
        let hi = max
            .min(usize::try_from(self.t).unwrap_or(usize::MAX))
            .max(1);
        let (f, xl) = (self.rect_fixed(), self.xl.max(1));
        // Time ~ a / m + b m: full blocks and their divisions against the powers and the
        // growth of the term adds while powers are short.
        let a = SERIES_NS_PRODUCT * f.steps / f.p as f64 * mul_work(f.p, f.p, self.stat())
            + SERIES_NS_DIV * f.steps;
        let b = SERIES_NS_PRODUCT * f.power + SERIES_NS_SCALAR * self.t as f64 * xl as f64 / 2.0;
        let ms = [(a / b).sqrt().round() as usize, 1, f.p.div_ceil(xl)].map(|m| m.clamp(1, hi));
        let mut best = (0, Work::default(), f64::INFINITY);
        for (i, &m) in ms.iter().enumerate() {
            if ms[..i].contains(&m) {
                continue;
            }
            let w = self.rect(&f, m);
            let ns = w.ns();
            if ns < best.2 {
                best = (m, w, ns);
            }
        }
        (best.0, best.1)
    }

    fn rect_m(&self) -> usize {
        self.rect_best().0
    }

    // Leaves of up to LEAF_TERMS terms or below LEAF_LIMBS (with static Node::limbs capped
    // at N, and static leaves split until they fit N), each term a scalar pass per merge
    // product; then merges level by level, and finalize's division. Dynamic trees from
    // DYN_BIN_SPLIT_PAR_CUTOFF limbs build their halves in parallel.
    fn bin_split(&self) -> Work {
        let (stat, fam, t) = (self.stat(), self.family, self.t as f64);
        let n = if stat { self.cap as f64 } else { f64::INFINITY };
        // Levels of splits: a node at level d has t / 2^d terms and E / 2^d limbs.
        let (e, cap) = (self.p.max(1) as u128, self.cap as u128);
        let (leaf_limbs, leaf_terms) = (self.leaf.0 as u128, self.leaf.1.max(1) as u128);
        let mut d = 0;
        while d < 60 {
            let small = e < leaf_limbs << d || stat && cap < leaf_limbs;
            if self.t as u128 <= leaf_terms << d || small && !(stat && e > cap << d) {
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
        let product = |a: f64, b: f64| {
            let (a, b) = (a.ceil().max(1.0) as usize, b.ceil().max(1.0) as usize);
            mul_work(a, b, stat)
        };
        // The threads at level j: 2^j while every level above splits in parallel.
        let threads = if self.par {
            rayon::current_num_threads() as f64
        } else {
            1.0
        };
        let par = (0..d)
            .take_while(|&j| e >= (DYN_BIN_SPLIT_PAR_CUTOFF as u128) << j)
            .count() as i32;
        let spread = |j: i32| ((1u64 << j.min(par)) as f64).min(threads).max(1.0);
        let pairs = fam.products();
        let (mut merges, mut passes) = (0.0, 0.0);
        for j in 0..d as i32 {
            let (mut level, mut out) = (0.0, 0.0);
            for &(a, b) in pairs {
                let (a, b) = (part(a, j + 1), part(b, j + 1));
                level += product(a, b);
                out += a + b;
            }
            let nodes = (1u64 << j) as f64;
            merges += nodes * level / spread(j);
            // Dynamic products zero their new buffers; the merged P is shifted into place
            // and summed (two passes) in either mode.
            let out = if stat { 0.0 } else { out };
            passes += nodes * (out + 2.0 * part(Part::P, j)) / spread(j);
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
        let mut leaf_arrays = 2.0 * fam.parts();
        if self.xl != 0 {
            // An x-leaf's R is its running term t: each term multiplies it by x, adds w t to
            // P and multiplies it by r(n) (two-limb passes, as mul_prim2 and add_term_wide),
            // and keeps exponents as Horner's terms do.
            leaf_product = t * product(t / leaves * self.xl as f64 / 2.0, self.xl as f64);
            leaf_calls += 1.0;
            leaf_arrays += 2.0;
            leaf_terms = t;
        }
        // Finalize divides f limbs by Q: a quotient limb estimate each, and Knuth's
        // products up to Newton's three (dynamic), or a reciprocal and a high product.
        let f = self.len + 2;
        let ql = (self.q.max(1) as usize).min(f);
        let finalize = if stat {
            ((f * ql) as f64).min(2.0 * mul_work(f, f, true)) + mul_work(f, f, true)
        } else {
            ((f * ql) as f64).min(3.0 * mul_work(f, f, false))
        };
        let (merge_n, leaf_s) = (leaves - 1.0, spread(d));
        let parts = fam.parts();
        Work {
            div: f as f64,
            scalar: t * leaf_term / leaf_s + passes,
            product: merges + leaf_product / leaf_s + finalize,
            term: leaf_terms / leaf_s,
            call: t * leaf_calls / leaf_s + merge_n * (pairs.len() + 1) as f64 + 4.0,
            // Each leaf allocates its parts and each merge its products' outputs; finalize
            // takes scratch for its division.
            alloc: if stat {
                0.0
            } else {
                leaves * parts + merge_n * pairs.len() as f64 + 2.0
            },
            // A leaf fills and moves its parts, a merge fills its P and moves its parts, and
            // finalize takes a reciprocal and a product in [u64; N] arrays.
            fill: self.fill(leaves * leaf_arrays + merge_n * (parts + 1.0) + 4.0),
        }
    }
}

// The Shape of a BBP or Engel range with N's leaves, frame shift k and cap (N, or 0).
fn bbp_shape<S: WideBBPSeries, N: Node>(
    k: u64,
    (a, b): (u64, u64),
    lz: (usize, usize),
    cap: usize,
) -> Shape {
    let mid = (a + (b - a) / 2) as u128;
    let limbs = (est_limbs(a, b, S::q, k), est_limbs(a, b, S::q, 0), 0);
    let coefs = (S::p(mid), S::q(mid));
    let k = (k, 0, k, S::SIGNED);
    Shape::new::<N>(Family::BBP, (a, b), lz, k, coefs, limbs, cap)
}

fn engel_shape<S: WideEngelSeries, N: Node>(
    k: u64,
    (a, b): (u64, u64),
    lz: (usize, usize),
    cap: usize,
) -> Shape {
    let mid = (a + (b - a) / 2) as u128;
    let limbs = (est_limbs(a, b, S::q, k), est_limbs(a, b, S::q, 0), 0);
    let coefs = (S::p(mid), S::q(mid));
    let k = (k, 0, k, S::SIGNED);
    Shape::new::<N>(Family::Engel, (a, b), lz, k, coefs, limbs, cap)
}

// A Hyper range's Shape: the frame shifts k - bits(x) a term, and P (the tree's width) is
// the larger of Q 2^(shift t) and R x^t, as Node::limbs estimates it.
fn hyper_shape<S: WideHyperSeries, N: Node>(
    (shift, x_bits, xl): (u64, u64, usize),
    (a, b): (u64, u64),
    lz: (usize, usize),
    cap: usize,
) -> Shape {
    let mid = (a + (b - a) / 2) as u128;
    let r = est_limbs(a, b, S::r, x_bits);
    let limbs = (
        est_limbs(a, b, S::q, shift).max(r),
        est_limbs(a, b, S::q, 0),
        r,
    );
    let k = (shift.saturating_sub(x_bits), xl, shift, S::SIGNED);
    let coefs = (S::p(mid), S::q(mid));
    Shape::new::<N>(Family::Hyper, (a, b), lz, k, coefs, limbs, cap)
}

impl<S: WideBBPSeries> SeriesNode for DynNodeBBP<S> {
    fn eligible(_: &Shift, _: u64, _: u64, _: usize, _: usize) -> Eligible {
        Eligible {
            horner: true,
            rect: false,
        }
    }

    fn costs(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize, m: Option<usize>) -> Costs {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        bbp_shape::<S, Self>(ctx.0, (a, b), (len, zeros), 0).costs(ok, m)
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

    fn costs(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize, m: Option<usize>) -> Costs {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        bbp_shape::<S, Self>(ctx.0, (a, b), (len, zeros), N).costs(ok, m)
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

    fn costs(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize, m: Option<usize>) -> Costs {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        engel_shape::<S, Self>(ctx.0, (a, b), (len, zeros), 0).costs(ok, m)
    }

    fn horner(ctx: &Shift, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let mut scratch = ScratchGuard::acquire();
        let acc = scratch.get(horner_room(out.len(), narrow, usize::MAX));
        let kernel = |bit: Bit, budget, acc: &mut [u64]| horner_engel::<S>(a, b, bit, budget, acc);
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

    fn costs(ctx: &Shift, a: u64, b: u64, len: usize, zeros: usize, m: Option<usize>) -> Costs {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        engel_shape::<S, Self>(ctx.0, (a, b), (len, zeros), N).costs(ok, m)
    }

    fn horner(ctx: &Shift, a: u64, b: u64, zeros: usize, out: &mut [u64]) -> bool {
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let mut acc = [0; N];
        let kernel = |bit: Bit, budget, acc: &mut [u64]| horner_engel::<S>(a, b, bit, budget, acc);
        horner_run(
            (ctx.0, a, zeros),
            out,
            &mut acc[..horner_room(out.len(), narrow, N)],
            S::SIGNED,
            kernel,
        )
    }
}

impl<S: WideHyperSeries> DynNodeHyper<S> {
    fn shape(ctx: &DynHyperCtx, ab: (u64, u64), lz: (usize, usize)) -> Shape {
        let xl = ctx.x.as_ref().map_or(0, Vec::len);
        hyper_shape::<S, Self>((ctx.shift, ctx.x_bits, xl), ab, lz, 0)
    }
}

impl<S: WideHyperSeries> SeriesNode for DynNodeHyper<S> {
    const RECT: bool = true;

    fn eligible(ctx: &DynHyperCtx, a: u64, b: u64, _: usize, _: usize) -> Eligible {
        let mut ok = hyper_eligible::<S>(a, b, ctx.shift, ctx.x_bits);
        ok.rect &= ctx.x.as_ref().is_some_and(|x| x.len() >= 2);
        ok
    }

    fn costs(
        ctx: &DynHyperCtx,
        a: u64,
        b: u64,
        len: usize,
        zeros: usize,
        m: Option<usize>,
    ) -> Costs {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        Self::shape(ctx, (a, b), (len, zeros)).costs(ok, m)
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
        let m = m.unwrap_or_else(|| Self::shape(ctx, (a, b), (out.len(), zeros)).rect_m());
        rect_dyn_m::<S>(ctx, a, b, zeros, out, m)
    }
}

impl<const N: usize, S: WideHyperSeries> StaticNodeHyper<N, S> {
    fn shape(ctx: &StaticHyperCtx<N>, ab: (u64, u64), lz: (usize, usize)) -> Shape {
        let xl = ctx.x.as_ref().map_or(0, |x| x.1);
        hyper_shape::<S, Self>((ctx.shift, ctx.x_bits, xl), ab, lz, N)
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

    fn costs(
        ctx: &StaticHyperCtx<N>,
        a: u64,
        b: u64,
        len: usize,
        zeros: usize,
        m: Option<usize>,
    ) -> Costs {
        let ok = Self::eligible(ctx, a, b, len, zeros);
        Self::shape(ctx, (a, b), (len, zeros)).costs(ok, m)
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
        let m = m.unwrap_or_else(|| Self::shape(ctx, (a, b), (out.len(), zeros)).rect_m());
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
    let n = acc_len(out.len());
    let (mut acc, mut tmp, mut xs) = ([0; N], [0; N], [0; N]);
    let bufs = [&mut acc[..n], &mut tmp[..n], &mut xs[..]];
    let mut store = [[0; N]; RECT_MAX_M];
    let mut pw = store.each_mut().map(|p| &mut p[..n]);
    let mut zs = [0; RECT_MAX_M];
    let x = (ctx.shift, &x[..*l]);
    let p = (&mut pw[..m], &mut zs[..m]);
    hyper_rect::<S>(x, a, b, zeros, out, bufs, p, hi_mul_static::<N>)
}
