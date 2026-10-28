use std::marker::PhantomData;

use crate::utils::{
    div::{div_dyn, rcp_static},
    mul::{hi_mul_static, mul_add_prim, mul_dyn, mul_prim, mul_prim2, mul_static},
    utils::{
        add_buf, add_mul, add_prim, buf_len, inc_buf, shl_buf, shr_buf, split_sh, sub_mul, trim_lz,
    },
    ScratchGuard, DYN_BBP_LEAF_LIMBS, DYN_BIN_SPLIT_MERGE_PAR_CUTOFF, DYN_BIN_SPLIT_PAR_CUTOFF,
    DYN_ENGEL_LEAF_LIMBS, DYN_HYPER_LEAF_LIMBS, DYN_HYPER_LEAF_TERMS, STATIC_BBP_LEAF_LIMBS,
    STATIC_ENGEL_LEAF_LIMBS, STATIC_HYPER_LEAF_LIMBS, STATIC_HYPER_LEAF_TERMS,
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
        drop_under(buf, len, e, c);
    }
}

#[cold]
#[inline(never)]
fn drop_under(buf: &mut [u64], len: &mut usize, e: &mut u64, c: u64) {
    let k = bit_len(c) + 1;
    *len = shr_bits(buf, *len, c, k);
    *e += k;
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

#[inline(always)]
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
    if d >= 0 && d % 64 == 0 {
        add_pass(p, len, e, pn, ((d / 64) as usize, x, w), neg, signed);
    } else {
        add_term_split(p, len, e, pn, (x, d), w, neg, signed);
    }
}

#[inline(never)]
fn add_term_split(
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
// A WIDE leaf takes u128 values, and otherwise every value fits u64.
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
        if t_len != 0 && WIDE {
            let ts = (&t[..t_len], et);
            add_term_wide(p, &mut p_len, &mut e, &mut pn, ts, w, S::neg(n), S::SIGNED);
        } else if t_len != 0 {
            let (ts, w) = ((&t[..t_len], et as i64 - e as i64), w as u64);
            add_term(p, &mut p_len, &mut e, &mut pn, ts, w, S::neg(n), S::SIGNED);
        }
        if WIDE {
            let c = mul_prim2(&mut t[..t_len], S::r(n));
            push_or_drop2(t, &mut t_len, &mut et, c);
        } else {
            let c = mul_prim(&mut t[..t_len], S::r(n) as u64);
            push_or_drop(t, &mut t_len, &mut et, c);
        }
    }
    let q_len = if WIDE {
        prod_terms_wide(a, b, S::q, q)
    } else {
        prod_terms(a, b, |n| S::q(n as u128) as u64, q)
    };
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

fn prod_terms_wide(a: u64, b: u64, f: impl Fn(u128) -> u128, buf: &mut [u64]) -> usize {
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
    mul_dyn(a, b, &mut v);
    trim_lz(&mut v);
    v
}

fn prod_terms(a: u64, b: u64, f: impl Fn(u64) -> u64, buf: &mut [u64]) -> usize {
    buf[0] = 1;
    let (mut len, mut n) = (1, a);
    while n < b {
        let mut u = f(n);
        n += 1;
        while n < b {
            let w = u as u128 * f(n) as u128;
            if w >> 64 != 0 {
                break;
            }
            u = w as u64;
            n += 1;
        }
        let c = mul_prim(&mut buf[..len], u);
        push(buf, &mut len, c);
    }
    len
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

// Moves u[..len] 2^eu to exponent e, dropping its bits below it, as (limb offset, length).
fn place(u: &mut [u64], len: usize, eu: u64, e: u64) -> (usize, usize) {
    if eu >= e {
        let (off, s) = split_sh(eu - e);
        (off, shl_bits(u, len, s))
    } else {
        (0, shr_bits(u, len, 0, e - eu))
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
        let terms = (b - a) as usize;
        let words = split_sh(ctx.0 * (b - a)).0 + 1;
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let (pl, ql) = leaf_lens(narrow, terms, words);
        let (mut p, mut q) = (vec![0; pl], vec![0; ql]);
        let (p_len, q_len, e, neg) = if narrow {
            bbp_leaf_terms::<Narrow<S>>(a, b, ctx.0, &mut p, &mut q)
        } else {
            bbp_leaf_terms_wide::<S>(a, b, ctx.0, &mut p, &mut q)
        };
        debug_assert_eq!(e, 0);
        p.truncate(p_len);
        q.truncate(q_len);
        DynNodeBBP {
            a,
            b,
            p,
            q,
            neg,
            _s: PhantomData,
        }
    }

    fn merge(ctx: &Shift, l: Self, r: Self) -> Self {
        let (sl, sb) = split_sh(ctx.0 * (r.b - r.a)); // r(b - m)
        let pl = l.p.len() + r.q.len();
        let tl = l.q.len() + r.p.len();

        let mut p = vec![0; (sl + pl + 1).max(tl) + 1];
        let mut scratch = ScratchGuard::acquire();
        let t = scratch.get(tl);
        let par = merge_par(l.q.len() + r.q.len());
        let (_, (t_len, q)) = join2(
            par,
            || mul_dyn(&l.p, &r.q, &mut p[sl..sl + pl]),
            || join2(par, || mul_exact(&l.q, &r.p, t), || product(&l.q, &r.q)),
        );
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);
        let (n, signs) = (p.len(), (l.neg, r.neg));
        let (len, neg) = p_sum(&mut p, n, (0, &t[..t_len]), signs, false, S::SIGNED);
        p.truncate(len);

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
        let shift = (ctx.0 * (self.b - self.a)) as i64 - 64 * zeros as i64;
        ratio_to_fraction_dyn(&self.p, &self.q, shift, out);
        S::SIGNED && self.neg && !self.p.is_empty()
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
    // Whether [a, b)'s exact P, at most prod q 2^(shift (t - 1)) t max p, fits N limbs.
    fn leaf_fits(shift: u64, a: u64, b: u64) -> bool {
        let t = b - a;
        let cap = 64 * N as u64;
        let (mut bits, mut p_bits) = (shift * (t - 1) + bit_len(t), 0);
        for n in a..b {
            let n = n as u128;
            bits += bit_len128(S::q(n));
            p_bits = p_bits.max(bit_len128(S::p(n)));
            if bits + p_bits > cap {
                return false;
            }
        }
        true
    }

    #[inline(never)]
    fn exact_leaf(shift: u64, a: u64, b: u64) -> Self {
        let (mut p, mut q) = ([0; N], [0; N]);
        let (p_len, q_len, e, neg) = if fits_u64(a, b, |n| S::p(n) | S::q(n)) {
            bbp_leaf_terms::<Narrow<S>>(a, b, shift, &mut p, &mut q)
        } else {
            bbp_leaf_terms_wide::<S>(a, b, shift, &mut p, &mut q)
        };
        debug_assert_eq!(e, 0, "leaf_fits keeps a leaf's P exact");

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

    fn leaf(ctx: &Shift, a: u64, b: u64, last: bool) -> Self {
        if b - a > 1 && !Self::leaf_fits(ctx.0, a, b) {
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
        let terms = (b - a) as usize;
        let words = split_sh(ctx.0 * (b - a)).0 + 1;
        let narrow = fits_u64(a, b, |n| S::p(n) | S::q(n));
        let (pl, ql) = leaf_lens(narrow, terms, words);
        let (mut p, mut q) = (vec![0; pl], vec![0; ql]);
        let (p_len, q_len, e, neg) = if narrow {
            engel_leaf_terms::<Narrow<S>>(a, b, ctx.0, &mut p, &mut q)
        } else {
            engel_leaf_terms_wide::<S>(a, b, ctx.0, &mut p, &mut q)
        };
        debug_assert_eq!(e, 0);
        p.truncate(p_len);
        q.truncate(q_len);
        DynNodeEngel {
            a,
            b,
            p,
            q,
            neg,
            _s: PhantomData,
        }
    }

    fn merge(ctx: &Shift, l: Self, r: Self) -> Self {
        let (sl, sb) = split_sh(ctx.0 * (r.b - r.a));
        let pl = l.p.len() + r.q.len();

        let mut p = vec![0; (sl + pl + 1).max(r.p.len()) + 1];
        let (_, q) = join2(
            merge_par(l.q.len() + r.q.len()),
            || mul_dyn(&l.p, &r.q, &mut p[sl..sl + pl]),
            || product(&l.q, &r.q),
        );
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);
        let n = p.len();
        let (len, neg) = p_sum(&mut p, n, (0, &r.p), (l.neg, r.neg), false, S::SIGNED);
        p.truncate(len);

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
        let shift = (ctx.0 * (self.b - self.a)) as i64 - 64 * zeros as i64;
        ratio_to_fraction_dyn(&self.p, &self.q, shift, out);
        S::SIGNED && self.neg && !self.p.is_empty()
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
        let (p_len, q_len, ep, neg) = if fits_u64(a, b, |n| S::p(n) | S::q(n)) {
            engel_leaf_terms::<Narrow<S>>(a, b, shift, &mut p, &mut q)
        } else {
            engel_leaf_terms_wide::<S>(a, b, shift, &mut p, &mut q)
        };
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
        let (uo, un) = place(&mut r.p, r.p_len, r.ep, ep);
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

    fn merge(ctx: &DynHyperCtx, left: Self, right: Self) -> Self {
        let lr = left
            .r
            .expect("a node that ends the sum can't merge as a left child");
        let (sl, sb) = split_sh(ctx.shift * (right.b - right.a));
        let pl = left.p.len() + right.q.len();
        let tl = right.p.len() + lr.len();

        let par = merge_par(left.q.len() + right.q.len());
        let mut p = vec![0; (sl + pl + 1).max(tl) + 1];
        let mut scratch = ScratchGuard::acquire();
        let t = scratch.get(tl);
        let ((_, t_len), (q, r)) = join2(
            par,
            || {
                join2(
                    par,
                    || mul_dyn(&left.p, &right.q, &mut p[sl..sl + pl]),
                    || mul_exact(&right.p, &lr, t),
                )
            },
            || {
                join2(
                    par,
                    || product(&left.q, &right.q),
                    || right.r.as_deref().map(|rr| product(&lr, rr)),
                )
            },
        );
        p[sl + pl] = shl_buf(&mut p[sl..sl + pl], sb);
        let (n, signs) = (p.len(), (left.neg, right.neg));
        let (len, neg) = p_sum(&mut p, n, (0, &t[..t_len]), signs, false, S::SIGNED);
        p.truncate(len);

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
        let shift = (ctx.shift * (self.b - self.a)) as i64 - 64 * zeros as i64;
        ratio_to_fraction_dyn(&self.p, &self.q, shift, out);
        S::SIGNED && self.neg && !self.p.is_empty()
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
