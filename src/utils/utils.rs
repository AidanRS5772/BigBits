#![allow(dead_code)]
use std::arch::asm;

#[inline(always)]
pub fn trim_lz(vec: &mut Vec<u64>) {
    if let Some(idx) = vec.iter().rposition(|&x| x != 0) {
        vec.truncate(idx + 1);
    } else {
        vec.clear();
    }
}

#[inline(always)]
pub fn buf_len(val: &[u64]) -> usize {
    return if let Some(idx) = val.iter().rposition(|&x| x != 0) {
        idx + 1
    } else {
        0
    };
}

#[inline(always)]
pub fn scmp(s: bool, ord: std::cmp::Ordering) -> std::cmp::Ordering {
    return if s { ord.reverse() } else { ord };
}

#[inline(always)]
pub fn push_prim2(buf: &mut Vec<u64>, mut prim: u128) {
    if prim > 0 {
        buf.push(prim as u64);
        prim >>= 64;
        if prim > 0 {
            buf.push(prim as u64);
        }
    }
}

#[inline(always)]
pub fn combine_u64(x0: u64, x1: u64) -> u128 {
    x0 as u128 | (x1 as u128) << 64
}

pub fn eq_buf(lhs: &[u64], rhs: &[u64]) -> bool {
    use std::cmp::Ordering::*;

    match lhs.len().cmp(&rhs.len()) {
        Greater => {
            if lhs[rhs.len()..].iter().any(|&x| x != 0) {
                return false;
            }
        }
        Less => {
            if rhs[lhs.len()..].iter().any(|&x| x != 0) {
                return false;
            }
        }
        Equal => {}
    }

    for (l, r) in lhs.iter().zip(rhs) {
        if l != r {
            return false;
        }
    }
    return true;
}

pub fn cmp_buf(lhs: &[u64], rhs: &[u64]) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;

    match lhs.len().cmp(&rhs.len()) {
        Greater => {
            if lhs[rhs.len()..].iter().any(|&x| x != 0) {
                return Greater;
            }
        }
        Less => {
            if rhs[lhs.len()..].iter().any(|&x| x != 0) {
                return Less;
            }
        }
        Equal => {}
    }

    for (l, r) in lhs.iter().zip(rhs).rev() {
        let cmp = l.cmp(r);
        if cmp != Equal {
            return cmp;
        }
    }

    return Equal;
}

#[inline]
pub fn signed_shl(val: u64, sh: i32) -> u64 {
    return if sh < 0 {
        val >> sh.unsigned_abs()
    } else {
        val << sh.unsigned_abs()
    };
}

#[inline]
pub fn signed_shr(val: u64, sh: i32) -> u64 {
    return if sh < 0 {
        val << sh.unsigned_abs()
    } else {
        val >> sh.unsigned_abs()
    };
}

#[inline]
pub fn lsb(val: u64, sh: i32) -> u64 {
    return if sh <= 0 {
        0
    } else {
        val & ((1 << sh.unsigned_abs()) - 1)
    };
}

// adds values with carry and propagates carry on ARM
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn add_with_carry_aarch64(l: &mut u64, s: u64, c: &mut u8) {
    asm!(
        "subs wzr, {c:w}, #1", // c -> cf
        "adcs {l}, {l}, {s}", // l+s+cf -> l , updates cf
        "cset {c:w}, cs", // cf -> c
        l = inout(reg) *l,
        s = in(reg) s,
        c = inout(reg) *c,
        options(nostack)
    );
}

// adds values with carry and propagates carry on x86
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn add_with_carry_x86_64(l: &mut u64, s: u64, c: &mut u8) {
    asm!(
        "add {c} , 0xFF", // c -> cc
        "adc {l}, {s}", // l+s+cf -> l , updates cf
        "setc {c}", // cf -> c
        c = inout(reg_byte) *c,
        l = inout(reg) *l,
        s = in(reg) s,
        options(nostack)
    );
}

//architecture wrapper
#[inline(always)]
pub(super) unsafe fn add_with_carry(l: &mut u64, s: u64, c: &mut u8) {
    #[cfg(target_arch = "aarch64")]
    add_with_carry_aarch64(l, s, c);

    #[cfg(target_arch = "x86_64")]
    add_with_carry_x86_64(l, s, c);
}

pub fn acc(lhs: &mut [u64], rhs: &[u64], comp: u8) -> bool {
    let mask = if comp == 0 { 0 } else { u64::MAX };
    let cf = comp;
    unsafe {
        let mut c = comp;
        for (l, s) in lhs.iter_mut().zip(rhs) {
            add_with_carry(l, *s ^ mask, &mut c);
        }
        if c != cf {
            for l in &mut lhs[rhs.len()..] {
                add_with_carry(l, mask, &mut c);
                if c == cf {
                    break;
                }
            }
        }

        return c != cf;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn inc_propagate_aarch64(l: &mut u64, c: &mut u8) {
    asm!(
        "adds {l}, {l}, {c}", // l + c -> l, sets CF
        "cset {c:w}, cs",     // CF -> c
        l = inout(reg) * l,
        c = inout(reg) * c,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn inc_propagate_x86_64(l: &mut u64, c: &mut u8) {
    asm!(
        "add {c}, 0xFF", // c -> CF (0xFF + 1 overflows, 0xFF + 0 doesn't)
        "adc {l}, 0",    // l + 0 + CF -> l
        "setc {c}",      // CF -> c
        c = inout(reg_byte) * c,
        l = inout(reg) * l,
        options(nostack)
    );
}

#[inline(always)]
unsafe fn inc_propagate(l: &mut u64, c: &mut u8) {
    #[cfg(target_arch = "aarch64")]
    inc_propagate_aarch64(l, c);
    #[cfg(target_arch = "x86_64")]
    inc_propagate_x86_64(l, c);
}

pub fn inc(lhs: &mut [u64]) -> bool {
    unsafe {
        let mut c: u8 = 1;
        for l in lhs.iter_mut() {
            inc_propagate(l, &mut c);
            if c == 0 {
                return false;
            }
        }
        true
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dec_propagate_aarch64(l: &mut u64, b: &mut u8) {
    asm!(
        "subs {l}, {l}, {b}", // l - b -> l, sets C = NOT borrow
        "cset {b:w}, cc",     // cc (carry clear) = borrow occurred -> b
        l = inout(reg) * l,
        b = inout(reg) * b,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dec_propagate_x86_64(l: &mut u64, b: &mut u8) {
    asm!(
        "add {b}, 0xFF", // b -> CF
        "sbb {l}, 0",    // l - 0 - CF -> l
        "setc {b}",      // CF -> b
        b = inout(reg_byte) * b,
        l = inout(reg) * l,
        options(nostack)
    );
}

#[inline(always)]
unsafe fn dec_propagate(l: &mut u64, b: &mut u8) {
    #[cfg(target_arch = "aarch64")]
    dec_propagate_aarch64(l, b);
    #[cfg(target_arch = "x86_64")]
    dec_propagate_x86_64(l, b);
}

pub fn dec(lhs: &mut [u64]) -> bool {
    unsafe {
        let mut b: u8 = 1;
        for l in lhs.iter_mut() {
            dec_propagate(l, &mut b);
            if b == 0 {
                return false;
            }
        }
        true
    }
}

pub fn twos_comp(buf: &mut [u64]) {
    for l in buf.iter_mut() {
        *l = !*l;
    }
    inc(buf);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shr_carry_aarch64(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    asm!(
        "lsl {tmp}, {e}, {ms:x}", // put the last bits of the e into tmp
        "lsr {e}, {e}, {r:x}", // shift e by rem
        "orr {e}, {e}, {c}", // put the last bits of previous e at the start of e
        "mov {c}, {tmp}", // put tmp into carry
        e = inout(reg) *e,
        c = inout(reg) *c,
        r = in(reg) sh,
        ms = in(reg) mv_sz,
        tmp = out(reg) _,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn shr_carry_x86_64(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    asm!(
        "mov {tmp}, {e}", // put e into tmp
        "mov cl , {ms}", // move into cl reg
        "shl {tmp}, cl", // get last digits of e into tmp
        "mov cl , {r}", // move into cl reg
        "shr {e}, cl", // shift e by rem
        "or {e}, {c}", // put the last bits of previous e at the start of e
        "mov {c}, {tmp}", // put tmp into carry
        e = inout(reg) *e,
        c = inout(reg) *c,
        r = in(reg_byte) sh,
        ms = in(reg_byte) mv_sz,
        tmp = out(reg) _,
        out("cl") _ ,
        options(nostack)
    );
}

unsafe fn shr_carry(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    #[cfg(target_arch = "aarch64")]
    shr_carry_aarch64(e, c, sh, mv_sz);

    #[cfg(target_arch = "x86_64")]
    shr_carry_x86_64(e, c, sh, mv_sz);
}

pub fn shr_buf(buf: &mut [u64], sh: u8) -> u64 {
    if sh == 0 {
        return 0;
    }

    assert!(sh < 64);
    let mv_sz = 64 - sh;
    unsafe {
        let mut c = 0;
        for e in buf.iter_mut().rev() {
            shr_carry(e, &mut c, sh, mv_sz);
        }
        return c;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shl_carry_aarch64(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    asm!(
        "lsr {tmp}, {e}, {ms:x}", // put the last bits of the e into tmp
        "lsl {e}, {e}, {r:x}", // shift e by rem
        "orr {e}, {e}, {c}", // put the last bits of previous e at the start of e
        "mov {c}, {tmp}", // put tmp into carry
        e = inout(reg) *e,
        c = inout(reg) *c,
        r = in(reg) sh,
        ms = in(reg) mv_sz,
        tmp = out(reg) _,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn shl_carry_x86_64(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    asm!(
        "mov {tmp}, {e}", // put e into tmp
        "mov cl , {ms}", // move into cl reg
        "shr {tmp}, cl", // get last digits of e into tmp
        "mov cl , {r}", // move into cl reg
        "shl {e}, cl", // shift e by rem
        "or {e}, {c}", // put the last bits of previous e at the start of e
        "mov {c}, {tmp}", // put tmp into carry
        e = inout(reg) *e,
        c = inout(reg) *c,
        r = in(reg_byte) sh,
        ms = in(reg_byte) mv_sz,
        tmp = out(reg) _,
        out("cl") _ ,
        options(nostack)
    );
}

unsafe fn shl_carry(e: &mut u64, c: &mut u64, sh: u8, mv_sz: u8) {
    #[cfg(target_arch = "aarch64")]
    shl_carry_aarch64(e, c, sh, mv_sz);

    #[cfg(target_arch = "x86_64")]
    shl_carry_x86_64(e, c, sh, mv_sz);
}

pub fn shl_buf(buf: &mut [u64], sh: u8) -> u64 {
    if sh == 0 {
        return 0;
    }

    assert!(sh < 64);
    let mv_sz = 64 - sh;
    unsafe {
        let mut c = 0;
        for e in buf.iter_mut() {
            shl_carry(e, &mut c, sh, mv_sz);
        }
        return c;
    }
}
