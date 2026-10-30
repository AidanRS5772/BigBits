#![allow(dead_code)]
use crate::utils::{rows_dispatch, Rows, MUL_LIMB_ADX_CUTOFF};
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
    if sh < 0 {
        val.checked_shr(sh.unsigned_abs()).unwrap_or(0)
    } else {
        val.checked_shl(sh.unsigned_abs()).unwrap_or(0)
    }
}

#[inline]
pub fn signed_shr(val: u64, sh: i32) -> u64 {
    if sh < 0 {
        val.checked_shl(sh.unsigned_abs()).unwrap_or(0)
    } else {
        val.checked_shr(sh.unsigned_abs()).unwrap_or(0)
    }
}

#[inline]
pub fn lsb(val: u64, sh: i32) -> u64 {
    if sh <= 0 {
        0
    } else if sh >= 64 {
        val
    } else {
        val & ((1 << sh.unsigned_abs()) - 1)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn add_asm_x86(lhs: *mut u64, rhs: *const u64, len: usize) -> bool {
    let carry: u8;
    asm!(
        "clc",
        "2:",
        "mov {tmp}, [{r}]",
        "adc [{l}], {tmp}",
        "lea {l}, [{l} + 8]",
        "lea {r}, [{r} + 8]",
        "dec {len}",
        "jnz 2b",
        "setc {c}",
        l = inout(reg) lhs => _,
        r = inout(reg) rhs => _,
        len = inout(reg) len => _,
        c = out(reg_byte) carry,
        tmp = out(reg) _,
        options(nostack)
    );
    return carry != 0;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sub_asm_x86(lhs: *mut u64, rhs: *const u64, len: usize) -> bool {
    let carry: u8;
    asm!(
        "stc",
        "2:",
        "mov {tmp}, [{r}]",
        "not {tmp}",
        "adc [{l}], {tmp}",
        "lea {l}, [{l} + 8]",
        "lea {r}, [{r} + 8]",
        "dec {len}",
        "jnz 2b",
        "setc {c}",
        l = inout(reg) lhs => _,
        r = inout(reg) rhs => _,
        len = inout(reg) len => _,
        c = out(reg_byte) carry,
        tmp = out(reg) _,
        options(nostack),
    );
    return carry == 0;
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn add_asm_aarch(lhs: *mut u64, rhs: *const u64, len: usize) -> bool {
    let carry: u64;
    asm!(
        "adds xzr, xzr, xzr",
        "2:",
        "ldr {tmp1}, [{l}]",
        "ldr {tmp2}, [{r}], #8",
        "adcs {tmp1}, {tmp1}, {tmp2}",
        "str {tmp1}, [{l}], #8",
        "sub {len}, {len}, #1",
        "cbnz {len}, 2b",
        "cset {c}, cs",
        l = inout(reg) lhs => _,
        r = inout(reg) rhs => _,
        len = inout(reg) len => _,
        c = out(reg) carry,
        tmp1 = out(reg) _,
        tmp2 = out(reg) _,
        options(nostack)
    );
    return carry != 0;
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sub_asm_aarch(lhs: *mut u64, rhs: *const u64, len: usize) -> bool {
    let carry: u64;
    asm!(
        "cmp xzr, xzr",
        "2:",
        "ldr {tmp1}, [{l}]",
        "ldr {tmp2}, [{r}], #8",
        "mvn {tmp2}, {tmp2}",
        "adcs {tmp1}, {tmp1}, {tmp2}",
        "str {tmp1}, [{l}], #8",
        "sub {len}, {len}, #1",
        "cbnz {len}, 2b",
        "cset {c}, cs",
        l = inout(reg) lhs => _,
        r = inout(reg) rhs => _,
        len = inout(reg) len => _,
        c = out(reg) carry,
        tmp1 = out(reg) _,
        tmp2 = out(reg) _,
        options(nostack)
    );
    return carry == 0;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn inc_asm_x86(buf: *mut u64, len: usize) -> bool {
    let carry: u8;
    asm!(
        "stc",
        "2:",
        "adc QWORD PTR [{l}], 0",
        "lea {l}, [{l} + 8]",
        "jnc 3f",
        "dec {len}",
        "jnz 2b",
        "3:",
        "setc {c}",
        l = inout(reg) buf => _,
        len = inout(reg) len => _,
        c = out(reg_byte) carry,
        options(nostack)
    );
    return carry != 0;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dec_asm_x86(buf: *mut u64, len: usize) -> bool {
    let carry: u8;
    asm!(
        "stc",
        "2:",
        "sbb QWORD PTR [{l}], 0",
        "lea {l}, [{l} + 8]",
        "jnc 3f",
        "dec {len}",
        "jnz 2b",
        "3:",
        "setc {c}",
        l = inout(reg) buf => _,
        len = inout(reg) len => _,
        c = out(reg_byte) carry,
        options(nostack)
    );
    return carry != 0;
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn inc_asm_aarch(buf: *mut u64, len: usize) -> bool {
    let carry: u64;
    asm!(
        "cmp xzr, xzr",
        "2:",
        "ldr {tmp}, [{l}]",
        "adcs {tmp}, {tmp}, xzr",
        "str {tmp}, [{l}], #8",
        "b.cc 3f",
        "sub {len}, {len}, #1",
        "cbnz {len}, 2b",
        "3:",
        "cset {c}, cs",
        l = inout(reg) buf => _,
        len = inout(reg) len => _,
        c = out(reg) carry,
        tmp = out(reg) _,
        options(nostack)
    );
    return carry != 0;
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dec_asm_aarch(buf: *mut u64, len: usize) -> bool {
    let carry: u64;
    asm!(
        "adds xzr, xzr, xzr",
        "2:",
        "ldr {tmp}, [{l}]",
        "sbcs {tmp}, {tmp}, xzr",
        "str {tmp}, [{l}], #8",
        "b.cs 3f",
        "sub {len}, {len}, #1",
        "cbnz {len}, 2b",
        "3:",
        "cset {c}, cs",
        l = inout(reg) buf => _,
        len = inout(reg) len => _,
        c = out(reg) carry,
        tmp = out(reg) _,
        options(nostack)
    );
    return carry == 0;
}

#[inline(always)]
unsafe fn add_asm(lhs: *mut u64, rhs: *const u64, len: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        add_asm_aarch(lhs, rhs, len)
    }

    #[cfg(target_arch = "x86_64")]
    {
        add_asm_x86(lhs, rhs, len)
    }
}

#[inline(always)]
unsafe fn sub_asm(lhs: *mut u64, rhs: *const u64, len: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        sub_asm_aarch(lhs, rhs, len)
    }

    #[cfg(target_arch = "x86_64")]
    {
        sub_asm_x86(lhs, rhs, len)
    }
}

#[inline(always)]
unsafe fn inc_asm(buf: *mut u64, len: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        inc_asm_aarch(buf, len)
    }

    #[cfg(target_arch = "x86_64")]
    {
        inc_asm_x86(buf, len)
    }
}

#[inline(always)]
unsafe fn dec_asm(buf: *mut u64, len: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        dec_asm_aarch(buf, len)
    }
    #[cfg(target_arch = "x86_64")]
    {
        dec_asm_x86(buf, len)
    }
}

#[inline]
pub fn add_buf(lhs: &mut [u64], rhs: &[u64]) -> bool {
    let rhs_len = rhs.len();
    let lhs_len = lhs.len();
    assert!(lhs_len >= rhs_len, "lhs must be longer then rhs");
    if rhs_len == 0 {
        return false;
    }
    let lhs_ptr = lhs.as_mut_ptr();
    let rhs_ptr = rhs.as_ptr();
    let mut carry: bool;
    unsafe {
        carry = add_asm(lhs_ptr, rhs_ptr, rhs_len);
        if carry && lhs_len > rhs_len {
            carry = inc_asm(lhs_ptr.add(rhs_len), lhs_len - rhs_len);
        }
    }
    return carry;
}

#[inline]
pub fn sub_buf(lhs: &mut [u64], rhs: &[u64]) -> bool {
    let rhs_len = rhs.len();
    let lhs_len = lhs.len();
    assert!(lhs_len >= rhs_len, "lhs must be longer then rhs");
    if rhs_len == 0 {
        return false;
    }
    let lhs_ptr = lhs.as_mut_ptr();
    let rhs_ptr = rhs.as_ptr();
    let mut carry: bool;
    unsafe {
        carry = sub_asm(lhs_ptr, rhs_ptr, rhs_len);
        if carry && lhs_len > rhs_len {
            carry = dec_asm(lhs_ptr.add(rhs_len), lhs_len - rhs_len);
        }
    }
    return carry;
}

#[inline]
pub fn inc_buf(buf: &mut [u64]) -> bool {
    let buf_len = buf.len();
    if buf_len == 0 {
        return true;
    }

    unsafe {
        return inc_asm(buf.as_mut_ptr(), buf_len);
    }
}

#[inline]
pub fn dec_buf(buf: &mut [u64]) -> bool {
    let buf_len = buf.len();
    if buf_len == 0 {
        return true;
    }

    unsafe {
        return dec_asm(buf.as_mut_ptr(), buf_len);
    }
}

#[inline]
pub fn add_prim(buf: &mut [u64], prim: u64) -> bool {
    if prim == 0 {
        return false;
    }
    let buf_len = buf.len();
    if buf_len == 0 {
        return true;
    }
    let (res, c) = buf[0].overflowing_add(prim);
    buf[0] = res;
    return if c && buf_len > 1 {
        unsafe { inc_asm(buf.as_mut_ptr().add(1), buf_len - 1) }
    } else {
        c
    };
}

#[inline]
pub fn sub_prim(buf: &mut [u64], prim: u64) -> bool {
    if prim == 0 {
        return false;
    }
    let buf_len = buf.len();
    if buf_len == 0 {
        return true;
    }
    let (res, c) = buf[0].overflowing_sub(prim);
    buf[0] = res;
    return if c && buf_len > 1 {
        unsafe { dec_asm(buf.as_mut_ptr().add(1), buf_len - 1) }
    } else {
        c
    };
}

// Adds s * d into x (equal lengths) and returns the carry limb owed to x's next limb.
#[inline(always)]
pub fn add_mul_limb(x: &mut [u64], s: &[u64], d: u64) -> u64 {
    assert!(x.len() == s.len(), "x and s must have equal length");
    if s.is_empty() {
        return 0;
    }
    add_mul_row(x, s, d)
}

// Subtracts s * d from x (equal lengths) and returns the borrow limb owed by x's next limb.
#[inline(always)]
pub fn sub_mul_limb(x: &mut [u64], s: &[u64], d: u64) -> u64 {
    assert!(x.len() == s.len(), "x and s must have equal length");
    if s.is_empty() {
        return 0;
    }
    sub_mul_row(x, s, d)
}

// The rows behind add_mul_limb and sub_mul_limb, checked first: the ADX row runs from
// MUL_LIMB_ADX_CUTOFF limbs, below which its bmi2/adx call costs more than it saves.
rows_dispatch!(#[inline(always)]
    fn add_mul_row(x: &mut [u64], s: &[u64], d: u64) -> u64 = add_mul_rows
    if s.len() >= MUL_LIMB_ADX_CUTOFF);
rows_dispatch!(#[inline(always)]
    fn sub_mul_row(x: &mut [u64], s: &[u64], d: u64) -> u64 = sub_mul_rows
    if s.len() >= MUL_LIMB_ADX_CUTOFF);

#[inline(always)]
fn add_mul_rows<K: Rows>(x: &mut [u64], s: &[u64], d: u64) -> u64 {
    unsafe { K::add(d, s, x) }
}

#[inline(always)]
fn sub_mul_rows<K: Rows>(x: &mut [u64], s: &[u64], d: u64) -> u64 {
    unsafe { K::sub(d, s, x) }
}

// Adds s * d into x (x.len() >= s.len()), carrying through x's upper limbs, and
// returns the carry limb out of the top of x.
#[inline]
pub fn add_mul(x: &mut [u64], s: &[u64], d: u64) -> u64 {
    assert!(x.len() >= s.len(), "x must be at least as long as s");
    let (lo, hi) = x.split_at_mut(s.len());
    let carry = add_mul_limb(lo, s, d);
    if hi.is_empty() {
        carry
    } else {
        add_prim(hi, carry) as u64
    }
}

// Subtracts s * d from x (x.len() >= s.len()), borrowing through x's upper limbs,
// and returns the borrow limb owed by the limb above x.
#[inline]
pub fn sub_mul(x: &mut [u64], s: &[u64], d: u64) -> u64 {
    assert!(x.len() >= s.len(), "x must be at least as long as s");
    let (lo, hi) = x.split_at_mut(s.len());
    let borrow = sub_mul_limb(lo, s, d);
    if hi.is_empty() {
        borrow
    } else {
        sub_prim(hi, borrow) as u64
    }
}

pub fn twos_comp(buf: &mut [u64]) {
    for l in buf.iter_mut() {
        *l = !*l;
    }
    inc_buf(buf);
}

// buf >> sh in place, returning the bits shifted out of the bottom (at the top of the
// returned limb). Plain Rust, reading each limb's upper neighbour before it is overwritten,
// which LLVM vectorizes: about 1.5 cycles a limb from 64 limbs, against 3 for one shrd asm
// block per limb, and ahead from 8 limbs (a few cycles behind at 4).
pub fn shr_buf(buf: &mut [u64], sh: u8) -> u64 {
    assert!(sh < 64, "shift right must be less then 64");
    if sh == 0 || buf.is_empty() {
        return 0;
    }
    let (r, l) = (sh as u32, 64 - sh as u32);
    let carry = buf[0] << l;
    for i in 0..buf.len() - 1 {
        buf[i] = (buf[i] >> r) | (buf[i + 1] << l);
    }
    *buf.last_mut().unwrap() >>= r;
    carry
}

// buf << sh in place, returning the bits shifted out of the top. It runs upward, carrying each
// limb's old value to the next, a first-order recurrence LLVM vectorizes (the in-place
// downward loop doesn't vectorize): about 1.5 cycles a limb, as shr_buf.
pub fn shl_buf(buf: &mut [u64], sh: u8) -> u64 {
    assert!(sh < 64, "shift left must be less then 64");
    if sh == 0 || buf.is_empty() {
        return 0;
    }
    let (l, r) = (sh as u32, 64 - sh as u32);
    let mut prev = 0u64;
    for x in buf.iter_mut() {
        let cur = *x;
        *x = (cur << l) | (prev >> r);
        prev = cur;
    }
    prev >> r
}

// Copies the top out.len() limbs of (src << sh) into out; reads one limb below
// the copied window for the shifted-in bits and zero-pads when src is shorter.
pub(crate) fn shl_top_copy(src: &[u64], out: &mut [u64], sh: u8) {
    let copy_len = src.len().min(out.len());
    let src_start = src.len() - copy_len;
    let out_start = out.len() - copy_len;

    out[..out_start].fill(0);
    out[out_start..].copy_from_slice(&src[src_start..]);

    if sh != 0 && copy_len != 0 {
        shl_buf(&mut out[out_start..], sh);
        if src_start != 0 {
            out[0] |= src[src_start - 1] >> (64 - sh);
        }
    }
}

pub(crate) fn split_sh(bits: u64) -> (usize, u8) {
    ((bits / 64) as usize, (bits % 64) as u8)
}

#[inline(always)]
pub fn end_ref(buf: &[u64], idx: usize) -> &[u64] {
    &buf[buf.len().saturating_sub(idx)..]
}

#[inline(always)]
pub fn end_mut(buf: &mut [u64], idx: usize) -> &mut [u64] {
    let len = buf.len();
    &mut buf[len.saturating_sub(idx)..]
}
