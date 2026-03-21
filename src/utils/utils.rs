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
    debug_assert!(lhs_len >= rhs_len, "lhs must be longer then rhs");
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
    debug_assert!(lhs_len >= rhs_len, "lhs must be longer then rhs");
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

    let buf_ptr = buf.as_mut_ptr();
    unsafe {
        return inc_asm(buf_ptr, buf_len);
    }
}

#[inline]
pub fn dec_buf(buf: &mut [u64]) -> bool {
    let buf_len = buf.len();
    if buf_len == 0 {
        return true;
    }

    let buf_ptr = buf.as_mut_ptr();
    unsafe {
        return dec_asm(buf_ptr, buf_len);
    }
}

pub fn twos_comp(buf: &mut [u64]) {
    for l in buf.iter_mut() {
        *l = !*l;
    }
    inc_buf(buf);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn shr_asm_x86(dst: &mut u64, src: u64, sh: u8) {
    asm!(
        "shrd {dst}, {src}, cl",
        dst = inout(reg) *dst,
        src = in(reg) src,
        in("cl") sh,
        options(nostack, nomem)
    );
}

unsafe fn shr_asm(dst: &mut u64, src: u64, sh: u8) {
    #[cfg(target_arch = "aarch64")]
    {
        *dst = (*dst >> sh) | (src << (64 - sh))
    }

    #[cfg(target_arch = "x86_64")]
    shr_asm_x86(dst, src, sh);
}

pub fn shr_buf(buf: &mut [u64], sh: u8) -> u64 {
    if sh == 0 || buf.is_empty() {
        return 0;
    }
    debug_assert!(sh < 64, "shift right must be less then 64");
    let carry = buf[0] << (64 - sh);
    for i in 0..buf.len() - 1 {
        let src = buf[i + 1];
        unsafe {
            shr_asm(&mut buf[i], src, sh);
        }
    }
    *buf.last_mut().unwrap() >>= sh;
    return carry;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn shl_asm_x86(dst: &mut u64, src: u64, sh: u8) {
    asm!(
        "shld {dst}, {src}, cl",
        dst = inout(reg) *dst,
        src = in(reg) src,
        in("cl") sh,
        options(nostack, nomem)
    );
}

unsafe fn shl_asm(dst: &mut u64, src: u64, sh: u8) {
    #[cfg(target_arch = "aarch64")]
    {
        *dst = (*dst << sh) | (src >> (64 - sh))
    }

    #[cfg(target_arch = "x86_64")]
    shl_asm_x86(dst, src, sh);
}

pub fn shl_buf(buf: &mut [u64], sh: u8) -> u64 {
    if sh == 0 || buf.is_empty() {
        return 0;
    }
    debug_assert!(sh < 64, "shift left must be less then 64");
    let carry = *buf.last().unwrap() >> (64 - sh);
    for i in (1..buf.len()).rev() {
        let src = buf[i - 1];
        unsafe { shl_asm(&mut buf[i], src, sh) }
    }
    buf[0] <<= sh;
    return carry;
}
