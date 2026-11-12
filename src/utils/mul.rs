#![allow(dead_code)]
use crate::utils::verify::Residues;
use crate::tuning::k;
use crate::utils::{utils::*, *};
use core::task;
use rayon::*;
use rustfft::num_traits::{One, Zero};
use rustfft::{num_complex::Complex, Fft, FftDirection, FftPlanner};
use std::arch::asm;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::ffi::c_short;
use std::marker::PhantomData;
use std::ops::*;
use std::sync::{Arc, LazyLock};
use std::thread::LocalKey;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// SCHOOL BOOK MULTIPLICATION
// Worst complexity lowest overhead optimized for small inputs. Base tier algorithm.
// General: O(ls)
// Primitive Optimizations: O(l)

// buf = buf * s + c in place, returning the top limb. The whole loop is one asm
// block, so the running carry stays in a register instead of passing through Rust
// between limbs.

// mul clobbers the flags, so each pass multiplies two limbs into registers and
// then closes the pair with one add/adc/adc chain. The pair's top limb can't
// overflow (c + s * buf[i..i + 2] < 2^192), and the block needs only caller-saved
// registers, so short calls pay no spills.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_add_prim_base(buf: &mut [u64], s: u64, c: u64) -> u64 {
    let (len, p) = (buf.len(), buf.as_mut_ptr());
    let top: u64;
    asm!(
        // An odd limb first, then pairs.
        "shr {n}, 1",
        "jnc 2f",
        "mov rax, [{p}]",
        "mul {s}",
        "add rax, {c}",
        "adc rdx, 0",
        "mov [{p}], rax",
        "mov {c}, rdx",
        "add {p}, 8",
        "2:",
        "test {n}, {n}",
        "jz 4f",
        "3:",
        "mov rax, [{p}]",
        "mul {s}",
        "mov {l0}, rax",
        "mov {h0}, rdx",
        "mov rax, [{p} + 8]",
        "mul {s}",
        "add {l0}, {c}",
        "adc rax, {h0}",
        "adc rdx, 0",
        "mov [{p}], {l0}",
        "mov [{p} + 8], rax",
        "mov {c}, rdx",
        "add {p}, 16",
        "dec {n}",
        "jnz 3b",
        "4:",
        s = in(reg) s,
        p = inout(reg) p => _,
        n = inout(reg) len => _,
        c = inout(reg) c => top,
        l0 = out(reg) _,
        h0 = out(reg) _,
        out("rax") _,
        out("rdx") _,
        options(nostack),
    );
    top
}

// mulx leaves the flags alone, so one adcx chain (CF) runs through the whole row:
// Adx's set row, with buf as both the row and its output.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,adx")]
unsafe fn mul_add_prim_adx(buf: &mut [u64], s: u64, c: u64) -> u64 {
    adx_set_row(s, buf.as_ptr(), buf.as_mut_ptr(), buf.len(), c)
}

// One carry flag, as in Base::set: each pass of four limbs multiplies them all,
// then adds the low limbs and the high limbs one up in a single adds/adcs chain.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_add_prim_base(buf: &mut [u64], s: u64, c: u64) -> u64 {
    let (len, p) = (buf.len(), buf.as_mut_ptr());
    let top: u64;
    asm!(
        "cbz {n}, 3f",
        "2:",
        "ldp {a0}, {a1}, [{p}]",
        "ldp {a2}, {a3}, [{p}, #16]",
        "umulh {h0}, {a0}, {s}",
        "mul {a0}, {a0}, {s}",
        "umulh {h1}, {a1}, {s}",
        "mul {a1}, {a1}, {s}",
        "umulh {h2}, {a2}, {s}",
        "mul {a2}, {a2}, {s}",
        "umulh {h3}, {a3}, {s}",
        "mul {a3}, {a3}, {s}",
        "adds {a0}, {a0}, {c}",
        "adcs {a1}, {a1}, {h0}",
        "adcs {a2}, {a2}, {h1}",
        "adcs {a3}, {a3}, {h2}",
        // No carry out: c + s * buf[i..i + 4] < 2^320.
        "adc {c}, {h3}, xzr",
        "stp {a0}, {a1}, [{p}], #16",
        "stp {a2}, {a3}, [{p}], #16",
        "sub {n}, {n}, #1",
        "cbnz {n}, 2b",
        "3:",

        "cbz {rem}, 5f",
        "4:",
        "ldr {a0}, [{p}]",
        "umulh {h0}, {a0}, {s}",
        "mul {a0}, {a0}, {s}",
        "adds {a0}, {a0}, {c}",
        "adc {c}, {h0}, xzr",                   // h0 <= 2^64 - 2, so no carry out
        "str {a0}, [{p}], #8",
        "sub {rem}, {rem}, #1",
        "cbnz {rem}, 4b",
        "5:",
        s = in(reg) s,
        p = inout(reg) p => _,
        n = inout(reg) len / 4 => _,
        rem = inout(reg) len % 4 => _,
        c = inout(reg) c => top,
        a0 = out(reg) _,
        a1 = out(reg) _,
        a2 = out(reg) _,
        a3 = out(reg) _,
        h0 = out(reg) _,
        h1 = out(reg) _,
        h2 = out(reg) _,
        h3 = out(reg) _,
        options(nostack),
    );
    top
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
unsafe fn mul_add_prim_base(buf: &mut [u64], s: u64, mut c: u64) -> u64 {
    for e in buf {
        let t = *e as u128 * s as u128 + c as u128;
        *e = t as u64;
        c = (t >> 64) as u64;
    }
    c
}

pub fn mul_prim(buf: &mut [u64], prim: u64) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    if prim == 0 {
        buf.fill(0);
        return 0;
    }
    mul_add_prim(buf, prim, 0)
}

// buf = buf * prim + add, returning the carry limb out of the top of buf.
#[inline]
pub fn mul_add_prim(buf: &mut [u64], prim: u64, add: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if buf.len() >= k::base::prim_adx() && adx_available() {
        return unsafe { mul_add_prim_adx(buf, prim, add) };
    }
    unsafe { mul_add_prim_base(buf, prim, add) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_asm_x86(a_val: u64, b_val: u64, acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) {
    asm!(
        "mul {b_val}",
        "add {acc0}, rax",
        "adc {acc1}, rdx",
        "adc {acc2}, 0",
        b_val = in(reg) b_val,
        inout("rax") a_val => _,
        out("rdx") _,
        acc0 = inout(reg) *acc0,
        acc1 = inout(reg) *acc1,
        acc2 = inout(reg) *acc2,
        options(nostack, nomem),
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_asm_aarch(a_val: u64, b_val: u64, acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) {
    asm!(
        "mul {lo}, {a_val}, {b_val}",
        "umulh {hi}, {a_val}, {b_val}",
        "adds {acc0}, {acc0}, {lo}",
        "adcs {acc1}, {acc1}, {hi}",
        "adc {acc2}, {acc2}, xzr",
        a_val = in(reg) a_val,
        b_val = in(reg) b_val,
        lo = out(reg) _,
        hi = out(reg) _,
        acc0 = inout(reg) *acc0,
        acc1 = inout(reg) *acc1,
        acc2 = inout(reg) *acc2,
        options(nostack, nomem),
    );
}

#[inline(always)]
unsafe fn mul_asm(a_val: u64, b_val: u64, acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) {
    #[cfg(target_arch = "aarch64")]
    mul_asm_aarch(a_val, b_val, acc0, acc1, acc2);

    #[cfg(target_arch = "x86_64")]
    mul_asm_x86(a_val, b_val, acc0, acc1, acc2);
}

pub fn mul_prim2(buf: &mut [u64], prim: u128) -> u128 {
    if buf.is_empty() {
        return 0;
    }
    if prim == 0 {
        buf.fill(0);
        return 0;
    }
    let (p0, p1) = (prim as u64, (prim >> 64) as u64);
    if p1 == 0 {
        return mul_add_prim(buf, p0, 0) as u128;
    }
    #[cfg(target_arch = "x86_64")]
    if adx_available() {
        return unsafe { mul_prim2_adx(buf, p0, p1) };
    }
    unsafe { mul_prim2_base(buf, p0, p1) }
}

// One pass, four limbs of b at a time. Each output limb sums four partial products,
// which takes three carry chains (b p0's, b p1's and their sum), and ADX has two
// flags. So each block runs b p1's row first on CF into registers (ra..rd, one
// limb up), then b p0's row on CF, adding the b p1 limbs on OF and storing. CF
// folds into the pending high limb after each row, which can't overflow (a mulx
// high limb is at most 2^64 - 2), while OF, the sum's carry, stays live to the
// end: mulx, adcx, mov, lea and jrcxz leave it alone. re carries b p1's limb that
// lands on the next block's first limb. A clc after each fold writes CF without
// reading it, so one row's CF chain doesn't wait on the other's (38% faster than
// the mul loop from 64 limbs, against 31% without). All 13 free registers are in
// use (one holds zero for the folds), so p0, p1 and len % 4 wait in xmm registers
// (movq costs port 0, but a stack copy costs store-forwarding at entry). It runs
// at every length: the mul loop is ahead below 8 limbs, by 6.5 ns at one limb and
// 2-3 ns from four, too little to dispatch on.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,adx")]
unsafe fn mul_prim2_adx(buf: &mut [u64], p0: u64, p1: u64) -> u128 {
    let (len, p) = (buf.len(), buf.as_mut_ptr());
    let (hp0, hp1, re, of): (u64, u64, u64, u64);
    asm!(
        "xor {z:e}, {z:e}",
        "xor {hp0:e}, {hp0:e}",
        "xor {hp1:e}, {hp1:e}",
        "xor {re:e}, {re:e}",
        "test rcx, rcx",                        // CF = OF = 0
        "jz 3f",
        "2:",
        // b p1 at limbs 1..4 of the block, into ra..rd.
        "movq rdx, {p1}",
        "mulx {x}, {ra}, [{p}]",
        "adcx {ra}, {hp1}",
        "mulx {hp1}, {rb}, [{p} + 8]",
        "adcx {rb}, {x}",
        "mulx {x}, {rc}, [{p} + 16]",
        "adcx {rc}, {hp1}",
        "mulx {hp1}, {rd}, [{p} + 24]",
        "adcx {rd}, {x}",
        "adcx {hp1}, {z}",                      // CF = 0
        "clc",                                  // so b p0's row needn't wait for it
        // b p0 at limbs 0..3, plus b p1's re, ra, rb, rc on OF.
        "movq rdx, {p0}",
        "mulx {x}, {lo}, [{p}]",
        "adcx {lo}, {hp0}",
        "adox {lo}, {re}",
        "mov [{p}], {lo}",
        "mulx {hp0}, {lo}, [{p} + 8]",
        "adcx {lo}, {x}",
        "adox {lo}, {ra}",
        "mov [{p} + 8], {lo}",
        "mulx {x}, {lo}, [{p} + 16]",
        "adcx {lo}, {hp0}",
        "adox {lo}, {rb}",
        "mov [{p} + 16], {lo}",
        "mulx {hp0}, {lo}, [{p} + 24]",
        "adcx {lo}, {x}",
        "adox {lo}, {rc}",
        "mov [{p} + 24], {lo}",
        "adcx {hp0}, {z}",                      // CF = 0
        "clc",
        "mov {re}, {rd}",
        "lea {p}, [{p} + 32]",
        "lea rcx, [rcx - 1]",
        "jrcxz 3f",                             // jrcxz only reaches 128 bytes
        "jmp 2b",
        "3:",
        // The last len % 4 limbs one at a time, the same two rows.
        "movq rcx, {rm}",
        "4:",
        "jrcxz 5f",
        "movq rdx, {p1}",
        "mulx {x}, {ra}, [{p}]",
        "adcx {ra}, {hp1}",
        "mov {hp1}, {x}",
        "adcx {hp1}, {z}",
        "movq rdx, {p0}",
        "mulx {x}, {lo}, [{p}]",
        "adcx {lo}, {hp0}",
        "adox {lo}, {re}",
        "mov [{p}], {lo}",
        "mov {hp0}, {x}",
        "adcx {hp0}, {z}",
        "mov {re}, {ra}",
        "lea {p}, [{p} + 8]",
        "lea rcx, [rcx - 1]",
        "jmp 4b",
        "5:",
        "adox {z}, {z}",                        // z = OF
        inout("rcx") len / 4 => _,
        out("rdx") _,
        p0 = in(xmm_reg) p0,
        p1 = in(xmm_reg) p1,
        rm = in(xmm_reg) (len % 4) as u64,
        p = inout(reg) p => _,
        hp0 = out(reg) hp0,
        hp1 = out(reg) hp1,
        re = out(reg) re,
        z = out(reg) of,
        lo = out(reg) _,
        x = out(reg) _,
        ra = out(reg) _,
        rb = out(reg) _,
        rc = out(reg) _,
        rd = out(reg) _,
        options(nostack),
    );
    // b p0's high limb, b p1's last low limb and OF land on limb len, b p1's high
    // limb on len + 1. The product's top two limbs can't overflow u128.
    hp0 as u128 + re as u128 + of as u128 + ((hp1 as u128) << 64)
}

// buf = buf * (p1 B + p0) in place for nonempty buf, returning the top two limbs.
#[inline(always)]
unsafe fn mul_prim2_base(buf: &mut [u64], p0: u64, p1: u64) -> u128 {
    let len = buf.len();

    let mut acc0: u64 = 0;
    let mut acc1: u64 = 0;
    let mut acc2: u64 = 0;

    let mut prev = buf[0];
    unsafe {
        mul_asm(prev, p0, &mut acc0, &mut acc1, &mut acc2);
    }

    for i in 1..len {
        let cur = unsafe { *buf.get_unchecked(i) };
        unsafe {
            *buf.get_unchecked_mut(i - 1) = acc0;
        }
        acc0 = acc1;
        acc1 = acc2;
        acc2 = 0;
        unsafe {
            mul_asm(prev, p1, &mut acc0, &mut acc1, &mut acc2);
            mul_asm(cur, p0, &mut acc0, &mut acc1, &mut acc2);
        }
        prev = cur;
    }

    buf[len - 1] = acc0;
    acc0 = acc1;
    acc1 = acc2;
    acc2 = 0;

    unsafe {
        mul_asm(prev, p1, &mut acc0, &mut acc1, &mut acc2);
    }

    (acc1 as u128) << 64 | acc0 as u128
}

// acc2:acc1:acc0 += Σ a[k] * b[-k] over k < len: one product column, with b
// pointing at its last limb and read descending.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_col(
    a: *const u64,
    b: *const u64,
    len: usize,
    acc0: &mut u64,
    acc1: &mut u64,
    acc2: &mut u64,
) {
    asm!(
        "test {n4}, {n4}",
        "jz 3f",
        "2:",
        "mov rax, [{a}]",
        "mul qword ptr [{b}]",
        "add {acc0}, rax",
        "adc {acc1}, rdx",
        "adc {acc2}, 0",
        "mov rax, [{a} + 8]",
        "mul qword ptr [{b} - 8]",
        "add {acc0}, rax",
        "adc {acc1}, rdx",
        "adc {acc2}, 0",
        "mov rax, [{a} + 16]",
        "mul qword ptr [{b} - 16]",
        "add {acc0}, rax",
        "adc {acc1}, rdx",
        "adc {acc2}, 0",
        "mov rax, [{a} + 24]",
        "mul qword ptr [{b} - 24]",
        "add {acc0}, rax",
        "adc {acc1}, rdx",
        "adc {acc2}, 0",
        "add {a}, 32",
        "sub {b}, 32",
        "dec {n4}",
        "jnz 2b",
        "3:",

        "test {rem}, {rem}",
        "jz 5f",
        "mov rax, [{a}]",
        "mul qword ptr [{b}]",
        "add {acc0}, rax",
        "adc {acc1}, rdx",
        "adc {acc2}, 0",
        "cmp {rem}, 1",
        "je 5f",
        "mov rax, [{a} + 8]",
        "mul qword ptr [{b} - 8]",
        "add {acc0}, rax",
        "adc {acc1}, rdx",
        "adc {acc2}, 0",
        "cmp {rem}, 2",
        "je 5f",
        "mov rax, [{a} + 16]",
        "mul qword ptr [{b} - 16]",
        "add {acc0}, rax",
        "adc {acc1}, rdx",
        "adc {acc2}, 0",
        "5:",
        a = inout(reg) a => _,
        b = inout(reg) b => _,
        n4 = inout(reg) len / 4 => _,
        rem = in(reg) len % 4,
        acc0 = inout(reg) *acc0,
        acc1 = inout(reg) *acc1,
        acc2 = inout(reg) *acc2,
        out("rax") _,
        out("rdx") _,
        options(nostack, readonly),
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_col(
    a: *const u64,
    b: *const u64,
    len: usize,
    acc0: &mut u64,
    acc1: &mut u64,
    acc2: &mut u64,
) {
    asm!(
        "cbz {n4}, 3f",
        "2:",
        "ldp {x0}, {x1}, [{a}]",
        "ldp {x2}, {x3}, [{a}, #16]",
        "ldp {y1}, {y0}, [{b}, #-8]",
        "ldp {y3}, {y2}, [{b}, #-24]",
        "umulh {h0}, {x0}, {y0}",
        "mul {x0}, {x0}, {y0}",
        "umulh {h1}, {x1}, {y1}",
        "mul {x1}, {x1}, {y1}",
        "umulh {h2}, {x2}, {y2}",
        "mul {x2}, {x2}, {y2}",
        "umulh {h3}, {x3}, {y3}",
        "mul {x3}, {x3}, {y3}",
        "adds {acc0}, {acc0}, {x0}",
        "adcs {acc1}, {acc1}, {h0}",
        "adc {acc2}, {acc2}, xzr",
        "adds {acc0}, {acc0}, {x1}",
        "adcs {acc1}, {acc1}, {h1}",
        "adc {acc2}, {acc2}, xzr",
        "adds {acc0}, {acc0}, {x2}",
        "adcs {acc1}, {acc1}, {h2}",
        "adc {acc2}, {acc2}, xzr",
        "adds {acc0}, {acc0}, {x3}",
        "adcs {acc1}, {acc1}, {h3}",
        "adc {acc2}, {acc2}, xzr",
        "add {a}, {a}, #32",
        "sub {b}, {b}, #32",
        "sub {n4}, {n4}, #1",
        "cbnz {n4}, 2b",
        "3:",

        "cbz {rem}, 5f",
        "4:",
        "ldr {x0}, [{a}], #8",
        "ldr {y0}, [{b}], #-8",
        "umulh {h0}, {x0}, {y0}",
        "mul {x0}, {x0}, {y0}",
        "adds {acc0}, {acc0}, {x0}",
        "adcs {acc1}, {acc1}, {h0}",
        "adc {acc2}, {acc2}, xzr",
        "sub {rem}, {rem}, #1",
        "cbnz {rem}, 4b",
        "5:",
        a = inout(reg) a => _,
        b = inout(reg) b => _,
        n4 = inout(reg) len / 4 => _,
        rem = inout(reg) len % 4 => _,
        acc0 = inout(reg) *acc0,
        acc1 = inout(reg) *acc1,
        acc2 = inout(reg) *acc2,
        x0 = out(reg) _,
        x1 = out(reg) _,
        x2 = out(reg) _,
        x3 = out(reg) _,
        y0 = out(reg) _,
        y1 = out(reg) _,
        y2 = out(reg) _,
        y3 = out(reg) _,
        h0 = out(reg) _,
        h1 = out(reg) _,
        h2 = out(reg) _,
        h3 = out(reg) _,
        options(nostack, readonly),
    );
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn mul_col(
    a: *const u64,
    b: *const u64,
    len: usize,
    acc0: &mut u64,
    acc1: &mut u64,
    acc2: &mut u64,
) {
    for k in 0..len {
        let lo = (*acc1 as u128) << 64 | *acc0 as u128;
        let (lo, carry) = lo.overflowing_add(*a.add(k) as u128 * *b.sub(k) as u128);
        *acc0 = lo as u64;
        *acc1 = (lo >> 64) as u64;
        *acc2 += carry as u64;
    }
}

#[inline(always)]
pub fn mul_elem(
    a: &[u64],
    b: &[u64],
    n: usize,
    acc0: &mut u64,
    acc1: &mut u64,
    acc2: &mut u64,
) -> u64 {
    // a[m] * b[n - m] for m in mi..=mf, both indices in bounds.
    let mi = n.saturating_sub(b.len() - 1);
    let mf = n.min(a.len() - 1);
    if mi <= mf {
        unsafe {
            mul_col(
                a.as_ptr().add(mi),
                b.as_ptr().add(n - mi),
                mf - mi + 1,
                acc0,
                acc1,
                acc2,
            );
        }
    }
    let out = *acc0;
    *acc0 = *acc1;
    *acc1 = *acc2;
    *acc2 = 0;
    return out;
}

// Rows (mod.rs) for mul_buf and its kin; callers check out.len() >= l.len() once per
// product. mulx/adcx/adox rows, run inside rows_dispatch!'s bmi2/adx copies. Two carry
// chains run side by side, adcx (CF) between products and adox (OF) into out, so the
// loops count with lea and jrcxz, which leave both flags alone.
#[cfg(target_arch = "x86_64")]
impl Rows for Adx {
    #[inline(always)]
    unsafe fn add(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        debug_assert!(out.len() >= l.len());
        adx_add_row(s, l.as_ptr(), out.as_mut_ptr(), l.len(), 0)
    }

    #[inline(always)]
    unsafe fn set(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        debug_assert!(out.len() >= l.len());
        adx_set_row(s, l.as_ptr(), out.as_mut_ptr(), l.len(), 0)
    }

    #[inline(always)]
    unsafe fn sub(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        debug_assert!(out.len() >= l.len());
        adx_sub_row(s, l.as_ptr(), out.as_mut_ptr(), l.len())
    }
}

// out[..len] += s * l[..len] + c, returning the top limb.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn adx_add_row(s: u64, l: *const u64, out: *mut u64, len: usize, c: u64) -> u64 {
    let top: u64;
    asm!(
        "test {h0}, 0",                         // CF = OF = 0
        // Four limbs per pass; the high limb alternates between h0 and h1.
        "2:",
        "jrcxz 3f",
        "mulx {h1}, {lo}, [{l}]",
        "adcx {lo}, {h0}",
        "adox {lo}, [{o}]",
        "mov [{o}], {lo}",
        "mulx {h0}, {lo}, [{l} + 8]",
        "adcx {lo}, {h1}",
        "adox {lo}, [{o} + 8]",
        "mov [{o} + 8], {lo}",
        "mulx {h1}, {lo}, [{l} + 16]",
        "adcx {lo}, {h0}",
        "adox {lo}, [{o} + 16]",
        "mov [{o} + 16], {lo}",
        "mulx {h0}, {lo}, [{l} + 24]",
        "adcx {lo}, {h1}",
        "adox {lo}, [{o} + 24]",
        "mov [{o} + 24], {lo}",
        "lea {l}, [{l} + 32]",
        "lea {o}, [{o} + 32]",
        "lea rcx, [rcx - 1]",
        "jmp 2b",
        "3:",
        // The last len % 4 limbs, leaving by jrcxz as rcx runs out.
        "mov rcx, {rem}",
        "jrcxz 5f",
        "mulx {h1}, {lo}, [{l}]",
        "adcx {lo}, {h0}",
        "adox {lo}, [{o}]",
        "mov [{o}], {lo}",
        "lea rcx, [rcx - 1]",
        "jrcxz 4f",
        "mulx {h0}, {lo}, [{l} + 8]",
        "adcx {lo}, {h1}",
        "adox {lo}, [{o} + 8]",
        "mov [{o} + 8], {lo}",
        "lea rcx, [rcx - 1]",
        "jrcxz 5f",
        "mulx {h1}, {lo}, [{l} + 16]",
        "adcx {lo}, {h0}",
        "adox {lo}, [{o} + 16]",
        "mov [{o} + 16], {lo}",
        "lea rcx, [rcx - 1]",
        "4:",
        "mov {h0}, {h1}",
        "5:",
        // Fold both carries into the top limb, which can't overflow
        // (out + c + s l < 2^(64(len + 1))). rcx is zero here.
        "adcx {h0}, rcx",
        "adox {h0}, rcx",
        inout("rdx") s => _,
        inout("rcx") len / 4 => _,
        rem = in(reg) len % 4,
        l = inout(reg) l => _,
        o = inout(reg) out => _,
        h0 = inout(reg) c => top,
        h1 = out(reg) _,
        lo = out(reg) _,
        options(nostack),
    );
    top
}

// out[..len] -= s * l[..len], returning the borrow limb owed by out[len]. adcx (CF) still
// carries between products, but OF can only add, so each product limb p is complemented
// and added with OF starting at 1: out + !p + 1 = out - p + B^len over the row, and the
// borrow is p's top limb plus one, less the OF that comes out.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn adx_sub_row(s: u64, l: *const u64, out: *mut u64, len: usize) -> u64 {
    let borrow: u64;
    asm!(
        "xor {h0:e}, {h0:e}",
        "mov {lo:e}, 0x7fffffff",
        "add {lo:e}, 1",                        // CF = 0, OF = 1
        "2:",
        "jrcxz 3f",
        "mulx {h1}, {lo}, [{l}]",
        "adcx {lo}, {h0}",
        "not {lo}",
        "adox {lo}, [{o}]",
        "mov [{o}], {lo}",
        "mulx {h0}, {lo}, [{l} + 8]",
        "adcx {lo}, {h1}",
        "not {lo}",
        "adox {lo}, [{o} + 8]",
        "mov [{o} + 8], {lo}",
        "mulx {h1}, {lo}, [{l} + 16]",
        "adcx {lo}, {h0}",
        "not {lo}",
        "adox {lo}, [{o} + 16]",
        "mov [{o} + 16], {lo}",
        "mulx {h0}, {lo}, [{l} + 24]",
        "adcx {lo}, {h1}",
        "not {lo}",
        "adox {lo}, [{o} + 24]",
        "mov [{o} + 24], {lo}",
        "lea {l}, [{l} + 32]",
        "lea {o}, [{o} + 32]",
        "lea rcx, [rcx - 1]",
        "jmp 2b",
        "3:",
        // The last len % 4 limbs, as in adx_add_row.
        "mov rcx, {rem}",
        "jrcxz 5f",
        "mulx {h1}, {lo}, [{l}]",
        "adcx {lo}, {h0}",
        "not {lo}",
        "adox {lo}, [{o}]",
        "mov [{o}], {lo}",
        "lea rcx, [rcx - 1]",
        "jrcxz 4f",
        "mulx {h0}, {lo}, [{l} + 8]",
        "adcx {lo}, {h1}",
        "not {lo}",
        "adox {lo}, [{o} + 8]",
        "mov [{o} + 8], {lo}",
        "lea rcx, [rcx - 1]",
        "jrcxz 5f",
        "mulx {h1}, {lo}, [{l} + 16]",
        "adcx {lo}, {h0}",
        "not {lo}",
        "adox {lo}, [{o} + 16]",
        "mov [{o} + 16], {lo}",
        "lea rcx, [rcx - 1]",
        "4:",
        "mov {h0}, {h1}",
        "5:",
        // p's top limb (rcx is zero here), then the borrow.
        "adcx {h0}, rcx",
        "lea {lo}, [{h0} + 1]",
        "cmovo {lo}, {h0}",
        inout("rdx") s => _,
        inout("rcx") len / 4 => _,
        rem = in(reg) len % 4,
        l = inout(reg) l => _,
        o = inout(reg) out => _,
        h0 = out(reg) _,
        h1 = out(reg) _,
        lo = out(reg) borrow,
        options(nostack),
    );
    borrow
}

// out[..len] = s * l[..len] + c, returning the top limb. out may be l itself:
// each limb is read before it is stored.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn adx_set_row(s: u64, l: *const u64, out: *mut u64, len: usize, c: u64) -> u64 {
    let top: u64;
    asm!(
        "test {h0}, 0",                         // CF = 0
        // Four limbs per pass, as in add.
        "2:",
        "jrcxz 3f",
        "mulx {h1}, {lo}, [{l}]",
        "adcx {lo}, {h0}",
        "mov [{o}], {lo}",
        "mulx {h0}, {lo}, [{l} + 8]",
        "adcx {lo}, {h1}",
        "mov [{o} + 8], {lo}",
        "mulx {h1}, {lo}, [{l} + 16]",
        "adcx {lo}, {h0}",
        "mov [{o} + 16], {lo}",
        "mulx {h0}, {lo}, [{l} + 24]",
        "adcx {lo}, {h1}",
        "mov [{o} + 24], {lo}",
        "lea {l}, [{l} + 32]",
        "lea {o}, [{o} + 32]",
        "lea rcx, [rcx - 1]",
        "jmp 2b",
        "3:",
        // The last len % 4 limbs, leaving by jrcxz as rcx runs out.
        "mov rcx, {rem}",
        "jrcxz 5f",
        "mulx {h1}, {lo}, [{l}]",
        "adcx {lo}, {h0}",
        "mov [{o}], {lo}",
        "lea rcx, [rcx - 1]",
        "jrcxz 4f",
        "mulx {h0}, {lo}, [{l} + 8]",
        "adcx {lo}, {h1}",
        "mov [{o} + 8], {lo}",
        "lea rcx, [rcx - 1]",
        "jrcxz 5f",
        "mulx {h1}, {lo}, [{l} + 16]",
        "adcx {lo}, {h0}",
        "mov [{o} + 16], {lo}",
        "lea rcx, [rcx - 1]",
        "4:",
        "mov {h0}, {h1}",
        "5:",
        // Fold the carry into the top limb, which can't overflow. rcx is zero here.
        "adcx {h0}, rcx",
        inout("rdx") s => _,
        inout("rcx") len / 4 => _,
        rem = in(reg) len % 4,
        l = inout(reg) l => _,
        o = inout(reg) out => _,
        h0 = inout(reg) c => top,
        h1 = out(reg) _,
        lo = out(reg) _,
        options(nostack),
    );
    top
}

// Base: the rows every CPU of the architecture runs, AArch64 asm and plain Rust elsewhere.

// AArch64 has one carry flag, so each pass of four limbs adds the low limbs, then
// the high limbs one up with the carry in.
#[cfg(target_arch = "aarch64")]
impl Rows for Base {
    #[inline(always)]
    unsafe fn add(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        debug_assert!(out.len() >= l.len());
        let (len, l, out) = (l.len(), l.as_ptr(), out.as_mut_ptr());
        let top: u64;
        asm!(
            "mov {c}, xzr",

            "cbz {n}, 3f",
            "2:",
            "ldp {a0}, {a1}, [{l}]",
            "ldp {a2}, {a3}, [{l}, #16]",
            "ldp {w0}, {w1}, [{o}]",
            "ldp {w2}, {w3}, [{o}, #16]",
            "umulh {h0}, {a0}, {s}",
            "mul {a0}, {a0}, {s}",
            "umulh {h1}, {a1}, {s}",
            "mul {a1}, {a1}, {s}",
            "umulh {h2}, {a2}, {s}",
            "mul {a2}, {a2}, {s}",
            "umulh {h3}, {a3}, {s}",
            "mul {a3}, {a3}, {s}",
            "adds {w0}, {w0}, {a0}",
            "adcs {w1}, {w1}, {a1}",
            "adcs {w2}, {w2}, {a2}",
            "adcs {w3}, {w3}, {a3}",
            "cset {t}, cs",
            "adds {w0}, {w0}, {c}",
            "adcs {w1}, {w1}, {h0}",
            "adcs {w2}, {w2}, {h1}",
            "adcs {w3}, {w3}, {h2}",
            // No carry out: out + c + s * l[0..4] < 2^320.
            "adc {c}, {h3}, {t}",
            "stp {w0}, {w1}, [{o}]",
            "stp {w2}, {w3}, [{o}, #16]",
            "add {l}, {l}, #32",
            "add {o}, {o}, #32",
            "sub {n}, {n}, #1",
            "cbnz {n}, 2b",
            "3:",

            "cbz {rem}, 5f",
            "4:",
            "ldr {a0}, [{l}], #8",
            "ldr {w0}, [{o}]",
            "umulh {h0}, {a0}, {s}",
            "mul {a0}, {a0}, {s}",
            "adds {w0}, {w0}, {a0}",
            "adc {h0}, {h0}, xzr",                  // h0 <= 2^64 - 2, so no carry out
            "adds {w0}, {w0}, {c}",
            "adc {c}, {h0}, xzr",
            "str {w0}, [{o}], #8",
            "sub {rem}, {rem}, #1",
            "cbnz {rem}, 4b",
            "5:",
            s = in(reg) s,
            l = inout(reg) l => _,
            o = inout(reg) out => _,
            n = inout(reg) len / 4 => _,
            rem = inout(reg) len % 4 => _,
            c = out(reg) top,
            t = out(reg) _,
            a0 = out(reg) _,
            a1 = out(reg) _,
            a2 = out(reg) _,
            a3 = out(reg) _,
            w0 = out(reg) _,
            w1 = out(reg) _,
            w2 = out(reg) _,
            w3 = out(reg) _,
            h0 = out(reg) _,
            h1 = out(reg) _,
            h2 = out(reg) _,
            h3 = out(reg) _,
            options(nostack),
        );
        top
    }

    #[inline(always)]
    unsafe fn set(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        debug_assert!(out.len() >= l.len());
        let (len, l, out) = (l.len(), l.as_ptr(), out.as_mut_ptr());
        let top: u64;
        asm!(
            "mov {c}, xzr",

            "cbz {n}, 3f",
            "2:",
            "ldp {a0}, {a1}, [{l}]",
            "ldp {a2}, {a3}, [{l}, #16]",
            "umulh {h0}, {a0}, {s}",
            "mul {a0}, {a0}, {s}",
            "umulh {h1}, {a1}, {s}",
            "mul {a1}, {a1}, {s}",
            "umulh {h2}, {a2}, {s}",
            "mul {a2}, {a2}, {s}",
            "umulh {h3}, {a3}, {s}",
            "mul {a3}, {a3}, {s}",
            "adds {a0}, {a0}, {c}",
            "adcs {a1}, {a1}, {h0}",
            "adcs {a2}, {a2}, {h1}",
            "adcs {a3}, {a3}, {h2}",
            // No carry out: c + s * l[0..4] < 2^320.
            "adc {c}, {h3}, xzr",
            "stp {a0}, {a1}, [{o}]",
            "stp {a2}, {a3}, [{o}, #16]",
            "add {l}, {l}, #32",
            "add {o}, {o}, #32",
            "sub {n}, {n}, #1",
            "cbnz {n}, 2b",
            "3:",

            "cbz {rem}, 5f",
            "4:",
            "ldr {a0}, [{l}], #8",
            "umulh {h0}, {a0}, {s}",
            "mul {a0}, {a0}, {s}",
            "adds {a0}, {a0}, {c}",
            "adc {c}, {h0}, xzr",                   // h0 <= 2^64 - 2, so no carry out
            "str {a0}, [{o}], #8",
            "sub {rem}, {rem}, #1",
            "cbnz {rem}, 4b",
            "5:",
            s = in(reg) s,
            l = inout(reg) l => _,
            o = inout(reg) out => _,
            n = inout(reg) len / 4 => _,
            rem = inout(reg) len % 4 => _,
            c = out(reg) top,
            a0 = out(reg) _,
            a1 = out(reg) _,
            a2 = out(reg) _,
            a3 = out(reg) _,
            h0 = out(reg) _,
            h1 = out(reg) _,
            h2 = out(reg) _,
            h3 = out(reg) _,
            options(nostack),
        );
        top
    }

    // One limb a pass, as utils' sub_mul loop was: the product's carry limb and the
    // subtraction's borrow share the one flag, so each limb adds the carry in, then
    // subtracts and counts its borrow into the next carry.
    #[inline(always)]
    unsafe fn sub(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        debug_assert!(out.len() >= l.len());
        if l.is_empty() {
            return 0;
        }
        let (len, l, out) = (l.len(), l.as_ptr(), out.as_mut_ptr());
        let borrow: u64;
        asm!(
            "mov {mc}, xzr",
            "2:",
            "ldr {w}, [{dst}]",
            "ldr {sv}, [{src}], #8",
            "mul   {lo}, {sv}, {d}",
            "umulh {hi}, {sv}, {d}",
            "adds {lo}, {lo}, {mc}",
            "adc  {mc}, {hi}, xzr",
            "subs {w}, {w}, {lo}",
            "cinc {mc}, {mc}, cc",
            "str {w}, [{dst}], #8",
            "sub {len}, {len}, #1",
            "cbnz {len}, 2b",
            dst = inout(reg) out => _,
            src = inout(reg) l => _,
            d = in(reg) s,
            len = inout(reg) len => _,
            mc = out(reg) borrow,
            w = out(reg) _,
            sv = out(reg) _,
            lo = out(reg) _,
            hi = out(reg) _,
            options(nostack),
        );
        borrow
    }
}

// x86 without ADX: add and sub thread one carry limb through rdx a limb at a time (mul
// clobbers the flags); set stays plain Rust. Against plain Rust rows, sub's loop is 4-14%
// faster at every length and add's ahead to 16 limbs, even to 64 and 6% behind from 128.
#[cfg(target_arch = "x86_64")]
impl Rows for Base {
    #[inline(always)]
    unsafe fn add(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        debug_assert!(out.len() >= l.len());
        if l.is_empty() {
            return 0;
        }
        let carry: u64;
        asm!(
            "mov {mc}, 0",
            "2:",
            "mov rax, [{src}]",
            "mul {d}",
            "add rax, {mc}",
            "adc rdx, 0",
            "add QWORD PTR [{dst}], rax",
            "adc rdx, 0",
            "mov {mc}, rdx",
            "lea {dst}, [{dst} + 8]",
            "lea {src}, [{src} + 8]",
            "dec {len}",
            "jnz 2b",
            dst = inout(reg) out.as_mut_ptr() => _,
            src = inout(reg) l.as_ptr() => _,
            len = inout(reg) l.len() => _,
            d = in(reg) s,
            mc = out(reg) carry,
            out("rax") _,
            out("rdx") _,
            options(nostack),
        );
        carry
    }

    unsafe fn set(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        plain_set_row(s, l, out)
    }

    #[inline(always)]
    unsafe fn sub(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        debug_assert!(out.len() >= l.len());
        if l.is_empty() {
            return 0;
        }
        let borrow: u64;
        asm!(
            "mov {mc}, 0",
            "2:",
            "mov rax, [{src}]",
            "mul {d}",
            "add rax, {mc}",
            "adc rdx, 0",
            "sub QWORD PTR [{dst}], rax",
            "adc rdx, 0",
            "mov {mc}, rdx",
            "lea {dst}, [{dst} + 8]",
            "lea {src}, [{src} + 8]",
            "dec {len}",
            "jnz 2b",
            dst = inout(reg) out.as_mut_ptr() => _,
            src = inout(reg) l.as_ptr() => _,
            len = inout(reg) l.len() => _,
            d = in(reg) s,
            mc = out(reg) borrow,
            out("rax") _,
            out("rdx") _,
            options(nostack),
        );
        borrow
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl Rows for Base {
    unsafe fn add(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        let mut carry = 0;
        for (o, &x) in out.iter_mut().zip(l) {
            let t = *o as u128 + s as u128 * x as u128 + carry as u128;
            *o = t as u64;
            carry = (t >> 64) as u64;
        }
        carry
    }

    unsafe fn set(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        plain_set_row(s, l, out)
    }

    // The borrow limb stays below B: s x + borrow < B^2 - B, so its high limb plus one fits.
    unsafe fn sub(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
        let mut borrow = 0;
        for (o, &x) in out.iter_mut().zip(l) {
            let t = s as u128 * x as u128 + borrow as u128;
            let (r, b) = o.overflowing_sub(t as u64);
            *o = r;
            borrow = (t >> 64) as u64 + b as u64;
        }
        borrow
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn plain_set_row(s: u64, l: &[u64], out: &mut [u64]) -> u64 {
    let mut carry = 0;
    for (o, &x) in out.iter_mut().zip(l) {
        let t = s as u128 * x as u128 + carry as u128;
        *o = t as u64;
        carry = (t >> 64) as u64;
    }
    carry
}

#[inline(always)]
fn mul_buf_rows<K: Rows>(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (a, b) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let len = a.len() + b.len() - 1;
    // The rows rely on this, so it is checked in release builds too.
    assert!(
        out.len() >= len,
        "out is not large enough for multiplication"
    );

    // Row i spans out[i..i + a.len()]; it starts once row i - 1 stores its top limb.
    let (&first, rows) = b.split_first().unwrap();
    let mut top = unsafe { K::set(first, a, out) };
    for (i, &s) in rows.iter().enumerate() {
        out[i + a.len()] = top;
        top = unsafe { K::add(s, a, &mut out[i + 1..]) };
    }
    if out.len() > len {
        out[len] = top;
        out[len + 1..].fill(0);
        return 0;
    }
    top
}

rows_dispatch!(pub fn mul_buf(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 = mul_buf_rows);

//KARATSUBA MULTIPLICATION
// Mid tier complexity overhead is still relatively small, Mid tier algorithm. It handles unbalanced inputs through a
// chunking and non-chunking version. The chunking boundary happens at 2s = l this is given to us
// by the nature of algorithm and doesn't need to be determined through testing.
// Time:
// Karatsuba: O(l^a - (l-s)^a)
// Chunking Karatsuba: O(ls^(a-1))
// where a = log_2(3) = 1.586...
// Space:
// Karatsuba: Omega(2l + 6log_2(l))
// Chunking Karatsuba: Omega(4s + 6log_2(s))

pub fn karatsuba_core(
    a: &[u64],
    b: &[u64],
    half: usize,       // (long + 1) / 2
    out: &mut [u64],   // >= long + short - 1
    cross: &mut [u64], // >= 2*half + 1
    scratch: &mut [u64],
) -> u64 {
    let (a0, a1) = a.split_at(half);
    let (b0, b1) = b.split_at(half);

    {
        let (a_sum, b_sum) = out[..2 * half + 2].split_at_mut(half + 1);
        let mut l_len = half;
        let mut s_len = half;

        a_sum[..l_len].copy_from_slice(a0);
        if add_buf(&mut a_sum[..l_len], a1) {
            a_sum[l_len] = 1;
            l_len += 1;
        }

        b_sum[..s_len].copy_from_slice(b0);
        if add_buf(&mut b_sum[..s_len], b1) {
            b_sum[s_len] = 1;
            s_len += 1;
        }

        karatsuba_mul(&a_sum[..l_len], &b_sum[..s_len], cross, scratch);
    }

    let (z0, z2) = out.split_at_mut(2 * half);
    karatsuba_mul(a0, b0, z0, scratch);
    let mut overflow = karatsuba_mul(a1, b1, z2, scratch);
    let z2_len = buf_len(z2);

    sub_buf(cross, z0);
    sub_buf(cross, &z2[..z2_len]);
    // overflow is the product limb at z2.len(), not at z2's trimmed length.
    if overflow > 0 {
        sub_prim(&mut cross[z2.len()..], overflow);
    }
    let out_cross = &mut out[half..];
    if out_cross.len() >= cross.len() {
        if add_buf(out_cross, cross) {
            overflow += 1;
        }
        return overflow;
    }

    let fit = out_cross.len();
    let carry = add_buf(out_cross, &cross[..fit]) as u64;
    let omitted = cross[fit];
    debug_assert_eq!(buf_len(&cross[fit + 1..]), 0);
    let (overflow, of1) = overflow.overflowing_add(omitted);
    let (overflow, of2) = overflow.overflowing_add(carry);
    debug_assert!(!of1 && !of2);
    overflow
}

pub fn chunking_karatsuba(
    long: &[u64],
    short: &[u64],
    out: &mut [u64],
    scratch: &mut [u64],
) -> u64 {
    out.fill(0);
    let s = short.len();
    let l = long.len();
    let half = (s + 1) / 2;
    let chunks = (l - s + 1) / s;
    {
        let (val_cross, rest) = scratch.split_at_mut(2 * s + 2 * half + 1);
        let (cross, val) = val_cross.split_at_mut(2 * half + 1);
        for i in 0..chunks {
            let of = s * i;
            karatsuba_core(&long[of..of + s], short, half, val, cross, rest);
            add_buf(&mut out[of..], &val);
        }
    }
    let of = chunks * s;
    let last_chunk = &long[of..];
    let (val, rest) = scratch.split_at_mut(last_chunk.len() + s - 1);
    let mut overflow = karatsuba_mul(last_chunk, short, val, rest);
    if add_buf(&mut out[of..], val) {
        overflow += 1;
    }
    return overflow;
}

fn chunking_karatsuba_static<const N: usize>(
    long: &[u64],
    short: &[u64],
    out: &mut [u64],
    val: &mut [u64; N],
) -> u64 {
    out.fill(0);
    let s = short.len();
    let l = long.len();
    let chunks = (l - s + 1) / s;
    let full_len = 2 * s;
    debug_assert!(full_len <= N);
    let mut work = [0; N];
    let mut cross = [0; N];

    for i in 0..chunks {
        let offset = s * i;
        let overflow = karatsuba_nonchunk_static::<N>(
            &long[offset..offset + s],
            short,
            &mut val[..full_len],
            &mut work,
            &mut cross,
        );
        debug_assert_eq!(overflow, 0);
        let carry = add_buf(&mut out[offset..], &val[..full_len]);
        debug_assert!(!carry);
    }

    let offset = chunks * s;
    let last_chunk = &long[offset..];
    let last_len = last_chunk.len() + s - 1;
    let mut overflow = karatsuba_nonchunk_static::<N>(
        last_chunk,
        short,
        &mut val[..last_len],
        &mut work,
        &mut cross,
    );
    if add_buf(&mut out[offset..], &val[..last_len]) {
        overflow += 1;
    }
    overflow
}

fn karatsuba_nonchunk_static<const N: usize>(
    a: &[u64],
    b: &[u64],
    out: &mut [u64],
    scratch: &mut [u64; N],
    cross: &mut [u64; N],
) -> u64 {
    let scratch_sz = find_karatsuba_scratch(a.len(), b.len());
    if scratch_sz <= N {
        return karatsuba_mul(a, b, out, scratch);
    }

    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let half = (long.len() + 1) / 2;
    debug_assert!(short.len() > half);
    karatsuba_core(long, short, half, out, &mut cross[..2 * half + 1], scratch)
}

enum KDispatch {
    Prim,
    Prim2,
    School,
    Chunking,
    Recurse,
}

// is_school's boundary for any cutoffs. For s > half it is the parabola
// s <= 1.125 c - (l - 1.5 c)^2 / (2 c), computed as its exact expansion 1.5 l - l^2 / (2 c): the
// old form cancelled two terms of size c, which read a large c (1e30 and up) as "never schoolbook"
// instead of 1.5 l, "always" (the random-knob test found it recursing forever). Both give the same
// boundary at every l below 2^21 for c from 3 to 100. c >= 3 keeps 3 x 3 on schoolbook, where a
// Karatsuba level's cross product is 3 x 3 again.
#[inline(always)]
pub fn is_school_at(l: usize, s: usize, chunk: usize, c: f64) -> bool {
    let half = (l + 1) / 2;
    if s <= half {
        s <= chunk
    } else {
        let l = l as f64;
        let boundary = (1.5 * l - l * l / (2.0 * c)).ceil() as usize;
        s <= boundary
    }
}

// Schoolbook or Karatsuba for l x s (l >= s) by the table `mul.kara` (read once per call or
// recursion level): Karatsuba's sub-products, its scratch sizing and the product dispatchers'
// schoolbook tier all ask this, so they agree. Its default is is_school_at(l, s, 63, 52.0).
pub fn is_school(l: usize, s: usize) -> bool {
    school_at(k::mul::kara(), l, s)
}

// is_school at an explicit table (invariant 6).
#[inline(always)]
pub fn school_at(kara: &crate::tuning::table::Table2, l: usize, s: usize) -> bool {
    kara.pick(s, l) == 0
}

fn kdispatch(l: usize, s: usize) -> KDispatch {
    if s == 1 {
        KDispatch::Prim
    } else if s == 2 {
        KDispatch::Prim2
    } else if is_school(l, s) {
        KDispatch::School
    } else if s <= (l + 1) / 2 {
        KDispatch::Chunking
    } else {
        KDispatch::Recurse
    }
}

fn karatsuba_mul(a: &[u64], b: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    match kdispatch(long.len(), short.len()) {
        KDispatch::Prim => {
            // mul_prim processes exactly the slice it's given, so out must be
            // truncated to long.len() before the call — any wider out (a
            // legitimate case: a scratch-backed buffer sized beyond the
            // strict a.len()+b.len()-1 convention) would otherwise feed
            // whatever garbage is beyond long.len() into the multiply as if
            // it were part of the multiplicand.
            out[..long.len()].copy_from_slice(long);
            let of = mul_prim(&mut out[..long.len()], short[0]);
            if out.len() > long.len() {
                out[long.len()] = of;
                out[long.len() + 1..].fill(0);
                return 0;
            }
            return of;
        }
        KDispatch::Prim2 => {
            // Same reasoning as Prim: mul_prim2 walks the full slice it's
            // given, so out must be truncated to long.len() first; its u128
            // return is the (up to) 2 legitimate overflow limbs beyond that.
            out[..long.len()].copy_from_slice(long);
            let of = mul_prim2(&mut out[..long.len()], combine_u64(short[0], short[1]));
            let (of_lo, of_hi) = (of as u64, (of >> 64) as u64);
            if out.len() > long.len() {
                out[long.len()] = of_lo;
                if out.len() > long.len() + 1 {
                    out[long.len() + 1] = of_hi;
                    out[long.len() + 2..].fill(0);
                    return 0;
                }
                return of_hi;
            }
            return of_lo;
        }
        KDispatch::School => {
            return mul_buf(long, short, out);
        }
        KDispatch::Chunking => {
            // Its buffers: a chunk's value and cross term, then the last chunk's product.
            let (l, s) = (long.len(), short.len());
            let last = l - (l - s + 1) / s * s;
            if scratch.len() < (2 * s + 2 * ((s + 1) / 2) + 1).max(last + s - 1) {
                return mul_buf(long, short, out);
            }
            return chunking_karatsuba(long, short, out, scratch);
        }
        KDispatch::Recurse => {}
    }
    let half = (long.len() + 1) / 2;
    if scratch.len() < 2 * half + 1 {
        // See find_karatsuba_scratch: only a non-monotone `mul.kara` table gets here.
        return mul_buf(long, short, out);
    }
    let (cross, rest) = scratch.split_at_mut(2 * half + 1);
    karatsuba_core(long, short, half, out, cross, rest)
}

// Scratch karatsuba_mul needs for l x s, by one largest stand-in per level (the cross product at
// half + 1 square). That holds for the boundaries dispatch has used, but not for every `mul.kara`
// table: one that makes a smaller sub-product Karatsuba and a larger one schoolbook can need more.
// karatsuba_mul therefore falls back to schoolbook wherever its scratch is too short, so any table
// is correct and such a table is only slower.
pub fn find_karatsuba_scratch(l_in: usize, s_in: usize) -> usize {
    let (l, s) = if l_in >= s_in {
        (l_in, s_in)
    } else {
        (s_in, l_in)
    };

    if s <= 2 || is_school(l, s) {
        return 0;
    }

    if s <= (l + 1) / 2 {
        let half = (s + 1) / 2;
        let phase1 = 2 * s + 2 * half + 1 + find_karatsuba_scratch(half + 1, half + 1);
        let chunks = (l - s + 1) / s;
        let last_len = l - chunks * s;
        let phase2 = (last_len + s - 1) + find_karatsuba_scratch(last_len, s);

        phase1.max(phase2)
    } else {
        let half = (l + 1) / 2;
        let inner = find_karatsuba_scratch(half + 1, half + 1)
            .max(find_karatsuba_scratch(l - half, s - half));
        2 * half + 1 + inner
    }
}

pub fn karatsuba_entry_dyn(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let mut scratch_guard = ScratchGuard::acquire();
    let scratch_sz = find_karatsuba_scratch(long.len(), short.len());
    karatsuba_mul(long, short, out, scratch_guard.get(scratch_sz))
}

pub fn karatsuba_entry_static<const N: usize>(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let scratch_sz = find_karatsuba_scratch(long.len(), short.len());
    let mut scratch = [0; N];
    if scratch_sz <= N {
        karatsuba_mul(long, short, out, &mut scratch)
    } else {
        match kdispatch(long.len(), short.len()) {
            KDispatch::Chunking => chunking_karatsuba_static::<N>(long, short, out, &mut scratch),
            KDispatch::Recurse => {
                let mut cross = [0; N];
                let half = (long.len() + 1) / 2;
                karatsuba_core(
                    long,
                    short,
                    half,
                    out,
                    &mut cross[..2 * half + 1],
                    &mut scratch,
                )
            }
            _ => karatsuba_mul(long, short, out, &mut scratch),
        }
    }
}

//FFT MULTIPLICATION
// Low tier complexity overhead is very high, High tier algorithm. An unbalanced product takes one
// transform of l + s digits. Chunking l into pieces of s only pays from about l = F_0 s^2, F_0 > 4
// (2l log(2s) against l log(l)), and nearly every such shape passes the NTT cutoff first, so there
// is no chunking version.
// Time: O((l+s)log(l+s))
// Space:
//  Complex Space: O(l+s)

// recombine's and sqr_recombine's twiddles: e^(2 pi i k / n) for k in 0..=n/4.
pub(crate) fn fft_twidles(n: usize) -> Box<[Complex<f64>]> {
    let th = 2.0 * PI / n as f64;
    (0..=n / 4).map(|k| Complex::cis(th * k as f64)).collect()
}

struct FFTCache {
    planner: FftPlanner<f64>,
    // One exact table per transform size n, so a size's table doesn't depend on the sizes the
    // thread saw before it, and a call reads it straight.
    twidles: HashMap<usize, Box<[Complex<f64>]>>,
}

impl FFTCache {
    fn new() -> Self {
        FFTCache {
            planner: FftPlanner::new(),
            twidles: HashMap::new(),
        }
    }

    fn prep_mul(
        &mut self,
        n: usize,
    ) -> (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>, &[Complex<f64>], usize) {
        let fwd = self.planner.plan_fft_forward(n);
        let bwd = self.planner.plan_fft_inverse(n / 2);
        let tw = self.twidles.entry(n).or_insert_with(|| fft_twidles(n));
        let scratch_sz = n + fwd
            .get_inplace_scratch_len()
            .max(bwd.get_inplace_scratch_len());
        (fwd, bwd, tw, 2 * scratch_sz)
    }

    fn prep_sqr(
        &mut self,
        m: usize,
    ) -> (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>, &[Complex<f64>], usize) {
        let fwd = self.planner.plan_fft_forward(m);
        let bwd = self.planner.plan_fft_inverse(m);
        let tw = self
            .twidles
            .entry(2 * m)
            .or_insert_with(|| fft_twidles(2 * m));
        let scratch_sz = m + fwd
            .get_inplace_scratch_len()
            .max(bwd.get_inplace_scratch_len());
        (fwd, bwd, tw, 2 * scratch_sz)
    }
}

thread_local! {
    static FFT_CACHE: RefCell<FFTCache> = RefCell::new(FFTCache::new());
}

// Transform sizes are 4 * o * 2^a for these odd parts. rustfft sends only 2^k and 3 * 2^k through
// its radix-4 kernel, and anything else through a mixed-radix split with two transposes, so a size
// costs more per point the more odd factors it has: about 1.2x for 5, 7, 9 and 15, 1.3-1.5x for
// 21, 25, 27, 35, 45, 49 and 63, and up to 1.9x past that (NEON). Those sizes lose to a slightly
// longer cheap one; these five average within 1% of the fastest size at or above each length. The
// measurements are in docs/fft_precision.md. AVX plans its factors differently: remeasure on x86.
const FFT_ODD_PARTS: [usize; 5] = [1, 3, 5, 7, 9];

/// The FFT's per-point cost for each odd part of FFT_ODD_PARTS, relative to a power of two: a
/// fixed constant per architecture, not a knob (docs/tuning_system.md, "Fixed constants"). AArch64:
/// the NEON medians of docs/fft_precision.md. x86-64 is unmeasured (rustfft's AVX planner chains
/// its radix steps differently), and other targets have no measurement: 1.0 throughout.
#[cfg(target_arch = "aarch64")]
pub const FFT_ODD_COST: [f64; 5] = [1.00, 1.00, 1.21, 1.18, 1.21];
#[cfg(not(target_arch = "aarch64"))]
pub const FFT_ODD_COST: [f64; 5] = [1.0; 5];

/// One real transform's work at size n (an FFT size): rho(o) n log2 n, for n's odd part o. The
/// FFT entries' work functions count it at the size they really use.
pub fn fft_transform_work(n: usize) -> f64 {
    let odd = n >> n.trailing_zeros();
    let rho = match FFT_ODD_PARTS.iter().position(|&o| o == odd) {
        Some(i) => FFT_ODD_COST[i],
        None => 1.0,
    };
    rho * n as f64 * (n as f64).log2()
}

/// fft_entry's transform size for an l x s product, and fft_sqr_entry's for an n-limb square.
pub fn fft_mul_size(l: usize, s: usize) -> usize {
    find_fft_size(4 * l + 4 * s - 1)
}
pub fn fft_sqr_size(n: usize) -> usize {
    find_fft_size(8 * n - 1)
}
/// fft_mid_mul's transform size for a middle product whose short operand has n limbs (operands
/// with full top limbs): 2n + 1 limbs of digits, so nothing wraps onto the band it reads.
pub fn fft_mid_size(n: usize) -> usize {
    find_fft_size(4 * (2 * n + 1))
}

/// The smallest transform size at or above `n` digits: 4 * o * 2^a with o in {1, 3, 5, 7, 9}.
pub const fn find_fft_size(n: usize) -> usize {
    let mut best = usize::MAX;
    let mut i = 0;
    while i < FFT_ODD_PARTS.len() {
        let mut size = 4 * FFT_ODD_PARTS[i];
        while size < n {
            size *= 2;
        }
        if size < best {
            best = size;
        }
        i += 1;
    }
    best
}

/// One FFT call's rounding: `err` is the largest distance from a scaled coefficient to the integer
/// it rounded to, and `fold_failed` whether the fold check ran and failed. The checked entries keep
/// a result only while `fold_failed` is false; the raw ones return it for the probe and the tests.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FftRounding {
    pub err: f64,
    pub fold_failed: bool,
}

// Each scale_and_round divides the inverse transform's coefficients by its length, rounds them to
// the nearest integer and returns the largest distance it rounded across.
// Coefficients are finite (bounded integers plus rounding error), so the maximum needs no NaN
// handling. A power-of-two length (POW2) multiplies by 1 / scale
// instead: that reciprocal is exact, so the result is bit-identical to the division's. Any other
// length divides, as an inexact reciprocal would move the distance the fold check's trigger reads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_and_round_x86<const POW2: bool>(x: &mut [Complex<f64>], scale: f64) -> f64 {
    let coefs = std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2);
    let factor = _mm256_set1_pd(if POW2 { 1.0 / scale } else { scale });
    let sign = _mm256_set1_pd(-0.0);
    // Two accumulators, so the maximum's loop-carried chain doesn't hold up the loop.
    let (mut err0, mut err1) = (_mm256_setzero_pd(), _mm256_setzero_pd());
    const ROUND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let chunks = coefs.len() / 8;
    let ptr = coefs.as_mut_ptr();
    for i in 0..chunks {
        let offset = i * 8;
        // Load 8 f64s, scale and round
        let (v0, v1) = (
            _mm256_loadu_pd(ptr.add(offset)),
            _mm256_loadu_pd(ptr.add(offset + 4)),
        );
        let (s0, s1) = if POW2 {
            (_mm256_mul_pd(v0, factor), _mm256_mul_pd(v1, factor))
        } else {
            (_mm256_div_pd(v0, factor), _mm256_div_pd(v1, factor))
        };
        let r0 = _mm256_round_pd::<ROUND>(s0);
        let r1 = _mm256_round_pd::<ROUND>(s1);
        // |scaled - rounded|: clear the sign bits
        err0 = _mm256_max_pd(err0, _mm256_andnot_pd(sign, _mm256_sub_pd(s0, r0)));
        err1 = _mm256_max_pd(err1, _mm256_andnot_pd(sign, _mm256_sub_pd(s1, r1)));
        _mm256_storeu_pd(ptr.add(offset), r0);
        _mm256_storeu_pd(ptr.add(offset + 4), r1);
    }
    let mut errs = [0.0f64; 4];
    _mm256_storeu_pd(errs.as_mut_ptr(), _mm256_max_pd(err0, err1));
    let tail = scale_and_round_slice(&mut coefs[chunks * 8..], scale);
    errs.iter().fold(tail, |m, &err| m.max(err))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_and_round_aarch<const POW2: bool>(x: &mut [Complex<f64>], scale: f64) -> f64 {
    let coefs = std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2);
    let factor = vdupq_n_f64(if POW2 { 1.0 / scale } else { scale });
    // Two accumulators, so the maximum's loop-carried chain doesn't hold up the loop.
    let (mut err0, mut err1) = (vdupq_n_f64(0.0), vdupq_n_f64(0.0));
    let chunks = coefs.len() / 4;
    let ptr = coefs.as_mut_ptr();
    for i in 0..chunks {
        let offset = i * 4;
        // Load 4 f64s, scale and round
        let (v0, v1) = (vld1q_f64(ptr.add(offset)), vld1q_f64(ptr.add(offset + 2)));
        let (s0, s1) = if POW2 {
            (vmulq_f64(v0, factor), vmulq_f64(v1, factor))
        } else {
            (vdivq_f64(v0, factor), vdivq_f64(v1, factor))
        };
        let r0 = vrndnq_f64(s0);
        let r1 = vrndnq_f64(s1);
        // |scaled - rounded|
        err0 = vmaxq_f64(err0, vabdq_f64(s0, r0));
        err1 = vmaxq_f64(err1, vabdq_f64(s1, r1));
        vst1q_f64(ptr.add(offset), r0);
        vst1q_f64(ptr.add(offset + 2), r1);
    }
    let tail = scale_and_round_slice(&mut coefs[chunks * 4..], scale);
    tail.max(vmaxvq_f64(vmaxq_f64(err0, err1)))
}

fn scale_and_round_slice(coefs: &mut [f64], scale: f64) -> f64 {
    let mut err = 0f64;
    for c in coefs {
        let scaled = *c / scale;
        *c = scaled.round_ties_even();
        err = err.max((scaled - *c).abs());
    }
    err
}

pub fn scale_and_round_standard(x: &mut [Complex<f64>], scale: f64) -> f64 {
    let coefs = unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2) };
    scale_and_round_slice(coefs, scale)
}

pub(crate) fn scale_and_round(x: &mut [Complex<f64>], scale: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                if (scale as usize).is_power_of_two() {
                    scale_and_round_x86::<true>(x, scale)
                } else {
                    scale_and_round_x86::<false>(x, scale)
                }
            };
        }
        return scale_and_round_standard(x, scale);
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe {
            if (scale as usize).is_power_of_two() {
                scale_and_round_aarch::<true>(x, scale)
            } else {
                scale_and_round_aarch::<false>(x, scale)
            }
        };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return scale_and_round_standard(x, scale);
    }
}

#[inline(always)]
fn separate_mul(x: Complex<f64>, y: Complex<f64>) -> Complex<f64> {
    let ar = x.re + y.re;
    let ai = x.im - y.im;
    let br = x.im + y.im;
    let bi = y.re - x.re;
    Complex::new(ar * br - ai * bi, ar * bi + ai * br)
}

// The pairs k..m/2 of recombine, one at a time.
#[inline]
fn recombine_pairs(x: &mut [Complex<f64>], m: usize, n: usize, tw: &[Complex<f64>], from: usize) {
    for k in from..m / 2 {
        let pk = separate_mul(x[k], x[n - k]);
        let pj = separate_mul(x[m + k], x[m - k]);
        let s = pk + pj;
        let d = pk - pj;
        let dw = d * tw[k];
        x[k] = Complex::new(s.re - dw.im, s.im + dw.re).scale(0.25);
        x[m - k] = Complex::new(s.re + dw.im, dw.re - s.im).scale(0.25);
    }
}

// Two consecutive complex values at p, split into a real and an imaginary register, and back. The
// _rev forms take p and p - 1, for the descending half of a conjugate pair.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_complex2(p: *const f64) -> (float64x2_t, float64x2_t) {
    let (a, b) = (vld1q_f64(p), vld1q_f64(p.add(2)));
    (vtrn1q_f64(a, b), vtrn2q_f64(a, b))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_complex2_rev(p: *const f64) -> (float64x2_t, float64x2_t) {
    let (a, b) = (vld1q_f64(p), vld1q_f64(p.sub(2)));
    (vtrn1q_f64(a, b), vtrn2q_f64(a, b))
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_complex2(p: *mut f64, re: float64x2_t, im: float64x2_t) {
    vst1q_f64(p, vtrn1q_f64(re, im));
    vst1q_f64(p.add(2), vtrn2q_f64(re, im));
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_complex2_rev(p: *mut f64, re: float64x2_t, im: float64x2_t) {
    vst1q_f64(p, vtrn1q_f64(re, im));
    vst1q_f64(p.sub(2), vtrn2q_f64(re, im));
}

// recombine_pairs two pairs at a time, returning the first k it left. Fused products, so its
// rounding differs slightly from the scalar loop's.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn recombine_pairs_aarch(
    x: &mut [Complex<f64>],
    m: usize,
    n: usize,
    tw: &[Complex<f64>],
) -> usize {
    debug_assert!(tw.len() >= m / 2 && x.len() >= n);
    let p = x.as_mut_ptr() as *mut f64;
    let t = tw.as_ptr() as *const f64;
    let quarter = vdupq_n_f64(0.25);
    let mut k = 1;
    while k + 1 < m / 2 {
        let (xr, xi) = load_complex2(p.add(2 * k));
        let (yr, yi) = load_complex2_rev(p.add(2 * (n - k)));
        let (ur, ui) = load_complex2(p.add(2 * (m + k)));
        let (vr, vi) = load_complex2_rev(p.add(2 * (m - k)));
        let (wr, wi) = load_complex2(t.add(2 * k));
        // separate_mul(x[k], x[n - k])
        let (ar, ai) = (vaddq_f64(xr, yr), vsubq_f64(xi, yi));
        let (br, bi) = (vaddq_f64(xi, yi), vsubq_f64(yr, xr));
        let pkr = vfmsq_f64(vmulq_f64(ar, br), ai, bi);
        let pki = vfmaq_f64(vmulq_f64(ar, bi), ai, br);
        // separate_mul(x[m + k], x[m - k])
        let (ar, ai) = (vaddq_f64(ur, vr), vsubq_f64(ui, vi));
        let (br, bi) = (vaddq_f64(ui, vi), vsubq_f64(vr, ur));
        let pjr = vfmsq_f64(vmulq_f64(ar, br), ai, bi);
        let pji = vfmaq_f64(vmulq_f64(ar, bi), ai, br);
        let (sr, si) = (vaddq_f64(pkr, pjr), vaddq_f64(pki, pji));
        let (dr, di) = (vsubq_f64(pkr, pjr), vsubq_f64(pki, pji));
        let dwr = vfmsq_f64(vmulq_f64(dr, wr), di, wi);
        let dwi = vfmaq_f64(vmulq_f64(dr, wi), di, wr);
        store_complex2(
            p.add(2 * k),
            vmulq_f64(vsubq_f64(sr, dwi), quarter),
            vmulq_f64(vaddq_f64(si, dwr), quarter),
        );
        store_complex2_rev(
            p.add(2 * (m - k)),
            vmulq_f64(vaddq_f64(sr, dwi), quarter),
            vmulq_f64(vsubq_f64(dwr, si), quarter),
        );
        k += 2;
    }
    k
}

// The two terms with no conjugate pair: no pair reads or writes them.
#[inline(always)]
fn recombine_ends(x: &mut [Complex<f64>], m: usize, n: usize) {
    let pk = x[0].re * x[0].im;
    let pj = x[m].re * x[m].im;
    x[0] = Complex::new(pk + pj, pk - pj);
    let k = m / 2;
    x[k] = separate_mul(x[k], x[n - k]).conj().scale(0.5);
}

pub(crate) fn recombine(x: &mut [Complex<f64>], m: usize, n: usize, tw: &[Complex<f64>]) {
    recombine_ends(x, m, n);
    #[cfg(target_arch = "aarch64")]
    let from = unsafe { recombine_pairs_aarch(x, m, n, tw) };
    #[cfg(not(target_arch = "aarch64"))]
    let from = 1;
    recombine_pairs(x, m, n, tw, from);
}

// recombine without NEON, for the tests.
#[cfg(test)]
pub(crate) fn recombine_standard(x: &mut [Complex<f64>], m: usize, n: usize, tw: &[Complex<f64>]) {
    recombine_ends(x, m, n);
    recombine_pairs(x, m, n, tw, 1);
}

fn fft_core(
    x: &mut [Complex<f64>],
    fwd: &dyn Fft<f64>,
    bwd: &dyn Fft<f64>,
    tw: &[Complex<f64>],
    fft_scratch: &mut [Complex<f64>],
) -> f64 {
    let n = x.len();
    fwd.process_with_scratch(x, fft_scratch);
    recombine(x, n / 2, n, tw);
    bwd.process_with_scratch(&mut x[..n / 2], fft_scratch);
    scale_and_round(&mut x[..n / 2], n as f64)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decompose_16_x86(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) -> (u64, u64) {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;
    let (mut c_l, mut c_s) = (0, 0);

    for (&l, &s) in long.iter().zip(short) {
        // Broadcast into both 64-bit lanes of a 128-bit register
        // [i16_0, i16_1, i16_2, i16_3] → [i32_0, i32_1, i32_2, i32_3]
        // [i32_0, i32_1, i32_2, i32_3] → [f64_0, f64_1, f64_2, f64_3]
        let l_broadcast = _mm_set1_epi64x(balance_limb(l, &mut c_l) as i64);
        let l_as_i32s = _mm_cvtepi16_epi32(l_broadcast);
        let l_f64s = _mm256_cvtepi32_pd(l_as_i32s);
        let s_broadcast = _mm_set1_epi64x(balance_limb(s, &mut c_s) as i64);
        let s_as_i32s = _mm_cvtepi16_epi32(s_broadcast);
        let s_f64s = _mm256_cvtepi32_pd(s_as_i32s);
        // result: [l0, s0, l2, s2]
        let lo = _mm256_unpacklo_pd(l_f64s, s_f64s);
        // result: [l1, s1, l3, s3]
        let hi = _mm256_unpackhi_pd(l_f64s, s_f64s);
        // result: [l0, s0, l1, s1]
        let first_four = _mm256_permute2f128_pd(lo, hi, 0x20);
        // result: [l2, s2, l3, s3]
        let second_four = _mm256_permute2f128_pd(lo, hi, 0x31);
        // Store 8 f64s = 4 Complex<f64> values
        _mm256_storeu_pd(out.add(idx), first_four);
        _mm256_storeu_pd(out.add(idx + 4), second_four);
        idx += 8;
    }
    let zero = _mm256_setzero_pd();
    for &l in &long[short.len()..] {
        let l_broadcast = _mm_set1_epi64x(balance_limb(l, &mut c_l) as i64);
        let l_as_i32s = _mm_cvtepi16_epi32(l_broadcast);
        let l_f64s = _mm256_cvtepi32_pd(l_as_i32s);
        // Interleave with zeros: same logic but s_f64s is replaced by zero
        let lo = _mm256_unpacklo_pd(l_f64s, zero);
        let hi = _mm256_unpackhi_pd(l_f64s, zero);
        let first_four = _mm256_permute2f128_pd(lo, hi, 0x20);
        let second_four = _mm256_permute2f128_pd(lo, hi, 0x31);
        _mm256_storeu_pd(out.add(idx), first_four);
        _mm256_storeu_pd(out.add(idx + 4), second_four);
        idx += 8;
    }

    // Zero any remaining output
    let total = x.len() * 2;
    if idx < total {
        std::ptr::write_bytes(out.add(idx), 0, total - idx);
    }
    (c_l, c_s)
}

// Balanced digits: a limb's four 16-bit digits as signed values in [-2^15, 2^15), and the carry
// into the next limb. x + bias + carry, with each digit's top bit then flipped, is y_i - 2^15 digit
// by digit, so the digits sum to x with no carry between the lanes. Mixed signs make a coefficient's
// digit products cancel, and an unsigned operand's huge zero-frequency term goes away with them.
const FFT_DIGIT_BIAS: u64 = 0x8000_8000_8000_8000;

#[inline(always)]
fn balance_limb(x: u64, carry: &mut u64) -> u64 {
    let (y, c1) = x.overflowing_add(FFT_DIGIT_BIAS);
    let (y, c2) = y.overflowing_add(*carry);
    *carry = (c1 | c2) as u64;
    y ^ FFT_DIGIT_BIAS
}

// An operand's top digit stays unsigned, so balancing adds no digit and no transform grows: a carry
// out of it (out of the top limb, or into the zero digit above it) goes back in as +2^16, leaving
// that digit at most 2^16. limbs may carry zero top limbs, as a middle product's operands can.
// `coefs` holds the operand's digits at offset + stride * index.
fn fft_absorb_top_carry(
    coefs: &mut [f64],
    offset: usize,
    stride: usize,
    limbs: &[u64],
    carry: u64,
) {
    let len16 = bit16_length(limbs.len(), limbs[limbs.len() - 1].max(1));
    let carried = if len16 == 4 * limbs.len() {
        carry != 0
    } else {
        std::mem::take(&mut coefs[offset + stride * len16]) != 0.0
    };
    if carried {
        coefs[offset + stride * (len16 - 1)] += 65536.0;
    }
}

// Digit i of a balanced limb.
#[inline(always)]
fn balanced_digit(limb: u64, i: usize) -> f64 {
    (limb >> (16 * i)) as u16 as i16 as f64
}

//converts a balanced limb to f64 in signed 16 bit chunks
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn u64_into_2x2f64s_aarch(x: u64) -> (float64x2_t, float64x2_t) {
    // Load into a 64-bit NEON register
    let x_raw = vdup_n_u64(x);
    // Reinterpret the same 64 bits as 4 × i16
    let x_i16x4 = vreinterpret_s16_u64(x_raw);
    // Sign Extend: 4 x i16 -> 4 x i32
    let x_i32x4 = vmovl_s16(x_i16x4);
    // Split: 4 x i32 -> 2 x i32 + 2 x i32
    let (x_i32_lo, x_i32_hi) = (vget_low_s32(x_i32x4), vget_high_s32(x_i32x4));
    // Sign Extend: 2 x i32 -> 2 x i64
    let x_i64_lo = vmovl_s32(x_i32_lo);
    let x_i64_hi = vmovl_s32(x_i32_hi);
    // Float Cast: 2 x i64 -> 2 x f64
    let x_f0 = vcvtq_f64_s64(x_i64_lo);
    let x_f1 = vcvtq_f64_s64(x_i64_hi);
    (x_f0, x_f1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decompose_16_aarch(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) -> (u64, u64) {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;
    let (mut c_l, mut c_s) = (0, 0);

    for (&l, &s) in long.iter().zip(short) {
        let (l_f0, l_f1) = u64_into_2x2f64s_aarch(balance_limb(l, &mut c_l));
        let (s_f0, s_f1) = u64_into_2x2f64s_aarch(balance_limb(s, &mut c_s));
        // [l0, l1] × [s0, s1] → [l0, s0]
        let pair0 = vzip1q_f64(l_f0, s_f0);
        // [l0, l1] × [s0, s1] → [l1, s1]
        let pair1 = vzip2q_f64(l_f0, s_f0);
        // [l2, l3] × [s2, s3] → [l2, s2]
        let pair2 = vzip1q_f64(l_f1, s_f1);
        // [l2, l3] × [s2, s3] → [l3, s3]
        let pair3 = vzip2q_f64(l_f1, s_f1);
        // Store 8 f64s = 4 Complex<f64> values
        vst1q_f64(out.add(idx), pair0);
        vst1q_f64(out.add(idx + 2), pair1);
        vst1q_f64(out.add(idx + 4), pair2);
        vst1q_f64(out.add(idx + 6), pair3);
        idx += 8;
    }

    let zero = vdupq_n_f64(0.0);
    for &l in &long[short.len()..] {
        let (l_f0, l_f1) = u64_into_2x2f64s_aarch(balance_limb(l, &mut c_l));
        // zip with zero register for im = 0.0
        vst1q_f64(out.add(idx), vzip1q_f64(l_f0, zero));
        vst1q_f64(out.add(idx + 2), vzip2q_f64(l_f0, zero));
        vst1q_f64(out.add(idx + 4), vzip1q_f64(l_f1, zero));
        vst1q_f64(out.add(idx + 6), vzip2q_f64(l_f1, zero));
        idx += 8;
    }

    let total = x.len() * 2;
    if idx < total {
        std::ptr::write_bytes(out.add(idx), 0, total - idx);
    }
    (c_l, c_s)
}

pub fn decompose_16_standard(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) -> (u64, u64) {
    let mut idx = 0;
    let (mut c_l, mut c_s) = (0, 0);
    for (&l, &s) in long.iter().zip(short) {
        let (l, s) = (balance_limb(l, &mut c_l), balance_limb(s, &mut c_s));
        for i in 0..4 {
            x[idx + i] = Complex::new(balanced_digit(l, i), balanced_digit(s, i));
        }
        idx += 4;
    }
    for &l in &long[short.len()..] {
        let l = balance_limb(l, &mut c_l);
        for i in 0..4 {
            x[idx + i] = Complex::new(balanced_digit(l, i), 0.0);
        }
        idx += 4;
    }
    if idx < x.len() {
        x[idx..].fill(Complex::zero());
    }
    (c_l, c_s)
}

// long + i short in balanced digits, with each operand's top digit unsigned.
pub(crate) fn decompose(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
    #[cfg(target_arch = "x86_64")]
    let (c_l, c_s) = if is_x86_feature_detected!("avx2") {
        unsafe { decompose_16_x86(long, short, x) }
    } else {
        decompose_16_standard(long, short, x)
    };

    #[cfg(target_arch = "aarch64")]
    let (c_l, c_s) = unsafe { decompose_16_aarch(long, short, x) };

    let coefs = unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2) };
    fft_absorb_top_carry(coefs, 0, 2, long, c_l);
    fft_absorb_top_carry(coefs, 1, 2, short, c_s);
}

// One limb's four 16-bit coefficients, which are signed integers of magnitude below 2^53:
// c0 + c1 2^16 + c2 2^32 + c3 2^48, added in two halves rather than shifted digit by digit.
#[inline(always)]
fn fft_limb_value(c: &[f64]) -> i128 {
    let [c0, c1, c2, c3] = *c else {
        unreachable!("chunks_exact(4)")
    };
    let lo = c0 as i64 as i128 + ((c1 as i64 as i128) << 16);
    let hi = c2 as i64 as i128 + ((c3 as i64 as i128) << 16);
    lo + (hi << 32)
}

#[inline(always)]
fn fft_accumulate_limb(chunks: &mut std::slice::ChunksExact<'_, f64>, carry: &mut i128) -> u64 {
    let total = *carry + fft_limb_value(chunks.next().unwrap());
    *carry = total >> 64;
    total as u64
}

// Writes out and returns the signed value above it. Every coefficient past that value's four is
// zero, as a product has at most 4 digits per limb, so the padding is left unread.
pub(crate) fn fft_accumulate_signed(x: &[Complex<f64>], out: &mut [u64]) -> i128 {
    let coefs = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len() * 2) };
    let mut chunks = coefs.chunks_exact(4);
    let mut carry = 0i128;

    for elem in out.iter_mut() {
        *elem = fft_accumulate_limb(&mut chunks, &mut carry);
    }
    if let Some(chunk) = chunks.next() {
        carry += fft_limb_value(chunk);
    }
    carry
}

// A product's limbs and the limb above them, which a nonnegative product can't make negative.
pub(crate) fn fft_accumulate(x: &[Complex<f64>], out: &mut [u64]) -> u64 {
    fft_accumulate_signed(x, out) as u64
}

#[inline(always)]
fn bit16_length(sz: usize, last: u64) -> usize {
    (sz - 1) * 4 + (last.ilog2() as usize) / 16 + 1
}

// THE FOLD CHECK AND ITS TRIGGER
//
// Every 16-bit FFT entry measures its call's rounding (scale_and_round) before it reads the
// coefficients: the largest distance between a coefficient and the integer it rounds to. One that
// reads above FFT_FOLD_TRIGGER runs the fold check (fft_fold_agrees), which tests the result itself,
// and is redone on the NTT only if that fails. A wrong limb needs some coefficient off by more than
// 0.5, which reads 1 - e, so a wrong result can only skip the check with one off by at least 0.75
// while every other reads within 0.25. Balanced digits keep the largest possible coefficient below
// 2^48 at every dispatch limit (asserted in mod.rs), so a coefficient's error is read in steps of at
// most 1/16; those limits are therefore soundness bounds as well as speed choices. FFT products and
// squares then check their result by its residues mod the NTT primes (always in debug builds, in
// release with FFT_RESIDUE_CHECK) and redo a mismatch too. The *_raw entries redo nothing, for the
// precision probe and the tests, and return the call's FftRounding beside the result.

static FFT_REROUTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FFT_RESIDUE_REROUTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FFT_FOLD_CHECKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FFT_FOLD_REROUTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Whether fft_entry and fft_sqr_entry check their results by residues.
const FFT_RESIDUES: bool = cfg!(debug_assertions) || FFT_RESIDUE_CHECK;

/// How many FFT calls the checked entries have redone on the NTT in this process, for any reason
/// (a diagnostic).
#[doc(hidden)]
pub fn fft_reroutes() -> u64 {
    FFT_REROUTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// How many of [`fft_reroutes`] passed the fold check or never ran it and failed only the residue
/// check: wrong results the fold check and its trigger missed. It should stay 0.
#[doc(hidden)]
pub fn fft_residue_reroutes() -> u64 {
    FFT_RESIDUE_REROUTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// How many fold checks the checked entries have run in this process (a diagnostic).
#[doc(hidden)]
pub fn fft_fold_checks() -> u64 {
    FFT_FOLD_CHECKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// How many of [`fft_fold_checks`] failed, each redone on the NTT.
#[doc(hidden)]
pub fn fft_fold_reroutes() -> u64 {
    FFT_FOLD_REROUTES.load(std::sync::atomic::Ordering::Relaxed)
}

// The largest transform each dispatcher reaches. A checked entry can only vouch for sizes up to these
// (see the limits in mod.rs), so past one it goes straight to the NTT.
const FFT_MUL_MAX_SIZE: usize = find_fft_size(4 * (FFT_16BIT_CUTOFF + 1));
const FFT_SQR_MAX_SIZE: usize = find_fft_size(8 * FFT_SQR_16BIT_CUTOFF);
const FFT_MID_MAX_SIZE: usize = find_fft_size(8 * FFT_MID_16BIT_CUTOFF + 4);
const FFT_BNM1_MAX_SIZE: usize = 4 * FFT_BNM1_16BIT_CUTOFF;

// THE FOLD CHECK
//
// The transform computes V = a b mod 2^(16 N) - 1, whatever its digits are, where V sums the N
// rounded coefficients at their digit positions. So V = a b must also hold mod any p dividing
// 2^(16 N) - 1, which is what lets a middle product and a cyclic square check themselves: neither
// keeps a whole product. A wrong V passes only if its coefficient errors, weighted by 2^(16 j) mod p,
// sum to zero. docs/fft_precision.md has how these primes were chosen.
//
// One prime per wrapped transform size, from an exhaustive search, each at least 40 bits with the
// longest repeat of those weights up to N/16 digits. Not longer: a smooth rounding wave leaves wrong
// coefficients at its evenly spaced crests, and copies s apart sum to x^t (x^N - 1) / (x^s - 1) = 0
// unless the repeat divides s, so primes repeating at N, N/2 or N/4 pass 9-28% of real wrong results.
// Sizes outside this table have no prime (see fft_fold_prime).
pub(crate) const FFT_FOLD_WRAPPED: &[(usize, u64)] = &[
    (65536, 2327042503868417),
    (73728, 2327042503868417),
    (81920, 2327042503868417),
    (98304, 2327042503868417),
    (114688, 2327042503868417),
    (131072, 2327042503868417),
    (147456, 1094462078977),
    (163840, 2327042503868417),
    (196608, 2327042503868417),
    (229376, 2327042503868417),
    (262144, 31065037602817),
    (294912, 31065037602817),
    (327680, 1753340313601),
    (393216, 31065037602817),
    (458752, 31065037602817),
    (524288, 31065037602817),
];

// Products and squares don't wrap, so any prime holds for them. This is the largest safe prime below
// the kernel's bound, so 2 has order (p - 1) / 2: no weights repeat or negate inside any transform.
pub(crate) const FFT_FOLD_PLAIN: u64 = 3689348814741904247;

// a mod p, four limbs a step. Only the accumulator chains between steps, so the products can run
// in parallel: 0.76 ns a limb on the M1 against div_prim's 3.27. p below 2^64 / 5 keeps the sum of
// five products and a limb inside 128 bits.
struct FftFold {
    p: u64,
    b: [u64; 6],
}

impl FftFold {
    fn new(p: u64) -> Self {
        debug_assert!((p as u128) * 5 < 1u128 << 64);
        let limb = ((1u128 << 64) % p as u128) as u64;
        let mut b = [1u64; 6];
        for k in 1..6 {
            b[k] = fold_mul(b[k - 1], limb, p);
        }
        FftFold { p, b }
    }

    fn rem(&self, x: &[u64]) -> u64 {
        let b = &self.b;
        let wide = |a: u64, c: u64| a as u128 * c as u128;
        let chunks = x.chunks_exact(4);
        let top = chunks.remainder().iter().zip(b);
        let mut acc: u128 = top.map(|(&l, &w)| wide(l, w)).sum();
        for c in chunks.rev() {
            acc = c[0] as u128
                + wide(c[1], b[1])
                + wide(c[2], b[2])
                + wide(c[3], b[3])
                + wide(acc as u64, b[4])
                + wide((acc >> 64) as u64, b[5]);
        }
        (acc % self.p as u128) as u64
    }
}

#[inline]
fn fold_mul(a: u64, b: u64, p: u64) -> u64 {
    (a as u128 * b as u128 % p as u128) as u64
}

fn fold_pow(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1 % p;
    while e > 0 {
        if e & 1 == 1 {
            r = fold_mul(r, a, p);
        }
        a = fold_mul(a, a, p);
        e >>= 1;
    }
    r
}

// The prime for a transform of n digits. A wrapped size may have none: those are too small for a
// reading to reach the trigger on any operand measured, and fall back to the NTT if one does.
fn fft_fold_prime(n: usize, wrapped: bool) -> Option<u64> {
    if !wrapped {
        return Some(FFT_FOLD_PLAIN);
    }
    FFT_FOLD_WRAPPED
        .iter()
        .find(|&&(size, _)| size == n)
        .map(|&(_, p)| p)
}

// Whether the rounded coefficients x (all n of them) agree with a b mod the prime for this
// transform, or false when a wrapped size has no prime. The operands are the unbalanced limbs: the
// digits only change the coefficients, not the value V has.
pub(crate) fn fft_fold_agrees(x: &[Complex<f64>], a: &[u64], b: &[u64], wrapped: bool) -> bool {
    let n = x.len() * 2;
    let Some(p) = fft_fold_prime(n, wrapped) else {
        return false;
    };
    let fold = FftFold::new(p);
    let mut guard = ScratchGuard::acquire();
    let v = guard.get(n / 4);
    let top = fft_accumulate_signed(x, v);
    // What lies above n / 4 limbs weighs 2^(16 n), which is 1 for a wrapped prime.
    let weight = fold_pow(fold.b[1], (n / 4) as u64, p);
    let top = top.rem_euclid(p as i128) as u64;
    let got = add_mod(fold.rem(v), fold_mul(top, weight, p), p);
    let ra = fold.rem(a);
    let rb = if std::ptr::eq(a, b) { ra } else { fold.rem(b) };
    fold_mul(ra, rb, p) == got
}

// The call's rounding, running the fold check if it read above the trigger, in a raw entry too, so
// the probe and the tests can see its verdict. NaN runs it as well, though finite inputs never make
// one.
#[inline(always)]
fn fft_fold_judge(
    err: f64,
    checked: bool,
    x: &[Complex<f64>],
    a: &[u64],
    b: &[u64],
    wrapped: bool,
) -> FftRounding {
    let mut rounding = FftRounding {
        err,
        fold_failed: false,
    };
    if !(err <= FFT_FOLD_TRIGGER) {
        rounding.fold_failed = !fft_fold_agrees(x, a, b, wrapped);
        if checked {
            FFT_FOLD_CHECKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if rounding.fold_failed {
                FFT_FOLD_REROUTES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    rounding
}

// long x short mod 2^(16 n) - 1, and `read`'s view of its n rounded coefficients. A checked call that
// fails the fold check doesn't reach `read`, so out stays untouched and the caller redoes it.
#[inline(always)]
fn fft_mul_coefs<R>(
    long: &[u64],
    short: &[u64],
    n: usize,
    wrapped: bool,
    checked: bool,
    read: impl FnOnce(&[Complex<f64>]) -> R,
) -> (Option<R>, FftRounding) {
    FFT_CACHE.with(|cell| {
        let fft_cache = &mut *cell.borrow_mut();
        let (fwd, bwd, tw, scratch_sz) = fft_cache.prep_mul(n);
        let mut scratch_gaurd = ScratchGuard::acquire();
        let complex_scratch: &mut [Complex<f64>] =
            unsafe { std::mem::transmute(scratch_gaurd.get(scratch_sz)) };
        let (x, fft_scratch) = complex_scratch.split_at_mut(n);

        decompose(long, short, x);
        let err = fft_core(x, fwd.as_ref(), bwd.as_ref(), tw, fft_scratch);
        let x = &x[..n / 2];
        let rounding = fft_fold_judge(err, checked, x, long, short, wrapped);
        if checked && rounding.fold_failed {
            return (None, rounding);
        }
        (Some(read(x)), rounding)
    })
}

// a^2 mod 2^(16 n) - 1, with n = 2 m, as fft_mul_coefs.
#[inline(always)]
fn fft_sqr_coefs<R>(
    a: &[u64],
    m: usize,
    wrapped: bool,
    checked: bool,
    read: impl FnOnce(&[Complex<f64>]) -> R,
) -> (Option<R>, FftRounding) {
    FFT_CACHE.with(|cell| {
        let fft_cache = &mut *cell.borrow_mut();
        let (fwd, bwd, tw, scratch_sz) = fft_cache.prep_sqr(m);
        let mut scratch_gaurd = ScratchGuard::acquire();
        let complex_scratch: &mut [Complex<f64>] =
            unsafe { std::mem::transmute(scratch_gaurd.get(scratch_sz)) };
        let (x, fft_scratch) = complex_scratch.split_at_mut(m);

        sqr_decompose(a, x);
        let err = fft_sqr_core(x, fwd.as_ref(), bwd.as_ref(), tw, fft_scratch);
        let rounding = fft_fold_judge(err, checked, x, a, a, wrapped);
        if checked && rounding.fold_failed {
            return (None, rounding);
        }
        (Some(read(x)), rounding)
    })
}

#[inline]
fn note_reroute() {
    FFT_REROUTES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn note_residue_reroute() {
    FFT_RESIDUE_REROUTES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// long × short into `out` by the 16-bit FFT. The call is redone on the NTT ([`ntt_entry_dyn`])
/// when its rounding fails the fold check, and, in debug builds or
/// with `FFT_RESIDUE_CHECK`, when its result fails the residue check ([`crate::utils::verify`]).
pub fn fft_entry(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    fft_entry_checked(long, short, out, true)
}

// fft_entry with the fold check optional: the tests turn it off to reach the residue check.
pub(crate) fn fft_entry_checked(
    long: &[u64],
    short: &[u64],
    out: &mut [u64],
    fold: bool,
) -> u64 {
    if let Some(overflow) = fft_entry_impl(long, short, out, fold).0 {
        if !FFT_RESIDUES
            || Residues::of_with_top(out, overflow) == Residues::of(long).mul(Residues::of(short))
        {
            return overflow;
        }
        note_residue_reroute();
    }
    note_reroute();
    ntt_entry_dyn(&long[..buf_len(long)], &short[..buf_len(short)], out)
}

/// [`fft_entry`] without any check: the FFT's own result, which may be wrong past the dispatch
/// limit, and the call's rounding. For the precision probe and the tests.
#[doc(hidden)]
pub fn fft_entry_raw(long: &[u64], short: &[u64], out: &mut [u64]) -> (u64, FftRounding) {
    let (overflow, rounding) = fft_entry_impl(long, short, out, false);
    (overflow.expect("the raw entry never trips"), rounding)
}

// The FFT product, or None (out untouched) when `checked` and the call's rounding trips.
fn fft_entry_impl(
    long: &[u64],
    short: &[u64],
    out: &mut [u64],
    checked: bool,
) -> (Option<u64>, FftRounding) {
    let l_len = buf_len(long);
    let s_len = buf_len(short);
    if l_len == 0 || s_len == 0 {
        out.fill(0);
        return (Some(0), FftRounding::default());
    }

    let long = &long[..l_len];
    let short = &short[..s_len];
    let (long, short) = if long.len() >= short.len() {
        (long, short)
    } else {
        (short, long)
    };
    let out_len = long.len() + short.len() - 1;
    debug_assert!(
        out.len() >= out_len,
        "out is not large enough for multiplication"
    );
    let (out_core, out_tail) = out.split_at_mut(out_len);
    let (l_len16, s_len16) = (
        bit16_length(long.len(), long[long.len() - 1]),
        bit16_length(short.len(), short[short.len() - 1]),
    );
    let n = find_fft_size(l_len16 + s_len16 - 1);
    if checked && n > FFT_MUL_MAX_SIZE {
        return (None, FftRounding::default());
    }
    let (overflow, rounding) =
        fft_mul_coefs(long, short, n, false, checked, |x| fft_accumulate(x, out_core));
    (
        overflow.map(|overflow| absorb_overflow(out_tail, overflow)),
        rounding,
    )
}

//NTT MULTIPLICATION

const NTT_RADIX_CNT: usize = 3;

const NTT_RADIX: [usize; NTT_RADIX_CNT] = [2, 3, 5];

pub(crate) const fn exp_cnt(mut p: u64) -> [usize; NTT_RADIX_CNT] {
    let mut exp = [0; NTT_RADIX_CNT];
    let mut idx = 0;
    while idx < NTT_RADIX_CNT {
        let q = NTT_RADIX[idx] as u64;
        while p % q == 0 {
            exp[idx] += 1;
            p /= q;
        }
        idx += 1;
    }
    return exp;
}

pub(crate) const fn neg_inv_mod_2_64(p: u64) -> u64 {
    let mut x: u64 = 1;
    while x.wrapping_mul(p) != u64::MAX {
        let mut tmp = x.wrapping_mul(p);
        tmp = tmp.wrapping_add(2);
        x = x.wrapping_mul(tmp);
    }
    return x;
}

pub(crate) const fn split_mul(a: u64, b: u64) -> (u64, u64) {
    const MASK: u64 = (1 << 32) - 1;
    let a0 = a & MASK;
    let a1 = a >> 32;
    let b0 = b & MASK;
    let b1 = b >> 32;
    let mut r11 = a1 * b1;
    let mut r00 = a0 * b0;
    let r10 = a1 * b0;
    let r01 = a0 * b1;
    let (mid_sum, c0) = r10.overflowing_add(r01);
    r11 += mid_sum >> 32;
    if c0 {
        r11 += 1 << 32;
    }
    let (lo_sum, c1) = r00.overflowing_add(mid_sum << 32);
    r00 = lo_sum;
    if c1 {
        r11 += 1;
    }
    (r11, r00)
}

pub(crate) const fn mod_mul(a: u64, b: u64, p: u64) -> u64 {
    const MASK: u64 = (1 << 32) - 1;
    let d = p << 1;
    let (mut hi, mut lo) = split_mul(a, b);
    hi = (hi << 1) | (lo >> 63);
    lo <<= 1;
    let n1 = lo >> 32;
    let n0 = lo & MASK;
    let d1 = d >> 32;
    let d0 = d & MASK;

    let mut q = hi / d1;
    let mut r = hi % d1;
    let mut c0 = q * d0;
    let mut c1 = (r << 32) + n1;
    if c0 > c1 {
        q -= 1;
        if d < (c0 - c1) {
            q -= 1;
        }
    }
    q &= MASK;

    let rem = (hi << 32).wrapping_add(n1).wrapping_sub(q.wrapping_mul(d));
    q = rem / d1;
    r = rem % d1;
    c0 = q * d0;
    c1 = (r << 32) + n0;
    if c0 > c1 {
        q -= 1;
        if d < (c0 - c1) {
            q -= 1;
        }
    }
    q &= MASK;

    (rem << 32).wrapping_add(n0).wrapping_sub(q.wrapping_mul(d)) >> 1
}

pub(crate) trait NTTPrime: Send + Sync {
    const P: u64;
    const EXP: [usize; NTT_RADIX_CNT] = exp_cnt(Self::P - 1);
    const R: u64 = (u64::MAX % Self::P) + 1;
    const P_INV: u64 = neg_inv_mod_2_64(Self::P);
    const R_SQR: u64 = mod_mul(Self::R, Self::R, Self::P);
    const R_CUB: u64 = mod_mul(Self::R, Self::R_SQR, Self::P);
    const P_LO: u64 = Self::P & 0xFFFFFFFF;
    const P_HI: u64 = Self::P >> 32;
}

// MIN_EXP = [46, 3, 2]
pub(crate) struct P1;
impl NTTPrime for P1 {
    const P: u64 = 5937362789990400001;
    // 2^46 * 3^3 * 5^5 + 1
}

pub(crate) struct P2;
impl NTTPrime for P2 {
    const P: u64 = 8122312296706867201;
    // 19 * 2^46 * 3^5 * 5^2 + 1
}

pub(crate) struct P3;
impl NTTPrime for P3 {
    const P: u64 = 7552325468867788801;
    // 53 * 2^46 * 3^4 * 5^2 + 1
}

#[repr(transparent)]
#[derive(Debug)]
pub(crate) struct Montgomery<P: NTTPrime> {
    val: u64,
    _phantom: PhantomData<P>,
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn reduce_x86(t: u128, p: u64, p_inv: u64) -> u64 {
    let t_lo = t as u64;
    let t_hi = (t >> 64) as u64;

    let val: u64;

    core::arch::asm!(
        // m = t_lo * p_inv mod 2^64
        "mov {m}, {t_lo}",
        "imul {m}, {p_inv}",

        // rdx:rax = m * p
        "mov rax, {m}",
        "mul {p}",

        // rdx = high(m*p) + t_hi + carry(low(m*p) + t_lo)
        "add rax, {t_lo}",
        "adc rdx, {t_hi}",

        // conditional subtract p
        "mov {tmp}, rdx",
        "sub {tmp}, {p}",
        "cmovnc rdx, {tmp}",

        m = out(reg) _,
        tmp = out(reg) _,

        t_lo = in(reg) t_lo,
        t_hi = in(reg) t_hi,
        p = in(reg) p,
        p_inv = in(reg) p_inv,

        out("rax") _,
        out("rdx") val,

        options(nostack, nomem, pure),
    );

    val
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reduce_aarch(t: u128, p: u64, p_inv: u64) -> u64 {
    let t_lo = t as u64;
    let t_hi = (t >> 64) as u64;

    let val: u64;

    core::arch::asm!(
        // m = t_lo * p_inv mod 2^64
        "mul {m}, {t_lo}, {p_inv}",

        // mp_lo = low64(m * p)
        // mp_hi = high64(m * p)
        "mul {mp_lo}, {m}, {p}",
        "umulh {mp_hi}, {m}, {p}",

        // val = t_hi + mp_hi + carry(t_lo + mp_lo)
        "adds {mp_lo}, {mp_lo}, {t_lo}",
        "adc {val}, {mp_hi}, {t_hi}",

        // conditional subtract p
        "subs {tmp}, {val}, {p}",
        "csel {val}, {tmp}, {val}, hs",

        m = out(reg) _,
        mp_lo = out(reg) _,
        mp_hi = out(reg) _,
        tmp = out(reg) _,

        val = out(reg) val,

        t_lo = in(reg) t_lo,
        t_hi = in(reg) t_hi,
        p = in(reg) p,
        p_inv = in(reg) p_inv,

        options(nostack, nomem, pure),
    );

    val
}

#[inline(always)]
pub fn reduce_no_asm(t: u128, p: u64, p_inv: u64) -> u64 {
    let t_lo = t as u64;
    let m = t_lo.wrapping_mul(p_inv);
    let mp = (m as u128) * (p as u128);

    let mut val = ((t + mp) >> 64) as u64;

    if val >= p {
        val -= p;
    }

    val
}

#[inline]
pub(crate) fn reduce_mont(t: u128, p: u64, p_inv: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return reduce_x86(t, p, p_inv);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return reduce_aarch(t, p, p_inv);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return reduce_no_asm(t, p, p_inv);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_reduce_x86(a: u64, b: u64, p: u64, p_inv: u64) -> u64 {
    let val: u64;

    core::arch::asm!(
        // rdx:rax = a * b
        "mul {b}",

        // Save t_lo and t_hi
        "mov {t_lo}, rax",
        "mov {t_hi}, rdx",

        // m = t_lo * p_inv mod 2^64
        "mov {m}, rax",
        "imul {m}, {p_inv}",

        // rdx:rax = m * p
        "mov rax, {m}",
        "mul {p}",

        // val = t_hi + high(m*p) + carry(t_lo + low(m*p))
        "add rax, {t_lo}",
        "adc rdx, {t_hi}",

        // Conditional subtract p
        "mov {tmp}, rdx",
        "sub {tmp}, {p}",
        "cmovnc rdx, {tmp}",

        t_lo = out(reg) _,
        t_hi = out(reg) _,
        m = out(reg) _,
        tmp = out(reg) _,

        inout("rax") a => _,
        b = in(reg) b,
        p = in(reg) p,
        p_inv = in(reg) p_inv,

        out("rdx") val,

        options(nostack, nomem, pure),
    );

    val
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_reduce_aarch(a: u64, b: u64, p: u64, p_inv: u64) -> u64 {
    let val: u64;

    core::arch::asm!(
        // t = a * b
        "mul {t_lo}, {a}, {b}",
        "umulh {t_hi}, {a}, {b}",

        // m = t_lo * p_inv mod 2^64
        "mul {m}, {t_lo}, {p_inv}",

        // mp = m * p
        "mul {mp_lo}, {m}, {p}",
        "umulh {mp_hi}, {m}, {p}",

        // val = t_hi + mp_hi + carry(t_lo + mp_lo)
        "adds {mp_lo}, {mp_lo}, {t_lo}",
        "adc {val}, {mp_hi}, {t_hi}",

        // if val >= p, val -= p
        "subs {tmp}, {val}, {p}",
        "csel {val}, {tmp}, {val}, hs",

        t_lo = out(reg) _,
        t_hi = out(reg) _,
        m = out(reg) _,
        mp_lo = out(reg) _,
        mp_hi = out(reg) _,
        tmp = out(reg) _,

        val = out(reg) val,

        a = in(reg) a,
        b = in(reg) b,
        p = in(reg) p,
        p_inv = in(reg) p_inv,

        options(nostack, nomem, pure),
    );

    val
}

#[inline(always)]
pub fn mul_reduce_no_asm(a: u64, b: u64, p: u64, p_inv: u64) -> u64 {
    let t = (a as u128) * (b as u128);

    let t_lo = t as u64;
    let m = t_lo.wrapping_mul(p_inv);
    let mp = (m as u128) * (p as u128);

    let mut val = ((t + mp) >> 64) as u64;

    if val >= p {
        val -= p;
    }

    val
}

#[inline]
fn mul_reduce_mont(a: u64, b: u64, p: u64, p_inv: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return mul_reduce_x86(a, b, p, p_inv);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return mul_reduce_aarch(a, b, p, p_inv);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return mul_reduce_no_asm(a, b, p, p_inv);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn add_mod_x86(mut a: u64, b: u64, p: u64) -> u64 {
    core::arch::asm!(
        "add {a}, {b}",
        "mov {tmp}, {a}",
        "sub {tmp}, {p}",
        "cmovnc {a}, {tmp}",
        a = inout(reg) a,
        b = in(reg) b,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn add_mod_aarch(mut a: u64, b: u64, p: u64) -> u64 {
    core::arch::asm!(
        "add {a}, {a}, {b}",
        "subs {tmp}, {a}, {p}",
        "csel {a}, {tmp}, {a}, hs",
        a = inout(reg) a,
        b = in(reg) b,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[inline(always)]
pub fn add_mod_no_asm(mut a: u64, b: u64, p: u64) -> u64 {
    a += b;
    if a >= p {
        a -= p;
    }
    a
}

#[inline]
fn add_mod(a: u64, b: u64, p: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        add_mod_x86(a, b, p)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        add_mod_aarch(a, b, p)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        add_mod_no_asm(a, b, p)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sub_mod_x86(mut a: u64, b: u64, p: u64) -> u64 {
    core::arch::asm!(
        "sub {a}, {b}",
        "sbb {tmp}, {tmp}",
        "and {tmp}, {p}",
        "add {a}, {tmp}",
        a = inout(reg) a,
        b = in(reg) b,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sub_mod_aarch(mut a: u64, b: u64, p: u64) -> u64 {
    core::arch::asm!(
        "subs {a}, {a}, {b}",
        "csel {tmp}, xzr, {p}, hs",
        "add {a}, {a}, {tmp}",
        a = inout(reg) a,
        b = in(reg) b,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[inline(always)]
pub fn sub_mod_no_asm(a: u64, b: u64, p: u64) -> u64 {
    let diff = a.wrapping_sub(b);
    let mask = ((a < b) as u64).wrapping_neg();
    diff.wrapping_add(p & mask)
}

#[inline(always)]
fn sub_mod(a: u64, b: u64, p: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return sub_mod_x86(a, b, p);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return sub_mod_aarch(a, b, p);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return sub_mod_no_asm(a, b, p);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn neg_mod_x86(mut a: u64, p: u64) -> u64 {
    core::arch::asm!(
        "test {a}, {a}",
        "mov {tmp}, {p}",
        "sub {tmp}, {a}",
        "cmovnz {a}, {tmp}",
        a = inout(reg) a,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neg_mod_aarch(mut a: u64, p: u64) -> u64 {
    core::arch::asm!(
        "cmp {a}, #0",
        "sub {tmp}, {p}, {a}",
        "csel {a}, xzr, {tmp}, eq",
        a = inout(reg) a,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[inline(always)]
pub fn neg_mod_no_asm(a: u64, p: u64) -> u64 {
    let mask = ((a != 0) as u64).wrapping_neg();
    p.wrapping_sub(a) & mask
}

#[inline]
fn neg_mod(a: u64, p: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return neg_mod_x86(a, p);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return neg_mod_aarch(a, p);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return neg_mod_no_asm(a, p);
    }
}

impl<P: NTTPrime> Montgomery<P> {
    const ZERO: Self = Montgomery {
        val: 0,
        _phantom: PhantomData,
    };

    const ONE: Self = Montgomery {
        val: P::R,
        _phantom: PhantomData,
    };

    const HALF: Self = Self::const_to((P::P + 1) / 2);

    const G: Self = Self::prim_root();

    const G3: Self = Self::G.const_pow((P::P - 1) / 3);

    const INV_G3: Self = Self::G3.const_pow(2);

    const G5: Self = Self::G.const_pow((P::P - 1) / 5);

    const INV_G5: Self = Self::G5.const_pow(4);

    #[inline(always)]
    const fn const_reduce(t_hi: u64, t_lo: u64) -> u64 {
        const MASK32: u64 = (1 << 32) - 1;
        let m = t_lo.wrapping_mul(P::P_INV);
        let (mp_hi, mp_lo) = split_mul(m, P::P);
        let (_, c) = mp_lo.overflowing_add(t_lo);
        let mut val = t_hi + mp_hi + c as u64;
        if val >= P::P {
            val -= P::P;
        }
        val
    }

    const fn const_to(a: u64) -> Self {
        let (t_hi, t_lo) = split_mul(a, P::R_SQR);
        Montgomery {
            val: Self::const_reduce(t_hi, t_lo),
            _phantom: PhantomData,
        }
    }

    const fn mul_mut(&mut self, rhs: Self) {
        let (t_hi, t_lo) = split_mul(self.val, rhs.val);
        self.val = Self::const_reduce(t_hi, t_lo);
    }

    const fn const_pow(self, pow: u64) -> Self {
        if pow == 0 {
            return Self::ONE;
        } else if pow == 1 {
            return self;
        }
        let mut val = self.const_pow(pow / 2);
        val.mul_mut(val);
        if pow & 1 == 1 {
            val.mul_mut(self);
        }
        return val;
    }

    const fn add_mut(&mut self, rhs: Self) {
        self.val += rhs.val;
        if self.val >= P::P {
            self.val -= P::P;
        }
    }

    const fn prim_root() -> Self {
        let mut g = Montgomery::ONE;
        let mut found = false;
        while !found {
            g.add_mut(Self::ONE);
            found = true;
            let mut idx = 0;
            while idx < NTT_RADIX_CNT && found {
                let q = NTT_RADIX[idx] as u64;
                found &= g.const_pow((P::P - 1) / q).val != P::R;
                idx += 1;
            }
        }
        return g;
    }

    const fn cast(x: u64) -> Self {
        Montgomery {
            val: x,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn to(a: u64) -> Self {
        Montgomery {
            val: mul_reduce_mont(a, P::R_SQR, P::P, P::P_INV),
            _phantom: PhantomData,
        }
    }

    fn to_mut(&mut self) {
        self.val = mul_reduce_mont(self.val, P::R_SQR, P::P, P::P_INV);
    }

    pub(crate) fn from(self) -> u64 {
        reduce_mont(self.val as u128, P::P, P::P_INV)
    }

    pub(crate) fn fuse_mul_to(a: u64, b: u64) -> Self {
        let c = mul_reduce_mont(a, b, P::P, P::P_INV);
        Montgomery {
            val: mul_reduce_mont(c, P::R_CUB, P::P, P::P_INV),
            _phantom: PhantomData,
        }
    }

    pub(crate) fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut result = Self::ONE;
        while exp > 0 {
            if exp & 1 == 1 {
                result *= base;
            }
            base *= base;
            exp >>= 1;
        }
        result
    }
}

impl<P: NTTPrime> Copy for Montgomery<P> {}

impl<P: NTTPrime> Clone for Montgomery<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: NTTPrime> PartialEq for Montgomery<P> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl<P: NTTPrime> Eq for Montgomery<P> {}

impl<P: NTTPrime> AddAssign for Montgomery<P> {
    fn add_assign(&mut self, rhs: Self) {
        self.val = add_mod(self.val, rhs.val, P::P);
    }
}

impl<P: NTTPrime> Add for Montgomery<P> {
    type Output = Montgomery<P>;
    fn add(self, rhs: Self) -> Self::Output {
        Montgomery {
            val: add_mod(self.val, rhs.val, P::P),
            _phantom: PhantomData,
        }
    }
}

impl<P: NTTPrime> SubAssign for Montgomery<P> {
    fn sub_assign(&mut self, rhs: Self) {
        self.val = sub_mod(self.val, rhs.val, P::P);
    }
}

impl<P: NTTPrime> Sub for Montgomery<P> {
    type Output = Montgomery<P>;
    fn sub(self, rhs: Self) -> Self::Output {
        Montgomery {
            val: sub_mod(self.val, rhs.val, P::P),
            _phantom: PhantomData,
        }
    }
}

impl<P: NTTPrime> Neg for Montgomery<P> {
    type Output = Montgomery<P>;
    fn neg(self) -> Self::Output {
        Montgomery {
            val: neg_mod(self.val, P::P),
            _phantom: PhantomData,
        }
    }
}

impl<P: NTTPrime> Mul for Montgomery<P> {
    type Output = Montgomery<P>;
    fn mul(self, rhs: Self) -> Self::Output {
        Montgomery {
            val: mul_reduce_mont(self.val, rhs.val, P::P, P::P_INV),
            _phantom: PhantomData,
        }
    }
}

impl<P: NTTPrime> MulAssign for Montgomery<P> {
    fn mul_assign(&mut self, rhs: Self) {
        self.val = mul_reduce_mont(self.val, rhs.val, P::P, P::P_INV);
    }
}

// A shared table: a handle cloned out of the cache survives the cache growing it.
struct DynNTTTwidles<P: NTTPrime> {
    tw: Arc<Vec<Montgomery<P>>>,
    max_n: usize,
    n: usize,
}

impl<P: NTTPrime> Clone for DynNTTTwidles<P> {
    fn clone(&self) -> Self {
        DynNTTTwidles {
            tw: Arc::clone(&self.tw),
            max_n: self.max_n,
            n: self.n,
        }
    }
}

impl<P: NTTPrime> DynNTTTwidles<P> {
    fn build(n: usize) -> Self {
        let w = Montgomery::G.pow((P::P - 1) / (n as u64));
        let mut tw = Vec::<Montgomery<P>>::with_capacity(n);
        let mut acc = Montgomery::ONE;
        for _ in 0..n {
            tw.push(acc);
            acc *= w;
        }
        DynNTTTwidles {
            tw: Arc::new(tw),
            max_n: n,
            n: n,
        }
    }

    fn ensure(&mut self, new_n: usize) {
        if new_n > self.max_n {
            debug_assert!(new_n % self.max_n == 0);
            let growth = new_n / self.max_n;
            debug_assert!(growth.is_power_of_two());
            let mut new_tw = Vec::<Montgomery<P>>::with_capacity(new_n);
            let w = Montgomery::G.pow((P::P - 1) / (new_n as u64));
            for &tw in self.tw.iter() {
                let mut acc = tw;
                new_tw.push(acc);
                for _ in 1..growth {
                    acc *= w;
                    new_tw.push(acc);
                }
            }
            self.tw = Arc::new(new_tw);
            self.max_n = new_n;
        }
        self.n = new_n;
    }
}

struct StaticNTTTwidles<const N: usize, P: NTTPrime> {
    tw: [Montgomery<P>; N],
    n: usize,
}

impl<P: NTTPrime, const N: usize> StaticNTTTwidles<N, P> {
    fn build(n: usize) -> Self {
        let w = Montgomery::G.pow((P::P - 1) / (n as u64));
        let mut tw = [Montgomery::ZERO; N];
        let mut acc = Montgomery::ONE;
        for i in 0..n {
            tw[i] = acc;
            acc *= w;
        }
        Self { tw, n }
    }
}

trait NTTDir {
    const INV: bool;
    fn advance(idx: usize, step: usize, len: usize) -> usize;
}

struct NTTForward {}

impl NTTDir for NTTForward {
    const INV: bool = false;
    fn advance(idx: usize, step: usize, len: usize) -> usize {
        add_mod(idx as u64, step as u64, len as u64) as usize
    }
}

struct NTTBackward {}

impl NTTDir for NTTBackward {
    const INV: bool = true;
    fn advance(idx: usize, step: usize, len: usize) -> usize {
        sub_mod(idx as u64, step as u64, len as u64) as usize
    }
}

struct TwiddleIter<'a, P: NTTPrime, D: NTTDir> {
    tw: &'a [Montgomery<P>],
    idx: usize,
    step: usize,
    len: usize,
    _dir: PhantomData<D>,
}

impl<'a, P: NTTPrime, D: NTTDir> Iterator for TwiddleIter<'a, P, D> {
    type Item = Montgomery<P>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        debug_assert!(self.idx < self.tw.len());
        debug_assert!(self.len <= self.tw.len());
        debug_assert!(self.step <= self.len);
        let x = unsafe { *self.tw.get_unchecked(self.idx) };
        self.idx = D::advance(self.idx, self.step, self.len);
        Some(x)
    }
}

trait NTTTwidles<P: NTTPrime>: Send + Sync {
    fn twidles_iter<'a, const R: usize, D: NTTDir>(
        &'a self,
        stride: usize,
    ) -> [TwiddleIter<'a, P, D>; R];
}

impl<const N: usize, P: NTTPrime> NTTTwidles<P> for StaticNTTTwidles<N, P> {
    fn twidles_iter<'a, const R: usize, D: NTTDir>(
        &'a self,
        stride: usize,
    ) -> [TwiddleIter<'a, P, D>; R] {
        std::array::from_fn(|r| TwiddleIter {
            tw: &self.tw,
            idx: 0,
            step: (r + 1) * stride,
            len: self.n,
            _dir: PhantomData,
        })
    }
}

impl<P: NTTPrime> NTTTwidles<P> for DynNTTTwidles<P> {
    fn twidles_iter<'a, const R: usize, D: NTTDir>(
        &'a self,
        stride: usize,
    ) -> [TwiddleIter<'a, P, D>; R] {
        let cache_stride = self.max_n / self.n;
        debug_assert_eq!(self.tw.len(), self.max_n);
        debug_assert_eq!(self.max_n % self.n, 0);
        std::array::from_fn(|r| {
            let step = (r + 1) * stride * cache_stride;
            debug_assert!(step <= self.max_n);
            TwiddleIter {
                tw: &self.tw,
                idx: 0,
                step,
                len: self.max_n,
                _dir: PhantomData,
            }
        })
    }
}

type DynNTTCache<P> = RefCell<HashMap<usize, DynNTTTwidles<P>>>;

thread_local! {
    static DYN_NTT_CACHE_P1: DynNTTCache<P1> = RefCell::new(HashMap::new());
    static DYN_NTT_CACHE_P2: DynNTTCache<P2> = RefCell::new(HashMap::new());
    static DYN_NTT_CACHE_P3: DynNTTCache<P3> = RefCell::new(HashMap::new());
}

// One prime's cyclic convolution of a and b (a squared when b is None) into res,
// whose length is the transform size. The twiddles are a handle cloned out of the
// cache, which holds no borrow of it: a thread blocked in rayon::join may run
// another convolution that needs the same cache.
fn dyn_convolution<P: NTTPrime>(
    cache: &'static LocalKey<DynNTTCache<P>>,
    a: &[u64],
    b: Option<&[u64]>,
    res: &mut [u64],
) {
    let n = res.len();
    let tw = cache.with(|cell| {
        let mut cache = cell.borrow_mut();
        let tw = cache
            .entry(n >> n.trailing_zeros())
            .or_insert_with(|| DynNTTTwidles::build(n));
        tw.ensure(n);
        tw.clone()
    });
    let mut g = ScratchGuard::acquire();
    match b {
        Some(b) => {
            let [bs, ns] = g.get_splits([n, ntt_convolution_scratch_len(n)]);
            ntt_convolution::<P>(a, b, res, &tw, bs, ns);
        }
        None => ntt_sqr_convolution::<P>(a, res, &tw, g.get(ntt_scratch_len(n))),
    }
}

// The three primes' convolutions, on rayon when par.
fn dyn_convolutions(a: &[u64], b: Option<&[u64]>, [r1, r2, r3]: [&mut [u64]; 3], par: bool) {
    let mut c1 = || dyn_convolution(&DYN_NTT_CACHE_P1, a, b, r1);
    let mut c2 = || dyn_convolution(&DYN_NTT_CACHE_P2, a, b, r2);
    let mut c3 = || dyn_convolution(&DYN_NTT_CACHE_P3, a, b, r3);
    if par {
        rayon::join(c1, || rayon::join(c2, c3));
    } else {
        c1();
        c2();
        c3();
    }
}

fn ntt<P: NTTPrime, D: NTTDir>(
    buf: &mut [Montgomery<P>],
    stride: usize,
    tw: &impl NTTTwidles<P>,
    scratch: &mut [Montgomery<P>],
) {
    debug_assert!(scratch.len() >= ntt_scratch_len(buf.len()));
    if buf.len() % 5 == 0 {
        ntt_5::<P, D>(buf, stride, tw, scratch);
    } else if buf.len() % 3 == 0 {
        ntt_3::<P, D>(buf, stride, tw, scratch);
    } else {
        ntt_2::<P, D>(buf, stride, tw, scratch);
    }
}

fn ntt_scratch_len(n: usize) -> usize {
    if n < 2 {
        return 0;
    }

    if n % 5 == 0 {
        ntt_radix_scratch_len(n, 5, k::mul::ntt_par_radix5())
    } else if n % 3 == 0 {
        ntt_radix_scratch_len(n, 3, k::mul::ntt_par_radix3())
    } else {
        0
    }
}

fn ntt_radix_scratch_len(n: usize, radix: usize, par_cutoff: usize) -> usize {
    let m = n / radix;
    let shuffle_scratch = (radix - 1) * m;
    let child_scratch = ntt_scratch_len(m);
    let recursive_scratch = if n > par_cutoff {
        radix * child_scratch
    } else {
        child_scratch
    };
    shuffle_scratch.max(recursive_scratch)
}

fn ntt_convolution_scratch_len(n: usize) -> usize {
    let ntt_scratch = ntt_scratch_len(n);
    if n > k::mul::ntt_par() {
        2 * ntt_scratch
    } else {
        ntt_scratch
    }
}

fn ntt_2<P: NTTPrime, D: NTTDir>(
    buf: &mut [Montgomery<P>],
    stride: usize,
    tw: &impl NTTTwidles<P>,
    _scratch: &mut [Montgomery<P>],
) {
    let n = buf.len();
    if n < 2 {
        return;
    }
    debug_assert!(n.is_power_of_two());

    let log_n = n.trailing_zeros();
    let shift = usize::BITS - log_n;
    for i in 1..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            buf.swap(i, j);
        }
    }

    let mut m_half = 1;
    while m_half < n {
        let m = m_half << 1;
        let stage_stride = stride * (n / m);
        let mut k = 0;
        while k < n {
            let [mut tw0] = tw.twidles_iter::<1, D>(stage_stride);
            for j in 0..m_half {
                let t = unsafe { tw0.next().unwrap_unchecked() };
                let [x0, x1] = unsafe { buf.get_disjoint_unchecked_mut([k + j, k + j + m_half]) };
                let u = *x0;
                let v = *x1 * t;
                *x0 = u + v;
                *x1 = u - v;
            }
            k += m;
        }
        m_half = m;
    }
}

fn split<const N: usize, P: NTTPrime>(buf: &mut [Montgomery<P>]) -> [&mut [Montgomery<P>]; N] {
    debug_assert!(
        buf.len() % N == 0,
        "length of buf must be evenly divided by N = {N}"
    );
    let m = buf.len() / N;
    let base = buf.as_mut_ptr();
    let mut offset = 0usize;
    std::array::from_fn(|_| unsafe {
        let s = std::slice::from_raw_parts_mut(base.add(offset), m);
        offset += m;
        s
    })
}

fn shuffle<const N: usize, P: NTTPrime>(buf: &mut [Montgomery<P>], scratch: &mut [Montgomery<P>]) {
    let m = buf.len() / N;
    debug_assert!(scratch.len() >= (N - 1) * m);
    for i in 0..m {
        buf[i] = buf[N * i];
        for j in 1..N {
            scratch[(j - 1) * m + i] = buf[N * i + j];
        }
    }
    for j in 1..N {
        let start = (j - 1) * m;
        buf[j * m..(j + 1) * m].copy_from_slice(&scratch[start..start + m]);
    }
}

fn ntt_3<P: NTTPrime, D: NTTDir>(
    buf: &mut [Montgomery<P>],
    stride: usize,
    tw: &impl NTTTwidles<P>,
    scratch: &mut [Montgomery<P>],
) {
    let n = buf.len();
    let m = buf.len() / 3;
    debug_assert!(scratch.len() >= ntt_scratch_len(n));
    shuffle::<3, P>(buf, scratch);

    let new_stride = 3 * stride;
    {
        let [b0, b1, b2] = split(buf);
        let child_scratch_len = ntt_scratch_len(m);
        if n > k::mul::ntt_par_radix3() {
            let (s0, rest) = scratch.split_at_mut(child_scratch_len);
            let (s1, rest) = rest.split_at_mut(child_scratch_len);
            let (s2, _) = rest.split_at_mut(child_scratch_len);
            let ntt0 = || ntt::<P, D>(b0, new_stride, tw, s0);
            let ntt1 = || ntt::<P, D>(b1, new_stride, tw, s1);
            let ntt2 = || ntt::<P, D>(b2, new_stride, tw, s2);
            rayon::join(ntt0, || rayon::join(ntt1, ntt2));
        } else {
            let child_scratch = &mut scratch[..child_scratch_len];
            ntt::<P, D>(b0, new_stride, tw, child_scratch);
            ntt::<P, D>(b1, new_stride, tw, child_scratch);
            ntt::<P, D>(b2, new_stride, tw, child_scratch);
        }
    }

    let g = if D::INV {
        Montgomery::INV_G3
    } else {
        Montgomery::G3
    };

    let [mut tw0, mut tw1] = tw.twidles_iter::<2, D>(stride);
    for i in 0..m {
        let [x0, x1, x2] = unsafe { buf.get_disjoint_unchecked_mut([i, i + m, i + 2 * m]) };
        let (t0, t1) = unsafe { (tw0.next().unwrap_unchecked(), tw1.next().unwrap_unchecked()) };
        let a = *x0;
        let b = *x1 * t0;
        let c = *x2 * t1;
        let diff = (b - c) * g;
        *x0 = a + b + c;
        *x1 = a - c + diff;
        *x2 = a - b - diff;
    }
}

fn ntt_5<P: NTTPrime, D: NTTDir>(
    buf: &mut [Montgomery<P>],
    stride: usize,
    tw: &impl NTTTwidles<P>,
    scratch: &mut [Montgomery<P>],
) {
    let n = buf.len();
    let m = buf.len() / 5;
    debug_assert!(scratch.len() >= ntt_scratch_len(n));
    shuffle::<5, P>(buf, scratch);

    let new_stride = 5 * stride;
    {
        let [b0, b1, b2, b3, b4] = split(buf);
        let child_scratch_len = ntt_scratch_len(m);
        if n > k::mul::ntt_par_radix5() {
            let (s0, rest) = scratch.split_at_mut(child_scratch_len);
            let (s1, rest) = rest.split_at_mut(child_scratch_len);
            let (s2, rest) = rest.split_at_mut(child_scratch_len);
            let (s3, rest) = rest.split_at_mut(child_scratch_len);
            let (s4, _) = rest.split_at_mut(child_scratch_len);
            let ntt0 = || ntt::<P, D>(b0, new_stride, tw, s0);
            let ntt1 = || ntt::<P, D>(b1, new_stride, tw, s1);
            let ntt2 = || ntt::<P, D>(b2, new_stride, tw, s2);
            let ntt3 = || ntt::<P, D>(b3, new_stride, tw, s3);
            let ntt4 = || ntt::<P, D>(b4, new_stride, tw, s4);
            rayon::join(
                || rayon::join(ntt0, ntt1),
                || rayon::join(ntt2, || rayon::join(ntt3, ntt4)),
            );
        } else {
            let child_scratch = &mut scratch[..child_scratch_len];
            ntt::<P, D>(b0, new_stride, tw, child_scratch);
            ntt::<P, D>(b1, new_stride, tw, child_scratch);
            ntt::<P, D>(b2, new_stride, tw, child_scratch);
            ntt::<P, D>(b3, new_stride, tw, child_scratch);
            ntt::<P, D>(b4, new_stride, tw, child_scratch);
        }
    }

    let g = if D::INV {
        Montgomery::INV_G5
    } else {
        Montgomery::G5
    };
    let g2 = g * g;
    let g3 = g2 * g;
    let g4 = g3 * g;
    let alpha = (g + g4) * Montgomery::HALF;
    let beta = (g2 + g3) * Montgomery::HALF;
    let gamma = (g - g4) * Montgomery::HALF;
    let delta = (g2 - g3) * Montgomery::HALF;

    let [mut tw0, mut tw1, mut tw2, mut tw3] = tw.twidles_iter::<4, D>(stride);
    for i in 0..m {
        let [x0, x1, x2, x3, x4] =
            unsafe { buf.get_disjoint_unchecked_mut([i, i + m, i + 2 * m, i + 3 * m, i + 4 * m]) };
        let (t0, t1, t2, t3) = unsafe {
            (
                tw0.next().unwrap_unchecked(),
                tw1.next().unwrap_unchecked(),
                tw2.next().unwrap_unchecked(),
                tw3.next().unwrap_unchecked(),
            )
        };
        let a = *x0;
        let b = *x1 * t0;
        let c = *x2 * t1;
        let d = *x3 * t2;
        let e = *x4 * t3;

        let sum1 = b + e;
        let dif1 = b - e;
        let sum2 = c + d;
        let dif2 = c - d;
        let as1 = alpha * sum1;
        let gd1 = gamma * dif1;
        let bs2 = beta * sum2;
        let dd2 = delta * dif2;
        let bs1 = beta * sum1;
        let dd1 = delta * dif1;
        let as2 = alpha * sum2;
        let gd2 = gamma * dif2;

        *x0 = a + sum1 + sum2;
        *x1 = a + (as1 + gd1) + (bs2 + dd2);
        *x2 = a + (bs1 + dd1) + (as2 - gd2);
        *x3 = a + (bs1 - dd1) + (as2 + gd2);
        *x4 = a + (as1 - gd1) + (bs2 - dd2);
    }
}

fn ntt_convolution<P: NTTPrime>(
    a_buf: &[u64],
    b_buf: &[u64],
    res: &mut [u64],
    tw: &impl NTTTwidles<P>,
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    let n = res.len();
    let ntt_scratch_len = ntt_scratch_len(n);
    debug_assert!(buf_scratch.len() >= n);
    debug_assert!(ntt_scratch.len() >= ntt_scratch_len);

    res[..a_buf.len()].copy_from_slice(a_buf);
    res[a_buf.len()..].fill(0);
    buf_scratch[..b_buf.len()].copy_from_slice(b_buf);
    buf_scratch[b_buf.len()..].fill(0);

    {
        let (a, b) = (&mut *res, &mut *buf_scratch);

        let ntt_a = move || {
            let mont_a: &mut [Montgomery<P>] = unsafe { std::mem::transmute(a) };
            for a in mont_a.iter_mut().take(a_buf.len()) {
                a.to_mut();
            }
            mont_a
        };

        let ntt_b = move || {
            let mont_b: &mut [Montgomery<P>] = unsafe { std::mem::transmute(b) };
            for b in mont_b.iter_mut().take(b_buf.len()) {
                b.to_mut();
            }
            mont_b
        };

        if n > k::mul::ntt_par() && ntt_scratch.len() >= 2 * ntt_scratch_len {
            let (a_ntt_scratch, b_ntt_scratch) = ntt_scratch.split_at_mut(ntt_scratch_len);
            let (mont_a, mont_b) = rayon::join(ntt_a, ntt_b);
            let mont_a_scratch: &mut [Montgomery<P>] =
                unsafe { std::mem::transmute(a_ntt_scratch) };
            let mont_b_scratch: &mut [Montgomery<P>] =
                unsafe { std::mem::transmute(&mut b_ntt_scratch[..ntt_scratch_len]) };
            rayon::join(
                || ntt::<P, NTTForward>(mont_a, 1, tw, mont_a_scratch),
                || ntt::<P, NTTForward>(mont_b, 1, tw, mont_b_scratch),
            );
        } else {
            let mont_a = ntt_a();
            let mont_ntt_scratch: &mut [Montgomery<P>] =
                unsafe { std::mem::transmute(&mut ntt_scratch[..ntt_scratch_len]) };
            ntt::<P, NTTForward>(mont_a, 1, tw, mont_ntt_scratch);

            let mont_b = ntt_b();
            let mont_ntt_scratch: &mut [Montgomery<P>] =
                unsafe { std::mem::transmute(&mut ntt_scratch[..ntt_scratch_len]) };
            ntt::<P, NTTForward>(mont_b, 1, tw, mont_ntt_scratch);
        }
    }

    let mont_a: &mut [Montgomery<P>] = unsafe { std::mem::transmute(res) };
    let mont_b: &mut [Montgomery<P>] = unsafe { std::mem::transmute(buf_scratch) };
    for (a, b) in mont_a.iter_mut().zip(mont_b.iter()) {
        *a *= *b;
    }

    ntt::<P, NTTBackward>(mont_a, 1, tw, mont_b);
}

struct CRT<P1: NTTPrime, P2: NTTPrime, P3: NTTPrime> {
    inv_p1_p2: Montgomery<P2>,
    inv_p1_p3: Montgomery<P3>,
    inv_p2_p3: Montgomery<P3>,
    p1p2_lo: u64,
    p1p2_hi: u64,
    _phantom: PhantomData<P1>,
}

impl<P1: NTTPrime, P2: NTTPrime, P3: NTTPrime> CRT<P1, P2, P3> {
    fn new() -> Self {
        let inv_p1_p2 = Montgomery::<P2>::to(P1::P).pow(P2::P - 2);
        let inv_p1_p3 = Montgomery::<P3>::to(P1::P).pow(P3::P - 2);
        let inv_p2_p3 = Montgomery::<P3>::to(P2::P).pow(P3::P - 2);
        let p1p2 = (P1::P as u128) * (P2::P as u128);
        CRT {
            inv_p1_p2,
            inv_p1_p3,
            inv_p2_p3,
            p1p2_lo: p1p2 as u64,
            p1p2_hi: (p1p2 >> 64) as u64,
            _phantom: PhantomData,
        }
    }

    fn crt(
        &self,
        r1: &mut u64,
        r2: &mut u64,
        r3: &mut u64,
        inv_n1: Montgomery<P1>,
        inv_n2: Montgomery<P2>,
        inv_n3: Montgomery<P3>,
    ) {
        let r1_p1 = Montgomery::cast(*r1) * inv_n1;
        let x1 = r1_p1.from();

        let r2_p2 = Montgomery::cast(*r2) * inv_n2;
        let x1_p2 = Montgomery::to(x1);
        let mont_x2 = (r2_p2 - x1_p2) * self.inv_p1_p2;
        let x2 = mont_x2.from();

        let r3_p3 = Montgomery::cast(*r3) * inv_n3;
        let x1_p3 = Montgomery::to(x1);
        let x2_p3 = Montgomery::to(x2);
        let mont_x3 = ((r3_p3 - x1_p3) * self.inv_p1_p3 - x2_p3) * self.inv_p2_p3;
        let x3 = mont_x3.from();

        let mask = u64::MAX as u128;

        let part1 = x1 as u128;

        let part2 = (x2 as u128) * (P1::P as u128);
        let part2_lo = part2 & mask;
        let part2_hi = part2 >> 64;

        let part3_bot = (x3 as u128) * (self.p1p2_lo as u128);
        let part3_top = (part3_bot >> 64) + (x3 as u128) * (self.p1p2_hi as u128);
        let part3_lo = part3_bot & mask;
        let part3_mid = part3_top & mask;
        let part3_hi = part3_top >> 64;

        let sum1 = part1 + part2_lo + part3_lo;
        let sum2 = (sum1 >> 64) + part2_hi + part3_mid;
        let sum3 = (sum2 >> 64) + part3_hi;

        *r1 = sum1 as u64;
        *r2 = sum2 as u64;
        *r3 = sum3 as u64;
    }
}

static CRT: LazyLock<CRT<P1, P2, P3>> = LazyLock::new(|| CRT::<P1, P2, P3>::new());

#[inline]
fn absorb_overflow(out_tail: &mut [u64], overflow: u64) -> u64 {
    if let Some((carry_limb, rest)) = out_tail.split_first_mut() {
        *carry_limb = overflow;
        rest.fill(0);
        0
    } else {
        overflow
    }
}

// n1/n2/n3 are the declared transform sizes the residues are scaled by; the
// res slices may be shorter (clamped to a static buffer) as long as they cover
// out_len.
fn ntt_accumulate(
    res1: &mut [u64],
    res2: &mut [u64],
    res3: &mut [u64],
    out: &mut [u64],
    out_len: usize,
    n1: usize,
    n2: usize,
    n3: usize,
) -> u64 {
    let inv_n1 = Montgomery::to(n1 as u64).pow(P1::P - 2);
    let inv_n2 = Montgomery::to(n2 as u64).pow(P2::P - 2);
    let inv_n3 = Montgomery::to(n3 as u64).pow(P3::P - 2);
    for i in 0..out_len {
        CRT.crt(
            &mut res1[i],
            &mut res2[i],
            &mut res3[i],
            inv_n1,
            inv_n2,
            inv_n3,
        );
    }

    let len1 = res1.len().min(out.len());
    out[..len1].copy_from_slice(&res1[..len1]);
    let len2 = res2.len().min(out.len().saturating_sub(1));
    let of1 = if len2 == 0 {
        0
    } else {
        add_buf(&mut out[1..], &res2[..len2]) as u64
    };
    let len3 = res3.len().min(out.len().saturating_sub(2));
    let of2 = if len3 == 0 {
        0
    } else {
        add_buf(&mut out[2..], &res3[..len3]) as u64
    };

    let res2_carry = out
        .len()
        .checked_sub(1)
        .and_then(|i| res2.get(i))
        .copied()
        .unwrap_or(0);
    let res3_carry = out
        .len()
        .checked_sub(2)
        .and_then(|i| res3.get(i))
        .copied()
        .unwrap_or(0);
    res2_carry + res3_carry + of1 + of2
}

fn ntt_log_prime_search(
    init: usize,
    target: usize,
    primes: &[usize],
    max_exps: &[usize],
) -> Option<usize> {
    debug_assert_eq!(primes.len(), max_exps.len());
    if init >= target {
        return Some(init);
    }

    let Some((&p, rest_primes)) = primes.split_first() else {
        return None;
    };

    let (&max_exp, rest_exps) = max_exps.split_first().unwrap();
    let mut best: Option<usize> = None;
    let mut val = init;
    for exp in 0..=max_exp {
        let cur = ntt_log_prime_search(val, target, rest_primes, rest_exps);
        best = match (best, cur) {
            (Some(b), Some(c)) => Some(b.min(c)),
            (b, c) => b.or(c),
        };
        if best == Some(target) {
            return best;
        }
        if exp < max_exp {
            match val.checked_mul(p) {
                Some(next) if best.map_or(true, |b| next < b) => val = next,
                _ => break,
            }
        }
    }

    best
}

fn find_ntt_size<P: NTTPrime>(n: usize) -> usize {
    ntt_log_prime_search(1, n, &NTT_RADIX, &P::EXP).expect("Input Is too large for NTT")
}

pub fn find_max_ntt_size(n: usize) -> usize {
    find_ntt_size::<P1>(n)
        .max(find_ntt_size::<P2>(n))
        .max(find_ntt_size::<P3>(n))
}

pub fn ntt_entry_dyn(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let out_len = a.len() + b.len() - 1;
    assert!(
        out.len() >= out_len,
        "out is not large enough for multiplication"
    );
    let n1 = find_ntt_size::<P1>(out_len);
    let n2 = find_ntt_size::<P2>(out_len);
    let n3 = find_ntt_size::<P3>(out_len);

    let mut res_guard = ScratchGuard::acquire();
    let [res1, res2, res3] = res_guard.get_splits([n1, n2, n3]);

    dyn_convolutions(
        a,
        Some(b),
        [&mut *res1, &mut *res2, &mut *res3],
        out_len > k::mul::ntt_par_conv(),
    );

    let (out_core, out_tail) = out.split_at_mut(out_len);
    let overflow = ntt_accumulate(res1, res2, res3, out_core, out_len, n1, n2, n3);
    absorb_overflow(out_tail, overflow)
}

fn ntt_log_prime_search_for_split(
    init: usize,
    target: usize,
    primes: &[usize],
    max_exps: &[usize],
) -> Option<usize> {
    debug_assert_eq!(primes.len(), max_exps.len());
    if init > target {
        return None;
    }

    let Some((&p, rest_primes)) = primes.split_first() else {
        return Some(init);
    };

    let (&max_exp, rest_exps) = max_exps.split_first().unwrap();
    let mut best: Option<usize> = None;
    let mut val = init;
    for exp in 0..=max_exp {
        let cur = ntt_log_prime_search_for_split(val, target, rest_primes, rest_exps);
        best = match (best, cur) {
            (Some(b), Some(c)) => Some(b.max(c)),
            (b, c) => b.or(c),
        };
        if best == Some(target) {
            return best;
        }
        if exp < max_exp {
            match val.checked_mul(p) {
                Some(next) if next <= target => val = next,
                _ => break,
            }
        }
    }

    best
}

pub(crate) fn find_ntt_size_for_split<const N: usize, P: NTTPrime>() -> usize {
    ntt_log_prime_search_for_split(1, N, &NTT_RADIX, &P::EXP).expect("Input Is too large for NTT")
}

pub(crate) fn acc_convolution<P: NTTPrime, F>(a_buf: &[u64], b_buf: &[u64], res: &mut [u64], op: F)
where
    F: Fn(u64, u64) -> Montgomery<P>,
{
    let mont_res: &mut [Montgomery<P>] = unsafe { std::mem::transmute(res) };
    for (i, &a) in a_buf.iter().enumerate() {
        for (j, &b) in b_buf.iter().enumerate() {
            mont_res[i + j] += op(a, b);
        }
    }
}

// Emulates a declared size-n cyclic convolution inside res (clamped to
// res.len()) by chunking long into pieces whose transforms fit N; every chunk
// is re-scaled so the accumulated values carry the size-n scale.
fn ntt_chunked_convolution<const N: usize, P: NTTPrime>(
    long: &[u64],
    short: &[u64],
    res: &mut [u64],
    n: usize,
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    let (long, short) = if long.len() >= short.len() {
        (long, short)
    } else {
        (short, long)
    };

    res.fill(0);
    let split_n = find_ntt_size_for_split::<N, P>();
    let chunk_len = split_n + 1 - short.len();
    debug_assert!(chunk_len > 0);

    let mut tmp = [0; N];
    let mut offset = 0;
    while offset < long.len() {
        let end = (offset + chunk_len).min(long.len());
        acc_scaled_cyclic_convolution::<N, P>(
            &long[offset..end],
            short,
            offset,
            n,
            res,
            &mut tmp,
            buf_scratch,
            ntt_scratch,
        );
        offset = end;
    }
}

fn ntt_static_convolution<const N: usize, P: NTTPrime>(
    long: &[u64],
    short: &[u64],
    res: &mut [u64],
    n: usize,
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    if n <= N {
        let mut tw = StaticNTTTwidles::<N, P>::build(n);
        ntt_convolution::<P>(
            long,
            short,
            &mut res[..n],
            &mut tw,
            &mut buf_scratch[..n],
            &mut ntt_scratch[..ntt_scratch_len(n)],
        );
    } else {
        ntt_chunked_convolution::<N, P>(long, short, res, n, buf_scratch, ntt_scratch);
    }
}

pub fn ntt_entry_static<const N: usize>(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    if long.is_empty() || short.is_empty() {
        return 0;
    }
    let out_len = long.len() + short.len() - 1;
    assert!(
        out_len <= N && out.len() >= out_len,
        "inputs exceed static capacity"
    );
    let mut res1 = [0; N];
    let mut res2 = [0; N];
    let mut res3 = [0; N];

    let n1 = find_ntt_size::<P1>(out_len);
    let n2 = find_ntt_size::<P2>(out_len);
    let n3 = find_ntt_size::<P3>(out_len);

    if out_len > k::mul::ntt_par_conv() {
        let conv1 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_static_convolution::<N, P1>(
                long,
                short,
                &mut res1,
                n1,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        let conv2 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_static_convolution::<N, P2>(
                long,
                short,
                &mut res2,
                n2,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        let conv3 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_static_convolution::<N, P3>(
                long,
                short,
                &mut res3,
                n3,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        rayon::join(conv1, || rayon::join(conv2, conv3));
    } else {
        let mut buf_scratch = [0; N];
        let mut ntt_scratch = [0; N];
        ntt_static_convolution::<N, P1>(
            long,
            short,
            &mut res1,
            n1,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
        ntt_static_convolution::<N, P2>(
            long,
            short,
            &mut res2,
            n2,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
        ntt_static_convolution::<N, P3>(
            long,
            short,
            &mut res3,
            n3,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
    }
    let (out_core, out_tail) = out.split_at_mut(out_len);
    let overflow = ntt_accumulate(
        &mut res1[..n1.min(N)],
        &mut res2[..n2.min(N)],
        &mut res3[..n3.min(N)],
        out_core,
        out_len,
        n1,
        n2,
        n3,
    );
    absorb_overflow(out_tail, overflow)
}

//END NTT MULTIPLICATION

pub fn fft_cost(l: usize, s: usize) -> f64 {
    let sum = (l + s) as f64;
    sum * sum.ln()
}

pub fn chunking_karatsuba_cost(l: usize, s: usize) -> f64 {
    const LOG_2_3: f64 = 1.584962501;
    let s_pow = (s as f64).powf(LOG_2_3 - 1.0);
    (l as f64) * s_pow
}

pub fn karatsuba_cost(l: usize, s: usize) -> f64 {
    const LOG_2_3: f64 = 1.584962501;
    let l_pow = (l as f64).powf(LOG_2_3);
    let sum_pow = ((l - s) as f64).powf(LOG_2_3);
    l_pow - sum_pow
}

pub fn is_karatsuba(l: usize, s: usize, chunk: f64, reg: f64) -> bool {
    let fft_cost = fft_cost(l, s);
    let half = (l + 1) / 2;
    if s <= half {
        chunking_karatsuba_cost(l, s) < chunk * fft_cost
    } else {
        karatsuba_cost(l, s) < reg * fft_cost
    }
}

// The work of an l x s product in schoolbook limb products, following mul_static's tiers, for
// the static series nodes' cost model (dynamic nodes price products by `MulEq`). Each tier's scale comes
// from the crossover that selects it, where both sides cost the same: Karatsuba's cost is
// scaled to l s at the school boundary (c^(2 - log2 3) for is_school's cutoff c), and the
// transform's is is_karatsuba's ratio times fft_cost. A two-limb short side counts four
// limb products a limb, as mul_prim2's 128-bit carries cost about four mul_prim passes.
// Only Karatsuba and above call libm.
pub fn mul_work(l: usize, s: usize) -> f64 {
    let (l, s) = if l >= s { (l, s) } else { (s, l) };
    if s <= 2 {
        return (l * s * s) as f64;
    }
    // Its own boundary, the formula its tiers are scaled at, not `mul.kara`: with the table's
    // boundary elsewhere the scaling below would jump. mul_ns replaces all of this.
    if is_school_at(l, s, k::mul::chunking_karatsuba(), k::mul::karatsuba()) {
        return l as f64 * s as f64;
    }
    const SCALE: f64 = 2.0 - 1.584962501;
    let (kara, cut, ratio) = if s <= (l + 1) / 2 {
        (chunking_karatsuba_cost(l, s), k::mul::chunking_karatsuba() as f64, k::mul::ntt_chunking_karatsuba())
    } else {
        (karatsuba_cost(l, s), k::mul::karatsuba(), k::mul::ntt_karatsuba())
    };
    cut.powf(SCALE) * kara.min(ratio * fft_cost(l, s))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulDynDispatch {
    Prim,
    Prim2,
    School,
    Karatsuba,
    FFT,
    NTT,
}

// The static product dispatch's knobs (invariants 6 and 7): the schoolbook boundary and
// is_karatsuba's ratios against the NTT, read once per call. (Dynamic products compare fitted
// equations instead: MulEq.)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MulKnobs {
    // mul.kara: the schoolbook tier.
    pub kara: &'static crate::tuning::table::Table2,
    pub chunk_ratio: f64,
    pub kara_ratio: f64,
}

impl MulKnobs {
    pub fn static_read() -> Self {
        MulKnobs {
            kara: k::mul::kara(),
            chunk_ratio: k::mul::ntt_chunking_karatsuba(),
            kara_ratio: k::mul::ntt_karatsuba(),
        }
    }
}

/// The fitted runtime equations of the dynamic product entries (docs/tuning_system.md, decision
/// 8): each algorithm's time in ns, in its theoretical form with constants fitted to its own
/// timings by `cargo tune` (knobs `mul.kara_rec_*`, `mul.chunk_rec_*`, `mul.fft_*`). Read once
/// per call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MulEq {
    /// mul_buf: scale l s + shift, as [scale, shift].
    pub school: [f64; 2],
    /// karatsuba_entry_dyn with s above half of l: scale karatsuba_cost(l, s) + school l s, as
    /// [scale, school]. The first term is Karatsuba's theoretical work, l^e - (l - s)^e at
    /// e = log2 3, and the second schoolbook's, which the recursion's leaves still do: its weight
    /// stands where a fitted exponent stood until 2026-11-11, so the equation is linear in its
    /// constants like every other.
    pub kara: [f64; 2],
    /// karatsuba_entry_dyn with s at most half of l (chunking): scale chunking_karatsuba_cost(l, s)
    /// + school l s, the first term l s^(e - 1) at e = log2 3.
    pub chunk: [f64; 2],
    /// fft_entry: ns per unit of rho(o) N log2 N at its transform size N.
    pub fft: f64,
}

// The forms above are what the timings need (the form analysis of 2026-11-11, `cargo tune tune
// --analyze`, on the M1): schoolbook's term a row, recursive Karatsuba's terms a limb and their
// shifts, and the FFT's term a point and its shift either solved negative or didn't lower the
// error on shapes held out, so the equations no longer carry them.

impl MulEq {
    pub fn read() -> Self {
        MulEq {
            school: [k::mul::school_scale(), k::mul::school_shift()],
            kara: [k::mul::kara_rec_scale(), k::mul::kara_rec_school()],
            chunk: [k::mul::chunk_rec_scale(), k::mul::chunk_rec_school()],
            fft: k::mul::fft_scale(),
        }
    }

    /// Schoolbook's predicted ns for l x s, l >= s.
    pub fn school_ns(&self, l: usize, s: usize) -> f64 {
        let [a, c] = self.school;
        a * l as f64 * s as f64 + c
    }

    /// mul_dyn's predicted ns for an l x s product, either order: the schoolbook equation where
    /// `kara` (mul.kara's table) keeps schoolbook, else the cheaper of Karatsuba's and the FFT's,
    /// as its dispatch chooses. Other algorithms' equations are built on it (division's).
    pub fn mul_ns(&self, l: usize, s: usize, kara: &crate::tuning::table::Table2) -> f64 {
        let (l, s) = if l >= s { (l, s) } else { (s, l) };
        if s == 0 {
            0.0
        } else if s <= 2 || school_at(kara, l, s) {
            self.school_ns(l, s)
        } else {
            self.kara_ns(l, s).min(self.fft_ns(l, s))
        }
    }

    /// Recursive Karatsuba's predicted ns for l x s, l >= s.
    pub fn kara_ns(&self, l: usize, s: usize) -> f64 {
        let (lf, sf) = (l as f64, s as f64);
        let ([a, q], work) = if s > (l + 1) / 2 { (self.kara, karatsuba_cost(l, s)) } else { (self.chunk, chunking_karatsuba_cost(l, s)) };
        a * work + q * lf * sf
    }

    /// The FFT's predicted ns for l x s, at the transform size it really uses.
    pub fn fft_ns(&self, l: usize, s: usize) -> f64 {
        self.fft_ns_at(fft_mul_size(l, s))
    }

    /// The FFT's predicted ns at transform size n: a product's, or a middle product's
    /// (`fft_mid_size`), which runs the same three transforms.
    pub fn fft_ns_at(&self, n: usize) -> f64 {
        self.fft * fft_transform_work(n)
    }
}

// mul_dyn's choice for l >= s. The table decides small products before the equations are read
// (constants in a normal build, eleven live values in a tune build).
pub fn dyn_dispatch(l: usize, s: usize) -> MulDynDispatch {
    match dyn_small_dispatch(l, s, k::mul::kara()) {
        Some(alg) => alg,
        None => dyn_large_dispatch(l, s, &MulEq::read()),
    }
}

// mul_dyn's choice at explicit knobs (invariant 6): `kara` is mul.kara's table.
pub fn dyn_dispatch_at(l: usize, s: usize, kara: &crate::tuning::table::Table2, eq: &MulEq) -> MulDynDispatch {
    dyn_small_dispatch(l, s, kara).unwrap_or_else(|| dyn_large_dispatch(l, s, eq))
}

// The single-limb, two-limb and schoolbook tiers, or None past the table's schoolbook boundary.
fn dyn_small_dispatch(l: usize, s: usize, kara: &crate::tuning::table::Table2) -> Option<MulDynDispatch> {
    if s == 1 {
        Some(MulDynDispatch::Prim)
    } else if s == 2 {
        Some(MulDynDispatch::Prim2)
    } else if school_at(kara, l, s) {
        Some(MulDynDispatch::School)
    } else {
        None
    }
}

// Past the schoolbook boundary: recursive Karatsuba or the transform, whichever fitted equation
// predicts less time (the user's rule for a decision with two inputs: fitted runtime equations
// compared per call). The FFT's equation is evaluated at the transform size it really uses, so
// the choice follows its size steps, which one ratio over (l, s) could not: on the M1 the
// equations lose 0.0-0.2% of measured time against the ratios' 0.2-0.7%, and 0-7% at worst
// against 8-18%. Past the FFT's limit the NTT takes the transform's place, decided by the same
// comparison (the FFT's equation at the size it would use: there is no fitted NTT equation), as
// a very unbalanced product still belongs to chunking Karatsuba there.
fn dyn_large_dispatch(l: usize, s: usize, eq: &MulEq) -> MulDynDispatch {
    if eq.kara_ns(l, s) < eq.fft_ns(l, s) {
        MulDynDispatch::Karatsuba
    } else if (l + s - 1) <= FFT_16BIT_CUTOFF {
        MulDynDispatch::FFT
    } else {
        MulDynDispatch::NTT
    }
}

pub fn mul_dyn(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let out_len = a.len() + b.len() - 1;
    debug_assert!(
        out.len() >= out_len,
        "out is not large enough for multiplication"
    );
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let out_core = &mut out[..out_len];
    let alg = dyn_dispatch(long.len(), short.len());
    let overflow = match alg {
        MulDynDispatch::Prim => {
            out_core[..long.len()].copy_from_slice(long);
            mul_prim(out_core, short[0])
        }
        MulDynDispatch::Prim2 => {
            out_core[..long.len()].copy_from_slice(long);
            out_core[long.len()] = 0;
            mul_prim2(out_core, combine_u64(short[0], short[1])) as u64
        }
        MulDynDispatch::School => mul_buf(long, short, out_core),
        MulDynDispatch::Karatsuba => karatsuba_entry_dyn(long, short, out_core),
        MulDynDispatch::FFT => fft_entry(long, short, out_core),
        MulDynDispatch::NTT => ntt_entry_dyn(long, short, out_core),
    };
    // Debug builds check every NTT product by its residues mod the NTT primes (linear); FFT
    // products check themselves in fft_entry and redo a mismatch.
    #[cfg(debug_assertions)]
    if matches!(alg, MulDynDispatch::NTT) {
        assert_eq!(
            Residues::of_with_top(out_core, overflow),
            Residues::of(long).mul(Residues::of(short)),
            "mul_dyn: the {}x{} NTT product fails its residue check",
            long.len(),
            short.len()
        );
    }
    if out.len() > out_len {
        out[out_len] = overflow;
        out[out_len + 1..].fill(0);
        0
    } else {
        overflow
    }
}

pub fn mul_vec(a: &[u64], b: &[u64]) -> (Vec<u64>, u64) {
    if a.is_empty() || b.is_empty() {
        return (Vec::new(), 0);
    }
    let mut out = vec![0; a.len() + b.len() - 1];
    let of = mul_dyn(a, b, &mut out);
    (out, of)
}

enum StaticDispatch {
    Prim,
    Prim2,
    School,
    Karatsuba,
    NTT,
}

fn static_dispatch(l: usize, s: usize) -> StaticDispatch {
    static_dispatch_at(l, s, &MulKnobs::static_read())
}

// mul_static's choice for l >= s at explicit knobs (invariant 6).
fn static_dispatch_at(l: usize, s: usize, kn: &MulKnobs) -> StaticDispatch {
    if s == 1 {
        StaticDispatch::Prim
    } else if s == 2 {
        StaticDispatch::Prim2
    } else if school_at(kn.kara, l, s) {
        StaticDispatch::School
    } else if is_karatsuba(l, s, kn.chunk_ratio, kn.kara_ratio) {
        StaticDispatch::Karatsuba
    } else {
        StaticDispatch::NTT
    }
}

pub fn mul_static<const N: usize>(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let out_len = a.len() + b.len() - 1;
    debug_assert!(
        out_len <= out.len(),
        "out is not large enough for multiplication"
    );
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let out_core = &mut out[..out_len];
    let overflow = match static_dispatch(long.len(), short.len()) {
        StaticDispatch::Prim => {
            out_core[..long.len()].copy_from_slice(long);
            mul_prim(out_core, short[0])
        }
        StaticDispatch::Prim2 => {
            out_core[..long.len()].copy_from_slice(long);
            out_core[long.len()] = 0;
            mul_prim2(out_core, combine_u64(short[0], short[1])) as u64
        }
        StaticDispatch::School => mul_buf(long, short, out_core),
        StaticDispatch::Karatsuba => karatsuba_entry_static::<N>(long, short, out_core),
        StaticDispatch::NTT => ntt_entry_static::<N>(long, short, out_core),
    };
    if out.len() > out_len {
        out[out_len] = overflow;
        out[out_len + 1..].fill(0);
        0
    } else {
        overflow
    }
}

pub fn mul_arr<const N: usize>(a: &[u64], b: &[u64]) -> ([u64; N], u64) {
    let mut out = [0; N];
    let of = mul_static::<N>(a, b, &mut out);
    (out, of)
}

//SQUARE MULTIPLICATION
// Simple hierarchy School < Karatsuba < FFT < NTT. These are optimized algorithms specifically
// for squaring buffers. They all have constant factor speed ups.
// School -> 50%
// Karatsuba -> 40% - 50%
// FTT -> 33%
// NTT -> 33%

// acc2:acc1:acc0 += 2 * s2:s1:s0.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn add_twice(s: [u64; 3], acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) {
    unsafe {
        asm!(
            "add {acc0}, {s0}",
            "adc {acc1}, {s1}",
            "adc {acc2}, {s2}",
            "add {acc0}, {s0}",
            "adc {acc1}, {s1}",
            "adc {acc2}, {s2}",
            s0 = in(reg) s[0],
            s1 = in(reg) s[1],
            s2 = in(reg) s[2],
            acc0 = inout(reg) *acc0,
            acc1 = inout(reg) *acc1,
            acc2 = inout(reg) *acc2,
            options(nostack, nomem),
        )
    };
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn add_twice(s: [u64; 3], acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) {
    unsafe {
        asm!(
            "adds {acc0}, {acc0}, {s0}",
            "adcs {acc1}, {acc1}, {s1}",
            "adc {acc2}, {acc2}, {s2}",
            "adds {acc0}, {acc0}, {s0}",
            "adcs {acc1}, {acc1}, {s1}",
            "adc {acc2}, {acc2}, {s2}",
            s0 = in(reg) s[0],
            s1 = in(reg) s[1],
            s2 = in(reg) s[2],
            acc0 = inout(reg) *acc0,
            acc1 = inout(reg) *acc1,
            acc2 = inout(reg) *acc2,
            options(nostack, nomem),
        )
    };
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
fn add_twice(s: [u64; 3], acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) {
    let lo = (*acc1 as u128) << 64 | *acc0 as u128;
    let (lo, carry) = lo.overflowing_add(((s[1] as u128) << 64 | s[0] as u128) << 1);
    *acc0 = lo as u64;
    *acc1 = (lo >> 64) as u64;
    *acc2 += (s[2] << 1 | s[1] >> 63) + carry as u64;
}

fn sqr_elem(buf: &[u64], n: usize, acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) -> u64 {
    // Off-diagonal products once, doubled, then the diagonal square (even n). The
    // column stays below n B^2 + B^2 < B^3.
    let mi = n.saturating_sub(buf.len() - 1);
    let mf = ((n + 1) / 2).min(buf.len());
    if mi < mf {
        let (mut s0, mut s1, mut s2) = (0, 0, 0);
        unsafe {
            let p = buf.as_ptr();
            mul_col(p.add(mi), p.add(n - mi), mf - mi, &mut s0, &mut s1, &mut s2);
        }
        add_twice([s0, s1, s2], acc0, acc1, acc2);
    }
    if n % 2 == 0 && n / 2 < buf.len() {
        unsafe {
            let d = *buf.get_unchecked(n / 2);
            mul_asm(d, d, acc0, acc1, acc2);
        }
    }
    let out = *acc0;
    *acc0 = *acc1;
    *acc1 = *acc2;
    *acc2 = 0;
    return out;
}

// out = 2 out + bit + Σ x[k]² B^(2k) for out.len() == 2 x.len() - 1, returning the
// limb that belongs at out[out.len()].
#[inline(always)]
fn double_add_sqr_diag(x: &[u64], out: &mut [u64], mut bit: u64) -> u64 {
    debug_assert_eq!(out.len(), 2 * x.len() - 1);
    let (&last, rest) = x.split_last().unwrap();
    let mut carry = 0u64;
    for (o, &v) in out.chunks_exact_mut(2).zip(rest) {
        let t = (o[1] as u128) << 64 | o[0] as u128;
        let doubled = t << 1 | bit as u128;
        bit = o[1] >> 63;
        // At most one of these carries: doubled + v² + 1 < 2^129.
        let (sum, c1) = doubled.overflowing_add(v as u128 * v as u128);
        let (sum, c2) = sum.overflowing_add(carry as u128);
        o[0] = sum as u64;
        o[1] = (sum >> 64) as u64;
        carry = (c1 | c2) as u64;
    }
    let sq = last as u128 * last as u128;
    let o = out.last_mut().unwrap();
    let top_bit = *o >> 63;
    let lo = ((*o << 1) | bit) as u128 + sq as u64 as u128 + carry as u128;
    *o = lo as u64;
    top_bit + (sq >> 64) as u64 + (lo >> 64) as u64
}

#[inline(always)]
fn sqr_buf_rows<K: Rows>(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let n = buf.len();
    let len = 2 * n - 1;
    // The rows rely on this, so it is checked in release builds too.
    assert!(
        out.len() >= len,
        "out is not large enough for multiplication"
    );

    // The off-diagonal half by rows, as in mul_buf: row i adds buf[i] * buf[i + 1..]
    // into out[2i + 1..] and stores its top limb at out[n + i]. Then double it and
    // add the squares.
    out[0] = 0;
    if n > 1 {
        out[n] = unsafe { K::set(buf[0], &buf[1..], &mut out[1..]) };
    }
    for i in 1..n - 1 {
        out[n + i] = unsafe { K::add(buf[i], &buf[i + 1..], &mut out[2 * i + 1..]) };
    }
    let top = double_add_sqr_diag(buf, &mut out[..len], 0);
    if out.len() > len {
        out[len] = top;
        out[len + 1..].fill(0);
        return 0;
    }
    top
}

rows_dispatch!(pub fn sqr_buf(buf: &[u64], out: &mut [u64]) -> u64 = sqr_buf_rows);

//KARATSUBA SQUARE

// pub(crate) so the tuner can time one level (a forced entry, docs/tuning_system.md).
pub(crate) fn karatsuba_sqr_core(
    buf: &[u64],
    half: usize,
    out: &mut [u64],
    cross: &mut [u64],
    scratch: &mut [u64],
) -> u64 {
    let (x0, x1) = buf.split_at(half);
    {
        let sum = &mut out[..=half];
        sum[..half].copy_from_slice(x0);
        let mut sum_len = half;
        if add_buf(&mut sum[..half], x1) {
            sum[half] = 1;
            sum_len += 1;
        }

        karatsuba_sqr_mul(&sum[..sum_len], cross, scratch);
    }
    out.fill(0);
    let (z0, z2) = out.split_at_mut(2 * half);
    karatsuba_sqr_mul(x0, z0, scratch);
    let mut overflow = karatsuba_sqr_mul(x1, z2, scratch);
    let z2_len = buf_len(z2);
    sub_buf(cross, z0);
    sub_buf(cross, &z2[..z2_len]);
    if overflow > 0 {
        sub_buf(&mut cross[z2.len()..], &[overflow]);
    }
    if add_buf(&mut out[half..], &cross) {
        overflow += 1;
    }

    return overflow;
}

fn karatsuba_sqr_mul(buf: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    if buf.len() <= k::mul::karatsuba_sqr() {
        return sqr_buf(buf, out);
    }
    let half = (buf.len() + 1) / 2;
    let (cross, rest) = scratch.split_at_mut(2 * half + 1);
    karatsuba_sqr_core(buf, half, out, cross, rest)
}

pub fn find_karatsuba_sqr_scratch_sz(n: usize) -> usize {
    if n <= k::mul::karatsuba_sqr() {
        return 0;
    }
    let half = (n + 1) / 2;
    (2 * half + 1) + find_karatsuba_sqr_scratch_sz(half + 1)
}

// One Karatsuba level at the top, as karatsuba_sqr_entry_static takes, at any
// buf.len() from 6 (at 3 and 5 the cross term is a limb wider than out[half..]).
pub fn karatsuba_sqr_entry_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    let half = (buf.len() + 1) / 2;
    let mut scratch_guard = ScratchGuard::acquire();
    let scratch = scratch_guard.get(2 * half + 1 + find_karatsuba_sqr_scratch_sz(half + 1));
    let (cross, rest) = scratch.split_at_mut(2 * half + 1);
    karatsuba_sqr_core(buf, half, out, cross, rest)
}

pub fn karatsuba_sqr_entry_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> u64 {
    let scratch_sz = find_karatsuba_sqr_scratch_sz(buf.len());
    let half = (buf.len() + 1) / 2;
    let mut scratch = [0; N];
    if scratch_sz > N {
        let mut cross = [0; N];
        karatsuba_sqr_core(buf, half, out, &mut cross[..2 * half + 1], &mut scratch)
    } else {
        let (cross, rest) = scratch.split_at_mut(2 * half + 1);
        karatsuba_sqr_core(buf, half, out, cross, rest)
    }
}

//FFT SQUARE

#[inline(always)]
fn seperate(x: Complex<f64>, y: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
    let a = (x.re + y.re) * 0.5;
    let b = (x.im - y.im) * 0.5;
    let c = (x.im + y.im) * 0.5;
    let d = (x.re - y.re) * 0.5;
    (Complex::new(a, b), Complex::new(c, -d))
}

// The pairs k..m/2 of sqr_recombine, one at a time.
#[inline]
fn sqr_recombine_pairs(x: &mut [Complex<f64>], m: usize, tw: &[Complex<f64>], from: usize) {
    for k in from..m / 2 {
        let (e, o) = seperate(x[k], x[m - k]);
        let (a, b) = seperate(e, tw[k] * o.conj());
        let s = (a + b) * (a - b);
        let t = e * o;
        x[k] = Complex::new(s.re - t.im, s.im + t.re).scale(2.0);
        x[m - k] = Complex::new(s.re + t.im, t.re - s.im).scale(2.0);
    }
}

// sqr_recombine_pairs two pairs at a time, as recombine_pairs_aarch.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sqr_recombine_pairs_aarch(
    x: &mut [Complex<f64>],
    m: usize,
    tw: &[Complex<f64>],
) -> usize {
    debug_assert!(tw.len() >= m / 2 && x.len() >= m);
    let p = x.as_mut_ptr() as *mut f64;
    let t = tw.as_ptr() as *const f64;
    // The scalar loop's halvings are left out, and the powers of two they would have taken come off
    // s and t at the end: scaling by them is exact, so the rounding doesn't change.
    let eighth = vdupq_n_f64(0.125);
    let half = vdupq_n_f64(0.5);
    let mut k = 1;
    while k + 1 < m / 2 {
        let (xr, xi) = load_complex2(p.add(2 * k));
        let (yr, yi) = load_complex2_rev(p.add(2 * (m - k)));
        let (wr, wi) = load_complex2(t.add(2 * k));
        // (e, o) = seperate(x[k], x[m - k]), both doubled
        let er = vaddq_f64(xr, yr);
        let ei = vsubq_f64(xi, yi);
        let or = vaddq_f64(xi, yi);
        let oi = vsubq_f64(yr, xr);
        // v = tw[k] * o.conj(), doubled
        let vr = vfmaq_f64(vmulq_f64(wr, or), wi, oi);
        let vi = vfmsq_f64(vmulq_f64(wi, or), wr, oi);
        // (a, b) = seperate(e, v), times 4
        let ar = vaddq_f64(er, vr);
        let ai = vsubq_f64(ei, vi);
        let br = vaddq_f64(ei, vi);
        let bi = vsubq_f64(vr, er);
        // s = (a + b) * (a - b), times 16, and t = e * o, times 4
        let (pr, pi) = (vaddq_f64(ar, br), vaddq_f64(ai, bi));
        let (qr, qi) = (vsubq_f64(ar, br), vsubq_f64(ai, bi));
        let sr = vfmsq_f64(vmulq_f64(pr, qr), pi, qi);
        let si = vfmaq_f64(vmulq_f64(pr, qi), pi, qr);
        let tr = vfmsq_f64(vmulq_f64(er, or), ei, oi);
        let ti = vfmaq_f64(vmulq_f64(er, oi), ei, or);
        // The output is 2 (s + i t) and its conjugate pair's.
        let (sr, si) = (vmulq_f64(sr, eighth), vmulq_f64(si, eighth));
        let (tr, ti) = (vmulq_f64(tr, half), vmulq_f64(ti, half));
        store_complex2(p.add(2 * k), vsubq_f64(sr, ti), vaddq_f64(si, tr));
        store_complex2_rev(p.add(2 * (m - k)), vaddq_f64(sr, ti), vsubq_f64(tr, si));
        k += 2;
    }
    k
}

// The two terms with no conjugate pair: no pair reads or writes them.
#[inline(always)]
fn sqr_recombine_ends(x: &mut [Complex<f64>], m: usize) {
    let Complex::<f64> { re, im } = x[0];
    x[0] = Complex::new(re * re + im * im, 2.0 * re * im);
    x[m / 2] = Complex::new(
        (x[m / 2].re + x[m / 2].im) * (x[m / 2].re - x[m / 2].im),
        2.0 * x[m / 2].re * x[m / 2].im,
    );
}

pub(crate) fn sqr_recombine(x: &mut [Complex<f64>], m: usize, tw: &[Complex<f64>]) {
    sqr_recombine_ends(x, m);
    #[cfg(target_arch = "aarch64")]
    let from = unsafe { sqr_recombine_pairs_aarch(x, m, tw) };
    #[cfg(not(target_arch = "aarch64"))]
    let from = 1;
    sqr_recombine_pairs(x, m, tw, from);
}

// sqr_recombine without NEON, for the tests.
#[cfg(test)]
pub(crate) fn sqr_recombine_standard(x: &mut [Complex<f64>], m: usize, tw: &[Complex<f64>]) {
    sqr_recombine_ends(x, m);
    sqr_recombine_pairs(x, m, tw, 1);
}

fn fft_sqr_core(
    x: &mut [Complex<f64>],
    fwd: &dyn Fft<f64>,
    bwd: &dyn Fft<f64>,
    tw: &[Complex<f64>],
    fft_scratch: &mut [Complex<f64>],
) -> f64 {
    let m = x.len();
    fwd.process_with_scratch(x, fft_scratch);
    sqr_recombine(x, m, tw);
    bwd.process_with_scratch(x, fft_scratch);
    scale_and_round(x, m as f64)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sqr_decompose_16_x86(buf: &[u64], x: &mut [Complex<f64>]) -> u64 {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;
    let mut carry = 0;
    for &b in buf {
        // loadl loads 64 bits into the low half of a 128-bit register
        let v = _mm_set1_epi64x(balance_limb(b, &mut carry) as i64);
        // pmovsxwd: sign-extend 4 i16s → 4 i32s
        let i32s = _mm_cvtepi16_epi32(v);
        // cvtepi32_pd: 4 i32 → 4 f64
        let f = _mm256_cvtepi32_pd(i32s);
        _mm256_storeu_pd(out.add(idx), f);
        idx += 4;
    }
    let written = buf.len() * 4;
    if written < x.len() * 2 {
        std::ptr::write_bytes(out.add(written), 0, x.len() * 2 - written);
    }
    carry
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sqr_decompose_16_aarch64(buf: &[u64], x: &mut [Complex<f64>]) -> u64 {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;
    let mut carry = 0;
    for &b in buf {
        let (f0, f1) = u64_into_2x2f64s_aarch(balance_limb(b, &mut carry));
        vst1q_f64(out.add(idx), f0);
        vst1q_f64(out.add(idx + 2), f1);
        idx += 4;
    }
    let written = buf.len() * 4;
    if written < x.len() * 2 {
        std::ptr::write_bytes(out.add(written), 0, x.len() * 2 - written);
    }
    carry
}

pub fn sqr_decompose_16_standard(buf: &[u64], x: &mut [Complex<f64>]) -> u64 {
    let coefs = unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2) };
    let mut idx = 0;
    let mut carry = 0;
    for &b in buf {
        let b = balance_limb(b, &mut carry);
        for i in 0..4 {
            coefs[idx + i] = balanced_digit(b, i);
        }
        idx += 4;
    }
    if idx < coefs.len() {
        coefs[idx..].fill(0.0);
    }
    carry
}

// buf in balanced digits, with its top digit unsigned.
pub(crate) fn sqr_decompose(buf: &[u64], x: &mut [Complex<f64>]) {
    #[cfg(target_arch = "x86_64")]
    let carry = if is_x86_feature_detected!("avx2") {
        unsafe { sqr_decompose_16_x86(buf, x) }
    } else {
        sqr_decompose_16_standard(buf, x)
    };

    #[cfg(target_arch = "aarch64")]
    let carry = unsafe { sqr_decompose_16_aarch64(buf, x) };

    let coefs = unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2) };
    fft_absorb_top_carry(coefs, 0, 1, buf, carry);
}

/// buf² into `out` by the 16-bit FFT, redone on the NTT ([`ntt_sqr_entry_dyn`]) when the fold check
/// trips or the result fails the residue check, as [`fft_entry`] does.
pub fn fft_sqr_entry(buf: &[u64], out: &mut [u64]) -> u64 {
    fft_sqr_entry_checked(buf, out, true)
}

// fft_sqr_entry with the fold check optional, as fft_entry_checked.
pub(crate) fn fft_sqr_entry_checked(buf: &[u64], out: &mut [u64], fold: bool) -> u64 {
    if let Some(overflow) = fft_sqr_entry_impl(buf, out, fold).0 {
        if !FFT_RESIDUES || {
            let r = Residues::of(buf);
            Residues::of_with_top(out, overflow) == r.mul(r)
        } {
            return overflow;
        }
        note_residue_reroute();
    }
    note_reroute();
    ntt_sqr_entry_dyn(&buf[..buf_len(buf)], out)
}

/// [`fft_sqr_entry`] without any check, returning the call's rounding beside the FFT's own result.
/// For the precision probe and the tests.
#[doc(hidden)]
pub fn fft_sqr_entry_raw(buf: &[u64], out: &mut [u64]) -> (u64, FftRounding) {
    let (overflow, rounding) = fft_sqr_entry_impl(buf, out, false);
    (overflow.expect("the raw entry never trips"), rounding)
}

fn fft_sqr_entry_impl(buf: &[u64], out: &mut [u64], checked: bool) -> (Option<u64>, FftRounding) {
    let len = buf_len(buf);
    if len == 0 {
        out.fill(0);
        return (Some(0), FftRounding::default());
    }

    let buf = &buf[..len];
    let out_len = 2 * len - 1;
    debug_assert!(
        out.len() >= out_len,
        "out is not large enough for multiplication"
    );
    let (out_core, out_tail) = out.split_at_mut(out_len);
    let bit_len = bit16_length(buf.len(), buf.last().copied().unwrap());
    let n = find_fft_size(2 * bit_len - 1);
    if checked && n > FFT_SQR_MAX_SIZE {
        return (None, FftRounding::default());
    }
    let m = n / 2;
    let (overflow, rounding) =
        fft_sqr_coefs(buf, m, false, checked, |x| fft_accumulate(x, out_core));
    (
        overflow.map(|overflow| absorb_overflow(out_tail, overflow)),
        rounding,
    )
}

// WRAPPED SQUARE (mod B^m - 1)

/// The smallest m >= `min` whose cyclic square mod B^m - 1 runs on an FFT of exactly 4m 16-bit
/// digits: m = o * 2^a with o in {1, 3, 5, 7, 9}, since every FFT size is 4 times one (GMP's
/// mpn_mulmod_bnm1_next_size). Any m >= min serves a caller that only needs the modulus to pass
/// a bound, so rounding up to this size lets the transform wrap at exactly B^m.
pub fn bnm1_size(min: usize) -> usize {
    find_fft_size(4 * min.max(1)) / 4
}

/// Whether [`sqr_mod_bnm1_dyn`] squares an `a_len`-limb operand mod B^m − 1 by the cyclic FFT
/// (above `mul.sqr_bnm1_fft` limbs, with 4m an FFT size of at most its dispatch limit,
/// `FFT_BNM1_16BIT_CUTOFF` modulus limbs), rather than by `sqr_dyn` and a fold.
pub fn bnm1_fft(a_len: usize, m: usize) -> bool {
    bnm1_fft_at(a_len, m, k::mul::sqr_bnm1_fft())
}

// bnm1_fft at an explicit `mul.sqr_bnm1_fft` (invariant 6).
pub fn bnm1_fft_at(a_len: usize, m: usize, cutoff: usize) -> bool {
    a_len > cutoff && m <= FFT_BNM1_16BIT_CUTOFF && find_fft_size(4 * m) == 4 * m
}

// a^2 mod B^m - 1, canonical, m = out.len(), for nonzero a with a.len() <= m and 4m an FFT size.
// It is fft_sqr_entry without the zero padding: a length-4m cyclic convolution of 16-bit digits is
// the square mod 2^(64m) - 1. When the fold check trips it is redone as the NTT's full square,
// folded (not through sqr_dyn, whose FFT square would round about as badly). There is no residue
// check: the primes don't divide B^m − 1. Public (hidden) for benches/probes/fft_precision.rs.
#[doc(hidden)]
pub fn fft_sqr_mod_bnm1(a: &[u64], out: &mut [u64]) {
    if fft_sqr_mod_bnm1_impl(a, out, true).0 {
        return;
    }
    note_reroute();
    let mut scratch = ScratchGuard::acquire();
    let full = scratch.get(2 * a.len());
    let top = ntt_sqr_entry_dyn(a, &mut full[..2 * a.len() - 1]);
    full[2 * a.len() - 1] = top;
    fold_mod_bnm1(full, out);
}

/// [`fft_sqr_mod_bnm1`] without the fold check's redo, returning the call's rounding. For the
/// precision probe and the tests.
#[doc(hidden)]
pub fn fft_sqr_mod_bnm1_raw(a: &[u64], out: &mut [u64]) -> FftRounding {
    fft_sqr_mod_bnm1_impl(a, out, false).1
}

// Whether the cyclic square was written (false: `checked` and its rounding tripped, out
// untouched), and its rounding.
fn fft_sqr_mod_bnm1_impl(a: &[u64], out: &mut [u64], checked: bool) -> (bool, FftRounding) {
    let n = 4 * out.len();
    debug_assert_eq!(find_fft_size(n), n, "4m must be an FFT size");
    debug_assert!(!a.is_empty() && a.len() <= out.len());
    if checked && n > FFT_BNM1_MAX_SIZE {
        return (false, FftRounding::default());
    }
    let half = n / 2;
    // All n coefficients land in exactly m limbs; the signed carry out of the top (below 2^35 in
    // magnitude, as each coefficient is below 2^50) counts multiples of B^m, which is 1.
    let (carry, rounding) =
        fft_sqr_coefs(a, half, true, checked, |x| fft_accumulate_signed(x, out));
    let Some(carry) = carry else {
        return (false, rounding);
    };
    // The carry is far below B^m, so its wrap can't carry again.
    if carry >= 0 {
        if add_prim(out, carry as u64) {
            inc_buf(out);
        }
    } else if sub_prim(out, carry.unsigned_abs() as u64) {
        dec_buf(out);
    }
    canon_mod_bnm1(out);
    (true, rounding)
}

/// Computes a² mod (B^m − 1), m = `out.len()`, `B = 2^64`, as the canonical residue in
/// [0, B^m − 1). Requires a's trimmed length to be at most m. When [`bnm1_fft`] allows (m from
/// [`bnm1_size`], at most `FFT_BNM1_16BIT_CUTOFF`, and a above `mul.sqr_bnm1_fft` limbs)
/// it is one cyclic FFT square, half the transform of a full square of a; otherwise it squares a
/// with `sqr_dyn` and folds the product.
pub fn sqr_mod_bnm1_dyn(a: &[u64], out: &mut [u64]) {
    let a = &a[..buf_len(a)];
    assert!(a.len() <= out.len(), "a must fit the modulus's m limbs");
    if a.is_empty() {
        out.fill(0);
        return;
    }
    if bnm1_fft(a.len(), out.len()) {
        fft_sqr_mod_bnm1(a, out);
    } else {
        let mut guard = ScratchGuard::acquire();
        let square = guard.get(2 * a.len());
        sqr_dyn(a, square);
        fold_mod_bnm1(square, out);
    }
}

//NTT SQUARE

fn ntt_sqr_convolution<P: NTTPrime>(
    buf: &[u64],
    res: &mut [u64],
    tw: &impl NTTTwidles<P>,
    ntt_scratch: &mut [u64],
) {
    res[..buf.len()].copy_from_slice(buf);
    res[buf.len()..].fill(0);
    let mont_res: &mut [Montgomery<P>] = unsafe { std::mem::transmute(res) };
    for elem in mont_res.iter_mut().take(buf.len()) {
        elem.to_mut();
    }
    let mont_scratch: &mut [Montgomery<P>] = unsafe { std::mem::transmute(ntt_scratch) };

    ntt::<P, NTTForward>(mont_res, 1, tw, mont_scratch);

    for elem in mont_res.iter_mut() {
        *elem *= *elem;
    }

    ntt::<P, NTTBackward>(mont_res, 1, tw, mont_scratch);
}

pub fn ntt_sqr_entry_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let out_len = 2 * buf.len() - 1;
    assert!(
        out.len() >= out_len,
        "out is not large enough for multiplication"
    );
    let n1 = find_ntt_size::<P1>(out_len);
    let n2 = find_ntt_size::<P2>(out_len);
    let n3 = find_ntt_size::<P3>(out_len);

    let mut scratch_gaurd = ScratchGuard::acquire();
    let [res1, res2, res3] = scratch_gaurd.get_splits([n1, n2, n3]);

    dyn_convolutions(
        buf,
        None,
        [&mut *res1, &mut *res2, &mut *res3],
        out_len > k::mul::ntt_par_conv(),
    );
    let (out_core, out_tail) = out.split_at_mut(out_len);
    let overflow = ntt_accumulate(res1, res2, res3, out_core, out_len, n1, n2, n3);
    absorb_overflow(out_tail, overflow)
}

fn ntt_static_sqr_convolution<const N: usize, P: NTTPrime>(
    buf: &[u64],
    res: &mut [u64],
    n: usize,
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    if n <= N {
        let mut tw = StaticNTTTwidles::<N, P>::build(n);
        ntt_sqr_convolution::<P>(
            buf,
            &mut res[..n],
            &mut tw,
            &mut ntt_scratch[..ntt_scratch_len(n)],
        );
    } else {
        ntt_chunked_convolution::<N, P>(buf, buf, res, n, buf_scratch, ntt_scratch);
    }
}

pub fn ntt_sqr_entry_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let out_len = 2 * buf.len() - 1;
    assert!(
        out_len <= N && out.len() >= out_len,
        "inputs exceed static capacity"
    );
    let mut res1 = [0; N];
    let mut res2 = [0; N];
    let mut res3 = [0; N];
    let n1 = find_ntt_size::<P1>(out_len);
    let n2 = find_ntt_size::<P2>(out_len);
    let n3 = find_ntt_size::<P3>(out_len);

    if out_len > k::mul::ntt_par_conv() {
        let conv1 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_static_sqr_convolution::<N, P1>(
                buf,
                &mut res1,
                n1,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        let conv2 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_static_sqr_convolution::<N, P2>(
                buf,
                &mut res2,
                n2,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        let conv3 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_static_sqr_convolution::<N, P3>(
                buf,
                &mut res3,
                n3,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        rayon::join(conv1, || rayon::join(conv2, conv3));
    } else {
        let mut buf_scratch = [0; N];
        let mut ntt_scratch = [0; N];
        ntt_static_sqr_convolution::<N, P1>(buf, &mut res1, n1, &mut buf_scratch, &mut ntt_scratch);
        ntt_static_sqr_convolution::<N, P2>(buf, &mut res2, n2, &mut buf_scratch, &mut ntt_scratch);
        ntt_static_sqr_convolution::<N, P3>(buf, &mut res3, n3, &mut buf_scratch, &mut ntt_scratch);
    }
    let (out_core, out_tail) = out.split_at_mut(out_len);
    let overflow = ntt_accumulate(
        &mut res1[..n1.min(N)],
        &mut res2[..n2.min(N)],
        &mut res3[..n3.min(N)],
        out_core,
        out_len,
        n1,
        n2,
        n3,
    );
    absorb_overflow(out_tail, overflow)
}

//END NTT SQUARE

enum DynSqrDispatch {
    School,
    Karatsuba,
    FFT,
    NTT,
}

fn dyn_sqr_dispatch(n: usize) -> DynSqrDispatch {
    dyn_sqr_dispatch_at(n, k::mul::karatsuba_sqr(), k::mul::fft_sqr())
}

// sqr_dyn's choice at explicit `mul.karatsuba_sqr` and `mul.fft_sqr` (invariant 6).
fn dyn_sqr_dispatch_at(n: usize, kara: usize, fft: usize) -> DynSqrDispatch {
    if n <= kara {
        DynSqrDispatch::School
    } else if n <= fft {
        DynSqrDispatch::Karatsuba
    } else if n <= FFT_SQR_16BIT_CUTOFF {
        DynSqrDispatch::FFT
    } else {
        DynSqrDispatch::NTT
    }
}

pub fn sqr_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let out_len = 2 * buf.len() - 1;
    assert!(
        out_len <= out.len(),
        "out is not large enough for multiplication"
    );
    let (out_core, out_tail) = out.split_at_mut(out_len);
    let alg = dyn_sqr_dispatch(buf.len());
    let overflow = match alg {
        DynSqrDispatch::School => sqr_buf(buf, out_core),
        DynSqrDispatch::Karatsuba => karatsuba_sqr_entry_dyn(buf, out_core),
        DynSqrDispatch::FFT => fft_sqr_entry(buf, out_core),
        DynSqrDispatch::NTT => ntt_sqr_entry_dyn(buf, out_core),
    };
    // Debug builds check every NTT square by its residues mod the NTT primes (linear); FFT squares
    // check themselves in fft_sqr_entry and redo a mismatch.
    #[cfg(debug_assertions)]
    if matches!(alg, DynSqrDispatch::NTT) {
        let r = Residues::of(buf);
        assert_eq!(
            Residues::of_with_top(out_core, overflow),
            r.mul(r),
            "sqr_dyn: the {}-limb NTT square fails its residue check",
            buf.len()
        );
    }
    absorb_overflow(out_tail, overflow)
}

pub fn sqr_vec(buf: &[u64]) -> (Vec<u64>, u64) {
    if buf.is_empty() {
        return (Vec::new(), 0);
    }
    let mut out = vec![0; 2 * buf.len() - 1];
    let of = sqr_dyn(buf, &mut out);
    (out, of)
}

enum StaticSqrDispatch {
    School,
    Karatsubaa,
    NTT,
}

fn static_sqr_dispatch(n: usize) -> StaticSqrDispatch {
    static_sqr_dispatch_at(n, k::mul::karatsuba_sqr(), k::mul::static_ntt_sqr())
}

// sqr_static's choice at explicit `mul.karatsuba_sqr` and `mul.static_ntt_sqr` (invariant 6).
fn static_sqr_dispatch_at(n: usize, kara: usize, ntt: usize) -> StaticSqrDispatch {
    if n <= kara {
        StaticSqrDispatch::School
    } else if n <= ntt {
        StaticSqrDispatch::Karatsubaa
    } else {
        StaticSqrDispatch::NTT
    }
}

pub fn sqr_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let out_len = 2 * buf.len() - 1;
    debug_assert!(
        out_len <= N,
        "multiplication exceeds static capacity N: {N}"
    );
    debug_assert!(
        out_len <= out.len(),
        "out is not large enough for multiplication"
    );
    let (out_core, out_tail) = out.split_at_mut(out_len);
    let overflow = match static_sqr_dispatch(buf.len()) {
        StaticSqrDispatch::School => sqr_buf(buf, out_core),
        StaticSqrDispatch::Karatsubaa => karatsuba_sqr_entry_static::<N>(buf, out_core),
        StaticSqrDispatch::NTT => ntt_sqr_entry_static::<N>(buf, out_core),
    };
    absorb_overflow(out_tail, overflow)
}

pub fn sqr_arr<const N: usize>(buf: &[u64]) -> ([u64; N], u64) {
    let mut out = [0; N];
    let of = sqr_static::<N>(buf, &mut out);
    (out, of)
}

// HIGH MULTIPLICATION
// Gets only the top part of a multiplication, only optimized for small inputs asymptotically same
// computation as a full product other then truncating inputs dynamically.

// Operand limbs kept below the window when truncating: two keep the carry error
// within one ulp, as hi_mul_buf's (with none it is a whole low limb).
const HI_MUL_GUARD_LIMBS: usize = 2;

// (guard, core): x's top `core` limbs and up to `guard` limbs below them.
fn hi_guard_split(x: &[u64], core: usize, guard: usize) -> (&[u64], &[u64]) {
    let core_start = x.len().saturating_sub(core);
    let guard_start = core_start.saturating_sub(guard);
    (&x[guard_start..core_start], &x[core_start..])
}

// The window below the top limb of full = (a_t B^|a_g| + a_g)(b_t B^|b_g| + b_g),
// returning the top limb; `core` writes a_t * b_t into full's top. The guard terms
// are linear add_mul passes, so the core product keeps its unguarded size (one
// limb past a transform size can cost up to 2x).
fn hi_guarded_window(
    full: &mut [u64],
    (a_g, a_t): (&[u64], &[u64]),
    (b_g, b_t): (&[u64], &[u64]),
    out: &mut [u64],
    core: impl FnOnce(&mut [u64]),
) -> u64 {
    let low = a_g.len() + b_g.len();
    core(&mut full[low..]);
    full[..low].fill(0);
    for (j, &limb) in b_g.iter().enumerate() {
        let carry = add_mul(&mut full[a_g.len() + j..], a_t, limb);
        debug_assert_eq!(carry, 0);
    }
    for (i, &limb) in a_g.iter().enumerate() {
        let carry = add_mul(&mut full[b_g.len() + i..], b_t, limb);
        debug_assert_eq!(carry, 0);
        let carry = add_mul(&mut full[i..], b_g, limb);
        debug_assert_eq!(carry, 0);
    }
    let n = full.len();
    out.copy_from_slice(&full[n - 1 - out.len()..n - 1]);
    full[n - 1]
}

// The static guard, clamped so the split paths' low buffers (out_len + 2 guard - 1
// limbs) fit N. With operands of at most N limbs, the windows from N - 1 are exact.
fn hi_mul_static_guard<const N: usize>(out_len: usize) -> usize {
    ((N + 1).saturating_sub(out_len) / 2).min(HI_MUL_GUARD_LIMBS)
}

// Adds acc0 + acc1 B, the guard columns' carry into a non-empty window, to out and
// its top limb, returning the new top.
fn add_guard_carry(out: &mut [u64], mut top: u64, acc0: u64, acc1: u64) -> u64 {
    top += add_prim(out, acc0) as u64;
    if out.len() > 1 {
        top += add_prim(&mut out[1..], acc1) as u64;
    } else {
        top += acc1;
    }
    top
}

#[inline(always)]
fn hi_mul_buf_rows<K: Rows>(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let d = (a.len() + b.len() - 1).saturating_sub(out.len());
    if d == 0 {
        return mul_buf_rows::<K>(a, b, out);
    }
    let (a, b) = if a.len() > b.len() { (a, b) } else { (b, a) };

    // The two columns below the window give the carry into it.
    let (mut acc0, mut acc1, mut acc2) = (0, 0, 0);
    if d >= 2 {
        mul_elem(a, b, d - 2, &mut acc0, &mut acc1, &mut acc2);
    }
    mul_elem(a, b, d - 1, &mut acc0, &mut acc1, &mut acc2);
    if out.is_empty() {
        return acc0;
    }

    // Row i adds b[i] * a[j] for every column i + j >= d and stores its top limb
    // at out[i + a.len() - d], as in mul_buf; the first row writes its span.
    let first = (d + 1).saturating_sub(a.len());
    let mut top = unsafe { K::set(b[first], &a[d - first..], out) };
    for i in first + 1..b.len() {
        out[i - 1 + a.len() - d] = top;
        let j0 = d.saturating_sub(i);
        top = unsafe { K::add(b[i], &a[j0..], &mut out[i + j0 - d..]) };
    }
    add_guard_carry(out, top, acc0, acc1)
}

// The window by columns: the two guard columns, then one column per limb.
pub fn hi_mul_buf_cols(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let d = (a.len() + b.len() - 1).saturating_sub(out.len());
    if d == 0 {
        return mul_buf(a, b, out);
    }
    let (mut acc0, mut acc1, mut acc2) = (0, 0, 0);
    if d >= 2 {
        mul_elem(a, b, d - 2, &mut acc0, &mut acc1, &mut acc2);
    }
    mul_elem(a, b, d - 1, &mut acc0, &mut acc1, &mut acc2);
    for (n, o) in (d..).zip(out.iter_mut()) {
        *o = mul_elem(a, b, n, &mut acc0, &mut acc1, &mut acc2);
    }
    acc0
}

pub fn hi_mul_buf(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    // Narrow windows give short rows, whose per-row cost loses to columns.
    if out.len() <= k::mul::hi_mul_row() {
        return hi_mul_buf_cols(a, b, out);
    }
    hi_mul_buf_by_rows(a, b, out)
}

// hi_mul_buf by rows at any width (benchmarks compare it with columns).
rows_dispatch!(
    pub fn hi_mul_buf_by_rows(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 = hi_mul_buf_rows
);

// A high product's algorithm (invariant 6): the whole product when it fits out, else hi_*_buf
// for windows of up to `cutoff` limbs (`mul.<dyn|static>_partial_<mul|sqr>`) and the guarded
// product of the operands' top limbs above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiAlg {
    Full,
    School,
    Guarded,
}

pub fn hi_alg(full_len: usize, out_len: usize, cutoff: usize) -> HiAlg {
    if full_len <= out_len {
        HiAlg::Full
    } else if out_len <= cutoff {
        HiAlg::School
    } else {
        HiAlg::Guarded
    }
}

pub fn hi_mul_dyn(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    match hi_alg(a.len() + b.len() - 1, out.len(), k::mul::dyn_partial_mul()) {
        HiAlg::Full => mul_dyn(a, b, out),
        HiAlg::School => hi_mul_buf(a, b, out),
        HiAlg::Guarded => hi_mul_guarded_dyn(a, b, out),
    }
}

// hi_mul_dyn's window from the product of the operands' guarded top limbs, for a
// product wider than out (benchmarks call it directly).
pub fn hi_mul_guarded_dyn(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    let ((a_g, a_t), (b_g, b_t)) = (
        hi_guard_split(a, out.len(), HI_MUL_GUARD_LIMBS),
        hi_guard_split(b, out.len(), HI_MUL_GUARD_LIMBS),
    );
    let full_len = a_g.len() + a_t.len() + b_g.len() + b_t.len();
    let mut scratch = ScratchGuard::acquire();
    hi_guarded_window(scratch.get(full_len), (a_g, a_t), (b_g, b_t), out, |p| {
        mul_dyn(a_t, b_t, p);
    })
}

pub fn hi_mul_static<const N: usize>(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    debug_assert!(a.len() <= N);
    debug_assert!(b.len() <= N);
    debug_assert!(
        out.len() <= N,
        "out is to large for static parameter N: {N}"
    );
    match hi_alg(a.len() + b.len() - 1, out.len(), k::mul::static_partial_mul()) {
        HiAlg::Full => mul_static::<N>(a, b, out),
        HiAlg::School => hi_mul_buf(a, b, out),
        HiAlg::Guarded => hi_mul_guarded_static::<N>(a, b, out),
    }
}

// hi_mul_static's counterpart of hi_mul_guarded_dyn.
pub fn hi_mul_guarded_static<const N: usize>(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    let out_len = out.len();
    let guard = hi_mul_static_guard::<N>(out_len);
    let ((a_g, a_t), (b_g, b_t)) = (
        hi_guard_split(a, out_len, guard),
        hi_guard_split(b, out_len, guard),
    );
    let full_len = a_g.len() + a_t.len() + b_g.len() + b_t.len();
    if full_len <= N {
        let mut full = [0; N];
        return hi_guarded_window(&mut full[..full_len], (a_g, a_t), (b_g, b_t), out, |p| {
            mul_static::<N>(a_t, b_t, p);
        });
    }

    // Too wide for one N-limb product: the same window from split sub-products.
    let trunc_a = &a[a.len() - a_g.len() - a_t.len()..];
    let trunc_b = &b[b.len() - b_g.len() - b_t.len()..];

    let (long, short) = if trunc_a.len() > trunc_b.len() {
        (trunc_a, trunc_b)
    } else {
        (trunc_b, trunc_a)
    };

    let k = out_len + 1;
    let beta_min = k.saturating_sub(long.len()).max(1);
    let beta_max = short.len().min(k - 1);
    let beta_pref = k / 2;
    let beta = beta_pref.clamp(beta_min, beta_max);
    let alpha = k - beta;
    let l_split_idx = long.len() - alpha;
    let s_split_idx = short.len() - beta;
    let (l0, l1) = long.split_at(l_split_idx);
    let (s0, s1) = short.split_at(s_split_idx);

    let mut of = mul_static::<N>(l1, s1, out);
    let mut low_carry = 0u64;
    let low_len = l0.len() + s0.len();
    let mut low = [0; N];

    if !l0.is_empty() && !s0.is_empty() {
        debug_assert!(low_len <= N);
        let low_of = mul_static::<N>(l0, s0, &mut low[..low_len]);
        debug_assert_eq!(low_of, 0);
    }

    if !l0.is_empty() {
        let tmp1_len = l0.len() + s1.len() - 1;
        let mut tmp1 = [0; N];
        let tmp1_of = mul_static::<N>(l0, s1, &mut tmp1[..tmp1_len]);

        low_carry += add_buf(&mut low[s0.len()..low_len], &tmp1[..l0.len()]) as u64;
        of += add_buf(out, &tmp1[l0.len()..tmp1_len]) as u64;
        of += add_prim(&mut out[tmp1_len - l0.len()..], tmp1_of) as u64;
    }

    if !s0.is_empty() {
        // Like tmp1, keep the top limb out of the buffer: l1 * s0 can have
        // N + 1 limbs when out_len == N.
        let tmp2_len = l1.len() + s0.len() - 1;
        let mut tmp2 = [0; N];
        let tmp2_of = mul_static::<N>(l1, s0, &mut tmp2[..tmp2_len]);

        low_carry += add_buf(&mut low[l0.len()..low_len], &tmp2[..s0.len()]) as u64;
        of += add_buf(out, &tmp2[s0.len()..tmp2_len]) as u64;
        of += add_prim(&mut out[tmp2_len - s0.len()..], tmp2_of) as u64;
    }

    of += add_prim(out, low_carry) as u64;

    return of;
}

// HIGH SQUARING

#[inline(always)]
fn hi_sqr_buf_rows<K: Rows>(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let n = buf.len();
    let d = (2 * n - 1).saturating_sub(out.len());
    if d == 0 {
        return sqr_buf_rows::<K>(buf, out);
    }

    // The two columns below the window give the carry into it.
    let (mut acc0, mut acc1, mut acc2) = (0, 0, 0);
    if d >= 2 {
        sqr_elem(buf, d - 2, &mut acc0, &mut acc1, &mut acc2);
    }
    sqr_elem(buf, d - 1, &mut acc0, &mut acc1, &mut acc2);
    if out.is_empty() {
        return acc0;
    }

    // The off-diagonal half's window: row i adds buf[i] * buf[j] for j > i and
    // i + j >= d and stores its top limb at out[i + n - d]; the first row writes
    // its span, and with no rows the window is just out[0].
    let first = (d + 1).saturating_sub(n);
    if first + 1 < n {
        out[first + n - d] = unsafe { K::set(buf[first], &buf[d - first..], out) };
    } else {
        out[0] = 0;
    }
    for i in first + 1..n - 1 {
        let j0 = (i + 1).max(d.saturating_sub(i));
        out[i + n - d] = unsafe { K::add(buf[i], &buf[j0..], &mut out[i + j0 - d..]) };
    }

    // Double it and add the squares from column d up; for odd d, column d - 1's
    // square is already in the guard, so out[0] is only doubled.
    let i0 = (d + 1) / 2;
    let q0 = 2 * i0 - d;
    let mut bit = 0;
    if q0 == 1 {
        bit = out[0] >> 63;
        out[0] <<= 1;
    }
    let top = double_add_sqr_diag(&buf[i0..], &mut out[q0..], bit);
    add_guard_carry(out, top, acc0, acc1)
}

// The window by columns: the two guard columns, then one column per limb.
pub fn hi_sqr_buf_cols(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let d = (2 * buf.len() - 1).saturating_sub(out.len());
    if d == 0 {
        return sqr_buf(buf, out);
    }
    let (mut acc0, mut acc1, mut acc2) = (0, 0, 0);
    if d >= 2 {
        sqr_elem(buf, d - 2, &mut acc0, &mut acc1, &mut acc2);
    }
    sqr_elem(buf, d - 1, &mut acc0, &mut acc1, &mut acc2);
    for (n, o) in (d..).zip(out.iter_mut()) {
        *o = sqr_elem(buf, n, &mut acc0, &mut acc1, &mut acc2);
    }
    acc0
}

pub fn hi_sqr_buf(buf: &[u64], out: &mut [u64]) -> u64 {
    // Narrow windows give short rows, whose per-row cost loses to columns.
    if out.len() <= k::mul::hi_sqr_row() {
        return hi_sqr_buf_cols(buf, out);
    }
    hi_sqr_buf_by_rows(buf, out)
}

// hi_sqr_buf by rows at any width (benchmarks compare it with columns).
rows_dispatch!(
    pub fn hi_sqr_buf_by_rows(buf: &[u64], out: &mut [u64]) -> u64 = hi_sqr_buf_rows
);

pub fn hi_sqr_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    match hi_alg(2 * buf.len() - 1, out.len(), k::mul::dyn_partial_sqr()) {
        HiAlg::Full => sqr_dyn(buf, out),
        HiAlg::School => hi_sqr_buf(buf, out),
        HiAlg::Guarded => hi_sqr_guarded_dyn(buf, out),
    }
}

// hi_sqr_dyn's counterpart of hi_mul_guarded_dyn.
pub fn hi_sqr_guarded_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    let (x_g, x_t) = hi_guard_split(buf, out.len(), HI_MUL_GUARD_LIMBS);
    let mut scratch = ScratchGuard::acquire();
    let full = scratch.get(2 * (x_g.len() + x_t.len()));
    hi_guarded_window(full, (x_g, x_t), (x_g, x_t), out, |p| {
        sqr_dyn(x_t, p);
    })
}

pub fn hi_sqr_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    debug_assert!(buf.len() <= N);
    debug_assert!(
        out.len() <= N,
        "out is to large for static parameter N: {N}"
    );
    match hi_alg(2 * buf.len() - 1, out.len(), k::mul::static_partial_sqr()) {
        HiAlg::Full => sqr_static::<N>(buf, out),
        HiAlg::School => hi_sqr_buf(buf, out),
        HiAlg::Guarded => hi_sqr_guarded_static::<N>(buf, out),
    }
}

// hi_sqr_static's counterpart of hi_mul_guarded_dyn.
pub fn hi_sqr_guarded_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> u64 {
    let out_len = out.len();
    let (x_g, x_t) = hi_guard_split(buf, out_len, hi_mul_static_guard::<N>(out_len));
    let full_len = 2 * (x_g.len() + x_t.len());
    if full_len <= N {
        let mut full = [0; N];
        return hi_guarded_window(&mut full[..full_len], (x_g, x_t), (x_g, x_t), out, |p| {
            sqr_static::<N>(x_t, p);
        });
    }

    // Too wide for one N-limb square: the same window from split sub-products.
    let trunc_buf = &buf[buf.len() - x_g.len() - x_t.len()..];
    let tmp_len = 2 * trunc_buf.len() - 1;

    let h = (out_len + 1) / 2;
    let (x0, x1) = trunc_buf.split_at(trunc_buf.len() - h);

    let hi_ofs = out.len() + 1 - 2 * h;
    let d = tmp_len - out_len;
    debug_assert_eq!(2 * x0.len() - hi_ofs, d);

    out[..hi_ofs].fill(0);
    let mut of = sqr_static::<N>(x1, &mut out[hi_ofs..]);

    let mut low = [0u64; N];
    let x0_sqr_len = 2 * x0.len() - 1;
    let x0_sqr_of = sqr_static::<N>(x0, &mut low[..x0_sqr_len]);
    let x0_high = if x0_sqr_len < d {
        low[x0_sqr_len] = x0_sqr_of;
        0
    } else {
        x0_sqr_of
    };

    let cross_len = x0.len() + x1.len() - 1;
    let mut cross = [0u64; N];
    let mut cross_of = mul_static::<N>(x0, x1, &mut cross[..cross_len]);
    let shl_of = shl_buf(&mut cross[..cross_len], 1);
    let cross_of_hi = cross_of >> 63;
    cross_of = (cross_of << 1) | shl_of;

    let cross_start = d - x0.len();
    let mut low_carry = 0u64;
    if cross_start > 0 {
        low_carry += add_buf(&mut low[x0.len()..d], &cross[..cross_start]) as u64;
    }

    of += add_prim(out, x0_high) as u64;
    of += add_prim(out, low_carry) as u64;
    if cross_start < cross_len {
        let cross_high_len = cross_len - cross_start;
        of += add_buf(out, &cross[cross_start..cross_len]) as u64;
        of += add_prim(&mut out[cross_high_len..], cross_of) as u64;
        of += add_prim(&mut out[cross_high_len + 1..], cross_of_hi) as u64;
    } else {
        of += add_prim(out, cross_of) as u64;
        of += add_prim(&mut out[1..], cross_of_hi) as u64;
    }

    return of;
}

//MIDDLE MULTIPLICATION
// Gets n middle limbs from a (2n-1) x n product.
// n: [0..n-1] x 2n-1: [0..2n-2] -> 3n-2: [0..3n-3] -> [n-1..2n-2]

#[inline]
fn debug_assert_mid_mul_operands(long: &[u64], short: &[u64]) {
    debug_assert!(!short.is_empty(), "short must not be empty");
    debug_assert_eq!(
        long.len(),
        short.len().saturating_mul(2).saturating_sub(1),
        "long must be exactly 2*short-1 limbs for middle product"
    );
}

#[inline]
fn debug_assert_mid_mul_shape(long: &[u64], short: &[u64], out: &[u64]) {
    debug_assert_mid_mul_operands(long, short);
    debug_assert_eq!(
        out.len(),
        short.len(),
        "size of out must be the same size as short"
    );
}

#[inline]
fn mul_column_parts(a: &[u64], b: &[u64], n: usize) -> (u64, u64, u64) {
    let (mut acc0, mut acc1, mut acc2) = (0, 0, 0);
    let out = mul_elem(a, b, n, &mut acc0, &mut acc1, &mut acc2);
    (out, acc0, acc1)
}

pub fn middle_correction(long: &[u64], short: &[u64]) -> (u64, u64) {
    debug_assert_mid_mul_operands(long, short);
    let n = short.len();
    if n < 2 {
        return (0, 0);
    }
    let (_, acc0, acc1) = mul_column_parts(long, short, n - 2);
    return (acc0, acc1);
}

pub fn mid_mul_buf(long: &[u64], short: &[u64], out: &mut [u64]) -> (u64, u64) {
    debug_assert_mid_mul_shape(long, short, out);
    let (mut acc0, mut acc1) = middle_correction(long, short);
    let mut acc2: u64 = 0;
    let n = short.len();
    for (i, elem) in ((n - 1)..=(2 * n - 2)).zip(out) {
        *elem = mul_elem(long, short, i, &mut acc0, &mut acc1, &mut acc2);
    }
    return (acc0, acc1);
}

#[inline]
fn add_mid_overflow(of: &mut (u64, u64), carry: bool) {
    if carry {
        let (lo, hi_carry) = of.0.overflowing_add(1);
        of.0 = lo;
        of.1 = of.1.wrapping_add(hi_carry as u64);
    }
}

fn fft_accumulate_mid(
    x: &[Complex<f64>],
    out: &mut [u64],
    next_col: (u64, u64),
    next_next_col_lo: u64,
) -> (u64, u64) {
    let coefs = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len() * 2) };
    let mut chunks = coefs.chunks_exact(4);
    let mut carry: i128 = 0;

    for elem in out.iter_mut() {
        *elem = fft_accumulate_limb(&mut chunks, &mut carry);
    }

    let next_limb = fft_accumulate_limb(&mut chunks, &mut carry);
    let next_next_limb = fft_accumulate_limb(&mut chunks, &mut carry);

    let (lo, carry_into_next) = next_limb.overflowing_sub(next_col.0);
    let hi = next_next_limb
        .wrapping_sub(next_next_col_lo)
        .wrapping_sub(next_col.1)
        .wrapping_sub(carry_into_next as u64);
    (lo, hi)
}

/// The middle product long[2n-1] × short[n] → out[n] by the 16-bit FFT, redone on the NTT
/// ([`ntt_mid_mul_dyn`], within the same documented error) when the fold check fails. The check
/// covers every coefficient of the transform, not only the band it reads. There is no residue
/// check, as the band omits an unknown part of the product. The band holds the product's value to
/// about 2^34 units of its lowest limb in either sign, which can flip every limb of a run of zeros
/// or all ones, so compare two middle products by value.
pub fn fft_mid_mul(long: &[u64], short: &[u64], out: &mut [u64]) -> (u64, u64) {
    match fft_mid_mul_impl(long, short, out, true).0 {
        Some(of) => of,
        None => {
            note_reroute();
            ntt_mid_mul_dyn(long, short, out)
        }
    }
}

/// [`fft_mid_mul`] without the fold check's redo, returning the call's rounding beside the FFT's own
/// result. For the precision probe and the tests.
#[doc(hidden)]
pub fn fft_mid_mul_raw(long: &[u64], short: &[u64], out: &mut [u64]) -> ((u64, u64), FftRounding) {
    let (of, rounding) = fft_mid_mul_impl(long, short, out, false);
    (of.expect("the raw entry never trips"), rounding)
}

fn fft_mid_mul_impl(
    long: &[u64],
    short: &[u64],
    out: &mut [u64],
    checked: bool,
) -> (Option<(u64, u64)>, FftRounding) {
    debug_assert_mid_mul_shape(long, short, out);
    let sz = short.len();
    // Inputs may carry zero top limbs (quotient-refinement operands are not
    // normalized); treating them as one 16-bit digit only over-sizes the fft.
    let (l_len, s_len, w) = (
        bit16_length(long.len(), long.last().copied().unwrap().max(1)),
        bit16_length(short.len(), short.last().copied().unwrap().max(1)),
        4,
    );
    let start_coef = (sz - 1) * w;
    let end_coef = (2 * sz + 1) * w;
    let n = find_fft_size((l_len + s_len - start_coef - 1).max(end_coef));
    if checked && n > FFT_MID_MAX_SIZE {
        return (None, FftRounding::default());
    }
    let (next_col_lo, next_col_hi, _) = mul_column_parts(long, short, 2 * sz - 1);
    let (next_next_col_lo, _, _) = mul_column_parts(long, short, 2 * sz);
    let next_col = (next_col_lo, next_col_hi);
    // Balanced coefficients below the band sum to under about 2^34 units of its lowest limb, in
    // either sign, so the band already holds the product's value and no column needs adding back.
    fft_mul_coefs(long, short, n, true, checked, |x| {
        fft_accumulate_mid(
            &x[start_coef / 2..end_coef / 2],
            out,
            next_col,
            next_next_col_lo,
        )
    })
}

// out[0] is coefficient lo of the convolution: s - 1 for a canonical middle
// product, lower when the static path drops long's low zero limbs. It reads the
// coefficients from lo - 2 up, which no wrapped column may reach: every prime's
// transform covers at least 2s + 1 of them (mid_ntt_size).
fn ntt_mid_accumulate_scaled(
    res1: &mut [u64],
    res2: &mut [u64],
    res3: &mut [u64],
    out: &mut [u64], // len == s
    lo: usize,
    n1: usize,
    n2: usize,
    n3: usize,
) -> (u64, u64) {
    let inv_n1 = Montgomery::to(n1 as u64).pow(P1::P - 2);
    let inv_n2 = Montgomery::to(n2 as u64).pow(P2::P - 2);
    let inv_n3 = Montgomery::to(n3 as u64).pow(P3::P - 2);

    let hi = lo + out.len();
    for i in lo - 2..hi {
        CRT.crt(
            &mut res1[i],
            &mut res2[i],
            &mut res3[i],
            inv_n1,
            inv_n2,
            inv_n3,
        );
    }

    out.copy_from_slice(&res1[lo..hi]);
    let of1 = add_buf(out, &res2[lo - 1..hi - 1]) as u64;
    let of2 = add_buf(out, &res3[lo - 2..hi - 2]) as u64;

    let top = res2[hi - 1] as u128 + res3[hi - 2] as u128 + of1 as u128 + of2 as u128;
    (top as u64, res3[hi - 1].wrapping_add((top >> 64) as u64))
}

fn acc_scaled_cyclic_convolution<const N: usize, P: NTTPrime>(
    chunk: &[u64],
    short: &[u64],
    offset: usize,
    modulus_len: usize,
    res: &mut [u64],
    tmp: &mut [u64],
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    let conv_len = chunk.len() + short.len() - 1;
    let n = find_ntt_size::<P>(conv_len);
    debug_assert!(n <= N);

    let mut tw = StaticNTTTwidles::<N, P>::build(n);
    ntt_convolution::<P>(
        chunk,
        short,
        &mut tmp[..n],
        &mut tw,
        &mut buf_scratch[..n],
        ntt_scratch,
    );

    let scale =
        Montgomery::<P>::to(modulus_len as u64) * Montgomery::<P>::to(n as u64).pow(P::P - 2);
    let mont_tmp: &mut [Montgomery<P>] = unsafe { std::mem::transmute(&mut tmp[..conv_len]) };
    let mont_res: &mut [Montgomery<P>] = unsafe { std::mem::transmute(res) };
    for (i, elem) in mont_tmp.iter().enumerate() {
        let mut idx = offset + i;
        if idx >= modulus_len {
            idx -= modulus_len;
        }
        if idx < mont_res.len() {
            mont_res[idx] += *elem * scale;
        }
    }
}

fn ntt_mid_static_convolution<const N: usize, P: NTTPrime>(
    long: &[u64],
    short: &[u64],
    res: &mut [u64],
    n: usize,
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    debug_assert!(long.len() <= N);
    debug_assert!(short.len() <= N);
    debug_assert!(res.len() >= long.len());

    if n <= N {
        let mut tw = StaticNTTTwidles::<N, P>::build(n);
        ntt_convolution::<P>(
            long,
            short,
            &mut res[..n],
            &mut tw,
            &mut buf_scratch[..n],
            ntt_scratch,
        );
        return;
    }

    ntt_chunked_convolution::<N, P>(long, short, res, n, buf_scratch, ntt_scratch);
}

// A middle product's transform size for prime P. The product's columns run to 3s - 3
// and wrap past the size, so at 2s or less they land on the coefficients the
// accumulation reads below the band; the three primes pick different sizes, so the
// CRT would then combine residues that disagree.
fn mid_ntt_size<P: NTTPrime>(s: usize) -> usize {
    find_ntt_size::<P>(2 * s + 1)
}

pub fn ntt_mid_mul_dyn(long: &[u64], short: &[u64], out: &mut [u64]) -> (u64, u64) {
    debug_assert_mid_mul_shape(long, short, out);
    let sz = short.len();
    let n1 = mid_ntt_size::<P1>(sz);
    let n2 = mid_ntt_size::<P2>(sz);
    let n3 = mid_ntt_size::<P3>(sz);
    let mut scratch_gaurd = ScratchGuard::acquire();
    let [res1, res2, res3] = scratch_gaurd.get_splits([n1, n2, n3]);

    let par = 2 * sz - 1 > k::mul::ntt_par_conv();
    dyn_convolutions(long, Some(short), [&mut *res1, &mut *res2, &mut *res3], par);
    let mut of = ntt_mid_accumulate_scaled(res1, res2, res3, out, sz - 1, n1, n2, n3);
    let (c0, c1) = middle_correction(long, short);
    add_mid_overflow(&mut of, add_prim(out, c0));
    add_mid_overflow(&mut of, add_prim(&mut out[1..], c1));
    return of;
}

// long may pass N only by low zero padding, long = T B^z: the convolution runs on T
// at the unchanged mid_ntt_size, with the band z coefficients lower.
pub fn ntt_mid_mul_static<const N: usize>(
    long: &[u64],
    short: &[u64],
    out: &mut [u64],
) -> (u64, u64) {
    debug_assert_mid_mul_shape(long, short, out);
    debug_assert!(short.len() <= N);
    debug_assert!(out.len() <= N);

    let z = long.len().saturating_sub(N);
    if short.len() < z + 3 {
        return mid_mul_buf(long, short, out);
    }
    let (pad, long_top) = long.split_at(z);
    assert!(
        pad.iter().all(|&l| l == 0),
        "static middle product: long past N must be low zero padding"
    );

    let sz = short.len();
    let n1 = mid_ntt_size::<P1>(sz);
    let n2 = mid_ntt_size::<P2>(sz);
    let n3 = mid_ntt_size::<P3>(sz);

    let mut res1 = [0; N];
    let mut res2 = [0; N];
    let mut res3 = [0; N];

    if 2 * sz - 1 > k::mul::ntt_par_conv() {
        let conv1 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_mid_static_convolution::<N, P1>(
                long_top,
                short,
                &mut res1,
                n1,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        let conv2 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_mid_static_convolution::<N, P2>(
                long_top,
                short,
                &mut res2,
                n2,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        let conv3 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_mid_static_convolution::<N, P3>(
                long_top,
                short,
                &mut res3,
                n3,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        rayon::join(conv1, || rayon::join(conv2, conv3));
    } else {
        let mut buf_scratch = [0; N];
        let mut ntt_scratch = [0; N];
        ntt_mid_static_convolution::<N, P1>(
            long_top,
            short,
            &mut res1,
            n1,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
        ntt_mid_static_convolution::<N, P2>(
            long_top,
            short,
            &mut res2,
            n2,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
        ntt_mid_static_convolution::<N, P3>(
            long_top,
            short,
            &mut res3,
            n3,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
    }

    let res1_len = n1.min(N);
    let res2_len = n2.min(N);
    let res3_len = n3.min(N);
    let mut of = ntt_mid_accumulate_scaled(
        &mut res1[..res1_len],
        &mut res2[..res2_len],
        &mut res3[..res3_len],
        out,
        sz - 1 - z,
        n1,
        n2,
        n3,
    );
    let (c0, c1) = middle_correction(long, short);
    add_mid_overflow(&mut of, add_prim(out, c0));
    add_mid_overflow(&mut of, add_prim(&mut out[1..], c1));
    return of;
}

pub(crate) enum DynMidDispatch {
    School,
    FFT,
    NTT,
}

fn dyn_mid_dispatch(n: usize) -> DynMidDispatch {
    dyn_mid_dispatch_at(n, k::mul::fft_mid())
}

// mid_mul_dyn's choice for a short operand of n limbs at an explicit `mul.fft_mid` (invariant 6).
pub(crate) fn dyn_mid_dispatch_at(n: usize, fft_mid: usize) -> DynMidDispatch {
    return if n < fft_mid {
        DynMidDispatch::School
    } else if n <= FFT_MID_16BIT_CUTOFF {
        DynMidDispatch::FFT
    } else {
        DynMidDispatch::NTT
    };
}

pub fn mid_mul_dyn(long: &[u64], short: &[u64], out: &mut [u64]) -> (u64, u64) {
    debug_assert_mid_mul_shape(long, short, out);
    match dyn_mid_dispatch(short.len()) {
        DynMidDispatch::School => mid_mul_buf(long, short, out),
        DynMidDispatch::FFT => fft_mid_mul(long, short, out),
        DynMidDispatch::NTT => ntt_mid_mul_dyn(long, short, out),
    }
}

pub fn mid_mul_static<const N: usize>(long: &[u64], short: &[u64], out: &mut [u64]) -> (u64, u64) {
    debug_assert_mid_mul_shape(long, short, out);
    if static_mid_ntt(short.len(), k::mul::ntt_mid()) {
        ntt_mid_mul_static::<N>(long, short, out)
    } else {
        mid_mul_buf(long, short, out)
    }
}

// Whether mid_mul_static takes the NTT for a short operand of n limbs at an explicit `mul.ntt_mid`
// (invariant 6).
pub fn static_mid_ntt(n: usize, ntt_mid: usize) -> bool {
    n >= ntt_mid
}

//FAST EXPONENTIATION
// does an integer powers via log(n) squaring s
// pow's binary form is `1 r_{k-1} ... r_0` (k = pow.ilog2() rest bits below
// the leading one). reverse_pow packs those rest bits in reverse (bit i holds
// r_{k-1-i}, the order powi_dyn_core/powi_static_core consume them in via
// repeated `& 1` then `>> 1`) with a sentinel 1-bit at position k so the
// recursion's `reverse_pow == 1` check fires exactly once all k rest bits are
// consumed.
fn reverse_pow(pow: usize) -> usize {
    let k = pow.ilog2() as usize;
    if k == 0 {
        return 1;
    }
    let rest = pow - (1 << k);
    let reversed_rest = rest.reverse_bits() >> (usize::BITS as usize - k);
    reversed_rest | (1 << k)
}

pub fn powi_sz(buf: &[u64], pow: usize) -> (usize, usize) {
    if pow == 0 {
        // x^0 = 1 always needs exactly one limb, independent of buf's size.
        return (1, 1);
    }
    let len = buf_len(buf);
    if len == 0 {
        return (0, 0);
    }
    let bit_length = (len - 1) * 64 + buf[len - 1].ilog2() as usize;
    (
        bit_length.saturating_mul(pow).div_ceil(64),
        bit_length
            .saturating_add(1)
            .saturating_mul(pow)
            .div_ceil(64),
    )
}

fn powi_dyn_core(buf: &[u64], reverse_pow: usize, src: &mut [u64], dst: &mut [u64]) {
    if reverse_pow == 1 {
        return;
    }
    let src_len = buf_len(src);
    sqr_dyn(&src[..src_len], dst);
    if reverse_pow & 1 == 1 {
        let dst_len = buf_len(dst);
        mul_dyn(&dst[..dst_len], buf, src);
        powi_dyn_core(buf, reverse_pow >> 1, src, dst);
        return;
    }
    powi_dyn_core(buf, reverse_pow >> 1, dst, src);
}

pub fn powi_dyn_entry(buf: &[u64], pow: usize, out: &mut [u64]) {
    out.fill(0);
    if pow == 0 {
        assert!(!out.is_empty(), "output is not large enough for x^0");
        out[0] = 1;
        return;
    }
    let buf = &buf[..buf_len(buf)];
    if buf.is_empty() {
        return;
    }
    let reverse_pow = reverse_pow(pow);
    let mut scratch = ScratchGuard::acquire();
    let tmp = scratch.get(out.len());
    tmp.fill(0);
    // powi_*_core leaves the running result in its `src` argument once
    // reverse_pow's k rest-bits are consumed (each 0 bit swaps src/dst, each
    // 1 bit keeps the same src). The number of swaps is
    // k - rest.count_ones() = pow.ilog2() - (pow.count_ones() - 1), which is
    // even iff pow.ilog2() + pow.count_ones() is odd — that's when `out`
    // needs to start in the src slot for the result to land back in `out`.
    if (pow.ilog2() + pow.count_ones()) % 2 == 1 {
        out[..buf.len()].copy_from_slice(buf);
        powi_dyn_core(buf, reverse_pow, out, tmp);
    } else {
        tmp[..buf.len()].copy_from_slice(buf);
        powi_dyn_core(buf, reverse_pow, tmp, out);
    }
}

fn powi_static_core<const N: usize>(
    buf: &[u64],
    reverse_pow: usize,
    src: &mut [u64],
    dst: &mut [u64],
) {
    if reverse_pow == 1 {
        return;
    }
    let src_len = buf_len(src);
    sqr_static::<N>(&src[..src_len], dst);
    if reverse_pow & 1 == 1 {
        let dst_len = buf_len(dst);
        mul_static::<N>(&dst[..dst_len], buf, src);
        powi_static_core::<N>(buf, reverse_pow >> 1, src, dst);
        return;
    }
    powi_static_core::<N>(buf, reverse_pow >> 1, dst, src)
}

pub fn powi_static_entry<const N: usize>(buf: &[u64], pow: usize, out: &mut [u64]) {
    let (min, _) = powi_sz(buf, pow);
    let capacity = N.min(out.len());
    debug_assert!(
        capacity >= min,
        "power exceeds output capacity: {}",
        capacity
    );
    out.fill(0);
    if pow == 0 {
        debug_assert!(!out.is_empty(), "output is not large enough for x^0");
        out[0] = 1;
        return;
    }
    let buf = &buf[..buf_len(buf)];
    if buf.is_empty() {
        return;
    }
    let reverse_pow = reverse_pow(pow);
    let mut tmp = [0; N];
    // See the parity comment in powi_dyn_entry.
    if (pow.ilog2() + pow.count_ones()) % 2 == 1 {
        out[..buf.len()].copy_from_slice(buf);
        powi_static_core::<N>(buf, reverse_pow, out, &mut tmp)
    } else {
        tmp[..buf.len()].copy_from_slice(buf);
        powi_static_core::<N>(buf, reverse_pow, &mut tmp, out)
    }
}
