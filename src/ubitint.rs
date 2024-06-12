#![allow(dead_code)]

use core::fmt;
use std::arch::asm;
use std::cmp::Ordering::*;
use std::ops::*;

// adds values with carry and propogates carry on aarch64
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn add_with_carry_aarch64(l: &mut usize, s: usize, c: &mut u8) {
    asm!(
        "adds {c}, {c}, #0xFFFFFFFFFFFFFFFF", // c -> cf
        "adcs {l}, {l}, {s}", // l+s+cf -> l , updates cf
        "cset {c}, cs", // cf -> c
        l = inout(reg) *l,
        s = in(reg) s,
        c = inout(reg) *c,
        options(nostack)
    );
}

// adds values with carry and propogates carry on x86
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn add_with_carry_x86_64(l: &mut usize, s: usize, c: &mut u8) {
    asm!(
        "add {c} , 0xFF", // carry -> cf
        "adc {l}, {s}", // l+s+cf -> l , updates cf
        "setc {c}", // cf -> c
        c = inout(reg_byte) *c,
        l = inout(reg) *l,
        s = in(reg) s,
        options(nostack)
    );
}

//propogates carry on aarch64
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn add_carry_aarch64(l: &mut usize, c: &mut usize) {
    asm!(
        "adds {l}, {l}, {c}", // l + c -> l
        "cset {c}, cs",       // cf -> c
        l = inout(reg) * l,
        c = inout(reg) * c,
        options(nostack)
    );
}

//propogates carry on x86
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn add_carry_x86_64(l: &mut usize, c: &mut u8) {
    asm!(
        "add {l}, {tmp}", // l + c -> l
        "setc {c}",     // cf -> c
        c = inout(reg_byte) * c,
        tmp = in(reg) *c as usize,
        l = inout(reg) * l,
        options(nostack)
    );
}

//the first input must be larger then the second because of the assymetry
//of the input being mutable and non-mutable
#[inline]
pub fn add_ubi(longer: &mut Vec<usize>, shorter: &[usize]) {
    unsafe {
        let mut c: u8 = 0;
        for (s, l) in shorter.iter().zip(longer.iter_mut()) {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(l, *s, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(l, *s, &mut c);
        }

        for l in longer.iter_mut().skip(shorter.len()) {
            if c == 0 {
                break;
            }

            #[cfg(target_arch = "aarch64")]
            add_carry_aarch64(l, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_carry_x86_64(l, &mut c);
        }

        if c == 1 {
            longer.push(1);
        }
    }
}

//adds then propogates the carry on aarch64
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn add_prop_carry_aarch64(l: &mut usize, s: usize) -> u8 {
    let c: u8;
    asm!(
        "adds {l}, {l}, {s}", // l + s -> l
        "cset {c}, cs", // cf -> c
        l = inout(reg) *l,
        s = in(reg) s,
        c = out(reg) c,
        options(nostack)
    );

    c
}

//adds then propogates the carry on x86
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn add_prop_carry_x86_64(l: &mut usize, s: usize) -> u8 {
    let c: u8;
    asm!(
        "add {l}, {s}", // l+s -> l
        "setc {c}", // cf -> c
        l = inout(reg) *l,
        s = in(reg) s,
        c = out(reg_byte) c,
        options(nostack)
    );

    c
}

// simplified algorithm for adding a primitive type less then or equal to usize
#[inline]
fn add_prim(ubi: &mut Vec<usize>, prim: usize) {
    unsafe {
        #[cfg(target_arch = "aarch64")]
        let mut c: u8 = add_prop_carry_aarch64(&mut ubi[0], prim);

        #[cfg(target_arch = "x86_64")]
        let mut c: u8 = add_prop_carry_x86_64(&mut ubi[0], prim);

        if c != 0 {
            for l in &mut ubi[1..] {
                if c == 0 {
                    break;
                }

                #[cfg(target_arch = "aarch64")]
                add_carry_aarch64(l, &mut c);

                #[cfg(target_arch = "x86_64")]
                add_carry_x86_64(l, &mut c);
            }

            if c == 1 {
                ubi.push(1)
            };
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn sub_carry_aarch64(l: &mut usize, c: &mut u8) {
    asm!(
        "eor {c}, {c}, #1",   //flip carry
        "subs {l}, {l}, {c}", // subtract fliped carry
        "cset {c}, cs",       // set carry
        l = inout(reg) * l,
        c = inout(reg) * c,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn sub_carry_x86_64(l: &mut usize, c: &mut u8) {
    asm!(
        "xor {tmp}, 1",   // flip carry
        "sub {l}, {tmp}", // subtract flipped carry
        "setc {c}",     // set carry
        l = inout(reg) *l,
        tmp = in(reg) *c as usize,
        c = inout(reg_byte) *c,
        options(nostack)
    );
}

//the rhs is the subtractor and lhs is the minued and is implcitly
//assumed that lhs > rhs and should be ensured before calling otherwise
//this will lead to undefined behavior
#[inline]
pub fn sub_ubi(lhs: &mut Vec<usize>, rhs: &[usize]) {
    unsafe {
        let mut c: u8 = 1;
        for (r, l) in rhs.iter().zip(lhs.iter_mut()) {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(l, !*r, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(l, !*r, &mut c);
        }

        if c == 0 {
            for l in lhs.iter_mut().skip(rhs.len()) {
                #[cfg(target_arch = "aarch64")]
                sub_carry_aarch64(l, &mut c);

                #[cfg(target_arch = "x86_64")]
                sub_carry_x86_64(l, &mut c);
            }
        }
    }

    if let Some(idx) = lhs.iter().rposition(|&x| x != 0) {
        lhs.truncate(idx + 1);
    } else {
        lhs.clear();
    }
}

//simplified algorithm for subtracting primitive type
#[inline]
pub fn sub_prim(lhs: &mut Vec<usize>, rhs: usize) {
    unsafe {
        #[cfg(target_arch = "aarch64")]
        let mut c: u8 = add_prop_carry_aarch64(&mut lhs[0], !rhs + 1);

        #[cfg(target_arch = "x86_64")]
        let mut c: u8 = add_prop_carry_x86_64(&mut lhs[0], !rhs + 1);

        if c == 0 {
            for l in &mut lhs[1..] {
                #[cfg(target_arch = "aarch64")]
                sub_carry_aarch64(l, &mut c);

                #[cfg(target_arch = "x86_64")]
                sub_carry_x86_64(l, &mut c);
            }
        }
    }

    if let Some(idx) = lhs.iter().rposition(|&x| x != 0) {
        lhs.truncate(idx + 1);
    } else {
        lhs.clear();
    }
}

//multiplying a UBitInt by all primitive types
#[inline]
fn mul_prim(ubi: &[usize], prim: u128) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(ubi.len() + 1);
    let mut carry: u128 = 0;

    for elem in ubi {
        let val = ((*elem as u128) * prim).wrapping_add(carry);
        out.push(val as usize);
        carry = val >> 64;
    }

    if carry > 0 {
        out.push(carry as usize);
    }

    out
}

//multiplying a UBitInt by a smaller UBitInt
#[inline]
fn mul_ubi_short(a: &[usize], b: &[usize]) -> Vec<usize> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let a_len = a.len() - 1;
    let b_len = b.len() - 1;

    let mut out: Vec<usize> = Vec::with_capacity(a_len + b_len + 1);

    let mask = (1u128 << 64) - 1;

    let mut carry: u128 = 0;
    for k in 0..=(a_len + b_len) {
        let mut term: u128 = carry;
        carry = 0;
        for j in k.saturating_sub(b_len)..=std::cmp::min(a_len, k) {
            let val = (a[j] as u128) * (b[k - j] as u128);
            term += val & mask;
            carry += val >> 64;
        }
        carry += term >> 64;
        out.push(term as usize);
    }

    if carry > 0 {
        out.push(carry as usize);
    }

    out
}

#[inline]
fn mul_ubi(a: &[usize], b: &[usize]) -> Vec<usize> {
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let half_len = long.len() / 2;
    if short.len() == 1 {
        return mul_prim(long, short[0] as u128);
    }
    if short.len() <= half_len || short.len() <= 64 {
        return mul_ubi_short(long, short);
    }

    let l1 = &long[half_len..];
    let l0 = &long[..half_len];
    let s1 = &short[half_len..];
    let s0 = &short[..half_len];

    let z2 = mul_ubi(l1, s1);
    let z0 = mul_ubi(l0, s0);

    let mut l1_vec = l1.to_vec();
    let mut s0_vec = s0.to_vec();

    add_ubi(&mut l1_vec, l0);
    add_ubi(&mut s0_vec, s1);
    let mut z1 = mul_ubi(&l1_vec, &s0_vec);
    sub_ubi(&mut z1, &z2);
    sub_ubi(&mut z1, &z0);

    let mut out2: Vec<usize> = vec![0; 2 * half_len];
    out2.extend(z2);
    let mut out1: Vec<usize> = vec![0; half_len];
    out1.extend(z1);
    add_ubi(&mut out2, &out1);
    add_ubi(&mut out2, &z0);

    out2
}

#[inline]
pub fn cmp_ubi(lhs: &[usize], rhs: &[usize]) -> std::cmp::Ordering {
    match lhs.len().cmp(&rhs.len()) {
        Greater => return Greater,
        Less => return Less,
        Equal => {
            for (l, r) in lhs.iter().zip(rhs.iter()).rev() {
                if l > r {
                    return Greater;
                } else if l < r {
                    return Less;
                }
            }

            Equal
        }
    }
}

#[inline]
fn log2_ubi(ubi: &[usize]) -> u128 {
    (64 - ubi[ubi.len() - 1].leading_zeros() as usize - 1) as u128 + ((ubi.len() - 1) as u128) * 64
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn shr_carry_aarch64(e: &mut usize, c: &mut usize, rem: usize, mv_sz: usize) {
    asm!(
        "lsl {tmp}, {e}, {ms}", // put the last bits of the e into tmp
        "lsr {e}, {e}, {r}", // shift e by rem
        "orr {e}, {e}, {c}", // put the last bits of previous e at the start of e
        "mov {c}, {tmp}", // put tmp into carry
        e = inout(reg) *e,
        c = inout(reg) *c,
        r = in(reg) rem,
        ms = in(reg) mv_sz,
        tmp = out(reg) _,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn shr_carry_x86_64(e: &mut usize, c: &mut usize, rem: u8, mv_sz: u8) {
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
        r = in(reg_byte) rem,
        ms = in(reg_byte) mv_sz,
        tmp = out(reg) _,
        out("cl") _ ,
        options(nostack)
    );
}

#[inline]
fn shr_ubi(ubi: &mut Vec<usize>, sh: u128) {
    if sh == 0 {
        return;
    }
    let div = (sh / 64) as usize;
    let rem = (sh % 64) as u8;
    let mv_sz = 64 - rem;
    ubi.drain(0..div);

    unsafe {
        if ubi.len() > 0 && rem != 0 {
            let mut carry = 0_usize;
            for elem in ubi.iter_mut().rev() {
                #[cfg(target_arch = "aarch64")]
                shr_carry_aarch64(elem, &mut carry, rem, mv_sz);

                #[cfg(target_arch = "x86_64")]
                shr_carry_x86_64(elem, &mut carry, rem, mv_sz);
            }

            if ubi[ubi.len() - 1] == 0 {
                ubi.pop();
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn shl_carry_aarch64(e: &mut usize, c: &mut usize, rem: u8, mv_sz: u8) {
    asm!(
        "lsr {tmp}, {e}, {ms}", // put the last bits of the e into tmp
        "lsl {e}, {e}, {r}", // shift e by rem
        "orr {e}, {e}, {c}", // put the last bits of previous e at the start of e
        "mov {c}, {tmp}", // put tmp into carry
        e = inout(reg) *e,
        c = inout(reg) *c,
        r = in(reg) rem,
        ms = in(reg) mv_sz,
        tmp = out(reg) _,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn shl_carry_x86_64(e: &mut usize, c: &mut usize, rem: u8, mv_sz: u8) {
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
        r = in(reg_byte) rem,
        ms = in(reg_byte) mv_sz,
        tmp = out(reg) _,
        out("cl") _ ,
        options(nostack)
    );
}

#[inline]
pub fn shl_ubi(ubi: &mut Vec<usize>, sh: u128) {
    if sh == 0 {
        return;
    }
    let div = (sh / 64) as usize;
    let rem = (sh % 64) as u8;
    let mv_sz = 64 - rem;

    let len = ubi.len();
    ubi.resize(len + div, 0);
    if div != 0 {
        ubi.copy_within(0..len, div);
        ubi[0..div].fill(0);
    }

    unsafe {
        if rem != 0 {
            let mut carry = 0_usize;
            for i in div..len + div {
                #[cfg(target_arch = "aarch64")]
                shl_carry_aarch64(&mut ubi[i], &mut carry, rem, mv_sz);

                #[cfg(target_arch = "x86_64")]
                shl_carry_x86_64(&mut ubi[i], &mut carry, rem, mv_sz);
            }

            if carry > 0 {
                ubi.push(carry);
            }
        }
    }
}

#[inline]
pub fn div_ubi(n: &mut Vec<usize>, d: &Vec<usize>) -> Vec<usize> {
    match cmp_ubi(n, d) {
        Less => return vec![],
        Equal => {
            *n = vec![];
            return vec![1];
        }
        Greater => (),
    }

    let n_log2 = log2_ubi(n);
    let d_log2 = log2_ubi(d);

    let mut sub = d.clone();
    let mut intit_log2 = n_log2 - d_log2;
    shl_ubi(&mut sub, intit_log2);

    if cmp_ubi(&sub, n) == Greater {
        shr_ubi(&mut sub, 1);
        intit_log2 -= 1;
    }

    let mut add: Vec<usize> = vec![0; (intit_log2 / 64) as usize];
    add.push(1_usize << (intit_log2 % 64) as usize);
    sub_ubi(n, &sub);
    let mut div = add.clone();

    while cmp_ubi(n, d) != Less {
        let log2 = log2_ubi(&sub) - log2_ubi(&n);
        shr_ubi(&mut sub, log2);
        shr_ubi(&mut add, log2);
        if cmp_ubi(&sub, &n) == Greater {
            shr_ubi(&mut sub, 1);
            shr_ubi(&mut add, 1);
        }
        sub_ubi(n, &sub);
        add_ubi(&mut div, &add);
    }

    div
}

#[derive(Debug, Clone, Default)]
pub struct UBitInt {
    pub data: Vec<usize>,
}

//converting primitives to UBitInt

pub trait ToUsizeVec {
    fn to_usize_vec(self) -> Vec<usize>;
}

impl ToUsizeVec for u128 {
    #[inline]
    fn to_usize_vec(self) -> Vec<usize> {
        let lower = (self & 0xFFFFFFFFFFFFFFFF) as usize;
        let upper = (self >> 64) as usize;
        vec![lower, upper]
    }
}

impl ToUsizeVec for u64 {
    #[inline]
    fn to_usize_vec(self) -> Vec<usize> {
        vec![self as usize]
    }
}

impl ToUsizeVec for usize {
    fn to_usize_vec(self) -> Vec<usize> {
        vec![self]
    }
}

macro_rules! impl_to_usize_vec_uprim {
    ($($t:ty),*) => {
        $(
            impl ToUsizeVec for $t{
                #[inline]
                fn to_usize_vec(self) -> Vec<usize>{
                    vec![self as usize]
                }
            }
        )*
    };
}

impl_to_usize_vec_uprim!(u32, u16, u8);

macro_rules! impl_to_usize_vec_iprim {
    ($($t:ty),*) => {
        $(
            impl ToUsizeVec for $t{
                #[inline]
                fn to_usize_vec(self) -> Vec<usize>{
                    self.unsigned_abs().to_usize_vec()
                }
            }
        )*
    };
}

impl_to_usize_vec_iprim!(i128, i64, isize, i32, i16, i8);

impl UBitInt {
    #[inline]
    pub fn get_data(&self) -> Vec<usize> {
        return self.data.clone();
    }

    #[inline]
    pub fn from<T: ToUsizeVec>(val: T) -> UBitInt {
        let mut data = val.to_usize_vec();
        if let Some(idx) = data.iter().rposition(|&x| x != 0) {
            data.truncate(idx + 1);
        } else {
            data.clear();
        }
        UBitInt { data }
    }

    #[inline]
    pub fn make(data: Vec<usize>) -> UBitInt {
        UBitInt { data }
    }

    #[inline]
    pub fn to(&self) -> Result<u128, String> {
        if self.data.is_empty() {
            return Ok(0);
        } else if self.data.len() == 1 {
            return Ok(self.data[0] as u128);
        } else if self.data.len() == 2 {
            let mut out = self.data[0] as u128;
            out |= (self.data[1] as u128) << 64;
            return Ok(out);
        } else {
            return Err("UBitInt is to large to be converted to a primitive".to_string());
        }
    }

    #[inline]
    pub fn from_str(num: &str) -> Result<UBitInt, String> {
        let mut out = UBitInt::from(0_usize);
        let mut base = UBitInt::from(1_usize);
        for c in num.chars().rev() {
            if let Some(digit) = c.to_digit(10) {
                out += digit * &base;
                base *= 10_usize;
            } else {
                return Err("Malformed string to convert to a UBitInt".to_string());
            }
        }

        Ok(out)
    }

    #[inline]
    pub fn log2(&self) -> u128 {
        log2_ubi(&self.data)
    }

    #[inline]
    pub fn mod2(&self) -> bool {
        (self.data[0] & 1) == 1
    }
}

impl fmt::Display for UBitInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ubi = self.clone();
        let mut out: String = "".to_string();
        while ubi.data.len() > 2 {
            let digit: UBitInt = &ubi % 10_usize;
            if let Ok(digit_str) = digit.to() {
                out = format!("{}{}", digit_str.to_string(), out);
            }
            ubi /= 10_usize;
        }

        if let Ok(last_digits) = ubi.to() {
            out = format!("{}{}", last_digits.to_string(), out);
        }

        write!(f, "{}", out)?;
        Ok(())
    }
}

// implementing ordering for UBitInts

impl PartialEq for UBitInt {
    fn eq(&self, other: &Self) -> bool {
        if self.data.len() != other.data.len() {
            return false;
        }

        for (l, r) in self.data.iter().zip(other.data.iter()) {
            if l != r {
                return false;
            }
        }

        return true;
    }
}

impl PartialEq<u128> for UBitInt {
    fn eq(&self, other: &u128) -> bool {
        if self.data.is_empty() {
            return *other == 0;
        }
        if self.data.len() > 2 {
            return false;
        }

        let rhs = UBitInt::from(*other);
        if rhs.data.len() != self.data.len() {
            return false;
        }

        for (l, r) in self.data.iter().zip(rhs.data.iter()) {
            if l != r {
                return false;
            }
        }

        return true;
    }
}

impl PartialEq<u64> for UBitInt {
    fn eq(&self, other: &u64) -> bool {
        if self.data.is_empty() {
            return *other == 0;
        }
        if self.data.len() != 1 {
            return false;
        }

        return self.data[0] == *other as usize;
    }
}

impl PartialEq<usize> for UBitInt {
    fn eq(&self, other: &usize) -> bool {
        if self.data.is_empty() {
            return *other == 0;
        }
        if self.data.len() != 1 {
            return false;
        }
        return self.data[0] == *other;
    }
}

macro_rules! impl_peq_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl PartialEq<$t> for UBitInt {
                fn eq(&self, other: &$t) -> bool {
                    if self.data.is_empty() {
                        return *other == 0;
                    }
                    if self.data.len() != 1 {
                        return false;
                    }
                    return self.data[0] == *other as usize;
                }
            }
        )*
    };
}

impl_peq_ubi_prim!(u32, u16, u8);

impl PartialOrd for UBitInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.data.len().cmp(&other.data.len()) {
            Greater => return Some(Greater),
            Less => return Some(Less),
            Equal => {
                for (l, r) in self.data.iter().zip(other.data.iter()).rev() {
                    if l > r {
                        return Some(Greater);
                    } else if r > l {
                        return Some(Less);
                    }
                }

                Some(Equal)
            }
        }
    }
}

impl PartialOrd<u128> for UBitInt {
    fn partial_cmp(&self, other: &u128) -> Option<std::cmp::Ordering> {
        if self.data.len() > 2 {
            return Some(Greater);
        }

        let rhs = UBitInt::from(*other).data;
        for (l, r) in self.data.iter().zip(rhs.iter()).rev() {
            if l > r {
                return Some(Greater);
            } else if r > l {
                return Some(Less);
            }
        }

        Some(Equal)
    }
}

impl PartialOrd<usize> for UBitInt {
    fn partial_cmp(&self, other: &usize) -> Option<std::cmp::Ordering> {
        match self.data.len().cmp(&1_usize) {
            Greater => return Some(Greater),
            Less => return Some(Less),
            Equal => return Some(self.data[0].cmp(other)),
        }
    }
}

macro_rules! impl_pord_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl PartialOrd<$t> for UBitInt {
                fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering> {
                    match self.data.len().cmp(&1_usize) {
                        Greater => return Some(Greater),
                        Less => return Some(Less),
                        Equal => return Some(self.data[0].cmp(&(*other as usize))),
                    }
                }
            }
        )*
    };
}

impl_pord_ubi_prim!(u64, u32, u16, u8);

impl Eq for UBitInt {}

impl Ord for UBitInt {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.data.len().cmp(&other.data.len()) {
            Greater => return Greater,
            Less => return Less,
            Equal => {
                for (l, r) in self.data.iter().zip(other.data.iter()).rev() {
                    if l > r {
                        return Greater;
                    } else if r > l {
                        return Less;
                    }
                }

                Equal
            }
        }
    }
}

// Implementing Add for UBitInt and all unisghned primitives

impl Add for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (mut longer, shorter) = if self.data.len() > rhs.data.len() {
            (self.data, rhs.data)
        } else {
            (rhs.data, self.data)
        };

        add_ubi(&mut longer, &shorter);
        UBitInt { data: longer }
    }
}

impl Add for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (mut longer, shorter) = if self.data.len() > rhs.data.len() {
            (self.data.clone(), &rhs.data)
        } else {
            (rhs.data.clone(), &self.data)
        };

        add_ubi(&mut longer, shorter);
        UBitInt { data: longer }
    }
}

impl Add<&UBitInt> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: &UBitInt) -> Self::Output {
        let (mut longer, shorter) = if self.data.len() > rhs.data.len() {
            (self.data, &rhs.data)
        } else {
            (rhs.data.clone(), &self.data)
        };

        add_ubi(&mut longer, &shorter);
        UBitInt { data: longer }
    }
}

impl Add<UBitInt> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: UBitInt) -> Self::Output {
        let (mut longer, shorter) = if self.data.len() > rhs.data.len() {
            (self.data.clone(), &rhs.data)
        } else {
            (rhs.data, &self.data)
        };

        add_ubi(&mut longer, &shorter);
        UBitInt { data: longer }
    }
}

impl Add<u128> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: u128) -> UBitInt {
        let prim = UBitInt::from(rhs);
        self + prim
    }
}

impl Add<u128> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: u128) -> UBitInt {
        let prim = UBitInt::from(rhs);
        self + prim
    }
}

impl Add<UBitInt> for u128 {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: UBitInt) -> UBitInt {
        rhs + self
    }
}

impl Add<&UBitInt> for u128 {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: &UBitInt) -> UBitInt {
        rhs + self
    }
}

impl Add<u64> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: u64) -> UBitInt {
        let mut ubi = self.data;
        add_prim(&mut ubi, rhs as usize);
        UBitInt { data: ubi }
    }
}

impl Add<u64> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: u64) -> UBitInt {
        let mut ubi = self.data.clone();
        add_prim(&mut ubi, rhs as usize);
        UBitInt { data: ubi }
    }
}

impl Add<UBitInt> for u64 {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: UBitInt) -> UBitInt {
        rhs + self
    }
}

impl Add<&UBitInt> for u64 {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: &UBitInt) -> UBitInt {
        rhs + self
    }
}

impl Add<usize> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: usize) -> Self::Output {
        let mut ubi = self.data;
        add_prim(&mut ubi, rhs);
        UBitInt { data: ubi }
    }
}

impl Add<usize> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: usize) -> Self::Output {
        let mut ubi = self.data.clone();
        add_prim(&mut ubi, rhs);
        UBitInt { data: ubi }
    }
}

impl Add<UBitInt> for usize {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: UBitInt) -> Self::Output {
        rhs + self
    }
}

impl Add<&UBitInt> for usize {
    type Output = UBitInt;

    #[inline]
    fn add(self, rhs: &UBitInt) -> Self::Output {
        rhs + self
    }
}

macro_rules! impl_add_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl Add<$t> for UBitInt {
                type Output = UBitInt;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    let mut ubi = self.data;
                    add_prim(&mut ubi, rhs as usize);
                    UBitInt { data: ubi }
                }
            }

            impl Add<$t> for &UBitInt {
                type Output = UBitInt;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    let mut ubi = self.data.clone();
                    add_prim(&mut ubi, rhs as usize);
                    UBitInt { data: ubi }
                }
            }

            impl Add<UBitInt> for $t {
                type Output = UBitInt;

                #[inline]
                fn add(self, rhs: UBitInt) -> Self::Output {
                    rhs + self
                }
            }

            impl Add<&UBitInt> for $t {
                type Output = UBitInt;

                #[inline]
                fn add(self, rhs: &UBitInt) -> Self::Output {
                    rhs + self
                }
            }
        )*
    };
}

impl_add_ubi_prim!(u32, u16, u8);

//implementing add assign for all UBitInt and unsighned primitives

impl AddAssign for UBitInt {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        if self.data.len() > rhs.data.len() {
            add_ubi(&mut self.data, &rhs.data);
        } else {
            let mut longer = rhs.data;
            add_ubi(&mut longer, &self.data);
            self.data = longer;
        }
    }
}

impl AddAssign<&UBitInt> for UBitInt {
    #[inline]
    fn add_assign(&mut self, rhs: &UBitInt) {
        if self.data.len() > rhs.data.len() {
            add_ubi(&mut self.data, &rhs.data);
        } else {
            let mut longer = rhs.data.clone();
            add_ubi(&mut longer, &self.data);
            self.data = longer;
        }
    }
}

impl AddAssign<u128> for UBitInt {
    #[inline]
    fn add_assign(&mut self, rhs: u128) {
        let prim = UBitInt::from(rhs);
        *self += prim;
    }
}

impl AddAssign<u64> for UBitInt {
    #[inline]
    fn add_assign(&mut self, rhs: u64) {
        add_prim(&mut self.data, rhs as usize);
    }
}

impl AddAssign<usize> for UBitInt {
    #[inline]
    fn add_assign(&mut self, rhs: usize) {
        add_prim(&mut self.data, rhs);
    }
}

macro_rules! impl_add_assign_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl AddAssign<$t> for UBitInt {
                #[inline]
                fn add_assign(&mut self, rhs: $t) {
                    add_prim(&mut self.data, rhs as usize);
                }
            }
        )*
    };
}

impl_add_assign_ubi_prim!(u32, u16, u8);

// implementing Sub for UBitInt and all unisghned primitives

impl Sub for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut lhs = self.data;
        sub_ubi(&mut lhs, &rhs.data);
        UBitInt { data: lhs }
    }
}

impl Sub for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut lhs = self.data.clone();
        sub_ubi(&mut lhs, &rhs.data);
        UBitInt { data: lhs }
    }
}

impl Sub<&UBitInt> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: &UBitInt) -> Self::Output {
        let mut lhs = self.data;
        sub_ubi(&mut lhs, &rhs.data);
        UBitInt { data: lhs }
    }
}

impl Sub<UBitInt> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: UBitInt) -> Self::Output {
        let mut lhs = self.data.clone();
        sub_ubi(&mut lhs, &rhs.data);
        UBitInt { data: lhs }
    }
}

impl Sub<u128> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: u128) -> Self::Output {
        let mut lhs = self.data;
        sub_ubi(&mut lhs, &UBitInt::from(rhs).data);
        UBitInt { data: lhs }
    }
}

impl Sub<u128> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: u128) -> Self::Output {
        let mut lhs = self.data.clone();
        sub_ubi(&mut lhs, &UBitInt::from(rhs).data);
        UBitInt { data: lhs }
    }
}

impl Sub<UBitInt> for u128 {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: UBitInt) -> Self::Output {
        let mut lhs = UBitInt::from(self).data;
        sub_ubi(&mut lhs, &rhs.data);
        UBitInt { data: lhs }
    }
}

impl Sub<&UBitInt> for u128 {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: &UBitInt) -> Self::Output {
        let mut lhs = UBitInt::from(self).data;
        sub_ubi(&mut lhs, &rhs.data);
        UBitInt { data: lhs }
    }
}

impl Sub<u64> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: u64) -> Self::Output {
        let mut lhs = self.data;

        sub_prim(&mut lhs, rhs as usize);

        UBitInt { data: lhs }
    }
}

impl Sub<u64> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: u64) -> Self::Output {
        let mut lhs = self.data.clone();

        sub_prim(&mut lhs, rhs as usize);

        UBitInt { data: lhs }
    }
}

impl Sub<UBitInt> for u64 {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: UBitInt) -> Self::Output {
        UBitInt {
            data: vec![(self as usize) - rhs.data[0]],
        }
    }
}

impl Sub<&UBitInt> for u64 {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: &UBitInt) -> Self::Output {
        
            UBitInt {
                data: vec![(self as usize) - rhs.data[0]],
            }
        
    }
}

impl Sub<usize> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: usize) -> Self::Output {
        let mut lhs = self.data;
        sub_prim(&mut lhs, rhs);
        UBitInt { data: lhs }
    }
}

impl Sub<usize> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: usize) -> Self::Output {
        let mut lhs = self.data.clone();
        sub_prim(&mut lhs, rhs);
        UBitInt { data: lhs }
    }
}

impl Sub<UBitInt> for usize {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: UBitInt) -> Self::Output {
        UBitInt {
            data: vec![self - rhs.data[0]],
        }
    }
}

impl Sub<&UBitInt> for usize {
    type Output = UBitInt;

    #[inline]
    fn sub(self, rhs: &UBitInt) -> Self::Output {
        UBitInt {
            data: vec![self - rhs.data[0]],
        }
    }
}

macro_rules! impl_sub_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl Sub<$t> for UBitInt {
                type Output = UBitInt;

                #[inline]
                fn sub(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    sub_prim(&mut lhs, rhs as usize);
                    UBitInt { data: lhs }
                }
            }

            impl Sub<$t> for &UBitInt {
                type Output = UBitInt;

                #[inline]
                fn sub(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data.clone();
                    sub_prim(&mut lhs, rhs as usize);
                    UBitInt { data: lhs }
                }
            }

            impl Sub<UBitInt> for $t {
                type Output = UBitInt;

                #[inline]
                fn sub(self, rhs: UBitInt) -> Self::Output {
                    UBitInt {
                        data: vec![(self as usize) - rhs.data[0]],
                    }
                }
            }

            impl Sub<&UBitInt> for $t {
                type Output = UBitInt;

                #[inline]
                fn sub(self, rhs: &UBitInt) -> Self::Output {
                    UBitInt {
                        data: vec![(self as usize) - rhs.data[0]],
                    }
                }
            }
        )*
    };
}

impl_sub_ubi_prim!(u32, u16, u8);

// implementing SubAssign for UBitInt and all unisghned primitives

impl SubAssign for UBitInt {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        sub_ubi(&mut self.data, &rhs.data);
    }
}

impl SubAssign<&UBitInt> for UBitInt {
    #[inline]
    fn sub_assign(&mut self, rhs: &UBitInt) {
        sub_ubi(&mut self.data, &rhs.data);
    }
}

impl SubAssign<u128> for UBitInt {
    #[inline]
    fn sub_assign(&mut self, rhs: u128) {
        let rhs_data = UBitInt::from(rhs).data;
        sub_ubi(&mut self.data, &rhs_data);
    }
}

impl SubAssign<u64> for UBitInt {
    #[inline]
    fn sub_assign(&mut self, rhs: u64) {
        
            sub_prim(&mut self.data, rhs as usize);

    }
}

impl SubAssign<usize> for UBitInt {
    #[inline]
    fn sub_assign(&mut self, rhs: usize) {
        sub_prim(&mut self.data, rhs);
    }
}

macro_rules! impl_sub_assign_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl SubAssign<$t> for UBitInt {
                #[inline]
                fn sub_assign(&mut self, rhs: $t) {
                    sub_prim(&mut self.data, rhs as usize);
                }
            }
        )*
    };
}

impl_sub_assign_ubi_prim!(u32, u16, u8);

// implementing mul for UbitInt and all unisghned primitives

impl Mul for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let data = if self.data.len() > rhs.data.len() {
            mul_ubi(&self.data, &rhs.data)
        } else {
            mul_ubi(&rhs.data, &self.data)
        };

        UBitInt { data }
    }
}

impl Mul for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let data = if self.data.len() > rhs.data.len() {
            mul_ubi(&self.data, &rhs.data)
        } else {
            mul_ubi(&rhs.data, &self.data)
        };

        UBitInt { data }
    }
}

impl Mul<&UBitInt> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn mul(self, rhs: &UBitInt) -> Self::Output {
        let data = if self.data.len() > rhs.data.len() {
            mul_ubi(&self.data, &rhs.data)
        } else {
            mul_ubi(&rhs.data, &self.data)
        };

        UBitInt { data }
    }
}

impl Mul<UBitInt> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn mul(self, rhs: UBitInt) -> Self::Output {
        let data = if self.data.len() > rhs.data.len() {
            mul_ubi(&self.data, &rhs.data)
        } else {
            mul_ubi(&rhs.data, &self.data)
        };

        UBitInt { data }
    }
}

impl Mul<u128> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn mul(self, rhs: u128) -> UBitInt {
        UBitInt {
            data: mul_prim(&self.data, rhs),
        }
    }
}

impl Mul<u128> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn mul(self, rhs: u128) -> UBitInt {
        UBitInt {
            data: mul_prim(&self.data, rhs),
        }
    }
}

impl Mul<UBitInt> for u128 {
    type Output = UBitInt;

    #[inline]
    fn mul(self, rhs: UBitInt) -> UBitInt {
        UBitInt {
            data: mul_prim(&rhs.data, self),
        }
    }
}

impl Mul<&UBitInt> for u128 {
    type Output = UBitInt;

    #[inline]
    fn mul(self, rhs: &UBitInt) -> UBitInt {
        UBitInt {
            data: mul_prim(&rhs.data, self),
        }
    }
}

macro_rules! impl_mul_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl Mul<$t> for UBitInt {
                type Output = UBitInt;

                #[inline]
                fn mul(self, rhs: $t) -> UBitInt {
                    UBitInt {
                        data: mul_prim(&self.data, rhs as u128),
                    }
                }
            }

            impl Mul<$t> for &UBitInt {
                type Output = UBitInt;

                #[inline]
                fn mul(self, rhs: $t) -> UBitInt {
                    UBitInt {
                        data: mul_prim(&self.data, rhs as u128),
                    }
                }
            }

            impl Mul<UBitInt> for $t {
                type Output = UBitInt;

                #[inline]
                fn mul(self, rhs: UBitInt) -> UBitInt {
                    UBitInt {
                        data: mul_prim(&rhs.data, self as u128),
                    }
                }
            }

            impl Mul<&UBitInt> for $t {
                type Output = UBitInt;

                #[inline]
                fn mul(self, rhs: &UBitInt) -> UBitInt {
                    UBitInt {
                        data: mul_prim(&rhs.data, self as u128),
                    }
                }
            }
        )*
    };
}

impl_mul_ubi_prim!(u64, usize, u32, u16, u8);

//implementing Mul Assign for UBitInt and all unsighned primitives

impl MulAssign for UBitInt {
    fn mul_assign(&mut self, rhs: Self) {
        self.data = mul_ubi(&self.data, &rhs.data);
    }
}

impl MulAssign<&UBitInt> for UBitInt {
    fn mul_assign(&mut self, rhs: &UBitInt) {
        self.data = mul_ubi(&self.data, &rhs.data);
    }
}

impl MulAssign<u128> for UBitInt {
    fn mul_assign(&mut self, rhs: u128) {
        self.data = mul_prim(&self.data, rhs);
    }
}

macro_rules! impl_mul_assign_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl MulAssign<$t> for UBitInt {
                fn mul_assign(&mut self, rhs: $t) {
                    self.data = mul_prim(&self.data, rhs as u128);
                }
            }
        )*
    };
}

impl_mul_assign_ubi_prim!(u64, usize, u32, u16, u8);

// implementing shl for UBitInt

impl Shl<u128> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn shl(self, rhs: u128) -> Self::Output {
        let mut lhs = self.data;
        shl_ubi(&mut lhs, rhs);
        UBitInt { data: lhs }
    }
}

impl Shl<u128> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn shl(self, rhs: u128) -> Self::Output {
        let mut lhs = self.data.clone();
        shl_ubi(&mut lhs, rhs);
        UBitInt { data: lhs }
    }
}

macro_rules! impl_shl_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl Shl<$t> for UBitInt {
                type Output = UBitInt;

                #[inline]
                fn shl(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    shl_ubi(&mut lhs, rhs as u128);
                    UBitInt { data: lhs }
                }
            }

            impl Shl<$t> for &UBitInt {
                type Output = UBitInt;

                #[inline]
                fn shl(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data.clone();
                    shl_ubi(&mut lhs, rhs as u128);
                    UBitInt { data: lhs }
                }
            }
        )*
    };
}

impl_shl_ubi_prim!(u64, usize, u32, u16, u8);

//implementing shl assign for UBitInt

impl ShlAssign<u128> for UBitInt {
    #[inline]
    fn shl_assign(&mut self, rhs: u128) {
        shl_ubi(&mut self.data, rhs);
    }
}

macro_rules! impl_shl_assign_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl ShlAssign<$t> for UBitInt {
                #[inline]
                fn shl_assign(&mut self, rhs: $t) {
                    shl_ubi(&mut self.data, rhs as u128);
                }
            }
        )*
    };
}

impl_shl_assign_ubi_prim!(u64, usize, u32, u16, u8);

// implementing shr for UBitInt

impl Shr<u128> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn shr(self, rhs: u128) -> Self::Output {
        let mut lhs = self.data;
        shr_ubi(&mut lhs, rhs);
        UBitInt { data: lhs }
    }
}

impl Shr<u128> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn shr(self, rhs: u128) -> Self::Output {
        let mut lhs = self.data.clone();
        shr_ubi(&mut lhs, rhs);
        UBitInt { data: lhs }
    }
}

macro_rules! impl_shr_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl Shr<$t> for UBitInt {
                type Output = UBitInt;

                #[inline]
                fn shr(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    shr_ubi(&mut lhs, rhs as u128);
                    UBitInt { data: lhs }
                }
            }

            impl Shr<$t> for &UBitInt {
                type Output = UBitInt;

                #[inline]
                fn shr(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data.clone();
                    shr_ubi(&mut lhs, rhs as u128);
                    UBitInt { data: lhs }
                }
            }
        )*
    };
}

impl_shr_ubi_prim!(u64, usize, u32, u16, u8);

//implementing shl assign for UBitInt

impl ShrAssign<u128> for UBitInt {
    #[inline]
    fn shr_assign(&mut self, rhs: u128) {
        shr_ubi(&mut self.data, rhs);
    }
}

macro_rules! impl_shr_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl ShrAssign<$t> for UBitInt {
                #[inline]
                fn shr_assign(&mut self, rhs: $t) {
                    shr_ubi(&mut self.data, rhs as u128);
                }
            }
        )*
    };
}

impl_shr_ubi_prim!(u64, usize, u32, u16, u8);

// implementing div for UBitInt and all unsigned primitive types

impl Div for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let mut lhs = self.data;
        UBitInt {
            data: div_ubi(&mut lhs, &rhs.data),
        }
    }
}

impl Div for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let mut lhs = self.data.clone();
        UBitInt {
            data: div_ubi(&mut lhs, &rhs.data),
        }
    }
}

impl Div<&UBitInt> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn div(self, rhs: &UBitInt) -> Self::Output {
        let mut lhs = self.data;
        UBitInt {
            data: div_ubi(&mut lhs, &rhs.data),
        }
    }
}

impl Div<UBitInt> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn div(self, rhs: UBitInt) -> Self::Output {
        let mut lhs = self.data.clone();
        UBitInt {
            data: div_ubi(&mut lhs, &rhs.data),
        }
    }
}

macro_rules! impl_div_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl Div<$t> for UBitInt {
                type Output = UBitInt;

                #[inline]
                fn div(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    UBitInt {
                        data: div_ubi(&mut lhs, &UBitInt::from(rhs).data),
                    }
                }
            }

            impl Div<$t> for &UBitInt {
                type Output = UBitInt;

                #[inline]
                fn div(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data.clone();
                    UBitInt {
                        data: div_ubi(&mut lhs, &UBitInt::from(rhs).data),
                    }
                }
            }

            impl Div<UBitInt> for $t {
                type Output = UBitInt;

                #[inline]
                fn div(self, rhs: UBitInt) -> Self::Output {
                    let mut lhs = UBitInt::from(self).data;
                    UBitInt {
                        data: div_ubi(&mut lhs, &rhs.data),
                    }
                }
            }

            impl Div<&UBitInt> for $t {
                type Output = UBitInt;

                #[inline]
                fn div(self, rhs: &UBitInt) -> Self::Output {
                    let mut lhs = UBitInt::from(self).data;
                    UBitInt {
                        data: div_ubi(&mut lhs, &rhs.data),
                    }
                }
            }
        )*
    };
}

impl_div_ubi_prim!(u128, u64, usize, u32, u16, u8);

//implementing Div Assign for UBitInt and all unsigned integers

impl DivAssign for UBitInt {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.data = div_ubi(&mut self.data, &rhs.data);
    }
}

impl DivAssign<&UBitInt> for UBitInt {
    #[inline]
    fn div_assign(&mut self, rhs: &UBitInt) {
        self.data = div_ubi(&mut self.data, &rhs.data);
    }
}

macro_rules! impl_div_assign_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl DivAssign<$t> for UBitInt {
                #[inline]
                fn div_assign(&mut self, rhs: $t) {
                    self.data = div_ubi(&mut self.data, &UBitInt::from(rhs).data);
                }
            }
        )*
    };
}

impl_div_assign_ubi_prim!(u128, u64, usize, u32, u16, u8);

//implementing Modulus for UBitInt and all unisghned primitives

impl Rem for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut lhs = self.data;
        div_ubi(&mut lhs, &rhs.data);
        UBitInt { data: lhs }
    }
}

impl Rem for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        let mut lhs = self.data.clone();
        div_ubi(&mut lhs, &rhs.data);
        UBitInt { data: lhs }
    }
}

impl Rem<&UBitInt> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn rem(self, rhs: &UBitInt) -> Self::Output {
        let mut lhs = self.data;
        div_ubi(&mut lhs, &rhs.data);
        UBitInt { data: lhs }
    }
}

impl Rem<UBitInt> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn rem(self, rhs: UBitInt) -> Self::Output {
        let mut lhs = self.data.clone();
        div_ubi(&mut lhs, &rhs.data);
        UBitInt { data: lhs }
    }
}

macro_rules! impl_rem_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl Rem<$t> for UBitInt {
                type Output = UBitInt;

                #[inline]
                fn rem(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    div_ubi(&mut lhs, &UBitInt::from(rhs).data);
                    UBitInt { data: lhs }
                }
            }

            impl Rem<$t> for &UBitInt {
                type Output = UBitInt;

                #[inline]
                fn rem(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data.clone();
                    div_ubi(&mut lhs, &UBitInt::from(rhs).data);
                    UBitInt { data: lhs }
                }
            }

            impl Rem<UBitInt> for $t {
                type Output = UBitInt;

                #[inline]
                fn rem(self, rhs: UBitInt) -> Self::Output {
                    let mut lhs = UBitInt::from(self).data;
                    div_ubi(&mut lhs, &rhs.data);
                    UBitInt { data: lhs }
                }
            }

            impl Rem<&UBitInt> for $t {
                type Output = UBitInt;

                #[inline]
                fn rem(self, rhs: &UBitInt) -> Self::Output {
                    let mut lhs = UBitInt::from(self).data;
                    div_ubi(&mut lhs, &rhs.data);
                    UBitInt { data: lhs }
                }
            }
        )*
    };
}

impl_rem_ubi_prim!(u128, u64, usize, u32, u16, u8);

//impl modulus assign for UBitInt and all unisghend primitive types

impl RemAssign for UBitInt {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        div_ubi(&mut self.data, &rhs.data);
    }
}

impl RemAssign<&UBitInt> for UBitInt {
    #[inline]
    fn rem_assign(&mut self, rhs: &UBitInt) {
        div_ubi(&mut self.data, &rhs.data);
    }
}

macro_rules! impl_rem_assign_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl RemAssign<$t> for UBitInt {
                #[inline]
                fn rem_assign(&mut self, rhs: $t) {
                    div_ubi(&mut self.data, &UBitInt::from(rhs).data);
                }
            }
        )*
    };
}

impl_rem_assign_ubi_prim!(u128, u64, usize, u32, u16, u8);

//implementing pow trait for UBitInt and all usighned primitives

pub trait Pow<RHS = Self> {
    type Output;
    fn pow(self, rhs: RHS) -> Self::Output;
}

impl Pow for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn pow(self, rhs: Self) -> Self::Output {
        if rhs == 0_usize {
            return UBitInt { data: vec![1] };
        } else if rhs == 1_usize {
            return self;
        } else if rhs.data[0] % 2 == 1 {
            return (&self * &self).pow(rhs >> 1_usize);
        } else {
            return (&self * &self).pow(rhs >> 1_usize) * self;
        }
    }
}

impl Pow for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn pow(self, rhs: Self) -> Self::Output {
        if *rhs == 0_usize {
            return UBitInt { data: vec![1] };
        } else if *rhs == 1_usize {
            return self.clone();
        } else if rhs.data[0] % 2 == 1 {
            return (self * self).pow(rhs >> 1_usize);
        } else {
            return self * self.pow(rhs >> 1_usize);
        }
    }
}

impl Pow<&UBitInt> for UBitInt {
    type Output = UBitInt;

    #[inline]
    fn pow(self, rhs: &UBitInt) -> Self::Output {
        if *rhs == 0_usize {
            return UBitInt { data: vec![1] };
        } else if *rhs == 1_usize {
            return self;
        } else if rhs.data[0] % 2 == 1 {
            return (&self * &self).pow(rhs >> 1_usize);
        } else {
            return (&self * &self).pow(rhs >> 1_usize) * self;
        }
    }
}

impl Pow<UBitInt> for &UBitInt {
    type Output = UBitInt;

    #[inline]
    fn pow(self, rhs: UBitInt) -> Self::Output {
        if rhs == 0_usize {
            return UBitInt { data: vec![1] };
        } else if rhs == 1_usize {
            return self.clone();
        } else if rhs.data[0] % 2 == 1 {
            return (self * self).pow(rhs >> 1_usize);
        } else {
            return self * (self * self).pow(rhs >> 1_usize);
        }
    }
}

macro_rules! impl_pow_ubi_prim {
    ($($t:ty),*) => {
        $(
            impl Pow<$t> for UBitInt {
                type Output = UBitInt;

                #[inline]
                fn pow(self, rhs: $t) -> Self::Output {
                    if rhs == 0 {
                        return UBitInt { data: vec![1] };
                    } else if rhs == 1 {
                        return self;
                    } else if rhs % 2 == 1 {
                        return (&self * &self).pow(rhs / 2);
                    } else {
                        return (&self * &self).pow(rhs / 2) * self;
                    }
                }
            }

            impl Pow<$t> for &UBitInt {
                type Output = UBitInt;

                #[inline]
                fn pow(self, rhs: $t) -> Self::Output {
                    if rhs == 0 {
                        return UBitInt { data: vec![1] };
                    } else if rhs == 1 {
                        return self.clone();
                    } else if rhs % 2 == 1 {
                        return (self * self).pow(rhs / 2);
                    } else {
                        return self * (self * self).pow(rhs / 2);
                    }
                }
            }
        )*
    };
}

impl_pow_ubi_prim!(u128, u64, usize, u32, u16, u8);
