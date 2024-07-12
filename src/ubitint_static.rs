#![allow(dead_code)]
#![allow(asm_sub_register)]

use crate::ubitint::*;
use std::arch::asm;
use std::fmt;
use std::ops::*;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn add4_with_carry_aarch64(lhs: &mut [usize], rhs: &[usize], c: &mut u8) {
    asm!(
        "adds {c}, {c}, #0xFFFFFFFFFFFFFFFF",
        "adcs {l0}, {l0}, {r0}",
        "adcs {l1}, {l1}, {r1}",
        "adcs {l2}, {l2}, {r2}",
        "adcs {l3}, {l3}, {r3}",
        "cset {c}, cs",
        l0 = inout(reg) lhs[0],
        l1 = inout(reg) lhs[1],
        l2 = inout(reg) lhs[2],
        l3 = inout(reg) lhs[3],
        r0 = in(reg) rhs[0],
        r1 = in(reg) rhs[1],
        r2 = in(reg) rhs[2],
        r3 = in(reg) rhs[3],
        c = inout(reg) *c,
        options(nostack)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn add4_with_carry_x86_64(lhs: &mut [usize], rhs: &[usize], c: &mut u8) {
    asm!(
        "add {c}, 0xFF",
        "adc {l0}, {r0}",
        "adc {l1}, {r1}",
        "adc {l2}, {r2}",
        "adc {l3}, {r3}",
        "setc {c}",
        l0 = inout(reg) lhs[0],
        l1 = inout(reg) lhs[1],
        l2 = inout(reg) lhs[2],
        l3 = inout(reg) lhs[3],
        r0 = in(reg) rhs[0],
        r1 = in(reg) rhs[1],
        r2 = in(reg) rhs[2],
        r3 = in(reg) rhs[3],
        c = inout(reg_byte) *c,
        options(nostack)
    );
}

#[inline]
pub fn add_ubis<const N: usize>(lhs: &mut [usize], rhs: &[usize], mut c: u8) -> bool {
    let mut i = 4;
    unsafe {
        while i <= N {
            #[cfg(target_arch = "aarch64")]
            add4_with_carry_aarch64(&mut lhs[i - 4..i], &rhs[i - 4..i], &mut c);

            #[cfg(target_arch = "x86_64")]
            add4_with_carry_x86_64(&mut lhs[i - 4..i], &rhs[i - 4..i], &mut c);

            i += 4;
        }

        i -= 4;
        while i < N {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(&mut lhs[i], rhs[i], &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(&mut lhs[i], rhs[i], &mut c);

            i += 1;
        }
    }

    return c == 1;
}

#[inline]
pub fn add_ubis_prim(ubis: &mut [usize], prim: usize) {
    unsafe {
        #[cfg(target_arch = "aarch64")]
        let mut c: u8 = add_prop_carry_aarch64(&mut ubis[0], prim);

        #[cfg(target_arch = "x86_64")]
        let mut c: u8 = add_prop_carry_x86_64(&mut ubis[0], prim);

        if c == 0 {
            for l in &mut ubis[1..] {
                if c == 0 {
                    break;
                }

                #[cfg(target_arch = "aarch64")]
                add_carry_aarch64(l, &mut c);

                #[cfg(target_arch = "x86_64")]
                add_carry_x86_64(l, &mut c);
            }
        }
    }
}

#[inline(always)]
pub fn one_comp(val: &mut [usize]) {
    for e in val {
        *e = !*e
    }
}

#[inline]
pub fn sub_ubis_prim(ubis: &mut [usize], prim: usize) {
    unsafe {
        #[cfg(target_arch = "aarch64")]
        let mut c: u8 = add_prop_carry_aarch64(&mut ubis[0], !prim + 1);

        #[cfg(target_arch = "x86_64")]
        let mut c: u8 = add_prop_carry_x86_64(&mut ubis[0], !prim + 1);

        if c == 0 {
            for l in &mut ubis[1..] {
                #[cfg(target_arch = "aarch64")]
                sub_carry_aarch64(l, &mut c);

                #[cfg(target_arch = "x86_64")]
                sub_carry_x86_64(l, &mut c);
            }
        }
    }
}

#[inline]
pub fn mul_ubis_prim(a: &mut [usize], b: u128) {
    let usz = 64 as u128;
    let mut carry = 0_u128;
    for val in a {
        let term = (*val as u128) * b + carry;
        carry = term >> usz;
        *val = term as usize;
    }
}

#[inline]
pub fn mul_ubis<const N: usize>(a: &[usize], b: &[usize]) -> [usize; N] {
    let mut out = [0; N];

    let mask = 0xFFFFFFFFFFFFFFFF;

    let mut carry: u128 = 0;
    for k in 0..N{
        let mut term: u128 = carry;
        carry = 0;
        for j in 0..=k {
            let val = (a[j] as u128) * (b[k - j] as u128);
            term += val & mask;
            carry += val >> 64;
        }
        carry += term >> 64;
        out[k] = term as usize;
    }

    out
}

#[inline]
pub fn shr_ubis<const N: usize>(ubis: &mut [usize], sh: u128) {
    if sh == 0 {
        return;
    }
    let div = (sh / 64) as usize;
    let rem = (sh % 64) as u8;
    let mv_sz = 64 - rem;
    if div != 0 {
        ubis.copy_within(div..N, 0);
        ubis[(N - div)..N].fill(0);
    }

    unsafe {
        if rem != 0 {
            let mut carry = 0_usize;
            for i in (0..(N - div)).rev() {
                #[cfg(target_arch = "aarch64")]
                shr_carry_aarch64(&mut ubis[i], &mut carry, rem, mv_sz);

                #[cfg(target_arch = "x86_64")]
                shr_carry_x86_64(&mut ubis[i], &mut carry, rem, mv_sz);
            }
        }
    }
}

#[inline]
pub fn shl_ubis<const N: usize>(ubis: &mut [usize], sh: u128) {
    if sh == 0 {
        return;
    }

    let div = (sh / 64) as usize;
    let rem = (sh % 64) as u8;
    let mv_sz = 64 - rem;
    if div != 0 {
        ubis.copy_within(0..(N - div), div);
        ubis[0..div].fill(0);
    }

    unsafe {
        if rem != 0 {
            let mut carry = 0_usize;
            for i in div..N {
                #[cfg(target_arch = "aarch64")]
                shl_carry_aarch64(&mut ubis[i], &mut carry, rem, mv_sz);

                #[cfg(target_arch = "x86_64")]
                shl_carry_x86_64(&mut ubis[i], &mut carry, rem, mv_sz);
            }
        }
    }
}

#[inline]
pub fn log2_ubis(ubis: &[usize]) -> u128 {
    if let Some(idx) = ubis.iter().rposition(|&x| x != 0) {
        return (idx as u128) * 64
            + (64 - ubis[idx].leading_zeros() as usize - 1) as u128;
    } else {
        return 0;
    }
}

#[inline]
pub fn div_ubis<const N: usize>(n: &mut UBitIntStatic<N>, d: UBitIntStatic<N>) -> UBitIntStatic<N> {
    match (*n).partial_cmp(&d) {
        Some(std::cmp::Ordering::Less) => return UBitIntStatic::<N> { data: [0; N] },
        Some(std::cmp::Ordering::Equal) => {
            *n = UBitIntStatic::<N> { data: [0; N] };
            let mut data = [0; N];
            data[0] = 1;
            return UBitIntStatic::<N> { data };
        }
        Some(std::cmp::Ordering::Greater) => (),
        None => unreachable!(),
    }

    let n_log2 = log2_ubis(&n.data);
    let d_log2 = log2_ubis(&d.data);

    let mut init_log2 = n_log2 - d_log2;
    let mut sub = d;
    sub <<= init_log2;

    if sub > *n {
        sub >>= 1_u128;
        init_log2 -= 1_u128;
    }

    let mut add = UBitIntStatic::<N>::from(1_usize);
    add <<= init_log2;
    let mut div = add;
    *n -= sub;

    while *n >= d {
        let log2 = log2_ubis(&sub.data) - log2_ubis(&n.data);
        sub >>= log2;
        add >>= log2;
        if sub > *n {
            sub >>= 1_u128;
            add >>= 1_u128;
        }
        *n -= sub;
        div += add;
    }

    return div;
}

#[derive(Debug, Clone, Copy)]
pub struct UBitIntStatic<const N: usize> {
    data: [usize; N],
}

pub trait ToUsizeArray<const N: usize> {
    fn to_usize_array(self) -> [usize; N];
}

impl<const N: usize> ToUsizeArray<N> for u128 {
    fn to_usize_array(self) -> [usize; N] {
        let mut out: [usize; N] = [0; N];
        
            let lower = (self & 0xFFFFFFFFFFFFFFFF) as usize;
            let upper = (self >> 64) as usize;
            if N < 2 {
                panic!("Static UBitInt is to small for type")
            }
            out[0] = lower;
            out[1] = upper;
        

        out
    }
}

impl<const N: usize> ToUsizeArray<N> for u64 {
    fn to_usize_array(self) -> [usize; N] {
        let mut out: [usize; N] = [0; N];
        
            out[0] = self as usize;
        

        out
    }
}

impl<const N: usize> ToUsizeArray<N> for usize {
    fn to_usize_array(self) -> [usize; N] {
        let mut out: [usize; N] = [0; N];
        out[0] = self;
        out
    }
}

macro_rules! impl_to_usize_arr_uprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> ToUsizeArray<N> for $t{
                #[inline]
                fn to_usize_array(self) -> [usize;N]{
                    let mut out: [usize; N] = [0; N];
                    out[0] = self as usize;
                    out
                }
            }
        )*
    };
}

impl_to_usize_arr_uprim!(u32, u16, u8);

impl<const N: usize> UBitIntStatic<N> {
    #[inline]
    pub fn get_data(&self) -> [usize;N] {
        self.data
    }

    #[inline]
    pub fn make(data: [usize; N]) -> UBitIntStatic<N> {
        UBitIntStatic::<N> { data }
    }

    #[inline]
    pub fn from<T: ToUsizeArray<N>>(val: T) -> UBitIntStatic<N> {
        UBitIntStatic {
            data: val.to_usize_array(),
        }
    }

    #[inline]
    pub fn to(&self) -> Result<u128, String> {
        if let Some(idx) = self.data.iter().rposition(|&x| x != 0) {
            if idx == 0 {
                return Ok(self.data[0] as u128);
            } else if idx == 1 {
                let mut out = self.data[0] as u128;
                out |= (self.data[1] as u128) << 64;
                return Ok(out);
            } else {
                return Err("Static UBitInt is to large to be converted to a u128".to_string());
            }
        } else {
            return Ok(0);
        }
    }

    #[inline]
    pub fn from_str(num: &str) -> Result<UBitIntStatic<N>, String> {
        use std::f64::consts::LN_10;
        use std::f64::consts::LN_2;
        if (num.len() as f64) * LN_10 > ((64 * N) as f64) * LN_2 {
            return Err("String is to long for Static UBitInt Size".to_string());
        }

        let mut out = UBitIntStatic::<N>::from(0_usize);
        let mut base = UBitIntStatic::<N>::from(1_usize);
        for c in num.chars().rev() {
            if let Some(digit) = c.to_digit(10) {
                out += digit * base;
                base *= 10_usize;
            } else {
                return Err("Malformed string to convert to a UBitInt".to_string());
            }
        }

        Ok(out)
    }

    #[inline]
    pub fn log2(self) -> u128 {
        log2_ubis(&self.data)
    }

    #[inline]
    pub fn mod2(self) -> bool {
        (self.data[0] & 1) == 1
    }
}

impl<const N: usize> fmt::Display for UBitIntStatic<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ubi = self.clone();
        let mut out: String = "".to_string();
        while ubi.log2() > 127 {
            let digit = ubi % 10_usize;
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

impl<const N: usize> PartialEq for UBitIntStatic<N> {
    #[inline]
    fn eq(&self, other: &UBitIntStatic<N>) -> bool {
        for (l, r) in self.data.iter().zip(&other.data) {
            if l != r {
                return false;
            }
        }

        return true;
    }
}

impl<const N:usize> PartialEq<usize> for UBitIntStatic<N>{
    #[inline]
    fn eq(&self, other: &usize) -> bool {
        if let Some(idx) = self.data.iter().rposition(|&x| x != 0){
            if idx > 0{
                return false
            }else{
                return self.data[0] == *other
            }
        }else{
            return *other == 0_usize;
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N:usize> PartialEq<u64> for UBitIntStatic<N>{
    #[inline]
    fn eq(&self, other: &u64) -> bool {
        if let Some(idx) = self.data.iter().rposition(|&x| x != 0){
            if idx > 0{
                return false
            }else{
                return self.data[0] == *other as usize
            }
        }else{
            return *other == 0_u64
        }
    }
}

macro_rules! impl_partial_eq_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> PartialEq<$t> for UBitIntStatic<N>{
                #[inline]
                fn eq(&self, other: &$t) -> bool {
                    if let Some(idx) = self.data.iter().rposition(|&x| x != 0){
                        if idx > 0{
                            return false
                        }else{
                            return self.data[0] == *other as usize
                        }
                    }else{
                        return *other == 0
                    }
                }
            }
        )*
    };
}

impl_partial_eq_ubis_prim!(u32, u16, u8);

impl<const N: usize> PartialOrd for UBitIntStatic<N> {
    #[inline]
    fn partial_cmp(&self, other: &UBitIntStatic<N>) -> Option<std::cmp::Ordering> {
        for (l, r) in self.data.iter().zip(&other.data).rev() {
            if l > r {
                return Some(std::cmp::Ordering::Greater);
            } else if l < r {
                return Some(std::cmp::Ordering::Less);
            }
        }

        return Some(std::cmp::Ordering::Equal);
    }
}

impl<const N:usize> PartialOrd<usize> for UBitIntStatic<N>{
    fn partial_cmp(&self, other: &usize) -> Option<std::cmp::Ordering> {
        if let Some(idx) = self.data.iter().rposition(|&x| x != 0){
            if idx > 0{
                return Some(std::cmp::Ordering::Greater)
            }else{
                return self.data[0].partial_cmp(other)
            }
        }else{
            if *other == 0_usize{
                return Some(std::cmp::Ordering::Equal)
            }else{
                return Some(std::cmp::Ordering::Greater)
            }
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N:usize> PartialOrd<u64> for UBitIntStatic<N>{
    fn partial_cmp(&self, other: &u64) -> Option<std::cmp::Ordering> {
        if let Some(idx) = self.data.iter().rposition(|&x| x != 0){
            if idx > 0{
                return Some(std::cmp::Ordering::Greater)
            }else{
                return self.data[0].partial_cmp(&(*other as usize))
            }
        }else{
            if *other == 0_u64{
                return Some(std::cmp::Ordering::Equal)
            }else{
                return Some(std::cmp::Ordering::Greater)
            }
        }
    }
}

macro_rules! impl_partial_ord_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> PartialOrd<$t> for UBitIntStatic<N>{
                fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering> {
                    if let Some(idx) = self.data.iter().rposition(|&x| x != 0){
                        if idx > 0{
                            return Some(std::cmp::Ordering::Greater)
                        }else{
                            return self.data[0].partial_cmp(&(*other as usize))
                        }
                    }else{
                        if *other == 0{
                            return Some(std::cmp::Ordering::Equal)
                        }else{
                            return Some(std::cmp::Ordering::Greater)
                        }
                    }
                }
            }
        )*
    };
}

impl_partial_ord_ubis_prim!(u32, u16, u8);

impl<const N:usize> Eq for UBitIntStatic<N>{

}

impl<const N:usize> Ord for UBitIntStatic<N>{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}


impl<const N: usize> Add for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;

    #[inline]
    fn add(self, rhs: UBitIntStatic<N>) -> Self::Output {
        let mut lhs = self.data;
        add_ubis::<N>(&mut lhs, &rhs.data, 0);
        UBitIntStatic::<N> { data: lhs }
    }
}

impl<const N: usize> Add<usize> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;

    #[inline]
    fn add(self, rhs: usize) -> Self::Output {
        let mut lhs = self.data;
        add_ubis_prim(&mut lhs, rhs);
        UBitIntStatic::<N> { data: lhs }
    }
}

impl<const N: usize> Add<UBitIntStatic<N>> for usize {
    type Output = UBitIntStatic<N>;

    #[inline]
    fn add(self, rhs: UBitIntStatic<N>) -> Self::Output {
        rhs + self
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> Add<u64> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;

    #[inline]
    fn add(self, rhs: u64) -> Self::Output {
        let mut lhs = self.data;
        add_ubis_prim(&mut lhs, rhs as usize);
        UBitIntStatic::<N> { data: lhs }
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> Add<UBitIntStatic<N>> for u64 {
    type Output = UBitIntStatic<N>;

    #[inline]
    fn add(self, rhs: UBitIntStatic<N>) -> Self::Output {
        rhs + self
    }
}

macro_rules! impl_add_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Add<$t> for UBitIntStatic<N>{
                type Output = UBitIntStatic<N>;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    add_ubis_prim(&mut lhs, rhs as usize);
                    UBitIntStatic::<N> { data: lhs }
                }
            }

            impl<const N:usize> Add<UBitIntStatic<N>> for $t{
                type Output = UBitIntStatic<N>;

                #[inline]
                fn add(self, rhs: UBitIntStatic<N>) -> Self::Output {
                    rhs + self
                }
            }
        )*
    };
}

impl_add_ubis_prim!(u32, u16, u8);

impl<const N: usize> AddAssign for UBitIntStatic<N> {
    fn add_assign(&mut self, rhs: Self) {
        add_ubis::<N>(&mut self.data, &rhs.data, 0);
    }
}

impl<const N: usize> AddAssign<usize> for UBitIntStatic<N> {
    fn add_assign(&mut self, rhs: usize) {
        add_ubis_prim(&mut self.data, rhs);
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> AddAssign<u64> for UBitIntStatic<N> {
    fn add_assign(&mut self, rhs: u64) {
        add_ubis_prim(&mut self.data, rhs as usize);
    }
}

macro_rules! impl_add_assign_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> AddAssign<$t> for UBitIntStatic<N>{
                fn add_assign(&mut self, rhs: $t) {
                    add_ubis_prim(&mut self.data, rhs as usize);
                }
            }
        )*
    };
}

impl_add_assign_ubis_prim!(u32, u16, u8);

impl<const N: usize> Sub for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut r = rhs.data;
        one_comp(&mut r);
        add_ubis::<N>(&mut r, &self.data, 1);
        UBitIntStatic::<N> { data: r }
    }
}

impl<const N: usize> Sub<usize> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;

    fn sub(self, rhs: usize) -> Self::Output {
        let mut l = self.data;
        add_ubis_prim(&mut l, rhs);
        UBitIntStatic::<N> { data: l }
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> Sub<u64> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;

    fn sub(self, rhs: u64) -> Self::Output {
        let mut l = self.data;
        add_ubis_prim(&mut l, rhs as usize);
        UBitIntStatic::<N> { data: l }
    }
}

macro_rules! impl_sub_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Sub<$t> for UBitIntStatic<N>{
                type Output = UBitIntStatic<N>;

                fn sub(self, rhs: $t) -> Self::Output {
                    let mut l = self.data;
                    add_ubis_prim(&mut l, rhs as usize);
                    UBitIntStatic::<N> {data:l}
                }
            }
        )*
    };
}

impl_sub_ubis_prim!(u32, u16, u8);

impl<const N: usize> SubAssign for UBitIntStatic<N> {
    fn sub_assign(&mut self, rhs: Self) {
        let mut r = rhs.data;
        one_comp(&mut r);
        add_ubis::<N>(&mut self.data, &r, 1);
    }
}

impl<const N: usize> SubAssign<usize> for UBitIntStatic<N> {
    fn sub_assign(&mut self, rhs: usize) {
        sub_ubis_prim(&mut self.data, rhs);
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> SubAssign<u64> for UBitIntStatic<N> {
    fn sub_assign(&mut self, rhs: u64) {
        sub_ubis_prim(&mut self.data, rhs as usize);
    }
}

macro_rules! impl_sub_assign_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> SubAssign<$t> for UBitIntStatic<N>{
                fn sub_assign(&mut self, rhs: $t) {
                    sub_ubis_prim(&mut self.data, rhs as usize);
                }
            }
        )*
    };
}

impl_sub_assign_ubis_prim!(u32, u16, u8);

impl<const N: usize> Mul for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        UBitIntStatic::<N> {
            data: mul_ubis::<N>(&self.data, &rhs.data),
        }
    }
}

macro_rules! impl_mul_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Mul<$t> for UBitIntStatic<N> {
                type Output = UBitIntStatic<N>;

                #[inline]
                fn mul(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    mul_ubis_prim(&mut lhs, rhs as u128);
                    UBitIntStatic::<N> {
                        data: lhs
                    }
                }
            }

            impl<const N: usize> Mul<UBitIntStatic<N>> for $t {
                type Output = UBitIntStatic<N>;

                #[inline]
                fn mul(self, rhs: UBitIntStatic<N>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}

impl_mul_ubis_prim!(u64, usize, u32, u16, u8);

impl<const N: usize> MulAssign for UBitIntStatic<N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = UBitIntStatic::<N> {
            data: mul_ubis(&self.data, &rhs.data),
        }
    }
}

macro_rules! impl_mul_assign_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> MulAssign<$t> for UBitIntStatic<N>{
                fn mul_assign(&mut self, rhs: $t) {
                    mul_ubis_prim(&mut self.data, rhs as u128);
                }
            }
        )*
    };
}

impl_mul_assign_ubis_prim!(u64, usize, u32, u16, u8);

impl<const N: usize> Shl<u128> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;

    fn shl(self, rhs: u128) -> Self::Output {
        let mut lhs = self.data;
        shl_ubis::<N>(&mut lhs, rhs);
        UBitIntStatic::<N> { data: lhs }
    }
}

macro_rules! impl_shl_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Shl<$t> for UBitIntStatic<N> {
                type Output = UBitIntStatic<N>;

                #[inline]
                fn shl(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    shl_ubis::<N>(&mut lhs, rhs as u128);
                    UBitIntStatic::<N> {
                        data: lhs
                    }
                }
            }
        )*
    };
}

impl_shl_ubis_prim!(u64, usize, u32, u16, u8);

impl<const N: usize> ShlAssign<u128> for UBitIntStatic<N> {
    fn shl_assign(&mut self, rhs: u128) {
        shl_ubis::<N>(&mut self.data, rhs);
    }
}

macro_rules! impl_shl_assign_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> ShlAssign<$t> for UBitIntStatic<N>{
                fn shl_assign(&mut self, rhs: $t) {
                    shl_ubis::<N>(&mut self.data, rhs as u128);
                }
            }
        )*
    };
}

impl_shl_assign_ubis_prim!(u64, usize, u32, u16, u8);

impl<const N: usize> Shr<u128> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;

    fn shr(self, rhs: u128) -> Self::Output {
        let mut lhs = self.data;
        shr_ubis::<N>(&mut lhs, rhs);
        UBitIntStatic::<N> { data: lhs }
    }
}

macro_rules! impl_shr_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Shr<$t> for UBitIntStatic<N> {
                type Output = UBitIntStatic<N>;

                #[inline]
                fn shr(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    shr_ubis::<N>(&mut lhs, rhs as u128);
                    UBitIntStatic::<N> {
                        data: lhs
                    }
                }
            }
        )*
    };
}

impl_shr_ubis_prim!(u64, usize, u32, u16, u8);

impl<const N: usize> ShrAssign<u128> for UBitIntStatic<N> {
    fn shr_assign(&mut self, rhs: u128) {
        shr_ubis::<N>(&mut self.data, rhs);
    }
}

macro_rules! impl_shr_assign_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> ShrAssign<$t> for UBitIntStatic<N>{
                fn shr_assign(&mut self, rhs: $t) {
                    shr_ubis::<N>(&mut self.data, rhs as u128);
                }
            }
        )*
    };
}

impl_shr_assign_ubis_prim!(u64, usize, u32, u16, u8);

impl<const N: usize> Div for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        div_ubis::<N>(&mut lhs, rhs)
    }
}

macro_rules! impl_div_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Div<$t> for UBitIntStatic<N>{
                type Output = UBitIntStatic<N>;

                fn div(self, rhs: $t) -> Self::Output {
                    let mut lhs = self;
                    div_ubis::<N>(&mut lhs, UBitIntStatic::<N>::from(rhs))
                }
            }
        )*
    };
}

impl_div_ubis_prim!(u128, u64, usize, u32, u16, u8);

impl<const N: usize> DivAssign for UBitIntStatic<N> {
    fn div_assign(&mut self, rhs: Self) {
        *self = div_ubis::<N>(self, rhs);
    }
}

macro_rules! impl_div_assign_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> DivAssign<$t> for UBitIntStatic<N>{
                fn div_assign(&mut self, rhs: $t) {
                    *self = div_ubis::<N>(self, UBitIntStatic::<N>::from(rhs));
                }
            }
        )*
    };
}

impl_div_assign_ubis_prim!(u128, u64, usize, u32, u16, u8);

impl<const N:usize> Rem for UBitIntStatic<N>{
    type Output = UBitIntStatic<N>;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        div_ubis(&mut lhs, rhs);
        lhs
    }
}

macro_rules! impl_rem_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Rem<$t> for UBitIntStatic<N>{
                type Output = UBitIntStatic<N>;
            
                fn rem(self, rhs: $t) -> Self::Output {
                    let mut lhs = self;
                    div_ubis(&mut lhs, UBitIntStatic::<N>::from(rhs));
                    lhs
                }
            }
        )*
    };
}

impl_rem_ubis_prim!(u128, u64, usize, u32, u16, u8);

impl<const N:usize> RemAssign for UBitIntStatic<N>{
    fn rem_assign(&mut self, rhs: Self) {
        div_ubis(self, rhs);
    }
}

macro_rules! impl_rem_assign_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> RemAssign<$t> for UBitIntStatic<N>{
                fn rem_assign(&mut self, rhs: $t) {
                    div_ubis(self, UBitIntStatic::<N>::from(rhs));
                }
            }
        )*
    };
}

impl_rem_assign_ubis_prim!(u128, u64, usize, u32, u16, u8);

pub trait DivRem< RHS = Self>{
    type Output;
    fn div_rem(self, rhs: RHS) -> Self::Output;
}

impl<const N:usize> DivRem for UBitIntStatic<N>{
    type Output = (UBitIntStatic<N>, UBitIntStatic<N>);

    fn div_rem(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        let div = div_ubis(&mut lhs, rhs);
        (div, self)
    }
}

macro_rules! impl_div_rem_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> DivRem<$t> for UBitIntStatic<N>{
                type Output = (UBitIntStatic<N>, UBitIntStatic<N>);

                fn div_rem(self, rhs: $t) -> Self::Output {
                    self.div_rem(UBitIntStatic::<N>::from(rhs))
                }
            }
        )*
    };
}

impl_div_rem_ubis_prim!(u128, u64, usize, u32, u16, u8);

pub trait Pow< RHS = Self> {
    type Output;
    fn pow(self, rhs: RHS) -> Self::Output;
}

impl<const N:usize> Pow for UBitIntStatic::<N>{
    type Output = UBitIntStatic<N>;

    fn pow(self, rhs: Self) -> Self::Output {
        if rhs == 0_usize{
            return UBitIntStatic::<N>::from(1_usize);
        }else if rhs == 1_usize{
            return self
        }else if rhs.mod2(){
            return self * (self * self).pow(rhs >> 1_usize)
        }else{
            return (self * self).pow(rhs >> 1_usize);
        }
    }
}

macro_rules! impl_pow_ubis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Pow<$t> for UBitIntStatic<N>{
                type Output = UBitIntStatic<N>;
            
                fn pow(self, rhs: $t) -> Self::Output {
                    if rhs == 0{
                        return UBitIntStatic::<N>::from(1_usize);
                    }else if rhs == 1{
                        return self
                    }else if rhs & 1 == 1{
                        return self * (self * self).pow(rhs >> 1)
                    }else{
                        return (self * self).pow(rhs >> 1);
                    }
                }
            }
        )*
    };
}

impl_pow_ubis_prim!(u128, u64, usize, u32, u16, u8);
