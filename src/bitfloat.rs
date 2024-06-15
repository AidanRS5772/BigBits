use crate::ubitint::*;
use std::ops::*;
use once_cell::sync::Lazy;

fn add_bf(a: &mut Vec<usize>, b: &[usize], sh: usize) -> isize {
    let mut sstart = a.len() as isize - (b.len() + sh) as isize;

    if sstart < 0 {
        let pad = vec![0; sstart.unsigned_abs()];
        a.splice(0..0, pad);
        sstart = 0;
    }
    let start = sstart.unsigned_abs();

    let mut c: u8 = 0;
    unsafe {
        for (a_elem, &b_elem) in a[start..].iter_mut().zip(b) {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(a_elem, b_elem, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(a_elem, b_elem, &mut c);
        }

        if c != 0 {
            let a_len = a.len();
            for a_elem in &mut a[a_len - sh..] {
                if c == 0 {
                    break;
                }

                #[cfg(target_arch = "aarch64")]
                add_carry_aarch64(a_elem, &mut c);

                #[cfg(target_arch = "x86_64")]
                add_carry_x86_64(a_elem, &mut c);
            }
        }
    }

    if c == 1 {
        a.push(1);
    }

    return c as isize;
}

fn sub_bf(a: &mut Vec<usize>, b: &[usize], sh: usize) -> isize {
    let mut sstart = a.len() as isize - (b.len() + sh) as isize;

    if sstart < 0 {
        let pad = vec![0; sstart.unsigned_abs()];
        a.splice(0..0, pad);
        sstart = 0;
    }
    let start = sstart.unsigned_abs();

    println!("a: {:?}, b: {:?}", a, b);

    unsafe {
        let mut c: u8 = 1;
        for (a_elem, &b_elem) in a[start..].iter_mut().zip(b) {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(a_elem, !b_elem, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(a_elem, !b_elem, &mut c);
        }

        if c == 0 {
            let a_len = a.len();
            for a_elem in &mut a[a_len - sh..] {
                #[cfg(target_arch = "aarch64")]
                add_carry_aarch64(a_elem, &mut c);

                #[cfg(target_arch = "x86_64")]
                add_carry_x86_64(a_elem, &mut c);
            }
        }
    }

    println!("a: {:?}", a);

    if let Some(idx) = a.iter().rposition(|&x| x != 0) {
        a.truncate(idx + 1);
        return -(a.len() as isize - idx as isize - 1);
    } else {
        a.clear();
        return 0;
    }
}

fn shl_bf(bf: &mut BitFloat, sh: u128){
    let div = (sh / 64) as isize;
    let rem = (sh % 64) as u8;
    bf.exp += div;

    let mv_sz = 64 - rem;

    if rem != 0{
        let mut carry:usize = 0;
        unsafe{
            for elem in  &mut bf.m{
                #[cfg(target_arch = "aarch64")]
                shl_carry_aarch64(elem, &mut carry, rem, mv_sz);

                #[cfg(target_arch = "x86_64")]
                shl_carry_x86_64(elem, &mut carry, rem, mv_sz);
            }
        }

        if carry > 0{
            bf.m.push(carry);
            bf.exp += 1;
        }
    }
}

fn shr_bf(bf: &mut BitFloat, sh: u128){
    let div = (sh / 64) as isize;
    let rem = (sh % 64) as u8;
    bf.exp -= div;

    let mv_sz = 64 - rem;

    if rem != 0{
        let mut carry:usize = 0;
        unsafe{
            for elem in bf.m.iter_mut().rev() {
                #[cfg(target_arch = "aarch64")]
                shr_carry_aarch64(elem, &mut carry, rem, mv_sz);

                #[cfg(target_arch = "x86_64")]
                shr_carry_x86_64(elem, &mut carry, rem, mv_sz);
            }
        }

        if bf.m[bf.m.len() - 1] == 0 {
            bf.m.pop();
            bf.exp -= 1;
        }
    }
}

pub static A:Lazy<BitFloat> = Lazy::new(|| {BitFloat::from(256.0/99.0)});
pub static B:Lazy<BitFloat> = Lazy::new(|| {BitFloat::from(-64.0/11.0)});
pub static C:Lazy<BitFloat> = Lazy::new(|| {BitFloat::from(140.0/33.0)});
pub static ONE:Lazy<BitFloat> = Lazy::new(|| {BitFloat::from(1)});

pub fn recipricol(d: &mut BitFloat) {
    let d_log2 = d.log2_int() + 1;
    *d >>= d_log2;

    println!("d: {:?}", (*d).to().unwrap());

    let mut x = &*C + (&*d).mul_stbl((&*A).mul_stbl(&*d) + &*B);

    println!("x: {:?}", x.to().unwrap());

    let max_i = ((d.m.len()*64) as f64 / 6.62935662007960961).log(3.0) as usize;

    println!("max i: {max_i}");
    println!("d*x: {:?}", (&*d).mul_stbl(&x));
    println!("1-d*x: {:?}", ((&*ONE) - (&*d).mul_stbl(&x)));

    // for _ in 0..max_i{
    //     let e = &*ONE - (&*d).mul_stbl(&x);
    //     let y = (&x).mul_stbl(&e);
    //     x += &y + (&y).mul_stbl(&e);

    //     println!("\ne: {}", e.to().unwrap());
    //     println!("y: {}", y.to().unwrap());
    //     println!("x: {}\n", x.to().unwrap());
    // }

    // *d = x;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitFloat {
    m: Vec<usize>,
    exp: isize,
    sign: bool,
}

pub trait MES {
    fn get_mes(self) -> (Vec<usize>, isize, bool);
}

impl MES for f64 {
    fn get_mes(self) -> (Vec<usize>, isize, bool) {
        let bits = self.to_bits();
        let exp_2 = ((bits >> 52) & 0x7FF) as isize - 1023;

        let exp = if exp_2 < 0 {
            exp_2 / 64 - 1
        } else {
            exp_2 / 64
        };

        let mut m_bin = (((bits & 0xFFFFFFFFFFFFF) | 0x10000000000000) as u128) << 12;

        if exp_2 < 0 {
            m_bin <<= 64 + exp_2 % 64;
        } else {
            m_bin <<= exp_2 % 64
        }

        let mut m: Vec<usize> = vec![];
        let m1: usize = (m_bin >> 64) as usize;
        let m2: usize = m_bin as usize;

        if m2 != 0 {
            m.push(m2);
        }
        if m1 != 0 {
            m.push(m1);
        }

        return (m, exp, self < 0.0);
    }
}

impl MES for f32 {
    fn get_mes(self) -> (Vec<usize>, isize, bool) {
        let bits = self.to_bits();
        let exp_2 = ((bits >> 23) & 0xFF) as isize - 127;

        let exp = if exp_2 < 0 {
            exp_2 / 64 - 1
        } else {
            exp_2 / 64
        };

        let mut m_bin = (((bits & 0x7FFFFF) | 0x800000) as u128) << 41;

        if exp_2 < 0 {
            m_bin <<= 64 + exp_2 % 64;
        } else {
            m_bin <<= exp_2 % 64
        }

        let mut m: Vec<usize> = vec![];
        let m1: usize = (m_bin >> 64) as usize;
        let m2: usize = m_bin as usize;

        if m2 != 0 {
            m.push(m2);
        }
        if m1 != 0 {
            m.push(m1);
        }

        return (m, exp, self < 0.0);
    }
}

impl MES for i128 {
    fn get_mes(self) -> (Vec<usize>, isize, bool) {
        let uval = self.unsigned_abs();

        let m1 = (uval >> 64) as usize;
        let m2 = uval as usize;

        let mut exp: isize = 0;
        let mut m: Vec<usize> = vec![];

        if m2 != 0 {
            m.push(m2);
        }
        if m1 != 0 {
            m.push(m1);
            exp = 1;
        }

        return (m, exp, self < 0);
    }
}

macro_rules! impl_MES_bf_iprim {
    ($($t:ty),*) => {
        $(
            impl MES for $t{
                fn get_mes(self) -> (Vec<usize>, isize, bool){
                    (vec![self as usize], 0, self < 0)
                }
            }
        )*
    };
}

impl_MES_bf_iprim!(i64, isize, i32, i16, i8);

impl BitFloat {
    pub fn get_m(&self) -> Vec<usize> {
        self.m.clone()
    }

    pub fn get_exp(&self) -> isize {
        self.exp
    }

    pub fn get_sign(&self) -> bool {
        self.sign
    }

    pub fn make(m: Vec<usize>, exp: isize, sign: bool) -> BitFloat {
        BitFloat { m, exp, sign }
    }

    pub fn from<T: MES>(val: T) -> BitFloat {
        let (m, exp, sign) = val.get_mes();
        BitFloat { m, exp, sign }
    }

    pub fn to(&self) -> Result<f64, String> {
        let exp: i32 = if let Ok(exp) = self.exp.try_into() {
            exp
        } else {
            return Err("BitFloat is too large or to small for f64".to_string());
        };

        let first_val = if let Some(&val) = self.m.get(self.m.len().saturating_sub(1)){
            val
        }else{
            return Ok(0.0);
        };

        let base = 2_f64.powi(64);
        let mut sc = 2_f64.powi(64 * exp);

        let mut out = sc * first_val as f64;

        if out.is_infinite() {
            return Err("BitFloat is to large for f64".to_string());
        }

        for &e in self.m[..(self.m.len() - 1)].iter().rev() {
            sc /= base;
            out += sc * e as f64;
        }

        Ok(out)
    }

    pub fn abs_cmp(&self, other: &BitFloat) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        let exp_ord = self.exp.cmp(&other.exp);
        if exp_ord != Equal {
            return exp_ord;
        }

        match (self.m.is_empty(), other.m.is_empty()){
            (true, true) => return Equal,
            (true, false) => return Less,
            (false, true) => return Greater,
            (false, false) => {},
        }

        let mut m_ord: std::cmp::Ordering;

        for (&a, b) in self.m.iter().rev().zip(other.m.iter().rev()) {
            m_ord = a.cmp(b);
            if m_ord != Equal {
                return m_ord;
            }
        }

        std::cmp::Ordering::Equal
    }

    pub fn log2_int(&self) -> i128{
        let mut out = self.exp as i128 * 64;
        out += 64 - self.m[self.m.len()-1].leading_zeros() as i128 - 1;
        return out;
    }
}

macro_rules! impl_peq_bf_iprim {
    ($($t:ty),*) => {
        $(
            impl PartialEq<$t> for BitFloat{
                fn eq(&self, other: &$t) -> bool {
                    *self == BitFloat::from(*other)
                }
            }
        )*
    };
}

impl_peq_bf_iprim!(f64, f32, i128, i64, isize, i32, i16, i8);

impl PartialOrd for BitFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.sign ^ other.sign {
            if self.sign {
                return Some(std::cmp::Ordering::Less);
            } else {
                return Some(std::cmp::Ordering::Greater);
            }
        }

        let exp_ord = self.exp.cmp(&other.exp);
        if exp_ord != std::cmp::Ordering::Equal {
            if self.sign {
                return Some(exp_ord.reverse());
            } else {
                return Some(exp_ord);
            }
        }

        let mut m_ord = std::cmp::Ordering::Equal;

        for (&a, b) in self.m.iter().rev().zip(other.m.iter().rev()) {
            m_ord = a.cmp(b);
            if m_ord != std::cmp::Ordering::Equal {
                break;
            }
        }

        if self.sign {
            return Some(m_ord.reverse());
        } else {
            return Some(m_ord);
        }
    }
}

macro_rules! impl_pord_bf_iprim {
    ($($t:ty),*) => {
        $(
            impl PartialOrd<$t> for BitFloat{
                fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering> {
                    self.partial_cmp(&BitFloat::from(*other))
                }
            }
        )*
    };
}

impl_pord_bf_iprim!(f64, f32, i128, i64, isize, i32, i16, i8);

impl Ord for BitFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Add for BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut a, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m, rhs.m, self.exp, self.sign),
            Less => (rhs.m, self.m, rhs.exp, rhs.sign),
            Equal => (self.m, rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ rhs.sign {
            sub_bf(&mut a, &b, sh)
        } else {
            add_bf(&mut a, &b, sh)
        };

        exp += exp_adj;

        BitFloat { m: a, exp, sign }
    }
}

impl Add for &BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut a, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m.clone(), &rhs.m, self.exp, self.sign),
            Less => (rhs.m.clone(), &self.m, rhs.exp, rhs.sign),
            Equal => (self.m.clone(), &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ rhs.sign {
            sub_bf(&mut a, b, sh)
        } else {
            add_bf(&mut a, b, sh)
        };

        exp += exp_adj;

        BitFloat { m: a, exp, sign }
    }
}

impl Add<BitFloat> for &BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: BitFloat) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut a, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m.clone(), &rhs.m, self.exp, self.sign),
            Less => (rhs.m, &self.m, rhs.exp, rhs.sign),
            Equal => (self.m.clone(), &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ rhs.sign {
            sub_bf(&mut a, b, sh)
        } else {
            add_bf(&mut a, b, sh)
        };

        exp += exp_adj;

        BitFloat { m: a, exp, sign }
    }
}

impl Add<&BitFloat> for BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: &BitFloat) -> Self::Output {
        use std::cmp::Ordering::*;

        let sh = (self.exp - rhs.exp).unsigned_abs();

        let (mut a, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m, &rhs.m, self.exp, self.sign),
            Less => (rhs.m.clone(), &self.m, rhs.exp, rhs.sign),
            Equal => (self.m, &rhs.m, self.exp, self.sign),
        };

        let exp_adj = if self.sign ^ rhs.sign {
            sub_bf(&mut a, b, sh)
        } else {
            add_bf(&mut a, b, sh)
        };

        exp += exp_adj;

        BitFloat { m: a, exp, sign }
    }
}

impl<T: MES> Add<T> for BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: T) -> Self::Output {
        self + BitFloat::from(rhs)
    }
}

impl<T: MES> Add<T> for &BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: T) -> Self::Output {
        self + BitFloat::from(rhs)
    }
}

macro_rules! impl_add_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Add<BitFloat> for $t {
                type Output = BitFloat;

                fn add(self, rhs: BitFloat) -> Self::Output {
                    rhs + self
                }
            }

            impl Add<&BitFloat> for $t {
                type Output = BitFloat;

                fn add(self, rhs: &BitFloat) -> Self::Output {
                    rhs + self
                }
            }
        )*
    }
}

impl_add_for_bitfloat!(f64, f32, i128, i64, isize, i32, i16, i8);

impl AddAssign for BitFloat{
    fn add_assign(&mut self, rhs: Self) {
        use std::cmp::Ordering::*;

        let sh = (self.exp - rhs.exp).unsigned_abs();

        match self.abs_cmp(&rhs) {
            Greater => {
                let exp_adj = if self.sign ^ rhs.sign {
                    sub_bf(&mut self.m, &rhs.m, sh)
                } else {
                    add_bf(&mut self.m, &rhs.m, sh)
                };

                self.exp += exp_adj;
            },
            Less => {
                let mut a = rhs.m;

                let exp_adj = if self.sign ^ rhs.sign {
                    sub_bf(&mut a, &self.m, sh)
                } else {
                    add_bf(&mut a, &self.m, sh)
                };

                self.m = a;
                self.exp = rhs.exp + exp_adj;
                self.sign = rhs.sign
            },
            Equal => {
                if self.sign ^ rhs.sign {
                    self.m = vec![];
                    self.exp = 0;
                    self.sign = false;
                } else {
                    let exp_adj = add_bf(&mut self.m, &rhs.m, sh);
                    self.exp += exp_adj;
                }
            },
        }
    }
}

impl AddAssign<&BitFloat> for BitFloat{
    fn add_assign(&mut self, rhs: &Self) {
        use std::cmp::Ordering::*;

        let sh = (self.exp - rhs.exp).unsigned_abs();
        match self.abs_cmp(&rhs) {
            Greater => {
                let exp_adj = if self.sign ^ rhs.sign {
                    sub_bf(&mut self.m, &rhs.m, sh)
                } else {
                    add_bf(&mut self.m, &rhs.m, sh)
                };

                self.exp += exp_adj;
            },
            Less => {
                let mut a = rhs.m.clone();

                let exp_adj = if self.sign ^ rhs.sign {
                    sub_bf(&mut a, &self.m, sh)
                } else {
                    add_bf(&mut a, &self.m, sh)
                };

                self.m = a;
                self.exp = rhs.exp + exp_adj;
                self.sign = rhs.sign
            },
            Equal => {
                if self.sign ^ rhs.sign {
                    self.m = vec![];
                    self.exp = 0;
                    self.sign = false;
                } else {
                    let exp_adj = add_bf(&mut self.m, &rhs.m, sh);
                    self.exp += exp_adj;
                }
            },
        }
    }
}

impl<T: MES> AddAssign<T> for BitFloat{
    fn add_assign(&mut self, rhs: T) {
        *self += BitFloat::from(rhs);
    }
}

impl Sub for BitFloat {
    type Output = BitFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut a, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m, &rhs.m, self.exp, self.sign),
            Less => (rhs.m, &self.m, rhs.exp, rhs.sign),
            Equal => (self.m, &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ rhs.sign {
            add_bf(&mut a, b, sh)
        } else {
            sub_bf(&mut a, b, sh)
        };

        exp = exp.saturating_add(exp_adj);

        BitFloat { m: a, exp, sign }
    }
}

impl Sub for &BitFloat{
    type Output = BitFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut a, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m.clone(), &rhs.m, self.exp, self.sign),
            Less => (rhs.m.clone(), &self.m, rhs.exp, rhs.sign),
            Equal => (self.m.clone(), &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ rhs.sign {
            add_bf(&mut a, b, sh)
        } else {
            sub_bf(&mut a, b, sh)
        };

        exp = exp.saturating_add(exp_adj);

        BitFloat { m: a, exp, sign }
    }
}

impl Sub<&BitFloat> for BitFloat{
    type Output = BitFloat;

    fn sub(self, rhs: &Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut a, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m, &rhs.m, self.exp, self.sign),
            Less => (rhs.m.clone(), &self.m, rhs.exp, rhs.sign),
            Equal => (self.m, &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ rhs.sign {
            add_bf(&mut a, b, sh)
        } else {
            sub_bf(&mut a, b, sh)
        };

        exp = exp.saturating_add(exp_adj);

        BitFloat { m: a, exp, sign }
    }
}

impl Sub<BitFloat> for &BitFloat{
    type Output = BitFloat;

    fn sub(self, rhs: BitFloat) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut a, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m.clone(), &rhs.m, self.exp, self.sign),
            Less => (rhs.m, &self.m, rhs.exp, rhs.sign),
            Equal => (self.m.clone(), &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        println!("\nsh: {sh}, a: {:?}, b: {:?}", a, b);

        let exp_adj = if self.sign ^ rhs.sign {
            add_bf(&mut a, b, sh)
        } else {
            sub_bf(&mut a, b, sh)
        };

        exp = exp.saturating_add(exp_adj);

        BitFloat { m: a, exp, sign }
    }
}

impl<T: MES> Sub<T> for BitFloat{
    type Output = BitFloat;

    fn sub(self, rhs: T) -> Self::Output {
        self + BitFloat::from(rhs)
    }
}

impl<T: MES> Sub<T> for &BitFloat{
    type Output = BitFloat;

    fn sub(self, rhs: T) -> Self::Output {
        self + BitFloat::from(rhs)
    }
}

macro_rules! impl_sub_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Sub<BitFloat> for $t {
                type Output = BitFloat;

                fn sub(self, rhs: BitFloat) -> Self::Output {
                    rhs + self
                }
            }

            impl Sub<&BitFloat> for $t {
                type Output = BitFloat;

                fn sub(self, rhs: &BitFloat) -> Self::Output {
                    rhs + self
                }
            }
        )*
    }
}

impl_sub_for_bitfloat!(f64, f32, i128, i64, isize, i32, i16, i8);

impl SubAssign for BitFloat{
    fn sub_assign(&mut self, rhs: Self) {
        use std::cmp::Ordering::*;

        let sh = (self.exp - rhs.exp).unsigned_abs();

        match self.abs_cmp(&rhs) {
            Greater => {
                let exp_adj = if self.sign ^ !rhs.sign {
                    sub_bf(&mut self.m, &rhs.m, sh)
                } else {
                    add_bf(&mut self.m, &rhs.m, sh)
                };

                self.exp += exp_adj;
            },
            Less => {
                let mut a = rhs.m;

                let exp_adj = if self.sign ^ !rhs.sign {
                    sub_bf(&mut a, &self.m, sh)
                } else {
                    add_bf(&mut a, &self.m, sh)
                };

                self.m = a;
                self.exp = rhs.exp + exp_adj;
                self.sign = rhs.sign
            },
            Equal => {
                if self.sign ^ !rhs.sign {
                    self.m = vec![];
                    self.exp = 0;
                    self.sign = false;
                } else {
                    let exp_adj = add_bf(&mut self.m, &rhs.m, sh);
                    self.exp += exp_adj;
                }
            },
        }
    }
}

impl SubAssign<&BitFloat> for BitFloat{
    fn sub_assign(&mut self, rhs: &Self) {
        use std::cmp::Ordering::*;

        let sh = (self.exp - rhs.exp).unsigned_abs();
        match self.abs_cmp(&rhs) {
            Greater => {
                let exp_adj = if self.sign ^ !rhs.sign {
                    sub_bf(&mut self.m, &rhs.m, sh)
                } else {
                    add_bf(&mut self.m, &rhs.m, sh)
                };

                self.exp += exp_adj;
            },
            Less => {
                let mut a = rhs.m.clone();

                let exp_adj = if self.sign ^ !rhs.sign {
                    sub_bf(&mut a, &self.m, sh)
                } else {
                    add_bf(&mut a, &self.m, sh)
                };

                self.m = a;
                self.exp = rhs.exp + exp_adj;
                self.sign = rhs.sign
            },
            Equal => {
                if self.sign ^ !rhs.sign {
                    self.m = vec![];
                    self.exp = 0;
                    self.sign = false;
                } else {
                    let exp_adj = add_bf(&mut self.m, &rhs.m, sh);
                    self.exp += exp_adj;
                }
            },
        }
    }
}

impl<T: MES> SubAssign<T> for BitFloat{
    fn sub_assign(&mut self, rhs: T) {
        *self += BitFloat::from(rhs);
    }
}

impl Mul for BitFloat {
    type Output = BitFloat;

    fn mul(self, rhs: Self) -> Self::Output {
        let m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul for &BitFloat{
    type Output = BitFloat;

    fn mul(self, rhs: Self) -> Self::Output {
        let m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<&BitFloat> for BitFloat{
    type Output = BitFloat;

    fn mul(self, rhs: &Self) -> Self::Output {
        let m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<BitFloat> for &BitFloat{
    type Output = BitFloat;

    fn mul(self, rhs: BitFloat) -> Self::Output {
        let m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl<T: MES> Mul<T> for BitFloat{
    type Output = BitFloat;

    fn mul(self, rhs: T) -> Self::Output {
        self * BitFloat::from(rhs)
    }
}

impl<T: MES> Mul<T> for &BitFloat{
    type Output = BitFloat;

    fn mul(self, rhs: T) -> Self::Output {
        self * BitFloat::from(rhs)
    }
}

macro_rules! impl_mul_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Mul<BitFloat> for $t{
                type Output = BitFloat;

                fn mul(self, rhs: BitFloat) -> Self::Output{
                    rhs * self
                }
            }

            impl Mul<&BitFloat> for $t{
                type Output = BitFloat;

                fn mul(self, rhs: &BitFloat) -> Self::Output{
                    rhs * self
                }
            }
        )*
    }
}

impl_mul_for_bitfloat!(f64, f32, i128, i64, isize, i32, i16, i8);

pub trait MulStbl<RHS = Self> {
    type Output;
    fn mul_stbl(self, rhs: RHS) -> Self::Output;
}

impl MulStbl for BitFloat {
    type Output = BitFloat;

    fn mul_stbl(self, rhs: Self) -> Self::Output {
        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        m.drain(..(std::cmp::min(rhs.m.len(),self.m.len())-1));

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl MulStbl for &BitFloat{
    type Output = BitFloat;

    fn mul_stbl(self, rhs: Self) -> Self::Output {
        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        m.drain(..(std::cmp::min(rhs.m.len(),self.m.len())-1));

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl MulStbl<&BitFloat> for BitFloat {
    type Output = BitFloat;

    fn mul_stbl(self, rhs: &Self) -> Self::Output {
        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        m.drain(..(std::cmp::min(rhs.m.len(),self.m.len())-1));

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl MulStbl<BitFloat> for &BitFloat{
    type Output = BitFloat;

    fn mul_stbl(self, rhs: BitFloat) -> Self::Output {
        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        m.drain(..(std::cmp::min(rhs.m.len(),self.m.len())-1));

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Shl<i128> for BitFloat{
    type Output = BitFloat;

    fn shl(self, rhs: i128) -> Self::Output {
        let mut bf = self;

        if rhs < 0{
            shr_bf(&mut bf, rhs.unsigned_abs());
        }else{
            shl_bf(&mut bf, rhs.unsigned_abs())
        }

        return bf;
    }
}

impl Shl<i128> for &BitFloat{
    type Output = BitFloat;

    fn shl(self, rhs: i128) -> Self::Output {
        let mut bf = self.clone();

        if rhs < 0{
            shr_bf(&mut bf, rhs.unsigned_abs());
        }else{
            shl_bf(&mut bf, rhs.unsigned_abs())
        }

        return bf;
    }
}

macro_rules! impl_shl_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Shl<$t> for BitFloat{
                type Output = BitFloat;
            
                fn shl(self, rhs: $t) -> Self::Output {
                    let mut bf = self;

                    if rhs < 0{
                        shr_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }else{
                        shl_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }
            
                    return bf;
                }
            }
            
            impl Shl<$t> for &BitFloat{
                type Output = BitFloat;
            
                fn shl(self, rhs: $t) -> Self::Output {
                    let mut bf = self.clone();

                    if rhs < 0{
                        shr_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }else{
                        shl_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }
            
                    return bf;
                }
            }
        )*
    }
}

impl_shl_for_bitfloat!(i64, isize, i32, i16, i8);

impl ShlAssign<i128> for BitFloat{
    fn shl_assign(&mut self, rhs: i128) {
        if rhs < 0{
            shr_bf(self, rhs.unsigned_abs());
        }else{
            shl_bf(self, rhs.unsigned_abs());
        }
    }
}

macro_rules! impl_shl_assign_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl ShlAssign<$t> for BitFloat{
                fn shl_assign(&mut self, rhs: $t) {
                    if rhs < 0{
                        shr_bf(self, rhs.unsigned_abs() as u128);
                    }else{
                        shl_bf(self, rhs.unsigned_abs() as u128);
                    }
                }
            }
        )*
    }
}

impl_shl_assign_for_bitfloat!(i64, isize, i32, i16, i8);

impl Shr<i128> for BitFloat{
    type Output = BitFloat;

    fn shr(self, rhs: i128) -> Self::Output {
        let mut bf = self;

        if rhs < 0{
            shl_bf(&mut bf, rhs.unsigned_abs());
        }else{
            shr_bf(&mut bf, rhs.unsigned_abs());
        }

        return bf;
    }
}

impl Shr<i128> for &BitFloat{
    type Output = BitFloat;

    fn shr(self, rhs: i128) -> Self::Output {
        let mut bf = self.clone();

        if rhs < 0{
            shl_bf(&mut bf, rhs.unsigned_abs());
        }else{
            shr_bf(&mut bf, rhs.unsigned_abs());
        }

        return bf;
    }
}

macro_rules! impl_shr_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Shr<$t> for BitFloat{
                type Output = BitFloat;
            
                fn shr(self, rhs: $t) -> Self::Output {
                    let mut bf = self;

                    if rhs < 0{
                        shl_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }else{
                        shr_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }
            
                    return bf;
                }
            }
            
            impl Shr<$t> for &BitFloat{
                type Output = BitFloat;
            
                fn shr(self, rhs: $t) -> Self::Output {
                    let mut bf = self.clone();

                    if rhs < 0{
                        shl_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }else{
                        shr_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }
            
                    return bf;
                }
            }
        )*
    }
}

impl_shr_for_bitfloat!(i64, isize, i32, i16, i8);

impl ShrAssign<i128> for BitFloat{
    fn shr_assign(&mut self, rhs: i128) {
        if rhs < 0{
            shl_bf(self, rhs.unsigned_abs());
        }else{
            shr_bf(self, rhs.unsigned_abs());
        }
    }
}

macro_rules! impl_shr_assign_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl ShrAssign<$t> for BitFloat{
                fn shr_assign(&mut self, rhs: $t) {
                    if rhs < 0{
                        shl_bf(self, rhs.unsigned_abs() as u128);
                    }else{
                        shr_bf(self, rhs.unsigned_abs() as u128);
                    }
                }
            }
        )*
    }
}

impl_shr_assign_for_bitfloat!(i64, isize, i32, i16, i8);