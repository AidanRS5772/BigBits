#![allow(dead_code)]

use crate::ubitint::*;
use core::fmt;
use std::cmp::Ordering::*;
use std::ops::*;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BitInt {
    pub val: UBitInt,
    pub sign: bool,
}

pub trait SignVal {
    fn sign_and_val(self) -> (bool, Vec<usize>);
}

macro_rules! impl_sign_val_bi_uprim {
    ($($t:ty),*) => {
        $(
            impl SignVal for $t {
                #[inline]
                fn sign_and_val(self) -> (bool, Vec<usize>) {
                    (false, self.to_usize_vec())
                }
            }
        )*
    };
}

impl_sign_val_bi_uprim!(u128, u64, usize, u32, u16, u8);

macro_rules! impl_sign_val_bi_iprim {
    ($($t:ty),*) => {
        $(
            impl SignVal for $t {
                #[inline]
                fn sign_and_val(self) -> (bool, Vec<usize>) {
                    (self < 0, self.to_usize_vec())
                }
            }
        )*
    };
}

impl_sign_val_bi_iprim!(i128, i64, isize, i32, i16, i8);

impl BitInt {
    #[inline]
    pub fn get_val(&self) -> UBitInt {
        return self.val.clone();
    }

    #[inline]
    pub fn get_sign(&self) -> bool {
        return self.sign;
    }

    #[inline]
    pub fn from<T: SignVal>(val: T) -> BitInt {
        let (sign, mut data) = val.sign_and_val();
        if let Some(idx) = data.iter().rposition(|&x| x != 0) {
            data.truncate(idx + 1);
        } else {
            data.clear();
        }

        BitInt {
            val: UBitInt { data },
            sign,
        }
    }

    #[inline]
    pub fn make(val: UBitInt, sign: bool) -> BitInt{
        BitInt{val,sign}
    }

    #[inline]
    pub fn from_ubi(ubi: &UBitInt) -> BitInt {
        BitInt {
            val: ubi.clone(),
            sign: false,
        }
    }

    #[inline]
    pub fn to(self) -> Result<i128, String> {
        if self.val > i128::MAX as u128 {
            return Err("BitInt is to large for i128".to_string());
        }

        let out = self.val.to()? as i128;
        if self.sign {
            return Ok(-out);
        } else {
            return Ok(out);
        }
    }

    #[inline]
    pub fn from_str(num: &str) -> Result<BitInt, String> {
        let mut sign = false;
        if let Some(first) = num.chars().next() {
            if first == '-' {
                sign = true;
            }
        } else {
            return Err("Malformed string to convert to a UBitInt".to_string());
        }
        let ubi = UBitInt::from_str(&num[1..])?;

        Ok(BitInt { val: ubi, sign })
    }

    #[inline]
    pub fn neg(&mut self) {
        self.sign = !self.sign;
    }

    #[inline]
    pub fn abs(&mut self) {
        self.sign = false;
    }

    #[inline]
    pub fn unsighned_abs(self) -> UBitInt {
        self.val
    }
}

impl fmt::Display for BitInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sign {
            write!(f, "-{}", self.val.to_string())?;
        } else {
            write!(f, "{}", self.val.to_string())?;
        }

        Ok(())
    }
}

impl Neg for BitInt {
    type Output = BitInt;

    #[inline]
    fn neg(self) -> Self::Output {
        BitInt {
            val: self.val,
            sign: !self.sign,
        }
    }
}

impl Neg for &BitInt {
    type Output = BitInt;

    #[inline]
    fn neg(self) -> Self::Output {
        BitInt {
            val: self.val.clone(),
            sign: !self.sign,
        }
    }
}

//implementing ordering for BitInt and all signed integers primitives

macro_rules! impl_peq_bi_prim {
    ($($t:ty),*) => {
        $(
            impl PartialEq<$t> for BitInt {
                #[inline]
                fn eq(&self, other: &$t) -> bool {
                    if self.sign != (*other < 0) {
                        return false;
                    }

                    return self.val == other.unsigned_abs();
                }
            }
        )*
    };
}

impl_peq_bi_prim!(i128, i64, isize, i32, i16, i8);

impl PartialOrd for BitInt {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.sign ^ other.sign {
            if self.sign {
                return Some(Less);
            } else {
                return Some(Greater);
            }
        }

        if self.sign {
            return Some(self.val.cmp(&other.val).reverse());
        } else {
            return self.val.partial_cmp(&other.val);
        }
    }
}

macro_rules! impl_pord_bi_prim {
    ($($t:ty),*) => {
        $(
            impl PartialOrd<$t> for BitInt {
                #[inline]
                fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering> {
                    if self.sign ^ (*other < 0) {
                        if self.sign {
                            return Some(Less);
                        } else {
                            return Some(Greater);
                        }
                    }

                    if self.sign {
                        return Some(
                            self.val
                                .partial_cmp(&other.unsigned_abs())
                                .unwrap()
                                .reverse(),
                        );
                    } else {
                        return self.val.partial_cmp(&other.unsigned_abs());
                    }
                }
            }
        )*
    };
}

impl_pord_bi_prim!(i128, i64, isize, i32, i16, i8);

impl Ord for BitInt {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.sign ^ other.sign {
            if self.sign {
                return Less;
            } else {
                return Greater;
            }
        }

        if self.sign {
            return self.val.cmp(&other.val).reverse();
        } else {
            return self.val.cmp(&other.val);
        }
    }
}

//implementing Add for BitInt and signed integer primitives

impl Add for BitInt {
    type Output = BitInt;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if self.sign ^ rhs.sign {
            match self.val.cmp(&rhs.val) {
                Greater => BitInt {
                    val: self.val - rhs.val,
                    sign: self.sign,
                },
                Less => BitInt {
                    val: rhs.val - self.val,
                    sign: rhs.sign,
                },
                Equal => BitInt {
                    val: UBitInt { data: vec![] },
                    sign: false,
                },
            }
        } else {
            BitInt {
                val: self.val + rhs.val,
                sign: self.sign,
            }
        }
    }
}

impl Add for &BitInt {
    type Output = BitInt;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if self.sign ^ rhs.sign {
            match self.val.cmp(&rhs.val) {
                Greater => BitInt {
                    val: &self.val - &rhs.val,
                    sign: self.sign,
                },
                Less => BitInt {
                    val: &rhs.val - &self.val,
                    sign: rhs.sign,
                },
                Equal => BitInt {
                    val: UBitInt { data: vec![] },
                    sign: false,
                },
            }
        } else {
            BitInt {
                val: &self.val + &rhs.val,
                sign: self.sign,
            }
        }
    }
}

impl Add<&BitInt> for BitInt {
    type Output = BitInt;

    #[inline]
    fn add(self, rhs: &BitInt) -> Self::Output {
        if self.sign ^ rhs.sign {
            match self.val.cmp(&rhs.val) {
                Greater => BitInt {
                    val: self.val - &rhs.val,
                    sign: self.sign,
                },
                Less => BitInt {
                    val: &rhs.val - self.val,
                    sign: rhs.sign,
                },
                Equal => BitInt {
                    val: UBitInt { data: vec![] },
                    sign: false,
                },
            }
        } else {
            BitInt {
                val: self.val + &rhs.val,
                sign: self.sign,
            }
        }
    }
}

impl Add<BitInt> for &BitInt {
    type Output = BitInt;

    #[inline]
    fn add(self, rhs: BitInt) -> Self::Output {
        if self.sign ^ rhs.sign {
            match self.val.cmp(&rhs.val) {
                Greater => BitInt {
                    val: &self.val - rhs.val,
                    sign: self.sign,
                },
                Less => BitInt {
                    val: rhs.val - &self.val,
                    sign: rhs.sign,
                },
                Equal => BitInt {
                    val: UBitInt { data: vec![] },
                    sign: false,
                },
            }
        } else {
            BitInt {
                val: &self.val + rhs.val,
                sign: self.sign,
            }
        }
    }
}

macro_rules! impl_add_bi_prim {
    ($($t:ty),*) => {
        $(
            impl Add<$t> for BitInt {
                type Output = BitInt;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    if self.sign ^ (rhs < 0) {
                        match self.val.partial_cmp(&rhs.unsigned_abs()).unwrap() {
                            Greater => BitInt {
                                val: self.val - rhs.unsigned_abs(),
                                sign: self.sign,
                            },
                            Less => BitInt {
                                val: rhs.unsigned_abs() - self.val,
                                sign: !self.sign,
                            },
                            Equal => BitInt {
                                val: UBitInt { data: vec![] },
                                sign: false,
                            },
                        }
                    } else {
                        BitInt {
                            val: self.val + rhs.unsigned_abs(),
                            sign: self.sign,
                        }
                    }
                }
            }

            impl Add<$t> for &BitInt {
                type Output = BitInt;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    if self.sign ^ (rhs < 0) {
                        match self.val.partial_cmp(&rhs.unsigned_abs()).unwrap() {
                            Greater => BitInt {
                                val: &self.val - rhs.unsigned_abs(),
                                sign: self.sign,
                            },
                            Less => BitInt {
                                val: rhs.unsigned_abs() - &self.val,
                                sign: !self.sign,
                            },
                            Equal => BitInt {
                                val: UBitInt { data: vec![] },
                                sign: false,
                            },
                        }
                    } else {
                        BitInt {
                            val: &self.val + rhs.unsigned_abs(),
                            sign: self.sign,
                        }
                    }
                }
            }

            impl Add<BitInt> for $t {
                type Output = BitInt;

                #[inline]
                fn add(self, rhs: BitInt) -> Self::Output {
                    if rhs.sign ^ (self < 0) {
                        match rhs.val.partial_cmp(&self.unsigned_abs()).unwrap() {
                            Greater => BitInt {
                                val: rhs.val - self.unsigned_abs(),
                                sign: rhs.sign,
                            },
                            Less => BitInt {
                                val: self.unsigned_abs() - rhs.val,
                                sign: !rhs.sign,
                            },
                            Equal => BitInt {
                                val: UBitInt { data: vec![] },
                                sign: false,
                            },
                        }
                    } else {
                        BitInt {
                            val: rhs.val + self.unsigned_abs(),
                            sign: rhs.sign,
                        }
                    }
                }
            }

            impl Add<&BitInt> for $t {
                type Output = BitInt;

                #[inline]
                fn add(self, rhs: &BitInt) -> Self::Output {
                    if rhs.sign ^ (self < 0) {
                        match rhs.val.partial_cmp(&self.unsigned_abs()).unwrap() {
                            Greater => BitInt {
                                val: &rhs.val - self.unsigned_abs(),
                                sign: rhs.sign,
                            },
                            Less => BitInt {
                                val: self.unsigned_abs() - &rhs.val,
                                sign: !rhs.sign,
                            },
                            Equal => BitInt {
                                val: UBitInt { data: vec![] },
                                sign: false,
                            },
                        }
                    } else {
                        BitInt {
                            val: &rhs.val + self.unsigned_abs(),
                            sign: rhs.sign,
                        }
                    }
                }
            }
        )*
    };
}

impl_add_bi_prim!(i128, i64, isize, i32, i16, i8);

//implementing AddAssign for BitInt and all signed primitives types

impl AddAssign for BitInt {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        if self.sign ^ rhs.sign {
            match self.val.cmp(&rhs.val) {
                Greater => {
                    self.val -= rhs.val;
                }
                Less => {
                    self.val = rhs.val - &self.val;
                    self.sign = rhs.sign;
                }
                Equal => {
                    *self = BitInt {
                        val: UBitInt { data: vec![] },
                        sign: false,
                    };
                }
            }
        } else {
            self.val += rhs.val;
        }
    }
}

impl AddAssign<&BitInt> for BitInt {
    #[inline]
    fn add_assign(&mut self, rhs: &BitInt) {
        if self.sign ^ rhs.sign {
            match self.val.cmp(&rhs.val) {
                Greater => {
                    self.val -= &rhs.val;
                }
                Less => {
                    self.val = &rhs.val - &self.val;
                    self.sign = rhs.sign;
                }
                Equal => {
                    *self = BitInt {
                        val: UBitInt { data: vec![] },
                        sign: false,
                    };
                }
            }
        } else {
            self.val += &rhs.val;
        }
    }
}

macro_rules! impl_add_assign_bi_prim {
    ($($t:ty),*) => {
        $(
            impl AddAssign<$t> for BitInt {
                #[inline]
                fn add_assign(&mut self, rhs: $t) {
                    if self.sign ^ (rhs < 0) {
                        match self.val.partial_cmp(&rhs.unsigned_abs()).unwrap() {
                            Greater => {
                                self.val -= rhs as u128;
                            }
                            Less => {
                                self.val = rhs as u128 - &self.val;
                                self.sign = rhs < 0;
                            }
                            Equal => {
                                *self = BitInt {
                                    val: UBitInt { data: vec![] },
                                    sign: false,
                                };
                            }
                        }
                    } else {
                        self.val += rhs as u128;
                    }
                }
            }
        )*
    };
}

impl_add_assign_bi_prim!(i128, i64, isize, i32, i16, i8);

//implementing for Sub for BitInt and signed primitive types

impl Sub for BitInt {
    type Output = BitInt;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if self.sign ^ rhs.sign {
            BitInt {
                val: self.val + rhs.val,
                sign: self.sign,
            }
        } else {
            match self.val.cmp(&rhs.val) {
                Greater => BitInt {
                    val: self.val - rhs.val,
                    sign: self.sign,
                },
                Less => BitInt {
                    val: rhs.val - self.val,
                    sign: rhs.sign,
                },
                Equal => BitInt {
                    val: UBitInt { data: vec![] },
                    sign: false,
                },
            }
        }
    }
}

impl Sub for &BitInt {
    type Output = BitInt;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if self.sign ^ rhs.sign {
            BitInt {
                val: &self.val + &rhs.val,
                sign: self.sign,
            }
        } else {
            match self.val.cmp(&rhs.val) {
                Greater => BitInt {
                    val: &self.val - &rhs.val,
                    sign: self.sign,
                },
                Less => BitInt {
                    val: &rhs.val - &self.val,
                    sign: rhs.sign,
                },
                Equal => BitInt {
                    val: UBitInt { data: vec![] },
                    sign: false,
                },
            }
        }
    }
}

impl Sub<&BitInt> for BitInt {
    type Output = BitInt;

    #[inline]
    fn sub(self, rhs: &BitInt) -> Self::Output {
        if self.sign ^ rhs.sign {
            BitInt {
                val: self.val + &rhs.val,
                sign: self.sign,
            }
        } else {
            match self.val.cmp(&rhs.val) {
                Greater => BitInt {
                    val: self.val - &rhs.val,
                    sign: self.sign,
                },
                Less => BitInt {
                    val: &rhs.val - self.val,
                    sign: rhs.sign,
                },
                Equal => BitInt {
                    val: UBitInt { data: vec![] },
                    sign: false,
                },
            }
        }
    }
}

impl Sub<BitInt> for &BitInt {
    type Output = BitInt;

    #[inline]
    fn sub(self, rhs: BitInt) -> Self::Output {
        if self.sign ^ rhs.sign {
            BitInt {
                val: &self.val + rhs.val,
                sign: self.sign,
            }
        } else {
            match self.val.cmp(&rhs.val) {
                Greater => BitInt {
                    val: &self.val - rhs.val,
                    sign: self.sign,
                },
                Less => BitInt {
                    val: rhs.val - &self.val,
                    sign: rhs.sign,
                },
                Equal => BitInt {
                    val: UBitInt { data: vec![] },
                    sign: false,
                },
            }
        }
    }
}

macro_rules! impl_sub_bi_prim {
    ($($t:ty),*) => {
        $(
            impl Sub<$t> for BitInt {
                type Output = BitInt;

                #[inline]
                fn sub(self, rhs: $t) -> Self::Output {
                    if self.sign ^ (rhs < 0) {
                        BitInt {
                            val: self.val + rhs.unsigned_abs(),
                            sign: self.sign,
                        }
                    } else {
                        match self.val.partial_cmp(&rhs.unsigned_abs()).unwrap() {
                            Greater => BitInt {
                                val: self.val - rhs.unsigned_abs(),
                                sign: self.sign,
                            },
                            Less => BitInt {
                                val: rhs.unsigned_abs() - self.val,
                                sign: !self.sign,
                            },
                            Equal => BitInt {
                                val: UBitInt { data: vec![] },
                                sign: false,
                            },
                        }
                    }
                }
            }

            impl Sub<$t> for &BitInt {
                type Output = BitInt;

                #[inline]
                fn sub(self, rhs: $t) -> Self::Output {
                    if self.sign ^ (rhs < 0) {
                        BitInt {
                            val: &self.val + rhs.unsigned_abs(),
                            sign: self.sign,
                        }
                    } else {
                        match self.val.partial_cmp(&rhs.unsigned_abs()).unwrap() {
                            Greater => BitInt {
                                val: &self.val - rhs.unsigned_abs(),
                                sign: self.sign,
                            },
                            Less => BitInt {
                                val: rhs.unsigned_abs() - &self.val,
                                sign: !self.sign,
                            },
                            Equal => BitInt {
                                val: UBitInt { data: vec![] },
                                sign: false,
                            },
                        }
                    }
                }
            }

            impl Sub<BitInt> for $t {
                type Output = BitInt;

                #[inline]
                fn sub(self, rhs: BitInt) -> Self::Output {
                    if rhs.sign ^ (self < 0) {
                        BitInt {
                            val: rhs.val + self.unsigned_abs(),
                            sign: rhs.sign,
                        }
                    } else {
                        match rhs.val.partial_cmp(&self.unsigned_abs()).unwrap() {
                            Greater => BitInt {
                                val: rhs.val - self.unsigned_abs(),
                                sign: rhs.sign,
                            },
                            Less => BitInt {
                                val: self.unsigned_abs() - rhs.val,
                                sign: !rhs.sign,
                            },
                            Equal => BitInt {
                                val: UBitInt { data: vec![] },
                                sign: false,
                            },
                        }
                    }
                }
            }

            impl Sub<&BitInt> for $t {
                type Output = BitInt;

                #[inline]
                fn sub(self, rhs: &BitInt) -> Self::Output {
                    if rhs.sign ^ (self < 0) {
                        BitInt {
                            val: &rhs.val + self.unsigned_abs(),
                            sign: rhs.sign,
                        }
                    } else {
                        match rhs.val.partial_cmp(&self.unsigned_abs()).unwrap() {
                            Greater => BitInt {
                                val: &rhs.val - self.unsigned_abs(),
                                sign: rhs.sign,
                            },
                            Less => BitInt {
                                val: self.unsigned_abs() - &rhs.val,
                                sign: !rhs.sign,
                            },
                            Equal => BitInt {
                                val: UBitInt { data: vec![] },
                                sign: false,
                            },
                        }
                    }
                }
            }
        )*
    };
}

impl_sub_bi_prim!(i128, i64, isize, i32, i16, i8);

//implementing Sub Assign for BitInt and signed primitive types

impl SubAssign for BitInt {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        if self.sign ^ rhs.sign {
            self.val += rhs.val;
        } else {
            match self.val.cmp(&rhs.val) {
                Greater => {
                    self.val -= rhs.val;
                }
                Less => {
                    self.val = rhs.val - &self.val;
                    self.sign = rhs.sign;
                }
                Equal => {
                    *self = BitInt {
                        val: UBitInt { data: vec![] },
                        sign: false,
                    };
                }
            }
        }
    }
}

impl SubAssign<&BitInt> for BitInt {
    #[inline]
    fn sub_assign(&mut self, rhs: &BitInt) {
        if self.sign ^ rhs.sign {
            self.val += &rhs.val;
        } else {
            match self.val.cmp(&rhs.val) {
                Greater => {
                    self.val -= &rhs.val;
                }
                Less => {
                    self.val = &rhs.val - &self.val;
                    self.sign = rhs.sign;
                }
                Equal => {
                    *self = BitInt {
                        val: UBitInt { data: vec![] },
                        sign: false,
                    };
                }
            }
        }
    }
}

macro_rules! impl_sub_assign_bi_prim {
    ($($t:ty),*) => {
        $(
            impl SubAssign<$t> for BitInt {
                #[inline]
                fn sub_assign(&mut self, rhs: $t) {
                    if self.sign ^ (rhs < 0) {
                        self.val += rhs.unsigned_abs();
                    } else {
                        match self.val.partial_cmp(&rhs.unsigned_abs()).unwrap() {
                            Greater => {
                                self.val -= rhs.unsigned_abs();
                            }
                            Less => {
                                self.val = rhs.unsigned_abs() - &self.val;
                                self.sign = rhs < 0;
                            }
                            Equal => {
                                *self = BitInt {
                                    val: UBitInt { data: vec![] },
                                    sign: false,
                                };
                            }
                        }
                    }
                }
            }
        )*
    };
}

impl_sub_assign_bi_prim!(i128, i64, isize, i32, i16, i8);

//implementing Mul for BitInt and signed primitive types

impl Mul for BitInt {
    type Output = BitInt;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        BitInt {
            val: self.val * rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul for &BitInt {
    type Output = BitInt;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        BitInt {
            val: &self.val * &rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<&BitInt> for BitInt {
    type Output = BitInt;

    #[inline]
    fn mul(self, rhs: &BitInt) -> Self::Output {
        BitInt {
            val: self.val * &rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<BitInt> for &BitInt {
    type Output = BitInt;

    #[inline]
    fn mul(self, rhs: BitInt) -> Self::Output {
        BitInt {
            val: &self.val * rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

macro_rules! impl_mul_bi_prim {
    ($($t:ty),*) => {
        $(
            impl Mul<$t> for BitInt {
                type Output = BitInt;

                #[inline]
                fn mul(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: self.val * rhs.unsigned_abs(),
                        sign: self.sign ^ (rhs < 0),
                    }
                }
            }

            impl Mul<$t> for &BitInt {
                type Output = BitInt;

                #[inline]
                fn mul(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: &self.val * rhs.unsigned_abs(),
                        sign: self.sign ^ (rhs < 0),
                    }
                }
            }

            impl Mul<BitInt> for $t {
                type Output = BitInt;

                #[inline]
                fn mul(self, rhs: BitInt) -> Self::Output {
                    BitInt {
                        val: self.unsigned_abs() * rhs.val,
                        sign: (self < 0) ^ rhs.sign,
                    }
                }
            }

            impl Mul<&BitInt> for $t {
                type Output = BitInt;

                #[inline]
                fn mul(self, rhs: &BitInt) -> Self::Output {
                    BitInt {
                        val: self.unsigned_abs() * &rhs.val,
                        sign: (self < 0) ^ rhs.sign,
                    }
                }
            }
        )*
    };
}

impl_mul_bi_prim!(i128, i64, isize, i32, i16, i8);

//implementing Mul Assign for BitInt and signed primitive types

impl MulAssign for BitInt {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.val *= rhs.val;
        self.sign ^= rhs.sign;
    }
}

impl MulAssign<&BitInt> for BitInt {
    #[inline]
    fn mul_assign(&mut self, rhs: &BitInt) {
        self.val *= &rhs.val;
        self.sign ^= rhs.sign;
    }
}

macro_rules! impl_mul_assign_bi_prim {
    ($($t:ty),*) => {
        $(
            impl MulAssign<$t> for BitInt {
                #[inline]
                fn mul_assign(&mut self, rhs: $t) {
                    self.val *= rhs.unsigned_abs();
                    self.sign ^= rhs < 0;
                }
            }
        )*
    };
}

impl_mul_assign_bi_prim!(i128, i64, isize, i32, i16, i8);

//implementing Div for BitInt and signed primitive types

impl Div for BitInt {
    type Output = BitInt;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        BitInt {
            val: self.val / rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Div for &BitInt {
    type Output = BitInt;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        BitInt {
            val: &self.val / &rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Div<&BitInt> for BitInt {
    type Output = BitInt;

    #[inline]
    fn div(self, rhs: &BitInt) -> Self::Output {
        BitInt {
            val: self.val / &rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Div<BitInt> for &BitInt {
    type Output = BitInt;

    #[inline]
    fn div(self, rhs: BitInt) -> Self::Output {
        BitInt {
            val: &self.val / rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

macro_rules! impl_div_bi_prim {
    ($($t:ty),*) => {
        $(
            impl Div<$t> for BitInt {
                type Output = BitInt;

                #[inline]
                fn div(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: self.val / rhs.unsigned_abs(),
                        sign: self.sign ^ (rhs < 0),
                    }
                }
            }

            impl Div<$t> for &BitInt {
                type Output = BitInt;

                #[inline]
                fn div(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: &self.val / rhs.unsigned_abs(),
                        sign: self.sign ^ (rhs < 0),
                    }
                }
            }

            impl Div<BitInt> for $t {
                type Output = BitInt;

                #[inline]
                fn div(self, rhs: BitInt) -> Self::Output {
                    BitInt {
                        val: self.unsigned_abs() / rhs.val,
                        sign: (self < 0) ^ rhs.sign,
                    }
                }
            }

            impl Div<&BitInt> for $t {
                type Output = BitInt;

                #[inline]
                fn div(self, rhs: &BitInt) -> Self::Output {
                    BitInt {
                        val: self.unsigned_abs() / &rhs.val,
                        sign: (self < 0) ^ rhs.sign,
                    }
                }
            }
        )*
    };
}

impl_div_bi_prim!(i128, i64, isize, i32, i16, i8);

//implementing Div Assign for BitInt and signed primitive types

impl DivAssign for BitInt {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.val /= rhs.val;
        self.sign ^= rhs.sign;
    }
}

impl DivAssign<&BitInt> for BitInt {
    #[inline]
    fn div_assign(&mut self, rhs: &BitInt) {
        self.val /= &rhs.val;
        self.sign ^= rhs.sign;
    }
}

macro_rules! impl_div_assign_bi_prim {
    ($($t:ty),*) => {
        $(
            impl DivAssign<$t> for BitInt {
                #[inline]
                fn div_assign(&mut self, rhs: $t) {
                    self.val /= rhs.unsigned_abs();
                    self.sign ^= rhs < 0;
                }
            }
        )*
    };
}

impl_div_assign_bi_prim!(i128, i64, isize, i32, i16, i8);

//implementing Modulus for BitInt and signed primitive types

impl Rem for BitInt {
    type Output = BitInt;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        BitInt {
            val: self.val % rhs.val,
            sign: self.sign,
        }
    }
}

impl Rem for &BitInt {
    type Output = BitInt;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        BitInt {
            val: &self.val % &rhs.val,
            sign: self.sign,
        }
    }
}

impl Rem<&BitInt> for BitInt {
    type Output = BitInt;

    #[inline]
    fn rem(self, rhs: &BitInt) -> Self::Output {
        BitInt {
            val: self.val % &rhs.val,
            sign: self.sign,
        }
    }
}

impl Rem<BitInt> for &BitInt {
    type Output = BitInt;

    #[inline]
    fn rem(self, rhs: BitInt) -> Self::Output {
        BitInt {
            val: &self.val % rhs.val,
            sign: self.sign,
        }
    }
}

macro_rules! impl_rem_bi_prim {
    ($($t:ty),*) => {
        $(
            impl Rem<$t> for BitInt {
                type Output = BitInt;

                #[inline]
                fn rem(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: self.val % rhs.unsigned_abs(),
                        sign: self.sign,
                    }
                }
            }

            impl Rem<$t> for &BitInt {
                type Output = BitInt;

                #[inline]
                fn rem(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: &self.val % rhs.unsigned_abs(),
                        sign: self.sign,
                    }
                }
            }

            impl Rem<BitInt> for $t {
                type Output = BitInt;

                #[inline]
                fn rem(self, rhs: BitInt) -> Self::Output {
                    BitInt {
                        val: self.unsigned_abs() % rhs.val,
                        sign: self < 0,
                    }
                }
            }

            impl Rem<&BitInt> for $t {
                type Output = BitInt;

                #[inline]
                fn rem(self, rhs: &BitInt) -> Self::Output {
                    BitInt {
                        val: self.unsigned_abs() % &rhs.val,
                        sign: self < 0,
                    }
                }
            }
        )*
    };
}

impl_rem_bi_prim!(i128, i64, isize, i32, i16, i8);

//implementing Mul Assign for BitInt and signed primitive types

impl RemAssign for BitInt {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        self.val %= rhs.val;
    }
}

impl RemAssign<&BitInt> for BitInt {
    #[inline]
    fn rem_assign(&mut self, rhs: &BitInt) {
        self.val %= &rhs.val;
    }
}

macro_rules! impl_rem_assign_bi_prim {
    ($($t:ty),*) => {
        $(
            impl RemAssign<$t> for BitInt {
                #[inline]
                fn rem_assign(&mut self, rhs: $t) {
                    self.val %= rhs.unsigned_abs();
                }
            }
        )*
    };
}

impl_rem_assign_bi_prim!(i128, i64, isize, i32, i16, i8);

//implementing Shr for BitInt

macro_rules! impl_shr_bi_prim {
    ($($t:ty),*) => {
        $(
            impl Shr<$t> for BitInt {
                type Output = BitInt;

                #[inline]
                fn shr(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: self.val >> rhs,
                        sign: self.sign,
                    }
                }
            }

            impl Shr<$t> for &BitInt {
                type Output = BitInt;

                #[inline]
                fn shr(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: &self.val >> rhs,
                        sign: self.sign,
                    }
                }
            }
        )*
    };
}

impl_shr_bi_prim!(u128, u64, usize, u32, u16, u8);

//implementing shr assign for BitInt

macro_rules! impl_shr_assign_bi_prim {
    ($($t:ty),*) => {
        $(
            impl ShrAssign<$t> for BitInt {
                #[inline]
                fn shr_assign(&mut self, rhs: $t) {
                    self.val >>= rhs;
                }
            }
        )*
    };
}

impl_shr_assign_bi_prim!(u128, u64, usize, u32, u16, u8);

//implementing shl for BitInt

macro_rules! impl_shl_bi_prim {
    ($($t:ty),*) => {
        $(
            impl Shl<$t> for BitInt {
                type Output = BitInt;

                #[inline]
                fn shl(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: self.val << rhs,
                        sign: self.sign,
                    }
                }
            }

            impl Shl<$t> for &BitInt {
                type Output = BitInt;

                #[inline]
                fn shl(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: &self.val << rhs,
                        sign: self.sign,
                    }
                }
            }
        )*
    };
}

impl_shl_bi_prim!(u128, u64, usize, u32, u16, u8);

//implementing shl assign for BitInt

macro_rules! impl_shl_assign_bi_prim {
    ($($t:ty),*) => {
        $(
            impl ShlAssign<$t> for BitInt {
                #[inline]
                fn shl_assign(&mut self, rhs: $t) {
                    self.val <<= rhs;
                }
            }
        )*
    };
}

impl_shl_assign_bi_prim!(u128, u64, usize, u32, u16, u8);

//implementing Pow for BitInt

impl Pow<UBitInt> for BitInt {
    type Output = BitInt;

    #[inline]
    fn pow(self, rhs: UBitInt) -> Self::Output {
        BitInt {
            val: self.val.pow(&rhs),
            sign: rhs % 2_usize == 1_usize,
        }
    }
}

impl Pow<&UBitInt> for BitInt {
    type Output = BitInt;

    #[inline]
    fn pow(self, rhs: &UBitInt) -> Self::Output {
        BitInt {
            val: self.val.pow(rhs),
            sign: rhs % 2_usize == 1_usize,
        }
    }
}

impl Pow<UBitInt> for &BitInt {
    type Output = BitInt;

    #[inline]
    fn pow(self, rhs: UBitInt) -> Self::Output {
        BitInt {
            val: (&self.val).pow(&rhs),
            sign: rhs % 2_usize == 1_usize,
        }
    }
}

impl Pow<&UBitInt> for &BitInt {
    type Output = BitInt;

    #[inline]
    fn pow(self, rhs: &UBitInt) -> Self::Output {
        BitInt {
            val: (&self.val).pow(rhs),
            sign: rhs % 2_usize == 1_usize,
        }
    }
}

macro_rules! impl_pow_bi_prim {
    ($($t:ty),*) => {
        $(
            impl Pow<$t> for BitInt {
                type Output = BitInt;

                #[inline]
                fn pow(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: self.val.pow(rhs),
                        sign: rhs % 2 == 1,
                    }
                }
            }

            impl Pow<$t> for &BitInt {
                type Output = BitInt;

                #[inline]
                fn pow(self, rhs: $t) -> Self::Output {
                    BitInt {
                        val: (&self.val).pow(rhs),
                        sign: rhs % 2 == 1,
                    }
                }
            }
        )*
    };
}

impl_pow_bi_prim!(u128, u64, usize, u32, u16, u8);
