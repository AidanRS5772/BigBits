#![allow(dead_code)]

use crate::bitint::*;
use crate::ubitint::*;
use core::fmt;
use std::cmp::Ordering::*;
use std::ops::*;

#[derive(Debug, Clone)]
pub struct BitFrac {
    pub n: UBitInt,
    pub d: UBitInt,
    pub sign: bool,
}

impl BitFrac {
    #[inline]
    pub fn get_num(&self) -> UBitInt {
        self.n.clone()
    }

    #[inline]
    pub fn get_den(&self) -> UBitInt {
        self.d.clone()
    }

    #[inline]
    pub fn get_sign(&self) -> bool {
        self.sign
    }

    #[inline]
    pub fn from<T: SignVal>(n: T, d: T) -> Result<BitFrac, String> {
        let (n_sign, mut n_data) = n.sign_and_val();
        let (d_sign, mut d_data) = d.sign_and_val();

        if let Some(idx) = n_data.iter().rposition(|&x| x != 0) {
            n_data.truncate(idx + 1);
        } else {
            n_data.clear();
        }

        if let Some(idx) = d_data.iter().rposition(|&x| x != 0) {
            d_data.truncate(idx + 1);
        } else {
            return Err("the denominator cannot be zero".to_string());
        }

        Ok(BitFrac {
            n: UBitInt { data: n_data },
            d: UBitInt { data: d_data },
            sign: n_sign ^ d_sign,
        })
    }

    #[inline]
    pub fn from_bi(n: BitInt, d: BitInt) -> BitFrac {
        BitFrac {
            n: n.get_val(),
            d: d.get_val(),
            sign: n.get_sign() ^ d.get_sign(),
        }
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
    pub fn simplify(&mut self) {
        let mut gcd = self.n.clone();
        let mut rem = self.d.clone();
        while rem > 0_usize {
            let tmp = rem.clone();
            rem = &gcd % rem;
            gcd = tmp;
        }

        self.n /= &gcd;
        self.d /= gcd;
    }

    #[inline]
    pub fn abs_cmp(&self, other: &BitFrac) -> std::cmp::Ordering {
        (&self.n * &other.d)
            .partial_cmp(&(&other.n * &self.d))
            .unwrap()
    }
}

impl fmt::Display for BitFrac {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sign {
            write!(f, "-{}  //  {}", self.n.to_string(), self.d.to_string())?;
        } else {
            write!(f, "{}  //  {}", self.n.to_string(), self.d.to_string())?;
        }
        Ok(())
    }
}

impl PartialEq for BitFrac {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.sign ^ other.sign {
            return false;
        }

        return &self.n * &other.d == &other.n * &self.d;
    }
}

macro_rules! impl_peq_bfc_prim {
    ($($t:ty),*) => {
        $(
            impl PartialEq<$t> for BitFrac {
                #[inline]
                fn eq(&self, other: &$t) -> bool {
                    if self.sign ^ (*other < 0) {
                        return false;
                    }

                    return self.n == other.unsigned_abs() * &self.d;
                }
            }
        )*
    };
}

impl_peq_bfc_prim!(i128, i64, isize, i32, i16, i8);

impl PartialOrd for BitFrac {
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
            return Some((&self.n * &other.d).cmp(&(&other.n * &self.d)).reverse());
        } else {
            return (&self.n * &other.d).partial_cmp(&(&other.n * &self.d));
        }
    }
}

macro_rules! impl_pord_bfc_prim {
    ($($t:ty),*) => {
        $(
            impl PartialOrd<$t> for BitFrac {
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
                            self.n
                                .partial_cmp(&(other.unsigned_abs() * &self.d))
                                .unwrap()
                                .reverse(),
                        );
                    } else {
                        return self.n.partial_cmp(&(other.unsigned_abs() * &self.d));
                    }
                }
            }
        )*
    };
}

impl_pord_bfc_prim!(i128, i64, isize, i32, i16, i8);

impl Eq for BitFrac {}

impl Ord for BitFrac {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Add for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if self.sign ^ rhs.sign {
            match self.abs_cmp(&rhs) {
                Greater => BitFrac {
                    n: self.n * &rhs.d - rhs.n * &self.d,
                    d: self.d * rhs.d,
                    sign: self.sign,
                },
                Less => BitFrac {
                    n: rhs.n * &self.d - self.n * &rhs.d,
                    d: self.d * rhs.d,
                    sign: rhs.sign,
                },
                Equal => BitFrac {
                    n: UBitInt { data: vec![0] },
                    d: UBitInt { data: vec![1] },
                    sign: false,
                },
            }
        } else {
            println!("here 4");
            BitFrac {
                n: self.n * &rhs.d + rhs.n * &self.d,
                d: self.d * rhs.d,
                sign: self.sign,
            }
        }
    }
}

impl Add for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if self.sign ^ rhs.sign {
            match self.abs_cmp(&rhs) {
                Greater => BitFrac {
                    n: &self.n * &rhs.d - &rhs.n * &self.d,
                    d: &self.d * &rhs.d,
                    sign: self.sign,
                },
                Less => BitFrac {
                    n: &rhs.n * &self.d - &self.n * &rhs.d,
                    d: &self.d * &rhs.d,
                    sign: rhs.sign,
                },
                Equal => BitFrac {
                    n: UBitInt { data: vec![0] },
                    d: UBitInt { data: vec![1] },
                    sign: false,
                },
            }
        } else {
            BitFrac {
                n: &self.n * &rhs.d + &rhs.n * &self.d,
                d: &self.d * &rhs.d,
                sign: self.sign,
            }
        }
    }
}

impl Add<&BitFrac> for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn add(self, rhs: &BitFrac) -> Self::Output {
        if self.sign ^ rhs.sign {
            match self.abs_cmp(&rhs) {
                Greater => BitFrac {
                    n: self.n * &rhs.d - &rhs.n * &self.d,
                    d: self.d * &rhs.d,
                    sign: self.sign,
                },
                Less => BitFrac {
                    n: &rhs.n * &self.d - self.n * &rhs.d,
                    d: self.d * &rhs.d,
                    sign: rhs.sign,
                },
                Equal => BitFrac {
                    n: UBitInt { data: vec![0] },
                    d: UBitInt { data: vec![1] },
                    sign: false,
                },
            }
        } else {
            BitFrac {
                n: self.n * &rhs.d + &rhs.n * &self.d,
                d: self.d * &rhs.d,
                sign: self.sign,
            }
        }
    }
}

impl Add<BitFrac> for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn add(self, rhs: BitFrac) -> Self::Output {
        if self.sign ^ rhs.sign {
            match self.abs_cmp(&rhs) {
                Greater => BitFrac {
                    n: &self.n * &rhs.d - rhs.n * &self.d,
                    d: &self.d * rhs.d,
                    sign: self.sign,
                },
                Less => BitFrac {
                    n: rhs.n * &self.d - &self.n * &rhs.d,
                    d: &self.d * rhs.d,
                    sign: rhs.sign,
                },
                Equal => BitFrac {
                    n: UBitInt { data: vec![0] },
                    d: UBitInt { data: vec![1] },
                    sign: false,
                },
            }
        } else {
            BitFrac {
                n: &self.n * &rhs.d + rhs.n * &self.d,
                d: &self.d * rhs.d,
                sign: self.sign,
            }
        }
    }
}

macro_rules! impl_add_bfc_prim {
    ($($t:ty),*) => {
        $(
            impl Add<$t> for BitFrac {
                type Output = BitFrac;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    let sc_rhs = rhs.unsigned_abs() * &self.d;
                    if self.sign ^ (rhs < 0) {
                        match self.n.cmp(&sc_rhs) {
                            Greater => BitFrac {
                                n: self.n - sc_rhs,
                                d: self.d,
                                sign: self.sign,
                            },
                            Less => BitFrac {
                                n: sc_rhs - self.n,
                                d: self.d,
                                sign: rhs < 0,
                            },
                            Equal => BitFrac {
                                n: UBitInt { data: vec![0] },
                                d: UBitInt { data: vec![1] },
                                sign: false,
                            },
                        }
                    } else {
                        BitFrac {
                            n: self.n + sc_rhs,
                            d: self.d,
                            sign: self.sign,
                        }
                    }
                }
            }

            impl Add<$t> for &BitFrac {
                type Output = BitFrac;

                #[inline]
                fn add(self, rhs: $t) -> Self::Output {
                    let sc_rhs = rhs.unsigned_abs() * &self.d;
                    if self.sign ^ (rhs < 0) {
                        match self.n.cmp(&sc_rhs) {
                            Greater => BitFrac {
                                n: &self.n - sc_rhs,
                                d: self.d.clone(),
                                sign: self.sign,
                            },
                            Less => BitFrac {
                                n: sc_rhs - &self.n,
                                d: self.d.clone(),
                                sign: rhs < 0,
                            },
                            Equal => BitFrac {
                                n: UBitInt { data: vec![0] },
                                d: UBitInt { data: vec![1] },
                                sign: false,
                            },
                        }
                    } else {
                        BitFrac {
                            n: &self.n + sc_rhs,
                            d: self.d.clone(),
                            sign: self.sign,
                        }
                    }
                }
            }

            impl Add<BitFrac> for $t {
                type Output = BitFrac;

                #[inline]
                fn add(self, rhs: BitFrac) -> Self::Output {
                    let sc_self = self.unsigned_abs() * &rhs.d;
                    if rhs.sign ^ (self < 0) {
                        match rhs.n.cmp(&sc_self) {
                            Greater => BitFrac {
                                n: rhs.n - sc_self,
                                d: rhs.d,
                                sign: rhs.sign,
                            },
                            Less => BitFrac {
                                n: sc_self - rhs.n,
                                d: rhs.d,
                                sign: self < 0,
                            },
                            Equal => BitFrac {
                                n: UBitInt { data: vec![0] },
                                d: UBitInt { data: vec![1] },
                                sign: false,
                            },
                        }
                    } else {
                        BitFrac {
                            n: rhs.n + sc_self,
                            d: rhs.d,
                            sign: rhs.sign,
                        }
                    }
                }
            }

            impl Add<&BitFrac> for $t {
                type Output = BitFrac;

                #[inline]
                fn add(self, rhs: &BitFrac) -> Self::Output {
                    let sc_self = self.unsigned_abs() * &rhs.d;
                    if rhs.sign ^ (self < 0) {
                        match rhs.n.cmp(&sc_self) {
                            Greater => BitFrac {
                                n: &rhs.n - sc_self,
                                d: rhs.d.clone(),
                                sign: rhs.sign,
                            },
                            Less => BitFrac {
                                n: sc_self - &rhs.n,
                                d: rhs.d.clone(),
                                sign: self < 0,
                            },
                            Equal => BitFrac {
                                n: UBitInt { data: vec![0] },
                                d: UBitInt { data: vec![1] },
                                sign: false,
                            },
                        }
                    } else {
                        BitFrac {
                            n: &rhs.n + sc_self,
                            d: rhs.d.clone(),
                            sign: rhs.sign,
                        }
                    }
                }
            }
        )*
    };
}

impl_add_bfc_prim!(i128, i64, isize, i32, i16, i8);

impl AddAssign for BitFrac {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        if self.sign ^ rhs.sign {
            match (*self).cmp(&rhs) {
                Greater => {
                    self.n *= &rhs.d;
                    self.n -= rhs.n * &self.d;
                    self.d *= rhs.d;
                }
                Less => {
                    self.n *= &rhs.d;
                    self.n = rhs.n * &self.d - &self.n;
                    self.d *= rhs.d;
                    self.sign = rhs.sign;
                }
                Equal => {
                    *self = BitFrac {
                        n: UBitInt { data: vec![0] },
                        d: UBitInt { data: vec![1] },
                        sign: false,
                    };
                }
            }
        } else {
            self.n *= &rhs.d;
            self.n += rhs.n * &self.d;
            self.d *= rhs.d;
        }
    }
}

impl AddAssign<&BitFrac> for BitFrac {
    #[inline]
    fn add_assign(&mut self, rhs: &BitFrac) {
        if self.sign ^ rhs.sign {
            match (*self).abs_cmp(rhs) {
                Greater => {
                    self.n *= &rhs.d;
                    self.n -= &rhs.n * &self.d;
                    self.d *= &rhs.d;
                }
                Less => {
                    self.n *= &rhs.d;
                    self.n = &rhs.n * &self.d - &self.n;
                    self.d *= &rhs.d;
                    self.sign = rhs.sign;
                }
                Equal => {
                    *self = BitFrac {
                        n: UBitInt { data: vec![0] },
                        d: UBitInt { data: vec![1] },
                        sign: false,
                    };
                }
            }
        } else {
            self.n *= &rhs.d;
            self.n += &rhs.n * &self.d;
            self.d *= &rhs.d;
        }
    }
}

macro_rules! impl_add_assign_bfc_prim {
    ($($t:ty),*) => {
        $(
            impl AddAssign<$t> for BitFrac{
                #[inline]
                fn add_assign(&mut self, rhs: $t){
                    let sc_rhs = rhs.unsigned_abs() * &self.d;
                    if self.sign ^ (rhs < 0) {
                        match self.n.cmp(&sc_rhs) {
                            Greater => {
                                self.n -= sc_rhs;
                            }
                            Less => {
                                self.n = sc_rhs - &self.n;
                                self.sign = rhs < 0;
                            }
                            Equal => {
                                *self = BitFrac {
                                    n: UBitInt { data: vec![0] },
                                    d: UBitInt { data: vec![1] },
                                    sign: false,
                                }
                            }
                        }
                    } else {
                        self.n += sc_rhs;
                    }
                }
            }
        )*
    };
}

impl_add_assign_bfc_prim!(i128, i64, isize, i32, i16, i8);

impl Sub for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if self.sign ^ rhs.sign {
            BitFrac {
                n: self.n * &rhs.d + rhs.n * &self.d,
                d: self.d * rhs.d,
                sign: self.sign,
            }
        } else {
            match self.abs_cmp(&rhs) {
                Greater => BitFrac {
                    n: self.n * &rhs.d - rhs.n * &self.d,
                    d: self.d * rhs.d,
                    sign: self.sign,
                },
                Less => BitFrac {
                    n: rhs.n * &self.d - self.n * &rhs.d,
                    d: self.d * rhs.d,
                    sign: rhs.sign,
                },
                Equal => BitFrac {
                    n: UBitInt { data: vec![0] },
                    d: UBitInt { data: vec![1] },
                    sign: false,
                },
            }
        }
    }
}

impl Sub for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if self.sign ^ rhs.sign {
            BitFrac {
                n: &self.n * &rhs.d + &rhs.n * &self.d,
                d: &self.d * &rhs.d,
                sign: self.sign,
            }
        } else {
            match self.abs_cmp(&rhs) {
                Greater => BitFrac {
                    n: &self.n * &rhs.d - &rhs.n * &self.d,
                    d: &self.d * &rhs.d,
                    sign: self.sign,
                },
                Less => BitFrac {
                    n: &rhs.n * &self.d - &self.n * &rhs.d,
                    d: &self.d * &rhs.d,
                    sign: rhs.sign,
                },
                Equal => BitFrac {
                    n: UBitInt { data: vec![0] },
                    d: UBitInt { data: vec![1] },
                    sign: false,
                },
            }
        }
    }
}

impl Sub<&BitFrac> for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn sub(self, rhs: &BitFrac) -> Self::Output {
        if self.sign ^ rhs.sign {
            BitFrac {
                n: self.n * &rhs.d + &rhs.n * &self.d,
                d: self.d * &rhs.d,
                sign: self.sign,
            }
        } else {
            match self.abs_cmp(&rhs) {
                Greater => BitFrac {
                    n: self.n * &rhs.d - &rhs.n * &self.d,
                    d: self.d * &rhs.d,
                    sign: self.sign,
                },
                Less => BitFrac {
                    n: &rhs.n * &self.d - self.n * &rhs.d,
                    d: self.d * &rhs.d,
                    sign: rhs.sign,
                },
                Equal => BitFrac {
                    n: UBitInt { data: vec![0] },
                    d: UBitInt { data: vec![1] },
                    sign: false,
                },
            }
        }
    }
}

impl Sub<BitFrac> for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn sub(self, rhs: BitFrac) -> Self::Output {
        if self.sign ^ rhs.sign {
            BitFrac {
                n: &self.n * &rhs.d + rhs.n * &self.d,
                d: &self.d * rhs.d,
                sign: self.sign,
            }
        } else {
            match self.abs_cmp(&rhs) {
                Greater => BitFrac {
                    n: &self.n * &rhs.d - rhs.n * &self.d,
                    d: &self.d * rhs.d,
                    sign: self.sign,
                },
                Less => BitFrac {
                    n: rhs.n * &self.d - &self.n * &rhs.d,
                    d: &self.d * rhs.d,
                    sign: rhs.sign,
                },
                Equal => BitFrac {
                    n: UBitInt { data: vec![0] },
                    d: UBitInt { data: vec![1] },
                    sign: false,
                },
            }
        }
    }
}

macro_rules! impl_sub_bfc_prim {
    ($($t:ty),*) => {
        $(
            impl Sub<$t> for BitFrac {
                type Output = BitFrac;

                #[inline]
                fn sub(self, rhs: $t) -> Self::Output {
                    let sc_rhs = rhs.unsigned_abs() * &self.d;
                    if self.sign ^ (rhs < 0) {
                        BitFrac {
                            n: self.n + sc_rhs,
                            d: self.d,
                            sign: self.sign,
                        }
                    } else {
                        match self.n.cmp(&sc_rhs) {
                            Greater => BitFrac {
                                n: self.n - sc_rhs,
                                d: self.d,
                                sign: self.sign,
                            },
                            Less => BitFrac {
                                n: sc_rhs - self.n,
                                d: self.d,
                                sign: rhs < 0,
                            },
                            Equal => BitFrac {
                                n: UBitInt { data: vec![0] },
                                d: UBitInt { data: vec![1] },
                                sign: false,
                            },
                        }
                    }
                }
            }

            impl Sub<$t> for &BitFrac {
                type Output = BitFrac;

                #[inline]
                fn sub(self, rhs: $t) -> Self::Output {
                    let sc_rhs = rhs.unsigned_abs() * &self.d;
                    if self.sign ^ (rhs < 0) {
                        BitFrac {
                            n: &self.n + sc_rhs,
                            d: self.d.clone(),
                            sign: self.sign,
                        }
                    } else {
                        match self.n.cmp(&sc_rhs) {
                            Greater => BitFrac {
                                n: &self.n - sc_rhs,
                                d: self.d.clone(),
                                sign: self.sign,
                            },
                            Less => BitFrac {
                                n: sc_rhs - &self.n,
                                d: self.d.clone(),
                                sign: rhs < 0,
                            },
                            Equal => BitFrac {
                                n: UBitInt { data: vec![0] },
                                d: UBitInt { data: vec![1] },
                                sign: false,
                            },
                        }
                    }
                }
            }

            impl Sub<BitFrac> for $t {
                type Output = BitFrac;

                #[inline]
                fn sub(self, rhs: BitFrac) -> Self::Output {
                    let sc_self = self.unsigned_abs() * &rhs.d;
                    if rhs.sign ^ (self < 0) {
                        BitFrac {
                            n: rhs.n + sc_self,
                            d: rhs.d,
                            sign: rhs.sign,
                        }
                    } else {
                        match rhs.n.cmp(&sc_self) {
                            Greater => BitFrac {
                                n: rhs.n - sc_self,
                                d: rhs.d,
                                sign: rhs.sign,
                            },
                            Less => BitFrac {
                                n: sc_self - rhs.n,
                                d: rhs.d,
                                sign: self < 0,
                            },
                            Equal => BitFrac {
                                n: UBitInt { data: vec![0] },
                                d: UBitInt { data: vec![1] },
                                sign: false,
                            },
                        }
                    }
                }
            }

            impl Sub<&BitFrac> for $t {
                type Output = BitFrac;

                #[inline]
                fn sub(self, rhs: &BitFrac) -> Self::Output {
                    let sc_self = self.unsigned_abs() * &rhs.d;
                    if rhs.sign ^ (self < 0) {
                        BitFrac {
                            n: &rhs.n + sc_self,
                            d: rhs.d.clone(),
                            sign: rhs.sign,
                        }
                    } else {
                        match rhs.n.cmp(&sc_self) {
                            Greater => BitFrac {
                                n: &rhs.n - sc_self,
                                d: rhs.d.clone(),
                                sign: rhs.sign,
                            },
                            Less => BitFrac {
                                n: sc_self - &rhs.n,
                                d: rhs.d.clone(),
                                sign: self < 0,
                            },
                            Equal => BitFrac {
                                n: UBitInt { data: vec![0] },
                                d: UBitInt { data: vec![1] },
                                sign: false,
                            },
                        }
                    }
                }
            }
        )*
    };
}

impl_sub_bfc_prim!(i128, i64, isize, i32, i16, i8);

impl SubAssign for BitFrac {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        if self.sign ^ rhs.sign {
            self.n *= &rhs.d;
            self.n += rhs.n * &self.d;
            self.d *= rhs.d;
        } else {
            match (*self).cmp(&rhs) {
                Greater => {
                    self.n *= &rhs.d;
                    self.n -= rhs.n * &self.d;
                    self.d *= rhs.d;
                }
                Less => {
                    self.n *= &rhs.d;
                    self.n = rhs.n * &self.d - &self.n;
                    self.d *= rhs.d;
                    self.sign = rhs.sign;
                }
                Equal => {
                    *self = BitFrac {
                        n: UBitInt { data: vec![0] },
                        d: UBitInt { data: vec![1] },
                        sign: false,
                    };
                }
            }
        }
    }
}

impl SubAssign<&BitFrac> for BitFrac {
    #[inline]
    fn sub_assign(&mut self, rhs: &BitFrac) {
        if self.sign ^ rhs.sign {
            self.n *= &rhs.d;
            self.n += &rhs.n * &self.d;
            self.d *= &rhs.d;
        } else {
            match (*self).cmp(&rhs) {
                Greater => {
                    self.n *= &rhs.d;
                    self.n -= &rhs.n * &self.d;
                    self.d *= &rhs.d;
                }
                Less => {
                    self.n *= &rhs.d;
                    self.n = &rhs.n * &self.d - &self.n;
                    self.d *= &rhs.d;
                    self.sign = rhs.sign;
                }
                Equal => {
                    *self = BitFrac {
                        n: UBitInt { data: vec![0] },
                        d: UBitInt { data: vec![1] },
                        sign: false,
                    };
                }
            }
        }
    }
}

macro_rules! impl_sub_assign_bfc_prim {
    ($($t:ty),*) => {
        $(
            impl SubAssign<$t> for BitFrac{
                #[inline]
                fn sub_assign(&mut self, rhs: $t){
                    let sc_rhs = rhs.unsigned_abs() * &self.d;
                    if self.sign ^ (rhs < 0) {
                        self.n += sc_rhs;
                    } else {
                        match self.n.cmp(&sc_rhs) {
                            Greater => {
                                self.n -= sc_rhs;
                            }
                            Less => {
                                self.n = sc_rhs - &self.n;
                                self.sign = rhs < 0;
                            }
                            Equal => {
                                *self = BitFrac {
                                    n: UBitInt { data: vec![0] },
                                    d: UBitInt { data: vec![1] },
                                    sign: false,
                                }
                            }
                        }
                    }
                }
            }
        )*
    };
}

impl_sub_assign_bfc_prim!(i128, i64, isize, i32, i16, i8);

impl Mul for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        BitFrac {
            n: self.n * rhs.n,
            d: self.d * rhs.d,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        BitFrac {
            n: &self.n * &rhs.n,
            d: &self.d * &rhs.d,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<&BitFrac> for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn mul(self, rhs: &BitFrac) -> Self::Output {
        BitFrac {
            n: self.n * &rhs.n,
            d: self.d * &rhs.d,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<BitFrac> for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn mul(self, rhs: BitFrac) -> Self::Output {
        BitFrac {
            n: &self.n * rhs.n,
            d: &self.d * rhs.d,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<BitInt> for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn mul(self, rhs: BitInt) -> Self::Output {
        BitFrac {
            n: &self.n * rhs.val,
            d: self.d,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<BitInt> for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn mul(self, rhs: BitInt) -> Self::Output {
        BitFrac {
            n: &self.n * rhs.val,
            d: self.d.clone(),
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<&BitInt> for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn mul(self, rhs: &BitInt) -> Self::Output {
        BitFrac {
            n: &self.n * &rhs.val,
            d: self.d,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<&BitInt> for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn mul(self, rhs: &BitInt) -> Self::Output {
        BitFrac {
            n: &self.n * &rhs.val,
            d: self.d.clone(),
            sign: self.sign ^ rhs.sign,
        }
    }
}

macro_rules! impl_mul_bfc_prim {
    ($($t:ty),*) => {
        $(
            impl Mul<$t> for BitFrac{
                type Output = BitFrac;

                #[inline]
                fn mul(self, rhs: $t) -> Self::Output{
                    BitFrac{
                        n: self.n*rhs.unsigned_abs(),
                        d: self.d,
                        sign: self.sign ^ (rhs < 0),
                    }
                }
            }

            impl Mul<$t> for &BitFrac{
                type Output = BitFrac;

                #[inline]
                fn mul(self, rhs: $t) -> Self::Output{
                    BitFrac{
                        n: &self.n*rhs.unsigned_abs(),
                        d: self.d.clone(),
                        sign: self.sign ^ (rhs < 0),
                    }
                }
            }

            impl Mul<BitFrac> for $t{
                type Output = BitFrac;

                #[inline]
                fn mul(self, rhs: BitFrac) -> Self::Output{
                    BitFrac{
                        n: self.unsigned_abs()*rhs.n,
                        d: rhs.d,
                        sign: rhs.sign ^ (self < 0),
                    }
                }
            }

            impl Mul<&BitFrac> for $t{
                type Output = BitFrac;

                #[inline]
                fn mul(self, rhs: &BitFrac) -> Self::Output{
                    BitFrac{
                        n: self.unsigned_abs()*&rhs.n,
                        d: rhs.d.clone(),
                        sign: rhs.sign ^ (self < 0),
                    }
                }
            }
        )*
    };
}

impl_mul_bfc_prim!(i128, i64, isize, i32, i16, i8);

impl MulAssign for BitFrac {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.n *= rhs.n;
        self.d *= rhs.d;
        self.sign ^= rhs.sign;
    }
}

impl MulAssign<&BitFrac> for BitFrac {
    #[inline]
    fn mul_assign(&mut self, rhs: &BitFrac) {
        self.n *= &rhs.n;
        self.d *= &rhs.d;
        self.sign ^= rhs.sign;
    }
}

impl MulAssign<BitInt> for BitFrac {
    #[inline]
    fn mul_assign(&mut self, rhs: BitInt) {
        self.n *= rhs.val;
        self.sign ^= rhs.sign;
    }
}

impl MulAssign<&BitInt> for BitFrac {
    #[inline]
    fn mul_assign(&mut self, rhs: &BitInt) {
        self.n *= &rhs.val;
        self.sign ^= rhs.sign;
    }
}

macro_rules! impl_mul_assign_bfc_prim {
    ($($t:ty),*) => {
        $(
            impl MulAssign<$t> for BitFrac{
                #[inline]
                fn mul_assign(&mut self, rhs: $t){
                    self.n *= rhs.unsigned_abs();
                    self.sign ^= rhs < 0;
                }
            }
        )*
    };
}

impl_mul_assign_bfc_prim!(i128, i64, isize, i32, i16, i8);

impl Div for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        BitFrac {
            n: self.n * rhs.d,
            d: self.d * rhs.n,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Div for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        BitFrac {
            n: &self.n * &rhs.d,
            d: &self.d * &rhs.n,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Div<&BitFrac> for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn div(self, rhs: &BitFrac) -> Self::Output {
        BitFrac {
            n: self.n * &rhs.d,
            d: self.d * &rhs.n,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Div<BitFrac> for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn div(self, rhs: BitFrac) -> Self::Output {
        BitFrac {
            n: &self.n * rhs.d,
            d: &self.d * rhs.n,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Div<BitInt> for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn div(self, rhs: BitInt) -> Self::Output {
        BitFrac {
            n: self.n,
            d: self.d * rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Div<BitInt> for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn div(self, rhs: BitInt) -> Self::Output {
        BitFrac {
            n: self.n.clone(),
            d: &self.d * rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Div<&BitInt> for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn div(self, rhs: &BitInt) -> Self::Output {
        BitFrac {
            n: self.n,
            d: self.d * &rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Div<&BitInt> for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn div(self, rhs: &BitInt) -> Self::Output {
        BitFrac {
            n: self.n.clone(),
            d: &self.d * &rhs.val,
            sign: self.sign ^ rhs.sign,
        }
    }
}

macro_rules! impl_div_bfc_prim {
    ($($t:ty),*) => {
        $(
            impl Div<$t> for BitFrac{
                type Output = BitFrac;

                #[inline]
                fn div(self, rhs: $t) -> Self::Output{
                    BitFrac{
                        n: self.n,
                        d: self.d*rhs.unsigned_abs(),
                        sign: self.sign ^ (rhs < 0),
                    }
                }
            }

            impl Div<$t> for &BitFrac{
                type Output = BitFrac;

                #[inline]
                fn div(self, rhs: $t) -> Self::Output{
                    BitFrac{
                        n: self.n.clone(),
                        d: &self.d*rhs.unsigned_abs(),
                        sign: self.sign ^ (rhs < 0),
                    }
                }
            }

            impl Div<BitFrac> for $t{
                type Output = BitFrac;

                #[inline]
                fn div(self, rhs: BitFrac) -> Self::Output{
                    BitFrac{
                        n: rhs.n,
                        d: rhs.d*self.unsigned_abs(),
                        sign: rhs.sign ^ (self < 0),
                    }
                }
            }

            impl Div<&BitFrac> for $t{
                type Output = BitFrac;

                #[inline]
                fn div(self, rhs: &BitFrac) -> Self::Output{
                    BitFrac{
                        n: rhs.n.clone(),
                        d: &rhs.d*self.unsigned_abs(),
                        sign: rhs.sign ^ (self < 0),
                    }
                }
            }
        )*
    };
}

impl_div_bfc_prim!(i128, i64, isize, i32, i16, i8);

impl DivAssign for BitFrac {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.n *= rhs.d;
        self.d *= rhs.n;
        self.sign ^= rhs.sign;
    }
}

impl DivAssign<&BitFrac> for BitFrac {
    #[inline]
    fn div_assign(&mut self, rhs: &BitFrac) {
        self.n *= &rhs.d;
        self.d *= &rhs.n;
        self.sign ^= rhs.sign;
    }
}

impl DivAssign<BitInt> for BitFrac {
    #[inline]
    fn div_assign(&mut self, rhs: BitInt) {
        self.d *= rhs.val;
        self.sign ^= rhs.sign;
    }
}

impl DivAssign<&BitInt> for BitFrac {
    #[inline]
    fn div_assign(&mut self, rhs: &BitInt) {
        self.d *= &rhs.val;
        self.sign ^= rhs.sign;
    }
}

macro_rules! impl_div_assign_bfc_prim {
    ($($t:ty),*) => {
        $(
            impl DivAssign<$t> for BitFrac{
                #[inline]
                fn div_assign(&mut self, rhs: $t){
                    self.d *= rhs.unsigned_abs();
                    self.sign ^= rhs < 0;
                }
            }
        )*
    };
}

impl_div_assign_bfc_prim!(i128, i64, isize, i32, i16, i8);

impl Neg for BitFrac {
    type Output = BitFrac;

    #[inline]
    fn neg(self) -> Self::Output {
        BitFrac {
            n: self.n,
            d: self.d,
            sign: !self.sign,
        }
    }
}

impl Neg for &BitFrac {
    type Output = BitFrac;

    #[inline]
    fn neg(self) -> Self::Output {
        BitFrac {
            n: self.n.clone(),
            d: self.d.clone(),
            sign: !self.sign,
        }
    }
}
