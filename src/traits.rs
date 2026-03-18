#![allow(dead_code)]
use std::ops::{Deref, DerefMut, Div, Neg, Rem};

use crate::utils::{lsb, signed_shl, signed_shr};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SmallBuf {
    limbs: [u64; 2],
    len: usize,
}

impl SmallBuf {
    pub const ZERO: Self = Self {
        limbs: [0; 2],
        len: 0,
    };

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Deref for SmallBuf {
    type Target = [u64];
    fn deref(&self) -> &Self::Target {
        &self.limbs[..self.len]
    }
}

impl DerefMut for SmallBuf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.limbs[..self.len]
    }
}

#[derive(Debug)]
pub enum FromErr {
    Overflow,
    Underflow,
}

impl TryFrom<&[u64]> for SmallBuf {
    type Error = FromErr;
    fn try_from(buf: &[u64]) -> Result<Self, Self::Error> {
        if buf.len() > 2 {
            return Err(FromErr::Overflow);
        }
        let mut limbs = [0; 2];
        limbs[..buf.len()].copy_from_slice(buf);
        return Ok(SmallBuf {
            limbs,
            len: buf.len(),
        });
    }
}

impl From<u128> for SmallBuf {
    fn from(value: u128) -> Self {
        let lo = value as u64;
        let hi = (value >> 64) as u64;
        let end = match (lo, hi) {
            (0, 0) => 0,
            (_, 0) => 1,
            (_, _) => 2,
        };
        SmallBuf {
            limbs: [lo, hi],
            len: end,
        }
    }
}

impl From<SmallBuf> for u128 {
    fn from(buf: SmallBuf) -> Self {
        *buf.get(0).unwrap_or(&0) as u128 | (*buf.get(1).unwrap_or(&0) as u128) << 64
    }
}

impl From<u64> for SmallBuf {
    fn from(value: u64) -> Self {
        return if value == 0 {
            SmallBuf::ZERO
        } else {
            SmallBuf {
                limbs: [value, 0],
                len: 1,
            }
        };
    }
}

impl TryFrom<SmallBuf> for u64 {
    type Error = FromErr;
    fn try_from(value: SmallBuf) -> Result<Self, Self::Error> {
        return if value.len > 1 {
            Err(FromErr::Overflow)
        } else if value.len == 0 {
            Ok(0)
        } else {
            Ok(value.limbs[0])
        };
    }
}

pub(crate) trait U:
    Default + Copy + Into<SmallBuf> + Div<Self, Output = Self> + Rem<Self, Output = Self>
{
}
impl U for u128 {}
impl U for u64 {}

pub(crate) trait I:
    Default + Copy + Div<Self, Output = Self> + Rem<Self, Output = Self>
{
    type U: Into<SmallBuf>;
    fn unsigned(self) -> Self::U;
    fn sign(self) -> bool;
}

impl I for i128 {
    type U = u128;
    fn unsigned(self) -> Self::U {
        self.unsigned_abs()
    }
    fn sign(self) -> bool {
        self.is_negative()
    }
}

impl I for i64 {
    type U = u64;
    fn unsigned(self) -> Self::U {
        self.unsigned_abs()
    }
    fn sign(self) -> bool {
        self.is_negative()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SEM {
    pub(crate) s: bool,
    pub(crate) e: i128,
    pub(crate) m: SmallBuf,
}

impl SEM {
    pub(crate) const ZERO: Self = SEM {
        s: false,
        e: i128::MIN,
        m: SmallBuf::ZERO,
    };

    pub(crate) const POS_INF: Self = SEM {
        s: false,
        e: i128::MAX,
        m: SmallBuf::ZERO,
    };

    pub(crate) const NEG_INF: Self = SEM {
        s: true,
        e: i128::MAX,
        m: SmallBuf::ZERO,
    };
}

impl From<f64> for SEM {
    fn from(value: f64) -> Self {
        let bits = value.to_bits();
        let s = bits >> 63 == 1;
        let exp_bits = ((bits >> 52) & 0x7FF) as i128;
        let mantissa_bits = bits & 0xFFFFFFFFFFFFF;

        let (sig, exp) = match (exp_bits, mantissa_bits) {
            (0x7FF, _) => return if s { SEM::NEG_INF } else { SEM::POS_INF },
            (0, 0) => return SEM::ZERO,
            (0, _) => {
                let sh = mantissa_bits.leading_zeros() - 11;
                let sig = mantissa_bits << sh;
                let exp = -1022 - (sh as i128);
                (sig, exp)
            }
            (_, _) => {
                let sig = mantissa_bits | (1 << 52);
                let exp = exp_bits - 1023;
                (sig, exp)
            }
        };

        let e = exp.div_euclid(64);
        let sh = 52 - (exp.rem_euclid(64) as i32);
        let m1 = signed_shr(sig, sh);
        let m0 = lsb(sig, sh) << (64 - sh).min(63).unsigned_abs();
        let m = if m0 == 0 {
            SmallBuf {
                limbs: [m1, 0],
                len: 1,
            }
        } else {
            SmallBuf {
                limbs: [m0, m1],
                len: 2,
            }
        };
        return SEM { s, e, m };
    }
}

impl From<f32> for SEM {
    fn from(value: f32) -> Self {
        let bits = value.to_bits();
        let s = bits >> 31 == 1;
        let exp_bits = ((bits >> 23) & 0xFF) as i128;
        let mantissa_bits = bits & 0x7FFFFF;

        let (sig, exp) = match (exp_bits, mantissa_bits) {
            (0xFF, _) => return if s { SEM::NEG_INF } else { SEM::POS_INF },
            (0, 0) => return SEM::ZERO,
            (0, _) => {
                let sh = mantissa_bits.leading_zeros() - 8;
                let sig = mantissa_bits << sh;
                let exp = -126 - (sh as i128);
                (sig, exp)
            }
            (_, _) => {
                let sig = mantissa_bits | (1 << 23);
                let exp = exp_bits - 127;
                (sig, exp)
            }
        };

        let e = exp.div_euclid(64);
        let sh = 23 - (exp.rem_euclid(64) as i32);
        let m1 = signed_shr(sig as u64, sh);
        let m0 = lsb(sig as u64, sh) << (64 - sh).min(63).unsigned_abs();
        let m = if m0 == 0 {
            SmallBuf {
                limbs: [m1, 0],
                len: 1,
            }
        } else {
            SmallBuf {
                limbs: [m0, m1],
                len: 2,
            }
        };
        return SEM { s, e, m };
    }
}

impl From<i128> for SEM {
    fn from(value: i128) -> Self {
        let uval = value.unsigned_abs();
        let m1 = (uval >> 64) as u64;
        let m0 = uval as u64;
        let limbs = match (m1, m0) {
            (0, 0) => return SEM::ZERO,
            (m1, 0) => [m1, 0],
            (0, m0) => [m0, 0],
            (m1, m0) => [m0, m1],
        };
        return SEM {
            s: value.is_negative(),
            e: if m1 != 0 { 1 } else { 0 },
            m: SmallBuf {
                limbs,
                len: if m1 != 0 && m0 != 0 { 2 } else { 1 },
            },
        };
    }
}

impl From<i64> for SEM {
    fn from(value: i64) -> Self {
        return SEM {
            s: value.is_negative(),
            e: 0,
            m: SmallBuf {
                limbs: [value.unsigned_abs(), 0],
                len: 1,
            },
        };
    }
}

impl TryFrom<SEM> for f64 {
    type Error = FromErr;
    fn try_from(value: SEM) -> Result<Self, Self::Error> {
        if value == SEM::ZERO {
            return Ok(0.0);
        }
        if value == SEM::POS_INF {
            return Ok(f64::INFINITY);
        }
        if value == SEM::NEG_INF {
            return Ok(f64::NEG_INFINITY);
        }

        let msb = value.m.last().unwrap().ilog2() as i32;
        let exp = value.e * 64 + msb as i128;

        if exp > 1023 {
            return Err(FromErr::Overflow);
        }
        if exp < -1074 {
            return Err(FromErr::Underflow);
        }

        let exp_biased = exp + 1023;
        let l = value.m.len - 1;
        let hi = *value.m.get(l).unwrap();
        let lo = *value.m.get(l - 1).unwrap_or(&0);
        let m_bits = if exp_biased > 0 {
            let hi_bits = signed_shl(lsb(hi, msb), 52 - msb);
            let lo_bits = if msb < 52 { lo >> (12 + msb) } else { 0 };
            hi_bits | lo_bits
        } else {
            let hi_bits = signed_shl(hi, 52 - msb);
            let lo_bits = if msb < 52 { lo >> (12 + msb) } else { 0 };
            (hi_bits | lo_bits) >> (1 - exp_biased) as u32
        };
        let exp_bits = (exp_biased.max(0) as u64) << 52;
        let sign_bit = if value.s { 1 << 63 } else { 0 };

        return Ok(f64::from_bits(sign_bit | exp_bits | m_bits));
    }
}

impl TryFrom<SEM> for f32 {
    type Error = FromErr;
    fn try_from(value: SEM) -> Result<Self, Self::Error> {
        if value == SEM::ZERO {
            return Ok(0.0);
        }
        if value == SEM::POS_INF {
            return Ok(f32::INFINITY);
        }
        if value == SEM::NEG_INF {
            return Ok(f32::NEG_INFINITY);
        }

        let msb = value.m.last().unwrap().ilog2() as i32;
        let exp = value.e * 64 + msb as i128;

        if exp > 127 {
            return Err(FromErr::Overflow);
        }
        if exp < -149 {
            return Err(FromErr::Underflow);
        }

        let exp_biased = exp + 127;
        let l = value.m.len - 1;
        let hi = *value.m.get(l).unwrap();
        let lo = *value.m.get(l - 1).unwrap_or(&0);
        let lo_bits = if msb < 23 {
            (lo >> (41 + msb)) as u32
        } else {
            0
        };
        let m_bits = if exp_biased > 0 {
            let hi_bits = signed_shl(lsb(hi, msb), 23 - msb) as u32;
            hi_bits | lo_bits
        } else {
            let hi_bits = signed_shl(hi, 23 - msb) as u32;
            (hi_bits | lo_bits) >> (1 - exp_biased) as u32
        };
        let exp_bits = ((exp_biased.max(0) as u64) << 23) as u32;
        let sign_bit = if value.s { 1 << 31 } else { 0 };

        return Ok(f32::from_bits(sign_bit | exp_bits | m_bits));
    }
}

impl TryFrom<SEM> for i128 {
    type Error = FromErr;
    fn try_from(value: SEM) -> Result<Self, Self::Error> {
        if value.e >= 2 {
            return Err(FromErr::Overflow);
        }

        let uval = if value.e == 0 {
            *value.m.last().unwrap() as u128
        } else if value.e == 1 {
            value.m.into()
        } else {
            0
        };

        let mut val: i128 = uval.try_into().map_err(|_| FromErr::Overflow)?;
        if value.s {
            val = val.neg();
        }
        return Ok(val);
    }
}

impl TryFrom<SEM> for i64 {
    type Error = FromErr;
    fn try_from(value: SEM) -> Result<Self, Self::Error> {
        if value.e >= 1 {
            return Err(FromErr::Overflow);
        }

        let uval = *value.m.last().unwrap();
        let mut val: i64 = uval.try_into().map_err(|_| FromErr::Overflow)?;
        if value.s {
            val = val.neg();
        }
        return Ok(val);
    }
}

#[derive(Debug)]
pub enum FromStrErr {
    Overflow,
    Empty,
    Whitespace,
    MalformedExpression,
}

pub trait Abs {
    type I;
    type U;
    fn unsigned_abs(self) -> Self::U;
    fn abs(self) -> Self::I;
}

pub trait PowI<RHS = Self> {
    type Output;
    fn powi(&self, rhs: RHS) -> Self::Output;
}

pub trait MulVariants {
    fn full_mul(&self, rhs: &Self) -> Self;
    fn man_mul(&self, rhs: &Self, l: Option<usize>, r: Option<usize>, prec: Option<usize>) -> Self;
}

pub trait Sqr {
    fn sqr(&self) -> Self;
}

pub trait SqrVariants {
    fn full_sqr(&self) -> Self;
    fn man_sqr(&self, in_prec: Option<usize>, out_prec: Option<usize>) -> Self;
}

pub trait DivVariants<RHS = Self> {
    type Output;
    fn full_div(self, rhs: RHS) -> Self::Output;
    fn man_div(
        self,
        rhs: RHS,
        l: Option<usize>,
        r: Option<usize>,
        prec: Option<usize>,
    ) -> Self::Output;
}

pub trait DivRem<RHS = Self> {
    type Q: Sized;
    type R: Sized;
    fn div_rem(self, rhs: RHS) -> (Self::Q, Self::R);
}

pub trait LogI {
    type Output;

    fn ilog2pow64(&self) -> Self::Output;
    fn ilog2(&self) -> Self::Output;
    fn ilog(&self, b: u128) -> Self::Output;
}

pub trait Rounding {
    fn ceil(&self) -> Self;
    fn ceil_mut(&mut self);
    fn floor(&self) -> Self;
    fn floor_mut(&mut self);
    fn round(&self) -> Self;
    fn round_mut(&mut self);
    fn trunc(&self) -> Self;
    fn trunc_mut(&mut self);
    fn fract(&self) -> Self;
    fn fract_mut(&mut self);
}

#[macro_export]
macro_rules! impl_commutative_peq_pord{
    (const $G:ident, $big:ident, $($prim:ty),+) => {
        $(
            impl<const $G: usize> PartialEq<$big<$G>> for $prim {
                fn eq(&self, rhs: &$big<$G>) -> bool {
                    rhs.eq(self)
                }
            }

            impl<const $G: usize> PartialOrd<$big<$G>> for $prim {
                fn partial_cmp(&self, rhs: &$big<$G>) -> Option<std::cmp::Ordering> {
                    Some(rhs.partial_cmp(self).unwrap().reverse())
                }
            }
        )+
    };
    ($big:ty, $($prim:ty),+) => {
        $(
            impl PartialEq<$big> for $prim {
                fn eq(&self, rhs: &$big) -> bool {
                    rhs.eq(self)
                }
            }
            impl PartialOrd<$big> for $prim {
                fn partial_cmp(&self, rhs: &$big) -> Option<std::cmp::Ordering> {
                    Some(rhs.partial_cmp(self).unwrap().reverse())
                }
            }
        )+
    };
}

#[macro_export]
macro_rules! impl_commutative_div_rem{
    (const $N:ident, $big:ident, $($prim:ty),+) => {
        $(
            impl<const $N: usize> Div<$big<$N>> for $prim
            where
                $prim: DivRem<$big<$N>>
            {
                type Output = <$prim as DivRem<$big<$N>>>::Q;
                fn div(self, rhs: $big<$N>) -> Self::Output{
                    self.div_rem(rhs).0
                }
            }

            impl<const $N: usize> Rem<$big<$N>> for $prim
            where
                $prim: DivRem<$big<$N>>
            {
                type Output = <$prim as DivRem<$big<$N>>>::R;
                fn rem(self, rhs: $big<$N>) -> Self::Output{
                    self.div_rem(rhs).1
                }
            }
        )+
    };
    ($big:ty, $($prim:ty),+) => {
        $(
            impl Div<$big> for $prim
            where
                $prim: DivRem<$big>
            {
                type Output = <$prim as DivRem<$big>>::Q;
                fn div(self, rhs: $big) -> Self::Output{
                    self.div_rem(rhs).0
                }
            }

            impl<'a> Div<&'a $big> for $prim
            where
                $prim: DivRem<&'a $big>
            {
                type Output = <$prim as DivRem<&'a $big>>::Q;
                fn div(self, rhs: &'a $big) -> Self::Output{
                    self.div_rem(rhs).0
                }
            }

            impl Rem<$big> for $prim
            where
                $prim: DivRem<$big>
            {
                type Output = <$prim as DivRem<$big>>::R;
                fn rem(self, rhs: $big) -> Self::Output{
                    self.div_rem(rhs).1
                }
            }

            impl<'a> Rem<&'a $big> for $prim
            where
                $prim: DivRem<&'a $big>
            {
                type Output = <$prim as DivRem<&'a $big>>::R;
                fn rem(self, rhs: &'a $big) -> Self::Output{
                    self.div_rem(rhs).1
                }
            }
        )+
    };
}

#[macro_export]
macro_rules! impl_commutative {
    (const $G:ident, $Trait:ident, $method:ident, $big:ident, |$x:ident| $transform:expr, $($prim:ty),+) => {
        $(
            impl<const $G: usize> $Trait<$big<$G>> for $prim
            where
                $big<$G>: $Trait<$prim, Output = $big<$G>>,
            {
                type Output = $big<$G>;
                fn $method(self, rhs: $big<$G>) -> Self::Output {
                    let $x: $big<$G> = rhs.$method(self);
                    $transform
                }
            }
        )+
    };
    ($Trait:ident, $method:ident, $big:ty, |$x:ident| $transform:expr, $($prim:ty),+) => {
        $(
            impl $Trait<$big> for $prim{
                type Output = $big;
                fn $method(self, rhs: $big) -> Self::Output {
                    let $x = rhs.$method(self);
                    $transform
                }
            }
            impl $Trait<&$big> for $prim{
                type Output = $big;
                fn $method(self, rhs: &$big) -> Self::Output {
                    let $x = rhs.$method(self);
                    $transform
                }
            }
        )+
    };
}
