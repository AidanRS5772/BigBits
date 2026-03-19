use super::traits::{Abs, DivRem, FromErr, FromStrErr, LogI, PowI, SmallBuf, Sqr, I};
use crate::bit_nums::ubitint::UBitInt;
use crate::utils::{div::*, mul::*, utils::*};
use crate::{impl_commutative, impl_commutative_div_rem, impl_commutative_peq_pord};
use core::fmt;
use std::ops::*;
use std::str::FromStr;

#[derive(Debug, Clone, Default)]
pub struct BitInt {
    data: Vec<u64>,
    sign: bool,
}

impl BitInt {
    #[inline]
    pub fn get_data(&self) -> &[u64] {
        return &self.data;
    }

    #[inline]
    pub fn get_sign(&self) -> bool {
        return self.sign;
    }

    pub fn make(data: Vec<u64>, sign: bool) -> Self {
        BitInt { data, sign }
    }

    pub fn zero() -> BitInt {
        BitInt {
            data: vec![],
            sign: false,
        }
    }

    pub fn one() -> BitInt {
        BitInt {
            data: vec![1],
            sign: false,
        }
    }

    #[inline]
    pub fn neg_mut(&mut self) {
        self.sign = !self.sign;
    }

    #[inline]
    pub fn abs_mut(&mut self) {
        self.sign = false;
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: I> From<T> for BitInt {
    fn from(value: T) -> Self {
        BitInt {
            data: value.unsigned().into().to_vec(),
            sign: value.sign(),
        }
    }
}

impl TryFrom<BitInt> for i128 {
    type Error = FromErr;
    fn try_from(value: BitInt) -> Result<Self, Self::Error> {
        let uval: u128 = SmallBuf::try_from(value.data.as_slice())?.into();
        let ival: i128 = uval.try_into().map_err(|_| FromErr::Overflow)?;
        Ok(if value.sign { -ival } else { ival })
    }
}

impl TryFrom<&BitInt> for i128 {
    type Error = FromErr;
    fn try_from(value: &BitInt) -> Result<Self, Self::Error> {
        let uval: u128 = SmallBuf::try_from(value.data.as_slice())?.into();
        let ival: i128 = uval.try_into().map_err(|_| FromErr::Overflow)?;
        Ok(if value.sign { -ival } else { ival })
    }
}

impl TryFrom<BitInt> for i64 {
    type Error = FromErr;
    fn try_from(value: BitInt) -> Result<Self, Self::Error> {
        let uval: u64 = SmallBuf::try_from(value.data.as_slice())?
            .try_into()
            .map_err(|_| FromErr::Overflow)?;
        let ival: i64 = uval.try_into().map_err(|_| FromErr::Overflow)?;
        Ok(if value.sign { -ival } else { ival })
    }
}

impl TryFrom<&BitInt> for i64 {
    type Error = FromErr;
    fn try_from(value: &BitInt) -> Result<Self, Self::Error> {
        let uval: u64 = SmallBuf::try_from(value.data.as_slice())?
            .try_into()
            .map_err(|_| FromErr::Overflow)?;
        let ival: i64 = uval.try_into().map_err(|_| FromErr::Overflow)?;
        Ok(if value.sign { -ival } else { ival })
    }
}

impl Abs for BitInt {
    type U = UBitInt;
    type I = BitInt;
    fn unsigned_abs(self) -> Self::U {
        UBitInt::make(self.data)
    }
    fn abs(self) -> Self::I {
        BitInt {
            data: self.data,
            sign: false,
        }
    }
}

impl Abs for &BitInt {
    type U = UBitInt;
    type I = BitInt;
    fn unsigned_abs(self) -> Self::U {
        UBitInt::make(self.data.clone())
    }
    fn abs(self) -> Self::I {
        BitInt {
            data: self.data.clone(),
            sign: false,
        }
    }
}

impl FromStr for BitInt {
    type Err = FromStrErr;
    fn from_str(str: &str) -> Result<Self, FromStrErr> {
        const CHUNK_SIZE: usize = 18;
        const CHUNK_BASE: i64 = 1_000_000_000_000_000_000;

        let mut result = BitInt::zero();
        let mut multiplier = BitInt::one();

        let neg = if let Some(last) = str.chars().next() {
            if last == '-' {
                true
            } else {
                false
            }
        } else {
            return Err(FromStrErr::Empty);
        };

        let mut end = str.len();
        while end > 0 {
            let mut start = end.saturating_sub(CHUNK_SIZE);
            if neg && start == 0 {
                start += 1;
            }
            let chunk_str = &str[start..end];

            if chunk_str.trim() != chunk_str {
                return Err(FromStrErr::Whitespace);
            }
            let chunk_value = chunk_str
                .parse::<i64>()
                .map_err(|_| FromStrErr::MalformedExpression)?;

            result += chunk_value * &multiplier;
            multiplier *= CHUNK_BASE;
            end = start;
        }

        if neg {
            result.sign = true;
        }

        Ok(result)
    }
}

impl fmt::Display for BitInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        const CHUNK_DIVISOR: u64 = 10_000_000_000_000_000_000;
        const CHUNK_DIGITS: usize = 19;

        let mut chunks: Vec<u64> = Vec::new();
        let mut ubi = self.unsigned_abs();

        while !ubi.is_zero() {
            let (div, rem) = ubi.div_rem(CHUNK_DIVISOR);
            chunks.push(rem);
            ubi = div;
        }

        if self.sign {
            write!(f, "-")?;
        }

        if let Some(&first) = chunks.last() {
            write!(f, "{}", first)?;
        }

        for &chunk in chunks.iter().rev().skip(1) {
            write!(f, "{:0width$}", chunk, width = CHUNK_DIGITS)?;
        }

        Ok(())
    }
}

impl PartialEq for BitInt {
    fn eq(&self, other: &Self) -> bool {
        if self.sign ^ other.sign {
            return false;
        }
        self.data == other.data
    }
}

impl<T: I> PartialEq<T> for BitInt {
    fn eq(&self, other: &T) -> bool {
        if self.sign ^ other.sign() {
            return false;
        }
        self.data[..] == other.unsigned().into()[..]
    }
}

impl PartialOrd for BitInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.sign ^ other.sign {
            return Some(scmp(self.sign, std::cmp::Ordering::Greater));
        }
        Some(scmp(self.sign, cmp_buf(&self.data, &other.data)))
    }
}

impl<T: I> PartialOrd<T> for BitInt {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        if self.sign ^ other.sign() {
            return Some(scmp(self.sign, std::cmp::Ordering::Greater));
        }
        Some(scmp(
            self.sign,
            cmp_buf(&self.data, &other.unsigned().into()),
        ))
    }
}

impl_commutative_peq_pord!(BitInt, i128, i64);

impl Eq for BitInt {}

impl Ord for BitInt {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

fn add_sub(lhs: &mut BitInt, rhs: &[u64], sub: bool) {
    let comp = lhs.sign ^ sub;
    if lhs.data.len() < rhs.len() {
        lhs.data.resize(rhs.len(), 0);
    }

    if acc(&mut lhs.data, &rhs, comp as u8) {
        if comp {
            twos_comp(&mut lhs.data);
        } else {
            lhs.data.push(1);
        }
        lhs.sign ^= comp;
    }
    if comp {
        trim_lz(&mut lhs.data);
    }
}

impl Add for BitInt {
    type Output = BitInt;
    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, &rhs.data, rhs.sign);
        return lhs;
    }
}

impl Add for &BitInt {
    type Output = BitInt;
    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, &rhs.data, rhs.sign);
        return lhs;
    }
}

impl Add<&BitInt> for BitInt {
    type Output = BitInt;
    fn add(self, rhs: &BitInt) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, &rhs.data, rhs.sign);
        return lhs;
    }
}

impl Add<BitInt> for &BitInt {
    type Output = BitInt;
    fn add(self, rhs: BitInt) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, &rhs.data, rhs.sign);
        return lhs;
    }
}

impl<T: I> Add<T> for BitInt {
    type Output = BitInt;
    fn add(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, &rhs.unsigned().into(), rhs.sign());
        return lhs;
    }
}

impl<T: I> Add<T> for &BitInt {
    type Output = BitInt;
    fn add(self, rhs: T) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, &rhs.unsigned().into(), rhs.sign());
        return lhs;
    }
}

impl_commutative!(Add, add, BitInt, |x| x, i128, i64);

impl AddAssign for BitInt {
    fn add_assign(&mut self, rhs: Self) {
        add_sub(self, &rhs.data, rhs.sign);
    }
}

impl AddAssign<&BitInt> for BitInt {
    fn add_assign(&mut self, rhs: &BitInt) {
        add_sub(self, &rhs.data, rhs.sign);
    }
}

impl AddAssign<UBitInt> for BitInt {
    fn add_assign(&mut self, rhs: UBitInt) {
        add_sub(self, &rhs.get_data(), false);
    }
}

impl AddAssign<&UBitInt> for BitInt {
    fn add_assign(&mut self, rhs: &UBitInt) {
        add_sub(self, &rhs.get_data(), false);
    }
}

impl<T: I> AddAssign<T> for BitInt {
    fn add_assign(&mut self, rhs: T) {
        add_sub(self, &rhs.unsigned().into(), rhs.sign());
    }
}

impl Neg for BitInt {
    type Output = BitInt;

    fn neg(self) -> Self::Output {
        BitInt {
            data: self.data,
            sign: !self.sign,
        }
    }
}

impl Neg for &BitInt {
    type Output = BitInt;

    fn neg(self) -> Self::Output {
        BitInt {
            data: self.data.clone(),
            sign: !self.sign,
        }
    }
}

impl Sub for BitInt {
    type Output = BitInt;
    fn sub(self, rhs: BitInt) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, &rhs.data, !rhs.sign);
        return lhs;
    }
}

impl Sub for &BitInt {
    type Output = BitInt;
    fn sub(self, rhs: &BitInt) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, &rhs.data, !rhs.sign);
        return lhs;
    }
}

impl Sub<&BitInt> for BitInt {
    type Output = BitInt;
    fn sub(self, rhs: &BitInt) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, &rhs.data, !rhs.sign);
        return lhs;
    }
}

impl Sub<BitInt> for &BitInt {
    type Output = BitInt;
    fn sub(self, rhs: BitInt) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, &rhs.data, !rhs.sign);
        return lhs;
    }
}

impl<T: I> Sub<T> for BitInt {
    type Output = BitInt;
    fn sub(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, &rhs.unsigned().into(), !rhs.sign());
        return lhs;
    }
}

impl<T: I> Sub<T> for &BitInt {
    type Output = BitInt;
    fn sub(self, rhs: T) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, &rhs.unsigned().into(), !rhs.sign());
        return lhs;
    }
}

impl_commutative!(Sub, sub, BitInt, |x| -x, i128, i64);

impl SubAssign for BitInt {
    fn sub_assign(&mut self, rhs: Self) {
        add_sub(self, &rhs.data, !rhs.sign);
    }
}

impl SubAssign<&BitInt> for BitInt {
    fn sub_assign(&mut self, rhs: &BitInt) {
        add_sub(self, &rhs.data, !rhs.sign);
    }
}

impl<T: I> SubAssign<T> for BitInt {
    fn sub_assign(&mut self, rhs: T) {
        add_sub(self, &rhs.unsigned().into(), !rhs.sign());
    }
}

impl Mul for BitInt {
    type Output = BitInt;
    fn mul(self, rhs: Self) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, &rhs.data);
        if c > 0 {
            data.push(c);
        }
        BitInt {
            data,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul for &BitInt {
    type Output = BitInt;
    fn mul(self, rhs: Self) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, &rhs.data);
        if c > 0 {
            data.push(c);
        }
        BitInt {
            data,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<&BitInt> for BitInt {
    type Output = BitInt;
    fn mul(self, rhs: &BitInt) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, &rhs.data);
        if c > 0 {
            data.push(c);
        }
        BitInt {
            data,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<BitInt> for &BitInt {
    type Output = BitInt;
    fn mul(self, rhs: BitInt) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, &rhs.data);
        if c > 0 {
            data.push(c);
        }
        BitInt {
            data,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<i128> for BitInt {
    type Output = BitInt;
    fn mul(self, rhs: i128) -> Self::Output {
        let mut bi = self;
        let c = mul_prim2(&mut bi.data, rhs.unsigned_abs());
        push_prim2(&mut bi.data, c);
        bi.sign ^= rhs < 0;
        return bi;
    }
}

impl Mul<i128> for &BitInt {
    type Output = BitInt;
    fn mul(self, rhs: i128) -> Self::Output {
        let mut bi = self.clone();
        let c = mul_prim2(&mut bi.data, rhs.unsigned_abs());
        push_prim2(&mut bi.data, c);
        bi.sign ^= rhs < 0;
        return bi;
    }
}

impl Mul<i64> for BitInt {
    type Output = BitInt;
    fn mul(self, rhs: i64) -> Self::Output {
        let mut bi = self;
        let c = mul_prim(&mut bi.data, rhs.unsigned_abs().into());
        if c > 0 {
            bi.data.push(c);
        }
        bi.sign ^= rhs.sign();
        return bi;
    }
}

impl Mul<i64> for &BitInt {
    type Output = BitInt;
    fn mul(self, rhs: i64) -> Self::Output {
        let mut bi = self.clone();
        let c = mul_prim(&mut bi.data, rhs.unsigned().into());
        if c > 0 {
            bi.data.push(c);
        }
        bi.sign ^= rhs.sign();
        return bi;
    }
}

impl_commutative!(Mul, mul, BitInt, |x| x, i128, i64);

impl MulAssign for BitInt {
    fn mul_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self).mul(rhs);
    }
}

impl MulAssign<&BitInt> for BitInt {
    fn mul_assign(&mut self, rhs: &BitInt) {
        *self = std::mem::take(self).mul(rhs);
    }
}

impl MulAssign<i128> for BitInt {
    fn mul_assign(&mut self, rhs: i128) {
        let c = mul_prim2(&mut self.data, rhs.unsigned_abs());
        if c > 0 {
            push_prim2(&mut self.data, c);
        }
        self.sign ^= rhs < 0;
    }
}

impl MulAssign<u128> for BitInt {
    fn mul_assign(&mut self, rhs: u128) {
        let c = mul_prim2(&mut self.data, rhs);
        if c > 0 {
            push_prim2(&mut self.data, c);
        }
    }
}

impl MulAssign<i64> for BitInt {
    fn mul_assign(&mut self, rhs: i64) {
        let c = mul_prim(&mut self.data, rhs.unsigned().into());
        if c > 0 {
            self.data.push(c);
        }
    }
}

impl Sqr for BitInt {
    fn sqr(&self) -> Self {
        let (mut data, c) = sqr_vec(&self.data);
        if c > 0 {
            data.push(c);
        }
        BitInt { data, sign: false }
    }
}

impl Shl<usize> for BitInt {
    type Output = BitInt;
    fn shl(self, rhs: usize) -> Self::Output {
        let mut lhs = self;
        let div = rhs / 64;
        lhs.data.resize(lhs.data.len() + div, 0);
        lhs.data.rotate_right(div);
        let c = shl_buf(&mut lhs.data, (rhs % 64) as u8);
        if c > 0 {
            lhs.data.push(c);
        }
        return lhs;
    }
}

impl Shl<usize> for &BitInt {
    type Output = BitInt;
    fn shl(self, rhs: usize) -> Self::Output {
        let mut lhs = self.clone();
        let div = rhs / 64;
        lhs.data.resize(lhs.data.len() + div, 0);
        lhs.data.rotate_right(div);
        let c = shl_buf(&mut lhs.data, (rhs % 64) as u8);
        if c > 0 {
            lhs.data.push(c);
        }
        return lhs;
    }
}

impl ShlAssign<usize> for BitInt {
    fn shl_assign(&mut self, rhs: usize) {
        let div = rhs / 64;
        self.data.resize(self.data.len() + div, 0);
        self.data.rotate_right(div);
        let c = shl_buf(&mut self.data, (rhs % 64) as u8);
        if c > 0 {
            self.data.push(c);
        }
    }
}

impl Shr<usize> for BitInt {
    type Output = BitInt;
    fn shr(self, rhs: usize) -> Self::Output {
        let mut lhs = self;
        let div = rhs / 64;
        lhs.data.drain(..div);
        shr_buf(&mut lhs.data, (rhs % 64) as u8);
        return lhs;
    }
}

impl Shr<usize> for &BitInt {
    type Output = BitInt;
    fn shr(self, rhs: usize) -> Self::Output {
        let mut lhs = self.clone();
        let div = rhs / 64;
        lhs.data.drain(..div);
        shr_buf(&mut lhs.data, (rhs % 64) as u8);
        return lhs;
    }
}

impl ShrAssign<usize> for BitInt {
    fn shr_assign(&mut self, rhs: usize) {
        let div = rhs / 64;
        self.data.drain(..div);
        shr_buf(&mut self.data, (rhs % 64) as u8);
    }
}

fn div_rem_bi(mut n: BitInt, d: &mut [u64], d_sign: bool) -> (BitInt, BitInt) {
    let mut q = div_vec(&mut n.data, d);
    trim_lz(&mut q);
    trim_lz(&mut n.data);
    (
        BitInt {
            data: q,
            sign: n.sign ^ d_sign,
        },
        n,
    )
}

impl DivRem for BitInt {
    type Q = BitInt;
    type R = BitInt;
    fn div_rem(self, rhs: Self) -> (Self::Q, Self::R) {
        let mut d = rhs.data;
        div_rem_bi(self, &mut d, rhs.sign)
    }
}

impl DivRem for &BitInt {
    type Q = BitInt;
    type R = BitInt;
    fn div_rem(self, rhs: Self) -> (Self::Q, Self::R) {
        let mut d = rhs.data.clone();
        div_rem_bi(self.clone(), &mut d, rhs.sign)
    }
}

impl DivRem<&BitInt> for BitInt {
    type Q = BitInt;
    type R = BitInt;
    fn div_rem(self, rhs: &BitInt) -> (Self::Q, Self::R) {
        let mut d = rhs.data.clone();
        div_rem_bi(self, &mut d, rhs.sign)
    }
}

impl DivRem<BitInt> for &BitInt {
    type Q = BitInt;
    type R = BitInt;
    fn div_rem(self, rhs: BitInt) -> (Self::Q, Self::R) {
        let mut d = rhs.data.clone();
        div_rem_bi(self.clone(), &mut d, rhs.sign)
    }
}

impl DivRem<i128> for BitInt {
    type Q = BitInt;
    type R = i128;
    fn div_rem(self, rhs: i128) -> (Self::Q, Self::R) {
        let mut d = SmallBuf::from(rhs.unsigned_abs());
        let (q, r) = div_rem_bi(self, &mut d, rhs < 0);
        (q, r.try_into().unwrap())
    }
}

impl DivRem<i128> for &BitInt {
    type Q = BitInt;
    type R = i128;
    fn div_rem(self, rhs: i128) -> (Self::Q, Self::R) {
        let mut d = SmallBuf::from(rhs.unsigned_abs());
        let (q, r) = div_rem_bi(self.clone(), &mut d, rhs < 0);
        (q, r.try_into().unwrap())
    }
}

impl DivRem<i64> for BitInt {
    type Q = BitInt;
    type R = i64;
    fn div_rem(self, rhs: i64) -> (Self::Q, Self::R) {
        let mut lhs = self;
        let r = div_prim(&mut lhs.data, rhs.unsigned_abs());
        lhs.sign ^= rhs.is_negative();
        (lhs, r.try_into().unwrap())
    }
}

impl DivRem<i64> for &BitInt {
    type Q = BitInt;
    type R = i64;
    fn div_rem(self, rhs: i64) -> (Self::Q, Self::R) {
        let mut lhs = self.clone();
        let r = div_prim(&mut lhs.data, rhs.unsigned_abs());
        lhs.sign ^= rhs.is_negative();
        (lhs, r.try_into().unwrap())
    }
}

impl<T: I> DivRem<BitInt> for T
where
    T: for<'a> TryFrom<&'a BitInt>,
{
    type Q = T;
    type R = T;
    fn div_rem(self, rhs: BitInt) -> (Self::Q, Self::R) {
        let Ok(prim) = T::try_from(&rhs) else {
            return (T::default(), self);
        };
        (self / prim, self % prim)
    }
}

impl<T: I> DivRem<&BitInt> for T
where
    T: for<'a> TryFrom<&'a BitInt>,
{
    type Q = T;
    type R = T;
    fn div_rem(self, rhs: &BitInt) -> (Self::Q, Self::R) {
        let Ok(prim) = T::try_from(rhs) else {
            return (T::default(), self);
        };
        (self / prim, self % prim)
    }
}

impl<T> Div<T> for BitInt
where
    BitInt: DivRem<T>,
{
    type Output = <BitInt as DivRem<T>>::Q;
    fn div(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).0
    }
}

impl<'a, T> Div<T> for &'a BitInt
where
    &'a BitInt: DivRem<T>,
{
    type Output = <&'a BitInt as DivRem<T>>::Q;
    fn div(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).0
    }
}

impl<T> Rem<T> for BitInt
where
    BitInt: DivRem<T>,
{
    type Output = <BitInt as DivRem<T>>::R;
    fn rem(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).1
    }
}

impl<'a, T> Rem<T> for &'a BitInt
where
    &'a BitInt: DivRem<T>,
{
    type Output = <&'a BitInt as DivRem<T>>::Q;
    fn rem(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).0
    }
}

impl_commutative_div_rem!(BitInt, i128, i64);

impl<T> DivAssign<T> for BitInt
where
    BitInt: DivRem<T, Q = BitInt>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = std::mem::take(self).div_rem(rhs).0;
    }
}

impl<T> RemAssign<T> for BitInt
where
    BitInt: DivRem<T, R = BitInt>,
{
    fn rem_assign(&mut self, rhs: T) {
        *self = std::mem::take(self).div_rem(rhs).1;
    }
}

impl PowI<usize> for BitInt {
    type Output = BitInt;
    fn powi(&self, rhs: usize) -> Self::Output {
        BitInt {
            data: powi_vec(&self.data, rhs),
            sign: self.sign ^ (rhs % 2 == 1),
        }
    }
}

impl LogI for BitInt {
    type Output = usize;

    fn ilog2pow64(&self) -> Self::Output {
        debug_assert!(
            !self.is_zero(),
            "attempt to calculate the logarithm of zero"
        );
        debug_assert!(!self.sign, "argument of integer logarithm must be positive");

        return self.data.len() - 1;
    }

    fn ilog2(&self) -> Self::Output {
        let l = self.ilog2pow64();
        return self.data[l].ilog2() as usize + 64 * l;
    }

    fn ilog(&self, b: u128) -> Self::Output {
        let mut log = self.ilog2() / (b.ilog2() as usize + 1);
        let u_arg = self.unsigned_abs();
        let mut b_ubi = UBitInt::from(b).powi(log + 1);

        while u_arg >= b_ubi {
            log += 1;
            b_ubi *= b;
        }

        return log;
    }
}
