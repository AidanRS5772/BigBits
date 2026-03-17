use crate::traits::{
    DivRem, FromErr, FromStrErr, Int, LogI, PowI, Prim, SmallBuf, Sqr, UPrim, UnsignedAbs,
};
use crate::ubitint::UBitInt;
use crate::utils::*;
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
    pub fn abs(&mut self) {
        self.sign = false;
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data.is_empty()
    }
}

impl From<UBitInt> for BitInt {
    fn from(value: UBitInt) -> Self {
        BitInt {
            data: value.get_data().to_vec(),
            sign: false,
        }
    }
}

impl From<&UBitInt> for BitInt {
    fn from(value: &UBitInt) -> Self {
        BitInt {
            data: value.get_data().to_vec(),
            sign: false,
        }
    }
}

impl<T: Int> From<T> for BitInt {
    fn from(value: T) -> Self {
        BitInt {
            data: value.unsigned().into().to_vec(),
            sign: value.sign(),
        }
    }
}

impl TryFrom<BitInt> for u128 {
    type Error = FromErr;
    fn try_from(value: BitInt) -> Result<Self, Self::Error> {
        if value.sign {
            return Err(FromErr::Sign);
        }
        let buf = SmallBuf::try_from(value.data.as_slice())?;
        Ok(buf.into())
    }
}

impl TryFrom<&BitInt> for u128 {
    type Error = FromErr;
    fn try_from(value: &BitInt) -> Result<Self, Self::Error> {
        if value.sign {
            return Err(FromErr::Sign);
        }
        let buf = SmallBuf::try_from(value.data.as_slice())?;
        Ok(buf.into())
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

macro_rules! impl_try_from_bi {
    ($(($u:ty, $i:ty)), +) => {
        $(
        impl TryFrom<BitInt> for $u{
            type Error = FromErr;
            fn try_from(value: BitInt) -> Result<Self, Self::Error> {
                if value.sign {
                    return Err(FromErr::Sign);
                }
                SmallBuf::try_from(value.data.as_slice())?.try_into()
            }
        }

        impl TryFrom<&BitInt> for $u{
            type Error = FromErr;
            fn try_from(value: &BitInt) -> Result<Self, Self::Error> {
                if value.sign {
                    return Err(FromErr::Sign);
                }
                SmallBuf::try_from(value.data.as_slice())?.try_into()
            }
        }


        impl TryFrom<BitInt> for $i{
            type Error = FromErr;
            fn try_from(value: BitInt) -> Result<Self, Self::Error> {
                let uval: $u = SmallBuf::try_from(value.data.as_slice())?.try_into()?;
                let ival: $i = uval.try_into().map_err(|_| FromErr::Overflow)?;
                Ok(if value.sign { -ival } else { ival })
            }
        }

        impl TryFrom<&BitInt> for $i{
            type Error = FromErr;
            fn try_from(value: &BitInt) -> Result<Self, Self::Error> {
                let uval: $u = SmallBuf::try_from(value.data.as_slice())?.try_into()?;
                let ival: $i = uval.try_into().map_err(|_| FromErr::Overflow)?;
                Ok(if value.sign { -ival } else { ival })
            }
        }
        )+
    };
}
impl_try_from_bi!((u64, i64), (u32, i32), (u16, i16), (u8, i8));

impl UnsignedAbs for BitInt {
    type Output = UBitInt;
    fn unsigned_abs(self) -> Self::Output {
        UBitInt::make(self.data)
    }
}

impl UnsignedAbs for &BitInt {
    type Output = UBitInt;
    fn unsigned_abs(self) -> Self::Output {
        UBitInt::make(self.data.clone())
    }
}

impl FromStr for BitInt {
    type Err = FromStrErr;
    fn from_str(str: &str) -> Result<Self, FromStrErr> {
        const CHUNK_SIZE: usize = 19;
        const CHUNK_BASE: u64 = 10_000_000_000_000_000_000;

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
                .parse::<u64>()
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

impl PartialEq<UBitInt> for BitInt {
    fn eq(&self, other: &UBitInt) -> bool {
        if self.sign {
            return false;
        }
        self.data == other.get_data()
    }
}

impl<T: Int> PartialEq<T> for BitInt {
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

impl PartialOrd<UBitInt> for BitInt {
    fn partial_cmp(&self, other: &UBitInt) -> Option<std::cmp::Ordering> {
        if self.sign {
            return Some(scmp(self.sign, std::cmp::Ordering::Greater));
        }
        Some(scmp(self.sign, cmp_buf(&self.data, other.get_data())))
    }
}

impl<T: Int> PartialOrd<T> for BitInt {
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

impl_commutative_peq_pord!(BitInt, UBitInt, u128, i128, u64, i64, u32, i32, u16, i16, u8, i8);

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

impl Add<UBitInt> for BitInt {
    type Output = BitInt;
    fn add(self, rhs: UBitInt) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, &rhs.get_data(), false);
        return lhs;
    }
}

impl Add<UBitInt> for &BitInt {
    type Output = BitInt;
    fn add(self, rhs: UBitInt) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, &rhs.get_data(), false);
        return lhs;
    }
}

impl Add<&UBitInt> for BitInt {
    type Output = BitInt;
    fn add(self, rhs: &UBitInt) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, &rhs.get_data(), false);
        return lhs;
    }
}

impl Add<&UBitInt> for &BitInt {
    type Output = BitInt;
    fn add(self, rhs: &UBitInt) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, &rhs.get_data(), false);
        return lhs;
    }
}

impl<T: Int> Add<T> for BitInt {
    type Output = BitInt;
    fn add(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, &rhs.unsigned().into(), rhs.sign());
        return lhs;
    }
}

impl<T: Int> Add<T> for &BitInt {
    type Output = BitInt;
    fn add(self, rhs: T) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, &rhs.unsigned().into(), rhs.sign());
        return lhs;
    }
}

impl_commutative!(
    Add,
    add,
    BitInt,
    |x| x,
    UBitInt,
    &UBitInt,
    i128,
    u128,
    i64,
    u64,
    i32,
    u32,
    i16,
    u16,
    i8,
    u8
);

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

impl<T: Int> AddAssign<T> for BitInt {
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

impl Sub<UBitInt> for BitInt {
    type Output = BitInt;
    fn sub(self, rhs: UBitInt) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, rhs.get_data(), true);
        return lhs;
    }
}

impl Sub<&UBitInt> for &BitInt {
    type Output = BitInt;
    fn sub(self, rhs: &UBitInt) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, rhs.get_data(), true);
        return lhs;
    }
}

impl Sub<&UBitInt> for BitInt {
    type Output = BitInt;
    fn sub(self, rhs: &UBitInt) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, rhs.get_data(), true);
        return lhs;
    }
}

impl Sub<UBitInt> for &BitInt {
    type Output = BitInt;
    fn sub(self, rhs: UBitInt) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, rhs.get_data(), true);
        return lhs;
    }
}

impl<T: Int> Sub<T> for BitInt {
    type Output = BitInt;
    fn sub(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        add_sub(&mut lhs, &rhs.unsigned().into(), !rhs.sign());
        return lhs;
    }
}

impl<T: Int> Sub<T> for &BitInt {
    type Output = BitInt;
    fn sub(self, rhs: T) -> Self::Output {
        let mut lhs = self.clone();
        add_sub(&mut lhs, &rhs.unsigned().into(), !rhs.sign());
        return lhs;
    }
}

impl_commutative!(
    Sub,
    sub,
    BitInt,
    |x| -x,
    UBitInt,
    &UBitInt,
    i128,
    u128,
    i64,
    u64,
    i32,
    u32,
    i16,
    u16,
    i8,
    u8
);

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

impl SubAssign<UBitInt> for BitInt {
    fn sub_assign(&mut self, rhs: UBitInt) {
        add_sub(self, rhs.get_data(), true);
    }
}

impl SubAssign<&UBitInt> for BitInt {
    fn sub_assign(&mut self, rhs: &UBitInt) {
        add_sub(self, rhs.get_data(), true);
    }
}

impl<T: Int> SubAssign<T> for BitInt {
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

impl Mul<UBitInt> for BitInt {
    type Output = BitInt;
    fn mul(self, rhs: UBitInt) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, rhs.get_data());
        if c > 0 {
            data.push(c);
        }
        BitInt {
            data,
            sign: self.sign,
        }
    }
}

impl Mul<UBitInt> for &BitInt {
    type Output = BitInt;
    fn mul(self, rhs: UBitInt) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, rhs.get_data());
        if c > 0 {
            data.push(c);
        }
        BitInt {
            data,
            sign: self.sign,
        }
    }
}

impl Mul<&UBitInt> for BitInt {
    type Output = BitInt;
    fn mul(self, rhs: &UBitInt) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, rhs.get_data());
        if c > 0 {
            data.push(c);
        }
        BitInt {
            data,
            sign: self.sign,
        }
    }
}

impl Mul<&UBitInt> for &BitInt {
    type Output = BitInt;
    fn mul(self, rhs: &UBitInt) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, rhs.get_data());
        if c > 0 {
            data.push(c);
        }
        BitInt {
            data,
            sign: self.sign,
        }
    }
}

impl Mul<u128> for BitInt {
    type Output = BitInt;
    fn mul(self, rhs: u128) -> Self::Output {
        let mut bi = self;
        let c = mul_prim2(&mut bi.data, rhs);
        push_prim2(&mut bi.data, c);
        return bi;
    }
}

impl Mul<u128> for &BitInt {
    type Output = BitInt;
    fn mul(self, rhs: u128) -> Self::Output {
        let mut bi = self.clone();
        let c = mul_prim2(&mut bi.data, rhs);
        push_prim2(&mut bi.data, c);
        return bi;
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

impl<T: Prim> Mul<T> for BitInt {
    type Output = BitInt;
    fn mul(self, rhs: T) -> Self::Output {
        let mut bi = self;
        let c = mul_prim(&mut bi.data, rhs.unsigned().into());
        if c > 0 {
            bi.data.push(c);
        }
        bi.sign ^= rhs.sign();
        return bi;
    }
}

impl<T: Prim> Mul<T> for &BitInt {
    type Output = BitInt;
    fn mul(self, rhs: T) -> Self::Output {
        let mut bi = self.clone();
        let c = mul_prim(&mut bi.data, rhs.unsigned().into());
        if c > 0 {
            bi.data.push(c);
        }
        bi.sign ^= rhs.sign();
        return bi;
    }
}

impl_commutative!(
    Mul,
    mul,
    BitInt,
    |x| x,
    UBitInt,
    &UBitInt,
    u128,
    i128,
    u64,
    i64,
    u32,
    i32,
    u16,
    i16,
    u8,
    i8
);

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

impl<T: Prim> MulAssign<T> for BitInt {
    fn mul_assign(&mut self, rhs: T) {
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

impl<T: UPrim> Shl<T> for BitInt {
    type Output = BitInt;
    fn shl(self, rhs: T) -> Self::Output {
        let mut bi = self;
        shl_vec(&mut bi.data, rhs.into());
        return bi;
    }
}

impl<T: UPrim> Shl<T> for &BitInt {
    type Output = BitInt;
    fn shl(self, rhs: T) -> Self::Output {
        let mut bi = self.clone();
        shl_vec(&mut bi.data, rhs.into());
        return bi;
    }
}

impl<T: UPrim> ShlAssign<T> for BitInt {
    fn shl_assign(&mut self, rhs: T) {
        shl_vec(&mut self.data, rhs.into());
    }
}

impl<T: UPrim> Shr<T> for BitInt {
    type Output = BitInt;
    fn shr(self, rhs: T) -> Self::Output {
        let mut bi = self;
        shl_vec(&mut bi.data, rhs.into());
        return bi;
    }
}

impl<T: UPrim> Shr<T> for &BitInt {
    type Output = BitInt;
    fn shr(self, rhs: T) -> Self::Output {
        let mut bi = self.clone();
        shl_vec(&mut bi.data, rhs.into());
        return bi;
    }
}

impl<T: UPrim> ShrAssign<T> for BitInt {
    fn shr_assign(&mut self, rhs: T) {
        shl_vec(&mut self.data, rhs.into());
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

impl DivRem<UBitInt> for BitInt {
    type Q = BitInt;
    type R = BitInt;
    fn div_rem(self, rhs: UBitInt) -> (Self::Q, Self::R) {
        let mut d = rhs.get_data().to_vec();
        div_rem_bi(self, &mut d, false)
    }
}

impl DivRem<UBitInt> for &BitInt {
    type Q = BitInt;
    type R = BitInt;
    fn div_rem(self, rhs: UBitInt) -> (Self::Q, Self::R) {
        let mut d = rhs.get_data().to_vec();
        div_rem_bi(self.clone(), &mut d, false)
    }
}

impl DivRem<&UBitInt> for BitInt {
    type Q = BitInt;
    type R = BitInt;
    fn div_rem(self, rhs: &UBitInt) -> (Self::Q, Self::R) {
        let mut d = rhs.get_data().to_vec();
        div_rem_bi(self, &mut d, false)
    }
}

impl DivRem<&UBitInt> for &BitInt {
    type Q = BitInt;
    type R = BitInt;
    fn div_rem(self, rhs: &UBitInt) -> (Self::Q, Self::R) {
        let mut d = rhs.get_data().to_vec();
        div_rem_bi(self.clone(), &mut d, false)
    }
}

impl DivRem<u128> for BitInt {
    type Q = BitInt;
    type R = BitInt;
    fn div_rem(self, rhs: u128) -> (Self::Q, Self::R) {
        let mut d = SmallBuf::from(rhs);
        div_rem_bi(self, &mut d, false)
    }
}

impl DivRem<u128> for &BitInt {
    type Q = BitInt;
    type R = BitInt;
    fn div_rem(self, rhs: u128) -> (Self::Q, Self::R) {
        let mut d = SmallBuf::from(rhs);
        div_rem_bi(self.clone(), &mut d, false)
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

impl<T: Prim> DivRem<T> for BitInt
where
    <T as Prim>::BigI: TryFrom<u64>,
{
    type Q = BitInt;
    type R = <T as Prim>::MedI;
    fn div_rem(self, rhs: T) -> (Self::Q, Self::R) {
        let mut lhs = self;
        let r = div_prim(&mut lhs.data, rhs.unsigned().into());
        let mut rem = <T as Prim>::MedI::try_from(r).ok().unwrap();
        lhs.sign ^= rhs.sign();
        if rhs.sign() {
            rem = rem.neg();
        }
        (lhs, rem)
    }
}

impl DivRem<BitInt> for u128 {
    type Q = BitInt;
    type R = u128;
    fn div_rem(self, rhs: BitInt) -> (Self::Q, Self::R) {
        let Ok(rem) = u128::try_from(&rhs) else {
            return (BitInt::zero(), self);
        };
        let div = BitInt::from(self);
        (div / rhs, self % rem)
    }
}

impl DivRem<&BitInt> for u128 {
    type Q = BitInt;
    type R = u128;
    fn div_rem(self, rhs: &BitInt) -> (Self::Q, Self::R) {
        let Ok(rem) = u128::try_from(rhs) else {
            return (BitInt::zero(), self);
        };
        let div = BitInt::from(self);
        (div / rhs, self % rem)
    }
}

impl DivRem<BitInt> for i128 {
    type Q = BitInt;
    type R = i128;
    fn div_rem(self, rhs: BitInt) -> (Self::Q, Self::R) {
        let Ok(rem) = i128::try_from(&rhs) else {
            return (BitInt::zero(), self);
        };
        let div = BitInt::from(self);
        (div / rhs, self % rem)
    }
}

impl DivRem<&BitInt> for i128 {
    type Q = BitInt;
    type R = i128;
    fn div_rem(self, rhs: &BitInt) -> (Self::Q, Self::R) {
        let Ok(rem) = i128::try_from(rhs) else {
            return (BitInt::zero(), self);
        };
        let div = BitInt::from(self);
        (div / rhs, self % rem)
    }
}

impl<T: Prim> DivRem<BitInt> for T
where
    T: TryFrom<BitInt> + Rem<T, Output = T> + for<'a> TryFrom<&'a BitInt>,
    <T as Prim>::BigI:
        Div<<T as Prim>::BigI, Output = <T as Prim>::BigI> + for<'a> TryFrom<&'a BitInt>,
{
    type Q = <T as Prim>::BigI;
    type R = T;
    fn div_rem(self, rhs: BitInt) -> (Self::Q, Self::R) {
        let Ok(rem) = T::try_from(&rhs) else {
            return (<T as Prim>::BigI::default(), self);
        };
        let Ok(div) = <T as Prim>::BigI::try_from(&rhs) else {
            return (<T as Prim>::BigI::default(), self);
        };
        (self.into() / div, self % rem)
    }
}

impl<T: Prim> DivRem<&BitInt> for T
where
    T: TryFrom<BitInt> + Rem<T, Output = T> + for<'a> TryFrom<&'a BitInt>,
    <T as Prim>::BigI:
        Div<<T as Prim>::BigI, Output = <T as Prim>::BigI> + for<'a> TryFrom<&'a BitInt>,
{
    type Q = <T as Prim>::BigI;
    type R = T;
    fn div_rem(self, rhs: &BitInt) -> (Self::Q, Self::R) {
        let Ok(rem) = T::try_from(rhs) else {
            return (<T as Prim>::BigI::default(), self);
        };
        let Ok(div) = <T as Prim>::BigI::try_from(rhs) else {
            return (<T as Prim>::BigI::default(), self);
        };
        (self.into() / div, self % rem)
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

impl_commutative_div_rem!(BitInt, u128, i128, u64, i64, u32, i32, u16, i16, u8, i8);

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

impl<T: UPrim> PowI<T> for BitInt {
    type Output = BitInt;
    fn powi(&self, rhs: T) -> Self::Output {
        BitInt {
            data: powi_vec(&self.data, rhs.into()),
            sign: self.sign ^ ((rhs.into() & 1) == 1),
        }
    }
}

impl LogI for BitInt {
    type Output = u64;

    fn ilog2pow64(&self) -> Self::Output {
        debug_assert!(
            !self.is_zero(),
            "attempt to calculate the logarithm of zero"
        );
        debug_assert!(!self.sign, "argument of integer logarithm must be positive");

        return (self.data.len() - 1) as u64;
    }
    fn ilog2(&self) -> Self::Output {
        let l = self.ilog2pow64();
        return self.data[l as usize].ilog2() as u64 + 64 * l;
    }

    fn ilog(&self, b: u128) -> Self::Output {
        let mut log = self.ilog2() / (b.ilog2() as u64 + 1);
        let mut b_ubi = UBitInt::from(b).powi(log + 1);

        while self >= &b_ubi {
            log += 1;
            b_ubi *= b;
        }

        return log;
    }
}
