use crate::traits::{DivRem, FromErr, FromStrErr, LogI, PowI, SmallBuf, Sqr, U};
use crate::{impl_commutative, impl_commutative_div_rem, impl_commutative_peq_pord, utils::*};
use core::fmt;
use std::ops::*;
use std::str::FromStr;

#[derive(Debug, Clone, Default)]
pub struct UBitInt {
    data: Vec<u64>,
}

impl UBitInt {
    pub fn get_data(&self) -> &[u64] {
        &self.data
    }

    pub fn make(data: Vec<u64>) -> UBitInt {
        UBitInt { data }
    }

    pub fn zero() -> UBitInt {
        UBitInt { data: vec![] }
    }

    pub fn one() -> UBitInt {
        UBitInt { data: vec![1] }
    }

    pub fn is_zero(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: Into<SmallBuf>> From<T> for UBitInt {
    fn from(value: T) -> Self {
        UBitInt {
            data: value.into().to_vec(),
        }
    }
}

impl TryFrom<UBitInt> for u128 {
    type Error = FromErr;
    fn try_from(value: UBitInt) -> Result<Self, Self::Error> {
        Ok(SmallBuf::try_from(value.data.as_slice())?.into())
    }
}

impl TryFrom<&UBitInt> for u128 {
    type Error = FromErr;
    fn try_from(value: &UBitInt) -> Result<Self, Self::Error> {
        Ok(SmallBuf::try_from(value.data.as_slice())?.into())
    }
}

impl TryFrom<UBitInt> for u64 {
    type Error = FromErr;
    fn try_from(value: UBitInt) -> Result<Self, Self::Error> {
        SmallBuf::try_from(value.data.as_slice())?.try_into()
    }
}

impl TryFrom<&UBitInt> for u64 {
    type Error = FromErr;
    fn try_from(value: &UBitInt) -> Result<Self, Self::Error> {
        SmallBuf::try_from(value.data.as_slice())?.try_into()
    }
}

impl FromStr for UBitInt {
    type Err = FromStrErr;
    fn from_str(str: &str) -> Result<Self, FromStrErr> {
        const CHUNK_SIZE: usize = 19;
        const CHUNK_BASE: u64 = 10_000_000_000_000_000_000;

        let mut result = UBitInt::zero();
        let mut multiplier = UBitInt::one();

        let mut end = str.len();
        if end == 0 {
            return Err(FromStrErr::Empty);
        }

        while end > 0 {
            let start = end.saturating_sub(CHUNK_SIZE);
            let chunk_str = &str[start..end];

            if chunk_str.trim() != chunk_str {
                return Err(FromStrErr::Whitespace);
            }
            if !chunk_str.chars().all(|c| c.is_ascii_digit()) {
                return Err(FromStrErr::MalformedExpression);
            }

            let chunk_value = chunk_str
                .parse::<u64>()
                .map_err(|_| FromStrErr::MalformedExpression)?;

            result += chunk_value * &multiplier;
            multiplier *= CHUNK_BASE;
            end = start;
        }

        Ok(result)
    }
}

impl fmt::Display for UBitInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        const CHUNK_DIVISOR: u64 = 10_000_000_000_000_000_000;
        const CHUNK_DIGITS: usize = 19;

        let mut chunks: Vec<u64> = Vec::new();
        let mut ubi = self.clone();

        while !ubi.is_zero() {
            let (div, rem) = ubi.div_rem(CHUNK_DIVISOR);
            chunks.push(rem);
            ubi = div;
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

impl PartialEq for UBitInt {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: U> PartialEq<T> for UBitInt {
    fn eq(&self, other: &T) -> bool {
        self.data[..] == (*other).into()[..]
    }
}

impl PartialOrd for UBitInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(cmp_buf(&self.data, &other.data))
    }
}

impl<T: U> PartialOrd<T> for UBitInt {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        Some(cmp_buf(&self.data, &(*other).into()))
    }
}

impl_commutative_peq_pord!(UBitInt, u128, u64);

impl Eq for UBitInt {}

impl Ord for UBitInt {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Add for UBitInt {
    type Output = UBitInt;
    fn add(self, rhs: Self) -> Self::Output {
        let (mut long, short) = if self.data.len() > rhs.data.len() {
            (self.data, &rhs.data)
        } else {
            (rhs.data, &self.data)
        };

        if acc(&mut long, short, 0) {
            long.push(1);
        }
        UBitInt { data: long }
    }
}

impl Add for &UBitInt {
    type Output = UBitInt;
    fn add(self, rhs: Self) -> Self::Output {
        let (mut longer, shorter) = if self.data.len() > rhs.data.len() {
            (self.data.clone(), &rhs.data)
        } else {
            (rhs.data.clone(), &self.data)
        };

        if acc(&mut longer, shorter, 0) {
            longer.push(1);
        }
        UBitInt { data: longer }
    }
}

impl Add<&UBitInt> for UBitInt {
    type Output = UBitInt;
    fn add(self, rhs: &UBitInt) -> Self::Output {
        let (mut longer, shorter) = if self.data.len() > rhs.data.len() {
            (self.data, &rhs.data)
        } else {
            (rhs.data.clone(), &self.data)
        };

        if acc(&mut longer, shorter, 0) {
            longer.push(1);
        }
        UBitInt { data: longer }
    }
}

impl Add<UBitInt> for &UBitInt {
    type Output = UBitInt;
    fn add(self, rhs: UBitInt) -> Self::Output {
        let (mut longer, shorter) = if self.data.len() > rhs.data.len() {
            (self.data.clone(), &rhs.data)
        } else {
            (rhs.data, &self.data)
        };

        if acc(&mut longer, &shorter, 0) {
            longer.push(1);
        }
        UBitInt { data: longer }
    }
}

impl<T: Into<SmallBuf>> Add<T> for UBitInt {
    type Output = UBitInt;
    fn add(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        let sml_buf = rhs.into();
        if lhs.data.len() < sml_buf.len() {
            lhs.data.resize(sml_buf.len(), 0);
        }
        if acc(&mut lhs.data, &sml_buf, 0) {
            lhs.data.push(1);
        }
        return lhs;
    }
}

impl<T: Into<SmallBuf>> Add<T> for &UBitInt {
    type Output = UBitInt;
    fn add(self, rhs: T) -> Self::Output {
        if self.is_zero() {
            return UBitInt::from(rhs);
        }
        let mut lhs = self.clone();
        let sml_buf = rhs.into();
        if lhs.data.len() < sml_buf.len() {
            lhs.data.resize(sml_buf.len(), 0);
        }
        if acc(&mut lhs.data, &sml_buf, 0) {
            lhs.data.push(1);
        }
        return lhs;
    }
}

impl_commutative!(Add, add, UBitInt, |x| x, u128, u64);

fn add_assign_ubi(lhs: &mut UBitInt, rhs: &[u64]) {
    if lhs.data.len() < rhs.len() {
        lhs.data.resize(rhs.len(), 0);
    }
    if acc(&mut lhs.data, rhs, 0) {
        lhs.data.push(1);
    }
}

impl AddAssign for UBitInt {
    fn add_assign(&mut self, rhs: Self) {
        add_assign_ubi(self, &rhs.data);
    }
}

impl AddAssign<&UBitInt> for UBitInt {
    fn add_assign(&mut self, rhs: &UBitInt) {
        add_assign_ubi(self, &rhs.data);
    }
}

impl<T: Into<SmallBuf>> AddAssign<T> for UBitInt {
    fn add_assign(&mut self, rhs: T) {
        add_assign_ubi(self, &rhs.into());
    }
}

fn sub_ubi(lhs: &mut UBitInt, rhs: &[u64]) -> bool {
    if lhs.data.len() < rhs.len() {
        return true;
    }
    if acc(&mut lhs.data, rhs, 1) {
        return true;
    }
    trim_lz(&mut lhs.data);
    return false;
}

impl Sub for UBitInt {
    type Output = UBitInt;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        let of = sub_ubi(&mut lhs, &rhs.data);
        debug_assert!(!of, "attempt to subtract with overflow");
        return lhs;
    }
}

impl Sub for &UBitInt {
    type Output = UBitInt;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut lhs = self.clone();
        let of = sub_ubi(&mut lhs, &rhs.data);
        debug_assert!(!of, "attempt to subtract with overflow");
        return lhs;
    }
}

impl Sub<&UBitInt> for UBitInt {
    type Output = UBitInt;
    fn sub(self, rhs: &UBitInt) -> Self::Output {
        let mut lhs = self;
        let of = sub_ubi(&mut lhs, &rhs.data);
        debug_assert!(!of, "attempt to subtract with overflow");
        return lhs;
    }
}

impl Sub<UBitInt> for &UBitInt {
    type Output = UBitInt;
    fn sub(self, rhs: UBitInt) -> Self::Output {
        let mut lhs = self.clone();
        let of = sub_ubi(&mut lhs, &rhs.data);
        debug_assert!(!of, "attempt to subtract with overflow");
        return lhs;
    }
}

impl<T: Into<SmallBuf>> Sub<T> for UBitInt {
    type Output = UBitInt;
    fn sub(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        let of = sub_ubi(&mut lhs, &rhs.into());
        debug_assert!(!of, "attempt to subtract with overflow");
        return lhs;
    }
}

impl<T: Into<SmallBuf>> Sub<T> for &UBitInt {
    type Output = UBitInt;
    fn sub(self, rhs: T) -> Self::Output {
        let mut lhs = self.clone();
        let of = sub_ubi(&mut lhs, &rhs.into());
        debug_assert!(!of, "attempt to subtract with overflow");
        return lhs;
    }
}

macro_rules! impl_sub_ubi_commute{
    ($($t:ty),*) => {
        $(
        impl Sub<UBitInt> for $t {
            type Output = $t;
            fn sub(self, rhs: UBitInt) -> Self::Output {
                self - Self::try_from(rhs).expect("attempt to subtract with overflow")
            }
        }

        impl Sub<&UBitInt> for $t {
            type Output = $t;
            fn sub(self, rhs: &UBitInt) -> Self::Output {
                self - Self::try_from(rhs).expect("attempt to subtract with overflow")
            }
        }
        )*
    };
}
impl_sub_ubi_commute!(u128, u64);

impl SubAssign for UBitInt {
    fn sub_assign(&mut self, rhs: Self) {
        let of = sub_ubi(self, &rhs.data);
        debug_assert!(!of, "attempt to subtract with overflow");
    }
}

impl SubAssign<&UBitInt> for UBitInt {
    fn sub_assign(&mut self, rhs: &UBitInt) {
        let of = sub_ubi(self, &rhs.data);
        debug_assert!(!of, "attempt to subtract with overflow");
    }
}

impl<T: Into<SmallBuf>> SubAssign<T> for UBitInt {
    fn sub_assign(&mut self, rhs: T) {
        let of = sub_ubi(self, &rhs.into());
        debug_assert!(!of, "attempt to subtract with overflow");
    }
}

impl Mul for UBitInt {
    type Output = UBitInt;
    fn mul(self, rhs: Self) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, &rhs.data);
        if c > 0 {
            data.push(c);
        }
        UBitInt { data }
    }
}

impl Mul for &UBitInt {
    type Output = UBitInt;
    fn mul(self, rhs: Self) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, &rhs.data);
        if c > 0 {
            data.push(c);
        }
        UBitInt { data }
    }
}

impl Mul<&UBitInt> for UBitInt {
    type Output = UBitInt;
    fn mul(self, rhs: &UBitInt) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, &rhs.data);
        if c > 0 {
            data.push(c);
        }
        UBitInt { data }
    }
}

impl Mul<UBitInt> for &UBitInt {
    type Output = UBitInt;
    fn mul(self, rhs: UBitInt) -> Self::Output {
        let (mut data, c) = mul_vec(&self.data, &rhs.data);
        if c > 0 {
            data.push(c);
        }
        UBitInt { data }
    }
}

fn mul_ubi_prim2(mut data: Vec<u64>, rhs: u128) -> UBitInt {
    let c = mul_prim2(&mut data, rhs);
    push_prim2(&mut data, c);
    UBitInt { data }
}

impl Mul<u128> for UBitInt {
    type Output = UBitInt;
    fn mul(self, rhs: u128) -> Self::Output {
        mul_ubi_prim2(self.data, rhs)
    }
}

impl Mul<u128> for &UBitInt {
    type Output = UBitInt;
    fn mul(self, rhs: u128) -> Self::Output {
        mul_ubi_prim2(self.data.clone(), rhs)
    }
}

fn mul_ubi_prim(mut data: Vec<u64>, rhs: u64) -> UBitInt {
    let c = mul_prim(&mut data, rhs);
    if c > 0 {
        data.push(c);
    }
    UBitInt { data }
}

impl Mul<u64> for UBitInt {
    type Output = UBitInt;
    fn mul(self, rhs: u64) -> Self::Output {
        mul_ubi_prim(self.data, rhs)
    }
}

impl Mul<u64> for &UBitInt {
    type Output = UBitInt;
    fn mul(self, rhs: u64) -> Self::Output {
        mul_ubi_prim(self.data.clone(), rhs)
    }
}

impl_commutative!(Mul, mul, UBitInt, |x| x, u128, u64);

fn mul_assign_ubi(a: &mut UBitInt, b: &[u64]) {
    let (data, c) = mul_vec(&a.data, &b);
    a.data = data;
    if c > 0 {
        a.data.push(c);
    }
}

impl MulAssign for UBitInt {
    fn mul_assign(&mut self, rhs: Self) {
        mul_assign_ubi(self, &rhs.data);
    }
}

impl MulAssign<&UBitInt> for UBitInt {
    fn mul_assign(&mut self, rhs: &UBitInt) {
        mul_assign_ubi(self, &rhs.data);
    }
}

impl MulAssign<u128> for UBitInt {
    fn mul_assign(&mut self, rhs: u128) {
        let c = mul_prim2(&mut self.data, rhs);
        push_prim2(&mut self.data, c);
    }
}

impl MulAssign<u64> for UBitInt {
    fn mul_assign(&mut self, rhs: u64) {
        let c = mul_prim(&mut self.data, rhs);
        if c > 0 {
            self.data.push(c);
        }
    }
}

impl Sqr for UBitInt {
    fn sqr(&self) -> Self {
        let (mut data, c) = sqr_vec(&self.data);
        if c > 0 {
            data.push(c);
        }
        UBitInt { data }
    }
}

impl Shl<usize> for UBitInt {
    type Output = UBitInt;
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

impl Shl<usize> for &UBitInt {
    type Output = UBitInt;
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

impl ShlAssign<usize> for UBitInt {
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

impl Shr<usize> for UBitInt {
    type Output = UBitInt;
    fn shr(self, rhs: usize) -> Self::Output {
        let mut lhs = self;
        let div = rhs / 64;
        lhs.data.drain(..div);
        shr_buf(&mut lhs.data, (rhs % 64) as u8);
        return lhs;
    }
}

impl Shr<usize> for &UBitInt {
    type Output = UBitInt;
    fn shr(self, rhs: usize) -> Self::Output {
        let mut lhs = self.clone();
        let div = rhs / 64;
        lhs.data.drain(..div);
        shr_buf(&mut lhs.data, (rhs % 64) as u8);
        return lhs;
    }
}

impl ShrAssign<usize> for UBitInt {
    fn shr_assign(&mut self, rhs: usize) {
        let div = rhs / 64;
        self.data.drain(..div);
        shr_buf(&mut self.data, (rhs % 64) as u8);
    }
}

fn div_rem_ubi(mut n: UBitInt, d: &mut [u64]) -> (UBitInt, UBitInt) {
    let mut q = div_vec(&mut n.data, d);
    trim_lz(&mut n.data);
    trim_lz(&mut q);
    (UBitInt { data: q }, n)
}

impl DivRem for UBitInt {
    type Q = UBitInt;
    type R = UBitInt;
    fn div_rem(self, rhs: Self) -> (UBitInt, UBitInt) {
        let mut d = rhs.data;
        div_rem_ubi(self, &mut d)
    }
}

impl DivRem for &UBitInt {
    type Q = UBitInt;
    type R = UBitInt;
    fn div_rem(self, rhs: Self) -> (UBitInt, UBitInt) {
        let mut d = rhs.data.clone();
        div_rem_ubi(self.clone(), &mut d)
    }
}

impl DivRem<&UBitInt> for UBitInt {
    type Q = UBitInt;
    type R = UBitInt;
    fn div_rem(self, rhs: &UBitInt) -> (Self::Q, Self::R) {
        let mut d = rhs.data.clone();
        div_rem_ubi(self, &mut d)
    }
}

impl DivRem<UBitInt> for &UBitInt {
    type Q = UBitInt;
    type R = UBitInt;
    fn div_rem(self, rhs: UBitInt) -> (Self::Q, Self::R) {
        let mut d = rhs.data;
        div_rem_ubi(self.clone(), &mut d)
    }
}

impl DivRem<u128> for UBitInt {
    type Q = UBitInt;
    type R = u128;
    fn div_rem(self, rhs: u128) -> (Self::Q, Self::R) {
        let (q, r) = div_rem_ubi(self, &mut SmallBuf::from(rhs));
        (q, r.try_into().unwrap())
    }
}

impl DivRem<u128> for &UBitInt {
    type Q = UBitInt;
    type R = u128;
    fn div_rem(self, rhs: u128) -> (Self::Q, Self::R) {
        let (q, r) = div_rem_ubi(self.clone(), &mut SmallBuf::from(rhs));
        (q, r.try_into().unwrap())
    }
}

impl DivRem<u64> for UBitInt {
    type Q = UBitInt;
    type R = u64;
    fn div_rem(self, rhs: u64) -> (Self::Q, Self::R) {
        let mut lhs = self;
        let rem = div_prim(&mut lhs.data, rhs.into());
        trim_lz(&mut lhs.data);
        (lhs, rem)
    }
}

impl DivRem<u64> for &UBitInt {
    type Q = UBitInt;
    type R = u64;
    fn div_rem(self, rhs: u64) -> (Self::Q, Self::R) {
        let mut lhs = self.clone();
        let rem = div_prim(&mut lhs.data, rhs.into());
        trim_lz(&mut lhs.data);
        (lhs, rem)
    }
}

impl<T: U> DivRem<UBitInt> for T
where
    T: TryFrom<UBitInt>,
{
    type Q = T;
    type R = T;
    fn div_rem(self, rhs: UBitInt) -> (Self::Q, Self::R) {
        let Ok(prim) = T::try_from(rhs) else {
            return (T::default(), self);
        };
        (self / prim, self % prim)
    }
}

impl<T: U> DivRem<&UBitInt> for T
where
    T: for<'a> TryFrom<&'a UBitInt>,
{
    type Q = T;
    type R = T;
    fn div_rem(self, rhs: &UBitInt) -> (Self::Q, Self::R) {
        let Ok(prim) = T::try_from(rhs) else {
            return (T::default(), self);
        };
        (self / prim, self % prim)
    }
}

impl<T> Div<T> for UBitInt
where
    UBitInt: DivRem<T>,
{
    type Output = <UBitInt as DivRem<T>>::Q;
    fn div(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).0
    }
}

impl<'a, T> Div<T> for &'a UBitInt
where
    &'a UBitInt: DivRem<T>,
{
    type Output = <&'a UBitInt as DivRem<T>>::Q;
    fn div(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).0
    }
}

impl<T> Rem<T> for UBitInt
where
    UBitInt: DivRem<T>,
{
    type Output = <UBitInt as DivRem<T>>::R;
    fn rem(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).1
    }
}

impl<'a, T> Rem<T> for &'a UBitInt
where
    &'a UBitInt: DivRem<T>,
{
    type Output = <&'a UBitInt as DivRem<T>>::R;
    fn rem(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).1
    }
}

impl_commutative_div_rem!(UBitInt, u128, u64);

impl<T> DivAssign<T> for UBitInt
where
    UBitInt: DivRem<T, Q = UBitInt>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = std::mem::take(self).div_rem(rhs).0;
    }
}

impl RemAssign for UBitInt {
    fn rem_assign(&mut self, rhs: Self) {
        let mut d = rhs.data;
        div_vec(&mut self.data, &mut d);
        trim_lz(&mut self.data);
    }
}

impl RemAssign<&UBitInt> for UBitInt {
    fn rem_assign(&mut self, rhs: &UBitInt) {
        let mut d = rhs.data.clone();
        div_vec(&mut self.data, &mut d);
        trim_lz(&mut self.data);
    }
}

impl PowI<usize> for UBitInt {
    type Output = UBitInt;
    fn powi(&self, rhs: usize) -> Self::Output {
        UBitInt {
            data: powi_vec(&self.data, rhs),
        }
    }
}

impl LogI for UBitInt {
    type Output = usize;

    fn ilog2pow64(&self) -> Self::Output {
        debug_assert!(
            !self.is_zero(),
            "attempt to calculate the logarithm of zero"
        );
        return self.data.len() - 1;
    }

    fn ilog2(&self) -> Self::Output {
        debug_assert!(
            !self.is_zero(),
            "attempt to calculate the logarithm of zero"
        );
        let l = self.data.len() - 1;
        return self.data[l].ilog2() as usize + 64 * l;
    }

    fn ilog(&self, b: u128) -> Self::Output {
        debug_assert!(
            !self.is_zero(),
            "attempt to calculate the logarithm of zero"
        );
        let mut log = self.ilog2() / (b.ilog2() as usize + 1);
        let mut b_ubi = UBitInt::from(b).powi(log + 1);

        while *self >= b_ubi {
            log += 1;
            b_ubi *= b;
        }

        return log;
    }
}
