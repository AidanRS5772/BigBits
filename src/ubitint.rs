use crate::traits::{DivRem, FromErr, FromStrErr, LogI, PowI, SmallBuf, Sqr, UInt, UPrim};
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

macro_rules! impl_try_from_prim {
    ($($t:ty),+) => {
        $(
        impl TryFrom<UBitInt> for $t{
            type Error = FromErr;
            fn try_from(value: UBitInt) -> Result<Self, Self::Error> {
                value.data.try_into::<SmallBuf>()?.try_into::<$t>()
            }
        }

        impl TryFrom<&UBitInt> for $t{
            type Error = FromErr;
            fn try_from(value: &UBitInt) -> Result<Self, Self::Error> {
                value.data.try_into::<SmallBuf>()?.try_into::<$t>()
            }
        }
        )+
    };
}
impl_try_from_prim!(u128, u64, usize, u32, u16, u8);

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

impl<T: UInt> PartialEq<T> for UBitInt {
    fn eq(&self, other: &T) -> bool {
        self.data[..] == other.into()[..]
    }
}

impl PartialOrd for UBitInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(cmp_buf(&self.data, &other.data))
    }
}

impl<T: UInt> PartialOrd<T> for UBitInt {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        Some(cmp_buf(&self.data, &other.into()))
    }
}

impl_commutative_peq_pord!(UBitInt, u128, u64, u32, u16, u8);

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

impl<T: UInt> Add<T> for UBitInt {
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

impl<T: UInt> Add<T> for &UBitInt {
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

impl_commutative!(Add, add, UBitInt, |x| x, u128, u64, u32, u16, u8);

fn add_assign_ubi(lhs: &mut UBitInt, rhs: &[u64]) {
    if lhs.data.len() < rhs.len() {
        lhs.resize(rhs.len(), 0);
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

impl<T: UInt> AddAssign<T> for UBitInt {
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

impl<T: UInt> Sub<T> for UBitInt {
    type Output = UBitInt;
    fn sub(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        let of = sub_ubi(&mut lhs, &rhs.into());
        debug_assert!(!of, "attempt to subtract with overflow");
        return lhs;
    }
}

impl<T: UInt> Sub<T> for &UBitInt {
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
                self - rhs.try_into().expect("attempt to subtract with overflow")
            }
        }

        impl Sub<&UBitInt> for $t {
            type Output = $t;
            fn sub(self, rhs: &UBitInt) -> Self::Output {
                self - rhs.try_into().expect("attempt to subtract with overflow")
            }
        }
        )*
    };
}
impl_sub_ubi_commute!(u128, u64, u32, u16, u8);

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
    let mut c = mul_prim2(&mut data, rhs);
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

impl<T: UPrim> Mul<T> for UBitInt {
    type Output = UBitInt;
    fn mul(self, rhs: T) -> Self::Output {
        mul_ubi_prim(self.data, rhs.into())
    }
}

impl<T: UPrim> Mul<T> for &UBitInt {
    type Output = UBitInt;
    fn mul(self, rhs: T) -> Self::Output {
        mul_ubi_prim(self.data.clone(), rhs.into())
    }
}

impl_commutative!(Mul, mul, UBitInt, |x| x, u128, u64, u32, u16, u8);

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

impl<T: UPrim> MulAssign<T> for UBitInt {
    fn mul_assign(&mut self, rhs: T) {
        let c = mul_prim(&mut self.data, rhs.into());
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

impl<T: UPrim> Shl<T> for UBitInt {
    type Output = UBitInt;
    fn shl(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        shl_vec(&mut lhs.data, rhs.into());
        return lhs;
    }
}

impl<T: UPrim> Shl<T> for &UBitInt {
    type Output = UBitInt;
    fn shl(self, rhs: T) -> Self::Output {
        let mut lhs = self.clone();
        shl_vec(&mut lhs.data, rhs.into());
        return lhs;
    }
}

impl<T: UPrim> ShlAssign<T> for UBitInt {
    fn shl_assign(&mut self, rhs: T) {
        shl_vec(&mut self.data, rhs.into());
    }
}

impl<T: UPrim> Shr<T> for UBitInt {
    type Output = UBitInt;
    fn shr(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        shr_vec(&mut lhs.data, rhs.into());
        return lhs;
    }
}

impl<T: UPrim> Shr<T> for &UBitInt {
    type Output = UBitInt;
    fn shr(self, rhs: T) -> Self::Output {
        let mut lhs = self.clone();
        shr_vec(&mut lhs.data, rhs.into());
        return lhs;
    }
}

impl<T: UPrim> ShrAssign<T> for UBitInt {
    fn shr_assign(&mut self, rhs: T) {
        shr_vec(&mut self.data, rhs.into());
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
    type R = UBitInt;
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

impl<T: UPrim> DivRem<T> for UBitInt {
    type Q = UBitInt;
    type R = T;
    fn div_rem(self, rhs: T) -> (Self::Q, Self::R) {
        let mut lhs = self;
        let rem = div_prim(&mut lhs.data, rhs.into());
        trim_lz(&mut lhs.data);
        (lhs, rem.try_into().unwrap())
    }
}

impl<T: UPrim + TryFrom<u64>> DivRem<T> for &UBitInt {
    type Q = UBitInt;
    type R = T;
    fn div_rem(self, rhs: T) -> (Self::Q, Self::R) {
        let mut lhs = self.clone();
        let rem = div_prim(&mut lhs.data, rhs.into());
        trim_lz(&mut lhs.data);
        (lhs, T::try_from(rem).unwrap())
    }
}

fn div_rem_ubi_bwd<I>(lhs: I, rhs: &UBitInt) -> (I, I)
where
    I: Default + Copy + Div<Output = I> + Rem<Output = I>,
    UBitInt: TryInto<I>,
{
    let Ok(val) = rhs.try_into() else {
        return (I::default(), lhs);
    };
    (lhs / val, lhs % val)
}

impl<T: UInt> DivRem<UBitInt> for T {
    type Q = T;
    type R = T;
    fn div_rem(self, rhs: UBitInt) -> (Self::Q, Self::R) {
        div_rem_ubi_bwd(self, &rhs)
    }
}

impl<T: UInt> DivRem<&UBitInt> for T {
    type Q = T;
    type R = T;
    fn div_rem(self, rhs: &UBitInt) -> (Self::Q, Self::R) {
        div_rem_ubi_bwd(self, rhs)
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

impl<T> Div<T> for &UBitInt
where
    UBitInt: DivRem<T>,
{
    type Output = <UBitInt as DivRem<T>>::Q;
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

impl<T> Rem<T> for &UBitInt
where
    UBitInt: DivRem<T>,
{
    type Output = <UBitInt as DivRem<T>>::R;
    fn rem(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).1
    }
}

impl_commutative_div_rem!(UBitInt, u128, u64, u32, u16, u8);

impl<T> DivAssign<T> for UBitInt
where
    UBitInt: DivRem<T, Q = UBitInt>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = self.div_rem(rhs).0;
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

impl<T: UPrim> PowI<T> for UBitInt {
    type Output = UBitInt;
    fn powi(&self, rhs: T) -> Self::Output {
        UBitInt {
            data: powi_vec(&self.data, rhs.into()),
        }
    }
}

impl LogI for UBitInt {
    type Output = u64;

    fn ilog2(&self) -> Self::Output {
        debug_assert!(
            !self.is_zero(),
            "attempt to calculate the logarithm of zero"
        );
        let l = (self.data.len() - 1) as u64;
        return self.data[l].ilog2() + 64 * l;
    }

    fn ilog2pow64(&self) -> Self::Output {
        debug_assert!(
            !self.is_zero(),
            "attempt to calculate the logarithm of zero"
        );
        return self.data.len() as u64 - 1;
    }

    fn ilog(&self, b: u128) -> Self::Output {
        debug_assert!(
            !self.is_zero(),
            "attempt to calculate the logarithm of zero"
        );
        let mut log = self.ilog2() / (b.ilog2() as u64 + 1);
        let mut b_ubi = UBitInt::from(b).powi(log + 1);

        while *self >= b_ubi {
            log += 1;
            b_ubi *= b;
        }

        return log;
    }
}
