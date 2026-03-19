use super::traits::{DivRem, FromErr, FromStrErr, LogI, PowI, SmallBuf, Sqr, U};
use crate::utils::{div::*, mul::*, utils::*};
use crate::{impl_commutative, impl_commutative_div_rem, impl_commutative_peq_pord};
use std::f64::consts::{LN_10, LN_2};
use std::fmt;
use std::ops::*;
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub struct UBitIntStatic<const N: usize> {
    data: [u64; N],
}

impl<const N: usize> UBitIntStatic<N> {
    pub fn get_data(&self) -> [u64; N] {
        self.data
    }

    pub fn make(data: [u64; N]) -> UBitIntStatic<N> {
        UBitIntStatic::<N> { data }
    }

    pub fn zero() -> UBitIntStatic<N> {
        UBitIntStatic { data: [0; N] }
    }

    pub fn one() -> UBitIntStatic<N> {
        let mut data = [0; N];
        data[0] = 1;
        UBitIntStatic { data }
    }

    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|&x| x == 0)
    }
}

impl<const N: usize, T: Into<SmallBuf>> From<T> for UBitIntStatic<N> {
    fn from(value: T) -> Self {
        let mut data = [0; N];
        let sml_buf = value.into();
        data[..sml_buf.len()].copy_from_slice(&sml_buf);
        UBitIntStatic { data }
    }
}

impl<const N: usize> TryFrom<UBitIntStatic<N>> for u128 {
    type Error = FromErr;
    fn try_from(value: UBitIntStatic<N>) -> Result<Self, Self::Error> {
        Ok(SmallBuf::try_from(value.data.as_slice())?.into())
    }
}

impl<const N: usize> TryFrom<UBitIntStatic<N>> for u64 {
    type Error = FromErr;
    fn try_from(value: UBitIntStatic<N>) -> Result<Self, Self::Error> {
        SmallBuf::try_from(value.data.as_slice())?.try_into()
    }
}

impl<const N: usize> FromStr for UBitIntStatic<N> {
    type Err = FromStrErr;
    fn from_str(str: &str) -> Result<Self, FromStrErr> {
        const CHUNK_SIZE: usize = 19;
        const CHUNK_BASE: u64 = 10_000_000_000_000_000_000;

        let mut end = str.len();
        if end == 0 {
            return Err(FromStrErr::Empty);
        }

        if (str.len() as f64) * LN_10 > (N as f64) * 64.0 * LN_2 {
            return Err(FromStrErr::Overflow);
        }

        let mut result = UBitIntStatic::zero();
        let mut multiplier = UBitIntStatic::one();
        while end > 0 {
            let start = end.saturating_sub(CHUNK_SIZE);
            let chunk_str = &str[start..end];

            if chunk_str.trim() != chunk_str {
                return Err(FromStrErr::Whitespace);
            }
            let chunk_value = chunk_str
                .parse::<u64>()
                .map_err(|_| FromStrErr::MalformedExpression)?;

            result += chunk_value * multiplier;
            multiplier *= CHUNK_BASE;
            end = start;
        }

        Ok(result)
    }
}

impl<const N: usize> fmt::Display for UBitIntStatic<N> {
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

impl<const N: usize> PartialEq for UBitIntStatic<N> {
    fn eq(&self, other: &UBitIntStatic<N>) -> bool {
        eq_buf(&self.data, &other.data)
    }
}

impl<const N: usize, T: U> PartialEq<T> for UBitIntStatic<N> {
    fn eq(&self, other: &T) -> bool {
        eq_buf(&self.data, &(*other).into())
    }
}

impl<const N: usize> PartialOrd for UBitIntStatic<N> {
    fn partial_cmp(&self, other: &UBitIntStatic<N>) -> Option<std::cmp::Ordering> {
        Some(cmp_buf(&self.data, &other.data))
    }
}

impl<const N: usize, T: U> PartialOrd<T> for UBitIntStatic<N> {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        Some(cmp_buf(&self.data, &(*other).into()))
    }
}

impl_commutative_peq_pord!(const N, UBitIntStatic, u128, u64);

impl<const N: usize> Eq for UBitIntStatic<N> {}

impl<const N: usize> Ord for UBitIntStatic<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<const N: usize> Add for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;
    fn add(self, rhs: UBitIntStatic<N>) -> Self::Output {
        let mut lhs = self.data;
        let of = acc(&mut lhs, &rhs.data, 0);
        debug_assert!(!of, "attempt to add with overflow");
        UBitIntStatic::<N> { data: lhs }
    }
}

impl<const N: usize, T: Into<SmallBuf>> Add<T> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;
    fn add(self, rhs: T) -> Self::Output {
        let mut lhs = self.data;
        let of = acc(&mut lhs, &rhs.into(), 0);
        debug_assert!(!of, "attempt to add with overflow");
        UBitIntStatic::<N> { data: lhs }
    }
}

impl_commutative!(const N, Add, add, UBitIntStatic, |x| x, u128, u64);

impl<const N: usize> AddAssign for UBitIntStatic<N> {
    fn add_assign(&mut self, rhs: Self) {
        let of = acc(&mut self.data, &rhs.data, 0);
        debug_assert!(!of, "attempt to add with overflow");
    }
}

impl<const N: usize, T: Into<SmallBuf>> AddAssign<T> for UBitIntStatic<N> {
    fn add_assign(&mut self, rhs: T) {
        let of = acc(&mut self.data, &rhs.into(), 0);
        debug_assert!(!of, "attempt to add with overflow");
    }
}

impl<const N: usize> Sub for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut lhs = self.data;
        let of = acc(&mut lhs, &rhs.data, 1);
        debug_assert!(!of, "attempt to subtract with overflow");
        UBitIntStatic::<N> { data: lhs }
    }
}

impl<const N: usize, T: Into<SmallBuf>> Sub<T> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;
    fn sub(self, rhs: T) -> Self::Output {
        let mut lhs = self.data;
        let of = acc(&mut lhs, &rhs.into(), 1);
        debug_assert!(!of, "attempt to subtract with overflow");
        UBitIntStatic { data: lhs }
    }
}

macro_rules! impl_sub_ubis_commute{
    ($($t:ty),*) => {
        $(
        impl<const N: usize> Sub<UBitIntStatic<N>> for $t {
            type Output = $t;
            fn sub(self, rhs: UBitIntStatic<N>) -> Self::Output {
                self - <$t>::try_from(rhs).expect("attempt to subtract with overflow")
            }
        }
        )*
    };
}
impl_sub_ubis_commute!(u128, u64);

impl<const N: usize> SubAssign for UBitIntStatic<N> {
    fn sub_assign(&mut self, rhs: Self) {
        let of = acc(&mut self.data, &rhs.data, 1);
        debug_assert!(!of, "attempt to subtract with overflow");
    }
}

impl<const N: usize, T: Into<SmallBuf>> SubAssign<T> for UBitIntStatic<N> {
    fn sub_assign(&mut self, rhs: T) {
        let of = acc(&mut self.data, &rhs.into(), 1);
        debug_assert!(!of, "attempt to subtract with overflow");
    }
}

impl<const N: usize> Mul for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;
    fn mul(self, rhs: Self) -> Self::Output {
        let (data, c) = mul_arr(&self.data, &rhs.data).expect("attempt to multiply with overflow");
        debug_assert!(c == 0, "attempt to multiply with overflow");
        UBitIntStatic::<N> { data }
    }
}

impl<const N: usize> Mul<u128> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;
    fn mul(self, rhs: u128) -> Self::Output {
        let mut data = self.data;
        let c = mul_prim2(&mut data, rhs);
        debug_assert!(c != 0, "attempt to multiply with overflow");
        UBitIntStatic { data }
    }
}

impl<const N: usize> Mul<u64> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;
    fn mul(self, rhs: u64) -> Self::Output {
        let mut data = self.data;
        let c = mul_prim(&mut data, rhs);
        debug_assert!(c != 0, "attempt to multiply with overflow");
        UBitIntStatic::<N> { data }
    }
}

impl_commutative!(const N, Mul, mul, UBitIntStatic, |x| x, u128, u64);

impl<const N: usize> MulAssign for UBitIntStatic<N> {
    fn mul_assign(&mut self, rhs: Self) {
        let (data, c) = mul_arr(&self.data, &rhs.data).expect("attempt to multiply with overflow");
        debug_assert!(c == 0, "attempt to multiply with overflow");
        self.data = data;
    }
}

impl<const N: usize> MulAssign<u128> for UBitIntStatic<N> {
    fn mul_assign(&mut self, rhs: u128) {
        let c = mul_prim2(&mut self.data, rhs);
        debug_assert!(c != 0, "attempt to multiply with overflow");
    }
}

impl<const N: usize> MulAssign<u64> for UBitIntStatic<N> {
    fn mul_assign(&mut self, rhs: u64) {
        let c = mul_prim(&mut self.data, rhs.into());
        debug_assert!(c != 0, "attempt to multiply with overflow");
    }
}

impl<const N: usize> Sqr for UBitIntStatic<N> {
    fn sqr(&self) -> Self {
        let (data, c) = sqr_arr(&self.data).expect("attempt to multiply with overflow");
        debug_assert!(c == 0, "attempt to multiply with overflow");
        UBitIntStatic { data }
    }
}

impl<const N: usize> Shl<usize> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;
    fn shl(self, rhs: usize) -> Self::Output {
        let mut lhs = self;
        let div = rhs / 64;
        lhs.data.rotate_right(div);
        shl_buf(&mut lhs.data, (rhs % 64) as u8);
        return lhs;
    }
}

impl<const N: usize> ShlAssign<usize> for UBitIntStatic<N> {
    fn shl_assign(&mut self, rhs: usize) {
        let div = rhs / 64;
        self.data.rotate_right(div);
        shl_buf(&mut self.data, (rhs % 64) as u8);
    }
}

impl<const N: usize> Shr<usize> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;
    fn shr(self, rhs: usize) -> Self::Output {
        let mut lhs = self;
        let div = rhs / 64;
        lhs.data[..div].fill(0);
        lhs.data.rotate_left(div);
        shr_buf(&mut lhs.data, (rhs % 64) as u8);
        return lhs;
    }
}

impl<const N: usize> ShrAssign<usize> for UBitIntStatic<N> {
    fn shr_assign(&mut self, rhs: usize) {
        let div = rhs / 64;
        self.data[..div].fill(0);
        self.data.rotate_left(div);
        shr_buf(&mut self.data, (rhs % 64) as u8);
    }
}

impl<const N: usize> DivRem for UBitIntStatic<N> {
    type Q = UBitIntStatic<N>;
    type R = UBitIntStatic<N>;
    fn div_rem(self, rhs: Self) -> (UBitIntStatic<N>, UBitIntStatic<N>) {
        let n_len = buf_len(&self.data);
        let mut n = self.data;
        let d_len = buf_len(&rhs.data);
        let mut d = rhs.data;
        let q = div_arr(&mut n[..n_len], &mut d[..d_len]);
        (UBitIntStatic { data: q }, UBitIntStatic { data: n })
    }
}

impl<const N: usize> DivRem<u128> for UBitIntStatic<N> {
    type Q = UBitIntStatic<N>;
    type R = u128;
    fn div_rem(self, rhs: u128) -> (Self::Q, Self::R) {
        let n_len = buf_len(&self.data);
        let mut n = self.data;
        let q = div_arr(&mut n[..n_len], &mut SmallBuf::from(rhs));
        (
            UBitIntStatic { data: q },
            SmallBuf::try_from(&n[..2]).unwrap().into(),
        )
    }
}

impl<const N: usize> DivRem<u64> for UBitIntStatic<N> {
    type Q = UBitIntStatic<N>;
    type R = u64;
    fn div_rem(self, rhs: u64) -> (Self::Q, Self::R) {
        let mut data = self.data;
        let rem = div_prim(&mut data, rhs.into());
        (UBitIntStatic { data }, rem)
    }
}

impl<const N: usize, T: U> DivRem<UBitIntStatic<N>> for T
where
    T: TryFrom<UBitIntStatic<N>>,
{
    type Q = T;
    type R = T;
    fn div_rem(self, rhs: UBitIntStatic<N>) -> (Self::Q, Self::R) {
        let Ok(prim) = T::try_from(rhs) else {
            return (T::default(), self);
        };
        (self / prim, self % prim)
    }
}

impl<const N: usize, T> Div<T> for UBitIntStatic<N>
where
    UBitIntStatic<N>: DivRem<T>,
{
    type Output = <UBitIntStatic<N> as DivRem<T>>::Q;
    fn div(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).0
    }
}

impl<const N: usize, T> Rem<T> for UBitIntStatic<N>
where
    UBitIntStatic<N>: DivRem<T>,
{
    type Output = <UBitIntStatic<N> as DivRem<T>>::R;
    fn rem(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).1
    }
}

impl_commutative_div_rem!(const N, UBitIntStatic, u128, u64);

impl<const N: usize, T> DivAssign<T> for UBitIntStatic<N>
where
    UBitIntStatic<N>: DivRem<T, Q = UBitIntStatic<N>>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = self.div_rem(rhs).0;
    }
}

impl<const N: usize> RemAssign for UBitIntStatic<N> {
    fn rem_assign(&mut self, rhs: Self) {
        let n_len = buf_len(&self.data);
        let d_len = buf_len(&rhs.data);
        let mut d = rhs.data;
        div_arr::<N>(&mut self.data[..n_len], &mut d[..d_len]);
    }
}

impl<const N: usize> PowI<usize> for UBitIntStatic<N> {
    type Output = UBitIntStatic<N>;
    fn powi(&self, rhs: usize) -> Self::Output {
        UBitIntStatic {
            data: powi_arr(&self.data, rhs).expect("attempt to take integer power with overflow"),
        }
    }
}

impl<const N: usize> LogI for UBitIntStatic<N> {
    type Output = usize;

    fn ilog2pow64(&self) -> Self::Output {
        let len = buf_len(&self.data);
        debug_assert!(len != 0, "attempt to calculate the logarithm of zero");
        return len - 1;
    }

    fn ilog2(&self) -> Self::Output {
        let l = self.ilog2pow64();
        return l * 64 + self.data[l].ilog2() as usize;
    }

    fn ilog(&self, b: u128) -> Self::Output {
        let mut log = self.ilog2() / (b.ilog2() as usize + 1);
        let mut b_ubis = UBitIntStatic::<N>::from(b).powi(log + 1);

        while *self >= b_ubis {
            log += 1;
            b_ubis *= b;
        }

        return log;
    }
}
