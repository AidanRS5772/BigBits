use crate::traits::{DivRem, FromErr, FromStrErr, Int, LogI, PowI, Prim, SmallBuf, Sqr, UPrim};
use crate::ubitint_static::UBitIntStatic;
use crate::{impl_commutative, impl_commutative_peq_pord};
use crate::{impl_commutative_div_rem, utils::*};
use std::f64::consts::{LN_10, LN_2};
use std::fmt;
use std::ops::*;
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub struct BitIntStatic<const N: usize> {
    data: [u64; N],
    sign: bool,
}

impl<const N: usize> BitIntStatic<N> {
    #[inline]
    pub fn get_data(&self) -> [u64; N] {
        self.data
    }

    #[inline]
    pub fn get_sign(&self) -> bool {
        self.sign
    }

    #[inline]
    pub fn make(data: [u64; N], sign: bool) -> Self {
        BitIntStatic { data, sign }
    }

    const ZERO: Self = BitIntStatic {
        data: [0; N],
        sign: false,
    };

    const ONE: Self = {
        let mut data = [0; N];
        data[0] = 1;
        BitIntStatic { data, sign: false }
    };

    const NEG_ONE: Self = {
        let mut data = [0; N];
        data[0] = 1;
        BitIntStatic { data, sign: true }
    };

    #[inline]
    pub fn mut_neg(&mut self) {
        self.sign ^= true;
    }

    #[inline]
    pub fn mut_abs(&mut self) {
        self.sign = false;
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|&x| x == 0)
    }
}

impl<const N: usize, T: Int> From<T> for BitIntStatic<N> {
    fn from(value: T) -> Self {
        let mut data = [0; N];
        let buf = SmallBuf::from(value);
        data[..buf.len()].copy_from_slice(&buf);
        BitIntStatic {
            data,
            sign: value.sign(),
        }
    }
}

impl<const N: usize> TryFrom<BitIntStatic<N>> for u128 {
    type Error = FromErr;
    fn try_from(value: BitIntStatic<N>) -> Result<Self, Self::Error> {
        if value.sign {
            return Err(FromErr::Sign);
        }
        let buf = SmallBuf::try_from(value.data.as_slice())?;
        Ok(buf.into())
    }
}

impl<const N: usize> TryFrom<BitIntStatic<N>> for i128 {
    type Error = FromErr;
    fn try_from(value: BitIntStatic<N>) -> Result<Self, Self::Error> {
        let uval: u128 = SmallBuf::try_from(value.data.as_slice())?.into();
        let ival: i128 = uval.try_into().map_err(|_| FromErr::Overflow)?;
        Ok(if value.sign { -ival } else { ival })
    }
}

macro_rules! impl_try_from_bi {
    ($(($u:ty, $i:ty)), +) => {
        $(
        impl<const N: usize> TryFrom<BitIntStatic<N>> for $u{
            type Error = FromErr;
            fn try_from(value: BitIntStatic<N>) -> Result<Self, Self::Error> {
                if value.sign {
                    return Err(FromErr::Sign);
                }
                SmallBuf::try_from(value.data.as_slice())?.try_into()
            }
        }

        impl<const N: usize> TryFrom<BitIntStatic<N>> for $i{
            type Error = FromErr;
            fn try_from(value: BitIntStatic<N>) -> Result<Self, Self::Error> {
                let uval: $u = SmallBuf::try_from(value.data.as_slice())?.try_into()?;
                let mut ival: $i = uval.try_into().map_err(|_| FromErr::Overflow)?;
                Ok(if value.sign { -ival } else { ival })
            }
        }
        )+
    };
}
impl_try_from_bi!((u64, i64), (u32, i32), (u16, i16), (u8, i8));

impl<const N: usize> FromStr for BitIntStatic<N> {
    type Err = FromStrErr;
    fn from_str(str: &str) -> Result<Self, FromStrErr> {
        const CHUNK_SIZE: usize = 19;
        const CHUNK_BASE: u64 = 10_000_000_000_000_000_000;

        if (str.len() as f64) * LN_10 > (N as f64) * 64.0 * LN_2 {
            return Err(FromStrErr::Overflow);
        }

        let neg = if let Some(last) = str.chars().next() {
            if last == '-' {
                true
            } else {
                false
            }
        } else {
            return Err(FromStrErr::Empty);
        };

        if neg {
            str.remove(0);
        }

        let mut result = BitIntStatic::zero();
        let mut multiplier = BitIntStatic::one();
        let mut end = str.len();
        while end > 0 {
            let mut start = end.saturating_sub(CHUNK_SIZE);
            let chunk_str = &str[start..end];

            if chunk_str.trim() != chunk_str {
                return Err(FromStrErr::Whitespace);
            }
            let chunk_value = chunk_str
                .parse::<usize>()
                .map_err(|_| FromStrErr::MalformedExpression)?;

            result += chunk_value * &multiplier;
            multiplier *= CHUNK_BASE;
            end = start;
        }

        result.sign = neg;
        Ok(result)
    }
}

impl<const N: usize> fmt::Display for BitIntStatic<N> {
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

impl<const N: usize> PartialEq for BitIntStatic<N> {
    fn eq(&self, other: &Self) -> bool {
        if self.sign ^ other.sign {
            return false;
        }
        self.data == other.data
    }
}

impl<const N: usize, T: Int> PartialEq<T> for BitIntStatic<N> {
    fn eq(&self, other: &T) -> bool {
        if self.sign ^ other.sign() {
            return false;
        }
        eq_buf(&self.data, &other.unsigned().into())
    }
}
impl<const N: usize> PartialOrd for BitIntStatic<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.sign ^ other.sign {
            return Some(scmp(self.sign, std::cmp::Ordering::Greater));
        }
        Some(scmp(self.sign, cmp_buf(&self.data, &other.data)))
    }
}

impl<const N: usize, T: Int> PartialOrd<T> for BitIntStatic<N> {
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

impl_commutative_peq_pord!(const N, BitIntStatic, u128, i128, u64, i64, u32, i32, u16, i16, u8, i8);

impl<const N: usize> Eq for BitIntStatic<N> {}

impl<const N: usize> Ord for BitIntStatic<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

fn add_sub<const N: usize>(lhs: &mut BitIntStatic<N>, rhs: &[u64], sub: bool) -> bool {
    let comp = lhs.sign ^ sub;
    let of = acc(&mut lhs.data, rhs, comp as u8);
    lhs.sign ^= of && comp;
    if of && comp {
        twos_comp(&mut lhs.data);
    }
    return of && !comp;
}

impl<const N: usize> Add for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        let of = add_sub(&mut lhs, &rhs.data, rhs.sign);
        debug_assert!(!of, "attempt to add with overflow");
        return lhs;
    }
}

impl<const N: usize, T: Int> Add<T> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn add(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        let of = add_sub(&mut lhs, &rhs.unsigned().into(), rhs.sign());
        debug_assert!(!of, "attempt to add with overflow");
        return lhs;
    }
}

impl_commutative!(const N, Add, add, BitIntStatic, |x| x, u128, i128, u64, i64, u32, i32, u16, i16, u8, i8);

impl<const N: usize> AddAssign for BitIntStatic<N> {
    fn add_assign(&mut self, rhs: Self) {
        let of = add_sub(self, &rhs.data, rhs.sign);
        debug_assert!(!of, "attempt to add with overflow");
    }
}

impl<const N: usize, T: Int> AddAssign<T> for BitIntStatic<N> {
    fn add_assign(&mut self, rhs: T) {
        let of = add_sub(self, &rhs.unsigned().into(), rhs.sign());
        debug_assert!(!of, "attempt to add with overflow");
    }
}

impl<const N: usize> Neg for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn neg(self) -> Self::Output {
        BitIntStatic::<N> {
            data: self.data,
            sign: !self.sign,
        }
    }
}

impl<const N: usize> Sub for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        let of = add_sub(&mut lhs, &rhs.data, !rhs.sign);
        debug_assert!(!of, "attempt to add with overflow");
        return lhs;
    }
}

impl<const N: usize, T: Int> Sub<T> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn sub(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        let of = add_sub(&mut lhs, &rhs.unsigned().into(), !rhs.sign());
        debug_assert!(!of, "attempt to add with overflow");
        return lhs;
    }
}

impl_commutative!(const N, Sub, sub, BitIntStatic, |x| -x, u128, i128, u64, i64, u32, i32, u16, i16, u8, i8);

impl<const N: usize> SubAssign for BitIntStatic<N> {
    fn sub_assign(&mut self, rhs: Self) {
        let of = add_sub(self, &rhs.data, !rhs.sign);
        debug_assert!(!of, "attempt to add with overflow");
    }
}

impl<const N: usize, T: Int> SubAssign<T> for BitIntStatic<N> {
    fn sub_assign(&mut self, rhs: T) {
        let of = add_sub(self, &rhs.unsigned().into(), !rhs.sign());
        debug_assert!(!of, "attempt to add with overflow")
    }
}

impl<const N: usize> Mul for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn mul(self, rhs: Self) -> Self::Output {
        let (data, c) = mul_arr(&self.data, &rhs.data).expect("attempt to multiply with overflow");
        debug_assert!(c == 0, "attempt to multiply with overflow");
        BitIntStatic {
            data,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl<const N: usize> Mul<i128> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn mul(self, rhs: i128) -> Self::Output {
        let mut data = self.data;
        let c = mul_prim2(&mut data, rhs.unsigned_abs());
        debug_assert!(c != 0, "attempt to multiply with overflow");
        BitIntStatic {
            data,
            sign: self.sign ^ (rhs < 0),
        }
    }
}

impl<const N: usize> Mul<u128> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn mul(self, rhs: u128) -> Self::Output {
        let mut data = self.data;
        let c = mul_prim2(&mut data, rhs);
        debug_assert!(c != 0, "attempt to multiply with overflow");
        BitIntStatic {
            data,
            sign: self.sign,
        }
    }
}

impl<const N: usize, T: Prim> Mul<T> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn mul(self, rhs: T) -> Self::Output {
        let mut data = self.data;
        let c = mul_prim(&mut data, rhs.unsigned().into());
        debug_assert!(c != 0, "attempt to multiply with overflow");
        BitIntStatic {
            data,
            sign: self.sign ^ rhs.sign(),
        }
    }
}

impl_commutative!(const N, Mul, mul, BitIntStatic, |x| x, u128, i128, u64, i64, u32, i32, u16, i16, u8, i8);

impl<const N: usize> MulAssign for BitIntStatic<N> {
    fn mul_assign(&mut self, rhs: Self) {
        let (data, c) = mul_arr(&self.data, &rhs.data).expect("attempt to multiply with overflow");
        debug_assert!(c == 0, "attempt to multiply with overflow");
        self.data = data;
        self.sign ^= rhs.sign;
    }
}

impl<const N: usize> MulAssign<i128> for BitIntStatic<N> {
    fn mul_assign(&mut self, rhs: i128) {
        let c = mul_prim2(&mut self.data, rhs.unsigned_abs());
        debug_assert!(c != 0, "attempt to multiply with overflow");
        self.sign ^= rhs < 0;
    }
}

impl<const N: usize> MulAssign<u128> for BitIntStatic<N> {
    fn mul_assign(&mut self, rhs: u128) {
        let c = mul_prim2(&mut self.data, rhs);
        debug_assert!(c != 0, "attempt to multiply with overflow");
    }
}

impl<const N: usize, T: Prim> MulAssign<T> for BitIntStatic<N> {
    fn mul_assign(&mut self, rhs: T) {
        let c = mul_prim(&mut self.data, rhs.unsigned().into());
        debug_assert!(c != 0, "attempt to multiply with overflow");
    }
}

impl<const N: usize> Sqr for BitIntStatic<N> {
    fn sqr(&self) -> Self {
        let (data, c) = sqr_arr(&self.data).expect("attempt to multiply with overflow");
        debug_assert!(c == 0, "attempt to multiply with overflow");
        BitIntStatic { data, sign: false }
    }
}

impl<const N: usize, T: UPrim> Shl<T> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn shl(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        shl_arr(&mut lhs.data, rhs.into());
        return lhs;
    }
}

impl<const N: usize, T: UPrim> ShlAssign<T> for BitIntStatic<N> {
    fn shl_assign(&mut self, rhs: T) {
        shl_arr(&mut self.data, rhs.into());
    }
}

impl<const N: usize, T: UPrim> Shr<T> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn shr(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        shl_arr(&mut lhs.data, rhs.into());
        return lhs;
    }
}

impl<const N: usize, T: UPrim> ShrAssign<T> for BitIntStatic<N> {
    fn shr_assign(&mut self, rhs: T) {
        shl_arr(&mut self.data, rhs.into());
    }
}

impl<const N: usize> DivRem for BitIntStatic<N> {
    type Q = BitIntStatic<N>;
    type R = UBitIntStatic<N>;
    fn div_rem(self, rhs: Self) -> (Self::Q, Self::R) {
        let n_len = buf_len(&self.data);
        let mut n = self.data;
        let d_len = buf_len(&rhs.data);
        let mut d = rhs.data[..d_len];
        let q = div_arr::<N>(&mut n[..n_len], &mut d);
        (
            BitIntStatic {
                data: q,
                sign: self.sign ^ rhs.sign,
            },
            UBitIntStatic::make(n),
        )
    }
}

impl<const N: usize> DivRem<i128> for BitIntStatic<N> {
    type Q = BitIntStatic<N>;
    type R = u128;
    fn div_rem(self, rhs: i128) -> (Self::Q, Self::R) {
        let n_len = buf_len(&self.data);
        let mut n = self.data;
        let mut d = SmallBuf::from(rhs.unsigned());
        let q = div_arr::<N>(&mut n[..n_len], &mut d);
        (
            BitIntStatic {
                data: q,
                sign: self.sign ^ rhs.sign(),
            },
            SmallBuf::from(&n[..2]).into(),
        )
    }
}

impl<const N: usize> DivRem<u128> for BitIntStatic<N> {
    type Q = BitIntStatic<N>;
    type R = u128;
    fn div_rem(self, rhs: u128) -> (Self::Q, Self::R) {
        let n_len = buf_len(&self.data);
        let mut n = self.data;
        let mut d = SmallBuf::from(rhs);
        let q = div_arr::<N>(&mut n[..n_len], &mut d);
        (
            BitIntStatic {
                data: q,
                sign: self.sign,
            },
            SmallBuf::try_from(&n[..2]).unwrap().into(),
        )
    }
}

impl<const N: usize, T: Prim> DivRem<T> for BitIntStatic<N>
where
    <T as Prim>::U: TryFrom<u64>,
{
    type Q = BitIntStatic<N>;
    type R = <T as Prim>::U;
    fn div_rem(self, rhs: T) -> (Self::Q, Self::R) {
        let mut lhs = self;
        let r = div_prim(&mut lhs.data, rhs.unsigned().into());
        lhs.sign ^= rhs.sign();
        (lhs, Self::R::try_from(r).unwrap())
    }
}

fn divrem_bis_bwd<const N: usize, I, Q, R>(lhs: I, rhs: &BitIntStatic<N>) -> (Q, R)
where
    I: Into<R> + Div<Q, Output = Q> + Rem<Q, Output = R>,
    Q: Default,
    R: Into<Q>,
{
    let Ok(val) = rhs.to() else {
        return (Q::default(), lhs.into());
    };
    (lhs / val, lhs % val)
}

impl<const N: usize, T: Int> DivRem<BitIntStatic<N>> for T {
    type Q = <T as Int>::I;
    type R = <T as Int>::U;
    fn div_rem(self, rhs: BitIntStatic<N>) -> (Self::Q, Self::R) {
        divrem_bis_bwd(self, &rhs)
    }
}

impl<const N: usize, T> Div<T> for BitIntStatic<N>
where
    BitIntStatic<N>: DivRem<T>,
{
    type Output = <BitIntStatic<N> as DivRem<T>>::Q;
    fn div(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).0
    }
}

impl<const N: usize, T> Rem<T> for BitIntStatic<N>
where
    BitIntStatic<N>: DivRem<T>,
{
    type Output = <BitIntStatic<N> as DivRem<T>>::R;
    fn rem(self, rhs: T) -> Self::Output {
        self.div_rem(rhs).1
    }
}

impl_commutative_div_rem!(const N, BitIntStatic, i128, u128, i64, u64, i32, u32, i16, u16, i8, u8);

impl<const N: usize, T> DivAssign<T> for BitIntStatic<N>
where
    BitIntStatic<N>: DivRem<T, Q = BitIntStatic<N>>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = self.div_rem(rhs).0;
    }
}

impl<const N: usize, T: UPrim> PowI<T> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;
    fn powi(&self, rhs: T) -> Self::Output {
        BitIntStatic {
            data: powi_arr(&self.data, rhs.into())
                .expect("attempt to take integer power with overflow"),
            sign: self.sign && (rhs.into() & 1 == 1),
        }
    }
}

impl<const N: usize> LogI for BitIntStatic<N> {
    type Output = u64;

    fn ilog2pow64(&self) -> Self::Output {
        let len = buf_len(&self.data) as u64;
        debug_assert!(len != 0, "attempt to calculate the logarithm of zero");
        debug_assert!(!self.sign, "argument of integer logarithm must be positive");

        return len - 1;
    }

    fn ilog2(&self) -> Self::Output {
        let l = self.ilog2pow64();
        return l * 64 + self.data[l].ilog2();
    }

    fn ilog(&self, b: u128) -> Self::Output {
        let mut log = self.ilog2() / (b.ilog2() as u64 + 1);
        let mut b_ubis = UBitIntStatic::<N>::from(b).powi(log + 1);

        while self >= &b_ubis {
            log += 1;
            b_ubis *= b;
        }

        return log;
    }
}
