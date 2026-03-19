use crate::{
    impl_commutative, impl_commutative_peq_pord,
    traits::{
        DivVariants, FromErr, FromStrErr, Int, MulVariants, Prim, Recipricol, Rounding, SmallBuf,
        Sqr, SqrVariants, SEM,
    },
    utils::*,
};
use std::{fmt, ops::*, str::FromStr};

#[derive(Debug, Clone)]
struct Mantissa {
    buf: Vec<u64>,
    start: usize,
    prev_sz: usize,
}

impl Mantissa {
    fn find_fib(n: usize) -> (usize, usize) {
        if n <= 2 {
            return (2, 1);
        }
        const SQRT_5: f64 = 5.0.sqrt();
        const PHI: f64 = (1.0 + SQRT_5) / 2.0;
        let i = ((n as f64) * SQRT_5).log(PHI) as i32;
        let mut f0 = (PHI.powi(i) / SQRT_5 + 0.5) as usize;
        let mut f1 = (f0 * PHI + 0.5) as usize;
        if f1 <= n {
            f1 += f0;
            f0 = f1 - f0;
        }
        (f1, f0)
    }

    fn new() -> Self {
        Mantissa {
            buf: vec![0; 2],
            start: 2,
            prev_sz: 1,
        }
    }

    fn make_take(mut buf: Vec<u64>) -> Self {
        let len = buf.len();
        let (sz, prev_sz) = Self::find_fib(len);
        let start = sz - len;
        buf.resize(sz, 0);
        buf.copy_within(..len, start);
        Mantissa {
            buf: buf,
            start,
            prev_sz,
        }
    }

    fn make_ref(buf: &[u64]) -> Self {
        let len = buf.len();
        let (sz, prev_sz) = Self::find_fib(len);
        let start = sz - len;
        let mut new_buf = vec![0; sz];
        new_buf[sz..].copy_from_slice(buf);
        Mantissa {
            buf: new_buf,
            start,
            prev_sz,
        }
    }

    fn len(&self) -> usize {
        self.buf.len() - self.start
    }

    fn is_empty(&self) -> bool {
        return self.start == self.buf.len();
    }

    fn push(&mut self, val: u64) {
        self.buf.push(val);
    }

    fn push_slice(&mut self, vals: &[u64]) {
        self.buf.extend_from_slice(vals);
    }

    fn push_zeros(&mut self, amt: usize) {
        self.buf.resize(self.buf.len() + amt, 0);
    }

    fn realloc(&mut self) {
        let len = self.buf.len();
        let new_start = self.start + self.prev_sz;
        self.buf.resize(len + self.prev_sz, 0);
        self.buf.copy_within(self.start..len, new_start);
        self.prev_sz = len;
        self.start = new_start;
    }

    fn put(&mut self, val: u64) {
        if self.start == 0 {
            self.realloc();
        }
        self.start -= 1;
        self.buf[self.start] = val;
    }

    fn put_slice(&mut self, vals: &[u64]) {
        if self.start < vals.len() {
            self.realloc();
            while self.start < vals.len() {
                self.realloc();
            }
        }
        self.start -= vals.len();
        self.buf[self.start..self.start + vals.len()].copy_from_slice(vals);
    }

    fn put_zeros(&mut self, amt: usize) {
        if amt == 0 {
            return;
        }
        if self.start < amt {
            self.realloc();
            while self.start < amt {
                self.realloc();
            }
        }
        self.start -= amt;
        self.buf[self.start..self.start + amt].fill(0);
    }

    fn drain(&mut self, amt: usize) {
        self.start = (self.start + amt).min(self.len());
    }

    fn trunc(&mut self, amt: usize) {
        self.buf.truncate(self.buf.len().saturating_sub(amt));
    }

    fn pop(&mut self) -> u64 {
        self.buf.pop().unwrap()
    }

    fn pull(&mut self) -> u64 {
        self.start += 1;
        self.buf[self.start - 1]
    }

    fn trim_zeros(&mut self) -> Option<usize> {
        let mut fnt_idx = self.start;
        let mut end_idx = self.buf.len() - 1;
        loop {
            if fnt_idx >= end_idx {
                self.start = self.buf.len();
                return None;
            }
            let fnt_zero = self.buf[fnt_idx] == 0;
            let end_zero = self.buf[end_idx] == 0;
            if !(fnt_zero || end_zero) {
                break;
            }
            if fnt_zero {
                fnt_idx += 1;
            }
            if end_zero {
                end_idx -= 1;
            }
        }
        self.start = fnt_idx;
        let end_truc_len = self.buf.len() - end_idx - 1;
        self.buf.truncate(end_idx + 1);
        return Some(end_truc_len);
    }

    fn take(&self, n: usize) -> &[u64] {
        let idx = self.buf.len().saturating_sub(n.max(1)).max(self.start);
        &self.buf[idx..]
    }

    fn top(&self) -> u64 {
        *self.last().unwrap()
    }
}

impl Deref for Mantissa {
    type Target = [u64];
    fn deref(&self) -> &[u64] {
        &self.buf[self.start..]
    }
}

impl DerefMut for Mantissa {
    fn deref_mut(&mut self) -> &mut [u64] {
        &mut self.buf[self.start..]
    }
}

impl PartialEq for Mantissa {
    fn eq(&self, other: &Self) -> bool {
        self.buf[self.start..] == other.buf[other.start..]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BitFloat {
    s: bool,
    e: i128,
    m: Mantissa,
}

impl BitFloat {
    pub fn get_m(&self) -> &[u64] {
        &self.m
    }

    pub fn get_e(&self) -> i128 {
        self.e
    }

    pub fn get_s(&self) -> bool {
        self.s
    }

    pub fn make(s: bool, e: i128, m: &[u64]) -> Self {
        BitFloat {
            s,
            e,
            m: Mantissa::make_ref(m),
        }
    }

    pub const ZERO: Self = BitFloat {
        s: false,
        e: i128::MIN,
        m: Mantissa::new(),
    };

    pub const ONE: Self = BitFloat {
        s: false,
        e: 0,
        m: Mantissa::make_ref(&[1]),
    };

    pub const NEG_ONE: Self = BitFloat {
        s: true,
        e: 0,
        m: Mantissa::make_ref(&[1]),
    };

    pub const INF: Self = BitFloat {
        s: false,
        e: i128::MAX,
        m: Mantissa::new(),
    };

    pub const NEG_INF: Self = BitFloat {
        s: true,
        e: i128::MAX,
        m: Mantissa::new(),
    };

    pub fn is_zero(&self) -> bool {
        self.e == i128::MIN
    }

    pub fn neg_mut(&mut self) {
        self.s ^= true;
    }

    pub fn abs(&self) -> Self {
        let mut out = self.clone();
        out.s = true;
        return out;
    }

    pub fn abs_mut(&mut self) {
        self.s = true;
    }
}

impl Rounding for BitFloat {
    fn floor(&self) -> Self {
        if self.e < 0 {
            return if self.s { Self::NEG_ONE } else { Self::ZERO };
        }

        let mut out = self.clone();
        let amt = self
            .m
            .len()
            .saturating_sub(usize::try_from(self.e).unwrap_or(self.m.len()));
        out.m.drain(amt);
        if out.s {
            if dec(&mut out.m) {
                out.m.pop();
                out.e -= 1;
            }
        }

        out
    }

    fn floor_mut(&mut self) {
        if self.e < 0 {
            if self.s {
                *self = Self::NEG_ONE;
            } else {
                *self = Self::ZERO;
            };
            return;
        }

        let amt = self
            .m
            .len()
            .saturating_sub(usize::try_from(self.e).unwrap_or(self.m.len()));
        self.m.drain(amt);
        if self.s {
            if dec(&mut self.m) {
                self.m.pop();
                self.e -= 1;
            }
        }
    }

    fn ceil(&self) -> Self {
        if self.e < 0 {
            return if self.s { Self::ZERO } else { Self::ONE };
        }

        let mut out = self.clone();
        let amt = self
            .m
            .len()
            .saturating_sub(usize::try_from(self.e).unwrap_or(self.m.len()));
        out.m.drain(amt);
        if !out.s {
            if inc(&mut out.m) {
                out.m.push(1);
                out.e += 1;
            }
        }

        out
    }

    fn ceil_mut(&mut self) {
        if self.e < 0 {
            if self.s {
                *self = Self::ZERO;
            } else {
                *self = Self::ONE;
            };
            return;
        }

        let amt = self
            .m
            .len()
            .saturating_sub(usize::try_from(self.e).unwrap_or(self.m.len()));
        self.m.drain(amt);
        if !self.s {
            if inc(&mut self.m) {
                self.m.push(1);
                self.e += 1;
            }
        }
    }

    fn round(&self) -> Self {
        if self.e < -1 {
            return Self::ZERO;
        }
        if self.e < 0 {
            return if self.m.top() >> 63 > 0 {
                if self.s {
                    Self::NEG_ONE
                } else {
                    Self::ONE
                }
            } else {
                Self::ZERO
            };
        }

        let mut out = self.clone();
        let amt = self
            .m
            .len()
            .saturating_sub(usize::try_from(self.e - 1).unwrap_or(self.m.len()));
        out.m.drain(amt);

        if out.m.pull() >> 63 > 0 {
            if out.s {
                if dec(&mut out.m) {
                    out.m.pop();
                    out.e -= 1;
                }
            } else {
                if inc(&mut out.m) {
                    out.m.push(1);
                    out.e += 1;
                }
            }
        }

        out
    }

    fn round_mut(&mut self) {
        if self.e < -1 {
            *self = Self::ZERO;
            return;
        }
        if self.e < 0 {
            if self.m.top() >> 63 > 0 {
                if self.s {
                    *self = Self::NEG_ONE;
                } else {
                    *self = Self::ONE;
                }
            } else {
                *self = Self::ZERO;
            };
            return;
        }

        let amt = self
            .m
            .len()
            .saturating_sub(usize::try_from(self.e - 1).unwrap_or(self.m.len()));
        self.m.drain(amt);

        if self.m.pull() >> 63 > 0 {
            if self.s {
                if dec(&mut self.m) {
                    self.m.pop();
                    self.e -= 1;
                }
            } else {
                if inc(&mut self.m) {
                    self.m.push(1);
                    self.e += 1;
                }
            }
        }
    }

    fn trunc(&self) -> Self {
        if self.e < 0 {
            return Self::ZERO;
        }

        let mut out = self.clone();
        let amt = self
            .m
            .len()
            .saturating_sub(usize::try_from(self.e).unwrap_or(self.m.len()));
        out.m.drain(amt);

        return out;
    }

    fn trunc_mut(&mut self) {
        if self.e < 0 {
            *self = Self::ZERO;
            return;
        }

        let amt = self
            .m
            .len()
            .saturating_sub(usize::try_from(self.e).unwrap_or(self.m.len()));
        self.m.drain(amt);
    }

    fn fract(&self) -> Self {
        if self.e < 0 {
            return self.clone();
        }

        let Ok(amt) = usize::try_from(self.e + 1) else {
            return Self::ZERO;
        };

        if amt > self.m.len() {
            return Self::ZERO;
        }

        let mut out = self.clone();
        out.m.trunc(amt);
        return out;
    }

    fn fract_mut(&mut self) {
        if self.e < 0 {
            return;
        }

        let Ok(amt) = usize::try_from(self.e + 1) else {
            *self = Self::ZERO;
            return;
        };
        if amt > self.m.len() {
            *self = Self::ZERO;
            return;
        }

        self.m.trunc(amt);
    }
}

impl<T: Into<SEM>> From<T> for BitFloat {
    fn from(value: T) -> Self {
        let sem = value.into();
        return BitFloat {
            s: sem.s,
            e: sem.e,
            m: Mantissa::make_ref(&sem.m),
        };
    }
}

macro_rules! impl_try_from_bf {
    ($($t:ty), +) => {
        $(
        impl TryFrom<BitFloat> for $t{
            type Error = FromErr;
            fn try_from(value: BitFloat) -> Result<$t, Self::Error>{
                let start = value.m.len().saturating_sub(2);
                let sem = SEM{
                    s: value.s,
                    e: value.e,
                    m: SmallBuf::try_from(&value.m[start..])?,
                };
                sem.try_into()
            }
        }

        impl TryFrom<&BitFloat> for $t{
            type Error = FromErr;
            fn try_from(value: &BitFloat) -> Result<$t, Self::Error>{
                let start = value.m.len().saturating_sub(2);
                let sem = SEM{
                    s: value.s,
                    e: value.e,
                    m: SmallBuf::try_from(&value.m[start..])?,
                };
                sem.try_into()
            }
        }
        )+
    };
}
impl_try_from_bf!(f64, f32, i128, u128, i64, u64, i32, u32, i16, u16, i8, u8);

impl FromStr for BitFloat {
    type Err = FromStrErr;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        todo!()
    }
}

impl fmt::Display for BitFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl<T: Into<SEM>> PartialEq<T> for BitFloat {
    fn eq(&self, other: &T) -> bool {
        let sem = other.into();
        if self.s ^ sem.s || self.e != sem.e {
            return false;
        }
        self.m[..] == sem.m[..]
    }
}

impl PartialOrd for BitFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        if self.s ^ other.s {
            return Some(scmp(self.s, Greater));
        }
        let exp_ord = self.e.cmp(&other.e);
        if !exp_ord.is_eq() {
            return Some(scmp(self.s, exp_ord));
        }

        return Some(scmp(self.s, cmp_buf(&self.m, &other.m)));
    }
}

impl<T: Into<SEM>> PartialOrd<T> for BitFloat {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        let sem = other.into();
        if self.s ^ sem.s {
            return Some(scmp(self.s, Greater));
        }

        let exp_ord = self.e.cmp(&sem.e);
        if !exp_ord.is_eq() {
            return Some(scmp(self.s, exp_ord));
        }

        return Some(scmp(self.s, cmp_buf(&self.m, &other.m)));
    }
}

impl_commutative_peq_pord!(BitFloat, f64, f32, i128, u128, i64, u64, i32, u32, i16, u16, i8, u8);

impl Eq for BitFloat {}

impl Ord for BitFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

fn add_sub_bf(bf: &mut BitFloat, e: i128, m: &[u64], sub: bool) {
    if m.is_empty() {
        return;
    }
    if bf.m.is_empty() {
        bf.e = e;
        bf.m = Mantissa::make(m);
        bf.s = sub;
        return;
    }

    let del_m = (bf.e - bf.m.len() as i128) - (e - m.len() as i128);
    let del_p = e - bf.e;

    if del_m > 0 {
        bf.m.put_zeros(del_m as usize);
    }
    if del_p > 0 {
        bf.m.push_zeros(del_p as usize);
        bf.e = e;
    }

    let start = (-del_m).max(0) as usize;
    let comp = bf.s ^ sub;
    if acc(&mut bf.m[start..], m, comp as u8) {
        if comp {
            twos_comp(&mut bf.m);
        } else {
            bf.m.push(1);
            bf.e += 1
        }
        bf.s ^= comp;
    }
    if comp {
        if let Some(val) = bf.m.trim_zeros() {
            bf.e -= val as i128;
        } else {
            bf.e = i128::MIN;
        }
    }
}

impl Add for BitFloat {
    type Output = BitFloat;
    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        add_sub_bf(&mut lhs, rhs.e, &rhs.m, rhs.s);
        return lhs;
    }
}

impl Add for &BitFloat {
    type Output = BitFloat;
    fn add(self, rhs: Self) -> Self::Output {
        let mut lhs = self.clone();
        add_sub_bf(&mut lhs, rhs.e, &rhs.m, rhs.s);
        return lhs;
    }
}

impl Add<&BitFloat> for BitFloat {
    type Output = BitFloat;
    fn add(self, rhs: &BitFloat) -> Self::Output {
        let mut lhs = self;
        add_sub_bf(&mut lhs, rhs.e, &rhs.m, rhs.s);
        return lhs;
    }
}

impl Add<BitFloat> for &BitFloat {
    type Output = BitFloat;
    fn add(self, rhs: BitFloat) -> Self::Output {
        let mut lhs = self.clone();
        add_sub_bf(&mut lhs, rhs.e, &rhs.m, rhs.s);
        return lhs;
    }
}

impl<T: Into<SEM>> Add<T> for BitFloat {
    type Output = BitFloat;
    fn add(self, rhs: T) -> Self::Output {
        let sem = rhs.into();
        let mut lhs = self;
        add_sub_bf(&mut lhs, sem.e, &sem.m, sem.s);
        return lhs;
    }
}

impl<T: Into<SEM>> Add<T> for &BitFloat {
    type Output = BitFloat;
    fn add(self, rhs: T) -> Self::Output {
        let sem = rhs.into();
        let mut lhs = self.clone();
        add_sub_bf(&mut lhs, sem.e, &sem.m, sem.s);
        return lhs;
    }
}

impl_commutative!(
    Add,
    add,
    BitFloat,
    |x| x,
    f64,
    f32,
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

impl AddAssign for BitFloat {
    fn add_assign(&mut self, rhs: Self) {
        add_sub_bf(self, rhs.e, &rhs.m, rhs.s);
    }
}

impl AddAssign<&BitFloat> for BitFloat {
    fn add_assign(&mut self, rhs: &BitFloat) {
        add_sub_bf(self, rhs.e, &rhs.m, rhs.s);
    }
}

impl<T: Into<SEM>> AddAssign<T> for BitFloat {
    fn add_assign(&mut self, rhs: T) {
        let sem = rhs.into();
        add_sub_bf(self, rhs.e, &rhs.m, rhs.s);
    }
}

impl Neg for BitFloat {
    type Output = BitFloat;
    fn neg(self) -> Self::Output {
        let mut out = self;
        out.neg_mut();
        return out;
    }
}

impl Neg for &BitFloat {
    type Output = BitFloat;
    fn neg(self) -> Self::Output {
        let mut out = self.clone();
        out.neg_mut();
        return out;
    }
}

impl Sub for BitFloat {
    type Output = BitFloat;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut lhs = self;
        add_sub_bf(&mut lhs, rhs.e, &rhs.m, !rhs.s);
        return lhs;
    }
}

impl Sub for &BitFloat {
    type Output = BitFloat;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut lhs = self.clone();
        add_sub_bf(&mut lhs, rhs.e, &rhs.m, !rhs.s);
        return lhs;
    }
}

impl Sub<&BitFloat> for BitFloat {
    type Output = BitFloat;
    fn sub(self, rhs: &BitFloat) -> Self::Output {
        let mut lhs = self;
        add_sub_bf(&mut lhs, rhs.e, &rhs.m, !rhs.s);
        return lhs;
    }
}

impl Sub<BitFloat> for &BitFloat {
    type Output = BitFloat;
    fn sub(self, rhs: BitFloat) -> Self::Output {
        let mut lhs = self.clone();
        add_sub_bf(&mut lhs, rhs.e, &rhs.m, !rhs.s);
        return lhs;
    }
}

impl<T: Into<SEM>> Sub<T> for BitFloat {
    type Output = BitFloat;
    fn sub(self, rhs: T) -> Self::Output {
        let sem = rhs.into();
        let mut lhs = self;
        add_sub_bf(&mut lhs, sem.e, &sem.m, !sem.m);
        return lhs;
    }
}

impl<T: Into<SEM>> Sub<T> for &BitFloat {
    type Output = BitFloat;
    fn sub(self, rhs: T) -> Self::Output {
        let sem = rhs.into();
        let mut lhs = self.clone();
        add_sub_bf(&mut lhs, sem.e, &sem.m, !sem.m);
        return lhs;
    }
}

impl_commutative!(
    Sub,
    sub,
    BitFloat,
    |x| -x,
    f64,
    f32,
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

fn shl_shr_bf(bf: &mut BitFloat, sh: u128, s: bool) {
    let div = (sh / 64) as i128;
    let rem = (sh % 64) as u8;
    if s {
        bf.e -= div;
        let c = shr_buf(&mut bf.m, rem);
        if c > 0 {
            bf.m.put(c);
        }
        if bf.m[bf.m.len() - 1] == 0 {
            bf.m.pop();
            bf.e -= 1;
        }
    } else {
        bf.e += div;
        let c = shl_buf(&mut bf.m, rem);
        if c > 0 {
            bf.m.push(c);
            bf.m.e += 1
        }
        if bf.m[0] == 0 {
            bf.m.pull();
        }
    }
}

impl<T: Int> Shl<T> for BitFloat
where
    T::U: Into<u128>,
{
    type Output = BitFloat;
    fn shl(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        shl_shr_bf(&mut lhs, rhs.unsigned().into(), rhs.sign());
        return lhs;
    }
}

impl<T: Int> Shl<T> for &BitFloat
where
    T::U: Into<u128>,
{
    type Output = BitFloat;
    fn shl(self, rhs: T) -> Self::Output {
        let mut lhs = self.clone();
        shl_shr_bf(&mut lhs, rhs.unsigned().into(), rhs.sign());
        return lhs;
    }
}

impl<T: Int> ShlAssign<T> for BitFloat
where
    T::U: Into<u128>,
{
    fn shl_assign(&mut self, rhs: T) {
        shl_shr_bf(self, rhs.unsigned().into(), rhs.sign());
    }
}

impl<T: Int> Shr<T> for BitFloat
where
    T::U: Into<u128>,
{
    type Output = BitFloat;
    fn shr(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        shl_shr_bf(&mut lhs, rhs.unsigned().into(), !rhs.sign());
        return lhs;
    }
}

impl<T: Int> Shr<T> for &BitFloat
where
    T::U: Into<u128>,
{
    type Output = BitFloat;
    fn shr(self, rhs: T) -> Self::Output {
        let mut lhs = self.clone();
        shl_shr_bf(&mut lhs, rhs.unsigned().into(), !rhs.sign());
        return lhs;
    }
}

impl<T: Int> ShrAssign<T> for BitFloat
where
    T::U: Into<u128>,
{
    fn shr_assign(&mut self, rhs: T) {
        shl_shr_bf(self, rhs.unsigned().into(), !rhs.sign());
    }
}

fn mul_bf(lhs: &BitFloat, rhs: &BitFloat) -> BitFloat {
    let mut e = lhs.e + rhs.e;
    let p = lhs.m.len().min(rhs.m.len());
    let (buf, c) = short_mul_vec(&lhs.m, &rhs.m, p + 1);
    let mut m = Mantissa::make_take(buf);
    let mut gaurd = m.pull();
    if c > 0 {
        m.push(c);
        e += 1;
        if c >> 32 > 0 {
            gaurd = m.pull()
        }
    }
    if gaurd >> 63 == 1 {
        if inc(&mut m) {
            m.push(1);
            e += 1
        }
    }
    BitFloat {
        s: lhs.s ^ rhs.s,
        e,
        m,
    }
}

impl Mul for BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: Self) -> Self::Output {
        mul_bf(&self, &rhs)
    }
}

impl Mul for &BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: Self) -> Self::Output {
        mul_bf(self, rhs)
    }
}

impl Mul<&BitFloat> for BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: &BitFloat) -> Self::Output {
        mul_bf(&self, rhs)
    }
}

impl Mul<BitFloat> for &BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: BitFloat) -> Self::Output {
        mul_bf(self, &rhs)
    }
}

fn mul_prim2_bf(bf: &mut BitFloat, sem: SEM) {
    bf.s ^= sem.s;
    bf.e += sem.e;
    let c = mul_prim2(&mut bf.m, u128::from(sem.m));
    if c > 0 {
        bf.m.push(c as u64);
        let mut gaurd = bf.m.pull();
        let of = (c >> 64) as u64;
        if of > 0 {
            bf.m.push(of);
            bf.e += 1;
            if of >> 32 > 0 {
                gaurd = bf.m.pull();
            }
        }
        if gaurd >> 63 > 0 {
            if inc(&mut bf.m) {
                bf.m.push(1);
                bf.e += 1
            }
        }
    }
}

impl Mul<f64> for BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: f64) -> Self::Output {
        let mut lhs = self;
        let sem = SEM::from(rhs);
        mul_prim2_bf(&mut lhs, sem);
        return lhs;
    }
}

impl Mul<f64> for &BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: f64) -> Self::Output {
        let mut lhs = self.clone();
        let sem = SEM::from(rhs);
        mul_prim2_bf(&mut lhs, sem);
        return lhs;
    }
}

impl Mul<f32> for BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: f32) -> Self::Output {
        let mut lhs = self;
        let sem = SEM::from(rhs);
        mul_prim2_bf(&mut lhs, sem);
        return lhs;
    }
}

impl Mul<f32> for &BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: f32) -> Self::Output {
        let mut lhs = self.clone();
        let sem = SEM::from(rhs);
        mul_prim2_bf(&mut lhs, sem);
        return lhs;
    }
}

impl Mul<i128> for BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: i128) -> Self::Output {
        let mut lhs = self;
        let sem = SEM::from(rhs);
        mul_prim2_bf(&mut lhs, sem);
        return lhs;
    }
}

impl Mul<i128> for &BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: i128) -> Self::Output {
        let mut lhs = self.clone();
        let sem = SEM::from(rhs);
        mul_prim2_bf(&mut lhs, sem);
        return lhs;
    }
}

impl Mul<u128> for BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: u128) -> Self::Output {
        let mut lhs = self;
        let sem = SEM::from(rhs);
        mul_prim2_bf(&mut lhs, sem);
        return lhs;
    }
}

impl Mul<u128> for &BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: u128) -> Self::Output {
        let mut lhs = self.clone();
        let sem = SEM::from(rhs);
        mul_prim2_bf(&mut lhs, sem);
        return lhs;
    }
}

impl<T: Prim> Mul<T> for BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: T) -> Self::Output {
        let mut lhs = self;
        let c = mul_prim(&mut lhs.m, rhs.unsigned().into());
        if c > 0 {
            lhs.m.push(c);
            lhs.e += 1;
            if c >> 32 > 0 {
                if lhs.m.pull() >> 63 > 0 {
                    if inc(&mut lhs.m) {
                        lhs.m.push(1);
                        lhs.e += 1;
                    }
                }
            }
        }
        lhs.s ^= rhs.sign();
        return lhs;
    }
}

impl<T: Prim> Mul<T> for &BitFloat {
    type Output = BitFloat;
    fn mul(self, rhs: T) -> Self::Output {
        let mut lhs = self.clone();
        let c = mul_prim(&mut lhs.m, rhs.unsigned().into());
        if c > 0 {
            lhs.m.push(c);
            lhs.e += 1;
        }
        lhs.s ^= rhs.sign();
        return lhs;
    }
}

impl_commutative!(
    Mul,
    mul,
    BitFloat,
    |x| x,
    f64,
    f32,
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

impl MulAssign for BitFloat {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs)
    }
}

impl MulAssign<&BitFloat> for BitFloat {
    fn mul_assign(&mut self, rhs: &BitFloat) {
        *self = self.mul(rhs)
    }
}

impl MulAssign<f64> for BitFloat {
    fn mul_assign(&mut self, rhs: f64) {
        let sem = SEM::from(rhs);
        mul_prim2_bf(self, sem);
    }
}

impl MulAssign<f32> for BitFloat {
    fn mul_assign(&mut self, rhs: f32) {
        let sem = SEM::from(rhs);
        mul_prim2_bf(self, sem);
    }
}

impl MulAssign<i128> for BitFloat {
    fn mul_assign(&mut self, rhs: i128) {
        let sem = SEM::from(rhs);
        mul_prim2_bf(self, sem);
    }
}

impl MulAssign<u128> for BitFloat {
    fn mul_assign(&mut self, rhs: u128) {
        let sem = SEM::from(rhs);
        mul_prim2_bf(self, sem);
    }
}

impl<T: Prim> MulAssign<T> for BitFloat {
    fn mul_assign(&mut self, rhs: T) {
        let c = mul_prim(&mut self.m, rhs.unsigned().into());
        if c > 0 {
            self.m.push(c);
            self.e += 1;
        }
        self.s ^= rhs.sign();
    }
}

impl MulVariants for BitFloat {
    fn full_mul(&self, rhs: &Self) -> Self {
        let mut e = self.e + rhs.e;
        let (mut vec, c) = mul_vec(&self.m, &rhs.m);
        if c > 0 {
            vec.push(c);
            e += 1;
        }
        let m = Mantissa::make_take(vec);
        BitFloat {
            s: self.s ^ rhs.s,
            e,
            m,
        }
    }

    fn man_mul(&self, rhs: &Self, l: Option<usize>, r: Option<usize>, prec: Option<usize>) -> Self {
        let mut e = self.e + rhs.e;
        let l = l.unwrap_or(self.m.len());
        let r = r.unwrap_or(rhs.m.len());
        let (mut vec, c) =
            short_mul_vec(self.m.take(l), rhs.m.take(r), prec.unwrap_or(l + r - 1) + 1);
        let mut m = Mantissa::make_take(vec);
        let mut gaurd = m.pull();
        if c > 0 {
            m.push(c);
            e += 1;
            if c >> 32 > 0 {
                gaurd = m.pull();
            }
        }
        if gaurd >> 63 > 0 {
            if inc(&mut m) {
                m.push(1);
                m.pull();
                e += 1;
            }
        }
        BitFloat {
            s: self.s ^ rhs.s,
            e,
            m,
        }
    }
}

impl Sqr for BitFloat {
    fn sqr(&self) -> Self {
        let mut e = 2 * self.e;
        let (vec, c) = short_sqr_vec(&self.m, self.m.len() + 1);
        let mut m = Mantissa::make_take(vec);
        let mut gaurd = m.pull();
        if c > 0 {
            m.push(c);
            e += 1;
            if c >> 32 > 0 {
                gaurd = m.pull();
            }
        }
        if gaurd >> 63 == 1 {
            if inc(&mut m) {
                m.push(1);
                e += 1
            }
        }
        BitFloat { s: false, e, m }
    }
}

impl SqrVariants for BitFloat {
    fn full_sqr(&self) -> Self {
        let mut e = 2 * self.e;
        let (vec, c) = sqr_vec(&self.m);
        let mut m = Mantissa::make_take(vec);
        if c > 0 {
            m.push(c);
            e += 1;
        }
        BitFloat { s: false, e, m }
    }

    fn man_sqr(&self, in_prec: Option<usize>, out_prec: Option<usize>) -> Self {
        let mut e = 2 * self.e;
        let in_prec = in_prec.unwrap_or(self.m.len());
        let out_prec = out_prec.unwrap_or(2 * in_prec - 1);
        let (vec, c) = short_sqr_vec(&self.m.take(in_prec), out_prec);
        let mut m = Mantissa::make_take(vec);
        let mut gaurd = m.pull();
        if c > 0 {
            m.push(c);
            e += 1;
            if c >> 32 > 0 {
                gaurd = m.pull();
            }
        }
        if gaurd >> 63 > 0 {
            if inc(&mut m) {
                m.push(1);
                m.pull();
                e += 1;
            }
        }
        BitFloat { s: false, e, m }
    }
}

fn div_bf(mut n: BitFloat, mut d: BitFloat, p: usize) -> BitFloat {
    let nlen = n.m.len();
    let dlen = d.m.len();
    let dp = dlen.min(p);
    let np = nlen.min(dp + p);

    d.m.drain(dlen.saturating_sub(dp));
    n.m.drain(nlen.saturating_sub(np));
    n.m.put_zeros((dp + p).saturating_sub(np));

    let mut m = Mantissa::make_take(div_vec(&mut n.m, &mut d.m));
    let mut e = n.exp - d.exp + (nlen as i128 - dlen as i128) - p;

    let overflow = m[p] > 0;
    if overflow {
        e += 1;
    } else {
        m.pop();
    }

    let round_up = if overflow && m.top() >> 32 > 0 {
        m.pull() >> 63 > 0
    } else {
        let c = shl_buf(&mut n.m, 1);
        c > 0 || cmp_buf(&n.m, &d.m).is_ge()
    };

    if round_up {
        if inc(&mut m) {
            m.push(1);
            e += 1;
        }
    }

    BitFloat { s: n.s ^ d.s, e, m }
}

impl Div for BitFloat {
    type Output = BitFloat;
    fn div(self, rhs: Self) -> Self::Output {
        let p = self.m.len().min(rhs.m.len());
        return div_bf(self, rhs, p);
    }
}

impl Div for &BitFloat {
    type Output = BitFloat;
    fn div(self, rhs: Self) -> Self::Output {
        let p = self.m.len().min(rhs.m.len());
        return div_bf(self.clone(), rhs.clone(), p);
    }
}

impl Div<&BitFloat> for BitFloat {
    type Output = BitFloat;
    fn div(self, rhs: &BitFloat) -> Self::Output {
        let p = self.m.len().min(rhs.m.len());
        return div_bf(self, rhs.clone(), p);
    }
}

impl Div<BitFloat> for &BitFloat {
    type Output = BitFloat;
    fn div(self, rhs: BitFloat) -> Self::Output {
        let p = self.m.len().min(rhs.m.len());
        return div_bf(self.clone(), rhs, p);
    }
}

fn div_prim_bf(mut n: BitFloat, mut d: SEM) -> BitFloat {
    let nlen = n.m.len();
    let dlen = d.m.len();
    let sign = n.s ^ d.s;

    if dlen == 1 {
        n.m.put(0);
        let rem = div_prim(&mut n.m, d.m[0]);
        let mut e = n.e - d.e - 1;

        let overflow = n.m[nlen] > 0;
        if overflow {
            e += 1;
        } else {
            n.m.pop();
        }

        let round_up = if overflow && n.m.top() >> 32 > 0 {
            n.m.pull() >> 63 > 0
        } else {
            rem >= (d.m[0] >> 1) + (d.m[0] & 1)
        };

        if round_up {
            if inc(&mut n.m) {
                n.m.push(1);
                e += 1;
            }
        }

        BitFloat { s: sign, e, m: n.m }
    } else {
        let lz = d.m[1].leading_zeros() as u8;
        shl_buf(&mut d.m, lz);
        let mut of = shl_buf(&mut n.m, lz);
        n.m.put_zeros(2);

        let mut q = vec![0u64; nlen + 1];
        div_buf_of(&mut n.m, &mut of, &d.m, &mut q);

        let mut m = Mantissa::make_take(q);
        let mut e = n.e - d.e - 2;

        let overflow = m[nlen] > 0;
        if overflow {
            e += 1;
        } else {
            m.pop();
        }

        let div = u128::from(d.m);
        let rem = (n.m[0] as u128) | ((n.m[1] as u128) << 64);
        let round_up = if overflow && m.top() >> 32 > 0 {
            m.pull() >> 63 > 0
        } else {
            (rem >> 127 > 0) | (rem << 1 >= div)
        };

        if round_up {
            if inc(&mut m) {
                m.push(1);
                e += 1;
            }
        }

        BitFloat { s: sign, e, m }
    }
}

impl<T: Into<SEM>> Div<T> for BitFloat {
    type Output = BitFloat;
    fn div(self, rhs: T) -> Self::Output {
        div_prim_bf(self, rhs.into())
    }
}

impl<T: Into<SEM>> Div<T> for &BitFloat {
    type Output = BitFloat;
    fn div(self, rhs: T) -> Self::Output {
        div_prim_bf(self.clone(), rhs.into())
    }
}

macro_rules! impl_div_commute_bf {
    ($($t:ty), *) => {
        $(
        impl Div<BitFloat> for $t{
            type Output = BitFloat;
            fn div(self, rhs: BitFloat) -> Self::Output{
                let mut out = div_bf(BitFloat::from(self), rhs, rhs.m.len());
                out.m.trim_zeros();
                return out;
            }
        }

        impl Div<&BitFloat> for $t{
            type Output = BitFloat;
            fn div(self, rhs: &BitFloat) -> Self::Output{
                let mut out = div_bf(BitFloat::from(self), rhs.clone(), rhs.m.len());
                out.m.trim_zeros();
                return out;
            }
        }
        )*
    };
}
impl_div_commute_bf!(f64, f32, i128, u128, i64, u64, i32, u32, i16, u16, i8, u8);

impl DivAssign for BitFloat {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.div(rhs);
    }
}

impl DivAssign<&BitFloat> for BitFloat {
    fn div_assign(&mut self, rhs: &BitFloat) {
        *self = self.div(&rhs);
    }
}

impl<T: Into<SEM>> DivAssign<T> for BitFloat {
    fn div_assign(&mut self, rhs: T) {
        *self = self.div(&rhs);
    }
}

impl DivVariants for BitFloat {
    type Output = BitFloat;

    fn full_div(self, rhs: Self) -> Self::Output {
        let p = self.m.len().max(rhs.m.len());
        return div_bf(self, rhs, p);
    }

    fn man_div(
        self,
        rhs: Self,
        l: Option<usize>,
        r: Option<usize>,
        prec: Option<usize>,
    ) -> Self::Output {
        let l = l.unwrap_or(self.m.len());
        let r = r.unwrap_or(rhs.m.len());
        let p = p.unwrap_or(l - r);

        let mut n = self;
        lhs.m.drain(lhs.m.len() - l);

        let mut d = rhs;
        rhs.m.drain(rhs.m.len() - r);

        return div_bf(n, d, p);
    }
}

impl DivVariants for &BitFloat {
    type Output = BitFloat;

    fn full_div(self, rhs: Self) -> Self::Output {
        let p = self.m.len().max(rhs.m.len());
        return div_bf(self.clone(), rhs.clone(), p);
    }

    fn man_div(
        self,
        rhs: Self,
        l: Option<usize>,
        r: Option<usize>,
        prec: Option<usize>,
    ) -> Self::Output {
        let l = l.unwrap_or(self.m.len());
        let r = r.unwrap_or(rhs.m.len());
        let p = p.unwrap_or(l - r);

        let mut n = self.clone();
        lhs.m.drain(lhs.m.len() - l);

        let mut d = rhs.clone();
        rhs.m.drain(rhs.m.len() - r);

        return div_bf(n, d, p);
    }
}

impl DivVariants<&BitFloat> for BitFloat {
    type Output = BitFloat;

    fn full_div(self, rhs: &BitFloat) -> Self::Output {
        let p = self.m.len().max(rhs.m.len());
        return div_bf(self, rhs.clone(), p);
    }

    fn man_div(
        self,
        rhs: &BitFloat,
        l: Option<usize>,
        r: Option<usize>,
        prec: Option<usize>,
    ) -> Self::Output {
        let l = l.unwrap_or(self.m.len());
        let r = r.unwrap_or(rhs.m.len());
        let p = p.unwrap_or(l - r);

        let mut n = self;
        lhs.m.drain(lhs.m.len() - l);

        let mut d = rhs.clone();
        rhs.m.drain(rhs.m.len() - r);

        return div_bf(n, d, p);
    }
}

impl DivVariants<BitFloat> for &BitFloat {
    type Output = BitFloat;

    fn full_div(self, rhs: BitFloat) -> Self::Output {
        let p = self.m.len().max(rhs.m.len());
        return div_bf(self.clone(), rhs, p);
    }

    fn man_div(
        self,
        rhs: BitFloat,
        l: Option<usize>,
        r: Option<usize>,
        prec: Option<usize>,
    ) -> Self::Output {
        let l = l.unwrap_or(self.m.len());
        let r = r.unwrap_or(rhs.m.len());
        let p = p.unwrap_or(l - r);

        let mut n = self.clone();
        lhs.m.drain(lhs.m.len() - l);

        let mut d = rhs;
        rhs.m.drain(rhs.m.len() - r);

        return div_bf(n, d, p);
    }
}
