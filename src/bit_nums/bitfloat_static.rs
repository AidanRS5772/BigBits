use crate::bitfloat::BitFloat;
use crate::constants::*;
use crate::ubitint::{add_carry, add_with_carry, shl_carry, shr_carry, sub_carry};
use core::fmt;
use std::f64::consts::{LN_10, LN_2, PI};
use std::{cmp, i128, ops::*};

fn add_bfs<const N: usize>(lhs: &mut [usize; N], rhs: &[usize; N], sh: usize) -> bool {
    unsafe {
        let mut c: u8 = 0;
        for (l, r) in lhs.iter_mut().zip(&rhs[sh..]) {
            add_with_carry(l, *r, &mut c);
        }

        if sh > 0 {
            for l in &mut lhs[(N - sh)..] {
                if c == 0 {
                    break;
                }
                add_carry(l, &mut c);
            }
        }

        return c == 1;
    }
}

fn add_bfs_prim2<const N: usize>(bfs: &mut [usize; N], prim: u128, sh: usize) -> bool {
    unsafe {
        let mut c: u8 = 0;
        if sh + 2 <= N {
            add_with_carry(&mut bfs[N - sh - 2], prim as usize, &mut c);
        }

        if sh + 1 <= N {
            add_with_carry(&mut bfs[N - sh - 1], (prim >> 64) as usize, &mut c);
        }

        if sh > 0 {
            for l in bfs[N - sh..].iter_mut() {
                if c == 0 {
                    break;
                }
                add_carry(l, &mut c);
            }
        }

        return c == 1;
    }
}

fn add_bfs_prim<const N: usize>(bfs: &mut [usize; N], prim: usize, sh: usize) -> bool {
    unsafe {
        let mut c: u8 = 0;
        if sh + 1 <= N {
            add_with_carry(&mut bfs[N - sh - 1], prim, &mut c);
        }

        if sh > 0 {
            for l in bfs[N - sh..].iter_mut() {
                if c == 0 {
                    break;
                }
                add_carry(l, &mut c);
            }
        }

        return c == 1;
    }
}

fn sub_bfs<const N: usize>(lhs: &mut [usize; N], rhs: &[usize; N], sh: usize) {
    unsafe {
        let mut c: u8 = 1;
        for (l, r) in lhs.iter_mut().zip(&rhs[sh..]) {
            add_with_carry(l, !*r, &mut c);
        }

        if c == 0 {
            for l in &mut lhs[(N - sh)..] {
                sub_carry(l, &mut c);
                if c != 0 {
                    break;
                }
            }
        }
    }
}

fn sub_bfs_prim2<const N: usize>(lhs: &mut [usize; N], rhs: u128, sh: usize) {
    unsafe {
        let r = !rhs;
        let mut c: u8 = 1;
        if sh + 2 < N {
            add_with_carry(&mut lhs[N - sh - 2], r as usize, &mut c);
        }

        if sh + 1 < N {
            add_with_carry(&mut lhs[N - sh - 1], (r >> 64) as usize, &mut c);
        }

        if c == 0 && sh > 0 {
            for l in lhs[N - sh..].iter_mut() {
                sub_carry(l, &mut c);
                if c != 0 {
                    break;
                }
            }
        }
    }
}

fn sub_bfs_prim<const N: usize>(lhs: &mut [usize; N], rhs: usize, sh: usize) {
    unsafe {
        let r = !rhs;
        let mut c: u8 = 1;
        if sh + 1 < N {
            add_with_carry(&mut lhs[N - sh - 1], (r >> 64) as usize, &mut c);
        }

        if c == 0 && sh > 0 {
            for l in lhs[N - sh..].iter_mut() {
                sub_carry(l, &mut c);
                if c != 0 {
                    break;
                }
            }
        }
    }
}

fn mul_bfs<const N: usize>(a: &[usize], b: &[usize]) -> ([usize; N], usize) {
    let mut out = [0_usize; N];
    let pa = a.len();
    let pb = b.len();
    let mask = u64::MAX as u128;
    let mut carry: u128 = 0;

    let d_tot = pa + pb - 1;
    let i_start = N.saturating_sub(d_tot);

    for i in i_start..N {
        let mut term: u128 = carry;
        carry = 0;

        let d = i + d_tot - N;
        let j_min = d.saturating_sub(pb - 1);
        let j_max = d.min(pa - 1);

        for j in j_min..=j_max {
            let val = (a[j] as u128) * (b[d - j] as u128);
            term += val & mask;
            carry += val >> 64;
        }
        carry += term >> 64;
        out[i] = term as usize;
    }

    (out, carry as usize)
}

fn mul_bfs_prim2<const N: usize>(a: &mut [usize], b: u128) -> usize {
    let mask = u64::MAX as u128;
    let b0 = b & mask;
    let b1 = b >> 64;

    let vali = (a[0] as u128) * b0;
    let mut carry = vali >> 64;

    for i in 0..N - 1 {
        let mut term = carry;
        carry = 0;

        let val0 = (a[i + 1] as u128) * b0;
        term += val0 & mask;
        carry += val0 >> 64;

        let val1 = (a[i] as u128) * b1;
        term += val1 & mask;
        carry += val1 >> 64;

        carry += term >> 64;
        a[i] = term as usize;
    }

    let valf = (a[N - 1] as u128) * b1;
    let termf = (valf & mask) + carry;
    a[N - 1] = termf as usize;

    ((valf >> 64) + (termf >> 64)) as usize
}

fn mul_bfs_prim(a: &mut [usize], b: usize) -> usize {
    let b_u128 = b as u128;
    let mut carry: u128 = 0;
    for a in a.iter_mut() {
        let val = (*a as u128) * b_u128 + carry;
        *a = val as usize;
        carry = val >> 64;
    }

    return carry as usize;
}

fn sqr_bfs<const N: usize>(a: &[usize]) -> ([usize; N], usize) {
    let p = a.len();
    let mut out = [0_usize; N];
    let mask = u64::MAX as u128;
    let mut carry: u128 = 0;

    let d_tot = 2 * p - 1;
    let i_min = N.saturating_sub(d_tot);

    for i in i_min..N {
        let mut term: u128 = carry;
        carry = 0;

        let d = i + d_tot - N;
        let j_min = d.saturating_sub(p - 1);
        let j_max = d.min(p - 1);

        for j in j_min..=((d - 1) / 2).min(j_max) {
            let val = (a[j] as u128) * (a[d - j] as u128);
            term += (val << 1) & mask;
            carry += val >> 63;
        }

        if d % 2 == 0 {
            let j = d / 2;
            let val = (a[j] as u128) * (a[j] as u128);
            term += val & mask;
            carry += val >> 64;
        }

        carry += term >> 64;
        out[i] = term as usize;
    }

    (out, carry as usize)
}

fn shl_bfs<const N: usize>(m: &mut [usize], sh: u8) -> i128 {
    let mv_sz = 64 - sh;
    let mut carry = 0;
    unsafe {
        for elem in m.iter_mut() {
            shl_carry(elem, &mut carry, sh, mv_sz);
        }

        let mut out = 0;
        if carry > 0 {
            m.rotate_left(1);
            m[N - 1] = carry;
            out = 1;
        }
        return out;
    }
}

fn shr_bfs<const N: usize>(m: &mut [usize], sh: u8) -> i128 {
    let mv_sz = 64 - sh;
    let mut carry = 0;
    unsafe {
        for elem in m.iter_mut().rev() {
            shr_carry(elem, &mut carry, sh, mv_sz);
        }

        let mut out = 0;
        if m[N - 1] == 0 {
            m.rotate_right(1);
            m[0] = carry;
            out = -1;
        }
        return out;
    }
}

fn rcp_bfs<const N: usize>(val: &BitFloatStatic<N>, p: usize) -> BitFloatStatic<N> {
    let iter_cnt = ((p as f64) * 64.0 / 53.0).log(3.0) as usize + 1;

    let sh = val.ilog2() - 1;
    let norm_val = *val >> sh;

    let rcp_fval = 1.0 / norm_val.to::<f64>().unwrap();
    let mut x = BitFloatStatic::from(rcp_fval);
    let e = 1 - x.partial_mul_man(norm_val, Some(1), Some(2));
    let y = x.partial_mul_man(e, Some(1), Some(1));
    x += y + y.partial_mul_man(e, Some(1), Some(1));

    let mut p = 53;
    for _ in 0..iter_cnt {
        p *= 3;
        let acc = 1 + p / 64;

        let e = 1 - x.partial_mul_man(norm_val, Some(acc), Some(2 * acc));
        let y = x.partial_mul_man(e, Some(acc), Some(acc));
        x += y + y.partial_mul_man(e, Some(acc), Some(acc));
    }

    x <<= sh;
    return x;
}

fn powi_prim<const N: usize>(x: BitFloatStatic<N>, n: u128) -> BitFloatStatic<N> {
    if n == 0_u128 {
        return const { BitFloatStatic::<N>::one() };
    } else if n == 1_u128 {
        return x.clone();
    } else if n % 2 == 1 {
        x * powi_prim(x * x, n / 2)
    } else {
        return powi_prim(x * x, n / 2);
    }
}

fn sqrt_h<const N: usize>(val: &BitFloatStatic<N>, p: usize) -> Result<BitFloatStatic<N>, String> {
    if val.sign {
        return Err("Can't take square root of negative value".to_string());
    }
    if *val == BitFloatStatic::zero() {
        return Ok(*val);
    }

    let sqrt_3_8 = sqrt_3_8::<N>();
    let sqrt_25_6 = sqrt_25_6::<N>();
    let iter_cnt = ((p as f64) * 64.0 / 53.0).log(3.0) as usize + 2;

    let mut sh = val.ilog2();
    if sh % 2 == 1 {
        sh -= 1;
    }
    let norm_val = *val >> sh;

    let f = norm_val.to::<f64>().unwrap();
    let rcp_sqrt_f = 1.0 / f.sqrt();
    let mut x = BitFloatStatic::from(rcp_sqrt_f);

    let s = norm_val * sqrt_3_8;
    let mut y = x
        .partial_sqr_man(Some(2))
        .partial_mul_man(s, Some(2), Some(2));
    let tmp = 1.875 - y.partial_mul_man(sqrt_25_6 - y, Some(2), Some(2));
    x = x.partial_mul_man(tmp, Some(2), Some(2));

    let mut p: usize = 53;
    for _ in 0..iter_cnt {
        p *= 3;
        let acc = 2 + p / 64;

        y = x
            .partial_sqr_man(Some(acc))
            .partial_mul_man(s, Some(acc), Some(2 * acc));
        let tmp = 1.875 - y.partial_mul_man(sqrt_25_6 - y, Some(acc), Some(acc));
        x = x.partial_mul_man(tmp, Some(acc), Some(acc));
    }
    x *= norm_val;
    x <<= sh / 2;

    Ok(x)
}

fn simple_exp<const N: usize>(
    a: usize,
    b: usize,
    x: BitFloatStatic<N>,
) -> (BitFloatStatic<N>, BitFloatStatic<N>, BitFloatStatic<N>) {
    let one = const { BitFloatStatic::one() };
    let (mut p, mut q, mut t) = if a == 0 {
        (one, one, one)
    } else {
        (x, BitFloatStatic::make_int(a as i128), x)
    };

    for i in (a + 1)..b {
        p = p.partial_mul(x);
        q *= i;

        t *= i;
        t += p;
    }

    (p, q, t)
}

const BIN_SPLIT_TAIL_THRESH: usize = 8;

fn bin_split_exp<const N: usize>(
    a: usize,
    b: usize,
    x: BitFloatStatic<N>,
) -> (BitFloatStatic<N>, BitFloatStatic<N>, BitFloatStatic<N>) {
    if b - a <= BIN_SPLIT_TAIL_THRESH {
        return simple_exp(a, b, x);
    }

    let m = (a + b) / 2;
    let (pl, ql, tl) = bin_split_exp(a, m, x);
    let (pr, qr, tr) = bin_split_exp(m, b, x);

    let p = pl.partial_mul(pr);
    let q = ql.partial_mul(qr);
    let mut t = tl.partial_mul(qr);
    t += pl.partial_mul(tr);

    return (p, q, t);
}

pub const EXP_L0: usize = 50;

pub fn exp_param(p: usize) -> (usize, usize) {
    let l = EXP_L0 - p.ilog2() as usize;
    let log = (p as f64).ln() + LN_2 * (7 + l) as f64 - 1.0;
    let log_log = log.ln();

    let k = LN_2 * 64.0 * (p as f64) / (log - log_log + log_log / log);

    (l, k as usize + 1)
}

fn arg_reduce_exp<const N: usize>(
    val: &BitFloatStatic<N>,
) -> Result<(BitFloatStatic<N>, i128), String> {
    let rcp_ln2 = rcp_ln2_static();
    let ln2 = ln2_static();
    let n = if let Ok(v) = (*val * rcp_ln2).round().to::<i128>() {
        v
    } else {
        return Err("Absolute value to large to take exponent".to_string());
    };
    return Ok((*val - n * ln2, n));
}

pub fn exp_bfs<const N: usize>(mut x: BitFloatStatic<N>, p: usize) -> BitFloatStatic<N> {
    let (l, k) = exp_param(p);
    x >>= l;

    let (_, q, t) = bin_split_exp(0, k, x);
    let mut res = t / q;

    for _ in 0..l {
        res = res.sqr();
    }

    return res;
}

fn simple_sin<const N: usize>(
    a: usize,
    b: usize,
    x: BitFloatStatic<N>,
) -> (BitFloatStatic<N>, BitFloatStatic<N>, BitFloatStatic<N>) {
    let one = const { BitFloatStatic::one() };
    let (mut p, mut q, mut t) = if a == 0 {
        (one, one, one)
    } else {
        let q = BitFloatStatic::make_int((2 * a * (2 * a + 1)) as i128);
        (x, q, x)
    };

    for i in a + 1..b {
        let qi = 2 * i * (2 * i + 1);
        p = p.partial_mul(x);
        q *= qi;

        t *= qi;
        t += p;
    }

    return (p, q, t);
}

fn bin_split_sin<const N: usize>(
    a: usize,
    b: usize,
    x: BitFloatStatic<N>,
) -> (BitFloatStatic<N>, BitFloatStatic<N>, BitFloatStatic<N>) {
    if b - a < BIN_SPLIT_TAIL_THRESH {
        return simple_sin(a, b, x);
    }

    let mid = (a + b) / 2;
    let (pl, ql, tl) = bin_split_sin(a, mid, x);
    let (pr, qr, tr) = bin_split_sin(mid, b, x);

    let p = pl.partial_mul(pr);
    let q = ql.partial_mul(qr);

    let mut t = tl.partial_mul(qr);
    t += pl.partial_mul(tr);

    return (p, q, t);
}

const TRIG_L0: usize = 20;

fn sin_params(p: usize) -> (usize, usize) {
    let l = TRIG_L0 - p.ilog2() as usize;
    let log = (p as f64).ln() + ((l + 7) as f64) * LN_2 + LN_2.ln() - PI.ln() - 1.0;
    let log_log = log.ln();

    let k = 32.0 * LN_2 * (p as f64) / (log - log_log + log_log / log);

    return (l, k as usize + 1);
}

fn arg_reduce_trig<const N: usize>(x: &BitFloatStatic<N>) -> (BitFloatStatic<N>, bool) {
    let pi = pi();
    let rcp_pi = rcp_pi();
    let n = (*x * rcp_pi).round();
    let sign = if n.exp > 0 {
        false
    } else {
        n.m[N - 1] % 2 == 1
    };

    return (*x - pi.partial_mul(n), sign);
}

pub fn cos_sin_bfs<const N: usize>(
    mut x: BitFloatStatic<N>,
    p: usize,
) -> (BitFloatStatic<N>, BitFloatStatic<N>) {
    let (l, k) = sin_params(p);
    x >>= l;

    let (_, q, t) = bin_split_sin(0, k, -x.sqr());
    let mut s = x * t / q;
    let c2: BitFloatStatic<N> = 1 - s.sqr();
    let mut c = c2.sqrt().unwrap();

    for _ in 0..l {
        let new_c = c.sqr() - s.sqr();
        let new_s = (s * c) << 1_u128;
        c = new_c;
        s = new_s;
    }

    return (c, s);
}

fn ln<const N: usize>(val: &BitFloatStatic<N>, p: usize) -> Result<BitFloatStatic<N>, String> {
    if val.sign {
        return Err("Can't take Natural Logirithm of a Negetive Number".to_string());
    }
    if val.exp == i128::MIN {
        return Ok(const { BitFloatStatic::neg_inf() });
    }

    let m_i128 = 32 * p as i128 - val.ilog2();
    let m = m_i128.max(0) as u128;
    let x = *val << m;

    let s: BitFloatStatic<N> = 4 / x;
    let mut a: BitFloatStatic<N> = (1 + s) >> 1_u128;
    let mut b: BitFloatStatic<N> = s.sqrt_man(2).unwrap();

    let k = (cmp::max(a, b).ilog2() - (a - b).abs().ilog2()) as usize;
    let iter_cnt = (64.0 * (p as f64) / (k as f64)).log2() as usize + 2;

    for _ in 0..iter_cnt {
        let k = (cmp::max(a, b).ilog2() - (a - b).abs().ilog2()) as usize;
        let acc = 2 + k / 64;

        let new_a = (a + b) >> 1_u8;
        let new_b = a
            .partial_mul_man(b, Some(acc), Some(acc))
            .sqrt_man(acc)
            .unwrap();
        a = new_a;
        b = new_b;
    }

    return Ok(pi::<N>() / a - m * ln2_static::<N>());
}

fn arr_to_u128(m: &[usize]) -> u128 {
    return ((m[1] as u128) << 64) | (m[0] as u128);
}

fn u128_into_arr(val: u128, m: &mut [usize]) {
    m[0] = val as usize;
    m[1] = (val >> 64) as usize;
}

fn twos_comp(slice: &mut [usize]) -> u8 {
    unsafe {
        let mut c: u8 = 1;
        for s in slice {
            *s = !*s;
            add_carry(s, &mut c);
        }
        return c;
    }
}

trait SEM {
    fn get_sem(self) -> (bool, i128, u128);
    fn make_from_sem(sign: bool, exp: i128, m: u128) -> Result<Self, String>
    where
        Self: Sized;
}

impl SEM for f64 {
    fn get_sem(self) -> (bool, i128, u128) {
        let bits = self.to_bits();
        let sign = (bits >> 63) == 1;
        let exp2 = ((bits >> 52) & 0x7FF) as i128 - 1023;

        let exp = exp2.div_euclid(64);
        let sh = exp2.rem_euclid(64) as u128;

        let m = (((bits & 0xFFFFFFFFFFFFF) | 0x10000000000000) as u128) << (sh + 12);

        (sign, exp, m)
    }

    fn make_from_sem(sign: bool, exp_bfs: i128, m_bfs: u128) -> Result<Self, String>
    where
        Self: Sized,
    {
        if m_bfs == 0 {
            return Ok(f64::from_bits(0));
        }

        let lz = m_bfs.leading_zeros() as i128;
        let exp = 64 * exp_bfs + 63 - lz;

        if exp < -1022 || exp > 1023 {
            return Err(format!("Exponent: {} is outside range for f64", exp));
        }

        let m = ((m_bfs << (lz as u32 + 1)) >> 76) as u64;

        let mut bits: u64 = if sign { 1 } else { 0 };

        bits <<= 11;
        bits |= (exp + 1023) as u64;

        bits <<= 52;
        bits |= m;

        return Ok(f64::from_bits(bits));
    }
}

impl SEM for f32 {
    fn get_sem(self) -> (bool, i128, u128) {
        let bits = self.to_bits();
        let sign = (bits >> 31) == 1;
        let exp2 = ((bits >> 23) & 0xFF) as i128 - 127;

        let exp = exp2.div_euclid(64);
        let sh = exp2.rem_euclid(64) as u128;

        let m = (((bits & 0x7FFFFF) | 0x800000) as u128) << (sh + 41);

        (sign, exp, m)
    }

    fn make_from_sem(sign: bool, exp_bfs: i128, m_bfs: u128) -> Result<Self, String>
    where
        Self: Sized,
    {
        if m_bfs == 0 {
            return Ok(f32::from_bits(0));
        }

        let lz = m_bfs.leading_zeros() as i128;
        let exp = 64 * exp_bfs + 63 - lz;
        if exp < -126 || exp > 127 {
            return Err(format!("Exponent: {} is outside range for f64", exp));
        }
        let m = ((m_bfs << (lz + 1)) >> 105) as u32;

        let mut bits: u32 = if sign { 1 } else { 0 };

        bits <<= 8;
        bits |= (exp + 127) as u32;

        bits <<= 23;
        bits |= m;

        return Ok(f32::from_bits(bits));
    }
}

impl SEM for i128 {
    fn get_sem(self) -> (bool, i128, u128) {
        let sign = self < 0;
        let mut m = self.unsigned_abs();
        let exp;
        if (m >> 64) > 0 {
            exp = 1;
        } else {
            m <<= 64;
            exp = 0;
        }

        (sign, exp, m)
    }

    fn make_from_sem(sign: bool, exp: i128, m: u128) -> Result<Self, String>
    where
        Self: Sized,
    {
        if exp == 0 {
            let val = (m >> 64) as i128;
            return if sign { Ok(-val) } else { Ok(val) };
        } else if exp == 1 {
            let val = m as i128;
            return if sign { Ok(-val) } else { Ok(val) };
        } else if exp < 0 {
            return Ok(0);
        } else {
            return Err(format!("Exponent: {} out of bounds of a i128", exp));
        }
    }
}

impl SEM for u128 {
    fn get_sem(self) -> (bool, i128, u128) {
        let mut m = self;
        let exp;
        if (m >> 64) > 0 {
            exp = 1;
        } else {
            m <<= 64;
            exp = 0;
        }

        (false, exp, m)
    }

    fn make_from_sem(sign: bool, exp: i128, m: u128) -> Result<Self, String>
    where
        Self: Sized,
    {
        if sign {
            return Err(format!("Sign is wrong for u128"));
        }
        if exp == 0 {
            return Ok(m >> 64);
        } else if exp == 1 {
            return Ok(m);
        } else if exp < 0 {
            return Ok(0);
        } else {
            return Err(format!("Exponent: {} out of bounds of a u128", exp));
        }
    }
}

macro_rules! impl_sem_iprim {
    ($($t:ty),*) => {
        $(
            impl SEM for $t{
                fn get_sem(self) -> (bool, i128, u128){
                    let sign = self < 0;
                    (sign, 0, self.abs() as u128)
                }

                fn make_from_sem(sign: bool, exp: i128, m: u128) -> Result<Self, String> where Self: Sized{
                    if exp == 0{
                        let val = (m >> 64) as $t;
                        return if sign {Ok(-val)} else {Ok(val)}
                    }else if exp < -1{
                        return Ok(0)
                    }else{
                        return Err(format!("Exponent: {} is out of range for {}", exp, stringify!($t)))
                    }
                }
            }
        )*
    };
}

macro_rules! impl_sem_uprim {
    ($($t:ty),*) => {
        $(
            impl SEM for $t{
                fn get_sem(self) -> (bool, i128, u128){
                    (false, 0, self as u128)
                }

                fn make_from_sem(sign: bool, exp: i128, m: u128) -> Result<Self, String> where Self: Sized{
                    if exp == 0{
                        return Ok((m >> 64) as $t);
                    }else if exp < -1{
                        return Ok(0)
                    }else{
                        return Err(format!("Exponent: {} is out of range for {}", exp, stringify!($t)))
                    }
                }
            }
        )*
    };
}

impl_sem_iprim!(i64, isize, i32, i16, i8);
impl_sem_uprim!(u64, usize, u32, u16, u8);

#[derive(Debug, Clone, Copy)]
pub struct BitFloatStatic<const N: usize> {
    m: [usize; N],
    exp: i128,
    sign: bool,
}

impl<const N: usize> BitFloatStatic<N> {
    pub fn get_sign(&self) -> bool {
        self.sign
    }

    pub fn get_exp(&self) -> i128 {
        self.exp
    }

    pub fn get_m(&self) -> [usize; N] {
        self.m
    }

    pub const fn zero() -> Self {
        return BitFloatStatic {
            m: [0; N],
            exp: i128::MIN,
            sign: false,
        };
    }

    pub const fn one() -> Self {
        let mut m = [0; N];
        m[N - 1] = 1;
        return BitFloatStatic {
            m,
            exp: 1,
            sign: false,
        };
    }

    pub const fn inf() -> Self {
        return BitFloatStatic {
            m: [usize::MAX; N],
            exp: i128::MAX,
            sign: false,
        };
    }

    pub const fn neg_inf() -> Self {
        let mut out = Self::inf();
        out.sign ^= true;
        return out;
    }

    pub const fn make(m: [usize; N], exp: i128, sign: bool) -> Self {
        assert!(N > 2);
        BitFloatStatic { m: m, exp, sign }
    }

    pub const fn make_const_int<const I: i128>() -> Self {
        let mut m = [0; N];
        let mut exp = 0;
        let ui: u128 = I.unsigned_abs();
        if ui < usize::MAX as u128 {
            m[N - 1] = ui as usize;
        } else {
            m[N - 1] = (ui >> 64) as usize;
            m[N - 2] = ui as usize;
            exp += 1;
        }
        BitFloatStatic {
            m,
            exp,
            sign: I < 0,
        }
    }

    pub fn make_int(i: i128) -> Self {
        if i == 0 {
            return const { BitFloatStatic::zero() };
        }
        let mut m = [0; N];
        let mut exp = 0;
        let ui: u128 = i.unsigned_abs();
        if ui < usize::MAX as u128 {
            m[N - 1] = ui as usize;
        } else {
            m[N - 1] = (ui >> 64) as usize;
            m[N - 2] = ui as usize;
            exp += 1;
        }
        BitFloatStatic {
            m,
            exp,
            sign: i < 0,
        }
    }

    pub fn from<T: SEM>(val: T) -> Self {
        if N < 2 {
            panic!("Mantissa is not large enough")
        }
        let (sign, exp, m_u128) = val.get_sem();
        let mut m = [0; N];
        m[N - 2] = m_u128 as usize;
        m[N - 1] = (m_u128 >> 64) as usize;
        BitFloatStatic::<N> { m, exp, sign }
    }

    pub fn from_str(val: &str) -> Result<Self, String> {
        if let Some(idx) = val.find('.') {
            let ten = const { BitFloatStatic::make_const_int::<10>() };
            let mut mul = ten.powi(-((val.len() - idx - 1) as i128));
            let mut sum = const { BitFloatStatic::zero() };
            for c in val.chars().rev() {
                if let Some(d) = c.to_digit(10) {
                    sum += BitFloatStatic::make_int(d as i128) * mul;
                    mul *= ten;
                } else {
                    if c == '.' {
                        continue;
                    } else {
                        return Err("Malformed string as input".to_string());
                    }
                }
            }

            return Ok(sum);
        } else {
            return Err("Malformed string as input".to_string());
        }
    }

    pub fn to<T: SEM>(&self) -> Result<T, String> {
        T::make_from_sem(self.sign, self.exp, arr_to_u128(&self.m[N - 2..]))
    }

    pub fn to_bf(&self) -> BitFloat {
        let mut m = self.m.to_vec();
        m.reverse();
        if let Some(idx) = m.iter().rposition(|&x| x != 0) {
            m.truncate(idx + 1);
        }

        BitFloat {
            m: m,
            exp: self.exp,
            sign: self.sign,
        }
    }

    pub fn mut_abs(&mut self) {
        self.sign = false;
    }

    pub fn abs(&self) -> Self {
        let mut out = *self;
        out.sign = false;
        out
    }

    pub fn mut_neg(&mut self) {
        self.sign ^= true;
    }

    pub fn abs_floor(&self) -> Self {
        if self.exp < 0 {
            Self::zero()
        } else {
            let mut m = self.m;
            m[..N.saturating_sub(self.exp as usize + 1)].fill(0);
            Self {
                m,
                exp: self.exp,
                sign: self.sign,
            }
        }
    }

    pub fn floor(&self) -> Self {
        let mut out = self.abs_floor();

        if self.sign {
            out -= 1;
        }

        out
    }

    pub fn ceil(&self) -> Self {
        let mut out = self.abs_floor();

        if !self.sign {
            out += 1;
        }

        out
    }

    pub fn abs_frac(&self) -> Self {
        (*self - self.abs_floor()).abs()
    }

    pub fn frac(&self) -> Self {
        *self - self.floor()
    }

    pub fn round(&self) -> Self {
        if self.exp < 0 {
            return Self::zero();
        } else {
            let fract = self.abs_frac();
            if fract < 0.5 {
                return self.floor();
            } else {
                return self.ceil();
            }
        }
    }

    pub fn ilog2(&self) -> i128 {
        return 64 * self.exp + self.m[N - 1].ilog2() as i128;
    }

    pub fn sqr(&self) -> Self {
        let (mut m, carry) = sqr_bfs(&self.m);
        let mut exp = 2 * self.exp;
        if carry > 0 {
            m.rotate_left(1);
            m[N - 1] = carry;
            exp += 1;
        }

        return Self {
            m,
            exp,
            sign: false,
        };
    }

    pub fn partial_sqr(&self) -> Self {
        let p = match self.m.iter().position(|&x| x != 0) {
            Some(idx) => idx,
            None => return Self::zero(),
        };

        let (mut m, carry) = sqr_bfs(&self.m[p..]);
        let mut exp = 2 * self.exp;
        if carry > 0 {
            m.rotate_left(1);
            m[N - 1] = carry;
            exp += 1;
        }

        return Self {
            m,
            exp,
            sign: false,
        };
    }

    pub fn partial_sqr_man(&self, p_opt: Option<usize>) -> Self {
        let p = match p_opt {
            Some(value) => N - value,
            None => match self.m.iter().position(|&x| x != 0) {
                Some(idx) => idx,
                None => return Self::zero(),
            },
        };

        let (mut m, carry) = sqr_bfs(&self.m[p..]);
        let mut exp = 2 * self.exp;
        if carry > 0 {
            m.rotate_left(1);
            m[N - 1] = carry;
            exp += 1;
        }

        return Self {
            m,
            exp,
            sign: false,
        };
    }

    pub fn partial_mul(self, rhs: Self) -> Self {
        let idxa = match self.m.iter().position(|&x| x != 0) {
            Some(idx) => idx,
            None => return BitFloatStatic::zero(),
        };

        let idxb = match self.m.iter().position(|&x| x != 0) {
            Some(idx) => idx,
            None => return BitFloatStatic::zero(),
        };

        let mut exp = self.exp + rhs.exp;
        let (mut m, carry) = mul_bfs(&self.m[idxa..], &rhs.m[idxb..]);
        if carry > 0 {
            m.rotate_left(1);
            m[N - 1] = carry;
            exp += 1;
        }

        Self {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }

    pub fn partial_mul_man(self, rhs: Self, pa_opt: Option<usize>, pb_opt: Option<usize>) -> Self {
        let idxa = match pa_opt {
            Some(value) => N - value.min(N),
            None => match self.m.iter().position(|&x| x != 0) {
                Some(idx) => idx,
                None => return BitFloatStatic::zero(),
            },
        };

        let idxb = match pb_opt {
            Some(value) => N - value.min(N),
            None => match self.m.iter().position(|&x| x != 0) {
                Some(idx) => idx,
                None => return BitFloatStatic::zero(),
            },
        };

        let (mut m, carry) = mul_bfs(&self.m[idxa..], &rhs.m[idxb..]);
        let mut exp = self.exp + rhs.exp;
        if carry > 0 {
            m.rotate_left(1);
            m[N - 1] = carry;
            exp += 1;
        }

        BitFloatStatic {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }

    pub fn rcp(&self) -> Self {
        rcp_bfs(self, N)
    }

    pub fn rcp_man(&self, p: usize) -> Self {
        rcp_bfs(self, p)
    }

    pub fn sqrt(&self) -> Result<Self, String> {
        sqrt_h(self, N)
    }

    pub fn sqrt_man(&self, p: usize) -> Result<Self, String> {
        sqrt_h(self, p.min(N))
    }

    pub fn exp(&self) -> Result<Self, String> {
        let (mut x, n) = arg_reduce_exp(self)?;
        x = exp_bfs(x, N);
        if n > 0 {
            return Ok(x << n as u128);
        } else {
            return Ok(x >> n.unsigned_abs());
        }
    }

    pub fn exp_man(&self, p: usize) -> Result<Self, String> {
        let (mut x, n) = arg_reduce_exp(self)?;
        x = exp_bfs(x, p.min(N));
        if n > 0 {
            return Ok(x << n as u128);
        } else {
            return Ok(x >> n.unsigned_abs());
        }
    }

    pub fn cos_sin(&self) -> (Self, Self) {
        let (x, sign) = arg_reduce_trig(self);
        let (mut cos, mut sin) = cos_sin_bfs(x, N);
        if sign {
            cos.mut_neg();
            sin.mut_neg();
        }
        return (cos, sin);
    }

    pub fn cos_sin_man(&self, p: usize) -> (Self, Self) {
        let (x, sign) = arg_reduce_trig(self);
        let (mut cos, mut sin) = cos_sin_bfs(x, p.min(N));
        if sign {
            cos.mut_neg();
            sin.mut_neg();
        }
        return (cos, sin);
    }

    pub fn cos(&self) -> Self {
        let (x, sign) = arg_reduce_trig(self);
        let (mut cos, _) = cos_sin_bfs(x, N);
        if sign {
            cos.mut_neg();
        }
        return cos;
    }

    pub fn cos_man(&self, p: usize) -> Self {
        let (x, sign) = arg_reduce_trig(self);
        let (mut cos, _) = cos_sin_bfs(x, p.min(N));
        if sign {
            cos.mut_neg();
        }
        return cos;
    }

    pub fn sin(&self) -> Self {
        let (x, sign) = arg_reduce_trig(self);
        let (_, mut sin) = cos_sin_bfs(x, N);
        if sign {
            sin.mut_neg();
        }
        return sin;
    }

    pub fn sin_man(&self, p: usize) -> Self {
        let (x, sign) = arg_reduce_trig(self);
        let (_, mut sin) = cos_sin_bfs(x, p.min(N));
        if sign {
            sin.mut_neg();
        }
        return sin;
    }

    pub fn ln(&self) -> Result<Self, String> {
        ln(&self, N)
    }

    pub fn ln_man(&self, p: usize) -> Result<Self, String> {
        ln(&self, p.min(N))
    }
}

impl<const N: usize> fmt::Display for BitFloatStatic<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let exp10 = ((self.ilog2() as f64) * LN_2 / LN_10) as i128;

        let digit_cnt = (64.0 * (N as f64) * LN_2 / LN_10) as usize;
        let ten = const { BitFloatStatic::make_const_int::<10>() };
        let mut m10 = self.abs() / ten.powi(exp10);

        let d = m10.m[N - 1];
        let mut m = format!("{}.", d);
        m10 -= d;
        m10 *= 10;
        for _ in 0..digit_cnt {
            let d = m10.m[N - 1];
            let d_char = if d < 10 {
                (b'0' + d as u8) as char
            } else {
                return Err(fmt::Error);
            };
            m.push(d_char);
            m10 -= d;
            m10 *= 10;
        }

        if self.sign {
            return write!(f, "-{} E {}", m, exp10);
        } else {
            return write!(f, "{} E {}", m, exp10);
        }
    }
}

impl<const N: usize> PartialEq for BitFloatStatic<N> {
    fn eq(&self, other: &Self) -> bool {
        if (self.sign ^ other.sign) || (self.exp != other.exp) {
            return false;
        }

        return self.m == other.m;
    }
}

macro_rules! impl_partial_eq_fprim{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> PartialEq<$t> for BitFloatStatic<N> {
                fn eq(&self, other: &$t) -> bool {
                    let (other_sign, other_exp, other_m) = other.get_sem();

                    if (other_sign ^ self.sign) || (self.exp != other_exp) {
                        return false;
                    }

                    if arr_to_u128(&self.m[N-2..]) != other_m{
                        return false;
                    }

                    return self.m[..N-2].iter().all(|&x| x == 0);
                }
            }

            impl<const N: usize> PartialEq<BitFloatStatic<N>> for $t{
                fn eq(&self, other: &BitFloatStatic<N>) -> bool{
                    other == self
                }
            }
        )*
    };
}

impl_partial_eq_fprim!(f64, f32);

impl<const N: usize> PartialEq<i128> for BitFloatStatic<N> {
    fn eq(&self, other: &i128) -> bool {
        if self.sign ^ (*other < 0) {
            return false;
        }

        if self.exp != 1 || self.exp != 0 {
            return false;
        }

        let ou = other.unsigned_abs();

        if self.exp == 0 {
            if self.m[N - 1] != ou as usize || self.m[N - 2] != 0 {
                return false;
            }
        }

        if self.exp == 1 {
            if arr_to_u128(&self.m[N - 2..]) != ou {
                return false;
            }
        }

        return self.m[..N - 2].iter().all(|&x| x == 0);
    }
}

impl<const N: usize> PartialEq<BitFloatStatic<N>> for i128 {
    fn eq(&self, other: &BitFloatStatic<N>) -> bool {
        *other == *self
    }
}

impl<const N: usize> PartialEq<isize> for BitFloatStatic<N> {
    fn eq(&self, other: &isize) -> bool {
        if self.sign ^ (*other < 0) {
            return false;
        }

        if self.exp != 0 {
            return false;
        }

        if self.m[N - 1] != other.unsigned_abs() {
            return false;
        }

        return self.m[..N - 1].iter().all(|&x| x == 0);
    }
}

impl<const N: usize> PartialEq<BitFloatStatic<N>> for isize {
    fn eq(&self, other: &BitFloatStatic<N>) -> bool {
        *other == *self
    }
}

macro_rules! impl_patial_eq_iprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> PartialEq<$t> for BitFloatStatic<N>{
                fn eq(&self, other: &$t) -> bool{
                    *self == *other as isize
                }
            }

            impl<const N: usize> PartialEq<BitFloatStatic<N>> for $t{
                fn eq(&self, other: &BitFloatStatic<N>) -> bool{
                    *self as isize == *other
                }
            }
        )*
    };
}

impl_patial_eq_iprim!(i64, i32, i16, i8);

impl<const N: usize> PartialEq<u128> for BitFloatStatic<N> {
    fn eq(&self, other: &u128) -> bool {
        if self.exp != 1 || self.exp != 0 {
            return false;
        }

        if self.exp == 0 {
            if self.m[N - 1] != *other as usize || self.m[N - 2] != 0 {
                return false;
            }
        }

        if self.exp == 1 {
            if arr_to_u128(&self.m[N - 2..]) != *other {
                return false;
            }
        }

        return self.m[..N - 2].iter().all(|&x| x == 0);
    }
}

impl<const N: usize> PartialEq<BitFloatStatic<N>> for u128 {
    fn eq(&self, other: &BitFloatStatic<N>) -> bool {
        *other == *self
    }
}

impl<const N: usize> PartialEq<usize> for BitFloatStatic<N> {
    fn eq(&self, other: &usize) -> bool {
        if self.exp != 0 {
            return false;
        }

        if self.m[N - 1] != *other {
            return false;
        }

        return self.m[..N - 1].iter().all(|&x| x == 0);
    }
}

impl<const N: usize> PartialEq<BitFloatStatic<N>> for usize {
    fn eq(&self, other: &BitFloatStatic<N>) -> bool {
        *other == *self
    }
}

macro_rules! impl_patial_eq_uprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> PartialEq<$t> for BitFloatStatic<N>{
                fn eq(&self, other: &$t) -> bool{
                    *self == *other as isize
                }
            }

            impl<const N: usize> PartialEq<BitFloatStatic<N>> for $t{
                fn eq(&self, other: &BitFloatStatic<N>) -> bool{
                    *self as isize == *other
                }
            }
        )*
    };
}

impl_patial_eq_uprim!(u64, u32, u16, u8);

#[inline]
fn sign_cmp(sign: bool) -> cmp::Ordering {
    if sign {
        return cmp::Ordering::Less;
    } else {
        return cmp::Ordering::Greater;
    }
}

#[inline]
fn sign_cmp_w_ord(sign: bool, ord: cmp::Ordering) -> cmp::Ordering {
    if sign {
        return ord.reverse();
    } else {
        return ord;
    }
}

impl<const N: usize> PartialOrd for BitFloatStatic<N> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        if self.sign ^ other.sign {
            return Some(sign_cmp(self.sign));
        }

        let exp_ord = self.exp.cmp(&other.exp);
        if exp_ord != cmp::Ordering::Equal {
            return Some(sign_cmp_w_ord(self.sign, exp_ord));
        }

        let mut m_ord = cmp::Ordering::Equal;
        for (l, r) in self.m.iter().zip(&other.m).rev() {
            m_ord = l.cmp(r);
            if m_ord != cmp::Ordering::Equal {
                break;
            }
        }

        return Some(sign_cmp_w_ord(self.sign, m_ord));
    }
}

macro_rules! impl_partial_ord_fprim{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> PartialOrd<$t> for BitFloatStatic<N>{
                fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering>{
                    let (other_sign, other_exp, other_m) = other.get_sem();

                    if self.sign ^ other_sign{
                        return Some(sign_cmp(self.sign));
                    }

                    let exp_ord = self.exp.cmp(&other_exp);
                    if exp_ord != cmp::Ordering::Equal {
                        return Some(sign_cmp_w_ord(self.sign, exp_ord));
                    }

                    let m_ord = arr_to_u128(&self.m[N-2..]).cmp(&other_m);
                    if m_ord != cmp::Ordering::Equal{
                        return Some(sign_cmp_w_ord(self.sign, m_ord));
                    }

                    if self.m[..N-2].iter().all(|&x| x == 0){
                        return Some(cmp::Ordering::Equal);
                    } else {
                        return Some(sign_cmp(self.sign));
                    }
                }
            }

            impl<const N: usize> PartialOrd<BitFloatStatic<N>> for $t{
                fn partial_cmp(&self, other: &BitFloatStatic<N>) -> Option<cmp::Ordering>{
                    Some(other.partial_cmp(self)?.reverse())
                }
            }
        )*
    };
}

impl_partial_ord_fprim!(f64, f32);

impl<const N: usize> PartialOrd<i128> for BitFloatStatic<N> {
    fn partial_cmp(&self, other: &i128) -> Option<cmp::Ordering> {
        if self.sign ^ (*other < 0) {
            return Some(sign_cmp(self.sign));
        }

        let ou = other.unsigned_abs();
        let other_exp = if ou > usize::MAX as u128 {
            1
        } else if ou != 0 {
            0
        } else {
            i128::MIN
        };
        let exp_ord = self.exp.cmp(&other_exp);
        if exp_ord != cmp::Ordering::Equal {
            return Some(sign_cmp_w_ord(self.sign, exp_ord));
        }

        if self.exp == 1 {
            let ord1 = self.m[N - 1].cmp(&((ou >> 64) as usize));
            if ord1 != cmp::Ordering::Equal {
                return Some(sign_cmp_w_ord(self.sign, ord1));
            }

            let ord2 = self.m[N - 2].cmp(&(ou as usize));
            if ord2 != cmp::Ordering::Equal {
                return Some(sign_cmp_w_ord(self.sign, ord1));
            }

            if self.m[..N - 2].iter().all(|&x| x == 0) {
                return Some(cmp::Ordering::Equal);
            } else {
                return Some(sign_cmp(self.sign));
            }
        } else {
            let ord = self.m[N - 1].cmp(&(ou as usize));
            if ord != cmp::Ordering::Equal {
                return Some(sign_cmp_w_ord(self.sign, ord));
            }

            if self.m[..N - 1].iter().all(|&x| x == 0) {
                return Some(cmp::Ordering::Equal);
            } else {
                return Some(sign_cmp(self.sign));
            }
        }
    }
}

impl<const N: usize> PartialOrd<BitFloatStatic<N>> for i128 {
    fn partial_cmp(&self, other: &BitFloatStatic<N>) -> Option<cmp::Ordering> {
        Some(other.partial_cmp(self).unwrap().reverse())
    }
}

impl<const N: usize> PartialOrd<isize> for BitFloatStatic<N> {
    fn partial_cmp(&self, other: &isize) -> Option<cmp::Ordering> {
        if self.sign ^ (*other < 0) || self.exp > 0 {
            return Some(sign_cmp(self.sign));
        }

        let ou = other.unsigned_abs();
        let other_exp = if ou != 0 { 0 } else { i128::MIN };
        let exp_ord = self.exp.cmp(&other_exp);
        if exp_ord != cmp::Ordering::Equal {
            return Some(sign_cmp_w_ord(self.sign, exp_ord));
        }

        let ord = self.m[N - 1].cmp(&ou);
        if ord != cmp::Ordering::Equal {
            return Some(sign_cmp_w_ord(self.sign, ord));
        }

        if self.m[..N - 1].iter().all(|&x| x == 0) {
            return Some(cmp::Ordering::Equal);
        } else {
            return Some(sign_cmp(self.sign));
        }
    }
}

impl<const N: usize> PartialOrd<BitFloatStatic<N>> for isize {
    fn partial_cmp(&self, other: &BitFloatStatic<N>) -> Option<cmp::Ordering> {
        Some(other.partial_cmp(self).unwrap().reverse())
    }
}

macro_rules! impl_partial_ord_iprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> PartialOrd<$t> for BitFloatStatic<N>{
                fn partial_cmp(&self, other: &$t) -> Option<cmp::Ordering> {
                    self.partial_cmp(&(*other as isize))
                }
            }

            impl<const N: usize> PartialOrd<BitFloatStatic<N>> for $t{
                fn partial_cmp(&self, other: &BitFloatStatic<N>) -> Option<cmp::Ordering> {
                    (*self as isize).partial_cmp(other)
                }
            }
        )*
    };
}

impl_partial_ord_iprim!(i64, i32, i16, i8);

impl<const N: usize> PartialOrd<u128> for BitFloatStatic<N> {
    fn partial_cmp(&self, other: &u128) -> Option<cmp::Ordering> {
        if self.sign {
            return Some(cmp::Ordering::Less);
        }

        let other_exp = if *other > usize::MAX as u128 {
            1
        } else if *other == 0 {
            0
        } else {
            i128::MAX
        };
        let exp_ord = self.exp.cmp(&other_exp);
        if exp_ord != cmp::Ordering::Equal {
            return Some(exp_ord);
        }

        if self.exp == 1 {
            let ord1 = self.m[N - 1].cmp(&((*other >> 64) as usize));
            if ord1 != cmp::Ordering::Equal {
                return Some(sign_cmp_w_ord(self.sign, ord1));
            }

            let ord2 = self.m[N - 2].cmp(&(*other as usize));
            if ord2 != cmp::Ordering::Equal {
                return Some(sign_cmp_w_ord(self.sign, ord1));
            }

            if self.m[..N - 2].iter().all(|&x| x == 0) {
                return Some(cmp::Ordering::Equal);
            } else {
                return Some(sign_cmp(self.sign));
            }
        } else {
            let ord = self.m[N - 1].cmp(&(*other as usize));
            if ord != cmp::Ordering::Equal {
                return Some(sign_cmp_w_ord(self.sign, ord));
            }

            if self.m[..N - 1].iter().all(|&x| x == 0) {
                return Some(cmp::Ordering::Equal);
            } else {
                return Some(sign_cmp(self.sign));
            }
        }
    }
}

impl<const N: usize> PartialOrd<BitFloatStatic<N>> for u128 {
    fn partial_cmp(&self, other: &BitFloatStatic<N>) -> Option<cmp::Ordering> {
        Some(other.partial_cmp(self).unwrap().reverse())
    }
}

impl<const N: usize> PartialOrd<usize> for BitFloatStatic<N> {
    fn partial_cmp(&self, other: &usize) -> Option<cmp::Ordering> {
        let other_exp = if *other != 0 { 0 } else { i128::MIN };
        let exp_ord = self.exp.cmp(&other_exp);
        if exp_ord != cmp::Ordering::Equal {
            return Some(sign_cmp_w_ord(self.sign, exp_ord));
        }

        let ord = self.m[N - 1].cmp(other);
        if ord != cmp::Ordering::Equal {
            return Some(sign_cmp_w_ord(self.sign, ord));
        }

        if self.m[..N - 1].iter().all(|&x| x == 0) {
            return Some(cmp::Ordering::Equal);
        } else {
            return Some(sign_cmp(self.sign));
        }
    }
}

impl<const N: usize> PartialOrd<BitFloatStatic<N>> for usize {
    fn partial_cmp(&self, other: &BitFloatStatic<N>) -> Option<cmp::Ordering> {
        Some(other.partial_cmp(self).unwrap().reverse())
    }
}

macro_rules! impl_partial_ord_uprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> PartialOrd<$t> for BitFloatStatic<N>{
                fn partial_cmp(&self, other: &$t) -> Option<cmp::Ordering> {
                    self.partial_cmp(&(*other as isize))
                }
            }

            impl<const N: usize> PartialOrd<BitFloatStatic<N>> for $t{
                fn partial_cmp(&self, other: &BitFloatStatic<N>) -> Option<cmp::Ordering> {
                    (*self as isize).partial_cmp(other)
                }
            }
        )*
    };
}

impl_partial_ord_uprim!(u64, u32, u16, u8);

impl<const N: usize> Eq for BitFloatStatic<N> {}

impl<const N: usize> Ord for BitFloatStatic<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<const N: usize> Neg for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn neg(self) -> Self::Output {
        return BitFloatStatic {
            m: self.m,
            exp: self.exp,
            sign: !self.sign,
        };
    }
}

pub fn abs_cmp<const N: usize>(
    lhs: &BitFloatStatic<N>,
    rhs: &BitFloatStatic<N>,
) -> std::cmp::Ordering {
    let exp_cmp = lhs.exp.cmp(&rhs.exp);
    if exp_cmp != std::cmp::Ordering::Equal {
        return exp_cmp;
    }

    for (l, r) in lhs.m.iter().zip(rhs.m.iter()) {
        let m_ord = l.cmp(r);
        if m_ord != std::cmp::Ordering::Equal {
            return m_ord;
        }
    }

    return std::cmp::Ordering::Equal;
}

fn add_sub<const N: usize>(
    lhs: &BitFloatStatic<N>,
    rhs: &BitFloatStatic<N>,
    sub: bool,
) -> BitFloatStatic<N> {
    let sh = (lhs.exp - rhs.exp).unsigned_abs();

    if sh >= N as u128 {
        if lhs.exp > rhs.exp {
            return *lhs;
        } else {
            return if sub { -*rhs } else { *rhs };
        }
    }

    if lhs.sign ^ rhs.sign ^ sub {
        let (mut m, mut exp, r, sign) = match abs_cmp(&lhs, &rhs) {
            std::cmp::Ordering::Greater => (lhs.m, lhs.exp, &rhs.m, lhs.sign),
            std::cmp::Ordering::Less => (rhs.m, rhs.exp, &lhs.m, rhs.sign ^ sub),
            std::cmp::Ordering::Equal => {
                return BitFloatStatic::zero();
            }
        };

        sub_bfs::<N>(&mut m, r, sh as usize);

        if let Some(idx) = m.iter().rposition(|&x| x != 0) {
            m.rotate_right(N - idx - 1);
            exp -= (N - idx - 1) as i128;
        }

        return BitFloatStatic::<N> { m, exp, sign };
    } else {
        let (mut m, mut exp, r) = if lhs.exp > rhs.exp {
            (lhs.m, lhs.exp, &rhs.m)
        } else {
            (rhs.m, rhs.exp, &lhs.m)
        };

        let carry = add_bfs::<N>(&mut m, r, sh as usize);
        if carry {
            m.rotate_left(1);
            m[N - 1] = 1;
            exp += 1;
        }
        return BitFloatStatic::<N> {
            m,
            exp,
            sign: lhs.sign,
        };
    }
}

impl<const N: usize> Add for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn add(self, rhs: Self) -> Self::Output {
        add_sub(&self, &rhs, false)
    }
}

fn add_sub_prim2<const N: usize, T: SEM>(
    lhs: &BitFloatStatic<N>,
    rhs: T,
    sub: bool,
) -> BitFloatStatic<N> {
    let (mut rhs_sign, rhs_exp, rhs_m) = rhs.get_sem();
    rhs_sign ^= sub;
    let sh = (lhs.exp - rhs_exp).unsigned_abs();

    if sh >= N as u128 {
        if lhs.exp > rhs_exp {
            return *lhs;
        } else {
            let mut m = [0; N];
            u128_into_arr(rhs_m, &mut m[N - 2..]);
            return BitFloatStatic {
                m,
                exp: rhs_exp,
                sign: rhs_sign,
            };
        }
    }

    if lhs.exp > rhs_exp {
        if lhs.sign ^ rhs_sign {
            let mut m = lhs.m;
            sub_bfs_prim2(&mut m, rhs_m, sh as usize);

            let mut exp = lhs.exp;
            if let Some(idx) = m.iter().rposition(|&x| x != 0) {
                m.rotate_right(N - idx - 1);
                exp -= (N - idx - 1) as i128;
            }

            return BitFloatStatic {
                m,
                exp,
                sign: lhs.sign,
            };
        } else {
            let mut m = lhs.m;
            let mut exp = lhs.exp;
            if add_bfs_prim2(&mut m, rhs_m, sh as usize) {
                m.rotate_left(1);
                exp += 1;
            }

            return BitFloatStatic {
                m,
                exp,
                sign: lhs.sign,
            };
        }
    }

    let mut m = [0; N];
    u128_into_arr(rhs_m, &mut m);
    let rbfs = BitFloatStatic {
        m,
        exp: rhs_exp,
        sign: rhs_sign,
    };
    *lhs + rbfs
}

macro_rules! impl_add_prim2 {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Add<$t> for BitFloatStatic<N> {
                type Output = BitFloatStatic<N>;

                fn add(self, rhs: $t) -> Self::Output {
                    add_sub_prim2(&self, rhs, false)
                }
            }

            impl<const N: usize> Add<BitFloatStatic<N>> for $t{
                type Output = BitFloatStatic<N>;

                fn add(self, rhs: BitFloatStatic<N>) -> Self::Output {
                    rhs + self
                }
            }
        )*
    };
}

impl_add_prim2!(f64, f32, i128, u128);

enum IUSize {
    ISize(isize),
    USize(usize),
}

fn add_sub_prim<const N: usize>(
    lhs: &BitFloatStatic<N>,
    rhs: IUSize,
    sub: bool,
) -> BitFloatStatic<N> {
    let rhs_sign = match rhs {
        IUSize::ISize(r) => (r < 0) ^ sub,
        IUSize::USize(_) => sub,
    };
    let rhs_m = match rhs {
        IUSize::ISize(r) => r.unsigned_abs(),
        IUSize::USize(r) => r,
    };
    let sh = lhs.exp.unsigned_abs();

    if sh >= N as u128 {
        if lhs.exp > 0 {
            return *lhs;
        } else {
            let mut m = [0; N];
            m[N - 1] = rhs_m;
            return BitFloatStatic {
                m,
                exp: 0,
                sign: rhs_sign,
            };
        }
    }

    if sh > 0 {
        if lhs.sign ^ rhs_sign {
            let mut m = lhs.m;
            sub_bfs_prim(&mut m, rhs_m, sh as usize);

            let mut exp = lhs.exp;
            if let Some(idx) = m.iter().rposition(|&x| x != 0) {
                m.rotate_right(N - idx - 1);
                exp -= (N - idx - 1) as i128;
            }

            return BitFloatStatic {
                m,
                exp,
                sign: lhs.sign,
            };
        } else {
            let mut m = lhs.m;
            let mut exp = lhs.exp;
            if add_bfs_prim(&mut m, rhs_m, sh as usize) {
                m.rotate_left(1);
                exp += 1;
            }

            return BitFloatStatic {
                m,
                exp,
                sign: lhs.sign,
            };
        }
    }

    let mut m = [0; N];
    m[N - 1] = rhs_m;
    let rbfs = BitFloatStatic {
        m,
        exp: 0,
        sign: rhs_sign,
    };
    *lhs + rbfs
}

impl<const N: usize> Add<isize> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn add(self, rhs: isize) -> Self::Output {
        add_sub_prim(&self, IUSize::ISize(rhs), false)
    }
}

impl<const N: usize> Add<usize> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn add(self, rhs: usize) -> Self::Output {
        add_sub_prim(&self, IUSize::USize(rhs), false)
    }
}

macro_rules! impl_add_iprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Add<$t> for BitFloatStatic<N> {
                type Output = BitFloatStatic<N>;

                fn add(self, rhs: $t) -> Self::Output {
                    add_sub_prim(&self, IUSize::ISize(rhs as isize), false)
                }
            }

            impl<const N: usize> Add<BitFloatStatic<N>> for $t{
                type Output = BitFloatStatic<N>;

                fn add(self, rhs: BitFloatStatic<N>) -> Self::Output {
                    rhs + self
                }
            }
        )*
    };
}

impl_add_iprim!(i64, i32, i16, i8);

macro_rules! impl_add_uprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Add<$t> for BitFloatStatic<N> {
                type Output = BitFloatStatic<N>;

                fn add(self, rhs: $t) -> Self::Output {
                    add_sub_prim(&self, IUSize::USize(rhs as usize), false)
                }
            }

            impl<const N: usize> Add<BitFloatStatic<N>> for $t{
                type Output = BitFloatStatic<N>;

                fn add(self, rhs: BitFloatStatic<N>) -> Self::Output {
                    rhs + self
                }
            }
        )*
    };
}

impl_add_uprim!(u64, u32, u16, u8);

fn add_sub_mut<const N: usize>(lhs: &mut BitFloatStatic<N>, rhs: &BitFloatStatic<N>, sub: bool) {
    let sh = (lhs.exp - rhs.exp).unsigned_abs();

    if sh >= N as u128 {
        if lhs.exp > rhs.exp {
            return;
        } else {
            if sub {
                *lhs = -*rhs;
            } else {
                *lhs = *rhs
            }
            return;
        }
    }

    if lhs.sign ^ rhs.sign ^ sub {
        match abs_cmp(&lhs, &rhs) {
            std::cmp::Ordering::Greater => {
                sub_bfs(&mut lhs.m, &rhs.m, sh as usize);
            }
            std::cmp::Ordering::Less => {
                let mut m = rhs.m;
                sub_bfs(&mut m, &lhs.m, sh as usize);
                lhs.m = m;
                lhs.exp = rhs.exp;
                lhs.sign = rhs.sign ^ sub;
            }
            std::cmp::Ordering::Equal => {
                *lhs = BitFloatStatic::zero();
                return;
            }
        }

        if let Some(idx) = lhs.m.iter().rposition(|&x| x != 0) {
            lhs.m.rotate_right(N - idx - 1);
            lhs.exp -= (N - idx - 1) as i128;
        }
    } else {
        let carry = if lhs.exp > rhs.exp {
            add_bfs(&mut lhs.m, &rhs.m, sh as usize)
        } else {
            let mut m = rhs.m;
            let carry = add_bfs(&mut m, &lhs.m, sh as usize);
            lhs.m = m;
            lhs.exp = rhs.exp;
            carry
        };

        if carry {
            lhs.m.rotate_left(1);
            lhs.m[N - 1] = 1;
            lhs.exp += 1;
        }
    }
}

impl<const N: usize> AddAssign for BitFloatStatic<N> {
    fn add_assign(&mut self, rhs: Self) {
        add_sub_mut(self, &rhs, false);
    }
}

fn add_sub_mut_prim2<const N: usize, T: SEM>(lhs: &mut BitFloatStatic<N>, rhs: T, sub: bool) {
    let (rhs_sign, rhs_exp, rhs_m) = rhs.get_sem();
    let sh = (lhs.exp - rhs_exp).unsigned_abs();

    if sh >= N as u128 {
        if lhs.exp > rhs_exp {
            return;
        } else {
            let mut m = [0; N];
            u128_into_arr(rhs_m, &mut m[N - 2..]);
            *lhs = BitFloatStatic {
                m,
                exp: rhs_exp,
                sign: rhs_sign,
            };
        }
    }

    if lhs.exp > rhs_exp {
        if lhs.sign ^ rhs_sign ^ sub {
            sub_bfs_prim2(&mut lhs.m, rhs_m, sh as usize);
            if let Some(idx) = lhs.m.iter().rposition(|&x| x != 0) {
                lhs.m.rotate_right(N - idx - 1);
                lhs.exp -= (N - idx - 1) as i128;
            }
        } else {
            if add_bfs_prim2(&mut lhs.m, rhs_m, sh as usize) {
                lhs.m.rotate_left(1);
                lhs.exp += 1;
            }
        }
    } else {
        let mut m = [0; N];
        u128_into_arr(rhs_m, &mut m);
        let rbfs = BitFloatStatic {
            m,
            exp: rhs_exp,
            sign: rhs_sign,
        };
        *lhs += rbfs;
    }
}

macro_rules! impl_add_assign_prim2 {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> AddAssign<$t> for BitFloatStatic<N> {
                fn add_assign(&mut self, rhs: $t){
                    add_sub_mut_prim2(self, rhs, false);
                }
            }
        )*
    };
}

impl_add_assign_prim2!(f64, f32, i128, u128);

fn add_sub_mut_prim<const N: usize>(lhs: &mut BitFloatStatic<N>, rhs: IUSize, sub: bool) {
    let rhs_sign = match rhs {
        IUSize::ISize(r) => (r < 0) ^ sub,
        IUSize::USize(_) => sub,
    };
    let rhs_m = match rhs {
        IUSize::ISize(r) => r.unsigned_abs(),
        IUSize::USize(r) => r,
    };
    let sh = lhs.exp.unsigned_abs();

    if sh >= N as u128 {
        if lhs.exp > 0 {
            return;
        } else {
            let mut m = [0; N];
            m[N - 1] = rhs_m;
            *lhs = BitFloatStatic {
                m,
                exp: 0,
                sign: rhs_sign,
            };
        }
    }

    if sh > 0 {
        if lhs.sign ^ rhs_sign ^ sub {
            sub_bfs_prim(&mut lhs.m, rhs_m, sh as usize);
            if let Some(idx) = lhs.m.iter().rposition(|&x| x != 0) {
                lhs.m.rotate_right(N - idx - 1);
                lhs.exp -= (N - idx - 1) as i128;
            }
        } else {
            if add_bfs_prim(&mut lhs.m, rhs_m, sh as usize) {
                lhs.m.rotate_left(1);
                lhs.exp += 1;
            }
        }
    } else {
        let mut m = [0; N];
        m[N - 1] = rhs_m;
        let rbfs = BitFloatStatic {
            m,
            exp: 0,
            sign: rhs_sign,
        };
        *lhs += rbfs;
    }
}

macro_rules! impl_add_assign_iprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> AddAssign<$t> for BitFloatStatic<N> {
                fn add_assign(&mut self, rhs: $t){
                    add_sub_mut_prim(self, IUSize::ISize(rhs as isize), false);
                }
            }
        )*
    };
}

impl_add_assign_iprim!(i64, i32, i16, i8);

macro_rules! impl_add_assign_uprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> AddAssign<$t> for BitFloatStatic<N> {
                fn add_assign(&mut self, rhs: $t){
                    add_sub_mut_prim(self, IUSize::USize(rhs as usize), false);
                }
            }
        )*
    };
}

impl_add_assign_uprim!(u64, u32, u16, u8);

impl<const N: usize> Sub for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        add_sub(&self, &rhs, true)
    }
}

macro_rules! impl_sub_prim2{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Sub<$t> for BitFloatStatic<N> {
                type Output = BitFloatStatic<N>;

                fn sub(self, rhs: $t) -> Self::Output {
                    add_sub_prim2(&self, rhs, true)
                }
            }

            impl<const N: usize> Sub<BitFloatStatic<N>> for $t{
                type Output = BitFloatStatic<N>;

                fn sub(self, rhs: BitFloatStatic<N>) -> Self::Output {
                    rhs - self
                }
            }
        )*
    };
}

impl_sub_prim2!(f64, f32, i128, u128);

impl<const N: usize> Sub<isize> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn sub(self, rhs: isize) -> Self::Output {
        add_sub_prim(&self, IUSize::ISize(rhs), true)
    }
}

impl<const N: usize> Sub<usize> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn sub(self, rhs: usize) -> Self::Output {
        add_sub_prim(&self, IUSize::USize(rhs), true)
    }
}

macro_rules! impl_sub_iprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Sub<$t> for BitFloatStatic<N> {
                type Output = BitFloatStatic<N>;

                fn sub(self, rhs: $t) -> Self::Output {
                    add_sub_prim(&self, IUSize::ISize(rhs as isize), true)
                }
            }

            impl<const N: usize> Sub<BitFloatStatic<N>> for $t{
                type Output = BitFloatStatic<N>;

                fn sub(self, rhs: BitFloatStatic<N>) -> Self::Output {
                    rhs - self
                }
            }
        )*
    };
}

impl_sub_iprim!(i64, i32, i16, i8);

macro_rules! impl_sub_uprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Sub<$t> for BitFloatStatic<N> {
                type Output = BitFloatStatic<N>;

                fn sub(self, rhs: $t) -> Self::Output {
                    add_sub_prim(&self, IUSize::USize(rhs as usize), true)
                }
            }

            impl<const N: usize> Sub<BitFloatStatic<N>> for $t{
                type Output = BitFloatStatic<N>;

                fn sub(self, rhs: BitFloatStatic<N>) -> Self::Output {
                    rhs - self
                }
            }
        )*
    };
}

impl_sub_uprim!(u64, u32, u16, u8);

impl<const N: usize> SubAssign for BitFloatStatic<N> {
    fn sub_assign(&mut self, rhs: Self) {
        add_sub_mut(self, &rhs, true);
    }
}

macro_rules! impl_sub_assign_prim2{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> SubAssign<$t> for BitFloatStatic<N> {

                fn sub_assign(&mut self, rhs: $t){
                    add_sub_mut_prim2(self, rhs, true);
                }
            }
        )*
    };
}

impl_sub_assign_prim2!(f64, f32, i128, u128);

impl<const N: usize> SubAssign<isize> for BitFloatStatic<N> {
    fn sub_assign(&mut self, rhs: isize) {
        add_sub_mut_prim(self, IUSize::ISize(rhs), true);
    }
}

macro_rules! impl_sub_assign_iprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> SubAssign<$t> for BitFloatStatic<N> {
                fn sub_assign(&mut self, rhs: $t){
                    add_sub_mut_prim(self, IUSize::ISize(rhs as isize), true);
                }
            }
        )*
    };
}

impl_sub_assign_iprim!(i64, i32, i16, i8);

impl<const N: usize> SubAssign<usize> for BitFloatStatic<N> {
    fn sub_assign(&mut self, rhs: usize) {
        add_sub_mut_prim(self, IUSize::USize(rhs), true);
    }
}

macro_rules! impl_sub_assign_uprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> SubAssign<$t> for BitFloatStatic<N> {
                fn sub_assign(&mut self, rhs: $t){
                    add_sub_mut_prim(self, IUSize::USize(rhs as usize), true);
                }
            }
        )*
    };
}

impl_sub_assign_uprim!(u64, u32, u16, u8);

impl<const N: usize> Mul for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn mul(self, rhs: Self) -> Self::Output {
        let (mut m, carry) = mul_bfs(&self.m, &rhs.m);

        let mut exp = self.exp.saturating_add(rhs.exp);

        if carry > 0 {
            m.rotate_left(1);
            m[N - 1] = carry;
            exp += 1;
        }

        BitFloatStatic {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

macro_rules! impl_mul_prim2{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Mul<$t> for BitFloatStatic<N> {
                type Output = BitFloatStatic<N>;

                fn mul(self, rhs: $t) -> Self::Output {
                    let (rhs_sign, rhs_exp, rhs_m) = rhs.get_sem();
                    let mut m = self.m;
                    let carry = mul_bfs_prim2::<N>(&mut m, rhs_m);
                    let mut exp = self.exp.saturating_add(rhs_exp);

                    if carry > 0 {
                        m.rotate_left(1);
                        m[N - 1] = carry;
                        exp += 1;
                    }

                    BitFloatStatic {
                        m,
                        exp,
                        sign: self.sign ^ rhs_sign,
                    }
                }
            }

            impl<const N: usize> Mul<BitFloatStatic<N>> for $t {
                type Output = BitFloatStatic<N>;

                fn mul(self, rhs: BitFloatStatic<N>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}

impl_mul_prim2!(f64, f32, i128, u128);

impl<const N: usize> Mul<isize> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn mul(self, rhs: isize) -> Self::Output {
        let mut m = self.m;
        let carry = mul_bfs_prim(&mut m, rhs.unsigned_abs());
        let mut exp = self.exp;

        if carry > 0 {
            m.rotate_left(1);
            m[N - 1] = carry;
            exp += 1;
        }

        BitFloatStatic {
            m,
            exp,
            sign: self.sign ^ (rhs < 0),
        }
    }
}

impl<const N: usize> Mul<BitFloatStatic<N>> for isize {
    type Output = BitFloatStatic<N>;

    fn mul(self, rhs: BitFloatStatic<N>) -> Self::Output {
        rhs * self
    }
}

impl<const N: usize> Mul<usize> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn mul(self, rhs: usize) -> Self::Output {
        let mut m = self.m;
        let carry = mul_bfs_prim(&mut m, rhs);
        let mut exp = self.exp;

        if carry > 0 {
            m.rotate_left(1);
            m[N - 1] = carry;
            exp += 1;
        }

        BitFloatStatic {
            m,
            exp,
            sign: self.sign,
        }
    }
}

impl<const N: usize> Mul<BitFloatStatic<N>> for usize {
    type Output = BitFloatStatic<N>;

    fn mul(self, rhs: BitFloatStatic<N>) -> Self::Output {
        rhs * self
    }
}

macro_rules! impl_mul_iprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Mul<$t> for BitFloatStatic<N>{
                type Output = BitFloatStatic<N>;

                fn mul(self, rhs: $t) -> Self::Output{
                    self * (rhs as isize)
                }
            }

            impl<const N: usize> Mul<BitFloatStatic<N>> for $t{
                type Output = BitFloatStatic<N>;

                fn mul(self, rhs: BitFloatStatic<N>) -> Self::Output{
                    (self as isize) * rhs
                }
            }
        )*
    };
}

impl_mul_iprim!(i64, i32, i16, i8);

macro_rules! impl_mul_uprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Mul<$t> for BitFloatStatic<N>{
                type Output = BitFloatStatic<N>;

                fn mul(self, rhs: $t) -> Self::Output{
                    self * (rhs as usize)
                }
            }

            impl<const N: usize> Mul<BitFloatStatic<N>> for $t{
                type Output = BitFloatStatic<N>;

                fn mul(self, rhs: BitFloatStatic<N>) -> Self::Output{
                    (self as usize) * rhs
                }
            }
        )*
    };
}

impl_mul_uprim!(u64, u32, u16, u8);

impl<const N: usize> MulAssign for BitFloatStatic<N> {
    fn mul_assign(&mut self, rhs: Self) {
        let (m, carry) = mul_bfs::<N>(&self.m, &rhs.m);
        self.m = m;
        self.exp += rhs.exp;

        if carry > 0 {
            self.m.rotate_left(1);
            self.m[N - 1] = carry;
            self.exp += 1;
        }

        self.sign ^= rhs.sign;
    }
}

macro_rules! impl_mul_assign_prim2{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> MulAssign<$t> for BitFloatStatic<N> {
                fn mul_assign(&mut self, rhs: $t){
                    let (rhs_sign, rhs_exp, rhs_m) = rhs.get_sem();
                    let carry = mul_bfs_prim2::<N>(&mut self.m, rhs_m);
                    self.exp += rhs_exp;
                    self.sign ^= rhs_sign;

                    if carry > 0 {
                        self.m.rotate_left(1);
                        self.m[N - 1] = carry;
                        self.exp += 1;
                    }
                }
            }
        )*
    };
}

impl_mul_assign_prim2!(f64, f32, i128, u128);

impl<const N: usize> MulAssign<isize> for BitFloatStatic<N> {
    fn mul_assign(&mut self, rhs: isize) {
        let carry = mul_bfs_prim(&mut self.m, rhs.unsigned_abs());
        self.sign ^= rhs < 0;

        if carry > 0 {
            self.m.rotate_left(1);
            self.m[N - 1] = carry;
            self.exp += 1;
        }
    }
}

impl<const N: usize> MulAssign<usize> for BitFloatStatic<N> {
    fn mul_assign(&mut self, rhs: usize) {
        let carry = mul_bfs_prim(&mut self.m, rhs);
        self.sign ^= rhs < 0;

        if carry > 0 {
            self.m.rotate_left(1);
            self.m[N - 1] = carry;
            self.exp += 1;
        }
    }
}

macro_rules! impl_mul_assign_iprim{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> MulAssign<$t> for BitFloatStatic<N> {
                fn mul_assign(&mut self, rhs: $t){
                    *self *= rhs as isize;
                }
            }
        )*
    };
}

impl_mul_assign_iprim!(i64, i32, i16, i8);

macro_rules! impl_mul_assign_uprim{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> MulAssign<$t> for BitFloatStatic<N> {
                fn mul_assign(&mut self, rhs: $t){
                    *self *= rhs as usize;
                }
            }
        )*
    };
}

impl_mul_assign_uprim!(u64, u32, u16, u8);

macro_rules! impl_shl_shr_uprim{
    ($($t:ty),*) => {
        $(
        impl<const N: usize> Shl<$t> for BitFloatStatic<N> {
            type Output = BitFloatStatic<N>;

            fn shl(self, rhs: $t) -> Self::Output {
                let mut out = self;
                let div = rhs / 64;
                let rem = (rhs % 64) as u8;

                out.exp += div as i128;
                out.exp += shl_bfs::<N>(&mut out.m, rem);

                out
            }
        }

        impl<const N: usize> ShlAssign<$t> for BitFloatStatic<N> {
            fn shl_assign(&mut self, rhs: $t) {
                let div = rhs / 64;
                let rem = (rhs % 64) as u8;
                self.exp += div as i128;
                self.exp += shl_bfs::<N>(&mut self.m, rem);
            }
        }

        impl<const N: usize> Shr<$t> for BitFloatStatic<N> {
            type Output = BitFloatStatic<N>;

            fn shr(self, rhs: $t) -> Self::Output {
                let mut out = self;
                let div = rhs / 64;
                let rem = (rhs % 64) as u8;

                out.exp += div as i128;
                out.exp += shr_bfs::<N>(&mut out.m, rem);

                out
            }
        }

        impl<const N: usize> ShrAssign<$t> for BitFloatStatic<N> {
            fn shr_assign(&mut self, rhs: $t) {
                let div = rhs / 64;
                let rem = (rhs % 64) as u8;
                self.exp += div as i128;
                self.exp += shr_bfs::<N>(&mut self.m, rem);
            }
        }
        )*
    };
}

impl_shl_shr_uprim!(u128, u64, usize, u32, u16, u8);

macro_rules! impl_shl_shr_iprim{
    ($($t:ty),*) => {
        $(
        impl<const N: usize> Shl<$t> for BitFloatStatic<N> {
            type Output = BitFloatStatic<N>;

            fn shl(self, rhs: $t) -> Self::Output {
                return if rhs > 0{
                    self << rhs.unsigned_abs()
                }else{
                    self << rhs.unsigned_abs()
                }
            }
        }

        impl<const N: usize> ShlAssign<$t> for BitFloatStatic<N> {
            fn shl_assign(&mut self, rhs: $t) {
                if rhs > 0{
                    *self <<= rhs.unsigned_abs();
                }else{
                    *self >>= rhs.unsigned_abs();
                }
            }
        }

        impl<const N: usize> Shr<$t> for BitFloatStatic<N> {
            type Output = BitFloatStatic<N>;

            fn shr(self, rhs: $t) -> Self::Output {
                return if rhs > 0{
                    self >> rhs.unsigned_abs()
                }else{
                    self >> rhs.unsigned_abs()
                }
            }
        }

        impl<const N: usize> ShrAssign<$t> for BitFloatStatic<N> {
            fn shr_assign(&mut self, rhs: $t) {
                if rhs > 0{
                    *self >>= rhs.unsigned_abs();
                }else{
                    *self <<= rhs.unsigned_abs();
                }
            }
        }
        )*
    };
}

impl_shl_shr_iprim!(i128, i64, isize, i32, i16, i8);

impl<const N: usize> Div for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.rcp()
    }
}

macro_rules! impl_div_prim{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Div<$t> for BitFloatStatic<N> {
                type Output = BitFloatStatic<N>;

                fn div(self, rhs: $t) -> Self::Output {
                    self * BitFloatStatic::<N>::from(rhs).rcp()
                }
            }

            impl<const N: usize> Div<BitFloatStatic<N>> for $t {
                type Output = BitFloatStatic<N>;

                fn div(self, rhs: BitFloatStatic<N>) -> Self::Output {
                    rhs * BitFloatStatic::<N>::from(self).rcp()
                }
            }
        )*
    };
}

impl_div_prim!(f64, f32, i128, isize, i64, i32, i16, i8, u128, usize, u64, u32, u16, u8);

impl<const N: usize> DivAssign for BitFloatStatic<N> {
    fn div_assign(&mut self, rhs: Self) {
        *self *= rhs.rcp();
    }
}

macro_rules! impl_div_assign_fprim{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> DivAssign<$t> for BitFloatStatic<N> {
                fn div_assign(&mut self, rhs: $t) {
                    *self *= BitFloatStatic::<N>::from(rhs).rcp()
                }
            }
        )*
    };
}

impl_div_assign_fprim!(f64, f32, i128, isize, i64, i32, i16, i8, u128, usize, u64, u32, u16, u8);

pub trait PowI<RHS = Self> {
    type Output;
    fn powi(self, n: RHS) -> Self::Output;
}

impl<const N: usize> PowI<i128> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn powi(self, rhs: i128) -> Self::Output {
        let uval = powi_prim(self, rhs.unsigned_abs());
        if rhs > 0 {
            return uval;
        } else {
            return BitFloatStatic::one() / uval;
        }
    }
}

macro_rules! impl_powi_iprim{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> PowI<$t> for BitFloatStatic<N> {
                type Output = BitFloatStatic<N>;

                fn powi(self, rhs: $t) -> Self::Output {
                    self.powi(rhs as i128)
                }
            }
        )*
    };
}

impl_powi_iprim!(i64, isize, i32, i16, i8);

impl<const N: usize> PowI<u128> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn powi(self, rhs: u128) -> Self::Output {
        powi_prim(self, rhs)
    }
}

macro_rules! impl_powi_uprim{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> PowI<$t> for BitFloatStatic<N> {
                type Output = BitFloatStatic<N>;

                fn powi(self, rhs: $t) -> Self::Output {
                    self.powi(rhs as u128)
                }
            }
        )*
    };
}

impl_powi_uprim!(u64, usize, u32, u16, u8);

pub trait PowF<RHS = Self> {
    type Output;
    fn powf(self, n: RHS) -> Result<Self::Output, String>;
}

impl<const N: usize> PowF for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn powf(self, n: Self) -> Result<Self::Output, String> {
        let ln_self = self
            .ln()
            .map_err(|_| "Can't take floating Point power of a negetive number.".to_string())?;
        (n * ln_self).exp()
    }
}

macro_rules! impl_powf_fprim{
    ($($t:ty),*) => {
        $(
            impl<const N: usize> PowF<$t> for BitFloatStatic<N>{
                type Output = BitFloatStatic<N>;

                fn powf(self, n: $t) -> Result<Self::Output, String> {
                    let ln_self = self.ln().map_err(|_| "Can't take floating Point power of a negetive number.".to_string())?;
                    (n*ln_self).exp()
                }
            }

            impl<const N: usize> PowF<BitFloatStatic<N>> for $t{
                type Output = BitFloatStatic<N>;

                fn powf(self, n: BitFloatStatic<N>) -> Result<Self::Output, String> {
                    (n*self.ln()).exp()
                }
            }
        )*
    };
}

impl_powf_fprim!(f64, f32);
