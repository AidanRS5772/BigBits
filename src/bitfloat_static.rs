use crate::bitfloat::MES;
use crate::ubitint::*;
// use std::ops::*;

const DEFAULT_SIZE: usize = 3;

pub fn add_bfs<const N:usize>(a: &mut [usize], b: &[usize], sh: usize) -> i128 {
    let mut c: u8 = 0;
    unsafe {
        for (a_elem, &b_elem) in a.iter_mut().zip(&b[sh..]) {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(a_elem, b_elem, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(a_elem, b_elem, &mut c);
        }

        if c != 0 {
            for a_elem in &mut a[N - sh..] {
                if c == 0 {
                    break;
                }

                #[cfg(target_arch = "aarch64")]
                add_carry_aarch64(a_elem, &mut c);

                #[cfg(target_arch = "x86_64")]
                add_carry_x86_64(a_elem, &mut c);
            }
        }
    }

    return c as i128;
}

pub fn sub_bfs<const N: usize>(a: &mut [usize], b: &[usize], sh: usize){
    unsafe {
        let mut c: u8 = 1;
        for (a_elem, &b_elem) in a.iter_mut().zip(&b[sh..]) {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(a_elem, !b_elem, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(a_elem, !b_elem, &mut c);
        }

        if c == 0 {
            let a_len = a.len();
            for a_elem in &mut a[a_len - sh..] {
                #[cfg(target_arch = "aarch64")]
                sub_carry_aarch64(a_elem, &mut c);

                #[cfg(target_arch = "x86_64")]
                sub_carry_x86_64(a_elem, &mut c);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitFloatStatic<const N: usize = DEFAULT_SIZE> {
    m: [usize; N],
    exp: i128,
    sign: bool,
}

impl<const N: usize> BitFloatStatic<N> {
    pub fn get_m(&self) -> [usize; N] {
        self.m
    }

    pub fn get_exp(&self) -> i128 {
        self.exp
    }

    pub fn get_sign(&self) -> bool {
        self.sign
    }

    pub fn from<T: MES>(val: T) -> BitFloatStatic<N> {
        let (m_vec, exp_isz, sign) = val.get_mes();

        let mut m = [0_usize; N];
        for (m_elem, &vec_elem) in m.iter_mut().rev().zip(m_vec.iter().rev()) {
            *m_elem = vec_elem;
        }

        BitFloatStatic::<N> {
            m,
            exp: exp_isz as i128,
            sign,
        }
    }

    pub fn to(&self) -> Result<f64, String> {
        let exp: i32 = if let Ok(exp) = self.exp.try_into() {
            exp
        } else {
            return Err("BitFloat is too large or to small for f64".to_string());
        };

        let base = 2_f64.powi(64);
        let mut sc = 2_f64.powi(64 * exp);
        let mut out = sc * self.m[N - 1] as f64;
        if out.is_infinite() {
            return Err("BitFloat is to large for f64".to_string());
        }

        for &e in self.m[..(N - 1)].iter().rev() {
            sc /= base;
            out += sc * e as f64;
        }

        if self.sign {
            return Ok(-out);
        } else {
            return Ok(out);
        }
    }

    pub fn abs_cmp(&self, other: &BitFloatStatic<N>) -> std::cmp::Ordering{
        use std::cmp::Ordering::*;
        let exp_ord = self.exp.cmp(&other.exp);
        if !exp_ord.is_eq(){
            return exp_ord
        }

        let mut ord = Equal;
        for i in 1..N{
            ord = self.m[N-i].cmp(&other.m[N-i]);
            if !ord.is_eq(){
                return ord;
            }
        }

        return ord;
    }
}

impl<const N: usize> PartialOrd for BitFloatStatic<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        if self.sign ^ other.sign {
            if self.sign {
                return Some(Less);
            } else {
                return Some(Greater);
            }
        }

        let exp_ord = self.exp.cmp(&other.exp);
        if !exp_ord.is_eq() {
            if self.sign {
                return Some(exp_ord.reverse());
            } else {
                return Some(exp_ord);
            }
        }

        let mut ord = Equal;
        for i in 1..N {
            ord = self.m[N-i].cmp(&other.m[N-i]);
            if !ord.is_eq() {
                break;
            }
        }

        if self.sign {
            return Some(ord.reverse());
        } else {
            return Some(ord);
        }
    }
}

impl<const N: usize> Ord for BitFloatStatic<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        return self.partial_cmp(other).unwrap();
    }
}

impl<const N:usize> Add for BitFloatStatic<N>{
    type Output = BitFloatStatic<N>;

    fn add(self, rhs: Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut m, b, mut exp, sign) = match self.abs_cmp(&rhs){
            Greater => (self.m, rhs.m, self.exp, self.sign),
            Less => (rhs.m, self.m, rhs.exp, rhs.sign),
            Equal => (self.m, rhs.m, self.exp, self.sign)
        };

        let sh = (self.exp - rhs.exp).unsigned_abs();




    }
}
