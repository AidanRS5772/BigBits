use crate::ubitint::*;
use crate::ubitint_static::*;
use std::ops::*;

fn twos_comp(bis: &mut [usize]) -> bool {
    one_comp(bis);

    unsafe {
        let mut c: u8 = 1;
        for l in bis {
            #[cfg(target_arch = "aarch64")]
            add_carry_aarch64(l, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_carry_x86_64(l, &mut c);
            if c == 0 {
                return false;
            }
        }
    }

    return true;
}

#[derive(Debug, Clone, Copy)]
pub struct BitIntStatic<const N: usize> {
    data: [usize; N],
    sign: bool,
}

macro_rules! impl_to_usize_arr_iprim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> ToUsizeArray<N> for $t{
                #[inline]
                fn to_usize_array(self) -> [usize;N]{
                    self.unsigned_abs().to_usize_array()
                }
            }
        )*
    };
}

impl_to_usize_arr_iprim!(i128, i64, isize, i32, i16, i8);

impl<const N: usize> BitIntStatic<N> {
    #[inline]
    pub fn get_data(&self) -> [usize; N] {
        self.data
    }

    #[inline]
    pub fn get_sign(&self) -> bool {
        self.sign
    }

    #[inline]
    pub fn make(data: [usize; N], sign: bool) -> BitIntStatic<N> {
        BitIntStatic::<N> { data, sign }
    }

    #[inline]
    pub fn from<T: ToUsizeArray<N> + std::cmp::PartialOrd + Default + std::marker::Copy>(
        val: T,
    ) -> BitIntStatic<N> {
        BitIntStatic::<N> {
            data: val.to_usize_array(),
            sign: val < T::default(),
        }
    }

    #[inline]
    pub fn neg(&mut self) {
        self.sign ^= true;
    }

    #[inline]
    pub fn abs(&mut self) {
        self.sign = false;
    }

    #[inline]
    pub fn unsighned_abs(&self) -> UBitIntStatic<N> {
        UBitIntStatic::<N>::make(self.data)
    }

    #[inline]
    pub fn to(&self) -> Result<i128, String> {
        let uval: u128 = self.unsighned_abs().to()?;
        let val_res: Result<i128, _> = uval.try_into();
        match val_res {
            Ok(val) => {
                if self.sign {
                    Ok(-val)
                } else {
                    Ok(val)
                }
            }
            Err(_) => Err("Static BitInt cant be converted to a i128".to_string()),
        }
    }
}

impl<const N: usize> PartialEq for BitIntStatic<N> {
    fn eq(&self, other: &Self) -> bool {
        if self.sign != other.sign {
            return false;
        }

        for (l, r) in self.data.iter().zip(&other.data) {
            if l != r {
                return false;
            }
        }

        return true;
    }
}

impl<const N: usize> Add for BitIntStatic<N> {
    type Output = BitIntStatic<N>;

    fn add(self, rhs: Self) -> Self::Output {
        let (mut lhs_d, mut lhs_s) = (self.data, self.sign);
        if lhs_s {
            lhs_s ^= twos_comp(&mut lhs_d)
        }
        let (mut rhs_d, mut rhs_s) = (rhs.data, rhs.sign);
        if rhs_s {
            rhs_s ^= twos_comp(&mut rhs_d);
        }

        let sign = lhs_s ^ rhs_s ^ add_ubis::<N>(&mut lhs_d, &rhs_d, 0);

        BitIntStatic::<N> { data: lhs_d, sign }
    }
}
