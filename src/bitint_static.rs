use crate::ubitint_static::Pow;
use crate::ubitint::*;
use crate::ubitint_static::*;
use std::ops::*;
use std::fmt;

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
    pub fn from_str(num: &str) -> Result<BitIntStatic<N>, String>{
        let sign:bool;
        let unum:&str;
        if let Some(first_char) = num.chars().next() {
            if first_char == '-'{
                sign = true;
                unum = &num[1..];
            }else{
                sign = false;
                unum = num;
            }
        } else {
            return Err("malformed string input".to_string())
        }

        return Ok(BitIntStatic::<N> { data: UBitIntStatic::<N>::from_str(unum)?.get_data(), sign})
    }

    #[inline]
    pub fn mut_neg(&mut self) {
        self.sign ^= true;
    }

    #[inline]
    pub fn abs(&mut self) {
        self.sign = false;
    }

    #[inline]
    pub fn mod2_abs(&self) -> bool{
        (self.data[0] & 1_usize) == 1
    }

    #[inline]
    pub fn unsigned_abs(&self) -> UBitIntStatic<N> {
        UBitIntStatic::<N>::make(self.data)
    }

    #[inline]
    pub fn to(&self) -> Result<i128, String> {
        let uval: u128 = self.unsigned_abs().to()?;
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

impl<const N: usize> fmt::Display for BitIntStatic<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sign{
            write!(f, "{}{}", "-",(*self).unsigned_abs().to_string())?;
        }else{
            write!(f, "{}",(*self).unsigned_abs().to_string())?;
        }

        Ok(())
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

impl<const N: usize> PartialEq<isize> for BitIntStatic<N> {
    fn eq(&self, other: &isize) -> bool {
        if self.sign ^ (*other < 0) {
            return false;
        }

        if let Some(idx) = self.data.iter().rposition(|&x| x != 0) {
            if idx > 0 {
                return false;
            } else {
                return self.data[0] == (*other).unsigned_abs();
            }
        } else {
            return *other == 0;
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> PartialEq<i64> for BitIntStatic<N> {
    fn eq(&self, other: &i64) -> bool {
        if self.sign ^ (*other < 0) {
            return false;
        }

        if let Some(idx) = self.data.iter().rposition(|&x| x != 0) {
            if idx > 0 {
                return false;
            } else {
                return self.data[0] == (*other).unsigned_abs() as usize;
            }
        } else {
            return *other == 0;
        }
    }
}

macro_rules! impl_partial_eq_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> PartialEq<$t> for BitIntStatic<N>{
                fn eq(&self, other: &$t) -> bool {
                    if self.sign ^ (*other < 0){
                        return false
                    }

                    if let Some(idx) = self.data.iter().rposition(|&x| x != 0){
                        if idx > 0{
                            return false
                        }else{
                            return self.data[0] == (*other).unsigned_abs() as usize
                        }
                    }else{
                        return *other == 0
                    }
                }
            }
        )*
    };
}

impl_partial_eq_bis_prim!(i32, i16, i8);

impl<const N: usize> PartialOrd for BitIntStatic<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.sign ^ other.sign {
            if self.sign {
                return Some(std::cmp::Ordering::Less);
            } else {
                return Some(std::cmp::Ordering::Greater);
            }
        }

        if self.sign {
            return Some(
                UBitIntStatic::make(self.data)
                    .cmp(&UBitIntStatic::make(other.data))
                    .reverse(),
            );
        } else {
            return UBitIntStatic::make(self.data).partial_cmp(&UBitIntStatic::make(other.data));
        }
    }
}

impl<const N: usize> PartialOrd<isize> for BitIntStatic<N> {
    fn partial_cmp(&self, other: &isize) -> Option<std::cmp::Ordering> {
        if self.sign ^ (*other < 0) {
            if self.sign {
                return Some(std::cmp::Ordering::Less);
            } else {
                return Some(std::cmp::Ordering::Greater);
            }
        }

        if self.sign {
            return Some(
                UBitIntStatic::make(self.data)
                    .partial_cmp(&(*other).unsigned_abs())
                    .unwrap()
                    .reverse(),
            );
        } else {
            return UBitIntStatic::make(self.data).partial_cmp(&(*other).unsigned_abs());
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> PartialOrd<i64> for BitIntStatic<N> {
    fn partial_cmp(&self, other: &i64) -> Option<std::cmp::Ordering> {
        if self.sign ^ (*other < 0) {
            if self.sign {
                return Some(std::cmp::Ordering::Less);
            } else {
                return Some(std::cmp::Ordering::Greater);
            }
        }

        if self.sign {
            return Some(
                UBitIntStatic::make(self.data)
                    .partial_cmp(&(*other).unsigned_abs())
                    .unwrap()
                    .reverse(),
            );
        } else {
            return UBitIntStatic::make(self.data).partial_cmp(&(*other).unsigned_abs());
        }
    }
}

macro_rules! impl_partial_ord_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> PartialOrd<$t> for BitIntStatic<N>{
                fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering> {
                    if self.sign ^ (*other < 0){
                        if self.sign{
                            return Some(std::cmp::Ordering::Less)
                        }else{
                            return Some(std::cmp::Ordering::Greater)
                        }
                    }

                    if self.sign{
                        return Some(UBitIntStatic::make(self.data).partial_cmp(&(*other).unsigned_abs()).unwrap().reverse())
                    }else{
                        return UBitIntStatic::make(self.data).partial_cmp(&(*other).unsigned_abs())
                    }
                }
            }
        )*
    };
}

impl_partial_ord_bis_prim!(i32, i16, i8);

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

        let mut sign = lhs_s ^ rhs_s ^ add_ubis::<N>(&mut lhs_d, &rhs_d, 0);

        if sign {
            sign ^= twos_comp(&mut lhs_d)
        }

        BitIntStatic::<N> { data: lhs_d, sign }
    }
}

impl<const N: usize> Add<isize> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;

    fn add(self, rhs: isize) -> Self::Output {
        if self.sign ^ (rhs < 0) {
            match self.unsigned_abs()
                .partial_cmp(&(rhs.unsigned_abs()))
                .unwrap()
            {
                std::cmp::Ordering::Greater => {
                    let mut lhs = self.data;
                    sub_ubis_prim(&mut lhs, rhs.unsigned_abs());
                    BitIntStatic::<N> {
                        data: lhs,
                        sign: self.sign,
                    }
                }
                std::cmp::Ordering::Less => {
                    let mut lhs: [usize; N] = [0; N];
                    lhs[0] = rhs.unsigned_abs() - self.data[0];
                    BitIntStatic::<N> {
                        data: lhs,
                        sign: !self.sign,
                    }
                }
                std::cmp::Ordering::Equal => BitIntStatic::<N> {
                    data: [0; N],
                    sign: false,
                },
            }
        } else {
            let mut lhs = self.data;
            add_ubis_prim(&mut lhs, rhs.unsigned_abs());
            BitIntStatic::<N> {
                data: lhs,
                sign: self.sign,
            }
        }
    }
}

impl<const N: usize> Add<BitIntStatic<N>> for isize {
    type Output = BitIntStatic<N>;

    #[inline]
    fn add(self, rhs: BitIntStatic<N>) -> Self::Output {
        rhs + self
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> Add<i64> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;

    fn add(self, rhs: i64) -> Self::Output {
        if self.sign ^ (rhs < 0) {
            match self.unsigned_abs()
                .partial_cmp(&(rhs.unsigned_abs()))
                .unwrap()
            {
                std::cmp::Ordering::Greater => {
                    let mut lhs = self.data;
                    sub_ubis_prim(&mut lhs, rhs.unsigned_abs() as usize);
                    BitIntStatic::<N> {
                        data: lhs,
                        sign: self.sign,
                    }
                }
                std::cmp::Ordering::Less => {
                    let mut lhs: [usize; N] = [0; N];
                    lhs[0] = rhs.unsigned_abs() as usize - self.data[0];
                    BitIntStatic::<N> {
                        data: lhs,
                        sign: !self.sign,
                    }
                }
                std::cmp::Ordering::Equal => BitIntStatic::<N> {
                    data: [0; N],
                    sign: false,
                },
            }
        } else {
            let mut lhs = self.data;
            add_ubis_prim(&mut lhs, rhs.unsigned_abs() as usize);
            BitIntStatic::<N> {
                data: lhs,
                sign: self.sign,
            }
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> Add<BitIntStatic<N>> for i64 {
    type Output = BitIntStatic<N>;

    fn add(self, rhs: BitIntStatic<N>) -> Self::Output {
        rhs + self
    }
}

macro_rules! impl_add_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Add<$t> for BitIntStatic<N>{
                type Output = BitIntStatic<N>;

                fn add(self, rhs: $t) -> Self::Output {
                    if self.sign ^ (rhs < 0){
                        match self.unsigned_abs().partial_cmp(&(rhs.unsigned_abs())).unwrap(){
                            std::cmp::Ordering::Greater => {
                                let mut lhs = self.data;
                                sub_ubis_prim(&mut lhs, rhs.unsigned_abs() as usize);
                                BitIntStatic::<N> { data: lhs, sign: self.sign}
                            },
                            std::cmp::Ordering::Less => {
                                let mut lhs: [usize; N] = [0;N];
                                lhs[0] = rhs.unsigned_abs() as usize - self.data[0];
                                BitIntStatic::<N> { data:lhs, sign: !self.sign }
                            },
                            std::cmp::Ordering::Equal =>{
                                BitIntStatic::<N> { data: [0;N], sign: false}
                            }
                        }
                    }else{
                        let mut lhs = self.data;
                        add_ubis_prim(&mut lhs, rhs.unsigned_abs() as usize);
                        BitIntStatic::<N> { data: lhs, sign: self.sign}
                    }
                }
            }

            impl<const N:usize> Add<BitIntStatic<N>> for $t{
                type Output = BitIntStatic<N>;

                fn add(self, rhs: BitIntStatic<N>) -> Self::Output {
                    rhs + self
                }
            }
        )*
    };
}

impl_add_bis_prim!(i32, i16, i8);

impl<const N: usize> AddAssign for BitIntStatic<N> {
    fn add_assign(&mut self, rhs: Self) {
        if self.sign {
            self.sign ^= twos_comp(&mut self.data)
        }

        let (mut rhs_d, mut rhs_s) = (rhs.data, rhs.sign);
        if rhs_s {
            rhs_s ^= twos_comp(&mut rhs_d);
        }

        self.sign ^= rhs_s ^ add_ubis::<N>(&mut self.data, &rhs_d, 0);

        if self.sign {
            self.sign ^= twos_comp(&mut self.data)
        }
    }
}

impl<const N: usize> AddAssign<isize> for BitIntStatic<N> {
    fn add_assign(&mut self, rhs: isize) {
        if self.sign ^ (rhs < 0) {
            match self.unsigned_abs()
                .partial_cmp(&(rhs.unsigned_abs()))
                .unwrap()
            {
                std::cmp::Ordering::Greater => {
                    sub_ubis_prim(&mut self.data, rhs.unsigned_abs());
                }
                std::cmp::Ordering::Less => {
                    let mut lhs: [usize; N] = [0; N];
                    lhs[0] = rhs.unsigned_abs() - self.data[0];
                    self.data = lhs;
                    self.sign ^= true;
                }
                std::cmp::Ordering::Equal => {
                    self.data = [0; N];
                    self.sign = false;
                }
            }
        } else {
            add_ubis_prim(&mut self.data, rhs.unsigned_abs());
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> AddAssign<i64> for BitIntStatic<N> {
    fn add_assign(&mut self, rhs: i64) {
        if self.sign ^ (rhs < 0) {
            match self.unsigned_abs()
                .partial_cmp(&(rhs.unsigned_abs()))
                .unwrap()
            {
                std::cmp::Ordering::Greater => {
                    sub_ubis_prim(&mut self.data, rhs.unsigned_abs() as usize);
                }
                std::cmp::Ordering::Less => {
                    let mut lhs: [usize; N] = [0; N];
                    lhs[0] = rhs.unsigned_abs() as usize - self.data[0];
                    self.data = lhs;
                    self.sign ^= true;
                }
                std::cmp::Ordering::Equal => {
                    self.data = [0; N];
                    self.sign = false;
                }
            }
        } else {
            add_ubis_prim(&mut self.data, rhs.unsigned_abs() as usize);
        }
    }
}

macro_rules! impl_add_assign_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> AddAssign<$t> for BitIntStatic<N>{
                fn add_assign(&mut self, rhs: $t) {
                    if self.sign ^ (rhs < 0){
                        match self.unsigned_abs().partial_cmp(&(rhs.unsigned_abs())).unwrap(){
                            std::cmp::Ordering::Greater => {
                                sub_ubis_prim(&mut self.data, rhs.unsigned_abs() as usize);
                            },
                            std::cmp::Ordering::Less => {
                                let mut lhs: [usize; N] = [0;N];
                                lhs[0] = rhs.unsigned_abs() as usize - self.data[0];
                                self.data = lhs;
                                self.sign ^= true;
                            },
                            std::cmp::Ordering::Equal =>{
                                self.data = [0;N];
                                self.sign = false;
                            }
                        }
                    }else{
                        add_ubis_prim(&mut self.data, rhs.unsigned_abs() as usize);
                    }
                }
            }
        )*
    };
}

impl_add_assign_bis_prim!(i32, i16, i8);

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
        let (mut lhs_d, mut lhs_s) = (self.data, self.sign);
        if lhs_s {
            lhs_s ^= twos_comp(&mut lhs_d)
        }

        let (mut rhs_d, mut rhs_s) = (rhs.data, !rhs.sign);
        if rhs_s {
            rhs_s ^= twos_comp(&mut rhs_d);
        }

        let mut sign = lhs_s ^ rhs_s ^ add_ubis::<N>(&mut lhs_d, &rhs_d, 0);

        if sign {
            sign ^= twos_comp(&mut lhs_d)
        }

        BitIntStatic::<N> { data: lhs_d, sign }
    }
}

impl<const N: usize> Sub<isize> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;

    fn sub(self, rhs: isize) -> Self::Output {
        if self.sign ^ (rhs >= 0) {
            match self.unsigned_abs()
                .partial_cmp(&(rhs.unsigned_abs()))
                .unwrap()
            {
                std::cmp::Ordering::Greater => {
                    let mut lhs = self.data;
                    sub_ubis_prim(&mut lhs, rhs.unsigned_abs());
                    BitIntStatic::<N> {
                        data: lhs,
                        sign: self.sign,
                    }
                }
                std::cmp::Ordering::Less => {
                    let mut lhs: [usize; N] = [0; N];
                    lhs[0] = rhs.unsigned_abs() - self.data[0];
                    BitIntStatic::<N> {
                        data: lhs,
                        sign: !self.sign,
                    }
                }
                std::cmp::Ordering::Equal => BitIntStatic::<N> {
                    data: [0; N],
                    sign: false,
                },
            }
        } else {
            let mut lhs = self.data;
            add_ubis_prim(&mut lhs, rhs.unsigned_abs());
            BitIntStatic::<N> {
                data: lhs,
                sign: self.sign,
            }
        }
    }
}

impl<const N: usize> Sub<BitIntStatic<N>> for isize {
    type Output = BitIntStatic<N>;

    fn sub(self, rhs: BitIntStatic<N>) -> Self::Output {
        let mut out = rhs - self;
        out.mut_neg();
        out
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> Sub<i64> for BitIntStatic<N> {
    type Output = BitIntStatic<N>;

    fn sub(self, rhs: i64) -> Self::Output {
        if self.sign ^ (rhs >= 0) {
            match self.unsigned_abs()
                .partial_cmp(&(rhs.unsigned_abs()))
                .unwrap()
            {
                std::cmp::Ordering::Greater => {
                    let mut lhs = self.data;
                    sub_ubis_prim(&mut lhs, rhs.unsigned_abs() as usize);
                    BitIntStatic::<N> {
                        data: lhs,
                        sign: self.sign,
                    }
                }
                std::cmp::Ordering::Less => {
                    let mut lhs: [usize; N] = [0; N];
                    lhs[0] = rhs.unsigned_abs() as usize - self.data[0];
                    BitIntStatic::<N> {
                        data: lhs,
                        sign: !self.sign,
                    }
                }
                std::cmp::Ordering::Equal => BitIntStatic::<N> {
                    data: [0; N],
                    sign: false,
                },
            }
        } else {
            let mut lhs = self.data;
            add_ubis_prim(&mut lhs, rhs.unsigned_abs() as usize);
            BitIntStatic::<N> {
                data: lhs,
                sign: self.sign,
            }
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N: usize> Sub<BitIntStatic<N>> for i64 {
    type Output = BitIntStatic<N>;

    fn sub(self, rhs: BitIntStatic<N>) -> Self::Output {
        let mut out = rhs - self;
        out.mut_neg();
        out
    }
}

macro_rules! impl_sub_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Sub<$t> for BitIntStatic<N>{
                type Output = BitIntStatic<N>;

                fn sub(self, rhs: $t) -> Self::Output {
                    if self.sign ^ (rhs >= 0){
                        match self.unsigned_abs().partial_cmp(&(rhs.unsigned_abs())).unwrap(){
                            std::cmp::Ordering::Greater => {
                                let mut lhs = self.data;
                                sub_ubis_prim(&mut lhs, rhs.unsigned_abs() as usize);
                                BitIntStatic::<N> { data: lhs, sign: self.sign}
                            },
                            std::cmp::Ordering::Less => {
                                let mut lhs: [usize; N] = [0;N];
                                lhs[0] = rhs.unsigned_abs() as usize - self.data[0];
                                BitIntStatic::<N> { data:lhs, sign: !self.sign }
                            },
                            std::cmp::Ordering::Equal =>{
                                BitIntStatic::<N> { data: [0;N], sign: false}
                            }
                        }
                    }else{
                        let mut lhs = self.data;
                        add_ubis_prim(&mut lhs, rhs.unsigned_abs() as usize);
                        BitIntStatic::<N> { data: lhs, sign: self.sign}
                    }
                }
            }

            impl<const N:usize> Sub<BitIntStatic<N>> for $t {
                type Output = BitIntStatic<N>;

                fn sub(self, rhs: BitIntStatic<N>) -> Self::Output {
                    let mut out = rhs - self;
                    out.mut_neg();
                    out
                }
            }
        )*
    };
}

impl_sub_bis_prim!(i32, i16, i8);

impl<const N: usize> SubAssign for BitIntStatic<N> {
    fn sub_assign(&mut self, rhs: Self) {
        if self.sign {
            self.sign ^= twos_comp(&mut self.data)
        }

        let (mut rhs_d, mut rhs_s) = (rhs.data, !rhs.sign);
        if rhs_s {
            rhs_s ^= twos_comp(&mut rhs_d);
        }

        self.sign ^= rhs_s ^ add_ubis::<N>(&mut self.data, &rhs_d, 0);

        if self.sign {
            self.sign ^= twos_comp(&mut self.data)
        }
    }
}

impl<const N:usize> SubAssign<isize> for BitIntStatic<N>{
    fn sub_assign(&mut self, rhs: isize) {
        if self.sign ^ (rhs >= 0) {
            match self.unsigned_abs()
                .partial_cmp(&(rhs.unsigned_abs()))
                .unwrap()
            {
                std::cmp::Ordering::Greater => {
                    sub_ubis_prim(&mut self.data, rhs.unsigned_abs());
                }
                std::cmp::Ordering::Less => {
                    let mut lhs: [usize; N] = [0; N];
                    lhs[0] = rhs.unsigned_abs() - self.data[0];
                    self.data = lhs;
                    self.sign ^= true;
                }
                std::cmp::Ordering::Equal => {
                    self.data = [0; N];
                    self.sign = false;
                }
            }
        } else {
            add_ubis_prim(&mut self.data, rhs.unsigned_abs());
        }
    }
}

#[cfg(target_pointer_width = "64")]
impl<const N:usize> SubAssign<i64> for BitIntStatic<N>{
    fn sub_assign(&mut self, rhs: i64) {
        if self.sign ^ (rhs >= 0) {
            match self.unsigned_abs()
                .partial_cmp(&(rhs.unsigned_abs()))
                .unwrap()
            {
                std::cmp::Ordering::Greater => {
                    sub_ubis_prim(&mut self.data, rhs.unsigned_abs() as usize);
                }
                std::cmp::Ordering::Less => {
                    let mut lhs: [usize; N] = [0; N];
                    lhs[0] = rhs.unsigned_abs() as usize - self.data[0];
                    self.data = lhs;
                    self.sign ^= true;
                }
                std::cmp::Ordering::Equal => {
                    self.data = [0; N];
                    self.sign = false;
                }
            }
        } else {
            add_ubis_prim(&mut self.data, rhs.unsigned_abs() as usize);
        }
    }
}

macro_rules! impl_sub_assign_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> SubAssign<$t> for BitIntStatic<N>{
                fn sub_assign(&mut self, rhs: $t) {
                    if self.sign ^ (rhs >= 0) {
                        match self.unsigned_abs()
                            .partial_cmp(&(rhs.unsigned_abs()))
                            .unwrap()
                        {
                            std::cmp::Ordering::Greater => {
                                sub_ubis_prim(&mut self.data, rhs.unsigned_abs() as usize);
                            }
                            std::cmp::Ordering::Less => {
                                let mut lhs: [usize; N] = [0; N];
                                lhs[0] = rhs.unsigned_abs() as usize - self.data[0];
                                self.data = lhs;
                                self.sign ^= true;
                            }
                            std::cmp::Ordering::Equal => {
                                self.data = [0; N];
                                self.sign = false;
                            }
                        }
                    } else {
                        add_ubis_prim(&mut self.data, rhs.unsigned_abs() as usize);
                    }
                }
            }
        )*
    };
}

impl_sub_assign_bis_prim!(i32, i16, i8);

impl<const N:usize> Mul for BitIntStatic<N>{
    type Output = BitIntStatic<N>;

    fn mul(self, rhs: Self) -> Self::Output {
        BitIntStatic::<N> { data: mul_ubis(&self.data, &rhs.data), sign: self.sign^rhs.sign}
    }
}

impl<const N:usize> Mul<i128> for BitIntStatic<N>{
    type Output = BitIntStatic<N>;

    fn mul(self, rhs: i128) -> Self::Output {
        let mut lhs = self.data;
        mul_ubis_prim(&mut lhs, rhs.unsigned_abs());
        BitIntStatic::<N> { data: lhs, sign: self.sign ^ (rhs < 0)}
    }
}

impl<const N:usize> Mul<BitIntStatic<N>> for i128{
    type Output = BitIntStatic<N>;

    fn mul(self, rhs: BitIntStatic<N>) -> Self::Output {
        rhs * self
    }
}

macro_rules! impl_mul_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Mul<$t> for BitIntStatic<N>{
                type Output = BitIntStatic<N>;
            
                fn mul(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    mul_ubis_prim(&mut lhs, rhs.unsigned_abs() as u128);
                    BitIntStatic::<N> { data: lhs, sign: self.sign ^ (rhs < 0)}
                }
            }

            impl<const N:usize> Mul<BitIntStatic<N>> for $t{
                type Output = BitIntStatic<N>;
            
                fn mul(self, rhs: BitIntStatic<N>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}

impl_mul_bis_prim!(i64, isize, i32, i16, i8);

impl<const N:usize> MulAssign for BitIntStatic<N>{
    fn mul_assign(&mut self, rhs: Self) {
        self.data = mul_ubis(&self.data, &rhs.data);
        self.sign ^= rhs.sign;
    }
}

impl<const N:usize> MulAssign<i128> for BitIntStatic<N>{
    fn mul_assign(&mut self, rhs: i128) {
        mul_ubis_prim(&mut self.data, rhs.unsigned_abs());
        self.sign ^= rhs < 0;
    }
}

macro_rules! impl_mul_assign_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> MulAssign<$t> for BitIntStatic<N>{
                fn mul_assign(&mut self, rhs: $t) {
                    mul_ubis_prim(&mut self.data, rhs.unsigned_abs() as u128);
                    self.sign ^= rhs < 0;
                }
            }
        )*
    };
}

impl_mul_assign_bis_prim!(i64, isize, i32, i16, i8);

impl<const N:usize> Div for BitIntStatic<N>{
    type Output = BitIntStatic<N>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut ulhs = self.unsigned_abs();
        let urhs = rhs.unsigned_abs();

        BitIntStatic::<N>{ data: div_ubis(&mut ulhs, urhs).get_data(), sign: self.sign^rhs.sign}
    }
}

macro_rules! impl_div_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Div<$t> for BitIntStatic<N>{
                type Output = BitIntStatic<N>;
            
                fn div(self, rhs: $t) -> Self::Output {
                    let mut ulhs = self.unsigned_abs();
                    let urhs = UBitIntStatic::<N>::from(rhs.unsigned_abs());
            
                    BitIntStatic::<N>{ data: div_ubis(&mut ulhs, urhs).get_data(), sign: self.sign^(rhs < 0)}
                }
            }
        )*
    };
}

impl_div_bis_prim!(i128, i64, isize, i32, i16, i8);

impl<const N:usize> DivAssign for BitIntStatic<N>{
    fn div_assign(&mut self, rhs: Self) {
        let mut ulhs = self.unsigned_abs();
        let urhs = rhs.unsigned_abs();

        self.data = div_ubis(&mut ulhs, urhs).get_data();
        self.sign ^= rhs < 0;
    }
}

macro_rules! impl_div_assign_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> DivAssign<$t> for BitIntStatic<N>{
                fn div_assign(&mut self, rhs: $t) {
                    let mut ulhs = self.unsigned_abs();
                    let urhs = UBitIntStatic::<N>::from(rhs.unsigned_abs());
            
                    self.data = div_ubis(&mut ulhs, urhs).get_data();
                    self.sign ^= rhs < 0;
                }
            }
        )*
    };
}

impl_div_assign_bis_prim!(i128, i64, isize, i32, i16, i8);

impl<const N:usize> Rem for BitIntStatic<N>{
    type Output = BitIntStatic<N>;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut ulhs = self.unsigned_abs();
        let urhs = rhs.unsigned_abs();
        div_ubis(&mut ulhs, urhs);

        if rhs.sign ^ self.sign{
            BitIntStatic::<N> { data: (urhs - ulhs).get_data(), sign: rhs.sign}
        }else{
            BitIntStatic::<N> { data: ulhs.get_data(), sign: rhs.sign}
        }
    }
}

macro_rules! impl_rem_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Rem<$t> for BitIntStatic<N>{
                type Output = BitIntStatic<N>;
            
                fn rem(self, rhs: $t) -> Self::Output {
                    let mut ulhs = self.unsigned_abs();
                    let urhs = UBitIntStatic::<N>::from(rhs.unsigned_abs());
                    div_ubis(&mut ulhs, urhs);
            
                    if (rhs < 0) ^ self.sign{
                        BitIntStatic::<N> { data: (urhs - ulhs).get_data(), sign: rhs < 0}
                    }else{
                        BitIntStatic::<N> { data: ulhs.get_data(), sign: rhs < 0}
                    }
                }
            }
        )*
    };
}

impl_rem_bis_prim!(i128, i64, isize, i32, i16, i8);

impl<const N:usize> RemAssign for BitIntStatic<N>{
    fn rem_assign(&mut self, rhs: Self) {
        let mut ulhs = self.unsigned_abs();
        let urhs = rhs.unsigned_abs();
        div_ubis(&mut ulhs, urhs);

        if rhs.sign ^ self.sign{
            self.data = (urhs - ulhs).get_data();
        }else{
            self.data = ulhs.get_data();
        }

        self.sign = rhs.sign;
    }
}

macro_rules! impl_rem_assign_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> RemAssign<$t> for BitIntStatic<N>{
                fn rem_assign(&mut self, rhs: $t){
                    let mut ulhs = self.unsigned_abs();
                    let urhs = UBitIntStatic::<N>::from(rhs.unsigned_abs());
                    div_ubis(&mut ulhs, urhs);
            
                    if (rhs < 0) ^ self.sign{
                        self.data = (urhs - ulhs).get_data();
                    }else{
                        self.data = ulhs.get_data();
                    }

                    self.sign = rhs < 0;
                }
            }
        )*
    };
}

impl_rem_assign_bis_prim!(i128, i64, isize, i32, i16, i8);

impl<const N:usize> Shl<u128> for BitIntStatic<N>{
    type Output = BitIntStatic<N>;

    fn shl(self, rhs: u128) -> Self::Output {
        let mut lhs = self.data;
        shl_ubis::<N>(&mut lhs, rhs);
        BitIntStatic::<N> { data:lhs , sign: self.sign}
    }
}

macro_rules! impl_shl_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Shl<$t> for BitIntStatic<N>{
                type Output = BitIntStatic<N>;
            
                fn shl(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    shl_ubis::<N>(&mut lhs, rhs as u128);
                    BitIntStatic::<N> { data: lhs, sign: self.sign}
                }
            }
        )*
    };
}

impl_shl_bis_prim!(u64, usize, u32, u16, u8);

impl<const N:usize> ShlAssign<u128> for BitIntStatic<N>{
    fn shl_assign(&mut self, rhs: u128) {
        shl_ubis::<N>(&mut self.data, rhs);
    }
}

macro_rules! impl_shl_assign_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> ShlAssign<$t> for BitIntStatic<N>{
                fn shl_assign(&mut self, rhs: $t) {
                    shl_ubis::<N>(&mut self.data, rhs as u128);
                }
            }
        )*
    };
}

impl_shl_assign_bis_prim!(u64, usize, u32, u16, u8);

impl<const N:usize> Shr<u128> for BitIntStatic<N>{
    type Output = BitIntStatic<N>;

    fn shr(self, rhs: u128) -> Self::Output {
        let mut lhs = self.data;
        shr_ubis::<N>(&mut lhs, rhs);
        BitIntStatic::<N> { data: lhs, sign: self.sign}
    }
}

macro_rules! impl_shr_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Shr<$t> for BitIntStatic<N>{
                type Output = BitIntStatic<N>;
            
                fn shr(self, rhs: $t) -> Self::Output {
                    let mut lhs = self.data;
                    shr_ubis::<N>(&mut lhs, rhs as u128);
                    BitIntStatic::<N> { data: lhs, sign: self.sign}
                }
            }
        )*
    };
}

impl_shr_bis_prim!(u64, usize, u32, u16, u8);

impl<const N:usize> ShrAssign<u128> for BitIntStatic<N>{
    fn shr_assign(&mut self, rhs: u128) {
        shr_ubis::<N>(&mut self.data, rhs);
    }
}

macro_rules! impl_shr_assign_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> ShrAssign<$t> for BitIntStatic<N>{
                fn shr_assign(&mut self, rhs: $t) {
                    shr_ubis::<N>(&mut self.data, rhs as u128);
                }
            }
        )*
    };
}

impl_shr_assign_bis_prim!(u64, usize, u32, u16, u8);

impl<const N:usize> Pow for BitIntStatic<N>{
    type Output = BitIntStatic<N>;

    fn pow(self, rhs: Self) -> Self::Output {
        let b = self.unsigned_abs();
        let exp = rhs.unsigned_abs();

        let abs_pow_data = b.pow(exp).get_data();
        if self.sign{
            return BitIntStatic::<N> { data: abs_pow_data, sign: self.mod2_abs()};
        }else{
            return BitIntStatic::<N> { data: abs_pow_data, sign: false};
        }
    }
}

macro_rules! impl_pow_bis_prim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> Pow<$t> for BitIntStatic<N>{
                type Output = BitIntStatic<N>;
            
                fn pow(self, rhs: $t) -> Self::Output {
                    let b = self.unsigned_abs();
                    let exp = UBitIntStatic::<N>::from(rhs);
            
                    let abs_pow_data = b.pow(exp).get_data();
                    if self.sign{
                        return BitIntStatic::<N> { data: abs_pow_data, sign: self.mod2_abs()};
                    }else{
                        return BitIntStatic::<N> { data: abs_pow_data, sign: false};
                    }
                }
            }
        )*
    };
}

impl_pow_bis_prim!(u128, u64, usize, u32, u16, u8);
