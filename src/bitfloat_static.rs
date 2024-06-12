use crate::{
    bitint_static::BitIntStatic,
    ubitint::*,
    ubitint_static::{DivRem, UBitIntStatic},
};
use once_cell::sync::Lazy;
use std::ops::*;

fn add_bfs<const N: usize>(lhs: &mut [usize], rhs: &[usize], sh: usize) -> bool {
    unsafe {
        let mut c: u8 = 0;
        for (l, r) in lhs.iter_mut().zip(&rhs[sh..]) {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(l, *r, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(l, *r, &mut c);
        }

        for l in &mut lhs[(N - sh)..] {
            if c == 0 {
                break;
            }
            #[cfg(target_arch = "aarch64")]
            add_carry_aarch64(l, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_carry_x86_64(l, &mut c);
        }

        return c == 1;
    }
}

fn sub_bfs<const N: usize>(lhs: &mut [usize], rhs: &[usize], sh: usize) {
    unsafe {
        let mut c: u8 = 1;
        for (l, r) in lhs.iter_mut().zip(&rhs[sh..]) {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(l, !*r, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(l, !*r, &mut c);
        }

        if c == 0 {
            for l in &mut lhs[(N - sh)..] {
                #[cfg(target_arch = "aarch64")]
                sub_carry_aarch64(l, &mut c);

                #[cfg(target_arch = "x86_64")]
                sub_carry_x86_64(l, &mut c);
            }
        }
    }
}

fn mul_bfs<const N: usize>(a: &[usize], b: &[usize]) -> ([usize; N], u128) {
    let mut out = [0_usize; N];

    let mask = u64::MAX as u128;

    let mut carry: u128 = 0;
    for i in 0..N {
        let mut term: u128 = carry;
        carry = 0;
        for j in i..N {
            let val = (a[j] as u128) * (b[N - 1 + i - j] as u128);
            term += val & mask;
            carry += val >> 64;
        }
        carry += term >> 64;
        out[i] = term as usize;
    }

    return (out, carry);
}

fn shl_bfs<const N: usize>(m: &mut [usize], sh: u8) -> i128 {
    let mv_sz = 64 - sh;
    let mut carry = 0;
    unsafe {
        for elem in m.iter_mut() {
            #[cfg(target_arch = "aarch64")]
            shl_carry_aarch64(elem, &mut carry, sh, mv_sz);

            #[cfg(target_arch = "x86_64")]
            shl_carry_x86_64(elem, &mut carry, sh, mv_sz);
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
            #[cfg(target_arch = "aarch64")]
            shr_carry_aarch64(elem, &mut carry, sh, mv_sz);

            #[cfg(target_arch = "x86_64")]
            shr_carry_x86_64(elem, &mut carry, sh, mv_sz);
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

#[macro_export]
macro_rules! make_int {
    ($val:expr, $N:expr) => {{
        let mut m = [0_usize; $N];
        m[$N - 1] = $val;
        BitFloatStatic::<$N>::make(m, 0, false)
    }};
}

fn div_bfs_gs<const N: usize>(n: &mut BitFloatStatic<N>, mut d: BitFloatStatic<N>) {
    n.exp -= d.exp + 1;
    d.exp = -1;

    let sh = d.m[N - 1].leading_zeros() as u8;
    if sh != 0 {
        shl_bfs::<N>(&mut d.m, sh);
        n.exp += shl_bfs::<N>(&mut n.m, sh);
    }

    n.abs();
    d.abs();

    let two = make_int!(2, N);
    let one = make_int!(1, N);

    let mut f = two - d;
    while f > one {
        d *= f;
        *n *= f;
        f = two - d;
    }
}

fn powi_ubi<const N: usize, const M: usize>(
    x: BitFloatStatic<N>,
    n: UBitIntStatic<M>,
) -> BitFloatStatic<N> {
    if n == 0_usize {
        return make_int!(1, N);
    } else if n == 1_usize {
        return x;
    } else if n.mod2() {
        x * powi_ubi(x * x, n >> 1_u128)
    } else {
        return powi_ubi(x * x, n >> 1_u128);
    }
}

pub static LN2_DATA: Lazy<Vec<usize>> = Lazy::new(|| {
    vec![
        12786308645202655659,
        14547668686819489455,
        4680158270178506285,
        9947632833883994667,
        16697225500131306648,
        6166925505658844291,
        8866180489242585516,
        11158079463888603658,
        13744767134432929394,
        16885391035760960380,
        7666328946881899146,
        640423544668653624,
        9913893022404239091,
        12996353481489890045,
        3660608522376630786,
        3459543747794943001,
        12866720777460081736,
        14178553729970976500,
    ]
});

pub static REP_LN2_DATA: Lazy<Vec<usize>> = Lazy::new(|| {
    vec![
        1,
        8166282121979093367,
        9011700246555294993,
        15469571501439759537,
        1606145530369565839,
        13562739736953412012,
        5310413354149083216,
        15256426431123078771,
        12135079278608564371,
        15518991392522668671,
        3014604011625205639,
        16322112991531077143,
        14367175982684688090,
        69839179731384734,
        1875675896573248823,
        1935475471239071041,
        5507552693231658938,
        8909034407454128537,
        16112723916186887149,
    ]
});

#[macro_export]
macro_rules! make_LN2 {
    ($N:expr) => {{
        let iter = (&*LN2_DATA).iter().chain(std::iter::repeat(&0_usize));
        let vec: Vec<usize> = iter.take($N).map(|&x| x).collect();
        let m: [usize; $N] = vec
            .into_iter()
            .rev()
            .collect::<Vec<usize>>()
            .try_into()
            .expect("wrong length");

        BitFloatStatic::<$N>::make(m, -1, false)
    }};
}

#[macro_export]
macro_rules! make_rep_LN2 {
    ($N:expr) => {{
        let iter = (&*REP_LN2_DATA).iter().chain(std::iter::repeat(&0_usize));
        let vec: Vec<usize> = iter.take($N).map(|&x| x).collect();
        let m: [usize; $N] = vec
            .into_iter()
            .rev()
            .collect::<Vec<usize>>()
            .try_into()
            .expect("wrong length");

        BitFloatStatic::<$N>::make(m, 1, false)
    }};
}

type MyLazyType = Lazy<
    std::iter::Chain<
        std::slice::Iter<'static, usize>,
        std::iter::Cycle<std::slice::Iter<'static, usize>>,
    >,
>;

pub static STATIC_HARMONICS: [MyLazyType; 33] = [
    Lazy::new(|| {
        let init = Box::new(vec![9223372036854775808]);
        let rep = Box::new(vec![0]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/2
    Lazy::new(|| {
        let init = Box::new(vec![12297829382473034410]);
        let rep = Box::new(vec![6148914691236517205]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/3
    Lazy::new(|| {
        let init = Box::new(vec![4611686018427387904]);
        let rep = Box::new(vec![0]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/4
    Lazy::new(|| {
        let init = Box::new(vec![10131064948296120970]);
        let rep = Box::new(vec![3689348814741910322]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/5
    Lazy::new(|| {
        let init = Box::new(vec![15372286728091293012]);
        let rep = Box::new(vec![12297829382473034410]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/6
    Lazy::new(|| {
        let init = Box::new(vec![7905747460161236406]);
        let rep = Box::new(vec![
            2635249153387078802,
            10540996613548315209,
            5270498306774157604,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/7
    Lazy::new(|| {
        let init = Box::new(vec![2305843009213693950]);
        let rep = Box::new(vec![0]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/8
    Lazy::new(|| {
        let init = Box::new(vec![4252738302106628413]);
        let rep = Box::new(vec![
            2049638230412172398,
            8198552921648689607,
            14347467612885206812,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/9
    Lazy::new(|| {
        let init = Box::new(vec![1275760113925725791]);
        let rep = Box::new(vec![11068046444225730967]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/10
    Lazy::new(|| {
        let init = Box::new(vec![6755947246084643949]);
        let rep = Box::new(vec![
            6707906935894382403,
            5030930201920786804,
            8384883669867978007,
            1676976733973595601,
            15092790605762360413,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/11
    Lazy::new(|| {
        let init = Box::new(vec![7686143364045646506]);
        let rep = Box::new(vec![6148914691236517205]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/12
    Lazy::new(|| {
        let init = Box::new(vec![11013875738701244365]);
        let rep = Box::new(vec![
            1418980313362273200,
            12770822820260458811,
            4256940940086819603,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/13
    Lazy::new(|| {
        let init = Box::new(vec![3952873730080618203]);
        let rep = Box::new(vec![
            10540996613548315209,
            5270498306774157604,
            2635249153387078802,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/14
    Lazy::new(|| {
        let init = Box::new(vec![2459565876494606882]);
        let rep = Box::new(vec![1229782938247303441]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/15
    Lazy::new(|| {
        let init = Box::new(vec![1152921504606846974]);
        let rep = Box::new(vec![0]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/16
    Lazy::new(|| {
        let init = Box::new(vec![12186111669619790943]);
        let rep = Box::new(vec![1085102592571150091]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/17
    Lazy::new(|| {
        let init = Box::new(vec![10215259773411907504]);
        let rep = Box::new(vec![
            10248191152060862004,
            4099276460824344803,
            16397105843297379214,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/18
    Lazy::new(|| {
        let init = Box::new(vec![4436022282994528602]);
        let rep = Box::new(vec![
            970881267037344817,
            8737931403336103397,
            4854406335186724109,
            6796168869261413753,
            5825287602224068931,
            15534100272597517150,
            10679693937410793040,
            3883525068149379287,
            16504981539634861972,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/19
    Lazy::new(|| {
        let init = Box::new(vec![13639831235185797706]);
        let rep = Box::new(vec![14757395258967641289]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/20
    Lazy::new(|| {
        let init = Box::new(vec![2059135504606161184]);
        let rep = Box::new(vec![
            878416384462359598,
            3513665537849438403,
            14054662151397753612,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/21
    Lazy::new(|| {
        let init = Box::new(vec![4273866306634753401]);
        let rep = Box::new(vec![
            3353953467947191199,
            11738837137815169210,
            13415813871788764811,
            10061860403841573608,
            16769767339735956014,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/22
    Lazy::new(|| {
        let init = Box::new(vec![965871173309366267]);
        let rep = Box::new(vec![
            2406097053092550208,
            9624388212370200843,
            1604064702061700140,
            6416258808246800562,
            7218291159277650632,
            10426420563401050913,
            4812194106185100421,
            802032351030850070,
            3208129404123400281,
            12832517616493601124,
            14436582318555301264,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/23
    Lazy::new(|| {
        let init = Box::new(vec![13066443718877599060]);
        let rep = Box::new(vec![12297829382473034410]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/24
    Lazy::new(|| {
        let init = Box::new(vec![12751593265824099710]);
        let rep = Box::new(vec![
            15495265021916023355,
            4427218577690292387,
            11805916207174113034,
            737869762948382064,
            8116567392432202711,
            15495265021916023357,
            4427218577690292387,
            11805916207174113034,
            737869762948382064,
            8116567392432202711,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/25
    Lazy::new(|| {
        let init = Box::new(vec![5335815181428988928, 9932862193535912406]);
        let rep = Box::new(vec![
            15608783446985005213,
            11351842506898185609,
            9932862193535912408,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/26
    Lazy::new(|| {
        let init = Box::new(vec![16801190198990128965, 683212743470724132]);
        let rep = Box::new(vec![
            8881765665119413741,
            4782489204295068937,
            6832127434707241339,
            15030680356355930946,
            10931403895531586142,
            12981042125943758544,
            2732850973882896535,
            17080318586768103348,
            683212743470724133,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/27
    Lazy::new(|| {
        let init = Box::new(vec![11199808901895084909]);
        let rep = Box::new(vec![
            5270498306774157604,
            2635249153387078802,
            10540996613548315209,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/28
    Lazy::new(|| {
        let init = Box::new(vec![5272946898442224699, 10177513971701821579]);
        let rep = Box::new(vec![
            12721892464627276976,
            15902365580784096220,
            15266270957552732371,
            636094623231363848,
            14630176334321368523,
            4452662362619546941,
            10177513971701821581,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/29
    Lazy::new(|| {
        let init = Box::new(vec![10453154975102079248]);
        let rep = Box::new(vec![9838263505978427528]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/30
    Lazy::new(|| {
        let init = Box::new(vec![1785168781326730801]);
        let rep = Box::new(vec![
            2380225041768974402,
            4760450083537948804,
            9520900167075897608,
            595056260442243600,
            1190112520884487201,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/31
    Lazy::new(|| {
        let init = Box::new(vec![576460752303423486]);
        let rep = Box::new(vec![0]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/32
    Lazy::new(|| {
        let init = Box::new(vec![3215661856662981158, 2235968978631460794]);
        let rep = Box::new(vec![
            13974806116446630012,
            8943875914525843207,
            558992244657865200,
            17328759584393821215,
            2235968978631460801,
            13974806116446630012,
            8943875914525843207,
            558992244657865200,
            17328759584393821215,
            2235968978631460801,
        ]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/33
    Lazy::new(|| {
        let init = Box::new(vec![1430218888451839311, 9765923333140350847]);
        let rep = Box::new(vec![9765923333140350855]);
        Box::leak(init).iter().chain(Box::leak(rep).iter().cycle())
    }), // 1/34
];

#[macro_export]
macro_rules! make_harmonic {
    ($n:expr, $N:expr) => {{
        let mut m: [usize; N] = [0; $N];
        let iter = STATIC_HARMONICS[$n - 2].clone();

        for (i, &val) in iter.take($N).enumerate() {
            m[N - i - 1] = val;
        }

        BitFloatStatic::<$N>::make(m, -1, false)
    }};
}

pub fn exp<const N: usize>(val: BitFloatStatic<N>) -> BitFloatStatic<N> {
    let ln2 = make_LN2!(N);
    let rln2 = make_rep_LN2!(N);

    let n = (val * rln2).round();
    println!("n: {:?}", n);

    let x = val - n * ln2;
    println!("x: {:?}", x);

    let mut out = make_int!(1, N) + x;
    let mut term = x;
    let mut state = true;
    for i in 2..34 {
        term *= x * make_harmonic!(i, N);
        out += term;

        if term.exp <= -(N as i128) {
            state = false;
            break;
        }
    }

    if state {
        let mut i = 35;
        while term.exp > -(N as i128) {
            term *= x / make_int!(i, N);
            out += term;
            i += 1;
        }
    }

    return out << n.m[N - 1] as u128;
}

#[derive(Debug, Clone, Copy)]
pub struct BitFloatStatic<const N: usize> {
    m: [usize; N],
    exp: i128,
    sign: bool,
}

pub trait SEM<const N: usize> {
    fn get_sem(self) -> (bool, i128, [usize; N]);
}

impl<const N: usize> SEM<N> for f64 {
    fn get_sem(self) -> (bool, i128, [usize; N]) {
        let bits = self.to_bits();
        let sign = (bits >> 63) == 1;
        let exp2 = ((bits >> 52) & 0x7FF) as i128 - 1023;

        let (exp, sh) = if exp2 >= 0 {
            (exp2 / 64, exp2 % 64)
        } else {
            (exp2 / 64 - 1, 64 - exp2 % 64)
        };

        let m2 = (((bits & 0xFFFFFFFFFFFFF) | 0x10000000000000) as u128) << (sh + 12);

        let mut m = [0; N];
        let upper_m2 = (m2 >> 64) as usize;

        if upper_m2 == 0 {
            m[N - 1] = m2 as usize;
        } else {
            m[N - 1] = upper_m2;
            m[N - 2] = m2 as usize;
        }

        (sign, exp, m)
    }
}

impl<const N: usize> SEM<N> for f32 {
    fn get_sem(self) -> (bool, i128, [usize; N]) {
        let bits = self.to_bits();
        let sign = (bits >> 31) == 1;
        let exp2 = ((bits >> 23) & 0xFF) as i128 - 127;

        let (exp, sh) = if exp2 >= 0 {
            (exp2 / 64, exp2 % 64 + 1)
        } else {
            (exp2 / 64 - 1, 64 - exp2 % 64 + 1)
        };

        let m2 = (((bits & 0x7FFFF) | 0x80000) as u128) << sh;

        let mut m = [0; N];
        let upper_m2 = (m2 >> 64) as usize;

        if upper_m2 == 0 {
            m[N - 1] = m2 as usize;
        } else {
            m[N - 1] = upper_m2;
            m[N - 2] = m2 as usize;
        }

        (sign, exp, m)
    }
}

impl<const N: usize> SEM<N> for i128 {
    fn get_sem(self) -> (bool, i128, [usize; N]) {
        let sign = self < 0;

        let (m0, m1) = (
            self.unsigned_abs() as usize,
            (self.unsigned_abs() >> 64) as usize,
        );
        let mut m = [0; N];
        let exp: i128;
        if m1 == 0 {
            exp = 0;
            m[N - 1] = m0;
        } else {
            exp = 1;
            m[N - 1] = m1;
            m[N - 2] = m0;
        }

        (sign, exp, m)
    }
}

impl<const N: usize> SEM<N> for u128 {
    fn get_sem(self) -> (bool, i128, [usize; N]) {
        let (m0, m1) = (self as usize, (self >> 64) as usize);
        let mut m = [0; N];
        let exp: i128;
        if m1 == 0 {
            exp = 0;
            m[N - 1] = m0;
        } else {
            exp = 1;
            m[N - 1] = m1;
            m[N - 2] = m0;
        }

        (false, exp, m)
    }
}

impl<const N: usize> SEM<N> for isize {
    fn get_sem(self) -> (bool, i128, [usize; N]) {
        let sign = self < 0;
        let mut m = [0; N];
        m[N - 1] = self.unsigned_abs();
        (sign, 0, m)
    }
}

impl<const N: usize> SEM<N> for usize {
    fn get_sem(self) -> (bool, i128, [usize; N]) {
        let mut m = [0; N];
        m[N - 1] = self;
        (false, 0, m)
    }
}

macro_rules! impl_sem_iprim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> SEM<N> for $t{
                fn get_sem(self) -> (bool, i128, [usize;N]){
                    let sign = self < 0;
                    let mut m:[usize;N] = [0;N];
                    m[N-1] = self.unsigned_abs() as usize;
                    (sign, 0, m)
                }
            }
        )*
    };
}

macro_rules! impl_sem_uprim {
    ($($t:ty),*) => {
        $(
            impl<const N:usize> SEM<N> for $t{
                fn get_sem(self) -> (bool, i128, [usize;N]){
                    let mut m:[usize;N] = [0;N];
                    m[N-1] = self as usize;
                    (false, 0, m)
                }
            }
        )*
    };
}

impl_sem_iprim!(i64, i32, i16, i8);
impl_sem_uprim!(u64, u32, u16, u8);

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

    pub fn make(m: [usize; N], exp: i128, sign: bool) -> BitFloatStatic<N> {
        if N < 2 {
            panic!("Mantissa is not large enough")
        }

        BitFloatStatic::<N> { m, exp, sign }
    }

    pub fn from<T: SEM<N>>(val: T) -> BitFloatStatic<N> {
        if N < 2 {
            panic!("Mantissa is not large enough")
        }
        let (sign, exp, m) = val.get_sem();
        BitFloatStatic::<N> { m, exp, sign }
    }

    pub fn to_bis(self) -> BitIntStatic<N> {
        let data = self.abs_floor().m;
        return BitIntStatic::make(data, self.sign);
    }

    pub fn m_cmp(self, rhs: &BitFloatStatic<N>) -> std::cmp::Ordering {
        for (l, r) in self.m.iter().zip(&rhs.m).rev() {
            if l > r {
                return std::cmp::Ordering::Greater;
            } else if l < r {
                return std::cmp::Ordering::Less;
            }
        }

        return std::cmp::Ordering::Equal;
    }

    pub fn mut_abs(&mut self) {
        self.sign = false;
    }

    pub fn abs(&self) -> BitFloatStatic<N> {
        let mut out = *self;
        out.sign = false;
        out
    }

    pub fn mut_neg(&mut self) {
        self.sign ^= true;
    }

    pub fn abs_floor(&self) -> BitFloatStatic<N> {
        if self.exp < 0 {
            BitFloatStatic::<N> {
                m: [0; N],
                exp: i128::MIN,
                sign: false,
            }
        } else {
            let mut m = self.m;
            m[..N.saturating_sub(self.exp as usize + 1)].fill(0);
            BitFloatStatic::<N> {
                m,
                exp: self.exp,
                sign: self.sign,
            }
        }
    }

    pub fn floor(&self) -> BitFloatStatic<N> {
        let mut out = self.abs_floor();

        if self.sign {
            out -= make_int!(1, N);
        }

        out
    }

    pub fn ceil(&self) -> BitFloatStatic<N> {
        let mut out = self.abs_floor();

        if !self.sign {
            out += make_int!(1, N);
        }

        out
    }

    pub fn abs_frac(&self) -> BitFloatStatic<N> {
        (*self - self.abs_floor()).abs()
    }

    pub fn frac(&self) -> BitFloatStatic<N> {
        *self - self.floor()
    }

    pub fn round(&self) -> BitFloatStatic<N> {
        if self.exp < 0 {
            BitFloatStatic::<N> {
                m: [0; N],
                exp: i128::MIN,
                sign: false,
            }
        } else {
            let fract = self.abs_frac();
            let half = make_int!(1, N) >> 1;

            if fract < half {
                return self.floor();
            } else {
                return self.ceil();
            }
        }
    }
}

impl<const N: usize> PartialEq for BitFloatStatic<N> {
    fn eq(&self, other: &Self) -> bool {
        if self.sign ^ other.sign {
            return false;
        }

        if self.exp != other.exp {
            return false;
        }

        return self.m == other.m;
    }
}

impl<const N: usize> PartialOrd for BitFloatStatic<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.sign ^ other.sign {
            if self.sign {
                return Some(std::cmp::Ordering::Less);
            } else {
                return Some(std::cmp::Ordering::Greater);
            }
        }

        let exp_ord = self.exp.cmp(&other.exp);
        if exp_ord != std::cmp::Ordering::Equal {
            if self.sign {
                return Some(exp_ord.reverse());
            } else {
                return Some(exp_ord);
            }
        }

        let mut ord = std::cmp::Ordering::Equal;
        for (l, r) in self.m.iter().zip(&other.m).rev() {
            if l > r {
                ord = std::cmp::Ordering::Greater;
                break;
            } else if l < r {
                ord = std::cmp::Ordering::Less;
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

impl<const N: usize> Add for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn add(self, rhs: Self) -> Self::Output {
        let ish = self.exp - rhs.exp;
        let sh = ish.unsigned_abs();

        if sh >= N as u128 {
            if ish > 0 {
                return self;
            } else {
                return rhs;
            }
        }

        if self.sign ^ rhs.sign {
            let (mut m, mut exp, r, sign) = match ish.cmp(&0) {
                std::cmp::Ordering::Greater => (self.m, self.exp, &rhs.m, self.sign),
                std::cmp::Ordering::Less => (rhs.m, rhs.exp, &self.m, !self.sign),
                std::cmp::Ordering::Equal => match self.m_cmp(&rhs) {
                    std::cmp::Ordering::Greater => (self.m, self.exp, &rhs.m, self.sign),
                    std::cmp::Ordering::Less => (rhs.m, rhs.exp, &self.m, !self.sign),
                    std::cmp::Ordering::Equal => {
                        return BitFloatStatic::<N> {
                            m: [0; N],
                            exp: i128::MIN,
                            sign: false,
                        }
                    }
                },
            };

            sub_bfs::<N>(&mut m, r, sh as usize);

            if let Some(idx) = m.iter().rposition(|&x| x != 0) {
                m.rotate_right(N - idx - 1);
                exp -= (N - idx - 1) as i128;
            }

            return BitFloatStatic::<N> { m, exp, sign };
        } else {
            let (mut m, mut exp, r) = if ish > 0 {
                (self.m, self.exp, &rhs.m)
            } else {
                (rhs.m, rhs.exp, &self.m)
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
                sign: self.sign,
            };
        }
    }
}

impl<const N: usize> AddAssign for BitFloatStatic<N> {
    fn add_assign(&mut self, rhs: Self) {
        let ish = self.exp - rhs.exp;
        let sh = ish.unsigned_abs();

        if sh >= N as u128 {
            if ish > 0 {
                return;
            } else {
                *self = rhs;
                return;
            }
        }

        if self.sign ^ rhs.sign {
            let (mut m, mut exp, r, sign) = match ish.cmp(&0) {
                std::cmp::Ordering::Greater => (self.m, self.exp, &rhs.m, self.sign),
                std::cmp::Ordering::Less => (rhs.m, rhs.exp, &self.m, !self.sign),
                std::cmp::Ordering::Equal => match self.m_cmp(&rhs) {
                    std::cmp::Ordering::Greater => (self.m, self.exp, &rhs.m, self.sign),
                    std::cmp::Ordering::Less => (rhs.m, rhs.exp, &self.m, !self.sign),
                    std::cmp::Ordering::Equal => {
                        self.m.fill(0);
                        self.exp = i128::MIN;
                        self.sign = false;
                        return;
                    }
                },
            };

            sub_bfs::<N>(&mut m, r, sh as usize);

            if let Some(idx) = m.iter().rposition(|&x| x != 0) {
                m.rotate_right(N - idx - 1);
                exp -= (N - idx - 1) as i128;
            }

            *self = BitFloatStatic::<N> { m, exp, sign };
        } else {
            let (mut m, mut exp, r) = if ish > 0 {
                (self.m, self.exp, &rhs.m)
            } else {
                (rhs.m, rhs.exp, &self.m)
            };

            let carry = add_bfs::<N>(&mut m, r, sh as usize);
            if carry {
                m.rotate_left(1);
                m[N - 1] = 1;
                exp += 1;
            }
            *self = BitFloatStatic::<N> {
                m,
                exp,
                sign: self.sign,
            };
        }
    }
}

impl<const N: usize> Sub for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let ish = self.exp - rhs.exp;
        let sh = ish.unsigned_abs();

        if sh >= N as u128 {
            if ish > 0 {
                return self;
            } else {
                return rhs;
            }
        }

        if self.sign == rhs.sign {
            let (mut m, mut exp, r, sign) = match ish.cmp(&0) {
                std::cmp::Ordering::Greater => (self.m, self.exp, &rhs.m, self.sign),
                std::cmp::Ordering::Less => (rhs.m, rhs.exp, &self.m, !self.sign),
                std::cmp::Ordering::Equal => match self.m_cmp(&rhs) {
                    std::cmp::Ordering::Greater => (self.m, self.exp, &rhs.m, self.sign),
                    std::cmp::Ordering::Less => (rhs.m, rhs.exp, &self.m, !self.sign),
                    std::cmp::Ordering::Equal => {
                        return BitFloatStatic::<N> {
                            m: [0; N],
                            exp: i128::MIN,
                            sign: false,
                        }
                    }
                },
            };

            sub_bfs::<N>(&mut m, r, sh as usize);

            if let Some(idx) = m.iter().rposition(|&x| x != 0) {
                m.rotate_right(N - idx - 1);
                exp -= (N - idx - 1) as i128;
            }

            return BitFloatStatic::<N> { m, exp, sign };
        } else {
            let (mut m, mut exp, r) = if ish > 0 {
                (self.m, self.exp, &rhs.m)
            } else {
                (rhs.m, rhs.exp, &self.m)
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
                sign: self.sign,
            };
        }
    }
}

impl<const N: usize> SubAssign for BitFloatStatic<N> {
    fn sub_assign(&mut self, rhs: Self) {
        let ish = self.exp - rhs.exp;
        let sh = ish.unsigned_abs();

        if sh >= N as u128 {
            if ish > 0 {
                return;
            } else {
                *self = rhs;
                return;
            }
        }

        if self.sign == rhs.sign {
            let (mut m, mut exp, r, sign) = match ish.cmp(&0) {
                std::cmp::Ordering::Greater => (self.m, self.exp, &rhs.m, self.sign),
                std::cmp::Ordering::Less => (rhs.m, rhs.exp, &self.m, !self.sign),
                std::cmp::Ordering::Equal => match self.m_cmp(&rhs) {
                    std::cmp::Ordering::Greater => (self.m, self.exp, &rhs.m, self.sign),
                    std::cmp::Ordering::Less => (rhs.m, rhs.exp, &self.m, !self.sign),
                    std::cmp::Ordering::Equal => {
                        self.m.fill(0);
                        self.exp = i128::MIN;
                        self.sign = false;
                        return;
                    }
                },
            };

            sub_bfs::<N>(&mut m, r, sh as usize);

            if let Some(idx) = m.iter().rposition(|&x| x != 0) {
                m.rotate_right(N - idx - 1);
                exp -= (N - idx - 1) as i128;
            }

            *self = BitFloatStatic::<N> { m, exp, sign };
        } else {
            let (mut m, mut exp, r) = if ish > 0 {
                (self.m, self.exp, &rhs.m)
            } else {
                (rhs.m, rhs.exp, &self.m)
            };

            let carry = add_bfs::<N>(&mut m, r, sh as usize);
            if carry {
                m.rotate_left(1);
                m[N - 1] = 1;
                exp += 1;
            }

            *self = BitFloatStatic::<N> {
                m,
                exp,
                sign: self.sign,
            };
        }
    }
}

impl<const N: usize> Mul for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn mul(self, rhs: Self) -> Self::Output {
        let (mut m, carry) = mul_bfs::<N>(&self.m, &rhs.m);

        let mut exp = self.exp.saturating_add(rhs.exp);

        if carry > 0 {
            m.rotate_left(1);
            m[N - 1] = carry as usize;
            exp += 1;
        }

        BitFloatStatic::<N> {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl<const N: usize> MulAssign for BitFloatStatic<N> {
    fn mul_assign(&mut self, rhs: Self) {
        let (m, carry) = mul_bfs::<N>(&self.m, &rhs.m);
        self.m = m;
        self.exp += rhs.exp;

        if carry > 0 {
            self.m.rotate_left(1);
            self.m[N - 1] = carry as usize;
            self.exp += 1;
        }

        self.sign ^= rhs.sign;
    }
}

impl<const N: usize> Neg for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn neg(self) -> Self::Output {
        BitFloatStatic::<N> {
            m: self.m,
            exp: self.exp,
            sign: !self.sign,
        }
    }
}

impl<const N: usize> Div for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn div(self, rhs: Self) -> Self::Output {
        let mut n = self;
        div_bfs_gs(&mut n, rhs);
        n.sign = self.sign ^ rhs.sign;
        return n;
    }
}

impl<const N: usize> DivAssign for BitFloatStatic<N> {
    fn div_assign(&mut self, rhs: Self) {
        let sign = self.sign;
        div_bfs_gs(self, rhs);
        self.sign = sign ^ rhs.sign;
    }
}

impl<const N: usize, const M: usize> Shl<UBitIntStatic<M>> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn shl(self, rhs: UBitIntStatic<M>) -> Self::Output {
        let mut out = self;
        let (div, rem) = rhs.div_rem(64_usize);

        out.exp += div.to().unwrap() as i128;
        let sh = rem.to().unwrap() as u8;
        out.exp += shl_bfs::<N>(&mut out.m, sh);

        out
    }
}

impl<const N: usize, const M: usize> ShlAssign<UBitIntStatic<M>> for BitFloatStatic<N> {
    fn shl_assign(&mut self, rhs: UBitIntStatic<M>) {
        let (div, rem) = rhs.div_rem(64_usize);
        self.exp += div.to().unwrap() as i128;
        let sh = rem.to().unwrap() as u8;
        self.exp += shl_bfs::<N>(&mut self.m, sh);
    }
}

impl<const N: usize> Shl<u128> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn shl(self, rhs: u128) -> Self::Output {
        let mut out = self;
        out.exp += (rhs / 64) as i128;
        let sh = (rhs % 64) as u8;
        out.exp += shl_bfs::<N>(&mut out.m, sh);

        out
    }
}

impl<const N: usize> ShlAssign<u128> for BitFloatStatic<N> {
    fn shl_assign(&mut self, rhs: u128) {
        self.exp += (rhs / 64) as i128;
        let sh = (rhs % 64) as u8;
        self.exp += shl_bfs::<N>(&mut self.m, sh);
    }
}

impl<const N: usize, const M: usize> Shr<UBitIntStatic<M>> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn shr(self, rhs: UBitIntStatic<M>) -> Self::Output {
        let mut out = self;
        let (div, rem) = rhs.div_rem(64_usize);

        out.exp -= div.to().unwrap() as i128;
        let sh = rem.to().unwrap() as u8;
        out.exp += shr_bfs::<N>(&mut out.m, sh);

        out
    }
}

impl<const N: usize, const M: usize> ShrAssign<UBitIntStatic<M>> for BitFloatStatic<N> {
    fn shr_assign(&mut self, rhs: UBitIntStatic<M>) {
        let (div, rem) = rhs.div_rem(64_usize);
        self.exp -= div.to().unwrap() as i128;
        let sh = rem.to().unwrap() as u8;
        self.exp += shl_bfs::<N>(&mut self.m, sh);
    }
}

impl<const N: usize> Shr<u128> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn shr(self, rhs: u128) -> Self::Output {
        let mut out = self;

        out.exp -= (rhs / 64) as i128;
        let sh = (rhs % 64) as u8;
        out.exp += shr_bfs::<N>(&mut out.m, sh);

        out
    }
}

impl<const N: usize> ShrAssign<u128> for BitFloatStatic<N> {
    fn shr_assign(&mut self, rhs: u128) {
        self.exp -= (rhs / 64) as i128;
        let sh = (rhs % 64) as u8;
        self.exp += shr_bfs::<N>(&mut self.m, sh);
    }
}

pub trait PowI<RHS = Self> {
    type Output;
    fn powi(self, exp: RHS) -> Self::Output;
}

impl<const N: usize, const M: usize> PowI<BitIntStatic<M>> for BitFloatStatic<N> {
    type Output = BitFloatStatic<N>;

    fn powi(self, exp: BitIntStatic<M>) -> Self::Output {
        let uexp = exp.unsigned_abs();
        let out = powi_ubi(self, uexp);
        if exp.get_sign() {
            return make_int!(1, N) / out;
        } else {
            return out;
        }
    }
}
