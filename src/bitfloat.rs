use crate::bitfrac::BitFrac;
use crate::bitint::*;
use crate::ubitint::*;
use core::fmt;
use once_cell::sync::Lazy;
use std::arch::asm;
use std::cmp::*;
use std::isize;
use std::ops::*;
use std::usize;

#[inline]
fn add_bf(m1: &mut Vec<usize>, m2: &[usize], sh: usize) -> bool {
    let mut idx = m2.len() + sh;
    if idx <= m1.len() {
        let mut c = 0_usize;
        for (s, l) in m2.iter().rev().zip(m1[..idx].iter_mut().rev()) {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                asm!(
                    "adds {l}, {l}, {s}",
                    "cset {tmp}, cs",
                    "adds {l}, {l}, {c}",
                    "cset {c}, cs",
                    "orr {c}, {c}, {tmp}",
                    l = inout(reg) *l,
                    s = in(reg) *s,
                    c = inout(reg) c,
                    tmp = out(reg) _
                )
            }
        }

        for l in m1[..idx - m2.len()].iter_mut().rev() {
            if c == 0 {
                break;
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                asm!(
                    "adds {l}, {l}, {c}", // l + c -> l
                    "cset {c}, cs",       // put overflow of l + c into c
                    l = inout(reg) *l,
                    c = inout(reg) c,
                );
            }
        }

        if let Some(idx) = m1.iter().rposition(|&x| x != 0) {
            m1.truncate(idx + 1);
        } else {
            m1.clear();
        }

        if c == 1 {
            m1.insert(0, 1);
            return true;
        }

        return false;
    } else {
        if sh > m1.len() {
            idx = sh - m1.len();
            m1.extend(std::iter::repeat(0).take(idx));
            m1.extend(m2.iter().cloned());
            return false;
        } else {
            idx = m1.len() - sh;
            let mut c = 0_usize;
            for (s, l) in m2[..idx].iter().rev().zip(m1.iter_mut().rev()) {
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    asm!(
                        "adds {l}, {l}, {s}",
                        "cset {tmp}, cs",
                        "adds {l}, {l}, {c}",
                        "cset {c}, cs",
                        "orr {c}, {c}, {tmp}",
                        l = inout(reg) *l,
                        s = in(reg) *s,
                        c = inout(reg) c,
                        tmp = out(reg) _
                    )
                }
            }

            for l in m1[..sh].iter_mut().rev() {
                if c == 0 {
                    break;
                }
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    asm!(
                        "adds {l}, {l}, {c}", // l + c -> l
                        "cset {c}, cs",       // put overflow of l + c into c
                        l = inout(reg) *l,
                        c = inout(reg) c,
                    );
                }
            }

            m1.extend(m2[idx..].iter().cloned());

            if c == 1 {
                m1.insert(0, 1);
                return true;
            }

            return false;
        }
    }
}

#[inline]
pub fn sub_bf(m1: &mut Vec<usize>, m2: &[usize], sh: usize) -> usize {
    let mut idx = m2.len() + sh;
    if idx <= m1.len() {
        let mut c = 1_usize;
        for (s, l) in m2.iter().rev().zip(m1[..idx].iter_mut().rev()) {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                asm!(
                    "adds {l}, {l}, {s}",
                    "cset {tmp}, cs",
                    "adds {l}, {l}, {c}",
                    "cset {c}, cs",
                    "orr {c}, {c}, {tmp}",
                    l = inout(reg) *l,
                    s = in(reg) !(*s),
                    c = inout(reg) c,
                    tmp = out(reg) _
                )
            }
        }
        if c == 0 {
            for l in m1[..idx - m2.len()].iter_mut().rev() {
                c ^= 1;
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    asm!(
                        "subs {l}, {l}, {c}",
                        "cset {c}, cs",
                        l = inout(reg) *l,
                        c = inout(reg) c
                    )
                }
            }
        }

        if let Some(idx) = m1.iter().rposition(|&x| x != 0) {
            m1.truncate(idx + 1);
        } else {
            m1.clear();
        }

        if let Some(idx) = m1.iter().position(|&x| x != 0) {
            m1.drain(..idx);
            return idx;
        } else {
            m1.clear();
            return 0;
        }
    } else {
        if sh > m1.len() {
            idx = sh - m1.len();
            let mut m1_len = m1.len();
            m1[m1_len - 1] -= 1;
            m1.extend(std::iter::repeat(usize::MAX).take(idx));
            m1.extend(m2.iter().map(|&x| !x));
            m1_len = m1.len();
            m1[m1_len - 1] += 1;
            return 0;
        } else {
            idx = m1.len() - sh;
            let mut c = 0_usize;
            for (s, l) in m2[..idx].iter().rev().zip(m1.iter_mut().rev()) {
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    asm!(
                        "adds {l}, {l}, {s}",
                        "cset {tmp}, cs",
                        "adds {l}, {l}, {c}",
                        "cset {c}, cs",
                        "orr {c}, {c}, {tmp}",
                        l = inout(reg) *l,
                        s = in(reg) !(*s),
                        c = inout(reg) c,
                        tmp = out(reg) _
                    )
                }
            }

            if c == 0 {
                for l in m1[..sh].iter_mut().rev() {
                    c ^= 1;
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        asm!(
                            "subs {l}, {l}, {c}",
                            "cset {c}, cs",
                            l = inout(reg) *l,
                            c = inout(reg) c
                        )
                    }
                }
            }

            m1.extend(m2[idx..].iter().map(|&x| !x));
            let m1_len = m1.len();
            m1[m1_len - 1] += 1;

            if let Some(idx) = m1.iter().position(|&x| x != 0) {
                m1.drain(..idx);
                return idx;
            } else {
                m1.clear();
                return 0;
            }
        }
    }
}

#[inline]
fn mul_bf(m1: &[usize], m2: &[usize], acc: usize) -> (Vec<usize>, bool) {
    if m2.is_empty() || m1.is_empty() {
        return (vec![], false);
    }
    let m1_len = m1.len() - 1;
    let m2_len = m2.len() - 1;

    let mut out: Vec<usize> = Vec::with_capacity(m1_len + m2_len + 1);

    let usz = (USZ_MEM * 8) as u128;
    let mask = (1u128 << usz) - 1;

    let mut carry: u128 = 0;
    for i in acc..=(m1_len + m2_len) {
        let mut term: u128 = carry;
        let mut next_carry: u128 = 0;
        for j in i.saturating_sub(m2_len)..=i.min(m1_len) {
            let val = (m1[m1_len - j] as u128) * (m2[m2_len - (i - j)] as u128);
            term += val & mask;
            next_carry += val >> usz;
        }
        carry = next_carry + (term >> usz);
        out.push(term as usize);
    }

    let mut is_carry = false;
    if carry > 0 {
        out.push(carry as usize);
        is_carry = true;
    }
    out.reverse();

    if let Some(idx) = out.iter().rposition(|&x| x != 0) {
        out.truncate(idx + 1);
    }

    (out, is_carry)
}

#[inline]
fn shl_bf_sup(m: &mut Vec<usize>, sh: usize) -> bool {
    let mv_sz = USZ_MEM * 8 - sh;
    let mut carry = 0_usize;
    for elem in m.iter_mut().rev() {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            asm!(
                "lsr {tmp}, {e}, {ms}",
                "lsl {e}, {e}, {sh}",
                "orr {e}, {e}, {c}",
                "mov {c}, {tmp}",
                e = inout(reg) *elem,
                c = inout(reg) carry,
                ms = in(reg) mv_sz,
                sh = in(reg) sh,
                tmp = out(reg) _,
            )
        }
    }

    let mut out = false;
    if carry > 0 {
        m.insert(0, carry);
        out = true;
    }
    if m[m.len() - 1] == 0 {
        m.pop();
    }
    return out;
}

pub static TWO: Lazy<BitFloat> = Lazy::new(|| BitFloat {
    m: vec![2],
    exp: 0,
    sign: false,
});

#[inline]
fn stop_crit(m: &[usize], acc: usize) -> bool {
    for e in m[1..acc.min(m.len())].iter() {
        if *e != 0 {
            return true;
        }
    }

    return false;
}

#[inline]
fn div_bf_gs(n: &mut BitFloat, d: &mut BitFloat) {
    let acc = (n.m.len().max(d.m.len()) + 1).max(3);
    n.exp -= d.exp + 1;
    d.exp = -1;
    let sh = d.m[0].leading_zeros();
    shl_bf_sup(&mut d.m, sh as usize);
    n.exp += shl_bf_sup(&mut n.m, sh as usize) as isize;
    let n_sign = n.sign;
    let d_sign = d.sign;
    n.abs();
    d.abs();

    let mut f = &*TWO - &*d;
    while stop_crit(&f.m, acc) {
        d.stbl_mul_assign(&f, acc);
        n.stbl_mul_assign(&f, acc);
        f = &*TWO - &*d;
    }

    n.sign = n_sign ^ d_sign;
}

pub static ONE: Lazy<BitFloat> = Lazy::new(|| BitFloat {
    m: vec![1],
    exp: 0,
    sign: false,
});

#[inline]
fn powi_ubi(x: BitFloat, n: UBitInt) -> BitFloat {
    if n == 0_usize {
        return (*ONE).clone();
    } else if n == 1_usize {
        return x.clone();
    } else if n.mod2() {
        return &x * powi_ubi(&x * &x, n >> 1_usize);
    } else {
        return powi_ubi(&x * &x, n >> 1_usize);
    }
}

#[inline]
fn powi_prim(x: BitFloat, n: u128) -> BitFloat {
    if n == 0 {
        return (*ONE).clone();
    } else if n == 1 {
        return x.clone();
    } else if n % 2 == 1 {
        return &x * powi_prim(&x * &x, n / 2);
    } else {
        return powi_prim(&x * &x, n / 2);
    }
}

pub static HARMONICS: [Lazy<BitFloat>; 34] = [
    Lazy::new(|| BitFloat {
        m: vec![9223372036854775808],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            6148914691236517205,
            6148914691236517205,
            6148914691236517205,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![4611686018427387904],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            3689348814741910323,
            3689348814741910323,
            3689348814741910323,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            3074457345618258602,
            12297829382473034410,
            12297829382473034410,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            2635249153387078802,
            5270498306774157604,
            10540996613548315209,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![2305843009213693952],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            2049638230412172401,
            14347467612885206812,
            8198552921648689607,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            1844674407370955161,
            11068046444225730969,
            11068046444225730969,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            1676976733973595601,
            8384883669867978007,
            5030930201920786804,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            1537228672809129301,
            6148914691236517205,
            6148914691236517205,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            1418980313362273201,
            4256940940086819603,
            12770822820260458811,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            1317624576693539401,
            2635249153387078802,
            5270498306774157604,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            1229782938247303441,
            1229782938247303441,
            1229782938247303441,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![1152921504606846976],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            1085102592571150095,
            1085102592571150095,
            1085102592571150095,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            1024819115206086200,
            16397105843297379214,
            4099276460824344803,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            970881267037344821,
            16504981539634861972,
            3883525068149379287,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            922337203685477580,
            14757395258967641292,
            14757395258967641292,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            878416384462359600,
            14054662151397753612,
            3513665537849438403,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            838488366986797800,
            13415813871788764811,
            11738837137815169210,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            802032351030850070,
            4812194106185100421,
            10426420563401050913,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            768614336404564650,
            12297829382473034410,
            12297829382473034410,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            737869762948382064,
            11805916207174113034,
            4427218577690292387,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            709490156681136600,
            11351842506898185609,
            15608783446985005213,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            683212743470724133,
            17080318586768103348,
            2732850973882896535,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            658812288346769700,
            10540996613548315209,
            2635249153387078802,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            636094623231363848,
            15266270957552732371,
            15902365580784096220,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![614891469123651720, 9838263505978427528, 9838263505978427528],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![595056260442243600, 9520900167075897608, 4760450083537948804],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![576460752303423488],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            558992244657865200,
            8943875914525843207,
            13974806116446630012,
        ],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![542551296285575047, 9765923333140350855, 9765923333140350855],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![527049830677415760, 8432797290838652167, 5797548137451573365],
        exp: -1,
        sign: false,
    }),
];

#[inline]
fn pow_pow_2(x: &BitFloat, n: u128, acc: usize) -> BitFloat {
    let mut out = x.clone();
    for _ in 0..n {
        out = (&out).stbl_mul(&out, acc);
    }

    out
}

pub static EXP: Lazy<BitFloat> = Lazy::new(|| BitFloat {
    m: vec![2, 13249961062380150784],
    exp: 0,
    sign: false,
});

#[inline]
pub fn exp_bf(x: &BitFloat) -> BitFloat {
    let sz = x.m.len().max(3);
    let acc = ((x.exp as i128 - x.m.len() as i128 + 1) as isize).min(-1);
    if x.exp < 0 {
        let mut out = &*ONE + x;
        let mut term = x.clone();

        let mut flag = true;
        for i in 0..34 {
            term.stbl_mul_assign(x.stbl_mul(&*HARMONICS[i], sz), sz);
            out += &term;
            if term.exp < acc {
                flag = false;
                break;
            }
        }

        if flag {
            let i: usize = 36;
            while term.exp >= acc {
                let div = BitFloat {
                    m: vec![i],
                    exp: 0,
                    sign: false,
                };
                term.stbl_mul_assign(x, sz);
                term /= div;
                out += &term;
            }
        }

        return out;
    } else {
        let mut val = x.clone();
        val.abs();
        val.exp = 0;
        let sh = (val.m[0].leading_zeros() + 1) as usize;
        shl_bf_sup(&mut val.m, sh);

        let mut sum = val.clone();
        val -= &*ONE;
        let mut term = val.clone();

        let mut flag = true;
        for i in 0..34 {
            term.stbl_mul_assign((&*HARMONICS[i]).stbl_mul(&val, sz), sz);
            sum += &term;

            if term.exp < acc {
                flag = false;
                break;
            }
        }

        if flag {
            let mut i: usize = 36;
            while term.exp >= acc {
                let div = BitFloat {
                    m: vec![i],
                    exp: 0,
                    sign: false,
                };
                term.stbl_mul_assign(&val, sz);
                term /= div;
                sum += &term;
                i += 1;
            }
        }

        sum.stbl_mul_assign(&*EXP, sz);
        let out = pow_pow_2(&sum, 64 * (x.exp as u128 + 1) - sh as u128, sz);

        if x.sign {
            return &*ONE / out;
        } else {
            return out;
        }
    }
}

pub static LN2: Lazy<BitFloat> = Lazy::new(|| BitFloat {
    m: vec![12786308645202655232],
    exp: -1,
    sign: false,
});

#[inline]
pub fn ln_bf(x: &BitFloat) -> BitFloat {
    let sz = x.m.len().max(3);
    let acc = ((x.exp as i128 - x.m.len() as i128 + 1) as isize).min(-1);
    let mut val = x.clone();
    val.abs();
    val.exp = 0;
    let sh = (val.m[0].leading_zeros() + 1) as usize;
    shl_bf_sup(&mut val.m, sh);
    val -= &*ONE;
    val /= &*TWO + &val;
    let mut sum = val.clone();
    let mut term = val.clone();
    val = (&val).stbl_mul(&val, sz);

    let mut flag = true;
    for i in 0..17 {
        term.stbl_mul_assign(&val, sz);
        let add = (&term).stbl_mul(&*HARMONICS[2 * i + 1], sz);
        sum += &add;
        if add.exp < acc {
            flag = false;
            break;
        }
    }

    if flag {
        let mut i: usize = 37;
        let mut add_exp = 0;
        while add_exp >= acc {
            let div = BitFloat {
                m: vec![i],
                exp: 0,
                sign: false,
            };
            term.stbl_mul_assign(&val, sz);
            let add = &term / div;
            add_exp = add.exp;
            sum += add;

            i += 2
        }
    }

    sum *= &*TWO;
    sum += BitFloat::from(64 * (x.exp as i128 + 1) - sh as i128) * &*LN2;

    return sum;
}

pub fn sqrt_bf(val: &BitFloat, acc: isize) -> BitFloat {
    let mut x = val.clone();

    loop {
        let term: BitFloat = (val / &x - &x) / 2.0;
        x += &term;
        if term.exp < acc {
            break;
        }
    }
    x
}

#[derive(Debug, Clone, PartialEq)]
pub struct BitFloat {
    pub m: Vec<usize>,
    pub exp: isize,
    pub sign: bool,
}

pub trait SignExpM {
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>);
}

impl SignExpM for f64 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        let sign = self < 0.0;
        let bits: u64 = self.to_bits();
        let mut exp_2 = (bits >> 52) as isize;
        exp_2 &= 0x7FF;
        exp_2 -= 1023;
        let mut m_2 = (bits & 0xFFFFFFFFFFFFF) as u128;
        m_2 |= 0x10000000000000;

        let exp = if exp_2 >= 0 {
            exp_2 / (USZ_MEM * 8) as isize
        } else {
            exp_2 / (USZ_MEM * 8) as isize - 1
        };

        let mut sh = 12 + exp_2 % 64;
        if sh < 0 {
            sh += 64;
        }
        m_2 <<= sh;

        let mut m = if USZ_MEM == 8 {
            let lower = (m_2 & 0xFFFFFFFFFFFFFFFF) as usize;
            let upper = (m_2 >> 64) as usize;
            vec![upper, lower]
        } else {
            let lowest = (m_2 & 0xFFFFFFFF) as usize;
            let next_lowest = ((m_2 >> 32) & 0xFFFFFFFF) as usize;
            let next_highest = ((m_2 >> 64) & 0xFFFFFFFF) as usize;
            let highest = ((m_2 >> 96) & 0xFFFFFFFF) as usize;
            vec![highest, next_highest, next_lowest, lowest]
        };

        if let Some(idx) = m.iter().rposition(|&x| x != 0) {
            m.truncate(idx + 1);
        } else {
            m.clear();
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        (sign, exp, m)
    }
}

impl SignExpM for f32 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        let sign = self < 0.0;
        let bits: u32 = self.to_bits();
        let mut exp_2 = (bits >> 23) as isize;
        exp_2 &= 0xFF;

        let mut m_2 = (bits & 0x7FFFFF) as u64;
        m_2 |= 0x800000;

        let exp = if exp_2 >= 0 {
            exp_2 / (USZ_MEM * 8) as isize
        } else {
            exp_2 / (USZ_MEM * 8) as isize - 1
        };

        let mut sh = 9 + exp_2 % 32;
        if sh < 0 {
            sh += 32;
        }
        m_2 <<= sh;

        let m = if USZ_MEM == 8 {
            if m_2 == 0 {
                vec![]
            } else {
                vec![m_2 as usize]
            }
        } else {
            let low = (m_2 & 0xFFFFFFFF) as usize;
            let high = ((m_2 >> 32) & 0xFFFFFFFF) as usize;
            let mut out: Vec<usize> = vec![];
            if high != 0 {
                out.push(high);
            }
            if low != 0 {
                out.push(low);
            }

            out
        };

        (sign, exp, m)
    }
}

impl SignExpM for u128 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        let (mut m, mut exp) = if USZ_MEM == 8 {
            let lower = (self & 0xFFFFFFFFFFFFFFFF) as usize;
            let upper = (self >> 64) as usize;
            (vec![upper, lower], 1_isize)
        } else {
            let lowest = (self & 0xFFFFFFFF) as usize;
            let next_lowest = ((self >> 32) & 0xFFFFFFFF) as usize;
            let next_highest = ((self >> 64) & 0xFFFFFFFF) as usize;
            let highest = ((self >> 96) & 0xFFFFFFFF) as usize;
            (vec![highest, next_highest, next_lowest, lowest], 3_isize)
        };

        if let Some(idx) = m.iter().rposition(|&x| x != 0) {
            m.truncate(idx + 1);
        } else {
            m.clear();
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
            exp -= idx as isize;
        } else {
            m.clear();
            exp = 0;
        }

        (false, exp, m)
    }
}

impl SignExpM for i128 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        let m_2 = self.unsigned_abs();

        let (mut m, mut exp) = if USZ_MEM == 8 {
            let lower = (m_2 & 0xFFFFFFFFFFFFFFFF) as usize;
            let upper = (m_2 >> 64) as usize;
            (vec![upper, lower], 1_isize)
        } else {
            let lowest = (m_2 & 0xFFFFFFFF) as usize;
            let next_lowest = ((m_2 >> 32) & 0xFFFFFFFF) as usize;
            let next_highest = ((m_2 >> 64) & 0xFFFFFFFF) as usize;
            let highest = ((m_2 >> 96) & 0xFFFFFFFF) as usize;
            (vec![highest, next_highest, next_lowest, lowest], 3_isize)
        };

        if let Some(idx) = m.iter().rposition(|&x| x != 0) {
            m.truncate(idx + 1);
        } else {
            m.clear();
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
            exp -= idx as isize;
        } else {
            m.clear();
            exp = 0;
        }

        (self < 0, exp, m)
    }
}

impl SignExpM for u64 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        let (m, exp) = if USZ_MEM == 8 {
            if self == 0 {
                (vec![], 0_isize)
            } else {
                (vec![self as usize], 0_isize)
            }
        } else {
            let low = (self & 0xFFFFFFFF) as usize;
            let high = ((self >> 32) & 0xFFFFFFFF) as usize;
            let mut out: Vec<usize> = vec![];
            if high != 0 {
                out.push(high);
            }
            if low != 0 {
                out.push(low);
            }
            let exp = (out.len() - 1) as isize;
            (out, exp)
        };

        (false, exp, m)
    }
}

impl SignExpM for i64 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        let m_2 = self.unsigned_abs();
        let (m, exp) = if USZ_MEM == 8 {
            if m_2 == 0 {
                (vec![], 0_isize)
            } else {
                (vec![m_2 as usize], 0_isize)
            }
        } else {
            let low = (m_2 & 0xFFFFFFFF) as usize;
            let high = ((m_2 >> 32) & 0xFFFFFFFF) as usize;
            let mut out: Vec<usize> = vec![];
            if high != 0 {
                out.push(high);
            }
            if low != 0 {
                out.push(low);
            }
            let exp = (out.len() - 1) as isize;
            (out, exp)
        };

        (self < 0, exp, m)
    }
}

impl SignExpM for usize {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        if self != 0 {
            (false, 0, vec![self])
        } else {
            (false, 0, vec![])
        }
    }
}

impl SignExpM for isize {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        if self != 0 {
            (self < 0, 0, vec![self.unsigned_abs()])
        } else {
            (false, 0, vec![])
        }
    }
}

impl SignExpM for u32 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        if self != 0 {
            (false, 0, vec![self as usize])
        } else {
            (false, 0, vec![])
        }
    }
}

impl SignExpM for i32 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        if self != 0 {
            (self < 0, 0, vec![self.unsigned_abs() as usize])
        } else {
            (false, 0, vec![])
        }
    }
}

impl SignExpM for u16 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        if self != 0 {
            (false, 0, vec![self as usize])
        } else {
            (false, 0, vec![])
        }
    }
}

impl SignExpM for i16 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        if self != 0 {
            (self < 0, 0, vec![self.unsigned_abs() as usize])
        } else {
            (false, 0, vec![])
        }
    }
}

impl SignExpM for u8 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        if self != 0 {
            (false, 0, vec![self as usize])
        } else {
            (false, 0, vec![])
        }
    }
}

impl SignExpM for i8 {
    #[inline]
    fn sign_exp_m(self) -> (bool, isize, Vec<usize>) {
        if self != 0 {
            (self < 0, 0, vec![self.unsigned_abs() as usize])
        } else {
            (false, 0, vec![])
        }
    }
}

impl BitFloat {
    pub fn get_m(&self) -> Vec<usize> {
        self.m.clone()
    }

    pub fn get_exp(&self) -> isize {
        self.exp
    }

    pub fn get_sign(&self) -> bool {
        self.sign
    }

    #[inline]
    pub fn from<T: SignExpM>(val: T) -> BitFloat {
        let (sign, exp, m) = val.sign_exp_m();
        BitFloat { sign, exp, m }
    }

    #[inline]
    pub fn from_bi(val: BitInt) -> BitFloat {
        let mut m = val.val.data;
        let exp = m.len().saturating_sub(1) as isize;
        m.reverse();

        if let Some(idx) = m.iter().rposition(|&x| x != 0) {
            m.truncate(idx + 1);
        } else {
            m.clear();
        }

        BitFloat {
            m,
            exp,
            sign: val.sign,
        }
    }

    #[inline]
    pub fn from_bfr(val: BitFrac) -> BitFloat {
        let n = BitFloat::from_bi(BitInt {
            val: val.n,
            sign: false,
        });

        let d = BitFloat::from_bi(BitInt {
            val: val.d,
            sign: false,
        });

        if val.sign {
            -n / d
        } else {
            n / d
        }
    }

    pub fn from_str(val: &str) -> Result<BitFloat, String> {
        if let Some(idx) = val.find('.') {
            let mut mul = (&*TEN).powi((idx - 1) as isize);
            let mut sum = BitFloat::from(0);
            for c in val.chars() {
                if let Some(dig) = c.to_digit(10) {
                    sum += BitFloat::from(dig) * &mul;
                    mul /= &*TEN;
                } else {
                    if c == '.' {
                        continue;
                    } else {
                        return Err("here 1 malformed string as an input".to_string());
                    }
                }
            }

            return Ok(sum);
        } else {
            return Err("here 2 malformed string as an input".to_string());
        }
    }

    #[inline]
    pub fn to(&self) -> Result<f64, String> {
        if self.exp > 1023 / (USZ_MEM * 8) as isize || self.exp < -1022 / (USZ_MEM * 8) as isize {
            return Err("BitFloat is to large or small for f64".to_string());
        }

        let mut out = 0.0;
        let base = (1_u128 << 64) as f64;
        let mut mul = base.powi(self.exp as i32);
        for elem in &self.m {
            out += (*elem as f64) * mul;
            mul /= base;
        }

        if self.sign {
            Ok(-out)
        } else {
            Ok(out)
        }
    }

    #[inline]
    pub fn abs_cmp(&self, other: &BitFloat) -> Ordering {
        use Ordering::*;
        let exp_cmp = self.exp.cmp(&other.exp);
        if exp_cmp != Equal {
            return exp_cmp;
        }

        for (l, r) in self.m.iter().zip(&other.m) {
            if l > r {
                return Greater;
            } else if l < r {
                return Less;
            }
        }

        return self.m.len().cmp(&other.m.len());
    }

    #[inline]
    pub fn abs(&mut self) {
        self.sign = false;
    }

    #[inline]
    pub fn neg(&mut self) {
        self.sign = !self.sign;
    }
}

pub static LN10: Lazy<BitFloat> = Lazy::new(|| BitFloat {
    m: vec![2, 5581709770980769792],
    exp: 0,
    sign: false,
});

pub static TEN: Lazy<BitFloat> = Lazy::new(|| BitFloat {
    m: vec![10],
    exp: 0,
    sign: false,
});

impl fmt::Display for BitFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sz = self.ln() / &*LN10;
        let exp: String;
        if sz.exp >= 0 {
            let mut exp_ubi_data = sz.m[..(sz.exp as usize + 1)].to_vec();
            exp_ubi_data.reverse();
            let mut exp_bi = BitInt {
                val: UBitInt { data: exp_ubi_data },
                sign: sz.sign,
            };
            if sz.sign {
                exp_bi -= 1;
            }
            exp = format!(" E {}", exp_bi.to_string());
        } else {
            exp = "".to_string();
        }

        let mut val = self.clone();
        val.exp = 0;

        let mut out = val.m[0].to_string();
        if let Some((index, _)) = out.bytes().enumerate().nth(1) {
            out.insert(index, '.');
        } else {
            out.push('.');
        }

        let mut sub = val.m[0] as u128 * 10;
        val *= 10.0;
        val -= BitFloat::from(sub);

        while !val.m.is_empty() {
            if val.exp < 0 {
                out = format!("{}{}", out, 0.to_string());
                val *= &*TEN;
            } else {
                out = format!("{}{}", out, val.m[0].to_string());
                sub = val.m[0] as u128 * 10;
                val *= &*TEN;
                val -= BitFloat::from(sub);
            };
        }

        out = format!("{}{}", out, exp);

        write!(f, "{}", out)?;
        Ok(())
    }
}

impl PartialEq<f64> for BitFloat {
    #[inline]
    fn eq(&self, other: &f64) -> bool {
        return *self == BitFloat::from(*other);
    }
}

impl PartialEq<f32> for BitFloat {
    #[inline]
    fn eq(&self, other: &f32) -> bool {
        return *self == BitFloat::from(*other);
    }
}

impl PartialOrd for BitFloat {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Ordering::*;
        if self.sign ^ other.sign {
            if self.sign {
                return Some(Less);
            } else {
                return Some(Greater);
            }
        }

        let cmp_exp = self.exp.cmp(&other.exp);
        if cmp_exp != Equal {
            if self.sign {
                return Some(cmp_exp.reverse());
            } else {
                return Some(cmp_exp);
            }
        }

        let mut cmp_m = Equal;
        for (l, r) in self.m.iter().zip(&other.m) {
            if l > r {
                cmp_m = Greater;
            } else if r < l {
                cmp_m = Less;
            }
        }

        if self.sign {
            return Some(cmp_m.reverse());
        } else {
            return Some(cmp_m);
        }
    }
}

impl PartialOrd<f64> for BitFloat {
    #[inline]
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.partial_cmp(&BitFloat::from(*other))
    }
}

impl PartialOrd<f32> for BitFloat {
    #[inline]
    fn partial_cmp(&self, other: &f32) -> Option<Ordering> {
        self.partial_cmp(&BitFloat::from(*other))
    }
}

impl Eq for BitFloat {}

impl Ord for BitFloat {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        use Ordering::*;
        if self.sign ^ other.sign {
            if self.sign {
                return Less;
            } else {
                return Greater;
            }
        }

        let cmp_exp = self.exp.cmp(&other.exp);
        if cmp_exp != Equal {
            if self.sign {
                return cmp_exp.reverse();
            } else {
                return cmp_exp;
            }
        }

        let mut cmp_m = Equal;
        for (l, r) in self.m.iter().zip(&other.m) {
            if l > r {
                cmp_m = Greater;
            } else if r < l {
                cmp_m = Less;
            }
        }

        if self.sign {
            return cmp_m.reverse();
        } else {
            return cmp_m;
        }
    }
}

impl Add for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            return rhs;
        }
        if rhs.m.is_empty() {
            return self;
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                let mut m = self.m;
                let mut exp = self.exp;
                if self.sign ^ rhs.sign {
                    exp -= sub_bf(&mut m, &rhs.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: self.sign,
                }
            }
            Less => {
                let mut m = rhs.m;
                let mut exp = rhs.exp;
                if self.sign ^ rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: rhs.sign,
                }
            }
            Equal => {
                if self.sign ^ rhs.sign {
                    return BitFloat {
                        sign: false,
                        exp: 0,
                        m: vec![],
                    };
                } else {
                    let mut m = self.m;
                    let mut exp = self.exp;
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                    BitFloat {
                        exp,
                        m,
                        sign: self.sign,
                    }
                }
            }
        }
    }
}

impl Add for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            return rhs.clone();
        }
        if rhs.m.is_empty() {
            return self.clone();
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                let mut m = self.m.clone();
                let mut exp = self.exp;
                if self.sign ^ rhs.sign {
                    exp -= sub_bf(&mut m, &rhs.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: self.sign,
                }
            }
            Less => {
                let mut m = rhs.m.clone();
                let mut exp = rhs.exp;
                if self.sign ^ rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: rhs.sign,
                }
            }
            Equal => {
                if self.sign ^ rhs.sign {
                    return BitFloat {
                        sign: false,
                        exp: 0,
                        m: vec![],
                    };
                } else {
                    let mut m = self.m.clone();
                    let mut exp = self.exp;
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                    BitFloat {
                        exp,
                        m,
                        sign: self.sign,
                    }
                }
            }
        }
    }
}

impl Add<&BitFloat> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: &BitFloat) -> Self::Output {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            return rhs.clone();
        }
        if rhs.m.is_empty() {
            return self;
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                let mut m = self.m;
                let mut exp = self.exp;
                if self.sign ^ rhs.sign {
                    exp -= sub_bf(&mut m, &rhs.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: self.sign,
                }
            }
            Less => {
                let mut m = rhs.m.clone();
                let mut exp = rhs.exp;
                if self.sign ^ rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: rhs.sign,
                }
            }
            Equal => {
                if self.sign ^ rhs.sign {
                    return BitFloat {
                        sign: false,
                        exp: 0,
                        m: vec![],
                    };
                } else {
                    let mut m = self.m;
                    let mut exp = self.exp;
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                    BitFloat {
                        exp,
                        m,
                        sign: self.sign,
                    }
                }
            }
        }
    }
}

impl Add<BitFloat> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: BitFloat) -> Self::Output {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            return rhs;
        }
        if rhs.m.is_empty() {
            return self.clone();
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                let mut m = self.m.clone();
                let mut exp = self.exp;
                if self.sign ^ rhs.sign {
                    exp -= sub_bf(&mut m, &rhs.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: self.sign,
                }
            }
            Less => {
                let mut m = rhs.m;
                let mut exp = rhs.exp;
                if self.sign ^ rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: rhs.sign,
                }
            }
            Equal => {
                if self.sign ^ rhs.sign {
                    return BitFloat {
                        sign: false,
                        exp: 0,
                        m: vec![],
                    };
                } else {
                    let mut m = self.m.clone();
                    let mut exp = self.exp;
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                    BitFloat {
                        exp,
                        m,
                        sign: self.sign,
                    }
                }
            }
        }
    }
}

impl Add<f64> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        self + BitFloat::from(rhs)
    }
}

impl Add<f64> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        self + BitFloat::from(rhs)
    }
}

impl Add<BitFloat> for f64 {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: BitFloat) -> Self::Output {
        BitFloat::from(self) + rhs
    }
}

impl Add<&BitFloat> for f64 {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: &BitFloat) -> Self::Output {
        BitFloat::from(self) + rhs
    }
}

impl Add<f32> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        self + BitFloat::from(rhs)
    }
}

impl Add<f32> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        self + BitFloat::from(rhs)
    }
}

impl Add<BitFloat> for f32 {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: BitFloat) -> Self::Output {
        BitFloat::from(self) + rhs
    }
}

impl Add<&BitFloat> for f32 {
    type Output = BitFloat;

    #[inline]
    fn add(self, rhs: &BitFloat) -> Self::Output {
        BitFloat::from(self) + rhs
    }
}

impl AddAssign for BitFloat {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            *self = rhs;
            return;
        }
        if rhs.m.is_empty() {
            return;
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                if self.sign ^ rhs.sign {
                    self.exp -= sub_bf(&mut self.m, &rhs.m, sh) as isize;
                } else {
                    self.exp += add_bf(&mut self.m, &rhs.m, sh) as isize;
                }
            }
            Less => {
                let mut m = rhs.m.clone();
                let mut exp = rhs.exp;
                if self.sign ^ rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                *self = BitFloat {
                    exp,
                    m,
                    sign: rhs.sign,
                };
            }
            Equal => {
                if self.sign ^ rhs.sign {
                    self.sign = false;
                    self.exp = 0;
                    self.m.clear();
                } else {
                    self.exp += add_bf(&mut self.m, &rhs.m, sh) as isize;
                }
            }
        }
    }
}

impl AddAssign<&BitFloat> for BitFloat {
    #[inline]
    fn add_assign(&mut self, rhs: &BitFloat) {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            *self = rhs.clone();
            return;
        }
        if rhs.m.is_empty() {
            return;
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                if self.sign ^ rhs.sign {
                    self.exp -= sub_bf(&mut self.m, &rhs.m, sh) as isize;
                } else {
                    self.exp += add_bf(&mut self.m, &rhs.m, sh) as isize;
                }
            }
            Less => {
                let mut m = rhs.m.clone();
                let mut exp = rhs.exp;
                if self.sign ^ rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                *self = BitFloat {
                    exp,
                    m,
                    sign: rhs.sign,
                };
            }
            Equal => {
                if self.sign ^ rhs.sign {
                    self.sign = false;
                    self.exp = 0;
                    self.m.clear();
                } else {
                    self.exp += add_bf(&mut self.m, &rhs.m, sh) as isize;
                }
            }
        }
    }
}

impl AddAssign<f64> for BitFloat {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
        *self += BitFloat::from(rhs);
    }
}

impl AddAssign<f32> for BitFloat {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        *self += BitFloat::from(rhs);
    }
}

impl Neg for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn neg(self) -> Self::Output {
        BitFloat {
            m: self.m,
            exp: self.exp,
            sign: !self.sign,
        }
    }
}

impl Neg for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn neg(self) -> Self::Output {
        BitFloat {
            m: self.m.clone(),
            exp: self.exp,
            sign: !self.sign,
        }
    }
}

impl Sub for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            return -rhs;
        }
        if rhs.m.is_empty() {
            return self;
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                let mut m = self.m;
                let mut exp = self.exp;
                if self.sign == rhs.sign {
                    exp -= sub_bf(&mut m, &rhs.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: self.sign,
                }
            }
            Less => {
                let mut m = rhs.m;
                let mut exp = rhs.exp;
                if self.sign == rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: !rhs.sign,
                }
            }
            Equal => {
                if self.sign == rhs.sign {
                    return BitFloat {
                        sign: false,
                        exp: 0,
                        m: vec![],
                    };
                } else {
                    let mut m = self.m;
                    let mut exp = self.exp;
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                    BitFloat {
                        exp,
                        m,
                        sign: self.sign,
                    }
                }
            }
        }
    }
}

impl Sub for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            return -rhs;
        }
        if rhs.m.is_empty() {
            return self.clone();
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                let mut m = self.m.clone();
                let mut exp = self.exp;
                if self.sign == rhs.sign {
                    exp -= sub_bf(&mut m, &rhs.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: self.sign,
                }
            }
            Less => {
                let mut m = rhs.m.clone();
                let mut exp = rhs.exp;
                if self.sign == rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: !rhs.sign,
                }
            }
            Equal => {
                if self.sign == rhs.sign {
                    return BitFloat {
                        sign: false,
                        exp: 0,
                        m: vec![],
                    };
                } else {
                    let mut m = self.m.clone();
                    let mut exp = self.exp;
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                    BitFloat {
                        exp,
                        m,
                        sign: self.sign,
                    }
                }
            }
        }
    }
}

impl Sub<&BitFloat> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn sub(self, rhs: &BitFloat) -> Self::Output {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            return -rhs;
        }
        if rhs.m.is_empty() {
            return self;
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                let mut m = self.m;
                let mut exp = self.exp;
                if self.sign == rhs.sign {
                    exp -= sub_bf(&mut m, &rhs.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: self.sign,
                }
            }
            Less => {
                let mut m = rhs.m.clone();
                let mut exp = rhs.exp;
                if self.sign == rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: !rhs.sign,
                }
            }
            Equal => {
                if self.sign == rhs.sign {
                    return BitFloat {
                        sign: false,
                        exp: 0,
                        m: vec![],
                    };
                } else {
                    let mut m = self.m;
                    let mut exp = self.exp;
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                    BitFloat {
                        exp,
                        m,
                        sign: self.sign,
                    }
                }
            }
        }
    }
}

impl Sub<BitFloat> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn sub(self, rhs: BitFloat) -> Self::Output {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            return -rhs;
        }
        if rhs.m.is_empty() {
            return self.clone();
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                let mut m = self.m.clone();
                let mut exp = self.exp;
                if self.sign == rhs.sign {
                    exp -= sub_bf(&mut m, &rhs.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: self.sign,
                }
            }
            Less => {
                let mut m = rhs.m;
                let mut exp = rhs.exp;
                if self.sign == rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                BitFloat {
                    exp,
                    m,
                    sign: !rhs.sign,
                }
            }
            Equal => {
                if self.sign == rhs.sign {
                    return BitFloat {
                        sign: false,
                        exp: 0,
                        m: vec![],
                    };
                } else {
                    let mut m = self.m.clone();
                    let mut exp = self.exp;
                    exp += add_bf(&mut m, &rhs.m, sh) as isize;
                    BitFloat {
                        exp,
                        m,
                        sign: self.sign,
                    }
                }
            }
        }
    }
}

impl Sub<f64> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        self - BitFloat::from(rhs)
    }
}

impl Sub<f64> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        self - BitFloat::from(rhs)
    }
}

impl Sub<f32> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        self - BitFloat::from(rhs)
    }
}

impl Sub<f32> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        self - BitFloat::from(rhs)
    }
}

impl SubAssign for BitFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            *self = -rhs;
            return;
        }
        if rhs.m.is_empty() {
            return;
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                if self.sign == rhs.sign {
                    self.exp -= sub_bf(&mut self.m, &rhs.m, sh) as isize;
                } else {
                    self.exp += add_bf(&mut self.m, &rhs.m, sh) as isize;
                }
            }
            Less => {
                let mut m = rhs.m.clone();
                let mut exp = rhs.exp;
                if self.sign == rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                *self = BitFloat {
                    exp,
                    m,
                    sign: !rhs.sign,
                };
            }
            Equal => {
                if self.sign == rhs.sign {
                    self.sign = false;
                    self.exp = 0;
                    self.m.clear();
                } else {
                    self.exp += add_bf(&mut self.m, &rhs.m, sh) as isize;
                }
            }
        }
    }
}

impl SubAssign<&BitFloat> for BitFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: &BitFloat) {
        use Ordering::*;
        let sh = (self.exp - rhs.exp).unsigned_abs();

        if self.m.is_empty() {
            *self = -rhs;
            return;
        }
        if rhs.m.is_empty() {
            return;
        }

        match self.abs_cmp(&rhs) {
            Greater => {
                if self.sign == rhs.sign {
                    self.exp -= sub_bf(&mut self.m, &rhs.m, sh) as isize;
                } else {
                    self.exp += add_bf(&mut self.m, &rhs.m, sh) as isize;
                }
            }
            Less => {
                let mut m = rhs.m.clone();
                let mut exp = rhs.exp;
                if self.sign == rhs.sign {
                    exp -= sub_bf(&mut m, &self.m, sh) as isize;
                } else {
                    exp += add_bf(&mut m, &self.m, sh) as isize;
                }
                *self = BitFloat {
                    exp,
                    m,
                    sign: !rhs.sign,
                };
            }
            Equal => {
                if self.sign == rhs.sign {
                    self.sign = false;
                    self.exp = 0;
                    self.m.clear();
                } else {
                    self.exp += add_bf(&mut self.m, &rhs.m, sh) as isize;
                }
            }
        }
    }
}

impl SubAssign<f64> for BitFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        *self -= BitFloat::from(rhs);
    }
}

impl SubAssign<f32> for BitFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        *self -= BitFloat::from(rhs);
    }
}

impl Mul for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let (m, carry) = mul_bf(&self.m, &rhs.m, 0);
        let exp = self.exp + rhs.exp + carry as isize;

        BitFloat {
            exp,
            m,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let (m, carry) = mul_bf(&self.m, &rhs.m, 0);
        let exp = self.exp + rhs.exp + carry as isize;

        BitFloat {
            exp,
            m,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<BitFloat> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: BitFloat) -> Self::Output {
        let (m, carry) = mul_bf(&self.m, &rhs.m, 0);
        let exp = self.exp + rhs.exp + carry as isize;

        BitFloat {
            exp,
            m,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<&BitFloat> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: &BitFloat) -> Self::Output {
        let (m, carry) = mul_bf(&self.m, &rhs.m, 0);
        let exp = self.exp + rhs.exp + carry as isize;

        BitFloat {
            exp,
            m,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<f64> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self * BitFloat::from(rhs)
    }
}

impl Mul<f64> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self * BitFloat::from(rhs)
    }
}

impl Mul<BitFloat> for f64 {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: BitFloat) -> Self::Output {
        rhs * BitFloat::from(self)
    }
}

impl Mul<&BitFloat> for f64 {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: &BitFloat) -> Self::Output {
        rhs * BitFloat::from(self)
    }
}

impl Mul<f32> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        self * BitFloat::from(rhs)
    }
}

impl Mul<f32> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        self * BitFloat::from(rhs)
    }
}

impl Mul<BitFloat> for f32 {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: BitFloat) -> Self::Output {
        rhs * BitFloat::from(self)
    }
}

impl Mul<&BitFloat> for f32 {
    type Output = BitFloat;

    #[inline]
    fn mul(self, rhs: &BitFloat) -> Self::Output {
        rhs * BitFloat::from(self)
    }
}

impl MulAssign for BitFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        let carry: bool;
        (self.m, carry) = mul_bf(&self.m, &rhs.m, 0);
        self.exp += rhs.exp + carry as isize;
        self.sign ^= rhs.sign;
    }
}

impl MulAssign<&BitFloat> for BitFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: &BitFloat) {
        let carry: bool;
        (self.m, carry) = mul_bf(&self.m, &rhs.m, 0);
        self.exp += rhs.exp + carry as isize;
        self.sign ^= rhs.sign;
    }
}

impl MulAssign<f64> for BitFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        *self *= BitFloat::from(rhs);
    }
}

impl MulAssign<f32> for BitFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self *= BitFloat::from(rhs);
    }
}

pub trait StblMul<RHS = Self> {
    type Output;
    fn stbl_mul(self, rhs: RHS, acc: usize) -> Self::Output;
}

impl StblMul for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn stbl_mul(self, rhs: Self, acc: usize) -> Self::Output {
        let prod_len = (self.m.len() + rhs.m.len()).saturating_sub(1);
        let (m, carry) = mul_bf(&self.m, &rhs.m, prod_len.saturating_sub(acc));
        let exp = self.exp + rhs.exp + carry as isize;

        BitFloat {
            exp,
            m,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl StblMul for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn stbl_mul(self, rhs: Self, acc: usize) -> Self::Output {
        let prod_len = (self.m.len() + rhs.m.len()).saturating_sub(1);
        let (m, carry) = mul_bf(&self.m, &rhs.m, prod_len.saturating_sub(acc));
        let exp = self.exp + rhs.exp + carry as isize;

        BitFloat {
            exp,
            m,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl StblMul<&BitFloat> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn stbl_mul(self, rhs: &BitFloat, acc: usize) -> Self::Output {
        let prod_len = (self.m.len() + rhs.m.len()).saturating_sub(1);
        let (m, carry) = mul_bf(&self.m, &rhs.m, prod_len.saturating_sub(acc));
        let exp = self.exp + rhs.exp + carry as isize;

        BitFloat {
            exp,
            m,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl StblMul<BitFloat> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn stbl_mul(self, rhs: BitFloat, acc: usize) -> Self::Output {
        let prod_len = (self.m.len() + rhs.m.len()).saturating_sub(1);
        let (m, carry) = mul_bf(&self.m, &rhs.m, prod_len.saturating_sub(acc));
        let exp = self.exp + rhs.exp + carry as isize;

        BitFloat {
            exp,
            m,
            sign: self.sign ^ rhs.sign,
        }
    }
}

pub trait StblMulAssign<RHS = Self> {
    fn stbl_mul_assign(&mut self, rhs: RHS, acc: usize);
}

impl StblMulAssign for BitFloat {
    #[inline]
    fn stbl_mul_assign(&mut self, rhs: Self, acc: usize) {
        let prod_len = (self.m.len() + rhs.m.len()).saturating_sub(1);
        let carry: bool;
        (self.m, carry) = mul_bf(&self.m, &rhs.m, prod_len.saturating_sub(acc));
        self.exp += rhs.exp + carry as isize;
        self.sign ^= rhs.sign;
    }
}

impl StblMulAssign<&BitFloat> for BitFloat {
    #[inline]
    fn stbl_mul_assign(&mut self, rhs: &BitFloat, acc: usize) {
        let prod_len = (self.m.len() + rhs.m.len()).saturating_sub(1);
        let carry: bool;
        (self.m, carry) = mul_bf(&self.m, &rhs.m, prod_len.saturating_sub(acc));
        self.exp += rhs.exp + carry as isize;
        self.sign ^= rhs.sign;
    }
}

impl Div for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let mut n = self;
        let mut d = rhs;
        div_bf_gs(&mut n, &mut d);
        n
    }
}

impl Div for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let mut n = self.clone();
        let mut d = rhs.clone();
        div_bf_gs(&mut n, &mut d);
        n
    }
}

impl Div<&BitFloat> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: &BitFloat) -> Self::Output {
        let mut n = self;
        let mut d = rhs.clone();
        div_bf_gs(&mut n, &mut d);
        n
    }
}

impl Div<BitFloat> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: BitFloat) -> Self::Output {
        let mut n = self.clone();
        let mut d = rhs;
        div_bf_gs(&mut n, &mut d);
        n
    }
}

impl Div<f64> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        self / BitFloat::from(rhs)
    }
}

impl Div<f64> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        self / BitFloat::from(rhs)
    }
}

impl Div<BitFloat> for f64 {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: BitFloat) -> Self::Output {
        BitFloat::from(self) / rhs
    }
}

impl Div<&BitFloat> for f64 {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: &BitFloat) -> Self::Output {
        BitFloat::from(self) / rhs
    }
}

impl Div<f32> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        self / BitFloat::from(rhs)
    }
}

impl Div<f32> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        self / BitFloat::from(rhs)
    }
}

impl Div<BitFloat> for f32 {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: BitFloat) -> Self::Output {
        BitFloat::from(self) / rhs
    }
}

impl Div<&BitFloat> for f32 {
    type Output = BitFloat;

    #[inline]
    fn div(self, rhs: &BitFloat) -> Self::Output {
        BitFloat::from(self) / rhs
    }
}

impl DivAssign for BitFloat {
    fn div_assign(&mut self, rhs: Self) {
        let mut d = rhs;
        div_bf_gs(self, &mut d);
    }
}

impl DivAssign<&BitFloat> for BitFloat {
    fn div_assign(&mut self, rhs: &BitFloat) {
        let mut d = rhs.clone();
        div_bf_gs(self, &mut d);
    }
}

impl DivAssign<f64> for BitFloat {
    fn div_assign(&mut self, rhs: f64) {
        *self /= BitFloat::from(rhs);
    }
}

impl DivAssign<f32> for BitFloat {
    fn div_assign(&mut self, rhs: f32) {
        *self /= BitFloat::from(rhs);
    }
}

pub trait PowI<RHS = Self> {
    type Output;
    fn powi(self, n: RHS) -> Self::Output;
}

impl PowI<BitInt> for BitFloat {
    type Output = BitFloat;
    fn powi(self, n: BitInt) -> Self::Output {
        if n.get_sign() {
            &*ONE / powi_ubi(self, n.unsighned_abs())
        } else {
            powi_ubi(self, n.unsighned_abs())
        }
    }
}

impl PowI<BitInt> for &BitFloat {
    type Output = BitFloat;
    fn powi(self, n: BitInt) -> Self::Output {
        if n.get_sign() {
            &*ONE / powi_ubi(self.clone(), n.unsighned_abs())
        } else {
            powi_ubi(self.clone(), n.unsighned_abs())
        }
    }
}

impl PowI<&BitInt> for BitFloat {
    type Output = BitFloat;
    fn powi(self, n: &BitInt) -> Self::Output {
        if n.get_sign() {
            &*ONE / powi_ubi(self, n.clone().unsighned_abs())
        } else {
            powi_ubi(self, n.clone().unsighned_abs())
        }
    }
}

impl PowI<&BitInt> for &BitFloat {
    type Output = BitFloat;
    fn powi(self, n: &BitInt) -> Self::Output {
        if n.get_sign() {
            &*ONE / powi_ubi(self.clone(), n.clone().unsighned_abs())
        } else {
            powi_ubi(self.clone(), n.clone().unsighned_abs())
        }
    }
}

impl PowI<i128> for BitFloat {
    type Output = BitFloat;

    fn powi(self, n: i128) -> Self::Output {
        if n < 0 {
            &*ONE / powi_prim(self, n.unsigned_abs())
        } else {
            powi_prim(self, n.unsigned_abs())
        }
    }
}

impl PowI<i128> for &BitFloat {
    type Output = BitFloat;

    fn powi(self, n: i128) -> Self::Output {
        if n < 0 {
            &*ONE / powi_prim(self.clone(), n.unsigned_abs())
        } else {
            powi_prim(self.clone(), n.unsigned_abs())
        }
    }
}

macro_rules! impl_powi_bf_prim {
    ($($t:ty),*) => {
        $(
            impl PowI<$t> for BitFloat {
                type Output = BitFloat;

                fn powi(self, n: $t) -> Self::Output {
                    if n < 0 {
                        &*ONE / powi_prim(self, n.unsigned_abs() as u128)
                    } else {
                        powi_prim(self, n.unsigned_abs() as u128)
                    }
                }
            }

            impl PowI<$t> for &BitFloat {
                type Output = BitFloat;

                fn powi(self, n: $t) -> Self::Output {
                    if n < 0 {
                        &*ONE / powi_prim(self.clone(), n.unsigned_abs() as u128)
                    } else {
                        powi_prim(self.clone(), n.unsigned_abs() as u128)
                    }
                }
            }
        )*
    };
}

impl_powi_bf_prim!(i64, isize, i32, i16, i8);

pub trait Exp {
    type Output;
    fn exp(self) -> Self::Output;
}

impl Exp for BitFloat {
    type Output = BitFloat;

    fn exp(self) -> Self::Output {
        exp_bf(&self)
    }
}

impl Exp for &BitFloat {
    type Output = BitFloat;

    fn exp(self) -> Self::Output {
        exp_bf(self)
    }
}

pub trait Ln {
    type Output;
    fn ln(self) -> Self::Output;
}

impl Ln for BitFloat {
    type Output = BitFloat;

    fn ln(self) -> Self::Output {
        ln_bf(&self)
    }
}

impl Ln for &BitFloat {
    type Output = BitFloat;

    fn ln(self) -> Self::Output {
        ln_bf(self)
    }
}

pub trait PowF<RHS = Self> {
    type Output;
    fn powf(self, rhs: RHS) -> Self::Output;
}

impl PowF for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn powf(self, rhs: Self) -> Self::Output {
        (rhs * self.ln()).exp()
    }
}

impl PowF for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn powf(self, rhs: Self) -> Self::Output {
        (rhs * self.ln()).exp()
    }
}

impl PowF<BitFloat> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn powf(self, rhs: BitFloat) -> Self::Output {
        (rhs * self.ln()).exp()
    }
}

impl PowF<&BitFloat> for BitFloat {
    type Output = BitFloat;

    #[inline]
    fn powf(self, rhs: &BitFloat) -> Self::Output {
        (rhs * self.ln()).exp()
    }
}

impl PowF<f64> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn powf(self, rhs: f64) -> Self::Output {
        (rhs * self.ln()).exp()
    }
}

impl PowF<f32> for &BitFloat {
    type Output = BitFloat;

    #[inline]
    fn powf(self, rhs: f32) -> Self::Output {
        (rhs * self.ln()).exp()
    }
}
