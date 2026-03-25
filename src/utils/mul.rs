#![allow(dead_code)]
use crate::utils::utils::*;
use rustfft::num_traits::{One, Zero};
use rustfft::{num_complex::Complex, Fft, FftDirection, FftPlanner};
use std::arch::asm;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_prim_asm_x86(val: u64, prim: u64, carry: u64) -> (u64, u64) {
    let lo: u64;
    let hi: u64;
    asm!(
        "mul {prim}",
        "add rax, {carry}",
        "adc rdx, 0",
        prim = in(reg) prim,
        carry = in(reg) carry,
        inout("rax") val => lo,
        out("rdx") hi,
        options(nostack, nomem),
    );
    (lo, hi)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_prim_asm_aarch(val: u64, prim: u64, carry: u64) -> (u64, u64) {
    let lo: u64;
    let hi: u64;
    asm!(
        "mul {lo}, {a_val}, {prim}",
        "umulh {hi}, {a_val}, {prim}",
        "adds {lo}, {lo}, {carry}",
        "adc {hi}, {hi}, xzr",
        a_val = in(reg) val,
        prim = in(reg) prim,
        carry = in(reg) carry,
        lo = out(reg) lo,
        hi = out(reg) hi,
        options(nostack, nomem),
    );
    (lo, hi)
}

#[inline(always)]
unsafe fn mul_prim_asm(val: u64, prim: u64, carry: u64) -> (u64, u64) {
    #[cfg(target_arch = "aarch64")]
    {
        mul_prim_asm_aarch(val, prim, carry)
    }
    #[cfg(target_arch = "x86_64")]
    {
        mul_prim_asm_x86(val, prim, carry)
    }
}

pub fn mul_prim(buf: &mut [u64], prim: u64) -> u64 {
    let mut carry: u64 = 0;
    for e in buf {
        let (lo, hi) = unsafe { mul_prim_asm(*e, prim, carry) };
        *e = lo;
        carry = hi;
    }
    carry
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_asm_x86(a_val: u64, b_val: u64, acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) {
    asm!(
        "mul {b_val}",
        "add {acc0}, rax",
        "adc {acc1}, rdx",
        "adc {acc2}, 0",
        b_val = in(reg) b_val,
        inout("rax") a_val => _,
        out("rdx") _,
        acc0 = inout(reg) *acc0,
        acc1 = inout(reg) *acc1,
        acc2 = inout(reg) *acc2,
        options(nostack, nomem),
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_asm_aarch(a_val: u64, b_val: u64, acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) {
    asm!(
        "mul {lo}, {a_val}, {b_val}",
        "umulh {hi}, {a_val}, {b_val}",
        "adds {acc0}, {acc0}, {lo}",
        "adcs {acc1}, {acc1}, {hi}",
        "adc {acc2}, {acc2}, xzr",
        a_val = in(reg) a_val,
        b_val = in(reg) b_val,
        lo = out(reg) _,
        hi = out(reg) _,
        acc0 = inout(reg) *acc0,
        acc1 = inout(reg) *acc1,
        acc2 = inout(reg) *acc2,
        options(nostack, nomem),
    );
}

#[inline(always)]
unsafe fn mul_asm(a_val: u64, b_val: u64, acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) {
    #[cfg(target_arch = "aarch64")]
    mul_asm_aarch(a_val, b_val, acc0, acc1, acc2);

    #[cfg(target_arch = "x86_64")]
    mul_asm_x86(a_val, b_val, acc0, acc1, acc2);
}

pub fn mul_prim2(buf: &mut [u64], prim: u128) -> u128 {
    if buf.is_empty() {
        return 0;
    }
    let p0 = prim as u64;
    let p1 = (prim >> 64) as u64;
    let len = buf.len();

    let mut acc0: u64 = 0;
    let mut acc1: u64 = 0;
    let mut acc2: u64 = 0;

    let mut prev = buf[0];
    unsafe {
        mul_asm(prev, p0, &mut acc0, &mut acc1, &mut acc2);
    }

    for i in 1..len {
        let cur = unsafe { *buf.get_unchecked(i) };
        unsafe {
            *buf.get_unchecked_mut(i - 1) = acc0;
        }
        acc0 = acc1;
        acc1 = acc2;
        acc2 = 0;
        unsafe {
            mul_asm(prev, p1, &mut acc0, &mut acc1, &mut acc2);
            mul_asm(cur, p0, &mut acc0, &mut acc1, &mut acc2);
        }
        prev = cur;
    }

    buf[len - 1] = acc0;
    acc0 = acc1;
    acc1 = acc2;
    acc2 = 0;

    unsafe {
        mul_asm(prev, p1, &mut acc0, &mut acc1, &mut acc2);
    }

    (acc1 as u128) << 64 | acc0 as u128
}

pub fn mul_buf(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let a_len = a.len() - 1;
    let b_len = b.len() - 1;

    let mut acc0: u64 = 0;
    let mut acc1: u64 = 0;
    let mut acc2: u64 = 0;
    for n in 0..out.len() {
        let mi = n.saturating_sub(b_len);
        let mf = n.min(a_len);
        unsafe {
            for m in mi..=mf {
                let a_val = *a.get_unchecked(m);
                let b_val = *b.get_unchecked(n - m);
                mul_asm(a_val, b_val, &mut acc0, &mut acc1, &mut acc2);
            }
            *out.get_unchecked_mut(n) = acc0;
        }
        acc0 = acc1;
        acc1 = acc2;
        acc2 = 0;
    }

    return acc0;
}

fn chunking_karatsuba(long: &[u64], short: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    out.fill(0);
    let s = short.len();
    let mut overflow = 0;

    let n_chunks = long.len() / s;
    let len = if long.len() % s == 0 {
        n_chunks - 1
    } else {
        n_chunks
    };

    for i in 0..len {
        let (val, rest) = scratch.split_at_mut(2 * s);
        let of = i * s;
        karatsuba_alg(&long[of..of + s], short, val, rest);
        add_buf(&mut out[of..], val);
    }

    let of = n_chunks * s;
    let chunk = &long[of..];
    let (val, rest) = scratch.split_at_mut(chunk.len() + s - 1);
    overflow += karatsuba_alg(short, chunk, val, rest);
    if add_buf(&mut out[of..], val) {
        overflow += 1;
    }

    return overflow;
}

enum KDispatch {
    Prim,
    Prim2,
    Base,
    Chunking,
    Recurse,
}

pub(crate) const KARATSUBA_CUTOFF: usize = 27;

fn karatsuba_dispatch(long: usize, short: usize) -> KDispatch {
    return if short == 1 {
        KDispatch::Prim
    } else if short == 2 {
        KDispatch::Prim2
    } else if long <= KARATSUBA_CUTOFF {
        KDispatch::Base
    } else if short < (long + 1) / 2 {
        if short <= KARATSUBA_CUTOFF {
            KDispatch::Base
        } else {
            KDispatch::Chunking
        }
    } else {
        KDispatch::Recurse
    };
}

fn karatsuba_core(
    long: &[u64],
    short: &[u64],
    half_len: usize,   // (long + 1) / 2
    out: &mut [u64],   // >= long + short - 1
    cross: &mut [u64], // >= 2*half + 1
    scratch: &mut [u64],
) -> u64 {
    let (l0, l1) = long.split_at(half_len);
    let (s0, s1) = short.split_at(half_len);

    {
        let (l_sum, s_sum) = out[..2 * half_len + 2].split_at_mut(half_len + 1);
        let mut l_len = half_len;
        let mut s_len = half_len;

        l_sum[..l_len].copy_from_slice(l0);
        if add_buf(&mut l_sum[..l_len], l1) {
            l_sum[l_len] = 1;
            l_len += 1;
        }

        s_sum[..s_len].copy_from_slice(s0);
        if add_buf(&mut s_sum[..s_len], s1) {
            s_sum[s_len] = 1;
            s_len += 1;
        }

        karatsuba_alg(&l_sum[..l_len], &s_sum[..s_len], cross, scratch);
    }

    let (z0, z2) = out.split_at_mut(2 * half_len);
    karatsuba_alg(l0, s0, z0, scratch);
    let mut c = karatsuba_alg(l1, s1, z2, scratch);

    sub_buf(cross, z0);
    sub_buf(cross, z2);
    if c > 0 {
        sub_buf(&mut cross[z2.len()..], &[c]);
    }
    if add_buf(&mut out[half_len..], &cross) {
        c += 1;
    }

    return c;
}

pub(super) fn karatsuba_alg(a: &[u64], b: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    match karatsuba_dispatch(long.len(), short.len()) {
        KDispatch::Prim => {
            out[..long.len()].copy_from_slice(long);
            return mul_prim(out, short[0]);
        }
        KDispatch::Prim2 => {
            out[..long.len()].copy_from_slice(long);
            out[out.len() - 1] = 0;
            return mul_prim2(out, short[0] as u128 | (short[0] as u128) << 64) as u64;
        }
        KDispatch::Base => {
            return mul_buf(long, short, out);
        }
        KDispatch::Chunking => {
            return chunking_karatsuba(long, short, out, scratch);
        }
        KDispatch::Recurse => {}
    }

    let half_len = (long.len() + 1) / 2;
    let (cross, rest) = scratch.split_at_mut(2 * half_len + 1);
    cross.fill(0);

    karatsuba_core(long, short, half_len, out, cross, rest)
}

pub(super) fn find_karatsuba_scratch_size(l: usize, s: usize) -> usize {
    if l <= KARATSUBA_CUTOFF || s <= 2 {
        return 0;
    }

    let init_half = (l + 1) / 2;

    let (mut n, mut total) = if s < init_half {
        if s <= KARATSUBA_CUTOFF {
            return 0;
        }
        (s, 2 * s)
    } else {
        (init_half + 1, 2 * init_half + 1)
    };

    while n > KARATSUBA_CUTOFF {
        let half = (n + 1) / 2;
        total += 2 * half + 1;
        n = half + 1;
    }

    total
}

struct TwidleTower {
    table: Vec<Complex<f64>>,
    veiw: Vec<Complex<f64>>,
    res: usize,
    gen: u8,
    base: usize,
}

impl TwidleTower {
    const MAX_GEN: u8 = 3;

    fn build(n: usize) -> Self {
        let len = n / 4 + 1;
        let th = 2.0 * PI / (n as f64);
        let mut table = Vec::<Complex<f64>>::with_capacity(len);
        table.push(Complex::new(1.0, 0.0));
        for k in 1..len {
            table.push(Complex::cis(th * k as f64));
        }

        TwidleTower {
            table,
            veiw: Vec::new(),
            res: n,
            gen: 0,
            base: 1,
        }
    }

    fn refine(&mut self, new_res: usize) {
        debug_assert!(new_res > self.res);
        debug_assert!(new_res % self.res == 0);

        let new_len = new_res / 4 + 1;
        let mut new_table = Vec::<Complex<f64>>::with_capacity(new_len);

        let growth = new_res / self.res;
        let mut steps = Vec::<Complex<f64>>::with_capacity(growth - 1);
        let th = 2.0 * PI / (new_res as f64);
        for i in 1..growth {
            steps.push(Complex::cis(th * i as f64));
        }

        for &v in &self.table[..self.table.len() - 1] {
            new_table.push(v);
            for &s in steps.iter() {
                new_table.push(v * s);
            }
        }
        new_table.push(self.table[self.table.len() - 1]);

        self.table = new_table;
        self.res = new_res;
        self.gen += 1;
        self.base *= growth;
    }

    fn rebuild(&mut self, new_res: usize) {
        debug_assert!(new_res > self.res);
        debug_assert!(new_res % self.res == 0);

        let new_len = new_res / 4 + 1;
        let growth = new_res / self.res;
        let new_base = self.base * growth;
        let mut new_table = Vec::<Complex<f64>>::with_capacity(new_len);
        new_table.push(Complex::one());
        let th = 2.0 * PI / (new_res as f64);
        for i in 1..new_len {
            if i % new_base != 0 {
                new_table.push(Complex::cis(th * i as f64));
            } else {
                new_table.push(self.table[i / growth]);
            }
        }

        self.table = new_table;
        self.res = new_res;
        self.gen = 0;
        self.base = 1;
    }

    fn get(&mut self, res: usize) -> &[Complex<f64>] {
        return match res.cmp(&self.res) {
            std::cmp::Ordering::Equal => &self.table,
            std::cmp::Ordering::Greater => {
                debug_assert!(res % self.res == 0);
                if self.gen >= Self::MAX_GEN {
                    self.rebuild(res);
                } else {
                    self.refine(res);
                }
                &self.table
            }
            std::cmp::Ordering::Less => {
                debug_assert!(self.res % res == 0);
                let len = res / 4 + 1;
                let stride = self.res / res;
                self.veiw.clear();
                for i in 0..len {
                    self.veiw.push(self.table[i * stride]);
                }
                &self.veiw
            }
        };
    }
}

struct FFTCache {
    planner: FftPlanner<f64>,
    twidles: HashMap<usize, TwidleTower>,
    scratch: Vec<Complex<f64>>,
}

thread_local! {
    static FFT_CACHE: RefCell<FFTCache> = RefCell::new(FFTCache { planner: FftPlanner::new(), twidles: HashMap::new(), scratch: Vec::new() });
}

const LENGTH_PRIMES: [usize; 3] = [3, 5, 7];

const fn log_prim_search(init: usize, target: usize, mut idx: usize) -> usize {
    if init >= target {
        return init;
    }

    if idx == 0 {
        let sh = target.ilog2().saturating_sub(init.ilog2());
        let mut result = init << sh;
        if result < target {
            result <<= 1;
        }
        return result;
    }

    idx -= 1;
    let p = LENGTH_PRIMES[idx];
    let mut val = init;
    let mut best = log_prim_search(val, target, idx);
    if best == target {
        return best;
    }
    val *= p;
    while val < best {
        let cur = log_prim_search(val, target, idx);
        if cur < best {
            best = cur;
            if best == target {
                return best;
            }
        }
        val *= p;
    }
    return best;
}

const fn find_fft_size(n: usize) -> usize {
    const INIT: usize = 4; // guarantee that size is a multiple of 4
    return log_prim_search(INIT, n, LENGTH_PRIMES.len());
}

fn u16_buf_len(buf: &[u64]) -> usize {
    let buf_len = buf_len(buf);
    let mut buf_u16_len = 4 * buf_len;
    let last = buf[buf_len];
    let mask = 0xFFFF;
    if (last >> 48) & mask == 0 {
        buf_u16_len -= 1;
        if (last >> 32) & mask == 0 {
            buf_u16_len -= 1;
            if (last >> 16) & mask == 0 {
                buf_u16_len -= 1;
            }
        }
    }
    return buf_u16_len;
}

#[inline(always)]
fn separate_mul(x: Complex<f64>, y: Complex<f64>) -> Complex<f64> {
    let ar = x.re + y.re;
    let ai = x.im - y.im;
    let br = x.im + y.im;
    let bi = y.re - x.re;
    Complex::new((ar * br - ai * bi) * 0.25, (ar * bi + ai * br) * 0.25)
}

fn recombine(x: &mut [Complex<f64>], m: usize, n: usize, tw: &[Complex<f64>]) {
    let pk = x[0].re * x[0].im;
    let pj = x[m].re * x[m].im;
    x[0] = Complex::new(pk + pj, pk - pj);

    for k in 1..m / 2 {
        let pk = separate_mul(x[k], x[n - k]);
        let pj = separate_mul(x[m + k], x[m - k]);
        let s = pk + pj;
        let d = pk - pj;
        let dw = d * tw[k];

        x[k] = s + Complex::<f64>::i() * dw;
        x[m - k] = s.conj() + Complex::<f64>::i() * dw.conj();
    }

    let k = m / 2;
    let wk = Complex::cis(PI * k as f64 / n as f64);
    let pk = separate_mul(x[k], x[n - k]);
    let s_re = 2.0 * pk.re;
    let d_im = 2.0 * pk.im;
    x[k] = Complex::new(s_re - d_im * wk.re, -d_im * wk.im);
}

fn accumulate(x: &[Complex<f64>], out: &mut [u64]) -> u64 {
    debug_assert!(2 * x.len() > out.len());
    let mut carry = 0;
    for i in 0..out.len() {
        let idx = i * 2;
        let a = x[idx].re.round_ties_even() as u128;
        let b = x[idx].im.round_ties_even() as u128;
        let c = x[idx + 1].re.round_ties_even() as u128;
        let d = x[idx + 1].im.round_ties_even() as u128;

        let val = carry + a + (b << 16) + (c << 32) + (d << 48);
        out[i] = val as u64;
        carry = val >> 64;
    }
    return carry as u64;
}

fn fft_mul_vec(a: &[u64], b: &[u64]) -> (Vec<u64>, u64) {
    let a_u16_len = u16_buf_len(a);
    let b_u16_len = u16_buf_len(b);
    let unpadded_sz = a_u16_len + b_u16_len - 1;

    let n = find_fft_size(unpadded_sz);
    let m = n / 2;

    let mut out = vec![0_u64; a.len() + b.len() - 1];
    let carry = FFT_CACHE.with(|cell| {
        let cache = &mut *cell.borrow_mut();

        let FFTCache {
            planner,
            twidles,
            scratch,
        } = cache;

        let fwd = planner.plan_fft_forward(n);
        let bwd = planner.plan_fft_inverse(m);
        let scratch_len = fwd
            .get_inplace_scratch_len()
            .max(bwd.get_inplace_scratch_len())
            + n;

        if scratch_len > scratch.len() {
            scratch.resize(scratch_len, Complex::zero());
        }

        let (x, fft_scratch) = scratch.split_at_mut(n);

        let mask = 0xFFFF;
        for i in 0..a_u16_len {
            x[i].re = ((a[i / 4] >> ((i % 4) * 16)) & mask) as f64;
        }
        for i in 0..b_u16_len {
            x[i].im = ((b[i / 4] >> ((i % 4) * 16)) & mask) as f64;
        }

        let fwd = planner.plan_fft_forward(n);
        let bwd = planner.plan_fft_inverse(m);

        fwd.process_with_scratch(x, fft_scratch);

        let tw = twidles
            .entry(n >> n.trailing_zeros())
            .or_insert_with(|| TwidleTower::build(n))
            .get(n);

        recombine(x, m, n, tw);

        bwd.process_with_scratch(&mut x[..m], fft_scratch);

        let scale = 1.0 / n as f64;
        for v in &mut x[..m] {
            *v = v.scale(scale)
        }
        accumulate(&x[..m], &mut out)
    });

    return (out, carry);
}

pub fn mul_vec(a: &[u64], b: &[u64]) -> (Vec<u64>, u64) {
    let mut out = vec![0_u64; a.len() + b.len() - 1];
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let c = match karatsuba_dispatch(long.len(), short.len()) {
        KDispatch::Prim => {
            out[..long.len()].copy_from_slice(long);
            mul_prim(&mut out, short[0])
        }
        KDispatch::Prim2 => {
            out[..long.len()].copy_from_slice(long);
            mul_prim2(&mut out, combine_u64(short[0], short[1])) as u64
        }
        KDispatch::Base => mul_buf(long, short, &mut out),
        KDispatch::Chunking => {
            let mut scratch = vec![0_u64; find_karatsuba_scratch_size(long.len(), short.len())];
            chunking_karatsuba(long, short, &mut out, &mut scratch)
        }
        KDispatch::Recurse => {
            let mut cross_scratch =
                vec![0_u64; find_karatsuba_scratch_size(long.len(), short.len())];
            let half = (long.len() + 1) / 2;
            let (cross, scratch) = cross_scratch.split_at_mut(2 * half + 1);
            karatsuba_core(long, short, half, &mut out, cross, scratch)
        }
    };

    return (out, c);
}

pub fn mul_arr<const N: usize>(a: &[u64], b: &[u64]) -> Result<([u64; N], u64), ()> {
    let a_len = buf_len(a);
    let b_len = buf_len(b);
    if a_len + b_len - 1 > N {
        return Err(());
    }

    let mut out = [0_u64; N];
    let (long, short) = if a_len > b_len { (a, b) } else { (b, a) };
    let c = match karatsuba_dispatch(long.len(), short.len()) {
        KDispatch::Prim => {
            out[..long.len()].copy_from_slice(long);
            mul_prim(&mut out, short[0])
        }
        KDispatch::Prim2 => {
            out[..long.len()].copy_from_slice(long);
            mul_prim2(&mut out, combine_u64(short[0], short[1])) as u64
        }
        KDispatch::Base => mul_buf(long, short, &mut out),
        KDispatch::Chunking => {
            let mut scratch = [0; N];
            chunking_karatsuba(long, short, &mut out, &mut scratch)
        }
        KDispatch::Recurse => {
            let mut scratch = [0; N];
            let scratch_sz = find_karatsuba_scratch_size(long.len(), short.len());
            let half_len = (long.len() + 1) / 2;
            if scratch_sz > N {
                let mut cross = [0; N];
                karatsuba_core(long, short, half_len, &mut out, &mut cross, &mut scratch)
            } else {
                let (cross, rest) = scratch.split_at_mut(2 * half_len + 1);
                karatsuba_core(long, short, half_len, &mut out, cross, rest)
            }
        }
    };
    return Ok((out, c));
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_double_asm_x86(
    a_val: u64,
    b_val: u64,
    acc0: &mut u64,
    acc1: &mut u64,
    acc2: &mut u64,
) {
    asm!(
        "mul {b_val}",
        "add rax, rax",
        "adc rdx, rdx",
        "adc {acc2}, 0",
        "add {acc0}, rax",
        "adc {acc1}, rdx",
        "adc {acc2}, 0",
        b_val = in(reg) b_val,
        inout("rax") a_val => _,
        out("rdx") _,
        acc0 = inout(reg) *acc0,
        acc1 = inout(reg) *acc1,
        acc2 = inout(reg) *acc2,
        options(nostack, nomem),
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_double_asm_aarch(
    a_val: u64,
    b_val: u64,
    acc0: &mut u64,
    acc1: &mut u64,
    acc2: &mut u64,
) {
    asm!(
        "mul {lo}, {a_val}, {b_val}",
        "umulh {hi}, {a_val}, {b_val}",
        "adds {lo}, {lo}, {lo}",
        "adcs {hi}, {hi}, {hi}",
        "adc {acc2}, {acc2}, xzr",
        "adds {acc0}, {acc0}, {lo}",
        "adcs {acc1}, {acc1}, {hi}",
        "adc {acc2}, {acc2}, xzr",
        a_val = in(reg) a_val,
        b_val = in(reg) b_val,
        lo = out(reg) _,
        hi = out(reg) _,
        acc0 = inout(reg) *acc0,
        acc1 = inout(reg) *acc1,
        acc2 = inout(reg) *acc2,
        options(nostack, nomem),
    );
}

#[inline(always)]
unsafe fn mul_double_asm(a_val: u64, b_val: u64, acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) {
    #[cfg(target_arch = "aarch64")]
    mul_double_asm_aarch(a_val, b_val, acc0, acc1, acc2);

    #[cfg(target_arch = "x86_64")]
    mul_double_asm_x86(a_val, b_val, acc0, acc1, acc2);
}

pub fn sqr_buf(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let len = buf.len() - 1;

    let mut acc0: u64 = 0;
    let mut acc1: u64 = 0;
    let mut acc2: u64 = 0;

    unsafe {
        mul_asm(buf[0], buf[0], &mut acc0, &mut acc1, &mut acc2);
        *out.get_unchecked_mut(0) = acc0;
    }
    acc0 = acc1;
    acc1 = acc2;
    acc2 = 0;

    for n in 1..out.len() {
        let mi = n.saturating_sub(len);
        let mf = ((n - 1) / 2).min(len);

        unsafe {
            for m in mi..=mf {
                let a_val = *buf.get_unchecked(m);
                let b_val = *buf.get_unchecked(n - m);
                mul_double_asm(a_val, b_val, &mut acc0, &mut acc1, &mut acc2);
            }
        }

        if n % 2 == 0 && n / 2 <= len {
            unsafe {
                let d = *buf.get_unchecked(n / 2);
                mul_asm(d, d, &mut acc0, &mut acc1, &mut acc2);
            }
        }

        out[n] = acc0;
        acc0 = acc1;
        acc1 = acc2;
        acc2 = 0;
    }

    acc0
}

pub(crate) const KARATSUBA_SQR_CUTOFF: usize = 26;

fn karatsuba_sqr_core(
    buf: &[u64],
    half_len: usize,
    out: &mut [u64],
    cross: &mut [u64],
    scratch: &mut [u64],
) -> u64 {
    let (x0, x1) = buf.split_at(half_len);

    {
        let sum = &mut out[..=half_len];
        sum[..half_len].copy_from_slice(x0);
        let mut sum_len = half_len;
        if add_buf(&mut sum[..half_len], x1) {
            sum[half_len] = 1;
            sum_len += 1;
        }

        karatsuba_sqr_alg(&sum[..sum_len], cross, scratch);
    }

    out.fill(0);
    let (z0, z2) = out.split_at_mut(2 * half_len);

    karatsuba_sqr_alg(x0, z0, scratch);
    let mut c = karatsuba_sqr_alg(x1, z2, scratch);

    sub_buf(cross, z0);
    sub_buf(cross, z2);
    if c > 0 {
        sub_buf(&mut cross[z2.len()..], &[c]);
    }
    if add_buf(&mut out[half_len..], &cross) {
        c += 1;
    }

    return c;
}

fn karatsuba_sqr_alg(buf: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    if buf.len() <= KARATSUBA_SQR_CUTOFF {
        return sqr_buf(buf, out);
    }
    let half = (buf.len() + 1) / 2;
    let (cross, rest) = scratch.split_at_mut(2 * half + 1);
    karatsuba_sqr_core(buf, half, out, cross, rest)
}

fn find_karatsuba_sqr_scratch_size(mut n: usize) -> usize {
    let mut total = 0;
    while n > KARATSUBA_SQR_CUTOFF {
        let half = (n + 1) / 2;
        total += 2 * half + 1;
        n = half + 1;
    }
    total
}

pub fn sqr_vec(buf: &[u64]) -> (Vec<u64>, u64) {
    let mut out = vec![0_u64; 2 * buf.len() - 1];
    let c = if buf.len() <= KARATSUBA_SQR_CUTOFF {
        sqr_buf(buf, &mut out)
    } else {
        let mut scratch = vec![0_u64; find_karatsuba_sqr_scratch_size(buf.len())];
        let half_len = (buf.len() + 1) / 2;
        let (cross, rest) = scratch.split_at_mut(2 * half_len + 1);
        karatsuba_sqr_core(buf, half_len, &mut out, cross, rest)
    };
    return (out, c);
}

pub fn sqr_arr<const N: usize>(buf: &[u64]) -> Result<([u64; N], u64), ()> {
    if 2 * buf.len() - 1 > N {
        return Err(());
    }

    let mut out = [0_u64; N];
    let c = if buf.len() <= KARATSUBA_SQR_CUTOFF {
        sqr_buf(buf, &mut out)
    } else {
        let mut scratch = [0_u64; N];
        let scratch_sz = find_karatsuba_sqr_scratch_size(buf.len());
        let half_len = (buf.len() + 1) / 2;
        if scratch_sz > N {
            let mut cross = [0; N];
            karatsuba_sqr_core(buf, half_len, &mut out, &mut cross, &mut scratch)
        } else {
            let (cross, rest) = scratch.split_at_mut(2 * half_len + 1);
            karatsuba_sqr_core(buf, half_len, &mut out, cross, rest)
        }
    };
    return Ok((out, c));
}

fn short_mul_buf(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let a_len = a.len() - 1;
    let b_len = b.len() - 1;
    let d = (a_len + b_len + 1).saturating_sub(out.len());

    let mask = u64::MAX as u128;
    let mut carry: u128 = 0;
    for n in d..(d + out.len()) {
        let mut term: u128 = carry;
        carry = 0;
        let mi = n.saturating_sub(b_len);
        let mf = n.min(a_len);
        for m in mi..=mf {
            let val = (a[m] as u128) * (b[n - m] as u128);
            term += val & mask;
            carry += val >> 64;
        }
        carry += term >> 64;
        out[n - d] = term as u64;
    }

    return carry as u64;
}

pub(crate) const SHORT_CUTOFF: usize = 32;

pub fn short_mul_vec(a: &[u64], b: &[u64], prec: usize) -> (Vec<u64>, u64) {
    let p = prec.min(a.len() + b.len() - 1);
    return if p <= SHORT_CUTOFF {
        let mut out = vec![0; p];
        let c = short_mul_buf(a, b, &mut out);
        (out, c)
    } else {
        let ap = a.len().min(p);
        let bp = b.len().min(p);
        let (mut out, c) = mul_vec(&a[a.len() - ap..], &b[b.len() - bp..]);
        out.drain(..out.len().saturating_sub(p));
        (out, c)
    };
}

fn short_sqr_buf(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let len = buf.len() - 1;
    let mask = u64::MAX as u128;
    let d = (2 * len + 1).saturating_sub(out.len());

    let mut carry = if d == 0 {
        let init_term = (buf[0] as u128) * (buf[0] as u128);
        out[0] = init_term as u64;
        init_term >> 64
    } else {
        0
    };
    for n in d.max(1)..(d + out.len()) {
        let mut term = carry;
        carry = 0;

        let mi = n.saturating_sub(len);
        let mf = ((n - 1) / 2).min(len);
        for m in mi..=mf {
            let val = (buf[m] as u128) * (buf[n - m] as u128);
            term += (val << 1) & mask;
            carry += val >> 63;
        }

        if n % 2 == 0 {
            let half_val = buf[n / 2] as u128;
            let val = half_val * half_val;
            term += val & mask;
            carry += val >> 64;
        }

        out[n - d] = term as u64;
        carry += term >> 64;
    }

    return carry as u64;
}

pub(crate) const SHORT_SQR_CUTOFF: usize = 32;

pub fn short_sqr_vec(buf: &[u64], prec: usize) -> (Vec<u64>, u64) {
    let p = prec.min(2 * buf.len() - 1);
    return if p <= SHORT_SQR_CUTOFF {
        let mut out = vec![0; p];
        let c = short_sqr_buf(buf, &mut out);
        (out, c)
    } else {
        let bufp = buf.len().min(p);
        let (mut out, c) = sqr_vec(&buf[buf.len() - bufp..]);
        out.drain(..out.len().saturating_sub(p));
        (out, c)
    };
}

pub fn powi_vec(buf: &[u64], pow: usize) -> Vec<u64> {
    if pow == 0 {
        return vec![1];
    }

    let l = buf_len(buf);
    let log2 = (l - 1) * 64 + buf[l - 1].ilog2() as usize;
    let tmp_sz = 1 + (log2 * pow) / 64;
    let scratch_sz = std::cmp::max(
        find_karatsuba_sqr_scratch_size(tmp_sz / 2),
        find_karatsuba_scratch_size(tmp_sz, l),
    );
    let mut tmp = vec![0_u64; tmp_sz];
    let mut out = vec![0_u64; tmp_sz];
    let mut scratch = vec![0_u64; scratch_sz];

    let mut len = buf.len();
    out[..len].copy_from_slice(buf);
    let mut io = true;

    let log = pow.ilog2() as usize;
    for i in (0..log).rev() {
        let sqr_len = 2 * len - 1;
        let (src, dst): (&[u64], &mut [u64]) = if io {
            (&out, &mut tmp)
        } else {
            (&tmp, &mut out)
        };
        let sqr_c = karatsuba_sqr_alg(&src[..len], &mut dst[..sqr_len], &mut scratch);
        len = sqr_len;
        if sqr_c > 0 {
            dst[len] = sqr_c;
            len += 1;
        }
        io = !io;

        if (pow >> i) & 1 == 1 {
            let mul_len = len + buf.len() - 1;
            let (src, dst): (&[u64], &mut [u64]) = if io {
                (&out, &mut tmp)
            } else {
                (&tmp, &mut out)
            };

            let mul_c = karatsuba_alg(&src[..len], &buf, &mut dst[..mul_len], &mut scratch);
            len = mul_len;
            if mul_c > 0 {
                dst[len] = mul_c;
                len += 1
            }
            io = !io;
        }
    }

    if !io {
        out[..len].copy_from_slice(&tmp[..len]);
    }
    trim_lz(&mut out);
    return out;
}

pub fn powi_arr<const N: usize>(buf: &[u64], pow: usize) -> Result<[u64; N], ()> {
    let l = buf_len(buf);
    let log2 = (l - 1) * 64 + buf[l - 1].ilog2() as usize;
    let tmp_sz = (log2 * pow) / 64;
    if tmp_sz > N {
        return Err(());
    }

    let mut out = [0; N];
    let mut tmp = [0; N];
    let mut scratch = [0; N];

    let buf_len = buf_len(buf);
    let mut len = buf_len;
    out[..len].copy_from_slice(&buf[..len]);
    let mut io = true;

    let log = pow.ilog2();
    for i in (1..log).rev() {
        let sqr_len = 2 * len - 1;
        let (src, dst): (&[u64], &mut [u64]) = if io {
            (&out, &mut tmp)
        } else {
            (&tmp, &mut out)
        };

        let sqr_c = karatsuba_sqr_alg(&src[..len], &mut dst[..sqr_len], &mut scratch);
        len = sqr_len;
        if sqr_c > 0 {
            dst[len] = sqr_c;
            len += 1;
        }
        io = !io;
        if (pow >> i) & 1 == 1 {
            let mul_len = len + buf_len - 1;
            let (src, dst): (&[u64], &mut [u64]) = if io {
                (&out, &mut tmp)
            } else {
                (&tmp, &mut out)
            };

            let mul_c = karatsuba_alg(
                &src[..len],
                &buf[..buf_len],
                &mut dst[..mul_len],
                &mut scratch,
            );
            len = mul_len;
            if mul_c > 0 {
                dst[len] = mul_c;
                len += 1
            }
            io = !io;
        }
    }

    let sqr_len = 2 * len - 1;
    if sqr_len > N {
        return Err(());
    }

    let (src, dst): (&[u64], &mut [u64]) = if io {
        (&out, &mut tmp)
    } else {
        (&tmp, &mut out)
    };

    let sqr_c = if find_karatsuba_sqr_scratch_size(len) > N {
        let mut cross = [0; N];
        karatsuba_sqr_core(
            &src[..len],
            (len + 1) / 2,
            &mut dst[..sqr_len],
            &mut cross,
            &mut scratch,
        )
    } else {
        karatsuba_sqr_alg(&src[..len], &mut dst[..sqr_len], &mut scratch)
    };

    len = sqr_len;
    if sqr_c > 0 {
        if len == N {
            return Err(());
        }
        dst[len] = sqr_c;
        len += 1;
    }
    io = !io;

    if pow & 1 == 1 {
        let mul_len = len + buf_len - 1;
        if mul_len > N {
            return Err(());
        }

        let (src, dst): (&[u64], &mut [u64]) = if io {
            (&out, &mut tmp)
        } else {
            (&tmp, &mut out)
        };

        let mul_c = if find_karatsuba_scratch_size(len, buf_len) > N {
            let mut cross = [0; N];
            karatsuba_core(
                &src[..len],
                &buf[..buf_len],
                (len + 1) / 2,
                &mut dst[..mul_len],
                &mut cross,
                &mut scratch,
            )
        } else {
            karatsuba_alg(
                &src[..len],
                &buf[..buf_len],
                &mut dst[..mul_len],
                &mut scratch,
            )
        };

        len = mul_len;
        if mul_c > 0 {
            if len == N {
                return Err(());
            }
            dst[len] = mul_c;
        }
        io = !io;
    }

    if !io {
        out.copy_from_slice(&tmp);
    }
    return Ok(out);
}
