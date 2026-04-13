#![allow(dead_code)]
use crate::utils::{utils::*, *};
use rustfft::num_traits::{One, Zero};
use rustfft::{num_complex::Complex, Fft, FftDirection, FftPlanner};
use std::arch::asm;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::marker::PhantomData;
use std::ops::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::sync::Arc;

// SCHOOL BOOK MULTIPLICATION
// Worst complexity lowest overhead optimized for small inputs. Base tier algorithm.
// General: O(ls)
// Primitive Optimizations: O(l)

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
    if buf.is_empty() {
        return 0;
    }
    if prim == 0 {
        buf.fill(0);
        return 0;
    }
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
    if prim == 0 {
        buf.fill(0);
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

//KARATSUBA MULTIPLICATION
// Mid tier complexity overhead is still relatively small, Mid tier algorithm. It handles unbalanced inputs through a
// chunking and non-chunking version. The chunking boundary happens at 2s = l this is given to us
// by the nature of algorithm and doesn't need to be determined through testing.
// Time:
// Karatsuba: O(l^a - (l-s)^a)
// Chunking Karatsuba: O(ls^(a-1))
// where a = log_2(3) = 1.586...
// Space:
// Karatsuba: Omega(2l + 6log_2(l))
// Chunking Karatsuba: Omega(4s + 6log_2(s))

fn karatsuba_core(
    a: &[u64],
    b: &[u64],
    half: usize,       // (long + 1) / 2
    out: &mut [u64],   // >= long + short - 1
    cross: &mut [u64], // >= 2*half + 1
    scratch: &mut [u64],
) -> u64 {
    let (a0, a1) = a.split_at(half);
    let (b0, b1) = b.split_at(half);

    {
        let (a_sum, b_sum) = out[..2 * half + 2].split_at_mut(half + 1);
        let mut l_len = half;
        let mut s_len = half;

        a_sum[..l_len].copy_from_slice(a0);
        if add_buf(&mut a_sum[..l_len], a1) {
            a_sum[l_len] = 1;
            l_len += 1;
        }

        b_sum[..s_len].copy_from_slice(b0);
        if add_buf(&mut b_sum[..s_len], b1) {
            b_sum[s_len] = 1;
            s_len += 1;
        }

        karatsuba_mul(&a_sum[..l_len], &b_sum[..s_len], cross, scratch);
    }

    let (z0, z2) = out.split_at_mut(2 * half);
    karatsuba_mul(a0, b0, z0, scratch);
    let mut overflow = karatsuba_mul(a1, b1, z2, scratch);
    let z2_len = buf_len(z2);

    sub_buf(cross, z0);
    sub_buf(cross, &z2[..z2_len]);
    if overflow > 0 {
        sub_prim(&mut cross[z2_len..], overflow);
    }
    if add_buf(&mut out[half..], &cross) {
        overflow += 1;
    }

    return overflow;
}

fn chunking_karatsuba(long: &[u64], short: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    out.fill(0);
    let s = short.len();
    let l = long.len();
    let half = (s + 1) / 2;
    let mut chunks = l / s;
    if l % s == 0 {
        chunks -= 1;
    }
    {
        let (val_cross, rest) = scratch.split_at_mut(2 * s + 2 * half + 1);
        let (cross, val) = val_cross.split_at_mut(2 * half + 1);
        for i in 0..chunks {
            let of = s * i;
            karatsuba_core(&long[of..of + s], short, half, val, cross, rest);
            add_buf(&mut out[of..], &val);
        }
    }
    let of = chunks * s;
    let last_chunk = &long[of..];
    let (val, rest) = scratch.split_at_mut(last_chunk.len() + s - 1);
    let mut overflow = karatsuba_mul(last_chunk, short, val, rest);
    if add_buf(&mut out[of..], val) {
        overflow += 1;
    }
    return overflow;
}

enum KDispatch {
    Prim,
    Prim2,
    School,
    Chunking,
    Recurse,
}

pub fn is_school(l: usize, s: usize) -> bool {
    let half = (l + 1) / 2;
    if s <= half {
        s <= CHUNKING_KARATSUBA_CUTOFF
    } else {
        let c = KARATSUBA_CUTOFF as f64;
        let val = l as f64 - 1.5 * c;
        let boundary = (1.125 * c - val * val / (2.0 * c)).ceil() as usize;
        s <= boundary
    }
}

fn kdispatch(l: usize, s: usize) -> KDispatch {
    if s == 1 {
        KDispatch::Prim
    } else if s == 2 {
        KDispatch::Prim2
    } else if is_school(l, s) {
        KDispatch::School
    } else if s <= (l + 1) / 2 {
        KDispatch::Chunking
    } else {
        KDispatch::Recurse
    }
}

fn karatsuba_mul(a: &[u64], b: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    match kdispatch(long.len(), short.len()) {
        KDispatch::Prim => {
            out[..long.len()].copy_from_slice(long);
            return mul_prim(out, short[0]);
        }
        KDispatch::Prim2 => {
            out[..long.len()].copy_from_slice(long);
            return mul_prim2(out, combine_u64(short[0], short[1])) as u64;
        }
        KDispatch::School => {
            return mul_buf(long, short, out);
        }
        KDispatch::Chunking => return chunking_karatsuba(long, short, out, scratch),
        KDispatch::Recurse => {}
    }
    let half = (long.len() + 1) / 2;
    let (cross, rest) = scratch.split_at_mut(2 * half + 1);
    karatsuba_core(long, short, half, out, cross, rest)
}

pub fn find_karatsuba_scratch(l: usize, s: usize) -> usize {
    match kdispatch(l, s) {
        KDispatch::Prim | KDispatch::Prim2 | KDispatch::School => 0,
        KDispatch::Chunking => {
            let half = (s + 1) / 2;
            2 * s + 2 * half + 1 + find_karatsuba_scratch(s, s)
        }
        KDispatch::Recurse => {
            let half = (l + 1) / 2;
            2 * half + 1 + find_karatsuba_scratch(half + 1, half + 1)
        }
    }
}

pub fn karatsuba_entry_dyn(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let mut scratch_guard = ScratchGuard::acquire();
    let scratch_sz = find_karatsuba_scratch(long.len(), short.len());
    karatsuba_mul(long, short, out, scratch_guard.get(scratch_sz))
}

pub fn karatsuba_entry_static<const N: usize>(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let scratch_sz = find_karatsuba_scratch(long.len(), short.len());
    let mut scratch = [0; N];
    if scratch_sz < N {
        karatsuba_mul(long, short, out, &mut scratch)
    } else {
        let mut cross = [0; N];
        let half = (long.len() + 1) / 2;
        karatsuba_core(
            long,
            short,
            half,
            out,
            &mut cross[..2 * half + 1],
            &mut scratch,
        )
    }
}

//FFT MULTIPLICATION
// Low tier complexity overhead is very high, High tier algorithm. It handles unbalanced inputs
// through a chunking and non-chunking version. The chunking version is recursive. The chunking
// boundary happens at F_0 s^2 = l , F_0 > 4 , the smaller the overhead of chunking the closer F_0
// is too 4.
// Time:
// FFT: O((l+s)log(l+s))
// Chunking FFT: O(2l*log(2s))
// Space:
// FFT:
//  Complex Space: O(l+s)
// Chunking FFT:
//  Complex Space: O(2s)
//  Real Space: O(2s)

struct FTTTwidleTower {
    table: Vec<Complex<f64>>,
    veiw: Vec<Complex<f64>>,
    res: usize,
    gen: u8,
    base: usize,
}

impl FTTTwidleTower {
    const MAX_GEN: u8 = 3;

    fn build(n: usize) -> Self {
        let len = n / 4 + 1;
        let th = 2.0 * PI / (n as f64);
        let mut table = Vec::<Complex<f64>>::with_capacity(len);
        table.push(Complex::new(1.0, 0.0));
        for k in 1..len {
            table.push(Complex::cis(th * k as f64));
        }

        FTTTwidleTower {
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
        new_table.push(Complex { re: 1.0, im: 0.0 });
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
    twidles: HashMap<usize, FTTTwidleTower>,
}

impl FFTCache {
    fn new() -> Self {
        FFTCache {
            planner: FftPlanner::new(),
            twidles: HashMap::new(),
        }
    }

    fn prep_mul(
        &mut self,
        n: usize,
    ) -> (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>, &[Complex<f64>], usize) {
        let fwd = self.planner.plan_fft_forward(n);
        let bwd = self.planner.plan_fft_inverse(n / 2);
        let tw = self
            .twidles
            .entry(n >> n.trailing_zeros())
            .or_insert_with(|| FTTTwidleTower::build(n))
            .get(n);
        let scratch_sz = n + fwd
            .get_inplace_scratch_len()
            .max(bwd.get_inplace_scratch_len());
        (fwd, bwd, tw, 2 * scratch_sz)
    }

    fn prep_sqr(
        &mut self,
        m: usize,
    ) -> (Arc<dyn Fft<f64>>, Arc<dyn Fft<f64>>, &[Complex<f64>], usize) {
        let fwd = self.planner.plan_fft_forward(m);
        let bwd = self.planner.plan_fft_inverse(m);
        let tw = self
            .twidles
            .entry(m >> m.trailing_zeros())
            .or_insert_with(|| FTTTwidleTower::build(m))
            .get(m);
        let scratch_sz = m + fwd
            .get_inplace_scratch_len()
            .max(bwd.get_inplace_scratch_len());
        (fwd, bwd, tw, 2 * scratch_sz)
    }
}

thread_local! {
    static FFT_CACHE: RefCell<FFTCache> = RefCell::new(FFTCache::new());
}

const FFT_RADIX: [usize; 3] = [3, 5, 7];

const fn log_prim_search(init: usize, target: usize, mut idx: usize, primes: &[usize]) -> usize {
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
    let p = primes[idx];
    let mut val = init;
    let mut best = log_prim_search(val, target, idx, primes);
    if best == target {
        return best;
    }
    val *= p;
    while val < best {
        let cur = log_prim_search(val, target, idx, primes);
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
    return log_prim_search(INIT, n, FFT_RADIX.len(), &FFT_RADIX);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_and_round_x86(x: &mut [Complex<f64>], scale: f64) {
    let coefs = std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2);
    let divisor = _mm256_set1_pd(scale);
    const ROUND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    let chunks = coefs.len() / 4;
    let ptr = coefs.as_mut_ptr();
    for i in 0..chunks {
        let offset = i * 4;
        // Load 4 f64s
        let v = _mm256_loadu_pd(ptr.add(offset));
        // Divide by scale
        let scaled = _mm256_div_pd(v, divisor);
        // Round
        let rounded = _mm256_round_pd::<ROUND>(scaled);
        // Store
        _mm256_storeu_pd(ptr.add(offset), rounded);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_and_round_aarch(x: &mut [Complex<f64>], scale: f64) {
    let coefs = std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2);
    let divisor = vdupq_n_f64(scale);
    let chunks = coefs.len() / 2;
    let ptr = coefs.as_mut_ptr();
    for i in 0..chunks {
        let offset = i * 2;
        // Load 2 f64s
        let v = vld1q_f64(ptr.add(offset));
        // Divide by scale
        let scaled = vdivq_f64(v, divisor);
        // Round
        let rounded = vrndnq_f64(scaled);
        vst1q_f64(ptr.add(offset), rounded);
    }
}

fn scale_and_round_standard(x: &mut [Complex<f64>], scale: f64) {
    let coefs = unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2) };
    for c in coefs {
        *c /= scale;
        *c = c.round_ties_even();
    }
}

fn scale_and_round(x: &mut [Complex<f64>], scale: f64) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            scale_and_round_x86(x, scale);
        }
    } else {
        scale_and_round_standard(x, scale);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        scale_and_round_aarch(x, scale);
    }
}

#[inline(always)]
fn separate_mul(x: Complex<f64>, y: Complex<f64>) -> Complex<f64> {
    let ar = x.re + y.re;
    let ai = x.im - y.im;
    let br = x.im + y.im;
    let bi = y.re - x.re;
    Complex::new(ar * br - ai * bi, ar * bi + ai * br)
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
        x[k] = Complex::new(s.re - dw.im, s.im + dw.re).scale(0.25);
        x[m - k] = Complex::new(s.re + dw.im, dw.re - s.im).scale(0.25);
    }
    let k = m / 2;
    x[k] = separate_mul(x[k], x[n - k]).conj().scale(0.5);
}

fn fft_core(
    x: &mut [Complex<f64>],
    fwd: &dyn Fft<f64>,
    bwd: &dyn Fft<f64>,
    tw: &[Complex<f64>],
    fft_scratch: &mut [Complex<f64>],
) {
    let n = x.len();
    fwd.process_with_scratch(x, fft_scratch);
    recombine(x, n / 2, n, tw);
    bwd.process_with_scratch(&mut x[..n / 2], fft_scratch);
    scale_and_round(x, n as f64);
}

enum BitSize {
    Bit16,
    Bit8,
}

fn bit_sz_dispatch(s: usize) -> BitSize {
    if s < FFT_BIT_CUTOFF {
        return BitSize::Bit16;
    } else {
        return BitSize::Bit8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decompose_16_x86(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;

    for (&l, &s) in long.iter().zip(short) {
        // Broadcast into both 64-bit lanes of a 128-bit register
        // [u16_0, u16_1, u16_2, u16_3] → [u32_0, u32_1, u32_2, u32_3]
        // [u32_0, u32_1, u32_2, u32_3] → [f64_0, f64_1, f64_2, f64_3]
        let l_broadcast = _mm_set1_epi64x(l as i64);
        let l_as_u32s = _mm_cvtepu16_epi32(l_broadcast);
        let l_f64s = _mm256_cvtepi32_pd(l_as_u32s);
        let s_broadcast = _mm_set1_epi64x(s as i64);
        let s_as_u32s = _mm_cvtepu16_epi32(s_broadcast);
        let s_f64s = _mm256_cvtepi32_pd(s_as_u32s);
        // result: [l0, s0, l2, s2]
        let lo = _mm256_unpacklo_pd(l_f64s, s_f64s);
        // result: [l1, s1, l3, s3]
        let hi = _mm256_unpackhi_pd(l_f64s, s_f64s);
        // result: [l0, s0, l1, s1]
        let first_four = _mm256_permute2f128_pd(lo, hi, 0x20);
        // result: [l2, s2, l3, s3]
        let second_four = _mm256_permute2f128_pd(lo, hi, 0x31);
        // Store 8 f64s = 4 Complex<f64> values
        _mm256_storeu_pd(out.add(idx), first_four);
        _mm256_storeu_pd(out.add(idx + 4), second_four);
        idx += 8;
    }
    let zero = _mm256_setzero_pd();
    for &l in &long[short.len()..] {
        let l_broadcast = _mm_set1_epi64x(l as i64);
        let l_as_u32s = _mm_cvtepu16_epi32(l_broadcast);
        let l_f64s = _mm256_cvtepi32_pd(l_as_u32s);
        // Interleave with zeros: same logic but s_f64s is replaced by zero
        let lo = _mm256_unpacklo_pd(l_f64s, zero);
        let hi = _mm256_unpackhi_pd(l_f64s, zero);
        let first_four = _mm256_permute2f128_pd(lo, hi, 0x20);
        let second_four = _mm256_permute2f128_pd(lo, hi, 0x31);
        _mm256_storeu_pd(out.add(idx), first_four);
        _mm256_storeu_pd(out.add(idx + 4), second_four);
        idx += 8;
    }

    // Zero any remaining output
    let total = x.len() * 2;
    if idx < total {
        std::ptr::write_bytes(out.add(idx), 0, total - idx);
    }
}

//converts u64 to f64 in 16 bit chunks
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn u64_into_2x2f64s_aarch(x: u64) -> (float64x2_t, float64x2_t) {
    // Load into a 64-bit NEON register
    let x_raw = vdup_n_u64(x);
    // Reinterpret the same 64 bits as 4 × u16
    let x_u16x4 = vreinterpret_u16_u64(x_raw);
    // Zero Extend: 4 x u16 -> 4 x u32
    let x_u32x4 = vmovl_u16(x_u16x4);
    // Split: 4 x u32 -> 2 x u32 + 2 x u32
    let (x_u32_lo, x_u32_hi) = (vget_low_u32(x_u32x4), vget_high_u32(x_u32x4));
    // Zero Extend: 2 x u32 -> 2 x u64
    let x_u64_lo = vmovl_u32(x_u32_lo);
    let x_u64_hi = vmovl_u32(x_u32_hi);
    // Float Cast: 2 x u64 -> 2 x f64
    let x_f0 = vcvtq_f64_u64(x_u64_lo);
    let x_f1 = vcvtq_f64_u64(x_u64_hi);
    (x_f0, x_f1)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decompose_16_aarch(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;

    for (&l, &s) in long.iter().zip(short) {
        let (l_f0, l_f1) = u64_into_2x2f64s_aarch(l);
        let (s_f0, s_f1) = u64_into_2x2f64s_aarch(s);
        // [l0, l1] × [s0, s1] → [l0, s0]
        let pair0 = vzip1q_f64(l_f0, s_f0);
        // [l0, l1] × [s0, s1] → [l1, s1]
        let pair1 = vzip2q_f64(l_f0, s_f0);
        // [l2, l3] × [s2, s3] → [l2, s2]
        let pair2 = vzip1q_f64(l_f1, s_f1);
        // [l2, l3] × [s2, s3] → [l3, s3]
        let pair3 = vzip2q_f64(l_f1, s_f1);
        // Store 8 f64s = 4 Complex<f64> values
        vst1q_f64(out.add(idx), pair0);
        vst1q_f64(out.add(idx + 2), pair1);
        vst1q_f64(out.add(idx + 4), pair2);
        vst1q_f64(out.add(idx + 6), pair3);
        idx += 8;
    }

    let zero = vdupq_n_f64(0.0);
    for &l in &long[short.len()..] {
        let (l_f0, l_f1) = u64_into_2x2f64s_aarch(l);
        // zip with zero register for im = 0.0
        vst1q_f64(out.add(idx), vzip1q_f64(l_f0, zero));
        vst1q_f64(out.add(idx + 2), vzip2q_f64(l_f0, zero));
        vst1q_f64(out.add(idx + 4), vzip1q_f64(l_f1, zero));
        vst1q_f64(out.add(idx + 6), vzip2q_f64(l_f1, zero));
        idx += 8;
    }

    let total = x.len() * 2;
    if idx < total {
        std::ptr::write_bytes(out.add(idx), 0, total - idx);
    }
}

fn decompose_16_standard(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
    const MASK: u64 = u16::MAX as u64;
    let mut idx = 0;
    for (&l, &s) in long.iter().zip(short) {
        x[idx].re = (l & MASK) as f64;
        x[idx].im = (s & MASK) as f64;
        x[idx + 1].re = (l >> 16 & MASK) as f64;
        x[idx + 1].im = (s >> 16 & MASK) as f64;
        x[idx + 2].re = (l >> 32 & MASK) as f64;
        x[idx + 2].im = (s >> 32 & MASK) as f64;
        x[idx + 3].re = (l >> 48) as f64;
        x[idx + 3].im = (s >> 48) as f64;
        idx += 4;
    }
    for &l in &long[short.len()..] {
        x[idx].re = (l & MASK) as f64;
        x[idx].im = 0.0;
        x[idx + 1].re = (l >> 16 & MASK) as f64;
        x[idx + 1].im = 0.0;
        x[idx + 2].re = (l >> 32 & MASK) as f64;
        x[idx + 2].im = 0.0;
        x[idx + 3].re = (l >> 48) as f64;
        x[idx + 3].im = 0.0;
        idx += 4;
    }
    if idx < x.len() {
        x[idx..].fill(Complex::zero());
    }
}

fn decompose_16(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            decompose_16_x86(long, short, x);
        }
    } else {
        decompose_16_standard(long, short, x);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        decompose_16_aarch(long, short, x);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decompose_8_x86(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;

    for (&l, &s) in long.iter().zip(short) {
        let lv = _mm_set1_epi64x(l as i64);
        let l_lo_u32 = _mm_cvtepu8_epi32(lv);
        let l_lo_f64 = _mm256_cvtepi32_pd(l_lo_u32);
        let lv_shifted = _mm_bsrli_si128(lv, 4);
        let l_hi_u32 = _mm_cvtepu8_epi32(lv_shifted);
        let l_hi_f64 = _mm256_cvtepi32_pd(l_hi_u32);

        let sv = _mm_set1_epi64x(s as i64);
        let s_lo_u32 = _mm_cvtepu8_epi32(sv);
        let s_lo_f64 = _mm256_cvtepi32_pd(s_lo_u32);
        let sv_shifted = _mm_bsrli_si128(sv, 4);
        let s_hi_u32 = _mm_cvtepu8_epi32(sv_shifted);
        let s_hi_f64 = _mm256_cvtepi32_pd(s_hi_u32);

        // [l0, s0, l2, s2]
        let lo_lo = _mm256_unpacklo_pd(l_lo_f64, s_lo_f64);
        // [l1, s1, l3, s3]
        let lo_hi = _mm256_unpackhi_pd(l_lo_f64, s_lo_f64);
        // low lanes of both → [l0, s0, l1, s1]
        let lo_first = _mm256_permute2f128_pd(lo_lo, lo_hi, 0x20);
        // high lanes of both → [l2, s2, l3, s3]
        let lo_second = _mm256_permute2f128_pd(lo_lo, lo_hi, 0x31);

        _mm256_storeu_pd(out.add(idx), lo_first);
        _mm256_storeu_pd(out.add(idx + 4), lo_second);

        // ---- Interleave high group (bytes 4-7 from each) ----

        let hi_lo = _mm256_unpacklo_pd(l_hi_f64, s_hi_f64);
        let hi_hi = _mm256_unpackhi_pd(l_hi_f64, s_hi_f64);
        let hi_first = _mm256_permute2f128_pd(hi_lo, hi_hi, 0x20);
        let hi_second = _mm256_permute2f128_pd(hi_lo, hi_hi, 0x31);

        _mm256_storeu_pd(out.add(idx + 8), hi_first);
        _mm256_storeu_pd(out.add(idx + 12), hi_second);

        idx += 16;
    }

    // ---- Overhang ----

    let zero = _mm256_setzero_pd();
    for &l in &long[short.len()..] {
        let lv = _mm_set1_epi64x(l as i64);

        let l_lo_f64 = _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(lv));
        let l_hi_f64 = _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_bsrli_si128(lv, 4)));

        let lo_lo = _mm256_unpacklo_pd(l_lo_f64, zero);
        let lo_hi = _mm256_unpackhi_pd(l_lo_f64, zero);
        _mm256_storeu_pd(out.add(idx), _mm256_permute2f128_pd(lo_lo, lo_hi, 0x20));
        _mm256_storeu_pd(out.add(idx + 4), _mm256_permute2f128_pd(lo_lo, lo_hi, 0x31));

        let hi_lo = _mm256_unpacklo_pd(l_hi_f64, zero);
        let hi_hi = _mm256_unpackhi_pd(l_hi_f64, zero);
        _mm256_storeu_pd(out.add(idx + 8), _mm256_permute2f128_pd(hi_lo, hi_hi, 0x20));
        _mm256_storeu_pd(
            out.add(idx + 12),
            _mm256_permute2f128_pd(hi_lo, hi_hi, 0x31),
        );

        idx += 16;
    }

    let total = x.len() * 2;
    if idx < total {
        std::ptr::write_bytes(out.add(idx), 0, total - idx);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn u64_into_2x4f64_aarch(l: u64) -> (float64x2_t, float64x2_t, float64x2_t, float64x2_t) {
    let l_raw = vdup_n_u64(l);
    let l_u8x8 = vreinterpret_u8_u64(l_raw);
    let l_u16x8 = vmovl_u8(l_u8x8);
    let (l_u16_lo, l_u16_hi) = (vget_low_u16(l_u16x8), vget_high_u16(l_u16x8));
    let l_u32_lo = vmovl_u16(l_u16_lo);
    let l_u32_hi = vmovl_u16(l_u16_hi);
    let (l_u32_lo_lo, l_u32_lo_hi) = (vget_low_u32(l_u32_lo), vget_high_u32(l_u32_lo));
    let (l_u32_hi_lo, l_u32_hi_hi) = (vget_low_u32(l_u32_hi), vget_high_u32(l_u32_hi));
    let l_u64_lo_lo = vmovl_u32(l_u32_lo_lo);
    let l_u64_lo_hi = vmovl_u32(l_u32_lo_hi);
    let l_u64_hi_lo = vmovl_u32(l_u32_hi_lo);
    let l_u64_hi_hi = vmovl_u32(l_u32_hi_hi);
    (
        vcvtq_f64_u64(l_u64_lo_lo),
        vcvtq_f64_u64(l_u64_lo_hi),
        vcvtq_f64_u64(l_u64_hi_lo),
        vcvtq_f64_u64(l_u64_hi_hi),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decompose_8_aarch(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;

    for (&l, &s) in long.iter().zip(short) {
        let (l_f0, l_f1, l_f2, l_f3) = u64_into_2x4f64_aarch(l);
        let (s_f0, s_f1, s_f2, s_f3) = u64_into_2x4f64_aarch(s);
        vst1q_f64(out.add(idx), vzip1q_f64(l_f0, s_f0)); // [l0, s0]
        vst1q_f64(out.add(idx + 2), vzip2q_f64(l_f0, s_f0)); // [l1, s1]
        vst1q_f64(out.add(idx + 4), vzip1q_f64(l_f1, s_f1)); // [l2, s2]
        vst1q_f64(out.add(idx + 6), vzip2q_f64(l_f1, s_f1)); // [l3, s3]
        vst1q_f64(out.add(idx + 8), vzip1q_f64(l_f2, s_f2)); // [l4, s4]
        vst1q_f64(out.add(idx + 10), vzip2q_f64(l_f2, s_f2)); // [l5, s5]
        vst1q_f64(out.add(idx + 12), vzip1q_f64(l_f3, s_f3)); // [l6, s6]
        vst1q_f64(out.add(idx + 14), vzip2q_f64(l_f3, s_f3)); // [l7, s7]
        idx += 16;
    }

    let zero = vdupq_n_f64(0.0);
    for &l in &long[short.len()..] {
        let (l_f0, l_f1, l_f2, l_f3) = u64_into_2x4f64_aarch(l);
        vst1q_f64(out.add(idx), vzip1q_f64(l_f0, zero));
        vst1q_f64(out.add(idx + 2), vzip2q_f64(l_f0, zero));
        vst1q_f64(out.add(idx + 4), vzip1q_f64(l_f1, zero));
        vst1q_f64(out.add(idx + 6), vzip2q_f64(l_f1, zero));
        vst1q_f64(out.add(idx + 8), vzip1q_f64(l_f2, zero));
        vst1q_f64(out.add(idx + 10), vzip2q_f64(l_f2, zero));
        vst1q_f64(out.add(idx + 12), vzip1q_f64(l_f3, zero));
        vst1q_f64(out.add(idx + 14), vzip2q_f64(l_f3, zero));
        idx += 16;
    }

    let total = x.len() * 2;
    if idx < total {
        std::ptr::write_bytes(out.add(idx), 0, total - idx);
    }
}

fn decompose_8_standard(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
    const MASK: u64 = u8::MAX as u64;
    let mut idx = 0;
    for (&l, &s) in long.iter().zip(short) {
        x[idx].re = (l & MASK) as f64;
        x[idx].im = (s & MASK) as f64;
        x[idx + 1].re = (l >> 8 & MASK) as f64;
        x[idx + 1].im = (s >> 8 & MASK) as f64;
        x[idx + 2].re = (l >> 16 & MASK) as f64;
        x[idx + 2].im = (s >> 16 & MASK) as f64;
        x[idx + 3].re = (l >> 24 & MASK) as f64;
        x[idx + 3].im = (s >> 24 & MASK) as f64;
        x[idx + 4].re = (l >> 32 & MASK) as f64;
        x[idx + 4].im = (s >> 32 & MASK) as f64;
        x[idx + 5].re = (l >> 40 & MASK) as f64;
        x[idx + 5].im = (s >> 40 & MASK) as f64;
        x[idx + 6].re = (l >> 48 & MASK) as f64;
        x[idx + 6].im = (s >> 48 & MASK) as f64;
        x[idx + 7].re = (l >> 56 & MASK) as f64;
        x[idx + 7].im = (s >> 56 & MASK) as f64;
        idx += 8;
    }
    for &l in &long[short.len()..] {
        x[idx].re = (l & MASK) as f64;
        x[idx].im = 0.0;
        x[idx + 1].re = (l >> 8 & MASK) as f64;
        x[idx + 1].im = 0.0;
        x[idx + 2].re = (l >> 16 & MASK) as f64;
        x[idx + 2].im = 0.0;
        x[idx + 3].re = (l >> 24 & MASK) as f64;
        x[idx + 3].im = 0.0;
        x[idx + 4].re = (l >> 32 & MASK) as f64;
        x[idx + 4].im = 0.0;
        x[idx + 5].re = (l >> 40 & MASK) as f64;
        x[idx + 5].im = 0.0;
        x[idx + 6].re = (l >> 48 & MASK) as f64;
        x[idx + 6].im = 0.0;
        x[idx + 7].re = (l >> 56 & MASK) as f64;
        x[idx + 7].im = 0.0;
        idx += 8;
    }
    if idx < x.len() {
        x[idx..].fill(Complex::zero());
    }
}

fn decompose_8(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { decompose_8_x86(long, short, x) }
    } else {
        decompose_8_standard(long, short, x);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        decompose_8_aarch(long, short, x);
    }
}

fn accumulate_16(x: &[Complex<f64>], out: &mut [u64]) -> u64 {
    let coefs = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len() * 2) };
    let mut carry = 0;
    for (chunk, elem) in coefs.chunks(4).zip(out.iter_mut()) {
        let mut tot = carry;
        for (i, c) in chunk.iter().enumerate() {
            let val = unsafe { c.to_int_unchecked::<u64>() } as u128;
            tot += val << (i * 16)
        }
        *elem = tot as u64;
        carry = tot >> 64;
    }
    return carry as u64;
}

fn accumulate_8(x: &[Complex<f64>], out: &mut [u64]) -> u64 {
    let coefs = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len() * 2) };
    let mut carry = 0;
    for (chunk, elem) in coefs.chunks(8).zip(out.iter_mut()) {
        let mut tot = carry;
        for (i, c) in chunk.iter().enumerate() {
            let val = unsafe { c.to_int_unchecked::<u64>() };
            tot += (val as u128) << (i * 8)
        }
        *elem = tot as u64;
        carry = tot >> 64;
    }
    return carry as u64;
}

#[inline(always)]
fn bit16_length(sz: usize, last: u64) -> usize {
    (sz - 1) * 4 + (last.ilog2() as usize + 1) / 16
}

#[inline(always)]
fn bit8_length(sz: usize, last: u64) -> usize {
    (sz - 1) * 8 + (last.ilog2() as usize + 1) / 8
}

pub fn fft_entry(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let bit_size = bit_sz_dispatch(short.len());
    let (l_len, s_len) = match bit_size {
        BitSize::Bit16 => (
            bit16_length(long.len(), long.last().copied().unwrap()),
            bit16_length(short.len(), short.last().copied().unwrap()),
        ),
        BitSize::Bit8 => (
            bit8_length(long.len(), long.last().copied().unwrap()),
            bit8_length(short.len(), short.last().copied().unwrap()),
        ),
    };
    let n = find_fft_size(l_len + s_len - 1);
    FFT_CACHE.with(|cell| {
        let fft_cache = &mut *cell.borrow_mut();
        let (fwd, bwd, tw, scratch_sz) = fft_cache.prep_mul(n);
        let mut scratch_gaurd = ScratchGuard::acquire();
        let complex_scratch: &mut [Complex<f64>] =
            unsafe { std::mem::transmute(scratch_gaurd.get(scratch_sz)) };
        let (x, fft_scratch) = complex_scratch.split_at_mut(n);

        match bit_size {
            BitSize::Bit16 => decompose_16(long, short, x),
            BitSize::Bit8 => decompose_8(long, short, x),
        }

        fft_core(x, fwd.as_ref(), bwd.as_ref(), tw, fft_scratch);

        match bit_size {
            BitSize::Bit16 => accumulate_16(x, out),
            BitSize::Bit8 => accumulate_8(x, out),
        }
    })
}

//NTT MULTIPLICATION

const NTT_RADIX: [usize; 1] = [2];

trait NTTPrime {
    const P: u64;
    const P_INV: u64;
    const R_SQR: u64;
    const R: u64;
    const G: u64;
    const K: u64;
    const M: u64;
}
// P = K*M + 1 where P is prime
// P_INV*P = -1 mod 2^64
// R_SQR = 2^128 mod P

struct P1;

impl NTTPrime for P1 {
    const P: u64 = 9097271247288401921;
    const P_INV: u64 = 9097271247288401919;
    const R_SQR: u64 = 6171946961806066123;
    const R: u64 = 252201579132747774;
    const G: u64 = 6;
    const K: u64 = 505;
    const M: u64 = 1 << 54;
}

#[repr(transparent)]
#[derive(Debug)]
struct Montgomery<P: NTTPrime> {
    val: u64,
    _phantom: PhantomData<P>,
}

impl<P: NTTPrime> Montgomery<P> {
    const ZERO: Self = Montgomery {
        val: 0,
        _phantom: PhantomData,
    };

    const ONE: Self = Montgomery {
        val: P::R,
        _phantom: PhantomData,
    };

    #[inline(always)]
    fn reduce(t: u128) -> u64 {
        let t_lo = t as u64;
        let m = t_lo.wrapping_mul(P::P_INV);
        let mp = (m as u128) * (P::P as u128);
        let mut val = ((t + mp) >> 64) as u64;
        if val >= P::P {
            val -= P::P;
        }
        return val;
    }

    fn to(a: u64) -> Self {
        let t = (a as u128) * (P::R_SQR as u128);
        Montgomery {
            val: Self::reduce(t),
            _phantom: PhantomData,
        }
    }

    fn to_mut(&mut self) {
        let t = (self.val as u128) * (P::R_SQR as u128);
        let m = (t as u64).wrapping_mul(P::P_INV);
        let mp = (m as u128) * (P::P as u128);
        self.val = ((t + mp) >> 64) as u64;
        if self.val >= P::P {
            self.val -= P::P;
        }
    }

    fn from(self) -> u64 {
        Self::reduce(self.val as u128)
    }

    fn from_mut(&mut self) {
        let m = self.val.wrapping_mul(P::P_INV);
        let mp = (m as u128) * (P::P as u128);
        self.val = ((self.val as u128 + mp) >> 64) as u64;
        if self.val >= P::P {
            self.val -= P::P
        }
    }

    fn pow(self, pow: u64) -> Self {
        if pow == 0 {
            return Self::ZERO;
        } else if pow == 1 {
            return self;
        }
        let mut val = self.pow(pow / 2);
        val *= val;
        if pow & 1 == 1 {
            val *= self;
        }
        return val;
    }
}

impl<P: NTTPrime> Copy for Montgomery<P> {}

impl<P: NTTPrime> Clone for Montgomery<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: NTTPrime> PartialEq for Montgomery<P> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl<P: NTTPrime> Eq for Montgomery<P> {}

impl<P: NTTPrime> Add for Montgomery<P> {
    type Output = Montgomery<P>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut val = self.val + rhs.val;
        if val >= P::P {
            val -= P::P;
        }
        return Montgomery {
            val,
            _phantom: PhantomData,
        };
    }
}

impl<P: NTTPrime> AddAssign for Montgomery<P> {
    fn add_assign(&mut self, rhs: Self) {
        self.val += rhs.val;
        if self.val >= P::P {
            self.val -= P::P;
        }
    }
}

impl<P: NTTPrime> Sub for Montgomery<P> {
    type Output = Montgomery<P>;
    fn sub(self, rhs: Self) -> Self::Output {
        let val = if self.val >= rhs.val {
            self.val - rhs.val
        } else {
            P::P - (rhs.val - self.val)
        };
        Montgomery {
            val,
            _phantom: PhantomData,
        }
    }
}

impl<P: NTTPrime> SubAssign for Montgomery<P> {
    fn sub_assign(&mut self, rhs: Self) {
        if self.val >= rhs.val {
            self.val -= rhs.val;
        } else {
            self.val = P::P - (rhs.val - self.val);
        }
    }
}

impl<P: NTTPrime> Neg for Montgomery<P> {
    type Output = Montgomery<P>;
    fn neg(self) -> Self::Output {
        if self.val == 0 {
            self
        } else {
            Montgomery {
                val: P::P - self.val,
                _phantom: PhantomData,
            }
        }
    }
}

impl<P: NTTPrime> Mul for Montgomery<P> {
    type Output = Montgomery<P>;
    fn mul(self, rhs: Self) -> Self::Output {
        let t = (self.val as u128) * (rhs.val as u128);
        Montgomery {
            val: Self::reduce(t),
            _phantom: PhantomData,
        }
    }
}

impl<P: NTTPrime> MulAssign for Montgomery<P> {
    fn mul_assign(&mut self, rhs: Self) {
        let t = (self.val as u128) * (rhs.val as u128);
        self.val = Self::reduce(t);
    }
}

fn ntt<P: NTTPrime>(buf: &mut [Montgomery<P>], w: Montgomery<P>, scratch: &mut [Montgomery<P>]) {
    if buf.len() <= NTT_CUTOFF {
        dft_naive(buf, w, scratch);
    } else {
        ntt_2(buf, w, scratch);
    }
}

fn dft_naive<P: NTTPrime>(
    buf: &mut [Montgomery<P>],
    w: Montgomery<P>,
    scratch: &mut [Montgomery<P>],
) {
    let n = buf.len();
    let mut t = Montgomery::<P>::ONE;
    for i in 0..n {
        let mut sum = Montgomery::<P>::ZERO;
        let mut l = Montgomery::<P>::ONE;
        for j in 0..n {
            sum += buf[j] * l;
            l *= t;
        }
        scratch[i] = sum;
        t *= w;
    }
    buf.copy_from_slice(&scratch[..n]);
}

fn shuffle_2<P: NTTPrime>(buf: &mut [Montgomery<P>], scratch: &mut [Montgomery<P>]) {
    let m = buf.len() / 2;
    for i in 0..m {
        buf[i] = buf[2 * i];
        scratch[i] = buf[2 * i + 1];
    }
    buf[m..].copy_from_slice(&scratch[..m]);
}

fn ntt_2<P: NTTPrime>(buf: &mut [Montgomery<P>], w: Montgomery<P>, scratch: &mut [Montgomery<P>]) {
    let m = buf.len() / 2;
    shuffle_2(buf, scratch);
    ntt(&mut buf[..m], w * w, scratch);
    ntt(&mut buf[m..], w * w, scratch);
    let mut t = Montgomery::<P>::ONE;
    for i in 0..m {
        let u = buf[i];
        let v = buf[i + m] * t;
        buf[i] = u + v;
        buf[i + m] = u - v;
        t *= w;
    }
}

fn ntt_mul_core<P: NTTPrime>(
    a: &[u64],
    b: &[u64],
    out: &mut [u64],
    n: usize,
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    out[..a.len()].copy_from_slice(a);
    out[a.len()..].fill(0);
    buf_scratch[..b.len()].copy_from_slice(b);
    buf_scratch[b.len()..].fill(0);
    let mont_a: &mut [Montgomery<P>] = unsafe { std::mem::transmute(out) };
    let mont_b: &mut [Montgomery<P>] = unsafe { std::mem::transmute(buf_scratch) };
    for i in 0..n {
        mont_a[i].to_mut();
        mont_b[i].to_mut();
    }
    let mont_scratch: &mut [Montgomery<P>] = unsafe { std::mem::transmute(ntt_scratch) };

    let w = Montgomery::<P>::to(P::G).pow((P::P - 1) / (n as u64));
    ntt(mont_a, w, mont_scratch);
    ntt(mont_b, w, mont_scratch);

    for i in 0..n {
        mont_a[i] *= mont_b[i];
    }

    let inv_w = Montgomery::<P>::to(P::G).pow((n - 1) as u64);
    ntt(mont_a, inv_w, mont_scratch);
    for i in 0..n {
        mont_a[i].from_mut();
    }
}

//END NTT MULTIPLICATION

pub fn fft_cost(l: usize, s: usize) -> f64 {
    let sum = (l + s) as f64;
    sum * sum.ln()
}

pub fn chunking_karatsuba_cost(l: usize, s: usize) -> f64 {
    const LOG_2_3: f64 = 1.584962501;
    let s_pow = (s as f64).powf(LOG_2_3 - 1.0);
    (l as f64) * s_pow
}

pub fn karatsuba_cost(l: usize, s: usize) -> f64 {
    const LOG_2_3: f64 = 1.584962501;
    let l_pow = (l as f64).powf(LOG_2_3);
    let sum_pow = ((l - s) as f64).powf(LOG_2_3);
    l_pow - sum_pow
}

pub fn is_karatsuba(l: usize, s: usize) -> bool {
    let fft_cost = fft_cost(l, s);
    let half = (l + 1) / 2;
    if s <= half {
        chunking_karatsuba_cost(l, s) < FFT_CHUNKING_KARATSUBA_CUTOFF * fft_cost
    } else {
        karatsuba_cost(l, s) < FFT_KARATSUBA_CUTOFF * fft_cost
    }
}

enum DynDispatch {
    Prim,
    Prim2,
    School,
    Karatsuba,
    FFT,
}

fn dyn_dispatch(l: usize, s: usize) -> DynDispatch {
    if s == 1 {
        DynDispatch::Prim
    } else if s == 2 {
        DynDispatch::Prim2
    } else if is_school(l, s) {
        DynDispatch::School
    } else if is_karatsuba(l, s) {
        DynDispatch::Karatsuba
    } else {
        DynDispatch::FFT
    }
}

pub fn mul_dyn(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    debug_assert!(
        out.len() >= a.len() + b.len() - 1,
        "out is not large enough for multiplication"
    );
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    match dyn_dispatch(long.len(), short.len()) {
        DynDispatch::Prim => {
            out[..long.len()].copy_from_slice(long);
            mul_prim(out, short[0])
        }
        DynDispatch::Prim2 => {
            out[..long.len()].copy_from_slice(long);
            mul_prim2(out, combine_u64(short[0], short[1])) as u64
        }
        DynDispatch::School => mul_buf(long, short, out),
        DynDispatch::Karatsuba => karatsuba_entry_dyn(long, short, out),
        DynDispatch::FFT => fft_entry(long, short, out),
    }
}

pub fn mul_vec(a: &[u64], b: &[u64]) -> (Vec<u64>, u64) {
    let mut out = vec![0; a.len() + b.len() - 1];
    let of = mul_dyn(a, b, &mut out);
    (out, of)
}

enum StaticDispatch {
    Prim,
    Prim2,
    School,
    Karatsuba,
}

fn static_dispatch(l: usize, s: usize) -> StaticDispatch {
    if s == 1 {
        StaticDispatch::Prim
    } else if s == 2 {
        StaticDispatch::Prim2
    } else if is_school(l, s) {
        StaticDispatch::School
    } else {
        StaticDispatch::Karatsuba
    }
}

pub fn mul_static<const N: usize>(a: &[u64], b: &[u64], out: &mut [u64]) -> Result<u64, ()> {
    if a.len() + b.len() - 1 > N {
        return Err(());
    }
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    Ok(match static_dispatch(long.len(), short.len()) {
        StaticDispatch::Prim => {
            out[..long.len()].copy_from_slice(long);
            mul_prim(out, short[0])
        }
        StaticDispatch::Prim2 => {
            out[..long.len()].copy_from_slice(long);
            mul_prim2(out, combine_u64(short[0], short[1])) as u64
        }
        StaticDispatch::School => mul_buf(long, short, out),
        StaticDispatch::Karatsuba => karatsuba_entry_static::<N>(long, short, out),
    })
}

pub fn mul_arr<const N: usize>(a: &[u64], b: &[u64]) -> Result<([u64; N], u64), ()> {
    let mut out = [0; N];
    let of = mul_static::<N>(a, b, &mut out)?;
    Ok((out, of))
}

//SQUARE MULTIPLICATION
//simple hierarchy School < Karatsuba < FFT

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

fn karatsuba_sqr_core(
    buf: &[u64],
    half: usize,
    out: &mut [u64],
    cross: &mut [u64],
    scratch: &mut [u64],
) -> u64 {
    let (x0, x1) = buf.split_at(half);
    {
        let sum = &mut out[..=half];
        sum[..half].copy_from_slice(x0);
        let mut sum_len = half;
        if add_buf(&mut sum[..half], x1) {
            sum[half] = 1;
            sum_len += 1;
        }

        karatsuba_sqr_mul(&sum[..sum_len], cross, scratch);
    }
    out.fill(0);
    let (z0, z2) = out.split_at_mut(2 * half);
    karatsuba_sqr_mul(x0, z0, scratch);
    let mut overflow = karatsuba_sqr_mul(x1, z2, scratch);
    let z2_len = buf_len(z2);
    sub_buf(cross, z0);
    sub_buf(cross, &z2[..z2_len]);
    if overflow > 0 {
        sub_buf(&mut cross[z2.len()..], &[overflow]);
    }
    if add_buf(&mut out[half..], &cross) {
        overflow += 1;
    }

    return overflow;
}

fn karatsuba_sqr_mul(buf: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    if buf.len() <= KARATSUBA_SQR_CUTOFF {
        return sqr_buf(buf, out);
    }
    let half = (buf.len() + 1) / 2;
    let (cross, rest) = scratch.split_at_mut(2 * half + 1);
    karatsuba_sqr_core(buf, half, out, cross, rest)
}

pub fn find_karatsuba_sqr_scratch_sz(n: usize) -> usize {
    return if n > KARATSUBA_SQR_CUTOFF {
        let half = (n + 1) / 2;
        2 * half + 1 + find_karatsuba_sqr_scratch_sz(half + 1)
    } else {
        0
    };
}

pub fn karatsuba_sqr_entry_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    let mut scratch_gaurd = ScratchGuard::acquire();
    let scratch_sz = find_karatsuba_sqr_scratch_sz(buf.len());
    let half = (buf.len() + 1) / 2;
    let (cross, rest) = scratch_gaurd.get(scratch_sz).split_at_mut(2 * half + 1);
    karatsuba_sqr_core(buf, half, out, cross, rest)
}

pub fn karatsuba_sqr_entry_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> u64 {
    let scratch_sz = find_karatsuba_sqr_scratch_sz(buf.len());
    let half = (buf.len() + 1) / 2;
    let mut scratch = [0; N];
    if scratch_sz > N {
        let mut cross = [0; N];
        karatsuba_sqr_core(buf, half, out, &mut cross[..2 * half + 1], &mut scratch)
    } else {
        let (cross, rest) = scratch.split_at_mut(2 * half + 1);
        karatsuba_sqr_core(buf, half, out, cross, rest)
    }
}

#[inline(always)]
fn seperate(x: Complex<f64>, y: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
    let a = (x.re + y.re) * 0.5;
    let b = (x.im - y.im) * 0.5;
    let c = (x.im + y.im) * 0.5;
    let d = (x.re - y.re) * 0.5;
    (Complex::new(a, b), Complex::new(c, -d))
}

#[inline(always)]
fn c_sqr(x: Complex<f64>) -> Complex<f64> {
    Complex::new((x.re + x.im) * (x.re - x.im), 2.0 * x.re * x.im)
}

fn sqr_recombine(x: &mut [Complex<f64>], m: usize, tw: &[Complex<f64>]) {
    let Complex::<f64> { re, im } = x[0];
    x[0] = Complex::new(re * re + im * im, 2.0 * re * im);
    for k in 1..m / 2 {
        let (e, o) = seperate(x[k], x[m - k]);
        let (a, b) = seperate(e, tw[k] * o.conj());
        let s = c_sqr(a) - c_sqr(b);
        let t = e * o;
        x[k] = Complex::new(s.re - t.im, s.im + t.re).scale(2.0);
        x[m - k] = Complex::new(s.re + t.im, t.re - s.im).scale(2.0)
    }
    x[m / 2] = c_sqr(x[m / 2]);
}

fn fft_sqr_core(
    x: &mut [Complex<f64>],
    fwd: &dyn Fft<f64>,
    bwd: &dyn Fft<f64>,
    tw: &[Complex<f64>],
    fft_scratch: &mut [Complex<f64>],
) {
    let m = x.len();
    fwd.process_with_scratch(x, fft_scratch);
    sqr_recombine(x, m, tw);
    bwd.process_with_scratch(x, fft_scratch);
    scale_and_round(x, m as f64);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sqr_decompose_16_x86(buf: &[u64], x: &mut [Complex<f64>]) {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;
    for &b in buf {
        // loadl loads 64 bits into the low half of a 128-bit register
        let v = _mm_set1_epi64x(b as i64);
        // pmovzxwd: zero-extend 4 u16s → 4 u32s
        let u32s = _mm_cvtepu16_epi32(v);
        // cvtepi32_pd: 4 i32 → 4 f64
        let f = _mm256_cvtepi32_pd(u32s);
        _mm256_storeu_pd(out.add(idx), f);
        idx += 4;
    }
    let written = buf.len() * 4;
    if written < x.len() * 2 {
        std::ptr::write_bytes(out.add(written), 0, x.len() * 2 - written);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sqr_decompose_16_aarch64(buf: &[u64], x: &mut [Complex<f64>]) {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;
    for &b in buf {
        let (f0, f1) = u64_into_2x2f64s_aarch(b);
        vst1q_f64(out.add(idx), f0);
        vst1q_f64(out.add(idx + 2), f1);
        idx += 4;
    }
    let written = buf.len() * 4;
    if written < x.len() * 2 {
        std::ptr::write_bytes(out.add(written), 0, x.len() * 2 - written);
    }
}

fn sqr_decompose_16_standard(buf: &[u64], x: &mut [Complex<f64>]) {
    let coefs = unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2) };
    const MASK: u64 = u16::MAX as u64;
    let mut idx = 0;
    for &b in buf {
        coefs[idx] = (b & MASK) as f64;
        coefs[idx + 1] = (b >> 16 & MASK) as f64;
        coefs[idx + 2] = (b >> 32 & MASK) as f64;
        coefs[idx + 3] = (b >> 48) as f64;
        idx += 4;
    }
    if idx < x.len() {
        x[idx..].fill(Complex::zero());
    }
}

fn sqr_decompose_16(buf: &[u64], x: &mut [Complex<f64>]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { sqr_decompose_16_x86(buf, x) }
    } else {
        sqr_decompose_16(buf, x);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        sqr_decompose_16_aarch64(buf, x);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sqr_decompose_8_x86(buf: &[u64], x: &mut [Complex<f64>]) {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;
    for &b in buf {
        let v = _mm_set1_epi64x(b as i64);
        // pmovzxbd: zero-extend low 4 bytes → 4 u32s
        let lo4 = _mm_cvtepu8_epi32(v);
        // shift right 4 bytes, then same widening for bytes 4-7
        let hi4 = _mm_cvtepu8_epi32(_mm_bsrli_si128(v, 4));
        _mm256_storeu_pd(out.add(idx), _mm256_cvtepi32_pd(lo4));
        _mm256_storeu_pd(out.add(idx + 4), _mm256_cvtepi32_pd(hi4));
        idx += 8;
    }
    let written = buf.len() * 8;
    if written < x.len() * 2 {
        std::ptr::write_bytes(out.add(written), 0, x.len() * 2 - written);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sqr_decompose_8_aarch64(buf: &[u64], x: &mut [Complex<f64>]) {
    let out = x.as_mut_ptr() as *mut f64;
    let mut idx = 0usize;
    for &b in buf {
        let (f0, f1, f2, f3) = u64_into_2x4f64_aarch(b);
        vst1q_f64(out.add(idx), f0);
        vst1q_f64(out.add(idx + 2), f1);
        vst1q_f64(out.add(idx + 4), f2);
        vst1q_f64(out.add(idx + 6), f3);
        idx += 8;
    }
    let written = buf.len() * 8;
    if written < x.len() * 2 {
        std::ptr::write_bytes(out.add(written), 0, x.len() * 2 - written);
    }
}

fn sqr_decompose_8_standard(buf: &[u64], x: &mut [Complex<f64>]) {
    let coefs = unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f64, x.len() * 2) };
    const MASK: u64 = u8::MAX as u64;
    let mut idx = 0;
    for &b in buf {
        coefs[idx] = (b & MASK) as f64;
        coefs[idx + 1] = (b >> 8 & MASK) as f64;
        coefs[idx + 2] = (b >> 16 & MASK) as f64;
        coefs[idx + 3] = (b >> 24 & MASK) as f64;
        coefs[idx + 4] = (b >> 32 & MASK) as f64;
        coefs[idx + 5] = (b >> 40 & MASK) as f64;
        coefs[idx + 6] = (b >> 48 & MASK) as f64;
        coefs[idx + 7] = (b >> 56 & MASK) as f64;
        idx += 8;
    }
    if idx < x.len() {
        x[idx..].fill(Complex::zero());
    }
}

fn sqr_decompose_8(buf: &[u64], x: &mut [Complex<f64>]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { sqr_decompose_8_x86(buf, x) }
    } else {
        sqr_decompose_8(buf, x);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        sqr_decompose_8_aarch64(buf, x);
    }
}

fn bit_size_sqr_dispatch(n: usize) -> BitSize {
    return if n <= FFT_SQR_BIT_CUTOFF {
        BitSize::Bit16
    } else {
        BitSize::Bit8
    };
}

pub fn fft_sqr_entry(buf: &[u64], out: &mut [u64]) -> u64 {
    let bit_size = bit_size_sqr_dispatch(buf.len());
    let bit_len = match bit_size {
        BitSize::Bit16 => bit16_length(buf.len(), buf.last().copied().unwrap()),
        BitSize::Bit8 => bit8_length(buf.len(), buf.last().copied().unwrap()),
    };
    let n = find_fft_size(2 * bit_len - 1);
    let m = n / 2;
    FFT_CACHE.with(|cell| {
        let fft_cache = &mut *cell.borrow_mut();
        let (fwd, bwd, tw, scratch_sz) = fft_cache.prep_sqr(m);
        let mut scratch_gaurd = ScratchGuard::acquire();
        let complex_scratch: &mut [Complex<f64>] =
            unsafe { std::mem::transmute(scratch_gaurd.get(scratch_sz)) };
        let (x, fft_scratch) = complex_scratch.split_at_mut(n);

        match bit_size {
            BitSize::Bit16 => sqr_decompose_16(buf, x),
            BitSize::Bit8 => sqr_decompose_8(buf, x),
        }
        fft_sqr_core(x, fwd.as_ref(), bwd.as_ref(), tw, fft_scratch);
        match bit_size {
            BitSize::Bit16 => accumulate_16(x, out),
            BitSize::Bit8 => accumulate_8(x, out),
        }
    })
}

enum DynSqrDispatch {
    School,
    Karatsuba,
    FFT,
}

fn dyn_sqr_dispatch(n: usize) -> DynSqrDispatch {
    if n <= KARATSUBA_SQR_CUTOFF {
        DynSqrDispatch::School
    } else if n <= FFT_SQR_CUTOFF {
        DynSqrDispatch::Karatsuba
    } else {
        DynSqrDispatch::FFT
    }
}

pub fn sqr_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    debug_assert!(
        2 * buf.len() - 1 <= out.len(),
        "out is not large enough for multiplication"
    );
    match dyn_sqr_dispatch(buf.len()) {
        DynSqrDispatch::School => sqr_buf(buf, out),
        DynSqrDispatch::Karatsuba => karatsuba_sqr_entry_dyn(buf, out),
        DynSqrDispatch::FFT => fft_sqr_entry(buf, out),
    }
}

pub fn sqr_vec(buf: &[u64]) -> (Vec<u64>, u64) {
    let mut out = vec![0; 2 * buf.len() - 1];
    let of = sqr_dyn(buf, &mut out);
    (out, of)
}

enum StaticSqrDispatch {
    School,
    Karatsubaa,
}

fn static_sqr_dispatch(n: usize) -> StaticSqrDispatch {
    if n <= KARATSUBA_SQR_CUTOFF {
        StaticSqrDispatch::School
    } else {
        StaticSqrDispatch::Karatsubaa
    }
}

pub fn sqr_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> Result<u64, ()> {
    if 2 * buf.len() - 1 > N {
        return Err(());
    }
    Ok(match static_sqr_dispatch(buf.len()) {
        StaticSqrDispatch::School => sqr_buf(buf, out),
        StaticSqrDispatch::Karatsubaa => karatsuba_sqr_entry_static::<N>(buf, out),
    })
}

pub fn sqr_arr<const N: usize>(buf: &[u64]) -> Result<([u64; N], u64), ()> {
    let mut out = [0; N];
    let of = sqr_static::<N>(buf, &mut out)?;
    Ok((out, of))
}

//SHORT MULTIPLICATION
// Gets only the top part of a multiplication, only optimized for small inputs asymptotically same
// computation as a full product other then truncating inputs dynamically.

pub fn short_mul_buf(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let a_len = a.len() - 1;
    let b_len = b.len() - 1;
    let d = (a_len + b_len).saturating_sub(out.len());

    let mut acc0: u64 = 0;
    let mut acc1: u64 = 0;
    let mut acc2: u64 = 0;
    for n in d..d + out.len() {
        let mi = n.saturating_sub(b_len);
        let mf = n.min(a_len);
        unsafe {
            for m in mi..=mf {
                let a_val = *a.get_unchecked(m);
                let b_val = *b.get_unchecked(n - m);
                mul_asm(a_val, b_val, &mut acc0, &mut acc1, &mut acc2);
            }
            *out.get_unchecked_mut(n - d) = acc0;
        }
        acc0 = acc1;
        acc1 = acc2;
        acc2 = 0;
    }

    return acc0;
}

pub fn short_mul_dyn(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.len() + b.len() - 1 < out.len() {
        return mul_dyn(a, b, out);
    }
    if out.len() <= SHORT_MUL_CUTOFF {
        return short_mul_buf(a, b, out);
    }
    let trunc_a = &a[a.len().saturating_sub(out.len())..];
    let trunc_b = &b[b.len().saturating_sub(out.len())..];
    let mut scratch = ScratchGuard::acquire();
    let val = scratch.get(trunc_a.len() + trunc_b.len() - 1);
    let of = mul_dyn(trunc_a, trunc_b, val);
    out.copy_from_slice(&val[val.len() - out.len()..]);
    return of;
}

pub fn short_mul_static<const N: usize>(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    debug_assert!(
        out.len() <= N,
        "out is to large for static parameter N: {N}"
    );
    if a.len() + b.len() - 1 < out.len() {
        return mul_static::<N>(a, b, out).unwrap();
    }
    if out.len() <= SHORT_MUL_CUTOFF {
        return short_mul_buf(a, b, out);
    }
    let trunc_a = &a[a.len().saturating_sub(out.len())..];
    let trunc_b = &b[b.len().saturating_sub(out.len())..];
    let mut scratch = [0; N];
    let val = &mut scratch[..trunc_a.len() + trunc_b.len() - 1];
    let of = mul_dyn(trunc_a, trunc_b, val);
    out.copy_from_slice(&val[val.len() - out.len()..]);
    return of;
}

//MIDDLE MULTIPLICATION
// gets the middle part n limbs of a n x 2n product
// n: [0..n-1] x 2n: [0..2n-1] -> 3n-1: [0..3n-2] -> [n .. 2n-1]

fn mid_mul_buf(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    // long (2n-1) , short (n)
    let n = short.len();
    let mut acc0: u64 = 0;
    let mut acc1: u64 = 0;
    let mut acc2: u64 = 0;
    for i in 0..n {
        unsafe {
            for j in 0..n {
                let a_val = *long.get_unchecked(j + i);
                let b_val = *short.get_unchecked(n - j - 1);
                mul_asm(a_val, b_val, &mut acc0, &mut acc1, &mut acc2);
            }
            *out.get_unchecked_mut(i) = acc0;
        }
        acc0 = acc1;
        acc1 = acc2;
        acc2 = 0;
    }
    return acc0;
}

fn fft_mid_mul(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let sz = short.len();
    let bit_size = bit_sz_dispatch(sz);
    let (l_len, s_len, w) = match bit_size {
        BitSize::Bit16 => (
            bit16_length(long.len(), long.last().copied().unwrap()),
            bit16_length(short.len(), short.last().copied().unwrap()),
            4,
        ),
        BitSize::Bit8 => (
            bit8_length(long.len(), long.last().copied().unwrap()),
            bit8_length(short.len(), short.last().copied().unwrap()),
            8,
        ),
    };
    let n = find_fft_size(l_len + s_len - (sz - 1) * w - 1);
    FFT_CACHE.with(|cell| {
        let fft_cache = &mut *cell.borrow_mut();
        let (fwd, bwd, tw, scratch_sz) = fft_cache.prep_mul(n);
        let mut scratch_gaurd = ScratchGuard::acquire();
        let complex_scratch: &mut [Complex<f64>] =
            unsafe { std::mem::transmute(scratch_gaurd.get(scratch_sz)) };
        let (x, fft_scratch) = complex_scratch.split_at_mut(n);

        match bit_size {
            BitSize::Bit16 => decompose_16(long, short, x),
            BitSize::Bit8 => decompose_8(long, short, x),
        }

        fft_core(x, fwd.as_ref(), bwd.as_ref(), tw, fft_scratch);

        match bit_size {
            BitSize::Bit16 => accumulate_16(&x[(sz - 1) * w..(2 * sz - 1) * w], out),
            BitSize::Bit8 => accumulate_8(&x[(sz - 1) * w..(2 * sz - 1) * w], out),
        }
    })
}

enum DynMidDispatch {
    School,
    Karatsuba,
    FFT,
}

fn dyn_mid_dispatch(n: usize) -> DynMidDispatch {
    return if n < KARATSUBA_MID_CUTOFF {
        DynMidDispatch::School
    } else if n < FFT_MID_CUTOFF {
        DynMidDispatch::Karatsuba
    } else {
        DynMidDispatch::FFT
    };
}

fn mid_mul_dyn(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    debug_assert_eq!(
        long.len(),
        2 * short.len() - 1,
        "incorrect size of long compared to short for middle product"
    );
    debug_assert_eq!(
        out.len(),
        short.len(),
        "size of out must be the same size as short"
    );
    match dyn_mid_dispatch(short.len()) {
        DynMidDispatch::School => mid_mul_buf(long, short, out),
        DynMidDispatch::Karatsuba => {
            let mut scratch_guard = ScratchGuard::acquire();
            let n = short.len();
            let [big_out, scratch] =
                scratch_guard.get_splits([3 * n - 2, find_karatsuba_scratch(2 * n - 1, n)]);
            let of = karatsuba_mul(long, short, big_out, scratch);
            out.copy_from_slice(&big_out[n - 1..2 * n - 1]);
            of
        }
        DynMidDispatch::FFT => fft_mid_mul(long, short, out),
    }
}

fn mid_mul_static<const N: usize>(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    debug_assert_eq!(
        long.len(),
        2 * short.len() - 1,
        "incorrect size of long compared to short for middle product"
    );
    debug_assert_eq!(
        out.len(),
        short.len(),
        "size of out must be the same size as short"
    );
    let n = short.len();
    if n < KARATSUBA_MID_CUTOFF {
        mid_mul_buf(long, short, out)
    } else {
        let mut big_out = [0; N];
        let of = karatsuba_entry_static::<N>(long, short, &mut big_out);
        out.copy_from_slice(&big_out[n - 1..2 * n - 1]);
        of
    }
}

//FAST EXPONENTIATION
// does an integer powers via log(n) squaring s
fn reverse_pow(pow: usize) -> usize {
    debug_assert!(pow < (1 << 63), "Bro this was a bad idea anyway");
    let marked_pow = (pow << 1) + 1;
    (marked_pow << marked_pow.leading_zeros()).reverse_bits()
}

pub fn powi_sz(buf: &[u64], pow: usize) -> (usize, usize) {
    let bit_length = (buf.len() - 1) * 64 + buf.last().copied().unwrap().ilog2() as usize;
    (
        (bit_length * pow + 63) / 64,
        ((bit_length + 1) * pow + 63) / 64,
    )
}

fn powi_dyn_core(buf: &[u64], reverse_pow: usize, src: &mut [u64], dst: &mut [u64]) {
    if reverse_pow == 1 {
        return;
    }
    let src_len = buf_len(src);
    sqr_dyn(&src[..src_len], dst);
    if reverse_pow & 1 == 1 {
        let dst_len = buf_len(dst);
        mul_dyn(&dst[..dst_len], buf, src);
        powi_dyn_core(buf, reverse_pow >> 1, src, dst);
        return;
    }
    powi_dyn_core(buf, reverse_pow >> 1, dst, src);
}

pub fn powi_dyn_entry(buf: &[u64], pow: usize, out: &mut [u64]) {
    if pow == 0 {
        out[0] = 1;
        return;
    }
    let reverse_pow = reverse_pow(pow);
    let mut scratch = ScratchGuard::acquire();
    let tmp = scratch.get(out.len());
    tmp.fill(0);
    if (pow.ilog2() + pow.count_ones()) % 2 == 0 {
        out[..buf.len()].copy_from_slice(buf);
        powi_dyn_core(buf, reverse_pow, out, tmp);
    } else {
        tmp[..buf.len()].copy_from_slice(buf);
        powi_dyn_core(buf, reverse_pow, tmp, out);
    }
}

pub fn powi_vec(buf: &[u64], pow: usize) -> Vec<u64> {
    let (_, max) = powi_sz(buf, pow);
    let mut out = vec![0; max];
    powi_dyn_entry(buf, pow, &mut out);
    trim_lz(&mut out);
    out
}

fn powi_static_core<const N: usize>(
    buf: &[u64],
    reverse_pow: usize,
    src: &mut [u64],
    dst: &mut [u64],
) -> Result<(), ()> {
    if reverse_pow == 1 {
        return Ok(());
    }
    let src_len = buf_len(src);
    sqr_static::<N>(&src[..src_len], dst)?;
    if reverse_pow & 1 == 1 {
        let dst_len = buf_len(dst);
        mul_static::<N>(&dst[..dst_len], buf, src)?;
        return powi_static_core::<N>(buf, reverse_pow >> 1, src, dst);
    }
    powi_static_core::<N>(buf, reverse_pow >> 1, dst, src)
}

pub fn powi_static_entry<const N: usize>(
    buf: &[u64],
    pow: usize,
    out: &mut [u64],
) -> Result<(), ()> {
    if pow == 0 {
        out[0] = 1;
        return Ok(());
    }
    let reverse_pow = reverse_pow(pow);
    let mut tmp = [0; N];
    if (pow.ilog2() + pow.count_ones()) % 2 == 0 {
        out[..buf.len()].copy_from_slice(buf);
        powi_static_core::<N>(buf, reverse_pow, out, &mut tmp)
    } else {
        tmp[..buf.len()].copy_from_slice(buf);
        powi_static_core::<N>(buf, reverse_pow, &mut tmp, out)
    }
}

pub fn powi_arr<const N: usize>(buf: &[u64], pow: usize) -> Result<[u64; N], ()> {
    let (min, _) = powi_sz(buf, pow);
    if N < min {
        return Err(());
    }
    let mut out = [0; N];
    powi_static_entry::<N>(buf, pow, &mut out)?;
    Ok(out)
}
