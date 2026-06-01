#![allow(dead_code)]
use crate::utils::{utils::*, *};
use core::task;
use rayon::*;
use rustfft::num_traits::{One, Zero};
use rustfft::{num_complex::Complex, Fft, FftDirection, FftPlanner};
use std::arch::asm;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::ffi::c_short;
use std::marker::PhantomData;
use std::ops::*;
use std::sync::{Arc, LazyLock};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

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

#[inline(always)]
pub fn mul_elem(
    a: &[u64],
    b: &[u64],
    n: usize,
    acc0: &mut u64,
    acc1: &mut u64,
    acc2: &mut u64,
) -> u64 {
    let mi = n.saturating_sub(b.len() - 1);
    let mf = n.min(a.len() - 1);
    unsafe {
        for m in mi..=mf {
            let a_val = *a.get_unchecked(m);
            let b_val = *b.get_unchecked(n - m);
            mul_asm(a_val, b_val, acc0, acc1, acc2);
        }
    }
    let out = *acc0;
    *acc0 = *acc1;
    *acc1 = *acc2;
    *acc2 = 0;
    return out;
}

pub fn mul_buf(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    let (mut acc0, mut acc1, mut acc2) = (0, 0, 0);
    for n in 0..out.len() {
        out[n] = mul_elem(a, b, n, &mut acc0, &mut acc1, &mut acc2);
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
    let chunks = (l - s + 1) / s;
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
        let c = KARATSUBA_CUTOFF;
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

pub fn find_karatsuba_scratch(l_in: usize, s_in: usize) -> usize {
    let (l, s) = if l_in >= s_in {
        (l_in, s_in)
    } else {
        (s_in, l_in)
    };

    if s <= 2 || is_school(l, s) {
        return 0;
    }

    if s <= (l + 1) / 2 {
        let half = (s + 1) / 2;
        let phase1 = 2 * s + 2 * half + 1 + find_karatsuba_scratch(half + 1, half + 1);
        let chunks = (l - s + 1) / s;
        let last_len = l - chunks * s;
        let phase2 = (last_len + s - 1) + find_karatsuba_scratch(last_len, s);

        phase1.max(phase2)
    } else {
        let half = (l + 1) / 2;
        let inner = find_karatsuba_scratch(half + 1, half + 1)
            .max(find_karatsuba_scratch(l - half, s - half));
        2 * half + 1 + inner
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

pub struct FTTTwidleTower {
    table: Vec<Complex<f64>>,
    veiw: Vec<Complex<f64>>,
    res: usize,
    gen: u8,
    base: usize,
}

impl FTTTwidleTower {
    const MAX_GEN: u8 = 3;

    pub fn build(n: usize) -> Self {
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

    pub fn refine(&mut self, new_res: usize) {
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

    pub fn rebuild(&mut self, new_res: usize) {
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

    pub fn get(&mut self, res: usize) -> &[Complex<f64>] {
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
            .or_insert_with(|| FTTTwidleTower::build(2 * m))
            .get(2 * m);
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

const fn fft_log_prim_search(
    init: usize,
    target: usize,
    mut idx: usize,
    primes: &[usize],
) -> usize {
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
    let mut best = fft_log_prim_search(val, target, idx, primes);
    if best == target {
        return best;
    }
    val *= p;
    while val < best {
        let cur = fft_log_prim_search(val, target, idx, primes);
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

pub const fn find_fft_size(n: usize) -> usize {
    return fft_log_prim_search(4, n, FFT_RADIX.len(), &FFT_RADIX);
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

pub fn scale_and_round_standard(x: &mut [Complex<f64>], scale: f64) {
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
    scale_and_round(&mut x[..n / 2], n as f64);
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

pub fn decompose_16_standard(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
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

fn decompose(long: &[u64], short: &[u64], x: &mut [Complex<f64>]) {
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

fn fft_accumulate(x: &[Complex<f64>], out: &mut [u64]) -> u64 {
    let coefs = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len() * 2) };
    let mut chunks = coefs.chunks(4);
    let mut carry: u128 = 0;

    for elem in out.iter_mut() {
        let chunk = chunks.next().unwrap();
        let mut tot = carry;
        for (i, c) in chunk.iter().enumerate() {
            let val = unsafe { c.to_int_unchecked::<u64>() } as u128;
            tot += val << (i * 16);
        }
        *elem = tot as u64;
        carry = tot >> 64;
    }

    // Drain any remaining chunk that didn't have a corresponding output limb
    for chunk in chunks {
        for (i, c) in chunk.iter().enumerate() {
            let val = unsafe { c.to_int_unchecked::<u64>() } as u128;
            carry += val << (i * 16);
        }
    }

    carry as u64
}

#[inline(always)]
fn bit16_length(sz: usize, last: u64) -> usize {
    (sz - 1) * 4 + (last.ilog2() as usize) / 16 + 1
}

pub fn fft_entry(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let l_len = buf_len(&long);
    let s_len = buf_len(&short);
    let (l_len16, s_len16) = (
        bit16_length(l_len, long[l_len - 1]),
        bit16_length(s_len, short[s_len - 1]),
    );
    let n = find_fft_size(l_len16 + s_len16 - 1);
    FFT_CACHE.with(|cell| {
        let fft_cache = &mut *cell.borrow_mut();
        let (fwd, bwd, tw, scratch_sz) = fft_cache.prep_mul(n);
        let mut scratch_gaurd = ScratchGuard::acquire();
        let complex_scratch: &mut [Complex<f64>] =
            unsafe { std::mem::transmute(scratch_gaurd.get(scratch_sz)) };
        let (x, fft_scratch) = complex_scratch.split_at_mut(n);

        decompose(long, short, x);
        fft_core(x, fwd.as_ref(), bwd.as_ref(), tw, fft_scratch);
        fft_accumulate(&mut x[..n / 2], out)
    })
}

//NTT MULTIPLICATION

const NTT_RADIX_CNT: usize = 3;

const NTT_RADIX: [usize; NTT_RADIX_CNT] = [2, 3, 5];

pub(crate) const fn exp_cnt(mut p: u64) -> [usize; NTT_RADIX_CNT] {
    let mut exp = [0; NTT_RADIX_CNT];
    let mut idx = 0;
    while idx < NTT_RADIX_CNT {
        let q = NTT_RADIX[idx] as u64;
        while p % q == 0 {
            exp[idx] += 1;
            p /= q;
        }
        idx += 1;
    }
    return exp;
}

pub(crate) const fn neg_inv_mod_2_64(p: u64) -> u64 {
    let mut x: u64 = 1;
    while x.wrapping_mul(p) != u64::MAX {
        let mut tmp = x.wrapping_mul(p);
        tmp = tmp.wrapping_add(2);
        x = x.wrapping_mul(tmp);
    }
    return x;
}

pub(crate) const fn split_mul(a: u64, b: u64) -> (u64, u64) {
    const MASK: u64 = (1 << 32) - 1;
    let a0 = a & MASK;
    let a1 = a >> 32;
    let b0 = b & MASK;
    let b1 = b >> 32;
    let mut r11 = a1 * b1;
    let mut r00 = a0 * b0;
    let r10 = a1 * b0;
    let r01 = a0 * b1;
    let (mid_sum, c0) = r10.overflowing_add(r01);
    r11 += mid_sum >> 32;
    if c0 {
        r11 += 1 << 32;
    }
    let (lo_sum, c1) = r00.overflowing_add(mid_sum << 32);
    r00 = lo_sum;
    if c1 {
        r11 += 1;
    }
    (r11, r00)
}

pub(crate) const fn mod_mul(a: u64, b: u64, p: u64) -> u64 {
    const MASK: u64 = (1 << 32) - 1;
    let d = p << 1;
    let (mut hi, mut lo) = split_mul(a, b);
    hi = (hi << 1) | (lo >> 63);
    lo <<= 1;
    let n1 = lo >> 32;
    let n0 = lo & MASK;
    let d1 = d >> 32;
    let d0 = d & MASK;

    let mut q = hi / d1;
    let mut r = hi % d1;
    let mut c0 = q * d0;
    let mut c1 = (r << 32) + n1;
    if c0 > c1 {
        q -= 1;
        if d < (c0 - c1) {
            q -= 1;
        }
    }
    q &= MASK;

    let rem = (hi << 32).wrapping_add(n1).wrapping_sub(q.wrapping_mul(d));
    q = rem / d1;
    r = rem % d1;
    c0 = q * d0;
    c1 = (r << 32) + n0;
    if c0 > c1 {
        q -= 1;
        if d < (c0 - c1) {
            q -= 1;
        }
    }
    q &= MASK;

    (rem << 32).wrapping_add(n0).wrapping_sub(q.wrapping_mul(d)) >> 1
}

pub(crate) trait NTTPrime: Send + Sync {
    const P: u64;
    const EXP: [usize; NTT_RADIX_CNT] = exp_cnt(Self::P - 1);
    const R: u64 = (u64::MAX % Self::P) + 1;
    const P_INV: u64 = neg_inv_mod_2_64(Self::P);
    const R_SQR: u64 = mod_mul(Self::R, Self::R, Self::P);
    const R_CUB: u64 = mod_mul(Self::R, Self::R_SQR, Self::P);
    const P_LO: u64 = Self::P & 0xFFFFFFFF;
    const P_HI: u64 = Self::P >> 32;
}

// MIN_EXP = [46, 3, 2]
pub(crate) struct P1;
impl NTTPrime for P1 {
    const P: u64 = 5937362789990400001;
    // 2^46 * 3^3 * 5^5 + 1
}

pub(crate) struct P2;
impl NTTPrime for P2 {
    const P: u64 = 8122312296706867201;
    // 19 * 2^46 * 3^5 * 5^2 + 1
}

pub(crate) struct P3;
impl NTTPrime for P3 {
    const P: u64 = 7552325468867788801;
    // 53 * 2^46 * 3^4 * 5^2 + 1
}

#[repr(transparent)]
#[derive(Debug)]
pub(crate) struct Montgomery<P: NTTPrime> {
    val: u64,
    _phantom: PhantomData<P>,
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn reduce_x86(t: u128, p: u64, p_inv: u64) -> u64 {
    let t_lo = t as u64;
    let t_hi = (t >> 64) as u64;

    let val: u64;

    core::arch::asm!(
        // m = t_lo * p_inv mod 2^64
        "mov {m}, {t_lo}",
        "imul {m}, {p_inv}",

        // rdx:rax = m * p
        "mov rax, {m}",
        "mul {p}",

        // rdx = high(m*p) + t_hi + carry(low(m*p) + t_lo)
        "add rax, {t_lo}",
        "adc rdx, {t_hi}",

        // conditional subtract p
        "mov {tmp}, rdx",
        "sub {tmp}, {p}",
        "cmovnc rdx, {tmp}",

        m = out(reg) _,
        tmp = out(reg) _,

        t_lo = in(reg) t_lo,
        t_hi = in(reg) t_hi,
        p = in(reg) p,
        p_inv = in(reg) p_inv,

        out("rax") _,
        out("rdx") val,

        options(nostack, nomem, pure),
    );

    val
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn reduce_aarch(t: u128, p: u64, p_inv: u64) -> u64 {
    let t_lo = t as u64;
    let t_hi = (t >> 64) as u64;

    let val: u64;

    core::arch::asm!(
        // m = t_lo * p_inv mod 2^64
        "mul {m}, {t_lo}, {p_inv}",

        // mp_lo = low64(m * p)
        // mp_hi = high64(m * p)
        "mul {mp_lo}, {m}, {p}",
        "umulh {mp_hi}, {m}, {p}",

        // val = t_hi + mp_hi + carry(t_lo + mp_lo)
        "adds {mp_lo}, {mp_lo}, {t_lo}",
        "adc {val}, {mp_hi}, {t_hi}",

        // conditional subtract p
        "subs {tmp}, {val}, {p}",
        "csel {val}, {tmp}, {val}, hs",

        m = out(reg) _,
        mp_lo = out(reg) _,
        mp_hi = out(reg) _,
        tmp = out(reg) _,

        val = out(reg) val,

        t_lo = in(reg) t_lo,
        t_hi = in(reg) t_hi,
        p = in(reg) p,
        p_inv = in(reg) p_inv,

        options(nostack, nomem, pure),
    );

    val
}

#[inline(always)]
pub fn reduce_no_asm(t: u128, p: u64, p_inv: u64) -> u64 {
    let t_lo = t as u64;
    let m = t_lo.wrapping_mul(p_inv);
    let mp = (m as u128) * (p as u128);

    let mut val = ((t + mp) >> 64) as u64;

    if val >= p {
        val -= p;
    }

    val
}

#[inline]
fn reduce_mont(t: u128, p: u64, p_inv: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return reduce_x86(t, p, p_inv);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return reduce_aarch(t, p, p_inv);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return reduce_no_asm(t, p, p_inv);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_reduce_x86(a: u64, b: u64, p: u64, p_inv: u64) -> u64 {
    let val: u64;

    core::arch::asm!(
        // rdx:rax = a * b
        "mul {b}",

        // Save t_lo and t_hi
        "mov {t_lo}, rax",
        "mov {t_hi}, rdx",

        // m = t_lo * p_inv mod 2^64
        "mov {m}, rax",
        "imul {m}, {p_inv}",

        // rdx:rax = m * p
        "mov rax, {m}",
        "mul {p}",

        // val = t_hi + high(m*p) + carry(t_lo + low(m*p))
        "add rax, {t_lo}",
        "adc rdx, {t_hi}",

        // Conditional subtract p
        "mov {tmp}, rdx",
        "sub {tmp}, {p}",
        "cmovnc rdx, {tmp}",

        t_lo = out(reg) _,
        t_hi = out(reg) _,
        m = out(reg) _,
        tmp = out(reg) _,

        inout("rax") a => _,
        b = in(reg) b,
        p = in(reg) p,
        p_inv = in(reg) p_inv,

        out("rdx") val,

        options(nostack, nomem, pure),
    );

    val
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_reduce_aarch(a: u64, b: u64, p: u64, p_inv: u64) -> u64 {
    let val: u64;

    core::arch::asm!(
        // t = a * b
        "mul {t_lo}, {a}, {b}",
        "umulh {t_hi}, {a}, {b}",

        // m = t_lo * p_inv mod 2^64
        "mul {m}, {t_lo}, {p_inv}",

        // mp = m * p
        "mul {mp_lo}, {m}, {p}",
        "umulh {mp_hi}, {m}, {p}",

        // val = t_hi + mp_hi + carry(t_lo + mp_lo)
        "adds {mp_lo}, {mp_lo}, {t_lo}",
        "adc {val}, {mp_hi}, {t_hi}",

        // if val >= p, val -= p
        "subs {tmp}, {val}, {p}",
        "csel {val}, {tmp}, {val}, hs",

        t_lo = out(reg) _,
        t_hi = out(reg) _,
        m = out(reg) _,
        mp_lo = out(reg) _,
        mp_hi = out(reg) _,
        tmp = out(reg) _,

        val = out(reg) val,

        a = in(reg) a,
        b = in(reg) b,
        p = in(reg) p,
        p_inv = in(reg) p_inv,

        options(nostack, nomem, pure),
    );

    val
}

#[inline(always)]
pub fn mul_reduce_no_asm(a: u64, b: u64, p: u64, p_inv: u64) -> u64 {
    let t = (a as u128) * (b as u128);

    let t_lo = t as u64;
    let m = t_lo.wrapping_mul(p_inv);
    let mp = (m as u128) * (p as u128);

    let mut val = ((t + mp) >> 64) as u64;

    if val >= p {
        val -= p;
    }

    val
}

#[inline]
fn mul_reduce_mont(a: u64, b: u64, p: u64, p_inv: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return mul_reduce_x86(a, b, p, p_inv);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return mul_reduce_aarch(a, b, p, p_inv);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return mul_reduce_no_asm(a, b, p, p_inv);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn add_mod_x86(mut a: u64, b: u64, p: u64) -> u64 {
    core::arch::asm!(
        "add {a}, {b}",
        "mov {tmp}, {a}",
        "sub {tmp}, {p}",
        "cmovnc {a}, {tmp}",
        a = inout(reg) a,
        b = in(reg) b,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn add_mod_aarch(mut a: u64, b: u64, p: u64) -> u64 {
    core::arch::asm!(
        "add {a}, {a}, {b}",
        "subs {tmp}, {a}, {p}",
        "csel {a}, {tmp}, {a}, hs",
        a = inout(reg) a,
        b = in(reg) b,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[inline(always)]
pub fn add_mod_no_asm(mut a: u64, b: u64, p: u64) -> u64 {
    a += b;
    if a >= p {
        a -= p;
    }
    a
}

#[inline]
fn add_mod(a: u64, b: u64, p: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        add_mod_x86(a, b, p)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        add_mod_aarch(a, b, p)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        add_mod_no_asm(a, b, p)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sub_mod_x86(mut a: u64, b: u64, p: u64) -> u64 {
    core::arch::asm!(
        "sub {a}, {b}",
        "sbb {tmp}, {tmp}",
        "and {tmp}, {p}",
        "add {a}, {tmp}",
        a = inout(reg) a,
        b = in(reg) b,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sub_mod_aarch(mut a: u64, b: u64, p: u64) -> u64 {
    core::arch::asm!(
        "subs {a}, {a}, {b}",
        "csel {tmp}, xzr, {p}, hs",
        "add {a}, {a}, {tmp}",
        a = inout(reg) a,
        b = in(reg) b,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[inline(always)]
pub fn sub_mod_no_asm(a: u64, b: u64, p: u64) -> u64 {
    let diff = a.wrapping_sub(b);
    let mask = ((a < b) as u64).wrapping_neg();
    diff.wrapping_add(p & mask)
}

#[inline(always)]
fn sub_mod(a: u64, b: u64, p: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return sub_mod_x86(a, b, p);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return sub_mod_aarch(a, b, p);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return sub_mod_no_asm(a, b, p);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn neg_mod_x86(mut a: u64, p: u64) -> u64 {
    core::arch::asm!(
        "test {a}, {a}",
        "mov {tmp}, {p}",
        "sub {tmp}, {a}",
        "cmovnz {a}, {tmp}",
        a = inout(reg) a,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neg_mod_aarch(mut a: u64, p: u64) -> u64 {
    core::arch::asm!(
        "cmp {a}, #0",
        "sub {tmp}, {p}, {a}",
        "csel {a}, xzr, {tmp}, eq",
        a = inout(reg) a,
        p = in(reg) p,
        tmp = out(reg) _,
        options(nostack, nomem, pure),
    );
    a
}

#[inline(always)]
pub fn neg_mod_no_asm(a: u64, p: u64) -> u64 {
    let mask = ((a != 0) as u64).wrapping_neg();
    p.wrapping_sub(a) & mask
}

#[inline]
fn neg_mod(a: u64, p: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        return neg_mod_x86(a, p);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return neg_mod_aarch(a, p);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return neg_mod_no_asm(a, p);
    }
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

    const HALF: Self = Self::const_to((P::P + 1) / 2);

    const G: Self = Self::prim_root();

    const G3: Self = Self::G.const_pow((P::P - 1) / 3);

    const INV_G3: Self = Self::G3.const_pow(2);

    const G5: Self = Self::G.const_pow((P::P - 1) / 5);

    const INV_G5: Self = Self::G5.const_pow(4);

    #[inline(always)]
    const fn const_reduce(t_hi: u64, t_lo: u64) -> u64 {
        const MASK32: u64 = (1 << 32) - 1;
        let m = t_lo.wrapping_mul(P::P_INV);
        let (mp_hi, mp_lo) = split_mul(m, P::P);
        let (_, c) = mp_lo.overflowing_add(t_lo);
        let mut val = t_hi + mp_hi + c as u64;
        if val >= P::P {
            val -= P::P;
        }
        val
    }

    const fn const_to(a: u64) -> Self {
        let (t_hi, t_lo) = split_mul(a, P::R_SQR);
        Montgomery {
            val: Self::const_reduce(t_hi, t_lo),
            _phantom: PhantomData,
        }
    }

    const fn mul_mut(&mut self, rhs: Self) {
        let (t_hi, t_lo) = split_mul(self.val, rhs.val);
        self.val = Self::const_reduce(t_hi, t_lo);
    }

    const fn const_pow(self, pow: u64) -> Self {
        if pow == 0 {
            return Self::ONE;
        } else if pow == 1 {
            return self;
        }
        let mut val = self.const_pow(pow / 2);
        val.mul_mut(val);
        if pow & 1 == 1 {
            val.mul_mut(self);
        }
        return val;
    }

    const fn add_mut(&mut self, rhs: Self) {
        self.val += rhs.val;
        if self.val >= P::P {
            self.val -= P::P;
        }
    }

    const fn prim_root() -> Self {
        let mut g = Montgomery::ONE;
        let mut found = false;
        while !found {
            g.add_mut(Self::ONE);
            found = true;
            let mut idx = 0;
            while idx < NTT_RADIX_CNT && found {
                let q = NTT_RADIX[idx] as u64;
                found &= g.const_pow((P::P - 1) / q).val != P::R;
                idx += 1;
            }
        }
        return g;
    }

    const fn cast(x: u64) -> Self {
        Montgomery {
            val: x,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn to(a: u64) -> Self {
        Montgomery {
            val: mul_reduce_mont(a, P::R_SQR, P::P, P::P_INV),
            _phantom: PhantomData,
        }
    }

    fn to_mut(&mut self) {
        self.val = mul_reduce_mont(self.val, P::R_SQR, P::P, P::P_INV);
    }

    pub(crate) fn from(self) -> u64 {
        reduce_mont(self.val as u128, P::P, P::P_INV)
    }

    pub(crate) fn fuse_mul_to(a: u64, b: u64) -> Self {
        let c = mul_reduce_mont(a, b, P::P, P::P_INV);
        Montgomery {
            val: mul_reduce_mont(c, P::R_CUB, P::P, P::P_INV),
            _phantom: PhantomData,
        }
    }

    pub(crate) fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut result = Self::ONE;
        while exp > 0 {
            if exp & 1 == 1 {
                result *= base;
            }
            base *= base;
            exp >>= 1;
        }
        result
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

impl<P: NTTPrime> AddAssign for Montgomery<P> {
    fn add_assign(&mut self, rhs: Self) {
        self.val = add_mod(self.val, rhs.val, P::P);
    }
}

impl<P: NTTPrime> Add for Montgomery<P> {
    type Output = Montgomery<P>;
    fn add(self, rhs: Self) -> Self::Output {
        Montgomery {
            val: add_mod(self.val, rhs.val, P::P),
            _phantom: PhantomData,
        }
    }
}

impl<P: NTTPrime> SubAssign for Montgomery<P> {
    fn sub_assign(&mut self, rhs: Self) {
        self.val = sub_mod(self.val, rhs.val, P::P);
    }
}

impl<P: NTTPrime> Sub for Montgomery<P> {
    type Output = Montgomery<P>;
    fn sub(self, rhs: Self) -> Self::Output {
        Montgomery {
            val: sub_mod(self.val, rhs.val, P::P),
            _phantom: PhantomData,
        }
    }
}

impl<P: NTTPrime> Neg for Montgomery<P> {
    type Output = Montgomery<P>;
    fn neg(self) -> Self::Output {
        Montgomery {
            val: neg_mod(self.val, P::P),
            _phantom: PhantomData,
        }
    }
}

impl<P: NTTPrime> Mul for Montgomery<P> {
    type Output = Montgomery<P>;
    fn mul(self, rhs: Self) -> Self::Output {
        Montgomery {
            val: mul_reduce_mont(self.val, rhs.val, P::P, P::P_INV),
            _phantom: PhantomData,
        }
    }
}

impl<P: NTTPrime> MulAssign for Montgomery<P> {
    fn mul_assign(&mut self, rhs: Self) {
        self.val = mul_reduce_mont(self.val, rhs.val, P::P, P::P_INV);
    }
}

struct DynNTTTwidles<P: NTTPrime> {
    tw: Vec<Montgomery<P>>,
    max_n: usize,
    n: usize,
}

impl<P: NTTPrime> DynNTTTwidles<P> {
    fn build(n: usize) -> Self {
        let w = Montgomery::G.pow((P::P - 1) / (n as u64));
        let mut tw = Vec::<Montgomery<P>>::with_capacity(n);
        let mut acc = Montgomery::ONE;
        for _ in 0..n {
            tw.push(acc);
            acc *= w;
        }
        DynNTTTwidles { tw, max_n: n, n: n }
    }

    fn ensure(&mut self, new_n: usize) {
        if new_n > self.max_n {
            debug_assert!(new_n % self.max_n == 0);
            let growth = new_n / self.max_n;
            debug_assert!(growth.is_power_of_two());
            let mut new_tw = Vec::<Montgomery<P>>::with_capacity(new_n);
            let w = Montgomery::G.pow((P::P - 1) / (new_n as u64));
            for &tw in &self.tw {
                let mut acc = tw;
                new_tw.push(acc);
                for _ in 1..growth {
                    acc *= w;
                    new_tw.push(acc);
                }
            }
            self.tw = new_tw;
            self.max_n = new_n;
        }
        self.n = new_n;
    }
}

struct StaticNTTTwidles<const N: usize, P: NTTPrime> {
    tw: [Montgomery<P>; N],
    n: usize,
}

impl<P: NTTPrime, const N: usize> StaticNTTTwidles<N, P> {
    fn build(n: usize) -> Self {
        let w = Montgomery::G.pow((P::P - 1) / (n as u64));
        let mut tw = [Montgomery::ZERO; N];
        let mut acc = Montgomery::ONE;
        for i in 0..n {
            tw[i] = acc;
            acc *= w;
        }
        Self { tw, n }
    }
}

trait NTTDir {
    const INV: bool;
    fn advance(idx: usize, step: usize, len: usize) -> usize;
}

struct NTTForward {}

impl NTTDir for NTTForward {
    const INV: bool = false;
    fn advance(idx: usize, step: usize, len: usize) -> usize {
        add_mod(idx as u64, step as u64, len as u64) as usize
    }
}

struct NTTBackward {}

impl NTTDir for NTTBackward {
    const INV: bool = true;
    fn advance(idx: usize, step: usize, len: usize) -> usize {
        sub_mod(idx as u64, step as u64, len as u64) as usize
    }
}

struct TwiddleIter<'a, P: NTTPrime, D: NTTDir> {
    tw: &'a [Montgomery<P>],
    idx: usize,
    step: usize,
    len: usize,
    _dir: PhantomData<D>,
}

impl<'a, P: NTTPrime, D: NTTDir> Iterator for TwiddleIter<'a, P, D> {
    type Item = Montgomery<P>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        debug_assert!(self.idx < self.tw.len());
        debug_assert!(self.len <= self.tw.len());
        debug_assert!(self.step <= self.len);
        let x = unsafe { *self.tw.get_unchecked(self.idx) };
        self.idx = D::advance(self.idx, self.step, self.len);
        Some(x)
    }
}

trait NTTTwidles<P: NTTPrime>: Send + Sync {
    fn twidles_iter<'a, const R: usize, D: NTTDir>(
        &'a self,
        stride: usize,
    ) -> [TwiddleIter<'a, P, D>; R];
}

impl<const N: usize, P: NTTPrime> NTTTwidles<P> for StaticNTTTwidles<N, P> {
    fn twidles_iter<'a, const R: usize, D: NTTDir>(
        &'a self,
        stride: usize,
    ) -> [TwiddleIter<'a, P, D>; R] {
        std::array::from_fn(|r| TwiddleIter {
            tw: &self.tw,
            idx: 0,
            step: (r + 1) * stride,
            len: self.n,
            _dir: PhantomData,
        })
    }
}

impl<P: NTTPrime> NTTTwidles<P> for DynNTTTwidles<P> {
    fn twidles_iter<'a, const R: usize, D: NTTDir>(
        &'a self,
        stride: usize,
    ) -> [TwiddleIter<'a, P, D>; R] {
        let cache_stride = self.max_n / self.n;
        debug_assert_eq!(self.tw.len(), self.max_n);
        debug_assert_eq!(self.max_n % self.n, 0);
        std::array::from_fn(|r| {
            let step = (r + 1) * stride * cache_stride;
            debug_assert!(step <= self.max_n);
            TwiddleIter {
                tw: &self.tw,
                idx: 0,
                step,
                len: self.max_n,
                _dir: PhantomData,
            }
        })
    }
}

thread_local! {
    static DYN_NTT_CACHE_P1: RefCell<HashMap<usize, DynNTTTwidles<P1>>> = RefCell::new(HashMap::new());
    static DYN_NTT_CACHE_P2: RefCell<HashMap<usize, DynNTTTwidles<P2>>> = RefCell::new(HashMap::new());
    static DYN_NTT_CACHE_P3: RefCell<HashMap<usize, DynNTTTwidles<P3>>> = RefCell::new(HashMap::new());
}

fn ntt<P: NTTPrime, D: NTTDir>(
    buf: &mut [Montgomery<P>],
    stride: usize,
    tw: &impl NTTTwidles<P>,
    scratch: &mut [Montgomery<P>],
) {
    debug_assert!(scratch.len() >= ntt_scratch_len(buf.len()));
    if buf.len() % 5 == 0 {
        ntt_5::<P, D>(buf, stride, tw, scratch);
    } else if buf.len() % 3 == 0 {
        ntt_3::<P, D>(buf, stride, tw, scratch);
    } else {
        ntt_2::<P, D>(buf, stride, tw, scratch);
    }
}

fn ntt_scratch_len(n: usize) -> usize {
    if n < 2 {
        return 0;
    }

    if n % 5 == 0 {
        ntt_radix_scratch_len(n, 5, NTT_PAR_CUTOFF_NTT_5)
    } else if n % 3 == 0 {
        ntt_radix_scratch_len(n, 3, NTT_PAR_CUTOFF_NTT_3)
    } else {
        0
    }
}

fn ntt_radix_scratch_len(n: usize, radix: usize, par_cutoff: usize) -> usize {
    let m = n / radix;
    let shuffle_scratch = (radix - 1) * m;
    let child_scratch = ntt_scratch_len(m);
    let recursive_scratch = if n > par_cutoff {
        radix * child_scratch
    } else {
        child_scratch
    };
    shuffle_scratch.max(recursive_scratch)
}

fn ntt_convolution_scratch_len(n: usize) -> usize {
    let ntt_scratch = ntt_scratch_len(n);
    if n > NTT_PAR_CUTOFF_NTT {
        2 * ntt_scratch
    } else {
        ntt_scratch
    }
}

fn ntt_2<P: NTTPrime, D: NTTDir>(
    buf: &mut [Montgomery<P>],
    stride: usize,
    tw: &impl NTTTwidles<P>,
    _scratch: &mut [Montgomery<P>],
) {
    let n = buf.len();
    if n < 2 {
        return;
    }
    debug_assert!(n.is_power_of_two());

    let log_n = n.trailing_zeros();
    let shift = usize::BITS - log_n;
    for i in 1..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            buf.swap(i, j);
        }
    }

    let mut m_half = 1;
    while m_half < n {
        let m = m_half << 1;
        let stage_stride = stride * (n / m);
        let mut k = 0;
        while k < n {
            let [mut tw0] = tw.twidles_iter::<1, D>(stage_stride);
            for j in 0..m_half {
                let t = unsafe { tw0.next().unwrap_unchecked() };
                let [x0, x1] = unsafe { buf.get_disjoint_unchecked_mut([k + j, k + j + m_half]) };
                let u = *x0;
                let v = *x1 * t;
                *x0 = u + v;
                *x1 = u - v;
            }
            k += m;
        }
        m_half = m;
    }
}

fn split<const N: usize, P: NTTPrime>(buf: &mut [Montgomery<P>]) -> [&mut [Montgomery<P>]; N] {
    debug_assert!(
        buf.len() % N == 0,
        "length of buf must be evenly divided by N = {N}"
    );
    let m = buf.len() / N;
    let base = buf.as_mut_ptr();
    let mut offset = 0usize;
    std::array::from_fn(|_| unsafe {
        let s = std::slice::from_raw_parts_mut(base.add(offset), m);
        offset += m;
        s
    })
}

fn shuffle<const N: usize, P: NTTPrime>(buf: &mut [Montgomery<P>], scratch: &mut [Montgomery<P>]) {
    let m = buf.len() / N;
    debug_assert!(scratch.len() >= (N - 1) * m);
    for i in 0..m {
        buf[i] = buf[N * i];
        for j in 1..N {
            scratch[(j - 1) * m + i] = buf[N * i + j];
        }
    }
    for j in 1..N {
        let start = (j - 1) * m;
        buf[j * m..(j + 1) * m].copy_from_slice(&scratch[start..start + m]);
    }
}

fn ntt_3<P: NTTPrime, D: NTTDir>(
    buf: &mut [Montgomery<P>],
    stride: usize,
    tw: &impl NTTTwidles<P>,
    scratch: &mut [Montgomery<P>],
) {
    let n = buf.len();
    let m = buf.len() / 3;
    debug_assert!(scratch.len() >= ntt_scratch_len(n));
    shuffle::<3, P>(buf, scratch);

    let new_stride = 3 * stride;
    {
        let [b0, b1, b2] = split(buf);
        let child_scratch_len = ntt_scratch_len(m);
        if n > NTT_PAR_CUTOFF_NTT_3 {
            let (s0, rest) = scratch.split_at_mut(child_scratch_len);
            let (s1, rest) = rest.split_at_mut(child_scratch_len);
            let (s2, _) = rest.split_at_mut(child_scratch_len);
            let ntt0 = || ntt::<P, D>(b0, new_stride, tw, s0);
            let ntt1 = || ntt::<P, D>(b1, new_stride, tw, s1);
            let ntt2 = || ntt::<P, D>(b2, new_stride, tw, s2);
            rayon::join(ntt0, || rayon::join(ntt1, ntt2));
        } else {
            let child_scratch = &mut scratch[..child_scratch_len];
            ntt::<P, D>(b0, new_stride, tw, child_scratch);
            ntt::<P, D>(b1, new_stride, tw, child_scratch);
            ntt::<P, D>(b2, new_stride, tw, child_scratch);
        }
    }

    let g = if D::INV {
        Montgomery::INV_G3
    } else {
        Montgomery::G3
    };

    let [mut tw0, mut tw1] = tw.twidles_iter::<2, D>(stride);
    for i in 0..m {
        let [x0, x1, x2] = unsafe { buf.get_disjoint_unchecked_mut([i, i + m, i + 2 * m]) };
        let (t0, t1) = unsafe { (tw0.next().unwrap_unchecked(), tw1.next().unwrap_unchecked()) };
        let a = *x0;
        let b = *x1 * t0;
        let c = *x2 * t1;
        let diff = (b - c) * g;
        *x0 = a + b + c;
        *x1 = a - c + diff;
        *x2 = a - b - diff;
    }
}

fn ntt_5<P: NTTPrime, D: NTTDir>(
    buf: &mut [Montgomery<P>],
    stride: usize,
    tw: &impl NTTTwidles<P>,
    scratch: &mut [Montgomery<P>],
) {
    let n = buf.len();
    let m = buf.len() / 5;
    debug_assert!(scratch.len() >= ntt_scratch_len(n));
    shuffle::<5, P>(buf, scratch);

    let new_stride = 5 * stride;
    {
        let [b0, b1, b2, b3, b4] = split(buf);
        let child_scratch_len = ntt_scratch_len(m);
        if n > NTT_PAR_CUTOFF_NTT_5 {
            let (s0, rest) = scratch.split_at_mut(child_scratch_len);
            let (s1, rest) = rest.split_at_mut(child_scratch_len);
            let (s2, rest) = rest.split_at_mut(child_scratch_len);
            let (s3, rest) = rest.split_at_mut(child_scratch_len);
            let (s4, _) = rest.split_at_mut(child_scratch_len);
            let ntt0 = || ntt::<P, D>(b0, new_stride, tw, s0);
            let ntt1 = || ntt::<P, D>(b1, new_stride, tw, s1);
            let ntt2 = || ntt::<P, D>(b2, new_stride, tw, s2);
            let ntt3 = || ntt::<P, D>(b3, new_stride, tw, s3);
            let ntt4 = || ntt::<P, D>(b4, new_stride, tw, s4);
            rayon::join(
                || rayon::join(ntt0, ntt1),
                || rayon::join(ntt2, || rayon::join(ntt3, ntt4)),
            );
        } else {
            let child_scratch = &mut scratch[..child_scratch_len];
            ntt::<P, D>(b0, new_stride, tw, child_scratch);
            ntt::<P, D>(b1, new_stride, tw, child_scratch);
            ntt::<P, D>(b2, new_stride, tw, child_scratch);
            ntt::<P, D>(b3, new_stride, tw, child_scratch);
            ntt::<P, D>(b4, new_stride, tw, child_scratch);
        }
    }

    let g = if D::INV {
        Montgomery::INV_G5
    } else {
        Montgomery::G5
    };
    let g2 = g * g;
    let g3 = g2 * g;
    let g4 = g3 * g;
    let alpha = (g + g4) * Montgomery::HALF;
    let beta = (g2 + g3) * Montgomery::HALF;
    let gamma = (g - g4) * Montgomery::HALF;
    let delta = (g2 - g3) * Montgomery::HALF;

    let [mut tw0, mut tw1, mut tw2, mut tw3] = tw.twidles_iter::<4, D>(stride);
    for i in 0..m {
        let [x0, x1, x2, x3, x4] =
            unsafe { buf.get_disjoint_unchecked_mut([i, i + m, i + 2 * m, i + 3 * m, i + 4 * m]) };
        let (t0, t1, t2, t3) = unsafe {
            (
                tw0.next().unwrap_unchecked(),
                tw1.next().unwrap_unchecked(),
                tw2.next().unwrap_unchecked(),
                tw3.next().unwrap_unchecked(),
            )
        };
        let a = *x0;
        let b = *x1 * t0;
        let c = *x2 * t1;
        let d = *x3 * t2;
        let e = *x4 * t3;

        let sum1 = b + e;
        let dif1 = b - e;
        let sum2 = c + d;
        let dif2 = c - d;
        let as1 = alpha * sum1;
        let gd1 = gamma * dif1;
        let bs2 = beta * sum2;
        let dd2 = delta * dif2;
        let bs1 = beta * sum1;
        let dd1 = delta * dif1;
        let as2 = alpha * sum2;
        let gd2 = gamma * dif2;

        *x0 = a + sum1 + sum2;
        *x1 = a + (as1 + gd1) + (bs2 + dd2);
        *x2 = a + (bs1 + dd1) + (as2 - gd2);
        *x3 = a + (bs1 - dd1) + (as2 + gd2);
        *x4 = a + (as1 - gd1) + (bs2 - dd2);
    }
}

fn ntt_convolution<P: NTTPrime>(
    a_buf: &[u64],
    b_buf: &[u64],
    res: &mut [u64],
    tw: &impl NTTTwidles<P>,
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    let n = res.len();
    let ntt_scratch_len = ntt_scratch_len(n);
    debug_assert!(buf_scratch.len() >= n);
    debug_assert!(ntt_scratch.len() >= ntt_scratch_len);

    res[..a_buf.len()].copy_from_slice(a_buf);
    res[a_buf.len()..].fill(0);
    buf_scratch[..b_buf.len()].copy_from_slice(b_buf);
    buf_scratch[b_buf.len()..].fill(0);

    {
        let (a, b) = (&mut *res, &mut *buf_scratch);

        let ntt_a = move || {
            let mont_a: &mut [Montgomery<P>] = unsafe { std::mem::transmute(a) };
            for a in mont_a.iter_mut().take(a_buf.len()) {
                a.to_mut();
            }
            mont_a
        };

        let ntt_b = move || {
            let mont_b: &mut [Montgomery<P>] = unsafe { std::mem::transmute(b) };
            for b in mont_b.iter_mut().take(b_buf.len()) {
                b.to_mut();
            }
            mont_b
        };

        if n > NTT_PAR_CUTOFF_NTT && ntt_scratch.len() >= 2 * ntt_scratch_len {
            let (a_ntt_scratch, b_ntt_scratch) = ntt_scratch.split_at_mut(ntt_scratch_len);
            let (mont_a, mont_b) = rayon::join(ntt_a, ntt_b);
            let mont_a_scratch: &mut [Montgomery<P>] =
                unsafe { std::mem::transmute(a_ntt_scratch) };
            let mont_b_scratch: &mut [Montgomery<P>] =
                unsafe { std::mem::transmute(&mut b_ntt_scratch[..ntt_scratch_len]) };
            rayon::join(
                || ntt::<P, NTTForward>(mont_a, 1, tw, mont_a_scratch),
                || ntt::<P, NTTForward>(mont_b, 1, tw, mont_b_scratch),
            );
        } else {
            let mont_a = ntt_a();
            let mont_ntt_scratch: &mut [Montgomery<P>] =
                unsafe { std::mem::transmute(&mut ntt_scratch[..ntt_scratch_len]) };
            ntt::<P, NTTForward>(mont_a, 1, tw, mont_ntt_scratch);

            let mont_b = ntt_b();
            let mont_ntt_scratch: &mut [Montgomery<P>] =
                unsafe { std::mem::transmute(&mut ntt_scratch[..ntt_scratch_len]) };
            ntt::<P, NTTForward>(mont_b, 1, tw, mont_ntt_scratch);
        }
    }

    let mont_a: &mut [Montgomery<P>] = unsafe { std::mem::transmute(res) };
    let mont_b: &mut [Montgomery<P>] = unsafe { std::mem::transmute(buf_scratch) };
    for (a, b) in mont_a.iter_mut().zip(mont_b.iter()) {
        *a *= *b;
    }

    ntt::<P, NTTBackward>(mont_a, 1, tw, mont_b);
}

struct CRT<P1: NTTPrime, P2: NTTPrime, P3: NTTPrime> {
    inv_p1_p2: Montgomery<P2>,
    inv_p1_p3: Montgomery<P3>,
    inv_p2_p3: Montgomery<P3>,
    p1p2_lo: u64,
    p1p2_hi: u64,
    _phantom: PhantomData<P1>,
}

impl<P1: NTTPrime, P2: NTTPrime, P3: NTTPrime> CRT<P1, P2, P3> {
    fn new() -> Self {
        let inv_p1_p2 = Montgomery::<P2>::to(P1::P).pow(P2::P - 2);
        let inv_p1_p3 = Montgomery::<P3>::to(P1::P).pow(P3::P - 2);
        let inv_p2_p3 = Montgomery::<P3>::to(P2::P).pow(P3::P - 2);
        let p1p2 = (P1::P as u128) * (P2::P as u128);
        CRT {
            inv_p1_p2,
            inv_p1_p3,
            inv_p2_p3,
            p1p2_lo: p1p2 as u64,
            p1p2_hi: (p1p2 >> 64) as u64,
            _phantom: PhantomData,
        }
    }

    fn crt(
        &self,
        r1: &mut u64,
        r2: &mut u64,
        r3: &mut u64,
        inv_n1: Montgomery<P1>,
        inv_n2: Montgomery<P2>,
        inv_n3: Montgomery<P3>,
    ) {
        let r1_p1 = Montgomery::cast(*r1) * inv_n1;
        let x1 = r1_p1.from();

        let r2_p2 = Montgomery::cast(*r2) * inv_n2;
        let x1_p2 = Montgomery::to(x1);
        let mont_x2 = (r2_p2 - x1_p2) * self.inv_p1_p2;
        let x2 = mont_x2.from();

        let r3_p3 = Montgomery::cast(*r3) * inv_n3;
        let x1_p3 = Montgomery::to(x1);
        let x2_p3 = Montgomery::to(x2);
        let mont_x3 = ((r3_p3 - x1_p3) * self.inv_p1_p3 - x2_p3) * self.inv_p2_p3;
        let x3 = mont_x3.from();

        let mask = u64::MAX as u128;

        let part1 = x1 as u128;

        let part2 = (x2 as u128) * (P1::P as u128);
        let part2_lo = part2 & mask;
        let part2_hi = part2 >> 64;

        let part3_bot = (x3 as u128) * (self.p1p2_lo as u128);
        let part3_top = (part3_bot >> 64) + (x3 as u128) * (self.p1p2_hi as u128);
        let part3_lo = part3_bot & mask;
        let part3_mid = part3_top & mask;
        let part3_hi = part3_top >> 64;

        let sum1 = part1 + part2_lo + part3_lo;
        let sum2 = (sum1 >> 64) + part2_hi + part3_mid;
        let sum3 = (sum2 >> 64) + part3_hi;

        *r1 = sum1 as u64;
        *r2 = sum2 as u64;
        *r3 = sum3 as u64;
    }
}

static CRT: LazyLock<CRT<P1, P2, P3>> = LazyLock::new(|| CRT::<P1, P2, P3>::new());

fn ntt_accumulate(
    res1: &mut [u64],
    res2: &mut [u64],
    res3: &mut [u64],
    out: &mut [u64],
    out_len: usize,
) -> u64 {
    let (n1, n2, n3) = (res1.len(), res2.len(), res3.len());
    let inv_n1 = Montgomery::to(n1 as u64).pow(P1::P - 2);
    let inv_n2 = Montgomery::to(n2 as u64).pow(P2::P - 2);
    let inv_n3 = Montgomery::to(n3 as u64).pow(P3::P - 2);
    for i in 0..out_len {
        CRT.crt(
            &mut res1[i],
            &mut res2[i],
            &mut res3[i],
            inv_n1,
            inv_n2,
            inv_n3,
        );
    }

    let len1 = n1.min(out.len());
    out[..len1].copy_from_slice(&res1[..len1]);
    let len2 = n2.min(out.len() - 1);
    let of1 = add_buf(&mut out[1..], &res2[..len2]) as u64;
    let len3 = n3.min(out.len() - 2);
    let of2 = add_buf(&mut out[2..], &res3[..len3]) as u64;

    res2.get(len2).copied().unwrap_or(0) + res3.get(len3).copied().unwrap_or(0) + of1 + of2
}

fn ntt_log_prime_search(
    init: usize,
    target: usize,
    primes: &[usize],
    max_exps: &[usize],
) -> Option<usize> {
    debug_assert_eq!(primes.len(), max_exps.len());
    if init >= target {
        return Some(init);
    }

    let Some((&p, rest_primes)) = primes.split_first() else {
        return None;
    };

    let (&max_exp, rest_exps) = max_exps.split_first().unwrap();
    let mut best: Option<usize> = None;
    let mut val = init;
    for exp in 0..=max_exp {
        let cur = ntt_log_prime_search(val, target, rest_primes, rest_exps);
        best = match (best, cur) {
            (Some(b), Some(c)) => Some(b.min(c)),
            (b, c) => b.or(c),
        };
        if best == Some(target) {
            return best;
        }
        if exp < max_exp {
            match val.checked_mul(p) {
                Some(next) if best.map_or(true, |b| next < b) => val = next,
                _ => break,
            }
        }
    }

    best
}

fn find_ntt_size<P: NTTPrime>(n: usize) -> usize {
    ntt_log_prime_search(1, n, &NTT_RADIX, &P::EXP).expect("Input Is too large for NTT")
}

pub fn find_max_ntt_size(n: usize) -> usize {
    find_ntt_size::<P1>(n)
        .max(find_ntt_size::<P2>(n))
        .max(find_ntt_size::<P3>(n))
}

pub fn ntt_entry_dyn(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    let out_len = a.len() + b.len() - 1;
    let n1 = find_ntt_size::<P1>(out_len);
    let n2 = find_ntt_size::<P2>(out_len);
    let n3 = find_ntt_size::<P3>(out_len);

    let mut res_guard = ScratchGuard::acquire();
    let [res1, res2, res3] = res_guard.get_splits([n1, n2, n3]);

    {
        let (r1, r2, r3) = (&mut *res1, &mut *res2, &mut *res3);

        let mut conv1 = move || {
            DYN_NTT_CACHE_P1.with(|cell| {
                let mut g = ScratchGuard::acquire();
                let [bs, ns] = g.get_splits([n1, ntt_convolution_scratch_len(n1)]);
                let cache = &mut *cell.borrow_mut();
                let tw = cache
                    .entry(n1 >> n1.trailing_zeros())
                    .or_insert_with(|| DynNTTTwidles::build(n1));
                tw.ensure(n1);
                ntt_convolution::<P1>(a, b, r1, tw, bs, ns);
            });
        };
        let mut conv2 = move || {
            DYN_NTT_CACHE_P2.with(|cell| {
                let mut g = ScratchGuard::acquire();
                let [bs, ns] = g.get_splits([n2, ntt_convolution_scratch_len(n2)]);
                let cache = &mut *cell.borrow_mut();
                let tw = cache
                    .entry(n2 >> n2.trailing_zeros())
                    .or_insert_with(|| DynNTTTwidles::build(n2));
                tw.ensure(n2);
                ntt_convolution::<P2>(a, b, r2, tw, bs, ns);
            });
        };
        let mut conv3 = move || {
            DYN_NTT_CACHE_P3.with(|cell| {
                let mut g = ScratchGuard::acquire();
                let [bs, ns] = g.get_splits([n3, ntt_convolution_scratch_len(n3)]);
                let cache = &mut *cell.borrow_mut();
                let tw = cache
                    .entry(n3 >> n3.trailing_zeros())
                    .or_insert_with(|| DynNTTTwidles::build(n3));
                tw.ensure(n3);
                ntt_convolution::<P3>(a, b, r3, tw, bs, ns);
            });
        };

        if out_len > NTT_PAR_CUTOFF_NTT_CONV {
            rayon::join(conv1, || rayon::join(conv2, conv3));
        } else {
            conv1();
            conv2();
            conv3();
        }
    }

    ntt_accumulate(res1, res2, res3, out, out_len)
}

fn ntt_log_prime_search_for_split(
    init: usize,
    target: usize,
    primes: &[usize],
    max_exps: &[usize],
) -> Option<usize> {
    debug_assert_eq!(primes.len(), max_exps.len());
    if init > target {
        return None;
    }

    let Some((&p, rest_primes)) = primes.split_first() else {
        return Some(init);
    };

    let (&max_exp, rest_exps) = max_exps.split_first().unwrap();
    let mut best: Option<usize> = None;
    let mut val = init;
    for exp in 0..=max_exp {
        let cur = ntt_log_prime_search_for_split(val, target, rest_primes, rest_exps);
        best = match (best, cur) {
            (Some(b), Some(c)) => Some(b.max(c)),
            (b, c) => b.or(c),
        };
        if best == Some(target) {
            return best;
        }
        if exp < max_exp {
            match val.checked_mul(p) {
                Some(next) if next <= target => val = next,
                _ => break,
            }
        }
    }

    best
}

pub(crate) fn find_ntt_size_for_split<const N: usize, P: NTTPrime>() -> usize {
    ntt_log_prime_search_for_split(1, N, &NTT_RADIX, &P::EXP).expect("Input Is too large for NTT")
}

pub(crate) fn acc_convolution<P: NTTPrime, F>(a_buf: &[u64], b_buf: &[u64], res: &mut [u64], op: F)
where
    F: Fn(u64, u64) -> Montgomery<P>,
{
    let mont_res: &mut [Montgomery<P>] = unsafe { std::mem::transmute(res) };
    for (i, &a) in a_buf.iter().enumerate() {
        for (j, &b) in b_buf.iter().enumerate() {
            mont_res[i + j] += op(a, b);
        }
    }
}

fn ntt_split_convolution<const N: usize, P: NTTPrime>(
    long: &[u64],
    short: &[u64],
    res: &mut [u64],
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    let n = find_ntt_size_for_split::<N, P>();
    let split = n + 1 - short.len();
    let (long_lo, long_hi) = long.split_at(split);
    let mut tw = StaticNTTTwidles::<N, P>::build(n);
    ntt_convolution::<P>(
        long_lo,
        short,
        &mut res[..n],
        &mut tw,
        &mut buf_scratch[..n],
        &mut ntt_scratch[..ntt_convolution_scratch_len(n)],
    );
    acc_convolution::<P, _>(short, long_hi, &mut res[split..], |a, b| {
        Montgomery::fuse_mul_to(a, b)
    });
}

fn ntt_static_convolution<const N: usize, P: NTTPrime>(
    long: &[u64],
    short: &[u64],
    res: &mut [u64],
    n: usize,
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    if n <= N {
        let mut tw = StaticNTTTwidles::<N, P>::build(n);
        ntt_convolution::<P>(
            long,
            short,
            &mut res[..n],
            &mut tw,
            &mut buf_scratch[..n],
            &mut ntt_scratch[..ntt_scratch_len(n)],
        );
    } else {
        ntt_split_convolution::<N, P>(long, short, res, buf_scratch, ntt_scratch);
    }
}

pub fn ntt_entry_static<const N: usize>(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let out_len = long.len() + short.len() - 1;
    let mut res1 = [0; N];
    let mut res2 = [0; N];
    let mut res3 = [0; N];

    let n1 = find_ntt_size::<P1>(out_len);
    let n2 = find_ntt_size::<P2>(out_len);
    let n3 = find_ntt_size::<P3>(out_len);

    if out_len > NTT_PAR_CUTOFF_NTT_CONV {
        let conv1 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_static_convolution::<N, P1>(
                long,
                short,
                &mut res1,
                n1,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        let conv2 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_static_convolution::<N, P2>(
                long,
                short,
                &mut res2,
                n2,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        let conv3 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_static_convolution::<N, P3>(
                long,
                short,
                &mut res3,
                n3,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        rayon::join(conv1, || rayon::join(conv2, conv3));
    } else {
        let mut buf_scratch = [0; N];
        let mut ntt_scratch = [0; N];
        ntt_static_convolution::<N, P1>(
            long,
            short,
            &mut res1,
            n1,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
        ntt_static_convolution::<N, P2>(
            long,
            short,
            &mut res2,
            n2,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
        ntt_static_convolution::<N, P3>(
            long,
            short,
            &mut res3,
            n3,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
    }
    ntt_accumulate(
        &mut res1[..n1],
        &mut res2[..n2],
        &mut res3[..n3],
        out,
        out_len,
    )
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

pub fn is_karatsuba(l: usize, s: usize, chunk: f64, reg: f64) -> bool {
    let fft_cost = fft_cost(l, s);
    let half = (l + 1) / 2;
    if s <= half {
        chunking_karatsuba_cost(l, s) < chunk * fft_cost
    } else {
        karatsuba_cost(l, s) < reg * fft_cost
    }
}

enum DynDispatch {
    Prim,
    Prim2,
    School,
    Karatsuba,
    FFT,
    NTT,
}

fn dyn_dispatch(l: usize, s: usize) -> DynDispatch {
    if s == 1 {
        DynDispatch::Prim
    } else if s == 2 {
        DynDispatch::Prim2
    } else if is_school(l, s) {
        DynDispatch::School
    } else if is_karatsuba(l, s, FFT_CHUNKING_KARATSUBA_CUTOFF, FFT_KARATSUBA_CUTOFF) {
        DynDispatch::Karatsuba
    } else if (l + s - 1) <= FFT_16BIT_CUTOFF {
        DynDispatch::FFT
    } else {
        DynDispatch::NTT
    }
}

pub fn mul_dyn(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let out_len = a.len() + b.len() - 1;
    debug_assert!(
        out.len() >= out_len,
        "out is not large enough for multiplication"
    );
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let out_core = &mut out[..out_len];
    let overflow = match dyn_dispatch(long.len(), short.len()) {
        DynDispatch::Prim => {
            out_core[..long.len()].copy_from_slice(long);
            mul_prim(out_core, short[0])
        }
        DynDispatch::Prim2 => {
            out_core[..long.len()].copy_from_slice(long);
            out_core[long.len()] = 0;
            mul_prim2(out_core, combine_u64(short[0], short[1])) as u64
        }
        DynDispatch::School => mul_buf(long, short, out_core),
        DynDispatch::Karatsuba => karatsuba_entry_dyn(long, short, out_core),
        DynDispatch::FFT => fft_entry(long, short, out_core),
        DynDispatch::NTT => ntt_entry_dyn(long, short, out_core),
    };
    if out.len() > out_len {
        out[out_len] = overflow;
        out[out_len + 1..].fill(0);
        0
    } else {
        overflow
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
    NTT,
}

fn static_dispatch(l: usize, s: usize) -> StaticDispatch {
    if s == 1 {
        StaticDispatch::Prim
    } else if s == 2 {
        StaticDispatch::Prim2
    } else if is_school(l, s) {
        StaticDispatch::School
    } else if is_karatsuba(l, s, NTT_CHUNKING_KARATSUBA_CUTOFF, NTT_KARATSUBA_CUTOFF) {
        StaticDispatch::Karatsuba
    } else {
        StaticDispatch::NTT
    }
}

pub fn mul_static<const N: usize>(a: &[u64], b: &[u64], out: &mut [u64]) -> Result<u64, ()> {
    if a.is_empty() || b.is_empty() {
        return Ok(0);
    }
    let out_len = a.len() + b.len() - 1;
    if out_len > out.len() {
        return Err(());
    }
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let out_core = &mut out[..out_len];
    let overflow = match static_dispatch(long.len(), short.len()) {
        StaticDispatch::Prim => {
            out_core[..long.len()].copy_from_slice(long);
            mul_prim(out_core, short[0])
        }
        StaticDispatch::Prim2 => {
            out_core[..long.len()].copy_from_slice(long);
            out_core[long.len()] = 0;
            mul_prim2(out_core, combine_u64(short[0], short[1])) as u64
        }
        StaticDispatch::School => mul_buf(long, short, out_core),
        StaticDispatch::Karatsuba => karatsuba_entry_static::<N>(long, short, out_core),
        StaticDispatch::NTT => ntt_entry_static::<N>(long, short, out_core),
    };
    Ok(if out.len() > out_len {
        out[out_len] = overflow;
        out[out_len + 1..].fill(0);
        0
    } else {
        overflow
    })
}

pub fn mul_arr<const N: usize>(a: &[u64], b: &[u64]) -> Result<([u64; N], u64), ()> {
    let mut out = [0; N];
    let of = mul_static::<N>(a, b, &mut out)?;
    Ok((out, of))
}

//SQUARE MULTIPLICATION
// Simple hierarchy School < Karatsuba < FFT < NTT. These are optimized algorithms specifically
// for squaring buffers. They all have constant factor speed ups.
// School -> 50%
// Karatsuba -> 40% - 50%
// FTT -> 33%
// NTT -> 33%

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

fn sqr_elem(buf: &[u64], n: usize, acc0: &mut u64, acc1: &mut u64, acc2: &mut u64) -> u64 {
    let mi = n.saturating_sub(buf.len() - 1);
    let mf = ((n + 1) / 2).min(buf.len());
    unsafe {
        for m in mi..mf {
            let a_val = *buf.get_unchecked(m);
            let b_val = *buf.get_unchecked(n - m);
            mul_double_asm(a_val, b_val, acc0, acc1, acc2);
        }
    }
    if n % 2 == 0 && n / 2 < buf.len() {
        unsafe {
            let d = *buf.get_unchecked(n / 2);
            mul_asm(d, d, acc0, acc1, acc2);
        }
    }
    let out = *acc0;
    *acc0 = *acc1;
    *acc1 = *acc2;
    *acc2 = 0;
    return out;
}

pub fn sqr_buf(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let (mut acc0, mut acc1, mut acc2) = (0, 0, 0);
    for n in 0..out.len() {
        out[n] = sqr_elem(buf, n, &mut acc0, &mut acc1, &mut acc2);
    }
    acc0
}

//KARATSUBA SQUARE

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
    if n <= KARATSUBA_SQR_CUTOFF {
        return 0;
    }
    let half = (n + 1) / 2;
    (2 * half + 1) + find_karatsuba_sqr_scratch_sz(half + 1)
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

//FFT SQUARE

#[inline(always)]
fn seperate(x: Complex<f64>, y: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
    let a = (x.re + y.re) * 0.5;
    let b = (x.im - y.im) * 0.5;
    let c = (x.im + y.im) * 0.5;
    let d = (x.re - y.re) * 0.5;
    (Complex::new(a, b), Complex::new(c, -d))
}

fn sqr_recombine(x: &mut [Complex<f64>], m: usize, tw: &[Complex<f64>]) {
    let Complex::<f64> { re, im } = x[0];
    x[0] = Complex::new(re * re + im * im, 2.0 * re * im);
    for k in 1..m / 2 {
        let (e, o) = seperate(x[k], x[m - k]);
        let (a, b) = seperate(e, tw[k] * o.conj());
        let s = (a + b) * (a - b);
        let t = e * o;
        x[k] = Complex::new(s.re - t.im, s.im + t.re).scale(2.0);
        x[m - k] = Complex::new(s.re + t.im, t.re - s.im).scale(2.0);
    }
    x[m / 2] = Complex::new(
        (x[m / 2].re + x[m / 2].im) * (x[m / 2].re - x[m / 2].im),
        2.0 * x[m / 2].re * x[m / 2].im,
    );
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

pub fn sqr_decompose_16_standard(buf: &[u64], x: &mut [Complex<f64>]) {
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

fn sqr_decompose(buf: &[u64], x: &mut [Complex<f64>]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { sqr_decompose_16_x86(buf, x) }
    } else {
        sqr_decompose(buf, x);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        sqr_decompose_16_aarch64(buf, x);
    }
}

pub fn fft_sqr_entry(buf: &[u64], out: &mut [u64]) -> u64 {
    let bit_len = bit16_length(buf.len(), buf.last().copied().unwrap());
    let n = find_fft_size(2 * bit_len - 1);
    let m = n / 2;
    FFT_CACHE.with(|cell| {
        let fft_cache = &mut *cell.borrow_mut();
        let (fwd, bwd, tw, scratch_sz) = fft_cache.prep_sqr(m);
        let mut scratch_gaurd = ScratchGuard::acquire();
        let complex_scratch: &mut [Complex<f64>] =
            unsafe { std::mem::transmute(scratch_gaurd.get(scratch_sz)) };
        let (x, fft_scratch) = complex_scratch.split_at_mut(m);

        sqr_decompose(buf, x);
        fft_sqr_core(x, fwd.as_ref(), bwd.as_ref(), tw, fft_scratch);
        fft_accumulate(x, out)
    })
}

//NTT SQUARE

fn ntt_sqr_convolution<P: NTTPrime>(
    buf: &[u64],
    res: &mut [u64],
    tw: &mut impl NTTTwidles<P>,
    ntt_scratch: &mut [u64],
) {
    res[..buf.len()].copy_from_slice(buf);
    res[buf.len()..].fill(0);
    let mont_res: &mut [Montgomery<P>] = unsafe { std::mem::transmute(res) };
    for elem in mont_res.iter_mut().take(buf.len()) {
        elem.to_mut();
    }
    let mont_scratch: &mut [Montgomery<P>] = unsafe { std::mem::transmute(ntt_scratch) };

    ntt::<P, NTTForward>(mont_res, 1, tw, mont_scratch);

    for elem in mont_res.iter_mut() {
        *elem *= *elem;
    }

    ntt::<P, NTTBackward>(mont_res, 1, tw, mont_scratch);
}

pub fn ntt_sqr_entry_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    let out_len = 2 * buf.len() - 1;
    let n1 = find_ntt_size::<P1>(out_len);
    let n2 = find_ntt_size::<P2>(out_len);
    let n3 = find_ntt_size::<P3>(out_len);

    let mut scratch_gaurd = ScratchGuard::acquire();
    let [res1, res2, res3] = scratch_gaurd.get_splits([n1, n2, n3]);

    {
        let (r1, r2, r3) = (&mut *res1, &mut *res2, &mut *res3);

        let mut conv1 = move || {
            DYN_NTT_CACHE_P1.with(|cell| {
                let mut g = ScratchGuard::acquire();
                let ns = g.get(ntt_scratch_len(n1));
                let cache = &mut *cell.borrow_mut();
                let tw = cache
                    .entry(n1 >> n1.trailing_zeros())
                    .or_insert_with(|| DynNTTTwidles::build(n1));
                tw.ensure(n1);
                ntt_sqr_convolution::<P1>(buf, r1, tw, ns);
            });
        };
        let mut conv2 = move || {
            DYN_NTT_CACHE_P2.with(|cell| {
                let mut g = ScratchGuard::acquire();
                let ns = g.get(ntt_scratch_len(n2));
                let cache = &mut *cell.borrow_mut();
                let tw = cache
                    .entry(n2 >> n2.trailing_zeros())
                    .or_insert_with(|| DynNTTTwidles::build(n2));
                tw.ensure(n2);
                ntt_sqr_convolution::<P2>(buf, r2, tw, ns);
            });
        };
        let mut conv3 = move || {
            DYN_NTT_CACHE_P3.with(|cell| {
                let mut g = ScratchGuard::acquire();
                let ns = g.get(ntt_scratch_len(n3));
                let cache = &mut *cell.borrow_mut();
                let tw = cache
                    .entry(n3 >> n3.trailing_zeros())
                    .or_insert_with(|| DynNTTTwidles::build(n3));
                tw.ensure(n3);
                ntt_sqr_convolution::<P3>(buf, r3, tw, ns);
            });
        };

        if out_len > NTT_PAR_CUTOFF_NTT_CONV {
            rayon::join(conv1, || rayon::join(conv2, conv3));
        } else {
            conv1();
            conv2();
            conv3();
        }
    }
    ntt_accumulate(res1, res2, res3, out, out_len)
}

fn ntt_split_sqr_convolution<const N: usize, P: NTTPrime>(
    buf: &[u64],
    res: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    let n = find_ntt_size_for_split::<N, P>();
    let split = (n + 1) / 2;
    let (buf_lo, buf_hi) = buf.split_at(split);
    let mut tw = StaticNTTTwidles::<N, P>::build(n);
    ntt_sqr_convolution::<P>(
        buf_lo,
        &mut res[..n],
        &mut tw,
        &mut ntt_scratch[..ntt_scratch_len(n)],
    );
    acc_convolution::<P, _>(buf_lo, buf_hi, &mut res[split..], |a, b| {
        let val = Montgomery::fuse_mul_to(a, b);
        val + val
    });
    acc_convolution::<P, _>(buf_hi, buf_hi, &mut res[2 * split..], |a, b| {
        Montgomery::fuse_mul_to(a, b)
    });
}

fn ntt_static_sqr_convolution<const N: usize, P: NTTPrime>(
    buf: &[u64],
    res: &mut [u64],
    n: usize,
    ntt_scratch: &mut [u64],
) {
    if n <= N {
        let mut tw = StaticNTTTwidles::<N, P>::build(n);
        ntt_sqr_convolution::<P>(
            buf,
            &mut res[..n],
            &mut tw,
            &mut ntt_scratch[..ntt_scratch_len(n)],
        );
    } else {
        ntt_split_sqr_convolution::<N, P>(buf, res, ntt_scratch);
    }
}

pub fn ntt_sqr_entry_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> u64 {
    let out_len = 2 * buf.len() - 1;
    let mut res1 = [0; N];
    let mut res2 = [0; N];
    let mut res3 = [0; N];
    let n1 = find_ntt_size::<P1>(out_len);
    let n2 = find_ntt_size::<P2>(out_len);
    let n3 = find_ntt_size::<P3>(out_len);

    if out_len > NTT_PAR_CUTOFF_NTT_CONV {
        let conv1 = || {
            let mut ntt_scratch = [0; N];
            ntt_static_sqr_convolution::<N, P1>(buf, &mut res1, n1, &mut ntt_scratch);
        };

        let conv2 = || {
            let mut ntt_scratch = [0; N];
            ntt_static_sqr_convolution::<N, P2>(buf, &mut res2, n2, &mut ntt_scratch);
        };

        let conv3 = || {
            let mut ntt_scratch = [0; N];
            ntt_static_sqr_convolution::<N, P3>(buf, &mut res3, n3, &mut ntt_scratch);
        };

        rayon::join(conv1, || rayon::join(conv2, conv3));
    } else {
        let mut ntt_scratch = [0; N];
        ntt_static_sqr_convolution::<N, P1>(buf, &mut res1, n1, &mut ntt_scratch);
        ntt_static_sqr_convolution::<N, P2>(buf, &mut res2, n2, &mut ntt_scratch);
        ntt_static_sqr_convolution::<N, P3>(buf, &mut res3, n3, &mut ntt_scratch);
    }
    ntt_accumulate(
        &mut res1[..n1],
        &mut res2[..n2],
        &mut res3[..n3],
        out,
        out_len,
    )
}

//END NTT SQUARE

enum DynSqrDispatch {
    School,
    FFT,
    NTT,
}

fn dyn_sqr_dispatch(n: usize) -> DynSqrDispatch {
    if n <= FFT_SQR_CUTOFF {
        DynSqrDispatch::School
    } else if n <= FFT_16BIT_CUTOFF {
        DynSqrDispatch::FFT
    } else {
        DynSqrDispatch::NTT
    }
}

pub fn sqr_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    debug_assert!(
        2 * buf.len() - 1 <= out.len(),
        "out is not large enough for multiplication"
    );
    match dyn_sqr_dispatch(buf.len()) {
        DynSqrDispatch::School => sqr_buf(buf, out),
        DynSqrDispatch::FFT => fft_sqr_entry(buf, out),
        DynSqrDispatch::NTT => ntt_sqr_entry_dyn(buf, out),
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
    NTT,
}

fn static_sqr_dispatch(n: usize) -> StaticSqrDispatch {
    if n <= KARATSUBA_SQR_CUTOFF {
        StaticSqrDispatch::School
    } else if n <= STATIC_NTT_SQR_CUTOFF {
        StaticSqrDispatch::Karatsubaa
    } else {
        StaticSqrDispatch::NTT
    }
}

pub fn sqr_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> Result<u64, ()> {
    if 2 * buf.len() - 1 > N {
        return Err(());
    }
    Ok(match static_sqr_dispatch(buf.len()) {
        StaticSqrDispatch::School => sqr_buf(buf, out),
        StaticSqrDispatch::Karatsubaa => karatsuba_sqr_entry_static::<N>(buf, out),
        StaticSqrDispatch::NTT => ntt_sqr_entry_static::<N>(buf, out),
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
    let d = (a.len() + b.len() - 1).saturating_sub(out.len());

    let (mut acc0, mut acc1, mut acc2) = (0, 0, 0);
    if d >= 2 {
        mul_elem(a, b, d - 2, &mut acc0, &mut acc1, &mut acc2);
    }
    if d >= 1 {
        mul_elem(a, b, d - 1, &mut acc0, &mut acc1, &mut acc2);
    }
    for n in d..d + out.len() {
        out[n - d] = mul_elem(a, b, n, &mut acc0, &mut acc1, &mut acc2);
    }
    return acc0;
}

pub fn short_mul_dyn(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    if a.len() + b.len() - 1 <= out.len() {
        return mul_dyn(a, b, out);
    }
    if out.len() <= SHORT_MUL_CUTOFF {
        return short_mul_buf(a, b, out);
    }
    let trunc_a = &a[a.len().saturating_sub(out.len())..];
    let trunc_b = &b[b.len().saturating_sub(out.len())..];
    let mut scratch = ScratchGuard::acquire();
    let tmp_len = trunc_a.len() + trunc_b.len() - 1;
    let tmp = scratch.get(tmp_len);
    let of = mul_dyn(trunc_a, trunc_b, tmp);
    out.copy_from_slice(&tmp[tmp_len - out.len()..]);
    return of;
}

pub fn short_mul_static<const N: usize>(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
    debug_assert!(a.len() <= N);
    debug_assert!(b.len() <= N);
    debug_assert!(
        out.len() <= N,
        "out is to large for static parameter N: {N}"
    );
    let out_len = out.len();
    if a.len() + b.len() - 1 <= out.len() {
        return mul_static::<N>(a, b, out).unwrap();
    }
    if out_len <= SHORT_MUL_CUTOFF {
        return short_mul_buf(a, b, out);
    }

    let trunc_a = &a[a.len().saturating_sub(out_len)..];
    let trunc_b = &b[b.len().saturating_sub(out_len)..];
    let tmp_len = trunc_a.len() + trunc_b.len() - 1;
    if trunc_a.len() + trunc_b.len() - 1 <= N {
        let mut tmp = [0; N];
        let of = mul_static::<N>(trunc_a, trunc_b, &mut tmp[..tmp_len]).unwrap();
        out.copy_from_slice(&tmp[tmp_len - out_len..tmp_len]);
        return of;
    }

    let (long, short) = if trunc_a.len() > trunc_b.len() {
        (trunc_a, trunc_b)
    } else {
        (trunc_b, trunc_a)
    };

    let k = out_len + 1;
    let beta_min = k.saturating_sub(long.len()).max(1);
    let beta_max = short.len().min(k - 1);
    let beta_pref = k / 2;
    let beta = beta_pref.clamp(beta_min, beta_max);
    let alpha = k - beta;
    let l_split_idx = long.len() - alpha;
    let s_split_idx = short.len() - beta;
    let (l0, l1) = long.split_at(l_split_idx);
    let (s0, s1) = short.split_at(s_split_idx);

    let mut of = mul_static::<N>(l1, s1, out).unwrap();
    let mut low_carry = 0u64;
    let low_len = l0.len() + s0.len();
    let mut low = [0; N];

    if !l0.is_empty() && !s0.is_empty() {
        debug_assert!(low_len <= N);
        let low_of = mul_static::<N>(l0, s0, &mut low[..low_len]).unwrap();
        debug_assert_eq!(low_of, 0);
    }

    if !l0.is_empty() {
        let tmp1_len = l0.len() + s1.len() - 1;
        let mut tmp1 = [0; N];
        let tmp1_of = mul_static::<N>(l0, s1, &mut tmp1[..tmp1_len]).unwrap();

        low_carry += add_buf(&mut low[s0.len()..low_len], &tmp1[..l0.len()]) as u64;
        of += add_buf(out, &tmp1[l0.len()..tmp1_len]) as u64;
        of += add_prim(&mut out[tmp1_len - l0.len()..], tmp1_of) as u64;
    }

    if !s0.is_empty() {
        let tmp2_len = l1.len() + s0.len();
        let mut tmp2 = [0; N];
        let tmp2_of = mul_static::<N>(l1, s0, &mut tmp2[..tmp2_len]).unwrap();
        debug_assert_eq!(tmp2_of, 0);

        low_carry += add_buf(&mut low[l0.len()..low_len], &tmp2[..s0.len()]) as u64;
        of += add_buf(out, &tmp2[s0.len()..tmp2_len]) as u64;
    }

    of += add_prim(out, low_carry) as u64;

    return of;
}

//SHORT SQUARING

pub fn short_sqr_buf(buf: &[u64], out: &mut [u64]) -> u64 {
    if buf.is_empty() {
        return 0;
    }
    let d = (2 * buf.len() - 1).saturating_sub(out.len());
    let (mut acc0, mut acc1, mut acc2) = (0, 0, 0);
    if d >= 2 {
        sqr_elem(buf, d - 2, &mut acc0, &mut acc1, &mut acc2);
    }
    if d >= 1 {
        sqr_elem(buf, d - 1, &mut acc0, &mut acc1, &mut acc2);
    }
    for n in d..d + out.len() {
        out[n - d] = sqr_elem(buf, n, &mut acc0, &mut acc1, &mut acc2);
    }
    return acc0;
}

pub fn short_sqr_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    if 2 * buf.len() - 1 <= out.len() {
        return sqr_dyn(buf, out);
    }
    if out.len() <= SHORT_SQR_CUTOFF {
        return short_sqr_buf(buf, out);
    }
    let trunc_buf = &buf[buf.len().saturating_sub(out.len())..];
    let mut scratch = ScratchGuard::acquire();
    let tmp_len = 2 * trunc_buf.len() - 1;
    let tmp = scratch.get(tmp_len);
    let of = sqr_dyn(trunc_buf, tmp);
    out.copy_from_slice(&tmp[tmp_len - out.len()..]);
    return of;
}

pub fn short_sqr_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> u64 {
    debug_assert!(buf.len() <= N);
    debug_assert!(
        out.len() <= N,
        "out is to large for static parameter N: {N}"
    );
    let out_len = out.len();
    if 2 * buf.len() - 1 <= out.len() {
        return sqr_static::<N>(buf, out).unwrap();
    }
    if out_len <= SHORT_SQR_CUTOFF {
        return short_sqr_buf(buf, out);
    }

    let trunc_buf = &buf[buf.len().saturating_sub(out.len())..];
    let tmp_len = 2 * trunc_buf.len() - 1;
    if tmp_len <= N {
        let mut tmp = [0; N];
        let of = sqr_static::<N>(trunc_buf, &mut tmp[..tmp_len]).unwrap();
        out.copy_from_slice(&tmp[tmp_len - out.len()..tmp_len]);
        return of;
    }

    let h = (out_len + 1) / 2;
    let (x0, x1) = trunc_buf.split_at(trunc_buf.len() - h);

    let hi_ofs = out.len() + 1 - 2 * h;
    let d = tmp_len - out_len;
    debug_assert_eq!(2 * x0.len() - hi_ofs, d);

    out[..hi_ofs].fill(0);
    let mut of = sqr_static::<N>(x1, &mut out[hi_ofs..]).unwrap();

    let mut low = [0u64; N];
    let x0_sqr_len = 2 * x0.len() - 1;
    let x0_sqr_of = sqr_static::<N>(x0, &mut low[..x0_sqr_len]).unwrap();
    let x0_high = if x0_sqr_len < d {
        low[x0_sqr_len] = x0_sqr_of;
        0
    } else {
        x0_sqr_of
    };

    let cross_len = x0.len() + x1.len() - 1;
    let mut cross = [0u64; N];
    let mut cross_of = mul_static::<N>(x0, x1, &mut cross[..cross_len]).unwrap();
    let shl_of = shl_buf(&mut cross[..cross_len], 1);
    let cross_of_hi = cross_of >> 63;
    cross_of = (cross_of << 1) | shl_of;

    let cross_start = d - x0.len();
    let mut low_carry = 0u64;
    if cross_start > 0 {
        low_carry += add_buf(&mut low[x0.len()..d], &cross[..cross_start]) as u64;
    }

    of += add_prim(out, x0_high) as u64;
    of += add_prim(out, low_carry) as u64;
    if cross_start < cross_len {
        let cross_high_len = cross_len - cross_start;
        of += add_buf(out, &cross[cross_start..cross_len]) as u64;
        of += add_prim(&mut out[cross_high_len..], cross_of) as u64;
        of += add_prim(&mut out[cross_high_len + 1..], cross_of_hi) as u64;
    } else {
        of += add_prim(out, cross_of) as u64;
        of += add_prim(&mut out[1..], cross_of_hi) as u64;
    }

    return of;
}

//MIDDLE MULTIPLICATION
// gets the middle part n limbs of a n x 2n product
// n: [0..n-1] x 2n-1: [0..2n-2] -> 3n-1: [0..3n-3] -> [n-1 .. 2n-2]

pub fn middle_correction(long: &[u64], short: &[u64]) -> (u64, u64) {
    debug_assert!(long.len() == 2 * short.len() - 1);
    let n = short.len();
    let (mut acc0, mut acc1, mut acc2) = (0, 0, 0);
    mul_elem(
        long,
        short,
        n.saturating_sub(3),
        &mut acc0,
        &mut acc1,
        &mut acc2,
    );
    mul_elem(
        long,
        short,
        n.saturating_sub(2),
        &mut acc0,
        &mut acc1,
        &mut acc2,
    );
    return (acc0, acc1);
}

pub fn mid_mul_buf(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    // long (2n-1) , short (n)
    let (mut acc0, mut acc1) = middle_correction(long, short);
    let mut acc2: u64 = 0;
    let n = short.len();
    for i in (n - 1)..=(2 * n - 2) {
        out[i + 1 - n] = mul_elem(long, short, i, &mut acc0, &mut acc1, &mut acc2);
    }
    return acc0;
}

pub fn fft_mid_mul(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let sz = short.len();
    let (l_len, s_len, w) = (
        bit16_length(long.len(), long.last().copied().unwrap()),
        bit16_length(short.len(), short.last().copied().unwrap()),
        4,
    );
    let n = find_fft_size(l_len + s_len - (sz - 1) * w - 1);
    let mut of = FFT_CACHE.with(|cell| {
        let fft_cache = &mut *cell.borrow_mut();
        let (fwd, bwd, tw, scratch_sz) = fft_cache.prep_mul(n);
        let mut scratch_gaurd = ScratchGuard::acquire();
        let complex_scratch: &mut [Complex<f64>] =
            unsafe { std::mem::transmute(scratch_gaurd.get(scratch_sz)) };
        let (x, fft_scratch) = complex_scratch.split_at_mut(n);

        decompose(long, short, x);
        fft_core(x, fwd.as_ref(), bwd.as_ref(), tw, fft_scratch);
        fft_accumulate(&x[(sz - 1) * w / 2..(2 * sz - 1) * w / 2], out)
    });
    let (c0, c1) = middle_correction(long, short);
    of += add_prim(out, c0) as u64;
    of += add_prim(&mut out[1..], c1) as u64;
    return of;
}

fn ntt_mid_accumulate(
    res1: &mut [u64],
    res2: &mut [u64],
    res3: &mut [u64],
    out: &mut [u64], // len == s
    s: usize,
) -> u64 {
    ntt_mid_accumulate_scaled(res1, res2, res3, out, s, res1.len(), res2.len(), res3.len())
}

fn ntt_mid_accumulate_scaled(
    res1: &mut [u64],
    res2: &mut [u64],
    res3: &mut [u64],
    out: &mut [u64], // len == s
    s: usize,
    n1: usize,
    n2: usize,
    n3: usize,
) -> u64 {
    let inv_n1 = Montgomery::to(n1 as u64).pow(P1::P - 2);
    let inv_n2 = Montgomery::to(n2 as u64).pow(P2::P - 2);
    let inv_n3 = Montgomery::to(n3 as u64).pow(P3::P - 2);

    let crt_start = s - 3;
    let crt_end = 2 * s - 1;
    for i in crt_start..crt_end {
        CRT.crt(
            &mut res1[i],
            &mut res2[i],
            &mut res3[i],
            inv_n1,
            inv_n2,
            inv_n3,
        );
    }

    out.copy_from_slice(&res1[s - 1..2 * s - 1]);
    let of1 = add_buf(out, &res2[s - 2..2 * s - 2]) as u64;
    let of2 = add_buf(out, &res3[s - 3..2 * s - 3]) as u64;

    res2[2 * s - 2] + res3[2 * s - 3] + of1 + of2
}

fn acc_scaled_cyclic_convolution<const N: usize, P: NTTPrime>(
    chunk: &[u64],
    short: &[u64],
    offset: usize,
    modulus_len: usize,
    res: &mut [u64],
    tmp: &mut [u64],
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    let conv_len = chunk.len() + short.len() - 1;
    let n = find_ntt_size::<P>(conv_len);
    debug_assert!(n <= N);

    let mut tw = StaticNTTTwidles::<N, P>::build(n);
    ntt_convolution::<P>(
        chunk,
        short,
        &mut tmp[..n],
        &mut tw,
        &mut buf_scratch[..n],
        ntt_scratch,
    );

    let scale =
        Montgomery::<P>::to(modulus_len as u64) * Montgomery::<P>::to(n as u64).pow(P::P - 2);
    let mont_tmp: &mut [Montgomery<P>] = unsafe { std::mem::transmute(&mut tmp[..conv_len]) };
    let mont_res: &mut [Montgomery<P>] = unsafe { std::mem::transmute(res) };
    for (i, elem) in mont_tmp.iter().enumerate() {
        let mut idx = offset + i;
        if idx >= modulus_len {
            idx -= modulus_len;
        }
        if idx < mont_res.len() {
            mont_res[idx] += *elem * scale;
        }
    }
}

fn ntt_mid_static_convolution<const N: usize, P: NTTPrime>(
    long: &[u64],
    short: &[u64],
    res: &mut [u64],
    n: usize,
    buf_scratch: &mut [u64],
    ntt_scratch: &mut [u64],
) {
    debug_assert!(long.len() <= N);
    debug_assert!(short.len() <= N);
    debug_assert!(res.len() >= long.len());

    if n <= N {
        let mut tw = StaticNTTTwidles::<N, P>::build(n);
        ntt_convolution::<P>(
            long,
            short,
            &mut res[..n],
            &mut tw,
            &mut buf_scratch[..n],
            ntt_scratch,
        );
        return;
    }

    res.fill(0);
    let split_n = find_ntt_size_for_split::<N, P>();
    let chunk_len = split_n + 1 - short.len();
    debug_assert!(chunk_len > 0);

    let mut tmp = [0; N];
    let mut offset = 0;
    while offset < long.len() {
        let end = (offset + chunk_len).min(long.len());
        acc_scaled_cyclic_convolution::<N, P>(
            &long[offset..end],
            short,
            offset,
            n,
            res,
            &mut tmp,
            buf_scratch,
            ntt_scratch,
        );
        offset = end;
    }
}

pub fn ntt_mid_mul_dyn(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let sz = short.len();
    let n1 = find_ntt_size::<P1>(2 * sz - 1);
    let n2 = find_ntt_size::<P2>(2 * sz - 1);
    let n3 = find_ntt_size::<P3>(2 * sz - 1);
    let mut scratch_gaurd = ScratchGuard::acquire();
    let [res1, res2, res3] = scratch_gaurd.get_splits([n1, n2, n3]);

    {
        let (r1, r2, r3) = (&mut *res1, &mut *res2, &mut *res3);

        let mut conv1 = move || {
            DYN_NTT_CACHE_P1.with(|cell| {
                let mut g = ScratchGuard::acquire();
                let [bs, ns] = g.get_splits([n1, ntt_convolution_scratch_len(n1)]);
                let cache = &mut *cell.borrow_mut();
                let tw = cache
                    .entry(n1 >> n1.trailing_zeros())
                    .or_insert_with(|| DynNTTTwidles::build(n1));
                tw.ensure(n1);
                ntt_convolution::<P1>(long, short, r1, tw, bs, ns);
            });
        };

        let mut conv2 = move || {
            DYN_NTT_CACHE_P2.with(|cell| {
                let mut g = ScratchGuard::acquire();
                let [bs, ns] = g.get_splits([n2, ntt_convolution_scratch_len(n2)]);
                let cache = &mut *cell.borrow_mut();
                let tw = cache
                    .entry(n2 >> n2.trailing_zeros())
                    .or_insert_with(|| DynNTTTwidles::build(n2));
                tw.ensure(n2);
                ntt_convolution::<P2>(long, short, r2, tw, bs, ns);
            });
        };
        let mut conv3 = move || {
            DYN_NTT_CACHE_P3.with(|cell| {
                let mut g = ScratchGuard::acquire();
                let [bs, ns] = g.get_splits([n3, ntt_convolution_scratch_len(n3)]);
                let cache = &mut *cell.borrow_mut();
                let tw = cache
                    .entry(n3 >> n3.trailing_zeros())
                    .or_insert_with(|| DynNTTTwidles::build(n3));
                tw.ensure(n3);
                ntt_convolution::<P3>(long, short, r3, tw, bs, ns);
            });
        };

        if 2 * sz - 1 > NTT_PAR_CUTOFF_NTT_CONV {
            rayon::join(conv1, || rayon::join(conv2, conv3));
        } else {
            conv1();
            conv2();
            conv3();
        }
    }
    let mut of = ntt_mid_accumulate(res1, res2, res3, out, sz);
    let (c0, c1) = middle_correction(long, short);
    of += add_prim(out, c0) as u64;
    of += add_prim(&mut out[1..], c1) as u64;
    return of;
}

pub fn ntt_mid_mul_static<const N: usize>(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
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
    debug_assert!(long.len() <= N);
    debug_assert!(short.len() <= N);
    debug_assert!(out.len() <= N);

    if short.len() < 3 {
        return mid_mul_buf(long, short, out);
    }

    let sz = short.len();
    let n1 = find_ntt_size::<P1>(2 * sz - 1);
    let n2 = find_ntt_size::<P2>(2 * sz - 1);
    let n3 = find_ntt_size::<P3>(2 * sz - 1);

    let mut res1 = [0; N];
    let mut res2 = [0; N];
    let mut res3 = [0; N];

    if 2 * sz - 1 > NTT_PAR_CUTOFF_NTT_CONV {
        let conv1 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_mid_static_convolution::<N, P1>(
                long,
                short,
                &mut res1,
                n1,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        let conv2 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_mid_static_convolution::<N, P2>(
                long,
                short,
                &mut res2,
                n2,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        let conv3 = || {
            let mut buf_scratch = [0; N];
            let mut ntt_scratch = [0; N];
            ntt_mid_static_convolution::<N, P3>(
                long,
                short,
                &mut res3,
                n3,
                &mut buf_scratch,
                &mut ntt_scratch,
            );
        };

        rayon::join(conv1, || rayon::join(conv2, conv3));
    } else {
        let mut buf_scratch = [0; N];
        let mut ntt_scratch = [0; N];
        ntt_mid_static_convolution::<N, P1>(
            long,
            short,
            &mut res1,
            n1,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
        ntt_mid_static_convolution::<N, P2>(
            long,
            short,
            &mut res2,
            n2,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
        ntt_mid_static_convolution::<N, P3>(
            long,
            short,
            &mut res3,
            n3,
            &mut buf_scratch,
            &mut ntt_scratch,
        );
    }

    let res1_len = n1.min(N);
    let res2_len = n2.min(N);
    let res3_len = n3.min(N);
    let mut of = ntt_mid_accumulate_scaled(
        &mut res1[..res1_len],
        &mut res2[..res2_len],
        &mut res3[..res3_len],
        out,
        sz,
        n1,
        n2,
        n3,
    );
    let (c0, c1) = middle_correction(long, short);
    of += add_prim(out, c0) as u64;
    of += add_prim(&mut out[1..], c1) as u64;
    return of;
}

enum DynMidDispatch {
    School,
    FFT,
    NTT,
}

fn dyn_mid_dispatch(n: usize) -> DynMidDispatch {
    return if n < FFT_MID_CUTOFF {
        DynMidDispatch::School
    } else if 2 * n < FFT_16BIT_CUTOFF {
        DynMidDispatch::FFT
    } else {
        DynMidDispatch::NTT
    };
}

pub fn mid_mul_dyn(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
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
        DynMidDispatch::FFT => fft_mid_mul(long, short, out),
        DynMidDispatch::NTT => ntt_mid_mul_dyn(long, short, out),
    }
}

pub fn mid_mul_static<const N: usize>(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
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
    if short.len() < NTT_MID_CUTOFF {
        mid_mul_buf(long, short, out)
    } else {
        ntt_mid_mul_static::<N>(long, short, out)
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
