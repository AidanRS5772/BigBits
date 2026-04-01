#![allow(dead_code)]
use crate::utils::{utils::*, Scratch, SCRATCH_POOL};
use rustfft::num_traits::{One, Zero};
use rustfft::{num_complex::Complex, Fft, FftDirection, FftPlanner};
use std::arch::asm;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;
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
// Mid tier complexity overhead is still reletively small, Mid tier algorithm. It handles unbalanced inputs through a
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

    sub_buf(cross, z0);
    sub_buf(cross, z2);
    if overflow > 0 {
        sub_buf(&mut cross[z2.len()..], &[overflow]);
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
    for i in 0..chunks {
        let of = s * i;
        let (val_cross, rest) = scratch.split_at_mut(2 * s + 2 * half + 1);
        let (cross, val) = val_cross.split_at_mut(2 * half + 1);
        karatsuba_core(&long[of..of + s], short, half, val, cross, rest);
        add_buf(&mut out[of..], &val);
    }
    let of = chunks * s;
    let (val, rest) = scratch.split_at_mut(s + l - of - 1);
    let mut overflow = karatsuba_mul(&long[of..], short, val, rest);
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

const CHUNKING_KARATSUBA_CUTOFF: usize = 32;
const BALENCED_KARATSUBA_CUTOFF: usize = 32;

fn is_school(l: usize, s: usize) -> bool {
    let half = (l + 1) / 2;
    if s <= half {
        s <= CHUNKING_KARATSUBA_CUTOFF
    } else {
        let c = BALENCED_KARATSUBA_CUTOFF as f64;
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
    } else if s < (l + 1) / 2 {
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

fn find_karatsuab_scratch(l: usize, s: usize) -> usize {
    match kdispatch(l, s) {
        KDispatch::Prim | KDispatch::Prim2 | KDispatch::School => 0,
        KDispatch::Chunking => 2 * s + find_karatsuab_scratch(s, s),
        KDispatch::Recurse => {
            let half = (l + 1) / 2;
            2 * half + 1 + find_karatsuab_scratch(half + 1, half + 1)
        }
    }
}

pub fn karatsuba_entry_dyn(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let scratch_sz = find_karatsuab_scratch(long.len(), short.len());
    SCRATCH_POOL.with(|cell| {
        let scratch = &mut *cell.borrow_mut();
        karatsuba_mul(long, short, out, scratch.get(scratch_sz))
    })
}

pub fn karatsuba_entry_static<const N: usize>(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    let scratch_sz = find_karatsuab_scratch(long.len(), short.len());
    let mut scratch = [0; N];
    if scratch_sz < N {
        karatsuba_mul(long, short, out, &mut scratch)
    } else {
        let mut cross = [0; N];
        let half = (long.len() + 1) / 2;
        karatsuba_core(long, short, half, out, &mut cross, &mut scratch)
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
    scratch_pool: Scratch<Complex<f64>>,
}

thread_local! {
    static FFT_CACHE: RefCell<FFTCache> = RefCell::new(FFTCache { planner: FftPlanner::new(), twidles: HashMap::new(), scratch_pool: Scratch::new()});
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

fn decompose(a: &[u64], b: &[u64], x: &mut [Complex<f64>]) {
    let max_len = x.len() / 4;
    let mask = 0xFFFF;
    for i in 0..max_len {
        let a_val = a.get(i).copied().unwrap_or(0);
        let b_val = b.get(i).copied().unwrap_or(0);
        let idx = i * 4;
        x[idx] = Complex {
            re: (a_val & mask) as f64,
            im: (b_val & mask) as f64,
        };
        x[idx + 1] = Complex {
            re: (a_val >> 16 & mask) as f64,
            im: (b_val >> 16 & mask) as f64,
        };
        x[idx + 2] = Complex {
            re: (a_val >> 32 & mask) as f64,
            im: (b_val >> 32 & mask) as f64,
        };
        x[idx + 3] = Complex {
            re: (a_val >> 48) as f64,
            im: (b_val >> 48) as f64,
        };
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

        x[k] = (s + Complex::<f64>::i() * dw).scale(0.25);
        x[m - k] = (s.conj() + Complex::<f64>::i() * dw.conj()).scale(0.25);
    }

    let k = m / 2;
    x[k] = separate_mul(x[k], x[n - k]).conj().scale(2.0);
}

fn accumulate(x: &[Complex<f64>], out: &mut [u64]) -> u64 {
    let mut carry = 0;
    for i in 0..out.len().min(x.len() / 2) {
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

fn fft_core(
    a: &[u64],
    b: &[u64],
    out: &mut [u64],
    n: usize,
    fwd: &dyn Fft<f64>,
    bwd: &dyn Fft<f64>,
    tw: &[Complex<f64>],
    x: &mut [Complex<f64>],
    fft_scratch: &mut [Complex<f64>],
) -> u64 {
    decompose(a, b, x);
    fwd.process_with_scratch(x, fft_scratch);
    recombine(x, n / 2, n, tw);
    bwd.process_with_scratch(&mut x[..n / 2], fft_scratch);
    let scale = 1.0 / n as f64;
    for v in &mut x[..n / 2] {
        *v = v.scale(scale)
    }
    accumulate(&x[..n / 2], out)
}

const CHUNKING_FFT_PROP: f64 = 4.0;

fn fft_chunk_boundary(s: usize) -> usize {
    (CHUNKING_FFT_PROP * (s * s) as f64).ceil() as usize
}

const CHUNKING_FFT_CHUNKING_KARATSUBA_CUTOFF: usize = 500;
const FFT_CHUNKING_KARATSUBA_CUTOFF: usize = 500;
const FFT_KARATSUBA_CUTOFF: usize = 500;

enum FFTDispatch {
    Prim,
    Prim2,
    School,
    Karatsuba,
    FFT,
    ChunkingFFT,
}

fn is_karatsuba(l: usize, s: usize) -> bool {
    const LOG_2_3: f64 = 1.584962501;
    if fft_chunk_boundary(s) <= l {
        s < CHUNKING_FFT_CHUNKING_KARATSUBA_CUTOFF
    } else {
        let half = (l + 1) / 2;
        let fft_cost = (l + s) * ((l + s).ilog2() as usize + 1);
        if s <= half {
            let k_chunk_cost = l * ((s as f64).powf(LOG_2_3).ceil() as usize);
            k_chunk_cost < FFT_CHUNKING_KARATSUBA_CUTOFF * fft_cost
        } else {
            let k_cost = (l as f64).powf(LOG_2_3).ceil() as usize
                - ((l - s) as f64).powf(LOG_2_3).ceil() as usize;
            k_cost < FFT_KARATSUBA_CUTOFF * fft_cost
        }
    }
}

fn fft_dispatch(l: usize, s: usize) -> FFTDispatch {
    if s == 1 {
        FFTDispatch::Prim
    } else if s == 2 {
        FFTDispatch::Prim2
    } else if is_school(l, s) {
        FFTDispatch::School
    } else if is_karatsuba(l, s) {
        FFTDispatch::Karatsuba
    } else {
        if fft_chunk_boundary(s) <= l {
            FFTDispatch::ChunkingFFT
        } else {
            FFTDispatch::FFT
        }
    }
}

fn fft_final(
    long: &[u64],
    short: &[u64],
    out: &mut [u64],
    fft_cache: &mut FFTCache,
    buf_scratch: &mut [u64],
) -> u64 {
    let (val, scratch) = buf_scratch.split_at_mut(long.len() + short.len() - 1);
    let mut overflow = match fft_dispatch(long.len(), short.len()) {
        FFTDispatch::Prim => {
            val[..long.len()].copy_from_slice(long);
            mul_prim(val, short[0])
        }
        FFTDispatch::Prim2 => {
            val[..long.len()].copy_from_slice(long);
            mul_prim2(val, combine_u64(short[0], short[1])) as u64
        }
        FFTDispatch::School => mul_buf(long, short, val),
        FFTDispatch::Karatsuba => karatsuba_mul(long, short, val, scratch),
        FFTDispatch::FFT => fft_mul(long, short, val, fft_cache),
        FFTDispatch::ChunkingFFT => fft_chunking(long, short, val, fft_cache, scratch),
    };
    if add_buf(out, val) {
        overflow += 1;
    }
    overflow
}

fn fft_chunking(
    long: &[u64],
    short: &[u64],
    out: &mut [u64],
    fft_cache: &mut FFTCache,
    buf_scratch: &mut [u64],
) -> u64 {
    out.fill(0);
    let l = long.len();
    let s = short.len();
    let n = find_fft_size(8 * s);
    let l_chunk_sz = n / 8;
    let mut chunk_cnt = l / l_chunk_sz;
    if l % l_chunk_sz == 0 {
        chunk_cnt -= 1;
    }
    {
        let FFTCache {
            planner,
            twidles,
            scratch_pool,
        } = fft_cache;
        let fwd = planner.plan_fft_forward(n);
        let bwd = planner.plan_fft_inverse(n / 2);
        let tw = twidles
            .entry(n >> n.trailing_zeros())
            .or_insert_with(|| TwidleTower::build(n))
            .get(n);
        let [x, fft_scratch] = scratch_pool.get_splits([
            n,
            fwd.get_inplace_scratch_len()
                .max(bwd.get_inplace_scratch_len()),
        ]);

        let val = &mut buf_scratch[..(l_chunk_sz + s)];
        for i in 0..chunk_cnt {
            let of = l_chunk_sz * i;
            fft_core(
                &long[of..of + l_chunk_sz],
                short,
                val,
                n,
                fwd.as_ref(),
                bwd.as_ref(),
                tw,
                x,
                fft_scratch,
            );
            add_buf(&mut out[of..], val);
        }
    }
    let of = chunk_cnt * l_chunk_sz;
    let last_chunk = &long[of..];
    let (last_long, last_short) = if last_chunk.len() > short.len() {
        (last_chunk, short)
    } else {
        (short, last_chunk)
    };
    fft_final(
        last_long,
        last_short,
        &mut out[of..],
        fft_cache,
        buf_scratch,
    )
}

fn u16_buf_len(buf: &[u64]) -> usize {
    let mut len = 4 * buf.len();
    let last = buf.last().copied().unwrap();
    if last >> 48 == 0 {
        len -= 1;
        if last >> 32 == 0 {
            len -= 1;
            if last >> 16 == 0 {
                len -= 1;
            }
        }
    }
    return len;
}

fn fft_mul(long: &[u64], short: &[u64], out: &mut [u64], fft_cache: &mut FFTCache) -> u64 {
    let u16_long_len = u16_buf_len(long);
    let u16_short_len = u16_buf_len(short);
    let n = find_fft_size(u16_long_len + u16_short_len - 1);
    let FFTCache {
        planner,
        twidles,
        scratch_pool,
    } = fft_cache;
    let fwd = planner.plan_fft_forward(n);
    let bwd = planner.plan_fft_inverse(n / 2);
    let tw = twidles
        .entry(n >> n.trailing_zeros())
        .or_insert_with(|| TwidleTower::build(n))
        .get(n);
    let [x, fft_scratch] = scratch_pool.get_splits([
        n,
        fwd.get_inplace_scratch_len()
            .max(bwd.get_inplace_scratch_len()),
    ]);
    fft_core(
        long,
        short,
        out,
        n,
        fwd.as_ref(),
        bwd.as_ref(),
        tw,
        x,
        fft_scratch,
    )
}

fn find_fft_chunking_buf_scratch(l: usize, s: usize) -> usize {
    let n = find_fft_size(8 * s);
    let l_chunk_sz = n / 8;
    let chunk_scratch_size = l_chunk_sz + s;

    let mut chunk_cnt = l / l_chunk_sz;
    if l % l_chunk_sz == 0 {
        chunk_cnt -= 1;
    }
    let l_last = l - chunk_cnt * l_chunk_sz;
    let mut last_chunk_scratch_size = l_last + s - 1;
    match fft_dispatch(l, s) {
        FFTDispatch::Prim | FFTDispatch::Prim2 | FFTDispatch::School | FFTDispatch::FFT => {}
        FFTDispatch::Karatsuba => {
            last_chunk_scratch_size += find_karatsuab_scratch(l_last, s);
        }
        FFTDispatch::ChunkingFFT => {
            last_chunk_scratch_size += find_fft_chunking_buf_scratch(l_last, s);
        }
    };

    return chunk_scratch_size.max(last_chunk_scratch_size);
}

fn fft_entry(long: &[u64], short: &[u64], out: &mut [u64]) -> u64 {
    FFT_CACHE.with(|cell| {
        let fft_cache = &mut *cell.borrow_mut();
        if fft_chunk_boundary(short.len()) < long.len() {
            let scratch_sz = find_fft_chunking_buf_scratch(long.len(), short.len());
            SCRATCH_POOL.with(|cell| {
                let scratch = &mut *cell.borrow_mut();
                fft_chunking(long, short, out, fft_cache, scratch.get(scratch_sz))
            })
        } else {
            fft_mul(long, short, out, fft_cache)
        }
    })
}

//END FFT MULTIPLICATION

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
        out.len() > a.len() + b.len() - 1,
        "out is not large enough for multiplication"
    );
    if a.len() == 0 || b.len() == 0 {
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

pub fn ensure_mul(a: usize, b: usize) {
    let (l, s) = (a.max(b), a.min(b));
    match dyn_dispatch(l, s) {
        DynDispatch::Prim | DynDispatch::Prim2 | DynDispatch::School => {}
        DynDispatch::Karatsuba => SCRATCH_POOL.with(|cell| {
            let scratch_pool = &mut *cell.borrow_mut();
            scratch_pool.ensure(find_karatsuab_scratch(l, s));
        }),
        DynDispatch::FFT => {
            FFT_CACHE.with(|cell| {
                let fft_cache = &mut *cell.borrow_mut();
                fft_cache.scratch_pool.ensure(4 * (a + b));
            });
            if fft_chunk_boundary(s) < l {
                SCRATCH_POOL.with(|cell| {
                    let scratch = &mut *cell.borrow_mut();
                    scratch.ensure(find_fft_chunking_buf_scratch(l, s));
                });
            }
        }
    }
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

fn mul_static<const N: usize>(a: &[u64], b: &[u64], out: &mut [u64]) -> Result<u64, ()> {
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

//SQUARE multiplication simple hierarchy School < Karatsuba < FFT

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

    sub_buf(cross, z0);
    sub_buf(cross, z2);
    if overflow > 0 {
        sub_buf(&mut cross[z2.len()..], &[overflow]);
    }
    if add_buf(&mut out[half..], &cross) {
        overflow += 1;
    }

    return overflow;
}

const KARATSUBA_SQR_CUTOFF: usize = 26;

fn karatsuba_sqr_mul(buf: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    if buf.len() <= KARATSUBA_SQR_CUTOFF {
        return sqr_buf(buf, out);
    }
    let half = (buf.len() + 1) / 2;
    let (cross, rest) = scratch.split_at_mut(2 * half + 1);
    karatsuba_sqr_core(buf, half, out, cross, rest)
}

fn find_karatsuba_sqr_scratch_sz(mut n: usize) -> usize {
    let mut tot = 0;
    while n > KARATSUBA_SQR_CUTOFF {
        let half = (n + 1) / 2;
        tot += 2 * half + 1;
        n = half + 1;
    }
    return tot;
}

fn karatsuba_sqr_entry_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
    let scratch_sz = find_karatsuba_sqr_scratch_sz(buf.len());
    SCRATCH_POOL.with(|cell| {
        let scratch = &mut *cell.borrow_mut();
        let half = (buf.len() + 1) / 2;
        let (cross, rest) = scratch.get(scratch_sz).split_at_mut(2 * half + 1);
        karatsuba_sqr_core(buf, half, out, cross, rest)
    })
}

fn karatsuba_sqr_entry_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> u64 {
    let scratch_sz = find_karatsuba_sqr_scratch_sz(buf.len());
    let half = (buf.len() + 1) / 2;
    let mut scratch = [0; N];
    if scratch_sz > N {
        let mut cross = [0; N];
        karatsuba_sqr_core(buf, half, out, &mut cross, &mut scratch)
    } else {
        let (cross, rest) = scratch.split_at_mut(2 * half + 1);
        karatsuba_sqr_core(buf, half, out, cross, rest)
    }
}

fn sqr_decompose(buf: &[u64], x: &mut [Complex<f64>]) {
    let max_len = x.len() / 2;
    let mask = 0xFFFF;
    for i in 0..max_len {
        let val = buf.get(i).copied().unwrap();
        let idx = 2 * i;
        x[idx] = Complex {
            re: (val & mask) as f64,
            im: (val >> 16 & mask) as f64,
        };
        x[idx + 1] = Complex {
            re: (val >> 32 & mask) as f64,
            im: (val >> 48) as f64,
        };
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
        x[k] = (s + Complex::<f64>::i() * t).scale(2.0);
        x[m - k] = (s.conj() + Complex::<f64>::i() * t.conj()).scale(2.0);
    }
    x[m / 2] = c_sqr(x[m / 2]);
}

fn fft_sqr_core(
    buf: &[u64],
    out: &mut [u64],
    m: usize,
    fwd: &dyn Fft<f64>,
    bwd: &dyn Fft<f64>,
    tw: &[Complex<f64>],
    x: &mut [Complex<f64>],
    fft_scratch: &mut [Complex<f64>],
) -> u64 {
    sqr_decompose(buf, x);
    fwd.process_with_scratch(x, fft_scratch);
    sqr_recombine(x, m, tw);
    bwd.process_with_scratch(x, fft_scratch);
    let scale = 1.0 / m as f64;
    for v in x.into_iter() {
        *v = v.scale(scale)
    }
    accumulate(x, out)
}

fn fft_sqr_entry(buf: &[u64], out: &mut [u64]) -> u64 {
    let u16_len = u16_buf_len(buf);
    let m = find_fft_size(2 * u16_len - 1) / 2;
    FFT_CACHE.with(|cell| {
        let FFTCache {
            planner,
            twidles,
            scratch_pool,
        } = &mut *cell.borrow_mut();
        let fwd = planner.plan_fft_forward(m);
        let bwd = planner.plan_fft_inverse(m);
        let [x, fft_scratch] = scratch_pool.get_splits([
            m,
            fwd.get_inplace_scratch_len()
                .max(bwd.get_inplace_scratch_len()),
        ]);
        let tw = twidles
            .entry(m >> m.trailing_zeros())
            .or_insert_with(|| TwidleTower::build(m))
            .get(m);
        fft_sqr_core(buf, out, m, fwd.as_ref(), bwd.as_ref(), tw, x, fft_scratch)
    })
}

enum DynSqrDispatch {
    School,
    Karatsuba,
    FFT,
}

const FFT_SQR_CUTOFF: usize = 500;

fn dyn_sqr_dispatch(n: usize) -> DynSqrDispatch {
    if n <= KARATSUBA_SQR_CUTOFF {
        DynSqrDispatch::School
    } else if n <= FFT_SQR_CUTOFF {
        DynSqrDispatch::Karatsuba
    } else {
        DynSqrDispatch::FFT
    }
}

fn sqr_dyn(buf: &[u64], out: &mut [u64]) -> u64 {
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

fn ensure_sqr(n: usize) {
    match dyn_sqr_dispatch(n) {
        DynSqrDispatch::School => {}
        DynSqrDispatch::Karatsuba => SCRATCH_POOL.with(|cell| {
            let scratch_pool = &mut *cell.borrow_mut();
            scratch_pool.ensure(find_karatsuba_sqr_scratch_sz(n));
        }),
        DynSqrDispatch::FFT => FFT_CACHE.with(|cell| {
            let fft_cache = &mut *cell.borrow_mut();
            fft_cache.scratch_pool.ensure(8 * n);
        }),
    }
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

fn sqr_static<const N: usize>(buf: &[u64], out: &mut [u64]) -> Result<u64, ()> {
    if 2 * buf.len() - 1 > N {
        return Err(());
    }
    Ok(match static_sqr_dispatch(buf.len()) {
        StaticSqrDispatch::School => sqr_buf(buf, out),
        StaticSqrDispatch::Karatsubaa => karatsuba_sqr_entry_static::<N>(buf, out),
    })
}

pub fn get_powi_size(n: usize, last: u64, pow: usize) -> usize {
    let log2 = 1 + last.ilog2() as usize + (n - 1) << 6;
    (log2 * pow + 63) >> 6
}

pub fn powi_vec(buf: &[u64], pow: usize) -> Vec<u64> {
    if pow == 0 {
        return vec![1];
    }
    let out_sz = get_powi_size(buf.len(), buf.last().copied().unwrap(), pow);
    let mut out = vec![0_u64; out_sz];
    let mut tmp = vec![0_u64; out_sz];
    ensure_sqr(1 + out_sz / 2);
    ensure_mul(out_sz, buf.len());

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

        let sqr_c = sqr_dyn(&src[..len], &mut dst[..sqr_len]);
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

            let mul_c = mul_dyn(&src[..len], &buf, &mut dst[..mul_len]);
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
    let out_sz = get_powi_size(buf.len(), buf.last().copied().unwrap(), pow);
    if out_sz > N {
        return Err(());
    }

    let mut out = [0; N];
    let mut tmp = [0; N];

    let buf_len = buf_len(buf);
    let mut len = buf_len;
    out[..len].copy_from_slice(&buf[..len]);
    let mut io = true;

    let log = pow.ilog2();
    for i in (0..log).rev() {
        let sqr_len = 2 * len - 1;
        let (src, dst): (&[u64], &mut [u64]) = if io {
            (&out, &mut tmp)
        } else {
            (&tmp, &mut out)
        };

        let sqr_c = sqr_static::<N>(&src[..len], &mut dst[..sqr_len]).unwrap();
        len = sqr_len;
        if sqr_c > 0 {
            if len == N {
                return Err(());
            }
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

            let mul_c = mul_static::<N>(&src[..len], &buf[..buf_len], &mut dst[..mul_len]).unwrap();
            len = mul_len;
            if mul_c > 0 {
                if len == N {
                    return Err(());
                }
                dst[len] = mul_c;
                len += 1
            }
            io = !io;
        }
    }

    if !io {
        out.copy_from_slice(&tmp);
    }
    return Ok(out);
}
