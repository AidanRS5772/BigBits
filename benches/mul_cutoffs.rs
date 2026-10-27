// Finds multiplication's algorithm crossovers, one family of shapes per process.
//
//   cargo bench-mul-cutoffs <family> [--seed S] [--rounds 60] [--measure-us 100] [--dry | --check]
//
// Families and the two sides each one times (lower side first):
//   prim    Prim / Prim2 arm vs mul_buf, s = 1 and 2 (dyn_dispatch's fixed tiers)
//   chunk   mul_buf vs chunking_karatsuba, s <= half (CHUNKING_KARATSUBA_CUTOFF)
//   kara    mul_buf vs one Karatsuba level, s > half (KARATSUBA_CUTOFF)
//   kfft    karatsuba_entry_dyn vs fft_entry (FFT_KARATSUBA_CUTOFF, FFT_CHUNKING_...)
//   kntt    karatsuba_entry_static vs ntt_entry_static (NTT_KARATSUBA_CUTOFF, NTT_CHUNKING_...)
//   ksqr    sqr_buf vs one Karatsuba square level (KARATSUBA_SQR_CUTOFF)
//   sqrfft  sqr_buf and karatsuba_sqr_entry_dyn vs fft_sqr_entry (FFT_SQR_CUTOFF)
//   sqrntt  karatsuba_sqr_entry_static vs ntt_sqr_entry_static (STATIC_NTT_SQR_CUTOFF)
//   hi      hi_mul_buf vs the guarded truncated product (PARTIAL_MUL_CUTOFF)
//   hisqr   hi_sqr_buf vs the guarded truncated square (PARTIAL_SQR_CUTOFF)
//   hirows  hi_*_buf by columns vs by rows (HI_MUL_ROW_CUTOFF, HI_SQR_ROW_CUTOFF)
//   mid     mid_mul_buf vs fft_mid_mul (FFT_MID_CUTOFF)
//   midntt  mid_mul_buf vs ntt_mid_mul_static (NTT_MID_CUTOFF)
//
// --dry prints the shapes without timing them; --check runs both sides once per input
// and compares them, also without timing.
//
// Per shape: up to 32 random inputs at random offsets within a cache line, then
// `rounds` rounds that each time both sides back to back in random order. Each side
// writes its own outputs, which are compared afterwards. Prints one CSV line per shape:
// family, path, l, s, flip, the median over rounds of ln(t_upper / t_lower), and the two
// trimmed mean times in ns. `flip` is the value of the shape's dispatch parameter at
// which it changes sides: the size for 1-D cutoffs, KARATSUBA_CUTOFF's c for kara, and
// the Karatsuba / transform cost ratio for kfft and kntt. benches/mul_cutoffs.py runs
// the processes and tests the results; see docs/mul_cutoff_tuning.md.
//
// Static entries run at N = the smallest 5-smooth bucket covering the call's output,
// scratch and NTT size, so their [0; N] buffers cost what a fitting N would. Forced
// static Karatsuba levels (chunk, kara) zero a scratch of that N and then run the
// same kernels, as karatsuba_entry_static does.

use big_bits::utils::{mul::*, utils::*, *};
use std::{env, hint::black_box, time::Instant};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(0x9E3779B97F4A7C15 ^ seed.wrapping_mul(0xD1B54A32D192ED03));
        for _ in 0..16 {
            r.next();
        }
        r
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

// len limbs at a random 0..8 limb offset into their own allocation.
struct Buf {
    v: Vec<u64>,
    off: usize,
    len: usize,
}

impl Buf {
    fn rand(len: usize, rng: &mut Rng) -> Self {
        let off = (rng.next() % 8) as usize;
        let mut v = vec![0u64; len + 8];
        for x in &mut v[off..off + len] {
            *x = rng.next();
        }
        // Operands are trimmed, as every caller's are.
        if len > 0 {
            v[off + len - 1] |= 1;
        }
        Buf { v, off, len }
    }

    fn zero(len: usize, rng: &mut Rng) -> Self {
        let off = (rng.next() % 8) as usize;
        Buf {
            v: vec![0u64; len + 8],
            off,
            len,
        }
    }

    fn s(&self) -> &[u64] {
        &self.v[self.off..self.off + self.len]
    }

    fn m(&mut self) -> &mut [u64] {
        &mut self.v[self.off..self.off + self.len]
    }
}

// One side's outputs: an output buffer and up to two returned limbs per input.
struct Side {
    outs: Vec<Buf>,
    rets: Vec<[u64; 2]>,
}

impl Side {
    fn new(inputs: usize, out_len: usize, rng: &mut Rng) -> Self {
        Side {
            outs: (0..inputs).map(|_| Buf::zero(out_len, rng)).collect(),
            rets: vec![[0; 2]; inputs],
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Check {
    Exact,
    // The two (ret, out) values may differ by this many units of out's first limb.
    Ulps(u128),
}

struct Cfg {
    family: String,
    rounds: usize,
    measure_ns: f64,
    dry: bool,
    check: bool,
}

struct Shape<'a> {
    path: &'a str,
    l: usize,
    s: usize,
    flip: f64,
}

// Inputs for a shape: few enough that one pass of the lower side takes about
// 200 us, between 2 and 32 of them. `work` estimates one call's limb products.
fn input_count(work: usize) -> usize {
    (6_400_000 / work.max(1)).clamp(2, 32)
}

fn trimmed_mean(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.truncate((v.len() * 9 / 10).max(1));
    v.iter().sum::<f64>() / v.len() as f64
}

fn diff_ulps(a: &[u64], ra: [u64; 2], b: &[u64], rb: [u64; 2]) -> u128 {
    // (ret, out) as one number each; their difference if it fits two limbs, or MAX.
    let mut x: Vec<u64> = a.to_vec();
    x.extend(ra);
    let mut y: Vec<u64> = b.to_vec();
    y.extend(rb);
    if cmp_buf(&x, &y) == std::cmp::Ordering::Less {
        std::mem::swap(&mut x, &mut y);
    }
    sub_buf(&mut x, &y);
    if x.len() > 2 && x[2..].iter().any(|&l| l != 0) {
        return u128::MAX;
    }
    x[0] as u128 | (x.get(1).copied().unwrap_or(0) as u128) << 64
}

fn compare(cfg: &Cfg, sh: &Shape, lo: &Side, up: &Side, check: Check) {
    for (i, (a, b)) in lo.outs.iter().zip(&up.outs).enumerate() {
        match check {
            Check::Exact => {
                assert!(
                    a.s() == b.s() && lo.rets[i] == up.rets[i],
                    "{} {} l={} s={}: outputs differ",
                    cfg.family,
                    sh.path,
                    sh.l,
                    sh.s
                );
            }
            Check::Ulps(tol) => {
                let d = diff_ulps(a.s(), lo.rets[i], b.s(), up.rets[i]);
                assert!(
                    d <= tol,
                    "{} {} l={} s={}: outputs differ by {d} ulps",
                    cfg.family,
                    sh.path,
                    sh.l,
                    sh.s
                );
            }
        }
    }
}

// Times lo and up (each runs input i on its own outputs) and prints the shape's line.
fn measure<L: FnMut(usize), U: FnMut(usize)>(
    cfg: &Cfg,
    rng: &mut Rng,
    sh: &Shape,
    inputs: usize,
    mut lo: L,
    mut up: U,
) {
    if cfg.dry {
        println!("{},{},{},{},{:.6},0,0,0", cfg.family, sh.path, sh.l, sh.s, sh.flip);
        return;
    }
    if cfg.check {
        for i in 0..inputs {
            lo(i);
            up(i);
        }
        return;
    }
    let mut pass = |k: usize| {
        for i in 0..inputs {
            if k == 0 {
                lo(i)
            } else {
                up(i)
            }
        }
    };

    // Warm both sides, then size each round's batch to at least measure_ns.
    let mut per = 0f64;
    for k in 0..2 {
        let t0 = Instant::now();
        pass(k);
        let first = t0.elapsed().as_nanos() as f64;
        let reps = ((2_000_000.0 / first.max(1.0)) as usize).clamp(2, 20);
        for _ in 0..reps {
            pass(k);
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            pass(k);
        }
        per = per.max(t0.elapsed().as_nanos() as f64 / reps as f64);
    }
    let passes = ((cfg.measure_ns / per).ceil() as usize).max(1);
    let calls = (passes * inputs) as f64;

    let (mut tl, mut tu, mut lr) = (vec![], vec![], vec![]);
    for _ in 0..cfg.rounds {
        let mut t = [0f64; 2];
        let order = if rng.next() & 1 == 0 { [0, 1] } else { [1, 0] };
        for k in order {
            let t0 = Instant::now();
            for _ in 0..passes {
                pass(k);
            }
            t[k] = t0.elapsed().as_nanos() as f64 / calls;
        }
        tl.push(t[0]);
        tu.push(t[1]);
        lr.push((t[1] / t[0]).ln());
    }
    lr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = lr.len();
    let med = if n % 2 == 0 {
        (lr[n / 2 - 1] + lr[n / 2]) / 2.0
    } else {
        lr[n / 2]
    };
    println!(
        "{},{},{},{},{:.6},{:.6},{:.2},{:.2}",
        cfg.family,
        sh.path,
        sh.l,
        sh.s,
        sh.flip,
        med,
        trimmed_mean(tl),
        trimmed_mean(tu)
    );
}

// Runs a two-operand shape: inputs are (long[l], short[s]) and each side writes
// out_len limbs, returning up to two limbs.
fn pair<L, U>(
    cfg: &Cfg,
    rng: &mut Rng,
    sh: Shape,
    out_len: usize,
    check: Check,
    work: usize,
    mut lo: L,
    mut up: U,
) where
    L: FnMut(&[u64], &[u64], &mut [u64]) -> [u64; 2],
    U: FnMut(&[u64], &[u64], &mut [u64]) -> [u64; 2],
{
    let inputs = input_count(work);
    let ins: Vec<(Buf, Buf)> = (0..inputs)
        .map(|_| (Buf::rand(sh.l, rng), Buf::rand(sh.s, rng)))
        .collect();
    let mut sl = Side::new(inputs, out_len, rng);
    let mut su = Side::new(inputs, out_len, rng);
    {
        let (sl, su) = (&mut sl, &mut su);
        let ins = &ins;
        measure(
            cfg,
            &mut Rng(rng.next() | 1),
            &sh,
            inputs,
            |i| {
                let r = lo(
                    black_box(ins[i].0.s()),
                    black_box(ins[i].1.s()),
                    black_box(sl.outs[i].m()),
                );
                sl.rets[i] = black_box(r);
            },
            |i| {
                let r = up(
                    black_box(ins[i].0.s()),
                    black_box(ins[i].1.s()),
                    black_box(su.outs[i].m()),
                );
                su.rets[i] = black_box(r);
            },
        );
    }
    if !cfg.dry {
        compare(cfg, &sh, &sl, &su, check);
    }
}

// A one-operand shape (squares): inputs are x[l].
fn single<L, U>(
    cfg: &Cfg,
    rng: &mut Rng,
    sh: Shape,
    out_len: usize,
    check: Check,
    work: usize,
    mut lo: L,
    mut up: U,
) where
    L: FnMut(&[u64], &mut [u64]) -> [u64; 2],
    U: FnMut(&[u64], &mut [u64]) -> [u64; 2],
{
    pair(cfg, rng, Shape { s: 0, ..sh }, out_len, check, work, |x, _, o| lo(x, o), |x, _, o| up(x, o))
}

fn r1(x: u64) -> [u64; 2] {
    [x, 0]
}

fn r2((a, b): (u64, u64)) -> [u64; 2] {
    [a, b]
}

// 5-smooth buckets about 8% apart, for static N. Each call site names a range
// (small: 8..600, mid: 30..2430, large: 300..9000, big: 300..65536), so that it instantiates its
// generic entries only for the buckets its shapes can need.
macro_rules! with_n {
    ($need:expr, small, |$N:ident| $body:expr) => {
        with_n!(@arms $need, $N, $body, [
            8, 9, 10, 12, 15, 18, 20, 24, 27, 30, 36, 40, 45, 50, 54, 60, 72, 80, 90, 100,
            108, 120, 135, 150, 162, 180, 200, 216, 240, 270, 300, 324, 360, 400, 432, 480,
            540, 600
        ])
    };
    ($need:expr, mid, |$N:ident| $body:expr) => {
        with_n!(@arms $need, $N, $body, [
            30, 36, 40, 45, 50, 54, 60, 72, 80, 90, 100, 108, 120, 135, 150, 162, 180, 200,
            216, 240, 270, 300, 324, 360, 400, 432, 480, 540, 600, 648, 720, 800, 864, 960,
            1080, 1200, 1296, 1440, 1600, 1728, 1875, 2048, 2250, 2430
        ])
    };
    ($need:expr, large, |$N:ident| $body:expr) => {
        with_n!(@arms $need, $N, $body, [
            300, 324, 360, 400, 432, 480, 540, 600, 648, 720, 800, 864, 960, 1080, 1200,
            1296, 1440, 1600, 1728, 1875, 2048, 2250, 2430, 2700, 2916, 3200, 3456, 3750,
            4096, 4500, 4860, 5400, 5832, 6400, 6912, 7500, 8192, 9000
        ])
    };
    ($need:expr, big, |$N:ident| $body:expr) => {
        with_n!(@arms $need, $N, $body, [
            300, 324, 360, 400, 432, 480, 540, 600, 648, 720, 800, 864, 960, 1080, 1200,
            1296, 1440, 1600, 1728, 1875, 2048, 2250, 2430, 2700, 2916, 3200, 3456, 3750,
            4096, 4500, 4860, 5400, 5832, 6400, 6912, 7500, 8192, 9000, 9720, 10800, 11664,
            12800, 13824, 15000, 16384, 18000, 19440, 21600, 23328, 25600, 27648, 30000,
            32768, 36000, 38880, 43200, 46656, 50625, 54675, 60000, 65536
        ])
    };
    (@arms $need:expr, $N:ident, $body:expr, [$($n:literal),*]) => {{
        let need: usize = $need;
        $(
            if need <= $n {
                const $N: usize = $n;
                $body
            } else
        )*
        {
            panic!("no static N bucket holds {need} limbs")
        }
    }};
}

fn bucket(need: usize) -> usize {
    if need <= 600 {
        with_n!(need, small, |N| N)
    } else {
        with_n!(need, big, |N| N)
    }
}

const HUGE_N: usize = 65536;

// The flip values of the size-only families.
fn sizes(ranges: &[(usize, usize, usize)]) -> Vec<usize> {
    let mut v = vec![];
    for &(a, b, step) in ranges {
        v.extend((a..=b).step_by(step));
    }
    v.sort();
    v.dedup();
    v
}

// Sizes from a to b about `pct` percent apart.
fn geo(a: usize, b: usize, pct: f64) -> Vec<usize> {
    let mut v = vec![];
    let mut x = a as f64;
    while x <= b as f64 {
        let n = x.round() as usize;
        if v.last() != Some(&n) {
            v.push(n);
        }
        x *= 1.0 + pct / 100.0;
    }
    v
}

// --- prim -------------------------------------------------------------------

fn fam_prim(cfg: &Cfg, rng: &mut Rng) {
    for s in [1usize, 2] {
        for l in sizes(&[(3, 64, 1), (72, 256, 8)]) {
            let sh = Shape {
                path: if s == 1 { "s1" } else { "s2" },
                l,
                s,
                flip: l as f64,
            };
            let out_len = l + s - 1;
            // The Prim and Prim2 arms of mul_dyn.
            let prim = |a: &[u64], b: &[u64], o: &mut [u64]| {
                o[..a.len()].copy_from_slice(a);
                if b.len() == 1 {
                    r1(mul_prim(o, b[0]))
                } else {
                    o[a.len()] = 0;
                    r1(mul_prim2(o, combine_u64(b[0], b[1])) as u64)
                }
            };
            pair(cfg, rng, sh, out_len, Check::Exact, l * s, prim, |a, b, o| r1(mul_buf(a, b, o)));
        }
    }
}

// --- chunk / kara -----------------------------------------------------------

// chunking_karatsuba's scratch for any (l, s), including shapes is_school keeps.
fn chunk_scratch(l: usize, s: usize) -> usize {
    let half = (s + 1) / 2;
    let phase1 = 2 * s + 2 * half + 1 + find_karatsuba_scratch(half + 1, half + 1);
    let chunks = (l - s + 1) / s;
    let last = l - chunks * s;
    let phase2 = (last + s - 1) + find_karatsuba_scratch(last, s);
    phase1.max(phase2)
}

// karatsuba_core's scratch at its top level for any l >= s > half.
fn kara_scratch(l: usize, s: usize) -> usize {
    let half = (l + 1) / 2;
    let inner = find_karatsuba_scratch(half + 1, half + 1).max(find_karatsuba_scratch(l - half, s - half));
    2 * half + 1 + inner
}

fn fam_chunk(cfg: &Cfg, rng: &mut Rng) {
    let mut zeros = vec![0u64; 1 << 16];
    for r in [2usize, 4, 16] {
        for s in sizes(&[(8, 64, 1), (68, 128, 4)]) {
            let l = r * s;
            let out_len = l + s - 1;
            let sz = chunk_scratch(l, s);
            let label = format!("r{r}");
            let sh = || Shape {
                path: &label,
                l,
                s,
                flip: s as f64,
            };
            pair(cfg, rng, sh(), out_len, Check::Exact, l * s, |a, b, o| r1(mul_buf(a, b, o)), |a, b, o| {
                let mut g = ScratchGuard::acquire();
                r1(chunking_karatsuba(a, b, o, g.get(sz)))
            });
            let n = bucket(out_len.max(sz));
            let label_s = format!("static_r{r}");
            let sh = Shape {
                path: &label_s,
                l,
                s,
                flip: s as f64,
            };
            let zeros = &mut zeros;
            pair(cfg, rng, sh, out_len, Check::Exact, l * s, |a, b, o| r1(mul_buf(a, b, o)), |a, b, o| {
                let scratch = black_box(&mut zeros[..n]);
                scratch.fill(0);
                r1(chunking_karatsuba(a, b, o, &mut scratch[..sz]))
            });
        }
    }
}

// The smallest KARATSUBA_CUTOFF that keeps (l, s) schoolbook, by bisection:
// is_school's region only grows with c.
fn kara_flip(l: usize, s: usize) -> f64 {
    let (mut lo, mut hi) = (0.5f64, 4096.0f64);
    if !is_school_at(l, s, 0, hi) {
        return f64::INFINITY;
    }
    for _ in 0..60 {
        let m = (lo + hi) / 2.0;
        if is_school_at(l, s, 0, m) {
            hi = m;
        } else {
            lo = m;
        }
    }
    hi
}

fn fam_kara(cfg: &Cfg, rng: &mut Rng) {
    let mut zeros = vec![0u64; 1 << 16];
    for l in 6..=240usize {
        let step = if l <= 30 {
            1
        } else if l <= 72 {
            2
        } else {
            3
        };
        if l % step != 0 {
            continue;
        }
        for s in ((l + 1) / 2 + 1)..=l {
            if s % step != 0 {
                continue;
            }
            let flip = kara_flip(l, s);
            if !(6.0..=90.0).contains(&flip) {
                continue;
            }
            let out_len = l + s - 1;
            let half = (l + 1) / 2;
            let sz = kara_scratch(l, s);
            let one_level = |a: &[u64], b: &[u64], o: &mut [u64], scratch: &mut [u64]| {
                let (cross, rest) = scratch.split_at_mut(2 * half + 1);
                r1(karatsuba_core(a, b, half, o, cross, rest))
            };
            let sh = Shape { path: "dyn", l, s, flip };
            pair(cfg, rng, sh, out_len, Check::Exact, l * s, |a, b, o| r1(mul_buf(a, b, o)), |a, b, o| {
                let mut g = ScratchGuard::acquire();
                one_level(a, b, o, g.get(sz))
            });
            let n = bucket(out_len.max(sz));
            let sh = Shape { path: "static", l, s, flip };
            let zeros = &mut zeros;
            pair(cfg, rng, sh, out_len, Check::Exact, l * s, |a, b, o| r1(mul_buf(a, b, o)), |a, b, o| {
                let scratch = black_box(&mut zeros[..n]);
                scratch.fill(0);
                one_level(a, b, o, &mut scratch[..sz])
            });
        }
    }
}

// --- kfft / kntt ------------------------------------------------------------

// Shapes around a Karatsuba / transform boundary: the regular region (s > half,
// flip = karatsuba_cost / fft_cost) and the chunking one (flip =
// chunking_karatsuba_cost / fft_cost), within the given flip ranges.
fn transform_shapes(
    reg: (f64, f64),
    fracs: &[f64],
    chunk: (f64, f64),
    ratios: &[usize],
    max_out: usize,
    pct: f64,
) -> Vec<(&'static str, usize, usize, f64)> {
    let mut v = vec![];
    for l in geo(48, 65536, pct) {
        for &frac in fracs {
            let s = ((l as f64 * frac).round() as usize).clamp((l + 1) / 2 + 1, l);
            if l + s - 1 > max_out || is_school(l, s) {
                continue;
            }
            let f = karatsuba_cost(l, s) / fft_cost(l, s);
            if (reg.0..=reg.1).contains(&f) {
                v.push(("reg", l, s, f));
            }
        }
    }
    for s in geo(8, 32768, pct) {
        for &r in ratios {
            let l = r * s;
            if l + s - 1 > max_out || is_school(l, s) {
                continue;
            }
            let f = chunking_karatsuba_cost(l, s) / fft_cost(l, s);
            if (chunk.0..=chunk.1).contains(&f) {
                v.push(("chunk", l, s, f));
            }
        }
    }
    v.sort_by_key(|&(p, l, s, _)| (p, l, s));
    v.dedup_by_key(|x| (x.0, x.1, x.2));
    v
}

fn fam_kfft(cfg: &Cfg, rng: &mut Rng) {
    for (path, l, s, flip) in
        transform_shapes((1.4, 4.0), &[0.55, 0.75, 1.0], (1.3, 3.6), &[2, 4, 32], FFT_16BIT_CUTOFF, 5.0)
    {
        let sh = Shape { path, l, s, flip };
        pair(
            cfg,
            rng,
            sh,
            l + s - 1,
            Check::Exact,
            l * s / 4,
            |a, b, o| r1(karatsuba_entry_dyn(a, b, o)),
            |a, b, o| r1(fft_entry(a, b, o)),
        );
    }
}

fn fam_kntt(cfg: &Cfg, rng: &mut Rng) {
    for (path, l, s, flip) in transform_shapes((2.4, 10.0), &[0.7, 1.0], (2.2, 9.0), &[2, 4], 1 << 15, 6.0) {
        let sh = Shape { path, l, s, flip };
        let out_len = l + s - 1;
        let need = out_len.max(find_karatsuba_scratch(l, s)).max(find_max_ntt_size(out_len));
        if need > HUGE_N {
            continue;
        }
        with_n!(need, big, |N| pair(
            cfg,
            rng,
            sh,
            out_len,
            Check::Exact,
            l * s / 4,
            |a, b, o| r1(karatsuba_entry_static::<N>(a, b, o)),
            |a, b, o| r1(ntt_entry_static::<N>(a, b, o)),
        ));
    }
}

// --- squares ----------------------------------------------------------------

fn fam_ksqr(cfg: &Cfg, rng: &mut Rng) {
    // From 6: karatsuba_sqr_core's cross term doesn't fit out at n = 3 or 5.
    for n in sizes(&[(6, 96, 1), (100, 192, 4)]) {
        let out_len = 2 * n - 1;
        let sh = Shape { path: "dyn", l: n, s: 0, flip: n as f64 };
        single(cfg, rng, sh, out_len, Check::Exact, n * n / 2, |x, o| r1(sqr_buf(x, o)), |x, o| {
            r1(karatsuba_sqr_entry_dyn(x, o))
        });
        let half = (n + 1) / 2;
        let need = out_len.max(2 * half + 1 + find_karatsuba_sqr_scratch_sz(half + 1));
        let sh = Shape { path: "static", l: n, s: 0, flip: n as f64 };
        with_n!(need, small, |N| single(cfg, rng, sh, out_len, Check::Exact, n * n / 2, |x, o| r1(sqr_buf(x, o)), |x, o| {
            r1(karatsuba_sqr_entry_static::<N>(x, o))
        }));
    }
}

fn fam_sqrfft(cfg: &Cfg, rng: &mut Rng) {
    for n in sizes(&[(64, 512, 4), (528, 1024, 16)]) {
        let out_len = 2 * n - 1;
        let sh = Shape { path: "school", l: n, s: 0, flip: n as f64 };
        single(cfg, rng, sh, out_len, Check::Exact, n * n / 2, |x, o| r1(sqr_buf(x, o)), |x, o| {
            r1(fft_sqr_entry(x, o))
        });
        let sh = Shape { path: "kara", l: n, s: 0, flip: n as f64 };
        single(cfg, rng, sh, out_len, Check::Exact, n * n / 4, |x, o| r1(karatsuba_sqr_entry_dyn(x, o)), |x, o| {
            r1(fft_sqr_entry(x, o))
        });
    }
}

fn fam_sqrntt(cfg: &Cfg, rng: &mut Rng) {
    for n in geo(384, 12288, 4.0) {
        let out_len = 2 * n - 1;
        let need = out_len.max(find_karatsuba_sqr_scratch_sz(n)).max(find_max_ntt_size(out_len));
        if need > HUGE_N {
            continue;
        }
        let sh = Shape { path: "static", l: n, s: 0, flip: n as f64 };
        with_n!(need, big, |N| single(
            cfg,
            rng,
            sh,
            out_len,
            Check::Exact,
            n * n / 4,
            |x, o| r1(karatsuba_sqr_entry_static::<N>(x, o)),
            |x, o| r1(ntt_sqr_entry_static::<N>(x, o)),
        ));
    }
}

// --- high products ----------------------------------------------------------

fn hi_windows() -> Vec<usize> {
    sizes(&[(32, 512, 8), (544, 1024, 32), (1152, 2048, 128)])
}

fn fam_hi(cfg: &Cfg, rng: &mut Rng) {
    let tol = Check::Ulps(64);
    for k in hi_windows() {
        for (path, w) in [("dyn_w1", k), ("dyn_w2", 2 * k)] {
            let sh = Shape { path, l: w, s: w, flip: k as f64 };
            pair(cfg, rng, sh, k, tol, k * k / 2, |a, b, o| r1(hi_mul_buf(a, b, o)), |a, b, o| {
                r1(hi_mul_guarded_dyn(a, b, o))
            });
        }
        // Room for the guarded product's 2k + 4 limbs, or operands that fill N.
        let w = 2 * k;
        let sh = Shape { path: "static_roomy", l: w, s: w, flip: k as f64 };
        with_n!(2 * k + 4, large, |N| pair(cfg, rng, sh, k, tol, k * k / 2, |a, b, o| r1(hi_mul_buf(a, b, o)), |a, b, o| {
            r1(hi_mul_guarded_static::<N>(a, b, o))
        }));
        let w = bucket(2 * k);
        let sh = Shape { path: "static_tight", l: w, s: w, flip: k as f64 };
        with_n!(w, large, |N| pair(cfg, rng, sh, k, tol, k * k / 2, |a, b, o| r1(hi_mul_buf(a, b, o)), |a, b, o| {
            r1(hi_mul_guarded_static::<N>(a, b, o))
        }));
    }
}

fn fam_hisqr(cfg: &Cfg, rng: &mut Rng) {
    let tol = Check::Ulps(64);
    for k in hi_windows() {
        for (path, w) in [("dyn_w1", k), ("dyn_w2", 2 * k)] {
            let sh = Shape { path, l: w, s: 0, flip: k as f64 };
            single(cfg, rng, sh, k, tol, k * k / 4, |x, o| r1(hi_sqr_buf(x, o)), |x, o| {
                r1(hi_sqr_guarded_dyn(x, o))
            });
        }
        let w = 2 * k;
        let sh = Shape { path: "static_roomy", l: w, s: 0, flip: k as f64 };
        with_n!(2 * k + 4, large, |N| single(cfg, rng, sh, k, tol, k * k / 4, |x, o| r1(hi_sqr_buf(x, o)), |x, o| {
            r1(hi_sqr_guarded_static::<N>(x, o))
        }));
        let w = bucket(2 * k);
        let sh = Shape { path: "static_tight", l: w, s: 0, flip: k as f64 };
        with_n!(w, large, |N| single(cfg, rng, sh, k, tol, k * k / 4, |x, o| r1(hi_sqr_buf(x, o)), |x, o| {
            r1(hi_sqr_guarded_static::<N>(x, o))
        }));
    }
}

fn fam_hirows(cfg: &Cfg, rng: &mut Rng) {
    let ws = sizes(&[(1, 128, 1), (136, 256, 8)]);
    for &w in &ws {
        let sh = Shape { path: "mul", l: w, s: w, flip: w as f64 };
        pair(cfg, rng, sh, w, Check::Exact, w * w / 2, |a, b, o| r1(hi_mul_buf_cols(a, b, o)), |a, b, o| {
            r1(hi_mul_buf_by_rows(a, b, o))
        });
    }
    for &w in &ws {
        let sh = Shape { path: "sqr", l: w, s: 0, flip: w as f64 };
        single(cfg, rng, sh, w, Check::Exact, w * w / 4, |x, o| r1(hi_sqr_buf_cols(x, o)), |x, o| {
            r1(hi_sqr_buf_by_rows(x, o))
        });
    }
}

// --- middle products --------------------------------------------------------

// The transform middle products drop the carries from the columns below the band,
// up to about (n + 1) B at its base, where mid_mul_buf adds the column just below.
fn mid_tol(n: usize) -> Check {
    Check::Ulps(((n as u128) + 2) << 64)
}

fn fam_mid(cfg: &Cfg, rng: &mut Rng) {
    for n in sizes(&[(32, 160, 1), (168, 320, 8)]) {
        let sh = Shape { path: "dyn", l: 2 * n - 1, s: n, flip: n as f64 };
        pair(cfg, rng, sh, n, mid_tol(n), n * n, |a, b, o| r2(mid_mul_buf(a, b, o)), |a, b, o| {
            r2(fft_mid_mul(a, b, o))
        });
    }
}

fn fam_midntt(cfg: &Cfg, rng: &mut Rng) {
    for n in geo(256, 3072, 2.5) {
        let long = 2 * n - 1;
        let need = long.max(find_max_ntt_size(long));
        let sh = Shape { path: "static", l: long, s: n, flip: n as f64 };
        with_n!(need, large, |N| pair(cfg, rng, sh, n, mid_tol(n), n * n, |a, b, o| r2(mid_mul_buf(a, b, o)), |a, b, o| {
            r2(ntt_mid_mul_static::<N>(a, b, o))
        }));
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).filter(|a| a != "--bench").collect();
    let family = args.first().cloned().unwrap_or_default();
    let opt = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
    };
    let seed: u64 = opt("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let cfg = Cfg {
        family: family.clone(),
        rounds: opt("--rounds").and_then(|s| s.parse().ok()).unwrap_or(60),
        measure_ns: opt("--measure-us").and_then(|s| s.parse::<f64>().ok()).unwrap_or(100.0) * 1000.0,
        dry: args.iter().any(|a| a == "--dry"),
        check: args.iter().any(|a| a == "--check"),
    };
    // The static entries put several [u64; N] buffers on the stack, up to N = 65536.
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(move || run(&family, &cfg, seed))
        .unwrap()
        .join()
        .unwrap();
}

fn run(family: &str, cfg: &Cfg, seed: u64) {
    let mut rng = Rng::new(seed);
    match family {
        "prim" => fam_prim(&cfg, &mut rng),
        "chunk" => fam_chunk(&cfg, &mut rng),
        "kara" => fam_kara(&cfg, &mut rng),
        "kfft" => fam_kfft(&cfg, &mut rng),
        "kntt" => fam_kntt(&cfg, &mut rng),
        "ksqr" => fam_ksqr(&cfg, &mut rng),
        "sqrfft" => fam_sqrfft(&cfg, &mut rng),
        "sqrntt" => fam_sqrntt(&cfg, &mut rng),
        "hi" => fam_hi(&cfg, &mut rng),
        "hisqr" => fam_hisqr(&cfg, &mut rng),
        "hirows" => fam_hirows(&cfg, &mut rng),
        "mid" => fam_mid(&cfg, &mut rng),
        "midntt" => fam_midntt(&cfg, &mut rng),
        _ => {
            eprintln!(
                "usage: mul_cutoffs <prim|chunk|kara|kfft|kntt|ksqr|sqrfft|sqrntt|hi|hisqr|hirows|mid|midntt> \
                 [--seed S] [--rounds R] [--measure-us U] [--dry | --check]"
            );
            std::process::exit(2);
        }
    }
}
