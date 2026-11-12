//! The workloads of the end-to-end check: public entries only, over shapes that sit around the
//! knobs (products, divisions, roots, exp and ln, the constants, theta and the AGM). One list,
//! included by the tuner (`tuner/check.rs`: `cargo tune check` and the gate that ends every
//! tune) and by `benches/tuning_bias.rs`, which compares two executables over it. It names the
//! crate as `big_bits` so the bench can include this file as it is; the crate declares itself
//! under that name in tune builds, and both declare `synth` (the synthetic series) beside this
//! module. Item names and operands are stable: outputs are compared by hash across builds, so
//! add new items at the end.

use super::synth::{x_of, Big, Den, Fact, Geo, Odd, Synth};
use big_bits::utils::series::{series_eval_dyn, BBPSeries, Bbp, Engel, EngelSeries, Hyper, HyperSeries, Series};
use big_bits::utils::{consts::consts::*, div::*, mul::*, sqrt::*, trans::*};
use std::hint::black_box;

// exp's scale for a full-precision X in [0, 1): exp reads x's top limb as worth 2^(64m), so
// m = -1 puts all of x below the point. Every exp workload of the tuner takes this scale. Until
// 2026-11-11 they passed m = -(x.len()), which put X below the last output limb: e^X came out
// exactly 1 in about a microsecond at any size (2.6 us at 1040 limbs, where a real one takes
// 4.5 ms), so no gate, sweep or bias run had timed exp's series.
pub const EXP_FRACTION: i64 = -1;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn vec(&mut self, n: usize) -> Vec<u64> {
        let mut v: Vec<u64> = (0..n).map(|_| self.next()).collect();
        if let Some(t) = v.last_mut() {
            *t |= 1 << 63;
        }
        v
    }
}

fn fnv(words: &[u64], h: &mut u64) {
    for w in words {
        for b in w.to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
}

pub struct Item {
    pub section: &'static str,
    pub name: String,
    // Runs the operation once; with `hash`, returns a hash of every output.
    pub run: Box<dyn FnMut(bool) -> u64>,
}

fn item(section: &'static str, name: String, run: impl FnMut(bool) -> u64 + 'static) -> Item {
    Item { section, name, run: Box::new(run) }
}

fn h(parts: &[&[u64]]) -> u64 {
    let mut x = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        fnv(p, &mut x);
    }
    x
}

fn mul_items(r: &mut Rng, v: &mut Vec<Item>) {
    for (l, s) in [(30, 30), (60, 40), (64, 64), (100, 100), (200, 60), (400, 400), (1000, 1000), (3000, 3000)] {
        let (a, b) = (r.vec(l), r.vec(s));
        let mut out = vec![0; l + s];
        v.push(item("mul", format!("mul_dyn {l}x{s}"), move |hs| {
            let t = mul_dyn(&a, &b, &mut out[..l + s - 1]);
            if hs { h(&[&out, &[t]]) } else { black_box(t) }
        }));
    }
    for (l, s) in [(64, 64), (400, 400), (1000, 1000)] {
        let (a, b) = (r.vec(l), r.vec(s));
        let mut out = vec![0; l + s];
        v.push(item("mul", format!("mul_static<2048> {l}x{s}"), move |hs| {
            let t = mul_static::<2048>(&a, &b, &mut out[..l + s - 1]);
            if hs { h(&[&out, &[t]]) } else { black_box(t) }
        }));
    }
    for n in [60, 80, 300, 400, 2000, 6000] {
        let a = r.vec(n);
        let mut out = vec![0; 2 * n];
        v.push(item("mul", format!("sqr_dyn {n}"), move |hs| {
            let t = sqr_dyn(&a, &mut out[..2 * n - 1]);
            if hs { h(&[&out, &[t]]) } else { black_box(t) }
        }));
    }
    for n in [60, 80, 3000, 4000] {
        let a = r.vec(n);
        let mut out = vec![0; 2 * n];
        v.push(item("mul", format!("sqr_static<8192> {n}"), move |hs| {
            let t = sqr_static::<8192>(&a, &mut out[..2 * n - 1]);
            if hs { h(&[&out, &[t]]) } else { black_box(t) }
        }));
    }
    for o in [20, 30, 300, 330, 600] {
        let (a, b) = (r.vec(o), r.vec(o));
        let (a2, b2) = (a.clone(), b.clone());
        let mut out = vec![0; o];
        v.push(item("mul", format!("hi_mul_dyn {o}"), move |hs| {
            let t = hi_mul_dyn(&a, &b, &mut out);
            if hs { h(&[&out, &[t]]) } else { black_box(t) }
        }));
        let mut out2 = vec![0; o];
        v.push(item("mul", format!("hi_mul_static<1024> {o}"), move |hs| {
            let t = hi_mul_static::<1024>(&a2, &b2, &mut out2);
            if hs { h(&[&out2, &[t]]) } else { black_box(t) }
        }));
        let mut out3 = vec![0; o];
        let a3 = r.vec(o);
        v.push(item("mul", format!("hi_sqr_dyn {o}"), move |hs| {
            let t = hi_sqr_dyn(&a3, &mut out3);
            if hs { h(&[&out3, &[t]]) } else { black_box(t) }
        }));
    }
    for n in [100, 130, 700] {
        let (long, short) = (r.vec(2 * n - 1), r.vec(n));
        let mut out = vec![0; n];
        v.push(item("mul", format!("mid_mul_dyn {n}"), move |hs| {
            let (x, y) = mid_mul_dyn(&long, &short, &mut out);
            if hs { h(&[&out, &[x, y]]) } else { black_box(x ^ y) }
        }));
        let (l2, s2) = (r.vec(2 * n - 1), r.vec(n));
        let mut out2 = vec![0; n];
        v.push(item("mul", format!("mid_mul_static<2048> {n}"), move |hs| {
            let (x, y) = mid_mul_static::<2048>(&l2, &s2, &mut out2);
            if hs { h(&[&out2, &[x, y]]) } else { black_box(x ^ y) }
        }));
    }
    for m in [300, 400, 1536] {
        let a = r.vec(m);
        let mut out = vec![0; m];
        v.push(item("mul", format!("sqr_mod_bnm1_dyn {m}"), move |hs| {
            sqr_mod_bnm1_dyn(&a, &mut out);
            if hs { h(&[&out]) } else { black_box(out[0]) }
        }));
    }
}

fn div_items(r: &mut Rng, v: &mut Vec<Item>) {
    for (nl, dl) in [(120, 60), (200, 100), (400, 150), (1000, 60), (2000, 1000), (6000, 3000)] {
        let (n, d) = (r.vec(nl), r.vec(dl));
        let mut q = vec![0; nl - dl];
        let (n2, d2) = (n.clone(), d.clone());
        v.push(item("div", format!("div_dyn {nl}/{dl}"), move |hs| {
            let t = div_dyn(&n, &d, &mut q);
            if hs { h(&[&q, &[t]]) } else { black_box(t) }
        }));
        let mut q2 = vec![0; nl - dl];
        let mut nw = n2.clone();
        v.push(item("div", format!("div_rem_dyn {nl}/{dl}"), move |hs| {
            nw.copy_from_slice(&n2);
            let t = div_rem_dyn(&mut nw, &d2, &mut q2);
            if hs { h(&[&q2, &nw, &[t]]) } else { black_box(t) }
        }));
    }
    for (nl, dl) in [(200, 100), (2000, 1000)] {
        let (n, d) = (r.vec(nl), r.vec(dl));
        let mut q = vec![0; nl - dl];
        let mut nw = n.clone();
        let (n2, d2) = (n.clone(), d.clone());
        let mut q2 = vec![0; nl - dl];
        v.push(item("div", format!("div_static<2048> {nl}/{dl}"), move |hs| {
            let t = div_static::<2048>(&n, &d, &mut q);
            if hs { h(&[&q, &[t]]) } else { black_box(t) }
        }));
        v.push(item("div", format!("div_rem_static<2048> {nl}/{dl}"), move |hs| {
            nw.copy_from_slice(&n2);
            let t = div_rem_static::<2048>(&mut nw, &d2, &mut q2);
            if hs { h(&[&q2, &nw, &[t]]) } else { black_box(t) }
        }));
    }
    for rl in [30, 40, 500] {
        let d = r.vec(rl + 1);
        let mut x = vec![0; rl];
        let d2 = d.clone();
        let mut x2 = vec![0; rl];
        v.push(item("div", format!("rcp_dyn {rl}"), move |hs| {
            rcp_dyn(&d, &mut x);
            if hs { h(&[&x]) } else { black_box(x[0]) }
        }));
        v.push(item("div", format!("rcp_static<1024> {rl}"), move |hs| {
            rcp_static::<1024>(&d2, &mut x2);
            if hs { h(&[&x2]) } else { black_box(x2[0]) }
        }));
    }
}

fn sqrt_items(r: &mut Rng, v: &mut Vec<Item>) {
    type F = fn(&mut [u64], &mut [u64]);
    let modes: [(&str, F, F); 3] = [
        ("sqrt", sqrt_dyn, sqrt_static::<4096>),
        ("sqrt_only", sqrt_only_dyn, sqrt_only_static::<4096>),
        ("sqrt_approx", sqrt_approx_dyn, sqrt_approx_static::<4096>),
    ];
    for xl in [16, 22, 32, 200, 700, 2000, 4000] {
        for (name, dynf, statf) in modes {
            let x0 = r.vec(xl);
            let (mut x, mut s) = (x0.clone(), vec![0; xl.div_ceil(2)]);
            v.push(item("sqrt", format!("{name}_dyn {xl}"), move |hs| {
                x.copy_from_slice(&x0);
                dynf(&mut x, &mut s);
                if hs { h(&[&x, &s]) } else { black_box(s[0]) }
            }));
            let x1 = r.vec(xl);
            let (mut y, mut t) = (x1.clone(), vec![0; xl.div_ceil(2)]);
            v.push(item("sqrt", format!("{name}_static<4096> {xl}"), move |hs| {
                y.copy_from_slice(&x1);
                statf(&mut y, &mut t);
                if hs { h(&[&y, &t]) } else { black_box(t[0]) }
            }));
        }
    }
}

fn exp_ln_items(section: &'static str, r: &mut Rng, v: &mut Vec<Item>, sizes: &[usize]) {
    for &o in sizes {
        // X in [0, 1) at full precision: x has o limbs, all of them below the point.
        let x = r.vec(o);
        let mut out = vec![0; o];
        v.push(item(section, format!("exp_dyn {o}"), move |hs| {
            let k = exp_dyn(&x, EXP_FRACTION, &mut out).unwrap();
            if hs { h(&[&out, &[k]]) } else { black_box(k) as u64 }
        }));
        let x2 = r.vec(o);
        let mut out2 = vec![0; o];
        v.push(item(section, format!("ln_dyn {o}"), move |hs| {
            let (k, neg) = ln_dyn(&x2, 0, &mut out2).unwrap();
            if hs { h(&[&out2, &[k as u64, neg as u64]]) } else { black_box(k) as u64 }
        }));
        if o + 4 <= 1100 {
            let x3 = r.vec(o);
            let mut out3 = vec![0; o];
            v.push(item(section, format!("exp_static<1100> {o}"), move |hs| {
                let k = exp_static::<1100>(&x3, EXP_FRACTION, &mut out3).unwrap();
                if hs { h(&[&out3, &[k]]) } else { black_box(k) as u64 }
            }));
            let x4 = r.vec(o);
            let mut out4 = vec![0; o];
            v.push(item(section, format!("ln_static<1100> {o}"), move |hs| {
                let (k, neg) = ln_static::<1100>(&x4, 0, &mut out4).unwrap();
                if hs { h(&[&out4, &[k as u64, neg as u64]]) } else { black_box(k) as u64 }
            }));
        }
    }
}

fn consts_items(v: &mut Vec<Item>) {
    // Past the 1024-limb tables the static constants are summed from scratch on every call.
    for o in [1030, 1060] {
        let mut out = vec![0; o];
        v.push(item("consts", format!("ln2_static<1100> {o}"), move |hs| {
            ln2_static::<1100>(&mut out).unwrap();
            if hs { h(&[&out]) } else { black_box(out[0]) }
        }));
        let mut out2 = vec![0; o];
        v.push(item("consts", format!("pi_static<1100> {o}"), move |hs| {
            pi_static::<1100>(&mut out2).unwrap();
            if hs { h(&[&out2]) } else { black_box(out2[0]) }
        }));
    }
}

fn trans_items(r: &mut Rng, v: &mut Vec<Item>) {
    for o in [64, 300, 1100] {
        let (a, b) = (r.vec(o), r.vec(o));
        let mut out = vec![0; o];
        v.push(item("trans", format!("agm_dyn {o}"), move |hs| {
            agm_dyn(&a, &b, &mut out);
            if hs { h(&[&out]) } else { black_box(out[0]) }
        }));
    }
    for o in [64, 300] {
        let (a, b) = (r.vec(o), r.vec(o));
        let mut out = vec![0; o];
        v.push(item("trans", format!("agm_static<1024> {o}"), move |hs| {
            agm_static::<1024>(&a, &b, &mut out).unwrap();
            if hs { h(&[&out]) } else { black_box(out[0]) }
        }));
    }
    for g in [100, 300, 1200] {
        let x = r.vec(1);
        let shift = 64 + 40;
        let (mut th2, mut th3) = (vec![0; g], vec![0; g + 1]);
        v.push(item("trans", format!("theta_dyn<5> {g}"), move |hs| {
            let z = theta_dyn::<5>(&x, shift, &mut th2, &mut th3);
            if hs { h(&[&th2, &th3, &[z as u64]]) } else { black_box(z) as u64 }
        }));
    }
    exp_ln_items("trans", r, v, &[4, 16, 64, 300, 1040]);
}

// Small and mid divisions over a spread of quotient and divisor widths (section "divgrid"), for a
// dispatcher's own cost and its choices where the algorithms cross. Registered last, so the
// earlier sections' operands (and hashes) don't move.
fn div_grid_items(r: &mut Rng, v: &mut Vec<Item>) {
    for (ql, dl) in [
        (4, 4), (8, 8), (8, 32), (12, 64), (24, 16), (24, 64), (32, 128), (48, 48), (64, 24), (64, 200), (96, 96),
        (130, 8), (150, 150), (200, 60), (300, 130), (300, 300), (600, 180), (600, 600), (30, 1000), (1200, 400),
    ] {
        let nl = ql + dl;
        let (n, d) = (r.vec(nl), r.vec(dl));
        let mut q = vec![0; ql];
        let (n2, d2) = (n.clone(), d.clone());
        v.push(item("divgrid", format!("div_dyn {nl}/{dl}"), move |hs| {
            let t = div_dyn(&n, &d, &mut q);
            if hs { h(&[&q, &[t]]) } else { black_box(t) }
        }));
        let mut q2 = vec![0; ql];
        let mut nw = n2.clone();
        v.push(item("divgrid", format!("div_rem_dyn {nl}/{dl}"), move |hs| {
            nw.copy_from_slice(&n2);
            let t = div_rem_dyn(&mut nw, &d2, &mut q2);
            if hs { h(&[&q2, &nw, &[t]]) } else { black_box(t) }
        }));
    }
}

// One series through the typed entry, as any caller sums it: `ctx` to w limbs below its zero
// limbs. Its outputs are within the entry's error, which the algorithm picked moves.
fn series_item<C: Series + 'static>(v: &mut Vec<Item>, name: String, ctx: C, w: usize) {
    let mut out = vec![0; w - ctx.zeros().min(w - 1)];
    v.push(item("series", name, move |hs| {
        let (neg, zeros) = series_eval_dyn(&ctx, &mut out);
        if hs { h(&[&out, &[neg as u64, zeros as u64]]) } else { black_box(out[0]) }
    }));
}

// The series evaluators' own workloads (section "series"): the synthetic family of `synth.rs`
// at shapes on both sides of each decision the dispatch makes, none of them a caller's series.
// (w, t, x): the precision in limbs, the terms the shift is set for, and x's bits (0: x = 1).
fn series_items(v: &mut Vec<Item>) {
    fn hyper<S: HyperSeries + Synth + 'static>(v: &mut Vec<Item>, (w, t, xb): (usize, u64, u64)) {
        let s = S::decay(64 * w as u64, t);
        let x: &'static [u64] = if xb == 0 { &[1] } else { Box::leak(x_of(xb, 131 * xb + w as u64).into_boxed_slice()) };
        series_item(v, format!("series hyper {} w{w} t{t} x{xb}", S::name()), Hyper::<S>::new(x, xb + s).expect("x 2^-shift <= 1/2"), w);
    }
    fn engel<S: EngelSeries + Synth + 'static>(v: &mut Vec<Item>, (w, t): (usize, u64)) {
        series_item(v, format!("series engel {} w{w} t{t}", S::name()), Engel::<S>::new(S::decay(64 * w as u64, t)), w);
    }
    fn bbp<S: BBPSeries + Synth + 'static>(v: &mut Vec<Item>, (w, t): (usize, u64)) {
        series_item(v, format!("series bbp {} w{w} t{t}", S::name()), Bbp::<S>::new(S::decay(64 * w as u64, t)), w);
    }
    // Small coefficients: a short series, a narrow x over many terms (Horner's ground), an x as
    // wide as the frame (rect's), a narrow x in a wide frame (binary splitting's), and the
    // boundaries between them.
    for shape in [(16, 16, 128), (64, 256, 64), (64, 64, 4096), (120, 500, 200), (256, 64, 1024), (500, 2000, 100), (1000, 256, 128), (1000, 64, 16384), (1000, 400, 4000)] {
        hyper::<Geo<8>>(v, shape);
    }
    // Large coefficients: two terms a division, and a tree that grows by 112 bits a term.
    for shape in [(64, 64, 256), (256, 256, 4096), (700, 100, 300)] {
        hyper::<Geo<56>>(v, shape);
    }
    // Convergence from the coefficients: a factorial's, at an x as wide as the frame (an
    // argument nothing has reduced) and at a narrow one; denominators of 32 bits; and x = 1.
    for shape in [(30, 200, 1920), (120, 600, 7680), (500, 2000, 512), (300, 2000, 0)] {
        hyper::<Fact>(v, shape);
    }
    for shape in [(120, 100, 2000), (500, 300, 0)] {
        hyper::<Den<32>>(v, shape);
    }
    for shape in [(8, 30), (120, 600), (700, 3000)] {
        engel::<Fact>(v, shape);
    }
    engel::<Den<32>>(v, (300, 200));
    for shape in [(8, 20), (120, 900), (700, 5000)] {
        bbp::<Odd>(v, shape);
    }
    bbp::<Big<40>>(v, (300, 600));
}

pub fn items() -> Vec<Item> {
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let mut v = Vec::new();
    mul_items(&mut r, &mut v);
    div_items(&mut r, &mut v);
    sqrt_items(&mut r, &mut v);
    // (Section "series" until 2026-11-11, when that section became the series evaluators' own
    // workloads, at the end. The names, operands and order here are unchanged.)
    exp_ln_items("trans", &mut r, &mut v, &[2, 8, 30, 120, 500]);
    consts_items(&mut v);
    trans_items(&mut r, &mut v);
    div_grid_items(&mut r, &mut v);
    series_items(&mut v);
    v
}

// Two-sided 95% Student t quantiles for 1..=30 degrees of freedom.
pub const T95: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];
