//! The end-to-end check's workloads: public entries only, over shapes around the knobs. One list,
//! included by the tuner (`tuner/check.rs`) and by `benches/tuning_bias.rs`, so it names the crate
//! as `big_bits` (which the crate declares itself as in tune builds). Names and operands are
//! stable, as outputs are compared by hash across builds: add new items at the end.

use super::synth::{x_static, Big, Den, Fact, Geo, Odd, Synth};
use big_bits::utils::series::{series_eval_dyn, BBPSeries, Bbp, Engel, EngelSeries, Hyper, HyperSeries, Series};
use big_bits::utils::{consts::consts::*, div::*, mul::*, sqrt::*, trans::*};
use std::hint::black_box;

// exp's scale for a full-precision X in [0, 1): x's top limb is worth 2^(64m), so m = -1 puts
// all of x below the point. (m = -(x.len()) gives exactly 1 in a microsecond: no workload.)
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

pub struct Item {
    pub section: &'static str,
    pub name: String,
    // Runs the operation once; with `hash`, returns a hash of every output.
    pub run: Box<dyn FnMut(bool) -> u64>,
}

fn item(section: &'static str, name: String, run: impl FnMut(bool) -> u64 + 'static) -> Item {
    Item { section, name, run: Box::new(run) }
}

// FNV-1a over the parts' limbs, as bytes.
fn h(parts: &[&[u64]]) -> u64 {
    let mut x = 0xcbf2_9ce4_8422_2325u64;
    for b in parts.iter().flat_map(|p| p.iter()).flat_map(|w| w.to_le_bytes()) {
        x = (x ^ b as u64).wrapping_mul(0x100_0000_01b3);
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

// `div_dyn` and `div_rem_dyn` of nl limbs by dl.
fn div_pair(section: &'static str, r: &mut Rng, v: &mut Vec<Item>, (nl, dl): (usize, usize)) {
    let (n, d) = (r.vec(nl), r.vec(dl));
    let mut q = vec![0; nl - dl];
    let (n2, d2) = (n.clone(), d.clone());
    v.push(item(section, format!("div_dyn {nl}/{dl}"), move |hs| {
        let t = div_dyn(&n, &d, &mut q);
        if hs { h(&[&q, &[t]]) } else { black_box(t) }
    }));
    let mut q2 = vec![0; nl - dl];
    let mut nw = n2.clone();
    v.push(item(section, format!("div_rem_dyn {nl}/{dl}"), move |hs| {
        nw.copy_from_slice(&n2);
        let t = div_rem_dyn(&mut nw, &d2, &mut q2);
        if hs { h(&[&q2, &nw, &[t]]) } else { black_box(t) }
    }));
}

fn div_items(r: &mut Rng, v: &mut Vec<Item>) {
    for shape in [(120, 60), (200, 100), (400, 150), (1000, 60), (2000, 1000), (6000, 3000)] {
        div_pair("div", r, v, shape);
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

fn exp_ln_items(r: &mut Rng, v: &mut Vec<Item>, sizes: &[usize]) {
    for &o in sizes {
        // X in [0, 1) at full precision.
        let x = r.vec(o);
        let mut out = vec![0; o];
        v.push(item("trans", format!("exp_dyn {o}"), move |hs| {
            let k = exp_dyn(&x, EXP_FRACTION, &mut out).unwrap();
            if hs { h(&[&out, &[k]]) } else { black_box(k) as u64 }
        }));
        let x2 = r.vec(o);
        let mut out2 = vec![0; o];
        v.push(item("trans", format!("ln_dyn {o}"), move |hs| {
            let (k, neg) = ln_dyn(&x2, 0, &mut out2).unwrap();
            if hs { h(&[&out2, &[k as u64, neg as u64]]) } else { black_box(k) as u64 }
        }));
        if o + 4 <= 1100 {
            let x3 = r.vec(o);
            let mut out3 = vec![0; o];
            v.push(item("trans", format!("exp_static<1100> {o}"), move |hs| {
                let k = exp_static::<1100>(&x3, EXP_FRACTION, &mut out3).unwrap();
                if hs { h(&[&out3, &[k]]) } else { black_box(k) as u64 }
            }));
            let x4 = r.vec(o);
            let mut out4 = vec![0; o];
            v.push(item("trans", format!("ln_static<1100> {o}"), move |hs| {
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
    exp_ln_items(r, v, &[4, 16, 64, 300, 1040]);
}

// Divisions over a spread of quotient and divisor widths, for a dispatcher's own cost and its
// choices where the algorithms cross.
const DIV_GRID: [(usize, usize); 20] = [
    (4, 4), (8, 8), (8, 32), (12, 64), (24, 16), (24, 64), (32, 128), (48, 48), (64, 24), (64, 200), (96, 96),
    (130, 8), (150, 150), (200, 60), (300, 130), (300, 300), (600, 180), (600, 600), (30, 1000), (1200, 400),
];

// One series through the typed entry: `ctx` to w limbs below its zero limbs.
fn series_item<C: Series + 'static>(v: &mut Vec<Item>, name: String, ctx: C, w: usize) {
    let mut out = vec![0; w - ctx.zeros().min(w - 1)];
    v.push(item("series", name, move |hs| {
        let (neg, zeros) = series_eval_dyn(&ctx, &mut out);
        if hs { h(&[&out, &[neg as u64, zeros as u64]]) } else { black_box(out[0]) }
    }));
}

// The series evaluators' own workloads: the synthetic family at shapes on both sides of each
// decision the dispatch makes. (w, t, x): limbs, the terms the shift is set for, x's bits (0: x = 1).
fn series_items(v: &mut Vec<Item>) {
    fn hyper<S: HyperSeries + Synth + 'static>(v: &mut Vec<Item>, (w, t, xb): (usize, u64, u64)) {
        let s = S::decay(64 * w as u64, t);
        let x: &'static [u64] = if xb == 0 { &[1] } else { x_static(xb, 131 * xb + w as u64) };
        series_item(v, format!("series hyper {} w{w} t{t} x{xb}", S::name()), Hyper::<S>::new(x, xb + s).expect("x 2^-shift <= 1/2"), w);
    }
    fn engel<S: EngelSeries + Synth + 'static>(v: &mut Vec<Item>, (w, t): (usize, u64)) {
        series_item(v, format!("series engel {} w{w} t{t}", S::name()), Engel::<S>::new(S::decay(64 * w as u64, t)), w);
    }
    fn bbp<S: BBPSeries + Synth + 'static>(v: &mut Vec<Item>, (w, t): (usize, u64)) {
        series_item(v, format!("series bbp {} w{w} t{t}", S::name()), Bbp::<S>::new(S::decay(64 * w as u64, t)), w);
    }
    // Small coefficients: a short series, Horner's ground, rect's, binary splitting's, and between them.
    for shape in [(16, 16, 128), (64, 256, 64), (64, 64, 4096), (120, 500, 200), (256, 64, 1024), (500, 2000, 100), (1000, 256, 128), (1000, 64, 16384), (1000, 400, 4000)] {
        hyper::<Geo<8>>(v, shape);
    }
    // Large coefficients: two terms a division, and a tree that grows 112 bits a term.
    for shape in [(64, 64, 256), (256, 256, 4096), (700, 100, 300)] {
        hyper::<Geo<56>>(v, shape);
    }
    // Convergence from the coefficients: a factorial's, 32-bit denominators, and x = 1.
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
    // (Names, operands and order are those of when these were section "series".)
    exp_ln_items(&mut r, &mut v, &[2, 8, 30, 120, 500]);
    consts_items(&mut v);
    trans_items(&mut r, &mut v);
    for (ql, dl) in DIV_GRID {
        div_pair("divgrid", &mut r, &mut v, (ql + dl, dl));
    }
    series_items(&mut v);
    v
}

// Two-sided 95% Student t quantiles for 1..=30 degrees of freedom.
pub const T95: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];
