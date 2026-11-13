//! Section `base` (x86-64 only): the two crossovers between a plain loop and its mulx/adcx
//! (ADX) twin, each a cutoff sweep with the knob forced to each side. The ADX loop runs from
//! n >= knob. On a CPU without ADX the section tunes nothing. Compile-checked only: its first
//! real run is the x86 handoff's (docs/tuning_implementation.md).

use crate::tuning::resolve::Value;
use crate::tuning::tuner::measure::{pad, Level, Probe};
use crate::tuning::tuner::out::Tuned;
use crate::tuning::tuner::sweep::{cutoff_sweep, with_knob};
use crate::utils::mul::mul_add_prim;
use crate::utils::utils::add_mul_limb;
use std::hint::black_box;

pub fn tune(level: Level) -> Vec<Tuned> {
    if !crate::utils::adx_available() {
        eprintln!("  base: this CPU has no ADX, so its two crossovers have nothing to choose between");
        return Vec::new();
    }
    // Plain below the knob, ADX from it: forced plain by a knob no length reaches, ADX by 0.
    let (plain, adx) = (Value::Int(1 << 40), Value::Int(0));
    let sizes: Vec<usize> = (1..=96).collect();
    let prim = cutoff_sweep(level, "base.prim_adx", &sizes, None, 1, &|n, p| {
        let at = |v: &Value| {
            let mut a = pad(n, n as u64 * 3 + 1, p);
            with_knob("base.prim_adx", v, Probe::new("mul_add_prim", move || {
                black_box(mul_add_prim(black_box(&mut a), black_box(0x9e37_79b9_7f4a_7c15), 0));
            }))
        };
        [at(&plain), at(&adx)]
    });
    let limb = cutoff_sweep(level, "base.limb_adx", &sizes, None, 1, &|n, p| {
        let at = |v: &Value| {
            let (mut x, s) = (pad(n, n as u64 * 5 + 2, p), pad(n, n as u64 * 7 + 3, p));
            with_knob("base.limb_adx", v, Probe::new("add_mul_limb", move || {
                black_box(add_mul_limb(black_box(&mut x), black_box(&s), black_box(0x9e37_79b9_7f4a_7c15)));
            }))
        };
        [at(&plain), at(&adx)]
    });
    vec![prim, limb]
}
