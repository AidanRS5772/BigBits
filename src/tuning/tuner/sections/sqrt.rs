//! Section `sqrt`: every decision is a one-input cutoff, swept through the public entries with
//! the knobs in question forced to one side or the other per probe.

use crate::tuning::resolve::Value;
use crate::tuning::tuner::measure::{pad, Level, Probe};
use crate::tuning::tuner::out::Tuned;
use crate::tuning::tuner::sweep::{cutoff_sweep, geo_to, keep};
use crate::tuning::{install, prepare, Handle};
use crate::utils::sqrt::{
    sqrt_approx_dyn, sqrt_approx_static, sqrt_dyn, sqrt_dyn_at, sqrt_only_dyn, sqrt_only_static, sqrt_static,
};
use std::hint::black_box;

type Root = fn(&mut [u64], &mut [u64]);
const MAX: u64 = usize::MAX as u64;

fn h(path: &str, v: u64) -> Handle {
    prepare(path, &Value::Int(v)).expect("a valid sqrt knob value")
}

// `f` on a full x of 2n limbs (root n), under `set`, restoring x each call.
fn root_probe(mut f: impl FnMut(&mut [u64], &mut [u64]) + 'static, n: usize, pass: usize, set: Vec<Handle>) -> Probe<'static> {
    let x0 = {
        let mut x = pad(2 * n, n as u64 * 3 + 1, pass);
        x[2 * n - 1] |= 1 << 63;
        x
    };
    let (mut x, mut s) = (pad(2 * n, 5, pass), pad(n, 7, pass));
    Probe::new("sqrt", move || {
        for &hd in &set {
            install(hd);
        }
        x.copy_from_slice(&x0);
        f(black_box(&mut x), &mut s);
        black_box(&s);
    })
}

pub fn tune(level: Level) -> Vec<Tuned> {
    let saved = crate::tuning::snapshot();
    let mut out = Vec::new();
    let small: Vec<usize> = (3..=64).collect();

    // The recursion's leaf: the binomial root against one Zimmermann stage over a binomial child.
    // Roots switch at n >= knob: the knob is one past the last binomial size.
    let leaf = cutoff_sweep(level, "sqrt.leaf", &small, None, 1, &|n, p| {
        [
            root_probe(sqrt_dyn as Root, n, p, vec![h("sqrt.dyn_rem", MAX)]),
            root_probe(sqrt_dyn as Root, n, p, vec![h("sqrt.dyn_rem", 3), h("sqrt.leaf", MAX)]),
        ]
    });
    // Later sweeps run under the values found so far.
    keep("sqrt", &mut out, leaf);

    // Each entry's top level, with the leaf as just tuned. Static at N = 128, close to the operands.
    let tops: [(&'static str, Root); 6] = [
        ("sqrt.dyn_rem", sqrt_dyn),
        ("sqrt.dyn_only", sqrt_only_dyn),
        ("sqrt.dyn_approx", sqrt_approx_dyn),
        ("sqrt.static_rem", sqrt_static::<128>),
        ("sqrt.static_only", sqrt_only_static::<128>),
        ("sqrt.static_approx", sqrt_approx_static::<128>),
    ];
    for (path, f) in tops {
        keep("sqrt", &mut out, cutoff_sweep(level, path, &small, None, 1, &|n, p| [root_probe(f, n, p, vec![h(path, MAX)]), root_probe(f, n, p, vec![h(path, 3)])]));
    }

    // The wrapped remainder at a stage of n root limbs: classic stages throughout against the top
    // stage alone taking the wrapped remainder.
    let coarse = geo_to(64.0, 8.0, 4000);
    out.push(cutoff_sweep(level, "sqrt.dyn_wrapped_rem", &coarse, Some(4000), 1, &|n, p| {
        let at = |cut: usize| root_probe(move |x, s| sqrt_dyn_at(x, s, cut), n, p, vec![]);
        [at(usize::MAX), at(n)]
    }));
    crate::tuning::restore(&saved);
    out
}
