//! Section `agm`: the tail caps, each a parameter sweep through the `_at` entries, which take
//! the cap. Static at N = 1100 only.

use crate::tuning::resolve::Value;
use crate::tuning::tuner::measure::{pad, Level, Probe};
use crate::tuning::tuner::out::Tuned;
use crate::tuning::tuner::sweep::{int, ints, param_sweep};
use crate::utils::trans::{agm_dyn_at, agm_static_at};
use std::hint::black_box;

pub fn tune(level: Level) -> Vec<Tuned> {
    // A spread of caps and sizes four apart; both defaults (10 and 4) are among them.
    let cands = ints(&[2, 4, 6, 8, 9, 10, 12, 14, 16, 19]);
    let cap = |v: &Value| int(v) as usize;
    // a and b of n limbs in an n-limb frame, as the logarithm calls it (b the smaller).
    let pair = |n: usize, pass: usize| {
        let (mut a, mut b) = (pad(n, n as u64 * 3 + 1, pass).to_vec(), pad(n, n as u64 * 5 + 2, pass).to_vec());
        a[n - 1] |= 1 << 63;
        b[n - 1] >>= 8;
        (a, b)
    };
    let sizes = |lo: f64, hi: usize| -> Vec<usize> {
        (0..40).map(|i| (lo * 4f64.powi(i)) as usize).take_while(|&n| n <= hi).collect()
    };
    let dynamic = param_sweep(level, "agm.dyn_tail_terms", &sizes(16.0, 4096), &cands, &|n, c, p| {
        let ((a, b), mut o, c) = (pair(n, p), pad(n, 7, p), cap(c));
        Probe::new("agm_dyn", move || {
            black_box(agm_dyn_at(black_box(&a), black_box(&b), c, &mut o));
        })
    });
    let stat = param_sweep(level, "agm.static_tail_terms", &sizes(16.0, 1024), &cands, &|n, c, p| {
        let ((a, b), mut o, c) = (pair(n, p), pad(n, 9, p), cap(c));
        Probe::new("agm_static", move || {
            black_box(agm_static_at::<1100>(black_box(&a), black_box(&b), c, &mut o).expect("N holds the frame"));
        })
    });
    vec![dynamic, stat]
}
