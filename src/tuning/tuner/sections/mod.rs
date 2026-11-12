//! Each section's tuner: what it measures (its leaf kernels' models and sweeps). Registered in
//! derive order; a section not listed has nothing measured yet.

pub mod agm;
#[cfg(target_arch = "x86_64")]
pub mod base;
pub mod bin_split;
pub mod div;
pub mod mul;
pub mod series;
pub mod sqrt;
pub mod trans;

use super::measure::Level;
use super::out::Tuned;

pub type Tune = fn(Level) -> Vec<Tuned>;

// `base` holds only the x86-64 ADX crossovers, so it is tuned only there.
#[cfg(target_arch = "x86_64")]
const BASE: &[(&str, Tune)] = &[("base", base::tune)];
#[cfg(not(target_arch = "x86_64"))]
const BASE: &[(&str, Tune)] = &[];

/// Every section's tuner, in dependency order.
pub fn sections() -> Vec<(&'static str, Tune)> {
    BASE.iter().chain(SECTIONS).copied().collect()
}

pub const SECTIONS: &[(&str, Tune)] = &[("mul", mul::tune), ("div", div::tune), ("sqrt", sqrt::tune), ("bin_split", bin_split::tune), ("series", series::tune), ("theta", trans::tune_theta), ("agm", agm::tune), ("exp", trans::tune_exp), ("ln", trans::tune_ln)];
