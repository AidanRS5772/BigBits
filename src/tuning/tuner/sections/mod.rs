//! Each section's tuner, in dependency order: what it measures (fits and sweeps), its dispatch
//! tables, and what it fits again once those are in force.

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
/// A dispatch table's measurement: its change points and its evidence.
pub type Table = fn(Level) -> (Vec<u32>, String);

/// A section's tuner, in the order the tune loop runs it: `tune`; then each of its dispatch
/// tables (the knob's key and its measurement), under `tune`'s values; then `late`, under the tables.
pub struct SectionTuner {
    pub name: &'static str,
    pub tune: Tune,
    pub tables: &'static [(&'static str, Table)],
    pub late: Option<Tune>,
}

const fn plain(name: &'static str, tune: Tune) -> SectionTuner {
    SectionTuner { name, tune, tables: &[], late: None }
}

// `base` holds only the x86-64 ADX crossovers, so it is tuned only there.
#[cfg(target_arch = "x86_64")]
const BASE: &[SectionTuner] = &[plain("base", base::tune)];
#[cfg(not(target_arch = "x86_64"))]
const BASE: &[SectionTuner] = &[];

/// Every section's tuner, in dependency order.
pub fn sections() -> Vec<&'static SectionTuner> {
    BASE.iter().chain(SECTIONS).collect()
}

pub const SECTIONS: &[SectionTuner] = &[
    // mul's second phase fits recursive Karatsuba's and the FFT's equations, under its table.
    SectionTuner { name: "mul", tune: mul::tune, tables: &[("kara", mul::kara_table)], late: Some(mul::tune_late) },
    // div's and the series' fit their equations again on the shapes their tables leave them.
    SectionTuner { name: "div", tune: div::tune, tables: &[("dyn_small", |l| div::small_table(l, false)), ("dyn_rem_small", |l| div::small_table(l, true))], late: Some(div::tune_late) },
    plain("sqrt", sqrt::tune),
    plain("bin_split", bin_split::tune),
    SectionTuner { name: "series", tune: series::tune, tables: &[("small", series::small_table)], late: Some(series::tune_late) },
    plain("theta", trans::tune_theta),
    plain("agm", agm::tune),
    plain("exp", trans::tune_exp),
    plain("ln", trans::tune_ln),
];
