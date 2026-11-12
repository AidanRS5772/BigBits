#![allow(unused_imports)]
#![allow(long_running_const_eval)]
extern crate core;
// The tuner's check workloads are shared with a benchmark, which names this crate from outside.
#[cfg(feature = "tune")]
extern crate self as big_bits;

pub mod bit_nums;
#[doc(hidden)]
pub mod tuning;
pub mod utils;

#[cfg(test)]
mod tests;

#[cfg(feature = "_bench_internals")]
#[doc(hidden)]
pub use utils::{mul::*, utils::*};
