//! A synthetic family of series, for tuning the series evaluators as general evaluators: none of
//! these is a caller's series, and nothing in the series section is fitted to one. Each member
//! sets one of the things a series' coefficients decide, apart from the others:
//!
//! - their size, which is what a term adds to the exact fraction (binary splitting's numbers)
//!   and what sets how many terms one of Horner's divisions covers: `Geo<LQ>`, q and r of LQ
//!   bits with r / q about 1, so they add size and no convergence;
//! - convergence of their own, on top of the shift's: `Den<LQ>`, q of LQ bits over r = 1, each
//!   term at least LQ - 1 bits below the last; and `Fact`, q(n) = n, a factorial's, whose terms
//!   converge faster the further they go;
//! - for BBP, whose terms don't compound: `Odd`, 1 / (2n + 1), and `Big<LQ>`, a q of LQ bits.
//!
//! The other dials are the evaluation's: the frame, the shift (through `Synth::decay`, as the
//! terms wanted) and x's width (`x_of`). Written against the crate's public paths, so
//! `check_items.rs` and `benches/tuning_bias.rs` can take the family too.

use big_bits::utils::series::{BBPSeries, EngelSeries, HyperSeries};

/// A member of the family.
pub trait Synth {
    fn name() -> String;
    /// The bits a term the shift should give (the decay s) for about `t` terms to reach `bits`
    /// bits: the member's own convergence taken off, and at least 1.
    fn decay(bits: u64, t: u64) -> u64;
}

/// q and r of LQ bits (8 to 62), r = q - 2: Hyper.
pub struct Geo<const LQ: u32>;
/// q of LQ bits (8 to 62), r = 1: Hyper and Engel.
pub struct Den<const LQ: u32>;
/// q(n) = n, r = 1: Hyper and Engel.
pub struct Fact;
/// q(n) = 2n + 1: BBP.
pub struct Odd;
/// q of LQ bits (8 to 62): BBP.
pub struct Big<const LQ: u32>;

// An odd number of exactly `bits` bits that changes from term to term.
const fn sized(bits: u32, n: u64) -> u64 {
    (1 << (bits - 1)) + 2 * (n & 31) + 3
}

// A tail of terms at most 2^-(per n) each, falling by half or more a term: under 2^(1 - per n).
fn geometric(n: u64, per: u64) -> u64 {
    n.saturating_mul(per).saturating_sub(1)
}

// The tail from y^n / n! for y <= 2^-s <= 1/2: under 2 y^n / n!, and n! >= (n / e)^n.
fn factorial(n: u64, s: i64) -> u64 {
    let nf = n as f64;
    let fact = (nf * (nf.log2() - std::f64::consts::LOG2_E)).max(0.0) as u64;
    n.saturating_mul(s.max(0) as u64).saturating_add(fact).saturating_sub(1)
}

impl<const LQ: u32> Synth for Geo<LQ> {
    fn name() -> String {
        format!("geo{LQ}")
    }
    fn decay(bits: u64, t: u64) -> u64 {
        (bits / t.max(1)).max(1)
    }
}

impl<const LQ: u32> HyperSeries for Geo<LQ> {
    const START: u64 = 1;
    // |term n| <= |x 2^-k|^n: p / r and every r / q are at most 1.
    fn tail_bits(n: u64, s: i64) -> u64 {
        geometric(n, s.max(0) as u64)
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        sized(LQ, n)
    }
    fn r(n: u64) -> u64 {
        sized(LQ, n) - 2
    }
}

impl<const LQ: u32> Synth for Den<LQ> {
    fn name() -> String {
        format!("den{LQ}")
    }
    fn decay(bits: u64, t: u64) -> u64 {
        (bits / t.max(1)).saturating_sub(LQ as u64 - 1).max(1)
    }
}

impl<const LQ: u32> HyperSeries for Den<LQ> {
    const START: u64 = 1;
    // Each term divides by a q of at least 2^(LQ - 1) more.
    fn tail_bits(n: u64, s: i64) -> u64 {
        geometric(n, s.max(0) as u64 + LQ as u64 - 1)
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        sized(LQ, n)
    }
    fn r(_: u64) -> u64 {
        1
    }
}

impl<const LQ: u32> EngelSeries for Den<LQ> {
    const START: u64 = 1;
    fn tail_bits(n: u64, s: i64) -> u64 {
        geometric(n, s.max(0) as u64 + LQ as u64 - 1)
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        sized(LQ, n)
    }
}

impl<const LQ: u32> Synth for Big<LQ> {
    fn name() -> String {
        format!("big{LQ}")
    }
    // A BBP term's q divides that term alone: the convergence is all the shift's.
    fn decay(bits: u64, t: u64) -> u64 {
        (bits / t.max(1)).max(1)
    }
}

impl<const LQ: u32> BBPSeries for Big<LQ> {
    const START: u64 = 1;
    // Term n is 2^(-k n) / q(n): the tail is under 2^(1 - k n - (LQ - 1)).
    fn tail_bits(n: u64, s: i64) -> u64 {
        geometric(n, s.max(0) as u64).saturating_add(LQ as u64 - 1)
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        sized(LQ, n)
    }
}

impl Synth for Fact {
    fn name() -> String {
        "fact".into()
    }
    // t s + log2 t! = bits, with log2 t! about t log2(t / e).
    fn decay(bits: u64, t: u64) -> u64 {
        let t = t.max(1) as f64;
        (bits as f64 / t - (t.log2() - std::f64::consts::LOG2_E).max(0.0)).max(1.0) as u64
    }
}

impl HyperSeries for Fact {
    const START: u64 = 1;
    fn tail_bits(n: u64, s: i64) -> u64 {
        factorial(n, s)
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
    fn r(_: u64) -> u64 {
        1
    }
}

impl EngelSeries for Fact {
    const START: u64 = 1;
    fn tail_bits(n: u64, s: i64) -> u64 {
        factorial(n, s)
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
}

impl Synth for Odd {
    fn name() -> String {
        "odd".into()
    }
    fn decay(bits: u64, t: u64) -> u64 {
        (bits / t.max(1)).max(1)
    }
}

impl BBPSeries for Odd {
    const START: u64 = 1;
    fn tail_bits(n: u64, s: i64) -> u64 {
        geometric(n, s.max(0) as u64)
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        2 * n + 1
    }
}

/// An odd x of exactly `bits` bits (at least 2) from `seed`.
pub fn x_of(bits: u64, seed: u64) -> Vec<u64> {
    let limbs = bits.div_ceil(64) as usize;
    let mut s = seed | 1;
    let mut x: Vec<u64> = (0..limbs)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        })
        .collect();
    let top = bits - 64 * (limbs as u64 - 1);
    x[limbs - 1] = (x[limbs - 1] >> (64 - top)) | 1 << (top - 1);
    x[0] |= 1;
    x
}
