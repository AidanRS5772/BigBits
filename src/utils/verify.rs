//! Linear-time checks of products by their residues mod the three NTT primes.
//!
//! `c = a·b` implies `c ≡ a·b` mod each of P1, P2 and P3, so mod P = P1·P2·P3 ≈ 2^188. The converse
//! fails only for an error D = c − a·b that all three primes divide. 2 is invertible mod each
//! prime, so any D = 2^s·E with 0 < |E| < P is caught whatever s is: an error whose nonzero bits
//! fit in a 188-bit window, such as one wrong FFT coefficient (±e·2^(16k)) or one wrong limb, is
//! always detected. An error spread wider escapes only if it vanishes mod all three primes at once.
//! 2^16 has multiplicative order about 2^56 mod each prime, so there is no periodic cancellation
//! like mod B − 1, where errors one limb apart cancel.
//!
//! Residues compose, so other identities check the same way: division (n = q·d + r) and square
//! roots (x = s² + r). The checks cost about one Montgomery reduction per limb per prime and never
//! allocate.
use crate::utils::mul::{mod_mul, reduce_mont, NTTPrime, P1, P2, P3};

/// The moduli, P1, P2 and P3 of the NTT.
pub const PRIMES: [u64; 3] = [P1::P, P2::P, P3::P];
// −p^-1 mod 2^64, B mod p and B² mod p, B = 2^64.
const P_INV: [u64; 3] = [P1::P_INV, P2::P_INV, P3::P_INV];
const B1: [u64; 3] = [P1::R, P2::R, P3::R];
const B2: [u64; 3] = [P1::R_SQR, P2::R_SQR, P3::R_SQR];

// Independent folds per prime. Each fold is a serial chain of dependent multiplies, so one chain
// per prime waits on multiply latency; four segments per prime keep the multipliers busy. Inputs
// shorter than LANES · MIN_SEGMENT limbs fold in one chain per prime.
const LANES: usize = 4;
const MIN_SEGMENT: usize = 32;

/// A number's residues mod P1, P2 and P3 (plain values in [0, p), not Montgomery form).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Residues(pub [u64; 3]);

// limb mod p for p > 2^62: limb < 2^64 < 4p, and 2p < 2^64.
#[inline(always)]
fn reduce_limb(limb: u64, p: u64) -> u64 {
    let x = if limb >= 2 * p { limb - 2 * p } else { limb };
    if x >= p {
        x - p
    } else {
        x
    }
}

// a·b·B^-1 mod PRIMES[i], for a, b < p.
#[inline(always)]
fn mont_mul(a: u64, b: u64, i: usize) -> u64 {
    reduce_mont(a as u128 * b as u128, PRIMES[i], P_INV[i])
}

// One step of the low-to-high fold: REDC(limb'·B + r) = limb' + r·B^-1 mod p, with limb' the limb
// reduced below p. With m = r·(−p^-1) mod B, (r + m·p)/B = hi(m·p) + [r ≠ 0], since the low halves
// sum to exactly B unless r = 0: two multiplies, where reduce_mont's general REDC takes three.
// limb' + hi(m·p) + 1 ≤ 2p − 1, so one subtraction brings it into [0, p).
#[inline(always)]
fn fold(r: u64, limb: u64, i: usize) -> u64 {
    let p = PRIMES[i];
    let m = r.wrapping_mul(P_INV[i]);
    let hi = ((m as u128 * p as u128) >> 64) as u64;
    let v = reduce_limb(limb, p) + hi + (r != 0) as u64;
    if v >= p {
        v - p
    } else {
        v
    }
}

#[inline(always)]
fn add_mod(a: u64, b: u64, i: usize) -> u64 {
    // Both are below p < 2^63, so the sum fits.
    let s = a + b;
    if s >= PRIMES[i] {
        s - PRIMES[i]
    } else {
        s
    }
}

// B^(e+1) mod each prime: the Montgomery form of B^e, by square and multiply on B's Montgomery
// form B² from 1's, B.
fn pow_b(mut e: usize) -> [u64; 3] {
    let (mut acc, mut base) = (B1, B2);
    while e > 0 {
        if e & 1 == 1 {
            acc = [0, 1, 2].map(|i| mont_mul(acc[i], base[i], i));
        }
        base = [0, 1, 2].map(|i| mont_mul(base[i], base[i], i));
        e >>= 1;
    }
    acc
}

impl Residues {
    pub const ZERO: Self = Residues([0; 3]);

    /// x mod P1, P2 and P3, for x little-endian limbs (any length, zero limbs allowed).
    pub fn of(x: &[u64]) -> Self {
        Self::fold_limbs(x, None)
    }

    /// x + top·B^x.len(): the multiplication entries' (out, returned top limb) convention.
    pub fn of_with_top(x: &[u64], top: u64) -> Self {
        Self::fold_limbs(x, Some(top))
    }

    pub fn of_limb(v: u64) -> Self {
        Residues(PRIMES.map(|p| reduce_limb(v, p)))
    }

    pub fn add(self, o: Self) -> Self {
        Residues([0, 1, 2].map(|i| add_mod(self.0[i], o.0[i], i)))
    }

    pub fn sub(self, o: Self) -> Self {
        let mut r = [0; 3];
        for i in 0..3 {
            r[i] = if self.0[i] >= o.0[i] {
                self.0[i] - o.0[i]
            } else {
                self.0[i] + PRIMES[i] - o.0[i]
            };
        }
        Residues(r)
    }

    pub fn mul(self, o: Self) -> Self {
        Residues([0, 1, 2].map(|i| mod_mul(self.0[i], o.0[i], PRIMES[i])))
    }

    /// self·B^k, B = 2^64.
    pub fn shl_limbs(self, k: usize) -> Self {
        if k == 0 {
            return self;
        }
        let bk = pow_b(k); // B^(k+1)
        Residues([0, 1, 2].map(|i| mont_mul(self.0[i], bk[i], i)))
    }

    // x's limbs, then `top` above them. After n limbs a fold holds their value·B^-(n-1).
    fn fold_limbs(x: &[u64], top: Option<u64>) -> Self {
        let n = x.len() + top.is_some() as usize;
        if n == 0 {
            return Self::ZERO;
        }
        let seg = x.len() / LANES;
        if seg < MIN_SEGMENT {
            let mut r = [0u64; 3];
            for &limb in x.iter().chain(&top) {
                r = [0, 1, 2].map(|i| fold(r[i], limb, i));
            }
            // r·B^(n-1) = mont_mul(r, B^n).
            let bn = pow_b(n - 1);
            return Residues([0, 1, 2].map(|i| mont_mul(r[i], bn[i], i)));
        }
        // Segments 0..3 hold seg limbs each; the last also takes x's rem leftover limbs and `top`.
        let (s0, rest) = x.split_at(seg);
        let (s1, rest) = rest.split_at(seg);
        let (s2, s3) = rest.split_at(seg);
        let mut r = [[0u64; 3]; LANES];
        for (((&a, &b), &c), &d) in s0.iter().zip(s1).zip(s2).zip(&s3[..seg]) {
            for i in 0..3 {
                r[0][i] = fold(r[0][i], a, i);
                r[1][i] = fold(r[1][i], b, i);
                r[2][i] = fold(r[2][i], c, i);
                r[3][i] = fold(r[3][i], d, i);
            }
        }
        let extra = n - LANES * seg;
        for &limb in s3[seg..].iter().chain(&top) {
            r[3] = [0, 1, 2].map(|i| fold(r[3][i], limb, i));
        }
        // With x_j segment j's value, r_j = x_j·B^-(seg-1) for j < 3 and r_3 = x_3·B^-(seg+extra-1),
        // so x·B^-(seg-1) = r_0 + W·r_1 + W²·r_2 + W³·B^extra·r_3, W = B^seg, by Horner.
        let wm = pow_b(seg); // W·B
        Residues([0, 1, 2].map(|i| {
            let mut acc = r[3][i];
            for _ in 0..extra {
                acc = mont_mul(acc, B2[i], i); // ·B
            }
            for j in (0..3).rev() {
                acc = add_mod(mont_mul(acc, wm[i], i), r[j][i], i);
            }
            // ·B^(seg-1) = mont_mul with W = mont_mul(W·B, 1).
            mont_mul(acc, mont_mul(wm[i], 1, i), i)
        }))
    }
}

/// Whether c ≡ a·b mod P1·P2·P3, with c read as an integer of any length. A correct product always
/// passes; see the module comment for which wrong ones are certain to fail.
pub fn verify_mul(a: &[u64], b: &[u64], c: &[u64]) -> bool {
    Residues::of(a).mul(Residues::of(b)) == Residues::of(c)
}

/// [`verify_mul`] for c = a².
pub fn verify_sqr(a: &[u64], c: &[u64]) -> bool {
    let r = Residues::of(a);
    r.mul(r) == Residues::of(c)
}
