use super::{mul_ref, rand_nonzero_vec, rand_vec};
use crate::utils::utils::{add_buf, sub_buf};
use crate::utils::verify::{verify_mul, verify_sqr, Residues, PRIMES};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// x mod p by Horner in u128 arithmetic, independent of the Montgomery fold.
fn horner(x: &[u64], p: u64) -> u64 {
    x.iter().rev().fold(0u64, |r, &l| {
        ((((r as u128) << 64) | l as u128) % p as u128) as u64
    })
}

fn horner_all(x: &[u64]) -> [u64; 3] {
    PRIMES.map(|p| horner(x, p))
}

// a · b as a full l + s limb buffer.
fn product(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut c = mul_ref(a, b);
    c.resize(a.len() + b.len(), 0);
    c
}

// c ± 2^bit, in place, as a wrong coefficient or limb would leave it.
fn nudge(c: &mut [u64], bit: usize, up: bool) {
    let mut delta = vec![0u64; c.len()];
    delta[bit / 64] = 1 << (bit % 64);
    if up {
        add_buf(c, &delta);
    } else {
        sub_buf(c, &delta);
    }
}

// Every length up to 40 (one fold chain per prime), and lengths around the lane path's
// 4 · 32-limb threshold with every leftover count, folded with and without a top limb.
#[test]
fn test_residues_match_horner() {
    let mut rng = StdRng::seed_from_u64(0x7e51);
    let lengths = (0..40).chain([
        127, 128, 129, 130, 131, 132, 255, 256, 257, 1000, 1001, 1002, 1003,
    ]);
    for len in lengths {
        for kind in 0..4 {
            let x: Vec<u64> = match kind {
                0 => (0..len).map(|_| rng.gen()).collect(),
                1 => vec![u64::MAX; len],
                // Limbs at and around each prime and 2p, where the limb reduction branches.
                2 => (0..len)
                    .map(|i| {
                        let p = PRIMES[i % 3];
                        [p - 1, p, p + 1, 2 * p - 1, 2 * p, 2 * p + 1][rng.gen_range(0..6)]
                    })
                    .collect(),
                _ => (0..len)
                    .map(|i| if i % 3 == 0 { 0 } else { rng.gen() })
                    .collect(),
            };
            assert_eq!(Residues::of(&x).0, horner_all(&x), "len {len} kind {kind}");
            let top: u64 = if kind == 1 { u64::MAX } else { rng.gen() };
            let mut with_top = x.clone();
            with_top.push(top);
            assert_eq!(
                Residues::of_with_top(&x, top).0,
                horner_all(&with_top),
                "len {len} kind {kind} with a top limb"
            );
        }
    }
}

#[test]
fn test_residue_arithmetic() {
    let mut rng = StdRng::seed_from_u64(0x7e52);
    for _ in 0..200 {
        let (la, lb, k) = (
            rng.gen_range(0..12),
            rng.gen_range(0..12),
            rng.gen_range(0..300),
        );
        let a = rand_vec(la, rng.gen());
        let b = rand_vec(lb, rng.gen());
        let top: u64 = rng.gen();
        let (ra, rb) = (Residues::of(&a), Residues::of(&b));
        for i in 0..3 {
            let p = PRIMES[i] as u128;
            let (x, y) = (ra.0[i] as u128, rb.0[i] as u128);
            assert_eq!(ra.add(rb).0[i] as u128, (x + y) % p);
            assert_eq!(ra.sub(rb).0[i] as u128, (x + p - y) % p);
            assert_eq!(ra.mul(rb).0[i] as u128, (x * y) % p);
        }
        let mut shifted = vec![0u64; k];
        shifted.extend_from_slice(&a);
        assert_eq!(ra.shl_limbs(k), Residues::of(&shifted));
        let mut with_top = a.clone();
        with_top.push(top);
        assert_eq!(Residues::of_with_top(&a, top), Residues::of(&with_top));
        assert_eq!(Residues::of_limb(top).0, horner_all(&[top]));
    }
}

#[test]
fn test_verify_accepts_products() {
    let mut rng = StdRng::seed_from_u64(0x7e53);
    for (la, lb) in [
        (1, 1),
        (2, 1),
        (5, 3),
        (17, 17),
        (40, 7),
        (100, 99),
        (300, 120),
    ] {
        for kind in 0..3 {
            let (a, b) = match kind {
                0 => (
                    rand_nonzero_vec(la, rng.gen()),
                    rand_nonzero_vec(lb, rng.gen()),
                ),
                1 => (vec![u64::MAX; la], vec![u64::MAX; lb]),
                _ => (rand_nonzero_vec(la, rng.gen()), vec![u64::MAX; lb]),
            };
            let c = product(&a, &b);
            assert!(verify_mul(&a, &b, &c), "{la}x{lb} kind {kind}");
            assert!(verify_mul(&b, &a, &c));
            // Extra zero limbs above, and the (out, top) split, don't change the integer.
            let mut padded = c.clone();
            padded.extend([0, 0, 0]);
            assert!(verify_mul(&a, &b, &padded));
            let (out, top) = c.split_at(c.len() - 1);
            assert_eq!(
                Residues::of_with_top(out, top[0]),
                Residues::of(&a).mul(Residues::of(&b))
            );
            let sq = product(&a, &a);
            assert!(verify_sqr(&a, &sq));
        }
    }
    // Zero operands: any all-zero c, including none.
    assert!(verify_mul(&[], &[1, 2], &[]));
    assert!(verify_mul(&[0, 0], &[5], &[0, 0, 0]));
    assert!(!verify_mul(&[0, 0], &[5], &[1]));
}

#[test]
fn test_verify_rejects_every_bit_flip() {
    let mut rng = StdRng::seed_from_u64(0x7e54);
    let a = rand_nonzero_vec(5, rng.gen());
    let b = rand_nonzero_vec(4, rng.gen());
    let c = product(&a, &b);
    for bit in 0..64 * c.len() {
        let mut bad = c.clone();
        bad[bit / 64] ^= 1 << (bit % 64);
        assert!(!verify_mul(&a, &b, &bad), "flip of bit {bit}");
    }
}

// The errors a wrong FFT leaves: ±1 in a 16-bit coefficient, alone, in pairs (including pairs one
// limb apart, which cancel mod B − 1), and in clusters, plus overwritten limbs.
#[test]
fn test_verify_rejects_coefficient_errors() {
    let mut rng = StdRng::seed_from_u64(0x7e55);
    let a = rand_nonzero_vec(600, rng.gen());
    let b = rand_nonzero_vec(500, rng.gen());
    let c = product(&a, &b);
    let digits = 4 * (c.len() - 1);
    assert!(verify_mul(&a, &b, &c));
    for k in (0..digits).step_by(7) {
        for up in [true, false] {
            let mut bad = c.clone();
            nudge(&mut bad, 16 * k, up);
            assert!(!verify_mul(&a, &b, &bad), "single error at digit {k}");
        }
    }
    for _ in 0..500 {
        let k1 = rng.gen_range(0..digits - 4);
        let k2 = if rng.gen() {
            k1 + 4
        } else {
            rng.gen_range(0..digits)
        };
        if k1 == k2 {
            continue;
        }
        let mut bad = c.clone();
        nudge(&mut bad, 16 * k1, true);
        nudge(&mut bad, 16 * k2, false);
        assert!(!verify_mul(&a, &b, &bad), "pair at digits {k1}, {k2}");
    }
    for _ in 0..200 {
        let mut bad = c.clone();
        for _ in 0..rng.gen_range(3..30) {
            nudge(&mut bad, 16 * rng.gen_range(0..digits), rng.gen());
        }
        if bad != c {
            assert!(!verify_mul(&a, &b, &bad), "cluster");
        }
        let mut bad = c.clone();
        let i = rng.gen_range(0..c.len());
        bad[i] = bad[i].wrapping_add(rng.gen_range(1..u64::MAX));
        assert!(!verify_mul(&a, &b, &bad), "limb {i} overwritten");
    }
}
