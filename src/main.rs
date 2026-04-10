#![allow(unused_imports)]
use big_bits::utils::mul::fft_entry;
use criterion::black_box;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_vec(n: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.gen()).collect()
}

fn main() {}
