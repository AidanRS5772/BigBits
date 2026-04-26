#![allow(unused_imports, dead_code)]
use big_bits::utils::mul::{fft_entry, ntt_entry_dyn};
use criterion::black_box;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_vec(n: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.gen()).collect()
}

fn main() {
    let n = 25 * 243 * (1 << 6);
    let a = random_vec(n);
    let b = random_vec(n);
    let mut out = vec![0;2*n];
    ntt_entry_dyn(&a, &b, &mut out);
}
