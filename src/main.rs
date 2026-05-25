#![allow(unused_imports, dead_code)]
use big_bits::utils::div::nr_rcp_core;
use big_bits::utils::mul::{mul_dyn, mul_vec};
use criterion::black_box;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_vec(n: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| rng.gen()).collect()
}

fn main() {
    let n = 64;
    let mut d = random_vec(2 * n);
    let mut rcp = vec![0; n];
    nr_rcp_core(&mut d, &mut rcp);
}
