#![allow(unused_imports, dead_code)]
use big_bits::utils::mul::{dyn_dispatch, mul_dyn, mul_vec};
use criterion::black_box;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// fn random_vec(n: usize) -> Vec<u64> {
//     let mut rng = StdRng::seed_from_u64(0);
//     (0..n).map(|_| rng.gen()).collect()
// }

fn main() {
    let m = 5;
    let n = 255;
    for v in m..n {
        println!("val = {v} -> {:?}", dyn_dispatch(v, v))
    }
}
