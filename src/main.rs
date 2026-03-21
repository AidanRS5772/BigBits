use big_bits::utils::mul::mul_vec;
use rand::Rng;

fn random_limbs(n: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}

fn main() {
    let a = random_limbs(45);
    let b = random_limbs(45);

    for _ in 0..10_000 {
        let (mut c, carry) = std::hint::black_box(mul_vec(&a, &b));
        c.push(carry);
    }
}
