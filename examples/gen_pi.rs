// Writes floor(pi * B^(limbs - 1)) to src/utils/consts/pi.bin as little-endian
// u64 limbs, least significant first, for the static pi cache to embed.
//
//   cargo run --release --example gen_pi [-- <limbs>]
//
// The cache sizes its table from the file, so any limb count works.

use big_bits::utils::consts::consts::pi_dyn;
use std::{env, fs};

const DEFAULT_LIMBS: usize = 1024;
const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/utils/consts/pi.bin");

fn main() {
    let limbs = env::args()
        .nth(1)
        .map(|s| s.parse().expect("limb count must be a positive integer"))
        .unwrap_or(DEFAULT_LIMBS);
    assert!(limbs > 0, "limb count must be a positive integer");

    // pi_dyn copies requests up to the current table's length from it, so ask for one limb
    // more than the table gets: from the current length on, that builds the series.
    let mut pi = vec![0u64; limbs + 1];
    pi_dyn(&mut pi);

    let bytes: Vec<u8> = pi[1..].iter().flat_map(|l| l.to_le_bytes()).collect();
    fs::write(PATH, &bytes).expect("failed to write pi.bin");
    println!("wrote {limbs} limbs ({} bytes) to {PATH}", bytes.len());
}
