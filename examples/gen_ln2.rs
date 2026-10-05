// Writes floor(ln(2) * B^limbs) to src/utils/consts/ln2.bin as little-endian
// u64 limbs, least significant first, for the static ln(2) cache to embed.
//
//   cargo run --release --example gen_ln2 [-- <limbs>]
//
// The cache sizes its table from the file, so any limb count works.

use big_bits::utils::consts::consts::ln2_dyn;
use std::{env, fs};

const DEFAULT_LIMBS: usize = 1024;
const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/utils/consts/ln2.bin");

fn main() {
    let limbs = env::args()
        .nth(1)
        .map(|s| s.parse().expect("limb count must be a positive integer"))
        .unwrap_or(DEFAULT_LIMBS);
    assert!(limbs > 0, "limb count must be a positive integer");

    let mut ln2 = vec![0u64; limbs];
    ln2_dyn(&mut ln2);

    let bytes: Vec<u8> = ln2.iter().flat_map(|l| l.to_le_bytes()).collect();
    fs::write(PATH, &bytes).expect("failed to write ln2.bin");
    println!("wrote {limbs} limbs ({} bytes) to {PATH}", bytes.len());
}
