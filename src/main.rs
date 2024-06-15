#![allow(unused_imports)]
#![allow(dead_code)]

use bit_ops_2::{
    bitfloat::*, bitfrac::*, bitint::*, bitint_static::*, ubitint::*, ubitint_static::*,
};
use criterion::measurement::ValueFormatter;
use rand::{rngs::OsRng, rngs::StdRng, Rng, RngCore, SeedableRng};
use std::arch::asm;
use std::f64::consts::*;
use std::fs;
use std::ptr::copy_nonoverlapping;
use std::result;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;

fn print_seed() {
    let mut rng = rand::thread_rng();
    let seed: [u8; 32] = rng.gen();

    // Print the seed in the desired format
    print!("[");
    for (i, byte) in seed.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("0x{:02X}", byte);
    }
    println!("];");
}

fn main() {
    let mut bf = BitFloat::from(PI);

    recipricol(&mut bf);
    println!("{:?}", bf);
}
