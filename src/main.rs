#![allow(unused_imports)]
#![allow(dead_code)]

use bit_ops_2::bitfloat_static::PowI;
use bit_ops_2::{
    bitfloat::*, bitfloat_static::*, bitfrac::*, bitint::*, bitint_static::*, make_LN2,
    make_harmonic, make_int, make_rep_LN2, ubitint::*, ubitint_static::*,
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
    let a = BitFloat::from(LN_2);
    let b = BitFloat::from(1.0/LN_2);

    println!("a: {:?}", a);
    println!("b: {:?}", b);
    println!("{:?}", a*b)
}
