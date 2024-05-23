#![allow(unused_imports)]

use bit_ops_2::{bitfloat::* , bitfrac::*, bitint::*, ubitint::*, ubitint_static::*, bitint_static::*};
use rand::{rngs::OsRng, rngs::StdRng, Rng, RngCore, SeedableRng};
use std::arch::asm;
use std::f64::consts::*;
use std::fs;
use std::ptr::copy_nonoverlapping;
use std::result;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;

pub fn print_seed() {
    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);

    // Print the random seed as a Rust array
    print!("[");
    for (i, byte) in seed.iter().enumerate() {
        print!("0x{:02X}", byte);
        if i < 31 {
            print!(", ");
        }
    }
    println!("];");
}

fn main() {
    let a = BitIntStatic::<3>::from(8);
    let b = BitIntStatic::<3>::from(-3);

    println!("{:?}", a+b);
}
