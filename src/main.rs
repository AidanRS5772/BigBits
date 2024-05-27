#![allow(unused_imports)]
#![allow(dead_code)]

use bit_ops_2::ubitint_static::Pow;
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

fn print_seed(){
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
    let a = BitIntStatic::<6>::from(2);
    let b = BitIntStatic::<6>::from(-12);

    println!("{:?}", a % b)
}
