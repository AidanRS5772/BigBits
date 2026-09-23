//! Line-oriented adapter for independent arbitrary-precision sqrt checks.
//! Input: decimal root width followed by little-endian hexadecimal input limbs.
//! Output: little-endian root limbs, then '|', then remainder limbs (hex).
//! Build with rustc --edition=2021 -O and --extern big_bits=... as in the seed probe.
use big_bits::utils::sqrt::binom_sqrt;
use std::io::{self, BufRead, Write};

fn main() {
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let mut fields = line.split_whitespace();
        let root_len: usize = fields.next().unwrap().parse().unwrap();
        let mut x: Vec<_> = fields
            .map(|v| u64::from_str_radix(v, 16).unwrap())
            .collect();
        let mut root = vec![u64::MAX; root_len];
        binom_sqrt(&mut x, &mut root);
        for limb in root {
            write!(output, "{limb:x} ").unwrap();
        }
        write!(output, "|").unwrap();
        for limb in x {
            write!(output, " {limb:x}").unwrap();
        }
        writeln!(output).unwrap();
    }
}
