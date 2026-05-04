use big_bits::utils::mul::ntt_entry_dyn;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    long_len: usize,
    short_len: usize,
    default_iters: usize,
}

const CASES: &[Case] = &[
    Case {
        name: "small-mixed",
        long_len: 192,
        short_len: 192,
        default_iters: 4_000,
    },
    Case {
        name: "radix2-balanced",
        long_len: 2_048,
        short_len: 2_048,
        default_iters: 800,
    },
    Case {
        name: "radix3-balanced",
        long_len: 24_576,
        short_len: 24_576,
        default_iters: 80,
    },
    Case {
        name: "radix5-balanced",
        long_len: 40_960,
        short_len: 40_960,
        default_iters: 45,
    },
    Case {
        name: "unbalanced-16x",
        long_len: 65_536,
        short_len: 4_096,
        default_iters: 90,
    },
    Case {
        name: "large-radix3",
        long_len: 98_304,
        short_len: 98_304,
        default_iters: 16,
    },
];

fn rand_vec(len: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen::<u64>()).collect()
}

fn run_case(case: Case, iters_override: Option<usize>) -> u64 {
    let iters = iters_override.unwrap_or(case.default_iters);
    let batch = if case.long_len >= 65_536 { 2 } else { 4 };
    let inputs: Vec<(Vec<u64>, Vec<u64>)> = (0..batch)
        .map(|i| {
            (
                rand_vec(case.long_len, 0x9e37_79b9_u64.wrapping_add(i as u64)),
                rand_vec(case.short_len, 0xc2b2_ae35_u64.wrapping_add(i as u64)),
            )
        })
        .collect();
    let mut out = vec![0u64; case.long_len + case.short_len - 1];

    for (a, b) in &inputs {
        let carry = ntt_entry_dyn(black_box(a), black_box(b), black_box(&mut out));
        black_box(carry);
    }

    let start = Instant::now();
    let mut checksum = 0u64;
    for i in 0..iters {
        let (a, b) = &inputs[i % inputs.len()];
        let carry = ntt_entry_dyn(black_box(a), black_box(b), black_box(&mut out));
        checksum ^= carry;
        checksum ^= out[i % out.len()];
        black_box(&out);
    }
    let elapsed = start.elapsed();
    let ns_per_iter = elapsed.as_nanos() / iters as u128;
    eprintln!(
        "{:<18} long={:<7} short={:<7} out={:<8} iters={:<5} total={:?} ns/iter={}",
        case.name,
        case.long_len,
        case.short_len,
        out.len(),
        iters,
        elapsed,
        ns_per_iter
    );
    black_box(checksum)
}

fn print_usage() {
    eprintln!("usage: profile_ntt [--list] [--case NAME|all] [--iters N]");
    eprintln!("cases:");
    for case in CASES {
        eprintln!(
            "  {:<18} long={:<7} short={:<7} default_iters={}",
            case.name, case.long_len, case.short_len, case.default_iters
        );
    }
}

fn main() {
    let mut case_name = "all";
    let mut iters_override = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--list" => {
                print_usage();
                return;
            }
            "--case" => case_name = args.next().expect("--case requires a name").leak(),
            "--iters" => {
                let iters = args
                    .next()
                    .expect("--iters requires a value")
                    .parse()
                    .expect("--iters must be a positive integer");
                iters_override = Some(iters);
            }
            _ => {
                print_usage();
                panic!("unknown argument: {arg}");
            }
        }
    }

    let mut checksum = 0u64;
    if case_name == "all" {
        for &case in CASES {
            checksum ^= run_case(case, iters_override);
        }
    } else {
        let case = CASES
            .iter()
            .copied()
            .find(|case| case.name == case_name)
            .unwrap_or_else(|| panic!("unknown case: {case_name}"));
        checksum ^= run_case(case, iters_override);
    }
    println!("{checksum}");
}
