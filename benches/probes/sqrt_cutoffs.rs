//! Build using scripts/python/tune_sqrt_cutoffs.py after cargo bench-cutoffs --no-run.
//! Complete-call timings alternate constant-cutoff variants on identical inputs.
pub use big_bits::utils;
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

#[allow(dead_code, unused_imports)]
mod measured {
    include!(env!("SQRT_TUNE_SOURCE"));
}

type Sqrt = fn(&mut [u64], &mut [u64]);

// Linux thread CPU time excludes descheduling; retain wall time separately.
#[repr(C)]
struct Timespec {
    seconds: std::os::raw::c_long,
    nanoseconds: std::os::raw::c_long,
}
extern "C" {
    fn clock_gettime(clock: i32, value: *mut Timespec) -> i32;
}
fn cpu_time() -> u128 {
    let mut time = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    assert_eq!(unsafe { clock_gettime(3, &mut time) }, 0);
    time.seconds as u128 * 1_000_000_000 + time.nanoseconds as u128
}

fn choose<const N: usize, const MODE: u8>(leaf: usize) -> Sqrt {
    macro_rules! entry {
        ($top:expr, $leaf:expr) => {
            if N == 0 {
                measured::run_dyn::<$top, $leaf, MODE>
            } else {
                measured::run_static::<N, $top, $leaf, MODE>
            }
        };
    }
    match leaf {
        0 => entry!({ usize::MAX }, 17),
        1 => entry!(17, 17),
        4 => entry!(3, 4),
        6 => entry!(3, 6),
        8 => entry!(3, 8),
        10 => entry!(3, 10),
        12 => entry!(3, 12),
        13 => entry!(3, 13),
        14 => entry!(3, 14),
        15 => entry!(3, 15),
        16 => entry!(3, 16),
        17 => entry!(3, 17),
        18 => entry!(3, 18),
        19 => entry!(3, 19),
        20 => entry!(3, 20),
        22 => entry!(3, 22),
        24 => entry!(3, 24),
        26 => entry!(3, 26),
        28 => entry!(3, 28),
        32 => entry!(3, 32),
        40 => entry!(3, 40),
        48 => entry!(3, 48),
        64 => entry!(3, 64),
        96 => entry!(3, 96),
        128 => entry!(3, 128),
        4096 => entry!(3, 4096),
        _ => panic!("unsupported leaf cutoff {leaf}"),
    }
}

fn mode_entry<const N: usize>(mode: u8, leaf: usize) -> Sqrt {
    match mode {
        0 => choose::<N, 0>(leaf),
        1 => choose::<N, 1>(leaf),
        2 => choose::<N, 2>(leaf),
        3 => choose::<N, 3>(leaf),
        _ => unreachable!(),
    }
}

fn entry(family: &str, capacity: usize, mode: u8, leaf: usize) -> Sqrt {
    if family == "dyn" {
        return mode_entry::<0>(mode, leaf);
    }
    match capacity {
        8 => mode_entry::<8>(mode, leaf),
        16 => mode_entry::<16>(mode, leaf),
        32 => mode_entry::<32>(mode, leaf),
        64 => mode_entry::<64>(mode, leaf),
        128 => mode_entry::<128>(mode, leaf),
        256 => mode_entry::<256>(mode, leaf),
        512 => mode_entry::<512>(mode, leaf),
        1024 => mode_entry::<1024>(mode, leaf),
        2048 => mode_entry::<2048>(mode, leaf),
        4096 => mode_entry::<4096>(mode, leaf),
        _ => panic!("unsupported capacity {capacity}"),
    }
}

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn inputs(n: usize, shape: &str, pattern: &str, seed: u64, count: usize) -> Vec<Vec<u64>> {
    let width = match shape {
        "full" => 2 * n,
        "near" => 2 * n - 1,
        "padded" => n + 1,
        "mid" => 3 * n / 2 + 1,
        _ => panic!("bad shape"),
    };
    let mut state = seed ^ (n as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ width as u64;
    (0..count)
        .map(|i| {
            let mut x: Vec<_> = (0..width).map(|_| random(&mut state)).collect();
            match pattern {
                "random" => x[width - 1] |= 1 << 63,
                "shifted" => x[width - 1] = (x[width - 1] >> (2 + (i % 31) * 2)) | 1,
                "mixed" => {
                    if i % 2 == 0 {
                        x[width - 1] |= 1 << 63;
                    } else {
                        x[width - 1] = (x[width - 1] >> (2 + (i % 31) * 2)) | 1;
                    }
                }
                "ones" => x.fill(u64::MAX),
                "power" => {
                    x.fill(0);
                    x[width - 1] = 1 << 62;
                }
                _ => panic!("bad pattern"),
            }
            x
        })
        .collect()
}

fn time(f: Sqrt, inputs: &[Vec<u64>], n: usize, duration: Duration) -> (f64, f64) {
    let mut x = vec![0; inputs[0].len()];
    let mut s = vec![0; n];
    let cpu_start = cpu_time();
    let start = Instant::now();
    let mut count = 0;
    loop {
        for input in inputs {
            x.copy_from_slice(black_box(input));
            f(black_box(&mut x), black_box(&mut s));
            black_box((&x, &s));
        }
        count += inputs.len();
        if start.elapsed() >= duration {
            break;
        }
    }
    let wall = start.elapsed().as_nanos();
    let cpu = cpu_time() - cpu_start;
    (cpu as f64 / count as f64, wall as f64 / count as f64)
}

fn validate(f: Sqrt, values: &[Vec<u64>], n: usize, mode: u8) {
    for value in values {
        let mut reference = value.clone();
        let mut floor = vec![0; n];
        utils::sqrt::binom_sqrt(&mut reference, &mut floor);
        let mut x = value.clone();
        let mut s = vec![u64::MAX; n];
        f(&mut x, &mut s);
        if mode == 2 {
            let mut ceil = floor.clone();
            let overflow = utils::utils::inc_buf(&mut ceil);
            assert!(s == floor || (!overflow && s == ceil), "approx root n={n}");
        } else {
            assert_eq!(s, floor, "root n={n}");
            if mode == 0 {
                assert_eq!(x, reference, "remainder n={n}");
            }
            if mode == 3 {
                assert!(
                    utils::utils::eq_buf(&x[..n + 1], &reference),
                    "core remainder n={n}"
                );
            }
        }
    }
}

fn numbers(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in s.split(',') {
        if let Some((a, b)) = part.split_once(':') {
            out.extend(a.parse::<usize>().unwrap()..=b.parse().unwrap());
        } else {
            out.push(part.parse().unwrap());
        }
    }
    out
}

fn main() {
    let mut sizes = numbers("8:64,80,96,128,192,256,512,1024");
    let mut leaves = numbers("0,1,8,12,16,17,20,24,32,48,64,4096");
    let mut families = String::from("dyn,static");
    let mut modes = String::from("rem,only,approx");
    let mut shapes = String::from("full");
    let mut pattern = String::from("mixed");
    let mut rounds = 5usize;
    let mut ms = 10u64;
    let mut seed = 0x5351525454554e45u64;
    let mut capacity = 0usize;
    let mut count = 32usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().expect("missing value");
        match arg.as_str() {
            "--sizes" => sizes = numbers(&value),
            "--leaves" => leaves = numbers(&value),
            "--families" => families = value,
            "--modes" => modes = value,
            "--shapes" => shapes = value,
            "--pattern" => pattern = value,
            "--rounds" => rounds = value.parse().unwrap(),
            "--ms" => ms = value.parse().unwrap(),
            "--seed" => seed = value.parse().unwrap(),
            "--capacity" => capacity = value.parse().unwrap(),
            "--count" => count = value.parse().unwrap(),
            _ => panic!("unknown {arg}"),
        }
    }
    assert!(sizes.iter().all(|&n| n >= 3));
    println!("family,mode,shape,pattern,root_limbs,capacity,leaf,round,ns,wall_ns");
    let mut checks = 0;
    for family in families.split(',') {
        assert!(family == "dyn" || family == "static");
        for &n in &sizes {
            let cap = if family == "dyn" {
                0
            } else if capacity == 0 {
                (2 * n).next_power_of_two()
            } else {
                capacity
            };
            assert!(family == "dyn" || cap >= 2 * n);
            for shape in shapes.split(',') {
                let values = inputs(n, shape, &pattern, seed, count);
                for mode_name in modes.split(',') {
                    let mode = match mode_name {
                        "rem" => 0,
                        "only" => 1,
                        "approx" => 2,
                        "core" => 3,
                        _ => panic!("mode"),
                    };
                    assert!(mode != 3 || (shape == "full" && pattern == "random"));
                    let candidates: Vec<_> = leaves
                        .iter()
                        .map(|&leaf| (leaf, entry(family, cap, mode, leaf)))
                        .collect();
                    for &(_, f) in &candidates {
                        validate(f, &values, n, mode);
                        checks += values.len();
                        time(f, &values, n, Duration::from_millis(1));
                    }
                    for round in 0..rounds {
                        for step in 0..candidates.len() {
                            let (leaf, f) = candidates[(round + step) % candidates.len()];
                            let (ns, wall_ns) = time(f, &values, n, Duration::from_millis(ms));
                            println!("{family},{mode_name},{shape},{pattern},{n},{cap},{leaf},{round},{ns:.3},{wall_ns:.3}");
                        }
                    }
                }
            }
        }
    }
    eprintln!("PASS: {checks} candidate checks against binomial root/remainder reference");
}
