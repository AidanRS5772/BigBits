//! Allocation-free timed loops for profiling the production binomial sqrt.
//! Example: cargo bench --profile prof --bench binom_sqrt_profile --
//! --limbs 16 --shape full --pattern random --mode binom --seconds 3

use big_bits::utils::sqrt::{
    binom_sqrt, binom_sqrt_core, sqrt_approx_dyn, sqrt_approx_static, sqrt_dyn, sqrt_only_dyn,
    sqrt_only_static, sqrt_static,
};
use big_bits::utils::{
    DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF, DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF,
    DYN_SQRT_REM_ZIMMERMAN_CUTOFF, STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF,
    STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF, STATIC_SQRT_REM_ZIMMERMAN_CUTOFF,
};
use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

type Sqrt = fn(&mut [u64], &mut [u64]);

fn copy_only(x: &mut [u64], s: &mut [u64]) {
    black_box((x, s));
}

fn static_entry<const N: usize>(mode: &str) -> Sqrt {
    match mode {
        "zimmer-static" => sqrt_static::<N>,
        "zimmer-static-only" => sqrt_only_static::<N>,
        "zimmer-static-approx" => sqrt_approx_static::<N>,
        _ => unreachable!(),
    }
}

fn main() {
    let mut root_len = 16usize;
    let mut shape = String::from("full");
    let mut pattern = String::from("random");
    let mut mode = String::from("binom");
    let mut seconds = 3.0f64;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--limbs" => root_len = args.next().expect("--limbs value").parse().unwrap(),
            "--shape" => shape = args.next().expect("--shape value"),
            "--pattern" => pattern = args.next().expect("--pattern value"),
            "--mode" => mode = args.next().expect("--mode value"),
            "--seconds" => seconds = args.next().expect("--seconds value").parse().unwrap(),
            _ => panic!("unknown option {arg}"),
        }
    }
    assert!(root_len > 0 && seconds.is_finite() && seconds > 0.0);
    let x_len = match shape.as_str() {
        "full" => 2 * root_len,
        "padded" => root_len + 1,
        _ => panic!("shape must be full or padded"),
    };
    let entry: Sqrt = match mode.as_str() {
        "binom" => binom_sqrt,
        "core" => binom_sqrt_core,
        "only" => sqrt_only_dyn,
        "approx" => sqrt_approx_dyn,
        "zimmer" => sqrt_dyn,
        "zimmer-only" => sqrt_only_dyn,
        "zimmer-approx" => sqrt_approx_dyn,
        "zimmer-static" | "zimmer-static-only" | "zimmer-static-approx" => {
            match (2 * root_len).next_power_of_two() {
                64 => static_entry::<64>(&mode),
                128 => static_entry::<128>(&mode),
                256 => static_entry::<256>(&mode),
                512 => static_entry::<512>(&mode),
                1024 => static_entry::<1024>(&mode),
                2048 => static_entry::<2048>(&mode),
                4096 => static_entry::<4096>(&mode),
                8192 => static_entry::<8192>(&mode),
                _ => panic!("static profiling supports root lengths 17..=4096"),
            }
        }
        "copy" => copy_only,
        _ => panic!("unknown sqrt profiling mode {mode}"),
    };
    let cutoff = match mode.as_str() {
        "only" | "zimmer-only" => Some(DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF),
        "approx" | "zimmer-approx" => Some(DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF),
        "zimmer" => Some(DYN_SQRT_REM_ZIMMERMAN_CUTOFF),
        "zimmer-static" => Some(STATIC_SQRT_REM_ZIMMERMAN_CUTOFF),
        "zimmer-static-only" => Some(STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF),
        "zimmer-static-approx" => Some(STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF),
        _ => None,
    };
    if let Some(cutoff) = cutoff {
        if mode.starts_with("zimmer") {
            assert!(root_len >= cutoff, "mode must dispatch to Zimmermann");
        } else {
            assert!(root_len < cutoff, "mode must dispatch to binomial");
        }
    }
    assert!(
        mode != "core" || pattern != "shifted",
        "core requires normalized input"
    );

    // Many deterministic inputs avoid training the predictor on one operand.
    let mut state = 0x5351_5254_5052_4f46u64;
    let inputs: Vec<Vec<u64>> = (0..256)
        .map(|_| {
            let mut x: Vec<u64> = (0..x_len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state
                })
                .collect();
            match pattern.as_str() {
                "random" => x[x_len - 1] |= 1 << 63,
                "shifted" => x[x_len - 1] = 1,
                "ones" => x.fill(u64::MAX),
                "power" => {
                    x.fill(0);
                    x[x_len - 1] = 1 << 62;
                }
                _ => panic!("pattern must be random, shifted, ones, or power"),
            }
            x
        })
        .collect();
    let mut work = vec![0; x_len];
    let mut root = vec![0; root_len];
    let mut run = |duration: Duration| {
        let start = Instant::now();
        let mut operations = 0u64;
        loop {
            for input in &inputs {
                work.copy_from_slice(black_box(input));
                entry(black_box(&mut work), black_box(&mut root));
                black_box(&root);
            }
            operations += inputs.len() as u64;
            if start.elapsed() >= duration {
                break;
            }
        }
        (operations, start.elapsed())
    };
    run(Duration::from_millis(100));
    let (operations, elapsed) = run(Duration::from_secs_f64(seconds));
    println!(
        "RESULT limbs={root_len} shape={shape} pattern={pattern} mode={mode} operations={operations} seconds={:.6} ns_per_op={:.3} checksum={:016x}",
        elapsed.as_secs_f64(),
        elapsed.as_nanos() as f64 / operations as f64,
        black_box(root.iter().chain(&work).fold(0u64, |v, &limb| v.rotate_left(7) ^ limb)),
    );
}
