use big_bits::utils::div::{div_dyn, div_rem_dyn, div_rem_static, div_static, rcp_dyn, rcp_static};
use std::{
    env,
    hint::black_box,
    process,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Model {
    Dynamic,
    Static,
}

impl Model {
    const fn label(self) -> &'static str {
        match self {
            Self::Dynamic => "dynamic",
            Self::Static => "static",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Operation {
    Div,
    DivRem,
    Rcp,
}

impl Operation {
    const fn label(self) -> &'static str {
        match self {
            Self::Div => "div",
            Self::DivRem => "div_rem",
            Self::Rcp => "rcp",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ProfileCase {
    name: &'static str,
    model: Model,
    operation: Operation,
    divisor_limbs: usize,
    output_limbs: usize,
    capacity: usize,
    expected_path: &'static str,
    batch: u64,
}

const fn dynamic_case(
    name: &'static str,
    operation: Operation,
    divisor_limbs: usize,
    output_limbs: usize,
    expected_path: &'static str,
    batch: u64,
) -> ProfileCase {
    ProfileCase {
        name,
        model: Model::Dynamic,
        operation,
        divisor_limbs,
        output_limbs,
        capacity: 0,
        expected_path,
        batch,
    }
}

const fn static_case(
    name: &'static str,
    operation: Operation,
    divisor_limbs: usize,
    output_limbs: usize,
    capacity: usize,
    expected_path: &'static str,
    batch: u64,
) -> ProfileCase {
    ProfileCase {
        name,
        model: Model::Static,
        operation,
        divisor_limbs,
        output_limbs,
        capacity,
        expected_path,
        batch,
    }
}

// Cases deliberately cover:
// - the Knuth/BZ divisor boundary;
// - NR and BZ in both Karatsuba- and transform-sized dispatch regions;
// - the allocation-model-specific reciprocal boundary;
// - dynamic FFT and static NTT middle/full products.
const CASES: &[ProfileCase] = &[
    dynamic_case("dyn_div_knuth", Operation::Div, 88, 88, "knuth", 512),
    dynamic_case(
        "dyn_div_nr_school",
        Operation::Div,
        89,
        89,
        "nr/school-middle",
        256,
    ),
    dynamic_case(
        "dyn_div_bz_karatsuba",
        Operation::Div,
        128,
        512,
        "bz/karatsuba-region",
        32,
    ),
    dynamic_case(
        "dyn_div_nr_fft",
        Operation::Div,
        2048,
        2048,
        "nr/fft-middle",
        1,
    ),
    dynamic_case(
        "dyn_div_nr_ntt",
        Operation::Div,
        65536,
        65536,
        "nr/ntt-middle",
        1,
    ),
    dynamic_case(
        "dyn_div_bz_fft_region",
        Operation::Div,
        1024,
        8192,
        "bz/fft-region",
        1,
    ),
    dynamic_case("dyn_div_rem_knuth", Operation::DivRem, 88, 88, "knuth", 512),
    dynamic_case(
        "dyn_div_rem_nr_school",
        Operation::DivRem,
        89,
        89,
        "nr/school-middle",
        256,
    ),
    dynamic_case(
        "dyn_div_rem_bz_karatsuba",
        Operation::DivRem,
        128,
        256,
        "bz/karatsuba-region",
        64,
    ),
    dynamic_case(
        "dyn_div_rem_nr_fft",
        Operation::DivRem,
        2048,
        2048,
        "nr/fft-middle",
        1,
    ),
    dynamic_case(
        "dyn_div_rem_nr_ntt",
        Operation::DivRem,
        65536,
        65536,
        "nr/ntt-middle",
        1,
    ),
    dynamic_case(
        "dyn_div_rem_bz_fft_region",
        Operation::DivRem,
        1024,
        4096,
        "bz/fft-region",
        1,
    ),
    dynamic_case(
        "dyn_rcp_knuth_boundary",
        Operation::Rcp,
        9,
        8,
        "knuth",
        2048,
    ),
    dynamic_case(
        "dyn_rcp_nr_boundary",
        Operation::Rcp,
        10,
        9,
        "nr/school-middle",
        2048,
    ),
    dynamic_case(
        "dyn_rcp_nr_fft",
        Operation::Rcp,
        2049,
        2048,
        "nr/fft-middle",
        2,
    ),
    dynamic_case(
        "dyn_rcp_nr_ntt",
        Operation::Rcp,
        65537,
        65536,
        "nr/ntt-middle",
        1,
    ),
    static_case(
        "static_div_knuth",
        Operation::Div,
        88,
        88,
        256,
        "knuth",
        512,
    ),
    static_case(
        "static_div_nr_school",
        Operation::Div,
        89,
        89,
        256,
        "nr/school-middle",
        256,
    ),
    static_case(
        "static_div_bz_karatsuba",
        Operation::Div,
        128,
        256,
        512,
        "bz/karatsuba-region",
        64,
    ),
    static_case(
        "static_div_nr_ntt",
        Operation::Div,
        2048,
        2048,
        4096,
        "nr/ntt-middle",
        1,
    ),
    static_case(
        "static_div_bz_ntt_region",
        Operation::Div,
        2048,
        8192,
        16384,
        "bz/ntt-region",
        1,
    ),
    static_case(
        "static_div_rem_knuth",
        Operation::DivRem,
        88,
        88,
        256,
        "knuth",
        512,
    ),
    static_case(
        "static_div_rem_nr_school",
        Operation::DivRem,
        89,
        89,
        256,
        "nr/school-middle",
        256,
    ),
    static_case(
        "static_div_rem_bz_karatsuba",
        Operation::DivRem,
        128,
        256,
        512,
        "bz/karatsuba-region",
        64,
    ),
    static_case(
        "static_div_rem_nr_ntt",
        Operation::DivRem,
        2048,
        2048,
        4096,
        "nr/ntt-middle",
        1,
    ),
    static_case(
        "static_div_rem_bz_ntt_region",
        Operation::DivRem,
        2048,
        4096,
        8192,
        "bz/ntt-region",
        1,
    ),
    static_case(
        "static_rcp_knuth_boundary",
        Operation::Rcp,
        101,
        100,
        128,
        "knuth",
        256,
    ),
    static_case(
        "static_rcp_nr_boundary",
        Operation::Rcp,
        102,
        101,
        128,
        "nr/school-middle",
        256,
    ),
    static_case(
        "static_rcp_nr_ntt",
        Operation::Rcp,
        2049,
        2048,
        4096,
        "nr/ntt-middle",
        1,
    ),
];

struct Workload {
    case: ProfileCase,
    numerator_template: Vec<u64>,
    numerator: Vec<u64>,
    divisor: Vec<u64>,
    output: Vec<u64>,
}

impl Workload {
    fn new(case: ProfileCase) -> Self {
        let numerator_len = match case.operation {
            Operation::Div | Operation::DivRem => case.divisor_limbs + case.output_limbs - 1,
            Operation::Rcp => 0,
        };
        let numerator_template = deterministic_limbs(numerator_len, 0x1234_5678_9abc_def0);
        let numerator = numerator_template.clone();
        let divisor = deterministic_limbs(case.divisor_limbs, 0xfedc_ba98_7654_3210);
        let output = vec![0; case.output_limbs];
        Self {
            case,
            numerator_template,
            numerator,
            divisor,
            output,
        }
    }

    #[inline(never)]
    fn execute(&mut self) {
        match self.case.model {
            Model::Dynamic => match self.case.operation {
                Operation::Div => div_dyn(
                    black_box(&self.numerator),
                    black_box(&self.divisor),
                    black_box(&mut self.output),
                ),
                Operation::DivRem => {
                    self.numerator.copy_from_slice(&self.numerator_template);
                    div_rem_dyn(
                        black_box(&mut self.numerator),
                        black_box(&self.divisor),
                        black_box(&mut self.output),
                    );
                }
                Operation::Rcp => rcp_dyn(black_box(&self.divisor), black_box(&mut self.output)),
            },
            Model::Static => match self.case.capacity {
                128 => self.execute_static::<128>(),
                256 => self.execute_static::<256>(),
                512 => self.execute_static::<512>(),
                1024 => self.execute_static::<1024>(),
                2048 => self.execute_static::<2048>(),
                4096 => self.execute_static::<4096>(),
                8192 => self.execute_static::<8192>(),
                16384 => self.execute_static::<16384>(),
                capacity => panic!("unsupported static capacity {capacity}"),
            },
        }
    }

    #[inline(never)]
    fn execute_static<const N: usize>(&mut self) {
        match self.case.operation {
            Operation::Div => div_static::<N>(
                black_box(&self.numerator),
                black_box(&self.divisor),
                black_box(&mut self.output),
            ),
            Operation::DivRem => {
                self.numerator.copy_from_slice(&self.numerator_template);
                div_rem_static::<N>(
                    black_box(&mut self.numerator),
                    black_box(&self.divisor),
                    black_box(&mut self.output),
                );
            }
            Operation::Rcp => {
                rcp_static::<N>(black_box(&self.divisor), black_box(&mut self.output))
            }
        }
    }

    fn checksum(&self) -> u64 {
        self.output
            .iter()
            .chain(self.numerator.iter())
            .fold(0u64, |sum, &limb| sum.rotate_left(7) ^ limb)
    }
}

fn deterministic_limbs(len: usize, mut state: u64) -> Vec<u64> {
    let mut limbs = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        limbs.push(state);
    }
    if let Some(first) = limbs.first_mut() {
        *first |= 1;
    }
    if let Some(last) = limbs.last_mut() {
        *last |= 1 << 63;
    }
    limbs
}

fn run_for(workload: &mut Workload, duration: Duration) -> (u64, Duration) {
    let start = Instant::now();
    let mut operations = 0u64;
    loop {
        for _ in 0..workload.case.batch {
            workload.execute();
            operations += 1;
        }
        if start.elapsed() >= duration {
            break;
        }
    }
    (operations, start.elapsed())
}

fn print_cases() {
    println!("name\tmodel\toperation\tdivisor_limbs\toutput_limbs\tcapacity\texpected_path");
    for case in CASES {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            case.name,
            case.model.label(),
            case.operation.label(),
            case.divisor_limbs,
            case.output_limbs,
            case.capacity,
            case.expected_path
        );
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: div_profile --list | --case NAME [--seconds N] [--warmup-ms N]\n\
         profiles one deterministic end-to-end production dispatcher"
    );
    process::exit(2);
}

fn main() {
    let mut args = env::args().skip(1);
    let mut case_name = None;
    let mut seconds = 3.0f64;
    let mut warmup_ms = 100u64;
    let mut list = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--list" => list = true,
            "--case" => case_name = args.next(),
            "--seconds" => {
                seconds = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--warmup-ms" => {
                warmup_ms = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            _ => usage(),
        }
    }

    if list {
        print_cases();
        return;
    }

    let case_name = case_name.unwrap_or_else(|| usage());
    let case = CASES
        .iter()
        .copied()
        .find(|case| case.name == case_name)
        .unwrap_or_else(|| {
            eprintln!("unknown profile case: {case_name}");
            print_cases();
            process::exit(2);
        });
    if !seconds.is_finite() || seconds <= 0.0 {
        usage();
    }

    println!(
        "PROFILE case={} model={} operation={} divisor_limbs={} output_limbs={} capacity={} expected_path={}",
        case.name,
        case.model.label(),
        case.operation.label(),
        case.divisor_limbs,
        case.output_limbs,
        case.capacity,
        case.expected_path,
    );

    let mut workload = Workload::new(case);
    workload.execute();
    if warmup_ms != 0 {
        let _ = run_for(&mut workload, Duration::from_millis(warmup_ms));
    }
    let (operations, elapsed) = run_for(&mut workload, Duration::from_secs_f64(seconds));
    let checksum = black_box(workload.checksum());
    println!(
        "RESULT operations={} elapsed_seconds={:.6} operations_per_second={:.3} checksum={checksum:016x}",
        operations,
        elapsed.as_secs_f64(),
        operations as f64 / elapsed.as_secs_f64(),
    );
}
