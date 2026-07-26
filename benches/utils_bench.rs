#![allow(dead_code)]

use big_bits::{utils::div::*, *};
use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion, Throughput,
};
use rand::Rng;

fn random_limbs(n: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}

fn random_normalized_limbs(n: usize) -> Vec<u64> {
    let mut limbs = random_limbs(n);
    if let Some(last) = limbs.last_mut() {
        *last |= 1 << 63;
    }
    limbs
}

fn random_unnormalized_limbs(n: usize) -> Vec<u64> {
    let mut limbs = random_limbs(n);
    if let Some(last) = limbs.last_mut() {
        *last = (*last & ((1 << 48) - 1)) | (1 << 47);
    }
    limbs
}

fn random_sh() -> u8 {
    let mut rng = rand::thread_rng();
    rng.gen_range(1..64)
}

fn random_div() -> u64 {
    let mut rng = rand::thread_rng();
    rng.gen::<u64>() | 1
}

const ARCH: &'static str = std::env::consts::ARCH;

fn bz_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        bz_div_wrapper_dyn(n, d, q, request);
    }
}

fn nr_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        nr_div_wrapper_dyn(n, d, q, request);
    }
}

fn nr_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        nr_div_rem_wrapper_dyn(n, d, q, request);
    }
}

fn nr_rcp_dyn(d: &[u64], rcp: &mut [u64]) {
    if reciprocal_preflight(d, rcp) {
        nr_rcp_wrapper_dyn(d, rcp);
    }
}

fn knuth_rcp_dyn(d: &[u64], rcp: &mut [u64]) {
    if reciprocal_preflight(d, rcp) {
        knuth_rcp_wrapper_dyn(d, rcp);
    }
}

fn set_up_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(250); // default is 100
    group.sampling_mode(criterion::SamplingMode::Flat);
    group.warm_up_time(std::time::Duration::from_secs(5)); // default is 3s
    group.measurement_time(std::time::Duration::from_secs(15));
}

macro_rules! bench_static_sizes {
    ($group:expr, $fn:ident, $(($n:literal, $N:literal)),*) => {
        $(
            $group.throughput(Throughput::Elements($n as u64));
            $group.bench_with_input(BenchmarkId::from_parameter($n), &$n, |bench, &_| {
                let a = random_limbs($n);
                let b = random_limbs($n);
                let mut out = vec![0u64; 2 * $n - 1];
                bench.iter(|| {
                    $fn::<$N>(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut out),
                    )
                });
            });
        )*
    };
}

macro_rules! bench_static_sqr_sizes {
    ($group:expr, $fn:ident, $(($n:literal, $N:literal)),*) => {
        $(
            $group.throughput(Throughput::Elements($n as u64));
            $group.bench_with_input(BenchmarkId::from_parameter($n), &$n, |bench, &_| {
                let a = random_limbs($n);
                let mut out = vec![0u64; 2 * $n - 1];
                bench.iter(|| {
                    $fn::<$N>(
                        black_box(&a),
                        black_box(&mut out),
                    )
                });
            });
        )*
    };
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("add_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let short = random_limbs(n);
            let long = random_limbs(n + 1);
            bench.iter_batched_ref(
                || long.clone(),
                |long| add_buf(black_box(long.as_mut_slice()), black_box(&short)),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_sub(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("sub_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let short = random_limbs(n);
            let long = random_limbs(n + 1);
            bench.iter_batched_ref(
                || long.clone(),
                |long| sub_buf(black_box(long.as_mut_slice()), black_box(&short)),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_shl(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("shl_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let src = random_limbs(n);
            let sh = random_sh();
            bench.iter_batched_ref(
                || src.clone(),
                |buf| shl_buf(black_box(buf.as_mut_slice()), black_box(sh)),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_shr(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("shr_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let src = random_limbs(n);
            let sh = random_sh();
            bench.iter_batched_ref(
                || src.clone(),
                |buf| shr_buf(black_box(buf.as_mut_slice()), black_box(sh)),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_school_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("school_mul_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let b = random_limbs(n);
            let mut out = vec![0; 2 * n - 1];
            bench.iter(|| {
                mul_buf(black_box(&a), black_box(&b), black_box(&mut out));
            });
        });
    }
    group.finish();
}

fn bench_karatsuba_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("karatsuba_mul_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let b = random_limbs(n);
            let mut out = vec![0; 2 * n - 1];
            bench.iter(|| karatsuba_entry_dyn(black_box(&a), black_box(&b), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_fft_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("fft_mul_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let b = random_limbs(n);
            let mut out = vec![0; 2 * n - 1];
            bench.iter(|| fft_entry(black_box(&a), black_box(&b), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_ntt_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("ntt_mul_buf/{ARCH}"));
    set_up_group(&mut group);
    let mut sizes: Vec<usize> = vec![1 << 5, 1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10];
    sizes.iter_mut().for_each(|x| *x *= 27 * 25);
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let b = random_limbs(n);
            let mut out = vec![0; 2 * n - 1];
            bench.iter(|| ntt_entry_dyn(black_box(&a), black_box(&b), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_gen_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("gen_mul_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![1 << 3, 1 << 6, 1 << 9, 1 << 12, 1 << 15, 1 << 18];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let b = random_limbs(n);
            let mut out = vec![0; 2 * n - 1];
            bench.iter(|| mul_dyn(black_box(&a), black_box(&b), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_static_karatsuba_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("static_karatsuba_mul_buf/{ARCH}"));
    set_up_group(&mut group);
    bench_static_sizes!(
        group,
        karatsuba_entry_static,
        (4, 16),
        (16, 64),
        (64, 256),
        (256, 1024),
        (1024, 4096),
        (4096, 16384)
    );
    group.finish();
}

fn bench_static_ntt_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("static_ntt_mul_buf/{ARCH}"));
    set_up_group(&mut group);
    bench_static_sizes!(
        group,
        ntt_entry_static,
        (4, 16),
        (16, 64),
        (64, 256),
        (256, 1024),
        (1024, 4096),
        (4096, 16384)
    );
    group.finish();
}

fn bench_school_sqr(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("school_sqr_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![8, 12, 16, 24, 32];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let mut out = vec![0; 2 * n - 1];
            bench.iter(|| sqr_buf(black_box(&a), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_fft_sqr(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("fft_sqr_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let mut out = vec![0; 2 * n - 1];
            bench.iter(|| fft_sqr_entry(black_box(&a), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_gen_sqr(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("gen_sqr_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let mut out = vec![0; 2 * n - 1];
            bench.iter(|| sqr_dyn(black_box(&a), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_static_karatsuba_sqr(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("static_karatsuba_sqr_buf/{ARCH}"));
    set_up_group(&mut group);
    bench_static_sqr_sizes!(
        group,
        karatsuba_sqr_entry_static,
        (8, 32),
        (12, 48),
        (16, 64),
        (24, 96),
        (32, 128)
    );
    group.finish();
}

fn bench_static_ntt_sqr(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("static_ntt_sqr_buf/{ARCH}"));
    set_up_group(&mut group);
    bench_static_sqr_sizes!(group, ntt_sqr_entry_static, (4, 16), (16, 64), (64, 256));
    group.finish();
}

fn bench_short_school_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("short_school_mul/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![112, 116, 120, 124, 128];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let b = random_limbs(n);
            let mut out = vec![0; n];
            bench.iter(|| short_mul_buf(black_box(&a), black_box(&b), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_short_gen_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("short_gen_mul/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![112, 116, 120, 124, 128];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let b = random_limbs(n);
            let mut out = vec![0; n];
            bench.iter(|| short_mul_dyn(black_box(&a), black_box(&b), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_short_school_sqr(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("short_school_sqr/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![64, 80, 96, 112, 128];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let mut out = vec![0; n];
            bench.iter(|| short_sqr_buf(black_box(&a), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_short_gen_sqr(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("short_gen_sqr/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![64, 80, 96, 112, 128];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let mut out = vec![0; n];
            bench.iter(|| short_sqr_dyn(black_box(&a), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_mid_school(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("mid_school_mul/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![64, 80, 96, 112, 128];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let short = random_limbs(n);
            let long = random_limbs(2 * n - 1);
            let mut out = vec![0; n];
            bench.iter(|| mid_mul_buf(black_box(&long), black_box(&short), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_mid_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("mid_fft_mul/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![64, 80, 96, 112, 128];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let short = random_limbs(n);
            let long = random_limbs(2 * n - 1);
            let mut out = vec![0; n];
            bench.iter(|| fft_mid_mul(black_box(&long), black_box(&short), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_div_prim(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("div_prim/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let src = random_limbs(n);
            let div = random_div();
            bench.iter_batched_ref(
                || src.clone(),
                |buf| div_prim(black_box(buf.as_mut_slice()), black_box(div)),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_knuth_div(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("knuth_div_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let short = random_normalized_limbs(n);
            let long = random_limbs(2 * n);
            bench.iter_batched_ref(
                || (long.clone(), 0u64, vec![0; n + 1]),
                |data| {
                    let (long, of, out) = data;
                    div_buf_of(
                        black_box(long.as_mut_slice()),
                        black_box(of),
                        black_box(short.as_slice()),
                        black_box(out.as_mut_slice()),
                    )
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_bz_div(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("bz_div_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        for (normalization, short) in [
            ("normalized", random_normalized_limbs(n)),
            ("unnormalized", random_unnormalized_limbs(n)),
        ] {
            let long = random_limbs(2 * n);
            group.bench_with_input(BenchmarkId::new(normalization, n), &n, |bench, &_| {
                bench.iter_batched_ref(
                    || vec![0; n + 1],
                    |out| {
                        bz_div_dyn(
                            black_box(long.as_slice()),
                            black_box(short.as_slice()),
                            black_box(out.as_mut_slice()),
                        )
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_nr_div(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("nr_div_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        for (normalization, short) in [
            ("normalized", random_normalized_limbs(n)),
            ("unnormalized", random_unnormalized_limbs(n)),
        ] {
            let long = random_limbs(2 * n);
            group.bench_with_input(BenchmarkId::new(normalization, n), &n, |bench, &_| {
                bench.iter_batched_ref(
                    || vec![0; n + 1],
                    |out| {
                        nr_div_dyn(
                            black_box(long.as_slice()),
                            black_box(short.as_slice()),
                            black_box(out.as_mut_slice()),
                        )
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_nr_div_rem(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("nr_div_rem_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        for (normalization, short) in [
            ("normalized", random_normalized_limbs(n)),
            ("unnormalized", random_unnormalized_limbs(n)),
        ] {
            let long = random_limbs(2 * n);
            group.bench_with_input(BenchmarkId::new(normalization, n), &n, |bench, &_| {
                bench.iter_batched_ref(
                    || (long.clone(), vec![0; n + 1]),
                    |data| {
                        let (num, out) = data;
                        nr_div_rem_dyn(
                            black_box(num.as_mut_slice()),
                            black_box(short.as_slice()),
                            black_box(out.as_mut_slice()),
                        )
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_rcp_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("rcp_setup/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        for (algorithm, reciprocal) in [
            ("knuth", knuth_rcp_dyn as fn(&[u64], &mut [u64])),
            ("nr", nr_rcp_dyn as fn(&[u64], &mut [u64])),
        ] {
            for (normalization, input) in [
                ("normalized", random_normalized_limbs(n)),
                ("unnormalized", random_unnormalized_limbs(n)),
            ] {
                group.bench_with_input(
                    BenchmarkId::new(format!("{algorithm}_{normalization}"), n),
                    &n,
                    |bench, &_| {
                        bench.iter_batched_ref(
                            || vec![0; n + 1],
                            |out| {
                                reciprocal(
                                    black_box(input.as_slice()),
                                    black_box(out.as_mut_slice()),
                                )
                            },
                            BatchSize::LargeInput,
                        );
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bz_div,
    bench_nr_div,
    bench_nr_div_rem,
    bench_rcp_setup
);

criterion_main!(benches);
