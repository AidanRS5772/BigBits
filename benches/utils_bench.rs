#![allow(dead_code)]

use big_bits::{
    utils::{div::*, BZ_CUTOFF},
    *,
};
use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use rand::Rng;

fn random_limbs(n: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}

fn random_sh() -> u8 {
    let mut rng = rand::thread_rng();
    rng.gen_range(0..64)
}

const ARCH: &'static str = std::env::consts::ARCH;

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
                let mut out = vec![0u64; 2 * $n];
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
                let mut out = vec![0u64; 2 * $n];
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
            let mut long = random_limbs(n + 1);
            bench.iter(|| add_buf(black_box(&mut long), black_box(&short)));
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
            let mut long = random_limbs(n + 1);
            bench.iter(|| sub_buf(black_box(&mut long), black_box(&short)));
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
            let mut buf = random_limbs(n);
            let sh = random_sh();
            bench.iter(|| shl_buf(black_box(&mut buf), sh));
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
            let mut buf = random_limbs(n);
            let sh = random_sh();
            bench.iter(|| shr_buf(black_box(&mut buf), sh));
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
            let mut out = vec![0; 2 * n];
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
            let mut out = vec![0; 2 * n];
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
            let mut out = vec![0; 2 * n];
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
            let mut out = vec![0; 2 * n];
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
            let mut out = vec![0; 2 * n];
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
            let mut out = vec![0; 2 * n];
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
            let mut out = vec![0; 2 * n];
            bench.iter(|| fft_sqr_entry(black_box(&a), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_gen_sqr(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("fft_sqr_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let mut out = vec![0; 2 * n];
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

fn bench_knuth_div(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("knuth_div_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let mut short = random_limbs(n - 1);
            short.push(u64::MAX);
            let mut long = random_limbs(2 * n);
            let mut out = vec![0; n + 1];
            bench.iter(|| {
                div_buf_of(
                    black_box(&mut long),
                    &mut 0,
                    black_box(&short),
                    black_box(&mut out),
                )
            });
        });
    }
    group.finish();
}

fn bench_bz_div(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("bz_div_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![BZ_CUTOFF - 1, BZ_CUTOFF + 1];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let mut short = random_limbs(n);
            let mut long = random_limbs(2 * n);
            let mut out = vec![0; n + 1];
            bench.iter(|| {
                bz_div_dyn(
                    black_box(&mut long),
                    black_box(&mut short),
                    black_box(&mut out),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bz_div);

criterion_main!(benches);
