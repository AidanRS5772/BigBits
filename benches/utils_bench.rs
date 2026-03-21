#![allow(dead_code)]

use big_bits::*;
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
    group.measurement_time(std::time::Duration::from_secs(20)); // default is 5s
    group.warm_up_time(std::time::Duration::from_secs(5)); // default is 3s
}

fn bench_add_buf(c: &mut Criterion) {
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

fn bench_sub_buf(c: &mut Criterion) {
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
            let mut out = vec![0; 2 * n - 1];
            bench.iter(|| mul_buf(black_box(&a), black_box(&b), black_box(&mut out)));
        });
    }
    group.finish();
}

fn bench_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("mul_buf/{ARCH}"));
    set_up_group(&mut group);
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let a = random_limbs(n);
            let b = random_limbs(n);
            bench.iter(|| mul_vec(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_shr, bench_school_mul, bench_mul);
criterion_main!(benches);
