#![allow(dead_code)]
use big_bits::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
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

fn bench_acc_buf(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("add_buf/{ARCH}"));
    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];
    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, &n| {
            let short = random_limbs(n);
            let mut long = random_limbs(n + 1);
            bench.iter(|| acc(black_box(&mut long), black_box(&short), 0));
        });
    }
    group.finish();
}

fn bench_shl_buf(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("shl_buf/{ARCH}"));
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

fn bench_shr_buf(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("shr_buf/{ARCH}"));
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

fn bench_school_mul_buf(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("school_mul_buf/{ARCH}"));
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

criterion_group!(benches, bench_acc_buf);
criterion_main!(benches);
