use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;

use BigBits::utils::*;

fn random_limbs(n: usize) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen()).collect()
}

fn bench_add_buf(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_buf");

    let sizes: Vec<usize> = vec![4, 16, 64, 256, 1024, 4096];

    for &n in &sizes {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let short = random_limbs(n);
            let mut long = random_limbs(n + 1);

            b.iter(|| {
                for l in long.iter_mut() {
                    *l = rand::random();
                }
                add_buf(&mut long, &short)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_add_buf);
criterion_main!(benches);
