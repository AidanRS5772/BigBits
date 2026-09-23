//! Stable inputs for before/after comparisons of division control flow.
use big_bits::utils::div::*;
use criterion::{black_box, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Duration;

fn register<const N: usize>(c: &mut Criterion, d_len: usize, body_len: usize) {
    let mut rng =
        StdRng::seed_from_u64(0x4449_5646 ^ d_len as u64 ^ (body_len as u64).rotate_left(32));
    // Exercise normalized and shifted inputs with exactly the same data in every build.
    let inputs: Vec<_> = (0..4)
        .map(|i| {
            let mut d: Vec<u64> = (0..d_len).map(|_| rng.gen()).collect();
            d[d_len - 1] = if i % 2 == 0 {
                d[d_len - 1] | (1 << 63)
            } else {
                (d[d_len - 1] >> 17) | 1
            };
            let mut n: Vec<u64> = (0..d_len + body_len).map(|_| rng.gen()).collect();
            *n.last_mut().unwrap() |= 1 << 63;
            (n, d)
        })
        .collect();
    let mut group = c.benchmark_group("division_flow");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(150));
    group.measurement_time(Duration::from_millis(500));
    for static_model in [false, true] {
        let model = if static_model { "static" } else { "dyn" };
        for forced in [false, true] {
            let path = if forced { "nr" } else { "public" };
            for remainder in [false, true] {
                let op = if remainder { "rem" } else { "div" };
                for spare in [false, true] {
                    let layout = if spare { "spare" } else { "exact" };
                    let name = format!("{model}/{op}/{path}/{layout}");
                    group.bench_function(
                        BenchmarkId::new(name, format!("{d_len}x{body_len}")),
                        |b| {
                            let mut n = inputs[0].0.clone();
                            let mut q = vec![0; body_len + usize::from(spare)];
                            let mut i = 0;
                            b.iter(|| {
                                let (input, d) = &inputs[i];
                                i = (i + 1) % inputs.len();
                                n.copy_from_slice(input);
                                let n = black_box(n.as_mut_slice());
                                let d = black_box(d.as_slice());
                                let q = black_box(q.as_mut_slice());
                                let overflow = match (static_model, remainder, forced) {
                                    (false, false, false) => div_dyn(n, d, q),
                                    (false, true, false) => div_rem_dyn(n, d, q),
                                    (true, false, false) => div_static::<N>(n, d, q),
                                    (true, true, false) => div_rem_static::<N>(n, d, q),
                                    (false, false, true) => {
                                        let r = division_preflight(n, d, q).unwrap();
                                        div_prepared_dyn(n, d, q, r, DivAlg::NR)
                                    }
                                    (false, true, true) => {
                                        let r = division_preflight(n, d, q).unwrap();
                                        div_rem_prepared_dyn(n, d, q, r, DivAlg::NR)
                                    }
                                    (true, false, true) => {
                                        let r = division_preflight(n, d, q).unwrap();
                                        div_prepared_static::<N>(n, d, q, r, DivAlg::NR)
                                    }
                                    (true, true, true) => {
                                        let r = division_preflight(n, d, q).unwrap();
                                        div_rem_prepared_static::<N>(n, d, q, r, DivAlg::NR)
                                    }
                                };
                                black_box(overflow);
                            });
                        },
                    );
                }
            }
            if d_len == body_len {
                group.bench_function(
                    BenchmarkId::new(format!("{model}/rcp/{path}"), d_len),
                    |b| {
                        let mut rcp = vec![0; body_len];
                        let mut i = 0;
                        b.iter(|| {
                            let d = black_box(inputs[i].1.as_slice());
                            i = (i + 1) % inputs.len();
                            let rcp = black_box(rcp.as_mut_slice());
                            match (static_model, forced) {
                                (false, false) => rcp_dyn(d, rcp),
                                (true, false) => rcp_static::<N>(d, rcp),
                                (false, true) => rcp_prepared_dyn(d, rcp, RcpAlg::NR),
                                (true, true) => rcp_prepared_static::<N>(d, rcp, RcpAlg::NR),
                            }
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

pub fn bench_division_flow(c: &mut Criterion) {
    register::<32>(c, 16, 16);
    register::<256>(c, 128, 128);
    register::<640>(c, 128, 512);
    register::<4096>(c, 2048, 2048);
    // With the universal cutoffs, this skew selects BZ in all public drivers.
    // The 173-limb partial top block also exercises padded-top recursion.
    assert!(use_bz_for_top_block(217, 174));
    register::<4608>(c, 217, 4296);
}
