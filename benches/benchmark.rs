#![allow(dead_code)]
#![allow(unused_imports)]
use bit_ops_2::{
    bitfloat::*, bitfloat_static::*, bitfrac::*, bitint_static::BitIntStatic, ubitint::*,
    ubitint_static::*,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::OsRng, rngs::StdRng, Rng, RngCore, SeedableRng};
use std::cmp::Ordering;

fn ubi_eq_bench(c: &mut Criterion) {
    let seed = [
        0x5E, 0x47, 0xB8, 0x64, 0x4A, 0x0C, 0xAC, 0x39, 0xFE, 0x2B, 0x74, 0x5F, 0xCB, 0x15, 0xD9,
        0x3A, 0xB2, 0x18, 0xEE, 0x65, 0x28, 0xC4, 0x07, 0x3E, 0xA1, 0x99, 0x41, 0xDF, 0x6F, 0xBE,
        0x12, 0x76,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitvec_pairs: Vec<(UBitInt, UBitInt)> = sizes
        .iter()
        .map(|&size| {
            let ubi1 = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            let ubi2 = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            (ubi1, ubi2)
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubi_equal_{}", arch));
    for (size, (ubi1, ubi2)) in sizes.iter().zip(bitvec_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(ubi1) == black_box(ubi2));
            },
        );
    }

    group.finish();
}

fn ubi_cmp_bench(c: &mut Criterion) {
    let seed = [
        0xFA, 0x88, 0x8B, 0x31, 0xD4, 0xE1, 0x48, 0x2C, 0xC3, 0x90, 0x81, 0xD5, 0xBE, 0x93, 0x22,
        0xE2, 0xC3, 0xB8, 0x79, 0x4F, 0x80, 0xF0, 0xD2, 0xA4, 0x87, 0x6D, 0x50, 0xF3, 0x42, 0xA4,
        0xB7, 0x7B,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitvec_pairs: Vec<(UBitInt, UBitInt)> = sizes
        .iter()
        .map(|&size| {
            let ubi1 = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            let ubi2 = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            (ubi1, ubi2)
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubi_cmp_{}", arch));
    for (size, (ubi1, ubi2)) in sizes.iter().zip(bitvec_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(ubi1).cmp(black_box(ubi2)));
            },
        );
    }

    group.finish();
}

fn ubi_add_bench(c: &mut Criterion) {
    let seed = [
        0x4B, 0xCA, 0x50, 0xB6, 0x62, 0x60, 0x79, 0xC2, 0x55, 0xF3, 0x51, 0x63, 0x88, 0x67, 0x05,
        0x1F, 0x11, 0xCE, 0x8D, 0x47, 0xF5, 0x45, 0x67, 0x7B, 0xD7, 0xB1, 0xC3, 0x36, 0x72, 0x2A,
        0x10, 0x93,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitvec_pairs: Vec<(UBitInt, UBitInt)> = sizes
        .iter()
        .map(|&size| {
            let ubi1 = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            let ubi2 = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            (ubi1, ubi2)
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubi_add_{}", arch));
    for (size, (ubi1, ubi2)) in sizes.iter().zip(bitvec_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(ubi1) + black_box(ubi2));
            },
        );
    }

    group.finish();
}

fn ubi_sub_bench(c: &mut Criterion) {
    let seed = [
        0xE7, 0xB9, 0x1C, 0x1F, 0x00, 0x98, 0x55, 0x91, 0x07, 0x3A, 0xD1, 0x00, 0x19, 0x39, 0x57,
        0xA3, 0x90, 0xA4, 0x68, 0xA7, 0xEC, 0x0E, 0x7F, 0xB3, 0x8D, 0x2B, 0xFF, 0x37, 0x99, 0x29,
        0xE3, 0x82,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitvec_pairs: Vec<(UBitInt, UBitInt)> = sizes
        .iter()
        .map(|&size| {
            let ubi1 = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            let ubi2 = UBitInt {
                data: (0..size * 9 / 10)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            (ubi1, ubi2)
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubi_sub_{}", arch));
    for (size, (ubi1, ubi2)) in sizes.iter().zip(bitvec_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(ubi1) - black_box(ubi2));
            },
        );
    }

    group.finish();
}

fn ubi_mul_bench(c: &mut Criterion) {
    let seed = [
        0x68, 0x99, 0x4D, 0xBF, 0x75, 0x06, 0x69, 0xA7, 0x0C, 0xAD, 0x65, 0x5E, 0x3A, 0xDD, 0x31,
        0x21, 0x76, 0x30, 0x9F, 0x4D, 0xFB, 0x8D, 0xAE, 0xAC, 0x6B, 0x63, 0x85, 0x97, 0xB1, 0x34,
        0x4B, 0x9A,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitvec_pairs: Vec<(UBitInt, UBitInt)> = sizes
        .iter()
        .map(|&size| {
            let ubi1 = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            let ubi2 = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            (ubi1, ubi2)
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubi_mul_{}", arch));
    for (size, (ubi1, ubi2)) in sizes.iter().zip(bitvec_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(ubi1) * black_box(ubi2));
            },
        );
    }

    group.finish();
}

fn ubi_shl_bench(c: &mut Criterion) {
    let seed = [
        0xE2, 0x7E, 0xF5, 0x72, 0x1E, 0xDA, 0x50, 0x9D, 0x37, 0x09, 0x6D, 0x03, 0x73, 0x80, 0x23,
        0xCB, 0xFA, 0x62, 0x3E, 0xB9, 0x1D, 0x43, 0x85, 0x99, 0xD8, 0x26, 0xB6, 0x0C, 0x45, 0xEF,
        0x14, 0xE0,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitvec_pairs: Vec<(UBitInt, u128)> = sizes
        .iter()
        .map(|&size| {
            let ubi = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            let rusize = rng.gen_range(0..64) as u128;
            (ubi, rusize * size)
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubi_shl_{}", arch));
    for (size, (ubi1, shl)) in sizes.iter().zip(bitvec_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(ubi1) << *shl);
            },
        );
    }

    group.finish();
}

fn ubi_shr_bench(c: &mut Criterion) {
    let seed = [
        0x99, 0x00, 0x53, 0x1C, 0x0E, 0x31, 0xE5, 0xC0, 0x3A, 0xAC, 0x29, 0x9C, 0xC7, 0x19, 0xD8,
        0x5A, 0xCD, 0x9D, 0xBF, 0x43, 0xFD, 0xDB, 0x3D, 0xC5, 0xCF, 0xB1, 0x92, 0xF9, 0x80, 0x7F,
        0x87, 0xC0,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitvec_pairs: Vec<(UBitInt, u128)> = sizes
        .iter()
        .map(|&size| {
            let ubi = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            let rusize = rng.gen_range(0..64) as u128;
            (ubi, rusize * size)
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubi_shr_{}", arch));
    for (size, (ubi1, shl)) in sizes.iter().zip(bitvec_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(ubi1) >> *shl);
            },
        );
    }

    group.finish();
}

fn ubi_div_bench(c: &mut Criterion) {
    let seed = [
        0x5E, 0xD4, 0x85, 0x8F, 0xDD, 0xCA, 0x58, 0x57, 0x10, 0x15, 0x83, 0xE9, 0x31, 0x3D, 0x75,
        0xD1, 0xA9, 0xD2, 0xCB, 0x38, 0xCA, 0x5D, 0x8F, 0x8E, 0xD7, 0x1F, 0x87, 0x5B, 0xD9, 0x0E,
        0x3B, 0x48,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitvec_pairs: Vec<(UBitInt, UBitInt)> = sizes
        .iter()
        .map(|&size| {
            let ubi1 = UBitInt {
                data: (0..size)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            let ubi2 = UBitInt {
                data: (0..size * 9 / 10)
                    .map(|_| rng.gen::<usize>())
                    .collect::<Vec<usize>>(),
            };
            (ubi1, ubi2)
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubi_div_{}", arch));
    for (size, (ubi1, ubi2)) in sizes.iter().zip(bitvec_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(ubi1) / black_box(ubi2));
            },
        );
    }

    group.finish();
}

fn bf_eq_bench(c: &mut Criterion) {
    let seed = [
        0x2A, 0x1B, 0x80, 0x2D, 0x2F, 0x29, 0x02, 0x44, 0x28, 0xFF, 0xBF, 0x82, 0x7C, 0x01, 0xB7,
        0x23, 0x36, 0xF4, 0xE0, 0x83, 0xA1, 0x12, 0xF4, 0x33, 0x35, 0xF2, 0x06, 0x3E, 0x03, 0x4C,
        0x73, 0x0E,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitfloat_pairs: Vec<(BitFloat, BitFloat)> = sizes
        .iter()
        .map(|&size| {
            let m1 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let m2 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let exp1 = rng.gen::<isize>();
            let exp2 = rng.gen::<isize>();
            let sign1 = rng.gen::<bool>();
            let sign2 = rng.gen::<bool>();
            (
                BitFloat {
                    sign: sign1,
                    exp: exp1,
                    m: m1,
                },
                BitFloat {
                    sign: sign2,
                    exp: exp2,
                    m: m2,
                },
            )
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("bf_eq_{}", arch));
    for (size, (bf1, bf2)) in sizes.iter().zip(bitfloat_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(bf1) == black_box(bf2));
            },
        );
    }

    group.finish();
}

fn bf_cmp_bench(c: &mut Criterion) {
    let seed = [
        0xBF, 0xF7, 0x00, 0xD6, 0xA3, 0x07, 0xF3, 0x92, 0xC1, 0xB2, 0xE4, 0xF0, 0x73, 0x2E, 0x33,
        0x13, 0x25, 0x38, 0x93, 0x7B, 0xDF, 0x3D, 0xC5, 0xCB, 0xE2, 0x70, 0x7D, 0x71, 0x4E, 0xC3,
        0xD5, 0xCD,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitfloat_pairs: Vec<(BitFloat, BitFloat)> = sizes
        .iter()
        .map(|&size| {
            let m1 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let m2 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let exp1 = rng.gen::<isize>();
            let exp2 = rng.gen::<isize>();
            let sign1 = rng.gen::<bool>();
            let sign2 = rng.gen::<bool>();
            (
                BitFloat {
                    sign: sign1,
                    exp: exp1,
                    m: m1,
                },
                BitFloat {
                    sign: sign2,
                    exp: exp2,
                    m: m2,
                },
            )
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("bf_cmp_{}", arch));
    for (size, (bf1, bf2)) in sizes.iter().zip(bitfloat_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| bf1.cmp(bf2));
            },
        );
    }

    group.finish();
}

fn bf_add_bench(c: &mut Criterion) {
    let seed = [
        0x4C, 0x49, 0xAA, 0xF2, 0x8A, 0x11, 0xE4, 0x6F, 0x32, 0xAE, 0xF9, 0xE4, 0x86, 0xDA, 0x85,
        0xDB, 0xA0, 0xB6, 0x84, 0x6F, 0x4D, 0x26, 0x68, 0x2D, 0x81, 0x79, 0x27, 0x32, 0x43, 0x8F,
        0x14, 0xD3,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64, 256];
    let bitfloat_pairs: Vec<(BitFloat, BitFloat)> = sizes
        .iter()
        .map(|&size| {
            let m1 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let m2 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let exp1 = rng.gen_range(-size..=size);
            let exp2 = rng.gen_range(-size..=size);
            let sign1 = rng.gen::<bool>();
            let sign2 = rng.gen::<bool>();
            (
                BitFloat {
                    sign: sign1,
                    exp: exp1,
                    m: m1,
                },
                BitFloat {
                    sign: sign2,
                    exp: exp2,
                    m: m2,
                },
            )
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("bf_add_{}", arch));
    for (size, (bf1, bf2)) in sizes.iter().zip(bitfloat_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(bf1) + black_box(bf2));
            },
        );
    }

    group.finish();
}

fn bf_mul_bench(c: &mut Criterion) {
    let seed = [
        0xBB, 0xBB, 0x95, 0x94, 0xB3, 0xC1, 0xC2, 0x34, 0xFA, 0x80, 0xCF, 0xF5, 0x6C, 0x99, 0x00,
        0x72, 0xE7, 0x1D, 0xFC, 0x33, 0x17, 0x5A, 0x4E, 0xF9, 0x22, 0xE5, 0x89, 0x30, 0x5C, 0x09,
        0x3E, 0x50,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitfloat_pairs: Vec<(BitFloat, BitFloat)> = sizes
        .iter()
        .map(|&size| {
            let m1 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let m2 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let exp1 = rng.gen_range(-size..=size);
            let exp2 = rng.gen_range(-size..=size);
            let sign1 = rng.gen::<bool>();
            let sign2 = rng.gen::<bool>();
            (
                BitFloat {
                    sign: sign1,
                    exp: exp1,
                    m: m1,
                },
                BitFloat {
                    sign: sign2,
                    exp: exp2,
                    m: m2,
                },
            )
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("bf_mul_{}", arch));
    for (size, (bf1, bf2)) in sizes.iter().zip(bitfloat_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(bf1) * black_box(bf2));
            },
        );
    }

    group.finish();
}

fn bf_stbl_mul_bench(c: &mut Criterion) {
    let seed = [
        0x90, 0x61, 0xFD, 0x96, 0x84, 0xD3, 0xCB, 0xCA, 0x94, 0x97, 0xA4, 0x80, 0xD0, 0x13, 0xCD,
        0x22, 0x54, 0xA0, 0xC9, 0x16, 0x43, 0xF7, 0x23, 0x81, 0xFF, 0xB2, 0x4F, 0xD4, 0xA4, 0x9C,
        0xDF, 0x66,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitfloat_pairs: Vec<(BitFloat, BitFloat)> = sizes
        .iter()
        .map(|&size| {
            let m1 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let m2 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let exp1 = rng.gen_range(-size..=size);
            let exp2 = rng.gen_range(-size..=size);
            let sign1 = rng.gen::<bool>();
            let sign2 = rng.gen::<bool>();
            (
                BitFloat {
                    sign: sign1,
                    exp: exp1,
                    m: m1,
                },
                BitFloat {
                    sign: sign2,
                    exp: exp2,
                    m: m2,
                },
            )
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("bf_stbl_mul_{}", arch));
    for (size, (bf1, bf2)) in sizes.iter().zip(bitfloat_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(bf1).stbl_mul(black_box(bf2), *size as usize));
            },
        );
    }

    group.finish();
}

fn bf_div_bench(c: &mut Criterion) {
    let seed = [
        0x59, 0x92, 0x43, 0x8E, 0x5D, 0x9D, 0x9E, 0xB7, 0x00, 0x5F, 0x70, 0x1E, 0xE7, 0xD4, 0xA6,
        0xB1, 0x19, 0xD6, 0xBD, 0xBF, 0x02, 0xA1, 0x8C, 0x46, 0x4C, 0x6F, 0xFB, 0x02, 0x1B, 0x32,
        0x41, 0x45,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitfloat_pairs: Vec<(BitFloat, BitFloat)> = sizes
        .iter()
        .map(|&size| {
            let m1 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let m2 = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let exp1 = rng.gen_range(-size..=size);
            let exp2 = rng.gen_range(-size..=size);
            let sign1 = rng.gen::<bool>();
            let sign2 = rng.gen::<bool>();
            (
                BitFloat {
                    sign: sign1,
                    exp: exp1,
                    m: m1,
                },
                BitFloat {
                    sign: sign2,
                    exp: exp2,
                    m: m2,
                },
            )
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("bf_div_{}", arch));
    for (size, (bf1, bf2)) in sizes.iter().zip(bitfloat_pairs.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(bf1) / black_box(bf2));
            },
        );
    }

    group.finish();
}

fn bf_exp_bench(c: &mut Criterion) {
    let seed = [
        0xAA, 0x95, 0xA7, 0x14, 0x32, 0xAD, 0x7F, 0x30, 0x8C, 0x54, 0x45, 0xD3, 0xC1, 0x1A, 0x67,
        0xBD, 0x8E, 0xF2, 0x2C, 0xD5, 0xB9, 0x16, 0x7B, 0x9D, 0x0E, 0x39, 0xAC, 0x53, 0x53, 0x5E,
        0x45, 0x2D,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitfloats: Vec<BitFloat> = sizes
        .iter()
        .map(|&size| {
            let m = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let exp = 0;
            let sign = rng.gen::<bool>();
            BitFloat { sign, exp, m }
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("bf_exp_{}", arch));
    for (size, bf) in sizes.iter().zip(bitfloats.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(bf).exp());
            },
        );
    }

    group.finish();
}

fn bf_ln_bench(c: &mut Criterion) {
    let seed = [
        0xAA, 0x2E, 0xE8, 0xA4, 0xB9, 0x03, 0xF1, 0x7E, 0x51, 0x28, 0xFD, 0x78, 0xDA, 0xD9, 0x43,
        0xA9, 0x2A, 0x16, 0xF6, 0xC4, 0x56, 0x48, 0xDB, 0xCB, 0x94, 0x03, 0x98, 0xB7, 0xCD, 0xA5,
        0x98, 0xD7,
    ];
    let mut rng = StdRng::from_seed(seed);

    let sizes = vec![4, 16, 64];
    let bitfloats: Vec<BitFloat> = sizes
        .iter()
        .map(|&size| {
            let m = (0..size)
                .map(|_| rng.gen::<usize>())
                .collect::<Vec<usize>>();
            let exp = 0;
            let sign = rng.gen::<bool>();
            BitFloat { sign, exp, m }
        })
        .collect();

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("bf_ln_{}", arch));
    for (size, bf) in sizes.iter().zip(bitfloats.iter()) {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bencher, &_size| {
                bencher.iter(|| black_box(bf).ln());
            },
        );
    }

    group.finish();
}

macro_rules! benchmark_ubis_op {
    ($group:expr, $rng:expr, $op:tt, $($size:expr),+) => {
        $(
            let mut data1 = [0; $size];
            let mut data2 = [0; $size];
            for i in 0..$size {
                data1[i] = $rng.gen::<usize>();
                data2[i] = $rng.gen::<usize>();
            }
            let (ubis_a, ubis_b) = (UBitIntStatic::make(data1), UBitIntStatic::make(data2));
            $group.bench_with_input(BenchmarkId::from_parameter($size), &$size, |bencher, _| {
                bencher.iter(|| black_box(ubis_a) $op black_box(ubis_b));
            });
        )+
    }
}

macro_rules! benchmark_ubis_sh {
    ($group:expr, $rng:expr, $op:tt, $($size:expr),+) => {
        $(
            let mut data1 = [0; $size];
            for i in 0..$size {
                data1[i] = $rng.gen::<usize>();
            }
            let (ubis_a, sh) = (UBitIntStatic::make(data1), ($size as u128)*($rng.gen_range::<u128, _>(0..=64)));
            $group.bench_with_input(BenchmarkId::from_parameter($size), &$size, |bencher, _| {
                bencher.iter(|| black_box(ubis_a) $op sh);
            });
        )+
    }
}

fn ubis_add_bench(c: &mut Criterion) {
    let seed = [
        0x4B, 0xCA, 0x50, 0xB6, 0x62, 0x60, 0x79, 0xC2, 0x55, 0xF3, 0x51, 0x63, 0x88, 0x67, 0x05,
        0x1F, 0x11, 0xCE, 0x8D, 0x47, 0xF5, 0x45, 0x67, 0x7B, 0xD7, 0xB1, 0xC3, 0x36, 0x72, 0x2A,
        0x10, 0x93,
    ];
    let mut rng = StdRng::from_seed(seed);

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubis_add_{}", arch));

    benchmark_ubis_op!(group, rng, +, 4, 16, 64);

    group.finish();
}

fn ubis_sub_bench(c: &mut Criterion) {
    let seed = [
        0x39, 0xB3, 0xF5, 0x19, 0x72, 0xCE, 0x16, 0xBD, 0xC7, 0x2E, 0x11, 0xFA, 0x00, 0xF3, 0xEB,
        0xD5, 0xA2, 0xB5, 0x4C, 0x10, 0xEC, 0xFE, 0x73, 0xE2, 0x01, 0x3E, 0xE8, 0xD5, 0x3F, 0x9A,
        0x34, 0x07,
    ];
    let mut rng = StdRng::from_seed(seed);

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubis_sub_{}", arch));

    benchmark_ubis_op!(group, rng, -, 4, 16, 64);

    group.finish();
}

fn ubis_mul_bench(c: &mut Criterion) {
    let seed = [
        0x28, 0x21, 0xDE, 0xBE, 0x95, 0xDD, 0x26, 0x8C, 0x86, 0x54, 0xE7, 0x78, 0xB4, 0xCC, 0x03,
        0x00, 0x94, 0x2D, 0x41, 0x4A, 0xAA, 0x79, 0x5B, 0x6E, 0xF7, 0xE6, 0x12, 0xE6, 0x5C, 0xB5,
        0x01, 0x7B,
    ];
    let mut rng = StdRng::from_seed(seed);

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubis_mul_{}", arch));

    benchmark_ubis_op!(group, rng, *, 4, 16, 64);

    group.finish();
}

fn ubis_shl_bench(c: &mut Criterion) {
    let seed = [
        0xAE, 0xF9, 0xBC, 0xAA, 0xED, 0x8B, 0xA4, 0x58, 0xF2, 0x91, 0xAC, 0x92, 0xC0, 0xD4, 0x71,
        0x4D, 0xE1, 0x8C, 0x00, 0xC9, 0x26, 0x01, 0xCB, 0xBD, 0x44, 0xE0, 0x07, 0x6B, 0x3B, 0xBC,
        0xDD, 0x1E,
    ];
    let mut rng = StdRng::from_seed(seed);

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubis_shl_{}", arch));

    benchmark_ubis_sh!(group, rng, <<, 4, 16, 64);

    group.finish();
}

fn ubis_shr_bench(c: &mut Criterion) {
    let seed = [
        0x95, 0x0F, 0x51, 0x12, 0x3F, 0x34, 0x24, 0x23, 0x57, 0x15, 0xF1, 0x8E, 0xD7, 0x0D, 0x31,
        0xE2, 0x27, 0xEE, 0xEE, 0x9F, 0xB9, 0x5B, 0xF7, 0x45, 0xDE, 0x15, 0x3A, 0x09, 0xB1, 0xDB,
        0x8B, 0x04,
    ];
    let mut rng = StdRng::from_seed(seed);

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubis_shr_{}", arch));

    benchmark_ubis_sh!(group, rng, >>, 4, 16, 64);

    group.finish();
}

fn ubis_div_bench(c: &mut Criterion) {
    let seed = [
        0x6A, 0xF2, 0xB9, 0x7D, 0x1E, 0x36, 0xB4, 0xC2, 0xCA, 0xA7, 0x09, 0x8D, 0x7F, 0xF9, 0xD6,
        0x61, 0x12, 0xE3, 0xE3, 0x72, 0x80, 0x69, 0x5A, 0x0B, 0x59, 0x21, 0xC2, 0xEE, 0x55, 0x3C,
        0x9D, 0xBB,
    ];
    let mut rng = StdRng::from_seed(seed);

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("ubis_div_{}", arch));

    benchmark_ubis_op!(group, rng, /, 4, 16, 64);

    group.finish();
}

macro_rules! benchmark_bfs_op {
    ($group:expr, $rng:expr, $op:tt, $($size:expr),+) => {
        $(
            let mut data1 = [0; $size];
            let mut data2 = [0; $size];
            for i in 0..$size {
                data1[i] = $rng.gen::<usize>();
                data2[i] = $rng.gen::<usize>();
            }

            let exp1 = $rng.gen_range::<i128, _>(-$size..=$size);
            let exp2 = $rng.gen_range::<i128, _>(-$size..=$size);
            let sign1 = $rng.gen::<bool>();
            let sign2 = $rng.gen::<bool>();

            let (bfs1, bfs2) = (BitFloatStatic::make(data1, exp1, sign1), BitFloatStatic::make(data2, exp2, sign2));
            $group.bench_with_input(BenchmarkId::from_parameter($size), &$size, |bencher, _| {
                bencher.iter(|| black_box(bfs1) $op black_box(bfs2));
            });
        )+
    }
}

fn bfs_add_bench(c: &mut Criterion) {
    let seed = [
        0xF4, 0x0B, 0xFB, 0x49, 0x9D, 0x85, 0xC8, 0x74, 0xBD, 0x6B, 0xF5, 0xD7, 0x9C, 0x09, 0xC3,
        0x45, 0x8E, 0xE9, 0xB9, 0x3C, 0xD2, 0x85, 0x2B, 0xEE, 0x2B, 0xD0, 0x09, 0x32, 0x7D, 0x91,
        0x8C, 0x8F,
    ];
    let mut rng = StdRng::from_seed(seed);

    #[cfg(target_arch = "x86_64")]
    let arch = "x86_64";
    #[cfg(target_arch = "aarch64")]
    let arch = "aarch64";

    let mut group = c.benchmark_group(format!("bfs_add_{}", arch));

    benchmark_bfs_op!(group, rng, +, 4, 16, 64, 256);

    group.finish();
}

// criterion_group!(
//     benches,
//     ubi_eq_bench,
//     ubi_cmp_bench,
//     ubi_add_bench,
//     ubi_sub_bench,
//     ubi_mul_bench,
//     ubi_shl_bench,
//     ubi_shr_bench,
//     ubi_div_bench,
// );

// criterion_group!(
//     benches,
//     bf_eq_bench,
//     bf_cmp_bench,
//     bf_add_bench,
//     bf_mul_bench,
//     bf_stbl_mul_bench,
//     bf_div_bench,
//     bf_exp_bench,
//     bf_ln_bench,
// );

// criterion_group!(
//     benches,
//     ubis_add_bench,
//     ubis_sub_bench,
//     ubis_mul_bench,
//     ubis_shl_bench,
//     ubis_shr_bench,
//     ubis_div_bench,
// );

// criterion_group!(
//     benches,
//     ubi_add_bench,
//     ubis_add_bench,
//     ubi_sub_bench,
//     ubis_sub_bench,
//     ubi_mul_bench,
//     ubis_mul_bench,
//     ubi_shl_bench,
//     ubis_shl_bench,
//     ubi_shr_bench,
//     ubis_shr_bench,
//     ubi_div_bench,
//     ubis_div_bench,
// );

criterion_group!(benches, bfs_add_bench, bf_add_bench);

criterion_main!(benches);
