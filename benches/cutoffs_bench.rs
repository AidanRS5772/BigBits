#![allow(dead_code)]

use big_bits::{utils::div::*, utils::*, *};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::{Duration, Instant},
};

const ARCH: &'static str = std::env::consts::ARCH;

pub type Point = (usize, usize);

// Shared benchmark helpers.

fn random_limbs(n: usize, rng: &mut impl Rng) -> Vec<u64> {
    let mut limbs: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
    if let Some(last) = limbs.last_mut() {
        *last |= 1;
    }
    limbs
}

fn random_normalized_limbs(n: usize, rng: &mut impl Rng) -> Vec<u64> {
    let mut limbs = random_limbs(n, rng);
    if let Some(last) = limbs.last_mut() {
        *last |= 1 << 63;
    }
    limbs
}

fn duration_ratio(num: Duration, den: Duration) -> f64 {
    num.as_nanos().max(1) as f64 / den.as_nanos().max(1) as f64
}

fn timed(run: impl FnOnce()) -> Duration {
    let start = Instant::now();
    run();
    start.elapsed()
}

fn time_pair_alternating(
    numerator_first: bool,
    numerator: impl FnOnce(),
    denominator: impl FnOnce(),
) -> (Duration, Duration) {
    if numerator_first {
        let numerator = timed(numerator);
        let denominator = timed(denominator);
        (numerator, denominator)
    } else {
        let denominator = timed(denominator);
        let numerator = timed(numerator);
        (numerator, denominator)
    }
}

fn alternating_ratio_bench<T>(
    iters: u64,
    inputs: &[T],
    mut measure: impl FnMut(&T, bool) -> (Duration, Duration),
) -> Duration {
    if inputs.is_empty() || iters == 0 {
        return Duration::ZERO;
    }

    let mut rng = rand::thread_rng();
    let mut ratio_sum = 0.0f64;
    for _ in 0..iters {
        for input in inputs {
            let (numerator, denominator) = measure(input, rng.gen());
            ratio_sum += duration_ratio(numerator, denominator);
        }
    }

    let avg_ratio_per_iter = ratio_sum / inputs.len() as f64;
    Duration::from_nanos((avg_ratio_per_iter * 1_000_000.0) as u64)
}

// Boundary search infrastructure.

#[inline]
fn neighbors(p: Point) -> impl Iterator<Item = Point> {
    [
        p.0.checked_add(1).map(|x| (x, p.1)),
        p.0.checked_sub(1).map(|x| (x, p.1)),
        p.1.checked_add(1).map(|y| (p.0, y)),
        p.1.checked_sub(1).map(|y| (p.0, y)),
    ]
    .into_iter()
    .flatten()
}

#[derive(Debug, Clone)]
struct BracketPair {
    soft_true_side: Point,
    soft_false_side: Point,
    hard_dist: u32,
}

pub struct BoundarySearch<H, S> {
    hard: H,
    soft: S,
    exploration_budget: usize,
}

impl<H, S> BoundarySearch<H, S>
where
    H: Fn(usize, usize) -> bool,
    S: Fn(usize, usize) -> bool,
{
    pub fn new(hard: H, soft: S) -> Self {
        Self {
            hard,
            soft,
            exploration_budget: 2_000_000,
        }
    }

    pub fn with_budget(mut self, budget: usize) -> Self {
        self.exploration_budget = budget;
        self
    }

    /// Find `count` bracket pairs straddling the soft boundary within the hard-true region.
    ///
    /// - `seed`: a point known to satisfy `hard(x, y) == true`.
    /// - `count`: how many `(usize, usize)` points to return.
    /// - `target_gap`: desired Manhattan distance between paired points on opposite
    ///   sides of the soft boundary. A gap of 1 gives the tightest possible bracket.
    ///
    /// Returns points near the soft boundary, all within the hard-true region,
    /// sorted so that points nearest the hard boundary come first.
    /// Points alternate: soft-true side, soft-false side, soft-true, soft-false, …
    pub fn find(&self, seed: Point, count: usize, target_gap: u32) -> Vec<(usize, usize)> {
        if !(self.hard)(seed.0, seed.1) {
            return vec![];
        }

        let target_gap = target_gap.max(1);

        // Phase 1: BFS to explore the hard-true region, recording soft values.
        let (hard_true_cells, soft_vals, hard_boundary_seeds) = self.explore(seed);

        // Phase 2: Multi-source BFS from hard boundary inward to compute hard_dist.
        let hard_dist = Self::compute_hard_dist(&hard_true_cells, &hard_boundary_seeds);

        // Phase 3: Find soft boundary crossings.
        let mut crossings = self.find_soft_crossings(&hard_true_cells, &soft_vals, &hard_dist);

        // Sort: nearest to hard boundary first.
        crossings.sort_unstable_by_key(|c| c.hard_dist);

        // Phase 4: Optionally expand crossings, then flatten into (usize, usize) output.
        let bracket_count = (count + 1) / 2; // each bracket yields 2 points
        let mut out = Vec::with_capacity(count);

        for crossing in crossings.iter().take(bracket_count) {
            let bp = if target_gap <= 1 {
                crossing.clone()
            } else {
                self.expand_crossing(crossing, target_gap, &hard_true_cells)
            };
            out.push(bp.soft_true_side);
            out.push(bp.soft_false_side);
        }

        out.truncate(count);
        out
    }

    /// BFS from seed within the hard-true region.
    fn explore(&self, seed: Point) -> (HashSet<Point>, HashMap<Point, bool>, Vec<Point>) {
        let mut hard_true: HashSet<Point> = HashSet::new();
        let mut soft_val: HashMap<Point, bool> = HashMap::new();
        let mut hard_false: HashSet<Point> = HashSet::new();
        let mut hard_boundary: Vec<Point> = Vec::new();
        let mut queue: VecDeque<Point> = VecDeque::new();

        hard_true.insert(seed);
        soft_val.insert(seed, (self.soft)(seed.0, seed.1));
        queue.push_back(seed);

        while let Some(p) = queue.pop_front() {
            let mut on_edge = false;

            for nb in neighbors(p) {
                if hard_true.contains(&nb) || hard_false.contains(&nb) {
                    if hard_false.contains(&nb) {
                        on_edge = true;
                    }
                    continue;
                }

                if (self.hard)(nb.0, nb.1) {
                    hard_true.insert(nb);
                    soft_val.insert(nb, (self.soft)(nb.0, nb.1));
                    if hard_true.len() < self.exploration_budget {
                        queue.push_back(nb);
                    }
                } else {
                    hard_false.insert(nb);
                    on_edge = true;
                }
            }

            // Points at usize 0 on either axis are implicitly on a boundary
            // (can't explore below 0), so treat them as on_edge too.
            if p.0 == 0 || p.1 == 0 {
                on_edge = true;
            }

            if on_edge {
                hard_boundary.push(p);
            }
        }

        (hard_true, soft_val, hard_boundary)
    }

    /// Multi-source BFS from hard boundary points inward.
    fn compute_hard_dist(
        hard_true: &HashSet<Point>,
        boundary_seeds: &[Point],
    ) -> HashMap<Point, u32> {
        let mut dist: HashMap<Point, u32> = HashMap::new();
        let mut queue: VecDeque<Point> = VecDeque::new();

        for &p in boundary_seeds {
            dist.insert(p, 0);
            queue.push_back(p);
        }

        while let Some(p) = queue.pop_front() {
            let d = dist[&p];
            for nb in neighbors(p) {
                if !dist.contains_key(&nb) && hard_true.contains(&nb) {
                    dist.insert(nb, d + 1);
                    queue.push_back(nb);
                }
            }
        }

        dist
    }

    /// Find edges between hard-true cells that disagree on soft.
    fn find_soft_crossings(
        &self,
        hard_true: &HashSet<Point>,
        soft_val: &HashMap<Point, bool>,
        hard_dist: &HashMap<Point, u32>,
    ) -> Vec<BracketPair> {
        let mut seen_edges: HashSet<(Point, Point)> = HashSet::new();
        let mut crossings = Vec::new();

        for &p in hard_true {
            let Some(&sp) = soft_val.get(&p) else {
                continue;
            };

            for nb in neighbors(p) {
                let edge = if p < nb { (p, nb) } else { (nb, p) };
                if !seen_edges.insert(edge) {
                    continue;
                }

                let Some(&sn) = soft_val.get(&nb) else {
                    continue;
                };
                if !hard_true.contains(&nb) || sp == sn {
                    continue;
                }

                let (st, sf) = if sp { (p, nb) } else { (nb, p) };
                let hd = std::cmp::min(
                    hard_dist.get(&p).copied().unwrap_or(u32::MAX),
                    hard_dist.get(&nb).copied().unwrap_or(u32::MAX),
                );

                crossings.push(BracketPair {
                    soft_true_side: st,
                    soft_false_side: sf,
                    hard_dist: hd,
                });
            }
        }

        crossings
    }

    /// Expand a tight (gap=1) crossing to the target gap.
    fn expand_crossing(
        &self,
        crossing: &BracketPair,
        target_gap: u32,
        hard_true: &HashSet<Point>,
    ) -> BracketPair {
        let half = (target_gap / 2).max(1);

        let expanded_true = self.walk_outward(crossing.soft_true_side, half, hard_true, true);
        let expanded_false = self.walk_outward(
            crossing.soft_false_side,
            target_gap - half,
            hard_true,
            false,
        );

        let st = expanded_true.unwrap_or(crossing.soft_true_side);
        let sf = expanded_false.unwrap_or(crossing.soft_false_side);

        BracketPair {
            soft_true_side: st,
            soft_false_side: sf,
            hard_dist: crossing.hard_dist,
        }
    }

    /// BFS from `origin` up to `depth` steps, staying within hard-true cells
    /// that match `want_soft`. Returns the point closest to exactly `depth` steps away.
    fn walk_outward(
        &self,
        origin: Point,
        depth: u32,
        hard_true: &HashSet<Point>,
        want_soft: bool,
    ) -> Option<Point> {
        if depth == 0 {
            return Some(origin);
        }

        let mut visited: HashSet<Point> = HashSet::new();
        let mut queue: VecDeque<(Point, u32)> = VecDeque::new();
        let mut best: Option<(Point, u32)> = None;

        visited.insert(origin);
        queue.push_back((origin, 0));

        while let Some((p, d)) = queue.pop_front() {
            if d > 0 {
                let err = d.abs_diff(depth);
                if best.map_or(true, |(_, best_err)| err < best_err) {
                    best = Some((p, err));
                }
                if err == 0 {
                    return Some(p);
                }
            }

            if d >= depth + 1 {
                continue;
            }

            for nb in neighbors(p) {
                if visited.contains(&nb) {
                    continue;
                }
                let is_hard = if hard_true.contains(&nb) {
                    true
                } else {
                    (self.hard)(nb.0, nb.1)
                };
                if !is_hard {
                    continue;
                }
                if (self.soft)(nb.0, nb.1) != want_soft {
                    continue;
                }
                visited.insert(nb);
                queue.push_back((nb, d + 1));
            }
        }

        best.map(|(p, _)| p)
    }
}

fn make_inputs(lengths: &Vec<(usize, usize)>) -> Vec<(Vec<u64>, Vec<u64>)> {
    let mut rng = rand::thread_rng();
    let mut inputs: Vec<(Vec<u64>, Vec<u64>)> = Vec::with_capacity(lengths.len());
    for (l, s) in lengths {
        inputs.push((random_limbs(*l, &mut rng), random_limbs(*s, &mut rng)));
    }
    inputs
}

fn ratio_bench(
    iters: u64,
    inputs: &[(Vec<u64>, Vec<u64>)],
    func_a: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> u64,
    func_b: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> u64,
) -> Duration {
    alternating_ratio_bench(iters, inputs, |(l, s), a_first| {
        let out_len = l.len() + s.len() - 1;
        let mut out_a = vec![0u64; out_len];
        let mut out_b = vec![0u64; out_len];

        time_pair_alternating(
            a_first,
            || {
                func_a(
                    black_box(l.as_slice()),
                    black_box(s.as_slice()),
                    black_box(out_a.as_mut_slice()),
                );
            },
            || {
                func_b(
                    black_box(l.as_slice()),
                    black_box(s.as_slice()),
                    black_box(out_b.as_mut_slice()),
                );
            },
        )
    })
}

fn avg_input(lengths: &Vec<(usize, usize)>) -> (f64, f64) {
    let mut sum_l = 0;
    let mut sum_s = 0;
    for (l, s) in lengths {
        sum_l += l;
        sum_s += s;
    }
    let n = lengths.len() as f64;
    (sum_l as f64 / n, sum_s as f64 / n)
}

const NUM_OF_LENGTHS: usize = 128;
const GAP: u32 = 4;

// Multiplication cutoff probes

fn bench_school_to_chunking_karatsuba(c: &mut Criterion) {
    let lengths = BoundarySearch::new(|l, s| (s <= (l + 1) / 2) && (s > 2), |l, s| is_school(l, s))
        .find((50, 20), NUM_OF_LENGTHS, GAP);
    let inputs = make_inputs(&lengths);
    assert!(!inputs.is_empty(), "find inputs failed");
    println!("Average Input: {:?}", avg_input(&lengths));
    println!("Number of Inputs: {}", inputs.len());
    println!("CHUNKING KARATSUBA CUTOFF = {FFT_CHUNKING_KARATSUBA_CUTOFF}");

    c.bench_function(&format!("school_to_chunking_karatsuba/{ARCH}"), |b| {
        b.iter_custom(|iters| {
            ratio_bench(
                iters,
                &inputs,
                &mut |a, b, out| mul_buf(a, b, out),
                &mut |a, b, out| karatsuba_entry_dyn(a, b, out),
            )
        })
    });
}

fn bench_school_to_karatsuba(c: &mut Criterion) {
    let lengths = BoundarySearch::new(
        |l, s| (l >= s) && (s > (l + 1) / 2) && (s > 2),
        |l, s| is_school(l, s),
    )
    .find((20, 18), NUM_OF_LENGTHS, GAP);
    let inputs = make_inputs(&lengths);
    assert!(!inputs.is_empty(), "find inputs failed");
    println!("Average Input: {:?}", avg_input(&lengths));
    println!("Number of Inputs: {}", inputs.len());
    println!("KARATSUBA CUTOFF = {KARATSUBA_CUTOFF}");

    c.bench_function(&format!("school_to_karatsuba/{ARCH}"), |b| {
        b.iter_custom(|iters| {
            ratio_bench(
                iters,
                &inputs,
                &mut |a, b, out| mul_buf(a, b, out),
                &mut |a, b, out| karatsuba_entry_dyn(a, b, out),
            )
        })
    });
}

fn karatsuba_mul_bench(a: &[u64], b: &[u64], out: &mut [u64], scratch: &mut [u64]) -> u64 {
    let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let half = (long.len() + 1) / 2;
    let (cross, rest) = scratch.split_at_mut(2 * half + 1);
    karatsuba_core(long, short, half, out, cross, rest)
}

fn bench_karatsuba_cutoff(c: &mut Criterion) {
    let lengths = BoundarySearch::new(
        |l, s| (l >= s) && (s > (l + 1) / 2) && (s > 2),
        |l, s| is_school(l, s),
    )
    .find((20, 18), NUM_OF_LENGTHS, GAP);
    let inputs = make_inputs(&lengths);
    assert!(!inputs.is_empty(), "find inputs failed");
    println!("Average Input: {:?}", avg_input(&lengths));
    println!("Number of Inputs: {}", inputs.len());
    println!("KARATSUBA CUTOFF = {KARATSUBA_CUTOFF}");
    let mut scratch = {
        let (l, s) = avg_input(&lengths);
        let sum = (l + s) as usize;
        vec![0; 2 * sum]
    };

    c.bench_function(&format!("karatsuba_cutoff/{ARCH}"), |b| {
        b.iter_custom(|iters| {
            ratio_bench(
                iters,
                &inputs,
                &mut |a, b, out| mul_buf(a, b, out),
                &mut |a, b, out| karatsuba_mul_bench(a, b, out, &mut scratch),
            )
        })
    });
}

fn bench_chunking_karatsuba_to_fft(c: &mut Criterion) {
    let lengths = BoundarySearch::new(
        |l, s| (s <= (l + 1) / 2) && !is_school(l, s),
        |l, s| is_karatsuba(l, s, FFT_CHUNKING_KARATSUBA_CUTOFF, FFT_KARATSUBA_CUTOFF),
    )
    .find((60, 25), NUM_OF_LENGTHS, GAP);
    let inputs = make_inputs(&lengths);
    assert!(!inputs.is_empty(), "find inputs failed");
    println!("Average Input: {:?}", avg_input(&lengths));
    println!("Number of Inputs: {}", inputs.len());
    println!("FFT CHUNKING KARATSUBA CUTOFF = {FFT_CHUNKING_KARATSUBA_CUTOFF}");

    c.bench_function(&format!("chunking_karatsuba_to_fft/{ARCH}"), |b| {
        b.iter_custom(|iters| {
            ratio_bench(
                iters,
                &inputs,
                &mut |a, b, out| karatsuba_entry_dyn(a, b, out),
                &mut |a, b, out| fft_entry(a, b, out),
            )
        })
    });
}

fn bench_karatsuba_to_fft(c: &mut Criterion) {
    let lengths = BoundarySearch::new(
        |l, s| (l >= s) && (s > (l + 1) / 2) && !is_school(l, s),
        |l, s| is_karatsuba(l, s, FFT_CHUNKING_KARATSUBA_CUTOFF, FFT_KARATSUBA_CUTOFF),
    )
    .find((150, 150), NUM_OF_LENGTHS, GAP);
    let inputs = make_inputs(&lengths);
    assert!(!inputs.is_empty(), "find inputs failed");
    println!("Average Input: {:?}", avg_input(&lengths));
    println!("Number of Inputs: {}", inputs.len());
    println!("FFT KARATSUBA CUTOFF = {FFT_KARATSUBA_CUTOFF}");

    c.bench_function(&format!("karatsuba_to_fft/{ARCH}"), |b| {
        b.iter_custom(|iters| {
            ratio_bench(
                iters,
                &inputs,
                &mut |a, b, out| karatsuba_entry_dyn(a, b, out),
                &mut |a, b, out| fft_entry(a, b, out),
            )
        })
    });
}

fn bench_static_chunking_karatsuba_to_ntt(c: &mut Criterion) {
    const N: usize = 7100;
    let lengths = BoundarySearch::new(
        |l, s| (s <= (l + 1) / 2) && !is_school(l, s),
        |l, s| is_karatsuba(l, s, NTT_CHUNKING_KARATSUBA_CUTOFF, NTT_KARATSUBA_CUTOFF),
    )
    .find((1500, 500), NUM_OF_LENGTHS, GAP);
    assert!(!lengths.is_empty(), "find inputs failed");

    let max_out = lengths.iter().map(|&(l, s)| l + s - 1).max().unwrap();
    let max_karatuba_scratch = lengths
        .iter()
        .map(|&(l, s)| find_karatsuba_scratch(l, s))
        .max()
        .unwrap();
    let max_ntt_scratch = lengths
        .iter()
        .map(|&(l, s)| find_max_ntt_size(l + s - 1))
        .max()
        .unwrap();
    assert!(max_out <= N && max_karatuba_scratch <= N && max_ntt_scratch <= N, "N = {N} to small: need out = {max_out}, karatsuba = {max_karatuba_scratch}, ntt = {max_ntt_scratch}\n set N = {}", 3 * max_out.max(max_karatuba_scratch).max(max_ntt_scratch) / 2);

    println!("Average Input: {:?}", avg_input(&lengths));
    println!("Number of Inputs: {}", lengths.len());
    println!("NTT CHUNKING KARATSUBA CUTOFF = {NTT_CHUNKING_KARATSUBA_CUTOFF}");
    let inputs = make_inputs(&lengths);

    c.bench_function(&format!("static_chunking_karatsuba_to_ntt/{ARCH}"), |b| {
        b.iter_custom(|iters| {
            ratio_bench(
                iters,
                &inputs,
                &mut |a, b, out| karatsuba_entry_static::<N>(a, b, out),
                &mut |a, b, out| ntt_entry_static::<N>(a, b, out),
            )
        })
    });
}

fn bench_static_karatsuba_to_ntt(c: &mut Criterion) {
    const N: usize = 7000;
    let lengths = BoundarySearch::new(
        |l, s| (l >= s) && (s > (l + 1) / 2) && !is_school(l, s),
        |l, s| is_karatsuba(l, s, NTT_CHUNKING_KARATSUBA_CUTOFF, NTT_KARATSUBA_CUTOFF),
    )
    .find((150, 150), NUM_OF_LENGTHS, GAP);
    assert!(!lengths.is_empty(), "find inputs failed");

    let max_out = lengths.iter().map(|&(l, s)| l + s - 1).max().unwrap();
    let max_karatuba_scratch = lengths
        .iter()
        .map(|&(l, s)| find_karatsuba_scratch(l, s))
        .max()
        .unwrap();
    let max_ntt_scratch = lengths
        .iter()
        .map(|&(l, s)| find_max_ntt_size(l + s - 1))
        .max()
        .unwrap();
    assert!(max_out <= N && max_karatuba_scratch <= N && max_ntt_scratch <= N, "N = {N} to small: need out = {max_out}, karatsuba = {max_karatuba_scratch}, ntt = {max_ntt_scratch}\n set N = {}", 3 * max_out.max(max_karatuba_scratch).max(max_ntt_scratch) / 2);

    println!("Average Input: {:?}", avg_input(&lengths));
    println!("Number of Inputs: {}", lengths.len());
    println!("NTT KARATSUBA CUTOFF = {NTT_KARATSUBA_CUTOFF}");
    let inputs = make_inputs(&lengths);

    c.bench_function(&format!("static_karatsuba_to_ntt/{ARCH}"), |b| {
        b.iter_custom(|iters| {
            ratio_bench(
                iters,
                &inputs,
                &mut |a, b, out| karatsuba_entry_static::<N>(a, b, out),
                &mut |a, b, out| ntt_entry_static::<N>(a, b, out),
            )
        })
    });
}

fn make_sqr_inputs(sz: usize, amt: usize) -> Vec<Vec<u64>> {
    let mut rng = rand::thread_rng();
    let min = sz.saturating_sub(amt / 2).max(4);
    let max = sz + amt / 2;
    let mut inputs: Vec<Vec<u64>> = Vec::with_capacity(amt);
    for i in min..max {
        inputs.push(random_limbs(i, &mut rng));
    }
    return inputs;
}

fn sqr_ratio_bench(
    iters: u64,
    inputs: &[Vec<u64>],
    func_a: &mut dyn FnMut(&[u64], &mut [u64]) -> u64,
    func_b: &mut dyn FnMut(&[u64], &mut [u64]) -> u64,
) -> Duration {
    alternating_ratio_bench(iters, inputs, |buf, a_first| {
        let out_len = 2 * buf.len() - 1;
        let mut out_a = vec![0u64; out_len];
        let mut out_b = vec![0u64; out_len];

        time_pair_alternating(
            a_first,
            || {
                func_a(black_box(buf.as_slice()), black_box(out_a.as_mut_slice()));
            },
            || {
                func_b(black_box(buf.as_slice()), black_box(out_b.as_mut_slice()));
            },
        )
    })
}

fn bench_sqr_school_to_fft(c: &mut Criterion) {
    let inputs = make_sqr_inputs(FFT_SQR_CUTOFF, 16);
    assert!(!inputs.is_empty(), "find inputs failed");
    println!("Number of Inputs: {}", inputs.len());
    println!("FFT SQR CUTOFF = {FFT_SQR_CUTOFF}");

    c.bench_function(&format!("sqr_karatsuba_to_fft/{ARCH}"), |b| {
        b.iter_custom(|iters| {
            sqr_ratio_bench(
                iters,
                &inputs,
                &mut |buf, out| sqr_buf(buf, out),
                &mut |buf, out| fft_sqr_entry(buf, out),
            )
        })
    });
}

const BZ_TOP_RATIO_D_WINDOWS: [(usize, usize); 9] = [
    (BZ_CUTOFF + 1, 128),
    (129, 192),
    (193, 256),
    (257, 384),
    (385, 512),
    (513, 768),
    (769, 1024),
    (1025, 1536),
    (1537, 2048),
];
const BZ_TOP_RATIO_Q_RADIUS: usize = 16;
const BZ_TOP_RATIO_SEARCH_BUDGET: usize = 50_000;
const BZ_TOP_RATIO_CASES_PER_WINDOW: usize = 16;
const BZ_TOP_RATIO_SCALES: [f64; 3] = [0.292, 0.295, 0.298];

struct BzTopInput {
    d_len: usize,
    q_len: usize,
    n: Vec<u64>,
    d: Vec<u64>,
}

fn bz_top_q_center(d_len: usize, padded_cost_scale: f64) -> usize {
    (padded_cost_scale * bz_top_block_padded_work(d_len) / d_len as f64)
        .ceil()
        .max(1.0) as usize
}

fn bz_top_boundary_hard(
    d_len: usize,
    q_len: usize,
    padded_cost_scale: f64,
    d_min: usize,
    d_max: usize,
) -> bool {
    if !(d_min..=d_max).contains(&d_len) {
        return false;
    }
    if q_len == 0 || q_len > d_len {
        return false;
    }

    let q_center = bz_top_q_center(d_len, padded_cost_scale).min(d_len);
    let lo = q_center.saturating_sub(BZ_TOP_RATIO_Q_RADIUS).max(1);
    let hi = q_center.saturating_add(BZ_TOP_RATIO_Q_RADIUS).min(d_len);
    (lo..=hi).contains(&q_len)
}

fn bz_top_boundary_lengths(padded_cost_scale: f64) -> Vec<(usize, usize)> {
    let mut lengths = Vec::new();

    for &(d_min, d_max) in &BZ_TOP_RATIO_D_WINDOWS {
        let seed_d = (d_min + d_max) / 2;
        let seed_q = bz_top_q_center(seed_d, padded_cost_scale)
            .min(seed_d)
            .max(1);
        let mut window_lengths = BoundarySearch::new(
            |d_len, q_len| bz_top_boundary_hard(d_len, q_len, padded_cost_scale, d_min, d_max),
            |d_len, q_len| use_bz_for_top_block_with_scale(d_len, q_len, padded_cost_scale),
        )
        .with_budget(BZ_TOP_RATIO_SEARCH_BUDGET)
        .find((seed_d, seed_q), BZ_TOP_RATIO_CASES_PER_WINDOW, GAP);
        lengths.append(&mut window_lengths);
    }

    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

fn make_bz_top_inputs(lengths: &[(usize, usize)]) -> Vec<BzTopInput> {
    let mut rng = rand::thread_rng();
    lengths
        .iter()
        .map(|&(d_len, q_len)| BzTopInput {
            d_len,
            q_len,
            n: random_limbs(d_len + q_len - 1, &mut rng),
            d: random_normalized_limbs(d_len, &mut rng),
        })
        .collect()
}

fn avg_bz_top_input(inputs: &[BzTopInput]) -> (f64, f64) {
    let sum_d = inputs.iter().map(|input| input.d_len).sum::<usize>();
    let sum_q = inputs.iter().map(|input| input.q_len).sum::<usize>();
    let n = inputs.len() as f64;
    (sum_d as f64 / n, sum_q as f64 / n)
}

fn run_bz_top_block_knuth(n: &mut [u64], d: &[u64], out: &mut [u64]) {
    let mut of = 0;
    div_buf_of(n, &mut of, d, out);
    debug_assert_eq!(of, 0);
}

fn run_bz_top_block_padded(n: &mut [u64], d: &[u64], out: &mut [u64]) {
    let dlen = d.len();
    let qlen = out.len();
    let mut scratch = ScratchGuard::acquire();
    let [top_n, q_tmp, div_scratch] = scratch.get_splits([2 * dlen, dlen, dlen]);

    top_n[..n.len()].copy_from_slice(n);
    top_n[n.len()..].fill(0);

    div_2_1(top_n, d, q_tmp, div_scratch, &mut |n, d, q| {
        mul_dyn(n, d, q);
    });

    out.copy_from_slice(&q_tmp[..qlen]);
    n.fill(0);
    n[..dlen].copy_from_slice(&top_n[..dlen]);
}

fn bz_top_block_ratio_bench(iters: u64, inputs: &[BzTopInput]) -> Duration {
    alternating_ratio_bench(iters, inputs, |input, bz_first| {
        let mut n_knuth = input.n.clone();
        let mut n_bz = input.n.clone();
        let mut q_knuth = vec![0u64; input.q_len];
        let mut q_bz = vec![0u64; input.q_len];

        let (t_bz, t_knuth) = time_pair_alternating(
            bz_first,
            || {
                run_bz_top_block_padded(
                    black_box(n_bz.as_mut_slice()),
                    black_box(input.d.as_slice()),
                    black_box(q_bz.as_mut_slice()),
                );
            },
            || {
                run_bz_top_block_knuth(
                    black_box(n_knuth.as_mut_slice()),
                    black_box(input.d.as_slice()),
                    black_box(q_knuth.as_mut_slice()),
                );
            },
        );

        debug_assert_eq!(q_bz, q_knuth);
        debug_assert_eq!(n_bz, n_knuth);

        (t_bz, t_knuth)
    })
}

fn bench_bz_top_block_dispatch_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("bz_top_block_dispatch_ratio/{ARCH}"));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    println!("BZ_CUTOFF = {BZ_CUTOFF}");
    println!("BZ_TOP_PADDED_COST_SCALE = {BZ_TOP_PADDED_COST_SCALE}");
    println!("Scale is the modeled q_len / d_len break-even ratio.");
    println!("Reported value is padded_bz_runtime / knuth_runtime.");
    println!("Below 1.0 means the boundary is conservative; decrease BZ_TOP_PADDED_COST_SCALE.");
    println!("Above 1.0 means the boundary is aggressive; increase BZ_TOP_PADDED_COST_SCALE.");

    for &scale in &BZ_TOP_RATIO_SCALES {
        let lengths = bz_top_boundary_lengths(scale);
        assert!(
            !lengths.is_empty(),
            "find BZ top-block boundary inputs failed"
        );
        let inputs = make_bz_top_inputs(&lengths);
        let (avg_d, avg_q) = avg_bz_top_input(&inputs);

        println!(
            "scale {scale:.3}: inputs={} avg_d_len={avg_d:.1} avg_q_len={avg_q:.1}",
            inputs.len()
        );

        group.bench_with_input(
            BenchmarkId::new("padded_over_knuth", format!("{scale:.3}x")),
            &scale,
            |bench, _| bench.iter_custom(|iters| bz_top_block_ratio_bench(iters, &inputs)),
        );
    }

    group.finish();
}

const NR_SEED_START_PRECISIONS: [usize; 11] = [
    896, 1024, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1408, 1536,
];
const NR_SEED_CASES_PER_PRECISION: usize = 4;

struct NrSeedTradeoffInput {
    denom: Vec<u64>,
}

fn run_nr_prior_seed_plus_step(d: &[u64], rcp: &mut [u64], prior_p: usize) {
    debug_assert!(rcp.len() >= prior_p + 1);
    debug_assert!(rcp.len() <= 2 * prior_p + 1);

    rcp.fill(0);
    bz_rcp_seed_dyn(end_ref(d, 2 * prior_p + 1), end_mut(rcp, prior_p + 1));

    let err_len = 2 * prior_p + 2;
    let mut scratch = ScratchGuard::acquire();
    let [err, cor] = scratch.get_splits([err_len, err_len]);
    let trunc = rcp.len() != 2 * prior_p + 1;
    if !nr_rcp_refine_step_dyn(end_ref(d, 2 * prior_p + 1), rcp, err, cor, prior_p, trunc) {
        bz_rcp_seed_dyn(d, rcp);
    }
}

fn make_nr_seed_tradeoff_inputs(p: usize) -> Vec<NrSeedTradeoffInput> {
    let mut rng = rand::thread_rng();
    (0..NR_SEED_CASES_PER_PRECISION)
        .map(|_| NrSeedTradeoffInput {
            denom: random_normalized_limbs(2 * p + 1, &mut rng),
        })
        .collect()
}

fn nr_seed_tradeoff_ratio_bench(iters: u64, inputs: &[NrSeedTradeoffInput], p: usize) -> Duration {
    let prior_p = p.div_ceil(2);
    alternating_ratio_bench(iters, inputs, |input, refined_first| {
        let mut direct = vec![0u64; p + 1];
        let mut refined = vec![0u64; p + 1];

        time_pair_alternating(
            refined_first,
            || {
                run_nr_prior_seed_plus_step(
                    black_box(input.denom.as_slice()),
                    black_box(refined.as_mut_slice()),
                    black_box(prior_p),
                );
            },
            || {
                bz_rcp_seed_dyn(
                    black_box(input.denom.as_slice()),
                    black_box(direct.as_mut_slice()),
                );
            },
        )
    })
}

fn bench_nr_seed_start_precision_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("nr_seed_start_precision_ratio/{ARCH}"));
    group.sample_size(60);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    println!("Reported value is (seed p/2 + one NR step to p) / direct seed at p.");
    println!("Below 1.0 favors lowering the direct-seed cutoff below p.");
    println!("Above 1.0 favors allowing direct seeds at p.");

    for &p in &NR_SEED_START_PRECISIONS {
        let inputs = make_nr_seed_tradeoff_inputs(p);
        println!(
            "p={p}: prior_p={} denom_len={} inputs={}",
            p.div_ceil(2),
            2 * p + 1,
            inputs.len()
        );

        group.bench_with_input(
            BenchmarkId::new("half_step_over_direct", p),
            &p,
            |bench, &p| bench.iter_custom(|iters| nr_seed_tradeoff_ratio_bench(iters, &inputs, p)),
        );
    }

    group.finish();
}

// Criterion setup.

fn cutoff_criterion() -> Criterion {
    Criterion::default()
        .sample_size(250)
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(10))
}

criterion_group! {
    name = benches;
    config = cutoff_criterion();
    targets = bench_nr_seed_start_precision_ratio
}
criterion_main!(benches);
