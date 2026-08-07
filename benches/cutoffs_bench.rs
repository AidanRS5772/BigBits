#![allow(dead_code)]

use big_bits::utils::sqrt::{binom_sqrt_core, correct_sqrt, reduce_sqrt_rem};
use big_bits::{utils::div::*, utils::*, *};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
    time::{Duration, Instant},
};

const ARCH: &'static str = std::env::consts::ARCH;

pub type Point = (usize, usize);

// Shared benchmark helpers.

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

fn bz_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        bz_div_rem_wrapper_dyn(n, d, q, request);
    }
}

fn nr_div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        nr_div_rem_wrapper_dyn(n, d, q, request);
    }
}

fn bz_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        bz_div_wrapper_static::<N>(n, d, q, request);
    }
}

fn nr_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        nr_div_wrapper_static::<N>(n, d, q, request);
    }
}

fn bz_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        bz_div_rem_wrapper_static::<N>(n, d, q, request);
    }
}

fn nr_div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) {
    if let Some(request) = division_preflight(n, d, q) {
        nr_div_rem_wrapper_static::<N>(n, d, q, request);
    }
}

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
        crossings.sort_unstable_by_key(|c| (c.hard_dist, c.soft_true_side, c.soft_false_side));

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

fn use_bz_for_top_block_with_scale(d_len: usize, q_len: usize, padded_cost_scale: f64) -> bool {
    d_len > BZ_CUTOFF
        && q_len != 0
        && padded_cost_scale * bz_2_1_cost(d_len) <= bz_top_block_knuth_work(d_len, q_len)
}

struct BzTopInput {
    d_len: usize,
    q_len: usize,
    n: Vec<u64>,
    d: Vec<u64>,
}

fn bz_top_q_center(d_len: usize, padded_cost_scale: f64) -> usize {
    (padded_cost_scale * bz_2_1_cost(d_len) / d_len as f64)
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
    knuth_div_rcp_seed_dyn(end_ref(d, 2 * prior_p + 1), end_mut(rcp, prior_p + 1));

    let err_len = 2 * prior_p + 2;
    let mut scratch = ScratchGuard::acquire();
    let [err, cor] = scratch.get_splits([err_len, err_len]);
    let trunc = rcp.len() != 2 * prior_p + 1;
    if !nr_refine_rcp(
        end_ref(d, 2 * prior_p + 1),
        rcp,
        err,
        cor,
        prior_p,
        trunc,
        &mut |a, b, o| mid_mul_dyn(a, b, o),
        &mut |a, b, o| short_mul_dyn(a, b, o),
    ) {
        knuth_div_rcp_seed_dyn(d, rcp);
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
                knuth_div_rcp_seed_dyn(
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

// Division BZ/NR cost-model tuning.
//
// Set:
//   DIV_TUNE_REGIME=karatsuba|transform|transition|dispatch|reciprocal
//   DIV_TUNE_FAMILY=dyn_div|dyn_rem|static_div|static_rem|dyn_rcp|static_rcp
//   DIV_TUNE_VALUES=comma-separated candidate ratios or reciprocal precisions
// For transition scans, also set DIV_TUNE_KARATSUBA_RATIO and
// DIV_TUNE_TRANSFORM_RATIO. Boundary and transition durations encode
// NR_runtime / BZ_runtime, with 1.000 ms as the break-even point. Dispatch
// durations encode tuned_runtime / original_guessed_runtime. Reciprocal
// durations encode NR_runtime / Knuth_runtime.

const DIV_TUNE_STATIC_CAPACITY: usize = 16_384;
const DIV_TUNE_CASES_PER_SCALE: usize = 8;
const DIV_TUNE_SEARCH_BUDGET: usize = 20_000;
const DIV_TUNE_KARATSUBA_D_CENTERS: [usize; 7] = [96, 128, 192, 256, 384, 512, 768];
const DIV_TUNE_TRANSFORM_D_CENTERS: [usize; 5] = [1024, 1280, 1536, 1792, 2048];
const DIV_TUNE_TRANSITION_SCALES: [usize; 12] = [
    256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048, 3072, 4096,
];

#[derive(Clone, Copy, Debug)]
enum DivisionTuneFamily {
    DynDiv,
    DynRem,
    StaticDiv,
    StaticRem,
    DynRcp,
    StaticRcp,
}

impl DivisionTuneFamily {
    fn from_env() -> Self {
        match env::var("DIV_TUNE_FAMILY").as_deref() {
            Ok("dyn_rem") => Self::DynRem,
            Ok("static_div") => Self::StaticDiv,
            Ok("static_rem") => Self::StaticRem,
            Ok("dyn_rcp") => Self::DynRcp,
            Ok("static_rcp") => Self::StaticRcp,
            Ok("dyn_div") | Err(_) => Self::DynDiv,
            Ok(other) => panic!("unknown DIV_TUNE_FAMILY={other}"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::DynDiv => "dyn_div",
            Self::DynRem => "dyn_rem",
            Self::StaticDiv => "static_div",
            Self::StaticRem => "static_rem",
            Self::DynRcp => "dyn_rcp",
            Self::StaticRcp => "static_rcp",
        }
    }

    fn is_reciprocal(self) -> bool {
        matches!(self, Self::DynRcp | Self::StaticRcp)
    }
}

#[derive(Clone, Copy, Debug)]
enum DivisionCostRegime {
    Karatsuba,
    Transform,
}

impl DivisionCostRegime {
    fn label(self) -> &'static str {
        match self {
            Self::Karatsuba => "karatsuba",
            Self::Transform => "transform",
        }
    }

    fn selects_bz(self, q_len: usize, d_len: usize, parameter: f64) -> bool {
        match self {
            Self::Karatsuba => (q_len as f64) * parameter > d_len as f64,
            Self::Transform => (q_len as f64).log2() * parameter > (d_len as f64).log2().powi(2),
        }
    }

    fn q_center(self, d_len: usize, parameter: f64) -> Option<usize> {
        let q = match self {
            Self::Karatsuba => d_len as f64 / parameter,
            Self::Transform => {
                let log_d = (d_len as f64).log2();
                2.0f64.powf(log_d * log_d / parameter)
            }
        };
        if !q.is_finite() || q < 8.0 || q > usize::MAX as f64 {
            None
        } else {
            Some(q.round() as usize)
        }
    }
}

struct DivisionTuneInput {
    q_len: usize,
    d_len: usize,
    n: Vec<u64>,
    d: Vec<u64>,
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an f64, got {value}"))
    })
}

fn env_f64_values(default: &[f64]) -> Vec<f64> {
    env::var("DIV_TUNE_VALUES").map_or_else(
        |_| default.to_vec(),
        |values| {
            values
                .split(',')
                .map(|value| {
                    value
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("invalid DIV_TUNE_VALUES entry {value}"))
                })
                .collect()
        },
    )
}

fn env_usize_values(name: &str, default: &[usize]) -> Vec<usize> {
    env::var(name).map_or_else(
        |_| default.to_vec(),
        |values| {
            values
                .split(',')
                .map(|value| {
                    value
                        .trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("invalid {name} entry {value}"))
                })
                .collect()
        },
    )
}

fn division_boundary_lengths_at(
    regime: DivisionCostRegime,
    parameter: f64,
    d_center: usize,
) -> Vec<(usize, usize)> {
    assert!(parameter.is_finite() && parameter > 0.0);
    let Some(q_center) = regime.q_center(d_center, parameter) else {
        return Vec::new();
    };
    if q_center + d_center - 1 > DIV_TUNE_STATIC_CAPACITY {
        return Vec::new();
    }

    let d_radius = 8;
    let q_radius = (q_center / 32).clamp(8, 256);
    let d_min = d_center.saturating_sub(d_radius).max(BZ_CUTOFF + 1);
    let d_max = d_center + d_radius;
    let q_min = q_center.saturating_sub(q_radius).max(8);
    let q_max = q_center.saturating_add(q_radius);
    let hard = |q_len: usize, d_len: usize| {
        (q_min..=q_max).contains(&q_len)
            && (d_min..=d_max).contains(&d_len)
            && q_len + d_len - 1 <= DIV_TUNE_STATIC_CAPACITY
    };

    BoundarySearch::new(hard, |q_len, d_len| {
        regime.selects_bz(q_len, d_len, parameter)
    })
    .with_budget(DIV_TUNE_SEARCH_BUDGET)
    .find((q_center, d_center), DIV_TUNE_CASES_PER_SCALE, GAP)
}

fn division_boundary_lengths(regime: DivisionCostRegime, parameter: f64) -> Vec<(usize, usize)> {
    let centers: &[usize] = match regime {
        DivisionCostRegime::Karatsuba => &DIV_TUNE_KARATSUBA_D_CENTERS,
        DivisionCostRegime::Transform => &DIV_TUNE_TRANSFORM_D_CENTERS,
    };
    let mut lengths = Vec::new();
    for &d_center in centers {
        lengths.extend(division_boundary_lengths_at(regime, parameter, d_center));
    }
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

fn division_boundary_lengths_at_scale(
    regime: DivisionCostRegime,
    parameter: f64,
    scale: usize,
) -> Vec<(usize, usize)> {
    assert!(parameter.is_finite() && parameter > 0.0);
    let (q_center, d_center) = match regime {
        DivisionCostRegime::Karatsuba if parameter <= 1.0 => {
            (scale, (scale as f64 * parameter).round() as usize)
        }
        DivisionCostRegime::Karatsuba => ((scale as f64 / parameter).round() as usize, scale),
        DivisionCostRegime::Transform => {
            let log_q = (scale as f64).log2();
            let d_at_q_scale = 2.0f64.powf((parameter * log_q).sqrt()).round() as usize;
            if d_at_q_scale <= scale {
                (scale, d_at_q_scale)
            } else {
                let q_at_d_scale = regime.q_center(scale, parameter).unwrap();
                (q_at_d_scale, scale)
            }
        }
    };
    if d_center <= BZ_CUTOFF || q_center + d_center - 1 > DIV_TUNE_STATIC_CAPACITY {
        return Vec::new();
    }

    let q_radius = (q_center / 128).clamp(4, 64);
    let d_radius = (d_center / 128).clamp(4, 64);
    let q_min = q_center.saturating_sub(q_radius).max(8);
    let q_max = q_center.saturating_add(q_radius);
    let d_min = d_center.saturating_sub(d_radius).max(BZ_CUTOFF + 1);
    let d_max = d_center.saturating_add(d_radius);
    let hard = |q_len: usize, d_len: usize| {
        (q_min..=q_max).contains(&q_len)
            && (d_min..=d_max).contains(&d_len)
            && q_len.max(d_len).abs_diff(scale) <= q_radius
            && q_len + d_len - 1 <= DIV_TUNE_STATIC_CAPACITY
    };

    BoundarySearch::new(hard, |q_len, d_len| {
        regime.selects_bz(q_len, d_len, parameter)
    })
    .with_budget(DIV_TUNE_SEARCH_BUDGET)
    .find((q_center, d_center), DIV_TUNE_CASES_PER_SCALE, GAP)
}

fn make_division_tuning_inputs(lengths: &[(usize, usize)], seed: u64) -> Vec<DivisionTuneInput> {
    let mut rng = StdRng::seed_from_u64(seed);
    lengths
        .iter()
        .map(|&(q_len, d_len)| DivisionTuneInput {
            q_len,
            d_len,
            n: random_limbs(d_len + q_len - 1, &mut rng),
            d: random_limbs(d_len, &mut rng),
        })
        .collect()
}

fn avg_division_tuning_input(inputs: &[DivisionTuneInput]) -> (f64, f64) {
    let count = inputs.len() as f64;
    let q_sum = inputs.iter().map(|input| input.q_len).sum::<usize>();
    let d_sum = inputs.iter().map(|input| input.d_len).sum::<usize>();
    (q_sum as f64 / count, d_sum as f64 / count)
}

fn division_tuning_ratio_bench(
    iters: u64,
    inputs: &[DivisionTuneInput],
    family: DivisionTuneFamily,
) -> Duration {
    alternating_ratio_bench(iters, inputs, |input, nr_first| {
        let mut q_nr = vec![0u64; input.q_len];
        let mut q_bz = vec![0u64; input.q_len];

        match family {
            DivisionTuneFamily::DynDiv => time_pair_alternating(
                nr_first,
                || {
                    nr_div_dyn(
                        black_box(input.n.as_slice()),
                        black_box(input.d.as_slice()),
                        black_box(q_nr.as_mut_slice()),
                    );
                },
                || {
                    bz_div_dyn(
                        black_box(input.n.as_slice()),
                        black_box(input.d.as_slice()),
                        black_box(q_bz.as_mut_slice()),
                    );
                },
            ),
            DivisionTuneFamily::DynRem => {
                let mut n_nr = input.n.clone();
                let mut n_bz = input.n.clone();
                time_pair_alternating(
                    nr_first,
                    || {
                        nr_div_rem_dyn(
                            black_box(n_nr.as_mut_slice()),
                            black_box(input.d.as_slice()),
                            black_box(q_nr.as_mut_slice()),
                        );
                    },
                    || {
                        bz_div_rem_dyn(
                            black_box(n_bz.as_mut_slice()),
                            black_box(input.d.as_slice()),
                            black_box(q_bz.as_mut_slice()),
                        );
                    },
                )
            }
            DivisionTuneFamily::StaticDiv => time_pair_alternating(
                nr_first,
                || {
                    nr_div_static::<DIV_TUNE_STATIC_CAPACITY>(
                        black_box(input.n.as_slice()),
                        black_box(input.d.as_slice()),
                        black_box(q_nr.as_mut_slice()),
                    );
                },
                || {
                    bz_div_static::<DIV_TUNE_STATIC_CAPACITY>(
                        black_box(input.n.as_slice()),
                        black_box(input.d.as_slice()),
                        black_box(q_bz.as_mut_slice()),
                    );
                },
            ),
            DivisionTuneFamily::StaticRem => {
                let mut n_nr = input.n.clone();
                let mut n_bz = input.n.clone();
                time_pair_alternating(
                    nr_first,
                    || {
                        nr_div_rem_static::<DIV_TUNE_STATIC_CAPACITY>(
                            black_box(n_nr.as_mut_slice()),
                            black_box(input.d.as_slice()),
                            black_box(q_nr.as_mut_slice()),
                        );
                    },
                    || {
                        bz_div_rem_static::<DIV_TUNE_STATIC_CAPACITY>(
                            black_box(n_bz.as_mut_slice()),
                            black_box(input.d.as_slice()),
                            black_box(q_bz.as_mut_slice()),
                        );
                    },
                )
            }
            DivisionTuneFamily::DynRcp | DivisionTuneFamily::StaticRcp => {
                unreachable!("reciprocal family used for division tuning")
            }
        }
    })
}

fn register_division_boundary_candidates(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    family: DivisionTuneFamily,
    regime: DivisionCostRegime,
    candidates: &[f64],
) {
    for &candidate in candidates {
        let lengths = division_boundary_lengths(regime, candidate);
        assert!(
            !lengths.is_empty(),
            "no boundary inputs for {} candidate {candidate}",
            regime.label()
        );
        let inputs = make_division_tuning_inputs(
            &lengths,
            candidate.to_bits() ^ (family as u64).wrapping_mul(0x9e37_79b9),
        );
        let (avg_q, avg_d) = avg_division_tuning_input(&inputs);
        println!(
            "{} {} candidate={candidate:.4}: inputs={} avg_q={avg_q:.1} avg_d={avg_d:.1}",
            family.label(),
            regime.label(),
            inputs.len()
        );

        group.bench_with_input(
            BenchmarkId::new(
                format!("{}/{}", family.label(), regime.label()),
                format!("{candidate:.4}"),
            ),
            &candidate,
            |bench, _| {
                bench.iter_custom(|iters| division_tuning_ratio_bench(iters, &inputs, family))
            },
        );
    }
}

fn register_division_transition_scan(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    family: DivisionTuneFamily,
) {
    let karatsuba_ratio = env_f64("DIV_TUNE_KARATSUBA_RATIO", 1.0);
    let transform_ratio = env_f64("DIV_TUNE_TRANSFORM_RATIO", 10.0);
    let scales = env_usize_values("DIV_TUNE_SCALES", &DIV_TUNE_TRANSITION_SCALES);

    println!(
        "{} transition scan: karatsuba_ratio={karatsuba_ratio:.4} transform_ratio={transform_ratio:.4}",
        family.label()
    );
    for scale in scales {
        for (regime, parameter) in [
            (DivisionCostRegime::Karatsuba, karatsuba_ratio),
            (DivisionCostRegime::Transform, transform_ratio),
        ] {
            let lengths = division_boundary_lengths_at_scale(regime, parameter, scale);
            if lengths.is_empty() {
                continue;
            }
            let inputs = make_division_tuning_inputs(
                &lengths,
                (scale as u64) ^ parameter.to_bits() ^ (family as u64).wrapping_mul(0x9e37_79b9),
            );
            let (avg_q, avg_d) = avg_division_tuning_input(&inputs);
            println!(
                "{} transition {} scale={scale}: inputs={} avg_q={avg_q:.1} avg_d={avg_d:.1}",
                family.label(),
                regime.label(),
                inputs.len()
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}/transition/{}", family.label(), regime.label()),
                    scale,
                ),
                &scale,
                |bench, _| {
                    bench.iter_custom(|iters| division_tuning_ratio_bench(iters, &inputs, family))
                },
            );
        }
    }
}

#[derive(Clone, Copy)]
struct DivisionTuneCutoffs {
    karatsuba_transform: usize,
    karatsuba_ratio: f64,
    transform_ratio: f64,
}

fn tuned_division_cutoffs(family: DivisionTuneFamily) -> DivisionTuneCutoffs {
    match family {
        DivisionTuneFamily::DynDiv => DivisionTuneCutoffs {
            karatsuba_transform: DYN_DIV_KARATSUBA_FFT_NR_BZ_CUTOFF,
            karatsuba_ratio: DYN_DIV_KARATSUBA_NR_BZ_CUTOFF,
            transform_ratio: DYN_DIV_FFT_NR_BZ_CUTOFF,
        },
        DivisionTuneFamily::DynRem => DivisionTuneCutoffs {
            karatsuba_transform: DYN_DIV_REM_KARATSUBA_FFT_NR_BZ_CUTOFF,
            karatsuba_ratio: DYN_DIV_REM_KARATSUBA_NR_BZ_CUTOFF,
            transform_ratio: DYN_DIV_REM_FFT_NR_BZ_CUTOFF,
        },
        DivisionTuneFamily::StaticDiv => DivisionTuneCutoffs {
            karatsuba_transform: STATIC_DIV_KARATSUBA_NTT_NR_BZ_CUTOFF,
            karatsuba_ratio: STATIC_DIV_KARATSUBA_NR_BZ_CUTOFF,
            transform_ratio: STATIC_DIV_NTT_NR_BZ_CUTOFF,
        },
        DivisionTuneFamily::StaticRem => DivisionTuneCutoffs {
            karatsuba_transform: STATIC_DIV_REM_KARATSUBA_NTT_NR_BZ_CUTOFF,
            karatsuba_ratio: STATIC_DIV_REM_KARATSUBA_NR_BZ_CUTOFF,
            transform_ratio: STATIC_DIV_REM_NTT_NR_BZ_CUTOFF,
        },
        DivisionTuneFamily::DynRcp | DivisionTuneFamily::StaticRcp => unreachable!(),
    }
}

fn cost_model_selects_bz(q_len: usize, d_len: usize, tuning: DivisionTuneCutoffs) -> bool {
    if q_len.max(d_len) < tuning.karatsuba_transform {
        (q_len as f64) * tuning.karatsuba_ratio > d_len as f64
    } else {
        (q_len as f64).log2() * tuning.transform_ratio > (d_len as f64).log2().powi(2)
    }
}

fn run_division_choice(
    family: DivisionTuneFamily,
    use_bz: bool,
    n: &mut [u64],
    d: &[u64],
    q: &mut [u64],
) {
    match (family, use_bz) {
        (DivisionTuneFamily::DynDiv, true) => bz_div_dyn(n, d, q),
        (DivisionTuneFamily::DynDiv, false) => nr_div_dyn(n, d, q),
        (DivisionTuneFamily::DynRem, true) => bz_div_rem_dyn(n, d, q),
        (DivisionTuneFamily::DynRem, false) => nr_div_rem_dyn(n, d, q),
        (DivisionTuneFamily::StaticDiv, true) => bz_div_static::<DIV_TUNE_STATIC_CAPACITY>(n, d, q),
        (DivisionTuneFamily::StaticDiv, false) => {
            nr_div_static::<DIV_TUNE_STATIC_CAPACITY>(n, d, q)
        }
        (DivisionTuneFamily::StaticRem, true) => {
            bz_div_rem_static::<DIV_TUNE_STATIC_CAPACITY>(n, d, q)
        }
        (DivisionTuneFamily::StaticRem, false) => {
            nr_div_rem_static::<DIV_TUNE_STATIC_CAPACITY>(n, d, q)
        }
        (DivisionTuneFamily::DynRcp | DivisionTuneFamily::StaticRcp, _) => unreachable!(),
    }
}

fn dispatch_comparison_ratio_bench(
    iters: u64,
    inputs: &[DivisionTuneInput],
    family: DivisionTuneFamily,
) -> Duration {
    let tuned = tuned_division_cutoffs(family);
    let guessed = DivisionTuneCutoffs {
        karatsuba_transform: 1000,
        karatsuba_ratio: 1.0,
        transform_ratio: 1.0,
    };
    alternating_ratio_bench(iters, inputs, |input, tuned_first| {
        let tuned_bz = cost_model_selects_bz(input.q_len, input.d_len, tuned);
        let guessed_bz = cost_model_selects_bz(input.q_len, input.d_len, guessed);
        debug_assert_ne!(tuned_bz, guessed_bz);

        let mut n_tuned = input.n.clone();
        let mut n_guessed = input.n.clone();
        let mut q_tuned = vec![0u64; input.q_len];
        let mut q_guessed = vec![0u64; input.q_len];
        time_pair_alternating(
            tuned_first,
            || {
                run_division_choice(
                    family,
                    tuned_bz,
                    black_box(&mut n_tuned),
                    black_box(&input.d),
                    black_box(&mut q_tuned),
                )
            },
            || {
                run_division_choice(
                    family,
                    guessed_bz,
                    black_box(&mut n_guessed),
                    black_box(&input.d),
                    black_box(&mut q_guessed),
                )
            },
        )
    })
}

fn register_dispatch_comparison(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    family: DivisionTuneFamily,
) {
    let tuned = tuned_division_cutoffs(family);
    let guessed = DivisionTuneCutoffs {
        karatsuba_transform: 1000,
        karatsuba_ratio: 1.0,
        transform_ratio: 1.0,
    };
    let d_centers = env_usize_values("DIV_TUNE_SCALES", &[128, 256, 512, 768, 1024, 1536, 2048]);
    let q_over_d = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0];

    for d_len in d_centers {
        let mut lengths = q_over_d
            .iter()
            .map(|ratio| ((d_len as f64 * ratio).round().max(8.0) as usize, d_len))
            .filter(|&(q_len, d_len)| {
                q_len + d_len - 1 <= DIV_TUNE_STATIC_CAPACITY
                    && cost_model_selects_bz(q_len, d_len, tuned)
                        != cost_model_selects_bz(q_len, d_len, guessed)
            })
            .collect::<Vec<_>>();
        lengths.sort_unstable();
        lengths.dedup();
        if lengths.is_empty() {
            continue;
        }
        println!(
            "{} tuned-vs-guess d_len={d_len}: differing_shapes={} {lengths:?}",
            family.label(),
            lengths.len()
        );
        for &(q_len, d_len) in &lengths {
            let inputs = make_division_tuning_inputs(
                &[(q_len, d_len)],
                (q_len as u64).rotate_left(17)
                    ^ d_len as u64
                    ^ (family as u64).wrapping_mul(0x9e37_79b9),
            );
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}/tuned_over_guess", family.label()),
                    format!("d{d_len}_q{q_len}"),
                ),
                &(q_len, d_len),
                |bench, _| {
                    bench.iter_custom(|iters| {
                        dispatch_comparison_ratio_bench(iters, &inputs, family)
                    })
                },
            );
        }
    }
}

struct ReciprocalTuneInput {
    d: Vec<u64>,
    precision: usize,
}

fn reciprocal_tuning_ratio_bench(
    iters: u64,
    inputs: &[ReciprocalTuneInput],
    family: DivisionTuneFamily,
) -> Duration {
    alternating_ratio_bench(iters, inputs, |input, nr_first| {
        let mut nr = vec![0u64; input.precision];
        let mut knuth = vec![0u64; input.precision];
        let inner_repetitions = (512 / input.precision.max(1)).clamp(1, 256);
        time_pair_alternating(
            nr_first,
            || {
                for _ in 0..inner_repetitions {
                    match family {
                        DivisionTuneFamily::DynRcp => {
                            nr_rcp_wrapper_dyn(black_box(&input.d), black_box(&mut nr))
                        }
                        DivisionTuneFamily::StaticRcp => nr_rcp_wrapper_static::<
                            DIV_TUNE_STATIC_CAPACITY,
                        >(
                            black_box(&input.d), black_box(&mut nr)
                        ),
                        _ => unreachable!(),
                    }
                }
            },
            || {
                for _ in 0..inner_repetitions {
                    match family {
                        DivisionTuneFamily::DynRcp => {
                            knuth_rcp_wrapper_dyn(black_box(&input.d), black_box(&mut knuth))
                        }
                        DivisionTuneFamily::StaticRcp => {
                            knuth_rcp_wrapper_static::<DIV_TUNE_STATIC_CAPACITY>(
                                black_box(&input.d),
                                black_box(&mut knuth),
                            )
                        }
                        _ => unreachable!(),
                    }
                }
            },
        )
    })
}

fn register_reciprocal_scan(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    family: DivisionTuneFamily,
) {
    assert!(family.is_reciprocal());
    let defaults: &[usize] = match family {
        DivisionTuneFamily::DynRcp => &[1, 2, 4, 5, 6, 7, 8, 9, 12, 16, 32, 64],
        DivisionTuneFamily::StaticRcp => &[64, 80, 96, 100, 101, 102, 104, 112, 128, 192, 256],
        _ => unreachable!(),
    };
    let precisions = env_usize_values("DIV_TUNE_VALUES", defaults);
    for precision in precisions {
        assert!(precision <= DIV_TUNE_STATIC_CAPACITY);
        let mut rng =
            StdRng::seed_from_u64((precision as u64) ^ (family as u64).wrapping_mul(0x9e37_79b9));
        let inputs = (0..4)
            .map(|_| ReciprocalTuneInput {
                d: random_limbs(precision + 1, &mut rng),
                precision,
            })
            .collect::<Vec<_>>();
        println!(
            "{} precision={precision} inputs={}",
            family.label(),
            inputs.len()
        );
        group.bench_with_input(
            BenchmarkId::new(family.label(), precision),
            &precision,
            |bench, _| {
                bench.iter_custom(|iters| reciprocal_tuning_ratio_bench(iters, &inputs, family))
            },
        );
    }
}

fn bench_division_tuning(c: &mut Criterion) {
    if env::var_os("SQRT_TUNE_FAMILY").is_some() {
        return;
    }
    let family = DivisionTuneFamily::from_env();
    let regime = env::var("DIV_TUNE_REGIME").unwrap_or_else(|_| "karatsuba".to_owned());
    let mut group = c.benchmark_group(format!("division_tuning/{ARCH}"));
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(750));
    group.measurement_time(Duration::from_secs(2));
    group.noise_threshold(0.02);

    if regime == "dispatch" {
        println!("Reported value is tuned_runtime / guessed_runtime.");
        println!("Below 1.000 ms favors the tuned dispatch.");
    } else if regime == "reciprocal" {
        println!("Reported value is NR_runtime / Knuth_runtime; 1.000 ms is break-even.");
        println!("Below 1.000 ms favors NR; above 1.000 ms favors Knuth.");
    } else {
        println!("Reported value is NR_runtime / BZ_runtime; 1.000 ms is break-even.");
        println!("Below 1.000 ms favors NR; above 1.000 ms favors BZ.");
    }
    match regime.as_str() {
        "karatsuba" => {
            assert!(!family.is_reciprocal());
            let candidates = env_f64_values(&[0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.05]);
            register_division_boundary_candidates(
                &mut group,
                family,
                DivisionCostRegime::Karatsuba,
                &candidates,
            );
        }
        "transform" => {
            assert!(!family.is_reciprocal());
            let candidates = env_f64_values(&[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
            register_division_boundary_candidates(
                &mut group,
                family,
                DivisionCostRegime::Transform,
                &candidates,
            );
        }
        "transition" => {
            assert!(!family.is_reciprocal());
            register_division_transition_scan(&mut group, family);
        }
        "dispatch" => {
            assert!(!family.is_reciprocal());
            register_dispatch_comparison(&mut group, family);
        }
        "reciprocal" => register_reciprocal_scan(&mut group, family),
        other => panic!("unknown DIV_TUNE_REGIME={other}"),
    }
    group.finish();
}

// Zimmermann/binomial square-root cutoff tuning.
//
// Set:
//   SQRT_TUNE_FAMILY=dyn|static (required, keeping the two runs independent)
//   SQRT_TUNE_VALUES=comma-separated root lengths
//
// Each result encodes one-Zimmermann-descent runtime / binomial runtime as a
// duration relative to 1.000 ms. Values below 1.000 ms favor descending.

const SQRT_TUNE_CASES_PER_POINT: usize = 4;
const SQRT_TUNE_STATIC_CAPACITY: usize = 512;
const SQRT_TUNE_DEFAULT_LENGTHS: [usize; 17] = [
    12, 14, 15, 16, 17, 18, 19, 20, 22, 24, 28, 32, 36, 40, 48, 56, 64,
];

#[derive(Clone, Copy)]
enum SqrtTuneFamily {
    Dyn,
    Static,
}

impl SqrtTuneFamily {
    fn label(self) -> String {
        match self {
            Self::Dyn => "dyn".to_owned(),
            Self::Static => format!("static_n{SQRT_TUNE_STATIC_CAPACITY}"),
        }
    }
}

struct SqrtTuneInput {
    x: Vec<u64>,
    root_len: usize,
}

fn sqrt_tune_families() -> Vec<SqrtTuneFamily> {
    match env::var("SQRT_TUNE_FAMILY")
        .expect("SQRT_TUNE_FAMILY must be dyn or static")
        .as_str()
    {
        "dyn" => vec![SqrtTuneFamily::Dyn],
        "static" => vec![SqrtTuneFamily::Static],
        other => panic!("unknown SQRT_TUNE_FAMILY={other}"),
    }
}

fn make_sqrt_tune_inputs(root_len: usize) -> Vec<SqrtTuneInput> {
    let mut rng =
        StdRng::seed_from_u64(0x5351_5254_u64 ^ (root_len as u64).wrapping_mul(0x9e37_79b9));
    (0..SQRT_TUNE_CASES_PER_POINT)
        .map(|_| SqrtTuneInput {
            x: random_normalized_limbs(2 * root_len, &mut rng),
            root_len,
        })
        .collect()
}

fn zimmermann_one_descent_core(
    x: &mut [u64],
    s: &mut [u64],
    div_rem_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64]) -> u64,
    sqr_alg: &mut dyn FnMut(&[u64], &mut [u64]) -> u64,
) {
    debug_assert_eq!(x.len(), 2 * s.len());
    let s_len = s.len();
    let lo = (s_len - 1) / 2;
    let (s_lo, s_hi) = s.split_at_mut(lo);

    // This is the forced cutoff decision: descend exactly once, then solve the
    // high half directly with the binomial algorithm.
    binom_sqrt_core(&mut x[2 * lo..], s_hi);
    let (reduced, saturated) = reduce_sqrt_rem(&mut x[2 * lo..s_len + lo + 1], s_hi);

    if saturated {
        let rem = &mut x[..s_len + 1];
        rem[2 * lo..].fill(0);
        add_buf(&mut rem[lo..], s_hi);
        add_buf(&mut rem[lo..], s_hi);
        s_lo.fill(u64::MAX);
    } else {
        let d_len = buf_len(&x[..s_len + lo]);
        let overflow = div_rem_alg(&mut x[lo..d_len], s_hi, s_lo);
        debug_assert_eq!(overflow, 0);
        if shr_buf(s_lo, 1) != 0 {
            add_buf(&mut x[lo..], s_hi);
        }
        if reduced {
            s_lo[lo - 1] |= 1 << 63;
        }
    }

    let (rem, s_lo_sqr) = x[..s_len + 2 * lo + 1].split_at_mut(s_len + 1);
    let overflow = sqr_alg(s_lo, s_lo_sqr);
    debug_assert_eq!(overflow, 0);
    if correct_sqrt(rem, s, s_lo_sqr) {
        dec_buf(s);
    }
}

fn zimmermann_one_descent_dyn(x: &mut [u64], s: &mut [u64]) {
    debug_assert!(x.last().is_some_and(|&top| top >= 1 << 62));
    debug_assert_eq!(x.len(), 2 * s.len());
    let mut div_rem = |n: &mut [u64], d: &[u64], q: &mut [u64]| div_rem_dyn(n, d, q);
    let mut sqr = |value: &[u64], out: &mut [u64]| sqr_dyn(value, out);
    zimmermann_one_descent_core(x, s, &mut div_rem, &mut sqr);
    x[s.len() + 1..].fill(0);
}

fn zimmermann_one_descent_static<const N: usize>(x: &mut [u64], s: &mut [u64]) {
    debug_assert!(x.last().is_some_and(|&top| top >= 1 << 62));
    debug_assert_eq!(x.len(), 2 * s.len());
    debug_assert!(x.len() <= N && s.len() <= N);
    let mut div_rem = |n: &mut [u64], d: &[u64], q: &mut [u64]| div_rem_static::<N>(n, d, q);
    let mut sqr = |value: &[u64], out: &mut [u64]| sqr_static::<N>(value, out);
    zimmermann_one_descent_core(x, s, &mut div_rem, &mut sqr);
    x[s.len() + 1..].fill(0);
}

fn sqrt_tuning_ratio_bench(
    iters: u64,
    inputs: &[SqrtTuneInput],
    family: SqrtTuneFamily,
) -> Duration {
    alternating_ratio_bench(iters, inputs, |input, zimmermann_first| {
        let mut zimmermann_x = input.x.clone();
        let mut binom_x = input.x.clone();
        let mut zimmermann_root = vec![0u64; input.root_len];
        let mut binom_root = vec![0u64; input.root_len];

        let times = time_pair_alternating(
            zimmermann_first,
            || match family {
                SqrtTuneFamily::Dyn => zimmermann_one_descent_dyn(
                    black_box(zimmermann_x.as_mut_slice()),
                    black_box(zimmermann_root.as_mut_slice()),
                ),
                SqrtTuneFamily::Static => {
                    zimmermann_one_descent_static::<SQRT_TUNE_STATIC_CAPACITY>(
                        black_box(zimmermann_x.as_mut_slice()),
                        black_box(zimmermann_root.as_mut_slice()),
                    )
                }
            },
            || {
                binom_sqrt_core(
                    black_box(binom_x.as_mut_slice()),
                    black_box(binom_root.as_mut_slice()),
                )
            },
        );

        assert_eq!(zimmermann_root, binom_root);
        assert_eq!(zimmermann_x, binom_x);
        times
    })
}

fn bench_sqrt_tuning(c: &mut Criterion) {
    if env::var_os("SQRT_TUNE_FAMILY").is_none() {
        return;
    }
    let lengths = env_usize_values("SQRT_TUNE_VALUES", &SQRT_TUNE_DEFAULT_LENGTHS);
    let families = sqrt_tune_families();
    assert!(!lengths.is_empty(), "SQRT_TUNE_VALUES cannot be empty");
    assert!(
        lengths.iter().all(|&root_len| root_len >= 3),
        "sqrt tuning requires root lengths of at least three limbs"
    );
    if families
        .iter()
        .any(|family| matches!(family, SqrtTuneFamily::Static))
    {
        assert!(
            lengths
                .iter()
                .all(|&root_len| 2 * root_len <= SQRT_TUNE_STATIC_CAPACITY),
            "static sqrt tuning inputs exceed SQRT_TUNE_STATIC_CAPACITY"
        );
    }

    let mut group = c.benchmark_group(format!("sqrt_tuning/{ARCH}"));
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(750));
    group.measurement_time(Duration::from_secs(2));
    group.noise_threshold(0.02);

    println!("Reported value is one Zimmermann descent / direct binomial sqrt.");
    println!("Below 1.000 ms favors Zimmermann; above 1.000 ms favors binomial.");
    println!("Dynamic and static results must be interpreted independently.");
    println!("Static measurements use N={SQRT_TUNE_STATIC_CAPACITY}.");

    for family in families {
        for &root_len in &lengths {
            let inputs = make_sqrt_tune_inputs(root_len);
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}_zimmermann_over_binom", family.label()),
                    root_len,
                ),
                &root_len,
                |bench, _| {
                    bench.iter_custom(|iters| sqrt_tuning_ratio_bench(iters, &inputs, family))
                },
            );
        }
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
    targets = bench_division_tuning, bench_sqrt_tuning
}
criterion_main!(benches);
