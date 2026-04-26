#![allow(dead_code)]

use big_bits::{utils::*, *};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::{Duration, Instant},
};

const ARCH: &'static str = std::env::consts::ARCH;

pub type Point = (usize, usize);

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
            exploration_budget: 200_000,
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
        let input = (
            (0..*l).map(|_| rng.gen()).collect(),
            (0..*s).map(|_| rng.gen()).collect(),
        );
        inputs.push(input);
    }
    inputs
}

fn ratio_bench(
    iters: u64,
    inputs: &[(Vec<u64>, Vec<u64>)],
    func_a: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> u64,
    func_b: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> u64,
) -> Duration {
    let mut rng = rand::thread_rng();
    let mut ratio_sum = 0.0f64;
    for _ in 0..iters {
        for (l, s) in inputs.iter() {
            let mut out_a = vec![0u64; l.len() + s.len()];
            let mut out_b = vec![0u64; l.len() + s.len()];

            let (t_a, t_b) = if rng.gen::<bool>() {
                let t_a = {
                    let start = Instant::now();
                    func_a(black_box(l), black_box(s), black_box(&mut out_a));
                    start.elapsed()
                };
                let t_b = {
                    let start = Instant::now();
                    func_b(black_box(l), black_box(s), black_box(&mut out_b));
                    start.elapsed()
                };
                (t_a, t_b)
            } else {
                let t_b = {
                    let start = Instant::now();
                    func_b(black_box(l), black_box(s), black_box(&mut out_b));
                    start.elapsed()
                };
                let t_a = {
                    let start = Instant::now();
                    func_a(black_box(l), black_box(s), black_box(&mut out_a));
                    start.elapsed()
                };
                (t_a, t_b)
            };

            ratio_sum += t_a.as_nanos() as f64 / t_b.as_nanos() as f64;
        }
    }
    let avg_ratio = ratio_sum / (inputs.len() as u64) as f64;
    Duration::from_nanos((avg_ratio * 1_000_000.0) as u64)
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

fn make_sqr_inputs(sz: usize, amt: usize) -> Vec<Vec<u64>>{
    let mut rng = rand::thread_rng();
    let min = sz.saturating_sub(amt/2).max(4);
    let max = sz + amt/2;
    let mut inputs: Vec<Vec<u64>> = Vec::with_capacity(amt);
    for i in min..max{
        inputs.push((0..i).map(|_| rng.gen()).collect());
    }
    return inputs;
}

fn sqr_ratio_bench(
    iters: u64,
    inputs: &[Vec<u64>],
    func_a: &mut dyn FnMut(&[u64], &mut [u64]) -> u64,
    func_b: &mut dyn FnMut(&[u64], &mut [u64]) -> u64,
) -> Duration {
    let mut rng = rand::thread_rng();
    let mut ratio_sum = 0.0f64;
    for _ in 0..iters {
        for buf in inputs.iter() {
            let mut out = vec![0u64; 2*buf.len()];
            let (t_a, t_b) = if rng.gen::<bool>() {
                let t_a = {
                    let start = Instant::now();
                    func_a(black_box(buf), black_box(&mut out));
                    start.elapsed()
                };
                let t_b = {
                    let start = Instant::now();
                    func_b(black_box(buf), black_box(&mut out));
                    start.elapsed()
                };
                (t_a, t_b)
            } else {
                let t_b = {
                    let start = Instant::now();
                    func_a(black_box(buf), black_box(&mut out));
                    start.elapsed()
                };
                let t_a = {
                    let start = Instant::now();
                    func_b(black_box(buf), black_box(&mut out));
                    start.elapsed()
                };
                (t_a, t_b)
            };
            ratio_sum += t_a.as_nanos().min(1) as f64 / t_b.as_nanos() as f64;
        }
    }
    let avg_ratio = ratio_sum / (inputs.len() as u64) as f64;
    Duration::from_nanos((avg_ratio * 1_000_000.0) as u64)
}

fn bench_sqr_school_to_fft(c: &mut Criterion){
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

fn cutoff_criterion() -> Criterion {
    Criterion::default()
        .sample_size(250)
        .warm_up_time(Duration::from_secs(5))
}

criterion_group! {
    name = benches;
    config = cutoff_criterion();
    targets = bench_sqr_school_to_fft
}
criterion_main!(benches);
