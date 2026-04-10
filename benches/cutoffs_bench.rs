#![allow(dead_code)]

use big_bits::*;
use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
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

fn set_up_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(250); // default is 100
    group.warm_up_time(std::time::Duration::from_secs(5)); // default is 3s
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

fn avg_bench(
    iters: u64,
    inputs: &Vec<(Vec<u64>, Vec<u64>)>,
    mut func: impl FnMut(&[u64], &[u64], &mut [u64]) -> u64,
) -> Duration {
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        for (l, s) in inputs.iter() {
            let mut out = vec![0; l.len() + s.len()];
            let start = Instant::now();
            func(black_box(l), black_box(s), black_box(&mut out));
            total += start.elapsed();
        }
    }
    total / inputs.len() as u32
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

const NUM_OF_LENGTHS: usize = 256;
const GAP: u32 = 8;

fn bench_school_to_chunking_karatsuba(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("school_to_chunking_karatsuba/{ARCH}"));
    set_up_group(&mut group);
    let lengths = BoundarySearch::new(|l, s| (s <= (l + 1) / 2) && (s > 2), |l, s| is_school(l, s))
        .find((20, 3), NUM_OF_LENGTHS, GAP);
    let inputs = make_inputs(&lengths);
    assert!(!inputs.is_empty(), "find inputs failed");
    println!("Average Input: {:?}", avg_input(&lengths));

    group.bench_function("school", |b| {
        b.iter_custom(|iters| avg_bench(iters, &inputs, |a, b, out| mul_buf(a, b, out)));
    });

    group.bench_function("chunking_karatsuba", |b| {
        b.iter_custom(|iters| {
            avg_bench(iters, &inputs, |a, b, out| karatsuba_entry_dyn(a, b, out))
        });
    });

    group.finish();
}

fn bench_school_to_karatsuba(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("school_to_karatsuba/{ARCH}"));
    set_up_group(&mut group);
    let lengths = BoundarySearch::new(
        |l, s| (l >= s) && (s > (l + 1) / 2) && (s > 2),
        |l, s| is_school(l, s),
    )
    .find((20, 18), NUM_OF_LENGTHS, GAP);
    let inputs = make_inputs(&lengths);
    assert!(!inputs.is_empty(), "find inputs failed");
    println!("Average Input: {:?}", avg_input(&lengths));

    group.bench_function("school", |b| {
        b.iter_custom(|iters| avg_bench(iters, &inputs, |a, b, out| mul_buf(a, b, out)));
    });

    group.bench_function("karatsuba", |b| {
        b.iter_custom(|iters| {
            avg_bench(iters, &inputs, |a, b, out| karatsuba_entry_dyn(a, b, out))
        });
    });

    group.finish();
}

fn bench_chunking_karatsuba_to_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("chunking_karatsuba_to_fft/{ARCH}"));
    set_up_group(&mut group);
    let lengths = BoundarySearch::new(
        |l, s| (s <= (l + 1) / 2) && !is_school(l, s),
        |l, s| is_karatsuba(l, s),
    )
    .find((60, 25), NUM_OF_LENGTHS, GAP);
    let inputs = make_inputs(&lengths);
    assert!(!inputs.is_empty(), "find inputs failed");
    println!("Average Input: {:?}", avg_input(&lengths));
    println!("Number of Inputs: {}", inputs.len());

    group.bench_function("karatsuba", |b| {
        b.iter_custom(|iters| {
            avg_bench(iters, &inputs, |a, b, out| karatsuba_entry_dyn(a, b, out))
        });
    });

    group.bench_function("fft", |b| {
        b.iter_custom(|iters| avg_bench(iters, &inputs, |a, b, out| fft_entry(a, b, out)));
    });

    group.finish();
}

fn bench_karatsuba_to_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("karatsuba_to_fft/{ARCH}"));
    set_up_group(&mut group);
    let lengths = BoundarySearch::new(
        |l, s| (l >= s) && (s > (l + 1) / 2) && !is_school(l, s),
        |l, s| is_karatsuba(l, s),
    )
    .find((150, 150), NUM_OF_LENGTHS, GAP);
    let inputs = make_inputs(&lengths);
    assert!(!inputs.is_empty(), "find inputs failed");
    println!("Average Input: {:?}", avg_input(&lengths));
    println!("Number of Inputs: {}", inputs.len());

    group.bench_function("karatsuba", |b| {
        b.iter_custom(|iters| {
            avg_bench(iters, &inputs, |a, b, out| karatsuba_entry_dyn(a, b, out))
        });
    });

    group.bench_function("fft", |b| {
        b.iter_custom(|iters| avg_bench(iters, &inputs, |a, b, out| fft_entry(a, b, out)));
    });

    group.finish();
}

criterion_group!(benches, bench_chunking_karatsuba_to_fft,);
criterion_main!(benches);
