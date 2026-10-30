# Series evaluation dispatch tuning

The series evaluators (`series_eval_dyn`, `series_eval_static` and the node-level
`series_eval_range` under them) pick among reverse Horner, rectangular splitting (rect) and
binary splitting by predicting each eligible algorithm's time and taking the fastest. The prediction is a
cost model in series.rs: Big-O counts of each kernel's primitive operations in the call's
parameters, times seven measured constants (`SERIES_NS_*` in `src/utils/mod.rs`) shared by
every series kind and both allocation modes. This document describes the model, how its
constants are fitted, and how well it predicts and dispatches. Binary splitting's leaf and
parallel cutoffs and the multiplication and division cutoffs are inputs to the model, not
part of it, and are unchanged. Existing exp/ln and constant-cache callers still call binary
splitting directly.

It replaces a score with 18 tuned constants (a relative weight per algorithm, family and
allocation mode, a product exponent and a rect block scale) plus a 65-entry table, fitted by
a discrete search over dispatch regret. Those weights absorbed effects the score didn't
model (static zero-filling, per-family pass counts, Rayon), so they couldn't be read or
carried over, and the grouped-division and trimmed-power kernel changes left them stale:
on held-out workloads the old dispatcher's balanced regret had risen from 1.0405 to 1.137.

## Reproduction

```sh
# Timings to fit: every algorithm forced, serially; rect at many block lengths; static
# storage far past the output.
python3 scripts/python/tune_series_cutoffs.py run --tag train --reps 3 -- --threads 1 --rounds 15 --ms 10
python3 scripts/python/tune_series_cutoffs.py run --tag train-blocks --reps 3 -- --threads 1 --sizes 32,128,512 --shapes chunk,deep,boundary --blocks --rounds 15 --ms 10
python3 scripts/python/tune_series_cutoffs.py run --tag train-oversize --reps 2 -- --threads 1 --modes static --oversize --sizes 1,2,4,8,16,32,64,128,256,512,1024 --rounds 15 --ms 10
python3 scripts/python/tune_series_cutoffs.py fit 'target/series_cutoffs/train*/*.csv' --output target/series_cutoffs/fit.json
# Paste the printed constants into src/utils/mod.rs and rebuild, then validate:
python3 scripts/python/tune_series_cutoffs.py run --tag heldout --reps 3 -- --holdout --sizes 3,12,48,192,768,3072 --rounds 15 --ms 10
python3 scripts/python/tune_series_cutoffs.py report 'target/series_cutoffs/heldout/*.csv'
python3 scripts/python/tune_series_cutoffs.py run --tag blocks --reps 3 -- --sizes 32,128,512 --shapes chunk,deep,boundary --blocks --rounds 15 --ms 10
python3 scripts/python/tune_series_cutoffs.py blocks 'target/series_cutoffs/blocks/*.csv'
```

The bench prints each row's work vector (`w_*` columns) from the library's own model, so
the script never re-implements it. After a change to the model, `--work-only` prints the
work vectors without timing anything, and `fit --work` refits earlier timings to them,
rect rows at the block length they ran:

```sh
cargo bench-series --threads 1 --work-only > target/series_cutoffs/work/train.csv
python3 scripts/python/tune_series_cutoffs.py fit 'target/series_cutoffs/train/*.csv' --work 'target/series_cutoffs/work/*.csv'
```

Use a fresh tag for each run. `--check` checks all algorithms without recording timings;
`--holdout` selects unseen polynomial, alternating and atanh shapes, starting indices,
term counts and random inputs; `--oversize` runs every static case at N = 8192.
`--sizes`, `--shapes`, `--modes dyn|static|both`, `--threads 1`, and `--execution pool`
select subsets and execution environments. `--blocks` also times explicit block lengths
around eight scales of sqrt(terms) (at most 16 for static rect, as in production).

Raw samples and logs go under `target/series_cutoffs/TAG/`. `metadata.json` captures
revision, compiler, platform, environment, command and production constants. No Python
packages are required. Each process changes its deterministic random seed. Candidates warm
up and calibrate independently, run in randomized order within each round, and target
10 ms batches. Analysis takes per-process medians, then the median across processes.

## Workloads and correctness

The default precision grid is 1, 2, 4, ..., 4096 output limbs. It includes dense/sparse,
short, signed and u128 BBP series; factorial, linear and wide Engel series; and Hyper
with x = 1, one-word x, wide exp chunks, a zero shift gap, deeply shifted terms, and
wide coefficients. Static capacities are 8, 32, 128, 512, 2048 and 8192, selected to fit
the output plus guards (or 8192 throughout with `--oversize`).

Before recording timings, every candidate must agree within one output ulp with an exact
dynamic node's shifted P/Q division, including the sign except for near-zero results.
Rect block-length candidates are checked too.

## The model

Each algorithm's `Work` counts seven kinds of operation. Its predicted time is their dot
product with the seven constants.

| Unit | Counts | Constant |
|---|---|---|
| div | limbs of `div_prim`/`div_prim2` passes, Knuth quotient limbs | `SERIES_NS_DIV` |
| scalar | limbs × scalar limbs² of linear passes: products and multiply-adds by a 1- or 2-limb scalar, additions, negations, copies | `SERIES_NS_SCALAR` |
| product | schoolbook limb products of multi-limb products (`mul::mul_work`) | `SERIES_NS_PRODUCT` |
| term | Horner, rect and x-leaf terms' group, weight and exponent bookkeeping | `SERIES_NS_TERM` |
| call | kernel calls: passes, products, divisions | `SERIES_NS_CALL` |
| alloc | heap allocations and scratch acquisitions | `SERIES_NS_ALLOC` |
| fill | limbs of static `[u64; N]` arrays zero-filled or moved | `SERIES_NS_FILL` |

The parameters are all O(1) from the series at its midpoint: the terms t, the output and
skipped limbs, the frame shift k and the leaves' shift, x's limbs, the bits of p and q, the
whole range's P, Q and R widths (`est_limbs`, as `Node::limbs` estimates them), N, and the
pool's threads. The counts are structural, read off the kernels:

- **Horner.** Groups take g = min(budget, 128 − bits(p)) / bits(q) terms, at most 32, as
  `group` stops. Each group divides the frame at its shallowest term once (`frame_sum`
  sums the frame's limbs at every g-th term), and Hyper multiplies by R (one or two limbs);
  BBP clears and adds its temp. Each term adds its weight (two passes) and pays its
  bookkeeping. With x ≠ 1 each term also multiplies by r and takes a high product by x
  (copied back). A signed term whose sign differs from the sum's flips it, since each term
  outweighs the sum so far, and negates the active limbs: one more pass per term.
- **Rect.** m − 1 powers (P × x products), one high product per block by x'^m before its
  terms, at the deeper block's frame, grouped divisions that don't cross a block, and term
  adds of each power's significant limbs times its weight.
- **Binary splitting.** The leaves follow the tree's own rule (more than `LEAF_TERMS` terms
  and at least `LEAF_LIMBS` limbs split; static leaves also split until they fit N). Each
  leaf term multiplies each part, about half built, by a coefficient: the parts that a
  merge multiplies (`Family::products`: BBP's lq·rp, lp·rq and lq·rq, Engel's last two,
  Hyper's rp·lr, lr·rr, lp·rq and lq·rq). P's 2^shift costs a limb move when the shift
  reaches a limb, and a bit shift unless its low bits fit beside q in the multiplier
  (`mul_shl`). An x-leaf's running term also takes a product by x and two two-limb passes
  per term. Each merge takes those products at its children's widths (capped at N for
  static nodes), zeroes its new buffers (dynamic), and shifts and sums the merged P.
  Finalize is a quotient-limb estimate per output limb plus Knuth's products, up to three
  products for Newton (dynamic), or a reciprocal and a high product (static). On a pool of
  two or more threads, a dynamic tree's levels from `DYN_BIN_SPLIT_PAR_CUTOFF` limbs split
  in parallel, so level j's work is divided by min(threads, 2^j).
- **Allocations and static arrays.** Dynamic Horner acquires scratch once and rect three
  times; each dynamic leaf allocates its parts and each merge its products. Static Horner
  zero-fills its two (BBP), one (Engel) or three (Hyper) N-limb arrays and rect its 19;
  static leaves fill and move their parts, merges fill a P and move their parts, and
  finalize uses four.

**Products.** `mul::mul_work(l, s, static)` is the work of an l × s product in schoolbook
limb products along `mul_dyn`'s (or `mul_static`'s) own tiers. Karatsuba's cost formula is
scaled by c^(2 − log2 3) for is_school's cutoff c, so it meets l·s at the school boundary,
and the transform's cost is `is_karatsuba`'s ratio times `fft_cost`. Since a crossover is
where both sides cost the same, the curve has no jumps (`test_mul_work_continuous`) and no
constants of its own; it follows mul.rs's cutoffs when they are retuned. A two-limb short
side counts four products a limb, as `mul_prim2`'s 128-bit carries cost about 3.6
`mul_prim` passes on this machine. Measured directly, `mul_dyn` runs at 0.33–0.50 ns per
unit from 32 to 2048 limbs, through school, Karatsuba and FFT alike.

**Rect's block length** is the model's own: rect's time is about a/m + b·m (full block
products and divisions against powers and longer term adds), and `rect_best` takes
whichever of sqrt(a/b), 1 (few terms, where powers don't pay) and the length whose power
fills the accumulator the model predicts fastest. After power trimming the best length is
no longer proportional to sqrt(t), so the old 0.5·sqrt(t) scale would not carry over.
Seven candidates (adding ±√2 and ±2 around the balance) chose the same lengths on all 18
block-sweep workloads with more than twice the evaluations; dropping 1 instead cost up to
2.2× on few-term chunks.

**Fit.** The constants minimize Σ((Σ c_i w_i + c_0)/T − 1)² over every forced timing T
(each algorithm, both modes, and rect at every measured block length) subject to c ≥ 0:
relative-error least squares, which is convex, has one solution and needs no search. The
script solves it exactly by trying every support of the seven columns (Lawson–Hanson's
optimum is the unconstrained solution on its own support). c_0 is a per-call intercept
that every algorithm shares, so it cancels in dispatch and isn't shipped. The fit uses
serial timings (`--threads 1`) so the constants measure kernels, while the parallel
division above models Rayon.

## Fitted constants

Measured 2026-10-30 on the Apple M1 (aarch64 macOS), Rust 1.95.0, release profile, from
1371 timings: the 364-combination training matrix, 18 block-sweep workloads and 154 static
workloads at N = 8192, serially, three processes each (two for N = 8192). Other sessions
shared the machine, so timings carry some scheduling noise.

| Constant | ns per unit | Sanity check (this machine) |
|---|---:|---|
| `SERIES_NS_DIV` | 3.574 | `div_prim`: 3.3 ns a limb, 11 cycles of its dependency chain |
| `SERIES_NS_SCALAR` | 0.1574 | `mul_prim`: 0.17–0.35 ns a limb |
| `SERIES_NS_PRODUCT` | 0.3847 | `mul_dyn`: 0.33–0.50 ns per `mul_work` unit, 32–2048 limbs |
| `SERIES_NS_TERM` | 9.334 | ~10 ns per Engel Horner term past its divisions |
| `SERIES_NS_CALL` | 0.9307 | |
| `SERIES_NS_ALLOC` | 9.377 | |
| `SERIES_NS_FILL` | 0.09114 | memset of an N-limb array: 0.09–0.2 ns a limb |

The intercept is 18.4 ns. Each constant is close to its kernel's own microbenchmark, so
the fit finds physical costs rather than compensating weights. Over the training timings,
choosing by the model's predictions gives balanced geometric regret 1.013.

## Prediction accuracy

The factor by which predictions miss, over the training timings (typical: geometric mean
of max(pred/T, T/pred); bias: geometric mean of pred/T):

| Mode | Family | Algorithm | n | Typical | p95 | Bias |
|---|---|---|---:|---:|---:|---:|
| dyn | bbp | BinSplit | 65 | 1.30 | 1.69 | 0.89 |
| dyn | bbp | Horner | 65 | 1.17 | 1.52 | 0.90 |
| dyn | engel | BinSplit | 39 | 1.36 | 1.95 | 0.87 |
| dyn | engel | Horner | 39 | 1.14 | 1.44 | 1.02 |
| dyn | hyper | BinSplit | 78 | 1.21 | 1.53 | 0.97 |
| dyn | hyper | Horner | 78 | 1.21 | 1.78 | 0.90 |
| dyn | hyper | Rect | 181 | 1.25 | 1.93 | 0.90 |
| static | bbp | BinSplit | 120 | 1.36 | 2.08 | 0.82 |
| static | bbp | Horner | 120 | 1.15 | 1.50 | 0.91 |
| static | engel | BinSplit | 72 | 1.24 | 1.95 | 0.88 |
| static | engel | Horner | 72 | 1.14 | 1.37 | 1.05 |
| static | hyper | BinSplit | 144 | 1.42 | 2.15 | 0.74 |
| static | hyper | Horner | 144 | 1.22 | 1.71 | 0.93 |
| static | hyper | Rect | 154 | 1.21 | 1.84 | 1.01 |

All cells together are typically 1.25× off, and within 1.84× at the 95th percentile.

## Validation

The held-out matrix has 168 workload/allocation combinations at 3, 12, 48, 192, 768 and
3072 output limbs, with a different starting index, term count, input seeds, quadratic
instead of linear coefficients, signed Hyper instead of exp chunks, and atanh instead of
the boundary shape. Nothing is fitted on it. Both dispatchers ran on the same kernels
(grouped divisions and trimmed powers), three processes of 15 rounds at default threads,
one serially. Regret is relative to each workload's fastest forced algorithm, balanced
across families and modes. *Selection* times the chosen algorithm forced; *actual*
times `series_eval_range` itself, so it includes evaluating the dispatcher.

| Held-out | Old score | Cost model |
|---|---:|---:|
| Selection, default threads | 1.101 | **1.023** |
| Selection, serial | 1.089 | **1.021** |
| Actual dispatch, default threads | 1.137 | **1.095** |
| Actual dispatch, serial | 1.125 | **1.093** |

| Selection by family (default threads) | Old score | Cost model |
|---|---:|---:|
| Dynamic BBP | 1.151 | 1.031 |
| Dynamic Engel | 1.000 | 1.013 |
| Dynamic Hyper | 1.147 | 1.050 |
| Static BBP | 1.189 | 1.016 |
| Static Engel | 1.000 | 1.010 |
| Static Hyper | 1.131 | 1.021 |

The old score's misses were up to 6.7× (binary splitting for sparse BBP and word Hyper
that Horner sums far faster); the model's worst selections are 1.6–1.7× (dynamic atanh at
192 limbs takes binary splitting over rect, word Hyper at 192 limbs binary splitting over
Horner).

**Evaluation cost** is what separates selection from actual regret. The model costs a
median 33 ns over the forced pick at 3 limbs, 34 at 12 and 45 at 48 (the old score 19, 18
and 24). For Hyper with a multi-limb x at 3 limbs, whose calls take 100–200 ns, that is
1.5× (old 1.14×): the rect estimate needs a block length. From 48 limbs up it is at most
5%. So the 5% aggregate target is met for selection but not for actual dispatch, which
the 3- and 12-limb calls hold at 1.095. `series_eval_range` evaluates the model once (and runs
rect at the length it found), its frame sums and tree levels use exact integer and shift
arithmetic, and libm runs only for Karatsuba-sized products;
with that and three block lengths rather than seven, a 3-limb Hyper dispatch went from
about 158 to 120 ns.

**Static at its capacity** (70 combinations, N = output + 4, one process): selection
1.016, actual 1.071.

**Rect's block length** (18 workloads, every explicit length around 0.25–4 × sqrt(t),
three processes): the model's length is 1.095 (dynamic) and 1.061 (static) above the best
measured one, against 1.123 and 1.097 for the old 0.5·sqrt(t) on the same kernels. The
model tends to longer blocks than the best, and the measured time steps with the FFT's
transform sizes (at 512 limbs, m = 47–49 costs 1.7× m = 128), which no smooth model
follows.

## Limitations

- **Multiplication crossovers.** `mul_work` trusts mul.rs's crossovers, which were tuned on
  x86 (and the static NTT's with its convolutions on Rayon). On this M1 the FFT just past
  its crossover is slower than Karatsuba, and the serial static NTT about 1.4× slower than
  assumed, so large dynamic x products and large static trees run up to ~2× over their
  predictions. Retuning the multiplier fixes both without touching the series model.
- **Oversized static storage.** With N far past the output, static binary splitting runs
  1.3–2.5× over its prediction, more for Hyper with x: its nodes move whole N-limb arrays
  more often than the counts above allow. Dispatch still avoids it where it matters, since
  Horner and rect touch only two or three such arrays.
- **Tiny calls.** Evaluating the model costs 30–45 ns (about 100 ns for Hyper with a
  multi-limb x, which needs a rect block length), which is 15–50% of a 3-limb call and
  keeps actual held-out regret at 1.095 although selection is at 1.023. Below a few hundred
  ns the shared intercept and scheduling noise are also as large as the differences
  between algorithms; small sparse BBP at 2–16 limbs is the largest remaining
  misprediction (up to 1.75× at 2–4 static limbs).
- **Dynamic NTT** past `FFT_16BIT_CUTOFF` output limbs is costed as the FFT.

## Verification

- `cargo test --lib`: 467 passed, including `test_mul_work_continuous`,
  `test_series_costs_edges` (finite, non-negative work and eligible picks over edge shapes)
  and `test_horner_cost_grows_with_terms`.
- `cargo bench-series --check` (and with `--blocks`) agrees with the exact sum within one
  ulp for every algorithm, block length and shape.
- Changed Rust files pass `rustfmt --check --edition 2021`; `git diff --check` passes.
