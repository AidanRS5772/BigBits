# Binary-splitting cutoff tuning

Machine: Apple M1 (4 performance + 4 efficiency cores), macOS, `bench` profile.
Harness: `benches/bin_split_cutoffs.rs` (`cargo bench-bin-split <family>`). Each
candidate builds the tree with `bin_split_tree_at` (finalize is excluded because it
doesn't depend on the tree's shape). Each sample repeats a build to about 20 ms.
Candidates alternate within each round, in forward and reversed rotations, and each
candidate reports its median over 7–11 rounds, relative to the workload's best. The
summary is the geometric mean over workloads. Leaf sweeps run serially (`par_limbs =
max`). The benchmark thread asks for the user-interactive QoS class: without that,
macOS sometimes ran it on an efficiency core and Engel rows swung by ±30%.

## The rule

`bin_split` used to take a term-count cutoff from each caller: 640 for ln(2), 32 for
`exp_dyn`. Each node type now owns its cutoffs, and a tree splits `[a, b)` while

    b - a > LEAF_TERMS  &&  Node::limbs(ctx, a, b) >= LEAF_LIMBS

where `Node::limbs` is an O(1) estimate of the node's largest part (P):
`t · (bitlen(q(mid)) + shift)` bits, or for Hyper the larger of that and
`t · (bitlen(r(mid)) + bitlen(x))`. Static nodes cap it at N. The parallel split uses
the same estimate (`DYN_BIN_SPLIT_PAR_CUTOFF`).

| Constant | Old | New | Plateau (≤ 2% of best geomean) |
|---|---|---:|---|
| `DYN_BBP_LEAF_LIMBS` | 640 terms (ln2) | 64 | 48–64; ln2 alone is flat over 48–256 |
| `STATIC_BBP_LEAF_LIMBS` | 640 terms (ln2) | 512 | 192–512 |
| `DYN_ENGEL_LEAF_LIMBS` | 32 terms (exp) | 96 | 48–128 |
| `STATIC_ENGEL_LEAF_LIMBS` | — | 128 | 128–256 |
| `DYN_HYPER_LEAF_LIMBS` / `_TERMS` | 32 terms (exp) | 48 / 8 | 24–48 limbs × 8–12 terms |
| `STATIC_HYPER_LEAF_LIMBS` / `_TERMS` | — | 64 / 8 | 64 / 8 only (neighbors +3%) |
| `DYN_BIN_SPLIT_PAR_CUTOFF` | 2048 terms | 256 limbs | 128–1024 |

## Why limbs, and why Hyper also keeps a term floor

A leaf builds its node one step at a time (a step is a packed group of terms), and each
step costs O(current length). Splitting halves that work but pays for the merge
products. Both sides depend on the node's size in limbs, not on its term count, and
bits per term vary by orders of magnitude between series: about 20 for ln(2), about
`shift` for exp's later chunks. A term cutoff is only right for the series it was
tuned on:

| Rule | BBP | Engel | Hyper | Worst row |
|---|---:|---:|---:|---|
| 640 terms (old ln2 cutoff) | 1.403 | 1.480 | 1.597 | 5.00× (Engel k=512, 4096) |
| 32 terms (old exp cutoff) | 1.040 | 1.107 | 1.040 | 1.37× (Engel k=0, 256) |
| New per-type cutoffs | **1.007** | **1.024** | **1.021** | 1.20× (Engel k=64, 256: 16 µs) |

Geometric-mean slowdown over every dynamic workload below. This run shared the machine
with other processes, so absolute times are about 1.7× slower, but candidates alternate,
so the ratios hold.

Limbs alone don't capture Hyper with a wide x. Each step there multiplies by x, so a
leaf of t steps costs about t²·M(s)/2 for an s-limb x. A merge costs M(t·s/2), which
grows only about as fast as t·M(s) once merges reach FFT sizes. The crossover then sits
at a few terms whatever the width. Limbs alone pick 512 limbs for 512-bit x, 1024 for
2048-bit and ≥ 2048 for 8192-bit, which is always about 8–32 terms. The term floor
captures this: in the same run, Hyper's best limbs-only cutoff scored 1.096 and its best
with a floor scored 1.047.

Two predictions did not hold:
- Sparse BBP and Engel (shift ≥ 64) prefer *smaller* leaves in limbs than dense ones.
  Their Q is small compared with P, so merges are cheap unbalanced products, while
  each leaf step still moves all of P. A term floor can only make leaves bigger, so it
  doesn't help them, and BBP and Engel keep `LEAF_TERMS = 1`.
- Static nodes zero and move whole `[u64; N]` arrays, which suggested leaves should grow
  with N. They don't: static ln2 is fastest at ≥ N limbs for small N, and at 256–512
  for N ≥ 1024. A plain constant fits every N.

## Workloads

- BBP: the four ln(2) series (shift 4) at 64–16384 output limbs, and
  `1 / ((4n + 3) 2^(kn))` at k = 64 and 256.
- Engel: `1 / n!` at shift k = 0 (e − 2) and k = 1, 8, 64, 512 (exp's first chunk at
  shift k), with terms from `exp_dyn`'s own term count.
- Hyper: x = 1 (`(2n+1)/(2n+4)` at shift 8), and `exp_dyn`'s chunk series `x^n / n!`
  with an x of 8–8192 bits at shift `2·bits(x)`, which is the shape of `exp_dyn`'s
  chunks.
- Static: ln(2) at N = 64–4096 with `len = N − 4`. Engel and Hyper on the same series
  at N = 64, 256 and 1024.

## Sweeps

Dynamic BBP (11 rounds):

| Workload | 32 | 48 | 64 | 96 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|
| ln2 64 | 1.071 | 1.067 | 1.016 | 1.024 | 1.009 | 1.000 | 1.043 |
| ln2 1024 | 1.041 | 1.013 | 1.028 | 1.000 | 1.018 | 1.015 | 1.025 |
| ln2 16384 | 1.023 | 1.015 | 1.007 | 1.015 | 1.000 | 1.015 | 1.026 |
| bbp k=64 1024 | 1.003 | 1.000 | 1.004 | 1.052 | 1.053 | 1.173 | 1.155 |
| bbp k=256 1024 | 1.074 | 1.008 | 1.008 | 1.000 | 1.002 | 1.072 | 1.073 |
| geomean (11 workloads) | 1.046 | 1.013 | **1.009** | 1.022 | 1.022 | 1.103 | 1.106 |

Dynamic Engel (11 rounds):

| Workload | 32 | 48 | 64 | 96 | 128 | 192 | 256 | 384 |
|---|---|---|---|---|---|---|---|---|
| engel k=0 1024 | 1.069 | 1.051 | 1.032 | 1.008 | 1.007 | 1.000 | 1.002 | 1.037 |
| engel k=1 4096 | 1.039 | 1.022 | 1.016 | 1.000 | 1.020 | 1.042 | 1.042 | 1.064 |
| engel k=64 1024 | 1.000 | 1.003 | 1.027 | 1.093 | 1.121 | 1.345 | 1.447 | 1.795 |
| engel k=512 4096 | 1.083 | 1.031 | 1.026 | 1.000 | 1.010 | 1.047 | 1.099 | 1.165 |
| geomean (20 workloads) | 1.085 | 1.044 | 1.041 | **1.023** | 1.041 | 1.102 | 1.121 | 1.259 |

Dynamic Hyper. First, limbs alone over a coarse grid (7 rounds, before the QoS fix):

| Workload | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|
| hyper x=1 1024 | 1.036 | 1.000 | 1.022 | 1.013 | 1.029 | 1.152 | 1.502 | 2.369 |
| exp xb=8 1024 | 1.000 | 1.030 | 1.062 | 1.152 | 1.402 | 1.997 | 3.265 | 6.331 |
| exp xb=512 1024 | 1.215 | 1.221 | 1.224 | 1.144 | 1.080 | 1.000 | 1.057 | 2.066 |
| exp xb=2048 1024 | 1.122 | 1.153 | 1.160 | 1.137 | 1.128 | 1.084 | 1.000 | 1.179 |
| geomean (20 workloads) | 1.096 | 1.084 | 1.084 | 1.112 | 1.166 | 1.295 | 1.497 | 1.901 |

Then limbs × term floor, geometric mean over the same 20 workloads. Each run is
relative to its own per-workload best, so compare cells within a table, not across
them. Coarse grid (7 rounds):

| limbs \ terms | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| 16 | 1.111 | 1.048 | 1.059 | 1.106 | 1.154 |
| 32 | 1.096 | **1.047** | **1.047** | 1.111 | 1.149 |
| 64 | 1.097 | 1.050 | 1.061 | 1.128 | 1.163 |
| 128 | 1.121 | 1.086 | 1.096 | 1.162 | 1.214 |

Fine grid (9 rounds):

| limbs \ terms | 4 | 6 | 8 | 12 |
|---|---|---|---|---|
| 16 | 1.044 | 1.047 | 1.030 | 1.033 |
| 24 | 1.050 | 1.049 | 1.027 | 1.028 |
| 32 | 1.041 | 1.037 | 1.026 | 1.024 |
| 48 | 1.038 | 1.031 | **1.020** | 1.017 |

Floors of 16 or more lose up to 2.5× on 8192-bit x (exp xb=8192 4096: 1.31 at 16, 2.48 at
32), and wide x leans toward the smaller floors, so the floor is 8 rather than 12.

Static BBP, ln(2) (11 rounds):

| N | 64 | 96 | 128 | 192 | 256 | 384 | 512 | max |
|---|---|---|---|---|---|---|---|---|
| 64 | 1.000 | 1.004 | 1.010 | 1.009 | 1.012 | 1.011 | 1.008 | 1.014 |
| 256 | 1.082 | 1.044 | 1.025 | 1.014 | 1.000 | 1.009 | 1.007 | 1.001 |
| 1024 | 1.041 | 1.026 | 1.058 | 1.037 | 1.037 | 1.035 | 1.000 | 1.072 |
| 2048 | 1.098 | 1.054 | 1.018 | 1.002 | 1.055 | 1.000 | 1.009 | 1.232 |
| 4096 | 1.034 | 1.109 | 1.111 | 1.036 | 1.000 | 1.051 | 1.007 | 1.477 |
| geomean | 1.051 | 1.047 | 1.044 | 1.019 | 1.021 | 1.021 | **1.006** | 1.146 |

Static Engel (9 rounds): geomean 2.163 / 1.577 / 1.269 / 1.200 / **1.162** / 1.178 /
1.326 / 1.551 at 8 / 16 / 32 / 64 / 128 / 256 / 512 / max. The worst case at 128 is
`engel k=64 N=64` (2.57×, a 2 µs build). Its whole range estimates below 64 limbs, so
every cutoff ≥ 64 builds one leaf.

Static Hyper (9 rounds), geomean by limbs, at term floor 1 / 8:

| | 8 | 16 | 32 | 64 | 128 | 256 | 512 | max |
|---|---|---|---|---|---|---|---|---|
| floor 1 | 1.762 | 1.454 | 1.300 | 1.166 | 1.130 | 1.153 | 1.249 | 1.497 |
| floor 8 | 1.375 | 1.208 | 1.091 | **1.058** | 1.090 | 1.143 | 1.241 | 1.491 |

Parallel cutoff (all 8 threads, default leaves, 9 rounds), geomean over ln2, Engel k=1
and exp xb=32/512 at 1024, 4096 and 16384 limbs:

| par limbs | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | max |
|---|---|---|---|---|---|---|---|---|
| geomean | 1.064 | 1.038 | **1.037** | 1.052 | 1.051 | 1.151 | 1.204 | 2.067 |

## End to end

The old term cutoffs against the new limb cutoffs on the production series (serial;
the same contended run as the universality table):

| Workload | old ms | new ms | new / old |
|---|---:|---:|---:|
| ln2 dyn 1024 | 27.37 | 27.43 | 1.002 |
| ln2 dyn 16384 | 1288.7 | 1288.3 | 1.000 |
| exp_dyn's series, 256 | 1.758 | 1.787 | 1.016 |
| exp_dyn's series, 1024 | 14.52 | 14.29 | 0.985 |
| exp_dyn's series, 4096 | 97.69 | 95.37 | 0.976 |
| ln2 static N=1028 | 52.56 | 52.65 | 1.002 |
| ln2 static N=4096 | 409.0 | 416.1 | 1.017 |

The production callers had already been tuned by hand, so they stay within noise. The
gain is that new series and shapes need no tuning: the ln(2) cutoff lost up to 5× on
other series, and the exp cutoff lost up to 1.21× on exp's own wide chunks
(exp xb=2048 1024) and 1.37× on dense Engel.

## exp_dyn's schedule: first-chunk width and chunk parallelism

`exp_dyn` sums one Engel series for x's top bit, then a series per chunk of x's
remaining bits (Engel when a chunk is a single bit, Hyper otherwise), and multiplies
the results. Each chunk spans `[s, ~2s)` of its own offset, so the widths double, but
the first chunk is at least `DYN_EXP_CHUNK_MIN_BITS` (16) wide.

Every chunk's bits are extracted up front into one pre-sized scratch buffer. The first
series then runs on its own. The chunks then fill one slot each (`Σ bl` limbs, carved
from one scratch acquisition), either serially or all at once on rayon. Either way they
combine in chunk order, so the output is bit-identical. The parallel choice is made
once per call:

    chunks >= 2  &&  chunks * v >= DYN_EXP_CHUNK_PAR_WORK (512)  &&  threads >= 2

Here `v` is the result's working limbs (`out.len()` + guard).

Both values are now plain constants in `src/utils/mod.rs`. The tuning harness, a
`bench-bin-split exp` family that overrode them per call, was removed once the fit
below was done. It timed whole exp calls at precision `l` = 4–4096 with
`m ∈ {0, 1, 2, 4, l/8, l/4, l/2, l}` leading zero limbs, plus `l` = 16384 at m = 0
(86 workloads, 9 rounds). A larger `m`, as argument reduction produces, means fewer
chunks: 7–17 at m = 0, 1 at m = l.

### First-chunk width

Near x's top a chunk's term count is set by the factorial, not by its offset, so the
plain doubling's 2-, 4- and 8-bit chunks each cost about as much as a wide one. At
16384 limbs, the chunks inside x's top limb were ~42% of serial time. Merging them
saves trees, finalizations and combines, but widens the merged chunk's P.

The width only matters at m = 0. Once m ≥ 1, the doubling's first chunk is already
about 64m bits wide. Geometric means over the m = 0 rows (0 is the plain doubling; since
the schedule became `[s, ~2s)` per chunk, widths 0–2 add one 2-bit chunk when x ≥ 1/2,
which these random x never are):

| min bits | 0 | 8 | **16** | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 thread | 1.144 | 1.087 | **1.021** | 1.025 | 1.033 | 1.148 | 1.294 | 1.713 | 2.171 |
| 8 threads (gate 512) | 1.138 | 1.085 | **1.028** | 1.045 | 1.045 | 1.176 | 1.335 | 1.734 | 2.249 |

There is no trend with precision. The per-row best lands on 16–32 at every l from 4 to
16384, so a constant fits:

| l (m = 0, 1 thread) | 0 | 8 | 16 | 24 | 32 | 48 | 64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.184 | 1.163 | 1.047 | 1.020 | 1.000 | 1.096 | 1.134 |
| 16 | 1.271 | 1.149 | 1.000 | 1.074 | 1.112 | 1.187 | 1.402 |
| 64 | 1.069 | 1.075 | 1.050 | 1.000 | 1.087 | 1.262 | 1.494 |
| 256 | 1.193 | 1.043 | 1.000 | 1.017 | 1.037 | 1.151 | 1.248 |
| 1024 | 1.189 | 1.087 | 1.017 | 1.077 | 1.000 | 1.143 | 1.139 |
| 4096 | 1.067 | 1.074 | 1.013 | 1.035 | 1.000 | 1.047 | 1.060 |
| 16384 | 1.100 | 1.063 | 1.000 | 1.031 | 1.010 | 1.098 | 1.088 |

This run shared the machine with a browser using about one core, and the load average
reached 6.6. The m ≥ 1 rows run identical schedules for every width ≤ 64, so they
measure the noise floor: their geometric means agree within 0.2% across candidates
(1.014–1.016). That is far below the gaps above.

### Parallel gate

Re-fit at a min width of 16 (candidates `--par 0,max`, i.e. always against never). Each
rule chooses per workload; scores are the geometric mean of the chosen path's time over
the best path's, at the rule's best threshold:

| Rule | 8 threads | worst row | 4 threads | worst row |
|---|---:|---:|---:|---:|
| `chunks >= 2 && v >= 96` | 1.004 | 1.204 | 1.005 | 1.215 |
| `chunks >= 3 && chunks * v >= 512` | 1.005 | 1.182 | 1.008 | 1.268 |
| **`chunks >= 2 && chunks * v >= 512`** | **1.001** | **1.050** | **1.003** | 1.268 |

448 and 512 tie, and 384–768 stay within 0.7%. An earlier fit, before the min width
(whose first series overlapped the chunks), also chose 512.

Why not precision alone:
- **Per-chunk cost depends on m, not just on precision.** In the earlier sweep, at
  w = 129, `l=128 m=0` took 0.299 ms but `l=64 m=64` 0.017 ms. With m > 0 each chunk sums
  only a few terms.
- **The break-even precision falls as chunks grow:** l ≈ 48–64 at 9 chunks, ≈ 128
  at 2–4. That is a roughly constant total `chunks * v`.
- **A lone chunk has nothing to run beside,** so `chunks >= 2`. Forcing one chunk
  parallel costs about nothing now (0.96–1.02×).

### End to end

`exp_dyn` with random x and m = 0. Mean ms per call, best of 3 runs alternating against
the parent commit (`1ffd2ee`), with a browser using about one core:

| limbs | 8 threads before | after | speedup | 1 thread before | after | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.018 | 0.017 | 1.06× | 0.018 | 0.017 | 1.06× |
| 64 | 0.153 | 0.115 | 1.33× | 0.153 | 0.129 | 1.19× |
| 256 | 1.574 | 0.651 | 2.42× | 1.345 | 1.142 | 1.18× |
| 1024 | 8.139 | 3.624 | 2.25× | 9.370 | 8.554 | 1.10× |
| 4096 | 44.21 | 22.44 | 1.97× | 64.12 | 60.41 | 1.06× |
| 16384 | 264.1 | 140.8 | 1.88× | 427.8 | 381.9 | 1.12× |

On one thread the gain is the min width alone. The combine is still serial, at about
30% of the remaining wall time in parallel mode.

### Argument scale: squares

`exp_reduced` runs the core on r·2^-j instead of r, then squares back j times
(u ← 2u + u², with u = e^x − 1). Near r's top the factorial sets each series' term
count, so moving r down by j bits trims the most expensive series, and each square
costs one `hi_sqr`. j is `DYN_EXP_SQR_BITS` (24) less r's own leading zero bits, and
it is 0 whenever the core would sum r's chunks in parallel.

The first rule tried was ln's: j = floor(A·log2(64·o)), the twin of `DYN_LN_SQRT_A`.
`cargo bench-bin-split exp` (two random X per size, from 1 to 16384 limbs, 9 rounds)
gives these geometric means relative to each row's best A:

| A | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 thread | 1.219 | 1.186 | 1.099 | 1.070 | 1.068 | 1.073 | 1.073 | 1.121 | 1.262 |
| 12 threads | 1.167 | 1.194 | 1.183 | 1.207 | 1.253 | 1.301 | 1.317 | 1.435 | 1.699 |

Neither fits. On one thread the best A falls from 4 at 1–2 limbs to 1–1.5 past 128,
because the best j hardly moves at all: about 24 bits at every size. On 12 threads the
best A is 0 from 256 limbs up. A fixed j per size, on one thread, relative to each
size's best:

| limbs | j=0 | 12 | 16 | 20 | **24** | 28 | 32 | 40 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.200 | 1.220 | 1.176 | 1.055 | **1.002** | 1.006 | 1.042 | 1.070 |
| 4 | 1.364 | 1.174 | 1.120 | 1.115 | **1.000** | 1.018 | 1.021 | 1.060 |
| 16 | 1.471 | 1.132 | 1.089 | 1.041 | **1.001** | 1.013 | 1.001 | 1.017 |
| 64 | 1.188 | 1.025 | 1.057 | 1.012 | **1.000** | 1.036 | 1.062 | 1.088 |
| 128 | 1.220 | 1.001 | 1.022 | 1.027 | **1.029** | | 1.157 | |
| 256 | 1.096 | 1.000 | 1.006 | 1.019 | **1.046** | | 1.108 | |
| 1024 | 1.032 | 1.009 | 1.006 | 1.010 | **1.010** | | 1.031 | |
| 4096 | 1.069 | 1.024 | 1.014 | 1.006 | **1.001** | | 1.005 | |

(128 to 4096 also ran j = 8: 1.014, 1.010, 1.011 and 1.045.) 24 is within 1.05× of the
best everywhere. At 128 limbs, ten more random X gave j = 0 1.159, j = 12 1.003 and
j = 24 1.047. Below 24 the small sizes lose 5–12%, so 24 is the smallest constant in the
noise band.

On 12 threads `exp_dyn_entry` sums its chunks on rayon from about 64 limbs on. The
squares stay serial, and the series they shorten were already overlapping, so they only
add time:

| limbs | j=0 | 4 | 8 | 12 | 16 | 24 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.111 | 1.066 | 1.022 | 1.009 | 1.004 | 1.102 |
| 128 | 1.015 | 1.014 | 1.038 | 1.120 | 1.218 | |
| 256 | 1.000 | 1.025 | 1.044 | 1.093 | | |

So `exp_reduced` asks the core's own parallel gate first (`exp_par`, on r's frame at j =
0) and takes no squares when it would go parallel. That gives up about 10% at 64 limbs
and keeps every larger size at its j = 0 time.

End to end, ms per call against j = 0 and against `ln_dyn` on the same X (means of the
two rows per size, 9 rounds):

| limbs | 1 thread j=0 | j=24 | gain | ln | 12 threads j=0 | j=24 | gain | ln |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0050 | 0.0040 | 1.24× | 0.0085 | 0.0050 | 0.0040 | 1.24× | 0.0085 |
| 4 | 0.0109 | 0.0080 | 1.36× | 0.0145 | 0.0111 | 0.0080 | 1.39× | 0.0145 |
| 16 | 0.0401 | 0.0295 | 1.36× | 0.0520 | 0.0388 | 0.0285 | 1.36× | 0.0530 |
| 32 | 0.103 | 0.076 | 1.35× | 0.130 | 0.105 | 0.078 | 1.36× | 0.134 |
| 64 | 0.269 | 0.236 | 1.14× | 0.339 | 0.249 | 0.250 | 1.00× | 0.270 |
| 128 | 1.043 | 1.053 | 0.99× | 1.276 | 0.611 | 0.608 | 1.01× | 0.552 |
| 256 | 3.041 | 2.930 | 1.04× | 3.740 | 1.402 | 1.384 | 1.01× | 1.272 |
| 512 | 7.144 | 6.989 | 1.02× | 8.622 | 2.914 | 2.829 | 1.03× | 2.657 |
| 1024 | 17.16 | 16.30 | 1.05× | 21.34 | 6.243 | 6.159 | 1.01× | 6.901 |
| 4096 | 95.15 | 91.71 | 1.04× | 126.8 | 32.11 | 31.06 | 1.03× | 36.95 |
| 16384 | 582.3 | 539.2 | 1.08× | 694.9 | 184.5 | 183.0 | 1.01× | 224.3 |

On 12 threads both columns from 64 limbs on run the same code, so their gap there is
noise. The one-thread 128-limb row drew two unlucky X; the ten-X repeat above has j = 24
1.11× ahead of j = 0. With the squares, exp is faster than ln at every size on one thread
(ln takes 1.2–2.1× as long). On 12 threads ln takes 1.1–2.1× as long, except at 128–512
limbs, where it takes about 0.92× as long: there exp's chunks run in parallel without
squares.

## Static nodes: bit-level normalization

Static parts used to keep limb exponents, so a truncated part kept its top N limbs
whatever its top limb held. A merge also reserved a spare carry limb, a whole limb when
the shift had bits, and `hi_mul_drop` zeroed the window's lowest limb about half the
time. Each part is now `m · 2^e` with e in bits: exact while it fits N limbs, then its
top 64N bits (P within two, leaf parts one spare bit once they drop). BBP keeps separate
P and Q exponents, so Q no longer carries 2^(shift·t).

Bits a node's P / Q loses below 64N (worst case), and static finalize's error at the
N − 2 bound against dyn, over BBP, Engel and Hyper series (signed and unsigned, deep,
default and single-leaf trees) at N = 4–256:

| Series | Bits lost before | after | N − 2 error before | after |
|---|---:|---:|---:|---:|
| BBP | 68–109 | 2–5 | 0 | 0 |
| Engel | 186–191 | 0–3 | up to 2^63 ulps | 0 |
| Hyper | 184–191 | 7–11 | up to 2^56 ulps | 0 |

`test_static_nodes_keep_bits` holds the loss to 16 bits, and the any-cutoff and leaf-size
sweeps now check within 1 ulp at N − 2 instead of N − 3. Finalize's own `len + 2 <= N`
bound comes from its reciprocal, so the public bounds don't change.

Cost, as new / old time, the min of 8 runs alternating the two builds (`bench` profile,
a browser using one to two cores):

| Workload | new / old |
|---|---|
| Static BBP trees, ln2 at N = 64–4096 | 0.89–0.92 |
| `ln2_static` past its table, N = 1028–4096 | 0.91–0.96 |
| Static Engel trees, N = 64–1024 | 0.99–1.02 |
| Static Hyper trees, N = 64–1024 | 0.67–1.10, geomean 0.98 |
| `exp_static_entry`, N = 8 / 16 / 32 / 64–1024 | 1.11 / 1.05 / 0.95 / 1.00–1.03 |
| `exp_static`, N = 16 / 64–1024 | 1.06 / 1.00–1.02 |
| Single leaves that drop bits nearly every term, N = 8–64 | 1.02–1.40 |
| Dynamic trees and `exp_dyn_entry` (shared leaf kernels) | 0.99–1.02 |

- BBP is faster because Q no longer holds 2^(shift·t), so its products are smaller.
- The leftover cost is in parts that drop bits: a drop is now a bit shift where it used
  to be a whole-limb memmove. That shows only at small N, where a whole exp series is
  one leaf, and in the drop-heavy test shapes.
- Hyper at N = 256, xb = 512 (1.10): the 2 extra limbs P keeps make one product's
  window 129 limbs instead of 127, which pushes `hi_mul_static`'s guarded product past
  N limbs onto its split path.

What kept the cost down:
- Each merge takes P's exponent from its operands' bit lengths first, and computes each
  product only down to it. Full 64N-bit windows for every product made merges up to 2×
  slower.
- `mul_shl` reads P's top limb only when a shift won't fit the free limbs. Waiting on it
  serialized the shift behind the multiply's carry chain, costing 30% on Engel at
  shift 64.
- A leaf adds a term at a bit offset by splitting the multiplier w·2^s across two limbs
  instead of shifting a copy of the term, inline when the offset is a whole number of
  limbs.
- The last x-leaf of a sum keeps its running term t only from P's exponent up, since t
  never becomes R. That took `exp_static_entry` at N = 16 from +12% to +4%.

## Constant caches: when to build the series in parallel

`ln2_dyn` and `pi_dyn` copy requests of up to their tables' length (1024 limbs) from
`ln2.bin` and `pi.bin`, as the static caches do, and build only past them. So the rule
below only runs past 1024 limbs, where it makes every ln2 build parallel, and π's from
about 7 new limbs on a 1024-limb cache. The tables below were measured before the dynamic
caches used the tables, building from small sizes, and still describe a shrunken table.
The first request past a table builds from scratch: 1024 → 1025 took 6.2 ms for ln2 and
1.56 ms for π, where extending a 1024-limb cache took 0.61 and 0.70 ms. In exchange, a
request of up to 1024 limbs on a cold cache costs about 2 µs instead of up to 6.4 ms
(ln2) and 1.5 ms (π), and `exp_dyn` at 64 and 512 limbs, whose ln2 now comes from the
table, took 0.56x and 0.48x as long. Cold builds, hits and extensions past the table were
unchanged (within an old-vs-old repeat's ±3%).

ln2 now sums four base-64 series (`Shift(6)`) from n = 0, ln 2 = 2/3·S(6,1) + 1/24·S(6,5)
+ 1/72·S(2,1) + 1/384·S(1,1) with S(A,B) = Σ 1/((An + B) 64^n), in place of the base-16
series from n = 1 with 2/3 as a constant. Each term adds 6 bits instead of 4, so a tree
has 2/3 the terms, and its Q about 2/3 the bits. Each weight is 2^-k/d with d = 3 or 9,
so a share scales a copy of the top limbs of its Q (all that finalize reads) by d and
moves 2^-k into the finalize shift. Against the base-16 series (11–41 rounds, median, an
old-vs-old repeat within ±3% except a few warm rows at ±5–11%):

| | 1025 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| `ln2_dyn` cold | 0.68 | 0.65–0.68 | 0.65 | 0.67 | 0.66 |
| `ln2_static`, N = len + 3 | 0.75 (1100) | 0.67 | 0.65 | | |

Warm extensions: 1100 → 1200 0.71–0.72, 2048 → 4096 0.66–0.68, and one limb (1100,
2048, 4096) 0.95–0.97, since the full-length finalize dominates those. Growing 1100 →
1150 a limb at a time: 0.94. `exp_static` at 1100 limbs, which sums ln2 past the table:
0.87, and `exp_dyn` at 2048: 0.84. With the table cut to one limb, so that the series run
below 1024 limbs: cold `ln2_dyn` 0.84 at 16 limbs, 0.75 at 32, 0.52–0.67 from 64 to 1024;
`ln2_static` 0.69–0.89 from 32 to 512; `exp_dyn` 0.77 at 64 and 0.83 at 512. From 2 to
8 limbs a fresh process paid about 1.5 µs once for touching `mul_prim` and a second
scratch buffer (1.08–1.25x); with those touched first it was 0.97–0.99. `ln2_build`
counts a cache's limbs as 3·terms/32 rounded up: rounded down, a one-limb extension (10
or 11 terms) counted as 0 new limbs and ran its four shares serially, at 2.6x. The ln2
parallel cutoffs above were measured on the base-16 series. Past the table they decide
nothing, since every ln2 build there is parallel.

`ln2_dyn` and `pi_dyn` keep their series nodes and extend them, so a request only sums
the terms past the cache. But extending also merges the new node into the cached one and
finalizes at the full length, and both cost more the larger the cache is. So the new
limbs alone mispredict, and the threshold falls quickly with the cache size. The rule, in
output limbs, is

    par = new * (cached^2 + HALF^2) >= CUTOFF * HALF^2

which is `new >= CUTOFF` on a cold cache, with the threshold halving at HALF cached
limbs and falling with its square past that. A linear `new * (cached + HALF)` fit the
earlier two-tree π and finalize-on-the-caller ln2, but not the current ones: their ties
fall about as fast as 1 / cached² from a few dozen cached limbs.

Measured on 8 threads (M-series, 11–15 rounds alternating, median), as parallel / serial,
with the mode forced for every call of the process (so the priming build of a warm row
ran in the same mode). "Serial" switches off only the cache-level parallelism; the
trees' own splits stay parallel.

ln2 runs its four series' shares, each extended and finalized, with nested `join2`:

| cold | 24 | 28 | 32 | 36 | 40 | 48 |
|---|---|---|---|---|---|---|
| ln2 | 1.18 | 1.14 | 0.88–1.03 | 0.88 | 0.78 | 0.67 |

| ln2 warm | Δ=1 | 2 | 4 | 6–8 | 10–12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| C=16 | | | | | 1.16 | 1.10 | 0.91 |
| C=32 | 2.08 | 1.96 | 1.35 | 1.32 | 1.18–1.27 | 0.83 | |
| C=64 | 1.11 | 1.11 | 0.93–1.00 | 0.79–1.03 | 0.57 | | |
| C=128 | 1.02 | 0.85 | 0.69 (Δ=3) | | | | |
| C=256 | 0.78 | 0.62–0.67 | 0.52 | | | | |
| C=1024 | 0.44 | | | | | | |

π spawns its one series into the pool with `rayon::in_place_scope`, where its top
merges can run in parallel, while the calling thread builds the root:

| cold | 64 | 96 | 128 | 160 |
|---|---|---|---|---|
| pi | 1.14 | 1.02 | 0.93 | 0.95 |

| pi warm | Δ=1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| C=256 | | | | 1.11 | 1.10 | 0.85 |
| C=1024 | | | 0.97 | 0.89 | 0.84 | 0.67 |
| C=4096 | 1.00 | 0.97 | 0.94 | 0.87 | | |

That gives CUTOFF = 32, HALF = 30 for ln2 and CUTOFF = 104, HALF = 256 for π. The
misfits are within a few percent: ln2 at C = 64, Δ = 6 (1.03) and C = 128, Δ = 1 (1.02).

Where a build runs matters beyond its own time, because each thread keeps its own FFT
plans (`mul.rs`'s thread-local `FFT_CACHE`) and scratch pool. Building π's root on a
worker made the next small serial extension on the caller 4–20% slower, which is why the
caller builds it. The first request after a switch between the serial and parallel paths
pays a few µs once: ln2 32 → 42 and 128 → 129 right after a parallel cold build cost
about 1.05x the old code, whose cold builds finalized on the caller. Raising ln2's CUTOFF
to 34 to keep cold 32 serial only moves that cost: the parallel 32 → 48 after it then
starts on cold workers (1.23x). Growing either cache a limb at a time in one process
shows no such cost (1.00x below 120 limbs).

`ln2_static` builds from scratch, so it compares its working limbs with
`STATIC_LN2_PAR_CUTOFF`. Each of its four scoped threads sums and finalizes one series,
and they cost about 60 µs to start. With the table bypassed (41 rounds): 1.54 at 24
limbs, 1.34 at 32, 0.95 at 40, 0.73 at 48, 0.56 at 64, 0.38 at 128 and 0.31 at 256.
That gives 40, down from 56 when the four finalizes ran serially on the caller. It only
matters if `ln2.bin` shrinks, since requests up to 1024 limbs copy the table.

`pi_static` runs its series (through V = Q′/T′) and its root on two threads from
`STATIC_PI_PAR_CUTOFF`, counted in limbs below π's integer limb, which only overlaps the
root's cost with the series: 4.94 at 24, 2.20 at 64, 1.36 at 128, 1.06–1.16 at 256, 1.05
at 384, 0.98 at 512 and 0.93 at 1000. That gives 480. Below `pi.bin`'s 1024 limbs it
copies the table, so past it the series always gets its own thread, with a stack sized
for N.
