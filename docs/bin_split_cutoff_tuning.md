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
