# exp and ln: profile below 1024 limbs

A profile of `exp_dyn` and `ln_dyn` from 1 to 512 output limbs, with the
optimization opportunities it points to. It covers the source as of 2026-10-24
(the constant `DYN_EXP_SQR_BITS = 24` squares, gated off when the core's chunks
run in parallel, and the `trans.rs` cleanup). The profiling did not change any
production code.

Below 1024 limbs neither function is bound by its asymptotic algorithm. The time
goes to three things the bit-burst structure multiplies:

- **Slow base-case kernels.** Schoolbook multiplication is 11–64% of all samples
  from 1 to 128 limbs (exp 20–64%, ln 11–57%). `mul_buf` runs at about 3.9–4.8 cycles per 64×64 limb
  product. An unrolled `mulx`/`adcx`/`adox` row prototype runs at 2.0–2.3.
- **Fixed costs per series.** Every chunk pays a finalize division, a combine
  product (exp) or chain step (ln), term counting and node allocations. At 1–4
  limbs these dominate. A plain fixed-point Taylor loop beats the exp core
  3–7.7× at 1 limb.
- **Weak argument reduction in ln.** After its 2–3 square roots, ln's series
  run on an argument of about 2^-5, against exp's 2^-24. On a 2^-24 argument the
  ln core is 1.8–2.5× cheaper from 16 to 512 limbs. exp's 24 squarings cost 18–38% of exp.

## Where time goes

Intel i7-8750H, rustc 1.99.1, `prof` profile plus `-C force-frame-pointers=yes`,
pinned to CPU 2. **The core ran at a fixed 2.17 GHz** (powersave governor, no turbo
boost, checked with a dependent-add loop), so absolute times are about 1.9× what
the chip does at full turbo. Shares and ratios are unaffected.

One rayon thread, with the caller as the pool's only worker, so every `join`
runs inline. Inputs are 64 random X ∈ [1, 2) with m = 0, cycled. For ln, X ∈ [1, 2)
takes the e = 0 path. X ∈ [2^63, 2^64) adds the e·ln 2 term and matches the
[1, 2) shares within a few points at every size, so only [1, 2) is shown. 3,000–3,900
samples per run. Timings come from separate unsampled runs.

### exp_dyn

| Limbs | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| µs per call | 4.4 | 6.1 | 8.5 | 14.9 | 31.2 | 80.8 | 243 | 917 | 2355 | 5911 |
| Series leaves | 31% | 32% | 32% | 32% | 41% | 34% | 27% | 17% | 17% | 14% |
| Series merges | – | – | – | – | – | 9% | 18% | 19% | 28% | 37% |
| Series finalize (division) | 15% | 14% | 16% | 18% | 19% | 18% | 13% | 9% | 8% | 10% |
| Series setup (tree, term counts) | 11% | 10% | 10% | 6% | 3% | 2% | 1% | 1% | 1% | – |
| Combine products | 5% | 5% | 5% | 7% | 6% | 9% | 11% | 15% | 14% | 13% |
| **Squares** | **18%** | **20%** | **23%** | **29%** | **24%** | **24%** | **29%** | **38%** | **32%** | **26%** |
| Reduction, setup, scratch | 20% | 18% | 13% | 8% | 6% | 3% | 2% | 1% | 1% | – |

### ln_dyn

| Limbs | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| µs per call | 8.7 | 11.1 | 15.6 | 28.5 | 58.2 | 142 | 336 | 942 | 2698 | 7468 |
| Series leaves | 35% | 38% | 39% | 49% | 59% | 51% | 39% | 29% | 22% | 17% |
| Series merges | – | – | – | – | 2% | 15% | 31% | 44% | 52% | 57% |
| Series finalize (division) | 13% | 13% | 14% | 15% | 15% | 14% | 14% | 12% | 13% | 11% |
| Series setup, a², A·S | 11% | 9% | 7% | 6% | 4% | 4% | 3% | 3% | 2% | 3% |
| Chain (quotients, n/d updates) | 25% | 25% | 21% | 15% | 10% | 7% | 5% | 5% | 5% | 4% |
| Square roots | 5% | 6% | 9% | 9% | 4% | 5% | 6% | 6% | 4% | 5% |
| Tail 2n/d | 4% | 3% | 3% | 3% | 3% | 2% | 2% | 1% | 2% | 2% |
| Setup, scratch | 6% | 6% | 5% | 3% | 2% | 2% | 1% | 1% | – | – |

At 4–8 limbs ln's square roots go through `binom_sqrt` with `sqrt_approx_dyn`
inlined away. They are counted under square roots above.

### The kernels underneath

Every sample is attributed by its innermost non-trivial frame, whichever phase
called it. The allocator row counts samples that pass through Rust's allocator
shims (`__rdl_alloc*`, `__rdl_dealloc`).

| Limbs | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp: schoolbook multiply | 20% | 24% | 31% | 42% | 46% | 53% | 64% | 34% | 28% | 28% |
| exp: FFT | – | – | – | – | – | – | – | 33% | 44% | 47% |
| exp: hardware divide (Knuth 2/1) | 7% | 8% | 11% | 11% | 13% | 12% | 8% | 5% | 4% | 3% |
| exp: bin_split bookkeeping, leaf code | 15% | 16% | 13% | 11% | 13% | 10% | 7% | 4% | 3% | 2% |
| exp: allocator | 15.0% | 11.5% | 11.7% | 6.2% | 3.2% | 4.0% | 3.6% | 2.2% | 2.4% | 2.2% |
| ln: schoolbook multiply | 11% | 13% | 17% | 26% | 36% | 44% | 49% | 57% | 51% | 43% |
| ln: FFT | – | – | – | – | – | – | – | – | 11% | 25% |
| ln: hardware divide (Knuth 2/1) | 12% | 11% | 12% | 11% | 10% | 9% | 8% | 5% | 4% | 2% |
| ln: bin_split bookkeeping, leaf code | 18% | 19% | 18% | 20% | 19% | 15% | 12% | 8% | 5% | 3% |
| ln: allocator | 10.6% | 8.4% | 6.7% | 4.7% | 3.7% | 4.4% | 3.8% | 3.6% | 3.2% | 2.1% |

"Schoolbook multiply" covers the column kernels (`mul_elem`, `sqr_elem` and the
`hi_*_buf` loops), the row primitives (`mul_prim`, `add_mul`, `sub_mul`), and the
base cases that Karatsuba recursion bottoms out in.

### Why the shape looks like this

The series per call, from the chunk schedule (`chunk_width`, `exp_terms`,
`ln_terms`). These are approximate: exp assumes r with one leading zero bit, so
j = 23 squares. ln assumes δ ≈ 2^-(1 + roots).

| Limbs | exp: top Engel terms | exp: Hyper chunk terms | ln: roots, top BBP terms | ln: Hyper chunk terms |
| ---: | ---: | --- | ---: | --- |
| 1 | 8 | 8, 4, 2 | 1, 31 | 18, 10, 5, 2, 1 |
| 16 | 40 | 40, 23, 12, 6, 3, 2 | 2, 143 | 95, 47, 23, 11, 5, 2, 1 |
| 64 | 138 | 138, 79, 42, 22, 11, 6, 3, 2 | 3, 421 | 301, 162, 84, 43, 21, 10, 5, 2 |
| 256 | 508 | 508, 297, 161, 84, 43, 22, 11, 6, 3, 2 | 3, 1650 | 1178, 634, 330, 168, 85, 42, 21, 10, 5, 2 |

Every chunk pays a finalize division and either a combine product (exp) or a
chain step (ln), however few terms it has. Up to about 16 limbs each series is a
single leaf (merges are 0%), so leaves and finalize carry the cost. ln's top
series and first chunks have about 3× exp's terms, because ln's argument starts
about 19 bits higher.

## 1. Faster base-case kernels (mulx / adcx / adox)

`mul_buf`, `sqr_buf` and the `hi_*_buf` truncated products are product-scanning
(column-wise). Every 64×64 product feeds a single `add/adc/adc` chain into a
three-word accumulator, so each product waits on the last one's carries. The row
primitive `add_mul_x86` is the classic `mul; add; adc; add [mem]; adc` loop with
one carry chain. Nothing in the crate uses BMI2/ADX, which this CPU has.

n × n products, ns per call, and cycles per limb product at 2.17 GHz. The
prototype rows are scratch code built on one unrolled ×4 `addmul_1` with
`mulx` and separate CF/OF carry chains. Every output was checked equal to
`mul_buf`.

| n | `mul_buf` | rows of `add_mul_limb` | rows of adx ×4 prototype | cycles per product (same order) |
| ---: | ---: | ---: | ---: | --- |
| 8 | 174 | 156 | 130 | 5.9, 5.3, 4.4 |
| 16 | 750 | 455 | 331 | 6.4, 3.9, 2.8 |
| 32 | 2549 | 1663 | 1066 | 5.4, 3.5, 2.3 |
| 64 | 8993 | 6676 | 3955 | 4.8, 3.5, 2.1 |
| 128 | 33036 | 30053 | 15415 | 4.4, 4.0, 2.0 |

`mul_buf` varied 15–20% between runs (7323 ns at n = 64 in another). The
comparison within a run is the point: the prototype is 2.1–2.3× faster from 32
limbs. A tuned basecase (GMP-style `mul_basecase` and `sqr_basecase`) should do
better still.

Changes to make:

- `addmul_1`, `submul_1` and `mul_1` rows with `mulx`/`adcx`/`adox`, unrolled.
  Use `lea` and `jrcxz` (or a counted unroll) for loop control, since `dec` and
  `inc` clobber OF.
- `mul_buf` as rows of these. `sqr_buf` as the half triangle plus a doubling and
  diagonal pass, which halves the product count. `hi_mul_buf`/`hi_sqr_buf` as
  truncated rows that skip the products below the guard window.
- Knuth's multiply-subtract (`sub_mul`) on the same row primitive.
- Runtime dispatch on `is_x86_feature_detected!("bmi2") && ("adx")`, cached in
  a static, with today's kernels as the fallback (and for aarch64).
- Then retune `KARATSUBA_CUTOFF`, `KARATSUBA_SQR_CUTOFF`, `FFT_SQR_CUTOFF`,
  `PARTIAL_MUL_CUTOFF`, `PARTIAL_SQR_CUTOFF` and the division cutoffs, since all
  of them move.

Expected gain (Amdahl on the shares above, not measured): with the schoolbook
share 2.1× faster, exp gets about 1.3× faster at 16 limbs and 1.5× at 64, and
ln about 1.35× at 64 and 1.4× at 128. Below 8 limbs the gain is smaller because
fixed costs dominate. Above 128 FFT takes over. The results are integer-exact,
so outputs stay bit-identical, and every other caller in the crate gains too.

## 2. Stronger argument reduction for ln (and exp)

The cores at a given argument size, against the full functions, in µs (one
run). Entries run at out.len() + 1 limbs, as the reductions call them.

| Limbs | `exp_dyn` | exp, no squares | exp core at 2^-1 | exp core at 2^-24 | `ln_dyn` | ln core at 2^-5 | 2^-10 | 2^-16 | 2^-24 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.1 | 5.4 | 6.2 | 3.0 | 7.8 | 5.4 | 4.7 | 3.6 | 3.2 |
| 4 | 8.0 | 11.7 | 13.5 | 5.2 | 11.9 | 8.5 | 8.0 | 6.3 | 5.5 |
| 16 | 28.3 | 37.7 | 42.5 | 19.8 | 51.8 | 31.8 | 24.8 | 16.8 | 16.4 |
| 64 | 232 | 306 | 296 | 156 | 361 | 277 | 178 | 132 | 112 |
| 128 | 907 | 936 | 919 | 549 | 919 | 795 | 580 | 426 | 340 |
| 256 | 2304 | 2321 | 2372 | 1555 | 2608 | 2267 | 1718 | 1213 | 1160 |
| 512 | 5674 | 6016 | 6024 | 4324 | 7593 | 6172 | 4708 | 4249 | 3375 |

- **ln:** its core runs near 2^-5 today. At 2^-24 it is 1.8–2.5× cheaper from 16
  to 512 limbs. The square roots that get it to 2^-5 cost another 5–9%.
- **exp:** the core at 2^-24 is 24–39% cheaper than the whole of `exp_dyn`.
  That's the most a reduction that skipped the squares could save.

### 2a. exp-based reduction for ln (uses only existing code)

Take y₀ ≈ ln u from f64 (53 bits). Compute 1 + δ = u·e^(−y₀) with one `exp_dyn`
of a 53-bit argument and one multiply. Then ln u = y₀ + ln(1 + δ), with |δ|
around 2^-50, where the ln core is cheap. Priced from today's functions (another
run, compared within it):

| Limbs | exp of 53-bit y₀ | ln core at 2^-50 | multiply | sum | `ln_dyn` same run | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.1 | 2.0 | 0.1 | 5.2 | 8.0 | 1.54× |
| 4 | 5.1 | 3.8 | 0.1 | 9.0 | 9.4 | 1.04× |
| 16 | 16.5 | 11.6 | 0.5 | 28.6 | 51.3 | 1.79× |
| 64 | 107 | 75 | 5.6 | 187 | 278 | 1.49× |
| 128 | 365 | 262 | 17 | 644 | 746 | 1.16× |
| 256 | 915 | 861 | 29 | 1804 | 2908 | 1.61× |
| 512 | 2069 | 2676 | 69 | 4814 | 7225 | 1.50× |

That run was noisy (`ln_dyn` at 64 limbs read 278 µs, against 336–361 µs in
other runs), but it gives about 1.5× at most sizes. It's also pessimistic: the
53-bit exp still pays its 24 squares, most of its cost at these sizes. Run with
fewer squares for a short argument, it should come out cheaper. The e·ln 2 term
works as today.

### 2b. Multi-prime argument reduction (both functions)

F. Johansson, *Computing elementary functions using multi-prime argument
reduction* (2022). Cache ln p for a few small primes, the way `ln2_dyn` caches
ln 2; Machin-like atanh formulas fit the existing bin_split machinery. Then:

- **ln:** pick small integers cᵢ with u·Π pᵢ^cᵢ within 2^-q of 1 (from a small
  precomputed table). One short × long multiply and one long ÷ short division
  give the reduced argument, and ln u = ln(u·Π pᵢ^cᵢ) − Σ cᵢ ln pᵢ. This replaces
  the square roots and should approach the 2–2.5× bound above.
- **exp:** generalize today's n·ln 2 to x = Σ cᵢ ln pᵢ + r with |r| < 2^-q, so
  e^x = Π pᵢ^cᵢ · e^r. The prime powers are exact rationals, so applying them
  costs one division instead of 24 squares, up to the 24–39% bound above.

This is the larger project of the two. It needs the prime-log caches and the
coefficient tables.

Both 2a and 2b change the last bits of the results (still within 1 ulp), so
they need the full oracle suites, not bit-identity checks.

## 3. A direct path at 1–4 limbs

A naive fixed-point Taylor loop (schoolbook `mul_buf` every term plus
`div_prim`, the shape of the test oracle) against `exp_dyn_entry` on the same
argument, in ns:

| Limbs | Argument | Taylor | `exp_dyn_entry` | Terms | Taylor speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 2^-1 | 1688 | 5268 | 28 | 3.1× |
| 1 | 2^-24 | 320 | 2453 | 6 | 7.7× |
| 2 | 2^-1 | 3683 | 7477 | 38 | 2.0× |
| 2 | 2^-24 | 708 | 3273 | 8 | 4.6× |
| 4 | 2^-1 | 10428 | 11040 | 54 | 1.1× |
| 4 | 2^-24 | 1922 | 5274 | 12 | 2.7× |
| 8 | 2^-24 | 7933 | 8633 | 21 | 1.1× |
| 16 | 2^-24 | 38795 | 18727 | 39 | 0.5× |

At 1–2 limbs the bit-burst machinery costs more than it saves. At 1 limb the 24
squares alone take about 1.6 µs of exp's 4.1 µs, and Taylor on the raw r is
nearly as fast as squares plus Taylor. Below a cutoff of about 4 limbs, a direct
path would make exp about 2× faster at 1–2 limbs (an estimate): Taylor, or Paterson–Stockmeyer
with term precision that drops as terms shrink, and fewer or no squares. ln can
do the same with its atanh series. The crossover moves up once item 1 lands.
Results change in the last bit, within 1 ulp.

## 4. Divide by reciprocal in Knuth division

The hardware `div` instruction (`div_rem_2_1_x86`), one per quotient digit, is
7–13% of both functions from 1 to 64 limbs. It runs in every finalize division,
ln's chunk quotients and its tail, and the exp reduction. Möller–Granlund 2/1
division with a precomputed reciprocal of the normalized top limb
(`udiv_qrnnd_preinv`) costs two multiplications and a few additions per digit,
against tens of cycles for `div r64` on this core. Expect most of that 7–13%
back. The change is exact.

## 5. Fewer allocations at tiny sizes

Heap allocations per call:

| Limbs | 1 | 4 | 16 | 64 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exp | 13 (0.9 KB) | 16 (1.8 KB) | 23 (5.8 KB) | 98 (35 KB) | 450 (224 KB) | 934 (550 KB) |
| ln | 18 (2.4 KB) | 21 (3.8 KB) | 29 (11 KB) | 122 (55 KB) | 564 (340 KB) | 1163 (827 KB) |

They're mostly the dynamic bin_split nodes' `Vec`s (`vec![0; n]` gives
`alloc_zeroed`), plus `exp_dyn_entry`'s `parts` and `get_split_vec` and ln's
`pool` and `chunks`. The allocator is 15% of exp and 11% of ln at 1 limb, and
2–4% from 16 limbs on. Options: draw node parts from `ScratchGuard`-style pools,
or give single-leaf series (every series up to about 16 limbs) a path that
builds P and Q in scratch and finalizes directly without a node. This only pays
at the smallest sizes.

## 6. Don't enter rayon from single-thread pools

`bin_split_tree_at` splits with `join2(N::PARALLEL && limbs >= cut.par_limbs, ..)`,
which checks node size only. The merges already require a worker
(`merge_par` checks `rayon::current_thread_index()`). So from about 128 limbs,
every series' top split injects into rayon from the caller's thread, even when
the pool has one thread and the work just moves to it. Same harness, ms per call:

| Limbs | exp, injected (`RAYON_NUM_THREADS=1`) | exp, caller is the worker | Cost | ln, injected | ln, caller is the worker | Cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.996 | 0.917 | +8.6% | 1.090 | 0.942 | +15.7% |
| 256 | 2.598 | 2.355 | +10.3% | 2.960 | 2.698 | +9.7% |
| 512 | 6.154 | 5.911 | +4.1% | 7.737 | 7.468 | +3.6% |

Gate the tree split on `rayon::current_num_threads() > 1` as well. Caveat: the
one-thread exp-squares tuning in `bin_split_cutoff_tuning.md` was measured with
this overhead included.

**Done (2026-10-28).** `bin_split_tree_at` now gates its split on
`rayon::current_num_threads() > 1`. With `RAYON_NUM_THREADS=1` and
`cargo bench-bin-split exp` (two X per size, 9 rounds), exp got 2.5–6% faster at
128–512 limbs and ln 2–11% faster, short of the table above because the merges
already ran inline. Outputs are unchanged.

## 7. Squaring dispatch and the squares themselves

Squares are 18–38% of exp. Through `hi_sqr_dyn` they use the column-wise
`hi_sqr_buf` up to `PARTIAL_SQR_CUTOFF` (128 output limbs), and a guarded full
FFT square above. ns per call at n limbs:

| n | `hi_sqr_dyn` | `hi_sqr_buf` | `sqr_dyn` | `sqr_buf` | `mul_dyn(u, u)` | `mul_buf(u, u)` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 353 | 348 | 386 | 378 | 561 | 533 |
| 33 | 1015 | 1007 | 1145 | 1142 | 1893 | 2068 |
| 34 | 1085 | 1070 | 4012 | 1199 | 1946 | 2174 |
| 48 | 2032 | 2018 | 4678 | 2355 | 3879 | 4181 |
| 65 | 3606 | 3566 | 7242 | 4412 | 5788 | 7372 |
| 96 | 7680 | 7623 | 9215 | 9739 | 11454 | 15240 |
| 129 | 14330 | 13445 | 14229 | 16971 | 17848 | 26703 |
| 257 | 30819 | 49976 | 31915 | 62871 | 37889 | 100237 |

- **`FFT_SQR_CUTOFF` (33) is too low.** `dyn_sqr_dispatch` has no Karatsuba
  tier, and the FFT square is 1.6–3.3× slower than schoolbook `sqr_buf` from 34
  to 65 limbs. The crossover is near 96. exp's squares don't hit this, but other
  callers of `sqr_dyn` do. Add a Karatsuba square tier or raise the cutoff.
- **`PARTIAL_SQR_CUTOFF` (128) is about right**: the column kernel still edges
  out the FFT path at 129 and loses by 257. With item 1, schoolbook and
  Karatsuba squaring move ahead of FFT until well past 129.
- **The square count.** The fixed-j sweep in `bin_split_cutoff_tuning.md`
  favored j ≈ 12 at 128–256 limbs (3–5% better than 24). A size-dependent count
  would take that. It matters less once squares get cheaper, and item 2b would
  remove them.

**Done (2026-10-28).** The mul.rs retune covered the dispatch bullets. `sqr_dyn` now has
a Karatsuba tier up to `FFT_SQR_CUTOFF = 340`, and the high-square crossover is a
measured `DYN_PARTIAL_SQR_CUTOFF = 464`. After item 6, the one-thread j sweep was run
again. j = 24 is still best (1.013× on the geometric mean from 1 to 4096 limbs). j ≈ 12
no longer wins at 128–256, and 512 favors j ≤ 8 by about 6%. So `DYN_EXP_SQR_BITS`
stays a constant 24. See `bin_split_cutoff_tuning.md`.

## 8. Smaller items

- **`exp_terms`** binary-searches a Stirling bound in f64 (`log2` per step, in
  libm) for every series. That's about 5% at 1–2 limbs. An integer bound (a
  `log2(n!)` table for small n, `ilog2` Stirling beyond) or memoizing per frame
  would remove it.
- **ln's chain at 1–4 limbs** sends 2–4-limb chunk quotients through the general
  `div_dyn` (preflight, dispatch, copies). The chain is 21–25% of ln there, and
  the quotient half of it is the general division. A direct 2/1 or 3/2 path for
  chunk widths up to 64 bits would skip that machinery.
- The static paths (`exp_static`, `ln_static`) weren't profiled. They share the
  cores and kernels, so items 1, 4 and 5 apply. Their zero-initialized
  `[u64; N]` buffers add an O(N) cost per array that the dynamic path doesn't
  have.

## Method and reproduction

The profile uses the PC sampler approach of `benches/binom_sqrt_sampler.c`:
`ITIMER_PROF` at 1,000 µs, loaded with `LD_PRELOAD`. A scratch copy added a
frame-pointer walk to the handler. It reads RBP from the signal context and
follows saved frame pointers within the main thread's stack bounds, so each
sample records a full call stack. With the harness built using
`-C force-frame-pointers=yes`, stacks reached 8–30 frames, and none of the 30
runs lost a sample.
`addr2line -a -f -C -i` resolves each address (return addresses minus one)
together with its inlined frames. That attributes samples to inlined callers too,
such as the `series` closures in `exp_dyn_entry`.

Phases are assigned by which `trans.rs` and `bin_split.rs` functions appear on a
sample's stack. For example:

- `exp_combine` is the combine phase.
- `Node>::finalize` or `ratio_to_fraction` under `trans::series` is finalize.
- `Node>::leaf` and `Node>::merge` are leaves and merges.
- `hi_sqr` under `exp_reduced` but outside the entry is the squares phase, along
  with `exp_reduced` lines inside the squares loop.
- `sqrt::` is ln's square roots, and `chunk_quotient` and `mul_shr` under
  `ln_chain` are the chain.

Kernels are the innermost frame after skipping `core::`/`alloc::`/`std::`
helpers. Unnamed libc samples (local memcpy/memset/malloc symbols) are attributed
to their first named caller.

The harness (`benches/trans_profile.rs` in the scratch copy) loops
`exp_dyn`/`ln_dyn` at one size over 64 random inputs, in a 1 s unsampled timing run
and a separate 3 s sampled run. It builds a one-thread global pool with `use_current_thread()`, or
with `PROFILE_INJECT=1` calls from outside a one-thread pool (item 6). A counting
`#[global_allocator]` gives item 5's numbers. The micro-benchmarks for items 1,
2, 3 and 7 are separate scratch benches (`row_micro`, `entry_micro`,
`taylor_micro`, `sqr_micro`); item 1's prototype kernel lives in `row_micro`.
These harnesses aren't checked in.

Caveats: these are sampled CPU-time shares, not hardware counters. Phase shares
at 1 limb rest on about 3,000 samples per run, so treat differences under a
point or two as noise. The micro-benchmarks are best-of-several on one core at
the fixed 2.17 GHz. Gains marked as estimates are Amdahl projections from the
measured shares.
