# Square-root cutoff tuning

Machine: Intel i7-8750H, `powersave` governor, pinned to one core (`taskset -c 3`).
Probe: `benches/probes/sqrt_cutoffs.rs`, built by `scripts/python/tune_sqrt_cutoffs.py`
from a constant-cutoff copy of `src/utils/sqrt.rs` under `target/`. Every timed
candidate is first validated against `binom_sqrt`; variants alternate within each
round and the median of 7-15 rounds of thread CPU time is used.

The values below are superseded by the
[retune after the Knuth rewrite](#retune-after-the-knuth-rewrite-2026-10-29); the method is unchanged.

The former single `ZIMMERMAN_SQRT_CUTOFF = 17` is replaced by seven constants:

| Constant | Old | New | 0.2% plateau |
|---|---:|---:|---|
| `ZIMMERMAN_SQRT_LEAF_CUTOFF` (recursive) | 17 | 15 | 12-20 end-to-end |
| `DYN_SQRT_REM_ZIMMERMAN_CUTOFF` | 17 | 20 | 18-23 |
| `DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF` | 17 | 15 | 12-17 |
| `DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF` | 17 | 15 | 11-17 |
| `STATIC_SQRT_REM_ZIMMERMAN_CUTOFF` | 17 | 20 | 18-23 |
| `STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF` | 17 | 15 | 12-17 |
| `STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF` | 17 | 14 | 11-16 |

## Recursive leaf

`core` mode times one Zimmermann descent whose half-width child is solved by
binomial sqrt against `binom_sqrt_core` at the same root width, which is exactly
the decision made at each level of `zimmermann_sqrt_core`. Ratio descent/binomial:

| Root limbs | 10 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|
| dyn | 1.066 | 1.053 | 1.038 | **0.988** | 0.969 | 0.966 | 0.942 | 0.924 | 0.882 | 0.834 |
| static, N=next_pow2(2n) | 1.070 | 1.048 | 1.013 | 1.007 | **0.966** | 0.971 | 0.925 | 0.925 | 0.907 | 0.846 |
| static, N=1024 | 1.060 | 1.060 | 1.034 | 1.012 | **0.994** | 0.958 | 0.949 | 0.963 | 0.928 | 0.851 |
| static, N=4096 | 1.193 | 1.169 | 1.112 | 1.094 | 1.015 | 1.018 | **0.977** | 0.967 | 0.930 | 0.861 |

Complete root-remainder calls at root widths 30-512 (dyn, static with tight N,
static with N=4096) sweeping the leaf over 12-20 are flat: every leaf's geometric
mean is within 1.0-2.8% of the per-size best, which is inside run-to-run noise.
The one-step crossover is therefore the deciding signal, and one shared leaf of 15
is used; the recursion is shared by both allocation models.

## Top-level cutoffs

Each output mode and allocation model compares the binomial path (`normalized` around `binom_sqrt_core`) against the
Zimmermann entry (leaf 15) for root widths 4-48 across four input shapes:
`full` (`x.len() = 2n`), `near` (`2n-1`), `mid` (`3n/2+1`) and `padded` (`n+1`),
with alternating normalized and shifted top limbs. The cutoff minimizes the
geometric-mean slowdown against the per-case best over all sizes and shapes. Two
independent runs (seeds differ; 7x10 ms and 15x15 ms) agree within one limb.

Geometric-mean Zimmermann/binomial over shapes, combined runs:

| Root limbs | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dyn rem | 1.14 | 1.13 | 1.11 | 1.11 | 1.07 | 1.04 | 1.01 | 1.01 | 0.98 | 0.97 | 0.96 |
| dyn only | 1.01 | 1.02 | 1.03 | 0.98 | 0.98 | 0.93 | 0.94 | 0.90 | 0.91 | 0.87 | 0.90 |
| dyn approx | 1.02 | 1.01 | 1.01 | 0.99 | 0.97 | 0.93 | 0.92 | 0.92 | 0.88 | 0.88 | 0.87 |
| static rem | 1.14 | 1.10 | 1.11 | 1.08 | 1.07 | 1.05 | 1.09 | 1.02 | 1.08 | 0.98 | 0.97 |
| static only | 1.03 | 1.01 | 1.02 | 0.98 | 0.98 | 0.95 | 0.94 | 0.91 | 0.91 | 0.88 | 0.89 |
| static approx | 1.00 | 1.00 | 0.99 | 0.99 | 0.95 | 0.94 | 0.95 | 0.92 | 0.89 | 0.87 | 0.88 |

Root-remainder calls switch later because the Zimmermann entry always pays for
the final low square and denormalization, while root-only calls usually return
before squaring.

The crossover also depends on the input shape, which dispatch does not see:
`full` and `padded` inputs favor Zimmermann near 10-12 (only/approx) or 17
(rem), while `near` and `mid` favor it only from 16-24. The constants are the
best single threshold on root width.

## Static capacity

The static constants were measured with `N = next_pow2(2n)`, the natural case in
which `x` nearly fills the capacity. Static division and squaring zero-initialize
`[u64; N]` stack scratch on every call, so their cost grows with `N` rather than
with the operands. With oversized capacity, the best cutoffs rise:

| Capacity | rem | only | approx |
|---|---:|---:|---:|
| next_pow2(2n) | 20 | 15 | 14 |
| 1024 | 23 | 15 | 20 |
| 4096 | >= 49 | 17 | 37 |

At N=4096, Zimmermann is 2.0-2.4x slower than binomial at 8 root limbs in the
rem and approx modes. The approx mode suffers most because quotient-only
`div_static` copies operands into additional capacity-sized buffers. A
capacity-aware threshold, or scratch that avoids zeroing all `N` limbs, would
remove this sensitivity.

## Reproduction

The probe and scripts below were deleted on 2026-11-08, when the sqrt constants became knobs
(see [Values before the tuner](#values-before-the-tuner-2026-11-08)). They still run at commit
`faa9971`, in a worktree (`git worktree add --detach <dir> faa9971`).

```bash
cargo bench-cutoffs --no-run
python3 scripts/python/tune_sqrt_cutoffs.py --rlib target/release/deps/libbig_bits-HASH.rlib \
  --output target/sqrt_cutoff_tuning/v4
cd target/sqrt_cutoff_tuning/v4
# Recursive leaf: one descent (4096) vs binomial (0).
taskset -c 3 ./sqrt_cutoffs --sizes 4:48 --leaves 0,4096 --families dyn,static \
  --modes core --pattern random --rounds 9 --ms 25
# Top level: binomial (0) vs Zimmermann with leaf 15.
taskset -c 3 ./sqrt_cutoffs --sizes 4:48 --leaves 0,15 --modes rem,only,approx \
  --shapes full,near,mid,padded --pattern mixed --rounds 7 --ms 10 --families dyn,static
```

## Retune after the Knuth rewrite (2026-10-29)

Binomial sqrt runs `knuth_est` once per root limb, and the Zimmermann descent divides with
`div_rem_*`. Both got faster with the Knuth rewrite, by different amounts, so every constant above
was remeasured. The probe and protocol are unchanged: `v5`, built against the retuned division
constants, on CPU 2. Leaves 9 and 11 were added to the probe.

| Constant | Before | After |
|---|---:|---:|
| `ZIMMERMAN_SQRT_LEAF_CUTOFF` | 15 | 11 |
| `DYN_SQRT_REM_ZIMMERMAN_CUTOFF` | 20 | 16 |
| `DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF` | 15 | 11 |
| `DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF` | 15 | 11 |
| `STATIC_SQRT_REM_ZIMMERMAN_CUTOFF` | 20 | 16 |
| `STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF` | 15 | 11 |
| `STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF` | 14 | 11 |

One descent / binomial (`core`, 9 × 25 ms):

| Root limbs | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|
| dyn | 1.056 | 1.035 | **0.982** | 0.996 | 0.946 | 0.949 | 0.890 |
| static, N = next_pow2(2n) | 1.047 | 1.031 | **0.984** | 0.991 | 0.949 | 0.939 | 0.887 |

Zimmermann (leaf 11) / binomial at the top level, geometric mean over the four shapes and both
runs (7 × 10 ms and 15 × 15 ms with another seed):

| Root limbs | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|
| dyn rem | 1.201 | 1.149 | 1.122 | 1.103 | 1.071 | 1.033 | **1.009** | 0.992 |
| dyn only | 1.071 | **1.009** | 1.013 | 0.972 | 0.979 | 0.909 | 0.910 | 0.869 |
| dyn approx | 1.048 | **1.005** | 0.989 | 0.966 | 0.947 | 0.907 | 0.886 | 0.871 |
| static rem | 1.186 | 1.131 | 1.112 | 1.092 | 1.064 | 1.024 | **0.998** | 0.986 |
| static only | 1.063 | **0.995** | 1.006 | 0.964 | 0.968 | 0.906 | 0.900 | 0.868 |
| static approx | 1.028 | **0.980** | 0.966 | 0.955 | 0.935 | 0.895 | 0.875 | 0.865 |

Each cutoff is the smallest width from which Zimmermann is never more than about 1.5% slower,
the single-thread noise band. Each lies inside the 0.2% plateau of the geometric-mean slowdown
that the earlier sections minimized. Those minimizers are 17, 13, 12, 16, 13 and 11 in table
order, and the two runs agree within one or two limbs. Zimmermann's lead also grows faster than
before: at 22 root limbs dyn rem takes 0.87 of binomial's time (0.96 before), and at 48 the six
modes take 0.49–0.56.

```bash
cargo bench-cutoffs --no-run
python3 scripts/python/tune_sqrt_cutoffs.py --rlib target/release/deps/libbig_bits-HASH.rlib \
  --output target/sqrt_cutoff_tuning/v5
cd target/sqrt_cutoff_tuning/v5
taskset -c 2 ./sqrt_cutoffs --sizes 4:48 --leaves 0,4096 --families dyn,static \
  --modes core --pattern random --rounds 9 --ms 25
taskset -c 2 ./sqrt_cutoffs --sizes 4:48 --leaves 0,11 --modes rem,only,approx \
  --shapes full,near,mid,padded --pattern mixed --rounds 7 --ms 10 --families dyn,static
taskset -c 2 ./sqrt_cutoffs --sizes 4:48 --leaves 0,11 --modes rem,only,approx \
  --shapes full,near,mid,padded --pattern mixed --rounds 15 --ms 15 --families dyn,static \
  --seed 12345
```

## Wrapped remainder (untuned, 2026-11-02)

Dynamic Zimmermann stages can form their remainder R = X − s² from residues mod B^m − 1
instead of from the division's remainder and the low root's square (see `sqrt.rs` in
`CLAUDE.md`, and recommendation 1 of `docs/agm_profile.md`). The roots and remainders are
bit-identical either way, so only the time moves. There are two placeholders, both `// GUESS`, and
the cyclic FFT's dispatch limit:

| Constant | Value | Decides |
|---|---:|---|
| `DYN_SQRT_WRAPPED_REM_CUTOFF` | 341 | Root limbs from which a dynamic stage (and `sqrt_dyn`'s top stage) takes the wrapped remainder |
| `DYN_SQR_BNM1_FFT_CUTOFF` | 340 | Operand limbs above which `sqr_mod_bnm1_dyn` runs the cyclic FFT rather than `sqr_dyn` and a fold |
| `FFT_BNM1_16BIT_CUTOFF` | 57344 | The cyclic FFT's dispatch limit in modulus limbs: the largest `bnm1_size` below where random operands tripped its old check (a 2^48 size gate, since removed) more than 1 in 2^10 times, not timed against the stages (`docs/fft_precision.md`). That was measured as 64800 at the old transform sizes and not remeasured; 57344 is the largest `bnm1_size` below it now. Stages whose `bnm1_size(k + 1)` passes it stay classic |

The placeholders follow `FFT_SQR_CUTOFF`: every stage whose wrapped square is on the FFT takes it.
That isn't measured, and it costs time at the small end. Release A/B against the classic stages
(cutoff `usize::MAX` through `sqrt_approx_dyn_at`/`sqrt_dyn_at`), M1, `prof`, one thread, minimum
of 15 alternating rounds, with identical outputs asserted. Wrapped time ÷ classic time:

| `sqrt_approx_dyn`, (g + 2) → g | 264 | 524 | 1044 | 2084 | 4164 | 8324 | 16644 |
|---|---:|---:|---:|---:|---:|---:|---:|
| wrapped / classic | 1.00 | 1.00 | 1.04 | 0.93 | 0.88 | 0.86 | 0.85 |

| `sqrt_dyn`, 2k → k | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|---:|---:|
| wrapped / classic | 1.00 | 1.12 | 1.04 | 0.95 | 0.88 | 0.87 | 0.87 |

A second run agreed within 1%. Through 524 limbs no stage reaches the cutoff. A stage of a few
hundred limbs loses, probably because its wrapped square of k limbs runs an FFT while the classic
square of k/2 is still Karatsuba, and the classic division keeps its remainder cheaply through BZ
there (neither cause was timed separately). To tune it, time the forced entries per root width, or one
stage at a time as `core` mode does for the leaf. The `one_descent` probe passes no
wrapped cutoff, so it still times the classic stage. A static version needs a cyclic NTT square
with all three primes at one transform length.

## Reciprocal carry (2026-11-05)

Each Zimmermann stage divides by the whole root of the stage below, so the last stage's reciprocal is
a reciprocal of the top of this stage's divisor. A dynamic root keeps it in an `RcpCarry`, and a stage's
NR division resumes its chain from there. The recursion is linear, so the carry lives in
`sqrt_dyn_output`'s division closure and the stages never see it. The schedules line up:
in a model of every root width to 40000 limbs, for full inputs and the AGM's (k + 2)-limb ones, the child's
precision passes the parent's second-to-last step by 1 or 2 limbs, so only the last refinement runs.

| Constant | Value | Meaning |
|---|---:|---|
| `DYN_SQRT_RCP_CARRY_CUTOFF` | 256 | Root limbs from which a root acquires the carry's scratch. Not a crossover: without it, roots of at most 32 limbs lost 5–8 ns (1–2%) to a carry they can't read |

New time ÷ old time, M1, one thread, release without LTO. Each process gets a different heap layout from a
padding global allocator (every allocation shifted by a random multiple of 16 bytes up to 8 KiB), and a
cell is the median over 9 processes of the fastest batches. A second copy of the old binary with other
seeds is the control, within about ±1% here. Comparing identical binaries without the padding hides
layout noise: 128-limb roots, which can't read a carry, read up to 16% faster by luck.

These builds don't pin code alignment, and relinking identical code has moved `sqrt_dyn` at 16384 limbs by
6% on its own, as rustfft's loops shift, so the table's large sizes share that uncertainty. Turning only the
seed off in one binary doesn't: `sqrt_approx_dyn` then takes 4.6% longer at g = 1044, 6.9% at 2084, 9.4% at
4164 and 11% at 8324. To pin alignment, build every crate and the harness with
`-C llvm-args=-align-all-functions=6 -C llvm-args=-align-all-nofallthru-blocks=6`, which brought an A/B of
builds doing the same work to within ±1%.

| Limbs | ~1024 | ~2048 | ~4096 | ~8192 | ~16384 |
|---|---:|---:|---:|---:|---:|
| `agm_dyn` (g = 1044 ...) | 0.974 | 0.968 | 0.950 | 0.933 | 0.922 |
| `ln_agm_dyn_at::<3>` | 0.979 | 0.962 | 0.949 | 0.939 | 0.922 |
| `sqrt_approx_dyn`, (g + 2) → g | 0.958 | 0.941 | 0.916 | 0.889 | 0.875 |
| `sqrt_dyn`, 2k → k | 0.974 | 0.941 | 0.929 | 0.907 | 0.891 |
| `sqrt_only_dyn`, 2k → k | 1.001 | 0.943 | 0.928 | 0.909 | 0.901 |

`sqrt_only_dyn` at 1024 has no gain because its top stage keeps a remainder, and `div_rem_dyn` takes BZ
there. Elsewhere nothing moved. As geometric means over their cases, 33 roots from 11 to 192 limbs came to 1.002
(control 1.000); `div_dyn`, `div_rem_dyn`, `hi_div_dyn`, `rcp_dyn`, `exp_dyn` and `ln_dyn_at` to 0.998–1.002; and
the static entries, whose logic didn't change, to 0.993–1.011, inside their control's case-by-case spread.

### Forcing NR at smaller sizes

Does a carry make NR worth taking lower than dispatch does? No, at least on this machine. The carry
makes NR cheaper, but NR's cost at small sizes is its remainder, and only NR stages can leave a carry.

One balanced division, n = 2d − 1 limbs, against Knuth (d ≤ 140) or BZ, one fixed-layout run, with the
child's reciprocal for "seeded":

| d | 48 | 64 | 96 | 128 | 192 | 256 | 384 | 512 | 768 | 1024 | 1536 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quotient only: NR | 1.00 | 0.96 | 0.88 | 0.84 | 0.87 | 0.82 | 0.91 | 0.97 | 1.03 | 1.05 | 0.90 | 0.89 | 0.83 |
| Quotient only: seeded NR | 0.88 | 0.83 | 0.79 | 0.77 | 0.80 | 0.77 | 0.85 | 0.90 | 0.94 | 0.94 | 0.81 | 0.79 | 0.71 |
| With remainder: NR | 1.77 | 1.56 | 1.50 | 1.40 | 1.47 | 1.32 | 1.37 | 1.37 | 1.34 | 1.34 | 1.17 | 1.14 | 1.04 |
| With remainder: seeded NR | 1.64 | 1.44 | 1.41 | 1.31 | 1.40 | 1.28 | 1.32 | 1.29 | 1.27 | 1.24 | 1.09 | 1.03 | 0.92 |

End to end, from the same kind of sweep (forced NR from a given divisor width, whole roots):

- **With remainder (classic stages):** it loses. `sqrt_only_dyn`'s top stage forced to NR takes 1.20× at 512
  limbs, 1.15× at 768, 1.07× at 1024 and 1.12× at 1536. `sqrt_approx_dyn` with its classic stages forced takes
  1.03–1.06× at 524 limbs and stays within noise from 1044 up, where those stages are a small share of the
  root. Running it with the seed off shows what the extra seeds repay: about a point (0 to 2.4), no more
  than the forced stages cost.
- **Lower wrapped cutoff (to make small stages quotient-only):** it loses. Below `FFT_SQR_CUTOFF` the wrapped
  square is a whole square and a fold, 3–3.7× the classic low-half square. A cutoff of 64–192 takes 1.06–1.25×
  from 384 to 768 limbs, and up to 1.04× above that.
- **Quotient only:** every wrapped stage already takes NR, so the only stage left is the top of a small
  `sqrt_approx_dyn`, whose child is Knuth and has no seed. Forcing it helps, but not because of the carry:
  0.974 at 110 limbs, 0.970 at 128, 0.946 at 160, 0.928 at 192 and 0.911 at 224, within 0.7% of that with
  the seed off. From 256 limbs dispatch already takes NR, and at 96 limbs and below NR wins nothing. This is `div_dyn`'s
  crossover (`DYN_DIV_FFT_NR_BZ_CUTOFF`, about 126 limbs on balanced shapes), which looks high on the M1,
  and belongs with the planned tuning revamp rather than in the root.

### The wrapped cutoff, with the carry

The sweep points the other way: at the placeholder, NR starts too low. Whole-root time ÷ time at 341:

| `DYN_SQRT_WRAPPED_REM_CUTOFF` = 1024 | 384 | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sqrt_dyn` | 0.859 | 0.891 | 0.927 | 1.003 | 1.010 | 1.000 | 1.013 | 0.999 | 0.999 |
| `sqrt_only_dyn` | 0.996 | 1.000 | 0.926 | 0.955 | 0.967 | 1.008 | 1.004 | 1.008 | 0.994 |

`sqrt_approx_dyn` stays within noise at the AGM's sizes (0.999 at g = 524, 0.988 at 1044, 0.999 at 2084, 1.000
at 4164, 1.005 at 8324, 0.990 at 16644). A cutoff of 1366 costs up to 14% (`sqrt_approx_dyn` at g = 2084),
and no wrapping up to 21%. The cutoff is left at the placeholder, as a tuning pass should also look at
`bnm1_size`: it took the smallest 7-smooth transform, and a root of exactly 2^k limbs needs m ≥ 2^k + 1, so
its transform was mixed-radix (m = 525 for k = 512). rustfft on NEON has cost up to twice as much per point at
such sizes, which may be part of the mid-size losses, but it wasn't timed separately here. These tables were
measured at those sizes. `bnm1_size` now follows `find_fft_size`'s stricter set (m = 576 for k = 512), and the
wrapped remainder's cutoffs haven't been remeasured with it.

The balanced row above also shows `div_rem_dyn` taking NR too early on the M1: dispatch switches from BZ at
d = 778, where NR takes 1.34× at 1024 limbs, 1.17× at 1536 and 1.14× at 2048. Its constants were measured
on an i7-8750H.

## Values before the tuner (2026-11-08)

The sqrt cutoffs moved from `src/utils/mod.rs` into the tuning schema (section `sqrt` of
`src/tuning/schema.rs`, Phase 1 of `docs/tuning_system.md`) with their values unchanged; code reads
them as `k::sqrt::<knob>()`.

| Knob | Was | Value | Notes from `mod.rs` |
|---|---|---|---|
| `sqrt.leaf` | `ZIMMERMAN_SQRT_LEAF_CUTOFF` | 11 | the recursive leaf, shared by dyn and static |
| `sqrt.dyn_rem`, `sqrt.static_rem` | `DYN_`/`STATIC_SQRT_REM_ZIMMERMAN_CUTOFF` | 16 | top level, root and remainder |
| `sqrt.dyn_only`, `sqrt.static_only` | `DYN_`/`STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF` | 11 | top level, root only |
| `sqrt.dyn_approx`, `sqrt.static_approx` | `DYN_`/`STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF` | 11 | top level, approximate root |
| `sqrt.dyn_wrapped_rem` | `DYN_SQRT_WRAPPED_REM_CUTOFF` | 341 | "Dynamic Zimmermann levels (and sqrt_dyn's top level) with at least this many root limbs form their remainder X - s^2 from one square mod B^m - 1 and divide quotient-only, rather than take the division's remainder and subtract the low root's square. GUESS: every level whose wrapped square is on the FFT." It was `DYN_SQR_BNM1_FFT_CUTOFF + 1`, and is a `Follows` of mul's cutoff once mul is declared |
| `sqrt.dyn_rcp_carry` | `DYN_SQRT_RCP_CARRY_CUTOFF` | 256 | "Not a crossover: a reciprocal needs two NR stages to be read (about 680 root limbs at the cutoffs above), so smaller roots just skip its scratch acquisition." |

The random-knob test (`cargo test --lib --features tune`) found that every cutoff, the six top
levels as well as the leaf, must be at least 3: Zimmermann's split takes `lo = (s.len() - 1) / 2`
low root limbs from a full x, which is none for roots of 1 or 2 limbs, and `s_lo[lo - 1]` then
underflows (a stage of 2 limbs would also recurse on itself). Those are the knobs' valid minimums;
every larger value, up to `usize::MAX`, gives correct roots.

