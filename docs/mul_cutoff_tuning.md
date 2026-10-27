# Multiplication cutoff tuning

Date: 2026-10-27. Machine: i7-8750H (Coffee Lake, 6 cores / 12 threads): one
pinned core, except the static NTT crossovers, which ran with rayon's full pool.

The row-wise schoolbook kernels (commits d1236ec, f924899) made `mul_buf`,
`sqr_buf`, `hi_mul_buf`, `hi_sqr_buf` and the column kernel behind
`mid_mul_buf` much faster, so every crossover in `mul.rs` from schoolbook to
something else, and every crossover above Karatsuba (whose leaves are
schoolbook), was retuned. This document records the method, the measurements
and the results.

## Crossovers

| Family | Constant | Lower side → upper side |
|---|---|---|
| `prim` | fixed `s == 1`/`s == 2` tiers | Prim / Prim2 arm → `mul_buf` (report only) |
| `chunk` | `CHUNKING_KARATSUBA_CUTOFF` | `mul_buf` → `chunking_karatsuba` (s ≤ half) |
| `kara` | `KARATSUBA_CUTOFF` | `mul_buf` → one Karatsuba level (s > half) |
| `kfft` | `FFT_KARATSUBA_CUTOFF`, `FFT_CHUNKING_KARATSUBA_CUTOFF` | `karatsuba_entry_dyn` → `fft_entry` |
| `kntt` | `NTT_KARATSUBA_CUTOFF`, `NTT_CHUNKING_KARATSUBA_CUTOFF` | `karatsuba_entry_static` → `ntt_entry_static` |
| `ksqr` | `KARATSUBA_SQR_CUTOFF` | `sqr_buf` → one Karatsuba square level |
| `sqrfft` | `FFT_SQR_CUTOFF` | `sqr_buf` / `karatsuba_sqr_entry_dyn` → `fft_sqr_entry` |
| `sqrntt` | `STATIC_NTT_SQR_CUTOFF` | `karatsuba_sqr_entry_static` → `ntt_sqr_entry_static` |
| `hi` | `DYN_`/`STATIC_PARTIAL_MUL_CUTOFF` (was `PARTIAL_MUL_CUTOFF`) | `hi_mul_buf` → guarded truncated product |
| `hisqr` | `DYN_`/`STATIC_PARTIAL_SQR_CUTOFF` (was `PARTIAL_SQR_CUTOFF`) | `hi_sqr_buf` → guarded truncated square |
| `hirows` | `HI_MUL_ROW_CUTOFF`, `HI_SQR_ROW_CUTOFF` | columns → rows (re-check) |
| `mid` | `FFT_MID_CUTOFF` | `mid_mul_buf` → `fft_mid_mul` |
| `midntt` | `NTT_MID_CUTOFF` | `mid_mul_buf` → `ntt_mid_mul_static` |

Not tuned: `FFT_16BIT_CUTOFF` is the f64 precision bound of the 16-bit FFT,
not a speed choice; the `NTT_PAR_*` cutoffs decide threading, which the
schoolbook rewrite doesn't affect and a single core can't measure;
`HI_MUL_GUARD_LIMBS` is an accuracy setting.

## Method

`benches/mul_cutoffs.rs` times one family per process. It forces each side
through the library's own entries (`karatsuba_core`, `chunking_karatsuba`,
`karatsuba_sqr_entry_dyn`, the `hi_*_guarded_*` paths, `hi_*_buf_cols` /
`hi_*_buf_by_rows`, and the existing transform entries). Per shape:

- up to 32 random trimmed inputs at random offsets within a cache line, fewer
  for large shapes (one pass of the lower side takes about 200 µs);
- batches calibrated to at least 100 µs, then 60 rounds (30 for the largest
  families), each timing both sides back to back in random order;
- the process's statistic is the median over rounds of ln(t_upper / t_lower);
- both sides' outputs are compared afterwards (exactly, or within the
  documented error of the guarded high products and transform middle
  products).

Static entries run at N = the smallest 5-smooth bucket (about 8% apart) that
holds the call's output, Karatsuba scratch and NTT size, so their `[0; N]`
buffers cost what a fitting N would. Forced static Karatsuba levels zero a
scratch of that N and then run the same kernels, as
`karatsuba_entry_static` does.

`benches/mul_cutoffs.py` builds the bench under three code layouts (`plain`,
`jcc` = `-C llvm-args=-x86-branches-within-32B-boundaries`, `jcc_align` =
that plus `-C llvm-args=-align-loops=64`; fat-LTO release), runs separate
processes pinned with `taskset -c 2` (which also gives rayon a one-thread
pool), and treats each process as one observation. Per shape it runs an
exact one-sided Wilcoxon signed-rank test in each direction and reports a
Hodges–Lehmann ratio with its 95% CI. A shape's verdict is U (upper side
significantly faster, p < 0.05), L (lower side significantly faster) or =.

How much replication a family gets depends on how long its calls take. The
full treatment above is for calls of a few hundred ns to a few µs (Stage A:
`prim`, `hirows`, `ksqr`, `chunk`, `kara`, `mid`), whose variance between
processes and layouts can be as large as the effects. Larger calls are run as
a few processes and compared by their averages (`analyze --simple`: each
shape's side is the mean of its per-process log ratios). That applies to
Stages B and C, and to `midntt` and `kfft`/`sqrfft`, which had already run 24
processes; they are reported by both methods where both exist, and the two
agree to within one grid step.

Cutoff rules:

- **1-D constants**: the smallest size from which the upper side is
  significant at every larger measured size (intersection–union, no
  multiplicity correction), as for `HI_MUL_ROW_CUTOFF` before.
- **Parametric boundaries** (`is_school`'s c, `is_karatsuba`'s reg and chunk
  ratios): each shape has a flip value, the parameter at which the formula
  moves it to the other side. The strict rule takes the largest upper region
  in which every measured shape is U, which is the largest flip among non-U
  shapes. None of the three formulas fits its measured boundary shape by
  shape (the parabola has the wrong shape, and FFT/NTT times step with the
  transform size), so the strict rule ends up set by one tie or one shape
  just past a transform step, and gives up large wins elsewhere. The values
  used are therefore the regret minimum instead: the mean, over the measured
  shapes, of the log time lost against each shape's faster side. Ties are
  broken toward the smallest value whose bootstrap interval overlaps the
  minimum's. The analysis reports the strict value, the shapes it would leave
  on the slower side, and the regret curve.
- Constants shared by dyn and static code apply the rule over both paths'
  shapes together.

Threads: everything ran pinned to one core except the families whose upper
side is the static NTT (`kntt`, `sqrntt`, `midntt`). `ntt_entry_static`,
`ntt_sqr_entry_static` and `ntt_mid_mul_static` run their three prime
convolutions with `rayon::join` above 512 output limbs, so those families ran
unpinned with rayon's full pool (`run --core all`), as a library user gets
by default.

Karatsuba's leaves use the constants being tuned. Stage A therefore measures
one Karatsuba level over schoolbook leaves (`build --set KARATSUBA_CUTOFF=1e9
--set CHUNKING_KARATSUBA_CUTOFF=usize::MAX --set
KARATSUBA_SQR_CUTOFF=usize::MAX`): near the true crossover the leaves are
below it, so they are schoolbook in the final configuration too. That forces
same-level products to schoolbook as well (`chunking_karatsuba`'s last chunk),
and large leaves that would recurse, so round 2 rebuilds with round 1's
`KARATSUBA_CUTOFF` and measures `chunk` and `kara` again; it is the fixed
point. Each later stage is measured on a build with the earlier stages'
results applied.

## Results

Ratios are upper / lower times: the Hodges–Lehmann estimate over processes
with its 95% CI in brackets for the fully tested families, the mean over
processes otherwise. Runs were on a laptop with a desktop session open: the
load average before each process was 1.5–2.2 (one of which is the measured
process itself). Pairing within each round and the per-process medians absorb
most of that.

| Constant | Old | New | Decided by |
|---|---|---|---|
| `CHUNKING_KARATSUBA_CUTOFF` | 22 | 63 | full test, every ratio |
| `KARATSUBA_CUTOFF` | 17.0 | 52.0 | regret minimum (parabola misfits) |
| `KARATSUBA_SQR_CUTOFF` | 14 | 71 | full test |
| `FFT_KARATSUBA_CUTOFF` | 1.92 | 1.96 | regret minimum |
| `FFT_CHUNKING_KARATSUBA_CUTOFF` | 1.74 | 1.88 | regret minimum |
| `NTT_KARATSUBA_CUTOFF` | 3.78 | 5.7 | regret minimum, all threads |
| `NTT_CHUNKING_KARATSUBA_CUTOFF` | 3.58 | 5.9 | regret minimum, all threads |
| `FFT_SQR_CUTOFF` | 33 (school → FFT) | 340 (Karatsuba → FFT) | averages; new dyn Karatsuba tier |
| `STATIC_NTT_SQR_CUTOFF` | 1014 | 3884 | averages, all threads |
| `DYN_PARTIAL_MUL_CUTOFF` | 90 (shared) | 312 | averages |
| `STATIC_PARTIAL_MUL_CUTOFF` | 90 (shared) | 408 | averages |
| `DYN_PARTIAL_SQR_CUTOFF` | 128 (shared) | 464 | averages |
| `STATIC_PARTIAL_SQR_CUTOFF` | 128 (shared) | 640 | averages |
| `HI_MUL_ROW_CUTOFF` | 30 | 24 | full test |
| `HI_SQR_ROW_CUTOFF` | 117 | 90 | full test |
| `FFT_MID_CUTOFF` | 90 | 125 | full test |
| `NTT_MID_CUTOFF` | 200 | 672 | averages, all threads (997 on one core) |

### Stage A (schoolbook leaves, 24 processes per family)

**`HI_MUL_ROW_CUTOFF` 30 → 24, `HI_SQR_ROW_CUTOFF` 117 → 90.** The re-check
after the row wrappers lost their bounds checks. Products: columns win to 22,
23–24 tie, rows win from 25 (0.9943 [0.9911, 0.9980]) at every size to 256.
Squares: rows are significant at 85–89, tie at 90 (0.9941 [0.9872, 1.0019])
and win from 91 to 256.

**`KARATSUBA_SQR_CUTOFF` 14 → 71.** One Karatsuba square level over
schoolbook leaves against `sqr_buf`: schoolbook wins to 67 (dyn) / 68
(static); dyn ties at 69 and static at 71 (0.9993 [0.9966, 1.0020]); both win
at every size from 72 to 192 (0.989 at 72, 0.978 at 76). With the old cutoff
the leaves themselves recursed, and one level lost 1.4–2.1× everywhere from
14 to 80.

**`FFT_MID_CUTOFF` 90 → 125.** `fft_mid_mul` is significant at 106–111 and
115–121, but its transform size steps up at 122–124 (1.0750 [1.0706, 1.0841]
at 122), and from 125 it wins at every size to 320 (0.8931 at 125).

**`NTT_MID_CUTOFF` on one core: 997.** The single-core static NTT middle
product loses to `mid_mul_buf` to 924 (8.4× at 64, 1.6× at 512), wins at 948,
loses at 971–996 (1.2197 [1.2011, 1.2241] at 996) and wins at every size from
1020 (0.8068) to 3024. Its NTT runs on rayon there, so the value used comes
from the all-threads run in Stage B.

**`CHUNKING_KARATSUBA_CUTOFF` 22 → 63 and `KARATSUBA_CUTOFF` 17 → 52.**
Round 1 forced every leaf to schoolbook. That understated Karatsuba wherever a
same-level product or a large leaf would really run Karatsuba:
`chunking_karatsuba`'s last chunk (s × [s, 2s)) is dispatched like any
product, and leaves past the crossover recurse. Round 1 had chunking never
consistently ahead at l = 2s, and the parabola's rule at c = 89.9. Round 2
rebuilt with c = 52.36 (round 1's regret minimum) and chunking leaves still
schoolbook, and is the fixed point: c = 52 changes no leaf near either
boundary.

- Chunking: schoolbook wins to s = 58–62 and chunking from 61–64 at every
  ratio (l = 2s: 62; 4s: 63; 16s: 61 dyn, 64 static), so the
  l-independent constant holds. The joint rule gives 63.
- `is_school`'s parabola doesn't fit the measured boundary. Squares are
  significantly faster with Karatsuba from n = 44 (c = 42). The smallest s at
  which every larger s is faster is about 0.7 l for l = 48–108, then just
  above half (s ≈ 63–66, the chunking crossover) for l ≥ 111:

  | l | 48 | 60 | 72 | 84 | 96 | 108 | 120 | 144 | 168 |
  |---|---|---|---|---|---|---|---|---|---|
  | measured s* | 44 | 52 | 54 | 60 | 66 | 72 | 66 | 75 | 87 |
  | parabola, c = 52 (school while s ≤) | 50 | 56 | 59 | 59 | 56 | 54 | 60 | 72 | 84 |

  The strict rule (every Karatsuba-side shape significant) needs c = 62.04,
  set by a single 108×69 shape (dyn 0.9996, a tie; static 1.0029). That
  keeps squares to 63 on schoolbook, 9% slower at 62×62 and 12% at 78×66, and
  doubles the mean time lost over the measured shapes (0.45%). The mean time
  lost is flat from c = 50 to 55 (0.253–0.261%) and rises on both sides, so c
  = 52, the smallest value within the noise of the minimum, is used. It
  sends l ≈ 96–123 with s just above half to Karatsuba, where Karatsuba is
  0–2% slower.
- **Proposed refit:** school iff s < max(T, ⌈0.7 l⌉) and s < S, with T ≈ 44
  and S ≈ 66 for s > half (S matching the chunking cutoff). That follows the
  measured s* within one grid step everywhere above.

**`prim` (report only).** `mul_buf` beats the `s == 1` Prim arm from l = 15
(0.9735 at 16, 0.7054 at 128, 0.6231 at 256) and the `s == 2` Prim2 arm from
l = 7 (0.9411 at 8, 0.6301 at 256); the Prim arms win below that (1.23 at
3×1, 1.16 at 3×2). `mul_prim`/`mul_prim2` are older single-limb kernels that
the ADX rows now beat; the fixed tiers in `dyn_dispatch`, `static_dispatch`
and `kdispatch` could move to `mul_buf` past those lengths, or `mul_prim`
could become a row kernel. Left unchanged: a dispatch change, not a constant.

### Stage B (Stage A applied)

`kfft` and `sqrfft` ran 24 processes on one core; `midntt` 24 processes and
`kntt`/`sqrntt` 3 processes (one per layout), all with rayon's full pool.

**`FFT_KARATSUBA_CUTOFF` 1.92 → 1.96, `FFT_CHUNKING_KARATSUBA_CUTOFF` 1.74 →
1.88.** `karatsuba_entry_dyn` against `fft_entry`. Squares are faster with the
FFT from cost ratio 2.04 (about 250 limbs), except just past FFT size steps
(307×307 at 2.22: 1.106). The strict rule would need 2.37 (regular, set by
391×215 at 1.0125) and 2.08 (chunking, set by 492×123 at 1.154, a step).
Mean time lost (regular): 0.48% at 1.92, 0.19% at 1.96, minimum 0.11% at
2.04, 0.14% at 2.08, rising to 1.3% at 2.40; 1.96 is the smallest within the
minimum's bootstrap interval. Chunking: 1.09% at 1.72, 0.33% at 1.88, minimum
0.31% at 1.92, 0.44% at 1.96. In limbs, squares now switch at about 230
instead of 220.

**Dyn Karatsuba square tier; `FFT_SQR_CUTOFF` 33 → 340 (now Karatsuba → FFT).**
`sqr_dyn` went from schoolbook straight to FFT. Against `fft_sqr_entry`,
schoolbook lasts only to 208 (FFT 0.9767 at 212), but `karatsuba_sqr_entry_dyn`
(one level, leaves at `KARATSUBA_SQR_CUTOFF`) is faster to 340, except just
past FFT steps at 288 and 316–336: FFT / Karatsuba is 2.09 at 100, 1.59 at 128,
1.32 at 200, 1.04 at 256, 1.08 at 340, 0.998 at 344, 0.84 at 400 and 0.50 at
1024. Since Karatsuba beats schoolbook from 72 (Stage A), `dyn_sqr_dispatch` is
now School (≤ 71) → Karatsuba (≤ 340) → FFT (≤ 2^16) → NTT. The full test
gives 344 (a tie at 344).

**`NTT_MID_CUTOFF` 200 → 672 (all threads).** `ntt_mid_mul_static` loses to
`mid_mul_buf` to 623 (1.16 at 511), wins at 638 (0.973), loses at 654–671
(1.215 at 654, past a transform step), and wins at every size from 687
(0.974; 0.80 at 759, 0.55 at 1020, 0.20 at 2482) to 3024. The full test agrees
(688: a tie at 687).

**`NTT_KARATSUBA_CUTOFF` 3.78 → 5.7, `NTT_CHUNKING_KARATSUBA_CUTOFF` 3.58 →
5.9 (all threads).** `karatsuba_entry_static` against `ntt_entry_static`.
NTT / Karatsuba on squares: 1.39 at 1583, 1.23 at 1886, 1.09 at 2119, 1.04 at
2381, 0.93 at 2524, 0.85 at 2836, 1.02 at 3006 (a step), 0.80 at 3186, 0.66 at
4264. So the static NTT now takes over at about 2500 limbs instead of 1000.
On one core it would be much later: the pilot had the NTT 1.4–1.8× slower to
3700 limbs, with Karatsuba's old leaves. Mean time lost: 0.02% at 5.7
(regular; 0.32% at 5.6, 0.16% at 5.9), 0.22% at 5.9 (chunking; 0.32% at 5.6).

**`STATIC_NTT_SQR_CUTOFF` 1014 → 3884 (all threads).** `ntt_sqr_entry_static`
against `karatsuba_sqr_entry_static`: 3.7× at 505, 2.2× at 1024, 1.3× at 2243,
then near 1 with transform steps (0.998 at 3070, 1.141 at 3320, 0.887 at
3453, 1.050 at 3884) and faster at every size from 4040 (0.901; 0.73 at
4915).

### Stage C (Stages A and B applied; one process per layout, one core)

The guarded path takes the full product of the operands' top `out.len()` limbs
(plus two guard limbs' linear terms), so it rides on the new full-product
dispatch: an FFT on the dyn path at these sizes, Karatsuba on the static one.
The two paths' crossovers are far apart, so the old shared `PARTIAL_*` pair is
now `DYN_`/`STATIC_` pairs. Shapes: windows k = 32–2048, operands k and 2k
wide (dyn), and for static either room for the whole guarded product (N ≥ 2k +
4) or operands that fill N (its split path).

| Constant | Old | New | Guarded / buf near the cutoff |
|---|---|---|---|
| `DYN_PARTIAL_MUL_CUTOFF` | 90 | 312 | 1.10 at 248; 0.92 at 256–296 but 1.04–1.09 at 304–312 (FFT step); 0.78 at 320, 0.52 at 512, 0.16 at 2048 |
| `STATIC_PARTIAL_MUL_CUTOFF` | 90 | 408 | 1.05 at 320, 1.00 at 384, 0.998 at 392, 0.985 at 408 (1.035 with operands filling N), 0.93 at 440, 0.88 at 512 |
| `DYN_PARTIAL_SQR_CUTOFF` | 128 | 464 | 1.18 at 312, 0.99 at 384, 1.05 at 408, 0.91 at 440, 0.94–1.00 at 464, 0.91–0.97 at 472, 0.75 at 512 |
| `STATIC_PARTIAL_SQR_CUTOFF` | 128 | 640 | 1.09 at 512, 1.00 at 640, 0.995 at 672, 0.94 at 768, 0.88 at 1024 |

(Ratios are dyn operands one window wide, or static with room, unless noted.)
The guarded static square stays close to `hi_sqr_buf` up to about 700 limbs,
because `sqr_static` is Karatsuba there, while the dyn one has the FFT.

## End to end: old constants (HEAD) against the new ones

Two copies of the library in one process (HEAD's `src/` and the retuned working
tree), each public entry timed against its twin in alternating, randomly ordered
rounds. One process per code layout, unpinned with rayon's full pool. New / old
time, geometric mean over the sizes in each band (the dispatch size: n for
squares, the window for high products, the short operand for middle products and
division):

| Entry | 1–16 | 17–64 | 65–256 | 257–1024 | 1025–4096 | 4097+ |
|---|---|---|---|---|---|---|
| `mul_dyn` n×n | 1.015 | **0.781** | **0.877** | 1.006 | 1.005 | 1.003 |
| `mul_dyn` 4s×s | 0.997 | 1.003 | **0.807** | 0.969 | 1.007 | 1.003 |
| `mul_dyn` 16s×s | | 1.007 | 0.998 | **0.821** | 0.968 | 1.004 |
| `mul_static` n×n | 1.012 | **0.768** | **0.867** | **0.884** | **0.716** | |
| `sqr_dyn` | 0.992 | **0.516** | **0.553** | 0.982 | 0.995 | 0.998 |
| `sqr_static` | 0.921 | **0.622** | **0.662** | **0.670** | **0.619** | |
| `hi_mul_dyn` (operands 2k) | 1.002 | 0.996 | **0.725** | 1.003 | 1.003 | |
| `hi_mul_static` | 1.018 | 0.994 | **0.699** | **0.874** | **0.687** | |
| `hi_sqr_dyn` | 0.972 | 0.986 | **0.678** | **0.911** | 0.998 | |
| `hi_sqr_static` | 0.976 | 0.988 | **0.676** | **0.619** | **0.546** | |
| `mid_mul_dyn` | 1.024 | 1.004 | 0.993 | 0.994 | 0.993 | |
| `mid_mul_static` | 1.032 | 0.994 | **0.739** | **0.691** | 0.996 | |
| `rcp_dyn` | 1.005 | 1.013 | **0.940** | **0.931** | 0.981 | |
| `div_dyn` (2k / k) | 0.996 | 0.997 | **0.908** | **0.951** | 0.983 | |
| `sqrt_approx_dyn` | 0.993 | 0.992 | **0.963** | **0.877** | **0.939** | |
| `exp_dyn` | 0.981 | 0.990 | 0.995 | 1.006 | 0.992 | |
| `ln_dyn` | 0.991 | **0.949** | **0.920** | **0.915** | **0.958** | |

Where dispatch didn't change (sizes 1–16, and the transform ranges), the two
copies differ by about ±1–3% (up to 1.10 for `mid_mul_static` at 5–7 limbs, about
40 ns): separately compiled code laid out differently, not the constants.
Single sizes where the new constants are slower:

- `mid_mul_dyn` at 111 (1.104) and 117 (1.061): the FFT middle product wins at
  106–121, but `FFT_MID_CUTOFF` follows the rule (faster at every larger size)
  past its 122–124 step. At 93–99 schoolbook is 9% faster than the old FFT.
  Starting the FFT at 106 instead would win those sizes and lose 3–7% at
  122–124.
- `sqr_dyn` at 320 (1.054) and 336 (1.045): the end of the Karatsuba tier, just
  before an FFT step. 304 is 0.764 and 352–408 are 0.995.
- `mul_dyn` at 224×224 (1.090): Karatsuba now, where the old cost ratio took a
  fast FFT size; the regret minimum accepts it (see Stage B).

## Follow-ups

- **`is_school` refit** (see Stage A): school iff s < max(T, ⌈0.7 l⌉) and s < S
  for s > half, with T ≈ 44 and S ≈ 66, follows the measured boundary.
- **`mul_prim`/`mul_prim2`**: `mul_buf` beats them from l = 15 (s = 1) and l = 7
  (s = 2), by up to 37%; the fixed `s == 1`/`s == 2` tiers are a dispatch change.
- **`ntt_mid_mul_*` exact-size bug** (fixed after this retune): when one prime's
  transform was exactly 2n − 1 long (2n − 1 = 81, 125, 243, 375, 405, 625, 1125,
  1215), `ntt_mid_accumulate_scaled` read two coefficients below the band onto
  which the wrap landed, and limb 1 was off by about 2^60. The transforms are now
  sized for 2n + 1 coefficients (`mid_ntt_size`). That grows the largest
  transform at about 1% of sizes from 672 to 4096 limbs, by at most 8%, so
  `NTT_MID_CUTOFF` stands.
- **`karatsuba_sqr_core` at n = 3 and 5**: its cross term is a limb wider than
  `out[half..]`. Dispatch never reaches it (the cutoff is 71); the entries document
  their n ≥ 6 floor.
- **Downstream cutoffs** tuned against the old multiplication:
  `DIV_KNUTH_CUTOFF`, `BZ_CUTOFF`, the BZ/NR cost models (`bz_2_1_cost` reads
  `dyn_dispatch`, whose tiers moved), `DYN_RCP_KNUTH_NR_CUTOFF` /
  `STATIC_RCP_KNUTH_NR_CUTOFF`, the sqrt cutoffs and the bin_split leaves. Their
  harnesses are `cargo bench-cutoffs` and `cargo bench-bin-split`.
- **`NTT_PAR_*`** decide the NTT's internal threading; they don't depend on the
  schoolbook kernels and weren't retuned.
