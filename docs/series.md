# Series evaluation (`src/utils/series.rs`)

General evaluators for BBP, Engel and Hyper series: reverse Horner, rectangular splitting (rect)
and binary splitting, with a dispatcher over them. They are for users' series and for more of this
library later, not helpers of exp and ln. How the knobs are measured is in `tuning_system.md`; the
record of how the equations were found is in `tuning_implementation.md`. Numbers are the M1's.

## Series kinds and contexts

Each kind has a narrow trait (`BBPSeries`, `EngelSeries`, `HyperSeries`: u64 coefficients) and a
wide one (`WideBBPSeries`, …: `p`/`q`/`r` return u128 and take a u128 index). Every narrow series is
a wide one through a blanket impl, and the nodes are bounded on the wide traits. A type implements
one of the two; with both in scope, call `<T as BBPSeries>::q`. Each kind trait requires:

- `const START`, the first term's index. Term n is weighted 2^(−k(n − START + 1)), with
  x^(n − START + 1) for Hyper.
- `tail_bits(n, s)`, a lower bound on −log2|R_n| for the tail from term n at decay s. 0 is always
  valid; any valid bound is correct, a monotone one gives the fewest terms.

A typed context picks the kind: `Bbp::<S>::new(shift)`, `Engel::<S>::new(shift)`,
`Hyper::<S>::new(x, shift)?`. A context of the wrong kind for its series is a compile error.
`Hyper::new` returns `HyperCtxError` unless x is trimmed and nonzero with x·2^-shift ≤ 1/2. A series
with two kinds (`Exp`: Engel and Hyper; `ATanh`: BBP and Hyper) is picked by the context. The
sealed `Series` trait maps a context to its dynamic and static nodes, derives the decay s (the
shift, or shift − ⌈log2 x⌉ for Hyper) and gives `zeros()` (= `tail_bits(START) / 64`) and
`range(bits)` (the fewest terms whose bound reaches `bits`, by doubling then bisection; panics past
2^40 terms).

## Entries

- `series_eval_dyn(ctx, out) -> (bool, usize)` and
  `series_eval_static::<N, C>(ctx, out) -> Result<(bool, usize), CapacityError>` sum to
  64·(out.len() + zeros) bits, write out = S·2^(64(out.len() + zeros)) within 2 units, and return
  (S < 0, zeros). Callers keep their own guard limbs and size `out` with `zeros()` first. The static
  entry returns `CapacityError` before touching `out` unless out.len() + 2 ≤ N and a Hyper x fits N.
- `series_eval_dyn_at`/`series_eval_static_at` (doc-hidden) force an algorithm.
- `series_eval_range::<N>(ctx, a, b, zeros, out) -> bool` (doc-hidden, node level) sums an explicit
  range, for tests. `series_eval_at` forces an algorithm there and panics on an ineligible one;
  `rect_dyn_m`/`rect_static_m` force rect's block length.
- `bin_split`/`bin_split_tree` build a node; `bin_split_tree_at` takes explicit `SplitCutoffs`. The
  constant caches call these directly, as π needs its node's P and Q.

Horner and rect agree with the exact sum within one unit of out's last limb, not bit for bit with
binary splitting.

## Binary splitting

Six nodes: `DynNode{BBP,Engel,Hyper}` (exact `Vec`) and `StaticNode{BBP,Engel,Hyper}<N>`
(`[u64; N]`, low bits dropped). A leaf whose values all fit u64 runs the narrow kernel through the
`Narrow<S>` view; any other leaf runs a wide kernel, one term a step (`mul_prim2`, two-limb
carries).

**The tree rule.** A tree splits `[a, b)` while it has more than `Node::leaf_terms()` terms and
`Node::limbs` (an O(1) estimate of the node's largest part, capped at N for static nodes) is at
least `Node::leaf_limbs()`. Those return the node kind's knobs (section `bin_split`:
`<dyn|static>_<bbp|engel|hyper>_leaf_limbs`, `<dyn|static>_hyper_leaf_terms`); `SplitCutoffs::of`
reads them once per tree. Any cutoff is correct.

- *Why limbs.* A leaf step costs O(current length) and a merge its products: both follow the node's
  size, and bits per term vary by orders of magnitude between series. A term cutoff tuned on ln 2
  lost up to 5× elsewhere.
- *Why Hyper also keeps a term floor.* With a wide x each leaf step is a full multiplication, so a
  leaf of t steps costs about t²·M(s)/2 against a merge's M(t·s/2): the crossover sits at a few terms
  whatever the width.
- Sparse BBP and Engel (shift ≥ 64) prefer smaller leaves in limbs than dense ones, as their merges
  are cheap unbalanced products. Static leaves don't grow with N.

**Parallel splits.** The same estimate against `bin_split.dyn_par`, on a pool of two or more
threads. A split started off the pool (the caller's own thread, which has to wake the pool) also
needs `bin_split.dyn_par_work` of limbs × terms (`SplitCutoffs::splits`): a 2-term tree at 1024
limbs took 15.8 µs split against 2.1 serial. On a pool worker the limb gate alone decides. A
merge's two products run in parallel from `bin_split.dyn_merge_par` limbs.

**Merges and finalize.** `Family::products` lists a merge's products (Hyper: P = lp·rq·2^(…) +
rp·lr, with Q and R). The Hyper x-leaf multiplies by x before each term, so a node sums
x^(n−a+1); a node built with `last = true` drops the R a later merge needs, so `bin_split_extend`
grows cached nodes with `last = false`. `Node::finalize(ctx, zeros, out)` writes
S·2^(64(out.len() + zeros)), skipping `zeros` known-zero leading limbs. Dynamic:
`ratio_to_fraction_dyn`, one `div_dyn` sized by the output. Static: `ratio_to_fraction_static`, a
(len + 2)-limb reciprocal and a high product, requiring out.len() + 2 ≤ N. Each static node's
`ratio(ctx)` gives (P, Q, shift).

**Static parts** are m·2^e with m in N limbs and e a bit exponent: exact while they fit, then their
top 64N bits. BBP keeps separate P and Q exponents. A merge takes P's exponent from its operands'
bit lengths before any product (`p_exponent`) and computes each product only down to it
(`product_at`/`hi_mul_from`); Q and R are normalized to their top bit (`hi_mul_norm`). A node loses
at most about 11 bits below 64N (`test_static_nodes_keep_bits`), so static and dynamic finalize
agree within 1 ulp up to finalize's N − 2 bound. A static leaf whose exact products wouldn't fit N
(a conservative bit-length bound, `prod_fits`) splits at its midpoint, so leaves stay exact and only
merges drop limbs.

**Static splits need about 640 bytes of stack per limb of N** (`static_split_stack`, four times
that in debug builds), more than a rayon worker has at large N, so static parallelism uses scoped
threads (`static_jobs`), never the pool.

## Horner and rect

Both run in the factored-shift frame F = S·2^(64w), w = len + zeros + 1 guard limb. The per-term
2^-k is never applied: term n's p(n) lands at bit 64w − k(n − a + 1), an O(1) add at an offset, so
the accumulator (len + 3 limbs narrow, len + 4 wide) is F from the first step, deep terms only
touch the limbs they reach, and there is no final shift.

**Horner** steps from n = b − 1 down:

| Kind | Step |
|---|---|
| BBP | `acc += p·2^d / q` |
| Engel | `acc = (acc + p·2^d) / q` |
| Hyper, x = 1 | `acc = (r·acc + p·2^d) / q` |
| Hyper, x ≠ 1 | `acc = x'(r·acc + p·2^d) / q`, x' = x·2^-bits(x) ∈ [1/2, 1) |

Consecutive terms whose ∏q fits the group budget share one division (`group`, `fold_group`:
`div_prim` to 64 bits, `div_prim2` to 128): a group of m terms is one multiplication by R = ∏r and
one division by Q = ∏q. The budget is 64 bits per accumulator limb past the single-term layout,
capped at 128 (`group_budget`); a group takes at least one term and at most `GROUP_MAX` = 32, and
stops early if p·Q would pass u128. Dynamic Horner always allocates the wide accumulator for the
128-bit budget; static does when N has room (`horner_room`). For x ≠ 1, x is normalized once
(`normalize`) and its products are `hi_mul` windows (`mul_frac`), with k − bits(x) as the frame's
shift. Grouping made Horner 3× faster at 16 limbs and 6–12× from 64 to 1024.

**Rect** (Hyper with x of two or more limbs, narrow coefficients only) precomputes x'^1..x'^m at
len + 3 limbs and records each power's zero low limbs (`low_zeros`), which its products and term
adds skip. Per block of m terms: one high product by a power, then Horner's scalar steps adding
p(n)·x'^(i+1). Static rect keeps at most `RECT_MAX_M` = 16 powers on its stack.

**Eligibility** (`SeriesNode::eligible`, one O(t) pass → `Eligible { horner, rect }`; binary
splitting always can): both need Hyper's r(n) ≤ q(n) and k ≥ bits(x). Static nodes need
out.len() + 3 ≤ N for narrow kernels and out.len() + 4 ≤ N for wide Horner. `const RECT` is true
only for the Hyper nodes.

## Dynamic dispatch

`SeriesNode::dispatch` for a dynamic node, in order:

1. **The small-series table** `series.small` (`SeriesNode::small`). Row = kind × 16 + the frame's
   octave (`small_frame`, 16 buckets to 65535 limbs); change points run over the terms; a cell is
   Horner, rect, binary splitting or "the equations" (`SMALL_PICKS`). A kind (`small_class`,
   `SMALL_KINDS` = 8) is the family and x's width: BBP, Engel, and Hyper with no x, an x to a limb,
   of 2 limbs, 3–16, 17–32, more. Coefficients past 64 bits have no class. The pick is checked
   against the range's own eligibility and falls to the equations otherwise, so any valid table is
   correct; rect's block length under a table pick is still the model's. The default table hands
   everything to the equations. 232 change points in the preset.
2. **Rect's guard** (`Shape::guards` at explicit `SeriesGuards`, `Shape::decide_at`): rect outright,
   at the balance block length, where rho reaches `series.rect_rho` (32) and x has at least
   `series.rect_x_limbs` (16) limbs. There is no guard on beta.
3. **The least fitted prediction** (`costs` → a `Costs` of each eligible algorithm's ns; `pick`).

The two ratios of a shape: **rho** (`Shape::rho`) is x's bits over the bits a term fixes (the
frame's bits over the terms); **beta** (`Shape::beta`) is the exact fraction's width (the tree's P)
over the frame. Rect against Horner follows rho; binary splitting follows beta. A bit-burst chunk
has rho 0.5–1 and beta 1–2 by construction, the triple point where the runner-up is a median 1.21×
from the winner: exp's and ln's chunks are the hardest place to dispatch and the least informative
to fit on.

### The equations

One equation an algorithm (`SeriesEq`): the algorithm's own `Parts`, read off its kernel
(`Shape::horner_parts`, `rect_parts`, `bin_split_parts`), times fitted prices. `Parts` has thirteen
columns (`Parts::N`). Multi-limb products are priced in ns by mul's equations (`MulEq::mul_ns`); a
high product by `SeriesEnv::hi`, which follows `hi_mul_dyn` (a `mul_prim` pass for a one-limb
operand, `HiEq` up to `mul.dyn_partial_mul`, the guarded product above); finalize's division by
`dyn_div_best`. A full product with a one- or two-limb side is a `mul_prim`/`mul_prim2` pass, not
schoolbook's equation, which is fitted from 10 limbs.

| Algorithm | Prices (`series.<alg>_ns_*`) |
|---|---|
| Horner (8) | product, div1, div2, mul1, light, term, call, step |
| Rect (5) | product, div1, add, term, one |
| Binary splitting (12) | product, division, div1, div2, mul1, mul2, add, light, term, call, alloc, step |

plus `series.horner_pass2` (a two-limb pass is priced at this ratio times `mul1`; measured on
`mul_prim2` against `mul_prim`, 3.46) and `series.bin_split_par_gain` (0.3 on the 8-thread pool).

- **Horner.** Per group a division by Q over the frame at the group's shallowest term
  (`Shape::groups`). For a q of one size the groups are whole (`whole_group`), taken from the deep
  end. For a q that grows they grow towards the shallow end: the range's bits of ∏q (`Shape::qs`,
  q as c·n^d through two Gauss points, summed by Stirling) over what a group holds give their number
  (`group_count`). A product by R runs over the sum as the group below left it (nothing for the
  deepest group). With x, a high product a term, summed in closed form where every frame is
  `hi_mul_buf`'s (`Shape::school_products`). Fixed costs are three: a term, a group (`step`), and
  the calls a term's own step makes besides (`call`: three for a step by a one-limb x, one
  otherwise).
- **Rect.** m − 1 powers, each a high product of the whole accumulator by x', a copy and a scan;
  per block but the deepest a high product by the block's power at that power's significant limbs,
  ceil(j·bits(x)/64) for power j (`power_step`, `power_limbs`); per group a division and a product
  by R at the group's own frame; per term its weight times its power over the lesser of the power's
  limbs and the limbs under the term's bit, once plus `spill` of a second pass. Rect takes Horner's
  prices for the kernels they share and fits its own for the rest. Its groups end at its blocks
  (`group_of`). Its block length is the fastest predicted of three candidates (`Shape::rect_len`:
  the a/m + b·m balance, 1, and the length whose power fills the accumulator).
- **Binary splitting.** The parts walk the tree `bin_split_tree_at` builds, level by level: each
  merge's products at its children's part sizes, each leaf's passes over parts that grow, and
  finalize (division's equations, or `div1`/`div2` a quotient limb by a Q of one or two limbs:
  Knuth's equation reads 27% high at two). An x-leaf's terms are `step`s. A tree wide enough to
  split has each level's work divided by 1 + gain·(threads − 1) and pays one wake
  (`bin_split.dyn_wake_ns`) off the pool.
- **One model of a group's size** for Horner and rect: `Shape::group_room`, `whole_group`,
  `group_count`.

Held out in the preset: Horner 3.4%, rect 2.7%, binary splitting 3.6%. On 960 unseen series the
equations' picks lose 0.07% and 0.02% on the mean (worst 12.4%), the entry 2.9% and 2.3%.

**What a decision costs.** 40–60 ns for a BBP or Engel range, 55–230 for a Hyper one up to 64
limbs, more where rect is predicted or binary splitting's tree is walked (200–570 ns at a high
beta). It is kept down by not predicting what can't matter and by closed forms; `sum_sampled` is
left for the guarded tier, rect's blocks and leaf products past schoolbook. The table exists
because this was 22% of a series under 1 µs.

## Static nodes

Approximate, like every static decision. `SeriesNode::TABLED` is false: no table, no guard.

- `horner_first` takes Horner without the model at up to `series.horner_limbs` (28) output limbs,
  for a Hyper range with a one-limb x or a BBP/Engel range of at most `series.horner_terms` (12)
  terms. Those two knobs stay unswept: the decision depends on precision, terms and x together.
- Otherwise the old model: `Work`, each algorithm's primitive counts (division limbs, scalar-pass
  limbs × scalar limbs², product work, terms, kernel calls, N-array limbs zero-filled or moved),
  times the six shared `series.ns_*` prices (`SeriesModel`). Product work is `mul::mul_work`,
  schoolbook limb products along `mul_static`'s tiers, each tier scaled at the crossover that
  selects it so it never jumps. Rect's block length is `rect_best`; the first candidate stands when
  no prediction is a number. On held-out workloads (2026-10-30) its selections were 1.02× the best
  and 1.25× off in predicted time.

## How the series section is tuned

- **On a synthetic family, on no caller's series** (`tuner/synth.rs`): `Geo<LQ>` (q and r of LQ
  bits: size and no convergence), `Den<LQ>` (an LQ-bit q over r = 1), `Fact` (q(n) = n), and for BBP
  `Odd` and `Big<LQ>`. Never put `Exp` or `ATanh` back into the section's fits, sweeps or gate
  items.
- Shapes are scattered over (frame, terms) and, with an x, (frame, terms, x's bits); frames 4–1024
  limbs. Each algorithm is timed where it competes and some way past (`RECT_FROM`, `HORNER_TO`).
- The table is measured on the kind's **members** (`member_cell`, 72 synthetic series), each summed
  by every algorithm that can, forced, with the decision timed beside them (`decide_probe`). A cell
  is the pick that costs its members least together, and **reading the equations is one of the
  picks**. A row ends where the decision is within the share of every member.
- The section fits twice from one set of timings: first on the series whose own decision is within
  the share, then, after the table, on what the table leaves (`tune_late`, `Rows::Table`), where the
  guard is also swept, on series of 50 µs and more.
- **A layout change** to the table means raising the section's `derive` number in
  `tuning/schema.rs` (5 now) and `cargo tune tune series --tables`, then `cargo tune series` to
  score it.
- `cargo tune series --timings FILE` keeps the timings between runs, so a change to an equation is
  read against the same measurements. `BIGBITS_TUNE_VERBOSE=1` writes `series_parts.tsv`,
  `series_workloads.tsv` and (with `--tables`) `series_cells.tsv` under `target/tuning/`.

Not covered by the tuner: coefficients past 64 bits (the wide kernels), frames past 1024 limbs.

## Known issues and open items

- Static Hyper binary splitting can panic in `hi_mul_shifted` when an R product is zero (r(n) = 0
  with enough terms to merge).
- Rect picks among three candidate block lengths; the best timed length is 8% faster on average.
- A lower bound from binary splitting's own equation would spare the tree's walk where it can't
  win. It is a shortcut computed from the equations, so proposed and not built.
- Binary splitting's `div1` and `div2` rest on a few series a half at the quick and normal levels.
- A holonomic (matrix) node for joint sin/cos and other D-finite functions is sketched in
  `holonomic_series.md`.

The reports behind the tree rule and the static model (`bin_split_cutoff_tuning.md`,
`series_cutoff_tuning.md`) are in git history at `fee4f90`; their harnesses are at `faa9971`.
