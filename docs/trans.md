# exp, ln, theta, the AGM and the constant caches

`src/utils/trans.rs`, `theta_plan.rs` and `consts/`. Their knobs are sections `exp`, `ln`, `agm`,
`theta` and `consts`. The theta–AGM logarithm's mathematics is in `ln_agm.md`; the series
evaluators under exp and ln are in `series.md`. Numbers are the M1's unless noted.

## Conventions

- A fixed-point argument `x` with scale `m: i64` means X = 2^(64m)·x with x's **top** limb its
  integer part: m = 0 is X in [0, 2^64), m = −1 a full-precision X in [0, 1). m = −(x.len()) puts X
  below an output of x's length. Top zero limbs fold into m (in i128).
- Results are o·2^(64k) with o in [1, 2^64), `out`'s top limb the integer part, and k returned.
- Each public function is a thin wrapper over a `*_reduced` routine that takes its ln 2, division,
  core and high-product functions as `fn` pointers, with pooled scratch (dynamic) or `[u64; N]`
  arrays (static). Serial and parallel paths give bit-identical output: slots are filled
  independently and combined in a fixed order.
- A static entry whose N is too small returns `Err(CapacityError { min_n })` before reading any
  input and leaves `out` untouched. It is a limit of the stack buffers, not an arithmetic failure.
- `*_at` entries (doc-hidden) take a knob's value explicitly, for tests and the tuner.

## exp

`exp_dyn(x, m, out) -> Result<i64, ExpError>` writes o = e^X / 2^(64k). x = 0 gives 1.
`ExpError::Overflow` once k = n/64 would pass i64::MAX (returned early when X ≥ 23·2^64).
`exp_static::<N>` needs N ≥ out.len() + 3 (out.len() + 4 once ln 2's out.len() + 2 limbs pass the
1024-limb table, where it sums them through `ln2_static_unguarded`).

**Reduction** (`exp_reduced`): one exact division with remainder of ⌊X·2^(64F)⌋·2^g by
⌊ln 2·2^(64F + g)⌋, F = out.len() + 1, g = `EXP_RED_GUARD_BITS` = 58. The quotient n has at most 3
limbs, so the division is linear in F; the remainder shifted down by g is r in [0, ln 2), and
out = e^r << (n mod 64). r's error is up to n times ln 2's truncation error, so ln 2 needs about 7
bits past F limbs for n < 2^69: a whole guard limb is not enough, and guard bits rather than a
second limb keep the numerator in out.len() + 3 limbs. When X is within the error of a multiple of
64·ln 2 either neighbouring k is valid.

**Squares.** r is scaled to r·2^-j (exactly, by a shift), the core gives u = e^(r·2^-j) − 1 in a
frame of fs = out.len() + ⌈(j + 4)/64⌉ limbs, and `exp_reduced` squares back j times as
u ← 2u + u² through `hi_sqr_*`, never forming 1 + u. j is `exp.<dyn|static>_sqr_bits` (24) less r's
own leading zero bits (`exp_sqr`), and **0 when the core would sum its chunks in parallel**: the
squares run serially while the series they shorten would overlap. The best j hardly moves with
precision (about 24 from 1 to 16384 limbs), so it is a constant, not a multiple of log2 of the
precision. The static entry takes the squares only when N has room (N ≥ fs + 3).

**The core** `exp_dyn_entry(x, m: u64, out)` is bit-burst e^x − 1 for a fraction
x·2^(−64(m + x.len())) < ln 2; it returns the zero limbs above `out`. It takes x's top bit, then
chunks of its remaining bits, combined as (1 + a)(1 + b) − 1 (`exp_combine`). Each chunk spans
[s, ~2s) of its own offset s (`chunk_width`), except that the first is at least
`exp.<dyn|static>_chunk_min_bits` wide: near x's top the factorial, not the offset, sets the term
count, so narrow chunks cost as much as wide ones. `chunk_bits` extracts every chunk into one
scratch buffer up front. `exp_series::<V>` sums each series into its slot through
`series_eval_dyn`/`series_eval_static` (the private `Eval` trait): an `Engel::<Exp>` context when
the bits are a single 1 (the top bit always is), otherwise `Hyper::<Exp>`. The top bit's series
runs on its own, straight into the result.

**Parallel chunks**, decided once per call: `chunks >= 2 && chunks * v >=
exp.dyn_chunk_par_work` on a pool of two or more threads, v the result's working limbs. Precision
alone predicts badly, since leading zero limbs shrink per-chunk work. The static twin
(`exp_static_entry::<N>`, N ≥ out.len() + 3) gives each chunk a scoped thread with a
`static_split_stack(N)` stack through `static_jobs` (at most 64 chunks) while the caller sums the
top bit's series; only that path touches the heap. Static matches dynamic exactly on almost every
input and within 1 in the last limb otherwise.

**Open:** the chunk width wants a serial and a parallel value. Run serially 64 is best at every
size; with the chunks on the pool (from about 256 limbs) 8–32 is, and 64 loses 19–38%. The preset's
32 (dynamic) and 64 (static) are a compromise: `exp_dyn` at 300–500 limbs is 7–12% slower than it
could be, `exp_static` at 500–1040 about 20%. Proposed, not built.

## ln

`ln_dyn(x, m, out) -> Result<(i64, bool), LnError>` writes o with |ln X| = o·2^(64k) and returns
(k, negative). X = 1 zeroes `out` and returns (0, false); X = 0 is `LnError::Zero`.
`ln_static::<N>` needs N ≥ out.len() + 4.

**Reduction** (`ln_reduced`, sharing `LnNorm` with the AGM path): X is normalized to
u = X·2^-e in [1, 2), except that X in [1/2, 1) stays u = X with e = 0, since −ln 2 + ln 2X would
cancel without bound just below 1. δ = |u − 1| comes straight from x's limbs (`Delta`). It then
takes j = ⌊A·log2(64·out.len())⌋ square roots (`ln.<dyn|static>_sqrt_a`), less the bits by which u
already sits near 1; each is one `sqrt_approx_*` of 4y·B shifted right 33 bits. The core runs on
δ_y, and |ln X| = |e·ln 2 ± 2^j·L| with ln 2 at o + 2 limbs.

**The core** `ln_dyn_entry(x, m: u64, neg, out)` is |ln(1 ± δ)| = 2 atanh(z), z = δ/(2 ± δ), for
δ < 1 (δ ≤ 1/2 when neg). Chunks split z, not z², because atanh has an addition rule:
atanh(z) = atanh(a) + atanh((z − a)/(1 − za)). `ln_chain` keeps the residual as a fraction n/d
updated by n −= a·d, d −= a·n (two short × long high products), so nothing is ever divided out. A
chunk's bits come from one short division of n's and d's top limbs (`chunk_quotient`), d rounded up
so a ≤ z. The first chunk is the residual's top bit, a `Bbp::<ATanh>` at shift 2k; later chunks
span [s, ~2s) (`ln.<dyn|static>_chunk_min_bits`) as a Hyper `ATanh` over A² (BBP when A = 1). Each
chunk adds 2a exactly and 2a·ATanh(a²) (`ln_chunk`, `ln_chunk_sum`, `mul_shr`). The chain stops once
the residual's top bit passes ⌈64w/3⌉ + 2, where its cubic term is below the frame, and the tail
2n/d is one `ratio_to_fraction_*`. The dynamic core runs the chain serially, then every series
serially or on rayon (`ln.dyn_chunk_par_work`) beside the tail. `ln_static_entry::<N>` needs
N ≥ out.len() + 3 and starts threads as the chain finds chunks.

**Series or AGM** (`ln_takes_agm` → `agm_wins`): the theta–AGM logarithm from
`ln.<dyn|static>_agm_limbs` output limbs on one thread, and on two or more threads only below
`ln.<dyn|static>_agm_par_until`. The series' chunks and their trees split across threads while the
AGM is one serial chain, so on a pool the series takes over again once it is large enough to
split; that is why the parallel knob is a bound, not a second cutoff. On the M1: 32 and 129. The
schema's defaults (1024 and 0) mean the AGM from 1024 limbs on one thread and the series on every
pool. Above the cutoff `agm_wins` still keeps X near 1, or near any power of 2, on the series
longer (per M(P): the series about L²/S, L = log2(P/max(λ, j0, 1)); the AGM about
κ(1 + w/P)·log2(P + w)). `ln_static` takes the AGM only when N also has room for its frame.

Left unswept on purpose: `ln.*_chunk_min_bits` (2) and `ln.*_sqrt_a` (0.25). Their best values
differ by size, and another square-root scale also moves the series-against-AGM cutoff, so sweeping
it made the whole section flip from run to run.

## theta

`theta_dyn::<T>(x, shift, th2, th3) -> usize` and `theta_static::<N, T>` compute
θ2(q⁴) = 2Σ_{k≥0} q^((2k+1)²) and θ3(q⁴) = 1 + 2Σ_{k≥1} q^((2k)²) for q = x·2^-shift < 1/4 (x
trimmed and nonzero, l = shift − bits(x) ≥ 2, asserted). θ3 keeps k = 1..T and θ2 k = 0..T, T a
const generic up to `THETA_MAX_TERMS` = 16.

- th3 = θ3·2^(64(th3.len() − 1)), top limb the integer part 1. th2 = θ2·2^(64(th2.len() + z)), and
  z = ⌊(l − 2)/64⌋ is returned. Each output is at most 1 unit below the truncated sum.
- The powers follow a compile-time plan (below). Each mantissa is kept normalized with a bit
  exponent, and the powers of 2^-l only decide where a term lands (`ThetaSchedule::place`). One
  backward pass (`Theta::new`) gives each value the widest window among the terms it feeds; a value
  feeding only terms below the frame never runs.
- The gate picks the plan from the frame alone: `PARALLEL` from `theta.<dyn|static>_par_limbs`,
  else `SERIAL`, threaded when there are two or more workers. Threads never change the bits.
- Threaded `PARALLEL` is a dataflow with one join (`Flow`): ops dispatch in plan order once their
  operands are done and a slot is free, from a pool of `slots + workers − 1` slots (`SlotPool`, a
  small `unsafe impl Sync` table whose safety rests on that scheduling), under one mutex. Dynamic
  runs on `rayon::scope`, static on `thread::scope` with a condvar, catching a panicking op.
- `theta_static` needs N ≥ max(th2.len() + 2, th3.len() + 1) and T's `SERIAL` plan to fit
  `THETA_STATIC_SLOTS` = 8 (to T = 9; a compile-time assert); it runs `PARALLEL` only if that fits
  too (to T = 6). Dynamic and static agree within a unit.
- `theta_dyn_at`/`theta_static_at` take a `ThetaMode` (`Auto`, `Serial`, `Parallel { threads }`).

### Plans (`theta_plan.rs`)

`ThetaPlans::<T>::SERIAL` and `PARALLEL`, built by integer-only `const fn`s (`MAX_VALUES` = 64).
Value 0 is q; op v makes value v = a·b from two earlier values, a square when a == b; `term` marks
q^(n²). `parallel_plan` is an earliest-finish DP (squares cost 2, products 3). `serial_plan` is a
greedy for least work, each op weighted by the window of the first term it feeds; `SERIAL` takes
whichever of the two does less work. `color` assigns in-order slots, `slots` is their live peak,
`width` the most ops sharing a DP start time. At T = 3 `SERIAL` is 2 4 8 9 16 25 18 36 13 49, about
0.59× the old recurrence's work. Evaluating a plan costs about 0.1 s of compile time at T = 3 and
6 s for all of T = 0..16.

ln's theta term count is a const generic, so `ln_dyn`/`ln_static` call `ln_agm_*_at::<T>` through
`knob_generic!`: a const argument in a normal build, a `match` over the knob's valid values in a
tune build (0..=16 dynamic, 0..=9 static).

## The AGM

`agm_dyn(a, b, out)` and `agm_static::<N>` run on integers at one scale (AGM(λa, λb) = λ·AGM(a, b)):
a, b and out share the caller's frame, shorter inputs zero-extended. They write the limit within a
few units and stop on their own. A zero input gives 0; a = b gives a.

- `agm_core` keeps the running a in `out`, with b and four scratch buffers at out.len() + 2 limbs.
  Each step takes only the top h = min(t, ⌈t/2⌉ + 2) limbs of the trimmed pair's product through
  `hi_mul_*`, larger operand first so swapped inputs make the same products; the geometric mean is
  one `sqrt_approx_*` of that window, read as padded. A perfect-square product can come out a unit
  low. An odd t is read as padded by one more limb and the root shifted back 32 bits, exactly.
- A unit in b_k < a_k costs up to about a_k/b_k units of the result, so the frame must give the
  smaller input its full precision. Its zero limbs cost little: trimming skips them.
- Before each step `agm_tail_ready` bounds the term count of the closed-form tail from the bit
  lengths of a, b and a − b; once it is at most `agm.<dyn|static>_tail_terms` (10 and 4),
  `agm_tail` finishes: AGM(m + D, m − D) = m·(1 − Σ g_i u^i), u = (D/m)², with `AGM_TAIL`'s 19
  falling dyadics g_i (the inverse of K's series, so the next term bounds the rest). The correction
  C = D·ε·H(u) needs no product by m; ε = D/m is one `ratio_to_fraction_*`, u = ε² by `sqr_shr`,
  Horner through `mul_shr` with frames shrinking 2t bits a level. Out is within 2 units of the limit
  of the pair the tail starts from.
- The gap doubles each step, so ending j steps early takes about 2^(j−1)…2^j − 1 terms. Dynamic's
  best is j = 3 at every size, static's j = 2 (its ε is a reciprocal and a product).
- `agm_static` needs N ≥ out.len() + 2. `agm_dyn_at`/`agm_static_at` take the cap (clamped to 19)
  and return (roots, tail terms).

## The theta–AGM logarithm

`ln_agm_dyn_at::<T>(x, m, out)` and `ln_agm_static_at::<N, T>` (doc-hidden) have `ln_dyn`'s
contract. q = X·2^-shift, so no reciprocal of X: ln X = c·ln 2 − π/(4·AGM(θ2², θ3²)),
c = e + 1 + l in i128. `agm_frame` sets the frames (`ln_agm.md` derives them): f = o + 1 fraction
limbs, more when X in [1/2, 2) makes the two terms cancel; l = max(2, ⌈(p + 3)/(4(T + 1)²)⌉);
g = f + 2 + ⌈l/64⌉ limbs for the AGM. The first AGM step is free (b1 = θ2θ3,
a1 = (θ2 + θ3)²/2 − b1). X = 2^e skips the AGM. The static entry needs N ≥ g + 2, which depends on
X, so it is checked after reading X but before touching `out`. Theta terms: `ln.dyn_agm_terms` = 5,
`ln.static_agm_terms` = 6.

## Constant caches (`consts/`)

`ln2_dyn` and `pi_dyn` copy requests of up to 1024 limbs from compile-time tables (`ln2.bin`,
`pi.bin`: the same as the static caches', no lock). Past that they share one growable cache each, a
`DynCache<T>` (the series nodes plus the constant with one guard limb) behind an `RwLock`, run by
`cached`: a hit copies the top limbs, a miss extends the nodes (`bin_split_extend`, with
`last = false` trees) and keeps the longer result. Nothing under the lock can panic halfway, so a
poisoned lock is used as is.

- **ln 2** = 2/3·S(6,1) + 1/24·S(6,5) + 1/72·S(2,1) + 1/384·S(1,1), four base-64 BBP series
  (`Ln2S<A, B>`, q = A·n + B, `Shift(6)`). Each weight is 2^-k/d with d = 3 or 9 (`LN2_W`):
  `ln2_share` scales a copy of the top limbs of the node's Q by d, folds 2^-k into the finalize
  shift, and the four shares are added, each extended and finalized in its own task.
- **π** from one Chudnovsky Hyper series (`Pi`; fits u128 below about 9.6M terms).
  `pi_denominator` truncates Q·2^(15t) and P to one limb past the output and forms
  T = 13591409·Q′ − P′; then one `sqrt_approx_dyn` (`pi_root`), one product and one `div_dyn`.
  `pi_static_sum` goes through the same denominator, then V = Q′/T′ by `ratio_to_fraction_static`
  and π = 426880·√10005·V as one `hi_mul_static`. It matches `pi_dyn` exactly.
- Static: `ln2_static::<N>` and `pi_static::<N>` copy the tables up to 1024 limbs and sum from
  scratch past them, needing N ≥ out.len() + 3. `ln2_static_unguarded` skips the guard limb
  (out.len() + 2, possibly a few ulps low) for `exp_static`.
- Term counts come from the series' tail bounds through `range`, a limb past the output
  (`ln2_end`, `pi_end`). Below the tables the series paths match them exactly
  (`test_const_series_match_tables`).
- **Parallel builds.** Dynamic: when new·(cached² + HALF²) ≥ CUTOFF·HALF² in output limbs
  (`cache_par`, saturating; `consts.dyn_ln2_par`/`_half`, `consts.dyn_pi_par`/`_half`): extending
  also merges into the cached nodes and finalizes at the full length, so a few new limbs on a large
  cache still pay. Past the tables every ln 2 build is parallel, and π's from about 7 new limbs.
  ln 2 runs its four shares with nested `join2`. π spawns its series into the pool with
  `rayon::in_place_scope` while **the caller builds the root, so the root's FFT plans stay in the
  caller's thread-local cache** (on a worker they cost small warm extensions 4–20%). Static:
  `static_jobs` on scoped threads from `consts.static_ln2_par`/`consts.static_pi_par` limbs.
- Regenerate the tables with `cargo run --release --example gen_ln2 [-- <limbs>]` and `gen_pi`.
  They write the dynamic series' output one limb past the table, less that limb.

## Facts to know when measuring

- **Every exp workload takes `check_items::EXP_FRACTION`** (m = −1). m = −(n) computes e^X = 1
  exactly in a microsecond: until 2026-11-11 every exp item, sweep and gate in the tuner timed that,
  so any `exp_dyn`/`exp_static` figure taken through them before that date is of a no-op. When a
  whole-call workload is suspiciously fast, or every candidate of a sweep ties, check that the call
  computes something.
- On one thread `ln_dyn` takes 1.2–2.2× `exp_dyn`'s time (about 0.96× at 256 limbs).
- Parallel chunks lose to one thread below about 128 limbs even with the squares on both sides.

## Findings not acted on

From a profile below 1024 limbs (i7, 2026-10-24) and of `agm_dyn` (M1, 2026-11-02). Items since
done (ADX rows, the Knuth rewrite, a Karatsuba square tier, no rayon entry on one thread, the
wrapped remainder, the reciprocal carry, the closed-form tail) are left out.

- **Stronger argument reduction for ln.** After its square roots ln's series run on an argument of
  about 2^-5 against exp's 2^-24; on a 2^-24 argument the ln core is 1.8–2.5× cheaper at 16–512
  limbs. (a) *exp-based*: take y₀ ≈ ln u from f64, compute 1 + δ = u·e^(−y₀) with one short exp and
  a multiply, then ln u = y₀ + ln(1 + δ): about 1.5× priced from existing functions. (b)
  *Multi-prime reduction* (Johansson 2022): cache ln p for a few small primes; replaces ln's roots
  and exp's 24 squares with one division. Both change last bits within 1 ulp, so they need the
  oracle suites.
- **A direct path at 1–4 limbs.** A plain fixed-point Taylor loop beat the exp core 3–7.7× at one
  limb: the bit-burst machinery costs more than it saves there.
- **Fewer allocations at tiny sizes.** 13–18 heap allocations a call at one limb (the dynamic
  nodes' `Vec`s), 11–15% of the time; nodes from pooled scratch, or a node-free single-leaf path.
- **ln's chain at 1–4 limbs** sends 2–4-limb chunk quotients through the general `div_dyn`.
- **The AGM.** Each iteration costs a product and a root, the root 0.8–2.5× the product; divisions
  inside the root are 41–64% of the AGM. Left: a narrower frame after the early steps (about 1.6%);
  squares in place of products with squares carried between steps (about 0.9×, untested);
  rectangular splitting of the tail's Horner pass for j = 4; folding the tail into the logarithm's
  own last step (about 2%). The chain is serial: threads only help inside each multiplication.

## Known issues

- `ln_static_at::<16600>` and `ln_agm_static_at::<16600, 6>` at 16384 output limbs abort with a
  stack overflow in a spawned worker (sized by `static_split_stack(N)`), even from a 1 GiB caller
  stack. Fine at N = 4200. Not investigated.
- `ln_dyn_at` (the series path) trips `ratio_to_fraction_dyn`'s "finalize expects S < 1" debug
  assertion through `ln_tail` for X = 1 + δ with a multi-bit δ far below 1 (`x = [5, 0 × (o + 2),
  1]`, m = 0, from about 150 output limbs). `ln_dyn` sends such X to the series. Static unchecked.

The reports these came from (`trans_profile.md`, `agm_profile.md`, the exp and constant-cache
sections of `bin_split_cutoff_tuning.md`) are in git history at `fee4f90`.
