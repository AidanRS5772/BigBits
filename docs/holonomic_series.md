# Holonomic series: a design sketch (2026-10-30, nothing built)

How a holonomic (D-finite) layer could sit on `series.rs`, for sin and cos jointly and for other
solutions of linear ODEs with polynomial coefficients. A sketch to start from, not a plan that was
approved.

## The idea

A holonomic function's Taylor coefficients satisfy a linear recurrence with polynomial coefficients.
As a first-order matrix recurrence:

    U_{n+1} = A(n)·U_n / q(n)        A(n): r×r, polynomial entries; q(n): scalar

Binary splitting carries over with matrices in place of scalars. A node over [a, b) holds

    M = A(b−1)···A(a)                               r×r
    Q = q(b−1)···q(a)                               scalar
    T = Σ_n q(b−1)···q(n) · E·A(n−1)···A(a)         k×r, one row per output functional E

    merge:  M = M_R·M_L    Q = Q_L·Q_R    T = Q_R·T_L + T_R·M_L        value = T·U_a / Q

`DynNodeHyper` is already the r = 1 case: its R is M, its P is T, and its merge is the T rule. A
holonomic node is Hyper with R and P promoted to matrices, and `last = true` dropping R becomes
dropping M.

## sin and cos: structure matters more than generality

For y'' = −y the step is A(n) = θ·J, J = [[0, −1], [1, 0]], q(n) = n + 1. Powers of J are ±I and ±J,
so M is never a dense matrix: it is θ^m times a phase i^m. The sum is e^(iθ) over Z[i]:

- Q and R stay real; R's phase is m mod 4.
- P becomes complex (two limb vectors). A merge costs 6 products. Separate sin and cos trees in θ²
  cost 8, as both compute the same Q and R.
- Finalize is one reciprocal of Q and two high products (`ratio_to_fraction_static`'s shape).
- Signs come from the phase. Rect's power table (iθ)^j = i^j·θ^j is real, so only the accumulator
  becomes a pair; a Horner step is a swap, a negate and the existing `mul_frac`.

A dense generic 2×2 node would do about 15 products a merge, so the framework has to let a series
declare its structure, or sin/cos pays about 2.5× for the generality.

**The baseline to beat first:** sin through the existing Hyper node in θ², then
cos − 1 = −s²/(1 + √(1 − s²)) through `sqrt_approx_*`. Four products a merge and one O(M(n)) root.
It may beat the joint tree at high precision, and it is an oracle either way.

The rest is exp with a different group:

- **Reduction** mod π/2 with `pi_dyn` and exp's exact division with guard bits. X has no overflow
  bound here, so the guard bits must grow with the quotient's bits, not stay at 58.
- **Halving** in the minus-identity form, u = cos − 1 and v = sin: u ← 2u² + 4u, v ← 2v + 2uv. Two
  products a doubling against exp's one, so the best count is smaller.
- **Bit-burst** is identical: the series is exp's at iθ, so the chunk schedule, `Exp`'s tail bound
  and probably the chunk-width knob transfer.
- **Combine** is rotation composition, 4 high products (3 Gauss-style):
  u = u_a + u_b + u_a·u_b − v_a·v_b, v = v_a + v_b + u_a·v_b + v_a·u_b.

The chunks combine through the addition formula, so sin/cos never needs analytic continuation.

## Three levels

1. **Constant-coefficient ODEs** (exp, sin/cos, sinh/cosh, e^(Az)). The step matrix is constant, so
   bit-burst is a product of transition matrices. Coefficients fit the u128 traits.
2. **Polynomial coefficients at a small rational point.** A(n) has small integer polynomial entries.
   This buys joint bases (Ai/Bi), a function with its derivative, and recurrences of order > 1. Many
   classical functions are already hypergeometric at 0 and go through the Hyper node one at a time.
3. **Analytic continuation**, for functions without an addition theorem (Chudnovsky, van der
   Hoeven): Taylor-shift the ODE to each z_k, which gives A(n) bigint coefficients, and carry the
   whole jet, so T is the full r×r transition matrix. This breaks the trait shape (`p(n) → u128`):
   leaves stop being products of machine words. Singular points are a further layer.

Tail bounds keep `tail_bits(n, s)`: exp's bound for level 1, majorant series (Mezzarobba–Salvy) in
general.

## How it plugs into `series.rs`

Reusable unchanged: `bin_split_tree_at`, `join2`, `SplitCutoffs` (they only call leaf, merge, limbs
and range; finalize is the tree code's one single-output assumption), `min_end`/`range`, the shift
and decay bookkeeping, `ratio_to_fraction_*`.

New:

1. **A node generic over the algebra** M and T live in, not a fixed dense matrix:

   ```rust
   trait Algebra {
       type M;  // Scalar: one vec (today's R); Phase: vec + i^m; Mat<R>: R×R signed vecs
       type T;  // Scalar: one vec (today's P); Gauss: two vecs; Mat<K, R>: K×R
       fn mm(r: &Self::M, l: &Self::M) -> Self::M;   // M_R·M_L
       fn tq(t: &Self::T, q: &[u64]) -> Self::T;     // Q_R·T_L, with the 2^shift offset
       fn tm(t: &Self::T, m: &Self::M) -> Self::T;   // T_R·M_L
       fn add(a: Self::T, b: Self::T) -> Self::T;
   }
   ```

   A kind trait gives q(n), the small-integer A(n), `START` and `tail_bits`. sin/cos is the generic
   node at Phase/Gauss, and Hyper is the Scalar case, a free cross-check against the existing node.
2. **`DynNodeHolo<S>` / `StaticNodeHolo<N, S>`.** Three changes from today: signed entries (merges
   need signed adds); one bit exponent per matrix (static nodes normalize to the largest entry, and
   since a product entry is a sum of r signed products that can cancel, the
   `test_static_nodes_keep_bits` analysis has to be redone); more stack (r² + kr + 1 N-arrays, so
   `static_split_stack` scales with it).
3. **Multi-output finalize**: a `JointNode: Node` with
   `finalize_joint(ctx, zeros: [usize; K], outs: [&mut [u64]; K]) -> [bool; K]`, a `Joint::<S>`
   context and `series_eval_joint_{dyn,static}`. All outputs share one absolute frame, which the
   combine needs; per-output zeros only trim leading limbs (cos − 1 has about twice sin's).
4. **Dispatch.** A `Family` per algebra with its list of merge products feeds binary splitting's
   parts as it does now; the dynamic equations' prices are per kernel, so they should transfer, and
   the series section's fit will say if they don't.

## Suggested order

1. The sin + sqrt baseline in `trans.rs`: the oracle and the number to beat.
2. The `Algebra` parameter with Phase/Gauss, the joint finalize, and `sin_cos_dyn` (reduction,
   doubling, bit-burst). Benchmark against step 1.
3. The dense `Mat<R>` algebra, checked by running sin/cos through it, then an Airy-style recurrence.
4. Level 3 only once functions without addition theorems are wanted.

## Reading

- van der Hoeven, *Fast evaluation of holonomic functions* (TCS, 1999)
- Chudnovsky & Chudnovsky's bit-burst papers (about 1988–90)
- Mezzarobba, *NumGfun* (ISSAC 2010) and his `ore_algebra` paper on rigorous D-finite evaluation
- Mezzarobba & Salvy, *Effective bounds for P-recursive sequences*
- Brent & Zimmermann, *Modern Computer Arithmetic*, ch. 4 (binary splitting, bit-burst, a sin/cos
  variant)
