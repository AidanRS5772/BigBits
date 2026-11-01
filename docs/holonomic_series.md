 ▐▛███▛█   Claude Code v2.1.314
▝▜██████▀  Opus 5.5 · Claude Pro
 ▝▝   ▝▝   ~/Programing/Rust/BigBits


❯ Im starting to think about how I would impliment and efficent series          
  repersentation within my current framwork for computing sin, and cos in joint 
  manner. Im thinking of using a holonomic frame work that is not only capable  
  of computing sin and cos but also other solutions to nth order differential   
  equations. This kind of thing is mentioned in the literature but Im unsure on 
  what the actual broad level implimentation looks like? Can you outline it in  
  broad strokes and how I could plug something like that into the existing      
  series code?                                                                  

⏺ I'll look at the series module's actual shape before sketching how a holonomic
  layer would fit.

  Searched for 2 patterns, listed 1 directory, ran 3 shell commands (ctrl+o to expand)

⏺ The core idea

  A holonomic (D-finite) function satisfies a linear ODE with polynomial
  coefficients. That makes its Taylor coefficients satisfy a linear recurrence
  with polynomial coefficients. Write that recurrence as a first-order matrix
  recurrence:

  U_{n+1} = A(n)·U_n / q(n)      A(n): r×r matrix with polynomial-in-n entries, 
  q(n): scalar

  Evaluation then becomes the binary splitting you already have, with matrices
  in place of scalars. A node over [a, b) holds:

  M = A(b−1)···A(a)                                  r×r
  Q = q(b−1)···q(a)                                  scalar
  T = Σ_n q(b−1)···q(n) · E·A(n−1)···A(a)            k×r, one row per output 
  functional E
  merge:  M = M_R·M_L    Q = Q_L·Q_R    T = Q_R·T_L + T_R·M_L        value = 
  T·U_a / Q

  DynNodeHyper is already the r = 1 case. Your R is M, your P is T, and the
  merge at src/utils/series.rs:1604 (P = lp·rq·2^(shift(b−m)) + rp·lr) is
  exactly the T rule. So a holonomic node is "Hyper with R and P promoted to
  matrices." last = true dropping R becomes dropping M.

  sin/cos: the structure matters more than the generality

  For y'' = −y in system form, the step is A(n) = θ·J with J = [[0,−1],[1,0]]
  and q(n) = n+1. Powers of J only take the values ±I and ±J. So M is never
  really a dense matrix: it's θ^m times a phase i^m. In other words, you're
  summing e^{iθ} over the Gaussian integers Z[i]:

  - Q and R stay real, as they are now. R's phase is just m mod 4, which costs
    nothing.
  - P becomes complex (two limb vectors). A merge then costs 6 products.
    Separate sin and cos trees in θ² would cost 8, because both trees end up
    with the same Q and R at the root and compute them twice.
  - Finalize can do one reciprocal of Q and two high products, the way
    ratio_to_fraction_static already works, instead of two divisions.
  - Signs come from the phase, so SIGNED isn't needed.
  - Rect and Horner carry over. Rect's power table (iθ)^j = i^j·θ^j is real, so
    rect_powers and low_zeros work unchanged; only the accumulator becomes a
    pair. A Horner step is swap + negate + the existing mul_frac.

  A dense generic 2×2 node would do about 15 products per merge here. So the
  framework has to let a series declare its structure, or sin/cos pays roughly
  2.5× for the generality.

  Baseline to beat first: compute sin through the existing Hyper in θ², then cos
  − 1 = −s²/(1 + √(1 − s²)) via sqrt_approx_*. That's 4 products per merge plus
  one O(M(n)) square root at the end, against a tree that costs O(M(n) log n).
  It may well beat the joint tree at high precision, and it gives you an oracle.

  The rest of sin/cos is exp with a different group

  - Reduction mod π/2. Use pi_dyn and the same exact division with guard bits
    that exp uses with ln 2. The error again scales with the quotient n. One
    difference: X has no overflow bound here, so the guard bits must grow with
    bits(n) instead of staying fixed at 58.
  - Halving. Keep the minus-identity form, u = cos − 1 and v = sin, and double
    with u ← 2u² + 4u, v ← 2v + 2uv. That's two products per doubling against
    exp_square's one, so the best j will be smaller.
  - Bit-burst. The schedule is identical, because the series is exp's series at
    iθ: same term counts, same bit sizes. exp_chunks, chunk_bits, Exp's tail
    bound and probably DYN_EXP_CHUNK_MIN_BITS all transfer.
  - Combine. This is rotation composition, the analog of exp_combine
    (src/utils/trans.rs:319):
  u = u_a + u_b + u_a·u_b − v_a·v_b
  v = v_a + v_b + u_a·v_b + v_a·u_b      (4 high products, or 3 Gauss-style)
    The leading-zero skipping from exp_combine applies too.

  Because the chunks combine through the addition formula, sin/cos never needs
  analytic continuation.

  Where it actually becomes holonomic

  1. Constant-coefficient ODEs (exp, sin/cos, sinh/cosh, any e^{Az}). The step
     matrix is constant and the solution is translation-invariant, so bit-burst
     is just a product of transition matrices. Coefficients are small integers
     and fit your u128 traits.
  2. Polynomial coefficients, expanded at a small rational point (usually 0).
     A(n) has small integer polynomial entries, still u128 per term. What this
     buys you is joint bases (Ai/Bi), a function together with its derivative,
     and recurrences of order > 1. Many classical functions (J_ν, erf, Ai) are
     already hypergeometric at 0, so your Hyper node handles them one at a time.
  3. Analytic continuation, for functions without an addition theorem. This is
     the actual Chudnovsky / van der Hoeven algorithm. To step from z_k to
     z_{k+1}:
     - Taylor-shift the ODE's polynomial coefficients to z_k and derive a new
       recurrence. Its coefficients involve z_k, which is a big rational, so
       A(n) has bigint polynomial coefficients.
     - Carry the whole jet (y, y′, …, y^(r−1)), which means T is the full r×r
       transition matrix.

     This level breaks the current trait shape (p(n) → u128): leaves stop being
     products of machine words. Singular points (Frobenius expansions, log
     terms) are a further layer on top.

  Tail bounds keep the tail_bits(n, s) interface. Use exp's bound for level 1
  and majorant series (Mezzarobba–Salvy) in general. Any valid lower bound
  works.

  How it plugs into series.rs

  Reusable unchanged:
  - bin_split_tree_at (series.rs:940), join2, SplitCutoffs: they only call leaf,
    merge, limbs and range. finalize is the only single-output assumption in
    the tree code.
  - min_end / range, the shift and decay bookkeeping, est_limbs,
    ratio_to_fraction_*.
  - The Work/Costs model, since Family::products is structural.

  New pieces:

  1. Make the node generic over the algebra M and T live in, not over a fixed
     dense matrix:
  trait Algebra {
      type M;  // Scalar: one vec (today's R); Phase: vec + i^m (sin/cos); 
  Mat<R>: R×R signed vecs
      type T;  // Scalar: one vec (today's P); Gauss: two vecs;          
  Mat<K,R>: K×R
      fn mm(r: &Self::M, l: &Self::M) -> Self::M;   // M_R·M_L
      fn tq(t: &Self::T, q: &[u64]) -> Self::T;     // Q_R·T_L (+ the 2^shift 
  offset, as dyn_merge does)
      fn tm(t: &Self::T, m: &Self::M) -> Self::T;   // T_R·M_L
      fn add(a: Self::T, b: Self::T) -> Self::T;
  }
     A HoloSeries kind trait then gives q(n), the small-integer A(n), START and
     tail_bits. Each entry's power of x is fixed by the recurrence's shape and
     declared by the trait, the same way HyperSeries keeps x out of r(n).
     Sin/cos is the generic node at Phase/Gauss, and Hyper itself is the Scalar
     case. That gives you a free cross-check against the existing node.
  2. DynNodeHolo<S> / StaticNodeHolo<N, S> implementing Node + SeriesNode. Three
     things change from today:
     - Signed entries. Today only P carries a sign. Matrix entries are signed,
       so merges need signed adds (cmp_buf + sub_buf).
     - One bit exponent per matrix. Static nodes should normalize each matrix to
       its largest entry, since products propagate error relative to the norm.
       A product entry is a sum of r signed products that can cancel, so the
       test_static_nodes_keep_bits analysis has to be redone.
     - More stack. A node holds r² + kr + 1 N-arrays instead of 3, so
       static_split_stack has to scale with that.
  3. Multi-output finalize. Add a sibling trait, JointNode: Node with
     finalize_joint(ctx, zeros: [usize; K], outs: [&mut [u64]; K]) -> [bool; K],
     plus a Joint::<S> context and series_eval_joint_{dyn,static}.
     Node::finalize can keep writing output 0. All outputs share one absolute
     frame, which is what the combine needs: the term count comes from one
     absolute bound, and per-output zeros only trim leading limbs (cos − 1 has
     about twice sin's leading zeros).
  4. Cost model. Add a Family per algebra with its list of products. The
     SERIES_NS_* constants are per unit, so they should transfer; refit if the
     residuals say otherwise.

  Suggested order

  1. Build the sin + sqrt baseline in trans.rs. It gives you the oracle and the
     number to beat.
  2. Add the Algebra parameter with Phase/Gauss, the joint finalize, and
     sin_cos_dyn (reduction, doubling, bit-burst). Benchmark it against step 1.
  3. Add the dense Mat<R> algebra. Check it against step 2 by running sin/cos
     through dense matrices, then test it on an Airy-style recurrence.
  4. Do level 3 (runtime ODE shifting and recurrences with bigint coefficients)
     only once you want functions without addition theorems.

  Reading

  - van der Hoeven, Fast evaluation of holonomic functions (TCS, 1999)
  - Chudnovsky & Chudnovsky's bit-burst papers (around 1988–90)
  - Mezzarobba, NumGfun (ISSAC 2010) and his ore_algebra paper on rigorous
    D-finite evaluation (2016)
  - Mezzarobba & Salvy, Effective bounds for P-recursive sequences
  - Brent & Zimmermann, Modern Computer Arithmetic, ch. 4: the binary-splitting
    / bit-burst sections, which include a sin/cos variant

✻ Sautéed for 5m 0s · done 5:13 PM