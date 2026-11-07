# Sharing FFT transforms in Newton division

An analysis of how far reusing FFT transforms could speed up Newton-Raphson division, and through it
the square root and the AGM. It covers the source as of 2026-11-06 (`70b5e3c`, branch `sqrt_opt`).
Nothing here is implemented. The transform counts and gains are estimates, worked out from the code
and from older profile shares, and no prototype has been timed. Only the section on large inputs
reports new measurements (M1, release).

The short version:

- **A transform can be shared within a Newton step, not across steps.** Between steps the operands
  change and the length doubles, so old spectra don't help.
- **Within a step the saving is small.** A reciprocal step goes from 6 transforms to 5, and a quotient
  refinement from 9 to 8.
- **That is worth about 5% on an AGM and 7% on a root.** It needs a new FFT interface, so it isn't the
  next thing to do. See the last section for what is.
- **It won't reach the published square-root counts** of about 1.3–1.8 M(n). The step that gets
  there, merging two products into one triple product, doesn't fit 16-bit digits in a double.

## The products in a Newton step

`nr_refine_rcp` takes a reciprocal x from precision p to 2p:

```
err = mid(d, x)      // d has 2p + 1 limbs, x has p + 1: the error band d x - B^k
cor = short(x, err)  // the correction x err
```

`nr_refine_quo` is the Karp–Markstein step at the end of a quotient, where x has h + 3 limbs:

```
hq  = short(n, x)        // the first estimate y
err = mid(d_top, y)      // d_top has 2h + 1 limbs, y has h + 1: the band d y - n
cor = short(x_norm, err) // the correction
```

`mid` is `mid_mul_dyn` and `short` is `hi_mul_dyn`. x appears in two products of each step, and that
repeat is the only reuse a step offers.

The digits each product needs, for a transform that holds it:

| Product | Digits today | Digits with the whole x |
|---|---:|---:|
| `mid(d, x)` | 8p + 12 | 8p + 12 |
| `short(x, err)` | 8p − 1 (`hi_mul_guarded_dyn`'s core of the top p limbs) | 8p + 7 |
| `short(n, x)` | 8h + 15 (core of the top h + 2 limbs) | 8h + 23 |
| `mid(d_top, y)` | 8h + 12 | 8h + 12 |
| `short(x_norm, err)` | about 8h + 15 | 8h + 15 |

So a shared x has to be the whole operand, not `hi_mul_guarded_dyn`'s truncated core. That also drops
the guard terms' linear passes from those products.

## What a product costs today

A product packs both operands into one complex array, a + i b, and runs a forward transform of length
N, a `recombine` pass, and an inverse of length N/2. In length-N/2 transforms, which is the unit from
here on, the forward is about 2 and the inverse 1: **3 per product**.

The packing is what blocks reuse. x only exists mixed with the other operand, so the next product has
to transform it again. `recombine` does separate the two spectra in passing, with
`A_k = (X_k + conj X_{N−k}) / 2` and `B_k = (X_k − conj X_{N−k}) / 2i`, but it multiplies them on the
spot and keeps neither.

At n = 2^19 a product's time is 84% transforms, 6.6% recombine, 5.1% accumulate, 2.4% rounding and
1.5% decomposing (`docs/fft_precision.md`). So transforms are most of a product, not all of it.

## Sharing within a step

The first product runs as it does now, and its recombine also writes out x's half spectrum. The second
needs only the transform it doesn't have:

```
// err = mid(d, x), as today, keeping x_hat from the separation
pack d + i x, forward N                 // 2
recombine, storing x_hat                // the extra writes
inverse N/2, read the band              // 1

// cor = short(x, err), reusing x_hat
pack err's digits, forward N/2          // 1 (err alone, as a real transform)
untangle e_hat
multiply x_hat * e_hat, tangle back
inverse N/2, read the top               // 1
```

That makes **5 instead of 6**. Separate real transforms of d and x would also cost 2 units, so packing
the first product isn't needed for this; keeping it just changes less code.

The real-transform path already exists: `sqr_decompose`, a length-N/2 forward and `sqr_recombine` do
this for squares. The second product needs that path with `A_k B_k` where a square has `A_k²`.

Each inverse still gets its own reading, and the fold check is unchanged, since it only needs the
operands and the rounded coefficients.

**The quotient refinement** gets the same treatment for `short(n, x)` and `short(x_norm, err)`, leaving
`mid(d_top, y)` alone: **8 instead of 9**. y comes out of the first product, and d_top is used only once.
x_norm is x before its shift by `sh`, so the two are only the same digits when `sh` is 0. Zimmermann
divisors have their top bit set, so that holds in the root; a general `div_dyn` with an unnormalized
divisor would have no reuse here.

### The transform length has to match

Both products must use one N. The first choice is the larger need, which is `mid`'s 8p + 12. Over
log-uniform p from 2^10 to 2^16:

| Pair | Sizes differ at |
|---|---:|
| 8p + 12 and 8p − 1 (today's `short`) | 0.37% of p |
| 8p + 12 and 8p + 7 (the whole x) | 0.15% of p |
| 8h + 23 and 8h + 15 (the quotient's pair) | 0.15% of h |

These are rare on average, but a precision just above a transform size hits them every time: at
p = 2^j, 8p − 1 fits 2^(j+3) and 8p + 12 doesn't. The next size up can be up to 1.2× longer, which costs
more than sharing saves, so such a step should take the ordinary path. The same goes for two more
cases:

- `nr_err_band` occasionally extends `err` past p + 1 limbs, up to `cap`, and the product may no longer
  fit N.
- Sharing only applies when both products are on the FFT tier. `mid_mul_dyn` uses schoolbook below
  `FFT_MID_CUTOFF`, `hi_mul_dyn` uses `hi_mul_buf` up to `DYN_PARTIAL_MUL_CUTOFF`, and past the dispatch
  limits both go to the NTT.

## Why not across steps

**x changes.** The next step's x is the old one shifted, minus a correction: x' = x B^p − c. Transforms
are linear, so x̂' = (x shifted)^ − ĉ, and the shift is a pointwise twiddle. But c is a truncated product,
so its spectrum isn't known and costs a full transform at the new length. That alone is what
transforming x' directly costs.

**The length doubles.** The even bins of a length-2N transform are the length-N bins of the same
sequence, if it fits N without wrapping, so half of the old x's new spectrum is free. The odd bins still
cost a length-N transform, and that comes on top of ĉ, which makes this dearer than transforming x'.

**d's window grows.** Each step reads `end_ref(d_work, 2p + 1)`, which adds lower limbs. A transform of
the widest window can't be subsampled for an earlier step either: sampling its bins gives the DFT of
that whole window wrapped to the shorter length, which is a different sequence from the shorter window.

The sharing across steps that does work is `RcpCarry`, which reuses the finished reciprocal rather than
a transform (`docs/sqrt_cutoff_tuning.md`).

## Why not merge the two products

The tempting next step is to skip the separate error band. With x' = 2x − d x², one could transform d
and x once, form `d̂ x̂²` pointwise, and run one inverse, with wrapping removing the parts already
known. That is 1 forward pair and 1 inverse at a longer N, roughly 25% less than two products in this
unit, and it is the kind of step behind the published 1.3–1.8 M(n) square-root counts. (Those figures
are from memory and weren't checked against the papers.)

It doesn't fit here because a triple product's coefficients are too large for a double. At 2^16 digits
of size 2^15, a coefficient of d x² can reach about 2^45 · 2^32 = 2^77, against the 2^53 a double holds
exactly, and the rounding reading needs several bits to spare on top of that. Digits small enough to fit
are about 6 bits, which makes the transform about 2.7× longer and loses far more than merging gains.

## What it would be worth

Adjusting the counts for the stages that don't shrink, in percent of a two-product step at n = 2^19:

| | Today | Shared | Change |
|---|---:|---:|---:|
| First product | 100 | about 103 (the extra writes, and an untangle and tangle pass) | +3% |
| Second product | 100 | about 71 (2 of 3 transforms, plus the passes) | −29% |
| Step | 200 | about 174 | **0.87×** |

The same reasoning puts a quotient refinement at about 0.91×. Applied to the shares of the AGM in
`docs/agm_profile.md` (from 4096 to 16384 limbs):

| Part | Share of the AGM | Saving | Off the AGM |
|---|---:|---:|---:|
| NR reciprocal chains | 13.3–17.5% | 13% | 1.7–2.3% |
| NR quotient refinement | 30.4–34.1% | 9% | 2.7–3.1% |
| **Total** | | | **about 4.5–5.5%** |

With the root at about 72% of the AGM, that is about 7% of a root.

Three things make this rough, and the first two lean optimistic:

- Those shares were measured on 2026-11-02, before the FFT was reworked and before the wrapped remainder
  and the reciprocal carry changed the root.
- The low steps of a chain are below the FFT tier and can't share, though the top steps dominate its time.
- It assumes the reused x̂, which came out of a packed transform, rounds as well as a fresh one. That
  hasn't been tested.

It would cost x̂'s storage, about 8 bytes per digit of N, and a new interface in `mul.rs`. `mid` and
`short` are callbacks shared with the static path, so the dynamic chain would need its own product steps.

## If this is ever tried

1. Prototype only the reciprocal step, in `nr_refine_rcp` on the dynamic path, with a pair like
   `fft_mid_mul_keeping_x` and `fft_mul_with_spectrum`.
2. Time `nr_rcp_chain` at several precisions first, and compare the reading of a second product that
   reuses x̂ with one that doesn't.
3. Continue only if the root comes in around 0.93× or better. Past that, add the quotient refinement.

Two side benefits would come with separated transforms: an operand's forward transform could run on its
own thread (recommendation 5 in `docs/agm_profile.md`), and the unbalanced products' packing noise would go
away.

## How it compares for large inputs

Measured 2026-11-06, one thread unless noted, in the shape the AGM uses (`n + 2` limbs in, `n` limbs out):

| n | Product | Square | Root | Root ÷ product |
|---:|---:|---:|---:|---:|
| 4096 | 0.29 ms | 0.21 ms | 0.71 ms | 2.45 |
| 16384 | 1.28 ms | 0.89 ms | 3.43 ms | 2.68 |
| 32768 | 2.80 ms | 1.83 ms | 7.43 ms | 2.65 |
| 65536 | 25.06 ms | 17.45 ms | 16.41 ms | 0.65 |
| 131072 | 55.14 ms | 38.03 ms | 38.34 ms | 0.70 |
| 262144 | 117.59 ms | 81.21 ms | 188.78 ms | 1.61 |

A top Zimmermann division is about 55% of a root in the FFT range, and the wrapped square about 8%.
`DYN_SQRT_WRAPPED_REM_CUTOFF` at 1024 instead of its default changes nothing at these sizes. The ratio
falls at 65536 because the product has crossed onto the NTT at about 53000 limbs, while the root's largest
stages work on half that and stay on the FFT.

Other options for the AGM, from the same run:

| Option | Gain on an AGM step | Status |
|---|---:|---|
| Letting the FFT run past its limits (scratch copy, identical roots on random operands, no correctness work) | 0.54× at 65536–131072 limbs and 0.36× at 262144 on one thread; 0.81–0.87× on 8 threads | Needs a check that always runs past the limits, since the limits are what keep readings meaningful (`docs/fft_precision.md`). Middle products and cyclic squares would only have the fold check. |
| Squares in place of products in the AGM, using a b = 2((a + b)/2)² − (a² + b²)/2 with squares carried between steps | About 0.91× in the FFT range and 0.82–0.88× above it, calculated from the timings, before the early iterations that would keep the product | Not prototyped. The subtraction loses precision while b is much smaller than a. |
| Sharing transforms, as above | About 0.95× | Not prototyped. New FFT interface. |

Above about 53000 limbs the FFT limits matter far more than anything here. Below that, squares in the AGM
look like at least the same gain as sharing for much less code.

## Reproduction

The size comparisons come from `find_fft_size`'s rule over 200000 log-uniform p. The timings came from a
scratch harness over `mul_dyn`, `sqr_dyn`, `hi_mul_dyn`, `hi_sqr_dyn`, `div_dyn`, `sqrt_approx_dyn`,
`sqrt_approx_dyn_at` and `sqr_mod_bnm1_dyn`: the fastest of at least three calls per size and 0.25 s of
calls, on one thread and on the default pool. The raised-limit runs set `FFT_16BIT_CUTOFF`,
`FFT_SQR_16BIT_CUTOFF`, `FFT_MID_16BIT_CUTOFF` and `FFT_BNM1_16BIT_CUTOFF` to 2^20–2^21 and removed the
assert beside them, in a copy of the tree only.
