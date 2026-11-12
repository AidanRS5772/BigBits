use crate::tuning::k;
use crate::utils::{
    div::{
        div_carry_dyn, div_dyn, div_rem_carry_dyn, div_rem_dyn, div_rem_static, div_static,
        knuth_est, mul_u64_asm, reciprocal_3_2, RcpCarry,
    },
    mul::{bnm1_size, sqr_dyn, sqr_mod_bnm1_dyn, sqr_static},
    utils::{
        add_buf, add_mul, buf_len, cmp_buf, combine_u64, dec_buf, fold_mod_bnm1, shl_buf, shr_buf,
        sub_buf, sub_mod_bnm1,
    },
    ScratchGuard, FFT_BNM1_16BIT_CUTOFF,
};

// x -= q_sqr; if that went negative, moves the remainder from s to s - 1 (rem + 2s - 1) and returns
// true, for the caller to decrement s.
#[inline]
pub fn correct_sqrt(x: &mut [u64], s: &[u64], q_sqr: &[u64]) -> bool {
    let negative = sub_buf(x, q_sqr);
    if negative {
        add_buf(x, s);
        add_buf(x, s);
        dec_buf(x);
    }
    negative
}

/// Computes `floor(sqrt(x * B^(2 * s.len() - x.len())))`, where `B = 2^64`, leaving the remainder in
/// `x`. `x` must be normalized, with `s.len() < x.len() <= 2 * s.len()`.
pub fn binom_sqrt_core(x: &mut [u64], s: &mut [u64]) {
    if s.is_empty() {
        return;
    }
    debug_assert!(x.len() > s.len());
    debug_assert!(2 * s.len() >= x.len());
    debug_assert!(x.last().copied().unwrap() >= (1 << 62));

    let (x_len, s_len) = (x.len(), s.len());
    let top = combine_u64(x[x_len - 2], x[x_len - 1]);
    let s_hi = top.isqrt();
    let rem = top - s_hi * s_hi;
    if s_len == 1 {
        s[0] = s_hi as u64;
        x[0] = rem as u64;
        x[1] = (rem >> 64) as u64;
        return;
    }

    // The second root limb from the next two x limbs (the lower one zero if x has only three).
    let a1 = x[x_len - 3];
    let (mut s_lo, u) = if rem == 2 * s_hi {
        (u64::MAX as u128, 2 * s_hi + a1 as u128)
    } else {
        let half = (rem << 63) | (a1 as u128 >> 1);
        let q = half / s_hi;
        (q, ((half - q * s_hi) << 1) | (a1 & 1) as u128)
    };
    let s_lo_sqr = s_lo * s_lo;
    let mut seed_rem = [
        if x_len > 3 { x[x_len - 4] } else { 0 },
        u as u64,
        (u >> 64) as u64,
    ];
    if correct_sqrt(
        &mut seed_rem,
        &[s_lo as u64, s_hi as u64],
        &[s_lo_sqr as u64, (s_lo_sqr >> 64) as u64],
    ) {
        s_lo -= 1;
    }
    let (s_lo, s_hi) = (s_lo as u64, s_hi as u64);
    s[s_len - 2] = s_lo;
    s[s_len - 1] = s_hi;
    let mut x_start = x_len.saturating_sub(4);
    x[x_len - 1] = 0;
    x[x_start..x_start + 3].copy_from_slice(&seed_rem);
    if s_len == 2 {
        return;
    }

    let mut x_end = x_start + 3;
    let mut j = s_len - 2;
    // The same for every root limb, as s's top two limbs don't change.
    let v = reciprocal_3_2(s_hi, s_lo);
    while j > 0 {
        let mut c = 0u64;
        while cmp_buf(&x[x_start..x_end], &s[j..]).is_ge() {
            sub_buf(&mut x[x_start..x_end], &s[j..]);
            c += 1;
        }
        x_end -= 1;
        if x_start < 2 {
            let shift = (2 * j - x_start).min(x_len - x_end);
            x.copy_within(..x_end, shift);
            x[..shift].fill(0);
            x_start += shift;
            x_end += shift;
        }
        j -= 1;
        x_start -= 2;
        let (rem, root) = (&mut x[x_start + 1..x_end], &s[j + 1..]);
        let (win, of) = rem.split_at_mut(root.len());
        let t = knuth_est(win, &mut of[0], root, s_hi, s_lo, v);
        let (q, adds) = if c == 2 {
            (u64::MAX, t + 2)
        } else {
            ((c << 63) | (t >> 1), t & 1)
        };
        for _ in 0..adds {
            add_buf(rem, root);
        }
        s[j] = q;
        let s_sqr = unsafe { mul_u64_asm(q, q) };
        if correct_sqrt(&mut x[x_start..x_end], &s[j..], &[s_sqr.1, s_sqr.0]) {
            s[j] -= 1;
        }
    }
}

fn sqrt_denormalization(x: &mut [u64], s: &mut [u64], sh: u8) {
    if sh == 0 {
        return;
    }
    let d = s[0] & ((1 << sh / 2) - 1);
    add_mul(x, s, 2 * d);
    let d_sqr = unsafe { mul_u64_asm(d, d) };
    sub_buf(x, &[d_sqr.1, d_sqr.0]);
    shr_buf(x, sh);
    shr_buf(s, sh / 2);
}

/// [`binom_sqrt_core`] on an `x` that needn't be normalized.
pub fn binom_sqrt(x: &mut [u64], s: &mut [u64]) {
    normalized(x, s, SqrtOutput::RootRem, binom_sqrt_core);
}

pub fn reduce_sqrt_rem(sqrt_r: &mut [u64], s_hi: &[u64]) -> (bool, bool) {
    if cmp_buf(sqrt_r, s_hi).is_lt() {
        return (false, false);
    }
    sub_buf(sqrt_r, s_hi);
    (true, cmp_buf(sqrt_r, s_hi).is_eq())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SqrtOutput {
    RootRem,
    Root,
    ApproxRoot,
    // RootRem for a recursive stage, leaving x above the remainder unspecified.
    Stage,
}

// The sqrt knobs one call reads (docs/tuning_system.md, invariants 6 and 7): read once at the top
// level and passed down the recursion, so every stage of a call sees the same values.
#[derive(Clone, Copy, Debug)]
struct SqrtKnobs {
    // Root width from which the top level runs Zimmermann rather than the binomial root.
    top: usize,
    // Root width below which a recursive stage is the binomial root (the recursion's leaf).
    leaf: usize,
    // Root width from which a dynamic stage takes the wrapped remainder; None for static roots.
    wrapped: Option<usize>,
    // Root width from which a dynamic root carries one reciprocal up its stages.
    rcp_carry: usize,
}

impl SqrtKnobs {
    fn dyn_read(output: SqrtOutput, wrapped: usize) -> Self {
        let top = match output {
            SqrtOutput::RootRem | SqrtOutput::Stage => k::sqrt::dyn_rem(),
            SqrtOutput::Root => k::sqrt::dyn_only(),
            SqrtOutput::ApproxRoot => k::sqrt::dyn_approx(),
        };
        Self { top, leaf: k::sqrt::leaf(), wrapped: Some(wrapped), rcp_carry: k::sqrt::dyn_rcp_carry() }
    }

    fn static_read(output: SqrtOutput) -> Self {
        let top = match output {
            SqrtOutput::RootRem | SqrtOutput::Stage => k::sqrt::static_rem(),
            SqrtOutput::Root => k::sqrt::static_only(),
            SqrtOutput::ApproxRoot => k::sqrt::static_approx(),
        };
        Self { top, leaf: k::sqrt::leaf(), wrapped: None, rcp_carry: usize::MAX }
    }
}

/// The algorithm for a root of `root_len` limbs: the binomial root below `cutoff`, Zimmermann from
/// it. At the top level `cutoff` is the entry's knob (`sqrt.dyn_rem`, …), in a recursive stage the
/// leaf (`sqrt.leaf`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SqrtAlg {
    Binomial,
    Zimmermann,
}

pub(crate) fn sqrt_alg(root_len: usize, cutoff: usize) -> SqrtAlg {
    if root_len < cutoff {
        SqrtAlg::Binomial
    } else {
        SqrtAlg::Zimmermann
    }
}

fn dyn_div(n: &mut [u64], d: &[u64], q: &mut [u64], remainder: bool) -> u64 {
    if remainder {
        div_rem_dyn(n, d, q)
    } else {
        div_dyn(n, d, q)
    }
}

fn static_div<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64], remainder: bool) -> u64 {
    if remainder {
        div_rem_static::<N>(n, d, q)
    } else {
        div_static::<N>(n, d, q)
    }
}

// A recursive stage: the exact root and remainder of a full x, which never asks for square scratch.
fn zimmermann_sqrt_core(
    x: &mut [u64],
    s: &mut [u64],
    div_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64], bool) -> u64,
    sqr_alg: &mut dyn FnMut(&[u64], &mut [u64]) -> u64,
    knobs: SqrtKnobs,
) {
    debug_assert_eq!(x.len(), 2 * s.len());
    if sqrt_alg(s.len(), knobs.leaf) == SqrtAlg::Binomial {
        binom_sqrt_core(x, s);
        return;
    }
    zimmerman_sqrt_entry(
        x,
        s,
        SqrtOutput::Stage,
        div_alg,
        sqr_alg,
        &mut |_, _| unreachable!(),
        knobs,
    );
}

// The sole entry into the recursive stack. One stage: s_hi from the child, then s_lo from dividing by
// 2 s_hi. Only the outer stage may skip remainder work, accept an approximate root or read a padded
// x (X = x B^p); it gets its square buffer for a padded x from with_square_scratch, lazily.
// knobs.wrapped is the root width from which a stage takes the wrapped remainder.
fn zimmerman_sqrt_entry(
    x: &mut [u64],
    s: &mut [u64],
    output: SqrtOutput,
    div_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64], bool) -> u64,
    sqr_alg: &mut dyn FnMut(&[u64], &mut [u64]) -> u64,
    with_square_scratch: &mut dyn FnMut(usize, &mut dyn FnMut(&mut [u64])),
    knobs: SqrtKnobs,
) {
    let (x_len, s_len) = (x.len(), s.len());
    let padded = x_len < 2 * s_len;
    let lo = if padded {
        s_len - x_len / 2
    } else {
        (s_len - 1) / 2
    };
    let hi = s_len - lo;
    let x_lo = x_len - 2 * hi;

    // R = X - s^2 lies in [-(2s - 1), 2s], narrower than B^m - 1 for m > s_len, so the wrapped
    // remainder reads it from one square mod B^m - 1 and the division can skip its own. X is folded
    // first, as the child overwrites it, and multiplying by B^p rotates it.
    let modulus = knobs
        .wrapped
        .filter(|&cutoff| {
            matches!(output, SqrtOutput::RootRem | SqrtOutput::Stage) && s_len >= cutoff
        })
        .map(|_| bnm1_size(s_len + 1))
        .filter(|&m| m <= FFT_BNM1_16BIT_CUTOFF);
    let mut scratch;
    let residues = if let Some(m) = modulus {
        scratch = ScratchGuard::acquire();
        let [x_hat, s_sqr] = scratch.get_splits([m, m]);
        fold_mod_bnm1(x, x_hat);
        x_hat.rotate_right(2 * s_len - x_len);
        Some((x_hat, s_sqr))
    } else {
        None
    };

    let (s_lo, s_hi) = s.split_at_mut(lo);
    zimmermann_sqrt_core(&mut x[x_lo..], s_hi, div_alg, sqr_alg, knobs);
    let (reduced, saturated) = reduce_sqrt_rem(&mut x[x_lo..x_lo + hi + 1], s_hi);

    // A padded x materializes its virtual low zeros, which a full x keeps below this window.
    let numerator = if padded {
        x.copy_within(..hi + x_lo + 1, lo - x_lo);
        x[..lo - x_lo].fill(0);
        x[s_len + 1..].fill(0);
        &mut x[..]
    } else {
        &mut x[lo..s_len + lo]
    };

    // floor(N / 2 s_hi), with the remainder if a later step reads it.
    let remainder = residues.is_none() && output != SqrtOutput::ApproxRoot;
    if saturated {
        if remainder {
            numerator[lo..].fill(0);
            add_buf(numerator, s_hi);
            add_buf(numerator, s_hi);
        }
        s_lo.fill(u64::MAX);
    } else {
        let n_len = buf_len(numerator);
        let overflow = div_alg(&mut numerator[..n_len], s_hi, s_lo, remainder);
        debug_assert_eq!(overflow, 0, "Zimmermann quotient exceeded low half");
        if shr_buf(s_lo, 1) != 0 && remainder {
            add_buf(numerator, s_hi);
        }
        if reduced {
            s_lo[lo - 1] |= 1 << 63;
        }
    }

    if let Some((x_hat, s_sqr)) = residues {
        sqr_mod_bnm1_dyn(s, s_sqr);
        sub_mod_bnm1(x_hat, s_sqr);
        let rem = &mut x[..s_len + 1];
        rem.copy_from_slice(&x_hat[..s_len + 1]);
        // R's window is below B^m / 4, so the top bit gives its sign. R < 0 means s is one too big:
        // R + 2s - 1 = w + 1 - B^m + 2s - 1, which is w + 2s at these limbs.
        let negative = x_hat[x_hat.len() - 1] >> 63 != 0;
        debug_assert!(
            x_hat[s_len + 1..]
                .iter()
                .all(|&limb| limb == if negative { u64::MAX } else { 0 }),
            "wrapped remainder outside its window"
        );
        if negative {
            add_buf(rem, s);
            add_buf(rem, s);
            dec_buf(s);
        }
        debug_assert!(rem[s_len] <= 1, "wrapped remainder above 2s");
    } else if output != SqrtOutput::ApproxRoot
        && (output != SqrtOutput::Root || numerator[lo..].iter().all(|&limb| limb == 0))
    {
        let mut finish = |rem: &mut [u64], s_lo_sqr: &mut [u64]| {
            let overflow = sqr_alg(&s[..lo], s_lo_sqr);
            debug_assert_eq!(overflow, 0, "Zimmermann low square exceeded its buffer");
            let too_big = if output == SqrtOutput::Root {
                cmp_buf(rem, s_lo_sqr).is_lt()
            } else {
                correct_sqrt(rem, s, s_lo_sqr)
            };
            if too_big {
                dec_buf(s);
            }
        };
        if padded {
            x.copy_within(..x_len - lo, lo);
            x[..lo].fill(0);
            with_square_scratch(2 * lo, &mut |s_lo_sqr| finish(x, s_lo_sqr));
        } else {
            let (rem, s_lo_sqr) = x[..s_len + 2 * lo + 1].split_at_mut(s_len + 1);
            finish(rem, s_lo_sqr);
        }
    }
    if output == SqrtOutput::RootRem {
        x[s_len + 1..].fill(0);
    }
}

// Every top level: a one-limb x (X = x B, whose remainder wouldn't fit x), or root between x's
// normalization and its undoing.
fn normalized(
    x: &mut [u64],
    s: &mut [u64],
    output: SqrtOutput,
    root: impl FnOnce(&mut [u64], &mut [u64]),
) {
    if x.len() == 1 && s.len() == 1 && output != SqrtOutput::RootRem {
        s[0] = ((x[0] as u128) << 64).isqrt() as u64;
        return;
    }
    debug_assert!(x.len() > s.len());
    debug_assert!(2 * s.len() >= x.len());
    let sh = x[x.len() - 1].leading_zeros() as u8 & !1_u8;
    shl_buf(x, sh);
    root(x, s);
    if output == SqrtOutput::RootRem {
        sqrt_denormalization(x, s, sh);
    } else {
        shr_buf(s, sh / 2);
    }
}

fn sqrt_dyn_output(x: &mut [u64], s: &mut [u64], output: SqrtOutput, wrapped_cutoff: usize) {
    let knobs = SqrtKnobs::dyn_read(output, wrapped_cutoff);
    if sqrt_alg(s.len(), knobs.top) == SqrtAlg::Binomial {
        normalized(x, s, output, binom_sqrt_core);
        return;
    }
    // Stages divide bottom up, each by the whole root of the stage below, so one reciprocal climbs
    // the recursion. The widest worth keeping is the top stage's child's, about 1/8 of the root.
    let (mut carry_scratch, mut carry, mut carried);
    let div: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64], bool) -> u64 =
        if s.len() < knobs.rcp_carry {
            &mut dyn_div
        } else {
            carry_scratch = ScratchGuard::acquire();
            carry = RcpCarry::new(carry_scratch.get(s.len() / 8 + 8));
            carried = |n: &mut [u64], d: &[u64], q: &mut [u64], remainder: bool| {
                if remainder {
                    div_rem_carry_dyn(n, d, q, &mut carry)
                } else {
                    div_carry_dyn(n, d, q, &mut carry)
                }
            };
            &mut carried
        };
    normalized(x, s, output, |x, s| {
        zimmerman_sqrt_entry(
            x,
            s,
            output,
            div,
            &mut sqr_dyn,
            &mut |square_len, finish| {
                let mut guard = ScratchGuard::acquire();
                finish(guard.get(square_len));
            },
            knobs,
        );
    });
}

fn sqrt_static_output<const N: usize>(x: &mut [u64], s: &mut [u64], output: SqrtOutput) {
    debug_assert!(
        x.len() <= N && s.len() <= N,
        "Zimmermann sqrt operands exceed static capacity"
    );
    let knobs = SqrtKnobs::static_read(output);
    if sqrt_alg(s.len(), knobs.top) == SqrtAlg::Binomial {
        normalized(x, s, output, binom_sqrt_core);
        return;
    }
    normalized(x, s, output, |x, s| {
        zimmerman_sqrt_entry(
            x,
            s,
            output,
            &mut static_div::<N>,
            &mut sqr_static::<N>,
            &mut |square_len, finish| {
                let mut s_lo_sqr = [0_u64; N];
                finish(&mut s_lo_sqr[..square_len]);
            },
            knobs,
        );
    });
}

/// Computes the exact root and remainder of `X = x * B^(2 * s.len() - x.len())`, where `B = 2^64`:
/// `floor(sqrt(X))` in `s`, the remainder in `x`. Requires trimmed nonzero `x` and
/// `s.len() < x.len() <= 2 * s.len()`.
pub fn sqrt_dyn(x: &mut [u64], s: &mut [u64]) {
    sqrt_dyn_output(x, s, SqrtOutput::RootRem, k::sqrt::dyn_wrapped_rem());
}

/// [`sqrt_dyn`] with the wrapped remainder from `wrapped_cutoff` root limbs (`usize::MAX`: none), for
/// tests and tuning. The output is the same.
#[doc(hidden)]
pub fn sqrt_dyn_at(x: &mut [u64], s: &mut [u64], wrapped_cutoff: usize) {
    sqrt_dyn_output(x, s, SqrtOutput::RootRem, wrapped_cutoff);
}

/// [`sqrt_dyn`]'s root only, with `x` left unspecified. Also takes `x.len() == s.len() == 1`
/// (`X = x * B`).
pub fn sqrt_only_dyn(x: &mut [u64], s: &mut [u64]) {
    sqrt_dyn_output(x, s, SqrtOutput::Root, k::sqrt::dyn_wrapped_rem());
}

/// [`sqrt_only_dyn`] with an explicit wrapped-remainder cutoff, as [`sqrt_dyn_at`].
#[doc(hidden)]
pub fn sqrt_only_dyn_at(x: &mut [u64], s: &mut [u64], wrapped_cutoff: usize) {
    sqrt_dyn_output(x, s, SqrtOutput::Root, wrapped_cutoff);
}

/// [`sqrt_only_dyn`], except that `s` is `floor(sqrt(X))` or `floor(sqrt(X)) + 1`. A numeric error
/// bound, not a guarantee of identical high limbs.
pub fn sqrt_approx_dyn(x: &mut [u64], s: &mut [u64]) {
    sqrt_dyn_output(x, s, SqrtOutput::ApproxRoot, k::sqrt::dyn_wrapped_rem());
}

/// [`sqrt_approx_dyn`] with an explicit wrapped-remainder cutoff, as [`sqrt_dyn_at`].
#[doc(hidden)]
pub fn sqrt_approx_dyn_at(x: &mut [u64], s: &mut [u64], wrapped_cutoff: usize) {
    sqrt_dyn_output(x, s, SqrtOutput::ApproxRoot, wrapped_cutoff);
}

/// [`sqrt_dyn`] without the heap, for operands of at most `N` limbs.
pub fn sqrt_static<const N: usize>(x: &mut [u64], s: &mut [u64]) {
    sqrt_static_output::<N>(x, s, SqrtOutput::RootRem);
}

/// [`sqrt_only_dyn`] without the heap, for operands of at most `N` limbs.
pub fn sqrt_only_static<const N: usize>(x: &mut [u64], s: &mut [u64]) {
    sqrt_static_output::<N>(x, s, SqrtOutput::Root);
}

/// [`sqrt_approx_dyn`] without the heap, for operands of at most `N` limbs.
pub fn sqrt_approx_static<const N: usize>(x: &mut [u64], s: &mut [u64]) {
    sqrt_static_output::<N>(x, s, SqrtOutput::ApproxRoot);
}

pub use self::{
    sqrt_approx_dyn as zimmerman_sqrt_approx_dyn,
    sqrt_approx_static as zimmerman_sqrt_approx_static, sqrt_dyn as zimmerman_sqrt_dyn,
    sqrt_only_dyn as zimmerman_sqrt_only_dyn, sqrt_only_static as zimmerman_sqrt_only_static,
    sqrt_static as zimmerman_sqrt_static,
};
