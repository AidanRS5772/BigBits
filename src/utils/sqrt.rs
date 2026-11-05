use crate::utils::{
    div::{
        div_buf_of, div_dyn, div_rem_dyn, div_rem_static, div_static, knuth_est,
        knuth_rcp_normalized, mul_u64_asm, reciprocal_3_2,
    },
    mul::{bnm1_size, mul_elem, sqr_dyn, sqr_mod_bnm1_dyn, sqr_static},
    utils::{
        add_buf, add_mul, buf_len, cmp_buf, combine_u64, dec_buf, end_mut, end_ref, fold_mod_bnm1,
        inc_buf, shl_buf, shr_buf, sub_buf, sub_mod_bnm1, twos_comp,
    },
    ScratchGuard, DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF, DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF,
    DYN_SQRT_REM_ZIMMERMAN_CUTOFF, DYN_SQRT_WRAPPED_REM_CUTOFF, FFT_BNM1_16BIT_CUTOFF,
    STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF, STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF,
    STATIC_SQRT_REM_ZIMMERMAN_CUTOFF, ZIMMERMAN_SQRT_LEAF_CUTOFF,
};

// rem += 2s - 1, the remainder of s - 1 from that of s: (s - 1)^2 = s^2 - 2s + 1.
#[inline]
fn add_root_step(rem: &mut [u64], s: &[u64]) {
    add_buf(rem, s);
    add_buf(rem, s);
    dec_buf(rem);
}

#[inline]
pub fn correct_sqrt(x: &mut [u64], s: &[u64], q_sqr: &[u64]) -> bool {
    let c = sub_buf(x, q_sqr);
    if c {
        add_root_step(x, s);
    }
    return c;
}

fn sqrt_4x2(x: &mut [u64; 4]) -> (u64, u64) {
    let x_hi = combine_u64(x[2], x[3]);
    let s_hi = x_hi.isqrt();
    let rem = x_hi - s_hi * s_hi;
    let (mut s_lo, u) = if rem == 2 * s_hi {
        (u64::MAX as u128, 2 * s_hi + x[1] as u128)
    } else {
        let half = (rem << 63) | ((x[1] as u128) >> 1);
        let q = half / s_hi;
        let v = half % s_hi;
        (q, (v << 1) | ((x[1] & 1) as u128))
    };
    let s_lo_sqr = s_lo * s_lo;
    let mut buf = [x[0], u as u64, (u >> 64) as u64];
    if correct_sqrt(
        &mut buf,
        &[s_lo as u64, s_hi as u64],
        &[s_lo_sqr as u64, (s_lo_sqr >> 64) as u64],
    ) {
        s_lo -= 1;
    }
    x[..3].copy_from_slice(&buf);
    x[3] = 0;
    (s_lo as u64, s_hi as u64)
}

// v = reciprocal_3_2(s1, s0), the same for every root limb, as s's top two limbs don't change.
fn binom_sqrt_est_reduced(x: &mut [u64], s: &[u64], s0: u64, s1: u64, v: u64, c: u64) -> u64 {
    let t = {
        let (win, of) = x.split_at_mut(s.len());
        knuth_est(win, &mut of[0], s, s1, s0, v)
    };
    let (q, d) = match c {
        0 => (t >> 1, t & 1),
        1 => ((1 << 63) | (t >> 1), t & 1),
        2 => (u64::MAX, t + 2),
        _ => unreachable!(),
    };
    for _ in 0..d {
        add_buf(x, s);
    }
    return q;
}

/// Computes `floor(sqrt(x * B^(2 * s.len() - x.len())))`, where `B = 2^64`.
/// The input must be normalized, and its length must satisfy
/// `s.len() < x.len() <= 2 * s.len()`. On return, `x` contains the remainder.
pub fn binom_sqrt_core(x: &mut [u64], s: &mut [u64]) {
    if s.is_empty() {
        return;
    }
    // assume correct size bounds
    debug_assert!(x.len() > s.len());
    debug_assert!(2 * s.len() >= x.len());
    // assume normalized x
    debug_assert!(x.last().copied().unwrap() >= (1 << 62));

    let x_len = x.len();
    let s_len = s.len();
    s.fill(0);

    // x.len() == 2
    if s_len == 1 {
        let term = combine_u64(x[0], x[1]);
        let sqrt = term.isqrt();
        let rem = term - sqrt * sqrt;
        s[0] = sqrt as u64;
        x[0] = rem as u64;
        x[1] = (rem >> 64) as u64;
        return;
    }

    let mut x_start = x_len.saturating_sub(4);
    let avail = x_len.min(4);

    let mut window = [0u64; 4];
    window[4 - avail..].copy_from_slice(&x[x_start..x_start + avail]);

    let (s_lo, s_hi) = sqrt_4x2(&mut window);
    s[s_len - 2] = s_lo;
    s[s_len - 1] = s_hi;

    x[x_start..x_start + avail].copy_from_slice(&window[..avail]);

    if s_len == 2 {
        return;
    }

    let mut x_end = x_start + 3;
    let mut j = s_len - 2;
    let v = reciprocal_3_2(s_hi, s_lo);
    while j > 0 {
        let mut c = 0;
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
        s[j] = binom_sqrt_est_reduced(&mut x[x_start + 1..x_end], &s[j + 1..], s_lo, s_hi, v, c);
        let s_sqr = unsafe { mul_u64_asm(s[j], s[j]) };
        if correct_sqrt(&mut x[x_start..x_end], &s[j..], &[s_sqr.1, s_sqr.0]) {
            s[j] -= 1;
        }
    }
}

fn sqrt_denormalization(x: &mut [u64], s: &mut [u64], sh: u8) {
    let d = s[0] & ((1 << sh / 2) - 1);
    add_mul(x, s, 2 * d);
    let d_sqr = unsafe { mul_u64_asm(d, d) };
    sub_buf(x, &[d_sqr.1, d_sqr.0]);
    shr_buf(x, sh);
    shr_buf(s, sh / 2);
}

pub fn binom_sqrt(x: &mut [u64], s: &mut [u64]) {
    // assume correct size bounds
    debug_assert!(x.len() > s.len());
    debug_assert!(x.len() <= 2 * s.len());

    let x_len = x.len();
    let sh = x[x_len - 1].leading_zeros() as u8 & !1_u8;
    if sh != 0 {
        shl_buf(x, sh);
    }
    binom_sqrt_core(x, s);
    if sh != 0 {
        sqrt_denormalization(x, s, sh);
    }
}

pub fn reduce_sqrt_rem(sqrt_r: &mut [u64], s_hi: &[u64]) -> (bool, bool) {
    let reduced = cmp_buf(sqrt_r, s_hi).is_ge();
    let saturated = if reduced {
        sub_buf(sqrt_r, s_hi);
        cmp_buf(sqrt_r, s_hi).is_eq()
    } else {
        false
    };
    (reduced, saturated)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SqrtOutput {
    RootRem,
    Root,
    ApproxRoot,
}

impl SqrtOutput {
    // Root widths from which each top-level entry prefers Zimmermann.
    fn dyn_cutoff(self) -> usize {
        match self {
            Self::RootRem => DYN_SQRT_REM_ZIMMERMAN_CUTOFF,
            Self::Root => DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF,
            Self::ApproxRoot => DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF,
        }
    }

    fn static_cutoff(self) -> usize {
        match self {
            Self::RootRem => STATIC_SQRT_REM_ZIMMERMAN_CUTOFF,
            Self::Root => STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF,
            Self::ApproxRoot => STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF,
        }
    }
}

// The dynamic divisions behind every Zimmermann stage: with or without the remainder.
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

// Forms a stage's remainder R = X - s^2 from one square mod B^m - 1 rather than from the
// division's remainder and the low root's square, at stages of at least `cutoff` root limbs whose
// modulus is at most `max_m` limbs.
// R lies in [-(2s - 1), 2s] (s is the root or one too big), and m > s.len() makes B^m - 1 wider
// than that window, so (X - s^2) mod B^m - 1 pins R down and the division can skip its remainder.
// Dynamic only: its stages take pooled scratch for the residues, and static passes None.
#[derive(Clone, Copy)]
struct WrappedRem {
    cutoff: usize,
    // The widest modulus sqr_mod squares on its cyclic transform; past it the square would be a
    // full one, which costs about what it saves.
    max_m: usize,
    sqr_mod: fn(&[u64], &mut [u64]),
}

impl WrappedRem {
    fn takes(&self, s_len: usize) -> bool {
        s_len >= self.cutoff && bnm1_size(s_len + 1) <= self.max_m
    }
}

// A wrapped stage's quotient: floor(N / 2s') into s_lo, from the numerator N = r' B^lo + a1 that
// reduce_sqrt_rem left (less s' when `reduced`). floor(floor(N / s') / 2) = floor(N / 2s'), so
// with no remainder to keep, halving needs no fix-up.
fn wrapped_quotient(
    numerator: &mut [u64],
    s_lo: &mut [u64],
    s_hi: &[u64],
    reduced: bool,
    saturated: bool,
    div_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64], bool) -> u64,
) {
    if saturated {
        s_lo.fill(u64::MAX);
        return;
    }
    let n_len = buf_len(numerator);
    let overflow = div_alg(&mut numerator[..n_len], s_hi, s_lo, false);
    debug_assert_eq!(overflow, 0, "Zimmermann quotient exceeded low half");
    shr_buf(s_lo, 1);
    if reduced {
        let lo = s_lo.len();
        s_lo[lo - 1] |= 1 << 63;
    }
}

// Writes a wrapped stage's remainder to rem (s.len() + 1 limbs) from w = (X - s^2) mod B^m - 1,
// canonical, m > s.len(). R's window is below B^m / 4 either way, so w's top bit gives its sign:
// R = w, or R = w - (B^m - 1) < 0 when s is one too big, which takes the usual one-step repair.
fn rem_from_wrapped(rem: &mut [u64], s: &mut [u64], w: &[u64]) {
    let k = s.len();
    debug_assert_eq!(rem.len(), k + 1);
    debug_assert!(w.len() > k);
    rem.copy_from_slice(&w[..k + 1]);
    if w[w.len() - 1] >> 63 != 0 {
        debug_assert!(
            w[k + 1..].iter().all(|&limb| limb == u64::MAX),
            "wrapped remainder below -2s"
        );
        // w - (B^m - 1) = w + 1 - B^m: w + 1 at k + 1 limbs is R in two's complement.
        inc_buf(rem);
        add_root_step(rem, s);
        dec_buf(s);
    } else {
        debug_assert!(
            w[k + 1..].iter().all(|&limb| limb == 0),
            "wrapped remainder above 2s"
        );
    }
    debug_assert!(rem[k] <= 1, "wrapped remainder above 2s");
}

// Every recursive stage computes an exact root and remainder.
fn zimmermann_sqrt_core(
    x: &mut [u64],
    s: &mut [u64],
    div_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64], bool) -> u64,
    sqr_alg: &mut dyn FnMut(&[u64], &mut [u64]) -> u64,
    wrapped: Option<WrappedRem>,
) {
    debug_assert_eq!(x.len(), 2 * s.len());
    if s.len() < ZIMMERMAN_SQRT_LEAF_CUTOFF {
        binom_sqrt_core(x, s);
        return;
    }
    let s_len = s.len();
    let lo = (s_len - 1) / 2;

    if let Some(w) = wrapped.filter(|w| w.takes(s_len)) {
        // X mod B^m - 1 is taken first, since the child overwrites X's top half.
        let m = bnm1_size(s_len + 1);
        let mut scratch = ScratchGuard::acquire();
        let [x_hat, s_sqr] = scratch.get_splits([m, m]);
        fold_mod_bnm1(x, x_hat);
        let (s_lo, s_hi) = s.split_at_mut(lo);
        zimmermann_sqrt_core(&mut x[2 * lo..], s_hi, div_alg, sqr_alg, wrapped);
        let (reduced, saturated) = reduce_sqrt_rem(&mut x[2 * lo..s_len + lo + 1], s_hi);
        wrapped_quotient(
            &mut x[lo..s_len + lo],
            s_lo,
            s_hi,
            reduced,
            saturated,
            div_alg,
        );
        (w.sqr_mod)(s, s_sqr);
        sub_mod_bnm1(x_hat, s_sqr);
        rem_from_wrapped(&mut x[..s_len + 1], s, x_hat);
        return;
    }

    let (s_lo, s_hi) = s.split_at_mut(lo);
    zimmermann_sqrt_core(&mut x[2 * lo..], s_hi, div_alg, sqr_alg, wrapped);
    let (reduced, saturated) = reduce_sqrt_rem(&mut x[2 * lo..s_len + lo + 1], s_hi);

    if saturated {
        let rem = &mut x[..s_len + 1];
        rem[2 * lo..].fill(0);
        add_buf(&mut rem[lo..], s_hi);
        add_buf(&mut rem[lo..], s_hi);
        s_lo.fill(u64::MAX);
    } else {
        let numerator = &mut x[lo..s_len + lo];
        let n_len = buf_len(numerator);
        let overflow = div_alg(&mut numerator[..n_len], s_hi, s_lo, true);
        debug_assert_eq!(overflow, 0, "Zimmermann quotient exceeded low half");
        if shr_buf(s_lo, 1) != 0 {
            add_buf(&mut x[lo..], s_hi);
        }
        if reduced {
            s_lo[lo - 1] |= 1 << 63;
        }
    }

    let (rem, s_lo_sqr) = x[..s_len + 2 * lo + 1].split_at_mut(s_len + 1);
    let overflow = sqr_alg(s_lo, s_lo_sqr);
    debug_assert_eq!(overflow, 0, "Zimmermann low square exceeded its buffer");
    if correct_sqrt(rem, s, s_lo_sqr) {
        dec_buf(s);
    }
}

// The sole entry into the recursive stack. Only this outer stage may skip
// remainder work or accept an approximate root. Padded inputs obtain their
// final square buffer lazily from the allocation-specific caller.
fn zimmerman_sqrt_entry(
    x: &mut [u64],
    s: &mut [u64],
    output: SqrtOutput,
    div_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64], bool) -> u64,
    sqr_alg: &mut dyn FnMut(&[u64], &mut [u64]) -> u64,
    with_square_scratch: &mut dyn FnMut(usize, &mut dyn FnMut(&mut [u64])),
    wrapped: Option<WrappedRem>,
) {
    let x_len = x.len();
    let s_len = s.len();
    let padded = x_len < 2 * s_len;
    let lo = if padded {
        s_len - x_len / 2
    } else {
        (s_len - 1) / 2
    };
    let hi = s_len - lo;
    let x_lo = x_len - 2 * hi;

    // A root-and-remainder top level from the wrapped cutoff forms its remainder as the core's
    // stages do. X = x B^p with p = 2 s_len - x_len < m, and multiplying by B^p mod B^m - 1
    // rotates the limbs.
    let wrapped_top = wrapped.filter(|w| output == SqrtOutput::RootRem && w.takes(s_len));
    let mut scratch;
    let residues = if wrapped_top.is_some() {
        let m = bnm1_size(s_len + 1);
        scratch = ScratchGuard::acquire();
        let [x_hat, s_sqr] = scratch.get_splits([m, m]);
        fold_mod_bnm1(x, x_hat);
        x_hat.rotate_right(2 * s_len - x_len);
        Some((x_hat, s_sqr))
    } else {
        None
    };

    let (s_lo, s_hi) = s.split_at_mut(lo);
    zimmermann_sqrt_core(&mut x[x_lo..], s_hi, div_alg, sqr_alg, wrapped);
    let (reduced, saturated) = reduce_sqrt_rem(&mut x[x_lo..x_lo + hi + 1], s_hi);

    // Both shapes use the same numerator once any virtual low zeros are
    // materialized. Full inputs retain their low lo limbs below this window.
    let numerator = if padded {
        x[x_lo + hi + 1..].fill(0);
        x.copy_within(..hi + x_lo + 1, lo - x_lo);
        x[..lo - x_lo].fill(0);
        &mut x[..]
    } else {
        &mut x[lo..s_len + lo]
    };

    if let (Some(w), Some((x_hat, s_sqr))) = (wrapped_top, residues) {
        wrapped_quotient(numerator, s_lo, s_hi, reduced, saturated, div_alg);
        (w.sqr_mod)(s, s_sqr);
        sub_mod_bnm1(x_hat, s_sqr);
        x[s_len + 1..].fill(0);
        rem_from_wrapped(&mut x[..s_len + 1], s, x_hat);
        return;
    }

    if saturated {
        if output != SqrtOutput::ApproxRoot {
            numerator[lo..].fill(0);
            add_buf(numerator, s_hi);
            add_buf(numerator, s_hi);
        }
        s_lo.fill(u64::MAX);
    } else {
        let n_len = buf_len(numerator);
        let overflow = div_alg(
            &mut numerator[..n_len],
            s_hi,
            s_lo,
            output != SqrtOutput::ApproxRoot,
        );
        debug_assert_eq!(overflow, 0, "Zimmermann quotient exceeded low half");
        if shr_buf(s_lo, 1) != 0 && output != SqrtOutput::ApproxRoot {
            add_buf(numerator, s_hi);
        }
        if reduced {
            s_lo[lo - 1] |= 1 << 63;
        }
    }

    match output {
        SqrtOutput::ApproxRoot => return,
        SqrtOutput::Root if numerator[lo..].iter().any(|&limb| limb != 0) => return,
        _ => {}
    }

    let mut finish = |rem: &mut [u64], s_lo_sqr: &mut [u64]| {
        let overflow = sqr_alg(&s[..lo], s_lo_sqr);
        debug_assert_eq!(overflow, 0, "Zimmermann low square exceeded its buffer");
        let correct = match output {
            SqrtOutput::RootRem => correct_sqrt(rem, s, s_lo_sqr),
            SqrtOutput::Root => cmp_buf(rem, s_lo_sqr).is_lt(),
            SqrtOutput::ApproxRoot => unreachable!(),
        };
        if correct {
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
        if output == SqrtOutput::RootRem {
            x[s_len + 1..].fill(0);
        }
    }
}

/// Computes the exact root and remainder of `X = value(x) * B^(2*s.len()-x.len())`,
/// where `B = 2^64`, using pooled dynamic scratch. Writes `floor(sqrt(X))` to `s`
/// and the remainder to `x`. Requires trimmed nonzero `x` and
/// `s.len() < x.len() <= 2*s.len()`.
pub fn zimmerman_sqrt_dyn(x: &mut [u64], s: &mut [u64]) {
    sqrt_dyn(x, s);
}

/// Computes only the exact root of `X = value(x) * B^(2*s.len()-x.len())`, where
/// `B = 2^64`, using pooled dynamic scratch. Writes `floor(sqrt(X))` to `s`.
/// Requires trimmed nonzero `x` and `s.len() < x.len() <= 2*s.len()`, or
/// `x.len() == s.len() == 1` (`X = x * B`).
/// The input `x` is used as scratch; its contents on return are unspecified.
pub fn zimmerman_sqrt_only_dyn(x: &mut [u64], s: &mut [u64]) {
    sqrt_only_dyn(x, s);
}

/// Computes an approximate root of `X = value(x) * B^(2*s.len()-x.len())`, where
/// `B = 2^64`, using pooled dynamic scratch. The returned integer in `s` is either
/// `floor(sqrt(X))` or `floor(sqrt(X)) + 1`; small inputs may be computed exactly.
/// This is a numeric error bound, not a guarantee of identical high limbs.
/// Requires trimmed nonzero `x` and `s.len() < x.len() <= 2*s.len()`, or
/// `x.len() == s.len() == 1` (`X = x * B`).
/// The input `x` is used as scratch; its contents on return are unspecified.
pub fn zimmerman_sqrt_approx_dyn(x: &mut [u64], s: &mut [u64]) {
    sqrt_approx_dyn(x, s);
}

// A one-limb x into a one-limb root, X = x B. It is below the s.len() < x.len() bound, which
// leaves x room for the remainder (up to 2 sqrt(X), two limbs here), so only the root outputs
// take it. Returns whether it did.
fn one_limb_root(x: &[u64], s: &mut [u64], output: SqrtOutput) -> bool {
    let one = x.len() == 1 && s.len() == 1 && output != SqrtOutput::RootRem;
    if one {
        s[0] = ((x[0] as u128) << 64).isqrt() as u64;
    }
    one
}

// wrapped_cutoff: the root width from which a stage forms its remainder from a wrapped square
// (DYN_SQRT_WRAPPED_REM_CUTOFF by default, usize::MAX for none).
fn sqrt_dyn_output(x: &mut [u64], s: &mut [u64], output: SqrtOutput, wrapped_cutoff: usize) {
    if one_limb_root(x, s, output) {
        return;
    }
    // assume correct size bounds
    debug_assert!(x.len() > s.len());
    debug_assert!(2 * s.len() >= x.len());

    if s.len() < output.dyn_cutoff() {
        binom_sqrt_output(x, s, output);
        return;
    }

    let x_len = x.len();
    let sh = x[x_len - 1].leading_zeros() as u8 & !1_u8;
    shl_buf(x, sh);

    let mut sqr = |value: &[u64], out: &mut [u64]| sqr_dyn(value, out);

    zimmerman_sqrt_entry(
        x,
        s,
        output,
        &mut dyn_div,
        &mut sqr,
        &mut |square_len, finish| {
            let mut guard = ScratchGuard::acquire();
            finish(guard.get(square_len));
        },
        Some(WrappedRem {
            cutoff: wrapped_cutoff,
            max_m: FFT_BNM1_16BIT_CUTOFF,
            sqr_mod: sqr_mod_bnm1_dyn,
        }),
    );

    if output == SqrtOutput::RootRem {
        sqrt_denormalization(x, s, sh);
    } else {
        shr_buf(s, sh / 2);
    }
}

/// Static-scratch counterpart of [`zimmerman_sqrt_dyn`]. Requires both operand
/// lengths to be at most `N`; writes the exact root to `s` and remainder to `x`.
pub fn zimmerman_sqrt_static<const N: usize>(x: &mut [u64], s: &mut [u64]) {
    sqrt_static::<N>(x, s);
}

/// Static-scratch counterpart of [`zimmerman_sqrt_only_dyn`], with the same
/// scaling and exact-root contract. Requires both operand lengths to be at most
/// `N`. The contents of `x` on return are unspecified. Does not allocate on the heap.
pub fn zimmerman_sqrt_only_static<const N: usize>(x: &mut [u64], s: &mut [u64]) {
    sqrt_only_static::<N>(x, s);
}

/// Static-scratch counterpart of [`zimmerman_sqrt_approx_dyn`], with the same
/// scaling and at-most-one-unit upward integer error. Requires both operand
/// lengths to be at most `N`. The contents of `x` on return are unspecified.
/// Does not allocate on the heap.
pub fn zimmerman_sqrt_approx_static<const N: usize>(x: &mut [u64], s: &mut [u64]) {
    sqrt_approx_static::<N>(x, s);
}

fn sqrt_static_output<const N: usize>(x: &mut [u64], s: &mut [u64], output: SqrtOutput) {
    if one_limb_root(x, s, output) {
        return;
    }
    // assume correct size bounds
    debug_assert!(x.len() > s.len());
    debug_assert!(2 * s.len() >= x.len());
    debug_assert!(
        x.len() <= N && s.len() <= N,
        "Zimmermann sqrt operands exceed static capacity"
    );

    if s.len() < output.static_cutoff() {
        binom_sqrt_output(x, s, output);
        return;
    }

    let x_len = x.len();
    let sh = x[x_len - 1].leading_zeros() as u8 & !1_u8;
    shl_buf(x, sh);

    let mut sqr = |value: &[u64], out: &mut [u64]| sqr_static::<N>(value, out);

    zimmerman_sqrt_entry(
        x,
        s,
        output,
        &mut static_div::<N>,
        &mut sqr,
        &mut |square_len, finish| {
            let mut s_lo_sqr = [0_u64; N];
            finish(&mut s_lo_sqr[..square_len]);
        },
        None,
    );

    if output == SqrtOutput::RootRem {
        sqrt_denormalization(x, s, sh);
    } else {
        shr_buf(s, sh / 2);
    }
}

fn binom_sqrt_output(x: &mut [u64], s: &mut [u64], output: SqrtOutput) {
    if output == SqrtOutput::RootRem {
        binom_sqrt(x, s);
    } else {
        let sh = x.last().unwrap().leading_zeros() as u8 & !1_u8;
        shl_buf(x, sh);
        binom_sqrt_core(x, s);
        shr_buf(s, sh / 2);
    }
}

/// Computes the exact root and remainder of `X = value(x) * B^(2*s.len()-x.len())`,
/// where `B = 2^64`. Writes `floor(sqrt(X))` to `s` and the remainder to `x`.
/// Requires trimmed nonzero `x` and `s.len() < x.len() <= 2*s.len()`.
///
/// Dispatches to binomial sqrt when `s.len() < DYN_SQRT_REM_ZIMMERMAN_CUTOFF`, or
/// Zimmermann sqrt otherwise, using pooled dynamic scratch when needed. Each
/// output mode and allocation model has its own top-level cutoff.
pub fn sqrt_dyn(x: &mut [u64], s: &mut [u64]) {
    sqrt_dyn_output(x, s, SqrtOutput::RootRem, DYN_SQRT_WRAPPED_REM_CUTOFF);
}

/// [`sqrt_dyn`] with Zimmermann stages of at least `wrapped_cutoff` root limbs (and a modulus of
/// at most `FFT_BNM1_16BIT_CUTOFF` limbs) forming their remainder from one square mod B^m − 1
/// (`usize::MAX`: none), in place of `DYN_SQRT_WRAPPED_REM_CUTOFF`. For tests and tuning; the
/// output is the same.
#[doc(hidden)]
pub fn sqrt_dyn_at(x: &mut [u64], s: &mut [u64], wrapped_cutoff: usize) {
    sqrt_dyn_output(x, s, SqrtOutput::RootRem, wrapped_cutoff);
}

/// Computes only the exact root with the scaling, input requirements, and
/// algorithms of [`sqrt_dyn`], switching at `DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF`.
/// Writes `floor(sqrt(X))` to `s`. Without a remainder to hold, it also takes
/// `x.len() == s.len() == 1` (`X = x * B`).
/// The input `x` is used as scratch; its contents on return are unspecified.
pub fn sqrt_only_dyn(x: &mut [u64], s: &mut [u64]) {
    sqrt_dyn_output(x, s, SqrtOutput::Root, DYN_SQRT_WRAPPED_REM_CUTOFF);
}

/// [`sqrt_only_dyn`] with an explicit wrapped-remainder cutoff, as [`sqrt_dyn_at`].
#[doc(hidden)]
pub fn sqrt_only_dyn_at(x: &mut [u64], s: &mut [u64], wrapped_cutoff: usize) {
    sqrt_dyn_output(x, s, SqrtOutput::Root, wrapped_cutoff);
}

/// Computes an approximate root with the scaling, input requirements, and
/// algorithms of [`sqrt_dyn`], switching at `DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF`.
/// The result in `s` is either `floor(sqrt(X))` or `floor(sqrt(X)) + 1`; the
/// binomial path is exact. Without a remainder to hold, it also takes
/// `x.len() == s.len() == 1` (`X = x * B`).
/// This is a numeric error bound, not a guarantee of identical high limbs.
/// The input `x` is used as scratch; its contents on return are unspecified.
pub fn sqrt_approx_dyn(x: &mut [u64], s: &mut [u64]) {
    sqrt_dyn_output(x, s, SqrtOutput::ApproxRoot, DYN_SQRT_WRAPPED_REM_CUTOFF);
}

/// [`sqrt_approx_dyn`] with an explicit wrapped-remainder cutoff, as [`sqrt_dyn_at`].
#[doc(hidden)]
pub fn sqrt_approx_dyn_at(x: &mut [u64], s: &mut [u64], wrapped_cutoff: usize) {
    sqrt_dyn_output(x, s, SqrtOutput::ApproxRoot, wrapped_cutoff);
}

/// Static-scratch counterpart of [`sqrt_dyn`], switching at
/// `STATIC_SQRT_REM_ZIMMERMAN_CUTOFF`.
/// Requires both operand lengths to be at most `N`. Does not allocate on the heap.
pub fn sqrt_static<const N: usize>(x: &mut [u64], s: &mut [u64]) {
    sqrt_static_output::<N>(x, s, SqrtOutput::RootRem);
}

/// Static-scratch counterpart of [`sqrt_only_dyn`], switching at
/// `STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF`.
/// Requires both operand lengths to be at most `N`. Does not allocate on the heap.
pub fn sqrt_only_static<const N: usize>(x: &mut [u64], s: &mut [u64]) {
    sqrt_static_output::<N>(x, s, SqrtOutput::Root);
}

/// Static-scratch counterpart of [`sqrt_approx_dyn`], switching at
/// `STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF`.
/// Requires both operand lengths to be at most `N`. Does not allocate on the heap.
pub fn sqrt_approx_static<const N: usize>(x: &mut [u64], s: &mut [u64]) {
    sqrt_static_output::<N>(x, s, SqrtOutput::ApproxRoot);
}
