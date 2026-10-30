use std::arch::asm;

use crate::utils::{
    mul::*, rows_dispatch, utils::*, Rows, ScratchGuard, BZ_CUTOFF, BZ_TOP_PADDED_COST_SCALE,
    DIV_KNUTH_CUTOFF, DYN_DIV_FFT_NR_BZ_CUTOFF, DYN_DIV_KARATSUBA_FFT_NR_BZ_CUTOFF,
    DYN_DIV_KARATSUBA_NR_BZ_CUTOFF, DYN_DIV_REM_FFT_NR_BZ_CUTOFF,
    DYN_DIV_REM_KARATSUBA_FFT_NR_BZ_CUTOFF, DYN_DIV_REM_KARATSUBA_NR_BZ_CUTOFF,
    DYN_RCP_KNUTH_NR_CUTOFF, STATIC_DIV_KARATSUBA_NR_BZ_CUTOFF,
    STATIC_DIV_KARATSUBA_NTT_NR_BZ_CUTOFF, STATIC_DIV_NTT_NR_BZ_CUTOFF,
    STATIC_DIV_REM_KARATSUBA_NR_BZ_CUTOFF, STATIC_DIV_REM_KARATSUBA_NTT_NR_BZ_CUTOFF,
    STATIC_DIV_REM_NTT_NR_BZ_CUTOFF, STATIC_RCP_KNUTH_NR_CUTOFF,
};

/// Algorithm selected by dispatch or supplied to a prepared division driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DivAlg {
    Knuth,
    BZ,
    NR,
}

/// Algorithm selected by dispatch or supplied to a prepared reciprocal driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RcpAlg {
    Knuth,
    NR,
}

#[derive(Clone, Copy)]
struct DivCutoffs {
    karatsuba_transform: usize,
    karatsuba_bz_nr: f64,
    transform_bz_nr: f64,
}

const DYN_DIV: DivCutoffs = DivCutoffs {
    karatsuba_transform: DYN_DIV_KARATSUBA_FFT_NR_BZ_CUTOFF,
    karatsuba_bz_nr: DYN_DIV_KARATSUBA_NR_BZ_CUTOFF,
    transform_bz_nr: DYN_DIV_FFT_NR_BZ_CUTOFF,
};
const DYN_DIV_REM: DivCutoffs = DivCutoffs {
    karatsuba_transform: DYN_DIV_REM_KARATSUBA_FFT_NR_BZ_CUTOFF,
    karatsuba_bz_nr: DYN_DIV_REM_KARATSUBA_NR_BZ_CUTOFF,
    transform_bz_nr: DYN_DIV_REM_FFT_NR_BZ_CUTOFF,
};
const STATIC_DIV: DivCutoffs = DivCutoffs {
    karatsuba_transform: STATIC_DIV_KARATSUBA_NTT_NR_BZ_CUTOFF,
    karatsuba_bz_nr: STATIC_DIV_KARATSUBA_NR_BZ_CUTOFF,
    transform_bz_nr: STATIC_DIV_NTT_NR_BZ_CUTOFF,
};
const STATIC_DIV_REM: DivCutoffs = DivCutoffs {
    karatsuba_transform: STATIC_DIV_REM_KARATSUBA_NTT_NR_BZ_CUTOFF,
    karatsuba_bz_nr: STATIC_DIV_REM_KARATSUBA_NR_BZ_CUTOFF,
    transform_bz_nr: STATIC_DIV_REM_NTT_NR_BZ_CUTOFF,
};

// By the logical quotient width q_len + 1 and the effective divisor width.
fn div_alg_dispatch(r: &DivisionRequest, c: DivCutoffs) -> DivAlg {
    let (q, d) = (r.q_len + 1, r.d_len);
    if q == 1 || d <= DIV_KNUTH_CUTOFF {
        return DivAlg::Knuth;
    }
    let bz = if q.max(d) < c.karatsuba_transform {
        (q as f64) * c.karatsuba_bz_nr > d as f64
    } else {
        let (log_q, log_d) = ((q as f64).log2(), (d as f64).log2());
        log_q * c.transform_bz_nr > log_d * log_d
    };
    if !bz && nr_split(d + q - 1, q).is_some() {
        DivAlg::NR
    } else if d <= BZ_CUTOFF {
        // BZ would only run Knuth leaves at these widths.
        DivAlg::Knuth
    } else {
        DivAlg::BZ
    }
}

pub(crate) fn rcp_alg_dispatch(precision: usize, knuth_nr_cutoff: usize) -> RcpAlg {
    debug_assert!(precision != 0);
    if precision <= knuth_nr_cutoff {
        RcpAlg::Knuth
    } else {
        RcpAlg::NR
    }
}
/// Divides trimmed `n` by trimmed nonzero `d` using pooled dynamic scratch.
///
/// `q` must contain at least [`div_quotient_len`] limbs. With an exact-sized
/// output the additional structural quotient limb is returned; a longer
/// output absorbs it and the function returns zero.
pub fn div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let Some(r) = division_preflight(n, d, q) else {
        return 0;
    };
    div_prepared_dyn(n, d, q, r, div_alg_dispatch(&r, DYN_DIV))
}

/// Dynamic division with the same quotient contract as [`div_dyn`], leaving
/// the exact remainder in `n`.
pub fn div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) -> u64 {
    let Some(r) = division_preflight(n, d, q) else {
        return 0;
    };
    div_rem_prepared_dyn(n, d, q, r, div_alg_dispatch(&r, DYN_DIV_REM))
}

/// Static-scratch division with the same quotient contract as [`div_dyn`].
pub fn div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let Some(r) = division_preflight(n, d, q) else {
        return 0;
    };
    div_prepared_static::<N>(n, d, q, r, div_alg_dispatch(&r, STATIC_DIV))
}

/// Static-scratch division with the same quotient and remainder contract as
/// [`div_rem_dyn`].
pub fn div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) -> u64 {
    let Some(r) = division_preflight(n, d, q) else {
        return 0;
    };
    div_rem_prepared_static::<N>(n, d, q, r, div_alg_dispatch(&r, STATIC_DIV_REM))
}

/// Computes the highest `q.len()` limbs of the quotient body and returns the
/// structural overflow limb. If the complete body fits, this delegates to
/// [`div_dyn`] and follows its overflow-absorption behavior.
pub fn hi_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let requested = q.len();
    let Some(r) = division_request(n, d, q, Some(requested)) else {
        return 0;
    };
    div_prepared_dyn(n, d, q, r, div_alg_dispatch(&r, DYN_DIV))
}

/// Static-scratch counterpart of [`hi_div_dyn`].
pub fn hi_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let requested = q.len();
    let Some(r) = division_request(n, d, q, Some(requested)) else {
        return 0;
    };
    div_prepared_static::<N>(n, d, q, r, div_alg_dispatch(&r, STATIC_DIV))
}

pub fn rcp_dyn(d: &[u64], rcp: &mut [u64]) {
    if !reciprocal_preflight(d, rcp) {
        return;
    }
    let algorithm = rcp_alg_dispatch(rcp.len(), DYN_RCP_KNUTH_NR_CUTOFF);
    rcp_prepared_dyn(d, rcp, algorithm);
}
pub fn rcp_static<const N: usize>(d: &[u64], rcp: &mut [u64]) {
    if !reciprocal_preflight(d, rcp) {
        return;
    }
    let algorithm = rcp_alg_dispatch(rcp.len(), STATIC_RCP_KNUTH_NR_CUTOFF);
    rcp_prepared_static::<N>(d, rcp, algorithm);
}
/// Minimum quotient-buffer width for trimmed little-endian operands.
///
/// A quotient can have one additional structural top limb. Standard division
/// returns that limb when the output has exactly this length and absorbs it
/// when the output is longer.
pub fn div_quotient_len(n_len: usize, d_len: usize) -> usize {
    assert!(d_len != 0, "Division by zero error");
    n_len.saturating_sub(d_len)
}
#[inline]
fn absorb_div_overflow(out_tail: &mut [u64], overflow: u64) -> u64 {
    if let Some((slot, tail)) = out_tail.split_first_mut() {
        *slot = overflow;
        tail.fill(0);
        0
    } else {
        overflow
    }
}
#[derive(Clone, Copy, Debug)]
pub struct DivisionRequest {
    /// First numerator limb that can affect the requested quotient window.
    n_start: usize,
    /// First nonzero divisor limb.
    d_start: usize,
    /// Divisor width after its low zero limbs.
    d_len: usize,
    /// Quotient body length.
    q_len: usize,
    /// Shift that bit-normalizes the divisor.
    normalization_shift: u8,
}

fn division_request(
    n: &[u64],
    d: &[u64],
    q: &mut [u64],
    requested_body_len: Option<usize>,
) -> Option<DivisionRequest> {
    assert!(
        d.last().is_some_and(|&top| top != 0),
        "division requires a trimmed nonzero divisor"
    );
    assert!(
        n.last().is_none_or(|&top| top != 0),
        "division requires a trimmed numerator"
    );
    let full_body_len = div_quotient_len(n.len(), d.len());
    let requested_body_len = requested_body_len.map_or(full_body_len, |len| len.min(full_body_len));
    debug_assert!(
        q.len() >= requested_body_len,
        "out is not large enough for division"
    );
    let _ = &q[requested_body_len..];

    if n.len() < d.len() {
        q.fill(0);
        return None;
    }
    if requested_body_len == 0 && n.len() == d.len() && cmp_buf(n, d).is_lt() {
        q.fill(0);
        return None;
    }

    q[requested_body_len..].fill(0);
    let quotient_skip = full_body_len - requested_body_len;
    let d_start = d.iter().position(|&limb| limb != 0).unwrap();
    let d_len = d.len() - d_start;

    Some(DivisionRequest {
        n_start: quotient_skip + d_start,
        d_start,
        d_len,
        q_len: requested_body_len,
        normalization_shift: d[d.len() - 1].leading_zeros() as u8,
    })
}
/// Validates and windows the operands of a standard division. `q` must hold at
/// least `div_quotient_len(n.len(), d.len())` limbs.
pub fn division_preflight(n: &[u64], d: &[u64], q: &mut [u64]) -> Option<DivisionRequest> {
    division_request(n, d, q, None)
}
/// Validates a reciprocal request; an empty output requests no work.
pub fn reciprocal_preflight(d: &[u64], rcp: &[u64]) -> bool {
    if rcp.is_empty() {
        return false;
    }
    assert!(
        d.last().is_some_and(|&top| top != 0),
        "reciprocal requires a trimmed nonzero divisor"
    );
    true
}
// The divisor's high window that can affect the requested precision (one limb past it), without
// its low zero limbs.
fn reciprocal_divisor_window(d: &[u64], precision: usize) -> &[u64] {
    debug_assert!(precision != 0);
    let window = &d[d.len() - d.len().min(precision.saturating_add(1))..];
    let first_nonzero = window.iter().position(|&limb| limb != 0).unwrap();
    &window[first_nonzero..]
}
#[inline]
fn debug_assert_rigid_division_shape(n: &[u64], d: &[u64], q: &[u64]) {
    debug_assert!(!d.is_empty());
    debug_assert!(n.len() >= d.len());
    debug_assert_eq!(q.len(), n.len() - d.len());
}
#[inline]
fn debug_assert_full_division_shape(n: &[u64], d: &[u64], q: &[u64]) {
    debug_assert!(!d.is_empty());
    debug_assert!(n.len() >= d.len());
    debug_assert_eq!(q.len(), n.len() - d.len() + 1);
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RemainderMode {
    Discard,
    Restore,
}

// An NR attempt writes the q_len + 1 limb quotient: into q when it has room for the structural
// top limb, else into the temporary tmp provides (body copied back, top limb returned). None means
// NR failed with n intact, and the driver retries with BZ.
fn nr_first(
    q: &mut [u64],
    q_len: usize,
    mut attempt: impl FnMut(&mut [u64]) -> bool,
    tmp: impl FnOnce(&mut dyn FnMut(&mut [u64]) -> Option<u64>) -> Option<u64>,
) -> Option<u64> {
    let full = q_len + 1;
    if q.len() >= full {
        return attempt(&mut q[..full]).then(|| {
            q[full..].fill(0);
            0
        });
    }
    tmp(&mut |t: &mut [u64]| {
        attempt(&mut t[..full]).then(|| {
            q.copy_from_slice(&t[..q_len]);
            t[q_len]
        })
    })
}

/// Executes the request returned by preflight for these operands and output.
/// Failed NR attempts retry through BZ without modifying the input numerator.
pub fn div_prepared_dyn(
    n: &[u64],
    d: &[u64],
    q: &mut [u64],
    request: DivisionRequest,
    mut algorithm: DivAlg,
) -> u64 {
    let (n, d) = (&n[request.n_start..], &d[request.d_start..]);
    if algorithm == DivAlg::NR {
        let attempt = |q: &mut [u64]| {
            debug_assert_full_division_shape(n, d, q);
            match nr_quo_est_dyn(n, d, q) {
                NrEstimate::Exact => true,
                NrEstimate::Failed => false,
                NrEstimate::NeedsCorrection => {
                    let mut scratch = ScratchGuard::acquire();
                    let [prod, num] = scratch.get_splits([n.len(), n.len()]);
                    num.copy_from_slice(n);
                    nr_exact_correction(num, d, q, prod, &mut |a, b, o| {
                        mul_dyn(a, b, o);
                    })
                }
            }
        };
        let tmp = |f: &mut dyn FnMut(&mut [u64]) -> Option<u64>| {
            f(ScratchGuard::acquire().get(request.q_len + 1))
        };
        if let Some(overflow) = nr_first(q, request.q_len, attempt, tmp) {
            return overflow;
        }
        algorithm = DivAlg::BZ;
    }
    let (q, q_tail) = q.split_at_mut(request.q_len);
    debug_assert_rigid_division_shape(n, d, q);
    if d.len() == 1 {
        return absorb_div_overflow(q_tail, div_prim_into(n, d[0], q).0);
    }
    let shift = request.normalization_shift;
    let mut scratch = ScratchGuard::acquire();
    let [n_work, d_work] = scratch.get_splits([n.len(), if shift == 0 { 0 } else { d.len() }]);
    n_work.copy_from_slice(n);
    let d = if shift == 0 {
        d
    } else {
        shl_top_copy(d, d_work, shift);
        d_work
    };
    let overflow = match algorithm {
        DivAlg::Knuth => knuth_div_rem_core(n_work, d, q, shift, RemainderMode::Discard),
        _ => bz_div_core_dyn(n_work, d, q, shift, RemainderMode::Discard),
    };
    absorb_div_overflow(q_tail, overflow)
}

/// Executes a prepared request with the selected algorithm. A failed NR
/// attempt retries through BZ with the original prepared operands.
pub fn div_rem_prepared_dyn(
    n: &mut [u64],
    d: &[u64],
    q: &mut [u64],
    request: DivisionRequest,
    mut algorithm: DivAlg,
) -> u64 {
    let (n, d) = (&mut n[request.n_start..], &d[request.d_start..]);
    if algorithm == DivAlg::NR {
        let attempt = |q: &mut [u64]| {
            debug_assert_full_division_shape(n, d, q);
            if nr_quo_est_dyn(n, d, q) == NrEstimate::Failed {
                return false;
            }
            // Even a proven quotient needs the windowed product for the remainder.
            let w = d.len() + 1;
            let mut scratch = ScratchGuard::acquire();
            let [prod, rem] = scratch.get_splits([d.len() + q.len().min(w) - 1, w]);
            if !nr_rem_finish(n, d, q, rem, prod, &mut |a, b, o| {
                mul_dyn(a, b, o);
            }) {
                return false;
            }
            n[..w].copy_from_slice(rem);
            n[w..].fill(0);
            true
        };
        let tmp = |f: &mut dyn FnMut(&mut [u64]) -> Option<u64>| {
            f(ScratchGuard::acquire().get(request.q_len + 1))
        };
        if let Some(overflow) = nr_first(q, request.q_len, attempt, tmp) {
            return overflow;
        }
        algorithm = DivAlg::BZ;
    }
    let (q, q_tail) = q.split_at_mut(request.q_len);
    debug_assert_rigid_division_shape(n, d, q);
    if d.len() == 1 {
        let (overflow, rem) = div_prim_into(n, d[0], q);
        n[0] = rem;
        n[1..].fill(0);
        return absorb_div_overflow(q_tail, overflow);
    }
    let shift = request.normalization_shift;
    let mut scratch;
    let d = if shift == 0 {
        d
    } else {
        scratch = ScratchGuard::acquire();
        let d_work = scratch.get(d.len());
        shl_top_copy(d, d_work, shift);
        d_work
    };
    let overflow = match algorithm {
        DivAlg::Knuth => knuth_div_rem_core(n, d, q, shift, RemainderMode::Restore),
        _ => bz_div_core_dyn(n, d, q, shift, RemainderMode::Restore),
    };
    absorb_div_overflow(q_tail, overflow)
}

/// Executes a prepared request with the selected algorithm. A failed NR
/// attempt retries through BZ with the original prepared operands.
pub fn div_prepared_static<const N: usize>(
    n: &[u64],
    d: &[u64],
    q: &mut [u64],
    request: DivisionRequest,
    mut algorithm: DivAlg,
) -> u64 {
    let (n, d) = (&n[request.n_start..], &d[request.d_start..]);
    assert!(
        n.len() <= N && d.len() <= N && q[..request.q_len].len() <= N,
        "prepared division operands exceed static capacity"
    );
    if algorithm == DivAlg::NR {
        let attempt = |q: &mut [u64]| {
            debug_assert_full_division_shape(n, d, q);
            match nr_quo_est_static::<N>(n, d, q) {
                NrEstimate::Exact => true,
                NrEstimate::Failed => false,
                NrEstimate::NeedsCorrection => {
                    let mut prod = [0u64; N];
                    let mut num = [0u64; N];
                    let num = &mut num[..n.len()];
                    num.copy_from_slice(n);
                    nr_exact_correction(num, d, q, &mut prod[..n.len()], &mut |a, b, o| {
                        mul_static::<N>(a, b, o);
                    })
                }
            }
        };
        // The full quotient fits too: q_len + 1 <= n.len() <= N.
        let tmp = |f: &mut dyn FnMut(&mut [u64]) -> Option<u64>| f(&mut [0u64; N]);
        if let Some(overflow) = nr_first(q, request.q_len, attempt, tmp) {
            return overflow;
        }
        algorithm = DivAlg::BZ;
    }
    let (q, q_tail) = q.split_at_mut(request.q_len);
    debug_assert_rigid_division_shape(n, d, q);
    if d.len() == 1 {
        return absorb_div_overflow(q_tail, div_prim_into(n, d[0], q).0);
    }
    let shift = request.normalization_shift;
    let mut n_work = [0u64; N];
    n_work[..n.len()].copy_from_slice(n);
    let n = &mut n_work[..n.len()];
    let mut d_work;
    let d = if shift == 0 {
        d
    } else {
        d_work = [0u64; N];
        shl_top_copy(d, &mut d_work[..d.len()], shift);
        &d_work[..d.len()]
    };
    let overflow = match algorithm {
        DivAlg::Knuth => knuth_div_rem_core(n, d, q, shift, RemainderMode::Discard),
        _ => bz_div_core_static::<N>(n, d, q, shift, RemainderMode::Discard),
    };
    absorb_div_overflow(q_tail, overflow)
}

/// Executes a prepared request with the selected algorithm. A failed NR
/// attempt retries through BZ with the original prepared operands.
pub fn div_rem_prepared_static<const N: usize>(
    n: &mut [u64],
    d: &[u64],
    q: &mut [u64],
    request: DivisionRequest,
    mut algorithm: DivAlg,
) -> u64 {
    let (n, d) = (&mut n[request.n_start..], &d[request.d_start..]);
    assert!(
        n.len() <= N && d.len() <= N && q[..request.q_len].len() <= N,
        "prepared division operands exceed static capacity"
    );
    if algorithm == DivAlg::NR {
        let attempt = |q: &mut [u64]| {
            debug_assert_full_division_shape(n, d, q);
            if nr_quo_est_static::<N>(n, d, q) == NrEstimate::Failed {
                return false;
            }
            // Even a proven quotient needs the windowed product for the remainder.
            let w = d.len() + 1;
            let mut prod = [0u64; N];
            let mut rem = [0u64; N];
            let (prod, rem) = (&mut prod[..d.len() + q.len().min(w) - 1], &mut rem[..w]);
            if !nr_rem_finish(n, d, q, rem, prod, &mut |a, b, o| {
                mul_static::<N>(a, b, o);
            }) {
                return false;
            }
            n[..w].copy_from_slice(rem);
            n[w..].fill(0);
            true
        };
        // The full quotient fits too: q_len + 1 <= n.len() <= N.
        let tmp = |f: &mut dyn FnMut(&mut [u64]) -> Option<u64>| f(&mut [0u64; N]);
        if let Some(overflow) = nr_first(q, request.q_len, attempt, tmp) {
            return overflow;
        }
        algorithm = DivAlg::BZ;
    }
    let (q, q_tail) = q.split_at_mut(request.q_len);
    debug_assert_rigid_division_shape(n, d, q);
    if d.len() == 1 {
        let (overflow, rem) = div_prim_into(n, d[0], q);
        n[0] = rem;
        n[1..].fill(0);
        return absorb_div_overflow(q_tail, overflow);
    }
    let shift = request.normalization_shift;
    let mut d_work;
    let d = if shift == 0 {
        d
    } else {
        d_work = [0u64; N];
        shl_top_copy(d, &mut d_work[..d.len()], shift);
        &d_work[..d.len()]
    };
    let overflow = match algorithm {
        DivAlg::Knuth => knuth_div_rem_core(n, d, q, shift, RemainderMode::Restore),
        _ => bz_div_core_static::<N>(n, d, q, shift, RemainderMode::Restore),
    };
    absorb_div_overflow(q_tail, overflow)
}

/// Executes a nonempty reciprocal request after [`reciprocal_preflight`]; Knuth is
/// both the classical path and NR's retry.
pub fn rcp_prepared_dyn(d: &[u64], rcp: &mut [u64], algorithm: RcpAlg) {
    let d = reciprocal_divisor_window(d, rcp.len());
    if algorithm == RcpAlg::NR && nr_rcp_attempt_dyn(d, rcp) {
        return;
    }
    if d.len() == 1 {
        rcp_prim(d[0], rcp);
        return;
    }

    let sh = d[d.len() - 1].leading_zeros() as u8;
    let mut scratch = ScratchGuard::acquire();
    if sh == 0 {
        let win = scratch.get(d.len());
        knuth_rcp_normalized(d, rcp, win, 1);
    } else {
        let [d_work, win] = scratch.get_splits([d.len(), d.len()]);
        shl_top_copy(d, d_work, sh);
        knuth_rcp_normalized(d_work, rcp, win, 1u64 << sh);
    }
}
/// Static counterpart of [`rcp_prepared_dyn`]; `d` and `rcp` must each fit N limbs.
pub fn rcp_prepared_static<const N: usize>(d: &[u64], rcp: &mut [u64], algorithm: RcpAlg) {
    assert!(d.len() <= N, "reciprocal divisor exceeds static capacity");
    assert!(rcp.len() <= N, "reciprocal output exceeds static capacity");
    let d = reciprocal_divisor_window(d, rcp.len());
    if algorithm == RcpAlg::NR && nr_rcp_attempt_static::<N>(d, rcp) {
        return;
    }
    if d.len() == 1 {
        rcp_prim(d[0], rcp);
        return;
    }

    let sh = d[d.len() - 1].leading_zeros() as u8;
    let mut win = [0u64; N];
    if sh == 0 {
        knuth_rcp_normalized(d, rcp, &mut win[..d.len()], 1);
    } else {
        let mut d_work = [0u64; N];
        shl_top_copy(d, &mut d_work[..d.len()], sh);
        knuth_rcp_normalized(&d_work[..d.len()], rcp, &mut win[..d.len()], 1u64 << sh);
    }
}
// Algorithm 3's 11-bit first estimates, floor((2^19 - 3 2^8) / d9) for d's top 9 bits d9.
static RECIPROCAL_TABLE: [u16; 256] = {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = (((1u64 << 19) - 3 * (1 << 8)) / (i as u64 + 256)) as u16;
        i += 1;
    }
    table
};

// floor((B^2 - 1) / d) - B for a normalized d, without a division: Möller and Granlund's
// algorithm 3 (the table, Newton steps to 22, 35 and 64 bits, a final correction), mod B where
// the paper is.
pub fn reciprocal_2_1(d: u64) -> u64 {
    debug_assert!(d >> 63 == 1);
    let d0 = d & 1;
    let d40 = (d >> 24) + 1;
    let d63 = (d >> 1) + d0;
    let v0 = RECIPROCAL_TABLE[(d >> 55) as usize - 256] as u64;
    let v1 = (v0 << 11)
        .wrapping_sub(v0.wrapping_mul(v0).wrapping_mul(d40) >> 40)
        .wrapping_sub(1);
    let e1 = (1u64 << 60).wrapping_sub(v1.wrapping_mul(d40));
    let v2 = (v1 << 13).wrapping_add(v1.wrapping_mul(e1) >> 47);
    let e2 = ((v2 >> 1) & d0.wrapping_neg()).wrapping_sub(v2.wrapping_mul(d63));
    let v3 = (v2 << 31).wrapping_add(((v2 as u128 * e2 as u128) >> 65) as u64);
    let t = v3 as u128 * d as u128 + d as u128;
    v3.wrapping_sub((t >> 64) as u64).wrapping_sub(d)
}
// floor((B^3 - 1) / (d1 B + d0)) - B for a normalized d1 d0: d1's 2/1 reciprocal lowered by up
// to 2 for d0 and up to 2 more for v d0 (algorithm 6), as masks: branches mispredict when the
// divisor changes between calls.
pub fn reciprocal_3_2(d1: u64, d0: u64) -> u64 {
    let v = reciprocal_2_1(d1);
    // d1 v + d0 carried: v -= 1, and once more (p -= d1) if p >= d1; then p -= d1.
    let (p, c1) = d1.wrapping_mul(v).overflowing_add(d0);
    let ge1 = c1 & (p >= d1);
    let v = v.wrapping_sub(c1 as u64 + ge1 as u64);
    let p = p
        .wrapping_sub(d1 & (ge1 as u64).wrapping_neg())
        .wrapping_sub(d1 & (c1 as u64).wrapping_neg());
    // p + hi(v d0) carried: v -= 1, and once more if (p lo(v d0)) >= (d1 d0).
    let (t1, t0) = unsafe { mul_u64_asm(v, d0) };
    let (p, c2) = p.overflowing_add(t1);
    let ge2 = c2 & ((p > d1) | ((p == d1) & (t0 >= d0)));
    v.wrapping_sub(c2 as u64 + ge2 as u64)
}
// (u2 u1 u0) / (d1 d0) for (u2 u1) < (d1 d0), with v = reciprocal_3_2(d1, d0): the quotient limb
// and the remainder's two limbs (algorithm 5). One asm block: each Knuth limb waits on the last
// one's step.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn div_3_2_limb(u2: u64, u1: u64, u0: u64, d1: u64, d0: u64, v: u64) -> (u64, u64, u64) {
    let (q, r1, r0): (u64, u64, u64);
    unsafe {
        asm!(
            // q q0 = v u2 + (u2 u1), in rdx rax.
            "mov rax, {v}",
            "mul {r1}",
            "add rax, {r0}",
            "adc rdx, {r1}",
            // r0 u0 = (u1 - q d1) u0 - d - q d0, the remainder for q + 1
            "sub {u0}, {d0}",
            "sbb {r0}, {d1}",
            "mov {q}, rdx",
            "mov {t}, {d1}",
            "imul {t}, rdx",
            "sub {r0}, {t}",
            "mov {t}, rax",
            "mov rax, {d0}",
            "mul rdx",
            "sub {u0}, rax",
            "sbb {r0}, rdx",
            // r1 rax = r0 u0 + d, kept when r0 >= q0; otherwise r0 u0 and q + 1 (q + CF).
            "mov {r1}, {r0}",
            "mov rax, {u0}",
            "add rax, {d0}",
            "adc {r1}, {d1}",
            "cmp {r0}, {t}",
            "cmovb {r1}, {r0}",
            "cmovb rax, {u0}",
            "adc {q}, 0",
            "mov {r0}, rax",
            // r1 r0 >= d (rare): q += 1, r1 r0 -= d
            "cmp {r0}, {d0}",
            "mov {t}, {r1}",
            "sbb {t}, {d1}",
            "jb 2f",
            "sub {r0}, {d0}",
            "sbb {r1}, {d1}",
            "add {q}, 1",
            "2:",
            r1 = inout(reg) u2 => r1,
            r0 = inout(reg) u1 => r0,
            u0 = inout(reg) u0 => _,
            v = in(reg) v,
            d1 = in(reg) d1,
            d0 = in(reg) d0,
            q = out(reg) q,
            t = out(reg) _,
            out("rax") _,
            out("rdx") _,
            options(pure, nomem, nostack),
        );
    }
    (q, r1, r0)
}
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn div_3_2_limb(u2: u64, u1: u64, u0: u64, d1: u64, d0: u64, v: u64) -> (u64, u64, u64) {
    let (q, r1, r0): (u64, u64, u64);
    unsafe {
        asm!(
            // q q0 = v u2 + (u2 u1)
            "mul {q0}, {v}, {r1}",
            "umulh {q}, {v}, {r1}",
            "adds {q0}, {q0}, {r0}",
            "adc {q}, {q}, {r1}",
            // r0 u0 = (u1 - q d1) u0 - d - q d0, the remainder for q + 1
            "msub {r0}, {q}, {d1}, {r0}",
            "subs {u0}, {u0}, {d0}",
            "sbc {r0}, {r0}, {d1}",
            "mul {t0}, {q}, {d0}",
            "umulh {t1}, {q}, {d0}",
            "subs {u0}, {u0}, {t0}",
            "sbc {r0}, {r0}, {t1}",
            // t1 t0 = r0 u0 + d, kept when r0 >= q0; otherwise r0 u0 and q + 1.
            "adds {t0}, {u0}, {d0}",
            "adc {t1}, {r0}, {d1}",
            "cmp {r0}, {q0}",
            "csel {r1}, {r0}, {t1}, lo",
            "csel {r0}, {u0}, {t0}, lo",
            "cinc {q}, {q}, lo",
            // r1 r0 >= d (rare): q += 1, r1 r0 -= d
            "cmp {r0}, {d0}",
            "sbcs xzr, {r1}, {d1}",
            "b.lo 2f",
            "subs {r0}, {r0}, {d0}",
            "sbc {r1}, {r1}, {d1}",
            "add {q}, {q}, #1",
            "2:",
            r1 = inout(reg) u2 => r1,
            r0 = inout(reg) u1 => r0,
            u0 = inout(reg) u0 => _,
            v = in(reg) v,
            d1 = in(reg) d1,
            d0 = in(reg) d0,
            q = out(reg) q,
            q0 = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    (q, r1, r0)
}
// One quotient limb of (of win) / d for a normalized d with (of win) < B d, leaving the remainder
// in win and of zero: a 3/2 step on the top three limbs against d1 d0 (v = reciprocal_3_2), and
// a multiply-subtract on the d.len() - 2 below, whose borrow comes off the step's remainder. The
// step's limb is at most one too big (rare add-back). Top limbs equal to d1 d0 give B - 1.
rows_dispatch!(
    pub fn knuth_est(win: &mut [u64], of: &mut u64, d: &[u64], d1: u64, d0: u64, v: u64) -> u64
    = knuth_est_rows
);

#[inline(always)]
fn knuth_est_rows<K: Rows>(
    win: &mut [u64],
    of: &mut u64,
    d: &[u64],
    d1: u64,
    d0: u64,
    v: u64,
) -> u64 {
    knuth_step::<K, false>(win, of, d, d1, d0, v)
}
// knuth_est's step; with COMP, of and win hold their complements: !x + q d = !(x - q d) over the
// low limbs with the borrow as the row's top limb, so the multiply-subtract is the add row (the
// ADX sub row complements every product limb). The division loops complement on the way in and
// out.
#[inline(always)]
fn knuth_step<K: Rows, const COMP: bool>(
    win: &mut [u64],
    of: &mut u64,
    d: &[u64],
    d1: u64,
    d0: u64,
    v: u64,
) -> u64 {
    let n = d.len();
    debug_assert_eq!(win.len(), n);
    let flip = |x: u64| if COMP { !x } else { x };
    let (n2, n1, n0) = (flip(*of), flip(win[n - 1]), flip(win[n - 2]));
    debug_assert!(n2 < d1 || (n2 == d1 && n1 <= d0));
    if n2 == d1 && n1 == d0 {
        let borrow = unsafe {
            if COMP {
                K::add(u64::MAX, d, win)
            } else {
                K::sub(u64::MAX, d, win)
            }
        };
        debug_assert_eq!(borrow, n2);
        *of = flip(0);
        return u64::MAX;
    }

    let (mut q, r1, r0) = div_3_2_limb(n2, n1, n0, d1, d0, v);
    let (lo, top) = win.split_at_mut(n - 2);
    let cy = unsafe {
        if COMP {
            K::add(q, &d[..n - 2], lo)
        } else {
            K::sub(q, &d[..n - 2], lo)
        }
    };
    let (mut r0, b0) = r0.overflowing_sub(cy);
    let (mut r1, b1) = r1.overflowing_sub(b0 as u64);
    if b1 {
        // Add d back: on !x the low limbs take !x - d = !(x + d), whose borrow is x + d's carry.
        q -= 1;
        let c = if COMP {
            sub_buf(lo, &d[..n - 2])
        } else {
            add_buf(lo, &d[..n - 2])
        };
        let (t, c1) = r0.overflowing_add(d0);
        let (t, c2) = t.overflowing_add(c as u64);
        r0 = t;
        r1 = r1.wrapping_add(d1).wrapping_add((c1 | c2) as u64);
    }
    top[0] = flip(r0);
    top[1] = flip(r1);
    *of = flip(0);
    q
}
/// Knuth division by a normalized multi-limb `d` with the numerator's overflow
/// limb `of`. `out` needs at least `n.len() - d.len()` limbs; the possible
/// additional quotient limb is returned or absorbed into a longer output.
pub fn div_buf_of(n: &mut [u64], of: &mut u64, d: &[u64], out: &mut [u64]) -> u64 {
    let q_len = n.len() - d.len();
    debug_assert!(out.len() >= q_len, "out is not large enough for division");
    let (out_body, out_tail) = out.split_at_mut(q_len);
    let v = reciprocal_3_2(d[d.len() - 1], d[d.len() - 2]);
    let overflow = knuth_loop(n, Some(of), d, out_body, v);
    absorb_div_overflow(out_tail, overflow)
}

/// Divide by normalized `d`, assuming the high divisor-width window of `n` is below `d`.
/// `out` must have exactly `n.len() - d.len()` limbs, and `d.len() >= 2`.
pub fn div_buf_body(n: &mut [u64], d: &[u64], out: &mut [u64]) {
    if n.len() > d.len() {
        let v = reciprocal_3_2(d[d.len() - 1], d[d.len() - 2]);
        knuth_loop(n, None, d, out, v);
    }
}

// Knuth's quotient limbs on a complemented n: (of n[q_len..]) first when of is given, then one per
// limb below. The remainder ends in n[..d.len()] with zeros above it, and of zero.
rows_dispatch!(
    fn knuth_loop(n: &mut [u64], of: Option<&mut u64>, d: &[u64], out: &mut [u64], v: u64) -> u64
    = knuth_loop_rows
);

#[inline(always)]
fn knuth_loop_rows<K: Rows>(
    n: &mut [u64],
    of: Option<&mut u64>,
    d: &[u64],
    out: &mut [u64],
    v: u64,
) -> u64 {
    let (dn, q_len) = (d.len(), n.len() - d.len());
    debug_assert_eq!(out.len(), q_len);
    let (d1, d0) = (d[dn - 1], d[dn - 2]);
    n.iter_mut().for_each(|x| *x = !*x);
    let top = of.map_or(0, |of| {
        *of = !*of;
        let q = knuth_step::<K, true>(&mut n[q_len..], of, d, d1, d0, v);
        *of = 0;
        q
    });
    for i in (0..q_len).rev() {
        let (win, of) = n[i..].split_at_mut(dn);
        out[i] = knuth_step::<K, true>(win, &mut of[0], d, d1, d0, v);
    }
    n[..dn].iter_mut().for_each(|x| *x = !*x);
    n[dn..].fill(0);
    top
}

fn knuth_div_rem_core(
    n: &mut [u64],
    d: &[u64],
    q: &mut [u64],
    normalization_shift: u8,
    remainder_mode: RemainderMode,
) -> u64 {
    debug_assert_rigid_division_shape(n, d, q);
    debug_assert!(d.len() >= 2);
    debug_assert_eq!(d[d.len() - 1].leading_zeros(), 0);

    let mut overflow = shl_buf(n, normalization_shift);
    let quotient_overflow = div_buf_of(n, &mut overflow, d, q);
    debug_assert_eq!(
        overflow, 0,
        "Knuth division left an overflow remainder limb"
    );
    if remainder_mode == RemainderMode::Restore {
        debug_assert!(n[d.len()..].iter().all(|&limb| limb == 0));
        shr_buf(&mut n[..d.len()], normalization_shift);
    }
    quotient_overflow
}
// n / d for a one-limb d: the low n.len() - 1 quotient limbs into q, returning the top one and
// the remainder. div_prim's loop runs over q (n's low limbs) from the top limb's remainder.
fn div_prim_into(n: &[u64], d: u64, q: &mut [u64]) -> (u64, u64) {
    debug_assert_eq!(q.len() + 1, n.len());
    let top = n[n.len() - 1];
    let (overflow, r) = (top / d, top % d);
    q.copy_from_slice(&n[..q.len()]);
    if q.is_empty() || d == 1 {
        return (overflow, r);
    }
    let s = d.leading_zeros();
    let dn = d << s;
    let r = (r << s) | ((q[q.len() - 1] >> 1) >> (63 - s));
    let rem = unsafe { div_prim_asm(q, dn, reciprocal_2_1(dn), s, r) };
    (overflow, rem >> s)
}
pub fn div_3_2(
    n: &mut [u64],
    d: &[u64],
    d_lo_len: usize,
    q: &mut [u64],
    scratch: &mut [u64],
    mul_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64]),
) {
    let q_len = q.len();
    let (d_lo, d_hi) = d.split_at(d_lo_len);
    if cmp_buf(&n[d.len()..], d_hi) == std::cmp::Ordering::Less {
        div_2_1(&mut n[d_lo_len..], d_hi, q, scratch, mul_alg);
    } else {
        q.fill(u64::MAX);
        sub_buf(&mut n[d.len()..], d_hi);
        add_buf(&mut n[d_lo_len..], d_hi);
    }

    let m = &mut scratch[..q_len + d_lo_len];
    *m.last_mut().unwrap() = 0;
    mul_alg(q, d_lo, m);
    if sub_buf(n, m) {
        dec_buf(q);
        if !add_buf(n, d) {
            dec_buf(q);
            add_buf(n, d);
        }
    }
}
pub fn div_2_1(
    n: &mut [u64],
    d: &[u64],
    q: &mut [u64],
    scratch: &mut [u64],
    mul_alg: &mut dyn FnMut(&mut [u64], &[u64], &mut [u64]),
) {
    let dlen = d.len();
    if dlen <= BZ_CUTOFF {
        // The 2/1 invariant already proves the structural top quotient limb is
        // zero, so do not pay for estimating it in every recursive base case.
        debug_assert!(cmp_buf(&n[dlen..], d).is_lt());
        div_buf_body(n, d, q);
        return;
    }

    let lo = dlen / 2;
    let hi = dlen - lo;
    let (q_lo, q_hi) = q.split_at_mut(lo);

    div_3_2(&mut n[lo..], d, lo, q_hi, scratch, mul_alg);
    div_3_2(&mut n[..dlen + lo], d, hi, q_lo, scratch, mul_alg);
}
pub fn bz_top_block_knuth_work(dlen: usize, qlen: usize) -> f64 {
    (dlen as f64) * (qlen as f64)
}
pub fn bz_2_1_cost(d: usize) -> f64 {
    const SCHOOL_TO_KARATSUBA: f64 = 3.394147384;
    const KARATSUBA_TO_FFT: f64 = 2.733850808;
    match dyn_dispatch(d, d) {
        MulDynDispatch::Prim | MulDynDispatch::Prim2 | MulDynDispatch::School => {
            d as f64 * d as f64
        }
        MulDynDispatch::Karatsuba => SCHOOL_TO_KARATSUBA * (d as f64).powf(1.5849625007),
        MulDynDispatch::FFT | MulDynDispatch::NTT => {
            KARATSUBA_TO_FFT * (d as f64) * (d as f64).log2() * (d as f64).log2()
        }
    }
}
pub fn use_bz_for_top_block(dlen: usize, qlen: usize) -> bool {
    if dlen <= BZ_CUTOFF {
        return false;
    }
    if qlen == 0 {
        return false;
    }

    BZ_TOP_PADDED_COST_SCALE * bz_2_1_cost(dlen) <= bz_top_block_knuth_work(dlen, qlen)
}
// Burnikel-Ziegler on a normalized-divisor request: the top block (Knuth, or with top_n of
// 2 d.len() limbs, BZ's 2/1 on the block padded to full size), then the 2/1 blocks below.
// scratch holds 2 d.len() limbs with top_n, else d.len().
fn bz_div_core(
    n: &mut [u64],
    d: &[u64],
    out: &mut [u64],
    shift: u8,
    mode: RemainderMode,
    top_n: Option<&mut [u64]>,
    scratch: &mut [u64],
    mut mul: impl FnMut(&mut [u64], &[u64], &mut [u64]),
) -> u64 {
    debug_assert_rigid_division_shape(n, d, out);
    let dlen = d.len();
    debug_assert!(dlen >= 2);
    debug_assert_eq!(d[dlen - 1].leading_zeros(), 0);
    let mut last_n = shl_buf(n, shift);
    let blocks = (n.len() - dlen) / dlen;
    let init_idx = dlen * blocks;
    let init_len = n.len() - init_idx;
    let overflow = if let Some(top_n) = top_n {
        let body = init_len - dlen;
        let (q_tmp, div_scratch) = scratch[..2 * dlen].split_at_mut(dlen);
        top_n[..init_len].copy_from_slice(&n[init_idx..]);
        top_n[init_len] = last_n;
        top_n[init_len + 1..].fill(0);
        div_2_1(top_n, d, q_tmp, div_scratch, &mut mul);
        out[init_idx..init_idx + body].copy_from_slice(&q_tmp[..body]);
        n[init_idx..init_idx + dlen].copy_from_slice(&top_n[..dlen]);
        n[init_idx + dlen..].fill(0);
        q_tmp[body]
    } else {
        div_buf_of(&mut n[init_idx..], &mut last_n, d, &mut out[init_idx..])
    };
    for i in (0..blocks).rev() {
        let idx = dlen * i;
        div_2_1(
            &mut n[idx..idx + 2 * dlen],
            d,
            &mut out[idx..idx + dlen],
            &mut scratch[..dlen],
            &mut mul,
        );
    }
    if mode == RemainderMode::Restore {
        debug_assert!(n[dlen..].iter().all(|&limb| limb == 0));
        shr_buf(&mut n[..dlen], shift);
    }
    overflow
}

fn bz_div_core_dyn(
    n: &mut [u64],
    d: &[u64],
    out: &mut [u64],
    shift: u8,
    mode: RemainderMode,
) -> u64 {
    let dlen = d.len();
    let top = use_bz_for_top_block(dlen, (n.len() - dlen) % dlen + 1);
    let mut scratch = ScratchGuard::acquire();
    let [top_n, s] = scratch.get_splits([if top { 2 * dlen } else { 0 }, 2 * dlen]);
    let mul = |a: &mut [u64], b: &[u64], o: &mut [u64]| {
        mul_dyn(a, b, o);
    };
    bz_div_core(n, d, out, shift, mode, top.then_some(top_n), s, mul)
}

fn bz_div_core_static<const N: usize>(
    n: &mut [u64],
    d: &[u64],
    out: &mut [u64],
    shift: u8,
    mode: RemainderMode,
) -> u64 {
    let dlen = d.len();
    let mul = |a: &mut [u64], b: &[u64], o: &mut [u64]| {
        mul_static::<N>(a, b, o);
    };
    let mut scratch = [0u64; N];
    if dlen <= N / 2 && use_bz_for_top_block(dlen, (n.len() - dlen) % dlen + 1) {
        let mut top_n = [0u64; N];
        let (top_n, s) = (&mut top_n[..2 * dlen], &mut scratch[..2 * dlen]);
        bz_div_core(n, d, out, shift, mode, Some(top_n), s, mul)
    } else {
        bz_div_core(n, d, out, shift, mode, None, &mut scratch[..dlen], mul)
    }
}

pub fn nr_rcp_schedule(p_target: usize, sizes: &mut [usize; 64]) -> usize {
    let mut steps = 0;
    let mut q = p_target;
    while q > 1 {
        sizes[steps] = q;
        steps += 1;
        q = (q >> 1) + (q & 1);
    }
    steps
}
// The signed error band d y - num around the cancellation point (num[i] aligned with e[i]; an
// empty num is the reciprocal's B^k), extended up while its limbs are neither 0 nor all ones.
// None past cap.
fn nr_err_band(
    d: &[u64],
    y: &[u64],
    num: &[u64],
    e: &mut [u64],
    p: usize,
    cap: usize,
    mid: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> (u64, u64),
) -> Option<(bool, usize)> {
    let sub_num_limb = |val: u64, idx: usize, borrow: &mut u64| {
        let (v, b1) = val.overflowing_sub(num.get(idx).copied().unwrap_or(0));
        let (v, b2) = v.overflowing_sub(*borrow);
        *borrow = (b1 | b2) as u64;
        v
    };
    debug_assert_eq!(d.len(), 2 * p + 1);
    debug_assert_eq!(y.len(), p + 1);
    let mut e_idx = p + 1;
    let (mut acc0, mut acc1) = mid(d, y, &mut e[..e_idx]);
    let mut acc2 = 0;
    let mut borrow = 0;
    if !num.is_empty() {
        borrow = sub_buf(&mut e[..e_idx], &num[..e_idx]) as u64;
    }
    let mut val = mul_elem(d, y, p + e_idx, &mut acc0, &mut acc1, &mut acc2);
    if !num.is_empty() {
        val = sub_num_limb(val, e_idx, &mut borrow);
    }
    while val != 0 && val != u64::MAX {
        e[e_idx] = val;
        e_idx += 1;
        if e_idx > cap {
            return None;
        }
        val = mul_elem(d, y, p + e_idx, &mut acc0, &mut acc1, &mut acc2);
        if !num.is_empty() {
            val = sub_num_limb(val, e_idx, &mut borrow);
        }
    }
    Some((val == u64::MAX, e_idx))
}
// Knuth long division of seed B^k by a bit-normalized d, through a sliding d.len()-limb window.
pub fn knuth_rcp_normalized(d: &[u64], rcp: &mut [u64], win: &mut [u64], seed: u64) {
    debug_assert!(d.len() >= 2);
    debug_assert_eq!(win.len(), d.len());

    win.fill(0);
    let v = reciprocal_3_2(d[d.len() - 1], d[d.len() - 2]);
    knuth_rcp_loop(d, rcp, win, seed, v);
}
rows_dispatch!(
    fn knuth_rcp_loop(d: &[u64], rcp: &mut [u64], win: &mut [u64], seed: u64, v: u64)
    = knuth_rcp_rows
);

// On the complemented window, as knuth_loop: the zero limbs shifted in are !0.
#[inline(always)]
fn knuth_rcp_rows<K: Rows>(d: &[u64], rcp: &mut [u64], win: &mut [u64], seed: u64, v: u64) {
    let mut of = !seed;
    let d1 = d[d.len() - 1];
    let d0 = d[d.len() - 2];
    win.iter_mut().for_each(|x| *x = !*x);
    for q in rcp.iter_mut().rev() {
        *q = knuth_step::<K, true>(win, &mut of, d, d1, d0, v);
        of = win[win.len() - 1];
        win.copy_within(0..win.len() - 1, 1);
        win[0] = u64::MAX;
    }
    win.iter_mut().for_each(|x| *x = !*x);
}
// Refines rcp from precision p to 2p, or 2p - 1 when trunc drops the correction's low limb.
pub fn nr_refine_rcp(
    d: &[u64],
    rcp: &mut [u64],
    err: &mut [u64],
    cor: &mut [u64],
    p: usize,
    trunc: bool,
    mid: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> (u64, u64),
    short: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> u64,
) -> bool {
    let skip = trunc as usize;
    debug_assert!(rcp.len() + skip == 2 * p + 1);

    let x = end_ref(rcp, p + 1);
    let cap = (2 * p + 1).min(err.len() - 1);
    let Some((neg, e_len)) = nr_err_band(d, x, &[], err, p, cap, mid) else {
        return false;
    };

    let extra = e_len - p - 1;
    let c = &mut cor[..e_len];
    c[e_len - 1] = short(x, &err[..e_len], &mut c[..e_len - 1]);
    if neg {
        sub_buf(&mut c[extra..], x);
        twos_comp(c);
        add_buf(rcp, &c[skip..]);
    } else {
        sub_buf(rcp, &c[skip..]);
    }

    true
}
// Knuth seed, then the ceiling-halving refine chain over a normalized divisor window. False when
// a band bails (rcp then only partly refined).
fn nr_rcp_chain(
    d_work: &[u64],
    rcp: &mut [u64],
    err: &mut [u64],
    cor: &mut [u64],
    seed_n: &mut [u64],
    sizes: &[usize],
    mid: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> (u64, u64),
    short: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> u64,
) -> bool {
    // Precision-1 seed: floor(B^(seed_r + 2) / top 3 limbs of d) + 1.
    let seed_r = rcp.len().min(2);
    let seed_n = &mut seed_n[..seed_r + 2];
    seed_n.fill(0);
    let mut seed_of = 1;
    div_buf_of(
        seed_n,
        &mut seed_of,
        end_ref(d_work, 3),
        end_mut(rcp, seed_r),
    );
    inc_buf(end_mut(rcp, seed_r));

    let mut p = 1;
    for &q in sizes.iter().rev() {
        if !nr_refine_rcp(
            end_ref(d_work, 2 * p + 1),
            end_mut(rcp, q + 1),
            err,
            cor,
            p,
            q & 1 == 1,
            mid,
            short,
        ) {
            return false;
        }
        p = q;
    }
    true
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NrEstimate {
    Exact,
    NeedsCorrection,
    Failed,
}

const GUARD: usize = 3;
const NR_DIV_MAX_CORRECTIONS: usize = 32;

fn nr_exact_correction(
    n: &mut [u64],
    d: &[u64],
    q: &mut [u64],
    prod: &mut [u64],
    mul: &mut dyn FnMut(&[u64], &[u64], &mut [u64]),
) -> bool {
    mul(d, q, prod);
    if sub_buf(n, prod) {
        let mut corrections = 0;
        loop {
            if corrections == NR_DIV_MAX_CORRECTIONS || dec_buf(q) {
                return false;
            }
            if add_buf(n, d) {
                break;
            }
            corrections += 1;
        }
        return true;
    }
    let mut corrections = 0;
    while cmp_buf(n, d).is_ge() {
        if corrections == NR_DIV_MAX_CORRECTIONS || inc_buf(q) {
            return false;
        }
        sub_buf(n, d);
        corrections += 1;
    }
    true
}
// Karp-Markstein refinement from x_norm, the reciprocal of d 2^sh (x, shifted down here, is d's):
// the guarded high product hq (its top h + 1 limbs anchored at digit l - 1), the band
// err = d q_hi - n, and q from the correction x_norm err. The band uses the normalized windows
// d_top and n_top: it drops the carries of the columns below it (up to about (h + 1) B), which
// dividing by an unnormalized top limb would scale past the guard. Failed on a rejected band or a
// wrapped quotient; otherwise the top guard limb says whether exact correction is needed.
fn nr_refine_quo(
    n: &[u64],
    d: &[u64],
    sh: u8,
    x: &mut [u64],
    x_norm: &mut [u64],
    d_top: &mut [u64],
    n_top: &mut [u64],
    hq: &mut [u64],
    err: &mut [u64],
    corr: &mut [u64],
    q: &mut [u64],
    h: usize,
    l: usize,
    mid: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> (u64, u64),
    short: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> u64,
) -> NrEstimate {
    x_norm.copy_from_slice(x);
    if shl_buf(x, sh) != 0 {
        return NrEstimate::Failed;
    }
    shl_top_copy(d, d_top, sh);
    shl_top_copy(n, n_top, sh);
    hq[h + 2] = short(n, x, &mut hq[..h + 2]);
    let y = end_ref(hq, h + 1);
    let Some((neg, e_len)) = nr_err_band(d_top, y, n_top, err, h, h + 3, mid) else {
        return NrEstimate::Failed;
    };

    // corr[i] is quotient digit i - GUARD; the window top tracks err's top so
    // k floats with e_len while the digit alignment stays fixed.
    let k = e_len + l - h + 1;
    let c = &mut corr[..k + 1];
    c[k] = short(x_norm, &err[..e_len], &mut c[..k]);
    if neg {
        sub_buf(&mut c[1..], end_ref(x_norm, k));
        twos_comp(c);
    }

    q[..l - 1].fill(0);
    q[l - 1] = hq[2];
    q[l..].copy_from_slice(&hq[3..]);
    let wrapped = if neg {
        add_buf(q, &c[GUARD..])
    } else {
        sub_buf(q, &c[GUARD..])
    };
    if wrapped {
        return NrEstimate::Failed;
    }

    // Truncated products can be off by a few ulps at the second guard limb, so
    // only the top guard limb proves the floor: an estimate within B^-1 of an
    // integer boundary is ambiguous.
    let guard = c[GUARD - 1];
    if guard == 0 || guard == u64::MAX {
        NrEstimate::NeedsCorrection
    } else {
        NrEstimate::Exact
    }
}
// NR's split of a q_len-limb quotient into h limbs from the reciprocal and l below, if the shape
// leaves room for the guard limbs and the quotient's error band.
fn nr_split(n_len: usize, q_len: usize) -> Option<(usize, usize)> {
    let h = q_len / 2 + 1;
    (q_len >= 2 * GUARD + 2 && n_len >= 2 * h + 1).then(|| (h, q_len - h))
}

// Newton division's estimate: a half-precision reciprocal and one quotient refinement put q within
// one of the true quotient.
fn nr_quo_est_dyn(n: &[u64], d: &[u64], q: &mut [u64]) -> NrEstimate {
    let Some((h, l)) = nr_split(n.len(), q.len()) else {
        return NrEstimate::Failed;
    };
    let x_len = h + 3;
    let mut sizes = [0usize; 64];
    let steps = nr_rcp_schedule(x_len - 1, &mut sizes);
    // The widest refinement reads 2p + 1 divisor limbs and may extend its band one past that.
    let pen_p = if steps >= 2 { sizes[1] } else { 1 };
    let d_len = 2 * pen_p + 1;
    let mut scratch = ScratchGuard::acquire();
    let [x, x_norm, hq, err, corr, rcp_err, rcp_cor, seed_n, d_work, d_top, n_top] = scratch
        .get_splits([
            x_len,
            x_len,
            h + 3,
            h + 4,
            l + 5,
            d_len + 1,
            d_len + 1,
            4,
            d_len,
            2 * h + 1,
            2 * h + 1,
        ]);
    let sh = d[d.len() - 1].leading_zeros() as u8;
    x.fill(0);
    shl_top_copy(d, d_work, sh);
    if !nr_rcp_chain(
        d_work,
        x,
        rcp_err,
        rcp_cor,
        seed_n,
        &sizes[..steps],
        &mut |a, b, o| mid_mul_dyn(a, b, o),
        &mut |a, b, o| hi_mul_dyn(a, b, o),
    ) {
        // Exact normalized quotient plus Newton division's upward bias.
        knuth_rcp_normalized(d_work, x, &mut rcp_err[..d_len], 1);
        inc_buf(x);
    }
    nr_refine_quo(
        n,
        d,
        sh,
        x,
        x_norm,
        d_top,
        n_top,
        hq,
        err,
        corr,
        q,
        h,
        l,
        &mut |a, b, o| mid_mul_dyn(a, b, o),
        &mut |a, b, o| hi_mul_dyn(a, b, o),
    )
}

fn nr_quo_est_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) -> NrEstimate {
    let Some((h, l)) = nr_split(n.len(), q.len()) else {
        return NrEstimate::Failed;
    };
    // n.len() <= N by the prepared capacity check and 2h + 1 <= n.len(), so every window fits.
    debug_assert!(2 * h + 1 <= N);
    let mut x = [0u64; N];
    let x = &mut x[..h + 3];
    let sh = d[d.len() - 1].leading_zeros() as u8;
    let reciprocal_ok = {
        let mut sizes = [0usize; 64];
        let steps = nr_rcp_schedule(x.len() - 1, &mut sizes);
        let pen_p = if steps >= 2 { sizes[1] } else { 1 };
        let d_len = 2 * pen_p + 1;
        let mut err = [0u64; N];
        let mut cor = [0u64; N];
        let mut seed_n = [0u64; 4];
        let mut d_work = [0u64; N];
        let d_work = &mut d_work[..d_len];
        shl_top_copy(d, d_work, sh);
        nr_rcp_chain(
            d_work,
            x,
            &mut err[..d_len + 1],
            &mut cor[..d_len + 1],
            &mut seed_n,
            &sizes[..steps],
            &mut |a, b, o| mid_mul_static::<N>(a, b, o),
            &mut |a, b, o| hi_mul_static::<N>(a, b, o),
        )
    };
    // The static seed can't safely outgrow N, so BZ retries instead.
    if !reciprocal_ok {
        return NrEstimate::Failed;
    }
    let mut x_norm = [0u64; N];
    let mut d_top = [0u64; N];
    let mut n_top = [0u64; N];
    let mut hq = [0u64; N];
    let mut err = [0u64; N];
    let mut corr = [0u64; N];
    nr_refine_quo(
        n,
        d,
        sh,
        x,
        &mut x_norm[..h + 3],
        &mut d_top[..2 * h + 1],
        &mut n_top[..2 * h + 1],
        &mut hq[..h + 3],
        &mut err[..h + 4],
        &mut corr[..l + 5],
        q,
        h,
        l,
        &mut |a, b, o| mid_mul_static::<N>(a, b, o),
        &mut |a, b, o| hi_mul_static::<N>(a, b, o),
    )
}

// Finishes a div_rem from the +-1 estimate: r = n - d q lies in (-d, 2d), so only the low
// d.len() + 1 limbs of the product matter, and the fix is a compare and one add or subtract. The
// remainder is built in rem, so a failure leaves n intact.
fn nr_rem_finish(
    n: &[u64],
    d: &[u64],
    q: &mut [u64],
    rem: &mut [u64],
    prod: &mut [u64],
    mul: &mut dyn FnMut(&[u64], &[u64], &mut [u64]),
) -> bool {
    let d_len = d.len();
    let w = d_len + 1;
    debug_assert!(n.len() >= w);
    debug_assert_eq!(rem.len(), w);
    rem.copy_from_slice(&n[..w]);

    let q_low = &q[..q.len().min(w)];
    debug_assert_eq!(prod.len(), d_len + q_low.len() - 1);
    mul(d, q_low, prod);
    sub_buf(rem, &prod[..w]);

    let mut corrections = 0;
    while rem[w - 1] == u64::MAX {
        // Negative window: q is one too big; the add wraps back into [0, d).
        if corrections == NR_DIV_MAX_CORRECTIONS || dec_buf(q) {
            return false;
        }
        add_buf(rem, d);
        corrections += 1;
    }
    while rem[w - 1] != 0 || cmp_buf(&rem[..d_len], d).is_ge() {
        // Window holds r + d: q is one too small.
        if corrections == NR_DIV_MAX_CORRECTIONS || inc_buf(q) {
            return false;
        }
        sub_buf(rem, d);
        corrections += 1;
    }
    true
}
// floor(B^R / d) for a one-limb d, R = rcp.len(): div_prim's loop over R zero limbs from remainder
// 1. d = 1 saturates to B^R - 1.
fn rcp_prim(d: u64, rcp: &mut [u64]) {
    if d == 0 {
        panic!("Division by zero error");
    }
    if d == 1 {
        rcp.fill(u64::MAX);
        return;
    }
    if rcp.is_empty() {
        return;
    }
    rcp.fill(0);
    let s = d.leading_zeros();
    let dn = d << s;
    unsafe { div_prim_asm(rcp, dn, reciprocal_2_1(dn), s, 1 << s) };
}
// Two low guard limbs past the requested precision absorb the truncated Newton products' error,
// in place of a verification product and correction pass.
const NR_RCP_GUARD_LIMBS: usize = 2;

fn nr_rcp_attempt_dyn(d: &[u64], rcp: &mut [u64]) -> bool {
    let work_len = rcp.len() + NR_RCP_GUARD_LIMBS;
    let sh = d[d.len() - 1].leading_zeros() as u8;
    let mut sizes = [0usize; 64];
    let steps = nr_rcp_schedule(work_len - 1, &mut sizes);
    let d_len = 2 * if steps >= 2 { sizes[1] } else { 1 } + 1;
    let mut scratch = ScratchGuard::acquire();
    let [work, err, cor, seed_n, d_work] =
        scratch.get_splits([work_len, d_len + 1, d_len + 1, work_len.min(2) + 2, d_len]);
    work.fill(0);
    shl_top_copy(d, d_work, sh);
    if !nr_rcp_chain(
        d_work,
        work,
        err,
        cor,
        seed_n,
        &sizes[..steps],
        &mut |a, b, o| mid_mul_dyn(a, b, o),
        &mut |a, b, o| hi_mul_dyn(a, b, o),
    ) || shl_buf(work, sh) != 0
    {
        return false;
    }
    rcp.copy_from_slice(&work[NR_RCP_GUARD_LIMBS..]);
    true
}

fn nr_rcp_attempt_static<const N: usize>(d: &[u64], rcp: &mut [u64]) -> bool {
    let work_len = rcp.len() + NR_RCP_GUARD_LIMBS;
    let sh = d[d.len() - 1].leading_zeros() as u8;
    let mut sizes = [0usize; 64];
    let steps = nr_rcp_schedule(work_len - 1, &mut sizes);
    let pen_p = if steps >= 2 { sizes[1] } else { 1 };
    let d_len = 2 * pen_p + 1;
    // Work and the last divisor window reach up to N + 3 limbs, past N only with d's low zero
    // padding (which mid_mul_static accepts). Bands stop a couple of limbs past p + 1, so the cap
    // at N only rejects bands that never occur, unless N is tiny, when Knuth retries.
    debug_assert!(d.len() <= N);
    let err_len = (d_len + 1).min(N);
    if pen_p + 2 > err_len {
        return false;
    }
    let mut work = [[0u64; N]; 2];
    let mut d_work = [[0u64; N]; 2];
    let (work, d_work) = (work.as_flattened_mut(), d_work.as_flattened_mut());
    let mut err = [0u64; N];
    let mut cor = [0u64; N];
    let mut seed_n = [0u64; 4];
    shl_top_copy(d, &mut d_work[..d_len], sh);
    if !nr_rcp_chain(
        &d_work[..d_len],
        &mut work[..work_len],
        &mut err[..err_len],
        &mut cor[..err_len],
        &mut seed_n,
        &sizes[..steps],
        &mut |a, b, o| mid_mul_static::<N>(a, b, o),
        &mut |a, b, o| hi_mul_static::<N>(a, b, o),
    ) || shl_buf(&mut work[..work_len], sh) != 0
    {
        return false;
    }
    rcp.copy_from_slice(&work[NR_RCP_GUARD_LIMBS..work_len]);
    true
}

// Divides buf in place by a normalized d with v = reciprocal_2_1(d) (Möller and Granlund,
// algorithm 4), returning the remainder << s; r starts as the bits buf << s pushes out of its top
// limb. Each step divides r B + u0 (u0 the next limb of buf << s): q1 q0 = v r + (r + 1) B + u0,
// rem = u0 - q1 d mod B, one fix when rem > q0 and a rare one when rem >= d. Only rem carries, and
// q1 d = (r + 1) d + c d + h d (h = hi(v r), c its carry) leaves only h d after the multiply.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn div_prim_asm(buf: &mut [u64], d: u64, v: u64, s: u32, mut r: u64) -> u64 {
    let len = buf.len();
    asm!(
        "2:",
        // u0 = (cur << s) | (nxt >> (64 - s)), with nxt = 0 below buf[0]. shld leaves cur at s = 0.
        "xor {nxt:e}, {nxt:e}",
        "test {n}, {n}",
        "jz 3f",
        "mov {nxt}, [{p} - 8]",
        "3:",
        "mov {u0}, {cur}",
        "shld {u0}, {nxt}, cl",
        // q1 q0 = v r + (r + 1) B + u0, with v r in rdx rax, and r = u0 - (r + 1) d beside it.
        "mov rax, {v}",
        "mul {r}",
        "mov {t}, {r}",
        "imul {t}, {d}",
        "lea {q}, [{r} + 1]",
        "mov {r}, {u0}",
        "sub {r}, {d}",
        "sub {r}, {t}",
        // q1 in t: in q itself the loop runs 7% slower.
        "mov {t}, {q}",
        "add rax, {u0}",
        "mov {u0}, 0",
        "cmovc {u0}, {d}",
        "adc {t}, rdx",
        // rem = r - c d - h d = u0 - q1 d
        "sub {r}, {u0}",
        "mov {u0}, rdx",
        "imul {u0}, {d}",
        "sub {r}, {u0}",
        // rem > q0: q1 -= 1, rem += d
        "lea {u0}, [{r} + {d}]",
        "cmp rax, {r}",
        "cmovb {r}, {u0}",
        "sbb {t}, 0",
        // rem >= d: q1 += 1, rem -= d
        "cmp {r}, {d}",
        "jae 5f",
        "4:",
        "mov [{p}], {t}",
        "mov {cur}, {nxt}",
        "lea {p}, [{p} - 8]",
        "sub {n}, 1",
        "jnc 2b",
        "jmp 6f",
        "5:",
        "sub {r}, {d}",
        "add {t}, 1",
        "jmp 4b",
        "6:",
        in("rcx") s as u64,
        p = inout(reg) buf.as_mut_ptr().add(len - 1) => _,
        n = inout(reg) len - 1 => _,
        cur = inout(reg) buf[len - 1] => _,
        r = inout(reg) r,
        v = in(reg) v,
        d = in(reg) d,
        nxt = out(reg) _,
        u0 = out(reg) _,
        t = out(reg) _,
        q = out(reg) _,
        out("rax") _,
        out("rdx") _,
        options(nostack),
    );
    r
}
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn div_prim_asm(buf: &mut [u64], d: u64, v: u64, s: u32, mut r: u64) -> u64 {
    let len = buf.len();
    asm!(
        "2:",
        // u0 = (cur << s) | (nxt >> (64 - s)), with nxt = 0 below buf[0]. nxt shifts by 1, then
        // 63 - s, as a register shift by 64 is one by 0.
        "mov {nxt}, xzr",
        "cbz {n}, 3f",
        "ldr {nxt}, [{p}, #-8]",
        "3:",
        "lsl {u0}, {cur}, {s}",
        "lsr {t}, {nxt}, #1",
        "lsr {t}, {t}, {ns}",
        "orr {u0}, {u0}, {t}",
        // q1 q0 = v r + (r + 1) B + u0, with h the high limb of v r, and r = u0 - (r + 1) d
        // beside the multiply.
        "mul {q0}, {v}, {r}",
        "umulh {h}, {v}, {r}",
        "add {q1}, {r}, #1",
        "msub {r}, {r}, {d}, {u0}",
        "sub {r}, {r}, {d}",
        "adds {q0}, {q0}, {u0}",
        "csel {t}, {d}, xzr, cs",
        "adc {q1}, {q1}, {h}",
        // rem = r - c d - h d = u0 - q1 d, with h d its own mul: M1 forwards a late accumulator
        // only from another multiply-add, so msub would multiply after r (11.6 cycles a limb
        // against 9.9).
        "mul {u0}, {h}, {d}",
        "sub {r}, {r}, {t}",
        "sub {r}, {r}, {u0}",
        // rem > q0: q1 -= 1, rem += d (C is clear when q0 < rem)
        "add {t}, {r}, {d}",
        "cmp {q0}, {r}",
        "csel {r}, {t}, {r}, lo",
        "sbc {q1}, {q1}, xzr",
        // rem >= d: q1 += 1, rem -= d
        "cmp {r}, {d}",
        "b.hs 5f",
        "4:",
        "str {q1}, [{p}], #-8",
        "mov {cur}, {nxt}",
        "subs {n}, {n}, #1",
        "b.hs 2b",
        "b 6f",
        "5:",
        "sub {r}, {r}, {d}",
        "add {q1}, {q1}, #1",
        "b 4b",
        "6:",
        s = in(reg) s as u64,
        ns = in(reg) 63 - s as u64,
        p = inout(reg) buf.as_mut_ptr().add(len - 1) => _,
        n = inout(reg) len - 1 => _,
        cur = inout(reg) buf[len - 1] => _,
        r = inout(reg) r,
        v = in(reg) v,
        d = in(reg) d,
        nxt = out(reg) _,
        u0 = out(reg) _,
        t = out(reg) _,
        h = out(reg) _,
        q0 = out(reg) _,
        q1 = out(reg) _,
        options(nostack),
    );
    r
}
pub fn div_prim(buf: &mut [u64], prim: u64) -> u64 {
    if prim == 0 {
        panic!("Division by zero error")
    }
    if prim == 1 || buf.is_empty() {
        return 0;
    }

    // Normalize prim and take its reciprocal.
    let s = prim.leading_zeros();
    let d = prim << s;
    let v = reciprocal_2_1(d);
    // Two shifts, so s = 0 gives 0 rather than a shift by 64.
    let r = (buf[buf.len() - 1] >> 1) >> (63 - s);
    unsafe { div_prim_asm(buf, d, v, s, r) >> s }
}
// div_prim's loop for a normalized two-limb d1 B + d0 with v = reciprocal_3_2 (algorithm 5),
// returning the remainder << s; r1 r0 starts as the top two limbs of buf << s. Each step folds in
// u0: q q0 = v r1 + (r1 r0), r1 r0 = (r0 - q d1) u0 - d - q d0 is the remainder for q + 1, d is
// added back when r1 >= q0, and r1 r0 >= d (rare) branches out of line. x86 forms both remainder
// candidates from copies taken before the products (adding d back afterwards puts copies on the
// chain), which needs a fourteenth register: v and d0 wait in xmm registers. aarch64 splits q into
// h + r1 + c (h = hi(v r1), c the carry into it), so r1 d and c d come off the chain while v r1
// multiplies and only h d waits on h: 12.2 cycles a limb on M1 against 13.2 for q d after q.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn div_prim2_asm(
    buf: &mut [u64],
    d1: u64,
    d0: u64,
    v: u64,
    s: u32,
    mut r1: u64,
    mut r0: u64,
) -> u128 {
    let len = buf.len();
    asm!(
        "2:",
        // u0 = ([p] << s) | ([p - 8] >> (64 - s)), with 0 below buf[0]. shld leaves [p] at s = 0.
        "xor {t:e}, {t:e}",
        "test {n}, {n}",
        "jz 3f",
        "mov {t}, [{p} - 8]",
        "3:",
        "mov {u0}, [{p}]",
        "shld {u0}, {t}, cl",
        // q q0 = v r1 + (r1 r0), in rdx rax.
        "movq rax, {v}",
        "mul {r1}",
        "add rax, {r0}",
        "adc rdx, {r1}",
        // C = (r0 - q d1) u0 - q d0 in r1 c0, the remainder for q, and A = C - d in r0 u0, the
        // remainder for q + 1.
        "mov {r1}, {r0}",
        "mov {c0}, {u0}",
        "movq {x}, {d0}",
        "sub {u0}, {x}",
        "sbb {r0}, {d1}",
        "mov {q}, rdx",
        "mov {t}, rax",
        "mov rax, {x}",
        "mul rdx",
        "mov {x}, {d1}",
        "imul {x}, {q}",
        "sub {r0}, {x}",
        "sub {r1}, {x}",
        "sub {u0}, rax",
        "sbb {r0}, rdx",
        "sub {c0}, rax",
        "sbb {r1}, rdx",
        // A and q + 1 (q + CF) when A's top limb is below q0, else C and q.
        "cmp {r0}, {t}",
        "cmovb {r1}, {r0}",
        "cmovae {u0}, {c0}",
        "adc {q}, 0",
        "mov {r0}, {u0}",
        // r1 r0 >= d, which needs r1 >= d1: q += 1, r1 r0 -= d
        "cmp {r1}, {d1}",
        "jae 7f",
        "4:",
        "mov [{p}], {q}",
        "lea {p}, [{p} - 8]",
        "sub {n}, 1",
        "jnc 2b",
        "jmp 6f",
        "7:",
        "movq {x}, {d0}",
        "ja 5f",
        "cmp {r0}, {x}",
        "jb 4b",
        "5:",
        "sub {r0}, {x}",
        "sbb {r1}, {d1}",
        "add {q}, 1",
        "jmp 4b",
        "6:",
        in("rcx") s as u64,
        p = inout(reg) buf.as_mut_ptr().add(len - 1) => _,
        n = inout(reg) len - 1 => _,
        r1 = inout(reg) r1,
        r0 = inout(reg) r0,
        v = in(xmm_reg) v,
        d0 = in(xmm_reg) d0,
        d1 = in(reg) d1,
        u0 = out(reg) _,
        t = out(reg) _,
        q = out(reg) _,
        c0 = out(reg) _,
        x = out(reg) _,
        out("rax") _,
        out("rdx") _,
        options(nostack),
    );
    ((r1 as u128) << 64) | r0 as u128
}
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn div_prim2_asm(
    buf: &mut [u64],
    d1: u64,
    d0: u64,
    v: u64,
    s: u32,
    mut r1: u64,
    mut r0: u64,
) -> u128 {
    let len = buf.len();
    asm!(
        "2:",
        // u0 = ([p] << s) | ([p - 8] >> (64 - s)), with 0 below buf[0]. The lower limb shifts by
        // 1, then 63 - s, as a register shift by 64 is one by 0.
        "mov {t}, xzr",
        "cbz {n}, 3f",
        "ldr {t}, [{p}, #-8]",
        "3:",
        "ldr {u0}, [{p}]",
        "lsl {u0}, {u0}, {s}",
        "lsr {t}, {t}, #1",
        "lsr {t}, {t}, {ns}",
        "orr {u0}, {u0}, {t}",
        // h q0 = v r1, and r1 d's low two limbs in a1 a0 beside it. madd would wait for a1 before
        // multiplying, as M1 forwards a late accumulator only from another multiply-add.
        "mul {q0}, {v}, {r1}",
        "umulh {h}, {v}, {r1}",
        "mul {a0}, {r1}, {d0}",
        "umulh {a1}, {r1}, {d0}",
        "mul {a2}, {r1}, {d1}",
        "add {a1}, {a1}, {a2}",
        // q0 += r0 with carry c, q = h + r1 + c, and c d in c1 t.
        "adds {q0}, {q0}, {r0}",
        "adc {q}, {h}, {r1}",
        "csel {t}, {d0}, xzr, cs",
        "csel {c1}, {d1}, xzr, cs",
        // r0 u0 = (r0 u0) - d - r1 d - c d
        "subs {u0}, {u0}, {d0}",
        "sbc {r0}, {r0}, {d1}",
        "subs {u0}, {u0}, {a0}",
        "sbc {r0}, {r0}, {a1}",
        "subs {u0}, {u0}, {t}",
        "sbc {r0}, {r0}, {c1}",
        // r0 u0 -= h d, the remainder for q + 1, and c1 t = r0 u0 + d, the one for q.
        "mul {a0}, {h}, {d0}",
        "umulh {a1}, {h}, {d0}",
        "mul {a2}, {h}, {d1}",
        "add {a1}, {a1}, {a2}",
        "subs {u0}, {u0}, {a0}",
        "sbc {r0}, {r0}, {a1}",
        "adds {t}, {u0}, {d0}",
        "adc {c1}, {r0}, {d1}",
        // r0 u0 and q + 1 when r0 < q0, else c1 t and q.
        "cmp {r0}, {q0}",
        "csel {r1}, {r0}, {c1}, lo",
        "csel {r0}, {u0}, {t}, lo",
        "cinc {q}, {q}, lo",
        // r1 r0 >= d, which needs r1 >= d1: q += 1, r1 r0 -= d
        "cmp {r1}, {d1}",
        "b.hs 5f",
        "4:",
        "str {q}, [{p}], #-8",
        "subs {n}, {n}, #1",
        "b.hs 2b",
        "b 6f",
        "5:",
        "b.hi 7f",
        "cmp {r0}, {d0}",
        "b.lo 4b",
        "7:",
        "subs {r0}, {r0}, {d0}",
        "sbc {r1}, {r1}, {d1}",
        "add {q}, {q}, #1",
        "b 4b",
        "6:",
        s = in(reg) s as u64,
        ns = in(reg) 63 - s as u64,
        p = inout(reg) buf.as_mut_ptr().add(len - 1) => _,
        n = inout(reg) len - 1 => _,
        r1 = inout(reg) r1,
        r0 = inout(reg) r0,
        v = in(reg) v,
        d1 = in(reg) d1,
        d0 = in(reg) d0,
        u0 = out(reg) _,
        t = out(reg) _,
        q = out(reg) _,
        q0 = out(reg) _,
        h = out(reg) _,
        a0 = out(reg) _,
        a1 = out(reg) _,
        a2 = out(reg) _,
        c1 = out(reg) _,
        options(nostack),
    );
    ((r1 as u128) << 64) | r0 as u128
}
// Divides buf in place by a two-limb prim and returns the remainder; the quotient has
// buf.len() - 1 limbs, so buf's top limb ends up zero. A one-limb prim goes to div_prim.
pub fn div_prim2(buf: &mut [u64], prim: u128) -> u128 {
    if prim <= u64::MAX as u128 {
        return div_prim(buf, prim as u64) as u128;
    }
    let len = buf.len();
    if len < 2 {
        let r = buf.first().map_or(0, |&x| x as u128);
        buf.fill(0);
        return r;
    }

    // Normalize prim and take its reciprocal v = floor((B^3 - 1) / d) - B.
    let s = prim.leading_zeros();
    let d = prim << s;
    let (d1, d0) = ((d >> 64) as u64, d as u64);
    let v = reciprocal_3_2(d1, d0);

    // The top two limbs of buf << s start the remainder (two shifts each, so s = 0 shifts
    // the limb below out entirely rather than by 0).
    let (top, next) = (buf[len - 1], buf[len - 2]);
    let r1 = (top >> 1) >> (63 - s);
    let r0 = (top << s) | ((next >> 1) >> (63 - s));
    buf[len - 1] = 0;
    unsafe { div_prim2_asm(&mut buf[..len - 1], d1, d0, v, s, r1, r0) >> s }
}
#[inline(always)]
pub unsafe fn mul_u64_asm(a: u64, b: u64) -> (u64, u64) {
    let t = a as u128 * b as u128;
    ((t >> 64) as u64, t as u64)
}
