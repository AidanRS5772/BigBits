use std::arch::asm;

use crate::utils::{
    mul::*, utils::*, ScratchGuard, BZ_CUTOFF, BZ_TOP_PADDED_COST_SCALE, DIV_KNUTH_CUTOFF,
    DYN_DIV_FFT_NR_BZ_CUTOFF, DYN_DIV_KARATSUBA_FFT_NR_BZ_CUTOFF, DYN_DIV_KARATSUBA_NR_BZ_CUTOFF,
    DYN_DIV_REM_FFT_NR_BZ_CUTOFF, DYN_DIV_REM_KARATSUBA_FFT_NR_BZ_CUTOFF,
    DYN_DIV_REM_KARATSUBA_NR_BZ_CUTOFF, DYN_RCP_KNUTH_NR_CUTOFF, STATIC_DIV_KARATSUBA_NR_BZ_CUTOFF,
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
    karatsuba_transform_cutoff: usize,
    karatsuba_bz_nr_ratio: f64,
    transform_bz_nr_ratio: f64,
}

fn div_alg_dispatch(q: usize, d: usize, tuning: DivCutoffs) -> DivAlg {
    debug_assert!(q != 0 && d != 0);
    let algo = if q == 1 || d <= DIV_KNUTH_CUTOFF {
        DivAlg::Knuth
    } else if q.max(d) < tuning.karatsuba_transform_cutoff {
        if (q as f64) * tuning.karatsuba_bz_nr_ratio > d as f64 {
            DivAlg::BZ
        } else {
            DivAlg::NR
        }
    } else {
        let log_q = (q as f64).log2();
        let log_d = (d as f64).log2();
        if log_q * tuning.transform_bz_nr_ratio > log_d * log_d {
            DivAlg::BZ
        } else {
            DivAlg::NR
        }
    };
    match algo {
        DivAlg::NR if nr_shape_supported(d + q - 1, q) => DivAlg::NR,
        DivAlg::Knuth => DivAlg::Knuth,
        // BZ would run only Knuth leaves at these widths. Avoid its block
        // setup, including when an unsupported NR shape needs a fallback.
        _ if d <= BZ_CUTOFF => DivAlg::Knuth,
        _ => DivAlg::BZ,
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

fn div_request_dyn(n: &[u64], d: &[u64], q: &mut [u64], request: DivisionRequest) -> u64 {
    let algorithm = div_alg_dispatch(
        request.q_len + 1,
        request.d_len,
        DivCutoffs {
            karatsuba_transform_cutoff: DYN_DIV_KARATSUBA_FFT_NR_BZ_CUTOFF,
            karatsuba_bz_nr_ratio: DYN_DIV_KARATSUBA_NR_BZ_CUTOFF,
            transform_bz_nr_ratio: DYN_DIV_FFT_NR_BZ_CUTOFF,
        },
    );
    div_prepared_dyn(n, d, q, request, algorithm)
}

fn div_request_static<const N: usize>(
    n: &[u64],
    d: &[u64],
    q: &mut [u64],
    request: DivisionRequest,
) -> u64 {
    let algorithm = div_alg_dispatch(
        request.q_len + 1,
        request.d_len,
        DivCutoffs {
            karatsuba_transform_cutoff: STATIC_DIV_KARATSUBA_NTT_NR_BZ_CUTOFF,
            karatsuba_bz_nr_ratio: STATIC_DIV_KARATSUBA_NR_BZ_CUTOFF,
            transform_bz_nr_ratio: STATIC_DIV_NTT_NR_BZ_CUTOFF,
        },
    );
    div_prepared_static::<N>(n, d, q, request, algorithm)
}

/// Divides trimmed `n` by trimmed nonzero `d` using pooled dynamic scratch.
///
/// `q` must contain at least [`div_quotient_len`] limbs. With an exact-sized
/// output the additional structural quotient limb is returned; a longer
/// output absorbs it and the function returns zero.
pub fn div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let Some(request) = division_preflight(n, d, q) else {
        return 0;
    };
    div_request_dyn(n, d, q, request)
}

/// Dynamic division with the same quotient contract as [`div_dyn`], leaving
/// the exact remainder in `n`.
pub fn div_rem_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) -> u64 {
    let Some(request) = division_preflight(n, d, q) else {
        return 0;
    };
    let algorithm = div_alg_dispatch(
        request.q_len + 1,
        request.d_len,
        DivCutoffs {
            karatsuba_transform_cutoff: DYN_DIV_REM_KARATSUBA_FFT_NR_BZ_CUTOFF,
            karatsuba_bz_nr_ratio: DYN_DIV_REM_KARATSUBA_NR_BZ_CUTOFF,
            transform_bz_nr_ratio: DYN_DIV_REM_FFT_NR_BZ_CUTOFF,
        },
    );
    div_rem_prepared_dyn(n, d, q, request, algorithm)
}

/// Static-scratch division with the same quotient contract as [`div_dyn`].
pub fn div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let Some(request) = division_preflight(n, d, q) else {
        return 0;
    };
    div_request_static::<N>(n, d, q, request)
}

/// Static-scratch division with the same quotient and remainder contract as
/// [`div_rem_dyn`].
pub fn div_rem_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) -> u64 {
    let Some(request) = division_preflight(n, d, q) else {
        return 0;
    };
    let algorithm = div_alg_dispatch(
        request.q_len + 1,
        request.d_len,
        DivCutoffs {
            karatsuba_transform_cutoff: STATIC_DIV_REM_KARATSUBA_NTT_NR_BZ_CUTOFF,
            karatsuba_bz_nr_ratio: STATIC_DIV_REM_KARATSUBA_NR_BZ_CUTOFF,
            transform_bz_nr_ratio: STATIC_DIV_REM_NTT_NR_BZ_CUTOFF,
        },
    );
    div_rem_prepared_static::<N>(n, d, q, request, algorithm)
}

/// Computes the highest `q.len()` limbs of the quotient body and returns the
/// structural overflow limb. If the complete body fits, this delegates to
/// [`div_dyn`] and follows its overflow-absorption behavior.
pub fn hi_div_dyn(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let requested = q.len();
    let Some(request) = division_request(n, d, q, Some(requested)) else {
        return 0;
    };
    div_request_dyn(n, d, q, request)
}

/// Static-scratch counterpart of [`hi_div_dyn`].
pub fn hi_div_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) -> u64 {
    let requested = q.len();
    let Some(request) = division_request(n, d, q, Some(requested)) else {
        return 0;
    };
    div_request_static::<N>(n, d, q, request)
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
    /// Effective divisor width after removing exact low zero limbs.
    d_len: usize,
    /// Quotient body length passed through the prepared execution driver.
    q_len: usize,
    /// Shift that bit-normalizes the effective divisor.
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

/// Performs checks and operand-window preparation shared by standard public
/// division entries.
///
/// `q` must hold at least `div_quotient_len(n.len(), d.len())` limbs. The
/// possible additional structural quotient limb is returned by the selected
/// algorithm or absorbed into a longer output.
pub fn division_preflight(n: &[u64], d: &[u64], q: &mut [u64]) -> Option<DivisionRequest> {
    division_request(n, d, q, None)
}

/// Performs validation shared by every public reciprocal entry. An empty
/// output requests zero precision and therefore no work.
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

/// Selects the part of the divisor that can affect the requested reciprocal
/// precision. Removing low zero limbs is exact. Keeping one limb beyond the
/// output precision bounds input-truncation error to the final output limb;
/// static drivers may keep only `precision` limbs when their capacity is
/// tight, which still bounds the numeric error to less than one whole limb.
fn reciprocal_divisor_window(d: &[u64], precision: usize, capacity: usize) -> &[u64] {
    debug_assert!(precision != 0 && capacity != 0);
    let max_window_len = d.len().min(precision.saturating_add(1)).min(capacity);
    let window = &d[d.len() - max_window_len..];
    let first_nonzero = window.iter().position(|&limb| limb != 0).unwrap();
    &window[first_nonzero..]
}

fn assert_static_division_capacity<const N: usize>(n: &[u64], d: &[u64], q: &[u64]) {
    assert!(
        n.len() <= N && d.len() <= N && q.len() <= N,
        "prepared division operands exceed static capacity"
    );
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

/// Executes the request returned by preflight for these operands and output.
/// Failed NR attempts retry through BZ without modifying the input numerator.
pub fn div_prepared_dyn(
    n: &[u64],
    d: &[u64],
    q: &mut [u64],
    request: DivisionRequest,
    mut algorithm: DivAlg,
) -> u64 {
    let n = &n[request.n_start..];
    let d = &d[request.d_start..];

    if algorithm == DivAlg::NR {
        let full_len = request.q_len + 1;
        if q.len() >= full_len {
            if nr_div_attempt_dyn(n, d, &mut q[..full_len]) {
                q[full_len..].fill(0);
                return 0;
            }
        } else {
            let mut scratch = ScratchGuard::acquire();
            let q_full = scratch.get(full_len);
            if nr_div_attempt_dyn(n, d, q_full) {
                q.copy_from_slice(&q_full[..request.q_len]);
                return q_full[request.q_len];
            }
        }
        algorithm = DivAlg::BZ;
    }
    let (q, q_tail) = q.split_at_mut(request.q_len);
    debug_assert_rigid_division_shape(n, d, q);
    if d.len() == 1 {
        let overflow = knuth_div_prim(n, d[0], q).0;
        return absorb_div_overflow(q_tail, overflow);
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
    let n = n_work;
    let overflow = match algorithm {
        DivAlg::Knuth => knuth_div_rem_core(n, d, q, shift, RemainderMode::Discard),
        DivAlg::BZ => bz_div_core_dyn(n, d, q, shift, RemainderMode::Discard),
        DivAlg::NR => unreachable!(),
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
    let n = &mut n[request.n_start..];
    let d = &d[request.d_start..];

    if algorithm == DivAlg::NR {
        let full_len = request.q_len + 1;
        if q.len() >= full_len {
            if nr_div_rem_attempt_dyn(n, d, &mut q[..full_len]) {
                q[full_len..].fill(0);
                return 0;
            }
        } else {
            let mut scratch = ScratchGuard::acquire();
            let q_full = scratch.get(full_len);
            if nr_div_rem_attempt_dyn(n, d, q_full) {
                q.copy_from_slice(&q_full[..request.q_len]);
                return q_full[request.q_len];
            }
        }
        algorithm = DivAlg::BZ;
    }
    let (q, q_tail) = q.split_at_mut(request.q_len);
    debug_assert_rigid_division_shape(n, d, q);
    if d.len() == 1 {
        let (overflow, rem) = knuth_div_prim(n, d[0], q);
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
        DivAlg::BZ => bz_div_core_dyn(n, d, q, shift, RemainderMode::Restore),
        DivAlg::NR => unreachable!(),
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
    let n = &n[request.n_start..];
    let d = &d[request.d_start..];
    assert_static_division_capacity::<N>(n, d, &q[..request.q_len]);
    if algorithm == DivAlg::NR {
        // The full quotient also fits: q_len + 1 <= n.len() <= N.
        let full_len = request.q_len + 1;
        if q.len() >= full_len {
            if nr_div_attempt_static::<N>(n, d, &mut q[..full_len]) {
                q[full_len..].fill(0);
                return 0;
            }
        } else {
            let mut storage = [0u64; N];
            let q_full = &mut storage[..full_len];
            if nr_div_attempt_static::<N>(n, d, q_full) {
                q.copy_from_slice(&q_full[..request.q_len]);
                return q_full[request.q_len];
            }
        }
        algorithm = DivAlg::BZ;
    }
    let (q, q_tail) = q.split_at_mut(request.q_len);
    debug_assert_rigid_division_shape(n, d, q);
    if d.len() == 1 {
        let overflow = knuth_div_prim(n, d[0], q).0;
        return absorb_div_overflow(q_tail, overflow);
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
        DivAlg::BZ => bz_div_core_static::<N>(n, d, q, shift, RemainderMode::Discard),
        DivAlg::NR => unreachable!(),
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
    let n = &mut n[request.n_start..];
    let d = &d[request.d_start..];
    assert_static_division_capacity::<N>(n, d, &q[..request.q_len]);
    if algorithm == DivAlg::NR {
        // The full quotient also fits: q_len + 1 <= n.len() <= N.
        let full_len = request.q_len + 1;
        if q.len() >= full_len {
            if nr_div_rem_attempt_static::<N>(n, d, &mut q[..full_len]) {
                q[full_len..].fill(0);
                return 0;
            }
        } else {
            let mut storage = [0u64; N];
            let q_full = &mut storage[..full_len];
            if nr_div_rem_attempt_static::<N>(n, d, q_full) {
                q.copy_from_slice(&q_full[..request.q_len]);
                return q_full[request.q_len];
            }
        }
        algorithm = DivAlg::BZ;
    }
    let (q, q_tail) = q.split_at_mut(request.q_len);
    debug_assert_rigid_division_shape(n, d, q);
    if d.len() == 1 {
        let (overflow, rem) = knuth_div_prim(n, d[0], q);
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
        DivAlg::BZ => bz_div_core_static::<N>(n, d, q, shift, RemainderMode::Restore),
        DivAlg::NR => unreachable!(),
    };
    absorb_div_overflow(q_tail, overflow)
}

/// Executes a nonempty reciprocal request after [`reciprocal_preflight`].
pub fn rcp_prepared_dyn(d: &[u64], rcp: &mut [u64], algorithm: RcpAlg) {
    let d = reciprocal_divisor_window(d, rcp.len(), usize::MAX);
    if algorithm == RcpAlg::NR && nr_rcp_attempt_dyn(d, rcp) {
        return;
    }
    // Knuth is both the selected classical path and the sole NR retry.
    if d.len() == 1 {
        knuth_rcp_prim(d[0], rcp);
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

/// Static counterpart of [`rcp_prepared_dyn`], with capacity measured against
/// the prepared divisor window and requested precision.
pub fn rcp_prepared_static<const N: usize>(d: &[u64], rcp: &mut [u64], algorithm: RcpAlg) {
    assert!(rcp.len() <= N, "reciprocal output exceeds static capacity");
    let d = reciprocal_divisor_window(d, rcp.len(), N);
    if algorithm == RcpAlg::NR && nr_rcp_attempt_static::<N>(d, rcp) {
        return;
    }
    // Knuth is both the selected classical path and the sole NR retry.
    if d.len() == 1 {
        knuth_rcp_prim(d[0], rcp);
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

pub fn knuth_est(win: &mut [u64], of: &mut u64, d: &[u64], d1: u64, d0: u64) -> u64 {
    let (mut qhat, rhat_hi, rhat_lo) = if *of >= d1 {
        let (rhat_lo, c) = win.last().unwrap().overflowing_add(d1);
        (u64::MAX, *of - d1 + c as u64, rhat_lo)
    } else {
        let mut r = *of;
        let mut q = *win.last().unwrap();
        unsafe { div_rem_2_1_asm(&mut q, &mut r, d1) };
        (q, 0, r)
    };

    if rhat_hi == 0 {
        let u0 = win[win.len() - 2];
        let (a_hi, a_lo) = unsafe { mul_u64_asm(qhat, d0) };
        if a_hi > rhat_lo || (a_hi == rhat_lo && a_lo > u0) {
            qhat -= 1;
            let (r, carry) = rhat_lo.overflowing_add(d1);
            if !carry {
                let (a_lo, borrow) = a_lo.overflowing_sub(d0);
                let a_hi = a_hi.wrapping_sub(borrow as u64);
                if a_hi > r || (a_hi == r && a_lo > u0) {
                    qhat -= 1;
                }
            }
        }
    }

    if sub_mul_of(win, of, d, qhat) {
        qhat -= 1;
        if add_buf(win, d) {
            *of = of.wrapping_add(1);
        }
    }

    return qhat;
}

/// Knuth division step for a normalized multi-limb divisor and an explicit
/// normalized-numerator overflow limb.
///
/// The minimum output length is `n.len() - d.len()`. The possible additional
/// quotient limb is returned or absorbed into a longer output.
pub fn div_buf_of(n: &mut [u64], of: &mut u64, d: &[u64], out: &mut [u64]) -> u64 {
    let d_len = d.len();
    let n_len = n.len();
    let q_len = n_len - d_len;
    debug_assert!(out.len() >= q_len, "out is not large enough for division");
    let (out_body, out_tail) = out.split_at_mut(q_len);
    let d1 = d[d_len - 1];
    let d0 = d[d_len - 2];
    let overflow = knuth_est(&mut n[q_len..], of, d, d1, d0);
    div_buf_body(n, d, out_body);
    absorb_div_overflow(out_tail, overflow)
}

/// Divide by normalized `d`, assuming the high divisor-width window of `n` is below `d`.
/// `out` must have exactly `n.len() - d.len()` limbs, and `d.len() >= 2`.
pub fn div_buf_body(n: &mut [u64], d: &[u64], out: &mut [u64]) {
    let d_len = d.len();
    let q_len = n.len() - d_len;
    debug_assert_eq!(out.len(), q_len);
    let d1 = d[d_len - 1];
    let d0 = d[d_len - 2];
    for i in (0..q_len).rev() {
        let (win, of) = n[i..].split_at_mut(d_len);
        out[i] = knuth_est(win, &mut of[0], d, d1, d0)
    }
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

fn knuth_div_prim(n: &[u64], d: u64, q: &mut [u64]) -> (u64, u64) {
    debug_assert_eq!(q.len() + 1, n.len());
    let mut rem = 0;
    let mut overflow = n[n.len() - 1];
    unsafe { div_rem_2_1_asm(&mut overflow, &mut rem, d) };
    for i in (0..q.len()).rev() {
        q[i] = n[i];
        unsafe { div_rem_2_1_asm(&mut q[i], &mut rem, d) };
    }
    (overflow, rem)
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

fn bz_div_top(
    n: &mut [u64],
    d: &[u64],
    out: &mut [u64],
    top_n: &mut [u64],
    q_tmp: &mut [u64],
    scratch: &mut [u64],
    last_n: u64,
    init_idx: usize,
    init_len: usize,
    mut mul: impl FnMut(&mut [u64], &[u64], &mut [u64]),
) -> u64 {
    let dlen = d.len();
    let init_body_len = init_len - dlen;
    top_n[..init_len].copy_from_slice(&n[init_idx..]);
    top_n[init_len] = last_n;
    top_n[init_len + 1..].fill(0);
    div_2_1(top_n, d, q_tmp, scratch, &mut mul);
    out[init_idx..init_idx + init_body_len].copy_from_slice(&q_tmp[..init_body_len]);
    n[init_idx..init_idx + dlen].copy_from_slice(&top_n[..dlen]);
    n[init_idx + dlen..].fill(0);
    q_tmp[init_body_len]
}

fn bz_div_core_dyn(
    n: &mut [u64],
    d: &[u64],
    out: &mut [u64],
    normalization_shift: u8,
    remainder_mode: RemainderMode,
) -> u64 {
    debug_assert_rigid_division_shape(n, d, out);
    let dlen = d.len();
    debug_assert!(dlen >= 2);
    debug_assert_eq!(d[dlen - 1].leading_zeros(), 0);
    let mut last_n = shl_buf(n, normalization_shift);
    let blocks = (n.len() - dlen) / dlen;
    let init_idx = dlen * blocks;
    let init_len = n.len() - init_idx;
    let init_qlen = init_len - dlen + 1;
    let quotient_overflow = if use_bz_for_top_block(dlen, init_qlen) {
        let mut scratch = ScratchGuard::acquire();
        let [top_n, q_tmp, div_scratch] = scratch.get_splits([2 * dlen, dlen, dlen]);
        bz_div_top(
            n,
            d,
            out,
            top_n,
            q_tmp,
            div_scratch,
            last_n,
            init_idx,
            init_len,
            |n, d, q| {
                mul_dyn(n, d, q);
            },
        )
    } else {
        div_buf_of(&mut n[init_idx..], &mut last_n, d, &mut out[init_idx..])
    };
    if blocks != 0 {
        let mut scratch_guard = ScratchGuard::acquire();
        let scratch = scratch_guard.get(dlen);
        for i in (0..blocks).rev() {
            let idx = dlen * i;
            div_2_1(
                &mut n[idx..idx + 2 * dlen],
                d,
                &mut out[idx..idx + dlen],
                scratch,
                &mut |n, d, q| {
                    mul_dyn(n, d, q);
                },
            );
        }
    }
    if remainder_mode == RemainderMode::Restore {
        debug_assert!(n[dlen..].iter().all(|&limb| limb == 0));
        shr_buf(&mut n[..dlen], normalization_shift);
    }
    quotient_overflow
}

fn bz_div_core_static<const N: usize>(
    n: &mut [u64],
    d: &[u64],
    out: &mut [u64],
    normalization_shift: u8,
    remainder_mode: RemainderMode,
) -> u64 {
    debug_assert_rigid_division_shape(n, d, out);
    let dlen = d.len();
    debug_assert!(dlen >= 2);
    debug_assert_eq!(d[dlen - 1].leading_zeros(), 0);
    let mut last_n = shl_buf(n, normalization_shift);
    let blocks = (n.len() - dlen) / dlen;
    let init_idx = dlen * blocks;
    let init_len = n.len() - init_idx;
    let init_qlen = init_len - dlen + 1;
    let mut scratch = [0u64; N];
    let quotient_overflow = if dlen <= N / 2 && use_bz_for_top_block(dlen, init_qlen) {
        let mut top_storage = [0u64; N];
        let top_n = &mut top_storage[..2 * dlen];
        let (q_tmp, div_scratch) = scratch[..2 * dlen].split_at_mut(dlen);
        bz_div_top(
            n,
            d,
            out,
            top_n,
            q_tmp,
            div_scratch,
            last_n,
            init_idx,
            init_len,
            |n, d, q| {
                mul_static::<N>(n, d, q);
            },
        )
    } else {
        div_buf_of(&mut n[init_idx..], &mut last_n, d, &mut out[init_idx..])
    };
    if blocks != 0 {
        let scratch = &mut scratch[..dlen];
        for i in (0..blocks).rev() {
            let idx = dlen * i;
            div_2_1(
                &mut n[idx..idx + 2 * dlen],
                d,
                &mut out[idx..idx + dlen],
                scratch,
                &mut |n, d, q| {
                    mul_static::<N>(n, d, q);
                },
            );
        }
    }
    if remainder_mode == RemainderMode::Restore {
        debug_assert!(n[dlen..].iter().all(|&limb| limb == 0));
        shr_buf(&mut n[..dlen], normalization_shift);
    }
    quotient_overflow
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

// Signed error band (d·y − num) around the cancellation point: fills e[..p+1]
// with the middle band of d·y minus the aligned numerator limbs (num[i] lines
// up with e[i]; empty num compares against zeros — the reciprocal case, where
// the numerator B^k has no limbs in the walked window), then extends upward
// while limbs are neither 0 nor all-ones. Bails with None past cap. The mid
// closure supplies the dyn or static middle product.
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

// Knuth long division over an implicit power-of-B numerator. `d` must be
// bit-normalized and `win` is the sliding d.len()-limb remainder window.
pub fn knuth_rcp_normalized(d: &[u64], rcp: &mut [u64], win: &mut [u64], seed: u64) {
    debug_assert!(d.len() >= 2);
    debug_assert_eq!(win.len(), d.len());

    win.fill(0);
    let mut of = seed;
    let d1 = d[d.len() - 1];
    let d0 = d[d.len() - 2];
    for q in rcp.iter_mut().rev() {
        *q = knuth_est(win, &mut of, d, d1, d0);
        of = win[win.len() - 1];
        win.copy_within(0..win.len() - 1, 1);
        win[0] = 0;
    }
}

// Refines rcp from precision p to 2p (trunc = false) or 2p - 1 (trunc = true,
// dropping the lowest limb of the correction).
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

// Knuth seed plus the ceiling-halving refine chain over a normalized top
// window of the divisor. Returns false when any refine band bails (rcp is then
// only partially refined; the caller falls back).
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
    // Precision-1 seed: floor(B^(seed_r + 2) / top 3 limbs of d) + 1 by knuth
    // division.
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

// Karp-Markstein quotient refinement: computes the guarded high product hq
// (h+3 limbs holding quotient digits l-3..q_len-1; its top h+1 limbs are the
// estimate anchored at digit l-1, one Newton feedback unit), measures
// err = d·q_hi − n around the cancellation point, and assembles q with the
// correction x·err. Reports failure on band rejection or quotient wrap;
// otherwise the top guard limb determines whether exact correction is needed.
fn nr_refine_quo(
    n: &[u64],
    d_top: &[u64],
    x: &[u64],
    hq: &mut [u64],
    err: &mut [u64],
    corr: &mut [u64],
    q: &mut [u64],
    h: usize,
    l: usize,
    mid: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> (u64, u64),
    short: &mut dyn FnMut(&[u64], &[u64], &mut [u64]) -> u64,
) -> NrEstimate {
    hq[h + 2] = short(n, x, &mut hq[..h + 2]);
    let y = end_ref(hq, h + 1);
    let Some((neg, e_len)) = nr_err_band(d_top, y, end_ref(n, 2 * h + 1), err, h, h + 3, mid)
    else {
        return NrEstimate::Failed;
    };

    // corr[i] is quotient digit i - GUARD; the window top tracks err's top so
    // k floats with e_len while the digit alignment stays fixed.
    let k = e_len + l - h + 1;
    let c = &mut corr[..k + 1];
    c[k] = short(x, &err[..e_len], &mut c[..k]);
    if neg {
        sub_buf(&mut c[1..], end_ref(x, k));
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

#[inline]
fn nr_shape_supported(n_len: usize, q_len: usize) -> bool {
    if q_len < 2 * GUARD + 2 {
        return false;
    }
    let h = q_len / 2 + 1;
    n_len >= 2 * h + 1
}

#[inline]
fn nr_split(q_len: usize, d_len: usize) -> (usize, usize) {
    let h = q_len / 2 + 1;
    debug_assert!(q_len >= 2 * GUARD + 2, "nr division requires q.len() >= 8");
    let l = q_len - h;
    debug_assert!(
        d_len + l >= h + 2,
        "nr division requires n.len() >= 2h+1 for the quotient error band"
    );
    (h, l)
}

// Dynamic front half of Newton division: half-precision reciprocal plus one
// quotient refinement. Fills q with an estimate within one of the true
// quotient. The guard verdict decides whether exact correction is needed.
fn nr_quo_est_dyn(n: &[u64], d: &[u64], q: &mut [u64]) -> NrEstimate {
    if !nr_shape_supported(n.len(), q.len()) {
        return NrEstimate::Failed;
    }
    let (h, l) = nr_split(q.len(), d.len());

    let x_len = h + 3;
    let mut sizes = [0usize; 64];
    let steps = nr_rcp_schedule(x_len - 1, &mut sizes);
    // The widest refinement reads 2p+1 divisor limbs and may extend its error
    // band one limb beyond that; the seed numerator never exceeds four limbs.
    let pen_p = if steps >= 2 { sizes[1] } else { 1 };
    let d_len = 2 * pen_p + 1;
    let rcp_err_len = d_len + 1;
    let mut scratch = ScratchGuard::acquire();
    let [x, hq, err, corr, rcp_err, rcp_cor, seed_n, d_work, d_top] = scratch.get_splits([
        x_len,
        h + 3,
        h + 4,
        l + 5,
        rcp_err_len,
        rcp_err_len,
        4,
        d_len,
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
        // Local reciprocal repair: exact normalized quotient plus the upward
        // bias required by Newton division. Public reciprocals have a different
        // precision contract and never use this +1 adjustment.
        knuth_rcp_normalized(d_work, x, &mut rcp_err[..d_len], 1);
        inc_buf(x);
    }
    if shl_buf(x, sh) != 0 {
        return NrEstimate::Failed;
    }

    shl_top_copy(d, d_top, 0);

    nr_refine_quo(
        n,
        d_top,
        x,
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
    if !nr_shape_supported(n.len(), q.len()) {
        return NrEstimate::Failed;
    }
    let (h, l) = nr_split(q.len(), d.len());
    // Prepared capacity proves n.len() <= N, and the shape check above
    // proves 2*h+1 <= n.len(). All refinement windows therefore fit.
    debug_assert!(2 * h + 1 <= N);

    let mut x = [0u64; N];
    let x = &mut x[..h + 3];

    let sh = d[d.len() - 1].leading_zeros() as u8;

    let reciprocal_ok = {
        let mut sizes = [0usize; 64];
        let steps = nr_rcp_schedule(x.len() - 1, &mut sizes);
        let pen_p = if steps >= 2 { sizes[1] } else { 1 };
        let d_len = 2 * pen_p + 1;
        let err_len = 2 * pen_p + 2;

        let mut err = [0u64; N];
        let mut cor = [0u64; N];
        let mut seed_n = [0u64; 4];
        let mut d_work = [0u64; N];
        let d_work = &mut d_work[..d_len];
        shl_top_copy(d, d_work, sh);

        nr_rcp_chain(
            d_work,
            x,
            &mut err[..err_len],
            &mut cor[..err_len],
            &mut seed_n,
            &sizes[..steps],
            &mut |a, b, o| mid_mul_static::<N>(a, b, o),
            &mut |a, b, o| hi_mul_static::<N>(a, b, o),
        )
    };

    if !reciprocal_ok {
        // The static seed cannot safely outgrow N, so let the caller fall back
        // to the complete Burnikel-Ziegler division.
        return NrEstimate::Failed;
    }
    if shl_buf(x, sh) != 0 {
        return NrEstimate::Failed;
    }

    let mut d_top = [0u64; N];
    let d_top = &mut d_top[..2 * h + 1];
    shl_top_copy(d, d_top, 0);

    let mut hq = [0u64; N];
    let mut err = [0u64; N];
    let mut corr = [0u64; N];
    nr_refine_quo(
        n,
        d_top,
        x,
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

// Finishes a div_rem from the +-1 quotient estimate: r = n - d*q lies in
// (-d, 2d), so only the low d.len()+1 limb window of the product matters
// (q mod B^(d.len()+1) alone determines it) and the fixup is a compare and one
// add/sub — no exact-correction multiply. The candidate is built separately
// so a failed correction can fall back without first restoring n.
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

// NR attempts may overwrite q on failure, but leave n intact for the
// prepared driver's BZ retry. Correction scratch is allocated only when used.
fn nr_div_attempt_dyn(n: &[u64], d: &[u64], q: &mut [u64]) -> bool {
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
}

fn nr_div_rem_attempt_dyn(n: &mut [u64], d: &[u64], q: &mut [u64]) -> bool {
    debug_assert_full_division_shape(n, d, q);
    if nr_quo_est_dyn(n, d, q) == NrEstimate::Failed {
        return false;
    }
    // Even a proven quotient needs the windowed product to produce a remainder.
    let q_low_len = q.len().min(d.len() + 1);
    let rem_len = d.len() + 1;
    let mut scratch = ScratchGuard::acquire();
    let [prod, rem] = scratch.get_splits([d.len() + q_low_len - 1, rem_len]);
    if !nr_rem_finish(n, d, q, rem, prod, &mut |a, b, o| {
        mul_dyn(a, b, o);
    }) {
        return false;
    }
    n[..rem_len].copy_from_slice(rem);
    n[rem_len..].fill(0);
    true
}

fn nr_div_attempt_static<const N: usize>(n: &[u64], d: &[u64], q: &mut [u64]) -> bool {
    debug_assert_full_division_shape(n, d, q);
    match nr_quo_est_static::<N>(n, d, q) {
        NrEstimate::Exact => true,
        NrEstimate::Failed => false,
        NrEstimate::NeedsCorrection => {
            let mut prod_storage = [0u64; N];
            let mut num_storage = [0u64; N];
            let prod = &mut prod_storage[..n.len()];
            let num = &mut num_storage[..n.len()];
            num.copy_from_slice(n);
            nr_exact_correction(num, d, q, prod, &mut |a, b, o| {
                mul_static::<N>(a, b, o);
            })
        }
    }
}

fn nr_div_rem_attempt_static<const N: usize>(n: &mut [u64], d: &[u64], q: &mut [u64]) -> bool {
    debug_assert_full_division_shape(n, d, q);
    if nr_quo_est_static::<N>(n, d, q) == NrEstimate::Failed {
        return false;
    }
    // Even a proven quotient needs the windowed product to produce a remainder.
    let q_low_len = q.len().min(d.len() + 1);
    let rem_len = d.len() + 1;
    let mut prod_storage = [0u64; N];
    let mut rem_storage = [0u64; N];
    let prod = &mut prod_storage[..d.len() + q_low_len - 1];
    let rem = &mut rem_storage[..rem_len];
    if !nr_rem_finish(n, d, q, rem, prod, &mut |a, b, o| {
        mul_static::<N>(a, b, o);
    }) {
        return false;
    }
    n[..rem_len].copy_from_slice(rem);
    n[rem_len..].fill(0);
    true
}

fn knuth_rcp_prim(d: u64, rcp: &mut [u64]) {
    if d == 0 {
        panic!("Division by zero error");
    }
    if d == 1 {
        rcp.fill(u64::MAX);
        return;
    }

    let mut rem = 1;
    for q in rcp.iter_mut().rev() {
        *q = 0;
        unsafe { div_rem_2_1_asm(q, &mut rem, d) };
    }
}

// Truncated Newton products can accumulate several ulps. Refining two low
// limbs beyond the public precision absorbs that error without an exact
// verification product or a final correction pass.
const NR_RCP_GUARD_LIMBS: usize = 2;

fn nr_rcp_attempt_dyn(d: &[u64], rcp: &mut [u64]) -> bool {
    let r = rcp.len();
    let work_len = r
        .checked_add(NR_RCP_GUARD_LIMBS)
        .expect("reciprocal size overflow");
    let sh = d[d.len() - 1].leading_zeros() as u8;
    let mut sizes = [0usize; 64];
    let steps = nr_rcp_schedule(work_len - 1, &mut sizes);
    let pen_p = if steps >= 2 { sizes[1] } else { 1 };
    let d_len = pen_p
        .checked_mul(2)
        .and_then(|len| len.checked_add(1))
        .expect("reciprocal size overflow");
    let err_len = d_len.checked_add(1).expect("reciprocal size overflow");
    let seed_len = work_len.min(2) + 2;

    let mut scratch = ScratchGuard::acquire();
    let [work, err, cor, seed_n, d_work] =
        scratch.get_splits([work_len, err_len, err_len, seed_len, d_len]);
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

    // The Newton chain is deliberately not corrected with a full product.
    // The two internal low guard limbs absorb accumulated ulp error. They are
    // discarded without verifying or correcting the least-significant result.
    rcp.copy_from_slice(&work[NR_RCP_GUARD_LIMBS..]);
    true
}

fn nr_rcp_attempt_static<const N: usize>(d: &[u64], rcp: &mut [u64]) -> bool {
    let r = rcp.len();
    let work_len = r
        .checked_add(NR_RCP_GUARD_LIMBS)
        .expect("reciprocal size overflow");
    let sh = d[d.len() - 1].leading_zeros() as u8;
    let mut sizes = [0usize; 64];
    let steps = nr_rcp_schedule(work_len - 1, &mut sizes);
    let pen_p = if steps >= 2 { sizes[1] } else { 1 };
    let d_len = pen_p
        .checked_mul(2)
        .and_then(|len| len.checked_add(1))
        .expect("reciprocal size overflow");
    let err_len = d_len.checked_add(1).expect("reciprocal size overflow");

    // Guard precision and schedule parity can make the widest refinement band
    // several limbs larger than the requested reciprocal. Tight static buffers
    // use Knuth instead of slicing beyond N.
    if work_len > N || d_len > N || err_len > N {
        return false;
    }

    let mut work = [0u64; N];
    let mut err = [0u64; N];
    let mut cor = [0u64; N];
    let mut seed_n = [0u64; 4];
    let mut d_work = [0u64; N];
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
    ) {
        return false;
    }

    if shl_buf(&mut work[..work_len], sh) != 0 {
        return false;
    }

    rcp.copy_from_slice(&work[NR_RCP_GUARD_LIMBS..work_len]);
    true
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn div_rem_2_1_x86(q: &mut u64, r: &mut u64, d: u64) {
    asm!(
        "div rcx",
        inout("rax") *q,
        inout("rdx") *r,
        in("rcx") d,
        options(pure, nomem, nostack),
    );
}

#[inline(always)]
unsafe fn div_rem_2_1_asm(q: &mut u64, r: &mut u64, d: u64) {
    #[cfg(target_arch = "aarch64")]
    {
        let val = ((*r as u128) << 64) | (*q as u128);
        let d_u128 = d as u128;
        *r = (val % d_u128) as u64;
        *q = (val / d_u128) as u64;
    }

    #[cfg(target_arch = "x86_64")]
    {
        div_rem_2_1_x86(q, r, d);
    }
}

pub fn div_prim(buf: &mut [u64], prim: u64) -> u64 {
    if prim == 0 {
        panic!("Division by zero error")
    }
    if prim == 1 {
        return 0;
    }

    let mut r = 0;
    for q in buf.iter_mut().rev() {
        unsafe {
            div_rem_2_1_asm(q, &mut r, prim);
        }
    }

    return r;
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sub_mul_of_aarch(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    let overflow: u64;
    asm!(
        "mov {mc}, xzr",
        "2:",
        "ldr {w}, [{win}]",
        "ldr {dv}, [{den}], #8",
        "mul   {lo}, {dv}, {q}",
        "umulh {hi}, {dv}, {q}",
        "adds {lo}, {lo}, {mc}",
        "adc  {mc}, {hi}, xzr",
        "subs {w}, {w}, {lo}",
        "cinc {mc}, {mc}, cc",
        "str {w}, [{win}], #8",
        "subs {len}, {len}, #1",
        "cbnz {len}, 2b",
        "ldr {w}, [{ofp}]",
        "subs {w}, {w}, {mc}",
        "cset {overflow}, cc",
        "str {w}, [{ofp}]",
        win = inout(reg) win => _,
        den = inout(reg) d => _,
        ofp = in(reg) of,
        q = in(reg) q,
        len = inout(reg) len => _,
        overflow = out(reg) overflow,
        mc = out(reg) _,
        w = out(reg) _,
        dv = out(reg) _,
        lo = out(reg) _,
        hi = out(reg) _,
        options(nostack),
    );
    overflow != 0
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sub_mul_of_x86(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    let borrow: u8;
    asm!(
        "mov {mc}, 0",
        "2:",
        "mov rax, [{den}]",
        "mul {q}",
        "add rax, {mc}",
        "adc rdx, 0",
        "sub QWORD PTR [{win}], rax",
        "adc rdx, 0",
        "mov {mc}, rdx",
        "lea {win}, [{win} + 8]",
        "lea {den}, [{den} + 8]",
        "dec {len}",
        "jnz 2b",
        "sub QWORD PTR [{ofp}], {mc}",
        "setc {b}",
        win = inout(reg) win => _,
        den = inout(reg) d => _,
        len = inout(reg) len => _,
        ofp = in(reg) of,
        q = in(reg) q,
        b = out(reg_byte) borrow,
        mc = out(reg) _,
        out("rax") _,
        out("rdx") _,
        options(nostack),
    );
    borrow != 0
}

#[inline(always)]
unsafe fn sub_mul_of_asm(win: *mut u64, of: *mut u64, d: *const u64, q: u64, len: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        sub_mul_of_aarch(win, of, d, q, len)
    }

    #[cfg(target_arch = "x86_64")]
    {
        sub_mul_of_x86(win, of, d, q, len)
    }
}

#[inline(always)]
fn sub_mul_of(win: &mut [u64], of: &mut u64, d: &[u64], q: u64) -> bool {
    unsafe { sub_mul_of_asm(win.as_mut_ptr(), of, d.as_ptr(), q, d.len()) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn mul_u64_x86(a: u64, b: u64) -> (u64, u64) {
    let lo: u64;
    let hi: u64;
    asm!(
        "mul {tmp}",
        tmp = in(reg) b,
        inout("rax") a => lo,
        out("rdx") hi,
        options(nostack, nomem),
    );
    (hi, lo)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_u64_aarch(a: u64, b: u64) -> (u64, u64) {
    let lo: u64;
    let hi: u64;
    asm!(
        "mul {lo}, {a}, {b}",
        "umulh {hi}, {a}, {b}",
        a = in(reg) a,
        b = in(reg) b,
        lo = out(reg) lo,
        hi = out(reg) hi,
        options(nostack, nomem),
    );
    (hi, lo)
}

#[inline(always)]
pub unsafe fn mul_u64_asm(a: u64, b: u64) -> (u64, u64) {
    #[cfg(target_arch = "aarch64")]
    {
        mul_u64_aarch(a, b)
    }
    #[cfg(target_arch = "x86_64")]
    {
        mul_u64_x86(a, b)
    }
}

#[cfg(test)]
#[path = "../tests/test_div_nr.rs"]
mod test_div_nr;
