use std::cell::RefCell;

pub mod consts;
pub mod div;
pub mod mul;
pub mod series;
pub mod sqrt;
pub mod theta_plan;
pub mod trans;
pub mod utils;
pub mod verify;

// A static entry's request needs more limbs than its N provides. This is a limit of
// the stack buffers, not an arithmetic failure: the same call succeeds once N >= min_n.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityError {
    pub min_n: usize,
}

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "request needs a static capacity of N >= {}", self.min_n)
    }
}

impl std::error::Error for CapacityError {}

// Row kernels on out[..l.len()]: += s l (add) and = s l (set) return the limb that belongs at
// out[l.len()], -= s l (sub) the borrow owed by it. Base runs on every CPU of the architecture
// (AArch64 asm, plain Rust elsewhere); Adx's mulx/adcx/adox rows (x86) assemble only inside
// bmi2/adx functions, which can't inline into others. The impls live in mul.rs with the asm.
pub(crate) trait Rows {
    unsafe fn add(s: u64, l: &[u64], out: &mut [u64]) -> u64;
    unsafe fn set(s: u64, l: &[u64], out: &mut [u64]) -> u64;
    unsafe fn sub(s: u64, l: &[u64], out: &mut [u64]) -> u64;
}

pub(crate) struct Base;

#[cfg(target_arch = "x86_64")]
pub(crate) struct Adx;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn adx_available() -> bool {
    is_x86_feature_detected!("bmi2") && is_x86_feature_detected!("adx")
}

// rows_dispatch!(fn name(args) -> ret = body) defines name, which runs body::<Adx> inside a
// bmi2/adx copy when the CPU has both (and the optional `if cond` holds), else body::<Base>.
// body is a safe function generic over K: Rows, written once; each copy fixes its rows at
// compile time, so the asm inlines into the whole operation and only the CPU check runs per
// call of name.
macro_rules! rows_dispatch {
    ($(#[$attr:meta])* $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty
        = $body:ident $(if $cond:expr)?) => {
        $(#[$attr])*
        $vis fn $name($($arg: $ty),*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "bmi2,adx")]
                fn adx($($arg: $ty),*) -> $ret {
                    $body::<$crate::utils::Adx>($($arg),*)
                }
                if true $(&& $cond)? && $crate::utils::adx_available() {
                    return unsafe { adx($($arg),*) };
                }
            }
            // The rows every CPU runs, optimized on their own before being inlined, with name's
            // attributes, so an #[inline(always)] name inlines this copy into its callers too.
            $(#[$attr])*
            fn base($($arg: $ty),*) -> $ret {
                $body::<$crate::utils::Base>($($arg),*)
            }
            base($($arg),*)
        }
    };
    ($(#[$attr:meta])* $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?)
        = $body:ident $(if $cond:expr)?) => {
        $crate::utils::rows_dispatch!(
            $(#[$attr])* $vis fn $name($($arg: $ty),*) -> () = $body $(if $cond)?
        );
    };
}
pub(crate) use rows_dispatch;

// The multiplication cutoffs, cost ratios and parallel gates are knobs (sections `base` and `mul`
// in src/tuning/schema.rs) since 2026-11-08; the soundness limits below are not.

// When the 16-bit FFT entries check a result. A call whose largest rounding distance
// (|x − round(x)| over every coefficient) passes this runs the fold check (mul.rs's fft_fold_agrees),
// keeps its result if that passes, and is redone on the NTT if it fails. A wrong limb needs a
// coefficient off by more than 0.5, which reads 1 − e, so a wrong result that reads at or below this
// needs a coefficient off by at least 0.75 and every other one within 0.25. A high reading alone
// proves nothing: it shows noise, and most such results are right.
pub const FFT_FOLD_TRIGGER: f64 = 0.25;
// Whether fft_entry and fft_sqr_entry also check every result by its residues mod the NTT primes
// (utils::verify) in release builds, redoing a mismatch on the NTT. Debug builds always do. It
// catches any wrong product whose error fits in 188 bits, whatever the rounding read, and costs a
// pass over the operands and the result: see docs/fft_precision.md for its share of the FFT's
// time. Middle products and squares mod B^m − 1 have no such check.
pub const FFT_RESIDUE_CHECK: bool = false;
// The 16-bit FFT entries' dispatch limits, past which the dynamic dispatchers take the NTT. The
// product, square and middle-product limits are the largest the assert below allows, so they're set
// by soundness, not speed: one limb more and a coefficient can reach 2^48. All three land on the
// 2^19-digit transform (524288). There the FFT beat the NTT with random operands on the M1: about 4×
// on one thread, and 1.3–2× against the NTT on the default 8-thread pool (docs/fft_precision.md).
// Products and squares used to stop at 458752; worst-case operands haven't been swept on 524288 for
// them yet (middle products were).
// Products: output limbs l + s − 1 (was 105839).
pub const FFT_16BIT_CUTOFF: usize = 131066;
// Squares: operand limbs (was 55296).
pub const FFT_SQR_16BIT_CUTOFF: usize = 65533;
// Middle products: short limbs n (was 64826, on the same transform).
pub const FFT_MID_16BIT_CUTOFF: usize = 65533;
// Squares mod B^m − 1: modulus limbs m, a bnm1_size. This one is still the old speed limit: the
// largest size below the first transform where uniformly random operands, in unsigned digits,
// tripped the old check (a reading of 0.4 or a 2^48 coefficient gate) more than 1 in 2^10 times.
// That was 64800 (transform 259200: 11 trips in 2^14, all at the old gate; 17 in 8845 at 259308,
// from the same gate), measured before find_fft_size's stricter sizes, and is no longer a
// bnm1_size. This is the largest one below it, so the dispatch is the same as it would be at 64800.
// The next one, 65536, would fail the assert below by 8.
pub const FFT_BNM1_16BIT_CUTOFF: usize = 57344;
// Why no coefficient gate is needed: a balanced coefficient sums at most 4 min(l, s) digit products
// of magnitude at most 2^30, plus a little for the operands' unsigned top digits, which is under
// 2^48 at every limit (2^47.99 for products, squares and middle products). Below 2^48 a
// coefficient's error is a multiple of at most 1/16, so readings stay fine enough for the fold
// check's trigger to mean something. Raising a limit past this needs a new answer to that, not only
// a new measurement.
const _: () = {
    let room = 1u64 << 18;
    assert!(4 * ((FFT_16BIT_CUTOFF as u64 + 1) / 2) + 8 < room);
    assert!(4 * FFT_SQR_16BIT_CUTOFF as u64 + 8 < room);
    assert!(4 * FFT_MID_16BIT_CUTOFF as u64 + 8 < room);
    assert!(4 * FFT_BNM1_16BIT_CUTOFF as u64 + 8 < room);
};

// The division and reciprocal cutoffs and cost models are knobs (section `div` in
// src/tuning/schema.rs) since 2026-11-08.

// The square-root cutoffs are knobs (section `sqrt` in src/tuning/schema.rs) since 2026-11-08.


// The constant caches' parallel gates are knobs (section `consts` in src/tuning/schema.rs) since
// 2026-11-08.

// The binary-splitting leaf and parallel cutoffs and the series cost model's coefficients are
// knobs (sections `bin_split` and `series` in src/tuning/schema.rs) since 2026-11-08.

// exp's, ln's, the AGM's and theta's knobs are in sections `exp`, `ln`, `agm` and `theta` of
// src/tuning/schema.rs since 2026-11-08; THETA_STATIC_SLOTS, a capacity, stays here.
// [u64; N] slots theta_static keeps for its powers. Its least-work plan must fit (to T = 9) and
// its parallel plan runs only if it fits too (to T = 6).
pub const THETA_STATIC_SLOTS: usize = 8;

thread_local! {
    static SCRATCH_POOL: RefCell<Vec<Vec<u64>>> = RefCell::new(Vec::new());
}

pub struct ScratchGuard {
    buf: Vec<u64>,
}

impl ScratchGuard {
    pub fn acquire() -> Self {
        let buf = SCRATCH_POOL.with(|p| p.borrow_mut().pop().unwrap_or_default());
        Self { buf }
    }

    pub fn get(&mut self, n: usize) -> &mut [u64] {
        if self.buf.len() < n {
            self.buf.resize(n, 0);
        }
        &mut self.buf[..n]
    }

    pub fn get_splits<const N: usize>(&mut self, sizes: [usize; N]) -> [&mut [u64]; N] {
        let tot = sizes
            .iter()
            .try_fold(0usize, |sum, &size| sum.checked_add(size))
            .expect("scratch split size overflow");
        if self.buf.len() < tot {
            self.buf.resize(tot, 0);
        }
        let base = self.buf.as_mut_ptr();
        let mut offset = 0usize;
        std::array::from_fn(|i| unsafe {
            let s = std::slice::from_raw_parts_mut(base.add(offset), sizes[i]);
            offset += sizes[i];
            s
        })
    }

    // get_splits for a count known only at run time.
    pub fn get_split_vec(&mut self, sizes: impl Iterator<Item = usize> + Clone) -> Vec<&mut [u64]> {
        let tot = sizes
            .clone()
            .try_fold(0usize, |sum, size| sum.checked_add(size))
            .expect("scratch split size overflow");
        let mut rest = self.get(tot);
        sizes
            .map(|size| {
                let (s, r) = std::mem::take(&mut rest).split_at_mut(size);
                rest = r;
                s
            })
            .collect()
    }
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.buf);
        SCRATCH_POOL.with(|p| p.borrow_mut().push(buf));
    }
}
