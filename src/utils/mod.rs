use std::cell::RefCell;

pub mod bin_split;
pub mod consts;
pub mod div;
pub mod mul;
pub mod sqrt;
pub mod utils;
pub mod trans;

pub const CHUNKING_KARATSUBA_CUTOFF: usize = 22;
pub const KARATSUBA_CUTOFF: f64 = 17.0;

pub const FFT_CHUNKING_KARATSUBA_CUTOFF: f64 = 1.74;
pub const FFT_KARATSUBA_CUTOFF: f64 = 1.92;
pub const FFT_16BIT_CUTOFF: usize = 1 << 16;

pub const NTT_CHUNKING_KARATSUBA_CUTOFF: f64 = 3.58;
pub const NTT_KARATSUBA_CUTOFF: f64 = 3.78;

pub const NTT_PAR_CUTOFF_NTT_CONV: usize = 512;
pub const NTT_PAR_CUTOFF_NTT: usize = 3600;
pub const NTT_PAR_CUTOFF_NTT_3: usize = 27648;
pub const NTT_PAR_CUTOFF_NTT_5: usize = 25600;

pub const KARATSUBA_SQR_CUTOFF: usize = 14;
pub const FFT_SQR_CUTOFF: usize = 33;
pub const STATIC_NTT_SQR_CUTOFF: usize = 1014;

pub const PARTIAL_MUL_CUTOFF: usize = 90;
pub const PARTIAL_SQR_CUTOFF: usize = 128;

pub const FFT_MID_CUTOFF: usize = 90;
pub const NTT_MID_CUTOFF: usize = 200;

pub const DIV_KNUTH_CUTOFF: usize = 88;
pub const BZ_CUTOFF: usize = 216;
pub const BZ_TOP_PADDED_COST_SCALE: f64 = 1.60;

pub const DYN_DIV_KARATSUBA_FFT_NR_BZ_CUTOFF: usize = 352;
pub const DYN_DIV_KARATSUBA_NR_BZ_CUTOFF: f64 = 0.60;
pub const DYN_DIV_FFT_NR_BZ_CUTOFF: f64 = 5.50;

pub const DYN_DIV_REM_KARATSUBA_FFT_NR_BZ_CUTOFF: usize = 192;
pub const DYN_DIV_REM_KARATSUBA_NR_BZ_CUTOFF: f64 = 24.0;
pub const DYN_DIV_REM_FFT_NR_BZ_CUTOFF: f64 = 6.17;

pub const DYN_RCP_KNUTH_NR_CUTOFF: usize = 11;

pub const STATIC_DIV_KARATSUBA_NTT_NR_BZ_CUTOFF: usize = 1664;
pub const STATIC_DIV_KARATSUBA_NR_BZ_CUTOFF: f64 = 0.65;
pub const STATIC_DIV_NTT_NR_BZ_CUTOFF: f64 = 9.485;

pub const STATIC_DIV_REM_KARATSUBA_NTT_NR_BZ_CUTOFF: usize = 3328;
pub const STATIC_DIV_REM_KARATSUBA_NR_BZ_CUTOFF: f64 = 0.82;
pub const STATIC_DIV_REM_NTT_NR_BZ_CUTOFF: f64 = 10.5;

pub const STATIC_RCP_KNUTH_NR_CUTOFF: usize = 100;

pub const ZIMMERMAN_SQRT_LEAF_CUTOFF: usize = 15;
pub const DYN_SQRT_REM_ZIMMERMAN_CUTOFF: usize = 20;
pub const DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF: usize = 15;
pub const DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF: usize = 15;
pub const STATIC_SQRT_REM_ZIMMERMAN_CUTOFF: usize = 20;
pub const STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF: usize = 15;
pub const STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF: usize = 14;

// ln(2) computes its four series in parallel from this many output limbs.
pub const LN2_PAR_CUTOFF: usize = 40;

// Binary splitting builds a range as one leaf once its node's estimated size
// (Node::limbs) drops below LEAF_LIMBS, or it has at most LEAF_TERMS terms.
// Hyper's terms can each multiply by a wide x, which makes its leaves cost
// more per limb, so it also keeps a few terms per leaf. See
// docs/bin_split_cutoff_tuning.md.
pub const DYN_BBP_LEAF_LIMBS: u64 = 64;
pub const STATIC_BBP_LEAF_LIMBS: u64 = 512;
pub const DYN_ENGEL_LEAF_LIMBS: u64 = 96;
pub const STATIC_ENGEL_LEAF_LIMBS: u64 = 128;
pub const DYN_HYPER_LEAF_LIMBS: u64 = 48;
pub const DYN_HYPER_LEAF_TERMS: u64 = 8;
pub const STATIC_HYPER_LEAF_LIMBS: u64 = 64;
pub const STATIC_HYPER_LEAF_TERMS: u64 = 8;

// Dynamic binary splitting builds a split's two halves in parallel from this
// many estimated limbs, and a merge's independent products once its Q reaches
// this many limbs.
pub const DYN_BIN_SPLIT_PAR_CUTOFF: u64 = 256;
pub const DYN_BIN_SPLIT_MERGE_PAR_CUTOFF: usize = 256;

// exp_dyn evaluates its chunk series in parallel once it has at least two chunks
// and chunks * working limbs reaches this, on a pool of two or more threads. Chunks
// cost about the same at a given precision, so this is one decision for the whole
// loop. See docs/bin_split_cutoff_tuning.md.
pub const DYN_EXP_CHUNK_PAR_WORK: usize = 512;

// exp_dyn's first chunk after x's top bit spans at least this many bits; later chunks
// each span [s, ~2s). Narrower chunks near the top all sum about as many terms, so each
// costs about as much as a wide one. See docs/bin_split_cutoff_tuning.md.
pub const DYN_EXP_CHUNK_MIN_BITS: u64 = 16;

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
