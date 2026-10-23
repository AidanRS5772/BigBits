use std::cell::RefCell;

pub mod bin_split;
pub mod consts;
pub mod div;
pub mod mul;
pub mod sqrt;
pub mod utils;
pub mod trans;

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

// The dynamic constant caches (ln2_dyn, pi_dyn) copy requests of up to their tables' 1024
// limbs from them, and past that build in parallel (rayon) when
// new * (cached^2 + HALF^2) >= CUTOFF * HALF^2, in output limbs: from CUTOFF new limbs on
// a cold cache, with the threshold halving at HALF cached limbs and falling with its
// square past that. Extending a cache also merges into the cached nodes and finalizes at
// the full length, which both cost more the larger the cache is, so a few new limbs on a
// large cache are worth parallelizing. Waking rayon's workers from outside the pool
// costs about 15-20 us. ln2 runs its four series' shares in parallel (measured on the
// earlier base-16 series): ties cold at ~31 limbs, warm at new ~ 20, 14, 6 and 1 for 16,
// 32, 64 and 128 cached. pi spawns its one series into the pool, where its own top
// merges can run in parallel, while the caller builds the root: ties cold at 96-128
// limbs, warm at new ~ 45, 8 and 1 for 256, 1024 and 4096 cached. Each thread keeps its
// own FFT plans and scratch, so the first request after a switch between the serial and
// parallel paths pays a few us once (ln2 32 -> 42 and 128 -> 129 after a parallel cold
// build, about 1.05x). Past the tables every ln2 build is parallel, and pi's from about 7
// new limbs; the smaller ties matter only if a table shrinks. See
// docs/bin_split_cutoff_tuning.md.
pub const DYN_LN2_PAR_CUTOFF: u64 = 32;
pub const DYN_LN2_PAR_HALF: u64 = 30;
pub const DYN_PI_PAR_CUTOFF: u64 = 104;
pub const DYN_PI_PAR_HALF: u64 = 256;

// ln2_static sums and finalizes each of its four series on its own scoped thread from
// this many limbs (out.len() plus its guard; measured on the earlier base-16 series).
// Four threads cost about 60 us to start: they lost at 32 limbs (1.34x) and won from 40
// (0.95x, 0.56x at 64). Below ln2.bin's 1024 limbs it copies the table instead, so this
// only matters if the table shrinks.
pub const STATIC_LN2_PAR_CUTOFF: usize = 40;

// pi_static builds its series and its root on two scoped threads from this many limbs
// below pi's integer limb (out.len() - 1 plus its guard), which only overlaps the root's
// cost with the series: they lost at 384 (1.05x) and won from 512 (0.98x, 0.93x at
// 1000). Below pi.bin's 1024 limbs it copies the table instead, so past it the series
// always gets a thread, and its stack sized for N.
pub const STATIC_PI_PAR_CUTOFF: usize = 480;

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

// exp_dyn_entry evaluates its chunk series in parallel once it has at least two chunks
// and chunks * working limbs reaches this, on a pool of two or more threads. Chunks
// cost about the same at a given precision, so this is one decision for the whole
// loop. See docs/bin_split_cutoff_tuning.md.
pub const DYN_EXP_CHUNK_PAR_WORK: usize = 512;

// exp_dyn_entry's first chunk after x's top bit spans at least this many bits; later chunks
// each span [s, ~2s). Narrower chunks near the top all sum about as many terms, so each
// costs about as much as a wide one. See docs/bin_split_cutoff_tuning.md.
pub const DYN_EXP_CHUNK_MIN_BITS: u64 = 16;

// exp_static_entry's counterparts: it gives each chunk series its own scoped thread once
// it has at least two chunks and chunks * working limbs reaches this. A thread
// costs about 10 us to start, far more than a rayon task: threads tied at 585 and
// won from 774 (0.67x) to 13325 (0.34x).
pub const STATIC_EXP_CHUNK_PAR_WORK: usize = 768;
pub const STATIC_EXP_CHUNK_MIN_BITS: u64 = 16;

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
