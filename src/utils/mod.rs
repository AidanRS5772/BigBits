use std::cell::RefCell;

pub mod consts;
pub mod div;
pub mod mul;
pub mod series;
pub mod sqrt;
pub mod trans;
pub mod utils;

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

// The multiplication cutoffs below were measured on an i7-8750H with
// `benches/mul_cutoffs.rs` over three code layouts, on one pinned core except where
// the static NTT is the upper side, which ran with rayon's full pool. Where a call
// takes a few hundred ns or a few us (the schoolbook crossovers), 24 processes and an
// exact Wilcoxon test per shape (p < 0.05) set the cutoff where the faster side is
// significant at every larger size; larger calls compare averages over processes. For
// the cost models, which no single value fits shape by shape, the value is the
// smallest within noise of the least mean time lost. See docs/mul_cutoff_tuning.md.

// is_school: schoolbook below, Karatsuba above (one level over schoolbook leaves
// against mul_buf). Chunking Karatsuba (s <= half) is significantly faster from
// s = 64 at l = 2s, 4s and 16s alike. For s > half the parabola fits only roughly:
// squares are faster from 44, but shapes with s just above half need s >= about
// 0.7 l or s >= 66, so no c satisfies every shape. c = 52 is the smallest within
// noise of the least mean time lost over the measured shapes (0.25%); squares
// switch from 55.
pub const CHUNKING_KARATSUBA_CUTOFF: usize = 63;
pub const KARATSUBA_CUTOFF: f64 = 52.0;

// is_karatsuba's cost ratios, Karatsuba against fft_entry (mul_dyn, one core). FFT
// time steps up with its transform size, so the boundary is a sawtooth: squares are
// faster with the FFT from about 250 limbs except just past a step (307x307 is 11%
// slower), and 1.96 switches them from about 230. Mean time lost over the measured
// shapes: 0.19% (regular) and 0.33% (chunking).
pub const FFT_CHUNKING_KARATSUBA_CUTOFF: f64 = 1.88;
pub const FFT_KARATSUBA_CUTOFF: f64 = 1.96;
// The 16-bit FFT's f64 precision bound, not a speed choice.
pub const FFT_16BIT_CUTOFF: usize = 1 << 16;

// is_karatsuba's cost ratios, karatsuba_entry_static against ntt_entry_static with
// the NTT's three convolutions on rayon: squares switch from about 2500 limbs. On one
// core the NTT stays 1.4x slower to at least 3700 limbs.
pub const NTT_CHUNKING_KARATSUBA_CUTOFF: f64 = 5.9;
pub const NTT_KARATSUBA_CUTOFF: f64 = 5.7;

pub const NTT_PAR_CUTOFF_NTT_CONV: usize = 512;
pub const NTT_PAR_CUTOFF_NTT: usize = 3600;
pub const NTT_PAR_CUTOFF_NTT_3: usize = 27648;
pub const NTT_PAR_CUTOFF_NTT_5: usize = 25600;

// sqr_buf up to this many limbs, a Karatsuba square level above: one level over
// schoolbook leaves is significantly faster at every size from 72 to 192 (0.989 at
// 72), schoolbook to 67, and the two tie at 69 (dyn) and 71 (static).
pub const KARATSUBA_SQR_CUTOFF: usize = 71;
// sqr_dyn squares by Karatsuba up to this many limbs and by FFT above:
// karatsuba_sqr_entry_dyn is faster than fft_sqr_entry to 340 except at 288 and
// 316-336, just past FFT steps, and the FFT is faster at every size from 344 to 1024.
// (Schoolbook alone would last only to 208 against the FFT.)
pub const FFT_SQR_CUTOFF: usize = 340;
// sqr_static squares by Karatsuba up to this many limbs and by NTT above, with the
// NTT's convolutions on rayon: the NTT wins from about 3070, but loses at 3320 and
// 3884 just past its transform steps, and wins at every size from 4040 to 12113.
pub const STATIC_NTT_SQR_CUTOFF: usize = 3884;

// hi_mul_* and hi_sqr_* take hi_mul_buf / hi_sqr_buf for windows of up to this many
// limbs, and above them the guarded product of the operands' top limbs. Dyn and
// static differ because the guarded product is an FFT on the dyn path and Karatsuba
// on the static one. With operands one and two windows wide, the guarded dyn product
// is faster at every window from 320 (0.78) to 2048 (0.16) but loses at 304-312,
// just past an FFT step (1.09). The static one is faster from 392 (0.998; 0.88 at
// 512), and from 440 with operands that fill N (its split path). Guarded squares:
// dyn from 472 (0.91-0.97; some windows from 384 already win), static from 672 (0.995;
// 0.88 at 1024).
pub const DYN_PARTIAL_MUL_CUTOFF: usize = 312;
pub const STATIC_PARTIAL_MUL_CUTOFF: usize = 408;
pub const DYN_PARTIAL_SQR_CUTOFF: usize = 464;
pub const STATIC_PARTIAL_SQR_CUTOFF: usize = 640;

// hi_mul_buf and hi_sqr_buf use columns for windows of up to this many limbs
// and rows above: rows are significantly faster at every wider window up to 256
// limbs, with operands as wide as the window, which gives rows their shortest
// rows. Columns are significantly faster up to 22 and 69 limbs; in between the
// two are within about 1%, and code layout alone moves each build's crossover by
// up to 20 limbs. (Re-measured after the row kernels lost their bounds checks;
// the cutoffs were 30 and 117.)
pub const HI_MUL_ROW_CUTOFF: usize = 24;
pub const HI_SQR_ROW_CUTOFF: usize = 90;

// mul_add_prim runs its plain mul loop below this many limbs and its mulx/adcx
// loop from it, behind the ADX feature check, which costs about 2 ns a call. The
// plain loop is ahead through 16 limbs (14% against 11% faster than the old
// per-limb loop) and the mulx loop from 20 (17% against 15%), single runs.
pub const MUL_PRIM_ADX_CUTOFF: usize = 20;

// add_mul_limb and sub_mul_limb run their one-limb loops below this many limbs and the
// ADX rows from it, behind the feature check and an out-of-line call. On an i7-8750H (24
// paired processes, exact Wilcoxon) the ADX row is faster from 9 limbs for add and 12 for
// sub; 10 favors add, which has most of the callers (sub loses 8% at 10 and 4% at 11).
pub const MUL_LIMB_ADX_CUTOFF: usize = 10;

// mid_mul_dyn uses mid_mul_buf below this and fft_mid_mul from it: the FFT is
// significantly faster at every size from 125 to 320, but at 122-124 its transform
// size steps up and it loses by up to 7.5%.
pub const FFT_MID_CUTOFF: usize = 125;
// mid_mul_static uses mid_mul_buf below this and ntt_mid_mul_static from it, with the
// NTT's convolutions on rayon: the NTT is faster at every size from 687 to 3024, and
// loses at 654-671 just past a transform step (on one core it would be 997).
pub const NTT_MID_CUTOFF: usize = 672;

pub const DIV_KNUTH_CUTOFF: usize = 68;
pub const BZ_CUTOFF: usize = 140;
pub const BZ_TOP_PADDED_COST_SCALE: f64 = 1.80;

pub const DYN_DIV_KARATSUBA_FFT_NR_BZ_CUTOFF: usize = 112;
pub const DYN_DIV_KARATSUBA_NR_BZ_CUTOFF: f64 = 1.5;
pub const DYN_DIV_FFT_NR_BZ_CUTOFF: f64 = 6.98;

pub const DYN_DIV_REM_KARATSUBA_FFT_NR_BZ_CUTOFF: usize = 778;
pub const DYN_DIV_REM_KARATSUBA_NR_BZ_CUTOFF: f64 = 4.4;
pub const DYN_DIV_REM_FFT_NR_BZ_CUTOFF: f64 = 7.15;

pub const DYN_RCP_KNUTH_NR_CUTOFF: usize = 36;

pub const STATIC_DIV_KARATSUBA_NTT_NR_BZ_CUTOFF: usize = 842;
pub const STATIC_DIV_KARATSUBA_NR_BZ_CUTOFF: f64 = 0.75;
pub const STATIC_DIV_NTT_NR_BZ_CUTOFF: f64 = 11.66;

pub const STATIC_DIV_REM_KARATSUBA_NTT_NR_BZ_CUTOFF: usize = 112;
pub const STATIC_DIV_REM_KARATSUBA_NR_BZ_CUTOFF: f64 = 4.4;
pub const STATIC_DIV_REM_NTT_NR_BZ_CUTOFF: f64 = 11.66;

pub const STATIC_RCP_KNUTH_NR_CUTOFF: usize = 36;

pub const ZIMMERMAN_SQRT_LEAF_CUTOFF: usize = 11;
pub const DYN_SQRT_REM_ZIMMERMAN_CUTOFF: usize = 16;
pub const DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF: usize = 11;
pub const DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF: usize = 11;
pub const STATIC_SQRT_REM_ZIMMERMAN_CUTOFF: usize = 16;
pub const STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF: usize = 11;
pub const STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF: usize = 11;

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

// series_eval predicts each eligible algorithm's time from its kernels' operation counts
// (series.rs's Work: Big-O in the call's terms, precision, shifts, x and N, with structural
// pass, product and array counts) and takes the fastest. These are the ns per unit of each
// count, shared by every series kind and both allocation modes, fitted by non-negative
// least squares on relative error over serial timings on an Apple M1 (aarch64). See
// docs/series_cutoff_tuning.md.
// A limb of a div_prim or div_prim2 pass, or a Knuth quotient limb.
pub const SERIES_NS_DIV: f64 = 3.574;
// A limb times scalar limb of a linear pass: a product or multiply-add by a scalar, an
// addition, a negation, a clear or a copy.
pub const SERIES_NS_SCALAR: f64 = 0.1574;
// A schoolbook limb product of a multi-limb product (mul::mul_work).
pub const SERIES_NS_PRODUCT: f64 = 0.3847;
// A Horner or rect term's group and weight bookkeeping.
pub const SERIES_NS_TERM: f64 = 9.334;
// A kernel call: a pass, product or division.
pub const SERIES_NS_CALL: f64 = 0.9307;
// A heap allocation or scratch acquisition.
pub const SERIES_NS_ALLOC: f64 = 9.377;
// A limb of a static [u64; N] array zero-filled or moved.
pub const SERIES_NS_FILL: f64 = 0.09114;

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

// exp_dyn runs its core on r 2^-j, j = this less the bits by which r already sits below 1,
// and squares the result back j times, unless the core would sum its chunks in parallel.
// 0 disables them. `cargo bench-bin-split exp` (1 to 4096 limbs, serial) put 24 within 1.05x
// of each size's best fixed j, 1.01x on the geometric mean; a log2(precision) scale fit
// worse (1.07x, 1.26x at 1 to 4 limbs), as the best j hardly moves with precision.
// Re-swept with the trees' rayon split gated on two or more threads: 24 was still best,
// 1.013x on the geometric mean and within 1.07x of each size's best (worst at 512).
pub const DYN_EXP_SQR_BITS: u64 = 24;
// GUESS: the dynamic value. exp_static only takes the squares when N has room for them.
pub const STATIC_EXP_SQR_BITS: u64 = 24;

// ln_dyn takes floor(A log2(64 out.len())) square roots before its series, less the bits
// by which its argument already sits near 1, so the series' argument starts that far down.
// 0 disables them. `cargo bench-bin-split ln` (2 to 2048 limbs) tied 0.25 and 0.5 within
// noise, 1.06x and 1.08x of each size's best; 0 was 1.29x and 1 was 1.27x.
pub const DYN_LN_SQRT_A: f64 = 0.25;
// GUESS: the dynamic value. ln_static only takes the square roots when N has room for them.
pub const STATIC_LN_SQRT_A: f64 = 0.25;

// ln_dyn_entry's first chunk after its top bit spans at least this many bits. Unlike exp's,
// an atanh chunk's term count depends only on its offset, so this starts at plain
// doubling. GUESS
pub const DYN_LN_CHUNK_MIN_BITS: u64 = 2;
pub const STATIC_LN_CHUNK_MIN_BITS: u64 = 2;

// ln_dyn_entry and ln_static_entry go parallel as exp's entries do. GUESS
pub const DYN_LN_CHUNK_PAR_WORK: usize = 512;
pub const STATIC_LN_CHUNK_PAR_WORK: usize = 768;

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
