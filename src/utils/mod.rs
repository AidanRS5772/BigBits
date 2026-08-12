use std::cell::RefCell;

pub mod div;
pub mod mul;
pub mod sqrt;
pub mod utils;

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

pub const BZ_CUTOFF: usize = 88;
pub const BZ_TOP_PADDED_COST_SCALE: f64 = 0.295;

pub const DYN_DIV_KARATSUBA_FFT_NR_BZ_CUTOFF: usize = 1568;
pub const DYN_DIV_KARATSUBA_NR_BZ_CUTOFF: f64 = 0.31;
pub const DYN_DIV_FFT_NR_BZ_CUTOFF: f64 = 7.84;

pub const DYN_DIV_REM_KARATSUBA_FFT_NR_BZ_CUTOFF: usize = 1280;
pub const DYN_DIV_REM_KARATSUBA_NR_BZ_CUTOFF: f64 = 0.58;
pub const DYN_DIV_REM_FFT_NR_BZ_CUTOFF: f64 = 8.70;

pub const DYN_RCP_KNUTH_NR_CUTOFF: usize = 8;

pub const STATIC_DIV_KARATSUBA_NTT_NR_BZ_CUTOFF: usize = 1664;
pub const STATIC_DIV_KARATSUBA_NR_BZ_CUTOFF: f64 = 0.65;
pub const STATIC_DIV_NTT_NR_BZ_CUTOFF: f64 = 9.485;

pub const STATIC_DIV_REM_KARATSUBA_NTT_NR_BZ_CUTOFF: usize = 3328;
pub const STATIC_DIV_REM_KARATSUBA_NR_BZ_CUTOFF: f64 = 0.82;
pub const STATIC_DIV_REM_NTT_NR_BZ_CUTOFF: f64 = 10.5;

pub const STATIC_RCP_KNUTH_NR_CUTOFF: usize = 100;

pub const ZIMMERMAN_SQRT_CUTOFF: usize = 17;

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
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.buf);
        SCRATCH_POOL.with(|p| p.borrow_mut().push(buf));
    }
}
