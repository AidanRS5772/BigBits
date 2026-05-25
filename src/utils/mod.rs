use std::cell::RefCell;

pub mod div;
pub mod mul;
pub mod utils;

pub const CHUNKING_KARATSUBA_CUTOFF: usize = 22;
pub const KARATSUBA_CUTOFF: f64 = 19.5;

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

pub const SHORT_MUL_CUTOFF: usize = 114;
pub const SHORT_SQR_CUTOFF: usize = 128;

pub const FFT_MID_CUTOFF: usize = 90;
pub const NTT_MID_CUTOFF: usize = 200;

pub const BZ_CUTOFF: usize = 120;

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
        let tot: usize = sizes.iter().sum();
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
