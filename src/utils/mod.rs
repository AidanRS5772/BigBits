use std::cell::RefCell;

pub mod div;
pub mod mul;
pub mod utils;

pub const CHUNKING_KARATSUBA_CUTOFF: usize = 15;
pub const KARATSUBA_CUTOFF: usize = 21;

pub const FFT_CHUNKING_KARATSUBA_CUTOFF: f64 = 1.834;
pub const FFT_KARATSUBA_CUTOFF: f64 = 1.907;
pub const FFT_BIT_CUTOFF: usize = 1 << 17; // GUESS

pub const DYN_NTT_CUTOFF: usize = 1 << 20; // GUESS
pub const NTT_CHUNKING_KARATSUBA_CUTOFF: f64 = 2.5; // GUESS
pub const NTT_KARATSUBA_CUTOFF: f64 = 2.75; // GUESS

pub const FFT_SQR_BIT_CUTOFF: usize = 1 << 16; // GUESS
pub const KARATSUBA_SQR_CUTOFF: usize = 26; // GUESS
pub const FFT_SQR_CUTOFF: usize = 100; // GUESS

pub const SHORT_MUL_CUTOFF: usize = 22; // GUESS

pub const KARATSUBA_MID_CUTOFF: usize = 30; // GUESS
pub const FFT_MID_CUTOFF: usize = 60; // GUESS

pub const BZ_CUTOFF: usize = 64; // GUESS

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
