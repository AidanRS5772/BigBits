use std::cell::RefCell;

pub mod div;
pub mod mul;
pub mod utils;

pub(crate) struct Scratch {
    buf: Vec<u64>,
}

impl Scratch {
    fn get(&mut self, n: usize) -> &mut [u64] {
        if self.buf.len() < n {
            self.buf.fill(0);
            self.buf.resize(n, 0);
        } else {
            self.buf[..n].fill(0);
        }
        return &mut self.buf[..n];
    }

    fn ensure(&mut self, n: usize) {
        if self.buf.len() < n {
            self.buf.resize(n, 0);
        }
    }
}

thread_local! {
     pub(crate) static SCRATCH_POOL: RefCell<Scratch> = RefCell::new(Scratch { buf: Vec::new() })
}

pub const KARATSUBA_CUTOFF: usize = 27;
pub const FFT_CUTOFF: usize = 1024;
pub const FFT_CHUNKING: f64 = 2.5;
