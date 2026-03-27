use std::cell::RefCell;

pub mod div;
pub mod mul;
pub mod utils;

pub(crate) struct Scratch<T> {
    buf: Vec<T>,
}

impl<T: Default + Copy> Scratch<T> {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn get(&mut self, n: usize) -> &mut [T] {
        if self.buf.len() < n {
            self.buf.resize(n, T::default());
        }
        &mut self.buf[..n]
    }

    fn ensure(&mut self, n: usize) {
        if self.buf.len() < n {
            self.buf.resize(n, T::default());
        }
    }
}

thread_local! {
     pub(crate) static SCRATCH_POOL: RefCell<Scratch<u64>> = RefCell::new(Scratch::new())
}

pub const KARATSUBA_CUTOFF: usize = 27;
pub const FFT_CUTOFF: usize = 1024;
pub const FFT_CHUNKING: f64 = 2.5;
