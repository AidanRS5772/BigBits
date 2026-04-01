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

    pub fn get_splits<const N: usize>(&mut self, sizes: [usize; N]) -> [&mut [T]; N] {
        let tot = sizes.iter().sum();
        if self.buf.len() < tot {
            self.buf.resize(tot, T::default());
        }
        let base = self.buf.as_mut_ptr();
        let mut offset = 0usize;
        std::array::from_fn(|i| {
            let size = sizes[i];
            let s = unsafe { std::slice::from_raw_parts_mut(base.add(offset), size) };
            offset += size;
            s
        })
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
