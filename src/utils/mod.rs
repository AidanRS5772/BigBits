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
}

thread_local! {
     pub(crate) static SCRATCH_POOL: RefCell<Scratch> = RefCell::new(Scratch { buf: Vec::new() })
}
