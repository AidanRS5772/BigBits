use std::{
    panic,
    sync::{RwLock, RwLockWriteGuard},
    thread::{self, Scope, ScopedJoinHandle},
};

use crate::utils::{
    CapacityError, LN2_PAR_CUTOFF, ScratchGuard, bin_split::{
        BBPSeries, DynHyperCtx, DynNodeBBP, DynNodeHyper, HyperSeries, Node, Shift, StaticNodeBBP, WideHyperSeries, bin_split, join2, ratio_to_fraction_dyn, ratio_to_fraction_static, static_split_stack
    }, utils::add_buf
};

//$ ln(2) = \frac{2}{3} + 1/2  \sum^{\infty}_{n=1} \left(\frac{1}{2n} + \frac{1}{4n+1}+\frac{1}{8n+4}+\frac{1}{16n+12}\right)16^{-n}$
//$ ln(2) = \frac{2}{3} + 1/4  \sum^{\infty}_{n=1} \frac{1}{n 16^n} + 1/2 \sum^{\infty}_{n=1} \frac{1}{(4n+1)16^n} + 1/8 \sum^{\infty}_{n=1} \frac{1}{(2n+1)16^n} + 1/8 \sum^{\infty}_{n=1} \frac{1}{(4n+3)16^n}$

const LN2_CTX: Shift = Shift(4);

struct Ln2S1;
impl BBPSeries for Ln2S1 {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
}

struct Ln2S2;
impl BBPSeries for Ln2S2 {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        2 * n + 1
    }
}

struct Ln2S3;
impl BBPSeries for Ln2S3 {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        4 * n + 1
    }
}

struct Ln2S4;
impl BBPSeries for Ln2S4 {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        4 * n + 3
    }
}

struct DynLn2 {
    s1: Option<DynNodeBBP<Ln2S1>>,
    s2: Option<DynNodeBBP<Ln2S2>>,
    s3: Option<DynNodeBBP<Ln2S3>>,
    s4: Option<DynNodeBBP<Ln2S4>>,
    buf: Vec<u64>,
}

static LN2_DYN: RwLock<DynLn2> = RwLock::new(DynLn2::new());

const TERM_GAURD: u64 = 4;

impl DynLn2 {
    const fn new() -> Self {
        DynLn2 {
            s1: None,
            s2: None,
            s3: None,
            s4: None,
            buf: Vec::new(),
        }
    }

    fn get(&self, out: &mut [u64]) -> bool {
        if out.len() >= self.buf.len() {
            return false;
        }
        out.copy_from_slice(&self.buf[self.buf.len() - out.len()..]);
        true
    }

    fn request(&mut self, out: &mut [u64]) {
        fn add_series<S: BBPSeries>(s: &DynNodeBBP<S>, shift: u64, acc: &mut [u64], t: &mut [u64]) {
            ratio_to_fraction_dyn(&s.p, &s.q, shift as i64, t);
            let carry = add_buf(acc, t);
            debug_assert!(!carry, "ln(2) < 1");
        }

        fn extend<S: BBPSeries>(node: &mut Option<DynNodeBBP<S>>, n: u64) -> &DynNodeBBP<S> {
            let s = match node.take() {
                Some(old) => {
                    let new = bin_split::<DynNodeBBP<S>>(&LN2_CTX, old.b, n);
                    DynNodeBBP::merge(&LN2_CTX, old, new)
                }
                None => bin_split::<DynNodeBBP<S>>(&LN2_CTX, 1, n),
            };
            node.insert(s)
        }

        if self.get(out) {
            return;
        }
        let len = out.len() + 1;

        // Each term adds 4 bits.
        let n = 1 + 16 * len as u64 + TERM_GAURD;
        let par = len >= LN2_PAR_CUTOFF;
        let ((s1, s2), (s3, s4)) = join2(
            par,
            || join2(par, || extend(&mut self.s1, n), || extend(&mut self.s2, n)),
            || join2(par, || extend(&mut self.s3, n), || extend(&mut self.s4, n)),
        );

        let k = LN2_CTX.0 * (s1.b - s1.a);

        let mut scratch = ScratchGuard::acquire();
        let t = scratch.get(len);

        self.buf.clear();
        self.buf.resize(len, 0xAAAA_AAAA_AAAA_AAAA);

        add_series(s1, k + 2, &mut self.buf, t);
        add_series(s2, k + 3, &mut self.buf, t);
        add_series(s3, k + 1, &mut self.buf, t);
        add_series(s4, k + 3, &mut self.buf, t);

        out.copy_from_slice(&self.buf[1..]);
    }
}

pub fn ln2_dyn(out: &mut [u64]) {
    fn ln2_write() -> RwLockWriteGuard<'static, DynLn2> {
        LN2_DYN.write().unwrap_or_else(|e| {
            let mut cache = e.into_inner();
            *cache = DynLn2::new();
            LN2_DYN.clear_poison();
            cache
        })
    }

    if let Ok(cache) = LN2_DYN.read() {
        if cache.get(out) {
            return;
        }
    }
    let mut work = {
        let mut cache = ln2_write();
        if cache.get(out) {
            return;
        }
        DynLn2 {
            s1: cache.s1.take(),
            s2: cache.s2.take(),
            s3: cache.s3.take(),
            s4: cache.s4.take(),
            buf: Vec::new(),
        }
    };
    work.request(out);
    let mut cache = ln2_write();
    if work.buf.len() > cache.buf.len() {
        *cache = work;
    }
}

const LN2_BIN: &[u8] = include_bytes!("ln2.bin");
pub(crate) const LN2_LIMBS: usize = LN2_BIN.len() / 8;
const _: () = assert!(LN2_BIN.len() % 8 == 0, "ln2.bin must hold whole u64 limbs");

struct StaticLn2<const N: usize>;

impl<const N: usize> StaticLn2<N> {
    fn get(out: &mut [u64]) -> bool {
        if out.len() > LN2_LIMBS {
            return false;
        }
        let bytes = &LN2_BIN[8 * (LN2_LIMBS - out.len())..];
        for (l, c) in out.iter_mut().zip(bytes.chunks_exact(8)) {
            *l = u64::from_le_bytes(c.try_into().unwrap());
        }
        true
    }

    // Past the table, sums the series at out.len() + guard limbs.
    fn request(out: &mut [u64], guard: usize) -> Result<(), CapacityError> {
        fn add_series_static<const N: usize, S: BBPSeries>(
            s: &StaticNodeBBP<N, S>,
            shift: u64,
            acc: &mut [u64],
            t: &mut [u64],
        ) {
            let (p, q, k) = s.ratio(&LN2_CTX);
            ratio_to_fraction_static::<N>(p, q, k + shift as i64, t);
            let carry = add_buf(acc, t);
            debug_assert!(!carry, "ln(2) < 1");
        }

        #[inline(never)]
        fn split<const N: usize, S: BBPSeries>(n: u64) -> StaticNodeBBP<N, S> {
            bin_split::<StaticNodeBBP<N, S>>(&LN2_CTX, 1, n)
        }

        fn spawn_split<'s, const N: usize, S: BBPSeries + 's>(
            sc: &'s Scope<'s, '_>,
            n: u64,
        ) -> Option<ScopedJoinHandle<'s, StaticNodeBBP<N, S>>> {
            thread::Builder::new()
                .stack_size(static_split_stack(N))
                .spawn_scoped(sc, move || split::<N, S>(n))
                .ok()
        }

        fn join_split<const N: usize, S: BBPSeries>(
            h: Option<ScopedJoinHandle<'_, StaticNodeBBP<N, S>>>,
            n: u64,
        ) -> StaticNodeBBP<N, S> {
            match h {
                Some(h) => h.join().unwrap_or_else(|e| panic::resume_unwind(e)),
                None => split::<N, S>(n),
            }
        }

        if Self::get(out) {
            return Ok(());
        }
        let len = out.len() + guard;
        // Static finalize needs 2 limbs past its output.
        if len + 2 > N {
            return Err(CapacityError { min_n: len + 2 });
        }

        // Each term adds 4 bits.
        let n = 1 + 16 * len as u64 + TERM_GAURD;
        let (s1, s2, s3, s4) = if len >= LN2_PAR_CUTOFF {
            thread::scope(|sc| {
                let (h1, h2, h3, h4) = (
                    spawn_split::<N, Ln2S1>(sc, n),
                    spawn_split::<N, Ln2S2>(sc, n),
                    spawn_split::<N, Ln2S3>(sc, n),
                    spawn_split::<N, Ln2S4>(sc, n),
                );
                (
                    join_split(h1, n),
                    join_split(h2, n),
                    join_split(h3, n),
                    join_split(h4, n),
                )
            })
        } else {
            let (s1, s2) = (split::<N, Ln2S1>(n), split::<N, Ln2S2>(n));
            let (s3, s4) = (split::<N, Ln2S3>(n), split::<N, Ln2S4>(n));
            (s1, s2, s3, s4)
        };

        let (mut buf, mut t) = ([0xAAAA_AAAA_AAAA_AAAA; N], [0; N]);
        let (buf, t) = (&mut buf[..len], &mut t[..len]);

        add_series_static(&s1, 2, buf, t);
        add_series_static(&s2, 3, buf, t);
        add_series_static(&s3, 1, buf, t);
        add_series_static(&s4, 3, buf, t);

        out.copy_from_slice(&buf[guard..]);
        Ok(())
    }
}

// Past ln2.bin's LN2_LIMBS it needs N >= out.len() + 3, and returns a CapacityError
// below that.
pub fn ln2_static<const N: usize>(out: &mut [u64]) -> Result<(), CapacityError> {
    StaticLn2::<N>::request(out, 1)
}

// ln2_static without its guard limb, so past the table it needs only
// N >= out.len() + 2, but its four series' rounding can leave out a few ulps low.
// For callers that keep their own guard bits.
pub(crate) fn ln2_static_unguarded<const N: usize>(out: &mut [u64]) -> Result<(), CapacityError> {
    StaticLn2::<N>::request(out, 0)
}

//$1/\pi = \frac{\sqrt{10005}}{4270934400} \; \sum^{\infty}_{n=0} (-1)^{n} \frac{(6n)!}{(3n)!(n!)^3 640320^{3n}} (545140134 \cdot n + 13591409)$
//$P_1 / Q_1 = \sum^{\infty}_{n=0} (-1)^{n} \; n \prod^{n}_{r=1} \frac{(6r-5)(6r-1)(2r-1)}{r^3 \cdot 333833583375 \cdot 2^15}$
//$P_2 / Q_2 = \sum^{\infty}_{n=0} (-1)^{n} \prod^{n}_{r=1} \frac{(6r-5)(6r-1)(2r-1)}{r^3 \cdot 333833583375 \cdot 2^15}$
//$\pi = \frac{4270934400 \cdot Q_1 Q_2}{\sqrt{10005}(545140134 \cdot P_1 Q_2 + 13591409 \cdot P_2 Q_1)}$

const PI_POW: u128 = 333_833_583_375;
const PI_CTX: DynHyperCtx = DynHyperCtx::new_shift(15);
struct PI1;
impl WideHyperSeries for PI1 {
    const SIGNED: bool = true;
    fn p(n: u128) -> u128 {
        n * (6 * n - 5) * (6 * n - 1) * (2 * n - 1)
    }
    fn q(n: u128) -> u128{
        n * n * n * PI_POW
    }
    fn r(n: u128) -> u128{
        (6 * n - 5) * (6 * n - 1) * (2 * n - 1)
    }
    fn neg(n: u128) -> bool {
        (n & 1) == 1
    }
}

struct PI2;
impl WideHyperSeries for PI2 {
    const SIGNED: bool = true;
    fn p(n: u128) -> u128 {
        (6 * n - 5) * (6 * n - 1) * (2 * n - 1)
    }
    fn q(n: u128) -> u128{
        n * n * n * PI_POW
    }
    fn r(n: u128) -> u128{
        n * (6 * n - 5) * (6 * n - 1) * (2 * n - 1)
    }
    fn neg(n: u128) -> bool {
        (n & 1) == 1
    }
}

struct DynPI{
    s1: Option<DynNodeHyper<PI1>>,
    s2: Option<DynNodeHyper<PI2>>,
    buf: Vec<u64>,
}

impl DynPI{
    const fn new() -> DynPI{
        DynPI{
            s1: None, s2: None, buf: Vec::new(),
        }
    }
}

static PI_DYN: RwLock<DynPI> = RwLock::new(DynPI::new());
