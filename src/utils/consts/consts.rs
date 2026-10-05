use std::sync::{RwLock, RwLockWriteGuard};

use crate::utils::{
    bin_split::{
        bin_split, ratio_to_fraction_dyn, ratio_to_fraction_static, BBPSeries, DynNodeBBP, Node,
        Shift, StaticNodeBBP,
    },
    mul::mul_prim,
    utils::{add_buf, buf_len, shl_buf, split_sh},
    ScratchGuard, LN2_TERM_CUTOFF,
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

static LN2: RwLock<DynLn2> = RwLock::new(DynLn2::new());

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
                    let new = bin_split::<DynNodeBBP<S>, LN2_TERM_CUTOFF>(&LN2_CTX, old.b, n);
                    DynNodeBBP::merge(&LN2_CTX, old, new)
                }
                None => bin_split::<DynNodeBBP<S>, LN2_TERM_CUTOFF>(&LN2_CTX, 1, n),
            };
            node.insert(s)
        }

        if self.get(out) {
            return;
        }
        let len = out.len() + 1;

        // Each term adds 4 bits.
        let n = 1 + 16 * len as u64 + TERM_GAURD;
        let ((s1, s2), (s3, s4)) = rayon::join(
            || rayon::join(|| extend(&mut self.s1, n), || extend(&mut self.s2, n)),
            || rayon::join(|| extend(&mut self.s3, n), || extend(&mut self.s4, n)),
        );

        let k = LN2_CTX.0 * (s1.b - s1.a);
        let (p1, q1) = (&s1.p, &s1.q);
        let (sl, sb) = split_sh(k + 3);
        let num_len = (sl + q1.len() + 1).max(p1.len() + 1) + 1;

        let mut scratch = ScratchGuard::acquire();
        let [num, p3, den, t] = scratch.get_splits([num_len, p1.len() + 1, q1.len() + 1, len]);

        num.fill(0);
        num[sl..sl + q1.len()].copy_from_slice(q1);
        num[sl + q1.len()] = shl_buf(&mut num[sl..sl + q1.len()], sb);
        p3[..p1.len()].copy_from_slice(p1);
        p3[p1.len()] = mul_prim(&mut p3[..p1.len()], 3);
        add_buf(num, p3);
        den[..q1.len()].copy_from_slice(q1);
        den[q1.len()] = mul_prim(&mut den[..q1.len()], 3);

        self.buf.clear();
        self.buf.resize(len, 0);
        ratio_to_fraction_dyn(
            &num[..buf_len(num)],
            &den[..buf_len(den)],
            (k + 2) as i64,
            &mut self.buf,
        );

        add_series(s2, k + 3, &mut self.buf, t);
        add_series(s3, k + 1, &mut self.buf, t);
        add_series(s4, k + 3, &mut self.buf, t);

        out.copy_from_slice(&self.buf[1..]);
    }
}

pub fn ln2_dyn(out: &mut [u64]) {
    fn ln2_write() -> RwLockWriteGuard<'static, DynLn2> {
        LN2.write().unwrap_or_else(|e| {
            let mut cache = e.into_inner();
            *cache = DynLn2::new();
            LN2.clear_poison();
            cache
        })
    }

    if let Ok(cache) = LN2.read() {
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

    fn request(out: &mut [u64]) {
        fn add_series_static<const N: usize, S: BBPSeries>(
            s: &StaticNodeBBP<N, S>,
            shift: u64,
            acc: &mut [u64],
            t: &mut [u64],
        ) {
            ratio_to_fraction_static::<N>(&s.p[..s.p_len], &s.q[..s.q_len], shift as i64, t);
            let carry = add_buf(acc, t);
            debug_assert!(!carry, "ln(2) < 1");
        }

        fn split<const N: usize, S: BBPSeries>(n: u64) -> StaticNodeBBP<N, S> {
            bin_split::<StaticNodeBBP<N, S>, LN2_TERM_CUTOFF>(&LN2_CTX, 1, n)
        }

        if Self::get(out) {
            return;
        }
        let len = out.len() + 1;
        assert!(
            len + 3 <= N,
            "ln(2) past the table needs out.len() + 4 <= N"
        );

        // Each term adds 4 bits. The splits run serially on the caller's stack,
        // as a rayon worker blocked in join can steal and stack another split.
        let n = 1 + 16 * len as u64 + TERM_GAURD;
        let (s1, s2) = (split::<N, Ln2S1>(n), split::<N, Ln2S2>(n));
        let (s3, s4) = (split::<N, Ln2S3>(n), split::<N, Ln2S4>(n));

        let (mut buf, mut t) = ([0xAAAA_AAAA_AAAA_AAAA; N], [0; N]);
        let (buf, t) = (&mut buf[..len], &mut t[..len]);

        add_series_static(&s1, 2, buf, t);
        add_series_static(&s2, 3, buf, t);
        add_series_static(&s3, 1, buf, t);
        add_series_static(&s4, 3, buf, t);

        out.copy_from_slice(&buf[1..]);
    }
}

pub fn ln2_static<const N: usize>(out: &mut [u64]) {
    StaticLn2::<N>::request(out);
}
