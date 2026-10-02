use std::sync::{RwLock, RwLockWriteGuard};

use crate::utils::{
    bin_split::{bin_split, ratio_to_fraction_dyn, BBPSeries, DynNodeBBP, Node, Shift},
    mul::mul_prim,
    utils::{add_buf, buf_len, shl_buf, split_sh},
    ScratchGuard, LN2_TERM_CUTOFF,
};

//$ ln(2) = \frac{2}{3} + 1/2  \sum^{\infty}_{n=1} \left(\frac{1}{2n} + \frac{1}{4n+1}+\frac{1}{8n+4}+\frac{1}{16n+12}\right)16^{-n}$
//$ ln(2) = \frac{2}{3} + 3/8 \sum^{\infty}_{n=1} \frac{3n+2}{n(4n+3)} 16^{-n} + 1/8 \sum^{\infty}_{n=1} \frac{12n+5}{(4n+1)(2n+1)} 16^{-n}$
//$ \frac{16*Q1*Q2 + 3*(3*P1*Q2 + P2*Q1)}{8*3*Q1*Q2}$

const LN2_CTX: Shift = Shift(4);

struct Ln2S1;
impl BBPSeries for Ln2S1 {
    fn p(n: u64) -> u64 {
        3 * n + 2
    }
    fn q(n: u64) -> u64 {
        n * (4 * n + 3)
    }
}

struct Ln2S2;
impl BBPSeries for Ln2S2 {
    fn p(n: u64) -> u64 {
        12 * n + 5
    }
    fn q(n: u64) -> u64 {
        (4 * n + 1) * (2 * n + 1)
    }
}

struct DynLn2 {
    s1: Option<DynNodeBBP<Ln2S1>>,
    s2: Option<DynNodeBBP<Ln2S2>>,
    buf: Vec<u64>,
}

static LN2: RwLock<DynLn2> = RwLock::new(DynLn2::new());

const TERM_GAURD: u64 = 4;

impl DynLn2 {
    const fn new() -> Self {
        DynLn2 {
            s1: None,
            s2: None,
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
        if self.get(out) {
            return;
        }
        let len = out.len() + 1;

        let n = 1 + 16 * len as u64 + TERM_GAURD;
        let a = self.s1.as_ref().map_or(1, |s| s.b);
        let new_s1 = bin_split::<DynNodeBBP<Ln2S1>, LN2_TERM_CUTOFF>(&LN2_CTX, a, n);
        let new_s2 = bin_split::<DynNodeBBP<Ln2S2>, LN2_TERM_CUTOFF>(&LN2_CTX, a, n);
        let s1 = match self.s1.take() {
            Some(old) => DynNodeBBP::merge(&LN2_CTX, old, new_s1),
            None => new_s1,
        };
        let s2 = match self.s2.take() {
            Some(old) => DynNodeBBP::merge(&LN2_CTX, old, new_s2),
            None => new_s2,
        };

        // T = P / (Q * 2^k) for each node.
        let k = LN2_CTX.0 * (s1.b - s1.a);
        let (p1, q1) = (&s1.p, &s1.q);
        let (sl, sb) = split_sh(k + 4);
        let num_len = (sl + q1.len() + 1).max(p1.len() + 1) + 1;

        let mut scratch = ScratchGuard::acquire();
        let [num, p9, den, t2] = scratch.get_splits([num_len, p1.len() + 1, q1.len() + 1, len]);

        // 2/3 + 3/8 * T1 = ((Q1 << (k + 4)) + 9 * P1) / (3 * Q1 * 2^(k + 3))
        num.fill(0);
        num[sl..sl + q1.len()].copy_from_slice(q1);
        num[sl + q1.len()] = shl_buf(&mut num[sl..sl + q1.len()], sb);
        p9[..p1.len()].copy_from_slice(p1);
        p9[p1.len()] = mul_prim(&mut p9[..p1.len()], 9);
        add_buf(num, p9);
        den[..q1.len()].copy_from_slice(q1);
        den[q1.len()] = mul_prim(&mut den[..q1.len()], 3);

        self.buf.clear();
        self.buf.resize(len, 0);
        ratio_to_fraction_dyn(
            &num[..buf_len(num)],
            &den[..buf_len(den)],
            (k + 3) as i64,
            &mut self.buf,
        );

        // 1/8 * T2 = P2 / (Q2 * 2^(k + 3))
        ratio_to_fraction_dyn(&s2.p, &s2.q, (k + 3) as i64, t2);
        let carry = add_buf(&mut self.buf, t2);
        debug_assert!(!carry, "ln(2) < 1");

        out.copy_from_slice(&self.buf[1..]);
        self.s1 = Some(s1);
        self.s2 = Some(s2);
    }
}

fn ln2_write() -> RwLockWriteGuard<'static, DynLn2> {
    LN2.write().unwrap_or_else(|e| {
        let mut cache = e.into_inner();
        *cache = DynLn2::new();
        LN2.clear_poison();
        cache
    })
}

pub fn ln2_dyn(out: &mut [u64]) {
    if let Ok(cache) = LN2.read() {
        if cache.get(out) {
            return;
        }
    }
    ln2_write().request(out);
}
