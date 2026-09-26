use std::marker::PhantomData;

use crate::{
    shl_buf, trim_lz,
    utils::{
        mul::{hi_mul_dyn, hi_mul_static, mul_dyn, mul_static},
        utils::{add_buf, buf_len, sub_buf, twos_comp},
        ScratchGuard,
    },
};

trait Node {
    fn merge(l: Self, r: Self) -> Self;
    fn leaf(a: u64, b: u64) -> Self;
}

fn bin_split<N, const CUTOFF: u64>(a: u64, b: u64) -> N
where
    N: Node,
{
    const { assert!(CUTOFF != 0) };
    debug_assert!(a < b);
    if b - a < CUTOFF {
        return N::leaf(a, b);
    }
    let m = (a + b) / 2;
    let l = bin_split::<N, CUTOFF>(a, m);
    let r = bin_split::<N, CUTOFF>(m, b);
    N::merge(l, r)
}

trait BBPSeries {
    const R: u64;
    fn p(n: u64) -> u64;
    fn q(n: u64) -> u64;
}

#[derive(Debug, Clone)]
pub struct DynNodeBBP<S: BBPSeries> {
    a: u64,
    b: u64,
    p: Vec<u64>,
    q: Vec<u64>,
    _s: PhantomData<S>,
}

impl<S: BBPSeries> Node for DynNodeBBP<S> {
    fn leaf(a: u64, b: u64) -> Self {
        let p = vec![S::p(a)];
        let q = vec![S::q(a)];
        for k in (a+1)..b{
            
        }
    }

    fn merge(l: Self, r: Self) -> Self {
        fn fit(mut v: Vec<u64>, n: usize) -> Vec<u64> {
            if v.capacity() < n {
                return vec![0u64; n];
            }
            v.clear();
            v.resize(n, 0);
            v
        }

        let DynNodeBBP {
            a,
            b: m,
            p: lp,
            q: lq,
            _s: _,
        } = l;
        let DynNodeBBP {
            a: _,
            b,
            p: rp,
            q: rq,
            _s: _,
        } = r;

        let sh = S::R * (b - m); // r(b - m)
        let (sl, sb) = ((sh / 64) as usize, (sh % 64) as u32);

        let pl = lp.len() + rq.len();
        let tl = lq.len() + rp.len();
        let ql = lq.len() + rq.len();

        let mut p;

        {
            let mut gaurd = ScratchGuard::acquire();
            let [t1, t2] = gaurd.get_splits([pl, tl]);
            mul_dyn(&lp, &rq, t1);
            let of = if sb != 0 { shl_buf(t1, sb as u8) } else { 0 };

            p = fit(lp, (sl + pl + (of != 0) as usize).max(tl));
            p[sl..sl + pl].copy_from_slice(t1);
            if of != 0 {
                p[sl + pl] = of;
            }

            mul_dyn(&lq, &rp, t2);
            if add_buf(&mut p, t2) {
                p.push(1);
            }
            trim_lz(&mut p);
        }

        let mut q = fit(rp, ql);
        mul_dyn(&lq, &rq, &mut q);
        trim_lz(&mut q);

        DynNodeBBP {
            a,
            b,
            p,
            q,
            _s: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct StaticNodeBBP<const N: usize> {
    p: [u64; N],
    q: [u64; N],
    neg: bool,
}
