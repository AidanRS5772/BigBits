use crate::{
    shl_buf, trim_lz,
    utils::{
        mul::{hi_mul_dyn, hi_mul_static, mul_dyn, mul_static},
        utils::{add_buf, buf_len, sub_buf, twos_comp},
        ScratchGuard,
    },
};

// ─── Original exploration (kept for reference) ─────────────────────────────
//
// use crate::{
//     add_buf, buf_len, cmp_buf, shl_buf, sub_buf, trim_lz, twos_comp,
//     utils::{mul::mul_dyn, ScratchGuard},
// };
//
// // Binary Split - BBP type
// struct Node<'a> {
//     p: &'a mut [u64],
//     p_len: usize,
//     q: &'a mut [u64],
//     q_len: usize,
//     neg: bool,
// }
//
// fn bbp_merge<'a>(x: &Node<'a>, y: &Node<'a>) -> Node<'a> {
//     let a_len = x.p_len + y.q_len;
//     let b_len = x.q_len + y.p_len;
//
//     let mut new_p = vec![0u64; a_len.max(b_len) + 1];
//     mul_dyn(x.p, y.q, &mut new_p);
//
//     let mut guard = ScratchGuard::acquire();
//     let t = guard.get(b_len);
//     mul_dyn(x.q, y.p, t);
//
//     let mut new_neg = x.neg;
//     if x.neg ^ y.neg {
//         if sub_buf(&mut new_p, t) {
//             twos_comp(&mut new_p);
//             new_neg = y.neg;
//         }
//     } else {
//         add_buf(&mut new_p, t);
//     }
//     trim_lz(&mut new_p);
//     new_neg &= !new_p.is_empty();
//
//     let mut new_q = vec![0; x.q_len + y.q_len];
//     mul_dyn(x.q, y.q, &mut new_q);
//     trim_lz(&mut new_q);
//
//     return Node {
//         p: &mut new_p,
//         p_len: new_p.len(),
//         q: &mut new_q,
//         q_len: new_q.len(),
//         neg: new_neg,
//     };
// }
//
// fn bin_split_bbp<'a, F>(a: u64, b: u64, term: &F) -> Node<'a>
// where
//     F: Fn(u64) -> Node<'a>,
// {
//     if b - a == 1 {
//          return term(a);
//     }
//     let m = (a + b) / 2;
//     let x = bin_split_bbp(a, m, term);
//     let y = bin_split_bbp(m, b, term);
//     bbp_merge(&x, &y)
// }

// ─── Storage abstraction ───────────────────────────────────────────────────

/// Owned limb storage for a binary-splitting node. The storage type also selects the
/// multiplication family, so `Vec<u64>` runs the dynamic kernels and `[u64; N]` the static ones.
// pub trait Limbs: Sized {
//     /// Largest logical length this storage can hold.
//     const CAP: usize;
//
//     /// Zeroed storage holding at least `len` limbs.
//     fn zeroed(len: usize) -> Self;
//     fn limbs(&self) -> &[u64];
//     fn limbs_mut(&mut self) -> &mut [u64];
//
//     fn mul(a: &[u64], b: &[u64], out: &mut [u64]) -> u64;
//     fn hi_mul(a: &[u64], b: &[u64], out: &mut [u64]) -> u64;
// }
//
// impl Limbs for Vec<u64> {
//     const CAP: usize = usize::MAX;
//
//     fn zeroed(len: usize) -> Self {
//         vec![0; len]
//     }
//     fn limbs(&self) -> &[u64] {
//         self
//
//     }
//     fn limbs_mut(&mut self) -> &mut [u64] {
//         self
//     }
//
//     fn mul(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
//         mul_dyn(a, b, out)
//     }
//     fn hi_mul(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
//         hi_mul_dyn(a, b, out)
//     }
// }
//
// impl<const N: usize> Limbs for [u64; N] {
//     const CAP: usize = N;
//
//     fn zeroed(len: usize) -> Self {
//         assert!(len <= N, "node needs {len} limbs but static storage holds {N}");
//         [0; N]
//     }
//     fn limbs(&self) -> &[u64] {
//         self
//     }
//     fn limbs_mut(&mut self) -> &mut [u64] {
//         self
//     }
//
//     fn mul(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
//         mul_static::<N>(a, b, out)
//     }
//     fn hi_mul(a: &[u64], b: &[u64], out: &mut [u64]) -> u64 {
//         hi_mul_static::<N>(a, b, out)
//     }
// }
//
// // ─── Node ──────────────────────────────────────────────────────────────────
//
// /// A signed rational `(-1)^neg * P / Q` covering one interval of a series. The fields
// /// `p_len`/`q_len` are the trimmed logical lengths; the backing storage may be longer.
// pub struct Node<B: Limbs> {
//     p: B,
//     p_len: usize,
//     q: B,
//     q_len: usize,
//     neg: bool,
// }
//
// impl<B: Limbs> Node<B> {
//     pub fn from_prims(p: u64, q: u64, neg: bool) -> Self {
//         assert!(q != 0, "node denominator must be nonzero");
//         let mut pb = B::zeroed(1);
//         let mut qb = B::zeroed(1);
//         pb.limbs_mut()[0] = p;
//         qb.limbs_mut()[0] = q;
//         Node {
//             p: pb,
//             p_len: (p != 0) as usize,
//             q: qb,
//             q_len: 1,
//             neg: neg && p != 0,
//         }
//     }
//
//     pub fn p(&self) -> &[u64] {
//         &self.p.limbs()[..self.p_len]
//     }
//     pub fn q(&self) -> &[u64] {
//         &self.q.limbs()[..self.q_len]
//     }
//     pub fn neg(&self) -> bool {
//         self.neg
//     }
// }
//
// // ─── BBP-type merge ────────────────────────────────────────────────────────
//
// /// Writes limbs `drop..a.len() + b.len()` of `a * b` into `out` and returns that window's width.
// /// When `drop > 0` the lowest limbs of the window come from a high product and are approximate.
// fn mul_window<B: Limbs>(a: &[u64], b: &[u64], drop: usize, out: &mut [u64]) -> usize {
//     if a.is_empty() || b.is_empty() {
//         return 0;
//     }
//     let full = a.len() + b.len();
//     if drop >= full {
//         return 0;
//     }
//     let w = full - drop;
//     if drop == 0 {
//         B::mul(a, b, &mut out[..w]);
//     } else {
//         out[w - 1] = B::hi_mul(a, b, &mut out[..w - 1]);
//     }
//     w
// }
//
// /// Sums two nodes: `P = Px*Qy ± Py*Qx`, `Q = Qx*Qy`.
// ///
// /// When `Qx*Qy` would exceed `cap` limbs, the same number of low limbs is dropped from
// /// all three products. `P/Q` is invariant under a common scale, so this keeps about `cap`
// /// limbs of relative precision without an exponent. `cap = usize::MAX` keeps it exact.
// ///
// /// Storage requirement: `cap + 2 <= B::CAP`, and every node satisfies `p_len <= q_len + 1`
// /// (partial sums below `2^64` in magnitude).
// pub fn bbp_merge<B: Limbs>(x: &Node<B>, y: &Node<B>, cap: usize) -> Node<B> {
//     let (xp, xq, yp, yq) = (x.p(), x.q(), y.p(), y.q());
//     let drop = (xq.len() + yq.len()).saturating_sub(cap);
//
//     let mut q = B::zeroed(xq.len() + yq.len() - drop);
//     mul_window::<B>(xq, yq, drop, q.limbs_mut());
//     let q_len = buf_len(q.limbs());
//
//     let t1_w = (xp.len() + yq.len()).saturating_sub(drop);
//     let t2_w = (yp.len() + xq.len()).saturating_sub(drop);
//     // +1 limb: the add can carry out, and sub_buf needs lhs >= rhs regardless of which is longer.
//     let p_w = t1_w.max(t2_w) + 1;
//
//     let mut p = B::zeroed(p_w);
//     let mut t = B::zeroed(t2_w);
//     mul_window::<B>(xp, yq, drop, p.limbs_mut());
//     mul_window::<B>(yp, xq, drop, t.limbs_mut());
//
//     let acc = &mut p.limbs_mut()[..p_w];
//     let rhs = &t.limbs()[..t2_w];
//     let mut neg = x.neg;
//     if x.neg != y.neg {
//         if sub_buf(acc, rhs) {
//             twos_comp(acc);
//             neg = y.neg;
//         }
//     } else {
//         add_buf(acc, rhs);
//     }
//     let p_len = buf_len(acc);
//
//     Node {
//         p,
//         p_len,
//         q,
//         q_len,
//         neg: neg && p_len != 0,
//     }
// }
//
// // ─── Driver ────────────────────────────────────────────────────────────────
//
// /// Sums `term(k)` over `k in [a, b)` by binary splitting. Each node lives only in its own
// /// call frame and its parent's, so peak memory is one pending node per recursion level.
// ///
// /// `cap` bounds the denominator width (see `bbp_merge`); it is clamped to what `B` can hold.
// /// Pass `usize::MAX` with `Vec<u64>` for an exact result. The final value is `P/Q`.
// pub fn bbp_split<B, F>(a: u64, b: u64, cap: usize, term: &F) -> Node<B>
// where
//     B: Limbs,
//     F: Fn(u64) -> Node<B>,
// {
//     assert!(a < b, "empty interval");
//     let cap = cap.min(B::CAP.saturating_sub(2)).max(1);
//     bbp_split_rec(a, b, cap, term)
// }
//
// fn bbp_split_rec<B, F>(a: u64, b: u64, cap: usize, term: &F) -> Node<B>
// where
//     B: Limbs,
//     F: Fn(u64) -> Node<B>,
// {
//     if b - a == 1 {
//         return term(a);
//     }
//     let m = a + (b - a) / 2;
//     let x = bbp_split_rec(a, m, cap, term);
//     let y = bbp_split_rec(m, b, cap, term);
//     bbp_merge(&x, &y, cap)
// }

trait Node {
    fn merge(l: Self, r: Self) -> Self;
}

fn bin_split<F, N, const CUTOFF: u64>(a: u64, b: u64, terms: &F) -> N
where
    N: Node,
    F: Fn(u64, u64) -> N,
{
    const { assert!(CUTOFF != 0) };
    debug_assert!(a < b);
    if b - a < CUTOFF {
        return terms(a, b);
    }
    let m = (a + b) / 2;
    let l = bin_split::<F, N, CUTOFF>(a, m, terms);
    let r = bin_split::<F, N, CUTOFF>(m, b, terms);
    N::merge(l, r)
}

#[derive(Debug, Clone)]
pub struct DynNodeBBP<const R: u64> {
    a: u64,
    b: u64,
    p: Vec<u64>,
    q: Vec<u64>,
}

impl<const R: u64> Node for DynNodeBBP<R> {
    fn merge(l: Self, r: Self) -> Self {
        let DynNodeBBP {
            a,
            b: m,
            p: lp,
            q: lq,
        } = l;
        let DynNodeBBP {
            a: _,
            b,
            p: rp,
            q: rq,
        } = r;

        let sh = R * (b - m); // r(b - m)
        let (sl, sb) = ((sh / 64) as usize, (sh % 64) as u32);

        let p_len = lp.len() + rq.len();
        let t_len = lq.len() + rp.len();
        let q_len = lq.len() + rq.len();
        let mut p = vec![0u64; (sl + p_len  + 1).max(tl) + 1];

        mul_dyn(&x.p, &y.q, &mut p[sl..sl + xl]);
        if sb != 0 {
            p[sl + xl] = shl_buf(&mut p[sl..sl + xl], sb as u8);
        }
        {
            let mut g = ScratchGuard::acquire();
            let t = g.get(tl);
            mul_dyn(&x.q, &y.p, t);
            let c = add_buf(&mut p, t);
            debug_assert!(!c); // headroom limb absorbs it
        }
        trim_lz(&mut p);

        let mut q = vec![0u64; x.q.len() + y.q.len()];
        mul_dyn(&x.q, &y.q, &mut q);
        trim_lz(&mut q);

        DynNodeBBP {
            a: x.a,
            b: y.b,
            p,
            q,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct StaticNodeBBP<const N: usize> {
    p: [u64; N],
    q: [u64; N],
    neg: bool,
}
