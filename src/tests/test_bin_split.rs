use crate::utils::bin_split::*;
use crate::utils::consts::consts::{ln2_dyn, ln2_static, LN2_LIMBS};
use crate::utils::utils::{add_prim, cmp_buf, sub_buf, trim_lz};
use crate::utils::LN2_TERM_CUTOFF;
use std::marker::PhantomData;

// A small leaf cutoff gives deep trees, so most work runs through merges.
const C: u64 = 16;

// Dynamic nodes are exact, unreduced fractions, so any split of [a, b), serial
// or parallel, must match one leaf summing the whole range bit for bit.

struct BbpLin;
impl BBPSeries for BbpLin {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        4 * n + 3
    }
}

struct BbpQuad;
impl BBPSeries for BbpQuad {
    fn p(n: u64) -> u64 {
        3 * n + 2
    }
    fn q(n: u64) -> u64 {
        n * (4 * n + 3)
    }
}

struct EngelFact;
impl EngelSeries for EngelFact {
    fn p(n: u64) -> u64 {
        n % 3
    }
    fn q(n: u64) -> u64 {
        n
    }
}

struct Hyp;
impl HyperSeries for Hyp {
    fn p(n: u64) -> u64 {
        n + 1
    }
    fn q(n: u64) -> u64 {
        2 * n + 2
    }
    fn r(n: u64) -> u64 {
        2 * n + 1
    }
}

fn check_bbp<S: BBPSeries>(shift: u64, a: u64, b: u64) {
    let ctx = Shift(shift);
    let want = DynNodeBBP::<S>::leaf(&ctx, a, b, true);
    for _ in 0..3 {
        let got = bin_split::<DynNodeBBP<S>, C>(&ctx, a, b);
        assert_eq!((got.a, got.b), (want.a, want.b));
        assert!(got.p == want.p && got.q == want.q, "BBP [{a}, {b}) differs");
    }
}

#[test]
fn test_bbp_tree_matches_single_leaf() {
    for b in [2, 17, 300, 2500, 9000] {
        check_bbp::<BbpLin>(4, 1, b);
    }
    check_bbp::<BbpQuad>(4, 1, 6000);
    check_bbp::<BbpLin>(70, 5, 3000);
}

#[test]
fn test_engel_tree_matches_single_leaf() {
    for b in [3, 40, 2500, 9000] {
        let ctx = Shift(1);
        let want = DynNodeEngel::<EngelFact>::leaf(&ctx, 2, b, true);
        for _ in 0..3 {
            let got = bin_split::<DynNodeEngel<EngelFact>, C>(&ctx, 2, b);
            assert!(got.p == want.p && got.q == want.q, "Engel [2, {b}) differs");
        }
    }
}

fn check_hyper(x: &[u64], shift: u64, b: u64) {
    let ctx = DynHyperCtx::new(x, shift);
    for last in [true, false] {
        let want = DynNodeHyper::<Hyp>::leaf(&ctx, 0, b, last);
        for _ in 0..3 {
            let got = bin_split_tree::<DynNodeHyper<Hyp>, C>(&ctx, 0, b, last);
            assert!(
                got.p == want.p && got.q == want.q && got.r == want.r,
                "Hyper x={x:?} [0, {b}) last={last} differs"
            );
        }
    }
}

#[test]
fn test_hyper_tree_matches_single_leaf() {
    for b in [2, 50, 2500, 6000] {
        check_hyper(&[1], 8, b);
    }
    check_hyper(&[0x9e37_79b9_7f4a_7c15, 0xff_ffff_ffff], 112, 3000);
    check_hyper(&[12345], 20, 3000);
}

// floor(ln(2) * 2^(64 * 32)), little-endian.
const LN2_32: [u64; 32] = [
    0x5f50b5185064c18b,
    0x078f735d1b2db31b,
    0xae313cdb6c606cb1,
    0x955d5179b1e17b9d,
    0x0c480a5417350d2c,
    0x074db6015cfe7aa3,
    0x6a9c7f8a5e148e82,
    0x25669b333564a337,
    0x4c1a1e0bd1d6095d,
    0xcccc4e659393514c,
    0xc943e732b479cd33,
    0x17460775db8990e5,
    0x7d2e23de1400b396,
    0xee569d6dfc1efa15,
    0x610d30f88fe551a2,
    0x07f4ca11fb5bfb90,
    0xda2d97c50f3fd5c6,
    0x655fa1872f20e3a2,
    0xf5dfa6bd38303248,
    0x72ce87b19d6548ca,
    0x256fa0ec7657f74b,
    0xb9ea9bc3b136603b,
    0x1acbda11317c387e,
    0x3e96ca16224ae8c5,
    0x27573b291169b825,
    0xed2eae35c1382144,
    0x559552fb4afa1b10,
    0xe7b876206debac98,
    0x8a0d175b8baafa2b,
    0x40f343267298b62d,
    0xc9e3b39803f2f6af,
    0xb17217f7d1cf79ab,
];

#[test]
fn test_ln2_dyn_concurrent_requests() {
    use rayon::prelude::*;
    // floor(ln(2) * B^m) is the top m limbs of floor(ln(2) * B^M) for m <= M.
    let mut full = vec![0; 600];
    ln2_dyn(&mut full);
    assert_eq!(&full[full.len() - 32..], &LN2_32[..]);
    let lens: Vec<usize> = (0..64).map(|i| 1 + (i * 37) % 700).collect();
    lens.par_iter().for_each(|&l| {
        let mut out = vec![0; l];
        ln2_dyn(&mut out);
        let m = l.min(full.len());
        assert_eq!(&out[l - m..], &full[full.len() - m..], "ln2 at {l} limbs");
    });
    let mut big = vec![0; 700];
    ln2_dyn(&mut big);
    assert_eq!(&big[big.len() - 600..], &full[..]);
}

// ln2.bin is ln2_dyn's output at LN2_LIMBS, and the static cache serves any
// request up to that length from it whatever N is.
#[test]
fn test_ln2_static_table() {
    let mut full = vec![0; LN2_LIMBS];
    ln2_dyn(&mut full);
    assert_eq!(&full[LN2_LIMBS - 32..], &LN2_32[..]);
    for l in [0, 1, 31, 32, 33, 500, LN2_LIMBS - 1, LN2_LIMBS] {
        let mut out = vec![0; l];
        ln2_static::<8>(&mut out);
        assert_eq!(&out[..], &full[LN2_LIMBS - l..], "ln2 at {l} limbs");
    }
}

// Past the table the static cache sums the series itself, and its guard limb
// absorbs the static nodes' truncation, so it matches ln2_dyn exactly.
#[test]
fn test_ln2_static_past_table() {
    fn check<const N: usize>(l: usize) {
        let (mut want, mut out) = (vec![0; l], vec![0; l]);
        ln2_dyn(&mut want);
        ln2_static::<N>(&mut out);
        assert_eq!(out, want, "ln2 at {l} limbs, N = {N}");
    }
    check::<{ LN2_LIMBS + 5 }>(LN2_LIMBS + 1);
    check::<{ LN2_LIMBS + 16 }>(LN2_LIMBS + 1);
    check::<{ LN2_LIMBS + 16 }>(LN2_LIMBS + 7);
}

#[test]
#[should_panic(expected = "out.len() + 4 <= N")]
fn test_ln2_static_past_table_small_n() {
    ln2_static::<{ LN2_LIMBS + 4 }>(&mut [0; LN2_LIMBS + 1]);
}

// Static nodes keep only their top N limbs, so their finalize can differ from
// the exact dynamic one only in rounding: by at most one in the lowest limb.
fn check_static<D, S>(dyn_ctx: &D::Ctx, static_ctx: &S::Ctx, a: u64, b: u64)
where
    D: Node + Send,
    S: Node + Send,
    D::Ctx: Sync,
    S::Ctx: Sync,
{
    check_static_at::<D, S, C>(dyn_ctx, static_ctx, a, b, 13, true);
}

// check_static with the static tree split at cutoff CUT into len limbs.
fn check_static_at<D, S, const CUT: u64>(
    dyn_ctx: &D::Ctx,
    static_ctx: &S::Ctx,
    a: u64,
    b: u64,
    len: usize,
    last: bool,
) where
    D: Node + Send,
    S: Node + Send,
    D::Ctx: Sync,
    S::Ctx: Sync,
{
    let (mut want, mut got) = (vec![0u64; len], vec![0u64; len]);
    let want_neg = bin_split_tree::<D, C>(dyn_ctx, a, b, last).finalize(dyn_ctx, &mut want);
    let got_neg = bin_split_tree::<S, CUT>(static_ctx, a, b, last).finalize(static_ctx, &mut got);
    let what = format!("static [{a}, {b}) at cutoff {CUT}, {len} limbs");
    assert_eq!(got_neg, want_neg, "{what} has the wrong sign");
    // Limbs are little-endian, so order them from the top.
    let (lo, hi) = if got.iter().rev().le(want.iter().rev()) {
        (&got, &want)
    } else {
        (&want, &got)
    };
    let mut lo = lo.clone();
    add_prim(&mut lo, 1);
    assert!(got == want || &lo == hi, "{what} is off by more than one");
}

#[test]
fn test_static_nodes_match_dyn() {
    let ctx = Shift(4);
    for b in [2, 17, 300, 2500] {
        check_static::<DynNodeBBP<BbpLin>, StaticNodeBBP<16, BbpLin>>(&ctx, &ctx, 1, b);
        check_static::<DynNodeBBP<BbpLin>, StaticNodeBBP<64, BbpLin>>(&ctx, &ctx, 1, b);
    }
    check_static::<DynNodeBBP<BbpQuad>, StaticNodeBBP<16, BbpQuad>>(&ctx, &ctx, 1, 1000);
    let ctx = Shift(1);
    for b in [3, 40, 2500] {
        check_static::<DynNodeEngel<EngelFact>, StaticNodeEngel<16, EngelFact>>(&ctx, &ctx, 2, b);
        check_static::<DynNodeEngel<EngelFact>, StaticNodeEngel<64, EngelFact>>(&ctx, &ctx, 2, b);
    }
    for (x, shift) in [
        (vec![1u64], 8u64),
        (vec![12345], 20),
        (vec![0x9e37_79b9_7f4a_7c15, 0xff_ffff_ffff], 112),
    ] {
        for b in [2, 50, 2500] {
            let d = DynHyperCtx::new(&x, shift);
            let s16 = StaticHyperCtx::<16>::new(&x, shift);
            let s64 = StaticHyperCtx::<64>::new(&x, shift);
            check_static::<DynNodeHyper<Hyp>, StaticNodeHyper<16, Hyp>>(&d, &s16, 0, b);
            check_static::<DynNodeHyper<Hyp>, StaticNodeHyper<64, Hyp>>(&d, &s64, 0, b);
        }
    }
}

// Signed series: term n's sign by pattern K, and PART 0 is the whole signed
// series, 1 only its positive terms and 2 only its negative ones, unsigned.
fn negative<const K: u8>(n: u64) -> bool {
    match K {
        0 => n % 2 == 1,
        1 => true,
        2 => (n / 5) % 2 == 1,
        _ => n.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 63 == 1,
    }
}

fn part<const K: u8, const PART: u8>(n: u64, p: u64) -> u64 {
    match PART {
        0 => p,
        1 if !negative::<K>(n) => p,
        2 if negative::<K>(n) => p,
        _ => 0,
    }
}

// Magnitudes that vary enough, with small q's and shifts, for partial sums to
// change sign mid-leaf and mid-merge.
fn mag(n: u64) -> u64 {
    n.wrapping_mul(7919) % 1000 + 1
}

struct SBbp<const K: u8, const PART: u8>;
impl<const K: u8, const PART: u8> BBPSeries for SBbp<K, PART> {
    const SIGNED: bool = PART == 0;
    fn p(n: u64) -> u64 {
        part::<K, PART>(n, mag(n))
    }
    fn q(n: u64) -> u64 {
        1 + n % 2
    }
    fn neg(n: u64) -> bool {
        PART == 0 && negative::<K>(n)
    }
}

struct SEngel<const K: u8, const PART: u8>;
impl<const K: u8, const PART: u8> EngelSeries for SEngel<K, PART> {
    const SIGNED: bool = PART == 0;
    fn p(n: u64) -> u64 {
        part::<K, PART>(n, mag(n) - 1)
    }
    fn q(n: u64) -> u64 {
        if n < 200 {
            1
        } else {
            n
        }
    }
    fn neg(n: u64) -> bool {
        PART == 0 && negative::<K>(n)
    }
}

struct SHyp<const K: u8, const PART: u8>;
impl<const K: u8, const PART: u8> HyperSeries for SHyp<K, PART> {
    const SIGNED: bool = PART == 0;
    fn p(n: u64) -> u64 {
        part::<K, PART>(n, mag(n))
    }
    fn q(n: u64) -> u64 {
        n + 2
    }
    fn r(n: u64) -> u64 {
        n + 3
    }
    fn neg(n: u64) -> bool {
        PART == 0 && negative::<K>(n)
    }
}

// pos - neg in sign-magnitude form.
fn signed_diff(pos: &[u64], neg: &[u64]) -> (Vec<u64>, bool) {
    let flip = cmp_buf(pos, neg) == std::cmp::Ordering::Less;
    let (big, small) = if flip { (neg, pos) } else { (pos, neg) };
    let mut d = big.to_vec();
    d.resize(d.len().max(small.len()), 0);
    sub_buf(&mut d, small);
    trim_lz(&mut d);
    (d, flip)
}

fn assert_signed(p: &[u64], neg: bool, want: (Vec<u64>, bool), what: &str) {
    assert_eq!(p, &want.0[..], "{what}: magnitude");
    assert!(want.0.is_empty() || neg == want.1, "{what}: sign");
}

fn check_signed_bbp<const K: u8>(shift: u64, a: u64, b: u64) {
    let ctx = Shift(shift);
    let pos = bin_split::<DynNodeBBP<SBbp<K, 1>>, C>(&ctx, a, b);
    let neg = bin_split::<DynNodeBBP<SBbp<K, 2>>, C>(&ctx, a, b);
    let want = signed_diff(&pos.p, &neg.p);
    let tree = bin_split::<DynNodeBBP<SBbp<K, 0>>, C>(&ctx, a, b);
    let leaf = DynNodeBBP::<SBbp<K, 0>>::leaf(&ctx, a, b, true);
    for (got, how) in [(&tree, "tree"), (&leaf, "leaf")] {
        let what = format!("BBP K={K} shift={shift} [{a}, {b}) {how}");
        assert_signed(&got.p, got.neg, want.clone(), &what);
        assert_eq!(got.q, pos.q, "{what}: Q");
    }
}

fn check_signed_engel<const K: u8>(shift: u64, a: u64, b: u64) {
    let ctx = Shift(shift);
    let pos = bin_split::<DynNodeEngel<SEngel<K, 1>>, C>(&ctx, a, b);
    let neg = bin_split::<DynNodeEngel<SEngel<K, 2>>, C>(&ctx, a, b);
    let want = signed_diff(&pos.p, &neg.p);
    let tree = bin_split::<DynNodeEngel<SEngel<K, 0>>, C>(&ctx, a, b);
    let leaf = DynNodeEngel::<SEngel<K, 0>>::leaf(&ctx, a, b, true);
    for (got, how) in [(&tree, "tree"), (&leaf, "leaf")] {
        let what = format!("Engel K={K} shift={shift} [{a}, {b}) {how}");
        assert_signed(&got.p, got.neg, want.clone(), &what);
        assert_eq!(got.q, pos.q, "{what}: Q");
    }
}

fn check_signed_hyper<const K: u8>(x: &[u64], shift: u64, b: u64) {
    let ctx = DynHyperCtx::new(x, shift);
    for last in [true, false] {
        let pos = bin_split_tree::<DynNodeHyper<SHyp<K, 1>>, C>(&ctx, 0, b, last);
        let neg = bin_split_tree::<DynNodeHyper<SHyp<K, 2>>, C>(&ctx, 0, b, last);
        let want = signed_diff(&pos.p, &neg.p);
        let tree = bin_split_tree::<DynNodeHyper<SHyp<K, 0>>, C>(&ctx, 0, b, last);
        let leaf = DynNodeHyper::<SHyp<K, 0>>::leaf(&ctx, 0, b, last);
        for (got, how) in [(&tree, "tree"), (&leaf, "leaf")] {
            let what = format!("Hyper K={K} x={x:?} [0, {b}) last={last} {how}");
            assert_signed(&got.p, got.neg, want.clone(), &what);
            assert_eq!((&got.q, &got.r), (&pos.q, &pos.r), "{what}: Q, R");
        }
    }
}

#[test]
fn test_signed_bbp_matches_unsigned_parts() {
    for b in [2, 3, 17, 300, 2500] {
        for shift in [0, 3, 70] {
            check_signed_bbp::<0>(shift, 1, b);
            check_signed_bbp::<1>(shift, 1, b);
            check_signed_bbp::<2>(shift, 1, b);
            check_signed_bbp::<3>(shift, 1, b);
        }
    }
}

#[test]
fn test_signed_engel_matches_unsigned_parts() {
    for b in [3, 4, 40, 2500] {
        for shift in [0, 1, 70] {
            check_signed_engel::<0>(shift, 2, b);
            check_signed_engel::<1>(shift, 2, b);
            check_signed_engel::<2>(shift, 2, b);
            check_signed_engel::<3>(shift, 2, b);
        }
    }
}

#[test]
fn test_signed_hyper_matches_unsigned_parts() {
    let xs: [&[u64]; 3] = [&[1], &[12345], &[0x9e37_79b9_7f4a_7c15, 0xff_ffff_ffff]];
    for x in xs {
        for b in [2, 50, 2500] {
            for shift in [0, 3] {
                check_signed_hyper::<0>(x, shift, b);
                check_signed_hyper::<1>(x, shift, b);
                check_signed_hyper::<2>(x, shift, b);
                check_signed_hyper::<3>(x, shift, b);
            }
        }
    }
}

// A series' terms with term n's sign by pattern K.
struct Signed<S, const K: u8>(PhantomData<S>);
impl<S: BBPSeries, const K: u8> BBPSeries for Signed<S, K> {
    const SIGNED: bool = true;
    fn p(n: u64) -> u64 {
        S::p(n)
    }
    fn q(n: u64) -> u64 {
        S::q(n)
    }
    fn neg(n: u64) -> bool {
        negative::<K>(n)
    }
}
impl<S: EngelSeries, const K: u8> EngelSeries for Signed<S, K> {
    const SIGNED: bool = true;
    fn p(n: u64) -> u64 {
        S::p(n)
    }
    fn q(n: u64) -> u64 {
        S::q(n)
    }
    fn neg(n: u64) -> bool {
        negative::<K>(n)
    }
}
impl<S: HyperSeries, const K: u8> HyperSeries for Signed<S, K> {
    const SIGNED: bool = true;
    fn p(n: u64) -> u64 {
        S::p(n)
    }
    fn q(n: u64) -> u64 {
        S::q(n)
    }
    fn r(n: u64) -> u64 {
        S::r(n)
    }
    fn neg(n: u64) -> bool {
        negative::<K>(n)
    }
}

// These signed sums cancel only mildly, so static nodes keep their precision.
#[test]
fn test_static_signed_nodes_match_dyn() {
    type L0 = Signed<BbpLin, 0>;
    type L3 = Signed<BbpLin, 3>;
    let ctx = Shift(4);
    for b in [2, 17, 300, 2500] {
        check_static::<DynNodeBBP<L0>, StaticNodeBBP<16, L0>>(&ctx, &ctx, 1, b);
        check_static::<DynNodeBBP<L3>, StaticNodeBBP<64, L3>>(&ctx, &ctx, 1, b);
    }
    type Q0 = Signed<BbpQuad, 0>;
    check_static::<DynNodeBBP<Q0>, StaticNodeBBP<16, Q0>>(&ctx, &ctx, 1, 1000);
    type E0 = Signed<EngelFact, 0>;
    type E3 = Signed<EngelFact, 3>;
    let ctx = Shift(1);
    for b in [3, 40, 2500] {
        check_static::<DynNodeEngel<E0>, StaticNodeEngel<16, E0>>(&ctx, &ctx, 2, b);
        check_static::<DynNodeEngel<E3>, StaticNodeEngel<64, E3>>(&ctx, &ctx, 2, b);
    }
    type H0 = Signed<Hyp, 0>;
    type H3 = Signed<Hyp, 3>;
    for (x, shift) in [
        (vec![1u64], 8u64),
        (vec![12345], 20),
        (vec![0x9e37_79b9_7f4a_7c15, 0xff_ffff_ffff], 112),
    ] {
        for b in [2, 50, 2500] {
            let d = DynHyperCtx::new(&x, shift);
            let s16 = StaticHyperCtx::<16>::new(&x, shift);
            let s64 = StaticHyperCtx::<64>::new(&x, shift);
            check_static::<DynNodeHyper<H0>, StaticNodeHyper<16, H0>>(&d, &s16, 0, b);
            check_static::<DynNodeHyper<H3>, StaticNodeHyper<64, H3>>(&d, &s64, 0, b);
        }
    }
}

// The ln(2) series with the largest terms, sum 1 / ((4n + 1) 16^n).
struct Ln2Like;
impl BBPSeries for Ln2Like {
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        4 * n + 1
    }
}

// u64::MAX makes [a, b) a single leaf, so all splitting happens inside it.
fn check_static_cutoffs<D, S>(dyn_ctx: &D::Ctx, static_ctx: &S::Ctx, a: u64, b: u64, len: usize)
where
    D: Node + Send,
    S: Node + Send,
    D::Ctx: Sync,
    S::Ctx: Sync,
{
    check_static_at::<D, S, 2>(dyn_ctx, static_ctx, a, b, len, true);
    check_static_at::<D, S, 3>(dyn_ctx, static_ctx, a, b, len, true);
    check_static_at::<D, S, LN2_TERM_CUTOFF>(dyn_ctx, static_ctx, a, b, len, true);
    check_static_at::<D, S, { u64::MAX }>(dyn_ctx, static_ctx, a, b, len, true);
}

fn static_any_cutoff_at<const N: usize>() {
    let len = N - 3;
    type L3 = Signed<BbpLin, 3>;
    let ctx = Shift(4);
    for (a, b) in [(1, 2), (1, 300), (1, 2500), (900, 1400)] {
        check_static_cutoffs::<DynNodeBBP<BbpLin>, StaticNodeBBP<N, BbpLin>>(&ctx, &ctx, a, b, len);
        check_static_cutoffs::<DynNodeBBP<Ln2Like>, StaticNodeBBP<N, Ln2Like>>(
            &ctx, &ctx, a, b, len,
        );
        check_static_cutoffs::<DynNodeBBP<L3>, StaticNodeBBP<N, L3>>(&ctx, &ctx, a, b, len);
    }
    check_static_cutoffs::<DynNodeBBP<BbpQuad>, StaticNodeBBP<N, BbpQuad>>(
        &ctx, &ctx, 1, 1000, len,
    );
    type E3 = Signed<EngelFact, 3>;
    let ctx = Shift(1);
    for (a, b) in [(2, 3), (2, 300), (2, 2500), (900, 1400)] {
        check_static_cutoffs::<DynNodeEngel<EngelFact>, StaticNodeEngel<N, EngelFact>>(
            &ctx, &ctx, a, b, len,
        );
        check_static_cutoffs::<DynNodeEngel<E3>, StaticNodeEngel<N, E3>>(&ctx, &ctx, a, b, len);
    }
    type H3 = Signed<Hyp, 3>;
    for (x, shift) in [
        (vec![1u64], 8u64),
        (vec![12345], 20),
        (vec![0x9e37_79b9_7f4a_7c15, 0xff_ffff_ffff], 112),
    ] {
        let d = DynHyperCtx::new(&x, shift);
        let s = StaticHyperCtx::<N>::new(&x, shift);
        for b in [2, 50, 2500] {
            check_static_cutoffs::<DynNodeHyper<Hyp>, StaticNodeHyper<N, Hyp>>(&d, &s, 0, b, len);
            check_static_cutoffs::<DynNodeHyper<H3>, StaticNodeHyper<N, H3>>(&d, &s, 0, b, len);
        }
    }
}

// Static leaves too big for N limbs split themselves, so every cutoff >= 2 is
// valid at every N.
#[test]
fn test_static_nodes_any_cutoff() {
    static_any_cutoff_at::<4>();
    static_any_cutoff_at::<8>();
    static_any_cutoff_at::<16>();
    static_any_cutoff_at::<64>();
}

// Leaves that overflowed their N limbs before static leaves could split.
#[test]
fn test_static_large_leaf_regressions() {
    let ctx = Shift(4);
    type D = DynNodeBBP<Ln2Like>;
    check_static_at::<D, StaticNodeBBP<8, Ln2Like>, LN2_TERM_CUTOFF>(&ctx, &ctx, 1, 53, 5, true);
    check_static_at::<D, StaticNodeBBP<64, Ln2Like>, LN2_TERM_CUTOFF>(
        &ctx, &ctx, 149, 501, 61, true,
    );
    let ctx = Shift(1);
    check_static_at::<DynNodeEngel<EngelFact>, StaticNodeEngel<8, EngelFact>, 384>(
        &ctx, &ctx, 2, 300, 5, true,
    );
    let (d, s) = (DynHyperCtx::new(&[1], 1), StaticHyperCtx::<8>::new(&[1], 1));
    check_static_at::<DynNodeHyper<Hyp>, StaticNodeHyper<8, Hyp>, 384>(&d, &s, 0, 300, 5, true);
}

// A fit bound that undercounts would overflow exactly at the size limit, so
// build a single leaf of every size from one term to well past it.
fn sweep_leaves<D, S>(
    dyn_ctx: &D::Ctx,
    static_ctx: &S::Ctx,
    a: u64,
    max_t: u64,
    len: usize,
    last: bool,
) where
    D: Node + Send,
    S: Node + Send,
    D::Ctx: Sync,
    S::Ctx: Sync,
{
    for t in 1..=max_t {
        check_static_at::<D, S, { u64::MAX }>(dyn_ctx, static_ctx, a, a + t, len, last);
    }
}

fn static_leaf_sweep_at<const N: usize>() {
    // Every term below costs at least 2 bits of Q, so 32N + 8 terms never fit.
    let (len, max_t) = (N - 3, 32 * N as u64 + 8);
    type L3 = Signed<BbpLin, 3>;
    type Q0 = Signed<BbpQuad, 0>;
    let ctx = Shift(4);
    for a in [1, 1000] {
        sweep_leaves::<DynNodeBBP<Ln2Like>, StaticNodeBBP<N, Ln2Like>>(
            &ctx, &ctx, a, max_t, len, true,
        );
        sweep_leaves::<DynNodeBBP<L3>, StaticNodeBBP<N, L3>>(&ctx, &ctx, a, max_t, len, true);
        sweep_leaves::<DynNodeBBP<Q0>, StaticNodeBBP<N, Q0>>(&ctx, &ctx, a, max_t, len, true);
    }
    type E3 = Signed<EngelFact, 3>;
    let ctx = Shift(1);
    for a in [2, 1000] {
        sweep_leaves::<DynNodeEngel<E3>, StaticNodeEngel<N, E3>>(&ctx, &ctx, a, max_t, len, true);
    }
    type H3 = Signed<Hyp, 3>;
    for (x, shift) in [(vec![1u64], 8u64), (vec![12345], 20)] {
        let d = DynHyperCtx::new(&x, shift);
        let s = StaticHyperCtx::<N>::new(&x, shift);
        // Only Hyper leaves depend on last: they keep R for a later merge.
        for last in [true, false] {
            sweep_leaves::<DynNodeHyper<H3>, StaticNodeHyper<N, H3>>(&d, &s, 0, max_t, len, last);
        }
    }
}

#[test]
fn test_static_leaf_size_sweep() {
    static_leaf_sweep_at::<4>();
    static_leaf_sweep_at::<5>();
    static_leaf_sweep_at::<8>();
    static_leaf_sweep_at::<16>();
}

// ln(3/2) = sum_{n >= 1} (-1)^(n + 1) / (n 2^n).
struct Ln3Over2;
impl BBPSeries for Ln3Over2 {
    const SIGNED: bool = true;
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
    fn neg(n: u64) -> bool {
        n % 2 == 0
    }
}

// -ln(3/2), the same terms with every sign flipped.
struct NegLn3Over2;
impl BBPSeries for NegLn3Over2 {
    const SIGNED: bool = true;
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n
    }
    fn neg(n: u64) -> bool {
        n % 2 == 1
    }
}

// e^(-1/2) / 2 = sum_{n >= 0} (-1)^n / (n! 2^(n + 1)).
struct ExpHalf;
impl HyperSeries for ExpHalf {
    const SIGNED: bool = true;
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n.max(1)
    }
    fn r(_: u64) -> u64 {
        1
    }
    fn neg(n: u64) -> bool {
        n % 2 == 1
    }
}

// Top limbs of floor(S * 2^256), from Python's decimal module.
const LN3_2_TOP: [u64; 4] = [
    0x67cc8fb2fe612fca,
    0xda35d9bd01488606,
    0x7d20ffb34547d7c2,
    0xb38ad78ec59e3b60,
];
const EXP_HALF_TOP: [u64; 4] = [
    0x4da2cbf1be5827f9,
    0xeb3ad1aa9866ebb3,
    0xf1a3dfc56874055d,
    0xe72574ac3080a18c,
];

fn top4(out: &[u64]) -> Vec<u64> {
    out.iter().rev().take(4).copied().collect()
}

#[test]
fn test_signed_series_values() {
    let mut out = vec![0u64; 8];
    let ctx = Shift(1);
    assert!(!bin_split::<DynNodeBBP<Ln3Over2>, C>(&ctx, 1, 600).finalize(&ctx, &mut out));
    assert_eq!(top4(&out), LN3_2_TOP);
    assert!(!bin_split::<StaticNodeBBP<16, Ln3Over2>, C>(&ctx, 1, 600).finalize(&ctx, &mut out));
    assert_eq!(top4(&out), LN3_2_TOP);
    assert!(bin_split::<DynNodeBBP<NegLn3Over2>, C>(&ctx, 1, 600).finalize(&ctx, &mut out));
    assert_eq!(top4(&out), LN3_2_TOP);
    assert!(bin_split::<StaticNodeBBP<16, NegLn3Over2>, C>(&ctx, 1, 600).finalize(&ctx, &mut out));
    assert_eq!(top4(&out), LN3_2_TOP);

    let ctx = DynHyperCtx::new(&[1], 1);
    assert!(!bin_split::<DynNodeHyper<ExpHalf>, C>(&ctx, 0, 200).finalize(&ctx, &mut out));
    assert_eq!(top4(&out), EXP_HALF_TOP);
    // With x = 2^64 and 65 bits of shift per term, S is e^(-1/2) / 2 / 2^64.
    let ctx = DynHyperCtx::new(&[0, 1], 65);
    assert!(!bin_split::<DynNodeHyper<ExpHalf>, C>(&ctx, 0, 200).finalize(&ctx, &mut out));
    assert_eq!(out[7], 0);
    assert_eq!(top4(&out)[1..], EXP_HALF_TOP[..3]);
    let ctx = StaticHyperCtx::<16>::new(&[0, 1], 65);
    assert!(!bin_split::<StaticNodeHyper<16, ExpHalf>, C>(&ctx, 0, 200).finalize(&ctx, &mut out));
    assert_eq!(top4(&out)[1..], EXP_HALF_TOP[..3]);
}
