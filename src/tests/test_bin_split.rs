use super::exact_fraction;
use crate::utils::series::*;
use crate::utils::consts::consts::{
    ln2_dyn, ln2_static, pi_dyn, pi_static, pi_static_sum, LN2_LIMBS, PI_LIMBS,
};
use crate::utils::div::div_dyn;
use crate::utils::mul::mul_dyn;
use crate::utils::utils::{add_buf, add_prim, cmp_buf, sub_buf, trim_lz};
use crate::utils::CapacityError;
use std::marker::PhantomData;

// Small leaves give deep trees, so most work runs through merges, and the
// low parallel cutoff runs most of the splits of a large tree in parallel.
const C: SplitCutoffs = SplitCutoffs {
    leaf_limbs: 2,
    leaf_terms: 1,
    par_limbs: 64,
};

// Serial cutoffs with leaves of at most leaf_terms terms, or below leaf_limbs.
fn leaves(leaf_limbs: u64, leaf_terms: u64) -> SplitCutoffs {
    SplitCutoffs {
        leaf_limbs,
        leaf_terms,
        par_limbs: u64::MAX,
    }
}

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
        let got = bin_split_tree_at::<DynNodeBBP<S>>(&ctx, a, b, true, C);
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
            let got = bin_split_tree_at::<DynNodeEngel<EngelFact>>(&ctx, 2, b, true, C);
            assert!(got.p == want.p && got.q == want.q, "Engel [2, {b}) differs");
        }
    }
}

fn check_hyper(x: &[u64], shift: u64, b: u64) {
    let ctx = DynHyperCtx::new(x, shift);
    for last in [true, false] {
        let want = DynNodeHyper::<Hyp>::leaf(&ctx, 0, b, last);
        for _ in 0..3 {
            let got = bin_split_tree_at::<DynNodeHyper<Hyp>>(&ctx, 0, b, last, C);
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
    // Requests up to LN2_LIMBS come from the table and longer ones from the series
    // built past it, which some of these extend concurrently.
    let mut full = vec![0; LN2_LIMBS + 100];
    ln2_dyn(&mut full);
    assert_eq!(&full[full.len() - 32..], &LN2_32[..]);
    let lens: Vec<usize> = (0..64).map(|i| 1 + (i * 37) % (LN2_LIMBS + 200)).collect();
    lens.par_iter().for_each(|&l| {
        let mut out = vec![0; l];
        ln2_dyn(&mut out);
        let m = l.min(full.len());
        assert_eq!(&out[l - m..], &full[full.len() - m..], "ln2 at {l} limbs");
    });
    let mut big = vec![0; LN2_LIMBS + 200];
    ln2_dyn(&mut big);
    assert_eq!(&big[big.len() - full.len()..], &full[..]);
}

// ln2.bin is the series built one limb past it, less that limb (examples/gen_ln2.rs),
// and both caches serve any request up to its length from it, whatever N is.
#[test]
fn test_ln2_static_table() {
    let mut full = vec![0; LN2_LIMBS + 1];
    ln2_dyn(&mut full);
    let full = &full[1..];
    assert_eq!(&full[LN2_LIMBS - 32..], &LN2_32[..]);
    for l in [0, 1, 31, 32, 33, 500, LN2_LIMBS - 1, LN2_LIMBS] {
        let (mut out, mut dyn_out) = (vec![0; l], vec![0; l]);
        ln2_static::<8>(&mut out).unwrap();
        ln2_dyn(&mut dyn_out);
        assert_eq!(&out[..], &full[LN2_LIMBS - l..], "ln2 at {l} limbs");
        assert_eq!(dyn_out, out, "ln2_dyn at {l} limbs");
    }
}

// Past the table the static cache sums the series itself, and its guard limb
// absorbs the static nodes' truncation, so it matches ln2_dyn exactly.
#[test]
fn test_ln2_static_past_table() {
    fn check<const N: usize>(l: usize) {
        let (mut want, mut out) = (vec![0; l], vec![0; l]);
        ln2_dyn(&mut want);
        ln2_static::<N>(&mut out).unwrap();
        assert_eq!(out, want, "ln2 at {l} limbs, N = {N}");
    }
    check::<{ LN2_LIMBS + 4 }>(LN2_LIMBS + 1);
    check::<{ LN2_LIMBS + 16 }>(LN2_LIMBS + 1);
    check::<{ LN2_LIMBS + 16 }>(LN2_LIMBS + 7);
}

#[test]
fn test_ln2_static_past_table_small_n() {
    let err = ln2_static::<{ LN2_LIMBS + 3 }>(&mut [0; LN2_LIMBS + 1]);
    assert_eq!(err, Err(CapacityError { min_n: LN2_LIMBS + 4 }));
}

// The top 32 limbs of floor(pi * B^1023), from Machin's formula.
const PI_32: [u64; 32] = [
    0xb4cc5c341141e8ce,
    0x55ca396a2aab10b6,
    0x5748986263e81440,
    0xe65525f3aa55ab94,
    0x78af2fda55605c60,
    0xd71577c1bd314b27,
    0x6c9e0e8bb01e8a3e,
    0x8e79dcb0603a180e,
    0xca417918b8db38ef,
    0xc5d1b023286085f0,
    0x9c30d5392af26013,
    0x7b54a41dc25a59b5,
    0x718bcd5882154aee,
    0x0d95748f728eb658,
    0xa458fea3f4933d7e,
    0x636920d871574e69,
    0x0801f2e2858efc16,
    0x24a19947b3916cf7,
    0xba7c9045f12c7f99,
    0xb8e1afed6a267e96,
    0x2ffd72dbd01adfb7,
    0xd1310ba698dfb5ac,
    0x9216d5d98979fb1b,
    0x3f84d5b5b5470917,
    0xc0ac29b7c97c50dd,
    0xbe5466cf34e90c6c,
    0x452821e638d01377,
    0x082efa98ec4e6c89,
    0xa4093822299f31d0,
    0x13198a2e03707344,
    0x243f6a8885a308d3,
    0x0000000000000003,
];

#[test]
fn test_pi_dyn_concurrent_requests() {
    use rayon::prelude::*;
    // floor(pi * B^(m - 1)) is the top m limbs of floor(pi * B^(M - 1)) for m <= M.
    // Requests up to PI_LIMBS come from the table and longer ones from the series
    // built past it, which some of these extend concurrently.
    let mut full = vec![0; PI_LIMBS + 100];
    pi_dyn(&mut full);
    assert_eq!(&full[full.len() - 32..], &PI_32[..]);
    let lens: Vec<usize> = (0..64).map(|i| 1 + (i * 37) % (PI_LIMBS + 200)).collect();
    lens.par_iter().for_each(|&l| {
        let mut out = vec![0; l];
        pi_dyn(&mut out);
        let m = l.min(full.len());
        assert_eq!(&out[l - m..], &full[full.len() - m..], "pi at {l} limbs");
    });
    let mut big = vec![0; PI_LIMBS + 200];
    pi_dyn(&mut big);
    assert_eq!(&big[big.len() - full.len()..], &full[..]);
}

// pi.bin is the series built one limb past it, less that limb (examples/gen_pi.rs), and
// both caches serve any request up to its length from it, whatever N is.
#[test]
fn test_pi_static_table() {
    let mut full = vec![0; PI_LIMBS + 1];
    pi_dyn(&mut full);
    let full = &full[1..];
    assert_eq!(&full[PI_LIMBS - 32..], &PI_32[..]);
    for l in [0, 1, 31, 32, 33, 500, PI_LIMBS - 1, PI_LIMBS] {
        let (mut out, mut dyn_out) = (vec![0; l], vec![0; l]);
        pi_static::<8>(&mut out).unwrap();
        pi_dyn(&mut dyn_out);
        assert_eq!(&out[..], &full[PI_LIMBS - l..], "pi at {l} limbs");
        assert_eq!(dyn_out, out, "pi_dyn at {l} limbs");
    }
}

// The static series at every length up to the N - 3 bound match pi_dyn (here pi.bin)
// exactly: the guard limb absorbs the static nodes' truncation and the finish's rounding.
// These all run serially (below STATIC_PI_PAR_CUTOFF); test_pi_static_past_table runs the
// threads against the dynamic series.
#[test]
fn test_pi_static_sum_matches_dyn() {
    fn check<const N: usize>() {
        for l in 1..=N - 3 {
            let (mut want, mut out) = (vec![0; l], vec![0; l]);
            pi_dyn(&mut want);
            pi_static_sum::<N>(&mut out).unwrap();
            assert_eq!(out, want, "pi at {l} limbs, N = {N}");
        }
    }
    check::<4>();
    check::<5>();
    check::<8>();
    check::<16>();
    check::<33>();
    check::<64>();
    check::<300>();
}

#[test]
fn test_pi_static_past_table() {
    fn check<const N: usize>(l: usize) {
        let (mut want, mut out) = (vec![0; l], vec![0; l]);
        pi_dyn(&mut want);
        pi_static::<N>(&mut out).unwrap();
        assert_eq!(out, want, "pi at {l} limbs, N = {N}");
    }
    check::<{ PI_LIMBS + 4 }>(PI_LIMBS + 1);
    check::<{ PI_LIMBS + 16 }>(PI_LIMBS + 1);
    check::<{ PI_LIMBS + 16 }>(PI_LIMBS + 13);
}

// Too small an N is a CapacityError, returned before out is touched.
#[test]
fn test_pi_static_small_n() {
    let mut out = [7; PI_LIMBS + 1];
    let err = pi_static::<{ PI_LIMBS + 3 }>(&mut out);
    assert_eq!(err, Err(CapacityError { min_n: PI_LIMBS + 4 }));
    assert!(out.iter().all(|&l| l == 7));
    assert_eq!(pi_static_sum::<4>(&mut [0; 2]), Err(CapacityError { min_n: 5 }));
}

// Static finalize up to its out.len() + 2 <= N bound, against the exact quotient
// of the same P and Q (so no node truncation): a reciprocal of len + 2 limbs and
// a high product stay within one of it. The last two cases put the high
// product's overflow limb past len + 1 limbs, where it has no slot.
#[test]
fn test_ratio_to_fraction_static_up_to_bound() {
    fn check<const N: usize>(seed: u64) {
        let mut seed = seed;
        let mut rand = |len: usize, top: u64| {
            seed += 1;
            let mut x = crate::tests::rand_vec(len, seed);
            x[len - 1] = top;
            x
        };
        for len in N - 4..=N - 2 {
            let cases = [
                (rand(N, 1 << 62), rand(N, 1 << 63 | 3), 0),
                (rand(N - 2, u64::MAX), rand(N, 3), -64),
                (rand(N, u64::MAX / 3), rand(N, u64::MAX), 7),
                (rand(N, 1 << 61 | 9), rand(N - 1, 1 << 63 | 1), 63),
                (rand(N, 1), rand(N - 1, u64::MAX), 1),
            ];
            for (i, (p, q, shift)) in cases.into_iter().enumerate() {
                let want = exact_fraction(&p, &q, shift, len);
                let mut got = vec![0; len];
                ratio_to_fraction_static::<N>(&p, &q, shift, &mut got);
                let (lo, hi) = if got.iter().rev().le(want.iter().rev()) {
                    (&got, &want)
                } else {
                    (&want, &got)
                };
                let mut lo = lo.clone();
                add_prim(&mut lo, 1);
                assert!(
                    got == want || &lo == hi,
                    "N = {N}, len = {len}, case {i} is off by more than one"
                );
            }
        }
    }
    check::<16>(30_000);
    check::<64>(30_100);
    check::<419>(30_200);
    check::<2048>(30_300);
}

#[test]
#[should_panic(expected = "out.len() + 2 <= N")]
fn test_ratio_to_fraction_static_past_bound() {
    ratio_to_fraction_static::<16>(&[1], &[2], 0, &mut [0; 15]);
}

// Static nodes keep only their top 64 N bits, so their finalize can differ from
// the exact dynamic one only in rounding: by at most one in the lowest limb.
fn check_static<D, S>(dyn_ctx: &D::Ctx, static_ctx: &S::Ctx, a: u64, b: u64)
where
    D: Node + Send,
    S: Node + Send,
    D::Ctx: Sync,
    S::Ctx: Sync,
{
    check_static_at::<D, S>(dyn_ctx, static_ctx, a, b, 13, true, C);
}

// check_static with the static tree split at cut into len limbs.
fn check_static_at<D, S>(
    dyn_ctx: &D::Ctx,
    static_ctx: &S::Ctx,
    a: u64,
    b: u64,
    len: usize,
    last: bool,
    cut: SplitCutoffs,
) where
    D: Node + Send,
    S: Node + Send,
    D::Ctx: Sync,
    S::Ctx: Sync,
{
    let (mut want, mut got) = (vec![0u64; len], vec![0u64; len]);
    let want_neg = bin_split_tree_at::<D>(dyn_ctx, a, b, last, C).finalize(dyn_ctx, 0, &mut want);
    let got_neg = bin_split_tree_at::<S>(static_ctx, a, b, last, cut).finalize(static_ctx, 0, &mut got);
    let what = format!("static [{a}, {b}) at {cut:?}, {len} limbs");
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
    let pos = bin_split_tree_at::<DynNodeBBP<SBbp<K, 1>>>(&ctx, a, b, true, C);
    let neg = bin_split_tree_at::<DynNodeBBP<SBbp<K, 2>>>(&ctx, a, b, true, C);
    let want = signed_diff(&pos.p, &neg.p);
    let tree = bin_split_tree_at::<DynNodeBBP<SBbp<K, 0>>>(&ctx, a, b, true, C);
    let leaf = DynNodeBBP::<SBbp<K, 0>>::leaf(&ctx, a, b, true);
    for (got, how) in [(&tree, "tree"), (&leaf, "leaf")] {
        let what = format!("BBP K={K} shift={shift} [{a}, {b}) {how}");
        assert_signed(&got.p, got.neg, want.clone(), &what);
        assert_eq!(got.q, pos.q, "{what}: Q");
    }
}

fn check_signed_engel<const K: u8>(shift: u64, a: u64, b: u64) {
    let ctx = Shift(shift);
    let pos = bin_split_tree_at::<DynNodeEngel<SEngel<K, 1>>>(&ctx, a, b, true, C);
    let neg = bin_split_tree_at::<DynNodeEngel<SEngel<K, 2>>>(&ctx, a, b, true, C);
    let want = signed_diff(&pos.p, &neg.p);
    let tree = bin_split_tree_at::<DynNodeEngel<SEngel<K, 0>>>(&ctx, a, b, true, C);
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
        let pos = bin_split_tree_at::<DynNodeHyper<SHyp<K, 1>>>(&ctx, 0, b, last, C);
        let neg = bin_split_tree_at::<DynNodeHyper<SHyp<K, 2>>>(&ctx, 0, b, last, C);
        let want = signed_diff(&pos.p, &neg.p);
        let tree = bin_split_tree_at::<DynNodeHyper<SHyp<K, 0>>>(&ctx, 0, b, last, C);
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

// Single-term leaves, leaves of up to two terms, the default cutoffs, and
// u64::MAX limbs, which makes [a, b) a single leaf so all splitting happens
// inside it.
fn check_static_cutoffs<D, S>(dyn_ctx: &D::Ctx, static_ctx: &S::Ctx, a: u64, b: u64, len: usize)
where
    D: Node + Send,
    S: Node + Send,
    D::Ctx: Sync,
    S::Ctx: Sync,
{
    for cut in [
        leaves(0, 1),
        leaves(0, 2),
        SplitCutoffs::of::<S>(),
        leaves(u64::MAX, 1),
    ] {
        check_static_at::<D, S>(dyn_ctx, static_ctx, a, b, len, true, cut);
    }
}

fn static_any_cutoff_at<const N: usize>() {
    let len = N - 2;
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

// Static leaves too big for N limbs split themselves, so every cutoff is valid
// at every N, up to finalize's N - 2 bound.
#[test]
fn test_static_nodes_any_cutoff() {
    static_any_cutoff_at::<4>();
    static_any_cutoff_at::<8>();
    static_any_cutoff_at::<16>();
    static_any_cutoff_at::<64>();
}

// Leaves that overflowed their N limbs before static leaves could split, each
// built as one leaf as the old term cutoffs did.
#[test]
fn test_static_large_leaf_regressions() {
    let one = leaves(u64::MAX, 1);
    let ctx = Shift(4);
    type D = DynNodeBBP<Ln2Like>;
    check_static_at::<D, StaticNodeBBP<8, Ln2Like>>(&ctx, &ctx, 1, 53, 5, true, one);
    check_static_at::<D, StaticNodeBBP<64, Ln2Like>>(&ctx, &ctx, 149, 501, 61, true, one);
    let ctx = Shift(1);
    check_static_at::<DynNodeEngel<EngelFact>, StaticNodeEngel<8, EngelFact>>(
        &ctx, &ctx, 2, 300, 5, true, one,
    );
    let (d, s) = (DynHyperCtx::new(&[1], 1), StaticHyperCtx::<8>::new(&[1], 1));
    check_static_at::<DynNodeHyper<Hyp>, StaticNodeHyper<8, Hyp>>(&d, &s, 0, 300, 5, true, one);
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
        let one = leaves(u64::MAX, 1);
        check_static_at::<D, S>(dyn_ctx, static_ctx, a, a + t, len, last, one);
    }
}

fn static_leaf_sweep_at<const N: usize>() {
    // Every term below costs at least 2 bits of Q, so 32N + 8 terms never fit.
    let (len, max_t) = (N - 2, 32 * N as u64 + 8);
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

// A single BBP term whose shift is past N limbs used to overflow its leaf, which folded
// 2^shift into Q. Each term here shrinks by more than N limbs, so the sum's top limbs are
// zero, and finalize skips them.
#[test]
fn test_static_bbp_shift_past_n() {
    fn check<const N: usize>(shift: u64, b: u64) {
        let ctx = Shift(shift);
        let zeros = (shift / 64) as usize;
        for cut in [leaves(u64::MAX, 1), C] {
            let (mut want, mut got) = (vec![0; N - 2], vec![0; N - 2]);
            bin_split_tree_at::<DynNodeBBP<Ln2Like>>(&ctx, 1, b, true, C).finalize(&ctx, zeros, &mut want);
            let s = bin_split_tree_at::<StaticNodeBBP<N, Ln2Like>>(&ctx, 1, b, true, cut);
            s.finalize(&ctx, zeros, &mut got);
            assert!(want.iter().any(|&l| l != 0), "shift {shift}: zeros skip the whole sum");
            let (lo, hi) = if got.iter().rev().le(want.iter().rev()) {
                (&got, &want)
            } else {
                (&want, &got)
            };
            let mut lo = lo.clone();
            add_prim(&mut lo, 1);
            assert!(got == want || &lo == hi, "N = {N}, shift {shift}, {cut:?}");
        }
    }
    check::<4>(64 * 4 + 10, 2);
    check::<4>(64 * 4 + 10, 6);
    check::<8>(64 * 9 + 33, 40);
    check::<16>(64 * 20, 300);
}

// Static parts keep their top 64 N bits once they drop any, so a node's P / Q stays
// within a few bits of rounding of 2^-64N relative to the exact ratio. Both sides divide
// exactly at N + 4 limbs, so only the node's own error shows.
trait StaticRatio: Node {
    fn ratio_of(&self, ctx: &Self::Ctx) -> (&[u64], &[u64], i64);
}
impl<const N: usize, S: WideBBPSeries> StaticRatio for StaticNodeBBP<N, S> {
    fn ratio_of(&self, ctx: &Shift) -> (&[u64], &[u64], i64) {
        self.ratio(ctx)
    }
}
impl<const N: usize, S: WideEngelSeries> StaticRatio for StaticNodeEngel<N, S> {
    fn ratio_of(&self, ctx: &Shift) -> (&[u64], &[u64], i64) {
        self.ratio(ctx)
    }
}
impl<const N: usize, S: WideHyperSeries> StaticRatio for StaticNodeHyper<N, S> {
    fn ratio_of(&self, ctx: &StaticHyperCtx<N>) -> (&[u64], &[u64], i64) {
        self.ratio(ctx)
    }
}

// Bits a static node loses below 64 N, the most observed being 10.
const STATIC_BITS_SLACK: i64 = 16;

fn check_bits<D, S>(dyn_ctx: &D::Ctx, static_ctx: &S::Ctx, a: u64, b: u64, n: usize)
where
    D: Node + Send,
    S: StaticRatio + Send,
    D::Ctx: Sync,
    S::Ctx: Sync,
{
    let bits = |x: &[u64]| {
        let x = &x[..x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1)];
        x.last().map_or(0, |&t| 64 * x.len() as i64 - t.leading_zeros() as i64)
    };
    let l = n + 4;
    let mut want = vec![0; l];
    bin_split_tree_at::<D>(dyn_ctx, a, b, true, C).finalize(dyn_ctx, 0, &mut want);
    for cut in [C, leaves(0, 2), SplitCutoffs::of::<S>(), leaves(u64::MAX, 1)] {
        let node = bin_split_tree_at::<S>(static_ctx, a, b, true, cut);
        let (p, q, shift) = node.ratio_of(static_ctx);
        let mut got = vec![0; l];
        ratio_to_fraction_dyn(p, q, shift, &mut got);
        let (mut err, small) = if cmp_buf(&got, &want) == std::cmp::Ordering::Less {
            (want.clone(), &got)
        } else {
            (got.clone(), &want)
        };
        sub_buf(&mut err, small);
        if err.iter().any(|&l| l != 0) {
            let lost = 64 * n as i64 - (bits(&want) - bits(&err));
            assert!(
                lost <= STATIC_BITS_SLACK,
                "static [{a}, {b}) at {cut:?}, N = {n}: lost {lost} bits"
            );
        }
    }
}

fn static_bits_at<const N: usize>() {
    let ctx = Shift(4);
    type L3 = Signed<BbpLin, 3>;
    for (a, b) in [(1, 300), (1, 2500)] {
        check_bits::<DynNodeBBP<BbpLin>, StaticNodeBBP<N, BbpLin>>(&ctx, &ctx, a, b, N);
        check_bits::<DynNodeBBP<L3>, StaticNodeBBP<N, L3>>(&ctx, &ctx, a, b, N);
    }
    check_bits::<DynNodeBBP<Ln2Like>, StaticNodeBBP<N, Ln2Like>>(&ctx, &ctx, 900, 1400, N);
    check_bits::<DynNodeBBP<BbpQuad>, StaticNodeBBP<N, BbpQuad>>(&ctx, &ctx, 1, 1000, N);
    let ctx = Shift(1);
    check_bits::<DynNodeEngel<EngelFact>, StaticNodeEngel<N, EngelFact>>(&ctx, &ctx, 2, 2500, N);
    check_bits::<DynNodeEngel<EngelFact>, StaticNodeEngel<N, EngelFact>>(&ctx, &ctx, 900, 1400, N);
    type H3 = Signed<Hyp, 3>;
    for (x, shift) in [
        (vec![1u64], 8u64),
        (vec![12345], 20),
        (vec![0x9e37_79b9_7f4a_7c15, 0xff_ffff_ffff], 112),
    ] {
        let d = DynHyperCtx::new(&x, shift);
        let s = StaticHyperCtx::<N>::new(&x, shift);
        check_bits::<DynNodeHyper<Hyp>, StaticNodeHyper<N, Hyp>>(&d, &s, 0, 2500, N);
        check_bits::<DynNodeHyper<H3>, StaticNodeHyper<N, H3>>(&d, &s, 0, 2500, N);
    }
}

#[test]
fn test_static_nodes_keep_bits() {
    static_bits_at::<4>();
    static_bits_at::<8>();
    static_bits_at::<16>();
    static_bits_at::<64>();
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

// The sum over [a, b) through a deep tree into out, returning its sign.
fn split_sum<N>(ctx: &N::Ctx, a: u64, b: u64, out: &mut [u64]) -> bool
where
    N: Node + Send,
    N::Ctx: Sync,
{
    bin_split_tree_at::<N>(ctx, a, b, true, C).finalize(ctx, 0, out)
}

#[test]
fn test_signed_series_values() {
    let mut out = vec![0u64; 8];
    let ctx = Shift(1);
    assert!(!split_sum::<DynNodeBBP<Ln3Over2>>(&ctx, 1, 600, &mut out));
    assert_eq!(top4(&out), LN3_2_TOP);
    assert!(!split_sum::<StaticNodeBBP<16, Ln3Over2>>(&ctx, 1, 600, &mut out));
    assert_eq!(top4(&out), LN3_2_TOP);
    assert!(split_sum::<DynNodeBBP<NegLn3Over2>>(&ctx, 1, 600, &mut out));
    assert_eq!(top4(&out), LN3_2_TOP);
    assert!(split_sum::<StaticNodeBBP<16, NegLn3Over2>>(&ctx, 1, 600, &mut out));
    assert_eq!(top4(&out), LN3_2_TOP);

    let ctx = DynHyperCtx::new(&[1], 1);
    assert!(!split_sum::<DynNodeHyper<ExpHalf>>(&ctx, 0, 200, &mut out));
    assert_eq!(top4(&out), EXP_HALF_TOP);
    // With x = 2^64 and 65 bits of shift per term, each term gains x / 2^65 = 1/2,
    // just as with x = 1 and one bit, so S is e^(-1/2) / 2 again.
    let ctx = DynHyperCtx::new(&[0, 1], 65);
    assert!(!split_sum::<DynNodeHyper<ExpHalf>>(&ctx, 0, 200, &mut out));
    assert_eq!(top4(&out), EXP_HALF_TOP);
    let ctx = StaticHyperCtx::<16>::new(&[0, 1], 65);
    assert!(!split_sum::<StaticNodeHyper<16, ExpHalf>>(&ctx, 0, 200, &mut out));
    assert_eq!(top4(&out), EXP_HALF_TOP);
}

// Node::limbs only shapes the tree, but an estimate far from the real size
// would silently move every cutoff.
fn check_estimate(what: &str, est: u64, parts: &[usize]) {
    let real = *parts.iter().max().unwrap() as u64;
    assert!(
        est <= 2 * real && real <= 2 * est,
        "{what}: estimated {est} limbs, built {real}"
    );
}

#[test]
fn test_leaf_limb_estimates() {
    for (shift, a, b) in [(4, 1, 2000), (4, 5000, 6000), (70, 5, 500)] {
        let ctx = Shift(shift);
        let n = DynNodeBBP::<Ln2Like>::leaf(&ctx, a, b, true);
        let est = DynNodeBBP::<Ln2Like>::limbs(&ctx, a, b);
        check_estimate(&format!("BBP shift {shift} [{a}, {b})"), est, &[n.p.len(), n.q.len()]);
    }
    let ctx = Shift(4);
    let n = DynNodeBBP::<BbpQuad>::leaf(&ctx, 1, 1000, true);
    let est = DynNodeBBP::<BbpQuad>::limbs(&ctx, 1, 1000);
    check_estimate("BBP quadratic q", est, &[n.p.len(), n.q.len()]);

    for (shift, a, b) in [(0, 2, 2000), (1, 1000, 3000), (70, 2, 500)] {
        let ctx = Shift(shift);
        let n = DynNodeEngel::<EngelFact>::leaf(&ctx, a, b, true);
        let est = DynNodeEngel::<EngelFact>::limbs(&ctx, a, b);
        check_estimate(&format!("Engel shift {shift} [{a}, {b})"), est, &[n.p.len(), n.q.len()]);
    }

    let x8: Vec<u64> = (1..=8u64).map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15)).collect();
    let xs: [(&[u64], u64, u64); 4] = [
        (&[1], 8, 2000),
        (&[12345], 20, 2000),
        (&[0x9e37_79b9_7f4a_7c15, 0xff_ffff_ffff], 112, 500),
        (&x8, 520, 100),
    ];
    for (x, shift, b) in xs {
        let ctx = DynHyperCtx::new(x, shift);
        let n = DynNodeHyper::<Hyp>::leaf(&ctx, 0, b, false);
        let est = DynNodeHyper::<Hyp>::limbs(&ctx, 0, b);
        let r = n.r.as_ref().map_or(0, |r| r.len());
        let what = format!("Hyper {}-limb x shift {shift} [0, {b})", x.len());
        check_estimate(&what, est, &[n.p.len(), n.q.len(), r]);
    }

    let ctx = DynHyperCtx::new(&[1], 15);
    let n = DynNodeHyper::<Chud>::leaf(&ctx, 1, 400, false);
    let est = DynNodeHyper::<Chud>::limbs(&ctx, 1, 400);
    let r = n.r.as_ref().map_or(0, |r| r.len());
    check_estimate("wide Chudnovsky [1, 400)", est, &[n.p.len(), n.q.len(), r]);
}

fn prod(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut v = vec![0; a.len() + b.len()];
    mul_dyn(a, b, &mut v);
    trim_lz(&mut v);
    v
}

fn shl(x: &[u64], k: u64) -> Vec<u64> {
    let (w, v) = ((k / 64) as usize, (k % 64) as u32);
    let mut out = vec![0; w + x.len() + 1];
    for (i, &l) in x.iter().enumerate() {
        out[w + i] |= l << v;
        if v != 0 {
            out[w + i + 1] |= l >> (64 - v);
        }
    }
    trim_lz(&mut out);
    out
}

// A dyn node's P, Q, term count and sign.
trait DynParts: Node {
    fn parts(&self) -> (&[u64], &[u64], u64, bool);
}
impl<S: WideBBPSeries> DynParts for DynNodeBBP<S> {
    fn parts(&self) -> (&[u64], &[u64], u64, bool) {
        (&self.p, &self.q, self.b - self.a, self.neg)
    }
}
impl<S: WideEngelSeries> DynParts for DynNodeEngel<S> {
    fn parts(&self) -> (&[u64], &[u64], u64, bool) {
        (&self.p, &self.q, self.b - self.a, self.neg)
    }
}
impl<S: WideHyperSeries> DynParts for DynNodeHyper<S> {
    fn parts(&self) -> (&[u64], &[u64], u64, bool) {
        (&self.p, &self.q, self.b - self.a, self.neg)
    }
}

// Whether two dyn nodes' sums S = P / (Q 2^(shift t)) are equal, exactly, with no
// division: P1 Q2 2^(s2 t2) = P2 Q1 2^(s1 t1), and their signs agree.
fn same_sum<A: DynParts, B: DynParts>(x: &A, x_shift: u64, y: &B, y_shift: u64) -> bool {
    let ((p1, q1, t1, n1), (p2, q2, t2, n2)) = (x.parts(), y.parts());
    let (a, b) = (x_shift * t1, y_shift * t2);
    let m = a.min(b);
    let equal = shl(&prod(p1, q2), b - m) == shl(&prod(p2, q1), a - m);
    equal && (p1.is_empty() || n1 == n2)
}

// Wide twins sum the same S as a narrow series, with terms from M on scaled by a 70-bit
// K in p and q (and r), which leaves each term as it was. Leaves below M run narrow,
// leaves past it wide, and leaves across it wide.
const K: u128 = (1 << 69) + 0x9e37;
fn k_at<const M: u64>(n: u128) -> u128 {
    if n < M as u128 {
        1
    } else {
        K
    }
}

struct Twin<S, const M: u64>(PhantomData<S>);
impl<S: BBPSeries, const M: u64> WideBBPSeries for Twin<S, M> {
    const SIGNED: bool = S::SIGNED;
    fn p(n: u128) -> u128 {
        S::p(n as u64) as u128 * k_at::<M>(n)
    }
    fn q(n: u128) -> u128 {
        S::q(n as u64) as u128 * k_at::<M>(n)
    }
    fn neg(n: u128) -> bool {
        S::neg(n as u64)
    }
}
impl<S: HyperSeries, const M: u64> WideHyperSeries for Twin<S, M> {
    const SIGNED: bool = S::SIGNED;
    fn p(n: u128) -> u128 {
        S::p(n as u64) as u128 * k_at::<M>(n)
    }
    fn q(n: u128) -> u128 {
        S::q(n as u64) as u128 * k_at::<M>(n)
    }
    fn r(n: u128) -> u128 {
        S::r(n as u64) as u128 * k_at::<M>(n)
    }
    fn neg(n: u128) -> bool {
        S::neg(n as u64)
    }
}

// An Engel series' q(n) 2^58 at shift k sums what q(n) does at shift k + 58. With
// q(n) = n it goes wide from n = 64.
struct Shifted58<S>(PhantomData<S>);
impl<S: EngelSeries> WideEngelSeries for Shifted58<S> {
    const SIGNED: bool = S::SIGNED;
    fn p(n: u128) -> u128 {
        S::p(n as u64) as u128
    }
    fn q(n: u128) -> u128 {
        (S::q(n as u64) as u128) << 58
    }
    fn neg(n: u128) -> bool {
        S::neg(n as u64)
    }
}

// A wide twin's sum equals its narrow series' exactly at every cutoff, whichever of its
// leaves run narrow or wide.
fn check_twin<N, W>(
    n_ctx: &N::Ctx,
    n_shift: u64,
    w_ctx: &W::Ctx,
    w_shift: u64,
    a: u64,
    b: u64,
    last: bool,
) where
    N: DynParts + Send,
    W: DynParts + Send,
    N::Ctx: Sync,
    W::Ctx: Sync,
{
    let narrow = bin_split_tree_at::<N>(n_ctx, a, b, last, C);
    for cut in [
        C,
        leaves(0, 2),
        SplitCutoffs::of::<W>(),
        leaves(u64::MAX, 1),
    ] {
        let wide = bin_split_tree_at::<W>(w_ctx, a, b, last, cut);
        assert!(
            same_sum(&wide, w_shift, &narrow, n_shift),
            "wide twin {} [{a}, {b}) last={last} at {cut:?}",
            std::any::type_name::<W>()
        );
    }
}

#[test]
fn test_wide_twins_match_narrow() {
    type L3 = Signed<BbpLin, 3>;
    let ctx = Shift(4);
    for b in [2, 300, 2500] {
        check_twin::<DynNodeBBP<BbpLin>, DynNodeBBP<Twin<BbpLin, 0>>>(&ctx, 4, &ctx, 4, 1, b, true);
        check_twin::<DynNodeBBP<BbpLin>, DynNodeBBP<Twin<BbpLin, 150>>>(
            &ctx, 4, &ctx, 4, 1, b, true,
        );
        check_twin::<DynNodeBBP<L3>, DynNodeBBP<Twin<L3, 150>>>(&ctx, 4, &ctx, 4, 1, b, true);
    }
    check_twin::<DynNodeBBP<BbpQuad>, DynNodeBBP<Twin<BbpQuad, 400>>>(
        &ctx, 4, &ctx, 4, 1, 1000, true,
    );

    // From a = 64 every term is wide.
    type E3 = Signed<EngelFact, 3>;
    for (k, a, b) in [(1, 2, 300), (1, 2, 2500), (1, 64, 600), (70, 2, 500)] {
        let (nc, wc) = (Shift(k + 58), Shift(k));
        type W<S> = DynNodeEngel<Shifted58<S>>;
        check_twin::<DynNodeEngel<EngelFact>, W<EngelFact>>(&nc, k + 58, &wc, k, a, b, true);
        check_twin::<DynNodeEngel<E3>, W<E3>>(&nc, k + 58, &wc, k, a, b, true);
    }

    type H3 = Signed<Hyp, 3>;
    for (x, shift) in [
        (vec![1u64], 8u64),
        (vec![12345], 20),
        (vec![0x9e37_79b9_7f4a_7c15, 0xff_ffff_ffff], 112),
    ] {
        let d = DynHyperCtx::new(&x, shift);
        for last in [true, false] {
            for b in [50, 2500] {
                type T<S, const M: u64> = DynNodeHyper<Twin<S, M>>;
                check_twin::<DynNodeHyper<Hyp>, T<Hyp, 0>>(&d, shift, &d, shift, 0, b, last);
                check_twin::<DynNodeHyper<Hyp>, T<Hyp, 100>>(&d, shift, &d, shift, 0, b, last);
                check_twin::<DynNodeHyper<H3>, T<H3, 100>>(&d, shift, &d, shift, 0, b, last);
            }
        }
    }
}

// Static wide twins match their narrow series' dyn sum within 1 ulp up to finalize's
// N - 2 bound, and keep all but a few bits of 64N.
fn static_wide_at<const N: usize>() {
    let len = N - 2;
    type L3 = Signed<BbpLin, 3>;
    let ctx = Shift(4);
    check_static_cutoffs::<DynNodeBBP<BbpLin>, StaticNodeBBP<N, Twin<BbpLin, 150>>>(
        &ctx, &ctx, 1, 2500, len,
    );
    check_static_cutoffs::<DynNodeBBP<L3>, StaticNodeBBP<N, Twin<L3, 150>>>(
        &ctx, &ctx, 1, 2500, len,
    );
    check_bits::<DynNodeBBP<BbpLin>, StaticNodeBBP<N, Twin<BbpLin, 150>>>(&ctx, &ctx, 1, 2500, N);

    type E3 = Signed<EngelFact, 3>;
    let (nc, wc) = (Shift(59), Shift(1));
    type W<const N: usize, S> = StaticNodeEngel<N, Shifted58<S>>;
    for a in [2, 64] {
        check_static_cutoffs::<DynNodeEngel<EngelFact>, W<N, EngelFact>>(&nc, &wc, a, 2500, len);
        check_static_cutoffs::<DynNodeEngel<E3>, W<N, E3>>(&nc, &wc, a, 2500, len);
        check_bits::<DynNodeEngel<EngelFact>, W<N, EngelFact>>(&nc, &wc, a, 2500, N);
    }

    type H3 = Signed<Hyp, 3>;
    for (x, shift) in [
        (vec![1u64], 8u64),
        (vec![12345], 20),
        (vec![0x9e37_79b9_7f4a_7c15, 0xff_ffff_ffff], 112),
    ] {
        let d = DynHyperCtx::new(&x, shift);
        let s = StaticHyperCtx::<N>::new(&x, shift);
        type T<const N: usize, S> = StaticNodeHyper<N, Twin<S, 100>>;
        check_static_cutoffs::<DynNodeHyper<Hyp>, T<N, Hyp>>(&d, &s, 0, 2500, len);
        check_static_cutoffs::<DynNodeHyper<H3>, T<N, H3>>(&d, &s, 0, 2500, len);
        check_bits::<DynNodeHyper<Hyp>, T<N, Hyp>>(&d, &s, 0, 2500, N);
    }
}

#[test]
fn test_static_wide_twins() {
    static_wide_at::<4>();
    static_wide_at::<8>();
    static_wide_at::<16>();
    static_wide_at::<64>();
}

// Chudnovsky's series for 1/π from n = 1, with 640320³ / 24 = 2^15 10005³ / 3: term n is
// (-1)^n (13591409 + 545140134 n) prod_{j <= n} (6j - 5)(2j - 1)(6j - 1) / (j³ 640320³ / 24).
// Its p passes 2^64 at about n = 146 and its q at n = 381, so leaves go from narrow to
// wide.
struct Chud;
impl WideHyperSeries for Chud {
    const SIGNED: bool = true;
    fn p(n: u128) -> u128 {
        Self::r(n) * (13591409 + 545140134 * n)
    }
    fn q(n: u128) -> u128 {
        n.pow(3) * 333_833_583_375
    }
    fn r(n: u128) -> u128 {
        (6 * n - 5) * (2 * n - 1) * (6 * n - 1)
    }
    fn neg(n: u128) -> bool {
        n % 2 == 1
    }
}

// Chud's sum over [1, b) as (P, Q, sign) with S = P / (Q 2^(15 (b - 1))), term by term with
// plain bignum products: P_n = P_(n-1) q(n) 2^15 ± a(n) R_n, R_n = R_(n-1) r(n) and
// Q_n = Q_(n-1) q(n). None of it runs through a leaf kernel.
fn chud_naive(b: u64) -> (Vec<u64>, Vec<u64>, bool) {
    let mul = |x: &[u64], m: u128| {
        let mut v = vec![0; x.len() + 2];
        mul_dyn(x, &[m as u64, (m >> 64) as u64], &mut v);
        trim_lz(&mut v);
        v
    };
    let add = |x: &[u64], y: &[u64]| {
        let (long, short) = if x.len() >= y.len() { (x, y) } else { (y, x) };
        let mut v = long.to_vec();
        v.push(0);
        add_buf(&mut v, short);
        trim_lz(&mut v);
        v
    };
    let (mut p, mut neg, mut q, mut r) = (vec![], false, vec![1u64], vec![1u64]);
    for n in 1..b as u128 {
        p = mul(&p, Chud::q(n) << 15);
        q = mul(&q, Chud::q(n));
        r = mul(&r, Chud::r(n));
        let t = mul(&r, 13591409 + 545140134 * n);
        (p, neg) = if p.is_empty() || neg == Chud::neg(n) {
            (add(&p, &t), Chud::neg(n))
        } else if neg {
            signed_diff(&t, &p)
        } else {
            signed_diff(&p, &t)
        };
    }
    (p, q, neg)
}

#[test]
fn test_chudnovsky_wide() {
    let d = DynHyperCtx::new(&[1], 15);
    let (p, q, neg) = chud_naive(500);
    for last in [true, false] {
        let leaf = DynNodeHyper::<Chud>::leaf(&d, 1, 500, last);
        let (lp, lq, _, ln) = leaf.parts();
        assert!(
            prod(lp, &q) == prod(&p, lq) && ln == neg,
            "Chudnovsky [1, 500) single leaf against the naive sum"
        );
        for cut in [C, leaves(0, 2), SplitCutoffs::of::<DynNodeHyper<Chud>>()] {
            let tree = bin_split_tree_at::<DynNodeHyper<Chud>>(&d, 1, 500, last, cut);
            assert!(
                same_sum(&tree, 15, &leaf, 15),
                "Chudnovsky [1, 500) last={last} at {cut:?}"
            );
        }
    }
    let s16 = StaticHyperCtx::<16>::new(&[1], 15);
    check_static_cutoffs::<DynNodeHyper<Chud>, StaticNodeHyper<16, Chud>>(&d, &s16, 1, 500, 14);
    check_bits::<DynNodeHyper<Chud>, StaticNodeHyper<16, Chud>>(&d, &s16, 1, 500, 16);
    let s64 = StaticHyperCtx::<64>::new(&[1], 15);
    check_static_cutoffs::<DynNodeHyper<Chud>, StaticNodeHyper<64, Chud>>(&d, &s64, 1, 500, 62);
    check_bits::<DynNodeHyper<Chud>, StaticNodeHyper<64, Chud>>(&d, &s64, 1, 500, 64);
    let s400 = StaticHyperCtx::<400>::new(&[1], 15);
    check_static_cutoffs::<DynNodeHyper<Chud>, StaticNodeHyper<400, Chud>>(&d, &s400, 1, 500, 398);
}
