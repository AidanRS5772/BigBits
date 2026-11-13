use super::exact_fraction;
use crate::utils::consts::consts::{Ln2S, Pi};
use crate::utils::mul::mul_work;
use crate::utils::series::*;
use crate::utils::trans::{ATanh, Exp};
use crate::utils::utils::{add_buf, buf_len, cmp_buf, sub_buf};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;

const ALGS: [SeriesAlg; 3] = [SeriesAlg::Horner, SeriesAlg::Rect, SeriesAlg::BinSplit];

struct BbpLin;
impl BBPSeries for BbpLin {
    no_tail!(1);
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        4 * n + 3
    }
}

struct BbpQuad;
impl BBPSeries for BbpQuad {
    no_tail!(1);
    fn p(n: u64) -> u64 {
        3 * n + 2
    }
    fn q(n: u64) -> u64 {
        n * (4 * n + 3)
    }
}

struct EngelFact;
impl EngelSeries for EngelFact {
    no_tail!(2);
    fn p(n: u64) -> u64 {
        n % 3
    }
    fn q(n: u64) -> u64 {
        n
    }
}

struct Hyp;
impl HyperSeries for Hyp {
    no_tail!(0);
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

fn mag(n: u64) -> u64 {
    n.wrapping_mul(7919) % 1000 + 1
}

fn negative(n: u64) -> bool {
    n.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 63 == 1
}

// Signed series whose partial sums change sign, with |S| < 1 from k >= 3.
struct SgBbp;
impl BBPSeries for SgBbp {
    no_tail!(1);
    const SIGNED: bool = true;
    fn p(n: u64) -> u64 {
        mag(n)
    }
    fn q(n: u64) -> u64 {
        mag(n) + 1 + n % 2
    }
    fn neg(n: u64) -> bool {
        negative(n)
    }
}

struct SgEngel;
impl EngelSeries for SgEngel {
    no_tail!(1);
    const SIGNED: bool = true;
    fn p(n: u64) -> u64 {
        mag(n) % 7
    }
    fn q(n: u64) -> u64 {
        n + 1
    }
    fn neg(n: u64) -> bool {
        n % 2 == 1
    }
}

// r(n) <= q(n), as Horner needs.
struct SgHyp;
impl HyperSeries for SgHyp {
    no_tail!(0);
    const SIGNED: bool = true;
    fn p(n: u64) -> u64 {
        mag(n) % 5
    }
    fn q(n: u64) -> u64 {
        n + 3
    }
    fn r(n: u64) -> u64 {
        n + 2
    }
    fn neg(n: u64) -> bool {
        negative(n)
    }
}

// Wide denominators are supported by Horner; r(n) > q(n) still is not.
struct WideQ;
impl WideBBPSeries for WideQ {
    no_tail!(1);
    fn p(_: u128) -> u128 {
        1
    }
    fn q(n: u128) -> u128 {
        (1 << 70) + n
    }
}

struct RBig;
impl HyperSeries for RBig {
    no_tail!(0);
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        n + 1
    }
    fn r(n: u64) -> u64 {
        n + 2
    }
}

// The exact sum's out from a dynamic node's exact P and Q, and its sign.
type Want = (Vec<u64>, bool);

fn finalize_shift(k: u64, a: u64, b: u64, zeros: usize) -> i64 {
    (k * (b - a)) as i64 - 64 * zeros as i64
}

fn want_bbp<S: WideBBPSeries>(k: u64, a: u64, b: u64, zeros: usize, len: usize) -> Want {
    let n = bin_split::<DynNodeBBP<S>>(&Shift(k), a, b);
    let neg = S::SIGNED && n.neg && !n.p.is_empty();
    (
        exact_fraction(&n.p, &n.q, finalize_shift(k, a, b, zeros), len),
        neg,
    )
}

fn want_engel<S: WideEngelSeries>(k: u64, a: u64, b: u64, zeros: usize, len: usize) -> Want {
    let n = bin_split::<DynNodeEngel<S>>(&Shift(k), a, b);
    let neg = S::SIGNED && n.neg && !n.p.is_empty();
    (
        exact_fraction(&n.p, &n.q, finalize_shift(k, a, b, zeros), len),
        neg,
    )
}

fn want_hyper<S: WideHyperSeries>(
    ctx: &DynHyperCtx,
    a: u64,
    b: u64,
    zeros: usize,
    len: usize,
) -> Want {
    let n = bin_split::<DynNodeHyper<S>>(ctx, a, b);
    let neg = S::SIGNED && n.neg && !n.p.is_empty();
    let shift = finalize_shift(ctx.shift, a, b, zeros);
    (exact_fraction(&n.p, &n.q, shift, len), neg)
}

// Whether x <= 1.
fn tiny(x: &[u64]) -> bool {
    buf_len(x) <= 1 && x.first().map_or(true, |&l| l <= 1)
}

// got is within one unit of want's last limb, with want's sign unless want is that small.
fn assert_close(got: &[u64], got_neg: bool, (want, want_neg): &Want, what: &str) {
    let (mut hi, lo) = match cmp_buf(got, want) {
        Ordering::Less => (want.clone(), got),
        _ => (got.to_vec(), &want[..]),
    };
    sub_buf(&mut hi, lo);
    assert!(
        tiny(&hi),
        "{what}: off by more than one\n got {got:x?}\nwant {want:x?}"
    );
    assert!(tiny(want) || got_neg == *want_neg, "{what}: sign");
}

fn elig(horner: bool, rect: bool) -> Eligible {
    Eligible { horner, rect }
}

// Every eligible algorithm, forced, and series_eval's own pick against want. Returns the
// eligibility.
fn check_algs<N>(
    ctx: &N::Ctx,
    (a, b): (u64, u64),
    zeros: usize,
    want: &Want,
    what: &str,
) -> Eligible
where
    N: SeriesNode,
    N::Ctx: Sync,
{
    let len = want.0.len();
    let ok = N::eligible(ctx, a, b, len, zeros);
    for alg in ALGS.into_iter().filter(|&alg| ok.allows(alg)) {
        let mut out = vec![0; len];
        let neg = series_eval_at::<N>(ctx, a, b, zeros, &mut out, alg);
        assert_close(&out, neg, want, &format!("{what} {alg:?}"));
    }
    let mut out = vec![0; len];
    let neg = series_eval_range::<N>(ctx, a, b, zeros, &mut out);
    assert_close(&out, neg, want, &format!("{what} series_eval"));
    ok
}

// x of exactly bits bits.
fn rand_x(bits: u64, rng: &mut StdRng) -> Vec<u64> {
    let len = bits.div_ceil(64) as usize;
    let mut x: Vec<u64> = (0..len).map(|_| rng.gen()).collect();
    let top = bits - 64 * (len as u64 - 1);
    x[len - 1] = (x[len - 1] >> (64 - top)) | 1 << (top - 1);
    x
}

// S < 2^(1 - k) for these BBP and Engel shapes, so (k - 1) / 64 limbs are zero.
fn shift_shapes() -> impl Iterator<Item = (u64, usize, u64, usize)> {
    [1, 3, 8, 63, 64, 65, 130, 200].into_iter().flat_map(|k| {
        [1, 2, 3, 5, 8].into_iter().flat_map(move |len| {
            [1, 2, 7, 40, 300].into_iter().flat_map(move |t| {
                let z = ((k - 1) / 64) as usize;
                [(k, len, t, 0), (k, len, t, z)]
            })
        })
    })
}

#[test]
fn test_horner_bbp_matches_exact() {
    for (k, len, t, zeros) in shift_shapes() {
        let what = format!("k {k} len {len} t {t} zeros {zeros}");
        let want = want_bbp::<BbpLin>(k, 1, 1 + t, zeros, len);
        let ok = check_algs::<DynNodeBBP<BbpLin>>(&Shift(k), (1, 1 + t), zeros, &want, &what);
        assert_eq!(ok, elig(true, false), "{what}");
        let want = want_bbp::<BbpQuad>(k, 1, 1 + t, zeros, len);
        check_algs::<DynNodeBBP<BbpQuad>>(&Shift(k), (1, 1 + t), zeros, &want, &what);
        let want = want_bbp::<ATanh>(k, 1, 1 + t, zeros, len);
        check_algs::<DynNodeBBP<ATanh>>(&Shift(k), (1, 1 + t), zeros, &want, &what);
    }
}

#[test]
fn test_horner_engel_matches_exact() {
    for (k, len, t, zeros) in shift_shapes() {
        let what = format!("k {k} len {len} t {t} zeros {zeros}");
        let want = want_engel::<Exp>(k, 1, 1 + t, zeros, len);
        let ok = check_algs::<DynNodeEngel<Exp>>(&Shift(k), (1, 1 + t), zeros, &want, &what);
        assert_eq!(ok, elig(true, false), "{what}");
        let want = want_engel::<EngelFact>(k, 2, 2 + t, zeros, len);
        check_algs::<DynNodeEngel<EngelFact>>(&Shift(k), (2, 2 + t), zeros, &want, &what);
    }
}

#[test]
fn test_horner_hyper_x1_matches_exact() {
    for (k, len, t, zeros) in shift_shapes() {
        let what = format!("k {k} len {len} t {t} zeros {zeros}");
        let ctx = DynHyperCtx::new_shift(k);
        let want = want_hyper::<Hyp>(&ctx, 0, t, zeros, len);
        let ok = check_algs::<DynNodeHyper<Hyp>>(&ctx, (0, t), zeros, &want, &what);
        assert_eq!(ok, elig(true, false), "{what}");
        let want = want_hyper::<ATanh>(&ctx, 1, 1 + t, zeros, len);
        check_algs::<DynNodeHyper<ATanh>>(&ctx, (1, 1 + t), zeros, &want, &what);
    }
}

// x of 1 to 4 limbs, with y = x 2^-k below 2^-extra, so S < 2^(1 - extra).
fn x_shapes(rng: &mut StdRng) -> Vec<(Vec<u64>, u64, usize, u64, usize)> {
    let mut shapes = Vec::new();
    for bits in [1, 2, 40, 64, 65, 128, 190, 256] {
        for extra in [1, 5, 64, 130] {
            for len in [1, 2, 5] {
                for t in [1, 2, 7, 30] {
                    let x = rand_x(bits, rng);
                    let z = ((extra - 1) / 64) as usize;
                    shapes.push((x.clone(), bits + extra, len, t, 0));
                    shapes.push((x, bits + extra, len, t, z));
                }
            }
        }
    }
    shapes
}

#[test]
fn test_horner_and_rect_hyper_x_match_exact() {
    let mut rng = StdRng::seed_from_u64(11);
    for (x, k, len, t, zeros) in x_shapes(&mut rng) {
        let what = format!("x {} limbs k {k} len {len} t {t} zeros {zeros}", x.len());
        let ctx = DynHyperCtx::new(&x, k);
        let want = want_hyper::<Exp>(&ctx, 1, 1 + t, zeros, len);
        let ok = check_algs::<DynNodeHyper<Exp>>(&ctx, (1, 1 + t), zeros, &want, &what);
        assert_eq!(ok, elig(true, x.len() >= 2), "{what}");
        let want = want_hyper::<ATanh>(&ctx, 1, 1 + t, zeros, len);
        check_algs::<DynNodeHyper<ATanh>>(&ctx, (1, 1 + t), zeros, &want, &what);
    }
}

// Every block length from 1 to t, including blocks that don't divide t.
#[test]
fn test_rect_every_block_length() {
    let mut rng = StdRng::seed_from_u64(12);
    for (bits, extra, len, t) in [
        (128, 1, 3, 20),
        (190, 5, 2, 13),
        (256, 64, 5, 9),
        (65, 3, 1, 25),
    ] {
        let x = rand_x(bits, &mut rng);
        let ctx = DynHyperCtx::new(&x, bits + extra);
        let want = want_hyper::<Exp>(&ctx, 1, 1 + t, 0, len);
        let swant = want_hyper::<SgHyp>(&ctx, 0, t, 0, len);
        for m in 1..=t as usize {
            let what = format!("x {bits} bits extra {extra} len {len} t {t} m {m}");
            let mut out = vec![0; len];
            let neg = rect_dyn_m::<Exp>(&ctx, 1, 1 + t, 0, &mut out, m);
            assert_close(&out, neg, &want, &what);
            let neg = rect_dyn_m::<SgHyp>(&ctx, 0, t, 0, &mut out, m);
            assert_close(&out, neg, &swant, &format!("{what} signed"));
        }
    }
}

#[test]
fn test_horner_signed_matches_exact() {
    let mut rng = StdRng::seed_from_u64(13);
    for k in [3, 8, 64, 100] {
        for len in [1, 2, 4] {
            for t in [1, 2, 9, 50] {
                let what = format!("k {k} len {len} t {t}");
                let want = want_bbp::<SgBbp>(k, 1, 1 + t, 0, len);
                check_algs::<DynNodeBBP<SgBbp>>(&Shift(k), (1, 1 + t), 0, &want, &what);
                let want = want_engel::<SgEngel>(k, 1, 1 + t, 0, len);
                check_algs::<DynNodeEngel<SgEngel>>(&Shift(k), (1, 1 + t), 0, &want, &what);
                let ctx = DynHyperCtx::new_shift(k);
                let want = want_hyper::<SgHyp>(&ctx, 0, t, 0, len);
                check_algs::<DynNodeHyper<SgHyp>>(&ctx, (0, t), 0, &want, &what);
                for bits in [30, 128] {
                    let x = rand_x(bits, &mut rng);
                    let ctx = DynHyperCtx::new(&x, bits + k);
                    let want = want_hyper::<SgHyp>(&ctx, 0, t, 0, len);
                    let what = format!("{what} x {bits} bits");
                    check_algs::<DynNodeHyper<SgHyp>>(&ctx, (0, t), 0, &want, &what);
                }
            }
        }
    }
}

// The typed entry, by its own pick and every eligible algorithm forced, against want, the
// exact sum of the range its tail bound gives.
fn check_typed<C: Series>(ctx: &C, want: &Want, what: &str) {
    let (len, zeros) = (want.0.len(), ctx.zeros());
    let (a, b) = ctx.range(64 * (len + zeros) as u64);
    let ok = C::Dyn::eligible(&ctx.dyn_ctx(), a, b, len, zeros);
    let mut out = vec![0; len];
    let (neg, z) = series_eval_dyn(ctx, &mut out);
    assert_eq!(z, zeros, "{what}");
    assert_close(&out, neg, want, &format!("{what} series_eval_dyn"));
    for alg in ALGS.into_iter().filter(|&alg| ok.allows(alg)) {
        let (neg, _) = series_eval_dyn_at(ctx, &mut out, alg);
        assert_close(&out, neg, want, &format!("{what} {alg:?}"));
    }
}

// exp's and ln's own series shapes in a w-limb frame, through the contexts trans.rs builds.
#[test]
fn test_series_eval_exp_ln_shapes() {
    let mut rng = StdRng::seed_from_u64(14);
    for w in [1, 2, 4, 8, 16, 40] {
        let bits = 64 * w as u64;
        // exp's top bit, an Engel series at its exact shift.
        for k in [1, 2, 25, 64, 65, 100] {
            let ctx = Engel::<Exp>::new(k);
            let (zeros, (a, b)) = (ctx.zeros(), ctx.range(bits));
            if zeros >= w {
                continue;
            }
            let what = format!("exp top w {w} k {k}");
            let want = want_engel::<Exp>(k, a, b, zeros, w - zeros);
            check_algs::<DynNodeEngel<Exp>>(&Shift(k), (a, b), zeros, &want, &what);
            check_typed(&ctx, &want, &what);
        }
        // exp's chunks: width bits at offset t, a Hyper series in x 2^-(t + width).
        for (t, width) in [(1, 16), (17, 32), (49, 64), (113, 128), (241, 256)] {
            let x = odd_x(width, &mut rng);
            let ctx = Hyper::<Exp>::new(&x, t + width).unwrap();
            let (zeros, (a, b)) = (ctx.zeros(), ctx.range(bits));
            if zeros >= w {
                continue;
            }
            let dctx = DynHyperCtx::new(&x, t + width);
            let what = format!("exp chunk w {w} t {t} width {width}");
            let want = want_hyper::<Exp>(&dctx, a, b, zeros, w - zeros);
            check_algs::<DynNodeHyper<Exp>>(&dctx, (a, b), zeros, &want, &what);
            check_typed(&ctx, &want, &what);
        }
        // ln's chunks: BBP ATanh at shift 2k when the chunk is a single bit, otherwise Hyper
        // in A^2 for an odd A of a_bits bits.
        for lead in [1, 3, 20, 40, 70] {
            let ctx = Bbp::<ATanh>::new(2 * lead);
            let (zeros, (a, b)) = (ctx.zeros(), ctx.range(bits));
            if zeros < w {
                let what = format!("ln top w {w} lead {lead}");
                let want = want_bbp::<ATanh>(2 * lead, a, b, zeros, w - zeros);
                check_algs::<DynNodeBBP<ATanh>>(&Shift(2 * lead), (a, b), zeros, &want, &what);
                check_typed(&ctx, &want, &what);
            }
            for a_bits in [2, 40, 130] {
                let k = 2 * (lead + a_bits);
                let a2 = super::mul_ref(&odd_x(a_bits, &mut rng), &odd_x(a_bits, &mut rng));
                let ctx = Hyper::<ATanh>::new(&a2, k).unwrap();
                let (zeros, (a, b)) = (ctx.zeros(), ctx.range(bits));
                if zeros >= w {
                    continue;
                }
                let dctx = DynHyperCtx::new(&a2, k);
                let what = format!("ln chunk w {w} lead {lead} A {a_bits} bits");
                let want = want_hyper::<ATanh>(&dctx, a, b, zeros, w - zeros);
                check_algs::<DynNodeHyper<ATanh>>(&dctx, (a, b), zeros, &want, &what);
                check_typed(&ctx, &want, &what);
            }
        }
    }
}

#[test]
fn test_series_eligibility_edges() {
    // Wide denominators now allow Horner; r(n) > q(n) still requires bin_split.
    let want = {
        let n = bin_split::<DynNodeBBP<WideQ>>(&Shift(3), 1, 20);
        (
            exact_fraction(&n.p, &n.q, finalize_shift(3, 1, 20, 0), 3),
            false,
        )
    };
    let ok = check_algs::<DynNodeBBP<WideQ>>(&Shift(3), (1, 20), 0, &want, "wide q");
    assert_eq!(ok, elig(true, false));
    let ctx = DynHyperCtx::new_shift(4);
    let want = want_hyper::<RBig>(&ctx, 0, 20, 0, 3);
    assert_eq!(
        check_algs::<DynNodeHyper<RBig>>(&ctx, (0, 20), 0, &want, "r > q"),
        elig(false, false)
    );

    // Only Hyper kinds have a rect kernel.
    assert!(!DynNodeBBP::<BbpLin>::RECT && !StaticNodeEngel::<8, Exp>::RECT);
    assert!(DynNodeHyper::<Exp>::RECT && StaticNodeHyper::<8, Exp>::RECT);

    // k below bits(x) leaves no shift for x'.
    let x = [u64::MAX, 7];
    let ok = DynNodeHyper::<Exp>::eligible(&DynHyperCtx::new(&x, 66), 1, 10, 2, 0);
    assert_eq!(ok, elig(false, false));
    let ok = DynNodeHyper::<Exp>::eligible(&DynHyperCtx::new(&x, 67), 1, 10, 2, 0);
    assert_eq!(ok, elig(true, true));

    // A static accumulator needs out.len() + 3 limbs.
    assert_eq!(
        StaticNodeEngel::<8, Exp>::eligible(&Shift(5), 1, 10, 5, 0),
        elig(true, false)
    );
    assert_eq!(
        StaticNodeEngel::<8, Exp>::eligible(&Shift(5), 1, 10, 6, 0),
        elig(false, false)
    );
    let ctx = StaticHyperCtx::<8>::new(&x, 70);
    assert_eq!(
        StaticNodeHyper::<8, Exp>::eligible(&ctx, 1, 10, 6, 0),
        elig(false, false)
    );
}

#[test]
#[should_panic(expected = "can't sum this range")]
fn test_series_eval_at_rejects_ineligible() {
    let mut out = [0; 2];
    series_eval_at::<DynNodeHyper<RBig>>(
        &DynHyperCtx::new_shift(3),
        1,
        20,
        0,
        &mut out,
        SeriesAlg::Horner,
    );
}

fn static_suite<const N: usize>() {
    let mut rng = StdRng::seed_from_u64(15 + N as u64);
    let lens = [1, N / 2, N - 3]
        .into_iter()
        .filter(|&l| l >= 1 && l + 3 <= N);
    for len in lens {
        for (k, t) in [(1, 30), (8, 200), (65, 5), (130, 40)] {
            let what = format!("N {N} len {len} k {k} t {t}");
            let zeros = ((k - 1) / 64) as usize;
            let want = want_bbp::<BbpQuad>(k, 1, 1 + t, zeros, len);
            let ok =
                check_algs::<StaticNodeBBP<N, BbpQuad>>(&Shift(k), (1, 1 + t), zeros, &want, &what);
            assert_eq!(ok, elig(true, false), "{what}");
            let want = want_engel::<Exp>(k, 1, 1 + t, zeros, len);
            check_algs::<StaticNodeEngel<N, Exp>>(&Shift(k), (1, 1 + t), zeros, &want, &what);
            let want = want_engel::<SgEngel>(k.max(3), 1, 1 + t, 0, len);
            check_algs::<StaticNodeEngel<N, SgEngel>>(
                &Shift(k.max(3)),
                (1, 1 + t),
                0,
                &want,
                &what,
            );
            let (dctx, sctx) = (DynHyperCtx::new_shift(k), StaticHyperCtx::<N>::new(&[1], k));
            let want = want_hyper::<Hyp>(&dctx, 0, t, zeros, len);
            check_algs::<StaticNodeHyper<N, Hyp>>(&sctx, (0, t), zeros, &want, &what);
        }
        for bits in [40, 64, 128, 190]
            .into_iter()
            .filter(|&b| b <= 64 * N as u64)
        {
            for t in [1, 9, 40] {
                let x = rand_x(bits, &mut rng);
                let k = bits + 3;
                let (dctx, sctx) = (DynHyperCtx::new(&x, k), StaticHyperCtx::<N>::new(&x, k));
                let what = format!("N {N} len {len} x {bits} bits t {t}");
                let want = want_hyper::<Exp>(&dctx, 1, 1 + t, 0, len);
                let ok = check_algs::<StaticNodeHyper<N, Exp>>(&sctx, (1, 1 + t), 0, &want, &what);
                assert_eq!(ok, elig(true, x.len() >= 2), "{what}");
                let want = want_hyper::<SgHyp>(&dctx, 0, t, 0, len);
                check_algs::<StaticNodeHyper<N, SgHyp>>(&sctx, (0, t), 0, &want, &what);
            }
        }
    }
}

#[test]
fn test_static_series_match_exact() {
    static_suite::<4>();
    static_suite::<8>();
    static_suite::<16>();
    static_suite::<64>();
}

// Includes u64/u128 boundaries, mixed widths, zero terms, and two-limb carries.
struct WideTerms<const SIGNED: bool>;
fn wide_coeffs(n: u128) -> (u128, u128, u128) {
    let m = u64::MAX as u128;
    match n % 8 {
        0 => (0, u128::MAX, 1),
        1 => (m, m, m),
        2 => (m + 1, m + 1, m + 1),
        3 => (m + 2, m + 20, m + 8),
        4 => (u128::MAX - 1, u128::MAX, u128::MAX),
        5 => ((1 << 127) + 3, u128::MAX, (1 << 127) + 5),
        6 => (1, (1 << 70) + 1, 1),
        _ => ((1 << 90) + 9, (1 << 90) + 17, 1),
    }
}

macro_rules! wide_terms {
    ($trait:ident $(, $r:ident)?) => {
        impl<const SIGNED: bool> $trait for WideTerms<SIGNED> {
            const SIGNED: bool = SIGNED;
            no_tail!(0);
            fn p(n: u128) -> u128 { wide_coeffs(n).0 }
            fn q(n: u128) -> u128 { wide_coeffs(n).1 }
            $(fn $r(n: u128) -> u128 { wide_coeffs(n).2 })?
            fn neg(n: u128) -> bool { SIGNED && negative(n as u64) }
        }
    };
}
wide_terms!(WideBBPSeries);
wide_terms!(WideEngelSeries);
wide_terms!(WideHyperSeries, r);

fn wide_dynamic_suite<const SIGNED: bool>() {
    let mut rng = StdRng::seed_from_u64(91);
    for k in [3, 63, 64, 65, 130, 200] {
        for len in [1, 2, 5] {
            for (a, t) in [(1, 1), (2, 1), (4, 1), (0, 9), (3, 40)] {
                for zeros in [0, ((k - 1) / 64) as usize] {
                    let b = a + t;
                    let what =
                        format!("wide signed {SIGNED} k {k} len {len} [{a},{b}) zeros {zeros}");
                    let want = want_bbp::<WideTerms<SIGNED>>(k, a, b, zeros, len);
                    assert_eq!(
                        check_algs::<DynNodeBBP<WideTerms<SIGNED>>>(
                            &Shift(k),
                            (a, b),
                            zeros,
                            &want,
                            &what
                        ),
                        elig(true, false)
                    );
                    let want = want_engel::<WideTerms<SIGNED>>(k, a, b, zeros, len);
                    assert_eq!(
                        check_algs::<DynNodeEngel<WideTerms<SIGNED>>>(
                            &Shift(k),
                            (a, b),
                            zeros,
                            &want,
                            &what
                        ),
                        elig(true, false)
                    );
                    for x in [vec![1], rand_x(64, &mut rng), rand_x(128, &mut rng)] {
                        let bits = 64 * (x.len() as u64 - 1)
                            + (64 - x.last().unwrap().leading_zeros()) as u64;
                        let shift = if x == [1] { k } else { k + bits };
                        let ctx = DynHyperCtx::new(&x, shift);
                        let want = want_hyper::<WideTerms<SIGNED>>(&ctx, a, b, zeros, len);
                        let narrow = a == 1 && t == 1;
                        assert_eq!(
                            check_algs::<DynNodeHyper<WideTerms<SIGNED>>>(
                                &ctx,
                                (a, b),
                                zeros,
                                &want,
                                &what
                            ),
                            elig(true, narrow && x.len() >= 2)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_wide_horner_matches_exact() {
    wide_dynamic_suite::<false>();
    wide_dynamic_suite::<true>();
}

fn wide_static_suite<const N: usize>() {
    let mut rng = StdRng::seed_from_u64(92 + N as u64);
    for len in [1, N - 4, N - 3] {
        let ok = elig(len + 4 <= N, false);
        for (k, a, b) in [(3, 0, 9), (65, 2, 19), (200, 4, 44)] {
            let zeros = ((k - 1) / 64) as usize;
            let what = format!("wide static N {N} len {len} k {k}");
            let want = want_bbp::<WideTerms<true>>(k, a, b, zeros, len);
            assert_eq!(
                check_algs::<StaticNodeBBP<N, WideTerms<true>>>(
                    &Shift(k),
                    (a, b),
                    zeros,
                    &want,
                    &what
                ),
                ok
            );
            let want = want_engel::<WideTerms<true>>(k, a, b, zeros, len);
            assert_eq!(
                check_algs::<StaticNodeEngel<N, WideTerms<true>>>(
                    &Shift(k),
                    (a, b),
                    zeros,
                    &want,
                    &what
                ),
                ok
            );
            for x in [vec![1], rand_x(128, &mut rng)] {
                let shift = k + if x.len() == 1 { 0 } else { 128 };
                let ctx = StaticHyperCtx::<N>::new(&x, shift);
                let want =
                    want_hyper::<WideTerms<true>>(&DynHyperCtx::new(&x, shift), a, b, zeros, len);
                assert_eq!(
                    check_algs::<StaticNodeHyper<N, WideTerms<true>>>(
                        &ctx,
                        (a, b),
                        zeros,
                        &want,
                        &what
                    ),
                    ok
                );
            }
        }
    }
}

#[test]
fn test_wide_horner_static_capacity() {
    wide_static_suite::<5>();
    wide_static_suite::<8>();
    wide_static_suite::<16>();
    wide_static_suite::<64>();

    // A range of a wide trait whose values fit u64 keeps narrow capacity and rect.
    let ctx = StaticHyperCtx::<8>::new(&[7, 3], 70);
    assert_eq!(
        StaticNodeHyper::<8, WideTerms<false>>::eligible(&ctx, 1, 2, 5, 0),
        elig(true, true)
    );
    assert_eq!(
        StaticNodeHyper::<8, WideTerms<false>>::eligible(&ctx, 1, 3, 5, 0),
        elig(false, false)
    );
    assert_eq!(
        StaticNodeHyper::<8, WideTerms<false>>::eligible(&ctx, 1, 3, 4, 0),
        elig(true, false)
    );
}

// A full u128 numerator with a narrow denominator exercises div_prim2's fallback and
// negative frame offsets that still leave significant bits of the numerator.
struct WideNumerator;
impl WideBBPSeries for WideNumerator {
    no_tail!(7);
    fn p(_: u128) -> u128 {
        u128::MAX
    }
    fn q(_: u128) -> u128 {
        1
    }
}
impl WideEngelSeries for WideNumerator {
    no_tail!(7);
    fn p(_: u128) -> u128 {
        u128::MAX
    }
    fn q(_: u128) -> u128 {
        1
    }
}
impl WideHyperSeries for WideNumerator {
    no_tail!(7);
    fn p(_: u128) -> u128 {
        u128::MAX
    }
    fn q(_: u128) -> u128 {
        1
    }
    fn r(_: u128) -> u128 {
        1
    }
}

#[test]
fn test_wide_horner_numerator_offsets() {
    for k in [129, 130, 191, 192, 193, 255, 256, 257] {
        for len in [1, 2, 4] {
            let zeros = ((k - 129) / 64) as usize;
            let what = format!("wide numerator k {k} len {len} zeros {zeros}");
            let want = want_bbp::<WideNumerator>(k, 7, 12, zeros, len);
            check_algs::<DynNodeBBP<WideNumerator>>(&Shift(k), (7, 12), zeros, &want, &what);
            check_algs::<StaticNodeBBP<8, WideNumerator>>(&Shift(k), (7, 12), zeros, &want, &what);
            let want = want_engel::<WideNumerator>(k, 7, 12, zeros, len);
            check_algs::<DynNodeEngel<WideNumerator>>(&Shift(k), (7, 12), zeros, &want, &what);
            check_algs::<StaticNodeEngel<8, WideNumerator>>(
                &Shift(k),
                (7, 12),
                zeros,
                &want,
                &what,
            );
            let ctx = DynHyperCtx::new(&[7, 3], k + 66);
            let want = want_hyper::<WideNumerator>(&ctx, 7, 12, zeros, len);
            check_algs::<DynNodeHyper<WideNumerator>>(&ctx, (7, 12), zeros, &want, &what);
            check_algs::<StaticNodeHyper<8, WideNumerator>>(
                &StaticHyperCtx::new(&[7, 3], k + 66),
                (7, 12),
                zeros,
                &want,
                &what,
            );
        }
    }
}

struct ZeroR;
impl WideHyperSeries for ZeroR {
    no_tail!(0);
    fn p(n: u128) -> u128 {
        u128::MAX - n
    }
    fn q(_: u128) -> u128 {
        u128::MAX
    }
    fn r(_: u128) -> u128 {
        0
    }
}

#[test]
fn test_wide_horner_zero_multiplier() {
    for x in [vec![1], vec![7, 3]] {
        let k = if x.len() == 1 { 3 } else { 69 };
        let dctx = DynHyperCtx::new(&x, k);
        let sctx = StaticHyperCtx::<8>::new(&x, k);
        let want = want_hyper::<ZeroR>(&dctx, 0, 20, 0, 4);
        let mut out = [u64::MAX; 4];
        let neg =
            series_eval_at::<DynNodeHyper<ZeroR>>(&dctx, 0, 20, 0, &mut out, SeriesAlg::Horner);
        assert_close(&out, neg, &want, "wide zero r dyn");
        // Force Horner independently of static binary splitting's zero-product merges.
        let neg = series_eval_at::<StaticNodeHyper<8, ZeroR>>(
            &sctx,
            0,
            20,
            0,
            &mut out,
            SeriesAlg::Horner,
        );
        assert_close(&out, neg, &want, "wide zero r static");
    }
}

struct Cancel;
impl WideBBPSeries for Cancel {
    no_tail!(0);
    const SIGNED: bool = true;
    fn p(_: u128) -> u128 {
        u128::MAX
    }
    fn q(_: u128) -> u128 {
        u128::MAX
    }
    fn neg(n: u128) -> bool {
        n % 2 == 0
    }
}
impl WideEngelSeries for Cancel {
    no_tail!(0);
    const SIGNED: bool = true;
    fn p(n: u128) -> u128 {
        if n % 2 == 0 {
            1
        } else {
            u128::MAX
        }
    }
    fn q(_: u128) -> u128 {
        u128::MAX
    }
    fn neg(n: u128) -> bool {
        n % 2 == 0
    }
}
impl WideHyperSeries for Cancel {
    no_tail!(0);
    const SIGNED: bool = true;
    fn p(_: u128) -> u128 {
        u128::MAX
    }
    fn q(_: u128) -> u128 {
        u128::MAX
    }
    fn r(_: u128) -> u128 {
        u128::MAX
    }
    fn neg(n: u128) -> bool {
        n % 2 == 0
    }
}

#[test]
fn test_wide_horner_exact_cancellation() {
    // Each pair cancels exactly even though its individual terms are large.
    // Use separate outputs so every kernel also has to overwrite a nonzero buffer.
    macro_rules! check_zero {
        ($node:ty, $ctx:expr) => {{
            let mut got = [u64::MAX; 4];
            let neg = series_eval_at::<$node>(&$ctx, 0, 2, 0, &mut got, SeriesAlg::Horner);
            assert_eq!(got, [0; 4]);
            assert!(!neg);
        }};
    }
    check_zero!(DynNodeBBP<Cancel>, Shift(0));
    check_zero!(StaticNodeBBP<8, Cancel>, Shift(0));
    check_zero!(DynNodeEngel<Cancel>, Shift(0));
    check_zero!(StaticNodeEngel<8, Cancel>, Shift(0));
    check_zero!(DynNodeHyper<Cancel>, DynHyperCtx::new_shift(0));
    check_zero!(StaticNodeHyper<8, Cancel>, StaticHyperCtx::new(&[1], 0));
}

// Larger product backends and the block sizes used by cutoff tuning must preserve the
// same error bound as the small, exhaustive tests above.
#[test]
fn test_series_large_precision_and_blocks() {
    let mut rng = StdRng::seed_from_u64(1831);
    for len in [63, 129, 257, 513, 1024] {
        let t = 4 * len as u64 + 3;
        let want = want_bbp::<BbpQuad>(8, 3, 3 + t, 0, len);
        check_algs::<DynNodeBBP<BbpQuad>>(&Shift(8), (3, 3 + t), 0, &want, "large BBP");
        let want = want_engel::<Exp>(1, 1, 1 + t / 4, 0, len);
        check_algs::<DynNodeEngel<Exp>>(&Shift(1), (1, 1 + t / 4), 0, &want, "large Engel");
        let x = rand_x(128, &mut rng);
        let ctx = DynHyperCtx::new(&x, 131);
        let want = want_hyper::<Exp>(&ctx, 1, 258, 0, len);
        check_algs::<DynNodeHyper<Exp>>(&ctx, (1, 258), 0, &want, "large Hyper");
    }
    let x = rand_x(128, &mut rng);
    let dctx = DynHyperCtx::new(&x, 258);
    let sctx = StaticHyperCtx::<132>::new(&x, 258);
    let want = want_hyper::<SgHyp>(&dctx, 3, 102, 2, 129);
    for m in 1..=16 {
        let mut out = vec![0; 129];
        let sign = rect_static_m::<132, SgHyp>(&sctx, 3, 102, 2, &mut out, m);
        assert_close(&out, sign, &want, &format!("static m={m}"));
        let sign = rect_dyn_m::<SgHyp>(&dctx, 3, 102, 2, &mut out, m);
        assert_close(&out, sign, &want, &format!("dynamic m={m}"));
    }
}

#[test]
fn test_series_large_static_capacity() {
    let want = want_bbp::<WideTerms<true>>(8, 3, 131, 0, 128);
    let ctx = Shift(8);
    check_algs::<StaticNodeBBP<132, WideTerms<true>>>(&ctx, (3, 131), 0, &want, "wide fits");
    let ok = check_algs::<StaticNodeBBP<131, WideTerms<true>>>(
        &ctx,
        (3, 131),
        0,
        &want,
        "wide capacity fallback",
    );
    assert!(!ok.horner);
    let want = want_bbp::<BbpQuad>(8, 3, 513, 0, 129);
    check_algs::<StaticNodeBBP<132, BbpQuad>>(&ctx, (3, 513), 0, &want, "tight narrow");
    check_algs::<StaticNodeBBP<512, BbpQuad>>(&ctx, (3, 513), 0, &want, "oversized narrow");
}

// Horner divides once per group of terms whose ∏q(n) fits 64 or 128 bits. q(n) = 2^SH + 2n + 1
// puts group boundaries at every size: dozens of terms (SH = 1, up to GROUP_MAX), several,
// two (SH = 31, 42, 63), one (SH = 64, 100). BIG_P's 2^60 p(n) ends groups by the weight
// bound p Q < 2^128 instead, and r(n) <= q(n) differs from q(n) by n mod 3.
struct Grp<const SH: u32, const BIG_P: bool, const SIGNED: bool>;
fn grp_coeffs<const SH: u32, const BIG_P: bool>(n: u128) -> (u128, u128, u128) {
    let q = (1 << SH) + 2 * n + 1;
    let p = if BIG_P { (1 << 60) + n } else { n % 5 };
    (p, q, q - n % 3)
}
macro_rules! grp_terms {
    ($trait:ident $(, $r:ident)?) => {
        impl<const SH: u32, const BIG_P: bool, const SIGNED: bool> $trait
            for Grp<SH, BIG_P, SIGNED>
        {
            const SIGNED: bool = SIGNED;
            no_tail!(0);
            fn p(n: u128) -> u128 { grp_coeffs::<SH, BIG_P>(n).0 }
            fn q(n: u128) -> u128 { grp_coeffs::<SH, BIG_P>(n).1 }
            $(fn $r(n: u128) -> u128 { grp_coeffs::<SH, BIG_P>(n).2 })?
            fn neg(n: u128) -> bool { SIGNED && negative(n as u64) }
        }
    };
}
grp_terms!(WideBBPSeries);
grp_terms!(WideEngelSeries);
grp_terms!(WideHyperSeries, r);

// q(n) = n^2 + 1 grows from 2 bits to 17 over the range, so group sizes shrink as n falls.
struct Growing;
impl HyperSeries for Growing {
    no_tail!(1);
    fn p(n: u64) -> u64 {
        n % 7 + 1
    }
    fn q(n: u64) -> u64 {
        n * n + 1
    }
    fn r(n: u64) -> u64 {
        n * n + 1 - n % 2
    }
}
impl BBPSeries for Growing {
    no_tail!(1);
    fn p(n: u64) -> u64 {
        n % 7 + 1
    }
    fn q(n: u64) -> u64 {
        n * n + 1
    }
}
impl EngelSeries for Growing {
    no_tail!(1);
    fn p(n: u64) -> u64 {
        n % 7 + 1
    }
    fn q(n: u64) -> u64 {
        n * n + 1
    }
}

// Dynamic and static (N from len + 3, the narrow bound and its 64-bit group budget, to roomy
// N with the 128-bit budget) BBP, Engel and Hyper, with x = 1, one limb and two limbs.
fn group_suite<T, const N: usize>(rng: &mut StdRng, (k, zeros): (u64, usize), what: &str)
where
    T: WideBBPSeries + WideEngelSeries + WideHyperSeries,
{
    for len in [1, 2, 5].into_iter().filter(|&l| l + 3 <= N) {
        for (a, t) in [(1, 1), (0, 7), (3, 40), (1, 300)] {
            let b = a + t;
            let what = format!("{what} N {N} k {k} len {len} [{a},{b}) zeros {zeros}");
            let want = want_bbp::<T>(k, a, b, zeros, len);
            check_algs::<DynNodeBBP<T>>(&Shift(k), (a, b), zeros, &want, &what);
            check_algs::<StaticNodeBBP<N, T>>(&Shift(k), (a, b), zeros, &want, &what);
            let want = want_engel::<T>(k, a, b, zeros, len);
            check_algs::<DynNodeEngel<T>>(&Shift(k), (a, b), zeros, &want, &what);
            check_algs::<StaticNodeEngel<N, T>>(&Shift(k), (a, b), zeros, &want, &what);
            for bits in [0, 40, 128] {
                let x = if bits == 0 {
                    vec![1]
                } else {
                    rand_x(bits, rng)
                };
                let shift = k + bits;
                let dctx = DynHyperCtx::new(&x, shift);
                let sctx = StaticHyperCtx::<N>::new(&x, shift);
                let want = want_hyper::<T>(&dctx, a, b, zeros, len);
                let what = format!("{what} x {bits} bits");
                check_algs::<DynNodeHyper<T>>(&dctx, (a, b), zeros, &want, &what);
                check_algs::<StaticNodeHyper<N, T>>(&sctx, (a, b), zeros, &want, &what);
            }
        }
    }
}

fn group_shapes<const SH: u32, const BIG_P: bool, const SIGNED: bool>(rng: &mut StdRng) {
    // S < 2^(-64 zeros): p(n) / q(n) 2^-k stays below it for these k.
    let shapes: &[(u64, usize)] = if BIG_P {
        &[(66, 0), (200, 2)]
    } else {
        &[(3, 0), (64, 0), (130, 2)]
    };
    let what = format!("SH {SH} big p {BIG_P} signed {SIGNED}");
    for &shape in shapes {
        group_suite::<Grp<SH, BIG_P, SIGNED>, 4>(rng, shape, &what);
        group_suite::<Grp<SH, BIG_P, SIGNED>, 8>(rng, shape, &what);
        group_suite::<Grp<SH, BIG_P, SIGNED>, 9>(rng, shape, &what);
        group_suite::<Grp<SH, BIG_P, SIGNED>, 64>(rng, shape, &what);
    }
}

#[test]
fn test_horner_group_boundaries() {
    let mut rng = StdRng::seed_from_u64(2001);
    group_shapes::<1, false, false>(&mut rng);
    group_shapes::<1, false, true>(&mut rng);
    group_shapes::<31, false, false>(&mut rng);
    group_shapes::<42, false, true>(&mut rng);
    group_shapes::<63, false, false>(&mut rng);
    group_shapes::<64, false, true>(&mut rng);
    group_shapes::<100, false, false>(&mut rng);
    group_shapes::<1, true, false>(&mut rng);
    group_shapes::<20, true, true>(&mut rng);
    group_shapes::<64, true, false>(&mut rng);
}

#[test]
fn test_horner_growing_groups() {
    let mut rng = StdRng::seed_from_u64(2002);
    for (k, zeros) in [(3, 0), (64, 0), (130, 2)] {
        for len in [1, 3, 8] {
            let (a, b) = (1, 1 + 700);
            let what = format!("growing k {k} len {len} zeros {zeros}");
            let want = want_bbp::<Growing>(k, a, b, zeros, len);
            check_algs::<DynNodeBBP<Growing>>(&Shift(k), (a, b), zeros, &want, &what);
            check_algs::<StaticNodeBBP<12, Growing>>(&Shift(k), (a, b), zeros, &want, &what);
            let want = want_engel::<Growing>(k, a, b, zeros, len);
            check_algs::<DynNodeEngel<Growing>>(&Shift(k), (a, b), zeros, &want, &what);
            for bits in [0, 64, 190] {
                let x = if bits == 0 {
                    vec![1]
                } else {
                    rand_x(bits, &mut rng)
                };
                let ctx = DynHyperCtx::new(&x, k + bits);
                let want = want_hyper::<Growing>(&ctx, a, b, zeros, len);
                let what = format!("{what} x {bits} bits");
                check_algs::<DynNodeHyper<Growing>>(&ctx, (a, b), zeros, &want, &what);
                let sctx = StaticHyperCtx::<12>::new(&x, k + bits);
                check_algs::<StaticNodeHyper<12, Growing>>(&sctx, (a, b), zeros, &want, &what);
            }
        }
    }
}

// Rect skips its powers' zero low limbs. x with zero low limbs, or zero limbs inside, gives
// powers whose zero counts differ from the plain padding, at every block length.
#[test]
fn test_rect_zero_limb_powers() {
    let mut rng = StdRng::seed_from_u64(2003);
    let top = |rng: &mut StdRng| rng.gen::<u64>() | 1 << 63;
    let xs = [
        vec![0, 0, top(&mut rng)],
        vec![0, top(&mut rng)],
        vec![5, 0, 0, top(&mut rng)],
        vec![0, rng.gen(), 0, top(&mut rng)],
        vec![0, 0, 0, 0, 0, 1 << 63],
    ];
    for x in xs {
        let bits = 64 * x.len() as u64;
        for (extra, len, t) in [(1, 3, 20), (5, 2, 13), (64, 5, 9), (3, 12, 30)] {
            let dctx = DynHyperCtx::new(&x, bits + extra);
            let sctx = StaticHyperCtx::<16>::new(&x, bits + extra);
            let want = want_hyper::<Exp>(&dctx, 1, 1 + t, 0, len);
            let swant = want_hyper::<SgHyp>(&dctx, 0, t, 0, len);
            for m in 1..=t as usize {
                let what = format!("x {x:x?} extra {extra} len {len} t {t} m {m}");
                let mut out = vec![0; len];
                let neg = rect_dyn_m::<Exp>(&dctx, 1, 1 + t, 0, &mut out, m);
                assert_close(&out, neg, &want, &what);
                let neg = rect_dyn_m::<SgHyp>(&dctx, 0, t, 0, &mut out, m);
                assert_close(&out, neg, &swant, &format!("{what} signed"));
                if m <= 16 && len + 3 <= 16 {
                    let neg = rect_static_m::<16, Exp>(&sctx, 1, 1 + t, 0, &mut out, m);
                    assert_close(&out, neg, &want, &format!("{what} static"));
                    let neg = rect_static_m::<16, SgHyp>(&sctx, 0, t, 0, &mut out, m);
                    assert_close(&out, neg, &swant, &format!("{what} static signed"));
                }
            }
        }
    }
}

#[test]
fn test_mul_work_continuous() {
    // Each tier's scale comes from the crossover that selects it, so the work never jumps
    // where the multiplier switches algorithms: neighbors differ by about their sizes.
    // mul_work follows mul_static's tiers alone since 2026-11-11 (dynamic nodes price their
    // products by MulEq).
    {
        let stat = true;
        let near = |a: (usize, usize), b: (usize, usize), grow: f64| {
            let r = mul_work(b.0, b.1) / mul_work(a.0, a.1);
            assert!(
                (0.85..=grow * 1.1).contains(&r),
                "mul_work {a:?} -> {b:?} (static {stat}): {r}"
            );
        };
        for l in [4, 40, 64, 100, 130, 200, 400, 1000, 3000, 10000] {
            for s in 3..l {
                near((l, s), (l, s + 1), (s + 1) as f64 / s as f64);
            }
        }
        for s in [3, 16, 52, 63, 64, 100, 200, 1000] {
            for l in s..8 * s + 64 {
                near((l, s), (l + 1, s), (l + 1) as f64 / l as f64);
            }
        }
    }
}

fn check_costs(costs: Costs, ok: Eligible, max_m: u64, what: &str) {
    for ns in ALGS.iter().filter_map(|&alg| costs.of(alg)) {
        assert!(ns.is_finite() && ns > 0.0, "{what}: {costs:?}");
    }
    assert_eq!(costs.horner.is_some(), ok.horner, "{what}");
    assert_eq!(costs.rect.is_some(), ok.rect, "{what}");
    assert!(ok.allows(costs.pick()), "{what}: picked {:?}", costs.pick());
    if let Some((m, _)) = costs.rect {
        assert!(
            m >= 1 && m as u64 <= max_m,
            "{what}: block length {m} of {max_m}"
        );
    }
}

fn check_node_costs<N: SeriesNode>(ctx: &N::Ctx, max_m: u64, what: &str) {
    let ranges = [(0, 1), (1, 2), (1, 17), (3, 300), (5, 3000)];
    for (a, b) in ranges {
        for len in [1, 2, 5, 64, 1000] {
            for zeros in [0, 3, 50] {
                let ok = N::eligible(ctx, a, b, len, zeros);
                let costs = N::costs(ctx, a, b, len, zeros, None);
                let what = format!("{what} [{a}, {b}) len {len} zeros {zeros}");
                check_costs(costs, ok, max_m.min(b - a), &what);
                // horner_first skips the model, only for ranges Horner can sum, and only for a
                // node with no table (a dynamic node reads the small-series table there).
                let first = N::horner_first(ctx, a, b, len, zeros);
                assert!((!first || ok.horner) && !(first && N::TABLED), "{what}");
                // Past it rect's guard settles what it can, and the least prediction decides
                // the rest.
                let g = N::guards(ctx, a, b, len, zeros);
                let (alg, m) = N::decide(ctx, a, b, len, zeros);
                assert!(ok.allows(alg), "{what}: decided {alg:?}");
                assert!(alg != SeriesAlg::Rect || m.is_some_and(|m| m >= 1 && m as u64 <= b - a), "{what}: rect at {m:?}");
                if g.rect_first {
                    assert_eq!(alg, SeriesAlg::Rect, "{what}");
                } else {
                    assert_eq!(alg, costs.pick(), "{what}");
                }
                assert!(!g.rect_first || ok.rect, "{what}");
                // The small-series table is read ahead of both, and only ever names an
                // algorithm that can sum the range.
                let small = N::small(ctx, a, b, len, zeros);
                assert!(small.map_or(true, |(alg, m)| ok.allows(alg) && (alg == SeriesAlg::Rect) == m.is_some()), "{what}: {small:?}");
                let want = small.map_or(if first { SeriesAlg::Horner } else { alg }, |s| s.0);
                assert_eq!(N::dispatch(ctx, a, b, len, zeros), want, "{what}");
                if ok.rect {
                    for m in [1, max_m.min(b - a)] {
                        let at = N::costs(ctx, a, b, len, zeros, Some(m as usize));
                        assert_eq!(at.rect.unwrap().0, m as usize, "{what}");
                        check_costs(at, ok, max_m.min(b - a), &what);
                    }
                }
            }
        }
    }
}

// The small-series table's layout: a frame's bucket and the frames it holds, a range's class,
// and a lookup at an explicit table (a pick, the equations past the extent or the buckets, and
// nothing for a class the table's last row doesn't reach).
#[test]
fn test_small_series_table_layout() {
    use crate::tuning::table::{row_index, Table2};
    // Buckets by octaves to 65535 limbs: every frame is in exactly the bucket whose span holds
    // it, and a class's `SMALL_ROWS` keys hold them all.
    assert_eq!(small_frame(0), None);
    assert_eq!(small_frame(65536), None);
    let mut last = 0;
    for len in 1..65536 {
        let b = small_frame(len).unwrap();
        let (lo, hi) = small_frame_span(b);
        assert!(lo <= len && len <= hi && b < SMALL_ROWS && (b == last || b == last + 1), "frame {len}: bucket {b} holds {lo}-{hi}");
        last = b;
    }
    assert_eq!((small_frame(1), small_frame(3), small_frame(5), small_frame(6), small_frame(11), small_frame(12), small_frame(255)), (Some(0), Some(1), Some(2), Some(2), Some(3), Some(3), Some(7)));
    assert_eq!((small_frame(256), small_frame(511), small_frame(512), small_frame(65535)), (Some(8), Some(8), Some(9), Some(SMALL_ROWS - 1)));
    assert_eq!((small_frame_span(5), small_frame_span(8), small_frame_span(15)), ((32, 63), (256, 511), (32768, 65535)));
    // A class is the kind alone: the family and x's width, whatever q's and r's sizes; none
    // past 64 bits, which the table doesn't hold.
    assert_eq!(small_class(0, 1, 0), Some(0));
    assert_eq!(small_class(0, 8, 0), small_class(0, 64, 0));
    assert_eq!(small_class(2, 8, 1), small_class(2, 40, 5), "neither q's size nor an r past 1 is in the class");
    assert_ne!(small_class(1, 8, 0), small_class(2, 8, 1));
    assert_eq!(small_class(1, 65, 0), None);
    assert_eq!(small_class(2, 8, 70), None);
    let kinds: Vec<usize> = [0u64, 2, 16, 17, 64, 65, 128, 129, 256, 1024, 1025, 2048, 2049].iter().map(|&b| small_kind_hyper(b)).collect();
    // No x, an x to a limb, of 2 limbs, of 3 to 16, of 17 to 32, and of more.
    assert_eq!(kinds, [2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7]);
    assert!(SMALL_KINDS == 8 && small_class(SMALL_KINDS - 1, 64, 64) == Some(SMALL_KINDS - 1));
    // A table by hand: class 3 at frames of 4-7 limbs runs Horner to 9 terms, binary splitting
    // from 10 and leaves 200 and more to the equations; at 8 limbs and up, rect from 3 terms.
    let key = |class: usize, frame: usize| (class * SMALL_ROWS + small_frame(frame).unwrap()) as u32;
    let t: &'static [u32] = Box::leak(Box::new([key(3, 4), 1, 0, key(3, 4), 10, 2, key(3, 4), 200, 3, key(3, 8), 1, 3, key(3, 8), 3, 1]));
    let table = Table2 { t, rows: Box::leak(row_index(t).into_boxed_slice()) };
    assert_eq!(small_pick(&table, 3, 4, 1), Some(SeriesAlg::Horner));
    assert_eq!(small_pick(&table, 3, 5, 9), Some(SeriesAlg::Horner));
    assert_eq!(small_pick(&table, 3, 5, 10), Some(SeriesAlg::BinSplit));
    assert_eq!(small_pick(&table, 3, 4, 199), Some(SeriesAlg::BinSplit));
    assert_eq!(small_pick(&table, 3, 7, 200), None, "past the extent");
    assert_eq!(small_pick(&table, 3, 8, 2), None);
    assert_eq!(small_pick(&table, 3, 15, 1 << 40), Some(SeriesAlg::Rect));
    assert_eq!(small_pick(&table, 3, 200, 5), Some(SeriesAlg::Rect), "a frame past the last row reads it");
    assert_eq!(small_pick(&table, 3, 65536, 5), None, "a frame past the buckets");
    assert_eq!(small_pick(&table, 2, 4, 5), Some(SeriesAlg::Horner), "below the first row: its first pick");
    // The default table hands everything to the equations.
    let none = Table2 { t: &[0, 0, 3], rows: &[0] };
    assert!((0..SMALL_KINDS).all(|c| [1, 30, 255, 4000].iter().all(|&f| small_pick(&none, c, f, 7).is_none())));
    // The live table through a node: its class, and a pick the range's own eligibility allows.
    type N = DynNodeHyper<Exp>;
    let ctx = DynHyperCtx::new(&[0x1234_5678], 80);
    assert_eq!(N::small_class(&ctx, 1, 20), small_class(small_kind_hyper(29), 4, 1));
    assert_eq!(<DynNodeBBP<ATanh> as SeriesNode>::small_class(&Shift(9), 1, 20), small_class(0, 5, 0));
    assert_eq!(<StaticNodeHyper<16, Exp> as SeriesNode>::small_class(&StaticHyperCtx::<16>::new(&[5], 80), 1, 20), None);
    if let Some((alg, m)) = N::small(&ctx, 1, 20, 3, 0) {
        assert!(N::eligible(&ctx, 1, 20, 3, 0).allows(alg) && (alg == SeriesAlg::Rect) == m.is_some());
        assert_eq!(N::dispatch(&ctx, 1, 20, 3, 0), alg);
    }
}

// The dispatch's guard at explicit knobs: rect outright from a rho and an x width; what `decide`
// then runs (binary splitting has no guard: its equation is read at any beta); and that rect at
// the guard's block length (the balance, unpredicted) sums the range.
#[test]
fn test_guards_settle_ranges_by_their_ratios() {
    type N = DynNodeHyper<Exp>;
    // An x of 40 limbs a bit below its shift, 300 terms into a 50-limb frame: an argument
    // nothing has reduced.
    let mut x = super::rand_vec(40, 0x5e71e5);
    x[39] |= 1 << 63;
    x[0] |= 1;
    let ctx = DynHyperCtx::new(&x, 64 * 40 + 1);
    let (a, b, len) = (1, 301, 50);
    let ok = N::eligible(&ctx, a, b, len, 0);
    assert!(ok.horner && ok.rect);
    let s = N::shape(&ctx, a, b, len, 0);
    // rho: x's 2560 bits over the frame's 3200 a term of 300. beta: the tree's P, 300 terms of
    // the shift's 2561 bits and q's, over the frame.
    assert!((s.rho() - 240.0).abs() < 1e-9 && s.x_limbs() == 40, "{}", s.rho());
    assert!((240.0..250.0).contains(&s.beta()), "{}", s.beta());
    let at = |rho: f64, x: usize| SeriesGuards { rect_rho: rho, rect_x: x };
    assert_eq!(s.guards(ok, &at(8.0, 16)), Guarded { rect_first: true });
    assert_eq!(s.guards(ok, &at(1e9, 0)), Guarded::default(), "off");
    assert!(s.guards(ok, &at(240.0, 40)).rect_first, "at its rho and its x");
    assert!(!s.guards(ok, &at(240.5, 40)).rect_first && !s.guards(ok, &at(240.0, 41)).rect_first);
    // Rect's guard needs rect.
    let (none, horner) = (Eligible { horner: false, rect: false }, Eligible { horner: true, rect: false });
    assert_eq!(s.guards(none, &at(0.0, 0)), Guarded::default());
    assert_eq!(s.guards(horner, &at(0.0, 0)), Guarded::default());
    // An x of one limb has no rho to speak of, and a narrow tree no beta.
    let narrow = N::shape(&DynHyperCtx::new(&[5], 3000), 1, 4, 200, 0);
    assert!(narrow.rho() < 0.01 && narrow.beta() < 1.0, "{} {}", narrow.rho(), narrow.beta());

    // decide: rect at a block length within the range under its guard; without it, the least
    // of the three predictions, binary splitting's among them at a beta of 240 (it is far the
    // dearest there, by its own equation and not by a guard).
    let costs = N::costs(&ctx, a, b, len, 0, None);
    let (alg, m) = s.decide_at(ok, Guarded { rect_first: true });
    let m = m.expect("rect's block length");
    assert!(alg == SeriesAlg::Rect && (1..=300).contains(&m), "{alg:?} {m}");
    assert!(costs.bin_split.is_finite() && costs.bin_split > 4.0 * costs.rect.unwrap().1.min(costs.horner.unwrap()), "{costs:?}");
    assert_eq!(s.decide_at(ok, Guarded::default()).0, costs.pick());
    assert_eq!(s.decide_at(none, Guarded::default()).0, SeriesAlg::BinSplit);
    // The live dispatch is `decide_at` under the live knobs' verdict.
    let live = N::guards(&ctx, a, b, len, 0);
    assert_eq!(live, s.guards(ok, &SeriesGuards::read()));
    assert_eq!(N::decide(&ctx, a, b, len, 0), s.decide_at(ok, live));
    assert_eq!(N::dispatch(&ctx, a, b, len, 0), s.decide_at(ok, live).0);
    // Static nodes have no guards.
    let sctx = StaticHyperCtx::<64>::new(&x, 64 * 40 + 1);
    assert_eq!(<StaticNodeHyper<64, Exp> as SeriesNode>::guards(&sctx, a, b, len, 0), Guarded::default());

    // Rect at the guard's block length against binary splitting, within a unit of the last limb.
    let (mut by_rect, mut by_tree) = (vec![0u64; len], vec![0u64; len]);
    N::rect(&ctx, a, b, 0, &mut by_rect, Some(m));
    series_eval_at::<N>(&ctx, a, b, 0, &mut by_tree, SeriesAlg::BinSplit);
    let (hi, lo) = if cmp_buf(&by_rect, &by_tree) == Ordering::Less { (&by_tree, &by_rect) } else { (&by_rect, &by_tree) };
    let mut d = hi.clone();
    sub_buf(&mut d, lo);
    assert!(buf_len(&d) <= 1 && d[0] <= 1, "rect at m = {m} is {} units off", d[0]);
}

// A dynamic node's equations over the same edge shapes: every part finite and non-negative, an
// algorithm's parts there exactly when it can sum the range, and the predictions the dispatch
// compares are those parts under the live prices (rect's at an explicit block length; with
// none it may read as Horner plus its fixed cost, where it can't win).
fn check_node_parts<N: SeriesNode>(ctx: &N::Ctx, what: &str) {
    let (env, eq) = (SeriesEnv::read(), SeriesEq::read());
    for (a, b) in [(0, 1), (1, 2), (1, 17), (3, 300), (5, 3000)] {
        for len in [1, 2, 5, 64, 1000] {
            for zeros in [0, 3, 50] {
                let what = format!("{what} [{a}, {b}) len {len} zeros {zeros}");
                let ok = N::eligible(ctx, a, b, len, zeros);
                let m = ok.rect.then_some(((b - a) as usize).min(3));
                let parts = N::parts(ctx, a, b, len, zeros, m, &env);
                let costs = N::costs(ctx, a, b, len, zeros, m);
                for (i, alg) in ALGS.into_iter().enumerate() {
                    assert_eq!(parts[i].is_some(), ok.allows(alg), "{what}: {alg:?}");
                    let Some(p) = parts[i] else { continue };
                    assert!(p.features().iter().all(|v| v.is_finite() && *v >= 0.0), "{what}: {alg:?} {p:?}");
                    let prices = [&eq.horner, &eq.rect, &eq.bin_split][i];
                    let (want, got) = (p.ns_with(prices, eq.wake), costs.of(alg).unwrap());
                    assert!((want - got).abs() <= 1e-9 * want, "{what}: {alg:?} {want} != {got}");
                }
            }
        }
    }
}

fn static_costs<const N: usize>(x: &[u64]) {
    // len up to N - 3 and N - 4, the narrow and wide Horner bounds, and past them.
    for k in [0, 1, 8, 130] {
        check_node_costs::<StaticNodeBBP<N, BbpLin>>(&Shift(k), 16, "static bbp");
        check_node_costs::<StaticNodeBBP<N, WideQ>>(&Shift(k), 16, "static wide bbp");
        check_node_costs::<StaticNodeEngel<N, Exp>>(&Shift(k), 16, "static engel");
    }
    let ctx = StaticHyperCtx::<N>::new(&[1], 5);
    check_node_costs::<StaticNodeHyper<N, Hyp>>(&ctx, 16, "static hyper x = 1");
    let ctx = StaticHyperCtx::<N>::new(x, 64 * x.len() as u64 + 3);
    check_node_costs::<StaticNodeHyper<N, Exp>>(&ctx, 16, "static hyper x");
}

#[test]
fn test_series_costs_edges() {
    let mut rng = StdRng::seed_from_u64(91);
    for k in [0, 1, 8, 130] {
        check_node_costs::<DynNodeBBP<BbpLin>>(&Shift(k), u64::MAX, "bbp");
        check_node_costs::<DynNodeBBP<SgBbp>>(&Shift(k), u64::MAX, "signed bbp");
        check_node_costs::<DynNodeBBP<WideQ>>(&Shift(k), u64::MAX, "wide bbp");
        check_node_costs::<DynNodeEngel<Exp>>(&Shift(k), u64::MAX, "engel");
        check_node_parts::<DynNodeBBP<BbpLin>>(&Shift(k), "bbp");
        check_node_parts::<DynNodeBBP<SgBbp>>(&Shift(k), "signed bbp");
        check_node_parts::<DynNodeBBP<WideQ>>(&Shift(k), "wide bbp");
        check_node_parts::<DynNodeEngel<Exp>>(&Shift(k), "engel");
        let ctx = DynHyperCtx::new_shift(k);
        check_node_costs::<DynNodeHyper<Hyp>>(&ctx, u64::MAX, "hyper x = 1");
        check_node_costs::<DynNodeHyper<RBig>>(&ctx, u64::MAX, "hyper r > q");
        check_node_costs::<DynNodeHyper<WideTerms<true>>>(&ctx, u64::MAX, "wide hyper");
        check_node_parts::<DynNodeHyper<Hyp>>(&ctx, "hyper x = 1");
        check_node_parts::<DynNodeHyper<RBig>>(&ctx, "hyper r > q");
        check_node_parts::<DynNodeHyper<WideTerms<true>>>(&ctx, "wide hyper");
    }
    for xl in [1, 2, 3, 256] {
        let x = rand_x(64 * xl as u64, &mut rng);
        for k in [64 * xl as u64, 64 * xl as u64 + 7, 64 * xl as u64 + 4000] {
            let ctx = DynHyperCtx::new(&x, k);
            check_node_costs::<DynNodeHyper<Exp>>(&ctx, u64::MAX, "hyper x");
            check_node_parts::<DynNodeHyper<Exp>>(&ctx, "hyper x");
        }
        // k below bits(x): bin_split only.
        let ctx = DynHyperCtx::new(&x, 64 * xl as u64 - 1);
        check_node_costs::<DynNodeHyper<Exp>>(&ctx, u64::MAX, "hyper k < bits(x)");
    }
    let x = rand_x(128, &mut rng);
    static_costs::<4>(&x[..1]);
    static_costs::<8>(&x);
    static_costs::<67>(&x);
    static_costs::<68>(&x);
}

#[test]
fn test_horner_cost_grows_with_terms() {
    // Horner's predicted time grows with its terms, which add frame limbs, divisions and steps:
    // twice the terms always cost more. One more term can take a little off, as the groups
    // realign and a first term left alone with q = 1 divides by nothing, but never much.
    let mut rng = StdRng::seed_from_u64(92);
    let x = rand_x(192, &mut rng);
    let (shift, hyper_x) = (Shift(3), DynHyperCtx::new(&x, 200));
    for len in [1, 8, 100] {
        let at = |b: u64| {
            [
                DynNodeBBP::<BbpQuad>::costs(&shift, 1, b, len, 0, None).horner,
                DynNodeEngel::<Exp>::costs(&shift, 1, b, len, 0, None).horner,
                DynNodeHyper::<Hyp>::costs(&DynHyperCtx::new_shift(3), 1, b, len, 0, None).horner,
                DynNodeHyper::<Exp>::costs(&hyper_x, 1, b, len, 0, None).horner,
            ]
            .map(|ns| ns.unwrap())
        };
        let mut last = at(2);
        for b in 3..600 {
            let (ns, half) = (at(b), at(1 + (b - 1) / 2));
            for i in 0..4 {
                assert!(ns[i] >= 0.8 * last[i], "kind {i} len {len} b {b}: {} after {}", ns[i], last[i]);
                assert!(ns[i] > half[i], "kind {i} len {len} b {b}: {} is not past {} at half the terms", ns[i], half[i]);
            }
            last = ns;
        }
    }
}

// The leaves and merges of the tree bin_split_tree_at builds over [a, b).
fn tree_nodes<N: Node>(ctx: &N::Ctx, a: u64, b: u64) -> (u64, u64) {
    if b - a <= N::leaf_terms().max(1) || N::limbs(ctx, a, b) < N::leaf_limbs() {
        return (1, 0);
    }
    let m = (a + b) / 2;
    let (l, r) = (tree_nodes::<N>(ctx, a, m), tree_nodes::<N>(ctx, m, b));
    (l.0 + r.0, l.1 + r.1 + 1)
}

#[test]
fn test_bin_split_parts_count_the_tree() {
    // Binary splitting's equation walks the tree the code builds: its allocations are the
    // tree's leaves and merges (two buffers a leaf; a merge's P and Q, and BBP's cross product;
    // finalize's scratch). The walk takes q's size at the range's midpoint for every node of a
    // level, where the tree's own estimate is smaller at its left nodes than at its right ones,
    // so with a level right at the leaf cutoff the two can differ by a node or two: within a
    // fifth, or two leaves and their merges where that is more (1000 terms at shift 7 under an
    // Engel leaf of 64 limbs: the walk's 250-term nodes are all leaves at 66 limbs' estimate,
    // and the tree splits the two on the right).
    let env = SeriesEnv::read();
    let allocs = |leaves: u64, merges: u64, per_merge: u64| (2 * leaves + per_merge * merges + 1) as f64;
    let near = |got: f64, want: f64, per_merge: u64| (got - want).abs() <= (0.2 * want).max((2 * (2 + per_merge)) as f64);
    for shift in [1u64, 7, 64, 200, 4096] {
        for t in [1u64, 2, 3, 5, 8, 13, 100, 257, 1000] {
            // Trees narrow enough to stay on the caller's thread, where the counts are whole.
            if t * (shift + 12) / 64 >= crate::tuning::k::bin_split::dyn_par().min(1 << 14) {
                continue;
            }
            let ctx = Shift(shift);
            let (leaves, merges) = tree_nodes::<DynNodeEngel<Exp>>(&ctx, 1, 1 + t);
            let got = DynNodeEngel::<Exp>::parts(&ctx, 1, 1 + t, 8, 0, None, &env)[2].unwrap().alloc;
            let want = allocs(leaves, merges, 2);
            assert!(near(got, want, 2), "engel shift {shift} t {t}: {got} allocations, the tree makes {want}");
            let (leaves, merges) = tree_nodes::<DynNodeBBP<BbpLin>>(&ctx, 1, 1 + t);
            let got = DynNodeBBP::<BbpLin>::parts(&ctx, 1, 1 + t, 8, 0, None, &env)[2].unwrap().alloc;
            let want = allocs(leaves, merges, 3);
            assert!(near(got, want, 3), "bbp shift {shift} t {t}: {got} allocations, the tree makes {want}");
            // A tree of one leaf, and one of one term a leaf, are counted exactly.
            if leaves == 1 || leaves == t {
                assert_eq!(got, want, "bbp shift {shift} t {t}");
            }
        }
    }
}

#[test]
fn test_horner_products_closed_form() {
    // Horner's products by a multi-limb x in hi_mul_buf's tier are summed in closed form over the
    // frames' stretches (the whole accumulator; falling, cut where the frame passes the kernels'
    // cutoff and where it passes x; one limb). Against the same sum taken term by term: a frame
    // of L limbs by x of xl costs hi_mul_buf's own equation (`HiEq`), x clipped to the frame.
    let mut rng = StdRng::seed_from_u64(94);
    let env = SeriesEnv::read();
    let hi = crate::utils::mul::HiEq::read();
    let partial = crate::tuning::k::mul::dyn_partial_mul();
    for xl in [2usize, 3, 7, 40] {
        let x = rand_x(64 * xl as u64, &mut rng);
        for extra in [0u64, 1, 9, 64, 200, 3000] {
            let shift = 64 * xl as u64 + extra;
            let ctx = DynHyperCtx::new(&x, shift);
            for (len, zeros, t) in [(1usize, 0usize, 1u64), (5, 0, 2), (5, 3, 40), (30, 0, 7), (64, 1, 300), (200, 0, 1000)] {
                let acc = len + 4;
                if acc > partial {
                    continue;
                }
                let (top, k) = ((len + zeros + 2) as f64, extra as f64);
                let want: f64 = (1..=t).map(|i| hi.ns((top - k * i as f64 / 64.0).clamp(1.0, acc as f64), xl as f64)).sum();
                let got = DynNodeHyper::<Exp>::parts(&ctx, 1, 1 + t, len, zeros, None, &env)[0].unwrap().product;
                assert!((got - want).abs() <= 1e-9 * want, "xl {xl} shift {shift} len {len} zeros {zeros} t {t}: {got} != {want}");
            }
        }
    }
}

#[test]
fn test_series_equations_share_horners_kernel_prices() {
    // Rect runs Horner's kernels, so its equation takes Horner's prices for their passes (the
    // two-limb divisions, the passes by one- and two-limb multipliers, the fills and copies)
    // and has its own for what is its own: its products and fixed cost, a term, its term adds
    // (add_mul passes of a weight over a power), and the one-limb divisions, which its groups
    // make in every range and Horner's long series hardly make at all. It has no price a call
    // or a group: its adds' calls and its groups' fixed costs are in its price a term.
    assert_eq!(SeriesEq::RECT, [0, 2, 6, 8, 11]);
    for eq in [SeriesEq::read()] {
        for c in 0..Parts::N {
            if SeriesEq::RECT.contains(&c) {
                continue;
            }
            assert_eq!(eq.rect[c], if c == 9 || c == 12 { 0.0 } else { eq.horner[c] }, "column {}", Parts::NAMES[c]);
        }
        // No price for work an algorithm doesn't do.
        for (name, cols, prices) in [("horner", &SeriesEq::HORNER[..], &eq.horner), ("bin_split", &SeriesEq::BIN_SPLIT[..], &eq.bin_split)] {
            for c in (0..Parts::N).filter(|c| !cols.contains(c)) {
                // Horner's price for two-limb passes is its one-limb price times the kernels' ratio.
                let want = if name == "horner" && c == 5 { crate::tuning::k::series::horner_pass2() * prices[4] } else { 0.0 };
                assert_eq!(prices[c], want, "{name}: column {}", Parts::NAMES[c]);
            }
        }
    }
}

// A Hyper series with a 32-bit q and r = 1: rect's narrow accumulator divides by a one-limb Q,
// so its groups hold exactly two terms.
struct Q32H;
impl HyperSeries for Q32H {
    const START: u64 = 1;
    fn tail_bits(_: u64, _: i64) -> u64 {
        0
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        (1 << 31) + 2 * n + 1
    }
    fn r(_: u64) -> u64 {
        1
    }
}

// An x of exactly `bits` bits whose limbs are all nonzero.
fn x_bits(bits: u64) -> Vec<u64> {
    let len = bits.div_ceil(64) as usize;
    let mut x = vec![0x9e37_79b9_7f4a_7c15u64; len];
    let top = bits - 64 * (len as u64 - 1);
    x[len - 1] = (x[len - 1] >> (64 - top)) | 1 << (top - 1);
    x
}

#[test]
fn test_rect_groups_divide_at_their_own_frames() {
    // Eight terms in blocks of four, groups of two: the kernel divides after terms 7, 5, 3 and 1
    // (deepest first), each time over the frame of that term, not twice over each block's
    // shallowest (which read 1.4x the kernel's limbs on a range of two blocks). 320 bits a term
    // from the shift: term i's frame is len + 2 - 5 i limbs. The sum a division takes is a q
    // above its term (the group's other q) in a top limb half full: 31 / 64 - 1 / 2 limbs more.
    let env = SeriesEnv::read();
    let (x, len) = (x_bits(128), 60);
    let ctx = DynHyperCtx::new(&x, 128 + 320);
    let r = DynNodeHyper::<Q32H>::parts(&ctx, 1, 9, len, 0, Some(4), &env)[1].unwrap();
    let frame = |i: usize| (len + 2 - 5 * i) as f64 + 31.0 / 64.0 - 0.5;
    let want = frame(1) + frame(3) + frame(5) + frame(7);
    assert!((r.div1 - want).abs() < 0.01, "{} limbs divided, the kernel's groups take {want}", r.div1);
    assert_eq!((r.div2, r.term, r.mul1, r.mul2), (0.0, 8.0, 0.0, 0.0), "{r:?}");
    // One block: the same four groups, at the same frames.
    let r = DynNodeHyper::<Q32H>::parts(&ctx, 1, 9, len, 0, Some(8), &env)[1].unwrap();
    assert!((r.div1 - want).abs() < 0.01, "{} limbs divided in one block", r.div1);
}

#[test]
fn test_rect_powers_are_as_wide_as_x_bits() {
    // Power j of x' has x's bits j times over, and the kernel skips its zero low limbs: an x of
    // two limbs and 66 bits makes powers of 2, 3, ..., 9 limbs (ceil(66 j / 64)) where one of
    // 128 bits makes 2, 4, ..., 16. One block of eight terms in a frame that holds every power
    // whole, so each term adds its power once, and a second time as often as its weight spills:
    // the weights are 1 and a 31-bit q by turns, 15.5 of 64 places on average.
    let env = SeriesEnv::read();
    let len = 200;
    let spill = 1.0 + 15.5 / 64.0;
    for (bits, limbs) in [(128u64, 72.0), (66, 44.0), (70, 44.0), (65, 44.0), (127, 72.0)] {
        let x = x_bits(bits);
        let r = DynNodeHyper::<Q32H>::parts(&DynHyperCtx::new(&x, bits + 64), 1, 9, len, 0, Some(8), &env)[1].unwrap();
        let want = spill * limbs;
        assert!((r.add - want).abs() < 1e-3 * want, "x of {bits} bits: {} limbs added, its powers have {want}", r.add);
    }
}

#[test]
fn test_rect_adds_closed_form() {
    // Rect's term adds are summed block by block in closed form: a term adds its power over the
    // power's limbs or the limbs under its own bit, whichever is fewer, and within a block the
    // one grows as the other falls. Against the same sum taken term by term.
    let env = SeriesEnv::read();
    for bits in [70u64, 128, 200, 1000] {
        let x = x_bits(bits);
        for k in [1u64, 20, 64, 300] {
            let ctx = DynHyperCtx::new(&x, bits + k);
            for (len, zeros, t, m) in [(40usize, 0usize, 60u64, 7usize), (40, 0, 60, 60), (300, 2, 500, 31), (300, 0, 90, 16), (64, 0, 1000, 5), (500, 1, 40, 40)] {
                let r = DynNodeHyper::<Q32H>::parts(&ctx, 1, 1 + t, len, zeros, Some(m), &env)[1].unwrap();
                let (p, xw, under, fall) = ((len + 3) as f64, bits as f64 / 64.0, (len + zeros + 1) as f64, k as f64 / 64.0);
                let mut want = 0.0;
                // The blocks from the deep end, each of m terms but the shallowest; the terms of
                // a block of n are in pairs but for one left over when n is odd.
                let n_blocks = t.div_ceil(m as u64);
                let first_len = t - (n_blocks - 1) * m as u64;
                for b in 0..n_blocks {
                    let (first, n) = if b == 0 { (1, first_len) } else { (1 + first_len + (b - 1) * m as u64, m as u64) };
                    let others = (2 * (n / 2)) as f64 / n as f64;
                    let h = others * 15.5 / 64.0;
                    for u in 0..n {
                        let (power, inside) = (((u + 1) as f64 * xw).ceil().min(p), under - fall * (first + u) as f64);
                        if inside <= 0.0 {
                            continue;
                        }
                        want += if power <= inside.min(p) { (1.0 + h) * power } else { (inside.min(p) - 0.5).max(0.0) + h * (inside.min(p) + 0.5) };
                    }
                }
                assert!((r.add - want).abs() <= 0.03 * want + 2.0, "x {bits} bits, k {k}, len {len}, zeros {zeros}, t {t}, m {m}: {} != {want}", r.add);
            }
        }
    }
}

// Horner's groups are counted whole for a q of constant size: 32 bits fill a 128-bit group with
// exactly four terms, so a range of eight makes two divisions, at the frames of its first and
// its fifth term (an average of 3.63 terms a group counted three, and read 39% high on such a
// range).
struct Q32;
impl EngelSeries for Q32 {
    const START: u64 = 1;
    fn tail_bits(_: u64, _: i64) -> u64 {
        0
    }
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        (1 << 31) + 2 * n + 1
    }
}

#[test]
fn test_horner_groups_are_whole() {
    type N = DynNodeEngel<Q32>;
    let env = SeriesEnv::read();
    // 64 limbs a term from the shift, in a frame of 454 limbs below 64 zero limbs: term i's
    // frame is 454 + 64 + 2 - 64 i limbs.
    let (len, zeros) = (454, 64);
    let frame = |i: usize| len + zeros + 2 - 64 * i;
    let h = N::parts(&Shift(4096), 1, 9, len, zeros, None, &env)[0].unwrap();
    assert_eq!((h.div2, h.div1, h.term), ((frame(1) + frame(5)) as f64, 0.0, 8.0), "{h:?}");
    // Nine terms: the shallowest group holds the one left over.
    let h = N::parts(&Shift(4096), 1, 10, len + 64, zeros, None, &env)[0].unwrap();
    assert_eq!(h.div2 + h.div1, (3 * 64 + frame(1) + frame(2) + frame(6)) as f64, "{h:?}");
    // Nine terms' shallowest group is one term: a one-limb Q, divided by div_prim.
    assert_eq!((h.div1, h.step), ((64 + frame(1)) as f64, 3.0), "{h:?}");
    // A q that grows fills groups that grow towards the shallow end: the range's bits over what
    // a group holds of its 128 (all but the half term it leaves unfilled). q(n) = n over 1000
    // terms is log2 1000! = 8529 bits, 69 groups.
    let e = <DynNodeEngel<Exp> as SeriesNode>::parts(&Shift(64), 1, 1001, 100, 1, None, &env)[0].unwrap();
    let walked = walk_groups(1, 1001, |_| 1, |n| n as u128).len() as f64;
    assert!((e.step - walked).abs() <= 1.0 && (walked - 69.0).abs() <= 1.0, "{} groups of 1000 terms, the kernel's rule makes {walked}", e.step);
    // And 32 terms of it, 118 bits in all, are one group: one division, over the first term's
    // frame, by a two-limb Q.
    let e = <DynNodeEngel<Exp> as SeriesNode>::parts(&Shift(1152), 1, 33, 564, 18, None, &env)[0].unwrap();
    assert_eq!((e.step, e.div1, e.div2), (1.0, 0.0, (564 + 18 + 2 - 18) as f64), "{e:?}");
}

// The kernel's grouping rule (`group` in series.rs) walked term by term from the deep end: a
// group takes terms while Q = ∏q and p Q stay within 128 bits, 32 at most. Each group's
// shallowest term and its Q's bits, deepest group first.
fn walk_groups(a: u64, b: u64, p: impl Fn(u64) -> u128, q: impl Fn(u64) -> u128) -> Vec<(u64, u32)> {
    let (mut out, mut n) = (Vec::new(), b);
    while n > a {
        let (mut m, mut q_all, mut p_max) = (0u64, 1u128, 0u128);
        for j in (a..n).rev() {
            let (pj, qj) = (p(j), q(j));
            let fits = q_all.checked_mul(qj).is_some_and(|nq| p_max.max(pj).checked_mul(nq).is_some());
            if m != 0 && (m == 32 || !fits) {
                break;
            }
            (q_all, p_max, m) = (q_all * qj, p_max.max(pj), m + 1);
        }
        n -= m;
        out.push((n, 128 - q_all.leading_zeros()));
    }
    out
}

#[test]
fn test_horner_groups_of_a_growing_q() {
    // Horner's divisions for a q that grows, against the kernel's rule walked term by term: the
    // groups' count, the limbs they divide (each at the frame of its shallowest term), and
    // which of them a one-limb Q divides (div_prim: the shallowest group, when the few terms
    // the others leave it fit a limb). q(n) = n (exp's) and 4 n + 3, in frames of a few terms to
    // a limb down to a term in twenty limbs, where a range is one group or two.
    let env = SeriesEnv::read();
    let frame = |len: usize, k: u64, i: u64| ((len + 2) as f64 - k as f64 * i as f64 / 64.0).clamp(1.0, (len + 4) as f64);
    let mut worst = (0.0f64, String::new());
    for (t, len) in [(27u64, 788usize), (38, 845), (47, 375), (77, 82), (96, 79), (132, 285), (165, 276), (215, 441), (1000, 100), (3000, 900)] {
        let k = 64 * len as u64 / t;
        for bbp in [false, true] {
            let (h, groups) = if bbp {
                (DynNodeBBP::<BbpLin>::parts(&Shift(k), 1, 1 + t, len, 0, None, &env)[0].unwrap(), walk_groups(1, 1 + t, |_| 1, |n| (4 * n + 3) as u128))
            } else {
                (DynNodeEngel::<Exp>::parts(&Shift(k), 1, 1 + t, len, 0, None, &env)[0].unwrap(), walk_groups(1, 1 + t, |_| 1, |n| n as u128))
            };
            let limbs = |one: bool| groups.iter().filter(|g| (g.1 <= 64) == one).map(|g| frame(len, k, g.0)).sum::<f64>();
            let (one, two) = (limbs(true), limbs(false));
            let what = format!("{} terms of {} in {len} limbs: {} groups, {:.0} + {:.0} limbs; the kernel's rule {} groups, {one:.0} + {two:.0}", t, if bbp { "4 n + 3" } else { "n" }, h.step, h.div1, h.div2, groups.len());
            assert!((h.step - groups.len() as f64).abs() <= 1.0, "{what}");
            let off = (h.div1 + h.div2) / (one + two) - 1.0;
            if off.abs() > worst.0 {
                worst = (off.abs(), what.clone());
            }
            // Within a group's worth at the first term's frame where the count is a group off (a
            // range a hair from a whole number of groups), and a tenth otherwise.
            let slack = if h.step == groups.len() as f64 { 0.1 * (one + two) } else { frame(len, k, 1) + 0.1 * (one + two) };
            assert!((h.div1 + h.div2 - one - two).abs() <= slack, "{what}");
        }
    }
    assert!(worst.0 < 0.5, "{}", worst.1);
}

// A Hyper series without x whose 8-bit q and r fill a 128-bit group with seventeen terms.
struct Geo8T;
impl HyperSeries for Geo8T {
    no_tail!(1);
    fn p(_: u64) -> u64 {
        1
    }
    fn q(n: u64) -> u64 {
        131 + 2 * (n & 31)
    }
    fn r(n: u64) -> u64 {
        129 + 2 * (n & 31)
    }
}

#[test]
fn test_horner_products_by_r_run_over_what_the_group_below_left() {
    // A group's product by R runs over the sum as the group below left it, so the deepest
    // group multiplies nothing: one group, no pass at all; and the shallowest group's R is its
    // own few terms', a one-limb pass when they fit a limb.
    let env = SeriesEnv::read();
    let ctx = DynHyperCtx::new_shift(6072);
    let one = DynNodeHyper::<Geo8T>::parts(&ctx, 1, 10, 760, 0, None, &env)[0].unwrap();
    assert_eq!((one.step, one.mul1, one.mul2), (1.0, 0.0, 0.0), "{one:?}");
    // 56 terms at 7.5 limbs a term: the kernel's rule makes groups from the deep end, and each
    // but the deepest multiplies the frame of the group below it.
    let (t, len, k) = (56u64, 413usize, 480u64);
    let h = DynNodeHyper::<Geo8T>::parts(&DynHyperCtx::new_shift(k), 1, 1 + t, len, 0, None, &env)[0].unwrap();
    let groups = walk_groups(1, 1 + t, |_| 1, |n| (131 + 2 * (n & 31)) as u128);
    let frame = |i: u64| ((len + 2) as f64 - k as f64 * i as f64 / 64.0).clamp(1.0, (len + 4) as f64);
    // Deepest first: a group's pass is over the frame of the group before it in this order.
    let passes: f64 = groups.windows(2).map(|w| frame(w[0].0)).sum();
    assert_eq!(h.step, groups.len() as f64, "{h:?}");
    assert!((h.mul1 + h.mul2 - passes).abs() <= 0.1 * passes, "{} limbs multiplied by R, the kernel's groups {passes}", h.mul1 + h.mul2);
}

#[test]
fn test_bin_split_finalize_and_x_leaf_terms() {
    // Finalize's division is a pass a quotient limb by a Q of one limb (div1) or two (div2: the
    // division's own equations start at four, and Knuth's reads 27% high at two), and
    // division's equation from three. Q32's terms add 31 bits each to Q.
    // Counted serial: a tree wide enough to split across the pool has its levels' counts
    // divided by the gain, and whether one is depends on the gates in force (it is under the
    // schema's defaults, which a target with no preset resolves).
    let mut env = SeriesEnv::read();
    env.par_gain = 0.0;
    let tree = |t: u64| DynNodeEngel::<Q32>::parts(&Shift(4096), 1, 1 + t, 300, 0, None, &env)[2].unwrap();
    let (one, two, more) = (tree(2), tree(4), tree(7));
    assert_eq!((one.div1, one.div2, one.division), (301.0, 0.0, 0.0), "{one:?}");
    assert_eq!((two.div1, two.div2, two.division), (0.0, 301.0, 0.0), "{two:?}");
    assert!(more.div1 == 0.0 && more.div2 == 0.0 && more.division > 0.0, "{more:?}");
    // Only an x-leaf's terms are steps.
    assert_eq!((one.step, two.step, more.step), (0.0, 0.0, 0.0));
    let mut rng = StdRng::seed_from_u64(95);
    let x = rand_x(64 * 3, &mut rng);
    for t in [1u64, 5, 40, 300] {
        let p = DynNodeHyper::<Exp>::parts(&DynHyperCtx::new(&x, 64 * 3 + 9), 1, 1 + t, 40, 0, None, &env)[2].unwrap();
        assert_eq!(p.step, t as f64, "{t} terms: {p:?}");
    }
}

// Typed contexts: term counts and zeros from each series' tail bound.

// floor(|S| 2^(64 len)) for S = a - b, from two Wants at len limbs.
fn signed_diff((a, an): &Want, (b, bn): &Want) -> Vec<u64> {
    let mut d = vec![0; a.len() + 1];
    if an == bn {
        let (hi, lo) = match cmp_buf(a, b) {
            Ordering::Less => (b, a),
            _ => (a, b),
        };
        d[..hi.len()].copy_from_slice(hi);
        sub_buf(&mut d, lo);
    } else {
        d[..a.len()].copy_from_slice(a);
        add_buf(&mut d, b);
    }
    d
}

fn bit_count(x: &[u64]) -> u64 {
    let x = &x[..buf_len(x)];
    x.last()
        .map_or(0, |&t| 64 * x.len() as u64 - t.leading_zeros() as u64)
}

// |R_n| < 2^-b for the tail R_n = S[START, end) - S[START, n), each floored at len limbs, with
// end far enough past n that its own tail doesn't matter: d + 2 < 2^(64 len - b).
fn assert_tail(full: &Want, part: &Want, b: u64, len: usize, what: &str) {
    let mut d = signed_diff(full, part);
    d.push(0);
    add_buf(&mut d, &[2]);
    assert!(
        bit_count(&d) <= 64 * len as u64 - b,
        "{what}: tail is not below 2^-{b}"
    );
}

fn tail_len(b: u64) -> usize {
    ((b + 128) / 64) as usize + 1
}

fn zero_want(len: usize) -> Want {
    (vec![0; len], false)
}

// Each in-crate tail bound against the exact tail.
#[test]
fn test_tail_bounds_hold() {
    let mut rng = StdRng::seed_from_u64(93);
    for s in [1, 2, 5, 13, 64, 100] {
        for n in [1, 2, 3, 5, 10, 30] {
            // e^y - 1 at y = 2^-s (Engel) and y = x 2^-k just below 2^-s (Hyper).
            let what = format!("exp s {s} n {n}");
            let ctx = Engel::<Exp>::new(s);
            let b = ctx.tail_bits(n);
            let len = tail_len(b);
            let end = ctx.range(64 * len as u64 + 64).1.max(n + 1) + 20;
            let full = want_engel::<Exp>(s, 1, end, 0, len);
            let part = if n > 1 {
                want_engel::<Exp>(s, 1, n, 0, len)
            } else {
                zero_want(len)
            };
            assert_tail(&full, &part, b, len, &what);

            let x = rand_x(70, &mut rng);
            let ctx = Hyper::<Exp>::new(&x, 70 + s).unwrap();
            let b = ctx.tail_bits(n);
            let len = tail_len(b);
            let end = ctx.range(64 * len as u64 + 64).1.max(n + 1) + 20;
            let dctx = DynHyperCtx::new(&x, 70 + s);
            let full = want_hyper::<Exp>(&dctx, 1, end, 0, len);
            let part = if n > 1 {
                want_hyper::<Exp>(&dctx, 1, n, 0, len)
            } else {
                zero_want(len)
            };
            assert_tail(&full, &part, b, len, &format!("{what} hyper"));

            // ATanh at y = 2^-s (BBP) and y = x 2^-k (Hyper).
            let what = format!("atanh s {s} n {n}");
            let ctx = Bbp::<ATanh>::new(s);
            let b = ctx.tail_bits(n);
            let len = tail_len(b);
            let end = ctx.range(64 * len as u64 + 64).1.max(n + 1) + 20;
            let full = want_bbp::<ATanh>(s, 1, end, 0, len);
            let part = if n > 1 {
                want_bbp::<ATanh>(s, 1, n, 0, len)
            } else {
                zero_want(len)
            };
            assert_tail(&full, &part, b, len, &what);

            let ctx = Hyper::<ATanh>::new(&x, 70 + s).unwrap();
            let b = ctx.tail_bits(n);
            let len = tail_len(b);
            let end = ctx.range(64 * len as u64 + 64).1.max(n + 1) + 20;
            let full = want_hyper::<ATanh>(&dctx, 1, end, 0, len);
            let part = if n > 1 {
                want_hyper::<ATanh>(&dctx, 1, n, 0, len)
            } else {
                zero_want(len)
            };
            assert_tail(&full, &part, b, len, &format!("{what} hyper"));
        }
    }
    // ln(2)'s four series, summed from 0 at shift 6.
    fn ln2<const A: u64, const B: u64>(n: u64) {
        let ctx = Bbp::<Ln2S<A, B>>::new(6);
        let b = ctx.tail_bits(n);
        let len = tail_len(b);
        let end = ctx.range(64 * len as u64 + 64).1.max(n + 1) + 20;
        let full = want_bbp::<Ln2S<A, B>>(6, 0, end, 0, len);
        let part = if n > 0 {
            want_bbp::<Ln2S<A, B>>(6, 0, n, 0, len)
        } else {
            zero_want(len)
        };
        assert_tail(&full, &part, b, len, &format!("ln2 {A}n + {B} n {n}"));
    }
    for n in [0, 1, 2, 3, 7, 31, 32, 100] {
        ln2::<6, 1>(n);
        ln2::<6, 5>(n);
        ln2::<2, 1>(n);
        ln2::<1, 1>(n);
    }
    // Chudnovsky's series at shift 15, signed.
    let ctx = Hyper::<Pi>::new(&[1], 15).unwrap();
    let dctx = DynHyperCtx::new_shift(15);
    for n in [1, 2, 3, 10, 11, 50] {
        let b = ctx.tail_bits(n);
        let len = tail_len(b);
        let end = ctx.range(64 * len as u64 + 64).1.max(n + 1) + 5;
        let full = want_hyper::<Pi>(&dctx, 1, end, 0, len);
        let part = if n > 1 {
            want_hyper::<Pi>(&dctx, 1, n, 0, len)
        } else {
            zero_want(len)
        };
        assert_tail(&full, &part, b, len, &format!("pi n {n}"));
    }
}

#[test]
fn test_typed_ctx_zeros_and_errors() {
    for t in [1, 2, 63, 64, 65, 128, 129, 1000] {
        let zeros = ((t - 1) / 64) as usize;
        assert_eq!(Engel::<Exp>::new(t).zeros(), zeros, "exp t {t}");
        let x = [5];
        assert_eq!(
            Hyper::<Exp>::new(&x, t + 3).unwrap().zeros(),
            zeros,
            "exp hyper t {t}"
        );
    }
    let err = |x: &[u64], shift| Hyper::<Exp>::new(x, shift).err();
    assert_eq!(err(&[], 5), Some(HyperCtxError::NotTrimmed));
    assert_eq!(err(&[0], 5), Some(HyperCtxError::NotTrimmed));
    assert_eq!(err(&[3, 0], 5), Some(HyperCtxError::NotTrimmed));
    assert_eq!(
        err(&[3], 2),
        Some(HyperCtxError::ShiftTooSmall { min_shift: 3 })
    );
    assert_eq!(
        err(&[4], 2),
        Some(HyperCtxError::ShiftTooSmall { min_shift: 3 })
    );
    assert_eq!(
        err(&[1], 0),
        Some(HyperCtxError::ShiftTooSmall { min_shift: 1 })
    );
    assert_eq!(
        err(&[0, 1], 64),
        Some(HyperCtxError::ShiftTooSmall { min_shift: 65 })
    );
    assert_eq!(
        err(&[1, 1], 65),
        Some(HyperCtxError::ShiftTooSmall { min_shift: 66 })
    );
    for (x, shift) in [
        (&[4][..], 3),
        (&[1], 1),
        (&[0, 1], 65),
        (&[1, 1], 66),
        (&[3], 3),
    ] {
        assert_eq!(err(x, shift), None, "x {x:?} shift {shift}");
    }
}

#[test]
#[should_panic(expected = "doesn't reach")]
fn test_typed_range_caps_a_weak_bound() {
    Bbp::<BbpLin>::new(3).range(64);
}

// An x below 2^width with its top and bottom bits set, as exp's and ln's chunks are.
fn odd_x(width: u64, rng: &mut StdRng) -> Vec<u64> {
    let mut x = rand_x(width, rng);
    x[0] |= 1;
    x
}

// got is within units of want's last limb.
fn assert_within(got: &[u64], want: &[u64], units: u64, what: &str) {
    let (mut hi, lo) = match cmp_buf(got, want) {
        Ordering::Less => (want.to_vec(), got),
        _ => (got.to_vec(), want),
    };
    sub_buf(&mut hi, lo);
    assert!(
        buf_len(&hi) <= 1 && hi.first().map_or(true, |&l| l <= units),
        "{what}: off by more than {units}\n got {got:x?}\nwant {want:x?}"
    );
}

// The default entries against a long exact sum: within a unit for the sum, under one more for
// its tail.
#[test]
fn test_typed_entries_match_exact_sum() {
    fn check<const N: usize, C: Series>(ctx: &C, want: impl Fn(usize, usize) -> Want, what: &str) {
        for len in [1, 2, 3, 5, 9, 30] {
            let what = format!("{what} len {len}");
            let (mut got, zeros) = (vec![0; len], ctx.zeros());
            let w = want(len, zeros);
            let (neg, z) = series_eval_dyn(ctx, &mut got);
            assert_eq!(z, zeros, "{what}");
            assert!(neg == w.1 || got.iter().all(|&l| l == 0), "{what}: sign");
            assert_within(&got, &w.0, 2, &format!("{what} dyn"));
            let (a, b) = ctx.range(64 * (len + zeros) as u64);
            let ok = C::Dyn::eligible(&ctx.dyn_ctx(), a, b, len, zeros);
            for alg in ALGS.into_iter().filter(|&alg| ok.allows(alg)) {
                series_eval_dyn_at(ctx, &mut got, alg);
                assert_within(&got, &w.0, 2, &format!("{what} dyn {alg:?}"));
            }
            if len + 2 <= N {
                let mut sgot = vec![0; len];
                let r = series_eval_static::<N, C>(ctx, &mut sgot);
                assert_eq!(r.map(|r| r.1), Ok(zeros), "{what}");
                assert_within(&sgot, &w.0, 2, &format!("{what} static"));
                assert_within(&sgot, &got, 2, &format!("{what} static vs dyn"));
            }
        }
    }
    // The exact sum far past where the bound stops, at len + zeros limbs.
    fn far<C: Series>(ctx: &C, len: usize, zeros: usize) -> u64 {
        ctx.range(64 * (len + zeros) as u64 + 256).1 + 10
    }
    let mut rng = StdRng::seed_from_u64(96);
    for k in [1, 3, 64, 65, 200] {
        let c = Engel::<Exp>::new(k);
        let want = |len, z| want_engel::<Exp>(k, 1, far(&c, len, z), z, len);
        check::<34, _>(&c, want, &format!("exp k {k}"));
        let c = Bbp::<ATanh>::new(k);
        let want = |len, z| want_bbp::<ATanh>(k, 1, far(&c, len, z), z, len);
        check::<34, _>(&c, want, &format!("atanh k {k}"));
        for xb in [1, 64, 130] {
            let x = odd_x(xb, &mut rng);
            let d = DynHyperCtx::new(&x, xb + k);
            let c = Hyper::<Exp>::new(&x, xb + k).unwrap();
            let want = |len, z| want_hyper::<Exp>(&d, 1, far(&c, len, z), z, len);
            check::<34, _>(&c, want, &format!("exp k {k} x {xb} bits"));
            let c = Hyper::<ATanh>::new(&x, xb + k).unwrap();
            let want = |len, z| want_hyper::<ATanh>(&d, 1, far(&c, len, z), z, len);
            check::<34, _>(&c, want, &format!("atanh k {k} x {xb} bits"));
        }
    }
    let c = Hyper::<Pi>::new(&[1], 15).unwrap();
    let d = DynHyperCtx::new_shift(15);
    let want = |len, z| want_hyper::<Pi>(&d, 1, far(&c, len, z), z, len);
    check::<34, _>(&c, want, "pi");
}

#[test]
fn test_typed_static_capacity() {
    let mut out = [u64::MAX; 7];
    let r = series_eval_static::<8, _>(&Engel::<Exp>::new(5), &mut out);
    assert_eq!(r, Err(crate::utils::CapacityError { min_n: 9 }));
    assert_eq!(out, [u64::MAX; 7], "a capacity error leaves out alone");
    let x = [3; 9];
    let r = series_eval_static::<8, _>(&Hyper::<Exp>::new(&x, 600).unwrap(), &mut out[..2]);
    assert_eq!(r, Err(crate::utils::CapacityError { min_n: 9 }));
    let r = series_eval_static::<8, _>(&Hyper::<Exp>::new(&x, 600).unwrap(), &mut out);
    assert_eq!(r, Err(crate::utils::CapacityError { min_n: 9 }));
    assert_eq!(out, [u64::MAX; 7]);
    let r = series_eval_static::<8, _>(&Hyper::<Exp>::new(&x[..8], 600).unwrap(), &mut out[..6]);
    assert!(r.is_ok());
    assert_eq!(
        series_eval_dyn(&Engel::<Exp>::new(130), &mut []),
        (false, 2)
    );
    assert_eq!(
        series_eval_static::<4, _>(&Engel::<Exp>::new(130), &mut []),
        Ok((false, 2))
    );
}

// The series section's random-knob check (test_tuning.rs): series_eval_range's own pick under the
// live cost model and Horner cutoffs, and every forced algorithm, against the exact sum, for BBP,
// Engel and Hyper ranges (x of 1 to 256 bits), dynamic and static.
#[cfg(feature = "tune")]
pub(super) fn random_knob_check(rng: &mut StdRng) {
    for _ in 0..4 {
        let k = [1u64, 3, 8, 63, 65, 130][rng.gen_range(0..6)];
        let (len, t) = (rng.gen_range(1..=40), rng.gen_range(1..=300));
        let z = ((k - 1) / 64) as usize;
        let what = format!("random-knob k {k} len {len} t {t}");
        let want = want_bbp::<BbpQuad>(k, 1, 1 + t, z, len);
        check_algs::<DynNodeBBP<BbpQuad>>(&Shift(k), (1, 1 + t), z, &want, &what);
        let want = want_engel::<Exp>(k, 1, 1 + t, z, len);
        check_algs::<DynNodeEngel<Exp>>(&Shift(k), (1, 1 + t), z, &want, &what);
        let bits = [1u64, 40, 64, 130, 256][rng.gen_range(0..5)];
        let x = rand_x(bits, rng);
        let ctx = DynHyperCtx::new(&x, bits + k);
        let want = want_hyper::<Exp>(&ctx, 1, 1 + t, 0, len);
        check_algs::<DynNodeHyper<Exp>>(&ctx, (1, 1 + t), 0, &want, &format!("{what} x {bits}"));
        if len + 4 <= 64 {
            let want = want_bbp::<BbpQuad>(k, 1, 1 + t, z, len);
            check_algs::<StaticNodeBBP<64, BbpQuad>>(&Shift(k), (1, 1 + t), z, &want, &what);
            let sctx = StaticHyperCtx::<64>::new(&x, bits + k);
            let want = want_hyper::<Exp>(&ctx, 1, 1 + t, 0, len);
            check_algs::<StaticNodeHyper<64, Exp>>(&sctx, (1, 1 + t), 0, &want, &what);
        }
    }
}
