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
    for stat in [false, true] {
        let near = |a: (usize, usize), b: (usize, usize), grow: f64| {
            let r = mul_work(b.0, b.1, stat) / mul_work(a.0, a.1, stat);
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
    let works = [costs.horner, costs.rect.map(|r| r.1), Some(costs.bin_split)];
    for w in works.into_iter().flatten() {
        let units = [w.div, w.scalar, w.product, w.term, w.call, w.alloc, w.fill];
        assert!(
            units.iter().all(|u| u.is_finite() && *u >= 0.0),
            "{what}: {w:?}"
        );
        assert!(w.ns().is_finite() && w.ns() > 0.0, "{what}: {w:?}");
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
                assert_eq!(N::dispatch(ctx, a, b, len, zeros), costs.pick(), "{what}");
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
        let ctx = DynHyperCtx::new_shift(k);
        check_node_costs::<DynNodeHyper<Hyp>>(&ctx, u64::MAX, "hyper x = 1");
        check_node_costs::<DynNodeHyper<RBig>>(&ctx, u64::MAX, "hyper r > q");
        check_node_costs::<DynNodeHyper<WideTerms<true>>>(&ctx, u64::MAX, "wide hyper");
    }
    for xl in [1, 2, 3, 256] {
        let x = rand_x(64 * xl as u64, &mut rng);
        for k in [64 * xl as u64, 64 * xl as u64 + 7, 64 * xl as u64 + 4000] {
            let ctx = DynHyperCtx::new(&x, k);
            check_node_costs::<DynNodeHyper<Exp>>(&ctx, u64::MAX, "hyper x");
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
    // More terms never make Horner cheaper: they add frame limbs, divisions and steps.
    let mut rng = StdRng::seed_from_u64(92);
    let x = rand_x(192, &mut rng);
    let (shift, hyper_x) = (Shift(3), DynHyperCtx::new(&x, 200));
    for len in [1, 8, 100] {
        let mut last = [0.0; 4];
        for b in 2..600 {
            let ns = [
                DynNodeBBP::<BbpQuad>::costs(&shift, 1, b, len, 0, None).horner,
                DynNodeEngel::<Exp>::costs(&shift, 1, b, len, 0, None).horner,
                DynNodeHyper::<Hyp>::costs(&DynHyperCtx::new_shift(3), 1, b, len, 0, None).horner,
                DynNodeHyper::<Exp>::costs(&hyper_x, 1, b, len, 0, None).horner,
            ]
            .map(|w| w.unwrap().ns());
            for (i, (&now, before)) in ns.iter().zip(&mut last).enumerate() {
                assert!(now >= *before, "kind {i} len {len} b {b}: {now} < {before}");
                *before = now;
            }
        }
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
