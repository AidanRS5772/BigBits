use super::exact_fraction;
use crate::utils::series::*;
use crate::utils::trans::{exp_terms, ln_terms, ATanh, Exp};
use crate::utils::utils::{buf_len, cmp_buf, sub_buf};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;

const ALGS: [SeriesAlg; 3] = [SeriesAlg::Horner, SeriesAlg::Rect, SeriesAlg::BinSplit];

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

fn mag(n: u64) -> u64 {
    n.wrapping_mul(7919) % 1000 + 1
}

fn negative(n: u64) -> bool {
    n.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 63 == 1
}

// Signed series whose partial sums change sign, with |S| < 1 from k >= 3.
struct SgBbp;
impl BBPSeries for SgBbp {
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
    fn p(_: u128) -> u128 {
        1
    }
    fn q(n: u128) -> u128 {
        (1 << 70) + n
    }
}

struct RBig;
impl HyperSeries for RBig {
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
    let neg = series_eval::<N>(ctx, a, b, zeros, &mut out);
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

// exp's and ln's own series shapes, as trans.rs's `series` sums them.
#[test]
fn test_series_eval_exp_ln_shapes() {
    let mut rng = StdRng::seed_from_u64(14);
    for w in [1, 2, 4, 8, 16, 40] {
        // exp's top bit, an Engel series at its exact shift.
        for k in [1, 2, 25, 64, 65, 100] {
            let (n, zeros) = (exp_terms(k, w), ((k - 1) / 64) as usize);
            if n == 0 || zeros >= w {
                continue;
            }
            let what = format!("exp top w {w} k {k}");
            let want = want_engel::<Exp>(k, 1, n + 1, zeros, w - zeros);
            check_algs::<DynNodeEngel<Exp>>(&Shift(k), (1, n + 1), zeros, &want, &what);
        }
        // exp's chunks: width bits at offset t, a Hyper series in x 2^-(t + width).
        for (t, width) in [(1, 16), (17, 32), (49, 64), (113, 128), (241, 256)] {
            let (n, zeros) = (exp_terms(t, w), ((t - 1) / 64) as usize);
            if n == 0 || zeros >= w {
                continue;
            }
            let x = rand_x(width, &mut rng);
            let ctx = DynHyperCtx::new(&x, t + width);
            let what = format!("exp chunk w {w} t {t} width {width}");
            let want = want_hyper::<Exp>(&ctx, 1, n + 1, zeros, w - zeros);
            check_algs::<DynNodeHyper<Exp>>(&ctx, (1, n + 1), zeros, &want, &what);
        }
        // ln's top chunk, BBP ATanh at shift 2 lead.
        for lead in [1, 3, 20, 40, 70] {
            let (n, zeros) = (ln_terms(2 * lead, w), ((2 * lead + 1) / 64) as usize);
            if n == 0 || zeros >= w {
                continue;
            }
            let what = format!("ln top w {w} lead {lead}");
            let want = want_bbp::<ATanh>(2 * lead, 1, n + 1, zeros, w - zeros);
            check_algs::<DynNodeBBP<ATanh>>(&Shift(2 * lead), (1, n + 1), zeros, &want, &what);
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
    fn p(_: u128) -> u128 {
        u128::MAX
    }
    fn q(_: u128) -> u128 {
        1
    }
}
impl WideEngelSeries for WideNumerator {
    fn p(_: u128) -> u128 {
        u128::MAX
    }
    fn q(_: u128) -> u128 {
        1
    }
}
impl WideHyperSeries for WideNumerator {
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
