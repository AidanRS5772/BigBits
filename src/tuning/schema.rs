// Every performance knob, declared once (docs/tuning_system.md, "The schema"). build.rs reads this
// file to generate the accessors (`k::<section>::<knob>()`), the `resolved::` constants and the
// live store; the crate reads it at run time for `tuning::get`/`set` and the tuner. Shared by
// build.rs and the crate, so it uses only `super::schema_types`.
//
// Phase 1 declares each section's old `mod.rs` constants as `Interim` knobs, under their own
// names, with today's values; the per-section checklist in docs/tuning_system.md says how. Keep
// sections in derive order (base, mul, div, sqrt, bin_split, series, consts, theta, agm, exp, ln).

use super::schema_types::{Alg, DefaultValue, Derivation, Kind, Knob, Range, Section, Threads, Ty};

// mul.kara's algorithms: schoolbook, and Karatsuba (chunking for s <= ceil(l / 2), a level above).
// Karatsuba needs s >= 4: on 3 x 3 a level's cross product is 3 x 3 again.
// series.small's picks, in `utils::series::SMALL_PICKS`' order: the last hands the range to the
// equations.
const SERIES_SMALL_ALGS: &[Alg] =
    &[Alg { name: "horner", min: 0 }, Alg { name: "rect", min: 0 }, Alg { name: "bin_split", min: 0 }, Alg { name: "model", min: 0 }];

const KARA_ALGS: &[Alg] = &[Alg { name: "school", min: 1 }, Alg { name: "karatsuba", min: 4 }];

// mul.kara's default: today's is_school_at(l, s, 63, 52.0) exactly, as change points over (s, l)
// (rows s = 1..64; every row from 64 on is Karatsuba). test_kara_default_is_todays_boundary checks it
// at every shape.
const KARA_DEFAULT: &[u32] = &[
    1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 5, 5, 0, 6, 6, 0, 7, 7, 0, 8, 8, 0,
    9, 9, 0, 10, 10, 0, 11, 11, 0, 12, 12, 0, 13, 13, 0, 14, 14, 0, 15, 15, 0, 16, 16, 0,
    17, 17, 0, 18, 18, 0, 19, 19, 0, 20, 20, 0, 21, 21, 0, 22, 22, 0, 23, 23, 0, 24, 24, 0,
    25, 25, 0, 26, 26, 0, 27, 27, 0, 28, 28, 0, 29, 29, 0, 30, 30, 0, 31, 31, 0, 32, 32, 0,
    33, 33, 0, 34, 34, 0, 35, 35, 0, 36, 36, 0, 37, 37, 0, 38, 38, 0, 39, 39, 0, 40, 40, 0,
    41, 41, 0, 42, 42, 0, 43, 43, 0, 44, 44, 0, 45, 45, 0, 46, 46, 0, 47, 47, 0, 48, 48, 0,
    49, 49, 0, 50, 50, 0, 51, 51, 0, 52, 52, 0, 53, 53, 0, 53, 104, 1, 53, 105, 0, 54, 54, 1,
    54, 55, 0, 54, 102, 1, 54, 107, 0, 55, 55, 1, 55, 57, 0, 55, 100, 1, 55, 109, 0, 56, 56, 1,
    56, 59, 0, 56, 98, 1, 56, 111, 0, 57, 57, 1, 57, 62, 0, 57, 95, 1, 57, 113, 0, 58, 58, 1,
    58, 66, 0, 58, 91, 1, 58, 115, 0, 59, 59, 1, 59, 71, 0, 59, 86, 1, 59, 117, 0, 60, 60, 1,
    60, 119, 0, 61, 61, 1, 61, 121, 0, 62, 62, 1, 62, 123, 0, 63, 63, 1, 63, 125, 0, 64, 64, 1,
];

pub const SCHEMA: &[Section] = &[
    Section {
        name: "base",
        version: 4,
        derive: 1,
        after: &[],
        doc: "The x86 ADX loops' crossovers. (Its cost units, pass, call, fill and the rest, were \
              removed 2026-11-10: nothing a dispatcher depends on read them.)",
        knobs: &[
            Knob::usize("prim_adx", "MUL_PRIM_ADX_CUTOFF", 20).arch("x86_64").kind(Kind::Sweep).doc(
                "Limbs from which mul_add_prim takes its mulx/adcx loop rather than the plain \
                 one, behind the ADX feature check (x86_64).",
            ),
            Knob::usize("limb_adx", "MUL_LIMB_ADX_CUTOFF", 10).arch("x86_64").kind(Kind::Sweep).doc(
                "Limbs from which add_mul_limb and sub_mul_limb take the ADX rows rather than \
                 their one-limb loops (x86_64).",
            ),
        ],
    },
    Section {
        name: "mul",
        version: 17,
        derive: 10,
        after: &["base"],
        doc: "Multiplication: schoolbook, Karatsuba, FFT and NTT for products, squares, high and \
              middle products and squares mod B^m - 1, and the NTT's parallel gates.",
        knobs: &[
            // Karatsuba's schoolbook boundary (Phase 2's pilot, derived from the models below).
            Knob::new(
                "kara",
                "",
                Ty::Table { algs: KARA_ALGS, measure: "short limbs s, long limbs l", inputs: 2 },
                DefaultValue::Table(KARA_DEFAULT),
            )
            .kind(Kind::Derived(Derivation::Table))
            .doc(
                "Schoolbook or Karatsuba for an l x s product (l >= s), by change points over \
                 (s, l): Karatsuba's sub-products, its scratch sizing and the product dispatchers' \
                 schoolbook tier all read it. Derived by comparing mul_buf with one Karatsuba level \
                 over schoolbook leaves (cargo tune derive). The default is the old is_school \
                 boundary (chunking_karatsuba 63, karatsuba 52.0).",
            ),
            // Leaf kernels' cost models (Phase 2's pilot): t = scale W + shift, in ns.
            Knob::f64("school_scale", "SERIES_NS_PRODUCT", 0.3567)
                .kind(Kind::Model { work: "mul_buf: l s limb products" })
                .doc("ns per limb product of mul_buf."),
            Knob::f64("school_shift", "", 20.4)
                .kind(Kind::Model { work: "mul_buf's fixed cost" })
                .doc("ns of mul_buf's fixed cost."),
            // A Karatsuba level's fitted form (decided 2026-11-09: a mid-range composite keeps its
            // theoretical form, with constants fitted to its own measured times, where assembling it
            // from base units misses its real overhead).
            Knob::f64("kara_scale", "", 1.039)
                .kind(Kind::Model { work: "karatsuba_core: its three sub-products' predicted time" })
                .doc("A Karatsuba level's time per ns of its three sub-products' predictions."),
            Knob::f64("kara_shift", "", 68.67)
                .kind(Kind::Model { work: "karatsuba_core's fixed cost" })
                .doc("ns of a Karatsuba level's fixed cost."),
            // Each fit's relative rms error. A derivation acts on a predicted difference only past
            // the combined error of the two models it compares, so a model never decides what it
            // can't resolve.
            Knob::f64("school_err", "", 0.04)
                .float_range(0.0, 1.0)
                .kind(Kind::Model { work: "the mul_buf fit's relative rms residual" })
                .doc("Relative rms error of mul_buf's model."),
            Knob::f64("kara_err", "", 0.02)
                .float_range(0.0, 1.0)
                .kind(Kind::Model { work: "the Karatsuba level fit's relative rms residual" })
                .doc("Relative rms error of the Karatsuba level's model."),
            Knob::f64("chunk_scale", "", 0.8506)
                .kind(Kind::Model { work: "chunking_karatsuba over schoolbook products: its chunk products' fitted schoolbook time" })
                .doc("Chunking Karatsuba's fitted form, scale (chunk products) + pass (l + s) + shift: ns per ns of its chunk products."),
            Knob::f64("chunk_pass", "", 2.532)
                .kind(Kind::Model { work: "chunking_karatsuba: l + s limbs (clearing out, adding each chunk in)" })
                .doc("Its ns per operand limb."),
            Knob::f64("chunk_err", "", 0.03)
                .float_range(0.0, 1.0)
                .kind(Kind::Model { work: "that fit's relative rms residual" })
                .doc("Its fit's relative rms error."),
            // Recursive Karatsuba and the FFT entries in their fitted forms, each fitted to its own
            // entry's times. mul_dyn's dispatch compares the product ones per call (`MulEq`:
            // kara_rec_*, chunk_rec_* and fft_*; since 2026-11-11), so their defaults matter on a
            // target with no preset: Karatsuba's are the M1's fit (2026-11-11, under the form with
            // a schoolbook weight), and fft_scale is set so the two cross about where the old
            // x86-tuned ratio (1.96) put the crossing, near 230 limbs.
            Knob::f64("kara_rec_scale", "", 1.838)
                .kind(Kind::Model { work: "karatsuba_entry_dyn, s above half of l: karatsuba_cost(l, s) = l^1.585 - (l - s)^1.585" })
                .doc("Recursive Karatsuba's fitted form (karatsuba_entry_dyn, s above half of l): ns per unit of karatsuba_cost(l, s) = l^1.585 - (l - s)^1.585."),
            Knob::f64("kara_rec_school", "", 0.02337)
                .kind(Kind::Model { work: "karatsuba_entry_dyn, s above half of l: l s limb products" })
                .doc(
                    "Its ns per limb product l s: schoolbook's share of the time, which the \
                     recursion's leaves still do. With the scale it stands where a fitted exponent \
                     stood until 2026-11-11 (the user's decision: every fit linear in its constants).",
                ),
            Knob::f64("kara_rec_err", "", 0.03)
                .float_range(0.0, 1.0)
                .kind(Kind::Model { work: "that fit's relative rms residual" })
                .doc("Its fit's relative rms error."),
            Knob::f64("chunk_rec_scale", "", 1.856)
                .kind(Kind::Model { work: "karatsuba_entry_dyn, s at most half of l (chunking): chunking_karatsuba_cost(l, s) = l s^0.585" })
                .doc("Recursive Karatsuba's fitted form (karatsuba_entry_dyn, s at most half of l (chunking)): ns per unit of chunking_karatsuba_cost(l, s) = l s^0.585."),
            Knob::f64("chunk_rec_school", "", 0.0275)
                .kind(Kind::Model { work: "karatsuba_entry_dyn, s at most half of l (chunking): l s limb products" })
                .doc("Its ns per limb product l s: schoolbook's share of the time, as mul.kara_rec_school."),
            Knob::f64("chunk_rec_err", "", 0.03)
                .float_range(0.0, 1.0)
                .kind(Kind::Model { work: "that fit's relative rms residual" })
                .doc("Its fit's relative rms error."),
            Knob::f64("fft_scale", "", 0.49)
                .kind(Kind::Model { work: "fft_entry: fft_transform_work(N) = rho(o) N log2 N at its transform size N" })
                .doc("fft_entry's ns per unit of transform work."),
            Knob::f64("fft_err", "", 0.03)
                .float_range(0.0, 1.0)
                .kind(Kind::Model { work: "that fit's relative rms residual" })
                .doc("fft_entry's fit's relative rms error."),
            // The old boundary's two parameters, which now only scale the static series model's
            // product work, and is_karatsuba's ratios against the NTT: the static products'
            // dispatch, swept at N = 8192. (The ratios against the FFT, which no decision had read
            // since dynamic products took fitted equations, were purged on 2026-11-11.)
            Knob::usize("chunking_karatsuba", "CHUNKING_KARATSUBA_CUTOFF", 63).doc(
                "The old is_school's short length for s <= half. Dispatch reads mul.kara now; this \
                 only scales mul_work's chunking tier (the series cost model) until mul_ns \
                 replaces mul_work.",
            ),
            Knob::f64("karatsuba", "KARATSUBA_CUTOFF", 52.0).float_range(3.0, f64::MAX).doc(
                "The old is_school's parabola parameter c (schoolbook up to ceil(1.5 l - l^2 / \
                 (2 c)) for s > half). Dispatch reads mul.kara now; this only scales mul_work's \
                 Karatsuba tier until mul_ns replaces it. At least 3, as is_school_at with a \
                 smaller c would recurse on 3 x 3.",
            ),
            Knob::f64("ntt_chunking_karatsuba", "NTT_CHUNKING_KARATSUBA_CUTOFF", 5.9)
                .kind(Kind::Sweep)
                .threads(Threads::Pool)
                .doc(
                    "is_karatsuba's ratio for chunking Karatsuba against ntt_entry_static \
                     (mul_static). A sweep like mul.fft_chunking_karatsuba, at the tuner's pool size.",
                ),
            Knob::f64("ntt_karatsuba", "NTT_KARATSUBA_CUTOFF", 5.7)
                .kind(Kind::Sweep)
                .threads(Threads::Pool)
                .doc(
                    "is_karatsuba's ratio for Karatsuba against ntt_entry_static (mul_static). \
                     A sweep like mul.fft_karatsuba, at the tuner's pool size.",
                ),
            Knob::usize("ntt_par_conv", "NTT_PAR_CUTOFF_NTT_CONV", 512).threads(Threads::Pool).kind(Kind::Sweep)
                .doc("Transform length from which the NTT runs its three primes' convolutions in parallel."),
            Knob::usize("ntt_par", "NTT_PAR_CUTOFF_NTT", 3600).threads(Threads::Pool).kind(Kind::Sweep)
                .doc("Transform length from which a convolution runs its two forward transforms in parallel."),
            Knob::usize("ntt_par_radix3", "NTT_PAR_CUTOFF_NTT_3", 27648).threads(Threads::Pool).kind(Kind::Sweep).doc(
                "Length from which a radix-3 step runs its sub-transforms in parallel; scratch \
                 sizing reads it too.",
            ),
            Knob::usize("ntt_par_radix5", "NTT_PAR_CUTOFF_NTT_5", 25600).threads(Threads::Pool).kind(Kind::Sweep).doc(
                "Length from which a radix-5 step runs its sub-transforms in parallel; scratch \
                 sizing reads it too.",
            ),
            Knob::usize("karatsuba_sqr", "KARATSUBA_SQR_CUTOFF", 71)
                .min(5)
                .kind(Kind::Sweep)
                .doc(
                    "sqr_buf up to this many limbs, a Karatsuba square level above (a one-input \
                     table with one change point). A sweep: sqr_buf against a level over sqr_buf \
                     leaves at every size from 40 to 140, as the two are tied over a wide range. At least 5: a Karatsuba square level needs 6 limbs.",
                ),
            Knob::usize("fft_sqr", "FFT_SQR_CUTOFF", 340)
                .kind(Kind::Sweep)
                .doc(
                    "sqr_dyn squares by Karatsuba up to this many limbs and by the FFT above. \
                     A sweep: the single cutoff with the least measured time lost (the FFT's time \
                     steps with its transform size, so the two cross more than once).",
                ),
            Knob::usize("static_ntt_sqr", "STATIC_NTT_SQR_CUTOFF", 3884)
                .kind(Kind::Sweep)
                .threads(Threads::Pool)
                .doc(
                    "sqr_static squares by Karatsuba up to this many limbs and by the NTT above. \
                     A sweep like mul.fft_sqr, at the tuner's pool size.",
                ),
            Knob::usize("dyn_partial_mul", "DYN_PARTIAL_MUL_CUTOFF", 312)
                .kind(Kind::Sweep)
                .doc("hi_mul_dyn takes hi_mul_buf for windows up to this many limbs, the guarded product above."),
            Knob::usize("static_partial_mul", "STATIC_PARTIAL_MUL_CUTOFF", 408)
                .kind(Kind::Sweep)
                .doc("hi_mul_static's counterpart of dyn_partial_mul."),
            Knob::usize("dyn_partial_sqr", "DYN_PARTIAL_SQR_CUTOFF", 464)
                .kind(Kind::Sweep)
                .doc("hi_sqr_dyn's counterpart of dyn_partial_mul."),
            Knob::usize("static_partial_sqr", "STATIC_PARTIAL_SQR_CUTOFF", 640)
                .kind(Kind::Sweep)
                .doc("hi_sqr_static's counterpart of dyn_partial_mul."),
            Knob::usize("hi_mul_row", "HI_MUL_ROW_CUTOFF", 24)
                .kind(Kind::Sweep)
                .doc("hi_mul_buf uses columns for windows up to this many limbs, rows above."),
            Knob::usize("hi_sqr_row", "HI_SQR_ROW_CUTOFF", 90)
                .kind(Kind::Sweep)
                .doc("hi_sqr_buf uses columns for windows up to this many limbs, rows above."),
            Knob::usize("fft_mid", "FFT_MID_CUTOFF", 125)
                .kind(Kind::Sweep)
                .doc("mid_mul_dyn uses mid_mul_buf below this short length and fft_mid_mul from it."),
            Knob::usize("ntt_mid", "NTT_MID_CUTOFF", 672).threads(Threads::Pool).kind(Kind::Sweep).doc(
                "mid_mul_static uses mid_mul_buf below this short length and ntt_mid_mul_static \
                 from it.",
            ),
            Knob::usize("sqr_bnm1_fft", "DYN_SQR_BNM1_FFT_CUTOFF", 0).follows("mul.fft_sqr", 0).kind(Kind::Sweep).doc(
                "sqr_mod_bnm1_dyn runs its cyclic FFT on operands above this many limbs. GUESS: a \
                 full square's FFT crossover (it follows mul.fft_sqr); the wrapped transform is \
                 half as long, so the true crossover is probably lower.",
            ),
        ],
    },
    Section {
        name: "div",
        version: 8,
        derive: 1,
        after: &["base", "mul"],
        doc: "Division and reciprocals: Knuth, Burnikel-Ziegler and Newton-Raphson, with \
              separate BZ-vs-NR cost models for dyn/static and div/div_rem, and BZ's top block.",
        knobs: &[
            Knob::usize("knuth", "DIV_KNUTH_CUTOFF", 68).doc(
                "Divisor width up to which every division (and every one-limb quotient) is Knuth; \
                 NR and BZ become eligible above it.",
            ),
            Knob::usize("bz", "BZ_CUTOFF", 140).min(3).kind(Kind::Sweep).doc(
                "BZ's recursive Knuth leaf: blocks with divisors of at most this many limbs are \
                 Knuth, and a non-NR choice below it skips BZ's block setup. At least 3: the 2/1 \
                 step halves the divisor, and Knuth needs at least 2 limbs (found by the \
                 random-knob test).",
            ),
            Knob::f64("bz_top_padded_cost_scale", "BZ_TOP_PADDED_COST_SCALE", 1.80).kind(Kind::Sweep).doc(
                "Static BZ's top block runs the padded 2/1 block when this times bz_2_1_cost is at \
                 most the Knuth top block's work (use_bz_for_top_block). Dynamic BZ compares its \
                 fitted equations instead (DivEq::bz_top_padded) since 2026-11-11; this and \
                 bz_2_1_cost's two scales are the static rule's, approximate: the scale is swept \
                 on dynamic BZ's forced top-block timings, not on static divisions.",
            ),
            Knob::f64("bz_2_1_school_to_karatsuba", "BZ_2_1_SCHOOL_TO_KARATSUBA", 3.394147384)
                .doc("bz_2_1_cost's scale on d^log2(3) where the block's products are Karatsuba."),
            Knob::f64("bz_2_1_karatsuba_to_fft", "BZ_2_1_KARATSUBA_TO_FFT", 2.733850808)
                .doc("bz_2_1_cost's scale on d log2^2 d where the block's products are transforms."),
            Knob::usize("dyn_knuth_d", "", 32)
                .kind(Kind::Derived(Derivation::Extent))
                .doc(
                    "div_dyn takes Knuth without reading its equations at divisors up to this width: \
                     the widest at which they pick Knuth for every quotient. Computed by the tuner \
                     from the fitted equations (it times nothing), as mul.kara is for small products.",
                ),
            Knob::usize("dyn_rem_knuth_d", "", 64)
                .kind(Kind::Derived(Derivation::Extent))
                .doc("div_rem_dyn's counterpart of dyn_knuth_d."),
            // The dynamic division algorithms' fitted runtime equations (DivEq), compared per call by
            // div_dyn and div_rem_dyn (2026-11-11). Their defaults are the M1's fit, so a target
            // with no preset dispatches by them until it is tuned.
            Knob::f64("knuth_ns_qd", "", 0.405)
                .kind(Kind::Model { work: "Knuth division (div_prepared_dyn forced): q d limb products" })
                .doc("Knuth's ns per quotient limb times divisor limb."),
            Knob::f64("knuth_ns_q", "", 9.5)
                .kind(Kind::Model { work: "Knuth division: q quotient digits" })
                .doc("Knuth's ns per quotient limb."),
            Knob::f64("knuth_ns_d", "", 0.44)
                .kind(Kind::Model { work: "Knuth division: d divisor limbs" })
                .doc("Knuth's ns per divisor limb."),
            Knob::f64("knuth_ns_shift", "", 22.0)
                .kind(Kind::Model { work: "Knuth division: fixed cost" })
                .doc("Knuth's fixed cost in ns."),
            Knob::f64("knuth_ns_err", "", 0.03)
                .float_range(0.0, 1.0)
                .kind(Kind::Model { work: "that fit's relative rms residual" })
                .doc("Knuth's fit's relative rms error."),
            // NR's products are priced one by one, by tier (DivEq::nr_parts), since 2026-11-10: a
            // multiple of M(h, h), a full product's price, fitted at 4.2-4.7%. The defaults here
            // are that form's offline fit to the M1's timings (1.7%; 2.6% held out), not yet a
            // tuner fit.
            Knob::f64("nr_ns_mid_school", "", 2.329)
                .kind(Kind::Model { work: "NR division, quotient only: its schoolbook middle products, each schoolbook's n x n time (short side below mul.fft_mid)" })
                .doc("NR's (quotient only) multiple of its schoolbook middle products' predicted time."),
            Knob::f64("nr_ns_mid_fft", "", 1.66)
                .kind(Kind::Model { work: "NR division, quotient only: its FFT middle products, each the FFT's time at fft_mid_size(n)" })
                .doc("Its multiple of its FFT middle products' predicted time."),
            Knob::f64("nr_ns_hi_school", "", 0.4061)
                .kind(Kind::Model { work: "NR division, quotient only: its schoolbook high products, each half a schoolbook product of its window (to mul.dyn_partial_mul limbs)" })
                .doc("Its multiple of its schoolbook high products' predicted time."),
            Knob::f64("nr_ns_hi_guarded", "", 0.5862)
                .kind(Kind::Model { work: "NR division, quotient only: its guarded high products, each mul's prediction for its operands' tops" })
                .doc("Its multiple of its guarded high products' predicted time."),
            Knob::f64("nr_ns_q", "", 3.231)
                .kind(Kind::Model { work: "NR division, quotient only: q" })
                .doc("Its ns per quotient limb."),
            Knob::f64("nr_ns_err", "", 0.03)
                .float_range(0.0, 1.0)
                .kind(Kind::Model { work: "that fit's relative rms residual" })
                .doc("Its fit's relative rms error."),
            Knob::f64("nr_rem_ns_mid_school", "", 2.096)
                .kind(Kind::Model { work: "NR division with remainder: its schoolbook middle products" })
                .doc("NR's (with remainder) multiple of its schoolbook middle products' predicted time."),
            Knob::f64("nr_rem_ns_mid_fft", "", 1.476)
                .kind(Kind::Model { work: "NR division with remainder: its FFT middle products" })
                .doc("Its multiple of its FFT middle products' predicted time."),
            Knob::f64("nr_rem_ns_hi_school", "", 0.622)
                .kind(Kind::Model { work: "NR division with remainder: its schoolbook high products" })
                .doc("Its multiple of its schoolbook high products' predicted time."),
            Knob::f64("nr_rem_ns_hi_guarded", "", 0.7064)
                .kind(Kind::Model { work: "NR division with remainder: its guarded high products" })
                .doc("Its multiple of its guarded high products' predicted time."),
            Knob::f64("nr_rem_ns_prod", "", 1.004)
                .kind(Kind::Model { work: "NR division with remainder: Mul(d, min(q - 1, d + 1)), nr_rem_finish's product" })
                .doc("Its multiple of the remainder's predicted product."),
            Knob::f64("nr_rem_ns_q", "", 3.61)
                .kind(Kind::Model { work: "NR division with remainder: q" })
                .doc("Its ns per quotient limb."),
            Knob::f64("nr_rem_ns_d", "", 1.512)
                .kind(Kind::Model { work: "NR division with remainder: d" })
                .doc("Its ns per divisor limb."),
            Knob::f64("nr_rem_ns_err", "", 0.03)
                .float_range(0.0, 1.0)
                .kind(Kind::Model { work: "that fit's relative rms residual" })
                .doc("Its fit's relative rms error."),
            // BZ's three constants were predicted (2026-11-10): the M1 was throttling when the
            // equation changed, so they come from the fit with each timing divided by its shape's
            // Knuth speed (1.04, 0.99, 0; 3.1% rms). Five tuner fits on the machine at one steady
            // speed then read 1.044-1.046, 0.981-0.984 and 0 at 2.2-2.4% rms, so the predicted
            // values stand as defaults (docs/tuning_implementation.md).
            Knob::f64("bz_ns_block", "", 1.031)
                .kind(Kind::Model {
                    work: "BZ: its 2/1 blocks, each the recursion's products sum 2^k M(d / 2^k) and its Knuth leaves, and a padded top block by its quotient's significant pieces",
                })
                .doc("BZ's multiple of its 2/1 blocks' predicted time, a padded top block included."),
            Knob::f64("bz_ns_top", "", 0.9764)
                .kind(Kind::Model { work: "BZ: a Knuth top block, Knuth's equation at (top, d)" })
                .doc("BZ's multiple of a Knuth top block's predicted time."),
            Knob::f64("bz_ns_err", "", 0.031)
                .float_range(0.0, 1.0)
                .kind(Kind::Model { work: "that fit's relative rms residual" })
                .doc("BZ's fit's relative rms error."),
            Knob::usize("dyn_rcp_knuth_nr", "DYN_RCP_KNUTH_NR_CUTOFF", 36).kind(Kind::Sweep)
                .doc("rcp_dyn is Knuth up to this precision and NR above."),
            Knob::usize("static_div_karatsuba_ntt_nr_bz", "STATIC_DIV_KARATSUBA_NTT_NR_BZ_CUTOFF", 842)
                .doc("div_static's BZ-vs-NR rule (static entries keep the cutoff rule; approximate): max(q, d) from which it uses the transform ratio (NTT products)."),
            Knob::f64("static_div_karatsuba_nr_bz", "STATIC_DIV_KARATSUBA_NR_BZ_CUTOFF", 0.75)
                .doc("div_static: BZ when q times this exceeds d (Karatsuba-sized products)."),
            Knob::f64("static_div_ntt_nr_bz", "STATIC_DIV_NTT_NR_BZ_CUTOFF", 11.66)
                .doc("div_static: BZ when log2 q times this exceeds log2^2 d (transform-sized products)."),
            Knob::usize("static_div_rem_karatsuba_ntt_nr_bz", "STATIC_DIV_REM_KARATSUBA_NTT_NR_BZ_CUTOFF", 112)
                .doc("div_rem_static's counterpart of static_div_karatsuba_ntt_nr_bz."),
            Knob::f64("static_div_rem_karatsuba_nr_bz", "STATIC_DIV_REM_KARATSUBA_NR_BZ_CUTOFF", 4.4)
                .doc("div_rem_static's counterpart of static_div_karatsuba_nr_bz."),
            Knob::f64("static_div_rem_ntt_nr_bz", "STATIC_DIV_REM_NTT_NR_BZ_CUTOFF", 11.66)
                .doc("div_rem_static's counterpart of static_div_ntt_nr_bz."),
            Knob::usize("static_rcp_knuth_nr", "STATIC_RCP_KNUTH_NR_CUTOFF", 36)
                .doc("rcp_static is Knuth up to this precision and NR above."),
        ],
    },
    Section {
        name: "sqrt",
        version: 2,
        derive: 1,
        after: &["base", "mul", "div"],
        doc: "Square roots: the binomial root or Zimmermann's recursion per output mode, the \
              recursion's leaf, the wrapped remainder and the carried reciprocal.",
        knobs: &[
            Knob::usize("leaf", "ZIMMERMAN_SQRT_LEAF_CUTOFF", 11).min(3).kind(Kind::Sweep).doc(
                "Root width below which a Zimmermann stage is the binomial root: the recursion's \
                 leaf, shared by dyn and static. At least 3, as a full stage of 2 root limbs would \
                 recurse on itself.",
            ),
            Knob::usize("dyn_rem", "DYN_SQRT_REM_ZIMMERMAN_CUTOFF", 16).min(3).kind(Kind::Sweep).doc(
                "Root width from which sqrt_dyn (root and remainder) runs Zimmermann rather than \
                 the binomial root. Every top-level cutoff is at least 3: Zimmermann's first stage \
                 needs a low half, and a full x under a root of 1 or 2 limbs has none (found by \
                 the random-knob test).",
            ),
            Knob::usize("dyn_only", "DYN_SQRT_ONLY_ZIMMERMAN_CUTOFF", 11).min(3).kind(Kind::Sweep).doc(
                "Root width from which sqrt_only_dyn runs Zimmermann.",
            ),
            Knob::usize("dyn_approx", "DYN_SQRT_APPROX_ZIMMERMAN_CUTOFF", 11).min(3).kind(Kind::Sweep).doc(
                "Root width from which sqrt_approx_dyn runs Zimmermann.",
            ),
            Knob::usize("static_rem", "STATIC_SQRT_REM_ZIMMERMAN_CUTOFF", 16).min(3).kind(Kind::Sweep).doc(
                "Root width from which sqrt_static runs Zimmermann (measured with N close to \
                 x.len()).",
            ),
            Knob::usize("static_only", "STATIC_SQRT_ONLY_ZIMMERMAN_CUTOFF", 11).min(3).kind(Kind::Sweep).doc(
                "Root width from which sqrt_only_static runs Zimmermann.",
            ),
            Knob::usize("static_approx", "STATIC_SQRT_APPROX_ZIMMERMAN_CUTOFF", 11).min(3).kind(Kind::Sweep).doc(
                "Root width from which sqrt_approx_static runs Zimmermann.",
            ),
            Knob::usize("dyn_wrapped_rem", "DYN_SQRT_WRAPPED_REM_CUTOFF", 341).kind(Kind::Sweep).doc(
                "Root width from which a dynamic Zimmermann stage forms its remainder from one \
                 square mod B^m - 1 and divides quotient-only. GUESS (341). It used to follow \
                 mul.sqr_bnm1_fft + 1, but that is another decision: when the cyclic square's own \
                 cutoff was measured at 80, following it made sqrt_dyn 40% slower at 200 limbs.",
            ),
            Knob::usize("dyn_rcp_carry", "DYN_SQRT_RCP_CARRY_CUTOFF", 256).doc(
                "Root width from which a dynamic root keeps one reciprocal between its stages' \
                 divisions. Not a crossover: a reciprocal is first read two NR stages up (about \
                 680 root limbs), so smaller roots just skip its scratch acquisition.",
            ),
        ],
    },
    Section {
        name: "bin_split",
        version: 3,
        derive: 1,
        after: &["base", "mul", "div"],
        doc: "Binary splitting: where a tree stops splitting into leaves (per node kind, dyn and \
              static), and when a dynamic tree builds its halves and a merge its products in \
              parallel.",
        knobs: &[
            Knob::u64("dyn_bbp_leaf_limbs", "DYN_BBP_LEAF_LIMBS", 64).kind(Kind::Sweep)
                .doc("A dynamic BBP tree splits while a node's estimated limbs reach this."),
            Knob::u64("static_bbp_leaf_limbs", "STATIC_BBP_LEAF_LIMBS", 512).kind(Kind::Sweep)
                .doc("A static BBP tree splits while a node's estimated limbs reach this."),
            Knob::u64("dyn_engel_leaf_limbs", "DYN_ENGEL_LEAF_LIMBS", 96).kind(Kind::Sweep)
                .doc("A dynamic Engel tree splits while a node's estimated limbs reach this."),
            Knob::u64("static_engel_leaf_limbs", "STATIC_ENGEL_LEAF_LIMBS", 128).kind(Kind::Sweep)
                .doc("A static Engel tree splits while a node's estimated limbs reach this."),
            Knob::u64("dyn_hyper_leaf_limbs", "DYN_HYPER_LEAF_LIMBS", 48).kind(Kind::Sweep)
                .doc("A dynamic Hyper tree splits while a node's estimated limbs reach this."),
            Knob::u64("dyn_hyper_leaf_terms", "DYN_HYPER_LEAF_TERMS", 8).kind(Kind::Sweep).doc(
                "A dynamic Hyper tree never splits a node of this many terms or fewer (a wide x \
                 makes each leaf step a full product).",
            ),
            Knob::u64("static_hyper_leaf_limbs", "STATIC_HYPER_LEAF_LIMBS", 64)
                .doc("A static Hyper tree splits while a node's estimated limbs reach this."),
            Knob::u64("static_hyper_leaf_terms", "STATIC_HYPER_LEAF_TERMS", 8).kind(Kind::Sweep)
                .doc("A static Hyper tree never splits a node of this many terms or fewer."),
            Knob::u64("dyn_par", "DYN_BIN_SPLIT_PAR_CUTOFF", 256).kind(Kind::Sweep).threads(Threads::Pool).doc(
                "A dynamic tree builds a split's halves in parallel (on a pool of two or more \
                 threads) from this many estimated limbs; also read by the series cost model.",
            ),
            Knob::usize("dyn_merge_par", "DYN_BIN_SPLIT_MERGE_PAR_CUTOFF", 256).kind(Kind::Sweep).threads(Threads::Pool).doc(
                "A dynamic merge (on a rayon worker) runs its two products in parallel from this \
                 many limbs of its operands' Q parts.",
            ),
            Knob::u64("dyn_par_work", "", 0).kind(Kind::Sweep).threads(Threads::Pool).doc(
                "A dynamic tree started off the pool (on its caller's own thread) builds a split's \
                 halves in parallel only from this many limbs times terms at the node, as well as \
                 dyn_par limbs: the split has to wake the pool. On a pool worker dyn_par alone \
                 decides. 0 (the default) is the behaviour before 2026-11-10.",
            ),
            Knob::f64("dyn_wake_ns", "", 14000.0).kind(Kind::Model { work: "one pool wake: a parallel split started off the pool" }).doc(
                "What a parallel split started off the pool costs beyond its halves' work, in ns: \
                 a few-term tree timed with the split and without. The series cost model prices \
                 binary splitting's wake with it.",
            ),
        ],
    },
    Section {
        name: "series",
        version: 5,
        // 3: the small-series table's layout (octave frames, four q buckets, ten kinds; rows of
        // 16 to a class). A table of another layout would be misread, so it falls through.
        derive: 3,
        // The dynamic equations are built on mul's and division's; the static model's mul_work
        // follows mul's dispatch.
        after: &["base", "mul", "div", "bin_split"],
        doc: "series_eval's dispatch among reverse Horner, rectangular splitting and binary \
              splitting. Dynamic nodes compare fitted runtime equations, one an algorithm: its \
              own parts (SeriesEq over Parts), with products priced by mul's equations and \
              finalize's division by division's (their defaults are the M1's fit). Static \
              nodes keep the shared ns per unit of work (approximate). Ahead of the equations, \
              a table for short series (series.small, derived from the series.short_* prices, \
              which are fitted on the series under its extent, as the main ones are on those \
              past it), the small ranges that go to Horner without them, and two guards on the shape's \
              ratios, rho (x's bits over the bits a term fixes) and beta (the exact fraction's \
              width over the frame). The equations are fitted, and the guards swept, on a \
              synthetic family of series scattered over frame, terms, x's width and the \
              coefficients' size (tuner/synth.rs): the evaluators are general, and nothing here \
              is fitted to a caller's series.",
        knobs: &[
            Knob::f64("horner_ns_product", "", 1.115)
                .kind(Kind::Model { work: "Horner (series_eval_dyn_at forced, the entry's own work taken off): its high products by a multi-limb x, by mul's equations" })
                .doc("Horner's multiple of its high products' predicted time (x of 3 limbs or more)."),
            Knob::f64("horner_ns_div1", "", 0.0)
                .kind(Kind::Model { work: "Horner: limbs of div_prim passes" })
                .doc("Horner's ns per limb of a division by a one-limb Q."),
            Knob::f64("horner_ns_div2", "", 3.893)
                .kind(Kind::Model { work: "Horner: limbs of div_prim2 passes" })
                .doc("Horner's ns per limb of a division by a two-limb Q."),
            Knob::f64("horner_ns_mul1", "", 0.4163)
                .kind(Kind::Model { work: "Horner: limbs of mul_prim and add_mul passes" })
                .doc("Horner's ns per limb of a pass by a one-limb multiplier."),
            Knob::f64("horner_ns_mul2", "", 1.336)
                .kind(Kind::Model { work: "Horner: limbs of mul_prim2 passes" })
                .doc("Horner's ns per limb of a pass by a two-limb multiplier."),
            Knob::f64("horner_ns_light", "", 0.1322)
                .kind(Kind::Model { work: "Horner: limbs zero-filled or copied" })
                .doc("Horner's ns per limb zero-filled or copied."),
            Knob::f64("horner_ns_term", "", 5.963)
                .kind(Kind::Model { work: "Horner: terms" })
                .doc("Horner's ns per term, for its coefficients and weights."),
            Knob::f64("horner_ns_call", "", 3.642)
                .kind(Kind::Model { work: "Horner: calls of its linear kernels" })
                .doc("Horner's ns per call of a linear kernel."),
            Knob::f64("rect_ns_product", "", 1.008)
                .kind(Kind::Model { work: "rect (forced, on Horner's prices for the linear kernels): its powers' and blocks' high products, by mul's equations" })
                .doc("Rect's multiple of its high products' predicted time."),
            Knob::f64("rect_ns_div1", "", 2.869)
                .kind(Kind::Model { work: "rect: limbs of div_prim passes (its groups end at its blocks, so it makes them in every range; Horner's long series hardly make one)" })
                .doc("Rect's ns per limb of a division by a one-limb Q."),
            Knob::f64("rect_ns_call", "", 5.137)
                .kind(Kind::Model { work: "rect: calls of its linear kernels" })
                .doc("Rect's ns per call of a linear kernel."),
            Knob::f64("rect_ns_one", "", 686.1)
                .kind(Kind::Model { work: "rect: 1" })
                .doc("Rect's fixed ns a call (its three scratch acquisitions among it)."),
            Knob::f64("bin_split_ns_product", "", 1.013)
                .kind(Kind::Model { work: "binary splitting (forced; trees that split across the pool left out): its merges' and x-leaves' products, by mul's equations" })
                .doc("Binary splitting's multiple of its products' predicted time."),
            Knob::f64("bin_split_ns_division", "", 0.9417)
                .kind(Kind::Model { work: "binary splitting: finalize's division, by division's equations" })
                .doc("Binary splitting's multiple of finalize's division's predicted time."),
            Knob::f64("bin_split_ns_div1", "", 1.385)
                .kind(Kind::Model { work: "binary splitting: limbs of finalize's div_prim pass (a one-limb Q)" })
                .doc("Binary splitting's ns per limb of a division by a one-limb Q."),
            Knob::f64("bin_split_ns_mul2", "", 0.8229)
                .kind(Kind::Model { work: "binary splitting: limbs of passes by a two-limb multiplier" })
                .doc("Binary splitting's ns per limb of a pass by a two-limb multiplier."),
            Knob::f64("bin_split_ns_add", "", 0.2491)
                .kind(Kind::Model { work: "binary splitting: limbs added or shifted" })
                .doc("Binary splitting's ns per limb of an addition or bit shift."),
            Knob::f64("bin_split_ns_light", "", 0.3143)
                .kind(Kind::Model { work: "binary splitting: limbs zero-filled or copied" })
                .doc("Binary splitting's ns per limb zero-filled or copied."),
            Knob::f64("bin_split_ns_term", "", 1.431)
                .kind(Kind::Model { work: "binary splitting: terms" })
                .doc("Binary splitting's ns per term."),
            Knob::f64("bin_split_ns_call", "", 4.435)
                .kind(Kind::Model { work: "binary splitting: calls of its linear kernels" })
                .doc("Binary splitting's ns per call of a linear kernel."),
            Knob::f64("bin_split_ns_alloc", "", 6.506)
                .kind(Kind::Model { work: "binary splitting: allocations" })
                .doc("Binary splitting's ns per allocation."),
            Knob::f64("short_horner_ns_product", "", 1.103)
                .kind(Kind::Model { work: "Horner (series_eval_dyn_at forced, the entry's own work taken off): its high products by a multi-limb x, by mul's equations" })
                .doc("Short series (the small-series table's derivation only): Horner's multiple of its high products' predicted time (x of 3 limbs or more)."),
            Knob::f64("short_horner_ns_div1", "", 2.857)
                .kind(Kind::Model { work: "Horner: limbs of div_prim passes" })
                .doc("Short series (the small-series table's derivation only): Horner's ns per limb of a division by a one-limb Q."),
            Knob::f64("short_horner_ns_div2", "", 3.712)
                .kind(Kind::Model { work: "Horner: limbs of div_prim2 passes" })
                .doc("Short series (the small-series table's derivation only): Horner's ns per limb of a division by a two-limb Q."),
            Knob::f64("short_horner_ns_mul1", "", 0.3092)
                .kind(Kind::Model { work: "Horner: limbs of mul_prim and add_mul passes" })
                .doc("Short series (the small-series table's derivation only): Horner's ns per limb of a pass by a one-limb multiplier."),
            Knob::f64("short_horner_ns_mul2", "", 0.6331)
                .kind(Kind::Model { work: "Horner: limbs of mul_prim2 passes" })
                .doc("Short series (the small-series table's derivation only): Horner's ns per limb of a pass by a two-limb multiplier."),
            Knob::f64("short_horner_ns_light", "", 0.2266)
                .kind(Kind::Model { work: "Horner: limbs zero-filled or copied" })
                .doc("Short series (the small-series table's derivation only): Horner's ns per limb zero-filled or copied."),
            Knob::f64("short_horner_ns_term", "", 6.356)
                .kind(Kind::Model { work: "Horner: terms" })
                .doc("Short series (the small-series table's derivation only): Horner's ns per term, for its coefficients and weights."),
            Knob::f64("short_horner_ns_call", "", 3.81)
                .kind(Kind::Model { work: "Horner: calls of its linear kernels" })
                .doc("Short series (the small-series table's derivation only): Horner's ns per call of a linear kernel."),
            Knob::f64("short_rect_ns_product", "", 1.266)
                .kind(Kind::Model { work: "rect (forced, on Horner's prices for the linear kernels): its powers' and blocks' high products, by mul's equations" })
                .doc("Short series (the small-series table's derivation only): Rect's multiple of its high products' predicted time."),
            Knob::f64("short_rect_ns_div1", "", 2.479)
                .kind(Kind::Model { work: "rect: limbs of div_prim passes (its groups end at its blocks, so it makes them in every range; Horner's long series hardly make one)" })
                .doc("Short series (the small-series table's derivation only): Rect's ns per limb of a division by a one-limb Q."),
            Knob::f64("short_rect_ns_call", "", 6.333)
                .kind(Kind::Model { work: "rect: calls of its linear kernels" })
                .doc("Short series (the small-series table's derivation only): Rect's ns per call of a linear kernel."),
            Knob::f64("short_rect_ns_one", "", 151.5)
                .kind(Kind::Model { work: "rect: 1" })
                .doc("Short series (the small-series table's derivation only): Rect's fixed ns a call (its three scratch acquisitions among it)."),
            Knob::f64("short_bin_split_ns_product", "", 1.02)
                .kind(Kind::Model { work: "binary splitting (forced; trees that split across the pool left out): its merges' and x-leaves' products, by mul's equations" })
                .doc("Short series (the small-series table's derivation only): Binary splitting's multiple of its products' predicted time."),
            Knob::f64("short_bin_split_ns_division", "", 0.9191)
                .kind(Kind::Model { work: "binary splitting: finalize's division, by division's equations" })
                .doc("Short series (the small-series table's derivation only): Binary splitting's multiple of finalize's division's predicted time."),
            Knob::f64("short_bin_split_ns_div1", "", 2.728)
                .kind(Kind::Model { work: "binary splitting: limbs of finalize's div_prim pass (a one-limb Q)" })
                .doc("Short series (the small-series table's derivation only): Binary splitting's ns per limb of a division by a one-limb Q."),
            Knob::f64("short_bin_split_ns_mul2", "", 0.7603)
                .kind(Kind::Model { work: "binary splitting: limbs of passes by a two-limb multiplier" })
                .doc("Short series (the small-series table's derivation only): Binary splitting's ns per limb of a pass by a two-limb multiplier."),
            Knob::f64("short_bin_split_ns_add", "", 0.2229)
                .kind(Kind::Model { work: "binary splitting: limbs added or shifted" })
                .doc("Short series (the small-series table's derivation only): Binary splitting's ns per limb of an addition or bit shift."),
            Knob::f64("short_bin_split_ns_light", "", 0.246)
                .kind(Kind::Model { work: "binary splitting: limbs zero-filled or copied" })
                .doc("Short series (the small-series table's derivation only): Binary splitting's ns per limb zero-filled or copied."),
            Knob::f64("short_bin_split_ns_term", "", 1.12)
                .kind(Kind::Model { work: "binary splitting: terms" })
                .doc("Short series (the small-series table's derivation only): Binary splitting's ns per term."),
            Knob::f64("short_bin_split_ns_call", "", 4.019)
                .kind(Kind::Model { work: "binary splitting: calls of its linear kernels" })
                .doc("Short series (the small-series table's derivation only): Binary splitting's ns per call of a linear kernel."),
            Knob::f64("short_bin_split_ns_alloc", "", 6.346)
                .kind(Kind::Model { work: "binary splitting: allocations" })
                .doc("Short series (the small-series table's derivation only): Binary splitting's ns per allocation."),
            Knob::f64("bin_split_par_gain", "", 0.3)
                .kind(Kind::Model { work: "trees that split across the pool: the gain under which the serial trees' prices predict them best" })
                .threads(Threads::Pool)
                .valid(Range::Float { min: 0.0, max: 1.0 })
                .doc("What a tree's parallel split gains: a level on n threads takes 1 / (1 + gain (n - 1)) of its serial time."),
            Knob::f64("ns_div", "SERIES_NS_DIV", 3.574)
                .doc("Static nodes: ns per limb of div_prim/div_prim2 passes and reciprocal limbs."),
            Knob::f64("ns_scalar", "SERIES_NS_SCALAR", 0.1574)
                .doc("Static nodes: ns per limb times scalar limbs of linear passes."),
            Knob::f64("ns_product", "SERIES_NS_PRODUCT", 0.3847)
                .doc("Static nodes: ns per schoolbook limb product of mul_work's product estimate."),
            Knob::f64("ns_term", "SERIES_NS_TERM", 9.334)
                .doc("Static nodes: ns per Horner or rect term, for group and weight bookkeeping."),
            Knob::f64("ns_call", "SERIES_NS_CALL", 0.9307).doc("Static nodes: ns per kernel call."),
            Knob::f64("ns_fill", "SERIES_NS_FILL", 0.09114)
                .doc("ns per limb of static [u64; N] arrays zero-filled or moved."),
            Knob::new(
                "small",
                "",
                Ty::Table { algs: SERIES_SMALL_ALGS, measure: "class x 16 + the frame's bucket, terms", inputs: 2 },
                DefaultValue::Table(&[0, 0, 3]),
            )
            .kind(Kind::Derived(Derivation::Table))
            .doc(
                "Dynamic nodes: what a short series runs, read ahead of everything else, by \
                 change points over (class x 16 + the frame's bucket, terms): Horner, rect, binary \
                 splitting, or the equations (every cell past the table's extent). A bucket is \
                 the octave of the frame's limbs; a class is the kind and x's width, q's size \
                 and whether r is past 1 (utils::series::small_class). Derived from the equations at each class's representative shape \
                 (cargo tune derive), so a short series reads no equation: evaluating them \
                 costs as much as a series under a microsecond does. The default hands every \
                 range to the equations.",
            ),
            Knob::f64("bin_split_beta", "", 8.0).kind(Kind::Sweep).threads(Threads::Pool).doc(
                "Dynamic nodes: binary splitting isn't predicted, or taken, where its tree's \
                 width (P's limbs) passes this many frames and another algorithm can sum the \
                 range. Its numbers are the exact fraction's, so past a few frames it loses in \
                 proportion (2-3x at 3 to 9 frames, 30-60x past 64 on the M1). A value no shape \
                 reaches (the sweep's last candidate, 1e9) turns the guard off.",
            ),
            Knob::f64("rect_rho", "", 32.0).kind(Kind::Sweep).doc(
                "Dynamic nodes: rect is taken without the equations, at its balance block \
                 length, where x has at least rect_x_limbs limbs and rho, x's bits over the bits \
                 a term fixes, reaches this: rect's products by x are about 2 / sqrt(rho) of \
                 Horner's. A value no shape reaches (1e9) turns the guard off.",
            ),
            Knob::usize("rect_x_limbs", "", 16).kind(Kind::Sweep).doc(
                "The narrowest x, in limbs, that the rect guard takes: under a few limbs a \
                 product by x is a small share of Horner's work, and rect's gain with it.",
            ),
            Knob::usize("horner_limbs", "SERIES_HORNER_LIMBS", 28).doc(
                "Output limbs up to which a small range goes to Horner without the cost model \
                 (its extent: at 1-4 limbs the model costs about as much as the sum).",
            ),
            Knob::u64("horner_terms", "SERIES_HORNER_TERMS", 12).doc(
                "Terms up to which a BBP or Engel range within horner_limbs goes to Horner \
                 without the model, which wrongly favors bin_split there.",
            ),
        ],
    },
    Section {
        name: "consts",
        version: 1,
        derive: 1,
        after: &["base", "mul", "div", "sqrt", "bin_split", "series"],
        doc: "The ln 2 and pi caches past their 1024-limb tables: when an extension builds in \
              parallel (dyn: rayon, from new * (cached^2 + half^2) >= cutoff * half^2 output \
              limbs; static: scoped threads, from a size).",
        knobs: &[
            Knob::u64("dyn_ln2_par", "DYN_LN2_PAR_CUTOFF", 32).threads(Threads::Pool)
                .doc("ln2_dyn's cache_par cutoff, in new output limbs at a cold cache."),
            Knob::u64("dyn_ln2_par_half", "DYN_LN2_PAR_HALF", 30).threads(Threads::Pool)
                .doc("ln2_dyn's cache_par half: the cached limbs at which the cutoff halves."),
            Knob::u64("dyn_pi_par", "DYN_PI_PAR_CUTOFF", 104).threads(Threads::Pool)
                .doc("pi_dyn's cache_par cutoff, in new output limbs at a cold cache."),
            Knob::u64("dyn_pi_par_half", "DYN_PI_PAR_HALF", 256).threads(Threads::Pool)
                .doc("pi_dyn's cache_par half."),
            Knob::usize("static_ln2_par", "STATIC_LN2_PAR_CUTOFF", 40).threads(Threads::Cpus)
                .doc("ln2_static sums its four series on scoped threads from this many limbs."),
            Knob::usize("static_pi_par", "STATIC_PI_PAR_CUTOFF", 480).threads(Threads::Cpus).doc(
                "pi_static sums its series on a scoped thread beside the root from this many \
                 limbs.",
            ),
        ],
    },
    Section {
        name: "theta",
        version: 2,
        derive: 1,
        after: &["base", "mul"],
        doc: "Theta's powers: the frame limbs from which it takes the PARALLEL plan (threaded on \
              two or more workers) rather than SERIAL.",
        knobs: &[
            Knob::usize("dyn_par_limbs", "DYN_THETA_PAR_LIMBS", 256).kind(Kind::Sweep).threads(Threads::Pool)
                .doc("theta_dyn takes the PARALLEL plan from this many frame limbs. GUESS."),
            Knob::usize("static_par_limbs", "STATIC_THETA_PAR_LIMBS", 1024).kind(Kind::Sweep).threads(Threads::Cpus)
                .doc("theta_static takes the PARALLEL plan from this many frame limbs. GUESS."),
        ],
    },
    Section {
        name: "agm",
        version: 2,
        derive: 1,
        after: &["base", "mul", "div", "sqrt"],
        doc: "The AGM: how many terms of its closed-form tail may replace its last steps.",
        knobs: &[
            Knob::usize("dyn_tail_terms", "DYN_AGM_TAIL_TERMS", 10).kind(Kind::Sweep)
                .search(Range::Int { min: 0, max: 19 })
                .doc("agm_dyn's tail cap (any value is valid: it is clamped to AGM_TAIL's 19)."),
            Knob::usize("static_tail_terms", "STATIC_AGM_TAIL_TERMS", 4).kind(Kind::Sweep)
                .search(Range::Int { min: 0, max: 19 })
                .doc("agm_static's tail cap."),
        ],
    },
    Section {
        name: "exp",
        version: 3,
        derive: 1,
        after: &["base", "mul", "div", "series", "consts"],
        doc: "exp: its bit-burst chunks (the first's width, when they run in parallel) and the \
              squares that shorten the series.",
        knobs: &[
            Knob::usize("dyn_chunk_par_work", "DYN_EXP_CHUNK_PAR_WORK", 1280).threads(Threads::Pool).kind(Kind::Sweep)
                .doc("exp_dyn_entry sums its chunks in parallel from chunks * working limbs this large."),
            Knob::u64("dyn_chunk_min_bits", "DYN_EXP_CHUNK_MIN_BITS", 16).kind(Kind::Sweep)
                .doc("exp_dyn_entry's first chunk is at least this many bits wide."),
            Knob::usize("static_chunk_par_work", "STATIC_EXP_CHUNK_PAR_WORK", 768).threads(Threads::Cpus)
                .doc("exp_static_entry's counterpart of dyn_chunk_par_work (scoped threads)."),
            Knob::u64("static_chunk_min_bits", "STATIC_EXP_CHUNK_MIN_BITS", 16).kind(Kind::Sweep)
                .doc("exp_static_entry's counterpart of dyn_chunk_min_bits."),
            Knob::u64("dyn_sqr_bits", "DYN_EXP_SQR_BITS", 24).kind(Kind::Sweep).int_range(0, 256).doc(
                "exp_dyn scales r by 2^-j and squares j times, j this less r's leading zeros. At \
                 most 256: each square is a full product, so the range bounds the work.",
            ),
            Knob::u64("static_sqr_bits", "STATIC_EXP_SQR_BITS", 24).kind(Kind::Sweep).int_range(0, 256)
                .doc("exp_static's counterpart of dyn_sqr_bits. GUESS."),
        ],
    },
    Section {
        name: "ln",
        version: 5,
        derive: 1,
        after: &["base", "mul", "div", "sqrt", "series", "consts", "theta", "agm"],
        doc: "ln: the square roots before the series, its chunks, and the series-vs-AGM choice \
              (agm_wins' one-thread cutoff and its bound on a pool, theta's term count).",
        knobs: &[
            Knob::f64("dyn_sqrt_a", "DYN_LN_SQRT_A", 0.25).float_range(0.0, 3.0).doc(
                "ln_dyn takes floor(A log2(64 out.len())) square roots before the series. At most \
                 3, the largest scale the tests cover.",
            ),
            Knob::f64("static_sqrt_a", "STATIC_LN_SQRT_A", 0.25).float_range(0.0, 3.0)
                .doc("ln_static's counterpart of dyn_sqrt_a. GUESS."),
            Knob::u64("dyn_chunk_min_bits", "DYN_LN_CHUNK_MIN_BITS", 2)
                .doc("ln_dyn_entry's first chunk is at least this many bits wide. GUESS."),
            Knob::u64("static_chunk_min_bits", "STATIC_LN_CHUNK_MIN_BITS", 2)
                .doc("ln_static_entry's counterpart of dyn_chunk_min_bits. GUESS."),
            Knob::usize("dyn_chunk_par_work", "DYN_LN_CHUNK_PAR_WORK", 512).threads(Threads::Pool).kind(Kind::Sweep)
                .doc("ln_dyn_entry sums its chunks in parallel from chunks * working limbs this large. GUESS."),
            Knob::usize("static_chunk_par_work", "STATIC_LN_CHUNK_PAR_WORK", 768).threads(Threads::Cpus)
                .doc("ln_static_entry's counterpart of dyn_chunk_par_work. GUESS."),
            Knob::usize("dyn_agm_limbs", "DYN_LN_AGM_LIMBS", 1024).kind(Kind::Sweep)
                .doc("agm_wins' anchor: ln_dyn never takes the AGM below this many output limbs."),
            Knob::usize("static_agm_limbs", "STATIC_LN_AGM_LIMBS", 1024)
                .doc("ln_static's counterpart of dyn_agm_limbs."),
            Knob::usize("dyn_agm_par_until", "", 0)
                .threads(Threads::Pool)
                .kind(Kind::Sweep)
                .doc(
                    "On a pool of two or more threads ln_dyn takes the AGM only below this many \
                     output limbs (and from dyn_agm_limbs): the series' chunks split across threads \
                     from some size and win from there. The parallel half of the serial/parallel \
                     pair, swept on the tune's pool; it replaced the series' Amdahl fraction \
                     (DYN_LN_SERIES_PAR_FRACTION, 0.85), which scaled the serial cutoff. Default 0: \
                     the series on every pool, as before.",
                ),
            Knob::usize("static_agm_par_until", "", 0)
                .threads(Threads::Cpus)
                .doc("ln_static's counterpart of dyn_agm_par_until, on two or more CPUs. Approximate (unswept)."),
            Knob::usize("dyn_agm_terms", "DYN_LN_AGM_TERMS", 5).kind(Kind::Sweep).int_range(0, 16).doc(
                "Theta's term count T in ln_dyn's AGM (a const generic): 0..=16, \
                 THETA_MAX_TERMS.",
            ),
            Knob::usize("static_agm_terms", "STATIC_LN_AGM_TERMS", 6).int_range(0, 9).doc(
                "Theta's term count T in ln_static's AGM: 0..=9, the T whose least-work plan fits \
                 THETA_STATIC_SLOTS.",
            ),
        ],
    },
];
