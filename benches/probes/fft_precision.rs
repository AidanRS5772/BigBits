//! Measures the 16-bit FFT entries' rounding against their fold check (FFT_FOLD_TRIGGER) and dispatch
//! limits (FFT_16BIT_CUTOFF, FFT_SQR_16BIT_CUTOFF,
//! FFT_MID_16BIT_CUTOFF, FFT_BNM1_16BIT_CUTOFF). For every FFT transform size in a range it runs
//! `count` operands at that size's widest shape through the unchecked (raw) entry and reports, per
//! size:
//! - fires: calls that run the fold check, trips: calls it fails, which a checked entry would redo
//!   on the NTT (FftRounding::fold_failed), needless: right results it ran on, and t30/t375: calls whose
//!   error passes 0.30 and 0.375, for the tail's shape;
//! - wrong: calls whose raw result is wrong, against the residue verifier (products, squares) or
//!   the NTT (middle products, cyclic squares; skipped by `no-oracle`);
//! - by_fold: wrong calls the fold check catches; blind: wrong calls it ran on and passed, quiet:
//!   wrong calls that read too low for it to run, and missed: their sum, which must never happen;
//! - err: the largest rounding error, and min_wrong: the smallest error among the wrong calls.
//! Modes: `worst` cycles the unsigned-digit kinds (all ones, a step, random digits all at least
//! 0xF000/0xFF00/0xE000; KIND=k forces any kind below); `rate` takes uniformly random limbs (the
//! dispatch limits' trip rates); `sweep` is the soundness run over padded and unbalanced shapes:
//! every transform size, FILLS, RATIOS (products) and KINDS, one row per shape and a summary at the
//! end, with REACHABLE=1 leaving out shapes past the dispatch limits. Rounding differs between rustfft's NEON and AVX paths, so measure on each architecture; see
//! docs/fft_precision.md.
//!
//! The shape: FILL=f takes the operands between the previous transform size (f near 0, the most
//! padding) and this one (f = 1, the widest and the default), and RATIO=r makes a product's long
//! side r times its short side. LIMBS=l overrides both with operands of l limbs (squares, cyclic
//! squares), l x l (products) or short side l (middle products), which must land on the size.
//!
//! Build after `cargo build --release` (substitute the rlib's actual hash):
//! rustc --edition=2021 -O -C lto=thin -C panic=abort -C codegen-units=1 \
//!   benches/probes/fft_precision.rs \
//!   --extern big_bits=target/release/deps/libbig_bits-HASH.rlib \
//!   -L dependency=target/release/deps -o target/fft_precision
//! target/fft_precision <worst|rate|sweep> <sqr|mul|mid|cyclic> <lo N> <hi N> <count> [seed]
//!   [no-oracle]
//! STOP=k ends a size after k + 1 trips (count then shows the calls made), for the rate limits'
//! "more than k trips" rule.
//! The size column is operand limbs for sqr, output limbs for mul, short limbs n for mid and
//! modulus limbs m for cyclic.
use big_bits::utils::mul::{
    bnm1_size, fft_entry_raw, fft_mid_mul_raw, fft_sqr_entry_raw, fft_sqr_mod_bnm1_raw,
    find_fft_size, ntt_mid_mul_dyn, ntt_sqr_entry_dyn, FftRounding,
};
use big_bits::utils::utils::{fold_mod_bnm1, sub_buf};
use big_bits::utils::verify::{verify_mul, verify_sqr};
use big_bits::utils::{
    FFT_16BIT_CUTOFF, FFT_BNM1_16BIT_CUTOFF, FFT_FOLD_TRIGGER, FFT_MID_16BIT_CUTOFF,
    FFT_SQR_16BIT_CUTOFF,
};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed ^ (*seed >> 29)
}

// An unsigned chirp: digit k is base + span (1 + cos(pi k^2 / D + phase)) / 2. Chirps collapse the
// transform's rounding pattern into spikes at the operands' length boundaries while its rms stays
// the same, so they are the operands a largest-distance check is least suited to.
fn chirp(len: usize, base: f64, span: f64, seed: &mut u64) -> Vec<u64> {
    let digits = 4 * len;
    let phase = (lcg(seed) >> 11) as f64 * (std::f64::consts::TAU / (1u64 << 53) as f64);
    let digit = |k: usize| {
        let angle = std::f64::consts::PI * (k as f64 * k as f64 / digits as f64) + phase;
        (base + span * (1.0 + angle.cos()) / 2.0) as u64
    };
    (0..len)
        .map(|i| {
            (0..4).fold(0u64, |limb, j| limb | digit(4 * i + j) << (16 * j))
        })
        .collect()
}

// The operand whose balanced digits are these, with any final borrow dropped (the top bit set below
// makes it positive): raw 0x7FFF or 0x8000 limbs would balance to one sign whatever their digits.
fn from_balanced(digits: impl Iterator<Item = i64>, len: usize) -> Vec<u64> {
    let mut v = vec![0u64; len];
    let mut carry = 0i64;
    for (k, d) in digits.take(4 * len).enumerate() {
        let t = d + carry;
        v[k / 4] |= (t.rem_euclid(1 << 16) as u64) << (16 * (k % 4));
        carry = t.div_euclid(1 << 16);
    }
    v
}

// kinds 0-4: all ones, a step, and random digits at least 0xF000, 0xFF00, 0xE000; kind 5: uniformly
// random limbs; kinds 6 and 7: random digits at least 0x8000 and 0xC000; kinds 8 and 9: unsigned
// chirps. Kinds 10 on are written in balanced digits, the operands balanced digits cancel least on:
// 10, all -2^15; 11, all 2^15 - 1; 12, random signs at full size; 13, random digits in [2^14, 2^15);
// 14, a chirp at full size; 15 and 16, full-size digits of period 3 and 12; 17 and 18, kinds 10 and
// 16 with random low bits, so that repeated calls differ. The top limb is nonzero, so every operand
// has its full length.
fn operand(kind: usize, len: usize, seed: &mut u64) -> Vec<u64> {
    let mut v: Vec<u64> = match kind {
        10 => from_balanced(std::iter::repeat(-(1 << 15)), len),
        11 => from_balanced(std::iter::repeat((1 << 15) - 1), len),
        12 => from_balanced(
            std::iter::repeat_with(|| if lcg(seed) >> 63 == 0 { 32767 } else { -32768 }),
            len,
        ),
        13 => from_balanced(
            std::iter::repeat_with(|| (1 << 14) + (lcg(seed) >> 50) as i64),
            len,
        ),
        14 => {
            let digits = 4 * len;
            let phase = (lcg(seed) >> 11) as f64 * (std::f64::consts::TAU / (1u64 << 53) as f64);
            from_balanced(
                (0..digits).map(|k| {
                    let angle = std::f64::consts::PI * (k as f64 * k as f64 / digits as f64) + phase;
                    (32767.0 * angle.cos()) as i64
                }),
                len,
            )
        }
        15 => from_balanced((0..).map(|k| if k % 3 == 2 { -32768 } else { 32767 }), len),
        16 => from_balanced((0..).map(|k| if k % 12 < 6 { 32767 } else { -32768 }), len),
        17 => from_balanced(
            std::iter::repeat_with(|| -(1 << 15) + (lcg(seed) >> 56) as i64),
            len,
        ),
        18 => from_balanced(
            (0..).map(|k| {
                let low = (lcg(seed) >> 56) as i64;
                if k % 12 < 6 {
                    32767 - low
                } else {
                    -32768 + low
                }
            }),
            len,
        ),
        8 => chirp(len, 0.0, 65535.0, seed),
        9 => chirp(len, 49152.0, 16383.0, seed),
        0 => vec![u64::MAX; len],
        1 => (0..len)
            .map(|i| if i >= len / 2 { u64::MAX } else { 0 })
            .collect(),
        2..=4 | 6 | 7 => {
            let digits = [
                0xF000_F000_F000_F000u64,
                0xFF00_FF00_FF00_FF00,
                0xE000_E000_E000_E000,
                0,
                0x8000_8000_8000_8000,
                0xC000_C000_C000_C000,
            ];
            (0..len).map(|_| lcg(seed) | digits[kind - 2]).collect()
        }
        _ => (0..len).map(|_| lcg(seed)).collect(),
    };
    if kind == 5 {
        if v[len - 1] == 0 {
            v[len - 1] = 1;
        }
    } else {
        v[len - 1] |= 1 << 63;
    }
    v
}

fn full(f: impl FnOnce(&mut [u64]) -> u64, len: usize) -> Vec<u64> {
    let mut out = vec![0; len];
    out[len - 1] = f(&mut out[..len - 1]);
    out
}

// The largest transform size below n.
fn previous_size(n: usize) -> usize {
    let mut below = n - 1;
    while below >= 4 && find_fft_size(below) >= n {
        below -= 1;
    }
    below
}

// The operands' lengths at transform size n: (l, s) for products, (l, 0) otherwise. The digits they
// need lie a fraction `fill` of the way from the previous transform size to n, and a product's l is
// `ratio` times its s.
fn shape(op: &str, n: usize, fill: f64, ratio: usize, limbs: Option<usize>) -> (usize, usize) {
    let prev = previous_size(n);
    let t = prev + (((n - prev) as f64 * fill).ceil() as usize).clamp(1, n - prev);
    let lands = |t: usize| {
        assert_eq!(find_fft_size(t), n, "{op} shape needs {t} digits, not transform {n}");
    };
    match op {
        "sqr" => {
            let l = limbs.unwrap_or_else(|| (t + 8) / 8).min((n + 1) / 8);
            lands(8 * l - 1);
            (l, 0)
        }
        "mul" => {
            let (l, s) = match limbs {
                Some(l) => (l, l),
                None => {
                    let sum = ((t + 4) / 4).min((n + 1) / 4);
                    let s = (sum / (ratio + 1)).max(1);
                    (sum - s, s)
                }
            };
            lands(4 * (l + s) - 1);
            (l, s)
        }
        "mid" => {
            let m = limbs.unwrap_or_else(|| (t + 3) / 8).min((n - 4) / 8);
            lands(8 * m + 4);
            (m, 0)
        }
        "cyclic" => (limbs.unwrap_or_else(|| ((t + 3) / 4).min(n / 4)), 0),
        _ => panic!("op is sqr, mul, mid or cyclic"),
    }
}

// Two integers of the same length, differing by more than their two low limbs. The middle products
// differ by design there, and comparing limbs would flag a tiny difference that carries across a
// run of ones.
fn differ(a: &[u64], b: &[u64]) -> bool {
    let sub = |x: &[u64], y: &[u64]| {
        let mut d = x.to_vec();
        sub_buf(&mut d, y);
        d.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1)
    };
    sub(a, b).min(sub(b, a)) > 2
}

// One call at transform size n: (shape size, rounding, wrong if checked).
fn run(
    op: &str,
    n: usize,
    (l, s): (usize, usize),
    kind: usize,
    oracle: bool,
    seed: &mut u64,
) -> (usize, FftRounding, Option<bool>) {
    match op {
        "sqr" => {
            let a = operand(kind, l, seed);
            let mut err = FftRounding::default();
            let got = full(
                |o| {
                    let (top, e) = fft_sqr_entry_raw(&a, o);
                    err = e;
                    top
                },
                2 * l,
            );
            (l, err, Some(!verify_sqr(&a, &got)))
        }
        "mul" => {
            let (a, b) = (operand(kind, l, seed), operand(kind, s, seed));
            let mut err = FftRounding::default();
            let got = full(
                |o| {
                    let (top, e) = fft_entry_raw(&a, &b, o);
                    err = e;
                    top
                },
                l + s,
            );
            (l + s - 1, err, Some(!verify_mul(&a, &b, &got)))
        }
        "mid" => {
            let m = l;
            let (long, short) = (operand(kind, 2 * m - 1, seed), operand(kind, m, seed));
            let mut got = vec![0; m + 2];
            let (gc, err) = fft_mid_mul_raw(&long, &short, &mut got[..m]);
            (got[m], got[m + 1]) = gc;
            let wrong = oracle.then(|| {
                let mut exact = vec![0; m + 2];
                (exact[m], exact[m + 1]) = ntt_mid_mul_dyn(&long, &short, &mut exact[..m]);
                differ(&got, &exact)
            });
            (m, err, wrong)
        }
        "cyclic" => {
            let m = n / 4;
            let a = operand(kind, l, seed);
            let mut got = vec![0; m];
            let err = fft_sqr_mod_bnm1_raw(&a, &mut got);
            let wrong = oracle.then(|| {
                let mut exact = vec![0; m];
                fold_mod_bnm1(&full(|o| ntt_sqr_entry_dyn(&a, o), 2 * l), &mut exact);
                got != exact
            });
            (m, err, wrong)
        }
        _ => panic!("op is sqr, mul, mid or cyclic"),
    }
}


// Every shape the soundness run covers at sizes [lo, hi], with a summary by largest coefficient.
// How a call fared: whether the fold check ran, and for a wrong call whether it caught it, ran and
// passed it, or never ran.
#[derive(Default)]
struct Tally {
    calls: usize,
    fires: usize,
    trips: usize,
    needless: usize,
    t30: usize,
    t375: usize,
    wrong: usize,
    by_fold: usize,
    blind: usize,
    quiet: usize,
    max_err: f64,
    min_wrong: f64,
}

impl Tally {
    fn new() -> Self {
        Tally { min_wrong: f64::INFINITY, ..Default::default() }
    }

    fn add(&mut self, r: FftRounding, bad: Option<bool>) {
        let fired = r.err > FFT_FOLD_TRIGGER;
        self.calls += 1;
        self.fires += fired as usize;
        self.trips += r.fold_failed as usize;
        self.t30 += (r.err > 0.30) as usize;
        self.t375 += (r.err > 0.375) as usize;
        self.max_err = self.max_err.max(r.err);
        if bad == Some(true) {
            self.wrong += 1;
            self.min_wrong = self.min_wrong.min(r.err);
            if !fired {
                self.quiet += 1;
            } else if r.fold_failed {
                self.by_fold += 1;
            } else {
                self.blind += 1;
            }
        } else if bad == Some(false) {
            self.needless += fired as usize;
        }
    }

    fn merge(&mut self, o: &Tally) {
        self.calls += o.calls;
        self.fires += o.fires;
        self.trips += o.trips;
        self.needless += o.needless;
        self.t30 += o.t30;
        self.t375 += o.t375;
        self.wrong += o.wrong;
        self.by_fold += o.by_fold;
        self.blind += o.blind;
        self.quiet += o.quiet;
        self.max_err = self.max_err.max(o.max_err);
        self.min_wrong = self.min_wrong.min(o.min_wrong);
    }

    fn row(&self) -> String {
        format!(
            "count={} fires={} trips={} needless={} t30={} t375={} wrong={} by_fold={} blind={} \
             quiet={} missed={} err={:.4} min_wrong={:.4}",
            self.calls,
            self.fires,
            self.trips,
            self.needless,
            self.t30,
            self.t375,
            self.wrong,
            self.by_fold,
            self.blind,
            self.quiet,
            self.blind + self.quiet,
            self.max_err,
            self.min_wrong
        )
    }
}

// Whether a dynamic dispatcher can send this shape to the FFT.
fn inside_limits(op: &str, (l, s): (usize, usize)) -> bool {
    match op {
        "sqr" => l <= FFT_SQR_16BIT_CUTOFF,
        "mul" => l + s - 1 <= FFT_16BIT_CUTOFF,
        "mid" => l <= FFT_MID_16BIT_CUTOFF,
        _ => l <= FFT_BNM1_16BIT_CUTOFF,
    }
}

fn sweep(op: &str, lo: usize, hi: usize, count: usize, seed: &mut u64, oracle: bool) {
    let list = |name: &str, default: &str| -> Vec<f64> {
        std::env::var(name)
            .unwrap_or_else(|_| default.to_string())
            .split(',')
            .map(|v| v.parse().unwrap())
            .collect()
    };
    let (fills, ratios, kinds) = (
        list("FILLS", "1,0.5,0.01"),
        list("RATIOS", "1,2,4,8,16"),
        list("KINDS", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18"),
    );
    let reachable = std::env::var("REACHABLE").is_ok();
    let mut total = Tally::new();
    let mut t = lo;
    while t <= hi {
        let n = find_fft_size(t);
        t = n + 1;
        if n > hi {
            break;
        }
        if op == "cyclic" && bnm1_size(n / 4) != n / 4 {
            continue;
        }
        for &fill in &fills {
            let ratios: &[f64] = if op == "mul" { &ratios } else { &[1.0] };
            for &ratio in ratios {
                for &kind in &kinds {
                    let kind = kind as usize;
                    let form = shape(op, n, fill, ratio as usize, None);
                    if form.0 == 0 || (op == "mul" && form.1 == 0) {
                        continue;
                    }
                    if reachable && !inside_limits(op, form) {
                        continue;
                    }
                    let mut tally = Tally::new();
                    for _ in 0..count {
                        let (_, r, bad) = run(op, n, form, kind, oracle, seed);
                        tally.add(r, bad);
                    }
                    println!(
                        "sweep {op} N={n} l={} s={} kind={kind} {}",
                        form.0,
                        form.1,
                        tally.row()
                    );
                    total.merge(&tally);
                }
            }
        }
    }
    println!("summary {op} {}", total.row());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mode, op) = (args[1].as_str(), args[2].as_str());
    let (lo, hi, count): (usize, usize, usize) = (
        args[3].parse().unwrap(),
        args[4].parse().unwrap(),
        args[5].parse().unwrap(),
    );
    let mut seed: u64 = args.get(6).map_or(1, |s| s.parse().unwrap());
    let oracle = args.get(7).map_or(true, |s| s != "no-oracle");
    let forced_kind: Option<usize> = std::env::var("KIND").ok().map(|k| k.parse().unwrap());
    let stop: Option<usize> = std::env::var("STOP").ok().map(|k| k.parse().unwrap());
    assert!(
        mode == "worst" || mode == "rate" || mode == "sweep",
        "mode is worst, rate or sweep"
    );
    if mode == "sweep" {
        sweep(op, lo, hi, count, &mut seed, oracle);
        return;
    }
    let env_num = |name: &str| std::env::var(name).ok().map(|v| v.parse::<f64>().unwrap());
    let (fill, ratio, limbs) = (
        env_num("FILL").unwrap_or(1.0),
        env_num("RATIO").unwrap_or(1.0) as usize,
        env_num("LIMBS").map(|v| v as usize),
    );
    let mut t = lo;
    while t <= hi {
        let n = find_fft_size(t);
        t = n + 1;
        if n > hi {
            break;
        }
        if op == "cyclic" && bnm1_size(n / 4) != n / 4 {
            continue;
        }
        let form = shape(op, n, fill, ratio, limbs);
        let mut size = 0;
        let mut tally = Tally::new();
        for i in 0..count {
            let kind = match (mode, forced_kind) {
                ("rate", _) => 5,
                (_, Some(k)) => k,
                _ => i % 5,
            };
            let (sz, r, bad) = run(op, n, form, kind, oracle, &mut seed);
            size = sz;
            tally.add(r, bad);
            if stop.is_some_and(|k| tally.trips > k) {
                break;
            }
        }
        println!("{mode} {op} N={n} size={size} {}", tally.row());
    }
}
