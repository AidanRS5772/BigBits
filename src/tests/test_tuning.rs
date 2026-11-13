// The tuning system's own tests (docs/tuning_system.md, "Validation"): the schema, the resolver
// and the file format in every build; the live store and the random-knob test with `tune`.
//
// Invariant 5: nothing may change a live knob while a library call runs. `cargo test` runs tests
// on many threads, so tests in this process only read knobs or set them to the value they already
// hold. Everything that changes them runs in `random_knobs_child`, which `random_knobs` starts in
// its own process.

use crate::tuning::file::{self, Num, Raw};
use crate::tuning::resolve::{self, check_table, flat, Layer, Origin, Target, Value};
use crate::tuning::schema_types::{Alg, Kind, Knob, Range, Section, Threads, Ty};
use crate::tuning::{self, PATHS, RESOLVED_BITS};

#[test]
fn test_schema_checks_pass() {
    if let Err(errs) = resolve::check_schema(tuning::schema()) {
        panic!("schema errors:\n  {}", errs.join("\n  "));
    }
    let paths: Vec<String> = flat(tuning::schema()).into_iter().map(|f| f.path).collect();
    assert_eq!(paths, PATHS.to_vec());
}

// A value's entry in RESOLVED_BITS: a scalar's bits, a table's hash.
fn value_bits(v: &Value) -> u64 {
    match v {
        Value::Table(t) => crate::tuning::hash::table_bits(t),
        v => v.bits().unwrap(),
    }
}

// The layers build.rs resolved with: the target's preset, if there is one.
fn build_layers() -> Vec<(String, file::File)> {
    let (arch, os) = (std::env::consts::ARCH, std::env::consts::OS);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut layers: Vec<(String, file::File)> = [format!("presets/{arch}-{os}.toml"), format!("presets/{arch}.toml")]
        .into_iter()
        .find_map(|n| std::fs::read_to_string(root.join(&n)).ok().map(|t| (n, file::parse(&t).unwrap())))
        .into_iter()
        .collect();
    // Then the machine's own file, as build.rs layers it: the one BIGBITS_TUNING named when this
    // was built (presets/local.toml in a checkout, which `cargo tune tune` writes), unless it
    // was tuned on another target. Without it this file's first test failed on any machine
    // that had tuned itself.
    if let Some(path) = option_env!("BIGBITS_TUNING").filter(|p| !p.is_empty()) {
        if let Ok(text) = std::fs::read_to_string(path) {
            let f = file::parse(&text).unwrap();
            if resolve::layer_target_ok(&f, &Target { arch, os }).is_ok() {
                layers.push((path.to_string(), f));
            }
        }
    }
    layers
}

// Every accessor returns the resolved value (in a tune build too: nothing here changes the cells).
#[test]
fn test_accessors_match_resolved() {
    assert_eq!(tuning::all_accessor_bits(), RESOLVED_BITS);
    let files = build_layers();
    let layers: Vec<Layer> = files.iter().map(|(n, f)| Layer { name: n, file: f }).collect();
    let target = Target { arch: std::env::consts::ARCH, os: std::env::consts::OS };
    let r = resolve::resolve(tuning::schema(), &layers, &target).expect("the build's values resolve");
    let bits: Vec<u64> = r.values.iter().map(value_bits).collect();
    assert_eq!(bits, RESOLVED_BITS.to_vec());
    for (p, want) in PATHS.iter().zip(&r.values) {
        assert_eq!(&tuning::get(p).unwrap(), want, "{p}");
    }
    // The spec of the current values parses back to them.
    let parsed = tuning::parse_spec(&tuning::spec()).unwrap();
    assert_eq!(parsed.len(), PATHS.len());
    for (i, v) in parsed {
        assert_eq!(value_bits(&v), RESOLVED_BITS[i]);
    }
}

// `src/utils` must read knobs through accessors: `resolved::` would bypass live values.
#[test]
fn test_utils_never_read_resolved() {
    fn walk(dir: &std::path::Path, hits: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, hits);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let text = std::fs::read_to_string(&p).unwrap();
                for (n, line) in text.lines().enumerate() {
                    if line.contains("resolved::") {
                        hits.push(format!("{}:{}", p.display(), n + 1));
                    }
                }
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/utils");
    let mut hits = Vec::new();
    walk(&root, &mut hits);
    // The scan itself must be able to find a name it looks for.
    assert!(std::fs::read_to_string(root.join("mod.rs")).unwrap().contains("ScratchGuard"));
    assert!(hits.is_empty(), "src/utils reads `resolved::` at {hits:?}");
}

// mul.sqr_bnm1_fft follows mul.fft_sqr until it is set itself (sqrt.dyn_wrapped_rem no longer
// follows it: see its doc).
#[test]
fn test_bnm1_cutoff_follows_fft_sqr_by_default() {
    let r = resolve::resolve(tuning::schema(), &[], &Target { arch: std::env::consts::ARCH, os: "" }).unwrap();
    let v = |p: &str| r.values[tuning::index(p).unwrap()].clone();
    assert_eq!(v("mul.sqr_bnm1_fft"), v("mul.fft_sqr"));
    assert_eq!(v("sqrt.dyn_wrapped_rem"), Value::Int(341));
}

// mul.kara's default is the old is_school boundary exactly, at every shape the recursion and the
// dispatchers can ask about (from s = 64 on every row is Karatsuba).
// `Table2::pick` without the row index (a binary search for the row): the lookup's oracle.
fn pick2(t: &[u32], s: u64, l: u64) -> usize {
    assert!(t.len() >= 3 && t.len() % 3 == 0);
    let (mut lo, mut hi) = (0, t.len() / 3);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if t[3 * mid] as u64 <= s {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if hi == 0 {
        return t[2] as usize;
    }
    let mut i = hi - 1;
    let row = t[3 * i];
    while t[3 * i + 1] as u64 > l && i > 0 && t[3 * (i - 1)] == row {
        i -= 1;
    }
    t[3 * i + 2] as usize
}

#[test]
fn test_kara_default_is_todays_boundary() {
    use crate::tuning::resolved::mul::{KARA as RESOLVED, KARA_ROWS};
    use crate::tuning::schema_types::DefaultValue;
    use crate::tuning::table::{row_index, Table2};
    use crate::utils::mul::{is_school_at, school_at};
    assert_eq!(row_index(RESOLVED), KARA_ROWS, "the generated index is the resolved table's");
    // The schema's default, whatever a preset resolves it to.
    let DefaultValue::Table(table) = tuning::knob("mul.kara").unwrap().default else { panic!("mul.kara is a table") };
    let rows: &'static [u32] = Box::leak(row_index(table).into_boxed_slice());
    let kara = Table2 { t: table, rows };
    for s in 3..300 {
        for l in s..2000 {
            let want = is_school_at(l, s, 63, 52.0);
            assert_eq!(school_at(&kara, l, s), want, "{l}x{s}");
            assert_eq!(pick2(table, s as u64, l as u64) == 0, want, "pick2 {l}x{s}");
        }
    }
}

// --- The runtime prediction layer (Phase 2) ---

#[test]
fn test_table_lookups() {
    use crate::tuning::table::pick1;
    let t = [1, 0, 9, 1, 40, 2];
    for (n, want) in [(0, 0), (1, 0), (8, 0), (9, 1), (39, 1), (40, 2), (u64::MAX, 2)] {
        assert_eq!(pick1(&t, n), want, "pick1 at {n}");
    }
    // Rows s = 1, 4, 10: within a row, the last change point at or below l.
    let t2 = [1, 1, 0, 4, 4, 0, 4, 12, 1, 10, 10, 0, 10, 20, 1, 10, 50, 2];
    for ((s, l), want) in [
        ((1, 1), 0),
        ((3, 100), 0),  // row 1
        ((4, 11), 0),
        ((4, 12), 1),   // row 4
        ((9, 1000), 1),
        ((10, 19), 0),
        ((10, 20), 1),
        ((12, 50), 2),  // row 10
        ((0, 0), 0),    // below every row: the first row's first
    ] {
        assert_eq!(pick2(&t2, s, l), want, "pick2 at ({s}, {l})");
        let rows = crate::tuning::table::row_index(&t2);
        let leaked = crate::tuning::table::Table2 { t: Box::leak(Box::new(t2)), rows: Box::leak(rows.into_boxed_slice()) };
        assert_eq!(leaked.pick(s as usize, l as usize), want, "Table2::pick at ({s}, {l})");
    }
}

// --- The tuner (tune builds) ---

// A parameter sweep takes the candidate losing least against the best one at each size.
#[cfg(feature = "tune")]
#[test]
fn test_param_sweep_choice() {
    use crate::tuning::tuner::sweep::pick_param;
    let cands = [Value::Int(0), Value::Int(1), Value::Int(2)];
    // Two sizes: candidate 1 is 5% off at the first and best at the second; 0 and 2 each win one
    // size and lose 30% at the other.
    let t = [100.0, 105.0, 130.0, 130.0, 100.0, 100.0];
    let (best, ev) = pick_param(2, &cands, &t, Some(&Value::Int(0)), 0.0);
    assert_eq!(best, 1, "{ev}");
    assert!(ev.contains("mean 2.50%") && ev.contains("the previous value 0 lost 15.00%"), "{ev}");
    // A previous value outside the candidates is left out of the evidence.
    assert!(!pick_param(2, &cands, &t, Some(&Value::Int(9)), 0.0).1.contains("previous"));
    // The previous value stays when it is within the tie band of the best (15% - 2.5% here),
    // and not when the band is narrower.
    let (kept, ev) = pick_param(2, &cands, &t, Some(&Value::Int(0)), 0.13);
    assert!(kept == 0 && ev.contains("kept") && ev.contains("mean 15.00%"), "{ev}");
    assert_eq!(pick_param(2, &cands, &t, Some(&Value::Int(0)), 0.12).0, 1);
}

#[cfg(feature = "tune")]
#[test]
fn test_fits_recover_coefficients() {
    use crate::tuning::tuner::fit::{analysis, fit_rel, fit_split, solve, ACCEPT_RMS};
    // t = 0.4 n + 12, with +-1% multiplicative noise.
    let ns: Vec<f64> = (1..=40).map(|i| (i * i) as f64).collect();
    let noise = |i: usize| 1.0 + 0.01 * (((i * 7919) % 21) as f64 / 10.0 - 1.0);
    let t: Vec<f64> = ns.iter().enumerate().map(|(i, n)| (0.4 * n + 12.0) * noise(i)).collect();
    let x: Vec<Vec<f64>> = ns.iter().map(|&n| vec![n, 1.0]).collect();
    let f = fit_rel(&x, &t);
    assert!((f.beta[0] / 0.4 - 1.0).abs() < 0.02 && (f.beta[1] / 12.0 - 1.0).abs() < 0.1, "{:?}", f.beta);
    assert!(f.rel_rms < 0.015, "{}", f.rel_rms);
    // A term the timings put below zero isn't needed: it is left out and reads 0, the rest are
    // solved again without it, and the fit says which term went and what it had solved.
    let rows: Vec<Vec<f64>> = (1..=10).map(|i| vec![i as f64, 1.0]).collect();
    let t: Vec<f64> = (1..=10).map(|i| 3.0 * i as f64 - 2.0).collect();
    let f = fit_rel(&rows, &t);
    // (On relative error the slope left is below 3: the small times, which the shift took most
    // from, weigh as much as the large.)
    assert!(f.beta[1] == 0.0 && f.beta[0] > 1.0 && f.beta[0] < 3.0, "{:?}", f.beta);
    assert!(f.dropped.len() == 1 && f.dropped[0].0 == 1 && (f.dropped[0].1 + 2.0).abs() < 1e-9, "{:?}", f.dropped);
    assert!(f.dropped_text(&["slope", "shift"]).contains("shift solved -2.000e0"), "{}", f.dropped_text(&["slope", "shift"]));
    // A term no row exercises reads 0 and isn't solved; the rest are.
    let z = solve(&[vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0]], &[2.0, 4.0, 6.0]);
    assert!((z.beta[0] - 2.0).abs() < 1e-12 && z.beta[1] == 0.0 && z.se[1] == 0.0, "{:?}", z.beta);
    // Two kinds sharing a shift, as base's passes are fitted.
    let rows: Vec<Vec<f64>> = (1..=8).flat_map(|n| [vec![n as f64, 0.0, 1.0], vec![0.0, n as f64, 1.0]]).collect();
    let t: Vec<f64> = rows.iter().map(|r| 0.2 * r[0] + 3.5 * r[1] + 1.0).collect();
    let f = fit_rel(&rows, &t);
    for (got, want) in f.beta.iter().zip([0.2, 3.5, 1.0]) {
        assert!((got - want).abs() < 1e-9, "{:?}", f.beta);
    }
    // Part of each time already priced (an equation taking some constants from another's fit):
    // t = 3 n + 0.5 m + 7 with the 0.5 m given, so only 3 and 7 are fitted.
    let rows: Vec<(f64, f64)> = (1..=12).map(|i| (i as f64, ((i * 5) % 7 + 1) as f64)).collect();
    let t: Vec<f64> = rows.iter().map(|&(n, m)| 3.0 * n + 0.5 * m + 7.0).collect();
    let x: Vec<Vec<f64>> = rows.iter().map(|&(n, _)| vec![n, 1.0]).collect();
    let fixed: Vec<f64> = rows.iter().map(|&(_, m)| 0.5 * m).collect();
    let f = crate::tuning::tuner::fit::fit_rel_over(&x, &t, &fixed);
    assert!((f.beta[0] - 3.0).abs() < 1e-9 && (f.beta[1] - 7.0).abs() < 1e-9 && f.rel_rms < 1e-12, "{:?}", f.beta);
    // A time between two powers (2.5 n^1.63, as recursive Karatsuba over schoolbook leaves) is
    // fitted linearly by weights on the two: n^log2(3) and n^2, in place of a fitted exponent.
    let ns: Vec<f64> = (0..24).map(|i| 96.0 * 2f64.powf(i as f64 / 6.0)).collect();
    let t: Vec<f64> = ns.iter().enumerate().map(|(i, n)| 2.5 * n.powf(1.63) * noise(i)).collect();
    let x: Vec<Vec<f64>> = ns.iter().map(|n| vec![n.powf(1.584962501), n * n]).collect();
    let f = fit_rel(&x, &t);
    assert!(f.dropped.is_empty() && f.rel_rms < 0.015, "{:?} rms {}", f.beta, f.rel_rms);
    // The standard errors say how well each constant is known: the n^2 weight, a small part of
    // every time, far less well than the main term's.
    assert!(f.se[0] / f.beta[0] < 0.02 && f.se[1] / f.beta[1] > f.se[0] / f.beta[0], "{:?} {:?}", f.beta, f.se);
    // Judged on shapes it didn't fit: a right form predicts the held-out half as well as it
    // fits, a wrong one (a line through a parabola) worse than it fits.
    let held: Vec<bool> = (0..ns.len()).map(|i| i % 2 == 1).collect();
    let good = fit_split(&x, &t, &vec![0.0; t.len()], &held);
    let (n_held, rms_held) = good.held.expect("some shapes were held out");
    assert!(n_held == 12 && rms_held < 0.02 && good.judged() == rms_held && good.accepted(), "{:?}", good.held);
    let line: Vec<Vec<f64>> = ns.iter().map(|n| vec![*n]).collect();
    let bad = fit_split(&line, &t, &vec![0.0; t.len()], &held);
    assert!(bad.held.unwrap().1 > ACCEPT_RMS && !bad.accepted(), "{:?}", bad.held);
    // The developer's analysis of a form: of t = 2 n^1.585 + 5 n with a junk term and a shift
    // offered too, the form to consider is the two terms the times are made of.
    let t: Vec<f64> = ns.iter().enumerate().map(|(i, n)| (2.0 * n.powf(1.584962501) + 5.0 * n) * noise(i)).collect();
    let x: Vec<Vec<f64>> = ns.iter().map(|n| vec![n.powf(1.584962501), *n, n.sqrt(), 1.0]).collect();
    let text = analysis("a test form", &["power", "linear", "root", "shift"], &x, &t, &vec![0.0; t.len()], &held);
    assert!(text.contains("the smallest within 0.2 points of the best") && text.contains("): power + linear at "), "{text}");
}

// A fit above the acceptance line reads REJECTED on its model's first form (report, propose and
// wait) and NOISY once the model has been revised (accepted as it is, a marked problem); a fit
// inside the line carries neither mark, whichever way it is asked.
#[cfg(feature = "tune")]
#[test]
fn test_fit_evidence_marks() {
    use crate::tuning::tuner::fit::{Fit, ACCEPT_RMS, NOISY, REJECTED};
    let fit = |rms: f64| Fit { beta: vec![1.0], se: vec![0.1], cond: 1.0, rel_rms: rms, max_rel: 2.0 * rms, worst: None, n: 100, held: None, noise: None, dropped: vec![] };
    let (good, bad) = (fit(ACCEPT_RMS), fit(ACCEPT_RMS + 0.02));
    for e in [good.evidence(), good.evidence_revised()] {
        assert!(!e.contains(REJECTED) && !e.contains(NOISY), "{e}");
    }
    assert!(bad.evidence().contains(REJECTED) && !bad.evidence().contains(NOISY), "{}", bad.evidence());
    assert!(bad.evidence_revised().contains(NOISY) && !bad.evidence_revised().contains(REJECTED), "{}", bad.evidence_revised());
    // Where shapes were held out, theirs is the rms the line reads, whatever the fitted ones'.
    let held_bad = Fit { held: Some((50, ACCEPT_RMS + 0.01)), ..good.clone() };
    let held_good = Fit { held: Some((50, ACCEPT_RMS)), ..bad.clone() };
    assert!(held_bad.evidence().contains(REJECTED) && !held_good.evidence().contains(REJECTED));
    // A term left out for solving negative is named in the evidence and rejects nothing.
    let short = Fit { dropped: vec![(0, -0.5, 0.1)], ..good };
    assert!(short.accepted() && !short.evidence().contains(REJECTED) && short.dropped_text(&["shift"]).contains("shift solved"));
}

// A fit's shapes are scattered over the area its algorithm competes on, and every other one is
// held out; its probes are timed twice, and the passes' difference is the noise.
#[cfg(feature = "tune")]
#[test]
fn test_fit_designs_cover_their_area() {
    use crate::tuning::tuner::design::{products, scatter};
    use crate::tuning::tuner::measure::pass_noise;
    // The points stay inside the area and fill it in both coordinates: every quarter of each
    // coordinate's log range holds about a quarter of them, as does each quadrant.
    let pts = scatter(0, 200, (10.0, 1000.0), (1.0, 8.0));
    // A report's points are further along the same sequence: none of a tune's.
    let unseen = scatter(crate::tuning::tuner::design::UNSEEN, 200, (10.0, 1000.0), (1.0, 8.0));
    assert!(unseen.iter().all(|p| !pts.contains(p)));
    assert!(pts.iter().all(|&(x, y)| (10.0..=1000.0).contains(&x) && (1.0..=8.0).contains(&y)));
    let (u, v): (Vec<f64>, Vec<f64>) = pts.iter().map(|&(x, y)| ((x / 10.0).log(100.0), y.log(8.0))).unzip();
    for q in 0..4 {
        let band = |w: &[f64]| w.iter().filter(|&&z| (0.25 * q as f64..0.25 * (q + 1) as f64).contains(&z)).count();
        assert!((40..=60).contains(&band(&u)) && (40..=60).contains(&band(&v)), "quarter {q}: {} {}", band(&u), band(&v));
        let quadrant = pts.iter().zip(u.iter().zip(&v)).filter(|(_, (&a, &b))| (a < 0.5) == (q % 2 == 0) && (b < 0.5) == (q / 2 == 0)).count();
        assert!((40..=60).contains(&quadrant), "quadrant {q}: {quadrant}");
    }
    // Three dials: each coordinate's quarters and each octant take their share.
    let pts3 = crate::tuning::tuner::design::scatter3(0, 400, (4.0, 1024.0), (1.0, 2.0), (1.0, 2.0));
    let unit: Vec<[f64; 3]> = pts3.iter().map(|&(x, y, z)| [(x / 4.0).log(256.0), y.log2(), z.log2()]).collect();
    assert!(unit.iter().all(|p| p.iter().all(|c| (0.0..=1.0).contains(c))));
    for c in 0..3 {
        for q in 0..4 {
            let n = unit.iter().filter(|p| (0.25 * q as f64..0.25 * (q + 1) as f64).contains(&p[c])).count();
            assert!((85..=115).contains(&n), "coordinate {c}, quarter {q}: {n}");
        }
    }
    for o in 0..8 {
        let n = unit.iter().filter(|p| (0..3).all(|c| (p[c] < 0.5) == (o >> c & 1 == 0))).count();
        assert!((35..=65).contains(&n), "octant {o}: {n}");
    }
    // Held out two by two, a dozen points still put both halves of the third coordinate (whose
    // step is near a half) in each half of the split; by parity they don't.
    let d3 = crate::tuning::tuner::design::Design::of3(0, 12, (4.0, 1024.0), (1.0, 2.0), (1.0, 2.0), |x, y, z| Some(((x * 1e6) as u64, (y * 1e6) as u64, (z * 1e6) as u64)));
    let low = |held: &[bool], half: bool| d3.shapes.iter().zip(held).filter(|(s, &h)| h == half && s.2 < 1_414_214).count();
    assert!(low(&d3.held, false).abs_diff(low(&d3.held, true)) >= 3, "by parity the halves are lopsided: {} {}", low(&d3.held, false), low(&d3.held, true));
    let pairs = crate::tuning::tuner::design::Design::of3(0, 12, (4.0, 1024.0), (1.0, 2.0), (1.0, 2.0), |x, y, z| Some(((x * 1e6) as u64, (y * 1e6) as u64, (z * 1e6) as u64))).in_pairs();
    assert_eq!(pairs.held.iter().filter(|&&h| h).count(), 6);
    assert!((2..=4).contains(&low(&pairs.held, false)) && (2..=4).contains(&low(&pairs.held, true)), "{} {}", low(&pairs.held, false), low(&pairs.held, true));
    // A design keeps only its algorithm's shapes, each once, and holds out every other one; the
    // held-out half spans the same sizes and ratios as the fitted half.
    let d = products(120, (24, 96), (1.0, 2.0), |l, s| s > (l + 1) / 2);
    assert!(d.shapes.len() > 80 && d.shapes.len() == d.held.len());
    assert!(d.shapes.iter().all(|&(l, s)| (24..=96).contains(&s) && l >= s && s > (l + 1) / 2));
    assert!(d.shapes.iter().enumerate().all(|(i, a)| !d.shapes[..i].contains(a)));
    for half in [false, true] {
        let of: Vec<(usize, usize)> = d.shapes.iter().zip(&d.held).filter(|(_, &h)| h == half).map(|(s, _)| *s).collect();
        let (lo, hi) = (of.iter().map(|x| x.1).min().unwrap(), of.iter().map(|x| x.1).max().unwrap());
        let ratios: Vec<f64> = of.iter().map(|&(l, s)| l as f64 / s as f64).collect();
        assert!(of.len() * 2 + 1 >= d.shapes.len() && lo <= 30 && hi >= 85, "held {half}: {} shapes, s {lo}-{hi}", of.len());
        assert!(ratios.iter().any(|&r| r < 1.15) && ratios.iter().any(|&r| r > 1.8), "held {half}");
    }
    // Two passes 2% apart at every probe read 2% / √2 of noise each; equal passes read none.
    let a = [100.0, 2000.0, 3.0e6];
    assert!(pass_noise(&a, &a) == 0.0);
    let n = pass_noise(&a, &a.map(|x| x * 1.02));
    assert!((n - 1.02f64.ln() / 2f64.sqrt()).abs() < 1e-12, "{n}");
}

// The synthetic family the series section is tuned on: each member's shift (`Synth::decay`)
// gives about the terms asked for, whether the convergence is the shift's alone, a
// denominator's or a factorial's; and its series sum to the same value by Horner, by rect where
// it can run and by binary splitting, to within their error.
#[cfg(feature = "tune")]
#[test]
fn test_synthetic_series_converge_and_agree() {
    use crate::tuning::tuner::synth::{x_of, Big, Den, Fact, Geo, Odd, Synth};
    use crate::utils::series::{series_eval_dyn_at, Bbp, Engel, Hyper, Series, SeriesAlg, SeriesNode};
    use crate::utils::utils::{buf_len, cmp_buf, sub_buf};
    fn check<C: Series>(what: &str, ctx: C, w: usize, t: u64) {
        let zeros = ctx.zeros();
        let len = w - zeros;
        let (a, b) = ctx.range(64 * w as u64);
        // The terms asked for, give or take the floor in the shift and the tail bound's slack.
        assert!((b - a) * 2 >= t && b - a <= t + t / 4 + 3, "{what}: {} terms for {t}", b - a);
        let ok = <C::Dyn as SeriesNode>::eligible(&ctx.dyn_ctx(), a, b, len, zeros);
        assert!(ok.horner, "{what}");
        let mut tree = vec![0u64; len];
        series_eval_dyn_at(&ctx, &mut tree, SeriesAlg::BinSplit);
        assert!(buf_len(&tree) > 0, "{what}: a sum of nothing");
        for alg in [SeriesAlg::Horner, SeriesAlg::Rect] {
            if !ok.allows(alg) {
                continue;
            }
            let mut out = vec![0u64; len];
            series_eval_dyn_at(&ctx, &mut out, alg);
            let (hi, lo) = if cmp_buf(&out, &tree) == std::cmp::Ordering::Less { (&tree, &out) } else { (&out, &tree) };
            let mut d = hi.clone();
            sub_buf(&mut d, lo);
            assert!(buf_len(&d) <= 1 && d[0] <= 2, "{what}: {alg:?} is {} units off", d[0]);
        }
    }
    fn hyper<S: crate::utils::series::HyperSeries + Synth>((w, t, xb): (usize, u64, u64)) {
        let x = if xb == 0 { vec![1] } else { x_of(xb, 7 * xb + 1) };
        assert!(xb == 0 || crate::utils::series::bits(&x) == xb && x[0] & 1 == 1);
        check(&format!("hyper {} w{w} t{t} x{xb}", S::name()), Hyper::<S>::new(&x, xb + S::decay(64 * w as u64, t)).unwrap(), w, t);
    }
    for shape in [(12, 30, 200), (40, 400, 64), (40, 20, 2560), (30, 6, 0), (60, 900, 1000)] {
        hyper::<Geo<8>>(shape);
        hyper::<Geo<56>>(shape);
        hyper::<Fact>(shape);
    }
    // A 32-bit denominator fixes 31 bits a term before any shift: at most a term to those.
    for shape in [(12, 10, 200), (40, 60, 64), (40, 20, 2560), (30, 6, 0)] {
        hyper::<Den<32>>(shape);
    }
    for (w, t) in [(12, 30), (40, 60), (60, 100)] {
        check(&format!("engel fact w{w} t{t}"), Engel::<Fact>::new(Fact::decay(64 * w as u64, t)), w, t);
        check(&format!("engel den32 w{w} t{t}"), Engel::<Den<32>>::new(<Den<32>>::decay(64 * w as u64, t)), w, t);
        check(&format!("bbp odd w{w} t{t}"), Bbp::<Odd>::new(Odd::decay(64 * w as u64, t)), w, t);
        check(&format!("bbp big40 w{w} t{t}"), Bbp::<Big<40>>::new(<Big<40>>::decay(64 * w as u64, t)), w, t);
    }
}

// The rule every table is measured by (`tuner/tables.rs`), on made-up timings: an index is the
// table's while the decision costs more than the share of the call, and takes the fastest
// algorithm timed there; the pick before stays within the tie band; a pick of one cell between
// two of another is dropped; the row ends at the first index within the share; and a larger
// share ends it sooner, from the same timings.
#[cfg(feature = "tune")]
#[test]
fn test_table_rule_on_made_up_timings() {
    use crate::tuning::tuner::tables::{assemble, row, rows_at, Cell, Row, SHARE};
    let cell = |size: u32, ns: &[(u32, f64)], decide: f64| Cell::one(size, ns.to_vec(), Some(decide));
    // A decision of 50 ns. Algorithm 0 is the faster to 4 terms, 1 from 6; at 5 they are within
    // 2% of each other. The call passes 1000 ns (20 decisions) at 9.
    let cells = vec![
        cell(1, &[(0, 100.0), (1, 180.0)], 50.0),
        cell(2, &[(0, 200.0), (1, 260.0)], 50.0),
        cell(4, &[(0, 400.0), (1, 420.0)], 50.0),
        cell(5, &[(0, 505.0), (1, 500.0)], 50.0),
        cell(6, &[(0, 640.0), (1, 560.0)], 50.0),
        cell(8, &[(0, 900.0), (1, 760.0)], 50.0),
        cell(9, &[(0, 1100.0), (1, 1001.0)], 50.0),
        cell(12, &[(0, 1500.0), (1, 1300.0)], 50.0),
    ];
    assert_eq!(SHARE, 0.05);
    assert_eq!(row(&cells, Some(9), 0.03, SHARE), [(1, 0), (6, 1), (9, 9)], "0 holds at 5 within the tie band, and 9 is the equations'");
    assert_eq!(row(&cells, Some(9), 0.0, SHARE), [(1, 0), (5, 1), (9, 9)], "with no tie band the faster at 5 takes it");
    // At a share of 10% the call only has to pass 500 ns, which the cell at 5 misses by a hair
    // (500 is the fastest there, and 50 / 500 is the share exactly: within it).
    assert_eq!(row(&cells, Some(9), 0.03, 0.10), [(1, 0), (5, 9)]);
    assert_eq!(row(&cells, Some(9), 0.03, 0.60), [(1, 9)], "a decision under the share from the first index: the row is the equations'");
    // A table with no equations (it decides at every size) reads every cell.
    assert_eq!(row(&cells, None, 0.03, SHARE), [(1, 0), (6, 1)]);
    // A pick that lasts one cell between two of another is dropped.
    let blip = vec![
        cell(1, &[(0, 100.0), (1, 150.0)], 50.0),
        cell(2, &[(0, 300.0), (1, 200.0)], 50.0),
        cell(3, &[(0, 300.0), (1, 450.0)], 50.0),
        cell(4, &[(0, 400.0), (1, 600.0)], 50.0),
    ];
    assert_eq!(row(&blip, Some(9), 0.03, SHARE), [(1, 0)]);
    // An index nothing can run at, or with no decision to read, never ends a row on its own.
    let bare = vec![Cell::one(1, vec![(0, 10.0)], None), Cell::one(2, vec![(0, 1e6)], None)];
    assert_eq!(row(&bare, Some(9), 0.03, SHARE), [(1, 0)]);
    assert!(!bare[1].past(SHARE) && cells[6].past(SHARE) && !cells[5].past(SHARE));
    // A cell's pick and its cost, the time under it over the fastest algorithm's: 1 where one
    // call was timed.
    assert_eq!(cells[0].fastest(), Some((0, 1.0)));
    assert_eq!((cells[0].cost(1), cells[0].cost(7)), (Some(1.8), None));
    assert_eq!(cells[0].share(), Some(0.5));
    // An index that stands for several kinds of call times one of each, and its pick is the one
    // that costs them least together: here 0, at 1.0 and 1.2 of the two calls' own bests, where
    // 1 is at 1.5 and 1.0. The decision's share is the largest among them, so the index is the
    // equations' only when it is within the share of every call.
    use crate::tuning::tuner::tables::Timed;
    let call = |ns: &[(u32, f64)], decide: f64| Timed { ns: ns.to_vec(), decide: Some(decide), note: Vec::new() };
    let two = Cell { size: 3, of: vec![call(&[(0, 100.0), (1, 150.0)], 50.0), call(&[(0, 240.0), (1, 200.0)], 50.0)], least: 1.0 };
    assert_eq!(two.fastest().map(|f| f.0), Some(0));
    assert!((two.cost(0).unwrap() - 1.1).abs() < 1e-12 && (two.cost(1).unwrap() - 1.25).abs() < 1e-12);
    assert_eq!(two.share(), Some(0.5));
    assert!(!two.past(0.3) && two.past(0.5), "the first call's decision is half of it, the second's a quarter");
    // A pick one of the calls can't run leaves that call to the equations: their pick taken as
    // its fastest, and the decision paid. Here 1 still costs least (1.0 and 1.1 against 1.67
    // and 1.0).
    let part = Cell { size: 3, of: vec![call(&[(0, 100.0), (1, 60.0)], 10.0), call(&[(0, 100.0)], 10.0)], least: 1.0 };
    assert_eq!(part.fastest().map(|f| f.0), Some(1));
    assert!((part.cost(1).unwrap() - 1.05).abs() < 1e-12 && (part.cost(0).unwrap() - (100.0 / 60.0 + 1.0) / 2.0).abs() < 1e-12);
    assert!(!Cell::one(1, vec![(0, 10.0)], None).bare() && Cell { size: 1, of: Vec::new(), least: 1.0 }.bare() && Cell::one(1, Vec::new(), Some(5.0)).bare());
    // Reading the equations is a pick of an index too, at its own cost: the decision paid on
    // every call. With one call the fastest algorithm always costs less (1 against 1 + its
    // share). Where an index stands for calls that no one algorithm serves, it costs least:
    // here either algorithm is 3x off for one call (a cost of 2), and the decision is a fifth
    // of each (1.2), so the index is the equations', and the row goes on to the next.
    let split = |size: u32| Cell { size, of: vec![call(&[(0, 100.0), (1, 300.0)], 20.0), call(&[(0, 300.0), (1, 100.0)], 20.0)], least: 1.0 };
    assert!((split(1).read().unwrap() - 1.2).abs() < 1e-12 && (split(1).cost(0).unwrap() - 2.0).abs() < 1e-12);
    assert_eq!(cells[0].read(), Some(1.5));
    let mixed = vec![cell(1, &[(0, 100.0), (1, 180.0)], 50.0), split(2), split(3), cell(4, &[(0, 400.0), (1, 300.0)], 50.0), cell(40, &[(0, 4000.0), (1, 3000.0)], 50.0)];
    assert_eq!(row(&mixed, Some(9), 0.03, SHARE), [(1, 0), (2, 9), (4, 1), (40, 9)]);
    // Such a row doesn't end at its first index, though it may start as the equations'.
    let starts = Row { key: 0, sizes: vec![2, 3, 4], points: row(&mixed[1..4], Some(9), 0.03, SHARE), cells: mixed[1..4].to_vec() };
    assert_eq!(starts.points, [(2, 9), (4, 1)]);
    assert!(!starts.all_equations(Some(9)));
    let ends = Row { key: 0, sizes: vec![40], points: row(&mixed[4..], Some(9), 0.03, SHARE), cells: mixed[4..].to_vec() };
    assert!(ends.points == [(40, 9)] && ends.all_equations(Some(9)));
    // A step of a coarse grid stands for calls shorter than the one timed: the rule is read at
    // the shortest. At 12 terms timed for a step from 10, the call that counts is 10 / 12 of it.
    let step = Cell { least: 10.0 / 12.0, ..cell(10, &[(0, 1100.0)], 50.0) };
    assert!(cell(10, &[(0, 1100.0)], 50.0).past(SHARE) && !step.past(SHARE) && (step.share().unwrap() - 50.0 / (1100.0 * 10.0 / 12.0)).abs() < 1e-12);
    // Rows into a table: a row that repeats the one before it is left out where the keys are of
    // one family, and the same rows at a larger share are a smaller table.
    let rows: Vec<Row> = [16u32, 17, 32].iter().map(|&key| Row { key, sizes: vec![1, 2, 4, 5, 6, 8, 9, 12], points: row(&cells, Some(9), 0.03, SHARE), cells: cells.clone() }).collect();
    let family = |a: u32, b: u32| a / 16 == b / 16;
    assert_eq!(assemble(&rows, family), [16, 1, 0, 16, 6, 1, 16, 9, 9, 32, 1, 0, 32, 6, 1, 32, 9, 9]);
    assert_eq!(assemble(&rows_at(&rows, Some(9), 0.03, 0.10), family), [16, 1, 0, 16, 5, 9, 32, 1, 0, 32, 5, 9]);
    // A row whose grid ended inside the table hands what is past the grid to the equations.
    let short = vec![Row { key: 1, sizes: vec![1, 2], points: vec![], cells: cells[..2].to_vec() }];
    assert_eq!(assemble(&rows_at(&short, Some(9), 0.03, SHARE), family), [1, 1, 0, 1, 3, 9]);
}

// The rule is system-wide: the schema's table knobs are exactly the tables of the dispatchers
// that read fitted equations (`dispatch::TABLES`), each measured by `tables::measure` and
// checked by `cargo tune dispatch`; and no knob is derived from the fits any more.
#[cfg(feature = "tune")]
#[test]
fn test_every_table_is_a_dispatchers() {
    use crate::tuning::tuner::dispatch::TABLES;
    let flat = resolve::flat(tuning::schema());
    let tables: Vec<&str> = flat.iter().filter(|f| f.knob.kind == Kind::Table).map(|f| f.path.as_str()).collect();
    let mut listed: Vec<&str> = TABLES.iter().map(|t| t.1).collect();
    listed.sort_unstable();
    let mut found = tables.clone();
    found.sort_unstable();
    assert_eq!(found, listed, "a table knob and a dispatcher's table are the same thing");
    for f in &flat {
        assert_eq!(matches!(f.knob.ty, Ty::Table { .. }), f.knob.kind == Kind::Table, "{}", f.path);
    }
}

// An equation is fitted on the calls it is read for: a fit after a section's tables leaves out
// what those tables decide. Division at tables made by hand (the entry's own table, a pick that
// can't take the shape handed on, past a row's extent), and the series by what its table names
// for a workload, whatever the workload's own decision costs.
#[cfg(feature = "tune")]
#[test]
fn test_fits_leave_out_what_tables_decide() {
    use crate::tuning::table::{row_index, Table2};
    use crate::tuning::tuner::sections::div::read_for;
    use crate::tuning::tuner::sections::series::{Rows, SeriesTiming};
    use crate::utils::series::{Guarded, SeriesAlg};
    let table = |t: &'static [u32]| Table2 { t, rows: Box::leak(row_index(t).into_boxed_slice()) };
    // Quotient only: at divisors of 8 to 39 limbs Knuth for quotients to 19 limbs, NR to 99, the
    // equations from 100; from 40, BZ; from 60, the equations. With remainder: Knuth at every
    // quotient for divisors to 15 limbs, the equations past.
    let quo = table(Box::leak(Box::new([8, 2, 0, 8, 20, 2, 8, 100, 3, 40, 2, 1, 60, 1, 3])));
    let rem = table(Box::leak(Box::new([1, 1, 0, 16, 1, 3])));
    let (tables, leaf) = ([&quo, &rem], 99);
    assert!(!read_for(false, 12, 20, tables, leaf), "the table names Knuth");
    assert!(!read_for(false, 50, 20, tables, leaf), "the table names NR");
    assert!(read_for(false, 100, 20, tables, leaf), "past the row's extent the equations are read");
    assert!(read_for(false, 30, 70, tables, leaf), "a row of the equations");
    assert!(read_for(false, 30, 45, tables, leaf), "BZ under its leaf can't take the shape: handed to the equations");
    assert!(!read_for(false, 30, 45, tables, 40), "and past its leaf the table's BZ stands");
    assert!(!read_for(true, 100, 12, tables, leaf) && read_for(true, 100, 16, tables, leaf), "with the remainder, that entry's table");
    assert!(read_for(true, 12, 20, tables, leaf) && !read_for(false, 12, 20, tables, leaf), "each entry by its own table");

    // A series: its fastest algorithm takes 1000 ns, and the table names an algorithm or doesn't.
    let timing = |decide: f64, small: Option<SeriesAlg>| SeriesTiming {
        name: String::new(),
        held: false,
        ns: [1000.0, f64::INFINITY, 2000.0],
        auto: f64::INFINITY,
        tree: f64::INFINITY,
        entry: 0.0,
        decide,
        parts: [None; 3],
        gained: Vec::new(),
        small,
        area: [true; 3],
        guard: Guarded { rect_first: false },
        rho: 0.0,
        beta: 0.0,
        dials: [0.0; 13],
        class: None,
    };
    // By its own decision (a first fit, before a table): within the share or not.
    assert!(timing(40.0, Some(SeriesAlg::Horner)).fitted(Rows::Share(0.05)) && !timing(60.0, None).fitted(Rows::Share(0.05)));
    assert!(timing(60.0, None).fitted(Rows::Share(0.10)));
    // By the table in force: a series it names an algorithm for is out though its decision is
    // within the share (its kind's row goes on for another member), and one it hands to the
    // equations is in though its decision is past it.
    assert!(!timing(40.0, Some(SeriesAlg::Horner)).fitted(Rows::Table), "the table decides: no equation is read");
    assert!(!timing(40.0, Some(SeriesAlg::Rect)).fitted(Rows::Table));
    assert!(timing(60.0, None).fitted(Rows::Table), "handed to the equations: they are read");
    assert!(timing(10.0, None).fitted(Rows::Table));
}

// Every class of the small-series table that a series can have is measured on a synthetic
// series of that class: the algorithms that can sum it (binary splitting always, rect with an x
// of two limbs or more) and the decision.
#[cfg(feature = "tune")]
#[test]
fn test_series_table_cells_are_of_their_class() {
    use crate::tuning::tuner::sections::series::{member_cell, member_kind, MEMBERS};
    use crate::utils::series::SMALL_KINDS;
    let (mut seen, mut kinds) = (0, vec![0usize; SMALL_KINDS]);
    for member in 0..MEMBERS {
        // A member is a width of x (ten of them, the first two BBP and Engel), q's bits (four)
        // and an r past 1 or not.
        let (rc, width) = (member % 2, member / 2 / 4);
        for (len, n) in [(1usize, 1u64), (20, 30), (300, 5)] {
            let mut probes = Vec::new();
            let plan = member_cell(member, len, n, &mut probes);
            assert_eq!(plan.is_some(), !(width < 2 && rc == 1), "member {member} at {len} limbs, {n} terms");
            let Some(plan) = plan else { continue };
            assert!(plan.decide && !plan.harness && plan.picks.contains(&2) && plan.picks.contains(&0), "member {member}: {plan:?}");
            assert_eq!(plan.picks.contains(&1), width >= 5, "member {member}: rect needs an x of two limbs");
            assert_eq!(probes.len(), plan.picks.len() + 1);
            // Its series' numbers and two ratios are kept with it, for the dump.
            assert_eq!(plan.note.len(), 15);
            // Each probe runs: the algorithms forced over the explicit range, and the decision.
            for p in probes.iter_mut() {
                (p.run)();
            }
            seen += 1;
        }
        if !(width < 2 && rc == 1) {
            kinds[member_kind(member)] += 1;
        }
    }
    assert_eq!(seen, 3 * 72);
    // Every kind of the table is timed on a spread of members: BBP and Engel on their four
    // sizes of q, Hyper on eight a width of x (two widths to a limb and at 3-16 limbs, one else).
    assert_eq!(kinds, [4, 4, 8, 16, 8, 16, 8, 8]);
}

// Which cores the tuner pins itself to on Linux, from each CPU's sysfs rating: the ones within
// 15% of the best, and nothing at all on a uniform machine.
#[cfg(feature = "tune")]
#[test]
fn test_fastest_cores_from_ratings() {
    use crate::tuning::tuner::measure::fastest_of;
    let rate = |r: &[u64]| -> Vec<(usize, u64)> { r.iter().cloned().enumerate().collect() };
    assert_eq!(fastest_of(&[]), None);
    assert_eq!(fastest_of(&rate(&[3_600_000; 8])), None, "a uniform CPU");
    // A hybrid Intel CPU: eight performance threads, two of them favored, and four efficiency cores.
    let intel = rate(&[5_300_000, 5_300_000, 5_100_000, 5_100_000, 5_100_000, 5_100_000, 5_100_000, 5_100_000, 3_900_000, 3_900_000, 3_900_000, 3_900_000]);
    assert_eq!(fastest_of(&intel), Some((0..8).collect()));
    // ARM capacities: one prime core, three big, four little.
    assert_eq!(fastest_of(&rate(&[250, 250, 250, 250, 800, 800, 800, 1024])), Some(vec![7]));
    assert_eq!(fastest_of(&rate(&[1024, 1024, 512, 512])), Some(vec![0, 1]));
    // Ratings a few percent apart are one kind of core.
    assert_eq!(fastest_of(&rate(&[4_000_000, 3_900_000, 3_800_000, 3_700_000])), None);
}

// The end-to-end gate's verdict, on made-up rounds (ln of time after over time before): values
// that change nothing are never dropped; a section is dropped when its mean is past the tie band
// with an interval that excludes 1, or when an exact output differs; one slow item among
// unchanged ones is listed, not a reason to drop. And which workloads each section's gate holds.
#[cfg(feature = "tune")]
#[test]
fn test_check_gate_verdicts() {
    use crate::tuning::tuner::check::{exact, in_gate, verdict, Rounds};
    let item = |name: &str, logs: &[f64], same: bool| Rounds { section: "mul", name: name.to_string(), logs: logs.to_vec(), same };
    let noise = [0.004, -0.003, 0.001, -0.002, 0.0, 0.002, -0.002];
    let unchanged = vec![item("mul_dyn 30x30", &noise, true), item("sqr_dyn 60", &noise.map(|x| -x), true)];
    for eps in [0.0, 0.005, 0.03] {
        let v = verdict(&unchanged, eps);
        assert!(!v.dropped && v.slower.is_empty() && v.differ.is_empty(), "eps {eps}: {v:?}");
    }

    // 5% slower on every round.
    let slow = [0.050, 0.047, 0.052, 0.049, 0.051];
    let v = verdict(&[item("mul_dyn 30x30", &slow, true)], 0.01);
    assert!(v.dropped && v.slower.len() == 1 && (v.all.ratio - 1.051).abs() < 0.002 && v.all.lo > 1.04, "{v:?}");
    assert!(!verdict(&[item("mul_dyn 30x30", &slow, true)], 0.10).dropped, "inside a 10% tie band");
    // Slower on average, but the rounds disagree too much to say so.
    let noisy = [0.20, -0.15, 0.18, -0.12, 0.10];
    let v = verdict(&[item("mul_dyn 30x30", &noisy, true)], 0.01);
    assert!(v.all.ratio > 1.01 && v.all.lo < 1.0 && !v.dropped, "{v:?}");
    // One item 5% slower among nine unchanged: the mean stays inside the band.
    let mut mixed: Vec<Rounds> = (0..9).map(|i| item(&format!("sqr_dyn {i}"), &noise[..5], true)).collect();
    mixed.push(item("mul_dyn 400x400", &slow, true));
    let v = verdict(&mixed, 0.01);
    assert!(!v.dropped && v.slower.len() == 1 && v.slower[0].name == "mul_dyn 400x400", "{v:?}");

    // An exact output that differs is a correctness failure, whatever the times say; an entry
    // specified only to within its error may differ with the algorithm chosen.
    let v = verdict(&[item("div_rem_dyn 200/100", &noise, false)], 0.03);
    assert!(v.dropped && v.differ == ["div_rem_dyn 200/100"], "{v:?}");
    assert!(!verdict(&[item("hi_mul_dyn 300", &noise, false)], 0.03).dropped);
    for name in ["mul_dyn 30x30", "mul_static<2048> 64x64", "sqr_dyn 60", "sqr_static<8192> 60", "sqr_mod_bnm1_dyn 300", "div_dyn 120/60", "div_rem_static<2048> 200/100"] {
        assert!(exact(name), "{name}");
    }
    for name in ["hi_mul_dyn 30", "hi_sqr_dyn 30", "mid_mul_dyn 100", "rcp_dyn 100", "sqrt_dyn 16", "sqrt_only_dyn 16", "exp_dyn 8", "ln2_static<1100> 1030"] {
        assert!(!exact(name), "{name}");
    }
    assert!(verdict(&[], 0.01).items == 0 && !verdict(&[], 0.01).dropped);

    // A section's gate: the workloads that read its knobs, directly or through a section built on it.
    let s = crate::tuning::schema();
    for sec in ["mul", "div", "divgrid", "sqrt", "series", "consts", "trans"] {
        assert!(in_gate(s, "mul", sec), "everything is built on mul: {sec}");
    }
    assert!(in_gate(s, "div", "divgrid") && in_gate(s, "div", "sqrt") && in_gate(s, "div", "trans") && !in_gate(s, "div", "mul"));
    assert!(in_gate(s, "bin_split", "series") && in_gate(s, "bin_split", "consts") && in_gate(s, "bin_split", "trans") && !in_gate(s, "bin_split", "sqrt"));
    assert!(in_gate(s, "ln", "trans") && !in_gate(s, "ln", "consts") && !in_gate(s, "ln", "sqrt"));
    // The series' own items are series evaluations, which no transcendental knob reaches; the
    // series section's gate holds them and everything built on it.
    assert!(!in_gate(s, "ln", "series") && !in_gate(s, "exp", "series"));
    assert!(in_gate(s, "series", "series") && in_gate(s, "series", "trans") && in_gate(s, "series", "consts") && !in_gate(s, "series", "sqrt"));
}

#[cfg(feature = "tune")]
#[test]
fn test_tuner_output_resolves() {
    use crate::tuning::tuner::out::{merge_into, section, Tuned};
    let path = std::env::temp_dir().join(format!("bigbits-tuning-test-{}.toml", std::process::id()));
    let _ = std::fs::remove_file(&path);
    // The values the file will be resolved over (the defaults), which its upstream hash records.
    let target = Target { arch: std::env::consts::ARCH, os: std::env::consts::OS };
    let values = resolve::resolve(tuning::schema(), &[], &target).unwrap().values;
    let mul = tuning::schema().iter().find(|s| s.name == "mul").unwrap();
    let tuned = vec![Tuned { key: "school_scale", value: Value::Float(0.31), evidence: "fit: test".into() }];
    merge_into(&path, section(mul, &tuned, "quick", &values), 8).unwrap();
    let f = file::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(f.sections[0].name, "meta");
    let r = resolve::resolve(tuning::schema(), &[Layer { name: "test", file: &f }], &target).unwrap();
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    let i = tuning::index("mul.school_scale").unwrap();
    assert_eq!(r.values[i], Value::Float(0.31));
    assert_eq!(f.section("mul").unwrap().get("school_scale").unwrap().comment.as_deref(), Some("fit: test"));
}

// --- The file format ---

const SAMPLE: &str = "# BigBits tuning file.\n# Second line.\n[meta]\nformat = 1\ncpu = \"Apple M1\"\n\n\
[sqrt]\nversion = 1\nderive = 2\nscale = 0.41        # fit: 9 shapes\nbig = 1_000_000\nneg = -3\n\
ext = inf\nflag = true\nexp = 1.5e-3\nstage = [1, 0,\n  12, 1]   # derived (table)\nsteps = [1, 5, 8, 10.5]\n";

#[test]
fn test_file_parses_the_format() {
    let f = file::parse(SAMPLE).unwrap();
    assert_eq!(f.header, vec!["BigBits tuning file.", "Second line."]);
    let meta = f.section("meta").unwrap();
    assert_eq!(meta.get("format").unwrap().value, Raw::Int(1));
    assert_eq!(meta.get("cpu").unwrap().value, Raw::Str("Apple M1".into()));
    let s = f.section("sqrt").unwrap();
    assert_eq!(s.get("scale").unwrap().value, Raw::Float(0.41));
    assert_eq!(s.get("scale").unwrap().comment.as_deref(), Some("fit: 9 shapes"));
    assert_eq!(s.get("big").unwrap().value, Raw::Int(1_000_000));
    assert_eq!(s.get("neg").unwrap().value, Raw::Int(-3));
    assert_eq!(s.get("ext").unwrap().value, Raw::Float(f64::INFINITY));
    assert_eq!(s.get("flag").unwrap().value, Raw::Bool(true));
    assert_eq!(s.get("exp").unwrap().value, Raw::Float(1.5e-3));
    let stage = s.get("stage").unwrap();
    assert_eq!(stage.value, Raw::Array(vec![Num::Int(1), Num::Int(0), Num::Int(12), Num::Int(1)]));
    assert_eq!(stage.comment.as_deref(), Some("derived (table)"));
    assert_eq!(
        s.get("steps").unwrap().value,
        Raw::Array(vec![Num::Int(1), Num::Int(5), Num::Int(8), Num::Float(10.5)])
    );
}

fn without_lines(mut f: file::File) -> file::File {
    for s in &mut f.sections {
        for e in &mut s.entries {
            e.line = 0;
        }
    }
    f
}

#[test]
fn test_file_round_trips() {
    let f = without_lines(file::parse(SAMPLE).unwrap());
    let text = file::write(&f);
    assert_eq!(without_lines(file::parse(&text).unwrap()), f, "{text}");
    // Floats keep their type and exact value; integral floats keep a decimal point.
    for v in [52.0, 0.1, 1e300, 5e-324, f64::INFINITY, 0.30000000000000004] {
        let t = file::float_text(v);
        assert_eq!(file::parse(&format!("[s]\nk = {t}\n")).unwrap().sections[0].entries[0].value, Raw::Float(v));
    }
}

#[test]
fn test_file_merge_replaces_whole_sections() {
    let mut a = file::parse("[meta]\nformat = 1\n[mul]\nx = 1\ny = 2\n[div]\nz = 3\n").unwrap();
    let b = file::parse("[mul]\nx = 5\n[sqrt]\nw = 4\n").unwrap();
    a.merge(&b);
    let names: Vec<&str> = a.sections.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["meta", "mul", "div", "sqrt"]);
    assert_eq!(a.section("mul").unwrap().entries.len(), 1, "y goes with the old section");
    assert_eq!(a.section("mul").unwrap().get("x").unwrap().value, Raw::Int(5));
}

#[test]
fn test_file_rejects_what_the_format_excludes() {
    for bad in [
        "k = 1\n",                       // before any section
        "[a]\nk = [1, [2]]\n",           // nested array
        "[a]\nk = [1, 2\n",              // unterminated array
        "[a]\nk = 1.\n",                 // TOML needs digits after the point
        "[a]\nk = \"a\\\"b\"\n",         // escapes
        "[a]\nk = 1\nk = 2\n",           // duplicate key
        "[a]\n[a]\n",                    // duplicate section
        "[a b]\n",                       // bad header
        "[a]\nk = {x = 1}\n",            // inline table
        "[a]\nk = nan\n",
        "[a]\nk = [\"s\"]\n",            // arrays hold numbers
    ] {
        assert!(file::parse(bad).is_err(), "accepted {bad:?}");
    }
}

// --- The resolver, on a schema of its own ---

const ALGS: &[Alg] = &[Alg { name: "school", min: 1 }, Alg { name: "kara", min: 4 }];

static TEST_SCHEMA: &[Section] = &[
    Section {
        name: "a",
        version: 2,
        derive: 3,
        after: &[],
        doc: "",
        knobs: &[
            Knob::usize("i", "OLD_I", 10).int_range(1, 100),
            Knob::f64("f", "", 0.5).float_range(0.0, 1.0),
            Knob::bool("b", "", false),
            Knob::usize("fol", "", 0).follows("a.i", 1),
            Knob::usize("adx", "", 20).arch("x86_64"),
            Knob::new("t", "", Ty::Table { algs: ALGS, measure: "n", inputs: 1 }, crate::tuning::schema_types::DefaultValue::Table(&[1, 0, 9, 1]))
                .kind(Kind::Table),
            Knob::new("st", "", Ty::U64, crate::tuning::schema_types::DefaultValue::Steps(&[(1, 5.0), (8, 11.0)]))
                .threads(Threads::Pool),
        ],
    },
    Section {
        name: "c",
        version: 1,
        derive: 1,
        after: &["a"],
        doc: "",
        knobs: &[Knob::u64("x", "", 3), Knob::u64("chain", "", 0).follows("a.fol", 2)],
    },
];

fn res(text: &str) -> Result<resolve::Resolved, Vec<String>> {
    let f = file::parse(text).unwrap();
    resolve::resolve(TEST_SCHEMA, &[Layer { name: "test", file: &f }], &Target { arch: "aarch64", os: "macos" })
}

fn val(r: &resolve::Resolved, path: &str) -> Value {
    let i = flat(TEST_SCHEMA).iter().position(|f| f.path == path).unwrap();
    r.values[i].clone()
}

#[test]
fn test_resolver_layers_and_follows() {
    resolve::check_schema(TEST_SCHEMA).unwrap();
    let r = resolve::resolve(TEST_SCHEMA, &[], &Target { arch: "aarch64", os: "macos" }).unwrap();
    assert_eq!(val(&r, "a.fol"), Value::Int(11));
    assert_eq!(val(&r, "c.chain"), Value::Int(13), "a Follows chain resolves in order");
    assert_eq!(val(&r, "a.st"), Value::Steps(vec![(1, 5.0), (8, 11.0)]));

    let r = res("[a]\nversion = 2\nderive = 3\ni = 20\nf = 1\n").unwrap();
    assert_eq!(val(&r, "a.i"), Value::Int(20));
    assert_eq!(val(&r, "a.f"), Value::Float(1.0), "a float knob reads an integer");
    assert_eq!(val(&r, "a.fol"), Value::Int(21), "Follows tracks the layered value");
    assert_eq!(val(&r, "c.chain"), Value::Int(23));
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);

    let r = res("[a]\nversion = 2\nderive = 3\nfol = 5\n").unwrap();
    assert_eq!(val(&r, "a.fol"), Value::Int(5), "a layer overrides a Follows default");

    // A later layer wins, key by key.
    let f1 = file::parse("[a]\nversion = 2\nderive = 3\ni = 20\nb = true\n").unwrap();
    let f2 = file::parse("[a]\nversion = 2\nderive = 3\ni = 30\n").unwrap();
    let layers = [Layer { name: "preset", file: &f1 }, Layer { name: "user", file: &f2 }];
    let r = resolve::resolve(TEST_SCHEMA, &layers, &Target { arch: "aarch64", os: "" }).unwrap();
    assert_eq!(val(&r, "a.i"), Value::Int(30));
    assert_eq!(val(&r, "a.b"), Value::Bool(true));
    let i = flat(TEST_SCHEMA).iter().position(|f| f.path == "a.i").unwrap();
    assert_eq!(r.origin[i], Origin::Layer(1));
}

#[test]
fn test_resolver_versions_fall_through() {
    let r = res("[a]\nversion = 1\nderive = 3\ni = 20\n").unwrap();
    assert_eq!(val(&r, "a.i"), Value::Int(10), "another version falls through whole");
    assert!(r.warnings.iter().any(|w| w.contains("falls through")));

    // A table's layout is the section's `derive` number: under another it would be misread, so
    // it falls through alone and the section's other values stay.
    let r = res("[a]\nversion = 2\nderive = 2\ni = 20\nt = [1, 0, 5, 1]\n").unwrap();
    assert_eq!(val(&r, "a.i"), Value::Int(20), "a stale layout number keeps the section's values");
    assert_eq!(val(&r, "a.t"), Value::Table(vec![1, 0, 9, 1]), "and drops its table");
    assert!(r.warnings.iter().any(|w| w.contains("derive")));
    let r = res("[a]\nversion = 2\nderive = 3\nt = [1, 0, 5, 1]\n").unwrap();
    assert_eq!(val(&r, "a.t"), Value::Table(vec![1, 0, 5, 1]));
}

#[test]
fn test_resolver_warnings_and_errors() {
    let r = res("[zzz]\nk = 1\n[a]\nversion = 2\nderive = 3\nnope = 1\nadx = 9\n").unwrap();
    assert_eq!(r.warnings.len(), 3, "{:?}", r.warnings);
    assert_eq!(val(&r, "a.adx"), Value::Int(20), "another architecture's value is ignored");
    for bad in [
        "[a]\nversion = 2\nderive = 3\ni = 0\n",     // below the valid range
        "[a]\nversion = 2\nderive = 3\ni = 101\n",   // above it
        "[a]\nversion = 2\nderive = 3\ni = 1.5\n",   // an integer knob doesn't read a float
        "[a]\nversion = 2\nderive = 3\nb = 1\n",     // nor a bool an integer
        "[a]\nversion = 2\nderive = 3\nt = [1, 0, 3, 1]\n", // kara picked below its minimum
        "[a]\nversion = 2\nderive = 3\nst = [2, 5]\n",      // steps start at one thread
        "[meta]\nformat = 2\n",
    ] {
        assert!(res(bad).is_err(), "accepted {bad:?}");
    }
}

#[test]
fn test_resolver_upstream_hash() {
    let fl = flat(TEST_SCHEMA);
    let r0 = resolve::resolve(TEST_SCHEMA, &[], &Target { arch: "aarch64", os: "" }).unwrap();
    let up = resolve::upstream_text(&fl, &r0.values, &TEST_SCHEMA[1]);
    assert!(up.starts_with("a:"), "{up}");
    let ok = res(&format!("[c]\nversion = 1\nderive = 1\nupstream = \"{up}\"\n")).unwrap();
    assert!(ok.warnings.is_empty(), "{:?}", ok.warnings);
    let stale = res(&format!("[a]\nversion = 2\nderive = 3\ni = 50\n[c]\nversion = 1\nderive = 1\nupstream = \"{up}\"\n")).unwrap();
    assert!(stale.warnings.iter().any(|w| w.contains("upstream")), "{:?}", stale.warnings);
}

// A machine's own tuning file is a second layer over the target's preset. Its sections replace
// the preset's, and only the top layer's upstream record of a section is checked: a preset's
// section under the machine's own is not stale, while one left standing over the machine's
// upstream values is. A file tuned on another target is refused.
#[test]
fn test_resolver_machine_layer_over_a_preset() {
    let fl = flat(TEST_SCHEMA);
    let target = Target { arch: "aarch64", os: "macos" };
    // A preset whose [c] was measured over its own [a].
    let preset_a = "[a]\nversion = 2\nderive = 3\nf = 0.25\n";
    let under_a = resolve::resolve(TEST_SCHEMA, &[Layer { name: "preset", file: &file::parse(preset_a).unwrap() }], &target).unwrap();
    let up_preset = resolve::upstream_text(&fl, &under_a.values, &TEST_SCHEMA[1]);
    let preset = file::parse(&format!("{preset_a}[c]\nversion = 1\nderive = 1\nupstream = \"{up_preset}\"\nx = 4\n")).unwrap();
    let alone = resolve::resolve(TEST_SCHEMA, &[Layer { name: "preset", file: &preset }], &target).unwrap();
    assert!(alone.warnings.is_empty(), "{:?}", alone.warnings);
    let two = |machine: &file::File| {
        resolve::resolve(TEST_SCHEMA, &[Layer { name: "preset", file: &preset }, Layer { name: "machine", file: machine }], &target).unwrap()
    };

    // The machine tuned [a] only: its values win knob by knob, and the preset's [c], measured
    // over other upstream values, is reported.
    let only_a = file::parse("[meta]\ntarget = \"aarch64-macos\"\n[a]\nversion = 2\nderive = 3\ni = 50\n").unwrap();
    let r = two(&only_a);
    assert_eq!((val(&r, "a.i"), val(&r, "a.f"), val(&r, "c.x")), (Value::Int(50), Value::Float(0.25), Value::Int(4)));
    assert!(r.warnings.iter().any(|w| w.starts_with("preset") && w.contains("[c]") && w.contains("upstream")), "{:?}", r.warnings);

    // The machine tuned [c] as well, over its own [a]: nothing is stale.
    let up_machine = resolve::upstream_text(&fl, &r.values, &TEST_SCHEMA[1]);
    assert_ne!(up_machine, up_preset);
    let both = file::parse(&format!("[a]\nversion = 2\nderive = 3\ni = 50\n[c]\nversion = 1\nderive = 1\nupstream = \"{up_machine}\"\nx = 9\n")).unwrap();
    let r = two(&both);
    assert_eq!(val(&r, "c.x"), Value::Int(9));
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);

    // Whose machine a file is: its [meta] target, when it has one.
    assert!(resolve::layer_target_ok(&only_a, &target).is_ok());
    assert!(resolve::layer_target_ok(&both, &target).is_ok(), "a file with no [meta] target is taken as it is");
    let err = resolve::layer_target_ok(&only_a, &Target { arch: "x86_64", os: "macos" }).unwrap_err();
    assert!(err.contains("aarch64-macos") && err.contains("x86_64-macos"), "{err}");
    assert!(resolve::layer_target_ok(&only_a, &Target { arch: "aarch64", os: "linux" }).is_err());
}

#[test]
fn test_schema_checks_catch_errors() {
    static BAD: &[(&str, &[Section])] = &[
        ("dup knob", &[Section { name: "a", version: 1, derive: 1, after: &[], doc: "", knobs: &[Knob::usize("i", "", 1), Knob::usize("i", "", 1)] }]),
        ("default out of range", &[Section { name: "a", version: 1, derive: 1, after: &[], doc: "", knobs: &[Knob::usize("i", "", 0).min(1)] }]),
        ("search outside valid", &[Section { name: "a", version: 1, derive: 1, after: &[], doc: "", knobs: &[Knob::usize("i", "", 5).int_range(1, 9).search(Range::Int { min: 0, max: 9 })] }]),
        ("default type", &[Section { name: "a", version: 1, derive: 1, after: &[], doc: "", knobs: &[Knob::new("i", "", Ty::Usize, crate::tuning::schema_types::DefaultValue::Float(1.0))] }]),
        ("unknown after", &[Section { name: "a", version: 1, derive: 1, after: &["b"], doc: "", knobs: &[] }]),
        ("after cycle", &[
            Section { name: "a", version: 1, derive: 1, after: &["b"], doc: "", knobs: &[] },
            Section { name: "b", version: 1, derive: 1, after: &["a"], doc: "", knobs: &[] },
        ]),
        ("follows unknown", &[Section { name: "a", version: 1, derive: 1, after: &[], doc: "", knobs: &[Knob::usize("i", "", 0).follows("a.z", 0)] }]),
        ("follows cycle", &[Section { name: "a", version: 1, derive: 1, after: &[], doc: "", knobs: &[Knob::usize("i", "", 0).follows("a.j", 0), Knob::usize("j", "", 0).follows("a.i", 0)] }]),
        ("follows another type", &[Section { name: "a", version: 1, derive: 1, after: &[], doc: "", knobs: &[Knob::f64("f", "", 1.0), Knob::usize("i", "", 0).follows("a.f", 0)] }]),
        ("bad name", &[Section { name: "a", version: 1, derive: 1, after: &[], doc: "", knobs: &[Knob::usize("Bad", "", 0)] }]),
        ("unknown arch", &[Section { name: "a", version: 1, derive: 1, after: &[], doc: "", knobs: &[Knob::usize("i", "", 0).arch("vax")] }]),
        ("steps on one thread", &[Section { name: "a", version: 1, derive: 1, after: &[], doc: "", knobs: &[Knob::new("s", "", Ty::U64, crate::tuning::schema_types::DefaultValue::Steps(&[(1, 1.0)]))] }]),
    ];
    for (what, schema) in BAD {
        assert!(resolve::check_schema(schema).is_err(), "check_schema accepted: {what}");
    }
}

#[test]
fn test_check_table() {
    assert!(check_table(ALGS, 1, &[1, 0, 9, 1]).is_ok());
    assert!(check_table(ALGS, 1, &[1, 0]).is_ok());
    assert!(check_table(ALGS, 1, &[1, 0, 9]).is_err(), "odd length");
    assert!(check_table(ALGS, 1, &[]).is_err());
    assert!(check_table(ALGS, 1, &[9, 0, 9, 1]).is_err(), "not ascending");
    assert!(check_table(ALGS, 1, &[1, 2]).is_err(), "no third algorithm");
    assert!(check_table(ALGS, 1, &[1, 1]).is_err(), "kara below its minimum");
    assert!(check_table(ALGS, 2, &[1, 1, 0, 4, 1, 0, 4, 10, 1, 5, 2, 0]).is_ok());
    assert!(check_table(ALGS, 2, &[4, 10, 1, 4, 9, 0]).is_err(), "l must ascend within a row");
    assert!(check_table(ALGS, 2, &[4, 10, 1, 3, 1, 0]).is_err(), "s must not descend");
}

#[test]
fn test_parse_spec_checks_values() {
    assert!(tuning::parse_spec("nope.nope=1").is_err());
    assert!(tuning::parse_spec("garbage").is_err());
    assert_eq!(tuning::parse_spec(" ; ").unwrap(), vec![]);
    for p in PATHS {
        let k = tuning::knob(p).unwrap();
        // A value outside the valid range, where one exists, is rejected.
        let outside = match k.valid {
            Range::Int { min, .. } if min > 0 => Some((min - 1).to_string()),
            Range::Int { max, .. } if max < u64::MAX => Some((max as u128 + 1).to_string()),
            Range::Float { min, .. } if min > f64::MIN => Some(file::float_text(min - min.abs() - 1.0)),
            _ => None,
        };
        if let Some(v) = outside {
            assert!(tuning::parse_spec(&format!("{p}={v}")).is_err(), "{p}={v} accepted");
        }
    }
}

// --- The live store (tune builds) ---

// Only no-op sets here (each knob to the value it holds); see the note at the top.
#[cfg(feature = "tune")]
#[test]
fn test_live_store_rejects_bad_values() {
    assert!(matches!(tuning::set("nope.nope", Value::Int(1)), Err(tuning::KnobError::Unknown(_))));
    for p in PATHS {
        let k = tuning::knob(p).unwrap();
        let v = tuning::get(p).unwrap();
        tuning::set(p, v.clone()).unwrap();
        let wrong = match k.ty {
            Ty::Bool => Value::Int(1),
            _ => Value::Bool(true),
        };
        assert!(tuning::set(p, wrong).is_err(), "{p} took a value of the wrong type");
        assert_eq!(tuning::get(p).unwrap(), v);
    }
    // Setting every knob to its own value changes no value (a table's cell may point at a new copy).
    let before: Vec<Value> = PATHS.iter().map(|p| tuning::get(p).unwrap()).collect();
    tuning::apply(&tuning::spec()).unwrap();
    assert_eq!(PATHS.iter().map(|p| tuning::get(p).unwrap()).collect::<Vec<_>>(), before);
    assert_eq!(tuning::all_accessor_bits(), RESOLVED_BITS);
}

#[cfg(feature = "tune")]
pub(super) mod random {
    //! The random-knob test (docs/tuning_system.md, "Validation" 1): invariant 1 as a test. Every
    //! value in a knob's valid range must give correct results, so for each knob vector the
    //! sections' checks run their operations against the suite's oracles.

    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// One section's check: operations of that section, against oracles, under the live knobs.
    pub(crate) type Check = fn(&mut StdRng);

    /// Each migrated section adds its check here.
    pub(crate) const SECTION_CHECKS: &[(&str, Check)] =
        &[
            ("mul", crate::tests::test_mul::random_knob_check),
            ("div", crate::tests::test_div::random_knob_check),
            ("sqrt", crate::tests::test_sqrt::random_knob_check),
            ("bin_split", crate::tests::test_bin_split::random_knob_check),
            ("series", crate::tests::test_series::random_knob_check),
            ("consts", crate::tests::test_bin_split::consts_random_knob_check),
            ("exp, ln, agm, theta", crate::tests::test_trans::random_knob_check),
        ];

    /// Sampling caps for integer knobs, by path: random values stay at or below the cap so that
    /// they fall among the checks' sizes; the knob's true maximum is still tried as an extreme.
    pub(crate) const CAPS: &[(&str, u64)] = &[
        ("base.prim_adx", 64),
        ("base.limb_adx", 64),
        ("mul.chunking_karatsuba", 300),
        ("mul.kara", 200),
        ("div.dyn_small", 300),
        ("div.dyn_rem_small", 300),
        ("mul.ntt_par_conv", 4096),
        ("mul.ntt_par", 4096),
        ("mul.ntt_par_radix3", 4096),
        ("mul.ntt_par_radix5", 4096),
        ("mul.karatsuba_sqr", 300),
        ("mul.fft_sqr", 500),
        ("mul.static_ntt_sqr", 500),
        ("mul.dyn_partial_mul", 600),
        ("mul.static_partial_mul", 600),
        ("mul.dyn_partial_sqr", 600),
        ("mul.static_partial_sqr", 600),
        ("mul.hi_mul_row", 300),
        ("mul.hi_sqr_row", 300),
        ("mul.fft_mid", 300),
        ("mul.ntt_mid", 300),
        ("mul.sqr_bnm1_fft", 600),
        ("div.knuth", 300),
        ("div.bz", 300),
        ("div.static_div_karatsuba_ntt_nr_bz", 600),
        ("div.static_div_rem_karatsuba_ntt_nr_bz", 600),
        ("div.dyn_rcp_knuth_nr", 300),
        ("div.static_rcp_knuth_nr", 300),
        ("sqrt.leaf", 48),
        ("sqrt.dyn_rem", 48),
        ("sqrt.dyn_only", 48),
        ("sqrt.dyn_approx", 48),
        ("sqrt.static_rem", 48),
        ("sqrt.static_only", 48),
        ("sqrt.static_approx", 48),
        ("sqrt.dyn_wrapped_rem", 320),
        ("sqrt.dyn_rcp_carry", 320),
        ("bin_split.dyn_bbp_leaf_limbs", 600),
        ("bin_split.static_bbp_leaf_limbs", 600),
        ("bin_split.dyn_engel_leaf_limbs", 600),
        ("bin_split.static_engel_leaf_limbs", 600),
        ("bin_split.dyn_hyper_leaf_limbs", 600),
        ("bin_split.dyn_hyper_leaf_terms", 64),
        ("bin_split.static_hyper_leaf_limbs", 600),
        ("bin_split.static_hyper_leaf_terms", 64),
        ("bin_split.dyn_par", 512),
        ("bin_split.dyn_merge_par", 512),
        ("series.horner_limbs", 64),
        ("series.horner_terms", 64),
        ("consts.dyn_ln2_par", 1100),
        ("consts.dyn_ln2_par_half", 1100),
        ("consts.dyn_pi_par", 1100),
        ("consts.dyn_pi_par_half", 1100),
        ("consts.static_ln2_par", 1100),
        ("consts.static_pi_par", 1100),
        ("theta.dyn_par_limbs", 64),
        ("theta.static_par_limbs", 64),
        ("agm.dyn_tail_terms", 19),
        ("agm.static_tail_terms", 19),
        ("exp.dyn_chunk_par_work", 4096),
        ("exp.dyn_chunk_min_bits", 64),
        ("exp.static_chunk_par_work", 4096),
        ("exp.static_chunk_min_bits", 64),
        ("ln.dyn_chunk_min_bits", 64),
        ("ln.static_chunk_min_bits", 64),
        ("ln.dyn_chunk_par_work", 4096),
        ("ln.static_chunk_par_work", 4096),
        ("ln.dyn_agm_limbs", 64),
        ("ln.static_agm_limbs", 64),
        ("ln.dyn_agm_par_until", 4096),
        ("ln.static_agm_par_until", 4096),
    ];
    const DEFAULT_CAP: u64 = 1024;

    fn cap(path: &str) -> u64 {
        CAPS.iter().find(|(p, _)| *p == path).map_or(DEFAULT_CAP, |&(_, c)| c)
    }

    // A table's algorithms and inputs.
    fn table_ty(k: &Knob) -> (&'static [Alg], u8) {
        match k.ty {
            Ty::Table { algs, inputs, .. } => (algs, inputs),
            _ => unreachable!("only table knobs have a table range"),
        }
    }

    // One change point (or row entry) for algorithm `a` at size `n`, in the table's layout.
    fn point(inputs: u8, n: u32, l: u32, a: u32) -> Vec<u32> {
        if inputs == 1 { vec![n, a] } else { vec![n, l, a] }
    }

    fn low(k: &Knob) -> Value {
        match k.valid {
            Range::Int { min, .. } => Value::Int(min),
            Range::Float { min, .. } => Value::Float(min),
            Range::Bool => Value::Bool(false),
            // The first algorithm everywhere.
            Range::Table => {
                let (algs, inputs) = table_ty(k);
                Value::Table(point(inputs, algs[0].min, algs[0].min, 0))
            }
        }
    }

    fn high(k: &Knob, path: &str, capped: bool) -> Value {
        match k.valid {
            Range::Int { min, max } => Value::Int(if capped { max.min(cap(path).max(min)) } else { max }),
            Range::Float { min, max } => Value::Float(if capped { max.min(min.max(1.0) * 1e3) } else { max }),
            Range::Bool => Value::Bool(true),
            // The last algorithm wherever it is eligible, the first below.
            Range::Table => {
                let (algs, inputs) = table_ty(k);
                let last = algs.len() - 1;
                let mut t = point(inputs, algs[0].min, algs[0].min, 0);
                t.extend(point(inputs, algs[last].min.max(algs[0].min + 1), algs[last].min.max(algs[0].min + 1), last as u32));
                Value::Table(t)
            }
        }
    }

    // A random valid table: a few rows (two inputs) or change points (one), ascending, each pick
    // eligible at its size.
    fn random_table(k: &Knob, path: &str, rng: &mut StdRng) -> Value {
        let (algs, inputs) = table_ty(k);
        let top = cap(path).max(8) as u32;
        let mut sizes: Vec<u32> = (0..rng.gen_range(1..5)).map(|_| rng.gen_range(algs[0].min..top)).collect();
        sizes.push(algs[0].min);
        sizes.sort_unstable();
        sizes.dedup();
        let mut t = Vec::new();
        for (i, &n) in sizes.iter().enumerate() {
            let ok: Vec<u32> = (0..algs.len() as u32).filter(|&a| algs[a as usize].min <= n).collect();
            let pick = |rng: &mut StdRng| if i == 0 { 0 } else { ok[rng.gen_range(0..ok.len())] };
            if inputs == 1 {
                t.extend([n, pick(rng)]);
            } else {
                let mut ls: Vec<u32> = (0..rng.gen_range(1..4)).map(|_| rng.gen_range(n..n.saturating_mul(4).max(n + 1))).collect();
                ls.push(n);
                ls.sort_unstable();
                ls.dedup();
                for l in ls {
                    t.extend([n, l, pick(rng)]);
                }
            }
        }
        Value::Table(t)
    }

    fn random(k: &Knob, path: &str, rng: &mut StdRng) -> Value {
        match k.valid {
            Range::Int { min, max } => {
                let hi = max.min(cap(path).max(min));
                // Log-uniform, so small values (where algorithms switch) are well covered.
                let (a, b) = ((min as f64 + 1.0).ln(), (hi as f64 + 1.0).ln());
                let v = (rng.gen_range(a..=b).exp() - 1.0).round() as u64;
                Value::Int(v.clamp(min, hi))
            }
            Range::Float { min, max } => {
                let hi = max.min(min.max(1.0) * 1e3);
                Value::Float(rng.gen_range(min..=hi))
            }
            Range::Bool => Value::Bool(rng.gen()),
            Range::Table => random_table(k, path, rng),
        }
    }

    fn knobs() -> Vec<(&'static str, &'static Knob)> {
        PATHS.iter().map(|&p| (p, tuning::knob(p).unwrap())).collect()
    }

    /// The knob vectors to try, each as `(name, values)`: everything low, everything high (capped
    /// and true maxima), the defaults, each knob alone at each extreme, then random vectors.
    fn vectors(seed: u64, n: usize) -> Vec<(String, Vec<Value>)> {
        let ks = knobs();
        let default: Vec<Value> = PATHS.iter().map(|p| tuning::get(p).unwrap()).collect();
        let mut out = vec![
            ("defaults".to_string(), default.clone()),
            ("everything low".to_string(), ks.iter().map(|(_, k)| low(k)).collect()),
            ("everything high".to_string(), ks.iter().map(|(p, k)| high(k, p, true)).collect()),
            ("everything at its maximum".to_string(), ks.iter().map(|(p, k)| high(k, p, false)).collect()),
        ];
        for (i, (p, k)) in ks.iter().enumerate() {
            for (what, v) in [("low", low(k)), ("high", high(k, p, true)), ("max", high(k, p, false))] {
                let mut vec = default.clone();
                vec[i] = v;
                out.push((format!("{p} {what}"), vec));
            }
        }
        let mut rng = StdRng::seed_from_u64(seed);
        for r in 0..n {
            out.push((format!("random {r}"), ks.iter().map(|(p, k)| random(k, p, &mut rng)).collect()));
        }
        out
    }

    fn spec_of(values: &[Value]) -> String {
        PATHS.iter().zip(values).map(|(p, v)| format!("{p}={}", v.text())).collect::<Vec<_>>().join(";")
    }

    fn env_u64(name: &str, default: u64) -> u64 {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    /// Runs every vector in this process. Only `random_knobs_child` may call it.
    fn run_vectors() {
        let seed = env_u64("BIGBITS_KNOB_SEED", 0x5eed);
        let list = match std::env::var("BIGBITS_KNOB_VECTOR") {
            Ok(spec) => {
                tuning::apply(&spec).expect("BIGBITS_KNOB_VECTOR is a valid spec");
                let v: Vec<Value> = PATHS.iter().map(|p| tuning::get(p).unwrap()).collect();
                tuning::reset();
                vec![("replayed".to_string(), v)]
            }
            Err(_) => vectors(seed, env_u64("BIGBITS_KNOB_VECTORS", 12) as usize),
        };
        for (name, values) in &list {
            let spec = spec_of(values);
            // Named first: a stack overflow aborts the process before catch_unwind can report.
            eprintln!("random-knob vector `{name}`: BIGBITS_KNOB_VECTOR='{spec}'");
            tuning::apply(&spec).unwrap_or_else(|e| panic!("vector {name}: {e}"));
            // The inputs depend only on the seed and the knob values, so a replay reproduces them.
            let vector_seed = seed ^ crate::tuning::hash::fnv1a(spec.bytes());
            for (section, check) in SECTION_CHECKS {
                let mut rng = StdRng::seed_from_u64(vector_seed);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(&mut rng)));
                if let Err(e) = result {
                    eprintln!("random-knob failure in section `{section}`, vector `{name}`; replay with");
                    eprintln!("BIGBITS_KNOB_VECTOR='{spec}' BIGBITS_KNOB_SEED={seed}");
                    std::panic::resume_unwind(e);
                }
            }
            tuning::reset();
        }
        println!("random-knob vectors passed: {} vectors x {} sections", list.len(), SECTION_CHECKS.len());
    }

    // Starts `random_knobs_child` in its own process, so no other test runs while knobs change.
    // BIGBITS_KNOB_SEED, BIGBITS_KNOB_VECTORS (random vectors, default 12) and
    // BIGBITS_KNOB_VECTOR (one spec to replay) pass through.
    #[test]
    fn random_knobs() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["tests::test_tuning::random::random_knobs_child", "--exact", "--ignored"])
            .args(["--test-threads=1", "--nocapture"])
            .env("BIGBITS_KNOB_CHILD", "1")
            .output()
            .expect("start the random-knob child");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "random-knob child failed:\n{stdout}\n{stderr}");
        assert!(stdout.contains("random-knob vectors passed"), "the child ran no vectors:\n{stdout}\n{stderr}");
    }

    #[test]
    #[ignore = "run by random_knobs in its own process"]
    fn random_knobs_child() {
        if std::env::var("BIGBITS_KNOB_CHILD").is_err() {
            return; // run directly with --ignored, other tests may be running
        }
        // Static entries keep [u64; N] buffers on the stack, and extreme knobs recurse deeply
        // (Karatsuba down to a few limbs), so the vectors run on a thread with a large stack.
        std::thread::Builder::new()
            .stack_size(1 << 30)
            .spawn(run_vectors)
            .unwrap()
            .join()
            .unwrap_or_else(|e| std::panic::resume_unwind(e));
    }
}
