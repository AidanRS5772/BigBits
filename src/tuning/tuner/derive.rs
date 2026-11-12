//! Derivations (docs/tuning_system.md, "Derivations"): computation on predictions, timing nothing.
//! The pilot derives Karatsuba's sub-product choice (`mul.kara`: schoolbook, chunking or one
//! Karatsuba level over schoolbook leaves) from mul's fitted schoolbook, level and chunking equations, and
//! compares it with the live `mul.kara` table, then turns it into a table of its own.

use crate::tuning::k;

/// The pilot's comparison of the derived choice with today's boundary.
pub struct KaraReport {
    pub shapes: usize,
    pub disagree: usize,
    /// The smallest square (s x s) the derived choice gives Karatsuba, and today's.
    pub square: (usize, usize),
    /// The smallest s at l = 4s the derived choice gives chunking, and today's.
    pub chunked: (usize, usize),
}

/// mul_buf's model at explicit constants: W = (l s, 1).
pub fn school_with(scale: f64, shift: f64) -> impl Fn(f64, f64) -> f64 {
    move |l: f64, s: f64| scale * l * s + shift
}

/// mul_buf's model at the live knobs.
pub fn school_model() -> impl Fn(f64, f64) -> f64 {
    school_with(k::mul::school_scale(), k::mul::school_shift())
}

/// chunking_karatsuba's chunk products for l x s (s <= ceil(l / 2)), each priced by `product`:
/// every full s x s chunk, and the last one of s - 1 to 2s - 2 limbs. The fitted `mul.chunk_scale`
/// multiplies exactly this, so the fit and the prediction share it.
pub fn chunk_products(l: usize, s: usize, mut product: impl FnMut(usize, usize) -> f64) -> f64 {
    let chunks = (l - s + 1) / s;
    let rest = l - chunks * s;
    chunks as f64 * product(s, s) + product(rest, s)
}

/// The three sub-products of a karatsuba_core level of l x s (the cross product at half + 1/2
/// square, as its sums carry in about half the calls), by `school`.
pub fn level_products(l: usize, s: usize, school: &dyn Fn(f64, f64) -> f64) -> f64 {
    let h = ((l + 1) / 2) as f64;
    let (l, s) = (l as f64, s as f64);
    school(h + 0.5, h + 0.5) + school(h, h) + school(l - h, s - h)
}

/// Predictions that follow the whole recursion, memoized by shape (the doc's derivation context):
/// a product's time is the least of schoolbook and Karatsuba, whose sub-products are predicted
/// the same way.
pub struct Pred {
    school: Box<dyn Fn(f64, f64) -> f64>,
    // A level: (scale of its sub-products, shift). Chunking: (scale of its chunk products, ns a limb).
    kara: (f64, f64),
    chunk: (f64, f64),
    margin: f64,
    memo: std::collections::HashMap<(u32, u32), f64>,
}

impl Pred {
    pub fn new() -> Self {
        Pred {
            school: Box::new(school_model()),
            kara: (k::mul::kara_scale(), k::mul::kara_shift()),
            chunk: (k::mul::chunk_scale(), k::mul::chunk_pass()),
            margin: (k::mul::school_err().powi(2) + k::mul::kara_err().powi(2)).sqrt(),
            memo: std::collections::HashMap::new(),
        }
    }

    pub fn school(&self, l: usize, s: usize) -> f64 {
        (self.school)(l as f64, s as f64)
    }

    /// The best prediction for l x s (either order).
    pub fn best(&mut self, l: usize, s: usize) -> f64 {
        let (l, s) = if l >= s { (l, s) } else { (s, l) };
        if s == 0 {
            return 0.0;
        }
        if s < 4 {
            return self.school(l, s);
        }
        if let Some(&v) = self.memo.get(&(l as u32, s as u32)) {
            return v;
        }
        // What dispatch will run there: Karatsuba only past the margin, as the table decides.
        let (sc, ka) = (self.school(l, s), self.kara(l, s));
        let v = if ka < sc * (1.0 - self.margin) { ka } else { sc };
        self.memo.insert((l as u32, s as u32), v);
        v
    }

    /// One karatsuba_core level of l x s (s > ceil(l / 2)) in its fitted form, over predicted
    /// sub-products; its cross product is half or half + 1 square, as its sums carry or not.
    fn level(&mut self, l: usize, s: usize) -> f64 {
        let h = (l + 1) / 2;
        let cross = (self.best(h, h) + self.best(h + 1, h + 1)) / 2.0;
        let products = cross + self.best(h, h) + self.best(l - h, s - h);
        self.kara.0 * products + self.kara.1
    }

    /// Karatsuba for l x s, s >= 4: a level, or chunking for s <= ceil(l / 2).
    pub fn kara(&mut self, l: usize, s: usize) -> f64 {
        if s > (l + 1) / 2 {
            return self.level(l, s);
        }
        // Chunking in its fitted form over its chunk products as dispatch will run them.
        let products = chunk_products(l, s, |a, b| self.best(a, b));
        self.chunk.0 * products + self.chunk.1 * (l + s) as f64
    }
}

/// The `mul.kara` table derived from the predictions (change points (s, l, Karatsuba?) over
/// 4 <= s <= l <= 4s, rows from s = 1), and its evidence. Karatsuba enters a row only past the
/// fits' margin, at least `eps`, and schoolbook takes a row back only where it wins by `eps`.
pub fn kara_table(eps: f64) -> (Vec<u32>, String) {
    const MAX_S: usize = 512;
    let mut p = Pred::new();
    let margin = p.margin.max(eps);
    let mut rows: Vec<Vec<(u32, u32)>> = Vec::new();
    for s in 1..=MAX_S {
        let mut pts: Vec<(u32, u32)> = Vec::new();
        for l in s..=4 * s {
            let a = if s < 4 {
                0
            } else {
                let (sc, ka) = (p.school(l, s), p.kara(l, s));
                match pts.last() {
                    None | Some(&(_, 0)) => (ka < sc * (1.0 - margin)) as u32,
                    Some(_) => !(sc < ka * (1.0 - eps)) as u32,
                }
            };
            if pts.last().map_or(true, |&(_, q)| q != a) {
                pts.push((l as u32, a));
            }
        }
        rows.push(pts);
    }
    let last = (0..MAX_S).rev().find(|&i| rows[i].len() != 1 || rows[i][0].1 != 1).map_or(0, |i| i + 1);
    let mut t = Vec::new();
    for (i, pts) in rows.iter().enumerate().take((last + 1).min(MAX_S)) {
        for &(l, a) in pts {
            t.extend([(i + 1) as u32, l, a]);
        }
    }
    let first = |p: &mut Pred, m: usize| {
        (4..=MAX_S).find(|&s| p.kara(m * s, s) < p.school(m * s, s) * (1.0 - margin)).unwrap_or(0)
    };
    let (square, chunk) = (first(&mut p, 1), first(&mut p, 4));
    let ev = format!(
        "derived (table): Karatsuba from {square} for squares and {chunk} at l = 4s, past a {:.1}% margin; every row from s = {} is Karatsuba throughout",
        100.0 * margin,
        last + 1
    );
    (t, ev)
}

pub fn kara_report(max_s: usize, max_l: usize) -> KaraReport {
    let kara = k::mul::kara();
    let mut p = Pred::new();
    let mut r = KaraReport { shapes: 0, disagree: 0, square: (0, 0), chunked: (0, 0) };
    for s in 4..=max_s {
        for l in s..=max_l.max(s) {
            let derived_school = p.school(l, s) <= p.kara(l, s);
            let today_school = crate::utils::mul::school_at(kara, l, s);
            r.shapes += 1;
            r.disagree += (derived_school != today_school) as usize;
            if l == s && r.square.0 == 0 && !derived_school {
                r.square.0 = s;
            }
            if l == s && r.square.1 == 0 && !today_school {
                r.square.1 = s;
            }
            if l == 4 * s && r.chunked.0 == 0 && !derived_school {
                r.chunked.0 = s;
            }
            if l == 4 * s && r.chunked.1 == 0 && !today_school {
                r.chunked.1 = s;
            }
        }
    }
    r
}

/// One shape of the pilot's check.
pub struct CheckRow {
    pub l: usize,
    pub s: usize,
    /// mul_buf, measured and predicted.
    pub school: (f64, f64),
    /// One karatsuba_core level over schoolbook leaves, measured and predicted.
    pub level: (f64, f64),
    /// The level's three sub-products (mul_buf at their sizes), measured and predicted.
    pub products: (f64, f64),
}

/// The pilot's check (docs/tuning_system.md, "Checking derived values"): at each l x s shape with
/// s > ceil(l / 2), time mul_buf against one karatsuba_core level over schoolbook leaves, and the
/// level's three sub-products alone, beside their predictions. Sub-products are forced to
/// schoolbook by setting `mul.kara` live to an all-schoolbook table (the old harness rebuilt with
/// `--set` for this), and restored afterwards.
pub fn kara_check(level: super::measure::Level, shapes: &[(usize, usize)]) -> Vec<CheckRow> {
    use super::measure::Probe;
    use crate::tuning;
    use crate::utils::mul::{karatsuba_core, mul_buf};
    use std::hint::black_box;

    let school = school_model();
    let saved = tuning::snapshot();
    // Schoolbook at every shape, so the level's sub-products are leaves.
    tuning::set("mul.kara", super::sections::mul::all_school()).unwrap();
    let mut rows = Vec::new();
    for &(l, s) in shapes {
        let h = (l + 1) / 2;
        assert!(s > h && s <= l, "the check covers karatsuba_core's shapes");
        let t = super::measure::time_min(level, |pass| {
            let pad = |n, seed| super::measure::pad(n, seed, pass);
            let (a, b, a2, b2) = (pad(l, 3), pad(s, 5), pad(l, 3), pad(s, 5));
            let (c1, c2) = (pad(h + 1, 11), pad(h + 1, 13));
            let (x0, y0, x1, y1) = (pad(h, 17), pad(h, 19), pad(l - h, 23), pad(s - h, 29));
            let (mut o1, mut o2, mut cross) = (pad(l + s - 1, 31), pad(l + s - 1, 37), pad(2 * h + 3, 41));
            let mut scratch = pad(4 * l + 8, 43);
            let (mut p1, mut p2, mut p3) = (pad(2 * h + 1, 47), pad(2 * h - 1, 53), pad(l + s - 2 * h - 1, 59));
            vec![
                Probe::new("school", move || {
                    black_box(mul_buf(black_box(&a), black_box(&b), &mut o1));
                }),
                Probe::new("level", move || {
                    black_box(karatsuba_core(black_box(&a2), black_box(&b2), h, &mut o2, &mut cross, &mut scratch));
                }),
                Probe::new("products", move || {
                    black_box(mul_buf(black_box(&c1), black_box(&c2), &mut p1));
                    black_box(mul_buf(black_box(&x0), black_box(&y0), &mut p2));
                    black_box(mul_buf(black_box(&x1), black_box(&y1), &mut p3));
                }),
            ]
        });
        let (lf, sf) = (l as f64, s as f64);
        // The check's level runs over schoolbook leaves, so its prediction does too.
        let sp = school(lf, sf);
        let prod_pred = level_products(l, s, &school);
        let lp = k::mul::kara_scale() * prod_pred + k::mul::kara_shift();
        rows.push(CheckRow { l, s, school: (t[0], sp), level: (t[1], lp), products: (t[2], prod_pred) });
    }
    tuning::restore(&saved);
    rows
}

/// The end-to-end check of derived values: `run`'s time at each shape under the settings `base`
/// and `derived` (each a list of knob values), installed in turn between interleaved rounds.
/// Returns (base ns, derived ns) per shape, the least over the level's passes.
pub fn compare<S: Copy>(
    level: super::measure::Level,
    base: &[(&str, crate::tuning::Value)],
    derived: &[(&str, crate::tuning::Value)],
    shapes: &[S],
    run: impl Fn(S, usize) -> Box<dyn FnMut()> + Copy,
) -> Vec<(f64, f64)> {
    use super::measure::{time_min, Probe};
    use crate::tuning::{install, prepare, Handle};
    let saved = crate::tuning::snapshot();
    let prep = |set: &[(&str, crate::tuning::Value)]| -> Vec<Handle> {
        set.iter().map(|(p, v)| prepare(p, v).expect("a valid knob value")).collect()
    };
    let (hb, hd) = (prep(base), prep(derived));
    let mut out = Vec::new();
    for &shape in shapes {
        let t = time_min(level, |pass| {
            [hb.clone(), hd.clone()]
                .into_iter()
                .map(|hs| {
                    let mut f = run(shape, pass);
                    Probe::new("compare", move || {
                        for &h in &hs {
                            install(h);
                        }
                        f();
                    })
                })
                .collect()
        });
        out.push((t[0], t[1]));
    }
    crate::tuning::restore(&saved);
    out
}

/// A series this long by its best prediction is left to the equations: reading them costs
/// 0.1-0.2 us, under 1% of it.
pub const SMALL_EXTENT_NS: f64 = 2e4;

// 1, 2, 3, ... and, from where a step of `per` to the octave passes 1, by such steps to `top`.
fn octave_steps(top: u32, per: f64) -> Vec<u32> {
    let mut v: Vec<u32> = Vec::new();
    let mut x = 1.0f64;
    while x <= top as f64 {
        let n = x.round() as u32;
        if v.last() != Some(&n) {
            v.push(n);
        }
        x = (x * 2f64.powf(1.0 / per)).max(x + 1.0);
    }
    v
}

/// The `series.small` table derived from the equations in force, and its evidence: for every
/// class and every bucket of frames (`small_frame`: at the bucket's middle frame), over terms up
/// to where the best prediction passes `SMALL_EXTENT_NS`, what the dispatch itself would decide
/// for the class's representative shape (`Shape::of_class`; guards and all) under the short
/// series' prices (`SeriesEq::read_short`). The grid is coarse
/// on purpose (frames by octaves, terms by thirds of one): next to a boundary the two algorithms
/// are within a few percent, and what the table buys is not reading the equations. Along the
/// terms a pick holds until another's prediction is under it by `eps` (at least 3%), and a pick
/// that lasts one step between two runs of another is dropped, so algorithms within the fits'
/// noise of each other don't trade cells. A row that repeats the one before it in its class is
/// left out (a lookup takes the row at or below its key).
pub fn series_small_table(eps: f64) -> (Vec<u32>, String) {
    series_small_table_at(eps, 3.0)
}

/// `series_small_table` at `terms` grid steps to the octave.
pub fn series_small_table_at(eps: f64, terms: f64) -> (Vec<u32>, String) {
    use crate::utils::series::{small_frame, small_frame_span, Costs, SeriesAlg, SeriesEq, SeriesGuards, Shape, SMALL_CLASSES, SMALL_PICKS, SMALL_ROWS};
    const MODEL: u32 = (SMALL_PICKS.len() - 1) as u32;
    let index = |alg: SeriesAlg| SMALL_PICKS.iter().position(|p| *p == Some(alg)).expect("every algorithm has a pick") as u32;
    let buckets = small_frame(65535).expect("the last frame the table holds") + 1;
    let terms = octave_steps(1 << 15, terms);
    // The prices fitted on the series under the extent: the table is theirs to decide.
    let (guards, eps, prices) = (SeriesGuards::read(), eps.max(0.03), SeriesEq::read_short());
    let (mut t, mut classes) = (Vec::new(), 0);
    // Cells inside the extent by pick (in SMALL_PICKS' order), each a (frame, terms) grid point.
    let mut cells = [0usize; 3];
    for class in 0..SMALL_CLASSES {
        if Shape::of_class(class, 1, 1).is_none() {
            continue;
        }
        classes += 1;
        let mut before: Option<Vec<(u32, u32)>> = None;
        for bucket in 0..buckets {
            let (lo, hi) = small_frame_span(bucket);
            let (len, key) = ((lo + hi) / 2, (class * SMALL_ROWS + bucket) as u32);
            // The row's cell at each step of the terms, to the first past the extent.
            let mut line: Vec<u32> = Vec::new();
            let mut held: Option<SeriesAlg> = None;
            for (i, &from) in terms.iter().enumerate() {
                // A step of the grid is decided at its last term count: binary splitting's
                // time steps up where its Q passes a limb, and a step read at its first count
                // would give it the counts past that too (a tree of 9 terms, taken at 8).
                let n = terms.get(i + 1).map_or(from, |&next| next - 1);
                let (shape, ok) = Shape::of_class(class, len, n as u64).expect("a class with a shape");
                let costs = shape.predictions_at(ok, &prices);
                let best = [SeriesAlg::Horner, SeriesAlg::Rect, SeriesAlg::BinSplit].iter().filter_map(|&a| costs.of(a)).fold(f64::INFINITY, f64::min);
                if best > SMALL_EXTENT_NS {
                    line.push(MODEL);
                    break;
                }
                // What the dispatch decides under these prices: the guards, then the least.
                let g = shape.guards(ok, &guards);
                let decided = if g.rect_first { SeriesAlg::Rect } else { Costs { bin_split: if g.no_bin_split { f64::INFINITY } else { costs.bin_split }, ..costs }.pick() };
                // The pick before stays while it is within eps of what would replace it.
                let alg = held.filter(|&h| costs.of(h).is_some_and(|ns| ns <= costs.of(decided).unwrap_or(f64::INFINITY) * (1.0 + eps))).unwrap_or(decided);
                held = Some(alg);
                line.push(index(alg));
            }
            for i in 1..line.len().saturating_sub(1) {
                if line[i - 1] == line[i + 1] && line[i] != line[i - 1] && line[i + 1] != MODEL {
                    line[i] = line[i - 1];
                }
            }
            let mut row: Vec<(u32, u32)> = Vec::new();
            for (&n, &cell) in terms.iter().zip(&line) {
                if cell != MODEL {
                    cells[cell as usize] += 1;
                }
                if row.last().map_or(true, |&(_, c)| c != cell) {
                    row.push((n, cell));
                }
            }
            if before.as_ref() != Some(&row) {
                for &(n, cell) in &row {
                    t.extend([key, n, cell]);
                }
                before = Some(row);
            }
        }
        // A class's keys are its own: `SMALL_ROWS` to a class, a bucket each (a frame past the
        // buckets is never looked up, and a class no series has is never asked for).
        debug_assert!(buckets <= SMALL_ROWS);
    }
    let all = cells.iter().sum::<usize>().max(1) as f64;
    let ev = format!(
        "derived (table): {} change points over {classes} classes, for series the equations put under {:.0} us; inside it Horner has {:.0}% of the cells, rect {:.0}%, binary splitting {:.0}%",
        t.len() / 3,
        SMALL_EXTENT_NS / 1e3,
        100.0 * cells[0] as f64 / all,
        100.0 * cells[1] as f64 / all,
        100.0 * cells[2] as f64 / all
    );
    (t, ev)
}
