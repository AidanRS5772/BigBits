//! Dispatch tables, measured. The rule, one for every dispatcher that compares fitted equations:
//! at an index where that decision costs more than `SHARE` of the call, the dispatcher reads a
//! table, whose cell is the algorithm that was fastest when each was timed there, forced; past
//! that it reads its equations. So a table is never computed from the equations, and each row
//! ends at the first index where the decision, timed at explicit knobs, is within the share.
//! `row` is the rule on one row's timings (nothing timed: tests give it made-up ones). `measure`
//! times a table through a `Spec`; every table is measured by it (`TABLES` in `tuner/dispatch.rs`).

use crate::tuning::tuner::measure::{less_harness, time_fit, Level, Probe};

/// The share of a call a dispatcher's decision may take: past it the index is the table's. The
/// user's number, for every dispatcher, and theirs to raise.
pub const SHARE: f64 = 0.05;

/// The shares a table's size is also reported at (those above the one it is measured at).
pub const OTHER_SHARES: [f64; 5] = [0.01, 0.02, 0.05, 0.10, 0.20];

static SHARE_NOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The share for this run of the tuner: `SHARE`, or what `--share` named. Nothing in the library reads it.
pub fn share() -> f64 {
    match SHARE_NOW.load(std::sync::atomic::Ordering::Relaxed) {
        0 => SHARE,
        bits => f64::from_bits(bits),
    }
}

pub fn set_share(share: f64) {
    SHARE_NOW.store(share.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// One call timed at an index.
#[derive(Clone, Debug, PartialEq)]
pub struct Timed {
    /// (pick, ns) of each algorithm that can run it, forced.
    pub ns: Vec<(u32, f64)>,
    /// ns of the dispatcher's decision for it, at explicit knobs. None where it reads no equations.
    pub decide: Option<f64>,
    /// For the spec's dumps (`Plan::note`).
    pub note: Vec<f64>,
}

impl Timed {
    pub fn best(&self) -> Option<f64> {
        self.ns.iter().map(|x| x.1).min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    // Its time under `pick` over its fastest algorithm's. A pick that can't run it leaves the call
    // to the equations: the decision paid, their pick taken as the fastest. None with no equations.
    fn cost(&self, pick: u32) -> Option<f64> {
        let best = self.best()?;
        match self.ns.iter().find(|x| x.0 == pick) {
            Some(x) => Some(x.1 / best),
            None => self.decide.map(|d| (best + d) / best),
        }
    }
}

/// One index's timings. Most tables time one call there; a row that stands for several kinds of
/// call times one of each, and its pick is the one that costs them least together.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub size: u32,
    pub of: Vec<Timed>,
    /// The shortest call the index stands for, as a fraction of the one timed (a step of a coarse
    /// grid stands for smaller sizes too). The rule is read at that call.
    pub least: f64,
}

impl Cell {
    /// An index where one call was timed.
    pub fn one(size: u32, ns: Vec<(u32, f64)>, decide: Option<f64>) -> Cell {
        Cell { size, of: vec![Timed { ns, decide, note: Vec::new() }], least: 1.0 }
    }

    /// Whether nothing was timed at the index.
    pub fn bare(&self) -> bool {
        self.of.iter().all(|t| t.ns.is_empty())
    }

    /// What `pick` costs at the index: the mean over the calls timed there of the call's time under
    /// it over its own fastest algorithm's. None for a pick none of them can run.
    pub fn cost(&self, pick: u32) -> Option<f64> {
        let timed: Vec<&Timed> = self.of.iter().filter(|t| !t.ns.is_empty()).collect();
        if !timed.iter().any(|t| t.ns.iter().any(|x| x.0 == pick)) {
            return None;
        }
        let mut sum = 0.0;
        for t in &timed {
            sum += t.cost(pick)?;
        }
        Some(sum / timed.len() as f64)
    }

    /// What reading the equations costs at the index, on `cost`'s scale: the decision paid, the
    /// equations' pick taken as the fastest. None where a call reads none.
    pub fn read(&self) -> Option<f64> {
        let timed: Vec<&Timed> = self.of.iter().filter(|t| !t.ns.is_empty()).collect();
        let mut sum = 0.0;
        for t in &timed {
            sum += 1.0 + t.decide? / t.best()?;
        }
        (!timed.is_empty()).then(|| sum / timed.len() as f64)
    }

    /// The algorithm that costs least at the index, and its cost.
    pub fn fastest(&self) -> Option<(u32, f64)> {
        let mut picks: Vec<u32> = self.of.iter().flat_map(|t| t.ns.iter().map(|x| x.0)).collect();
        picks.sort_unstable();
        picks.dedup();
        picks.into_iter().filter_map(|p| Some((p, self.cost(p)?))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// The decision's share of the shortest call the index stands for: the largest among its calls.
    pub fn share(&self) -> Option<f64> {
        self.of.iter().filter_map(|t| Some(t.decide? / (t.best()? * self.least))).max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Whether the index is the equations': the decision within `share` of every call timed there.
    pub fn past(&self, share: f64) -> bool {
        self.share().is_some_and(|s| s <= share)
    }
}

/// A row's change points (size, pick) from its cells, in ascending size. An index whose decision
/// is past `share` takes the pick that costs least there; the pick before it stays while within
/// `eps` of the least, and a pick of one cell between two of another is dropped. With
/// `equations` (the pick that hands an index to the equations) the row ends at the first index
/// within the share, or where nothing was timed; and reading the equations is itself a pick, at
/// its own cost (`Cell::read`), where no one algorithm serves the calls an index stands for.
pub fn row(cells: &[Cell], equations: Option<u32>, eps: f64, share: f64) -> Vec<(u32, u32)> {
    let mut line: Vec<(u32, u32)> = Vec::new();
    let mut held: Option<u32> = None;
    // Whether the last entry of `line` is the index the row ends on.
    let mut ended = false;
    for c in cells {
        let Some((fast, cost)) = c.fastest() else {
            line.extend(equations.map(|e| (c.size, e)));
            ended = equations.is_some();
            break;
        };
        if let Some(e) = equations.filter(|_| c.past(share)) {
            line.push((c.size, e));
            ended = true;
            break;
        }
        // The equations as a pick: taken where they cost less than any algorithm.
        let read = equations.and_then(|e| Some((e, c.read()?)));
        let (fast, cost) = read.filter(|r| r.1 < cost).unwrap_or((fast, cost));
        let cost_of = |pick: u32| if Some(pick) == equations { read.map(|r| r.1) } else { c.cost(pick) };
        let pick = held.filter(|&h| cost_of(h).is_some_and(|t| t <= cost * (1.0 + eps))).unwrap_or(fast);
        held = Some(pick);
        line.push((c.size, pick));
        ended = false;
    }
    // The index a row ends on is no neighbour.
    let inner = line.len() - ended as usize;
    for i in 1..inner.saturating_sub(1) {
        if line[i - 1].1 == line[i + 1].1 && line[i].1 != line[i - 1].1 {
            line[i].1 = line[i - 1].1;
        }
    }
    let mut points: Vec<(u32, u32)> = Vec::new();
    for &(size, pick) in &line {
        if points.last().is_none_or(|p| p.1 != pick) {
            points.push((size, pick));
        }
    }
    points
}

/// A row's pick at `size`: its last change point's at or below it.
pub fn pick_at(points: &[(u32, u32)], size: u32) -> Option<u32> {
    points.iter().rev().find(|p| p.0 <= size).map(|p| p.1)
}

/// The probes of one call at an index, in the order a `Spec` pushed them: a harness probe if
/// `harness` (the probes' own work, taken off each), one for each of `picks`, the decision's if
/// `decide`.
#[derive(Clone, Debug)]
pub struct Plan {
    pub harness: bool,
    pub picks: Vec<u32>,
    pub decide: bool,
    /// `Cell::least` for the index (its first plan's).
    pub least: f64,
    /// `Timed::note` for the call.
    pub note: Vec<f64>,
}

impl Default for Plan {
    fn default() -> Self {
        Plan { harness: false, picks: Vec::new(), decide: false, least: 1.0, note: Vec::new() }
    }
}

/// A table to measure.
pub trait Spec {
    /// The pick that hands an index to the equations; None for a table that decides at every size.
    fn equations(&self) -> Option<u32>;

    /// The next rows to measure together, each a key and its ascending grid; None when done. A
    /// group's rows don't depend on each other, and every group before it is settled.
    fn group(&mut self) -> Option<Vec<(u32, Vec<u32>)>>;

    /// Pushes the probes of one index and returns a plan for each call timed there, in order.
    fn probes(&self, key: u32, size: u32, pass: usize, probes: &mut Vec<Probe<'static>>) -> Vec<Plan>;

    /// A group's rows as measured: for the spec to keep, and to make live where later rows need them.
    fn settled(&mut self, rows: Vec<Row>);
}

/// A measured row: its key, its grid, its change points, and its cells.
#[derive(Clone, Debug)]
pub struct Row {
    pub key: u32,
    pub sizes: Vec<u32>,
    pub points: Vec<(u32, u32)>,
    pub cells: Vec<Cell>,
}

impl Row {
    /// Whether the row ends at its first index, so the equations have all of it.
    pub fn all_equations(&self, equations: Option<u32>) -> bool {
        self.cells.len() <= 1 && self.points.first().is_some_and(|p| Some(p.1) == equations)
    }
}

/// What `measure` timed, for a table's evidence.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tally {
    /// Indices timed, and those the table keeps.
    pub cells: usize,
    pub kept: usize,
    /// Rows that reach the equations; the sum and the largest of the decision's share where each does.
    pub ended: usize,
    pub end_share: f64,
    pub worst_end: f64,
    /// The rms difference between the two timing passes.
    pub noise: f64,
    /// Indices inside the table that are the equations' as their least-cost pick, and the decision's
    /// largest share there.
    pub handed: usize,
    pub worst_handed: f64,
}

impl Tally {
    /// The tally as a table's evidence.
    pub fn text(&self) -> String {
        let ends = if self.ended == 0 {
            String::new()
        } else {
            format!(
                "; {} rows reach the equations, where reading them is {:.1}% of the call on average and {:.1}% at most (the rule: {:.0}%)",
                self.ended,
                100.0 * self.end_share / self.ended as f64,
                100.0 * self.worst_end,
                100.0 * share()
            )
        };
        let handed = if self.handed == 0 {
            String::new()
        } else {
            format!(
                "; at {} of the table's indices no one algorithm serves the calls the index stands for, and the equations are read (the decision up to {:.0}% of the call there)",
                self.handed,
                100.0 * self.worst_handed
            )
        };
        format!("{} indices timed, {} of them the table's{ends}{handed}; passes differ {:.1}%", self.cells, self.kept, 100.0 * self.noise)
    }
}

// The sizes of a row timed at a time, where a row can end before its grid.
const PER_WAVE: usize = 3;

/// Times a table: group by group, `PER_WAVE` sizes of every open row at a time (each wave one
/// `time_fit`), a row closing where it reaches the equations or the end of its grid (a spec
/// with no equations: every row over its whole grid at once). `eps` is the tie band.
pub fn measure<S: Spec>(level: Level, spec: &mut S, eps: f64) -> Tally {
    let share = share();
    let equations = spec.equations();
    let ends = |c: &Cell| equations.is_some() && (c.past(share) || c.bare());
    let mut tally = Tally::default();
    let (mut noise2, mut probes_timed) = (0.0, 0usize);
    while let Some(group) = spec.group() {
        let mut cells: Vec<Vec<Cell>> = vec![Vec::new(); group.len()];
        let mut open: Vec<usize> = (0..group.len()).filter(|&r| !group[r].1.is_empty()).collect();
        while !open.is_empty() {
            // This wave's indices, (row, size).
            let wave: Vec<(usize, u32)> = open
                .iter()
                .flat_map(|&r| {
                    let (from, sizes) = (cells[r].len(), &group[r].1);
                    let to = if equations.is_some() { (from + PER_WAVE).min(sizes.len()) } else { sizes.len() };
                    sizes[from..to].iter().map(move |&size| (r, size))
                })
                .collect();
            let mut plans: Vec<Vec<Plan>> = Vec::new();
            let timed = time_fit(level, |pass| {
                let mut probes = Vec::new();
                plans = wave.iter().map(|&(r, size)| spec.probes(group[r].0, size, pass, &mut probes)).collect();
                probes
            });
            noise2 += timed.ns.len() as f64 * timed.noise * timed.noise;
            probes_timed += timed.ns.len();
            let mut at = 0;
            for (&(r, size), plans) in wave.iter().zip(&plans) {
                let mut of = Vec::new();
                for plan in plans {
                    let harness = if plan.harness { timed.ns[at] } else { 0.0 };
                    at += plan.harness as usize;
                    let ns = plan.picks.iter().enumerate().map(|(i, &p)| (p, less_harness(timed.ns[at + i], harness))).collect();
                    at += plan.picks.len();
                    let decide = plan.decide.then(|| timed.ns[at]);
                    at += plan.decide as usize;
                    of.push(Timed { ns, decide, note: plan.note.clone() });
                }
                cells[r].push(Cell { size, of, least: plans.first().map_or(1.0, |p| p.least) });
            }
            open.retain(|&r| !cells[r].iter().any(ends) && cells[r].len() < group[r].1.len());
        }
        let rows = group
            .iter()
            .zip(cells)
            .map(|((key, sizes), mut cells)| {
                // A wave's cells past the index where the row ends aren't the row's.
                if let Some(end) = cells.iter().position(ends) {
                    cells.truncate(end + 1);
                }
                let points = points_of(&cells, sizes, equations, eps, share);
                tally.cells += cells.len();
                // The table's own indices handed to the equations as a pick.
                for c in cells.iter().filter(|c| !ends(c)) {
                    let pick = pick_at(&points, c.size);
                    if pick.is_some() && pick == equations {
                        tally.handed += 1;
                        tally.worst_handed = tally.worst_handed.max(c.share().unwrap_or(0.0));
                    }
                }
                match cells.last().filter(|c| ends(c)) {
                    Some(last) => {
                        tally.kept += cells.len() - 1;
                        tally.ended += 1;
                        let end = last.share().unwrap_or(0.0);
                        tally.end_share += end;
                        tally.worst_end = tally.worst_end.max(end);
                    }
                    None => tally.kept += cells.len(),
                }
                Row { key: *key, sizes: sizes.clone(), points, cells }
            })
            .collect();
        spec.settled(rows);
    }
    tally.noise = if probes_timed == 0 { 0.0 } else { (noise2 / probes_timed as f64).sqrt() };
    tally
}

// A row's change points at `share`; where its grid ended inside the table, the equations past it.
fn points_of(cells: &[Cell], sizes: &[u32], equations: Option<u32>, eps: f64, share: f64) -> Vec<(u32, u32)> {
    let mut points = row(cells, equations, eps, share);
    if let (Some(e), Some(last)) = (equations, sizes.last()) {
        if points.last().is_none_or(|p| p.1 != e) {
            points.push((last + 1, e));
        }
    }
    points
}

/// The rows as they would be at a larger `share`, from the same cells.
pub fn rows_at(rows: &[Row], equations: Option<u32>, eps: f64, share: f64) -> Vec<Row> {
    rows.iter().map(|r| Row { points: points_of(&r.cells, &r.sizes, equations, eps, share), ..r.clone() }).collect()
}

/// "N at 10%, M at 20%": the table's change points at each of `OTHER_SHARES` above the run's share.
pub fn sizes_above(rows: &[Row], equations: Option<u32>, eps: f64, same: impl Fn(u32, u32) -> bool) -> String {
    let size = |at: f64| assemble(&rows_at(rows, equations, eps, at), &same).len() / 3;
    let v: Vec<String> = OTHER_SHARES.iter().filter(|&&s| s > share() * 1.001).map(|&s| format!("{} at {:.0}%", size(s), 100.0 * s)).collect();
    v.join(", ")
}

/// A table `[key, size, pick, ...]` from rows in ascending key order. A row that repeats the one
/// before it is left out where `same(before, key)`: a lookup takes the row at or below its key.
pub fn assemble(rows: &[Row], same: impl Fn(u32, u32) -> bool) -> Vec<u32> {
    let mut t = Vec::new();
    let mut before: Option<&Row> = None;
    for r in rows {
        if r.points.is_empty() || before.is_some_and(|b| b.points == r.points && same(b.key, r.key)) {
            continue;
        }
        for &(size, pick) in &r.points {
            t.extend([r.key, size, pick]);
        }
        before = Some(r);
    }
    t
}

/// A grid along a row: 1, 2, 3, ... and, from where a step of `per` to the octave passes 1, by such steps to `top`.
pub fn octave_steps(from: u32, top: u32, per: f64) -> Vec<u32> {
    let mut v: Vec<u32> = Vec::new();
    let mut x = from.max(1) as f64;
    while x <= top as f64 {
        let n = x.round() as u32;
        if v.last() != Some(&n) {
            v.push(n);
        }
        x = (x * 2f64.powf(1.0 / per)).max(x + 1.0);
    }
    v
}
