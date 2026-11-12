//! Timing (docs/tuning_system.md, "Fitting a cost model" 2): each probe runs in batches long enough
//! that the clock's resolution doesn't matter. There are two ways to time a set of probes.
//!
//! - For a fit (`time_fit`): a fixed few batches of every probe, taken in rounds that each visit
//!   every probe once in a shuffled order, and the whole set twice. A fit pools its shapes, so its
//!   constants gain from more shapes and not from a tighter time at each (the user's decision,
//!   2026-11-11): there is no spread bound. The rounds spread a probe's batches in time, so a
//!   disturbance longer than a batch costs several probes one batch each, which their medians
//!   drop, and not one probe all of its batches; the shuffle keeps a machine that drifts from
//!   bending the fit along the sizes; and the two passes' difference measures the noise.
//! - For a sweep, whose decision is the ratio of two times at one size (`time_min`,
//!   `time_fixed`): rounds interleave every probe so drift cancels between the two sides, and a
//!   probe's value is the median of its rounds.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Asks the OS to run the calling thread on performance cores. On Apple silicon a loaded machine
/// can put a process on its efficiency cores, which run about half as fast, and the first base
/// fits on the M1 varied 2.5x between runs that way; the user-interactive QoS class keeps
/// measurements on one kind of core. On Linux a hybrid CPU's fastest cores are read from sysfs
/// and the thread is pinned to them (`fastest_of`): the measuring thread always, a pool worker
/// only when the whole pool fits on those cores, as a pool larger than that runs on the slower
/// ones in real use too. Written from the kernel's documentation and compile-checked only: it
/// has not been run on a hybrid Linux machine. Elsewhere, and on a uniform CPU, it does nothing.
pub fn prefer_fast_cores() {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
        // SAFETY: sets only the calling thread's scheduling class.
        unsafe {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        }
    }
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    {
        static FAST: std::sync::OnceLock<Option<Vec<usize>>> = std::sync::OnceLock::new();
        let Some(set) = FAST.get_or_init(|| fastest_of(&linux_core_ratings())) else { return };
        if rayon::current_thread_index().is_some() && rayon::current_num_threads() > set.len() {
            return;
        }
        extern "C" {
            fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
        }
        // A cpu_set_t: 1024 bits in 64-bit words.
        let mut mask = [0u64; 16];
        for &cpu in set.iter().filter(|&&c| c < 1024) {
            mask[cpu / 64] |= 1 << (cpu % 64);
        }
        // SAFETY: pid 0 is the calling thread, and the mask is a whole cpu_set_t of the size passed.
        unsafe {
            sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr());
        }
    }
}

/// Each CPU's speed rating as Linux's sysfs gives it: `cpu_capacity` (ARM's scheduler capacity),
/// else `cpufreq/cpuinfo_max_freq`. Empty off Linux, or where neither file exists.
pub fn linux_core_ratings() -> Vec<(usize, u64)> {
    let mut rated = Vec::new();
    for cpu in 0..1024 {
        let dir = format!("/sys/devices/system/cpu/cpu{cpu}");
        if !std::path::Path::new(&dir).exists() {
            break;
        }
        let read = |f: &str| std::fs::read_to_string(format!("{dir}/{f}")).ok().and_then(|s| s.trim().parse::<u64>().ok());
        if let Some(r) = read("cpu_capacity").or_else(|| read("cpufreq/cpuinfo_max_freq")) {
            rated.push((cpu, r));
        }
    }
    rated
}

/// The CPUs to measure on, from (cpu, rating) pairs: those within 15% of the highest rating, or
/// None when that is all of them (a uniform machine) or nothing is rated. The margin keeps a
/// hybrid Intel CPU's performance cores together (two "favored" ones report a slightly higher
/// top frequency) while leaving out efficiency cores, which sit a quarter or more below.
pub fn fastest_of(rated: &[(usize, u64)]) -> Option<Vec<usize>> {
    let top = rated.iter().map(|r| r.1).max()?;
    let fast: Vec<usize> = rated.iter().filter(|r| r.1 * 100 >= top * 85).map(|r| r.0).collect();
    (fast.len() < rated.len()).then_some(fast)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Quick,
    Normal,
    Thorough,
}

impl Level {
    pub fn parse(s: &str) -> Option<Level> {
        match s {
            "quick" => Some(Level::Quick),
            "normal" => Some(Level::Normal),
            "thorough" => Some(Level::Thorough),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Level::Quick => "quick",
            Level::Normal => "normal",
            Level::Thorough => "thorough",
        }
    }
    /// The tie band ε.
    pub fn eps(self) -> f64 {
        match self {
            Level::Quick => 0.03,
            Level::Normal => 0.01,
            Level::Thorough => 0.005,
        }
    }
    /// The end-to-end gate's allowance: how much slower a section's workloads may run on average
    /// under its tuned values than under the values before them, before the section is dropped.
    /// 3%, 2% and 1% (the user's decision, 2026-11-10), wider than the tie band at normal and
    /// thorough so that what one tune differs from the next by noise doesn't reject a section.
    pub fn gate(self) -> f64 {
        match self {
            Level::Quick => 0.03,
            Level::Normal => 0.02,
            Level::Thorough => 0.01,
        }
    }
    pub fn rounds(self) -> usize {
        match self {
            Level::Quick => 5,
            Level::Normal => 11,
            Level::Thorough => 21,
        }
    }
    /// The spread a probe's samples must come under before it is settled: (their interquartile
    /// range) / (their median). The level's thoroughness is this bound; `rounds` is only the
    /// fewest samples taken and `max_rounds` the most.
    pub fn spread(self) -> f64 {
        match self {
            Level::Quick => 0.06,
            Level::Normal => 0.04,
            Level::Thorough => 0.02,
        }
    }
    pub fn max_rounds(self) -> usize {
        6 * self.rounds()
    }
    /// The most time, in ns, one probe's samples may take in one pass. A probe that takes
    /// milliseconds a call (a whole tree, a whole exp) is one call a sample, and chasing the
    /// spread bound to `max_rounds` cost such a probe half a minute: with whole sections of them
    /// a thorough tune passed an hour. Within the budget a probe takes as many of its minimum
    /// rounds as fit (at least 5), then extra rounds only while unsettled and under budget.
    pub fn probe_budget_ns(self) -> f64 {
        match self {
            Level::Quick => 5e7,
            Level::Normal => 2.5e8,
            Level::Thorough => 1e9,
        }
    }
    /// Samples of each probe in a whole-algorithm measurement (`time_fixed`): a fixed count, as
    /// such probes are one call of milliseconds a sample and most never meet the spread bound.
    pub fn whole_rounds(self) -> usize {
        match self {
            Level::Quick => 5,
            Level::Normal => 7,
            Level::Thorough => 11,
        }
    }
    /// A batch's target length in ns.
    pub fn batch_ns(self) -> f64 {
        match self {
            Level::Quick => 2e5,
            Level::Normal => 1e6,
            Level::Thorough => 2e6,
        }
    }
    /// How many passes `time_min` takes.
    pub fn passes(self) -> usize {
        match self {
            Level::Quick => 1,
            Level::Normal => 3,
            Level::Thorough => 5,
        }
    }
    /// Rounds in one pass of `time_fit`, each one batch of every probe: enough for a median to
    /// drop one disturbed batch, at every level.
    pub fn fit_batches(self) -> usize {
        3
    }
    /// How many shapes a fit samples, from its count at the quick level: a level's thoroughness
    /// for a fit is its shapes, twice as many at normal and four times at thorough.
    pub fn fit_shapes(self, quick: usize) -> usize {
        quick * match self {
            Level::Quick => 1,
            Level::Normal => 2,
            Level::Thorough => 4,
        }
    }
}

/// Whether `BIGBITS_TUNE_VERBOSE` is set: every probe's time, and the diagnostics files.
pub fn verbose() -> bool {
    std::env::var_os("BIGBITS_TUNE_VERBOSE").is_some()
}

/// One thing to time: a closure, run in batches.
pub struct Probe<'a> {
    pub name: String,
    pub run: Box<dyn FnMut() + 'a>,
}

impl<'a> Probe<'a> {
    pub fn new(name: impl Into<String>, run: impl FnMut() + 'a) -> Self {
        Probe { name: name.into(), run: Box::new(run) }
    }
}

// Probes measured, and those whose samples never came under the level's spread.
static PROBES: AtomicUsize = AtomicUsize::new(0);
static UNSETTLED: AtomicUsize = AtomicUsize::new(0);

/// (sweep probes measured by `time_probes`, those that hit `max_rounds` or their time budget
/// still above the level's spread) since the last call, which resets both. Many unsettled probes
/// mean a busy machine, or one changing speed: tune again later. (Fit probes have `fit_noise`.)
pub fn settle_counts() -> (usize, usize) {
    (PROBES.swap(0, Ordering::Relaxed), UNSETTLED.swap(0, Ordering::Relaxed))
}

/// What `time_fit` measured.
pub struct Timed {
    /// ns per call of each probe: the least of its two passes (a disturbance only adds time).
    pub ns: Vec<f64>,
    /// One pass's noise, relative: `pass_noise` of the two passes.
    pub noise: f64,
}

// Fit probes measured, and the sum of their squared log pass ratios.
static FIT_NOISE: std::sync::Mutex<(usize, f64)> = std::sync::Mutex::new((0, 0.0));

/// (fit probes measured, their passes' noise as `pass_noise` gives it) since the last call, which
/// resets both: the tuner's check of the machine for fits. The noise takes in whatever doesn't
/// repeat from one pass to the next: timer noise, a machine changing speed, a buffer's placement.
pub fn fit_noise() -> (usize, f64) {
    let (n, sq) = std::mem::take(&mut *FIT_NOISE.lock().unwrap());
    (n, (sq / (2.0 * n.max(1) as f64)).sqrt())
}

/// The noise of one of two measurements of the same probes, relative: the rms of ln(a / b), over
/// √2 as the difference of two carries both's.
pub fn pass_noise(a: &[f64], b: &[f64]) -> f64 {
    let sq: f64 = a.iter().zip(b).map(|(x, y)| (x / y).ln().powi(2)).sum();
    (sq / (2.0 * a.len().max(1) as f64)).sqrt()
}

// 0..n in an order that depends on `seed` (Fisher-Yates over xorshift).
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, (xorshift(&mut x) % (i as u64 + 1)) as usize);
    }
    order
}

/// ns per call of the probes `build(pass)` makes, for a fit. In each of two passes the probes are
/// built anew (with `pass` to move their buffers: see `pad`), each is run once (warm caches and
/// scratch pools) and calibrated, and then `level.fit_batches()` rounds each time one batch of
/// every probe, in an order shuffled anew each round; a probe's time in the pass is the median
/// of its batches. The value is the least of the two passes, and their difference the noise.
///
/// Batches taken back to back, one probe at a time, were tried first (2026-11-11) and dropped:
/// a disturbance of a few milliseconds took all of one probe's batches, and in a process's first
/// second 5-7% of a fit's probes read up to 2.4x slow in one pass that way.
pub fn time_fit<'a>(level: Level, mut build: impl FnMut(usize) -> Vec<Probe<'a>>) -> Timed {
    let passes = [0, 1].map(|pass| {
        let mut probes = build(pass);
        let n = probes.len();
        let order = |round: usize| shuffled(n, (16 * pass + round) as u64 + 1);
        let mut iters = vec![0u64; n];
        for i in order(0) {
            iters[i] = calibrate(&mut probes[i], level);
        }
        let mut samples: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(level.fit_batches())).collect();
        for round in 0..level.fit_batches() {
            for i in order(round + 1) {
                samples[i].push(batch(&mut probes[i], iters[i]));
            }
        }
        let t: Vec<f64> = samples.into_iter().map(median).collect();
        if verbose() {
            for (p, ns) in probes.iter().zip(&t) {
                eprintln!("    {:<28} {ns:>12.2} ns", p.name);
            }
        }
        t
    });
    let noise = pass_noise(&passes[0], &passes[1]);
    {
        let mut total = FIT_NOISE.lock().unwrap();
        total.0 += passes[0].len();
        total.1 += 2.0 * passes[0].len() as f64 * noise * noise;
    }
    Timed { ns: passes[0].iter().zip(&passes[1]).map(|(a, b)| a.min(*b)).collect(), noise }
}

// ns per call over one batch of `iters` calls.
fn batch(p: &mut Probe, iters: u64) -> f64 {
    let t = Instant::now();
    for _ in 0..iters {
        (p.run)();
    }
    t.elapsed().as_nanos() as f64 / iters as f64
}

// Runs the probe once (warm caches and scratch pools) and finds how many calls make a batch of
// the level's length.
fn calibrate(p: &mut Probe, level: Level) -> u64 {
    (p.run)();
    let mut n = 1u64;
    loop {
        let per = batch(p, n);
        if per * n as f64 >= level.batch_ns() / 4.0 || n >= 1 << 24 {
            return ((level.batch_ns() / per.max(1.0)).ceil() as u64).clamp(1, 1 << 26);
        }
        n *= 4;
    }
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

// The value a fraction `f` of the way through sorted values, at the nearest index.
fn quantile(sorted: &[f64], f: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * f).round() as usize]
}

/// (interquartile range) / median of the samples.
pub fn spread(samples: &[f64]) -> f64 {
    let v = sorted(samples.to_vec());
    (quantile(&v, 0.75) - quantile(&v, 0.25)) / quantile(&v, 0.5)
}

/// ns per call of each probe: the median of its samples. Every probe is run warm (once, then
/// through calibration) before any sample is kept. Each round times one batch of every probe
/// still unsettled, in turn; a probe is settled once it has `level.rounds()` samples whose spread
/// is under `level.spread()`, and rounds stop at `level.max_rounds()`.
pub fn time_probes(probes: &mut [Probe], level: Level) -> Vec<f64> {
    sample(probes, level, None)
}

/// ns per call of each probe from exactly `level.whole_rounds()` interleaved samples (their
/// median), with no spread bound: for whole-algorithm probes, whose sweeps are judged by whether
/// two such measurements choose the same value (`sweep::param_sweep`, `sweep::cutoff_sweep_whole`).
pub fn time_fixed(probes: &mut [Probe], level: Level) -> Vec<f64> {
    sample(probes, level, Some(level.whole_rounds()))
}

fn sample(probes: &mut [Probe], level: Level, fixed: Option<usize>) -> Vec<f64> {
    let all: Vec<usize> = (0..probes.len()).collect();
    sample_at(probes, level, fixed, &all)
}

/// `sample` over the probes at `idx` only (the others don't run), in that order.
fn sample_at(probes: &mut [Probe], level: Level, fixed: Option<usize>, idx: &[usize]) -> Vec<f64> {
    let iters: Vec<u64> = idx.iter().map(|&i| calibrate(&mut probes[i], level)).collect();
    // Everything below indexes by position in idx.
    // Each probe's buffer at its most samples, so none grows between timed batches (a `vec!` of
    // one would clone it, and a clone keeps no capacity).
    let most = fixed.unwrap_or(level.max_rounds());
    let mut samples: Vec<Vec<f64>> = (0..idx.len()).map(|_| Vec::with_capacity(most)).collect();
    let mut spent = vec![0.0f64; idx.len()];
    let mut open: Vec<usize> = (0..idx.len()).collect();
    let mut unsettled = 0;
    for _ in 0..level.max_rounds() {
        for &i in &open {
            let t = Instant::now();
            for _ in 0..iters[i] {
                (probes[idx[i]].run)();
            }
            let ns = t.elapsed().as_nanos() as f64;
            spent[i] += ns;
            samples[i].push(ns / iters[i] as f64);
        }
        // A probe leaves once it has its minimum rounds (fewer if its budget ran out first, but
        // at least 5) and is either settled or out of budget.
        if let Some(rounds) = fixed {
            if samples.first().map_or(true, |v| v.len() >= rounds) {
                open.clear();
                break;
            }
            continue;
        }
        open.retain(|&i| {
            let (n, over) = (samples[i].len(), spent[i] >= level.probe_budget_ns());
            if n < 5 || (n < level.rounds() && !over) {
                return true;
            }
            let settled = spread(&samples[i]) <= level.spread();
            if !settled && over {
                unsettled += 1;
            }
            !settled && !over
        });
        if open.is_empty() {
            break;
        }
    }
    if fixed.is_none() {
        PROBES.fetch_add(idx.len(), Ordering::Relaxed);
        UNSETTLED.fetch_add(unsettled + open.len(), Ordering::Relaxed);
    }
    let t: Vec<f64> = samples.into_iter().map(median).collect();
    if verbose() {
        for (&i, ns) in idx.iter().zip(&t) {
            eprintln!("    {:<28} {ns:>12.2} ns", probes[i].name);
        }
    }
    t
}

pub fn median(v: Vec<f64>) -> f64 {
    let v = sorted(v);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// ns per call of the probes `build(pass)` makes: the least over the level's passes of
/// `time_probes`. Each pass builds its probes anew, with `pass` to vary where their buffers land
/// (see `pad`): a disturbance, such as an unlucky buffer placement that slows one probe for a
/// whole pass, only ever adds time, so the least pass is the cleanest.
pub fn time_min<'a>(level: Level, mut build: impl FnMut(usize) -> Vec<Probe<'a>>) -> Vec<f64> {
    let mut best: Vec<f64> = Vec::new();
    for pass in 0..level.passes() {
        let mut probes = build(pass);
        let t = time_probes(&mut probes, level);
        if best.is_empty() {
            best = t;
        } else {
            for (b, v) in best.iter_mut().zip(t) {
                *b = b.min(v);
            }
        }
    }
    best
}

/// Runs `f` on a rayon pool of one thread, so a dynamic entry inside it sees one thread and stays
/// serial: for decisions the library defines on one thread and scales for more.
pub fn on_one_thread<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(1).stack_size(64 << 20).build().expect("a one-thread pool"))
        .install(f)
}

/// A buffer of `n` limbs from `seed`, placed `13 * pass + 1` limbs into its allocation, so each
/// pass puts a probe's buffers at different offsets from each other.
pub fn pad(n: usize, seed: u64, pass: usize) -> PadBuf {
    let off = 13 * pass + 1;
    let mut x = seed | 1;
    PadBuf { v: (0..n + off).map(|_| xorshift(&mut x)).collect(), off }
}

// One step of a 64-bit xorshift generator, from a nonzero state.
fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

pub struct PadBuf {
    v: Vec<u64>,
    off: usize,
}

impl std::ops::Deref for PadBuf {
    type Target = [u64];
    fn deref(&self) -> &[u64] {
        &self.v[self.off..]
    }
}

impl std::ops::DerefMut for PadBuf {
    fn deref_mut(&mut self) -> &mut [u64] {
        &mut self.v[self.off..]
    }
}
