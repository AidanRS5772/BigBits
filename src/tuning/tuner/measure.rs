//! Timing: each probe runs in batches long enough that the clock's resolution doesn't matter.
//! For a fit (`time_fit`): a fixed few batches of every probe, in rounds that visit every probe
//! once in a shuffled order, the whole set twice; no spread bound, as a fit gains from more
//! shapes and not from a tighter time at each. For a sweep, whose decision is the ratio of two
//! times at one size (`time_min`, `time_fixed`): rounds interleave every probe so drift cancels,
//! and a probe's value is the median of its rounds.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Asks the OS to run the calling thread on performance cores. macOS: the user-interactive QoS
/// class (a loaded M1 can put a process on its efficiency cores, half as fast). Linux: pinned to
/// a hybrid CPU's fastest cores (`fastest_of`), a pool worker only when the whole pool fits on
/// them; compile-checked only, never run on a hybrid Linux machine. Elsewhere nothing.
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

/// Each CPU's speed rating from sysfs: `cpu_capacity`, else `cpufreq/cpuinfo_max_freq`.
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

/// The CPUs to measure on: those within 15% of the highest rating (a hybrid Intel CPU's
/// performance cores, not its efficiency ones), or None when that is all of them or none is rated.
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
    // A value by level: [quick, normal, thorough].
    fn of<T: Copy>(self, v: [T; 3]) -> T {
        v[self as usize]
    }
    pub fn parse(s: &str) -> Option<Level> {
        [Level::Quick, Level::Normal, Level::Thorough].into_iter().find(|l| l.name() == s)
    }
    pub fn name(self) -> &'static str {
        self.of(["quick", "normal", "thorough"])
    }
    /// The tie band.
    pub fn eps(self) -> f64 {
        self.of([0.03, 0.01, 0.005])
    }
    /// The end-to-end gate's allowance: how much slower a section's workloads may run on average
    /// under its tuned values before it is dropped. Wider than the tie band, so that noise from
    /// one tune to the next doesn't reject a section.
    pub fn gate(self) -> f64 {
        self.of([0.03, 0.02, 0.01])
    }
    /// The fewest samples of a sweep probe.
    pub fn rounds(self) -> usize {
        self.of([5, 11, 21])
    }
    /// The spread a sweep probe's samples must come under to settle, (interquartile range) /
    /// median: the level's thoroughness for a sweep.
    pub fn spread(self) -> f64 {
        self.of([0.06, 0.04, 0.02])
    }
    pub fn max_rounds(self) -> usize {
        6 * self.rounds()
    }
    /// The most ns one probe's samples may take in a pass: a probe of milliseconds a call is one
    /// call a sample, and chasing the spread bound to `max_rounds` cost such probes minutes.
    pub fn probe_budget_ns(self) -> f64 {
        self.of([5e7, 2.5e8, 1e9])
    }
    /// Samples of each probe in a whole-algorithm measurement (`time_fixed`).
    pub fn whole_rounds(self) -> usize {
        self.of([5, 7, 11])
    }
    /// A batch's target length in ns.
    pub fn batch_ns(self) -> f64 {
        self.of([2e5, 1e6, 2e6])
    }
    /// How many passes `time_min` takes.
    pub fn passes(self) -> usize {
        self.of([1, 3, 5])
    }
    /// The level below, for what is timed in great numbers and only has to be put in order (a
    /// table's indices).
    pub fn lighter(self) -> Level {
        self.of([Level::Quick, Level::Quick, Level::Normal])
    }
    /// How many shapes a fit samples, from its count at the quick level: the level's
    /// thoroughness for a fit.
    pub fn fit_shapes(self, quick: usize) -> usize {
        quick * self.of([1, 2, 4])
    }
}

/// Whether `BIGBITS_TUNE_VERBOSE` is set: every probe's time, and the dump files.
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

// Sweep probes measured, and those whose samples never came under the level's spread.
static PROBES: AtomicUsize = AtomicUsize::new(0);
static UNSETTLED: AtomicUsize = AtomicUsize::new(0);

/// (sweep probes measured, those still unsettled at `max_rounds` or their budget) since the last
/// call, which resets both. Many unsettled: a busy machine, or one changing speed.
pub fn settle_counts() -> (usize, usize) {
    (PROBES.swap(0, Ordering::Relaxed), UNSETTLED.swap(0, Ordering::Relaxed))
}

/// What `time_fit` measured.
pub struct Timed {
    /// ns per call of each probe: the least of its two passes (a disturbance only adds time).
    pub ns: Vec<f64>,
    /// One pass's noise, relative (`pass_noise`).
    pub noise: f64,
}

// Fit probes measured, and the sum of their squared log pass ratios.
static FIT_NOISE: std::sync::Mutex<(usize, f64)> = std::sync::Mutex::new((0, 0.0));

/// (fit probes measured, their passes' noise) since the last call, which resets both.
pub fn fit_noise() -> (usize, f64) {
    let (n, sq) = std::mem::take(&mut *FIT_NOISE.lock().unwrap());
    (n, (sq / (2.0 * n.max(1) as f64)).sqrt())
}

/// The noise of one of two measurements of the same probes: the rms of ln(a / b), over sqrt 2.
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

/// Rounds in a pass of `time_fit`: enough for a median to drop one disturbed batch.
const FIT_BATCHES: usize = 3;

/// ns per call of the probes `build(pass)` makes, for a fit. In each of two passes the probes are
/// built anew (`pass` moves their buffers: `pad`), run once and calibrated, then timed a batch
/// each in `FIT_BATCHES` rounds, shuffled anew each round; a probe's time in a pass is the median
/// of its batches, the value the least of the two passes, their difference the noise. (Batches
/// taken back to back lost all of a probe's batches to one disturbance.)
pub fn time_fit<'a>(level: Level, mut build: impl FnMut(usize) -> Vec<Probe<'a>>) -> Timed {
    let passes = [0, 1].map(|pass| {
        let mut probes = build(pass);
        let n = probes.len();
        let order = |round: usize| shuffled(n, (16 * pass + round) as u64 + 1);
        let mut iters = vec![0u64; n];
        for i in order(0) {
            iters[i] = calibrate(&mut probes[i], level);
        }
        let mut samples: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(FIT_BATCHES)).collect();
        for round in 0..FIT_BATCHES {
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

// Runs the probe once (warm caches and scratch pools), then finds how many calls make a batch.
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

/// (interquartile range) / median of the samples.
pub fn spread(samples: &[f64]) -> f64 {
    let v = sorted(samples.to_vec());
    // The value a fraction of the way through, at the nearest index.
    let at = |f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
    (at(0.75) - at(0.25)) / at(0.5)
}

/// ns per call from exactly `level.whole_rounds()` interleaved samples, no spread bound: for
/// whole-algorithm probes, whose sweeps are judged by whether two such measurements agree.
pub fn time_fixed(probes: &mut [Probe], level: Level) -> Vec<f64> {
    sample(probes, level, Some(level.whole_rounds()))
}

// ns per call of each probe, the median of its samples. Each round times one batch of every
// probe still unsettled; a probe settles once it has `level.rounds()` samples whose spread is
// under `level.spread()`, and rounds stop at `level.max_rounds()`. With `fixed`: that many
// rounds of every probe, no spread bound.
fn sample(probes: &mut [Probe], level: Level, fixed: Option<usize>) -> Vec<f64> {
    let n = probes.len();
    let iters: Vec<u64> = probes.iter_mut().map(|p| calibrate(p, level)).collect();
    // Each probe's buffer at its most samples: none grows between timed batches.
    let most = fixed.unwrap_or(level.max_rounds());
    let mut samples: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(most)).collect();
    let mut spent = vec![0.0f64; n];
    let mut open: Vec<usize> = (0..n).collect();
    let mut unsettled = 0;
    for _ in 0..most {
        for &i in &open {
            let t = Instant::now();
            for _ in 0..iters[i] {
                (probes[i].run)();
            }
            let ns = t.elapsed().as_nanos() as f64;
            spent[i] += ns;
            samples[i].push(ns / iters[i] as f64);
        }
        // A fixed count is every round of every probe.
        if fixed.is_some() {
            continue;
        }
        // A probe leaves once it has its minimum rounds (at least 5) and is settled or out of budget.
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
        PROBES.fetch_add(n, Ordering::Relaxed);
        UNSETTLED.fetch_add(unsettled + open.len(), Ordering::Relaxed);
    }
    let t: Vec<f64> = samples.into_iter().map(median).collect();
    if verbose() {
        for (p, ns) in probes.iter().zip(&t) {
            eprintln!("    {:<28} {ns:>12.2} ns", p.name);
        }
    }
    t
}

/// A timing less the harness's own work inside it, timed on its own; the floor only keeps a
/// mistimed pair finite.
pub fn less_harness(t: f64, harness: f64) -> f64 {
    (t - harness).max(0.25 * t)
}

/// Writes a developer's dump to target/tuning/`name`; a failure is nothing to stop a tune for.
pub fn dump(name: &str, text: &str) {
    let _ = std::fs::create_dir_all("target/tuning");
    let _ = std::fs::write(format!("target/tuning/{name}"), text);
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
/// `sample`, each pass building its probes anew (a disturbance only adds time).
pub fn time_min<'a>(level: Level, mut build: impl FnMut(usize) -> Vec<Probe<'a>>) -> Vec<f64> {
    let mut best: Vec<f64> = Vec::new();
    for pass in 0..level.passes() {
        let mut probes = build(pass);
        let t = sample(&mut probes, level, None);
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

/// Runs `f` on a rayon pool of one thread, so a dynamic entry inside it stays serial.
pub fn on_one_thread<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(1).stack_size(64 << 20).build().expect("a one-thread pool"))
        .install(f)
}

/// A buffer of `n` limbs from `seed`, `13 * pass + 1` limbs into its allocation: each pass puts
/// a probe's buffers at different offsets from each other.
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
