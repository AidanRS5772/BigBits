//! Bias and identity checks for the tuning system's migrations (docs/tuning_system.md,
//! "Validation" 2, and docs/tuning_implementation.md, Stage 1). It times public entries only, so
//! it builds against the `faa9971` baseline too: copy this file and
//! `src/tuning/tuner/check_items.rs` (the items) into that worktree, point the `#[path]` below at
//! the copy, and add the `[[bench]]` entry.
//!
//!   tuning_bias --worker --section S [--rounds R]   time S's items: `name<TAB>median ns<TAB>hash`
//!   tuning_bias --hash [--section S]                 outputs only: `name<TAB>hash`
//!   tuning_bias --compare A B [--procs P] [--section S] [--rounds R]
//!       runs executables A and B as workers, alternating over P processes each, and prints per
//!       item the geometric mean of B/A over the paired processes with a 95% interval, and
//!       whether their output hashes agree.
//!
//! Sections: mul, div, sqrt, series (the series evaluators on synthetic series: `synth.rs`),
//! consts, trans (theta, agm, exp, ln; exp's and ln's items that were "series" until 2026-11-11
//! are here), all. Shapes sit around the Phase 1 values, so they cross
//! the knobs being migrated.
//!
//! Bias check (normal vs tune build of one tree):
//!   cargo bench --bench tuning_bias --no-run; cargo bench --bench tuning_bias --features tune --no-run
//!   (copy each printed executable aside), then tuning_bias --compare NORMAL TUNE --section S
//! Identity check (this tree vs the faa9971 worktree): --hash on both, or --compare.

use std::process::Command;
use std::time::Instant;

// The items, shared with the tuner's end-to-end check.
#[path = "../src/tuning/tuner/synth.rs"]
mod synth;
#[path = "../src/tuning/tuner/check_items.rs"]
mod check_items;
use check_items::{items, Item, T95};

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn selected(section: &str) -> Vec<Item> {
    items().into_iter().filter(|i| section == "all" || i.section == section).collect()
}

fn worker(section: &str, rounds: usize) {
    let mut its = selected(section);
    // Batch sizes give each timed batch about 2 ms, so the clock's resolution doesn't matter.
    let batch: Vec<u32> = its
        .iter_mut()
        .map(|it| {
            (it.run)(false);
            let t = Instant::now();
            (it.run)(false);
            let ns = t.elapsed().as_nanos().max(1) as f64;
            (2e6 / ns).clamp(1.0, 1e6) as u32
        })
        .collect();
    let mut samples = vec![Vec::with_capacity(rounds); its.len()];
    for _ in 0..rounds {
        // Rounds interleave every item, so drift cancels between the two builds' processes.
        for (i, it) in its.iter_mut().enumerate() {
            let t = Instant::now();
            for _ in 0..batch[i] {
                (it.run)(false);
            }
            samples[i].push(t.elapsed().as_nanos() as f64 / batch[i] as f64);
        }
    }
    for (i, it) in its.iter_mut().enumerate() {
        samples[i].sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = samples[i][samples[i].len() / 2];
        println!("{}\t{med:.1}\t{:016x}", it.name, (it.run)(true));
    }
}

fn run_worker(exe: &str, section: &str, rounds: usize) -> Vec<(String, f64, String)> {
    let out = Command::new(exe)
        .args(["--worker", "--section", section, "--rounds", &rounds.to_string()])
        .output()
        .expect("run worker");
    assert!(out.status.success(), "{exe} failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut p = l.split('\t');
            Some((p.next()?.to_string(), p.next()?.parse().ok()?, p.next()?.to_string()))
        })
        .collect()
}

fn compare(a: &str, b: &str, procs: usize, section: &str, rounds: usize) {
    let mut ra = Vec::new();
    let mut rb = Vec::new();
    for p in 0..procs {
        // Alternate which runs first, so a trend in machine load cancels too.
        if p % 2 == 0 {
            ra.push(run_worker(a, section, rounds));
            rb.push(run_worker(b, section, rounds));
        } else {
            rb.push(run_worker(b, section, rounds));
            ra.push(run_worker(a, section, rounds));
        }
    }
    println!("{:<34} {:>9} {:>16}  hashes", "item (B/A)", "ratio", "95% interval");
    let mut all = Vec::new();
    let mut mismatched = 0;
    for (i, (name, _, ha)) in ra[0].iter().enumerate() {
        let logs: Vec<f64> = (0..procs).map(|p| (rb[p][i].1 / ra[p][i].1).ln()).collect();
        let n = logs.len() as f64;
        let mean = logs.iter().sum::<f64>() / n;
        let sd = (logs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
        let half = T95[(procs - 2).min(29)] * sd / n.sqrt();
        let same = rb[0][i].2 == *ha;
        mismatched += !same as usize;
        all.push(mean);
        println!(
            "{name:<34} {:>9.4} [{:.4}, {:.4}]  {}",
            mean.exp(),
            (mean - half).exp(),
            (mean + half).exp(),
            if same { "same" } else { "DIFFER" }
        );
    }
    let g = (all.iter().sum::<f64>() / all.len() as f64).exp();
    println!("geometric mean over {} items: {g:.4}; {mismatched} hash mismatches", all.len());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let section: String = arg(&args, "--section", "all".to_string());
    let rounds: usize = arg(&args, "--rounds", 11);
    if args.iter().any(|a| a == "--worker") {
        worker(&section, rounds);
    } else if args.iter().any(|a| a == "--hash") {
        for mut it in selected(&section) {
            println!("{}\t{:016x}", it.name, (it.run)(true));
        }
    } else if let Some(i) = args.iter().position(|a| a == "--compare") {
        let procs: usize = arg(&args, "--procs", 6);
        assert!(procs >= 2, "--procs needs at least 2");
        compare(&args[i + 1], &args[i + 2], procs, &section, rounds);
    } else if args.iter().any(|a| a == "--bench") {
        // `cargo bench` passes --bench: print the hashes, the cheapest useful default.
        for mut it in selected(&section) {
            println!("{}\t{:016x}", it.name, (it.run)(true));
        }
    } else {
        eprintln!("usage: tuning_bias --worker|--hash|--compare A B [--section S] [--procs P] [--rounds R]");
    }
}
