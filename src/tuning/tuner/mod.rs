//! The tuner (docs/tuning_system.md, "The tuner"), in tune builds only. `cargo tune` runs `cli()`
//! through `examples/tune.rs`. It measures each section's model knobs and sweeps, sets them live,
//! and merges them into a tuning file with their evidence.

pub mod check;
mod check_items;
pub mod derive;
pub mod design;
pub mod fit;
pub mod measure;
pub mod out;
pub mod sections;
pub mod sweep;
pub mod synth;

use crate::tuning::{self, file, resolve, schema_types::Kind, schema_types::Section, Value};
use measure::Level;
use std::path::PathBuf;

/// The machine the tuner runs on: tuning files are resolved for it and record it.
pub(crate) const HOST: resolve::Target<'static> = resolve::Target { arch: std::env::consts::ARCH, os: std::env::consts::OS };

const USAGE: &str = "usage: cargo tune <command>
  tune [SECTION ...] [--level quick|normal|thorough] [--out FILE] [--write-preset] [--threads N] [--no-gate] [--analyze]
      measure the sections' model knobs (all registered sections when none are named) and merge
      them into FILE. The default is this machine's own file, the one BIGBITS_TUNING names
      (presets/local.toml in a checkout), which the next build layers over the target's preset;
      --write-preset writes presets/<target>.toml itself. Each section ends with the end-to-end
      check as a gate: if its values make its workloads slower on average than the values
      before it, by more than 3% (quick), 2% (normal) or 1% (thorough), the section is dropped
      and those values stay (--no-gate: keep it anyway). --analyze is for whoever writes an
      equation, not for tuning a machine: each fit also prints which of its form's terms the
      timings need (each constant against its standard error, and every form short of some terms)
  check [--file FILE] [SECTION ...] [--level L]
      the end-to-end check on its own: public calls timed under FILE's values (default: this
      machine's own file) and under the layers below it, the preset and the defaults
  derive [--file FILE] [mul] [div] [series] [--check] [--compare]
                       derive from the measured values (FILE's, else the build's); the pilot reports
                       mul.kara (schoolbook / one Karatsuba level) against today's is_school, and
                       --check times both sides and the level's parts against their predictions;
                       --compare times mul_dyn under the build's table and the derived one.
                       Naming sections rewrites only theirs in FILE: mul's table, div's Knuth
                       guards, the series' small-series table
  equations [--level L]
                       Karatsuba against the FFT for dynamic products: the measured time lost by
                       comparing the fitted equations and by mul_dyn itself, on shapes the
                       equations weren't fitted on
  division [--level L] [--out FILE] [--refit]
                       Knuth, BZ and NR forced over quotient and divisor widths no tune fits the
                       equations to, for div_dyn and div_rem_dyn: timings to FILE, and on the half
                       held out the measured time lost by the fitted equations' picks (--refit:
                       fitted now, to the other half) and by the entries themselves
  series [--level L] [--set 'PATH=VALUE;...'] [--timings FILE] [--rederive]
                       the series equations and guards on synthetic series no tune takes: the
                       measured time lost by the picks under the live prices and under refitted
                       ones, and by the entry itself (--set: with those knob values)
  list [SECTION]       every knob: kind, valid range, live value
  status [--file FILE] the layers a build of this checkout resolves (the target's preset, then
                       this machine's file) and which supplies each section; with --file, that
                       one file against this crate's schema";

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

// The flags that take no value; every other `--flag` is followed by one.
const SWITCHES: [&str; 7] = ["--write-preset", "--refit", "--check", "--compare", "--no-gate", "--analyze", "--rederive"];

fn switch(args: &[String], name: &str) -> bool {
    debug_assert!(SWITCHES.contains(&name), "`positional` must know {name} takes no value");
    args.iter().any(|a| a == name)
}

fn level_flag(args: &[String]) -> Result<Level, String> {
    match flag(args, "--level") {
        Some(l) => Level::parse(&l).ok_or_else(|| format!("unknown level `{l}`")),
        None => Ok(Level::Normal),
    }
}

fn positional(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") {
            i += if SWITCHES.contains(&args[i].as_str()) { 1 } else { 2 };
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    out
}

/// The tuner's entry point, reading the process's arguments.
pub fn cli() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("cargo tune: {e}\n\n{USAGE}");
        std::process::exit(2);
    }
}

pub fn run(args: &[String]) -> Result<(), String> {
    let Some(cmd) = args.first() else { return Err("no command".into()) };
    let rest = &args[1..];
    match cmd.as_str() {
        "tune" => tune(rest),
        "list" => {
            list(positional(rest).first().map(String::as_str));
            Ok(())
        }
        "derive" => derive(flag(rest, "--file"), &positional(rest), switch(rest, "--check"), switch(rest, "--compare")),
        "equations" => {
            let level = level_flag(rest)?;
            setup_pool(None)?;
            measure::settle_counts();
            sections::mul::equations(level);
            samples_line(level);
            Ok(())
        }
        "division" => {
            let level = level_flag(rest)?;
            setup_pool(None)?;
            let out = flag(rest, "--out").unwrap_or_else(|| "target/tuning/division.tsv".into());
            measure::settle_counts();
            sections::div::division(level, std::path::Path::new(&out), switch(rest, "--refit")).map_err(|e| e.to_string())?;
            fit_line();
            Ok(())
        }
        "series" => {
            let level = level_flag(rest)?;
            setup_pool(None)?;
            // --set 'path=value;...': knob values for this run, for experiments.
            if let Some(set) = flag(rest, "--set") {
                tuning::apply(&set).map_err(|e| e.to_string())?;
            }
            measure::settle_counts();
            // --rederive: the small-series table derived now from the equations in force, for
            // this run alone (a change to its classes or grid read without a tune).
            if switch(rest, "--rederive") {
                let (table, evidence) = derive::series_small_table(level.eps());
                tuning::set("series.small", Value::Table(table)).map_err(|e| e.to_string())?;
                println!("series.small, for this run: {evidence}");
            }
            let cache = flag(rest, "--timings").map(PathBuf::from);
            sections::series::report(level, cache.as_deref());
            fit_line();
            Ok(())
        }
        "check" => {
            let level = level_flag(rest)?;
            setup_pool(None)?;
            check_file(flag(rest, "--file"), &positional(rest), level)
        }
        "status" => match flag(rest, "--file") {
            Some(f) => status(&f),
            None => status_layers(),
        },
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command `{other}`")),
    }
}

fn tune(args: &[String]) -> Result<(), String> {
    let level = level_flag(args)?;
    let named = positional(args);
    for n in &named {
        if !sections::sections().iter().any(|(s, _)| s == n) {
            return Err(format!("section `{n}` has nothing to measure yet"));
        }
    }
    let local = machine_file();
    let out = if switch(args, "--write-preset") {
        if let Some(l) = local.as_ref().filter(|l| l.exists()) {
            eprintln!(
                "warning: {} is in force over the preset in this checkout: this tune starts from its values, and builds keep using it until it is deleted",
                l.display()
            );
        }
        PathBuf::from(format!("presets/{}.toml", HOST.name()))
    } else {
        match flag(args, "--out") {
            Some(o) => PathBuf::from(o),
            None => local.clone().unwrap_or_else(|| PathBuf::from("target/tuning/latest.toml")),
        }
    };
    let threads: usize = match flag(args, "--threads") {
        Some(t) => t.parse().map_err(|_| format!("bad --threads `{t}`"))?,
        None => std::thread::available_parallelism().map_or(1, |c| c.get()),
    };
    // The file being merged into is the layer below: sections not measured now keep its values,
    // so what is measured and derived here reads them (mul's equations under a `tune div`). A file
    // another machine's target wrote (a checkout shared between machines, or a cross run) isn't
    // this machine's to merge into.
    if out.exists() {
        let name = out.to_string_lossy();
        let f = read_file(&name)?;
        resolve::layer_target_ok(&f, &HOST).map_err(|why| format!("{name}: {why}; delete it or name another file with --out"))?;
        load(&name, f)?;
    }
    setup_pool(Some(threads))?;
    fit::set_analyze(switch(args, "--analyze"));
    measure::settle_counts();
    measure::fit_noise();
    sweep::stability_counts();
    sweep::edge_count();
    // Evidence of every rejected fit, and of every noisy one whose model was already revised
    // once (one fit sets several knobs with the same evidence).
    let (mut rejected, mut noisy): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    let mut note = |section: &str, tuned: &[out::Tuned]| {
        for t in tuned {
            let line = format!("{section}.{}: {}", t.key, t.evidence);
            for (mark, list) in [(fit::REJECTED, &mut rejected), (fit::NOISY, &mut noisy)] {
                if t.evidence.contains(mark) && !list.iter().any(|r| r.ends_with(&t.evidence)) {
                    list.push(line.clone());
                }
            }
        }
    };
    // The gate (unless --no-gate): each section's values against the values before it, on its
    // own workloads and those of every section built on it. (section, why) of each one dropped.
    let gate = !switch(args, "--no-gate");
    let mut dropped: Vec<(String, String)> = Vec::new();
    let mut kept: Vec<String> = Vec::new();
    for (name, tune_section) in &sections::sections() {
        if !named.is_empty() && !named.iter().any(|n| n == name) {
            continue;
        }
        eprintln!("tuning {name} ({})...", level.name());
        let (before, file_before) = (tuning::snapshot(), std::fs::read_to_string(&out).ok());
        let tuned = tune_section(level);
        set_tuned(name, &tuned)?;
        note(name, &tuned);
        let values: Vec<Value> = tuning::PATHS.iter().map(|p| tuning::get(p).unwrap()).collect();
        let section = section(name);
        let fs = out::section(section, &tuned, level.name(), &values);
        // Each section is written as soon as it finishes, so an interrupted tune keeps it.
        out::merge_into(&out, vec![fs], threads).map_err(|e| e.to_string())?;
        if *name == "mul" {
            // mul's second phase runs under the leaf boundaries derived from its first, and
            // before the sections that are measured on top of mul.
            derive_into(&out, level.eps(), true, true)?;
            eprintln!("tuning mul, second phase ({})...", level.name());
            let tuned = sections::mul::tune_late(level);
            set_tuned("mul", &tuned)?;
            note("mul", &tuned);
            out::set_derived(&out, section, &tuned).map_err(|e| e.to_string())?;
        }
        if *name == "series" {
            // The small-series table follows from the equations just fitted, and is in force
            // for the gate and for the sections measured on top of the series.
            derive_small(&out, level.eps())?;
        }
        if gate {
            let after = tuning::snapshot();
            let rounds = check::compare(&before, &after, level, &|it| check::in_gate(tuning::schema(), name, it.section));
            let v = check::verdict(&rounds, level.gate());
            check::report(name, &v);
            if v.dropped {
                // The values and the file go back to what they were before this section.
                tuning::restore(&before);
                match &file_before {
                    Some(text) => std::fs::write(&out, text),
                    None => std::fs::remove_file(&out),
                }
                .map_err(|e| format!("{}: {e}", out.display()))?;
                let why = if v.differ.is_empty() {
                    format!("its workloads ran {:.1}% slower than under the values before it [{:.4}, {:.4}]", 100.0 * (v.all.ratio - 1.0), v.all.lo, v.all.hi)
                } else {
                    format!("exact outputs differ under its values ({})", v.differ.join(", "))
                };
                eprintln!("  DROPPED {name}: {why}; the values before it stay");
                dropped.push((name.to_string(), why));
                continue;
            }
        }
        kept.push(name.to_string());
    }
    // The derived values follow the measured ones this tune kept: mul's table when mul was tuned
    // and kept (a table rederived from fits that didn't move would pass no gate), and div's
    // guards when either section's equations moved.
    let is_kept = |s: &str| kept.iter().any(|k| k == s);
    derive_into(&out, level.eps(), is_kept("mul"), is_kept("mul") || is_kept("div"))?;
    if out.exists() {
        eprintln!("wrote {}", out.display());
    } else {
        eprintln!("nothing written: no section was kept");
    }
    if local.as_ref() == Some(&out) && out.exists() {
        eprintln!("  it is this machine's own file: the next build of this checkout layers it over the preset (delete it to build with the preset again)");
    }
    if gate {
        eprintln!(
            "gate: {} of {} sections dropped (more than {:.0}% slower on average than the values before them, which stay in force)",
            dropped.len(),
            dropped.len() + kept.len(),
            100.0 * level.gate()
        );
        for (name, why) in &dropped {
            eprintln!("  {name}: {why}");
        }
    }
    samples_line(level);
    fit_line();
    let (whole, unstable) = sweep::stability_counts();
    eprintln!("stability: {unstable} of {whole} whole-algorithm sweeps disagreed between their two measurements (the previous value is kept)");
    let edges = sweep::edge_count();
    if edges > 0 {
        eprintln!("edges: {edges} cutoff sweeps chose a value at an end of the sizes they timed, so the real crossover may lie outside (their evidence says which end)");
    }
    eprintln!(
        "fits: {} rejected (relative rms above {:.0}%: report each with a proposed form and wait); {} noisy but accepted (above it after one revision of the model: marked problems)",
        rejected.len(),
        100.0 * fit::ACCEPT_RMS,
        noisy.len()
    );
    for r in rejected.iter().chain(&noisy) {
        eprintln!("  {r}");
    }
    Ok(())
}

// Every tune ends with a derive, so a file's derived knobs match its measured ones: computes each
// derived knob from the live values, sets it live and writes it to `out`. This first step is the
// leaf boundaries, which mul's second phase is measured under.
fn derive_into(out: &std::path::Path, eps: f64, kara: bool, guards: bool) -> Result<(), String> {
    if kara {
        let (table, evidence) = derive::kara_table(eps);
        let v = Value::Table(table);
        tuning::set("mul.kara", v.clone()).map_err(|e| e.to_string())?;
        eprintln!("  mul.kara: {evidence}");
        let tuned = [out::Tuned { key: "kara", value: v, evidence }];
        out::set_derived(out, section("mul"), &tuned).map_err(|e| e.to_string())?;
    }
    if !guards {
        return Ok(());
    }
    // div's Knuth guards follow from its equations and mul's, and time nothing either.
    let tuned = sections::div::guards();
    set_tuned("div", &tuned)?;
    out::set_derived(out, section("div"), &tuned).map_err(|e| e.to_string())
}

// The series' small-series table, derived from the equations in force (it times nothing): set
// live and written to `out`.
fn derive_small(out: &std::path::Path, eps: f64) -> Result<(), String> {
    let (table, evidence) = derive::series_small_table(eps);
    let v = Value::Table(table);
    tuning::set("series.small", v.clone()).map_err(|e| e.to_string())?;
    eprintln!("  series.small: {evidence}");
    let tuned = [out::Tuned { key: "small", value: v, evidence }];
    out::set_derived(out, section("series"), &tuned).map_err(|e| e.to_string())
}

// The schema's declaration of a section the tuner names.
fn section(name: &str) -> &'static Section {
    tuning::schema().iter().find(|s| s.name == name).expect("the tuner's sections are declared")
}

// Sets `section`'s tuned values live and prints each with its evidence.
fn set_tuned(section: &str, tuned: &[out::Tuned]) -> Result<(), String> {
    for t in tuned {
        tuning::set(&format!("{section}.{}", t.key), t.value.clone()).map_err(|e| e.to_string())?;
        eprintln!("  {section}.{} = {}   # {}", t.key, t.value.text(), t.evidence);
    }
    Ok(())
}

fn read_file(path: &str) -> Result<file::File, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    file::parse(&text).map_err(|e| format!("{path}: {e}"))
}

// What `files` resolve to on this machine, lowest layer first.
fn resolve_host(files: &[(String, file::File)]) -> Result<resolve::Resolved, String> {
    let layers: Vec<resolve::Layer> = files.iter().map(|(name, file)| resolve::Layer { name, file }).collect();
    resolve::resolve(tuning::schema(), &layers, &HOST).map_err(|e| e.join("
"))
}

// Sets every resolved value live.
fn set_values(r: resolve::Resolved) -> Result<(), String> {
    for (p, v) in tuning::PATHS.iter().zip(r.values) {
        tuning::set(p, v).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn list(section: Option<&str>) {
    for f in resolve::flat(tuning::schema()) {
        if section.is_some_and(|s| s != f.section.name) {
            continue;
        }
        let kind = match f.knob.kind {
            Kind::Interim => "interim".to_string(),
            Kind::Model { .. } => "model".into(),
            Kind::Sweep => "sweep".into(),
            Kind::Derived(d) => format!("derived {d:?}").to_lowercase(),
        };
        let v = tuning::get(&f.path).map(|v| v.text()).unwrap_or_default();
        let v = if v.len() > 60 { format!("{}…", &v[..60]) } else { v };
        println!("{:<44} {:<18} {v}", f.path, kind);
    }
}

// Sets every value `files` resolve to live (lowest layer first), and returns them as a snapshot.
fn set_live(files: &[(String, file::File)]) -> Result<tuning::Snapshot, String> {
    set_values(resolve_host(files)?)?;
    Ok(tuning::snapshot())
}

// `cargo tune check`: FILE's values (this machine's own file when none is named) against the
// layers below it, over every workload or those of the named sections' gates.
fn check_file(path: Option<String>, named: &[String], level: Level) -> Result<(), String> {
    for n in named {
        if !tuning::schema().iter().any(|s| s.name == n) {
            return Err(format!("no section `{n}`"));
        }
    }
    let machine = machine_file();
    let path = match (path, &machine) {
        (Some(p), _) => PathBuf::from(p),
        (None, Some(m)) => m.clone(),
        (None, None) => return Err("no --file, and BIGBITS_TUNING isn't set here (run through `cargo tune`)".into()),
    };
    let name = path.display().to_string();
    let f = read_file(&name)?;
    // The layers below FILE: the preset, unless FILE is the preset (then the schema's defaults).
    // The machine's own file is left out, as FILE stands in its place.
    let same = |a: &str, b: &std::path::Path| match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => std::path::Path::new(a) == b,
    };
    let mut layers = layers_in_force()?;
    layers.retain(|(n, _)| !same(n, &path) && !machine.as_ref().is_some_and(|m| same(n, m)));
    let below = if layers.is_empty() { "the schema's defaults".to_string() } else { layers.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ") };
    let before = set_live(&layers)?;
    layers.push((name.clone(), f));
    let after = set_live(&layers)?;
    println!("check ({}): {name} against {below}; ratios are time under {name} over time under {below}", level.name());
    let rounds = check::compare(&before, &after, level, &|it| named.is_empty() || named.iter().any(|n| check::in_gate(tuning::schema(), n, it.section)));
    for sec in ["mul", "div", "divgrid", "sqrt", "series", "consts", "trans"] {
        let of: Vec<check::Rounds> = rounds.iter().filter(|r| r.section == sec).cloned().collect();
        if !of.is_empty() {
            check::report(sec, &check::verdict(&of, level.gate()));
        }
    }
    let v = check::verdict(&rounds, level.gate());
    check::report("all", &v);
    if v.dropped {
        println!("{name} loses to {below} on these workloads");
    }
    Ok(())
}

// This machine's own tuning file: the one BIGBITS_TUNING names (cargo passes the checkout's
// [env] setting, presets/local.toml, to `cargo run` as it does to build.rs).
fn machine_file() -> Option<PathBuf> {
    std::env::var_os("BIGBITS_TUNING").filter(|p| !p.is_empty()).map(PathBuf::from)
}

// The layers a build of this checkout resolves, found as build.rs finds them (from the
// checkout's root): the target's preset, then the machine's own file.
fn layers_in_force() -> Result<Vec<(String, file::File)>, String> {
    let parse = |name: &str, text: &str| file::parse(text).map_err(|e| format!("{name}: {e}"));
    let mut files = Vec::new();
    for name in HOST.preset_paths() {
        if let Ok(text) = std::fs::read_to_string(&name) {
            files.push((name.clone(), parse(&name, &text)?));
            break;
        }
    }
    if let Some(path) = machine_file() {
        let name = path.display().to_string();
        if let Ok(text) = std::fs::read_to_string(&path) {
            let f = parse(&name, &text)?;
            match resolve::layer_target_ok(&f, &HOST) {
                Ok(()) => files.push((name, f)),
                Err(why) => println!("{name}: {why}; builds ignore it"),
            }
        }
    }
    Ok(files)
}

fn status_layers() -> Result<(), String> {
    let files = layers_in_force()?;
    println!("layers for {}, lowest first: the schema's defaults{}", HOST.name(), files.iter().map(|(n, _)| format!(", {n}")).collect::<String>());
    match machine_file() {
        Some(p) if !p.exists() => println!("no file of this machine's own yet ({}): `cargo tune tune` writes it", p.display()),
        None => println!("BIGBITS_TUNING isn't set here (run through `cargo tune`), so no machine file is looked for"),
        _ => {}
    }
    let r = resolve_host(&files)?;
    for sec in tuning::schema() {
        // The top layer that holds the section at the crate's version supplies it.
        let from = files.iter().rev().find_map(|(name, f)| {
            let fs = f.section(sec.name)?;
            matches!(fs.value("version"), Some(file::Raw::Int(v)) if *v == sec.version as i128).then_some((name, fs))
        });
        match from {
            Some((name, fs)) => {
                let text = |key: &str| fs.get(key).map(|e| file::raw_text(&e.value)).unwrap_or_else(|| "?".into());
                println!("[{}] {name}: tuned {} at {}", sec.name, text("tuned"), text("level"));
            }
            None => println!("[{}] defaults", sec.name),
        }
    }
    for w in r.warnings {
        println!("warning: {w}");
    }
    Ok(())
}

fn status(path: &str) -> Result<(), String> {
    let f = read_file(path)?;
    let r = resolve::resolve(tuning::schema(), &[resolve::Layer { name: path, file: &f }], &HOST).map_err(|e| e.join("\n"))?;
    for s in f.sections.iter().filter(|s| s.name != "meta") {
        let tuned = s.get("tuned").map(|e| file::raw_text(&e.value)).unwrap_or_default();
        println!("[{}] tuned {tuned}, {} values", s.name, s.entries.len());
    }
    for w in r.warnings {
        println!("warning: {w}");
    }
    Ok(())
}

// Sets every value the tuning file `f` (read from `path`) resolves to live, so measurements and
// derivations read it.
fn load(path: &str, f: file::File) -> Result<(), String> {
    let r = resolve_host(&[(path.to_string(), f)])?;
    for w in &r.warnings {
        eprintln!("warning: {w}");
    }
    set_values(r)
}

// Measurements run as a caller's code does: on the calling thread, with parallel entries on
// rayon's global pool (a private pool's worker measured the parallel NTT 9% faster than a caller
// sees it). The global pool gets `threads` workers (all CPUs when None), on performance cores,
// with stacks large enough for static entries' [u64; N] arrays. Only the first call configures it.
fn setup_pool(threads: Option<usize>) -> Result<(), String> {
    static DONE: std::sync::Once = std::sync::Once::new();
    let mut result = Ok(());
    DONE.call_once(|| {
        let mut b = rayon::ThreadPoolBuilder::new().stack_size(64 << 20).start_handler(|_| measure::prefer_fast_cores());
        if let Some(t) = threads {
            b = b.num_threads(t);
        }
        result = b.build_global().map_err(|e| e.to_string());
    });
    measure::prefer_fast_cores();
    result
}

// mul's table and the cutoffs measured under it: what `derive --compare` switches between the
// build's values and FILE's.
const COMPARED: [&str; 6] =
    ["mul.kara", "mul.karatsuba_sqr", "mul.fft_sqr", "mul.static_ntt_sqr", "mul.ntt_karatsuba", "mul.ntt_chunking_karatsuba"];

// The fits' check of the machine: how far the two passes of every fit probe since the last call
// were apart. It takes in whatever doesn't repeat from one pass to the next.
fn fit_line() {
    let (probes, noise) = measure::fit_noise();
    if probes > 0 {
        eprintln!(
            "fit probes: {probes} measured twice, the passes differ {:.1}% rms{}",
            100.0 * noise,
            if noise > fit::ACCEPT_RMS { " -- more than the 5% a fit is judged at: the machine was busy or changing speed, so don't trust these fits" } else { "" },
        );
    }
}

// How many of the sweeps' measurements since the last call never settled: the sign the tuner
// gives that the machine was busy or changing speed while it swept.
fn samples_line(level: Level) {
    let (probes, unsettled) = measure::settle_counts();
    eprintln!(
        "samples: {unsettled} of {probes} sweep measurements unsettled (above a {:.1}% spread after {} rounds or {:.2} s){}",
        100.0 * level.spread(),
        level.max_rounds(),
        level.probe_budget_ns() / 1e9,
        if unsettled * 20 > probes { " -- more than 5%: the machine was busy or changing speed, so don't trust these numbers; measure again when it is quiet and on full power" } else { "" },
    );
}

// The level FILE's mul section was tuned at (normal when it doesn't say). The table's
// derivation leaves Karatsuba inside a row only where schoolbook wins by the level's tie band,
// so rederiving a file reproduces the table its tune wrote only at that tune's level.
pub(crate) fn file_level(path: &str) -> Level {
    read_file(path)
        .ok()
        .and_then(|f| match f.section("mul").and_then(|s| s.value("level")) {
            Some(file::Raw::Str(l)) => Level::parse(l),
            _ => None,
        })
        .unwrap_or(Level::Normal)
}

// The sections with derived values, in the order `derive` writes them.
const DERIVED: [&str; 3] = ["mul", "div", "series"];

fn derive(file: Option<String>, named: &[String], check: bool, compare: bool) -> Result<(), String> {
    if let Some(other) = named.iter().find(|n| !DERIVED.contains(&n.as_str())) {
        return Err(format!("derive: section '{other}' has no derived values (those that do: {})", DERIVED.join(", ")));
    }
    // Every section's, or only the named ones': a table whose layout changed is rederived
    // without rewriting another section's.
    let wanted = |s: &str| named.is_empty() || named.iter().any(|n| n == s);
    let eps = file.as_deref().map_or(Level::Normal, file_level).eps();
    // The build's table (the schema default, or a preset's), before FILE's replaces it.
    let build: Vec<(&str, Value)> = COMPARED.iter().map(|p| (*p, tuning::get(p).expect("declared"))).collect();
    if let Some(f) = &file {
        load(f, read_file(f)?)?;
    }
    setup_pool(None)?;
    if wanted("mul") {
        let r = derive::kara_report(160, 640);
        let (table, evidence) = derive::kara_table(eps);
        println!("{evidence} ({} change points)", table.len() / 3);
        println!(
            "  against the live table: {} of {} shapes differ (squares from {} vs {}, l = 4s from {} vs {})",
            r.disagree, r.shapes, r.square.0, r.square.1, r.chunked.0, r.chunked.1
        );
    }
    if let Some(f) = &file {
        // Rewrites FILE's derived values; times nothing.
        derive_into(std::path::Path::new(f), eps, wanted("mul"), wanted("div"))?;
        if wanted("series") {
            derive_small(std::path::Path::new(f), eps)?;
        }
    }
    let derived: Vec<(&str, Value)> = build.iter().map(|(p, _)| (*p, tuning::get(p).unwrap())).collect();
    if compare {
        use std::hint::black_box;
        let report = |what: &str, names: Vec<String>, rows: Vec<(f64, f64)>| {
            println!("compare {what}: under the build's values and the derived ones (ns)");
            let mut logs = Vec::new();
            for (name, (b, d)) in names.iter().zip(rows) {
                logs.push((d / b).ln());
                if (d / b - 1.0).abs() > 0.01 {
                    println!("  {name:>10} {b:>10.0} {d:>10.0}  {:.3}", d / b);
                }
            }
            let worst = logs.iter().cloned().fold(f64::MIN, f64::max);
            println!(
                "  derived / build: geometric mean {:.4} over {} shapes, worst {:.3} (shapes within 1% not listed)",
                (logs.iter().sum::<f64>() / logs.len() as f64).exp(), logs.len(), worst.exp()
            );
        };
        let sizes: Vec<usize> = sweep::geo(30.0, 5.0).take(34).collect();
        let shapes: Vec<(usize, usize)> = sizes.iter().flat_map(|&s| [(s, s), (s + s / 2, s), (3 * s, s)]).collect();
        let rows = derive::compare(Level::Normal, &build, &derived, &shapes, |(l, s), pass| {
            let (a, b) = (measure::pad(l, 61, pass), measure::pad(s, 67, pass));
            let mut o = measure::pad(l + s - 1, 71, pass);
            Box::new(move || {
                black_box(crate::utils::mul::mul_dyn(black_box(&a), black_box(&b), &mut o));
            })
        });
        report("mul_dyn", shapes.iter().map(|(l, s)| format!("{l}x{s}")).collect(), rows);
        let rows = derive::compare(Level::Normal, &build, &derived, &sizes, |n, pass| {
            let a = measure::pad(n, 73, pass);
            let mut o = measure::pad(2 * n - 1, 79, pass);
            Box::new(move || {
                black_box(crate::utils::mul::sqr_dyn(black_box(&a), &mut o));
            })
        });
        report("sqr_dyn", sizes.iter().map(|n| n.to_string()).collect(), rows);
        // The static entries at N = 8192, on the default pool.
        let st: Vec<usize> = sweep::geo(256.0, 5.0).take(18).collect();
        let st_shapes: Vec<(usize, usize)> = st.iter().flat_map(|&s| [(s, s), (2 * s, s)]).filter(|&(l, s)| l + s <= 6000).collect();
        let rows = derive::compare(Level::Normal, &build, &derived, &st_shapes, |(l, s), pass| {
            let (a, b) = (measure::pad(l, 83, pass), measure::pad(s, 89, pass));
            let mut o = measure::pad(l + s - 1, 97, pass);
            Box::new(move || {
                black_box(crate::utils::mul::mul_static::<8192>(black_box(&a), black_box(&b), &mut o));
            })
        });
        report("mul_static::<8192>", st_shapes.iter().map(|(l, s)| format!("{l}x{s}")).collect(), rows);
        let st_sq: Vec<usize> = st.iter().cloned().filter(|&n| n <= 4000).collect();
        let rows = derive::compare(Level::Normal, &build, &derived, &st_sq, |n, pass| {
            let a = measure::pad(n, 101, pass);
            let mut o = measure::pad(2 * n - 1, 103, pass);
            Box::new(move || {
                black_box(crate::utils::mul::sqr_static::<8192>(black_box(&a), &mut o));
            })
        });
        report("sqr_static::<8192>", st_sq.iter().map(|n| n.to_string()).collect(), rows);
    }
    if check {
        let shapes: Vec<(usize, usize)> =
            [16, 24, 32, 40, 48, 56, 64, 80, 96, 128].iter().flat_map(|&s| [(s, s), (s + s / 3, s)]).collect();
        println!("check (ns, measured / predicted):");
        println!("  {:>9} {:>17} {:>17} {:>17} {:>17} {:>13}", "l x s", "mul_buf", "level", "level's products", "level overhead", "level/school");
        for r in derive::kara_check(Level::Normal, &shapes) {
            println!(
                "  {:>9} {:>8.0} /{:>7.0} {:>8.0} /{:>7.0} {:>8.0} /{:>7.0} {:>8.0} /{:>7.0} {:>6.3} /{:>5.3}",
                format!("{}x{}", r.l, r.s),
                r.school.0, r.school.1, r.level.0, r.level.1, r.products.0, r.products.1,
                r.level.0 - r.products.0, r.level.1 - r.products.1,
                r.level.0 / r.school.0, r.level.1 / r.school.1
            );
        }
    }
    Ok(())
}
