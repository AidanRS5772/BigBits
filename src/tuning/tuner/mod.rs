//! The tuner (docs/tuning_system.md), in tune builds only: `cargo tune` runs `cli()` through
//! `examples/tune.rs`. It measures each section's knobs, sets them live, and merges them into a
//! tuning file with their evidence.

pub mod check;
mod check_items;
pub mod design;
pub mod dispatch;
pub mod fit;
pub mod measure;
pub mod out;
pub mod sections;
pub mod sweep;
pub mod tables;
pub mod synth;

use crate::tuning::{self, file, resolve, schema_types::Kind, Value};
use measure::Level;
use std::path::PathBuf;

/// The machine the tuner runs on: tuning files are resolved for it and record it.
pub(crate) const HOST: resolve::Target<'static> = resolve::Target { arch: std::env::consts::ARCH, os: std::env::consts::OS };

const USAGE: &str = "usage: cargo tune <command>
  tune [SECTION ...] [--level quick|normal|thorough] [--out FILE] [--write-preset] [--threads N] [--no-gate] [--analyze] [--tables]
      measure the sections' knobs (all registered sections when none are named), and their
      dispatchers' tables after them (mul.kara, div.dyn_small and div.dyn_rem_small,
      series.small: each index of a table timed, and a row ended where reading the dispatcher's
      equations is within 5% of the call), and merge them into FILE. A section's equations are
      fitted again after its tables, from the same timings, on the shapes they are read for
      (not those the tables decide). --tables measures only the
      tables of the sections named (after a change to a table's layout). --share X on tune,
      dispatch or series puts another number in the rule's place for the run (0.02 for 2%):
      what the tables and the equations would be under it. The default is this machine's own file, the one BIGBITS_TUNING names
      (presets/local.toml in a checkout), which the next build layers over the target's preset;
      --write-preset writes the target's preset itself (the file a build resolves:
      presets/<arch>-<os>.toml, else presets/<arch>.toml; the first when neither exists). Each section ends with the end-to-end
      check as a gate: if its values make its workloads slower on average than the values
      before it, by more than 3% (quick), 2% (normal) or 1% (thorough), the section is dropped
      and those values stay (--no-gate: keep it anyway). --analyze is for whoever writes an
      equation, not for tuning a machine: each fit also prints which of its form's terms the
      timings need (each constant against its standard error, and every form short of some terms)
  check [--file FILE] [SECTION ...] [--level L]
      the end-to-end check on its own: public calls timed under FILE's values (default: this
      machine's own file) and under the layers below it, the preset and the defaults
  dispatch [--level L]
      the rule every dispatcher is held to, checked: for mul_dyn, div_dyn, div_rem_dyn and the
      series' entry, on indices where each reads its equations under the values in force, what
      that decision costs against the call (at most 5%: an index past it belongs in a table)
  equations [--level L]
                       Karatsuba against the FFT for dynamic products: the measured time lost by
                       comparing the fitted equations and by mul_dyn itself, on shapes the
                       equations weren't fitted on
  division [--level L] [--out FILE] [--refit]
                       Knuth, BZ and NR forced over quotient and divisor widths no tune fits the
                       equations to, for div_dyn and div_rem_dyn: timings to FILE, and on the half
                       held out the measured time lost by the fitted equations' picks (--refit:
                       fitted now, to the other half) and by the entries themselves
  series [--level L] [--set 'PATH=VALUE;...'] [--timings FILE]
                       the series equations, rect's guard and the table on synthetic series no tune takes:
                       where the equations are read, the measured time lost by the picks under
                       the live prices and under refitted ones, and by the entry itself; where
                       the table decides, the time its picks lose (--set: with those knob values)
  list [SECTION]       every knob: kind, valid range, live value
  status [--file FILE] the layers a build of this checkout resolves (the target's preset, then
                       this machine's file) and which supplies each section; with --file, that
                       one file against this crate's schema";

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

// The flags that take no value; every other `--flag` is followed by one.
const SWITCHES: [&str; 5] = ["--write-preset", "--refit", "--no-gate", "--analyze", "--tables"];

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
    // --share X: the rule's number for this run, in place of 5%.
    if let Some(v) = flag(rest, "--share") {
        let share: f64 = v.parse().ok().filter(|s: &f64| *s > 0.0 && *s < 1.0).ok_or_else(|| format!("bad --share `{v}` (a fraction of the call, as 0.05)"))?;
        tables::set_share(share);
    }
    // A command that times something: its level, on the pool.
    let ready = || -> Result<Level, String> {
        let level = level_flag(rest)?;
        setup_pool(None)?;
        Ok(level)
    };
    match cmd.as_str() {
        "tune" => tune(rest),
        "list" => {
            list(positional(rest).first().map(String::as_str));
            Ok(())
        }
        "derive" => Err("nothing is derived any more: the tables are measured (`cargo tune tune SECTION --tables` measures a section's alone)".into()),
        "dispatch" => {
            let level = ready()?;
            measure::settle_counts();
            let ok = dispatch::report(level);
            fit_line();
            if ok {
                Ok(())
            } else {
                Err("a dispatcher's decision is past the rule's share of its call: measure its table again (`cargo tune tune SECTION --tables`)".into())
            }
        }
        "equations" => {
            let level = ready()?;
            measure::settle_counts();
            sections::mul::equations(level);
            samples_line(level);
            Ok(())
        }
        "division" => {
            let level = ready()?;
            let out = flag(rest, "--out").unwrap_or_else(|| "target/tuning/division.tsv".into());
            measure::settle_counts();
            sections::div::division(level, std::path::Path::new(&out), switch(rest, "--refit")).map_err(|e| e.to_string())?;
            fit_line();
            Ok(())
        }
        "series" => {
            let level = ready()?;
            // --set 'path=value;...': knob values for this run.
            if let Some(set) = flag(rest, "--set") {
                tuning::apply(&set).map_err(|e| e.to_string())?;
            }
            measure::settle_counts();
            let cache = flag(rest, "--timings").map(PathBuf::from);
            sections::series::report(level, cache.as_deref());
            fit_line();
            Ok(())
        }
        "check" => check_file(flag(rest, "--file"), &positional(rest), ready()?),
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
        if !sections::sections().iter().any(|s| s.name == n) {
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
        // The preset a build of this target resolves; `<arch>-<os>.toml` for a target's first.
        let paths = HOST.preset_paths();
        PathBuf::from(paths.iter().find(|p| std::path::Path::new(p).exists()).unwrap_or(&paths[0]))
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
    // The file merged into is the layer below: sections not measured now keep its values. A file
    // of another machine's target isn't this machine's to merge into.
    if out.exists() {
        let name = out.to_string_lossy();
        let f = read_file(&name)?;
        resolve::layer_target_ok(&f, &HOST).map_err(|why| format!("{name}: {why}; delete it or name another file with --out"))?;
        for w in set_live(&[(name.to_string(), f)])? {
            eprintln!("warning: {w}");
        }
    }
    setup_pool(Some(threads))?;
    fit::set_analyze(switch(args, "--analyze"));
    measure::settle_counts();
    measure::fit_noise();
    sweep::stability_counts();
    sweep::edge_count();
    // Each rejected fit's evidence, and each noisy one's (one fit sets several knobs with the same).
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
    // The gate: each section's values against those before it, on its workloads and those built on it.
    let gate = !switch(args, "--no-gate");
    let tables_only = switch(args, "--tables");
    let mut dropped: Vec<(String, String)> = Vec::new();
    let mut kept = 0;
    for tuner in sections::sections() {
        let name = tuner.name;
        if !named.is_empty() && !named.iter().any(|n| n == name) {
            continue;
        }
        // `--tables`: only the sections that have one.
        if tables_only && tuner.tables.is_empty() {
            continue;
        }
        eprintln!("{} {name} ({})...", if tables_only { "measuring the tables of" } else { "tuning" }, level.name());
        let (before, file_before) = (tuning::snapshot(), std::fs::read_to_string(&out).ok());
        let section = tuning::schema().iter().find(|s| s.name == name).expect("the tuner's sections are declared");
        if !tables_only {
            let tuned = (tuner.tune)(level);
            set_tuned(name, &tuned)?;
            note(name, &tuned);
            let values: Vec<Value> = tuning::PATHS.iter().map(|p| tuning::get(p).unwrap()).collect();
            let fs = out::section(section, &tuned, level.name(), &values);
            // Written as soon as it finishes: an interrupted tune keeps it.
            out::merge_into(&out, fs, threads).map_err(|e| e.to_string())?;
        }
        // The tables are measured once the section's other values are in force, and are in force for
        // its second phase (its equations fitted again on the shapes the tables leave them), for the
        // gate and for the sections built on it.
        for &(key, table) in tuner.tables {
            let (table, evidence) = table(level);
            let tuned = [out::Tuned { key, value: Value::Table(table), evidence }];
            tuned[0].set_live(name)?;
            eprintln!("  {name}.{key}: {}", tuned[0].evidence);
            out::set_in_section(&out, section, &tuned).map_err(|e| e.to_string())?;
        }
        if let Some(late) = tuner.late.filter(|_| !tables_only) {
            eprintln!("tuning {name}, second phase ({})...", level.name());
            let tuned = late(level);
            set_tuned(name, &tuned)?;
            note(name, &tuned);
            out::set_in_section(&out, section, &tuned).map_err(|e| e.to_string())?;
        }
        let after = tuning::snapshot();
        if gate && after == before {
            // Every sweep kept its value: both sides would time the same values.
            eprintln!("  check {name}: its values are the ones before it, so there is nothing to compare");
        } else if gate {
            let rounds = check::compare(&before, &after, level, &|it| check::in_gate(tuning::schema(), name, it.section));
            let v = check::verdict(&rounds, level.gate());
            check::report(name, &v);
            if v.dropped {
                // The values and the file go back.
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
        kept += 1;
    }
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
            dropped.len() + kept,
            100.0 * level.gate()
        );
        for (name, why) in &dropped {
            eprintln!("  {name}: {why}");
        }
    }
    // The rule the tables were measured by, checked under the values now in force.
    for v in dispatch::check(level) {
        eprintln!("dispatch: {}", dispatch::line(&v));
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

// Sets `section`'s tuned values live and prints each with its evidence.
fn set_tuned(section: &str, tuned: &[out::Tuned]) -> Result<(), String> {
    for t in tuned {
        t.set_live(section)?;
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
    resolve::resolve(tuning::schema(), &layers, &HOST).map_err(|e| e.join("\n"))
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
            Kind::Table => "table (measured)".into(),
        };
        let v = tuning::get(&f.path).map(|v| v.text()).unwrap_or_default();
        let v = if v.len() > 60 { format!("{}…", &v[..60]) } else { v };
        println!("{:<44} {:<18} {v}", f.path, kind);
    }
}

// Sets every value `files` resolve to live; returns the resolver's warnings.
fn set_live(files: &[(String, file::File)]) -> Result<Vec<String>, String> {
    let r = resolve_host(files)?;
    for (p, v) in tuning::PATHS.iter().zip(r.values) {
        tuning::set(p, v).map_err(|e| e.to_string())?;
    }
    Ok(r.warnings)
}

// `cargo tune check`: FILE's values (default: this machine's own file) against the layers below
// it, over every workload or those of the named sections' gates.
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
    // The layers below FILE: the preset, unless FILE is the preset. The machine's file is left out:
    // FILE stands in its place.
    let same = |a: &str, b: &std::path::Path| match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => std::path::Path::new(a) == b,
    };
    let mut layers = layers_in_force()?;
    layers.retain(|(n, _)| !same(n, &path) && !machine.as_ref().is_some_and(|m| same(n, m)));
    let below = if layers.is_empty() { "the schema's defaults".to_string() } else { layers.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ") };
    set_live(&layers)?;
    let before = tuning::snapshot();
    layers.push((name.clone(), f));
    set_live(&layers)?;
    let after = tuning::snapshot();
    println!("check ({}): {name} against {below}; ratios are time under {name} over time under {below}", level.name());
    let rounds = check::compare(&before, &after, level, &|it| named.is_empty() || named.iter().any(|n| check::in_gate(tuning::schema(), n, it.section)));
    for (sec, _) in check::ITEM_SECTIONS {
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

// This machine's own tuning file: the one BIGBITS_TUNING names.
fn machine_file() -> Option<PathBuf> {
    std::env::var_os("BIGBITS_TUNING").filter(|p| !p.is_empty()).map(PathBuf::from)
}

// The layers a build of this checkout resolves, as build.rs finds them: the preset, then the machine's file.
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
    let files = [(path.to_string(), read_file(path)?)];
    let r = resolve_host(&files)?;
    for s in files[0].1.sections.iter().filter(|s| s.name != "meta") {
        let tuned = s.get("tuned").map(|e| file::raw_text(&e.value)).unwrap_or_default();
        println!("[{}] tuned {tuned}, {} values", s.name, s.entries.len());
    }
    for w in r.warnings {
        println!("warning: {w}");
    }
    Ok(())
}

// Measurements run as a caller's code does: on the calling thread, parallel entries on rayon's
// global pool, which gets `threads` workers (all CPUs when None) on performance cores, with
// stacks for static entries' arrays. Only the first call configures it.
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

// The fits' check of the machine: how far apart the two passes of every fit probe since the last call were.
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

// The sweeps' check: how many of their measurements since the last call never settled.
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
