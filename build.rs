// Generates the knob accessors from the schema (docs/tuning_system.md, "build.rs"). It checks the
// schema, layers the values (the schema's defaults, then the target's preset, then the machine's
// own file, which `BIGBITS_TUNING` names), and writes $OUT_DIR/tuning.rs, which
// `src/tuning/mod.rs` includes. The shared files are the crate's own `src/tuning` modules, so
// there is one source of truth and no build dependency.
#![allow(dead_code)]

#[path = "src/tuning/file.rs"]
mod file;
#[path = "src/tuning/hash.rs"]
mod hash;
#[path = "src/tuning/resolve.rs"]
mod resolve;
#[path = "src/tuning/schema.rs"]
mod schema;
#[path = "src/tuning/schema_types.rs"]
mod schema_types;
#[path = "src/tuning/table.rs"]
mod table;

#[path = "build/codegen.rs"]
mod codegen;

use std::{env, fs, path::PathBuf};

fn main() {
    for p in [
        "build.rs",
        "build/codegen.rs",
        "src/tuning/file.rs",
        "src/tuning/hash.rs",
        "src/tuning/resolve.rs",
        "src/tuning/schema.rs",
        "src/tuning/schema_types.rs",
        "src/tuning/table.rs",
    ] {
        println!("cargo:rerun-if-changed={p}");
    }

    if let Err(errs) = resolve::check_schema(schema::SCHEMA) {
        panic!("src/tuning/schema.rs has {} error(s):\n  {}", errs.len(), errs.join("\n  "));
    }
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target = resolve::Target { arch: &arch, os: &os };
    // The target's preset, if one exists (presets/<arch>-<os>.toml, else <arch>.toml).
    println!("cargo:rerun-if-changed=presets");
    let mut files = Vec::new();
    for name in target.preset_paths() {
        if let Ok(text) = fs::read_to_string(&name) {
            let file = file::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
            files.push((name, file));
            break;
        }
    }
    // Then the machine's own tuning file, the one BIGBITS_TUNING names. This checkout's
    // .cargo/config.toml points it at presets/local.toml, which `cargo tune tune` writes and git
    // ignores, so a machine that has tuned itself builds with its own values and any other with
    // the preset. A file tuned on another target (a cross build would find the host's) is skipped.
    println!("cargo:rerun-if-env-changed=BIGBITS_TUNING");
    if let Some(path) = env::var_os("BIGBITS_TUNING").filter(|p| !p.is_empty()).map(PathBuf::from) {
        let name = path.display().to_string();
        match fs::read_to_string(&path) {
            Ok(text) => {
                println!("cargo:rerun-if-changed={name}");
                let file = file::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
                match resolve::layer_target_ok(&file, &target) {
                    Ok(()) => files.push((name, file)),
                    Err(why) => println!("cargo:warning={name}: {why}; the file is ignored"),
                }
            }
            Err(_) => {
                // No file yet. Inside presets/ that is a machine that hasn't tuned, and the
                // directory is watched, so writing the file reruns this script. Anywhere else
                // cargo could only watch for a missing file by rerunning this script on every
                // build, so the mistake is reported instead.
                let in_presets = path.parent().and_then(|d| fs::canonicalize(d).ok()) == fs::canonicalize("presets").ok();
                if !in_presets {
                    println!("cargo:warning=BIGBITS_TUNING names {name}, which doesn't exist; the preset or the defaults apply");
                }
            }
        }
    }
    let layers: Vec<resolve::Layer> =
        files.iter().map(|(name, file)| resolve::Layer { name, file }).collect();
    let resolved = resolve::resolve(schema::SCHEMA, &layers, &target).unwrap_or_else(|errs| {
        panic!("tuning values don't resolve:\n  {}", errs.join("\n  "));
    });
    for w in &resolved.warnings {
        println!("cargo:warning={w}");
    }
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("tuning.rs");
    fs::write(&out, codegen::generate(schema::SCHEMA, &resolved)).expect("write tuning.rs");
}
