//! Performance knobs: the cutoffs, runtime-equation constants and parallel gates that choose
//! among correct algorithms (docs/tuning_system.md). Every knob is declared once in `schema.rs`;
//! build.rs generates an accessor `k::<section>::<knob>()` and a constant
//! `resolved::<section>::<KNOB>` for each. In a normal build an accessor is a `const fn`
//! returning the resolved value. With the `tune` feature it reads a live cell that `set`/`apply`
//! change between calls (never while a library call runs); code that needs a knob in a const
//! context reads `resolved::`, and only tests may: `src/utils` always reads accessors.
//! Doc-hidden and unstable until Phase 3.

pub mod file;
pub mod hash;
pub mod resolve;
pub mod schema;
pub mod schema_types;
pub mod table;
#[cfg(feature = "tune")]
pub mod tuner;

include!(concat!(env!("OUT_DIR"), "/tuning.rs"));

pub use resolve::Value;
use schema_types::{Knob, Section};

/// The schema every knob is declared in.
pub fn schema() -> &'static [Section] {
    schema::SCHEMA
}

/// `path`'s index in schema order (the order of `PATHS` and of the live cells).
pub fn index(path: &str) -> Option<usize> {
    PATHS.iter().position(|p| *p == path)
}

/// The declaration of the knob at `path` (`section.name`).
pub fn knob(path: &str) -> Option<&'static Knob> {
    let (s, k) = path.split_once('.')?;
    schema::SCHEMA.iter().find(|x| x.name == s)?.knobs.iter().find(|x| x.name == k)
}

/// A knob's current value: the resolved value in a normal build, the live one in a tune build.
pub fn get(path: &str) -> Option<Value> {
    let (i, k) = (index(path)?, knob(path)?);
    if TABLE_OF[i] != usize::MAX {
        return Some(Value::Table(table_now(TABLE_OF[i]).to_vec()));
    }
    Some(Value::from_bits(k.ty, cell_bits(i)))
}

#[cfg(not(feature = "tune"))]
fn table_now(j: usize) -> &'static [u32] {
    TABLE_INIT[j].t
}

#[cfg(feature = "tune")]
fn table_now(j: usize) -> &'static [u32] {
    table_cell(j).t
}

/// A tune build's live table `j` (the generated table accessors read it).
#[cfg(feature = "tune")]
#[inline(always)]
pub fn table_cell(j: usize) -> &'static table::Table2 {
    // SAFETY: every pointer stored in TABLE_CELLS points at a `Table2` that is never freed
    // (TABLE_INIT's entries, or a leaked one from `prepare`).
    unsafe { &*TABLE_CELLS[j].load(std::sync::atomic::Ordering::Acquire) }
}

#[cfg(not(feature = "tune"))]
fn cell_bits(i: usize) -> u64 {
    RESOLVED_BITS[i]
}

#[cfg(feature = "tune")]
fn cell_bits(i: usize) -> u64 {
    CELLS[i].load(std::sync::atomic::Ordering::Relaxed)
}

/// Every knob's current value as `path=value` entries joined by `;`, the form `apply` reads.
pub fn spec() -> String {
    PATHS
        .iter()
        .map(|p| format!("{p}={}", get(p).expect("PATHS are knobs").text()))
        .collect::<Vec<_>>()
        .join(";")
}

#[derive(Clone, Debug, PartialEq)]
pub enum KnobError {
    Unknown(String),
    Invalid { path: String, msg: String },
}

impl std::fmt::Display for KnobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KnobError::Unknown(p) => write!(f, "no knob `{p}`"),
            KnobError::Invalid { path, msg } => write!(f, "`{path}`: {msg}"),
        }
    }
}

impl std::error::Error for KnobError {}

/// Parses `path=value` entries joined by `;` (whitespace around them ignored), each value written
/// as in a tuning file, into checked values.
pub fn parse_spec(spec: &str) -> Result<Vec<(usize, Value)>, KnobError> {
    let mut out = Vec::new();
    for item in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let (path, text) = item
            .split_once('=')
            .ok_or_else(|| KnobError::Invalid { path: item.into(), msg: "expected `path=value`".into() })?;
        let path = path.trim();
        let (i, k) = index(path).zip(knob(path)).ok_or_else(|| KnobError::Unknown(path.into()))?;
        let invalid = |msg: String| KnobError::Invalid { path: path.into(), msg };
        let parsed = file::parse(&format!("[s]\nk = {}\n", text.trim())).map_err(|e| invalid(e.msg))?;
        let raw = &parsed.sections[0].entries[0].value;
        let v = resolve::parse_raw(k, raw).map_err(invalid)?;
        resolve::check_value(k, &v).map_err(invalid)?;
        out.push((i, v));
    }
    Ok(out)
}

/// Sets the live value of the knob at `path`, after checking its type and valid range. A table
/// is leaked: readers may still hold the old slice.
#[cfg(feature = "tune")]
pub fn set(path: &str, v: Value) -> Result<(), KnobError> {
    prepare(path, &v).map(install)
}

/// Sets every knob a `parse_spec` string names, or none of them if any entry is invalid.
#[cfg(feature = "tune")]
pub fn apply(spec: &str) -> Result<(), KnobError> {
    for (i, v) in parse_spec(spec)? {
        set(PATHS[i], v)?;
    }
    Ok(())
}

/// Every live value, to put back with `restore`: scalar cells, and table pointers (as addresses).
#[cfg(feature = "tune")]
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot(Vec<u64>, Vec<usize>);

#[cfg(feature = "tune")]
pub fn snapshot() -> Snapshot {
    let tables = TABLE_CELLS.iter().map(|c| c.load(std::sync::atomic::Ordering::Acquire) as usize).collect();
    Snapshot((0..NKNOBS).map(cell_bits).collect(), tables)
}

#[cfg(feature = "tune")]
pub fn restore(s: &Snapshot) {
    for (c, &b) in CELLS.iter().zip(&s.0) {
        c.store(b, std::sync::atomic::Ordering::Relaxed);
    }
    for (c, &p) in TABLE_CELLS.iter().zip(&s.1) {
        c.store(p as *mut table::Table2, std::sync::atomic::Ordering::Release);
    }
}

/// Puts every live value back to its resolved value.
#[cfg(feature = "tune")]
pub fn reset() {
    let tables = (0..NTABLES).map(|j| &TABLE_INIT[j] as *const table::Table2 as usize).collect();
    restore(&Snapshot(RESOLVED_BITS.to_vec(), tables));
}

/// The tuner's entry point (`cargo tune`): docs/tuning_system.md, "The command line".
#[cfg(feature = "tune")]
pub fn cli() {
    tuner::cli()
}

/// A knob value ready to install with one store. `prepare` looks its path up (and leaks a copy
/// of a table) on every call, so code that switches between values many times prepares each once.
#[cfg(feature = "tune")]
#[derive(Clone, Copy)]
pub enum Handle {
    Scalar(usize, u64),
    Table(usize, &'static table::Table2),
}

/// Checks `v` as the value of the knob at `path` and prepares it for `install`.
#[cfg(feature = "tune")]
pub fn prepare(path: &str, v: &Value) -> Result<Handle, KnobError> {
    let (i, k) = index(path).zip(knob(path)).ok_or_else(|| KnobError::Unknown(path.into()))?;
    resolve::check_value(k, v).map_err(|msg| KnobError::Invalid { path: path.into(), msg })?;
    Ok(match v {
        Value::Table(t) => {
            let two = matches!(k.ty, schema_types::Ty::Table { inputs: 2, .. });
            let rows = if two { table::row_index(t) } else { Vec::new() };
            Handle::Table(
                TABLE_OF[i],
                Box::leak(Box::new(table::Table2 {
                    t: Box::leak(t.clone().into_boxed_slice()),
                    rows: Box::leak(rows.into_boxed_slice()),
                })),
            )
        }
        v => {
            let bits = v
                .bits()
                .ok_or_else(|| KnobError::Invalid { path: path.into(), msg: "step functions arrive later".into() })?;
            Handle::Scalar(i, bits)
        }
    })
}

/// Makes a prepared value the live one: one store.
#[cfg(feature = "tune")]
#[inline]
pub fn install(h: Handle) {
    match h {
        Handle::Scalar(i, bits) => CELLS[i].store(bits, std::sync::atomic::Ordering::Relaxed),
        Handle::Table(j, t) => TABLE_CELLS[j].store(t as *const _ as *mut _, std::sync::atomic::Ordering::Release),
    }
}
