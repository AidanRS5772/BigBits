//! Writing tuning files: each tuned section with its versions, level, date and upstream hash,
//! and each value with its evidence.

use crate::tuning::file::{self, File, FileSection, Raw};
use crate::tuning::resolve::{self, Value};
use crate::tuning::schema_types::Section;
use crate::tuning::tuner::fit::Fit;
use std::path::Path;

/// A tuned value and the evidence for it.
#[derive(Clone, Debug)]
pub struct Tuned {
    pub key: &'static str,
    pub value: Value,
    pub evidence: String,
}

impl Tuned {
    /// Makes the value live, as `section`'s knob.
    pub fn set_live(&self, section: &str) -> Result<(), String> {
        crate::tuning::set(&format!("{section}.{}", self.key), self.value.clone()).map_err(|e| e.to_string())
    }
}

fn today() -> String {
    // Days since the epoch to a civil date, without a date crate.
    let days = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64 / 86400;
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02}")
}

/// Records a fit's knobs: its constants under `keys`, with `evidence` and any term left out for
/// solving negative (written as 0).
pub fn push_fit(out: &mut Vec<Tuned>, keys: &[&'static str], fit: &Fit, evidence: String) {
    let evidence = evidence + &fit.dropped_text(keys);
    for (key, v) in keys.iter().zip(&fit.beta) {
        out.push(Tuned { key, value: Value::Float(*v), evidence: evidence.clone() });
    }
}

// The tuning file at `path` and its text (an empty file and no text when there is none yet).
fn open(path: &Path) -> std::io::Result<(File, Option<String>)> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let f = file::parse(&text).map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
            Ok((f, Some(text)))
        }
        Err(_) => Ok((File::default(), None)),
    }
}

// Writes `f` to `path`, unless that is the text `read` from it: build.rs watches these files,
// and a write that changes nothing would rebuild the crate.
fn save(path: &Path, f: &File, read: Option<String>) -> std::io::Result<()> {
    let text = file::write(f);
    if read.as_deref() == Some(text.as_str()) {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, text)
}

fn set_tuned(fs: &mut FileSection, tuned: &[Tuned]) {
    for t in tuned {
        fs.set(t.key, t.value.to_raw(), Some(t.evidence.clone()));
    }
}

/// A file section for `section`'s tuned values, with the upstream hash of `values` (every knob's
/// resolved value, in schema order).
pub fn section(section: &'static Section, tuned: &[Tuned], level: &str, values: &[Value]) -> FileSection {
    let flat = resolve::flat(crate::tuning::schema());
    let mut fs = FileSection { name: section.name.to_string(), entries: Vec::new() };
    fs.set("version", Raw::Int(section.version as i128), None);
    fs.set("derive", Raw::Int(section.derive as i128), None);
    fs.set("tuned", Raw::Str(today()), None);
    fs.set("level", Raw::Str(level.to_string()), None);
    fs.set("upstream", Raw::Str(resolve::upstream_text(&flat, values, section)), None);
    set_tuned(&mut fs, tuned);
    fs
}

/// Merges `section` into the tuning file at `path` (created if missing), replacing the section
/// whole, and records the machine in [meta].
pub fn merge_into(path: &Path, section: FileSection, threads: usize) -> std::io::Result<()> {
    let (mut f, read) = open(path)?;
    if f.header.is_empty() {
        f.header = vec![
            "BigBits tuning file. Written by `cargo tune`. Edit with care: values outside a knob's".into(),
            "valid range fail the build, and a section whose version differs from the crate's is ignored.".into(),
        ];
    }
    let meta = f.section_mut("meta");
    meta.set("format", Raw::Int(1), None);
    meta.set("crate", Raw::Str(env!("CARGO_PKG_VERSION").into()), None);
    meta.set("target", Raw::Str(super::HOST.name()), None);
    meta.set("threads", Raw::Int(threads as i128), Some("the most threads checks went up to".into()));
    f.merge(&File { sections: vec![section], ..File::default() });
    // Keep [meta] first.
    f.sections.sort_by_key(|s| s.name != "meta");
    save(path, &f, read)
}

/// Sets more values in a section of the file at `path` (a table, a second phase), recording the
/// section's `derive` number, its tables' layout.
pub fn set_in_section(path: &Path, section: &'static Section, tuned: &[Tuned]) -> std::io::Result<()> {
    let (mut f, read) = open(path)?;
    let fs = f.section_mut(section.name);
    if fs.get("version").is_none() {
        fs.set("version", Raw::Int(section.version as i128), None);
    }
    fs.set("derive", Raw::Int(section.derive as i128), None);
    set_tuned(fs, tuned);
    save(path, &f, read)
}
