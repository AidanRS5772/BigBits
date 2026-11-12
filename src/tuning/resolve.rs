// Resolving knob values (docs/tuning_system.md, "build.rs"): the schema's defaults, then each layer
// (the target's preset, then the machine's own file, `BIGBITS_TUNING`), then `Follows` defaults. Also the
// schema's own checks and each value's validity (invariant 1). Shared by build.rs, which runs it
// at build time, and the crate, where the tuner's `derive --file` (and Phase 4's runtime loading)
// run the same code. Std only; other shared files through `super::`.

use super::file::{File, Num, Raw};
use super::hash;
use super::schema_types::{Alg, DefaultValue, Knob, Range, Section, Ty};

/// A knob's value. Usize and U64 knobs hold `Int`.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(u64),
    Float(f64),
    Bool(bool),
    Table(Vec<u32>),
    Steps(Vec<(u32, f64)>),
}

impl Value {
    /// A scalar as the bits a live cell stores: integers as themselves, floats by `to_bits`,
    /// bools as 0 or 1.
    pub fn bits(&self) -> Option<u64> {
        match *self {
            Value::Int(v) => Some(v),
            Value::Float(v) => Some(v.to_bits()),
            Value::Bool(v) => Some(v as u64),
            _ => None,
        }
    }

    /// The scalar of type `ty` that `bits` holds.
    pub fn from_bits(ty: Ty, bits: u64) -> Value {
        match ty {
            Ty::Usize | Ty::U64 => Value::Int(bits),
            Ty::F64 => Value::Float(f64::from_bits(bits)),
            Ty::Bool => Value::Bool(bits != 0),
            Ty::Table { .. } => panic!("a table has no scalar bits"),
        }
    }

    pub fn to_raw(&self) -> Raw {
        match self {
            Value::Int(v) => Raw::Int(*v as i128),
            Value::Float(v) => Raw::Float(*v),
            Value::Bool(v) => Raw::Bool(*v),
            Value::Table(t) => Raw::Array(t.iter().map(|&x| Num::Int(x as i128)).collect()),
            Value::Steps(s) => Raw::Array(
                s.iter().flat_map(|&(t, v)| [Num::Int(t as i128), Num::Float(v)]).collect(),
            ),
        }
    }

    /// The canonical text the upstream hash reads.
    pub fn text(&self) -> String {
        super::file::raw_text(&self.to_raw())
    }
}

/// A knob in schema order, with its path `section.name`. Codegen, the live store and the resolver
/// all index knobs by this order.
#[derive(Clone, Debug)]
pub struct FlatKnob {
    pub path: String,
    pub section: &'static Section,
    pub knob: &'static Knob,
}

pub fn flat(schema: &'static [Section]) -> Vec<FlatKnob> {
    let mut out = Vec::new();
    for s in schema {
        for k in s.knobs {
            out.push(FlatKnob { path: format!("{}.{}", s.name, k.name), section: s, knob: k });
        }
    }
    out
}

/// Keys of a file section that describe the section rather than set a knob.
pub const META_KEYS: [&str; 6] = ["version", "derive", "tuned", "level", "upstream", "pinned"];

/// Architectures a knob's `arch` may name (`CARGO_CFG_TARGET_ARCH` spellings).
pub const KNOWN_ARCHES: [&str; 6] = ["x86_64", "aarch64", "x86", "arm", "riscv64", "wasm32"];

fn ident(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(f) if f.is_ascii_lowercase())
        && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn is_int(ty: Ty) -> bool {
    matches!(ty, Ty::Usize | Ty::U64)
}

fn steps(knob: &Knob) -> bool {
    matches!(knob.default, DefaultValue::Steps(_))
}

fn range_text(r: Range) -> String {
    match r {
        Range::Int { min, max } => format!("{min}..={max}"),
        Range::Float { min, max } => format!("{min:?}..={max:?}"),
        Range::Bool => "a bool".into(),
        Range::Table => "a table".into(),
    }
}

fn in_range(v: f64, r: Range) -> bool {
    match r {
        Range::Int { min, max } => v >= min as f64 && v <= max as f64,
        Range::Float { min, max } => v >= min && v <= max,
        _ => false,
    }
}

/// Checks a table's structure: change points ascend and every pick is eligible at its sizes.
/// A one-input table is `[size, alg, …]`; a two-input one is `[s, l, alg, …]` with `s`
/// non-decreasing and `l` ascending within each row. An algorithm's `min` is in the table's
/// measure: the size for one input, the shorter input `s` for two.
pub fn check_table(algs: &[Alg], inputs: u8, t: &[u32]) -> Result<(), String> {
    let w = match inputs {
        1 => 2,
        2 => 3,
        _ => return Err(format!("a table has 1 or 2 inputs, not {inputs}")),
    };
    if t.is_empty() || t.len() % w != 0 {
        return Err(format!("a {inputs}-input table needs a non-empty multiple of {w} numbers"));
    }
    let mut prev: Option<(u32, u32)> = None;
    for c in t.chunks(w) {
        let (s, l, a) = if w == 2 { (c[0], c[0], c[1]) } else { (c[0], c[1], c[2]) };
        let alg = algs
            .get(a as usize)
            .ok_or_else(|| format!("pick {a} is not one of the {} algorithms", algs.len()))?;
        if s < alg.min {
            return Err(format!("{} is picked at {s}, below its minimum {}", alg.name, alg.min));
        }
        if let Some((ps, pl)) = prev {
            let ok = if w == 2 { s > ps } else { s > ps || (s == ps && l > pl) };
            if !ok {
                return Err(format!("change points must ascend ({ps},{pl} then {s},{l})"));
            }
        }
        prev = Some((s, l));
    }
    Ok(())
}

/// Checks `v` against `knob`'s type and valid range (invariant 1).
pub fn check_value(knob: &Knob, v: &Value) -> Result<(), String> {
    let bad = |what: String| Err(format!("{what} is outside the valid range {}", range_text(knob.valid)));
    match (knob.ty, v) {
        (Ty::Usize | Ty::U64, Value::Int(x)) => {
            if matches!(knob.valid, Range::Int { min, max } if *x < min || *x > max) {
                return bad(x.to_string());
            }
        }
        (Ty::F64, Value::Float(x)) => {
            if x.is_nan() || !in_range(*x, knob.valid) {
                return bad(format!("{x:?}"));
            }
        }
        (Ty::F64 | Ty::Usize | Ty::U64, Value::Steps(s)) if steps(knob) => {
            if s.is_empty() || s[0].0 != 1 {
                return Err("a step function starts at 1 thread".into());
            }
            if s.windows(2).any(|w| w[1].0 <= w[0].0) {
                return Err("a step function's thread counts must ascend".into());
            }
            if let Some(&(_, x)) = s.iter().find(|&&(_, x)| x.is_nan() || !in_range(x, knob.valid)) {
                return bad(format!("{x:?}"));
            }
        }
        (Ty::Bool, Value::Bool(_)) => {}
        (Ty::Table { algs, inputs, .. }, Value::Table(t)) => check_table(algs, inputs, t)?,
        (ty, v) => return Err(format!("{v:?} is not a value of type {ty:?}")),
    }
    Ok(())
}

/// Reads a file value as a value of `knob`'s type. Floats accept integers; integers don't accept
/// floats.
pub fn parse_raw(knob: &Knob, raw: &Raw) -> Result<Value, String> {
    let int = |v: i128| u64::try_from(v).map_err(|_| format!("{v} doesn't fit a u64"));
    let v = match (knob.ty, raw) {
        (Ty::Usize | Ty::U64, Raw::Int(v)) if !steps(knob) => Value::Int(int(*v)?),
        (Ty::F64, Raw::Float(v)) if !steps(knob) => Value::Float(*v),
        (Ty::F64, Raw::Int(v)) if !steps(knob) => Value::Float(*v as f64),
        (Ty::Bool, Raw::Bool(v)) => Value::Bool(*v),
        (Ty::Table { .. }, Raw::Array(a)) => Value::Table(
            a.iter()
                .map(|n| match *n {
                    Num::Int(v) => u32::try_from(v).map_err(|_| format!("{v} doesn't fit a u32")),
                    Num::Float(v) => Err(format!("table entries are integers, not {v:?}")),
                })
                .collect::<Result<_, _>>()?,
        ),
        (_, Raw::Array(a)) if steps(knob) => {
            if a.len() % 2 != 0 {
                return Err("a step function is pairs of (threads, value)".into());
            }
            let mut s = Vec::new();
            for p in a.chunks(2) {
                let t = match p[0] {
                    Num::Int(t) => u32::try_from(t).map_err(|_| format!("bad thread count {t}"))?,
                    Num::Float(t) => return Err(format!("thread counts are integers, not {t:?}")),
                };
                let v = match p[1] {
                    Num::Int(v) => v as f64,
                    Num::Float(v) => v,
                };
                s.push((t, v));
            }
            Value::Steps(s)
        }
        (ty, raw) => return Err(format!("`{}` is not a {ty:?} value", super::file::raw_text(raw))),
    };
    Ok(v)
}

fn default_type_ok(knob: &Knob) -> bool {
    match (knob.ty, knob.default) {
        (_, DefaultValue::Follows { .. }) => is_int(knob.ty) || knob.ty == Ty::F64,
        (Ty::Usize | Ty::U64, DefaultValue::Int(_)) => true,
        (Ty::F64, DefaultValue::Float(_)) => true,
        (Ty::Usize | Ty::U64 | Ty::F64, DefaultValue::Steps(_)) => true,
        (Ty::Bool, DefaultValue::Bool(_)) => true,
        (Ty::Table { .. }, DefaultValue::Table(_)) => true,
        _ => false,
    }
}

fn range_type_ok(ty: Ty, r: Range) -> bool {
    matches!(
        (ty, r),
        (Ty::Usize | Ty::U64, Range::Int { .. })
            | (Ty::F64, Range::Float { .. })
            | (Ty::Bool, Range::Bool)
            | (Ty::Table { .. }, Range::Table)
    )
}

fn subrange(inner: Range, outer: Range) -> bool {
    match (inner, outer) {
        (Range::Int { min: a, max: b }, Range::Int { min: c, max: d }) => a >= c && b <= d,
        (Range::Float { min: a, max: b }, Range::Float { min: c, max: d }) => a >= c && b <= d,
        (a, b) => a == b,
    }
}

/// A non-`Follows` default as a value.
pub fn default_value(knob: &Knob) -> Option<Value> {
    Some(match knob.default {
        DefaultValue::Int(v) => Value::Int(v),
        DefaultValue::Float(v) => Value::Float(v),
        DefaultValue::Bool(v) => Value::Bool(v),
        DefaultValue::Table(t) => Value::Table(t.to_vec()),
        DefaultValue::Steps(s) => Value::Steps(s.to_vec()),
        DefaultValue::Follows { .. } => return None,
    })
}

/// The schema's own checks. Every failure is listed; build.rs turns them into a build error,
/// because the schema is the crate's own data.
pub fn check_schema(schema: &'static [Section]) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();
    let flat = flat(schema);
    for (i, s) in schema.iter().enumerate() {
        if !ident(s.name) {
            errs.push(format!("section name `{}` must be a lowercase identifier", s.name));
        }
        if schema[..i].iter().any(|t| t.name == s.name) {
            errs.push(format!("section `{}` is declared twice", s.name));
        }
        for a in s.after {
            if !schema.iter().any(|t| t.name == *a) {
                errs.push(format!("section `{}` lists unknown upstream section `{a}`", s.name));
            }
        }
    }
    // Upstream sections must not form a cycle.
    fn visit(schema: &[Section], i: usize, state: &mut [u8], errs: &mut Vec<String>) {
        if state[i] == 2 {
            return;
        }
        if state[i] == 1 {
            errs.push(format!("upstream sections form a cycle through `{}`", schema[i].name));
            return;
        }
        state[i] = 1;
        for a in schema[i].after {
            if let Some(j) = schema.iter().position(|t| t.name == *a) {
                visit(schema, j, state, errs);
            }
        }
        state[i] = 2;
    }
    let mut state = vec![0u8; schema.len()];
    for i in 0..schema.len() {
        visit(schema, i, &mut state, &mut errs);
    }
    for (i, f) in flat.iter().enumerate() {
        let k = f.knob;
        let p = &f.path;
        if !ident(k.name) {
            errs.push(format!("knob name `{p}` must be a lowercase identifier"));
        }
        if flat[..i].iter().any(|g| g.path == *p) {
            errs.push(format!("knob `{p}` is declared twice"));
        }
        if !default_type_ok(k) {
            errs.push(format!("`{p}`'s default {:?} doesn't match its type {:?}", k.default, k.ty));
        }
        if !range_type_ok(k.ty, k.valid) || !range_type_ok(k.ty, k.search) {
            errs.push(format!("`{p}`'s ranges don't match its type {:?}", k.ty));
        } else if !subrange(k.search, k.valid) {
            errs.push(format!("`{p}`'s search range must lie inside its valid range"));
        }
        if let Some(v) = default_value(k) {
            if let Err(e) = check_value(k, &v) {
                errs.push(format!("`{p}`'s default: {e}"));
            }
        }
        if let Some(a) = k.arch {
            if !KNOWN_ARCHES.contains(&a) {
                errs.push(format!("`{p}` names unknown architecture `{a}`"));
            }
        }
        if steps(k) && k.threads == super::schema_types::Threads::One {
            errs.push(format!("`{p}` is a step function of the thread count but reads one thread"));
        }
        if let DefaultValue::Follows { from, .. } = k.default {
            match flat.iter().find(|g| g.path == from) {
                None => errs.push(format!("`{p}` follows unknown knob `{from}`")),
                Some(g) if is_int(g.knob.ty) != is_int(k.ty) || steps(g.knob) => {
                    errs.push(format!("`{p}` follows `{from}`, whose type differs"))
                }
                Some(_) => {
                    // Follow the chain; a knob reached twice is a cycle.
                    let mut seen = vec![p.clone()];
                    let mut cur = from;
                    while let Some(g) = flat.iter().find(|g| g.path == cur) {
                        if seen.contains(&g.path) {
                            errs.push(format!("`{p}`'s Follows chain is a cycle"));
                            break;
                        }
                        seen.push(g.path.clone());
                        match g.knob.default {
                            DefaultValue::Follows { from, .. } => cur = from,
                            _ => break,
                        }
                    }
                }
            }
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

/// One layer over the defaults: a preset or a user's tuning file.
pub struct Layer<'a> {
    pub name: &'a str,
    pub file: &'a File,
}

/// The target the values are resolved for. Arch-specific knobs take values only from layers for
/// their own architecture.
pub struct Target<'a> {
    pub arch: &'a str,
    pub os: &'a str,
}

impl Target<'_> {
    /// `<arch>-<os>`: what a tuning file's [meta] target records, and a preset is named after.
    pub fn name(&self) -> String {
        format!("{}-{}", self.arch, self.os)
    }

    /// Where the target's preset is looked for, first match wins.
    pub fn preset_paths(&self) -> [String; 2] {
        [format!("presets/{}.toml", self.name()), format!("presets/{}.toml", self.arch)]
    }
}

/// Whether a machine's own tuning file may be layered for `target`. `cargo tune` records the
/// machine it measured as [meta] target = "<arch>-<os>"; a file that names another target is
/// another machine's (or the host's, in a cross build), and its measurements don't apply. A file
/// that names none is taken as it is.
pub fn layer_target_ok(file: &File, target: &Target) -> Result<(), String> {
    let Some(Raw::Str(tuned_on)) = file.section("meta").and_then(|m| m.value("target")) else {
        return Ok(());
    };
    let build = target.name();
    if *tuned_on == build {
        Ok(())
    } else {
        Err(format!("it was tuned on {tuned_on}, and this build is for {build}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    Default,
    Follows,
    /// Set by the layer at this index.
    Layer(usize),
}

#[derive(Clone, Debug)]
pub struct Resolved {
    /// Every knob's value, in `flat` order.
    pub values: Vec<Value>,
    pub origin: Vec<Origin>,
    pub warnings: Vec<String>,
}

/// The hash of `section`'s resolved measured values (`Kind::measured`, plus its pinned knobs).
pub fn section_hash(
    flat: &[FlatKnob],
    values: &[Value],
    section: &str,
    pinned: &[&str],
) -> u64 {
    let pairs: Vec<(&str, String)> = flat
        .iter()
        .zip(values)
        .filter(|(f, _)| f.section.name == section && (f.knob.kind.measured() || pinned.contains(&f.knob.name)))
        .map(|(f, v)| (f.knob.name, v.text()))
        .collect();
    hash::hash_pairs(&pairs)
}

/// The `upstream` string recorded with a section's derived values: `section:hash` for each of its
/// upstream sections, from the given resolved values.
pub fn upstream_text(flat: &[FlatKnob], values: &[Value], section: &Section) -> String {
    let parts: Vec<String> = section
        .after
        .iter()
        .map(|a| format!("{a}:{:016x}", section_hash(flat, values, a, &[])))
        .collect();
    parts.join(" ")
}

/// Layers `layers` over `schema`'s defaults for `target`, resolves `Follows` defaults, and checks
/// every value. Warnings: unknown sections and keys, a section whose `version` differs (it falls
/// through whole), one whose `derive` differs (its derived values fall through), a stale upstream
/// hash, and a value for another architecture. Errors: a value of the wrong type or outside its
/// valid range, and a tuning file of an unknown format.
pub fn resolve(
    schema: &'static [Section],
    layers: &[Layer],
    target: &Target,
) -> Result<Resolved, Vec<String>> {
    let flat = flat(schema);
    let mut values: Vec<Value> =
        flat.iter().map(|f| default_value(f.knob).unwrap_or(Value::Int(0))).collect();
    let mut origin = vec![Origin::Default; flat.len()];
    let mut warnings = Vec::new();
    let mut errs = Vec::new();
    // (layer, section, recorded upstream text) to check once everything is resolved.
    let mut upstream_checks: Vec<(usize, &'static Section, String)> = Vec::new();

    for (li, layer) in layers.iter().enumerate() {
        let lname = layer.name;
        if let Some(meta) = layer.file.section("meta") {
            match meta.value("format") {
                None | Some(Raw::Int(1)) => {}
                Some(v) => errs.push(format!(
                    "{lname}: [meta] format {} is not one this crate reads (1)",
                    super::file::raw_text(v)
                )),
            }
        }
        for fs in &layer.file.sections {
            if fs.name == "meta" {
                continue;
            }
            let Some(sec) = schema.iter().find(|s| s.name == fs.name) else {
                warnings.push(format!("{lname}: unknown section [{}] ignored", fs.name));
                continue;
            };
            match fs.value("version") {
                Some(Raw::Int(v)) if *v == sec.version as i128 => {}
                other => {
                    let had = other.map_or("none".to_string(), super::file::raw_text);
                    warnings.push(format!(
                        "{lname}: [{}] has version {had}, the crate's is {}; the section falls through",
                        sec.name, sec.version
                    ));
                    continue;
                }
            }
            let derive_ok = matches!(fs.value("derive"), Some(Raw::Int(d)) if *d == sec.derive as i128);
            let pinned: Vec<&str> = match fs.value("pinned") {
                Some(Raw::Str(s)) => s.split_whitespace().collect(),
                _ => Vec::new(),
            };
            let mut stale_derived = false;
            for e in &fs.entries {
                if META_KEYS.contains(&e.key.as_str()) {
                    continue;
                }
                let Some(i) = flat.iter().position(|f| f.section.name == sec.name && f.knob.name == e.key)
                else {
                    warnings.push(format!("{lname}: unknown key `{}` in [{}] ignored", e.key, sec.name));
                    continue;
                };
                let k = flat[i].knob;
                if !k.kind.measured() && !pinned.contains(&k.name) && !derive_ok {
                    stale_derived = true;
                    continue;
                }
                if let Some(a) = k.arch {
                    if a != target.arch {
                        warnings.push(format!(
                            "{lname}: `{}` applies only to {a}, not {}; ignored",
                            flat[i].path, target.arch
                        ));
                        continue;
                    }
                }
                match parse_raw(k, &e.value).and_then(|v| check_value(k, &v).map(|_| v)) {
                    Ok(v) => {
                        values[i] = v;
                        origin[i] = Origin::Layer(li);
                    }
                    Err(msg) => errs.push(format!("{lname}, line {}: `{}`: {msg}", e.line, flat[i].path)),
                }
            }
            if stale_derived {
                warnings.push(format!(
                    "{lname}: [{}]'s derived values are from another derive version (the crate's is {}); \
                     they fall through: run `cargo tune derive`",
                    sec.name, sec.derive
                ));
            }
            if let Some(Raw::Str(u)) = fs.value("upstream") {
                upstream_checks.push((li, sec, u.clone()));
            }
        }
    }

    // Follows defaults, in chain order: repeat until nothing changes (check_schema rules out cycles).
    loop {
        let mut changed = false;
        for i in 0..flat.len() {
            if origin[i] != Origin::Default {
                continue;
            }
            let DefaultValue::Follows { from, plus } = flat[i].knob.default else { continue };
            let j = flat.iter().position(|f| f.path == from).expect("checked by check_schema");
            if origin[j] == Origin::Default && matches!(flat[j].knob.default, DefaultValue::Follows { .. }) {
                continue; // its source isn't resolved yet
            }
            values[i] = match (&values[j], is_int(flat[i].knob.ty)) {
                (Value::Int(v), true) => Value::Int((*v as i128 + plus as i128).clamp(0, u64::MAX as i128) as u64),
                (Value::Float(v), false) => Value::Float(v + plus as f64),
                (v, _) => {
                    errs.push(format!("`{}` follows `{from}`, whose value {v:?} has another type", flat[i].path));
                    continue;
                }
            };
            origin[i] = Origin::Follows;
            changed = true;
        }
        if !changed {
            break;
        }
    }

    for (i, f) in flat.iter().enumerate() {
        // Layer values were checked as they were read; defaults by check_schema, but a Follows
        // value is only known now.
        if matches!(origin[i], Origin::Layer(_)) {
            continue;
        }
        if let Err(msg) = check_value(f.knob, &values[i]) {
            errs.push(format!("`{}` resolves to an invalid value: {msg}", f.path));
        }
    }

    // A section that several layers set is the top one's (the tuner writes sections whole), so
    // only that layer's record is checked: a preset's section under a machine's own isn't stale.
    let mut top: Vec<(usize, &'static Section, String)> = Vec::new();
    for c in upstream_checks {
        top.retain(|x| x.1.name != c.1.name);
        top.push(c);
    }
    for (li, sec, recorded) in top {
        let now = upstream_text(&flat, &values, sec);
        if now != recorded {
            warnings.push(format!(
                "{}: [{}] was derived from different upstream values ({recorded}; now {now}): \
                 run `cargo tune derive`",
                layers[li].name, sec.name
            ));
        }
    }

    if errs.is_empty() {
        Ok(Resolved { values, origin, warnings })
    } else {
        Err(errs)
    }
}
