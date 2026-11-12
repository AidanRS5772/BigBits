// The tuning-file format (docs/tuning_system.md, "The tuning file format"): a subset of TOML that
// presets, user files and the tuner share. `#` comments, `[section]` headers and `key = value`
// lines, where a value is an integer (decimal, `_` allowed), a float (`inf` included), `true` or
// `false`, a double-quoted string without escapes, or a flat array of numbers that may span lines.
// Shared by build.rs and the crate, so std only; parsed by hand so users don't compile a TOML
// parser.

use std::fmt::Write as _;

/// One number in an array: tables hold integers, step functions may hold floats.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Num {
    Int(i128),
    Float(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Raw {
    Int(i128),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Num>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub key: String,
    pub value: Raw,
    /// The trailing comment, without its `#`: a value's evidence (invariant 9).
    pub comment: Option<String>,
    pub line: usize,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct FileSection {
    pub name: String,
    pub entries: Vec<Entry>,
}

impl FileSection {
    pub fn get(&self, key: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.key == key)
    }
    /// The value of `key`, if the section sets it.
    pub fn value(&self, key: &str) -> Option<&Raw> {
        self.get(key).map(|e| &e.value)
    }
    /// Sets `key`, replacing an existing entry in place, else appending.
    pub fn set(&mut self, key: &str, value: Raw, comment: Option<String>) {
        match self.entries.iter_mut().find(|e| e.key == key) {
            Some(e) => {
                e.value = value;
                e.comment = comment;
            }
            None => self.entries.push(Entry { key: key.to_string(), value, comment, line: 0 }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct File {
    /// Comment lines before the first section, without their `#`.
    pub header: Vec<String>,
    pub sections: Vec<FileSection>,
}

impl File {
    pub fn section(&self, name: &str) -> Option<&FileSection> {
        self.sections.iter().find(|s| s.name == name)
    }
    pub fn section_mut(&mut self, name: &str) -> &mut FileSection {
        if let Some(i) = self.sections.iter().position(|s| s.name == name) {
            return &mut self.sections[i];
        }
        self.sections.push(FileSection { name: name.to_string(), entries: Vec::new() });
        self.sections.last_mut().unwrap()
    }
    /// Replaces (or adds) each of `other`'s sections whole, keeping the rest of `self`: how the
    /// tuner merges a finished section into an existing file.
    pub fn merge(&mut self, other: &File) {
        for s in &other.sections {
            match self.sections.iter().position(|t| t.name == s.name) {
                Some(i) => self.sections[i] = s.clone(),
                None => self.sections.push(s.clone()),
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    pub line: usize,
    pub msg: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.msg)
    }
}

// Splits `line` at a `#` outside a string.
fn split_comment(line: &str) -> (&str, Option<&str>) {
    let mut in_str = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_str = !in_str,
            '#' if !in_str => return (&line[..i], Some(&line[i + 1..])),
            _ => {}
        }
    }
    (line, None)
}

fn parse_num(t: &str) -> Option<Num> {
    let t = t.trim();
    if t.is_empty() {
        return None;
    }
    let clean: String = t.chars().filter(|&c| c != '_').collect();
    match clean.as_str() {
        "inf" | "+inf" => return Some(Num::Float(f64::INFINITY)),
        "-inf" => return Some(Num::Float(f64::NEG_INFINITY)),
        _ => {}
    }
    let is_float = clean.contains(['.', 'e', 'E']);
    if !is_float {
        if let Ok(v) = clean.parse::<i128>() {
            return Some(Num::Int(v));
        }
        return None;
    }
    // TOML requires digits on both sides of a decimal point.
    let mantissa = clean.split(['e', 'E']).next().unwrap();
    if let Some((a, b)) = mantissa.split_once('.') {
        let a = a.trim_start_matches(['+', '-']);
        if a.is_empty() || b.is_empty() {
            return None;
        }
    }
    clean.parse::<f64>().ok().filter(|v| !v.is_nan()).map(Num::Float)
}

fn parse_scalar(t: &str, line: usize) -> Result<Raw, ParseError> {
    let t = t.trim();
    let err = |msg: String| ParseError { line, msg };
    if t == "true" {
        return Ok(Raw::Bool(true));
    }
    if t == "false" {
        return Ok(Raw::Bool(false));
    }
    if let Some(rest) = t.strip_prefix('"') {
        let body = rest.strip_suffix('"').ok_or_else(|| err(format!("unterminated string `{t}`")))?;
        if body.contains(['"', '\\']) {
            return Err(err(format!("strings may not contain quotes or escapes: `{t}`")));
        }
        return Ok(Raw::Str(body.to_string()));
    }
    match parse_num(t) {
        Some(Num::Int(v)) => Ok(Raw::Int(v)),
        Some(Num::Float(v)) => Ok(Raw::Float(v)),
        None => Err(err(format!("not a value in this format: `{t}`"))),
    }
}

fn parse_array(body: &str, line: usize) -> Result<Raw, ParseError> {
    let mut out = Vec::new();
    for part in body.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue; // a trailing comma, or a line break
        }
        if p.starts_with('[') {
            return Err(ParseError { line, msg: "nested arrays are not allowed".into() });
        }
        out.push(parse_num(p).ok_or_else(|| ParseError {
            line,
            msg: format!("array elements must be numbers: `{p}`"),
        })?);
    }
    Ok(Raw::Array(out))
}

fn valid_name(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

pub fn parse(text: &str) -> Result<File, ParseError> {
    let mut file = File::default();
    let mut lines = text.lines().enumerate().map(|(i, l)| (i + 1, l));
    while let Some((n, raw)) = lines.next() {
        let (content, comment) = split_comment(raw);
        let content = content.trim();
        if content.is_empty() {
            if file.sections.is_empty() {
                if let Some(c) = comment {
                    file.header.push(c.trim().to_string());
                }
            }
            continue;
        }
        if let Some(rest) = content.strip_prefix('[') {
            let name = rest
                .strip_suffix(']')
                .map(str::trim)
                .filter(|s| valid_name(s))
                .ok_or(ParseError { line: n, msg: format!("bad section header `{content}`") })?;
            if file.section(name).is_some() {
                return Err(ParseError { line: n, msg: format!("section [{name}] appears twice") });
            }
            file.sections.push(FileSection { name: name.to_string(), entries: Vec::new() });
            continue;
        }
        let (key, value) = content
            .split_once('=')
            .ok_or(ParseError { line: n, msg: format!("expected `key = value`: `{content}`") })?;
        let key = key.trim();
        if !valid_name(key) {
            return Err(ParseError { line: n, msg: format!("bad key `{key}`") });
        }
        let section = file
            .sections
            .last_mut()
            .ok_or(ParseError { line: n, msg: format!("`{key}` comes before any [section]") })?;
        if section.get(key).is_some() {
            return Err(ParseError { line: n, msg: format!("`{key}` appears twice in [{}]", section.name) });
        }
        let value = value.trim();
        let mut comment = comment.map(|c| c.trim().to_string());
        let parsed = if let Some(rest) = value.strip_prefix('[') {
            // A flat array, possibly over several lines; the comment of its last line is kept.
            let mut body = rest.to_string();
            while !body.contains(']') {
                let (_, next) = lines
                    .next()
                    .ok_or(ParseError { line: n, msg: format!("unterminated array for `{key}`") })?;
                let (c, cm) = split_comment(next);
                body.push(',');
                body.push_str(c);
                if cm.is_some() {
                    comment = cm.map(|c| c.trim().to_string());
                }
            }
            let (inner, after) = body.split_once(']').unwrap();
            if !after.trim().is_empty() {
                return Err(ParseError { line: n, msg: format!("text after the array for `{key}`") });
            }
            parse_array(inner, n)?
        } else {
            parse_scalar(value, n)?
        };
        section.entries.push(Entry { key: key.to_string(), value: parsed, comment, line: n });
    }
    Ok(file)
}

/// A float in the format: shortest round-trip, always with a `.` or exponent so it reads back as
/// a float, and `inf` spelled as TOML spells it.
pub fn float_text(v: f64) -> String {
    if v == f64::INFINITY {
        return "inf".into();
    }
    if v == f64::NEG_INFINITY {
        return "-inf".into();
    }
    format!("{v:?}")
}

pub fn num_text(n: Num) -> String {
    match n {
        Num::Int(v) => v.to_string(),
        Num::Float(v) => float_text(v),
    }
}

pub fn raw_text(r: &Raw) -> String {
    match r {
        Raw::Int(v) => v.to_string(),
        Raw::Float(v) => float_text(*v),
        Raw::Bool(v) => v.to_string(),
        Raw::Str(s) => format!("\"{s}\""),
        Raw::Array(a) => {
            let items: Vec<String> = a.iter().map(|&n| num_text(n)).collect();
            format!("[{}]", items.join(", "))
        }
    }
}

/// Writes a file that `parse` reads back to the same `File` (line numbers aside). Values line up
/// their evidence comments within each section.
pub fn write(file: &File) -> String {
    let mut out = String::new();
    for h in &file.header {
        let _ = writeln!(out, "# {h}");
    }
    for (i, s) in file.sections.iter().enumerate() {
        if i > 0 || !file.header.is_empty() {
            out.push('\n');
        }
        let _ = writeln!(out, "[{}]", s.name);
        let lines: Vec<String> = s.entries.iter().map(|e| format!("{} = {}", e.key, raw_text(&e.value))).collect();
        let width = lines.iter().map(|l| l.len()).max().unwrap_or(0).min(40);
        for (l, e) in lines.iter().zip(&s.entries) {
            match &e.comment {
                Some(c) => {
                    let _ = writeln!(out, "{l:<width$}  # {c}");
                }
                None => {
                    let _ = writeln!(out, "{l}");
                }
            }
        }
    }
    out
}
