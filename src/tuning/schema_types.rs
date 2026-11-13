// The schema's types: plain `const` data describing every performance knob (see
// docs/tuning_system.md, "The schema"). Shared by build.rs and the crate through `#[path]`, so this
// file uses only core/std and refers to the other shared files only through `super::`.

/// An algorithm a dispatcher's table can pick, with the smallest size at which it is eligible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Alg {
    pub name: &'static str,
    pub min: u32,
}

/// A knob's type. Scalars keep the Rust type of the constant they replaced, so call sites change
/// only names. A table is a dispatcher's change points (Phase 2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ty {
    Usize,
    U64,
    F64,
    Bool,
    /// One input: `[size, alg, size, alg, …]`. Two inputs: `[s, l, alg, …]`, row by row.
    Table {
        algs: &'static [Alg],
        measure: &'static str,
        inputs: u8,
    },
}

/// A knob's declared default. `Follows` takes another knob's resolved value plus a constant,
/// after layering, unless a layer sets this knob itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DefaultValue {
    Int(u64),
    Float(f64),
    Bool(bool),
    Table(&'static [u32]),
    /// A step function of the thread count: (threads, value) from each count up.
    Steps(&'static [(u32, f64)]),
    Follows { from: &'static str, plus: i64 },
}

/// A range of values. `valid` is a correctness contract (invariant 1); `search` is where a sweep
/// looks, always inside `valid`. A table's validity is structural (ascending change points,
/// eligible picks), so its range is `Table`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Range {
    Int { min: u64, max: u64 },
    Float { min: f64, max: f64 },
    Bool,
    Table,
}

/// How a knob's value is decided. `Interim` is an old `mod.rs` constant migrated as it was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Interim,
    Model { work: &'static str },
    Sweep,
    /// A dispatcher's table, measured: at each index where reading the dispatcher's equations would
    /// cost more than `tuner::tables::SHARE` of the call, the algorithm that was fastest there. Its
    /// layout is the section's `derive` number.
    Table,
}

/// Which thread count a knob's predictions or value read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threads {
    One,
    Pool,
    Cpus,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Knob {
    pub name: &'static str,
    /// The `mod.rs` constant this knob replaced, or "" for one the method adds.
    pub old: &'static str,
    pub ty: Ty,
    pub default: DefaultValue,
    pub valid: Range,
    pub search: Range,
    pub kind: Kind,
    pub threads: Threads,
    /// The target architecture it applies to, if not all (`CARGO_CFG_TARGET_ARCH` spelling).
    pub arch: Option<&'static str>,
    pub doc: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Section {
    pub name: &'static str,
    /// Bumped when the section's measured knobs or their meaning change (invariant 10).
    pub version: u32,
    /// The layout of its tables (`Kind::Table`), bumped when one changes: a file's section loses its
    /// tables, and nothing else, when only this number is stale. (Named for the knobs once computed
    /// from measured ones; none are now.)
    pub derive: u32,
    /// Upstream sections, whose measured values are hashed into this section's upstream hash.
    pub after: &'static [&'static str],
    pub doc: &'static str,
    pub knobs: &'static [Knob],
}

const fn full_range(ty: Ty) -> Range {
    match ty {
        Ty::Usize => Range::Int { min: 0, max: usize::MAX as u64 },
        Ty::U64 => Range::Int { min: 0, max: u64::MAX },
        Ty::F64 => Range::Float { min: 0.0, max: f64::MAX },
        Ty::Bool => Range::Bool,
        Ty::Table { .. } => Range::Table,
    }
}

// Builders, so a schema entry fits on a few lines without a macro:
// `Knob::usize("leaf", "ZIMMERMAN_SQRT_LEAF_CUTOFF", 11).min(1).doc("…")`.
impl Knob {
    pub const fn new(name: &'static str, old: &'static str, ty: Ty, default: DefaultValue) -> Knob {
        Knob {
            name,
            old,
            ty,
            default,
            valid: full_range(ty),
            search: full_range(ty),
            kind: Kind::Interim,
            threads: Threads::One,
            arch: None,
            doc: "",
        }
    }
    pub const fn usize(name: &'static str, old: &'static str, v: usize) -> Knob {
        Knob::new(name, old, Ty::Usize, DefaultValue::Int(v as u64))
    }
    pub const fn u64(name: &'static str, old: &'static str, v: u64) -> Knob {
        Knob::new(name, old, Ty::U64, DefaultValue::Int(v))
    }
    pub const fn f64(name: &'static str, old: &'static str, v: f64) -> Knob {
        Knob::new(name, old, Ty::F64, DefaultValue::Float(v))
    }
    pub const fn bool(name: &'static str, old: &'static str, v: bool) -> Knob {
        Knob::new(name, old, Ty::Bool, DefaultValue::Bool(v))
    }
    /// Sets the valid range (and the search range, until `search` narrows it).
    pub const fn valid(mut self, r: Range) -> Knob {
        self.valid = r;
        self.search = r;
        self
    }
    /// An integer knob valid from `min` to its type's maximum.
    pub const fn min(self, min: u64) -> Knob {
        let max = match self.valid {
            Range::Int { max, .. } => max,
            _ => panic!("min() needs an integer knob"),
        };
        self.valid(Range::Int { min, max })
    }
    /// An integer knob valid from `min` to `max`.
    pub const fn int_range(self, min: u64, max: u64) -> Knob {
        self.valid(Range::Int { min, max })
    }
    /// A float knob valid from `min` to `max`.
    pub const fn float_range(self, min: f64, max: f64) -> Knob {
        self.valid(Range::Float { min, max })
    }
    pub const fn search(mut self, r: Range) -> Knob {
        self.search = r;
        self
    }
    pub const fn kind(mut self, k: Kind) -> Knob {
        self.kind = k;
        self
    }
    pub const fn threads(mut self, t: Threads) -> Knob {
        self.threads = t;
        self
    }
    pub const fn arch(mut self, a: &'static str) -> Knob {
        self.arch = Some(a);
        self
    }
    pub const fn follows(mut self, from: &'static str, plus: i64) -> Knob {
        self.default = DefaultValue::Follows { from, plus };
        self
    }
    pub const fn doc(mut self, d: &'static str) -> Knob {
        self.doc = d;
        self
    }
}
