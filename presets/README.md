# Tuning presets

One tuning file per target the developers measure (docs/tuning_system.md, "Presets"). `build.rs`
layers `presets/<arch>-<os>.toml` (for example `x86_64-linux.toml`), else `presets/<arch>.toml`
(for example `aarch64.toml`), over the schema's defaults: the first of the two that exists, and
only that one. A file here is written only by `cargo tune tune --write-preset` on the target's
preset machine (it writes the file a build of that machine resolves, or `<arch>-<os>.toml` for a
target's first), and edited by hand only to record a measurement made some other way, with a
comment saying so. Partial files are normal: a section or key a preset leaves out keeps the
schema default.

`aarch64.toml` was measured on one machine, an M1 (8 threads) under macOS, as its
`[meta] target = "aarch64-macos"` records, and since 2026-11-13 it stands for every aarch64
target until another is measured (it was `aarch64-macos.toml` before). Its values are correct
anywhere and tuned only there: on other aarch64 hardware `cargo tune check --file
presets/aarch64.toml` says whether they beat the defaults, and `cargo tune tune` gives the
machine its own file. A preset for one OS (`aarch64-linux.toml`) takes that target over whole.
