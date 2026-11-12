# Tuning presets

One tuning file per target the developers measure (docs/tuning_system.md, "Presets"). `build.rs`
layers `presets/<arch>-<os>.toml` (for example `aarch64-macos.toml`), else `presets/<arch>.toml`
(for example `x86_64.toml`), over the schema's defaults. A file here is written only by
`cargo tune <section> --write-preset` on the target's preset machine, and edited by hand only to
record a measurement made some other way, with a comment saying so. Partial files are normal: a
section or key a preset leaves out keeps the schema default.
