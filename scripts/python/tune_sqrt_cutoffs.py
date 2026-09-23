#!/usr/bin/env python3
"""Build constant-cutoff sqrt probes from the current production source.

Only the generated copy under target/ is parameterized. Arithmetic and outer
wrappers are taken verbatim from src/utils/sqrt.rs; no tuning API enters src/.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def build(output, rlib):
    output.mkdir(parents=True, exist_ok=True)
    original = Path("src/utils/sqrt.rs").read_text()
    source = original
    for name in ["zimmermann_sqrt_core", "zimmerman_sqrt_entry"]:
        source = re.sub(r"\b" + name + r"\(", name + "::<LEAF>(", source)
        source = replace_once(source, "fn " + name + "::<LEAF>(",
                              "fn " + name + "<const LEAF: usize>(")
    source = replace_once(source, "fn sqrt_dyn_output(",
                          "fn sqrt_dyn_output<const TOP: usize, const LEAF: usize>(")
    source = replace_once(source, "fn sqrt_static_output<const N: usize>(",
                          "fn sqrt_static_output<const N: usize, const TOP: usize, const LEAF: usize>(")
    source = replace_once(source, "if s.len() < ZIMMERMAN_SQRT_LEAF_CUTOFF {", "if s.len() < LEAF {")
    source = replace_once(source, "if s.len() < output.dyn_cutoff() {", "if s.len() < TOP {")
    source = replace_once(source, "if s.len() < output.static_cutoff() {", "if s.len() < TOP {")
    # Production wrappers keep compiling; probes call run_dyn/run_static instead.
    source = source.replace("sqrt_dyn_output(x,", "sqrt_dyn_output::<17, 17>(x,")
    source = source.replace("sqrt_static_output::<N>(x,", "sqrt_static_output::<N, 17, 17>(x,")
    # Force precisely the outer core step while letting its child use LEAF.
    # Calling core::<4096> directly would simply measure binomial twice.
    core = source[source.index("fn zimmermann_sqrt_core<"):source.index("// The sole entry into the recursive stack.")]
    core = replace_once(core, "fn zimmermann_sqrt_core<", "fn one_descent<")
    core = replace_once(core, "    if s.len() < LEAF {\n        binom_sqrt_core(x, s);\n        return;\n    }\n", "")
    source += "\n" + core
    source += r'''
pub fn run_dyn<const TOP: usize, const LEAF: usize, const MODE: u8>(x: &mut [u64], s: &mut [u64]) {
    if MODE == 3 {
        if TOP == usize::MAX { binom_sqrt_core(x, s); }
        else { one_descent::<LEAF>(x, s, &mut div_rem_dyn, &mut sqr_dyn); }
    } else {
        let output = match MODE { 0 => SqrtOutput::RootRem, 1 => SqrtOutput::Root, 2 => SqrtOutput::ApproxRoot, _ => unreachable!() };
        sqrt_dyn_output::<TOP, LEAF>(x, s, output);
    }
}
pub fn run_static<const N: usize, const TOP: usize, const LEAF: usize, const MODE: u8>(x: &mut [u64], s: &mut [u64]) {
    if MODE == 3 {
        if TOP == usize::MAX { binom_sqrt_core(x, s); }
        else { one_descent::<LEAF>(x, s, &mut div_rem_static::<N>, &mut sqr_static::<N>); }
    } else {
        let output = match MODE { 0 => SqrtOutput::RootRem, 1 => SqrtOutput::Root, 2 => SqrtOutput::ApproxRoot, _ => unreachable!() };
        sqrt_static_output::<N, TOP, LEAF>(x, s, output);
    }
}
'''
    generated = (output / "sqrt_generated.rs").resolve()
    generated.write_text(source)
    (output / "sqrt_original.rs").write_text(original)
    env = os.environ.copy()
    env["SQRT_TUNE_SOURCE"] = str(generated)
    command = ["rustc", "--edition=2021", "-O", "-g", "-C", "lto=thin", "-C", "codegen-units=1",
               "benches/probes/sqrt_cutoffs.rs", "--extern", f"big_bits={rlib}",
               "-L", f"dependency={rlib.parent}", "-o", str(output / "sqrt_cutoffs")]
    subprocess.run(command, env=env, check=True)
    paths = ["src/utils/sqrt.rs", "src/utils/div.rs", "src/utils/mul.rs", "src/utils/mod.rs",
             "benches/probes/sqrt_cutoffs.rs", "scripts/python/tune_sqrt_cutoffs.py"]
    metadata = {"source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                "command": command, "rlib": str(rlib),
                "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
                "binary_sha256": hashlib.sha256((output / "sqrt_cutoffs").read_bytes()).hexdigest()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("target/sqrt_cutoff_tuning"))
    parser.add_argument("--rlib", type=Path, required=True)
    args = parser.parse_args()
    build(args.output, args.rlib)
