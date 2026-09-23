#!/usr/bin/env python3
"""Profile Zimmermann sqrt using the shared single-thread CPU sampler.

Requires the binom_sqrt_profile benchmark, built with --profile prof. The
zimmer* modes assert that the root width reaches the production sqrt cutoff.
"""

import argparse
import json
import os
from pathlib import Path
import re
import statistics

from profile_binom_sqrt import analyze, run


def phase(frames, symbol):
    names = " ".join(name for name, _ in frames) + " " + symbol
    for needles, category in [
        (("sqrt_denormalization", "sqrt::add_mul"), "sqrt denormalization"),
        (("sub_mul_of",), "division multiply/subtract"),
        (("knuth_est", "div_rem_2_1"), "division quotient estimate"),
        (("sqrt_4x2", "int_sqrt", "isqrt", "udivti", "udivmodti", "specialized_div_rem"), "binomial seed"),
        (("binom_sqrt_est_reduced", "binom_sqrt_core"), "binomial control"),
        (("sqr_elem", "sqr_buf", "mul_double_asm", "sqr_prim", "sqr_entry", "sqr_core", "sqr_mul"), "squaring kernels"),
        (("rustfft", "fft_", "sqr_recombine", "scale_and_round", "decompose", "recombine"), "FFT kernels (shared)"),
        (("ntt", "mont", "mul_reduce", "shuffle"), "NTT kernels (shared)"),
        (("mul_asm", "mul_elem", "mul_buf", "mul_prim", "karatsuba", "hi_mul", "mid_mul"), "multiplication kernels (shared)"),
        (("add_buf", "sub_buf", "add_asm", "sub_asm", "inc_asm", "dec_asm", "add_prim", "sub_prim", "inc_buf", "dec_buf"), "limb addition/subtraction (shared)"),
        (("shl_", "shr_"), "limb shifts (shared)"),
        (("cmp_buf", "buf_len", "eq_buf"), "comparison/trimming (shared)"),
        (("ScratchGuard", "SCRATCH_POOL", "RefCell", "alloc::alloc", "__rust_alloc", "__rust_dealloc", "_int_malloc", "malloc", "free"), "scratch/allocation"),
        (("memcpy", "memmove", "memset", "copy_nonoverlapping", "write_bytes"), "memory operations (shared)"),
        (("big_bits::utils::div::",), "division control/other"),
        (("big_bits::utils::sqrt::",), "sqrt control/other"),
    ]:
        if any(needle in names for needle in needles):
            return category
    return "harness/runtime/unresolved"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="17,32,64,128,256,512,1024,2048")
    parser.add_argument("--shapes", default="full,padded")
    parser.add_argument("--patterns", default="random")
    parser.add_argument("--modes", default="zimmer")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--timing-seconds", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    sampler = (args.output / "sampler.so").resolve()
    run(["gcc", "-O2", "-Wall", "-Wextra", "-fPIC", "-shared",
         "benches/binom_sqrt_sampler.c", "-o", str(sampler), "-ldl"])
    summary = []
    for size in map(int, args.sizes.split(",")):
        for shape in args.shapes.split(","):
            for pattern in args.patterns.split(","):
                for mode in args.modes.split(","):
                    name = f"n{size}_{shape}_{pattern}_{mode}"
                    base = [str(binary), "--limbs", str(size), "--shape", shape,
                            "--pattern", pattern, "--mode", mode]
                    logs = []
                    ns = []
                    for _ in range(args.repeats):
                        log = run(base + ["--seconds", str(args.timing_seconds)])
                        logs.append(log)
                        ns.append(float(re.search(r"ns_per_op=([0-9.]+)", log)[1]))
                    env = os.environ.copy()
                    env["LD_PRELOAD"] = str(sampler)
                    env["BIGBITS_PROFILE_OUT"] = str((args.output / f"{name}.tsv").resolve())
                    logs.append(run(base + ["--seconds", str(args.seconds)], env=env))
                    (args.output / f"{name}.log").write_text("".join(logs))
                    result = analyze(args.output / f"{name}.tsv", binary, phase)
                    result.update(case=name, limbs=size, shape=shape, pattern=pattern,
                                  mode=mode, median_ns=statistics.median(ns), timing_ns=ns)
                    (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
                    summary.append({k: v for k, v in result.items() if k != "addresses"})
                    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                    top = ", ".join(f"{key} {value:.1f}%" for key, value in list(result["phases_pct"].items())[:4])
                    print(f"{name}: {result['median_ns']:.1f} ns, {result['samples']} samples; {top}", flush=True)


if __name__ == "__main__":
    main()
