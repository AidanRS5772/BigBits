#!/usr/bin/env python3
"""Collect CPU-PC samples and separate unsampled timings for binomial sqrt.

Build with `cargo build --profile prof --bench binom_sqrt_profile`, then pass
the executable under target/prof/deps with --binary. Requires Linux, GCC,
addr2line. Uses ITIMER_PROF rather than privileged perf events.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
import re
import statistics
import subprocess


def resolve(binary: Path, addresses: list[str]) -> dict[str, list[tuple[str, str]]]:
    result = subprocess.run(
        ["addr2line", "-a", "-f", "-C", "-i", "-e", str(binary)],
        input="\n".join(addresses) + "\n", text=True, capture_output=True, check=True,
    )
    resolved = {}
    current = None
    lines = iter(result.stdout.splitlines())
    for line in lines:
        if re.fullmatch(r"0x[0-9a-f]+", line):
            current = f"{int(line, 16):x}"
            resolved[current] = []
        else:
            resolved[current].append((line, next(lines)))
    return resolved


def phase(frames: list[tuple[str, str]], symbol: str) -> str:
    names = " ".join(name for name, _ in frames) + " " + symbol
    for needle, category in [
        ("sub_mul_of", "quotient multiply/subtract"),
        ("knuth_est", "quotient estimate/correction"),
        ("sqrt_4x2", "initial root seed"),
        ("isqrt", "initial root seed"),
        ("int_sqrt", "initial root seed"),
        ("udivti", "initial root seed (u128 division)"),
        ("udivmodti", "initial root seed (u128 division)"),
        ("umodti", "initial root seed (u128 division)"),
        ("specialized_div_rem", "initial root seed (u128 division)"),
        ("correct_sqrt", "root correction"),
        ("sqrt_denormalization", "denormalization"),
        ("add_mul", "denormalization"),
        ("shl_buf", "normalization shifts"),
        ("shr_buf", "normalization shifts"),
        ("cmp_buf", "remainder comparison"),
        ("add_buf", "limb addition (shared callers)"),
        ("sub_buf", "limb subtraction (shared callers)"),
        ("binom_sqrt_est_reduced", "root digit estimate bookkeeping"),
        ("binom_sqrt_core", "core control/memory"),
        ("big_bits::utils::sqrt::binom_sqrt", "wrapper/control"),
        ("memcpy", "memory (includes input reset)"),
        ("memmove", "memory (includes padding)"),
        ("memset", "zero fill"),
    ]:
        if needle in names:
            return category
    return "harness/runtime/unresolved"


def analyze(path: Path, binary: Path, classify=phase) -> dict:
    with path.open() as stream:
        header = stream.readline().strip()
        rows = list(csv.DictReader(stream, delimiter="\t"))
    own = [row for row in rows if Path(row["module"]).resolve() == binary]
    resolved = resolve(binary, [row["offset"] for row in own])
    counts = Counter()
    hotspots = Counter()
    total = sum(int(row["count"]) for row in rows)
    for row in rows:
        frames = resolved.get(row["offset"], []) if row in own else []
        count = int(row["count"])
        counts[classify(frames, row["symbol"])] += count
        if frames:
            name, location = frames[0]
            hotspots[f"{name} @ {location}"] += count
        else:
            hotspots[f"{row['module']}:{row['symbol']}"] += count
        row["frames"] = frames
    if total < 100:
        raise RuntimeError(f"Too few CPU samples: {total}")
    return {
        "sampler": header, "samples": total,
        "phases_pct": {name: 100 * n / total for name, n in counts.most_common()},
        "hotspots_pct": {name: 100 * n / total for name, n in hotspots.most_common()},
        "addresses": rows,
    }


def run(command: list[str], **kwargs) -> str:
    return subprocess.run(command, text=True, capture_output=True, check=True, **kwargs).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("target/binom_sqrt_profiles"))
    parser.add_argument("--sizes", default="1,2,4,8,16,32,64,128,256,512,1024")
    parser.add_argument("--shapes", default="full,padded")
    parser.add_argument("--patterns", default="random")
    parser.add_argument("--seconds", type=float, default=3)
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
                name = f"n{size}_{shape}_{pattern}"
                base = [str(binary), "--limbs", str(size),
                        "--shape", shape, "--pattern", pattern, "--mode", "binom"]
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
                result = analyze(args.output / f"{name}.tsv", binary)
                result.update(case=name, limbs=size, shape=shape, pattern=pattern,
                              median_ns=statistics.median(ns), timing_ns=ns)
                (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
                summary.append({k: v for k, v in result.items() if k != "addresses"})
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                top = ", ".join(f"{key} {value:.1f}%" for key, value in list(result["phases_pct"].items())[:4])
                print(f"{name}: {result['median_ns']:.1f} ns, {result['samples']} samples; {top}", flush=True)


if __name__ == "__main__":
    main()
