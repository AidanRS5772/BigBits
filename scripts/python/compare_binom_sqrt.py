#!/usr/bin/env python3
"""Compare saved binom_sqrt_profile executables on identical deterministic inputs.

All variants call binom_sqrt directly, bypassing algorithm cutoffs. Build each
executable with the same Rust toolchain/profile. Example:
  python3 scripts/python/compare_binom_sqrt.py --baseline /tmp/baseline \
    --candidate /tmp/current --div-only /tmp/div_only --output target/sqrt_compare
"""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--div-only", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=0.2)
    args = parser.parse_args()
    assert args.rounds > 0 and args.seconds > 0
    os.sched_setaffinity(0, {args.cpu})
    variants = {"baseline": args.baseline.resolve()}
    if args.div_only:
        variants["div_only"] = args.div_only.resolve()
    variants["current"] = args.candidate.resolve()
    cases = [(n, "full", "random") for n in (1, 2, 3, 4, 8, 16, 32, 48, 63, 64, 128, 256)]
    cases += [(n, "padded", "random") for n in (2, 4, 8, 16, 32, 63)]
    cases += [(n, "full", p) for p in ("shifted", "ones", "power") for n in (2, 8, 16, 32, 63)]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps({
        "binaries": {k: str(v) for k, v in variants.items()},
        "cpu": args.cpu, "rounds": args.rounds, "seconds": args.seconds,
        "cases": cases,
    }, indent=2) + "\n")
    summaries = []
    with (args.output / "rounds.csv").open("w", newline="") as raw:
        writer = csv.DictWriter(raw, fieldnames=[
            "limbs", "shape", "pattern", "round", "variant", "ns_per_op", "checksum"
        ])
        writer.writeheader()
        for n, shape, pattern in cases:
            timings = {name: [] for name in variants}
            checksums = set()
            names = list(variants)
            for round_index in range(args.rounds):
                for step in range(len(names)):
                    name = names[(round_index + step) % len(names)]
                    result = subprocess.run([
                        str(variants[name]), "--limbs", str(n), "--shape", shape,
                        "--pattern", pattern, "--mode", "binom", "--seconds", str(args.seconds),
                    ], text=True, capture_output=True, check=True)
                    line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
                    fields = dict(item.split("=", 1) for item in line.split()[1:])
                    ns = float(fields["ns_per_op"])
                    checksums.add(fields["checksum"])
                    timings[name].append(ns)
                    writer.writerow({
                        "limbs": n, "shape": shape, "pattern": pattern,
                        "round": round_index, "variant": name,
                        "ns_per_op": ns, "checksum": fields["checksum"],
                    })
                    raw.flush()
            assert len(checksums) == 1, (n, shape, pattern, checksums)
            summary = {"limbs": n, "shape": shape, "pattern": pattern, "variants": {}}
            for name, samples in timings.items():
                summary["variants"][name] = {
                    "median_ns": statistics.median(samples),
                    "min_ns": min(samples), "max_ns": max(samples),
                }
            baseline = summary["variants"]["baseline"]["median_ns"]
            candidate = summary["variants"]["current"]["median_ns"]
            summary["reduction_percent"] = 100 * (1 - candidate / baseline)
            summaries.append(summary)
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(f"n={n:3d} {shape:6s} {pattern:7s}: "
                  f"{baseline:10.3f} -> {candidate:10.3f} ns "
                  f"({summary['reduction_percent']:+.2f}% less time)", flush=True)


if __name__ == "__main__":
    main()
