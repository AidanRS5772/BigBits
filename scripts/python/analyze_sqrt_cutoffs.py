#!/usr/bin/env python3
"""Summarize and fit the complete-call constant-cutoff sqrt probe CSV files."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict


def read(paths):
    values = defaultdict(list)
    for path in paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = tuple(row[k] for k in ("family", "mode", "shape", "pattern", "root_limbs", "capacity", "leaf"))
                values[key].append(float(row["ns"]))
    return {key: statistics.median(ns) for key, ns in values.items()}


def fit(data, family, mode=None):
    cases = defaultdict(dict)
    for key, ns in data.items():
        f, m, shape, pattern, n, cap, leaf = key
        if f == family and (mode is None or m == mode) and m != "core":
            cases[(m, shape, pattern, int(n), cap)][int(leaf)] = ns
    if not cases:
        return None
    leaves = sorted(set.intersection(*(set(v) for v in cases.values())) - {0, 1, 4096})
    assert leaves and all(0 in values and 17 in values for values in cases.values())
    scores = []
    for leaf in leaves:
        for top in range(3, 130):
            ratios = []
            for (_, _, _, n, _), times in cases.items():
                base = times[0 if n < 17 else 17]
                runtime = times[0 if n < top else leaf]
                ratios.append(math.log(runtime / base))
            scores.append({"top": top, "leaf": leaf, "relative": math.exp(statistics.mean(ratios))})
    shared = sorted((x for x in scores if x["top"] == x["leaf"]), key=lambda x: x["relative"])
    split = sorted(scores, key=lambda x: x["relative"])
    return {"family": family, "mode": mode or "all", "cases": len(cases),
            "shared": shared[:8], "split": split[:12]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = read(args.csv)
    if args.fit:
        result = [fit(data, family, mode) for family in ("dyn", "static") for mode in (None, "rem", "only", "approx")]
        text = json.dumps([r for r in result if r], indent=2) + "\n"
    else:
        cases = defaultdict(dict)
        for key, ns in data.items():
            cases[key[:-1]][int(key[-1])] = ns
        lines = ["family,mode,shape,pattern,root_limbs,capacity,binom_ns,best_leaf,best_ns,ratio"]
        for case, values in cases.items():
            choices = {k: v for k, v in values.items() if k not in (0, 1)}
            if not choices or 0 not in values:
                continue
            best = min(choices, key=choices.get)
            lines.append(",".join(case) + f",{values[0]:.3f},{best},{choices[best]:.3f},{choices[best]/values[0]:.6f}")
        text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
