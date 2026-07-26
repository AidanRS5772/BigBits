#!/usr/bin/env python3
"""Summarize Samply division profiles by source owner and division phase."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DIV_FILE = "/src/utils/div.rs"
MUL_FILE = "/src/utils/mul.rs"
UTILS_FILE = "/src/utils/utils.rs"
HARNESS_FILE = "/benches/div_profile.rs"


@dataclass(frozen=True)
class LogicalFrame:
    function: str
    file: str | None
    line: int | None


@dataclass
class ProfileSummary:
    case: str
    metadata: dict[str, str]
    samples: float
    categories: Counter[str]
    phases: Counter[str]
    div_hotspots: Counter[str]
    division_hotspots: Counter[str]
    mul_hotspots: Counter[str]
    other_hotspots: Counter[str]

    def percent(self, count: float) -> float:
        return 100.0 * count / self.samples if self.samples else 0.0


class SymbolResolver:
    def __init__(self, profile: dict[str, Any], symbols: dict[str, Any]) -> None:
        self.strings: list[str] = symbols["string_table"]
        self.by_code_id = {
            item.get("code_id"): item for item in symbols["data"] if item.get("code_id")
        }
        self.by_debug_name = {
            item.get("debug_name"): item
            for item in symbols["data"]
            if item.get("debug_name")
        }
        self.library_symbols: list[tuple[dict[int, int], list[dict[str, Any]]] | None] = []
        for library in profile.get("libs", []):
            item = self.by_code_id.get(library.get("codeId"))
            if item is None:
                item = self.by_debug_name.get(library.get("debugName"))
            if item is None:
                self.library_symbols.append(None)
                continue
            known = {int(address): int(index) for address, index in item["known_addresses"]}
            self.library_symbols.append((known, item["symbol_table"]))

    def _text(self, index: int | None) -> str | None:
        return self.strings[index] if index is not None else None

    def resolve(self, thread: dict[str, Any], frame_index: int) -> list[LogicalFrame]:
        frames = thread["frameTable"]
        funcs = thread["funcTable"]
        resources = thread["resourceTable"]
        func_index = frames["func"][frame_index]
        resource_index = funcs["resource"][func_index]
        library_index = resources["lib"][resource_index]
        address = int(frames["address"][frame_index])

        if library_index is not None and library_index < len(self.library_symbols):
            library = self.library_symbols[library_index]
            if library is not None:
                known, symbol_table = library
                symbol_index = known.get(address)
                if symbol_index is not None:
                    symbol = symbol_table[symbol_index]
                    inline_frames = symbol.get("frames")
                    if inline_frames:
                        return [
                            LogicalFrame(
                                self._text(frame.get("function")) or "UNKNOWN",
                                self._text(frame.get("file")),
                                frame.get("line"),
                            )
                            for frame in inline_frames
                        ]
                    return [
                        LogicalFrame(
                            self._text(symbol.get("symbol")) or "UNKNOWN", None, None
                        )
                    ]

        profile_strings = thread["stringArray"]
        name_index = funcs["name"][func_index]
        file_index = funcs.get("fileName", [None] * funcs["length"])[func_index]
        line = funcs.get("lineNumber", [None] * funcs["length"])[func_index]
        return [
            LogicalFrame(
                profile_strings[name_index],
                profile_strings[file_index] if file_index is not None else None,
                line,
            )
        ]


def stack_frames(
    thread: dict[str, Any], stack_index: int, resolver: SymbolResolver
) -> list[LogicalFrame]:
    stack_table = thread["stackTable"]
    physical: list[int] = []
    current: int | None = stack_index
    while current is not None:
        physical.append(stack_table["frame"][current])
        current = stack_table["prefix"][current]
    physical.reverse()

    logical: list[LogicalFrame] = []
    for frame_index in physical:
        # Samply's sidecar stores inline frames from innermost to outermost.
        logical.extend(reversed(resolver.resolve(thread, frame_index)))
    return logical


def source_owner(frame: LogicalFrame) -> str | None:
    file = frame.file or ""
    if file.endswith(DIV_FILE):
        return "division"
    if file.endswith(MUL_FILE):
        return "multiplication"
    if file.endswith(UTILS_FILE):
        return "limb_helpers"
    if file.endswith(HARNESS_FILE):
        return "harness"
    return None


def nearest_owner(stack: list[LogicalFrame], skip_leaf: bool = False) -> str | None:
    frames = reversed(stack[:-1] if skip_leaf else stack)
    helper_seen = False
    for frame in frames:
        owner = source_owner(frame)
        if owner == "limb_helpers":
            helper_seen = True
            continue
        if owner is not None:
            if helper_seen and owner in {"division", "multiplication"}:
                return f"{owner}_helpers"
            return owner
    return "limb_helpers" if helper_seen else None


def classify(stack: list[LogicalFrame], ntt_case: bool) -> str:
    if not stack:
        return "other"
    leaf_owner = source_owner(stack[-1])
    if leaf_owner == "division":
        return "division_direct"
    if leaf_owner == "multiplication":
        return "multiplication_direct"
    if leaf_owner == "harness":
        return "harness"
    if leaf_owner == "limb_helpers":
        owner = nearest_owner(stack, skip_leaf=True)
        if owner == "division":
            return "division_helpers"
        if owner == "multiplication":
            return "multiplication_helpers"
        return "other"

    owner = nearest_owner(stack)
    if owner in {"division", "division_helpers"}:
        return "division_support"
    if owner in {"multiplication", "multiplication_helpers"}:
        return "multiplication_support"
    if owner == "harness":
        return "harness"
    if ntt_case and any(
        "rayon" in frame.function or "rayon-core" in (frame.file or "")
        for frame in stack
    ):
        return "multiplication_support"
    return "other"


def division_phase(stack: list[LogicalFrame]) -> str:
    division_functions = [
        frame.function
        for frame in reversed(stack)
        if (frame.file or "").endswith(DIV_FILE)
    ]
    if not division_functions:
        return "outside_division_stack"
    for function in division_functions:
        if "nr_err_band" in function:
            continue
        if any(
            token in function
            for token in ("nr_refine_rcp", "nr_rcp_chain", "nr_div_rcp", "nr_rcp_wrapper")
        ):
            return "nr_reciprocal"
        if any(token in function for token in ("nr_refine_quo", "nr_quo_est")):
            return "nr_quotient"
        if "nr_rem_finish" in function:
            return "nr_remainder"
        if "nr_exact_correction" in function:
            return "nr_exact_correction"
        if any(
            token in function
            for token in ("knuth", "div_buf_of", "div_rem_2_1", "div_prim")
        ):
            return "knuth"
        if any(
            token in function
            for token in ("bz_", "div_2_1", "div_3_2")
        ):
            return "burnikel_ziegler"
    return "dispatch_wrapper"


def concise_function(frame: LogicalFrame) -> str:
    name = frame.function
    for prefix in (
        "big_bits::utils::div::",
        "big_bits::utils::mul::",
        "big_bits::utils::utils::",
    ):
        name = name.replace(prefix, "")
    if frame.line is not None and frame.file:
        return f"{name}:{frame.line}"
    return name


def load_case_metadata(directory: Path) -> dict[str, dict[str, str]]:
    path = directory / "cases.tsv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["name"]: row
            for row in csv.DictReader(stream, delimiter="\t")
        }


def sidecar_path(profile_path: Path) -> Path:
    if profile_path.name.endswith(".json.gz"):
        return profile_path.with_name(profile_path.name[:-3] + ".syms.json")
    return profile_path.with_suffix(profile_path.suffix + ".syms.json")


def summarize_profile(
    profile_path: Path, metadata: dict[str, str]
) -> ProfileSummary:
    with gzip.open(profile_path, "rt", encoding="utf-8") as stream:
        profile = json.load(stream)
    with sidecar_path(profile_path).open(encoding="utf-8") as stream:
        symbols = json.load(stream)

    resolver = SymbolResolver(profile, symbols)
    categories: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    div_hotspots: Counter[str] = Counter()
    division_hotspots: Counter[str] = Counter()
    mul_hotspots: Counter[str] = Counter()
    other_hotspots: Counter[str] = Counter()
    samples = 0.0
    ntt_case = "ntt" in metadata.get("expected_path", "")

    for thread in profile.get("threads", []):
        sample_table = thread.get("samples", {})
        stacks = sample_table.get("stack", [])
        cpu_deltas = sample_table.get("threadCPUDelta")
        if cpu_deltas and any(delta for delta in cpu_deltas if delta is not None):
            # Wall-clock sample counts overstate idle Rayon workers. Attribute
            # each sampled stack by CPU time consumed since the prior sample.
            weights = [
                max(float(delta or 0), 0.0) / 1000.0 for delta in cpu_deltas
            ]
        else:
            weights = [float(weight) for weight in (sample_table.get("weight") or [1] * len(stacks))]
        for stack_index, weight in zip(stacks, weights):
            if stack_index is None or weight == 0:
                continue
            logical = stack_frames(thread, stack_index, resolver)
            category = classify(logical, ntt_case)
            categories[category] += weight
            phases[division_phase(logical)] += weight
            samples += weight
            if logical:
                if category == "division_direct":
                    div_hotspots[concise_function(logical[-1])] += weight
                if category.startswith("division_"):
                    division_hotspots[concise_function(logical[-1])] += weight
                if category.startswith("multiplication_"):
                    mul_hotspots[concise_function(logical[-1])] += weight
                if category == "other":
                    other_hotspots[concise_function(logical[-1])] += weight

    case = profile_path.name.removesuffix(".json.gz")
    return ProfileSummary(
        case,
        metadata,
        samples,
        categories,
        phases,
        div_hotspots,
        division_hotspots,
        mul_hotspots,
        other_hotspots,
    )


def aggregate(summary: ProfileSummary, prefix: str) -> float:
    return sum(
        count
        for category, count in summary.categories.items()
        if category.startswith(prefix)
    )


def top(counter: Counter[str], count: int = 3) -> str:
    return ", ".join(f"{name} ({value:g})" for name, value in counter.most_common(count))


def write_csv(path: Path, summaries: Iterable[ProfileSummary]) -> None:
    fields = [
        "case",
        "model",
        "operation",
        "expected_path",
        "cpu_ms",
        "div_rs_direct_pct",
        "division_total_pct",
        "multiplication_total_pct",
        "harness_pct",
        "other_pct",
        "top_div_hotspot",
        "top_phase",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "case": summary.case,
                    "model": summary.metadata.get("model", ""),
                    "operation": summary.metadata.get("operation", ""),
                    "expected_path": summary.metadata.get("expected_path", ""),
                    "cpu_ms": f"{summary.samples:.0f}",
                    "div_rs_direct_pct": f"{summary.percent(summary.categories['division_direct']):.2f}",
                    "division_total_pct": f"{summary.percent(aggregate(summary, 'division_')):.2f}",
                    "multiplication_total_pct": f"{summary.percent(aggregate(summary, 'multiplication_')):.2f}",
                    "harness_pct": f"{summary.percent(summary.categories['harness']):.2f}",
                    "other_pct": f"{summary.percent(summary.categories['other']):.2f}",
                    "top_div_hotspot": top(summary.div_hotspots, 1),
                    "top_phase": top(summary.phases, 1),
                }
            )


def write_markdown(path: Path, summaries: list[ProfileSummary]) -> None:
    lines = [
        "# Division profile sample summary",
        "",
        "Percentages are CPU-sample shares. `div.rs direct` counts instructions "
        "attributed directly to `src/utils/div.rs`; `division total` adds helper/runtime "
        "work whose nearest BigBits caller is division. `multiplication` includes "
        "`mul.rs`, FFT/NTT dependencies, and runtime work below multiplication frames.",
        "",
        "| case | expected path | CPU ms | div.rs direct | division total | multiplication | harness | other |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        expected = summary.metadata.get("expected_path", "")
        lines.append(
            f"| `{summary.case}` | {expected} | {summary.samples:.0f} | "
            f"{summary.percent(summary.categories['division_direct']):.1f}% | "
            f"{summary.percent(aggregate(summary, 'division_')):.1f}% | "
            f"{summary.percent(aggregate(summary, 'multiplication_')):.1f}% | "
            f"{summary.percent(summary.categories['harness']):.1f}% | "
            f"{summary.percent(summary.categories['other']):.1f}% |"
        )

    lines.extend(["", "## Per-case hotspots", ""])
    for summary in summaries:
        lines.extend(
            [
                f"### `{summary.case}`",
                "",
                f"- Division phases: {top(summary.phases, 5) or 'none'}",
                f"- Direct `div.rs` leaves: {top(summary.div_hotspots, 5) or 'none'}",
                f"- All division-owned leaves: {top(summary.division_hotspots, 5) or 'none'}",
                f"- Multiplication leaves: {top(summary.mul_hotspots, 5) or 'none'}",
                f"- Other leaves: {top(summary.other_hotspots, 5) or 'none'}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    metadata = load_case_metadata(args.profile_dir)
    summaries = []
    for profile_path in sorted(args.profile_dir.glob("*.json.gz")):
        if not sidecar_path(profile_path).exists():
            continue
        case = profile_path.name.removesuffix(".json.gz")
        if metadata and case not in metadata:
            continue
        summaries.append(summarize_profile(profile_path, metadata.get(case, {})))

    if not summaries:
        raise SystemExit("no Samply profiles with symbol sidecars found")
    if args.csv:
        write_csv(args.csv, summaries)
    if args.markdown:
        write_markdown(args.markdown, summaries)

    for summary in summaries:
        print(
            f"{summary.case}: cpu_ms={summary.samples:.0f} "
            f"div.rs={summary.percent(summary.categories['division_direct']):.1f}% "
            f"division={summary.percent(aggregate(summary, 'division_')):.1f}% "
            f"multiplication={summary.percent(aggregate(summary, 'multiplication_')):.1f}% "
            f"harness={summary.percent(summary.categories['harness']):.1f}% "
            f"other={summary.percent(summary.categories['other']):.1f}%"
        )


if __name__ == "__main__":
    main()
