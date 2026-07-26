#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output_dir="${DIV_PROFILE_OUTPUT_DIR:-target/division_profiles}"
seconds="${DIV_PROFILE_SECONDS:-3}"
rate="${DIV_PROFILE_RATE:-1000}"
warmup_ms="${DIV_PROFILE_WARMUP_MS:-100}"

mkdir -p "$output_dir"

build_output="$(mktemp)"
trap 'rm -f "$build_output"' EXIT
cargo build --profile prof --bench div_profile --message-format=json >"$build_output"
profile_bin="$(
    python3 -c '
import json
import sys

path = sys.argv[1]
executable = None
with open(path, encoding="utf-8") as stream:
    for line in stream:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = message.get("target", {})
        if (
            message.get("reason") == "compiler-artifact"
            and target.get("name") == "div_profile"
            and "bench" in target.get("kind", [])
            and message.get("executable")
        ):
            executable = message["executable"]
if executable is None:
    raise SystemExit("cargo did not report the div_profile executable")
print(executable)
' "$build_output"
)"

"$profile_bin" --list >"$output_dir/cases.tsv"

if (($#)); then
    cases=("$@")
else
    cases=()
    while IFS=$'\t' read -r case_name _; do
        cases+=("$case_name")
    done < <(tail -n +2 "$output_dir/cases.tsv")
fi

for case_name in "${cases[@]}"; do
    echo "profiling $case_name"
    samply record \
        --save-only \
        --unstable-presymbolicate \
        --reuse-threads \
        --include-args=4 \
        --rate "$rate" \
        --output "$output_dir/$case_name.json.gz" \
        -- "$profile_bin" \
        --case "$case_name" \
        --seconds "$seconds" \
        --warmup-ms "$warmup_ms" 2>&1 | tee "$output_dir/$case_name.log"
done

python3 scripts/python/analyze_div_profiles.py \
    "$output_dir" \
    --markdown "$output_dir/summary.md" \
    --csv "$output_dir/summary.csv"
