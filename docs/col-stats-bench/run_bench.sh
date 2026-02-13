#!/usr/bin/env bash
set -euo pipefail

# Col-stats benchmark: compare ingest encode stage with and without --col-stats.
#
# Usage:
#   ./run_bench.sh [OPTIONS]
#
# Options:
#   --data-dir DIR       Path to generated benchmark data (default: /mnt/localssd/col-stats-bench)
#   --iterations N       Number of iterations per mode (default: 3)
#   --encode-only        Kill process after encode stage (default: run full pipeline)
#   --sizes "100M 500M"  Space-separated list of dataset sizes to test
#   --help               Show this help
#
# Prerequisites:
#   - Build: cd ~/kumo-diskgraph && cargo build --release --bin ingest
#   - Generate data: python generate.py --rows-per-table 100000000 --output-dir /mnt/localssd/col-stats-bench/100M

BIN="${COL_STATS_BIN:-$(dirname "$0")/../../target/release/ingest}"
DATA_BASE="${COL_STATS_DATA:-/mnt/localssd/col-stats-bench}"
OUTPUT_BASE="${COL_STATS_OUTPUT:-/mnt/localssd/col-stats-bench-output}"
ITERATIONS=3
ENCODE_ONLY=false
SIZES="100M"

usage() {
    sed -n '3,/^$/s/^# //p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_BASE="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --encode-only) ENCODE_ONLY=true; shift ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binary not found: $BIN"
    echo "Build with: cd ~/kumo-diskgraph && cargo build --release --bin ingest"
    exit 1
fi

# Increase file descriptor limit
ulimit -n 100000 2>/dev/null || true

RESULTS_FILE="${OUTPUT_BASE}/bench_results_$(date +%Y%m%d_%H%M%S).json"
mkdir -p "$OUTPUT_BASE"

echo "{"  > "$RESULTS_FILE"
echo '  "timestamp": "'$(date -Iseconds)'",' >> "$RESULTS_FILE"
echo '  "binary": "'$BIN'",' >> "$RESULTS_FILE"
echo '  "iterations": '$ITERATIONS',' >> "$RESULTS_FILE"
echo '  "encode_only": '$ENCODE_ONLY',' >> "$RESULTS_FILE"
echo '  "results": {' >> "$RESULTS_FILE"

first_size=true

for SIZE in $SIZES; do
    DATA_DIR="${DATA_BASE}/${SIZE}"
    CONFIG="${DATA_DIR}/diskgraph_config.json"

    if [[ ! -f "$CONFIG" ]]; then
        echo "SKIP: no config at $CONFIG (generate data first)"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "  Dataset: $SIZE  ($CONFIG)"
    echo "================================================================"

    if $first_size; then first_size=false; else echo ',' >> "$RESULTS_FILE"; fi
    echo '    "'$SIZE'": {' >> "$RESULTS_FILE"

    # Arrays to collect timings
    baseline_times=()
    colstats_times=()

    run_ingest() {
        local label="$1"
        local extra_args="$2"
        local output_dir="${OUTPUT_BASE}/${SIZE}_${label}"
        local log_file="${OUTPUT_BASE}/${SIZE}_${label}.log"

        rm -rf "$output_dir"

        echo -n "  [$label] running... "

        if $ENCODE_ONLY; then
            # Run in background, kill after encode stage
            "$BIN" --source "$CONFIG" --output "$output_dir" --dest local --parallel $extra_args \
                > "$log_file" 2>&1 &
            local pid=$!

            # Wait for encode stage complete
            local timeout=3600
            local waited=0
            while kill -0 $pid 2>/dev/null; do
                if grep -q "encode stage complete" "$log_file" 2>/dev/null; then
                    kill $pid 2>/dev/null || true
                    wait $pid 2>/dev/null || true
                    break
                fi
                sleep 1
                waited=$((waited + 1))
                if [[ $waited -ge $timeout ]]; then
                    echo "TIMEOUT after ${timeout}s"
                    kill $pid 2>/dev/null || true
                    wait $pid 2>/dev/null || true
                    return 1
                fi
            done
        else
            # Run full pipeline
            "$BIN" --source "$CONFIG" --output "$output_dir" --dest local --parallel $extra_args \
                > "$log_file" 2>&1 || true
        fi

        # Extract encode time from log
        local encode_line
        encode_line=$(grep "encode stage complete" "$log_file" 2>/dev/null || echo "")
        if [[ -z "$encode_line" ]]; then
            echo "FAILED (no encode stage complete)"
            return 1
        fi

        # Parse elapsed time (formats: "1m 23s", "45.2s", "1h 2m 3s")
        local elapsed_str
        elapsed_str=$(echo "$encode_line" | sed 's/.*elapsed=//')
        local total_seconds=0

        # Extract hours, minutes, seconds
        local hours=0 minutes=0 seconds=0
        if [[ "$elapsed_str" =~ ([0-9]+)h ]]; then hours=${BASH_REMATCH[1]}; fi
        if [[ "$elapsed_str" =~ ([0-9]+)m ]]; then minutes=${BASH_REMATCH[1]}; fi
        if [[ "$elapsed_str" =~ ([0-9.]+)s ]]; then seconds=${BASH_REMATCH[1]}; fi
        total_seconds=$(echo "$hours * 3600 + $minutes * 60 + $seconds" | bc)

        echo "${total_seconds}s (${elapsed_str})"
        echo "$total_seconds"

        # Cleanup output to save disk
        rm -rf "$output_dir"
    }

    for iter in $(seq 1 "$ITERATIONS"); do
        echo ""
        echo "--- Iteration $iter/$ITERATIONS ---"

        # Baseline (no col stats)
        result=$(run_ingest "baseline_${iter}" "" | tail -1)
        if [[ -n "$result" && "$result" != "FAILED"* ]]; then
            baseline_times+=("$result")
        fi

        # With col stats
        result=$(run_ingest "colstats_${iter}" "--col-stats" | tail -1)
        if [[ -n "$result" && "$result" != "FAILED"* ]]; then
            colstats_times+=("$result")
        fi
    done

    # Compute averages
    compute_avg() {
        local -n arr=$1
        if [[ ${#arr[@]} -eq 0 ]]; then echo "0"; return; fi
        local sum=0
        for v in "${arr[@]}"; do
            sum=$(echo "$sum + $v" | bc)
        done
        echo "scale=2; $sum / ${#arr[@]}" | bc
    }

    baseline_avg=$(compute_avg baseline_times)
    colstats_avg=$(compute_avg colstats_times)

    if [[ "$baseline_avg" != "0" && "$baseline_avg" != "" ]]; then
        overhead=$(echo "scale=2; ($colstats_avg - $baseline_avg) / $baseline_avg * 100" | bc 2>/dev/null || echo "?")
    else
        overhead="?"
    fi

    echo ""
    echo "--- Results for $SIZE ---"
    echo "  Baseline  encode avg: ${baseline_avg}s  (${baseline_times[*]:-none})"
    echo "  ColStats  encode avg: ${colstats_avg}s  (${colstats_times[*]:-none})"
    echo "  Overhead: ${overhead}%"

    # Write JSON
    echo '      "baseline_times": ['$(IFS=,; echo "${baseline_times[*]:-}")'],' >> "$RESULTS_FILE"
    echo '      "colstats_times": ['$(IFS=,; echo "${colstats_times[*]:-}")'],' >> "$RESULTS_FILE"
    echo '      "baseline_avg": '$baseline_avg',' >> "$RESULTS_FILE"
    echo '      "colstats_avg": '$colstats_avg',' >> "$RESULTS_FILE"
    echo '      "overhead_pct": "'$overhead'"' >> "$RESULTS_FILE"
    echo '    }' >> "$RESULTS_FILE"
done

echo '  }' >> "$RESULTS_FILE"
echo '}' >> "$RESULTS_FILE"

echo ""
echo "Results saved to: $RESULTS_FILE"
