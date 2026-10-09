#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Scan for sensitive values with the repository's Gitleaks configuration.
#
# - No argument: scan the current tree before a release.
# - --diff BASE HEAD: scan the commits reachable from HEAD but not from BASE.
#   CI uses this for each pull request and push.
# - --history / --history-report: the one-time full-history baseline scan; the
#   report option writes a redacted artifact that can be shared for review.
#
# GITLEAKS_BIN selects a Gitleaks executable (CI downloads a pinned release);
# otherwise the version pinned in .mise.toml runs through mise.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

scan_mode="dir"
report_dir=""
diff_base=""
diff_head=""

usage() {
    echo "Usage: $0 [--diff BASE HEAD | --history |" \
        "--history-report ABSOLUTE_DIRECTORY]" >&2
}

run_gitleaks() {
    if [[ -n "${GITLEAKS_BIN:-}" ]]; then
        "$GITLEAKS_BIN" "$@"
    else
        mise exec -- gitleaks "$@"
    fi
}

resolve_commit() {
    local reference="$1" label="$2"
    if ! git rev-parse --verify --quiet --end-of-options \
        "${reference}^{commit}"; then
        echo "Gitleaks diff $label is not an available commit: $reference" >&2
        exit 2
    fi
}

case "${1:-}" in
    "") ;;
    --diff)
        if (( $# != 3 )) || [[ -z "$2" || -z "$3" ]]; then
            echo "--diff requires non-empty BASE and HEAD commits." >&2
            exit 2
        fi
        scan_mode="git"
        diff_base="$2"
        diff_head="$3"
        shift 3
        ;;
    --history)
        scan_mode="git"
        shift
        ;;
    --history-report)
        if (( $# != 2 )) || [[ "$2" != /* ]]; then
            echo "--history-report requires an absolute output directory." >&2
            exit 2
        fi
        scan_mode="git"
        report_dir="$2"
        shift 2
        ;;
    *)
        usage
        exit 2
        ;;
esac
if (( $# != 0 )); then
    usage
    exit 2
fi

cd "$REPO_ROOT"
gitleaks_args=(
    "$scan_mode"
    --config "$REPO_ROOT/.gitleaks.toml"
    --no-banner
    --redact=100
)
if [[ -n "$diff_head" ]]; then
    head_commit="$(resolve_commit "$diff_head" head)"
    if [[ "$diff_base" =~ ^0+$ ]]; then
        # A push that creates a branch has no previous commit; scan only HEAD.
        gitleaks_args+=("--log-opts=${head_commit}^!")
    else
        base_commit="$(resolve_commit "$diff_base" base)"
        gitleaks_args+=("--log-opts=${base_commit}..${head_commit}")
    fi
elif [[ "$scan_mode" == "git" ]]; then
    if [[ "$(git rev-parse --is-shallow-repository)" != "false" ]]; then
        echo "Full-history scans require a non-shallow Git checkout." >&2
        exit 2
    fi
    gitleaks_args+=(--log-opts=--all)
fi

if [[ -z "$report_dir" ]]; then
    run_gitleaks "${gitleaks_args[@]}" .
    exit
fi

umask 077
mkdir -p "$report_dir"
raw_report="$(mktemp "${TMPDIR:-/tmp}/kumo-gitleaks-report.XXXXXX")"
cleanup() {
    rm -f -- "$raw_report"
}
trap cleanup EXIT

set +e
run_gitleaks "${gitleaks_args[@]}" \
    --report-format json \
    --report-path "$raw_report" \
    .
scan_status=$?
set -e

if [[ -s "$raw_report" ]]; then
    mise exec -- python3 "$SCRIPT_DIR/summarize_gitleaks_report.py" "$raw_report" \
        --json-output "$report_dir/gitleaks-history-sanitized.json" \
        --markdown-output "$report_dir/gitleaks-history.md"
    echo "Sanitized report: $report_dir/gitleaks-history.md"
elif (( scan_status != 0 )); then
    echo "Gitleaks failed before producing a report." >&2
fi

exit "$scan_status"
