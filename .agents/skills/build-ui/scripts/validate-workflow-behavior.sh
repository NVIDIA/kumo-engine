#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CHECKS_FILE="$ROOT/.agents/skills/build-ui/references/behavioral-checks.tsv"

fail() {
  echo "$1" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$ROOT/$path" ]] || fail "MISSING FILE: $path"
}

require_text() {
  local path="$1"
  local pattern="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -F -q -- "$pattern" "$ROOT/$path" && return 0
  elif command -v grep >/dev/null 2>&1; then
    grep -F -q -- "$pattern" "$ROOT/$path" && return 0
  else
    fail "validate-workflow-behavior.sh requires rg or grep on PATH."
  fi

  fail "MISSING BEHAVIOR: $path :: $pattern"
}

main() {
  require_file ".agents/skills/build-ui/SKILL.md"
  require_file ".agents/skills/build-ui/references/decision-tables.md"
  require_file ".agents/skills/build-ui/references/headless-mode.md"
  require_file ".agents/skills/build-ui/references/workflow-test-scenarios.md"
  [[ -f "$CHECKS_FILE" ]] || fail "MISSING FILE: .agents/skills/build-ui/references/behavioral-checks.tsv"

  local line_no=0
  local checked=0
  local scenario path pattern

  while IFS=$'\t' read -r scenario path pattern; do
    line_no=$((line_no + 1))

    if [[ $line_no -eq 1 ]]; then
      [[ "$scenario"$'\t'"$path"$'\t'"$pattern" == "scenario"$'\t'"path"$'\t'"pattern" ]] || fail "INVALID HEADER: .agents/skills/build-ui/references/behavioral-checks.tsv"
      continue
    fi

    [[ -n "$scenario" && -n "$path" && -n "$pattern" ]] || fail "INVALID ROW: .agents/skills/build-ui/references/behavioral-checks.tsv:$line_no"

    require_file "$path"
    require_text "$path" "$pattern"
    checked=$((checked + 1))
  done < "$CHECKS_FILE"

  [[ $checked -gt 0 ]] || fail "NO BEHAVIORAL CHECKS: .agents/skills/build-ui/references/behavioral-checks.tsv"

  echo "Workflow behavior validation passed ($checked checks)."
}

main "$@"
