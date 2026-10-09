#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Run regression tests for diff-scan.sh
#
# Usage: bash run-tests.sh
#
# Compares actual output of diff-scan.sh against expected.tsv.
# Exits 0 on match, 1 on mismatch.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCAN="${SCRIPT_DIR}/../diff-scan.sh"
SAMPLE="${SCRIPT_DIR}/sample.patch"
EXPECTED="${SCRIPT_DIR}/expected.tsv"

if [[ ! -f "$SCAN" ]]; then
  echo "FAIL: diff-scan.sh not found at $SCAN" >&2
  exit 1
fi

if [[ ! -f "$SAMPLE" ]]; then
  echo "FAIL: sample.patch not found at $SAMPLE" >&2
  exit 1
fi

if [[ ! -f "$EXPECTED" ]]; then
  echo "FAIL: expected.tsv not found at $EXPECTED" >&2
  exit 1
fi

# Run scanner, capture stdout (findings) and stderr (summary)
ACTUAL=$(bash "$SCAN" --file "$SAMPLE" 2>/dev/null)
SUMMARY=$(bash "$SCAN" --file "$SAMPLE" 2>&1 >/dev/null)

# Compare findings
if diff <(echo "$ACTUAL") "$EXPECTED" > /dev/null 2>&1; then
  ACTUAL_COUNT=$(echo "$ACTUAL" | wc -l)
  ACTUAL_FILES=$(echo "$ACTUAL" | cut -f2 | sort -u | wc -l)
  echo "PASS: ${ACTUAL_COUNT} findings in ${ACTUAL_FILES} files — matches expected output"

  # Verify summary matches
  EXPECTED_SUMMARY="# diff-scan: ${ACTUAL_COUNT} findings in ${ACTUAL_FILES} files"
  if [[ "$SUMMARY" == "$EXPECTED_SUMMARY" ]]; then
    echo "PASS: stderr summary matches"
  else
    echo "WARN: stderr summary mismatch"
    echo "  expected: $EXPECTED_SUMMARY"
    echo "  actual:   $SUMMARY"
  fi

  exit 0
else
  echo "FAIL: Output differs from expected"
  echo ""
  echo "--- Expected ---"
  cat "$EXPECTED"
  echo ""
  echo "--- Actual ---"
  echo "$ACTUAL"
  echo ""
  echo "--- Diff (< expected, > actual) ---"
  diff "$EXPECTED" <(echo "$ACTUAL") || true
  exit 1
fi
