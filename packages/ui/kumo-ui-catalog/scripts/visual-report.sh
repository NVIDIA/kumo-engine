#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Run visual regression tests and generate console report
echo "Running visual regression tests..."

npm run test:visual 2>&1 | tee /tmp/visual-report.txt

echo ""
echo "========================================"
echo "   VISUAL REGRESSION SCORE REPORT"
echo "========================================"

# Extract pass/fail counts
passed=$(grep -E "Tests.*passed" /tmp/visual-report.txt | grep -oE "[0-9]+ passed" | head -1 | grep -oE "[0-9]+")
failed=$(grep -E "Tests.*failed" /tmp/visual-report.txt | grep -oE "[0-9]+ failed" | head -1 | grep -oE "[0-9]+")

if [ -z "$passed" ]; then passed=0; fi
if [ -z "$failed" ]; then failed=0; fi

total=$((passed + failed))

if [ $total -gt 0 ]; then
    score=$(awk "BEGIN {printf \"%.1f\", ($passed / $total) * 100}")
else
    score="N/A"
fi

echo ""
echo "Tests Passed:  $passed"
echo "Tests Failed:  $failed (visual mismatches)"
echo "Total Tests:   $total"
echo "----------------------------------------"
echo "Pass Rate:     ${score}%"
echo ""

echo "========================================"
echo "     FAILED TESTS"
echo "========================================"
echo ""

grep -E "FAIL|✗|×|❌" /tmp/visual-report.txt | grep -v "Tests.*failed" | head -20

echo ""
echo "========================================"
echo "For detailed diffs, run: npm run test:visual:html"
echo "========================================"
