#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Run storybook tests with a11y error mode and generate report
echo "Running accessibility tests..."

# Temporarily set a11y to error mode via environment or run tests
npm run test:storybook -- --run 2>&1 | tee /tmp/a11y-report.txt

echo ""
echo "========================================"
echo "     ACCESSIBILITY SCORE REPORT"
echo "========================================"

# Extract pass/fail counts
passed=$(grep -E "Tests.*passed" /tmp/a11y-report.txt | grep -oE "[0-9]+ passed" | head -1 | grep -oE "[0-9]+")
failed=$(grep -E "Tests.*failed" /tmp/a11y-report.txt | grep -oE "[0-9]+ failed" | head -1 | grep -oE "[0-9]+")

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
echo "Tests Failed:  $failed (a11y violations)"
echo "Total Tests:   $total"
echo "----------------------------------------"
echo "A11y Score:    ${score}%"
echo ""

echo "========================================"
echo "     VIOLATIONS BY TYPE"
echo "========================================"
echo ""

grep -oE '\([a-z-]+\)' /tmp/a11y-report.txt | grep -v chromium | grep -v wrapped | sort | uniq -c | sort -rn | head -10

echo ""
echo "========================================"
echo "For detailed fixes, see the test output above"
echo "or run: npm run storybook"
echo "========================================"
