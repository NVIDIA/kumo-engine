#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# diff-scan.sh — Pre-scan a GitHub PR diff for [HIGH] violations plus the
# zero-judgment [MEDIUM] migration rules (design-token, arbitrary-tailwind).
#
# Parses the unified diff to extract added lines with correct source
# file paths and line numbers (single awk pass), then matches against
# known violation patterns. Supports inline suppression via
# "review-ui-ignore: <rule-id>" comments.
#
# Usage:
#   diff-scan.sh <pr_number>
#   diff-scan.sh --file <diff_path>
#
# Output (TSV to stdout, one finding per line):
#   rule_id	file_path	line_number	matched_text
#
# Summary (to stderr):
#   # diff-scan: N findings in M files
#
# Exit codes:
#   0 — scan completed (findings may or may not exist)
#   1 — usage error or gh CLI failure

usage() {
  echo "Usage: $0 <pr_number>" >&2
  echo "       $0 --file <diff_path>" >&2
  exit 1
}

DIFF_FILE=""
PR_NUMBER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file)
      DIFF_FILE="${2:-}"
      [[ -z "$DIFF_FILE" ]] && usage
      shift 2
      ;;
    -h|--help) usage ;;
    *)
      PR_NUMBER="$1"
      shift
      ;;
  esac
done

[[ -z "$DIFF_FILE" && -z "$PR_NUMBER" ]] && usage

# Use mktemp to avoid collisions when running concurrent reviews
TMPFILE=$(mktemp "${TMPDIR:-/tmp}/review-ui-XXXXXX.patch")
trap 'rm -f "$TMPFILE"' EXIT

if [[ -n "$DIFF_FILE" ]]; then
  cp "$DIFF_FILE" "$TMPFILE"
else
  if ! gh pr diff "$PR_NUMBER" > "$TMPFILE" 2>/dev/null; then
    echo "error: failed to fetch diff for PR #${PR_NUMBER}" >&2
    exit 1
  fi
fi

# Single-pass awk script:
#   1. Parses diff structure (file paths from "diff --git", line numbers from @@ hunks)
#   2. Filters to ui/src/ paths, excluding gen/ and PQLParsing/
#   3. Extracts inline suppression comments (review-ui-ignore: rule-id[, rule-id])
#   4. Skips known exception files per-pattern
#   5. Detects commented-out code (avoids console.log false positives)
#   6. Matches violation patterns on added lines only
#   7. Outputs: rule_id\tfile_path\tline_number\tmatched_text
#   8. Prints summary to stderr
#
# Note: \047 is octal for single quote (avoids shell quoting issues in awk)

awk '
function emit(rule, f, l, content) {
  printf "%s\t%s\t%d\t%s\n", rule, f, l, content
  findings++
  if (!(f in seen_files)) { file_count++; seen_files[f] = 1 }
}

# Check if a keyword appears inside a single-line comment (// ...)
function in_comment(content, keyword,    ki, si) {
  # Line is entirely a comment
  if (content ~ /^[[:space:]]*\/\//) return 1
  # Keyword appears after a // comment marker
  ki = index(content, keyword)
  si = index(content, "//")
  if (si > 0 && si < ki) return 1
  return 0
}

# [MEDIUM] arbitrary-tailwind, per-token. Returns 1 if the line contains a
# *hardcoded* arbitrary Tailwind value (px or hex #) that is not a dynamic
# var(--*) value. Each -[...] token is checked independently so a line mixing
# an allowed dynamic value with a hardcoded one — e.g. w-[var(--x)] h-[44px] —
# still flags h-[44px]. A line-level "skip if the line contains var(--" would
# let the hardcoded token escape. Viewport units (vh/vw/vmin/vmax) carry no
# px/# so they never match and need no special case.
function has_hardcoded_arbitrary(content,    s, rest, ci, val) {
  s = content
  while (match(s, /-\[/)) {
    rest = substr(s, RSTART + RLENGTH)   # text after this "-["
    ci = index(rest, "]")                # arbitrary values contain no nested "]"
    if (ci == 0) break
    val = substr(rest, 1, ci - 1)        # token contents between [ ]
    if ((val ~ /px/ || val ~ /#/) && val !~ /var\(--/) return 1
    s = substr(rest, ci + 1)             # continue scanning after this token
  }
  return 0
}

/^diff --git/ {
  file = ""
  for (i = 1; i <= NF; i++) {
    if (substr($i, 1, 2) == "b/") {
      file = substr($i, 3)
      break
    }
  }
  line = 0
  next
}

/^@@ / {
  for (i = 1; i <= NF; i++) {
    if (substr($i, 1, 1) == "+") {
      split(substr($i, 2), p, ",")
      line = p[1]
      break
    }
  }
  next
}

{
  if (line == 0 || file == "") next

  ch = substr($0, 1, 1)

  if (ch == "+") {
    c = substr($0, 2)

    # ── Scope filter: ui/src/ only, exclude generated code ──
    if (file !~ /^ui\/src\//)                   { line++; next }
    if (file ~ /^ui\/src\/gen\//)               { line++; next }
    if (file ~ /^ui\/src\/utils\/PQLParsing\//) { line++; next }

    # 1 = test file, 0 = production file
    t = (file ~ /__tests__|\.test\.|\.spec\./) ? 1 : 0

    # ── Inline suppression: extract rule IDs after "review-ui-ignore:" ──
    suppress = ""
    if (c ~ /review-ui-ignore:/) {
      si = index(c, "review-ui-ignore:")
      suppress = substr(c, si + 17)
    }

    # ══════════════════════════════════════════════════════════════
    # Pattern checks — each includes:
    #   - Scope guard (production vs test)
    #   - Exception file exclusion
    #   - Inline suppression check
    # ══════════════════════════════════════════════════════════════

    # ── [HIGH] console.log in production code ──
    # Exception: utils/errorHandler.ts (implements the logging utility)
    # Skip: commented-out console.log
    if (!t && c ~ /console\.log/ \
        && file !~ /utils\/errorHandler\.ts$/ \
        && suppress !~ /console-log/ \
        && !in_comment(c, "console.log")) {
      emit("console-log", file, line, c)
    }

    # ── [HIGH] Deep relative imports (3+ levels) ──
    if (c ~ /from ["\047]\.\.\/\.\.\/\.\./ \
        && suppress !~ /deep-import/) {
      emit("deep-import", file, line, c)
    }

    # ── [HIGH] dangerouslySetInnerHTML (XSS) ──
    if (c ~ /dangerouslySetInnerHTML/ \
        && suppress !~ /xss/) {
      emit("xss", file, line, c)
    }

    # ── [HIGH] Direct localStorage/sessionStorage access ──
    # Exception: utils/storage.ts (IS the storage abstraction)
    if (c ~ /localStorage\.[gs]et|localStorage\.remove|sessionStorage\.[gs]et|sessionStorage\.remove/ \
        && file !~ /utils\/storage\.ts$/ \
        && suppress !~ /direct-storage/) {
      emit("direct-storage", file, line, c)
    }

    # ── [HIGH] Direct axios import (not from services/) ──
    # Exception: services/ directory (axios service layer)
    # Exception: hooks/tables/ (file uploads with progress tracking)
    if (c ~ /from ["\047]axios["\047]/ \
        && file !~ /\/services\// \
        && file !~ /hooks\/tables\// \
        && suppress !~ /axios-bypass/) {
      emit("axios-bypass", file, line, c)
    }

    # ── [HIGH] process.env.REACT_APP in production code ──
    # Exception: services/axios_client.ts (getBaseURL uses process.env.NODE_ENV)
    if (!t && c ~ /process\.env\.REACT_APP/ \
        && file !~ /services\/axios_client\.ts$/ \
        && suppress !~ /process-env/) {
      emit("process-env", file, line, c)
    }

    # ── [HIGH] Hardcoded route in navigate() — quotes ──
    if ((c ~ /navigate\(["\047]\//) \
        && suppress !~ /hardcoded-route/) {
      emit("hardcoded-route", file, line, c)
    }

    # ── [HIGH] Hardcoded route in navigate() — template literals ──
    if (c ~ /navigate\(`\// \
        && suppress !~ /hardcoded-route/) {
      emit("hardcoded-route-tpl", file, line, c)
    }

    # ── [HIGH] Hardcoded route in <Link to=> ──
    if (c ~ /<Link[[:space:]].*to=["\047]\//) {
      if (suppress !~ /hardcoded-route/) {
        emit("hardcoded-route", file, line, c)
      }
    }

    # ── [HIGH] new QueryClient() in test files ──
    if (t && c ~ /new QueryClient/ \
        && suppress !~ /raw-queryclient/) {
      emit("raw-queryclient", file, line, c)
    }

    # ── [HIGH] vi.mock on stores ──
    if (c ~ /vi\.mock.*stores\// \
        && suppress !~ /mock-store/) {
      emit("mock-store", file, line, c)
    }

    # ── [HIGH] vi.mock on axios ──
    if (c ~ /vi\.mock.*axios/ \
        && suppress !~ /mock-axios/) {
      emit("mock-axios", file, line, c)
    }

    # ── [HIGH] vi.mock on react-router ──
    if (c ~ /vi\.mock.*react-router/ \
        && suppress !~ /mock-router/) {
      emit("mock-router", file, line, c)
    }

    # ── [HIGH] Third-party form libraries ──
    if ((c ~ /from ["\047]react-hook-form["\047]/ \
         || c ~ /from ["\047]formik["\047]/ \
         || c ~ /from ["\047]yup["\047]/) \
        && suppress !~ /form-library/) {
      emit("form-library", file, line, c)
    }

    # ── [HIGH] Hardcoded query keys (string literal in queryKey array) ──
    if (c ~ /queryKey.*\[["\047]/ \
        && suppress !~ /hardcoded-querykey/) {
      emit("hardcoded-querykey", file, line, c)
    }

    # ── [MEDIUM] Default neutral Tailwind palette where a Kumo token exists ──
    # Production code only: test files legitimately assert legacy class names.
    # Families: slate/gray/zinc/neutral/stone. Documented exceptions
    # (tooltip-dark, semantic error/warning reds) use review-ui-ignore.
    if (!t && c ~ /(text|bg|border|ring|fill|stroke|divide|placeholder|from|via|to|outline|accent|caret|decoration)-(slate|gray|zinc|neutral|stone)-[0-9]/ \
        && suppress !~ /design-token/) {
      emit("design-token", file, line, c)
    }

    # ── [MEDIUM] Arbitrary Tailwind values: hardcoded pixels or hex colors ──
    # Allowed (not flagged): dynamic var(--*) and viewport units (vh/vw/vmin/vmax).
    # Production code only. Per-token scan (see has_hardcoded_arbitrary) so a
    # line mixing an allowed dynamic value with a hardcoded one still flags the
    # hardcoded token instead of escaping on the line-wide var(--) match.
    if (!t && suppress !~ /arbitrary-tailwind/ \
        && has_hardcoded_arbitrary(c)) {
      emit("arbitrary-tailwind", file, line, c)
    }

    line++
  } else if (ch == " ") {
    line++
  }
  # "-" lines (deletions): do not increment line counter
}

END {
  printf "# diff-scan: %d findings in %d files\n", findings, file_count > "/dev/stderr"
}
' "$TMPFILE"
