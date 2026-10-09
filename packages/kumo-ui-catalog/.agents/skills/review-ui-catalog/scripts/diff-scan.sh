#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pre-scan PR diff for mechanical violations in kumo-ui-catalog.
#
# Usage: bash .agents/skills/review-ui-catalog/scripts/diff-scan.sh <pr_number>
#
# Output: TSV to stdout: rule_id\tfile_path\tline_number\tmatched_text
# Summary to stderr: # diff-scan: N findings in M files

set -euo pipefail

PR_NUMBER="${1:?Usage: diff-scan.sh <pr_number>}"

# gh pr diff has no path argument; filter its repository-root paths below.
# A fetch error must fail the scan instead of looking like an empty diff.
DIFF=$(gh pr diff "$PR_NUMBER")

if [[ -z "$DIFF" ]]; then
  echo "# diff-scan: 0 findings in 0 files" >&2
  exit 0
fi

FINDINGS=0
FILES_WITH_FINDINGS=()

# Parse diff: extract file path and new lines (+ lines) with line numbers
scan_diff() {
  local current_file=""
  local catalog_file=""
  local new_line_num=0

  while IFS= read -r line; do
    # Added lines belong to the destination path, including for renames.
    if [[ "$line" =~ ^diff\ --git\ a/(.+)\ b/(.+)$ ]]; then
      current_file="${BASH_REMATCH[2]}"
      case "$current_file" in
        packages/kumo-ui-catalog/src/components/*|packages/kumo-ui-catalog/src/stories/*|packages/kumo-ui-catalog/src/animations/*|packages/kumo-ui-catalog/src/hooks/*|packages/kumo-ui-catalog/src/lib/*|packages/kumo-ui-catalog/src/utils/*|packages/kumo-ui-catalog/src/types/*|packages/kumo-ui-catalog/src/icons/*|packages/kumo-ui-catalog/src/index.ts|packages/kumo-ui-catalog/src/colors.css)
          catalog_file="${current_file#packages/kumo-ui-catalog/}" ;;
        *) current_file=""; catalog_file="" ;;
      esac
      continue
    fi

    [[ -n "$current_file" ]] || continue

    # Track line numbers from hunk headers
    if [[ "$line" =~ ^@@.*\+([0-9]+) ]]; then
      new_line_num="${BASH_REMATCH[1]}"
      continue
    fi

    # Only scan added lines
    if [[ "$line" =~ ^\+ ]] && [[ ! "$line" =~ ^\+\+\+ ]]; then
      local content="${line:1}" # strip leading +

      # Skip suppressed lines
      if [[ "$content" == *"review-ui-catalog-ignore:"* ]]; then
        new_line_num=$((new_line_num + 1))
        continue
      fi

      # Only scan component source files (not stories/tests)
      if [[ "$catalog_file" == src/components/* ]]; then
        run_component_rules "$current_file" "$new_line_num" "$content"
      fi

      # Scan all source files for general rules
      run_general_rules "$current_file" "$new_line_num" "$content"

      # Scan story files
      if [[ "$catalog_file" == src/stories/* ]]; then
        run_story_rules "$current_file" "$new_line_num" "$content"
      fi

      new_line_num=$((new_line_num + 1))
      continue
    fi

    # Unchanged lines still increment line counter
    if [[ ! "$line" =~ ^- ]]; then
      new_line_num=$((new_line_num + 1))
    fi
  done <<< "$DIFF"
}

emit() {
  local rule="$1" file="$2" line_num="$3" matched="$4"
  printf '%s\t%s\t%s\t%s\n' "$rule" "$file" "$line_num" "$matched"
  FINDINGS=$((FINDINGS + 1))
  # Track unique files
  local found=false
  for f in "${FILES_WITH_FINDINGS[@]+"${FILES_WITH_FINDINGS[@]}"}"; do
    [[ "$f" == "$file" ]] && found=true && break
  done
  $found || FILES_WITH_FINDINGS+=("$file")
}

run_component_rules() {
  local file="$1" line_num="$2" content="$3"

  # inline-styles: style={{ in components
  if [[ "$content" =~ style=\{\{ ]] && [[ "$content" != *"review-ui-catalog-ignore"* ]]; then
    emit "inline-styles" "$file" "$line_num" "$content"
  fi

  # hardcoded-colors: hex colors (#xxx or #xxxxxx) in component code
  if echo "$content" | grep -qE '#[0-9a-fA-F]{3,8}[^a-zA-Z]' 2>/dev/null; then
    # Exclude comments and CSS variable definitions
    if [[ "$content" != *"//"* ]] && [[ "$content" != *"--"* ]]; then
      emit "hardcoded-colors" "$file" "$line_num" "$content"
    fi
  fi

  # cn-usage: string template for className (backtick with className)
  if [[ "$content" =~ className=\` ]] || [[ "$content" =~ className=\{\" ]]; then
    emit "cn-usage" "$file" "$line_num" "$content"
  fi

  # barrel-self-import: importing from the barrel inside source
  if [[ "$content" =~ from\ \"@kumo-ai/kumo-ui-catalog\" ]] || \
     [[ "$content" =~ from\ \"\.\.\/index\" ]] || \
     [[ "$content" =~ from\ \"\.\.\/\.\.\/index\" ]]; then
    emit "barrel-self-import" "$file" "$line_num" "$content"
  fi

  # focus-states: focus: without focus-visible:
  if [[ "$content" =~ \".*focus:.*\" ]] && [[ "$content" != *"focus-visible:"* ]] && [[ "$content" != *"focus-within:"* ]]; then
    emit "focus-states" "$file" "$line_num" "$content"
  fi
}

run_general_rules() {
  local file="$1" line_num="$2" content="$3"

  # any-type: explicit any without justification
  if [[ "$content" =~ :[[:space:]]*any[^a-zA-Z] ]] || [[ "$content" =~ \<any\> ]] || [[ "$content" =~ as\ any ]]; then
    # Skip if there's an eslint-disable or ts-expect-error on same line
    if [[ "$content" != *"eslint-disable"* ]] && [[ "$content" != *"ts-expect-error"* ]]; then
      emit "any-type" "$file" "$line_num" "$content"
    fi
  fi

  # strict-compliance: ts-ignore without comment
  if [[ "$content" =~ @ts-ignore ]] && [[ "$content" != *"//"*"@ts-ignore"*"—"* ]] && [[ "$content" != *"//"*"@ts-ignore"*"-"* ]]; then
    emit "strict-compliance" "$file" "$line_num" "$content"
  fi
}

run_story_rules() {
  local file="$1" line_num="$2" content="$3"

  # story-tags: Meta without autodocs (check for tags line without autodocs)
  # Only flag if we see a tags array that doesn't include autodocs
  if [[ "$content" =~ tags:.*\[.*\] ]] && [[ "$content" != *"autodocs"* ]]; then
    emit "story-tags" "$file" "$line_num" "$content"
  fi
}

scan_diff

echo "# diff-scan: ${FINDINGS} findings in ${#FILES_WITH_FINDINGS[@]} files" >&2
