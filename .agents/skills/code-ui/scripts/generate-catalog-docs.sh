#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Auto-generate catalog-components.md from kumo-ui-catalog's index.ts exports.
#
# Usage: bash .agents/skills/code-ui/scripts/generate-catalog-docs.sh [catalog_path]
#
# If catalog_path is not provided, uses the in-tree catalog.
# Pass --check to compare without writing the committed output.
# Outputs to .agents/skills/code-ui/references/catalog-components.md

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
EXPECTED_OUTPUT="${REPO_ROOT}/.agents/skills/code-ui/references/catalog-components.md"
TMPDIR_WORK=$(mktemp -d)
trap 'rm -rf "$TMPDIR_WORK"' EXIT

OUTPUT="${TMPDIR_WORK}/catalog-components.md"
CHECK=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK=1
  shift
fi
CATALOG_PATH="${1:-${REPO_ROOT}/packages/kumo-ui-catalog}"

if [[ -z "$CATALOG_PATH" || ! -f "${CATALOG_PATH}/package.json" ]]; then
  echo "ERROR: Cannot find kumo-ui-catalog. Provide path as argument." >&2
  exit 1
fi

CATALOG_PATH="$(cd "$CATALOG_PATH" && pwd)"
VERSION=$(grep '"version"' "${CATALOG_PATH}/package.json" | head -1 | sed 's/.*: *"\(.*\)".*/\1/')

# Determine index file
INDEX_FILE=""
for candidate in "${CATALOG_PATH}/src/index.ts" "${CATALOG_PATH}/dist/index.d.ts"; do
  if [[ -f "$candidate" ]]; then
    INDEX_FILE="$candidate"
    break
  fi
done

if [[ -z "$INDEX_FILE" ]]; then
  echo "ERROR: Cannot find index.ts or index.d.ts in catalog." >&2
  exit 1
fi

echo "Generating catalog docs from: ${INDEX_FILE} (v${VERSION})"

# Classify a source path into a category
classify_path() {
  local path="$1"
  if [[ "$path" == *"/ui/"* ]]; then echo "01_UI-Primitives"
  elif [[ "$path" == *"/blocks/chart"* ]]; then echo "02_Charts-(Block)"
  elif [[ "$path" == *"/blocks/data-table"* ]]; then echo "03_Data-Table-(Block)"
  elif [[ "$path" == *"/blocks/layout"* ]]; then echo "04_Layout-(Block)"
  elif [[ "$path" == *"/blocks/sidebar"* ]]; then echo "05_Sidebar-(Block)"
  elif [[ "$path" == *"/blocks/visual-graph"* ]]; then echo "06_Graph-Visualization-(Block)"
  elif [[ "$path" == *"/animations/"* ]]; then echo "07_Loading-&-Animation"
  elif [[ "$path" == *"/icons/"* ]]; then echo "08_Icons"
  elif [[ "$path" == *"sonner"* || "$path" == *"toast"* ]]; then echo "09_Toast"
  elif [[ "$path" == *"/lib/"* || "$path" == *"/utils"* ]]; then echo "10_Utility"
  elif [[ "$path" == *"@xyflow"* || "$path" == *"react-flow"* ]]; then echo "11_React-Flow-(re-export)"
  elif [[ "$path" == *"@tanstack"* ]]; then echo "12_TanStack-Table-(re-export)"
  else echo "13_Other"
  fi
}

# Pre-process: join multi-line exports into single lines
# e.g., "export {\n  Foo,\n  Bar\n} from ..." becomes one line
FLAT_INDEX="${TMPDIR_WORK}/index_flat.ts"
tr '\n' ' ' < "$INDEX_FILE" | sed 's/export/\nexport/g' | grep '^export' > "$FLAT_INDEX"

# Parse exports into categorized temp files
n_components=0
n_types=0

while IFS= read -r line; do
  [[ "$line" =~ ^[[:space:]]*$ ]] && continue

  if [[ "$line" =~ ^export ]]; then
    # Get source path
    source_path=""
    if [[ "$line" =~ from[[:space:]]*\"(.+)\" ]]; then
      source_path="${BASH_REMATCH[1]}"
    fi

    # Extract names between { and }
    if [[ "$line" =~ \{(.+)\} ]]; then
      names_str="${BASH_REMATCH[1]}"
    else
      continue
    fi

    is_type_export=false
    [[ "$line" =~ ^export[[:space:]]+type ]] && is_type_export=true

    IFS=',' read -ra name_arr <<< "$names_str"
    for name in "${name_arr[@]}"; do
      name=$(echo "$name" | xargs)
      [[ -z "$name" ]] && continue
      [[ "$name" == "type" ]] && continue

      if $is_type_export; then
        # Type export
        n_types=$((n_types + 1))
        cat_key="type"
        if [[ "$source_path" == *"@xyflow"* ]]; then cat_key="type_reactflow"
        elif [[ "$source_path" == *"@tanstack"* ]]; then cat_key="type_tanstack"
        fi
        echo "$name" >> "${TMPDIR_WORK}/${cat_key}.txt"
      else
        # Component or function export
        n_components=$((n_components + 1))
        cat=$(classify_path "$source_path")
        echo "$name" >> "${TMPDIR_WORK}/${cat}.txt"
      fi
    done
  fi
done < "$FLAT_INDEX"

# Generate markdown header
cat > "$OUTPUT" << 'HEADER'
# UI Catalog Component Reference

> **Auto-generated** — run `bash .agents/skills/code-ui/scripts/generate-catalog-docs.sh` to refresh.
> Source: `@kumo-ai/kumo-ui-catalog` — import everything from this single entry point.
> Source directory: `packages/kumo-ui-catalog` in this repository.

## Import Pattern

```tsx
import { Button, Card, DataTable, cn } from "@kumo-ai/kumo-ui-catalog";
```

HEADER

echo "**Version:** ${VERSION} | **Explicit named exports:** ${n_components} components/functions, ${n_types} types" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# Output component groups in order
echo "## Components by Category" >> "$OUTPUT"
echo "" >> "$OUTPUT"

for catfile in $(ls "${TMPDIR_WORK}"/[0-9]*.txt 2>/dev/null | LC_ALL=C sort); do
  filename=$(basename "$catfile" .txt)
  # Strip sort prefix and convert dashes to spaces (e.g., "01_UI-Primitives" -> "UI Primitives")
  cat_name="${filename#*_}"
  cat_name="${cat_name//-/ }"

  names=$(LC_ALL=C sort "$catfile" | sed 's/.*/ `&`/' | paste -sd',' | sed 's/,/, /g; s/^ //')
  echo "### ${cat_name}" >> "$OUTPUT"
  echo "${names}" >> "$OUTPUT"
  echo "" >> "$OUTPUT"
done

# Types section
echo "## Type Exports" >> "$OUTPUT"
echo "" >> "$OUTPUT"

for type_key in "type_tanstack:TanStack Table" "type_reactflow:React Flow" "type:Custom"; do
  key="${type_key%%:*}"
  label="${type_key##*:}"
  file="${TMPDIR_WORK}/${key}.txt"
  if [[ -f "$file" ]]; then
    names=$(LC_ALL=C sort "$file" | sed 's/.*/ `&`/' | paste -sd',' | sed 's/,/, /g; s/^ //')
    echo "**${label}:** ${names}" >> "$OUTPUT"
    echo "" >> "$OUTPUT"
  fi
done

# Footer
cat >> "$OUTPUT" << 'FOOTER'
---

> **Note:** This file lists explicit named exports from the entry point; wildcard
> re-exports are not expanded. For detailed props documentation, see the
> [Storybook](http://localhost:6006) or read component source files in the in-tree catalog.
> Keep authored props guidance in ordinary documentation; regeneration replaces
> this entire index.
FOOTER

echo ""
if (( CHECK )); then
  if ! cmp -s "$OUTPUT" "$EXPECTED_OUTPUT"; then
    echo "Catalog documentation is stale; run this script without --check." >&2
    exit 1
  fi
  echo "Catalog documentation matches the in-tree exports."
else
  mkdir -p "$(dirname "$EXPECTED_OUTPUT")"
  cp "$OUTPUT" "$EXPECTED_OUTPUT"
  echo "Generated: ${EXPECTED_OUTPUT}"
fi
echo "Components: ${n_components}, Types: ${n_types}"
