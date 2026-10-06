#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Check freshness of agent documentation vs source of truth files.
# Flags when key source files have been modified more recently than the docs.
#
# Usage: bash .agents/skills/code-ui/scripts/check-docs-freshness.sh
#
# Exit codes:
#   0 - all docs are up to date
#   1 - one or more docs may be stale

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
STALE=0

# Map: doc file -> space-separated list of source-of-truth files it covers
declare -A DOC_SOURCES=(
  ["docs/development/ui.md"]="ui/src/utils/Constants/routes.ts ui/src/utils/Constants/enums.ts ui/src/utils/reactQueryKeys.ts ui/src/utils/queryInvalidation.ts ui/src/utils/storage.ts ui/src/utils/errorHandler.ts ui/src/utils/timeConverter.ts ui/src/routes/Routes.tsx ui/src/routes/Loaders.ts ui/src/tests/testUtil.ts ui/src/tests/mockAxios.ts ui/src/tests/setup.ts"
  [".agents/skills/review-ui/SKILL.md"]="docs/development/ui-review-kumo.md docs/development/ui-review-react.md docs/development/ui-review-testing.md"
  [".agents/skills/code-ui/SKILL.md"]="docs/development/ui.md .agents/skills/code-ui/references/catalog-components.md"
)

check_freshness() {
  local doc="$1"
  shift
  local sources=("$@")
  local doc_path="${REPO_ROOT}/${doc}"

  if [[ ! -f "$doc_path" ]]; then
    echo "MISSING: ${doc}"
    STALE=1
    return
  fi

  local doc_mtime
  doc_mtime=$(stat -c %Y "$doc_path" 2>/dev/null || stat -f %m "$doc_path")

  for src in "${sources[@]}"; do
    local src_path="${REPO_ROOT}/${src}"
    [[ -f "$src_path" ]] || continue

    local src_mtime
    src_mtime=$(stat -c %Y "$src_path" 2>/dev/null || stat -f %m "$src_path")

    if (( src_mtime > doc_mtime )); then
      echo "STALE: ${doc} (source changed: ${src})"
      STALE=1
    fi
  done
}

echo "=== Agent Documentation Freshness Check ==="
echo ""

for doc in "${!DOC_SOURCES[@]}"; do
  # shellcheck disable=SC2086
  check_freshness "$doc" ${DOC_SOURCES[$doc]}
done

# Check catalog components doc vs installed package
CATALOG_DOC="${REPO_ROOT}/.agents/skills/code-ui/references/catalog-components.md"
CATALOG_PKG="${REPO_ROOT}/ui/node_modules/@kumo-ai/kumo-ui-catalog/package.json"
if [[ -f "$CATALOG_DOC" && -f "$CATALOG_PKG" ]]; then
  doc_mtime=$(stat -c %Y "$CATALOG_DOC" 2>/dev/null || stat -f %m "$CATALOG_DOC")
  pkg_mtime=$(stat -c %Y "$CATALOG_PKG" 2>/dev/null || stat -f %m "$CATALOG_PKG")
  if (( pkg_mtime > doc_mtime )); then
    echo "STALE: catalog-components.md (catalog package updated)"
    STALE=1
  fi
fi

echo ""
if (( STALE == 0 )); then
  echo "All docs are up to date."
else
  echo "Some docs may need updating. Review the STALE entries above."
fi

exit $STALE
