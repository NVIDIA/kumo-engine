#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

require_file() {
  local path="$1"
  [[ -f "$ROOT/$path" ]] || { echo "MISSING FILE: $path" >&2; exit 1; }
}

require_text() {
  local path="$1"
  local pattern="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -F -q -- "$pattern" "$ROOT/$path" && return 0
  elif command -v grep >/dev/null 2>&1; then
    grep -F -q -- "$pattern" "$ROOT/$path" && return 0
  else
    echo "ERROR: validate-output-schemas.sh requires rg or grep on PATH." >&2
    exit 1
  fi

  echo "MISSING TEXT: $path :: $pattern" >&2
  exit 1
}

require_abs_text() {
  local path="$1"
  local pattern="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -F -q -- "$pattern" "$path" && return 0
  elif command -v grep >/dev/null 2>&1; then
    grep -F -q -- "$pattern" "$path" && return 0
  else
    echo "ERROR: validate-output-schemas.sh requires rg or grep on PATH." >&2
    exit 1
  fi

  echo "MISSING TEXT: $path :: $pattern" >&2
  exit 1
}

main() {
  require_file ".agents/skills/build-ui/SKILL.md"
  require_file ".agents/skills/build-ui/references/headless-mode.md"
  require_file ".agents/skills/build-ui/references/output-contracts.md"
  require_file ".agents/skills/build-ui/references/run-state-template.json"
  require_file ".agents/skills/build-ui/scripts/update-headless-artifact.sh"
  require_file ".agents/skills/build-ui/scripts/write-headless-artifact.sh"

  require_text ".agents/skills/build-ui/SKILL.md" "Checkpoint 1"
  require_text ".agents/skills/build-ui/SKILL.md" "Verification Status"
  require_text ".agents/skills/build-ui/SKILL.md" "Checkpoint 2"
  require_text ".agents/skills/build-ui/SKILL.md" "Resume Mismatch"
  require_text ".agents/skills/build-ui/SKILL.md" "Handoff"
  require_text ".agents/skills/build-ui/SKILL.md" "references/output-contracts.md"
  require_text ".agents/skills/build-ui/SKILL.md" "references/run-state-template.json"

  require_text ".agents/skills/build-ui/references/output-contracts.md" "## Checkpoint 1"
  require_text ".agents/skills/build-ui/references/output-contracts.md" "## Verification Status"
  require_text ".agents/skills/build-ui/references/output-contracts.md" "## Checkpoint 2"
  require_text ".agents/skills/build-ui/references/output-contracts.md" "## Resume Mismatch"
  require_text ".agents/skills/build-ui/references/output-contracts.md" "## Handoff"
  require_text ".agents/skills/build-ui/references/output-contracts.md" "## Headless CI Artifacts"
  require_text ".agents/skills/build-ui/references/output-contracts.md" "## Screenshot Plan"
  require_text ".agents/skills/build-ui/references/output-contracts.md" "## Headless GitHub Comment"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Required CI Artifacts"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Pre-flight Publication"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Artifact-First Debugging"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Screenshot Plan Contract"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Stale-Branch Protection"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Bounded Auto-Fix Behavior"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "BLOCKER_API_MISSING"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "BLOCKER_WORKFLOW_FAILURE"

  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"mode\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"target\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"flags\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"validateOnly\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "preflight|pr-inspection"
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"workBranch\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"lastCompletedPhase\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "Phase 0: Pre-flight"
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"blockerType\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "adjacent-fix"
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"verificationStatus\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"baseShaAtStart\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"baseShaAtEnd\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"staleBaseBranch\""
  require_text ".agents/skills/build-ui/references/run-state-template.json" "\"jobStatus\""
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'Recommended `run-state` payload fields in CI:'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'Recommended `artifact-manifest` payload fields:'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'Recommended `verification-status` payload fields in workflow-owned screenshot mode:'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'screenshot-plan.json'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'screenshotFailureReason'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'Recommended `blocker` payload fields for screenshot-related blockers:'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'invalid screenshot-plan schema should produce a structured blocker before Playwright runs'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'wrote-override'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'failed-override'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'merged-primary'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'rewrote-multi'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'selectedRecipes'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'screenshotOverride'
  require_text ".agents/skills/build-ui/references/output-contracts.md" '_recipeKeys'
  require_text ".agents/skills/build-ui/references/output-contracts.md" '_rationale'
  require_text ".agents/skills/build-ui/references/output-contracts.md" '_requiredFeatureFlags'
  require_text ".agents/skills/build-ui/references/output-contracts.md" '_screenshotOverride'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'visualCoverage'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'requiredFeatureFlags'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'presentation'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'paste'
  require_text ".agents/skills/build-ui/references/headless-mode.md" 'BLOCKER_SCREENSHOT_COVERAGE_MISSING'
  require_text ".agents/skills/build-ui/references/headless-mode.md" 'BLOCKER_SCREENSHOT_STATUS_EVALUATION_FAILED'
  require_text ".agents/skills/build-ui/references/headless-mode.md" 'BLOCKER_SCREENSHOT_UPLOAD_FAILED'
  require_text ".agents/skills/build-ui/references/output-contracts.md" 'BLOCKER_SCREENSHOT_STATUS_EVALUATION_FAILED'

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  bash "$ROOT/.agents/skills/build-ui/scripts/write-headless-artifact.sh" --type preflight --target 12345 --output "$tmp_dir/preflight.json" >/dev/null
  bash "$ROOT/.agents/skills/build-ui/scripts/write-headless-artifact.sh" --type blocker --target 12345 --output "$tmp_dir/blocker.json" >/dev/null
  bash "$ROOT/.agents/skills/build-ui/scripts/write-headless-artifact.sh" --type artifact-manifest --target 12345 --output "$tmp_dir/artifact-manifest.json" >/dev/null
  bash "$ROOT/.agents/skills/build-ui/scripts/update-headless-artifact.sh" --input "$tmp_dir/preflight.json" --payload-json '{"jobStatus":"in_progress","staleBaseBranch":false}' >/dev/null
  require_abs_text "$tmp_dir/preflight.json" "\"artifactType\": \"preflight\""
  require_abs_text "$tmp_dir/preflight.json" "\"headless\": true"
  require_abs_text "$tmp_dir/preflight.json" "\"target\": \"12345\""
  require_abs_text "$tmp_dir/preflight.json" "\"jobStatus\": \"in_progress\""
  require_abs_text "$tmp_dir/preflight.json" "\"staleBaseBranch\": false"
  require_abs_text "$tmp_dir/blocker.json" "\"artifactType\": \"blocker\""
  require_abs_text "$tmp_dir/artifact-manifest.json" "\"artifactType\": \"artifact-manifest\""

  echo "Output schema validation passed."
}

main "$@"
