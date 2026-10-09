#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

require_file() {
  local path="$1"
  if [[ ! -f "$ROOT/$path" ]]; then
    echo "MISSING FILE: $path" >&2
    exit 1
  fi
}

require_text() {
  local path="$1"
  local pattern="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -F -q -- "$pattern" "$ROOT/$path" && return 0
  elif command -v grep >/dev/null 2>&1; then
    grep -F -q -- "$pattern" "$ROOT/$path" && return 0
  else
    echo "ERROR: validate-skill-structure.sh requires rg or grep on PATH." >&2
    exit 1
  fi

  if [[ $? -ne 0 ]]; then
    echo "MISSING TEXT: $path :: $pattern" >&2
    exit 1
  fi
}

main() {
  require_file ".agents/skills/build-ui/SKILL.md"
  require_file ".agents/skills/build-ui/references/decision-tables.md"
  require_file ".agents/skills/build-ui/references/completion-criteria.md"
  require_file ".agents/skills/build-ui/references/behavioral-checks.tsv"
  require_file ".agents/skills/build-ui/references/headless-mode.md"
  require_file ".agents/skills/build-ui/references/handoff-examples.md"
  require_file ".agents/skills/build-ui/references/output-contracts.md"
  require_file ".agents/skills/build-ui/references/pr-workflow.md"
  require_file ".agents/skills/build-ui/references/run-state-template.json"
  require_file ".agents/skills/build-ui/references/screenshot-operator-guide.md"
  require_file ".agents/skills/build-ui/references/workflow-test-scenarios.md"
  require_file ".agents/skills/build-ui/scripts/check-preflight.sh"
  require_file ".agents/skills/build-ui/scripts/summarize-issue.sh"
  require_file ".agents/skills/build-ui/scripts/update-headless-artifact.sh"
  require_file ".agents/skills/build-ui/scripts/validate-output-schemas.sh"
  require_file ".agents/skills/build-ui/scripts/validate-workflow-behavior.sh"
  require_file ".agents/skills/build-ui/scripts/write-headless-artifact.sh"
  require_file ".agents/skills/code-ui/SKILL.md"
  require_file ".agents/skills/code-ui/references/catalog-components.md"
  require_file ".agents/skills/code-ui/references/design-tokens.md"
  require_file ".agents/skills/code-ui/references/integration-checklist.md"
  require_file ".agents/skills/code-ui/references/testing-playbook.md"

  require_text ".agents/skills/build-ui/SKILL.md" "## Ownership Boundaries"
  require_text ".agents/skills/build-ui/SKILL.md" "## Required Outputs"
  require_text ".agents/skills/build-ui/SKILL.md" "## Minimum Path"
  require_text ".agents/skills/build-ui/SKILL.md" "## Phase Entry And Exit Criteria"
  require_text ".agents/skills/build-ui/SKILL.md" 'Construct a `/code-ui` task using this handoff contract:'
  require_text ".agents/skills/build-ui/SKILL.md" "references/headless-mode.md"
  require_text ".agents/skills/build-ui/SKILL.md" "references/handoff-examples.md"
  require_text ".agents/skills/build-ui/SKILL.md" "Validation ownership:"
  require_text ".agents/skills/build-ui/SKILL.md" "Failure Classification"
  require_text ".agents/skills/build-ui/SKILL.md" "Evidence"
  require_text ".agents/skills/build-ui/SKILL.md" "## Resume Capability"
  require_text ".agents/skills/build-ui/SKILL.md" "Resume stage mapping:"
  require_text ".agents/skills/build-ui/SKILL.md" 'Use the exact phase label from this table in `lastCompletedPhase`.'
  require_text ".agents/skills/build-ui/SKILL.md" "### Run State Artifact"
  require_text ".agents/skills/build-ui/SKILL.md" "Resume Mismatch"
  require_text ".agents/skills/build-ui/SKILL.md" "## Workflow Validation"
  require_text ".agents/skills/build-ui/SKILL.md" "validate-workflow-behavior.sh"
  require_text ".agents/skills/build-ui/SKILL.md" "validate-output-schemas.sh"
  require_text ".agents/skills/build-ui/SKILL.md" "check-preflight.sh"
  require_text ".agents/skills/build-ui/SKILL.md" "## Final Handoff"

  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Checkpoint Behavior"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Phase Behavior in Headless Mode"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Reporting"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Required CI Artifacts"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "## Screenshot Plan Contract"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "BLOCKER_SCOPE_TOO_LARGE"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "screenshot-plan.json"
  require_text ".agents/skills/build-ui/references/headless-mode.md" "owns screenshot execution directly"
  require_text ".agents/skills/build-ui/references/headless-mode.md" 'must validate the `screenshot-plan.json` schema'
  require_text ".agents/skills/build-ui/references/headless-mode.md" "include the concrete failure reason"

  require_text ".agents/skills/build-ui/references/decision-tables.md" "## Shared Component Blast Radius"
  require_text ".agents/skills/build-ui/references/decision-tables.md" "## Final Diff Sanity"
  require_text ".agents/skills/build-ui/references/decision-tables.md" 'Do not use `pre-existing` without evidence.'

  require_text ".agents/skills/build-ui/references/screenshot-operator-guide.md" "## Failure Triage"
  require_text ".agents/skills/build-ui/references/screenshot-operator-guide.md" "## Local Reproduction"
  require_text ".agents/skills/build-ui/references/screenshot-operator-guide.md" "## Catalog Maintenance Rules"

  require_text ".agents/skills/build-ui/references/handoff-examples.md" "## Trivial Bug Fix Handoff"
  require_text ".agents/skills/build-ui/references/handoff-examples.md" "## New Page Handoff"
  require_text ".agents/skills/build-ui/references/handoff-examples.md" "## Cross-Surface Workflow Handoff"

  require_text ".agents/skills/code-ui/SKILL.md" "## Ownership Boundaries"
  require_text ".agents/skills/code-ui/SKILL.md" '`/code-ui` does not own:'
  require_text ".agents/skills/code-ui/SKILL.md" "references/integration-checklist.md"
  require_text ".agents/skills/code-ui/SKILL.md" "references/testing-playbook.md"
  require_text ".agents/skills/code-ui/SKILL.md" "## Step 3a: Apply Integration Checks"
  require_text ".agents/skills/code-ui/SKILL.md" "## Step 6: Test Strategy"

  require_text ".agents/skills/build-ui/references/workflow-test-scenarios.md" "## PR Mode Continuation"
  require_text ".agents/skills/build-ui/references/workflow-test-scenarios.md" "## Catalog Dependency"
  require_text ".agents/skills/build-ui/references/workflow-test-scenarios.md" "## Large PR Splitting"
  require_text ".agents/skills/build-ui/references/workflow-test-scenarios.md" "## Shared Component Blast Radius Review"
  require_text ".agents/skills/build-ui/references/workflow-test-scenarios.md" "## Resume Mismatch"
  require_text ".agents/skills/build-ui/references/workflow-test-scenarios.md" "## Pre-Existing Failure Claim"
  require_text ".agents/skills/build-ui/references/workflow-test-scenarios.md" "## Final Diff Sanity"

  bash "$ROOT/.agents/skills/build-ui/scripts/validate-workflow-behavior.sh" >/dev/null
  bash "$ROOT/.agents/skills/build-ui/scripts/validate-output-schemas.sh" >/dev/null

  echo "Skill structure validation passed."
}

main "$@"
