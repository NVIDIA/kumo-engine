#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: write-headless-artifact.sh --type <type> --target <target> --output <file>
Types: preflight, run-state, checkpoint-1, verification-status, checkpoint-2, handoff, blocker, artifact-manifest
EOF
  exit 1
}

artifact_type=""
target=""
output=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --type)
      artifact_type="${2:-}"
      shift 2
      ;;
    --target)
      target="${2:-}"
      shift 2
      ;;
    --output)
      output="${2:-}"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

[[ -n "$artifact_type" && -n "$target" && -n "$output" ]] || usage

case "$artifact_type" in
  preflight|run-state|checkpoint-1|verification-status|checkpoint-2|handoff|blocker|artifact-manifest)
    ;;
  *)
    echo "Unsupported artifact type: $artifact_type" >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "$output")"

jq -n \
  --arg artifactType "$artifact_type" \
  --arg generatedAt "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  --arg target "$target" \
  '{
    artifactType: $artifactType,
    generatedAt: $generatedAt,
    headless: true,
    target: $target,
    payload: {}
  }' >"$output"

echo "Wrote $artifact_type artifact to $output"
