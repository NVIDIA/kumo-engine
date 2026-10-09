#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: update-headless-artifact.sh --input <file> [--payload-file <file> | --payload-json <json>]
Merges the provided payload into the artifact's existing .payload object and refreshes generatedAt.
EOF
  exit 1
}

input=""
payload_file=""
payload_json=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      input="${2:-}"
      shift 2
      ;;
    --payload-file)
      payload_file="${2:-}"
      shift 2
      ;;
    --payload-json)
      payload_json="${2:-}"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

[[ -n "$input" ]] || usage
[[ -f "$input" ]] || { echo "Artifact file not found: $input" >&2; exit 1; }

if [[ -n "$payload_file" ]]; then
  [[ -f "$payload_file" ]] || { echo "Payload file not found: $payload_file" >&2; exit 1; }
  payload_json="$(cat "$payload_file")"
fi

[[ -n "$payload_json" ]] || usage

tmp="$(mktemp)"
jq \
  --arg now "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  --argjson payload "$payload_json" \
  '.generatedAt = $now | .payload = ((.payload // {}) + $payload)' \
  "$input" >"$tmp"
mv "$tmp" "$input"

echo "Updated artifact $input"
