#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Parse a /build-ui command from a GitHub comment body.
# Usage: parse-build-ui-command.sh "<comment-body>"
# Outputs (via GITHUB_OUTPUT): issue_number, flags

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 '<comment-body>'" >&2
  exit 1
fi

body="$1"

# Match: /build-ui <number> [optional flags]
if [[ "$body" =~ ^/build-ui[[:space:]]+([0-9]+)(.*)?$ ]]; then
  issue_number="${BASH_REMATCH[1]}"
  flags=$(echo "${BASH_REMATCH[2]}" | xargs)  # trim whitespace
else
  echo "ERROR: Could not parse /build-ui command. Expected: /build-ui <issue-number> [flags]" >&2
  exit 1
fi

# Validate issue number is reasonable
if [[ "$issue_number" -le 0 ]]; then
  echo "ERROR: Invalid issue number: $issue_number" >&2
  exit 1
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "issue_number=$issue_number" >> "$GITHUB_OUTPUT"
  echo "flags=$flags" >> "$GITHUB_OUTPUT"
else
  echo "issue_number=$issue_number"
  echo "flags=$flags"
fi
