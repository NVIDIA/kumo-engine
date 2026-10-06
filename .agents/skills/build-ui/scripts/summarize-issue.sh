#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <issue-number-or-url>" >&2
  exit 1
fi

issue="$1"
issue_number="$issue"
if [[ "$issue_number" =~ /issues/([0-9]+)([/?#].*)?$ ]]; then
  issue_number="${BASH_REMATCH[1]}"
elif [[ "$issue_number" =~ ^#([0-9]+)$ ]]; then
  issue_number="${BASH_REMATCH[1]}"
fi

# This helper assumes gh can resolve the current repository context.
# Run it from inside the target repo, or configure a default repo with:
#   gh repo set-default <owner>/<repo>

# Fetch issue metadata. The --json comments field can fail on repos with
# Projects Classic enabled (GraphQL deprecation error). Fetch metadata and
# comments separately to handle this gracefully.
metadata=$(gh issue view "$issue" --json number,title,body,labels,assignees,milestone \
  --jq '{
    number,
    title,
    labels: [.labels[].name],
    assignees: [.assignees[].login],
    milestone: (.milestone.title // ""),
    body
  }' 2>&1) || {
  echo "ERROR: Failed to fetch issue metadata: $metadata" >&2
  exit 1
}

# Try gh issue view for comments first, fall back to REST API. Do not silently
# drop comments: issue comments often contain reproduction steps, Figma links,
# and screenshot override intent.
if ! comments=$(gh issue view "$issue" --json comments --jq '[.comments[].body]' 2>/dev/null); then
  comments=$(gh api "repos/{owner}/{repo}/issues/${issue_number}/comments" --jq '[.[].body]' 2>&1) || {
    echo "ERROR: Failed to fetch issue comments: $comments" >&2
    exit 1
  }
fi

# Merge metadata and comments into single JSON output
echo "$metadata" | jq --argjson comments "$comments" '. + {comments: $comments}'
