#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

EVENT_NAME="${EVENT_NAME:-${GITHUB_EVENT_NAME:-}}"
EVENT_PATH="${GITHUB_EVENT_PATH:-}"
CONFIG_PATH=".github/claude-repos.json"

if [ -z "$EVENT_NAME" ]; then
  echo "ERROR: EVENT_NAME is required" >&2
  exit 1
fi

if [ -z "$EVENT_PATH" ] || [ ! -f "$EVENT_PATH" ]; then
  echo "ERROR: GITHUB_EVENT_PATH must point to an existing file" >&2
  exit 1
fi

case "$EVENT_NAME" in
  repository_dispatch)
    TRIGGER_TEXT=$(jq -r '.client_payload.comment_body // ""' "$EVENT_PATH")
    echo "Handling repository_dispatch event"
    echo "is_multi_repo=false" >> "$GITHUB_OUTPUT"
    echo "original_request<<EOF" >> "$GITHUB_OUTPUT"
    echo "$TRIGGER_TEXT" >> "$GITHUB_OUTPUT"
    echo "EOF" >> "$GITHUB_OUTPUT"
    ;;
  issue_comment)
    TRIGGER_TEXT=$(jq -r '.comment.body // ""' "$EVENT_PATH")
    ;;
  pull_request_review_comment)
    TRIGGER_TEXT=$(jq -r '.comment.body // ""' "$EVENT_PATH")
    ;;
  pull_request_review)
    TRIGGER_TEXT=$(jq -r '.review.body // ""' "$EVENT_PATH")
    ;;
  issues)
    TRIGGER_TEXT=$(jq -r '.issue.body // ""' "$EVENT_PATH")
    ;;
  *)
    TRIGGER_TEXT=""
    ;;
esac

echo "Trigger text: $TRIGGER_TEXT"

if [ "$EVENT_NAME" != "repository_dispatch" ]; then
  if [ -f "$CONFIG_PATH" ]; then
    KEYWORDS=$(jq -r '.multi_repo_keywords | join("|")' "$CONFIG_PATH")

    if echo "$TRIGGER_TEXT" | grep -qiE "($KEYWORDS)"; then
      echo "Multi-repo request detected!"
      echo "is_multi_repo=true" >> "$GITHUB_OUTPUT"

      REPOS=$(jq -c '.repositories' "$CONFIG_PATH")
      echo "repos=$REPOS" >> "$GITHUB_OUTPUT"

      echo "original_request<<EOF" >> "$GITHUB_OUTPUT"
      echo "$TRIGGER_TEXT" >> "$GITHUB_OUTPUT"
      echo "EOF" >> "$GITHUB_OUTPUT"
    else
      echo "Single-repo request"
      echo "is_multi_repo=false" >> "$GITHUB_OUTPUT"
    fi
  else
    echo "No claude-repos.json found, defaulting to single-repo mode"
    echo "is_multi_repo=false" >> "$GITHUB_OUTPUT"
  fi
fi
