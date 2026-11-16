#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [ $# -ne 2 ] && [ $# -ne 3 ]
then
  echo "usage: $0 <pr num> <target version> <optional: --branch-ready>"
  echo
  echo "  <target version>: Kumo release branch (e.g., v1.9)"
  exit 1
fi

PR_NUM="${1}"
VERSION_INPUT="${2}"

# Normalize the target to include the v prefix.
VERSION_NUM="${VERSION_INPUT#v}"
VERSION_STR="v${VERSION_NUM}"

PR_SHA="$(gh pr view $PR_NUM --json=mergeCommit |jq -r '.["mergeCommit"]["oid"]')"

if [ "$PR_SHA" == "null" ] || [ -z "$PR_SHA" ]; then
  echo "Error: PR #$PR_NUM is not merged or has no merge commit."
  exit 1
fi

PR_DESC="$(gh pr view $PR_NUM --json=body | jq -r '.body')"
PR_URL="$(gh pr view $PR_NUM --json=url | jq -r '.url')"

BACKPORT_BRANCH="backport-$PR_NUM-$VERSION_NUM"

REMOTE="$(git remote)"
CURRENT_BRANCH="$(git branch --show-current)"

if [ "$3" != "--branch-ready" ]
then
  git fetch

  git branch -D $BACKPORT_BRANCH || true

  git checkout -b $BACKPORT_BRANCH $REMOTE/$VERSION_STR --no-track

  if ! git cherry-pick $PR_SHA
  then
    cat <<EOF

Merge conflict detected, manual merge required.
Complete the cherry-pick by fixing conflicts and:

    git commit

Return to your previous branch:

    git checkout $CURRENT_BRANCH

then complete the backport with:

    $0 $@ --branch-ready

EOF
    exit 1
  fi

  git checkout $CURRENT_BRANCH
fi

git push -f -u $REMOTE $BACKPORT_BRANCH:$BACKPORT_BRANCH

# Use VERSION_STR for the title to clearly identify the target.
TITLE="[backport $VERSION_STR] $(git log --format=%s -n 1 $PR_SHA)"

# Create PR and capture the URL
BACKPORT_PR_URL=$(gh pr create \
  --base="$VERSION_STR" \
  --head="$BACKPORT_BRANCH" \
  --title="$TITLE" \
  --body="

**If Backport / Hotfix PR:**
- Related master PR: ${PR_URL}") || {
  echo "Error: Failed to create backport PR"
  exit 1
}

BACKPORT_PR_NUM=$(echo "$BACKPORT_PR_URL" | sed -n 's/.*\/pull\/\([0-9]*\).*/\1/p')

if [ -z "$BACKPORT_PR_NUM" ] || ! [[ "$BACKPORT_PR_NUM" =~ ^[0-9]+$ ]]; then
  echo "Error: Failed to extract PR number from URL: $BACKPORT_PR_URL"
  exit 1
fi

# Add the backport label
gh pr edit "${BACKPORT_PR_NUM}" --add-label "pr-type/backport"

echo "Created backport PR #${BACKPORT_PR_NUM} at ${BACKPORT_PR_URL} with pr-type/backport label"
