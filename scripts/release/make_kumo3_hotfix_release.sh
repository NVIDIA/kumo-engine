#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Cut a kumo3 (v3.0+) hotfix release from a release branch:
# 1. tag `vX.Y.{Z+1}` on the branch tip, where `vX.Y.Z` is its latest patch tag,
# 2. publish a GitHub release whose notes are the backports since that tag, and
# 3. build the kumo3 image set (kumo-ui, kumo-backend-cpu, kumo-backend-gpu).
#
# Run it from the release branch, with the backports already merged (see
# backport_pr.sh).
#
# Usage examples:
#
#    # Show exactly what would happen, changing nothing:
#    ./scripts/release/make_kumo3_hotfix_release.sh --dry-run false
#
#    # Cut the next patch on the current release branch, interactively:
#    ./scripts/release/make_kumo3_hotfix_release.sh false
#
#    # Non-interactive, builds images automatically. This is how
#    # check_updated_release_branches.sh invokes it.
#    ./scripts/release/make_kumo3_hotfix_release.sh true
#
# Kumo 2 hotfixes are no longer created from this repository. Emergency
# v2.21 maintenance runs in kumo-ai/kumo-v2-maintenance.

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo
  echo "Usage: ./scripts/release/make_kumo3_hotfix_release.sh [--dry-run] <CI> [TAG] [PREV_TAG]"
  echo
  echo "  --dry-run (optional): print every mutating command without running it"
  echo "  CI: 'true' to run non-interactively and build images automatically"
  echo "  TAG (optional): the tag to create. Default: PREV_TAG with the patch"
  echo "    number incremented."
  echo "  PREV_TAG (optional): the tag release notes are generated from. All"
  echo "    commits between PREV_TAG and HEAD are listed in the release."
  echo "    Default: the highest patch tag on the current release branch."
  echo
  exit 1
fi

CI=$1
TAG=$2
PREV_TAG=$3

# CI drives two decisions independently: 'false' prompts, 'true' builds. A
# value that is neither -- a typo such as 'flase' -- matches neither test, so
# the script would tag, push and publish with no confirmation and then build no
# images for what it just published. Reject it here, before anything mutates.
if [[ "$CI" != "true" && "$CI" != "false" ]]; then
  echo "CI must be exactly 'true' or 'false', but got '${CI}'." >&2
  exit 1
fi

# Every mutating command goes through here, so --dry-run output is exactly the
# list of commands a real run would execute -- not a separate description of
# them that can drift.
run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+ %s\n' "$*"
  else
    "$@"
  fi
}

BRANCH=$(git rev-parse --abbrev-ref HEAD)
# TODO: Currently, the branch looks like heads/v1.41
# so we need to remove the heads/ prefix. This is needed
# if we need to handle a tag and a branch with the same
# name. Let's delete this once we fix the issue.
BRANCH=${BRANCH#heads/}

# BRANCH, TAG and PREV_TAG are all checked because they are the whole surface a
# version can arrive on: TAG is either passed in or derived from PREV_TAG, and
# PREV_TAG is either passed in or looked up by BRANCH prefix. Comparing the
# captured major numerically, rather than matching digits, is what makes v10
# read as newer than v3 instead of sorting below it as a string.
for candidate in "$BRANCH" "$TAG" "$PREV_TAG"; do
  if [[ "$candidate" =~ ^v([0-9]+) ]] && (( BASH_REMATCH[1] < 3 )); then
    echo "'${candidate}' is on a pre-v3 line, which is not released from" >&2
    echo "this repository. For supported v2.21 emergency maintenance, see:" >&2
    echo >&2
    echo "  https://github.com/kumo-ai/kumo-v2-maintenance" >&2
    echo >&2
    exit 1
  fi
done

if [[ ! "$BRANCH" =~ ^v[0-9]+\.[0-9]+$ ]]; then
  echo "This script tags the tip of a release branch, so it must be run from" >&2
  echo "one -- a branch named like 'v3.0'. The current branch is '${BRANCH}'." >&2
  exit 1
fi

# All preflight checks run before any tag or release is created, so a missing
# tool cannot strand a half-finished hotfix with a public tag already pushed.
preflight() {
  if ! command -v gh &> /dev/null; then
    echo "github CLI not found. Install it: https://cli.github.com/" >&2
    exit 1
  fi

  local gh_version
  gh_version=$(gh --version | perl -pe 'if(($v)=/([0-9]+([.][0-9]+)+)/){print"$v\n";exit}$_=""')
  if ! { echo "2.28.0"; echo "$gh_version"; } | sort -V -C; then
    echo "github cli ${gh_version} is out of date. Please upgrade to at least 2.28.0" >&2
    exit 1
  fi

  # build_kumo_images.py imports packaging, which is not stdlib and is not
  # importable outside the venv.
  if ! python3 -c "import packaging" 2>/dev/null; then
    echo "build_kumo_images.py needs the 'packaging' module, which is not" >&2
    echo "importable by this python3. Set up the venv first:" >&2
    echo "  make install-venv   # then re-run from inside the venv" >&2
    exit 1
  fi

  if ! gh auth status &> /dev/null; then
    echo "gh is not authenticated. Run: gh auth login" >&2
    exit 1
  fi
}

preflight

echo "Fetching latest changes from origin"
git fetch origin --tags

if [ -z "$PREV_TAG" ]; then
  # `git tag -l <glob>` rather than a grep on the branch name: the branch name
  # contains a '.', which as a grep pattern matches any character, so 'v3.0'
  # would also pick up tags on a 'v3.0X' branch.
  PREV_TAG=$(git tag -l "${BRANCH}.*" |
    grep -E "^v[0-9]+\.[0-9]+\.[0-9]+$" | sort -V | tail -1)

  if [ -z "$PREV_TAG" ]; then
    echo "Could not find a patch tag on '${BRANCH}' to base this hotfix on." >&2
    echo "Expected a tag such as '${BRANCH}.0'. If the release branch was cut" >&2
    echo "but never tagged, finish that first:" >&2
    echo "  ./scripts/release/start_kumo3_release.sh ${CI} ${BRANCH}" >&2
    echo "Otherwise pass the tags explicitly:" >&2
    echo "  ./scripts/release/make_kumo3_hotfix_release.sh ${CI} <TAG> <PREV_TAG>" >&2
    exit 1
  fi
fi

if [ -z "$TAG" ]; then
  TAG=$(echo "${PREV_TAG}" | awk -F. -v OFS=. '{$NF += 1 ; print}')
fi

# The floating minor tag, rebuilt alongside the patch tag so that 'v3.0' keeps
# pointing at the newest v3.0.x images.
FLOATING_TAG=$BRANCH

# kumo3 builds exactly three images: kumo-ui, kumo-backend-cpu,
# kumo-backend-gpu. `--app kumo3` supports no target other than saas.
BUILD_CMD=(python3 scripts/release/build_kumo_images.py)
BUILD_ARGS=(--app kumo3 --target saas --publish_to_acr)

# The same pair of builds is printed as copy-pasteable instructions and run
# directly, so the command signature lives here only.
build_command_lines() {
  local tag
  for tag in "$TAG" "$FLOATING_TAG"; do
    printf '  %s %s %s\n' "${BUILD_CMD[*]}" "$tag" "${BUILD_ARGS[*]}"
  done
}

# Ask origin, not refs/tags. "Has this already been released?" is a question
# about origin, and a local tag left behind by a run that died between
# `git tag` and `git push origin <tag>` answers it wrongly in both directions:
# it makes an unpublished patch look shipped, and its absence in a shallow or
# single-branch clone makes a shipped one look unpublished.
#
# Cutting a hotfix is two public steps -- push the tag, publish the release --
# so there are three states here, not two. The middle one, tag pushed but
# release never created, is the one that used to dead-end: refusing there left
# the tag orphaned, and since the next run's PREV_TAG lookup picks that tag up
# and cuts the patch after it, the orphan never got a release at all. Probe for
# each state separately, the same way start_kumo3_release.sh does.
RESUME=false
if git ls-remote --exit-code origin "refs/tags/${TAG}" > /dev/null 2>&1; then
  if gh release view "$TAG" > /dev/null 2>&1; then
    echo "Tag '${TAG}' and its GitHub release both exist, so this hotfix was" >&2
    echo "already cut. To ship a further fix, merge the backport and re-run:" >&2
    echo "the next patch number will be picked up automatically." >&2
    exit 1
  fi
  # Finish the cut rather than refusing it. Everything below skips the tag it
  # would otherwise create, so this publishes the release for the tag origin
  # already has -- it never creates a second one.
  RESUME=true
  echo "Tag '${TAG}' is on origin but no GitHub release was published for it:"
  echo "a previous cut stopped part-way. Resuming it -- no new tag is created."
fi

if ! git ls-remote --exit-code origin "refs/heads/${BRANCH}" > /dev/null 2>&1; then
  echo "Release branch '${BRANCH}' does not exist on origin. This script tags" >&2
  echo "what origin already has; push the branch first." >&2
  exit 1
fi

# Ensure that the local branch is up to date with the remote branch, in case
# someone else was pushing hotfixes.
run git rebase "origin/${BRANCH}"

# The rebase above goes through run(), so a dry run skips it and HEAD is
# whatever the local checkout happened to be. Reading HEAD there would make a
# dry run on a stale branch report no commits -- and refuse, below, as an empty
# release -- for a hotfix a real run would cut without complaint. Describe the
# range against origin instead, which is what the rebase would have produced.
# A real run keeps reading HEAD, so its behaviour is unchanged.
if [[ "$RESUME" == "true" ]]; then
  # The tag is already public and points where it points. Describing HEAD
  # would put commits into the notes that the tag does not contain.
  CHANGES_REF=$TAG
elif [[ "$DRY_RUN" == "true" ]]; then
  CHANGES_REF="origin/${BRANCH}"
else
  CHANGES_REF=HEAD
fi
CHANGES=$(git log "${PREV_TAG}".."${CHANGES_REF}" --oneline)

# An empty range means the tip is already released as PREV_TAG, so cutting
# here would publish a release identical to it. The sweep never reaches this
# (it only acts on an untagged tip), but a hand re-run does, and the tag and
# the GitHub release are both public by the time anyone notices. A resume is
# exempt: its tag is public already, so refusing would strand it again.
if [[ -z "$CHANGES" && "$RESUME" == "false" ]]; then
  echo "No commits on '${BRANCH}' since '${PREV_TAG}', so '${TAG}' would be an" >&2
  echo "empty release. Merge the backport first:" >&2
  echo "  ./scripts/release/backport_pr.sh <PULL_REQUEST> ${BRANCH}" >&2
  exit 1
fi

echo
if [[ "$RESUME" == "true" ]]; then
  echo "This will publish the release for existing tag '${TAG}', covering:"
else
  echo "This will create a release tag '${TAG}' with the following hotfixes:"
fi
echo
echo "${CHANGES}"

if [[ "$CI" == "false" && "$DRY_RUN" == "false" ]]; then
  echo
  if [[ "$RESUME" == "false" ]]; then
    echo "The tag will be pushed to 'origin/${BRANCH}' and the previous tag"
    echo "'${PREV_TAG}' will be used to generate release notes."
    echo
  fi
  echo "If this is not correct, please specify <TAG> and <PREV_TAG> explicitly:"
  echo
  echo "  ./scripts/release/make_kumo3_hotfix_release.sh ${CI} <TAG> <PREV_TAG>"
  echo
  read -p "Do you wish to continue? [Y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

if [[ "$RESUME" == "false" ]]; then
  run git tag "$TAG"
  run git push origin "$TAG"
fi

# A hotfix does not certify a line that was not certified, so if the previous
# release was a prerelease this one is published as a prerelease too.
#
# The field is `isPrerelease`. Asking `gh` for a field it does not know is an
# error, and it would be a silent one here: the call is deliberately allowed to
# fail, because a first hotfix on a tag that was never published as a release
# is a normal state, so a wrong field name reads as "no release exists" and
# quietly drops the prerelease flag.
set +e
IS_PRERELEASE=$(gh release view "$PREV_TAG" --json isPrerelease --jq '.isPrerelease' 2>/dev/null)
VIEW_STATUS=$?
set -e

RELEASE_ARGS=(--verify-tag --title "$TAG" --notes "$CHANGES")
if [[ "$VIEW_STATUS" -ne 0 ]]; then
  echo "No existing GitHub release found for ${PREV_TAG}. Creating a normal release for ${TAG}."
elif [[ "$IS_PRERELEASE" == "true" ]]; then
  echo "${PREV_TAG} is a pre-release, so ${TAG} is published as one too."
  RELEASE_ARGS+=(--prerelease)
fi

run gh release create "$TAG" "${RELEASE_ARGS[@]}"

if [[ "$CI" == "true" ]]; then
  # Same pair build_command_lines() prints, enumerated the same way so the
  # printed and executed paths cannot drift apart.
  for tag in "$TAG" "$FLOATING_TAG"; do
    run "${BUILD_CMD[@]}" "$tag" "${BUILD_ARGS[@]}"
  done
else
  echo
  if [[ "$DRY_RUN" == "true" ]]; then
    # A dry run created nothing, so the build commands below would push images
    # for a tag that does not exist. Say so before printing them.
    echo "Dry run: nothing above was executed. Do not run these yet -- a real"
    echo "run would print them once the tag exists:"
  else
    echo "Next, build the kumo3 images (kumo-ui, kumo-backend-cpu, kumo-backend-gpu):"
  fi
  echo
  build_command_lines
fi

if [[ "$DRY_RUN" == "true" ]]; then
  # check_updated_release_branches.sh records the release by grepping for
  # "New version is". A dry run must not hand it a version that was never cut.
  echo "Dry run: tag '${TAG}' was NOT created."
else
  echo "New version is ${TAG}"
fi
