#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Cut a kumo3 (v3.0+) release:
# 1. create release branch `vX.Y` from origin/master,
# 2. tag `vX.Y.0` and publish a GitHub release, and
# 3. build the kumo3 image set (kumo-ui, kumo-backend-cpu, kumo-backend-gpu).
#
# Usage examples:
#
#    # Cut v3.0 interactively:
#    ./scripts/release/start_kumo3_release.sh false v3.0
#
#    # Show exactly what would happen, changing nothing:
#    ./scripts/release/start_kumo3_release.sh --dry-run false v3.0
#
#    # Non-interactive, builds images automatically:
#    ./scripts/release/start_kumo3_release.sh true v3.0
#
# Kumo 2 releases are no longer created from this repository. Emergency
# v2.21 maintenance runs in kumo-ai/kumo-v2-maintenance.

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo
  echo "Usage: ./scripts/release/start_kumo3_release.sh [--dry-run] <CI> <VERSION> [BASE_REF]"
  echo
  echo "  --dry-run (optional): print every mutating command without running it"
  echo "  CI: 'true' to run non-interactively and build images automatically"
  echo "  VERSION: the minor version of the release, such as 'v3.0'"
  echo "  BASE_REF (optional): branch or commit to cut from. Default: origin/master"
  echo
  exit 1
fi

CI=$1
VERSION=$2
BASE_REF=${3:-origin/master}

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

if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+$ ]]; then
  echo "kumo3 release branches look like 'v3.0', but got '${VERSION}'" >&2
  exit 1
fi

MAJOR=${VERSION#v}
MAJOR=${MAJOR%%.*}
MINOR=${VERSION##*.}

if (( MAJOR < 3 )); then
  echo "v${MAJOR}.x is not released from this repository." >&2
  echo "For supported v2.21 emergency maintenance, see:" >&2
  echo "  https://github.com/kumo-ai/kumo-v2-maintenance" >&2
  exit 1
fi

# All preflight checks run before any branch, tag, or release is created, so a
# missing tool cannot strand a half-finished release with a public tag already
# pushed.
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

  # Needed to build the compare URL the oversized-notes fallback links to.
  # Read it from the checkout rather than hardcoding a slug, so a fork or a
  # renamed repo does not silently produce a link to somewhere else.
  REPO_SLUG=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
  if [[ -z "$REPO_SLUG" ]]; then
    echo "Could not determine the GitHub owner/repo for this checkout." >&2
    exit 1
  fi
}

preflight

git fetch origin --tags

# Release notes are generated from the previous *minor release*, never from a
# hotfix tag: hotfix tags such as v2.21.35 live on a maintenance branch, so
# diffing against one produces a misleading changelog. Only `.0` tags qualify.
previous_release_tag() {
  local pattern=$1
  git tag -l | grep -E "$pattern" | sort -V | tail -n 1
}

if (( MINOR == 0 )); then
  PREV_VERSION=$(previous_release_tag "^v$(( MAJOR - 1 ))\.[0-9]+\.0$")
else
  PREV_VERSION=$(previous_release_tag "^v${MAJOR}\.$(( MINOR - 1 ))\.0$")
fi

NOTES_ARGS=()
if [[ -n "$PREV_VERSION" ]]; then
  NOTES_ARGS=(--notes-start-tag "$PREV_VERSION")
  echo "Release notes baseline: ${PREV_VERSION}"
else
  echo "Warning: no previous release tag found. Release notes will cover"
  echo "whatever range GitHub picks automatically."
fi

BRANCH=$VERSION
TAG=${VERSION}.0
FLOATING_TAG=$VERSION

# kumo3 builds exactly three images: kumo-ui, kumo-backend-cpu,
# kumo-backend-gpu. `--app kumo3` supports no target other than saas.
BUILD_CMD=(python3 scripts/release/build_kumo_images.py)
BUILD_ARGS=(--app kumo3 --target saas --publish_to_acr)

# The same pair of builds is printed as copy-pasteable instructions from two
# different places, so the command signature lives here only.
build_command_lines() {
  local tag
  for tag in "$TAG" "$FLOATING_TAG"; do
    printf '  %s %s %s\n' "${BUILD_CMD[*]}" "$tag" "${BUILD_ARGS[*]}"
  done
}

# The GitHub releases API rejects a body over 125,000 characters with
# "HTTP 422: Validation Failed -- body is too long". A major-version cut
# always diffs against the previous major's last `.0` tag, so its changelog
# is hundreds of commits long and lands near that ceiling every time; v3.0.0
# came in at ~137k. Sit well below the edge rather than on it: the check
# counts bytes and the API counts characters, so a body with any multi-byte
# character in it measures larger here than it does there -- the safe
# direction -- and the headroom also covers the fallback body's own text.
GITHUB_RELEASE_BODY_LIMIT=125000
RELEASE_BODY_BUDGET=120000

# Printed instead of the changelog when the changelog does not fit. A
# truncated list is worse than a linked one: it reads as complete, so nobody
# goes looking for the rest.
compact_release_body() {
  local size=$1
  cat <<EOF
**[Full changelog: ${PREV_VERSION}...${TAG}](https://github.com/${REPO_SLUG}/compare/${PREV_VERSION}...${TAG})**

The auto-generated changelog for this range is ${size} bytes, over the
${GITHUB_RELEASE_BODY_LIMIT}-character limit the GitHub API enforces on a release body, so it
is linked above in full rather than truncated to fit here.
EOF
}

# Generate the notes up front instead of letting `gh release create
# --generate-notes` do it inline. Inline generation moves the size check to
# GitHub, which reports it as a 422 -- after the branch and tag are already
# pushed and public, leaving a half-finished release for a human to clean up.
create_release() {
  if (( ${#NOTES_ARGS[@]} == 0 )); then
    # No baseline tag to diff against, so GitHub picks the range. Nothing to
    # size up in advance; leave this path exactly as it was.
    run gh release create "$TAG" --verify-tag --generate-notes --title "$TAG"
    return
  fi

  # Not a `local`: the EXIT trap below runs after this function's locals are
  # gone. Generating notes is read-only, so it runs on a dry run too -- that
  # is what makes the dry run able to say which body a real cut would use.
  NOTES_FILE=$(mktemp)
  trap 'rm -f "$NOTES_FILE"' EXIT

  if ! gh api "repos/${REPO_SLUG}/releases/generate-notes" \
      -f tag_name="$TAG" -f previous_tag_name="$PREV_VERSION" --jq .body \
      > "$NOTES_FILE"; then
    # An unreachable or unauthenticated gh is not an oversized changelog, and
    # must not be answered with the fallback body -- that would publish a
    # release claiming its changelog was too long to inline when in truth it
    # was never fetched. Stop, and say which failure this is.
    echo "Could not generate release notes for ${PREV_VERSION}..${TAG} (gh error" >&2
    echo "above). This is not the oversized-changelog case; no release was" >&2
    echo "published. Retry, or publish by hand once gh works:" >&2
    echo "  gh release create ${TAG} --verify-tag --generate-notes --title ${TAG} ${NOTES_ARGS[*]}" >&2
    exit 1
  fi

  local size
  size=$(wc -c < "$NOTES_FILE")
  if (( size > RELEASE_BODY_BUDGET )); then
    echo "Release notes for ${PREV_VERSION}..${TAG} are ${size} bytes, over the"
    echo "${RELEASE_BODY_BUDGET}-byte budget. Linking the compare view instead of inlining."
    compact_release_body "$size" > "$NOTES_FILE"
  fi

  run gh release create "$TAG" --verify-tag --title "$TAG" \
    --notes-file "$NOTES_FILE"
}

# Ask origin rather than a local remote-tracking ref, matching the tag probe
# below. refs/remotes/origin/${BRANCH} does not exist in a `--single-branch`
# or `--depth` clone even when the branch is plainly there on origin, and the
# checkout configuration of the CI job running this is not defined in this
# repository. `git fetch origin --tags` above already talks to origin, so this
# adds no round-trip the script was not making.
if git ls-remote --exit-code origin "refs/heads/${BRANCH}" > /dev/null 2>&1; then
  # A cut is three public steps -- push branch, push tag, publish release -- and
  # it can die after any of them. Telling the operator to backport is only right
  # once all three are done; for the other two states it is actively wrong
  # advice, so probe for each one separately.
  #
  # Probe the remote, not refs/tags: "was this published" is a question about
  # origin. A cut that died between `git tag` and `git push origin <tag>` leaves
  # a local tag behind, and reading that as a shipped release is exactly the
  # mistake this avoids.
  if ! git ls-remote --exit-code origin "refs/tags/${TAG}" > /dev/null 2>&1; then
    echo "Release branch 'origin/${BRANCH}' exists but tag '${TAG}' was never" >&2
    echo "pushed to origin." >&2
    echo "A previous cut stopped part-way. Finish it by hand:" >&2
    echo "  git checkout ${BRANCH} && git pull origin ${BRANCH}" >&2
    echo "  git tag ${TAG}   # skip this if the tag already exists locally" >&2
    echo "  git push origin ${TAG}" >&2
    echo "  gh release create ${TAG} --verify-tag --generate-notes --title ${TAG} ${NOTES_ARGS[*]}" >&2
    echo "Or discard it and start over:" >&2
    echo "  git push origin --delete ${BRANCH}" >&2
  elif ! gh release view "$TAG" > /dev/null 2>&1; then
    # Branch and tag are both public, but `gh release create` never succeeded.
    # The tag alone makes this look shipped, so without this probe the operator
    # would be sent to backport a release that was never published.
    echo "Release branch 'origin/${BRANCH}' and tag '${TAG}' exist on origin," >&2
    echo "but no GitHub release was published for '${TAG}'." >&2
    echo "A previous cut stopped part-way. Finish it by hand:" >&2
    echo "  gh release create ${TAG} --verify-tag --generate-notes --title ${TAG} ${NOTES_ARGS[*]}" >&2
    echo "Then build the images:" >&2
    build_command_lines >&2
    echo "Or discard it and start over:" >&2
    echo "  git push origin --delete ${TAG} && git push origin --delete ${BRANCH}" >&2
  else
    echo "Release branch already exists at 'origin/${BRANCH}'." >&2
    echo "To ship a fix on it, backport instead of cutting a new release:" >&2
    echo "  ./scripts/release/backport_pr.sh <PULL_REQUEST> ${BRANCH}" >&2
  fi
  exit 1
fi

if git show-ref -q --verify "refs/tags/${TAG}"; then
  echo "Tag '${TAG}' already exists." >&2
  exit 1
fi

echo
echo "Base commit for ${BRANCH}:"
git log -1 "$BASE_REF"
echo

if [[ "$CI" == "false" && "$DRY_RUN" == "false" ]]; then
  read -p "Cut release branch '${BRANCH}' and tag '${TAG}'? [Y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

run git checkout -b "$BRANCH" "$BASE_REF"
run git push origin "$BRANCH"
run git tag "$TAG"
run git push origin "$TAG"
create_release

if [[ "$CI" == "true" ]]; then
  run "${BUILD_CMD[@]}" "$TAG" "${BUILD_ARGS[@]}"
  run "${BUILD_CMD[@]}" "$FLOATING_TAG" "${BUILD_ARGS[@]}"
else
  echo
  if [[ "$DRY_RUN" == "true" ]]; then
    # A dry run created nothing, so the build commands below would push images
    # for a tag that does not exist. Say so before printing them.
    echo "Dry run: nothing above was executed. Branch '${BRANCH}' and tag"
    echo "'${TAG}' were NOT created. Do not run these yet -- a real run would"
    echo "print them once the tag exists:"
  else
    echo "Release branch '${BRANCH}' and tag '${TAG}' created."
    echo
    echo "Next, build the kumo3 images (kumo-ui, kumo-backend-cpu, kumo-backend-gpu):"
  fi
  echo
  build_command_lines
fi
