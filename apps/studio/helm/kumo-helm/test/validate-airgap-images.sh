#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validates that every rendered image comes from the EXPECTED registry (i.e. nothing
# leaks to a public registry) — for an air-gapped / own-registry install.
# Usage:   ./validate-airgap-images.sh [values-file...]
#   The chart is rendered WITH your values file(s), so it checks the registry you will
#   actually deploy. EXPECTED_REGISTRY defaults to NVCR; override it to validate
#   your own mirror, e.g.:
#     EXPECTED_REGISTRY=111111111111.dkr.ecr.us-west-2.amazonaws.com \
#       ./validate-airgap-images.sh my-values.yaml
#
# Exit codes: 0 = every image OK; 1 = an image from an unexpected registry;
#             2 = the chart could not be rendered (a tooling/values error, NOT
#             an image problem).

set -euo pipefail

# The registry all images must come from. Override via env for your own mirror.
# NOTE: with no values file, the chart renders its DEFAULT image.registry, so this
# default must stay in sync with values.yaml's default image.registry — if you change
# the chart default, change this too (or always pass EXPECTED_REGISTRY explicitly).
EXPECTED_REGISTRY="${EXPECTED_REGISTRY:-nvcr.io}"
CHART_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HELM_BIN=${HELM:-helm}

# Build helm args as an array so values-file paths with spaces survive (a string +
# unquoted expansion would word-split them).
VALUES_ARGS=()
for f in "$@"; do
  VALUES_ARGS+=( -f "$f" )
done

# Render the kumo-helm chart WITH the passed values (so the registry under test is the
# one you'll actually deploy — do NOT override image.registry here). The other
# overrides only replace required placeholders; the check looks at image
# registries, not the storage or node-pool configuration.
# Capture stderr and the exit status so a render failure is reported as a render failure
# (exit 2) instead of being swallowed by `set -e`/pipefail and surfacing as a contentless
# exit 1 that looks like an image leak.
if ! RENDER=$("$HELM_BIN" template test "$CHART_DIR" \
      ${VALUES_ARGS[@]+"${VALUES_ARGS[@]}"} \
      --set global.tenant=test \
      --set image.tag=test-version \
      --set kumoConfig.storage.dataBucket=s3://test-bucket \
      --set kumoConfig.diskgraph.storageClass=ebs-gp3-xfs \
      --set kumoConfig.diskgraph.nodeTypes.test.ssd_gib=800 \
      --set postgres.authMode=aws_iam \
      --set postgres.host=state.example.us-west-2.rds.amazonaws.com \
      --set postgres.database=kumo \
      --set postgres.username=kumo \
      --set postgres.region=us-west-2 \
      --set-string 'serviceAccounts.control.annotations.eks\.amazonaws\.com/role-arn=arn:aws:iam::000000000000:role/kumo-control' \
      --set-string 'serviceAccounts.worker.annotations.eks\.amazonaws\.com/role-arn=arn:aws:iam::000000000000:role/kumo-worker' \
      --set-string 'serviceAccounts.controller.annotations.eks\.amazonaws\.com/role-arn=arn:aws:iam::000000000000:role/kumo-controller' 2>&1); then
  echo "ERROR: the chart failed to render — cannot validate images (this is NOT an"
  echo "       image-registry problem). helm output:" >&2
  echo "$RENDER" >&2
  exit 2
fi

# Read only INDENTED `image:` lines: real manifest image fields are nested under
# containers, so anchoring to `^\s+` keeps helm/K8s warning lines — which we merged in via
# `2>&1` above and which are NOT indented (e.g. `coalesce.go:NNN: ... .image: Not a table.`)
# — from being mis-parsed as bogus images and flipping a clean chart to a false FAIL. POSIX
# character classes keep this check portable to macOS, whose grep has no `-P` support.
IMAGES=$(printf '%s\n' "$RENDER" |
  sed -nE 's/^[[:space:]]+image:[[:space:]]*"?([^"[:space:]]+)"?.*$/\1/p' |
  sort -u || true)
if [[ -z "$IMAGES" ]]; then
  echo "No images found in the rendered output — nothing to validate."
  exit 0
fi

FAILED=0
echo "Checking rendered images..."
while IFS= read -r img; do
  # Skip template placeholders
  [[ "$img" == *"{image_id}"* ]] && continue
  [[ "$img" == *"{"* ]] && continue

  # Match on the registry followed by '/' so a look-alike host that merely PREFIXES the
  # expected registry (e.g. <expected>.evil.io/...) is not accepted as OK.
  if [[ "$img" == "$EXPECTED_REGISTRY/"* ]]; then
    echo "  OK:   $img"
  else
    echo "  FAIL: $img (not from $EXPECTED_REGISTRY)"
    FAILED=1
  fi
done <<< "$IMAGES"

echo ""
if [[ $FAILED -eq 1 ]]; then
  echo "ERROR: image(s) not from $EXPECTED_REGISTRY. Air-gapped install will fail."
  echo "       (set EXPECTED_REGISTRY=<your-registry> if you mirror to a different one.)"
  exit 1
fi
echo "All images point to $EXPECTED_REGISTRY."
exit 0
