#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Single source of truth helper for the Makefile to resolve KUMO_IMAGE_TAG for
# running the build sub-make, preflight checks, and compose up.
#
# Precedence (highest first):
#   1. KUMO_IMAGE_TAG already set in the environment
#   2. KUMO_IMAGE_TAG in the override file (.env.compose-local.local)
#   3. KUMO_IMAGE_TAG in the base env file (.env.compose-local)
#   4. fallback: latest
#
# Usage: resolve-image-tag.sh <base-env-file> [override-env-file ...]
set -euo pipefail

tag="${KUMO_IMAGE_TAG:-}"
if [[ -z "$tag" && $# -gt 0 ]]; then
  # Only feed files that actually exist (e.g. handle missing override file)
  files=()
  for f in "$@"; do
    [[ -f "$f" ]] && files+=("$f")
  done

  if (( ${#files[@]} > 0 )); then
    tag="$(awk -F= '/^KUMO_IMAGE_TAG=/{v=$2} END{print v}' "${files[@]}" 2>/dev/null || true)"
  fi
fi

printf '%s\n' "${tag:-latest}"
