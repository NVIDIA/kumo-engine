#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${EXPECTED_VERSION:?EXPECTED_VERSION is required}"

wheels=("$@")
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "Expected exactly one wheel, found ${#wheels[@]}." >&2
  exit 1
fi

python -m twine check "${wheels[0]}"
python "$SCRIPT_DIR/release_artifact.py" \
  --expected-version "$EXPECTED_VERSION" \
  "${wheels[0]}"
sha256sum "${wheels[0]}"
