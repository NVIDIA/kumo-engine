#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if (( $# != 0 )); then
    echo "Usage: $0 (scans the full configured Python source tree)" >&2
    exit 2
fi

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
if command -v mise >/dev/null 2>&1; then
    uvx_command=(mise exec -- uvx)
elif command -v uvx >/dev/null 2>&1; then
    uvx_command=(uvx)
else
    echo "uvx is required. Install the repository tools with 'mise install'." >&2
    exit 2
fi

exec "${uvx_command[@]}" --from vulture==2.14 vulture
