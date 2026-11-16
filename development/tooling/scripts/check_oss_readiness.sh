#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

diff_args=()
if (( $# != 0 )); then
    if (( $# != 3 )) || [[ "$1" != "--diff" ]]; then
        echo "Usage: $0 [--diff BASE COMMIT]" >&2
        exit 2
    fi
    diff_args=("$@")
fi

overall_status=0
"$SCRIPT_DIR/check_public_references.py" "${diff_args[@]}" \
    || overall_status=1
"$SCRIPT_DIR/check_dead_code.sh" \
    || overall_status=1
exit "$overall_status"
