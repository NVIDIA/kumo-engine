#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Resolve the SDK and Python environment from either the Kumo monorepo or a
# historical standalone checkout. SDK launchers source this file before using
# package-relative paths.
KUMO_SDK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KUMO_WORKSPACE_ROOT="$(git -C "$KUMO_SDK_ROOT" rev-parse --show-toplevel)"

if [[ -d "$KUMO_WORKSPACE_ROOT/packages/kumo-sdk" ]]; then
    KUMO_SDK_ROOT="$KUMO_WORKSPACE_ROOT/packages/kumo-sdk"
fi

if [[ -z "${KUMO_SDK_PYTHON:-}" ]]; then
    if [[ -x "$KUMO_WORKSPACE_ROOT/.venv/bin/python" ]]; then
        KUMO_SDK_PYTHON="$KUMO_WORKSPACE_ROOT/.venv/bin/python"
    else
        KUMO_SDK_PYTHON="$KUMO_SDK_ROOT/.venv/bin/python"
    fi
fi

if [[ ! -x "$KUMO_SDK_PYTHON" ]]; then
    echo "SDK Python not found at $KUMO_SDK_PYTHON" >&2
    echo "Run 'mise exec -- make install-venv' from $KUMO_WORKSPACE_ROOT or set KUMO_SDK_PYTHON." >&2
    return 1
fi

cd "$KUMO_SDK_ROOT" || return 1
export KUMO_SDK_PYTHON KUMO_SDK_ROOT KUMO_WORKSPACE_ROOT
export PYTHONPATH="$KUMO_SDK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
