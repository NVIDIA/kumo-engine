#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

TENANT=$1

if [ -z "$TENANT" ]; then
    echo "Usage: npm run remote -- <tenant>"
    echo "Example: npm run remote -- ui-test-1.staging"
    exit 1
fi

bash scripts/setup-tsconfig.sh --reset
UI_REMOTE=true npm run update-env
UI_REMOTE=true API_TARGET="https://${TENANT}.kumoai.cloud/internal" vite
