#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <terraform.tfvars>" >&2
    exit 2
fi
terraform -chdir=terraform destroy -var-file="$1"
