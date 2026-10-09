#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ -n "${DISKGRAPH_DATA_ROOT:-}" ]]; then
  DEFAULT_ROOT="${DISKGRAPH_DATA_ROOT}"
elif [[ -d /mnt/localssd && -w /mnt/localssd ]]; then
  DEFAULT_ROOT="/mnt/localssd"
else
  DEFAULT_ROOT="/tmp"
fi
DEST_ROOT="${1:-${DEFAULT_ROOT%/}/datasets/mimic}"
S3_PREFIX="s3://kumo-public-datasets/MIMIC_IV_demo"

rm -rf "${DEST_ROOT}"
mkdir -p "${DEST_ROOT}"

aws s3 sync "${S3_PREFIX}" "${DEST_ROOT}"

echo "Next: ./diskgraph-infer/scripts/infer_config.sh ${DEST_ROOT}"
