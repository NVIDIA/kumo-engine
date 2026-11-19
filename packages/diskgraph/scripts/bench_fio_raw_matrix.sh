#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Read-only raw md/per-drive IOPS matrix with enough workers and queue depth.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
targets=${FIO_TARGETS:-/dev/md127 /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1}
qd_points=${FIO_QD_POINTS:-64 128 256 512 1024 2048}
workers=${FIO_WORKERS:-32}

if [[ ${FIO_ALLOW_RAW_READ:-0} != 1 ]]; then
    printf 'Set FIO_ALLOW_RAW_READ=1 to acknowledge read-only raw-device tests.\n' >&2
    exit 2
fi

for target in $targets; do
    [[ -b $target ]] || {
        printf 'skipping missing block device: %s\n' "$target" >&2
        continue
    }
    for aggregate_qd in $qd_points; do
        depth=$(( (aggregate_qd + workers - 1) / workers ))
        actual_qd=$(( workers * depth ))
        printf '\n### target=%s requested_qd=%s workers=%s depth=%s actual_qd=%s ###\n' \
            "$target" "$aggregate_qd" "$workers" "$depth" "$actual_qd"
        FIO_ALLOW_RAW_READ=1 \
        FIO_WORKERS=$workers \
        FIO_IODEPTH=$depth \
            "$repo_root/scripts/bench_fio_iops.sh" "$target"
    done
done
