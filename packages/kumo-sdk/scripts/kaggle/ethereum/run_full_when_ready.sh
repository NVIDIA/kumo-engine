#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Drives the full-scale (380M transaction) run unattended.
#
# The Snowflake CTAS and the local run are each long enough that neither wants
# a babysitter, and the run cannot start until the table exists. This waits for
# the build, launches the run on the instance-store NVMe, and samples free disk
# throughout -- the peak is the number that decides whether the full fixture
# fits, and it is only observable while the run is in flight.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../_env.sh"
BUILD_LOG=/tmp/eth_build_full.log
RUN_LOG=/tmp/eth_full.log
DISK_LOG=/tmp/eth_full_disk.log

# The build was launched separately; wait for it rather than re-running it.
while pgrep -f 'build_ethereum_tables.py --scale full' >/dev/null; do
    sleep 60
done

if ! grep -q 'held out for scoring' "$BUILD_LOG"; then
    echo "build did not finish cleanly; not starting the run" >&2
    tail -5 "$BUILD_LOG" >&2
    exit 1
fi

# One sample a minute is enough to catch the ingest peak, which builds over
# hours rather than spiking.
(
    while true; do
        echo "$(date +%H:%M:%S) $(df -B1 --output=used,avail /mnt/localssd |
            tail -1)"
        sleep 60
    done
) > "$DISK_LOG" 2>&1 &
SAMPLER=$!

"$KUMO_SDK_PYTHON" -u scripts/kaggle/ethereum/run_ethereum_local.py \
    --scale full --no-publish-snapshots > "$RUN_LOG" 2>&1
STATUS=$?

kill "$SAMPLER" 2>/dev/null
echo "run exited with status $STATUS"
exit "$STATUS"
