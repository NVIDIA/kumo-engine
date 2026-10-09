#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds one BCCC tier and runs it, unattended, sampling free disk throughout.
#
# Both halves are long enough that neither wants a babysitter, and the run
# cannot start until the tables exist. The disk sample is the point of the
# wrapper as much as the sequencing: the ingest peak is well above the resting
# index size and is only observable while the run is in flight -- and this
# fixture is 315 feature columns wide, so the index is far larger per row than
# either of the previous two.
#
#   scripts/kaggle/bccc/run_bccc_tier.sh <scale> [truth-sample-pct]
set -u

SCALE=${1:?usage: run_bccc_tier.sh <scale> [truth-sample-pct]}
TRUTH_PCT=${2:-100}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../_env.sh"
BUILD_LOG=/tmp/bccc_build_${SCALE}.log
RUN_LOG=/tmp/bccc_run_${SCALE}.log
DISK_LOG=/tmp/bccc_disk_${SCALE}.log

"$KUMO_SDK_PYTHON" -u scripts/kaggle/bccc/build_bccc_tables.py \
    --scale "$SCALE" > "$BUILD_LOG" 2>&1
# The build prints a per-class table and then a TOTAL line; it exits non-zero
# on a degenerate slice, which for a multiclass fixture means "some class did
# not survive the sample" as often as it means "nothing was held out".
if ! grep -q '^TOTAL' "$BUILD_LOG"; then
    echo "build did not finish cleanly; not starting the run" >&2
    tail -20 "$BUILD_LOG" >&2
    exit 1
fi

# One sample a minute is enough to catch the ingest peak, which builds over
# minutes to hours rather than spiking.
(
    while true; do
        echo "$(date +%H:%M:%S) $(df -B1 --output=used,avail /mnt/localssd |
            tail -1) rss=$(ps -eo rss,comm --sort=-rss | head -2 | tail -1)"
        sleep 60
    done
) > "$DISK_LOG" 2>&1 &
SAMPLER=$!

"$KUMO_SDK_PYTHON" -u scripts/kaggle/bccc/run_bccc_local.py \
    --scale "$SCALE" --no-publish-snapshots \
    --truth-sample-pct "$TRUTH_PCT" > "$RUN_LOG" 2>&1
STATUS=$?

kill "$SAMPLER" 2>/dev/null
echo "run exited with status $STATUS"
exit "$STATUS"
