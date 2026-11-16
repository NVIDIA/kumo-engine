#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds one NOAA tier if it is missing, then runs one matrix row on it,
# unattended, sampling free disk throughout.
#
#   scripts/kaggle/noaa/run_noaa_tier.sh <row> <scale> [extra run args...]
#
# `<row>` is 5, 8 or 11. The tier tables are shared by all three, so the build
# is skipped when the run log for another row on the same scale shows the tables
# already exist -- rebuilding them would be several minutes of Snowflake for
# nothing, and would invalidate the "same rows, one word different" property
# that makes the row 5 / row 8 comparison worth anything.
#
# Two things this wrapper exists for beyond sequencing:
#
# * **The disk sample.** The ingest peak sits well above the resting index size
#   and is only observable while the run is in flight. One sample a minute is
#   enough; it builds over minutes, it does not spike.
# * **A wait-loop that cannot be fooled.** Row 4 lost a run to a loop that
#   matched the benign `sklearn` traceback that every run emits. Nothing here
#   greps the log for failure: the run's own exit status is the verdict, and
#   `END TO END OK` on the last line is the confirmation.
set -u

ROW=${1:?usage: run_noaa_tier.sh <row: 5|8|11> <scale> [extra args...]}
SCALE=${2:?usage: run_noaa_tier.sh <row: 5|8|11> <scale> [extra args...]}
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../_env.sh"
BUILD_LOG=/tmp/noaa_build_${SCALE}.log
RUN_LOG=/tmp/noaa_run_r${ROW}_${SCALE}.log
DISK_LOG=/tmp/noaa_disk_r${ROW}_${SCALE}.log

# The build is idempotent and prints a TOTAL line last; if a previous build of
# this scale already finished, it is not repeated.
if ! grep -q '^TOTAL' "$BUILD_LOG" 2>/dev/null; then
    "$KUMO_SDK_PYTHON" -u scripts/kaggle/noaa/build_noaa_tables.py \
        --scale "$SCALE" > "$BUILD_LOG" 2>&1
    if ! grep -q '^TOTAL' "$BUILD_LOG"; then
        echo "build did not finish cleanly; not starting the run" >&2
        tail -20 "$BUILD_LOG" >&2
        exit 1
    fi
fi

# `used avail` in bytes plus the largest resident process, once a minute.
(
    while true; do
        echo "$(date +%H:%M:%S) $(df -B1 --output=used,avail /mnt/localssd |
            tail -1) rss=$(ps -eo rss,comm --sort=-rss | head -2 | tail -1)"
        sleep 60
    done
) > "$DISK_LOG" 2>&1 &
SAMPLER=$!

"$KUMO_SDK_PYTHON" -u scripts/kaggle/noaa/run_noaa_local.py \
    --row "$ROW" --scale "$SCALE" --no-publish-snapshots \
    "$@" > "$RUN_LOG" 2>&1
STATUS=$?

kill "$SAMPLER" 2>/dev/null
echo "row $ROW at $SCALE exited with status $STATUS"
tail -1 "$RUN_LOG"
exit "$STATUS"
