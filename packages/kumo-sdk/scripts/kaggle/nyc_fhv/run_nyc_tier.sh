#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds one NYC FHV tier if it is missing, then runs one matrix row on it,
# unattended, sampling free disk throughout.
#
#   scripts/kaggle/nyc_fhv/run_nyc_tier.sh <row> <scale> [extra run args...]
#
# `<row>` is 9, 12 or 16, and the two tier families are named apart because the
# rows do not share one:
#
#   row 9        r9tiny | r9smoke | r9small | r9med | r9big
#                a uniform hash sample of trips across the whole 2019-2022 span,
#                built with a 20% holdout because `trip_seconds` is a static
#                column target.
#   rows 12, 16  z30 | z90 | z365 | zfull
#                every trip inside a contiguous window from 2019-02-01, because
#                both targets are aggregates over a zone's trips and a row
#                sample would divide the label.
#
# Two things this wrapper exists for beyond sequencing:
#
# * **The disk sample.** The ingest peak sits well above the resting index size
#   and is only observable while the run is in flight.
# * **A wait-loop that cannot be fooled.** Row 4 lost a run to a loop that
#   matched the benign `sklearn` traceback that every run emits. Nothing here
#   greps the log for failure: the run's own exit status is the verdict, and
#   `END TO END OK` on the last line is the confirmation.
set -u

ROW=${1:?usage: run_nyc_tier.sh <row: 9|12|16> <scale> [extra args...]}
SCALE=${2:?usage: run_nyc_tier.sh <row: 9|12|16> <scale> [extra args...]}
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../_env.sh"
TAG=${TAG:-}
BUILD_LOG=/tmp/nyc_build_${SCALE}.log
RUN_LOG=/tmp/nyc_run_r${ROW}_${SCALE}${TAG}.log
DISK_LOG=/tmp/nyc_disk_r${ROW}_${SCALE}${TAG}.log

# Row 9 needs the holdout; rows 12 and 16 must not have one -- a temporal
# aggregate target gets no prediction-mode filter at all (NOAA S2), so blanking
# trip_seconds would only remove a feature.
if [ "$ROW" = "9" ]; then
    BUILD_ARGS="--holdout-pct 20"
    BUILD_LOG=/tmp/nyc_build_${SCALE}_h20.log
else
    BUILD_ARGS=""
fi

if ! grep -q '^TOTAL' "$BUILD_LOG" 2>/dev/null; then
    # shellcheck disable=SC2086
    "$KUMO_SDK_PYTHON" -u scripts/kaggle/nyc_fhv/build_nyc_tables.py \
        --scale "$SCALE" $BUILD_ARGS > "$BUILD_LOG" 2>&1
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

"$KUMO_SDK_PYTHON" -u scripts/kaggle/nyc_fhv/run_nyc_local.py \
    --row "$ROW" --scale "$SCALE" --no-publish-snapshots \
    "$@" > "$RUN_LOG" 2>&1
STATUS=$?

kill "$SAMPLER" 2>/dev/null
echo "row $ROW at $SCALE exited with status $STATUS"
tail -1 "$RUN_LOG"
exit "$STATUS"
