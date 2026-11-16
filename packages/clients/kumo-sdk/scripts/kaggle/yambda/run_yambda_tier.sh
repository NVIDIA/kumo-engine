#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds one Yambda tier if it is missing, then runs one matrix row on it,
# unattended, sampling free disk and RSS throughout.
#
#   scripts/kaggle/yambda/run_yambda_tier.sh <row: 6|7|10|13|15> <scale> [args...]
#
# `<scale>` is y_tiny | y_smoke | y_small | y_part | y_pool -- a hash sample of
# *users* carried with all of their interactions, because four of the five rows
# have `users.user_id` as the entity and an aggregate over `interactions` as the
# target, so an interaction sample would divide the label (NOAA S5, BCCC S6).
#
# Row 10 additionally needs `--holdout-pct 20` at build time (its target is a
# static column, so its prediction table filters on `IS NULL`), and row 13 needs
# `--lp-holdout-pct 20` (its target is "every item this user ever touched",
# which is exactly what the graph contains, so scoring it without a holdout
# measures memorisation).  Rows 7 and 15 need neither: a temporal aggregate
# target gets no prediction-mode filter at all and its truth is the future.
#
# Three things this wrapper exists for beyond sequencing:
#
# * **The disk and RSS sample.** DiskGraph ingest is memory-bound, not
#   disk-bound (NYC S9: a 255M-row ingest was OOM-killed at 13.5 GB RSS on this
#   15 GB box), and the peak is only observable while the run is in flight.
# * **A wait-loop that cannot be fooled.** Row 4 lost a run to a loop that
#   matched the benign `sklearn` traceback every run emits.  Nothing here greps
#   the log for failure: the run's exit status is the verdict and
#   `END TO END OK` on the last line is the confirmation.
# * **`PYTHONPATH`.** Without it `kumoai_studio` resolves to the fat wheel and the run
#   dies in the first second on a stale backend API.
set -u

ROW=${1:?usage: run_yambda_tier.sh <row> <scale> [extra args...]}
SCALE=${2:?usage: run_yambda_tier.sh <row> <scale> [extra args...]}
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../_env.sh"
TAG=${TAG:-}
BUILD_ARGS=""
BUILD_KEY=$SCALE
case "$ROW" in
    10) BUILD_ARGS="--holdout-pct 20";    BUILD_KEY="${SCALE}_h20" ;;
    13) BUILD_ARGS="--lp-holdout-pct 20"; BUILD_KEY="${SCALE}_lp"  ;;
esac
BUILD_LOG=/tmp/yambda_build_${BUILD_KEY}.log
RUN_LOG=/tmp/yambda_run_r${ROW}_${SCALE}${TAG}.log
DISK_LOG=/tmp/yambda_disk_r${ROW}_${SCALE}${TAG}.log

if ! grep -q '^TOTAL' "$BUILD_LOG" 2>/dev/null; then
    # shellcheck disable=SC2086
    "$KUMO_SDK_PYTHON" -u scripts/kaggle/yambda/build_yambda_tables.py \
        --scale "$SCALE" $BUILD_ARGS > "$BUILD_LOG" 2>&1
    if ! grep -q '^TOTAL' "$BUILD_LOG"; then
        echo "build did not finish cleanly; not starting the run" >&2
        tail -20 "$BUILD_LOG" >&2
        exit 1
    fi
fi

# `used avail` in bytes plus the largest resident process, every 30 s.
(
    while true; do
        echo "$(date +%H:%M:%S) $(df -B1 --output=used,avail /mnt/localssd |
            tail -1) rss=$(ps -eo rss,comm --sort=-rss | head -2 | tail -1)"
        sleep 30
    done
) > "$DISK_LOG" 2>&1 &
SAMPLER=$!

"$KUMO_SDK_PYTHON" -u scripts/kaggle/yambda/run_yambda_local.py \
    --row "$ROW" --scale "$SCALE" --no-publish-snapshots \
    "$@" > "$RUN_LOG" 2>&1
STATUS=$?

kill "$SAMPLER" 2>/dev/null
echo "row $ROW at $SCALE exited with status $STATUS"
tail -1 "$RUN_LOG"
exit "$STATUS"
