#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds one BCCC link-prediction tier if it is missing, then runs matrix row 14
# on it, unattended, sampling free disk and RSS throughout.
#
#   scripts/kaggle/bccc_lp/run_bccc_lp_tier.sh <scale> [extra run args...]
#
# `<scale>` is lp_tiny | lp_smoke | lp_small | lp_part | lp_full -- a hash
# sample of *source hosts* carried with all of their flows, because row 14's
# entity is `hosts.host_id` and its target is the set of destinations a host
# reaches, so a flow sample would divide the label (NOAA S5, NYC S9).
#
# Three things this wrapper exists for beyond sequencing:
#
# * **The disk and RSS sample.** DiskGraph ingest is memory-bound, not
#   disk-bound (NYC S9: a 255M-row ingest was OOM-killed at 13.5 GB RSS on this
#   15 GB box), and the peak is only observable while the run is in flight.
# * **A wait-loop that cannot be fooled.** Row 4 lost a run to a loop that
#   matched the benign `sklearn` traceback that every run emits. Nothing here
#   greps the log for failure: the run's own exit status is the verdict, and
#   `END TO END OK` on the last line is the confirmation.
# * **`PYTHONPATH`.** Without it `kumoai_studio` resolves to the fat wheel and the run
#   dies in the first second on a stale backend API.
set -u

SCALE=${1:?usage: run_bccc_lp_tier.sh <scale> [extra args...]}
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../_env.sh"
TAG=${TAG:-}
BUILD_LOG=/tmp/bccc_lp_build_${SCALE}.log
RUN_LOG=/tmp/bccc_lp_run_${SCALE}${TAG}.log
DISK_LOG=/tmp/bccc_lp_disk_${SCALE}${TAG}.log

if ! grep -q '^held out:' "$BUILD_LOG" 2>/dev/null; then
    "$KUMO_SDK_PYTHON" -u scripts/kaggle/bccc_lp/build_bccc_lp_tables.py \
        --scale "$SCALE" > "$BUILD_LOG" 2>&1
    if ! grep -q '^held out:' "$BUILD_LOG"; then
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

"$KUMO_SDK_PYTHON" -u scripts/kaggle/bccc_lp/run_bccc_lp_local.py \
    --scale "$SCALE" --no-publish-snapshots \
    "$@" > "$RUN_LOG" 2>&1
STATUS=$?

kill "$SAMPLER" 2>/dev/null
echo "row 14 at $SCALE exited with status $STATUS"
tail -1 "$RUN_LOG"
exit "$STATUS"
