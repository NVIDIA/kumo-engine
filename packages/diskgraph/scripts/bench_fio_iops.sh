#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Saturating, read-only 4 KiB random-read fio baseline.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/bench_fio_iops.sh TARGET

TARGET may be a regular file, a directory of edge files, or a raw block device. Raw-device tests require
FIO_ALLOW_RAW_READ=1 as an explicit safety acknowledgement. Every test is
read-only and uses 32 workers x iodepth 32 (aggregate QD1024) by default.

Environment overrides:
  FIO_WORKERS       Worker count (default: 32)
  FIO_IODEPTH       Per-worker depth (default: 32)
  FIO_RUNTIME       Measured seconds (default: 30)
  FIO_RAMP_TIME     Warmup seconds (default: 5)
  FIO_ENGINE        I/O engine (default: io_uring)
  FIO_FILE_COUNT    Largest files used when TARGET is a directory (default: 128)
  FIO_ALLOW_RAW_READ=1  Required when TARGET is a block device

Examples:
  scripts/bench_fio_iops.sh /mnt/localssd/index/nodes/type/shard-00000/edges.bin
  FIO_ALLOW_RAW_READ=1 scripts/bench_fio_iops.sh /dev/md127
  FIO_ALLOW_RAW_READ=1 scripts/bench_fio_iops.sh /dev/nvme0n1
EOF
}

[[ ${1:-} != -h && ${1:-} != --help ]] || {
    usage
    exit 0
}
[[ $# == 1 ]] || {
    usage >&2
    exit 2
}

target=$1
workers=${FIO_WORKERS:-32}
iodepth=${FIO_IODEPTH:-32}
runtime=${FIO_RUNTIME:-30}
ramp_time=${FIO_RAMP_TIME:-5}
engine=${FIO_ENGINE:-io_uring}

[[ -e $target ]] || {
    printf 'target does not exist: %s\n' "$target" >&2
    exit 2
}
if [[ -b $target && ${FIO_ALLOW_RAW_READ:-0} != 1 ]]; then
    printf 'refusing raw block-device access without FIO_ALLOW_RAW_READ=1\n' >&2
    exit 2
fi
if [[ ! -f $target && ! -b $target ]]; then
    if [[ ! -d $target ]]; then
        printf 'target is not a regular file, directory, or block device: %s\n' "$target" >&2
        exit 2
    fi
fi

fio_target=$target
if [[ -d $target ]]; then
    file_count=${FIO_FILE_COUNT:-128}
    mapfile -t selected_files < <(
        find "$target" -type f -name edges.bin -printf '%s\t%p\n' |
            sort -nr |
            sed -n "1,${file_count}p" |
            cut -f2-
    )
    if (( ${#selected_files[@]} == 0 )); then
        printf 'no edges.bin files under directory: %s\n' "$target" >&2
        exit 2
    fi
    printf 'selected %s largest edge files\n' "${#selected_files[@]}"
    jobfile=$(mktemp --tmpdir dg-fio-multifile.XXXXXX.fio)
    trap 'rm -f -- "$jobfile"' EXIT
    {
        printf '[global]\n'
        printf 'rw=randread\nbs=4k\ndirect=1\nallow_file_create=0\n'
        printf 'ioengine=%s\niodepth=%s\ntime_based=1\n' "$engine" "$iodepth"
        printf 'runtime=%s\nramp_time=%s\ngroup_reporting=1\n' "$runtime" "$ramp_time"
        printf 'norandommap=1\nrandrepeat=0\nfile_service_type=random\n'
        for ((job = 0; job < workers; job++)); do
            job_files=()
            for ((idx = job; idx < ${#selected_files[@]}; idx += workers)); do
                job_files+=("${selected_files[idx]}")
            done
            (( ${#job_files[@]} > 0 )) || continue
            joined=$(IFS=:; printf '%s' "${job_files[*]}")
            printf '\n[job-%03d]\nfilename=%s\n' "$job" "$joined"
        done
    } >"$jobfile"
    exec fio "$jobfile"
fi

aggregate_qd=$((workers * iodepth))
printf 'read-only fio: target=%s workers=%s iodepth=%s aggregate_qd=%s\n' \
    "$target" "$workers" "$iodepth" "$aggregate_qd"

exec fio \
    --name=4k-randread-qd"$aggregate_qd" \
    --filename="$fio_target" \
    --file_service_type=random \
    --readonly \
    --rw=randread \
    --bs=4k \
    --direct=1 \
    --ioengine="$engine" \
    --numjobs="$workers" \
    --iodepth="$iodepth" \
    --time_based \
    --runtime="$runtime" \
    --ramp_time="$ramp_time" \
    --group_reporting \
    --norandommap=1 \
    --randrepeat=0
