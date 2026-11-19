#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# These Linux tests require working io_uring and O_DIRECT. A denied operation
# must fail this check; fork cases need separate processes with fresh globals.
set -euo pipefail

diskgraph_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$diskgraph_root"

export DG_DIRECT_CACHE_GB=1
export DG_IO_QD=4
export DG_IO_RINGS=2

run_case() {
    local test_name="$1"
    shift
    local output
    if ! output=$(cargo test --locked -p diskgraph-index "$@" "$test_name" \
        -- --ignored --exact --nocapture 2>&1); then
        printf '%s\n' "$output" >&2
        return 1
    fi
    printf '%s\n' "$output"
    # Cargo exits successfully when an exact test name no longer exists.
    if ! grep -Fq 'test result: ok. 1 passed; 0 failed; 0 ignored;' <<< "$output"; then
        printf 'Expected one executed native test: %s\n' "$test_name" >&2
        return 1
    fi
}

for test_name in \
    native_roundtrip \
    canceled_batches_release_reservations \
    read_error_recovers_cache \
    fork_before_first_read \
    fork_after_first_read \
    fork_with_populated_cache \
    fork_handles_inherited_work; do
    run_case "$test_name" --test page_io_readiness
done

run_case page_io::readiness_tests::fork_during_initialization --lib
run_case page_io::readiness_tests::fork_with_locked_cache --lib
run_case backends_checksum_identical_bytes --test page_bench_checksum
