#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Real-primary-key benchmark for the original hotspot-based pruned Pinterest graph.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

if [[ $# != 2 ]]; then
    printf 'Usage: %s INDEX_DIR SOURCE_DISKGRAPH_CONFIG\n' "$0" >&2
    exit 2
fi

index_dir=$1
source_config=$2

# Historical pin-uniform traversal: pin fans out to the four neighboring
# entity families, then those entities follow the reverse directions to pins.
forward='edge_board_pin_dst_to_src:10,edge_pin_pin_src_to_dst:10,edge_query_pin_dst_to_src:10,edge_pin_item_src_to_dst:10'
reverse='edge_board_pin_src_to_dst:10,edge_pin_pin_dst_to_src:10,edge_query_pin_src_to_dst:10,edge_pin_item_dst_to_src:10'

export DG_BENCH_SEED_TYPE=${DG_BENCH_SEED_TYPE:-pin}
export DG_BENCH_PLAN=${DG_BENCH_PLAN:-"$forward;$reverse"}
export DG_BENCH_PKEY_E2E=${DG_BENCH_PKEY_E2E:-1}
export DG_BENCH_SEED_SOURCE_CONFIG=${DG_BENCH_SEED_SOURCE_CONFIG:-$source_config}
export DG_BENCH_BATCH_SIZE=${DG_BENCH_BATCH_SIZE:-512}
export DG_BENCH_THREADS=${DG_BENCH_THREADS:-160}
export DG_BENCH_BACKENDS=${DG_BENCH_BACKENDS:-"mmap direct"}
export DG_BENCH_DIRECT_IO_RINGS=${DG_BENCH_DIRECT_IO_RINGS:-32}
export DG_BENCH_DIRECT_IO_QD=${DG_BENCH_DIRECT_IO_QD:-32}
export DG_BENCH_IO_COALESCE_PAGES=${DG_BENCH_IO_COALESCE_PAGES:-16}
export DG_BENCH_DIRECT_CACHE_GB=${DG_BENCH_DIRECT_CACHE_GB:-320}
export DG_BENCH_MEMORY_MAX=${DG_BENCH_MEMORY_MAX:-500G}
export DG_BENCH_WARMUP_SEEDS=${DG_BENCH_WARMUP_SEEDS:-20000000}
export DG_BENCH_NUM_SEEDS=${DG_BENCH_NUM_SEEDS:-10000000}
export DG_BENCH_EVICT_CACHE=${DG_BENCH_EVICT_CACHE:-1}

exec "$repo_root/scripts/bench_sampling_iops.sh" "$index_dir"
