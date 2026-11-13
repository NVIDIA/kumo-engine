#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reproducible production-engine sampling/IOPS sweep for an existing index.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
    cat <<'EOF'
Usage: scripts/bench_sampling_iops.sh INDEX_DIR

Runs dg-engine-bench in a temporary memory-capped user scope. Before every
point, clean cache pages belonging to INDEX_DIR are evicted with
POSIX_FADV_DONTNEED so sweep order does not bias the result.

Environment overrides:
  DG_BENCH_BINARY       Harness path (default: target/release/dg-engine-bench)
  DG_BENCH_CARGO_FEATURES Build features (default: jemalloc)
  DG_BENCH_SEED_TYPE    Seed node type (default: users)
  DG_BENCH_PLAN         Per-layer relation plan (default: 3-hop all alt edges;
                        use "default" for all relations at every layer)
  DG_BENCH_LAYERS       Layers used with DG_BENCH_PLAN=default (default: 3)
  DG_BENCH_FANOUT       Fanout used with DG_BENCH_PLAN=default (default: 10)
  DG_BENCH_FETCH_FEATURES 1 to materialize sampled-node features (default: 0)
  DG_BENCH_PKEY_E2E       1 to resolve real input PKs and return sampled PKs (default: 0)
  DG_BENCH_SEED_SOURCE_CONFIG Original local diskgraph_config.json used to
                        draw PKs uniformly across every source-table row
  DG_BENCH_SEED_RANK_COVERAGE Optional cumulative key:request fractions for
                        skewed roots (example: "0.0106:0.60,0.0389:0.80,1:1")
  DG_BENCH_SEED_RANK_PERMUTATION_SEED Rank-to-source-ID permutation seed
  DG_BENCH_THREADS      Space-separated sweep (default: "32 48 64 96")
  DG_BENCH_BACKENDS     Space-separated modes (default: "mmap buffered direct")
  DG_BENCH_IO_QD        Buffered per-ring depth (default: 128)
  DG_BENCH_IO_RINGS     Persistent owner rings (default: 8)
  DG_BENCH_DIRECT_IO_QD Direct per-ring depth (default: 32)
  DG_BENCH_DIRECT_IO_RINGS Direct owner rings (default: 32)
  DG_BENCH_IO_BATCH_US  Owner coalescing delay in microseconds (default: 0)
  DG_BENCH_IO_BATCH_MULT Owner batch target in ring-depth multiples (default: 8)
  DG_BENCH_IO_COALESCE_PAGES Maximum adjacent 4K pages per SQE (default: 16)
  DG_BENCH_IO_EAGER_NODE_PAGES Read complete records spanning at most N pages in the header wave (default: 0)
  DG_BENCH_DIRECT_CACHE_GB Direct cache payload cap (default: 0/cacheless)
  DG_BENCH_NUM_SEEDS    Measured seeds per point (default: 10000000)
  DG_BENCH_WARMUP_SEEDS Discarded warmup seeds (default: 2000000)
  DG_BENCH_BATCH_SIZE   Seeds per frontier request (default: 512)
  DG_BENCH_MAX_NEIGHBORS_PER_SEED Stop adding new nodes after this many
                        distinct neighbors per root (unset keeps fixed fanout BFS)
  DG_BENCH_MEMORY_MAX   systemd MemoryMax (default: 96G; empty disables scope)
  DG_BENCH_DEVICES      Devices counted by /proc/diskstats
  DG_BENCH_EVICT_CACHE  1 to evict index pages per point, 0 to preserve cache
  DG_BENCH_SEED         Deterministic RNG seed (default: 42)

Example for the alt-detection baseline:
  scripts/bench_sampling_iops.sh \
    /mnt/localssd/diskgraph-index/alt_detection_full_scale-zstd-1c1f7a9f-full-monitor
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

index_dir=$1
binary=${DG_BENCH_BINARY:-$repo_root/target/release/dg-engine-bench}
bench_features=${DG_BENCH_CARGO_FEATURES:-jemalloc}
seed_type=${DG_BENCH_SEED_TYPE:-users}
forward_plan='edge_user_ip_edges_src_to_dst:10,edge_user_mac_edges_src_to_dst:10,edge_user_email_edges_src_to_dst:10,edge_user_btid_edges_src_to_dst:10'
reverse_plan='edge_user_ip_edges_dst_to_src:10,edge_user_mac_edges_dst_to_src:10,edge_user_email_edges_dst_to_src:10,edge_user_btid_edges_dst_to_src:10'
default_plan="${forward_plan};${reverse_plan};${forward_plan}"
plan=${DG_BENCH_PLAN:-$default_plan}
layers=${DG_BENCH_LAYERS:-3}
fanout=${DG_BENCH_FANOUT:-10}
fetch_features=${DG_BENCH_FETCH_FEATURES:-0}
pkey_e2e=${DG_BENCH_PKEY_E2E:-0}
seed_source_config=${DG_BENCH_SEED_SOURCE_CONFIG:-}
seed_rank_coverage=${DG_BENCH_SEED_RANK_COVERAGE:-}
seed_rank_permutation_seed=${DG_BENCH_SEED_RANK_PERMUTATION_SEED:-0}
thread_list=${DG_BENCH_THREADS:-32 48 64 96}
backends=${DG_BENCH_BACKENDS:-mmap buffered direct}
io_qd=${DG_BENCH_IO_QD:-128}
io_rings=${DG_BENCH_IO_RINGS:-8}
direct_io_qd=${DG_BENCH_DIRECT_IO_QD:-32}
direct_io_rings=${DG_BENCH_DIRECT_IO_RINGS:-32}
io_batch_us=${DG_BENCH_IO_BATCH_US:-0}
io_batch_mult=${DG_BENCH_IO_BATCH_MULT:-8}
io_coalesce_pages=${DG_BENCH_IO_COALESCE_PAGES:-16}
io_eager_node_pages=${DG_BENCH_IO_EAGER_NODE_PAGES:-0}
direct_cache_gb=${DG_BENCH_DIRECT_CACHE_GB:-0}
num_seeds=${DG_BENCH_NUM_SEEDS:-10000000}
warmup_seeds=${DG_BENCH_WARMUP_SEEDS:-2000000}
batch_size=${DG_BENCH_BATCH_SIZE:-512}
max_neighbors_per_seed=${DG_BENCH_MAX_NEIGHBORS_PER_SEED:-}
memory_max=${DG_BENCH_MEMORY_MAX-96G}
devices=${DG_BENCH_DEVICES:-nvme0n1,nvme1n1,nvme2n1,nvme3n1}
evict_cache=${DG_BENCH_EVICT_CACHE:-1}
rng_seed=${DG_BENCH_SEED:-42}

[[ -f $index_dir/index_metadata.json ]] || {
    printf 'missing index metadata: %s/index_metadata.json\n' "$index_dir" >&2
    exit 2
}
if [[ $pkey_e2e == 1 && ! -f $seed_source_config ]]; then
    printf 'DG_BENCH_PKEY_E2E=1 requires a local DG_BENCH_SEED_SOURCE_CONFIG: %s\n' \
        "${seed_source_config:-<unset>}" >&2
    exit 2
fi

if [[ ! -x $binary ]]; then
    (cd "$repo_root" && cargo build --release -p diskgraph-engine --bin dg-engine-bench --features "$bench_features")
fi

for threads in $thread_list; do
  for backend in $backends; do
    point_io_qd=$io_qd
    point_io_rings=$io_rings
    if [[ $backend == direct ]]; then
        point_io_qd=$direct_io_qd
        point_io_rings=$direct_io_rings
    fi
    if [[ $evict_cache == 1 ]]; then
        python3 "$repo_root/scripts/evict_file_cache.py" "$index_dir"
    fi

    command=(
        "$binary"
        --index "$index_dir"
        --seed-type "$seed_type"
        --num-seeds "$num_seeds"
        --warmup-seeds "$warmup_seeds"
        --batch-size "$batch_size"
        --threads "$threads"
        --engine-threads "$threads"
        --prefetch off
        --devices "$devices"
        --seed "$rng_seed"
        --backend "$backend"
        --io-qd "$point_io_qd"
        --io-rings "$point_io_rings"
        --io-batch-us "$io_batch_us"
        --io-batch-mult "$io_batch_mult"
        --io-coalesce-pages "$io_coalesce_pages"
        --io-eager-node-pages "$io_eager_node_pages"
        --direct-cache-gb "$direct_cache_gb"
        --verify
        --prefault-offsets
    )
    if [[ $plan == default ]]; then
        command+=(--layers "$layers" --fanout "$fanout")
    else
        command+=(--plan "$plan")
    fi
    if [[ $fetch_features == 1 ]]; then
        command+=(--fetch-features)
    fi
    if [[ $pkey_e2e == 1 ]]; then
        command+=(--pkey-e2e --seed-source-config "$seed_source_config")
        if [[ -n $seed_rank_coverage ]]; then
            command+=(
                --seed-rank-coverage "$seed_rank_coverage"
                --seed-rank-permutation-seed "$seed_rank_permutation_seed"
            )
        fi
    fi
    if [[ -n $max_neighbors_per_seed ]]; then
        command+=(--max-neighbors-per-seed "$max_neighbors_per_seed")
    fi

    printf '\n### sampling point: backend=%s threads=%s rings=%s qd/ring=%s batch-us=%s batch-mult=%s coalesce-pages=%s eager-pages=%s cache-gb=%s ###\n' \
        "$backend" "$threads" "$point_io_rings" "$point_io_qd" "$io_batch_us" "$io_batch_mult" \
        "$io_coalesce_pages" "$io_eager_node_pages" "$direct_cache_gb"
    if [[ -n $memory_max ]]; then
        systemd-run --user --scope -p "MemoryMax=$memory_max" -p MemorySwapMax=0 "${command[@]}"
    else
        "${command[@]}"
    fi
  done
done
