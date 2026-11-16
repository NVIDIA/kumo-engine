// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DISKGRAPH_VERIFY_INDEX=<index dir> [DISKGRAPH_VERIFY_SEED_TYPE=customers]
//! [DISKGRAPH_VERIFY_ANCHOR=<unix s>] cargo test --release -p diskgraph-engine
//! --test per_seed_recording_real_graph -- --ignored

use diskgraph_core::sampling::*;
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_engine::output::InMemoryOutput;
use diskgraph_engine::state::StatePool;
use std::collections::HashMap;

const PLAN6: [u32; 6] = [24, 8, 6, 8, 6, 8];

fn request(seed_type: &str, seeds: &[u64], anchors: &[i64], hops: usize) -> BatchSamplingRequest {
    let layers = PLAN6[..hops]
        .iter()
        .map(|&fanout| LayerSpec {
            fanouts: HashMap::new(),
            default_fanout: Some(FanoutSpec {
                fanout,
                mode: SamplingMode::TemporalMostRecent,
                absent_key_policy: AbsentKeyPolicy::UniformFallback,
            }),
        })
        .collect();
    BatchSamplingRequest {
        dataset_id: diskgraph_core::dataset::DatasetId::nil(),
        seed_nodes: [(seed_type.to_string(), seeds.to_vec())].into(),
        seed_timestamps: Some([(seed_type.to_string(), anchors.to_vec())].into()),
        seed_timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
        plan: SamplingPlan { layers },
        flags: SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: false,
            swap_edge_direction: true,
        },
    }
}

#[test]
#[ignore = "needs DISKGRAPH_VERIFY_INDEX"]
fn per_seed_recording_matches_legacy_on_real_graph() {
    let index = std::env::var("DISKGRAPH_VERIFY_INDEX").expect("DISKGRAPH_VERIFY_INDEX");
    let seed_type =
        std::env::var("DISKGRAPH_VERIFY_SEED_TYPE").unwrap_or_else(|_| "customers".into());
    let anchor: i64 =
        std::env::var("DISKGRAPH_VERIFY_ANCHOR").map_or(1_734_566_399, |s| s.parse().unwrap());
    let mut engine = DiskGraphEngine::load(std::path::Path::new(&index), Some(0), Some(4)).unwrap();
    let (lo, hi) = engine.node_type_id_range(&seed_type).unwrap();
    let (batch, batches) = (512usize, 6usize);
    let n = batch * batches;
    let seeds: Vec<u64> = (0..n as u64)
        .map(|i| lo + i * (hi - lo) / n as u64)
        .collect();
    let anchors: Vec<i64> = (0..n)
        .map(|i| {
            if (i / batch) % 2 == 0 {
                anchor
            } else {
                anchor - (i as i64 * 7_919) % 31_536_000
            }
        })
        .collect();

    for hops in 2..=6 {
        let mut runs = Vec::new();
        for legacy in [false, true, true] {
            engine.legacy_recording = legacy;
            let pool = StatePool::new();
            let mut out = Vec::new();
            for (s, a) in seeds.chunks(batch).zip(anchors.chunks(batch)) {
                let mut state = pool.acquire();
                let mut output = InMemoryOutput::new();
                engine
                    .sample(&request(&seed_type, s, a, hops), &mut state, &mut output)
                    .unwrap();
                let mut types: Vec<_> = state.type_states.iter().collect();
                types.sort_by_key(|(t, _)| **t);
                let nodes: Vec<Vec<(u64, u32, u32, u32)>> = types
                    .iter()
                    .map(|(_, ts)| {
                        ts.nodes
                            .iter()
                            .map(|r| (r.global_id, r.seed_index, r.local_id, r.seed_type_id))
                            .collect()
                    })
                    .collect();
                let frontier: Vec<Vec<u64>> = types
                    .iter()
                    .map(|(_, ts)| ts.next_frontier.iter().map(|r| r.global_id).collect())
                    .collect();
                out.push((
                    output.edges,
                    nodes,
                    frontier,
                    state.stats.nodes_per_type.clone(),
                    state.stats.edges_per_type.clone(),
                ));
            }
            runs.push(out);
        }
        let edges: usize = runs[0].iter().map(|r| r.0.len()).sum();
        assert!(edges > 0);
        assert!(
            runs[1] == runs[2],
            "hops {hops}: legacy recording is not deterministic across states"
        );
        assert!(
            runs[0] == runs[1],
            "hops {hops}: per-seed recording differs from legacy"
        );
        println!("hops {hops}: {n} seeds, {edges} edges identical");
    }
}
