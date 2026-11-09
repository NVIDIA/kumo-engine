// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use diskgraph_index::layout::RelationLayout;

fn sample(times: &[u8], fanout: u32, seed: u64, mode: SamplingMode) -> Result<Vec<u64>> {
    let mut engine = DiskGraphEngine::new(
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        None,
        TimeUnit::Seconds,
        Some(seed),
        Some(1),
    );
    engine.relation_info = vec![0];
    let data: Vec<u8> = (0..times.len() as u64).flat_map(u64::to_le_bytes).collect();
    let segment = RelationSegment {
        layout: RelationLayout {
            relation_id: 0,
            dst_bits: 64,
            dest_sort_key_bits: None,
            entry_bytes: 8,
            dst_mask: u64::MAX,
            dest_sort_key_mask: None,
        },
        count: times.len() as u32,
        data: &data,
    };
    let node = NodeRecord {
        global_id: 0,
        seed_index: 0,
        local_id: 0,
        seed_type_id: 0,
    };
    let spec = FanoutSpec {
        fanout,
        mode,
        absent_key_policy: AbsentKeyPolicy::UniformFallback,
    };
    let cutoff = NodeCutoff {
        type_id: 0,
        times,
        seed_time: 1,
    };
    let mut result = Vec::new();
    engine.sample_segment_edges(
        &node,
        0,
        &segment,
        &spec,
        &[],
        &mut Vec::new(),
        &mut Vec::new(),
        Some(&cutoff),
        |id| {
            result.push(id);
            Ok(())
        },
    )?;
    Ok(result)
}

#[test]
fn filtered_samples_cover_sparse_dense_and_fanout_boundaries() {
    for degree in [0, 10, 10000] {
        for every in [1, 2, 1000, usize::MAX] {
            let times: Vec<u8> = (0..degree)
                .map(|i| {
                    if every != usize::MAX && i % every == 0 {
                        1
                    } else {
                        3
                    }
                })
                .collect();
            let visible = times.iter().filter(|&&t| t == 1).count();
            for fanout in [0, 1, 3, 100, 10000, 100000] {
                for mode in [SamplingMode::Uniform, SamplingMode::DeterministicUniform] {
                    let first = sample(&times, fanout, 41, mode.clone()).unwrap();
                    assert_eq!(first.len(), visible.min(fanout as usize));
                    assert!(first.iter().all(|&id| times[id as usize] == 1));
                    assert!(first.windows(2).all(|w| w[0] < w[1]));
                    assert_eq!(first, sample(&times, fanout, 41, mode).unwrap());
                }
            }
        }
    }
}

#[test]
fn sparse_fallback_samples_all_visible_positions() {
    let mut times = vec![3; 1000];
    for i in [7, 113, 337, 999] {
        times[i] = 1;
    }
    let mut counts = [0usize; 1000];
    for seed in 0..1000 {
        for id in sample(&times, 2, seed, SamplingMode::Uniform).unwrap() {
            counts[id as usize] += 1;
        }
    }
    for i in [7, 113, 337, 999] {
        assert!(
            (400..600).contains(&counts[i]),
            "position {i}: {}",
            counts[i]
        );
    }
}

#[test]
fn missing_fact_ids_are_hidden() {
    let cutoff = NodeCutoff {
        type_id: 0,
        times: &[0, 1, 2, 3],
        seed_time: 1,
    };
    assert!(cutoff.visible(0));
    assert!(cutoff.visible(1));
    for id in [2, 3, 4, u64::MAX] {
        assert!(!cutoff.visible(id));
    }
}

#[test]
fn zero_fanout_does_not_accept_unsupported_weighted_modes() {
    use diskgraph_core::sampling::Weighting;
    let weighting = Weighting::ExponentialDecay { lambda: 1.0 };
    for mode in [
        SamplingMode::Weighted {
            weighting: weighting.clone(),
        },
        SamplingMode::TemporalWeighted { weighting },
    ] {
        let error = sample(&[1, 2], 0, 41, mode).unwrap_err();
        assert!(error.to_string().contains("not supported"));
    }
}
