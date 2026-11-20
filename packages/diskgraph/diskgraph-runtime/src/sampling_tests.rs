// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{make_worker_offsets, SeedSchedule, SeedScheduleConfig};
use serde_json::json;

fn config() -> SeedScheduleConfig {
    SeedScheduleConfig {
        stage_key: "task/train".into(),
        ranges: vec![(10, 12), (20, 32)],
        num_workers: 2,
        worker_id: 0,
        shuffle: true,
        rnd_seed: Some(42),
        epoch: 3,
        batch_size: 2,
        drop_last: false,
        rhs: false,
        selected_stage_node_ids: None,
        weighted_seed_sampling_json: None,
        bucket_extension_root: None,
    }
}

#[test]
fn metadata_schedule_preserves_flat_shards_and_shuffle() {
    for worker in 0..2 {
        let mut c = config();
        c.worker_id = worker;
        let offsets = make_worker_offsets(16, 2, worker, true, Some(42), 3, "task/train")
            .expect("reference offsets");
        let expected: Vec<_> = offsets
            .into_iter()
            .map(|offset| {
                if offset < 3 {
                    10 + offset
                } else {
                    20 + offset - 3
                }
            })
            .collect();
        let mut schedule = SeedSchedule::new(c.clone()).expect("schedule");
        assert_eq!(schedule.num_samples(), expected.len());
        let mut actual = Vec::new();
        while let Some(ids) = schedule.next_seed_ids().expect("seeds") {
            assert!(ids.len() <= 2);
            actual.extend(ids);
        }
        assert_eq!(actual, expected);
    }
}

#[test]
fn metadata_schedule_preserves_selected_multiplicity_and_order() {
    let mut c = config();
    c.shuffle = false;
    c.selected_stage_node_ids = Some(vec![12, 20, 12, 21, 10]);
    let mut schedule = SeedSchedule::new(c.clone()).expect("schedule");
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![12, 12]));
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![10]));
    assert_eq!(schedule.next_seed_ids().unwrap(), None);
}

#[test]
fn metadata_schedule_preserves_rhs_range_and_tail() {
    let mut c = config();
    c.ranges = vec![(20, 24)];
    c.stage_key = "items".into();
    c.rhs = true;
    c.shuffle = false;
    c.num_workers = 1;
    let mut schedule = SeedSchedule::new(c.clone()).expect("schedule");
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![20, 21]));
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![22, 23]));
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![24]));
    assert_eq!(schedule.next_seed_ids().unwrap(), None);
    c.drop_last = true;
    let mut schedule = SeedSchedule::new(c.clone()).unwrap();
    assert_eq!(schedule.worker_batch_count(), 2);
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![20, 21]));
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![22, 23]));
    assert_eq!(schedule.next_seed_ids().unwrap(), None);
}

#[test]
fn metadata_schedule_rejects_invalid_requests() {
    for c in [
        SeedScheduleConfig {
            batch_size: 0,
            ..config()
        },
        SeedScheduleConfig {
            ranges: vec![(0, 5), (5, 9)],
            ..config()
        },
        SeedScheduleConfig {
            num_workers: 0,
            ..config()
        },
        SeedScheduleConfig {
            worker_id: 2,
            ..config()
        },
        SeedScheduleConfig {
            rnd_seed: None,
            ..config()
        },
    ] {
        assert!(SeedSchedule::new(c).is_err());
    }
}

#[test]
fn metadata_schedule_rejects_selected_ids_before_worker_sharding() {
    for ids in [vec![999], vec![10, 999]] {
        let mut c = config();
        c.shuffle = false;
        c.selected_stage_node_ids = Some(ids);
        assert!(matches!(
            SeedSchedule::new(c.clone()),
            Err(super::RuntimeError::Data(message))
                if message == "selected_stage_node_ids contains seed id outside stage table"
        ));
    }
}

#[test]
fn metadata_schedule_maps_maximum_node_id_without_overflow() {
    let mut c = config();
    c.ranges = vec![(0, 0), (u64::MAX, u64::MAX)];
    c.shuffle = false;
    c.num_workers = 1;
    let mut schedule = SeedSchedule::new(c.clone()).unwrap();
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![0, u64::MAX]));
    assert_eq!(schedule.next_seed_ids().unwrap(), None);
}

#[test]
fn metadata_schedule_requires_nonempty_rhs_range() {
    let mut c = config();
    c.rhs = true;
    for ranges in [vec![], vec![(24, 20)], vec![(20, 21), (24, 25)]] {
        c.ranges = ranges;
        assert!(matches!(
            SeedSchedule::new(c.clone()),
            Err(super::RuntimeError::Config(message))
                if message == "RHS requires one nonempty dense node range"
        ));
    }
    c.ranges = vec![(24, 24)];
    let mut schedule = SeedSchedule::new(c.clone()).unwrap();
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![24]));
    assert_eq!(schedule.next_seed_ids().unwrap(), None);
}

#[test]
fn metadata_schedule_reuses_grouped_schedule() {
    use std::io::Write;
    let root = tempfile::tempdir().unwrap();
    let bin = root.path().join("buckets.bin");
    let mut file = std::fs::File::create(&bin).unwrap();
    for id in [10u64, 11, 12, 20, 21] {
        file.write_all(&id.to_le_bytes()).unwrap();
    }
    std::fs::write(
        root.path().join("diskgraph_manifest.json"),
        json!({
            "task_dataset": {"stages": {"train": {
                "table_name": "train", "bucket_index": {
                    "path": "buckets.bin", "buckets": [
                        {"ts": 1, "offset": 0, "count": 3},
                        {"ts": 2, "offset": 3, "count": 2},
                    ]
                }
            }}}
        })
        .to_string(),
    )
    .unwrap();
    let mut c = config();
    c.bucket_extension_root = Some(root.path().to_path_buf());
    c.ranges = vec![(20, 32), (10, 12)];
    let mut actual = SeedSchedule::new(c.clone()).unwrap();
    let mut reference = super::BucketSchedule::build(
        vec![(1, 0, 3), (2, 3, 2)],
        bin,
        42,
        3,
        "task/train".into(),
        2,
        2,
        0,
    )
    .unwrap();
    assert_eq!(
        actual.num_samples(),
        reference.worker_batch_count() as usize
    );
    while let Some(expected) = reference.next_node_ids().unwrap() {
        assert_eq!(actual.next_seed_ids().unwrap(), Some(expected));
    }
    assert_eq!(actual.next_seed_ids().unwrap(), None);
    c.num_workers = 1;
    c.drop_last = true;
    let mut tails = SeedSchedule::new(c.clone()).unwrap();
    let mut sizes = Vec::new();
    while let Some(ids) = tails.next_seed_ids().unwrap() {
        sizes.push(ids.len());
    }
    sizes.sort_unstable();
    assert_eq!(sizes, vec![1, 2, 2]);
    c.ranges = vec![(999, 999)];
    let mut invalid = SeedSchedule::new(c.clone()).unwrap();
    assert!(matches!(
        invalid.next_seed_ids(),
        Err(super::RuntimeError::Data(message))
            if message == "grouped seed id outside stage table"
    ));
}

#[test]
fn metadata_schedule_reuses_weighted_draws() {
    use std::io::Write;
    let root = tempfile::tempdir().unwrap();
    let ids_path = root.path().join("ids.bin");
    let mut file = std::fs::File::create(&ids_path).unwrap();
    for id in [10u64, 11, 12] {
        file.write_all(&id.to_le_bytes()).unwrap();
    }
    let weighted = json!({"num_samples": 12, "buckets": [{
        "name": "all", "node_ids_path": ids_path.to_str().unwrap(),
        "count": 3, "mass": 3.0,
    }]})
    .to_string();
    let mut c = config();
    c.weighted_seed_sampling_json = Some(weighted.clone());
    let expected = super::make_weighted_stage_node_ids(
        &weighted,
        |id| (10..=12).contains(&id),
        "task/train",
        2,
        0,
        Some(42),
        3,
    )
    .unwrap();
    let mut actual = SeedSchedule::new(c.clone()).unwrap();
    let mut ids = Vec::new();
    while let Some(batch) = actual.next_seed_ids().unwrap() {
        ids.extend(batch);
    }
    assert_eq!(ids, expected);
}
