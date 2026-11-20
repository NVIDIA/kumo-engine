// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{make_worker_offsets, SeedSchedule};
use serde_json::{json, Value};

fn config() -> Value {
    json!({
        "stage_key": "task/train", "ranges": [[10, 12], [20, 32]],
        "num_workers": 2, "worker_id": 0, "shuffle": true,
        "rnd_seed": 42, "epoch": 3, "batch_size": 2, "drop_last": false,
        "rhs": false,
    })
}

#[test]
fn metadata_schedule_preserves_flat_shards_and_shuffle() {
    for worker in 0..2 {
        let mut c = config();
        c["worker_id"] = json!(worker);
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
        let mut schedule = SeedSchedule::from_json(&c.to_string()).expect("schedule");
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
    c["shuffle"] = json!(false);
    c["selected_stage_node_ids"] = json!([12, 20, 12, 21, 10]);
    let mut schedule = SeedSchedule::from_json(&c.to_string()).expect("schedule");
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![12, 12]));
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![10]));
    assert_eq!(schedule.next_seed_ids().unwrap(), None);
}

#[test]
fn metadata_schedule_preserves_rhs_range_and_tail() {
    let mut c = config();
    c["ranges"] = json!([[20, 24]]);
    c["stage_key"] = json!("items");
    c["rhs"] = json!(true);
    c["shuffle"] = json!(false);
    c["num_workers"] = json!(1);
    let mut schedule = SeedSchedule::from_json(&c.to_string()).expect("schedule");
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![20, 21]));
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![22, 23]));
    assert_eq!(schedule.next_seed_ids().unwrap(), Some(vec![24]));
    assert_eq!(schedule.next_seed_ids().unwrap(), None);
    c["drop_last"] = json!(true);
    let mut schedule = SeedSchedule::from_json(&c.to_string()).unwrap();
    assert_eq!(schedule.worker_batch_count(), 2);
    schedule.next_seed_ids().unwrap();
    schedule.next_seed_ids().unwrap();
    assert_eq!(schedule.next_seed_ids().unwrap(), None);
}

#[test]
fn metadata_schedule_rejects_invalid_requests() {
    for (field, value) in [
        ("batch_size", json!(0)),
        ("ranges", json!([[0, 5], [5, 9]])),
        ("num_workers", json!(0)),
        ("worker_id", json!(2)),
        ("rnd_seed", Value::Null),
        ("unexpected", json!(true)),
        ("selected_stage_node_ids", json!([999])),
        ("selected_stage_node_ids", json!([10, 999])),
    ] {
        let mut c = config();
        c[field] = value;
        assert!(SeedSchedule::from_json(&c.to_string()).is_err(), "{field}");
    }
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
    c["bucket_extension_root"] = json!(root.path().to_str().unwrap());
    let mut actual = SeedSchedule::from_json(&c.to_string()).unwrap();
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
    c["weighted_seed_sampling_json"] = json!(weighted);
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
    let mut actual = SeedSchedule::from_json(&c.to_string()).unwrap();
    let mut ids = Vec::new();
    while let Some(batch) = actual.next_seed_ids().unwrap() {
        ids.extend(batch);
    }
    assert_eq!(ids, expected);
}
