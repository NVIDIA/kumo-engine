// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use arrow_array::Int64Array;
use diskgraph_train_table::event_stream::event_stream_order;

#[test]
fn merge_is_stable_and_keeps_anchors_separate() {
    let parts = vec![
        (
            Int64Array::from(vec![0, 2, 3]),
            Int64Array::from(vec![10, 20, 5]),
            true,
        ),
        (
            Int64Array::from(vec![0, 2, 3]),
            Int64Array::from(vec![20, 30, 1]),
            true,
        ),
    ];

    let (offsets, order) = event_stream_order(&parts, None, false, false).unwrap();

    assert_eq!(offsets, [0, 4, 6]);
    assert_eq!(order, [0, 1, 3, 4, 5, 2]);
    let (_, latest) = event_stream_order(&parts, Some(2), true, false).unwrap();
    assert_eq!(latest, [3, 4, 5, 2]);
    let (_, reversed) = event_stream_order(&parts, Some(2), true, true).unwrap();
    assert_eq!(reversed, [4, 3, 2, 5]);
}

#[test]
fn merge_rejects_invalid_offsets_and_declared_order() {
    let invalid_offsets = vec![(
        Int64Array::from(vec![0, 3]),
        Int64Array::from(vec![10, 20]),
        true,
    )];
    let invalid_order = vec![(
        Int64Array::from(vec![0, 2]),
        Int64Array::from(vec![20, 10]),
        true,
    )];

    assert!(event_stream_order(&invalid_offsets, None, false, false).is_err());
    assert!(event_stream_order(&invalid_order, None, false, false).is_err());
}

#[test]
fn query_merge_resolves_sequences_and_rejects_invalid_names() {
    use std::sync::Arc;

    use arrow_array::RecordBatch;
    use arrow_schema::Schema;
    use diskgraph_core::time::TimeUnit;
    use diskgraph_train_table::{
        batch_aggregate::{QueryResult, QueryStats, QueryTimings, SequenceResult},
        event_stream::EventMerge,
    };

    let result = QueryResult {
        sequences: vec![SequenceResult {
            clock: TimeUnit::Seconds,
            name: "orders".into(),
            offsets: vec![0, 2, 3],
            node_ids: vec![1, 2, 3],
            timestamps: vec![Some(10), Some(20), Some(5)],
            batch: RecordBatch::new_empty(Arc::new(Schema::empty())),
        }],
        aggregates: vec![],
        timings: QueryTimings::default(),
        stats: QueryStats::default(),
    };
    let plan: EventMerge = serde_json::from_value(serde_json::json!({
        "name": "events", "sequences": ["orders"], "ascending": [true],
        "maximum": 1, "keep_last": true, "descending": false,
    }))
    .unwrap();

    let merged = result.merged_event_orders(&[plan]).unwrap();

    assert_eq!(
        merged,
        vec![("events".to_owned(), vec![0, 1, 2], vec![1, 2])]
    );
    for (sequences, ascending, message) in [
        (vec!["missing"], vec![true], "unknown merge sequence"),
        (vec!["orders"], vec![], "counts differ"),
    ] {
        let plan = serde_json::from_value(serde_json::json!({
            "name": "events", "sequences": sequences, "ascending": ascending,
            "maximum": 1, "keep_last": true, "descending": false,
        }))
        .unwrap();
        assert!(result
            .merged_event_orders(&[plan])
            .unwrap_err()
            .contains(message));
    }
}
