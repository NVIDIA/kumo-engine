use std::fs;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use diskgraph_core::metadata::{EdgeRelationMetadata, NodeTypeMetadata};
use diskgraph_index::{
    builder::{ShardBuildParams, ShardBuilder},
    layout::RelationSegment,
    reader::ShardReader,
    streaming::{EdgeShardInput, NodeShardInput, ParquetSource},
};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::tempdir;

#[derive(Clone)]
struct EdgeRow {
    src: i64,
    dst: i64,
    dest_sort_key: Option<i64>,
}

#[test]
fn build_and_read_round_trip_basic() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0, 1]).unwrap();
    write_edges(
        &edge_path,
        &[
            EdgeRow {
                src: 0,
                dst: 10,
                dest_sort_key: None,
            },
            EdgeRow {
                src: 0,
                dst: 11,
                dest_sort_key: None,
            },
            EdgeRow {
                src: 1,
                dst: 12,
                dest_sort_key: None,
            },
        ],
        false,
    )
    .unwrap();

    let node_meta = node_meta("User", 12);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::LocalFile(node_path),
            },
            &[EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::LocalFile(edge_path),
                has_dest_sort_key: false,
            }],
        )
        .unwrap();

    let reader = ShardReader::open(&output_dir, built.layouts.clone()).unwrap();
    let header = reader.header();
    assert_eq!(header.nodes, 2);
    assert_eq!(header.total_offsets, 3);

    let rel0 = 0;
    let seg0 = decode_relation(&reader, 0, rel0);
    assert_eq!(seg0.len(), 2);
    assert_eq!(seg0[0].0, 10);
    assert_eq!(seg0[1].0, 11);

    // Node with no edges should yield empty relation data and zero-length slice.
    let seg1 = decode_relation(&reader, 1, rel0);
    assert_eq!(seg1.len(), 1);
    assert_eq!(seg1[0].0, 12);
}

#[test]
fn build_handles_all_zero_edges() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    write_nodes(&node_path, &[0, 1]).unwrap();

    let node_meta = node_meta("User", 1);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::LocalFile(node_path),
            },
            &[],
        )
        .unwrap();

    let reader = ShardReader::open(&output_dir, built.layouts.clone()).unwrap();

    for node in 0..=1 {
        let seg = decode_relation(&reader, node, 0);
        assert!(seg.is_empty(), "expected empty relation for node {}", node);
    }
}

#[test]
fn build_rejects_dangling_edges() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0]).unwrap();
    write_edges(
        &edge_path,
        &[EdgeRow {
            src: 1,
            dst: 42,
            dest_sort_key: None,
        }],
        false,
    )
    .unwrap();

    let node_meta = node_meta("User", 42);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let msg = match builder.build(
        &NodeShardInput {
            node_type: "User".into(),
            shard_index: 0,
            parquet: ParquetSource::LocalFile(node_path),
        },
        &[EdgeShardInput {
            relation: "Follows".into(),
            source_shard_index: 0,
            parquet: ParquetSource::LocalFile(edge_path),
            has_dest_sort_key: false,
        }],
    ) {
        Ok(_) => panic!("expected DanglingEdges error"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("edges remain after exhausting node ids"),
        "expected DanglingEdges, got {msg}"
    );
}

#[test]
fn build_rejects_edge_ordering_violation() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0]).unwrap();
    // dest_sort_key must be descending; make it ascending to trigger violation.
    write_edges(
        &edge_path,
        &[
            EdgeRow {
                src: 0,
                dst: 1,
                dest_sort_key: Some(1),
            },
            EdgeRow {
                src: 0,
                dst: 2,
                dest_sort_key: Some(2),
            },
        ],
        true,
    )
    .unwrap();

    let node_meta = node_meta("User", 2);
    let relation_meta = relation_meta("Temporal", "User", "User", Some(8));
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let msg = match builder.build(
        &NodeShardInput {
            node_type: "User".into(),
            shard_index: 0,
            parquet: ParquetSource::LocalFile(node_path),
        },
        &[EdgeShardInput {
            relation: "Temporal".into(),
            source_shard_index: 0,
            parquet: ParquetSource::LocalFile(edge_path),
            has_dest_sort_key: true,
        }],
    ) {
        Ok(_) => panic!("expected EdgeOrderingViolation error"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("input edges not sorted by src_id"),
        "expected EdgeOrderingViolation, got {msg}"
    );
}

#[test]
fn offsets_span_multiple_chunks_and_edges_decode_correctly() {
    // Ensure offset delta encoding handles repeated offsets and multiple chunks.
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    let nodes: Vec<i64> = (0..=129).collect(); // 130 nodes => 131 offsets (seed + per node), spans >2 chunks of 64
    write_nodes(&node_path, &nodes).unwrap();
    write_edges(
        &edge_path,
        &[
            EdgeRow {
                src: 0,
                dst: 10,
                dest_sort_key: None,
            },
            EdgeRow {
                src: 0,
                dst: 11,
                dest_sort_key: None,
            },
            EdgeRow {
                src: 63,
                dst: 63,
                dest_sort_key: None,
            },
            EdgeRow {
                src: 64,
                dst: 120,
                dest_sort_key: None,
            },
            EdgeRow {
                src: 64,
                dst: 121,
                dest_sort_key: None,
            },
            EdgeRow {
                src: 129,
                dst: 5,
                dest_sort_key: None,
            },
        ],
        false,
    )
    .unwrap();

    let node_meta = node_meta("User", 129);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::LocalFile(node_path),
            },
            &[EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::LocalFile(edge_path),
                has_dest_sort_key: false,
            }],
        )
        .unwrap();
    let reader = ShardReader::open(&output_dir, built.layouts.clone()).unwrap();

    // Nodes with edges.
    assert_eq!(decode_relation(&reader, 0, 0), vec![(10, None), (11, None)]);
    assert_eq!(decode_relation(&reader, 63, 0), vec![(63, None)]);
    assert_eq!(
        decode_relation(&reader, 64, 0),
        vec![(120, None), (121, None)]
    );
    assert_eq!(decode_relation(&reader, 129, 0), vec![(5, None)]);

    // Nodes without edges across chunk boundary should decode as empty.
    assert!(decode_relation(&reader, 1, 0).is_empty());
    assert!(decode_relation(&reader, 70, 0).is_empty());
    assert!(decode_relation(&reader, 100, 0).is_empty());
}

#[test]
fn sort_key_round_trip_decodes_correctly() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0, 1]).unwrap();
    write_edges(
        &edge_path,
        &[
            EdgeRow {
                src: 0,
                dst: 10,
                dest_sort_key: Some(5),
            },
            EdgeRow {
                src: 0,
                dst: 11,
                dest_sort_key: Some(3),
            },
            EdgeRow {
                src: 1,
                dst: 12,
                dest_sort_key: Some(9),
            },
        ],
        true,
    )
    .unwrap();

    let node_meta = node_meta("User", 12);
    let relation_meta = relation_meta("Temporal", "User", "User", Some(8));
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::LocalFile(node_path),
            },
            &[EdgeShardInput {
                relation: "Temporal".into(),
                source_shard_index: 0,
                parquet: ParquetSource::LocalFile(edge_path),
                has_dest_sort_key: true,
            }],
        )
        .unwrap();
    let reader = ShardReader::open(&output_dir, built.layouts.clone()).unwrap();

    assert_eq!(
        decode_relation(&reader, 0, 0),
        vec![(10, Some(5)), (11, Some(3))] // order preserved by sort key desc
    );
    assert_eq!(decode_relation(&reader, 1, 0), vec![(12, Some(9))]);
}

#[test]
fn multi_relation_counts_and_edges_decode_correctly() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path_a = dir.path().join("edges_a.parquet");
    let edge_path_b = dir.path().join("edges_b.parquet");
    write_nodes(&node_path, &[0, 1, 2]).unwrap();
    // Relation A: no sort key.
    write_edges(
        &edge_path_a,
        &[EdgeRow {
            src: 0,
            dst: 10,
            dest_sort_key: None,
        }],
        false,
    )
    .unwrap();
    // Relation B: with sort key.
    write_edges(
        &edge_path_b,
        &[
            EdgeRow {
                src: 0,
                dst: 20,
                dest_sort_key: Some(5),
            },
            EdgeRow {
                src: 0,
                dst: 21,
                dest_sort_key: Some(3),
            },
            EdgeRow {
                src: 1,
                dst: 22,
                dest_sort_key: Some(9),
            },
        ],
        true,
    )
    .unwrap();

    let node_meta = node_meta("User", 22);
    let follows = relation_meta("Follows", "User", "User", None);
    let temporal = relation_meta("Temporal", "User", "User", Some(8));
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![follows.clone(), temporal.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::LocalFile(node_path),
            },
            &[
                EdgeShardInput {
                    relation: "Follows".into(),
                    source_shard_index: 0,
                    parquet: ParquetSource::LocalFile(edge_path_a),
                    has_dest_sort_key: false,
                },
                EdgeShardInput {
                    relation: "Temporal".into(),
                    source_shard_index: 0,
                    parquet: ParquetSource::LocalFile(edge_path_b),
                    has_dest_sort_key: true,
                },
            ],
        )
        .unwrap();
    let reader = ShardReader::open(&output_dir, built.layouts.clone()).unwrap();

    let view0 = reader.edges_for_node(0).unwrap();
    assert_eq!(view0.raw.len(), 7, "unexpected raw len for node 0");
    assert_eq!(
        view0.raw.to_vec(),
        vec![1u8, 2, 10, 180, 0, 117, 0],
        "unexpected raw bytes for node0"
    );
    let view1 = reader.edges_for_node(1).unwrap();
    assert_eq!(view1.raw.len(), 4, "unexpected raw len for node 1");
    assert_eq!(
        view1.raw.to_vec(),
        vec![0u8, 1, 54, 1],
        "unexpected raw bytes for node1"
    );

    // Relation order: Follows (idx 0), Temporal (idx 1).
    assert_eq!(decode_relation(&reader, 0, 0), vec![(10, None)]);
    assert_eq!(
        decode_relation(&reader, 0, 1),
        vec![(20, Some(5)), (21, Some(3))]
    );

    assert!(decode_relation(&reader, 1, 0).is_empty());
    assert_eq!(decode_relation(&reader, 1, 1), vec![(22, Some(9))]);
}

#[test]
fn multi_relation_mixed_dst_widths_and_partial_edges() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path_a = dir.path().join("edges_a.parquet"); // small dst ids
    let edge_path_b = dir.path().join("edges_b.parquet"); // larger dst ids + sort key
    write_nodes(&node_path, &[0, 1, 2]).unwrap();

    // Relation A: no sort key, small dst ids.
    write_edges(
        &edge_path_a,
        &[EdgeRow {
            src: 0,
            dst: 3, // small dst
            dest_sort_key: None,
        }],
        false,
    )
    .unwrap();
    // Relation B: with sort key, larger dst ids; node 0 has multiple edges, node 1 empty.
    write_edges(
        &edge_path_b,
        &[
            EdgeRow {
                src: 0,
                dst: 5000,
                dest_sort_key: Some(9),
            },
            EdgeRow {
                src: 0,
                dst: 4000,
                dest_sort_key: Some(2),
            },
            EdgeRow {
                src: 2,
                dst: 6000,
                dest_sort_key: Some(1),
            },
        ],
        true,
    )
    .unwrap();

    let node_meta = node_meta("User", 6000);
    let follows = relation_meta("Follows", "User", "User", None);
    let temporal = relation_meta("Temporal", "User", "User", Some(8));
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![follows.clone(), temporal.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::LocalFile(node_path),
            },
            &[
                EdgeShardInput {
                    relation: "Follows".into(),
                    source_shard_index: 0,
                    parquet: ParquetSource::LocalFile(edge_path_a),
                    has_dest_sort_key: false,
                },
                EdgeShardInput {
                    relation: "Temporal".into(),
                    source_shard_index: 0,
                    parquet: ParquetSource::LocalFile(edge_path_b),
                    has_dest_sort_key: true,
                },
            ],
        )
        .unwrap();
    let reader = ShardReader::open(&output_dir, built.layouts.clone()).unwrap();

    // Node 0: edges in both relations; relation B should be sorted by sort key desc.
    assert_eq!(decode_relation(&reader, 0, 0), vec![(3, None)]);
    assert_eq!(
        decode_relation(&reader, 0, 1),
        vec![(5000, Some(9)), (4000, Some(2))]
    );
    // Node 1: empty in both relations.
    assert!(decode_relation(&reader, 1, 0).is_empty());
    assert!(decode_relation(&reader, 1, 1).is_empty());
    // Node 2: only relation B has an edge.
    assert!(decode_relation(&reader, 2, 0).is_empty());
    assert_eq!(decode_relation(&reader, 2, 1), vec![(6000, Some(1))]);
}

#[test]
fn truncates_large_node_without_sort_key() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    // Two nodes; node 0 has a huge fanout, node 1 has a small edge.
    write_nodes(&node_path, &[0, 1]).unwrap();
    let big_count = 70_000;
    let mut edges = Vec::with_capacity(big_count as usize + 1);
    for i in 0..big_count {
        edges.push(EdgeRow {
            src: 0,
            dst: i as i64,
            dest_sort_key: None,
        });
    }
    edges.push(EdgeRow {
        src: 1,
        dst: 999,
        dest_sort_key: None,
    });
    write_edges(&edge_path, &edges, false).unwrap();

    let node_meta = node_meta("User", 999);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::LocalFile(node_path),
            },
            &[EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::LocalFile(edge_path),
                has_dest_sort_key: false,
            }],
        )
        .unwrap();
    let reader = ShardReader::open(&output_dir, built.layouts.clone()).unwrap();
    let segments0 = reader
        .edges_for_node(0)
        .unwrap()
        .relation_segments()
        .unwrap();
    let seg = &segments0[0];
    let entry_bytes = seg.layout.entry_bytes as u64;
    let budget = u16::MAX as u64; // base_span is 0 for first node.
    let node_bytes =
        varint_len_u64(seg.count as u64) as u64 + (seg.count as u64).saturating_mul(entry_bytes);
    assert!(
        seg.count < big_count as u32,
        "expected truncation below original"
    );
    assert!(
        node_bytes <= budget,
        "truncated node payload should fit budget"
    );
    assert!(built.build_stats.truncated_edges_total > 0);
    // Next node should still decode correctly.
    let segments1 = reader
        .edges_for_node(1)
        .unwrap()
        .relation_segments()
        .unwrap();
    assert_eq!(segments1[0].count, 1);
    let decoded1 = decode_relation(&reader, 1, 0);
    assert_eq!(decoded1, vec![(999, None)]);
}

#[test]
fn truncates_large_node_with_sort_key() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0, 1]).unwrap();
    let big_count = 70_000;
    let mut edges = Vec::with_capacity(big_count as usize + 1);
    for i in 0..big_count {
        edges.push(EdgeRow {
            src: 0,
            dst: (i + 1000) as i64,
            dest_sort_key: Some((big_count - i) as i64),
        });
    }
    // Node 1 has two edges with sort keys.
    edges.push(EdgeRow {
        src: 1,
        dst: 500,
        dest_sort_key: Some(8),
    });
    edges.push(EdgeRow {
        src: 1,
        dst: 600,
        dest_sort_key: Some(7),
    });
    write_edges(&edge_path, &edges, true).unwrap();

    let node_meta = node_meta("User", 70_600);
    let relation_meta = relation_meta("Temporal", "User", "User", Some(8));
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::LocalFile(node_path),
            },
            &[EdgeShardInput {
                relation: "Temporal".into(),
                source_shard_index: 0,
                parquet: ParquetSource::LocalFile(edge_path),
                has_dest_sort_key: true,
            }],
        )
        .unwrap();
    let reader = ShardReader::open(&output_dir, built.layouts.clone()).unwrap();
    let segments0 = reader
        .edges_for_node(0)
        .unwrap()
        .relation_segments()
        .unwrap();
    let seg0 = &segments0[0];
    let entry_bytes = seg0.layout.entry_bytes as u64;
    let budget = u16::MAX as u64;
    let node_bytes =
        varint_len_u64(seg0.count as u64) as u64 + (seg0.count as u64).saturating_mul(entry_bytes);
    assert!(
        seg0.count < big_count as u32,
        "expected truncation with sort key"
    );
    assert!(
        node_bytes <= budget,
        "truncated node payload should fit budget"
    );
    assert!(built.build_stats.truncated_edges_total > 0);

    // Node 1 should decode its two edges with sort keys intact and ordered.
    let decoded = decode_relation(&reader, 1, 0);
    assert_eq!(decoded, vec![(500, Some(8)), (600, Some(7))]);
}

#[test]
fn reader_rejects_invalid_header() {
    let dir = tempdir().unwrap();
    let offsets = dir.path().join("offsets.bin");
    let edges = dir.path().join("edges.bin");
    let header_size = std::mem::size_of::<diskgraph_index::layout::OffsetFileHeader>();
    let mut buf = vec![0u8; header_size];
    // total_offsets = 1 (bytes 12..20 for u64 after version + nodes)
    buf[12..20].copy_from_slice(&1u64.to_le_bytes());
    fs::write(&offsets, buf).unwrap();
    fs::write(&edges, vec![]).unwrap();
    let layouts = vec![relation_meta("Follows", "User", "User", None)];
    let node_types = vec![node_meta("User", 0)];
    let layouts = relation_layouts(layouts, node_types);
    let reader = ShardReader::open(dir.path(), layouts).unwrap();
    let msg = match reader.edges_for_node(0) {
        Ok(_) => panic!("expected offset chunk error"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("offset chunk missing")
            || msg.contains("offset chunk range mismatch")
            || msg.contains("node index out of range"),
        "expected offset chunk error, got {msg}"
    );
}

#[test]
fn reader_rejects_offsets_out_of_bounds() {
    let (_tmp, output_dir, layouts) = build_simple_shard(&[0], &[]); // zero-edge node
                                                                     // Corrupt the base offset to point past the edge blob (which is empty).
    let offsets_path = output_dir.join("offsets.bin");
    let mut bytes = fs::read(&offsets_path).unwrap();
    // Header is bincode-serialized OffsetFileHeader, then chunk: count(u16), base(u64)
    let header_len = std::mem::size_of::<diskgraph_index::layout::OffsetFileHeader>();
    // write base = large number
    let base_pos = header_len + 2;
    bytes[base_pos..base_pos + 8].copy_from_slice(&1_000_000u64.to_le_bytes());
    fs::write(&offsets_path, bytes).unwrap();

    let reader = ShardReader::open(&output_dir, layouts).unwrap();
    let msg = match reader.edges_for_node(0) {
        Ok(_) => panic!("expected offset bounds error"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("offset points outside edge blob"),
        "expected offset bounds error, got {msg}"
    );
}

#[test]
fn reader_rejects_node_index_out_of_range() {
    let (_tmp, output_dir, layouts) = build_simple_shard(&[0], &[]);
    let reader = ShardReader::open(&output_dir, layouts).unwrap();
    let msg = match reader.edges_for_node(10) {
        Ok(_) => panic!("expected node index out of range"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("node index out of range"),
        "expected node index error, got {msg}"
    );
}

#[test]
fn reader_rejects_offset_chunk_mismatch() {
    let (_tmp, output_dir, layouts) = build_simple_shard(&[0, 1], &[]);
    let offsets_path = output_dir.join("offsets.bin");
    let mut bytes = fs::read(&offsets_path).unwrap();
    // Tamper with the first chunk count to a larger number.
    let header_len = std::mem::size_of::<diskgraph_index::layout::OffsetFileHeader>();
    bytes[header_len] = 10;
    bytes[header_len + 1] = 0;
    fs::write(&offsets_path, bytes).unwrap();
    let reader = ShardReader::open(&output_dir, layouts).unwrap();
    let msg = match reader.edges_for_node(5) {
        Ok(_) => panic!("expected offset chunk mismatch"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("offset chunk range mismatch")
            || msg.contains("offset chunk missing")
            || msg.contains("node index out of range"),
        "expected metadata error, got {msg}"
    );
}

#[test]
fn reader_rejects_truncated_edge_counts() {
    let (_tmp, output_dir, layouts) = build_simple_shard(
        &[0],
        &[EdgeRow {
            src: 0,
            dst: 1,
            dest_sort_key: None,
        }],
    );
    let edges_path = output_dir.join("edges.bin");
    let mut bytes = fs::read(&edges_path).unwrap();
    // Replace the edge blob with an incomplete varint to force decode failure.
    bytes.clear();
    bytes.push(0x80); // varint with continuation but no following byte
    fs::write(&edges_path, bytes).unwrap();
    let reader = ShardReader::open(&output_dir, layouts).unwrap();
    let msg = match reader.edges_for_node(0) {
        Ok(view) => match view.relation_segments() {
            Ok(_) => panic!("expected relation_segments error"),
            Err(e) => e.to_string(),
        },
        Err(err) => err.to_string(),
    };
    let lower = msg.to_lowercase();
    assert!(
        lower.contains("unexpected eof")
            || lower.contains("invalid data")
            || lower.contains("offset points outside edge blob"),
        "expected EOF/invalid data, got {msg}"
    );
}

#[test]
fn builder_errors_when_node_column_missing() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    write_custom_nodes(&node_path, "wrong", &[0, 1]).unwrap();
    let node_meta = node_meta("User", 1);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: dir.path().join("out"),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let msg = match builder.build(
        &NodeShardInput {
            node_type: "User".into(),
            shard_index: 0,
            parquet: ParquetSource::LocalFile(node_path),
        },
        &[],
    ) {
        Ok(_) => panic!("expected invalid column error"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("invalid column type"),
        "expected InvalidColumn, got {msg}"
    );
}

#[test]
fn builder_errors_when_node_column_wrong_type() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    write_wrong_type_nodes(&node_path, "id", &["a", "b"]).unwrap();
    let node_meta = node_meta("User", 1);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: dir.path().join("out"),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let msg = match builder.build(
        &NodeShardInput {
            node_type: "User".into(),
            shard_index: 0,
            parquet: ParquetSource::LocalFile(node_path),
        },
        &[],
    ) {
        Ok(_) => panic!("expected invalid column type error"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("invalid column type"),
        "expected InvalidColumn type, got {msg}"
    );
}

#[test]
fn builder_errors_when_edge_columns_missing() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0]).unwrap();
    // Missing dest_sort_key even though metadata expects it.
    write_edges(
        &edge_path,
        &[EdgeRow {
            src: 0,
            dst: 1,
            dest_sort_key: None,
        }],
        false,
    )
    .unwrap();

    let node_meta = node_meta("User", 1);
    let relation_meta = relation_meta("Temporal", "User", "User", Some(8));
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: dir.path().join("out"),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let msg = match builder.build(
        &NodeShardInput {
            node_type: "User".into(),
            shard_index: 0,
            parquet: ParquetSource::LocalFile(node_path),
        },
        &[EdgeShardInput {
            relation: "Temporal".into(),
            source_shard_index: 0,
            parquet: ParquetSource::LocalFile(edge_path),
            has_dest_sort_key: true,
        }],
    ) {
        Ok(_) => panic!("expected missing dest_sort_key error"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("invalid column type"),
        "expected InvalidColumn for missing dest_sort_key, got {msg}"
    );
}

#[test]
fn builder_errors_when_edge_column_wrong_type() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0]).unwrap();
    write_wrong_type_edges(&edge_path, &["src_id"], &["dst_id"]).unwrap();

    let node_meta = node_meta("User", 1);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: dir.path().join("out"),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let msg = match builder.build(
        &NodeShardInput {
            node_type: "User".into(),
            shard_index: 0,
            parquet: ParquetSource::LocalFile(node_path),
        },
        &[EdgeShardInput {
            relation: "Follows".into(),
            source_shard_index: 0,
            parquet: ParquetSource::LocalFile(edge_path),
            has_dest_sort_key: false,
        }],
    ) {
        Ok(_) => panic!("expected invalid column type error"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("invalid column type"),
        "expected InvalidColumn type, got {msg}"
    );
}

#[test]
fn builder_errors_on_unknown_relation_name() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0]).unwrap();
    write_edges(
        &edge_path,
        &[EdgeRow {
            src: 0,
            dst: 1,
            dest_sort_key: None,
        }],
        false,
    )
    .unwrap();

    let node_meta = node_meta("User", 1);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: dir.path().join("out"),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let msg = match builder.build(
        &NodeShardInput {
            node_type: "User".into(),
            shard_index: 0,
            parquet: ParquetSource::LocalFile(node_path),
        },
        &[EdgeShardInput {
            relation: "Unknown".into(),
            source_shard_index: 0,
            parquet: ParquetSource::LocalFile(edge_path),
            has_dest_sort_key: false,
        }],
    ) {
        Ok(_) => panic!("expected UnknownRelation error"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("UnknownRelation") || msg.contains("unknown relation in shard input"),
        "expected UnknownRelation, got {msg}"
    );
}

#[test]
fn builder_errors_on_corrupted_parquet() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    // Write non-parquet bytes to edge file.
    write_nodes(&node_path, &[0]).unwrap();
    fs::write(&edge_path, b"not parquet").unwrap();

    let node_meta = node_meta("User", 0);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: dir.path().join("out"),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let msg = match builder.build(
        &NodeShardInput {
            node_type: "User".into(),
            shard_index: 0,
            parquet: ParquetSource::LocalFile(node_path),
        },
        &[EdgeShardInput {
            relation: "Follows".into(),
            source_shard_index: 0,
            parquet: ParquetSource::LocalFile(edge_path),
            has_dest_sort_key: false,
        }],
    ) {
        Ok(_) => panic!("expected parquet error"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.to_lowercase().contains("parquet"),
        "expected parquet error, got {msg}"
    );
}

#[test]
fn builder_fails_on_missing_edge_file() {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    write_nodes(&node_path, &[0]).unwrap();
    let node_meta = node_meta("User", 0);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: dir.path().join("out"),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let msg = match builder.build(
        &NodeShardInput {
            node_type: "User".into(),
            shard_index: 0,
            parquet: ParquetSource::LocalFile(node_path),
        },
        &[EdgeShardInput {
            relation: "Follows".into(),
            source_shard_index: 0,
            parquet: ParquetSource::LocalFile(dir.path().join("missing.parquet")),
            has_dest_sort_key: false,
        }],
    ) {
        Ok(_) => panic!("expected IO error for missing edge file"),
        Err(err) => err.to_string(),
    };
    assert!(
        msg.contains("No such file") || msg.contains("os error"),
        "expected IO error for missing file, got {msg}"
    );
}

fn node_meta(name: &str, max_id: u64) -> NodeTypeMetadata {
    NodeTypeMetadata {
        name: name.into(),
        max_id,
        shards: vec![],
    }
}

fn relation_meta(
    name: &str,
    src: &str,
    dst: &str,
    dest_sort_key_bits: Option<u8>,
) -> EdgeRelationMetadata {
    EdgeRelationMetadata {
        name: name.into(),
        src_type: src.into(),
        dst_type: dst.into(),
        shards: vec![],
        dest_sort_key_bits,
    }
}

fn write_nodes(path: &Path, ids: &[i64]) -> anyhow::Result<()> {
    let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef],
    )?;
    write_parquet(path, &schema, &[batch])
}

fn write_edges(path: &Path, edges: &[EdgeRow], with_sort_key: bool) -> anyhow::Result<()> {
    let mut fields = vec![
        Field::new("src_id", DataType::Int64, false),
        Field::new("dst_id", DataType::Int64, false),
    ];
    if with_sort_key {
        fields.push(Field::new("dest_sort_key", DataType::Int64, false));
    }
    let schema = Schema::new(fields);
    let src: Vec<i64> = edges.iter().map(|e| e.src).collect();
    let dst: Vec<i64> = edges.iter().map(|e| e.dst).collect();
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(src)) as ArrayRef,
        Arc::new(Int64Array::from(dst)) as ArrayRef,
    ];
    if with_sort_key {
        let dsk: Vec<i64> = edges
            .iter()
            .map(|e| e.dest_sort_key.unwrap_or_default())
            .collect();
        cols.push(Arc::new(Int64Array::from(dsk)) as ArrayRef);
    }
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), cols)?;
    write_parquet(path, &schema, &[batch])
}

fn write_parquet(path: &Path, schema: &Schema, batches: &[RecordBatch]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let props = WriterProperties::builder().build();
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    Ok(())
}

fn decode_relation(
    reader: &ShardReader,
    node_idx: u64,
    relation_idx: usize,
) -> Vec<(u64, Option<u64>)> {
    let view = reader.edges_for_node(node_idx).unwrap();
    let segments = view.relation_segments().unwrap();
    let seg = &segments[relation_idx];
    decode_segment(seg)
}

fn decode_segment(seg: &RelationSegment) -> Vec<(u64, Option<u64>)> {
    let mut out = Vec::new();
    if seg.count == 0 {
        return out;
    }
    let mut buf = [0u8; 16];
    for chunk in seg.data.chunks(seg.layout.entry_bytes) {
        buf[..seg.layout.entry_bytes].copy_from_slice(chunk);
        let val = u128::from_le_bytes(buf);
        let dst = (val & seg.layout.dst_mask as u128) as u64;
        let sort = seg
            .layout
            .dest_sort_key_bits
            .map(|bits| (val >> seg.layout.dst_bits as u128) & ((1u128 << bits) - 1))
            .map(|v| v as u64);
        out.push((dst, sort));
        buf.fill(0);
    }
    out
}

fn relation_layouts(
    relation_metas: Vec<EdgeRelationMetadata>,
    node_types: Vec<NodeTypeMetadata>,
) -> Arc<Vec<diskgraph_index::layout::RelationLayout>> {
    let layouts =
        diskgraph_index::builder::compute_relation_layouts(&relation_metas, &node_types).unwrap();
    Arc::new(layouts)
}

fn build_simple_shard(
    nodes: &[i64],
    edges: &[EdgeRow],
) -> (
    tempfile::TempDir,
    std::path::PathBuf,
    Arc<Vec<diskgraph_index::layout::RelationLayout>>,
) {
    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, nodes).unwrap();
    write_edges(&edge_path, edges, false).unwrap();
    let node_meta = node_meta("User", *nodes.iter().max().unwrap_or(&0) as u64);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let output_dir = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: output_dir.clone(),
        node_meta: node_meta.clone(),
        node_types: vec![node_meta.clone()],
        relation_metas: vec![relation_meta.clone()],
        shard_index: 0,
    });
    let built = builder
        .build(
            &NodeShardInput {
                node_type: "User".into(),
                shard_index: 0,
                parquet: ParquetSource::LocalFile(node_path),
            },
            &[EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::LocalFile(edge_path),
                has_dest_sort_key: false,
            }],
        )
        .unwrap();
    (dir, output_dir, built.layouts.clone())
}

fn write_custom_nodes(path: &Path, col: &str, ids: &[i64]) -> anyhow::Result<()> {
    let schema = Schema::new(vec![Field::new(col, DataType::Int64, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef],
    )?;
    write_parquet(path, &schema, &[batch])
}

fn write_wrong_type_nodes(path: &Path, col: &str, vals: &[&str]) -> anyhow::Result<()> {
    let schema = Schema::new(vec![Field::new(col, DataType::Utf8, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(vals.to_vec())) as ArrayRef],
    )?;
    write_parquet(path, &schema, &[batch])
}

fn write_wrong_type_edges(path: &Path, src_col: &[&str], dst_col: &[&str]) -> anyhow::Result<()> {
    let schema = Schema::new(vec![
        Field::new("src_id", DataType::Utf8, false),
        Field::new("dst_id", DataType::Utf8, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(StringArray::from(src_col.to_vec())) as ArrayRef,
            Arc::new(StringArray::from(dst_col.to_vec())) as ArrayRef,
        ],
    )?;
    write_parquet(path, &schema, &[batch])
}

fn varint_len_u64(v: u64) -> usize {
    if v < (1 << 7) {
        1
    } else if v < (1 << 14) {
        2
    } else if v < (1 << 21) {
        3
    } else if v < (1 << 28) {
        4
    } else if v < (1 << 35) {
        5
    } else if v < (1 << 42) {
        6
    } else if v < (1 << 49) {
        7
    } else if v < (1 << 56) {
        8
    } else if v < (1 << 63) {
        9
    } else {
        10
    }
}
