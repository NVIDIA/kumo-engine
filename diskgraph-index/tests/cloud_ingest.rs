#![cfg(feature = "cloud-ingest-tests")]

use std::sync::Arc;

use diskgraph_index::builder::{ShardBuildParams, ShardBuilder};
use diskgraph_index::reader::ShardReader;
use diskgraph_index::streaming::{
    AdlsLocation, EdgeShardInput, GcsLocation, NodeShardInput, ParquetSource, S3Location,
};
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use tempfile::tempdir;
use tokio::runtime::Runtime;

#[derive(Debug, Clone)]
struct EdgeRow {
    src: i64,
    dst: i64,
    dest_sort_key: Option<i64>,
}

fn node_meta(name: &str, max_id: u64) -> diskgraph_core::metadata::NodeTypeMetadata {
    diskgraph_core::metadata::NodeTypeMetadata {
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
) -> diskgraph_core::metadata::EdgeRelationMetadata {
    diskgraph_core::metadata::EdgeRelationMetadata {
        name: name.into(),
        src_type: src.into(),
        dst_type: dst.into(),
        shards: vec![],
        dest_sort_key_bits,
    }
}

fn write_nodes(path: &std::path::Path, ids: &[i64]) -> anyhow::Result<()> {
    use arrow_array::{ArrayRef, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::arrow_writer::ArrowWriter;

    let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    let batch = arrow_array::RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef],
    )?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn write_edges(
    path: &std::path::Path,
    edges: &[EdgeRow],
    with_sort_key: bool,
) -> anyhow::Result<()> {
    use arrow_array::{ArrayRef, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::arrow_writer::ArrowWriter;

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
    let batch = arrow_array::RecordBatch::try_new(Arc::new(schema.clone()), cols)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), None)?;
    writer.write(&batch)?;
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

/// Helper to upload a local file to an ObjectStore and return the (store, location).
fn upload_to_store(
    rt: &Runtime,
    store: Arc<dyn ObjectStore>,
    bucket_key: &str,
    local_path: &std::path::Path,
) -> (Arc<dyn ObjectStore>, ObjectPath) {
    let location = ObjectPath::from(bucket_key);
    let bytes = std::fs::read(local_path).expect("read local parquet");
    rt.block_on(async {
        store
            .put(&location, bytes.into())
            .await
            .expect("upload to store");
    });
    (store, location)
}

#[test]
fn s3_ingest_round_trip() {
    let rt = Runtime::new().unwrap();
    // Use in-memory object store to exercise the object-store reader without Docker.
    let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());

    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0, 1]).unwrap();
    write_edges(
        &edge_path,
        &[
            EdgeRow {
                src: 0,
                dst: 1,
                dest_sort_key: None,
            },
            EdgeRow {
                src: 1,
                dst: 0,
                dest_sort_key: None,
            },
        ],
        false,
    )
    .unwrap();

    let (_s, node_loc) = upload_to_store(&rt, store.clone(), "nodes.parquet", &node_path);
    let (_s2, edge_loc) = upload_to_store(&rt, store.clone(), "edges.parquet", &edge_path);

    let node_meta = node_meta("User", 10);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let out = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: out.clone(),
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
                parquet: ParquetSource::InMemory {
                    store: store.clone(),
                    location: node_loc,
                },
            },
            &[EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::InMemory {
                    store: store.clone(),
                    location: edge_loc,
                },
                has_dest_sort_key: false,
            }],
        )
        .unwrap();

    let reader = ShardReader::open(&out, built.layouts.clone()).unwrap();
    let decoded = decode_relation(&reader, 0, 0);
    assert_eq!(decoded, vec![(1, None)]);
}

#[test]
fn gcs_ingest_round_trip() {
    let rt = Runtime::new().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());

    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0]).unwrap();
    write_edges(
        &edge_path,
        &[EdgeRow {
            src: 0,
            dst: 42,
            dest_sort_key: None,
        }],
        false,
    )
    .unwrap();

    let (_s, node_loc) = upload_to_store(&rt, store.clone(), "nodes.parquet", &node_path);
    let (_s2, edge_loc) = upload_to_store(&rt, store.clone(), "edges.parquet", &edge_path);

    let node_meta = node_meta("User", 100);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let out = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: out.clone(),
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
                parquet: ParquetSource::InMemory {
                    store: store.clone(),
                    location: node_loc,
                },
            },
            &[EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::InMemory {
                    store: store.clone(),
                    location: edge_loc,
                },
                has_dest_sort_key: false,
            }],
        )
        .unwrap();
    let reader = ShardReader::open(&out, built.layouts.clone()).unwrap();
    let decoded = decode_relation(&reader, 0, 0);
    assert_eq!(decoded, vec![(42, None)]);
}

#[test]
fn adls_ingest_round_trip() {
    let rt = Runtime::new().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(object_store::memory::InMemory::new());

    let dir = tempdir().unwrap();
    let node_path = dir.path().join("nodes.parquet");
    let edge_path = dir.path().join("edges.parquet");
    write_nodes(&node_path, &[0, 1]).unwrap();
    write_edges(
        &edge_path,
        &[EdgeRow {
            src: 1,
            dst: 7,
            dest_sort_key: None,
        }],
        false,
    )
    .unwrap();

    let (_s, node_loc) = upload_to_store(&rt, store.clone(), "nodes.parquet", &node_path);
    let (_s2, edge_loc) = upload_to_store(&rt, store.clone(), "edges.parquet", &edge_path);

    let node_meta = node_meta("User", 10);
    let relation_meta = relation_meta("Follows", "User", "User", None);
    let out = dir.path().join("out");
    let builder = ShardBuilder::new(ShardBuildParams {
        output_dir: out.clone(),
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
                parquet: ParquetSource::InMemory {
                    store: store.clone(),
                    location: node_loc,
                },
            },
            &[EdgeShardInput {
                relation: "Follows".into(),
                source_shard_index: 0,
                parquet: ParquetSource::InMemory {
                    store: store.clone(),
                    location: edge_loc,
                },
                has_dest_sort_key: false,
            }],
        )
        .unwrap();

    let reader = ShardReader::open(&out, built.layouts.clone()).unwrap();
    let decoded = decode_relation(&reader, 1, 0);
    assert_eq!(decoded, vec![(7, None)]);
}
