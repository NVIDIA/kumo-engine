use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, EdgeShardMetadata, NodeShardMetadata, NodeTypeMetadata,
};
use diskgraph_index::graph::{GraphIndexBuilder, GraphIndexReader};
use std::fs;
use std::io::Write;
use std::path::Path;
use tempfile::tempdir;

#[derive(Clone)]
struct EdgeRow {
    src: i64,
    dst: i64,
}

#[test]
fn graph_builder_and_reader_multi_table() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("dataset");
    fs::create_dir_all(&root).unwrap();

    let edges_root = root.join("edges");
    fs::create_dir_all(&edges_root).unwrap();

    let follows0 = edges_root.join("follows/part-00000.bin");
    let follows1 = edges_root.join("follows/part-00001.bin");
    let likes0 = edges_root.join("likes/part-00000.bin");
    let likes1 = edges_root.join("likes/part-00001.bin");
    let tagged0 = edges_root.join("tagged/part-00000.bin");
    fs::create_dir_all(follows0.parent().unwrap()).unwrap();
    fs::create_dir_all(likes0.parent().unwrap()).unwrap();
    fs::create_dir_all(tagged0.parent().unwrap()).unwrap();

    write_edges(
        &follows0,
        &[EdgeRow { src: 0, dst: 1 }, EdgeRow { src: 1, dst: 0 }],
    )
    .unwrap();
    write_edges(&follows1, &[EdgeRow { src: 2, dst: 3 }]).unwrap();
    write_edges(
        &likes0,
        &[EdgeRow { src: 0, dst: 0 }, EdgeRow { src: 1, dst: 1 }],
    )
    .unwrap();
    write_edges(&likes1, &[EdgeRow { src: 2, dst: 0 }]).unwrap();
    write_edges(&tagged0, &[EdgeRow { src: 0, dst: 0 }]).unwrap();

    let metadata = DatasetMetadata {
        timestamp_unit: diskgraph_core::time::TimeUnit::Seconds,
        dataset_name: "graph-index-test".into(),
        node_types: vec![
            NodeTypeMetadata {
                name: "User".into(),
                max_id: 3,
                has_node_data: false,
                node_data_compression: None,
                primary_key_map: None,
                create_time_col: None,
                end_time_col: None,
                shards: vec![
                    NodeShardMetadata {
                        min_id: 0,
                        max_id: 1,
                        node_data_paths: None,
                    },
                    NodeShardMetadata {
                        min_id: 2,
                        max_id: 3,
                        node_data_paths: None,
                    },
                ],
            },
            NodeTypeMetadata {
                name: "Item".into(),
                max_id: 1,
                has_node_data: false,
                node_data_compression: None,
                primary_key_map: None,
                create_time_col: None,
                end_time_col: None,
                shards: vec![NodeShardMetadata {
                    min_id: 0,
                    max_id: 1,
                    node_data_paths: None,
                }],
            },
        ],
        edge_relations: vec![
            EdgeRelationMetadata {
                name: "Follows".into(),
                src_type: "User".into(),
                dst_type: "User".into(),
                shards: vec![
                    EdgeShardMetadata {
                        path: "edges/follows/part-00000.bin".into(),
                        src_min_id: 0,
                        src_max_id: 1,
                    },
                    EdgeShardMetadata {
                        path: "edges/follows/part-00001.bin".into(),
                        src_min_id: 2,
                        src_max_id: 3,
                    },
                ],
                dest_sort_key_bits: None,
                dest_sort_key: None,
                format: diskgraph_core::metadata::EdgeStorageFormat::Binary,
                reverse_of: None,
            },
            EdgeRelationMetadata {
                name: "Likes".into(),
                src_type: "User".into(),
                dst_type: "Item".into(),
                shards: vec![
                    EdgeShardMetadata {
                        path: "edges/likes/part-00000.bin".into(),
                        src_min_id: 0,
                        src_max_id: 1,
                    },
                    EdgeShardMetadata {
                        path: "edges/likes/part-00001.bin".into(),
                        src_min_id: 2,
                        src_max_id: 3,
                    },
                ],
                dest_sort_key_bits: None,
                dest_sort_key: None,
                format: diskgraph_core::metadata::EdgeStorageFormat::Binary,
                reverse_of: None,
            },
            EdgeRelationMetadata {
                name: "Tagged".into(),
                src_type: "Item".into(),
                dst_type: "User".into(),
                shards: vec![EdgeShardMetadata {
                    path: "edges/tagged/part-00000.bin".into(),
                    src_min_id: 0,
                    src_max_id: 1,
                }],
                dest_sort_key_bits: None,
                dest_sort_key: None,
                format: diskgraph_core::metadata::EdgeStorageFormat::Binary,
                reverse_of: None,
            },
        ],
    };

    let meta_path = root.join("index_metadata.json");
    fs::write(&meta_path, serde_json::to_vec_pretty(&metadata).unwrap()).unwrap();

    let output = dir.path().join("index");
    let builder = GraphIndexBuilder::from_metadata_path(&meta_path, &output).unwrap();
    builder.build(None).unwrap();

    let reader = GraphIndexReader::open(&output.join("index_metadata.json"), &output).unwrap();
    let user = reader.table("User").unwrap();
    let item = reader.table("Item").unwrap();

    assert_eq!(user.shard_count(), 2);
    assert_eq!(item.shard_count(), 1);
    assert_eq!(user.node_count(), 4);
    assert_eq!(item.node_count(), 2);

    let user0 = user.edges_for_node(0).unwrap();
    let user0_segments = user0.relation_segments().unwrap();
    assert_eq!(user0_segments.len(), 2);
    assert_eq!(user0_segments[0].count, 1);
    assert_eq!(user0_segments[1].count, 1);

    let user2 = user.edges_for_node(2).unwrap();
    let user2_segments = user2.relation_segments().unwrap();
    assert_eq!(user2_segments.len(), 2);
    assert_eq!(user2_segments[0].count, 1);
    assert_eq!(user2_segments[1].count, 1);

    let item0 = item.edges_for_node(0).unwrap();
    let item0_segments = item0.relation_segments().unwrap();
    assert_eq!(item0_segments.len(), 1);
    assert_eq!(item0_segments[0].count, 1);
}

fn write_edges(path: &Path, edges: &[EdgeRow]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(fs::File::create(path)?);
    writer.write_all(&[8u8, 8u8, 0u8])?;
    for edge in edges {
        write_u64_width(&mut writer, edge.src as u64, 8)?;
        write_u64_width(&mut writer, edge.dst as u64, 8)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_u64_width(
    writer: &mut std::io::BufWriter<std::fs::File>,
    value: u64,
    width: u8,
) -> anyhow::Result<()> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes[..width as usize])?;
    Ok(())
}
