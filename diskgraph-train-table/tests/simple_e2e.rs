use std::fs::{self, File};
use std::path::Path;

use diskgraph_core::metadata::{
    DatasetMetadata, NodeDataCompression, NodeShardMetadata, NodeTypeMetadata,
};
use diskgraph_rowpack::{
    ColumnContext, ColumnEncoding, IngestedType, RowPackContext, RowPackEncoder, RowPackScratch,
    ValueRef,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::tempdir;

use diskgraph_train_table::config::{JobConfig, QueryOptions, SplitConfig, SplitType};
use diskgraph_train_table::pipeline::run_job;

#[test]
fn non_temporal_predict_comparison_column_ignores_null_e2e() {
    let dir = tempdir().unwrap();
    let index_root = dir.path().join("index");
    fs::create_dir_all(&index_root).unwrap();

    let ctx = RowPackContext {
        version: 3,
        columns: vec![
            ColumnContext {
                name: "customer_id".to_string(),
                logical_type: IngestedType::Int64,
                encoding: ColumnEncoding::Int {
                    signed: true,
                    varint: true,
                    bits: 64,
                },
            },
            ColumnContext {
                name: "active".to_string(),
                logical_type: IngestedType::Float64,
                encoding: ColumnEncoding::Float {
                    storage: diskgraph_rowpack::FloatStorage::F64,
                },
            },
        ],
        estimated_row_bytes: 16,
    };
    write_rowpack_ctx(&index_root.join("nodes/customers/node_data.ctx"), &ctx);

    let metadata = DatasetMetadata {
        dataset_name: "test".to_string(),
        node_types: vec![NodeTypeMetadata {
            name: "customers".to_string(),
            max_id: 2,
            has_node_data: true,
            node_data_compression: Some(NodeDataCompression {
                codec: "rowpack".to_string(),
                context_path: "nodes/customers/node_data.ctx".to_string(),
            }),
            primary_key_map: None,
            shards: vec![NodeShardMetadata {
                min_id: 0,
                max_id: 2,
                node_data_paths: None,
            }],
        }],
        edge_relations: Vec::new(),
    };
    fs::write(
        index_root.join("index_metadata.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();

    let encoder = RowPackEncoder::new(ctx.clone()).expect("encoder");
    let mut scratch = RowPackScratch::new(encoder.context());
    let mut row1 = Vec::new();
    encoder
        .encode_row_into(
            &[ValueRef::Int64(1), ValueRef::Float64(1.0)],
            &mut row1,
            &mut scratch,
        )
        .expect("encode row");
    let mut row2 = Vec::new();
    encoder
        .encode_row_into(
            &[ValueRef::Int64(2), ValueRef::Null],
            &mut row2,
            &mut scratch,
        )
        .expect("encode row");
    write_node_data(
        &index_root.join("nodes/customers/shard-00000/node_data.bin"),
        vec![row1, row2],
    );

    let output_path = dir.path().join("train.parquet");
    let config = JobConfig {
        job_name: None,
        index_path: index_root.to_string_lossy().to_string(),
        query: "PREDICT customers.active = 1 FOR EACH customers.customer_id".to_string(),
        output_parquet: output_path.to_string_lossy().to_string(),
        split: SplitConfig {
            split_type: SplitType::Random,
            fractions: vec![1.0, 0.0, 0.0],
            seed: Some(0),
            time: None,
            range: None,
        },
        options: QueryOptions::default(),
        validation_output: None,
    };

    run_job(&config).unwrap();

    let file = File::open(&output_path).unwrap();
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    assert_eq!(batch.num_rows(), 1);
    let target = batch
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::BooleanArray>()
        .unwrap();
    assert!(target.value(0));
}

fn write_rowpack_ctx(path: &Path, ctx: &RowPackContext) {
    let bytes = ctx.to_bytes().expect("ctx bytes");
    fs::create_dir_all(path.parent().unwrap()).expect("ctx dir");
    fs::write(path, bytes).expect("write ctx");
}

fn write_node_data(path: &Path, rows: Vec<Vec<u8>>) {
    fs::create_dir_all(path.parent().unwrap()).expect("node data dir");
    let mut file = File::create(path).expect("node data");
    for row in rows {
        let len = row.len() as u32;
        std::io::Write::write_all(&mut file, &len.to_le_bytes()).expect("write len");
        std::io::Write::write_all(&mut file, &row).expect("write row");
    }
}
