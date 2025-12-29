use diskgraph_ingest::source_metadata::{ColumnSpec, ColumnType, SourceMetadata, TableFormat, TableSpec};

#[test]
fn validate_minimal_metadata() {
    let meta = SourceMetadata {
        dataset_name: "test".into(),
        tables: vec![TableSpec {
            name: "User".into(),
            format: TableFormat::Parquet,
            paths: vec!["/tmp/user.parquet".into()],
            columns: vec![
                ColumnSpec {
                    name: "id".into(),
                    dtype: ColumnType::Int64,
                    fixed_len: None,
                },
                ColumnSpec {
                    name: "name".into(),
                    dtype: ColumnType::String,
                    fixed_len: Some(8),
                },
            ],
            primary_key: Some("id".into()),
            create_time: None,
            weight: None,
            edge_table: None,
        }],
        relationships: Vec::new(),
    };
    assert!(meta.validate().is_ok());
}

#[test]
fn validate_missing_pk_column() {
    let meta = SourceMetadata {
        dataset_name: "test".into(),
        tables: vec![TableSpec {
            name: "User".into(),
            format: TableFormat::Parquet,
            paths: vec!["/tmp/user.parquet".into()],
            columns: vec![ColumnSpec {
                name: "name".into(),
                dtype: ColumnType::String,
                fixed_len: Some(8),
            }],
            primary_key: Some("id".into()),
            create_time: None,
            weight: None,
            edge_table: None,
        }],
        relationships: Vec::new(),
    };
    assert!(meta.validate().is_err());
}
