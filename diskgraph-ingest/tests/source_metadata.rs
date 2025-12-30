use diskgraph_ingest::source_metadata::{
    ColumnSpec, ColumnType, EdgeTableSpec, ForeignKeySpec, RelationshipSpec, SourceMetadata,
    TableFormat, TableSpec,
};

fn base_table(name: &str) -> TableSpec {
    TableSpec {
        name: name.into(),
        format: TableFormat::Parquet,
        paths: vec!["/tmp/table.parquet".into()],
        columns: vec![
            ColumnSpec {
                name: "id".into(),
                dtype: ColumnType::Int64,
                fixed_len: None,
                precision: None,
                scale: None,
            },
            ColumnSpec {
                name: "name".into(),
                dtype: ColumnType::String,
                fixed_len: Some(8),
                precision: None,
                scale: None,
            },
        ],
        primary_key: Some("id".into()),
        create_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
    }
}

fn base_metadata() -> SourceMetadata {
    SourceMetadata {
        dataset_name: "test".into(),
        tables: vec![base_table("User"), base_table("Device")],
        relationships: Vec::new(),
        rowpack: None,
    }
}

fn assert_err_contains(meta: SourceMetadata, needle: &str) {
    let err = meta.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains(needle),
        "expected error to contain '{}', got '{}'",
        needle,
        msg
    );
}

#[test]
fn validate_minimal_metadata() {
    let meta = SourceMetadata {
        dataset_name: "test".into(),
        tables: vec![base_table("User")],
        relationships: Vec::new(),
        rowpack: None,
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
                precision: None,
                scale: None,
            }],
            primary_key: Some("id".into()),
            create_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
        }],
        relationships: Vec::new(),
        rowpack: None,
    };
    assert!(meta.validate().is_err());
}

#[test]
fn validate_empty_dataset_name() {
    let mut meta = base_metadata();
    meta.dataset_name = "   ".into();
    assert_err_contains(meta, "dataset_name is required");
}

#[test]
fn validate_duplicate_table_names() {
    let mut meta = base_metadata();
    meta.tables.push(base_table("User"));
    assert_err_contains(meta, "duplicate table name User");
}

#[test]
fn validate_table_no_paths() {
    let mut meta = base_metadata();
    meta.tables[0].paths.clear();
    assert_err_contains(meta, "table User has no paths");
}

#[test]
fn validate_table_no_columns() {
    let mut meta = base_metadata();
    meta.tables[0].columns.clear();
    assert_err_contains(meta, "table User has no columns");
}

#[test]
fn validate_duplicate_column_names() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "id".into(),
        dtype: ColumnType::Int64,
        fixed_len: None,
        precision: None,
        scale: None,
    });
    assert_err_contains(meta, "table User has duplicate column id");
}

#[test]
fn validate_string_pk_requires_fixed_len() {
    let mut meta = base_metadata();
    meta.tables[0].columns[0].name = "pk".into();
    meta.tables[0].columns[0].dtype = ColumnType::String;
    meta.tables[0].columns[0].fixed_len = None;
    meta.tables[0].primary_key = Some("pk".into());
    assert_err_contains(meta, "primary key pk requires fixed_len");
}

#[test]
fn validate_pk_invalid_type() {
    let mut meta = base_metadata();
    meta.tables[0].columns[0].dtype = ColumnType::Float32;
    assert_err_contains(meta, "primary_key id must be int or string/bytes");
}

#[test]
fn validate_weight_invalid_type() {
    let mut meta = base_metadata();
    meta.tables[0].weight = Some("name".into());
    assert_err_contains(meta, "weight column name must be numeric");
}

#[test]
fn validate_create_time_invalid_type() {
    let mut meta = base_metadata();
    meta.tables[0].create_time = Some("name".into());
    assert_err_contains(meta, "create_time column name must be date/timestamp");
}

#[test]
fn validate_decimal_missing_precision() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "amount".into(),
        dtype: ColumnType::Decimal128,
        fixed_len: None,
        precision: None,
        scale: Some(2),
    });
    assert_err_contains(meta, "missing precision");
}

#[test]
fn validate_decimal_missing_scale() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "amount".into(),
        dtype: ColumnType::Decimal128,
        fixed_len: None,
        precision: Some(10),
        scale: None,
    });
    assert_err_contains(meta, "missing scale");
}

#[test]
fn validate_decimal_precision_out_of_range() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "amount".into(),
        dtype: ColumnType::Decimal128,
        fixed_len: None,
        precision: Some(50),
        scale: Some(2),
    });
    assert_err_contains(meta, "precision out of range");
}

#[test]
fn validate_decimal_scale_out_of_range() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "amount".into(),
        dtype: ColumnType::Decimal128,
        fixed_len: None,
        precision: Some(10),
        scale: Some(20),
    });
    assert_err_contains(meta, "scale out of range");
}

#[test]
fn validate_decimal_precision_on_non_decimal() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "bad".into(),
        dtype: ColumnType::Int64,
        fixed_len: None,
        precision: Some(10),
        scale: None,
    });
    assert_err_contains(meta, "precision/scale only valid for decimal");
}

#[test]
fn validate_edge_fk_missing_column() {
    let mut meta = base_metadata();
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        format: TableFormat::Parquet,
        paths: vec!["/tmp/edge.parquet".into()],
        columns: vec![ColumnSpec {
            name: "src_id".into(),
            dtype: ColumnType::Int64,
            fixed_len: None,
            precision: None,
            scale: None,
        }],
        primary_key: None,
        create_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                column: "src_id".into(),
                ref_table: "User".into(),
                ref_column: "id".into(),
            },
            dst_fk: ForeignKeySpec {
                column: "dst_id".into(),
                ref_table: "Device".into(),
                ref_column: "id".into(),
            },
        }),
        rowpack: None,
    });
    assert_err_contains(meta, "edge fk column dst_id not found");
}

#[test]
fn validate_edge_fk_missing_table() {
    let mut meta = base_metadata();
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        format: TableFormat::Parquet,
        paths: vec!["/tmp/edge.parquet".into()],
        columns: vec![
            ColumnSpec {
                name: "src_id".into(),
                dtype: ColumnType::Int64,
                fixed_len: None,
                precision: None,
                scale: None,
            },
            ColumnSpec {
                name: "dst_id".into(),
                dtype: ColumnType::Int64,
                fixed_len: None,
                precision: None,
                scale: None,
            },
        ],
        primary_key: None,
        create_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                column: "src_id".into(),
                ref_table: "Missing".into(),
                ref_column: "id".into(),
            },
            dst_fk: ForeignKeySpec {
                column: "dst_id".into(),
                ref_table: "Device".into(),
                ref_column: "id".into(),
            },
        }),
        rowpack: None,
    });
    assert_err_contains(meta, "edge table Edge references missing table Missing");
}

#[test]
fn validate_edge_fk_not_pk() {
    let mut meta = base_metadata();
    meta.tables[1].primary_key = Some("name".into());
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        format: TableFormat::Parquet,
        paths: vec!["/tmp/edge.parquet".into()],
        columns: vec![
            ColumnSpec {
                name: "src_id".into(),
                dtype: ColumnType::Int64,
                fixed_len: None,
                precision: None,
                scale: None,
            },
            ColumnSpec {
                name: "dst_id".into(),
                dtype: ColumnType::Int64,
                fixed_len: None,
                precision: None,
                scale: None,
            },
        ],
        primary_key: None,
        create_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                column: "src_id".into(),
                ref_table: "User".into(),
                ref_column: "id".into(),
            },
            dst_fk: ForeignKeySpec {
                column: "dst_id".into(),
                ref_table: "Device".into(),
                ref_column: "id".into(),
            },
        }),
        rowpack: None,
    });
    assert_err_contains(meta, "edge table Edge fk dst_id must reference primary key Device.id");
}

#[test]
fn validate_relationship_missing_table() {
    let mut meta = base_metadata();
    meta.relationships.push(RelationshipSpec {
        name: Some("bad".into()),
        src_table: "Missing".into(),
        src_column: "id".into(),
        dst_table: "User".into(),
        dst_column: "id".into(),
    });
    assert_err_contains(meta, "relationship src table Missing not found");
}

#[test]
fn validate_relationship_missing_column() {
    let mut meta = base_metadata();
    meta.relationships.push(RelationshipSpec {
        name: Some("bad".into()),
        src_table: "User".into(),
        src_column: "missing".into(),
        dst_table: "Device".into(),
        dst_column: "id".into(),
    });
    assert_err_contains(meta, "relationship src column missing not found in User");
}

#[test]
fn validate_relationship_no_pk() {
    let mut meta = base_metadata();
    meta.tables[0].primary_key = None;
    meta.tables[1].primary_key = None;
    meta.relationships.push(RelationshipSpec {
        name: Some("bad".into()),
        src_table: "User".into(),
        src_column: "id".into(),
        dst_table: "Device".into(),
        dst_column: "id".into(),
    });
    assert_err_contains(meta, "must reference at least one primary key");
}

#[test]
fn validate_relationship_incompatible_types() {
    let mut meta = base_metadata();
    meta.tables[1].columns[0].dtype = ColumnType::String;
    meta.tables[1].columns[0].fixed_len = Some(8);
    meta.relationships.push(RelationshipSpec {
        name: Some("bad".into()),
        src_table: "User".into(),
        src_column: "id".into(),
        dst_table: "Device".into(),
        dst_column: "id".into(),
    });
    assert_err_contains(meta, "have incompatible types");
}

#[test]
fn validate_relationship_duplicate_reverse_pk_pk() {
    let mut meta = base_metadata();
    meta.relationships.push(RelationshipSpec {
        name: Some("a".into()),
        src_table: "User".into(),
        src_column: "id".into(),
        dst_table: "Device".into(),
        dst_column: "id".into(),
    });
    meta.relationships.push(RelationshipSpec {
        name: Some("b".into()),
        src_table: "Device".into(),
        src_column: "id".into(),
        dst_table: "User".into(),
        dst_column: "id".into(),
    });
    assert_err_contains(meta, "has reverse already declared");
}

#[test]
fn validate_empty_table_name() {
    let mut meta = base_metadata();
    meta.tables[0].name = "  ".into();
    assert_err_contains(meta, "table name cannot be empty");
}

#[test]
fn validate_empty_column_name() {
    let mut meta = base_metadata();
    meta.tables[0].columns[0].name = " ".into();
    assert_err_contains(meta, "table User has empty column name");
}

#[test]
fn validate_duplicate_relationship_same_direction() {
    let mut meta = base_metadata();
    meta.relationships.push(RelationshipSpec {
        name: Some("a".into()),
        src_table: "User".into(),
        src_column: "id".into(),
        dst_table: "Device".into(),
        dst_column: "id".into(),
    });
    meta.relationships.push(RelationshipSpec {
        name: Some("b".into()),
        src_table: "User".into(),
        src_column: "id".into(),
        dst_table: "Device".into(),
        dst_column: "id".into(),
    });
    assert_err_contains(meta, "is duplicated");
}
