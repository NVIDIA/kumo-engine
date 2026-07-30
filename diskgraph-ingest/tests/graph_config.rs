use diskgraph_ingest::graph_config::{
    ColumnSpec, DataSource, DirectionSpec, EdgeTableSpec, FKeySpec, ForeignKeySpec, GraphConfig,
    IngestedType, SemanticType, TableSpec, TimeSpec, WeightSpec,
};

fn base_table(name: &str) -> TableSpec {
    TableSpec {
        name: name.into(),
        columns: vec![
            ColumnSpec {
                name: "id".into(),
                ingested_dtype: IngestedType::Int64,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            ColumnSpec {
                name: "name".into(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: Some(8),
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
        ],
        primary_key: Some("id".into()),
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/table.parquet".into()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    }
}

fn base_metadata() -> GraphConfig {
    GraphConfig {
        dataset_name: "test".into(),
        tables: vec![base_table("User"), base_table("Device")],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    }
}

fn assert_err_contains(meta: GraphConfig, needle: &str) {
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
    let meta = GraphConfig {
        dataset_name: "test".into(),
        tables: vec![base_table("User")],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    };
    assert!(meta.validate().is_ok());
}

#[test]
fn validate_rejects_unsupported_direct_multicategorical_dtype() {
    let mut table = base_table("User");
    table.columns.push(ColumnSpec {
        name: "bad_multicat".into(),
        ingested_dtype: IngestedType::Int64,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: None,
        scale: None,
        semantic_type: Some(SemanticType::Multicategorical),
        multicategorical_delimiter: None,
    });
    let meta = GraphConfig {
        dataset_name: "test".into(),
        tables: vec![table],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    };
    assert_err_contains(
        meta,
        "bad_multicat stype=multicategorical requires a list-shaped",
    );
}

#[test]
fn validate_missing_pk_column() {
    let meta = GraphConfig {
        dataset_name: "test".into(),
        tables: vec![TableSpec {
            name: "User".into(),
            columns: vec![ColumnSpec {
                name: "name".into(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: Some(8),
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            }],
            primary_key: Some("id".into()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec!["/tmp/user.parquet".into()],
            },

            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        }],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
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
fn validate_case_colliding_table_names() {
    let mut meta = base_metadata();
    meta.tables.push(base_table("user"));
    assert_err_contains(meta, "collide under lowercase normalization");
}

#[test]
fn validate_table_no_paths() {
    let mut meta = base_metadata();
    meta.tables[0].replace_with_local_parquet(Vec::new());
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
        ingested_dtype: IngestedType::Int64,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    assert_err_contains(meta, "table User has duplicate column id");
}

#[test]
fn validate_uuid_pk_allowed() {
    let mut meta = base_metadata();
    meta.tables[0].columns[0].ingested_dtype = IngestedType::Uuid;
    assert!(meta.validate().is_ok());
}

#[test]
fn validate_fixed_len_requires_string_bytes() {
    let mut meta = base_metadata();
    meta.tables[0].columns[0].fixed_len = Some(8);
    meta.tables[0].columns[0].ingested_dtype = IngestedType::Int64;
    assert_err_contains(meta, "fixed_len only valid for string/bytes");
}

#[test]
fn validate_pk_invalid_type() {
    let mut meta = base_metadata();
    meta.tables[0].columns[0].ingested_dtype = IngestedType::Bool;
    assert_err_contains(
        meta,
        "primary_key id must be numeric or string/bytes/uuid/uuid256/ip",
    );
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
        ingested_dtype: IngestedType::Decimal128,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: None,
        scale: Some(2),
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    assert_err_contains(meta, "missing precision");
}

#[test]
fn validate_decimal_missing_scale() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "amount".into(),
        ingested_dtype: IngestedType::Decimal128,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: Some(10),
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    assert_err_contains(meta, "missing scale");
}

#[test]
fn validate_decimal_precision_out_of_range() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "amount".into(),
        ingested_dtype: IngestedType::Decimal128,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: Some(50),
        scale: Some(2),
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    assert_err_contains(meta, "precision out of range");
}

#[test]
fn validate_decimal_scale_out_of_range() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "amount".into(),
        ingested_dtype: IngestedType::Decimal128,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: Some(10),
        scale: Some(20),
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    assert_err_contains(meta, "scale out of range");
}

#[test]
fn validate_decimal_precision_on_non_decimal() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "bad".into(),
        ingested_dtype: IngestedType::Int64,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: Some(10),
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    assert_err_contains(meta, "precision/scale only valid for decimal");
}

#[test]
fn validate_decimal_precision_allowed_with_source_decimal() {
    let mut meta = base_metadata();
    meta.tables[0].columns.push(ColumnSpec {
        name: "castable".into(),
        ingested_dtype: IngestedType::Int64,
        source_dtype: Some("Decimal128(10, 2)".into()),
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: Some(10),
        scale: Some(2),
        semantic_type: None,
        multicategorical_delimiter: None,
    });
    assert!(meta.validate().is_ok());
}

#[test]
fn validate_edge_fk_missing_column() {
    let mut meta = base_metadata();
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        columns: vec![ColumnSpec {
            name: "src_id".into(),
            ingested_dtype: IngestedType::Int64,
            source_dtype: None,
            timestamp_format: None,
            timestamp_unit: None,
            fixed_len: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        }],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "User".into(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "dst_id".into(),
                ref_table: "Device".into(),
            },
            forward: None,
            reverse: None,
        }),
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/edge.parquet".into()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    });
    assert_err_contains(meta, "edge fk column dst_id not found");
}

#[test]
fn validate_edge_fk_missing_table() {
    let mut meta = base_metadata();
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        columns: vec![
            ColumnSpec {
                name: "src_id".into(),
                ingested_dtype: IngestedType::Int64,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            ColumnSpec {
                name: "dst_id".into(),
                ingested_dtype: IngestedType::Int64,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
        ],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "Missing".into(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "dst_id".into(),
                ref_table: "Device".into(),
            },
            forward: None,
            reverse: None,
        }),
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/edge.parquet".into()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    });
    assert_err_contains(meta, "edge table Edge references missing table Missing");
}

#[test]
fn validate_edge_fk_ref_table_no_pk() {
    let mut meta = base_metadata();
    meta.tables[1].primary_key = None; // Device has no PK
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        columns: vec![
            ColumnSpec {
                name: "src_id".into(),
                ingested_dtype: IngestedType::Int64,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            ColumnSpec {
                name: "dst_id".into(),
                ingested_dtype: IngestedType::Int64,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
        ],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "User".into(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "dst_id".into(),
                ref_table: "Device".into(),
            },
            forward: None,
            reverse: None,
        }),
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/edge.parquet".into()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    });
    assert_err_contains(meta, "references Device which has no primary key");
}

// Relationship validation tests removed - relationships field no longer exists in GraphConfig

#[test]
fn validate_edge_fk_int_string_ok() {
    let mut meta = base_metadata();
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        columns: vec![
            ColumnSpec {
                name: "src_id".into(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            ColumnSpec {
                name: "dst_id".into(),
                ingested_dtype: IngestedType::Int64,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
        ],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "User".into(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "dst_id".into(),
                ref_table: "Device".into(),
            },
            forward: None,
            reverse: None,
        }),
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/edge.parquet".into()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    });
    assert!(meta.validate().is_ok());
}

#[test]
fn validate_edge_fk_string_int_ok() {
    let mut meta = base_metadata();
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        columns: vec![
            ColumnSpec {
                name: "src_id".into(),
                ingested_dtype: IngestedType::Int64,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            ColumnSpec {
                name: "dst_id".into(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
        ],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "User".into(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "dst_id".into(),
                ref_table: "Device".into(),
            },
            forward: None,
            reverse: None,
        }),
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/edge.parquet".into()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    });
    assert!(meta.validate().is_ok());
}

#[test]
fn validate_edge_fk_string_bytes_ok() {
    let mut meta = base_metadata();
    meta.tables[0].columns[0].name = "pk".into();
    meta.tables[0].columns[0].ingested_dtype = IngestedType::Bytes;
    meta.tables[0].columns[0].fixed_len = Some(8);
    meta.tables[0].primary_key = Some("pk".into());
    meta.tables[1].columns[0].name = "pk".into();
    meta.tables[1].columns[0].ingested_dtype = IngestedType::String;
    meta.tables[1].columns[0].fixed_len = Some(8);
    meta.tables[1].primary_key = Some("pk".into());
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        columns: vec![
            ColumnSpec {
                name: "src_id".into(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: Some(8),
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            ColumnSpec {
                name: "dst_id".into(),
                ingested_dtype: IngestedType::Bytes,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: Some(8),
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
        ],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "Device".into(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "dst_id".into(),
                ref_table: "User".into(),
            },
            forward: None,
            reverse: None,
        }),
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/edge.parquet".into()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    });
    assert!(meta.validate().is_ok());
}

#[test]
fn validate_edge_fk_string_bytes_len_mismatch() {
    let mut meta = base_metadata();
    meta.tables[0].columns[0].name = "pk".into();
    meta.tables[0].columns[0].ingested_dtype = IngestedType::Bytes;
    meta.tables[0].columns[0].fixed_len = Some(8);
    meta.tables[0].primary_key = Some("pk".into());
    meta.tables[1].columns[0].name = "pk".into();
    meta.tables[1].columns[0].ingested_dtype = IngestedType::String;
    meta.tables[1].columns[0].fixed_len = Some(8);
    meta.tables[1].primary_key = Some("pk".into());
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        columns: vec![
            ColumnSpec {
                name: "src_id".into(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: Some(4),
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
            ColumnSpec {
                name: "dst_id".into(),
                ingested_dtype: IngestedType::Bytes,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: Some(8),
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            },
        ],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "Device".into(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "dst_id".into(),
                ref_table: "User".into(),
            },
            forward: None,
            reverse: None,
        }),
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/edge.parquet".into()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    });
    assert!(meta.validate().is_ok());
}

#[test]
fn validate_edge_fk_uuid_ok() {
    let mut meta = base_metadata();
    meta.tables[0].columns[0].name = "pk".into();
    meta.tables[0].columns[0].ingested_dtype = IngestedType::Uuid;
    meta.tables[0].primary_key = Some("pk".into());
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        columns: vec![ColumnSpec {
            name: "src_id".into(),
            ingested_dtype: IngestedType::Uuid,
            source_dtype: None,
            timestamp_format: None,
            timestamp_unit: None,
            fixed_len: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        }],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "User".into(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "User".into(),
            },
            forward: None,
            reverse: None,
        }),
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/edge.parquet".into()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    });
    assert!(meta.validate().is_ok());
}

#[test]
fn validate_edge_fk_uuid_mismatch() {
    let mut meta = base_metadata();
    meta.tables[0].columns[0].name = "pk".into();
    meta.tables[0].columns[0].ingested_dtype = IngestedType::Uuid;
    meta.tables[0].primary_key = Some("pk".into());
    meta.tables.push(TableSpec {
        name: "Edge".into(),
        columns: vec![ColumnSpec {
            name: "src_id".into(),
            ingested_dtype: IngestedType::Bytes,
            source_dtype: None,
            timestamp_format: None,
            timestamp_unit: None,
            fixed_len: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        }],
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "User".into(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "src_id".into(),
                ref_table: "User".into(),
            },
            forward: None,
            reverse: None,
        }),
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/edge.parquet".into()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    });
    assert_err_contains(meta, "incompatible types");
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

// ============================================================================
// Tests for new schema types: DataSource, TimeSpec, WeightSpec, DirectionSpec,
// FKeySpec
// ============================================================================

mod new_schema_types {
    use super::*;

    // DataSource tests
    #[test]
    fn datasource_parquet_serialize() {
        let source = DataSource::Parquet {
            paths: vec!["s3://bucket/data/".into()],
        };
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains(r#""type":"parquet""#));
        assert!(json.contains(r#""paths""#));
    }

    #[test]
    fn datasource_parquet_deserialize() {
        let json = r#"{"type":"parquet","paths":["s3://bucket/data/"]}"#;
        let source: DataSource = serde_json::from_str(json).unwrap();
        match source {
            DataSource::Parquet { paths } => {
                assert_eq!(paths, vec!["s3://bucket/data/"]);
            }
            _ => panic!("expected Parquet variant"),
        }
    }

    #[test]
    fn datasource_snowflake_serialize() {
        let source = DataSource::Snowflake {
            connection_id: "conn1".into(),
            table_or_view: "my_table".into(),
        };
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains(r#""type":"snowflake""#));
        assert!(json.contains(r#""connection_id":"conn1""#));
        assert!(json.contains(r#""table_or_view":"my_table""#));
    }

    #[test]
    fn datasource_snowflake_deserialize() {
        let json = r#"{"type":"snowflake","connection_id":"conn1","table_or_view":"users"}"#;
        let source: DataSource = serde_json::from_str(json).unwrap();
        match source {
            DataSource::Snowflake {
                connection_id,
                table_or_view,
            } => {
                assert_eq!(connection_id, "conn1");
                assert_eq!(table_or_view, "users");
            }
            _ => panic!("expected Snowflake variant"),
        }
    }

    #[test]
    fn datasource_databricks_serialize() {
        let source = DataSource::Databricks {
            server_hostname: "adb.example.net".into(),
            warehouse_id: "warehouse".into(),
            catalog: "main".into(),
            schema: "default".into(),
            sql: "SELECT id FROM main.default.events".into(),
            auth_profile: Some("db_conn".into()),
        };
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains(r#""type":"databricks""#));
        assert!(json.contains(r#""server_hostname":"adb.example.net""#));
        assert!(json.contains(r#""warehouse_id":"warehouse""#));
        assert!(json.contains(r#""catalog":"main""#));
        assert!(json.contains(r#""schema":"default""#));
        assert!(json.contains(r#""sql":"SELECT id FROM main.default.events""#));
        assert!(json.contains(r#""auth_profile":"db_conn""#));
    }

    #[test]
    fn datasource_databricks_deserialize() {
        let json = r#"{"type":"databricks","server_hostname":"adb.example.net","warehouse_id":"wh","catalog":"cat","schema":"sch","sql":"SELECT id FROM cat.sch.tbl","auth_profile":"DATABRICKS"}"#;
        let source: DataSource = serde_json::from_str(json).unwrap();
        match source {
            DataSource::Databricks {
                server_hostname,
                warehouse_id,
                catalog,
                schema,
                sql,
                auth_profile,
            } => {
                assert_eq!(server_hostname, "adb.example.net");
                assert_eq!(warehouse_id, "wh");
                assert_eq!(catalog, "cat");
                assert_eq!(schema, "sch");
                assert_eq!(sql, "SELECT id FROM cat.sch.tbl");
                assert_eq!(auth_profile.as_deref(), Some("DATABRICKS"));
            }
            _ => panic!("expected Databricks variant"),
        }
    }

    #[test]
    fn documented_utf8_dtype_deserializes_as_string() {
        let column: ColumnSpec = serde_json::from_str(r#"{"name":"id","dtype":"utf8"}"#)
            .expect("the documented Databricks dtype must deserialize");
        assert_eq!(column.ingested_dtype, IngestedType::String);
    }

    #[test]
    fn datasource_bigquery_serialize() {
        let source = DataSource::BigQuery {
            connection_id: "bq_conn".into(),
            project: "my-project".into(),
            dataset: "my_dataset".into(),
            table: "my_table".into(),
        };
        let json = serde_json::to_string(&source).unwrap();
        assert!(json.contains(r#""type":"big_query""#));
        assert!(json.contains(r#""connection_id":"bq_conn""#));
        assert!(json.contains(r#""project":"my-project""#));
        assert!(json.contains(r#""dataset":"my_dataset""#));
        assert!(json.contains(r#""table":"my_table""#));
    }

    #[test]
    fn datasource_bigquery_deserialize() {
        let json = r#"{"type":"big_query","connection_id":"bq","project":"proj","dataset":"ds","table":"tbl"}"#;
        let source: DataSource = serde_json::from_str(json).unwrap();
        match source {
            DataSource::BigQuery {
                connection_id,
                project,
                dataset,
                table,
            } => {
                assert_eq!(connection_id, "bq");
                assert_eq!(project, "proj");
                assert_eq!(dataset, "ds");
                assert_eq!(table, "tbl");
            }
            _ => panic!("expected BigQuery variant"),
        }
    }

    // TimeSpec tests
    #[test]
    fn timespec_serialize_full() {
        let spec = TimeSpec {
            table: "events".into(),
            column: "created_at".into(),
            bitwidth: Some(16),
            anchor_time: Some(1609459200),
            unit: "day".into(),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""table":"events""#));
        assert!(json.contains(r#""column":"created_at""#));
        assert!(json.contains(r#""bitwidth":16"#));
        assert!(json.contains(r#""anchor_time":1609459200"#));
        assert!(json.contains(r#""unit":"day""#));
    }

    #[test]
    fn timespec_serialize_minimal() {
        let spec = TimeSpec {
            table: "t".into(),
            column: "ts".into(),
            bitwidth: None,
            anchor_time: None,
            unit: "s".into(),
        };
        let json = serde_json::to_string(&spec).unwrap();
        // bitwidth and anchor_time should be omitted
        assert!(!json.contains("bitwidth"));
        assert!(!json.contains("anchor_time"));
    }

    #[test]
    fn timespec_deserialize() {
        let json = r#"{"table":"orders","column":"order_time","unit":"ms"}"#;
        let spec: TimeSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.table, "orders");
        assert_eq!(spec.column, "order_time");
        assert_eq!(spec.unit, "ms");
        assert!(spec.bitwidth.is_none());
        assert!(spec.anchor_time.is_none());
    }

    #[test]
    fn timespec_deserialize_full() {
        let json = r#"{"table":"t","column":"c","bitwidth":32,"anchor_time":0,"unit":"hour"}"#;
        let spec: TimeSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.bitwidth, Some(32));
        assert_eq!(spec.anchor_time, Some(0));
        assert_eq!(spec.unit, "hour");
    }

    // WeightSpec tests
    #[test]
    fn weightspec_serialize() {
        let spec = WeightSpec {
            table: "purchases".into(),
            expr: "quantity * price".into(),
            dtype: "float32".into(),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""table":"purchases""#));
        assert!(json.contains(r#""expr":"quantity * price""#));
        assert!(json.contains(r#""dtype":"float32""#));
    }

    #[test]
    fn weightspec_deserialize() {
        let json = r#"{"table":"edges","expr":"weight","dtype":"float16"}"#;
        let spec: WeightSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.table, "edges");
        assert_eq!(spec.expr, "weight");
        assert_eq!(spec.dtype, "float16");
    }

    #[test]
    fn weightspec_complex_expr() {
        let spec = WeightSpec {
            table: "t".into(),
            expr: "log(amount + 1) * factor".into(),
            dtype: "float32".into(),
        };
        let json = serde_json::to_string(&spec).unwrap();
        let parsed: WeightSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.expr, "log(amount + 1) * factor");
    }

    // DirectionSpec tests
    #[test]
    fn directionspec_empty() {
        let spec = DirectionSpec::default();
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn directionspec_with_time_only() {
        let spec = DirectionSpec {
            time: Some(TimeSpec {
                table: "t".into(),
                column: "ts".into(),
                bitwidth: None,
                anchor_time: None,
                unit: "day".into(),
            }),
            weight: None,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""time""#));
        assert!(!json.contains(r#""weight""#));
    }

    #[test]
    fn directionspec_with_weight_only() {
        let spec = DirectionSpec {
            time: None,
            weight: Some(WeightSpec {
                table: "t".into(),
                expr: "w".into(),
                dtype: "byte".into(),
            }),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(!json.contains(r#""time""#));
        assert!(json.contains(r#""weight""#));
    }

    #[test]
    fn directionspec_with_both() {
        let spec = DirectionSpec {
            time: Some(TimeSpec {
                table: "t".into(),
                column: "ts".into(),
                bitwidth: Some(16),
                anchor_time: None,
                unit: "s".into(),
            }),
            weight: Some(WeightSpec {
                table: "t".into(),
                expr: "amount".into(),
                dtype: "uint16".into(),
            }),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""time""#));
        assert!(json.contains(r#""weight""#));
    }

    #[test]
    fn directionspec_deserialize_empty() {
        let json = "{}";
        let spec: DirectionSpec = serde_json::from_str(json).unwrap();
        assert!(spec.time.is_none());
        assert!(spec.weight.is_none());
    }

    // FKeySpec tests
    #[test]
    fn fkeyspec_minimal() {
        let spec = FKeySpec {
            src_column: "user_id".into(),
            dst_table: "users".into(),
            forward: None,
            reverse: None,
            bidirectional: true,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""src_column":"user_id""#));
        assert!(json.contains(r#""dst_table":"users""#));
        assert!(!json.contains(r#""forward""#));
        assert!(!json.contains(r#""reverse""#));
        // Default bidirectional (true) should be skipped.
        assert!(!json.contains(r#""bidirectional""#));
    }

    #[test]
    fn fkeyspec_with_directions() {
        let spec = FKeySpec {
            src_column: "user_id".into(),
            dst_table: "users".into(),
            forward: Some(DirectionSpec {
                time: Some(TimeSpec {
                    table: "users".into(),
                    column: "created_at".into(),
                    bitwidth: Some(16),
                    anchor_time: None,
                    unit: "day".into(),
                }),
                weight: None,
            }),
            reverse: Some(DirectionSpec {
                time: Some(TimeSpec {
                    table: "transactions".into(),
                    column: "txn_time".into(),
                    bitwidth: Some(16),
                    anchor_time: None,
                    unit: "day".into(),
                }),
                weight: Some(WeightSpec {
                    table: "transactions".into(),
                    expr: "amount".into(),
                    dtype: "float16".into(),
                }),
            }),
            bidirectional: true,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""forward""#));
        assert!(json.contains(r#""reverse""#));

        let parsed: FKeySpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.src_column, "user_id");
        assert!(parsed.forward.is_some());
        assert!(parsed.reverse.is_some());
    }

    #[test]
    fn fkeyspec_deserialize() {
        let json = r#"{"src_column":"item_id","dst_table":"items"}"#;
        let spec: FKeySpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.src_column, "item_id");
        assert_eq!(spec.dst_table, "items");
        assert!(spec.forward.is_none());
        assert!(spec.reverse.is_none());
    }

    // TableSpec with new fields tests
    #[test]
    fn tablespec_with_source_field() {
        let table = TableSpec {
            name: "events".into(),
            columns: vec![],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec!["s3://bucket/events/".into()],
            },

            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        };
        let json = serde_json::to_string(&table).unwrap();
        assert!(json.contains(r#""source""#));
        assert!(json.contains(r#""type":"parquet""#));
    }

    #[test]
    fn tablespec_with_fkeys() {
        let table = TableSpec {
            name: "transactions".into(),
            columns: vec![ColumnSpec {
                name: "user_id".into(),
                ingested_dtype: IngestedType::Int64,
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            }],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec!["s3://bucket/data/".into(), "/local/path/".into()],
            },

            source_size: None,
            fkeys: vec![FKeySpec {
                src_column: "user_id".into(),
                dst_table: "users".into(),
                forward: None,
                reverse: None,
                bidirectional: true,
            }],
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        };
        let json = serde_json::to_string(&table).unwrap();
        assert!(json.contains(r#""fkeys""#));
        assert!(json.contains(r#""src_column":"user_id""#));
    }

    #[test]
    fn tablespec_with_end_time() {
        let table = TableSpec {
            name: "subscriptions".into(),
            columns: vec![],
            primary_key: None,
            create_time: Some("start_date".into()),
            end_time: Some("end_date".into()),
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec!["/data/subs.parquet".into()],
            },

            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        };
        let json = serde_json::to_string(&table).unwrap();
        assert!(json.contains(r#""create_time":"start_date""#));
        assert!(json.contains(r#""end_time":"end_date""#));
    }

    // EdgeTableSpec with new fields tests
    #[test]
    fn edgetablespec_with_directions() {
        let spec = EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "user_id".into(),
                ref_table: "users".into(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "item_id".into(),
                ref_table: "items".into(),
            },
            forward: Some(DirectionSpec {
                time: Some(TimeSpec {
                    table: "purchases".into(),
                    column: "purchase_time".into(),
                    bitwidth: Some(32),
                    anchor_time: None,
                    unit: "s".into(),
                }),
                weight: Some(WeightSpec {
                    table: "purchases".into(),
                    expr: "quantity * price".into(),
                    dtype: "float32".into(),
                }),
            }),
            reverse: Some(DirectionSpec {
                time: Some(TimeSpec {
                    table: "purchases".into(),
                    column: "purchase_time".into(),
                    bitwidth: Some(32),
                    anchor_time: None,
                    unit: "s".into(),
                }),
                weight: None,
            }),
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains(r#""forward""#));
        assert!(json.contains(r#""reverse""#));
        assert!(json.contains(r#""quantity * price""#));
    }

    // Round-trip tests
    #[test]
    fn datasource_roundtrip_parquet() {
        let original = DataSource::Parquet {
            paths: vec!["path1".into(), "path2".into()],
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: DataSource = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn datasource_roundtrip_snowflake() {
        let original = DataSource::Snowflake {
            connection_id: "conn".into(),
            table_or_view: "tbl".into(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: DataSource = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn datasource_roundtrip_databricks() {
        let original = DataSource::Databricks {
            server_hostname: "adb.example.net".into(),
            warehouse_id: "wh".into(),
            catalog: "cat".into(),
            schema: "sch".into(),
            sql: "SELECT id FROM cat.sch.tbl".into(),
            auth_profile: Some("db_conn".into()),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: DataSource = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn datasource_roundtrip_bigquery() {
        let original = DataSource::BigQuery {
            connection_id: "conn".into(),
            project: "proj".into(),
            dataset: "ds".into(),
            table: "tbl".into(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: DataSource = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn timespec_roundtrip() {
        let original = TimeSpec {
            table: "events".into(),
            column: "ts".into(),
            bitwidth: Some(16),
            anchor_time: Some(1000000),
            unit: "ms".into(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: TimeSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn weightspec_roundtrip() {
        let original = WeightSpec {
            table: "edges".into(),
            expr: "w1 + w2".into(),
            dtype: "float16".into(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: WeightSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn directionspec_roundtrip() {
        let original = DirectionSpec {
            time: Some(TimeSpec {
                table: "t".into(),
                column: "c".into(),
                bitwidth: None,
                anchor_time: None,
                unit: "day".into(),
            }),
            weight: Some(WeightSpec {
                table: "t".into(),
                expr: "w".into(),
                dtype: "byte".into(),
            }),
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: DirectionSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn fkeyspec_roundtrip() {
        let original = FKeySpec {
            src_column: "fk_col".into(),
            dst_table: "dst".into(),
            forward: Some(DirectionSpec::default()),
            reverse: None,
            bidirectional: true,
        };
        let json = serde_json::to_string(&original).unwrap();
        let parsed: FKeySpec = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed);
    }
}

// TT-NG timestamp grouping gate: the `group_by_timestamp` field must be OMITTED
// from serialized JSON when false (byte-identical to pre-feature output) and
// present when true. This pins the "byte-identical when disabled" guarantee.
#[test]
fn group_by_timestamp_omitted_when_false() {
    let table = base_table("t");
    assert!(!table.group_by_timestamp, "default is false");
    let json = serde_json::to_string(&table).unwrap();
    assert!(
        !json.contains("group_by_timestamp"),
        "field must be omitted when false; got: {json}"
    );
    // Round-trips back to false.
    let back: TableSpec = serde_json::from_str(&json).unwrap();
    assert!(!back.group_by_timestamp);
}

#[test]
fn group_by_timestamp_present_when_true() {
    let mut table = base_table("t");
    table.group_by_timestamp = true;
    let json = serde_json::to_string(&table).unwrap();
    assert!(json.contains("\"group_by_timestamp\":true"));
    let back: TableSpec = serde_json::from_str(&json).unwrap();
    assert!(back.group_by_timestamp);
}

#[test]
fn databricks_auth_profile_is_persisted() {
    let mut config = GraphConfig {
        dataset_name: "dataset".into(),
        tables: vec![base_table("customers")],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    };
    config.tables[0].source = DataSource::Databricks {
        server_hostname: "adb.example.net".into(),
        warehouse_id: "wh".into(),
        catalog: "main".into(),
        schema: "default".into(),
        sql: "SELECT id FROM main.default.customers".into(),
        auth_profile: Some("production".into()),
    };

    let persisted = serde_json::to_string(&config).unwrap();
    assert!(persisted.contains(r#""auth_profile":"production""#));
    assert!(persisted.contains("SELECT id FROM main.default.customers"));
}

#[test]
fn databricks_auth_profile_cannot_be_empty() {
    let mut config = GraphConfig {
        dataset_name: "dataset".into(),
        tables: vec![base_table("customers")],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    };
    config.tables[0].source = DataSource::Databricks {
        server_hostname: "adb.example.net".into(),
        warehouse_id: "wh".into(),
        catalog: "main".into(),
        schema: "default".into(),
        sql: "SELECT id FROM main.default.customers".into(),
        auth_profile: Some(String::new()),
    };

    let error = config.validate().unwrap_err().to_string();
    assert!(error.contains("auth_profile cannot be empty"));
}
