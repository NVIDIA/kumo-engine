//! Unit tests for kumo_config module.

use super::*;
use crate::graph_config::IngestedType;

// =============================================================================
// Dtype Mapping Tests
// =============================================================================

#[test]
fn test_map_dtype_bool() {
    assert_eq!(map_kumo_dtype("bool"), Some(IngestedType::Bool));
}

#[test]
fn test_map_dtype_int_variants() {
    assert_eq!(map_kumo_dtype("int"), Some(IngestedType::Int64));
    assert_eq!(map_kumo_dtype("int64"), Some(IngestedType::Int64));
    assert_eq!(map_kumo_dtype("int32"), Some(IngestedType::Int32));
    assert_eq!(map_kumo_dtype("int16"), Some(IngestedType::Int16));
}

#[test]
fn test_map_dtype_uint_byte() {
    assert_eq!(map_kumo_dtype("byte"), Some(IngestedType::UInt8));
}

#[test]
fn test_map_dtype_float_variants() {
    assert_eq!(map_kumo_dtype("float"), Some(IngestedType::Float32));
    assert_eq!(map_kumo_dtype("float32"), Some(IngestedType::Float32));
    assert_eq!(map_kumo_dtype("float64"), Some(IngestedType::Float64));
}

#[test]
fn test_map_dtype_string() {
    assert_eq!(map_kumo_dtype("string"), Some(IngestedType::String));
}

#[test]
fn test_map_dtype_temporal() {
    assert_eq!(map_kumo_dtype("date"), Some(IngestedType::Timestamp));
    assert_eq!(map_kumo_dtype("time"), Some(IngestedType::Timestamp));
    assert_eq!(map_kumo_dtype("timedelta"), Some(IngestedType::Int64));
}

#[test]
fn test_map_dtype_lists() {
    assert_eq!(
        map_kumo_dtype("floatlist"),
        Some(IngestedType::Float32Array)
    );
    assert_eq!(map_kumo_dtype("intlist"), Some(IngestedType::String));
    assert_eq!(map_kumo_dtype("stringlist"), Some(IngestedType::String));
}

#[test]
fn test_map_dtype_binary() {
    assert_eq!(map_kumo_dtype("binary"), Some(IngestedType::Bytes));
}

#[test]
fn test_map_dtype_unknown() {
    assert_eq!(map_kumo_dtype("unknown_type"), None);
    assert_eq!(map_kumo_dtype("unsupported"), None);
    assert_eq!(map_kumo_dtype(""), None);
}

// =============================================================================
// Timestamp Unit Mapping Tests
// =============================================================================

#[test]
fn test_map_timestamp_unit_seconds() {
    use crate::graph_config::TimestampUnit;
    assert_eq!(map_timestamp_unit("s"), Some(TimestampUnit::Seconds));
    assert_eq!(map_timestamp_unit("seconds"), Some(TimestampUnit::Seconds));
    assert_eq!(map_timestamp_unit("sec"), Some(TimestampUnit::Seconds));
}

#[test]
fn test_map_timestamp_unit_millis() {
    use crate::graph_config::TimestampUnit;
    assert_eq!(map_timestamp_unit("ms"), Some(TimestampUnit::Millis));
    assert_eq!(map_timestamp_unit("millis"), Some(TimestampUnit::Millis));
    assert_eq!(
        map_timestamp_unit("milliseconds"),
        Some(TimestampUnit::Millis)
    );
}

#[test]
fn test_map_timestamp_unit_micros() {
    use crate::graph_config::TimestampUnit;
    assert_eq!(map_timestamp_unit("us"), Some(TimestampUnit::Micros));
    assert_eq!(map_timestamp_unit("micros"), Some(TimestampUnit::Micros));
    assert_eq!(
        map_timestamp_unit("microseconds"),
        Some(TimestampUnit::Micros)
    );
}

#[test]
fn test_map_timestamp_unit_nanos() {
    use crate::graph_config::TimestampUnit;
    assert_eq!(map_timestamp_unit("ns"), Some(TimestampUnit::Nanos));
    assert_eq!(map_timestamp_unit("nanos"), Some(TimestampUnit::Nanos));
    assert_eq!(
        map_timestamp_unit("nanoseconds"),
        Some(TimestampUnit::Nanos)
    );
}

#[test]
fn test_map_timestamp_unit_unknown() {
    assert_eq!(map_timestamp_unit("unknown"), None);
    assert_eq!(map_timestamp_unit(""), None);
}

// =============================================================================
// Conversion Tests
// =============================================================================

fn make_kumo_config(tables: Vec<KumoTable>) -> KumoGraphConfig {
    KumoGraphConfig {
        name: Some("test_graph".to_string()),
        data_source_dict: Default::default(),
        tables,
    }
}

fn make_convert_options() -> ConvertOptions {
    ConvertOptions {
        data_persist_dir: "s3://test-bucket/customer/prod/data-persist-dir".to_string(),
        dataset_name: "test_dataset".to_string(),
        table_data_versions: None,
    }
}

#[test]
fn test_convert_simple_node_table() {
    let kumo_table = KumoTable {
        table_name: "users".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("users".to_string()),
        data_version: Some("12345".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "user_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![
            KumoColumn {
                name: "name".to_string(),
                dtype: Some("string".to_string()),
                stype: None,
                timestamp_unit: None,
            },
            KumoColumn {
                name: "age".to_string(),
                dtype: Some("int32".to_string()),
                stype: None,
                timestamp_unit: None,
            },
        ],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    assert_eq!(result.dataset_name, "test_dataset");
    assert_eq!(result.tables.len(), 1);

    let table = &result.tables[0];
    assert_eq!(table.name, "users");
    assert_eq!(table.primary_key, Some("user_id".to_string()));
    assert_eq!(table.columns.len(), 3); // user_id, name, age
    assert!(table.edge_table.is_none());
    assert!(table.fkeys.is_empty());
}

#[test]
fn test_convert_table_with_fkeys() {
    let kumo_table = KumoTable {
        table_name: "orders".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("orders".to_string()),
        data_version: Some("12345".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "order_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![KumoForeignKey {
            name: "user_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            dst_table: "postgres.users".to_string(),
            dst_key: "user_id".to_string(),
            dst_data_source: None,
        }],
        cols: vec![KumoColumn {
            name: "amount".to_string(),
            dtype: Some("float64".to_string()),
            stype: None,
            timestamp_unit: None,
        }],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    assert_eq!(table.name, "orders");
    assert!(table.edge_table.is_none()); // Has pkey + extra column, not edge table
    assert_eq!(table.fkeys.len(), 1);
    assert_eq!(table.fkeys[0].src_column, "user_id");
    assert_eq!(table.fkeys[0].dst_table, "users"); // Namespace stripped
}

#[test]
fn test_convert_edge_table() {
    // Edge table: 2 fkeys, only time column, no extra data columns
    let kumo_table = KumoTable {
        table_name: "purchases".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("purchases".to_string()),
        data_version: Some("12345".to_string()),
        pkey: None, // No primary key
        fkeys: vec![
            KumoForeignKey {
                name: "user_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "users".to_string(),
                dst_key: "id".to_string(),
                dst_data_source: None,
            },
            KumoForeignKey {
                name: "item_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "items".to_string(),
                dst_key: "id".to_string(),
                dst_data_source: None,
            },
        ],
        cols: vec![], // No extra columns
        time_col: Some(KumoTimeCol {
            name: "purchase_time".to_string(),
            dtype: Some("time".to_string()),
            stype: None,
            timestamp_unit: Some("s".to_string()),
            is_pseudo: false,
        }),
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    assert_eq!(table.name, "purchases");
    assert!(table.edge_table.is_some()); // Should be edge table
    assert!(table.fkeys.is_empty()); // fkeys go into edge_table instead

    let edge = table.edge_table.as_ref().unwrap();
    assert_eq!(edge.src_fk.edge_column, "user_id");
    assert_eq!(edge.src_fk.ref_table, "users");
    assert_eq!(edge.dst_fk.edge_column, "item_id");
    assert_eq!(edge.dst_fk.ref_table, "items");
}

#[test]
fn test_convert_not_edge_table_with_extra_columns() {
    // Has 2 fkeys but also extra data columns -> not edge table
    let kumo_table = KumoTable {
        table_name: "purchases".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("purchases".to_string()),
        data_version: Some("12345".to_string()),
        pkey: None,
        fkeys: vec![
            KumoForeignKey {
                name: "user_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "users".to_string(),
                dst_key: "id".to_string(),
                dst_data_source: None,
            },
            KumoForeignKey {
                name: "item_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "items".to_string(),
                dst_key: "id".to_string(),
                dst_data_source: None,
            },
        ],
        cols: vec![KumoColumn {
            name: "review_text".to_string(), // Non-numeric extra column
            dtype: Some("string".to_string()),
            stype: None,
            timestamp_unit: None,
        }],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    assert!(table.edge_table.is_none()); // Not edge table due to extra column
    assert_eq!(table.fkeys.len(), 2);
}

#[test]
fn test_convert_path_building() {
    let kumo_table = KumoTable {
        table_name: "users".to_string(),
        data_source: Some("my_db".to_string()),
        source_name: Some("users_table".to_string()),
        data_version: Some("v123".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    // Check source instead of paths
    let source = &table.source;
    match source {
        crate::graph_config::DataSource::Parquet { paths } => {
            assert_eq!(paths.len(), 1);
            assert_eq!(
                paths[0],
                "s3://test-bucket/customer/prod/data-persist-dir/ingest/fpv3.my_db.users_table/v123/"
            );
        }
        _ => panic!("Expected Parquet source"),
    }
}

#[test]
fn test_convert_dst_table_namespace_strip() {
    let kumo_table = KumoTable {
        table_name: "orders".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("orders".to_string()),
        data_version: Some("12345".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "order_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![KumoForeignKey {
            name: "user_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            dst_table: "schema.datasource.users".to_string(), // Multiple dots
            dst_key: "user_id".to_string(),
            dst_data_source: None,
        }],
        cols: vec![],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    assert_eq!(table.fkeys[0].dst_table, "users"); // Only last part kept
}

#[test]
fn test_convert_time_col_to_create_time() {
    let kumo_table = KumoTable {
        table_name: "events".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("events".to_string()),
        data_version: Some("12345".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "event_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![],
        time_col: Some(KumoTimeCol {
            name: "event_time".to_string(),
            dtype: Some("time".to_string()),
            stype: None,
            timestamp_unit: Some("ms".to_string()),
            is_pseudo: false,
        }),
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    assert_eq!(table.create_time, Some("event_time".to_string()));

    // Check that the time column has the correct timestamp unit
    let time_col = table
        .columns
        .iter()
        .find(|c| c.name == "event_time")
        .unwrap();
    assert_eq!(
        time_col.timestamp_unit,
        Some(crate::graph_config::TimestampUnit::Millis)
    );
}

#[test]
fn test_convert_missing_data_source_error() {
    let kumo_table = KumoTable {
        table_name: "users".to_string(),
        data_source: None, // Missing!
        source_name: None,
        data_version: Some("12345".to_string()),
        pkey: None,
        fkeys: vec![],
        cols: vec![KumoColumn {
            name: "id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            timestamp_unit: None,
        }],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options);
    assert!(result.is_err());
}

#[test]
fn test_convert_missing_data_version_error() {
    let kumo_table = KumoTable {
        table_name: "users".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: None,
        data_version: None, // Missing!
        pkey: None,
        fkeys: vec![],
        cols: vec![KumoColumn {
            name: "id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            timestamp_unit: None,
        }],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options);
    assert!(result.is_err());
}

// =============================================================================
// Additional Comprehensive Tests
// =============================================================================

#[test]
fn test_convert_edge_table_with_weight_column() {
    // Edge table: 2 fkeys + time_col + numeric weight column
    let kumo_table = KumoTable {
        table_name: "purchases".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("purchases".to_string()),
        data_version: Some("12345".to_string()),
        pkey: None,
        fkeys: vec![
            KumoForeignKey {
                name: "user_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "users".to_string(),
                dst_key: "id".to_string(),
                dst_data_source: None,
            },
            KumoForeignKey {
                name: "item_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "items".to_string(),
                dst_key: "id".to_string(),
                dst_data_source: None,
            },
        ],
        cols: vec![KumoColumn {
            name: "price".to_string(), // Numeric weight-like column
            dtype: Some("float64".to_string()),
            stype: None,
            timestamp_unit: None,
        }],
        time_col: Some(KumoTimeCol {
            name: "purchase_time".to_string(),
            dtype: Some("time".to_string()),
            stype: None,
            timestamp_unit: Some("s".to_string()),
            is_pseudo: false,
        }),
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    assert!(table.edge_table.is_some()); // Should still be edge table with numeric weight
    assert!(table.fkeys.is_empty());
}

#[test]
fn test_convert_source_name_defaults_to_table_name() {
    let kumo_table = KumoTable {
        table_name: "my_table".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: None, // Not specified - should default to table_name
        data_version: Some("12345".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    // Source path should use table_name as source_name
    let source = &table.source;
    match source {
        crate::graph_config::DataSource::Parquet { paths } => {
            assert!(paths[0].contains("fpv3.postgres.my_table/"));
        }
        _ => panic!("Expected Parquet source"),
    }
}

#[test]
fn test_convert_end_time_col() {
    let kumo_table = KumoTable {
        table_name: "subscriptions".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("subscriptions".to_string()),
        data_version: Some("12345".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "subscription_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![],
        time_col: Some(KumoTimeCol {
            name: "start_date".to_string(),
            dtype: Some("time".to_string()),
            stype: None,
            timestamp_unit: Some("s".to_string()),
            is_pseudo: false,
        }),
        end_time_col: Some(KumoTimeCol {
            name: "end_date".to_string(),
            dtype: Some("time".to_string()),
            stype: None,
            timestamp_unit: Some("s".to_string()),
            is_pseudo: false,
        }),
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    assert_eq!(table.create_time, Some("start_date".to_string()));
    assert_eq!(table.end_time, Some("end_date".to_string()));

    // Both time columns should be in the columns list
    let col_names: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
    assert!(col_names.contains(&"start_date"));
    assert!(col_names.contains(&"end_date"));
}

#[test]
fn test_convert_column_deduplication() {
    // Same column name in pkey and cols - should only appear once
    let kumo_table = KumoTable {
        table_name: "users".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("users".to_string()),
        data_version: Some("12345".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "user_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![
            KumoColumn {
                name: "user_id".to_string(), // Duplicate of pkey
                dtype: Some("int64".to_string()),
                stype: None,
                timestamp_unit: None,
            },
            KumoColumn {
                name: "name".to_string(),
                dtype: Some("string".to_string()),
                stype: None,
                timestamp_unit: None,
            },
        ],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    // Should have exactly 2 columns (user_id, name) not 3
    assert_eq!(table.columns.len(), 2);

    let user_id_count = table.columns.iter().filter(|c| c.name == "user_id").count();
    assert_eq!(user_id_count, 1);
}

#[test]
fn test_convert_single_fkey_not_edge_table() {
    // Table with only 1 fkey should not be edge table
    let kumo_table = KumoTable {
        table_name: "orders".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("orders".to_string()),
        data_version: Some("12345".to_string()),
        pkey: None,
        fkeys: vec![KumoForeignKey {
            name: "user_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            dst_table: "users".to_string(),
            dst_key: "id".to_string(),
            dst_data_source: None,
        }],
        cols: vec![],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    assert!(table.edge_table.is_none()); // Not edge table with only 1 fkey
    assert_eq!(table.fkeys.len(), 1);
}

#[test]
fn test_convert_three_fkeys_not_edge_table() {
    // Table with 3+ fkeys should not be edge table
    let kumo_table = KumoTable {
        table_name: "transactions".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("transactions".to_string()),
        data_version: Some("12345".to_string()),
        pkey: None,
        fkeys: vec![
            KumoForeignKey {
                name: "user_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "users".to_string(),
                dst_key: "id".to_string(),
                dst_data_source: None,
            },
            KumoForeignKey {
                name: "item_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "items".to_string(),
                dst_key: "id".to_string(),
                dst_data_source: None,
            },
            KumoForeignKey {
                name: "store_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "stores".to_string(),
                dst_key: "id".to_string(),
                dst_data_source: None,
            },
        ],
        cols: vec![],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    let table = &result.tables[0];
    assert!(table.edge_table.is_none()); // Not edge table with 3 fkeys
    assert_eq!(table.fkeys.len(), 3);
}

#[test]
fn test_convert_no_columns_error() {
    // Table with no columns at all should error
    let kumo_table = KumoTable {
        table_name: "empty".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("empty".to_string()),
        data_version: Some("12345".to_string()),
        pkey: None,
        fkeys: vec![],
        cols: vec![],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ConvertError::NoColumns { .. }
    ));
}

#[test]
fn test_convert_multi_table_graph() {
    // Full graph with nodes and edges
    let users_table = KumoTable {
        table_name: "users".to_string(),
        data_source: Some("db".to_string()),
        source_name: Some("users".to_string()),
        data_version: Some("v1".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "user_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![KumoColumn {
            name: "name".to_string(),
            dtype: Some("string".to_string()),
            stype: None,
            timestamp_unit: None,
        }],
        time_col: None,
        end_time_col: None,
    };

    let items_table = KumoTable {
        table_name: "items".to_string(),
        data_source: Some("db".to_string()),
        source_name: Some("items".to_string()),
        data_version: Some("v1".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "item_id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![KumoColumn {
            name: "price".to_string(),
            dtype: Some("float64".to_string()),
            stype: None,
            timestamp_unit: None,
        }],
        time_col: None,
        end_time_col: None,
    };

    let purchases_table = KumoTable {
        table_name: "purchases".to_string(),
        data_source: Some("db".to_string()),
        source_name: Some("purchases".to_string()),
        data_version: Some("v1".to_string()),
        pkey: None,
        fkeys: vec![
            KumoForeignKey {
                name: "user_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "db.users".to_string(),
                dst_key: "user_id".to_string(),
                dst_data_source: None,
            },
            KumoForeignKey {
                name: "item_id".to_string(),
                dtype: Some("int64".to_string()),
                stype: None,
                dst_table: "db.items".to_string(),
                dst_key: "item_id".to_string(),
                dst_data_source: None,
            },
        ],
        cols: vec![],
        time_col: Some(KumoTimeCol {
            name: "timestamp".to_string(),
            dtype: Some("time".to_string()),
            stype: None,
            timestamp_unit: Some("ms".to_string()),
            is_pseudo: false,
        }),
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![users_table, items_table, purchases_table]);
    let options = make_convert_options();

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();

    assert_eq!(result.tables.len(), 3);

    // Users table - node table
    let users = result.tables.iter().find(|t| t.name == "users").unwrap();
    assert_eq!(users.primary_key, Some("user_id".to_string()));
    assert!(users.edge_table.is_none());
    assert!(users.fkeys.is_empty());

    // Items table - node table
    let items = result.tables.iter().find(|t| t.name == "items").unwrap();
    assert_eq!(items.primary_key, Some("item_id".to_string()));
    assert!(items.edge_table.is_none());
    assert!(items.fkeys.is_empty());

    // Purchases table - edge table
    let purchases = result
        .tables
        .iter()
        .find(|t| t.name == "purchases")
        .unwrap();
    assert!(purchases.primary_key.is_none());
    assert!(purchases.edge_table.is_some());
    assert!(purchases.fkeys.is_empty());

    let edge = purchases.edge_table.as_ref().unwrap();
    assert_eq!(edge.src_fk.ref_table, "users");
    assert_eq!(edge.dst_fk.ref_table, "items");
}

#[test]
fn test_is_numeric_weight_dtype() {
    use crate::graph_config::IngestedType;
    use crate::utils::is_numeric_weight_dtype;

    // Numeric types - all signed integers
    assert!(is_numeric_weight_dtype(&IngestedType::Int8));
    assert!(is_numeric_weight_dtype(&IngestedType::Int16));
    assert!(is_numeric_weight_dtype(&IngestedType::Int32));
    assert!(is_numeric_weight_dtype(&IngestedType::Int64));

    // Numeric types - all unsigned integers
    assert!(is_numeric_weight_dtype(&IngestedType::UInt8));
    assert!(is_numeric_weight_dtype(&IngestedType::UInt16));
    assert!(is_numeric_weight_dtype(&IngestedType::UInt32));
    assert!(is_numeric_weight_dtype(&IngestedType::UInt64));

    // Numeric types - floats
    assert!(is_numeric_weight_dtype(&IngestedType::Float32));
    assert!(is_numeric_weight_dtype(&IngestedType::Float64));

    // Numeric types - decimals
    assert!(is_numeric_weight_dtype(&IngestedType::Decimal128));
    assert!(is_numeric_weight_dtype(&IngestedType::Decimal256));

    // Non-numeric types
    assert!(!is_numeric_weight_dtype(&IngestedType::String));
    assert!(!is_numeric_weight_dtype(&IngestedType::Bool));
    assert!(!is_numeric_weight_dtype(&IngestedType::Date));
    assert!(!is_numeric_weight_dtype(&IngestedType::Timestamp));
    assert!(!is_numeric_weight_dtype(&IngestedType::Bytes));
}

// =============================================================================
// JSON Deserialization Tests
// =============================================================================

#[test]
fn test_deserialize_kumo_config_from_json() {
    // Test that we can deserialize a real Kumo config JSON
    let json = r#"{
        "name": "test_graph",
        "data_source_dict": {},
        "tables": [
            {
                "table_name": "users",
                "data_source": "postgres",
                "source_name": "users",
                "data_version": "12345",
                "pkey": {
                    "name": "user_id",
                    "dtype": "int64",
                    "stype": "ID",
                    "drop_duplicates": false
                },
                "fkeys": [],
                "cols": [
                    {
                        "name": "email",
                        "dtype": "string",
                        "stype": "text"
                    }
                ],
                "time_col": null,
                "end_time_col": null
            }
        ]
    }"#;

    let config: KumoGraphConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.name, Some("test_graph".to_string()));
    assert_eq!(config.tables.len(), 1);
    assert_eq!(config.tables[0].table_name, "users");
    assert!(config.tables[0].pkey.is_some());
}

#[test]
fn test_deserialize_with_optional_fields() {
    // Test that optional fields are handled correctly
    let json = r#"{
        "name": null,
        "tables": [
            {
                "table_name": "minimal",
                "data_source": "db",
                "data_version": "1",
                "pkey": null,
                "fkeys": [],
                "cols": [{"name": "id", "dtype": "int64"}],
                "time_col": null,
                "end_time_col": null
            }
        ]
    }"#;

    let config: KumoGraphConfig = serde_json::from_str(json).unwrap();
    assert!(config.name.is_none());
    assert!(config.tables[0].pkey.is_none());
    assert!(config.tables[0].source_name.is_none());
}

// =============================================================================
// table_data_versions Tests
// =============================================================================

#[test]
fn test_table_data_versions_overrides_data_version() {
    let kumo_table = KumoTable {
        table_name: "users".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("users".to_string()),
        data_version: Some("old_version".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let mut versions = std::collections::HashMap::new();
    versions.insert("users".to_string(), "new_version_123".to_string());

    let options = ConvertOptions {
        data_persist_dir: "s3://test-bucket/customer/prod/data-persist-dir".to_string(),
        dataset_name: "test_dataset".to_string(),
        table_data_versions: Some(versions),
    };

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();
    let table = &result.tables[0];
    match &table.source {
        crate::graph_config::DataSource::Parquet { paths } => {
            assert!(
                paths[0].contains("new_version_123"),
                "Expected path to contain new_version_123, got: {}",
                paths[0]
            );
            assert!(
                !paths[0].contains("old_version"),
                "Expected path to NOT contain old_version, got: {}",
                paths[0]
            );
        }
        _ => panic!("Expected Parquet source"),
    }
}

#[test]
fn test_table_data_versions_missing_entry_errors() {
    let kumo_table = KumoTable {
        table_name: "users".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("users".to_string()),
        data_version: None,
        pkey: Some(KumoPrimaryKey {
            name: "id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    // Provide a map but without the "users" entry.
    let versions = std::collections::HashMap::new();
    let options = ConvertOptions {
        data_persist_dir: "s3://test-bucket/data".to_string(),
        dataset_name: "test".to_string(),
        table_data_versions: Some(versions),
    };

    let result = convert_kumo_to_diskgraph(&kumo_config, &options);
    assert!(result.is_err());
}

#[test]
fn test_table_data_versions_none_uses_inline() {
    let kumo_table = KumoTable {
        table_name: "users".to_string(),
        data_source: Some("postgres".to_string()),
        source_name: Some("users".to_string()),
        data_version: Some("inline_v42".to_string()),
        pkey: Some(KumoPrimaryKey {
            name: "id".to_string(),
            dtype: Some("int64".to_string()),
            stype: None,
            drop_duplicates: false,
        }),
        fkeys: vec![],
        cols: vec![],
        time_col: None,
        end_time_col: None,
    };

    let kumo_config = make_kumo_config(vec![kumo_table]);
    let options = ConvertOptions {
        data_persist_dir: "s3://test-bucket/data".to_string(),
        dataset_name: "test".to_string(),
        table_data_versions: None, // Use inline data_version
    };

    let result = convert_kumo_to_diskgraph(&kumo_config, &options).unwrap();
    match &result.tables[0].source {
        crate::graph_config::DataSource::Parquet { paths } => {
            assert!(paths[0].contains("inline_v42"));
        }
        _ => panic!("Expected Parquet source"),
    }
}

// =============================================================================
// detect_timestamp_formats Tests
// =============================================================================

#[test]
fn test_detect_timestamp_formats_from_parquet() {
    use crate::config_converter::detect_timestamp_formats;
    use crate::graph_config::{ColumnSpec, DataSource, GraphConfig, IngestedType, TableSpec};
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    // Create a temp parquet file with a string column containing timestamps.
    let dir = tempfile::tempdir().unwrap();
    let parquet_path = dir.path().join("test.parquet");

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("created_at", DataType::Utf8, true),
    ]));

    let id_array = StringArray::from(vec!["a", "b", "c"]);
    let ts_array = StringArray::from(vec![
        Some("2024-01-15T10:30:00"),
        Some("2024-02-20T14:45:30"),
        None,
    ]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id_array), Arc::new(ts_array)],
    )
    .unwrap();

    let file = std::fs::File::create(&parquet_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Build a GraphConfig with a Timestamp column pointing to the parquet file.
    let mut config = GraphConfig {
        dataset_name: "test".to_string(),
        tables: vec![TableSpec {
            name: "events".to_string(),
            columns: vec![
                ColumnSpec {
                    name: "id".to_string(),
                    source_dtype: None,
                    ingested_dtype: IngestedType::String,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    precision: None,
                    scale: None,
                    semantic_type: None,
                },
                ColumnSpec {
                    name: "created_at".to_string(),
                    source_dtype: None,
                    ingested_dtype: IngestedType::Timestamp,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    precision: None,
                    scale: None,
                    semantic_type: None,
                },
            ],
            primary_key: Some("id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec![parquet_path.to_string_lossy().to_string()],
            },
            fkeys: vec![],
        }],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    };

    // Run detection (no cloud transfer needed for local files).
    detect_timestamp_formats(&mut config, &None).unwrap();

    // The created_at column should now have a detected timestamp format.
    let col = &config.tables[0].columns[1];
    assert_eq!(col.name, "created_at");
    assert!(
        col.timestamp_format.is_some(),
        "Expected a detected timestamp format, got None"
    );
    // The parser may match "%Y-%m-%d %H:%M:%S" (space) or "%Y-%m-%dT%H:%M:%S" (T)
    // depending on format priority — both are valid for ISO 8601 with T separator.
    let fmt = col.timestamp_format.as_ref().unwrap();
    assert!(
        fmt.contains("%Y-%m-%d") && fmt.contains("%H:%M:%S"),
        "Expected date+time format, got {:?}",
        fmt
    );

    // The id column should remain unchanged.
    assert_eq!(config.tables[0].columns[0].timestamp_format, None);
}

#[test]
fn test_detect_timestamp_formats_skips_native_timestamp_columns() {
    use crate::config_converter::detect_timestamp_formats;
    use crate::graph_config::{ColumnSpec, DataSource, GraphConfig, IngestedType, TableSpec};
    use arrow_array::{RecordBatch, TimestampMicrosecondArray};
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    // Create a parquet file with a native Arrow timestamp column (not string).
    let dir = tempfile::tempdir().unwrap();
    let parquet_path = dir.path().join("native_ts.parquet");

    let schema = Arc::new(Schema::new(vec![Field::new(
        "event_time",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        true,
    )]));

    let ts_array = TimestampMicrosecondArray::from(vec![Some(1700000000_000_000i64), Some(1700100000_000_000)]);
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(ts_array)]).unwrap();

    let file = std::fs::File::create(&parquet_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let mut config = GraphConfig {
        dataset_name: "test".to_string(),
        tables: vec![TableSpec {
            name: "events".to_string(),
            columns: vec![ColumnSpec {
                name: "event_time".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::Timestamp,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
            }],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec![parquet_path.to_string_lossy().to_string()],
            },
            fkeys: vec![],
        }],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    };

    detect_timestamp_formats(&mut config, &None).unwrap();

    // Native timestamp column should NOT get a format string (it's not Utf8).
    assert_eq!(config.tables[0].columns[0].timestamp_format, None);
}

#[test]
fn test_detect_timestamp_formats_integer_yyyymmdd() {
    use crate::config_converter::detect_timestamp_formats;
    use crate::graph_config::{ColumnSpec, DataSource, GraphConfig, IngestedType, TableSpec};
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    // Create a parquet file with Int64 column containing YYYYMMDD values.
    let dir = tempfile::tempdir().unwrap();
    let parquet_path = dir.path().join("yyyymmdd.parquet");

    let schema = Arc::new(Schema::new(vec![Field::new(
        "date_col",
        DataType::Int64,
        false,
    )]));

    let int_array = Int64Array::from(vec![20230408, 20240115, 20250301]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(int_array)]).unwrap();

    let file = std::fs::File::create(&parquet_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let mut config = GraphConfig {
        dataset_name: "test".to_string(),
        tables: vec![TableSpec {
            name: "events".to_string(),
            columns: vec![ColumnSpec {
                name: "date_col".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::Timestamp,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
            }],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec![parquet_path.to_string_lossy().to_string()],
            },
            fkeys: vec![],
        }],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    };

    detect_timestamp_formats(&mut config, &None).unwrap();

    // Should detect %Y%m%d format, NOT epoch seconds unit
    let col = &config.tables[0].columns[0];
    assert_eq!(
        col.timestamp_format.as_deref(),
        Some("%Y%m%d"),
        "Expected %Y%m%d format, got format={:?}, unit={:?}",
        col.timestamp_format,
        col.timestamp_unit
    );
    assert_eq!(col.timestamp_unit, None);
}

#[test]
fn test_detect_timestamp_formats_multi_value_sampling() {
    use crate::config_converter::detect_timestamp_formats;
    use crate::graph_config::{ColumnSpec, DataSource, GraphConfig, IngestedType, TableSpec};
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    // Create a parquet file with 20 values: 18 YYYYMMDD + 2 outliers (small epoch-like).
    let dir = tempfile::tempdir().unwrap();
    let parquet_path = dir.path().join("multi_sample.parquet");

    let schema = Arc::new(Schema::new(vec![Field::new(
        "date_col",
        DataType::Int64,
        false,
    )]));

    let mut values: Vec<i64> = (0..18)
        .map(|i| 20230101 + i * 100) // 20230101, 20230201, 20230301, ...
        .collect();
    // Add 2 outlier values that look like small epoch seconds
    values.push(86400);  // 1 day in seconds
    values.push(172800); // 2 days in seconds
    let int_array = Int64Array::from(values);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(int_array)]).unwrap();

    let file = std::fs::File::create(&parquet_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let mut config = GraphConfig {
        dataset_name: "test".to_string(),
        tables: vec![TableSpec {
            name: "events".to_string(),
            columns: vec![ColumnSpec {
                name: "date_col".to_string(),
                source_dtype: None,
                ingested_dtype: IngestedType::Timestamp,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
            }],
            primary_key: None,
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec![parquet_path.to_string_lossy().to_string()],
            },
            fkeys: vec![],
        }],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    };

    detect_timestamp_formats(&mut config, &None).unwrap();

    // Majority vote should pick %Y%m%d (8 out of 10 sampled values)
    let col = &config.tables[0].columns[0];
    assert_eq!(
        col.timestamp_format.as_deref(),
        Some("%Y%m%d"),
        "Expected majority vote to pick %Y%m%d, got format={:?}, unit={:?}",
        col.timestamp_format,
        col.timestamp_unit
    );
}

#[test]
fn test_detect_timestamp_formats_skips_empty_parquet_files() {
    use crate::config_converter::detect_timestamp_formats;
    use crate::graph_config::{
        ColumnSpec, DataSource, GraphConfig, IngestedType, TableSpec, TimestampUnit,
    };
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    // Create a directory with an empty parquet file first, then a non-empty one.
    // This simulates PySpark writing empty partition files (e.g., part-00000 with 0 rows).
    let dir = tempfile::tempdir().unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("t_dat", DataType::Int64, false),
    ]));

    // File 1: empty (schema only, 0 rows) — mimics empty Spark partition
    let empty_path = dir.path().join("part-00000.parquet");
    let file = std::fs::File::create(&empty_path).unwrap();
    let writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
    writer.close().unwrap(); // Close without writing any batches → 0 rows

    // File 2: non-empty with nanosecond epoch timestamps
    let data_path = dir.path().join("part-00001.parquet");
    let id_array = StringArray::from(vec!["a", "b", "c"]);
    let ts_array = Int64Array::from(vec![
        1537401600_000_000_000i64, // 2018-09-20 in nanos
        1538006400_000_000_000i64, // 2018-09-27 in nanos
        1600732800_000_000_000i64, // 2020-09-22 in nanos
    ]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id_array), Arc::new(ts_array)],
    )
    .unwrap();
    let file = std::fs::File::create(&data_path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    // Point the config at the directory (not individual files).
    let mut config = GraphConfig {
        dataset_name: "test".to_string(),
        tables: vec![TableSpec {
            name: "transactions".to_string(),
            columns: vec![
                ColumnSpec {
                    name: "id".to_string(),
                    source_dtype: None,
                    ingested_dtype: IngestedType::String,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    precision: None,
                    scale: None,
                    semantic_type: None,
                },
                ColumnSpec {
                    name: "t_dat".to_string(),
                    source_dtype: None,
                    ingested_dtype: IngestedType::Timestamp,
                    fixed_len: None,
                    timestamp_format: None,
                    timestamp_unit: None,
                    precision: None,
                    scale: None,
                    semantic_type: None,
                },
            ],
            primary_key: Some("id".to_string()),
            create_time: Some("t_dat".to_string()),
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec![dir.path().to_string_lossy().to_string()],
            },
            fkeys: vec![],
        }],
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    };

    detect_timestamp_formats(&mut config, &None).unwrap();

    // Detection should have skipped the empty file and read from part-00001,
    // correctly detecting nanosecond epoch timestamps.
    let col = &config.tables[0].columns[1];
    assert_eq!(col.name, "t_dat");
    assert_eq!(
        col.timestamp_unit,
        Some(TimestampUnit::Nanos),
        "Expected Nanos from magnitude detection after skipping empty file, \
         got format={:?}, unit={:?}",
        col.timestamp_format,
        col.timestamp_unit
    );
}
