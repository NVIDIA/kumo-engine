use super::{map_kumo_dtype, unify_types, IngestedType};

#[test]
fn test_map_kumo_dtype() {
    assert_eq!(map_kumo_dtype("int"), Ok(IngestedType::Int64));
    assert_eq!(map_kumo_dtype("float"), Ok(IngestedType::Float32));
    assert_eq!(map_kumo_dtype("string"), Ok(IngestedType::String));
    assert_eq!(map_kumo_dtype("timedelta"), Ok(IngestedType::Int64));
    assert!(map_kumo_dtype("unknown").is_err());
}

#[test]
fn test_unify_types() {
    use IngestedType::*;
    // Equal
    assert_eq!(unify_types(&Int64, &Int64), Some(Int64));

    // String vs Uuid/Ip
    assert_eq!(unify_types(&String, &Uuid), Some(String));
    assert_eq!(unify_types(&Uuid, &String), Some(String));
    assert_eq!(unify_types(&String, &Ip), Some(String));

    // Bytes vs Uuid/Ip
    assert_eq!(unify_types(&Bytes, &Uuid), Some(Bytes));

    // Int
    assert_eq!(unify_types(&Int64, &Int32), Some(Int64));
    assert_eq!(unify_types(&Int32, &Int64), Some(Int64));

    // Mismatch
    assert_eq!(unify_types(&Int64, &String), None);
}

use crate::*;
use arrow_schema::{DataType, Field, Schema};
use diskgraph_ingest::graph_config::{ColumnSpec, DataSource, TableSpec};
use std::sync::Arc;

fn table(name: &str, cols: &[(&str, IngestedType)]) -> TableSpec {
    TableSpec {
        name: name.to_string(),
        columns: cols
            .iter()
            .map(|(n, t)| ColumnSpec {
                name: n.to_string(),
                ingested_dtype: t.clone(),
                source_dtype: None,
                fixed_len: None,
                timestamp_format: None,
                timestamp_unit: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            })
            .collect(),
        primary_key: None,
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet { paths: vec![] },
        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    }
}

fn fkey_exists(fkeys: &[InferredFKey], src_table: &str, src_col: &str, dst_table: &str) -> bool {
    fkeys.iter().any(|fk| {
        fk.src_table == src_table && fk.src_column == src_col && fk.dst_table == dst_table
    })
}

// Backwards-compatible wrapper for tests that pass dst_col (which is now always the PK)
fn rel_exists(
    fkeys: &[InferredFKey],
    src_table: &str,
    src_col: &str,
    dst_table: &str,
    _dst_col: &str, // Ignored - now always the primary key of dst_table
) -> bool {
    fkey_exists(fkeys, src_table, src_col, dst_table)
}

#[test]
fn infer_pk_id() {
    let mut t = table("users", &[("id", IngestedType::Int64)]);
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key.as_deref(), Some("id"));
}

#[test]
fn infer_pk_table_id() {
    let mut t = table("orders", &[("order_id", IngestedType::Int64)]);
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key.as_deref(), Some("order_id"));
}

#[test]
fn infer_pk_last_token_only_when_single_id() {
    let mut t = table("kaggle_hm_articles", &[("article_id", IngestedType::Int64)]);
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key.as_deref(), Some("article_id"));
}

#[test]
fn infer_pk_uuid() {
    let mut t = table("sessions", &[("uuid", IngestedType::String)]);
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key.as_deref(), Some("uuid"));
}

#[test]
fn infer_pk_prefers_customer_id_over_postal_code() {
    let mut t = TableSpec {
        name: "kaggle_hm_customers".to_string(),
        columns: vec![
            ColumnSpec {
                name: "customer_id".to_string(),
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
                name: "postal_code".to_string(),
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
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet {
            paths: vec!["/tmp/customers.parquet".to_string()],
        },

        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    };
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key.as_deref(), Some("customer_id"));
}

#[test]
fn infer_pk_ignores_user_id_in_join_table() {
    let mut t = table(
        "user_interest",
        &[
            ("user_id", IngestedType::Int64),
            ("interest_id", IngestedType::Int64),
        ],
    );
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key, None);
}

#[test]
fn rel_avito_infers_ads_and_users_primary_keys() {
    let mut tables = vec![
        table(
            "AdsInfo",
            &[
                ("AdID", IngestedType::Int64),
                ("LocationID", IngestedType::Int64),
                ("CategoryID", IngestedType::Int64),
            ],
        ),
        table(
            "UserInfo",
            &[
                ("UserID", IngestedType::Int64),
                ("UserAgentID", IngestedType::Int64),
                ("UserDeviceID", IngestedType::Int64),
            ],
        ),
    ];
    infer_primary_keys(&mut tables, None);
    let ads = tables.iter().find(|t| t.name == "AdsInfo").unwrap();
    let users = tables.iter().find(|t| t.name == "UserInfo").unwrap();
    assert_eq!(ads.primary_key.as_deref(), Some("AdID"));
    assert_eq!(users.primary_key.as_deref(), Some("UserID"));
}

#[test]
fn rel_event_infers_join_keys_and_foreign_keys() {
    let mut tables = vec![
        table(
            "events",
            &[
                ("event_id", IngestedType::Int64),
                ("user_id", IngestedType::Int64),
                ("start_time", IngestedType::Timestamp),
            ],
        ),
        table(
            "users",
            &[
                ("user_id", IngestedType::Int64),
                ("joinedAt", IngestedType::Timestamp),
            ],
        ),
        table(
            "event_attendees",
            &[
                ("event", IngestedType::Int64),
                ("user_id", IngestedType::Int64),
                ("start_time", IngestedType::Timestamp),
            ],
        ),
        table(
            "event_interest",
            &[
                ("user", IngestedType::Int64),
                ("event", IngestedType::Int64),
                ("timestamp", IngestedType::Timestamp),
            ],
        ),
    ];
    infer_primary_keys(&mut tables, None);
    infer_create_time(&mut tables);
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    drop_join_table_primary_keys(&mut tables, &inferred_fkeys);
    infer_primary_key_relationships(&tables, &mut inferred_fkeys);

    let attendees = tables.iter().find(|t| t.name == "event_attendees").unwrap();
    assert_eq!(attendees.primary_key, None);
    assert!(rel_exists(
        &inferred_fkeys,
        "event_attendees",
        "event",
        "events",
        "event_id"
    ));
    assert!(rel_exists(
        &inferred_fkeys,
        "event_attendees",
        "user_id",
        "users",
        "user_id"
    ));
    assert!(rel_exists(
        &inferred_fkeys,
        "event_interest",
        "event",
        "events",
        "event_id"
    ));
    assert!(rel_exists(
        &inferred_fkeys,
        "event_interest",
        "user",
        "users",
        "user_id"
    ));
}

#[test]
fn rel_salt_prefers_customer_id_and_address_fk() {
    let mut tables = vec![
        table("address", &[("ADDRESSID", IngestedType::Int64)]),
        table(
            "customer",
            &[
                ("CUSTOMER", IngestedType::Int64),
                ("ADDRESSID", IngestedType::Int64),
            ],
        ),
    ];
    infer_primary_keys(&mut tables, None);
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);

    let customer = tables.iter().find(|t| t.name == "customer").unwrap();
    assert_eq!(customer.primary_key.as_deref(), Some("CUSTOMER"));
    assert!(rel_exists(
        &inferred_fkeys,
        "customer",
        "ADDRESSID",
        "address",
        "ADDRESSID"
    ));
}

#[test]
fn rel_stack_adds_related_post_fk_and_skips_history_type_fk() {
    let mut tables = vec![
        table("posts", &[("Id", IngestedType::Int64)]),
        table("users", &[("Id", IngestedType::Int64)]),
        table(
            "postLinks",
            &[
                ("Id", IngestedType::Int64),
                ("PostId", IngestedType::Int64),
                ("RelatedPostId", IngestedType::Int64),
                ("CreationDate", IngestedType::Timestamp),
            ],
        ),
        table(
            "postHistory",
            &[
                ("Id", IngestedType::Int64),
                ("PostId", IngestedType::Int64),
                ("UserId", IngestedType::Int64),
                ("PostHistoryTypeId", IngestedType::Int64),
                ("CreationDate", IngestedType::Timestamp),
            ],
        ),
    ];
    infer_primary_keys(&mut tables, None);
    infer_create_time(&mut tables);
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    drop_join_table_primary_keys(&mut tables, &inferred_fkeys);
    infer_primary_key_relationships(&tables, &mut inferred_fkeys);

    assert!(rel_exists(
        &inferred_fkeys,
        "postLinks",
        "RelatedPostId",
        "posts",
        "Id"
    ));
    assert!(!rel_exists(
        &inferred_fkeys,
        "postHistory",
        "PostHistoryTypeId",
        "postHistory",
        "Id"
    ));
}

#[test]
fn rel_trial_handles_join_tables_and_fk_direction() {
    let mut tables = vec![
        table("conditions", &[("condition_id", IngestedType::Int64)]),
        table("studies", &[("nct_id", IngestedType::String)]),
        table(
            "designs",
            &[
                ("id", IngestedType::Int64),
                ("nct_id", IngestedType::String),
            ],
        ),
        table(
            "conditions_studies",
            &[
                ("id", IngestedType::Int64),
                ("nct_id", IngestedType::String),
                ("condition_id", IngestedType::Int64),
                ("date", IngestedType::Date),
            ],
        ),
    ];
    infer_primary_keys(&mut tables, None);
    infer_create_time(&mut tables);
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    drop_join_table_primary_keys(&mut tables, &inferred_fkeys);
    infer_primary_key_relationships(&tables, &mut inferred_fkeys);

    let join = tables
        .iter()
        .find(|t| t.name == "conditions_studies")
        .unwrap();
    assert_eq!(join.primary_key, None);
    assert!(rel_exists(
        &inferred_fkeys,
        "conditions_studies",
        "condition_id",
        "conditions",
        "condition_id"
    ));
    assert!(rel_exists(
        &inferred_fkeys,
        "conditions_studies",
        "nct_id",
        "studies",
        "nct_id"
    ));
    assert!(!rel_exists(
        &inferred_fkeys,
        "conditions_studies",
        "id",
        "designs",
        "id"
    ));
}

#[test]
fn infer_edge_table_with_timestamp_and_weight() {
    let mut tables = vec![
        table("users", &[("user_id", IngestedType::Int64)]),
        table("items", &[("item_id", IngestedType::Int64)]),
        table(
            "user_item_edges",
            &[
                ("user_id", IngestedType::Int64),
                ("item_id", IngestedType::Int64),
                ("created_at", IngestedType::Timestamp),
                ("weight", IngestedType::Float32),
            ],
        ),
    ];
    infer_primary_keys(&mut tables, None);
    infer_create_time(&mut tables);
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    drop_join_table_primary_keys(&mut tables, &inferred_fkeys);
    prune_fkeys_without_primary_key(&tables, &mut inferred_fkeys);
    infer_primary_key_relationships(&tables, &mut inferred_fkeys);
    infer_edge_tables(&mut tables, &inferred_fkeys);
    prune_fkeys_for_edge_tables(&tables, &mut inferred_fkeys);

    let edge = tables.iter().find(|t| t.name == "user_item_edges").unwrap();
    let edge_spec = edge.edge_table.as_ref().unwrap();
    assert_eq!(edge_spec.src_fk.edge_column, "user_id");
    assert_eq!(edge_spec.src_fk.ref_table, "users");
    assert_eq!(edge_spec.dst_fk.edge_column, "item_id");
    assert_eq!(edge_spec.dst_fk.ref_table, "items");
    assert!(
        !inferred_fkeys
            .iter()
            .any(|fk| fk.src_table == "user_item_edges"),
        "edge table relationships should be pruned"
    );
}

// Tests for edge sort keys removed - infer_edge_sort_keys function removed

#[test]
fn infer_fk_relationship() {
    let tables = vec![
        TableSpec {
            name: "users".to_string(),
            columns: vec![ColumnSpec {
                name: "user_id".to_string(),
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
            primary_key: Some("user_id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "orders".to_string(),
            columns: vec![ColumnSpec {
                name: "user_id".to_string(),
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
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
    ];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    assert_eq!(inferred_fkeys.len(), 1);
    assert_eq!(inferred_fkeys[0].src_table, "orders");
    assert_eq!(inferred_fkeys[0].dst_table, "users");
}

#[test]
fn infer_fk_skips_incompatible() {
    let tables = vec![
        TableSpec {
            name: "devices".to_string(),
            columns: vec![ColumnSpec {
                name: "device_id".to_string(),
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
            primary_key: Some("device_id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "sessions".to_string(),
            columns: vec![ColumnSpec {
                name: "device_id".to_string(),
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
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
    ];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    assert!(inferred_fkeys.is_empty());
    assert!(!notes.is_empty());
}

#[test]
fn infer_pk_prefers_rare_key() {
    let admits = table(
        "admits",
        &[
            ("subject_id", IngestedType::Int64),
            ("hadm_id", IngestedType::Int64),
            ("admittime", IngestedType::Timestamp),
            ("dischtime", IngestedType::Timestamp),
        ],
    );
    let pats = table("pats", &[("subject_id", IngestedType::Int64)]);
    let encounters = table("encounters", &[("subject_id", IngestedType::Int64)]);
    let mut tables = vec![admits, pats, encounters];
    infer_primary_keys(&mut tables, None);
    assert_eq!(tables[0].primary_key.as_deref(), Some("hadm_id"));
}

#[test]
fn infer_fk_prefers_best_token_match() {
    let tables = vec![
        TableSpec {
            name: "d_items".to_string(),
            columns: vec![ColumnSpec {
                name: "itemid".to_string(),
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
            primary_key: Some("itemid".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "d_lab".to_string(),
            columns: vec![ColumnSpec {
                name: "itemid".to_string(),
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
            primary_key: Some("itemid".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "labevents".to_string(),
            columns: vec![ColumnSpec {
                name: "itemid".to_string(),
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
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
    ];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    assert_eq!(inferred_fkeys.len(), 1);
    assert_eq!(inferred_fkeys[0].src_table, "labevents");
    assert_eq!(inferred_fkeys[0].dst_table, "d_lab");
}

#[test]
fn infer_fk_self_reference_suffix() {
    let tables = vec![TableSpec {
        name: "poe".to_string(),
        columns: vec![
            ColumnSpec {
                name: "poe_id".to_string(),
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
                name: "discontinue_of_poe_id".to_string(),
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
        primary_key: Some("poe_id".to_string()),
        create_time: None,
        end_time: None,
        weight: None,
        edge_table: None,
        rowpack: None,
        source: DataSource::Parquet { paths: vec![] },
        source_size: None,
        fkeys: Vec::new(),
        data_version: None,
        kumo_connector_id: None,
        kumo_source_table_name: None,
        group_by_timestamp: false,
    }];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    assert_eq!(inferred_fkeys.len(), 1);
    assert_eq!(inferred_fkeys[0].src_table, "poe");
    assert_eq!(inferred_fkeys[0].dst_table, "poe");
    assert_eq!(inferred_fkeys[0].src_column, "discontinue_of_poe_id");
}

#[test]
fn infer_pk_skips_code_with_version() {
    let mut t = table(
        "d_icd_diag",
        &[
            ("icd_code", IngestedType::String),
            ("icd_version", IngestedType::Int16),
        ],
    );
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key, None);
}

#[test]
fn infer_pk_token_id_match() {
    let mut t = table(
        "icu_stays",
        &[
            ("subject_id", IngestedType::Int64),
            ("hadm_id", IngestedType::Int64),
            ("stay_id", IngestedType::Int64),
        ],
    );
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key.as_deref(), Some("stay_id"));
}

#[test]
fn infer_pk_token_cd_match() {
    let mut t = table(
        "hcpcs",
        &[
            ("subject_id", IngestedType::Int64),
            ("hadm_id", IngestedType::Int64),
            ("hcpcs_cd", IngestedType::String),
        ],
    );
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key.as_deref(), Some("hcpcs_cd"));
}

#[test]
fn infer_pk_small_two_ids_requires_strong_match() {
    let mut t = table(
        "encounters",
        &[
            ("subject_id", IngestedType::Int64),
            ("hadm_id", IngestedType::Int64),
            ("date", IngestedType::Date),
        ],
    );
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key, None);
}

#[test]
fn infer_fk_base_match_allows_code() {
    let tables = vec![
        TableSpec {
            name: "d_hcpcs".to_string(),
            columns: vec![ColumnSpec {
                name: "code".to_string(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            }],
            primary_key: Some("code".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "hcpcs".to_string(),
            columns: vec![ColumnSpec {
                name: "hcpcs_cd".to_string(),
                ingested_dtype: IngestedType::String,
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
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
    ];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    assert_eq!(inferred_fkeys.len(), 1);
    assert_eq!(inferred_fkeys[0].src_table, "hcpcs");
    assert_eq!(inferred_fkeys[0].dst_table, "d_hcpcs");
}

#[test]
fn infer_fk_avoids_generic_code_match() {
    let tables = vec![
        TableSpec {
            name: "d_hcpcs".to_string(),
            columns: vec![ColumnSpec {
                name: "code".to_string(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            }],
            primary_key: Some("code".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "d_icd_diag".to_string(),
            columns: vec![ColumnSpec {
                name: "icd_code".to_string(),
                ingested_dtype: IngestedType::String,
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
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
    ];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    assert!(inferred_fkeys.is_empty());
}

#[test]
fn infer_pk_overlap_ratio_match() {
    let mut t = table(
        "admits",
        &[
            ("subject_id", IngestedType::Int64),
            ("hadm_id", IngestedType::Int64),
            ("admittime", IngestedType::Timestamp),
            ("dischtime", IngestedType::Timestamp),
        ],
    );
    infer_primary_keys(std::slice::from_mut(&mut t), None);
    assert_eq!(t.primary_key.as_deref(), Some("hadm_id"));
}

#[test]
fn infer_fk_prefers_shorter_table_on_tie() {
    let tables = vec![
        TableSpec {
            name: "poe".to_string(),
            columns: vec![ColumnSpec {
                name: "poe_id".to_string(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            }],
            primary_key: Some("poe_id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "poe_detail".to_string(),
            columns: vec![ColumnSpec {
                name: "poe_id".to_string(),
                ingested_dtype: IngestedType::String,
                source_dtype: None,
                timestamp_format: None,
                timestamp_unit: None,
                fixed_len: None,
                precision: None,
                scale: None,
                semantic_type: None,
                multicategorical_delimiter: None,
            }],
            primary_key: Some("poe_id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "emar".to_string(),
            columns: vec![ColumnSpec {
                name: "poe_id".to_string(),
                ingested_dtype: IngestedType::String,
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
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
    ];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    assert_eq!(inferred_fkeys.len(), 1);
    assert_eq!(inferred_fkeys[0].dst_table, "poe");
}

#[test]
fn infer_fk_exact_pk_name_match() {
    let tables = vec![
        TableSpec {
            name: "admits".to_string(),
            columns: vec![ColumnSpec {
                name: "hadm_id".to_string(),
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
            primary_key: Some("hadm_id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "services".to_string(),
            columns: vec![ColumnSpec {
                name: "hadm_id".to_string(),
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
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
    ];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    assert_eq!(inferred_fkeys.len(), 1);
    assert_eq!(inferred_fkeys[0].dst_table, "admits");
}

#[test]
fn infer_fk_itemid_prefers_lab_table() {
    let tables = vec![
        TableSpec {
            name: "d_items".to_string(),
            columns: vec![ColumnSpec {
                name: "itemid".to_string(),
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
            primary_key: Some("itemid".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "d_lab".to_string(),
            columns: vec![ColumnSpec {
                name: "itemid".to_string(),
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
            primary_key: Some("itemid".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "labevents".to_string(),
            columns: vec![ColumnSpec {
                name: "itemid".to_string(),
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
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
    ];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    assert_eq!(inferred_fkeys.len(), 1);
    assert_eq!(inferred_fkeys[0].dst_table, "d_lab");
}

#[test]
fn infer_pk_pk_relationship_when_table_names_overlap() {
    let tables = vec![
        TableSpec {
            name: "user".to_string(),
            columns: vec![ColumnSpec {
                name: "user_id".to_string(),
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
            primary_key: Some("user_id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "user_profile".to_string(),
            columns: vec![ColumnSpec {
                name: "user_id".to_string(),
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
            primary_key: Some("user_id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
    ];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    infer_primary_key_relationships(&tables, &mut inferred_fkeys);
    assert_eq!(inferred_fkeys.len(), 1);
    assert!(inferred_fkeys[0].src_table == "user" || inferred_fkeys[0].src_table == "user_profile");
}

#[test]
fn infer_pk_pk_skips_generic_id() {
    let tables = vec![
        TableSpec {
            name: "alpha".to_string(),
            columns: vec![ColumnSpec {
                name: "id".to_string(),
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
            primary_key: Some("id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
        TableSpec {
            name: "beta".to_string(),
            columns: vec![ColumnSpec {
                name: "id".to_string(),
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
            primary_key: Some("id".to_string()),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet { paths: vec![] },
            source_size: None,
            fkeys: Vec::new(),
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        },
    ];
    let mut inferred_fkeys = Vec::new();
    let mut notes = Vec::new();
    infer_foreign_keys(&tables, &mut inferred_fkeys, &mut notes, None);
    infer_primary_key_relationships(&tables, &mut inferred_fkeys);
    assert!(inferred_fkeys.is_empty());
}

#[test]
fn infer_create_time_from_timestamp_col() {
    let mut t = table(
        "events",
        &[
            ("created_at", IngestedType::Timestamp),
            ("id", IngestedType::Int64),
        ],
    );
    infer_create_time(std::slice::from_mut(&mut t));
    assert_eq!(t.create_time.as_deref(), Some("created_at"));
}

#[test]
fn csv_infers_timestamp_from_name_and_ints() {
    let mut stats = ColumnStats::default();
    for v in ["1693526400", "1693612800", "1693699200"] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("created_at");
    assert!(matches!(col.ingested_dtype, IngestedType::Timestamp));
}

#[test]
fn csv_rejects_timestamp_out_of_range() {
    let mut stats = ColumnStats::default();
    for v in ["12", "13", "14"] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("created_at");
    assert!(matches!(col.ingested_dtype, IngestedType::Int64));
}

#[test]
fn csv_infers_date_from_iso() {
    let mut stats = ColumnStats::default();
    for v in ["2024-01-01", "2024-01-02"] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("event_date");
    assert!(matches!(col.ingested_dtype, IngestedType::Date));
}

#[test]
fn csv_infers_bool() {
    let mut stats = ColumnStats::default();
    for v in ["true", "false", "true"] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("flag");
    assert!(matches!(col.ingested_dtype, IngestedType::Bool));
}

#[test]
fn csv_infers_float32_from_float64() {
    let mut stats = ColumnStats::default();
    for v in ["1.5", "2.25", "3.0"] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("score");
    assert!(matches!(col.ingested_dtype, IngestedType::Float32));
}

#[test]
fn csv_infers_uuid_dtype() {
    let mut stats = ColumnStats::default();
    for v in [
        "fdda48b5ba510e70ddcbe3932fc6d7e0",
        "fdda48b5ba510e70ddcbe3932fc6d7e0",
    ] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("user_id");
    assert!(matches!(col.ingested_dtype, IngestedType::Uuid));
    assert_eq!(col.fixed_len, None);
}

#[test]
fn csv_infers_uuid256_dtype() {
    let mut stats = ColumnStats::default();
    for v in [
        "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
        "ffffffffffffffffffffffffffffffff00000000000000000000000000000000",
    ] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("user_id");
    assert!(matches!(col.ingested_dtype, IngestedType::Uuid256));
    assert_eq!(col.fixed_len, None);
}

#[test]
fn csv_infers_ip_dtype() {
    let mut stats = ColumnStats::default();
    for v in ["192.168.1.1", "2001:db8::1"] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("ip");
    assert!(matches!(col.ingested_dtype, IngestedType::Ip));
    assert_eq!(col.fixed_len, None);
}

#[test]
fn csv_infers_fixed_len() {
    let mut stats = ColumnStats::default();
    for v in ["ABC", "DEF"] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("code");
    assert!(matches!(col.ingested_dtype, IngestedType::String));
    assert_eq!(col.fixed_len, Some(3));
}

#[test]
fn parquet_lists_infer_float16_arrays() {
    let schema = Schema::new(vec![
        Field::new(
            "vec32",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        ),
        Field::new(
            "vec64",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        ),
    ]);
    let cols = infer_parquet_schema_from_arrow(&schema, None).unwrap();
    let vec32 = cols.iter().find(|c| c.name == "vec32").unwrap();
    let vec64 = cols.iter().find(|c| c.name == "vec64").unwrap();
    assert!(matches!(vec32.ingested_dtype, IngestedType::Float16Array));
    assert!(matches!(vec64.ingested_dtype, IngestedType::Float16Array));
}

#[test]
fn parquet_infers_float64_from_float64() {
    let schema = Schema::new(vec![Field::new(
        "production_budget",
        DataType::Float64,
        true,
    )]);
    let cols = infer_parquet_schema_from_arrow(&schema, None).unwrap();
    let col = cols.iter().find(|c| c.name == "production_budget").unwrap();
    assert!(
        matches!(col.ingested_dtype, IngestedType::Float64),
        "expected Float64 to preserve precision, got {:?}",
        col.ingested_dtype
    );
}

#[test]
fn parquet_infers_float32_from_float32() {
    let schema = Schema::new(vec![Field::new("weight", DataType::Float32, true)]);
    let cols = infer_parquet_schema_from_arrow(&schema, None).unwrap();
    let col = cols.iter().find(|c| c.name == "weight").unwrap();
    assert!(matches!(col.ingested_dtype, IngestedType::Float32));
}

#[test]
fn parquet_list_float64_unchanged() {
    let schema = Schema::new(vec![Field::new(
        "embedding",
        DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
        true,
    )]);
    let cols = infer_parquet_schema_from_arrow(&schema, None).unwrap();
    let col = cols.iter().find(|c| c.name == "embedding").unwrap();
    assert!(matches!(col.ingested_dtype, IngestedType::Float16Array));
}

#[test]
fn csv_infers_timestamp_from_common_format() {
    let mut stats = ColumnStats::default();
    for v in ["2024-01-01 12:34:56", "2024-01-01 12:35:56"] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("charttime");
    assert!(matches!(col.ingested_dtype, IngestedType::Timestamp));
}

#[test]
fn csv_infers_timestamp_from_fractional_seconds() {
    let mut stats = ColumnStats::default();
    for v in ["2024-01-01 12:34:56.123", "2024-01-01 12:35:56.987"] {
        stats.observe(v);
    }
    let col = stats.to_column_spec("storetime");
    assert!(matches!(col.ingested_dtype, IngestedType::Timestamp));
}

#[test]
fn infer_create_time_prefers_start_over_end() {
    let mut t = table(
        "events",
        &[
            ("start_time", IngestedType::Timestamp),
            ("end_time", IngestedType::Timestamp),
        ],
    );
    infer_create_time(std::slice::from_mut(&mut t));
    assert_eq!(t.create_time.as_deref(), Some("start_time"));
}

#[test]
fn infer_create_time_skips_negative_only() {
    let mut t = table(
        "events",
        &[
            ("end_time", IngestedType::Timestamp),
            ("stop_time", IngestedType::Timestamp),
        ],
    );
    infer_create_time(std::slice::from_mut(&mut t));
    assert_eq!(t.create_time, None);
}

#[test]
fn infer_create_time_prefers_created_over_updated() {
    let mut t = table(
        "accounts",
        &[
            ("created_at", IngestedType::Timestamp),
            ("updated_at", IngestedType::Timestamp),
        ],
    );
    infer_create_time(std::slice::from_mut(&mut t));
    assert_eq!(t.create_time.as_deref(), Some("created_at"));
}

#[test]
fn infer_create_time_ignores_negative_token() {
    let mut t = table(
        "orders",
        &[
            ("authorized_time", IngestedType::Timestamp),
            ("voided_time", IngestedType::Timestamp),
        ],
    );
    infer_create_time(std::slice::from_mut(&mut t));
    assert_eq!(t.create_time.as_deref(), Some("authorized_time"));
}

#[test]
fn infer_create_time_prefers_positive_when_present() {
    let mut t = table(
        "shipping",
        &[
            ("shipped_at", IngestedType::Timestamp),
            ("returned_at", IngestedType::Timestamp),
        ],
    );
    infer_create_time(std::slice::from_mut(&mut t));
    assert_eq!(t.create_time.as_deref(), Some("shipped_at"));
}

#[test]
fn infer_create_time_prefers_chart_over_store() {
    let mut t = table(
        "chart",
        &[
            ("charttime", IngestedType::Timestamp),
            ("storetime", IngestedType::Timestamp),
        ],
    );
    infer_create_time(std::slice::from_mut(&mut t));
    assert_eq!(t.create_time.as_deref(), Some("charttime"));
}

#[test]
fn infer_create_time_prefers_start_over_store() {
    let mut t = table(
        "inputevents",
        &[
            ("starttime", IngestedType::Timestamp),
            ("storetime", IngestedType::Timestamp),
        ],
    );
    infer_create_time(std::slice::from_mut(&mut t));
    assert_eq!(t.create_time.as_deref(), Some("starttime"));
}

#[test]
fn infer_create_time_prefers_in_over_out() {
    let mut t = table(
        "transfers",
        &[
            ("intime", IngestedType::Timestamp),
            ("outtime", IngestedType::Timestamp),
        ],
    );
    infer_create_time(std::slice::from_mut(&mut t));
    assert_eq!(t.create_time.as_deref(), Some("intime"));
}

#[test]
fn compatible_numeric_to_var_string() {
    let a = ColumnSpec {
        name: "user_id".to_string(),
        ingested_dtype: IngestedType::Int64,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    };
    let b = ColumnSpec {
        name: "id".to_string(),
        ingested_dtype: IngestedType::String,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    };
    assert!(compatible_key_columns(&a, &b));
}

#[test]
fn incompatible_uuid_and_bytes() {
    let a = ColumnSpec {
        name: "device_id".to_string(),
        ingested_dtype: IngestedType::Uuid,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    };
    let b = ColumnSpec {
        name: "id".to_string(),
        ingested_dtype: IngestedType::Bytes,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    };
    assert!(!compatible_key_columns(&a, &b));
}

#[test]
fn compatible_uuid() {
    let a = ColumnSpec {
        name: "id".to_string(),
        ingested_dtype: IngestedType::Uuid,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    };
    let b = ColumnSpec {
        name: "id".to_string(),
        ingested_dtype: IngestedType::Uuid,
        source_dtype: None,
        timestamp_format: None,
        timestamp_unit: None,
        fixed_len: None,
        precision: None,
        scale: None,
        semantic_type: None,
        multicategorical_delimiter: None,
    };
    assert!(compatible_key_columns(&a, &b));
}
