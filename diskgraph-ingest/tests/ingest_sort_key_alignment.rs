// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression test for sort key / node data misalignment when an edge table
//! is split across multiple parquet files and processed in parallel.
//!
//! **The Bug**: `NodeBatcher` and `SortKeyComponentBatcher` flush at different
//! rates (node data is variable-size ~100-200 bytes/row, sort keys are fixed
//! 4-8 bytes/row) and go to separate channel pools routed by different hashes
//! (`table_hash` for nodes, `rel_hash` for sort keys). With concurrent file
//! processing the interleaving of batches from different files can differ
//! between the node temp file and the sort key temp file, causing row N in one
//! to correspond to a different source row than row N in the other.
//!
//! **Verification strategy**:
//!   - Create an edge table `Purchase(user_id, item_id, event_time)` where
//!     `event_time = user_id * 1000 + item_id` (each row has a UNIQUE time).
//!   - Split the edge table across 4 parquet files with 50 000 rows each.
//!   - After ingest, iterate every `User` node and every forward edge
//!     (`edge_Purchase_src_to_dst`). For each edge, decode the dest node id
//!     back to an `item_id` via PK lookup, and decode the embedded sort key
//!     back to a timestamp. The sort key must equal
//!     `encode_ts(user_id * 1000 + item_id)` for the (user_id, item_id) pair
//!     that produced this edge.
//!   - If the sort keys were misaligned with node data, the decoded timestamp
//!     will belong to a *different* source row, and the assertion will fail.

use anyhow::{anyhow, Result};
use arrow_array::{ArrayRef, Int64Array, RecordBatch, TimestampSecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_ingest::{ingest, Dest, IngestOptions};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_parquet(path: &Path, schema: Schema, columns: Vec<ArrayRef>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(8192))
        .build();
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))?;
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Encode a timestamp the same way the ingest pipeline does for a 32-bit
/// sort key component with `anchor_time=0, unit="s"` (i.e. step=1).
fn encode_ts_i32(value: i64, anchor: i64, step: i64) -> u64 {
    let delta = value.saturating_sub(anchor).div_euclid(step.max(1));
    let clamped = delta.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let raw = clamped as u32 as u64;
    raw ^ 0x8000_0000
}

/// Extract (dest_node_id, sort_key) pairs for a given relation on a given
/// source node.
fn extract_relation_entries(
    table: &diskgraph_index::graph::TableReader,
    rel_name: &str,
    node_id: u64,
) -> Result<Vec<(u64, u64)>> {
    let rel_idx = table
        .relations()
        .iter()
        .position(|rel| rel.name == rel_name)
        .ok_or_else(|| anyhow!("missing relation {rel_name}"))?;
    let view = table.edges_for_node(node_id)?;
    let segments = view.relation_segments()?;
    let seg = &segments[rel_idx];
    let mut buf = [0u8; 16];
    let mut out = Vec::new();
    for chunk in seg.data.chunks(seg.layout.entry_bytes) {
        buf[..seg.layout.entry_bytes].copy_from_slice(chunk);
        let val = u128::from_le_bytes(buf);
        let dst_mask = seg.layout.dst_mask as u128;
        let dst = (val & dst_mask) as u64;
        let sort = if let Some(bits) = seg.layout.dest_sort_key_bits {
            ((val >> seg.layout.dst_bits as u128) & ((1u128 << bits) - 1)) as u64
        } else {
            0
        };
        out.push((dst, sort));
        buf.fill(0);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Test data generation
// ---------------------------------------------------------------------------

const NUM_USERS: i64 = 100;
const NUM_ITEMS: i64 = 100;
const ROWS_PER_FILE: usize = 50_000;
const NUM_FILES: usize = 4;
const TOTAL_ROWS: usize = ROWS_PER_FILE * NUM_FILES;
const ANCHOR_TIME: i64 = 0;
const TS_STEP: i64 = 1;

/// Deterministic timestamp for a given (user_id, item_id) pair.
/// Each value is unique across the entire dataset.
fn expected_ts(user_id: i64, item_id: i64) -> i64 {
    user_id * 1000 + item_id
}

/// Generate the edge table rows for a single parquet file.
/// `file_idx` selects which slice of the global row space this file covers.
fn generate_edge_file_arrays(file_idx: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let start = file_idx * ROWS_PER_FILE;
    let mut user_ids = Vec::with_capacity(ROWS_PER_FILE);
    let mut item_ids = Vec::with_capacity(ROWS_PER_FILE);
    let mut timestamps = Vec::with_capacity(ROWS_PER_FILE);
    for row in start..start + ROWS_PER_FILE {
        // Distribute rows across users/items deterministically but with
        // mixing so that different files have overlapping user/item ranges.
        let uid = (row as i64) % NUM_USERS;
        let iid = (row as i64 / NUM_USERS) % NUM_ITEMS;
        let ts = expected_ts(uid, iid);
        user_ids.push(uid);
        item_ids.push(iid);
        timestamps.push(ts);
    }
    (user_ids, item_ids, timestamps)
}

// ---------------------------------------------------------------------------
// The actual test
// ---------------------------------------------------------------------------

/// This test is `#[ignore]` because it creates a 200 K row dataset and runs
/// the full ingest pipeline, which takes a few seconds.
#[test]
#[ignore]
fn sort_key_alignment_multi_file_edge_table() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    // -- 1. Write the User node table (100 rows) ----------------------------
    let user_ids: Vec<i64> = (0..NUM_USERS).collect();
    write_parquet(
        &input_root.join("users.parquet"),
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from(user_ids)) as ArrayRef],
    )?;

    // -- 2. Write the Item node table (100 rows) ----------------------------
    let item_ids: Vec<i64> = (0..NUM_ITEMS).collect();
    write_parquet(
        &input_root.join("items.parquet"),
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from(item_ids)) as ArrayRef],
    )?;

    // -- 3. Write the Purchase edge table across NUM_FILES parquet files -----
    let edge_schema = Schema::new(vec![
        Field::new("user_id", DataType::Int64, false),
        Field::new("item_id", DataType::Int64, false),
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
    ]);
    let mut edge_paths = Vec::new();
    for file_idx in 0..NUM_FILES {
        let path = input_root.join(format!("purchases_part{file_idx}.parquet"));
        let (uids, iids, tss) = generate_edge_file_arrays(file_idx);
        write_parquet(
            &path,
            edge_schema.clone(),
            vec![
                Arc::new(Int64Array::from(uids)) as ArrayRef,
                Arc::new(Int64Array::from(iids)) as ArrayRef,
                Arc::new(TimestampSecondArray::from(tss)) as ArrayRef,
            ],
        )?;
        edge_paths.push(path);
    }

    // -- 4. Build the diskgraph config --------------------------------------
    let edge_path_strs: Vec<String> = edge_paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let source = serde_json::json!({
        "dataset_name": "sort_key_alignment_test",
        "tables": [
            {
                "name": "User",
                "source": {
                    "type": "parquet",
                    "paths": [input_root.join("users.parquet").to_string_lossy()]
                },
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "Item",
                "source": {
                    "type": "parquet",
                    "paths": [input_root.join("items.parquet").to_string_lossy()]
                },
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "Purchase",
                "source": {
                    "type": "parquet",
                    "paths": edge_path_strs
                },
                "edge_table": {
                    "src_fk": {"edge_column": "user_id", "ref_table": "User"},
                    "dst_fk": {"edge_column": "item_id", "ref_table": "Item"},
                    "forward": {
                        "time": {
                            "table": "Purchase",
                            "column": "event_time",
                            "bitwidth": 32,
                            "anchor_time": 0,
                            "unit": "s"
                        }
                    },
                    "reverse": {
                        "time": {
                            "table": "Purchase",
                            "column": "event_time",
                            "bitwidth": 32,
                            "anchor_time": 0,
                            "unit": "s"
                        }
                    }
                },
                "columns": [
                    {"name": "user_id", "ingested_dtype": "int64"},
                    {"name": "item_id", "ingested_dtype": "int64"},
                    {"name": "event_time", "ingested_dtype": "timestamp",
                     "timestamp_unit": "seconds"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    // -- 5. Run ingest -------------------------------------------------------
    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
            parallel: true, // essential: the bug requires parallel file processing
            index_mem_budget_bytes: None,
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
            resume_from_phase: None,
            source_loader_threads: None,
            source_encoder_threads: None,
            pk_map_threads: None,
            edges_map_threads: None,
            index_threads: None,
            parent_index_path: None,
            resolve_columns: Vec::new(),
        },
    )?;

    // Sanity: both edge relations should carry a 32-bit sort key.
    let forward = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "edge_Purchase_src_to_dst")
        .ok_or_else(|| anyhow!("missing forward edge relation"))?;
    let reverse = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "edge_Purchase_dst_to_src")
        .ok_or_else(|| anyhow!("missing reverse edge relation"))?;
    assert_eq!(
        forward.dest_sort_key_bits,
        Some(32),
        "forward sort key should be 32 bits"
    );
    assert_eq!(
        reverse.dest_sort_key_bits,
        Some(32),
        "reverse sort key should be 32 bits"
    );

    // -- 6. Open the index and build a PK -> node id map for Items ----------
    let reader =
        GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let user_table = reader.table("User").unwrap();
    let item_table = reader.table("Item").unwrap();

    // Build item_node_id -> item_pk_value map by iterating all item nodes.
    let mut item_node_to_pk: HashMap<u64, i64> = HashMap::new();
    for item_pk in 0..NUM_ITEMS {
        let key = item_table.parse_pk_key(&item_pk.to_string())?;
        let node_id = item_table
            .node_id_from_pk(&key)?
            .ok_or_else(|| anyhow!("item pk {item_pk} not found in index"))?;
        item_node_to_pk.insert(node_id, item_pk);
    }

    // Build user_node_id -> user_pk_value map.
    let mut user_node_to_pk: HashMap<u64, i64> = HashMap::new();
    for user_pk in 0..NUM_USERS {
        let key = user_table.parse_pk_key(&user_pk.to_string())?;
        let node_id = user_table
            .node_id_from_pk(&key)?
            .ok_or_else(|| anyhow!("user pk {user_pk} not found in index"))?;
        user_node_to_pk.insert(node_id, user_pk);
    }

    // -- 7. Build the ground truth set of (user_id, item_id, encoded_ts) ----
    //        from the raw data we generated.
    let mut expected_edges: HashSet<(i64, i64, u64)> = HashSet::new();
    for file_idx in 0..NUM_FILES {
        let (uids, iids, _) = generate_edge_file_arrays(file_idx);
        for i in 0..ROWS_PER_FILE {
            let uid = uids[i];
            let iid = iids[i];
            let ts = expected_ts(uid, iid);
            let encoded = encode_ts_i32(ts, ANCHOR_TIME, TS_STEP);
            expected_edges.insert((uid, iid, encoded));
        }
    }

    // -- 8. Verify FORWARD edges (User -> Item via edge_Purchase_src_to_dst) -
    let mut total_forward_edges: u64 = 0;
    let mut misaligned_forward: Vec<String> = Vec::new();
    for user_node_id in 0..user_table.node_count() {
        let user_pk = *user_node_to_pk
            .get(&user_node_id)
            .ok_or_else(|| anyhow!("user node {user_node_id} not in pk map"))?;
        let entries =
            extract_relation_entries(&user_table, "edge_Purchase_src_to_dst", user_node_id)?;
        for (dst_node_id, sort_key) in &entries {
            total_forward_edges += 1;
            let item_pk = match item_node_to_pk.get(dst_node_id) {
                Some(pk) => *pk,
                None => {
                    misaligned_forward.push(format!(
                        "forward: user_node={user_node_id} (pk={user_pk}) -> \
                         item_node={dst_node_id} NOT in pk map"
                    ));
                    continue;
                }
            };
            let ts = expected_ts(user_pk, item_pk);
            let expected_encoded = encode_ts_i32(ts, ANCHOR_TIME, TS_STEP);
            if *sort_key != expected_encoded {
                misaligned_forward.push(format!(
                    "forward: user_pk={user_pk}, item_pk={item_pk}: \
                     sort_key={sort_key:#010x} but expected={expected_encoded:#010x} \
                     (ts={ts})"
                ));
            }
        }
    }

    // -- 9. Verify REVERSE edges (Item -> User via edge_Purchase_dst_to_src) -
    let mut total_reverse_edges: u64 = 0;
    let mut misaligned_reverse: Vec<String> = Vec::new();
    for item_node_id in 0..item_table.node_count() {
        let item_pk = *item_node_to_pk
            .get(&item_node_id)
            .ok_or_else(|| anyhow!("item node {item_node_id} not in pk map"))?;
        let entries =
            extract_relation_entries(&item_table, "edge_Purchase_dst_to_src", item_node_id)?;
        for (dst_node_id, sort_key) in &entries {
            total_reverse_edges += 1;
            let user_pk = match user_node_to_pk.get(dst_node_id) {
                Some(pk) => *pk,
                None => {
                    misaligned_reverse.push(format!(
                        "reverse: item_node={item_node_id} (pk={item_pk}) -> \
                         user_node={dst_node_id} NOT in pk map"
                    ));
                    continue;
                }
            };
            let ts = expected_ts(user_pk, item_pk);
            let expected_encoded = encode_ts_i32(ts, ANCHOR_TIME, TS_STEP);
            if *sort_key != expected_encoded {
                misaligned_reverse.push(format!(
                    "reverse: item_pk={item_pk}, user_pk={user_pk}: \
                     sort_key={sort_key:#010x} but expected={expected_encoded:#010x} \
                     (ts={ts})"
                ));
            }
        }
    }

    // -- 10. Report results -------------------------------------------------
    // Some edges may be deduplicated if the same (user_id, item_id) pair
    // appears multiple times (which it does in our data).  Count unique pairs.
    let expected_unique = expected_edges.len() as u64;
    eprintln!(
        "Forward edges: {total_forward_edges} (expected unique pairs: {expected_unique})"
    );
    eprintln!(
        "Reverse edges: {total_reverse_edges} (expected unique pairs: {expected_unique})"
    );

    if !misaligned_forward.is_empty() {
        let count = misaligned_forward.len();
        let sample: Vec<_> = misaligned_forward.iter().take(20).collect();
        panic!(
            "SORT KEY MISALIGNMENT detected on {count} forward edges \
             (out of {total_forward_edges}).\n\
             First {} samples:\n{}",
            sample.len(),
            sample
                .iter()
                .map(|s| format!("  {s}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    if !misaligned_reverse.is_empty() {
        let count = misaligned_reverse.len();
        let sample: Vec<_> = misaligned_reverse.iter().take(20).collect();
        panic!(
            "SORT KEY MISALIGNMENT detected on {count} reverse edges \
             (out of {total_reverse_edges}).\n\
             First {} samples:\n{}",
            sample.len(),
            sample
                .iter()
                .map(|s| format!("  {s}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    // All edges verified: sort keys match node data.
    assert!(
        total_forward_edges > 0,
        "expected at least some forward edges"
    );
    assert!(
        total_reverse_edges > 0,
        "expected at least some reverse edges"
    );

    // The data has TOTAL_ROWS edge rows but many (user_id, item_id) pairs
    // will repeat (200K rows across 100*100=10K unique pairs).  After dedup
    // we should still see all unique pairs present.
    // Note: the ingest pipeline does NOT dedup edges, so we expect TOTAL_ROWS
    // edges in each direction.
    assert_eq!(
        total_forward_edges, TOTAL_ROWS as u64,
        "expected {TOTAL_ROWS} forward edges (one per input row)"
    );
    assert_eq!(
        total_reverse_edges, TOTAL_ROWS as u64,
        "expected {TOTAL_ROWS} reverse edges (one per input row)"
    );

    Ok(())
}

/// Smaller variant of the alignment test (1 000 rows, 2 files) that runs
/// without `#[ignore]` so it executes in the normal test suite. This variant
/// is too small to reliably trigger the race condition, but it validates the
/// test infrastructure and catches trivial encoding errors.
#[test]
fn sort_key_alignment_small() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    const S_USERS: i64 = 10;
    const S_ITEMS: i64 = 10;
    const S_ROWS_PER_FILE: usize = 500;
    const S_NUM_FILES: usize = 2;

    // Users
    write_parquet(
        &input_root.join("users.parquet"),
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from((0..S_USERS).collect::<Vec<_>>())) as ArrayRef],
    )?;

    // Items
    write_parquet(
        &input_root.join("items.parquet"),
        Schema::new(vec![Field::new("id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from((0..S_ITEMS).collect::<Vec<_>>())) as ArrayRef],
    )?;

    // Edge files
    let edge_schema = Schema::new(vec![
        Field::new("user_id", DataType::Int64, false),
        Field::new("item_id", DataType::Int64, false),
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
    ]);
    let mut edge_paths = Vec::new();
    for file_idx in 0..S_NUM_FILES {
        let path = input_root.join(format!("purchases_part{file_idx}.parquet"));
        let start = file_idx * S_ROWS_PER_FILE;
        let mut uids = Vec::with_capacity(S_ROWS_PER_FILE);
        let mut iids = Vec::with_capacity(S_ROWS_PER_FILE);
        let mut tss = Vec::with_capacity(S_ROWS_PER_FILE);
        for row in start..start + S_ROWS_PER_FILE {
            let uid = (row as i64) % S_USERS;
            let iid = (row as i64 / S_USERS) % S_ITEMS;
            let ts = uid * 1000 + iid;
            uids.push(uid);
            iids.push(iid);
            tss.push(ts);
        }
        write_parquet(
            &path,
            edge_schema.clone(),
            vec![
                Arc::new(Int64Array::from(uids)) as ArrayRef,
                Arc::new(Int64Array::from(iids)) as ArrayRef,
                Arc::new(TimestampSecondArray::from(tss)) as ArrayRef,
            ],
        )?;
        edge_paths.push(path);
    }

    let edge_path_strs: Vec<String> = edge_paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let source = serde_json::json!({
        "dataset_name": "sort_key_alignment_small_test",
        "tables": [
            {
                "name": "User",
                "source": {
                    "type": "parquet",
                    "paths": [input_root.join("users.parquet").to_string_lossy()]
                },
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "Item",
                "source": {
                    "type": "parquet",
                    "paths": [input_root.join("items.parquet").to_string_lossy()]
                },
                "primary_key": "id",
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "Purchase",
                "source": {
                    "type": "parquet",
                    "paths": edge_path_strs
                },
                "edge_table": {
                    "src_fk": {"edge_column": "user_id", "ref_table": "User"},
                    "dst_fk": {"edge_column": "item_id", "ref_table": "Item"},
                    "forward": {
                        "time": {
                            "table": "Purchase",
                            "column": "event_time",
                            "bitwidth": 32,
                            "anchor_time": 0,
                            "unit": "s"
                        }
                    },
                    "reverse": {
                        "time": {
                            "table": "Purchase",
                            "column": "event_time",
                            "bitwidth": 32,
                            "anchor_time": 0,
                            "unit": "s"
                        }
                    }
                },
                "columns": [
                    {"name": "user_id", "ingested_dtype": "int64"},
                    {"name": "item_id", "ingested_dtype": "int64"},
                    {"name": "event_time", "ingested_dtype": "timestamp",
                     "timestamp_unit": "seconds"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions::default(),
    )?;

    let forward = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == "edge_Purchase_src_to_dst")
        .ok_or_else(|| anyhow!("missing forward edge relation"))?;
    assert_eq!(forward.dest_sort_key_bits, Some(32));

    let reader =
        GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let user_table = reader.table("User").unwrap();
    let item_table = reader.table("Item").unwrap();

    // item node_id -> item PK
    let mut item_node_to_pk: HashMap<u64, i64> = HashMap::new();
    for pk in 0..S_ITEMS {
        let key = item_table.parse_pk_key(&pk.to_string())?;
        let nid = item_table.node_id_from_pk(&key)?.unwrap();
        item_node_to_pk.insert(nid, pk);
    }
    // user node_id -> user PK
    let mut user_node_to_pk: HashMap<u64, i64> = HashMap::new();
    for pk in 0..S_USERS {
        let key = user_table.parse_pk_key(&pk.to_string())?;
        let nid = user_table.node_id_from_pk(&key)?.unwrap();
        user_node_to_pk.insert(nid, pk);
    }

    let mut total = 0u64;
    let mut bad = 0u64;
    for user_nid in 0..user_table.node_count() {
        let user_pk = user_node_to_pk[&user_nid];
        let entries =
            extract_relation_entries(&user_table, "edge_Purchase_src_to_dst", user_nid)?;
        for (dst_nid, sort_key) in entries {
            total += 1;
            let item_pk = item_node_to_pk[&dst_nid];
            let ts = user_pk * 1000 + item_pk;
            let expected = encode_ts_i32(ts, 0, 1);
            if sort_key != expected {
                bad += 1;
            }
        }
    }
    assert_eq!(
        bad, 0,
        "{bad} out of {total} forward edges have misaligned sort keys"
    );
    assert_eq!(total, (S_ROWS_PER_FILE * S_NUM_FILES) as u64);

    Ok(())
}

// ---------------------------------------------------------------------------
// Node-table FK alignment test
// ---------------------------------------------------------------------------

/// Regression test for sort key / node data misalignment on **node tables**
/// with FK relationships (not edge tables). The reverse edges
/// (`customer → account_daily`) must carry sort keys that match the
/// destination `account_daily` node's actual `action_date`.
///
/// **Setup**:
///   - `customer` table: 100 rows, PK = `customer_id`
///   - `account_daily` table: 200K rows across 4 parquet files,
///     PK = `id`, FK `customer_id` → `customer`,
///     sort key on `action_date` column
///   - `action_date = customer_id * 10_000 + sequential_within_customer`
///     so each row has a unique timestamp and we can verify alignment.
///
/// **Verification**: For each `customer` node, iterate reverse edges
/// (`customer_id_to_customer`), look up the destination `account_daily`
/// node, and verify the embedded sort key matches that node's `action_date`.
#[test]
#[ignore]
fn sort_key_alignment_node_table_fk() -> Result<()> {
    let dir = tempdir()?;
    let input_root = dir.path().join("input");
    let output_root = dir.path().join("output");

    const N_CUSTOMERS: i64 = 100;
    const ROWS_PER_FK_FILE: usize = 50_000;
    const N_FK_FILES: usize = 4;
    const TOTAL_FK_ROWS: usize = ROWS_PER_FK_FILE * N_FK_FILES;

    // Deterministic timestamp for each account_daily row.
    let action_date_for = |customer_id: i64, seq: i64| -> i64 {
        customer_id * 10_000 + seq
    };

    // -- 1. Write the customer node table (100 rows) --------------------------
    let cust_ids: Vec<i64> = (0..N_CUSTOMERS).collect();
    write_parquet(
        &input_root.join("customers.parquet"),
        Schema::new(vec![Field::new("customer_id", DataType::Int64, false)]),
        vec![Arc::new(Int64Array::from(cust_ids)) as ArrayRef],
    )?;

    // -- 2. Write the account_daily node table across N_FK_FILES parquet files -
    let ad_schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("customer_id", DataType::Int64, false),
        Field::new(
            "action_date",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
    ]);

    // Track (id, customer_id, action_date) for verification.
    let mut all_ids = Vec::with_capacity(TOTAL_FK_ROWS);
    let mut all_cids = Vec::with_capacity(TOTAL_FK_ROWS);
    let mut all_dates = Vec::with_capacity(TOTAL_FK_ROWS);

    let mut ad_paths = Vec::new();
    let mut per_customer_seq: Vec<i64> = vec![0; N_CUSTOMERS as usize];

    for file_idx in 0..N_FK_FILES {
        let path = input_root.join(format!("account_daily_part{file_idx}.parquet"));
        let start = file_idx * ROWS_PER_FK_FILE;
        let mut ids = Vec::with_capacity(ROWS_PER_FK_FILE);
        let mut cids = Vec::with_capacity(ROWS_PER_FK_FILE);
        let mut dates = Vec::with_capacity(ROWS_PER_FK_FILE);

        for row in start..start + ROWS_PER_FK_FILE {
            let id = row as i64;
            // Mix customers across files so each file has rows for many customers.
            let cid = (row as i64) % N_CUSTOMERS;
            let seq = per_customer_seq[cid as usize];
            per_customer_seq[cid as usize] += 1;
            let ts = action_date_for(cid, seq);

            ids.push(id);
            cids.push(cid);
            dates.push(ts);

            all_ids.push(id);
            all_cids.push(cid);
            all_dates.push(ts);
        }

        write_parquet(
            &path,
            ad_schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)) as ArrayRef,
                Arc::new(Int64Array::from(cids)) as ArrayRef,
                Arc::new(TimestampSecondArray::from(dates)) as ArrayRef,
            ],
        )?;
        ad_paths.push(path);
    }

    // -- 3. Build the diskgraph config ----------------------------------------
    let ad_path_strs: Vec<String> = ad_paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    let source = serde_json::json!({
        "dataset_name": "sort_key_alignment_node_fk_test",
        "tables": [
            {
                "name": "customer",
                "source": {
                    "type": "parquet",
                    "paths": [input_root.join("customers.parquet").to_string_lossy()]
                },
                "primary_key": "customer_id",
                "columns": [
                    {"name": "customer_id", "ingested_dtype": "int64"}
                ]
            },
            {
                "name": "account_daily",
                "source": {
                    "type": "parquet",
                    "paths": ad_path_strs
                },
                "primary_key": "id",
                "fkeys": [
                    {
                        "src_column": "customer_id",
                        "dst_table": "customer",
                        "reverse": {
                            "time": {
                                "table": "account_daily",
                                "column": "action_date",
                                "bitwidth": 32,
                                "anchor_time": 0,
                                "unit": "s"
                            }
                        }
                    }
                ],
                "columns": [
                    {"name": "id", "ingested_dtype": "int64"},
                    {"name": "customer_id", "ingested_dtype": "int64"},
                    {"name": "action_date", "ingested_dtype": "timestamp",
                     "timestamp_unit": "seconds"}
                ]
            }
        ]
    });
    let source_path = dir.path().join("diskgraph_config.json");
    fs::write(&source_path, serde_json::to_vec_pretty(&source)?)?;

    // -- 4. Run ingest --------------------------------------------------------
    let metadata = ingest(
        &source_path,
        output_root.to_str().unwrap(),
        Dest::Local,
        IngestOptions {
            profile_dir: None,
            progress: None,
            parallel: true, // essential: the bug requires parallel file processing
            index_mem_budget_bytes: None,
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: diskgraph_ingest::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
            resume_from_phase: None,
            source_loader_threads: None,
            source_encoder_threads: None,
            pk_map_threads: None,
            edges_map_threads: None,
            index_threads: None,
            parent_index_path: None,
            resolve_columns: Vec::new(),
        },
    )?;

    // The reverse FK relation name follows the pattern:
    // {src_table}_{src_col}_to_{dst_table}_{dst_pk}_reverse
    let reverse_rel_name = "account_daily_customer_id_to_customer_customer_id_reverse";
    let reverse_rel = metadata
        .edge_relations
        .iter()
        .find(|r| r.name == reverse_rel_name)
        .ok_or_else(|| {
            let names: Vec<_> = metadata.edge_relations.iter().map(|r| &r.name).collect();
            anyhow!(
                "missing reverse edge relation '{reverse_rel_name}'; \
                 available: {names:?}"
            )
        })?;
    assert_eq!(
        reverse_rel.dest_sort_key_bits,
        Some(32),
        "reverse sort key should be 32 bits"
    );

    // -- 5. Open the index and build lookup maps ------------------------------
    let reader =
        GraphIndexReader::open(&output_root.join("index_metadata.json"), &output_root)?;
    let customer_table = reader.table("customer").unwrap();
    let ad_table = reader.table("account_daily").unwrap();

    // Build account_daily node_id -> (customer_id, action_date) map.
    let mut ad_node_to_info: HashMap<u64, (i64, i64)> = HashMap::new();
    for i in 0..TOTAL_FK_ROWS {
        let ad_id = all_ids[i];
        let key = ad_table.parse_pk_key(&ad_id.to_string())?;
        let node_id = ad_table
            .node_id_from_pk(&key)?
            .ok_or_else(|| anyhow!("account_daily pk {ad_id} not found in index"))?;
        ad_node_to_info.insert(node_id, (all_cids[i], all_dates[i]));
    }

    // Build customer node_id -> customer_pk map.
    let mut cust_node_to_pk: HashMap<u64, i64> = HashMap::new();
    for cid in 0..N_CUSTOMERS {
        let key = customer_table.parse_pk_key(&cid.to_string())?;
        let node_id = customer_table
            .node_id_from_pk(&key)?
            .ok_or_else(|| anyhow!("customer pk {cid} not found in index"))?;
        cust_node_to_pk.insert(node_id, cid);
    }

    // -- 6. Verify reverse edges (customer → account_daily) -------------------
    let mut total_reverse: u64 = 0;
    let mut misaligned: Vec<String> = Vec::new();

    for cust_node_id in 0..customer_table.node_count() {
        let cust_pk = *cust_node_to_pk
            .get(&cust_node_id)
            .ok_or_else(|| anyhow!("customer node {cust_node_id} not in pk map"))?;
        let entries =
            extract_relation_entries(&customer_table, reverse_rel_name, cust_node_id)?;
        for (dst_node_id, sort_key) in &entries {
            total_reverse += 1;
            let (owner_cid, action_date) = match ad_node_to_info.get(dst_node_id) {
                Some(info) => *info,
                None => {
                    misaligned.push(format!(
                        "reverse: customer_pk={cust_pk} -> ad_node={dst_node_id} NOT in map"
                    ));
                    continue;
                }
            };
            // Sanity: this reverse edge should point to an account_daily
            // owned by this customer.
            if owner_cid != cust_pk {
                misaligned.push(format!(
                    "reverse: customer_pk={cust_pk} -> ad_node={dst_node_id} \
                     but that node belongs to customer {owner_cid}"
                ));
                continue;
            }
            let expected_encoded = encode_ts_i32(action_date, ANCHOR_TIME, TS_STEP);
            if *sort_key != expected_encoded {
                misaligned.push(format!(
                    "reverse: customer_pk={cust_pk}, ad_node={dst_node_id}: \
                     sort_key={sort_key:#010x} but expected={expected_encoded:#010x} \
                     (action_date={action_date})"
                ));
            }
        }
    }

    // -- 7. Report results ----------------------------------------------------
    eprintln!(
        "Reverse FK edges: {total_reverse} (expected {TOTAL_FK_ROWS})"
    );

    if !misaligned.is_empty() {
        let count = misaligned.len();
        let sample: Vec<_> = misaligned.iter().take(20).collect();
        panic!(
            "SORT KEY MISALIGNMENT detected on {count} reverse FK edges \
             (out of {total_reverse}).\n\
             First {} samples:\n{}",
            sample.len(),
            sample
                .iter()
                .map(|s| format!("  {s}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    assert!(
        total_reverse > 0,
        "expected at least some reverse FK edges"
    );
    assert_eq!(
        total_reverse, TOTAL_FK_ROWS as u64,
        "expected {TOTAL_FK_ROWS} reverse FK edges (one per account_daily row)"
    );

    Ok(())
}
