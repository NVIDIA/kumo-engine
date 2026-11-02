// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Encode proven additions with frozen contexts and stable node IDs.
use super::*;
use crate::mapping::to_index_key;
use diskgraph_index::delta::{
    DeltaLog, DeltaManifest, DeltaTable, EdgeDelta, NodeAppender, MANIFEST,
};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_index::pkmap::Key as IndexKey;
use diskgraph_rowpack::{RowPackContext, RowPackDecodeScratch, RowPackScratch, ValueRef};
use std::fs::File;

#[derive(serde::Deserialize)]
pub struct AppendRequest {
    pub parent: PathBuf,
    pub output: PathBuf,
    pub tables: BTreeMap<String, Vec<PathBuf>>,
    #[serde(default)]
    pub replacements: BTreeMap<String, Vec<PathBuf>>,
    #[serde(default)]
    pub expansion: Option<GraphConfig>,
}

pub(super) fn index_key(
    access: &mut RowAccess<'_>,
    table: &TableSpec,
    column: &str,
    dtype: &IngestedType,
) -> Result<IndexKey> {
    let index = column_index(table, column)?;
    let mut bytes = Vec::new();
    key_from_access_into_buf(access, index, &table.columns[index], dtype, &mut bytes)
        .map_err(|e| anyhow!("append key {}.{} requires full table encoding: {e:?}", table.name, column))?;
    let key = read_key_from_buf_for_type(&mut bytes.as_slice(), dtype)?;
    Ok(to_index_key(key))
}


fn batches(paths: &[PathBuf], mut consume: impl FnMut(RecordBatch) -> Result<()>) -> Result<()> {
    for path in paths {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?
            .with_batch_size(8192)
            .build()?;
        for batch in reader {
            consume(batch?)?;
        }
    }
    Ok(())
}

// Resolve only requested keys from the compact, retained PK streams. Scanning
// these streams avoids random feature-page reads for every incoming FK. Exact
// keys and the original first-valid-row rule preserve membership semantics.
fn resolve_keys(
    request: &AppendRequest,
    state: &EncodeState,
    metadata: &DatasetMetadata,
    deltas: &DeltaManifest,
) -> Result<BTreeMap<String, HashMap<IndexKey, Option<u64>>>> {
    let mut wanted: BTreeMap<String, HashMap<IndexKey, Option<u64>>> = BTreeMap::new();
    for (name, paths) in &request.tables {
        let spec = table_by_name(&state.source, name)?;
        let pk = spec
            .primary_key
            .as_ref()
            .context("append requires a primary key")?;
        let mut columns = vec![(
            name.as_str(),
            pk.as_str(),
            &spec.columns[column_index(spec, pk)?].ingested_dtype,
        )];
        for fk in &spec.fkeys {
            let target = table_by_name(&state.source, &fk.dst_table)?;
            let key = target
                .primary_key
                .as_ref()
                .context("missing destination PK")?;
            columns.push((
                &fk.dst_table,
                &fk.src_column,
                &target.columns[column_index(target, key)?].ingested_dtype,
            ));
        }
        let mut access = RowAccess::new(build_row_access_hints(spec), RowAccessStats::new(None));
        batches(paths, |batch| {
            let cols = build_access_cols(&batch, spec, &HashSet::new())?;
            access.set_cols(&cols);
            for row in 0..batch.num_rows() {
                access.set_row(row);
                for (target, column, dtype) in &columns {
                    if cols[column_index(spec, column)?].is_null(row) {
                        continue;
                    }
                    wanted
                        .entry((*target).to_owned())
                        .or_default()
                        .entry(index_key(&mut access, spec, column, dtype)?)
                        .or_insert(None);
                }
            }
            access.set_cols(&[]);
            Ok(())
        })?;
    }
    for (name, keys) in &mut wanted {
        let spec = table_by_name(&state.source, name)?;
        let column = &spec.columns[column_index(spec, spec.primary_key.as_ref().unwrap())?];
        let node = metadata
            .node_types
            .iter()
            .find(|node| node.name == *name)
            .unwrap();
        let delta = &deltas.tables[name];
        for (shard, range) in node.shards.iter().take(delta.base_shards).enumerate() {
            let path = request
                .parent
                .join("_incremental/encode")
                .join(temp_pkey_path(name, shard));
            let mut id = range.min_id;
            read_temp_rows(&path, |row| {
                if row.last() == Some(&1) {
                    let key = to_index_key(read_key_from_buf(&mut &row[..row.len() - 1], column)?);
                    if let Some(value) = keys.get_mut(&key) {
                        value.get_or_insert(id);
                    }
                }
                id += 1;
                Ok(())
            })?;
        }
        for path in &delta.primary_key_logs {
            diskgraph_index::delta::visit_primary_keys(&request.parent.join(path), |key, id| {
                if let Some(value) = keys.get_mut(&key) {
                    value.get_or_insert(id);
                }
                Ok(())
            })?;
        }
    }
    Ok(wanted)
}

fn save_metadata(root: &Path, metadata: &DatasetMetadata, deltas: &DeltaManifest) -> Result<()> {
    // These files were deliberately excluded from hardlinking the parent.
    fs::write(
        root.join("index_metadata.json"),
        serde_json::to_vec(metadata)?,
    )?;
    fs::write(root.join(MANIFEST), serde_json::to_vec(deltas)?)?;
    Ok(())
}

pub fn append(request: &AppendRequest) -> Result<()> {
    let mut timings = BTreeMap::new();
    let mut timer = std::time::Instant::now();
    let parent = request.parent.canonicalize()?;
    ensure!(!request.output.exists(), "append output already exists");
    let config = incremental::IncrementalConfig {
        parent: Some(parent.clone()),
        force_tables: Vec::new(),
        require_reuse: Vec::new(),
        reuse_encoders: true,
        source_fingerprints: None,
        snapshot_id: None,
        parent_snapshot_id: None,
    };
    config.validate_output(&request.output)?;
    let original = GraphIndexReader::open(&parent.join("index_metadata.json"), &parent)?;
    let state: EncodeState =
        serde_json::from_slice(&fs::read(parent.join("_incremental/encode_state.json"))?)?;
    let mut metadata = original.metadata().clone();
    let mut deltas = DeltaManifest::load(&parent)?.unwrap_or_else(|| DeltaManifest {
        version: 1,
        tables: BTreeMap::new(),
    });
    for table in &metadata.node_types {
        if !deltas.tables.contains_key(&table.name) {
            let reader = original.table(&table.name).unwrap();
            deltas.tables.insert(
                table.name.clone(),
                DeltaTable {
                    base_shards: table.shards.len(),
                    shard_layouts: vec![reader.layouts().as_ref().clone(); table.shards.len()],
                    edge_logs: Vec::new(),
                    primary_key_logs: Vec::new(),
                    feature_file: None,
                    edge_counts: BTreeMap::new(),
                },
            );
        }
    }
    timings.insert("open_parent", timer.elapsed().as_secs_f64());
    timer = std::time::Instant::now();
    clone_snapshot(&parent, &request.output)?;
    if let Some(source) = &request.expansion {
        return crate::expansion::expand(request, source, &state, &original, metadata, deltas);
    }
    timings.insert("clone", timer.elapsed().as_secs_f64());
    timer = std::time::Instant::now();
    replace_features(request, &state, &original, &mut deltas)?;
    timings.insert("replace_features", timer.elapsed().as_secs_f64());
    timer = std::time::Instant::now();
    let mut resolved = resolve_keys(request, &state, &metadata, &deltas)?;
    timings.insert("resolve_keys", timer.elapsed().as_secs_f64());
    timer = std::time::Instant::now();
    let mut first_ids = BTreeMap::new();
    for (name, paths) in &request.tables {
        let spec = table_by_name(&state.source, name)?;
        let reader = original
            .table(name)
            .context("append table missing in parent")?;
        let pk = spec
            .primary_key
            .as_ref()
            .context("append requires a primary key")?;
        let pk_dtype = &spec.columns[column_index(spec, pk)?].ingested_dtype;
        let node = metadata
            .node_types
            .iter_mut()
            .find(|n| n.name == *name)
            .unwrap();
        let first = node.max_id.checked_add(1).context("node ID overflow")?;
        first_ids.insert(name.clone(), first);
        let shard = node.shards.len();
        let directory = request
            .output
            .join(format!("nodes/{}/shard-{shard:05}", name.to_lowercase()));
        let mut writer = NodeAppender::new(&directory, reader.relations().len())?;
        let ctx =
            RowPackContext::from_bytes(&fs::read(parent.join(node_context_output_path(name)))?)?;
        let encoder = RowPackEncoder::new(ctx)?;
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut access = RowAccess::new(build_row_access_hints(spec), RowAccessStats::new(None));
        let mut features = Vec::new();
        let mut records = Vec::new();
        let mut time_bounds: BTreeMap<usize, (i64, i64)> = spec.columns.iter().enumerate()
            .filter(|(_, c)| matches!(c.ingested_dtype, IngestedType::Timestamp | IngestedType::Date))
            .map(|(i, _)| (i, (i64::MAX, i64::MIN))).collect();
        let mut id = first;
        batches(paths, |batch| {
            let cols = build_access_cols(&batch, spec, &HashSet::new())?;
            access.set_cols(&cols);
            for row in 0..batch.num_rows() {
                access.set_row(row);
                let key = index_key(&mut access, spec, pk, pk_dtype)?;
                ensure!(
                    resolved[name].get(&key).copied().flatten().is_none(),
                    "append contains an existing or duplicate primary key in {name}"
                );
                if let Some(time) = &spec.create_time {
                    let index = column_index(spec, time)?;
                    ensure!(
                        access.get_timestamp(index)?.is_some(),
                        "append has a null timestamp"
                    );
                }
                for (&column, (minimum, maximum)) in &mut time_bounds {
                    if let Some(value) = access.get_timestamp(column)? {
                        *minimum = (*minimum).min(value);
                        *maximum = (*maximum).max(value);
                    }
                }
                encoder.encode_row_from_access(&mut access, &mut features, &mut scratch)?;
                writer.push(&features)?;
                records.push((bincode::serialize(&key)?, id));
                resolved.get_mut(name).unwrap().insert(key, Some(id));
                id = id.checked_add(1).context("node ID overflow")?;
            }
            access.set_cols(&[]);
            Ok(())
        })?;
        let count = writer.finish(&directory)?;
        ensure!(count > 0, "append table {name} has no new rows");
        node.max_id = id - 1;
        node.shards.push(NodeShardMetadata {
            min_id: first,
            max_id: id - 1,
            node_data_paths: None,
        });
        let delta = deltas.tables.get_mut(name).unwrap();
        delta.shard_layouts.push(reader.layouts().as_ref().clone());
        let key_path = format!("_incremental/keys-{}-{shard}.bin", name.to_lowercase());
        diskgraph_index::delta::write_primary_keys(&request.output.join(&key_path), records)?;
        delta.primary_key_logs.push(key_path);
        refresh_time_bounds(&request.output, spec, &time_bounds, count)?;
    }
    timings.insert("encode", timer.elapsed().as_secs_f64());
    timer = std::time::Instant::now();
    save_metadata(&request.output, &metadata, &deltas)?;
    let combined =
        GraphIndexReader::open(&request.output.join("index_metadata.json"), &request.output)?;
    timings.insert("open_combined", timer.elapsed().as_secs_f64());
    timer = std::time::Instant::now();
    let mut edges: BTreeMap<String, Vec<EdgeDelta>> = BTreeMap::new();
    let mut missing_keys = 0u64;
    let mut timestamps = BTreeMap::new();
    for (name, paths) in &request.tables {
        let spec = table_by_name(&state.source, name)?;
        let mut id = first_ids[name];
        let mut access = RowAccess::new(build_row_access_hints(spec), RowAccessStats::new(None));
        batches(paths, |batch| {
            let cols = build_access_cols(&batch, spec, &HashSet::new())?;
            access.set_cols(&cols);
            for row in 0..batch.num_rows() {
                access.set_row(row);
                for fk in &spec.fkeys {
                    let index = column_index(spec, &fk.src_column)?;
                    if cols[index].is_null(row) {
                        continue;
                    }
                    let dst_spec = table_by_name(&state.source, &fk.dst_table)?;
                    let dst_pk = dst_spec
                        .primary_key
                        .as_ref()
                        .context("missing destination PK")?;
                    let dtype = &dst_spec.columns[column_index(dst_spec, dst_pk)?].ingested_dtype;
                    let key = index_key(&mut access, spec, &fk.src_column, dtype)?;
                    let destination = resolved[&fk.dst_table].get(&key).copied().flatten();
                    let Some(destination) = destination else {
                        missing_keys += 1;
                        continue;
                    };
                    let forward =
                        format!("{}_{}_to_{}_{}", name, fk.src_column, fk.dst_table, dst_pk);
                    for (relation_name, src_table, src, dst) in [
                        (forward.clone(), name.as_str(), id, destination),
                        (
                            format!("{forward}_reverse"),
                            fk.dst_table.as_str(),
                            destination,
                            id,
                        ),
                    ] {
                        let table = combined.table(src_table).unwrap();
                        let Some((relation, definition)) = table
                            .relations()
                            .iter()
                            .enumerate()
                            .find(|(_, rel)| rel.name == relation_name)
                        else {
                            continue;
                        };
                        let sort_key = sort_key(
                            &combined,
                            &parent,
                            &deltas,
                            &mut timestamps,
                            definition,
                            src,
                            dst,
                        )?;
                        edges
                            .entry(src_table.to_owned())
                            .or_default()
                            .push(EdgeDelta {
                                source: src,
                                relation: relation as u16,
                                destination: dst,
                                sort_key,
                            });
                    }
                }
                id += 1;
            }
            access.set_cols(&[]);
            Ok(())
        })?;
    }
    drop(combined);
    for (name, mut entries) in edges {
        let delta = deltas.tables.get_mut(&name).unwrap();
        let relations = original.table(&name).unwrap();
        for entry in &entries {
            *delta
                .edge_counts
                .entry(relations.relations()[entry.relation as usize].name.clone())
                .or_default() += 1;
        }
        let path = format!(
            "_incremental/edges-{}-{}.bin",
            name.to_lowercase(),
            delta.edge_logs.len()
        );
        DeltaLog::write(&request.output.join(&path), &mut entries)?;
        delta.edge_logs.push(path);
    }
    save_metadata(&request.output, &metadata, &deltas)?;
    timings.insert("edges", timer.elapsed().as_secs_f64());
    fs::write(
        request.output.join("_incremental/append_stats.json"),
        serde_json::to_vec(&serde_json::json!({"key_lookup_failures": missing_keys}))?,
    )?;
    fs::write(
        request.output.join("_incremental/append_timings.json"),
        serde_json::to_vec(&timings)?,
    )?;
    Ok(())
}

// The SDK proves unchanged structure against retained normalized source rows.
// Only the edited rows reach this encoder; the original key maps and edges stay intact.
fn replace_features(request: &AppendRequest, state: &EncodeState,
                    original: &GraphIndexReader, deltas: &mut DeltaManifest) -> Result<()> {
    for (name, paths) in &request.replacements {
        let spec = table_by_name(&state.source, name)?;
        let reader = original.table(name).context("replacement table missing")?;
        let pk = spec.primary_key.as_ref().context("replacement requires primary key")?;
        let dtype = &spec.columns[column_index(spec, pk)?].ingested_dtype;
        let ctx = RowPackContext::from_bytes(&fs::read(request.parent.join(node_context_output_path(name)))?)?;
        let encoder = RowPackEncoder::new(ctx)?;
        let mut scratch = RowPackScratch::new(encoder.context());
        let mut access = RowAccess::new(build_row_access_hints(spec), RowAccessStats::new(None));
        let delta = deltas.tables.get_mut(name).unwrap();
        let mut updates = diskgraph_index::delta::load_features(&request.parent, delta)?;
        let mut seen = HashSet::new();
        let mut features = Vec::new();
        batches(paths, |batch| {
            let cols = build_access_cols(&batch, spec, &HashSet::new())?;
            access.set_cols(&cols);
            for row in 0..batch.num_rows() {
                access.set_row(row);
                let key = index_key(&mut access, spec, pk, dtype)?;
                let id = reader.node_id_from_pk(&key)?.context("replacement key missing")?;
                ensure!(seen.insert(id), "duplicate replacement key");
                encoder.encode_row_from_access(&mut access, &mut features, &mut scratch)?;
                updates.insert(id, features.clone());
            }
            access.set_cols(&[]);
            Ok(())
        })?;
        let path = format!("_incremental/features-{}.bin", name.to_lowercase());
        let target = request.output.join(&path);
        if target.exists() { fs::remove_file(&target)?; }
        fs::write(target, bincode::serialize(&updates)?)?;
        delta.feature_file = Some(path);
        deltas.version = deltas.version.max(2);
    }
    Ok(())
}

fn sort_key(
    graph: &GraphIndexReader,
    parent: &Path,
    deltas: &DeltaManifest,
    timestamps: &mut BTreeMap<String, Vec<SortKeyComponentStore>>,
    relation: &EdgeRelationMetadata,
    src: u64,
    dst: u64,
) -> Result<u64> {
    let Some(sort) = &relation.dest_sort_key else {
        return Ok(0);
    };
    ensure!(
        sort.weight.is_none(),
        "weighted append requires full table encoding"
    );
    let Some(time) = &sort.timestamp else {
        return Ok(0);
    };
    let node = if time.table == relation.src_type {
        src
    } else {
        dst
    };
    let table = graph.table(&time.table).context("sort-key table missing")?;
    let (shard, local) = table
        .shard_for_node(node)
        .context("sort-key node missing")?;
    let base_shards = deltas.tables[&time.table].base_shards;
    if shard < base_shards {
        if !timestamps.contains_key(&relation.name) {
            // The retained dense timestamp columns avoid random reads of old
            // feature/adjacency pages. Their encoding is already frozen.
            let width = match time.encoding.kind {
                EdgeSortKeyTimestampKind::Int8 => 1,
                EdgeSortKeyTimestampKind::Int16 => 2,
                EdgeSortKeyTimestampKind::Int32 => 4,
            };
            let root = parent.join("_incremental/encode");
            let retained = root.join(temp_sort_key_component_path(
                &relation.name, SortKeyComponentKind::Timestamp, shard)).is_file();
            timestamps.insert(
                relation.name.clone(),
                if retained { load_sort_key_component_shards(
                    &root,
                    &relation.name,
                    SortKeyComponentKind::Timestamp,
                    base_shards,
                    width,
                    false,
                )? } else { Vec::new() },
            );
        }
        if let Some(values) = timestamps[&relation.name].get(shard) {
            return values.get(local);
        }
    }
    let view = table.shard(shard).unwrap().features_for_node(local)?;
    let data = view.node_data()?.context("sort-key node has no features")?;
    let decoder = table
        .node_data_decoder()
        .context("sort-key decoder missing")?;
    let column = decoder
        .context()
        .columns
        .iter()
        .position(|c| c.name == time.column)
        .context("sort-key column missing")?;
    let mut scratch = RowPackDecodeScratch::default();
    let timestamp = match decoder.decode_column_value(data, column, &mut scratch)? {
        ValueRef::Timestamp(v) | ValueRef::Int64(v) => v,
        ValueRef::Date(v) => i64::from(v) * 86400,
        _ => return Err(anyhow!("append timestamp missing or invalid")),
    };
    let bits = match time.encoding.kind {
        EdgeSortKeyTimestampKind::Int8 => 8,
        EdgeSortKeyTimestampKind::Int16 => 16,
        EdgeSortKeyTimestampKind::Int32 => 32,
    };
    let diff = timestamp
        .checked_sub(time.encoding.anchor)
        .context("timestamp overflow")?;
    let step = time.encoding.step.max(1);
    let value = diff.div_euclid(step) + i64::from(diff.rem_euclid(step) != 0);
    ensure!(
        value > -(1i64 << (bits - 1)) && value < (1i64 << (bits - 1)),
        "append timestamp exceeds frozen sort-key range"
    );
    Ok(ordered_signed_to_u64(value, bits))
}

// Replace linked statistics files; fitted rowpack contexts remain immutable.
fn refresh_time_bounds(
    root: &Path,
    table: &TableSpec,
    bounds: &BTreeMap<usize, (i64, i64)>,
    rows: u64,
) -> Result<()> {
    if bounds.is_empty() {
        return Ok(());
    }
    for kind in ["full", "min"] {
        let path = root.join(format!("kumo_col_stats/col_stats_{kind}/{}.json", table.name));
        let mut stats: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        for (&column, &(minimum, maximum)) in bounds {
            for (metric, incoming) in [("MIN", minimum), ("MAX", maximum)] {
                if incoming == i64::MAX || incoming == i64::MIN {
                    continue;
                }
                let key = format!("{}__kumo.timestamp__{metric}__False", table.columns[column].name);
                let Some(stat) = stats.get_mut(&key) else { continue; };
                let prior = stat["value"].as_str().and_then(|value| {
                    chrono::NaiveDateTime::parse_from_str(
                        value.trim_start_matches("kumo_timestamp_"), "%Y-%m-%d %H:%M:%S",
                    ).ok()
                }).map(|value| value.and_utc().timestamp()).unwrap_or(incoming);
                let value = if metric == "MIN" { prior.min(incoming) } else { prior.max(incoming) };
                stat["value"] = serde_json::json!(col_stats::format_kumo_timestamp(value));
                stat["num_entries"] = serde_json::json!(stat["num_entries"].as_u64().unwrap_or(0) + rows);
            }
        }
        fs::remove_file(&path)?;
        fs::write(path, serde_json::to_vec(&stats)?)?;
    }
    Ok(())
}

fn clone_snapshot(parent: &Path, output: &Path) -> Result<()> {
    // New mutable metadata is copied, never linked. Base/cache binaries stay
    // immutable. The original cache versions remain unchanged: a later full
    // table rebuild must not mistake that cache for the appended history.
    fs::create_dir_all(output)?;
    for entry in fs::read_dir(parent)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == "index_metadata.json" || name == "_SUCCESS" || name == "_incremental" {
            continue;
        }
        if name == "ingest_stats.json" || name == "_MANIFEST.json" {
            fs::copy(entry.path(), output.join(name))?;
        } else {
            incremental::clone_tree(&entry.path(), &output.join(name))?;
        }
    }
    fs::create_dir_all(output.join("_incremental"))?;
    for entry in fs::read_dir(parent.join("_incremental"))? {
        let entry = entry?;
        if entry.file_name() == "deltas.json" {
            continue;
        }
        let destination = output.join("_incremental").join(entry.file_name());
        if matches!(
            entry.file_name().to_str(),
            Some(
                "update.json"
                    | "source_fingerprints.json"
                    | "append_timings.json"
                    | "append_stats.json"
            )
        ) {
            fs::copy(entry.path(), destination)?;
        } else {
            incremental::clone_tree(&entry.path(), &destination)?;
        }
    }
    Ok(())
}

/// Materialize merged adjacency into a new snapshot, preserving node IDs and
/// frozen feature/PK encodings. The parent and its readers remain untouched.
pub fn compact(parent: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "compaction output already exists");
    incremental::IncrementalConfig {
        parent: Some(parent.to_path_buf()),
        force_tables: Vec::new(),
        require_reuse: Vec::new(),
        reuse_encoders: true,
        source_fingerprints: None,
        snapshot_id: None,
        parent_snapshot_id: None,
    }
    .validate_output(output)?;
    let reader = GraphIndexReader::open(&parent.join("index_metadata.json"), parent)?;
    let mut deltas = DeltaManifest::load(parent)?.context("missing compaction manifest")?;
    clone_snapshot(parent, output)?;
    let mut metadata = reader.metadata().clone();
    for node in &mut metadata.node_types {
        let delta = deltas.tables.get_mut(&node.name).unwrap();
        let merge_shards = node.shards.len() > delta.base_shards + 1;
        let merge_keys = delta.primary_key_logs.len() > 1
            || delta
                .primary_key_logs
                .iter()
                .any(|path| !parent.join(path).with_extension("idx").exists());
        if delta.edge_logs.is_empty() && !merge_shards && !merge_keys && delta.feature_file.is_none() {
            continue;
        }
        let table = reader.table(&node.name).unwrap();
        let layouts = table.layouts();
        let old_shards = node.shards.clone();
        if merge_shards {
            node.shards.truncate(delta.base_shards);
            node.shards.push(NodeShardMetadata {
                min_id: old_shards[delta.base_shards].min_id,
                max_id: old_shards.last().unwrap().max_id,
                node_data_paths: None,
            });
        }
        let mut shard_layouts = delta.shard_layouts[..node.shards.len()].to_vec();
        for (shard_idx, range) in node.shards.iter().enumerate() {
            if delta.edge_logs.is_empty() && delta.feature_file.is_none() && !(merge_shards && shard_idx == delta.base_shards) {
                continue;
            }
            let directory = output.join(format!(
                "nodes/{}/shard-{shard_idx:05}",
                node.name.to_lowercase()
            ));
            for name in [
                diskgraph_index::layout::EDGES_FILENAME,
                diskgraph_index::layout::OFFSETS_FILENAME,
            ] {
                fs::remove_file(directory.join(name))?;
            }
            let mut writer = NodeAppender::new(&directory, table.relations().len())?;
            if range.min_id <= range.max_id {
                for id in range.min_id..=range.max_id {
                    let view = diskgraph_index::delta::merge(
                        table.edges_for_node(id)?,
                        &[],
                        layouts.clone(),
                    )?;
                    writer.push_view(&view)?;
                }
            }
            writer.finish(&directory)?;
            shard_layouts[shard_idx] = layouts.as_ref().clone();
        }
        for shard_idx in node.shards.len()..old_shards.len() {
            fs::remove_dir_all(output.join(format!(
                "nodes/{}/shard-{shard_idx:05}",
                node.name.to_lowercase()
            )))?;
        }
        if merge_keys {
            let mut keys = Vec::new();
            for path in delta.primary_key_logs.drain(..) {
                diskgraph_index::delta::visit_primary_keys(&parent.join(&path), |key, id| {
                    keys.push((bincode::serialize(&key)?, id));
                    Ok(())
                })?;
                let path = output.join(path);
                fs::remove_file(&path)?;
                if path.with_extension("idx").exists() {
                    fs::remove_file(path.with_extension("idx"))?;
                }
            }
            let path = format!(
                "_incremental/keys-{}-compacted.bin",
                node.name.to_lowercase()
            );
            diskgraph_index::delta::write_primary_keys(&output.join(&path), keys)?;
            delta.primary_key_logs.push(path);
        }
        for path in delta.edge_logs.drain(..) {
            fs::remove_file(output.join(path))?;
        }
        if let Some(path) = delta.feature_file.take() { fs::remove_file(output.join(path))?; }
        delta.shard_layouts = shard_layouts;
        delta.edge_counts.clear();
    }
    save_metadata(output, &metadata, &deltas)?;
    Ok(())
}
