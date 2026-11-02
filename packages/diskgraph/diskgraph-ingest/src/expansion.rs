// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Add tables and relationships without replacing existing node records.
use super::*;
use crate::append::{index_key, AppendRequest};
use diskgraph_index::delta::{DeltaLog, DeltaManifest, DeltaTable, EdgeDelta, MANIFEST};
use diskgraph_index::graph::{GraphIndexReader, TableReader};
use diskgraph_index::pkmap::{self, Key as IndexKey};
use parquet::arrow::ProjectionMask;
use serde_json::Value;

fn read(root: &Path, name: &str) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(root.join(name))?)?)
}

fn write(root: &Path, name: &str, value: &impl serde::Serialize) -> Result<()> {
    let path = root.join(name);
    if path.exists() {
        fs::remove_file(&path)?;
    } // Break parent metadata hardlinks.
    fs::write(path, serde_json::to_vec(value)?)?;
    Ok(())
}

// Only used for PKs read from the exact retained source of a snapshot with no
// dropped/duplicate rows. Arbitrary FK membership is checked in an exact map.
fn known_id(table: &TableReader, appended: &HashMap<IndexKey, u64>, key: &IndexKey) -> Result<u64> {
    if let Some(id) = appended.get(key) {
        return Ok(*id);
    }
    let map = table.pk_map().context("expansion requires primary keys")?;
    let shard = pkmap::shard_for_key(key, map.shard_count());
    let local = map
        .shard(shard)
        .and_then(|s| s.lookup(key))
        .context("retained primary key missing")?;
    Ok(table.shard_id_range(shard).unwrap().0 + u64::from(local))
}

fn visit(
    table: &TableSpec,
    columns: &HashSet<String>,
    mut row: impl FnMut(&TableSpec, &mut RowAccess<'_>, &[ArrayRef], usize) -> Result<()>,
) -> Result<()> {
    let mut projected = table.clone();
    projected.columns.retain(|c| columns.contains(&c.name));
    let mut access = RowAccess::new(
        build_row_access_hints(&projected),
        RowAccessStats::new(None),
    );
    let crate::graph_config::DataSource::Parquet { paths } = &table.source else {
        return Err(anyhow!("expansion requires normalized Parquet"));
    };
    for path in paths {
        let builder = ParquetRecordBatchReaderBuilder::try_new(fs::File::open(path)?)?;
        let indices = builder
            .schema()
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(i, field)| columns.contains(field.name()).then_some(i));
        let projection = ProjectionMask::roots(builder.parquet_schema(), indices);
        for batch in builder
            .with_projection(projection)
            .with_batch_size(8192)
            .build()?
        {
            let batch = batch?;
            let cols = build_access_cols(&batch, &projected, &HashSet::new())?;
            access.set_cols(&cols);
            for i in 0..batch.num_rows() {
                access.set_row(i);
                row(&projected, &mut access, &cols, i)?;
            }
            access.set_cols(&[]);
        }
    }
    Ok(())
}

pub(super) fn expand(
    request: &AppendRequest,
    source: &GraphConfig,
    old: &EncodeState,
    original: &GraphIndexReader,
    mut metadata: DatasetMetadata,
    mut deltas: DeltaManifest,
) -> Result<()> {
    let started = std::time::Instant::now();
    ensure!(
        request.tables.is_empty() && request.replacements.is_empty(),
        "expansion cannot also append rows"
    );
    let existing: HashSet<_> = old.source.tables.iter().map(|t| t.name.clone()).collect();
    let declared = source;
    let mut source = source.clone();
    for table in source.tables.iter().filter(|t| existing.contains(&t.name)) {
        let crate::graph_config::DataSource::Parquet { paths } = &table.source else {
            return Err(anyhow!(
                "expansion requires full table encoding for non-retained sources"
            ));
        };
        let retained = request
            .parent
            .join("_incremental/sources")
            .join(&table.name);
        let expected: HashSet<_> = fs::read_dir(retained)?
            .map(|p| p?.path().canonicalize())
            .collect::<std::io::Result<_>>()?;
        let actual: HashSet<_> = paths
            .iter()
            .map(|p| Path::new(p).canonicalize())
            .collect::<std::io::Result<_>>()?;
        ensure!(
            actual == expected,
            "expansion requires exact retained sources"
        );
    }
    source.num_shards = Some(
        read(&request.parent, "_incremental/manifest.json")?["num_shards"]
            .as_u64()
            .context("missing shard count")? as usize,
    );
    let mut relationships = Vec::new();
    for table in &mut source.tables {
        ensure!(
            table.primary_key.is_some() && table.edge_table.is_none(),
            "keyless expansion requires full table encoding"
        );
        let old_table = old.source.tables.iter().find(|t| t.name == table.name);
        for fk in &table.fkeys {
            let value = serde_json::to_value(fk)?;
            if old_table.is_some_and(|t| {
                t.fkeys
                    .iter()
                    .any(|f| serde_json::to_value(f).ok() == Some(value.clone()))
            }) {
                continue;
            }
            ensure!(
                [&fk.forward, &fk.reverse]
                    .into_iter()
                    .flatten()
                    .all(|d| d.weight.is_none()),
                "weighted expansion requires full table encoding"
            );
            ensure!(
                fk.dst_table != table.name,
                "self-relation expansion requires full table encoding"
            );
            relationships.push((table.name.clone(), fk.clone()));
        }
        if let Some(previous) = old_table {
            // The SDK checks unchanged declarations and exact source hashes.
            // Keep the actual fitted column types, including inferred widths.
            table.columns.clone_from(&previous.columns);
        }
    }
    ensure!(
        existing
            .iter()
            .all(|n| source.tables.iter().any(|t| &t.name == n)),
        "expansion cannot remove tables"
    );
    let mut added_source = source.clone();
    added_source.tables.retain(|t| !existing.contains(&t.name));
    ensure!(
        !added_source.tables.is_empty(),
        "expansion requires new tables"
    );
    for table in &mut added_source.tables {
        table.fkeys.clear();
    }

    let staging = tempfile::tempdir_in(request.output.parent().unwrap())?;
    let fragment = staging.path().join("tables");
    let options = IngestOptions {
        col_stats: true,
        cat_count_strategy: col_stats::CategoryCountStrategy::Hybrid,
        incremental: Some(incremental::IncrementalConfig {
            parent: None,
            force_tables: vec![],
            require_reuse: vec![],
            reuse_encoders: true,
            source_fingerprints: None,
            snapshot_id: None,
            parent_snapshot_id: None,
        }),
        ..Default::default()
    };
    let (added_meta, added_stats) = ingest_with_stats_from_source(
        added_source,
        None,
        fragment.to_str().context("non-UTF8 expansion path")?,
        Dest::Local,
        options,
        None,
        None,
    )?;
    ensure!(
        added_stats.pk_duplicate_total == 0
            && added_stats.key_parse_failures == 0
            && added_stats.dropped_rows_counts.values().all(|&n| n == 0)
            && added_stats
                .pk_null_timestamp_drops
                .values()
                .all(|&n| n == 0),
        "filtered expansion requires full table encoding"
    );
    let added_graph = GraphIndexReader::open(&fragment.join("index_metadata.json"), &fragment)?;
    for node in &added_meta.node_types {
        for path in [
            format!("nodes/{}", node.name.to_lowercase()),
            format!("kumo_col_stats/col_stats_full/{}.json", node.name),
            format!("kumo_col_stats/col_stats_min/{}.json", node.name),
        ] {
            incremental::clone_tree(&fragment.join(&path), &request.output.join(path))?;
        }
        let reader = added_graph.table(&node.name).unwrap();
        deltas.tables.insert(
            node.name.clone(),
            DeltaTable {
                base_shards: node.shards.len(),
                shard_layouts: vec![vec![]; node.shards.len()],
                edge_logs: vec![],
                primary_key_logs: vec![],
                feature_file: None,
                edge_counts: BTreeMap::new(),
            },
        );
        ensure!(
            reader.relations().is_empty(),
            "new table unexpectedly contains relationships"
        );
    }
    incremental::clone_tree(
        &fragment.join("_incremental/encode"),
        &request.output.join("_incremental/encode"),
    )?;
    metadata.node_types.extend(added_meta.node_types.clone());
    let added_state: EncodeState =
        serde_json::from_value(read(&fragment, "_incremental/encode_state.json")?)?;
    for table in &mut source.tables {
        if let Some(added) = added_state
            .source
            .tables
            .iter()
            .find(|t| t.name == table.name)
        {
            table.columns.clone_from(&added.columns);
        }
    }
    let mut appended = BTreeMap::new();
    for (name, delta) in &deltas.tables {
        let keys = appended.entry(name.clone()).or_insert_with(HashMap::new);
        for path in &delta.primary_key_logs {
            diskgraph_index::delta::visit_primary_keys(&request.output.join(path), |key, id| {
                keys.insert(key, id);
                Ok(())
            })?;
        }
    }
    let table_reader = |name: &str| {
        original
            .table(name)
            .or_else(|| added_graph.table(name))
            .context("expansion table missing")
    };
    let mut missing = 0u64;
    let mut stats_relations = Vec::new();
    let mut stats_counts = HashMap::new();
    let mut stats_destinations = HashMap::new();
    for (name, fk) in relationships {
        let table = table_by_name(&source, &name)?;
        let target = table_by_name(&source, &fk.dst_table)?;
        let pk = table.primary_key.as_ref().unwrap();
        let dst_pk = target.primary_key.as_ref().unwrap();
        let dtype = &target.columns[column_index(target, dst_pk)?].ingested_dtype;
        let base = format!("{name}_{}_to_{}_{dst_pk}", fk.src_column, fk.dst_table);
        let mut directions = vec![(base.clone(), &name, &fk.dst_table, &fk.forward, false)];
        if fk.bidirectional {
            directions.push((
                format!("{base}_reverse"),
                &fk.dst_table,
                &name,
                &fk.reverse,
                true,
            ));
        }
        let mut source_columns: HashSet<String> =
            [pk.clone(), fk.src_column.clone()].into_iter().collect();
        let mut target_columns: HashSet<String> = [dst_pk.clone()].into_iter().collect();
        let mut times: BTreeMap<String, Vec<Option<i64>>> = BTreeMap::new();
        let mut relation_ids = Vec::new();
        for (rel_name, src, dst, direction, reverse) in &directions {
            if let Some(time) = direction.as_ref().and_then(|d| d.time.as_ref()) {
                ensure!(
                    time.table == name || time.table == fk.dst_table,
                    "expansion sort key must reference an endpoint"
                );
                if time.table == name {
                    source_columns.insert(time.column.clone());
                } else {
                    target_columns.insert(time.column.clone());
                    times.entry(time.column.clone()).or_insert_with(|| {
                        vec![None; table_reader(&fk.dst_table).unwrap().node_count() as usize]
                    });
                }
            }
            let sort = direction.as_ref().and_then(build_relation_sort_key_config);
            let relation = metadata
                .edge_relations
                .iter()
                .filter(|r| &r.src_type == *src)
                .count();
            relation_ids.push(u16::try_from(relation)?);
            metadata.edge_relations.push(EdgeRelationMetadata {
                name: rel_name.clone(),
                src_type: (*src).clone(),
                dst_type: (*dst).clone(),
                shards: vec![],
                dest_sort_key_bits: sort.as_ref().map(|s| s.width_bits),
                dest_sort_key: sort.as_ref().map(build_sort_key_metadata),
                format: EdgeStorageFormat::Binary,
                reverse_of: fk.bidirectional.then(|| {
                    if *reverse {
                        base.clone()
                    } else {
                        format!("{base}_reverse")
                    }
                }),
            });
        }
        let target_reader = table_reader(&fk.dst_table)?;
        let source_reader = table_reader(&name)?;
        let mut destinations = HashMap::new();
        visit(target, &target_columns, |spec, access, _, _| {
            let key = index_key(access, spec, dst_pk, dtype)?;
            let id = known_id(&target_reader, &appended[&fk.dst_table], &key)?;
            for (column, values) in &mut times {
                values[id as usize] = access.get_timestamp(column_index(spec, column)?)?;
            }
            destinations.insert(key, id);
            Ok(())
        })?;
        let mut edges = vec![Vec::new(); directions.len()];
        let mut reached = vec![false; target_reader.node_count() as usize];
        let mut edge_count = 0;
        visit(table, &source_columns, |spec, access, cols, row| {
            let src_key = index_key(
                access,
                spec,
                pk,
                &table.columns[column_index(table, pk)?].ingested_dtype,
            )?;
            let id = known_id(&source_reader, &appended[&name], &src_key)?;
            // Null FK is not a failed membership lookup.
            if cols[column_index(spec, &fk.src_column)?].is_null(row) {
                return Ok(());
            }
            let key = index_key(access, spec, &fk.src_column, dtype)?;
            let Some(&destination) = destinations.get(&key) else {
                missing += 1;
                return Ok(());
            };
            reached[destination as usize] = true;
            edge_count += 1;
            for (i, (_, _, _, direction, reverse)) in directions.iter().enumerate() {
                let sort_key = if let Some(time) = direction.as_ref().and_then(|d| d.time.as_ref())
                {
                    let value = if time.table == name {
                        access.get_timestamp(column_index(spec, &time.column)?)?
                    } else {
                        times[&time.column][destination as usize]
                    };
                    encode_timestamp_value(value, &time_spec_to_encoding(time))
                } else {
                    0
                };
                edges[i].push(EdgeDelta {
                    source: if *reverse { destination } else { id },
                    relation: relation_ids[i],
                    destination: if *reverse { id } else { destination },
                    sort_key,
                });
            }
            Ok(())
        })?;
        for (relation, _, _, _, _) in &directions {
            stats_counts.insert(relation.clone(), edge_count);
        }
        stats_destinations.insert(
            base.clone(),
            reached.iter().filter(|&&used| used).count() as u64,
        );
        stats_relations.push(RelationshipRuntime {
            name_forward: base.clone(),
            name_reverse: format!("{base}_reverse"),
            src_table: name.clone(),
            src_column: fk.src_column.clone(),
            dst_table: fk.dst_table.clone(),
            bidirectional: fk.bidirectional,
        });
        for ((_, owner, _, _, _), mut entries) in directions.into_iter().zip(edges) {
            if entries.is_empty() { continue; }
            let delta = deltas.tables.get_mut(owner).unwrap();
            let path = format!(
                "_incremental/edges-{}-{}.bin",
                owner.to_lowercase(),
                delta.edge_logs.len()
            );
            DeltaLog::write(&request.output.join(&path), &mut entries)?;
            delta.edge_logs.push(path);
        }
    }
    if let Some(bytes) = IngestContext::build_edge_stats_json(
        &source,
        &stats_relations,
        &stats_counts,
        &stats_destinations,
        &metadata,
    )? {
        let path = col_stats_export::EDGE_STATS_FILENAME;
        let mut stats = if request.output.join(path).is_file() {
            read(&request.output, path)?
        } else {
            serde_json::json!({})
        };
        stats.as_object_mut().unwrap().extend(
            serde_json::from_slice::<Value>(&bytes)?
                .as_object()
                .unwrap()
                .clone(),
        );
        write(&request.output, path, &stats)?;
    }
    deltas.version = 3;
    write(&request.output, "index_metadata.json", &metadata)?;
    write(&request.output, MANIFEST, &deltas)?;
    write(&request.output, "diskgraph_config.json", declared)?;
    for (path, fields) in [
        (
            "_incremental/manifest.json",
            vec!["schemas", "versions", "trimmed_strings"],
        ),
        (
            "_incremental/encode_state.json",
            vec![
                "pk_sampled_max_key_len",
                "dropped_rows_counts",
                "logical_source_rows_by_table",
                "logical_arrow_source_bytes_by_table",
            ],
        ),
        ("col_stats.json", vec![]),
        ("_incremental/col_stats.json", vec![]),
    ] {
        let mut value = read(&request.output, path)?;
        let new = read(&fragment, path)?;
        if fields.is_empty() {
            value
                .as_object_mut()
                .unwrap()
                .extend(new.as_object().unwrap().clone());
        }
        for field in fields {
            value[field]
                .as_object_mut()
                .unwrap()
                .extend(new[field].as_object().unwrap().clone());
        }
        if path.ends_with("manifest.json") {
            // Staging may be skipped only when the retained encoding includes
            // every FK and sort-key stream required by the expanded schema.
            for table in &declared.tables {
                if value["schemas"][&table.name] != incremental::schema(table, declared)? {
                    value["versions"][&table.name] = Value::Null;
                }
            }
        }
        if path.ends_with("encode_state.json") {
            value["source"] = serde_json::to_value(&source)?;
        }
        write(&request.output, path, &value)?;
    }
    write(
        &request.output,
        "_incremental/expansion_stats.json",
        &serde_json::json!({
            "added": added_stats, "edge_counts": stats_counts, "key_lookup_failures": missing, "seconds": started.elapsed().as_secs_f64(),
        }),
    )?;
    Ok(())
}
