// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Immutable Encode outputs. Reuse happens before producers open source files.

use super::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

const CACHE: &str = "_incremental";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IncrementalConfig {
    pub parent: Option<PathBuf>,
    #[serde(default)]
    pub force_tables: Vec<String>,
    #[serde(default)]
    pub require_reuse: Vec<String>,
    #[serde(default = "reuse_default")]
    pub reuse_encoders: bool,
    #[serde(default)]
    pub source_fingerprints: Option<Value>,
    #[serde(default)]
    pub snapshot_id: Option<String>,
    #[serde(default)]
    pub parent_snapshot_id: Option<String>,
}

impl IncrementalConfig {
    pub fn validate_output(&self, output: &Path) -> Result<()> {
        let Some(parent) = &self.parent else {
            return Ok(());
        };
        let parent = parent.canonicalize()?;
        let output = std::env::current_dir()?.join(output);
        if output.exists() {
            let output = output.canonicalize()?;
            ensure!(
                !output.starts_with(&parent) && !parent.starts_with(&output),
                "incremental output and parent must not overlap"
            );
        } else {
            let ancestor = output
                .ancestors()
                .find(|p| p.exists())
                .context("incremental output has no existing ancestor")?;
            ensure!(
                !ancestor.canonicalize()?.starts_with(&parent),
                "incremental output must not be inside its parent"
            );
        }
        Ok(())
    }
}

fn reuse_default() -> bool {
    true
}

#[derive(Serialize, Deserialize)]
pub(crate) struct CacheManifest {
    version: u32,
    num_shards: usize,
    options: Value,
    schemas: BTreeMap<String, Value>,
    versions: BTreeMap<String, Option<String>>,
    #[serde(default)]
    pub(super) trimmed_strings: BTreeMap<String, u64>,
    #[serde(default)]
    reused_tables: Vec<String>,
    #[serde(default)]
    inherited_encoders: Vec<String>,
}

pub(super) fn schema(table: &TableSpec, source: &GraphConfig) -> Result<Value> {
    let mut value = serde_json::to_value(table)?;
    for key in ["source", "source_size", "data_version"] {
        value.as_object_mut().unwrap().remove(key);
    }
    // Encoding emits FK keys and per-relation sort-key streams as well as
    // features. A changed incoming relation can require new sort-key output.
    let incoming: Vec<_> = source
        .tables
        .iter()
        .flat_map(|other| {
            other
                .fkeys
                .iter()
                .filter(|fk| {
                    fk.dst_table == table.name
                        && [&fk.forward, &fk.reverse]
                            .into_iter()
                            .flatten()
                            .any(|direction| {
                                direction
                                    .time
                                    .as_ref()
                                    .is_some_and(|time| time.table == table.name)
                                    || direction
                                        .weight
                                        .as_ref()
                                        .is_some_and(|weight| weight.table == table.name)
                            })
                })
                .map(move |fk| serde_json::json!({"table": other.name, "key": fk}))
        })
        .collect();
    Ok(serde_json::json!({"table": value, "incoming": incoming, "rowpack": source.rowpack}))
}

/// Never hardlink bitmaps: PkMap clears bits in place. All other Encode files
/// are immutable after the writer pools drain; downstream phases only read
/// or unlink them. Replacing a destination link never changes its source.
pub(super) fn clone_tree(source: &Path, destination: &Path) -> Result<()> {
    if source.is_dir() {
        fs::create_dir_all(destination)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            clone_tree(&entry.path(), &destination.join(entry.file_name()))?;
        }
    } else {
        fs::create_dir_all(destination.parent().context("cache path has no parent")?)?;
        if destination.exists() {
            fs::remove_file(destination)?;
        }
        if source.file_name().is_some_and(|name| name == "bitmap.bin")
            || fs::hard_link(source, destination).is_err()
        {
            fs::copy(source, destination)?;
        }
    }
    Ok(())
}

impl IngestContext {
    fn encode_paths(&self, name: &str) -> Vec<String> {
        let mut paths = vec![
            format!("_tmp/nodes/{}", name.to_lowercase()),
            format!("_tmp/fkeys/{}", name.to_lowercase()),
            format!("_tmp/fkeys_dst/{}", name.to_lowercase()),
            node_context_output_path(name),
            format!("kumo_col_stats/col_stats_full/{name}.json"),
            format!("kumo_col_stats/col_stats_min/{name}.json"),
        ];
        for rel in &self.relation_defs {
            if let Some(sort) = &rel.sort_key {
                for (kind, owner) in [
                    (
                        "timestamp",
                        sort.timestamp.as_ref().map(|s| s.table.as_str()),
                    ),
                    ("weight", sort.weight.as_ref().map(|s| s.table.as_str())),
                ] {
                    if owner == Some(name) {
                        paths.push(format!("_tmp/sort_keys/{}/{kind}", rel.name.to_lowercase()));
                    }
                }
            }
        }
        paths
    }

    pub(super) fn prepare_incremental(&mut self) -> Result<()> {
        let Some(config) = self.incremental.clone() else {
            return Ok(());
        };
        ensure!(
            self.stage_sidecars.is_empty()
                && self.resolve_columns.is_empty()
                && self.source.tables.iter().all(|t| t.edge_table.is_none()),
            "incremental cache supports base graph node tables only"
        );
        let mut manifest = CacheManifest {
            version: 1,
            num_shards: self.num_shards,
            options: serde_json::json!({"col_stats": self.col_stats,
                "category_counts": format!("{:?}", self.cat_count_strategy),
                "row_subsample": self.row_subsample,
                "edge_format": self.source.edge_format,
                "coercion": self.source.coercion_mode}),
            schemas: self
                .source
                .tables
                .iter()
                .map(|t| Ok((t.name.clone(), schema(t, &self.source)?)))
                .collect::<Result<_>>()?,
            versions: self
                .source
                .tables
                .iter()
                .map(|t| (t.name.clone(), t.data_version.clone()))
                .collect(),
            trimmed_strings: BTreeMap::new(),
            reused_tables: Vec::new(),
            inherited_encoders: Vec::new(),
        };
        for name in &config.force_tables {
            ensure!(
                manifest.schemas.contains_key(name),
                "unknown forced table {name}"
            );
        }
        if let Some(parent) = &config.parent {
            let root = require_local_output(&self.target)?;
            ensure!(
                parent.canonicalize()? != root.canonicalize()?,
                "incremental output must differ from its parent"
            );
            ensure!(
                parent.join("index_metadata.json").is_file(),
                "parent index is incomplete"
            );
            let cache = parent.join(CACHE);
            let old: CacheManifest =
                serde_json::from_slice(&fs::read(cache.join("manifest.json"))?)?;
            ensure!(old.version == 1, "unsupported incremental cache version");
            let state: EncodeState =
                serde_json::from_slice(&fs::read(cache.join("encode_state.json"))?)?;
            for table in self.source.tables.clone() {
                let previous_schema = old.schemas.get(&table.name);
                let current_schema = &manifest.schemas[&table.name];
                let same_schema = previous_schema == Some(current_schema);
                let same_encoder = previous_schema.is_some_and(|previous| {
                    previous["table"]["columns"] == current_schema["table"]["columns"]
                        && previous["table"]["rowpack"] == current_schema["table"]["rowpack"]
                        && previous["rowpack"] == current_schema["rowpack"]
                });
                if !same_schema && !(config.reuse_encoders && same_encoder) {
                    if config.reuse_encoders && old.schemas.contains_key(&table.name) {
                        tracing::warn!(table = %table.name,
                            "encoding schema changed; fitting a new encoder context");
                    }
                    continue;
                }
                if config.reuse_encoders && same_encoder {
                    let bytes = fs::read(parent.join(node_context_output_path(&table.name)))?;
                    self.inherited_encoders.insert(table.name.clone(), bytes);
                }
                if same_schema
                    && old.num_shards == manifest.num_shards
                    && old.options == manifest.options
                    && table.data_version.is_some()
                    && old.versions.get(&table.name) == manifest.versions.get(&table.name)
                    && !config.force_tables.contains(&table.name)
                {
                    for path in self.encode_paths(&table.name) {
                        let source = cache.join("encode").join(&path);
                        if source.exists() {
                            clone_tree(&source, &root.join(&path))?;
                        }
                    }
                    self.reused_tables.insert(table.name.clone());
                }
                if config.reuse_encoders || self.reused_tables.contains(&table.name) {
                    let previous = table_by_name(&state.source, &table.name)?;
                    self.source
                        .tables
                        .iter_mut()
                        .find(|t| t.name == table.name)
                        .unwrap()
                        .columns
                        .clone_from(&previous.columns);
                }
            }
        }
        ensure!(config.require_reuse.iter().all(|name| self.reused_tables.contains(name)),
            "unstaged table requires encoding");
        manifest.reused_tables = self.reused_tables.iter().cloned().collect();
        manifest.reused_tables.sort();
        manifest.inherited_encoders = self.inherited_encoders.keys().cloned().collect();
        manifest.inherited_encoders.sort();
        self.cache_manifest = Some(manifest);
        for name in &self.reused_tables {
            self.table_contexts
                .insert(name.clone(), node_context_output_path(name));
        }
        Ok(())
    }

    pub(super) fn restore_incremental_plan(&mut self) -> Result<()> {
        if self.incremental.is_none() {
            return Ok(());
        }
        let root = require_local_output(&self.target)?;
        let manifest: CacheManifest =
            serde_json::from_slice(&fs::read(root.join(CACHE).join("manifest.json"))?)?;
        self.reused_tables = manifest.reused_tables.iter().cloned().collect();
        for name in &manifest.inherited_encoders {
            self.inherited_encoders.insert(
                name.clone(),
                fs::read(root.join(node_context_output_path(name)))?,
            );
        }
        self.cache_manifest = Some(manifest);
        Ok(())
    }

    pub(super) fn restore_incremental_counts(&mut self) -> Result<()> {
        if self.reused_tables.is_empty() {
            return Ok(());
        }
        let parent = self
            .incremental
            .as_ref()
            .and_then(|c| c.parent.as_ref())
            .unwrap();
        let cache = parent.join(CACHE);
        let old: EncodeState = serde_json::from_slice(&fs::read(cache.join("encode_state.json"))?)?;
        let old_manifest: CacheManifest =
            serde_json::from_slice(&fs::read(cache.join("manifest.json"))?)?;
        for name in &self.reused_tables {
            let trimmed = old_manifest.trimmed_strings.get(name).copied().unwrap_or(0);
            self.trimmed_strings.fetch_add(trimmed, Ordering::Relaxed);
            self.cache_manifest
                .as_mut()
                .unwrap()
                .trimmed_strings
                .insert(name.clone(), trimmed);
            if let Some(value) = old.pk_sampled_max_key_len.get(name) {
                self.pk_sampled_max_key_len.insert(name.clone(), *value);
            }
            if let Some(value) = old.dropped_rows_counts.get(name) {
                self.dropped_rows_counts.insert(name.clone(), *value);
            }
            if let Some(value) = old.logical_source_rows_by_table.get(name) {
                self.logical_source_rows_by_table
                    .insert(name.clone(), *value);
                self.logical_source_rows += value;
            }
            if let Some(value) = old.logical_arrow_source_bytes_by_table.get(name) {
                self.logical_arrow_source_bytes_by_table
                    .insert(name.clone(), *value);
                self.logical_arrow_source_bytes += value;
            }
            for (context, counts) in &old.key_failures_by_context {
                if context.starts_with(&format!("pkey:{name}."))
                    || context.starts_with(&format!("fkey:{name}."))
                {
                    self.key_parse_failures += counts.parse_failures;
                    self.key_lookup_failures += counts.lookup_failures;
                    self.key_membership_failures += counts.membership_failures;
                    self.key_failures_by_context
                        .insert(context.clone(), counts.clone());
                }
            }
        }
        let root = require_local_output(&self.target)?.to_path_buf();
        if self.col_stats {
            let path = root.join("col_stats.json");
            let mut current: BTreeMap<String, Value> = if path.exists() {
                serde_json::from_slice(&fs::read(&path)?)?
            } else {
                BTreeMap::new()
            };
            let old: BTreeMap<String, Value> =
                serde_json::from_slice(&fs::read(cache.join("col_stats.json"))?)?;
            for (name, value) in old {
                if self.reused_tables.contains(&name) {
                    current.insert(name, value);
                }
            }
            self.target
                .write_bytes("col_stats.json", serde_json::to_vec_pretty(&current)?)?;
        }
        self.repopulate_node_stats_from_meta()?;
        Ok(())
    }

    pub(super) fn save_encode_cache(&self) -> Result<()> {
        let Some(manifest) = &self.cache_manifest else {
            return Ok(());
        };
        let root = require_local_output(&self.target)?;
        let cache = root.join(CACHE);
        fs::create_dir_all(&cache)?;
        for table in &self.source.tables {
            for path in self.encode_paths(&table.name) {
                if root.join(&path).exists() {
                    clone_tree(&root.join(&path), &cache.join("encode").join(&path))?;
                }
            }
        }
        fs::copy(
            root.join(ENCODE_STATE_PATH),
            cache.join("encode_state.json"),
        )?;
        if self.col_stats {
            clone_tree(&root.join("col_stats.json"), &cache.join("col_stats.json"))?;
        }
        fs::write(
            cache.join("manifest.json"),
            serde_json::to_vec_pretty(manifest)?,
        )?;
        if let Some(fingerprints) = self
            .incremental
            .as_ref()
            .and_then(|c| c.source_fingerprints.as_ref())
        {
            let mut fingerprints = fingerprints.clone();
            for (name, count) in &self.logical_source_rows_by_table {
                if let Some(table) = fingerprints.get_mut(name) {
                    table["row_count"] = serde_json::json!(count);
                }
            }
            fs::write(
                cache.join("source_fingerprints.json"),
                serde_json::to_vec(&fingerprints)?,
            )?;
        }
        Ok(())
    }
}

impl IngestContext {
    pub(super) fn restore_cached_edges(&mut self) -> Result<()> {
        let Some(parent) = self.incremental.as_ref().and_then(|c| c.parent.clone()) else {
            return Ok(());
        };
        let root = require_local_output(&self.target)?;
        let cache = parent.join(CACHE);
        let before: ResumeStatsState =
            serde_json::from_slice(&fs::read(cache.join("pk_map_state.json"))?)?;
        let after: ResumeStatsState =
            serde_json::from_slice(&fs::read(cache.join("edges_map_state.json"))?)?;
        for rel in &self.relationships {
            if !self.reused_tables.contains(&rel.src_table)
                || !self.reused_tables.contains(&rel.dst_table)
            {
                continue;
            }
            for name in [&rel.name_forward, &rel.name_reverse] {
                let source = cache.join("edges").join(name.to_lowercase());
                if source.exists() {
                    clone_tree(
                        &source,
                        &root.join("_tmp/edges_map").join(name.to_lowercase()),
                    )?;
                    self.reused_relations.insert(name.clone());
                }
                for (current, old) in [
                    (&mut self.edge_counts, &after.edge_counts),
                    (
                        &mut self.edge_dst_nonzero_counts,
                        &after.edge_dst_nonzero_counts,
                    ),
                    (&mut self.pk_duplicate_drops, &after.pk_duplicate_drops),
                    (
                        &mut self.edge_null_timestamp_drops,
                        &after.edge_null_timestamp_drops,
                    ),
                ] {
                    if let Some(value) = old.get(name) {
                        current.insert(name.clone(), *value);
                    }
                }
            }
            for context in [
                format!("fkey:{}.{}", rel.src_table, rel.src_column),
                format!("dst_lookup:{}", rel.name_forward),
            ] {
                if let Some(counts) = after.key_failures_by_context.get(&context) {
                    let empty = KeyFailureCounts::default();
                    let previous = before
                        .key_failures_by_context
                        .get(&context)
                        .unwrap_or(&empty);
                    let delta = KeyFailureCounts {
                        parse_failures: counts
                            .parse_failures
                            .saturating_sub(previous.parse_failures),
                        lookup_failures: counts
                            .lookup_failures
                            .saturating_sub(previous.lookup_failures),
                        membership_failures: counts
                            .membership_failures
                            .saturating_sub(previous.membership_failures),
                        non_null_parse_failures: counts
                            .non_null_parse_failures
                            .saturating_sub(previous.non_null_parse_failures),
                    };
                    self.key_parse_failures += delta.parse_failures;
                    self.key_lookup_failures += delta.lookup_failures;
                    self.key_membership_failures += delta.membership_failures;
                    self.key_failures_by_context
                        .entry(context)
                        .or_default()
                        .merge(&delta);
                }
            }
        }
        Ok(())
    }

    pub(super) fn save_edge_cache(&self) -> Result<()> {
        if self.incremental.is_none() {
            return Ok(());
        }
        let root = require_local_output(&self.target)?;
        let cache = root.join(CACHE);
        if root.join("_tmp/edges_map").exists() {
            clone_tree(&root.join("_tmp/edges_map"), &cache.join("edges"))?;
        }
        fs::copy(
            root.join(PK_MAP_STATE_PATH),
            cache.join("pk_map_state.json"),
        )?;
        fs::copy(
            root.join(EDGES_MAP_STATE_PATH),
            cache.join("edges_map_state.json"),
        )?;
        let mut reused: Vec<_> = self.reused_tables.iter().collect();
        reused.sort();
        let mut rebuilt: Vec<_> = self
            .source
            .tables
            .iter()
            .map(|t| &t.name)
            .filter(|name| !self.reused_tables.contains(*name))
            .collect();
        rebuilt.sort();
        let mut edges: Vec<_> = self.reused_relations.iter().collect();
        edges.sort();
        let config = self.incremental.as_ref().unwrap();
        let chain_length = if let Some(parent) = &config.parent {
            let info: Value =
                serde_json::from_slice(&fs::read(parent.join(CACHE).join("update.json"))?)?;
            info["update_chain_length"].as_u64().unwrap_or(0) + 1
        } else {
            0
        };
        let tables: BTreeMap<_, _> = self.source.tables.iter().map(|table| {
            let reused = self.reused_tables.contains(&table.name);
            (table.name.clone(), serde_json::json!({
                "status": if reused { "reused" } else { "rebuilt" },
                "rows": self.table_rows.get(&table.name),
                "encoder_state": if reused || self.inherited_encoders.contains_key(&table.name) {
                    "inherited"
                } else { "fitted" },
            }))
        }).collect();
        fs::write(
            cache.join("update.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "reused_tables": reused, "rebuilt_tables": rebuilt, "reused_edges": edges,
                "parent": config.parent,
                "reuse_encoders": config.reuse_encoders,
                "snapshot_id": config.snapshot_id,
                "parent_snapshot_id": config.parent_snapshot_id,
                "created_at": chrono::Utc::now().to_rfc3339(),
                "creation_mode": if config.parent.is_some() { "incremental" } else { "full" },
                "update_chain_length": chain_length, "tables": tables,
            }))?,
        )?;
        Ok(())
    }

    pub(super) fn save_index_commits(&self) -> Result<()> {
        if self.incremental.is_none() {
            return Ok(());
        }
        let root = require_local_output(&self.target)?;
        clone_tree(
            &root.join("_tmp/index_commits"),
            &root.join(CACHE).join("index_commits"),
        )
    }

    pub(super) fn restore_cached_index_shards(&self) -> Result<()> {
        let Some(parent) = self.incremental.as_ref().and_then(|c| c.parent.as_ref()) else {
            return Ok(());
        };
        // Append/compaction changes shard layouts; retained commit markers
        // describe the original Encode generation, not these repacked shards.
        if parent.join(CACHE).join("deltas.json").exists() {
            return Ok(());
        }
        let root = require_local_output(&self.target)?;
        let previous: DatasetMetadata = serde_json::from_slice(&fs::read(parent.join("index_metadata.json"))?)?;
        for table in &self.reused_tables {
            // A removed relation also changes the on-disk record layout.
            if previous.edge_relations.iter().any(|rel| {
                (rel.src_type == *table || rel.dst_type == *table)
                    && !self.reused_relations.contains(&rel.name)
            }) || self.relation_defs.iter().any(|rel| {
                (rel.src_table == *table || rel.dst_table == *table)
                    && !self.reused_relations.contains(&rel.name)
            }) {
                continue;
            }
            for shard in 0..self.num_shards {
                let name = format!("shard-{shard:05}");
                let relative = PathBuf::from("nodes")
                    .join(table.to_lowercase())
                    .join(&name);
                let marker = parent
                    .join(CACHE)
                    .join("index_commits")
                    .join(table.to_lowercase())
                    .join(format!("{name}.json"));
                ensure!(
                    marker.is_file(),
                    "parent snapshot has no committed shard {table}/{name}"
                );
                clone_tree(&parent.join(&relative), &root.join(&relative))?;
                clone_tree(
                    &marker,
                    &index_shard_commit_path(root, &table.to_lowercase(), &name),
                )?;
            }
        }
        Ok(())
    }
}
