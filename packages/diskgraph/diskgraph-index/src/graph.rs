// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::builder::{
    compute_relation_layouts, estimate_edge_record_bytes, shard_node_count, BuildStats,
    EdgeBufferPool, MemoryBudget, ShardBuildParams, ShardBuilder,
};
use crate::layout::{
    OffsetFileHeader, RelationLayout, EDGES_FILENAME, LAYOUT_VERSION, OFFSETS_FILENAME,
};
use crate::pkmap::{self, Key, PkHashTable};
use crate::reader::ShardReader;
use crate::streaming::EdgeShardInput;
use anyhow::{anyhow, ensure, Context, Result};
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, NodeDataCompression, NodeShardMetadata,
    NodeTypeMetadata, PrimaryKeyMap,
};
use diskgraph_core::tt_builder::Value;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const TEMP_META_MAGIC: [u8; 4] = *b"DGTP";
const TEMP_META_VERSION: u8 = 2;
pub const INDEX_SHARD_COMMIT_DIR: &str = "_tmp/index_commits";
const INDEX_SHARD_COMMIT_VERSION: u32 = 1;

#[derive(Debug, Deserialize, Serialize)]
struct IndexShardCommit {
    version: u32,
    shard_key: String,
    expected_nodes: u64,
    offsets_len: u64,
    edges_len: u64,
    build_stats_len: u64,
}

pub struct PreparedResumableJobs {
    pub jobs: Vec<ShardBuildJob>,
    pub resumed_shards: Vec<ResumedShard>,
}

pub struct ResumedShard {
    pub shard_key: Arc<str>,
    pub expected_nodes: u64,
}

/// Validate and enumerate every durable Index shard commit below `root`.
///
/// Temporary marker files are not durable commits and are ignored. Every
/// published marker is checked against its path and exact output files because
/// its scratch inputs may already have been reclaimed.
pub fn validated_index_shard_commits(root: &Path) -> Result<Vec<String>> {
    let commit_root = root.join(INDEX_SHARD_COMMIT_DIR);
    if !commit_root.exists() {
        return Ok(Vec::new());
    }
    ensure_plain_directory(root, "index output root")?;
    ensure!(
        fs::symlink_metadata(&commit_root)?.file_type().is_dir(),
        "index shard commit root is not a directory: {}",
        commit_root.display()
    );
    let nodes_root = root.join("nodes");
    ensure_plain_directory(&nodes_root, "committed nodes root")?;

    let mut shard_keys = Vec::new();
    for table_entry in fs::read_dir(&commit_root)? {
        let table_entry = table_entry?;
        let table_path = table_entry.path();
        let table_type = table_entry.file_type()?;
        ensure!(
            table_type.is_dir() && !table_type.is_symlink(),
            "unexpected entry in index shard commit root: {}",
            table_path.display()
        );
        let table_dir = table_entry.file_name().into_string().map_err(|_| {
            anyhow!(
                "non-UTF8 index shard commit table path: {}",
                table_path.display()
            )
        })?;
        let output_table_dir = nodes_root.join(&table_dir);
        ensure_plain_directory(&output_table_dir, "committed node table")?;

        for marker_entry in fs::read_dir(&table_path)? {
            let marker_entry = marker_entry?;
            let marker_path = marker_entry.path();
            let marker_type = marker_entry.file_type()?;
            ensure!(
                marker_type.is_file() && !marker_type.is_symlink(),
                "unexpected entry in index shard commit table: {}",
                marker_path.display()
            );
            let marker_name = marker_entry.file_name().into_string().map_err(|_| {
                anyhow!(
                    "non-UTF8 index shard commit path: {}",
                    marker_path.display()
                )
            })?;
            if marker_name.ends_with(".json.tmp") {
                continue;
            }
            ensure!(
                marker_name.ends_with(".json"),
                "unexpected index shard commit file: {}",
                marker_path.display()
            );
            let shard_dir = marker_name.trim_end_matches(".json");
            ensure!(
                !shard_dir.is_empty(),
                "empty shard name in index commit marker {}",
                marker_path.display()
            );
            let output_dir = output_table_dir.join(shard_dir);
            ensure_plain_directory(&output_dir, "committed node shard")?;
            let commit = read_validated_shard_commit(&output_dir, &marker_path, None)?
                .ok_or_else(|| {
                    anyhow!(
                        "index shard commit disappeared: {}",
                        marker_path.display()
                    )
                })?;
            let (node_type, committed_shard_dir) =
                commit.shard_key.split_once('/').ok_or_else(|| {
                    anyhow!(
                        "invalid shard key '{}' in index commit {}",
                        commit.shard_key,
                        marker_path.display()
                    )
                })?;
            ensure!(
                node_type.to_lowercase() == table_dir && committed_shard_dir == shard_dir,
                "index shard commit path/key mismatch at {}: found {}",
                marker_path.display(),
                commit.shard_key
            );
            shard_keys.push(commit.shard_key);
        }
    }
    shard_keys.sort_unstable();
    ensure!(
        shard_keys.windows(2).all(|pair| pair[0] != pair[1]),
        "duplicate index shard commit key"
    );
    Ok(shard_keys)
}

fn ensure_plain_directory(path: &Path, kind: &str) -> Result<()> {
    let file_type = fs::symlink_metadata(path)
        .with_context(|| format!("stat {kind} {}", path.display()))?
        .file_type();
    ensure!(
        file_type.is_dir() && !file_type.is_symlink(),
        "{kind} is not a plain directory: {}",
        path.display()
    );
    Ok(())
}

pub struct GraphIndexBuilder {
    metadata: DatasetMetadata,
    source: GraphDataSource,
    output_root: PathBuf,
    /// Coordinated layout fix (write side). When ingesting a sub-index whose
    /// FKs cross into a parent index (a dst_table that lives in the parent,
    /// not this config), the parent's dst node type isn't in
    /// `metadata.node_types`. `compute_relation_layouts` would then fall back
    /// to `dst_bits=64` for cross-boundary edges. Populating this with the
    /// parent's `node_types` lets layout computation pick up the real
    /// `max_id` → 1-2 byte dst encoding for typical fixtures (≈8x edge size
    /// savings on the hot path). The persisted `index_metadata.json` is
    /// unaffected — these types live only in the in-memory build scope.
    external_node_types: Vec<NodeTypeMetadata>,
}

#[derive(Clone)]
pub enum GraphDataSource {
    Local { root: PathBuf },
}

#[derive(Clone, Debug)]
pub enum GraphBuildStage {
    BuildShard { shard: Arc<str> },
    Complete { shard: Arc<str> },
}

#[derive(Clone, Debug)]
pub struct GraphBuildProgress {
    pub stage: GraphBuildStage,
    pub current: u64,
    pub total: Option<u64>,
}

pub trait GraphProgressSink: Send + Sync {
    fn report(&self, update: GraphBuildProgress);
}

pub struct GraphBuildOutput {
    pub shard_stats: HashMap<String, BuildStats>,
    pub total_shard_ms: u64,
    pub total_shards: u64,
}

impl GraphIndexBuilder {
    pub fn from_metadata_path(metadata_path: &Path, output_root: &Path) -> Result<Self> {
        let bytes = fs::read(metadata_path).context("read index_metadata.json")?;
        let metadata: DatasetMetadata =
            serde_json::from_slice(&bytes).context("parse index_metadata.json")?;
        let root = metadata_path
            .parent()
            .ok_or_else(|| anyhow!("index_metadata.json path has no parent"))?
            .to_path_buf();
        Ok(Self::from_metadata(
            metadata,
            GraphDataSource::Local { root },
            output_root,
        ))
    }

    pub fn from_metadata(
        metadata: DatasetMetadata,
        source: GraphDataSource,
        output_root: &Path,
    ) -> Self {
        Self {
            metadata,
            source,
            output_root: output_root.to_path_buf(),
            external_node_types: Vec::new(),
        }
    }

    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Coordinated layout fix (write side): include `types` in the layout
    /// computation scope so `compute_relation_layouts` finds the real
    /// `max_id` for any cross-boundary FK dst. Pass parent's
    /// `metadata.node_types` here when building a sub-index that references
    /// a parent. The persisted `index_metadata.json` stays clean — these
    /// types only enter the in-memory layout-computation `node_types` Vec.
    pub fn with_external_node_types(mut self, types: Vec<NodeTypeMetadata>) -> Self {
        self.external_node_types = types;
        self
    }

    pub fn build(&self, progress: Option<&dyn GraphProgressSink>) -> Result<GraphBuildOutput> {
        fs::create_dir_all(&self.output_root)?;
        let metadata_path = self.output_root.join("index_metadata.json");
        let metadata_file = fs::File::create(&metadata_path)?;
        let mut metadata_writer = BufWriter::new(metadata_file);
        serde_json::to_writer_pretty(&mut metadata_writer, &self.metadata)?;
        metadata_writer.flush()?;

        let jobs = self.prepare_jobs_with_pool(None)?;
        let mut shard_stats = HashMap::new();
        let mut total_shard_ms = 0u64;
        let mut total_shards = 0u64;
        for job in jobs {
            let shard_key = job.shard_key.clone();
            let output = job.run(progress)?;
            shard_stats.insert(shard_key.to_string(), output.build_stats);
            total_shard_ms = total_shard_ms.saturating_add(output.wall_ms);
            total_shards = total_shards.saturating_add(1);
        }
        Ok(GraphBuildOutput {
            shard_stats,
            total_shard_ms,
            total_shards,
        })
    }

    pub fn prepare_jobs(&self) -> Result<Vec<ShardBuildJob>> {
        self.prepare_jobs_with_pool(None)
    }

    pub fn prepare_jobs_with_pool(
        &self,
        edge_buffer_pool: Option<Arc<EdgeBufferPool>>,
    ) -> Result<Vec<ShardBuildJob>> {
        Ok(self.prepare_jobs_impl(edge_buffer_pool, false)?.jobs)
    }

    /// Prepare only incomplete shards and discard inputs for shards whose
    /// committed outputs survived an interrupted index build.
    ///
    /// Callers using this mode must also call
    /// [`ShardBuildJob::cleanup_consumed_inputs`] after each successful job.
    pub fn prepare_resumable_jobs_with_pool(
        &self,
        edge_buffer_pool: Option<Arc<EdgeBufferPool>>,
    ) -> Result<PreparedResumableJobs> {
        self.prepare_jobs_impl(edge_buffer_pool, true)
    }

    fn prepare_jobs_impl(
        &self,
        edge_buffer_pool: Option<Arc<EdgeBufferPool>>,
        resume_completed_shards: bool,
    ) -> Result<PreparedResumableJobs> {
        let mut jobs = Vec::new();
        let mut resumed_shards = Vec::new();
        let input_cleanup_root = self.input_cleanup_root();
        // Layout-computation scope: local types + external (parent) types
        // for cross-boundary FK width lookup. Externals are append-only —
        // `compute_relation_layouts` uses `find` so name collisions favor
        // the local type (good — local always wins when both exist).
        let local_names: std::collections::HashSet<&str> =
            self.metadata.node_types.iter().map(|n| n.name.as_str()).collect();
        let mut layout_node_types = self.metadata.node_types.clone();
        for ext in &self.external_node_types {
            if !local_names.contains(ext.name.as_str()) {
                layout_node_types.push(ext.clone());
            }
        }
        let node_types = Arc::new(layout_node_types);
        for node_type in &self.metadata.node_types {
            let relations: Vec<EdgeRelationMetadata> = self
                .metadata
                .edge_relations
                .iter()
                .filter(|rel| rel.src_type == node_type.name)
                .cloned()
                .collect();
            let relation_metas = Arc::new(relations);
            let node_meta = Arc::new(node_type.clone());
            for (shard_idx, shard_meta) in node_type.shards.iter().enumerate() {
                let shard_key: Arc<str> =
                    Arc::from(format!("{}/shard-{:05}", node_type.name, shard_idx));
                let output_dir = self
                    .output_root
                    .join("nodes")
                    .join(node_type.name.to_lowercase())
                    .join(format!("shard-{:05}", shard_idx));
                let expected_nodes = shard_node_count(node_type, shard_idx)?;
                let commit_path = self.shard_commit_path(node_type, shard_idx);
                let input_cleanup_paths = self.shard_input_cleanup_paths(
                    node_type,
                    shard_meta,
                    relation_metas.as_ref(),
                    shard_idx,
                )?;
                if resume_completed_shards
                    && shard_output_is_committed(
                        &output_dir,
                        &commit_path,
                        Some(shard_key.as_ref()),
                        Some(expected_nodes),
                    )?
                {
                    let reclaimed =
                        remove_consumed_input_paths(&input_cleanup_paths, &input_cleanup_root)?;
                    tracing::info!(
                        shard = shard_key.as_ref(),
                        reclaimed_bytes = reclaimed,
                        "index resume: reusing committed shard output"
                    );
                    resumed_shards.push(ResumedShard {
                        shard_key,
                        expected_nodes,
                    });
                    continue;
                }
                let output_dir = Arc::new(output_dir);
                let (node_data, node_data_max_len) = if node_type.has_node_data {
                    let paths = shard_meta
                        .node_data_paths
                        .clone()
                        .ok_or_else(|| anyhow!("node data paths missing for {shard_key}"))?;
                    let resolved: Vec<PathBuf> = paths
                        .into_iter()
                        .map(|path| self.resolve_path(&path))
                        .collect();
                    let max_len = read_node_data_max_len(&resolved)?;
                    (Some(Arc::new(resolved)), max_len)
                } else {
                    (None, 0)
                };
                let edge_inputs = relation_metas
                    .iter()
                    .map(|rel| {
                        let shard = rel.shards.get(shard_idx).ok_or_else(|| {
                            anyhow!("missing edge shard {} index {}", rel.name, shard_idx)
                        })?;
                        Ok(EdgeShardInput {
                            relation: rel.name.clone(),
                            source_shard_index: shard_idx,
                            paths: self.collect_edge_paths(&shard.path)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let job = ShardBuildJob {
                    shard_key,
                    output_dir,
                    node_meta: Arc::clone(&node_meta),
                    node_types: Arc::clone(&node_types),
                    relation_metas: Arc::clone(&relation_metas),
                    shard_index: shard_idx,
                    node_data_paths: node_data,
                    node_data_max_len,
                    edge_inputs: Arc::new(edge_inputs),
                    expected_nodes,
                    edge_buffer_pool: edge_buffer_pool.clone(),
                    input_cleanup_paths: Arc::new(input_cleanup_paths),
                    input_cleanup_root: Arc::new(input_cleanup_root.clone()),
                    commit_path: Arc::new(commit_path),
                    output_sync_root: Arc::new(self.output_root.clone()),
                    resumable: resume_completed_shards,
                };
                jobs.push(job);
            }
        }
        Ok(PreparedResumableJobs {
            jobs,
            resumed_shards,
        })
    }

    fn shard_input_cleanup_paths(
        &self,
        node_type: &NodeTypeMetadata,
        shard_meta: &NodeShardMetadata,
        relation_metas: &[EdgeRelationMetadata],
        shard_idx: usize,
    ) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        if node_type.has_node_data {
            let node_paths = shard_meta.node_data_paths.as_ref().ok_or_else(|| {
                anyhow!(
                    "node data paths missing for {}/shard-{:05}",
                    node_type.name,
                    shard_idx
                )
            })?;
            for path in node_paths {
                let data_path = self.resolve_path(path);
                if let Some(parent) = data_path.parent() {
                    paths.push(parent.join("node_data.meta"));
                }
                paths.push(data_path);
            }
        }
        for relation in relation_metas {
            let shard = relation.shards.get(shard_idx).ok_or_else(|| {
                anyhow!(
                    "missing edge shard {} index {}",
                    relation.name,
                    shard_idx
                )
            })?;
            paths.push(self.resolve_path(&shard.path));
        }
        Ok(paths)
    }

    fn resolve_path(&self, rel_path: &str) -> PathBuf {
        match &self.source {
            GraphDataSource::Local { root } => root.join(rel_path),
        }
    }

    fn input_cleanup_root(&self) -> PathBuf {
        match &self.source {
            GraphDataSource::Local { root } => root.join("_tmp"),
        }
    }

    fn shard_commit_path(&self, node_type: &NodeTypeMetadata, shard_idx: usize) -> PathBuf {
        match &self.source {
            GraphDataSource::Local { root } => index_shard_commit_path(
                root,
                &node_type.name.to_lowercase(),
                &format!("shard-{:05}", shard_idx),
            ),
        }
    }

    fn collect_edge_paths(&self, rel_path: &str) -> Result<Vec<PathBuf>> {
        let path = self.resolve_path(rel_path);
        if path.is_dir() {
            let mut paths = Vec::new();
            for entry in fs::read_dir(&path)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("bin") {
                    paths.push(path);
                }
            }
            paths.sort();
            Ok(paths)
        } else {
            Ok(vec![path])
        }
    }
}

fn read_node_data_max_len(paths: &[PathBuf]) -> Result<u32> {
    let Some(path) = paths.first() else {
        return Ok(0);
    };
    let Some(dir) = path.parent() else {
        return Ok(0);
    };
    let meta_path = dir.join("node_data.meta");
    if !meta_path.exists() {
        return Ok(0);
    }
    let bytes = fs::read(&meta_path).with_context(|| format!("read {}", meta_path.display()))?;
    if bytes.len() < 4 + 1 + 1 + 8 + 4 + 4 {
        return Err(anyhow!("node data meta truncated"));
    }
    if bytes[0..4] != TEMP_META_MAGIC {
        return Err(anyhow!("node data meta magic mismatch"));
    }
    if bytes[4] != TEMP_META_VERSION {
        return Err(anyhow!("node data meta version mismatch"));
    }
    let max_row_len = u32::from_le_bytes(bytes[14..18].try_into().unwrap());
    Ok(max_row_len)
}

pub struct ShardBuildJob {
    shard_key: Arc<str>,
    output_dir: Arc<PathBuf>,
    node_meta: Arc<NodeTypeMetadata>,
    node_types: Arc<Vec<NodeTypeMetadata>>,
    relation_metas: Arc<Vec<EdgeRelationMetadata>>,
    shard_index: usize,
    node_data_paths: Option<Arc<Vec<PathBuf>>>,
    node_data_max_len: u32,
    edge_inputs: Arc<Vec<EdgeShardInput>>,
    expected_nodes: u64,
    edge_buffer_pool: Option<Arc<EdgeBufferPool>>,
    input_cleanup_paths: Arc<Vec<PathBuf>>,
    input_cleanup_root: Arc<PathBuf>,
    commit_path: Arc<PathBuf>,
    output_sync_root: Arc<PathBuf>,
    resumable: bool,
}

impl ShardBuildJob {
    pub fn shard_key(&self) -> &str {
        self.shard_key.as_ref()
    }

    pub fn estimated_edge_record_bytes(&self) -> Result<u64> {
        estimate_edge_record_bytes(self.edge_inputs.as_ref())
    }

    /// Remove the temporary node and mapped-edge inputs consumed by this job.
    /// The caller must invoke this only after [`Self::run_with_budget`]
    /// succeeds and the shard output is committed.
    pub fn cleanup_consumed_inputs(&self) -> Result<u64> {
        remove_consumed_input_paths(
            self.input_cleanup_paths.as_ref(),
            self.input_cleanup_root.as_ref(),
        )
    }

    pub fn run(
        &self,
        progress: Option<&dyn GraphProgressSink>,
    ) -> Result<crate::builder::ShardBuildOutput> {
        self.run_with_budget(progress, None)
    }

    pub fn run_with_budget(
        &self,
        progress: Option<&dyn GraphProgressSink>,
        mem_budget: Option<Arc<MemoryBudget>>,
    ) -> Result<crate::builder::ShardBuildOutput> {
        let start = std::time::Instant::now();
        fs::create_dir_all(&*self.output_dir)?;
        let params = ShardBuildParams {
            output_dir: Arc::clone(&self.output_dir),
            node_meta: Arc::clone(&self.node_meta),
            node_types: Arc::clone(&self.node_types),
            relation_metas: Arc::clone(&self.relation_metas),
            shard_index: self.shard_index,
            node_data_paths: self.node_data_paths.as_ref().map(Arc::clone),
            node_data_max_len: self.node_data_max_len,
            mem_budget,
            edge_buffer_pool: self.edge_buffer_pool.clone(),
        };
        let builder = ShardBuilder::new(params);
        let adapter =
            progress.map(|sink| ShardProgressAdapter::new(Arc::clone(&self.shard_key), sink));
        let mut output = builder.build_with_progress(
            &self.edge_inputs,
            Some(self.expected_nodes),
            adapter
                .as_ref()
                .map(|a| a as &dyn crate::builder::ProgressSink),
            None,
        )?;
        if self.resumable {
            commit_shard_output(
                self.output_dir.as_ref(),
                self.commit_path.as_ref(),
                self.shard_key.as_ref(),
                self.expected_nodes,
                self.output_sync_root.as_ref(),
                self.input_cleanup_root
                    .parent()
                    .ok_or_else(|| anyhow!("index scratch root has no parent"))?,
            )?;
        }
        if let Some(sink) = progress {
            sink.report(GraphBuildProgress {
                stage: GraphBuildStage::Complete {
                    shard: Arc::clone(&self.shard_key),
                },
                current: 0,
                total: None,
            });
        }
        output.wall_ms = start.elapsed().as_millis() as u64;
        Ok(output)
    }
}

pub fn index_shard_commit_path(root: &Path, table_dir: &str, shard_dir: &str) -> PathBuf {
    root.join(INDEX_SHARD_COMMIT_DIR)
        .join(table_dir)
        .join(format!("{shard_dir}.json"))
}

/// A commit marker is published only after all shard outputs are synced.
/// A missing marker means the output is provisional and can be rebuilt. An
/// invalid marker is an integrity error: its inputs may already be reclaimed,
/// so silently rebuilding would be unsafe.
pub fn shard_output_is_committed(
    output_dir: &Path,
    commit_path: &Path,
    expected_shard_key: Option<&str>,
    expected_nodes: Option<u64>,
) -> Result<bool> {
    let Some(commit) = read_validated_shard_commit(
        output_dir,
        commit_path,
        expected_shard_key,
    )? else {
        return Ok(false);
    };
    if let Some(expected) = expected_nodes {
        ensure!(
            commit.expected_nodes == expected,
            "index shard commit node count mismatch at {}: expected {}, found {}",
            commit_path.display(),
            expected,
            commit.expected_nodes
        );
    }
    Ok(true)
}

/// Return the durable node count for a committed shard whose Encode metadata
/// may already have been reclaimed. The marker and all committed output files
/// are validated before its count is trusted.
pub fn committed_shard_expected_nodes(
    output_dir: &Path,
    commit_path: &Path,
    expected_shard_key: Option<&str>,
) -> Result<Option<u64>> {
    Ok(read_validated_shard_commit(
        output_dir,
        commit_path,
        expected_shard_key,
    )?
    .map(|commit| commit.expected_nodes))
}

fn read_validated_shard_commit(
    output_dir: &Path,
    commit_path: &Path,
    expected_shard_key: Option<&str>,
) -> Result<Option<IndexShardCommit>> {
    let marker_type = match fs::symlink_metadata(commit_path) {
        Ok(metadata) => metadata.file_type(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("stat index shard commit {}", commit_path.display()))
        }
    };
    ensure!(
        marker_type.is_file() && !marker_type.is_symlink(),
        "index shard commit is not a plain file: {}",
        commit_path.display()
    );
    let payload = fs::read(commit_path)
        .with_context(|| format!("read index shard commit {}", commit_path.display()))?;
    let commit: IndexShardCommit = serde_json::from_slice(&payload)
        .with_context(|| format!("parse index shard commit {}", commit_path.display()))?;
    ensure!(
        commit.version == INDEX_SHARD_COMMIT_VERSION,
        "unsupported index shard commit version {} at {}",
        commit.version,
        commit_path.display()
    );
    if let Some(expected) = expected_shard_key {
        ensure!(
            commit.shard_key == expected,
            "index shard commit key mismatch at {}: expected {}, found {}",
            commit_path.display(),
            expected,
            commit.shard_key
        );
    }
    validate_committed_shard_files(output_dir, &commit)
        .with_context(|| format!("validate committed index shard {}", commit.shard_key))?;
    Ok(Some(commit))
}

fn commit_shard_output(
    output_dir: &Path,
    commit_path: &Path,
    shard_key: &str,
    expected_nodes: u64,
    output_sync_root: &Path,
    commit_sync_root: &Path,
) -> Result<()> {
    let offsets_len = sync_file_len(&output_dir.join(OFFSETS_FILENAME))?;
    let edges_len = sync_file_len(&output_dir.join(EDGES_FILENAME))?;
    let build_stats_path = output_dir.join("build_stats.json");
    let build_stats_len = sync_file_len(&build_stats_path)?;
    let report = fs::File::open(&build_stats_path)?;
    serde_json::from_reader::<_, serde_json::Value>(report)
        .context("parse completed shard build_stats.json")?;
    sync_directory_chain(output_dir, output_sync_root)?;

    let commit = IndexShardCommit {
        version: INDEX_SHARD_COMMIT_VERSION,
        shard_key: shard_key.to_string(),
        expected_nodes,
        offsets_len,
        edges_len,
        build_stats_len,
    };
    write_durable_commit(
        commit_path,
        &serde_json::to_vec(&commit)?,
        commit_sync_root,
    )?;
    validate_committed_shard_files(output_dir, &commit)
}

fn validate_committed_shard_files(output_dir: &Path, commit: &IndexShardCommit) -> Result<()> {
    ensure_plain_directory(output_dir, "committed node shard")?;
    for (name, expected_len) in [
        (OFFSETS_FILENAME, commit.offsets_len),
        (EDGES_FILENAME, commit.edges_len),
        ("build_stats.json", commit.build_stats_len),
    ] {
        let path = output_dir.join(name);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("stat committed shard file {}", path.display()))?;
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "committed shard path is not a plain file: {}",
            path.display()
        );
        let actual = metadata.len();
        ensure!(
            actual == expected_len,
            "committed shard file size changed for {}: expected {}, found {}",
            path.display(),
            expected_len,
            actual
        );
    }

    let report = fs::File::open(output_dir.join("build_stats.json"))?;
    serde_json::from_reader::<_, serde_json::Value>(report)
        .context("parse committed shard build_stats.json")?;

    let offsets_path = output_dir.join(OFFSETS_FILENAME);
    let mut offsets = fs::File::open(&offsets_path)?;
    let header_len = bincode::serialized_size(&OffsetFileHeader::new(0, 0))? as usize;
    let mut header_bytes = vec![0u8; header_len];
    offsets
        .read_exact(&mut header_bytes)
        .with_context(|| format!("read committed offsets header {}", offsets_path.display()))?;
    let header: OffsetFileHeader = bincode::deserialize(&header_bytes)?;
    ensure!(
        header.version == LAYOUT_VERSION,
        "committed offsets layout version mismatch: expected {}, found {}",
        LAYOUT_VERSION,
        header.version
    );
    ensure!(
        header.nodes == commit.expected_nodes
            && header.total_offsets == commit.expected_nodes.saturating_add(1),
        "committed offsets count mismatch: expected {} nodes, found {} nodes and {} offsets",
        commit.expected_nodes,
        header.nodes,
        header.total_offsets
    );
    Ok(())
}

fn sync_file_len(path: &Path) -> Result<u64> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    // Windows FlushFileBuffers requires GENERIC_WRITE on the file handle.
    #[cfg(windows)]
    options.write(true);
    let file = options
        .open(path)
        .with_context(|| format!("open {} for sync", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync completed shard file {}", path.display()))?;
    Ok(file.metadata()?.len())
}

fn sync_directory(path: &Path) -> Result<()> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;

        // CreateFile requires this flag to open a directory, and
        // FlushFileBuffers requires write access. Keep flush failures fatal:
        // unsupported filesystems must not silently weaken shard durability.
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        options.write(true).custom_flags(FILE_FLAG_BACKUP_SEMANTICS);
    }
    options
        .open(path)
        .and_then(|dir| dir.sync_all())
        .with_context(|| format!("sync directory {}", path.display()))
}

fn sync_directory_chain(start: &Path, stop: &Path) -> Result<()> {
    let mut current = Some(start);
    while let Some(path) = current {
        sync_directory(path)?;
        if path == stop {
            return Ok(());
        }
        current = path.parent();
    }
    Err(anyhow!(
        "directory {} is not below durable root {}",
        start.display(),
        stop.display()
    ))
}

fn write_durable_commit(path: &Path, payload: &[u8], durable_root: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("index shard commit has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let tmp_path = path.with_extension("json.tmp");
    let _ = fs::remove_file(&tmp_path);
    let mut file = fs::File::create(&tmp_path)?;
    file.write_all(payload)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp_path, path)?;
    sync_directory_chain(parent, durable_root)
}

fn remove_consumed_input_paths(paths: &[PathBuf], allowed_root: &Path) -> Result<u64> {
    let mut removed_bytes = 0u64;
    let mut unique = HashSet::new();
    let allowed_root = allowed_root
        .canonicalize()
        .with_context(|| format!("canonicalize index scratch root {}", allowed_root.display()))?;
    for path in paths {
        if !unique.insert(path.clone()) || !path.exists() {
            continue;
        }
        let link_metadata = fs::symlink_metadata(path)
            .with_context(|| format!("stat consumed index input {}", path.display()))?;
        ensure!(
            !link_metadata.file_type().is_symlink(),
            "refusing to remove symlinked index input {}",
            path.display()
        );
        let canonical = path
            .canonicalize()
            .with_context(|| format!("canonicalize consumed index input {}", path.display()))?;
        ensure!(
            canonical.starts_with(&allowed_root) && canonical != allowed_root,
            "refusing to remove index input outside {}: {}",
            allowed_root.display(),
            canonical.display()
        );
        removed_bytes = removed_bytes.saturating_add(if link_metadata.is_dir() {
            directory_bytes(&canonical)?
        } else {
            link_metadata.len()
        });
        if link_metadata.is_dir() {
            fs::remove_dir_all(&canonical).with_context(|| {
                format!("remove consumed index input {}", canonical.display())
            })?;
        } else {
            fs::remove_file(&canonical).with_context(|| {
                format!("remove consumed index input {}", canonical.display())
            })?;
        }
        if let Some(parent) = canonical.parent() {
            sync_directory(parent)?;
        }
    }
    Ok(removed_bytes)
}

fn directory_bytes(root: &Path) -> Result<u64> {
    let mut total = 0u64;
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).with_context(|| format!("read_dir {}", dir.display()))? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_symlink() {
                total = total.saturating_add(fs::symlink_metadata(entry.path())?.len());
            } else {
                total = total.saturating_add(entry.metadata()?.len());
            }
        }
    }
    Ok(total)
}

struct ShardProgressAdapter<'a> {
    shard: Arc<str>,
    sink: &'a dyn GraphProgressSink,
}

impl<'a> ShardProgressAdapter<'a> {
    fn new(shard: Arc<str>, sink: &'a dyn GraphProgressSink) -> Self {
        Self { shard, sink }
    }
}

impl<'a> crate::builder::ProgressSink for ShardProgressAdapter<'a> {
    fn report(&self, update: crate::builder::BuildProgress) {
        let stage = match update.stage {
            crate::builder::BuildStage::BuildIndex => GraphBuildStage::BuildShard {
                shard: Arc::clone(&self.shard),
            },
        };
        self.sink.report(GraphBuildProgress {
            stage,
            current: update.current,
            total: update.total,
        });
    }
}

pub struct GraphIndexReader {
    metadata: DatasetMetadata,
    tables: HashMap<String, Arc<TableReader>>,
    /// Per-column semantic types loaded from `diskgraph_config.json` at index
    /// root. Empty when the file is absent (legacy indexes pre-#28400
    /// follow-up). Used by TT-gen resolver to dispatch per-token observation
    /// for multicategorical sources without per-query plan-JSON plumbing.
    column_semantic_types: HashMap<(String, String), String>,
    /// Unified view of edge relations across parent + every loaded extension.
    /// Extension relations have their `src_type` rewritten to the namespaced
    /// `<tt_id>/<src_type>` form so a consumer (e.g. the sampler engine)
    /// can resolve them via the merged `node_type_to_id` lookup. `dst_type`
    /// is left bare when the destination is a parent table (the common case
    /// for seed-table forward FKs).
    ///
    /// Populated by `open` to mirror `metadata.edge_relations`, and extended
    /// by every `merge_extension_into` call. Read via
    /// `merged_edge_relations()`.
    merged_edge_relations: Vec<EdgeRelationMetadata>,
    /// Filesystem root of the TT extension this reader was opened against
    /// (`<parent_root>/train_tables/<tt_id>/`). Populated only by
    /// `open_with_tt`; `None` for plain `open()`. Sample-time consumers use
    /// this to locate sidecar files (`weight.bin`, `split.bin`, etc.) that
    /// live alongside the sub-index but aren't part of its rowpack catalog.
    /// One source of truth: the reader knows where its TT lives; callers
    /// don't have to thread the path through.
    tt_extension_root: Option<PathBuf>,
}

impl GraphIndexReader {
    pub fn open(metadata_path: &Path, root: &Path) -> Result<Self> {
        Self::open_impl(metadata_path, root, &[], None)
    }

    /// Open ONLY the named tables of an index (mmaps, pk maps, and rowpack
    /// decoders are per-table costs). Used by callers that need a handful of
    /// verified-lookup handles into a large parent index — e.g. the TT-ingest
    /// membership verifier — where a full `open` would mmap every table ×
    /// shard and load every pk hash for nothing. Tables absent from the index
    /// are silently omitted (callers observe absence via [`Self::table`]).
    /// Layout computation still sees ALL node types, so edge widths of the
    /// opened tables stay correct.
    pub fn open_subset(
        metadata_path: &Path,
        root: &Path,
        only_tables: &std::collections::HashSet<String>,
    ) -> Result<Self> {
        Self::open_impl(metadata_path, root, &[], Some(only_tables))
    }

    /// Open an index but augment its `node_types` with the supplied
    /// external entries when computing edge-relation layouts. Used by
    /// `merge_extension_into` so the extension's edges (which point at
    /// the parent's tables) can be sized correctly: without the parent's
    /// `users` in the layout-computation scope, `compute_relation_layouts`
    /// would fall back to `dst_bits=64` and the read view would not match
    /// the on-disk byte width Phase 1 produced.
    pub(crate) fn open_with_external_node_types(
        metadata_path: &Path,
        root: &Path,
        external_node_types: &[NodeTypeMetadata],
    ) -> Result<Self> {
        Self::open_impl(metadata_path, root, external_node_types, None)
    }

    fn open_impl(
        metadata_path: &Path,
        root: &Path,
        external_node_types: &[NodeTypeMetadata],
        only_tables: Option<&std::collections::HashSet<String>>,
    ) -> Result<Self> {
        let bytes = fs::read(metadata_path).context("read index_metadata.json")?;
        let metadata: DatasetMetadata =
            serde_json::from_slice(&bytes).context("parse index_metadata.json")?;
        let mut tables = HashMap::new();
        tracing::info!(
            node_types = metadata.node_types.len(),
            edge_relations = metadata.edge_relations.len(),
            external_node_types = external_node_types.len(),
            "Loaded index metadata"
        );
        // Build a unified node_types view for layout computation. The
        // extension's own node_types take precedence; externals fill in
        // referenced-but-absent dst types so `bits_for_id` resolves
        // correctly.
        let mut layout_node_types: Vec<NodeTypeMetadata> = metadata.node_types.clone();
        let local_names: std::collections::HashSet<&str> =
            metadata.node_types.iter().map(|n| n.name.as_str()).collect();
        for ext in external_node_types {
            if !local_names.contains(ext.name.as_str()) {
                layout_node_types.push(ext.clone());
            }
        }
        for node_type in &metadata.node_types {
            if let Some(only) = only_tables {
                if !only.contains(&node_type.name) {
                    continue;
                }
            }
            let (node_data_compression, node_data_decoder) = match &node_type.node_data_compression
            {
                Some(cfg) => {
                    if !node_type.has_node_data {
                        return Err(anyhow!(
                            "node data compression set but has_node_data is false for {}",
                            node_type.name
                        ));
                    }
                    let ctx_path = root.join(&cfg.context_path);
                    let bytes = fs::read(&ctx_path)
                        .with_context(|| format!("read node data ctx {}", ctx_path.display()))?;
                    let ctx = diskgraph_rowpack::RowPackContext::from_bytes(&bytes)?;
                    let decoder = diskgraph_rowpack::RowPackDecoder::new(ctx)?;
                    (Some(cfg.clone()), Some(decoder))
                }
                None => (None, None),
            };
            let relations: Vec<EdgeRelationMetadata> = metadata
                .edge_relations
                .iter()
                .filter(|rel| rel.src_type == node_type.name)
                .cloned()
                .collect();
            let layouts = compute_relation_layouts(&relations, &layout_node_types)?;
            let layouts = Arc::new(layouts);
            let mut shard_readers: Vec<Arc<ShardReader>> = Vec::new();
            for (shard_idx, _shard_meta) in node_type.shards.iter().enumerate() {
                let dir = root
                    .join("nodes")
                    .join(node_type.name.to_lowercase())
                    .join(format!("shard-{:05}", shard_idx));
                let reader =
                    ShardReader::open(&dir, Arc::clone(&layouts), node_type.has_node_data)?;
                shard_readers.push(Arc::new(reader));
            }
            let pk_map = match &node_type.primary_key_map {
                Some(pk) if pk.mapping_kind == "ptrhash" => {
                    Some(load_pk_hash_table(root, node_type, pk)?)
                }
                Some(pk) => {
                    tracing::warn!(
                        "unsupported primary key map kind {} for {}",
                        pk.mapping_kind,
                        node_type.name
                    );
                    None
                }
                None => None,
            };
            let table = TableReader::new(
                node_type,
                relations,
                Arc::clone(&layouts),
                shard_readers,
                node_data_compression,
                node_data_decoder,
                node_type.primary_key_map.clone(),
                pk_map,
            )?;
            tables.insert(node_type.name.clone(), Arc::new(table));
        }
        let column_semantic_types = load_column_semantic_types(root);
        let merged_edge_relations = metadata.edge_relations.clone();
        tracing::info!(
            tables = tables.len(),
            stype_columns = column_semantic_types.len(),
            "GraphIndexReader open completed"
        );
        Ok(Self {
            metadata,
            tables,
            column_semantic_types,
            merged_edge_relations,
            // `open` / `open_with_external_node_types` produce a reader over
            // a single index. `open_with_tt` overwrites this after
            // `merge_extension_into` so sidecar lookups land in the right
            // place.
            tt_extension_root: None,
        })
    }

    /// Edge relations across parent + every loaded extension. Extension
    /// relations have `src_type` rewritten to the namespaced
    /// `<tt_id>/<src_type>` form so the engine's `node_type_to_id` lookup
    /// resolves them. For the runtime case (one tt at a time) this is just
    /// parent's edge_relations + the single extension's one or two
    /// relations, sequentially. No new state for the multi-extension case
    /// because we don't load multiple at runtime.
    pub fn merged_edge_relations(&self) -> &[EdgeRelationMetadata] {
        &self.merged_edge_relations
    }

    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    pub fn table(&self, name: &str) -> Option<Arc<TableReader>> {
        self.tables.get(name).cloned()
    }

    pub fn tables(&self) -> Vec<Arc<TableReader>> {
        let mut tables: Vec<Arc<TableReader>> = self.tables.values().cloned().collect();
        tables.sort_by(|a, b| a.node_type().cmp(b.node_type()));
        tables
    }

    /// Sorted catalog keys (these are the namespaced names — `<tt_id>/<bare>`
    /// for extension tables, bare for parent tables). The engine builds its
    /// `node_type_to_id` map keyed by these names so cross-boundary edge
    /// relations (which carry namespaced src_type per `merged_edge_relations`)
    /// resolve correctly.
    pub fn table_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tables.keys().cloned().collect();
        names.sort();
        names
    }

    /// Look up the kumo `stype` for a column from the persisted graph config.
    /// Returns `None` when the index has no `diskgraph_config.json` (legacy)
    /// or the column has no declared `semantic_type`.
    pub fn column_semantic_type(&self, table: &str, column: &str) -> Option<&str> {
        self.column_semantic_types
            .get(&(table.to_string(), column.to_string()))
            .map(String::as_str)
    }

    /// Open the parent DG index together with exactly one seed-table
    /// extension. A bare `tt_ref` keeps legacy behavior and opens
    /// `<parent_root>/train_tables/<tt_ref>/`; explicit
    /// `train_tables/<id>` and `pred_tables/<id>` refs open that subdir.
    /// This is the runtime API: kumo always opens parent + the one seed
    /// table being trained or scored against. There is no multi-tt-at-once
    /// runtime case.
    ///
    /// The extension's tables are merged under namespaced keys
    /// `<id>/<table_name>`; parent tables keep their bare names. The
    /// extension's edge_relations are merged into `merged_edge_relations`
    /// with namespaced `src_type` so the sampler engine's name lookup
    /// resolves them.
    ///
    /// Errors loudly when the extension doesn't exist or isn't finalized
    /// (no `index_metadata.json` under the subdir). Callers should use
    /// `discover_tts` first if they need to ask "what's available".
    pub fn open_with_tt(parent_root: &Path, tt_ref: &str) -> Result<Self> {
        if tt_ref.is_empty() {
            return Err(anyhow!("empty tt_ref"));
        }
        let (subdir_rel, namespace) = if let Some((prefix, id)) = tt_ref.split_once('/') {
            if id.is_empty() || id.contains('/') {
                return Err(anyhow!("invalid extension reference `{tt_ref}`"));
            }
            match prefix {
                "train_tables" | "pred_tables" => (tt_ref.to_string(), id.to_string()),
                _ => return Err(anyhow!("invalid extension reference `{tt_ref}`")),
            }
        } else {
            (format!("train_tables/{tt_ref}"), tt_ref.to_string())
        };
        Self::open_with_extension_subdir(parent_root, &subdir_rel, &namespace)
    }

    /// Generalization of [`open_with_tt`] to an extension at an arbitrary
    /// relative `subdir` under the parent (e.g. `pred_tables/job1` for the
    /// batch-prediction path, or `train_tables/<tt_id>` for training). The
    /// extension's tables are merged under the `<namespace>/<table_name>`
    /// keys and its cross-boundary edges are decoded against the parent's
    /// node types — the same recipe `open_with_tt` uses.
    pub fn open_with_extension_subdir(
        parent_root: &Path,
        subdir_rel: &str,
        namespace: &str,
    ) -> Result<Self> {
        let parent_meta_path = parent_root.join("index_metadata.json");
        let mut reader = Self::open(&parent_meta_path, parent_root)?;
        let subdir = parent_root.join(subdir_rel);
        let ext_meta_path = subdir.join("index_metadata.json");
        if !ext_meta_path.is_file() {
            return Err(anyhow!(
                "extension `{namespace}` has no finalized index at {}",
                subdir.display()
            ));
        }
        merge_extension_into(&mut reader, &subdir, namespace)?;
        // Remember the extension root so sample-time helpers can locate
        // sidecars (weight.bin, split.bin, ...) without callers threading
        // the path through.
        reader.tt_extension_root = Some(subdir);
        Ok(reader)
    }

    /// Filesystem root of the TT extension this reader was opened against,
    /// when applicable. `None` for plain `open()` readers. Sample-time
    /// consumers read this to locate sidecar files that live next to
    /// `index_metadata.json` (e.g. `weight.bin`).
    pub fn tt_extension_root(&self) -> Option<&Path> {
        self.tt_extension_root.as_deref()
    }
}

/// List the tt_ids of every finalized seed-table extension under
/// `<parent_root>/train_tables/<tt_id>/`. A subdir is considered finalized
/// iff it contains an `index_metadata.json`. Mid-ingest or stale subdirs
/// are excluded so callers can iterate the result without each one needing
/// to re-check finalization. Returns an empty `Vec` (not an error) when
/// no extensions exist or the `train_tables/` dir is absent — the empty
/// state is valid for a freshly built parent index.
pub fn discover_tts(parent_root: &Path) -> Result<Vec<String>> {
    let train_tables_dir = parent_root.join("train_tables");
    if !train_tables_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut ids = Vec::new();
    for entry in fs::read_dir(&train_tables_dir)
        .with_context(|| format!("read train_tables dir {}", train_tables_dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        // Finalization gate: skip subdirs without index_metadata.json.
        // This is the same signal `open_with_tt` checks, kept consistent
        // so `discover_tts` results are always loadable.
        if entry.path().join("index_metadata.json").is_file() {
            ids.push(name);
        }
    }
    ids.sort();
    Ok(ids)
}

/// Open the extension at `subdir` and merge its tables into `reader` under
/// the `<tt_id>/<table_name>` namespace. Parent tables in `reader` are
/// untouched. The extension is opened with parent's `node_types` supplied
/// as externals so its edge-layout computation (compute_relation_layouts)
/// resolves dst types like `users` (which live in the parent, not the
/// extension's own metadata) and produces the correct dst_bits / entry_bytes.
/// Open the extension and merge it into the reader.
///
/// IMPORTANT (Phase 3 follow-up): this opens the extension via the plain
/// `Self::open`, NOT `open_with_external_node_types`. Reason: Phase 1's
/// ingest writes the seed's edges with `dst_max_id=0` (the parent isn't in
/// `max_ids`) → `compute_relation_layouts` produces dst_bits=64 fallback at
/// write side, entry_bytes=8. The read side here also uses dst_bits=64
/// fallback for the same reason. Write and read AGREE at 8-byte entries —
/// fine for the cross-boundary Phase 2 walk test which masks with
/// dst_mask=u64::MAX.
///
/// The sampler engine (Phase 3) builds a merged-view layout that would
/// resolve `users` to its real max_id and produce dst_bits=4 / entry_bytes=1
/// — which is what the on-disk format SHOULD be but isn't, because Phase 1
/// writes assuming the fallback. Closing the loop end-to-end requires
/// Phase 1's ingest to know parent's max_id at edge-write time AND Phase 2
/// to recompute the read layout with the same parent max_id. The two
/// sides must move together; either alone breaks read/write parity. This
/// is the Phase 3 sampler enablement work.
fn merge_extension_into(
    reader: &mut GraphIndexReader,
    subdir: &Path,
    tt_id: &str,
) -> Result<()> {
    let ext_meta_path = subdir.join("index_metadata.json");
    // Coordinated layout fix (read side): the extension's seed→parent edges
    // were written with `dst_bits = bits_for_id(parent.users.max_id)` at
    // ingest time. To decode them we MUST re-compute the layout with the
    // same `node_types` scope — that's the parent's, not the extension's.
    // Without this scope the extension's open falls back to dst_bits=64
    // and reads garbage from the (smaller) on-disk packed edges.
    let parent_node_types = reader.metadata.node_types.clone();
    let ext = GraphIndexReader::open_with_external_node_types(
        &ext_meta_path,
        subdir,
        &parent_node_types,
    )
    .with_context(|| format!("open extension {tt_id} at {}", subdir.display()))?;
    for (bare_name, table_arc) in ext.tables.iter() {
        let namespaced = format!("{tt_id}/{bare_name}");
        if reader.tables.contains_key(&namespaced) {
            return Err(anyhow!(
                "extension table `{namespaced}` collides with existing entry — \
                 indicates a duplicate tt_id load"
            ));
        }
        reader.tables.insert(namespaced, Arc::clone(table_arc));
    }
    // Merge column_semantic_types under the namespaced table name as well.
    for ((table, column), stype) in ext.column_semantic_types.iter() {
        let namespaced_table = format!("{tt_id}/{table}");
        reader
            .column_semantic_types
            .insert((namespaced_table, column.clone()), stype.clone());
    }
    // Merge edge_relations into the unified view, rewriting src_type to use
    // the namespaced form. dst_type stays bare for FKs pointing at parent
    // tables. If a future extension ever introduces FKs whose dst is another
    // extension table, that namespace rewrite belongs here too.
    for rel in &ext.metadata.edge_relations {
        let mut merged_rel = rel.clone();
        merged_rel.src_type = format!("{tt_id}/{}", rel.src_type);
        merged_rel.name = format!("{tt_id}/{}", rel.name);
        reader.merged_edge_relations.push(merged_rel);
    }
    Ok(())
}

/// Best-effort load of `{root}/diskgraph_config.json`. Returns an empty map
/// when the file is missing or malformed; the resolver treats absence as
/// "stype unknown" and falls back to the legacy whole-cell observe path.
fn load_column_semantic_types(root: &Path) -> HashMap<(String, String), String> {
    let path = root.join("diskgraph_config.json");
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(_) => return HashMap::new(),
    };
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(err) => {
            tracing::warn!(
                "diskgraph_config.json present but unparseable: {err}; \
                 multicat dispatch will fall back to whole-cell"
            );
            return HashMap::new();
        }
    };
    let mut out = HashMap::new();
    let Some(tables) = value.get("tables").and_then(|t| t.as_array()) else {
        return out;
    };
    for table in tables {
        let Some(table_name) = table.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        let Some(columns) = table.get("columns").and_then(|c| c.as_array()) else {
            continue;
        };
        for col in columns {
            let Some(col_name) = col.get("name").and_then(|n| n.as_str()) else {
                continue;
            };
            let Some(stype) = col.get("semantic_type").and_then(|s| s.as_str()) else {
                continue;
            };
            out.insert(
                (table_name.to_string(), col_name.to_string()),
                stype.to_string(),
            );
        }
    }
    out
}

pub struct TableReader {
    node_type: String,
    relations: Vec<EdgeRelationMetadata>,
    layouts: Arc<Vec<RelationLayout>>,
    /// Shards held as `Arc<ShardReader>` so downstream consumers (the
    /// sampler engine) can share the same mmap'd handles without
    /// re-opening them. Cheap to clone; the underlying mmap pages are
    /// shared via the OS page cache regardless, but Arc keeps the file
    /// descriptors + virtual mappings owned by one place.
    shards: Vec<Arc<ShardReader>>,
    shard_ranges: Vec<NodeShardMetadata>,
    shard_offsets: Vec<u64>,
    total_nodes: u64,
    node_data_compression: Option<NodeDataCompression>,
    node_data_decoder: Option<diskgraph_rowpack::RowPackDecoder>,
    primary_key_map: Option<PrimaryKeyMap>,
    pk_map: Option<PkHashTable>,
    create_time_col: Option<String>,
    end_time_col: Option<String>,
}

impl TableReader {
    #[allow(clippy::too_many_arguments)]
    fn new(
        node_type: &NodeTypeMetadata,
        relations: Vec<EdgeRelationMetadata>,
        layouts: Arc<Vec<RelationLayout>>,
        shards: Vec<Arc<ShardReader>>,
        node_data_compression: Option<NodeDataCompression>,
        node_data_decoder: Option<diskgraph_rowpack::RowPackDecoder>,
        primary_key_map: Option<PrimaryKeyMap>,
        pk_map: Option<PkHashTable>,
    ) -> Result<Self> {
        let mut shard_offsets = Vec::with_capacity(node_type.shards.len());
        let mut offset = 0u64;
        for shard in &node_type.shards {
            shard_offsets.push(offset);
            if shard.max_id >= shard.min_id {
                offset = offset.saturating_add(shard.max_id - shard.min_id + 1);
            }
        }
        Ok(Self {
            node_type: node_type.name.clone(),
            relations,
            layouts,
            shards,
            shard_ranges: node_type.shards.clone(),
            shard_offsets,
            total_nodes: offset,
            node_data_compression,
            node_data_decoder,
            primary_key_map,
            pk_map,
            create_time_col: node_type.create_time_col.clone(),
            end_time_col: node_type.end_time_col.clone(),
        })
    }

    pub fn node_type(&self) -> &str {
        &self.node_type
    }

    pub fn primary_key_map(&self) -> Option<&PrimaryKeyMap> {
        self.primary_key_map.as_ref()
    }

    pub fn pk_map(&self) -> Option<&PkHashTable> {
        self.pk_map.as_ref()
    }

    pub fn node_data_compression(&self) -> Option<&NodeDataCompression> {
        self.node_data_compression.as_ref()
    }

    pub fn node_data_decoder(&self) -> Option<&diskgraph_rowpack::RowPackDecoder> {
        self.node_data_decoder.as_ref()
    }

    pub fn create_time_col(&self) -> Option<&str> {
        self.create_time_col.as_deref()
    }

    pub fn end_time_col(&self) -> Option<&str> {
        self.end_time_col.as_deref()
    }

    pub fn relations(&self) -> &[EdgeRelationMetadata] {
        &self.relations
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn node_count(&self) -> u64 {
        self.total_nodes
    }

    pub fn layouts(&self) -> Arc<Vec<RelationLayout>> {
        Arc::clone(&self.layouts)
    }

    pub fn shard(&self, index: usize) -> Option<&ShardReader> {
        self.shards.get(index).map(|a| a.as_ref())
    }

    /// Borrow the shared `Arc<ShardReader>` for `index`. Used by callers
    /// (notably the sampler engine) that need to retain a handle to the
    /// shard alongside its `min_id`/`max_id` range without re-opening it.
    pub fn shard_arc(&self, index: usize) -> Option<Arc<ShardReader>> {
        self.shards.get(index).cloned()
    }

    /// Per-shard contiguous id range as `(min_id, max_id)`. The engine
    /// uses this for `find_shard` binary search; in the reader's view
    /// these come from the metadata loaded at open time.
    pub fn shard_id_range(&self, index: usize) -> Option<(u64, u64)> {
        self.shard_ranges
            .get(index)
            .map(|r| (r.min_id, r.max_id))
    }

    pub fn edges_for_node(&self, node_id: u64) -> Result<crate::layout::EdgeListView<'_>> {
        let (shard_idx, local_idx) = self
            .shard_for_node(node_id)
            .ok_or_else(|| anyhow!("node id {node_id} not in shard ranges"))?;
        let reader = self
            .shards
            .get(shard_idx)
            .ok_or_else(|| anyhow!("missing shard reader {shard_idx}"))?;
        reader.edges_for_node(local_idx)
    }

    pub fn decode_node_data(
        &self,
        view: &crate::layout::EdgeListView<'_>,
    ) -> Result<Option<Vec<u8>>> {
        let Some(raw) = view.node_data()? else {
            return Ok(None);
        };
        let Some(compression) = &self.node_data_compression else {
            return Ok(Some(raw.to_vec()));
        };
        if compression.codec != "rowpack" {
            return Err(anyhow!("unsupported node data codec {}", compression.codec));
        }
        let decoder = self
            .node_data_decoder
            .as_ref()
            .ok_or_else(|| anyhow!("node data ctx missing for {}", self.node_type))?;
        let out = decoder.decode_msgpack(raw)?;
        Ok(Some(out))
    }

    pub fn decode_node_data_json(
        &self,
        view: &crate::layout::EdgeListView<'_>,
    ) -> Result<Option<serde_json::Value>> {
        let Some(raw) = view.node_data()? else {
            return Ok(None);
        };
        let Some(compression) = &self.node_data_compression else {
            return Ok(None);
        };
        if compression.codec != "rowpack" {
            return Ok(None);
        }
        let decoder = self
            .node_data_decoder
            .as_ref()
            .ok_or_else(|| anyhow!("node data ctx missing for {}", self.node_type))?;
        let out = decoder.decode_json(raw)?;
        Ok(Some(out))
    }

    pub fn parse_pk_key(&self, input: &str) -> Result<Key> {
        let pk = self
            .primary_key_map
            .as_ref()
            .ok_or_else(|| anyhow!("table {} has no primary key map", self.node_type))?;
        pkmap::parse_key_from_str(pk, input)
    }

    pub fn node_id_from_pk(&self, key: &Key) -> Result<Option<u64>> {
        let pk_map = match &self.pk_map {
            Some(map) => map,
            None => return Ok(None),
        };
        if key.key_type() != pk_map.key_type() {
            return Ok(None);
        }
        let shard_idx = pkmap::shard_for_key(key, self.shards.len());
        let shard = pk_map
            .shard(shard_idx)
            .ok_or_else(|| anyhow!("missing pk shard {}", shard_idx))?;
        let local = match shard.lookup(key) {
            Some(value) => value,
            None => return Ok(None),
        };
        let min_id = self
            .shard_ranges
            .get(shard_idx)
            .map(|range| range.min_id)
            .unwrap_or(0);
        let node_id = min_id.saturating_add(local as u64);
        if let Some(validated) = self.validate_pk_at_node(node_id, key)? {
            if validated {
                return Ok(Some(node_id));
            }
            return Ok(None);
        }
        Ok(Some(node_id))
    }

    /// Check whether `node_id` is the canonical (first-occurrence) node for its
    /// primary key value. Duplicate PK rows get separate node IDs during ingest,
    /// but only the first occurrence is stored in the MPHF. Returns `true` if
    /// this node is canonical or if no pk_map is available (conservative).
    pub fn is_canonical_node(&self, node_id: u64, pk_value: &Value) -> bool {
        let pk_map = match &self.pk_map {
            Some(map) => map,
            None => return true, // no pk_map → assume valid
        };
        let key = match Self::value_to_pk_key(pk_value, pk_map.key_type()) {
            Some(k) => k,
            None => return true, // can't convert → assume valid
        };
        let shard_idx = pkmap::shard_for_key(&key, self.shards.len());
        let shard = match pk_map.shard(shard_idx) {
            Some(s) => s,
            None => return true,
        };
        let canonical_local = match shard.lookup(&key) {
            Some(id) => id,
            None => return true, // not in MPHF → assume valid
        };
        let min_id = self
            .shard_ranges
            .get(shard_idx)
            .map(|r| r.min_id)
            .unwrap_or(0);
        let canonical_node_id = min_id.saturating_add(canonical_local as u64);
        canonical_node_id == node_id
    }

    /// Check whether `value` exists as a primary key in this table.
    ///
    /// # Panics
    /// Panics if the table has no PK map (caller must verify via
    /// `pk_map().is_some()` at resolve time) or if the underlying
    /// shard data is corrupted.
    ///
    /// Lock-free, read-only lookup on the memory-mapped MPHF index —
    /// safe to call from multiple threads concurrently.
    pub fn contains_pk_value(&self, value: &Value) -> bool {
        let pk_map = self
            .pk_map
            .as_ref()
            .expect("contains_pk_value called on table without PK map");
        let key = match Self::value_to_pk_key(value, pk_map.key_type()) {
            Some(k) => k,
            None => return false,
        };
        self.node_id_from_pk(&key)
            .expect("corrupted index: PK shard lookup failed")
            .is_some()
    }

    fn value_to_pk_key(value: &Value, key_type: pkmap::KeyType) -> Option<Key> {
        match (value, key_type) {
            (Value::Int(i), pkmap::KeyType::Int64) => Some(Key::Int(*i as u64)),
            (Value::Str(s), pkmap::KeyType::VarBytes) => Some(Key::Bytes(s.as_bytes().to_vec())),
            (Value::Float(f), pkmap::KeyType::Int64) => Some(Key::Int(*f as u64)),
            _ => None,
        }
    }

    pub fn node_id_from_offset(&self, offset: u64) -> Option<u64> {
        if offset >= self.total_nodes {
            return None;
        }
        for (idx, base) in self.shard_offsets.iter().enumerate() {
            let shard = &self.shard_ranges[idx];
            let shard_len = if shard.max_id >= shard.min_id {
                shard.max_id - shard.min_id + 1
            } else {
                0
            };
            if offset < base.saturating_add(shard_len) {
                let local = offset.saturating_sub(*base);
                return Some(shard.min_id + local);
            }
        }
        None
    }

    /// Verify that the node at `node_id` actually stores `key` as its primary
    /// key. This is the membership check behind [`Self::node_id_from_pk`],
    /// exposed for callers (e.g. stage-extension ingest) that resolve ids
    /// through a standalone minimal-perfect-hash pkmap — which has no
    /// membership verification and silently maps orphan keys onto arbitrary
    /// in-range nodes — and need to reject those false positives.
    ///
    /// Returns `Ok(None)` when the node cannot be verified — no primary-key
    /// map, no rowpack node data, no decoder, or a stored PK type with no
    /// comparison arm (e.g. Decimal/Date/Timestamp PK columns); callers must
    /// treat that as "unknown", not as a mismatch.
    pub fn validate_pk_at_node(&self, node_id: u64, key: &Key) -> Result<Option<bool>> {
        let Some(pk) = self.primary_key_map.as_ref() else {
            return Ok(None);
        };
        let Some(compression) = self.node_data_compression.as_ref() else {
            return Ok(None);
        };
        if compression.codec != "rowpack" {
            return Ok(None);
        }
        let Some(decoder) = self.node_data_decoder.as_ref() else {
            return Ok(None);
        };
        let view = self.edges_for_node(node_id)?;
        view.pk_matches(pk, decoder, key)
            .map_err(|e| anyhow!("node lookup failed: {e}"))
    }

    pub fn shard_for_node(&self, node_id: u64) -> Option<(usize, u64)> {
        for (idx, shard) in self.shard_ranges.iter().enumerate() {
            if node_id >= shard.min_id && node_id <= shard.max_id {
                return Some((idx, node_id - shard.min_id));
            }
        }
        None
    }

    /// Returns the rowpack column indices for `(create_time_col, end_time_col)`.
    /// Returns `None` for each if the table lacks that temporal column or has no
    /// rowpack decoder.
    pub fn temporal_col_indices(&self) -> (Option<usize>, Option<usize>) {
        let Some(decoder) = self.node_data_decoder.as_ref() else {
            return (None, None);
        };
        let ctx = decoder.context();
        let create_idx = self.create_time_col.as_deref().and_then(|name| {
            ctx.columns.iter().position(|c| c.name == name)
        });
        let end_idx = self.end_time_col.as_deref().and_then(|name| {
            ctx.columns.iter().position(|c| c.name == name)
        });
        (create_idx, end_idx)
    }

    /// Check whether the entity identified by `node_id` is temporally valid at
    /// `observation_ts` (epoch seconds). Returns `true` if the entity's
    /// `create_time_col <= observation_ts` and `(end_time_col IS NULL OR
    /// end_time_col > observation_ts)`.
    ///
    /// `create_col_idx` / `end_col_idx` are pre-resolved rowpack column indices
    /// from [`temporal_col_indices`]. If both are `None`, always returns `true`.
    pub fn is_node_temporally_valid(
        &self,
        node_id: u64,
        observation_ts: i64,
        create_col_idx: Option<usize>,
        end_col_idx: Option<usize>,
        scratch: &mut diskgraph_rowpack::RowPackDecodeScratch,
    ) -> Result<bool> {
        self.is_node_valid_between(
            node_id,
            observation_ts,
            observation_ts,
            create_col_idx,
            end_col_idx,
            scratch,
        )
    }

    /// [`is_node_temporally_valid`] with separate create and end reference times.
    pub fn is_node_valid_between(
        &self,
        node_id: u64,
        create_ref_ts: i64,
        end_ref_ts: i64,
        create_col_idx: Option<usize>,
        end_col_idx: Option<usize>,
        scratch: &mut diskgraph_rowpack::RowPackDecodeScratch,
    ) -> Result<bool> {
        if create_col_idx.is_none() && end_col_idx.is_none() {
            return Ok(true);
        }
        let view = self.edges_for_node(node_id)?;
        let Some(raw) = view.node_data()? else {
            // No node data → can't check → conservatively include
            return Ok(true);
        };
        let decoder = match self.node_data_decoder.as_ref() {
            Some(d) => d,
            None => return Ok(true),
        };

        // Check create_time_col <= create_ref_ts
        if let Some(col_idx) = create_col_idx {
            let val = decoder.decode_column_value(raw, col_idx, scratch)?;
            let ts_secs = temporal_value_to_epoch_secs(&val);
            if let Some(secs) = ts_secs {
                if secs > create_ref_ts {
                    return Ok(false);
                }
            }
            // NULL create_time → conservatively include
        }

        // Check end_time_col IS NULL OR end_time_col > end_ref_ts
        if let Some(col_idx) = end_col_idx {
            let val = decoder.decode_column_value(raw, col_idx, scratch)?;
            let ts_secs = temporal_value_to_epoch_secs(&val);
            if let Some(secs) = ts_secs {
                if secs <= end_ref_ts {
                    return Ok(false);
                }
            }
            // NULL end_time → entity still active → include
        }

        Ok(true)
    }

    /// Like [`contains_pk_value`] but also checks temporal validity.
    /// Returns `true` only if the PK exists AND the node is temporally valid.
    pub fn contains_pk_value_temporal(
        &self,
        value: &Value,
        observation_ts: i64,
        create_col_idx: Option<usize>,
        end_col_idx: Option<usize>,
        scratch: &mut diskgraph_rowpack::RowPackDecodeScratch,
    ) -> bool {
        self.contains_pk_value_valid_between(
            value,
            observation_ts,
            observation_ts,
            create_col_idx,
            end_col_idx,
            scratch,
        )
    }

    /// [`contains_pk_value_temporal`] with separate create and end reference times.
    pub fn contains_pk_value_valid_between(
        &self,
        value: &Value,
        create_ref_ts: i64,
        end_ref_ts: i64,
        create_col_idx: Option<usize>,
        end_col_idx: Option<usize>,
        scratch: &mut diskgraph_rowpack::RowPackDecodeScratch,
    ) -> bool {
        let pk_map = match self.pk_map.as_ref() {
            Some(m) => m,
            None => return false,
        };
        let key = match Self::value_to_pk_key(value, pk_map.key_type()) {
            Some(k) => k,
            None => return false,
        };
        let node_id = match self.node_id_from_pk(&key) {
            Ok(Some(id)) => id,
            _ => return false,
        };
        // If no temporal columns, just PK existence is enough
        if create_col_idx.is_none() && end_col_idx.is_none() {
            return true;
        }
        self.is_node_valid_between(
            node_id,
            create_ref_ts,
            end_ref_ts,
            create_col_idx,
            end_col_idx,
            scratch,
        )
        .unwrap_or(true) // on error, conservatively include
    }
}

/// Convert a rowpack `ValueRef` representing a temporal column to epoch seconds.
fn temporal_value_to_epoch_secs(val: &diskgraph_rowpack::ValueRef<'_>) -> Option<i64> {
    match val {
        diskgraph_rowpack::ValueRef::Timestamp(secs) => Some(*secs),
        diskgraph_rowpack::ValueRef::Int64(ns) => Some(*ns / 1_000_000_000),
        diskgraph_rowpack::ValueRef::Date(days) => Some(*days as i64 * 86400),
        diskgraph_rowpack::ValueRef::Null => None,
        _ => None,
    }
}

fn load_pk_hash_table(
    root: &Path,
    node_type: &NodeTypeMetadata,
    pk: &PrimaryKeyMap,
) -> Result<PkHashTable> {
    let key_type = pkmap::key_type_from_metadata(pk)?;
    let hash_template = pk
        .hash_path
        .as_ref()
        .ok_or_else(|| anyhow!("missing pk hash path for {}", node_type.name))?;
    let mut shards = Vec::with_capacity(node_type.shards.len());
    for shard_idx in 0..node_type.shards.len() {
        let data_rel = pkmap::resolve_shard_path(&pk.mapping_path, shard_idx);
        let hash_rel = pkmap::resolve_shard_path(hash_template, shard_idx);
        let data_path = root.join(data_rel);
        let hash_path = root.join(hash_rel);
        let shard = pkmap::PkHashShard::load(&hash_path, &data_path, key_type)?;
        shards.push(shard);
    }
    Ok(PkHashTable::new(key_type, shards))
}

#[cfg(test)]
mod scratch_cleanup_tests {
    use super::*;

    #[test]
    fn sync_completed_file_preserves_contents() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("offsets.bin");
        let contents = b"completed shard offsets";
        fs::write(&path, contents).unwrap();

        assert_eq!(sync_file_len(&path).unwrap(), contents.len() as u64);
        assert_eq!(fs::read(&path).unwrap(), contents);
    }

    #[test]
    fn sync_missing_file_does_not_create_it() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("missing.bin");

        assert!(sync_file_len(&path).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn sync_directory_flushes_completed_entry() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("completed.bin");
        fs::write(&path, b"completed").unwrap();
        sync_file_len(&path).unwrap();

        sync_directory(root.path()).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"completed");
        assert!(sync_directory(&root.path().join("missing")).is_err());
    }

    #[test]
    fn committed_shard_requires_durable_marker_and_matching_files() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("nodes/users/shard-00000");
        fs::create_dir_all(&output).unwrap();
        let mut header = OffsetFileHeader::new(0, 1);
        header.version = LAYOUT_VERSION;
        fs::write(
            output.join(OFFSETS_FILENAME),
            bincode::serialize(&header).unwrap(),
        )
        .unwrap();
        fs::write(output.join(EDGES_FILENAME), b"").unwrap();
        fs::write(output.join("build_stats.json"), b"{}").unwrap();
        let marker = index_shard_commit_path(root.path(), "users", "shard-00000");
        assert!(
            !shard_output_is_committed(&output, &marker, Some("Users/shard-00000"), Some(0))
                .unwrap()
        );

        commit_shard_output(
            &output,
            &marker,
            "Users/shard-00000",
            0,
            root.path(),
            root.path(),
        )
        .unwrap();
        assert!(
            shard_output_is_committed(&output, &marker, Some("Users/shard-00000"), Some(0))
                .unwrap()
        );
        assert_eq!(
            committed_shard_expected_nodes(&output, &marker, Some("Users/shard-00000"))
                .unwrap(),
            Some(0)
        );
        fs::write(marker.with_extension("json.tmp"), b"provisional").unwrap();
        assert_eq!(
            validated_index_shard_commits(root.path()).unwrap(),
            vec!["Users/shard-00000"]
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            for name in [OFFSETS_FILENAME, EDGES_FILENAME, "build_stats.json"] {
                let committed_file = output.join(name);
                let contents = fs::read(&committed_file).unwrap();
                let external_file = root.path().join(format!("external-{name}"));
                fs::write(&external_file, &contents).unwrap();
                fs::remove_file(&committed_file).unwrap();
                symlink(&external_file, &committed_file).unwrap();
                assert!(validated_index_shard_commits(root.path()).is_err());
                fs::remove_file(&committed_file).unwrap();
                fs::write(&committed_file, contents).unwrap();
            }

            let real_output = root.path().join("real-shard-output");
            fs::rename(&output, &real_output).unwrap();
            symlink(&real_output, &output).unwrap();
            assert!(validated_index_shard_commits(root.path()).is_err());
            fs::remove_file(&output).unwrap();
            fs::rename(&real_output, &output).unwrap();

            let table_dir = root.path().join("nodes/users");
            let real_table_dir = root.path().join("real-users-table");
            fs::rename(&table_dir, &real_table_dir).unwrap();
            symlink(&real_table_dir, &table_dir).unwrap();
            assert!(validated_index_shard_commits(root.path()).is_err());
            fs::remove_file(&table_dir).unwrap();
            fs::rename(&real_table_dir, &table_dir).unwrap();
        }
        fs::write(output.join(EDGES_FILENAME), b"changed").unwrap();
        assert!(
            shard_output_is_committed(&output, &marker, Some("Users/shard-00000"), Some(0))
                .is_err()
        );
        assert!(
            committed_shard_expected_nodes(&output, &marker, Some("Users/shard-00000")).is_err()
        );
        assert!(validated_index_shard_commits(root.path()).is_err());
    }

    #[test]
    fn validated_commit_inventory_rejects_unexpected_entries() {
        let root = tempfile::tempdir().unwrap();
        let commit_root = root.path().join(INDEX_SHARD_COMMIT_DIR);
        fs::create_dir_all(&commit_root).unwrap();
        fs::write(commit_root.join("not-a-table"), b"unexpected").unwrap();

        assert!(validated_index_shard_commits(root.path()).is_err());
    }

    #[test]
    fn consumed_input_cleanup_handles_files_directories_and_duplicates() {
        let root = tempfile::tempdir().unwrap();
        let node_data = root.path().join("node_data.bin");
        let edge_shard = root.path().join("edge-shard");
        fs::create_dir(&edge_shard).unwrap();
        fs::write(&node_data, b"node").unwrap();
        fs::write(edge_shard.join("part-00000.bin"), b"edges").unwrap();

        let removed = remove_consumed_input_paths(
            &[node_data.clone(), edge_shard.clone(), node_data.clone()],
            root.path(),
        )
        .unwrap();

        assert_eq!(removed, 9);
        assert!(!node_data.exists());
        assert!(!edge_shard.exists());
    }
}
