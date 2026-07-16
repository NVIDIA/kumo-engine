use crate::builder::{
    compute_relation_layouts, estimate_edge_record_bytes, shard_node_count, BuildStats,
    EdgeBufferPool, MemoryBudget, ShardBuildParams, ShardBuilder,
};
use crate::layout::RelationLayout;
use crate::pkmap::{self, Key, PkHashTable};
use crate::reader::ShardReader;
use crate::streaming::EdgeShardInput;
use anyhow::{anyhow, Context, Result};
use diskgraph_core::metadata::{
    DatasetMetadata, EdgeRelationMetadata, NodeDataCompression, NodeShardMetadata,
    NodeTypeMetadata, PrimaryKeyMap,
};
use diskgraph_core::tt_builder::Value;
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const TEMP_META_MAGIC: [u8; 4] = *b"DGTP";
const TEMP_META_VERSION: u8 = 2;

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
        let mut jobs = Vec::new();
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
                let output_dir = Arc::new(
                    self.output_root
                        .join("nodes")
                        .join(node_type.name.to_lowercase())
                        .join(format!("shard-{:05}", shard_idx)),
                );
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
                    expected_nodes: shard_node_count(node_type, shard_idx)?,
                    edge_buffer_pool: edge_buffer_pool.clone(),
                };
                jobs.push(job);
            }
        }
        Ok(jobs)
    }

    fn resolve_path(&self, rel_path: &str) -> PathBuf {
        match &self.source {
            GraphDataSource::Local { root } => root.join(rel_path),
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
}

impl ShardBuildJob {
    pub fn shard_key(&self) -> &str {
        self.shard_key.as_ref()
    }

    pub fn estimated_edge_record_bytes(&self) -> Result<u64> {
        estimate_edge_record_bytes(self.edge_inputs.as_ref())
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
        let result = builder.build_with_progress(
            &self.edge_inputs,
            Some(self.expected_nodes),
            adapter
                .as_ref()
                .map(|a| a as &dyn crate::builder::ProgressSink),
            None,
        );
        if let Some(sink) = progress {
            sink.report(GraphBuildProgress {
                stage: GraphBuildStage::Complete {
                    shard: Arc::clone(&self.shard_key),
                },
                current: 0,
                total: None,
            });
        }
        result.map(|mut output| {
            output.wall_ms = start.elapsed().as_millis() as u64;
            output
        })
    }
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

        // Check create_time_col <= observation_ts
        if let Some(col_idx) = create_col_idx {
            let val = decoder.decode_column_value(raw, col_idx, scratch)?;
            let ts_secs = temporal_value_to_epoch_secs(&val);
            if let Some(secs) = ts_secs {
                if secs > observation_ts {
                    return Ok(false);
                }
            }
            // NULL create_time → conservatively include
        }

        // Check end_time_col IS NULL OR end_time_col > observation_ts
        if let Some(col_idx) = end_col_idx {
            let val = decoder.decode_column_value(raw, col_idx, scratch)?;
            let ts_secs = temporal_value_to_epoch_secs(&val);
            if let Some(secs) = ts_secs {
                if secs <= observation_ts {
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
        self.is_node_temporally_valid(
            node_id,
            observation_ts,
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
