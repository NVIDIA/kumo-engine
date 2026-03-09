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
        }
    }

    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
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
        let node_types = Arc::new(self.metadata.node_types.clone());
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
}

impl GraphIndexReader {
    pub fn open(metadata_path: &Path, root: &Path) -> Result<Self> {
        let bytes = fs::read(metadata_path).context("read index_metadata.json")?;
        let metadata: DatasetMetadata =
            serde_json::from_slice(&bytes).context("parse index_metadata.json")?;
        let mut tables = HashMap::new();
        tracing::info!(
            node_types = metadata.node_types.len(),
            edge_relations = metadata.edge_relations.len(),
            "Loaded index metadata"
        );
        for node_type in &metadata.node_types {
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
            let layouts = compute_relation_layouts(&relations, &metadata.node_types)?;
            let layouts = Arc::new(layouts);
            let mut shard_readers = Vec::new();
            for (shard_idx, _shard_meta) in node_type.shards.iter().enumerate() {
                let dir = root
                    .join("nodes")
                    .join(node_type.name.to_lowercase())
                    .join(format!("shard-{:05}", shard_idx));
                let reader =
                    ShardReader::open(&dir, Arc::clone(&layouts), node_type.has_node_data)?;
                shard_readers.push(reader);
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
        tracing::info!(tables = tables.len(), "GraphIndexReader open completed");
        Ok(Self { metadata, tables })
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
}

pub struct TableReader {
    node_type: String,
    relations: Vec<EdgeRelationMetadata>,
    layouts: Arc<Vec<RelationLayout>>,
    shards: Vec<ShardReader>,
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
        shards: Vec<ShardReader>,
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
        self.shards.get(index)
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

    fn validate_pk_at_node(&self, node_id: u64, key: &Key) -> Result<Option<bool>> {
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
