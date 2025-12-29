use crate::builder::{ShardBuildParams, ShardBuilder, BuildStats};
use crate::layout::RelationLayout;
use crate::reader::ShardReader;
use crate::streaming::{EdgeShardInput, ParquetSource};
use crate::builder::{compute_relation_layouts, shard_node_count};
use anyhow::{anyhow, Context, Result};
use diskgraph_core::metadata::{DatasetMetadata, EdgeRelationMetadata, NodeShardMetadata, NodeTypeMetadata};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use serde_json;

pub struct GraphIndexBuilder {
    metadata: DatasetMetadata,
    source: GraphDataSource,
    output_root: PathBuf,
    max_workers: usize,
}

#[derive(Clone)]
pub enum GraphDataSource {
    Local { root: PathBuf },
    S3 {
        bucket: String,
        prefix: String,
        region: Option<String>,
        endpoint: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
        virtual_hosted_style_request: Option<bool>,
    },
    Gcs {
        bucket: String,
        prefix: String,
    },
    Adls {
        account: String,
        container: String,
        prefix: String,
        access_key: Option<String>,
        sas_key: Option<String>,
        bearer_token: Option<String>,
        client_id: Option<String>,
        client_secret: Option<String>,
        tenant_id: Option<String>,
        endpoint: Option<String>,
    },
}

#[derive(Clone, Debug)]
pub enum GraphBuildStage {
    Download {
        shard: Arc<str>,
        file: Arc<str>,
        index: usize,
        total: usize,
    },
    BuildShard {
        shard: Arc<str>,
    },
    Complete {
        shard: Arc<str>,
    },
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
}

impl GraphIndexBuilder {
    pub fn from_metadata_path(metadata_path: &Path, output_root: &Path) -> Result<Self> {
        let bytes = fs::read(metadata_path).context("read metadata.json")?;
        let metadata: DatasetMetadata =
            serde_json::from_slice(&bytes).context("parse metadata.json")?;
        let root = metadata_path
            .parent()
            .ok_or_else(|| anyhow!("metadata.json path has no parent"))?
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
            max_workers: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        }
    }

    pub fn with_max_workers(mut self, workers: usize) -> Self {
        self.max_workers = workers.max(1);
        self
    }

    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    pub fn build(&self, progress: Option<&dyn GraphProgressSink>) -> Result<GraphBuildOutput> {
        fs::create_dir_all(&self.output_root)?;
        let metadata_path = self.output_root.join("metadata.json");
        fs::write(metadata_path, serde_json::to_vec_pretty(&self.metadata)?)?;

        let jobs = self.prepare_jobs()?;
        let jobs = Arc::new(Mutex::new(jobs));
        let abort = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::scope(|scope| {
            for _ in 0..self.max_workers {
                let jobs = Arc::clone(&jobs);
                let tx = tx.clone();
                let abort = Arc::clone(&abort);
                scope.spawn(move || {
                    loop {
                        if abort.load(Ordering::Relaxed) {
                            break;
                        }
                        let job = {
                            let mut guard = jobs.lock().unwrap();
                            guard.pop()
                        };
                        let Some(job) = job else { break };
                        let result = job.run(progress, &abort);
                        if result.is_err() {
                            abort.store(true, Ordering::Relaxed);
                        }
                        let _ = tx.send((job.shard_key.clone(), result));
                    }
                });
            }
        });
        drop(tx);

        let mut shard_stats = HashMap::new();
        for (shard_key, result) in rx {
            let output = result?;
            shard_stats.insert(shard_key, output.build_stats);
        }
        Ok(GraphBuildOutput { shard_stats })
    }

    fn prepare_jobs(&self) -> Result<Vec<ShardBuildJob>> {
        let mut jobs = Vec::new();
        for node_type in &self.metadata.node_types {
            let relations: Vec<EdgeRelationMetadata> = self
                .metadata
                .edge_relations
                .iter()
                .filter(|rel| rel.src_type == node_type.name)
                .cloned()
                .collect();
            for (shard_idx, shard_meta) in node_type.shards.iter().enumerate() {
                let shard_key = format!("{}/shard-{:05}", node_type.name, shard_idx);
                let output_dir = self
                    .output_root
                    .join("nodes")
                    .join(&node_type.name)
                    .join(format!("shard-{:05}", shard_idx));
                let node_data = if node_type.has_node_data {
                    let path = shard_meta
                        .node_data_path
                        .clone()
                        .ok_or_else(|| anyhow!("node data path missing for {shard_key}"))?;
                    Some(self.resolve_parquet(&path))
                } else {
                    None
                };
                let edge_inputs = relations
                    .iter()
                    .map(|rel| {
                        let shard = rel
                            .shards
                            .get(shard_idx)
                            .ok_or_else(|| anyhow!("missing edge shard {} index {}", rel.name, shard_idx))?;
                        Ok(EdgeShardInput {
                            relation: rel.name.clone(),
                            source_shard_index: shard_idx,
                            parquet: self.resolve_parquet(&shard.path),
                            has_dest_sort_key: rel.dest_sort_key_bits.is_some(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let job = ShardBuildJob {
                    shard_key,
                    output_dir,
                    node_meta: node_type.clone(),
                    node_types: self.metadata.node_types.clone(),
                    relation_metas: relations.clone(),
                    shard_index: shard_idx,
                    node_data,
                    edge_inputs,
                    expected_nodes: shard_node_count(node_type, shard_idx)?,
                };
                jobs.push(job);
            }
        }
        Ok(jobs)
    }

    fn resolve_parquet(&self, rel_path: &str) -> ParquetSource {
        match &self.source {
            GraphDataSource::Local { root } => {
                ParquetSource::LocalFile(root.join(rel_path))
            }
            GraphDataSource::S3 {
                bucket,
                prefix,
                region,
                endpoint,
                access_key_id,
                secret_access_key,
                session_token,
                virtual_hosted_style_request,
            } => ParquetSource::S3(crate::streaming::S3Location {
                bucket: bucket.clone(),
                key: join_prefix(prefix, rel_path),
                region: region.clone(),
                endpoint: endpoint.clone(),
                access_key_id: access_key_id.clone(),
                secret_access_key: secret_access_key.clone(),
                session_token: session_token.clone(),
                virtual_hosted_style_request: *virtual_hosted_style_request,
            }),
            GraphDataSource::Gcs { bucket, prefix } => ParquetSource::Gcs(crate::streaming::GcsLocation {
                bucket: bucket.clone(),
                key: join_prefix(prefix, rel_path),
                service_account_path: None,
                service_account_key: None,
                application_credentials_path: None,
            }),
            GraphDataSource::Adls {
                account,
                container,
                prefix,
                access_key,
                sas_key,
                bearer_token,
                client_id,
                client_secret,
                tenant_id,
                endpoint,
            } => ParquetSource::Adls(crate::streaming::AdlsLocation {
                account: account.clone(),
                container: container.clone(),
                key: join_prefix(prefix, rel_path),
                access_key: access_key.clone(),
                sas_key: sas_key.clone(),
                bearer_token: bearer_token.clone(),
                client_id: client_id.clone(),
                client_secret: client_secret.clone(),
                tenant_id: tenant_id.clone(),
                endpoint: endpoint.clone(),
            }),
        }
    }
}

struct ShardBuildJob {
    shard_key: String,
    output_dir: PathBuf,
    node_meta: NodeTypeMetadata,
    node_types: Vec<NodeTypeMetadata>,
    relation_metas: Vec<EdgeRelationMetadata>,
    shard_index: usize,
    node_data: Option<ParquetSource>,
    edge_inputs: Vec<EdgeShardInput>,
    expected_nodes: u64,
}

impl ShardBuildJob {
    fn run(
        &self,
        progress: Option<&dyn GraphProgressSink>,
        abort: &AtomicBool,
    ) -> Result<crate::builder::ShardBuildOutput> {
        fs::create_dir_all(&self.output_dir)?;
        let params = ShardBuildParams {
            output_dir: self.output_dir.clone(),
            node_meta: self.node_meta.clone(),
            node_types: self.node_types.clone(),
            relation_metas: self.relation_metas.clone(),
            shard_index: self.shard_index,
            node_data: self.node_data.clone(),
        };
        let builder = ShardBuilder::new(params);
        let shard_key: Arc<str> = Arc::from(self.shard_key.as_str());
        let adapter = progress.map(|sink| ShardProgressAdapter::new(Arc::clone(&shard_key), sink));
        let result = builder.build_with_progress(
            &self.edge_inputs,
            Some(self.expected_nodes),
            adapter.as_ref().map(|a| a as &dyn crate::builder::ProgressSink),
            Some(&|| abort.load(Ordering::Relaxed)),
        );
        if let Some(sink) = progress {
            sink.report(GraphBuildProgress {
                stage: GraphBuildStage::Complete { shard: shard_key },
                current: 0,
                total: None,
            });
        }
        result
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
            crate::builder::BuildStage::Download { file, index, total } => {
                GraphBuildStage::Download {
                    shard: Arc::clone(&self.shard),
                    file,
                    index,
                    total,
                }
            }
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
        let bytes = fs::read(metadata_path).context("read metadata.json")?;
        let metadata: DatasetMetadata =
            serde_json::from_slice(&bytes).context("parse metadata.json")?;
        let mut tables = HashMap::new();
        for node_type in &metadata.node_types {
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
                    .join(&node_type.name)
                    .join(format!("shard-{:05}", shard_idx));
                let reader = ShardReader::open(&dir, Arc::clone(&layouts), node_type.has_node_data)?;
                shard_readers.push(reader);
            }
            let table = TableReader::new(node_type, relations, Arc::clone(&layouts), shard_readers)?;
            tables.insert(node_type.name.clone(), Arc::new(table));
        }
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
}

impl TableReader {
    fn new(
        node_type: &NodeTypeMetadata,
        relations: Vec<EdgeRelationMetadata>,
        layouts: Arc<Vec<RelationLayout>>,
        shards: Vec<ShardReader>,
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
        })
    }

    pub fn node_type(&self) -> &str {
        &self.node_type
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

    fn shard_for_node(&self, node_id: u64) -> Option<(usize, u64)> {
        for (idx, shard) in self.shard_ranges.iter().enumerate() {
            if node_id >= shard.min_id && node_id <= shard.max_id {
                return Some((idx, node_id - shard.min_id));
            }
        }
        None
    }
}

fn join_prefix(prefix: &str, path: &str) -> String {
    if prefix.is_empty() {
        return path.to_string();
    }
    let trimmed = prefix.trim_end_matches('/');
    let rel = path.trim_start_matches('/');
    format!("{}/{}", trimmed, rel)
}
