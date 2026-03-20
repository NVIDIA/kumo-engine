use arrow::pyarrow::ToPyArrow;
use arrow_array::builder::{ArrayBuilder, UInt64Builder};
use arrow_array::{Array, RecordBatch};
use crossbeam_channel::{bounded, Receiver, Sender};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use tracing::{debug, error, info};

use diskgraph_core::features::{BatchFeatureRequest, FeatureRequest};
use diskgraph_core::sampling::{BatchSamplingRequest, SamplingFlags, SamplingPlan};
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_engine::output::OutputHandler;
use diskgraph_engine::state::{SamplingState, StatePool};

mod split_assignments;
mod tt_builder;
mod tt_executor;

#[pyclass(eq, eq_int, name = "TimeUnit")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyTimeUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

#[pymethods]
impl PyTimeUnit {
    fn __hash__(&self) -> u64 {
        *self as u64
    }
}

impl From<PyTimeUnit> for diskgraph_core::time::TimeUnit {
    fn from(unit: PyTimeUnit) -> Self {
        match unit {
            PyTimeUnit::Seconds => diskgraph_core::time::TimeUnit::Seconds,
            PyTimeUnit::Milliseconds => diskgraph_core::time::TimeUnit::Milliseconds,
            PyTimeUnit::Microseconds => diskgraph_core::time::TimeUnit::Microseconds,
            PyTimeUnit::Nanoseconds => diskgraph_core::time::TimeUnit::Nanoseconds,
            PyTimeUnit::Minutes => diskgraph_core::time::TimeUnit::Minutes,
            PyTimeUnit::Hours => diskgraph_core::time::TimeUnit::Hours,
            PyTimeUnit::Days => diskgraph_core::time::TimeUnit::Days,
            PyTimeUnit::Weeks => diskgraph_core::time::TimeUnit::Weeks,
            PyTimeUnit::Months => diskgraph_core::time::TimeUnit::Months,
            PyTimeUnit::Years => diskgraph_core::time::TimeUnit::Years,
        }
    }
}

/// Events yielded by the sampling iterator.
#[derive(Debug)]
enum SampleEvent {
    EdgeBatch { rel_id: usize, batch: RecordBatch },
    NodeBatch { type_id: usize, batch: RecordBatch },
    FeatureBatch { type_id: usize, batch: RecordBatch },
    Stats { stats: String },
    Error { message: String },
}

/// Bridge between Rust Engine and Python Iterator.
struct PyOutputHandler {
    sender: Sender<SampleEvent>,
    edge_builders: HashMap<u16, (UInt64Builder, UInt64Builder)>,
    chunk_size: usize,
    relation_src_type: Vec<usize>,
    relation_dst_type: Vec<usize>,
    edge_max_ids: HashMap<u16, (u32, u32)>,
}

impl PyOutputHandler {
    fn new(
        sender: Sender<SampleEvent>,
        relation_src_type: Vec<usize>,
        relation_dst_type: Vec<usize>,
    ) -> Self {
        let chunk_size = std::env::var("EDGE_BATCH_SIZE_ROWS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64 * 1024);
        Self {
            sender,
            edge_builders: HashMap::new(),
            chunk_size,
            relation_src_type,
            relation_dst_type,
            edge_max_ids: HashMap::new(),
        }
    }

    fn flush_edge_builder(&mut self, rel_id: u16) -> diskgraph_engine::errors::Result<()> {
        let (src_b, dst_b) = if let Some(b) = self.edge_builders.get_mut(&rel_id) {
            b
        } else {
            return Ok(());
        };

        if src_b.is_empty() {
            return Ok(());
        }

        let len = src_b.len();
        let src_array = src_b.finish();
        let dst_array = dst_b.finish();

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("row_type", arrow_schema::DataType::Int8, false),
            arrow_schema::Field::new("id_a", arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new("id_b", arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new("meta_type", arrow_schema::DataType::UInt64, false),
        ]));

        let row_type = arrow_array::Int8Array::from(vec![0; len]);
        let meta_type = arrow_array::UInt64Array::from(vec![rel_id as u64; len]);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(row_type) as Arc<dyn Array>,
                Arc::new(src_array) as Arc<dyn Array>,
                Arc::new(dst_array) as Arc<dyn Array>,
                Arc::new(meta_type) as Arc<dyn Array>,
            ],
        )
        .map_err(|e| diskgraph_engine::errors::EngineError::Internal(e.to_string()))?;

        let _ = self.sender.send(SampleEvent::EdgeBatch {
            rel_id: rel_id as usize,
            batch,
        });
        Ok(())
    }
}

impl OutputHandler for PyOutputHandler {
    fn handle_edge(
        &mut self,
        _hop: usize,
        rel_id: u16,
        src_local_id: u32,
        dst_local_id: u32,
    ) -> diskgraph_engine::errors::Result<()> {
        let maxes = self.edge_max_ids.entry(rel_id).or_insert((0, 0));
        maxes.0 = maxes.0.max(src_local_id);
        maxes.1 = maxes.1.max(dst_local_id);

        let chunk_size = self.chunk_size;
        let (src_b, dst_b) = self.edge_builders.entry(rel_id).or_insert_with(|| {
            (
                UInt64Builder::with_capacity(chunk_size),
                UInt64Builder::with_capacity(chunk_size),
            )
        });

        src_b.append_value(src_local_id as u64);
        dst_b.append_value(dst_local_id as u64);

        if src_b.len() >= chunk_size {
            self.flush_edge_builder(rel_id)?;
        }
        Ok(())
    }

    fn handle_schema(
        &mut self,
        _schema: arrow_schema::SchemaRef,
    ) -> diskgraph_engine::errors::Result<()> {
        Ok(())
    }

    fn handle_feature_batch(&mut self, batch: RecordBatch) -> diskgraph_engine::errors::Result<()> {
        let _ = self
            .sender
            .send(SampleEvent::FeatureBatch { type_id: 0, batch });
        Ok(())
    }

    fn handle_fused_features(
        &mut self,
        type_id: usize,
        batch: RecordBatch,
    ) -> diskgraph_engine::errors::Result<()> {
        let _ = self
            .sender
            .send(SampleEvent::FeatureBatch { type_id, batch });
        Ok(())
    }

    fn finalize(&mut self, state: &SamplingState) -> diskgraph_engine::errors::Result<()> {
        // Validation logic
        for (&rel_id, &(max_src, max_dst)) in &self.edge_max_ids {
            let rel_idx = rel_id as usize;
            if let (Some(&src_type_id), Some(&dst_type_id)) = (
                self.relation_src_type.get(rel_idx),
                self.relation_dst_type.get(rel_idx),
            ) {
                let src_count = state
                    .type_states
                    .get(&src_type_id)
                    .map(|s| s.nodes.len())
                    .unwrap_or(0);
                let dst_count = state
                    .type_states
                    .get(&dst_type_id)
                    .map(|s| s.nodes.len())
                    .unwrap_or(0);

                if src_count > 0 && (max_src as usize) >= src_count {
                    return Err(diskgraph_engine::errors::EngineError::DataCorruption(
                        format!(
                            "invalid src_local_id for rel_id={}: max_src={} >= src_count={}",
                            rel_id, max_src, src_count
                        ),
                    ));
                }
                if dst_count > 0 && (max_dst as usize) >= dst_count {
                    return Err(diskgraph_engine::errors::EngineError::DataCorruption(
                        format!(
                            "invalid dst_local_id for rel_id={}: max_dst={} >= dst_count={}",
                            rel_id, max_dst, dst_count
                        ),
                    ));
                }
            }
        }

        let rel_ids: Vec<u16> = self.edge_builders.keys().cloned().collect();
        for rel_id in rel_ids {
            self.flush_edge_builder(rel_id)?;
        }

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("row_type", arrow_schema::DataType::Int8, false),
            arrow_schema::Field::new("id_a", arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new("id_b", arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new("meta_type", arrow_schema::DataType::UInt64, false),
        ]));

        for (type_id, type_state) in &state.type_states {
            if type_state.nodes.is_empty() {
                continue;
            }
            let len = type_state.nodes.len();
            let row_type = arrow_array::Int8Array::from(vec![1; len]);
            let id_a = arrow_array::UInt64Array::from_iter_values(
                type_state.nodes.iter().map(|n| n.seed_index as u64),
            );
            let id_b = arrow_array::UInt64Array::from_iter_values(
                type_state.nodes.iter().map(|n| n.global_id),
            );
            let meta_type = arrow_array::UInt64Array::from(vec![*type_id as u64; len]);

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(row_type) as Arc<dyn Array>,
                    Arc::new(id_a) as Arc<dyn Array>,
                    Arc::new(id_b) as Arc<dyn Array>,
                    Arc::new(meta_type) as Arc<dyn Array>,
                ],
            )
            .map_err(|e| diskgraph_engine::errors::EngineError::Internal(e.to_string()))?;

            let _ = self.sender.send(SampleEvent::NodeBatch {
                type_id: *type_id,
                batch,
            });
        }

        let stats_json = serde_json::to_string(&state.stats).unwrap_or_default();
        let _ = self.sender.send(SampleEvent::Stats { stats: stats_json });
        Ok(())
    }
}

#[pyclass]
pub struct SamplerEngine {
    engine: Arc<DiskGraphEngine>,
    pool: StatePool,
}

#[pymethods]
impl SamplerEngine {
    #[new]
    #[pyo3(signature = (path, random_seed=None))]
    pub fn new(path: String, random_seed: Option<u64>) -> PyResult<Self> {
        debug!(path = %path, random_seed = ?random_seed, "Initializing SamplerEngine");
        let engine = DiskGraphEngine::load(std::path::Path::new(&path), random_seed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        info!(path = %path, "SamplerEngine initialized");
        Ok(Self {
            engine: Arc::new(engine),
            pool: StatePool::new(),
        })
    }

    #[pyo3(signature = (seed_nodes, plan_json, fused, seed_timestamps=None, swap_edge_direction=None, seed_timestamp_unit=None))]
    pub fn sample(
        &self,
        py: Python<'_>,
        seed_nodes: Bound<'_, PyDict>,
        plan_json: String,
        fused: bool,
        seed_timestamps: Option<Bound<'_, PyDict>>,
        swap_edge_direction: Option<bool>,
        seed_timestamp_unit: Option<PyTimeUnit>,
    ) -> PyResult<PySampleIterator> {
        let mut seeds = HashMap::new();
        for (key, value) in seed_nodes.into_iter() {
            let type_name: String = key.extract()?;
            let ids: Vec<u64> = value.extract()?;
            seeds.insert(type_name, ids);
        }

        let mut timestamps = None;
        if let Some(st) = seed_timestamps {
            let mut ts_map = HashMap::new();
            for (key, value) in st.into_iter() {
                let type_name: String = key.extract()?;
                let ts: Vec<i64> = value.extract()?;
                ts_map.insert(type_name, ts);
            }
            timestamps = Some(ts_map);
        }

        let plan: SamplingPlan = serde_json::from_str(&plan_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid plan JSON: {}", e)))?;

        let flags = SamplingFlags {
            replace: false,
            directed: true,
            disjoint: true,
            return_edge_id: false,
            adaptive: false,
            fused_feature_fetching: fused,
            swap_edge_direction: swap_edge_direction.unwrap_or(true),
        };

        let parsed_unit = seed_timestamp_unit
            .map(|u| u.into())
            .unwrap_or(diskgraph_core::time::TimeUnit::Nanoseconds);

        let request = BatchSamplingRequest {
            dataset_id: uuid::Uuid::nil(),
            seed_nodes: seeds,
            seed_timestamps: timestamps,
            seed_timestamp_unit: parsed_unit,
            plan,
            flags,
        };

        let seed_type = request
            .seed_nodes
            .keys()
            .next()
            .cloned()
            .unwrap_or_default();
        let seed_count = request
            .seed_nodes
            .values()
            .next()
            .map_or(0usize, |v| v.len());
        info!(
            seed_type = %seed_type,
            seed_count,
            hops = request.plan.layers.len(),
            fused_feature_fetching = request.flags.fused_feature_fetching,
            swap_edge_direction = request.flags.swap_edge_direction,
            "Pybind sampling request started"
        );

        let (tx, rx) = bounded(1);
        let engine = Arc::clone(&self.engine);
        let mut state = self.pool.acquire();

        py.detach(move || {
            thread::spawn(move || {
                let mut handler = PyOutputHandler::new(
                    tx.clone(),
                    engine.relation_src_type.clone(),
                    engine.relation_info.clone(),
                );
                match engine.sample(&request, &mut state, &mut handler) {
                    Ok(()) => {
                        info!(
                            total_nodes = state.stats.total_nodes,
                            total_edges = state.stats.total_edges,
                            "Pybind sampling request finished"
                        );
                    }
                    Err(e) => {
                        error!(error = %e, "Pybind sampling request failed");
                        let _ = tx.send(SampleEvent::Error {
                            message: e.to_string(),
                        });
                    }
                }
            });
        });

        Ok(PySampleIterator { receiver: rx })
    }

    pub fn fetch_features(
        &self,
        py: Python<'_>,
        batch: Bound<'_, PyList>,
    ) -> PyResult<PySampleIterator> {
        let mut req_batch = Vec::new();
        for item in batch.iter() {
            let dict: Bound<'_, PyDict> = item.extract()?;
            let node_type: String = dict.get_item("node_type")?.expect("node_type").extract()?;
            let node_ids: Vec<u64> = dict.get_item("node_ids")?.expect("node_ids").extract()?;
            let feature_names: Vec<String> = dict
                .get_item("feature_names")?
                .expect("feature_names")
                .extract()?;

            req_batch.push(FeatureRequest {
                node_type,
                node_ids,
                feature_names,
            });
        }

        let request = BatchFeatureRequest {
            dataset_id: uuid::Uuid::nil(),
            batch: req_batch,
        };

        let req_count = request.batch.len();
        info!(requests = req_count, "Pybind feature request started");

        let (tx, rx) = bounded(1);
        let engine = Arc::clone(&self.engine);

        py.detach(move || {
            thread::spawn(move || {
                let mut handler = PyOutputHandler::new(tx.clone(), Vec::new(), Vec::new());

                for feature_req in request.batch {
                    let node_type_id = match engine.node_type_to_id.get(&feature_req.node_type) {
                        Some(&id) => id,
                        None => {
                            error!(node_type = %feature_req.node_type, "Unknown node type in pybind feature request");
                            let _ = tx.send(SampleEvent::Error {
                                message: format!("Unknown node type: {}", feature_req.node_type),
                            });
                            return;
                        }
                    };

                    if let Err(e) = engine.fetch_features_for_type(
                        node_type_id,
                        &feature_req.node_ids,
                        &feature_req.feature_names,
                        &mut handler,
                    ) {
                        error!(error = %e, node_type = %feature_req.node_type, "Pybind feature request failed");
                        let _ = tx.send(SampleEvent::Error {
                            message: e.to_string(),
                        });
                        return;
                    }
                }
                info!("Pybind feature request finished");
            });
        });

        Ok(PySampleIterator { receiver: rx })
    }

    pub fn get_type_mapping(&self) -> PyResult<String> {
        let mapping = self.engine.get_type_name_mapping();
        serde_json::to_string(&mapping).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass]
pub struct PySampleIterator {
    receiver: Receiver<SampleEvent>,
}

#[pymethods]
impl PySampleIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let event = py.detach(|| self.receiver.recv());

        match event {
            Ok(SampleEvent::EdgeBatch { rel_id, batch }) => {
                let dict = PyDict::new(py);
                dict.set_item("type", "edge")?;
                dict.set_item("rel_id", rel_id)?;
                dict.set_item("batch", batch.to_pyarrow(py)?)?;
                Ok(Some(dict.into()))
            }
            Ok(SampleEvent::NodeBatch { type_id, batch }) => {
                let dict = PyDict::new(py);
                dict.set_item("type", "node")?;
                dict.set_item("type_id", type_id)?;
                dict.set_item("batch", batch.to_pyarrow(py)?)?;
                Ok(Some(dict.into()))
            }
            Ok(SampleEvent::FeatureBatch { type_id, batch }) => {
                let dict = PyDict::new(py);
                dict.set_item("type", "feature")?;
                dict.set_item("type_id", type_id)?;
                dict.set_item("batch", batch.to_pyarrow(py)?)?;
                Ok(Some(dict.into()))
            }
            Ok(SampleEvent::Stats { stats }) => {
                let dict = PyDict::new(py);
                dict.set_item("type", "stats")?;
                dict.set_item("stats", stats)?;
                Ok(Some(dict.into()))
            }
            Ok(SampleEvent::Error { message }) => Err(PyRuntimeError::new_err(message)),
            Err(_) => Ok(None),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct FkeyConfig {
    #[pyo3(get, set)]
    pub column_name: String,
    #[pyo3(get, set)]
    pub target_table: String,
}

#[pymethods]
impl FkeyConfig {
    #[new]
    fn new(column_name: String, target_table: String) -> Self {
        Self {
            column_name,
            target_table,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ExportConfig {
    #[pyo3(get, set)]
    pub train_table_path: String,
    #[pyo3(get, set)]
    pub index_path: String,
    #[pyo3(get, set)]
    pub output_dir: String,
    #[pyo3(get, set)]
    pub row_idx_col: String,
    #[pyo3(get, set)]
    pub train_table_name: String,
    #[pyo3(get, set)]
    pub fkeys: Vec<FkeyConfig>,
}

#[pymethods]
impl ExportConfig {
    #[new]
    #[pyo3(signature = (train_table_path, index_path, output_dir, fkeys, row_idx_col="arange".to_string(), train_table_name="custom_train_table".to_string()))]
    fn new(
        train_table_path: String,
        index_path: String,
        output_dir: String,
        fkeys: Vec<FkeyConfig>,
        row_idx_col: String,
        train_table_name: String,
    ) -> Self {
        Self {
            train_table_path,
            index_path,
            output_dir,
            row_idx_col,
            train_table_name,
            fkeys,
        }
    }
}

#[pyfunction]
pub fn export_tt_artifacts(py: Python<'_>, config: &ExportConfig) -> PyResult<()> {
    info!(
        train_table_path = %config.train_table_path,
        index_path = %config.index_path,
        output_dir = %config.output_dir,
        fkeys = config.fkeys.len(),
        "Pybind export_tt_artifacts started"
    );
    let fkeys = config
        .fkeys
        .iter()
        .map(|f| diskgraph_engine::export_tt_artifacts::FkeyConfig {
            column_name: f.column_name.clone(),
            target_table: f.target_table.clone(),
        })
        .collect();

    let internal_config = diskgraph_engine::export_tt_artifacts::ExportConfig {
        train_table_path: config.train_table_path.clone(),
        index_path: config.index_path.clone(),
        output_dir: config.output_dir.clone(),
        row_idx_col: config.row_idx_col.clone(),
        train_table_name: config.train_table_name.clone(),
        fkeys,
    };

    // Release the GIL during export — this is a long-running CPU/IO operation
    // and holding the GIL would block heartbeats and other Python threads in the
    // Temporal worker.
    py.allow_threads(move || {
        diskgraph_engine::export_tt_artifacts::process_export(&internal_config)
    })
    .map(|_| {
        info!("Pybind export_tt_artifacts finished");
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (kumo_config_json, data_persist_dir, dataset_name, table_data_versions_json=None, source_paths_base=None, source_paths_json=None))]
pub fn convert_kumo_config(
    kumo_config_json: &str,
    data_persist_dir: &str,
    dataset_name: &str,
    table_data_versions_json: Option<&str>,
    source_paths_base: Option<&str>,
    source_paths_json: Option<&str>,
) -> PyResult<String> {
    info!(
        dataset_name = %dataset_name,
        data_persist_dir = %data_persist_dir,
        has_table_data_versions = table_data_versions_json.is_some(),
        has_source_paths_base = source_paths_base.is_some(),
        has_source_paths = source_paths_json.is_some(),
        "Pybind convert_kumo_config started"
    );
    use diskgraph_ingest::config_converter::{
        convert_kumo_to_diskgraph, ConvertOptions, KumoGraphConfig,
    };

    let kumo_config: KumoGraphConfig = serde_json::from_str(kumo_config_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid kumo config JSON: {e}")))?;

    let table_data_versions: Option<HashMap<String, String>> = match table_data_versions_json {
        Some(json) => Some(serde_json::from_str(json).map_err(|e| {
            PyValueError::new_err(format!("Invalid table_data_versions JSON: {e}"))
        })?),
        None => None,
    };

    let source_paths: Option<HashMap<String, String>> = match source_paths_json {
        Some(json) => Some(serde_json::from_str(json).map_err(|e| {
            PyValueError::new_err(format!("Invalid source_paths JSON: {e}"))
        })?),
        None => None,
    };

    let options = ConvertOptions {
        data_persist_dir: data_persist_dir.to_string(),
        dataset_name: dataset_name.to_string(),
        table_data_versions,
        source_paths_base: source_paths_base.map(|s| s.to_string()),
        source_paths,
    };

    // convert_kumo_to_diskgraph handles everything: schema, timestamp detection, num_shards.
    let mut diskgraph_config = convert_kumo_to_diskgraph(&kumo_config, &options)
        .map_err(|e| PyRuntimeError::new_err(format!("Config conversion failed: {e}")))?;

    // Expand cloud directory prefixes (s3://, gs://, az://) to individual
    // parquet file paths.  The converter outputs fpv3 directory paths but
    // ingest() expects individual file paths (it does HEAD per path).
    diskgraph_ingest::io::expand_cloud_paths_in_config(&mut diskgraph_config)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to expand cloud paths: {e}")))?;

    serde_json::to_string(&diskgraph_config)
        .map(|s| {
            info!("Pybind convert_kumo_config finished");
            s
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize config: {e}")))
}

#[pyfunction]
#[pyo3(signature = (config_json, output_dir, col_stats=true, cat_count_strategy="hybrid"))]
pub fn ingest(
    py: Python<'_>,
    config_json: &str,
    output_dir: &str,
    col_stats: bool,
    cat_count_strategy: &str,
) -> PyResult<String> {
    info!(
        output_dir = %output_dir,
        col_stats,
        cat_count_strategy = %cat_count_strategy,
        "Pybind ingest started"
    );
    let source: diskgraph_ingest::GraphConfig = serde_json::from_str(config_json)
        .map_err(|e| PyValueError::new_err(format!("Invalid config JSON: {e}")))?;

    let strategy = match cat_count_strategy {
        "hybrid" => diskgraph_ingest::col_stats::CategoryCountStrategy::Hybrid,
        "graduation" => diskgraph_ingest::col_stats::CategoryCountStrategy::Graduation,
        "none" => diskgraph_ingest::col_stats::CategoryCountStrategy::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown cat_count_strategy: {other}"
            )))
        }
    };

    let options = diskgraph_ingest::IngestOptions {
        profile_dir: None,
        progress: None,
        parallel: true,
        index_mem_budget_bytes: None,
        peak_mem_stats: None,
        col_stats,
        cat_count_strategy: strategy,
        row_subsample: false,
    };

    let output_dir = output_dir.to_string();

    // Release the GIL during ingest — this is a long-running CPU/IO operation
    // and holding the GIL would block heartbeats and other activities in the
    // Temporal worker.
    let result = py.allow_threads(move || {
        diskgraph_ingest::ingest_with_stats_from_source(
            source,
            None,
            &output_dir,
            diskgraph_ingest::Dest::Local,
            options,
        )
    });

    let (_metadata, stats) =
        result.map_err(|e| PyRuntimeError::new_err(format!("Ingest failed: {e}")))?;

    serde_json::to_string(&stats)
        .map(|s| {
            info!(
                total_nodes = stats.node_counts.values().copied().sum::<u64>(),
                total_edges = stats.edge_counts.values().copied().sum::<u64>(),
                "Pybind ingest finished"
            );
            s
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to serialize stats: {e}")))
}

#[pymodule]
#[pyo3(name = "_c")]
fn diskgraph_pybind(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(export_tt_artifacts, m)?)?;
    m.add_function(wrap_pyfunction!(convert_kumo_config, m)?)?;
    m.add_function(wrap_pyfunction!(ingest, m)?)?;
    m.add_class::<ExportConfig>()?;
    m.add_class::<FkeyConfig>()?;
    m.add_class::<PyTimeUnit>()?;
    m.add_class::<SamplerEngine>()?;
    m.add_class::<PySampleIterator>()?;
    m.add_class::<tt_builder::DiskGraphTTBuilder>()?;
    m.add_class::<tt_builder::EdgeRef>()?;
    m.add_class::<tt_builder::RowFilter>()?;
    m.add_class::<tt_builder::StageFilter>()?;
    m.add_function(wrap_pyfunction!(tt_executor::execute_tt_plan, m)?)?;
    m.add_function(wrap_pyfunction!(tt_executor::execute_tt_plan_materialize, m)?)?;
    m.add_function(wrap_pyfunction!(tt_executor::execute_tt_plan_materialize_with_split, m)?)?;
    m.add_function(wrap_pyfunction!(tt_executor::build_tt_stats_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(tt_executor::build_split_stats_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(tt_executor::persist_tt_artifact_bundle, m)?)?;
    m.add_class::<tt_executor::TTPlanIterator>()?;
    m.add_function(wrap_pyfunction!(split_assignments::assign_by_ratio_on_perm, m)?)?;
    m.add_function(wrap_pyfunction!(split_assignments::compute_dense_split_assignments_core, m)?)?;
    m.add_function(wrap_pyfunction!(split_assignments::apply_entity_include_mask, m)?)?;
    m.add_function(wrap_pyfunction!(split_assignments::assign_manual_split, m)?)?;
    m.add_function(wrap_pyfunction!(split_assignments::assign_backtesting_split, m)?)?;
    m.add_function(wrap_pyfunction!(split_assignments::assign_approx_date_offset_split, m)?)?;
    m.add_function(wrap_pyfunction!(split_assignments::assign_date_offset_split, m)?)?;
    m.add_function(wrap_pyfunction!(split_assignments::assign_time_range_split, m)?)?;
    Ok(())
}
