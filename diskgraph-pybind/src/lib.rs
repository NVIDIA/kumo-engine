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

use diskgraph_core::features::{BatchFeatureRequest, FeatureRequest};
use diskgraph_core::sampling::{BatchSamplingRequest, SamplingFlags, SamplingPlan};
use diskgraph_engine::engine::DiskGraphEngine;
use diskgraph_engine::output::OutputHandler;
use diskgraph_engine::state::{SamplingState, StatePool};

mod tt_builder;
mod tt_executor;

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
    pub fn new(path: String) -> PyResult<Self> {
        let engine = DiskGraphEngine::load(std::path::Path::new(&path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            engine: Arc::new(engine),
            pool: StatePool::new(),
        })
    }

    #[pyo3(signature = (seed_nodes, plan_json, fused, seed_timestamps=None, swap_edge_direction=None))]
    pub fn sample(
        &self,
        py: Python<'_>,
        seed_nodes: Bound<'_, PyDict>,
        plan_json: String,
        fused: bool,
        seed_timestamps: Option<Bound<'_, PyDict>>,
        swap_edge_direction: Option<bool>,
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

        let request = BatchSamplingRequest {
            dataset_id: uuid::Uuid::nil(),
            seed_nodes: seeds,
            seed_timestamps: timestamps,
            plan,
            flags,
        };

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
                if let Err(e) = engine.sample(&request, &mut state, &mut handler) {
                    let _ = tx.send(SampleEvent::Error {
                        message: e.to_string(),
                    });
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

        let (tx, rx) = bounded(1);
        let engine = Arc::clone(&self.engine);

        py.detach(move || {
            thread::spawn(move || {
                let mut handler = PyOutputHandler::new(tx.clone(), Vec::new(), Vec::new());

                for feature_req in request.batch {
                    let node_type_id = match engine.node_type_to_id.get(&feature_req.node_type) {
                        Some(&id) => id,
                        None => {
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
                        let _ = tx.send(SampleEvent::Error {
                            message: e.to_string(),
                        });
                        return;
                    }
                }
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

#[pymodule]
#[pyo3(name = "_c")]
fn diskgraph_pybind(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SamplerEngine>()?;
    m.add_class::<PySampleIterator>()?;
    m.add_class::<tt_builder::DiskGraphTTBuilder>()?;
    m.add_class::<tt_builder::EdgeRef>()?;
    m.add_class::<tt_builder::RowFilter>()?;
    m.add_class::<tt_builder::StageFilter>()?;
    m.add_function(wrap_pyfunction!(tt_executor::execute_tt_plan, m)?)?;
    Ok(())
}
