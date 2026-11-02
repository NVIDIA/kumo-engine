// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-independent Arrow components shared by native and Python consumers.
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema};
use diskgraph_engine::output::{GraphChannel, RuntimeBatch};
use serde_json::{json, Value};
use std::sync::Arc;

/// A complete minibatch represented by Arrow components and their output layout.
///
/// Local consumers construct this directly from RuntimeBatch. Flight transports
/// the same components and layout; its client validates references before
/// returning this representation. Neither construction nor reference resolution
/// serializes Arrow data or applies a transport byte limit.
#[derive(Debug)]
pub struct ArrowBatch {
    /// Native output structure, with Arrow values replaced by column/range references.
    pub layout: Value,
    /// Record batches with independent schemas and row counts, in layout order.
    pub components: Vec<RecordBatch>,
}

impl TryFrom<RuntimeBatch> for ArrowBatch {
    type Error = anyhow::Error;

    /// Consume a native batch, moving owned vectors into Arrow arrays.
    ///
    /// Feature and target buffers are shared. Returns an error when fields
    /// grouped into one component disagree on their row counts.
    fn try_from(batch: RuntimeBatch) -> anyhow::Result<Self> {
        let mut encoder = BatchEncoder::default();
        let layout = encoder.batch(batch)?;
        Ok(Self {
            layout,
            components: encoder.parts,
        })
    }
}

#[derive(Default)]
struct BatchEncoder {
    parts: Vec<RecordBatch>,
}
impl BatchEncoder {
    fn record_batch(&mut self, batch: RecordBatch) -> Value {
        let slot = self.parts.len();
        let num_columns = batch.num_columns();
        self.parts.push(batch);
        json!({"component":slot,"columns":[0,num_columns]})
    }
    fn array(&mut self, array: impl Into<ArrayRef>) -> anyhow::Result<Value> {
        let array = array.into();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            array.data_type().clone(),
            true,
        )]));
        let slot = self.parts.len();
        self.parts.push(RecordBatch::try_new(schema, vec![array])?);
        Ok(json!({"component":slot,"column":0}))
    }
    fn group(
        &mut self,
        columns: Vec<(&str, ArrayRef)>,
        target: Option<&RecordBatch>,
    ) -> anyhow::Result<serde_json::Map<String, Value>> {
        let slot = self.parts.len();
        let mut references = serde_json::Map::new();
        let mut fields =
            Vec::with_capacity(columns.len() + target.map_or(0, RecordBatch::num_columns));
        let mut arrays = Vec::with_capacity(fields.capacity());
        for (index, (name, array)) in columns.into_iter().enumerate() {
            references.insert(name.into(), json!({"component":slot,"column":index}));
            fields.push(Arc::new(Field::new(name, array.data_type().clone(), true)));
            arrays.push(array);
        }
        let mut schema_metadata = Default::default();
        if let Some(target) = target {
            anyhow::ensure!(
                arrays
                    .first()
                    .is_some_and(|array| array.len() == target.num_rows()),
                "Target row count does not match seed rows"
            );
            let start = arrays.len();
            fields.extend(target.schema().fields().iter().cloned());
            arrays.extend(target.columns().iter().cloned());
            schema_metadata = target.schema().metadata().clone();
            references.insert(
                "target".into(),
                json!({"component":slot,"columns":[start,arrays.len()]}),
            );
        }
        let schema = Arc::new(Schema::new_with_metadata(fields, schema_metadata));
        self.parts.push(RecordBatch::try_new(schema, arrays)?);
        Ok(references)
    }
    fn batch(&mut self, batch: RuntimeBatch) -> anyhow::Result<Value> {
        use arrow_array::{BooleanArray, Float32Array, Int64Array, UInt32Array, UInt64Array};
        let task = batch.task;
        let mut seeds = vec![
            (
                "input_ids",
                Arc::new(UInt32Array::from(task.input_ids)) as ArrayRef,
            ),
            (
                "tt_stage_node_ids",
                Arc::new(UInt64Array::from(task.tt_stage_node_ids)) as ArrayRef,
            ),
            (
                "seed_node_ids",
                Arc::new(UInt64Array::from(task.seed_node_ids)) as ArrayRef,
            ),
        ];
        if let Some(values) = task.seed_timestamps {
            seeds.push(("seed_timestamps", Arc::new(Int64Array::from(values))));
        }
        if let Some(values) = task.weight {
            seeds.push(("weight", Arc::new(Float32Array::from(values))));
        }
        let has_positives = !task.lp_pos_counts.is_empty();
        if has_positives {
            seeds.push((
                "lp_pos_counts",
                Arc::new(UInt32Array::from(task.lp_pos_counts)),
            ));
        }
        if !task.lp_exclude_counts.is_empty() {
            seeds.push((
                "lp_exclude_counts",
                Arc::new(UInt32Array::from(task.lp_exclude_counts)),
            ));
        }
        let mut t = self.group(seeds, task.target.as_ref())?;
        t.insert(
            "neg_mask".into(),
            self.array(Arc::new(BooleanArray::from(task.neg_mask)) as ArrayRef)?,
        );
        t.insert("amount_m".into(), json!(task.amount_m));
        t.insert("num_positives".into(), json!(task.num_positives));
        t.insert("rhs_seed_type".into(), json!(task.rhs_seed_type));
        let mut positives: Vec<(&str, ArrayRef)> = Vec::new();
        if has_positives {
            positives.push((
                "lp_pos_rhs_global",
                Arc::new(Int64Array::from(task.lp_pos_rhs_global)),
            ));
            positives.push((
                "lp_pos_mask",
                Arc::new(BooleanArray::from(task.lp_pos_mask)),
            ));
        }
        if let Some(values) = task.lp_pos_weight {
            positives.push(("lp_pos_weight", Arc::new(Float32Array::from(values))));
        }
        if let Some(values) = task.lp_pos_local {
            positives.push(("lp_pos_local", Arc::new(UInt32Array::from(values))));
        }
        if !positives.is_empty() {
            t.extend(self.group(positives, None)?);
        }
        if t.contains_key("lp_exclude_counts") {
            t.insert(
                "lp_exclude_rhs_global".into(),
                self.array(Arc::new(Int64Array::from(task.lp_exclude_rhs_global)) as ArrayRef)?,
            );
        }
        let mut graphs = serde_json::Map::new();
        for (name, channel) in batch.graphs {
            graphs.insert(name, self.graph(channel)?);
        }
        let mut features = serde_json::Map::new();
        for (name, bank) in batch.raw_features {
            let num_rows = bank
                .batches
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>();
            let records = bank
                .batches
                .into_iter()
                .map(|batch| self.record_batch(batch))
                .collect::<Vec<_>>();
            features.insert(
                name.clone(),
                json!({"key":name,"num_rows":num_rows,"record_batches":records}),
            );
        }
        Ok(json!({"task":t,"graphs":graphs,"raw_features":features}))
    }
    fn graph(&mut self, channel: GraphChannel) -> anyhow::Result<Value> {
        use arrow_array::{UInt32Array, UInt64Array};
        let mut nodes = serde_json::Map::new();
        let mut batches = serde_json::Map::new();
        let mut assignments = channel.batch_dict;
        for (name, ids) in channel.node_ids {
            let mut columns = vec![("node_ids", Arc::new(UInt64Array::from(ids)) as ArrayRef)];
            if let Some(ids) = assignments.remove(&name) {
                columns.push(("batch_dict", Arc::new(UInt32Array::from(ids))));
            }
            let mut references = self.group(columns, None)?;
            nodes.insert(name.clone(), references.remove("node_ids").unwrap());
            if let Some(reference) = references.remove("batch_dict") {
                batches.insert(name, reference);
            }
        }
        anyhow::ensure!(
            assignments.is_empty(),
            "Seed assignments have no matching node IDs"
        );
        let mut counts = serde_json::Map::new();
        for (name, ids) in channel.num_sampled_nodes {
            counts.insert(
                name.clone(),
                self.array(Arc::new(UInt64Array::from(ids)) as ArrayRef)?,
            );
        }
        let mut edges = Vec::new();
        for (id, (src, dst)) in channel.edges.into_iter().enumerate() {
            anyhow::ensure!(src.len() == dst.len(), "Edge endpoint row counts differ");
            if src.is_empty() {
                continue;
            }
            let (s, r, d) = channel
                .rel_id_to_names
                .get(&(id as u16))
                .cloned()
                .unwrap_or_else(|| ("?".into(), format!("rel_{id}"), "?".into()));
            let mut references = self.group(
                vec![
                    ("src", Arc::new(UInt32Array::from(src)) as ArrayRef),
                    ("dst", Arc::new(UInt32Array::from(dst)) as ArrayRef),
                ],
                None,
            )?;
            edges.push(json!({"edge_type":{"src":s,"rel":r,"dst":d},
                "src":references.remove("src").unwrap(),
                "dst":references.remove("dst").unwrap()}));
        }
        let mut edge_counts = Vec::new();
        for ((s, r, d), values) in channel.num_sampled_edges {
            edge_counts.push(json!({"edge_type":{"src":s,"rel":r,"dst":d},"counts":self.array(Arc::new(UInt64Array::from(values)) as ArrayRef)?}));
        }
        Ok(
            json!({"node_ids":nodes,"batch_dict":batches,"edge_index_dict":edges,"num_sampled_nodes_dict":counts,"num_sampled_edges_dict":edge_counts}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int64Array;
    use diskgraph_engine::output::{RawFeatureBank, TaskBag};
    use std::collections::HashMap;

    #[test]
    fn complete_batch_preserves_task_graph_and_feature_payloads() {
        let target = RecordBatch::try_from_iter(vec![(
            "target",
            Arc::new(Int64Array::from(vec![7])) as ArrayRef,
        )])
        .unwrap();
        let mut batch = RuntimeBatch {
            task: TaskBag {
                input_ids: vec![0],
                tt_stage_node_ids: vec![13],
                seed_node_ids: vec![5],
                target: Some(target.clone()),
                num_positives: 1,
                lp_pos_counts: vec![1],
                lp_pos_rhs_global: vec![9],
                lp_pos_mask: vec![false],
                ..Default::default()
            },
            ..Default::default()
        };
        batch.graphs.insert(
            "main".into(),
            GraphChannel {
                node_ids: HashMap::from([("users".into(), vec![5])]),
                ..Default::default()
            },
        );
        batch.raw_features.insert(
            "users".into(),
            RawFeatureBank {
                schema: Some(target.schema()),
                batches: vec![target.clone()],
            },
        );
        let ArrowBatch {
            layout: value,
            components: parts,
        } = ArrowBatch::try_from(batch).unwrap();
        assert_eq!(value["task"]["num_positives"], 1);
        assert!(value["task"]["lp_pos_counts"].is_object());
        assert!(value["graphs"]["main"]["node_ids"]["users"].is_object());
        assert_eq!(value["raw_features"]["users"]["num_rows"], 1);
        let crate::ArrowValue::RecordBatch(decoded) =
            crate::resolve_reference(&value["task"]["target"], &parts)
                .unwrap()
                .unwrap()
        else {
            panic!("Expected target record batch")
        };
        assert_eq!(decoded, target);
    }

    #[test]
    fn groups_seed_node_and_edge_columns_and_preserves_target_schema() {
        use arrow_array::StringArray;
        let target = RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(
                vec![
                    Field::new("seed_node_ids", arrow_schema::DataType::Int64, false)
                        .with_metadata([("role".into(), "label".into())].into()),
                    Field::new("TARGET2", arrow_schema::DataType::Utf8, true),
                ],
                [("origin".into(), "target_schema".into())].into(),
            )),
            vec![
                Arc::new(Int64Array::from(vec![1, 0])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("yes"), None])) as ArrayRef,
            ],
        )
        .unwrap();
        let features = RecordBatch::try_from_iter([(
            "age",
            Arc::new(Int64Array::from(vec![30, 40, 25])) as ArrayRef,
        )])
        .unwrap();
        let batch = RuntimeBatch {
            task: TaskBag {
                input_ids: vec![0, 1],
                tt_stage_node_ids: vec![100, 101],
                seed_node_ids: vec![10, 20],
                seed_timestamps: Some(vec![1000, 2000]),
                weight: Some(vec![1.0, 0.5]),
                target: Some(target.clone()),
                num_positives: 2,
                ..Default::default()
            },
            graphs: [(
                "main".into(),
                GraphChannel {
                    node_ids: [("users".into(), vec![10, 20, 30])].into(),
                    batch_dict: [("users".into(), vec![0, 1, 0])].into(),
                    num_sampled_nodes: [("users".into(), vec![2, 1])].into(),
                    edges: vec![(vec![2], vec![0])],
                    rel_id_to_names: [(0, ("users".into(), "follows".into(), "users".into()))]
                        .into(),
                    num_sampled_edges: [(
                        ("users".into(), "follows".into(), "users".into()),
                        vec![1],
                    )]
                    .into(),
                },
            )]
            .into(),
            raw_features: [(
                "users".into(),
                RawFeatureBank {
                    schema: Some(features.schema()),
                    batches: vec![features],
                },
            )]
            .into(),
        };
        let ArrowBatch {
            layout,
            components: parts,
        } = ArrowBatch::try_from(batch).unwrap();
        assert_eq!(parts.len(), 7);
        assert_eq!(
            layout["task"]["target"],
            json!({"component":0,"columns":[5,7]})
        );
        let crate::ArrowValue::RecordBatch(decoded) =
            crate::resolve_reference(&layout["task"]["target"], &parts)
                .unwrap()
                .unwrap()
        else {
            panic!("Expected target record batch")
        };
        assert_eq!(decoded, target);
        assert_eq!(decoded.schema(), target.schema());
        assert!(Arc::ptr_eq(decoded.column(0), target.column(0)));
        let graph = &layout["graphs"]["main"];
        assert_eq!(
            graph["node_ids"]["users"]["component"],
            graph["batch_dict"]["users"]["component"]
        );
        assert_eq!(
            graph["edge_index_dict"][0]["src"]["component"],
            graph["edge_index_dict"][0]["dst"]["component"]
        );
        assert_ne!(
            graph["num_sampled_nodes_dict"]["users"]["component"],
            graph["num_sampled_edges_dict"][0]["counts"]["component"]
        );
        crate::layout::validate_layout(&layout, &parts).unwrap();
    }

    #[test]
    fn ragged_link_labels_keep_their_own_row_domain() {
        let batch = RuntimeBatch {
            task: TaskBag {
                input_ids: vec![0, 1],
                tt_stage_node_ids: vec![100, 101],
                seed_node_ids: vec![10, 20],
                num_positives: 2,
                amount_m: 2,
                neg_mask: vec![true, true, false, true],
                lp_pos_counts: vec![2, 1],
                lp_pos_rhs_global: vec![5, 6, 7],
                lp_pos_mask: vec![false, false, true],
                lp_pos_weight: Some(vec![1.0, 0.5, 0.2]),
                lp_pos_local: Some(vec![0, 1, 2]),
                lp_exclude_counts: vec![1, 0],
                lp_exclude_rhs_global: vec![9],
                ..Default::default()
            },
            ..Default::default()
        };
        let ArrowBatch {
            layout,
            components: parts,
        } = ArrowBatch::try_from(batch).unwrap();
        assert_eq!(
            parts.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
            vec![2, 4, 3, 1]
        );
        assert_eq!(parts[0].num_columns(), 5);
        assert_eq!(parts[2].num_columns(), 4);
        let task = &layout["task"];
        assert_eq!(
            task["lp_pos_counts"]["component"],
            task["input_ids"]["component"]
        );
        assert_eq!(
            task["lp_exclude_counts"]["component"],
            task["input_ids"]["component"]
        );
        for name in [
            "lp_pos_rhs_global",
            "lp_pos_mask",
            "lp_pos_weight",
            "lp_pos_local",
        ] {
            assert_eq!(task[name]["component"], 2);
        }
        crate::layout::validate_layout(&layout, &parts).unwrap();
    }

    #[test]
    fn inconsistent_row_alignment_is_rejected() {
        let mut encoder = BatchEncoder::default();
        let target = RecordBatch::try_from_iter([(
            "target",
            Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
        )])
        .unwrap();
        let batch = RuntimeBatch {
            task: TaskBag {
                input_ids: vec![0],
                tt_stage_node_ids: vec![0],
                seed_node_ids: vec![0],
                target: Some(target),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(encoder
            .batch(batch)
            .unwrap_err()
            .to_string()
            .contains("Target row count"));
        let graph = GraphChannel {
            node_ids: [("users".into(), vec![0, 1])].into(),
            batch_dict: [("users".into(), vec![0])].into(),
            ..Default::default()
        };
        assert!(encoder.graph(graph).is_err());
        let graph = GraphChannel {
            edges: vec![(vec![0], vec![])],
            ..Default::default()
        };
        assert!(encoder
            .graph(graph)
            .unwrap_err()
            .to_string()
            .contains("Edge endpoint"));
    }
    #[test]
    fn conversion_moves_vectors_and_shares_feature_chunks() {
        use arrow_array::UInt64Array;
        let seed_ids = vec![10_u64, 20];
        let seed_buffer = seed_ids.as_ptr();
        let feature_chunk = RecordBatch::try_from_iter([(
            "age",
            Arc::new(Int64Array::from(vec![30, 40])) as ArrayRef,
        )])
        .unwrap();
        let feature_buffer = feature_chunk.column(0).clone();
        let batch = RuntimeBatch {
            task: TaskBag {
                input_ids: vec![0, 1],
                tt_stage_node_ids: vec![100, 101],
                seed_node_ids: seed_ids,
                ..Default::default()
            },
            raw_features: [(
                "users".into(),
                RawFeatureBank {
                    schema: Some(feature_chunk.schema()),
                    batches: vec![feature_chunk.clone(), feature_chunk],
                },
            )]
            .into(),
            ..Default::default()
        };
        let batch = ArrowBatch::try_from(batch).unwrap();
        crate::validate_layout(&batch.layout, &batch.components).unwrap();
        let Some(crate::ArrowValue::Array(seeds)) =
            crate::resolve_reference(&batch.layout["task"]["seed_node_ids"], &batch.components)
                .unwrap()
        else {
            panic!("Expected seed array")
        };
        let seeds = seeds.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(seeds.values().as_ptr(), seed_buffer);
        assert!(batch.layout["task"].get("target").is_none());
        assert!(batch.layout["task"].get("weight").is_none());
        assert_eq!(batch.layout["raw_features"]["users"]["num_rows"], 4);
        let chunks = batch.layout["raw_features"]["users"]["record_batches"]
            .as_array()
            .unwrap();
        assert_eq!(chunks.len(), 2);
        for chunk in chunks {
            let Some(crate::ArrowValue::RecordBatch(chunk)) =
                crate::resolve_reference(chunk, &batch.components).unwrap()
            else {
                panic!("Expected feature record batch")
            };
            assert!(Arc::ptr_eq(chunk.column(0), &feature_buffer));
        }
    }
}
