// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encode native task batches without constructing Python objects.
use crate::protocol::MAX_BATCH_BYTES;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{Field, Schema};
use diskgraph_engine::output::{GraphChannel, RuntimeBatch};
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Default)]
pub(crate) struct BatchEncoder {
    pub(crate) parts: Vec<Vec<u8>>,
    bytes: usize,
}
impl BatchEncoder {
    fn record_batch(&mut self, batch: &RecordBatch, array: bool) -> anyhow::Result<Value> {
        let mut payload = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut payload, &batch.schema())?;
            writer.write(batch)?;
            writer.finish()?;
        }
        self.bytes += payload.len();
        anyhow::ensure!(
            self.bytes <= MAX_BATCH_BYTES,
            "Task batch exceeds 64 MiB; reduce batch size or fanout"
        );
        let slot = self.parts.len();
        self.parts.push(payload);
        Ok(json!({"__arrow__":slot,"array":array}))
    }
    fn array(&mut self, array: impl Into<ArrayRef>) -> anyhow::Result<Value> {
        let array = array.into();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            array.data_type().clone(),
            true,
        )]));
        self.record_batch(&RecordBatch::try_new(schema, vec![array])?, true)
    }
    pub(crate) fn batch(&mut self, batch: RuntimeBatch) -> anyhow::Result<Value> {
        use arrow_array::{BooleanArray, Float32Array, Int64Array, UInt32Array, UInt64Array};
        let task = batch.task;
        let mut t = serde_json::Map::new();
        t.insert(
            "input_ids".into(),
            self.array(Arc::new(UInt32Array::from(task.input_ids)) as ArrayRef)?,
        );
        t.insert(
            "tt_stage_node_ids".into(),
            self.array(Arc::new(UInt64Array::from(task.tt_stage_node_ids)) as ArrayRef)?,
        );
        t.insert(
            "seed_node_ids".into(),
            self.array(Arc::new(UInt64Array::from(task.seed_node_ids)) as ArrayRef)?,
        );
        t.insert(
            "neg_mask".into(),
            self.array(Arc::new(BooleanArray::from(task.neg_mask)) as ArrayRef)?,
        );
        if let Some(values) = task.seed_timestamps {
            t.insert(
                "seed_timestamps".into(),
                self.array(Arc::new(Int64Array::from(values)) as ArrayRef)?,
            );
        }
        if let Some(values) = task.weight {
            t.insert(
                "weight".into(),
                self.array(Arc::new(Float32Array::from(values)) as ArrayRef)?,
            );
        }
        if let Some(values) = task.lp_pos_weight {
            t.insert(
                "lp_pos_weight".into(),
                self.array(Arc::new(Float32Array::from(values)) as ArrayRef)?,
            );
        }
        if let Some(values) = task.lp_pos_local {
            t.insert(
                "lp_pos_local".into(),
                self.array(Arc::new(UInt32Array::from(values)) as ArrayRef)?,
            );
        }
        if let Some(target) = task.target {
            t.insert("target".into(), self.record_batch(&target, false)?);
        }
        t.insert("amount_m".into(), json!(task.amount_m));
        t.insert("num_positives".into(), json!(task.num_positives));
        t.insert("rhs_seed_type".into(), json!(task.rhs_seed_type));
        if !task.lp_pos_counts.is_empty() {
            t.insert(
                "lp_pos_counts".into(),
                self.array(Arc::new(UInt32Array::from(task.lp_pos_counts)) as ArrayRef)?,
            );
            t.insert(
                "lp_pos_rhs_global".into(),
                self.array(Arc::new(Int64Array::from(task.lp_pos_rhs_global)) as ArrayRef)?,
            );
            t.insert(
                "lp_pos_mask".into(),
                self.array(Arc::new(BooleanArray::from(task.lp_pos_mask)) as ArrayRef)?,
            );
        }
        if !task.lp_exclude_counts.is_empty() {
            t.insert(
                "lp_exclude_counts".into(),
                self.array(Arc::new(UInt32Array::from(task.lp_exclude_counts)) as ArrayRef)?,
            );
            t.insert(
                "lp_exclude_rhs_global".into(),
                self.array(Arc::new(Int64Array::from(task.lp_exclude_rhs_global)) as ArrayRef)?,
            );
        }
        let mut graphs = serde_json::Map::new();
        for (name, channel) in batch.graphs {
            graphs.insert(name.clone(), self.graph(channel)?);
        }
        let mut features = serde_json::Map::new();
        for (name, bank) in batch.raw_features {
            let records = bank
                .batches
                .iter()
                .map(|b| self.record_batch(b, false))
                .collect::<anyhow::Result<Vec<_>>>()?;
            features.insert(name.clone(), json!({"key":name,"num_rows":bank.batches.iter().map(RecordBatch::num_rows).sum::<usize>(),"record_batches":records}));
        }
        Ok(json!({"task":t,"graphs":graphs,"raw_features":features}))
    }
    fn graph(&mut self, channel: GraphChannel) -> anyhow::Result<Value> {
        use arrow_array::{UInt32Array, UInt64Array};
        let mut nodes = serde_json::Map::new();
        for (name, ids) in channel.node_ids {
            nodes.insert(
                name.clone(),
                self.array(Arc::new(UInt64Array::from(ids)) as ArrayRef)?,
            );
        }
        let mut batches = serde_json::Map::new();
        for (name, ids) in channel.batch_dict {
            batches.insert(
                name.clone(),
                self.array(Arc::new(UInt32Array::from(ids)) as ArrayRef)?,
            );
        }
        let mut counts = serde_json::Map::new();
        for (name, ids) in channel.num_sampled_nodes {
            counts.insert(
                name.clone(),
                self.array(Arc::new(UInt64Array::from(ids)) as ArrayRef)?,
            );
        }
        let mut edges = Vec::new();
        for (id, (src, dst)) in channel.edges.into_iter().enumerate() {
            if src.is_empty() {
                continue;
            }
            let (s, r, d) = channel
                .rel_id_to_names
                .get(&(id as u16))
                .cloned()
                .unwrap_or_else(|| ("?".into(), format!("rel_{id}"), "?".into()));
            edges.push(json!({"edge_type":{"src":s,"rel":r,"dst":d},
                "src":self.array(Arc::new(UInt32Array::from(src)) as ArrayRef)?,
                "dst":self.array(Arc::new(UInt32Array::from(dst)) as ArrayRef)?}));
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
    use crate::protocol::envelope;
    use crate::runtime::TaskOptions;
    use crate::storage::resolve_beneath;
    use arrow_array::{Array, Int64Array, ListArray};
    use arrow_ipc::reader::StreamReader;
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
        let mut encoder = BatchEncoder::default();
        let value = encoder.batch(batch).unwrap();
        assert_eq!(value["task"]["num_positives"], 1);
        assert!(value["task"]["lp_pos_counts"].is_object());
        assert!(value["graphs"]["main"]["node_ids"]["users"].is_object());
        assert_eq!(value["raw_features"]["users"]["num_rows"], 1);
        let slot = value["task"]["target"]["__arrow__"].as_u64().unwrap() as usize;
        let mut reader =
            StreamReader::try_new(std::io::Cursor::new(&encoder.parts[slot]), None).unwrap();
        assert_eq!(reader.next().unwrap().unwrap(), target);
        let wrapped = envelope(json!({"kind":"batch","value":value}), encoder.parts).unwrap();
        assert_eq!(wrapped.num_rows(), 1);
        let components = wrapped
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert!(!components.value(0).is_empty());
    }

    #[test]
    fn task_options_reject_unknown_fields() {
        let options = json!({"stage_table":"train_seeds","sampling_plan_json":"{}","num_workers":1,"worker_id":0,"shuffle":false,"epoch":0,"batch_size":2});
        let parsed: TaskOptions = serde_json::from_value(options.clone()).unwrap();
        assert!(parsed.disjoint);
        assert!(parsed.swap_edge_direction);
        assert!(parsed.use_seed_timestamps);
        assert!(!parsed.replace);
        let mut unsupported = options;
        unsupported["ignore_me"] = json!(true);
        assert!(serde_json::from_value::<TaskOptions>(unsupported).is_err());
    }

    #[test]
    fn paths_cannot_escape_the_served_graph() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("task")).unwrap();
        assert!(resolve_beneath(root.path(), "task").is_ok());
        assert!(resolve_beneath(root.path(), "../task").is_err());
        assert!(resolve_beneath(root.path(), "/task").is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/", root.path().join("outside")).unwrap();
            assert!(resolve_beneath(root.path(), "outside").is_err());
        }
    }
}
