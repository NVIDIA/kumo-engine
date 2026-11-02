// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::paths::resolve_beneath;
use diskgraph_runtime::{RuntimeBatchIterator, RuntimeError, RuntimeIndex};
use serde_json::{json, Value};
use std::{collections::HashMap, path::Path, sync::Arc};
use tonic::Status;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRequest {
    pub kind: String,
    pub version: u32,
    pub extension_ref: String,
    pub engine_seed: Option<u64>,
    pub num_threads: Option<usize>,
    pub options: TaskOptions,
}
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskOptions {
    pub stage_table: String,
    pub sampling_plan_json: String,
    pub num_workers: u64,
    pub worker_id: u64,
    pub shuffle: bool,
    #[serde(default)]
    pub rnd_seed: Option<u64>,
    pub epoch: u64,
    pub batch_size: usize,
    #[serde(default)]
    pub drop_last: bool,
    #[serde(default)]
    pub fused: Option<bool>,
    #[serde(default = "default_true")]
    pub swap_edge_direction: bool,
    #[serde(default)]
    pub neg_rhs_node_type: Option<String>,
    #[serde(default)]
    pub neg_amount_m: Option<usize>,
    #[serde(default)]
    pub neg_rnd_seed: Option<u64>,
    #[serde(default)]
    pub replace: bool,
    #[serde(default = "default_true")]
    pub disjoint: bool,
    #[serde(default)]
    pub adaptive: bool,
    #[serde(default)]
    pub entity_node_type: Option<String>,
    #[serde(default)]
    pub target_node_type: Option<String>,
    #[serde(default)]
    pub rhs_seed_node_type: Option<String>,
    #[serde(default)]
    pub rhs_seed_time: Option<i64>,
    #[serde(default)]
    pub group_by_timestamp: bool,
    #[serde(default)]
    pub weighted_seed_sampling_json: Option<String>,
    #[serde(default)]
    pub selected_stage_node_ids: Option<Vec<u64>>,
    #[serde(default = "default_true")]
    pub use_seed_timestamps: bool,
    #[serde(default)]
    pub pseudo_seed_time: Option<i64>,
    #[serde(default)]
    pub disable_pseudo_time: bool,
    #[serde(default)]
    pub projections: Option<HashMap<String, Vec<String>>>,
}
fn default_true() -> bool {
    true
}

pub(crate) fn runtime_error(error: RuntimeError) -> Status {
    match error {
        RuntimeError::Config(message) => Status::invalid_argument(message),
        RuntimeError::Data(message) => Status::failed_precondition(message),
        RuntimeError::TargetCategoryLimit(message) => Status::out_of_range(message),
        RuntimeError::System(message) => Status::internal(message),
    }
}

impl TaskRequest {
    /// Build an epoch request with the native wire version and validate its shard.
    pub fn new(
        extension_ref: String,
        options: TaskOptions,
        engine_seed: Option<u64>,
        num_threads: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(num_threads > 0, "Sampling thread count must be positive");
        anyhow::ensure!(
            options.batch_size > 0,
            "Sampling batch size must be positive"
        );
        anyhow::ensure!(
            options.num_workers > 0 && options.worker_id < options.num_workers,
            "Sampling worker_id must be less than positive num_workers"
        );
        Ok(Self {
            kind: "epoch".into(),
            version: crate::protocol::VERSION,
            extension_ref,
            engine_seed,
            num_threads: Some(num_threads),
            options,
        })
    }
}

impl TaskOptions {
    pub(crate) fn open(
        mut self,
        index: Arc<RuntimeIndex>,
        extension: &Path,
    ) -> Result<RuntimeBatchIterator, Status> {
        if let Some(weighted) = self.weighted_seed_sampling_json.as_mut() {
            let mut value: Value = serde_json::from_str(weighted)
                .map_err(|e| Status::invalid_argument(e.to_string()))?;
            let buckets = value["buckets"]
                .as_array_mut()
                .ok_or_else(|| Status::invalid_argument("Weighted sampling requires buckets"))?;
            for bucket in buckets {
                for key in ["node_ids_path", "cdf_path"] {
                    if let Some(path) = bucket.get(key).and_then(Value::as_str) {
                        bucket[key] = json!(resolve_beneath(extension, path)?);
                    }
                }
            }
            *weighted = value.to_string();
        }
        index
            .iter(
                self.stage_table,
                self.sampling_plan_json,
                self.num_workers,
                self.worker_id,
                self.shuffle,
                self.rnd_seed,
                self.epoch,
                self.batch_size,
                self.drop_last,
                self.fused,
                self.swap_edge_direction,
                self.neg_rhs_node_type,
                self.neg_amount_m,
                self.neg_rnd_seed,
                self.replace,
                self.disjoint,
                self.adaptive,
                self.entity_node_type,
                self.target_node_type,
                self.rhs_seed_node_type,
                self.rhs_seed_time,
                self.group_by_timestamp,
                self.weighted_seed_sampling_json,
                self.selected_stage_node_ids,
                self.use_seed_timestamps,
                self.pseudo_seed_time,
                self.disable_pseudo_time,
                self.projections,
            )
            .map_err(runtime_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options() -> TaskOptions {
        serde_json::from_value(json!({
            "stage_table":"train_seeds", "sampling_plan_json":"{\"layers\":[]}",
            "num_workers":2, "worker_id":1, "shuffle":true, "epoch":7, "batch_size":4,
            "selected_stage_node_ids":[u64::MAX,0], "projections":{"users":["age"]},
        }))
        .unwrap()
    }
    #[test]
    fn native_request_serialization_preserves_options_and_sets_wire_identity() {
        let request =
            TaskRequest::new("train_tables/task".into(), options(), Some(u64::MAX), 2).unwrap();
        let ticket = serde_json::to_vec(&request).unwrap();
        let decoded: TaskRequest = serde_json::from_slice(&ticket).unwrap();
        assert_eq!(decoded.kind, "epoch");
        assert_eq!(decoded.version, crate::protocol::VERSION);
        assert_eq!(decoded.engine_seed, Some(u64::MAX));
        assert_eq!(decoded.num_threads, Some(2));
        assert_eq!(
            decoded.options.selected_stage_node_ids,
            Some(vec![u64::MAX, 0])
        );
        assert_eq!(decoded.options.projections.unwrap()["users"], vec!["age"]);
        assert_eq!(
            (
                decoded.options.num_workers,
                decoded.options.worker_id,
                decoded.options.epoch
            ),
            (2, 1, 7)
        );
        assert!(decoded.options.disjoint);
        assert!(decoded.options.swap_edge_direction);
        assert!(decoded.options.use_seed_timestamps);
        assert_eq!(decoded.options.sampling_plan_json, "{\"layers\":[]}");
    }
    #[test]
    fn invalid_native_requests_are_rejected_before_serialization() {
        for (workers, worker_id, batch_size, threads) in
            [(0, 0, 1, 1), (1, 1, 1, 1), (1, 0, 0, 1), (1, 0, 1, 0)]
        {
            let mut options = options();
            options.num_workers = workers;
            options.worker_id = worker_id;
            options.batch_size = batch_size;
            assert!(TaskRequest::new("train_tables/task".into(), options, None, threads).is_err());
        }
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
}
