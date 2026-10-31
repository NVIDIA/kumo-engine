// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::storage::resolve_beneath;
use diskgraph_runtime::{RuntimeBatchIterator, RuntimeError, RuntimeIndex};
use serde_json::{json, Value};
use std::{collections::HashMap, path::Path, sync::Arc};
use tonic::Status;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskRequest {
    pub kind: String,
    pub version: u32,
    pub extension_ref: String,
    pub engine_seed: Option<u64>,
    pub num_threads: Option<usize>,
    pub options: TaskOptions,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskOptions {
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
