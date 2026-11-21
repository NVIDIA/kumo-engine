// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Complete runtime output and optional temporal history execution.

use std::{collections::HashMap, sync::Arc};

use diskgraph_engine::{
    engine::DiskGraphEngine,
    output::{GraphChannel, RawFeatureBank, SamplingBatch, TaskBag},
};
use diskgraph_index::graph::GraphIndexReader;
use diskgraph_train_table::{
    batch_aggregate::{PreparedTemporalPlan, QueryResult},
    event_stream::{EventMerge, EventOrders},
};

use crate::{Result, RuntimeError};

/// Complete sampled batch, including configured history and event merges.
#[derive(Default)]
pub struct RuntimeBatch {
    /// Seed identities, supervision and link-prediction labels.
    pub task: TaskBag,
    /// Sampled graph channels, including configured neighborhoods.
    pub graphs: HashMap<String, GraphChannel>,
    /// Arrow feature chunks keyed by graph/task namespace.
    pub raw_features: HashMap<String, RawFeatureBank>,
    /// Temporal query output when history is configured.
    pub history: Option<QueryResult>,
    /// Named event streams represented by per-anchor offsets and event indices.
    pub history_merges: EventOrders,
}

impl From<SamplingBatch> for RuntimeBatch {
    fn from(batch: SamplingBatch) -> Self {
        Self {
            task: batch.task,
            graphs: batch.graphs,
            raw_features: batch.raw_features,
            history: None,
            history_merges: Vec::new(),
        }
    }
}

/// Prepared history context shared by batch delivery and projected length queries.
pub struct RuntimeHistory {
    root_type: String,
    reader: Arc<GraphIndexReader>,
    engine: Arc<DiskGraphEngine>,
    plan: PreparedTemporalPlan,
    merges: Vec<EventMerge>,
}

impl RuntimeHistory {
    pub(crate) fn prepare(
        context: (Arc<GraphIndexReader>, Arc<DiskGraphEngine>),
        root_type: &str,
        plan: &str,
        merges: &str,
    ) -> Result<Self> {
        let (reader, engine) = context;
        let plan = PreparedTemporalPlan::prepare_history(&reader, root_type, plan)
            .map_err(RuntimeError::config)?;
        let merges: Vec<EventMerge> = serde_json::from_str(merges)
            .map_err(|error| RuntimeError::config(format!("invalid history merges: {error}")))?;
        for merge in &merges {
            if merge.sequences.len() != merge.ascending.len() {
                return Err(RuntimeError::config("merge sequence/order counts differ"));
            }
            for sequence in &merge.sequences {
                plan.require_timed_sequence(sequence)
                    .map_err(RuntimeError::config)?;
            }
        }
        Ok(Self {
            root_type: root_type.to_owned(),
            reader,
            engine,
            plan,
            merges,
        })
    }

    /// Prepare a projected history query using the same graph and root type.
    /// Returns a configuration error if the query is invalid.
    pub fn prepare_query(&self, plan: &str) -> Result<PreparedTemporalPlan> {
        PreparedTemporalPlan::prepare_history(&self.reader, &self.root_type, plan)
            .map_err(RuntimeError::config)
    }

    /// Execute a prepared query for aligned seed IDs and index-clock cutoffs.
    /// Returns a configuration error if temporal execution fails.
    pub fn execute_query(
        &self,
        plan: &PreparedTemporalPlan,
        roots: &[u64],
        cutoffs: &[i64],
    ) -> Result<QueryResult> {
        plan.execute(&self.reader, roots, cutoffs, &self.engine.thread_pool)
            .map_err(RuntimeError::config)
    }

    /// Return event merge specifications in declaration order.
    pub fn merges(&self) -> &[EventMerge] {
        &self.merges
    }

    pub(crate) fn execute(&self, batch: &mut RuntimeBatch) -> Result<()> {
        let default_cutoffs;
        let cutoffs = match batch.task.seed_timestamps.as_deref() {
            Some(cutoffs) => cutoffs,
            None => {
                default_cutoffs = vec![0; batch.task.seed_node_ids.len()];
                &default_cutoffs
            }
        };
        let result = self.execute_query(&self.plan, &batch.task.seed_node_ids, cutoffs)?;
        batch.history_merges = result
            .merged_event_orders(&self.merges)
            .map_err(RuntimeError::config)?;
        batch.history = Some(result);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphChannel, RawFeatureBank, RuntimeBatch, SamplingBatch};

    #[test]
    fn sampling_output_moves_into_complete_batch_without_copying() {
        let mut sampled = SamplingBatch::default();
        sampled.task.seed_node_ids = vec![1, 3, 5];
        sampled
            .graphs
            .insert("main".into(), GraphChannel::default());
        sampled
            .raw_features
            .insert("users".into(), RawFeatureBank::default());
        let seed_buffer = sampled.task.seed_node_ids.as_ptr();

        let batch = RuntimeBatch::from(sampled);

        assert_eq!(batch.task.seed_node_ids.as_ptr(), seed_buffer);
        assert_eq!(batch.task.seed_node_ids, [1, 3, 5]);
        assert!(batch.graphs.contains_key("main"));
        assert!(batch.raw_features.contains_key("users"));
        assert!(batch.history.is_none());
        assert!(batch.history_merges.is_empty());
    }
}
