use std::collections::HashSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct TrainTableStatsPayload {
    pub version: u32,
    pub num_total_rows: u64,
    pub num_train_rows: u64,
    pub num_entities: u64,
    pub num_targets: u64,
    pub num_links: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct TTExecutionSummary {
    pub rows_written: u64,
    pub stats: Option<TrainTableStatsPayload>,
}

#[derive(Default)]
pub(crate) struct ChunkStats {
    pub(crate) num_total_rows: u64,
    pub(crate) num_train_rows: u64,
    pub(crate) num_links: u64,
    pub(crate) entities: HashSet<u64>,
    pub(crate) targets: HashSet<u64>,
}

#[derive(Default)]
pub(crate) struct StatsMerger {
    num_train_rows: u64,
    num_links: u64,
    entities: HashSet<u64>,
    targets: HashSet<u64>,
}

impl StatsMerger {
    pub(crate) fn merge(&mut self, chunk: ChunkStats) {
        self.num_train_rows += chunk.num_train_rows;
        self.num_links += chunk.num_links;
        self.entities.extend(chunk.entities);
        self.targets.extend(chunk.targets);
    }

    pub(crate) fn finalize(
        self,
        num_total_rows: u64,
        is_link_prediction: bool,
    ) -> TrainTableStatsPayload {
        TrainTableStatsPayload {
            version: 1,
            num_total_rows,
            num_train_rows: self.num_train_rows,
            num_entities: self.entities.len() as u64,
            num_targets: self.targets.len() as u64,
            num_links: if is_link_prediction {
                Some(self.num_links)
            } else {
                None
            },
        }
    }
}
