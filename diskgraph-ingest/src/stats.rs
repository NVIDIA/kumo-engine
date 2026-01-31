use serde::Serialize;
use std::collections::HashMap;
use crate::graph_config::GraphConfig;

#[derive(Clone, Debug, Serialize)]
pub struct IngestReport {
    pub stats: IngestStats,
    pub source: GraphConfig,
}

#[derive(Clone, Debug, Serialize)]
pub struct IngestStats {
    pub output: String,
    pub node_tables: usize,
    pub edge_relations: usize,
    pub total_nodes: u64,
    pub total_edges: u64,
    pub node_counts: HashMap<String, u64>,
    pub edge_counts: HashMap<String, u64>,
    pub key_parse_failures: u64,
    pub key_lookup_failures: u64,
    pub key_failure_samples: Vec<String>,
    pub key_failures_by_context: HashMap<String, KeyFailureCounts>,
    pub pk_duplicate_counts: HashMap<String, u64>,
    pub pk_duplicate_drops: HashMap<String, u64>,
    pub pk_duplicate_total: u64,
    pub pk_duplicate_samples: HashMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trimmed_strings: Option<u64>,
    pub perf: PerfStats,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_shards: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct KeyFailureCounts {
    pub parse_failures: u64,
    pub lookup_failures: u64,
}

#[derive(Default)]
pub struct KeyFailureState {
    pub parse_failures: u64,
    pub lookup_failures: u64,
    pub samples: Vec<String>,
    pub by_context: HashMap<String, KeyFailureCounts>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PerfStats {
    pub wall_time_ms: u64,
    pub context_wall_ms: u64,
    pub encode_wall_ms: u64,
    pub pk_map_wall_ms: u64,
    pub edges_map_wall_ms: u64,
    pub index_wall_ms: u64,
    pub finalize_wall_ms: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub temp_edges_src_bytes: u64,
    pub temp_fkeys_bytes: u64,
    pub temp_edges_map_bytes: u64,
    pub temp_node_data_bytes: u64,
    pub temp_bytes_total: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_context_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_encode_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_pk_map_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_edges_map_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_index_rss_anon_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_finalize_rss_anon_bytes: Option<u64>,
}
