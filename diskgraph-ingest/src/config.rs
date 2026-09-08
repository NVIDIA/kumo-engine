use crate::progress::ProgressCallback;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

pub const PEAK_MEM_STAGE_COUNT: usize = 6;

/// Pipeline phases in execution order. Used by `IngestOptions::resume_from_phase`
/// to pick a starting point for resumed ingests; numeric ordering reflects
/// execution order so `<`/`>` comparisons answer "is this phase before/after".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestPhase {
    Encode = 0,
    PkMap = 1,
    EdgesMap = 2,
    Index = 3,
}

pub const ALL_PHASES: [IngestPhase; 4] = [
    IngestPhase::Encode,
    IngestPhase::PkMap,
    IngestPhase::EdgesMap,
    IngestPhase::Index,
];

impl IngestPhase {
    pub fn name(self) -> &'static str {
        match self {
            Self::Encode => "encode",
            Self::PkMap => "pk_map",
            Self::EdgesMap => "edges_map",
            Self::Index => "index",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "encode" => Some(Self::Encode),
            "pk_map" => Some(Self::PkMap),
            "edges_map" => Some(Self::EdgesMap),
            "index" => Some(Self::Index),
            _ => None,
        }
    }

    /// Phases that must already be complete before this one can run.
    pub fn predecessors(self) -> impl Iterator<Item = IngestPhase> {
        ALL_PHASES.iter().copied().take_while(move |p| *p < self)
    }
}

pub struct PeakMemStats {
    stage_index: AtomicUsize,
    overall_max: AtomicU64,
    stage_max: Vec<AtomicU64>,
}

impl Default for PeakMemStats {
    fn default() -> Self {
        Self::new()
    }
}

impl PeakMemStats {
    pub fn new() -> Self {
        Self {
            stage_index: AtomicUsize::new(usize::MAX),
            overall_max: AtomicU64::new(0),
            stage_max: (0..PEAK_MEM_STAGE_COUNT)
                .map(|_| AtomicU64::new(0))
                .collect(),
        }
    }

    pub fn set_stage(&self, idx: Option<usize>) {
        let value = idx.unwrap_or(usize::MAX);
        self.stage_index.store(value, Ordering::Relaxed);
    }

    pub fn stage_index(&self) -> usize {
        self.stage_index.load(Ordering::Relaxed)
    }

    pub fn update_overall(&self, bytes: u64) {
        update_max(&self.overall_max, bytes);
    }

    pub fn update_stage(&self, idx: usize, bytes: u64) {
        if let Some(slot) = self.stage_max.get(idx) {
            update_max(slot, bytes);
        }
    }

    pub fn overall_max(&self) -> u64 {
        self.overall_max.load(Ordering::Relaxed)
    }

    pub fn stage_max(&self, idx: usize) -> u64 {
        self.stage_max
            .get(idx)
            .map(|v| v.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
}

fn update_max(target: &AtomicU64, bytes: u64) {
    let mut current = target.load(Ordering::Relaxed);
    while bytes > current {
        match target.compare_exchange(current, bytes, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(prev) => current = prev,
        }
    }
}

/// Scalar vs list arity of a [`ResolveColumnSpec`] reference column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveKind {
    Scalar,
    List,
}

/// One FK-free LP resolution column: resolve `src_col`'s PK reference value(s)
/// against `dst_table`'s pkmap to graph node-id(s) and write them into a NEW
/// rowpack column `node_id_col`, WITHOUT emitting any FK edge for `src_col`.
///
/// The ingest pipeline appends `node_id_col` to the table's column list
/// (`UInt64` for [`ResolveKind::Scalar`], `Int64Array` for
/// [`ResolveKind::List`]); the value is filled at node-write time from the
/// parent pkmap, not read from the source parquet.
#[derive(Clone, Debug)]
pub struct ResolveColumnSpec {
    /// On-disk stage table this resolution applies to (e.g. `train_seeds`).
    pub table: String,
    /// Source column in the stage parquet holding the PK reference value(s).
    pub src_col: String,
    /// Parent node table whose pkmap resolves the reference values.
    pub dst_table: String,
    /// Scalar vs list arity of `src_col`.
    pub kind: ResolveKind,
    /// Name of the NEW rowpack column the resolved node-id(s) are written to.
    pub node_id_col: String,
}

#[derive(Clone)]
pub struct IngestOptions {
    pub profile_dir: Option<PathBuf>,
    pub progress: Option<ProgressCallback>,
    pub parallel: bool,
    pub index_mem_budget_bytes: Option<u64>,
    pub peak_mem_stats: Option<Arc<PeakMemStats>>,
    /// Collect per-column statistics during the encode stage.
    pub col_stats: bool,
    /// Category count collection strategy.
    pub cat_count_strategy: crate::col_stats::CategoryCountStrategy,
    /// Enable per-row 30% subsampling for category counts (scale ÷0.3).
    pub row_subsample: bool,
    /// If `Some(P)`, skip phases earlier than P (their completion markers must exist
    /// in `<output>/_tmp/_phase_complete_<name>`); start running at P. Phase P's
    /// own outputs are wiped on entry to clear partial state from a prior crash.
    /// `None` (default) preserves today's behaviour: run all phases unconditionally.
    pub resume_from_phase: Option<IngestPhase>,
    /// Maximum concurrent source loaders/downloads during Encode.
    pub source_loader_threads: Option<usize>,
    /// Maximum Encode workers shared by node and edge tables.
    pub source_encoder_threads: Option<usize>,
    pub pk_map_threads: Option<usize>,
    pub edges_map_threads: Option<usize>,
    pub index_threads: Option<usize>,
    /// Parent DG index whose pkmap shards resolve `resolve_columns` (below).
    /// `None` for an ordinary base-graph ingest.
    pub parent_index_path: Option<PathBuf>,
    /// Resolution columns. When non-empty, the listed columns are
    /// resolved to node-ids at node-write time (using `parent_index_path`'s
    /// pkmap) and written as rowpack columns with NO FK edge. Empty (default)
    /// preserves the existing edge-materializing behaviour.
    pub resolve_columns: Vec<ResolveColumnSpec>,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            profile_dir: None,
            progress: None,
            parallel: true,
            index_mem_budget_bytes: None,
            peak_mem_stats: None,
            col_stats: false,
            cat_count_strategy: crate::col_stats::CategoryCountStrategy::None,
            row_subsample: false,
            resume_from_phase: None,
            source_loader_threads: None,
            source_encoder_threads: None,
            pk_map_threads: None,
            edges_map_threads: None,
            index_threads: None,
            parent_index_path: None,
            resolve_columns: Vec::new(),
        }
    }
}
