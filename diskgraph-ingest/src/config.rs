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
    Context = 0,
    Encode = 1,
    PkMap = 2,
    EdgesMap = 3,
    Index = 4,
}

pub const ALL_PHASES: [IngestPhase; 5] = [
    IngestPhase::Context,
    IngestPhase::Encode,
    IngestPhase::PkMap,
    IngestPhase::EdgesMap,
    IngestPhase::Index,
];

impl IngestPhase {
    pub fn name(self) -> &'static str {
        match self {
            Self::Context => "context",
            Self::Encode => "encode",
            Self::PkMap => "pk_map",
            Self::EdgesMap => "edges_map",
            Self::Index => "index",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "context" => Some(Self::Context),
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
    /// Per-phase worker pool overrides. When `Some(n)`, that phase's `make_pool`
    /// uses `n`. `None` falls back to `default_worker_threads()`
    /// (≈ `available_parallelism × 1.05`).
    pub context_threads: Option<usize>,
    pub encode_threads: Option<usize>,
    pub pk_map_threads: Option<usize>,
    pub edges_map_threads: Option<usize>,
    pub index_threads: Option<usize>,
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
            context_threads: None,
            encode_threads: None,
            pk_map_threads: None,
            edges_map_threads: None,
            index_threads: None,
        }
    }
}
