use crate::progress::ProgressCallback;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

pub const PEAK_MEM_STAGE_COUNT: usize = 6;

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
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            profile_dir: None,
            progress: None,
            parallel: true,
            index_mem_budget_bytes: None,
            peak_mem_stats: None,
        }
    }
}
