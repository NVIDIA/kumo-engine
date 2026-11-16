// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use diskgraph_index::graph::{GraphBuildProgress, GraphBuildStage, GraphProgressSink};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::io::IsTerminal;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProgressPhase {
    Encode,
    PkMap,
    EdgesPassA,
    EdgesPassB,
    EdgesPassC,
    Index,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressKind {
    Start,
    Advance,
    Finish,
}

#[derive(Clone, Debug)]
pub struct ProgressEvent {
    pub phase: ProgressPhase,
    pub table: String,
    pub shard: Option<usize>,
    pub kind: ProgressKind,
    pub delta: u64,
    pub total: Option<u64>,
    pub status: Option<String>,
}

pub type ProgressCallback = std::sync::Arc<dyn Fn(ProgressEvent) + Send + Sync>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProgressKey {
    phase: ProgressPhase,
    table: String,
    shard: Option<usize>,
}

struct ProgressState {
    mp: MultiProgress,
    bars: HashMap<ProgressKey, ProgressBar>,
}

pub struct IndexProgressAdapter {
    cb: ProgressCallback,
    state: Mutex<IndexProgressTracker>,
}

struct IndexProgressTracker {
    shard_states: HashMap<(String, Option<usize>), IndexProgressState>,
    table_states: HashMap<String, TableProgressState>,
}

#[derive(Clone, Copy)]
struct IndexProgressState {
    current: u64,
}

#[derive(Clone, Copy)]
struct TableProgressState {
    current: u64,
    total: Option<u64>,
    remaining_shards: Option<usize>,
    started: bool,
    finished: bool,
}

#[derive(Clone, Copy)]
pub struct TableIndexTotals {
    pub total_nodes: u64,
    pub total_shards: usize,
}

impl IndexProgressAdapter {
    pub fn new(cb: ProgressCallback, table_totals: HashMap<String, TableIndexTotals>) -> Self {
        let mut table_states = HashMap::new();
        for (table, totals) in table_totals {
            table_states.insert(
                table,
                TableProgressState {
                    current: 0,
                    total: Some(totals.total_nodes),
                    remaining_shards: Some(totals.total_shards),
                    started: false,
                    finished: false,
                },
            );
        }
        Self {
            cb,
            state: Mutex::new(IndexProgressTracker {
                shard_states: HashMap::new(),
                table_states,
            }),
        }
    }

    fn shard_key(shard: &str) -> (String, Option<usize>) {
        if let Some((table, rest)) = shard.split_once("/shard-") {
            if let Ok(idx) = rest.parse::<usize>() {
                return (table.to_string(), Some(idx));
            }
        }
        (shard.to_string(), None)
    }
}

impl GraphProgressSink for IndexProgressAdapter {
    fn report(&self, update: GraphBuildProgress) {
        let (table, shard) = match &update.stage {
            GraphBuildStage::BuildShard { shard } | GraphBuildStage::Complete { shard } => {
                Self::shard_key(shard.as_ref())
            }
        };
        let key = (table.clone(), shard);
        let mut state = self.state.lock().unwrap();
        match update.stage {
            GraphBuildStage::BuildShard { .. } => {
                let delta = match state.shard_states.entry(key) {
                    Entry::Vacant(v) => {
                        v.insert(IndexProgressState {
                            current: update.current,
                        });
                        update.current
                    }
                    Entry::Occupied(mut o) => {
                        let prev = o.get().current;
                        if update.current > prev {
                            let delta = update.current.saturating_sub(prev);
                            o.insert(IndexProgressState {
                                current: update.current,
                            });
                            delta
                        } else {
                            0
                        }
                    }
                };
                let table_state =
                    state
                        .table_states
                        .entry(table.clone())
                        .or_insert(TableProgressState {
                            current: 0,
                            total: None,
                            remaining_shards: None,
                            started: false,
                            finished: false,
                        });
                if !table_state.started && !table_state.finished {
                    table_state.started = true;
                    (self.cb)(ProgressEvent {
                        phase: ProgressPhase::Index,
                        table: table.clone(),
                        shard: None,
                        kind: ProgressKind::Start,
                        delta: 0,
                        total: table_state.total,
                        status: None,
                    });
                }
                if delta > 0 {
                    table_state.current = table_state.current.saturating_add(delta);
                    (self.cb)(ProgressEvent {
                        phase: ProgressPhase::Index,
                        table,
                        shard: None,
                        kind: ProgressKind::Advance,
                        delta,
                        total: table_state.total,
                        status: None,
                    });
                }
            }
            GraphBuildStage::Complete { .. } => {
                state.shard_states.remove(&key);
                let table_state =
                    state
                        .table_states
                        .entry(table.clone())
                        .or_insert(TableProgressState {
                            current: 0,
                            total: None,
                            remaining_shards: None,
                            started: false,
                            finished: false,
                        });
                if let Some(remaining) = table_state.remaining_shards.as_mut() {
                    if *remaining > 0 {
                        *remaining = remaining.saturating_sub(1);
                    }
                }
                let should_finish = match table_state.remaining_shards {
                    Some(remaining) => remaining == 0,
                    None => true,
                };
                if should_finish && !table_state.finished {
                    table_state.finished = true;
                    (self.cb)(ProgressEvent {
                        phase: ProgressPhase::Index,
                        table,
                        shard: None,
                        kind: ProgressKind::Finish,
                        delta: 0,
                        total: table_state.total.or(Some(table_state.current)),
                        status: None,
                    });
                }
            }
        }
    }
}

pub fn build_progress() -> Option<ProgressCallback> {
    if !std::io::stderr().is_terminal() {
        return Some(build_log_progress());
    }
    let state = Arc::new(Mutex::new(ProgressState {
        mp: MultiProgress::new(),
        bars: HashMap::new(),
    }));
    let cb = Arc::new(move |event: ProgressEvent| {
        let mut guard = state.lock().unwrap();
        let label = progress_label(&event);
        let key = ProgressKey {
            phase: event.phase,
            table: event.table.clone(),
            shard: event.shard,
        };
        match event.kind {
            ProgressKind::Start => {
                if guard.bars.contains_key(&key) {
                    return;
                }
                let bar = match event.total {
                    Some(total) => {
                        let bar = guard.mp.add(ProgressBar::new(total));
                        bar.set_style(bar_style());
                        bar
                    }
                    None => {
                        let bar = guard.mp.add(ProgressBar::new_spinner());
                        bar.set_style(spinner_style());
                        bar
                    }
                };
                bar.set_message(label);
                if let Some(total) = event.total {
                    bar.set_prefix(format!("({}/{})", format_count(0), format_count(total)));
                }
                guard.bars.insert(key, bar);
            }
            ProgressKind::Advance => {
                if let Some(bar) = guard.bars.get(&key) {
                    bar.inc(event.delta);
                    if let Some(total) = bar.length() {
                        bar.set_prefix(format!(
                            "({}/{})",
                            format_count(bar.position()),
                            format_count(total)
                        ));
                    } else if event.status.is_some() {
                        bar.set_message(label);
                    } else {
                        let count = format_count(bar.position());
                        bar.set_message(format!("{} {}", label, count));
                    }
                }
            }
            ProgressKind::Finish => {
                if let Some(bar) = guard.bars.remove(&key) {
                    if let Some(total) = event.total.or_else(|| bar.length()) {
                        bar.set_length(total);
                        if bar.position() < total {
                            bar.set_position(total);
                        }
                        bar.set_prefix(format!(
                            "({}/{})",
                            format_count(bar.position()),
                            format_count(total)
                        ));
                    }
                    bar.finish_with_message(label);
                }
            }
        }
    });
    Some(cb)
}

struct LogProgressState {
    last_logged_pct: i32,
    current: u64,
}

fn build_log_progress() -> ProgressCallback {
    let states = Arc::new(Mutex::new(HashMap::<ProgressKey, LogProgressState>::new()));
    Arc::new(move |event: ProgressEvent| {
        let mut guard = states.lock().unwrap();
        let key = ProgressKey {
            phase: event.phase,
            table: event.table.clone(),
            shard: event.shard,
        };

        let state = guard.entry(key).or_insert(LogProgressState {
            last_logged_pct: -1,
            current: 0,
        });

        match event.kind {
            ProgressKind::Start => {
                tracing::debug!(
                    phase = ?event.phase,
                    table = %event.table,
                    shard = ?event.shard,
                    total = ?event.total,
                    "Ingestion stage started"
                );
            }
            ProgressKind::Advance => {
                state.current += event.delta;
                if let Some(total) = event.total {
                    if total > 0 {
                        let pct = (state.current as f64 / total as f64 * 100.0) as i32;
                        // Log every 25% instead of 10% to reduce verbosity
                        if pct >= state.last_logged_pct + 25 || (pct >= 100 && state.last_logged_pct < 100) {
                            tracing::info!(
                                phase = ?event.phase,
                                table = %event.table,
                                shard = ?event.shard,
                                progress_pct = pct,
                                current = state.current,
                                total = total,
                                "Ingestion stage progress"
                            );
                            state.last_logged_pct = pct;
                        }
                    }
                }
            }
            ProgressKind::Finish => {
                tracing::debug!(
                    phase = ?event.phase,
                    table = %event.table,
                    shard = ?event.shard,
                    total = ?event.total.unwrap_or(state.current),
                    "Ingestion stage finished"
                );
            }
        }
    })
}

fn progress_label(event: &ProgressEvent) -> String {
    let phase = match event.phase {
        ProgressPhase::Encode => "encode",
        ProgressPhase::PkMap => "pk_map",
        ProgressPhase::EdgesPassA => "edges_map",
        ProgressPhase::EdgesPassB => "edges_b",
        ProgressPhase::EdgesPassC => "edges_c",
        ProgressPhase::Index => "index",
    };
    if matches!(event.phase, ProgressPhase::Index) {
        let label = format!("{} {}", phase, event.table);
        return match &event.status {
            Some(status) => format!("{} {}", label, status),
            None => label,
        };
    }
    match event.shard {
        Some(shard) => {
            let label = format!("{} {} shard-{:05}", phase, event.table, shard);
            match &event.status {
                Some(status) => format!("{} {}", label, status),
                None => label,
            }
        }
        None => {
            let label = format!("{} {}", phase, event.table);
            match &event.status {
                Some(status) => format!("{} {}", label, status),
                None => label,
            }
        }
    }
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {percent}% {prefix}")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
}

pub fn format_bytes(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut value = bytes as f64;
    let mut unit = units[0];
    for next in &units[1..] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = next;
    }
    if unit == "B" {
        format!("{} {}", bytes, unit)
    } else {
        format!("{:.1} {}", value, unit)
    }
}

pub fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        return format!("{}ms", ms);
    }
    let secs = ms / 1000;
    let rem_ms = ms % 1000;
    if secs < 60 {
        return format!("{}.{}s", secs, rem_ms / 100);
    }
    let mins = secs / 60;
    let rem_secs = secs % 60;
    format!("{}m {}s", mins, rem_secs)
}

pub fn format_count(value: u64) -> String {
    if value >= 1_000_000_000 {
        format!("{:.1}B", value as f64 / 1_000_000_000.0)
    } else if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}
