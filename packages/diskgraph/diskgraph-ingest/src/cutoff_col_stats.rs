// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Column statistics of the rows created at or before each cutoff, under `col_stats/<dir>/`.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use diskgraph_core::time::TimeUnit;
use diskgraph_rowpack::{ColumnValueObserver, FloatListView, IntListView, StringListView};
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::col_stats::{
    write_col_stats, CategoryCountStrategy, ColStatsTable, FileColStats, StatSet,
    TableCategoryCounts,
};
use crate::graph_config::{ColumnSpec, SemanticType, TableSpec};
use crate::io::{Dest, OutputTarget};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CutoffColStatsRequest {
    /// Index-clock ticks; `None` counts every row.
    pub cutoffs: Vec<Option<i64>>,
    /// The `col_stats/<dir>/` of each cutoff.
    pub dirs: Vec<String>,
    /// Table, then column, to the semantic types whose statistics it collects.
    pub columns: BTreeMap<String, BTreeMap<String, Vec<SemanticType>>>,
    /// The t-digest quantiles every numerical column publishes.
    #[serde(default)]
    pub quantiles: Vec<f64>,
}

/// One table's statistics at every cutoff.
pub struct CutoffTable {
    name: String,
    columns: Vec<ColumnSpec>,
    stat_sets: Vec<StatSet>,
    temporal: bool,
    cutoffs: Vec<Option<i64>>,
    categories: Vec<Arc<TableCategoryCounts>>,
    stats: Option<CutoffSinks>,
}

/// A worker's statistics at every cutoff; a row reaches the cutoffs it counts at.
pub struct CutoffSinks {
    cutoffs: Vec<Option<i64>>,
    temporal: bool,
    pub stats: Vec<FileColStats>,
    active: Vec<bool>,
    row: Row,
}

impl CutoffColStatsRequest {
    pub fn validate(&self, tables: &[TableSpec]) -> Result<()> {
        ensure!(
            self.cutoffs.len() == self.dirs.len(),
            "one directory per cutoff"
        );
        ensure!(
            self.dirs.iter().collect::<HashSet<_>>().len() == self.dirs.len(),
            "each cutoff needs its own directory"
        );
        for (name, columns) in &self.columns {
            let table = tables
                .iter()
                .find(|table| &table.name == name && table.edge_table.is_none())
                .ok_or_else(|| anyhow!("column statistics need a node table, got {name}"))?;
            for column in columns.keys() {
                ensure!(
                    table.columns.iter().any(|c| &c.name == column),
                    "table {name} has no column {column}"
                );
            }
        }
        Ok(())
    }

    /// The requested statistics of each column; the others collect nothing.
    pub fn table(&self, table: &TableSpec) -> Option<CutoffTable> {
        let requested = self.columns.get(&table.name)?;
        let stat_sets: Vec<StatSet> = table
            .columns
            .iter()
            .map(|column| {
                requested
                    .get(&column.name)
                    .into_iter()
                    .flatten()
                    .fold(StatSet::default(), |set, &stype| set | Some(stype).into())
            })
            .collect();
        let categories = self
            .cutoffs
            .iter()
            .map(|_| {
                Arc::new(TableCategoryCounts::new(
                    CategoryCountStrategy::Hybrid,
                    &stat_sets,
                    false,
                ))
            })
            .collect();
        Some(CutoffTable {
            name: table.name.clone(),
            columns: table.columns.clone(),
            stat_sets,
            temporal: table.create_time.is_some(),
            cutoffs: self.cutoffs.clone(),
            categories,
            stats: None,
        })
    }

    /// Adds each table's statistics to its `col_stats/<dir>/<table>.col_stats.json`.
    pub fn publish(
        &self,
        index_dir: &Path,
        clock: TimeUnit,
        tables: Vec<CutoffTable>,
    ) -> Result<()> {
        if tables.is_empty() {
            return Ok(());
        }
        let mut per_cutoff: Vec<Vec<ColStatsTable<'_>>> =
            self.dirs.iter().map(|_| Vec::new()).collect();
        for mut table in tables {
            let stats = match table.stats.take() {
                Some(stats) => stats,
                None => table.sinks()?,
            };
            for ((out, stats), categories) in
                per_cutoff.iter_mut().zip(stats.stats).zip(table.categories)
            {
                out.push(ColStatsTable {
                    name: table.name.clone(),
                    stats: stats
                        .with_clock(clock)
                        .with_quantiles(self.quantiles.clone()),
                    column_names: table.columns.iter().map(|c| c.name.clone()).collect(),
                    ingested_types: table
                        .columns
                        .iter()
                        .map(|c| c.ingested_dtype.clone())
                        .collect(),
                    table_cat: Some(categories),
                    text_sample_stats: None,
                    data_version: None,
                });
            }
        }
        for (dir, tables) in self.dirs.iter().zip(per_cutoff) {
            let dir = index_dir.join("col_stats").join(dir);
            std::fs::create_dir_all(&dir)?;
            for table in tables {
                add_to_file(&dir, table)?;
            }
        }
        Ok(())
    }
}

/// Table, column and statistic, each value kept as written.
type StatsFile = BTreeMap<String, BTreeMap<String, BTreeMap<String, Box<RawValue>>>>;

/// Adds `table` to `<dir>/<table>.col_stats.json`; a column already there keeps its keys and
/// gains the new ones.
fn add_to_file(dir: &Path, table: ColStatsTable<'_>) -> Result<()> {
    let path = dir.join(format!("{}.col_stats.json", file_name_part(&table.name)));
    let tmp = tempfile::Builder::new().prefix(".").tempdir_in(dir)?;
    write_col_stats(
        &OutputTarget::parse(&tmp.path().to_string_lossy(), Dest::Local)?,
        vec![table],
    )?;
    let written = tmp.path().join("col_stats.json");
    // Writers take turns, so none drops another's columns.
    let lock = File::options()
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join(".lock"))?;
    lock.lock()?;
    if path.is_file() {
        let mut stats: StatsFile = serde_json::from_slice(&std::fs::read(&path)?)?;
        let added: StatsFile = serde_json::from_slice(&std::fs::read(&written)?)?;
        for (name, columns) in added {
            let table = stats.entry(name).or_default();
            for (column, values) in columns {
                let known = table.entry(column).or_default();
                for (key, value) in values {
                    known.entry(key).or_insert(value);
                }
            }
        }
        std::fs::write(&written, serde_json::to_vec(&stats)?)?;
    }
    File::open(&written)?.sync_all()?;
    std::fs::rename(&written, &path)?;
    Ok(())
}

/// `name` with every byte but ASCII letters, digits, `-` and `_` percent-encoded.
fn file_name_part(name: &str) -> String {
    name.bytes()
        .map(|byte| match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' => char::from(byte).to_string(),
            _ => format!("%{byte:02X}"),
        })
        .collect()
}

impl CutoffTable {
    pub fn sinks(&self) -> Result<CutoffSinks> {
        Ok(CutoffSinks {
            cutoffs: self.cutoffs.clone(),
            temporal: self.temporal,
            stats: self
                .categories
                .iter()
                .map(|categories| {
                    FileColStats::new_with_stat_sets(
                        &self.columns,
                        &self.stat_sets,
                        Arc::clone(categories),
                    )
                })
                .collect::<Result<_>>()?,
            active: vec![true; self.cutoffs.len()],
            row: Row::default(),
        })
    }

    pub fn add(&mut self, mut sinks: CutoffSinks) {
        for stats in &mut sinks.stats {
            stats.flush_categories();
        }
        match &mut self.stats {
            Some(total) => {
                for (total, stats) in total.stats.iter_mut().zip(&mut sinks.stats) {
                    total.merge(stats);
                }
            }
            None => self.stats = Some(sinks),
        }
    }
}

/// One observed value; strings and bytes live in the row's buffers.
enum Seen {
    I64(i64),
    F64(f64),
    Str(Range<usize>),
    Bytes(Range<usize>),
    Decimal(f64, Range<usize>),
    Timestamp(i64),
    FloatList(FloatListView),
    IntList(IntListView),
    StringList(StringListView),
    Null,
    ParseFailure,
    BeyondClock,
}

/// A row's observations, held until the row is known to be stored.
#[derive(Default)]
struct Row {
    seen: Vec<(usize, Seen)>,
    text: String,
    bytes: Vec<u8>,
}

impl Row {
    fn text(&mut self, value: &str) -> Range<usize> {
        let start = self.text.len();
        self.text.push_str(value);
        start..self.text.len()
    }

    fn bytes(&mut self, value: &[u8]) -> Range<usize> {
        let start = self.bytes.len();
        self.bytes.extend_from_slice(value);
        start..self.bytes.len()
    }

    fn replay(&self, stats: &mut FileColStats) {
        for (col, seen) in &self.seen {
            let col = *col;
            match seen {
                Seen::I64(value) => stats.observe_i64(col, *value),
                Seen::F64(value) => stats.observe_f64(col, *value),
                Seen::Str(range) => stats.observe_string(col, &self.text[range.clone()]),
                Seen::Bytes(range) => stats.observe_bytes(col, &self.bytes[range.clone()]),
                Seen::Decimal(value, range) => {
                    stats.observe_decimal(col, *value, &|| self.text[range.clone()].to_owned())
                }
                Seen::Timestamp(ticks) => stats.observe_timestamp(col, *ticks),
                Seen::FloatList(view) => stats.observe_float_list(col, view),
                Seen::IntList(view) => stats.observe_int64_list(col, view),
                Seen::StringList(view) => stats.observe_string_list(col, view),
                Seen::Null => stats.observe_null(col),
                Seen::ParseFailure => stats.observe_parse_failure(col),
                Seen::BeyondClock => stats.observe_beyond_clock(col),
            }
        }
    }
}

impl CutoffSinks {
    fn push(&mut self, col_idx: usize, seen: impl FnOnce(&mut Row) -> Seen) {
        if self.active.contains(&true) && self.stats[0].collects(col_idx) {
            let seen = seen(&mut self.row);
            self.row.seen.push((col_idx, seen));
        }
    }
}

impl ColumnValueObserver for CutoffSinks {
    fn observe_i64(&mut self, col_idx: usize, value: i64) {
        self.push(col_idx, |_| Seen::I64(value));
    }
    fn observe_f64(&mut self, col_idx: usize, value: f64) {
        self.push(col_idx, |_| Seen::F64(value));
    }
    fn observe_string(&mut self, col_idx: usize, value: &str) {
        self.push(col_idx, |row| Seen::Str(row.text(value)));
    }
    fn observe_bytes(&mut self, col_idx: usize, value: &[u8]) {
        self.push(col_idx, |row| Seen::Bytes(row.bytes(value)));
    }
    fn observe_decimal(&mut self, col_idx: usize, value: f64, text: &dyn Fn() -> String) {
        let keyed = self.stats.iter().any(|s| s.counts_categories(col_idx));
        self.push(col_idx, |row| match keyed {
            true => Seen::Decimal(value, row.text(&text())),
            false => Seen::F64(value),
        });
    }
    fn observe_timestamp(&mut self, col_idx: usize, ticks: i64) {
        self.push(col_idx, |_| Seen::Timestamp(ticks));
    }
    fn observe_float_list(&mut self, col_idx: usize, view: &FloatListView) {
        self.push(col_idx, |_| Seen::FloatList(view.clone()));
    }
    fn observe_int64_list(&mut self, col_idx: usize, view: &IntListView) {
        self.push(col_idx, |_| Seen::IntList(view.clone()));
    }
    fn observe_string_list(&mut self, col_idx: usize, view: &StringListView) {
        self.push(col_idx, |_| Seen::StringList(view.clone()));
    }
    fn observe_null(&mut self, col_idx: usize) {
        self.push(col_idx, |_| Seen::Null);
    }
    fn observe_parse_failure(&mut self, col_idx: usize) {
        self.push(col_idx, |_| Seen::ParseFailure);
    }
    fn observe_beyond_clock(&mut self, col_idx: usize) {
        self.push(col_idx, |_| Seen::BeyondClock);
    }

    fn wants_create_time(&self) -> bool {
        true
    }

    fn start_row(&mut self, create_time: Option<i64>) {
        for (active, cutoff) in self.active.iter_mut().zip(&self.cutoffs) {
            *active = match (cutoff, create_time) {
                (Some(cutoff), Some(time)) => time <= *cutoff,
                (Some(_), None) => !self.temporal,
                (None, _) => true,
            };
        }
    }

    fn finish_row(&mut self, stored: bool) {
        if stored {
            for (stats, active) in self.stats.iter_mut().zip(&self.active) {
                if *active {
                    self.row.replay(stats);
                }
            }
        }
        self.row.seen.clear();
        self.row.text.clear();
        self.row.bytes.clear();
    }
}
