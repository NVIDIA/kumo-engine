// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Column statistics of the rows created at or before each cutoff, under `col_stats/<dir>/`.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use diskgraph_core::time::TimeUnit;
use diskgraph_rowpack::{ColumnValueObserver, FloatListView, IntListView, StringListView};
use serde::Deserialize;

use crate::col_stats::{
    write_col_stats, CategoryCountStrategy, ColStatsTable, FileColStats, TableCategoryCounts,
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
    /// Table, then column, to the semantic type it is collected as.
    pub columns: BTreeMap<String, BTreeMap<String, SemanticType>>,
    /// The t-digest quantiles every numerical column publishes.
    #[serde(default)]
    pub quantiles: Vec<f64>,
}

/// One table's statistics at every cutoff.
pub struct CutoffTable {
    name: String,
    pub columns: Vec<ColumnSpec>,
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

    /// The requested columns as their requested types; the others collect nothing.
    pub fn table(&self, table: &TableSpec) -> Option<CutoffTable> {
        let requested = self.columns.get(&table.name)?;
        let columns: Vec<ColumnSpec> = table
            .columns
            .iter()
            .map(|column| ColumnSpec {
                semantic_type: Some(
                    requested
                        .get(&column.name)
                        .copied()
                        .unwrap_or(SemanticType::Unsupported),
                ),
                ..column.clone()
            })
            .collect();
        let stypes: Vec<_> = columns.iter().map(|c| c.semantic_type).collect();
        let categories = self
            .cutoffs
            .iter()
            .map(|_| {
                Arc::new(TableCategoryCounts::new(
                    CategoryCountStrategy::Hybrid,
                    &stypes,
                    false,
                ))
            })
            .collect();
        Some(CutoffTable {
            name: table.name.clone(),
            columns,
            temporal: table.create_time.is_some(),
            cutoffs: self.cutoffs.clone(),
            categories,
            stats: None,
        })
    }

    /// Writes each cutoff's `col_stats.json`; a directory another writer published stays.
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
            let stats = table.stats.take().unwrap_or_else(|| table.sinks());
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
        let root = index_dir.join("col_stats");
        std::fs::create_dir_all(&root)?;
        let written = self
            .dirs
            .iter()
            .zip(per_cutoff)
            .map(|(dir, tables)| {
                let tmp = tempfile::Builder::new()
                    .prefix(&format!(".{dir}."))
                    .tempdir_in(&root)?;
                write_col_stats(
                    &OutputTarget::parse(&tmp.path().to_string_lossy(), Dest::Local)?,
                    tables,
                )?;
                Ok((root.join(dir), tmp))
            })
            .collect::<Result<Vec<_>>>()?;
        for (published, tmp) in written {
            if let Err(error) = std::fs::rename(tmp.path(), &published) {
                // Another writer published the same statistics first.
                ensure!(published.join("col_stats.json").is_file(), error);
            }
        }
        Ok(())
    }
}

impl CutoffTable {
    pub fn sinks(&self) -> CutoffSinks {
        CutoffSinks {
            cutoffs: self.cutoffs.clone(),
            temporal: self.temporal,
            stats: self
                .categories
                .iter()
                .map(|categories| {
                    FileColStats::new_with_categories(&self.columns, Arc::clone(categories))
                })
                .collect(),
            active: vec![true; self.cutoffs.len()],
        }
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

macro_rules! forward {
    ($($name:ident($($arg:ident: $ty:ty),*);)*) => {$(
        fn $name(&mut self, $($arg: $ty),*) {
            for (stats, active) in self.stats.iter_mut().zip(&self.active) {
                if *active {
                    stats.$name($($arg),*);
                }
            }
        }
    )*};
}

impl ColumnValueObserver for CutoffSinks {
    forward! {
        observe_i64(col_idx: usize, value: i64);
        observe_f64(col_idx: usize, value: f64);
        observe_string(col_idx: usize, value: &str);
        observe_bytes(col_idx: usize, value: &[u8]);
        observe_decimal(col_idx: usize, value: f64, text: &dyn Fn() -> String);
        observe_timestamp(col_idx: usize, ticks: i64);
        observe_float_list(col_idx: usize, view: &FloatListView);
        observe_int64_list(col_idx: usize, view: &IntListView);
        observe_string_list(col_idx: usize, view: &StringListView);
        observe_null(col_idx: usize);
        observe_parse_failure(col_idx: usize);
        observe_beyond_clock(col_idx: usize);
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
}
