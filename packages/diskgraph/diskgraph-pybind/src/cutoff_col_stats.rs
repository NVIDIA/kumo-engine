// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use ahash::AHashMap;
use anyhow::{anyhow, ensure, Result};
use arrow::array::Array;
use arrow::compute::kernels::length::length;
use arrow::compute::{cast, filter_record_batch};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, UInt64Type};
use arrow_array::{ArrayRef, Int64Array};
use arrow_schema::DataType;
use compact_str::CompactString;
use diskgraph_core::time::convert_trunc;
use diskgraph_ingest::column_scan::{
    at_cutoff, create_times, decode_chunk, node_chunks, stored_tables, StoredTable,
};
use diskgraph_ingest::cutoff_col_stats::CutoffColStatsRequest;
use diskgraph_ingest::graph_config::{read_persisted_config, ColumnSpec, IngestedType};
use pyo3::prelude::*;
use rayon::prelude::*;
use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{
    AddedToken, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer,
};

use crate::bp_col_stats::drive_from_arrow_column;
use crate::{map_anyhow_err, DiskgraphConfigError};

const CHUNK_NODES: u64 = 1 << 16;
const TEXT_CHUNK_NODES: u64 = 1 << 10;
const TEXT_WINDOW_CHUNKS: usize = 1 << 10;
const TOKENIZER_TRAINER: &str = "tokenizers 0.23.2 onig";

type WordCounts = AHashMap<CompactString, u64>;

/// Recomputes `request` from an index's stored rows.
fn from_index(index_dir: &Path, request: &CutoffColStatsRequest) -> Result<()> {
    let config = read_persisted_config(index_dir)?;
    request.validate(&config.tables)?;
    let names: HashSet<String> = request.columns.keys().cloned().collect();
    let (stored_tables, clock) = stored_tables(index_dir, &names)?;
    let mut tables = Vec::new();
    for spec in &config.tables {
        let (Some(mut table), Some(wanted)) =
            (request.table(spec), request.columns.get(&spec.name))
        else {
            continue;
        };
        let stored = &stored_tables[&spec.name];
        ensure!(
            stored
                .columns
                .iter()
                .map(|(name, _)| name)
                .eq(spec.columns.iter().map(|c| &c.name)),
            "table {} stores other columns than its config lists",
            spec.name
        );
        let requested: Vec<usize> = (0..spec.columns.len())
            .filter(|&idx| wanted.contains_key(&spec.columns[idx].name))
            .collect();
        let create_time = stored.create_time;
        let decoded: Vec<usize> = requested.iter().copied().chain(create_time).collect();
        let parts = node_chunks(&stored.reader, CHUNK_NODES)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|chunk| {
                let (_, batch) = decode_chunk(&stored.reader, chunk, &decoded)?;
                let times = create_time.map(|_| create_times(batch.column(requested.len()), clock));
                let mut sinks = table.sinks()?;
                for (stats, cutoff) in sinks.stats.iter_mut().zip(&request.cutoffs) {
                    let keep = at_cutoff(times.as_deref(), batch.num_rows(), *cutoff);
                    let rows = filter_record_batch(&batch, &keep)?;
                    for (position, &col_idx) in requested.iter().enumerate() {
                        // Stored values are observed by their type, as ingest observes them.
                        let column = ColumnSpec {
                            semantic_type: None,
                            ..spec.columns[col_idx].clone()
                        };
                        let (values, column) = match column.ingested_dtype {
                            // Ingest observes a u64's bits as an i64.
                            IngestedType::UInt64 => (
                                Arc::new(Int64Array::from_iter(
                                    rows.column(position)
                                        .as_primitive::<UInt64Type>()
                                        .iter()
                                        .map(|value| value.map(|value| value as i64)),
                                )) as ArrayRef,
                                ColumnSpec {
                                    ingested_dtype: IngestedType::Int64,
                                    ..column
                                },
                            ),
                            _ => (rows.column(position).clone(), column),
                        };
                        drive_from_arrow_column(col_idx, &values, &column, stats, None, clock)?;
                    }
                }
                Ok(sinks)
            })
            .collect::<Result<Vec<_>>>()?;
        for sinks in parts {
            table.add(sinks);
        }
        tables.push(table);
    }
    request.publish(index_dir, clock, tables)
}

/// Recomputes column statistics at cutoffs from an index's stored rows, into `col_stats/<dir>/`.
#[pyfunction]
#[pyo3(signature = (index_dir, request_json, num_threads=None))]
pub fn cutoff_col_stats(
    py: Python<'_>,
    index_dir: &str,
    request_json: &str,
    num_threads: Option<usize>,
) -> PyResult<()> {
    let request: CutoffColStatsRequest = serde_json::from_str(request_json)
        .map_err(|e| DiskgraphConfigError::new_err(format!("invalid request_json: {e}")))?;
    py.detach(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or(0))
            .build()?;
        pool.install(|| from_index(Path::new(index_dir), &request))
    })
    .map_err(map_anyhow_err)
}

fn word_counts(
    py: Python<'_>,
    index_dir: &Path,
    columns: &[(String, String)],
    cutoff: Option<i64>,
    pool: &rayon::ThreadPool,
    pre_tokenizer: &ByteLevel,
) -> PyResult<WordCounts> {
    let config = read_persisted_config(index_dir).map_err(map_anyhow_err)?;
    for (table, column) in columns {
        let spec = config
            .tables
            .iter()
            .find(|spec| spec.name == *table)
            .and_then(|spec| spec.columns.iter().find(|spec| spec.name == *column))
            .ok_or_else(|| {
                DiskgraphConfigError::new_err(format!("index has no column {table}.{column}"))
            })?;
        if spec.ingested_dtype != IngestedType::String {
            return Err(DiskgraphConfigError::new_err(format!(
                "column {table}.{column} holds {:?} values, not strings",
                spec.ingested_dtype
            )));
        }
    }
    let names = columns.iter().map(|(table, _)| table.clone()).collect();
    let (tables, clock) = stored_tables(index_dir, &names).map_err(map_anyhow_err)?;
    let scans = columns
        .iter()
        .map(|(table, column)| {
            let stored = &tables[table];
            let index = stored
                .columns
                .iter()
                .position(|(name, _)| name == column)
                .ok_or_else(|| {
                    anyhow!("table {table} stores other columns than its config lists")
                })?;
            let decoded: Vec<usize> = std::iter::once(index).chain(stored.create_time).collect();
            Ok((stored, decoded))
        })
        .collect::<Result<Vec<_>>>()
        .map_err(map_anyhow_err)?;
    let chunks: Vec<_> = scans
        .iter()
        .enumerate()
        .flat_map(|(scan, (stored, _))| {
            node_chunks(&stored.reader, TEXT_CHUNK_NODES).map(move |chunk| (scan, chunk))
        })
        .collect();
    let maps: Vec<Mutex<WordCounts>> = (0..pool.current_num_threads())
        .map(|_| Mutex::default())
        .collect();
    for window in chunks.chunks(TEXT_WINDOW_CHUNKS) {
        py.detach(|| {
            pool.install(|| {
                window.par_iter().try_for_each(|&(scan, chunk)| {
                    let (stored, decoded) = &scans[scan];
                    let (_, batch) = decode_chunk(&stored.reader, chunk, decoded)?;
                    let times =
                        (batch.num_columns() > 1).then(|| create_times(batch.column(1), clock));
                    let keep = at_cutoff(times.as_deref(), batch.num_rows(), cutoff);
                    let values = batch.column(0).as_string::<i64>();
                    let thread = rayon::current_thread_index().expect("a pool thread");
                    let mut words = maps[thread].lock().expect("unpoisoned");
                    for row in
                        (0..batch.num_rows()).filter(|&row| keep.value(row) && values.is_valid(row))
                    {
                        let mut value = PreTokenizedString::from(values.value(row));
                        pre_tokenizer
                            .pre_tokenize(&mut value)
                            .map_err(|e| anyhow!(e))?;
                        for (word, _, _) in
                            value.get_splits(OffsetReferential::Original, OffsetType::Byte)
                        {
                            *words.entry(word.into()).or_default() += 1;
                        }
                    }
                    Ok(())
                })
            })
        })
        .map_err(map_anyhow_err)?;
        py.check_signals()?;
    }
    Ok(py.detach(|| {
        let mut maps = maps
            .into_iter()
            .map(|map| map.into_inner().expect("unpoisoned"));
        let mut words = maps.next().unwrap_or_default();
        for map in maps {
            for (word, count) in map {
                *words.entry(word).or_default() += count;
            }
        }
        words
    }))
}

/// A byte-level BPE tokenizer's JSON, trained as `tokenizers`' `train_from_iterator` trains it on
/// string columns' non-null values over the rows counted at `cutoff`.
#[pyfunction]
#[pyo3(signature = (
    index_dir, columns, cutoff, *, vocab_size, min_frequency, special_tokens, unk_token,
    num_threads=None
))]
#[allow(clippy::too_many_arguments)]
pub fn train_tokenizer(
    py: Python<'_>,
    index_dir: &str,
    columns: Vec<(String, String)>,
    cutoff: Option<i64>,
    vocab_size: usize,
    min_frequency: u64,
    special_tokens: Vec<String>,
    unk_token: String,
    num_threads: Option<usize>,
) -> PyResult<String> {
    if num_threads == Some(0) {
        return Err(DiskgraphConfigError::new_err(
            "num_threads must be positive",
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.unwrap_or(0))
        .build()
        .map_err(|e| map_anyhow_err(e.into()))?;
    let pre_tokenizer = ByteLevel::default().add_prefix_space(false);
    let words = word_counts(
        py,
        Path::new(index_dir),
        &columns,
        cutoff,
        &pool,
        &pre_tokenizer,
    )?;
    py.detach(|| -> tokenizers::Result<String> {
        let trainer = BpeTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(min_frequency)
            .show_progress(false)
            .special_tokens(
                special_tokens
                    .into_iter()
                    .map(|t| AddedToken::from(t, true))
                    .collect(),
            )
            .build();
        let mut model = BPE::builder().unk_token(unk_token).build()?;
        let special = pool.install(|| trainer.do_train(&words, &mut model))?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
        tokenizer.add_special_tokens(special)?;
        tokenizer.to_string(false)
    })
    .map_err(|e| map_anyhow_err(anyhow!(e)))
}

/// Each node's token count over `columns` of `table`: one per scalar, a list's elements, and a
/// string's pieces under its tokenizer JSON, saturated to `u16`. Written to `path` as one
/// little-endian `u16` per node ID, 0 for a node stored without data.
#[pyfunction]
#[pyo3(signature = (index_dir, table, columns, path, *, num_threads=None))]
pub fn token_lengths(
    py: Python<'_>,
    index_dir: &str,
    table: &str,
    columns: Vec<(String, Option<String>)>,
    path: &str,
    num_threads: Option<usize>,
) -> PyResult<()> {
    if num_threads == Some(0) {
        return Err(DiskgraphConfigError::new_err(
            "num_threads must be positive",
        ));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads.unwrap_or(0))
        .build()
        .map_err(|e| map_anyhow_err(e.into()))?;
    let (mut tables, _) = stored_tables(Path::new(index_dir), &HashSet::from([table.to_string()]))
        .map_err(map_anyhow_err)?;
    let stored = tables.remove(table).expect("opened table");
    let mut decoded = Vec::new();
    let mut tokenizers = Vec::new();
    for (column, tokenizer) in columns {
        let index = stored
            .columns
            .iter()
            .position(|(name, _)| *name == column)
            .ok_or_else(|| {
                DiskgraphConfigError::new_err(format!("index has no column {table}.{column}"))
            })?;
        let kind = stored.columns[index].1;
        if tokenizer.is_some() && kind != diskgraph_rowpack::IngestedType::String {
            return Err(DiskgraphConfigError::new_err(format!(
                "column {table}.{column} holds {kind:?} values, not strings"
            )));
        }
        decoded.push(index);
        let tokenizer = tokenizer
            .map(|json| Tokenizer::from_str(&json))
            .transpose()
            .map_err(|e| DiskgraphConfigError::new_err(format!("invalid tokenizer: {e}")))?;
        tokenizers.push(tokenizer.map(|mut tokenizer| {
            tokenizer.with_padding(None);
            tokenizer
        }));
    }
    let mut chunks: Vec<_> = node_chunks(&stored.reader, TEXT_CHUNK_NODES)
        .map(|chunk| {
            let (first, _) = stored
                .reader
                .shard_id_range(chunk.0)
                .expect("a stored shard");
            (first + chunk.1, chunk)
        })
        .collect();
    chunks.sort_unstable();
    let path = Path::new(path);
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(format!(".{}", uuid::Uuid::new_v4().simple()));
    let mut file = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| map_anyhow_err(e.into()))?,
    );
    let written = (|| -> PyResult<()> {
        let mut next = 0;
        for window in chunks.chunks(TEXT_WINDOW_CHUNKS) {
            let lengths = py
                .detach(|| {
                    pool.install(|| {
                        window
                            .par_iter()
                            .map(|&(_, chunk)| chunk_lengths(&stored, chunk, &decoded, &tokenizers))
                            .collect::<Result<Vec<_>>>()
                    })
                })
                .map_err(map_anyhow_err)?;
            for (&(first, _), lengths) in window.iter().zip(lengths) {
                let zeros = std::iter::repeat_n(0, (first - next) as usize);
                for length in zeros.chain(lengths.iter().copied()) {
                    file.write_all(&u16::to_le_bytes(length))?;
                }
                next = first + lengths.len() as u64;
            }
            py.check_signals()?;
        }
        file.flush()?;
        file.get_ref().sync_all()?;
        Ok(fs::rename(&temporary, path)?)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written
}

fn chunk_lengths(
    stored: &StoredTable,
    chunk: (usize, u64, u64),
    decoded: &[usize],
    tokenizers: &[Option<Tokenizer>],
) -> Result<Vec<u16>> {
    let (nodes, batch) = decode_chunk(&stored.reader, chunk, decoded)?;
    let mut totals = vec![0u64; batch.num_rows()];
    for (column, tokenizer) in batch.columns().iter().zip(tokenizers) {
        match (tokenizer, column.data_type()) {
            (Some(tokenizer), _) => {
                for (total, value) in totals.iter_mut().zip(column.as_string::<i64>()) {
                    if let Some(value) = value {
                        let pieces = tokenizer.encode_fast(value, true).map_err(|e| anyhow!(e))?;
                        *total += pieces.len() as u64;
                    }
                }
            }
            (None, DataType::List(_) | DataType::LargeList(_)) => {
                let counts = cast(&length(column)?, &DataType::Int64)?;
                for (total, count) in totals.iter_mut().zip(counts.as_primitive::<Int64Type>()) {
                    *total += count.unwrap_or(0) as u64;
                }
            }
            (None, _) => totals.iter_mut().for_each(|total| *total += 1),
        }
    }
    let mut lengths = vec![0; (chunk.2 - chunk.1) as usize];
    for (node, total) in nodes.into_iter().zip(totals) {
        lengths[(node - chunk.1) as usize] = total.min(u16::MAX.into()) as u16;
    }
    Ok(lengths)
}

/// `value` in index clock `from` as ticks of index clock `to`, truncated.
#[pyfunction]
pub fn convert_ticks(value: i64, from: &str, to: &str) -> PyResult<i64> {
    let parse = crate::tt_builder::parse_index_clock;
    Ok(convert_trunc(value, parse(from)?, parse(to)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cutoff_col_stats, m)?)?;
    m.add_function(wrap_pyfunction!(convert_ticks, m)?)?;
    m.add_function(wrap_pyfunction!(train_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(token_lengths, m)?)?;
    m.add("TOKENIZER_TRAINER", TOKENIZER_TRAINER)
}
