// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Realized metrics of batch predictions against their labels, joined through
//! hash-partitioned spill files; metrics follow torchmetrics and PyG.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use arrow::array::{
    Array, AsArray, BooleanArray, FixedSizeBinaryArray, Float64Array, ListBuilder, StringBuilder,
    TimestampNanosecondArray, UInt32Array,
};
use arrow::compute::{cast, interleave, take, take_record_batch};
use arrow::datatypes::{Float64Type, TimestampNanosecondType};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};

use crate::bp_col_stats::{column, float64s, row_group_batches, rows};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EvaluateTask {
    Binary,
    Regression,
    Forecasting,
    Multiclass,
    Multilabel,
    Ranking,
}

impl EvaluateTask {
    fn grouped(self) -> bool {
        !matches!(self, Self::Binary | Self::Regression)
    }

    fn sets(self) -> bool {
        matches!(self, Self::Multilabel | Self::Ranking)
    }

    /// One actuals row per matched prediction row.
    fn exploded(self) -> bool {
        matches!(self, Self::Forecasting | Self::Multilabel)
    }

    fn score(self) -> &'static str {
        match self {
            Self::Binary => "True_PROB",
            Self::Regression | Self::Forecasting => "TARGET_PRED",
            Self::Multiclass | Self::Multilabel | Self::Ranking => "SCORE",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct EvaluateSpec {
    pub task: EvaluateTask,
    pub prediction_keys: Vec<String>,
    pub label_keys: Vec<String>,
    pub label_target: String,
    #[serde(default)]
    pub actuals_dir: Option<PathBuf>,
    /// (input, output) column names.
    #[serde(default)]
    pub actual_columns: Vec<(String, String)>,
    #[serde(default)]
    pub prediction_columns: Vec<(String, String)>,
    /// (output column, UTC nanoseconds).
    #[serde(default)]
    pub constant_timestamps: Vec<(String, i64)>,
    #[serde(default)]
    pub ks: Vec<usize>,
    pub spill_dir: PathBuf,
    #[serde(default = "default_partition_rows")]
    pub partition_rows: usize,
}

fn default_partition_rows() -> usize {
    1_000_000
}

const MAX_CONCURRENT_JOINS: usize = 8;
const KEY: &str = "__key";
const LABEL_SEPARATOR: char = '|';

type Batches = Box<dyn Iterator<Item = Result<RecordBatch>> + Send>;

fn common_type(a: &DataType, b: &DataType) -> DataType {
    let int = |t: &DataType| t.is_integer() || *t == DataType::Boolean;
    if a.is_temporal() || b.is_temporal() {
        DataType::Timestamp(TimeUnit::Nanosecond, None)
    } else if int(a) && int(b) {
        DataType::Int64
    } else if [a, b].iter().all(|t| int(t) || t.is_floating()) {
        DataType::Float64
    } else {
        DataType::Utf8
    }
}

/// 128-bit hash of each row of `arrays` cast to `types`; `None` on a null.
fn hashes(arrays: &[&ArrayRef], types: &[DataType]) -> Result<Vec<Option<u128>>> {
    let arrays = arrays
        .iter()
        .zip(types)
        .map(|(array, to)| match (to, array.data_type()) {
            (DataType::Timestamp(unit, _), DataType::Timestamp(_, tz)) => {
                cast(array, &DataType::Timestamp(*unit, tz.clone()))
            }
            _ => cast(array, to),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rows = rows(&arrays)?;
    Ok((0..rows.num_rows())
        .map(|row| {
            arrays.iter().all(|array| array.is_valid(row)).then(|| {
                let half = |seed: u64| {
                    let mut hasher = DefaultHasher::new();
                    seed.hash(&mut hasher);
                    rows.row(row).hash(&mut hasher);
                    hasher.finish()
                };
                u128::from(half(1)) << 64 | u128::from(half(2))
            })
        })
        .collect())
}

fn classes(array: &ArrayRef, class: &DataType) -> Result<Vec<Option<u64>>> {
    let hashes = hashes(&[array], std::slice::from_ref(class))?;
    Ok(hashes.into_iter().map(|h| h.map(|h| h as u64)).collect())
}

fn floats(array: &ArrayRef) -> Result<Vec<Option<f64>>> {
    Ok(float64s(array)?.iter().collect())
}

fn keys(batch: &RecordBatch) -> Result<impl Iterator<Item = u128> + '_> {
    let keys = column(batch, KEY)?.as_fixed_size_binary();
    Ok((0..keys.len()).map(|row| u128::from_le_bytes(keys.value(row).try_into().unwrap())))
}

/// Sorted, distinct class hashes per label; a set is a list or `|`-separated.
fn label_classes(target: &ArrayRef, sets: bool, class: &DataType) -> Result<Vec<Option<Vec<u64>>>> {
    if !sets {
        let classes = classes(target, class)?;
        return Ok(classes.into_iter().map(|c| c.map(|c| vec![c])).collect());
    }
    let target = match target.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => split(target)?,
        _ => target.clone(),
    };
    let item = Field::new_list_field(class.clone(), true);
    let list = cast(&target, &DataType::List(Arc::new(item)))?;
    let list = list.as_list::<i32>();
    let items = classes(list.values(), class)?;
    let offsets = list.value_offsets();
    Ok((0..list.len())
        .map(|row| {
            list.is_valid(row).then(|| {
                let range = offsets[row] as usize..offsets[row + 1] as usize;
                let mut row: Vec<u64> = items[range].iter().flatten().copied().collect();
                row.sort_unstable();
                row.dedup();
                row
            })
        })
        .collect())
}

fn split(strings: &ArrayRef) -> Result<ArrayRef> {
    let strings = cast(strings, &DataType::Utf8)?;
    let mut lists = ListBuilder::new(StringBuilder::new());
    for value in strings.as_string::<i32>() {
        for item in value.into_iter().flat_map(|v| v.split(LABEL_SEPARATOR)) {
            if !item.is_empty() {
                lists.values().append_value(item);
            }
        }
        lists.append(value.is_some());
    }
    Ok(Arc::new(lists.finish()))
}

fn label_steps(target: &ArrayRef) -> Result<Vec<Option<Vec<Option<f64>>>>> {
    let item = Field::new_list_field(DataType::Float64, true);
    let list = cast(target, &DataType::List(Arc::new(item)))?;
    let list = list.as_list::<i32>();
    let values = list.values().as_primitive::<Float64Type>();
    let offsets = list.value_offsets();
    Ok((0..list.len())
        .map(|row| {
            list.is_valid(row).then(|| {
                (offsets[row] as usize..offsets[row + 1] as usize)
                    .map(|i| values.is_valid(i).then(|| values.value(i)))
                    .collect()
            })
        })
        .collect())
}

type Sink = (PathBuf, Mutex<Option<FileWriter<BufWriter<File>>>>);

fn spill((path, writer): &Sink, batch: &RecordBatch) -> Result<()> {
    let mut writer = writer.lock().unwrap();
    if writer.is_none() {
        let file = BufWriter::new(File::create(path)?);
        *writer = Some(FileWriter::try_new(file, &batch.schema())?);
    }
    Ok(writer.as_mut().unwrap().write(batch)?)
}

fn unspill((path, writer): Sink) -> Result<Batches> {
    let Some(mut writer) = writer.into_inner().unwrap() else {
        return Ok(Box::new(std::iter::empty()));
    };
    writer.finish()?;
    let reader = FileReader::try_new(File::open(path)?, None)?;
    Ok(Box::new(reader.map(|batch| Ok(batch?))))
}

type RowGroups = Vec<(usize, usize)>;

fn row_groups(paths: &[&Path]) -> Result<(RowGroups, usize, SchemaRef)> {
    let (mut groups, mut rows, mut schema) = (Vec::new(), 0, Arc::new(Schema::empty()));
    for (file, path) in paths.iter().enumerate() {
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
        schema = builder.schema().clone();
        rows += builder.metadata().file_metadata().num_rows() as usize;
        groups.extend((0..builder.metadata().num_row_groups()).map(|group| (file, group)));
    }
    Ok((groups, rows, schema))
}

/// Routes each row with a non-null key to its partition's sink.
fn scatter(
    paths: &[&Path],
    groups: &RowGroups,
    keys: &[String],
    types: &[DataType],
    keep: &[String],
    sinks: &[Sink],
) -> Result<()> {
    let columns: Vec<String> = keys.iter().chain(keep).cloned().collect();
    groups.par_iter().try_for_each(|&(file, group)| {
        for batch in row_group_batches(paths[file], group, &[], &columns)? {
            let batch = batch?;
            let key_columns = keys
                .iter()
                .map(|name| column(&batch, name))
                .collect::<Result<Vec<_>>>()?;
            let hashes = hashes(&key_columns, types)?;
            let mut parts = vec![Vec::new(); sinks.len()];
            for (row, key) in hashes.iter().enumerate() {
                if let Some(key) = key {
                    parts[((key >> 64) as u64 % sinks.len() as u64) as usize].push(row as u32);
                }
            }
            let key = hashes.into_iter().map(|key| key.map(u128::to_le_bytes));
            let key = FixedSizeBinaryArray::try_from_sparse_iter_with_size(key, 16)?;
            let mut out = Vec::new();
            for name in keep {
                out.push((name.as_str(), column(&batch, name)?.clone(), true));
            }
            out.push((KEY, Arc::new(key), true));
            let batch = RecordBatch::try_from_iter_with_nullable(out)?;
            for (sink, rows) in sinks.iter().zip(parts) {
                if !rows.is_empty() {
                    spill(sink, &take_record_batch(&batch, &UInt32Array::from(rows))?)?;
                }
            }
        }
        Ok(())
    })
}

type Step = [f64; 6];
const STEP_NAMES: [&str; 6] = [
    "rows",
    "abs_error",
    "squared_error",
    "error",
    "target",
    "target_squared",
];

#[derive(Default)]
struct Accumulator {
    rows: u64,
    labelled: u64,
    /// Binary: score order key shifted left, label in bit 0.
    pairs: Vec<u64>,
    count: u64,
    sum: f64,
    squares: f64,
    steps: Vec<Step>,
    /// Per class: pairs and true positive, false positive, false negative.
    classes: BTreeMap<u64, (Vec<u64>, [u64; 3])>,
    /// Per k: MAP, precision, recall and F1 sums.
    at_k: Vec<[f64; 4]>,
    k: usize,
}

impl Accumulator {
    fn merge(mut self, mut other: Self) -> Self {
        self.rows += other.rows;
        self.labelled += other.labelled;
        self.pairs.append(&mut other.pairs);
        self.count += other.count;
        self.sum += other.sum;
        self.squares += other.squares;
        add(&mut self.steps, &other.steps);
        for (class, (mut pairs, counts)) in other.classes {
            let (all, total) = self.classes.entry(class).or_default();
            all.append(&mut pairs);
            total.iter_mut().zip(counts).for_each(|(t, c)| *t += c);
        }
        add(&mut self.at_k, &other.at_k);
        self.k = self.k.max(other.k);
        self
    }

    fn error(&mut self, prediction: f64, label: f64) {
        let error = prediction - label;
        self.count += 1;
        self.sum += error.abs();
        self.squares += error * error;
    }

    fn step(&mut self, step: usize, prediction: f64, label: f64) {
        self.error(prediction, label);
        if self.steps.len() <= step {
            self.steps.resize(step + 1, [0.0; 6]);
        }
        let error = prediction - label;
        let values = [1.0, error.abs(), error * error, error, label, label * label];
        self.steps[step]
            .iter_mut()
            .zip(values)
            .for_each(|(s, v)| *s += v);
    }

    fn label(&mut self, entry: &Entry, label: bool) {
        let Some(class) = entry.class else { return };
        let (pairs, counts) = self.classes.entry(class).or_default();
        if let Some(score) = entry.score.filter(|s| s.is_finite()) {
            pairs.push(order_key(score) << 1 | u64::from(label));
        }
        match (entry.predicted, label) {
            (true, true) => counts[0] += 1,
            (true, false) => counts[1] += 1,
            (false, true) => counts[2] += 1,
            (false, false) => {}
        }
    }

    fn rank(&mut self, ranked: &[Entry], items: &[u64], ks: &[usize]) {
        let labels = items.len() as f64;
        let (mut hits, mut numerator, mut at) = (0.0, 0.0, Vec::with_capacity(ranked.len()));
        for (rank, entry) in ranked.iter().enumerate() {
            if entry.class.is_some_and(|c| items.binary_search(&c).is_ok()) {
                hits += 1.0;
                numerator += hits / (rank + 1) as f64;
            }
            at.push((hits, numerator));
        }
        if self.at_k.len() < ks.len() {
            self.at_k.resize(ks.len(), [0.0; 4]);
        }
        for (sums, &k) in self.at_k.iter_mut().zip(ks) {
            let (hits, numerator) = at[k.min(at.len()) - 1];
            let (precision, recall) = (hits / k as f64, hits / labels);
            let f1 = if hits > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            let values = [numerator / labels.min(k as f64), precision, recall, f1];
            sums.iter_mut().zip(values).for_each(|(s, v)| *s += v);
        }
        self.count += 1;
    }
}

fn add<const N: usize>(sums: &mut Vec<[f64; N]>, other: &[[f64; N]]) {
    if sums.len() < other.len() {
        sums.resize(other.len(), [0.0; N]);
    }
    for (sum, values) in sums.iter_mut().zip(other) {
        sum.iter_mut().zip(values).for_each(|(s, v)| *s += v);
    }
}

fn order_key(score: f64) -> u64 {
    let bits = (score as f32).to_bits();
    u64::from(if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    })
}

struct Entry {
    score: Option<f64>,
    class: Option<u64>,
    time: i64,
    predicted: bool,
    at: (usize, usize),
}

fn grouped_entries(
    batch: &RecordBatch,
    index: usize,
    task: EvaluateTask,
    class: &DataType,
) -> Result<Vec<Entry>> {
    let rows = batch.num_rows();
    let scores = floats(column(batch, task.score())?)?;
    let classes = match task {
        EvaluateTask::Forecasting => vec![None; rows],
        _ => classes(column(batch, "CLASS")?, class)?,
    };
    let times: Vec<i64> = match task {
        EvaluateTask::Forecasting => {
            let time = cast(
                column(batch, "TIMESTAMP")?,
                &DataType::Timestamp(TimeUnit::Nanosecond, None),
            )?;
            time.as_primitive::<TimestampNanosecondType>()
                .values()
                .to_vec()
        }
        _ => vec![0; rows],
    };
    let predicted = match task {
        EvaluateTask::Multilabel => floats(column(batch, "PREDICTED")?)?,
        _ => vec![None; rows],
    };
    Ok((0..rows)
        .map(|row| Entry {
            score: scores[row],
            class: classes[row],
            time: times[row],
            predicted: predicted[row].is_some_and(|p| p != 0.0),
            at: (index, row),
        })
        .collect())
}

/// The first label of a key counts.
fn join(
    predictions: Batches,
    labels: Batches,
    spec: &EvaluateSpec,
    class: &DataType,
    actuals: Option<PathBuf>,
) -> Result<Accumulator> {
    let task = spec.task;
    let mut first = HashMap::new();
    let mut groups: HashMap<u128, Vec<Entry>> = HashMap::new();
    let mut kept: Vec<Vec<ArrayRef>> = Vec::new();
    for (index, batch) in predictions.enumerate() {
        let batch = batch?;
        kept.push(
            spec.prediction_columns
                .iter()
                .map(|(name, _)| column(&batch, name).cloned())
                .collect::<Result<_>>()?,
        );
        if !task.grouped() {
            let scores = floats(column(&batch, task.score())?)?;
            for (row, (key, score)) in keys(&batch)?.zip(scores).enumerate() {
                first.entry(key).or_insert((score, (index, row)));
            }
            continue;
        }
        let entries = grouped_entries(&batch, index, task, class)?;
        for (key, entry) in keys(&batch)?.zip(entries) {
            groups.entry(key).or_default().push(entry);
        }
    }
    let mut acc = Accumulator {
        rows: (first.len() + groups.len()) as u64,
        k: groups.values().map(Vec::len).max().unwrap_or(0),
        ..Default::default()
    };

    let mut writer: Option<ArrowWriter<File>> = None;
    for batch in labels {
        let batch = batch?;
        let target = column(&batch, &spec.label_target)?;
        let (mut rows, mut picked, mut values) = (Vec::new(), Vec::new(), Vec::new());
        if !task.grouped() {
            for ((row, key), label) in keys(&batch)?.enumerate().zip(floats(target)?) {
                let Some(label) = label else { continue };
                let Some((score, at)) = first.remove(&key) else {
                    continue;
                };
                acc.labelled += 1;
                rows.push(row as u32);
                picked.push(at);
                match score {
                    Some(score) if task == EvaluateTask::Binary => {
                        if score.is_finite() {
                            acc.pairs
                                .push(order_key(score) << 1 | u64::from(label != 0.0));
                        }
                    }
                    Some(prediction) => acc.error(prediction, label),
                    None => {}
                }
            }
        } else if task == EvaluateTask::Forecasting {
            for ((row, key), steps) in keys(&batch)?.enumerate().zip(label_steps(target)?) {
                let Some(steps) = steps else { continue };
                let Some(mut entries) = groups.remove(&key) else {
                    continue;
                };
                acc.labelled += 1;
                entries.sort_by_key(|entry| entry.time);
                for (step, (entry, label)) in entries.iter().zip(steps).enumerate() {
                    if let (Some(prediction), Some(label)) = (entry.score, label) {
                        acc.step(step, prediction, label);
                    }
                    rows.push(row as u32);
                    picked.push(entry.at);
                    values.push(label);
                }
            }
        } else {
            let labels = label_classes(target, task.sets(), class)?;
            for ((row, key), items) in keys(&batch)?.enumerate().zip(labels) {
                let Some(items) = items else { continue };
                let Some(mut ranked) = groups.remove(&key) else {
                    continue;
                };
                acc.labelled += 1;
                // Highest score first; ties keep their order.
                ranked.sort_by(|a, b| {
                    let score = |entry: &Entry| entry.score.unwrap_or(f64::NEG_INFINITY);
                    score(b).total_cmp(&score(a))
                });
                let hit =
                    |entry: &Entry| entry.class.is_some_and(|c| items.binary_search(&c).is_ok());
                match task {
                    EvaluateTask::Multiclass => {
                        acc.count += 1;
                        acc.sum += f64::from(u8::from(hit(&ranked[0])));
                    }
                    EvaluateTask::Ranking if !items.is_empty() => {
                        acc.rank(&ranked, &items, &spec.ks)
                    }
                    _ => {}
                }
                if task != EvaluateTask::Multilabel {
                    rows.push(row as u32);
                    picked.push(ranked[0].at);
                    continue;
                }
                for entry in &ranked {
                    acc.label(entry, hit(entry));
                    rows.push(row as u32);
                    picked.push(entry.at);
                    values.push(Some(f64::from(u8::from(hit(entry)))));
                }
            }
        }
        let Some(path) = actuals.as_ref().filter(|_| !rows.is_empty()) else {
            continue;
        };
        let out = actuals_batch(spec, &batch, rows, &values, &kept, &picked)?;
        if writer.is_none() {
            writer = Some(ArrowWriter::try_new(
                File::create(path)?,
                out.schema(),
                None,
            )?);
        }
        writer.as_mut().unwrap().write(&out)?;
    }
    if let Some(writer) = writer {
        writer.close()?;
    }
    Ok(acc)
}

/// The matched label rows with their predictions' columns.
fn actuals_batch(
    spec: &EvaluateSpec,
    labels: &RecordBatch,
    rows: Vec<u32>,
    values: &[Option<f64>],
    kept: &[Vec<ArrayRef>],
    picked: &[(usize, usize)],
) -> Result<RecordBatch> {
    let rows = UInt32Array::from(rows);
    let mut out = spec
        .actual_columns
        .iter()
        .map(|(name, out)| {
            let array = if spec.task.exploded() && *name == spec.label_target {
                exploded_target(spec.task, values)
            } else {
                take(column(labels, name)?, &rows, None)?
            };
            Ok((out.as_str(), array, true))
        })
        .collect::<Result<Vec<_>>>()?;
    for (name, ns) in &spec.constant_timestamps {
        let values = vec![*ns; picked.len()];
        let array = TimestampNanosecondArray::from(values).with_timezone("UTC");
        out.push((name.as_str(), Arc::new(array), true));
    }
    for (column, (_, name)) in spec.prediction_columns.iter().enumerate() {
        let arrays: Vec<&dyn Array> = kept.iter().map(|batch| batch[column].as_ref()).collect();
        out.push((name.as_str(), interleave(&arrays, picked)?, true));
    }
    Ok(RecordBatch::try_from_iter_with_nullable(out)?)
}

fn exploded_target(task: EvaluateTask, values: &[Option<f64>]) -> ArrayRef {
    if task == EvaluateTask::Multilabel {
        let hits: BooleanArray = values.iter().map(|v| v.map(|v| v != 0.0)).collect();
        return Arc::new(hits);
    }
    Arc::new(Float64Array::from(values.to_vec()))
}

/// Exact AUROC, AP and AUPRC; None unless both labels occur.
fn ranking_metrics(pairs: &mut [u64]) -> [Option<f64>; 3] {
    pairs.par_sort_unstable();
    let positives = pairs.iter().filter(|pair| *pair & 1 == 1).count() as f64;
    let negatives = pairs.len() as f64 - positives;
    if positives == 0.0 || negatives == 0.0 {
        return [None; 3];
    }
    let groups: Vec<(f64, f64)> = pairs
        .chunk_by(|a, b| a >> 1 == b >> 1)
        .map(|group| {
            let positive = group.iter().filter(|pair| *pair & 1 == 1).count() as f64;
            (positive, group.len() as f64 - positive)
        })
        .collect();
    let (mut below, mut area) = (0.0, 0.0);
    for &(positive, negative) in &groups {
        area += positive * (below + negative / 2.0);
        below += negative;
    }
    let (mut true_positive, mut false_positive) = (0.0, 0.0);
    let (mut previous_recall, mut previous_precision) = (0.0, 1.0);
    let (mut precision_sum, mut trapezoids) = (0.0, 0.0);
    for &(positive, negative) in groups.iter().rev() {
        true_positive += positive;
        false_positive += negative;
        let recall = true_positive / positives;
        let precision = true_positive / (true_positive + false_positive);
        precision_sum += (recall - previous_recall) * precision;
        trapezoids += (recall - previous_recall) * (precision + previous_precision) / 2.0;
        (previous_recall, previous_precision) = (recall, precision);
    }
    [
        Some(area / (positives * negatives)),
        Some(precision_sum),
        Some(trapezoids),
    ]
}

fn metrics(mut acc: Accumulator, spec: &EvaluateSpec) -> JsonValue {
    let ratio = |num: f64| (acc.count > 0).then(|| num / acc.count as f64);
    match spec.task {
        EvaluateTask::Binary => {
            let [auroc, ap, _] = ranking_metrics(&mut acc.pairs);
            json!({"auroc": auroc, "ap": ap})
        }
        EvaluateTask::Regression | EvaluateTask::Forecasting => {
            json!({"mae": ratio(acc.sum), "rmse": ratio(acc.squares).map(f64::sqrt)})
        }
        EvaluateTask::Multiclass => json!({"acc": ratio(acc.sum)}),
        EvaluateTask::Multilabel => multilabel_metrics(acc.classes),
        EvaluateTask::Ranking => {
            let mut out = serde_json::Map::new();
            for (i, &k) in spec.ks.iter().enumerate().filter(|(_, &k)| k <= acc.k) {
                let sums = acc.at_k.get(i).copied().unwrap_or_default();
                for (name, sum) in ["map", "precision", "recall", "f1"].iter().zip(sums) {
                    out.insert(format!("{name}@{k}"), json!(ratio(sum)));
                }
            }
            JsonValue::Object(out)
        }
    }
}

fn multilabel_metrics(classes: BTreeMap<u64, (Vec<u64>, [u64; 3])>) -> JsonValue {
    let (mut pooled, mut per_class, mut f1, mut total) =
        (Vec::new(), Vec::new(), Vec::new(), [0u64; 3]);
    for (_, (mut pairs, counts)) in classes {
        let [tp, fp, fn_] = counts.map(|c| c as f64);
        if tp + fp + fn_ > 0.0 {
            f1.push(2.0 * tp / (2.0 * tp + fp + fn_));
        }
        total.iter_mut().zip(counts).for_each(|(t, c)| *t += c);
        per_class.push(ranking_metrics(&mut pairs));
        pooled.append(&mut pairs);
    }
    let mean = |values: Vec<f64>| {
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    let macro_ = |i: usize| mean(per_class.iter().filter_map(|m| m[i]).collect());
    let [auroc, ap, auprc] = ranking_metrics(&mut pooled);
    let [tp, fp, fn_] = total.map(|c| c as f64);
    let share = |num: f64, den: f64| if den > 0.0 { num / den } else { 0.0 };
    json!({
        "auroc_macro": macro_(0), "auroc_micro": auroc,
        "ap_macro": macro_(1), "ap_micro": ap,
        "auprc_macro": macro_(2), "auprc_micro": auprc,
        "f1_micro": share(2.0 * tp, 2.0 * tp + fp + fn_),
        "precision_micro": share(tp, tp + fp),
        "recall_micro": share(tp, tp + fn_),
        "f1_macro": mean(f1).unwrap_or(0.0),
    })
}

fn by_horizon(steps: &[Step]) -> JsonValue {
    let columns = STEP_NAMES.iter().enumerate().map(|(i, name)| {
        (
            format!("horizon_{name}"),
            json!(steps.iter().map(|step| step[i]).collect::<Vec<_>>()),
        )
    });
    JsonValue::Object(columns.collect())
}

pub fn evaluate_prediction_parquet(
    prediction_paths: &[&Path],
    label_paths: &[&Path],
    spec: &EvaluateSpec,
) -> Result<JsonValue> {
    if spec.prediction_keys.is_empty() || spec.prediction_keys.len() != spec.label_keys.len() {
        bail!("prediction_keys and label_keys must pair up");
    }
    let (prediction_groups, prediction_rows, prediction_schema) = row_groups(prediction_paths)?;
    let (label_groups, label_rows, label_schema) = row_groups(label_paths)?;
    let types = spec
        .prediction_keys
        .iter()
        .zip(&spec.label_keys)
        .map(|(p, l)| {
            Ok(common_type(
                prediction_schema.field_with_name(p)?.data_type(),
                label_schema.field_with_name(l)?.data_type(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut prediction_columns = vec![spec.task.score().to_string()];
    prediction_columns.extend(spec.prediction_columns.iter().map(|(name, _)| name.clone()));
    let class = match spec.task {
        EvaluateTask::Multiclass | EvaluateTask::Multilabel | EvaluateTask::Ranking => {
            prediction_columns.push("CLASS".to_string());
            let mut target = label_schema
                .field_with_name(&spec.label_target)?
                .data_type();
            if spec.task.sets() {
                target = match target {
                    DataType::List(item)
                    | DataType::LargeList(item)
                    | DataType::FixedSizeList(item, _) => item.data_type(),
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => &DataType::Utf8,
                    _ => bail!("label sets must be lists or strings, got {target}"),
                };
            }
            common_type(
                prediction_schema.field_with_name("CLASS")?.data_type(),
                target,
            )
        }
        _ => DataType::Null,
    };
    match spec.task {
        EvaluateTask::Forecasting => prediction_columns.push("TIMESTAMP".to_string()),
        EvaluateTask::Multilabel => prediction_columns.push("PREDICTED".to_string()),
        _ => {}
    }
    prediction_columns.sort();
    prediction_columns.dedup();
    let mut label_columns = vec![spec.label_target.clone()];
    label_columns.extend(spec.actual_columns.iter().map(|(name, _)| name.clone()));
    label_columns.sort();
    label_columns.dedup();

    let partitions = prediction_rows
        .max(label_rows)
        .div_ceil(spec.partition_rows.max(1))
        .max(1);
    std::fs::create_dir_all(&spec.spill_dir)?;
    if let Some(dir) = &spec.actuals_dir {
        std::fs::create_dir_all(dir)?;
    }
    let sinks = |side: &str| -> Vec<Sink> {
        (0..partitions)
            .map(|p| {
                (
                    spec.spill_dir.join(format!("{side}-{p:05}.arrow")),
                    Mutex::default(),
                )
            })
            .collect()
    };
    let (predictions, labels) = (sinks("predictions"), sinks("labels"));
    scatter(
        prediction_paths,
        &prediction_groups,
        &spec.prediction_keys,
        &types,
        &prediction_columns,
        &predictions,
    )?;
    scatter(
        label_paths,
        &label_groups,
        &spec.label_keys,
        &types,
        &label_columns,
        &labels,
    )?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(MAX_CONCURRENT_JOINS)
        .build()?;
    let acc = pool.install(|| {
        predictions
            .into_par_iter()
            .zip(labels)
            .enumerate()
            .map(|(p, (predictions, labels))| {
                let actuals =
                    (spec.actuals_dir.as_ref()).map(|dir| dir.join(format!("part-{p:05}.parquet")));
                join(
                    unspill(predictions)?,
                    unspill(labels)?,
                    spec,
                    &class,
                    actuals,
                )
            })
            .try_reduce(Accumulator::default, |a, b| Ok(a.merge(b)))
    })?;
    let _ = std::fs::remove_dir_all(&spec.spill_dir);
    let (rows, labelled) = (acc.rows, acc.labelled);
    let mut out = json!({"rows": rows, "labelled": labelled});
    if spec.task == EvaluateTask::Forecasting {
        out["by_horizon"] = by_horizon(&acc.steps);
    }
    out["metrics"] = metrics(acc, spec);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Float64Array, Int64Array, Int64Builder, ListBuilder, StringArray, TimestampMicrosecondArray,
    };
    use parquet::file::properties::WriterProperties;
    use tempfile::tempdir;

    fn write(path: &Path, columns: Vec<(&str, ArrayRef)>, rows_per_group: usize) {
        let batch = RecordBatch::try_from_iter(columns).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_group))
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(path).unwrap(), batch.schema(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn spec(task: EvaluateTask, dir: &Path, partition_rows: usize) -> EvaluateSpec {
        EvaluateSpec {
            task,
            prediction_keys: vec!["ENTITY".into(), "ANCHOR_TIMESTAMP".into()],
            label_keys: vec!["ENTITY1".into(), "TIMESTAMP".into()],
            label_target: "TARGET1".into(),
            actuals_dir: Some(dir.join("actuals")),
            actual_columns: vec![
                ("ENTITY1".into(), "ENTITY".into()),
                ("TIMESTAMP".into(), "ANCHOR_TIMESTAMP".into()),
                ("TARGET1".into(), "TARGET".into()),
            ],
            prediction_columns: vec![],
            constant_timestamps: Vec::new(),
            ks: Vec::new(),
            spill_dir: dir.join("spill"),
            partition_rows,
        }
    }

    fn evaluate(dir: &Path, spec: &EvaluateSpec) -> JsonValue {
        let predictions = dir.join("predictions.parquet");
        let labels = dir.join("labels.parquet");
        evaluate_prediction_parquet(&[predictions.as_path()], &[labels.as_path()], spec).unwrap()
    }

    fn actuals(dir: &Path) -> Vec<RecordBatch> {
        std::fs::read_dir(dir.join("actuals"))
            .unwrap()
            .flat_map(|entry| {
                let file = File::open(entry.unwrap().path()).unwrap();
                ParquetRecordBatchReaderBuilder::try_new(file)
                    .unwrap()
                    .build()
                    .unwrap()
                    .map(Result::unwrap)
            })
            .collect()
    }

    fn actual_column<T: arrow::datatypes::ArrowPrimitiveType>(
        dir: &Path,
        name: &str,
    ) -> Vec<T::Native> {
        actuals(dir)
            .iter()
            .flat_map(|batch| {
                let array = cast(column(batch, name).unwrap(), &T::DATA_TYPE).unwrap();
                array.as_primitive::<T>().values().to_vec()
            })
            .collect()
    }

    /// Mann-Whitney AUROC with ties counted half, by brute force.
    fn brute_auroc(scores: &[f32], labels: &[bool]) -> f64 {
        let (mut wins, mut pairs) = (0.0, 0.0);
        for (i, &si) in scores.iter().enumerate() {
            for (j, &sj) in scores.iter().enumerate() {
                if labels[i] && !labels[j] {
                    pairs += 1.0;
                    wins += if si > sj {
                        1.0
                    } else if si == sj {
                        0.5
                    } else {
                        0.0
                    };
                }
            }
        }
        wins / pairs
    }

    #[test]
    fn binary_join_matches_brute_force_in_one_and_many_partitions() {
        let n = 3000i64;
        let entity: Vec<i64> = (0..n).collect();
        // Several anchors; tz-aware microseconds against naive nanoseconds.
        let anchor_us: Vec<i64> = (0..n).map(|i| (i % 3) * 86_400_000_000).collect();
        let scores: Vec<f32> = (0..n).map(|i| ((i * 7919) % 101) as f32 / 100.0).collect();
        let labels: Vec<bool> = (0..n).map(|i| (i * 31 + i / 7) % 5 < 2).collect();
        // Predictions for every entity; labels for all but the last 500.
        let m = (n - 500) as usize;
        for partition_rows in [1_000_000, 700] {
            let dir = tempdir().unwrap();
            write(
                &dir.path().join("predictions.parquet"),
                vec![
                    (
                        "ENTITY",
                        Arc::new(Int64Array::from(entity.clone())) as ArrayRef,
                    ),
                    (
                        "ANCHOR_TIMESTAMP",
                        Arc::new(
                            TimestampMicrosecondArray::from(anchor_us.clone()).with_timezone("UTC"),
                        ) as ArrayRef,
                    ),
                    (
                        "True_PROB",
                        Arc::new(arrow::array::Float32Array::from(scores.clone())) as ArrayRef,
                    ),
                ],
                256,
            );
            write(
                &dir.path().join("labels.parquet"),
                vec![
                    (
                        "ENTITY1",
                        Arc::new(Int64Array::from(
                            entity[..m].iter().rev().copied().collect::<Vec<_>>(),
                        )) as ArrayRef,
                    ),
                    (
                        "TIMESTAMP",
                        Arc::new(TimestampNanosecondArray::from(
                            anchor_us[..m]
                                .iter()
                                .rev()
                                .map(|t| t * 1000)
                                .collect::<Vec<_>>(),
                        )) as ArrayRef,
                    ),
                    (
                        "TARGET1",
                        Arc::new(BooleanArray::from(
                            labels[..m].iter().rev().copied().collect::<Vec<_>>(),
                        )) as ArrayRef,
                    ),
                ],
                300,
            );
            let mut spec_ = spec(EvaluateTask::Binary, dir.path(), partition_rows);
            spec_.prediction_columns = vec![("True_PROB".into(), "True_PROB".into())];
            let out = evaluate(dir.path(), &spec_);
            assert_eq!(out["rows"], json!(n));
            assert_eq!(out["labelled"], json!(m));
            let auroc = out["metrics"]["auroc"].as_f64().unwrap();
            assert!((auroc - brute_auroc(&scores[..m], &labels[..m])).abs() < 1e-12);
            assert!(out["metrics"]["ap"].as_f64().unwrap() > 0.0);
            let entities = actual_column::<arrow::datatypes::Int64Type>(dir.path(), "ENTITY");
            let emitted = actual_column::<arrow::datatypes::Float32Type>(dir.path(), "True_PROB");
            assert_eq!(entities.len(), m);
            for (entity, score) in entities.iter().zip(emitted) {
                assert_eq!(score, scores[*entity as usize]);
            }
            assert!(!dir.path().join("spill").exists());
        }
    }

    #[test]
    fn average_precision_and_auprc_match_step_and_trapezoid_definitions() {
        // Scores 0.9(+), 0.8(-), 0.8(+), 0.1(-): (recall, precision) at the
        // thresholds 0.9, 0.8, 0.1 is (0.5, 1), (1, 2/3), (1, 1/2).
        let mut pairs = vec![
            order_key(0.9) << 1 | 1,
            order_key(0.8) << 1,
            order_key(0.8) << 1 | 1,
            order_key(0.1) << 1,
        ];
        let [auroc, ap, auprc] = ranking_metrics(&mut pairs);
        assert!((auroc.unwrap() - 0.875).abs() < 1e-12);
        assert!((ap.unwrap() - (0.5 * 1.0 + 0.5 * 2.0 / 3.0)).abs() < 1e-12);
        let trapezoids = 0.5 * (1.0 + 1.0) / 2.0 + 0.5 * (1.0 + 2.0 / 3.0) / 2.0;
        assert!((auprc.unwrap() - trapezoids).abs() < 1e-12);
        assert_eq!(ranking_metrics(&mut [order_key(0.5) << 1 | 1]), [None; 3]);
    }

    #[test]
    fn regression_multiclass_and_ranking_metrics() {
        let dir = tempdir().unwrap();
        let mut spec_ = spec(EvaluateTask::Regression, dir.path(), 1_000_000);
        spec_.prediction_keys.truncate(1);
        spec_.label_keys.truncate(1);
        spec_.actual_columns = vec![
            ("ENTITY1".into(), "ENTITY".into()),
            ("TARGET1".into(), "TARGET".into()),
        ];
        spec_.constant_timestamps =
            vec![("ANCHOR_TIMESTAMP".into(), 7), ("EVALUATED_AT".into(), 9)];
        write(
            &dir.path().join("predictions.parquet"),
            vec![
                (
                    "ENTITY",
                    Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
                ),
                (
                    "TARGET_PRED",
                    Arc::new(Float64Array::from(vec![1.0, 3.0, 4.0])) as ArrayRef,
                ),
            ],
            2,
        );
        write(
            &dir.path().join("labels.parquet"),
            vec![
                (
                    "ENTITY1",
                    Arc::new(StringArray::from(vec!["c", "a", "b"])) as ArrayRef,
                ),
                (
                    "TARGET1",
                    Arc::new(Float64Array::from(vec![5.0, 1.0, 2.0])) as ArrayRef,
                ),
            ],
            2,
        );
        let out = evaluate(dir.path(), &spec_);
        assert_eq!(
            out["metrics"],
            json!({"mae": 2.0 / 3.0, "rmse": (2.0f64 / 3.0).sqrt()})
        );
        assert_eq!(
            actual_column::<arrow::datatypes::TimestampNanosecondType>(
                dir.path(),
                "ANCHOR_TIMESTAMP"
            ),
            vec![7, 7, 7]
        );
        assert_eq!(
            actual_column::<arrow::datatypes::TimestampNanosecondType>(dir.path(), "EVALUATED_AT"),
            vec![9, 9, 9]
        );

        // Two anchors of entity 1 (a hit and a miss at rank 1) and entity 2.
        spec_ = spec(EvaluateTask::Multiclass, dir.path(), 1_000_000);
        spec_.prediction_columns = vec![
            ("CLASS".into(), "CLASS".into()),
            ("SCORE".into(), "SCORE".into()),
        ];
        let entity = Int64Array::from(vec![1, 1, 1, 1, 2, 2]);
        let anchor = TimestampMicrosecondArray::from(vec![0, 0, 1, 1, 0, 0]);
        write(
            &dir.path().join("predictions.parquet"),
            vec![
                ("ENTITY", Arc::new(entity.clone()) as ArrayRef),
                ("ANCHOR_TIMESTAMP", Arc::new(anchor.clone()) as ArrayRef),
                (
                    "CLASS",
                    Arc::new(Int64Array::from(vec![10, 11, 12, 10, 11, 12])) as ArrayRef,
                ),
                (
                    "SCORE",
                    Arc::new(Float64Array::from(vec![0.9, 0.5, 0.7, 0.2, 0.8, 0.1])) as ArrayRef,
                ),
            ],
            3,
        );
        write(
            &dir.path().join("labels.parquet"),
            vec![
                (
                    "ENTITY1",
                    Arc::new(Int64Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "TIMESTAMP",
                    Arc::new(TimestampNanosecondArray::from(vec![0, 1000, 0])) as ArrayRef,
                ),
                (
                    "TARGET1",
                    Arc::new(Float64Array::from(vec![10.0, 10.0, 11.0])) as ArrayRef,
                ),
            ],
            3,
        );
        let out = evaluate(dir.path(), &spec_);
        assert_eq!(out["metrics"], json!({"acc": 2.0 / 3.0}));
        let top: Vec<(i64, f64)> =
            actual_column::<arrow::datatypes::Int64Type>(dir.path(), "CLASS")
                .into_iter()
                .zip(actual_column::<arrow::datatypes::Float64Type>(
                    dir.path(),
                    "SCORE",
                ))
                .collect();
        assert_eq!(top, vec![(10, 0.9), (12, 0.7), (11, 0.8)]);
        assert_eq!(
            (out["rows"].clone(), out["labelled"].clone()),
            (json!(3), json!(3))
        );

        spec_ = spec(EvaluateTask::Ranking, dir.path(), 1_000_000);
        spec_.ks = vec![1, 2, 3];
        let mut targets = ListBuilder::new(Int64Builder::new());
        for items in [vec![11, 13], vec![], vec![11]] {
            for item in items {
                targets.values().append_value(item);
            }
            targets.append(true);
        }
        write(
            &dir.path().join("labels.parquet"),
            vec![
                (
                    "ENTITY1",
                    Arc::new(Int64Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "TIMESTAMP",
                    Arc::new(TimestampNanosecondArray::from(vec![0, 1000, 0])) as ArrayRef,
                ),
                ("TARGET1", Arc::new(targets.finish()) as ArrayRef),
            ],
            3,
        );
        // (1, 0) ranks 10, 11 and hits 11 at rank 2 of two labels; (1, 1)
        // has no labels and is skipped; entity 2 hits 11 at rank 1. Only two
        // classes are returned, so there is no k = 3.
        let expected = json!({
            "map@1": 0.5, "precision@1": 0.5, "recall@1": 0.5, "f1@1": 0.5,
            "map@2": (0.5 / 2.0 + 1.0) / 2.0, "precision@2": 0.5, "recall@2": 0.75,
            "f1@2": (0.5 + 2.0 / 3.0) / 2.0,
        });
        assert_eq!(evaluate(dir.path(), &spec_)["metrics"], expected);
        write(
            &dir.path().join("labels.parquet"),
            vec![
                (
                    "ENTITY1",
                    Arc::new(Int64Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "TIMESTAMP",
                    Arc::new(TimestampNanosecondArray::from(vec![0, 1000, 0])) as ArrayRef,
                ),
                (
                    "TARGET1",
                    Arc::new(StringArray::from(vec!["11|13", "", "11"])) as ArrayRef,
                ),
            ],
            3,
        );
        assert_eq!(evaluate(dir.path(), &spec_)["metrics"], expected);
    }

    fn brute_ap_and_auprc(scores: &[f32], labels: &[bool]) -> (f64, f64) {
        let mut thresholds = scores.to_vec();
        thresholds.sort_by(|a, b| b.total_cmp(a));
        thresholds.dedup();
        let positives = labels.iter().filter(|l| **l).count() as f64;
        let (mut ap, mut auprc, mut previous) = (0.0, 0.0, (0.0, 1.0));
        for t in thresholds {
            let above = |positive: bool| {
                (0..scores.len())
                    .filter(|&i| scores[i] >= t && labels[i] == positive)
                    .count() as f64
            };
            let (tp, fp) = (above(true), above(false));
            let (recall, precision) = (tp / positives, tp / (tp + fp));
            ap += (recall - previous.0) * precision;
            auprc += (recall - previous.0) * (precision + previous.1) / 2.0;
            previous = (recall, precision);
        }
        (ap, auprc)
    }

    #[test]
    fn multilabel_join_matches_brute_force() {
        let (entities, classes) = (400usize, 6usize);
        let score = |e: usize, c: usize| ((e * 7919 + c * 104_729) % 97) as f32 / 96.0;
        let label = |e: usize, c: usize| (e * 31 + c * 17 + e / 5) % 7 < 2 + c % 3;
        let mut entity = Vec::new();
        let (mut class, mut scores, mut predicted) = (Vec::new(), Vec::new(), Vec::new());
        for e in 0..entities {
            for c in 0..classes {
                entity.push(e as i64);
                class.push(format!("c{c}"));
                scores.push(score(e, c));
                predicted.push(score(e, c) > 0.5);
            }
        }
        let dir = tempdir().unwrap();
        write(
            &dir.path().join("predictions.parquet"),
            vec![
                ("ENTITY", Arc::new(Int64Array::from(entity)) as ArrayRef),
                ("CLASS", Arc::new(StringArray::from(class)) as ArrayRef),
                (
                    "SCORE",
                    Arc::new(arrow::array::Float32Array::from(scores)) as ArrayRef,
                ),
                (
                    "PREDICTED",
                    Arc::new(BooleanArray::from(predicted)) as ArrayRef,
                ),
            ],
            500,
        );
        // Entity 0 has no label row; the class "c9" is never scored.
        let labelled: Vec<usize> = (1..entities).collect();
        let sets: Vec<String> = labelled
            .iter()
            .map(|&e| {
                let mut set: Vec<String> = (0..classes)
                    .filter(|&c| label(e, c))
                    .map(|c| format!("c{c}"))
                    .collect();
                set.push("c9".into());
                set.join("|")
            })
            .collect();
        write(
            &dir.path().join("labels.parquet"),
            vec![
                (
                    "ENTITY1",
                    Arc::new(Int64Array::from(
                        labelled.iter().map(|&e| e as i64).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ),
                ("TARGET1", Arc::new(StringArray::from(sets)) as ArrayRef),
            ],
            150,
        );
        let mut spec_ = spec(EvaluateTask::Multilabel, dir.path(), 900);
        spec_.prediction_keys.truncate(1);
        spec_.label_keys.truncate(1);
        spec_.actual_columns = vec![
            ("ENTITY1".into(), "ENTITY".into()),
            ("TARGET1".into(), "TARGET".into()),
        ];
        spec_.prediction_columns = vec![("CLASS".into(), "CLASS".into())];
        let out = evaluate(dir.path(), &spec_);
        assert_eq!(
            (out["rows"].clone(), out["labelled"].clone()),
            (json!(entities), json!(entities - 1))
        );

        let scored = |c: Option<usize>| -> (Vec<f32>, Vec<bool>, Vec<bool>) {
            let pairs = labelled.iter().flat_map(|&e| {
                (0..classes)
                    .filter(move |&x| c.is_none_or(|c| c == x))
                    .map(move |x| (e, x))
            });
            let s: Vec<f32> = pairs.clone().map(|(e, x)| score(e, x)).collect();
            let p = s.iter().map(|s| *s > 0.5).collect();
            (s, p, pairs.map(|(e, x)| label(e, x)).collect())
        };
        let (mut per_class, mut f1s) = (Vec::new(), Vec::new());
        for c in 0..classes {
            let (s, p, l) = scored(Some(c));
            let (ap, auprc) = brute_ap_and_auprc(&s, &l);
            per_class.push([brute_auroc(&s, &l), ap, auprc]);
            let count = |want: (bool, bool)| {
                p.iter().zip(&l).filter(|(p, l)| (**p, **l) == want).count() as f64
            };
            let (tp, fp, fn_) = (
                count((true, true)),
                count((true, false)),
                count((false, true)),
            );
            f1s.push(2.0 * tp / (2.0 * tp + fp + fn_));
        }
        let (s, p, l) = scored(None);
        let (ap, auprc) = brute_ap_and_auprc(&s, &l);
        let tp = p.iter().zip(&l).filter(|(p, l)| **p && **l).count() as f64;
        let predicted = p.iter().filter(|p| **p).count() as f64;
        let actual = l.iter().filter(|l| **l).count() as f64;
        let mean = |i: usize| per_class.iter().map(|m| m[i]).sum::<f64>() / classes as f64;
        let expected = [
            ("auroc_macro", mean(0)),
            ("ap_macro", mean(1)),
            ("auprc_macro", mean(2)),
            ("auroc_micro", brute_auroc(&s, &l)),
            ("ap_micro", ap),
            ("auprc_micro", auprc),
            ("f1_micro", 2.0 * tp / (predicted + actual)),
            ("precision_micro", tp / predicted),
            ("recall_micro", tp / actual),
            ("f1_macro", f1s.iter().sum::<f64>() / classes as f64),
        ];
        for (name, value) in expected {
            assert!(
                (out["metrics"][name].as_f64().unwrap() - value).abs() < 1e-12,
                "{name}"
            );
        }
        let rows = actuals(dir.path());
        let targets: Vec<bool> = rows
            .iter()
            .flat_map(|batch| column_bools(batch, "TARGET"))
            .collect();
        let classes_out: Vec<String> = rows
            .iter()
            .flat_map(|batch| {
                let array = column(batch, "CLASS").unwrap().as_string::<i32>().clone();
                array
                    .iter()
                    .map(|c| c.unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        let entities_out = actual_column::<arrow::datatypes::Int64Type>(dir.path(), "ENTITY");
        assert_eq!(targets.len(), (entities - 1) * classes);
        for ((e, c), target) in entities_out.iter().zip(&classes_out).zip(&targets) {
            assert_eq!(*target, label(*e as usize, c[1..].parse().unwrap()));
        }
    }

    fn column_bools(batch: &RecordBatch, name: &str) -> Vec<bool> {
        column(batch, name)
            .unwrap()
            .as_boolean()
            .iter()
            .map(Option::unwrap)
            .collect()
    }

    #[test]
    fn forecast_join_matches_brute_force_per_step() {
        let (entities, steps) = (300i64, 4usize);
        let day = 86_400_000_000_000i64;
        let prediction = |e: i64, h: usize| (e * 13 + h as i64 * 7) as f64 % 50.0;
        let actual = |e: i64, h: usize| ((e * 29 + h as i64 * 11) % 41) as f64;
        let (mut entity, mut anchor, mut time, mut value) = (vec![], vec![], vec![], vec![]);
        for e in 0..entities {
            // Steps written newest first; the join orders them by TIMESTAMP.
            for h in (0..steps).rev() {
                entity.push(e);
                anchor.push(e % 2 * 30 * day);
                time.push(e % 2 * 30 * day + h as i64 * 7 * day);
                value.push(prediction(e, h));
            }
        }
        let dir = tempdir().unwrap();
        write(
            &dir.path().join("predictions.parquet"),
            vec![
                ("ENTITY", Arc::new(Int64Array::from(entity)) as ArrayRef),
                (
                    "ANCHOR_TIMESTAMP",
                    Arc::new(TimestampNanosecondArray::from(anchor)) as ArrayRef,
                ),
                (
                    "TIMESTAMP",
                    Arc::new(TimestampMicrosecondArray::from(
                        time.iter().map(|t| t / 1000).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ),
                (
                    "TARGET_PRED",
                    Arc::new(Float64Array::from(value)) as ArrayRef,
                ),
            ],
            200,
        );
        // Entity 0 has a label row with three of the four steps.
        let mut futures = ListBuilder::new(arrow::array::Float64Builder::new());
        for e in 0..entities {
            for h in 0..if e == 0 { steps - 1 } else { steps } {
                futures.values().append_value(actual(e, h));
            }
            futures.append(true);
        }
        write(
            &dir.path().join("labels.parquet"),
            vec![
                (
                    "ENTITY1",
                    Arc::new(Int64Array::from((0..entities).collect::<Vec<_>>())) as ArrayRef,
                ),
                (
                    "TIMESTAMP",
                    Arc::new(TimestampNanosecondArray::from(
                        (0..entities).map(|e| e % 2 * 30 * day).collect::<Vec<_>>(),
                    )) as ArrayRef,
                ),
                ("future_labels", Arc::new(futures.finish()) as ArrayRef),
            ],
            70,
        );
        let mut spec_ = spec(EvaluateTask::Forecasting, dir.path(), 400);
        spec_.label_target = "future_labels".into();
        spec_.actual_columns[2].0 = "future_labels".into();
        spec_.prediction_columns = vec![
            ("TIMESTAMP".into(), "TIMESTAMP".into()),
            ("TARGET_PRED".into(), "TARGET_PRED".into()),
        ];
        let out = evaluate(dir.path(), &spec_);
        assert_eq!(
            (out["rows"].clone(), out["labelled"].clone()),
            (json!(entities), json!(entities))
        );

        let mut sums = vec![[0.0f64; 6]; steps];
        for e in 0..entities {
            for (h, sum) in sums
                .iter_mut()
                .enumerate()
                .take(if e == 0 { steps - 1 } else { steps })
            {
                let (p, a) = (prediction(e, h), actual(e, h));
                let values = [1.0, (p - a).abs(), (p - a) * (p - a), p - a, a, a * a];
                sum.iter_mut().zip(values).for_each(|(s, v)| *s += v);
            }
        }
        for (i, name) in STEP_NAMES.iter().enumerate() {
            let column: Vec<f64> = sums.iter().map(|s| s[i]).collect();
            assert_eq!(
                out["by_horizon"][format!("horizon_{name}")],
                json!(column),
                "{name}"
            );
        }
        let rows: f64 = sums.iter().map(|s| s[0]).sum();
        let mae = sums.iter().map(|s| s[1]).sum::<f64>() / rows;
        let rmse = (sums.iter().map(|s| s[2]).sum::<f64>() / rows).sqrt();
        assert!((out["metrics"]["mae"].as_f64().unwrap() - mae).abs() < 1e-12);
        assert!((out["metrics"]["rmse"].as_f64().unwrap() - rmse).abs() < 1e-12);

        let entities_out = actual_column::<arrow::datatypes::Int64Type>(dir.path(), "ENTITY");
        let times =
            actual_column::<arrow::datatypes::TimestampNanosecondType>(dir.path(), "TIMESTAMP");
        let targets = actual_column::<arrow::datatypes::Float64Type>(dir.path(), "TARGET");
        let predictions = actual_column::<arrow::datatypes::Float64Type>(dir.path(), "TARGET_PRED");
        assert_eq!(targets.len(), rows as usize);
        for i in 0..targets.len() {
            let e = entities_out[i];
            let h = ((times[i] - e % 2 * 30 * day) / (7 * day)) as usize;
            assert_eq!(
                (targets[i], predictions[i]),
                (actual(e, h), prediction(e, h))
            );
        }
    }
}
