// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Join batch-prediction outputs with generated labels and compute realized
//! metrics. Both sides are hash-partitioned on their key into Arrow IPC
//! spill files of about `partition_rows` rows each, and at most
//! `MAX_CONCURRENT_JOINS` partitions are joined at once. Metrics follow the
//! fit-time definitions: exact AUROC and AP as in torchmetrics, MAE/RMSE,
//! top-1 accuracy, and MAP@k / precision@k as in PyG.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use arrow::array::TimestampNanosecondArray;
use arrow::array::{Array, AsArray, BooleanArray, FixedSizeBinaryArray, UInt32Array};
use arrow::compute::{cast, filter, interleave, take_record_batch};
use arrow::datatypes::Float64Type;
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::row::{RowConverter, SortField};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};

use crate::bp_col_stats::row_group_batches;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EvaluateTask {
    Binary,
    Regression,
    Multiclass,
    Link,
}

#[derive(Clone, Debug, Deserialize)]
pub struct EvaluateSpec {
    pub task: EvaluateTask,
    pub prediction_keys: Vec<String>,
    pub label_keys: Vec<String>,
    pub label_target: String,
    #[serde(default)]
    pub actuals_dir: Option<PathBuf>,
    /// (label column, output column) pairs written to `actuals_dir`.
    #[serde(default)]
    pub actual_columns: Vec<(String, String)>,
    /// (prediction column, output column) pairs written next to them, from
    /// the matched prediction (the top-scored row of a grouped task).
    #[serde(default)]
    pub prediction_columns: Vec<(String, String)>,
    /// (output column, UTC nanoseconds) pairs written as constant timestamps.
    #[serde(default)]
    pub constant_timestamps: Vec<(String, i64)>,
    pub spill_dir: PathBuf,
    #[serde(default = "default_partition_rows")]
    pub partition_rows: usize,
}

fn default_partition_rows() -> usize {
    1_000_000
}

const MAX_CONCURRENT_JOINS: usize = 8;
const KEY: &str = "__key";

type Batches = Box<dyn Iterator<Item = Result<RecordBatch>> + Send>;

/// The type a pair of columns is compared as.
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
    let fields = arrays
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let rows = RowConverter::new(fields)?.convert_columns(&arrays)?;
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

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    batch
        .column_by_name(name)
        .with_context(|| format!("missing column '{name}'"))
}

fn floats(array: &ArrayRef) -> Result<Vec<Option<f64>>> {
    Ok(cast(array, &DataType::Float64)?
        .as_primitive::<Float64Type>()
        .iter()
        .collect())
}

fn keys(batch: &RecordBatch) -> Result<impl Iterator<Item = u128> + '_> {
    let keys = column(batch, KEY)?.as_fixed_size_binary();
    Ok((0..keys.len()).map(|row| u128::from_le_bytes(keys.value(row).try_into().unwrap())))
}

/// Sorted, distinct class hashes of each label.
fn label_classes(target: &ArrayRef, link: bool, class: &DataType) -> Result<Vec<Option<Vec<u64>>>> {
    if !link {
        let classes = classes(target, class)?;
        return Ok(classes.into_iter().map(|c| c.map(|c| vec![c])).collect());
    }
    let item = Field::new_list_field(class.clone(), true);
    let list = cast(target, &DataType::List(Arc::new(item)))?;
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

/// Read `keep` columns, hash `keys`, and route rows with a key to `sinks`.
fn scatter(
    paths: &[&Path],
    groups: &RowGroups,
    (keys, types): (&[String], &[DataType]),
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

#[derive(Default)]
struct Accumulator {
    rows: u64,
    labelled: u64,
    /// Binary: score order key shifted left, label in bit 0.
    pairs: Vec<u64>,
    count: u64,
    sum: f64,
    squares: f64,
    /// Link: MAP numerators by label count.
    map_by_labels: BTreeMap<usize, f64>,
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
        for (labels, value) in other.map_by_labels {
            *self.map_by_labels.entry(labels).or_default() += value;
        }
        self.k = self.k.max(other.k);
        self
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

/// A grouped prediction row: score, class and (batch, row).
type Ranked = (f64, Option<u64>, (usize, usize));

/// Score each label row against the predictions with its key; the first
/// label of a key counts.
fn join(
    predictions: Batches,
    labels: Batches,
    spec: &EvaluateSpec,
    (score, class): (&str, &DataType),
    actuals: Option<PathBuf>,
) -> Result<Accumulator> {
    let grouped = matches!(spec.task, EvaluateTask::Multiclass | EvaluateTask::Link);
    let mut first = HashMap::new();
    let mut groups: HashMap<u128, Vec<Ranked>> = HashMap::new();
    let mut kept: Vec<Vec<ArrayRef>> = Vec::new();
    for (index, batch) in predictions.enumerate() {
        let batch = batch?;
        kept.push(
            spec.prediction_columns
                .iter()
                .map(|(name, _)| column(&batch, name).cloned())
                .collect::<Result<_>>()?,
        );
        let scores = floats(column(&batch, score)?)?;
        if !grouped {
            for (row, (key, score)) in keys(&batch)?.zip(scores).enumerate() {
                first.entry(key).or_insert((score, (index, row)));
            }
            continue;
        }
        let classes = classes(column(&batch, "CLASS")?, class)?;
        for (row, ((key, score), class)) in keys(&batch)?.zip(scores).zip(classes).enumerate() {
            let score = score.unwrap_or(f64::NEG_INFINITY);
            groups
                .entry(key)
                .or_default()
                .push((score, class, (index, row)));
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
        let mut matched = vec![false; batch.num_rows()];
        let mut picked = Vec::new();
        if !grouped {
            for ((row, key), label) in keys(&batch)?.enumerate().zip(floats(target)?) {
                let Some(label) = label else { continue };
                let Some((score, at)) = first.remove(&key) else {
                    continue;
                };
                matched[row] = true;
                picked.push(at);
                match score {
                    Some(score) if spec.task == EvaluateTask::Binary => {
                        if score.is_finite() {
                            acc.pairs
                                .push(order_key(score) << 1 | u64::from(label != 0.0));
                        }
                    }
                    Some(prediction) => {
                        let error = prediction - label;
                        acc.count += 1;
                        acc.sum += error.abs();
                        acc.squares += error * error;
                    }
                    None => {}
                }
            }
        } else {
            let link = spec.task == EvaluateTask::Link;
            let labels = label_classes(target, link, class)?;
            for ((row, key), items) in keys(&batch)?.enumerate().zip(labels) {
                let Some(items) = items else { continue };
                let Some(mut ranked) = groups.remove(&key) else {
                    continue;
                };
                matched[row] = true;
                // Highest score first; ties keep their order.
                ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
                picked.push(ranked[0].2);
                let hit =
                    |class: Option<u64>| class.is_some_and(|c| items.binary_search(&c).is_ok());
                if !link {
                    acc.count += 1;
                    acc.sum += f64::from(u8::from(hit(ranked[0].1)));
                } else if !items.is_empty() {
                    let (mut hits, mut numerator) = (0usize, 0.0);
                    for (rank, (_, class, _)) in ranked.iter().enumerate() {
                        if hit(*class) {
                            hits += 1;
                            numerator += hits as f64 / (rank + 1) as f64;
                        }
                    }
                    acc.count += 1;
                    acc.sum += hits as f64;
                    *acc.map_by_labels.entry(items.len()).or_default() += numerator;
                }
            }
        }
        let matched = BooleanArray::from(matched);
        acc.labelled += matched.true_count() as u64;
        let Some(path) = actuals.as_ref().filter(|_| matched.true_count() > 0) else {
            continue;
        };
        let mut out = spec
            .actual_columns
            .iter()
            .map(|(name, out)| Ok((out.as_str(), filter(column(&batch, name)?, &matched)?, true)))
            .collect::<Result<Vec<_>>>()?;
        for (name, ns) in &spec.constant_timestamps {
            let values = vec![*ns; picked.len()];
            let array = TimestampNanosecondArray::from(values).with_timezone("UTC");
            out.push((name.as_str(), Arc::new(array), true));
        }
        for (column, (_, name)) in spec.prediction_columns.iter().enumerate() {
            let arrays: Vec<&dyn Array> = kept.iter().map(|batch| batch[column].as_ref()).collect();
            out.push((name.as_str(), interleave(&arrays, &picked)?, true));
        }
        let out = RecordBatch::try_from_iter_with_nullable(out)?;
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

/// Exact AUROC and average precision from (score, label) order keys.
fn ranking_metrics(mut pairs: Vec<u64>) -> (Option<f64>, Option<f64>) {
    pairs.par_sort_unstable();
    let positives = pairs.iter().filter(|pair| *pair & 1 == 1).count() as f64;
    let negatives = pairs.len() as f64 - positives;
    if positives == 0.0 || negatives == 0.0 {
        return (None, None);
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
    let (mut true_positive, mut false_positive, mut previous_recall, mut precision_sum) =
        (0.0, 0.0, 0.0, 0.0);
    for &(positive, negative) in groups.iter().rev() {
        true_positive += positive;
        false_positive += negative;
        let recall = true_positive / positives;
        precision_sum +=
            (recall - previous_recall) * true_positive / (true_positive + false_positive);
        previous_recall = recall;
    }
    (Some(area / (positives * negatives)), Some(precision_sum))
}

fn metrics(acc: Accumulator, task: EvaluateTask) -> JsonValue {
    let ratio = |num: f64| (acc.count > 0).then(|| num / acc.count as f64);
    match task {
        EvaluateTask::Binary => {
            let (auroc, ap) = ranking_metrics(acc.pairs);
            json!({"auroc": auroc, "ap": ap})
        }
        EvaluateTask::Regression => {
            json!({"mae": ratio(acc.sum), "rmse": ratio(acc.squares).map(f64::sqrt)})
        }
        EvaluateTask::Multiclass => json!({"acc": ratio(acc.sum)}),
        EvaluateTask::Link => {
            let k = acc.k.max(1);
            let map: f64 = acc
                .map_by_labels
                .iter()
                .map(|(labels, numerator)| numerator / (*labels).min(k) as f64)
                .sum();
            let precision = ratio(acc.sum / k as f64);
            json!({format!("map@{k}"): ratio(map), format!("precision@{k}"): precision})
        }
    }
}

/// Join `prediction_paths` with `label_paths` and compute the realized
/// metrics; returns `{"rows", "labelled", "metrics"}`.
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
    let score = match spec.task {
        EvaluateTask::Binary => "True_PROB",
        EvaluateTask::Regression => "TARGET_PRED",
        EvaluateTask::Multiclass | EvaluateTask::Link => "SCORE",
    };
    let mut prediction_columns = vec![score.to_string()];
    prediction_columns.extend(spec.prediction_columns.iter().map(|(name, _)| name.clone()));
    let mut class = DataType::Null;
    if matches!(spec.task, EvaluateTask::Multiclass | EvaluateTask::Link) {
        prediction_columns.push("CLASS".to_string());
    }
    prediction_columns.sort();
    prediction_columns.dedup();
    if matches!(spec.task, EvaluateTask::Multiclass | EvaluateTask::Link) {
        let mut target = label_schema
            .field_with_name(&spec.label_target)?
            .data_type();
        if spec.task == EvaluateTask::Link {
            let (DataType::List(item)
            | DataType::LargeList(item)
            | DataType::FixedSizeList(item, _)) = target
            else {
                bail!("link-prediction labels must be lists, got {target}");
            };
            target = item.data_type();
        }
        class = common_type(
            prediction_schema.field_with_name("CLASS")?.data_type(),
            target,
        );
    }
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
    let prediction_keys = (spec.prediction_keys.as_slice(), types.as_slice());
    scatter(
        prediction_paths,
        &prediction_groups,
        prediction_keys,
        &prediction_columns,
        &predictions,
    )?;
    let label_keys = (spec.label_keys.as_slice(), types.as_slice());
    scatter(
        label_paths,
        &label_groups,
        label_keys,
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
                    (score, &class),
                    actuals,
                )
            })
            .try_reduce(Accumulator::default, |a, b| Ok(a.merge(b)))
    })?;
    let _ = std::fs::remove_dir_all(&spec.spill_dir);
    let (rows, labelled) = (acc.rows, acc.labelled);
    Ok(json!({"rows": rows, "labelled": labelled, "metrics": metrics(acc, spec.task)}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BooleanArray, Float64Array, Int64Array, Int64Builder, ListBuilder, StringArray,
        TimestampMicrosecondArray, TimestampNanosecondArray,
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
    fn average_precision_matches_step_definition() {
        // Scores 0.9(+), 0.8(-), 0.8(+), 0.1(-): thresholds 0.9, 0.8, 0.1.
        let pairs = vec![
            order_key(0.9) << 1 | 1,
            order_key(0.8) << 1,
            order_key(0.8) << 1 | 1,
            order_key(0.1) << 1,
        ];
        let (auroc, ap) = ranking_metrics(pairs);
        assert!((auroc.unwrap() - 0.875).abs() < 1e-12);
        assert!((ap.unwrap() - (0.5 * 1.0 + 0.5 * 2.0 / 3.0)).abs() < 1e-12);
        assert_eq!(ranking_metrics(vec![order_key(0.5) << 1 | 1]), (None, None));
    }

    #[test]
    fn regression_multiclass_and_link_metrics() {
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

        spec_ = spec(EvaluateTask::Link, dir.path(), 1_000_000);
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
        // has no labels and is skipped; entity 2 hits 11 at rank 1.
        let out = evaluate(dir.path(), &spec_);
        assert_eq!(
            out["metrics"],
            json!({"map@2": (0.5 / 2.0 + 1.0) / 2.0, "precision@2": 0.5})
        );
    }
}
