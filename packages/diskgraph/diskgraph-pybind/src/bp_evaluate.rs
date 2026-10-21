// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Join batch-prediction outputs with generated labels and compute realized
//! metrics, streaming both sides from Parquet.
//!
//! Both sides are hash-partitioned on their join key; a partition holds about
//! `partition_rows` rows of either side and at most `MAX_CONCURRENT_JOINS`
//! partitions are joined at once, so memory follows the partition size, not
//! the output size (binary metrics also keep 8 bytes per labelled row for the
//! exact ranking). With more than one partition both sides are spilled to
//! Arrow IPC files first. Metrics follow the
//! definitions used for fit-time `test_*` metrics: AUROC and average
//! precision as in torchmetrics (exact, ties counted half), MAE/RMSE,
//! top-1 accuracy, and MAP@k / precision@k as in PyG's link-prediction
//! metrics over examples with at least one label.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use arrow::array::{Array, AsArray, ListArray, UInt32Array};
use arrow::compute::{cast, take};
use arrow::datatypes::{Float64Type, Int64Type, UInt64Type};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow_array::{ArrayRef, Float64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowWriter, ProjectionMask};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EvaluateTask {
    /// `True_PROB` against a 0/1 label: `auroc`, `ap`.
    Binary,
    /// `TARGET_PRED` against a number: `mae`, `rmse`.
    Regression,
    /// The top-`SCORE` `CLASS` of each prediction against one label: `acc`.
    Multiclass,
    /// Ranked `CLASS` rows against a list of labels: `map@k`, `precision@k`.
    Link,
}

/// What to join and score; columns are matched by name.
#[derive(Clone, Debug, Deserialize)]
pub struct EvaluateSpec {
    pub task: EvaluateTask,
    /// Prediction key columns, paired by position with `label_keys`.
    pub prediction_keys: Vec<String>,
    pub label_keys: Vec<String>,
    pub label_target: String,
    /// Directory for the matched label rows (`part-*.parquet`), renamed by
    /// `actual_columns` (label name, output name); none when unset.
    #[serde(default)]
    pub actuals_dir: Option<PathBuf>,
    #[serde(default)]
    pub actual_columns: Vec<(String, String)>,
    pub spill_dir: PathBuf,
    #[serde(default = "default_partition_rows")]
    pub partition_rows: usize,
}

fn default_partition_rows() -> usize {
    1_000_000
}

const MAX_CONCURRENT_JOINS: usize = 8;

/// Keys are already uniform 128-bit hashes; fold them instead of rehashing.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*byte);
        }
    }

    fn write_u128(&mut self, value: u128) {
        self.0 = value as u64 ^ (value >> 64) as u64;
    }
}

type KeyMap<V> = HashMap<u128, V, std::hash::BuildHasherDefault<KeyHasher>>;

const KEY_HI: &str = "__key_hi";
const KEY_LO: &str = "__key_lo";
const SCORE: &str = "__score";
const CLASS: &str = "__class";

#[derive(Clone, Copy, PartialEq)]
enum KeyKind {
    Integer,
    Time,
    Text,
}

fn key_kind(prediction: &DataType, label: &DataType) -> KeyKind {
    let temporal = |t: &DataType| {
        matches!(
            t,
            DataType::Timestamp(..) | DataType::Date32 | DataType::Date64
        )
    };
    if temporal(prediction) || temporal(label) {
        KeyKind::Time
    } else if prediction.is_integer() && label.is_integer() {
        KeyKind::Integer
    } else {
        KeyKind::Text
    }
}

fn hasher(seed: u64) -> std::collections::hash_map::DefaultHasher {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    hasher
}

/// 128-bit hashes of each row's key; `None` when a key value is null.
fn key_hashes(
    batch: &RecordBatch,
    names: &[String],
    kinds: &[KeyKind],
) -> Result<Vec<Option<u128>>> {
    let columns: Vec<ArrayRef> = names
        .iter()
        .zip(kinds)
        .map(|(name, kind)| {
            let array = batch.column(batch.schema().index_of(name)?);
            Ok(match kind {
                KeyKind::Integer => cast(array, &DataType::Int64)?,
                KeyKind::Time => cast(
                    &cast(
                        array,
                        &DataType::Timestamp(TimeUnit::Nanosecond, timezone(array)),
                    )?,
                    &DataType::Int64,
                )?,
                KeyKind::Text => cast(array, &DataType::Utf8)?,
            })
        })
        .collect::<Result<_>>()?;
    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let (mut hi, mut lo) = (hasher(1), hasher(2));
        let mut present = true;
        for (array, kind) in columns.iter().zip(kinds) {
            if array.is_null(row) {
                present = false;
                break;
            }
            if *kind == KeyKind::Text {
                let value = array.as_string::<i32>().value(row);
                value.hash(&mut hi);
                value.hash(&mut lo);
            } else {
                let value = array.as_primitive::<Int64Type>().value(row);
                value.hash(&mut hi);
                value.hash(&mut lo);
            }
        }
        out.push(present.then(|| (u128::from(hi.finish()) << 64) | u128::from(lo.finish())));
    }
    Ok(out)
}

fn timezone(array: &ArrayRef) -> Option<Arc<str>> {
    match array.data_type() {
        DataType::Timestamp(_, tz) => tz.clone(),
        _ => None,
    }
}

/// Identity of a class or label value: integers, integral floats and
/// booleans by number, other floats by value, everything else as text.
fn value_hashes(array: &ArrayRef) -> Result<Vec<Option<u64>>> {
    let tagged = |tag: u8, write: &dyn Fn(&mut std::collections::hash_map::DefaultHasher)| {
        let mut hasher = hasher(3);
        tag.hash(&mut hasher);
        write(&mut hasher);
        hasher.finish()
    };
    let number = |value: f64| {
        if value.fract() == 0.0 && value.abs() < 9.0e15 {
            tagged(0, &|h| (value as i64).hash(h))
        } else {
            tagged(1, &|h| value.to_bits().hash(h))
        }
    };
    let data_type = array.data_type();
    Ok(
        if data_type.is_integer() || *data_type == DataType::Boolean {
            let values = cast(array, &DataType::Int64)?;
            values
                .as_primitive::<Int64Type>()
                .iter()
                .map(|value| value.map(|value| tagged(0, &|h| value.hash(h))))
                .collect()
        } else if data_type.is_floating() {
            let values = cast(array, &DataType::Float64)?;
            values
                .as_primitive::<Float64Type>()
                .iter()
                .map(|value| value.map(number))
                .collect()
        } else {
            let values = cast(array, &DataType::Utf8)?;
            values
                .as_string::<i32>()
                .iter()
                .map(|value| value.map(|value| tagged(2, &|h| value.hash(h))))
                .collect()
        },
    )
}

fn float_values(array: &ArrayRef) -> Result<Vec<Option<f64>>> {
    Ok(cast(array, &DataType::Float64)?
        .as_primitive::<Float64Type>()
        .iter()
        .collect())
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    Ok(batch.column(
        batch
            .schema()
            .index_of(name)
            .with_context(|| format!("missing column '{name}'"))?,
    ))
}

fn key_columns(keys: &[Option<u128>]) -> (UInt64Array, UInt64Array) {
    let hi: UInt64Array = keys
        .iter()
        .map(|key| key.map(|key| (key >> 64) as u64))
        .collect();
    let lo: UInt64Array = keys.iter().map(|key| key.map(|key| key as u64)).collect();
    (hi, lo)
}

fn read_key(batch: &RecordBatch, row: usize) -> Option<u128> {
    let hi = batch.column(0).as_primitive::<UInt64Type>();
    let lo = batch.column(1).as_primitive::<UInt64Type>();
    (!hi.is_null(row)).then(|| (u128::from(hi.value(row)) << 64) | u128::from(lo.value(row)))
}

/// Prediction rows reduced to key, score and class identity.
fn canonical_predictions(
    batch: &RecordBatch,
    spec: &EvaluateSpec,
    kinds: &[KeyKind],
) -> Result<RecordBatch> {
    let keys = key_hashes(batch, &spec.prediction_keys, kinds)?;
    let (hi, lo) = key_columns(&keys);
    let score_name = match spec.task {
        EvaluateTask::Binary => "True_PROB",
        EvaluateTask::Regression => "TARGET_PRED",
        EvaluateTask::Multiclass | EvaluateTask::Link => "SCORE",
    };
    let score = Float64Array::from(float_values(column(batch, score_name)?)?);
    let class: UInt64Array = match spec.task {
        EvaluateTask::Multiclass | EvaluateTask::Link => {
            value_hashes(column(batch, "CLASS")?)?.into_iter().collect()
        }
        _ => UInt64Array::new_null(batch.num_rows()),
    };
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new(KEY_HI, DataType::UInt64, true),
            Field::new(KEY_LO, DataType::UInt64, true),
            Field::new(SCORE, DataType::Float64, true),
            Field::new(CLASS, DataType::UInt64, true),
        ])),
        vec![Arc::new(hi), Arc::new(lo), Arc::new(score), Arc::new(class)],
    )?)
}

/// Label rows with their key hash in front of the original columns.
fn canonical_labels(
    batch: &RecordBatch,
    spec: &EvaluateSpec,
    kinds: &[KeyKind],
) -> Result<RecordBatch> {
    let keys = key_hashes(batch, &spec.label_keys, kinds)?;
    let (hi, lo) = key_columns(&keys);
    let mut fields = vec![
        Arc::new(Field::new(KEY_HI, DataType::UInt64, true)),
        Arc::new(Field::new(KEY_LO, DataType::UInt64, true)),
    ];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(hi), Arc::new(lo)];
    for (field, array) in batch.schema().fields().iter().zip(batch.columns()) {
        fields.push(Arc::new(field.as_ref().clone().with_nullable(true)));
        columns.push(array.clone());
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

fn partition_of(key: u128, partitions: usize) -> usize {
    ((key >> 64) as u64 % partitions as u64) as usize
}

enum Sink {
    Memory(Mutex<Vec<RecordBatch>>),
    Spill(
        Mutex<Option<Box<FileWriter<BufWriter<File>>>>>,
        PathBuf,
        SchemaRef,
    ),
}

impl Sink {
    fn push(&self, batch: RecordBatch) -> Result<()> {
        match self {
            Sink::Memory(batches) => batches.lock().unwrap().push(batch),
            Sink::Spill(writer, path, schema) => {
                let mut writer = writer.lock().unwrap();
                if writer.is_none() {
                    *writer = Some(Box::new(FileWriter::try_new(
                        BufWriter::new(File::create(path)?),
                        schema,
                    )?));
                }
                writer.as_mut().unwrap().write(&batch)?;
            }
        }
        Ok(())
    }

    fn into_batches(self) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>> + Send>> {
        Ok(match self {
            Sink::Memory(batches) => Box::new(batches.into_inner().unwrap().into_iter().map(Ok)),
            Sink::Spill(writer, path, _) => match writer.into_inner().unwrap() {
                None => Box::new(std::iter::empty()),
                Some(mut writer) => {
                    writer.finish()?;
                    let reader = FileReader::try_new(File::open(&path)?, None)?;
                    Box::new(reader.map(|batch| batch.map_err(Into::into)))
                }
            },
        })
    }
}

type RowGroups = Vec<(usize, usize)>;

fn work_items(paths: &[&Path]) -> Result<(RowGroups, usize, SchemaRef)> {
    let mut work = Vec::new();
    let mut rows = 0usize;
    let mut schema = None;
    for (idx, path) in paths.iter().enumerate() {
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
        schema.get_or_insert_with(|| builder.schema().clone());
        rows += builder.metadata().file_metadata().num_rows() as usize;
        work.extend((0..builder.metadata().num_row_groups()).map(|rg| (idx, rg)));
    }
    let schema = schema.unwrap_or_else(|| Arc::new(Schema::empty()));
    Ok((work, rows, schema))
}

/// Read every row group in parallel, canonicalize, and route rows to sinks.
fn scatter(
    paths: &[&Path],
    work: &[(usize, usize)],
    columns: &[String],
    sinks: &[Sink],
    canonical: &(dyn Fn(&RecordBatch) -> Result<RecordBatch> + Sync),
) -> Result<()> {
    work.par_iter().try_for_each(|&(idx, rg)| -> Result<()> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(paths[idx])?)?;
        let indices = columns
            .iter()
            .map(|name| {
                builder
                    .schema()
                    .index_of(name)
                    .with_context(|| format!("missing column '{name}'"))
            })
            .collect::<Result<Vec<_>>>()?;
        let projection = ProjectionMask::roots(builder.parquet_schema(), indices);
        for batch in builder
            .with_row_groups(vec![rg])
            .with_projection(projection)
            .build()?
        {
            let batch = batch?;
            let order = columns
                .iter()
                .map(|name| batch.schema().index_of(name))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let batch = canonical(&batch.project(&order)?)?;
            if sinks.len() == 1 {
                sinks[0].push(batch)?;
                continue;
            }
            let mut rows: Vec<Vec<u32>> = vec![Vec::new(); sinks.len()];
            for row in 0..batch.num_rows() {
                if let Some(key) = read_key(&batch, row) {
                    rows[partition_of(key, sinks.len())].push(row as u32);
                }
            }
            for (sink, rows) in sinks.iter().zip(rows) {
                if !rows.is_empty() {
                    let indices = UInt32Array::from(rows);
                    let columns = batch
                        .columns()
                        .iter()
                        .map(|array| take(array, &indices, None))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    sink.push(RecordBatch::try_new(batch.schema(), columns)?)?;
                }
            }
        }
        Ok(())
    })
}

enum Label {
    Positive(bool),
    Number(f64),
    Class(u64),
    Items(Vec<u64>),
}

#[derive(Default)]
struct Accumulator {
    rows: u64,
    labelled: u64,
    /// Binary: float32 order key of the score shifted left, label in bit 0.
    pairs: Vec<u64>,
    abs_error: f64,
    squared_error: f64,
    count: u64,
    correct: u64,
    /// Link: examples with labels, hits, and MAP numerators by label count.
    examples: u64,
    hits: f64,
    map_by_labels: BTreeMap<usize, f64>,
    k: usize,
}

impl Accumulator {
    fn merge(mut self, mut other: Self) -> Self {
        self.rows += other.rows;
        self.labelled += other.labelled;
        self.pairs.append(&mut other.pairs);
        self.abs_error += other.abs_error;
        self.squared_error += other.squared_error;
        self.count += other.count;
        self.correct += other.correct;
        self.examples += other.examples;
        self.hits += other.hits;
        for (labels, value) in other.map_by_labels {
            *self.map_by_labels.entry(labels).or_default() += value;
        }
        self.k = self.k.max(other.k);
        self
    }
}

fn order_key(score: f64) -> u64 {
    let bits = (score as f32).to_bits();
    let ordered = if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    };
    u64::from(ordered)
}

fn read_label(
    batch: &RecordBatch,
    target: &ArrayRef,
    task: EvaluateTask,
    row: usize,
    hashes: &LabelValues,
) -> Option<Label> {
    let _ = batch;
    if target.is_null(row) {
        return None;
    }
    Some(match (task, hashes) {
        (EvaluateTask::Binary, LabelValues::Numbers(values)) => {
            Label::Positive(values[row]? != 0.0)
        }
        (EvaluateTask::Regression, LabelValues::Numbers(values)) => Label::Number(values[row]?),
        (EvaluateTask::Multiclass, LabelValues::Classes(values)) => Label::Class(values[row]?),
        (EvaluateTask::Link, LabelValues::Lists(list, values)) => {
            let offsets = list.value_offsets();
            let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
            let mut items: Vec<u64> = values[start..end].iter().flatten().copied().collect();
            items.sort_unstable();
            items.dedup();
            Label::Items(items)
        }
        _ => return None,
    })
}

enum LabelValues {
    Numbers(Vec<Option<f64>>),
    Classes(Vec<Option<u64>>),
    Lists(ListArray, Vec<Option<u64>>),
}

fn label_values(target: &ArrayRef, task: EvaluateTask) -> Result<LabelValues> {
    Ok(match task {
        EvaluateTask::Binary | EvaluateTask::Regression => {
            LabelValues::Numbers(float_values(target)?)
        }
        EvaluateTask::Multiclass => LabelValues::Classes(value_hashes(target)?),
        EvaluateTask::Link => {
            let list = cast(
                target,
                &DataType::List(Arc::new(Field::new_list_field(
                    target_item_type(target)?,
                    true,
                ))),
            )?;
            let list = list.as_list::<i32>().clone();
            let values = value_hashes(list.values())?;
            LabelValues::Lists(list, values)
        }
    })
}

fn target_item_type(target: &ArrayRef) -> Result<DataType> {
    match target.data_type() {
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            Ok(field.data_type().clone())
        }
        other => bail!("link-prediction labels must be lists, got {other}"),
    }
}

fn join_partition(
    labels: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    predictions: Box<dyn Iterator<Item = Result<RecordBatch>> + Send>,
    spec: &EvaluateSpec,
    actuals: Option<PathBuf>,
) -> Result<Accumulator> {
    let mut label_batches = Vec::new();
    let mut by_key: KeyMap<(Label, u32, u32, bool)> = KeyMap::default();
    for batch in labels {
        let batch = batch?;
        let target = column(&batch, &spec.label_target)?.clone();
        let values = label_values(&target, spec.task)?;
        let batch_idx = label_batches.len() as u32;
        for row in 0..batch.num_rows() {
            if let (Some(key), Some(label)) = (
                read_key(&batch, row),
                read_label(&batch, &target, spec.task, row, &values),
            ) {
                by_key.insert(key, (label, batch_idx, row as u32, false));
            }
        }
        label_batches.push(batch);
    }

    let mut acc = Accumulator::default();
    let mut groups: KeyMap<Vec<(f64, Option<u64>)>> = KeyMap::default();
    let mut seen: KeyMap<()> = KeyMap::default();
    for batch in predictions {
        let batch = batch?;
        let score = batch.column(2).as_primitive::<Float64Type>();
        let class = batch.column(3).as_primitive::<UInt64Type>();
        for row in 0..batch.num_rows() {
            let Some(key) = read_key(&batch, row) else {
                continue;
            };
            let value = (!score.is_null(row)).then(|| score.value(row));
            match spec.task {
                EvaluateTask::Binary | EvaluateTask::Regression => {
                    if seen.insert(key, ()).is_some() {
                        continue;
                    }
                    acc.rows += 1;
                    let Some(entry) = by_key.get_mut(&key) else {
                        continue;
                    };
                    entry.3 = true;
                    acc.labelled += 1;
                    match (&entry.0, value) {
                        (Label::Positive(positive), Some(score)) if score.is_finite() => {
                            acc.pairs.push(order_key(score) << 1 | u64::from(*positive));
                        }
                        (Label::Number(target), Some(prediction)) => {
                            let error = prediction - target;
                            acc.abs_error += error.abs();
                            acc.squared_error += error * error;
                            acc.count += 1;
                        }
                        _ => {}
                    }
                }
                EvaluateTask::Multiclass | EvaluateTask::Link => {
                    groups.entry(key).or_default().push((
                        value.unwrap_or(f64::NEG_INFINITY),
                        (!class.is_null(row)).then(|| class.value(row)),
                    ));
                }
            }
        }
    }
    for (key, mut rows) in groups {
        acc.rows += 1;
        acc.k = acc.k.max(rows.len());
        let Some(entry) = by_key.get_mut(&key) else {
            continue;
        };
        entry.3 = true;
        acc.labelled += 1;
        // Highest score first; equal scores keep their output order.
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        match &entry.0 {
            Label::Class(target) => {
                acc.count += 1;
                acc.correct += u64::from(rows[0].1 == Some(*target));
            }
            Label::Items(items) if !items.is_empty() => {
                acc.examples += 1;
                let (mut hits, mut numerator) = (0usize, 0.0);
                for (rank, (_, class)) in rows.iter().enumerate() {
                    if class.is_some_and(|class| items.binary_search(&class).is_ok()) {
                        hits += 1;
                        numerator += hits as f64 / (rank + 1) as f64;
                    }
                }
                acc.hits += hits as f64;
                *acc.map_by_labels.entry(items.len()).or_default() += numerator;
            }
            _ => {}
        }
    }

    if let Some(path) = actuals {
        write_actuals(&path, &label_batches, &by_key, spec)?;
    }
    Ok(acc)
}

fn write_actuals(
    path: &Path,
    batches: &[RecordBatch],
    by_key: &KeyMap<(Label, u32, u32, bool)>,
    spec: &EvaluateSpec,
) -> Result<()> {
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); batches.len()];
    for (_, batch, row, matched) in by_key.values() {
        if *matched {
            rows[*batch as usize].push(*row);
        }
    }
    let mut writer: Option<ArrowWriter<File>> = None;
    for (batch, mut rows) in batches.iter().zip(rows) {
        if rows.is_empty() {
            continue;
        }
        rows.sort_unstable();
        let indices = UInt32Array::from(rows);
        let mut fields = Vec::new();
        let mut columns = Vec::new();
        for (name, output) in &spec.actual_columns {
            let array = column(batch, name)?;
            fields.push(Field::new(output, array.data_type().clone(), true));
            columns.push(take(array, &indices, None)?);
        }
        let out = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
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
    Ok(())
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

fn metrics(acc: Accumulator, task: EvaluateTask) -> BTreeMap<String, Option<f64>> {
    let ratio = |num: f64, den: u64| (den > 0).then(|| num / den as f64);
    let mut out = BTreeMap::new();
    match task {
        EvaluateTask::Binary => {
            let (auroc, ap) = ranking_metrics(acc.pairs);
            out.insert("auroc".to_string(), auroc);
            out.insert("ap".to_string(), ap);
        }
        EvaluateTask::Regression => {
            out.insert("mae".to_string(), ratio(acc.abs_error, acc.count));
            out.insert(
                "rmse".to_string(),
                ratio(acc.squared_error, acc.count).map(f64::sqrt),
            );
        }
        EvaluateTask::Multiclass => {
            out.insert("acc".to_string(), ratio(acc.correct as f64, acc.count));
        }
        EvaluateTask::Link => {
            let k = acc.k.max(1);
            let map: f64 = acc
                .map_by_labels
                .iter()
                .map(|(labels, numerator)| numerator / (*labels).min(k) as f64)
                .sum();
            out.insert(format!("map@{k}"), ratio(map, acc.examples));
            out.insert(
                format!("precision@{k}"),
                ratio(acc.hits / k as f64, acc.examples),
            );
        }
    }
    out
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
    let (prediction_work, prediction_rows, prediction_schema) = work_items(prediction_paths)?;
    let (label_work, label_rows, label_schema) = work_items(label_paths)?;
    let kinds: Vec<KeyKind> = spec
        .prediction_keys
        .iter()
        .zip(&spec.label_keys)
        .map(|(prediction, label)| {
            let field = |schema: &Schema, name: &str| {
                schema
                    .field_with_name(name)
                    .map(|f| f.data_type().clone())
                    .with_context(|| format!("missing column '{name}'"))
            };
            Ok(key_kind(
                &field(&prediction_schema, prediction)?,
                &field(&label_schema, label)?,
            ))
        })
        .collect::<Result<_>>()?;
    let partitions = prediction_rows
        .max(label_rows)
        .div_ceil(spec.partition_rows.max(1))
        .max(1);
    if partitions > 1 {
        std::fs::create_dir_all(&spec.spill_dir)?;
    }
    if let Some(dir) = &spec.actuals_dir {
        std::fs::create_dir_all(dir)?;
    }

    let mut prediction_columns = spec.prediction_keys.clone();
    prediction_columns.push(
        match spec.task {
            EvaluateTask::Binary => "True_PROB",
            EvaluateTask::Regression => "TARGET_PRED",
            _ => "SCORE",
        }
        .to_string(),
    );
    if matches!(spec.task, EvaluateTask::Multiclass | EvaluateTask::Link) {
        prediction_columns.push("CLASS".to_string());
    }
    let mut label_columns = spec.label_keys.clone();
    label_columns.push(spec.label_target.clone());
    label_columns.extend(spec.actual_columns.iter().map(|(name, _)| name.clone()));
    label_columns.sort();
    label_columns.dedup();

    let sinks = |side: &str, schema: SchemaRef| -> Vec<Sink> {
        (0..partitions)
            .map(|p| {
                if partitions == 1 {
                    Sink::Memory(Mutex::new(Vec::new()))
                } else {
                    Sink::Spill(
                        Mutex::new(None),
                        spec.spill_dir.join(format!("{side}-{p:05}.arrow")),
                        schema.clone(),
                    )
                }
            })
            .collect()
    };
    let canonical_prediction_schema = canonical_predictions(
        &RecordBatch::new_empty(project(&prediction_schema, &prediction_columns)?),
        spec,
        &kinds,
    )?
    .schema();
    let prediction_sinks = sinks("predictions", canonical_prediction_schema);
    scatter(
        prediction_paths,
        &prediction_work,
        &prediction_columns,
        &prediction_sinks,
        &|batch| canonical_predictions(batch, spec, &kinds),
    )?;
    let canonical_label_schema = canonical_labels(
        &RecordBatch::new_empty(project(&label_schema, &label_columns)?),
        spec,
        &kinds,
    )?
    .schema();
    let label_sinks = sinks("labels", canonical_label_schema);
    scatter(
        label_paths,
        &label_work,
        &label_columns,
        &label_sinks,
        &|batch| canonical_labels(batch, spec, &kinds),
    )?;

    let jobs: Vec<(usize, Sink, Sink)> = prediction_sinks
        .into_iter()
        .zip(label_sinks)
        .enumerate()
        .map(|(p, (prediction, label))| (p, prediction, label))
        .collect();
    let mut acc = Accumulator::default();
    let mut jobs = jobs.into_iter();
    loop {
        let wave: Vec<_> = jobs.by_ref().take(MAX_CONCURRENT_JOINS).collect();
        if wave.is_empty() {
            break;
        }
        let partial = wave
            .into_par_iter()
            .map(|(p, prediction, label)| {
                let actuals = spec
                    .actuals_dir
                    .as_ref()
                    .map(|dir| dir.join(format!("part-{p:05}.parquet")));
                join_partition(
                    label.into_batches()?,
                    prediction.into_batches()?,
                    spec,
                    actuals,
                )
            })
            .try_reduce(Accumulator::default, |a, b| Ok(a.merge(b)))?;
        acc = acc.merge(partial);
    }
    if partitions > 1 {
        let _ = std::fs::remove_dir_all(&spec.spill_dir);
    }
    let (rows, labelled) = (acc.rows, acc.labelled);
    Ok(json!({"rows": rows, "labelled": labelled, "metrics": metrics(acc, spec.task)}))
}

fn project(schema: &Schema, names: &[String]) -> Result<SchemaRef> {
    Ok(Arc::new(Schema::new(
        names
            .iter()
            .map(|name| {
                schema
                    .field_with_name(name)
                    .cloned()
                    .with_context(|| format!("missing column '{name}'"))
            })
            .collect::<Result<Vec<_>>>()?,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BooleanArray, Int64Array, Int64Builder, ListBuilder, StringArray,
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
            spill_dir: dir.join("spill"),
            partition_rows,
        }
    }

    fn evaluate(dir: &Path, spec: &EvaluateSpec) -> JsonValue {
        let predictions = dir.join("predictions.parquet");
        let labels = dir.join("labels.parquet");
        evaluate_prediction_parquet(&[predictions.as_path()], &[labels.as_path()], spec).unwrap()
    }

    fn actual_rows(dir: &Path) -> usize {
        std::fs::read_dir(dir.join("actuals"))
            .unwrap()
            .map(|entry| {
                let file = File::open(entry.unwrap().path()).unwrap();
                ParquetRecordBatchReaderBuilder::try_new(file)
                    .unwrap()
                    .metadata()
                    .file_metadata()
                    .num_rows() as usize
            })
            .sum()
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
            let out = evaluate(
                dir.path(),
                &spec(EvaluateTask::Binary, dir.path(), partition_rows),
            );
            assert_eq!(out["rows"], json!(n));
            assert_eq!(out["labelled"], json!(m));
            let auroc = out["metrics"]["auroc"].as_f64().unwrap();
            assert!((auroc - brute_auroc(&scores[..m], &labels[..m])).abs() < 1e-12);
            assert!(out["metrics"]["ap"].as_f64().unwrap() > 0.0);
            assert_eq!(actual_rows(dir.path()), m);
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

        // Two anchors of entity 1 (a hit and a miss at rank 1) and entity 2.
        spec_ = spec(EvaluateTask::Multiclass, dir.path(), 1_000_000);
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
