use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Date32Builder, Float32Builder, Float64Builder, Int16Builder,
    Int32Builder, Int64Builder, Int8Builder, ListBuilder, StringBuilder,
    TimestampMicrosecondBuilder, TimestampMillisecondBuilder, TimestampNanosecondBuilder,
    TimestampSecondBuilder, UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::config::{
    ColumnConfig, ColumnType, DatasetConfig, FkDistribution, GeneratorSpec, KeyEncoding,
    TableConfig, TimestampUnit,
};
use crate::output::OutputTarget;
use crate::progress::ProgressState;
use diskgraph_fs::CloudTransferStatsHandle;

pub fn run(config: DatasetConfig, output: OutputTarget, workers: usize) -> Result<()> {
    let plan = build_plan(&config)?;
    let table_names = plan
        .tables
        .iter()
        .map(|table| table.name.clone())
        .collect::<Vec<_>>();
    output.prepare(&table_names)?;

    let total_files = plan.tasks.len() as u64;
    let progress = Arc::new(Mutex::new(ProgressState::new()));
    let completed = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicBool::new(false));
    let upload_stats = output.stats_handle();
    let progress_handle = spawn_progress(
        Arc::clone(&progress),
        Arc::clone(&completed),
        Arc::clone(&done),
        total_files,
        upload_stats,
    );

    let tasks = Arc::new(Mutex::new(VecDeque::from(plan.tasks.clone())));
    let plan = Arc::new(plan);
    let mut handles = Vec::new();
    for _ in 0..workers {
        let plan = Arc::clone(&plan);
        let tasks = Arc::clone(&tasks);
        let completed = Arc::clone(&completed);
        let output = output.clone();
        let handle = thread::spawn(move || worker_loop(plan, tasks, completed, output));
        handles.push(handle);
    }

    let mut errors = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => errors.push(err),
            Err(_) => errors.push(anyhow!("worker thread panicked")),
        }
    }

    done.store(true, Ordering::Relaxed);
    if let Err(err) = progress_handle.join() {
        eprintln!("progress thread error: {err:?}");
    }

    if !errors.is_empty() {
        for err in errors {
            eprintln!("generation error: {err:#}");
        }
        bail!("generation failed");
    }

    if let Ok(progress) = progress.lock() {
        progress.finish("Done");
    }

    Ok(())
}

fn spawn_progress(
    progress: Arc<Mutex<ProgressState>>,
    completed: Arc<AtomicU64>,
    done: Arc<AtomicBool>,
    total_files: u64,
    upload_stats: Option<CloudTransferStatsHandle>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut last = 0u64;
        loop {
            let current = completed.load(Ordering::Relaxed);
            if let Ok(mut progress) = progress.lock() {
                if last == 0 {
                    progress.start_stage("Generating shards", Some(total_files));
                }
                progress.set_position(current);
                if let Some(stats) = upload_stats.as_ref() {
                    let snapshot = stats.snapshot();
                    let mut status = format!(
                        "upload buf={}/{} queued={} inflight_files={} inflight_parts={} started={} err={}",
                        snapshot.available_buffers,
                        snapshot.total_buffers,
                        snapshot.queued_jobs,
                        snapshot.inflight_files,
                        snapshot.inflight_parts,
                        snapshot.parts_started,
                        snapshot.parts_errors
                    );
                    if let Some(err) = snapshot.last_error.as_ref() {
                        status.push_str(" last_err=");
                        status.push_str(&truncate_status(err, 60));
                    }
                    progress.set_status(Some(status));
                }
            }
            last = current;
            if done.load(Ordering::Relaxed) || current >= total_files {
                break;
            }
            thread::sleep(Duration::from_millis(250));
        }
    })
}

fn truncate_status(input: &str, max_len: usize) -> String {
    if input.len() <= max_len {
        return input.to_string();
    }
    let mut out = input
        .chars()
        .take(max_len.saturating_sub(3))
        .collect::<String>();
    out.push_str("...");
    out
}

fn worker_loop(
    plan: Arc<Plan>,
    tasks: Arc<Mutex<VecDeque<ShardTask>>>,
    completed: Arc<AtomicU64>,
    output: OutputTarget,
) -> Result<()> {
    loop {
        let task = {
            let mut guard = tasks.lock().expect("task lock");
            guard.pop_front()
        };
        let Some(task) = task else { break };
        generate_shard(&plan, &task, &output, &completed)?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Plan {
    tables: Vec<TablePlan>,
    tasks: Vec<ShardTask>,
    seed: u64,
    batch_size: usize,
    row_group_size: usize,
}

#[derive(Clone, Debug)]
struct TablePlan {
    name: String,
    shards: usize,
    shard_ranges: Vec<RowRange>,
    schema: Arc<Schema>,
    columns: Vec<ColumnPlan>,
    table_seed: u64,
}

#[derive(Clone, Debug)]
struct RowRange {
    start: u64,
    len: u64,
}

#[derive(Clone, Debug)]
struct ColumnPlan {
    name: String,
    dtype: ColumnType,
    arrow_type: DataType,
    generator: ColumnGenerator,
    null_ratio: Option<f64>,
    timestamp_unit: Option<TimestampUnit>,
    col_seed: u64,
}

#[derive(Clone, Debug)]
enum ColumnGenerator {
    PrimaryKey(KeyGenerator),
    ForeignKey(ForeignKeyGenerator),
    Generator(PreparedGenerator),
}

#[derive(Clone, Debug)]
struct KeyGenerator {
    dtype: ColumnType,
    fixed_len: Option<usize>,
    encoding: KeyEncoding,
    table_seed: u64,
}

#[derive(Clone, Debug)]
struct ForeignKeyGenerator {
    dst_rows: u64,
    dst_key: KeyGenerator,
    distribution: FkDistribution,
}

#[derive(Clone, Debug)]
enum PreparedGenerator {
    Constant(GeneratedValue),
    Sequence {
        start: i64,
        step: i64,
    },
    UniformInt {
        min: i64,
        max: i64,
    },
    UniformUInt {
        min: u64,
        max: u64,
    },
    UniformFloat {
        min: f64,
        max: f64,
    },
    Bool {
        true_ratio: f64,
    },
    String {
        min_len: usize,
        max_len: usize,
        alphabet: Vec<u8>,
        prefix: String,
    },
    Bytes {
        min_len: usize,
        max_len: usize,
    },
    Uuid {
        mode: UuidMode,
        format: UuidFormat,
    },
    Timestamp {
        start_ms: i64,
        step_ms: i64,
    },
    Date {
        start_days: i32,
        step_days: i32,
    },
    Categorical {
        values: Vec<String>,
        cumulative: Vec<f64>,
    },
    ArrayF32 {
        len: usize,
        min: f32,
        max: f32,
    },
    ArrayF64 {
        len: usize,
        min: f64,
        max: f64,
    },
    ArrayI64 {
        len: usize,
        min: i64,
        max: i64,
    },
    ArrayStr {
        len: usize,
        min_len: usize,
        max_len: usize,
        alphabet: Vec<u8>,
    },
}

#[derive(Clone, Debug)]
enum GeneratedValue {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
    Date(i32),
    Timestamp(i64),
    ArrayF32(Vec<f32>),
    ArrayF64(Vec<f64>),
    ArrayI64(Vec<i64>),
    ArrayStr(Vec<String>),
}

#[derive(Clone, Copy, Debug)]
enum UuidMode {
    V4,
    FromId,
}

#[derive(Clone, Copy, Debug)]
enum UuidFormat {
    Dashed,
    Hex,
    Bytes,
}

#[derive(Clone, Debug)]
struct ShardTask {
    table_idx: usize,
    shard_idx: usize,
}

fn build_plan(config: &DatasetConfig) -> Result<Plan> {
    let mut tables = Vec::new();
    let row_counts = allocate_rows(config)?;
    let table_lookup: HashMap<&str, usize> = config
        .tables
        .iter()
        .enumerate()
        .map(|(idx, table)| (table.name.as_str(), idx))
        .collect();

    let mut pk_columns: Vec<Option<(usize, KeyGenerator)>> = Vec::new();
    for table in config.tables.iter() {
        let table_seed = mix64(config.seed ^ hash_name(&table.name));
        let pk = resolve_primary_key(table, table_seed)
            .with_context(|| format!("table {} primary key config", table.name))?;
        pk_columns.push(pk);
    }

    let fk_map = build_relationship_map(config, &table_lookup, &pk_columns, &row_counts)?;

    for (idx, table) in config.tables.iter().enumerate() {
        let row_count = row_counts[idx];
        let table_seed = mix64(config.seed ^ hash_name(&table.name));
        let pk = pk_columns[idx].as_ref();
        let columns = build_column_plans(table, table_seed, pk, idx, &fk_map)?;
        let schema = build_schema(&columns)?;
        let shard_ranges = shard_ranges(row_count, table.shards)?;
        tables.push(TablePlan {
            name: table.name.clone(),
            shards: table.shards,
            shard_ranges,
            schema,
            columns,
            table_seed,
        });
    }

    let mut tasks = Vec::new();
    for (table_idx, table) in tables.iter().enumerate() {
        for shard_idx in 0..table.shards {
            tasks.push(ShardTask {
                table_idx,
                shard_idx,
            });
        }
    }

    Ok(Plan {
        tables,
        tasks,
        seed: config.seed,
        batch_size: config.batch_size,
        row_group_size: config.row_group_size,
    })
}

fn allocate_rows(config: &DatasetConfig) -> Result<Vec<u64>> {
    let mut fixed = vec![0u64; config.tables.len()];
    let mut fractions = vec![0f64; config.tables.len()];
    let mut has_fraction = false;
    let mut fixed_total = 0u64;
    for (idx, table) in config.tables.iter().enumerate() {
        if let Some(rows) = table.rows {
            fixed[idx] = rows;
            fixed_total = fixed_total
                .checked_add(rows)
                .ok_or_else(|| anyhow!("row count overflow"))?;
        } else if let Some(percent) = table.row_percent {
            fractions[idx] = percent / 100.0;
            has_fraction = true;
        } else if let Some(frac) = table.row_fraction {
            fractions[idx] = frac;
            has_fraction = true;
        } else {
            bail!(
                "table {} must define rows, row_percent, or row_fraction",
                table.name
            );
        }
    }

    if fixed_total > config.total_rows {
        bail!(
            "sum of fixed rows {} exceeds total_rows {}",
            fixed_total,
            config.total_rows
        );
    }

    let remaining = config.total_rows - fixed_total;
    if has_fraction {
        let mut raw = Vec::new();
        let mut sum_base = 0u64;
        for (idx, frac) in fractions.iter().enumerate() {
            if *frac <= 0.0 {
                raw.push((idx, 0f64));
                continue;
            }
            let value = (remaining as f64) * frac;
            let base = value.floor() as u64;
            sum_base += base;
            raw.push((idx, value - base as f64));
            fixed[idx] += base;
        }
        let mut leftover = remaining.saturating_sub(sum_base);
        raw.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, _) in raw {
            if leftover == 0 {
                break;
            }
            fixed[idx] += 1;
            leftover -= 1;
        }
    }

    let assigned: u64 = fixed.iter().sum();
    if assigned != config.total_rows {
        bail!(
            "assigned rows {} does not match total_rows {}",
            assigned,
            config.total_rows
        );
    }
    Ok(fixed)
}

fn resolve_primary_key(
    table: &TableConfig,
    table_seed: u64,
) -> Result<Option<(usize, KeyGenerator)>> {
    let mut pk_idx = None;
    for (idx, column) in table.columns.iter().enumerate() {
        if column.primary_key {
            if pk_idx.is_some() {
                bail!("table {} has multiple primary_key columns", table.name);
            }
            pk_idx = Some(idx);
        }
    }
    let Some(pk_idx) = pk_idx else {
        return Ok(None);
    };
    let pk_col = &table.columns[pk_idx];
    validate_key_column(pk_col)?;
    let encoding = pk_col
        .key_encoding
        .unwrap_or(default_key_encoding(&pk_col.dtype));
    Ok(Some((
        pk_idx,
        KeyGenerator {
            dtype: pk_col.dtype,
            fixed_len: pk_col.fixed_len,
            encoding,
            table_seed,
        },
    )))
}

fn build_column_plans(
    table: &TableConfig,
    table_seed: u64,
    pk: Option<&(usize, KeyGenerator)>,
    table_idx: usize,
    fk_map: &HashMap<(usize, usize), ForeignKeyGenerator>,
) -> Result<Vec<ColumnPlan>> {
    let mut columns = Vec::new();
    let pk_index = pk.map(|(idx, _)| *idx);
    for (idx, column) in table.columns.iter().enumerate() {
        let arrow_type = column_arrow_type(column)?;
        let col_seed = mix64(table_seed ^ hash_name(&column.name));
        if let Some(null_ratio) = column.null_ratio {
            if !(0.0..=1.0).contains(&null_ratio) {
                bail!(
                    "column {} null_ratio {} must be between 0 and 1",
                    column.name,
                    null_ratio
                );
            }
        }
        let fk_generator = fk_map.get(&(table_idx, idx));
        let generator = if pk_index == Some(idx) {
            if column.gen.is_some() {
                bail!("primary key column {} should not define gen", column.name);
            }
            let (_, pk_generator) = pk.expect("pk exists");
            ColumnGenerator::PrimaryKey(pk_generator.clone())
        } else if let Some(fk) = fk_generator {
            ColumnGenerator::ForeignKey(fk.clone())
        } else {
            let Some(gen) = column.gen.clone() else {
                bail!("column {} missing gen definition", column.name);
            };
            ColumnGenerator::Generator(prepare_generator(column, gen)?)
        };
        columns.push(ColumnPlan {
            name: column.name.clone(),
            dtype: column.dtype,
            arrow_type,
            generator,
            null_ratio: if pk_index == Some(idx) || fk_generator.is_some() {
                None
            } else {
                column.null_ratio
            },
            timestamp_unit: column.timestamp_unit,
            col_seed,
        });
    }
    Ok(columns)
}

fn build_schema(columns: &[ColumnPlan]) -> Result<Arc<Schema>> {
    let mut fields = Vec::new();
    for column in columns {
        let field = Field::new(
            &column.name,
            column.arrow_type.clone(),
            column.null_ratio.is_some(),
        );
        fields.push(field);
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn build_relationship_map(
    config: &DatasetConfig,
    table_lookup: &HashMap<&str, usize>,
    pk_columns: &[Option<(usize, KeyGenerator)>],
    row_counts: &[u64],
) -> Result<HashMap<(usize, usize), ForeignKeyGenerator>> {
    let mut map = HashMap::new();
    for rel in &config.relationships {
        let src_idx = *table_lookup
            .get(rel.src_table.as_str())
            .ok_or_else(|| anyhow!("unknown src_table {}", rel.src_table))?;
        let dst_idx = *table_lookup
            .get(rel.dst_table.as_str())
            .ok_or_else(|| anyhow!("unknown dst_table {}", rel.dst_table))?;
        let Some((dst_pk_idx, dst_pk)) = pk_columns[dst_idx].as_ref() else {
            bail!(
                "relationship dst_table {} must define a primary key",
                rel.dst_table
            );
        };
        let dst_table = &config.tables[dst_idx];
        let dst_col = &dst_table.columns[*dst_pk_idx];
        if dst_col.name != rel.dst_column {
            bail!(
                "relationship dst_column {} does not match primary key {} for table {}",
                rel.dst_column,
                dst_col.name,
                dst_table.name
            );
        }
        let src_table = &config.tables[src_idx];
        let src_col_idx = src_table
            .columns
            .iter()
            .position(|col| col.name == rel.src_column)
            .ok_or_else(|| anyhow!("missing column {} in {}", rel.src_column, src_table.name))?;
        let src_col = &src_table.columns[src_col_idx];
        if src_col.primary_key {
            bail!(
                "relationship src_column {} on {} cannot be the primary key",
                src_col.name,
                src_table.name
            );
        }
        if src_col.gen.is_some() {
            bail!(
                "relationship src_column {} on {} should not define gen",
                src_col.name,
                src_table.name
            );
        }
        validate_fk_compat(src_col, dst_col)?;
        let distribution = rel.distribution.clone().unwrap_or(FkDistribution::Uniform);
        let fk = ForeignKeyGenerator {
            dst_rows: row_counts[dst_idx],
            dst_key: dst_pk.clone(),
            distribution,
        };
        if map.insert((src_idx, src_col_idx), fk).is_some() {
            bail!(
                "duplicate relationship for {}.{}",
                rel.src_table,
                rel.src_column
            );
        }
    }
    Ok(map)
}

fn generate_shard(
    plan: &Plan,
    task: &ShardTask,
    output: &OutputTarget,
    completed: &AtomicU64,
) -> Result<()> {
    let table = &plan.tables[task.table_idx];
    let range = &table.shard_ranges[task.shard_idx];
    let sink = output.parquet_sink(&table.name, task.shard_idx)?;
    let props = WriterProperties::builder()
        .set_max_row_group_size(plan.row_group_size)
        .build();
    let mut writer = ArrowWriter::try_new(sink, Arc::clone(&table.schema), Some(props))
        .with_context(|| format!("create parquet writer for {}", table.name))?;

    let mut remaining = range.len;
    let mut row_start = range.start;
    while remaining > 0 {
        let batch_len = std::cmp::min(plan.batch_size as u64, remaining) as usize;
        let batch = build_batch(table, row_start, batch_len, plan.seed)?;
        writer.write(&batch)?;
        row_start += batch_len as u64;
        remaining -= batch_len as u64;
    }

    let sink = writer.into_inner()?;
    let completion = sink.finish()?;
    completion.wait()?;
    completed.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn build_batch(table: &TablePlan, row_start: u64, len: usize, seed: u64) -> Result<RecordBatch> {
    let mut arrays = Vec::with_capacity(table.columns.len());
    for column in &table.columns {
        arrays.push(generate_column_array(table, column, row_start, len, seed)?);
    }
    RecordBatch::try_new(Arc::clone(&table.schema), arrays)
        .map_err(|e| anyhow!("record batch: {e}"))
}

fn generate_column_array(
    table: &TablePlan,
    column: &ColumnPlan,
    row_start: u64,
    len: usize,
    seed: u64,
) -> Result<ArrayRef> {
    let array: ArrayRef = match column.dtype {
        ColumnType::Int8 => {
            let mut builder = Int8Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Int16 => {
            let mut builder = Int16Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Int32 => {
            let mut builder = Int32Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Int64 => {
            let mut builder = Int64Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::UInt8 => {
            let mut builder = UInt8Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::UInt16 => {
            let mut builder = UInt16Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::UInt32 => {
            let mut builder = UInt32Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::UInt64 => {
            let mut builder = UInt64Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Float32 => {
            let mut builder = Float32Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Float64 => {
            let mut builder = Float64Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Bool => {
            let mut builder = BooleanBuilder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::String => {
            let mut builder = StringBuilder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Bytes => {
            let mut builder = BinaryBuilder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Uuid | ColumnType::Uuid256 | ColumnType::Ip => {
            let mut builder = StringBuilder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Date => {
            let mut builder = Date32Builder::new();
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Timestamp => {
            let unit = column.timestamp_unit.unwrap_or(TimestampUnit::Millis);
            match unit {
                TimestampUnit::Seconds => {
                    let mut builder = TimestampSecondBuilder::new();
                    for i in 0..len {
                        append_value(&mut builder, column, table, row_start + i as u64, seed)?;
                    }
                    Arc::new(builder.finish())
                }
                TimestampUnit::Millis => {
                    let mut builder = TimestampMillisecondBuilder::new();
                    for i in 0..len {
                        append_value(&mut builder, column, table, row_start + i as u64, seed)?;
                    }
                    Arc::new(builder.finish())
                }
                TimestampUnit::Micros => {
                    let mut builder = TimestampMicrosecondBuilder::new();
                    for i in 0..len {
                        append_value(&mut builder, column, table, row_start + i as u64, seed)?;
                    }
                    Arc::new(builder.finish())
                }
                TimestampUnit::Nanos => {
                    let mut builder = TimestampNanosecondBuilder::new();
                    for i in 0..len {
                        append_value(&mut builder, column, table, row_start + i as u64, seed)?;
                    }
                    Arc::new(builder.finish())
                }
            }
        }
        ColumnType::Float32Array => {
            let values = Float32Builder::new();
            let mut builder = ListBuilder::new(values);
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Float64Array => {
            let values = Float64Builder::new();
            let mut builder = ListBuilder::new(values);
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Int8Array => {
            let values = Int8Builder::new();
            let mut builder = ListBuilder::new(values);
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Int16Array => {
            let values = Int16Builder::new();
            let mut builder = ListBuilder::new(values);
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Int32Array => {
            let values = Int32Builder::new();
            let mut builder = ListBuilder::new(values);
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::Int64Array => {
            let values = Int64Builder::new();
            let mut builder = ListBuilder::new(values);
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
        ColumnType::StringArray => {
            let values = StringBuilder::new();
            let mut builder = ListBuilder::new(values);
            for i in 0..len {
                append_value(&mut builder, column, table, row_start + i as u64, seed)?;
            }
            Arc::new(builder.finish())
        }
    };
    Ok(array)
}

fn append_value<T>(
    builder: &mut T,
    column: &ColumnPlan,
    table: &TablePlan,
    row_id: u64,
    seed: u64,
) -> Result<()>
where
    T: ValueAppender,
{
    let value = column.generate_value(table, row_id, seed)?;
    builder.append_value(value)?;
    Ok(())
}

trait ValueAppender {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()>;
}

impl ValueAppender for Int8Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Int(v)) => self.append_value(v as i8),
            Some(_) => bail!("type mismatch for int8"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for Int16Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Int(v)) => self.append_value(v as i16),
            Some(_) => bail!("type mismatch for int16"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for Int32Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Int(v)) => self.append_value(v as i32),
            Some(_) => bail!("type mismatch for int32"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for Int64Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Int(v)) => self.append_value(v),
            Some(_) => bail!("type mismatch for int64"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for UInt8Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::UInt(v)) => self.append_value(v as u8),
            Some(_) => bail!("type mismatch for uint8"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for UInt16Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::UInt(v)) => self.append_value(v as u16),
            Some(_) => bail!("type mismatch for uint16"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for UInt32Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::UInt(v)) => self.append_value(v as u32),
            Some(_) => bail!("type mismatch for uint32"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for UInt64Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::UInt(v)) => self.append_value(v),
            Some(_) => bail!("type mismatch for uint64"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for Float32Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Float(v)) => self.append_value(v as f32),
            Some(_) => bail!("type mismatch for float32"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for Float64Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Float(v)) => self.append_value(v),
            Some(_) => bail!("type mismatch for float64"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for BooleanBuilder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Bool(v)) => self.append_value(v),
            Some(_) => bail!("type mismatch for bool"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for StringBuilder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::String(v)) => self.append_value(&v),
            Some(_) => bail!("type mismatch for string"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for BinaryBuilder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Bytes(v)) => self.append_value(&v),
            Some(_) => bail!("type mismatch for bytes"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for Date32Builder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Date(v)) => self.append_value(v),
            Some(_) => bail!("type mismatch for date"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for TimestampSecondBuilder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Timestamp(v)) => self.append_value(v),
            Some(_) => bail!("type mismatch for timestamp"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for TimestampMillisecondBuilder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Timestamp(v)) => self.append_value(v),
            Some(_) => bail!("type mismatch for timestamp"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for TimestampMicrosecondBuilder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Timestamp(v)) => self.append_value(v),
            Some(_) => bail!("type mismatch for timestamp"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for TimestampNanosecondBuilder {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::Timestamp(v)) => self.append_value(v),
            Some(_) => bail!("type mismatch for timestamp"),
            None => self.append_null(),
        }
        Ok(())
    }
}

impl ValueAppender for ListBuilder<Float32Builder> {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::ArrayF32(values)) => {
                let inner = self.values();
                for value in values {
                    inner.append_value(value);
                }
                self.append(true);
            }
            Some(_) => bail!("type mismatch for float32 array"),
            None => {
                self.append(false);
            }
        }
        Ok(())
    }
}

impl ValueAppender for ListBuilder<Float64Builder> {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::ArrayF64(values)) => {
                let inner = self.values();
                for value in values {
                    inner.append_value(value);
                }
                self.append(true);
            }
            Some(_) => bail!("type mismatch for float64 array"),
            None => {
                self.append(false);
            }
        }
        Ok(())
    }
}

impl ValueAppender for ListBuilder<Int8Builder> {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::ArrayI64(values)) => {
                let inner = self.values();
                for value in values {
                    inner.append_value(value as i8);
                }
                self.append(true);
            }
            Some(_) => bail!("type mismatch for int8 array"),
            None => {
                self.append(false);
            }
        }
        Ok(())
    }
}

impl ValueAppender for ListBuilder<Int16Builder> {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::ArrayI64(values)) => {
                let inner = self.values();
                for value in values {
                    inner.append_value(value as i16);
                }
                self.append(true);
            }
            Some(_) => bail!("type mismatch for int16 array"),
            None => {
                self.append(false);
            }
        }
        Ok(())
    }
}

impl ValueAppender for ListBuilder<Int32Builder> {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::ArrayI64(values)) => {
                let inner = self.values();
                for value in values {
                    inner.append_value(value as i32);
                }
                self.append(true);
            }
            Some(_) => bail!("type mismatch for int32 array"),
            None => {
                self.append(false);
            }
        }
        Ok(())
    }
}

impl ValueAppender for ListBuilder<Int64Builder> {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::ArrayI64(values)) => {
                let inner = self.values();
                for value in values {
                    inner.append_value(value);
                }
                self.append(true);
            }
            Some(_) => bail!("type mismatch for int64 array"),
            None => {
                self.append(false);
            }
        }
        Ok(())
    }
}

impl ValueAppender for ListBuilder<StringBuilder> {
    fn append_value(&mut self, value: Option<GeneratedValue>) -> Result<()> {
        match value {
            Some(GeneratedValue::ArrayStr(values)) => {
                let inner = self.values();
                for value in values {
                    inner.append_value(&value);
                }
                self.append(true);
            }
            Some(_) => bail!("type mismatch for string array"),
            None => {
                self.append(false);
            }
        }
        Ok(())
    }
}

impl ColumnPlan {
    fn generate_value(
        &self,
        table: &TablePlan,
        row_id: u64,
        seed: u64,
    ) -> Result<Option<GeneratedValue>> {
        if let Some(null_ratio) = self.null_ratio {
            let random = rand_f64(mix64(seed ^ self.col_seed ^ row_id));
            if random < null_ratio {
                return Ok(None);
            }
        }
        let value = match &self.generator {
            ColumnGenerator::PrimaryKey(pk) => pk.value_for_id(row_id, self.timestamp_unit)?,
            ColumnGenerator::ForeignKey(fk) => {
                let target = fk.target_id(row_id, seed ^ self.col_seed);
                fk.dst_key.value_for_id(target, self.timestamp_unit)?
            }
            ColumnGenerator::Generator(gen) => gen.value_for_row(
                row_id,
                seed ^ table.table_seed ^ self.col_seed,
                self.dtype,
                self.timestamp_unit,
            )?,
        };
        Ok(Some(value))
    }
}

impl KeyGenerator {
    fn value_for_id(
        &self,
        id: u64,
        timestamp_unit: Option<TimestampUnit>,
    ) -> Result<GeneratedValue> {
        match self.dtype {
            ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::Int64 => {
                Ok(GeneratedValue::Int(id as i64))
            }
            ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
                Ok(GeneratedValue::UInt(id))
            }
            ColumnType::String => {
                let bytes = encode_key_bytes(self.encoding, self.table_seed, id);
                let value = String::from_utf8_lossy(&bytes).to_string();
                let value = apply_fixed_len_string(value, self.fixed_len);
                Ok(GeneratedValue::String(value))
            }
            ColumnType::Bytes => {
                let mut bytes = encode_key_bytes(self.encoding, self.table_seed, id);
                apply_fixed_len_bytes(&mut bytes, self.fixed_len);
                Ok(GeneratedValue::Bytes(bytes))
            }
            ColumnType::Uuid => Ok(GeneratedValue::String(hex_encode(&uuid_from_id(
                self.table_seed,
                id,
            )))),
            ColumnType::Uuid256 => Ok(GeneratedValue::String(hex_encode(&uuid256_from_id(
                self.table_seed,
                id,
            )))),
            ColumnType::Ip => Ok(GeneratedValue::String(format_ip(ip_from_id(
                self.table_seed,
                id,
            )))),
            ColumnType::Timestamp => {
                let unit = timestamp_unit.unwrap_or(TimestampUnit::Millis);
                let value = match unit {
                    TimestampUnit::Seconds => id as i64,
                    TimestampUnit::Millis => (id as i64) * 1_000,
                    TimestampUnit::Micros => (id as i64) * 1_000_000,
                    TimestampUnit::Nanos => (id as i64) * 1_000_000_000,
                };
                Ok(GeneratedValue::Timestamp(value))
            }
            _ => bail!("unsupported key type {:?}", self.dtype),
        }
    }
}

impl ForeignKeyGenerator {
    fn target_id(&self, row_id: u64, seed: u64) -> u64 {
        if self.dst_rows == 0 {
            return 0;
        }
        match self.distribution.clone() {
            FkDistribution::Uniform => mix64(seed ^ row_id) % self.dst_rows,
            FkDistribution::Sequential { offset } => (row_id + offset) % self.dst_rows,
            FkDistribution::Hotspot {
                hot_fraction,
                hot_share,
            } => {
                let hot_size = ((self.dst_rows as f64) * hot_fraction).max(1.0) as u64;
                let random = rand_f64(mix64(seed ^ row_id));
                if random < hot_share {
                    mix64(seed ^ row_id ^ 0xabc) % hot_size
                } else {
                    hot_size + (mix64(seed ^ row_id ^ 0xdef) % (self.dst_rows - hot_size))
                }
            }
        }
    }
}

impl PreparedGenerator {
    fn value_for_row(
        &self,
        row_id: u64,
        seed: u64,
        dtype: ColumnType,
        timestamp_unit: Option<TimestampUnit>,
    ) -> Result<GeneratedValue> {
        match self {
            PreparedGenerator::Constant(value) => Ok(value.clone()),
            PreparedGenerator::Sequence { start, step } => {
                let value = start + *step * row_id as i64;
                match dtype {
                    ColumnType::UInt8
                    | ColumnType::UInt16
                    | ColumnType::UInt32
                    | ColumnType::UInt64 => Ok(GeneratedValue::UInt(value as u64)),
                    _ => Ok(GeneratedValue::Int(value)),
                }
            }
            PreparedGenerator::UniformInt { min, max } => {
                let span = (*max - *min + 1).max(1);
                let value = *min + (mix64(seed ^ row_id) as i64 % span);
                Ok(GeneratedValue::Int(value))
            }
            PreparedGenerator::UniformUInt { min, max } => {
                let span = (*max - *min + 1).max(1);
                let value = *min + (mix64(seed ^ row_id) % span);
                Ok(GeneratedValue::UInt(value))
            }
            PreparedGenerator::UniformFloat { min, max } => {
                let random = rand_f64(mix64(seed ^ row_id));
                Ok(GeneratedValue::Float(min + (max - min) * random))
            }
            PreparedGenerator::Bool { true_ratio } => {
                let random = rand_f64(mix64(seed ^ row_id));
                Ok(GeneratedValue::Bool(random < *true_ratio))
            }
            PreparedGenerator::String {
                min_len,
                max_len,
                alphabet,
                prefix,
            } => {
                let len = pick_len(*min_len, *max_len, seed ^ row_id);
                let mut value = build_string(seed ^ row_id, len, alphabet);
                if !prefix.is_empty() {
                    value.insert_str(0, prefix);
                }
                Ok(GeneratedValue::String(value))
            }
            PreparedGenerator::Bytes { min_len, max_len } => {
                let len = pick_len(*min_len, *max_len, seed ^ row_id);
                let bytes = build_bytes(seed ^ row_id, len);
                Ok(GeneratedValue::Bytes(bytes))
            }
            PreparedGenerator::Uuid { mode, format } => {
                let bytes = match mode {
                    UuidMode::V4 => uuid_from_id(seed, row_id),
                    UuidMode::FromId => uuid_from_id(seed ^ 0x55aa, row_id),
                };
                match (format, dtype) {
                    (UuidFormat::Bytes, _) => Ok(GeneratedValue::Bytes(bytes.to_vec())),
                    (UuidFormat::Hex, ColumnType::Bytes) => {
                        Ok(GeneratedValue::Bytes(hex_encode(&bytes).into_bytes()))
                    }
                    (UuidFormat::Dashed, ColumnType::Bytes) => Ok(GeneratedValue::Bytes(
                        format_uuid(bytes, *format).into_bytes(),
                    )),
                    (UuidFormat::Hex, _) => Ok(GeneratedValue::String(hex_encode(&bytes))),
                    (UuidFormat::Dashed, _) => {
                        Ok(GeneratedValue::String(format_uuid(bytes, *format)))
                    }
                }
            }
            PreparedGenerator::Timestamp { start_ms, step_ms } => {
                let value_ms = *start_ms + (*step_ms * row_id as i64);
                let unit = timestamp_unit.unwrap_or(TimestampUnit::Millis);
                let value = match unit {
                    TimestampUnit::Seconds => value_ms / 1000,
                    TimestampUnit::Millis => value_ms,
                    TimestampUnit::Micros => value_ms * 1000,
                    TimestampUnit::Nanos => value_ms * 1_000_000,
                };
                Ok(GeneratedValue::Timestamp(value))
            }
            PreparedGenerator::Date {
                start_days,
                step_days,
            } => {
                let value = *start_days + (*step_days * row_id as i32);
                Ok(GeneratedValue::Date(value))
            }
            PreparedGenerator::Categorical { values, cumulative } => {
                let random = rand_f64(mix64(seed ^ row_id));
                let idx = cumulative.iter().position(|v| random <= *v).unwrap_or(0);
                Ok(GeneratedValue::String(values[idx].clone()))
            }
            PreparedGenerator::ArrayF32 { len, min, max } => {
                let mut values = Vec::with_capacity(*len);
                for idx in 0..*len {
                    let rand = rand_f64(mix64(seed ^ row_id ^ idx as u64));
                    values.push((*min as f64 + (*max - *min) as f64 * rand) as f32);
                }
                Ok(GeneratedValue::ArrayF32(values))
            }
            PreparedGenerator::ArrayF64 { len, min, max } => {
                let mut values = Vec::with_capacity(*len);
                for idx in 0..*len {
                    let rand = rand_f64(mix64(seed ^ row_id ^ idx as u64));
                    values.push(*min + (*max - *min) * rand);
                }
                Ok(GeneratedValue::ArrayF64(values))
            }
            PreparedGenerator::ArrayI64 { len, min, max } => {
                let span = ((*max as i128 - *min as i128) + 1).max(1) as u64;
                let mut values = Vec::with_capacity(*len);
                for idx in 0..*len {
                    let value = *min + (mix64(seed ^ row_id ^ idx as u64) % span) as i64;
                    values.push(value);
                }
                Ok(GeneratedValue::ArrayI64(values))
            }
            PreparedGenerator::ArrayStr {
                len,
                min_len,
                max_len,
                alphabet,
            } => {
                let mut values = Vec::with_capacity(*len);
                for idx in 0..*len {
                    let str_seed = mix64(seed ^ row_id ^ idx as u64);
                    let elem_len = pick_len(*min_len, *max_len, str_seed);
                    values.push(build_string(str_seed, elem_len, alphabet));
                }
                Ok(GeneratedValue::ArrayStr(values))
            }
        }
    }
}

fn column_arrow_type(column: &ColumnConfig) -> Result<DataType> {
    Ok(match column.dtype {
        ColumnType::Int8 => DataType::Int8,
        ColumnType::Int16 => DataType::Int16,
        ColumnType::Int32 => DataType::Int32,
        ColumnType::Int64 => DataType::Int64,
        ColumnType::UInt8 => DataType::UInt8,
        ColumnType::UInt16 => DataType::UInt16,
        ColumnType::UInt32 => DataType::UInt32,
        ColumnType::UInt64 => DataType::UInt64,
        ColumnType::Float32 => DataType::Float32,
        ColumnType::Float64 => DataType::Float64,
        ColumnType::Bool => DataType::Boolean,
        ColumnType::String => DataType::Utf8,
        ColumnType::Bytes => DataType::Binary,
        ColumnType::Uuid => DataType::Utf8,
        ColumnType::Uuid256 => DataType::Utf8,
        ColumnType::Ip => DataType::Utf8,
        ColumnType::Date => DataType::Date32,
        ColumnType::Timestamp => {
            let unit = match column.timestamp_unit.unwrap_or(TimestampUnit::Millis) {
                TimestampUnit::Seconds => TimeUnit::Second,
                TimestampUnit::Millis => TimeUnit::Millisecond,
                TimestampUnit::Micros => TimeUnit::Microsecond,
                TimestampUnit::Nanos => TimeUnit::Nanosecond,
            };
            DataType::Timestamp(unit, None)
        }
        ColumnType::Float32Array => {
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true)))
        }
        ColumnType::Float64Array => {
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true)))
        }
        ColumnType::Int8Array => {
            DataType::List(Arc::new(Field::new("item", DataType::Int8, true)))
        }
        ColumnType::Int16Array => {
            DataType::List(Arc::new(Field::new("item", DataType::Int16, true)))
        }
        ColumnType::Int32Array => {
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true)))
        }
        ColumnType::Int64Array => {
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true)))
        }
        ColumnType::StringArray => {
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)))
        }
    })
}

fn prepare_generator(column: &ColumnConfig, gen: GeneratorSpec) -> Result<PreparedGenerator> {
    match gen {
        GeneratorSpec::Constant { value } => Ok(PreparedGenerator::Constant(parse_constant(
            column.dtype,
            value,
        )?)),
        GeneratorSpec::Sequence { start, step } => {
            ensure_dtype(
                column.dtype,
                &[
                    ColumnType::Int8,
                    ColumnType::Int16,
                    ColumnType::Int32,
                    ColumnType::Int64,
                    ColumnType::UInt8,
                    ColumnType::UInt16,
                    ColumnType::UInt32,
                    ColumnType::UInt64,
                ],
            )?;
            Ok(PreparedGenerator::Sequence { start, step })
        }
        GeneratorSpec::UniformInt { min, max } => {
            ensure_dtype(
                column.dtype,
                &[
                    ColumnType::Int8,
                    ColumnType::Int16,
                    ColumnType::Int32,
                    ColumnType::Int64,
                ],
            )?;
            if min > max {
                bail!("uniform_int min {} exceeds max {}", min, max);
            }
            Ok(PreparedGenerator::UniformInt { min, max })
        }
        GeneratorSpec::UniformUInt { min, max } => {
            ensure_dtype(
                column.dtype,
                &[
                    ColumnType::UInt8,
                    ColumnType::UInt16,
                    ColumnType::UInt32,
                    ColumnType::UInt64,
                ],
            )?;
            if min > max {
                bail!("uniform_uint min {} exceeds max {}", min, max);
            }
            Ok(PreparedGenerator::UniformUInt { min, max })
        }
        GeneratorSpec::UniformFloat { min, max } => {
            ensure_dtype(column.dtype, &[ColumnType::Float32, ColumnType::Float64])?;
            if min > max {
                bail!("uniform_float min {} exceeds max {}", min, max);
            }
            Ok(PreparedGenerator::UniformFloat { min, max })
        }
        GeneratorSpec::Bool { true_ratio } => {
            ensure_dtype(column.dtype, &[ColumnType::Bool])?;
            if !(0.0..=1.0).contains(&true_ratio) {
                bail!("bool true_ratio {} must be between 0 and 1", true_ratio);
            }
            Ok(PreparedGenerator::Bool { true_ratio })
        }
        GeneratorSpec::String {
            len,
            min_len,
            max_len,
            alphabet,
            prefix,
        } => {
            ensure_dtype(column.dtype, &[ColumnType::String])?;
            let (min_len, max_len) = resolve_len(len, min_len, max_len)?;
            let alphabet = resolve_alphabet(alphabet);
            let prefix = prefix.unwrap_or_default();
            Ok(PreparedGenerator::String {
                min_len,
                max_len,
                alphabet,
                prefix,
            })
        }
        GeneratorSpec::Bytes {
            len,
            min_len,
            max_len,
        } => {
            ensure_dtype(column.dtype, &[ColumnType::Bytes])?;
            let (min_len, max_len) = resolve_len(len, min_len, max_len)?;
            Ok(PreparedGenerator::Bytes { min_len, max_len })
        }
        GeneratorSpec::Uuid { mode, format } => {
            ensure_dtype(
                column.dtype,
                &[ColumnType::String, ColumnType::Bytes, ColumnType::Uuid],
            )?;
            let mode = match mode.as_deref().unwrap_or("from_id") {
                "v4" => UuidMode::V4,
                "from_id" => UuidMode::FromId,
                other => bail!("unknown uuid mode {}", other),
            };
            let default_format = if matches!(column.dtype, ColumnType::Bytes) {
                "bytes"
            } else if matches!(column.dtype, ColumnType::Uuid) {
                "hex"
            } else {
                "dashed"
            };
            let format = match format.as_deref().unwrap_or(default_format) {
                "dashed" => UuidFormat::Dashed,
                "hex" => UuidFormat::Hex,
                "bytes" => UuidFormat::Bytes,
                other => bail!("unknown uuid format {}", other),
            };
            if column.dtype == ColumnType::String && matches!(format, UuidFormat::Bytes) {
                bail!("uuid bytes format requires bytes dtype");
            }
            if column.dtype == ColumnType::Uuid && matches!(format, UuidFormat::Bytes) {
                bail!("uuid bytes format not allowed for uuid dtype");
            }
            Ok(PreparedGenerator::Uuid { mode, format })
        }
        GeneratorSpec::Timestamp {
            start_epoch_ms,
            step_ms,
        } => {
            ensure_dtype(column.dtype, &[ColumnType::Timestamp])?;
            Ok(PreparedGenerator::Timestamp {
                start_ms: start_epoch_ms,
                step_ms,
            })
        }
        GeneratorSpec::Date {
            start_ymd,
            step_days,
        } => {
            ensure_dtype(column.dtype, &[ColumnType::Date])?;
            let start_days = parse_date_days(&start_ymd)?;
            Ok(PreparedGenerator::Date {
                start_days,
                step_days,
            })
        }
        GeneratorSpec::Categorical { values, weights } => {
            ensure_dtype(column.dtype, &[ColumnType::String])?;
            let cumulative = build_cumulative(&values, weights)?;
            Ok(PreparedGenerator::Categorical { values, cumulative })
        }
        GeneratorSpec::ArrayFloat32 { len, min, max } => {
            ensure_dtype(column.dtype, &[ColumnType::Float32Array])?;
            if min > max {
                bail!("array_float32 min {} exceeds max {}", min, max);
            }
            Ok(PreparedGenerator::ArrayF32 { len, min, max })
        }
        GeneratorSpec::ArrayFloat64 { len, min, max } => {
            ensure_dtype(column.dtype, &[ColumnType::Float64Array])?;
            if min > max {
                bail!("array_float64 min {} exceeds max {}", min, max);
            }
            Ok(PreparedGenerator::ArrayF64 { len, min, max })
        }
        GeneratorSpec::ArrayInt64 { len, min, max } => {
            ensure_dtype(column.dtype, &[ColumnType::Int64Array])?;
            if min > max {
                bail!("array_int64 min {} exceeds max {}", min, max);
            }
            Ok(PreparedGenerator::ArrayI64 { len, min, max })
        }
        GeneratorSpec::ArrayString {
            len,
            min_len,
            max_len,
            alphabet,
        } => {
            ensure_dtype(column.dtype, &[ColumnType::StringArray])?;
            let (min_len, max_len) = resolve_len(None, min_len, max_len)?;
            let alphabet = resolve_alphabet(alphabet);
            Ok(PreparedGenerator::ArrayStr {
                len,
                min_len,
                max_len,
                alphabet,
            })
        }
    }
}

fn ensure_dtype(dtype: ColumnType, allowed: &[ColumnType]) -> Result<()> {
    if !allowed.contains(&dtype) {
        bail!("generator not compatible with dtype {:?}", dtype);
    }
    Ok(())
}

fn resolve_len(
    len: Option<usize>,
    min_len: Option<usize>,
    max_len: Option<usize>,
) -> Result<(usize, usize)> {
    if let Some(len) = len {
        return Ok((len, len));
    }
    let min_len = min_len.unwrap_or(4);
    let max_len = max_len.unwrap_or(min_len);
    if min_len > max_len {
        bail!("min_len {} exceeds max_len {}", min_len, max_len);
    }
    Ok((min_len, max_len))
}

fn parse_constant(dtype: ColumnType, value: serde_json::Value) -> Result<GeneratedValue> {
    Ok(match dtype {
        ColumnType::Int8 | ColumnType::Int16 | ColumnType::Int32 | ColumnType::Int64 => {
            GeneratedValue::Int(value.as_i64().unwrap_or(0))
        }
        ColumnType::UInt8 | ColumnType::UInt16 | ColumnType::UInt32 | ColumnType::UInt64 => {
            GeneratedValue::UInt(value.as_u64().unwrap_or(0))
        }
        ColumnType::Float32 | ColumnType::Float64 => {
            GeneratedValue::Float(value.as_f64().unwrap_or(0.0))
        }
        ColumnType::Bool => GeneratedValue::Bool(value.as_bool().unwrap_or(false)),
        ColumnType::String => {
            GeneratedValue::String(value.as_str().unwrap_or_default().to_string())
        }
        ColumnType::Bytes | ColumnType::Uuid | ColumnType::Uuid256 | ColumnType::Ip => {
            let bytes = value.as_str().unwrap_or_default().as_bytes().to_vec();
            GeneratedValue::Bytes(bytes)
        }
        ColumnType::Date => GeneratedValue::Date(value.as_i64().unwrap_or(0) as i32),
        ColumnType::Timestamp => GeneratedValue::Timestamp(value.as_i64().unwrap_or(0)),
        ColumnType::Float32Array => {
            let mut values = Vec::new();
            if let Some(array) = value.as_array() {
                for entry in array {
                    values.push(entry.as_f64().unwrap_or(0.0) as f32);
                }
            }
            GeneratedValue::ArrayF32(values)
        }
        ColumnType::Float64Array => {
            let mut values = Vec::new();
            if let Some(array) = value.as_array() {
                for entry in array {
                    values.push(entry.as_f64().unwrap_or(0.0));
                }
            }
            GeneratedValue::ArrayF64(values)
        }
        ColumnType::Int8Array
        | ColumnType::Int16Array
        | ColumnType::Int32Array
        | ColumnType::Int64Array => {
            let mut values = Vec::new();
            if let Some(array) = value.as_array() {
                for entry in array {
                    values.push(entry.as_i64().unwrap_or(0));
                }
            }
            GeneratedValue::ArrayI64(values)
        }
        ColumnType::StringArray => {
            let mut values = Vec::new();
            if let Some(array) = value.as_array() {
                for entry in array {
                    values.push(entry.as_str().unwrap_or_default().to_string());
                }
            }
            GeneratedValue::ArrayStr(values)
        }
    })
}

fn resolve_alphabet(name: Option<String>) -> Vec<u8> {
    match name.as_deref().unwrap_or("alnum") {
        "hex" => b"0123456789abcdef".to_vec(),
        "lower" => b"abcdefghijklmnopqrstuvwxyz".to_vec(),
        "upper" => b"ABCDEFGHIJKLMNOPQRSTUVWXYZ".to_vec(),
        "digits" => b"0123456789".to_vec(),
        _ => b"abcdefghijklmnopqrstuvwxyz0123456789".to_vec(),
    }
}

fn build_cumulative(values: &[String], weights: Option<Vec<f64>>) -> Result<Vec<f64>> {
    if values.is_empty() {
        bail!("categorical values cannot be empty");
    }
    let weights = weights.unwrap_or_else(|| vec![1.0; values.len()]);
    if weights.len() != values.len() {
        bail!("categorical weights length mismatch");
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        bail!("categorical weights must be positive");
    }
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for weight in weights {
        acc += weight / total;
        cumulative.push(acc);
    }
    Ok(cumulative)
}

fn parse_date_days(value: &str) -> Result<i32> {
    let date = chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .map_err(|e| anyhow!("invalid date {}: {}", value, e))?;
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    Ok((date - epoch).num_days() as i32)
}

fn validate_key_column(column: &ColumnConfig) -> Result<()> {
    match column.dtype {
        ColumnType::Int8
        | ColumnType::Int16
        | ColumnType::Int32
        | ColumnType::Int64
        | ColumnType::UInt8
        | ColumnType::UInt16
        | ColumnType::UInt32
        | ColumnType::UInt64
        | ColumnType::String
        | ColumnType::Bytes
        | ColumnType::Uuid
        | ColumnType::Uuid256
        | ColumnType::Ip => {}
        _ => bail!(
            "invalid key dtype {:?} for column {}",
            column.dtype,
            column.name
        ),
    }
    if column.dtype == ColumnType::String && column.key_encoding == Some(KeyEncoding::U64Le) {
        bail!("key_encoding u64_le is only valid for bytes columns");
    }
    if column.fixed_len.is_some() && !matches!(column.dtype, ColumnType::String | ColumnType::Bytes)
    {
        bail!("fixed_len only valid for string/bytes for {}", column.name);
    }
    Ok(())
}

fn validate_fk_compat(src: &ColumnConfig, dst: &ColumnConfig) -> Result<()> {
    if src.dtype != dst.dtype {
        bail!(
            "foreign key type mismatch {} ({:?}) -> {} ({:?})",
            src.name,
            src.dtype,
            dst.name,
            dst.dtype
        );
    }
    if src.fixed_len != dst.fixed_len {
        bail!(
            "foreign key fixed_len mismatch {} ({:?}) -> {} ({:?})",
            src.name,
            src.fixed_len,
            dst.name,
            dst.fixed_len
        );
    }
    Ok(())
}

fn shard_ranges(total: u64, shards: usize) -> Result<Vec<RowRange>> {
    if shards == 0 {
        bail!("shards must be >= 1");
    }
    let base = total / shards as u64;
    let mut rem = total % shards as u64;
    let mut ranges = Vec::with_capacity(shards);
    let mut start = 0u64;
    for _ in 0..shards {
        let extra = if rem > 0 {
            rem -= 1;
            1
        } else {
            0
        };
        let len = base + extra;
        ranges.push(RowRange { start, len });
        start += len;
    }
    Ok(ranges)
}

fn default_key_encoding(dtype: &ColumnType) -> KeyEncoding {
    match dtype {
        ColumnType::String | ColumnType::Bytes => KeyEncoding::Decimal,
        _ => KeyEncoding::Decimal,
    }
}

fn encode_key_bytes(encoding: KeyEncoding, seed: u64, id: u64) -> Vec<u8> {
    match encoding {
        KeyEncoding::Decimal => id.to_string().into_bytes(),
        KeyEncoding::Hex => format!("{:016x}", id).into_bytes(),
        KeyEncoding::U64Le => {
            let value = mix64(seed ^ id);
            value.to_le_bytes().to_vec()
        }
    }
}

fn apply_fixed_len_bytes(bytes: &mut Vec<u8>, fixed_len: Option<usize>) {
    let Some(fixed_len) = fixed_len else { return };
    if bytes.len() > fixed_len {
        bytes.truncate(fixed_len);
    } else if bytes.len() < fixed_len {
        bytes.resize(fixed_len, 0);
    }
}

fn apply_fixed_len_string(mut value: String, fixed_len: Option<usize>) -> String {
    let Some(fixed_len) = fixed_len else {
        return value;
    };
    if value.len() > fixed_len {
        value.truncate(fixed_len);
    } else if value.len() < fixed_len {
        let padding = "0".repeat(fixed_len - value.len());
        value.push_str(&padding);
    }
    value
}

fn uuid_from_id(seed: u64, id: u64) -> [u8; 16] {
    let hi = mix64(seed ^ id);
    let lo = mix64(seed ^ id ^ 0x9e3779b97f4a7c15);
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&hi.to_be_bytes());
    bytes[8..].copy_from_slice(&lo.to_be_bytes());
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    bytes
}

fn uuid256_from_id(seed: u64, id: u64) -> [u8; 32] {
    let v0 = mix64(seed ^ id);
    let v1 = mix64(seed ^ id ^ 0x9e3779b97f4a7c15);
    let v2 = mix64(seed ^ id ^ 0x94d049bb133111eb);
    let v3 = mix64(seed ^ id ^ 0x2545f4914f6cdd1d);
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(&v0.to_be_bytes());
    bytes[8..16].copy_from_slice(&v1.to_be_bytes());
    bytes[16..24].copy_from_slice(&v2.to_be_bytes());
    bytes[24..32].copy_from_slice(&v3.to_be_bytes());
    bytes
}

fn ip_from_id(seed: u64, id: u64) -> [u8; 16] {
    let v4 = (mix64(seed ^ id) as u32).to_be_bytes();
    let mut bytes = [0u8; 16];
    bytes[10] = 0xff;
    bytes[11] = 0xff;
    bytes[12..16].copy_from_slice(&v4);
    bytes
}

fn format_uuid(bytes: [u8; 16], format: UuidFormat) -> String {
    match format {
        UuidFormat::Hex => hex_encode(&bytes),
        _ => {
            let hex = hex_encode(&bytes);
            format!(
                "{}-{}-{}-{}-{}",
                &hex[0..8],
                &hex[8..12],
                &hex[12..16],
                &hex[16..20],
                &hex[20..32]
            )
        }
    }
}

fn format_ip(bytes: [u8; 16]) -> String {
    let addr = std::net::Ipv6Addr::from(bytes);
    if let Some(v4) = addr.to_ipv4() {
        v4.to_string()
    } else {
        addr.to_string()
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(hex_char(byte >> 4));
        out.push(hex_char(byte & 0x0f));
    }
    out
}

fn hex_char(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        _ => (b'a' + (value - 10)) as char,
    }
}

fn build_string(seed: u64, len: usize, alphabet: &[u8]) -> String {
    if alphabet.is_empty() || len == 0 {
        return String::new();
    }
    let mut value = Vec::with_capacity(len);
    let mut current = seed;
    for _ in 0..len {
        current = mix64(current);
        let idx = (current % alphabet.len() as u64) as usize;
        value.push(alphabet[idx]);
    }
    String::from_utf8_lossy(&value).to_string()
}

fn build_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut current = seed;
    for _ in 0..len {
        current = mix64(current);
        out.push((current & 0xff) as u8);
    }
    out
}

fn pick_len(min_len: usize, max_len: usize, seed: u64) -> usize {
    if min_len == max_len {
        min_len
    } else {
        let span = (max_len - min_len + 1) as u64;
        min_len + (mix64(seed) % span) as usize
    }
}

fn rand_f64(value: u64) -> f64 {
    let value = value >> 11;
    (value as f64) / ((1u64 << 53) as f64)
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn hash_name(value: &str) -> u64 {
    let mut hash = 1469598103934665603u64;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}
