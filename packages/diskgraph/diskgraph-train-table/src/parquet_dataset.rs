// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use arrow_array::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::{Path, PathBuf};

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::errors::{TTGenError, TTGenResult};

pub const TT_DATASET_DIR: &str = "tt.parquet";
pub const TT_PART_ROW_GROUP_ROWS: usize = 1_048_576;
pub const TT_PART_MAX_ROW_GROUPS: usize = 64;
pub const TT_PART_MAX_ROWS: usize = TT_PART_ROW_GROUP_ROWS * TT_PART_MAX_ROW_GROUPS;

pub fn part_name(index: usize) -> String {
    format!("part-{index:05}.parquet")
}

/// Return the exact Pass-A dataset parts in numeric order.
///
/// The dataset contract is strict: `path` is a directory containing only
/// contiguous, canonically named `part-NNNNN.parquet` regular files.
pub fn ordered_tt_dataset_parts(path: &Path) -> TTGenResult<Vec<PathBuf>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(TTGenError::InvalidPlan(format!(
            "training table path must be a non-symlink directory: {}",
            path.display()
        )));
    }

    let mut indexed = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        let entry_metadata = std::fs::symlink_metadata(&entry_path)?;
        let name = entry.file_name().into_string().map_err(|_| {
            TTGenError::InvalidPlan("non-UTF8 training table part name".to_string())
        })?;
        let index = name
            .strip_prefix("part-")
            .and_then(|value| value.strip_suffix(".parquet"))
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|index| part_name(*index) == name)
            .ok_or_else(|| {
                TTGenError::InvalidPlan(format!(
                    "unexpected training table dataset entry: {}",
                    entry_path.display()
                ))
            })?;
        if entry_metadata.file_type().is_symlink() || !entry_metadata.is_file() {
            return Err(TTGenError::InvalidPlan(format!(
                "training table part must be a regular file: {}",
                entry_path.display()
            )));
        }
        indexed.push((index, entry_path));
    }
    indexed.sort_by_key(|(index, _)| *index);
    for (expected, (actual, _)) in indexed.iter().enumerate() {
        if *actual != expected {
            return Err(TTGenError::InvalidPlan(format!(
                "training table parts must be contiguous from part-00000.parquet; expected part {} but found {}",
                expected, actual
            )));
        }
    }
    Ok(indexed.into_iter().map(|(_, path)| path).collect())
}

pub fn first_tt_dataset_schema(path: &Path) -> TTGenResult<arrow_schema::SchemaRef> {
    let first = ordered_tt_dataset_parts(path)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            TTGenError::InvalidPlan(format!("empty training table dataset: {}", path.display()))
        })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(first)?)?;
    Ok(builder.schema().clone())
}

/// Shared incremental writer for generated TT and task anchor datasets.
/// Rotation bounds both rows per file and the number of footer row groups.
pub struct ParquetDatasetWriter {
    path: PathBuf,
    maximum: usize,
    part: usize,
    rows_in_part: usize,
    total: u64,
    writer: Option<ArrowWriter<File>>,
}
impl ParquetDatasetWriter {
    pub fn new(path: &Path, maximum: usize) -> TTGenResult<Self> {
        if maximum == 0 {
            return Err(TTGenError::InvalidPlan(
                "max rows per part must be positive".into(),
            ));
        }
        std::fs::create_dir(path)?;
        Ok(Self {
            path: path.into(),
            maximum,
            part: 0,
            rows_in_part: 0,
            total: 0,
            writer: None,
        })
    }
    pub fn write(&mut self, batch: &RecordBatch) -> TTGenResult<()> {
        // Parquet has no seconds timestamp logical type. Preserve time values
        // by promoting that physical representation, as PyArrow's writer does.
        let fields = batch
            .schema()
            .fields()
            .iter()
            .map(|field| {
                if let arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Second, zone) =
                    field.data_type()
                {
                    std::sync::Arc::new(field.as_ref().clone().with_data_type(
                        arrow_schema::DataType::Timestamp(
                            arrow_schema::TimeUnit::Millisecond,
                            zone.clone(),
                        ),
                    ))
                } else {
                    field.clone()
                }
            })
            .collect::<Vec<_>>();
        let schema = std::sync::Arc::new(arrow_schema::Schema::new_with_metadata(
            fields,
            batch.schema().metadata().clone(),
        ));
        let columns = batch
            .columns()
            .iter()
            .zip(schema.fields())
            .map(|(column, field)| arrow::compute::cast(column, field.data_type()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| TTGenError::SystemError(e.to_string()))?;
        let batch = RecordBatch::try_new(schema, columns)
            .map_err(|e| TTGenError::SystemError(e.to_string()))?;
        let mut offset = 0;
        // Empty datasets still carry their schema and anchor binding.
        loop {
            if self.rows_in_part == self.maximum {
                self.writer.take().expect("nonempty part").close()?;
                self.part += 1;
                self.rows_in_part = 0;
            }
            if self.writer.is_none() {
                let properties = WriterProperties::builder()
                    .set_max_row_group_row_count(Some(TT_PART_ROW_GROUP_ROWS))
                    .build();
                self.writer = Some(ArrowWriter::try_new(
                    File::create(self.path.join(part_name(self.part)))?,
                    batch.schema(),
                    Some(properties),
                )?);
            }
            let rows = (batch.num_rows() - offset).min(self.maximum - self.rows_in_part);
            self.writer
                .as_mut()
                .expect("initialized writer")
                .write(&batch.slice(offset, rows))?;
            self.rows_in_part += rows;
            self.total += rows as u64;
            offset += rows;
            if offset == batch.num_rows() {
                return Ok(());
            }
        }
    }
    pub fn close(mut self) -> TTGenResult<u64> {
        if let Some(writer) = self.writer.take() {
            writer.close()?;
        }
        Ok(self.total)
    }
}
