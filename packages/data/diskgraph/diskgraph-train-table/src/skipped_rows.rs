// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A prediction source's skipped rows, streamed to one Parquet part per worker.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::custom_tt_row_errors::RowFailureKind;
use crate::errors::{TTGenError, TTGenResult};

pub const SKIPPED_ROWS_DIR: &str = "skipped";

const ROWS_PER_FLUSH: usize = 64 * 1024;
const COLUMNS: [&str; 2] = ["ENTITY1", "TIMESTAMP"];

pub struct SkippedRowsWriter {
    path: PathBuf,
    writer: Option<ArrowWriter<File>>,
}

impl SkippedRowsWriter {
    pub fn new(dir: &Path, part: &str) -> Self {
        Self {
            path: dir.join(format!("part-{part}.parquet")),
            writer: None,
        }
    }

    pub fn write(
        &mut self,
        batch: &RecordBatch,
        row_base: u64,
        skips: &[(u64, Vec<RowFailureKind>)],
    ) -> TTGenResult<()> {
        if skips.is_empty() {
            return Ok(());
        }
        let rows = UInt64Array::from_iter_values(skips.iter().map(|(row, _)| row - row_base));
        let mut fields = Vec::new();
        let mut columns: Vec<ArrayRef> = Vec::new();
        for name in COLUMNS {
            if let Some(column) = batch.column_by_name(name) {
                columns.push(arrow_select::take::take(column.as_ref(), &rows, None).map_err(failed)?);
                fields.push(Field::new(name, column.data_type().clone(), true));
            }
        }
        columns.push(Arc::new(StringArray::from_iter_values(skips.iter().map(
            |(_, kinds)| {
                kinds
                    .iter()
                    .map(|kind| kind.name())
                    .collect::<Vec<_>>()
                    .join(",")
            },
        ))));
        fields.push(Field::new("condition", arrow_schema::DataType::Utf8, false));
        let part = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(failed)?;
        let writer = match &mut self.writer {
            Some(writer) => writer,
            None => {
                std::fs::create_dir_all(self.path.parent().unwrap_or(Path::new(".")))?;
                let properties = WriterProperties::builder()
                    .set_max_row_group_row_count(Some(ROWS_PER_FLUSH))
                    .build();
                self.writer.insert(ArrowWriter::try_new(
                    File::create(&self.path)?,
                    part.schema(),
                    Some(properties),
                )?)
            }
        };
        writer.write(&part).map_err(failed)
    }

    pub fn finish(self) -> TTGenResult<()> {
        if let Some(writer) = self.writer {
            writer.close().map_err(failed)?;
        }
        Ok(())
    }
}

fn failed(error: impl std::fmt::Display) -> TTGenError {
    TTGenError::Parquet(format!("skipped rows: {error}"))
}
