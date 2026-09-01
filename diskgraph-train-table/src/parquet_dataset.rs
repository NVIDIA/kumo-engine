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
