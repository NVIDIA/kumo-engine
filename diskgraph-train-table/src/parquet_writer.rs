use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use arrow_array::builder::{
    BinaryBuilder, BooleanBuilder, Float64Builder, Int64Builder, ListBuilder, StringBuilder,
    UInt64Builder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::exec::{TargetValue, Value};
use crate::split::SplitLabel;

#[derive(Debug, Clone)]
pub struct TrainTableRow {
    pub entity: Value,
    pub target: TargetValue,
    pub timestamp: i64,
    pub weight: Option<f64>,
    pub split: SplitLabel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarType {
    Int64,
    UInt64,
    Bool,
    Utf8,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetType {
    Number,
    Count,
    List(ScalarType),
    Scalar(ScalarType),
}

pub fn write_train_table(path: impl AsRef<Path>, rows: &[TrainTableRow]) -> Result<()> {
    if rows.is_empty() {
        return Err(anyhow!("no rows to write"));
    }
    let entity_type = infer_entity_type(rows)?;
    let target_type = infer_target_type(rows)?;
    let include_weight = rows.iter().any(|row| row.weight.is_some());

    let (columns, schema) = build_columns(rows, entity_type, target_type, include_weight)?;
    let batch = RecordBatch::try_new(schema.clone(), columns)?;

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn infer_entity_type(rows: &[TrainTableRow]) -> Result<ScalarType> {
    match rows[0].entity {
        Value::Int(_) => Ok(ScalarType::Int64),
        Value::UInt(_) => Ok(ScalarType::UInt64),
        Value::Bool(_) => Ok(ScalarType::Bool),
        Value::String(_) => Ok(ScalarType::Utf8),
        Value::Bytes(_) => Ok(ScalarType::Binary),
    }
}

fn infer_target_type(rows: &[TrainTableRow]) -> Result<TargetType> {
    let mut list_type: Option<ScalarType> = None;
    let mut target_kind: Option<TargetType> = None;
    for row in rows {
        let kind = match &row.target {
            TargetValue::Number(_) => TargetType::Number,
            TargetValue::Count(_) => TargetType::Count,
            TargetValue::List(values) => {
                if list_type.is_none() {
                    list_type = infer_list_value_type(values);
                }
                TargetType::List(list_type.unwrap_or(ScalarType::Utf8))
            }
            TargetValue::Scalar(value) => {
                TargetType::Scalar(infer_scalar_type(value).unwrap_or(ScalarType::Utf8))
            }
        };
        match target_kind {
            None => target_kind = Some(kind),
            Some(existing) if existing == kind => {}
            Some(_) => return Err(anyhow!("mixed target types in rows")),
        }
    }
    target_kind.ok_or_else(|| anyhow!("unable to infer target type"))
}

fn infer_list_value_type(values: &[Value]) -> Option<ScalarType> {
    if let Some(value) = values.iter().next() {
        return Some(match value {
            Value::Int(_) => ScalarType::Int64,
            Value::UInt(_) => ScalarType::UInt64,
            Value::Bool(_) => ScalarType::Bool,
            Value::String(_) => ScalarType::Utf8,
            Value::Bytes(_) => ScalarType::Binary,
        });
    }
    None
}

fn infer_scalar_type(value: &Value) -> Option<ScalarType> {
    Some(match value {
        Value::Int(_) => ScalarType::Int64,
        Value::UInt(_) => ScalarType::UInt64,
        Value::Bool(_) => ScalarType::Bool,
        Value::String(_) => ScalarType::Utf8,
        Value::Bytes(_) => ScalarType::Binary,
    })
}

fn build_columns(
    rows: &[TrainTableRow],
    entity_type: ScalarType,
    target_type: TargetType,
    include_weight: bool,
) -> Result<(Vec<ArrayRef>, Arc<Schema>)> {
    let mut columns = Vec::new();
    let mut fields = Vec::new();

    columns.push(build_entity_array(rows, entity_type)?);
    fields.push(Field::new(
        "entity",
        scalar_type_to_datatype(entity_type),
        false,
    ));

    columns.push(build_target_array(rows, target_type)?);
    fields.push(Field::new(
        "target",
        target_type_to_datatype(target_type),
        false,
    ));

    columns.push(build_timestamp_array(rows));
    fields.push(Field::new("timestamp", DataType::Int64, false));

    if include_weight {
        columns.push(build_weight_array(rows));
        fields.push(Field::new("weight", DataType::Float64, true));
    }

    columns.push(build_split_array(rows));
    fields.push(Field::new("split", DataType::Utf8, false));

    Ok((columns, Arc::new(Schema::new(fields))))
}

fn build_entity_array(rows: &[TrainTableRow], scalar_type: ScalarType) -> Result<ArrayRef> {
    Ok(match scalar_type {
        ScalarType::Int64 => {
            let mut builder = Int64Builder::new();
            for row in rows {
                if let Value::Int(v) = row.entity {
                    builder.append_value(v);
                } else {
                    return Err(anyhow!("entity type mismatch"));
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::UInt64 => {
            let mut builder = UInt64Builder::new();
            for row in rows {
                if let Value::UInt(v) = row.entity {
                    builder.append_value(v);
                } else {
                    return Err(anyhow!("entity type mismatch"));
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::Bool => {
            let mut builder = BooleanBuilder::new();
            for row in rows {
                if let Value::Bool(v) = row.entity {
                    builder.append_value(v);
                } else {
                    return Err(anyhow!("entity type mismatch"));
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::Utf8 => {
            let mut builder = StringBuilder::new();
            for row in rows {
                if let Value::String(ref v) = row.entity {
                    builder.append_value(v);
                } else {
                    return Err(anyhow!("entity type mismatch"));
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::Binary => {
            let mut builder = BinaryBuilder::new();
            for row in rows {
                if let Value::Bytes(ref v) = row.entity {
                    builder.append_value(v);
                } else {
                    return Err(anyhow!("entity type mismatch"));
                }
            }
            Arc::new(builder.finish())
        }
    })
}

fn build_target_array(rows: &[TrainTableRow], target_type: TargetType) -> Result<ArrayRef> {
    Ok(match target_type {
        TargetType::Number => {
            let mut builder = Float64Builder::new();
            for row in rows {
                if let TargetValue::Number(v) = row.target {
                    builder.append_value(v);
                } else {
                    return Err(anyhow!("target type mismatch"));
                }
            }
            Arc::new(builder.finish())
        }
        TargetType::Count => {
            let mut builder = UInt64Builder::new();
            for row in rows {
                if let TargetValue::Count(v) = row.target {
                    builder.append_value(v);
                } else {
                    return Err(anyhow!("target type mismatch"));
                }
            }
            Arc::new(builder.finish())
        }
        TargetType::List(scalar_type) => build_list_array(rows, scalar_type)?,
        TargetType::Scalar(scalar_type) => build_scalar_target_array(rows, scalar_type)?,
    })
}

fn build_scalar_target_array(rows: &[TrainTableRow], scalar_type: ScalarType) -> Result<ArrayRef> {
    Ok(match scalar_type {
        ScalarType::Int64 => {
            let mut builder = Int64Builder::new();
            for row in rows {
                match &row.target {
                    TargetValue::Scalar(Value::Int(v)) => builder.append_value(*v),
                    _ => return Err(anyhow!("target type mismatch")),
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::UInt64 => {
            let mut builder = UInt64Builder::new();
            for row in rows {
                match &row.target {
                    TargetValue::Scalar(Value::UInt(v)) => builder.append_value(*v),
                    _ => return Err(anyhow!("target type mismatch")),
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::Bool => {
            let mut builder = BooleanBuilder::new();
            for row in rows {
                match &row.target {
                    TargetValue::Scalar(Value::Bool(v)) => builder.append_value(*v),
                    _ => return Err(anyhow!("target type mismatch")),
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::Utf8 => {
            let mut builder = StringBuilder::new();
            for row in rows {
                match &row.target {
                    TargetValue::Scalar(Value::String(v)) => builder.append_value(v),
                    _ => return Err(anyhow!("target type mismatch")),
                }
            }
            Arc::new(builder.finish())
        }
        ScalarType::Binary => {
            let mut builder = BinaryBuilder::new();
            for row in rows {
                match &row.target {
                    TargetValue::Scalar(Value::Bytes(v)) => builder.append_value(v),
                    _ => return Err(anyhow!("target type mismatch")),
                }
            }
            Arc::new(builder.finish())
        }
    })
}

fn build_list_array(rows: &[TrainTableRow], scalar_type: ScalarType) -> Result<ArrayRef> {
    match scalar_type {
        ScalarType::Int64 => {
            build_list_with_builder(rows, ListBuilder::new(Int64Builder::new()), scalar_type)
        }
        ScalarType::UInt64 => {
            build_list_with_builder(rows, ListBuilder::new(UInt64Builder::new()), scalar_type)
        }
        ScalarType::Bool => {
            build_list_with_builder(rows, ListBuilder::new(BooleanBuilder::new()), scalar_type)
        }
        ScalarType::Utf8 => {
            build_list_with_builder(rows, ListBuilder::new(StringBuilder::new()), scalar_type)
        }
        ScalarType::Binary => {
            build_list_with_builder(rows, ListBuilder::new(BinaryBuilder::new()), scalar_type)
        }
    }
}

fn build_list_with_builder<T>(
    rows: &[TrainTableRow],
    mut builder: ListBuilder<T>,
    scalar_type: ScalarType,
) -> Result<ArrayRef>
where
    T: arrow_array::builder::ArrayBuilder,
{
    for row in rows {
        match &row.target {
            TargetValue::List(values) => {
                for value in values {
                    append_list_value(&mut builder, scalar_type, value)?;
                }
                builder.append(true);
            }
            _ => return Err(anyhow!("target type mismatch")),
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn append_list_value<T: arrow_array::builder::ArrayBuilder>(
    builder: &mut ListBuilder<T>,
    scalar_type: ScalarType,
    value: &Value,
) -> Result<()> {
    match scalar_type {
        ScalarType::Int64 => {
            let inner = builder
                .values()
                .as_any_mut()
                .downcast_mut::<Int64Builder>()
                .unwrap();
            if let Value::Int(v) = value {
                inner.append_value(*v);
            } else {
                return Err(anyhow!("list value type mismatch"));
            }
        }
        ScalarType::UInt64 => {
            let inner = builder
                .values()
                .as_any_mut()
                .downcast_mut::<UInt64Builder>()
                .unwrap();
            if let Value::UInt(v) = value {
                inner.append_value(*v);
            } else {
                return Err(anyhow!("list value type mismatch"));
            }
        }
        ScalarType::Bool => {
            let inner = builder
                .values()
                .as_any_mut()
                .downcast_mut::<BooleanBuilder>()
                .unwrap();
            if let Value::Bool(v) = value {
                inner.append_value(*v);
            } else {
                return Err(anyhow!("list value type mismatch"));
            }
        }
        ScalarType::Utf8 => {
            let inner = builder
                .values()
                .as_any_mut()
                .downcast_mut::<StringBuilder>()
                .unwrap();
            if let Value::String(v) = value {
                inner.append_value(v);
            } else {
                return Err(anyhow!("list value type mismatch"));
            }
        }
        ScalarType::Binary => {
            let inner = builder
                .values()
                .as_any_mut()
                .downcast_mut::<BinaryBuilder>()
                .unwrap();
            if let Value::Bytes(v) = value {
                inner.append_value(v);
            } else {
                return Err(anyhow!("list value type mismatch"));
            }
        }
    }
    Ok(())
}

fn build_timestamp_array(rows: &[TrainTableRow]) -> ArrayRef {
    let mut builder = Int64Builder::new();
    for row in rows {
        builder.append_value(row.timestamp);
    }
    Arc::new(builder.finish())
}

fn build_weight_array(rows: &[TrainTableRow]) -> ArrayRef {
    let mut builder = Float64Builder::new();
    for row in rows {
        if let Some(weight) = row.weight {
            builder.append_value(weight);
        } else {
            builder.append_null();
        }
    }
    Arc::new(builder.finish())
}

fn build_split_array(rows: &[TrainTableRow]) -> ArrayRef {
    let mut builder = StringBuilder::new();
    for row in rows {
        builder.append_value(row.split.as_str());
    }
    Arc::new(builder.finish())
}

fn scalar_type_to_datatype(scalar_type: ScalarType) -> DataType {
    match scalar_type {
        ScalarType::Int64 => DataType::Int64,
        ScalarType::UInt64 => DataType::UInt64,
        ScalarType::Bool => DataType::Boolean,
        ScalarType::Utf8 => DataType::Utf8,
        ScalarType::Binary => DataType::Binary,
    }
}

fn target_type_to_datatype(target_type: TargetType) -> DataType {
    match target_type {
        TargetType::Number => DataType::Float64,
        TargetType::Count => DataType::UInt64,
        TargetType::List(scalar_type) => DataType::List(Arc::new(Field::new(
            "item",
            scalar_type_to_datatype(scalar_type),
            true,
        ))),
        TargetType::Scalar(scalar_type) => scalar_type_to_datatype(scalar_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    #[test]
    fn writes_basic_parquet() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("train.parquet");
        let rows = vec![
            TrainTableRow {
                entity: Value::Int(1),
                target: TargetValue::Number(2.0),
                timestamp: 10,
                weight: None,
                split: SplitLabel::Train,
            },
            TrainTableRow {
                entity: Value::Int(2),
                target: TargetValue::Number(3.0),
                timestamp: 20,
                weight: None,
                split: SplitLabel::Val,
            },
        ];
        write_train_table(&path, &rows).unwrap();

        let file = File::open(&path).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema().fields().len(), 4);
        assert_eq!(batch.schema().field(0).name(), "entity");
        assert_eq!(batch.schema().field(1).name(), "target");
        assert_eq!(batch.schema().field(2).name(), "timestamp");
        assert_eq!(batch.schema().field(3).name(), "split");
    }

    #[test]
    fn writes_list_and_weight() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("train.parquet");
        let rows = vec![
            TrainTableRow {
                entity: Value::String("u1".to_string()),
                target: TargetValue::List(vec![Value::String("a".to_string())]),
                timestamp: 10,
                weight: Some(0.5),
                split: SplitLabel::Train,
            },
            TrainTableRow {
                entity: Value::String("u2".to_string()),
                target: TargetValue::List(vec![Value::String("b".to_string())]),
                timestamp: 20,
                weight: Some(0.9),
                split: SplitLabel::Test,
            },
        ];
        write_train_table(&path, &rows).unwrap();

        let file = File::open(&path).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema().fields().len(), 5);
        assert_eq!(batch.schema().field(3).name(), "weight");
        assert_eq!(batch.schema().field(4).name(), "split");
    }

    #[test]
    fn writes_scalar_target() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("train.parquet");
        let rows = vec![
            TrainTableRow {
                entity: Value::UInt(1),
                target: TargetValue::Scalar(Value::Bool(true)),
                timestamp: 10,
                weight: None,
                split: SplitLabel::Train,
            },
            TrainTableRow {
                entity: Value::UInt(2),
                target: TargetValue::Scalar(Value::Bool(false)),
                timestamp: 20,
                weight: None,
                split: SplitLabel::Val,
            },
        ];
        write_train_table(&path, &rows).unwrap();

        let file = File::open(&path).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema().fields().len(), 4);
        assert_eq!(batch.schema().field(1).data_type(), &DataType::Boolean);
    }
}
