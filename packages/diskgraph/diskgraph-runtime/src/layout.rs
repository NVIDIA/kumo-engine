// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve positional Arrow references without copying payload buffers.
use anyhow::{ensure, Context};
use arrow_array::{ArrayRef, RecordBatch};
use serde_json::Value;
use std::ops::Range;

/// An array or record batch selected from a received component.
pub enum ArrowValue {
    Array(ArrayRef),
    RecordBatch(RecordBatch),
}

enum Selection {
    Column(usize),
    Columns(Range<usize>),
}

fn index(value: &Value) -> anyhow::Result<usize> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .context("Invalid Arrow component or column index")
}

fn reference<'a>(
    value: &Value,
    components: &'a [RecordBatch],
) -> anyhow::Result<Option<(&'a RecordBatch, Selection)>> {
    let Some(values) = value.as_object() else {
        return Ok(None);
    };
    let Some(component) = values.get("component") else {
        return Ok(None);
    };
    // Node/namespace dictionaries may themselves contain a type named component.
    if component.is_object() && values.values().all(Value::is_object) {
        return Ok(None);
    }
    ensure!(
        values.len() == 2 && (values.contains_key("column") ^ values.contains_key("columns")),
        "Invalid Arrow component reference"
    );
    let batch = components
        .get(index(component)?)
        .context("Invalid Arrow component reference")?;
    let selection = if let Some(column) = values.get("column") {
        let column = index(column)?;
        ensure!(
            column < batch.num_columns(),
            "Arrow column index out of bounds"
        );
        Selection::Column(column)
    } else {
        let columns = values["columns"]
            .as_array()
            .context("Invalid Arrow column range")?;
        ensure!(columns.len() == 2, "Invalid Arrow column range");
        let start = index(&columns[0])?;
        let end = index(&columns[1])?;
        ensure!(
            start <= end && end <= batch.num_columns(),
            "Arrow column range out of bounds"
        );
        Selection::Columns(start..end)
    };
    Ok(Some((batch, selection)))
}

/// Resolve a column as an array or a half-open column range as a record batch.
///
/// Returns None for ordinary layout containers and scalars. Invalid references
/// return an error; selected arrays share their original Arrow buffers.
pub fn resolve_reference(
    value: &Value,
    components: &[RecordBatch],
) -> anyhow::Result<Option<ArrowValue>> {
    reference(value, components)?
        .map(|(batch, selection)| match selection {
            Selection::Column(column) => Ok(ArrowValue::Array(batch.column(column).clone())),
            Selection::Columns(columns) => Ok(ArrowValue::RecordBatch(
                batch.project(&columns.collect::<Vec<_>>())?,
            )),
        })
        .transpose()
}

/// Validate every positional Arrow reference in a layout, including nested containers.
///
/// Returns an error for invalid component IDs, columns or half-open ranges.
pub fn validate_layout(value: &Value, components: &[RecordBatch]) -> anyhow::Result<()> {
    if reference(value, components)?.is_some() {
        return Ok(());
    }
    match value {
        Value::Object(values) => {
            for value in values.values() {
                validate_layout(value, components)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                validate_layout(value, components)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use serde_json::json;
    use std::sync::Arc;

    fn batch() -> RecordBatch {
        RecordBatch::try_from_iter([
            (
                "TARGET1",
                Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            ),
            (
                "TARGET2",
                Arc::new(StringArray::from(vec![Some("yes"), None])) as ArrayRef,
            ),
        ])
        .unwrap()
        .with_schema(Arc::new(arrow_schema::Schema::new_with_metadata(
            vec![
                arrow_schema::Field::new("TARGET1", arrow_schema::DataType::Int64, false),
                arrow_schema::Field::new("TARGET2", arrow_schema::DataType::Utf8, true),
            ],
            [("origin".into(), "targets".into())].into(),
        )))
        .unwrap()
    }
    #[test]
    fn ranges_preserve_column_order_metadata_and_buffers() {
        let components = [batch()];
        let Some(ArrowValue::RecordBatch(selected)) =
            resolve_reference(&json!({"component":0,"columns":[0,2]}), &components).unwrap()
        else {
            panic!("Expected record batch")
        };
        assert_eq!(selected, components[0]);
        assert_eq!(selected.schema(), components[0].schema());
        assert!(Arc::ptr_eq(selected.column(0), components[0].column(0)));
        let Some(ArrowValue::Array(selected)) =
            resolve_reference(&json!({"component":0,"column":1}), &components).unwrap()
        else {
            panic!("Expected array")
        };
        assert!(Arc::ptr_eq(&selected, components[0].column(1)));
        let Some(ArrowValue::RecordBatch(empty)) =
            resolve_reference(&json!({"component":0,"columns":[1,1]}), &components).unwrap()
        else {
            panic!("Expected empty projection")
        };
        assert_eq!((empty.num_rows(), empty.num_columns()), (2, 0));
    }
    #[test]
    fn invalid_references_fail_before_export() {
        for value in [
            json!({"component":1,"column":0}),
            json!({"component":{},"columns":[0,1]}),
            json!({"component":null,"column":0}),
            json!({"component":-1,"column":0}),
            json!({"component":false,"column":0}),
            json!({"component":0,"column":2}),
            json!({"component":0,"columns":[0,3]}),
            json!({"component":0,"columns":[2,1]}),
            json!({"component":0,"columns":[0]}),
            json!({"component":0,"columns":[true,1]}),
            json!({"component":0,"columns":"all"}),
            json!({"component":0,"column":0,"as":"record_batch"}),
        ] {
            assert!(validate_layout(&value, &[batch()]).is_err(), "{value}");
        }
    }
    #[test]
    fn reference_field_names_can_be_node_type_names() {
        let marker = json!({"component":0,"column":0});
        validate_layout(
            &json!({"component":marker,"column":marker,"columns":marker}),
            &[batch()],
        )
        .unwrap();
        validate_layout(&json!({"component":marker,"columns":marker}), &[batch()]).unwrap();
    }
}
