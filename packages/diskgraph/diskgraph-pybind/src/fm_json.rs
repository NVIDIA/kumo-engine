// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! FM's normalized scalar Arrow rows, encoded without Python cell objects.
use std::fmt::{self, Write};

use arrow::array::{
    Array, BooleanArray, Date32Array, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::pyarrow::FromPyArrow;
use arrow::record_batch::RecordBatch;
use chrono::{Datelike, NaiveDate};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[derive(Debug)]
enum EncodeError {
    Budget,
    Unsupported(DataType),
    NonFinite,
    Calendar,
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget => f.write_str("Payload exceeds byte budget; reduce the workload batch"),
            Self::Unsupported(dtype) => write!(f, "Unsupported native arrays Arrow type: {dtype}"),
            Self::NonFinite => f.write_str("Out of range float values are not JSON compliant"),
            Self::Calendar => f.write_str("Arrow timestamp outside calendar range"),
        }
    }
}

struct Writer {
    data: String,
    limit: usize,
}

// Formatting writes directly into the bounded output, including escaped strings.
impl Write for Writer {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        if value.len() > self.limit - self.data.len() {
            return Err(fmt::Error);
        }
        self.data.push_str(value);
        Ok(())
    }
}

impl From<fmt::Error> for EncodeError {
    fn from(_: fmt::Error) -> Self {
        Self::Budget
    }
}

impl Writer {
    fn string(&mut self, value: &str) -> Result<(), EncodeError> {
        self.write_char('"')?;
        for ch in value.chars() {
            match ch {
                '"' => self.write_str("\\\"")?,
                '\\' => self.write_str("\\\\")?,
                '\u{8}' => self.write_str("\\b")?,
                '\u{c}' => self.write_str("\\f")?,
                '\n' => self.write_str("\\n")?,
                '\r' => self.write_str("\\r")?,
                '\t' => self.write_str("\\t")?,
                ' '..='~' => self.write_char(ch)?,
                _ => {
                    for unit in ch.encode_utf16(&mut [0; 2]) {
                        write!(self, "\\u{unit:04x}")?;
                    }
                }
            }
        }
        self.write_char('"')?;
        Ok(())
    }

    fn float(&mut self, value: f64) -> Result<(), EncodeError> {
        // serde_json's shortest-roundtrip formatter rounds decimal ties to even,
        // as CPython does. Normalize its fixed/scientific layout to json.dumps.
        let text = serde_json::to_string(&value).expect("finite float serialization");
        let unsigned = text.strip_prefix('-').unwrap_or(&text);
        let (mantissa, exponent) = unsigned.split_once('e').unwrap_or((unsigned, "0"));
        let exponent: i32 = exponent.parse().expect("float exponent");
        let point = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
        let digits = mantissa.replace('.', "");
        let leading = digits.len() - digits.trim_start_matches('0').len();
        let digits = digits.trim_start_matches('0').trim_end_matches('0');
        if value.is_sign_negative() {
            self.write_char('-')?;
        }
        if digits.is_empty() {
            self.write_str("0.0")?;
            return Ok(());
        }
        let exponent = exponent + point - leading as i32 - 1;
        if !(-4..16).contains(&exponent) {
            self.write_str(&digits[..1])?;
            if digits.len() > 1 {
                self.write_char('.')?;
                self.write_str(&digits[1..])?;
            }
            write!(self, "e{exponent:+03}")?;
        } else {
            let point = exponent + 1;
            if point <= 0 {
                self.write_str("0.")?;
                for _ in 0..-point {
                    self.write_char('0')?;
                }
                self.write_str(digits)?;
            } else if point as usize >= digits.len() {
                self.write_str(digits)?;
                for _ in digits.len()..point as usize {
                    self.write_char('0')?;
                }
                self.write_str(".0")?;
            } else {
                self.write_str(&digits[..point as usize])?;
                self.write_char('.')?;
                self.write_str(&digits[point as usize..])?;
            }
        }
        Ok(())
    }

    fn cell(&mut self, column: &Column<'_>, row: usize) -> Result<(), EncodeError> {
        if column.array().is_null(row) {
            self.write_str("null")?;
            return Ok(());
        }
        match column {
            Column::Int(a) => {
                let n = a.value(row);
                if !(-9_007_199_254_740_991..=9_007_199_254_740_991).contains(&n) {
                    write!(self, "\"{n}\"")?;
                } else {
                    write!(self, "{n}")?;
                }
            }
            Column::Float(a) => {
                let n = a.value(row);
                if !n.is_finite() {
                    return Err(EncodeError::NonFinite);
                }
                self.float(n)?;
            }
            Column::Bool(a) => self.write_str(if a.value(row) { "true" } else { "false" })?,
            Column::String(a) => self.string(a.value(row))?,
            Column::Date(a) => {
                let days = a.value(row).checked_add(719_163).ok_or(EncodeError::Calendar)?;
                let date = NaiveDate::from_num_days_from_ce_opt(days).ok_or(EncodeError::Calendar)?;
                write!(
                    self,
                    "\"{:04}-{:02}-{:02}\"",
                    date.year(),
                    date.month(),
                    date.day()
                )?;
            }
            Column::Timestamp(a) => {
                let date = chrono::DateTime::from_timestamp_micros(a.value(row))
                    .ok_or(EncodeError::Calendar)?;
                write!(self, "\"{}Z\"", date.format("%Y-%m-%dT%H:%M:%S%.6f"))?;
            }
        }
        Ok(())
    }
}

// Resolve types once per batch rather than downcasting each cell.
enum Column<'a> {
    Int(&'a Int64Array),
    Float(&'a Float64Array),
    Bool(&'a BooleanArray),
    String(&'a StringArray),
    Date(&'a Date32Array),
    Timestamp(&'a TimestampMicrosecondArray),
}

impl<'a> Column<'a> {
    fn new(a: &'a dyn Array) -> Result<Self, EncodeError> {
        Ok(match a.data_type() {
            DataType::Int64 => Self::Int(a.as_any().downcast_ref().unwrap()),
            DataType::Float64 => Self::Float(a.as_any().downcast_ref().unwrap()),
            DataType::Boolean => Self::Bool(a.as_any().downcast_ref().unwrap()),
            DataType::Utf8 => Self::String(a.as_any().downcast_ref().unwrap()),
            DataType::Date32 => Self::Date(a.as_any().downcast_ref().unwrap()),
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) if tz.as_ref() == "UTC" => {
                Self::Timestamp(a.as_any().downcast_ref().unwrap())
            }
            other => return Err(EncodeError::Unsupported(other.clone())),
        })
    }
    fn array(&self) -> &dyn Array {
        match self {
            Self::Int(a) => *a,
            Self::Float(a) => *a,
            Self::Bool(a) => *a,
            Self::String(a) => *a,
            Self::Date(a) => *a,
            Self::Timestamp(a) => *a,
        }
    }
}

fn encode(batches: &[RecordBatch], limit: usize) -> Result<Vec<u8>, EncodeError> {
    let mut out = Writer {
        data: String::new(),
        limit,
    };
    out.write_char('[')?;
    let mut first = true;
    for batch in batches {
        let columns = batch
            .columns()
            .iter()
            .map(|a| Column::new(a.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        // Match zip(*columns) for the zero-column case.
        let row_count = if columns.is_empty() { 0 } else { batch.num_rows() };
        for row in 0..row_count {
            if !first {
                out.write_char(',')?;
            }
            first = false;
            out.write_char('[')?;
            for (index, column) in columns.iter().enumerate() {
                if index != 0 {
                    out.write_char(',')?;
                }
                out.cell(column, row)?;
            }
            out.write_char(']')?;
        }
    }
    out.write_char(']')?;
    Ok(out.data.into_bytes())
}

/// Encode normalized FM Arrow record batches as bounded compact ASCII JSON.
#[pyfunction]
pub fn encode_fm_rows<'py>(
    py: Python<'py>,
    batches: Vec<Bound<'py, PyAny>>,
    max_bytes: i64,
) -> PyResult<Bound<'py, PyBytes>> {
    let limit = usize::try_from(max_bytes)
        .map_err(|_| crate::DiskgraphConfigError::new_err("Negative payload byte budget"))?;
    // Import all Arrow data while attached. No Python iterator or stream callback
    // runs in the detached section, and Arrow buffers stay owned until it ends.
    let batches = batches
        .iter()
        .map(RecordBatch::from_pyarrow_bound)
        .collect::<PyResult<Vec<_>>>()?;
    let data = py
        .detach(|| encode(&batches, limit))
        .map_err(|e| crate::DiskgraphConfigError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &data))
}
