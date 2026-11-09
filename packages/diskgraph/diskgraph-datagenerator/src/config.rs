// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetConfig {
    pub total_rows: u64,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_row_group_size")]
    pub row_group_size: usize,
    #[serde(default)]
    pub parquet_compression: ParquetCompression,
    pub tables: Vec<TableConfig>,
    #[serde(default)]
    pub relationships: Vec<RelationshipConfig>,
}

#[derive(Debug, Default, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParquetCompression {
    #[default]
    Uncompressed,
    Snappy,
    Zstd,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TableConfig {
    pub name: String,
    #[serde(default)]
    pub row_fraction: Option<f64>,
    #[serde(default)]
    pub row_percent: Option<f64>,
    #[serde(default)]
    pub rows: Option<u64>,
    pub shards: usize,
    pub columns: Vec<ColumnConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ColumnConfig {
    pub name: String,
    pub dtype: ColumnType,
    #[serde(default)]
    pub primary_key: bool,
    #[serde(default)]
    pub fixed_len: Option<usize>,
    #[serde(default)]
    pub key_encoding: Option<KeyEncoding>,
    #[serde(default)]
    pub timestamp_unit: Option<TimestampUnit>,
    #[serde(default)]
    pub null_ratio: Option<f64>,
    #[serde(default)]
    pub gen: Option<GeneratorSpec>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ColumnType {
    Int8,
    Int16,
    Int32,
    Int64,
    #[serde(alias = "uint8")]
    UInt8,
    #[serde(alias = "uint16")]
    UInt16,
    #[serde(alias = "uint32")]
    UInt32,
    #[serde(alias = "uint64")]
    UInt64,
    Float32,
    Float64,
    Bool,
    String,
    Bytes,
    Uuid,
    Uuid256,
    Ip,
    Date,
    Timestamp,
    Float32Array,
    Float64Array,
    Int8Array,
    Int16Array,
    Int32Array,
    Int64Array,
    StringArray,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TimestampUnit {
    Seconds,
    Millis,
    Micros,
    Nanos,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KeyEncoding {
    Decimal,
    Hex,
    U64Le,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RelationshipConfig {
    pub src_table: String,
    pub src_column: String,
    pub dst_table: String,
    pub dst_column: String,
    #[serde(default)]
    pub distribution: Option<FkDistribution>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FkDistribution {
    Uniform,
    Sequential {
        #[serde(default)]
        offset: u64,
    },
    Hotspot {
        hot_fraction: f64,
        hot_share: f64,
    },
    /// A deterministic broken-power-law degree distribution.
    ///
    /// `minimum_degree` first assigns every referenced key that many rows.
    /// Remaining rows follow the power-law segments by popularity rank. The
    /// rank-to-key permutation prevents popular keys from occupying one
    /// contiguous primary-key range, while generation remains stateless and
    /// can be split across arbitrary Parquet shards.
    Organic {
        #[serde(default)]
        minimum_degree: u64,
        permutation_seed: u64,
        segments: Vec<PowerLawSegment>,
    },
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct PowerLawSegment {
    /// Inclusive upper bound of this popularity-rank interval. A value of
    /// 0.5 means that residual edges are assigned only to the most popular
    /// half of the keys unless another segment extends the range.
    pub end_fraction: f64,
    /// Density within the interval is proportional to rank^-exponent.
    pub exponent: f64,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GeneratorSpec {
    Constant {
        value: serde_json::Value,
    },
    Sequence {
        #[serde(default)]
        start: i64,
        #[serde(default = "default_step")]
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
        #[serde(default = "default_true_ratio")]
        true_ratio: f64,
    },
    String {
        #[serde(default)]
        len: Option<usize>,
        #[serde(default)]
        min_len: Option<usize>,
        #[serde(default)]
        max_len: Option<usize>,
        #[serde(default)]
        alphabet: Option<String>,
        #[serde(default)]
        prefix: Option<String>,
    },
    Bytes {
        #[serde(default)]
        len: Option<usize>,
        #[serde(default)]
        min_len: Option<usize>,
        #[serde(default)]
        max_len: Option<usize>,
    },
    Uuid {
        #[serde(default)]
        mode: Option<String>,
        #[serde(default)]
        format: Option<String>,
    },
    Timestamp {
        start_epoch_ms: i64,
        #[serde(default = "default_step_ms")]
        step_ms: i64,
    },
    Date {
        start_ymd: String,
        #[serde(default = "default_step_days")]
        step_days: i32,
    },
    Categorical {
        values: Vec<String>,
        #[serde(default)]
        weights: Option<Vec<f64>>,
    },
    ArrayFloat32 {
        len: usize,
        min: f32,
        max: f32,
    },
    ArrayFloat64 {
        len: usize,
        min: f64,
        max: f64,
    },
    ArrayInt64 {
        len: usize,
        min: i64,
        max: i64,
    },
    ArrayString {
        len: usize,
        #[serde(default)]
        min_len: Option<usize>,
        #[serde(default)]
        max_len: Option<usize>,
        #[serde(default)]
        alphabet: Option<String>,
    },
}

fn default_batch_size() -> usize {
    100_000
}

fn default_row_group_size() -> usize {
    1_000_000
}

fn default_step() -> i64 {
    1
}

fn default_step_ms() -> i64 {
    1_000
}

fn default_step_days() -> i32 {
    1
}

fn default_true_ratio() -> f64 {
    0.5
}

#[cfg(test)]
mod tests {
    use super::{DatasetConfig, FkDistribution, ParquetCompression};

    #[test]
    fn parquet_compression_defaults_to_uncompressed() {
        let config: DatasetConfig = serde_json::from_str(r#"{"total_rows":0,"tables":[]}"#)
            .expect("minimal dataset config");
        assert_eq!(config.parquet_compression, ParquetCompression::Uncompressed);
    }

    #[test]
    fn parquet_compression_parses_zstd() {
        let config: DatasetConfig =
            serde_json::from_str(r#"{"total_rows":0,"parquet_compression":"zstd","tables":[]}"#)
                .expect("zstd dataset config");
        assert_eq!(config.parquet_compression, ParquetCompression::Zstd);
    }

    #[test]
    fn organic_relationship_distribution_parses() {
        let distribution: FkDistribution = serde_json::from_str(
            r#"{
                "type":"organic",
                "minimum_degree":1,
                "permutation_seed":7,
                "segments":[
                    {"end_fraction":0.01,"exponent":0.3},
                    {"end_fraction":0.5,"exponent":0.8}
                ]
            }"#,
        )
        .expect("organic distribution");
        let FkDistribution::Organic {
            minimum_degree,
            permutation_seed,
            segments,
        } = distribution
        else {
            panic!("expected organic distribution")
        };
        assert_eq!(minimum_degree, 1);
        assert_eq!(permutation_seed, 7);
        assert_eq!(segments.len(), 2);
    }
}
