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
    pub tables: Vec<TableConfig>,
    #[serde(default)]
    pub relationships: Vec<RelationshipConfig>,
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
