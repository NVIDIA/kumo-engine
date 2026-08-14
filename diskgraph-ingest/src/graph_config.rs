use anyhow::{anyhow, ensure, Context, Result};
use diskgraph_core::metadata::EdgeStorageFormat;
use diskgraph_rowpack::RowPackConfig;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::coercion::{coercion_kind, validate_fk_pk, CoercionKind, CoercionMode, CoercionParams};

/// Filename for the graph config persisted to the index root at the end of
/// ingest. Carries per-column `semantic_type` so downstream consumers (TT-gen,
/// BP) can dispatch on stype without per-query plan JSON plumbing.
pub const DISKGRAPH_CONFIG_FILENAME: &str = "diskgraph_config.json";

/// Read a `GraphConfig` from `{index_root}/diskgraph_config.json`. Returns the
/// resolved post-ingest config (i.e., with `semantic_type` populated on every
/// column).
pub fn read_persisted_config(index_root: &Path) -> Result<GraphConfig> {
    let path = index_root.join(DISKGRAPH_CONFIG_FILENAME);
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let config: GraphConfig =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    Ok(config)
}

/// Build a `(table, column) → SemanticType` lookup from a resolved config.
/// Columns without an explicit `semantic_type` are skipped.
pub fn column_semantic_types(config: &GraphConfig) -> HashMap<(String, String), SemanticType> {
    let mut out = HashMap::new();
    for table in &config.tables {
        for col in &table.columns {
            if let Some(stype) = col.semantic_type {
                out.insert((table.name.clone(), col.name.clone()), stype);
            }
        }
    }
    out
}

fn source_dtype_is_decimal(dtype: Option<&str>) -> bool {
    let Some(dtype) = dtype else {
        return false;
    };
    let dtype = dtype.trim_start();
    dtype.starts_with("Decimal128")
        || dtype.starts_with("Decimal256")
        || dtype.starts_with("Decimal")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    pub dataset_name: String,
    pub tables: Vec<TableSpec>,
    #[serde(default)]
    pub rowpack: Option<RowPackConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_shards: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_format: Option<EdgeStorageFormat>,
    /// Coercion mode for FK-PK type compatibility.
    /// Default is Lossless (allows safe and conditional coercions).
    #[serde(default, skip_serializing_if = "is_default_coercion_mode")]
    pub coercion_mode: CoercionMode,
}

fn is_default_coercion_mode(mode: &CoercionMode) -> bool {
    *mode == CoercionMode::default()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSpec {
    pub name: String,
    pub columns: Vec<ColumnSpec>,
    #[serde(default)]
    pub primary_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub create_time: Option<String>,
    /// End time column (for temporal graphs with validity windows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub edge_table: Option<EdgeTableSpec>,
    #[serde(default)]
    pub rowpack: Option<RowPackConfig>,
    /// Data source configuration.
    pub source: DataSource,
    /// Total byte size of all source files, populated during path expansion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_size: Option<u64>,
    /// Foreign key relationships (edges embedded in table).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fkeys: Vec<FKeySpec>,
    /// Data version identifier (carried from config converter for kumo stats output).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_version: Option<String>,
    /// Kumo connector ID (carried from config converter for edge stats output).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kumo_connector_id: Option<String>,
    /// Kumo source table name (carried from config converter for edge stats output).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kumo_source_table_name: Option<String>,
    /// TT-NG temporal-LP timestamp grouping: when true, ingest spills each
    /// node's `TIMESTAMP` value during encode and, after the index is built,
    /// emits `<table>/bucket_index.bin` (final node-ids grouped contiguously by
    /// timestamp) plus a per-bucket `{ts,offset,count}` list. Default `false`
    /// produces byte-identical output to before. See [`crate::ts_grouping`].
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub group_by_timestamp: bool,
}

impl TableSpec {
    /// Get local Parquet/CSV paths. Kept slice-based for API compatibility.
    pub fn get_paths(&self) -> &[String] {
        match &self.source {
            DataSource::Parquet { paths } | DataSource::Csv { paths } => paths,
            _ => &[],
        }
    }

    /// Get mutable local Parquet/CSV paths.
    pub fn get_paths_mut(&mut self) -> &mut Vec<String> {
        match &mut self.source {
            DataSource::Parquet { paths } | DataSource::Csv { paths } => paths,
            _ => panic!("get_paths_mut called on non-file source"),
        }
    }

    /// Iterate local or manifest paths without discarding identity.
    pub fn source_paths(&self) -> Box<dyn ExactSizeIterator<Item = &str> + '_> {
        Box::new(self.source_objects().map(|object| object.path))
    }

    pub(crate) fn source_objects(
        &self,
    ) -> Box<dyn ExactSizeIterator<Item = SourceObjectRef<'_>> + '_> {
        match &self.source {
            DataSource::Parquet { paths } | DataSource::Csv { paths } => {
                Box::new(paths.iter().map(|path| SourceObjectRef {
                    path,
                    manifest: None,
                }))
            }
            DataSource::ObjectStoreParquet { objects, .. } => {
                Box::new(objects.iter().map(|object| SourceObjectRef {
                    path: &object.path,
                    manifest: Some(object),
                }))
            }
            _ => Box::new(std::iter::empty()),
        }
    }

    /// Explicitly replace a remote source with staged local Parquet files.
    pub fn replace_with_local_parquet(&mut self, paths: Vec<String>) {
        self.source = DataSource::Parquet { paths };
    }

    pub(crate) fn validate_object_store_manifest(&self) -> Result<&ObjectStoreObject> {
        let DataSource::ObjectStoreParquet {
            objects,
            auth_profile,
            authority: source_authority,
        } = &self.source
        else {
            return Err(anyhow!("table {} is not an object-store source", self.name));
        };
        ensure!(
            auth_profile == "ambient",
            "table {} object-store auth_profile {auth_profile:?} is unsupported; only explicit ambient mode is available",
            self.name
        );
        let first = objects
            .first()
            .ok_or_else(|| anyhow!("table {} has no manifest objects", self.name))?;
        let mut paths = HashSet::new();
        let mut manifest_authority = None;
        let mut total = 0u64;
        for object in objects {
            ensure!(
                object.size > 0,
                "table {} object-store manifest path {} has zero size",
                self.name,
                object.path
            );
            ensure!(
                paths.insert(object.path.as_str()),
                "table {} object-store manifest repeats {}",
                self.name,
                object.path
            );
            let valid_identity = [object.e_tag.as_deref(), object.version.as_deref()]
                .into_iter()
                .flatten()
                .any(|value| !value.trim().is_empty());
            ensure!(
                valid_identity,
                "table {} object-store manifest path {} has no valid immutable identity",
                self.name,
                object.path
            );
            let current = parse_cloud_object_authority(&object.path)?;
            ensure!(
                manifest_authority
                    .as_ref()
                    .is_none_or(|first| first == &current),
                "all cloud paths must share the same bucket/container"
            );
            manifest_authority = Some(current);
            total = total.checked_add(object.size).ok_or_else(|| {
                anyhow!(
                    "table {} object-store manifest size overflows u64",
                    self.name
                )
            })?;
        }
        if let Some(size) = self.source_size {
            ensure!(
                size == total,
                "table {} source_size {size} does not match object-store manifest size {total}",
                self.name
            );
        }
        if let Some(source_authority) = source_authority {
            source_authority.validate_for(
                manifest_authority
                    .as_ref()
                    .expect("a non-empty manifest has one provider authority"),
            )?;
        }
        Ok(first)
    }

    /// Get the table format based on the source type.
    pub fn format(&self) -> TableFormat {
        match &self.source {
            DataSource::Parquet { .. } | DataSource::ObjectStoreParquet { .. } => {
                TableFormat::Parquet
            }
            DataSource::Csv { .. } => TableFormat::Csv,
            _ => TableFormat::Parquet, // Default for non-file sources
        }
    }
}

/// The semantic type of a column, mirroring kumo-ml's `Stype`.
///
/// Determines which encoder and statistics are appropriate for a column.
/// Can be inferred from `IngestedType` + column name heuristics (Phase 1),
/// or refined later using column statistics like num_unique (Phase 2).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticType {
    /// Numeric values (integers, floats, decimals).
    Numerical,
    /// Low-cardinality discrete values (booleans, enums, small int sets).
    Categorical,
    /// Unique identifiers (primary keys, foreign keys).
    Id,
    /// Free-form text with semantic content.
    Text,
    /// Date/time values.
    Timestamp,
    /// Ordered lists (embeddings, float arrays).
    Sequence,
    /// Latitude/longitude coordinate pair. Stored as a fixed length-2 numeric
    /// list so downstream geo encoders can validate the raw coordinate shape.
    Latlong,
    /// Concatenation of multiple categories (e.g. comma-separated tags).
    Multicategorical,
    /// Column type not supported for ML encoding.
    Unsupported,
    /// Image URLs stored as strings. Kumo's training pipeline treats these
    /// as `Stype.image` → `Image()` encoder → `UnsupportedPreMatMapper`
    /// (the column is skipped during tensor creation). Col-stats emit only
    /// the universal stats (COUNT, NUM_NA, NA_FRACTION) under the
    /// `kumo.image` stype key so the strict-path cache check passes.
    ImageUrl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_dtype: Option<String>,
    #[serde(alias = "dtype")]
    pub ingested_dtype: IngestedType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_len: Option<usize>,
    // For timestamp/date columns inferred from strings, this captures the parse format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp_format: Option<String>,
    // For integer timestamps, indicates the epoch unit to interpret the value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp_unit: Option<TimestampUnit>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<i32>,
    /// Semantic type for ML encoding. Inferred from dtype/name if not provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_type: Option<SemanticType>,
    /// For stype=multicategorical + IngestedType=String: the delimiter used to
    /// split each row into tokens. `None` means the whole cell is one token.
    /// Absent (serde skip) for non-String multicat and for non-multicat columns.
    /// Populated by `config_converter` at convert time (sampling the source
    /// parquet), or supplied by the user on the upstream Kumo config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multicategorical_delimiter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IngestedType {
    Int8,
    Int16,
    Int64,
    Int32,
    UInt8,
    UInt16,
    UInt64,
    UInt32,
    Float64,
    Float32,
    Bool,
    #[serde(alias = "utf8")]
    String,
    Bytes,
    Uuid,
    Uuid256,
    Ip,
    Date,
    Timestamp,
    Decimal128,
    Decimal256,
    Float32Array,
    Float64Array,
    Float16Array,
    Int8Array,
    Int16Array,
    Int32Array,
    Int64Array,
    StringArray,
}

impl IngestedType {
    /// Returns true if this type represents an integer or boolean value.
    /// Used to emit category count keys as JSON numbers instead of strings.
    pub fn is_integer_like(&self) -> bool {
        matches!(
            self,
            IngestedType::Int8
                | IngestedType::Int16
                | IngestedType::Int32
                | IngestedType::Int64
                | IngestedType::UInt8
                | IngestedType::UInt16
                | IngestedType::UInt32
                | IngestedType::UInt64
                | IngestedType::Bool
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TimestampUnit {
    Seconds,
    Millis,
    Micros,
    Nanos,
}

impl TimestampUnit {
    /// Convert an integer epoch value in this unit to epoch seconds.
    /// Ceil-rounds so a sub-second value never lands in an earlier second,
    /// matching rowpack's `timestamp_from_int`.
    #[inline]
    pub fn scale_to_secs(self, value: i64) -> i64 {
        fn div_ceil(value: i64, divisor: i64) -> i64 {
            let d = value.div_euclid(divisor);
            let r = value.rem_euclid(divisor);
            if r > 0 {
                d + 1
            } else {
                d
            }
        }
        match self {
            TimestampUnit::Seconds => value,
            TimestampUnit::Millis => div_ceil(value, 1_000),
            TimestampUnit::Micros => div_ceil(value, 1_000_000),
            TimestampUnit::Nanos => div_ceil(value, 1_000_000_000),
        }
    }

    /// Infer the epoch unit of an integer timestamp from its magnitude
    /// (covers ~1970 to ~2286): `|v| < 1e10` → seconds, `< 1e13` → millis,
    /// `< 1e16` → micros, else nanos; `0` → None. Shared fallback when no
    /// explicit unit is supplied.
    #[inline]
    pub fn from_epoch_magnitude(value: i64) -> Option<TimestampUnit> {
        let abs = value.unsigned_abs();
        if abs == 0 {
            return None;
        }
        Some(if abs < 10_000_000_000 {
            TimestampUnit::Seconds
        } else if abs < 10_000_000_000_000 {
            TimestampUnit::Millis
        } else if abs < 10_000_000_000_000_000 {
            TimestampUnit::Micros
        } else {
            TimestampUnit::Nanos
        })
    }

    /// Canonical `s`/`ms`/`us`/`ns` token; the inverse of `map_timestamp_unit`.
    pub fn token(self) -> &'static str {
        match self {
            TimestampUnit::Seconds => "s",
            TimestampUnit::Millis => "ms",
            TimestampUnit::Micros => "us",
            TimestampUnit::Nanos => "ns",
        }
    }

    pub fn to_arrow(self) -> arrow_schema::TimeUnit {
        match self {
            TimestampUnit::Seconds => arrow_schema::TimeUnit::Second,
            TimestampUnit::Millis => arrow_schema::TimeUnit::Millisecond,
            TimestampUnit::Micros => arrow_schema::TimeUnit::Microsecond,
            TimestampUnit::Nanos => arrow_schema::TimeUnit::Nanosecond,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TableFormat {
    Csv,
    #[default]
    Parquet,
}

/// One exact object captured in a direct object-store source manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObjectStoreObject {
    pub path: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e_tag: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl ObjectStoreObject {
    pub(crate) fn condition(&self) -> diskgraph_fs::CloudObjectCondition {
        diskgraph_fs::CloudObjectCondition {
            size: self.size,
            e_tag: self.e_tag.clone(),
            version: self.version.clone(),
        }
    }
}

/// Non-secret provider routing selected while the source manifest is captured.
///
/// Credentials remain process-local. This block only prevents DiskGraph from
/// silently rebuilding a client with a different provider authority.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObjectStoreAuthority {
    S3 { region: String },
    Gcs,
    Adls { cloud: AzureCloud },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AzureCloud {
    Public,
    UsGovernment,
    China,
}

#[derive(Clone, Copy)]
pub(crate) struct SourceObjectRef<'a> {
    pub(crate) path: &'a str,
    pub(crate) manifest: Option<&'a ObjectStoreObject>,
}

impl SourceObjectRef<'_> {
    pub(crate) fn condition(&self) -> Option<diskgraph_fs::CloudObjectCondition> {
        self.manifest.map(ObjectStoreObject::condition)
    }
}

/// Data source configuration - abstracts where table data comes from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataSource {
    /// Parquet files (local, s3://, gs://, or az://).
    ///
    /// Cloud paths are retained for backwards compatibility and are promoted
    /// to an in-memory exact manifest before schema inspection or ingest. New
    /// callers should prefer `ObjectStoreParquet` when they already have one.
    Parquet { paths: Vec<String> },
    /// Exact cloud Parquet manifest with explicit authentication.
    ObjectStoreParquet {
        objects: Vec<ObjectStoreObject>,
        auth_profile: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        authority: Option<ObjectStoreAuthority>,
    },
    /// CSV files (local or cloud storage).
    Csv {
        /// List of file paths (local, s3://, gs://, az://).
        paths: Vec<String>,
    },
    /// Snowflake data warehouse.
    Snowflake {
        /// Connection identifier referencing a connection config.
        connection_id: String,
        /// Table or view name in Snowflake.
        table_or_view: String,
    },
    /// Databricks.
    Databricks {
        /// Databricks workspace hostname, without an HTTP path.
        server_hostname: String,
        /// SQL warehouse identifier.
        warehouse_id: String,
        /// Catalog name.
        catalog: String,
        /// Schema name.
        schema: String,
        /// Final validated read-only SQL executed by the helper. Result column
        /// names, including aliases, must match `ColumnSpec::name`.
        sql: String,
        /// Optional Databricks unified-authentication profile. When omitted,
        /// the SDK uses notebook-native authentication, standard environment
        /// variables, or the default local profile.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        auth_profile: Option<String>,
    },
    /// Google BigQuery.
    BigQuery {
        /// Connection identifier referencing a connection config.
        connection_id: String,
        /// GCP project ID.
        project: String,
        /// BigQuery dataset name.
        dataset: String,
        /// Table name.
        table: String,
    },
}

/// Time-based sort key specification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimeSpec {
    /// The table containing the timestamp column.
    pub table: String,
    /// The timestamp column name.
    pub column: String,
    /// Bit width for encoding: 8, 16, or 32. Null for auto-detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bitwidth: Option<u8>,
    /// Anchor time in epoch seconds. Null for auto-detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor_time: Option<i64>,
    /// Time unit: "s", "ms", "day", "hour", etc.
    pub unit: String,
}

/// Weight-based sort key specification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightSpec {
    /// The table containing columns referenced in the expression.
    pub table: String,
    /// Weight expression - column name or math expression.
    pub expr: String,
    /// Encoding dtype: "byte", "uint16", "uint32", "float16", "float32"
    pub dtype: String,
}

/// Direction-specific configuration for edge traversal.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DirectionSpec {
    /// Time-based sort key for this direction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<TimeSpec>,
    /// Weight-based sort key for this direction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<WeightSpec>,
}

/// Foreign key specification defining a 1:N relationship.
/// Defines an edge from this table to another table's primary key.
///
/// `Default` sets `bidirectional=true` so literal builders can use
/// `..Default::default()`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FKeySpec {
    /// Column on this table that references the destination table.
    pub src_column: String,
    /// Destination table name. Must have a primary key.
    pub dst_table: String,
    /// Forward direction config (src -> dst).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward: Option<DirectionSpec>,
    /// Reverse direction config (dst -> src).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverse: Option<DirectionSpec>,
    /// True when the FK produces both forward and reverse edges (legacy
    /// behavior). False emits forward edges only — no reverse RelationDef,
    /// no reverse temp files, no reverse_batcher. Default true preserves
    /// legacy bidirectional behavior.
    #[serde(
        default = "default_bidirectional",
        skip_serializing_if = "is_default_bidirectional"
    )]
    pub bidirectional: bool,
}

fn default_bidirectional() -> bool {
    true
}
fn is_default_bidirectional(v: &bool) -> bool {
    *v
}

impl Default for FKeySpec {
    fn default() -> Self {
        Self {
            src_column: String::new(),
            dst_table: String::new(),
            forward: None,
            reverse: None,
            bidirectional: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTableSpec {
    /// Source foreign key.
    pub src_fk: ForeignKeySpec,
    /// Destination foreign key.
    pub dst_fk: ForeignKeySpec,
    /// Forward direction config (src -> dst).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward: Option<DirectionSpec>,
    /// Reverse direction config (dst -> src).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reverse: Option<DirectionSpec>,
}

/// Foreign key reference in an edge table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForeignKeySpec {
    /// Column in the edge table containing the foreign key value.
    pub edge_column: String,
    /// Referenced node table (must have a primary key - joins on PK).
    pub ref_table: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CloudSource {
    S3 { bucket: String },
    Gcs { bucket: String },
    Adls { account: String, container: String },
}

impl ObjectStoreAuthority {
    fn validate_for(&self, source: &CloudSource) -> Result<()> {
        match (self, source) {
            (Self::S3 { region }, CloudSource::S3 { .. }) => {
                let valid_region = !region.is_empty()
                    && region.len() <= 64
                    && !region.starts_with('-')
                    && !region.ends_with('-')
                    && region.bytes().all(|byte| {
                        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'
                    });
                ensure!(valid_region, "invalid S3 source region {region:?}");
            }
            (Self::Gcs, CloudSource::Gcs { .. }) => {}
            (Self::Adls { cloud }, CloudSource::Adls { .. }) => {
                ensure!(
                    *cloud == AzureCloud::Public,
                    "ADLS source cloud {cloud:?} is not supported by the current object-store client"
                );
            }
            _ => {
                return Err(anyhow!(
                    "object-store source authority does not match manifest provider"
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn parse_cloud_source(path: &str) -> Result<Option<CloudSource>> {
    if let Some(rest) = path.strip_prefix("s3://") {
        let mut parts = rest.splitn(2, '/');
        let bucket = parts.next().unwrap_or("").trim().to_string();
        if bucket.is_empty() {
            return Err(anyhow!("invalid s3 path {path}"));
        }
        return Ok(Some(CloudSource::S3 { bucket }));
    }
    if let Some(rest) = path.strip_prefix("gs://") {
        let mut parts = rest.splitn(2, '/');
        let bucket = parts.next().unwrap_or("").trim().to_string();
        if bucket.is_empty() {
            return Err(anyhow!("invalid gs path {path}"));
        }
        return Ok(Some(CloudSource::Gcs { bucket }));
    }
    if let Some(rest) = path.strip_prefix("az://") {
        let mut parts = rest.splitn(3, '/');
        let account = parts.next().unwrap_or("").trim().to_string();
        let container = parts.next().unwrap_or("").trim().to_string();
        if account.is_empty() || container.is_empty() {
            return Err(anyhow!("invalid az path {path}"));
        }
        return Ok(Some(CloudSource::Adls { account, container }));
    }
    Ok(None)
}

pub(crate) fn parse_cloud_object_authority(path: &str) -> Result<CloudSource> {
    let authority = parse_cloud_source(path)?
        .ok_or_else(|| anyhow!("object-store manifest path {path} is not a cloud URL"))?;
    let rest = path.split_once("://").unwrap().1;
    let index = if matches!(&authority, CloudSource::Adls { .. }) {
        2
    } else {
        1
    };
    let key = rest
        .split('/')
        .nth(index)
        .unwrap_or("")
        .split(['?', '#'])
        .next()
        .unwrap_or("");
    ensure!(
        !key.is_empty(),
        "object-store manifest path {path} has no object key"
    );
    Ok(authority)
}

impl GraphConfig {
    /// Normalize columns that need a canonical on-disk shape.
    ///
    /// Normalize multicategorical "pseudo list" columns to the native list
    /// shape on disk. For any column with `(stype=Multicategorical,
    /// dtype=String, multicategorical_delimiter=Some(_))`, flip
    /// `ingested_dtype` to `StringArray`. The parquet column stays `Utf8`
    /// scalar; the access layer (`RowAccess::string_list_view`) sees the
    /// delimiter on the per-column hint and splits scalar→list at read time
    /// using GraphEngine-parity drop-empty semantics. Result: rowpack
    /// persists a native list, downstream readers (TT-gen, infer,
    /// tensor mapper) never re-split.
    ///
    /// For `stype=Latlong`, force `fixed_len=2` so rowpack can enforce the
    /// coordinate pair shape and use the compact fixed-array representation.
    ///
    /// Called automatically by `ingest_with_stats_from_source` and by
    /// `convert_kumo_to_diskgraph`. Idempotent. See kumo-ai/kumo#28478.
    pub fn normalize_multicat_pseudo_list(&mut self) {
        for table in self.tables.iter_mut() {
            for col in table.columns.iter_mut() {
                if col.semantic_type == Some(SemanticType::Multicategorical)
                    && col.ingested_dtype == IngestedType::String
                    && col.multicategorical_delimiter.is_some()
                {
                    col.ingested_dtype = IngestedType::StringArray;
                }
                if col.semantic_type == Some(SemanticType::Latlong) {
                    col.fixed_len = Some(2);
                }
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();
        if self.dataset_name.trim().is_empty() {
            errors.push("dataset_name is required".to_string());
        }
        if self.tables.is_empty() {
            errors.push("at least one table is required".to_string());
        }
        if let Some(shards) = self.num_shards {
            if shards == 0 {
                errors.push("num_shards must be >= 1".to_string());
            }
        }
        if let Some(format) = self.edge_format {
            if matches!(format, EdgeStorageFormat::Parquet) {
                errors.push("parquet output format is no longer supported".to_string());
            }
        }
        let mut table_names = HashSet::new();
        // Disk artifacts use lowercase table directory names, so reject
        // case-insensitive collisions up front (e.g. User vs user).
        let mut table_names_ci: HashMap<String, String> = HashMap::new();
        let mut table_map: HashMap<&str, &TableSpec> = HashMap::new();
        for table in &self.tables {
            let mut cloud_source: Option<CloudSource> = None;
            if table.name.trim().is_empty() {
                errors.push("table name cannot be empty".to_string());
                continue;
            }
            if !table_names.insert(table.name.as_str()) {
                errors.push(format!("duplicate table name {}", table.name));
            }
            let lower_name = table.name.to_lowercase();
            if let Some(existing) = table_names_ci.get(&lower_name) {
                if existing != &table.name {
                    errors.push(format!(
                        "table names '{}' and '{}' collide under lowercase normalization",
                        existing, table.name
                    ));
                }
            } else {
                table_names_ci.insert(lower_name, table.name.clone());
            }
            match &table.source {
                DataSource::Parquet { paths } | DataSource::Csv { paths } => {
                    if paths.is_empty() {
                        errors.push(format!("table {} has no paths", table.name));
                    }
                }
                DataSource::ObjectStoreParquet { .. } => {
                    if let Err(error) = table.validate_object_store_manifest() {
                        errors.push(error.to_string());
                    }
                }
                DataSource::Databricks {
                    server_hostname,
                    warehouse_id,
                    catalog,
                    schema,
                    sql,
                    auth_profile,
                } => {
                    for (field, value) in [
                        ("server_hostname", server_hostname),
                        ("warehouse_id", warehouse_id),
                        ("catalog", catalog),
                        ("schema", schema),
                        ("sql", sql),
                    ] {
                        if value.trim().is_empty() {
                            errors.push(format!(
                                "table {} Databricks {} cannot be empty",
                                table.name, field
                            ));
                        }
                    }
                    if server_hostname.contains('/') || server_hostname.contains(':') {
                        errors.push(format!(
                            "table {} Databricks server_hostname must be a hostname without scheme or path",
                            table.name
                        ));
                    }
                    if auth_profile
                        .as_ref()
                        .is_some_and(|profile| profile.trim().is_empty())
                    {
                        errors.push(format!(
                            "table {} Databricks auth_profile cannot be empty",
                            table.name
                        ));
                    }
                }
                DataSource::Snowflake { .. } | DataSource::BigQuery { .. } => {
                    errors.push(format!(
                        "table {} uses a source that is not implemented",
                        table.name
                    ));
                }
            }
            if table.columns.is_empty() {
                errors.push(format!("table {} has no columns", table.name));
            }
            let mut col_names = HashSet::new();
            // Track per-table path types: a single table must not mix
            // local and cloud paths, but different tables CAN use
            // different path types (e.g. one table local, another S3).
            let mut table_has_cloud = false;
            let mut table_has_local = false;
            for path in table.get_paths() {
                match parse_cloud_source(path) {
                    Ok(Some(source)) => {
                        table_has_cloud = true;
                        if let Some(existing) = &cloud_source {
                            if existing != &source {
                                errors.push(
                                    "all cloud paths must share the same bucket/container"
                                        .to_string(),
                                );
                            }
                        } else {
                            cloud_source = Some(source);
                        }
                    }
                    Ok(None) => {
                        table_has_local = true;
                    }
                    Err(err) => errors.push(err.to_string()),
                }
            }
            if table_has_cloud && table_has_local {
                errors.push(format!("table {} mixes local and cloud paths", table.name));
            }
            for col in &table.columns {
                if col.name.trim().is_empty() {
                    errors.push(format!("table {} has empty column name", table.name));
                    continue;
                }
                if !col_names.insert(col.name.as_str()) {
                    errors.push(format!(
                        "table {} has duplicate column {}",
                        table.name, col.name
                    ));
                }
                if col.fixed_len.is_some()
                    && !matches!(
                        col.ingested_dtype,
                        IngestedType::String
                            | IngestedType::Bytes
                            | IngestedType::Float16Array
                            | IngestedType::Float32Array
                            | IngestedType::Float64Array
                            | IngestedType::Int8Array
                            | IngestedType::Int16Array
                            | IngestedType::Int32Array
                            | IngestedType::Int64Array
                    )
                {
                    errors.push(format!(
                        "table {} column {} fixed_len only valid for string/bytes/numeric arrays",
                        table.name, col.name
                    ));
                }
                if col.semantic_type == Some(SemanticType::Latlong) {
                    if !matches!(
                        col.ingested_dtype,
                        IngestedType::Float16Array
                            | IngestedType::Float32Array
                            | IngestedType::Float64Array
                            | IngestedType::Int8Array
                            | IngestedType::Int16Array
                            | IngestedType::Int32Array
                            | IngestedType::Int64Array
                    ) {
                        errors.push(format!(
                            "table {} column {} latlong requires a numeric array dtype",
                            table.name, col.name
                        ));
                    }
                    if col.fixed_len != Some(2) {
                        errors.push(format!(
                            "table {} column {} latlong requires fixed_len=2",
                            table.name, col.name
                        ));
                    }
                }
                if col.semantic_type == Some(SemanticType::Multicategorical)
                    && !matches!(
                        col.ingested_dtype,
                        IngestedType::StringArray
                            | IngestedType::Int64Array
                            | IngestedType::Float16Array
                            | IngestedType::Float32Array
                            | IngestedType::Float64Array
                    )
                {
                    // Multicat requires a list-shaped storage on disk. Scalar
                    // string columns with a declared delimiter are flipped to
                    // StringArray by `normalize_multicat_pseudo_list` before
                    // validation runs, so anything still scalar at this point
                    // is a misdeclaration. A single-value-per-row column —
                    // whether int or string — is `categorical` semantics, not
                    // multicategorical.
                    let hint = if col.ingested_dtype == IngestedType::String {
                        "set multicategorical_delimiter, switch to a native \
                         list dtype (string_array / int64_array), or use \
                         stype=categorical for single-token columns"
                    } else {
                        "use stype=categorical for single-value columns, or \
                         a native list dtype for multi-token columns"
                    };
                    errors.push(format!(
                        "table {} column {} stype=multicategorical requires a \
                         list-shaped ingested_dtype (string_array / int64_array \
                         / floatN_array); got {:?}. {hint}.",
                        table.name, col.name, col.ingested_dtype
                    ));
                }
                let is_decimal = matches!(
                    col.ingested_dtype,
                    IngestedType::Decimal128 | IngestedType::Decimal256
                );
                if is_decimal {
                    let precision = match col.precision {
                        Some(v) => v,
                        None => {
                            errors.push(format!(
                                "table {} decimal column {} missing precision",
                                table.name, col.name
                            ));
                            continue;
                        }
                    };
                    let scale = match col.scale {
                        Some(v) => v,
                        None => {
                            errors.push(format!(
                                "table {} decimal column {} missing scale",
                                table.name, col.name
                            ));
                            continue;
                        }
                    };
                    let max_precision = match col.ingested_dtype {
                        IngestedType::Decimal128 => 38,
                        IngestedType::Decimal256 => 76,
                        _ => 0,
                    };
                    if precision == 0 || precision > max_precision {
                        errors.push(format!(
                            "table {} decimal column {} precision out of range",
                            table.name, col.name
                        ));
                    }
                    if scale.unsigned_abs() > precision {
                        errors.push(format!(
                            "table {} decimal column {} scale out of range",
                            table.name, col.name
                        ));
                    }
                    if scale < i8::MIN as i32 || scale > i8::MAX as i32 {
                        errors.push(format!(
                            "table {} decimal column {} scale out of range",
                            table.name, col.name
                        ));
                    }
                } else if (col.precision.is_some() || col.scale.is_some())
                    && !source_dtype_is_decimal(col.source_dtype.as_deref())
                {
                    errors.push(format!(
                        "table {} column {} precision/scale only valid for decimal",
                        table.name, col.name
                    ));
                }
            }
            if let Some(pk) = &table.primary_key {
                if !col_names.contains(pk.as_str()) {
                    errors.push(format!("table {} primary_key {} not found", table.name, pk));
                } else if let Some(col) = table.columns.iter().find(|c| c.name == *pk) {
                    if !matches!(
                        col.ingested_dtype,
                        IngestedType::Int8
                            | IngestedType::Int16
                            | IngestedType::Int64
                            | IngestedType::Int32
                            | IngestedType::UInt8
                            | IngestedType::UInt16
                            | IngestedType::UInt64
                            | IngestedType::UInt32
                            | IngestedType::Float32
                            | IngestedType::Float64
                            | IngestedType::Decimal128
                            | IngestedType::Decimal256
                            | IngestedType::String
                            | IngestedType::Bytes
                            | IngestedType::Uuid
                            | IngestedType::Uuid256
                            | IngestedType::Ip
                    ) {
                        errors.push(format!(
                        "table {} primary_key {} must be numeric or string/bytes/uuid/uuid256/ip",
                        table.name, pk
                    ));
                    }
                }
            }
            if let Some(weight) = &table.weight {
                if !col_names.contains(weight.as_str()) {
                    errors.push(format!(
                        "table {} weight column {} not found",
                        table.name, weight
                    ));
                } else if let Some(col) = table.columns.iter().find(|c| c.name == *weight) {
                    if !matches!(
                        col.ingested_dtype,
                        IngestedType::Int8
                            | IngestedType::Int16
                            | IngestedType::Int64
                            | IngestedType::Int32
                            | IngestedType::UInt8
                            | IngestedType::UInt16
                            | IngestedType::UInt64
                            | IngestedType::UInt32
                            | IngestedType::Float64
                            | IngestedType::Float32
                    ) {
                        errors.push(format!(
                            "table {} weight column {} must be numeric",
                            table.name, weight
                        ));
                    }
                }
            }
            if let Some(ts) = &table.create_time {
                if !col_names.contains(ts.as_str()) {
                    errors.push(format!(
                        "table {} create_time column {} not found",
                        table.name, ts
                    ));
                } else if let Some(col) = table.columns.iter().find(|c| c.name == *ts) {
                    if !matches!(
                        col.ingested_dtype,
                        IngestedType::Date | IngestedType::Timestamp
                    ) {
                        errors.push(format!(
                            "table {} create_time column {} must be date/timestamp",
                            table.name, ts
                        ));
                    }
                }
            }
            if let Some(edge) = &table.edge_table {
                for fk in [&edge.src_fk, &edge.dst_fk] {
                    if !col_names.contains(fk.edge_column.as_str()) {
                        errors.push(format!(
                            "table {} edge fk column {} not found",
                            table.name, fk.edge_column
                        ));
                    }
                }
            }
            table_map.insert(table.name.as_str(), table);
        }

        let mut warnings = Vec::new();

        // Validate embedded fkeys on regular tables
        for table in &self.tables {
            for fkey in &table.fkeys {
                let ref_table = match table_map.get(fkey.dst_table.as_str()) {
                    Some(t) => *t,
                    None => {
                        errors.push(format!(
                            "table {} fkey {} references missing table {}",
                            table.name, fkey.src_column, fkey.dst_table
                        ));
                        continue;
                    }
                };
                let pk_name = match &ref_table.primary_key {
                    Some(pk) => pk.as_str(),
                    None => {
                        errors.push(format!(
                            "table {} fkey {} references {} which has no primary key",
                            table.name, fkey.src_column, fkey.dst_table
                        ));
                        continue;
                    }
                };
                if let Some(fk_col) = table.columns.iter().find(|c| c.name == fkey.src_column) {
                    if let Some(pk_col) = ref_table.columns.iter().find(|c| c.name == pk_name) {
                        let result = check_fk_pk_types(
                            fk_col,
                            pk_col,
                            &table.name,
                            &fkey.dst_table,
                            self.coercion_mode,
                        );
                        if let Some(err) = result.error {
                            errors.push(err.to_string());
                        }
                        if let Some(warn) = result.warning {
                            warnings.push(warn);
                        }
                    }
                } else {
                    errors.push(format!(
                        "table {} fkey column {} not found",
                        table.name, fkey.src_column
                    ));
                }
            }
        }

        // Validate edge tables
        for table in &self.tables {
            if let Some(edge) = &table.edge_table {
                for fk in [&edge.src_fk, &edge.dst_fk] {
                    let ref_table = match table_map.get(fk.ref_table.as_str()) {
                        Some(t) => *t,
                        None => {
                            errors.push(format!(
                                "edge table {} references missing table {}",
                                table.name, fk.ref_table
                            ));
                            continue;
                        }
                    };
                    let pk_name = match &ref_table.primary_key {
                        Some(pk) => pk.as_str(),
                        None => {
                            errors.push(format!(
                                "edge table {} references {} which has no primary key",
                                table.name, fk.ref_table
                            ));
                            continue;
                        }
                    };
                    if let Some(fk_col) = table.columns.iter().find(|c| c.name == fk.edge_column) {
                        if let Some(pk_col) = ref_table.columns.iter().find(|c| c.name == pk_name) {
                            let result = check_fk_pk_types(
                                fk_col,
                                pk_col,
                                &table.name,
                                &fk.ref_table,
                                self.coercion_mode,
                            );
                            if let Some(err) = result.error {
                                errors.push(err.to_string());
                            }
                            if let Some(warn) = result.warning {
                                warnings.push(warn);
                            }
                        }
                    }
                }
            }
        }

        // Print warnings to stderr
        for warn in &warnings {
            eprintln!("[WARN] {warn}");
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(anyhow!(errors.join("\n")))
        }
    }

    /// Resolve FK column types to match their referenced PK types.
    ///
    /// For FK→PK relationships where coercion is Safe or Conditional,
    /// this updates the FK column's `ingested_dtype` to match the PK type.
    /// This allows Arrow→IngestedType conversion to handle the casting,
    /// ensuring FK and PK produce the same KeyType for hash lookup.
    ///
    /// Returns a new GraphConfig with resolved types.
    /// Should be called after `validate()` succeeds.
    pub fn resolve_fk_types(&self) -> GraphConfig {
        let mut resolved = self.clone();

        // Build table lookup for PK column access
        let table_map: HashMap<&str, &TableSpec> =
            self.tables.iter().map(|t| (t.name.as_str(), t)).collect();

        // Track which columns need type updates: (table_name, column_name) -> new_type
        let mut type_updates: HashMap<(String, String), IngestedType> = HashMap::new();

        // Process embedded fkeys on regular tables
        for table in &self.tables {
            for fkey in &table.fkeys {
                if let Some(ref_table) = table_map.get(fkey.dst_table.as_str()) {
                    if let Some(pk_name) = &ref_table.primary_key {
                        if let Some(fk_col) =
                            table.columns.iter().find(|c| c.name == fkey.src_column)
                        {
                            if let Some(pk_col) =
                                ref_table.columns.iter().find(|c| c.name == *pk_name)
                            {
                                if let Some(new_type) = resolve_fk_type_to_pk(
                                    &fk_col.ingested_dtype,
                                    &pk_col.ingested_dtype,
                                    self.coercion_mode,
                                    pk_col.scale,
                                ) {
                                    type_updates.insert(
                                        (table.name.clone(), fkey.src_column.clone()),
                                        new_type,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Process edge tables
        for table in &self.tables {
            if let Some(edge) = &table.edge_table {
                for fk in [&edge.src_fk, &edge.dst_fk] {
                    if let Some(ref_table) = table_map.get(fk.ref_table.as_str()) {
                        if let Some(pk_name) = &ref_table.primary_key {
                            if let Some(fk_col) =
                                table.columns.iter().find(|c| c.name == fk.edge_column)
                            {
                                if let Some(pk_col) =
                                    ref_table.columns.iter().find(|c| c.name == *pk_name)
                                {
                                    if let Some(new_type) = resolve_fk_type_to_pk(
                                        &fk_col.ingested_dtype,
                                        &pk_col.ingested_dtype,
                                        self.coercion_mode,
                                        pk_col.scale,
                                    ) {
                                        type_updates.insert(
                                            (table.name.clone(), fk.edge_column.clone()),
                                            new_type,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply type updates to resolved config
        for table in &mut resolved.tables {
            for col in &mut table.columns {
                if let Some(new_type) = type_updates.get(&(table.name.clone(), col.name.clone())) {
                    if col.ingested_dtype != *new_type {
                        eprintln!(
                            "[INFO] FK type resolution: {}.{} {:?} → {:?} (to match PK)",
                            table.name, col.name, col.ingested_dtype, new_type
                        );
                        col.ingested_dtype = new_type.clone();
                    }
                }
            }
        }

        resolved
    }
}

/// Determine if FK type should be resolved to PK type.
/// Returns Some(pk_type) if coercion is Safe or Conditional and types differ.
/// Returns None if types are the same or coercion is not allowed.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn resolve_fk_type_to_pk(
    fk_type: &IngestedType,
    pk_type: &IngestedType,
    mode: CoercionMode,
    pk_scale: Option<i32>,
) -> Option<IngestedType> {
    if fk_type == pk_type {
        return None; // Same type, no resolution needed
    }

    let params = CoercionParams { pk_scale };
    let kind = coercion_kind(fk_type.clone(), pk_type.clone(), params);

    match (mode, kind) {
        // Safe coercion: resolve FK to PK type
        (CoercionMode::Lossless, CoercionKind::Safe) => Some(pk_type.clone()),

        // Conditional coercion: resolve FK to PK type (parsing will happen at runtime)
        (CoercionMode::Lossless, CoercionKind::Conditional) => Some(pk_type.clone()),

        // Strict mode, Lossy, or Incompatible: don't resolve (validate() will catch errors)
        _ => None,
    }
}

/// Result of FK-PK type compatibility check.
pub struct FkPkCheckResult {
    /// Whether the types are compatible (allows ingestion to proceed).
    pub allowed: bool,
    /// Warning message for conditional coercions.
    pub warning: Option<String>,
    /// Error message for incompatible/lossy coercions.
    pub error: Option<String>,
}

/// Check FK-PK type compatibility using the coercion system.
/// Delegates to `coercion::validate_fk_pk` to avoid duplicating coercion logic.
fn check_fk_pk_types(
    fk_col: &ColumnSpec,
    pk_col: &ColumnSpec,
    fk_table: &str,
    pk_table: &str,
    mode: CoercionMode,
) -> FkPkCheckResult {
    let result = validate_fk_pk(
        fk_col.ingested_dtype.clone(),
        pk_col.ingested_dtype.clone(),
        fk_table,
        &fk_col.name,
        pk_table,
        &pk_col.name,
        mode,
        pk_col.scale,
    );
    FkPkCheckResult {
        allowed: result.allowed,
        warning: result.warning.map(|w| w.to_string()),
        error: result.error.map(|e| e.to_string()),
    }
}

// ── Semantic type inference ─────────────────────────────────────────────────

/// Thresholds matching kumo-ml's `EncoderGenerator` constants.
const MAX_AUTO_CATEGORICAL_THRESHOLD: usize = 3;
const MAX_CATEGORIES: usize = 1_000_000;
const UNIQUE_INT_THRESHOLD: f64 = 0.05;

/// Column role context for semantic type inference.
pub struct ColumnRole<'a> {
    pub is_pk: bool,
    pub is_fk: bool,
    pub name: &'a str,
}

/// Sample statistics for a single column, computed during the context phase.
pub struct ColumnSampleStats {
    /// Number of unique values observed in the sample.
    pub num_unique: usize,
    /// Total non-null values observed.
    pub count: usize,
}

/// Infer the semantic type of a column.
///
/// Uses IngestedType for unambiguous cases, then falls back to sample stats
/// and column role for ambiguous types (integers, strings).
pub fn infer_semantic_type(
    col: &ColumnSpec,
    role: &ColumnRole<'_>,
    sample: Option<&ColumnSampleStats>,
) -> SemanticType {
    // PK/FK columns are always Id.
    if role.is_pk || role.is_fk {
        return SemanticType::Id;
    }

    // Unambiguous from IngestedType alone.
    match col.ingested_dtype {
        IngestedType::Bool => return SemanticType::Categorical,
        IngestedType::Date | IngestedType::Timestamp => return SemanticType::Timestamp,
        IngestedType::Uuid | IngestedType::Uuid256 | IngestedType::Ip => {
            return SemanticType::Id;
        }
        IngestedType::Float32Array
        | IngestedType::Float64Array
        | IngestedType::Float16Array
        | IngestedType::Int64Array => {
            return SemanticType::Sequence;
        }
        IngestedType::StringArray => {
            return SemanticType::Multicategorical;
        }
        IngestedType::Float32
        | IngestedType::Float64
        | IngestedType::Decimal128
        | IngestedType::Decimal256 => return SemanticType::Numerical,
        IngestedType::Bytes => return SemanticType::Unsupported,
        // Int and String types need stats — fall through.
        _ => {}
    }

    // Name heuristic: columns containing "id" are likely identifiers.
    if contains_identifier(role.name) {
        return SemanticType::Id;
    }

    // If no sample stats, use defaults.
    let Some(sample) = sample else {
        return match col.ingested_dtype {
            IngestedType::String => SemanticType::Categorical,
            _ => SemanticType::Numerical, // int types
        };
    };

    if sample.count == 0 {
        return SemanticType::Unsupported;
    }

    let num_unique = sample.num_unique;
    let unique_fraction = num_unique as f64 / sample.count as f64;

    // Binary/ternary: always categorical.
    if num_unique <= MAX_AUTO_CATEGORICAL_THRESHOLD {
        return SemanticType::Categorical;
    }

    // Within categorical cardinality range.
    if num_unique <= MAX_CATEGORIES {
        // High unique fraction for ints → numerical (not categorical).
        if is_int_type(&col.ingested_dtype) && unique_fraction > UNIQUE_INT_THRESHOLD {
            return SemanticType::Numerical;
        }
        return SemanticType::Categorical;
    }

    // High cardinality: numerical for ints, text for strings.
    match col.ingested_dtype {
        IngestedType::String => SemanticType::Text,
        _ => SemanticType::Numerical,
    }
}

/// Check if a column name looks like an identifier.
fn contains_identifier(name: &str) -> bool {
    let name_lower = name.to_lowercase();
    if name_lower == "id" {
        return true;
    }
    if name_lower.ends_with("_id") || name_lower.starts_with("id_") {
        return true;
    }
    if name_lower.contains("_id_") {
        return true;
    }
    if name_lower.contains("uuid") || name_lower.contains("guid") {
        return true;
    }
    false
}

fn is_int_type(dtype: &IngestedType) -> bool {
    matches!(
        dtype,
        IngestedType::Int8
            | IngestedType::Int16
            | IngestedType::Int32
            | IngestedType::Int64
            | IngestedType::UInt8
            | IngestedType::UInt16
            | IngestedType::UInt32
            | IngestedType::UInt64
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_unit_scale_to_secs() {
        assert_eq!(
            TimestampUnit::Millis.scale_to_secs(1_659_304_801_388),
            1_659_304_802
        );
        assert_eq!(
            TimestampUnit::Seconds.scale_to_secs(1_659_304_801),
            1_659_304_801
        );
        assert_eq!(
            TimestampUnit::Micros.scale_to_secs(1_659_304_801_000_000),
            1_659_304_801
        );
        assert_eq!(
            TimestampUnit::Nanos.scale_to_secs(1_659_304_801_000_000_000),
            1_659_304_801
        );
        // Ceil rounding: sub-second remainder rounds up.
        assert_eq!(TimestampUnit::Millis.scale_to_secs(1_001), 2);
        assert_eq!(TimestampUnit::Millis.scale_to_secs(1_000), 1);
    }

    #[test]
    fn timestamp_unit_from_epoch_magnitude() {
        assert_eq!(TimestampUnit::from_epoch_magnitude(0), None);
        assert_eq!(
            TimestampUnit::from_epoch_magnitude(1_659_304_801),
            Some(TimestampUnit::Seconds)
        );
        // 13 digits → milliseconds.
        assert_eq!(
            TimestampUnit::from_epoch_magnitude(1_659_304_801_388),
            Some(TimestampUnit::Millis)
        );
        assert_eq!(
            TimestampUnit::from_epoch_magnitude(1_659_304_801_000_000),
            Some(TimestampUnit::Micros)
        );
        assert_eq!(
            TimestampUnit::from_epoch_magnitude(1_659_304_801_000_000_000),
            Some(TimestampUnit::Nanos)
        );
        // Boundaries.
        assert_eq!(
            TimestampUnit::from_epoch_magnitude(9_999_999_999),
            Some(TimestampUnit::Seconds)
        );
        assert_eq!(
            TimestampUnit::from_epoch_magnitude(10_000_000_000),
            Some(TimestampUnit::Millis)
        );
    }

    fn make_column(name: &str, dtype: IngestedType) -> ColumnSpec {
        ColumnSpec {
            name: name.to_string(),
            source_dtype: None,
            ingested_dtype: dtype,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: None,
            multicategorical_delimiter: None,
        }
    }

    fn make_table(name: &str, columns: Vec<ColumnSpec>, primary_key: Option<&str>) -> TableSpec {
        TableSpec {
            name: name.to_string(),
            columns,
            primary_key: primary_key.map(String::from),
            create_time: None,
            end_time: None,
            weight: None,
            edge_table: None,
            rowpack: None,
            source: DataSource::Parquet {
                paths: vec!["test.parquet".to_string()],
            },
            source_size: None,
            fkeys: vec![],
            data_version: None,
            kumo_connector_id: None,
            kumo_source_table_name: None,
            group_by_timestamp: false,
        }
    }

    fn manifest_config() -> GraphConfig {
        serde_json::from_str(r#"{"dataset_name":"manifest","tables":[{"name":"first","columns":[{"name":"id","ingested_dtype":"int64"}],"primary_key":"id","source":{"type":"object_store_parquet","objects":[{"path":"s3://first/data.parquet","size":123,"e_tag":"etag"}],"auth_profile":"ambient"}},{"name":"second","columns":[{"name":"id","ingested_dtype":"int64"}],"primary_key":"id","source":{"type":"object_store_parquet","objects":[{"path":"gs://second/data.parquet","size":456,"version":"generation"}],"auth_profile":"ambient"}}]}"#).unwrap()
    }

    #[test]
    fn object_store_manifest_contract_fails_closed() {
        let mut config = manifest_config();
        config.validate().unwrap();
        let mut json = serde_json::to_value(&config).unwrap();
        let decoded: GraphConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), json);

        json["tables"][0]["source"]
            .as_object_mut()
            .unwrap()
            .remove("auth_profile");
        assert!(serde_json::from_value::<GraphConfig>(json).is_err());

        if let DataSource::ObjectStoreParquet { objects, .. } = &mut config.tables[0].source {
            objects[0].path = "s3://first".to_string();
        }
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("no object key"));
        if let DataSource::ObjectStoreParquet { objects, .. } = &mut config.tables[0].source {
            objects[0].path = "s3://first/data.parquet".to_string();
        }
        config.tables[0].source_size = Some(1);
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("manifest size"));
        config.tables[0].source_size = None;
        if let DataSource::ObjectStoreParquet { auth_profile, .. } = &mut config.tables[0].source {
            *auth_profile = "role".to_string();
        }
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("explicit ambient"));

        config.tables[0].replace_with_local_parquet(vec!["/staged.parquet".to_string()]);
        assert!(matches!(
            config.tables[0].source,
            DataSource::Parquet { .. }
        ));
        if let DataSource::Parquet { paths } = &mut config.tables[0].source {
            paths[0] = "s3://bucket/data.parquet".to_string();
        }
        config.validate().unwrap();
    }

    #[test]
    fn object_store_authority_is_provider_scoped_and_backward_compatible() {
        let mut config = manifest_config();
        config.validate().unwrap();
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(
            legacy["tables"][0]["source"].get("authority").is_none(),
            "legacy manifests must keep their existing serialized shape"
        );

        let DataSource::ObjectStoreParquet { authority, .. } = &mut config.tables[0].source else {
            panic!("expected object-store source");
        };
        *authority = Some(ObjectStoreAuthority::S3 {
            region: "us-west-2".to_string(),
        });
        config.validate().unwrap();
        let serialized = serde_json::to_value(&config).unwrap();
        assert_eq!(
            serialized["tables"][0]["source"]["authority"]["region"],
            "us-west-2"
        );
        let decoded = serde_json::from_value::<GraphConfig>(serialized.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), serialized);

        let mut unknown_field = serialized.clone();
        unknown_field["tables"][0]["source"]["authority"]["endpoint"] =
            serde_json::json!("https://wrong.invalid");
        assert!(serde_json::from_value::<GraphConfig>(unknown_field)
            .unwrap_err()
            .to_string()
            .contains("unknown field `endpoint`"));

        let DataSource::ObjectStoreParquet { authority, .. } = &mut config.tables[0].source else {
            unreachable!();
        };
        *authority = Some(ObjectStoreAuthority::Gcs);
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("does not match manifest provider"));

        let DataSource::ObjectStoreParquet { authority, .. } = &mut config.tables[0].source else {
            unreachable!();
        };
        *authority = Some(ObjectStoreAuthority::S3 {
            region: "us.west-2".to_string(),
        });
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("invalid S3 source region"));

        let DataSource::ObjectStoreParquet {
            objects, authority, ..
        } = &mut config.tables[0].source
        else {
            unreachable!();
        };
        objects[0].path = "az://account/container/data.parquet".to_string();
        *authority = Some(ObjectStoreAuthority::Adls {
            cloud: AzureCloud::Public,
        });
        config.validate().unwrap();

        let DataSource::ObjectStoreParquet { authority, .. } = &mut config.tables[0].source else {
            unreachable!();
        };
        *authority = Some(ObjectStoreAuthority::Adls {
            cloud: AzureCloud::UsGovernment,
        });
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("not supported by the current object-store client"));
    }

    // Note: Coercion kind tests (Same, Safe, Conditional, Lossy, Incompatible) are in
    // coercion/tests_matrix.rs (Matrix 4). These tests only verify config transformation.

    #[test]
    fn test_resolve_fk_types_embedded_fkey() {
        // orders.customer_id (String) → customers.id (Int64)
        let customers = make_table(
            "customers",
            vec![make_column("id", IngestedType::Int64)],
            Some("id"),
        );
        let mut orders = make_table(
            "orders",
            vec![
                make_column("id", IngestedType::Int64),
                make_column("customer_id", IngestedType::String),
            ],
            Some("id"),
        );
        orders.fkeys.push(FKeySpec {
            src_column: "customer_id".to_string(),
            dst_table: "customers".to_string(),
            forward: None,
            reverse: None,
            bidirectional: true,
        });

        let config = GraphConfig {
            dataset_name: "test".to_string(),
            tables: vec![customers, orders],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::Lossless,
        };

        let resolved = config.resolve_fk_types();

        let resolved_orders = resolved.tables.iter().find(|t| t.name == "orders").unwrap();
        let resolved_customer_id = resolved_orders
            .columns
            .iter()
            .find(|c| c.name == "customer_id")
            .unwrap();

        // String FK resolved to Int64 to match PK
        assert_eq!(resolved_customer_id.ingested_dtype, IngestedType::Int64);
    }

    #[test]
    fn test_resolve_fk_types_edge_table() {
        // Edge table with two FKs to different PK types
        let users = make_table(
            "users",
            vec![make_column("id", IngestedType::Int64)],
            Some("id"),
        );
        let products = make_table(
            "products",
            vec![make_column("id", IngestedType::Uuid)],
            Some("id"),
        );
        let mut reviews = make_table(
            "reviews",
            vec![
                make_column("user_id", IngestedType::String),
                make_column("product_id", IngestedType::String),
            ],
            None,
        );
        reviews.edge_table = Some(EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: "user_id".to_string(),
                ref_table: "users".to_string(),
            },
            dst_fk: ForeignKeySpec {
                edge_column: "product_id".to_string(),
                ref_table: "products".to_string(),
            },
            forward: None,
            reverse: None,
        });

        let config = GraphConfig {
            dataset_name: "test".to_string(),
            tables: vec![users, products, reviews],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::Lossless,
        };

        let resolved = config.resolve_fk_types();

        let resolved_reviews = resolved
            .tables
            .iter()
            .find(|t| t.name == "reviews")
            .unwrap();
        let resolved_user_id = resolved_reviews
            .columns
            .iter()
            .find(|c| c.name == "user_id")
            .unwrap();
        let resolved_product_id = resolved_reviews
            .columns
            .iter()
            .find(|c| c.name == "product_id")
            .unwrap();

        // String → Int64 (to match users.id)
        assert_eq!(resolved_user_id.ingested_dtype, IngestedType::Int64);
        // String → Uuid (to match products.id)
        assert_eq!(resolved_product_id.ingested_dtype, IngestedType::Uuid);
    }

    #[test]
    fn test_resolve_fk_types_int32_to_int64_safe() {
        // Int32→Int64 is Safe, but Int64→Int32 is Lossy (asymmetric)
        // This test catches FK/PK argument swap bugs in resolve_fk_type_to_pk
        let customers = make_table(
            "customers",
            vec![make_column("id", IngestedType::Int64)],
            Some("id"),
        );
        let mut orders = make_table(
            "orders",
            vec![
                make_column("id", IngestedType::Int64),
                make_column("customer_id", IngestedType::Int32),
            ],
            Some("id"),
        );
        orders.fkeys.push(FKeySpec {
            src_column: "customer_id".to_string(),
            dst_table: "customers".to_string(),
            forward: None,
            reverse: None,
            bidirectional: true,
        });

        let config = GraphConfig {
            dataset_name: "test".to_string(),
            tables: vec![customers, orders],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::Lossless,
        };

        let resolved = config.resolve_fk_types();
        let resolved_orders = resolved.tables.iter().find(|t| t.name == "orders").unwrap();
        let resolved_col = resolved_orders
            .columns
            .iter()
            .find(|c| c.name == "customer_id")
            .unwrap();

        // Int32 FK should be resolved to Int64 to match PK
        assert_eq!(resolved_col.ingested_dtype, IngestedType::Int64);
    }

    #[test]
    fn test_resolve_fk_types_strict_mode_no_resolution() {
        // In Strict mode, even Safe coercions should NOT resolve FK types
        // Types must match exactly in Strict mode
        let customers = make_table(
            "customers",
            vec![make_column("id", IngestedType::Int64)],
            Some("id"),
        );
        let mut orders = make_table(
            "orders",
            vec![
                make_column("id", IngestedType::Int64),
                make_column("customer_id", IngestedType::Int32),
            ],
            Some("id"),
        );
        orders.fkeys.push(FKeySpec {
            src_column: "customer_id".to_string(),
            dst_table: "customers".to_string(),
            forward: None,
            reverse: None,
            bidirectional: true,
        });

        let config = GraphConfig {
            dataset_name: "test".to_string(),
            tables: vec![customers, orders],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::Strict,
        };

        let resolved = config.resolve_fk_types();
        let resolved_orders = resolved.tables.iter().find(|t| t.name == "orders").unwrap();
        let resolved_col = resolved_orders
            .columns
            .iter()
            .find(|c| c.name == "customer_id")
            .unwrap();

        // In Strict mode, FK should NOT be resolved - stays Int32
        assert_eq!(resolved_col.ingested_dtype, IngestedType::Int32);
    }

    #[test]
    fn test_resolve_fk_type_to_pk_lossy_returns_none() {
        // Lossy coercions should never resolve, even if called directly
        // (In practice, validate() rejects Lossy before resolve is called)
        use super::resolve_fk_type_to_pk;

        // Float64→Int64 is Lossy
        let result = resolve_fk_type_to_pk(
            &IngestedType::Float64,
            &IngestedType::Int64,
            CoercionMode::Lossless,
            None,
        );
        assert!(
            result.is_none(),
            "Lossy coercion Float64→Int64 should not resolve"
        );

        // Int64→Int32 is Lossy
        let result = resolve_fk_type_to_pk(
            &IngestedType::Int64,
            &IngestedType::Int32,
            CoercionMode::Lossless,
            None,
        );
        assert!(
            result.is_none(),
            "Lossy coercion Int64→Int32 should not resolve"
        );
    }

    fn make_multicat_column(name: &str, dtype: IngestedType, delim: Option<&str>) -> ColumnSpec {
        ColumnSpec {
            name: name.to_string(),
            source_dtype: None,
            ingested_dtype: dtype,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: None,
            precision: None,
            scale: None,
            semantic_type: Some(SemanticType::Multicategorical),
            multicategorical_delimiter: delim.map(str::to_string),
        }
    }

    #[test]
    fn normalize_flips_string_multicat_with_delim_to_stringarray() {
        let col = make_multicat_column("tags", IngestedType::String, Some(","));
        let mut cfg = GraphConfig {
            dataset_name: "t".into(),
            tables: vec![make_table("items", vec![col], None)],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::default(),
        };
        cfg.normalize_multicat_pseudo_list();
        assert_eq!(
            cfg.tables[0].columns[0].ingested_dtype,
            IngestedType::StringArray,
        );
        // Delim is preserved on the spec — it drives the access-layer split
        // at read time even after the dtype flip.
        assert_eq!(
            cfg.tables[0].columns[0]
                .multicategorical_delimiter
                .as_deref(),
            Some(","),
        );
    }

    #[test]
    fn normalize_leaves_string_multicat_without_delim_alone() {
        let col = make_multicat_column("tags", IngestedType::String, None);
        let mut cfg = GraphConfig {
            dataset_name: "t".into(),
            tables: vec![make_table("items", vec![col], None)],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::default(),
        };
        cfg.normalize_multicat_pseudo_list();
        assert_eq!(
            cfg.tables[0].columns[0].ingested_dtype,
            IngestedType::String,
            "no delim ⇒ single-token-per-row, scalar string encoding stays",
        );
    }

    #[test]
    fn normalize_leaves_native_stringarray_alone() {
        let col = make_multicat_column("tags", IngestedType::StringArray, None);
        let mut cfg = GraphConfig {
            dataset_name: "t".into(),
            tables: vec![make_table("items", vec![col], None)],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::default(),
        };
        cfg.normalize_multicat_pseudo_list();
        assert_eq!(
            cfg.tables[0].columns[0].ingested_dtype,
            IngestedType::StringArray,
        );
    }

    #[test]
    fn normalize_leaves_non_multicat_columns_alone() {
        // Non-multicat semantic type with an irrelevant delimiter should not flip.
        let mut col = make_column("x", IngestedType::String);
        col.semantic_type = Some(SemanticType::Text);
        col.multicategorical_delimiter = Some(",".into());
        let mut cfg = GraphConfig {
            dataset_name: "t".into(),
            tables: vec![make_table("items", vec![col], None)],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::default(),
        };
        cfg.normalize_multicat_pseudo_list();
        assert_eq!(
            cfg.tables[0].columns[0].ingested_dtype,
            IngestedType::String,
        );
    }

    #[test]
    fn normalize_is_idempotent() {
        let col = make_multicat_column("tags", IngestedType::String, Some(","));
        let mut cfg = GraphConfig {
            dataset_name: "t".into(),
            tables: vec![make_table("items", vec![col], None)],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::default(),
        };
        cfg.normalize_multicat_pseudo_list();
        let after_first = cfg.tables[0].columns[0].ingested_dtype.clone();
        cfg.normalize_multicat_pseudo_list();
        let after_second = cfg.tables[0].columns[0].ingested_dtype.clone();
        assert_eq!(after_first, after_second, "pass should be idempotent");
        assert_eq!(after_first, IngestedType::StringArray);
    }

    fn make_latlong_column(
        name: &str,
        dtype: IngestedType,
        fixed_len: Option<usize>,
    ) -> ColumnSpec {
        let mut col = make_column(name, dtype);
        col.semantic_type = Some(SemanticType::Latlong);
        col.fixed_len = fixed_len;
        col
    }

    #[test]
    fn normalize_latlong_forces_fixed_len_two() {
        let col = make_latlong_column("location", IngestedType::Float32Array, None);
        let mut cfg = GraphConfig {
            dataset_name: "t".into(),
            tables: vec![make_table("items", vec![col], None)],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::default(),
        };

        cfg.normalize_multicat_pseudo_list();

        assert_eq!(cfg.tables[0].columns[0].fixed_len, Some(2));
        cfg.validate().expect("normalized latlong should validate");
    }

    #[test]
    fn validate_latlong_requires_numeric_array() {
        let col = make_latlong_column("location", IngestedType::String, Some(2));
        let cfg = GraphConfig {
            dataset_name: "t".into(),
            tables: vec![make_table("items", vec![col], None)],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::default(),
        };

        let err = cfg.validate().expect_err("string latlong should fail");
        assert!(
            err.to_string()
                .contains("table items column location latlong requires a numeric array dtype"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_latlong_requires_fixed_len_two() {
        let col = make_latlong_column("location", IngestedType::Float32Array, Some(3));
        let cfg = GraphConfig {
            dataset_name: "t".into(),
            tables: vec![make_table("items", vec![col], None)],
            rowpack: None,
            num_shards: None,
            edge_format: None,
            coercion_mode: CoercionMode::default(),
        };

        let err = cfg.validate().expect_err("length-3 latlong should fail");
        assert!(
            err.to_string()
                .contains("table items column location latlong requires fixed_len=2"),
            "unexpected error: {err}"
        );
    }
}
