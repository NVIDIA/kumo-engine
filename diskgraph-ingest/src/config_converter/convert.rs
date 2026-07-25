//! Conversion from Kumo GraphConfig to DiskGraph GraphConfig.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use serde::Deserialize;

use crate::errors::ConvertError;
use crate::graph_config::{
    ColumnSpec, DataSource, DirectionSpec, EdgeTableSpec, FKeySpec, ForeignKeySpec, GraphConfig,
    IngestedType, SemanticType, TableSpec, TimeSpec,
};

use super::mapping::{map_kumo_dtype, map_kumo_stype, map_timestamp_unit};
use super::types::{KumoForeignKey, KumoGraphConfig, KumoTable};
use crate::io::expand_source_paths;

/// Time column info for a table, used for cross-table sort key lookups.
struct TableTimeInfo {
    column: String,
    unit: String,
}

/// Options for the conversion.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Base path for data persist directory.
    /// e.g., "s3://kumo-prod-dataplane-bucket/demo/prod/data-persist-dir"
    pub data_persist_dir: String,
    /// Dataset name for the output GraphConfig.
    pub dataset_name: String,
    /// Per-table data versions, keyed by table_name.
    /// When provided, these override `kumo_table.data_version` for path building.
    pub table_data_versions: Option<HashMap<String, String>>,
    /// Override base path for source data. When set, table paths become
    /// `{source_paths_base}/{source_name}/` instead of the fpv3 pattern.
    /// Lower priority than `source_paths`.
    pub source_paths_base: Option<String>,
    /// Per-table file-path overrides, keyed by table_name.
    /// Used when a table has no direct source override.
    pub source_paths: Option<HashMap<String, String>>,
    /// Per-table direct source overrides, keyed by table_name.
    /// A matched source takes priority and bypasses file-path expansion.
    pub source_specs: Option<HashMap<String, DirectSourceSpec>>,
}

/// A direct source and the estimated logical bytes it will produce.
#[derive(Debug, Clone, Deserialize)]
pub struct DirectSourceSpec {
    pub source: DataSource,
    pub source_size: u64,
}

/// Convert a Kumo GraphConfig to a DiskGraph GraphConfig.
pub fn convert_kumo_to_diskgraph(
    kumo_config: &KumoGraphConfig,
    options: &ConvertOptions,
) -> Result<GraphConfig, ConvertError> {
    // Build a map of table_name -> time info so FK sort keys can reference
    // the destination table's create_time (matching diskgraph-infer behavior).
    let mut time_info_map: HashMap<String, TableTimeInfo> = HashMap::new();
    for kumo_table in &kumo_config.tables {
        if let Some(tc) = &kumo_table.time_col {
            let unit = tc
                .timestamp_unit
                .as_deref()
                .map(map_time_unit_for_spec)
                .unwrap_or_else(|| "s".to_string());
            let table_name = kumo_table.table_name.clone();
            time_info_map.insert(
                table_name,
                TableTimeInfo {
                    column: tc.name.clone(),
                    unit,
                },
            );
        }
    }

    let mut tables = Vec::new();

    for kumo_table in &kumo_config.tables {
        let table_spec = convert_table(kumo_table, options, &time_info_map)?;
        tables.push(table_spec);
    }

    let mut config = GraphConfig {
        dataset_name: options.dataset_name.clone(),
        tables,
        rowpack: None,
        num_shards: None,
        edge_format: None,
        coercion_mode: Default::default(),
    };

    // Best-effort: detect timestamp formats, multicat delimiters, and estimate num_shards.
    // These require I/O (reading parquet files, listing cloud objects) and may fail
    // when files aren't accessible (e.g., in tests). The config is still valid without them.
    //
    // Run all converter inference in one separate OS thread so this works even
    // when called from inside a tokio runtime (e.g., validate-s3-ingest). The I/O
    // helpers create their own tokio runtimes internally, which panics if nested
    // inside an existing one.
    let inference_err: Option<ConvertError> = std::thread::scope(|s| {
        let cfg = &mut config;
        s.spawn(move || -> Option<ConvertError> {
            let transfer = match crate::io::build_input_transfer(cfg) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!("could not build cloud transfer: {e}");
                    return None;
                }
            };

            // Hard errors (e.g. time_col format detection failure) are
            // propagated. Soft warnings for non-critical columns are
            // logged via tracing::warn! inside detect_timestamp_formats.
            if let Err(e) = super::inference::detect_timestamp_formats(cfg, &transfer) {
                return Some(ConvertError::InferenceError(format!("{e:#}")));
            }

            if let Err(e) = super::inference::detect_multicategorical_delimiters(cfg, &transfer) {
                return Some(e);
            }

            // Validate any user-declared multicategorical delimiters against
            // the source data. Post-#28478 a wrong delimiter bakes into the
            // rowpack on disk because ingest pre-splits, so a mistake here is
            // irreversible without re-ingest. Surface mismatches loudly with
            // column + sample-row context. The check piggybacks on the same
            // bounded sampling budget the auto-detect path uses.
            if let Err(e) = super::inference::validate_explicit_multicat_delimiters(cfg, &transfer)
            {
                return Some(e);
            }

            // After delimiters are finalized + validated, normalize the on-disk
            // shape of multicategorical "pseudo list" columns: a `String`
            // source with a delimiter becomes a `StringArray` on disk so
            // downstream consumers (TT-gen, infer, tensor mapper) only ever see
            // native lists. The translation happens at read time in
            // `RowAccess::string_list_view`, driven by
            // `RowAccessHints.multicategorical_delimiter`. The same pass is
            // also called unconditionally at ingest entry
            // (`ingest_with_stats_from_source`); applying it here too keeps
            // `convert_kumo_to_diskgraph`'s returned config in normalized form
            // for callers that inspect it before ingest. The pass is
            // idempotent. See kumo-ai/kumo#28478.
            cfg.normalize_multicat_pseudo_list();

            let total_bytes: u64 = cfg.tables.iter().filter_map(|t| t.source_size).sum();
            if total_bytes > 0 {
                cfg.num_shards = Some(super::inference::estimate_num_shards(total_bytes));
            }
            None
        })
        .join()
        .ok()
        .flatten()
    });

    if let Some(e) = inference_err {
        return Err(e);
    }

    config.normalize_multicat_pseudo_list();
    Ok(config)
}

/// Convert a single Kumo table to a DiskGraph TableSpec.
fn convert_table(
    kumo_table: &KumoTable,
    options: &ConvertOptions,
    time_info_map: &HashMap<String, TableTimeInfo>,
) -> Result<TableSpec, ConvertError> {
    // Resolve data version (used for path building and kumo stats output)
    let data_version = resolve_data_version(kumo_table, options)?;

    let (source, source_size) = if let Some(spec) = options
        .source_specs
        .as_ref()
        .and_then(|sources| sources.get(&kumo_table.table_name))
    {
        (spec.source.clone(), Some(spec.source_size))
    } else {
        let dir_paths = build_table_paths(kumo_table, options)?;
        let (source, source_size) = expand_source_paths(dir_paths, &kumo_table.table_name)?;
        (source, Some(source_size))
    };

    // Build columns from all sources (pkey, fkeys, cols, time_col)
    let columns = build_columns(kumo_table)?;

    if columns.is_empty() {
        return Err(ConvertError::NoColumns {
            table: kumo_table.table_name.clone(),
        });
    }

    // Get primary key name
    let primary_key = kumo_table.pkey.as_ref().map(|pk| pk.name.clone());

    // Get create_time from time_col
    let create_time = kumo_table.time_col.as_ref().map(|tc| tc.name.clone());

    // Get end_time from end_time_col
    let end_time = kumo_table.end_time_col.as_ref().map(|tc| tc.name.clone());

    // Get timestamp unit for time-based sorting
    let time_unit = kumo_table
        .time_col
        .as_ref()
        .and_then(|tc| tc.timestamp_unit.as_deref())
        .map(map_time_unit_for_spec)
        .unwrap_or_else(|| "s".to_string()); // Default to seconds

    // Detect if this should be an edge table
    let (edge_table, fkeys) = detect_edge_table_or_fkeys(
        kumo_table,
        &columns,
        create_time.as_deref(),
        &kumo_table.table_name,
        &time_unit,
        time_info_map,
    );

    Ok(TableSpec {
        name: kumo_table.table_name.clone(),
        columns,
        primary_key,
        create_time,
        end_time,
        weight: None,
        edge_table,
        rowpack: None,
        source,
        source_size,
        fkeys,
        data_version: Some(data_version),
        kumo_connector_id: kumo_table.data_source.clone(),
        kumo_source_table_name: kumo_table
            .source_name
            .clone()
            .or_else(|| Some(kumo_table.table_name.clone())),
        group_by_timestamp: false,
    })
}

/// Resolve the data_version for a table from options or inline config.
fn resolve_data_version(
    kumo_table: &KumoTable,
    options: &ConvertOptions,
) -> Result<String, ConvertError> {
    if let Some(versions) = &options.table_data_versions {
        versions
            .get(&kumo_table.table_name)
            .ok_or_else(|| ConvertError::MissingField {
                table: kumo_table.table_name.clone(),
                field: "table_data_versions entry".to_string(),
            })
            .cloned()
    } else {
        kumo_table
            .data_version
            .as_ref()
            .ok_or_else(|| ConvertError::MissingField {
                table: kumo_table.table_name.clone(),
                field: "data_version".to_string(),
            })
            .cloned()
    }
}

/// Build the source data paths for a table.
///
/// Priority:
/// 1. `source_paths[table_name]` — per-table explicit override
/// 2. `source_paths_base` — `{base}/{source_name}/`
/// 3. Default fpv3 — `{data_persist_dir}/ingest/fpv3.{data_source}.{source_name}/{version}/`
fn build_table_paths(
    kumo_table: &KumoTable,
    options: &ConvertOptions,
) -> Result<Vec<String>, ConvertError> {
    // Priority 1: per-table explicit path
    if let Some(paths_map) = &options.source_paths {
        if let Some(path) = paths_map.get(&kumo_table.table_name) {
            let mut p = path.clone();
            if !p.ends_with('/') {
                p.push('/');
            }
            return Ok(vec![p]);
        }
    }

    // source_name defaults to table_name if not specified
    let source_name = kumo_table
        .source_name
        .as_ref()
        .unwrap_or(&kumo_table.table_name);

    // Priority 2: base path override — {source_paths_base}/{source_name}/
    if let Some(base) = &options.source_paths_base {
        let base = base.trim_end_matches('/');
        let path = format!("{}/{}/", base, source_name);
        return Ok(vec![path]);
    }

    // Priority 3: default fpv3 pattern
    let data_source =
        kumo_table
            .data_source
            .as_ref()
            .ok_or_else(|| ConvertError::MissingField {
                table: kumo_table.table_name.clone(),
                field: "data_source".to_string(),
            })?;

    let data_version = resolve_data_version(kumo_table, options)?;

    // Build fpv3 path: {data_persist_dir}/ingest/fpv3.{data_source}.{source_name}/{data_version}/
    let base = options.data_persist_dir.trim_end_matches('/');
    let path = format!(
        "{}/ingest/fpv3.{}.{}/{}/",
        base, data_source, source_name, data_version
    );

    Ok(vec![path])
}

/// Build ColumnSpec list from all column sources in a Kumo table.
fn build_columns(kumo_table: &KumoTable) -> Result<Vec<ColumnSpec>, ConvertError> {
    // A column name can appear in `cols`, `time_col`, and `end_time_col`
    // (pkey / fkeys don't carry `timestamp_format`). If two positions declare
    // disagreeing non-empty formats, error out rather than letting iteration
    // order pick a winner.
    check_timestamp_format_consistency(kumo_table)?;

    let mut columns = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();

    // Helper to add a column if not already seen.
    // `multicategorical_delimiter` is only meaningful for regular columns with
    // stype="multicategorical" + dtype="string"; other callers pass None.
    let mut add_column = |name: &str,
                          dtype: Option<&str>,
                          stype: Option<&str>,
                          timestamp_unit: Option<&str>,
                          timestamp_format: Option<&str>,
                          multicategorical_delimiter: Option<&str>|
     -> Result<(), ConvertError> {
        if seen_names.contains(name) {
            return Ok(());
        }
        seen_names.insert(name.to_string());

        let (timestamp_unit, timestamp_format) =
            normalize_timestamp_hints(timestamp_unit, timestamp_format);

        // Reject malformed formats here; without this, `"%Q"` etc. would
        // silently null every row at ingest time.
        if let Some(fmt) = timestamp_format {
            if !is_valid_chrono_format(fmt) {
                return Err(ConvertError::InvalidTimestampFormat {
                    table: kumo_table.table_name.clone(),
                    column: name.to_string(),
                    format: fmt.to_string(),
                });
            }
        }

        let ingested_dtype = match dtype {
            Some(dt) => map_kumo_dtype(dt).map_err(|unknown| ConvertError::UnknownDtype {
                table: kumo_table.table_name.clone(),
                column: name.to_string(),
                dtype: unknown.to_string(),
            })?,
            None => IngestedType::String,
        };

        let ts_unit = timestamp_unit.and_then(map_timestamp_unit);
        let semantic_type = stype.and_then(map_kumo_stype);

        // Error on dtype="string" + stype="sequence": the data is likely a
        // JSON-encoded float array that rowpack can't parse from a string column.
        // The Kumo config should use dtype="floatlist" instead.
        if ingested_dtype == IngestedType::String && semantic_type == Some(SemanticType::Sequence) {
            return Err(ConvertError::StringSequenceUnsupported {
                table: kumo_table.table_name.clone(),
                column: name.to_string(),
                dtype: dtype.unwrap_or("(none)").to_string(),
            });
        }

        // Multicategorical columns are accepted for a specific set of ingested
        // dtypes; see `ColumnStatsCollector::from_column_type` for the full
        // input-kind matrix. Unsupported combinations (e.g. a single int64
        // scalar with stype=multicategorical) are rejected here with a
        // clear error rather than silently degrading to Skip.
        if semantic_type == Some(SemanticType::Multicategorical) {
            let ok = matches!(
                ingested_dtype,
                IngestedType::String
                    | IngestedType::StringArray
                    | IngestedType::Int64Array
                    | IngestedType::Float16Array
                    | IngestedType::Float32Array
                    | IngestedType::Float64Array
            );
            if !ok {
                return Err(ConvertError::MulticategoricalOnUnsupportedInput {
                    table: kumo_table.table_name.clone(),
                    column: name.to_string(),
                    dtype: ingested_dtype.clone(),
                });
            }
        }

        if semantic_type == Some(SemanticType::Latlong) {
            let ok = matches!(
                ingested_dtype,
                IngestedType::Float16Array
                    | IngestedType::Float32Array
                    | IngestedType::Float64Array
                    | IngestedType::Int8Array
                    | IngestedType::Int16Array
                    | IngestedType::Int32Array
                    | IngestedType::Int64Array
            );
            if !ok {
                return Err(ConvertError::LatlongOnUnsupportedInput {
                    table: kumo_table.table_name.clone(),
                    column: name.to_string(),
                    dtype: ingested_dtype.clone(),
                });
            }
        }

        columns.push(ColumnSpec {
            name: name.to_string(),
            source_dtype: dtype.map(|s| s.to_string()),
            ingested_dtype,
            fixed_len: (semantic_type == Some(SemanticType::Latlong)).then_some(2),
            // User-declared format wins: `detect_timestamp_formats` skips
            // columns where this is `Some`, and the rowpack parser uses it
            // via `RowAccessHints.timestamp_format`.
            timestamp_format: timestamp_format.map(|s| s.to_string()),
            timestamp_unit: ts_unit,
            precision: None,
            scale: None,
            semantic_type,
            // Detection (`detect_multicategorical_delimiters`) only runs when
            // the user did not supply an explicit delimiter on the Kumo
            // config. Pre-populated here so the explicit path is preserved
            // even for columns that aren't `String + Multicategorical`
            // (harmless for other shapes; the collector ignores the field).
            multicategorical_delimiter: multicategorical_delimiter.map(str::to_string),
        });

        Ok(())
    };

    // Add primary key column
    if let Some(pkey) = &kumo_table.pkey {
        add_column(
            &pkey.name,
            pkey.dtype.as_deref(),
            pkey.stype.as_deref(),
            None,
            None,
            None,
        )?;
    }

    // Add foreign key columns
    for fkey in &kumo_table.fkeys {
        add_column(
            &fkey.name,
            fkey.dtype.as_deref(),
            fkey.stype.as_deref(),
            None,
            None,
            None,
        )?;
    }

    // Add regular columns
    for col in &kumo_table.cols {
        add_column(
            &col.name,
            col.dtype.as_deref(),
            col.stype.as_deref(),
            col.timestamp_unit.as_deref(),
            col.timestamp_format.as_deref(),
            col.multicategorical_delimiter.as_deref(),
        )?;
    }

    // Add time column
    if let Some(time_col) = &kumo_table.time_col {
        add_column(
            &time_col.name,
            time_col.dtype.as_deref(),
            time_col.stype.as_deref(),
            time_col.timestamp_unit.as_deref(),
            time_col.timestamp_format.as_deref(),
            None,
        )?;
    }

    // Add end time column
    if let Some(end_time_col) = &kumo_table.end_time_col {
        add_column(
            &end_time_col.name,
            end_time_col.dtype.as_deref(),
            end_time_col.stype.as_deref(),
            end_time_col.timestamp_unit.as_deref(),
            end_time_col.timestamp_format.as_deref(),
            None,
        )?;
    }

    Ok(columns)
}

/// Detect if a table should be an edge table or have fkeys.
///
/// Edge table criteria (same as diskgraph-infer):
/// - Exactly 2 foreign keys
/// - All other columns are only create_time or weight-like (numeric)
///
/// Sort key direction convention (matching diskgraph-infer):
/// - FK forward: timestamp from **destination** table's create_time
/// - FK reverse: timestamp from **source** (current) table's create_time
/// - Edge table: timestamp from edge table's own create_time (both directions)
fn detect_edge_table_or_fkeys(
    kumo_table: &KumoTable,
    columns: &[ColumnSpec],
    create_time: Option<&str>,
    table_name: &str,
    time_unit: &str,
    time_info_map: &HashMap<String, TableTimeInfo>,
) -> (Option<EdgeTableSpec>, Vec<FKeySpec>) {
    let fkeys = &kumo_table.fkeys;

    // Build direction spec for the current (source) table's create_time.
    // Used for: FK reverse direction, edge table both directions.
    let src_direction_spec = create_time.map(|col| DirectionSpec {
        time: Some(TimeSpec {
            table: table_name.to_string(),
            column: col.to_string(),
            bitwidth: None,
            anchor_time: None,
            unit: time_unit.to_string(),
        }),
        weight: None,
    });

    // Check if this should be an edge table
    if should_be_edge_table(fkeys, columns, create_time) {
        // Create EdgeTableSpec — edge tables use their own create_time for both directions
        let src_fk = &fkeys[0];
        let dst_fk = &fkeys[1];

        let edge_table = EdgeTableSpec {
            src_fk: ForeignKeySpec {
                edge_column: src_fk.name.clone(),
                ref_table: strip_namespace(&src_fk.dst_table),
            },
            dst_fk: ForeignKeySpec {
                edge_column: dst_fk.name.clone(),
                ref_table: strip_namespace(&dst_fk.dst_table),
            },
            forward: src_direction_spec.clone(),
            reverse: src_direction_spec,
        };

        (Some(edge_table), Vec::new())
    } else {
        // Create FKeySpecs with correct per-direction sort keys:
        // - forward: use destination table's create_time
        // - reverse: use source (current) table's create_time
        let fkey_specs: Vec<FKeySpec> = fkeys
            .iter()
            .map(|fk| {
                let dst_table_name = strip_namespace(&fk.dst_table);
                let forward_spec = time_info_map
                    .get(&dst_table_name)
                    .map(|info| DirectionSpec {
                        time: Some(TimeSpec {
                            table: dst_table_name.clone(),
                            column: info.column.clone(),
                            bitwidth: None,
                            anchor_time: None,
                            unit: info.unit.clone(),
                        }),
                        weight: None,
                    });
                FKeySpec {
                    src_column: fk.name.clone(),
                    dst_table: dst_table_name,
                    forward: forward_spec,
                    reverse: src_direction_spec.clone(),
                    bidirectional: true,
                }
            })
            .collect();

        (None, fkey_specs)
    }
}

/// Check if a table should be treated as an edge table.
///
/// Always returns false — the config converter should never auto-classify
/// tables as edge tables. Edge table detection is left to diskgraph-infer.
fn should_be_edge_table(
    _fkeys: &[KumoForeignKey],
    _columns: &[ColumnSpec],
    _create_time: Option<&str>,
) -> bool {
    false
}

/// Strip namespace prefix from a table name.
/// e.g., "datasource.tablename" -> "tablename"
fn strip_namespace(table_name: &str) -> String {
    table_name
        .rsplit('.')
        .next()
        .unwrap_or(table_name)
        .to_string()
}

/// Reject two positions declaring disagreeing non-empty `timestamp_format`
/// for the same column. Empty strings and `None` are treated as "unset".
fn check_timestamp_format_consistency(kumo_table: &KumoTable) -> Result<(), ConvertError> {
    // First non-empty value per column name is the reference for comparison.
    let mut first_format: HashMap<String, String> = HashMap::new();

    let mut check = |name: &str, fmt: Option<&str>| -> Result<(), ConvertError> {
        let Some(fmt) = normalize_ts_format(fmt) else {
            return Ok(());
        };
        if let Some(prev) = first_format.get(name) {
            if prev != fmt {
                return Err(ConvertError::TimestampFormatConflict {
                    table: kumo_table.table_name.clone(),
                    column: name.to_string(),
                    first: prev.clone(),
                    second: fmt.to_string(),
                });
            }
        } else {
            first_format.insert(name.to_string(), fmt.to_string());
        }
        Ok(())
    };

    // pkey / fkeys don't carry a `timestamp_format` field today, so only
    // `cols`, `time_col`, and `end_time_col` participate in the check.
    for col in &kumo_table.cols {
        check(&col.name, col.timestamp_format.as_deref())?;
    }
    if let Some(tc) = &kumo_table.time_col {
        check(&tc.name, tc.timestamp_format.as_deref())?;
    }
    if let Some(etc) = &kumo_table.end_time_col {
        check(&etc.name, etc.timestamp_format.as_deref())?;
    }
    Ok(())
}

/// Treat `Some("")` as "unset" (returns `None`). Callers can then use
/// `is_none()` to mean "no user-supplied hint".
fn normalize_ts_format(fmt: Option<&str>) -> Option<&str> {
    fmt.filter(|s| !s.is_empty())
}

/// Kumo historically routes `TimestampUnit` values through
/// `metadata.TS_FORMAT`, which reaches this converter as `timestamp_format`.
/// If that value is a known unit shorthand, consume it as a unit hint instead
/// of validating it as a chrono format string.
fn normalize_timestamp_hints<'a>(
    timestamp_unit: Option<&'a str>,
    timestamp_format: Option<&'a str>,
) -> (Option<&'a str>, Option<&'a str>) {
    let timestamp_format = normalize_ts_format(timestamp_format);
    if let Some(fmt) = timestamp_format {
        if map_timestamp_unit(fmt).is_some() {
            return (timestamp_unit.or(Some(fmt)), None);
        }
    }
    (timestamp_unit, timestamp_format)
}

/// Reject malformed strftime specifiers (e.g. `"%Q"`) and literal-only
/// strings with no date/time directive (e.g. Java syntax `"yyyy-MM-dd"`).
/// `Item::Error` flags the first; absence of any `Numeric`/`Fixed` item
/// flags the second — otherwise such strings silently null every row.
pub(super) fn is_valid_chrono_format(format: &str) -> bool {
    use chrono::format::{Item, StrftimeItems};
    let mut saw_specifier = false;
    for item in StrftimeItems::new(format) {
        match item {
            Item::Error => return false,
            Item::Numeric(_, _) | Item::Fixed(_) => saw_specifier = true,
            _ => {}
        }
    }
    saw_specifier
}

/// Map kumo timestamp unit to TimeSpec unit string.
fn map_time_unit_for_spec(unit: &str) -> String {
    match unit {
        "s" | "seconds" | "sec" => "s".to_string(),
        "ms" | "millis" | "milliseconds" => "ms".to_string(),
        "us" | "micros" | "microseconds" => "us".to_string(),
        "ns" | "nanos" | "nanoseconds" => "ns".to_string(),
        // For date types, use "day"
        "day" | "days" => "day".to_string(),
        _ => "s".to_string(), // Default to seconds
    }
}
