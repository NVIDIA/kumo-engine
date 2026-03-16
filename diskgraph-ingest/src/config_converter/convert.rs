//! Conversion from Kumo GraphConfig to DiskGraph GraphConfig.

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::errors::ConvertError;
use crate::graph_config::{
    ColumnSpec, DataSource, DirectionSpec, EdgeTableSpec, FKeySpec, ForeignKeySpec, GraphConfig,
    IngestedType, SemanticType, TableSpec, TimeSpec,
};

use super::mapping::{map_kumo_dtype, map_kumo_stype, map_timestamp_unit};
use super::types::{KumoForeignKey, KumoGraphConfig, KumoTable};

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

    // Best-effort: detect timestamp formats and estimate num_shards.
    // These require I/O (reading parquet files, listing cloud objects) and may fail
    // when files aren't accessible (e.g., in tests). The config is still valid without them.
    //
    // Run on a separate OS thread so this works even when called from inside a
    // tokio runtime (e.g., validate-s3-ingest). The I/O helpers create their own
    // tokio runtimes internally, which panics if nested inside an existing one.
    std::thread::scope(|s| {
        let cfg = &mut config;
        s.spawn(move || {
            let transfer = match crate::io::build_input_transfer(cfg) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Warning: could not build cloud transfer: {e}");
                    return;
                }
            };

            if let Err(e) = super::inference::detect_timestamp_formats(cfg, &transfer) {
                eprintln!("Warning: timestamp format detection failed: {e}");
            }

            match super::inference::compute_total_bytes(cfg, &transfer) {
                Ok(total_bytes) if total_bytes > 0 => {
                    cfg.num_shards =
                        Some(super::inference::estimate_num_shards(total_bytes));
                }
                Err(e) => {
                    eprintln!("Warning: num_shards estimation failed: {e}");
                }
                _ => {}
            }
        })
        .join()
        .ok();
    });

    Ok(config)
}

/// Convert a single Kumo table to a DiskGraph TableSpec.
fn convert_table(
    kumo_table: &KumoTable,
    options: &ConvertOptions,
    time_info_map: &HashMap<String, TableTimeInfo>,
) -> Result<TableSpec, ConvertError> {
    // Build the S3 path
    let paths = build_table_paths(kumo_table, options)?;

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
        source: DataSource::Parquet { paths },
        fkeys,
    })
}

/// Build the S3 paths for a table using the fpv3 pattern.
fn build_table_paths(
    kumo_table: &KumoTable,
    options: &ConvertOptions,
) -> Result<Vec<String>, ConvertError> {
    let data_source =
        kumo_table
            .data_source
            .as_ref()
            .ok_or_else(|| ConvertError::MissingField {
                table: kumo_table.table_name.clone(),
                field: "data_source".to_string(),
            })?;

    // source_name defaults to table_name if not specified
    let source_name = kumo_table
        .source_name
        .as_ref()
        .unwrap_or(&kumo_table.table_name);

    // Look up data_version: prefer table_data_versions map, fall back to kumo_table.data_version.
    let data_version = if let Some(versions) = &options.table_data_versions {
        versions
            .get(&kumo_table.table_name)
            .ok_or_else(|| ConvertError::MissingField {
                table: kumo_table.table_name.clone(),
                field: "table_data_versions entry".to_string(),
            })?
            .clone()
    } else {
        kumo_table
            .data_version
            .as_ref()
            .ok_or_else(|| ConvertError::MissingField {
                table: kumo_table.table_name.clone(),
                field: "data_version".to_string(),
            })?
            .clone()
    };

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
    let mut columns = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();

    // Helper to add a column if not already seen
    let mut add_column = |name: &str,
                          dtype: Option<&str>,
                          stype: Option<&str>,
                          timestamp_unit: Option<&str>|
     -> Result<(), ConvertError> {
        if seen_names.contains(name) {
            return Ok(());
        }
        seen_names.insert(name.to_string());

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
        if ingested_dtype == IngestedType::String
            && semantic_type == Some(SemanticType::Sequence)
        {
            return Err(ConvertError::StringSequenceUnsupported {
                table: kumo_table.table_name.clone(),
                column: name.to_string(),
                dtype: dtype.unwrap_or("(none)").to_string(),
            });
        }

        // Error on stype="multicategorical": diskgraph does not yet support
        // column stats collection or encoding for multicategorical columns.
        if semantic_type == Some(SemanticType::Multicategorical) {
            return Err(ConvertError::MulticategoricalUnsupported {
                table: kumo_table.table_name.clone(),
                column: name.to_string(),
            });
        }

        columns.push(ColumnSpec {
            name: name.to_string(),
            source_dtype: dtype.map(|s| s.to_string()),
            ingested_dtype,
            fixed_len: None,
            timestamp_format: None,
            timestamp_unit: ts_unit,
            precision: None,
            scale: None,
            semantic_type,
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
        )?;
    }

    // Add foreign key columns
    for fkey in &kumo_table.fkeys {
        add_column(
            &fkey.name,
            fkey.dtype.as_deref(),
            fkey.stype.as_deref(),
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
        )?;
    }

    // Add time column
    if let Some(time_col) = &kumo_table.time_col {
        add_column(
            &time_col.name,
            time_col.dtype.as_deref(),
            time_col.stype.as_deref(),
            time_col.timestamp_unit.as_deref(),
        )?;
    }

    // Add end time column
    if let Some(end_time_col) = &kumo_table.end_time_col {
        add_column(
            &end_time_col.name,
            end_time_col.dtype.as_deref(),
            end_time_col.stype.as_deref(),
            end_time_col.timestamp_unit.as_deref(),
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
