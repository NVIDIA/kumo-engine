use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use diskgraph_ingest::databricks::{
    DatabricksHelper, DatabricksQueryStream, QueryMetrics, QueryStreamOptions,
};
use diskgraph_ingest::{
    ingest_with_stats_from_source, inspect_source_schema, Dest, GraphConfig, IngestOptions,
    IngestStats,
};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const MINIMUM_PIPELINE_RATIO: f64 = 0.85;

#[derive(Debug, Parser)]
#[command(
    about = "Benchmark Databricks download/decode, local Encode, and combined streaming"
)]
struct Args {
    /// Databricks graph config used for download/decode and combined modes.
    #[arg(long)]
    databricks_config: PathBuf,
    /// Equivalent local Parquet/CSV graph config used for encode-only mode.
    #[arg(long)]
    local_config: PathBuf,
    /// Optional mode-0600 Databricks unified-authentication config file.
    #[arg(long)]
    config_file: Option<PathBuf>,
    /// Parent directory for the two benchmark graph outputs.
    #[arg(long)]
    output_dir: PathBuf,
    /// Permit loopback HTTP links from a synthetic local Cloud Fetch fixture.
    #[arg(long)]
    allow_http_loopback: bool,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    rows: u64,
    download_decode_seconds: f64,
    encode_only_seconds: f64,
    combined_steady_seconds: f64,
    query_startup_seconds: f64,
    download_decode_rows_per_second: f64,
    encode_rows_per_second: f64,
    combined_rows_per_second: f64,
    slower_component_rows_per_second: f64,
    combined_to_bottleneck_ratio: f64,
    databricks_wire_bytes: u64,
    databricks_decoded_ipc_bytes: u64,
    memory_high_water_bytes: u64,
    passed: bool,
}

fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs_f64().max(1e-9)
}

fn load_config(path: &Path) -> Result<GraphConfig> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let config = serde_json::from_slice::<GraphConfig>(&bytes)
        .with_context(|| format!("parse {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

fn decoded_row_multiset(
    table: &diskgraph_index::graph::TableReader,
) -> Result<Vec<Option<String>>> {
    let mut rows = Vec::with_capacity(table.node_count() as usize);
    for offset in 0..table.node_count() {
        let node_id = table
            .node_id_from_offset(offset)
            .ok_or_else(|| anyhow!("benchmark node offset is invalid"))?;
        let view = table.edges_for_node(node_id)?;
        rows.push(
            table
                .decode_node_data_json(&view)?
                .map(|row| serde_json::to_string(&row))
                .transpose()?,
        );
    }
    rows.sort_unstable();
    Ok(rows)
}

fn decoded_node_identity(
    table: &diskgraph_index::graph::TableReader,
    node_id: u64,
) -> Result<String> {
    let view = table.edges_for_node(node_id)?;
    let row = table
        .decode_node_data_json(&view)?
        .ok_or_else(|| anyhow!("benchmark node has no decodable data"))?;
    if let Some(primary_key) = table.primary_key_map() {
        let value = row.get(&primary_key.column).ok_or_else(|| {
            anyhow!(
                "benchmark node data is missing PK '{}' in table {}",
                primary_key.column,
                table.node_type()
            )
        })?;
        return Ok(format!("pk:{}", serde_json::to_string(value)?));
    }
    Ok(format!("row:{}", serde_json::to_string(&row)?))
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SemanticEdge {
    source_table: String,
    relation: String,
    source_identity: String,
    destination_identity: String,
    sort_key: Option<u64>,
}

fn semantic_edges(
    graph: &diskgraph_index::graph::GraphIndexReader,
) -> Result<Vec<SemanticEdge>> {
    let mut edges = Vec::new();
    for source_table in graph.tables() {
        for offset in 0..source_table.node_count() {
            let source_id = source_table
                .node_id_from_offset(offset)
                .ok_or_else(|| anyhow!("benchmark source node offset is invalid"))?;
            let source_identity = decoded_node_identity(&source_table, source_id)?;
            let view = source_table.edges_for_node(source_id)?;
            let segments = view.relation_segments()?;
            if segments.len() != source_table.relations().len() {
                bail!(
                    "benchmark relation segment count differs for table {}",
                    source_table.node_type()
                );
            }
            for (relation, segment) in source_table.relations().iter().zip(segments) {
                let destination_table = graph.table(&relation.dst_type).ok_or_else(|| {
                    anyhow!(
                        "benchmark relation '{}' is missing destination table {}",
                        relation.name,
                        relation.dst_type
                    )
                })?;
                if segment.layout.entry_bytes > 16 {
                    bail!("benchmark relation entry is too wide to decode");
                }
                for entry in segment.data.chunks_exact(segment.layout.entry_bytes) {
                    let mut bytes = [0u8; 16];
                    bytes[..segment.layout.entry_bytes].copy_from_slice(entry);
                    let packed = u128::from_le_bytes(bytes);
                    let destination_id =
                        (packed & segment.layout.dst_mask as u128) as u64;
                    let destination_identity =
                        decoded_node_identity(&destination_table, destination_id)?;
                    let sort_key = segment.layout.dest_sort_key_bits.map(|bits| {
                        ((packed >> segment.layout.dst_bits as u128)
                            & ((1u128 << bits) - 1)) as u64
                    });
                    edges.push(SemanticEdge {
                        source_table: source_table.node_type().to_string(),
                        relation: relation.name.clone(),
                        source_identity: source_identity.clone(),
                        destination_identity,
                        sort_key,
                    });
                }
            }
        }
    }
    edges.sort_unstable();
    Ok(edges)
}

fn compare_graph_semantics(left_root: &Path, right_root: &Path) -> Result<()> {
    let left = diskgraph_index::graph::GraphIndexReader::open(
        &left_root.join("index_metadata.json"),
        left_root,
    )?;
    let right = diskgraph_index::graph::GraphIndexReader::open(
        &right_root.join("index_metadata.json"),
        right_root,
    )?;
    for left_table in left.tables() {
        let right_table = right.table(left_table.node_type()).ok_or_else(|| {
            anyhow!(
                "benchmark graph is missing table {}",
                left_table.node_type()
            )
        })?;
        let node_count = left_table.node_count();
        if node_count != right_table.node_count() {
            bail!(
                "benchmark PK table row counts differ for {}",
                left_table.node_type()
            );
        }
        if decoded_row_multiset(&left_table)? != decoded_row_multiset(&right_table)? {
            bail!(
                "benchmark node rows differ for table {}",
                left_table.node_type()
            );
        }
        let Some(primary_key) = left_table.primary_key_map() else {
            if right_table.primary_key_map().is_some() {
                bail!(
                    "benchmark PK metadata differs for table {}",
                    left_table.node_type()
                );
            }
            continue;
        };
        let right_primary_key = right_table.primary_key_map().ok_or_else(|| {
            anyhow!(
                "benchmark graph is missing PK map for table {}",
                left_table.node_type()
            )
        })?;
        if right_primary_key.column != primary_key.column {
            bail!(
                "benchmark PK columns differ for table {}",
                left_table.node_type()
            );
        }
        for offset in 0..node_count {
            let node_id = left_table
                .node_id_from_offset(offset)
                .ok_or_else(|| anyhow!("benchmark PK sample offset is invalid"))?;
            let view = left_table.edges_for_node(node_id)?;
            let row = left_table.decode_node_data_json(&view)?.ok_or_else(|| {
                anyhow!(
                    "benchmark could not decode node data for PK row {} in table {}",
                    offset,
                    left_table.node_type()
                )
            })?;
            let value = row.get(&primary_key.column).ok_or_else(|| {
                anyhow!(
                    "benchmark node data is missing PK '{}' in table {}",
                    primary_key.column,
                    left_table.node_type()
                )
            })?;
            let encoded = match value {
                serde_json::Value::String(value) => value.clone(),
                serde_json::Value::Number(value) => value.to_string(),
                _ => {
                    bail!(
                        "benchmark PK '{}' has unsupported JSON representation in table {}",
                        primary_key.column,
                        left_table.node_type()
                    )
                }
            };
            let key = left_table.parse_pk_key(&encoded)?;
            let left_id = left_table.node_id_from_pk(&key)?.ok_or_else(|| {
                anyhow!(
                    "benchmark left PK lookup failed for table {}",
                    left_table.node_type()
                )
            })?;
            let right_id = right_table.node_id_from_pk(&key)?.ok_or_else(|| {
                anyhow!(
                    "benchmark right PK lookup failed for table {}",
                    left_table.node_type()
                )
            })?;
            let left_view = left_table.edges_for_node(left_id)?;
            let right_view = right_table.edges_for_node(right_id)?;
            let left_row = left_table.decode_node_data_json(&left_view)?;
            let right_row = right_table.decode_node_data_json(&right_view)?;
            if left_row != right_row {
                bail!(
                    "benchmark node data differs for PK '{}' in table {}",
                    encoded,
                    left_table.node_type()
                );
            }
            // Duplicate PK rows legitimately resolve to one canonical node,
            // so parity is the resolved row rather than its physical node ID.
            // Duplicate counts/drops are compared in semantic_stats above.
        }
    }
    if semantic_edges(&left)? != semantic_edges(&right)? {
        bail!("benchmark graph edge membership or sort keys differ");
    }
    Ok(())
}

fn merge_metrics(total: &mut QueryMetrics, next: QueryMetrics) {
    total.rows = total.rows.saturating_add(next.rows);
    total.wire_bytes = total.wire_bytes.saturating_add(next.wire_bytes);
    total.decoded_ipc_bytes = total
        .decoded_ipc_bytes
        .saturating_add(next.decoded_ipc_bytes);
    total.query_startup_nanos = total
        .query_startup_nanos
        .saturating_add(next.query_startup_nanos);
}

fn benchmark_download_decode(
    config: &GraphConfig,
    config_file: Option<&Path>,
    allow_http_loopback: bool,
) -> Result<(Duration, QueryMetrics)> {
    let mut helper = DatabricksHelper::spawn()?;
    let hello = helper.hello()?;
    if hello.protocol != 1 || hello.connector != "4.3.0" || hello.thrift != "0.22.0" {
        bail!("Databricks helper protocol version mismatch");
    }
    let mut elapsed = Duration::ZERO;
    let mut metrics = QueryMetrics::default();
    for table in &config.tables {
        if !matches!(
            table.source,
            diskgraph_ingest::graph_config::DataSource::Databricks { .. }
        ) {
            bail!("download benchmark config contains a non-Databricks table");
        }
        let mut options = QueryStreamOptions::default();
        if allow_http_loopback {
            options.allow_loopback_http_for_benchmark();
        }
        let mut stream =
            DatabricksQueryStream::open(&mut helper, &table.source, config_file, options)?;
        let start = Instant::now();
        let mut observed_rows = 0u64;
        for batch in stream.by_ref() {
            observed_rows = observed_rows.saturating_add(batch?.num_rows() as u64);
        }
        elapsed += start.elapsed();
        let table_metrics = stream.metrics();
        if observed_rows != table_metrics.rows {
            bail!("download benchmark row accounting mismatch");
        }
        merge_metrics(&mut metrics, table_metrics);
    }
    if metrics.rows == 0 {
        bail!("download benchmark returned zero rows");
    }
    Ok((elapsed, metrics))
}

fn benchmark_databricks_ingest(
    config: GraphConfig,
    output: &Path,
    config_file: Option<&Path>,
    allow_http_loopback: bool,
    isolate_consumer: bool,
) -> Result<IngestStats> {
    if allow_http_loopback {
        std::env::set_var("DISKGRAPH_DATABRICKS_BENCHMARK_ALLOW_HTTP_LOOPBACK", "1");
    }
    if isolate_consumer {
        std::env::set_var("DISKGRAPH_DATABRICKS_BENCHMARK_ISOLATE_CONSUMER", "1");
    }
    let result = ingest_with_stats_from_source(
        config,
        None,
        output
            .to_str()
            .ok_or_else(|| anyhow!("benchmark output path is not UTF-8"))?,
        Dest::Local,
        IngestOptions {
            databricks_config_file: config_file.map(Path::to_path_buf),
            ..IngestOptions::default()
        },
        None,
        None,
    );
    if allow_http_loopback {
        std::env::remove_var("DISKGRAPH_DATABRICKS_BENCHMARK_ALLOW_HTTP_LOOPBACK");
    }
    if isolate_consumer {
        std::env::remove_var("DISKGRAPH_DATABRICKS_BENCHMARK_ISOLATE_CONSUMER");
    }
    let (_, stats) = result?;
    Ok(stats)
}

fn semantic_stats(stats: &IngestStats) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(stats)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("benchmark stats did not serialize as an object"))?;
    // Runtime, transport, path, and resource observations intentionally vary
    // between isolated/local/combined runs. Every other present or future
    // IngestStats field is semantic and must compare exactly.
    object.remove("output");
    let perf = object
        .get_mut("perf")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| anyhow!("benchmark stats perf did not serialize as an object"))?;
    // Retain logical SOURCE_SIZE (`bytes_read`/`source_total_bytes`), source
    // rows, and semantic temp/output byte counts. Remove only observations
    // that legitimately differ across runs and transports.
    for field in [
        "wall_time_ms",
        "encode_wall_nanos",
        "source_input_wait_nanos",
        "context_wall_ms",
        "encode_wall_ms",
        "pk_map_wall_ms",
        "edges_map_wall_ms",
        "index_wall_ms",
        "finalize_wall_ms",
        // PtrHash serialization is intentionally non-canonical across
        // equivalent parallel builds, so aggregate physical output bytes are
        // a resource observation rather than a semantic source statistic.
        "bytes_written",
        "databricks_wire_bytes",
        "databricks_decoded_ipc_bytes",
        "databricks_query_startup_nanos",
        "encode_memory_high_water_bytes",
        "peak_rss_anon_bytes",
        "peak_context_rss_anon_bytes",
        "peak_encode_rss_anon_bytes",
        "peak_pk_map_rss_anon_bytes",
        "peak_edges_map_rss_anon_bytes",
        "peak_index_rss_anon_bytes",
        "peak_finalize_rss_anon_bytes",
    ] {
        perf.remove(field);
    }
    Ok(value)
}

fn equivalent_stats(left: &IngestStats, right: &IngestStats) -> Result<bool> {
    Ok(semantic_stats(left)? == semantic_stats(right)?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let databricks_config = load_config(&args.databricks_config)?;
    fs::create_dir_all(&args.output_dir)?;
    let mut local_config = load_config(&args.local_config)?;
    local_config.dataset_name = databricks_config.dataset_name.clone();
    let local_base_dir = args.local_config.parent().ok_or_else(|| {
        anyhow!(
            "local benchmark config has no parent: {}",
            args.local_config.display()
        )
    })?;
    let local_schema = inspect_source_schema(
        local_config.clone(),
        Some(local_base_dir),
        args.config_file.as_deref(),
    )?;
    let databricks_schema = inspect_source_schema(
        databricks_config.clone(),
        None,
        args.config_file.as_deref(),
    )?;
    if local_schema != databricks_schema {
        bail!(
            "benchmark source schemas differ:\nlocal={}\ndatabricks={}",
            serde_json::to_string(&local_schema)?,
            serde_json::to_string(&databricks_schema)?
        );
    }
    let local_output = args.output_dir.join("current-path");
    let (_, local_stats) = ingest_with_stats_from_source(
        local_config,
        Some(local_base_dir),
        local_output
            .to_str()
            .ok_or_else(|| anyhow!("benchmark output path is not UTF-8"))?,
        Dest::Local,
        IngestOptions::default(),
        None,
        None,
    )?;
    let (download_elapsed, download_metrics) = benchmark_download_decode(
        &databricks_config,
        args.config_file.as_deref(),
        args.allow_http_loopback,
    )?;

    // Use a separate Databricks pass to measure the production consumer over
    // the exact Cloud Fetch Arrow workload. Its rendezvous mode pauses the sole
    // download/decode worker while each batch is active, removing upstream
    // contention from measured consumer service time. Local Parquet remains
    // only as the correctness/current-path comparison below.
    let encode_output = args.output_dir.join("encode-probe");
    let encode_stats = benchmark_databricks_ingest(
        databricks_config.clone(),
        &encode_output,
        args.config_file.as_deref(),
        args.allow_http_loopback,
        true,
    )?;

    let combined_output = args.output_dir.join("combined");
    let combined_stats = benchmark_databricks_ingest(
        databricks_config,
        &combined_output,
        args.config_file.as_deref(),
        args.allow_http_loopback,
        false,
    )?;

    let rows = combined_stats.perf.source_rows;
    if download_metrics.rows != rows
        || local_stats.perf.source_rows != rows
        || encode_stats.perf.source_rows != rows
    {
        bail!(
            "benchmark source row counts are not equivalent: download={}, local={}, encode={}, combined={}",
            download_metrics.rows,
            local_stats.perf.source_rows,
            encode_stats.perf.source_rows,
            rows
        );
    }
    if !equivalent_stats(&local_stats, &combined_stats)? {
        bail!(
            "benchmark local and combined Databricks stats are not equivalent:\nlocal={}\ncombined={}",
            semantic_stats(&local_stats)?,
            semantic_stats(&combined_stats)?
        );
    }
    if !equivalent_stats(&encode_stats, &combined_stats)? {
        bail!(
            "benchmark isolated and combined Databricks stats are not equivalent:\nisolated={}\ncombined={}",
            semantic_stats(&encode_stats)?,
            semantic_stats(&combined_stats)?
        );
    }
    compare_graph_semantics(&local_output, &combined_output)?;
    compare_graph_semantics(&encode_output, &combined_output)?;
    let download_seconds = duration_seconds(download_elapsed);
    // The probe is an independent production consumer run over the exact
    // decoded Arrow workload. Removing iterator wait and query startup leaves
    // its active Encode time as the isolated encode ceiling.
    let encode_nanos = encode_stats
        .perf
        .encode_wall_nanos
        .saturating_sub(encode_stats.perf.source_input_wait_nanos)
        .saturating_sub(encode_stats.perf.databricks_query_startup_nanos);
    if encode_nanos == 0 {
        bail!("encode ceiling measurement had no active Encode time");
    }
    let encode_seconds = duration_seconds(Duration::from_nanos(encode_nanos));
    let combined_steady_nanos = combined_stats
        .perf
        .encode_wall_nanos
        .saturating_sub(combined_stats.perf.databricks_query_startup_nanos);
    let combined_seconds = duration_seconds(Duration::from_nanos(combined_steady_nanos));
    let download_rate = rows as f64 / download_seconds;
    let encode_rate = rows as f64 / encode_seconds;
    let combined_rate = rows as f64 / combined_seconds;
    let bottleneck_rate = download_rate.min(encode_rate);
    let ratio = combined_rate / bottleneck_rate;
    let passed = ratio >= MINIMUM_PIPELINE_RATIO;
    let report = BenchmarkReport {
        rows,
        download_decode_seconds: download_seconds,
        encode_only_seconds: encode_seconds,
        combined_steady_seconds: combined_seconds,
        query_startup_seconds: combined_stats.perf.databricks_query_startup_nanos as f64
            / 1_000_000_000.0,
        download_decode_rows_per_second: download_rate,
        encode_rows_per_second: encode_rate,
        combined_rows_per_second: combined_rate,
        slower_component_rows_per_second: bottleneck_rate,
        combined_to_bottleneck_ratio: ratio,
        databricks_wire_bytes: combined_stats.perf.databricks_wire_bytes,
        databricks_decoded_ipc_bytes: combined_stats.perf.databricks_decoded_ipc_bytes,
        memory_high_water_bytes: combined_stats.perf.encode_memory_high_water_bytes,
        passed,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !passed {
        bail!(
            "combined throughput ratio {ratio:.3} is below required {:.3}",
            MINIMUM_PIPELINE_RATIO
        );
    }
    Ok(())
}
