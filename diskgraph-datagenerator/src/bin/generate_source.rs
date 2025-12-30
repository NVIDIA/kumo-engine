//! Generate raw CSV/Parquet inputs plus source_metadata.json for diskgraph-ingest.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Datelike, Timelike, Utc};
use arrow_array::builder::{Float32Builder, Int64Builder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use clap::{Parser, ValueEnum};
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, GcsConfig, S3Config,
};
use object_store::WriteMultipart;
use object_store::{path::Path as ObjectPath, ObjectStore};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokio::runtime::Runtime;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate raw CSV/Parquet datasets + source_metadata.json.")]
struct Args {
    /// Output target; local path or s3/gs/az URL (e.g., s3://bucket/prefix).
    #[arg(long, value_name = "PATH_OR_URL")]
    output: String,
    /// Destination kind.
    #[arg(long, value_enum, default_value = "local")]
    dest: Dest,
    /// Output file format.
    #[arg(long, value_enum, default_value = "parquet")]
    format: OutputFormat,
    /// Total nodes across all node tables (supports 10k, 5m, 2b).
    #[arg(long, default_value = "1000000", value_parser = parse_human_u64)]
    nodes: u64,
    /// Rows in the edge table (order_items).
    #[arg(long, default_value = "500000", value_parser = parse_human_u64)]
    edge_rows: u64,
    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Dest {
    Local,
    S3,
    Gcs,
    Adls,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum OutputFormat {
    Csv,
    Parquet,
}

#[derive(Clone)]
struct NodeTypeDef {
    name: &'static str,
    weight: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args = Args::parse();
    let target = OutputTarget::parse(&args)?;
    target.prepare()?;

    let node_defs = vec![
        NodeTypeDef { name: "User", weight: 30 },
        NodeTypeDef { name: "Product", weight: 25 },
        NodeTypeDef { name: "Order", weight: 15 },
        NodeTypeDef { name: "Session", weight: 10 },
        NodeTypeDef { name: "Merchant", weight: 10 },
        NodeTypeDef { name: "Category", weight: 10 },
    ];
    let node_counts = distribute_counts(args.nodes, &node_defs);

    let mut rng = StdRng::seed_from_u64(args.seed);
    let table_counts: HashMap<&str, u64> = node_defs
        .iter()
        .zip(node_counts.iter())
        .map(|(d, c)| (d.name, *c))
        .collect();
    for (def, count) in node_defs.iter().zip(node_counts.iter()) {
        match args.format {
            OutputFormat::Parquet => {
                write_node_parquet(&target, def.name, *count, &table_counts, &mut rng)?
            }
            OutputFormat::Csv => {
                write_node_csv(&target, def.name, *count, &table_counts, &mut rng)?
            }
        }
    }

    match args.format {
        OutputFormat::Parquet => {
            write_edge_parquet(&target, args.edge_rows, &table_counts, &mut rng)?
        }
        OutputFormat::Csv => write_edge_csv(&target, args.edge_rows, &table_counts, &mut rng)?,
    }

    write_source_metadata(&target, args.format)?;
    info!("Done. Output written to {}", target.display());
    Ok(())
}

fn write_node_parquet(
    target: &OutputTarget,
    name: &str,
    rows: u64,
    counts: &HashMap<&str, u64>,
    rng: &mut StdRng,
) -> Result<()> {
    let mut fields = vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("weight", DataType::Float32, false),
        Field::new("created_at", DataType::Timestamp(arrow_schema::TimeUnit::Second, None), false),
    ];
    if name == "User" {
        fields.push(Field::new("created_at_str", DataType::Utf8, false));
    }
    fields.push(Field::new(
        "embedding",
        DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
        false,
    ));
    if name == "Order" || name == "Session" {
        fields.push(Field::new("user_id", DataType::Int64, false));
    }
    if name == "Order" || name == "Product" {
        fields.push(Field::new("merchant_id", DataType::Int64, false));
    }
    if name == "Product" {
        fields.push(Field::new("category_id", DataType::Int64, false));
    }
    let schema = Schema::new(fields);
    let path = target.join(&format!("tables/{}/part-00000.parquet", name.to_lowercase()));
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    let props = WriterProperties::builder().set_max_row_group_size(50_000).build();
    let sink = target.sink(&path)?;
    let mut writer = ArrowWriter::try_new(sink, Arc::new(schema.clone()), Some(props))?;

    let mut remaining = rows;
    let mut next_id = 0u64;
    while remaining > 0 {
        let take = remaining.min(50_000) as usize;
        let mut id_builder = Int64Builder::new();
        let mut name_builder = arrow_array::builder::StringBuilder::new();
        let mut weight_builder = arrow_array::builder::Float32Builder::new();
        let mut ts_builder = arrow_array::builder::TimestampSecondBuilder::new();
        let mut ts_str_builder = if name == "User" {
            Some(arrow_array::builder::StringBuilder::new())
        } else {
            None
        };
        let mut list_builder = arrow_array::builder::ListBuilder::new(Float32Builder::new());
        for _ in 0..take {
            let id = next_id as i64;
            id_builder.append_value(id);
            name_builder.append_value(format!("{}-{:08}", name, id));
            weight_builder.append_value(rng.gen_range(0.0f32..100.0));
            let ts = 1_700_000_000 + id;
            ts_builder.append_value(ts);
            if let Some(builder) = ts_str_builder.as_mut() {
                builder.append_value(format_non_iso_ts(ts));
            }
            list_builder.values().append_value(rng.gen_range(0.0..1.0));
            list_builder.values().append_value(rng.gen_range(0.0..1.0));
            list_builder.values().append_value(rng.gen_range(0.0..1.0));
            list_builder.append(true);
            next_id += 1;
        }
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(id_builder.finish()) as ArrayRef,
            Arc::new(name_builder.finish()) as ArrayRef,
            Arc::new(weight_builder.finish()) as ArrayRef,
            Arc::new(ts_builder.finish()) as ArrayRef,
        ];
        if let Some(mut builder) = ts_str_builder {
            cols.push(Arc::new(builder.finish()) as ArrayRef);
        }
        cols.push(Arc::new(list_builder.finish()) as ArrayRef);
        if name == "Order" || name == "Session" {
            let max_user = *counts.get("User").unwrap_or(&1);
            let mut user_builder = Int64Builder::new();
            for _ in 0..take {
                user_builder.append_value(rng.gen_range(0..max_user) as i64);
            }
            cols.push(Arc::new(user_builder.finish()) as ArrayRef);
        }
        if name == "Order" || name == "Product" {
            let max_merchant = *counts.get("Merchant").unwrap_or(&1);
            let mut merchant_builder = Int64Builder::new();
            for _ in 0..take {
                merchant_builder.append_value(rng.gen_range(0..max_merchant) as i64);
            }
            cols.push(Arc::new(merchant_builder.finish()) as ArrayRef);
        }
        if name == "Product" {
            let max_category = *counts.get("Category").unwrap_or(&1);
            let mut category_builder = Int64Builder::new();
            for _ in 0..take {
                category_builder.append_value(rng.gen_range(0..max_category) as i64);
            }
            cols.push(Arc::new(category_builder.finish()) as ArrayRef);
        }
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), cols)?;
        writer.write(&batch)?;
        remaining -= take as u64;
    }
    let sink = writer.into_inner()?;
    sink.finish()?;
    Ok(())
}

fn write_node_csv(
    target: &OutputTarget,
    name: &str,
    rows: u64,
    counts: &HashMap<&str, u64>,
    rng: &mut StdRng,
) -> Result<()> {
    let path = target.join(&format!("tables/{}/part-00000.csv", name.to_lowercase()));
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut sink = target.sink(&path)?;
    let mut headers = vec!["id", "name", "weight", "created_at"];
    if name == "User" {
        headers.push("created_at_str");
    }
    headers.push("embedding");
    if name == "Order" || name == "Session" {
        headers.push("user_id");
    }
    if name == "Order" || name == "Product" {
        headers.push("merchant_id");
    }
    if name == "Product" {
        headers.push("category_id");
    }
    writeln!(sink, "{}", headers.join(","))?;
    for id in 0..rows {
        let name_val = format!("{}-{:08}", name, id);
        let weight = rng.gen_range(0.0f32..100.0);
        let ts = 1_700_000_000 + id;
        let embedding = format!(
            "[{:.3},{:.3},{:.3}]",
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0)
        );
        let mut row = vec![
            id.to_string(),
            name_val,
            weight.to_string(),
            ts.to_string(),
        ];
        if name == "User" {
            row.push(format_non_iso_ts(ts as i64));
        }
        row.push(embedding);
        if name == "Order" || name == "Session" {
            let max_user = *counts.get("User").unwrap_or(&1);
            row.push(rng.gen_range(0..max_user).to_string());
        }
        if name == "Order" || name == "Product" {
            let max_merchant = *counts.get("Merchant").unwrap_or(&1);
            row.push(rng.gen_range(0..max_merchant).to_string());
        }
        if name == "Product" {
            let max_category = *counts.get("Category").unwrap_or(&1);
            row.push(rng.gen_range(0..max_category).to_string());
        }
        write_csv_row(&mut sink, &row)?;
    }
    sink.finish()?;
    Ok(())
}

fn write_edge_parquet(
    target: &OutputTarget,
    rows: u64,
    counts: &HashMap<&str, u64>,
    rng: &mut StdRng,
) -> Result<()> {
    let schema = Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("product_id", DataType::Int64, false),
        Field::new("weight", DataType::Int32, false),
        Field::new("created_at", DataType::Timestamp(arrow_schema::TimeUnit::Second, None), false),
    ]);
    let path = target.join("tables/order_items/part-00000.parquet");
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    let props = WriterProperties::builder().set_max_row_group_size(100_000).build();
    let sink = target.sink(&path)?;
    let mut writer = ArrowWriter::try_new(sink, Arc::new(schema.clone()), Some(props))?;

    let mut remaining = rows;
    while remaining > 0 {
        let take = remaining.min(100_000) as usize;
        let mut order_builder = Int64Builder::new();
        let mut product_builder = Int64Builder::new();
        let mut weight_builder = arrow_array::builder::Int32Builder::new();
        let mut ts_builder = arrow_array::builder::TimestampSecondBuilder::new();
        for _ in 0..take {
            let max_order = *counts.get("Order").unwrap_or(&1);
            let max_product = *counts.get("Product").unwrap_or(&1);
            order_builder.append_value(rng.gen_range(0..max_order) as i64);
            product_builder.append_value(rng.gen_range(0..max_product) as i64);
            weight_builder.append_value(rng.gen_range(1..5));
            ts_builder.append_value(rng.gen_range(1_700_000_000..1_800_000_000));
        }
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(order_builder.finish()) as ArrayRef,
                Arc::new(product_builder.finish()) as ArrayRef,
                Arc::new(weight_builder.finish()) as ArrayRef,
                Arc::new(ts_builder.finish()) as ArrayRef,
            ],
        )?;
        writer.write(&batch)?;
        remaining -= take as u64;
    }
    let sink = writer.into_inner()?;
    sink.finish()?;
    Ok(())
}

fn write_edge_csv(
    target: &OutputTarget,
    rows: u64,
    counts: &HashMap<&str, u64>,
    rng: &mut StdRng,
) -> Result<()> {
    let path = target.join("tables/order_items/part-00000.csv");
    if let OutputTarget::Local { .. } = target {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut sink = target.sink(&path)?;
    writeln!(sink, "order_id,product_id,weight,created_at")?;
    for _ in 0..rows {
        write_csv_row(
            &mut sink,
            &[
                rng.gen_range(0..*counts.get("Order").unwrap_or(&1)).to_string(),
                rng.gen_range(0..*counts.get("Product").unwrap_or(&1)).to_string(),
                rng.gen_range(1..5).to_string(),
                rng.gen_range(1_700_000_000..1_800_000_000).to_string(),
            ],
        )?;
    }
    sink.finish()?;
    Ok(())
}

fn write_source_metadata(target: &OutputTarget, format: OutputFormat) -> Result<()> {
    let ext = match format {
        OutputFormat::Csv => "csv",
        OutputFormat::Parquet => "parquet",
    };
    let json = serde_json::json!({
        "dataset_name": "sample_source",
        "tables": [
            {
                "name": "User",
                "format": format!("{ext}"),
                "paths": [format!("tables/user/part-00000.{ext}")],
                "primary_key": "id",
                "weight": "weight",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "name", "dtype": "string", "fixed_len": 12},
                    {"name": "weight", "dtype": "float32"},
                    {"name": "created_at", "dtype": "timestamp"},
                    {"name": "created_at_str", "dtype": "string", "fixed_len": 19},
                    {"name": "embedding", "dtype": "float32_array"}
                ]
            },
            {
                "name": "Product",
                "format": format!("{ext}"),
                "paths": [format!("tables/product/part-00000.{ext}")],
                "primary_key": "id",
                "weight": "weight",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "name", "dtype": "string", "fixed_len": 12},
                    {"name": "weight", "dtype": "float32"},
                    {"name": "created_at", "dtype": "timestamp"},
                    {"name": "embedding", "dtype": "float32_array"},
                    {"name": "merchant_id", "dtype": "int64"},
                    {"name": "category_id", "dtype": "int64"}
                ]
            },
            {
                "name": "Order",
                "format": format!("{ext}"),
                "paths": [format!("tables/order/part-00000.{ext}")],
                "primary_key": "id",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "name", "dtype": "string", "fixed_len": 12},
                    {"name": "weight", "dtype": "float32"},
                    {"name": "created_at", "dtype": "timestamp"},
                    {"name": "embedding", "dtype": "float32_array"},
                    {"name": "user_id", "dtype": "int64"},
                    {"name": "merchant_id", "dtype": "int64"}
                ]
            },
            {
                "name": "Session",
                "format": format!("{ext}"),
                "paths": [format!("tables/session/part-00000.{ext}")],
                "primary_key": "id",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "name", "dtype": "string", "fixed_len": 12},
                    {"name": "weight", "dtype": "float32"},
                    {"name": "created_at", "dtype": "timestamp"},
                    {"name": "embedding", "dtype": "float32_array"},
                    {"name": "user_id", "dtype": "int64"}
                ]
            },
            {
                "name": "Merchant",
                "format": format!("{ext}"),
                "paths": [format!("tables/merchant/part-00000.{ext}")],
                "primary_key": "id",
                "weight": "weight",
                "create_time": "created_at",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "name", "dtype": "string", "fixed_len": 12},
                    {"name": "weight", "dtype": "float32"},
                    {"name": "created_at", "dtype": "timestamp"},
                    {"name": "embedding", "dtype": "float32_array"}
                ]
            },
            {
                "name": "Category",
                "format": format!("{ext}"),
                "paths": [format!("tables/category/part-00000.{ext}")],
                "primary_key": "id",
                "columns": [
                    {"name": "id", "dtype": "int64"},
                    {"name": "name", "dtype": "string", "fixed_len": 12}
                ]
            },
            {
                "name": "OrderItem",
                "format": format!("{ext}"),
                "paths": [format!("tables/order_items/part-00000.{ext}")],
                "edge_table": {
                    "src_fk": {"column": "order_id", "ref_table": "Order", "ref_column": "id"},
                    "dst_fk": {"column": "product_id", "ref_table": "Product", "ref_column": "id"}
                },
                "weight": "weight",
                "create_time": "created_at",
                "columns": [
                    {"name": "order_id", "dtype": "int64"},
                    {"name": "product_id", "dtype": "int64"},
                    {"name": "weight", "dtype": "int32"},
                    {"name": "created_at", "dtype": "timestamp"}
                ]
            }
        ],
        "relationships": [
            {
                "name": "order_user",
                "src_table": "Order",
                "src_column": "user_id",
                "dst_table": "User",
                "dst_column": "id"
            },
            {
                "name": "order_merchant",
                "src_table": "Order",
                "src_column": "merchant_id",
                "dst_table": "Merchant",
                "dst_column": "id"
            },
            {
                "name": "session_user",
                "src_table": "Session",
                "src_column": "user_id",
                "dst_table": "User",
                "dst_column": "id"
            },
            {
                "name": "product_merchant",
                "src_table": "Product",
                "src_column": "merchant_id",
                "dst_table": "Merchant",
                "dst_column": "id"
            },
            {
                "name": "product_category",
                "src_table": "Product",
                "src_column": "category_id",
                "dst_table": "Category",
                "dst_column": "id"
            }
        ]
    });
    target.write_bytes("source_metadata.json", serde_json::to_vec_pretty(&json)?)?;
    Ok(())
}

fn distribute_counts(total: u64, defs: &[NodeTypeDef]) -> Vec<u64> {
    let total_weight: u64 = defs.iter().map(|d| d.weight as u64).sum();
    let mut counts = Vec::with_capacity(defs.len());
    let mut remaining = total;
    for (idx, def) in defs.iter().enumerate() {
        if idx == defs.len() - 1 {
            counts.push(remaining);
        } else {
            let share = total.saturating_mul(def.weight as u64) / total_weight.max(1);
            let take = share.min(remaining);
            counts.push(take);
            remaining = remaining.saturating_sub(take);
        }
    }
    counts
}

fn format_non_iso_ts(ts: i64) -> String {
    let dt = DateTime::<Utc>::from_timestamp(ts, 0)
        .map(|dt| dt.naive_utc())
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap().naive_utc());
    format!(
        "{:02}/{:02}/{:04} {:02}:{:02}:{:02}",
        dt.month(),
        dt.day(),
        dt.year(),
        dt.hour(),
        dt.minute(),
        dt.second()
    )
}

fn write_csv_row(mut sink: impl Write, fields: &[String]) -> io::Result<()> {
    for (idx, field) in fields.iter().enumerate() {
        if idx > 0 {
            sink.write_all(b",")?;
        }
        if field.contains(',') || field.contains('"') || field.contains('\n') {
            let escaped = field.replace('"', "\"\"");
            write!(sink, "\"{}\"", escaped)?;
        } else {
            sink.write_all(field.as_bytes())?;
        }
    }
    sink.write_all(b"\n")?;
    Ok(())
}

enum OutputTarget {
    Local { root: PathBuf },
    S3 {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        display: String,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
    Gcs {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        display: String,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
    Adls {
        prefix: String,
        store: Arc<dyn ObjectStore>,
        display: String,
        rt: Arc<Runtime>,
        part_size: usize,
        part_concurrency: usize,
    },
}

impl OutputTarget {
    fn parse(args: &Args) -> Result<Self> {
        match args.dest {
            Dest::Local => Ok(OutputTarget::Local {
                root: PathBuf::from(&args.output),
            }),
            Dest::S3 => {
                let (bucket, prefix) = parse_s3_url(&args.output)?;
                let s3 = build_s3_client(S3Config {
                    bucket: bucket.clone(),
                    region: None,
                    endpoint: None,
                    allow_http: None,
                    virtual_hosted: None,
                    profile: std::env::var("S3_PROFILE").ok(),
                    access_key_id: None,
                    secret_access_key: None,
                    session_token: None,
                })?;
                Ok(OutputTarget::S3 {
                    prefix,
                    store: s3.store,
                    display: args.output.clone(),
                    rt: Arc::new(Runtime::new()?),
                    part_size: 64 * 1024 * 1024,
                    part_concurrency: 10,
                })
            }
            Dest::Gcs => {
                let (bucket, prefix) = parse_gs_url(&args.output)?;
                let gcs = build_gcs_client(GcsConfig { bucket })?;
                Ok(OutputTarget::Gcs {
                    prefix,
                    store: gcs.store,
                    display: args.output.clone(),
                    rt: Arc::new(Runtime::new()?),
                    part_size: 64 * 1024 * 1024,
                    part_concurrency: 10,
                })
            }
            Dest::Adls => {
                let (account, container, prefix) = parse_az_url(&args.output)?;
                let adls = build_adls_client(AdlsConfig {
                    account,
                    container,
                    endpoint: std::env::var("AZURE_ENDPOINT").ok(),
                    allow_http: None,
                    access_key: std::env::var("AZURE_ACCESS_KEY").ok(),
                    sas_key: std::env::var("AZURE_SAS_KEY").ok(),
                    bearer_token: std::env::var("AZURE_BEARER_TOKEN").ok(),
                    client_id: std::env::var("AZURE_CLIENT_ID").ok(),
                    client_secret: std::env::var("AZURE_CLIENT_SECRET").ok(),
                    tenant_id: std::env::var("AZURE_TENANT_ID").ok(),
                })?;
                Ok(OutputTarget::Adls {
                    prefix,
                    store: adls.store,
                    display: args.output.clone(),
                    rt: Arc::new(Runtime::new()?),
                    part_size: 64 * 1024 * 1024,
                    part_concurrency: 10,
                })
            }
        }
    }

    fn prepare(&self) -> Result<()> {
        if let OutputTarget::Local { root } = self {
            fs::create_dir_all(root)?;
        }
        Ok(())
    }

    fn join(&self, name: &str) -> PathBuf {
        match self {
            OutputTarget::Local { root } => root.join(name),
            OutputTarget::S3 { prefix, .. }
            | OutputTarget::Gcs { prefix, .. }
            | OutputTarget::Adls { prefix, .. } => {
                if prefix.is_empty() {
                    PathBuf::from(name)
                } else {
                    PathBuf::from(format!("{}/{}", prefix.trim_end_matches('/'), name))
                }
            }
        }
    }

    fn write_bytes(&self, name: &str, bytes: Vec<u8>) -> Result<()> {
        match self {
            OutputTarget::Local { root } => {
                fs::write(root.join(name), bytes)?;
            }
            OutputTarget::S3 { store, rt, prefix, .. }
            | OutputTarget::Gcs { store, rt, prefix, .. }
            | OutputTarget::Adls { store, rt, prefix, .. } => {
                let obj = if prefix.is_empty() {
                    ObjectPath::from(name)
                } else {
                    ObjectPath::from(format!("{}/{}", prefix.trim_end_matches('/'), name))
                };
                rt.block_on(store.put(&obj, Bytes::from(bytes).into()))
                    .map_err(anyhow::Error::from)?;
            }
        }
        Ok(())
    }

    fn display(&self) -> String {
        match self {
            OutputTarget::Local { root } => root.display().to_string(),
            OutputTarget::S3 { display, .. }
            | OutputTarget::Gcs { display, .. }
            | OutputTarget::Adls { display, .. } => display.clone(),
        }
    }

    fn sink(&self, path: &Path) -> Result<OutputSink> {
        match self {
            OutputTarget::Local { .. } => {
                let file = fs::File::create(path)?;
                Ok(OutputSink::Local(file))
            }
            OutputTarget::S3 {
                store,
                rt,
                part_size,
                part_concurrency,
                ..
            }
            | OutputTarget::Gcs {
                store,
                rt,
                part_size,
                part_concurrency,
                ..
            }
            | OutputTarget::Adls {
                store,
                rt,
                part_size,
                part_concurrency,
                ..
            } => {
                let obj = ObjectPath::from(path.to_string_lossy().to_string());
                let upload = rt.block_on(store.put_multipart(&obj)).map_err(to_io_err)?;
                Ok(OutputSink::Multipart(MultipartSink {
                    writer: Some(WriteMultipart::new_with_chunk_size(upload, *part_size)),
                    rt: rt.clone(),
                    max_concurrency: *part_concurrency,
                    handle: rt.handle().clone(),
                }))
            }
        }
    }
}

enum OutputSink {
    Local(fs::File),
    Multipart(MultipartSink),
}

impl OutputSink {
    fn finish(self) -> io::Result<()> {
        match self {
            OutputSink::Local(mut file) => file.flush(),
            OutputSink::Multipart(sink) => sink.finish(),
        }
    }
}

impl Write for OutputSink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        match self {
            OutputSink::Local(file) => file.write(data),
            OutputSink::Multipart(sink) => sink.write(data),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            OutputSink::Local(file) => file.flush(),
            OutputSink::Multipart(sink) => sink.flush(),
        }
    }
}

struct MultipartSink {
    writer: Option<WriteMultipart>,
    rt: Arc<Runtime>,
    max_concurrency: usize,
    handle: tokio::runtime::Handle,
}

impl MultipartSink {
    fn finish(mut self) -> io::Result<()> {
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> io::Result<()> {
        let mut writer = match self.writer.take() {
            Some(writer) => writer,
            None => return Ok(()),
        };
        let rt = self.rt.clone();
        Self::retry_with_backoff(|| rt.block_on(writer.wait_for_capacity(0)))?;
        self.rt.block_on(writer.finish()).map_err(to_io_err)?;
        Ok(())
    }

    fn retry_with_backoff<T>(mut op: impl FnMut() -> Result<T, object_store::Error>) -> io::Result<T> {
        const RETRIES: usize = 3;
        const BASE_DELAY_MS: u64 = 250;
        let mut last_err: Option<object_store::Error> = None;
        for attempt in 0..=RETRIES {
            match op() {
                Ok(val) => return Ok(val),
                Err(err) => {
                    last_err = Some(err);
                    if attempt == RETRIES {
                        break;
                    }
                    let delay = BASE_DELAY_MS.saturating_mul(1 << attempt);
                    std::thread::sleep(Duration::from_millis(delay));
                }
            }
        }
        if let Some(err) = last_err {
            Err(to_io_err(err))
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "unknown multipart error",
            ))
        }
    }
}

impl Write for MultipartSink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if let Some(writer) = self.writer.as_mut() {
            let rt = self.rt.clone();
            let max_concurrency = self.max_concurrency;
            Self::retry_with_backoff(|| rt.block_on(writer.wait_for_capacity(max_concurrency)))?;
            let _guard = self.handle.enter();
            writer.write(data);
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            let rt = self.rt.clone();
            Self::retry_with_backoff(|| rt.block_on(writer.wait_for_capacity(0)))?;
        }
        Ok(())
    }
}

impl Drop for MultipartSink {
    fn drop(&mut self) {
        if let Err(err) = self.finish_inner() {
            eprintln!("MultipartSink finish error on drop: {err}");
        }
    }
}

fn to_io_err(err: object_store::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err)
}

fn parse_s3_url(raw: &str) -> Result<(String, String)> {
    let url = url::Url::parse(raw)
        .map_err(|e| anyhow::anyhow!("invalid s3 url {raw}: {e}"))?;
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("missing bucket in s3 url"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    Ok((bucket, key))
}

fn parse_gs_url(raw: &str) -> Result<(String, String)> {
    let url = url::Url::parse(raw)
        .map_err(|e| anyhow::anyhow!("invalid gs url {raw}: {e}"))?;
    let bucket = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("missing bucket in gs url"))?
        .to_string();
    let key = url.path().trim_start_matches('/').to_string();
    Ok((bucket, key))
}

fn parse_az_url(raw: &str) -> Result<(String, String, String)> {
    let url = url::Url::parse(raw)
        .map_err(|e| anyhow::anyhow!("invalid az url {raw}: {e}"))?;
    let account = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("missing account in az url"))?
        .to_string();
    let mut segments = url
        .path()
        .trim_start_matches('/')
        .split('/')
        .filter(|s| !s.is_empty());
    let container = segments
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing container in az url"))?
        .to_string();
    let key = segments.collect::<Vec<_>>().join("/");
    Ok((account, container, key))
}

fn parse_human_u64(raw: &str) -> Result<u64, String> {
    let (number, suffix) = split_number_suffix(raw)?;
    let multiplier = match suffix.as_str() {
        "" => 1u64,
        "k" => 1_000u64,
        "m" => 1_000_000u64,
        "b" => 1_000_000_000u64,
        "t" => 1_000_000_000_000u64,
        _ => return Err(format!("invalid suffix: {raw}")),
    };
    number
        .checked_mul(multiplier)
        .ok_or_else(|| format!("value out of range: {raw}"))
}

fn split_number_suffix(raw: &str) -> Result<(u64, String), String> {
    let cleaned = raw.trim().replace('_', "");
    if cleaned.is_empty() {
        return Err("empty value".into());
    }
    let lower = cleaned.to_ascii_lowercase();
    let mut idx = 0;
    for (i, ch) in lower.char_indices() {
        if !ch.is_ascii_digit() {
            idx = i;
            break;
        }
        idx = i + ch.len_utf8();
    }
    let (num_part, suffix_part) = lower.split_at(idx);
    if num_part.is_empty() {
        return Err(format!("missing number: {raw}"));
    }
    let number: u64 = num_part
        .parse()
        .map_err(|_| format!("invalid number: {raw}"))?;
    Ok((number, suffix_part.to_string()))
}
