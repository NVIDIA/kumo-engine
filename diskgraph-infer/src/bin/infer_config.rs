use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::Parser;
use diskgraph_fs::{
    build_adls_client, build_gcs_client, build_s3_client, AdlsConfig, CloudTransfer,
    CloudTransferConfig, GcsConfig, S3Config,
};
use diskgraph_infer::{infer_source_metadata, parse_root_source, InferOptions};
use diskgraph_ingest::graph_config::{EdgeTableSpec, TableSpec};
use object_store::path::Path as ObjectPath;
use url::Url;

const CONFIG_FILENAME: &str = "diskgraph_config.json";

#[derive(Parser, Debug)]
#[command(author, version, about = "Infer diskgraph_config.json from CSV/Parquet tables.")]
struct Args {
    /// Root directory or cloud URL (s3://, gs://, az://).
    #[arg(long)]
    input: String,
    /// Output path for diskgraph_config.json (defaults to <input>/diskgraph_config.json).
    #[arg(long)]
    output: Option<String>,
    /// Validate schema consistency across all files in each table.
    #[arg(long)]
    validate_all: bool,
    /// Path to graph config.json.
    #[arg(long)]
    graph_config: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = parse_root_source(&args.input)?;

    let graph_config = if let Some(path) = args.graph_config {
        let bytes = std::fs::read(&path)?;
        Some(serde_json::from_slice(&bytes)?)
    } else {
        None
    };

    let options = InferOptions {
        validate_dir_schemas: args.validate_all,
        graph_config,
        ..InferOptions::default()
    };
    eprintln!("infer_config: inferring metadata");
    let result = infer_source_metadata(&root, options)?;
    let output_target = match args.output.as_deref() {
        Some(path) => parse_output_target(path)?,
        None => default_output_target(&root, &args.input)?,
    };
    let output_label = output_target_label(&output_target);
    eprintln!("infer_config: writing {}", output_label);
    let payload = serde_json::to_vec_pretty(&result.metadata)?;
    write_output(&output_target, payload)?;
    println!("Tables:");
    for table in &result.metadata.tables {
        let pk = table.primary_key.as_deref().unwrap_or("none");
        let create_time = table.create_time.as_deref().unwrap_or("none");
        println!(
            "  {} (files={}, primary_key={}, create_time={})",
            table.name,
            table.get_paths().len(),
            pk,
            create_time
        );
    }
    let tables_without_pk: Vec<_> = result
        .metadata
        .tables
        .iter()
        .filter(|table| table.primary_key.is_none())
        .map(|table| table.name.as_str())
        .collect();
    if !tables_without_pk.is_empty() {
        println!("Tables without primary key:");
        for name in tables_without_pk {
            println!("  {}", name);
        }
    }
    // Print fkeys embedded in tables
    let tables_with_fkeys: Vec<_> = result
        .metadata
        .tables
        .iter()
        .filter(|table| !table.fkeys.is_empty())
        .collect();
    if !tables_with_fkeys.is_empty() {
        println!("Foreign Keys:");
        for table in tables_with_fkeys {
            for fkey in &table.fkeys {
                println!(
                    "  {}.{} -> {}",
                    table.name, fkey.src_column, fkey.dst_table
                );
            }
        }
    }
    let edge_tables: Vec<_> = result
        .metadata
        .tables
        .iter()
        .filter_map(|table| table.edge_table.as_ref().map(|edge| (table, edge)))
        .collect();
    if !edge_tables.is_empty() {
        println!("Edge table relationships:");
        print_edge_table_relationships(&edge_tables);
    }
    if !result.warnings.ignored_files.is_empty() {
        println!("Ignored files (schema mismatch):");
        for path in &result.warnings.ignored_files {
            println!("  {}", path);
        }
        if !result.warnings.schema_mismatches.is_empty() {
            println!("Schema diffs:");
            for mismatch in &result.warnings.schema_mismatches {
                println!("  {}", mismatch.path);
                for diff in &mismatch.diffs {
                    println!("    [{}] {} -> {}", diff.index, fmt_column(&diff.left), fmt_column(&diff.right));
                }
            }
        }
    }
    if !result.warnings.notes.is_empty() {
        println!("Notes:");
        for note in &result.warnings.notes {
            println!("  {}", note);
        }
    }
    println!("Wrote {}", output_label);
    if let Some(cmd) = ingest_command_hint(&root, &args.input) {
        println!("Next: {}", cmd);
    }
    Ok(())
}

fn default_output_target(
    root: &diskgraph_infer::RootSource,
    input: &str,
) -> Result<OutputTarget> {
    let target = match root {
        diskgraph_infer::RootSource::Local(path) => OutputTarget::Local(path.join(CONFIG_FILENAME)),
        diskgraph_infer::RootSource::S3 { bucket, prefix } => OutputTarget::S3 {
            bucket: bucket.clone(),
            key: join_prefix(prefix, CONFIG_FILENAME),
        },
        diskgraph_infer::RootSource::Gcs { bucket, prefix } => OutputTarget::Gcs {
            bucket: bucket.clone(),
            key: join_prefix(prefix, CONFIG_FILENAME),
        },
        diskgraph_infer::RootSource::Adls {
            account,
            container,
            prefix,
        } => OutputTarget::Adls {
            account: account.clone(),
            container: container.clone(),
            key: join_prefix(prefix, CONFIG_FILENAME),
        },
    };
    if output_target_label(&target).is_empty() {
        return Err(anyhow!("invalid output for input {}", input));
    }
    Ok(target)
}

fn ingest_command_hint(root: &diskgraph_infer::RootSource, input: &str) -> Option<String> {
    let script = match root {
        diskgraph_infer::RootSource::Local(_) => "ingest_local.sh",
        diskgraph_infer::RootSource::S3 { .. } => "ingest_s3.sh",
        diskgraph_infer::RootSource::Gcs { .. } => "ingest_gcs.sh",
        diskgraph_infer::RootSource::Adls { .. } => "ingest_adls.sh",
    };
    Some(format!("./diskgraph-ingest/scripts/{} {}", script, input))
}

#[derive(Debug)]
enum OutputTarget {
    Local(PathBuf),
    S3 { bucket: String, key: String },
    Gcs { bucket: String, key: String },
    Adls {
        account: String,
        container: String,
        key: String,
    },
}

fn parse_output_target(output: &str) -> Result<OutputTarget> {
    if let Ok(url) = Url::parse(output) {
        match url.scheme() {
            "s3" => {
                let bucket = url.host_str().unwrap_or_default().to_string();
                let key = url.path().trim_start_matches('/').to_string();
                if bucket.is_empty() || key.is_empty() {
                    return Err(anyhow!("invalid s3 output path {}", output));
                }
                Ok(OutputTarget::S3 { bucket, key })
            }
            "gs" | "gcs" => {
                let bucket = url.host_str().unwrap_or_default().to_string();
                let key = url.path().trim_start_matches('/').to_string();
                if bucket.is_empty() || key.is_empty() {
                    return Err(anyhow!("invalid gs output path {}", output));
                }
                Ok(OutputTarget::Gcs { bucket, key })
            }
            "az" | "adls" => {
                let account = url.host_str().unwrap_or_default().to_string();
                let mut parts = url.path().trim_start_matches('/').splitn(2, '/');
                let container = parts.next().unwrap_or_default().to_string();
                let key = parts.next().unwrap_or("").to_string();
                if account.is_empty() || container.is_empty() || key.is_empty() {
                    return Err(anyhow!("invalid az output path {}", output));
                }
                Ok(OutputTarget::Adls {
                    account,
                    container,
                    key,
                })
            }
            "file" => {
                let path = url
                    .to_file_path()
                    .map_err(|_| anyhow!("invalid file url {}", output))?;
                Ok(OutputTarget::Local(path))
            }
            _ => Err(anyhow!("unsupported output scheme {}", url.scheme())),
        }
    } else {
        Ok(OutputTarget::Local(PathBuf::from(output)))
    }
}

fn write_output(target: &OutputTarget, bytes: Vec<u8>) -> Result<()> {
    match target {
        OutputTarget::Local(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, bytes)?;
        }
        OutputTarget::S3 { bucket, key } => {
            let client = build_s3_client(S3Config {
                bucket: bucket.clone(),
                region: None,
                endpoint: None,
                allow_http: None,
                virtual_hosted: None,
                profile: None,
                access_key_id: None,
                secret_access_key: None,
                session_token: None,
            })?;
            let transfer = CloudTransfer::new(client.store, CloudTransferConfig::from_env());
            transfer.upload_bytes(ObjectPath::from(key.as_str()), bytes)?;
        }
        OutputTarget::Gcs { bucket, key } => {
            let client = build_gcs_client(GcsConfig {
                bucket: bucket.clone(),
            })?;
            let transfer = CloudTransfer::new(client.store, CloudTransferConfig::from_env());
            transfer.upload_bytes(ObjectPath::from(key.as_str()), bytes)?;
        }
        OutputTarget::Adls {
            account,
            container,
            key,
        } => {
            let client = build_adls_client(AdlsConfig {
                account: account.clone(),
                container: container.clone(),
                endpoint: None,
                allow_http: None,
                access_key: None,
                sas_key: None,
                bearer_token: None,
                client_id: None,
                client_secret: None,
                tenant_id: None,
            })?;
            let transfer = CloudTransfer::new(client.store, CloudTransferConfig::from_env());
            transfer.upload_bytes(ObjectPath::from(key.as_str()), bytes)?;
        }
    }
    Ok(())
}

fn output_target_label(target: &OutputTarget) -> String {
    match target {
        OutputTarget::Local(path) => path.display().to_string(),
        OutputTarget::S3 { bucket, key } => format!("s3://{}/{}", bucket, key),
        OutputTarget::Gcs { bucket, key } => format!("gs://{}/{}", bucket, key),
        OutputTarget::Adls {
            account,
            container,
            key,
        } => format!("az://{}/{}/{}", account, container, key),
    }
}

fn join_prefix(prefix: &str, name: &str) -> String {
    let trimmed = prefix.trim_end_matches('/');
    if trimmed.is_empty() {
        name.to_string()
    } else {
        format!("{}/{}", trimmed, name)
    }
}

fn print_edge_table_relationships(edge_tables: &[(&TableSpec, &EdgeTableSpec)]) {
    for (table, edge) in edge_tables {
        println!(
            "  {}: {} -> {}; {} -> {}",
            table.name,
            edge.src_fk.edge_column,
            edge.src_fk.ref_table,
            edge.dst_fk.edge_column,
            edge.dst_fk.ref_table
        );
    }
}

fn fmt_column(col: &Option<diskgraph_ingest::graph_config::ColumnSpec>) -> String {
    match col {
        Some(spec) => format!("{}:{:?}", spec.name, spec.ingested_dtype),
        None => "<missing>".to_string(),
    }
}
