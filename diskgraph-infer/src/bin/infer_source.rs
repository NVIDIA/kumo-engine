use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::Parser;
use diskgraph_infer::{infer_source_metadata, parse_root_source, InferOptions};
use diskgraph_ingest::source_metadata::{
    relationship_name, EdgeTableSortKeysSpec, EdgeTableSpec, EdgeSortKeysSpec, RelationshipSpec,
    SortKeySpec, TableSpec, TimestampEncodingKind, WeightEncodingKind,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Infer source_metadata.json from CSV/Parquet tables.")]
struct Args {
    /// Root directory or cloud URL (s3://, gs://, az://).
    #[arg(long)]
    input: String,
    /// Output path for source_metadata.json (defaults to <input>/source_metadata.json for local dirs).
    #[arg(long)]
    output: Option<PathBuf>,
    /// Validate schema consistency across all files in each table.
    #[arg(long)]
    validate_all: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = parse_root_source(&args.input)?;
    let options = InferOptions {
        validate_dir_schemas: args.validate_all,
        ..InferOptions::default()
    };
    eprintln!("infer_source: inferring metadata");
    let result = infer_source_metadata(&root, options)?;
    let output_path = match args.output {
        Some(path) => path,
        None => default_output(&args.input, &root)?,
    };
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    eprintln!("infer_source: writing {}", output_path.display());
    std::fs::write(&output_path, serde_json::to_vec_pretty(&result.metadata)?)?;
    println!("Tables:");
    for table in &result.metadata.tables {
        let format = match table.format {
            diskgraph_ingest::source_metadata::TableFormat::Csv => "csv",
            diskgraph_ingest::source_metadata::TableFormat::Parquet => "parquet",
        };
        let pk = table.primary_key.as_deref().unwrap_or("none");
        let create_time = table.create_time.as_deref().unwrap_or("none");
        println!(
            "  {} (format={}, files={}, primary_key={}, create_time={})",
            table.name,
            format,
            table.paths.len(),
            pk,
            create_time
        );
    }
    if !result.metadata.relationships.is_empty() {
        println!("Relationships:");
        print_relationships(
            &result.metadata.relationships,
            result.metadata.edge_sort_keys.as_ref(),
        );
    }
    let edge_tables: Vec<_> = result
        .metadata
        .tables
        .iter()
        .filter_map(|table| table.edge_table.as_ref().map(|edge| (table, edge)))
        .collect();
    if !edge_tables.is_empty() {
        println!("Edge table relationships:");
        print_edge_table_relationships(
            &edge_tables,
            result.metadata.edge_sort_keys.as_ref(),
        );
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
    println!("Wrote {}", output_path.display());
    if let Some(cmd) = ingest_command_hint(&root, &args.input) {
        println!("Next: {}", cmd);
    }
    Ok(())
}

fn default_output(input: &str, root: &diskgraph_infer::RootSource) -> Result<PathBuf> {
    let path = match root {
        diskgraph_infer::RootSource::Local(path) => path.join("source_metadata.json"),
        _ => PathBuf::from("source_metadata.json"),
    };
    if path.as_os_str().is_empty() {
        return Err(anyhow!("invalid output for input {}", input));
    }
    Ok(path)
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

fn print_relationships(rels: &[RelationshipSpec], keys: Option<&EdgeSortKeysSpec>) {
    for rel in rels {
        let name = rel.name.as_deref().unwrap_or("foreignkey");
        println!(
            "  {}: {}.{} -> {}.{}",
            name, rel.src_table, rel.src_column, rel.dst_table, rel.dst_column
        );
        if let Some(keys) = keys {
            if let Some(spec) = keys.relationships.get(&relationship_name(rel)) {
                print_directional_keys("forward", spec.forward.as_ref());
                print_directional_keys("reverse", spec.reverse.as_ref());
            }
        }
    }
}

fn print_edge_table_relationships(
    edge_tables: &[(&TableSpec, &EdgeTableSpec)],
    keys: Option<&EdgeSortKeysSpec>,
) {
    for (table, edge) in edge_tables {
        println!(
            "  {}: {} -> {}.{}; {} -> {}.{}",
            table.name,
            edge.src_fk.column,
            edge.src_fk.ref_table,
            edge.src_fk.ref_column,
            edge.dst_fk.column,
            edge.dst_fk.ref_table,
            edge.dst_fk.ref_column
        );
        if let Some(keys) = keys {
            if let Some(spec) = keys.edge_tables.get(&table.name) {
                print_edge_table_keys(spec);
            }
        }
    }
}

fn print_edge_table_keys(spec: &EdgeTableSortKeysSpec) {
    print_directional_keys("src_to_dst", spec.src_to_dst.as_ref());
    print_directional_keys("dst_to_src", spec.dst_to_src.as_ref());
}

fn print_directional_keys(label: &str, spec: Option<&SortKeySpec>) {
    let Some(spec) = spec else {
        return;
    };
    let desc = format_sort_key_spec(spec);
    if desc.is_empty() {
        return;
    }
    println!("    {}: {}", label, desc);
}

fn format_sort_key_spec(spec: &SortKeySpec) -> String {
    let mut parts = Vec::new();
    if let Some(ts) = &spec.timestamp {
        parts.push(format!(
            "ts={}.{} ({})",
            ts.table,
            ts.column,
            format_timestamp_kind(&ts.encoding.kind)
        ));
    }
    if let Some(weight) = &spec.weight {
        parts.push(format!(
            "weight={}.{} ({})",
            weight.table,
            weight.expr,
            format_weight_kind(&weight.encoding.dtype)
        ));
    }
    parts.join(", ")
}

fn format_timestamp_kind(kind: &TimestampEncodingKind) -> &'static str {
    match kind {
        TimestampEncodingKind::Int8 => "int8",
        TimestampEncodingKind::Int16 => "int16",
        TimestampEncodingKind::Int32 => "int32",
    }
}

fn format_weight_kind(kind: &WeightEncodingKind) -> &'static str {
    match kind {
        WeightEncodingKind::Byte => "byte",
        WeightEncodingKind::UInt16 => "uint16",
        WeightEncodingKind::UInt32 => "uint32",
        WeightEncodingKind::Float16 => "float16",
        WeightEncodingKind::Float32 => "float32",
    }
}

fn fmt_column(col: &Option<diskgraph_ingest::source_metadata::ColumnSpec>) -> String {
    match col {
        Some(spec) => format!("{}:{:?}", spec.name, spec.ingested_dtype),
        None => "<missing>".to_string(),
    }
}
