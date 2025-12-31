use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::Parser;
use diskgraph_infer::{infer_source_metadata, parse_root_source, InferOptions};

#[derive(Parser, Debug)]
#[command(author, version, about = "Infer source_metadata.json from CSV/Parquet tables.")]
struct Args {
    /// Root directory or cloud URL (s3://, gs://, az://).
    #[arg(long)]
    input: String,
    /// Output path for source_metadata.json (defaults to <input>/source_metadata.json for local dirs).
    #[arg(long)]
    output: Option<PathBuf>,
    /// Disable schema validation within directories.
    #[arg(long)]
    no_validate: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = parse_root_source(&args.input)?;
    let options = InferOptions {
        validate_dir_schemas: !args.no_validate,
        ..InferOptions::default()
    };
    let result = infer_source_metadata(&root, options)?;
    let output_path = match args.output {
        Some(path) => path,
        None => default_output(&args.input, &root)?,
    };
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output_path, serde_json::to_vec_pretty(&result.metadata)?)?;
    println!("Wrote {}", output_path.display());
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
        for rel in &result.metadata.relationships {
            let name = rel.name.as_deref().unwrap_or("foreignkey");
            println!(
                "  {}: {}.{} -> {}.{}",
                name, rel.src_table, rel.src_column, rel.dst_table, rel.dst_column
            );
        }
    }
    if !result.warnings.ignored_files.is_empty() {
        println!("Ignored files (schema mismatch):");
        for path in &result.warnings.ignored_files {
            println!("  {}", path);
        }
    }
    if !result.warnings.notes.is_empty() {
        println!("Notes:");
        for note in &result.warnings.notes {
            println!("  {}", note);
        }
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
