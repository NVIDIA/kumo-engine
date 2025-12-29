use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use diskgraph_ingest::{ingest, Dest, IngestOptions, MappingKind};

#[derive(Parser, Debug)]
#[command(author, version, about = "Ingest raw CSV/Parquet into diskgraph index format.")]
struct Args {
    /// Path to source_metadata.json
    #[arg(long)]
    source: PathBuf,
    /// Output destination root (path or URL).
    #[arg(long)]
    output: String,
    /// Destination kind.
    #[arg(long, value_enum, default_value = "local")]
    dest: DestArg,
    /// Primary key mapping strategy.
    #[arg(long, value_enum, default_value = "hash")]
    mapping: MappingArg,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DestArg {
    Local,
    S3,
    Gcs,
    Adls,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum MappingArg {
    Hash,
    Sorted,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let args = Args::parse();
    let options = IngestOptions {
        mapping_kind: match args.mapping {
            MappingArg::Hash => MappingKind::Hash,
            MappingArg::Sorted => MappingKind::Sorted,
        },
    };
    let dest = match args.dest {
        DestArg::Local => Dest::Local,
        DestArg::S3 => Dest::S3,
        DestArg::Gcs => Dest::Gcs,
        DestArg::Adls => Dest::Adls,
    };
    ingest(&args.source, &args.output, dest, options)?;
    Ok(())
}
