use std::path::PathBuf;

use anyhow::{Context, Result};

use diskgraph_train_table::config::JobConfig;
use diskgraph_train_table::pipeline::run_job;

fn main() -> Result<()> {
    let args = parse_args();
    let config = JobConfig::from_path(&args.config_path)
        .with_context(|| format!("load config from {}", args.config_path.display()))?;
    run_job(&config)
}

struct Args {
    config_path: PathBuf,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut config_path: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                if let Some(value) = args.next() {
                    config_path = Some(PathBuf::from(value));
                }
            }
            _ => {}
        }
    }

    let config_path = config_path.unwrap_or_else(|| {
        eprintln!("usage: generate --config /path/to/job.json");
        std::process::exit(2);
    });

    Args { config_path }
}
