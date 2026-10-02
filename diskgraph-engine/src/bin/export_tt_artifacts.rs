// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use diskgraph_engine::export_tt_artifacts::{process_export, ExportConfig};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the JSON configuration file
    #[arg(short, long)]
    config: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config_data = std::fs::read_to_string(&args.config)?;
    let config: ExportConfig = serde_json::from_str(&config_data)?;
    process_export(&config)
}
