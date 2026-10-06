// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use diskgraph_explain::{run, ExplainConfig};
use std::env;
use std::fs;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let config_path = env::args().nth(1).expect("usage: explain <config.json>");
    let bytes = fs::read(&config_path)?;
    let config: ExplainConfig = serde_json::from_slice(&bytes)?;
    run(config)
}
