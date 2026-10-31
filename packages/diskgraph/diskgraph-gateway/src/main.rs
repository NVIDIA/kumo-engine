// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use diskgraph_gateway::{SamplingServer, ServerConfig};
use std::{net::IpAddr, path::PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Serve native DiskGraph epoch shards over Arrow Flight")]
struct Args {
    #[arg(short, long)]
    data: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    #[arg(short, long, default_value_t = 50051)]
    port: u16,
    #[arg(long, default_value_t = 1)]
    max_streams: usize,
    #[arg(long, default_value_t = 2)]
    max_threads: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let mut server = SamplingServer::start(ServerConfig {
        index_path: args.data,
        address: (args.host, args.port).into(),
        max_streams: args.max_streams,
        max_threads: args.max_threads,
    })?;
    let signals = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    signals.block_on(async {
        #[cfg(unix)]
        {
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
            tokio::select! {
                _ = terminate.recv() => {},
                result = tokio::signal::ctrl_c() => result?,
            }
        }
        #[cfg(not(unix))]
        tokio::signal::ctrl_c().await?;
        Ok::<(), std::io::Error>(())
    })?;
    server.close()
}
