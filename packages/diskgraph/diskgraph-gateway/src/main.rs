// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;
use diskgraph_gateway::SamplerGateway;
use std::{net::IpAddr, path::PathBuf};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

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
    /// Atomically publish the bound port and process id after binding.
    #[arg(long)]
    ready_file: Option<PathBuf>,
    /// Exit when this owning process disappears (used by the Python launcher).
    #[arg(long)]
    parent_pid: Option<u32>,
}
async fn parent_exited(parent: Option<u32>) {
    if let Some(expected) = parent {
        #[cfg(unix)]
        loop {
            // getppid reads process state and does not dereference caller memory.
            if unsafe { libc::getppid() } as u32 != expected {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
        #[cfg(not(unix))]
        let _ = expected;
    }
    std::future::pending::<()>().await;
}

#[tokio::main(worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let listener = tokio::net::TcpListener::bind((args.host, args.port)).await?;
    let address = listener.local_addr()?;
    let gateway = SamplerGateway::new(
        args.data,
        args.max_streams,
        args.max_threads,
        address.port(),
    )?;
    let shutdown = gateway.clone();
    let service = arrow_flight::flight_service_server::FlightServiceServer::new(gateway)
        .max_decoding_message_size(128 * 1024 * 1024)
        .max_encoding_message_size(128 * 1024 * 1024);
    if let Some(path) = args.ready_file {
        let temporary = path.with_extension("tmp");
        std::fs::write(
            &temporary,
            serde_json::to_vec(
                &serde_json::json!({"port":address.port(),"pid":std::process::id()}),
            )?,
        )?;
        std::fs::rename(temporary, path)?;
    }
    tracing::info!(%address,pid=std::process::id(),"SAMPLING_LISTENER_READY");
    Server::builder()
        .add_service(service)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
            #[cfg(unix)]
            {
                let mut terminate =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("SIGTERM handler");
                tokio::select! {_=terminate.recv()=>{},_=tokio::signal::ctrl_c()=>{},_=parent_exited(args.parent_pid)=>{}}
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
            }
            shutdown.shutdown();
            tracing::info!("SAMPLING_LISTENER_STOPPING");
        })
        .await?;
    tracing::info!("SAMPLING_LISTENER_STOPPED");
    Ok(())
}
