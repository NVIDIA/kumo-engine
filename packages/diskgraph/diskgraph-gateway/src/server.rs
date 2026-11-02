// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned native server threads, shared by the extension and standalone CLI.
use crate::SamplerGateway;
use anyhow::{anyhow, Context};
use arrow_flight::flight_service_server::FlightServiceServer;
use serde_json::Value;
use std::{
    net::{SocketAddr, TcpListener},
    path::PathBuf,
    sync::mpsc,
    thread::{self, JoinHandle},
    time::Duration,
};
use tokio_stream::wrappers::TcpListenerStream;

/// Local graph, network address, and native sampling budgets.
pub struct ServerConfig {
    pub index_path: PathBuf,
    pub address: SocketAddr,
    pub max_streams: usize,
    pub max_threads: usize,
}

/// A Rust Flight listener with its own runtime and bounded shutdown.
pub struct SamplingServer {
    gateway: SamplerGateway,
    address: SocketAddr,
    thread: Option<JoinHandle<anyhow::Result<()>>>,
}

impl SamplingServer {
    /// Bind the listener and start native serving without calling Python.
    pub fn start(config: ServerConfig) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(config.address).context("Bind sampling listener")?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let gateway = SamplerGateway::new(
            config.index_path,
            config.max_streams,
            config.max_threads,
            address.port(),
        )?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("diskgraph-flight")
            .enable_all()
            .build()?;
        let worker = gateway.clone();
        let (ready, receiver) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("diskgraph-flight-owner".into())
            .spawn(move || {
                let result = runtime.block_on(async move {
                    let listener = tokio::net::TcpListener::from_std(listener)?;
                    let _ = ready.send(());
                    serve(listener, worker).await
                });
                runtime.shutdown_timeout(Duration::from_secs(5));
                if let Err(error) = &result {
                    tracing::error!(%error, "SAMPLING_LISTENER_FAILED");
                }
                result
            })?;
        let mut server = Self {
            gateway,
            address,
            thread: Some(thread),
        };
        if receiver.recv().is_err() {
            server.close()?;
            return Err(anyhow!("Sampling listener exited before readiness"));
        }
        Ok(server)
    }

    /// Return the actual bound address, including an automatically selected port.
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// Return native counters and readiness, including background server failure.
    pub fn health(&self) -> Value {
        let mut health = self.gateway.health();
        health["ready"] = Value::Bool(
            self.thread
                .as_ref()
                .is_some_and(|thread| !thread.is_finished())
                && !*self.gateway.state.stop.borrow(),
        );
        health
    }

    /// Cancel producers and join the serving thread; repeated calls are harmless.
    pub fn close(&mut self) -> anyhow::Result<()> {
        self.gateway.shutdown();
        let result = if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| anyhow!("Sampling listener thread panicked"))?
        } else {
            Ok(())
        };
        self.gateway.state.indexes.lock().clear();
        result
    }
}

impl Drop for SamplingServer {
    fn drop(&mut self) {
        // Python can drop a handle while holding the GIL. Never join native
        // logging threads here; explicit close releases the GIL before joining.
        self.gateway.shutdown();
    }
}

async fn serve(listener: tokio::net::TcpListener, gateway: SamplerGateway) -> anyhow::Result<()> {
    let address = listener.local_addr()?;
    let mut stopping = gateway.state.stop.subscribe();
    let mut shutdown = stopping.clone();
    let service = FlightServiceServer::new(gateway)
        .max_decoding_message_size(128 * 1024 * 1024)
        .max_encoding_message_size(128 * 1024 * 1024);
    let serving = tonic::transport::Server::builder()
        .add_service(service)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
            let _ = shutdown.wait_for(|stop| *stop).await;
        });
    tokio::pin!(serving);
    tracing::info!(%address, pid=std::process::id(), "SAMPLING_LISTENER_READY");
    let finished = tokio::select! {
        result = &mut serving => Some(result),
        _ = stopping.wait_for(|stop| *stop) => None,
    };
    if let Some(result) = finished {
        result?;
    } else {
        tracing::info!(%address, "SAMPLING_LISTENER_STOPPING");
        match tokio::time::timeout(Duration::from_secs(5), serving).await {
            Ok(result) => result?,
            Err(_) => tracing::warn!(%address, "SAMPLING_LISTENER_FORCED_CLOSE"),
        }
    }
    tracing::info!(%address, "SAMPLING_LISTENER_STOPPED");
    Ok(())
}
