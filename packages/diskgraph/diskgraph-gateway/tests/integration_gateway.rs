// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise the actual tonic listener without loading a Python interpreter.
use arrow_flight::{
    flight_service_client::FlightServiceClient, flight_service_server::FlightServiceServer, Action,
    Empty,
};
use diskgraph_gateway::{SamplerGateway, SamplingServer, ServerConfig};
use futures::StreamExt;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::Request;

#[tokio::test]
async fn native_listener_reports_its_process_and_shutdown_state() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("index_metadata.json"), b"{}").unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let gateway = SamplerGateway::new(root.path().to_path_buf(), 1, 2, address.port()).unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(FlightServiceServer::new(gateway.clone()))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                let _ = stopped.await;
            }),
    );
    let mut client = FlightServiceClient::new(
        tonic::transport::Endpoint::from_shared(format!("http://{address}"))
            .unwrap()
            .connect()
            .await
            .unwrap(),
    );
    let actions = client
        .list_actions(Request::new(Empty {}))
        .await
        .unwrap()
        .into_inner()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(actions.len(), 2);
    let mut result = client
        .do_action(Action {
            r#type: "health".into(),
            body: vec![].into(),
        })
        .await
        .unwrap()
        .into_inner();
    let body = result.message().await.unwrap().unwrap().body;
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["pid"], std::process::id());
    assert_eq!(value["ready"], true);
    assert_eq!(value["active_streams"], 0);
    gateway.shutdown();
    stop.send(()).unwrap();
    server.await.unwrap().unwrap();
}

#[test]
fn owned_listener_closes_and_releases_its_bound_port() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("index_metadata.json"), b"{}").unwrap();
    let mut server = SamplingServer::start(ServerConfig {
        index_path: root.path().to_path_buf(),
        address: "127.0.0.1:0".parse().unwrap(),
        max_streams: 1,
        max_threads: 1,
    })
    .unwrap();
    assert_eq!(server.health()["pid"], std::process::id());
    assert_eq!(server.health()["ready"], true);
    assert!(std::net::TcpListener::bind(server.address()).is_err());
    server.close().unwrap();
    server.close().unwrap();
    assert_eq!(server.health()["ready"], false);
    let _replacement = std::net::TcpListener::bind(server.address()).unwrap();
}

#[test]
fn owned_listener_rejects_invalid_budgets() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("index_metadata.json"), b"{}").unwrap();
    assert!(SamplingServer::start(ServerConfig {
        index_path: root.path().to_path_buf(),
        address: "127.0.0.1:0".parse().unwrap(),
        max_streams: 0,
        max_threads: 1,
    })
    .is_err());
}
