// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise the actual tonic listener without loading a Python interpreter.
use arrow_flight::{
    flight_service_client::FlightServiceClient, flight_service_server::FlightServiceServer, Action,
    Empty,
};
use diskgraph_gateway::SamplerGateway;
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
