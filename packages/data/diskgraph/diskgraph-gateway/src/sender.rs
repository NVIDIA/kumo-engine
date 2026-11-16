// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use arrow_array::RecordBatch;
use arrow_flight::FlightData;
use arrow_ipc::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
use arrow_schema::Schema;
use std::sync::Arc;
use tokio::sync::mpsc;
use tonic::Status;

pub struct FlightSender {
    sender: mpsc::Sender<Result<FlightData, Status>>,
    options: IpcWriteOptions,
}

impl FlightSender {
    pub fn new(sender: mpsc::Sender<Result<FlightData, Status>>) -> Self {
        Self {
            sender,
            options: IpcWriteOptions::default(),
        }
    }

    pub fn send_schema(&self, schema: Arc<Schema>) -> Result<(), Status> {
        let schema_flight_data = arrow_flight::SchemaAsIpc::new(&schema, &self.options).into();
        self.blocking_send(schema_flight_data)
    }

    pub async fn send_schema_async(&self, schema: Arc<Schema>) -> Result<(), Status> {
        let schema_flight_data = arrow_flight::SchemaAsIpc::new(&schema, &self.options).into();
        self.send_async(schema_flight_data).await
    }

    pub fn send_raw(&self, data: FlightData) -> Result<(), Status> {
        self.blocking_send(data)
    }

    #[allow(deprecated)]
    pub fn send_batch(&self, batch: RecordBatch) -> Result<(), Status> {
        let generator = IpcDataGenerator::default();
        let mut tracker = DictionaryTracker::new(false);
        let (dictionaries, encoded_data) = generator
            .encoded_batch(&batch, &mut tracker, &self.options)
            .map_err(|e| Status::internal(e.to_string()))?;

        for dict in dictionaries {
            let flight_data: FlightData = dict.into();
            self.blocking_send(flight_data)?;
        }

        let flight_data: FlightData = encoded_data.into();
        self.blocking_send(flight_data)?;
        Ok(())
    }

    #[allow(deprecated)]
    pub fn send_batch_with_metadata(
        &self,
        batch: RecordBatch,
        metadata: bytes::Bytes,
    ) -> Result<(), Status> {
        let generator = IpcDataGenerator::default();
        let mut tracker = DictionaryTracker::new(false);
        let (dictionaries, encoded_data) = generator
            .encoded_batch(&batch, &mut tracker, &self.options)
            .map_err(|e| Status::internal(e.to_string()))?;

        for dict in dictionaries {
            let flight_data: FlightData = dict.into();
            self.blocking_send(flight_data)?;
        }

        let mut flight_data: FlightData = encoded_data.into();
        flight_data.app_metadata = metadata;
        self.blocking_send(flight_data)?;
        Ok(())
    }

    #[allow(deprecated)]
    pub async fn send_batch_async(&self, batch: RecordBatch) -> Result<(), Status> {
        let generator = IpcDataGenerator::default();
        let mut tracker = DictionaryTracker::new(false);
        let (dictionaries, encoded_data) = generator
            .encoded_batch(&batch, &mut tracker, &self.options)
            .map_err(|e| Status::internal(e.to_string()))?;

        for dict in dictionaries {
            let flight_data: FlightData = dict.into();
            self.send_async(flight_data).await?;
        }

        let flight_data: FlightData = encoded_data.into();
        self.send_async(flight_data).await?;
        Ok(())
    }

    fn blocking_send(&self, data: FlightData) -> Result<(), Status> {
        self.sender
            .blocking_send(Ok(data))
            .map_err(|_| Status::unavailable("channel closed"))
    }

    async fn send_async(&self, data: FlightData) -> Result<(), Status> {
        self.sender
            .send(Ok(data))
            .await
            .map_err(|_| Status::unavailable("channel closed"))
    }
}
