// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatedbConfigCommonJobStatus as JobStatus } from "gen/v0/types";

export enum sTypes {
    Numerical = "Numerical",
    Categorical = "Categorical",
    Timestamp = "Timestamp",
    Embedding = "Sequence",
    ID = "ID",
    MultiCategorical = "Multicategorical",
    Text = "Text",
    Image = "Image"
}

export enum jobStatus {
    Queued = "QUEUED",
    Running = "RUNNING",
    Done = "DONE",
    Failed = "FAILED",
    Cancelled = "CANCELLED",
    Untrained = "UNTRAINED",
    None = ""
}

export type JobStatusType =
    | jobStatus
    | (typeof JobStatus)[keyof typeof JobStatus];

export enum queryTabType {
    Overview = "Overview",
    Training = "Training",
    Evaluation = "Evaluation",
    Explanation = "Explanation"
}

export enum storageKeys {
    userDetails = "userDetails",
    accessToken = "accessToken",
    refreshToken = "refreshToken",
    lastUsedConn = "lastUsedConn",
    navPath = "navigationPath",
    isLocalUpload = "is_local_upload",
    selectedProject = "selected-project",
    selectedGroup = "selected-group",
    directConnectLoggedOut = "kumoDirectConnectLoggedOut"
}

export enum kumoResource {
    Connector = "Connector",
    Table = "Table",
    Graph = "Graph",
    Project = "Project"
}

export enum outputStorage {
    S3 = "S3",
    GCS = "GCS",
    ADLS = "ADLS",
    Snowflake = "Snowflake",
    Web = "Web Download",
    Databricks = "Databricks",
    Bigquery = "BigQuery",
    LocalDownloadOnly = "Local Download Only"
}

export enum runMode {
    Normal = "NORMAL",
    Fast = "FAST",
    Best = "BEST"
}

export enum artifactType {
    Predictions = "PREDICTIONS",
    Embeddings = "EMBEDDINGS"
}

export enum inputState {
    None = "none",
    Validating = "validating",
    Valid = "valid",
    Invalid = "invalid",
    Warning = "warning"
}

export enum errorStatusCode {
    UnProcessableEntityError = 422,
    InternalServerError = 500,
    // Gateway/proxy errors: the upstream (e.g. nginx) returns these, often with
    // a raw HTML body rather than a structured Kumo error - see #30031.
    BadGateway = 502,
    ServiceUnavailable = 503,
    GatewayTimeout = 504,
    Success = 200
}

export enum dataSplitKeys {
    HoldOut = "test",
    Training = "train",
    Validation = "val"
}

export enum CtaSourceName {
    Graph = "Graph",
    TrainingJob = "Training Job",
    Table = "Table"
}

export enum PageNames {
    Connector = "Connectors",
    Table = "Tables",
    Graph = "Graphs",
    Jobs = "Jobs",
    NewModel = "New Model",
    Predictions = "Predictions"
}
