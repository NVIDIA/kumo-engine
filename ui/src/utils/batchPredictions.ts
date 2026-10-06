// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BatchPredictionJobOutputJs } from "types/batchPredictions";

import { getIdHash } from "./Helpers/stringHelpers";
import { outputStorage } from "./Constants";

export const initializeOutputs = (
    jobID: string,
    artifact: string,
    storageType: string,
    outputs: {
        [key: string]: BatchPredictionJobOutputJs;
    } | null = null
) => {
    let tmpOutputs: { [key: string]: BatchPredictionJobOutputJs };
    const selectedQueryId = getIdHash(jobID);
    let tableName = selectedQueryId + "_" + artifact;

    if (
        storageType === outputStorage.Snowflake ||
        storageType === outputStorage.Bigquery ||
        storageType === outputStorage.Databricks
    ) {
        tableName = tableName?.toUpperCase().replaceAll(/[^A-Z0-9]/g, "_");
    } else {
        tableName = tableName?.toLowerCase().replaceAll(/ /g, "-");
    }
    if (outputs == null) {
        tmpOutputs = {};
    } else {
        tmpOutputs = outputs;
    }
    tmpOutputs[artifact] = {
        artifact_type: artifact,
        directory: "",
        table_schema: "",
        table_name: tableName,
        extension: ".parquet",
        errorInfo: { color: "", info: "" }
    };
    return tmpOutputs;
};
