// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JobSource as GeneratedJobSource } from "gen/v0/types";
import { RunMode } from "gen/v1/types";

import { JSGraph } from "./graphs";

// Creating custom `JobSource` with `StatedbConfigCommonJobSource`, `NONE` and
// `WEB` since backend doesn't have these fields.
export type JobSource = GeneratedJobSource | "" | "WEB";
export const JobSource = {
    ...GeneratedJobSource,
    NONE: "",
    WEB: "WEB"
} as const;

export enum TagType {
    TRAINING_JOB = "TRAINING_JOB",
    BATCH_PREDICTION_JOB = "BATCH_PREDICTION_JOB"
}

export enum JobOverviewType {
    TRAINING_JOB = "TRAINING_JOB",
    BATCH_PREDICTION_JOB = "BATCH_PREDICTION_JOB",
    TRAIN_TABLE_JOB = "TRAIN_TABLE_JOB",
    PREDICTION_TABLE_JOB = "PREDICTION_TABLE_JOB",
    GRAPH_SNAPSHOT_JOB = "GRAPH_SNAPSHOT_JOB"
}

export type CreateTrainingJobArgs = {
    name: string;
    graph_id: string;
    query_yaml: string;
    query_dialect: string;
    model_planner_yaml: string;
    description: string;
    run_mode: RunMode | null;
    graph_name: string | null;
    graph?: JSGraph | null;
    user?: string | null;
    // Set when the user edits an existing graph template before training.
    // Passed to the backend to store "<name> (modified)" as the snapshot graph name.
    base_graph_name?: string | null;
};
