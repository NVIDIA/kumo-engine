// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    GenerateTrainTableRequest,
    JobStatusReport,
    KumoapiCommonJobStatus,
    Metric
} from "gen/v0/types";

import { Nullable } from "utils/Helpers/typeGuards";

export type GraphSnapshotID = string;
export type TrainingJobID = string;
export type FPID = string;

export type JobTagKey = string;
export type JobTagValue = string;

export interface TrialProgress {
    trial_id: string;
    best_train_metric?: Metric;
    best_validation_metric?: Metric;
}

export interface TrainTableGeneration {
    jobId: string;
    jobStatusReport: JobStatusReport | null;
    createdAt: string;
    created: string;
    dateCreated: string;
    tags?: JobTags;
    config: GenerateTrainTableRequest;
    source?: string;
    queryYaml?: string;
    runTime?: string;
    graphSnapshotId: Nullable<string>;
    status: KumoapiCommonJobStatus;
    archived: boolean;
    user?: string | null;
}

export interface JobTags {
    [key: string]: string;
}

export type TopKFeatures = {
    table_name: string;
    name: string;
    value: string;
    stype: string;
    score: number;
};

export type TopKFeaturesGlobal = {
    table_name: string;
    name: string;
    value: string;
    stype: string;
    score: number;
};

export type EntityNodePredRfm = {
    entity_table_name: string;
    count: { [key: string]: string | number };
    avg_rel_time: { [key: string]: (string | number)[] };
    most_recent_rel_time: { [key: string]: (string | number)[] };
    unique_counts: { [key: string]: { [key: string]: string | number } };
};

export type CategoryNodePredRfm = {
    entity_table_name: string;
    count: { [key: string]: string | number };
    avg_rel_time: { [key: string]: (string | number)[] };
    most_recent_rel_time: { [key: string]: (string | number)[] };
    unique_counts: { [key: string]: { [key: string]: string | number } };
};
export type SubgraphSummary = {
    property: string;
    current?: string | number;
    global?: string | number;
};
