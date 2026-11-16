// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    GeneratePredictionTableJobResourceTags,
    GeneratePredictionTableRequest,
    JobEventLogEntry,
    JobSource as GeneratedJobSource,
    JobStatusReport,
    KumoapiCommonJobStatus,
    KumoCoreTypesEnumsTaskType
} from "gen/v0/types";
import {
    BatchPredictionJobConfigOutputsItem,
    BatchPredictionJobOutput,
    FilterOverrideArgs,
    GraphLink,
    OutputArtifactStats
} from "gen/v1/types";

import { JobStatusType } from "utils/Constants";
import { Nullable } from "utils/Helpers/typeGuards";

import { JobSource } from "./jobs";
import { PredictiveQueryID, TaskType } from "./predictiveQueries";
import { TrainingJobID } from "./training_job";
import { CTAInfo, ErrorInfo } from "./warnings";

export type BatchPredictionJobAnchorTime = {
    anchor_time: string | null;
};
export type BatchPredictionJobID = string;

export type BatchPredictionJobOutputJs = {
    artifact_type: string;
    directory: string;
    table_schema: string;
    table_name: string;
    extension: string;
    errorInfo: ErrorInfo;
};

export type UnlinkedBatchPredictionJobData = CTAInfo & {
    id: string;
    description: string;
    workflow_id: Nullable<string>;
    name?: string;
    predictiveQuery: string;
    predictiveQueryId: PredictiveQueryID;
    predictiveQuery_link: string;
    status: JobStatusType;
    location?: string;
    log?: JobEventLogEntry;
    created: string;
    dateCreated?: string;
    /** Raw ISO timestamp used as the sort key, never a display string. */
    sortCreated: string;
    numRows?: string | number | null;
    numCols?: string | number;
    num_rows_original: string | number;
    outputs?: BatchPredictionJobConfigOutputsItem[];
    outputsList?: string;
    runtime: string;
    id_link?: string;
    taskType?: TaskType | KumoCoreTypesEnumsTaskType;
    threshold?: number | null;
    threshold_lable?: boolean;
    graphName?: string;
    graphId?: string;
    graphSnapshotId?: string | null;
    failureInfo?: string;
    stacktrace?: string;
    size?: number | string;
    keyColumns?: string | null;
    filterOverrideArgs?: FilterOverrideArgs | null;
    artifactStats?: OutputArtifactStats[] | null;
    outputPaths?: string[];
    classes?: number | null;
    numParallelWorkersLimit?: number;
    isJobUI?: boolean;
    trainingJobID?: TrainingJobID;
    source?: JobSource | GeneratedJobSource | null;
    tags?: { [key: string]: string };
    execution?: BatchPredictionJobOutput;
    generatePredTableJobId?: string | null;
    explanations?: boolean;
    archived?: boolean;
    user?: string | null;
    graph_link?: GraphLink;
    kumoVersion?: string | null;
    downloadSupported?: boolean;
};
export interface PredictionTableGeneration {
    jobId: string;
    jobStatusReport: JobStatusReport;
    createdAt: string;
    created: string;
    dateCreated?: string;
    tags?: GeneratePredictionTableJobResourceTags;
    config: GeneratePredictionTableRequest;
    source?: string;
    queryYaml?: string;
    runTime?: string;
    graphSnapshotId: Nullable<string>;
    status: KumoapiCommonJobStatus;
    user?: string | null;
}
