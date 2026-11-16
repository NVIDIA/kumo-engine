// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    BinaryClassificationMetrics,
    ErrWarnResponse,
    JobEventLogEntry,
    KumoapiCommonJobStatus,
    QueryType as QueryTypeNew,
    RestConfigGraphV2GraphDefinition as GraphDefinition,
    TrainingJobProgress,
    TrainingJobResults as GeneratedTrainingJobResults
} from "gen/v0/types";
import {
    KumoCoreTypesEnumsTaskType as NewTaskType,
    OutputArtifactStats,
    QueryDialect,
    RunMode
} from "gen/v1/types";

import { JobStatusType } from "utils/Constants";

import { Graph } from "./graphs";
import { JobSource } from "./jobs";
import { JobTagKey, JobTagValue, TrainingJobID } from "./training_job";

export type PredictiveQueryID = string;

export type TaskType =
    | (typeof NewTaskType)[keyof typeof NewTaskType]
    | ""
    | "time_to_next_event"
    | "logistic_regression";

export const TaskType = {
    ...NewTaskType,
    LOGISTIC_REGRESSION: "logistic_regression",
    TIME_TO_NEXT_EVENT: "time_to_next_event",
    NONE: ""
} as const;

export enum QueryType {
    STATIC = "static",
    TEMPORAL = "temporal"
}

enum CallToActionLevel {
    WARNING = "WARNING",
    CRITICAL = "CRITICAL"
}

export type CallToAction = {
    message: string;
    level: CallToActionLevel;
};

export type PredictiveQueryInfo = {
    id: string;
    queryName: string;
    queryName_link?: string;
    queryYaml: string;
    query_dialect: string | QueryDialect;
    trainingConfigYaml: string;
    queryDescription: string;
    advancedOptionsYaml: string;
    modelPlannerYaml: string | null;
    runMode: RunMode | null;
    graphName: string;
    created: string | Date | undefined;
    updated: string | Date | undefined;
    dateCreated: string | Date | undefined;
    dateUpdated: string | Date | undefined;
    trainingJobId: TrainingJobID | null;
    latestTrainingJobId: TrainingJobID | null;
    trainingStatus: JobStatusType | undefined;
    statusKey: string;
    log: JobEventLogEntry;
    queryType?: QueryTypeNew | null;
    taskType: TaskType;
    trainingStatusMessage: string;
    callToActions: Array<CallToAction>;
    timePeriods: boolean;
    trainedOnData: string;
    evaluatedOnData: string;
    testSplitSize: number | undefined;
    trainSplitSize: number | undefined;
    runtime: string;
    estTimings?: EstimatedInfo | null;
    estExperiments?: EstimatedInfo | null;
    description?: string;
    queryProgressStats?: QueryProgressStats | null;
    results?: TrainingJobResults;
    status: string | null;
    queryId?: string | null;
    graphSnapshotId?: string | null;
    trainTableJobId?: string;
    trainTableJobStatus?: KumoapiCommonJobStatus;
    isV2: boolean;
    jobSource?: JobSource | null;
    generationPlan?: string | null;
    trainingPlan?: string | null;
    tags?: { [key: JobTagKey]: JobTagValue } | null;
    trialProgress?: TrainingJobProgress | null;
    graphDefinition?: GraphDefinition | null;
    graphVersion?: string | null;
    archived?: boolean;
    kumoVersion?: string | null;
    user?: string | null;
} & Partial<ErrWarnResponse>;

export type ChartPlotPoints = {
    color: string;
    value: Array<number>;
    label: string;
};

export type QueryProgressStats = {
    xAxisLabels: Array<string | number>;
    datasets: Array<ChartPlotPoints>;
    xLabel: string;
    yLabel: string;
    mode: string;
};

export type EstimatedInfo = {
    text: string;
    value: string | number;
};

export type ValidateQueryStringRequest = {
    pquery_resource: {
        name?: string;
        query_string: string;
        graph: Graph;
        query_dialect: QueryDialect;
    };
    model_plan: string;
};

export type ValidateQueryStringResponse = {
    items: [
        {
            type: number;
            description: string;
            title: string;
            cta: null;
        }
    ];
};

export type RegressionHistogramDistribution = {
    label: string;
    value: string;
};

export type ExtendedBinaryClassificationMetrics =
    BinaryClassificationMetrics & {
        // extra UI-only properties
        rocData?: Array<xyCoordinates>;
        prData?: Array<xyCoordinates>;
        gainData?: Array<xyCoordinates>;
        confMatrix?: Array<Array<ConfusionMatrixCell>[]>;
        threshLen?: number;
        random_guess?: number;
    };

export type ExtendedBinaryClassificationClasses = {
    [key: string]: ExtendedBinaryClassificationMetrics;
};

type xyCoordinates = { x: number | string; y: number | string };

export type ConfusionMatrixCell = { count: number; percentage: number };

export type TimePeriods = {
    test?: TimePeriod;
    train?: TimePeriod;
    val?: TimePeriod;
};

export type TimePeriod = {
    start: string;
    end: string;
    is_pseudo: boolean;
};

export type DataSplit = {
    name?: string;
    label?: string;
    value?: string;
    samples?: number | string;
    prediction_horizon?: string;
};

export type TimePeriodData = {
    exists: boolean;
    trainedOnData: string | null;
    evaluatedOnData: string | null;
    trainSplitSize: number | undefined;
    testSplitSize: number | undefined;
};

export type TrainingJobResults = GeneratedTrainingJobResults & {
    key_cols: string[];
    output_artifact_stats?: OutputArtifactStats[] | null;
    num_rows?: number;
    num_cols?: number;
    size_bytes?: number;
    output_paths?: string[];
};

export type PredictiveQueryMetadata = {
    task_type: TaskType;
    query_type: QueryType;
};
