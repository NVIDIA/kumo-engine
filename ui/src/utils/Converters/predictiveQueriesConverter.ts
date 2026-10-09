// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    Metric,
    RestConfigGraphV2GraphDefinition as GraphDefinition,
    TimePeriodEnd,
    TimePeriodStart,
    TrainingJobOutput as TrainingJobOutputNew,
    TrainingWorkflowResponse,
    TrialProgress
} from "gen/v0/types";
import { EditableQuery, FilterOverrideArgs } from "gen/v1/types";
import yaml from "js-yaml";

import {
    ChartPlotPoints,
    EstimatedInfo,
    PredictiveQueryInfo,
    QueryProgressStats,
    TaskType,
    TimePeriodData,
    TrainingJobResults
} from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";
import { parseNumbers } from "utils/Helpers/numberHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { TAG_KEY_ARCHIVED } from "utils/jobs";
import {
    formatToLongLocalDateString,
    formatToShortLocalDateString,
    jobRunTimeFromJobResponse,
    parseDateTime
} from "utils/timeConverter";

export function getFiltersFromBpOverride(data: EditableQuery) {
    let entity_filter_override = "";
    const temporal_entity_filter_override = "";
    let target_filter_override = "";
    if (data?.editable_fields.length === 1) {
        const field = data?.editable_fields[0];
        if (field?.field_name === "Entity filter") {
            entity_filter_override = field?.text;
        } else if (field?.field_name === "Target filter") {
            target_filter_override = field?.text;
        }
    } else if (data?.editable_fields?.length > 1) {
        const entityFilter = data.editable_fields.find(
            (field) => field?.field_name === "Entity filter"
        );
        const targetFilter = data.editable_fields.find(
            (field) => field?.field_name === "Target filter"
        );

        if (entityFilter) {
            entity_filter_override = entityFilter.text;
        }

        if (targetFilter) {
            target_filter_override = targetFilter.text;
        }
    }
    const filterOvrideArgs: FilterOverrideArgs = {
        entity_filter_override: entity_filter_override,
        temporal_entity_filter_override: temporal_entity_filter_override,
        target_filter_override: target_filter_override
    };
    return filterOvrideArgs;
}

export function convertResultsToTrainingJobResults(
    results: TrainingJobResults | null | undefined,
    taskType?: string
): TrainingJobResults | undefined {
    if (!results) {
        return undefined;
    }

    // Only add @0.5 threshold for binary and multilabel classification
    // For multiclass, predictions are based on argmax, not a threshold
    const shouldShowThreshold =
        taskType === TaskType.BINARY_CLASSIFICATION ||
        taskType === TaskType.MULTILABEL_CLASSIFICATION ||
        taskType === TaskType.MULTILABEL_RANKING;

    const transformMetrics = (metrics: Metric[]) => {
        return metrics.map((metric) => {
            switch (metric.name) {
                case "f1":
                    return {
                        ...metric,
                        name: shouldShowThreshold
                            ? "F1@0.5 (Macro Avg)"
                            : "F1 (Macro Avg)"
                    };
                case "precision":
                    return { ...metric, name: "PRECISION (Macro Avg)" };
                case "recall":
                    return { ...metric, name: "RECALL (Macro Avg)" };
                default:
                    return metric;
            }
        });
    };

    const transformedResults = { ...results };

    // Apply transformations to all metric types
    if (transformedResults.test_metrics) {
        transformedResults.test_metrics = transformMetrics(
            transformedResults.test_metrics
        );
    }

    if (transformedResults.train_metrics) {
        transformedResults.train_metrics = transformMetrics(
            transformedResults.train_metrics
        );
    }

    if (transformedResults.val_metrics) {
        transformedResults.val_metrics = transformMetrics(
            transformedResults.val_metrics
        );
    }

    return transformedResults;
}

export function generateProgressStatsforChart(
    data: TrialProgress[],
    mode: string
) {
    if (!data || !Array.isArray(data)) {
        return null;
    }
    const xAxisLabels = [];
    const points: ChartPlotPoints = {
        color: "#ff6666",
        value: [],
        label: "Best validation metric"
    };
    const xLabel = "Experiments";
    const yLabel = data.find((trial) => trial.best_train_metric?.name)
        ?.best_train_metric?.name;

    //If the yLabel is null means no data to plot in chart
    if (!yLabel) {
        return null;
    }
    for (let i = 0; i < data.length; i++) {
        if (isValid(data[i]?.best_validation_metric?.val)) {
            const validation = data[i]?.best_validation_metric?.val;
            points.value.push(validation as number);
            xAxisLabels.push(xAxisLabels.length + 1);
        }
    }

    // If no validation metrics were found, return null
    if (points.value.length === 0) {
        return null;
    }

    const pqStatus: QueryProgressStats = {
        xAxisLabels: xAxisLabels,
        datasets: [points],
        xLabel: xLabel,
        yLabel: yLabel,
        mode: mode
    };
    return pqStatus;
}

type JobTimePeriods = NonNullable<
    TrainingJobOutputNew["results"]
>["time_periods"];
type JobSplitSizes = NonNullable<
    TrainingJobOutputNew["results"]
>["split_sizes"];
type Boundary = TimePeriodStart | null | undefined;

/**
 * Pick a boundary from two candidates: when both are valid take the earliest
 * (or latest); otherwise fall back to whichever single one is valid, else "".
 */
function chooseBoundary(
    primary: Boundary,
    secondary: Boundary,
    mode: "earliest" | "latest"
): TimePeriodStart {
    if (primary && secondary && isValid(primary) && isValid(secondary)) {
        const primaryWins =
            mode === "earliest" ? primary < secondary : primary > secondary;
        return primaryWins ? primary : secondary;
    }
    if (isValid(primary)) {
        return primary as string;
    }
    if (isValid(secondary)) {
        return secondary as string;
    }
    return "";
}

/** Resolve the [start, end] training window, normalized via parseDateTimePeriods. */
function computeTrainWindow(tp: JobTimePeriods): {
    start: TimePeriodStart;
    end: TimePeriodEnd;
} {
    let start = chooseBoundary(tp?.val?.start, tp?.train?.start, "earliest");
    let end = chooseBoundary(tp?.val?.end, tp?.train?.end, "latest");
    if (isValid(start) && isValid(end)) {
        const period = parseDateTimePeriods(
            new Date(start).toISOString(),
            new Date(end).toISOString()
        );
        start = period?.start;
        end = period?.end;
    }
    return { start, end };
}

/** Resolve the [start, end] evaluation/test window. */
function computeTestWindow(tp: JobTimePeriods): { start: string; end: string } {
    const testStart = tp?.test?.start;
    const testEnd = tp?.test?.end;
    if (isValid(testStart) && isValid(testEnd)) {
        return {
            start: parseDateTime(testStart as string),
            end: parseDateTime(testEnd as string)
        };
    }
    return { start: "", end: "" };
}

/**
 * Train split size: sum of train+val when both valid, otherwise whichever
 * single one is valid, else null.
 */
function computeTrainSplitSize(splitSizes: JobSplitSizes): number | null {
    const train = splitSizes?.train;
    const val = splitSizes?.val;
    if (train && val && isValid(train) && isValid(val)) {
        return train + val;
    }
    if (isValid(train)) {
        return train as number;
    }
    if (isValid(val)) {
        return val as number;
    }
    return null;
}

/** UI label for a time window — "{start} - {end} ({n} Samples)", or just the
 * sample count for pseudo time periods. Null when the window isn't complete. */
function buildPeriodLabel(
    start: Boundary,
    end: Boundary,
    splitSize: number | null,
    pseudoTime: boolean
): string | null {
    if (!isValid(start) || !isValid(end) || !isValid(splitSize)) {
        return null;
    }
    if (pseudoTime) {
        return `${splitSize} Samples`;
    }
    return `${start} - ${end} (${splitSize} Samples)`;
}

export function timePeriodsFromQueryResponse(
    trainingJob: TrainingJobOutputNew
): TimePeriodData {
    const results = trainingJob?.results;
    const tp = results?.time_periods;
    const pseudoTime = Boolean(
        tp?.test?.is_pseudo || tp?.train?.is_pseudo || tp?.val?.is_pseudo
    );

    let trainStartTime: TimePeriodStart = "";
    let trainEndTime: TimePeriodEnd = "";
    let testStartTime = "";
    let testEndTime = "";
    if (!pseudoTime) {
        ({ start: trainStartTime, end: trainEndTime } = computeTrainWindow(tp));
        ({ start: testStartTime, end: testEndTime } = computeTestWindow(tp));
    }

    const testSplitSize = results?.split_sizes?.test as number;
    const trainSplitSize = computeTrainSplitSize(results?.split_sizes);

    const trainedOnData = buildPeriodLabel(
        trainStartTime,
        trainEndTime,
        trainSplitSize,
        pseudoTime
    );
    const evaluatedOnData = buildPeriodLabel(
        testStartTime,
        testEndTime,
        testSplitSize,
        pseudoTime
    );

    return {
        exists: isValid(trainedOnData) || isValid(evaluatedOnData),
        trainedOnData: trainedOnData,
        evaluatedOnData: evaluatedOnData,
        trainSplitSize: trainSplitSize as number,
        testSplitSize: testSplitSize
    } as TimePeriodData;
}

/* This function takes 2 dates as input and returns output in the format
YYYY-MM-DD HH:MM if the date parts(DD) in the inputs are identical,
otherwise output is in YYYY-MM-DD format */
export function parseDateTimePeriods(
    startTime: string,
    endTime: string
): { start: string; end: string } {
    const timePeriod: { start: string; end: string } = {
        start: "",
        end: ""
    };

    // Slpit date and time
    const startDateSplitText = startTime?.split("T");
    const startDatePart = startDateSplitText[0];
    //Slipt time text
    const startTimeSplitText = startDateSplitText[1]?.split(":");
    // Slpit date and time
    const endDateSplitText = endTime?.split("T");
    const endDatePart = endDateSplitText[0];
    //Slipt time text
    const endTimeSplitText = endDateSplitText[1]?.split(":");
    /* Prepare Date Time in YYYY-MM-DD HH:MM or YYYY-MM-DD format */
    if (new Date(endTime).getDate() === new Date(startTime).getDate()) {
        timePeriod.start = `${startDatePart} ${startTimeSplitText[0]}:${startTimeSplitText[1]} UTC`;
        timePeriod.end = `${endDatePart} ${endTimeSplitText[0]}:${endTimeSplitText[1]} UTC`;
    } else {
        timePeriod.start = `${startDatePart} UTC`;
        timePeriod.end = `${endDatePart} UTC`;
    }

    return timePeriod;
}

const QUERY_DESC_RUNNING =
    "We are running neural architecture search " +
    "experiments to find the most predictive set of parameters " +
    "for your pQuery";
const QUERY_DESC_DONE =
    "We finished running neural architecture " +
    "search experiments, and trained a model with the " +
    "most predictive set of parameters for your pQuery";

type TrainingExecution = NonNullable<
    NonNullable<TrainingWorkflowResponse["training_job"]>["execution"]
>;

interface EstimatesResult {
    estTimings: EstimatedInfo;
    estExperiments: EstimatedInfo;
    progressStats: QueryProgressStats | null;
    queryDescription: string;
}

/**
 * Estimated remaining-time string for a Running job, or null when the
 * progress/start/estimated-end timestamps aren't all available yet.
 */
function computeRemainingTimeText(
    progress: TrainingExecution["progress"],
    execution: TrainingExecution | undefined
): string | null {
    let startTime = null;
    let estimatedEnd = null;
    let jobStartTime = null;
    const currenTime = new Date();

    if (progress && isValid(progress?.start_time)) {
        startTime = new Date(progress.start_time!);
    }
    // During the 1st epoch of the trainer, the startTime is defined but the
    // estimatedEnd is not yet available. In that case we avoid using the
    // estimatedEnd to calculate the remaining time.
    if (progress && isValid(progress?.estimated_end_time)) {
        estimatedEnd = new Date(progress.estimated_end_time as string);
    }
    if (isValid(execution?.start_time)) {
        jobStartTime = new Date(execution!.start_time as string);
    }
    if (
        !estimatedEnd ||
        !jobStartTime ||
        !isValid(startTime) ||
        !isValid(estimatedEnd) ||
        !isValid(jobStartTime)
    ) {
        return null;
    }

    const diffMin = Math.ceil(
        (estimatedEnd.getTime() - jobStartTime.getTime()) / 60000
    );
    let remngTime = parseNumbers(
        Math.ceil((estimatedEnd.getTime() - currenTime.getTime()) / 60000)
    ) as string | number;
    if (remngTime === "-") {
        // parseNumbers returns "-" for value = -1, causing NaN to be shown for
        // estTimings. Handle it here.
        remngTime = "-1";
    }
    return (
        Math.abs(Number(remngTime)) +
        " minutes (out of " +
        parseNumbers(diffMin) +
        ")"
    );
}

/** Estimates/progress for a Done training job. */
function computeDoneEstimates(execution: TrainingExecution): EstimatesResult {
    const result = execution?.results;
    // start_time/last_updated_at are optional on the generated response type;
    // jobRunTimeFromJobResponse already accepts Nullish and normalizes
    // internally, so pass them through rather than wrapping in `new Date()`
    // (which would require narrowing them to non-undefined first).
    const totalTrainingTime = jobRunTimeFromJobResponse(
        execution.start_time,
        execution.last_updated_at
    );
    return {
        queryDescription: QUERY_DESC_DONE,
        estTimings: {
            text: "Total Training Time",
            value: totalTrainingTime as string | number
        },
        estExperiments: {
            text: "Total Experiments",
            value: result?.total_iterations?.toString() as string
        },
        progressStats: generateProgressStatsforChart(
            execution?.progress?.trial_progress as TrialProgress[],
            execution?.progress?.metric_mode as string
        )
    };
}

/** Estimates/progress for a Running training job. */
function computeRunningEstimates(
    execution: TrainingExecution | undefined
): Omit<EstimatesResult, "queryDescription"> {
    const estTimings: EstimatedInfo = {
        text: "Estimated Time Remaining",
        value: "Calculating..."
    };
    const estExperiments: EstimatedInfo = {
        text: "Estimated Experiments Remaining",
        value: "Calculating..."
    };
    let progressStats: QueryProgressStats | null = null;

    if (isValid(execution?.progress)) {
        const progress = execution!.progress;
        const remaining = computeRemainingTimeText(progress, execution);
        if (remaining !== null) {
            estTimings.value = remaining;
        }
        // No. of iterations is correct if the progress is available:
        const totalIteration =
            (progress?.iterations_completed ?? 0) +
            (progress?.iterations_remaining ?? 0);
        estExperiments.value =
            (progress?.iterations_completed ?? 0) + " of " + totalIteration;
        progressStats = generateProgressStatsforChart(
            progress?.trial_progress as TrialProgress[],
            progress?.metric_mode as string
        );
    }
    return { estTimings, estExperiments, progressStats };
}

/** Dispatch estimates by job status, mirroring the original default/Done/Running logic. */
function computeEstimates(
    data: TrainingWorkflowResponse,
    status: string
): EstimatesResult {
    const execution = data?.training_job?.execution;
    if (status === jobStatus.Done && isValid(execution?.results)) {
        return computeDoneEstimates(execution as TrainingExecution);
    }
    if (status === jobStatus.Running) {
        return {
            ...computeRunningEstimates(execution),
            queryDescription: QUERY_DESC_RUNNING
        };
    }
    return {
        queryDescription: QUERY_DESC_RUNNING,
        estTimings: { text: "Estimated Time Remaining", value: "N/A" },
        estExperiments: {
            text: "Estimated Experiments Remaining",
            value: "N/A"
        },
        progressStats: null
    };
}

/** Identity, query-string, and date fields. */
function buildIdentityFields(
    data: TrainingWorkflowResponse
): Partial<PredictiveQueryInfo> {
    const trainingJob = data?.training_job;
    const execution = trainingJob?.execution;
    const queryYaml = data?.pquery_definition?.query_string;
    const jobId = trainingJob?.id;
    return {
        id: jobId,
        queryName: jobId,
        queryName_link: "",
        queryYaml: queryYaml,
        query_dialect: queryYaml,
        trainingConfigYaml: queryYaml,
        dateCreated: formatToShortLocalDateString(
            execution?.start_time ?? undefined
        ),
        dateUpdated: formatToLongLocalDateString(execution?.last_updated_at),
        created: execution?.start_time ?? undefined,
        updated: execution?.last_updated_at,
        trainingJobId: jobId,
        latestTrainingJobId: jobId,
        queryId: trainingJob?.pquery_id
    };
}

/** Graph, plan, and generation-job fields. */
function buildGraphAndPlanFields(
    data: TrainingWorkflowResponse
): Partial<PredictiveQueryInfo> {
    const trainingJob = data?.training_job;
    const genJob = data?.generation_job;
    const pqueryDef = data?.pquery_definition;
    return {
        modelPlannerYaml: trainingJob
            ? yaml.dump(trainingJob.advanced_options)
            : null,
        generationPlan: genJob?.config ? yaml.dump(genJob.config.plan) : null,
        trainingPlan: trainingJob
            ? yaml.dump(trainingJob.advanced_options)
            : null,
        graphName: data?.legacy_graph_id as string,
        graphSnapshotId:
            data?.graph_snapshot_id ?? genJob?.config?.graph_snapshot_id,
        trainTableJobId: genJob?.job_id as string,
        trainTableJobStatus: genJob?.job_status_report?.status,
        graphDefinition: (pqueryDef?.graph as GraphDefinition) ?? null,
        graphVersion: trainingJob?.graph_data_version ?? null
    };
}

/** Status, results, and training-progress fields. */
function buildStatusFields(
    data: TrainingWorkflowResponse
): Partial<PredictiveQueryInfo> {
    const trainingJob = data?.training_job;
    const execution = trainingJob?.execution;
    return {
        trainingStatus: execution?.status,
        log: execution?.log,
        // This logic is handled in the backend for queries
        trainingStatusMessage: (execution?.failure_info
            ? execution?.failure_info?.message
            : execution?.log?.detail) as string,
        taskType: data?.task_type,
        queryType: data?.query_type,
        results: convertResultsToTrainingJobResults(
            execution?.results as TrainingJobResults,
            data?.task_type
        ),
        trialProgress: execution?.progress
    };
}

/** Time-period and job-metadata (tags / version / user) fields. */
function buildMetaFields(
    data: TrainingWorkflowResponse,
    timePeriods: TimePeriodData | null
): Partial<PredictiveQueryInfo> {
    const trainingJob = data?.training_job;
    return {
        timePeriods: timePeriods?.exists,
        trainedOnData: timePeriods?.trainedOnData as string,
        evaluatedOnData: timePeriods?.evaluatedOnData as string,
        testSplitSize: timePeriods?.testSplitSize,
        trainSplitSize: timePeriods?.trainSplitSize,
        tags: trainingJob?.tags,
        archived: trainingJob?.tags?.archived === TAG_KEY_ARCHIVED,
        kumoVersion: trainingJob?.kumo_version,
        user: trainingJob?.user
    };
}

export function parseTrainingWorkflowResponse(
    data: TrainingWorkflowResponse
): PredictiveQueryInfo {
    const execution = data?.training_job?.execution;
    const status = execution?.status ?? "";
    const timePeriods = timePeriodsFromQueryResponse(execution);

    const statusKey = execution?.failure_info
        ? "Reason for Failure:"
        : "Progress";

    const { estTimings, estExperiments, progressStats, queryDescription } =
        computeEstimates(data, status);

    let runTime = "-";
    if (
        status !== jobStatus.Queued &&
        status !== jobStatus.Running &&
        status !== jobStatus.Untrained
    ) {
        runTime = jobRunTimeFromJobResponse(
            execution?.start_time,
            execution?.last_updated_at
        );
    }

    // TODO(frg100): Handle CTA / graphs
    return {
        ...buildIdentityFields(data),
        ...buildGraphAndPlanFields(data),
        ...buildStatusFields(data),
        ...buildMetaFields(data, timePeriods),
        queryDescription: queryDescription,
        advancedOptionsYaml: "",
        statusKey: statusKey,
        callToActions: [],
        runtime: runTime,
        runMode: null,
        description: "",
        estTimings: estTimings,
        estExperiments: estExperiments,
        queryProgressStats: progressStats,
        status: status,
        isV2: true,
        jobSource: data?.job_source
    } as PredictiveQueryInfo;
}
