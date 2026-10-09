// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BatchPredictionJobResponse, OutputArtifactStats } from "gen/v1/types";

import { UnlinkedBatchPredictionJobData } from "types/batchPredictions";
import { TaskType, TrainingJobResults } from "types/predictiveQueries";

import { getProjectPath, jobStatus } from "utils/Constants";
import { bytesToSize, parseNumbers } from "utils/Helpers/numberHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { TAG_KEY_ARCHIVED } from "utils/jobs";
import { isObjectStoreOutputStorage } from "utils/objectStoreTablePath";
import {
    formatToShortLocalDateString,
    jobRunTimeFromJobResponse
} from "utils/timeConverter";

/*
    Starting from #12167 we use output_artifact_stats to return
    stats instead of the raw fields which we're deprecating.
    This function will use the new stats if available, otherwise
    fall back to the old stats for backwards compatibility
*/
export const getOutputArtifactStatsFromResults = (
    results: TrainingJobResults
) => {
    const artifactStats = results?.output_artifact_stats;
    const hasArtifactStats = artifactStats && artifactStats.length > 0;
    const outputPaths = hasArtifactStats
        ? artifactStats.map((stat) => stat?.output_path)
        : results?.output_paths;
    const totalNumRows = hasArtifactStats
        ? artifactStats.reduce((curr, stat) => curr + stat?.num_rows, 0)
        : results?.num_rows;
    const totalNumCols = hasArtifactStats
        ? artifactStats.reduce((curr, stat) => curr + stat?.num_cols, 0)
        : results?.num_cols;
    const totalSizeBytes = hasArtifactStats
        ? artifactStats.reduce((curr, stat) => curr + stat?.size_bytes, 0)
        : results?.size_bytes;
    return {
        artifactStats,
        outputPaths,
        totalNumRows,
        totalSizeBytes,
        totalNumCols
    };
};

interface UnlinkedDerived {
    isJobUI: boolean;
    preditiveQueryId: string | null | undefined;
    graphIdParsed: string | undefined;
    outputsList: string;
    runTime: string;
    numRows: string | number | null;
    numCols: string | number;
    size: string | number;
    keyColumn: string[] | null | undefined;
    outputPaths: string[];
    artifactStats: OutputArtifactStats[] | null | undefined;
}

/** Concatenate each configured output into a human-readable list string. */
function buildOutputsList(pred: BatchPredictionJobResponse): string {
    let outputsList = "";
    pred?.config?.outputs?.forEach((e) => {
        outputsList += e?.artifact_type + ": ";
        if (isObjectStoreOutputStorage(e?.storage_type) && "file_path" in e) {
            outputsList += e.file_path;
        }
        if (
            e?.storage_type === "SNOWFLAKE" &&
            "connector_id" in e &&
            "table_name" in e
        ) {
            outputsList += e?.connector_id + "." + e?.table_name;
        }
    });
    return outputsList;
}

/** Run time, or "-" for queued/running/untrained jobs. */
function computeUnlinkedRunTime(pred: BatchPredictionJobResponse): string {
    const status = pred?.execution?.status;
    if (
        status !== jobStatus.Queued &&
        status !== jobStatus.Running &&
        status !== jobStatus.Untrained
    ) {
        return jobRunTimeFromJobResponse(
            pred?.execution?.start_time,
            pred?.execution?.last_updated_at
        );
    }
    return "-";
}

/** Row/col/size/key/output-path stats derived from the job results. */
function computeUnlinkedOutputStats(
    results: TrainingJobResults | null | undefined
): Pick<
    UnlinkedDerived,
    | "numRows"
    | "numCols"
    | "size"
    | "keyColumn"
    | "outputPaths"
    | "artifactStats"
> {
    if (!isValid(results)) {
        return {
            numRows: null,
            numCols: "N/A",
            size: "N/A",
            keyColumn: null,
            outputPaths: [],
            artifactStats: null
        };
    }
    const stats = getOutputArtifactStatsFromResults(
        results as TrainingJobResults
    );
    return {
        numRows: parseNumbers(stats?.totalNumRows, { millionSystem: true }),
        numCols: parseNumbers(stats?.totalNumCols, {
            millionSystem: true
        }) as string,
        size: bytesToSize(stats?.totalSizeBytes as number),
        keyColumn: results?.key_cols,
        outputPaths: stats?.outputPaths as string[],
        artifactStats: stats?.artifactStats
    };
}

/** Identity, status, and timestamp fields. */
function buildUnlinkedJobCoreFields(
    pred: BatchPredictionJobResponse,
    d: UnlinkedDerived,
    projectId?: string | null
): Partial<UnlinkedBatchPredictionJobData> {
    const projectPaths = getProjectPath(projectId);
    return {
        id: pred?.id,
        workflow_id: pred?.batch_prediction_workflow_id,
        description: pred?.description ?? "",
        predictiveQuery: pred?.predictive_query_link?.name,
        predictiveQueryId: d.preditiveQueryId as string,
        predictiveQuery_link: d.isJobUI
            ? projectPaths.TrainingJob + d.preditiveQueryId
            : projectPaths.TrainingJob +
              pred.predictive_query_link?.training_job_id,
        status: (pred?.execution?.status as jobStatus) ?? jobStatus.None,
        created: pred.timestamps?.created_at,
        dateCreated: formatToShortLocalDateString(pred.timestamps?.created_at),
        // Sort key stays raw ISO so ordering never depends on the display
        // format or the viewer's timezone.
        sortCreated: pred.timestamps?.created_at ?? "",
        isJobUI: d.isJobUI,
        log: pred?.execution?.log
    };
}

/** Output stats, artifacts, and runtime fields. */
function buildUnlinkedJobOutputFields(
    pred: BatchPredictionJobResponse,
    d: UnlinkedDerived,
    projectId?: string | null
): Partial<UnlinkedBatchPredictionJobData> {
    return {
        numRows: d.numRows,
        numCols: d.numCols,
        num_rows_original: (pred?.execution.results
            ? pred?.execution?.results?.num_rows
            : "-") as string,
        outputsList: d.outputsList,
        outputs: pred?.config?.outputs,
        runtime: d.runTime,
        id_link: getProjectPath(projectId).ModelJobsList,
        size: d.size,
        taskType: pred?.config?.task_type,
        keyColumns: d.keyColumn ? d.keyColumn.join(", ") : null,
        outputPaths: d.outputPaths,
        artifactStats: d.artifactStats as unknown as OutputArtifactStats[]
    };
}

/** Graph, snapshot, and prediction-config fields. */
function buildUnlinkedJobGraphFields(
    pred: BatchPredictionJobResponse,
    d: UnlinkedDerived
): Partial<UnlinkedBatchPredictionJobData> {
    return {
        graphId: pred?.graph_link?.id as string,
        graphName: pred?.graph_link?.name,
        explanations: pred?.config?.explanations,
        graphSnapshotId:
            pred.config?.graph_data_version === null
                ? null
                : "graph-" +
                  d.graphIdParsed +
                  "@" +
                  pred.config?.graph_data_version,
        classes: pred?.config?.num_classes_to_return,
        numParallelWorkersLimit: pred?.config?.num_parallel_workers_limit,
        trainingJobID: pred?.config?.training_job_id
    };
}

/** Tags, ownership, and version metadata fields. */
function buildUnlinkedJobMetaFields(
    pred: BatchPredictionJobResponse
): Partial<UnlinkedBatchPredictionJobData> {
    return {
        tags: pred?.tags,
        archived: pred?.tags?.archived === TAG_KEY_ARCHIVED,
        execution: pred?.execution,
        source: pred?.job_source,
        generatePredTableJobId: pred?.config?.generate_pred_table_job_id,
        user: pred?.user,
        graph_link: pred?.graph_link,
        kumoVersion: pred?.kumo_version
    };
}

/** Mutate the job with binary-classification threshold and failure details. */
function applyFailureAndThreshold(
    prediction: UnlinkedBatchPredictionJobData,
    pred: BatchPredictionJobResponse
): void {
    if (prediction.taskType === TaskType.BINARY_CLASSIFICATION) {
        prediction.threshold = pred?.config?.binary_classification_threshold;
        prediction.threshold_lable = true;
    }
    const failureInfo = pred?.execution?.failure_info;
    prediction.failureInfo = isValid(failureInfo)
        ? (failureInfo?.message ?? "")
        : "";
    if (failureInfo && failureInfo?.stacktrace) {
        prediction.stacktrace = failureInfo?.stacktrace;
    }
}

function buildUnlinkedPredictionJob(
    pred: BatchPredictionJobResponse,
    projectId?: string | null
): UnlinkedBatchPredictionJobData {
    const d: UnlinkedDerived = {
        isJobUI: !pred.config.predictive_query_id.startsWith("NA"),
        preditiveQueryId: pred.predictive_query_link?.id,
        graphIdParsed: pred?.graph_link?.id?.split("-")[1],
        outputsList: buildOutputsList(pred),
        runTime: computeUnlinkedRunTime(pred),
        ...computeUnlinkedOutputStats(pred?.execution?.results)
    };
    const prediction: UnlinkedBatchPredictionJobData = {
        ...buildUnlinkedJobCoreFields(pred, d, projectId),
        ...buildUnlinkedJobOutputFields(pred, d, projectId),
        ...buildUnlinkedJobGraphFields(pred, d),
        ...buildUnlinkedJobMetaFields(pred),
        downloadSupported: pred.download_supported === true
    } as UnlinkedBatchPredictionJobData;
    applyFailureAndThreshold(prediction, pred);
    return prediction;
}

export function parseUnlinkedPredictionJobs(
    predictions: Array<BatchPredictionJobResponse>,
    projectId?: string | null
) {
    const tmpArray = (predictions ?? []).map((prediction) =>
        buildUnlinkedPredictionJob(prediction, projectId)
    );
    // sort created date column data in the descending order
    tmpArray.sort((a, b) => (a?.sortCreated > b?.sortCreated ? -1 : 1));
    return tmpArray;
}
