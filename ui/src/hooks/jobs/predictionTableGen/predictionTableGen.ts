// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";
import {
    useCancelGenpredtableJobsById,
    useGetGenpredtableJobsAnchorTimeById,
    useGetGenpredtableJobsById,
    useGetGenpredtableJobsEntitiesById
} from "gen/v0/api/jobs/jobs";
import { GeneratePredictionTableResponse, JobType } from "gen/v0/types";
import {
    useCreateV1BatchPredictionJobsSamplesForOutputSchema,
    useGetV1BatchPredictionJobsOutputSchema,
    useGetV1BatchPredictionJobsPredictionTableSchemaById
} from "gen/v1/api/batch-prediction-job/batch-prediction-job";
import {
    ColumnArgs,
    KumoCoreTypesEnumsTaskType as GeneratedTaskType,
    QueryType
} from "gen/v1/types";
import { useCacheStaleTime } from "stores/cacheStore";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { PredictionTableGeneration } from "types/batchPredictions";
import { CustomError } from "types/common";
import { TaskType } from "types/predictiveQueries";
import { HandleErrorFunc } from "types/validation";

import { jobStatus } from "utils/Constants";
import { parsePredictionTableGenResponse } from "utils/Converters/jobsConverter";
import { isValid } from "utils/Helpers/typeGuards";
import { TAG_KEY_ARCHIVED } from "utils/jobs";
import {
    invalidateJobQueries,
    invalidateJobsList,
    invalidateSnapshotsGraph
} from "utils/queryInvalidation";
import { ReactQueryKeys } from "utils/reactQueryKeys";

/** Invalidate caches on prediction-table-gen status transitions. */
function handlePredTableStatusTransition(
    queryClient: Parameters<typeof invalidateJobsList>[0],
    prevStatus: jobStatus,
    predictionStatus: jobStatus,
    parsed: PredictionTableGeneration | null,
    id: string
): void {
    if (prevStatus !== jobStatus.None && prevStatus !== predictionStatus) {
        invalidateJobsList(queryClient);
        if (isValid(parsed?.config?.graph_snapshot_id)) {
            invalidateSnapshotsGraph(
                queryClient,
                parsed?.config?.graph_snapshot_id as string
            );
        }
    }
    if (
        prevStatus === jobStatus.None &&
        (predictionStatus === jobStatus.Queued ||
            predictionStatus === jobStatus.Running)
    ) {
        // Invalidate specific job cache on status change so fresh data is
        // fetched and the UI updates immediately for all transitions.
        invalidateJobQueries(
            queryClient,
            JobType.GENERATE_PREDICTION_TABLE_JOB,
            id as string
        );
    }
}

export const useGenPredTableSummary = (
    id: string,
    handleError: HandleErrorFunc,
    setPageLoading: (loading: boolean) => void,
    setStatus?: (status: jobStatus) => void,
    setIsArchived?: (archived: boolean) => void
) => {
    const { queryOption } = useQueryOptions();
    const projectId = useProjectId();
    const [predictionTableDetails, setPredictionTableDetails] =
        useState<PredictionTableGeneration | null>(null);
    const [bpJobId, setBpJobId] = useState<string | null>(null);
    const [queryId, setQueryId] = useState<string | null>(null);
    const [prevStatus, setPrevStatus] = useState<jobStatus>(jobStatus.None);

    const queryClient = useQueryClient();

    const {
        data: response,
        isLoading,
        isError,
        error,
        isFetching
    } = useGetGenpredtableJobsById<{
        data: GeneratePredictionTableResponse;
    }>(
        id,
        { project_id: projectId ?? "" },
        queryOption.genPredTableJobsById(id),
        queryClient
    );

    // handle errors
    useEffect(() => {
        if (isError) {
            invalidateJobsList(queryClient);
            setPageLoading(false);
            handleError(error as CustomError);
        }
    }, [isError, error, handleError, setPageLoading, queryClient]);

    // process data
    useEffect(() => {
        if (isLoading) return;
        setPageLoading(false);
        if (response?.data?.resource) {
            const parsed = parsePredictionTableGenResponse(response?.data);
            setPredictionTableDetails(parsed);
            setBpJobId(response.data.resource?.tags?.parent_job ?? null);
            setQueryId(response.data.resource?.config?.pquery_id ?? null);
            const predictionStatus =
                response.data.resource?.job_status_report?.status;
            handlePredTableStatusTransition(
                queryClient,
                prevStatus,
                predictionStatus as jobStatus,
                parsed,
                id
            );

            setPrevStatus(predictionStatus as jobStatus);
            if (setStatus) setStatus(predictionStatus as jobStatus);
            if (setIsArchived)
                setIsArchived(
                    response.data.resource?.tags?.archived === TAG_KEY_ARCHIVED
                );
        }
    }, [
        isLoading,
        setPageLoading,
        response,
        setStatus,
        setIsArchived,
        queryClient,
        prevStatus,
        id
    ]);

    return {
        isDataLoading: isLoading || isFetching,
        predictionTableDetails,
        bpJobId,
        queryId
    };
};

export const useGenpredtableAnchorTime = (
    id: string,
    predictionTableDetails: PredictionTableGeneration | null,
    handleError: HandleErrorFunc,
    setPageLoading: (loading: boolean) => void
) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId();
    const [anchorTime, setAnchorTime] = useState<string | null>(null);
    const queryClient = useQueryClient();
    const refetchAfterDone = useRef(false);

    const {
        data: response,
        isLoading,
        isError,
        error,
        refetch
    } = useGetGenpredtableJobsAnchorTimeById<{ data: string | null }>(
        id,
        { project_id: projectId ?? "" },
        getQueryOptionsById(ReactQueryKeys.PredtableJobsAnchorTimeById, id),
        queryClient
    );

    // Handle errors
    useEffect(() => {
        if (isError) {
            setPageLoading(false);
            handleError(error as CustomError);
        }
    }, [isError, error, setPageLoading, handleError]);

    // Refetch when job is done and anchor_time is "inferred" or null
    useEffect(() => {
        const status = predictionTableDetails?.status;
        const anchorTimeValue =
            predictionTableDetails?.config?.plan?.anchor_time;
        if (
            status === jobStatus.Done &&
            (anchorTimeValue === "inferred" || anchorTimeValue === null) &&
            !refetchAfterDone.current &&
            !response?.data
        ) {
            refetchAfterDone.current = true;
            refetch();
        }
    }, [predictionTableDetails, refetch, response?.data]);

    // Process anchor time
    useEffect(() => {
        if (isLoading) return;

        setPageLoading(false);

        setAnchorTime((prev) => {
            const anchorTimeFromApi = response?.data;
            if (anchorTimeFromApi) return anchorTimeFromApi;

            if (prev) return prev; // already set
            const anchorTimeValue =
                predictionTableDetails?.config?.plan?.anchor_time;
            if (isValid(anchorTimeValue) && anchorTimeValue !== "inferred") {
                const numValue = Number(anchorTimeValue);
                if (!isNaN(numValue)) {
                    return new Date(numValue * 1000)
                        .toISOString()
                        .split(".")[0];
                }
            }
            return prev;
        });
    }, [
        isLoading,
        response?.data,
        predictionTableDetails?.config?.plan?.anchor_time,
        setPageLoading
    ]);

    return {
        isDataLoading: isLoading,
        anchorTime
    };
};

export const usePredictionTableSchema = (
    predictiveQueryID?: string,
    taskType?: TaskType,
    queryType?: QueryType,
    trainingTableID?: string
): [Array<{ name: string; colType: string }>, boolean] => {
    const canFetchById = Boolean(predictiveQueryID);
    const canFetchByTaskArgs = Boolean(taskType && queryType);
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const schemaByIdQuery =
        useGetV1BatchPredictionJobsPredictionTableSchemaById(
            predictiveQueryID ?? "",
            { project_id: projectId },
            getQueryOptionsById(
                ReactQueryKeys.BpPredictionSchemaById,
                predictiveQueryID,
                canFetchById && Boolean(projectId)
            )
        );
    const schemaByTaskQuery = useGetV1BatchPredictionJobsOutputSchema(
        {
            task_type: taskType as GeneratedTaskType,
            query_type: queryType as QueryType
        },

        getQueryOptionsById(
            ReactQueryKeys.BpPredictionSchemaByTaskType,
            trainingTableID,
            !canFetchById && canFetchByTaskArgs
        )
    );

    const data: ColumnArgs[] =
        schemaByIdQuery.data?.data ?? schemaByTaskQuery.data?.data ?? [];

    const transformToColumnArgs = (
        data: ColumnArgs[]
    ): Array<{ name: string; colType: string }> => {
        return data.map((col) => ({
            name: col.name,
            colType: col.dtype as string
        }));
    };
    const outputSchema = transformToColumnArgs(data);

    const loading = schemaByIdQuery.isLoading || schemaByTaskQuery.isLoading;

    return [outputSchema, loading];
};

export const usePredictionTableSchemaSamples = (
    trainingTableID?: string,
    taskType?: TaskType | null
): [Array<{ name: string; sample_rows: Array<string | number> }>] => {
    const canFetch = Boolean(trainingTableID && taskType);
    const { getQueryOptionsById } = useQueryOptions();

    const query = useCreateV1BatchPredictionJobsSamplesForOutputSchema(
        {
            train_table_job_id: trainingTableID as string,
            task_type: taskType as GeneratedTaskType
        },
        getQueryOptionsById(
            ReactQueryKeys.BpJobsSamplesForOutputSchemaById,
            trainingTableID,
            canFetch
        )
    );
    if (query.isError) {
        console.error("Prediction sample fetch error:", query.error);
    }

    return [query?.data?.data ?? []];
};

interface UseCancelGenpredtableJobsProps {
    bpId: string;
    id: string;
    onSuccess?: () => void;
    handleError: (error: CustomError, title: string) => void;
}

export const useCancelGenpredtableJobs = ({
    bpId,
    id,
    onSuccess,
    handleError
}: UseCancelGenpredtableJobsProps) => {
    const queryClient = useQueryClient();
    const projectId = useProjectId();
    const { mutate: cancelGenpredtableJobs } = useCancelGenpredtableJobsById(
        {
            mutation: {
                onSuccess: () => {
                    invalidateJobsList(queryClient);
                    invalidateJobQueries(
                        queryClient,
                        JobType.GENERATE_PREDICTION_TABLE_JOB,
                        id as string
                    );
                    invalidateJobQueries(
                        queryClient,
                        JobType.BATCH_PREDICTION_JOB,
                        bpId as string
                    );
                    invalidateJobsList(queryClient);
                    onSuccess?.();
                },
                onError: (error) => {
                    if (error) {
                        const title =
                            "Unable to Cancel Prediction Table Generation.";
                        handleError(error as CustomError, title);
                    }
                }
            }
        },
        queryClient
    );

    return {
        cancelGenpredtableJobs: ({ id }: { id: string }) =>
            cancelGenpredtableJobs({
                id,
                params: { project_id: projectId ?? "" }
            })
    };
};

export const usePredictionTableNumEntities = ({
    id,
    status
}: {
    id: string;
    status: jobStatus;
}) => {
    const { long } = useCacheStaleTime();
    const { getQueryOptions } = useQueryOptions();
    const projectId = useProjectId();
    const { data } = useGetGenpredtableJobsEntitiesById(
        id,
        { project_id: projectId ?? "" },
        getQueryOptions({
            queryKey: [ReactQueryKeys.PredictionTableNumEntities, id, status],
            staleTime: long,
            enabled: !!id
        })
    );

    return (data?.data as string | null) || null;
};
