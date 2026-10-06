// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";
import {
    useCancelGentraintableJobsById,
    useGetGentraintableJobsById,
    useGetGentraintableJobsErrorsById,
    useGetGentraintableJobsGenerationStatsById,
    useGetGentraintableJobsSplitStatsById
} from "gen/v0/api/jobs/jobs";
import { useGetTrainingJobsTrainTableSplitStatsById } from "gen/v0/api/training-jobs/training-jobs";
import { ErrWarnDetails, JobType } from "gen/v0/types";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";
import { TrainTableGenStats } from "types/stats";
import { TrainTableGeneration } from "types/training_job";

import { jobStatus } from "utils/Constants";
import { copyObject, getKeys } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import {
    invalidateJobQueries,
    invalidateJobsList
} from "utils/queryInvalidation";
import { ReactQueryKeys } from "utils/reactQueryKeys";
import { parseDateTime } from "utils/timeConverter";

export const useTrainingTableDetails = (id: string | undefined) => {
    const { queryOption } = useQueryOptions();
    const projectId = useProjectId();
    const {
        data: trainTableGenerationResponse,
        isLoading,
        isError,
        error
    } = useGetGentraintableJobsById(
        id ?? "",
        { project_id: projectId ?? "" },
        queryOption.traintableJobsById(id as string)
    );

    return {
        data: trainTableGenerationResponse,
        isLoading,
        isError,
        error
    };
};

export const useTrainTableGenerationStats = (
    trainingJobId: string | undefined,
    queryStatus: jobStatus
) => {
    const { queryOption } = useQueryOptions();
    const projectId = useProjectId();

    const [trainTableGenStats, setStats] = useState<TrainTableGenStats | null>(
        null
    );

    const {
        data: response,
        isLoading,
        isError
    } = useGetGentraintableJobsGenerationStatsById(
        trainingJobId ?? "",
        { project_id: projectId ?? "" },
        queryOption.traintableJobsGenerationStatsById(
            trainingJobId as string,
            queryStatus
        )
    );

    useEffect(() => {
        if (response && response.data) {
            const stats: TrainTableGenStats = copyObject(
                response.data
            ) as TrainTableGenStats;
            stats?.iteration_data?.forEach((stat) => {
                const tmpKeys = getKeys(stat);
                stat.isActive = false;
                tmpKeys.forEach((key) => {
                    if (["min_date", "max_date", "anchor_date"].includes(key)) {
                        const t = stat[key];
                        if (typeof t === "string") {
                            stat[key] = parseDateTime(t, false, true);
                            stat[key + "_toSort"] = parseDateTime(
                                t,
                                false,
                                true
                            );
                        }
                    }
                });
            });
            setStats(stats);
        }
    }, [response]);

    return {
        stats: trainTableGenStats,
        receivedResponse: isError || !isLoading
    };
};

export const useSplitStats = (
    id: string,
    isSourceSDK: boolean,
    queryStatus: jobStatus
) => {
    const { getQueryOptionsByIdAndJobStatus } = useQueryOptions();
    const projectId = useProjectId();
    const {
        data: trainingSplitStatsResponse,
        isError: istrainingSplitStatsError,
        error: trainingSplitStatsError
    } = useGetTrainingJobsTrainTableSplitStatsById(
        id,
        { project_id: projectId ?? "" },
        getQueryOptionsByIdAndJobStatus(
            ReactQueryKeys.TrainingJobsTrainTableSplitStatsById,
            id,
            !!projectId && !!id && !isSourceSDK,
            queryStatus
        )
    );

    const {
        data: generationSplitStatsResponse,
        isError: isgenerationSplitStatsError,
        error: generationSplitStatsError
    } = useGetGentraintableJobsSplitStatsById(
        id,
        { project_id: projectId ?? "" },
        getQueryOptionsByIdAndJobStatus(
            ReactQueryKeys.TraintableJobsSplitStatsById,
            id,
            !!isSourceSDK && !!id,
            queryStatus
        )
    );

    if (istrainingSplitStatsError || isgenerationSplitStatsError) {
        console.warn(trainingSplitStatsError || generationSplitStatsError);
    }

    const data = isSourceSDK
        ? generationSplitStatsResponse?.data
        : trainingSplitStatsResponse?.data;

    return data;
};

interface UseCancelTrainTableJobProps {
    jobId: string;
    trainingJobId?: string | null;
    onSuccess?: () => void;
    handleError: (error: CustomError, title: string) => void;
}

export const useCancelTrainTableJob = ({
    jobId,
    trainingJobId,
    onSuccess,
    handleError
}: UseCancelTrainTableJobProps) => {
    const queryClient = useQueryClient();
    const projectId = useProjectId();

    const { mutate: cancelTrainTable } = useCancelGentraintableJobsById(
        {
            mutation: {
                onSuccess: () => {
                    invalidateJobQueries(
                        queryClient,
                        JobType.GENERATE_TRAIN_TABLE_JOB,
                        jobId
                    );
                    if (trainingJobId) {
                        invalidateJobQueries(
                            queryClient,
                            JobType.TRAINING_JOB,
                            trainingJobId
                        );
                    }
                    invalidateJobsList(queryClient);
                    onSuccess?.();
                },
                onError: (error) => {
                    if (error) {
                        const title = "Unable to Cancel Training Table Job. ";
                        handleError(error as CustomError, title);
                    }
                }
            }
        },
        queryClient
    );

    return {
        cancelTrainTable: ({ id }: { id: string }) =>
            cancelTrainTable({
                id,
                params: { project_id: projectId ?? "" }
            })
    };
};

export const useTraintableJobErrors = ({
    jobId,
    trainTableDetails
}: {
    jobId?: string;
    trainTableDetails: TrainTableGeneration | null;
}) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId();

    const { data: errorsResponse, isLoading: fetchingErrors } =
        useGetGentraintableJobsErrorsById(
            jobId ?? "",
            { project_id: projectId ?? "" },
            getQueryOptionsById(
                ReactQueryKeys.TraintableJobsErrorsById +
                    "-" +
                    trainTableDetails?.status,
                jobId,
                isValid(jobId)
            )
        );
    const { warnings, errors } = useMemo(() => {
        return {
            errors:
                errorsResponse?.data?.items?.filter(
                    (t) => `${t.type}` !== `1`
                ) ?? [],
            warnings:
                (errorsResponse?.data?.items
                    ?.filter((t) => `${t.type}` === `1`)
                    .map((t) => ({
                        title: t.title,
                        description: t.description,
                        is_dismissible: true,
                        is_error: false
                    })) as ErrWarnDetails[]) ?? []
        };
    }, [errorsResponse?.data?.items]);

    return {
        errors,
        warnings,
        fetchingErrors
    };
};
