// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useGetEvalXaiById } from "gen/v0/api/evaluation-statistics/evaluation-statistics";
import { useGetTrainingJobsXaiById } from "gen/v0/api/training-jobs/training-jobs";
import { useGetV1BatchPredictionJobsXaiById } from "gen/v1/api/batch-prediction-job/batch-prediction-job";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useXai = ({
    trainingJobId,
    queryId,
    bpjobId
}: {
    trainingJobId: string | undefined;
    queryId: string | undefined;
    bpjobId: string | undefined;
}) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    // Training Job XAI
    const {
        data: trainingJobsXaiResponse,
        isLoading: trainingJobsXaiLoading,
        error: trainingJobsXaiError,
        isError: isTrainingJobsXaiError
    } = useGetTrainingJobsXaiById(
        trainingJobId ?? "",
        getQueryOptionsById(
            ReactQueryKeys.TrainingJobsXaiById,
            trainingJobId,
            Boolean(projectId && trainingJobId)
        )
    );

    // Evaluation XAI
    const {
        data: evalXaiResponse,
        isLoading: evalXaiLoading,
        error: evalXaiError,
        isError: isEvalXaiError
    } = useGetEvalXaiById(
        queryId ?? "",
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.EvalXaiById,
            queryId,
            Boolean(projectId && queryId && !trainingJobId)
        )
    );

    // Batch Prediction XAI
    const {
        data: bpXaiResponse,
        isLoading: bpXaiLoading,
        error: bpXaiError,
        isError: isBpXaiError
    } = useGetV1BatchPredictionJobsXaiById(
        bpjobId ?? "",
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.BatchPredictionJobsXaiById,
            bpjobId,
            Boolean(projectId) && !!bpjobId && !trainingJobId && !queryId
        )
    );

    const XaiResponse =
        trainingJobsXaiResponse || evalXaiResponse || bpXaiResponse;
    const isLoading = trainingJobsXaiLoading || bpXaiLoading || evalXaiLoading;
    const isError = isTrainingJobsXaiError || isEvalXaiError || isBpXaiError;
    const error = trainingJobsXaiError || evalXaiError || bpXaiError;

    return { data: XaiResponse, isLoading, isError, error };
};
