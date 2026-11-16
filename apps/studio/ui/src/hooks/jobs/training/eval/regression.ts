// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react";

import {
    useGetEvalRegressionHeatmapById,
    useGetEvalRegressionHistogramById
} from "gen/v0/api/evaluation-statistics/evaluation-statistics";
import {
    useGetTrainingJobsRegressionHeatmapById,
    useGetTrainingJobsRegressionHistogramById
} from "gen/v0/api/training-jobs/training-jobs";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useRegressionHeatmap = ({
    queryId,
    trainingJobId,
    handleError
}: {
    queryId: string;
    trainingJobId?: string;
    handleError: (err: Error, title?: string) => void;
}) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const {
        data: trainingRegressionHeatmapResponse,
        isLoading: trainingRegressionHeatmapLoading,
        error: trainingRegressionHeatmapError
    } = useGetTrainingJobsRegressionHeatmapById(
        trainingJobId ?? "",
        getQueryOptionsById(
            ReactQueryKeys.TrainingJobsRegressionHeatmapById,
            trainingJobId,
            Boolean(projectId && trainingJobId)
        )
    );

    const {
        data: evalRegressionHeatmapResponse,
        isLoading: evalRegressionHeatmapLoading,
        error: evalRegressionHeatmapError
    } = useGetEvalRegressionHeatmapById(
        queryId ?? "",
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.EvalRegressionHeatmapById,
            queryId,
            Boolean(projectId && !trainingJobId && queryId)
        )
    );

    const error = (
        trainingJobId
            ? trainingRegressionHeatmapError
            : evalRegressionHeatmapError
    ) as Error | null;

    useEffect(() => {
        if (error) {
            handleError(error, "Failed to get Regression Heatmap Stats!");
        }
    }, [error, handleError]);

    return {
        isLoading: trainingJobId
            ? trainingRegressionHeatmapLoading
            : evalRegressionHeatmapLoading,
        data: trainingJobId
            ? trainingRegressionHeatmapResponse?.data
            : evalRegressionHeatmapResponse?.data
    };
};

export const useRegressionHistogram = ({
    queryId,
    trainingJobId,
    handleError
}: {
    queryId: string;
    trainingJobId?: string;
    handleError: (err: Error, title?: string) => void;
}) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const {
        data: trainingRegressionHistogramResponse,
        isLoading: trainingRegressionHistogramLoading,
        error: trainingRegressionHistogramError
    } = useGetTrainingJobsRegressionHistogramById(
        trainingJobId ?? "",
        getQueryOptionsById(
            ReactQueryKeys.TrainingJobsRegressionHistogramById,
            trainingJobId,
            Boolean(projectId && trainingJobId)
        )
    );

    const {
        data: evalRegressionHistogramResponse,
        isLoading: evalRegressionHistogramLoading,
        error: evalRegressionHistogramError
    } = useGetEvalRegressionHistogramById(
        queryId ?? "",
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.EvalRegressionHistogramById,
            queryId,
            Boolean(projectId && !trainingJobId && queryId)
        )
    );

    const error = (
        trainingJobId
            ? trainingRegressionHistogramError
            : evalRegressionHistogramError
    ) as Error | null;

    useEffect(() => {
        if (error) {
            handleError(error, "Failed to get Regression Heatmap Stats!");
        }
    }, [error, handleError]);

    return {
        isLoading: trainingJobId
            ? trainingRegressionHistogramLoading
            : evalRegressionHistogramLoading,
        data: trainingJobId
            ? trainingRegressionHistogramResponse
            : evalRegressionHistogramResponse
    };
};
