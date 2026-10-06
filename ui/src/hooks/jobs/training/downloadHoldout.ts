// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";
import { useGetTrainingJobsDownloadHoldoutById } from "gen/v0/api/training-jobs/training-jobs";
import { OutputPresignedUrlResponse } from "gen/v0/types";
import { useGetV1PredictiveQueriesDownloadHoldoutById } from "gen/v1/api/predictive-queries/predictive-queries";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";

import { parseErrorMessage } from "utils/errorHandler";
import { ReactQueryKeys } from "utils/reactQueryKeys";

interface DownloadHoldoutProps {
    id: string;
    isTrainingJob?: boolean;
    enabled?: boolean;
}

/**
 * Custom hook to fetch download holdout presigned URL for either training jobs or predictive queries
 */
export const useDownloadHoldout = ({
    id,
    isTrainingJob = false,
    enabled = false
}: DownloadHoldoutProps): {
    presignedUrl: string;
    isError: boolean;
    error: string;
    refetch: () => void;
} => {
    const [presignedUrl, setPresignedUrl] = useState<string>("");

    const queryClient = useQueryClient();
    const { getQueryOptions } = useQueryOptions();
    const projectId = useProjectId() ?? "";

    // Query for training job download holdout
    const {
        data: trainingJobResponse,
        isError: isTrainingJobError,
        error: trainingJobError,
        refetch: trainingJobRefetch
    } = useGetTrainingJobsDownloadHoldoutById<{
        data: OutputPresignedUrlResponse;
    }>(
        id,
        { project_id: projectId },
        getQueryOptions({
            queryKey: [ReactQueryKeys.PredictiveQueriesDownloadHoldoutById],
            enabled: isTrainingJob && !!id && enabled && Boolean(projectId)
        }),
        queryClient
    );

    // Query for predictive query download holdout
    const {
        data: pqResponse,
        isError: isPqError,
        error: pqError,
        refetch: pqRefetch
    } = useGetV1PredictiveQueriesDownloadHoldoutById<{
        data: OutputPresignedUrlResponse;
    }>(
        id,
        { project_id: projectId },
        getQueryOptions({
            queryKey: [ReactQueryKeys.PredictiveQueriesDownloadHoldoutById, id],
            enabled: !isTrainingJob && !!id && enabled && Boolean(projectId)
        }),
        queryClient
    );

    const isError = isTrainingJob ? isTrainingJobError : isPqError;
    const error = isTrainingJob ? trainingJobError : pqError;
    const response = isTrainingJob ? trainingJobResponse : pqResponse;
    const refetch = isTrainingJob ? trainingJobRefetch : pqRefetch;

    useEffect(() => {
        if (response?.data) {
            setPresignedUrl(response.data.presigned_url);
        }
    }, [isError, error, response, enabled]);

    return {
        presignedUrl,
        isError: enabled && isError,
        error: isError && error ? parseErrorMessage(error as CustomError) : "",
        refetch
    };
};
