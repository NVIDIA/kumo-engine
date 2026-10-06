// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useGetTrainingJobsEncodersById } from "gen/v0/api/training-jobs/training-jobs";
import { useGetV1PredictiveQueriesEncodersById } from "gen/v1/api/predictive-queries/predictive-queries";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useGetQueryEncoders = (
    queryId?: string,
    trainingJobId?: string,
    trainingStatus?: string
): { [k: string]: string } => {
    const { getQueryOptionsByIdAndJobStatus } = useQueryOptions();
    const projectId = useProjectId() ?? "";

    const { data: trainingEncodersResponse } = useGetTrainingJobsEncodersById(
        trainingJobId ?? "",
        getQueryOptionsByIdAndJobStatus(
            ReactQueryKeys.TrainingJobsEncodersById,
            trainingJobId as string,
            Boolean(projectId && trainingJobId),
            trainingStatus as string
        )
    );

    const { data: predictiveEncodersResponse } =
        useGetV1PredictiveQueriesEncodersById(
            queryId ?? "",
            { project_id: projectId },
            getQueryOptionsByIdAndJobStatus(
                ReactQueryKeys.PredictiveQueriesEncodersById,
                queryId as string,
                Boolean(projectId && !trainingJobId && queryId)
            )
        );

    const encoders = trainingJobId
        ? (trainingEncodersResponse?.data ?? {})
        : (predictiveEncodersResponse?.data?.encoders ?? {});

    return encoders;
};
