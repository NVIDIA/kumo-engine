// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useGetTrainingJobsSdkByTrainingJobId } from "gen/v0/api/training-jobs/training-jobs";
import { useGetV1BatchPredictionJobsSdkByPredictionJobId } from "gen/v1/api/batch-prediction-job/batch-prediction-job";
import { useGetV1GraphsSdkByGraphId } from "gen/v1/api/graphs/graphs";
import { useGetV1TablesSdkByTableId } from "gen/v1/api/tables/tables";
import { useCacheStaleTime } from "stores/cacheStore";

import { useProjectId } from "hooks/project/projects";

import { SDKObjectType } from "types/constants";

export const useSDKCode = (id: string, type: SDKObjectType) => {
    const { medium } = useCacheStaleTime();
    const projectId = useProjectId();
    // Flags to decide which hook to run
    const isTable = type === SDKObjectType.Table;
    const isGraph = type === SDKObjectType.Graph;
    const isTraining = type === SDKObjectType.TrainingJob;
    const isPrediction = type === SDKObjectType.PredictionJob;

    const tableQuery = useGetV1TablesSdkByTableId(
        id,
        { project_id: projectId ?? "" },
        {
            query: {
                enabled: isTable && Boolean(projectId),
                staleTime: medium,
                retry: false
            }
        }
    );

    const graphQuery = useGetV1GraphsSdkByGraphId(
        id,
        { project_id: projectId ?? "" },
        {
            query: {
                enabled: isGraph && Boolean(projectId),
                staleTime: medium,
                retry: false
            }
        }
    );

    const trainingJob = useGetTrainingJobsSdkByTrainingJobId(
        id,
        { project_id: projectId ?? "" },
        {
            query: {
                enabled: isTraining && Boolean(projectId),
                staleTime: medium,
                retry: false
            }
        }
    );

    const predictionJob = useGetV1BatchPredictionJobsSdkByPredictionJobId(
        id,
        { project_id: projectId ?? "" },
        {
            query: {
                enabled: isPrediction && Boolean(projectId),
                staleTime: medium,
                retry: false
            }
        }
    );

    const query = isTable
        ? tableQuery
        : isGraph
          ? graphQuery
          : isTraining
            ? trainingJob
            : predictionJob;

    const { data, isLoading, error, isError } = query;

    return { data, isLoading, error, isError };
};
