// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { getGetIdentitiesQueryKey } from "gen/v0/api/identities/identities";
import { JobType } from "gen/v0/types";

import { ReactQueryKeys } from "./reactQueryKeys";
import { getSelectedGroup } from "./storage";

export const invalidateJobsList = (queryClient: QueryClient) => {
    queryClient.invalidateQueries({
        queryKey: [ReactQueryKeys.JobList]
    });
    queryClient.removeQueries({
        queryKey: [ReactQueryKeys.JobList]
    });
};

export const invalidateActiveJobs = (queryClient: QueryClient) => {
    queryClient.invalidateQueries({
        queryKey: [ReactQueryKeys.ActiveJobs]
    });
};

export const invalidateTrainingTags = (queryClient: QueryClient) => {
    queryClient.invalidateQueries({
        queryKey: [ReactQueryKeys.TrainingTags]
    });
    queryClient.removeQueries({
        queryKey: [ReactQueryKeys.TrainingTags]
    });
};

export const invalidatePredictionTags = (queryClient: QueryClient) => {
    queryClient.invalidateQueries({
        queryKey: [ReactQueryKeys.PredictionTags]
    });
    queryClient.removeQueries({
        queryKey: [ReactQueryKeys.PredictionTags]
    });
};

export const invalidateSnapshotsGraph = (
    queryClient: QueryClient,
    graph_snapshot_id?: string
) => {
    const queryKey = graph_snapshot_id
        ? [ReactQueryKeys.SnapshotsGraph, graph_snapshot_id]
        : [ReactQueryKeys.SnapshotsGraph];

    queryClient.removeQueries({
        queryKey: queryKey
    });
};

export const invalidateJobQueries = (
    queryClient: QueryClient,
    jobType: JobType,
    jobId: string
) => {
    queryClient.invalidateQueries({
        queryKey: [ReactQueryKeys.JobList]
    });

    switch (jobType) {
        case JobType.TRAINING_JOB:
            queryClient.invalidateQueries({
                queryKey: [ReactQueryKeys.TrainingTags]
            });
            queryClient.invalidateQueries({
                queryKey: [ReactQueryKeys.TrainingJobsWorkflowById, jobId]
            });
            break;

        case JobType.BATCH_PREDICTION_JOB:
            queryClient.invalidateQueries({
                queryKey: [ReactQueryKeys.PredictionTags]
            });
            queryClient.invalidateQueries({
                queryKey: [ReactQueryKeys.BatchPredictionJobsById, jobId]
            });
            break;

        case JobType.GENERATE_TRAIN_TABLE_JOB:
            queryClient.invalidateQueries({
                queryKey: [ReactQueryKeys.TraintableJobsById, jobId]
            });
            queryClient.invalidateQueries({
                queryKey: [ReactQueryKeys.TrainingTags]
            });
            break;

        case JobType.GENERATE_PREDICTION_TABLE_JOB:
            queryClient.invalidateQueries({
                queryKey: [ReactQueryKeys.GenPredTableJobsById, jobId]
            });
            queryClient.invalidateQueries({
                queryKey: [ReactQueryKeys.PredictionTags]
            });
            break;

        default:
            break;
    }
};

export const invalidateProjectTableList = (
    queryClient: QueryClient,
    projectId: string,
    connectorId: string
) => {
    const queryKey = [ReactQueryKeys.TablesList, projectId, connectorId];
    queryClient.invalidateQueries({
        queryKey: queryKey,
        refetchType: "none" // prevents automatic refetch
    });
    queryClient.removeQueries({
        queryKey: queryKey
    });
};

export const invalidateProjectSourceTables = (
    queryClient: QueryClient,
    projectId: string,
    connectorId: string
) => {
    const queryKey = [ReactQueryKeys.SourceTable, projectId, connectorId];
    queryClient.invalidateQueries({
        queryKey: queryKey,
        refetchType: "none" // prevents automatic refetch
    });
    queryClient.removeQueries({
        queryKey: queryKey
    });
};

export const invalidateConnectorList = (
    queryClient: QueryClient,
    projectId?: string
) => {
    const queryKey = projectId
        ? [ReactQueryKeys.ConnectorList, projectId]
        : [ReactQueryKeys.ConnectorList];
    // Only invalidate (no `removeQueries`) so the active connector list query
    // refetches in place instead of dropping its cache and flashing empty.
    queryClient.invalidateQueries({
        queryKey: queryKey
    });
};

export const invalidateTablesList = (queryClient: QueryClient) => {
    queryClient.invalidateQueries({
        queryKey: [ReactQueryKeys.TablesList]
    });
    queryClient.removeQueries({
        queryKey: [ReactQueryKeys.TablesList]
    });
};

export const invalidateGraphsList = (
    queryClient: QueryClient,
    graphId?: string
) => {
    queryClient.invalidateQueries({
        queryKey: [ReactQueryKeys.GraphList]
    });
    queryClient.removeQueries({
        queryKey: [ReactQueryKeys.GraphList]
    });
    if (graphId) {
        queryClient.invalidateQueries({
            queryKey: [ReactQueryKeys.GraphInfo, graphId]
        });
        queryClient.removeQueries({
            queryKey: [ReactQueryKeys.GraphInfo, graphId]
        });
    }
};

export const invalidateGroupsList = (queryClient: QueryClient) => {
    queryClient.invalidateQueries({
        queryKey: [ReactQueryKeys.Groups]
    });
    queryClient.removeQueries({
        queryKey: [ReactQueryKeys.Groups]
    });
};

export const invalidateUserGroupsList = (queryClient: QueryClient) => {
    queryClient.invalidateQueries({
        queryKey: [ReactQueryKeys.UserGroups]
    });
    queryClient.removeQueries({
        queryKey: [ReactQueryKeys.UserGroups]
    });
};

// Matches the orval-generated keys for GET /v0/projects/{id}/identities/,
// which the connector form's project-scoped identity lists subscribe to.
const projectIdentitiesKeyPattern = /^\/v0\/projects\/[^/]+\/identities\//;

export const invalidateIdentityLists = (queryClient: QueryClient) => {
    // The admin identity list subscribes to the orval-generated query key, so
    // reuse that key helper to stay in sync if the endpoint path ever changes.
    // Project-scoped identity lists (one per project) are matched by path
    // predicate instead. Only invalidate (no `removeQueries`) so active lists
    // refetch in place instead of dropping their cache and flashing empty.
    queryClient.invalidateQueries({
        queryKey: getGetIdentitiesQueryKey()
    });
    queryClient.invalidateQueries({
        predicate: ({ queryKey }) =>
            typeof queryKey[0] === "string" &&
            projectIdentitiesKeyPattern.test(queryKey[0])
    });
};

export const invalidateProjectInfo = (
    queryClient: QueryClient,
    projectId: string
) => {
    queryClient.invalidateQueries({
        queryKey: [ReactQueryKeys.ProjectList, projectId]
    });
    queryClient.removeQueries({
        queryKey: [ReactQueryKeys.ProjectList, projectId]
    });

    // Invalidate Groups project list as well
    const groupId = getSelectedGroup();
    if (groupId) {
        queryClient.invalidateQueries({
            queryKey: [ReactQueryKeys.ProjectList, groupId]
        });
        queryClient.removeQueries({
            queryKey: [ReactQueryKeys.ProjectList, groupId]
        });
    }
};
