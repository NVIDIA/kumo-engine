// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoaderFunctionArgs } from "react-router";

import { prefetchGetProjectsConnectorsByProjectIdQuery } from "gen/v0/api/connectors/connectors";
import { prefetchGetJobsQuery } from "gen/v0/api/jobs/jobs";
import { JobType } from "gen/v0/types";
import { prefetchGetV1GraphsQuery } from "gen/v1/api/graphs/graphs";
import { prefetchGetV1TablesQuery } from "gen/v1/api/tables/tables";
import { getDefaultStaleTimes } from "stores/cacheStore";

import {
    getJobsListPageSize,
    getJobsListQueryParams,
    getTrainingJobsSelectListQueryParams
} from "utils/jobs";
import { queryClient } from "utils/queryClient";
import { ReactQueryKeys } from "utils/reactQueryKeys";
import { projectQueryParams } from "utils/routeProject";

const getLoaderQueryOptions = (queryKey: string[]) => {
    const { medium } = getDefaultStaleTimes();
    return {
        query: {
            queryKey,
            staleTime: medium
        }
    };
};

export const connectorsLoader = ({ params }: LoaderFunctionArgs) => {
    const projectId = params.projectId;
    if (projectId) {
        prefetchGetProjectsConnectorsByProjectIdQuery(queryClient, projectId, {
            ...getLoaderQueryOptions([ReactQueryKeys.ConnectorList, projectId])
        });
    }
    return null;
};

export const tablesLoader = ({ params }: LoaderFunctionArgs) => {
    const projectId = params.projectId;
    if (projectId) {
        prefetchGetV1TablesQuery(
            queryClient,
            { project_id: projectId },
            {
                ...getLoaderQueryOptions([ReactQueryKeys.TablesList, projectId])
            }
        );
    }
    return null;
};

export const graphsLoader = ({ params }: LoaderFunctionArgs) => {
    const projectId = params.projectId;
    if (projectId) {
        prefetchGetV1GraphsQuery(
            queryClient,
            { project_id: projectId },
            {
                ...getLoaderQueryOptions([ReactQueryKeys.GraphList, projectId])
            }
        );
    }
    return null;
};

export const modelsLoader = ({ request, params }: LoaderFunctionArgs) => {
    const url = new URL(request.url);
    const searchParams = Object.fromEntries(url.searchParams.entries());
    const projectId = params.projectId;
    const jobsListQuery = getJobsListQueryParams(
        searchParams,
        getJobsListPageSize(),
        projectId ?? ""
    );
    const query = {
        ...jobsListQuery,
        job_type: [JobType.TRAINING_JOB],
        ...(projectQueryParams(projectId) ?? {})
    };

    prefetchGetJobsQuery(queryClient, query, {
        ...getLoaderQueryOptions([
            ReactQueryKeys.JobList,
            JSON.stringify(query)
        ])
    });
    return null;
};

export const predictionsLoader = ({ request, params }: LoaderFunctionArgs) => {
    const url = new URL(request.url);
    const searchParams = Object.fromEntries(url.searchParams.entries());
    const projectId = params.projectId;
    const jobsListQuery = getJobsListQueryParams(
        searchParams,
        getJobsListPageSize(),
        projectId ?? ""
    );
    const query = {
        ...jobsListQuery,
        job_type: [JobType.BATCH_PREDICTION_JOB],
        ...(projectQueryParams(projectId) ?? {})
    };

    prefetchGetJobsQuery(queryClient, query, {
        ...getLoaderQueryOptions([
            ReactQueryKeys.JobList,
            JSON.stringify(query)
        ])
    });
    return null;
};

export const newPredictionLoader = ({ params }: LoaderFunctionArgs) => {
    const trainingJobsQuery = getTrainingJobsSelectListQueryParams(
        params.projectId ?? ""
    );

    prefetchGetJobsQuery(queryClient, trainingJobsQuery, {
        ...getLoaderQueryOptions([
            ReactQueryKeys.JobList,
            JSON.stringify(trainingJobsQuery)
        ])
    });
    return null;
};
