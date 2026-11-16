// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useMemo } from "react";
import { useLocation } from "react-router";

import { prefetchGetProjectsConnectorsByProjectIdQuery } from "gen/v0/api/connectors/connectors";
import { prefetchGetJobsQuery } from "gen/v0/api/jobs/jobs";
import { prefetchGetV1GraphsQuery } from "gen/v1/api/graphs/graphs";
import { prefetchGetV1TablesQuery } from "gen/v1/api/tables/tables";

import { useJobsListPageSize } from "hooks/jobs/list/jobList";
import { useProjectId } from "hooks/project/projects";

import {
    getCachedJobId,
    getJobsListQueryParams,
    getTrainingJobsSelectListQueryParams
} from "utils/jobs";
import { ReactQueryKeys } from "utils/reactQueryKeys";

import { useQueryOptions } from "../misc/queryOptions";

import { usePrefetch } from "./prefetch";

export const useSidebarLinksPrefetch = () => {
    const { prefetchParallel } = usePrefetch();
    const prefetchConfigs = usePrefetchConfigs();

    const handleSidebarLinkHover = useCallback(
        (hoverLink: string) => {
            switch (hoverLink) {
                case "Models":
                    prefetchParallel(prefetchConfigs.ModelList);
                    break;
                case "Predictions":
                    prefetchParallel(prefetchConfigs.PredictionList);
                    break;
                case "Tables":
                    prefetchParallel(prefetchConfigs.TablesList);
                    break;
                case "Graphs":
                    prefetchParallel(prefetchConfigs.GraphList);
                    break;
                case "Connectors":
                    prefetchParallel(prefetchConfigs.ConnectorList);
                    break;
                case "NewPrediction":
                    prefetchParallel(prefetchConfigs.NewPredictionList);
                    break;
                default:
                    break;
            }
        },
        [
            prefetchConfigs.ConnectorList,
            prefetchConfigs.GraphList,
            prefetchConfigs.ModelList,
            prefetchConfigs.NewPredictionList,
            prefetchConfigs.PredictionList,
            prefetchConfigs.TablesList,
            prefetchParallel
        ]
    );

    return handleSidebarLinkHover;
};

const usePrefetchConfigs = () => {
    const location = useLocation();
    const { getQueryOptions } = useQueryOptions();
    const projectId = useProjectId();

    const jobsListPageSize = useJobsListPageSize();

    const jobsListQueryParams = useMemo(() => {
        if (
            location.pathname.includes("models") ||
            location.pathname.includes("predictions")
        ) {
            return Object.fromEntries(
                new URLSearchParams(location?.search).entries()
            );
        }
        return {};
    }, [location]);

    const jobsListQuery = useMemo(() => {
        return getJobsListQueryParams(
            jobsListQueryParams,
            jobsListPageSize,
            projectId ?? ""
        );
    }, [jobsListQueryParams, jobsListPageSize, projectId]);

    const trainingJobsListQuery = useMemo(() => {
        return getTrainingJobsSelectListQueryParams(projectId ?? "");
    }, [projectId]);

    const prefetchConfigs = useMemo(() => {
        const projectParams = { project_id: projectId ?? "" };
        return {
            ConnectorList: [
                {
                    prefetchFn: prefetchGetProjectsConnectorsByProjectIdQuery,
                    params: [projectId ?? ""],
                    options: getQueryOptions({
                        queryKey: [
                            ReactQueryKeys.ConnectorList,
                            projectId ?? ""
                        ]
                    })
                }
            ],
            TablesList: [
                {
                    prefetchFn: prefetchGetV1TablesQuery,
                    params: [projectParams],
                    options: getQueryOptions({
                        queryKey: [ReactQueryKeys.TablesList]
                    })
                }
            ],
            GraphList: [
                {
                    prefetchFn: prefetchGetV1GraphsQuery,
                    params: [projectParams],
                    options: getQueryOptions({
                        queryKey: [ReactQueryKeys.GraphList]
                    })
                }
            ],
            ModelList: [
                {
                    prefetchFn: prefetchGetJobsQuery,
                    params: [
                        {
                            ...jobsListQuery,
                            job_type: ["TRAINING_JOB"],
                            ...projectParams
                        }
                    ],
                    options: getQueryOptions({
                        queryKey: [
                            ReactQueryKeys.JobList,
                            getCachedJobId({
                                ...jobsListQuery,
                                job_type: ["TRAINING_JOB"],
                                ...projectParams
                            })
                        ]
                    })
                }
            ],
            PredictionList: [
                {
                    prefetchFn: prefetchGetJobsQuery,
                    params: [
                        {
                            ...jobsListQuery,
                            job_type: ["BATCH_PREDICTION_JOB"],
                            ...projectParams
                        }
                    ],
                    options: getQueryOptions({
                        queryKey: [
                            ReactQueryKeys.JobList,
                            getCachedJobId({
                                ...jobsListQuery,
                                job_type: ["BATCH_PREDICTION_JOB"],
                                ...projectParams
                            })
                        ]
                    })
                }
            ],
            NewPredictionList: [
                {
                    prefetchFn: prefetchGetJobsQuery,
                    params: [
                        {
                            ...trainingJobsListQuery,
                            ...projectParams
                        }
                    ],
                    options: getQueryOptions({
                        queryKey: [
                            ReactQueryKeys.JobList,
                            getCachedJobId({
                                ...trainingJobsListQuery,
                                ...projectParams
                            })
                        ]
                    })
                }
            ]
        };
    }, [getQueryOptions, jobsListQuery, projectId, trainingJobsListQuery]);

    return prefetchConfigs;
};
