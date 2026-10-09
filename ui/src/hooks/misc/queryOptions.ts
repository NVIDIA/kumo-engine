// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable @typescript-eslint/no-explicit-any */
import { useRef } from "react";

import {
    Query,
    QueryFunction,
    queryOptions,
    UseQueryOptions
} from "@tanstack/react-query";
import { AxiosResponse } from "axios";
import { getTensorboardManyScalars } from "gen/v0/api/tensorboard/tensorboard";
import {
    EdgeHealthStats,
    GeneratePredictionTableResponse,
    GenerateTrainTableResponse,
    GenericJobSource,
    GraphSnapshotResponse,
    JobScalarTags,
    ListJobsResponse,
    SourceColumn,
    TrainTableGenerationProgress
} from "gen/v0/types";
import { BatchPredictionJobResponse } from "gen/v1/types";
import { useCacheStaleTime } from "stores/cacheStore";

import { BatchPredictionJobAnchorTime } from "types/batchPredictions";
import { JobSource } from "types/jobs";

import { jobStatus, kumoResource, splitKey } from "utils/Constants";
import { edgeStatsFromEdgeStatsResponse } from "utils/Converters/edgeStatsConverter";
import { getKeys } from "utils/Helpers/objectHelpers";
import { filterTags } from "utils/Helpers/tagHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { getCacheTime } from "utils/jobs";
import { ReactQueryKeys } from "utils/reactQueryKeys";
import { validateResourceName } from "utils/validator";

// Keys whose only (or additional) override is a `long` staleTime. `long` is a
// runtime cache value, so it is applied at call time rather than baked into the
// static table below.
const STALE_LONG_KEYS = new Set<string>([
    ReactQueryKeys.SnapshotsColStatsById,
    ReactQueryKeys.PredictiveQueriesResourceById,
    ReactQueryKeys.BatchPredictionJobsPredictionStatsById,
    ReactQueryKeys.BatchPredictionJobsDownloadBpById,
    ReactQueryKeys.PredtableJobsAnchorTimeById,
    ReactQueryKeys.SnapshotsStatsById,
    ReactQueryKeys.TrainingJobsBinaryClassificationById,
    ReactQueryKeys.EvalBinaryClassificationById,
    ReactQueryKeys.TrainingJobsAvailableXaiEntitiesById,
    ReactQueryKeys.EvalAvailableXaiEntitiesById,
    ReactQueryKeys.TrainingJobsRegressionHeatmapById,
    ReactQueryKeys.EvalRegressionHeatmapById,
    ReactQueryKeys.TrainingJobsRegressionHistogramById,
    ReactQueryKeys.EvalRegressionHistogramById,
    ReactQueryKeys.TrainingJobsXaiById,
    ReactQueryKeys.EvalXaiById,
    ReactQueryKeys.BatchPredictionJobsXaiById,
    ReactQueryKeys.TrainingJobsEntityOutputByEntityId,
    ReactQueryKeys.EvalEntityOutputByEntityId,
    ReactQueryKeys.TrainingJobsSubgraphByEntityId,
    ReactQueryKeys.TrainingJobsEntityHistoricalItemsByEntityId,
    ReactQueryKeys.EvalEntityHistoricalItemsByEntityId,
    ReactQueryKeys.TrainingJobsWarningsById,
    ReactQueryKeys.TraintableJobsErrorsById,
    ReactQueryKeys.TrainingJobsNodePredRfmByEntityId,
    ReactQueryKeys.TrainingJobsCategoryNodePredRfmByCategory,
    ReactQueryKeys.TrainingJobsTopkFeatureByTopk,
    ReactQueryKeys.TrainingJobsCategoryAggrFeatureByOp
]);

// Static (long-independent) portion of each simple override. Built once at
// module scope; keys in STALE_LONG_KEYS additionally get `staleTime: long`.
const STATIC_QUERY_OVERRIDES: Record<string, Partial<UseQueryOptions>> = {
    [ReactQueryKeys.TablesById]: { retry: false },
    [ReactQueryKeys.ColStatsByTableId]: {
        refetchOnWindowFocus: false,
        retry: false
    },
    [ReactQueryKeys.SnapshotsColStatsById]: { retry: false },
    [ReactQueryKeys.PredictiveQueriesResourceById]: { retry: false },
    [ReactQueryKeys.BatchPredictionJobsPredictionStatsById]: { retry: false },
    [ReactQueryKeys.BatchPredictionJobsDownloadBpById]: {
        refetchInterval: false
    },
    [ReactQueryKeys.PredtableJobsAnchorTimeById]: { refetchInterval: false },
    [ReactQueryKeys.TrainingJobsDownloadHoldoutById]: {
        staleTime: 0,
        gcTime: 0,
        refetchInterval: false,
        retry: false
    },
    [ReactQueryKeys.PredictiveQueriesDownloadHoldoutById]: {
        staleTime: 0,
        gcTime: 0,
        refetchInterval: false,
        retry: false
    }
};

// Merge the static + long overrides for `key` onto `query`. Returns false when
// `key` has no simple override, so the caller falls through to the switch.
function applySimpleOverride<T extends { staleTime?: unknown }>(
    query: T,
    key: string,
    long: number
): boolean {
    const staticOverride = STATIC_QUERY_OVERRIDES[key];
    const needsStaleLong = STALE_LONG_KEYS.has(key);
    if (!staticOverride && !needsStaleLong) {
        return false;
    }
    if (staticOverride) {
        Object.assign(query, staticOverride);
    }
    if (needsStaleLong) {
        query.staleTime = long;
    }
    return true;
}

export const useQueryOptions = () => {
    const { medium, short, long } = useCacheStaleTime();
    const pollQueuingCount = useRef(10);
    const refetchAfterDone = useRef(false);

    const getQueryOptions = ({
        queryKey,
        staleTime,
        enabled = true,
        queryFn,
        refetchOnWindowFocus,
        retry,
        refetchInterval
    }: Partial<UseQueryOptions>): any => {
        const qKey = queryKey as string[];
        const query = queryOptions({
            queryKey: qKey,
            staleTime: staleTime === undefined ? medium : staleTime,
            enabled: enabled
        });

        if (qKey?.length === 0) {
            return {};
        }

        if (!applySimpleOverride(query, qKey[0], long)) {
            switch (qKey[0]) {
                case ReactQueryKeys.BatchPredictionJobsPredictionAnchorTimeById:
                    query.staleTime = long;
                    query.select = (data) =>
                        (data as AxiosResponse<BatchPredictionJobAnchorTime>)
                            .data;
                    break;
                case ReactQueryKeys.JobList:
                    query.staleTime = short;
                    query.select = (data) => {
                        const response =
                            data as AxiosResponse<ListJobsResponse>;
                        if (response?.data?.jobs) {
                            response.data.jobs.forEach((j) => {
                                if (j.source === null) {
                                    //Overwrite null to avoid `<Missing>` from Jobs List Table
                                    j.source =
                                        JobSource.NONE as GenericJobSource;
                                }
                            });
                        }
                        return response;
                    };
                    break;
                case ReactQueryKeys.ConnectorsById:
                    query.retry = false;
                    break;
                case ReactQueryKeys.SnapshotsGraph:
                    query.staleTime = long;
                    query.refetchInterval = (query) => {
                        if (!query.state.data) return;
                        const resp = query?.state
                            ?.data as AxiosResponse<GraphSnapshotResponse>;
                        const status =
                            resp.data.snapshot_resource.stages?.INGEST?.status;

                        if (
                            [
                                jobStatus.Running,
                                jobStatus.Queued,
                                jobStatus.Untrained
                            ].includes(status as jobStatus)
                        ) {
                            return short;
                        }
                        return false;
                    };
                    break;
                case ReactQueryKeys.TrainingJobsEvalMetricsById:
                case ReactQueryKeys.PredictiveQueriesEvalMetricsById:
                    query.staleTime = long;
                    query.refetchInterval = (query) => {
                        if (!query.state.data) return 4000;
                        const resp = query?.state?.data as AxiosResponse<any>;
                        const hasMetrics =
                            resp?.data?.metrics &&
                            getKeys(resp.data.metrics).length > 0;
                        // Poll every 4 seconds if metrics aren't available yet
                        return hasMetrics ? false : 4000;
                    };
                    break;
            }
        }

        if (queryFn) {
            query.queryFn = queryFn as
                | QueryFunction<unknown, readonly unknown[]>
                | undefined;
        }

        if (refetchOnWindowFocus) {
            query.refetchOnWindowFocus = refetchOnWindowFocus;
        }

        if (retry !== undefined && retry !== null) {
            query.retry = retry;
        }

        if (refetchInterval) {
            query.refetchInterval = refetchInterval;
        }

        return { query };
    };

    const getQueryOptionsById = (
        queryKey: string,
        id: string | null | undefined,
        enabled?: boolean,
        refetchInterval?:
            | number
            | false
            | ((
                  query: Query<unknown, Error, unknown, readonly unknown[]>
              ) => number | false | undefined)
            | undefined
    ): any => {
        return getQueryOptions({
            queryKey: [queryKey, id],
            enabled: enabled ?? !!id,
            refetchInterval: refetchInterval ?? false
        });
    };

    const getQueryOptionsByEntityId = (
        queryKey: string,
        id: string | null | undefined,
        entityId: string,
        enabled?: boolean
    ): any => {
        return getQueryOptions({
            queryKey: [queryKey, id ?? "", entityId],
            enabled: enabled ?? !!id
        });
    };

    const getQueryOptionsByIdAndJobStatus = (
        queryKey: string,
        id: string,
        enabled?: boolean,
        status?: string | jobStatus
    ): any => {
        const staleTime = getCacheTime(status as jobStatus, short, long);

        return getQueryOptions({
            queryKey: [queryKey, id, status ?? ""],
            staleTime: staleTime,
            enabled: enabled ?? !!id
        });
    };

    const getModelPlansQueryOptionsById = (
        queryKey: string,
        id: string,
        status: jobStatus,
        bestExperimentIndex: number,
        enabled?: boolean
    ): any => {
        const staleTime = getCacheTime(status, short, long);

        return getQueryOptions({
            queryKey: [queryKey, id, status, bestExperimentIndex.toString()],
            staleTime: staleTime,
            enabled: enabled ?? !!id
        });
    };

    const queryOption = {
        nameValidation: (
            input: string,
            resource: kumoResource,
            entityId: string,
            projectId?: string | null
        ) =>
            queryOptions({
                queryKey: [
                    ReactQueryKeys.NameValidation,
                    input,
                    resource,
                    entityId,
                    projectId ?? ""
                ],
                queryFn: ({ signal }) =>
                    validateResourceName(
                        resource,
                        input.split(splitKey)[0],
                        entityId,
                        signal,
                        projectId
                    ),
                enabled: !!input, // only run if there's input
                staleTime: short
            }),
        connectorTableCols: (
            tableName: string,
            connectorId: string,
            enabled: boolean
        ): any => {
            const query = queryOptions({
                queryKey: [
                    ReactQueryKeys.ConnectorTableCols,
                    tableName,
                    connectorId
                ],
                staleTime: medium,
                retry: false,
                enabled: enabled,
                select: (resp: AxiosResponse<SourceColumn>) => {
                    if (enabled) {
                        if (resp?.data) {
                            return resp;
                        }
                    }
                    return undefined;
                }
            });
            return { query };
        },
        graphEdgeStats: (
            id: string,
            graphSnapshot: GraphSnapshotResponse | null
        ): any => {
            const query = queryOptions({
                queryKey: [ReactQueryKeys.GraphEdgeStats, id],
                enabled: !!graphSnapshot && !!id,
                staleTime: long,
                select: (response: AxiosResponse<EdgeHealthStats>) => {
                    if (!response?.data || !graphSnapshot) return null;

                    return edgeStatsFromEdgeStatsResponse(response.data).map(
                        (e) => ({
                            ...e,
                            col_a_table_id:
                                graphSnapshot.snapshot_resource?.table_ids?.[
                                    e.col_a_table_name
                                ],
                            col_b_table_id:
                                graphSnapshot.snapshot_resource?.table_ids?.[
                                    e.col_b_table_name
                                ]
                        })
                    );
                }
            });

            return { query };
        },
        genPredTableJobsById: (id: string): any => {
            const query = queryOptions({
                queryKey: [ReactQueryKeys.GenPredTableJobsById, id],
                enabled: !!id,
                staleTime: long,
                refetchInterval: (query) => {
                    const q = query?.state
                        ?.data as AxiosResponse<GeneratePredictionTableResponse>;
                    const status = q?.data?.resource?.job_status_report?.status;
                    if (!status) return 15000;
                    if (
                        status === jobStatus.Done ||
                        status === jobStatus.Failed ||
                        status === jobStatus.Cancelled
                    ) {
                        return false;
                    }
                    if (
                        status === jobStatus.Queued &&
                        pollQueuingCount.current > 0
                    ) {
                        pollQueuingCount.current -= 1;
                        return 5000;
                    }
                    return 15000;
                }
            });

            return { query };
        },
        batchPredictionJobsById: (
            id: string,
            refetchTrigger: number | undefined
        ): any => {
            const query = queryOptions({
                queryKey: [ReactQueryKeys.BatchPredictionJobsById, id],
                enabled: !!id,
                staleTime: long,
                refetchInterval: (query) => {
                    const q = query?.state
                        ?.data as AxiosResponse<BatchPredictionJobResponse>;
                    const status = q?.data?.execution?.status;
                    if (
                        status === jobStatus.Queued ||
                        status === jobStatus.Running
                    ) {
                        return 10000;
                    }
                    // refetch once if trigger changed
                    if (refetchTrigger && refetchTrigger > 0) {
                        return 0;
                    }
                    return false;
                }
            });

            return { query };
        },
        traintableJobsGenerationStatsById: (
            id: string,
            status: jobStatus
        ): any => {
            const staleTime = getCacheTime(status, short, long);
            const query = queryOptions({
                queryKey: [
                    ReactQueryKeys.TraintableJobsGenerationStatsById,
                    id
                ],
                enabled: !!id,
                staleTime: staleTime,
                refetchInterval: (query) => {
                    if (!id || status !== jobStatus.Running) {
                        return false;
                    }
                    if (!query?.state?.data) {
                        return 10000;
                    }
                    const { num_expected_iterations, num_finished_iterations } =
                        query?.state?.data as TrainTableGenerationProgress;
                    if (
                        num_expected_iterations === 0 ||
                        num_expected_iterations === undefined ||
                        num_expected_iterations !== num_finished_iterations
                    ) {
                        return 10000;
                    }

                    return false;
                }
            });

            return { query };
        },
        traintableJobsById: (id: string): any => {
            const query = queryOptions({
                queryKey: [ReactQueryKeys.TraintableJobsById, id],
                enabled: !!id,
                staleTime: (query) => {
                    const resp = query?.state
                        ?.data as AxiosResponse<GenerateTrainTableResponse>;
                    const status =
                        resp?.data?.resource?.job_status_report?.status ?? null;
                    return getCacheTime(status as jobStatus, short, long);
                },
                refetchInterval: (query) => {
                    const resp = query?.state
                        ?.data as AxiosResponse<GenerateTrainTableResponse>;

                    if (!resp?.data) return 15000; // default 15s

                    const status =
                        resp.data.resource?.job_status_report?.status;
                    if (
                        status === jobStatus.Done ||
                        status === jobStatus.Failed ||
                        status === jobStatus.Cancelled
                    ) {
                        return false;
                    }
                    if (
                        status === jobStatus.Queued &&
                        pollQueuingCount.current > 0
                    ) {
                        pollQueuingCount.current -= 1;
                        return 5000;
                    }
                    return 15000;
                }
            });

            return { query };
        },
        trainingJobsWorkflowById: (
            id: string,
            status: jobStatus,
            refetchInterval:
                | number
                | false
                | ((
                      query: Query<unknown, Error, unknown, string[]>
                  ) => number | false | undefined)
                | undefined
        ): any => {
            const staleTime = getCacheTime(status, short, long);
            const query = queryOptions({
                queryKey: [ReactQueryKeys.TrainingJobsWorkflowById, id],
                staleTime: staleTime,
                refetchInterval: refetchInterval,
                refetchOnMount: true
            });

            return { query };
        },
        tensorboardTagsByJobId: (id: string, status: string): any => {
            const query = queryOptions({
                queryKey: [ReactQueryKeys.TensorboardTagsByJobId, id],
                staleTime: (query) => {
                    const resp = query?.state
                        ?.data as AxiosResponse<JobScalarTags>;
                    const trails = getKeys(resp?.data?.trial_tags ?? {});
                    const uniqueTags = filterTags(
                        trails,
                        resp?.data?.trial_tags ?? {}
                    );
                    if (status === jobStatus.Done && uniqueTags?.length > 0) {
                        return long;
                    }
                    return 0;
                },
                enabled:
                    status !== jobStatus.Failed &&
                    status !== jobStatus.Cancelled &&
                    isValid(id),
                refetchInterval: (query) => {
                    const resp = query.state
                        ?.data as AxiosResponse<JobScalarTags>;
                    const trialTags = resp?.data?.trial_tags ?? {};
                    const trails = getKeys(trialTags);
                    const tags = filterTags(trails, trialTags);

                    if (
                        status === jobStatus.Failed ||
                        status === jobStatus.Cancelled ||
                        (status === jobStatus.Done && tags?.length > 0)
                    ) {
                        return false;
                    }
                    return 30000;
                }
            });

            return { query };
        },
        tensorboardManyScalars: (
            id: string,
            trails: string[],
            tags: any[],
            tagsId: string,
            status: string
        ) =>
            trails.map((trail) =>
                queryOptions({
                    queryKey: [
                        ReactQueryKeys.TensorboardManyScalars,
                        id,
                        trail,
                        ...tags,
                        tagsId,
                        status === jobStatus.Done && tags?.length > 0
                            ? tagsId.substring(0, tagsId.lastIndexOf("-")) // to avoid api call for dependacy changes
                            : tagsId
                    ],
                    queryFn: () =>
                        getTensorboardManyScalars({
                            job_id: id,
                            tags: tags,
                            trial: trail
                        }),
                    enabled: tags?.length > 0,
                    staleTime: 0
                })
            ),
        trainingJobsWarningsById: (
            id: string | null,
            status?: string | null
        ): any => {
            const query = queryOptions({
                enabled: !!id,
                queryKey: [ReactQueryKeys.TrainingJobsWarningsById, id],
                staleTime: long,
                refetchInterval: (): false | number => {
                    if (status === jobStatus.Running) {
                        return 10000; // poll every 10s if running
                    }

                    if (!status && !refetchAfterDone.current) {
                        refetchAfterDone.current = true;
                        return 0;
                    }

                    return false; // stop polling otherwise
                }
            });

            return { query };
        }
    };

    return {
        getQueryOptions,
        getQueryOptionsById,
        getQueryOptionsByEntityId,
        getQueryOptionsByIdAndJobStatus,
        getModelPlansQueryOptionsById,
        queryOption
    };
};
