// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";
import {
    useGetSnapshotsColStatsById,
    useGetSnapshotsEdgeStatsById,
    useGetSnapshotsGraph,
    useGetSnapshotsStatsById
} from "gen/v0/api/snapshots/snapshots";
import {
    ErrWarnDetails,
    GraphSnapshotRequest,
    GraphSnapshotResponse,
    GraphSnapshotStatsResponse,
    RestConfigGraphV2GraphDefinition as GraphDefinition
} from "gen/v0/types";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { AxiosResponse, CustomError } from "types/common";
import { EdgeStats } from "types/graphs";
import { ExtendedColStatsResponse } from "types/snapshots";
import { GraphSnapshotID } from "types/training_job";
import { HandleErrorFunc } from "types/validation";

import { jobStatus, JobStatusType } from "utils/Constants";
import { invalidateSnapshotsGraph } from "utils/queryInvalidation";
import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useGraphSnapshot = (
    graph_snapshot_id: string | undefined,
    graph_definiton: GraphDefinition | undefined | null,
    graph_version: string | null | undefined,
    setPageLoading: (isLoading: boolean) => void,
    handleError?: HandleErrorFunc,
    setStatus?: (status: jobStatus) => void,
    setWarnings?: (status: ErrWarnDetails[]) => void
) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const [prevStatus, setPrevStatus] = useState<JobStatusType>(jobStatus.None);
    const queryClient = useQueryClient();

    const request: GraphSnapshotRequest | null = graph_snapshot_id
        ? {
              graph_snapshot_id,
              graph_definiton: graph_definiton ?? undefined,
              graph_version: graph_version ?? undefined
          }
        : null;

    const {
        data: response,
        isLoading,
        isError,
        error
    } = useGetSnapshotsGraph(
        request!,
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.SnapshotsGraph,
            graph_snapshot_id,
            Boolean(graph_snapshot_id && projectId),
            (query) => {
                if (!query.state.data) return 15000;
                const resp = query?.state
                    ?.data as AxiosResponse<GraphSnapshotResponse>;
                const status = resp?.data?.snapshot_resource?.stages?.INGEST
                    ?.status as jobStatus;
                return [
                    jobStatus.Done,
                    jobStatus.Failed,
                    jobStatus.Cancelled
                ].includes(status)
                    ? false
                    : 15000;
            }
        )
    );

    // Handle errors

    useEffect(() => {
        if (!isLoading) {
            setPageLoading(false);
        }
        if (isError && error) {
            const title = "Unable to fetch graph snapshots";
            if (handleError) handleError(error as CustomError, title);
        }
    }, [isLoading, isError, error, setPageLoading, handleError]);

    useEffect(
        function handleIngestStatusEffect() {
            const ingestStatus =
                response?.data?.snapshot_resource?.stages?.INGEST?.status;
            const warnings =
                response?.data?.snapshot_resource?.stages?.INGEST?.warnings;
            if (ingestStatus && setStatus) {
                setStatus(ingestStatus as jobStatus);
            }

            if (
                prevStatus === jobStatus.None &&
                (ingestStatus === jobStatus.Queued ||
                    ingestStatus === jobStatus.Running)
            ) {
                invalidateSnapshotsGraph(queryClient, graph_snapshot_id);
            }
            if (warnings && warnings?.length > 0) {
                setWarnings?.(
                    warnings.map((warning) => ({
                        title: "",
                        description: warning,
                        is_error: false,
                        is_dismissible: true
                    }))
                );
            }
            setPrevStatus(ingestStatus as JobStatusType);
        },
        [
            response,
            setStatus,
            prevStatus,
            queryClient,
            graph_snapshot_id,
            setWarnings
        ]
    );

    return {
        isLoading,
        graphSnapshot: response?.data ?? null,
        stages: response?.data?.snapshot_resource?.stages ?? null
    };
};

export const useGraphEdges = (
    id: GraphSnapshotID,
    graphSnapshot: GraphSnapshotResponse | null
): {
    graphEdges: EdgeStats[] | null | undefined;
    graphEdgeStatsError: unknown;
    isGraphEdgeStatsError: boolean;
} => {
    const { queryOption } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const {
        data: graphEdges,
        error: graphEdgeStatsError,
        isError: isGraphEdgeStatsError
    } = useGetSnapshotsEdgeStatsById<EdgeStats[] | null>(
        id,
        { project_id: projectId },
        queryOption.graphEdgeStats(id, graphSnapshot)
    );

    return {
        graphEdges,
        graphEdgeStatsError,
        isGraphEdgeStatsError
    };
};

export const useGraphSnapshotStats = (
    id: GraphSnapshotID
): { totalRows: number | null; totalSizeBytes: number | null } => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const { data: response } = useGetSnapshotsStatsById<{
        data: GraphSnapshotStatsResponse | null;
    }>(
        id,
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.SnapshotsStatsById,
            id,
            Boolean(projectId)
        )
    );

    const data = response?.data ?? null;

    return {
        totalRows: data?.total_rows ?? null,
        totalSizeBytes: data?.total_size_bytes ?? null
    };
};

export const useSnapshotColumnStats = (snapshotId: string | null) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const [columnStats, setColumnStats] =
        useState<ExtendedColStatsResponse | null>(null);

    const [processingSamples, setFetchingSamples] = useState<boolean>(false);

    const {
        data: colStatsResponse,
        isError,
        error
    } = useGetSnapshotsColStatsById<{ data: ExtendedColStatsResponse }>(
        snapshotId ?? "",
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.SnapshotsColStatsById,
            snapshotId,
            Boolean(snapshotId && projectId)
        )
    );

    useEffect(
        function prepareSampleRowsEffect() {
            if (!snapshotId || !colStatsResponse?.data) return;
            setFetchingSamples(true);
            try {
                const data: ExtendedColStatsResponse = {
                    ...colStatsResponse.data,
                    columns:
                        colStatsResponse.data.columns?.map((c) => ({ ...c })) ??
                        []
                };

                data?.columns?.forEach((element) => {
                    if (element?.sample_rows) {
                        const samples = element?.sample_rows as string[];
                        let nonEmpty: string | null | undefined = samples?.find(
                            (d) => d !== null
                        );
                        if (nonEmpty === undefined) {
                            nonEmpty = null; // secure fallback
                        }
                        element.sample_rows =
                            nonEmpty === null ? nonEmpty : samples;
                        element.nonEmpty = nonEmpty;
                    } else if (!data.statistics_available) {
                        element.sample_rows = [];
                        element.isSampleAvailable = true;
                    }
                });

                setColumnStats(data);
            } catch (err) {
                console.error(err);
            } finally {
                setFetchingSamples(false);
            }
        },
        [colStatsResponse?.data, snapshotId]
    );

    useEffect(() => {
        if (isError && error) {
            console.error(error);
            setFetchingSamples(false);
        }
    }, [isError, error]);

    return {
        columnStats,
        fetchingSamples: processingSamples,
        isError,
        error
    };
};
