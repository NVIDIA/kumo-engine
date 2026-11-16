// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from "react";

import {
    useGetSnapshotsColStatsById,
    useGetSnapshotsTableById
} from "gen/v0/api/snapshots/snapshots";
import { TableSnapshotResponse } from "gen/v0/types";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";
import { ExtendedColStatsResponse } from "types/snapshots";
import { JSTable } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { tableSnapshotToJSTable } from "utils/Converters/snapshotConverter";
import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useGetSnapshotsTableInfo = (
    tableId: string | null,
    handleError: (err: Error) => void,
    enabled: boolean = true
): {
    snapshotTableInfo: JSTable | null;
    fetchSnapshotTableInfo: () => void;
    loading: boolean;
} => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const {
        data: tableInfo,
        isLoading: loading,
        refetch: fetchSnapshotTableInfo,
        error,
        isError
    } = useGetSnapshotsTableById(
        tableId ?? "",
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.SnapshotsTableById,
            tableId,
            enabled && Boolean(projectId)
        )
    );

    useEffect(() => {
        if (loading) return;
        if (isError) {
            handleError(error as unknown as CustomError);
        }
    }, [error, handleError, isError, loading]);

    // Convert API response into your internal JSTable
    const snapshotTableInfo: JSTable | null = useMemo(() => {
        return tableInfo
            ? tableSnapshotToJSTable(
                  tableInfo.data as TableSnapshotResponse,
                  true
              )
            : null;
    }, [tableInfo]);

    return { snapshotTableInfo, fetchSnapshotTableInfo, loading };
};

export const useGetSnapshotsColumnStatsInfo = (
    id: string | null,
    handleError: HandleErrorFunc,
    isMounted: React.MutableRefObject<boolean>,
    enabled: boolean = true
) => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId() ?? "";
    const [columnStats, setColumnStats] =
        useState<ExtendedColStatsResponse | null>(null);

    const {
        data: columnStatsResponse,
        isLoading: loading,
        refetch: fetchColumnStats,
        error,
        isError
    } = useGetSnapshotsColStatsById(
        id ?? "",
        { project_id: projectId },
        getQueryOptionsById(
            ReactQueryKeys.ColStatsBySnapshotId,
            id,
            enabled && Boolean(projectId),
            columnStats?.statistics_available !== true &&
                id &&
                isMounted?.current
                ? 2000
                : false
        )
    );

    // Handle errors
    useEffect(() => {
        if (isError) {
            handleError(
                error as CustomError,
                "Failed to get the column statistics!"
            );
        }
    }, [error, handleError, isError]);

    // Process data
    useEffect(() => {
        if (!columnStatsResponse?.data) {
            setColumnStats(null);
            return;
        }
        setColumnStats({ ...columnStatsResponse.data });
    }, [columnStatsResponse?.data]);

    return {
        columnStats,
        loading: !!id && loading,
        fetchColumnStats,
        error,
        isError
    };
};
