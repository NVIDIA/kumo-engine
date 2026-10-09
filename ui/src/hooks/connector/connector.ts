// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";

import {
    getFromLocalStorage,
    removeFromLocalStorage
} from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import {
    useDeleteProjectsConnectorsByConnectorId,
    useGetProjectsColumnsByConnectorId,
    useGetProjectsConnectorsByConnectorId,
    useGetProjectsConnectorsByProjectId,
    useGetProjectsSampleByConnectorId
} from "gen/v0/api/connectors/connectors";
import {
    GetProjectsColumnsByConnectorIdParams,
    GetProjectsSampleByConnectorIdParams,
    SourceColumn
} from "gen/v0/types";
import {
    useDeleteV1TablesByTableAlias,
    useGetV1TablesListTablesByConnectorId
} from "gen/v1/api/tables/tables";
import { TableResourceResponse } from "gen/v1/types";
import { useCacheStaleTime } from "stores/cacheStore";

import { useQueryOptions } from "hooks/misc/queryOptions";
import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";
import { JsConnector } from "types/connectors";
import { ConnectorDropdownOption } from "types/inputs";
import { JSTable } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { getProjectPath, storageKeys } from "utils/Constants";
import { connectorListFromConnectorResp } from "utils/Converters/connectorsConverter";
import { connectorFromConnectorResp } from "utils/Converters/connectorsConverter";
import { tableFromTableResponse } from "utils/Converters/tablesConverter";
import { isValid } from "utils/Helpers/typeGuards";
import { useCustomNavigation } from "utils/navigator";
import {
    invalidateProjectTableList,
    invalidateTablesList
} from "utils/queryInvalidation";
import { ReactQueryKeys } from "utils/reactQueryKeys";

export const useConnectors = (
    handleError?: HandleErrorFunc,
    setPageLoading?: (v: boolean) => void,
    enabled: boolean = true
) => {
    const navigate = useCustomNavigation();
    const projectId = useProjectId() ?? "";
    const queryClient = useQueryClient();
    const { getQueryOptions } = useQueryOptions();
    const connectorListQueryKey = useMemo(
        () => [ReactQueryKeys.ConnectorList, projectId],
        [projectId]
    );

    const {
        data: connectorListResponse,
        isLoading,
        error,
        isError
    } = useGetProjectsConnectorsByProjectId(
        projectId,
        getQueryOptions({
            queryKey: connectorListQueryKey,
            enabled: enabled && Boolean(projectId)
        })
    );

    const connectorList = useMemo(
        () => connectorListFromConnectorResp(connectorListResponse?.data ?? []),
        [connectorListResponse?.data]
    );

    useEffect(() => {
        if (isError) {
            setPageLoading?.(false);
            handleError?.(error as unknown as CustomError);
        } else if (!isLoading) {
            setPageLoading?.(false);
        }
    }, [error, handleError, isError, isLoading, setPageLoading]);

    const deleteConnectorMutation = useDeleteProjectsConnectorsByConnectorId(
        {}
    );

    const deleteConnector = useCallback(
        (
            id: string,
            name: string,
            clearSelected: (v: JsConnector | null) => void
        ) => {
            if (!projectId) return;
            deleteConnectorMutation.mutate(
                { projectId, connectorId: id },
                {
                    onSuccess: () => {
                        // Remove deleted entity from `connectorListResponse`
                        // and update the cached data.
                        if (connectorListResponse?.data) {
                            const updatedItems =
                                connectorListResponse.data.filter(
                                    (item) => item.id !== id
                                );

                            const tmpResponse = { ...connectorListResponse };
                            tmpResponse.data = updatedItems;
                            invalidateProjectTableList(
                                queryClient,
                                projectId,
                                id
                            );
                            // Updating cache
                            queryClient.setQueryData(
                                connectorListQueryKey,
                                tmpResponse
                            );
                            clearSelected(null);
                        }
                        const lastUsed = getFromLocalStorage<{
                            value: string;
                            type: string;
                        }>(storageKeys.lastUsedConn);
                        if (
                            isValid(lastUsed) &&
                            isValid(lastUsed?.value) &&
                            lastUsed?.value === id
                        ) {
                            removeFromLocalStorage(storageKeys.lastUsedConn);
                        }
                        navigate(getProjectPath(projectId).Connectors, {
                            replace: true
                        });
                    },
                    onError(error) {
                        if (handleError) {
                            const title = "Unable to delete connector " + name;
                            handleError(error as CustomError, title);
                        }
                    }
                }
            );
        },
        [
            connectorListResponse,
            deleteConnectorMutation,
            handleError,
            navigate,
            projectId,
            queryClient,
            connectorListQueryKey
        ]
    );

    return {
        isDataLoading: isLoading,
        isConnectorsPresent: connectorList.length > 0,
        isDBConnectorsPresent: connectorList.length > 0,
        connectorList,
        deleteConnector
    };
};

export const useConnectorDropdownOptions = (
    handleError: (err: Error) => void,
    enabled = true
): {
    connectors: Array<ConnectorDropdownOption>;
} => {
    const [connectors, setConnectors] = useState<
        Array<ConnectorDropdownOption>
    >([]);
    const {
        isDataLoading: fetchingConnectors,
        connectorList: connectorListOriginal
    } = useConnectors(undefined, undefined, enabled);
    const getConnectorList = useCallback(async () => {
        if (fetchingConnectors) {
            return;
        }
        try {
            if (connectorListOriginal) {
                setConnectors(
                    connectorListOriginal
                        .filter(({ id }) => id !== null)
                        .map((c) => ({
                            value: c?.id,
                            label: c?.name,
                            type: c?.config?.type,
                            config: c?.config
                        }))
                );
            }
        } catch (error) {
            handleError(error as CustomError);
        }
    }, [connectorListOriginal, fetchingConnectors, handleError]);

    useEffect(() => {
        getConnectorList();
    }, [getConnectorList]);

    return { connectors };
};

/**
 *  This hook invokes REST API call(inside `useEffect`) and fetches all the tables
 *  and sets it to state variable.
 *
 *  The `response` will converted to `Array<JsConnector>` format and stored in
 *  state variable to use it in React Components
 *
 */
export const useConnectedTables = (
    connectorId: string,
    handleError: (err: Error, title?: string) => void,
    setLoading: React.Dispatch<React.SetStateAction<boolean>>,
    setConnectedTables?: (tables: JSTable[]) => void
) => {
    const queryClient = useQueryClient();
    const [tables, setTables] = useState<Array<JSTable>>([]);
    const { medium } = useCacheStaleTime();
    const projectId = useProjectId();
    function sortTablesByTime(data: TableResourceResponse[]) {
        let tablesByTime: Array<JSTable> = [];
        if (data) {
            tablesByTime = data.map((tr) => tableFromTableResponse(tr));
            // sort created date and updated date column data in
            // descending order
            tablesByTime.sort((a, b) =>
                (a?.created as string) > (b?.created as string) ? -1 : 1
            );
            tablesByTime.sort((a, b) =>
                (a?.updated as string) > (b?.updated as string) ? -1 : 1
            );
        }
        return tablesByTime;
    }
    const {
        data: tableListResponse,
        isLoading,
        error,
        isError
    } = useGetV1TablesListTablesByConnectorId(
        connectorId,
        { project_id: projectId ?? "" },
        {
            query: {
                queryKey: [ReactQueryKeys.TablesList, projectId, connectorId],
                staleTime: medium,
                retry: false,
                enabled: Boolean(projectId && connectorId)
            }
        }
    );

    useEffect(() => {
        if (!connectorId) {
            setLoading(false);
            return;
        }
        if (isError) {
            setLoading(false);
            handleError(error as CustomError);
        }
        if (!tableListResponse) return;

        if (!isLoading) {
            setLoading(false);
            setTables(sortTablesByTime(tableListResponse.data));
            if (setConnectedTables) {
                setConnectedTables?.(sortTablesByTime(tableListResponse.data));
            }
        }
    }, [
        error,
        handleError,
        isError,
        isLoading,
        tableListResponse,
        setLoading,
        connectorId,
        setConnectedTables
    ]);

    const deleteTablesMutation = useDeleteV1TablesByTableAlias({});

    const deleteTable = (nameAlias: string) => {
        setLoading(true);
        deleteTablesMutation.mutate(
            {
                tableAlias: nameAlias as string,
                params: { project_id: projectId ?? "" }
            },
            {
                onSuccess: () => {
                    setLoading(false);
                    setTables(
                        tables?.filter(
                            (table) => table?.name_alias !== nameAlias
                        )
                    );
                    if (tableListResponse) {
                        const updatedItems = tableListResponse?.data?.filter(
                            (item) => item.name_alias !== nameAlias
                        );
                        tableListResponse.data = updatedItems;
                        queryClient.setQueryData(
                            [ReactQueryKeys.TablesList, projectId, connectorId],
                            tableListResponse
                        );
                    }
                    if (projectId) {
                        invalidateProjectTableList(
                            queryClient,
                            projectId,
                            connectorId
                        );
                    }
                    invalidateTablesList(queryClient);
                },
                onError(error) {
                    setLoading(false);
                    const title = "Unable to delete table " + nameAlias;
                    handleError(error as CustomError, title);
                }
            }
        );
    };

    return { isLoading, setTables, tables, deleteTable };
};

export const useConnectorDetails = (
    connectorId: string,
    handleError: (err: Error, title?: string) => void
) => {
    const projectId = useProjectId() ?? "";
    const { getQueryOptionsById } = useQueryOptions();
    const {
        data: connectorResponse,
        isLoading,
        error,
        isError
    } = useGetProjectsConnectorsByConnectorId(
        projectId,
        connectorId,
        getQueryOptionsById(
            ReactQueryKeys.ConnectorsById,
            connectorId,
            isValid(connectorId) && Boolean(projectId)
        )
    );

    useEffect(() => {
        if (isError && error) {
            handleError(error as CustomError);
        }
    }, [error, handleError, isError]);

    return {
        isLoading,
        connectorResponse: connectorResponse?.data
            ? connectorFromConnectorResp(connectorResponse.data)
            : undefined
    };
};

export const useGetColumns = (
    config: {
        connectorId?: string;
        // `query` (a defining SELECT) opts the project-scoped discovery endpoint
        // into Custom SQL column discovery: the backend describes the query's
        // output columns instead of a raw source table. It rides in `params` so
        // it enters the request and the generated cache key (alongside
        // projectId + connectorId).
        params?: { source_table_name?: string; query?: string };
    },
    handleError: (err: Error) => void,
    setLoading: (v: boolean) => void,
    enabled: boolean = true
) => {
    const projectId = useProjectId() ?? "";
    const sourceTableName = config?.params?.source_table_name ?? "";
    const sqlQuery = config?.params?.query;
    const columnsParams: GetProjectsColumnsByConnectorIdParams & {
        query?: string;
    } = {
        table_name: sourceTableName,
        ...(sqlQuery ? { query: sqlQuery } : {})
    };
    const { data, isLoading, error, isError, isFetching } =
        useGetProjectsColumnsByConnectorId(
            projectId,
            config.connectorId ?? "",
            columnsParams,
            {
                query: {
                    enabled: Boolean(
                        enabled &&
                        projectId &&
                        config.connectorId &&
                        // Custom SQL discovers from `query` with no source table,
                        // so either identifies a valid request.
                        (sourceTableName || sqlQuery)
                    ),
                    // A Custom SQL `query` describe is a warehouse round-trip and
                    // its failures (bad syntax / unknown function / missing table)
                    // are deterministic, so skip the global `retry: 1` — a second
                    // describe only doubles the latency before the warehouse's
                    // reason is shown.
                    ...(sqlQuery ? { retry: false } : {})
                }
            }
        );

    useEffect(() => {
        if (enabled && data?.data.cols && !isLoading) {
            setLoading(false);
        }
        if (isError && error) {
            setLoading(false);
            handleError(error as Error);
        }
    }, [
        data?.data.cols,
        enabled,
        error,
        handleError,
        isError,
        isLoading,
        setLoading
    ]);

    return {
        columns: data?.data.cols,
        sourceTableName: data?.data.table_name ?? null,
        isLoading: isLoading || isFetching,
        columnWarnings: null
    };
};

export const useGetSampleRows = (
    config: {
        connectorId?: string;
        // `query` (a defining SELECT) makes the project-scoped sample endpoint
        // return the Custom SQL query's bounded output instead of a raw source
        // table. It rides in `params` so it enters the request and the generated
        // cache key.
        params: {
            source_table_name?: string;
            num_rows?: number;
            query?: string;
        };
    },
    handleError: (err: Error, title?: string) => void,
    enabled: boolean = true
) => {
    const projectId = useProjectId() ?? "";
    const sourceTableName = config.params.source_table_name ?? "";
    const sqlQuery = config.params.query;
    const sampleParams: GetProjectsSampleByConnectorIdParams & {
        query?: string;
    } = {
        table_name: sourceTableName,
        sample_rows: config.params.num_rows,
        ...(sqlQuery ? { query: sqlQuery } : {})
    };
    const { data, isLoading, isError, error } =
        useGetProjectsSampleByConnectorId(
            projectId,
            config.connectorId ?? "",
            sampleParams,
            {
                query: {
                    enabled: Boolean(
                        enabled &&
                        projectId &&
                        config.connectorId &&
                        (sourceTableName || sqlQuery)
                    )
                }
            }
        );

    useEffect(() => {
        if (isError && error) {
            handleError(error as Error, "Failed to fetch sample rows");
        }
    }, [error, handleError, isError]);

    const columnsWithRows = useMemo(() => {
        if (!data?.data) return null;
        const columns = data.data.cols ?? [];
        const serialized = data.data.sample_rows;
        if (!serialized) return { columns, rows: [] };
        try {
            const parsed = JSON.parse(serialized) as {
                data?: Record<string, unknown>[];
            };
            return {
                columns,
                rows: (parsed.data ?? []).map((row) =>
                    columns.map((column) => row[column.name] ?? null)
                )
            };
        } catch (parseError) {
            handleError(
                parseError as Error,
                "Failed to parse source sample rows"
            );
            return { columns: [] as SourceColumn[], rows: [] };
        }
    }, [data?.data, handleError]);

    return {
        columnsWithRows,
        isLoading
    };
};
