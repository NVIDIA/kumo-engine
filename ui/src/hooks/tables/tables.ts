// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, {
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState
} from "react";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import axios, { AxiosProgressEvent, AxiosRequestConfig } from "axios";
import {
    useCreateProjectsCompleteByProjectId,
    useCreateProjectsDeleteByProjectId,
    useCreateProjectsStartByProjectId
} from "gen/v0/api/connectors/connectors";
import {
    CompleteFileUploadRequest,
    DeleteUploadedFileRequest,
    StartFileUploadRequest
} from "gen/v0/types";
import {
    getV1TablesByTableIdOrName,
    useCreateV1Tables,
    useCreateV1TablesRefreshByTableId,
    useDeleteV1TablesByTableAlias,
    useGetV1Tables,
    useGetV1TablesByTableIdOrName,
    useGetV1TablesColStatsByTableId,
    useValidateV1Tables
} from "gen/v1/api/tables/tables";
import {
    CreateV1TablesParams,
    HTTPValidationError,
    KumoapiTableTableDefinition,
    TableResourceResponse
} from "gen/v1/types";
import { useCacheStaleTime } from "stores/cacheStore";

import { useProjectId } from "hooks/project/projects";

import axiosClient from "services/axios_client";

import { CustomError } from "types/common";
import { ExtendedColStatsResponse } from "types/snapshots";
import { JSTable, TableListResponse } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { getProjectPath, storageKeys } from "utils/Constants";
import {
    getConnectorIDFromTableDefSource,
    tableFromTableResponse,
    tableListFromListResp
} from "utils/Converters/tablesConverter";
import { useCustomNavigation } from "utils/navigator";
import { ReactQueryKeys } from "utils/reactQueryKeys";

import { useQueryOptions } from "../misc/queryOptions";

export const useColumnStatsInfo = (
    id: string | null,
    handleError: HandleErrorFunc,
    isMounted: React.MutableRefObject<boolean>,
    enabled: boolean = true
) => {
    const pollingRef = useRef<NodeJS.Timeout | null>(null);

    const [columnStats, setColumnStats] =
        useState<ExtendedColStatsResponse | null>(null);

    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId();

    const {
        data: columnStatsResponse,
        isLoading: loading,
        refetch: fetchColumnStats,
        error,
        isError
    } = useGetV1TablesColStatsByTableId(
        id ?? "",
        { project_id: projectId ?? "" },
        getQueryOptionsById(
            ReactQueryKeys.ColStatsByTableId,
            id,
            enabled && Boolean(projectId)
        )
    );

    useEffect(() => {
        if (isError) {
            handleError(
                error as CustomError,
                "Failed to get the column statistics!"
            );
        }
    }, [error, handleError, isError]);

    useEffect(() => {
        if (!columnStatsResponse?.data) {
            setColumnStats(null);
            return;
        }
        const processedData = { ...columnStatsResponse.data };
        setColumnStats(processedData);
    }, [columnStatsResponse?.data]);

    // Polling logic - use columnStats from state
    useEffect(() => {
        if (pollingRef?.current) {
            clearTimeout(pollingRef?.current);
            pollingRef.current = null;
        }

        if (
            columnStats?.statistics_available !== true &&
            id &&
            isMounted?.current
        ) {
            pollingRef.current = setTimeout(() => {
                if (isMounted?.current) {
                    fetchColumnStats();
                }
            }, 2000);
        }

        return () => {
            if (pollingRef.current) {
                clearTimeout(pollingRef.current);
                pollingRef.current = null;
            }
        };
    }, [columnStats, id, fetchColumnStats, isMounted]);

    // Clean up on unmount
    useEffect(() => {
        return () => {
            if (pollingRef.current) {
                clearTimeout(pollingRef.current);
                pollingRef.current = null;
            }
        };
    }, []);

    return {
        columnStats,
        loading: !!id && loading,
        fetchColumnStats,
        error,
        isError
    };
};

export const useGetTableInfo = (
    tableId: string | null,
    handleError: (err: Error) => void,
    enabled: boolean = true
): {
    selectedTableInfo: JSTable | null;
    fetchTableInfo: () => void;
    loading: boolean;
} => {
    const { getQueryOptionsById } = useQueryOptions();
    const projectId = useProjectId();
    const {
        data: tableInfo,
        isLoading: loading,
        refetch: fetchTableInfo,
        error,
        isError
    } = useGetV1TablesByTableIdOrName(
        tableId ?? "",
        { project_id: projectId ?? "" },
        getQueryOptionsById(
            ReactQueryKeys.TablesById,
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
    const selectedTableInfo: JSTable | null = useMemo(() => {
        return tableInfo?.data
            ? tableFromTableResponse(tableInfo.data, true)
            : null;
    }, [tableInfo?.data]);

    return { selectedTableInfo, fetchTableInfo, loading };
};

export const useGetMultipleTableInfo = (
    projectId?: string | null
): {
    getTables: (
        params: {
            tableIds: string[];
            tableNames: string[];
        },
        callback: (d: {
            tableNames: string[];
            data: (TableResourceResponse | null)[];
        }) => void
    ) => Promise<void>;
} => {
    const { medium } = useCacheStaleTime();
    const queryClient = useQueryClient();

    const { mutate } = useMutation({
        mutationFn: async ({
            tableIds,
            tableNames
        }: {
            tableIds: string[];
            tableNames: string[];
        }) => {
            if (!projectId) {
                throw new Error("A project must be selected.");
            }
            const queries = tableIds.map((id) => {
                const cachedData = queryClient.getQueryData([
                    ReactQueryKeys.TablesById,
                    id
                ]);
                if (cachedData) {
                    return {
                        id,
                        data: (cachedData as { data: TableResourceResponse })
                            .data,
                        cached: true,
                        success: true
                    };
                }
                return getV1TablesByTableIdOrName(`${id}`, {
                    project_id: projectId
                });
            });
            const responses = await Promise.allSettled(queries);
            const data: (TableResourceResponse | null)[] = [];
            responses.forEach((res, i) => {
                if (res.status === "fulfilled" && res?.value?.data) {
                    const cacheKeys = [ReactQueryKeys.TablesById, tableIds[i]];
                    queryClient.setQueryData(cacheKeys, {
                        data: res.value.data
                    });
                    // Set query defaults for this specific query key
                    queryClient.setQueryDefaults(cacheKeys, {
                        staleTime: medium,
                        gcTime: medium * 2
                    });
                    data.push(res?.value?.data);
                } else {
                    data.push(null);
                }
            });

            return {
                tableNames,
                data: data
            };
        }
    });

    const getTables = useCallback(
        async (
            params: {
                tableIds: string[];
                tableNames: string[];
            },
            callback: (d: {
                tableNames: string[];
                data: (TableResourceResponse | null)[];
            }) => void
        ) => {
            try {
                mutate(
                    {
                        tableIds: params.tableIds,
                        tableNames: params.tableNames
                    },
                    {
                        onSuccess(data) {
                            callback(data);
                        }
                    }
                );
            } catch {
                /* empty */
            }
        },
        [mutate]
    );

    return {
        getTables
    };
};

export const useTables = (
    handleError?: HandleErrorFunc,
    setPageLoading?: (v: boolean) => void,
    enabled: boolean = true,
    projectId?: string
) => {
    const selectedProject = useProjectId() ?? projectId;
    const [connectorPresent, setConnectorPresent] = useState(true);
    const [tableList, setTables] = useState<Array<TableListResponse>>([]);
    const queryClient = useQueryClient();
    const navigate = useCustomNavigation();
    const { getQueryOptions } = useQueryOptions();

    const {
        data: tablesListResponse,
        isLoading,
        error,
        isError,
        refetch
    } = useGetV1Tables(
        { project_id: selectedProject ?? "" },
        getQueryOptions({
            queryKey: [ReactQueryKeys.TablesList, selectedProject ?? ""],
            enabled: enabled && Boolean(selectedProject)
        })
    );

    useEffect(() => {
        if (isError) {
            setPageLoading?.(false);
            handleError?.(error as unknown as CustomError);
        }
    }, [error, handleError, setPageLoading, isError]);

    useEffect(() => {
        if (isLoading) return;

        const rows = tablesListResponse?.data ?? [];
        if (rows.length > 0) {
            const resp = tableListFromListResp(rows);
            const sortedTables = [...resp].sort((a, b) => {
                const dateA = new Date(a?.timestamps?.updated_at || "");
                const dateB = new Date(b?.timestamps?.updated_at || "");
                return dateB?.getTime() - dateA?.getTime();
            });
            setTables(sortedTables);
            setConnectorPresent(true);
            setPageLoading?.(false);
        } else {
            setTables([]);
            setConnectorPresent(false);
            setPageLoading?.(false);
        }
    }, [tablesListResponse?.data, isLoading, setPageLoading]);

    const deleteTableMutation = useDeleteV1TablesByTableAlias({});

    const deleteTable = (
        id: string,
        name: string,
        clearSelected: (v: TableListResponse | null) => void
    ) => {
        if (!selectedProject) {
            handleError?.(
                new Error("A project must be selected."),
                "Unable to delete table"
            );
            return;
        }
        setPageLoading?.(true);
        const params = {
            tableAlias: name as string,
            params: { project_id: selectedProject }
        };
        deleteTableMutation.mutate(params, {
            onSuccess: () => {},
            onError(error) {
                setPageLoading?.(false);
                const title = "Unable to delete table " + name;
                handleError?.(error as CustomError, title);
            }
        });
        setPageLoading?.(false);

        // Remove deleted entity from `tablesListResponse`
        // and update the cached data.
        if (tablesListResponse?.data) {
            const updatedItems = tablesListResponse?.data.filter(
                (item) => item?.id?.split("-").pop() !== id
            );
            const tmpResponse = { ...tablesListResponse };
            tmpResponse.data = updatedItems;
            queryClient?.setQueryData(
                [ReactQueryKeys.TablesList, selectedProject ?? ""],
                tmpResponse
            );
            clearSelected(null);
            queryClient.invalidateQueries({
                queryKey: [ReactQueryKeys.TablesList, id]
            });
            queryClient.invalidateQueries({
                queryKey: [ReactQueryKeys.ColStatsByTableId, id]
            });
        }

        navigate(getProjectPath(selectedProject).Tables, { replace: true });
    };

    const refreshTableMutation = useCreateV1TablesRefreshByTableId({
        mutation: {
            onSuccess: async (data, variables) => {
                const { tableId } = variables;
                await refetch();
                queryClient.invalidateQueries({
                    queryKey: [ReactQueryKeys.ColStatsByTableId, tableId]
                });

                queryClient.invalidateQueries({
                    queryKey: [ReactQueryKeys.TablesList, tableId]
                });
                setPageLoading?.(false);
            },
            onError: (error: HTTPValidationError | null) => {
                setPageLoading?.(false);
                handleError?.(error as CustomError, "Request failed");
            }
        }
    });

    const refreshTable = useCallback(
        (tableId: string) => {
            if (!selectedProject) {
                handleError?.(
                    new Error("A project must be selected."),
                    "Unable to refresh table"
                );
                return;
            }
            setPageLoading?.(true);
            refreshTableMutation.mutate({
                tableId,
                params: { project_id: selectedProject }
            });
        },
        [handleError, refreshTableMutation, setPageLoading, selectedProject]
    );

    const refetchTables = useCallback(async () => {
        await refetch();
    }, [refetch]);

    return {
        isDataLoading: isLoading,
        connectorPresent,
        tableList,
        tablesPresent: tableList.length > 0,
        deleteTable,
        refreshTable,
        refetchTables
    };
};

export const useFileUpload = (handleUploadError: HandleErrorFunc) => {
    const projectId = useProjectId();
    const startUploadMutation = useCreateProjectsStartByProjectId({});
    const completeUploadMutation = useCreateProjectsCompleteByProjectId({});
    const deleteFileMutation = useCreateProjectsDeleteByProjectId({});

    const requireProjectId = useCallback(() => {
        if (!projectId) {
            throw new Error("A project is required for file uploads");
        }
        return projectId;
    }, [projectId]);

    const startFileUpload = useCallback(
        async (config: StartFileUploadRequest) => {
            try {
                // Use mutateAsync instead of mutate
                const response = await startUploadMutation.mutateAsync({
                    projectId: requireProjectId(),
                    data: config
                });

                return response;
            } catch (error) {
                if (handleUploadError) {
                    handleUploadError(error as CustomError);
                }
                // Re-throw so the caller can handle it
                throw error;
            }
        },
        [handleUploadError, requireProjectId, startUploadMutation]
    );

    const completeFileUpload = useCallback(
        (config: CompleteFileUploadRequest, successHandler: () => void) => {
            completeUploadMutation.mutate(
                { projectId: requireProjectId(), data: config },
                {
                    onSuccess: () => {
                        if (successHandler) {
                            successHandler();
                        }
                    },
                    onError(error) {
                        if (handleUploadError) {
                            handleUploadError(error as CustomError);
                        }
                    }
                }
            );
        },
        [handleUploadError, completeUploadMutation, requireProjectId]
    );

    const deleteUploadedFile = useCallback(
        (config: DeleteUploadedFileRequest, successHandler?: () => void) => {
            deleteFileMutation.mutate(
                { projectId: requireProjectId(), data: config },
                {
                    onSuccess: () => {
                        successHandler?.();
                    },
                    onError(e) {
                        console.warn("Failed to delete table: ", e);
                    }
                }
            );
        },
        [deleteFileMutation, requireProjectId]
    );

    const upload = useCallback(
        async (
            url: string,
            chunk: FormData | Blob,
            onUploadProgress?: (progressEvent: AxiosProgressEvent) => void,
            signal?: AbortSignal
        ) => {
            const config: AxiosRequestConfig = { onUploadProgress };
            if (url.startsWith("/")) {
                return axiosClient.put(url, chunk, {
                    ...config,
                    signal,
                    headers: {
                        "Content-Type": "application/octet-stream"
                    }
                });
            }
            // Per-request: axios has no merge rule for `signal`, so an
            // instance default is not reliably carried through.
            return axios.create(config).put(url, chunk, { signal });
        },
        []
    );

    return { startFileUpload, completeFileUpload, deleteUploadedFile, upload };
};

export const useCreateTable = (handleError: HandleErrorFunc) => {
    const createTableMutation = useCreateV1Tables({});
    const validateMutation = useValidateV1Tables({});
    const projectId = useProjectId();

    const createTable = useCallback(
        async (
            tableArgs: KumoapiTableTableDefinition,
            params: CreateV1TablesParams
        ) => {
            if (tableArgs?.source_table) {
                const connectorId = getConnectorIDFromTableDefSource(
                    tableArgs.source_table
                );
                const connector = {
                    value: connectorId,
                    label: connectorId,
                    description: ""
                };

                storeToLocalStorage(storageKeys.lastUsedConn, connector);
            }
            try {
                const response = await createTableMutation.mutateAsync({
                    data: tableArgs,
                    params: {
                        ...params,
                        project_id: projectId ?? ""
                    }
                });

                return response;
            } catch (error) {
                if (handleError) {
                    const title = "Failed to create table!";
                    handleError(error as CustomError, title);
                }
            }
        },
        [createTableMutation, handleError, projectId]
    );

    const validateTables = useCallback(
        async (tableArgs: KumoapiTableTableDefinition) => {
            try {
                // Use mutateAsync instead of mutate
                const response = await validateMutation.mutateAsync({
                    data: tableArgs,
                    params: { project_id: projectId ?? "" }
                });

                return response;
            } catch (error) {
                if (handleError) {
                    const title = "Failed to create table!";
                    handleError(error as CustomError, title);
                }
            }
        },
        [handleError, projectId, validateMutation]
    );

    return { createTable, validateTables };
};
