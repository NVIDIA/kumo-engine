// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useQueryClient } from "@tanstack/react-query";
import { ResourceTimestamps } from "gen/v0/types";
import {
    useCreateV1GraphsSuggestLinks,
    useDeleteV1GraphsByGraphAlias,
    useGetV1Graphs,
    useGetV1GraphsByGraphIdOrAlias
} from "gen/v1/api/graphs/graphs";
import { useCacheStaleTime } from "stores/cacheStore";

import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";
import {
    ColumnGroup,
    GraphFlowInfo,
    GraphListResponse,
    GraphTables,
    JSGraph
} from "types/graphs";
import { HandleErrorFunc } from "types/validation";

import { errorStatusCode } from "utils/Constants";
import { jsGraphFromGraphResponse } from "utils/Converters/graphsConverter";
import { createGraphBody, getpKeys } from "utils/graph";
import { getKeys } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { useCustomNavigation } from "utils/navigator";
import { ReactQueryKeys } from "utils/reactQueryKeys";
import { getScopedRoutes } from "utils/scopedRoutes";
import { formatToShortLocalDateString } from "utils/timeConverter";

import { useQueryOptions } from "../misc/queryOptions";

/**
 * This method calls graph list hooks from the generated code,
 * converts the results into a `JSGraph[]`, and returns it.
 *
 *
 * @param handleError
 * @param setPageLoading
 * @param setSelectedGraph
 * @returns `{ isDataLoading: boolean;
    graphList: GraphListResponse[];
    deleteGraph: (graphInfo: JSGraph) => void}
 */
export const useGraphs = ({
    enabled = true,
    projectId,
    handleError,
    setPageLoading,
    setSelectedGraph
}: {
    enabled?: boolean;
    projectId?: string | null;
    handleError?: HandleErrorFunc;
    setPageLoading?: (isLoading: boolean) => void;
    setSelectedGraph?: (response: GraphListResponse | null) => void;
}) => {
    const navigate = useCustomNavigation();
    const [graphList, setGraphList] = useState<Array<GraphListResponse>>([]);
    const queryClient = useQueryClient();
    const { getQueryOptions } = useQueryOptions();
    const [isGraphsPresent, setIsGraphsPresent] = useState<boolean>(false);

    const {
        data: graphListResponse,
        isLoading,
        error,
        isError
    } = useGetV1Graphs(
        { project_id: projectId ?? "" },
        getQueryOptions({
            queryKey: [ReactQueryKeys.GraphList, projectId ?? ""],
            enabled: enabled && Boolean(projectId)
        })
    );
    useEffect(() => {
        if (isError) {
            setPageLoading?.(false);
            handleError?.(error as unknown as CustomError);
        }
    }, [error, handleError, setPageLoading, isError]);

    /**
     * This method will be called once the graphs UI is mounted
     *
     * Restructure graphListResponse as UI needed!
     */
    useEffect(() => {
        if (isLoading) {
            return;
        }
        if (graphListResponse?.data && graphListResponse.data?.length > 0) {
            const transformed: GraphListResponse[] = graphListResponse.data.map(
                (graph) => ({
                    name: graph.name,
                    id: graph.id ?? graph.name,
                    timestamps: graph.timestamps as ResourceTimestamps,
                    dateCreated: graph.timestamps?.created_at
                        ? formatToShortLocalDateString(
                              graph.timestamps.created_at
                          )
                        : undefined,
                    num_tables: graph.num_tables ?? undefined,
                    num_links: graph.num_links ?? undefined,
                    created_by_name: graph.created_by_name
                })
            );

            setGraphList(transformed);
            setPageLoading?.(false);
            setIsGraphsPresent(transformed?.length > 0);
        } else {
            setPageLoading?.(false);
        }
    }, [graphListResponse?.data, queryClient, isLoading, setPageLoading]);

    const deleteGraphMutation = useDeleteV1GraphsByGraphAlias({});

    const deleteGraph = useCallback(
        (graphInfo: JSGraph) => {
            setPageLoading?.(true);
            const onError = (error: unknown) => {
                setPageLoading?.(false);
                const title = `Unable to delete graph ${graphInfo?.name}`;
                handleError?.(error as CustomError, title);
            };

            if (projectId) {
                deleteGraphMutation.mutate(
                    {
                        graphAlias: graphInfo.name ?? "",
                        params: { project_id: projectId }
                    },
                    { onError }
                );
            }
            setPageLoading?.(false);
            setSelectedGraph?.(null);

            // Remove deleted entity from GraphList
            setGraphList((prevGraphs) =>
                (prevGraphs as GraphListResponse[]).filter(
                    (graph) => graph.name !== graphInfo?.name
                )
            );
            // Remove deleted entity from `graphListResponse`
            // and update the cached data.
            if (graphListResponse?.data) {
                const updatedGraphs = graphListResponse.data.filter(
                    (graph) => graph.name !== graphInfo?.name
                );
                queryClient.setQueryData(
                    [ReactQueryKeys.GraphList, projectId ?? ""],
                    {
                        ...graphListResponse,
                        data: updatedGraphs
                    }
                );
            }

            navigate(getScopedRoutes(projectId).Graphs, { replace: true });
        },
        [
            setPageLoading,
            deleteGraphMutation,
            setSelectedGraph,
            graphListResponse,
            navigate,
            projectId,
            handleError,
            queryClient
        ]
    );
    return {
        isDataLoading: isLoading,
        graphList,
        deleteGraph,
        isGraphsPresent
    };
};

/**
 * Suggest links for the current project and graph definition.
 *
 * Only the latest invocation may merge groups or clear its loading state.
 * Responses for an older invocation, project, or graph definition are ignored
 * so they cannot overwrite newer graph edits or surface stale errors.
 */
export const useSuggestLinks = (
    graphFlowInfo: GraphFlowInfo,
    allTables: GraphTables,
    groups: Array<ColumnGroup>,
    setGroups: (g: Array<ColumnGroup>) => void,
    handleError: (err: Error, title?: string) => void,
    setLoading: (load: boolean) => void
): {
    suggestLinks: () => void;
} => {
    const { mutate: suggestLink } = useCreateV1GraphsSuggestLinks({});
    const projectId = useProjectId();
    const requestKey = useMemo(
        () => [projectId, graphFlowInfo] as const,
        [projectId, graphFlowInfo]
    );
    const currentRequestKey = useRef(requestKey);
    currentRequestKey.current = requestKey;
    const latestRequest = useRef<Record<never, never> | null>(null);

    useEffect(
        () => () => {
            latestRequest.current = null;
        },
        []
    );

    const suggestLinks = useCallback(async () => {
        const request = {};
        latestRequest.current = request;
        const isLatestRequest = () => request === latestRequest.current;
        const isCurrentRequest = () =>
            isLatestRequest() && requestKey === currentRequestKey.current;
        const finishRequest = () => {
            if (isLatestRequest()) {
                setLoading(false);
            }
        };
        try {
            setLoading(true);
            if (!projectId) {
                throw new Error("A project must be selected.");
            }
            const requestBody = createGraphBody(graphFlowInfo, []);
            if (getKeys(requestBody?.tables ?? {}).length <= 1) {
                finishRequest();
                return;
            }
            suggestLink(
                {
                    data: requestBody,
                    params: { project_id: projectId }
                },
                {
                    onSuccess: (response) => {
                        if (!isCurrentRequest()) {
                            finishRequest();
                            return;
                        }
                        if (response?.data) {
                            const suggested: Array<ColumnGroup> = response.data;
                            const grps: Array<ColumnGroup> = [...groups];
                            let updateGroups: boolean = false;
                            for (let i = 0; i < suggested.length; i++) {
                                const group: ColumnGroup = suggested[i];
                                let replaced: boolean = false;
                                // Set the group state
                                group.state = "saved";
                                group.updated = true;
                                // Check if the incoming suggested group is already present,
                                // and remove them from groups.
                                // Get the pkey of suggested group
                                const inputPkey = getpKeys(
                                    group.columns,
                                    allTables
                                );
                                if (inputPkey?.length > 0) {
                                    for (let j = 0; j < grps?.length; j++) {
                                        const pkeys = getpKeys(
                                            grps[j].columns,
                                            allTables
                                        );
                                        if (pkeys?.length > 0) {
                                            if (
                                                pkeys[0]?.table_name ===
                                                    inputPkey[0]?.table_name &&
                                                pkeys[0]?.col_name ===
                                                    inputPkey[0]?.col_name
                                            ) {
                                                // Match found. Remove the group and
                                                // replace with the new suggested link.
                                                grps[j] = group; //Replace
                                                replaced = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (!replaced) grps.push(group);
                                updateGroups = true;
                            }
                            if (updateGroups) setGroups(grps);
                        }
                        finishRequest();
                    },
                    onError: (error) => {
                        if (!isCurrentRequest()) {
                            finishRequest();
                            return;
                        }
                        if (
                            (error as { response?: { status?: number } })
                                ?.response?.status !==
                            errorStatusCode.InternalServerError
                        ) {
                            const title: string = "Suggest linkages failed!";
                            handleError(error as CustomError, title);
                        }
                        finishRequest();
                    }
                }
            );
        } catch (error) {
            if (
                (error as { response?: { status?: number } })?.response
                    ?.status !== errorStatusCode.InternalServerError
            ) {
                const title: string = "Suggest linkages failed!";
                handleError(error as CustomError, title);
            }
            finishRequest();
        }
    }, [
        setLoading,
        graphFlowInfo,
        suggestLink,
        groups,
        setGroups,
        allTables,
        handleError,
        projectId,
        requestKey
    ]);

    return { suggestLinks };
};
export const useGetGraphInfo = (
    graphId: string | null,
    handleError: (err: Error) => void,
    setLoading?: (load: boolean) => void,
    enabled: boolean = true
): {
    selectedGraphInfo: JSGraph | null;
    fetchGraphInfo: () => void;
    isLoading: boolean;
} => {
    const { medium } = useCacheStaleTime();
    const projectId = useProjectId();

    const {
        data: graphInfo,
        isLoading: isLoading,
        refetch: fetchGraphInfo,
        error,
        isError
    } = useGetV1GraphsByGraphIdOrAlias(
        graphId ?? "",
        { project_id: projectId ?? "" },
        {
            query: {
                queryKey: [ReactQueryKeys.GraphInfo, graphId],
                staleTime: medium,
                retry: false,
                enabled: enabled && isValid(graphId) && Boolean(projectId)
            }
        }
    );
    // Convert API response into your internal JSGraph
    const selectedGraphInfo: JSGraph | null = useMemo(() => {
        return graphInfo?.data
            ? jsGraphFromGraphResponse(graphInfo?.data, true)
            : null;
    }, [graphInfo?.data]);

    useEffect(() => {
        if (isError) {
            handleError(error as unknown as CustomError);
            setLoading?.(false);
        } else if (graphInfo?.data && !isLoading) {
            setLoading?.(false);
        }
    }, [error, graphInfo?.data, handleError, isError, isLoading, setLoading]);

    return { selectedGraphInfo, fetchGraphInfo, isLoading };
};
