// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef } from "react";

import { useQueryClient } from "@tanstack/react-query";
import {
    useCreateV1Graphs,
    useValidateV1Graphs
} from "gen/v1/api/graphs/graphs";
import {
    KumoapiGraphGraphDefinitionTables as GraphDefinitionTables,
    TableResourceResponse
} from "gen/v1/types";

import { useProjectId } from "hooks/project/projects";
import { useGetMultipleTableInfo } from "hooks/tables/tables";

import { CustomError } from "types/common";
import { ColumnGroup, GraphFlowInfo, GraphTables } from "types/graphs";
import { FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { errorStatusCode } from "utils/Constants";
import { parseErrorMessage, prepareCustomError } from "utils/errorHandler";
import { createGraphBody } from "utils/graph";
import { getKeys } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { invalidateGraphsList } from "utils/queryInvalidation";

export const useGraphValidation = () => {
    const { mutateAsync: validateGraph } = useValidateV1Graphs();
    const projectId = useProjectId();

    const validateGraphConfig = useCallback(
        async (
            graphFlowInfo: GraphFlowInfo,
            groups: ColumnGroup[]
        ): Promise<{ errors: string[]; statusCode: errorStatusCode }> => {
            let errors: string[] = [];
            let statusCode = errorStatusCode.Success;

            const requestBody = createGraphBody(graphFlowInfo, groups);
            try {
                if (!projectId) {
                    throw new Error("A project must be selected.");
                }
                const response = await validateGraph({
                    data: requestBody,
                    params: { project_id: projectId }
                });
                if (response?.data?.errors?.length) {
                    statusCode = errorStatusCode.UnProcessableEntityError;
                    errors = response?.data?.errors
                        ?.map((e) => e.message)
                        .filter((e) => e !== null && e !== undefined);
                }
            } catch (err) {
                errors = [parseErrorMessage(err as CustomError)];
                statusCode = (err as { response: { status: errorStatusCode } })
                    ?.response?.status;
            }
            return { errors, statusCode };
        },
        [projectId, validateGraph]
    );

    return { validateGraph: validateGraphConfig };
};

export const useCreateGraph = (invalidateCache: boolean = true) => {
    const queryClient = useQueryClient();
    const { mutateAsync: createGraph } = useCreateV1Graphs();
    const selectedProject = useProjectId();
    const createNewGraph = useCallback(
        async (
            graphFlowInfo: GraphFlowInfo,
            groups: ColumnGroup[],
            handleError?: HandleErrorFunc,
            isEdit: boolean = false
        ): Promise<{
            graphId: string | null;
            error: string;
            statusCode: errorStatusCode;
        }> => {
            let graphId: string | null = null;
            let error: string = "";
            let statusCode = errorStatusCode.Success;
            const requestBody = createGraphBody(graphFlowInfo, groups);

            try {
                if (!selectedProject) {
                    throw new Error("A project must be selected.");
                }
                const response = await createGraph({
                    data: requestBody,
                    params: {
                        name_alias: graphFlowInfo.name?.trim(),
                        force_rename: isEdit,
                        project_id: selectedProject
                    }
                });
                if (response?.data) {
                    graphId = response.data.id;
                    if (invalidateCache) {
                        invalidateGraphsList(
                            queryClient,
                            isEdit ? graphFlowInfo.name?.trim() : undefined
                        );
                    }
                }
            } catch (e) {
                statusCode = (e as { response: { status: errorStatusCode } })
                    ?.response?.status;
                error = parseErrorMessage(e as CustomError);
                const err = prepareCustomError(error, statusCode);
                handleError?.(err, "Graph Creation failed!");
            }

            return { graphId, error, statusCode };
        },
        [createGraph, invalidateCache, queryClient, selectedProject]
    );

    return { createGraph: createNewGraph };
};

export const useTableSelectionForGraphs = (
    graphFlowInfo: GraphFlowInfo,
    allTables: GraphTables,
    setAllTables: React.Dispatch<React.SetStateAction<GraphTables>>,
    setGraphFlowInfo: React.Dispatch<React.SetStateAction<GraphFlowInfo>>,
    updateGroups: (config: GraphFlowInfo) => void,
    handleError: HandleErrorFunc,
    tableIdMap: { [key: string]: string },
    suggestLinks: () => void,
    setLoading?: (loading: boolean) => void,
    projectId?: string | null
) => {
    //Used to call suggestLinks
    const suggestLinksRef = useRef(false);

    const { getTables } = useGetMultipleTableInfo(projectId);

    const updateGfcAndGroupsWithTableInfo = useCallback(
        (responses: {
            tableNames: string[];
            data: (TableResourceResponse | null)[];
        }) => {
            try {
                //Add this table to allTables.
                const allTbls: GraphTables = { ...allTables };
                //Add this table to graph flow info.
                const config: GraphFlowInfo = { ...graphFlowInfo };
                const tablesNotFound: string[] = [];
                let updateAllTable = false;
                responses?.data.forEach((response, i) => {
                    // Using names from `responses.tableNames` instead of table response
                    // is because, incase if the API fails, we have show alert msg with name
                    const name = responses.tableNames[i];
                    if (response) {
                        const data = { ...response };
                        if (allTbls && !(name in allTables)) {
                            allTbls[name] = data?.table;
                            updateAllTable = true;
                        }
                        if (config && name) {
                            config.tables[name] = data?.table;
                        }
                    } else {
                        tablesNotFound.push(name);
                    }
                    config.selected_tables.push(name);
                });

                if (updateAllTable) {
                    setAllTables(allTbls);
                }
                if (config.flow_type === FlowType.CREATE) {
                    suggestLinksRef.current = true;
                } else {
                    setLoading?.(false);
                }

                setGraphFlowInfo(config);
                updateGroups(config);
                if (tablesNotFound?.length > 0) {
                    const err = `Details for table(s) ${tablesNotFound.join(", ")} not found`;
                    const error: CustomError = {
                        response: {
                            data: err
                        },
                        name: "",
                        message: err
                    };
                    handleError(error, "Table(s) not found!");
                }
            } catch (error) {
                console.warn(error);
            }
        },
        [
            allTables,
            graphFlowInfo,
            handleError,
            setAllTables,
            setGraphFlowInfo,
            setLoading,
            updateGroups
        ]
    );

    //Gets a specified table
    const onGetTables = useCallback(
        async (tableNames: string[]) => {
            setLoading?.(true);
            getTables(
                {
                    tableIds: tableNames.map((name) => `${tableIdMap[name]}`),
                    tableNames
                },
                updateGfcAndGroupsWithTableInfo
            );
        },
        [getTables, tableIdMap, setLoading, updateGfcAndGroupsWithTableInfo]
    );

    const onSelectTables = useCallback(
        (selectedTables: string[], unselectedTables: string[]) => {
            const config: GraphFlowInfo = { ...graphFlowInfo };
            const configSelTables: Array<string | null> = [
                ...config.selected_tables
            ];
            let updateUI = false;
            const newTables: string[] = [];
            //First add selected rows to graphConfig if not already added.
            if (selectedTables?.length > 0 && config) {
                selectedTables.forEach((t) => {
                    //Add if NOT present in graphConfig
                    if (!configSelTables.includes(t)) {
                        //If table details present in allTables , get it from there
                        if (t in allTables) {
                            //Get table from allTables
                            config.tables[t] = allTables[t];
                            config.selected_tables.push(t);
                            updateUI = true;
                        } else {
                            //Gets and adds to config and allTables
                            newTables.push(t);
                        }
                    }
                });
            }

            //Next deselect rows from graphConfig if any.
            if (unselectedTables?.length > 0 && config) {
                unselectedTables.forEach((t) => {
                    //Remove if present in graphConfig
                    if (configSelTables.includes(t)) {
                        const updated = Object.fromEntries(
                            Object.entries(config.tables).filter(
                                ([key]) => key !== t
                            )
                        );
                        config.tables = updated;
                        config.selected_tables.splice(
                            config.selected_tables.indexOf(t),
                            1
                        );
                        updateUI = true;
                    }
                });
            }
            if (updateUI) {
                if (config.flow_type === FlowType.CREATE) {
                    suggestLinksRef.current = true;
                }
                setGraphFlowInfo(config);
                updateGroups(config);
            }
            // Get all New tables data
            if (newTables?.length > 0) {
                onGetTables(newTables);
            }
        },
        [graphFlowInfo, allTables, setGraphFlowInfo, updateGroups, onGetTables]
    );
    useEffect(() => {
        if (
            graphFlowInfo?.flow_type === FlowType.CREATE &&
            suggestLinksRef.current
        ) {
            suggestLinksRef.current = false;
            suggestLinks();
        }
    }, [graphFlowInfo, suggestLinks]);

    return { onSelectTables };
};

export const useBackwardLinks = (
    tables?: GraphDefinitionTables,
    groups?: ColumnGroup[]
) => {
    const { backwardLinks, isPkeyExists } = useMemo(() => {
        const bcklinks: {
            [key: string]: {
                tableName: string;
                columnName: string;
                pkeyName: string;
            }[];
        } = {};

        if (!tables) {
            return { backwardLinks: bcklinks, isPkeyExists: true };
        }

        // If there is no tables selected, set to true to avoid warning!
        let pkeyExists = getKeys(tables)?.length < 1;
        getKeys(tables).forEach((table) => {
            if (!isValid(tables[table]?.pkey)) {
                bcklinks[table] = [];
            } else {
                pkeyExists = true;
            }
        });
        groups?.forEach((g) => {
            const pkeyTable = g?.columns?.find(
                (c) => c?.table_name && isValid(tables[c?.table_name]?.pkey)
            );
            g?.columns?.forEach((c) => {
                const pkey = c?.table_name ? tables[c?.table_name]?.pkey : "";
                if (pkey !== c?.col_name) {
                    if (!bcklinks[c?.table_name]) {
                        bcklinks[c?.table_name] = [];
                    }
                    bcklinks[c?.table_name].push({
                        tableName: pkeyTable?.table_name || "",
                        columnName: c?.col_name,
                        pkeyName: pkeyTable?.col_name || ""
                    });
                }
            });
        });
        return { backwardLinks: bcklinks, isPkeyExists: pkeyExists };
    }, [tables, groups]);

    return { backwardLinks, isPkeyExists };
};
