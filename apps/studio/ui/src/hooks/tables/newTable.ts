// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from "react";

import { useQueryClient } from "@tanstack/react-query";
import {
    CreateV1TablesParams,
    KumoapiTableTableDefinition,
    ValidationResponse
} from "gen/v1/types";

import { useProjectId } from "hooks/project/projects";
import { useCreateTable } from "hooks/tables/tables";

import { CustomError } from "types/common";
import { GraphTables } from "types/graphs";
import { FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { getProjectPath, storageKeys } from "utils/Constants";
import { isValid } from "utils/Helpers/typeGuards";
import { useCustomNavigation } from "utils/navigator";
import { invalidateTablesList } from "utils/queryInvalidation";
import { removeFromSessionStorage } from "utils/storage";
import {
    applyKeyOverrides,
    applyS3Overrides,
    buildCustomSqlTableArgs,
    buildValidationError,
    S3SourceOverrideContext
} from "utils/tables";

/**
 * Volatile form/flow state read by `submitRequest` at call time. Passed in on
 * every call (rather than closed over) so the returned callback stays stable.
 */
export interface SubmitTableContext extends S3SourceOverrideContext {
    tableArgs: KumoapiTableTableDefinition;
    tableName: string;
}

/**
 * Volatile state read by `submitSqlTable`. The SQL draft and connector are
 * lifted from the authoring surface so the create request can be built without
 * re-reading the (source-table-shaped) form state.
 */
export interface SubmitSqlTableContext {
    tableArgs: KumoapiTableTableDefinition;
    tableName: string;
    connectorId: string;
    sql: string;
    // Edit reuses the create mutation with `force_rename` so the server
    // overwrites the existing table (bumping its data_version) instead of
    // rejecting the name; create/duplicate mint a new table.
    flowType: FlowType;
}

/**
 * Stable collaborators wired once by the owning component.
 */
export interface UseSubmitTableConfig {
    handleError: HandleErrorFunc;
    addTable?: (table: GraphTables) => void;
    refetchTable?: (id: string) => void;
    setLoading: (v: boolean) => void;
    setAlertPkey: (v: boolean) => void;
    setCancelFetchSample: (v: boolean) => void;
}

/**
 * Encapsulates the "Add / Save table" orchestration: validate the table
 * definition, then either hand it to `addTable` (graph-builder flow) or create
 * it via the API and navigate to / refetch the result.
 */
export const useSubmitTable = ({
    handleError,
    addTable,
    refetchTable,
    setLoading,
    setAlertPkey,
    setCancelFetchSample
}: UseSubmitTableConfig) => {
    const { createTable, validateTables } = useCreateTable(handleError);
    const queryClient = useQueryClient();
    const navigate = useCustomNavigation();
    const projectId = useProjectId();
    const projectPaths = getProjectPath(projectId);

    const submitRequest = useCallback(
        async (ctx: SubmitTableContext, timeCol?: string, pkey?: string) => {
            const { tableArgs, tableName, flowType } = ctx;
            setAlertPkey(false);
            setCancelFetchSample(true);
            setLoading(true);
            const tmpArgs = { ...tableArgs };
            applyKeyOverrides(tmpArgs, timeCol, pkey);
            applyS3Overrides(tmpArgs, ctx);
            try {
                if (!projectId) {
                    throw new Error("A project must be selected.");
                }
                const validationResponse = await validateTables(tmpArgs);
                if (validationResponse?.data) {
                    const response: ValidationResponse =
                        validationResponse.data;

                    if (response?.errors && response.errors.length < 1) {
                        if (addTable) {
                            setLoading(false);
                            addTable({
                                [tableName]: tmpArgs
                            } as unknown as GraphTables);
                        } else {
                            const tableParams: CreateV1TablesParams = {
                                project_id: projectId,
                                name_alias: tableName,
                                force_rename: flowType === FlowType.EDIT
                            };
                            const createResponse = await createTable(
                                tmpArgs,
                                tableParams
                            );
                            invalidateTablesList(queryClient);
                            setLoading(false);
                            if (createResponse?.data) {
                                const id = createResponse.data.id;
                                if (refetchTable) {
                                    refetchTable(tableName);
                                } else if (isValid(id)) {
                                    removeFromSessionStorage(
                                        storageKeys.isLocalUpload
                                    );
                                    navigate(projectPaths.Tables + tableName);
                                }
                            }
                        }
                    } else {
                        handleError(
                            buildValidationError(response),
                            "Table Validation failed!"
                        );
                    }
                }
            } catch (error) {
                const title = "Failed to create table!";
                handleError(error as CustomError, title);
                setLoading(false);
            }
        },
        [
            validateTables,
            addTable,
            createTable,
            queryClient,
            refetchTable,
            navigate,
            projectPaths.Tables,
            projectId,
            handleError,
            setAlertPkey,
            setCancelFetchSample,
            setLoading
        ]
    );

    /**
     * Create a Custom SQL table from an already-validated authoring surface.
     *
     * Unlike `submitRequest`, this skips the front-end `/validate` call: that
     * endpoint probes a physical source table, which a SQL definition does not
     * have. The definition was already validated in the editor, and the create
     * path re-validates the SELECT server-side (routing it to the Custom SQL
     * activity), so the only client work is to build the `CustomSQLSourceTable`
     * definition and hand it to the same project-scoped create mutation raw
     * tables use. A create-time validation failure surfaces via `handleError`.
     */
    const submitSqlTable = useCallback(
        async (ctx: SubmitSqlTableContext) => {
            const { tableArgs, tableName, connectorId, sql, flowType } = ctx;
            setCancelFetchSample(true);
            setLoading(true);
            const sqlArgs = buildCustomSqlTableArgs(
                tableArgs,
                connectorId,
                sql,
                tableName
            );
            try {
                if (!projectId) {
                    throw new Error("A project must be selected.");
                }
                if (addTable) {
                    setLoading(false);
                    addTable({
                        [tableName]: sqlArgs
                    } as unknown as GraphTables);
                    return;
                }
                const tableParams: CreateV1TablesParams = {
                    project_id: projectId,
                    name_alias: tableName,
                    force_rename: flowType === FlowType.EDIT
                };
                const createResponse = await createTable(sqlArgs, tableParams);
                invalidateTablesList(queryClient);
                setLoading(false);
                if (createResponse?.data) {
                    const id = createResponse.data.id;
                    if (refetchTable) {
                        refetchTable(tableName);
                    } else if (isValid(id)) {
                        navigate(projectPaths.Tables + tableName);
                    }
                }
            } catch (error) {
                handleError(error as CustomError, "Failed to create table!");
                setLoading(false);
            }
        },
        [
            addTable,
            createTable,
            queryClient,
            refetchTable,
            navigate,
            projectPaths.Tables,
            projectId,
            handleError,
            setCancelFetchSample,
            setLoading
        ]
    );

    return { submitRequest, submitSqlTable };
};
