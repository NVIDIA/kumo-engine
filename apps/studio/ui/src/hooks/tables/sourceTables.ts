// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from "react";

import { useIsMounted } from "@kumo-ai/kumo-ui-catalog";
import { useQueryClient } from "@tanstack/react-query";
import axios from "axios";
import {
    getProjectsSourceTableConfigByConnectorId,
    getProjectsSourceTablesByConnectorId
} from "gen/v0/api/connectors/connectors";
import { SourceTableListResponse } from "gen/v0/types";
import { KumoapiSourceTableFileType } from "gen/v1/types";
import { useCacheStaleTime } from "stores/cacheStore";

import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";
import { ConnectorDropdownOption, TableDropdownOption } from "types/inputs";
import { FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { isAnyUriObjectStoreConnector } from "utils/objectStoreTablePath";
import { ReactQueryKeys } from "utils/reactQueryKeys";

export interface UseSourceTablesProps {
    selectedConnector: ConnectorDropdownOption;
    flowType: FlowType;
    handleError: HandleErrorFunc;
    setLoading: (v: boolean) => void;
    connectorId?: string;
    sourceTableName?: string;
    resetTableSelection: () => void;
    clearErrors?: () => void;
    reloadSignal?: number;
}

/** Resolve an exact object-store coordinate and its server-detected type. */
export const getObjectStoreSourceTableConfig = (
    projectId: string,
    connectorId: string,
    tableName: string
) =>
    getProjectsSourceTableConfigByConnectorId(projectId, connectorId, {
        table_name: tableName
    });

export const useSourceTables = ({
    selectedConnector,
    flowType,
    handleError,
    setLoading,
    connectorId,
    sourceTableName,
    reloadSignal
}: UseSourceTablesProps) => {
    const projectId = useProjectId();
    const queryClient = useQueryClient();
    const { medium } = useCacheStaleTime();
    const isMounted = useIsMounted();
    const abortControllerRef = useRef<AbortController | null>(null);
    const requestSequence = useRef(0);

    const [fetchSourceTables, setFetchSourceTables] = useState(false);
    const [sourceTables, setSourceTables] = useState<TableDropdownOption[]>(
        sourceTableName && flowType === FlowType.EDIT
            ? [{ value: sourceTableName, label: sourceTableName }]
            : []
    );

    const fetchTables = useCallback(async () => {
        abortControllerRef.current?.abort();
        if (!projectId || !connectorId) {
            setSourceTables([]);
            setFetchSourceTables(false);
            setLoading(false);
            return;
        }
        if (isAnyUriObjectStoreConnector(selectedConnector?.config)) {
            setSourceTables([]);
            setFetchSourceTables(false);
            setLoading(false);
            return;
        }

        const sequence = ++requestSequence.current;
        const controller = new AbortController();
        abortControllerRef.current = controller;
        const cacheKey = [ReactQueryKeys.SourceTable, projectId, connectorId];
        setFetchSourceTables(true);

        try {
            let data =
                queryClient.getQueryData<SourceTableListResponse>(cacheKey);
            if (!data || reloadSignal !== undefined) {
                const response = await getProjectsSourceTablesByConnectorId(
                    projectId,
                    connectorId,
                    undefined,
                    controller.signal
                );
                data = response.data;
                queryClient.setQueryDefaults(cacheKey, {
                    staleTime: medium,
                    gcTime: medium * 2
                });
                queryClient.setQueryData(cacheKey, data);
            }
            if (!isMounted.current || sequence !== requestSequence.current) {
                return;
            }
            setSourceTables(
                (data.table_names ?? []).map((name) => ({
                    value: name,
                    label: name,
                    valid: true,
                    connectorId,
                    fileType: name.toLowerCase().endsWith(".csv")
                        ? KumoapiSourceTableFileType.CSV
                        : KumoapiSourceTableFileType.PARQUET
                }))
            );
        } catch (error) {
            if (!axios.isCancel(error) && isMounted.current) {
                handleError(error as CustomError);
            }
        } finally {
            if (isMounted.current && sequence === requestSequence.current) {
                setFetchSourceTables(false);
                setLoading(false);
            }
        }
    }, [
        connectorId,
        handleError,
        isMounted,
        medium,
        projectId,
        queryClient,
        reloadSignal,
        selectedConnector?.config,
        setLoading
    ]);

    useEffect(() => {
        if (flowType === FlowType.CREATE) {
            void fetchTables();
        }
        return () => abortControllerRef.current?.abort();
    }, [fetchTables, flowType]);

    return { sourceTables, fetchSourceTables };
};
