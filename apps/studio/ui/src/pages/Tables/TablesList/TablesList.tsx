// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useMemo, useState } from "react";

import {
    AddCircle,
    Button,
    DeleteIcon,
    Duplicate,
    Edit,
    Replay,
    Select,
    useIsMounted
} from "@kumo-ai/kumo-ui-catalog";
import { useGetProjectsConnectorByProjectId } from "gen/v0/api/connectors/connectors";
import {
    Column,
    KumoapiTableTableDefinition,
    KumoapiTableTableDefinitionSourceTable
} from "gen/v1/types";

import EmptyEntities from "components/Empty/EmptyEntities";
import ListOverview from "components/ListOverview/ListOverview";
import TableOverview from "components/Tables/Overview/TableOverview";
import Validation from "components/Validation/Validation";

import { useConnectorDetails } from "hooks/connector/connector";
import { useProjectId } from "hooks/project/projects";
import {
    useColumnStatsInfo,
    useGetTableInfo,
    useTables
} from "hooks/tables/tables";
import { useValidationState } from "hooks/validation/validation";

import { MenuItemType } from "types/common";
import { FlowType, TableKind, TableListResponse } from "types/tables";

import {
    ENABLE_CUSTOM_SQL_TABLES,
    getProjectPath,
    PageNames
} from "utils/Constants";
import { isValid } from "utils/Helpers/typeGuards";
import { useCustomNavigation } from "utils/navigator";

/**
 * Tables List Component
 *
 * Fetches list of connected Tables from all the connectors
 * and renders it in UI
 *
 */
export const resolveTableConnectorDetails = (
    connectorId: string,
    projectUploadConnectorId: string
) => {
    const isProjectUploadConnector =
        connectorId !== "" && connectorId === projectUploadConnectorId;
    return {
        isProjectUploadConnector,
        connectorDetailsId:
            projectUploadConnectorId && !isProjectUploadConnector
                ? connectorId
                : ""
    };
};

const TablesList: React.FC = () => {
    const projectId = useProjectId();

    const [isPageLoading, setPageLoading] = useState(true);
    const isMounted = useIsMounted();

    const [banner, alert, handleError, clearErrors, dialogRef] =
        useValidationState(setPageLoading);

    const [selectedTable, setSelectedTable] =
        useState<TableListResponse | null>(null);

    const {
        isDataLoading,
        connectorPresent,
        tableList,
        deleteTable,
        refreshTable
    } = useTables(handleError, setPageLoading, true, projectId as string);

    const projectPaths = getProjectPath(projectId);

    // Raw/SQL kind filter (default: All). Filtering happens here so the shared
    // ListOverview stays entity-agnostic; the control is passed down as a slot.
    const [kindFilter, setKindFilter] = useState<string>("all");
    const kindFilterOptions = [
        { value: "all", label: "All tables" },
        { value: TableKind.Raw, label: "Raw" },
        { value: TableKind.Sql, label: "Custom SQL" }
    ];
    const visibleTables = useMemo(
        () =>
            kindFilter === "all"
                ? tableList
                : tableList.filter(
                      (t) => (t.table_kind ?? TableKind.Raw) === kindFilter
                  ),
        [tableList, kindFilter]
    );

    // Drop the selection (and its detail pane) when the filter hides it.
    useEffect(() => {
        if (
            selectedTable &&
            !visibleTables.some((t) => t.id === selectedTable.id)
        ) {
            setSelectedTable(null);
        }
    }, [visibleTables, selectedTable]);

    const { columnStats: columnStatsData, loading: fetchingSamplesData } =
        useColumnStatsInfo(
            selectedTable?.id ?? null,
            handleError,
            isMounted,
            isValid(selectedTable?.id)
        );

    const navigate = useCustomNavigation();

    const { selectedTableInfo, loading } = useGetTableInfo(
        selectedTable?.id as string,
        handleError,
        selectedTable != null
    );

    const { data: uploadConnectorResponse } =
        useGetProjectsConnectorByProjectId(projectId ?? "", {
            query: { enabled: Boolean(projectId) }
        });
    const selectedConnectorId = selectedTableInfo?.connectorId ?? "";
    const { connectorDetailsId, isProjectUploadConnector } =
        resolveTableConnectorDetails(
            selectedConnectorId,
            uploadConnectorResponse?.data?.connector_id ?? ""
        );
    const { connectorResponse: connectorConfig } = useConnectorDetails(
        connectorDetailsId,
        handleError
    );

    const buttons = [
        <Button
            id={"create-new-table"}
            onClick={() => {
                navigate(projectPaths.NewTable);
            }}>
            <AddCircle /> Add table
        </Button>
    ];

    const seletedTableArgs: KumoapiTableTableDefinition = {
        cols: selectedTableInfo?.table?.cols as Column[],
        source_table: selectedTableInfo?.table
            .source_table as KumoapiTableTableDefinitionSourceTable,
        pkey: selectedTableInfo?.table?.pkey
            ? selectedTableInfo?.table?.pkey
            : "",
        time_col: selectedTableInfo?.table?.time_col
            ? selectedTableInfo?.table?.time_col
            : "",
        end_time_col: selectedTableInfo?.table?.end_time_col
            ? selectedTableInfo?.table?.end_time_col
            : ""
    };

    const onDeleteConfirm = useCallback(() => {
        dialogRef?.current?.toggleDialog();
        clearErrors();
        if (selectedTableInfo?.name_alias) {
            deleteTable(
                selectedTableInfo.id,
                selectedTableInfo.name_alias,
                setSelectedTable
            );
        }
    }, [
        dialogRef,
        selectedTableInfo?.id,
        selectedTableInfo?.name_alias,
        deleteTable,
        clearErrors
    ]);

    const onDeleteClick = () => {
        dialogRef?.current?.toggleDialog(
            "Delete Table",
            `Are you sure you want to delete table "${selectedTableInfo?.name_alias}"? This action cannot be undone.`,
            {},
            false,
            "Delete",
            "destructive"
        );
    };

    const onSummaryRefreshClick = useCallback(async () => {
        if (isValid(selectedTable?.id) && selectedTable?.id) {
            refreshTable(selectedTable.id as string);
        }
    }, [selectedTable?.id, refreshTable]);

    const onCopyTableClick = () => {
        const copyTableArgs: KumoapiTableTableDefinition = seletedTableArgs;
        navigate(projectPaths.NewTable, {
            state: {
                copytableargs: copyTableArgs,
                flowtype: FlowType.DUPLICATE,
                tableName: selectedTableInfo?.name_alias + "_copy"
            }
        });
    };

    const onTableEditClick = () => {
        const editTableArgs: KumoapiTableTableDefinition = seletedTableArgs;
        navigate(projectPaths.EditTable, {
            state: {
                createtableargs: editTableArgs,
                flowtype: FlowType.EDIT,
                tableName: selectedTableInfo?.name_alias,
                s3ConnctorConfig: selectedTableInfo?.s3Connector,
                islocal: false
            }
        });
    };

    const menuItems: MenuItemType[] = [
        {
            content: (
                <>
                    <Edit className="size-4" /> {" Edit "}
                </>
            ),
            disabled: false,
            onSelect: () => {
                onTableEditClick();
            },
            tooltipText: ""
        },
        {
            content: (
                <>
                    <DeleteIcon className="size-4" /> {" Delete "}
                </>
            ),
            disabled: false,
            onSelect: () => {
                onDeleteClick();
            },
            tooltipText: ""
        },
        {
            content: (
                <>
                    <Duplicate className="size-4 fill-icon-secondary" />
                    {" Duplicate "}
                </>
            ),
            disabled: false,
            onSelect: () => {
                onCopyTableClick();
            },
            tooltipText: ""
        },
        {
            content: (
                <>
                    <Replay className="size-4" /> {" Re-ingest "}
                </>
            ),
            disabled: false,
            onSelect: () => {
                onSummaryRefreshClick();
            },
            tooltipText: ""
        }
    ];

    // Combine all loading state
    const isLoading = isDataLoading || isPageLoading;

    if (!isLoading && tableList.length === 0) {
        return (
            <EmptyEntities
                currentPage={PageNames.Table}
                title={PageNames.Table}
            />
        );
    }

    return (
        <ListOverview
            title={"tables"}
            data={ENABLE_CUSTOM_SQL_TABLES ? visibleTables : tableList}
            loadingStatus={isLoading}
            entityPresent={connectorPresent}
            filterControl={
                ENABLE_CUSTOM_SQL_TABLES ? (
                    <Select
                        id="table-kind-filter"
                        options={kindFilterOptions}
                        sortOptions={false}
                        selectedValues={[kindFilter]}
                        onValuesChange={(values: string[]) =>
                            setKindFilter(values[0] ?? "all")
                        }
                        search={false}
                        mode="single"
                        variant="secondary"
                        size="sm"
                        popoverTriggerClassName="flex w-44 h-9 py-2 px-4 items-center gap-2"
                        isInValid={false}
                        selectedOptionPrefix="Kind:"
                    />
                ) : undefined
            }
            buttons={!projectId ? [] : buttons}
            selectedEntity={selectedTable}
            setSelectedEntity={(table) => {
                clearErrors();
                setSelectedTable(table);
            }}
            overviewComponent={
                <>
                    <Validation
                        pageLoading={isLoading}
                        banner={banner}
                        alert={alert}
                        ctaMsgs={null}
                        dialogRef={dialogRef}
                        onDialogConfirm={onDeleteConfirm}
                        hideWarnings={() => {}}
                        clearErrors={clearErrors}
                    />
                    <TableOverview
                        loadingStatus={loading}
                        tableInfo={selectedTableInfo}
                        actionItems={menuItems}
                        columnStats={columnStatsData}
                        fetchingSamples={fetchingSamplesData}
                        connectorConfig={connectorConfig}
                        isProjectUploadConnector={isProjectUploadConnector}
                    />
                </>
            }
        />
    );
};

export default TablesList;
