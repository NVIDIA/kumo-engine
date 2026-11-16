// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useState } from "react";

import {
    Card,
    ConnectorChip,
    GridCol,
    GridRow,
    Skeleton,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import { OverviewDropdown } from "@kumo-ai/kumo-ui-catalog";
import { PropertiesOverviewCard } from "@kumo-ai/kumo-ui-catalog";
import {
    KumoapiTableTableDefinition,
    KumoapiTableTableDefinitionSourceTable
} from "gen/v1/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import TableView from "components/@ui/TablePlus/ConnectedTables";
import { useConnectorProperties } from "components/Overview/JobProperties";
import Validation from "components/Validation/Validation";

import { useConnectedTables } from "hooks/connector/connector";
import { useValidationState } from "hooks/validation/validation";

import { MenuItemType } from "types/common";
import { JsConnector } from "types/connectors";
import { FlowType, JSTable } from "types/tables";

import { getProjectPath } from "utils/Constants";
import { getSourceInfoFromConnectorConfig } from "utils/Converters/tablesConverter";
import { useCustomNavigation } from "utils/navigator";

export interface OverviewProps {
    actionItems: MenuItemType[];
    connectorInfo: JsConnector | null;
    loadingStatus: boolean;
    onFetchConnectedTables?: (tables: JSTable[]) => void;
    isProjectView?: boolean;
}

const Overview = ({
    actionItems,
    connectorInfo,
    onFetchConnectedTables,
    isProjectView = false
}: OverviewProps) => {
    const [loading, setLoading] = useState(true);
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [banner, alert, handleError, clearErrors, dialogRef, openDialog] =
        useValidationState(setLoading);

    const { isLoading, tables, deleteTable } = useConnectedTables(
        connectorInfo?.id as string,
        handleError,
        setLoading,
        onFetchConnectedTables
    );
    const navigate = useCustomNavigation();

    const overviewProperty = useConnectorProperties({
        info: connectorInfo,
        isProjectView: isProjectView
    });

    // Clear errors when use changes the Connector
    useEffect(() => {
        clearErrors();
    }, [clearErrors, connectorInfo?.id]);

    const onConfirmDelete = useCallback(
        (value?: { index?: number; isConnector?: boolean; data?: JSTable }) => {
            dialogRef.current?.toggleDialog();
            if (value?.isConnector !== true) {
                deleteTable(value?.data?.name_alias as string);
            }
        },
        [deleteTable, dialogRef]
    );

    const onConnectTableClick = () => {
        if (connectorInfo) {
            const tableArgs: KumoapiTableTableDefinition = {
                cols: [],
                source_table: getSourceInfoFromConnectorConfig(
                    connectorInfo.config,
                    connectorInfo.id
                ) as KumoapiTableTableDefinitionSourceTable,
                pkey: "",
                time_col: "",
                end_time_col: ""
            };
            navigate(getProjectPath(connectorInfo.projectId).NewTable, {
                state: {
                    createtableargs: tableArgs,
                    flowtype: FlowType.CREATE,
                    tableName: "",
                    islocal: false
                }
            });
        }
    };
    if (!connectorInfo) {
        return null;
    }

    return (
        <>
            <Validation
                data-testid="validate-delete"
                pageLoading={false}
                banner={banner}
                alert={alert}
                ctaMsgs={null}
                dialogRef={dialogRef}
                onDialogConfirm={onConfirmDelete}
                hideWarnings={() => {}}
                clearErrors={clearErrors}
            />
            <GridRow
                xs={12}
                sm={12}
                md={12}
                className="flex justify-between items-center mb-0">
                <GridCol xs={12} sm={12} md="auto" className="min-w-0">
                    <label data-testid="header" className="block min-w-0">
                        {!loading ? (
                            <div className="flex min-w-0 gap-4 items-center">
                                <Title
                                    variant="primary-heading"
                                    className="min-w-0 flex-1 text-title-primary !leading-5 font-bold text-xl">
                                    <span
                                        className="block truncate"
                                        title={connectorInfo.name}>
                                        {connectorInfo.name}
                                    </span>
                                </Title>
                                <ConnectorChip
                                    type={connectorInfo.config?.type as string}
                                />
                            </div>
                        ) : (
                            <Skeleton className="w-60 h-8" />
                        )}
                    </label>
                </GridCol>
                {loading ? null : (
                    <GridCol xs={12} sm={12} md="auto">
                        <OverviewDropdown
                            id="connector-overview-dropdown"
                            actionItems={actionItems}
                        />
                    </GridCol>
                )}
            </GridRow>
            {loading ? (
                <Skeleton className="h-7" />
            ) : (
                <div data-testid="connector-overview-properties">
                    <PropertiesOverviewCard
                        properties={overviewProperty}
                        cardClassName={"mt-2 mb-2 p-4"}
                        hideTitle
                    />
                </div>
            )}
            <Card className="p-4 border-none gap-4 rounded-lg h-full">
                <TableView
                    tables={tables}
                    fetchingTables={isLoading || loading}
                    onConnectTableClick={onConnectTableClick}
                    dialogRef={dialogRef}
                    disableAdd={false}
                    projectId={connectorInfo.projectId}
                />
            </Card>
        </>
    );
};

export default WithErrorBoundary(Overview);
