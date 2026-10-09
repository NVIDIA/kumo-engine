// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useCallback } from "react";

import {
    AddCircle,
    Button,
    DataTable,
    EmptyTable,
    GridCol,
    GridRow,
    SectionHeader
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { DialogRef } from "types/common";
import { JSTable } from "types/tables";

import { ConnectedTablesColumns } from "../TableColumns/ConnectedTables";

export interface ConnectedTablesProps<TObject extends object> {
    tables: Array<JSTable> | [];
    onConnectTableClick: () => void;
    dialogRef: RefObject<DialogRef<TObject>>;
    fetchingTables: boolean;
    projectId: string;
    disableAdd?: boolean;
}

const ConnectedTables = <TObject extends object>({
    tables,
    onConnectTableClick,
    dialogRef,
    fetchingTables,
    projectId,
    disableAdd = false
}: ConnectedTablesProps<TObject>) => {
    const onTableDeleteClick = useCallback(
        (rowData: JSTable) => {
            const data = rowData;
            if (rowData) {
                dialogRef.current?.toggleDialog(
                    "Delete Table",
                    `Are you sure you want to delete table "${data?.name_alias}"? This action cannot be undone.`,
                    { data: data, isConnector: false } as TObject,
                    false,
                    "Delete",
                    "destructive"
                );
            }
        },
        [dialogRef]
    );

    const columns = ConnectedTablesColumns(onTableDeleteClick, projectId);

    return (
        <>
            {(fetchingTables || tables?.length > 0) && (
                <>
                    <GridRow className="flex justify-between items-center">
                        <GridCol>
                            <SectionHeader
                                title="Connected Tables"
                                titleSize="md"
                                description="You have connected the following tables from this connector"
                            />
                        </GridCol>
                        {tables?.length > 0 && !disableAdd && (
                            <GridCol>
                                <Button
                                    id="connect-table-2"
                                    onClick={onConnectTableClick}>
                                    <AddCircle />
                                    {" Add table"}
                                </Button>
                            </GridCol>
                        )}
                    </GridRow>
                    <DataTable
                        id="connector_tables"
                        disableRowSelectionHover={true}
                        columns={columns}
                        data={tables}
                        isLoading={fetchingTables}
                        pageCount={
                            tables?.length > 0
                                ? Math.ceil(tables?.length / 10)
                                : 1
                        }
                        defaultPageSize={10}
                        enableMultiSort={true}
                        enableGlobalFilter={tables?.length > 3 ? true : false}
                        columnsVisibility={
                            disableAdd ? { "action": false } : {}
                        }
                    />
                </>
            )}
            {fetchingTables === false && tables?.length == 0 && (
                <EmptyTable
                    id={"no-conn-table"}
                    label={"No connected tables"}
                    showLoading={false}>
                    {!disableAdd && (
                        <Button
                            id="connect-table-2"
                            onClick={onConnectTableClick}>
                            <AddCircle />
                            {" Add table"}
                        </Button>
                    )}
                </EmptyTable>
            )}
        </>
    );
};

export default WithErrorBoundary(ConnectedTables);
