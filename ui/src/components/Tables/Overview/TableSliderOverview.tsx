// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Button,
    CloseSmall,
    DataTable,
    Description,
    Form,
    FormField,
    FormWrapper,
    GridCol,
    GridRow,
    Label,
    SectionHeader,
    Select,
    Sheet,
    SheetContent,
    SheetHeader,
    Skeleton,
    useIsMounted
} from "@kumo-ai/kumo-ui-catalog";
import { ColStatsResponse } from "gen/v0/types";
import { Column } from "gen/v1/types/column";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { TableSliderOverviewColumns } from "components/@ui/TableColumns/TableSliderOverview";
import Preprocessing from "components/Tables/Connect/Preprocess";

import { useGetGraphInfo } from "hooks/graph/graphs";
import { useColumnStatsInfo, useGetTableInfo } from "hooks/tables/tables";
import { useValidationState } from "hooks/validation/validation";

import { GraphTables } from "types/graphs";
import { DataSourceType, FlowType, TableDetails } from "types/tables";

import {
    getConnectorIDFromTableSource,
    getSourceNameFromTableSource
} from "utils/Converters/tablesConverter";
import { getKeys } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";

export interface TableSliderOverviewProps {
    name: string;
    tableId: string;
    graphNameOrId?: string;
    allTablesInGraph: GraphTables;
    onClose: () => void;
    updateTable?: (n: string, t: KumoapiTableTableDefinition) => void;
    callCreateGraph?: () => void;
    suggestLinks: () => void;
    editable?: boolean;
}

const TableSliderOverview = ({
    name,
    tableId,
    graphNameOrId,
    allTablesInGraph, //Each may contain only selected columns for the graph
    onClose,
    updateTable,
    callCreateGraph,
    suggestLinks,
    editable = false
}: TableSliderOverviewProps) => {
    const getDefaultTableData = (): TableDetails =>
        (allTablesInGraph[name] ?? {}) as TableDetails;
    const getDefaultCreateArgs = () => {
        const tableInGraph: KumoapiTableTableDefinition = allTablesInGraph[
            name
        ] as KumoapiTableTableDefinition;
        const cols: Array<Column> = tableInGraph ? [...tableInGraph.cols] : [];

        const tArgs: KumoapiTableTableDefinition = {
            cols: cols,
            source_table: tableInGraph?.source_table,
            pkey: tableInGraph?.pkey,
            time_col: tableInGraph?.time_col,
            end_time_col: tableInGraph?.end_time_col
        };
        return tArgs;
    };

    const isMounted = useIsMounted();
    const [show, setShow] = useState(true);
    const [loading, setLoading] = useState(
        allTablesInGraph[name] ? false : true
    );
    const [, setGetGraphLoading] = useState(true);
    const [tableIdVar, setTableIdVar] = useState<string | null>(null);
    const [loadingTable, setLoadingTable] = useState(true);
    const [table, setTable] = useState<TableDetails>(getDefaultTableData());
    const [connectorNotExists, setConnectorNotExists] = useState<boolean>(
        table?.source_table?.data_source_type === DataSourceType.S3 &&
            getConnectorIDFromTableSource(table?.source_table) === ""
    );
    const [sourceTableNotExists, setSourceTableNotExists] = useState<boolean>(
        table?.source_table?.data_source_type === DataSourceType.S3 &&
            getSourceNameFromTableSource(table?.source_table) === ""
    );

    const [, , handleError] = useValidationState(setLoading);

    const [tableArgs, setTableArgs] = useState<KumoapiTableTableDefinition>(
        getDefaultCreateArgs() as KumoapiTableTableDefinition
    );

    const { selectedGraphInfo, isLoading } = useGetGraphInfo(
        graphNameOrId ?? null,
        handleError,
        setGetGraphLoading,
        isValid(graphNameOrId)
    );

    useEffect(() => {
        //Sets table ID appropriately
        if (!isValid(tableIdVar)) {
            if (isValid(graphNameOrId)) {
                let table_id = tableId;
                if (!isLoading && selectedGraphInfo) {
                    const fpid_mapping =
                        selectedGraphInfo?.table_name_to_fpid_mapping;
                    if (fpid_mapping && name in fpid_mapping) {
                        if (fpid_mapping[name].startsWith("NA.NA-"))
                            table_id = fpid_mapping[name]!;
                        else table_id = "NA.NA-" + fpid_mapping[name]!;
                    }
                }
                setTableIdVar(table_id);
            } else {
                setTableIdVar(tableId);
            }
        }
    }, [
        graphNameOrId,
        isLoading,
        name,
        selectedGraphInfo,
        tableId,
        tableIdVar
    ]);

    const { selectedTableInfo, loading: tableLoading } = useGetTableInfo(
        tableIdVar as string,
        handleError,
        isValid(tableIdVar)
    );

    useEffect(() => {
        if (getKeys(table).length < 1 && selectedTableInfo && !tableLoading) {
            setLoading(true);
            setTable(selectedTableInfo?.table);
            setConnectorNotExists(
                selectedTableInfo?.table?.source_table?.data_source_type ===
                    DataSourceType.S3 &&
                    getConnectorIDFromTableSource(
                        selectedTableInfo?.table?.source_table
                    ) === ""
            );
            setSourceTableNotExists(
                selectedTableInfo?.table?.source_table?.data_source_type ===
                    DataSourceType.S3 &&
                    getSourceNameFromTableSource(
                        selectedTableInfo?.table?.source_table
                    ) === ""
            );

            setTableArgs({} as KumoapiTableTableDefinition);
            if (isValid(name)) {
                //Prepare table args.
                const tableInGraph = allTablesInGraph[name] as TableDetails;
                const cols: Array<Column> = tableInGraph
                    ? [...(tableInGraph.cols as Column[])]
                    : [];

                const targs: KumoapiTableTableDefinition = {
                    cols: cols,
                    source_table: tableInGraph?.source_table,
                    pkey: tableInGraph?.pkey,
                    time_col: tableInGraph?.time_col,
                    end_time_col: tableInGraph?.end_time_col
                };

                setTableArgs(targs);
            }
            setLoading(false);
        }
    }, [
        allTablesInGraph,
        graphNameOrId,
        name,
        selectedTableInfo,
        table,
        tableId,
        tableLoading
    ]);

    const { columnStats, loading: fetchingSamples } = useColumnStatsInfo(
        editable || !tableIdVar ? null : tableIdVar,
        handleError,
        isMounted,
        editable || !isValid(tableIdVar) ? false : true
    );

    const onChange = useCallback(
        (open: boolean) => {
            setTimeout(() => {
                setShow(open);
            }, 200);
            if (!open) {
                setTimeout(() => {
                    onClose();
                }, 1000);
            }
        },
        [onClose]
    );

    const onSaveTableClick = useCallback(() => {
        const nTableArgs = { ...tableArgs };
        if (
            isValid(tableArgs.pkey) &&
            typeof tableArgs.pkey === "string" &&
            !tableArgs.cols.map((c) => c.name).includes(tableArgs.pkey)
        ) {
            nTableArgs.pkey = "";
        }
        if (
            isValid(tableArgs.end_time_col) &&
            typeof tableArgs.end_time_col === "string" &&
            !tableArgs.cols.map((c) => c.name).includes(tableArgs.end_time_col)
        ) {
            nTableArgs.end_time_col = "";
        }
        if (
            isValid(tableArgs.time_col) &&
            typeof tableArgs.time_col === "string" &&
            !tableArgs.cols.map((c) => c.name).includes(tableArgs.time_col)
        ) {
            nTableArgs.time_col = "";
        }
        setTableArgs(nTableArgs);
        if (updateTable) updateTable(name, nTableArgs);
        onChange(false);

        suggestLinks?.();
        callCreateGraph?.();
    }, [tableArgs, updateTable, name, onChange, callCreateGraph, suggestLinks]);

    const columns = useMemo(
        () =>
            editable
                ? []
                : TableSliderOverviewColumns(
                      table,
                      fetchingSamples,
                      columnStats as ColStatsResponse
                  ),
        [editable, table, fetchingSamples, columnStats]
    );

    return (
        <div id="graph-side-window-view">
            <Sheet open={show} onOpenChange={onChange}>
                <SheetContent>
                    <SheetHeader>
                        <GridRow className="flex items-center  justify-between">
                            <GridCol>
                                <SectionHeader
                                    title={name}
                                    titleClassName="!font-medium"
                                />
                            </GridCol>
                            <GridCol>
                                <Button
                                    id="table-overview"
                                    data-testid="close-icon"
                                    aria-label="Close"
                                    className="px-2 [&_svg]:size-6"
                                    variant="outline"
                                    onClick={() => {
                                        onChange(false);
                                    }}>
                                    <CloseSmall />
                                </Button>
                            </GridCol>
                        </GridRow>
                        {editable && (
                            <Description
                                variant="secondary"
                                size="sm"
                                className="leading-5">
                                {
                                    "The edits made here will be saved and used as a new temp table without altering the original table."
                                }
                            </Description>
                        )}
                        <Spacer height={12} />
                        {loading ? (
                            <Skeleton className="w-220 h-8" />
                        ) : (
                            <>
                                <GridRow className="flex flex-wrap">
                                    <GridCol>
                                        <Form>
                                            <FormWrapper>
                                                <FormField className="gap-0">
                                                    <Label
                                                        htmlFor="Source"
                                                        className="text-title-secondary mb-1">
                                                        Source
                                                    </Label>
                                                    <div className="w-72.5">
                                                        <Select
                                                            placeholder={
                                                                "file_type" in
                                                                    table.source_table &&
                                                                table
                                                                    .source_table
                                                                    .file_type
                                                                    ? "File"
                                                                    : "Connector"
                                                            }
                                                            id={"select-source"}
                                                            options={[]}
                                                            onValuesChange={() => {}}
                                                            disabled={true}
                                                        />{" "}
                                                    </div>
                                                </FormField>
                                            </FormWrapper>
                                        </Form>
                                    </GridCol>
                                    {!connectorNotExists && (
                                        <GridCol>
                                            <Form>
                                                <FormWrapper>
                                                    <FormField className="gap-0">
                                                        <Label
                                                            htmlFor="Connector"
                                                            className="text-title-secondary mb-1">
                                                            Connector
                                                        </Label>
                                                        <div className="w-72.5">
                                                            <Select
                                                                placeholder={
                                                                    getConnectorIDFromTableSource(
                                                                        table?.source_table
                                                                    ) as string
                                                                }
                                                                id={
                                                                    "select-connector"
                                                                }
                                                                options={[]}
                                                                onValuesChange={() => {}}
                                                                disabled={true}
                                                            />{" "}
                                                        </div>
                                                    </FormField>
                                                </FormWrapper>
                                            </Form>
                                        </GridCol>
                                    )}
                                    {!sourceTableNotExists && (
                                        <GridCol>
                                            <Form>
                                                <FormWrapper>
                                                    <FormField className="gap-0">
                                                        <Label
                                                            htmlFor="Source Table"
                                                            className="text-title-secondary mb-1">
                                                            Source table
                                                        </Label>{" "}
                                                        <div className="w-72.5">
                                                            <Select
                                                                placeholder={
                                                                    getSourceNameFromTableSource(
                                                                        table?.source_table
                                                                    ) as string
                                                                }
                                                                id={
                                                                    "select-source"
                                                                }
                                                                options={[]}
                                                                onValuesChange={() => {}}
                                                                disabled={true}
                                                            />{" "}
                                                        </div>
                                                    </FormField>
                                                </FormWrapper>
                                            </Form>
                                        </GridCol>
                                    )}
                                </GridRow>
                                {editable ? (
                                    <>
                                        <Preprocessing
                                            setLoading={setLoadingTable}
                                            isLoading={loadingTable}
                                            flowType={FlowType.EDIT}
                                            tableArgs={tableArgs}
                                            handleError={handleError}
                                            setTableArgs={setTableArgs}
                                            selectedConnector={{
                                                label: getConnectorIDFromTableSource(
                                                    table?.source_table
                                                ),
                                                value: getConnectorIDFromTableSource(
                                                    table?.source_table
                                                )
                                            }}
                                            cancelFetchSampleCal={false}
                                            connectorId={getConnectorIDFromTableSource(
                                                table?.source_table
                                            )}
                                            selectedSourceTableName={getSourceNameFromTableSource(
                                                table?.source_table
                                            )}
                                            selectFromTableArgs={true}
                                            connectorNotExists={
                                                connectorNotExists
                                            }
                                            sourcetableNotExists={
                                                sourceTableNotExists
                                            }
                                            setColumnNamesAdjusted={() => {}}
                                            columnNamesAdjusted={false}
                                            setUnnamedConSourceTable={() => {}}
                                        />
                                        <Spacer height={16} />
                                        <GridRow xs={12} sm={12} md={12}>
                                            <GridCol xs={12} sm={12} md="auto">
                                                <Button
                                                    onClick={onSaveTableClick}
                                                    id="save-table"
                                                    disabled={loadingTable}>
                                                    Save
                                                </Button>
                                            </GridCol>
                                        </GridRow>
                                    </>
                                ) : (
                                    table?.cols && (
                                        <>
                                            <Spacer height={16} />
                                            <DataTable
                                                id="preprocessing"
                                                columns={columns}
                                                disableRowSelectionHover={true}
                                                data={table?.cols}
                                                isLoading={false}
                                                pageCount={
                                                    table?.cols?.length > 0
                                                        ? Math.ceil(
                                                              table?.cols
                                                                  ?.length / 50
                                                          )
                                                        : 1
                                                }
                                                defaultPageSize={50}
                                                enableMultiSort={true}
                                                enableGlobalFilter={true}
                                            />
                                        </>
                                    )
                                )}
                            </>
                        )}
                    </SheetHeader>
                </SheetContent>
            </Sheet>
        </div>
    );
};

export default WithErrorBoundary(TableSliderOverview);
