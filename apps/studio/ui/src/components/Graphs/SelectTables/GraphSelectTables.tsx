// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef, useState } from "react";

import {
    ColumnDef,
    DataTable,
    RowSelectionState
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { GraphSelectTablesColumns } from "components/@ui/TableColumns/GraphSelectTables";

import { useTableSelectionForGraphs } from "hooks/graph/newGraph";

import { GraphFlowInfo, GraphTables } from "types/graphs";
import { TableListResponse } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import TableSliderOverview from "../../Tables/Overview/TableSliderOverview";

export interface GraphSelectTablesProps {
    tableList: TableListResponse[] | null;
    graphFlowInfo: GraphFlowInfo;
    setGraphFlowInfo: React.Dispatch<React.SetStateAction<GraphFlowInfo>>;
    allTables: GraphTables;
    setAllTables: React.Dispatch<React.SetStateAction<GraphTables>>;
    tableIdMap: { [key: string]: string };
    updateGroups: (config: GraphFlowInfo) => void;
    suggestLinks: () => void;
    handleError: HandleErrorFunc;
    setLoading: (value: boolean) => void;
    projectId?: string | null;
}

const GraphSelectTables = ({
    tableList,
    graphFlowInfo,
    setGraphFlowInfo,
    allTables,
    setAllTables,
    tableIdMap,
    updateGroups,
    suggestLinks,
    handleError,
    setLoading,
    projectId
}: GraphSelectTablesProps) => {
    const [previewName, setPreviewName] = useState<string>("");
    const [showSlider, setShowSlider] = useState<boolean>(false);

    const rowSelectionStateRef = useRef<{
        selected: string[];
        unSelected: string[];
    }>({ selected: [], unSelected: [] });

    const [rowSelectionState, setRowSelectionState] =
        useState<RowSelectionState>(
            tableList
                ? Object.fromEntries(
                      tableList.flatMap((l, index) =>
                          graphFlowInfo?.selected_tables?.includes(l.name)
                              ? [[index, true]]
                              : []
                      )
                  )
                : {}
        );

    const { onSelectTables } = useTableSelectionForGraphs(
        graphFlowInfo,
        allTables,
        setAllTables,
        setGraphFlowInfo,
        updateGroups,
        handleError,
        tableIdMap,
        suggestLinks,
        setLoading,
        projectId ?? undefined
    );

    const { selectedRows, unselectedRows } = useMemo(() => {
        const selected: TableListResponse[] = [];
        const unselected: TableListResponse[] = [];

        tableList?.forEach((table, index) => {
            if (rowSelectionState[index]) {
                selected.push(table);
            } else {
                unselected.push(table);
            }
        });

        return { selectedRows: selected, unselectedRows: unselected };
    }, [tableList, rowSelectionState]);

    useEffect(() => {
        const selected = selectedRows?.map((s) => s?.name);
        const unSelected = unselectedRows?.map((s) => s?.name);

        if (rowSelectionStateRef.current) {
            // To avoid multiple calls to onSelectTables
            const previosSelected = [
                ...rowSelectionStateRef.current.selected
            ].sort();
            const previosUnselected = [
                ...rowSelectionStateRef.current.unSelected
            ].sort();

            const currrentSelected = [...selected].sort();
            const currentUnselected = [...unSelected].sort();

            if (
                previosSelected.join(",") === currrentSelected.join(",") &&
                previosUnselected.join(",") === currentUnselected.join(",")
            ) {
                return;
            }
        }
        rowSelectionStateRef.current = {
            selected: selected,
            unSelected: unSelected
        };
        onSelectTables(
            selectedRows?.map((s) => s?.name),
            unselectedRows?.map((s) => s?.name)
        );
    }, [onSelectTables, selectedRows, unselectedRows]);

    const columns: ColumnDef<TableListResponse>[] = useMemo(
        () =>
            GraphSelectTablesColumns(showSlider, setShowSlider, setPreviewName),
        [showSlider]
    );

    return (
        <div data-testid="graph-select-tables-panel">
            {tableList && tableList?.length > 0 && (
                <DataTable
                    id="graph-select-table"
                    columns={columns}
                    disableRowSelectionHover={true}
                    data={tableList}
                    isLoading={false}
                    pageCount={
                        tableList?.length > 0
                            ? Math.ceil(tableList?.length / 10)
                            : 1
                    }
                    defaultPageSize={10}
                    enableMultiSort={true}
                    enableGlobalFilter={true}
                    manualRowSelectionState={rowSelectionState}
                    setManualRowSelectionState={setRowSelectionState}
                />
            )}
            {tableList && tableList?.length > 0 && showSlider && (
                <TableSliderOverview
                    name={previewName}
                    tableId={tableIdMap[previewName]}
                    allTablesInGraph={allTables}
                    onClose={() => setShowSlider(!showSlider)}
                    suggestLinks={() => {}}
                />
            )}
        </div>
    );
};

export default WithErrorBoundary(GraphSelectTables);
