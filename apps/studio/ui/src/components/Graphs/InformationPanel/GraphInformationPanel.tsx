// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { memo, useState } from "react";

import { useWindowDimensions } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import {
    ColumnGroup,
    EdgeStats,
    GraphFlowInfo,
    GraphTables,
    JSGraph
} from "types/graphs";
import { JSTable } from "types/tables";

import { isObjectNotEmpty } from "utils/Helpers/objectHelpers";

import TableGraphView from "./TableGraphView";
import TableView from "./TableView";

/**
 * Wrapper component for Grah View with switch button.
 *
 * Can switch from Graph View to Time Range View
 *
 * @param graphInfo graph info
 * @param tables List of tables
 * @param weakGraphEdges List of graph edge statistics with poor edge health
 * @returns ```<GraphInformationPanel />```
 */
export interface GraphInformationPanelProps {
    graphInfo: JSGraph;
    tables: Array<JSTable>;
    weakGraphEdges: Array<EdgeStats>;
    enableEditTable?: boolean;
    gfc: GraphFlowInfo;
    setGfc: React.Dispatch<React.SetStateAction<GraphFlowInfo>>;
    allTables: GraphTables;
    setAllTables: React.Dispatch<React.SetStateAction<GraphTables>>;
    groups: Array<ColumnGroup>;
    setGroups: React.Dispatch<React.SetStateAction<Array<ColumnGroup>>>;
    tableIds?: { [key: string]: string };
    callCreateGraph?: () => void;
    suggestLinks: () => void;
    customHeight?: number;
    graphView?: boolean;
    setGraphView?: React.Dispatch<React.SetStateAction<boolean>>;
    columnsPerRow?: number;
}

const GraphInformationPanel: React.FC<GraphInformationPanelProps> = ({
    graphInfo,
    tables,
    weakGraphEdges,
    enableEditTable = false,
    gfc,
    setGfc,
    allTables,
    setAllTables,
    groups,
    setGroups,
    tableIds = {},
    callCreateGraph,
    suggestLinks = () => {},
    customHeight,
    graphView,
    setGraphView,
    columnsPerRow
}) => {
    const { height, width } = useWindowDimensions();
    const [selectedTable, setSelectedTable] = useState<string | null>(null);
    const [showGraph, setShowGraph] = useState<boolean>(true);
    const isGraphView = graphView ?? showGraph;
    const updateGraphView = setGraphView ?? setShowGraph;
    return (
        <>
            <TableGraphView
                graphData={graphInfo}
                tableData={tables}
                weakEdges={weakGraphEdges}
                setSelectedTable={(newtable: string) => {
                    setSelectedTable(newtable);

                    updateGraphView(false);
                }}
                height={customHeight ?? height * 0.75}
                width={width}
                open={isGraphView}
                enableEditTable={enableEditTable}
                gfc={gfc as GraphFlowInfo}
                setGfc={setGfc}
                allTables={allTables as GraphTables}
                setAllTables={setAllTables}
                groups={groups}
                setGroups={setGroups}
                callCreateGraph={callCreateGraph}
                suggestLinks={suggestLinks}
                columnsPerRow={columnsPerRow}
            />
            {!isGraphView && (
                <TableView
                    tables={tables}
                    selectedTable={selectedTable as string}
                    navigateToGraph={() => {
                        updateGraphView(true);
                        setSelectedTable(null);
                    }}
                    tableIds={isObjectNotEmpty(tableIds) ? tableIds : {}}
                />
            )}
        </>
    );
};

export default memo(WithErrorBoundary(GraphInformationPanel));
