// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useMemo } from "react";

import { Card, CardContent } from "@kumo-ai/kumo-ui-catalog";

import GraphInformationPanel from "components/Graphs/InformationPanel/GraphInformationPanel";

import { ColumnGroup, GraphFlowInfo, GraphTables, JSGraph } from "types/graphs";
import { FlowType, JSTable } from "types/tables";

import { tablesFromGraphResponse } from "utils/Converters/graphsConverter";
import { getKeys } from "utils/Helpers/objectHelpers";

/**
 * Graph Configuration Panel Component
 *
 */
const GraphConfigPanel = ({
    gfc,
    setGfc,
    allTables,
    setAllTables,
    groups,
    setGroups,
    tableIdMap,
    suggestLinks
}: {
    gfc: GraphFlowInfo;
    setGfc: React.Dispatch<React.SetStateAction<GraphFlowInfo>>;
    allTables: GraphTables;
    setAllTables: React.Dispatch<React.SetStateAction<GraphTables>>;
    groups: Array<ColumnGroup>;
    setGroups: React.Dispatch<React.SetStateAction<Array<ColumnGroup>>>;
    tableIdMap: { [key: string]: string };
    suggestLinks: () => void;
}) => {
    const defgraphInfo: JSGraph = useMemo(() => {
        return {
            id: "",
            name: gfc?.name,
            tables: gfc?.tables,
            groups: groups,
            totalLinks: 0,
            totalTables: getKeys(gfc?.tables).length,
            created: "",
            updated: "",
            dateCreated: "",
            dateUpdated: "",
            isActive: true
        };
    }, [groups, gfc]);

    // Create tables from graphInfo

    const tables: JSTable[] = useMemo(() => {
        if (defgraphInfo !== null) {
            return tablesFromGraphResponse(defgraphInfo);
        } else {
            return [];
        }
    }, [defgraphInfo]);

    const graphInfoPanel = useMemo(() => {
        if (!defgraphInfo) return null;

        return (
            <GraphInformationPanel
                graphInfo={defgraphInfo}
                tables={tables}
                weakGraphEdges={[]}
                enableEditTable={true}
                gfc={gfc}
                setGfc={setGfc}
                allTables={allTables}
                setAllTables={setAllTables}
                groups={groups}
                setGroups={setGroups}
                tableIds={tableIdMap}
                suggestLinks={
                    gfc.flow_type === FlowType.EDIT ? () => {} : suggestLinks
                }
            />
        );
    }, [
        defgraphInfo,
        tables,
        gfc,
        setGfc,
        allTables,
        setAllTables,
        groups,
        setGroups,
        tableIdMap,
        suggestLinks
    ]);

    return (
        <>
            {defgraphInfo && (
                <Card className="m-0 p-0">
                    <CardContent className="!p-0">{graphInfoPanel}</CardContent>
                </Card>
            )}
        </>
    );
};

export default GraphConfigPanel;
