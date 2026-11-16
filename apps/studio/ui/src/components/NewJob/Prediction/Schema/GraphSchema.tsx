// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import { GraphSnapshotResponse } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import GraphInformationPanel from "components/Graphs/InformationPanel/GraphInformationPanel";

import { useGraphSnapshot } from "hooks/snapshot/graphSnapshot";

import { ColumnGroup, GraphFlowInfo, GraphTables } from "types/graphs";

import {
    graphSnapshotToGraphInformationTables,
    graphSnapshotToJSGraph
} from "utils/Converters/snapshotConverter";
import { defFlowInfo } from "utils/graph";
import { copyObject } from "utils/Helpers/objectHelpers";

export interface GraphSchemaProps {
    graphSnapshotId: string;
}

const GraphSchema: React.FC<GraphSchemaProps> = ({ graphSnapshotId }) => {
    const [, setLoading] = useState<boolean>(false);

    const { graphSnapshot } = useGraphSnapshot(
        graphSnapshotId as string,
        null,
        null,
        setLoading
    );

    // graphFlowInfo ,allTables, groups are defined here just to pass it to
    // GraphInformationPanel , but is not used.
    const [groups, setGroups] = useState<Array<ColumnGroup>>([]);
    const [graphFlowInfo, setGraphFlowInfo] = useState<GraphFlowInfo>(
        copyObject(defFlowInfo)
    );
    // List of tables with individual table details
    const [allTables, setAllTables] = useState<GraphTables>({});

    const graphInfo = graphSnapshotToJSGraph(
        graphSnapshotId as string,
        graphSnapshot as GraphSnapshotResponse
    );
    const tables = graphSnapshotToGraphInformationTables(
        graphSnapshot as GraphSnapshotResponse
    );

    return (
        graphSnapshotId &&
        graphInfo && (
            <GraphInformationPanel
                graphInfo={graphInfo}
                tables={tables}
                weakGraphEdges={[]}
                gfc={graphFlowInfo}
                setGfc={setGraphFlowInfo}
                allTables={allTables}
                setAllTables={setAllTables}
                groups={groups}
                setGroups={setGroups}
                tableIds={graphInfo.table_name_to_fpid_mapping}
                suggestLinks={() => {}}
            />
        )
    );
};

export default WithErrorBoundary(GraphSchema);
