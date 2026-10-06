// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import { buildTableColorMap } from "@kumo-ai/kumo-ui-catalog";
import type { KumoapiExplainSubgraphSubgraph } from "gen/v0/types";

import { extractNodesFromSubgraphWithHops } from "utils/Converters/explorerConverter";
import { logStackTrace } from "utils/errorHandler";
import {
    computeScoreTierMap,
    DEFAULT_MAX_HOP
} from "utils/subgraphTreeViewUtils";

// Link-prediction predicted items appear as hop-0 disjoint (TLP) or
// unreached (SLP, undefined hopLevel) nodes; we strip them from LHS arrays
// and surface them via predictedItemTableName for the RHS panel.
export function useSubgraphData(
    subgraph: KumoapiExplainSubgraphSubgraph | undefined,
    excludeDisjointHopZero = false
) {
    const { nodeArr, parentMap, childrenMap, rootId, predictedItemTableName } =
        useMemo(() => {
            const {
                nodeArr: rawNodeArr,
                parentMap: rawParentMap,
                childrenMap: rawChildrenMap
            } = extractNodesFromSubgraphWithHops(subgraph);
            const rootId = subgraph?.subgraph_store
                ? `node_${subgraph.subgraph_store.root_id}`
                : undefined;

            if (!excludeDisjointHopZero) {
                return {
                    nodeArr: rawNodeArr,
                    parentMap: rawParentMap,
                    childrenMap: rawChildrenMap,
                    rootId,
                    predictedItemTableName: undefined
                };
            }

            const excludedNodes = rawNodeArr.filter(
                (n) => !n.data.hopLevel && n.id !== rootId
            );
            // Current TLP/SLP tasks predict items of a single target table.
            // Surface a multi-table case loudly so we notice if heterogeneous
            // link-prediction ever arrives — UI today only labels/colors by
            // the first table.
            const uniqueTables = new Set(
                excludedNodes.map((n) => n.data.tableName).filter(Boolean)
            );
            if (uniqueTables.size > 1) {
                logStackTrace(
                    "useSubgraphData: predictions panel found multiple predicted tables; using first only",
                    new Error(`Tables: ${[...uniqueTables].join(", ")}`)
                );
            }
            const predictedItemTableName = excludedNodes[0]?.data.tableName;
            if (excludedNodes.length === 0) {
                return {
                    nodeArr: rawNodeArr,
                    parentMap: rawParentMap,
                    childrenMap: rawChildrenMap,
                    rootId,
                    predictedItemTableName: undefined
                };
            }
            const excludedIds = new Set(excludedNodes.map((n) => n.id));
            const nodeArr = rawNodeArr.filter((n) => !excludedIds.has(n.id));
            const parentMap: Record<string, string> = {};
            for (const [k, v] of Object.entries(rawParentMap)) {
                if (excludedIds.has(k) || excludedIds.has(v)) continue;
                parentMap[k] = v;
            }
            const childrenMap: Record<string, string[]> = {};
            for (const [k, vs] of Object.entries(rawChildrenMap)) {
                if (excludedIds.has(k)) continue;
                childrenMap[k] = vs.filter((v) => !excludedIds.has(v));
            }
            return {
                nodeArr,
                parentMap,
                childrenMap,
                rootId,
                predictedItemTableName
            };
        }, [subgraph, excludeDisjointHopZero]);

    const { tableColorMap, scoreTierMap, allTableNames, totalNodes } =
        useMemo(() => {
            const allTableNames = [
                ...new Set(nodeArr.map((n) => n.data.tableName).filter(Boolean))
            ];
            // Color map includes predictedItemTable even without LHS nodes;
            // allTableNames stays clean for the filter UI.
            const tablesForColorMap =
                predictedItemTableName &&
                !allTableNames.includes(predictedItemTableName)
                    ? [...allTableNames, predictedItemTableName]
                    : allTableNames;
            return {
                tableColorMap: buildTableColorMap(
                    tablesForColorMap,
                    nodeArr.find((n) => n.id === rootId)?.data.tableName
                ),
                scoreTierMap: computeScoreTierMap(nodeArr),
                allTableNames,
                totalNodes: nodeArr.length
            };
        }, [nodeArr, rootId, predictedItemTableName]);

    const allChipBoundaryParents = useMemo(() => {
        const hopLevel = new Map(
            nodeArr.map((n) => [n.id, n.data.hopLevel ?? 0])
        );
        return new Set(
            nodeArr
                .filter((n) =>
                    (childrenMap[n.id] ?? []).some(
                        (cId) => (hopLevel.get(cId) ?? 0) > DEFAULT_MAX_HOP
                    )
                )
                .map((n) => n.id)
        );
    }, [nodeArr, childrenMap]);

    return {
        nodeArr,
        parentMap,
        childrenMap,
        tableColorMap,
        scoreTierMap,
        allTableNames,
        totalNodes,
        allChipBoundaryParents,
        rootId,
        predictedItemTableName
    };
}
