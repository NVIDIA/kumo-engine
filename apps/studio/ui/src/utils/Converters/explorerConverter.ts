// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    AggregatedNodes,
    Feature,
    KumoapiExplainSubgraphSubgraph,
    SubgraphNode,
    SubgraphStoreEdges
} from "gen/v0/types";

import {
    GraphEdge,
    GraphNode,
    GraphNodeData,
    GraphXaiScores,
    XaiColumnArgs
} from "types/explorer";
import { ColumnArgs } from "types/tables";

import { sTypes } from "utils/Constants";
import { getKeys, getValues } from "utils/Helpers/objectHelpers";

/**
 * This method loop through the nested SubGraphResponse and
 * and creates of array of GraphNode with unique Node Id
 * This unique Node Id is used to create Node and Link/Edges
 * between nodes for NodeGraph
 *
 * @param node
 * @returns `GraphNode[]`
 *
 */

function isSubgraphNode(
    node: SubgraphNode | AggregatedNodes
): node is SubgraphNode {
    return "pkey_name" in node;
}

export const extractNodesFromSubgraph = (
    data: KumoapiExplainSubgraphSubgraph | undefined
) => {
    const nodeArr: GraphNode[] = [];
    const nodes = { ...data?.subgraph_store?.nodes };
    const nodeKeys =
        data?.feature_store?.features && data?.subgraph_store?.nodes
            ? getKeys(data?.subgraph_store?.nodes)
            : [];
    nodeKeys?.forEach((key, i) => {
        const nodeId = nodes[key].id;
        const feature = data?.feature_store?.features?.[key];
        const columns: Feature[] = getValues(feature);
        const tableKeys = getKeys(data?.subgraph_store?.tables);
        let tableName = "";
        tableKeys.forEach((t) => {
            const tableIds = data?.subgraph_store?.tables?.[t];
            if (tableIds?.includes(Number(key))) {
                tableName = t;
                return;
            }
        });

        const totalScore = columns
            ?.map((c) => c.score ?? 0)
            ?.reduce((a, b) => (a ?? 0) + (b ?? 0), 0);

        let pkey_value: string | number = nodes[key].id;
        // If pkey_name in nodes[key], nodes[key].id is number and
        // if it's >= 0 then use .pkey instead of .id which is the
        // pkey value we want to show to the frontend

        if (isSubgraphNode(nodes[key])) {
            if (nodes[key].id >= 0) {
                pkey_value = (nodes[key].pkey as string) ?? nodes[key].id;
            } else {
                pkey_value =
                    nodes[key].pkey == null
                        ? "<AGGREGATED>"
                        : (nodes[key].pkey as string);
            }

            // If pkey_name in nodes[key], nodes[key].id is number and
            // it's < 0 with pkey is null, then it means that this node
            // is an aggregated node, so we need to set the pkey_value to
            // "AGGREGATED"
        }

        const graphNodeData: GraphNodeData = {
            cols: columns?.map(
                (c) =>
                    ({ ...c, grad_score: c?.score ?? 0 }) as ColumnArgs &
                        XaiColumnArgs
            ),
            tableName: tableName,
            idName: nodes[key]?.pkey_name ?? "",
            value: pkey_value as string,
            totalScore: totalScore ?? 0
        };

        nodeArr.push({
            id: `${pkey_value}`,
            nodeId: `node_${nodeId}`,
            data: graphNodeData,
            position: { x: i * 100, y: i * 100 }
        });
    });
    return { nodeArr };
};

export const createLinksFromEdges = (
    edges: SubgraphStoreEdges | undefined,
    nodeArr: GraphNode[]
): GraphEdge[] => {
    const links: GraphEdge[] = [];
    const edgeKeys = getKeys(edges);
    let index = 0;
    edgeKeys.forEach((edge) => {
        const source = nodeArr?.find(
            (c) => `${c.nodeId}` === `node_${edge}`
        )?.id;
        if (source) {
            const pkeysLinks = getKeys(edges?.[edge]);
            pkeysLinks.forEach((key) => {
                const edgeLinks = edges?.[edge][key];

                edgeLinks?.forEach((link) => {
                    const target = nodeArr?.find(
                        (c) => `${c.nodeId}` === `node_${link}`
                    )?.id;
                    if (target) {
                        links.push({
                            id: `e-${index + 1}`,
                            source: `${source}`,
                            target: `${target}`,
                            type: "straight"
                        });
                    }
                    index++;
                });
            });
        }
    });
    return links;
};
/**
 * Loop through columns to get the global avg score and prediction
 * from matching bucket of xaiData
 *
 **/
export const updateColumnsWithXaiData = (
    columns: (ColumnArgs & XaiColumnArgs)[],
    tableName: string,
    xaiData: GraphXaiScores,
    anchorTime: string
) => {
    const tables = getKeys(xaiData);

    if (tableName?.includes(".")) {
        tableName = tableName.split(".")?.[1];
    }
    columns?.forEach((col) => {
        const finds = tables?.filter((e) => {
            const tableParts = e?.split(".");
            if (tableParts?.length < 2) {
                return false;
            }
            const extractedColumnName = tableParts.pop(); // Getting last element, column name
            // Getting next last element, table name (Note that pop() removes the last element from the array)
            const extractedTableName = tableParts.pop();

            return (
                extractedTableName === tableName &&
                extractedColumnName?.startsWith(col?.name as string)
            );
        });

        col.isXaiUpdated = true;
        if (finds?.length > 0) {
            let data = xaiData[finds[0]];
            // Check for `hop`, If many hop exists for a column, use lowest
            const includesHop = finds.filter((e) => e?.includes(`hop`));
            if (includesHop?.length > 0) {
                const lowestHop = getLowestHop(includesHop);
                if (lowestHop) data = xaiData[lowestHop];
            }

            // Find the xai bucket index to get prediction value
            const valueIndex = findXaiBucketIndex(
                col.stype as string,
                col.value as string | number,
                data.labels as (string | number)[],
                anchorTime
            );

            // Set the prediction and global avg, If the matching bucket found
            // in the `data.labels`
            if (typeof valueIndex === "number" && valueIndex >= 0) {
                col.xai_prediction = data?.predictions
                    ? Number(data.predictions[valueIndex])
                    : 0;
                col.xai_global_avg = data?.globalAvgScore;
            }
        }
    });
    return columns;
};

function getLowestHop(arr: string[]): string | null {
    if (arr?.length === 0) return null;

    let lowestHopElement = arr[0];
    let lowestHopValue = extractHopValue(lowestHopElement);

    for (const item of arr) {
        const hopValue = extractHopValue(item);
        if (hopValue < lowestHopValue) {
            lowestHopValue = hopValue;
            lowestHopElement = item;
        }
    }

    return lowestHopElement;
}

function extractHopValue(item: string): number {
    const match = item?.match(/\((\d+)_hop\)/);
    return match
        ? match?.length > 1
            ? parseInt(match[1], 10)
            : Infinity
        : Infinity; // Return a large number if no match
}

function findInRangeIndex(
    value: number | string,
    intervals: (number | string)[]
): number | null {
    for (let i = 0; i < intervals?.length; i++) {
        const interval = intervals[i];
        if (interval) {
            const [start, end] = parseRangeValues(interval);
            if (Number(value) < start) {
                return i; // Return the index if the value less than range
            } else if (isValueInRange(value, start, end)) {
                return i; // Return the index if the value is in range
            } else if (Number(value) > end && i === intervals.length - 1) {
                return i; // Return the index if the value greater than range
            }
        }
    }
    return -1; // Return null if the value doesn't fall into any interval
}

function parseRangeValues(interval: number | string): [number, number] {
    const trimmed = interval
        ?.toString()
        ?.replaceAll(/[()[\]]/g, "")
        ?.replaceAll(",", "-")
        ?.replaceAll("months", ""); // Remove brackets and months(timestamp)
    const [start, end] = trimmed.split(" - ").map(Number);
    return [start, end];
}

function isValueInRange(
    value: number | string,
    start: number,
    end: number
): boolean {
    return Number(start) <= Number(value) && Number(value) <= Number(end);
}

type SubgraphEdges = Record<string, Record<string, number[]>>;

/**
 * Build an undirected adjacency list. Edges may be stored in one direction
 * only (e.g. child→parent), so normalise to undirected before BFS from root.
 */
function buildAdjacency(edges: SubgraphEdges): Record<number, number[]> {
    const adj: Record<number, number[]> = {};
    const addAdj = (a: number, b: number) => {
        (adj[a] ??= []).push(b);
        (adj[b] ??= []).push(a);
    };
    for (const [srcStr, relations] of Object.entries(edges)) {
        const src = Number(srcStr);
        for (const targets of Object.values(relations)) {
            for (const tgt of targets) addAdj(src, tgt);
        }
    }
    return adj;
}

/** BFS from rootId assigning hop levels and parent/children maps. */
function bfsHopLevels(
    adj: Record<number, number[]>,
    rootId: number
): {
    hopLevels: Record<string, number>;
    parentMap: Record<string, string>;
    childrenMap: Record<string, string[]>;
} {
    const hopLevels: Record<string, number> = {};
    const parentMap: Record<string, string> = {};
    const childrenMap: Record<string, string[]> = {};
    hopLevels[`node_${rootId}`] = 0;
    const queue: number[] = [rootId];

    while (queue.length > 0) {
        const current = queue.shift()!;
        const currentKey = `node_${current}`;
        for (const neighborId of adj[current] ?? []) {
            const neighborKey = `node_${neighborId}`;
            if (hopLevels[neighborKey] === undefined) {
                hopLevels[neighborKey] = (hopLevels[currentKey] ?? 0) + 1;
                parentMap[neighborKey] = currentKey;
                if (!childrenMap[currentKey]) childrenMap[currentKey] = [];
                childrenMap[currentKey].push(neighborKey);
                queue.push(neighborId);
            }
        }
    }
    return { hopLevels, parentMap, childrenMap };
}

/** Build a table → nodeKey lookup. */
function buildNodeTableMap(
    tables: Record<string, number[]>
): Record<string, string> {
    const nodeTableMap: Record<string, string> = {};
    for (const [tableName, nodeIds] of Object.entries(tables)) {
        for (const id of nodeIds ?? []) {
            nodeTableMap[`node_${id}`] = tableName;
        }
    }
    return nodeTableMap;
}

/** Build a single explorer GraphNode from a subgraph node + its features. */
function buildExplorerNode(
    node: SubgraphNode | AggregatedNodes,
    feature: Parameters<typeof getValues>[0],
    tableName: string,
    hopLevel: number
): GraphNode {
    const columns: Feature[] = getValues(feature);
    const totalScore =
        columns?.reduce((sum, c) => sum + (c.score ?? 0), 0) ?? 0;

    let pkey_value: string | number = node.id;
    let hasPrimaryKey = false;
    if (isSubgraphNode(node)) {
        hasPrimaryKey = true;
        pkey_value =
            node.id >= 0
                ? ((node.pkey as string) ?? node.id)
                : node.pkey == null
                  ? "<AGGREGATED>"
                  : (node.pkey as string);
    }

    const aggregatedCount = !isSubgraphNode(node)
        ? (node as AggregatedNodes).count
        : undefined;

    const nodeKey = `node_${node.id}`;
    const graphNodeData: GraphNodeData = {
        cols: columns?.map(
            (c) =>
                ({ ...c, grad_score: c?.score ?? 0 }) as ColumnArgs &
                    XaiColumnArgs
        ),
        tableName,
        idName: isSubgraphNode(node) ? (node.pkey_name ?? "") : "",
        value: String(pkey_value),
        totalScore,
        hopLevel,
        hasPrimaryKey,
        aggregatedCount
    };

    return {
        id: nodeKey,
        nodeId: nodeKey,
        position: { x: 0, y: 0 },
        data: graphNodeData
    };
}

// BFS from root_id — returns nodeArr with hop levels, parentMap, and childrenMap.
export const extractNodesFromSubgraphWithHops = (
    data: KumoapiExplainSubgraphSubgraph | undefined
): {
    nodeArr: GraphNode[];
    parentMap: Record<string, string>;
    childrenMap: Record<string, string[]>;
} => {
    if (!data?.subgraph_store) {
        return { nodeArr: [], parentMap: {}, childrenMap: {} };
    }
    const store = data.subgraph_store;
    const nodes = store.nodes ?? {};
    const edges = store.edges ?? {};
    const tables = store.tables ?? {};

    const adj = buildAdjacency(edges as SubgraphEdges);
    const { hopLevels, parentMap, childrenMap } = bfsHopLevels(
        adj,
        store.root_id
    );
    const nodeTableMap = buildNodeTableMap(tables as Record<string, number[]>);
    const features = data.feature_store?.features ?? {};

    const nodeArr: GraphNode[] = [];
    // key === String(node.id) per API contract — both nodes and features are indexed by the same string key
    for (const [key, node] of Object.entries(nodes)) {
        const nodeKey = `node_${node.id}`;
        nodeArr.push(
            buildExplorerNode(
                node,
                features[key],
                nodeTableMap[nodeKey] ?? "",
                hopLevels[nodeKey]
            )
        );
    }

    return { nodeArr, parentMap, childrenMap };
};

function findXaiBucketIndex(
    stype: string,
    value: number | string,
    bucketLables: (string | number)[],
    anchorTime: string
) {
    switch (stype.toLowerCase()) {
        case sTypes.Categorical.toLowerCase():
        case sTypes.ID.toLowerCase(): {
            // Checking whether value directly matches the bucket values
            let i: number = bucketLables?.findIndex(
                (d) => d === value || Number(d) === Number(value)
            );

            // If no match found, checking with `other` or `n/a` values
            if (i === -1) {
                i = bucketLables?.findIndex(
                    (d) =>
                        d?.toString()?.toLowerCase()?.includes("n/a") ||
                        d?.toString()?.toLowerCase()?.includes("other")
                );
            }
            return i;
        }

        case sTypes.Numerical.toLowerCase():
            return findInRangeIndex(value, bucketLables ?? []);
        case sTypes.Timestamp.toLowerCase(): {
            try {
                const d1: Date = new Date(anchorTime);
                const d2: Date = new Date(value);
                const yearsDifference: number =
                    d1?.getFullYear() - d2?.getFullYear();
                const monthsDifference: number =
                    d1?.getMonth() - d2?.getMonth();
                const diff: number = yearsDifference * 12 + monthsDifference;
                return findInRangeIndex(diff, bucketLables ?? []);
            } catch {
                return -1;
            }
        }

        default: {
            // Checking whether value directly matches the bucket values
            const i: number = bucketLables?.findIndex(
                (d) => d === value || Number(d) === Number(value)
            );
            return i;
        }
    }
}
