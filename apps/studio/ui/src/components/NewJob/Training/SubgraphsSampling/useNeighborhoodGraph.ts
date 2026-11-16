// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import { HopGraphEdge, HopGraphNode } from "@kumo-ai/kumo-ui-catalog";
import yaml from "js-yaml";

import {
    ColumnGroup,
    EdgeBudget,
    HopEdgeConfig,
    JSGraph,
    NeighborSamplingInfo,
    SchemaEdge
} from "types/graphs";

// ─── Regex for "table.col->table.col" edge spec keys ─────────────────────────
const EDGE_SPEC_RE = /^(.+)\.([^.]+)->(.+)\.([^.]+)$/;

// ─── PQL parsing ─────────────────────────────────────────────────────────────
/**
 * Returns the matched identifiers from the FOR EACH clause.
 * Handles quoted identifiers and 1–3 dot-separated parts.
 */
function parseForEachIdentifiers(pq: string): string[] | null {
    const flat = pq.replace(/\s+/g, " ");
    const m = flat.match(
        /FOR EACH\s+((?:"[^"]+"|[A-Za-z0-9_]+)(?:\.(?:"[^"]+"|[A-Za-z0-9_]+)){0,2})/i
    );
    if (!m) return null;
    return m[1]
        .split(".")
        .map((p) => p.replace(/^"|"$/g, "").trim())
        .filter(Boolean);
}

/**
 * Resolves the target table from the FOR EACH identifiers, matching
 * against the actual table names present in the graph.
 */
function resolveTargetTable(
    predictiveQuery: string | null | undefined,
    graph: JSGraph | null | undefined
): string | null {
    const ids = parseForEachIdentifiers(predictiveQuery ?? "");
    if (!ids || ids.length === 0) return null;
    const graphTables = new Set(Object.keys(graph?.tables ?? {}));
    // Try full joined identifier first, then each part individually
    const joined = ids.join(".");
    if (graphTables.has(joined)) return joined;
    for (const part of ids) {
        if (graphTables.has(part)) return part;
    }
    // Best-effort: return last identifier (matches PR behaviour)
    return ids[ids.length - 1] ?? null;
}

// ─── Model plan parsing ───────────────────────────────────────────────────────
/**
 * Parses neighbor_sampling.num_neighbors from the model plan YAML.
 *
 * Supports:
 *  - Modern hop-map: num_neighbors is an array whose items are objects with
 *    hop1/hop2/… keys, e.g. [{hop1:{default:12,…}, hop2:{default:12}}]
 *  - Per-item hop-map: each array item has exactly one hopN key,
 *    e.g. [{hop1:{default:12}}, {hop2:{default:12}}]
 *  - Legacy array-of-arrays: [[12,12],[12,12]]
 */
function parseNeighborSampling(
    modelPlanYaml: string | null | undefined
): NeighborSamplingInfo {
    if (!modelPlanYaml) return { hops: [], hasData: false };

    try {
        const parsed = yaml.load(modelPlanYaml, {
            schema: yaml.JSON_SCHEMA
        }) as Record<string, unknown>;
        const rawNeighbors = (
            parsed as {
                neighbor_sampling?: {
                    num_neighbors?: Array<
                        Array<number | string | null> | Record<string, unknown>
                    >;
                };
            }
        )?.neighbor_sampling?.num_neighbors;

        if (!Array.isArray(rawNeighbors) || rawNeighbors.length === 0) {
            return { hops: [], hasData: false };
        }

        const hops: HopEdgeConfig[] = [];

        rawNeighbors.forEach((entry) => {
            // ── Legacy array format: [12, 12] or [[12,12]] ────────────────
            if (Array.isArray(entry)) {
                entry.forEach((hopValue, idx) => {
                    const n =
                        typeof hopValue === "number"
                            ? hopValue
                            : parseInt(String(hopValue), 10);
                    if (!isNaN(n)) {
                        const hopNum = idx + 1;
                        const existing = hops.find((h) => h.hop === hopNum);
                        if (existing) {
                            existing.defaultBudget = Math.max(
                                existing.defaultBudget,
                                n
                            );
                        } else {
                            hops.push({
                                hop: hopNum,
                                defaultBudget: n,
                                edgeBudgets: []
                            });
                        }
                    }
                });
                return;
            }

            // ── Object format: { hopN: { default: 12, "a.x->b.y": 5 } } ──
            if (entry && typeof entry === "object") {
                Object.entries(entry as Record<string, unknown>).forEach(
                    ([hopKey, hopValue]) => {
                        const hopNum = parseInt(
                            hopKey.replace(/hop\s*/i, ""),
                            10
                        );
                        if (isNaN(hopNum) || hopNum <= 0) return;
                        if (hopValue === null) return;

                        let defaultBudget = 0;
                        const edgeBudgets: EdgeBudget[] = [];

                        if (
                            hopValue &&
                            typeof hopValue === "object" &&
                            !Array.isArray(hopValue)
                        ) {
                            const hopObj = hopValue as Record<string, unknown>;

                            if ("default" in hopObj) {
                                const d = hopObj.default;
                                defaultBudget =
                                    typeof d === "number"
                                        ? d
                                        : parseInt(String(d), 10) || 0;
                            }

                            // Parse explicit edge budgets like "a.x->b.y": 5
                            Object.entries(hopObj).forEach(([key, val]) => {
                                if (key === "default") return;
                                const m = key.match(EDGE_SPEC_RE);
                                if (m) {
                                    const [, st, sc, tt, tc] = m;
                                    const budget =
                                        val === "inferred"
                                            ? defaultBudget
                                            : typeof val === "number"
                                              ? val
                                              : parseInt(String(val), 10) || 0;
                                    edgeBudgets.push({
                                        edge: {
                                            sourceTable: st,
                                            sourceColumn: sc,
                                            targetTable: tt,
                                            targetColumn: tc
                                        },
                                        budget
                                    });
                                }
                            });
                        } else {
                            defaultBudget =
                                typeof hopValue === "number"
                                    ? hopValue
                                    : parseInt(String(hopValue), 10) || 0;
                        }

                        const existing = hops.find((h) => h.hop === hopNum);
                        if (existing) {
                            existing.defaultBudget = Math.max(
                                existing.defaultBudget,
                                defaultBudget
                            );
                            existing.edgeBudgets.push(...edgeBudgets);
                        } else {
                            hops.push({
                                hop: hopNum,
                                defaultBudget,
                                edgeBudgets
                            });
                        }
                    }
                );
            }
        });

        hops.sort((a, b) => a.hop - b.hop);
        return { hops, hasData: hops.length > 0 };
    } catch {
        return { hops: [], hasData: false };
    }
}

// ─── Schema helpers ───────────────────────────────────────────────────────────
function extractSchemaEdges(
    groups: ColumnGroup[],
    tables: Record<string, { pkey?: string | null }> = {}
): SchemaEdge[] {
    // Build the set of "table_name.col_name" strings that are primary keys,
    // matching the approach used by TableGraphView.
    const primaryKeys = new Set<string>();
    Object.entries(tables).forEach(([tableName, def]) => {
        if (def?.pkey) primaryKeys.add(`${tableName}.${def.pkey}`);
    });

    const edges: SchemaEdge[] = [];
    const seenEdges = new Set<string>();

    const addEdge = (t1: string, c1: string, t2: string, c2: string) => {
        if (t1 === t2) return;
        const key = `${t1}.${c1}->${t2}.${c2}`;
        if (seenEdges.has(key)) return;
        seenEdges.add(key);
        edges.push({ table1: t1, col1: c1, table2: t2, col2: c2 });
    };

    for (const group of groups) {
        if ((group.columns?.length ?? 0) < 2) continue;

        const pkCols = group.columns.filter(({ table_name, col_name }) =>
            primaryKeys.has(`${table_name}.${col_name}`)
        );
        const fkCols = group.columns.filter(
            (c) => !primaryKeys.has(`${c.table_name}.${c.col_name}`)
        );

        // If no PK found via graph.tables, fall back to first column as anchor
        // (mirrors PR's fallback for groups without a resolvable PK)
        if (pkCols.length === 0 && group.columns.length >= 2) {
            pkCols.push(group.columns[0]);
            fkCols.splice(
                fkCols.findIndex(
                    (c) =>
                        c.table_name === group.columns[0].table_name &&
                        c.col_name === group.columns[0].col_name
                ),
                1
            );
        }

        // PK → each FK (and reverse), avoiding FK-to-FK edges
        for (const pk of pkCols) {
            for (const fk of fkCols) {
                addEdge(pk.table_name, pk.col_name, fk.table_name, fk.col_name);
                addEdge(fk.table_name, fk.col_name, pk.table_name, pk.col_name);
            }
        }

        // Junction tables: connect each PK pair (many-to-many)
        for (let i = 0; i < pkCols.length; i++) {
            for (let j = i + 1; j < pkCols.length; j++) {
                const a = pkCols[i],
                    b = pkCols[j];
                addEdge(a.table_name, a.col_name, b.table_name, b.col_name);
                addEdge(b.table_name, b.col_name, a.table_name, a.col_name);
            }
        }
    }
    return edges;
}

function buildTableAdjacency(
    schemaEdges: SchemaEdge[]
): Map<string, Set<string>> {
    const adj = new Map<string, Set<string>>();
    for (const e of schemaEdges) {
        if (!adj.has(e.table1)) adj.set(e.table1, new Set());
        if (!adj.has(e.table2)) adj.set(e.table2, new Set());
        adj.get(e.table1)!.add(e.table2);
        adj.get(e.table2)!.add(e.table1);
    }
    return adj;
}

/**
 * Heuristic: a column is likely the primary key of its table.
 * Strips common plural suffixes (matches the PR's isPrimaryKeyColumn).
 */
function isPKColumn(tableName: string, colName: string): boolean {
    const lowerTable = tableName.toLowerCase();
    const lowerCol = colName.toLowerCase();
    const stem = lowerTable
        .replace(/ies$/, "y")
        .replace(/ses$/, "s")
        .replace(/es$/, "")
        .replace(/s$/, "");
    if (lowerCol.includes(stem)) return true;
    if (lowerCol === "id") return true;
    if (lowerCol === `${stem}_id` || lowerCol === `${stem}id`) return true;
    return false;
}

function hasPrimaryKey(
    tableName: string,
    schemaEdges: SchemaEdge[],
    tablesWithPKey: Set<string>
): boolean {
    // Authoritative check first (from graph.tables[name].pkey)
    if (tablesWithPKey.has(tableName)) return true;
    // Fall back to heuristic for schemas without pkey metadata
    return schemaEdges.some(
        (e) =>
            (e.table1 === tableName && isPKColumn(tableName, e.col1)) ||
            (e.table2 === tableName && isPKColumn(tableName, e.col2))
    );
}

function hasSchemaConnection(
    sourceTable: string,
    targetTable: string,
    schemaEdges: SchemaEdge[]
): boolean {
    return schemaEdges.some(
        (e) =>
            (e.table1 === sourceTable && e.table2 === targetTable) ||
            (e.table1 === targetTable && e.table2 === sourceTable)
    );
}

/**
 * Returns the sampling budget for a given directed hop edge.
 * Returns 0 if the edge has no FK relationship in the schema (matches PR's
 * strict validation). Prefers explicit edge entries over the hop default.
 */
function getEdgeBudget(
    sourceTable: string,
    targetTable: string,
    hopConfig: HopEdgeConfig,
    schemaEdges: SchemaEdge[]
): number {
    if (!hasSchemaConnection(sourceTable, targetTable, schemaEdges)) {
        return 0;
    }
    for (const eb of hopConfig.edgeBudgets) {
        const e = eb.edge;
        if (
            (e.sourceTable === sourceTable && e.targetTable === targetTable) ||
            (e.sourceTable === targetTable && e.targetTable === sourceTable)
        ) {
            return eb.budget;
        }
    }
    return hopConfig.defaultBudget;
}

// ─── Neighbourhood builder ────────────────────────────────────────────────────
/**
 * Traverses the schema graph hop-by-hop from the target table.
 *
 * - Each table appears at most once per hop (deduped by `tablesAtHop`).
 * - Double-back prevention: a table that appeared N-2 hops ago is skipped if
 *   it has a primary key (mirrors the PR's logic exactly).
 */
type BuildNeighborhoodResult = {
    nodes: HopGraphNode[];
    edges: HopGraphEdge[];
    configuredHopCount: number;
    renderedHopCount: number;
};

function buildNeighborhood(
    graph: JSGraph,
    targetTable: string,
    neighborInfo: NeighborSamplingInfo
): BuildNeighborhoodResult {
    const tables = (graph.tables ?? {}) as Record<
        string,
        { pkey?: string | null }
    >;
    const schemaEdges = extractSchemaEdges(graph.groups ?? [], tables);
    const tablesWithPKey = new Set<string>(
        Object.entries(tables)
            .filter(([, def]) => def?.pkey)
            .map(([name]) => name)
    );
    const tableAdj = buildTableAdjacency(schemaEdges);
    if (!tableAdj.has(targetTable)) {
        tableAdj.set(targetTable, new Set());
    }

    const maxHop = neighborInfo.hops.reduce((m, h) => Math.max(m, h.hop), 0);
    if (maxHop === 0)
        return {
            nodes: [],
            edges: [],
            configuredHopCount: 0,
            renderedHopCount: 0
        };

    // ── Pre-compute hop adjacency ─────────────────────────────────────────
    // hopAdj[h][srcTable] = [{targetTable, budget}]
    const hopAdj = new Map<
        number,
        Map<string, { targetTable: string; budget: number }[]>
    >();

    for (let h = 1; h <= maxHop; h++) {
        const hopConfig = neighborInfo.hops.find((hc) => hc.hop === h);
        if (!hopConfig) continue;

        const adjForHop = new Map<
            string,
            { targetTable: string; budget: number }[]
        >();

        // Source tables for hop h = unique target tables from hop h-1
        const sourceTables =
            h === 1
                ? [targetTable]
                : Array.from(
                      new Set(
                          Array.from(hopAdj.get(h - 1)?.values() ?? []).flatMap(
                              (targets) =>
                                  targets
                                      .filter((t) => t.budget > 0)
                                      .map((t) => t.targetTable)
                          )
                      )
                  );

        sourceTables.forEach((src) => {
            const neighbors = tableAdj.get(src) ?? new Set<string>();
            const targets: { targetTable: string; budget: number }[] = [];
            neighbors.forEach((tgt) => {
                const budget = getEdgeBudget(src, tgt, hopConfig, schemaEdges);
                if (budget > 0) targets.push({ targetTable: tgt, budget });
            });
            if (targets.length > 0) adjForHop.set(src, targets);
        });

        hopAdj.set(h, adjForHop);
    }

    // ── Traverse hops and emit nodes / edges ──────────────────────────────
    const nodes: HopGraphNode[] = [];
    const edges: HopGraphEdge[] = [];
    const hopLayers = new Map<number, { id: string; table: string }[]>();
    const tablesAtHop = new Map<number, Set<string>>();

    // Root
    const rootId = `hop0:${targetTable}`;
    nodes.push({ id: rootId, table: targetTable, hop: 0 });
    hopLayers.set(0, [{ id: rootId, table: targetTable }]);
    tablesAtHop.set(0, new Set([targetTable]));

    for (let h = 1; h <= maxHop; h++) {
        const adjForHop = hopAdj.get(h);
        if (!adjForHop) continue;

        const prevTables = tablesAtHop.get(h - 1) ?? new Set<string>();
        const nodesThisHop: { id: string; table: string }[] = [];
        const tablesThisHop = new Set<string>();

        prevTables.forEach((src) => {
            const targets = adjForHop.get(src) ?? [];
            targets.forEach(({ targetTable: tgt, budget }) => {
                // Double-back prevention: skip table T at hop H if T appeared
                // at hop H-2 AND T has a primary key (dimension table).
                // Fact tables (no PK) are allowed to cycle — e.g. a transaction
                // table can appear at hop 1 and again at hop 3.
                // Mirrors the PR's logic exactly.
                const twoAgo = h - 2;
                if (twoAgo >= 0) {
                    const atTwoAgo = tablesAtHop.get(twoAgo);
                    if (
                        atTwoAgo?.has(tgt) &&
                        hasPrimaryKey(tgt, schemaEdges, tablesWithPKey)
                    ) {
                        return;
                    }
                }

                if (!tablesThisHop.has(tgt)) {
                    tablesThisHop.add(tgt);
                    const nodeId = `hop${h}:${tgt}:0`;
                    nodesThisHop.push({ id: nodeId, table: tgt });
                    nodes.push({
                        id: nodeId,
                        table: tgt,
                        hop: h,
                        neighborCount: budget
                    });
                }

                const prevNode = (hopLayers.get(h - 1) ?? []).find(
                    (n) => n.table === src
                );
                const thisNode = nodesThisHop.find((n) => n.table === tgt);
                if (prevNode && thisNode) {
                    const edgeId = `e:${prevNode.id}→${thisNode.id}`;
                    if (!edges.some((e) => e.id === edgeId)) {
                        edges.push({
                            id: edgeId,
                            sourceId: prevNode.id,
                            targetId: thisNode.id,
                            samplingBudget: budget
                        });
                    }
                }
            });
        });

        hopLayers.set(h, nodesThisHop);
        tablesAtHop.set(h, tablesThisHop);
    }

    const renderedHopCount = Array.from(hopLayers.entries()).filter(
        ([h, layer]) => h > 0 && layer.length > 0
    ).length;

    return { nodes, edges, configuredHopCount: maxHop, renderedHopCount };
}

// ─── Hook ─────────────────────────────────────────────────────────────────────
export interface NeighborhoodGraphResult {
    nodes: HopGraphNode[];
    edges: HopGraphEdge[];
    targetTable: string | null;
    isReady: boolean;
    notReadyReason:
        | "missing-inputs"
        | "no-target-table"
        | "no-schema-edges"
        | "no-hops"
        | "invalid-hops"
        | null;
    /** Total hops configured in the model plan (0 when not ready). */
    configuredHopCount: number;
    /** Hops actually rendered in the graph (0 when not ready). */
    renderedHopCount: number;
}

export function useNeighborhoodGraph(
    graph: JSGraph | null,
    predictiveQuery: string,
    modelPlanYaml: string
): NeighborhoodGraphResult {
    return useMemo((): NeighborhoodGraphResult => {
        const EMPTY: NeighborhoodGraphResult = {
            nodes: [],
            edges: [],
            targetTable: null,
            isReady: false,
            notReadyReason: "missing-inputs",
            configuredHopCount: 0,
            renderedHopCount: 0
        };

        if (!graph || !predictiveQuery.trim() || !modelPlanYaml.trim()) {
            return EMPTY;
        }

        const targetTable = resolveTargetTable(predictiveQuery, graph);
        if (!targetTable) {
            return { ...EMPTY, notReadyReason: "no-target-table" };
        }

        const schemaEdges = extractSchemaEdges(
            graph.groups ?? [],
            (graph.tables ?? {}) as Record<string, { pkey?: string | null }>
        );
        if (schemaEdges.length === 0) {
            return { ...EMPTY, targetTable, notReadyReason: "no-schema-edges" };
        }

        const neighborInfo = parseNeighborSampling(modelPlanYaml);
        if (!neighborInfo.hasData) {
            return { ...EMPTY, targetTable, notReadyReason: "no-hops" };
        }

        const { nodes, edges, configuredHopCount, renderedHopCount } =
            buildNeighborhood(graph, targetTable, neighborInfo);

        if (edges.length === 0) {
            return { ...EMPTY, targetTable, notReadyReason: "invalid-hops" };
        }

        return {
            nodes,
            edges,
            targetTable,
            isReady: true,
            notReadyReason: null,
            configuredHopCount,
            renderedHopCount
        };
    }, [graph, predictiveQuery, modelPlanYaml]);
}
