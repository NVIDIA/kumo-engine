// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { graphDetail } from "pages/Graphs/__tests__/MockResponse/GraphDetail.response";

import { JSGraph } from "types/graphs";

import { useNeighborhoodGraph } from "../useNeighborhoodGraph";

// Build a JSGraph from the existing GraphDetail fixture.
// Tables: customers (pkey: customer_id), articles (pkey: article_id),
//         transactions (no pkey — fact table).
// Schema edges: customers ↔ transactions (customer_id),
//               articles  ↔ transactions (article_id).
const graph: JSGraph = {
    id: graphDetail.id,
    name: graphDetail.name_alias,
    tables: graphDetail.graph.tables as JSGraph["tables"],
    groups: graphDetail.graph.col_groups.map((g) => ({
        columns: g.columns
    })),
    totalLinks: graphDetail.num_links,
    totalTables: graphDetail.num_tables
};

// H&M style queries
const query = "PREDICT age FOR EACH customers";

const oneHopYaml = `
neighbor_sampling:
  num_neighbors:
    - [12]
`;

const twoHopYaml = `
neighbor_sampling:
  num_neighbors:
    - [12, 13]
`;

const objectFormatYaml = `
neighbor_sampling:
  num_neighbors:
    - hop1:
        default: 15
      hop2:
        default: 20
`;

describe("useNeighborhoodGraph", () => {
    // ── missing-inputs ─────────────────────────────────────────────────────────

    it("returns missing-inputs when graph is null", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(null, query, oneHopYaml)
        );
        expect(result.current.isReady).toBe(false);
        expect(result.current.notReadyReason).toBe("missing-inputs");
        expect(result.current.nodes).toHaveLength(0);
        expect(result.current.edges).toHaveLength(0);
    });

    it("returns missing-inputs when predictiveQuery is blank", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, "   ", oneHopYaml)
        );
        expect(result.current.isReady).toBe(false);
        expect(result.current.notReadyReason).toBe("missing-inputs");
    });

    it("returns missing-inputs when modelPlanYaml is blank", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, query, "")
        );
        expect(result.current.isReady).toBe(false);
        expect(result.current.notReadyReason).toBe("missing-inputs");
    });

    // ── no-hops ────────────────────────────────────────────────────────────────

    it("returns no-hops when model plan has no neighbor_sampling", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, query, "model_type: GNN")
        );
        expect(result.current.isReady).toBe(false);
        expect(result.current.notReadyReason).toBe("no-hops");
    });

    // ── no-target-table ────────────────────────────────────────────────────────

    it("returns no-target-table when FOR EACH clause is entirely missing", () => {
        const noForEachQuery = "PREDICT x";
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, noForEachQuery, oneHopYaml)
        );
        expect(result.current.isReady).toBe(false);
        expect(result.current.notReadyReason).toBe("no-target-table");
    });

    it("is not ready when FOR EACH names a table with no schema connections", () => {
        // resolveTargetTable uses best-effort fallback, returning the identifier.
        // An unknown table has no FK edges so the neighborhood is empty → not ready.
        const unknownQuery = "PREDICT x FOR EACH unknown_table";
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, unknownQuery, oneHopYaml)
        );
        expect(result.current.isReady).toBe(false);
    });

    // ── entity node ────────────────────────────────────────────────────────────

    it("places entity (customers) at hop 0 with no neighborCount", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, query, oneHopYaml)
        );
        expect(result.current.isReady).toBe(true);
        const entity = result.current.nodes.find((n) => n.hop === 0);
        expect(entity?.table).toBe("customers");
        expect(entity?.neighborCount).toBeUndefined();
    });

    // ── hop-1 nodes ────────────────────────────────────────────────────────────

    it("includes transactions at hop 1 (customers → transactions)", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, query, oneHopYaml)
        );
        const hop1Tables = result.current.nodes
            .filter((n) => n.hop === 1)
            .map((n) => n.table);
        expect(hop1Tables).toContain("transactions");
    });

    it("sets neighborCount on hop-1 nodes from YAML default", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, query, oneHopYaml)
        );
        const hop1 = result.current.nodes.find(
            (n) => n.hop === 1 && n.table === "transactions"
        );
        expect(hop1?.neighborCount).toBe(12);
    });

    // ── edges ──────────────────────────────────────────────────────────────────

    it("produces edges from entity node to hop-1 nodes", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, query, oneHopYaml)
        );
        expect(result.current.edges.length).toBeGreaterThan(0);
        for (const edge of result.current.edges) {
            expect(edge.sourceId).toMatch(/^hop0:/);
        }
    });

    // ── two-hop graph ──────────────────────────────────────────────────────────

    it("builds hop-2 nodes (articles reachable via transactions)", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, query, twoHopYaml)
        );
        expect(result.current.isReady).toBe(true);
        const hop2Tables = result.current.nodes
            .filter((n) => n.hop === 2)
            .map((n) => n.table);
        expect(hop2Tables).toContain("articles");
    });

    it("sets hop-2 neighborCount from second array entry", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, query, twoHopYaml)
        );
        const hop2 = result.current.nodes.find(
            (n) => n.hop === 2 && n.table === "articles"
        );
        expect(hop2?.neighborCount).toBe(13);
    });

    // ── double-back prevention ─────────────────────────────────────────────────

    it("double-back: customers (has PK) does NOT appear at hop 2", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, query, twoHopYaml)
        );
        const hop2Tables = result.current.nodes
            .filter((n) => n.hop === 2)
            .map((n) => n.table);
        expect(hop2Tables).not.toContain("customers");
    });

    // ── object-format YAML ─────────────────────────────────────────────────────

    it("parses object-format num_neighbors (hop1/hop2 keys)", () => {
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, query, objectFormatYaml)
        );
        expect(result.current.isReady).toBe(true);
        const hop1 = result.current.nodes.find((n) => n.hop === 1);
        expect(hop1?.neighborCount).toBe(15);
        const hop2 = result.current.nodes.find((n) => n.hop === 2);
        expect(hop2?.neighborCount).toBe(20);
    });

    // ── configuredHopCount / renderedHopCount ──────────────────────────────────

    it("configuredHopCount equals hops in YAML, renderedHopCount equals hops with nodes", () => {
        const { result: r1 } = renderHook(() =>
            useNeighborhoodGraph(graph, query, oneHopYaml)
        );
        expect(r1.current.configuredHopCount).toBe(1);
        expect(r1.current.renderedHopCount).toBe(1);

        const { result: r2 } = renderHook(() =>
            useNeighborhoodGraph(graph, query, twoHopYaml)
        );
        expect(r2.current.configuredHopCount).toBe(2);
        expect(r2.current.renderedHopCount).toBe(2);
    });

    it("renderedHopCount < configuredHopCount when double-back prevention prunes an entire hop", () => {
        // Minimal graph: customers (PK) ↔ transactions (no PK).
        // With 2 configured hops:
        //   hop 1 → transactions (allowed, no PK)
        //   hop 2 → customers (blocked: has PK, appeared at hop 0)
        // → hop 2 is entirely empty, so renderedHopCount should be 1.
        const twoTableGraph: JSGraph = {
            totalLinks: 1,
            totalTables: 2,
            tables: {
                customers: { pkey: "customer_id" },
                transactions: { pkey: null }
            } as unknown as JSGraph["tables"],
            groups: [
                {
                    columns: [
                        { table_name: "customers", col_name: "customer_id" },
                        { table_name: "transactions", col_name: "customer_id" }
                    ]
                }
            ]
        };
        const { result } = renderHook(() =>
            useNeighborhoodGraph(
                twoTableGraph,
                "PREDICT age FOR EACH customers",
                twoHopYaml
            )
        );
        expect(result.current.configuredHopCount).toBe(2);
        expect(result.current.renderedHopCount).toBe(1);
    });

    // ── dotted FOR EACH clause ─────────────────────────────────────────────────

    it("resolves entity from dotted FOR EACH (schema.customers)", () => {
        const dottedQuery = "PREDICT age FOR EACH schema.customers";
        const { result } = renderHook(() =>
            useNeighborhoodGraph(graph, dottedQuery, oneHopYaml)
        );
        expect(result.current.isReady).toBe(true);
        expect(result.current.targetTable).toBe("customers");
    });
});
