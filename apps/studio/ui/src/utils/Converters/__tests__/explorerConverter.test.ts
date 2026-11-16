// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    KumoapiExplainSubgraphSubgraph,
    SubgraphStoreEdges
} from "gen/v0/types";
import { describe, expect, it } from "vitest";

import { GraphXaiScores, SubgraphEdge } from "types/explorer";

import {
    createLinksFromEdges,
    extractNodesFromSubgraph,
    updateColumnsWithXaiData
} from "../explorerConverter";

import {
    anchorTime,
    columns,
    mockEdges,
    mockNodeArr,
    subgraphMockData,
    xaiData
} from "./MockResponse/mockResponses";

describe("extractNodesFromSubgraph", () => {
    it("Should return correct number of nodes", () => {
        const { nodeArr } = extractNodesFromSubgraph(
            subgraphMockData as unknown as KumoapiExplainSubgraphSubgraph
        );
        expect(nodeArr.length).toBe(4);
    });

    it.each([
        {
            description:
                "Should assign <AGGREGATED> for negative ID with null pkey",
            nodeId: "<AGGREGATED>",
            findByType: "id" as const,
            expectedDefined: true,
            expectedValue: "<AGGREGATED>"
        },
        {
            description: "Should fallback to .id if pkey_name is missing",
            nodeId: "node_0",
            findByType: "nodeId" as const,
            expectedIdName: "",
            expectedValue: 0
        },
        {
            description: "Should correctly assign tableName for ARTICLES",
            nodeId: "node_12",
            findByType: "nodeId" as const,
            expectedTableName: "ARTICLES"
        },
        {
            description: "Should correctly assign tableName for CUSTOMERS",
            nodeId: "node_2",
            findByType: "nodeId" as const,
            expectedTableName: "CUSTOMERS"
        },
        {
            description: "Should calculate totalScore correctly",
            nodeId: "node_0",
            findByType: "nodeId" as const,
            expectedTotalScore: 1.05
        },
        {
            description: "Should map grad_score from score for each column",
            nodeId: "node_12",
            findByType: "nodeId" as const,
            expectedGradScore: 0.1
        }
    ])(
        "$description",
        ({
            nodeId,
            findByType,
            expectedDefined,
            expectedValue,
            expectedIdName,
            expectedTableName,
            expectedTotalScore,
            expectedGradScore
        }) => {
            const { nodeArr } = extractNodesFromSubgraph(
                subgraphMockData as unknown as KumoapiExplainSubgraphSubgraph
            );

            const node = nodeArr.find((n) =>
                findByType === "id" ? n.id === nodeId : n.nodeId === nodeId
            );

            if (expectedDefined !== undefined) {
                expect(node).toBeDefined();
            }
            if (expectedValue !== undefined) {
                expect(node?.data.value).toBe(expectedValue);
            }
            if (expectedIdName !== undefined) {
                expect(node?.data.idName).toBe(expectedIdName);
            }
            if (expectedTableName !== undefined) {
                expect(node?.data.tableName).toBe(expectedTableName);
            }
            if (expectedTotalScore !== undefined) {
                expect(node?.data.totalScore).toBeCloseTo(expectedTotalScore);
            }
            if (expectedGradScore !== undefined) {
                const gradScoreValues = node?.data.cols.map(
                    (col) => col.grad_score
                );
                expect(gradScoreValues?.[0]).toBe(expectedGradScore);
            }
        }
    );
});

describe("createLinksFromEdges", () => {
    it("Should create correct number of graph edges", () => {
        const links = createLinksFromEdges(
            mockEdges as SubgraphStoreEdges,
            mockNodeArr
        );
        expect(links.length).toBe(2);
    });

    it("Should generate links with correct source and target", () => {
        const links = createLinksFromEdges(
            mockEdges as SubgraphStoreEdges,
            mockNodeArr
        );
        expect(links).toEqual([
            {
                id: "e-1",
                source: "0",
                target: "2",
                type: "straight"
            },
            {
                id: "e-2",
                source: "12",
                target: "0",
                type: "straight"
            }
        ]);
    });

    it("Should skip invalid source or target nodeIds", () => {
        const edgesWithInvalidTarget: { [key: string]: SubgraphEdge } = {
            "0": {
                customer_id: [999]
            }
        };

        const links = createLinksFromEdges(
            edgesWithInvalidTarget as SubgraphStoreEdges,
            mockNodeArr
        );
        expect(links).toEqual([]);
    });

    it("Should assign unique edge IDs", () => {
        const links = createLinksFromEdges(
            mockEdges as SubgraphStoreEdges,
            mockNodeArr
        );
        const ids = links.map((e) => e.id);
        expect(new Set(ids).size).toBe(ids.length);
    });
});

describe("updateColumnsWithXaiData", () => {
    it("updates XAI predictions and global avg on matching columns", () => {
        const result = updateColumnsWithXaiData(
            columns,
            "Ratings",
            xaiData,
            anchorTime
        );
        expect(result[0].isXaiUpdated).toBe(true);
        expect(result[0].xai_prediction).toBeCloseTo(4.849);
        expect(result[0].xai_global_avg).toBeCloseTo(4.82996);

        expect(result[1].isXaiUpdated).toBe(true);
        expect(result[1].xai_prediction).toBeCloseTo(4.851);
        expect(result[1].xai_global_avg).toBeCloseTo(4.82972283);
    });

    it("Should not crash if xaiData does not have a matching column", () => {
        const columns = [
            {
                table_name: "Ratings_test",
                name: "non_existing_column",
                value: 2,
                stype: "numerical",
                score: 0.1,
                grad_score: 0.1,
                isXaiUpdated: false
            }
        ];

        const xaiData: GraphXaiScores = {
            "Ratings_test.timestamp_(1_hop)": {
                globalAvgScore: 4.8,
                labels: [],
                predictions: []
            }
        };

        const updated = updateColumnsWithXaiData(
            columns,
            "Ratings_test",
            xaiData,
            "Mon Jun 23, 2014"
        );

        expect(updated[0].isXaiUpdated).toBe(true);
        expect(updated[0].xai_prediction).toBeUndefined();
        expect(updated[0].xai_global_avg).toBeUndefined();
    });

    it("Should handle tableName with dot notation", () => {
        const columns = [
            {
                table_name: "schema.Ratings",
                name: "rating",
                value: 5,
                stype: "numerical",
                score: 0.5,
                grad_score: 0.5,
                isXaiUpdated: false
            }
        ];

        const xaiData: GraphXaiScores = {
            "Ratings.rating_(1_hop)": {
                globalAvgScore: 4.5,
                labels: ["(0 - 3)", "(3 - 5)", "(5 - 7)"],
                predictions: [4.1, 4.5, 4.9]
            }
        };

        const updated = updateColumnsWithXaiData(
            columns,
            "schema.Ratings",
            xaiData,
            "Mon Jun 23, 2014"
        );

        expect(updated[0].isXaiUpdated).toBe(true);
        expect(updated[0].xai_prediction).toBe(4.5);
    });

    it("Should select lowest hop when multiple hops exist", () => {
        const columns = [
            {
                table_name: "Ratings",
                name: "rating",
                value: 5,
                stype: "numerical",
                score: 0.5,
                grad_score: 0.5,
                isXaiUpdated: false
            }
        ];

        const xaiData: GraphXaiScores = {
            "Ratings.rating_(3_hop)": {
                globalAvgScore: 4.3,
                labels: ["(0 - 5)", "(5 - 10)"],
                predictions: [4.3, 4.7]
            },
            "Ratings.rating_(1_hop)": {
                globalAvgScore: 4.5,
                labels: ["(0 - 3)", "(3 - 5)", "(5 - 7)"],
                predictions: [4.1, 4.5, 4.9]
            },
            "Ratings.rating_(2_hop)": {
                globalAvgScore: 4.4,
                labels: ["(0 - 5)", "(5 - 10)"],
                predictions: [4.4, 4.8]
            }
        };

        const updated = updateColumnsWithXaiData(
            columns,
            "Ratings",
            xaiData,
            "Mon Jun 23, 2014"
        );

        expect(updated[0].xai_global_avg).toBe(4.5);
        expect(updated[0].xai_prediction).toBe(4.5);
    });

    it.each([
        {
            description: "Should handle categorical stype with direct match",
            tableName: "Products",
            columnName: "category",
            value: "Electronics",
            stype: "categorical",
            xaiKey: "Products.category",
            globalAvgScore: 4.6,
            labels: ["Books", "Electronics", "Clothing"],
            predictions: [4.2, 4.8, 4.4],
            expectedPrediction: 4.8
        },
        {
            description:
                "Should handle categorical stype with fallback to other/n/a",
            tableName: "Products",
            columnName: "category",
            value: "UnknownCategory",
            stype: "categorical",
            xaiKey: "Products.category",
            globalAvgScore: 4.6,
            labels: ["Books", "Electronics", "Other"],
            predictions: [4.2, 4.8, 4.3],
            expectedPrediction: 4.3
        },
        {
            description: "Should handle ID stype",
            tableName: "Users",
            columnName: "user_id",
            value: 123,
            stype: "ID",
            xaiKey: "Users.user_id",
            globalAvgScore: 4.7,
            labels: [123, 456, 789],
            predictions: [4.9, 4.5, 4.7],
            expectedPrediction: 4.9
        }
    ])(
        "$description",
        ({
            tableName,
            columnName,
            value,
            stype,
            xaiKey,
            globalAvgScore,
            labels,
            predictions,
            expectedPrediction
        }) => {
            const columns = [
                {
                    table_name: tableName,
                    name: columnName,
                    value: value,
                    stype: stype,
                    score: 0.8,
                    grad_score: 0.8,
                    isXaiUpdated: false
                }
            ];

            const xaiData: GraphXaiScores = {
                [xaiKey]: {
                    globalAvgScore: globalAvgScore,
                    labels: labels,
                    predictions: predictions
                }
            };

            const updated = updateColumnsWithXaiData(
                columns,
                tableName,
                xaiData,
                "Mon Jun 23, 2014"
            );

            expect(updated[0].xai_prediction).toBe(expectedPrediction);
        }
    );

    it("Should handle timestamp stype with date calculation", () => {
        const columns = [
            {
                table_name: "Events",
                name: "event_date",
                value: "2014-01-01",
                stype: "timestamp",
                score: 0.7,
                grad_score: 0.7,
                isXaiUpdated: false
            }
        ];

        const xaiData: GraphXaiScores = {
            "Events.event_date": {
                globalAvgScore: 4.55,
                labels: ["(0 - 3) months", "(3 - 6) months", "(6 - 12) months"],
                predictions: [4.9, 4.6, 4.3]
            }
        };

        const updated = updateColumnsWithXaiData(
            columns,
            "Events",
            xaiData,
            "Mon Jun 23, 2014"
        );

        expect(updated[0].isXaiUpdated).toBe(true);
        expect(updated[0].xai_prediction).toBeDefined();
    });

    it("Should handle timestamp stype with invalid date gracefully", () => {
        const columns = [
            {
                table_name: "Events",
                name: "event_date",
                value: "invalid-date",
                stype: "timestamp",
                score: 0.7,
                grad_score: 0.7,
                isXaiUpdated: false
            }
        ];

        const xaiData: GraphXaiScores = {
            "Events.event_date": {
                globalAvgScore: 4.55,
                labels: ["(0 - 3) months", "(3 - 6) months"],
                predictions: [4.9, 4.6]
            }
        };

        const updated = updateColumnsWithXaiData(
            columns,
            "Events",
            xaiData,
            "Mon Jun 23, 2014"
        );

        expect(updated[0].isXaiUpdated).toBe(true);
        expect(updated[0].xai_prediction).toBeUndefined();
    });

    it.each([
        {
            description: "Should handle numerical stype with range matching",
            value: 150,
            labels: ["(0 - 100)", "(100 - 200)", "(200 - 500)"],
            predictions: [4.2, 4.7, 5.0],
            expectedPrediction: 4.7
        },
        {
            description: "Should handle numerical value less than first range",
            value: 5,
            labels: ["(10 - 100)", "(100 - 200)"],
            predictions: [4.2, 4.7],
            expectedPrediction: 4.2
        },
        {
            description:
                "Should handle numerical value greater than last range",
            value: 600,
            labels: ["(0 - 100)", "(100 - 200)"],
            predictions: [4.2, 4.7],
            expectedPrediction: 4.7
        }
    ])("$description", ({ value, labels, predictions, expectedPrediction }) => {
        const columns = [
            {
                table_name: "Sales",
                name: "amount",
                value: value,
                stype: "numerical",
                score: 0.9,
                grad_score: 0.9,
                isXaiUpdated: false
            }
        ];

        const xaiData: GraphXaiScores = {
            "Sales.amount": {
                globalAvgScore: 4.65,
                labels: labels,
                predictions: predictions
            }
        };

        const updated = updateColumnsWithXaiData(
            columns,
            "Sales",
            xaiData,
            "Mon Jun 23, 2014"
        );

        expect(updated[0].xai_prediction).toBe(expectedPrediction);
    });

    it("Should skip xaiData entries with less than 2 table parts", () => {
        const columns = [
            {
                table_name: "Products",
                name: "price",
                value: 50,
                stype: "numerical",
                score: 0.5,
                grad_score: 0.5,
                isXaiUpdated: false
            }
        ];

        const xaiData: GraphXaiScores = {
            InvalidFormat: {
                globalAvgScore: 4.5,
                labels: ["(0 - 100)"],
                predictions: [4.5]
            },
            "Products.price": {
                globalAvgScore: 4.7,
                labels: ["(0 - 100)", "(100 - 200)"],
                predictions: [4.5, 4.9]
            }
        };

        const updated = updateColumnsWithXaiData(
            columns,
            "Products",
            xaiData,
            "Mon Jun 23, 2014"
        );

        expect(updated[0].xai_global_avg).toBe(4.7);
    });

    it("Should handle unknown stype with default case", () => {
        const columns = [
            {
                table_name: "Custom",
                name: "custom_field",
                value: "test_value",
                stype: "unknown_type",
                score: 0.5,
                grad_score: 0.5,
                isXaiUpdated: false
            }
        ];

        const xaiData: GraphXaiScores = {
            "Custom.custom_field": {
                globalAvgScore: 4.6,
                labels: ["test_value", "other_value"],
                predictions: [4.8, 4.4]
            }
        };

        const updated = updateColumnsWithXaiData(
            columns,
            "Custom",
            xaiData,
            "Mon Jun 23, 2014"
        );

        expect(updated[0].xai_prediction).toBe(4.8);
        expect(updated[0].xai_global_avg).toBe(4.6);
    });
});
