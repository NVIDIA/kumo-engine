// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { describe, expect, it, vi } from "vitest";

import { QueryType } from "types/predictiveQueries";
import { RegressionTargetDistribution, TimeFrameStats } from "types/stats";

import { RegressionStatsColumns } from "../RegressionStats";

type ColumnDefinition = ReturnType<typeof RegressionStatsColumns>[number];

const mockTargetDistribution: RegressionTargetDistribution = {
    target_mean: 150.5,
    target_min: 10,
    target_percentile_25: 50,
    target_median: 100,
    target_percentile_75: 200,
    target_max: 500,
    target_mode: 75,
    target_mode_count: 10,
    target_mode_percentage: 0.25,
    histogram: {
        bins: [0, 10, 20],
        counts: [5, 10, 15],
        bin_width: 10,
        num_bins: 3
    }
};

const createMockRow = (
    overrides?: Partial<TimeFrameStats>
): Row<TimeFrameStats> =>
    ({
        original: {
            num_entities: 1000,
            num_train_rows: 1000,
            num_targets: 0,
            max_date: "",
            min_date: "",
            anchor_date: "",
            target_distribution: mockTargetDistribution,
            ...overrides
        }
    }) as Row<TimeFrameStats>;

const renderCell = (column: ColumnDefinition, row: Row<TimeFrameStats>) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    return column.cell({ row } as CellContext<TimeFrameStats, number>);
};

const getColumn = (columns: ColumnDefinition[], id: string) => {
    const column = columns.find((col) => col.id === id);
    if (!column) {
        throw new Error(`Column with id "${id}" not found`);
    }
    return column;
};

describe("RegressionStatsColumns", () => {
    describe("Column Configuration", () => {
        it("Should generate base columns without distribution", () => {
            const columns = RegressionStatsColumns(
                false,
                QueryType.TEMPORAL,
                false
            );

            expect(columns).toHaveLength(10);

            const columnIds = columns.map((col) => col.id);
            expect(columnIds).toContain("num_entities");
            expect(columnIds).toContain("target_distribution.target_mean");
            expect(columnIds).toContain("target_distribution.target_mode");
            expect(columnIds).toContain("target_distribution.target_min");
            expect(columnIds).toContain(
                "target_distribution.target_percentile_25"
            );
            expect(columnIds).toContain("target_distribution.target_median");
            expect(columnIds).toContain(
                "target_distribution.target_percentile_75"
            );
            expect(columnIds).toContain("target_distribution.target_max");
        });

        it("Should include expand column when distributionAvailable is true", () => {
            const columns = RegressionStatsColumns(
                false,
                QueryType.TEMPORAL,
                true
            );

            expect(columns).toHaveLength(11);
            expect(columns[10].id).toBe("expand");
            expect(columns[10].header).toBe("Target Distribution");
            expect(columns[10].size).toBe(150);
        });

        it("Should not include expand column when distributionAvailable is false", () => {
            const columns = RegressionStatsColumns(
                false,
                QueryType.TEMPORAL,
                false
            );
            const expandColumn = columns.find((col) => col.id === "expand");

            expect(expandColumn).toBeUndefined();
        });

        it("Should include common training table stats columns", () => {
            const columns = RegressionStatsColumns(
                true,
                QueryType.TEMPORAL,
                false
            );

            const columnIds = columns.map((col) => col.id);
            expect(columnIds).toContain("data_split");
            expect(columnIds).toContain("min_date");
            expect(columnIds).toContain("num_train_rows");
        });

        it("Should work with temporal and non-temporal query types", () => {
            const temporalColumns = RegressionStatsColumns(
                false,
                QueryType.TEMPORAL,
                false
            );
            const nonTemporalColumns = RegressionStatsColumns(
                false,
                "non-temporal",
                false
            );

            expect(temporalColumns).toBeDefined();
            expect(temporalColumns.length).toBeGreaterThan(0);
            expect(nonTemporalColumns).toBeDefined();
            expect(nonTemporalColumns.length).toBeGreaterThan(0);
        });

        it("Should work with split stats enabled and disabled", () => {
            const withSplit = RegressionStatsColumns(
                true,
                QueryType.TEMPORAL,
                false
            );
            const withoutSplit = RegressionStatsColumns(
                false,
                QueryType.TEMPORAL,
                false
            );

            expect(withSplit).toBeDefined();
            expect(withSplit.length).toBeGreaterThan(0);
            expect(withoutSplit).toBeDefined();
            expect(withoutSplit.length).toBeGreaterThan(0);
        });

        it("Should work with default isSplitStats parameter", () => {
            const columns = RegressionStatsColumns(
                undefined,
                QueryType.TEMPORAL,
                false
            );

            expect(columns).toBeDefined();
            expect(columns.length).toBeGreaterThan(0);
        });
    });

    describe("Column Properties", () => {
        const columns = RegressionStatsColumns(
            false,
            QueryType.TEMPORAL,
            false
        );

        const columnConfig = [
            {
                id: "num_entities",
                header: undefined,
                size: undefined,
                accessorKey: "num_entities"
            },
            {
                id: "target_distribution.target_mean",
                header: "Mean",
                size: 50,
                accessorKey: "target_distribution.target_mean"
            },
            {
                id: "target_distribution.target_mode",
                header: "Mode",
                size: 75,
                accessorKey: "target_distribution.target_mode"
            },
            {
                id: "target_distribution.target_min",
                header: "Min",
                size: 50,
                accessorKey: "target_distribution.target_min"
            },
            {
                id: "target_distribution.target_percentile_25",
                header: "25%",
                size: 50,
                accessorKey: "target_distribution.target_percentile_25"
            },
            {
                id: "target_distribution.target_median",
                header: "Median",
                size: 50,
                accessorKey: "target_distribution.target_median"
            },
            {
                id: "target_distribution.target_percentile_75",
                header: "75%",
                size: 50,
                accessorKey: "target_distribution.target_percentile_75"
            },
            {
                id: "target_distribution.target_max",
                header: "Max",
                size: 50,
                accessorKey: "target_distribution.target_max"
            }
        ];

        columnConfig.forEach(({ id, header, size, accessorKey }) => {
            it(`Should have correct configuration for ${id}`, () => {
                const column = columns.find((col) => col.id === id);

                expect(column).toBeDefined();
                expect(column?.id).toBe(id);
                if (header) expect(column?.header).toBe(header);
                if (size) expect(column?.size).toBe(size);
                expect(column?.id).toBe(accessorKey);
            });
        });
    });

    describe("Cell Rendering", () => {
        const columns = RegressionStatsColumns(
            false,
            QueryType.TEMPORAL,
            false
        );

        describe("Num Entities Column", () => {
            it("Should render num_entities cell correctly", () => {
                const entitiesColumn = getColumn(columns, "num_entities");
                const mockRow = createMockRow({ num_entities: 12345 });

                const cellValue = renderCell(entitiesColumn, mockRow);

                expect(cellValue).toBeDefined();
            });

            it("Should handle large numbers in entities", () => {
                const entitiesColumn = getColumn(columns, "num_entities");
                const mockRow = createMockRow({ num_entities: 9999999 });

                const cellValue = renderCell(entitiesColumn, mockRow);

                expect(cellValue).toBeDefined();
            });
        });

        describe("Distribution Columns", () => {
            const distributionColumns = [
                "target_distribution.target_mean",
                "target_distribution.target_mode",
                "target_distribution.target_min",
                "target_distribution.target_percentile_25",
                "target_distribution.target_median",
                "target_distribution.target_percentile_75",
                "target_distribution.target_max"
            ];

            it("Should render all distribution columns correctly", () => {
                distributionColumns.forEach((columnId) => {
                    const column = getColumn(columns, columnId);
                    const mockRow = createMockRow();

                    const cellValue = renderCell(column, mockRow);

                    expect(cellValue).toBeDefined();
                });
            });

            it("Should handle missing target_distribution gracefully", () => {
                const meanColumn = getColumn(
                    columns,
                    "target_distribution.target_mean"
                );
                const mockRow = createMockRow({
                    target_distribution: {} as RegressionTargetDistribution
                });

                const cellValue = renderCell(meanColumn, mockRow);

                expect(cellValue).toBeDefined();
            });

            it("Should handle edge case values in distribution", () => {
                const edgeCases = [
                    {
                        columnId: "target_distribution.target_min",
                        distribution: {
                            ...mockTargetDistribution,
                            target_min: -100
                        }
                    },
                    {
                        columnId: "target_distribution.target_mean",
                        distribution: {
                            ...mockTargetDistribution,
                            target_mean: 0
                        }
                    },
                    {
                        columnId: "target_distribution.target_max",
                        distribution: {
                            ...mockTargetDistribution,
                            target_max: 999999999
                        }
                    },
                    {
                        columnId: "target_distribution.target_mode",
                        distribution: {
                            ...mockTargetDistribution,
                            target_mode_percentage: 0
                        }
                    }
                ];

                edgeCases.forEach(({ columnId, distribution }) => {
                    const column = getColumn(columns, columnId);
                    const mockRow = createMockRow({
                        target_distribution: distribution
                    });

                    const cellValue = renderCell(column, mockRow);

                    expect(cellValue).toBeDefined();
                });
            });
        });
    });

    describe("Expand Column", () => {
        const columns = RegressionStatsColumns(false, QueryType.TEMPORAL, true);
        const expandColumn = getColumn(columns, "expand");

        it("Should render expand button with correct text when collapsed", () => {
            const collapsedRow = {
                original: createMockRow().original,
                getIsExpanded: () => false,
                toggleExpanded: vi.fn()
            } as unknown as Row<TimeFrameStats>;

            const button = renderCell(expandColumn, collapsedRow);

            expect(button.props.children).toBe("See Details");
        });

        it("Should render expand button with correct text when expanded", () => {
            const expandedRow = {
                original: createMockRow().original,
                getIsExpanded: () => true,
                toggleExpanded: vi.fn()
            } as unknown as Row<TimeFrameStats>;

            const button = renderCell(expandColumn, expandedRow);

            expect(button.props.children).toBe("Close Details");
        });

        it("Should call toggleExpanded when expand button is clicked", () => {
            const toggleExpandedMock = vi.fn();
            const mockRow = {
                original: createMockRow().original,
                getIsExpanded: () => false,
                toggleExpanded: toggleExpandedMock
            } as unknown as Row<TimeFrameStats>;

            const button = renderCell(expandColumn, mockRow);
            const mockEvent = {
                stopPropagation: vi.fn()
            } as unknown as React.MouseEvent<HTMLButtonElement>;

            button.props.onClick(mockEvent);

            expect(mockEvent.stopPropagation).toHaveBeenCalled();
            expect(toggleExpandedMock).toHaveBeenCalled();
        });
    });
});
