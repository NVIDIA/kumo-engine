// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Statistics } from "types/stats";
import { ColumnArgs } from "types/tables";

import { sTypes } from "utils/Constants";

import ColumnStatsChart, { ColumnStatsChartProps } from "../ColumnStatsChart";

const defaultProps: ColumnStatsChartProps = {
    data: {} as ColumnArgs,
    isSnapShot: false
};

const numericalDataMock: ColumnArgs = {
    stype: sTypes.Numerical,
    statistics: {
        counts: {
            total: 1000,
            num_unique: 500,
            missing_percentage: 5.5
        },
        summary_stats: [
            ["Mean", 75.5],
            ["Median", 70],
            ["Min", 0],
            ["Max", 100]
        ],
        distribution_points: [
            [0, 10],
            [10, 20],
            [20, 30]
        ]
    } as unknown as Statistics,
    sample_rows: ["10", "20", "30"]
} as ColumnArgs;

const categoricalDataMock: ColumnArgs = {
    stype: sTypes.Categorical,
    statistics: {
        counts: {
            total: 500,
            num_unique: 10,
            missing_percentage: 2
        },
        distribution_category: [
            ["Category A", 100],
            ["Category B", 80],
            ["Category C", 60]
        ]
    } as unknown as Statistics,
    sample_rows: ["Category A", "Category B"]
};

const timestampDataMock: ColumnArgs = {
    stype: sTypes.Timestamp,
    statistics: {
        counts: {
            total: 1000,
            missing_percentage: 1
        },
        distribution_date: [
            ["2024-01-01", "2024-01-31", 100],
            ["2024-02-01", "2024-02-28", 80]
        ]
    } as unknown as Statistics,
    sample_rows: ["2024-01-15", "2024-02-15"]
};

const idDataMock: ColumnArgs = {
    stype: sTypes.ID,
    statistics: {
        counts: {
            total: 1000,
            num_unique: 1000
        },
        distribution_category: [
            ["ID1", 1],
            ["ID2", 1]
        ]
    } as unknown as Statistics,
    sample_rows: ["ID1", "ID2"]
};

describe("ColumnStatsChart", () => {
    const renderColumnStatsChart = async (props: ColumnStatsChartProps) => {
        await act(async () => {
            render(<ColumnStatsChart {...props} />);
        });
    };

    const verifyStatsWithTestIds = (
        stats: Array<{ testId: string; label: string; value: string }>
    ) => {
        stats.forEach(({ testId, label, value }) => {
            const statBlock = screen.getByTestId(testId);
            const utils = within(statBlock);
            expect(utils.getByText(label)).toBeInTheDocument();
            expect(utils.getByText(value)).toBeInTheDocument();
        });
    };

    const verifyStatsInContainer = (
        container: HTMLElement,
        stats: Array<{ label: string; value: string }>
    ) => {
        stats.forEach(({ label, value }) => {
            const statBlock = within(container).getByText(label).closest("div");
            expect(statBlock).toBeInTheDocument();
            expect(within(statBlock!).getByText(value)).toBeInTheDocument();
        });
    };

    const verifyStatsContainer = () => {
        const statsContainer = screen.getByTestId("stats");
        expect(statsContainer).toBeInTheDocument();
        expect(within(statsContainer).getByText("Stats")).toBeInTheDocument();
        return statsContainer;
    };

    const verifyDistribution = (chartId: "bar-chart" | "area-chart") => {
        expect(screen.getByTestId("distribution")).toBeInTheDocument();
        expect(screen.getByText("Distribution")).toBeInTheDocument();
        expect(document.querySelector(`#${chartId}`)).toBeInTheDocument();
    };

    it("Should display error message when no statistics available", async () => {
        await renderColumnStatsChart({
            ...defaultProps,
            data: {
                stype: sTypes.Numerical,
                statistics: undefined,
                sample_rows: null
            } as ColumnArgs
        });

        expect(
            screen.getByText(
                "We are unable to display column statistics at this time"
            )
        ).toBeInTheDocument();
    });

    describe("Data type rendering", () => {
        it("Should render all stats for numerical data", async () => {
            await renderColumnStatsChart({
                ...defaultProps,
                data: numericalDataMock
            });

            const statsContainer = verifyStatsContainer();

            verifyStatsWithTestIds([
                { testId: "count", label: "Count", value: "1,000" },
                { testId: "unique", label: "Unique", value: "—" },
                { testId: "missing", label: "Missing", value: "5.500%" }
            ]);

            verifyStatsInContainer(statsContainer, [
                { label: "Mean", value: "75.50" },
                { label: "Median", value: "70" },
                { label: "Min", value: "0" },
                { label: "Max", value: "100" }
            ]);
        });

        it.each([
            {
                description: "Should render AreaChart when unique values > 20",
                numUnique: 25,
                expectedChartId: "area-chart" as const
            },
            {
                description: "Should render BarChart when unique values ≤ 20",
                numUnique: 15,
                expectedChartId: "bar-chart" as const
            }
        ])("$description", async ({ numUnique, expectedChartId }) => {
            await renderColumnStatsChart({
                ...defaultProps,
                data: {
                    ...numericalDataMock,
                    statistics: {
                        ...numericalDataMock.statistics!,
                        counts: {
                            ...numericalDataMock.statistics!.counts,
                            num_unique: numUnique
                        },
                        distribution_points: Array(numUnique)
                            .fill(0)
                            .map((_, i) => [i, i * 10])
                    }
                } as ColumnArgs
            });

            verifyDistribution(expectedChartId);
        });

        it.each([
            {
                description:
                    "Should render all stats and BarChart for categorical data",
                mockData: categoricalDataMock,
                expectedStats: [
                    { testId: "count", label: "Count", value: "500" },
                    { testId: "unique", label: "Unique", value: "10" },
                    { testId: "missing", label: "Missing", value: "2%" }
                ]
            },
            {
                description:
                    "Should render all stats and BarChart for timestamp data",
                mockData: timestampDataMock,
                expectedStats: [
                    { testId: "count", label: "Count", value: "1,000" },
                    { testId: "missing", label: "Missing", value: "1%" }
                ]
            },
            {
                description: "Should render stats and BarChart for ID type",
                mockData: idDataMock,
                expectedStats: [
                    { testId: "count", label: "Count", value: "1,000" },
                    { testId: "unique", label: "Unique", value: "1,000" }
                ]
            }
        ])("$description", async ({ mockData, expectedStats }) => {
            await renderColumnStatsChart({
                ...defaultProps,
                data: mockData
            });

            verifyStatsContainer();
            verifyStatsWithTestIds(expectedStats);
            verifyDistribution("bar-chart");
        });
    });

    describe("Edge cases", () => {
        it("Should not display Mode for ID type", async () => {
            await renderColumnStatsChart({
                ...defaultProps,
                data: {
                    ...idDataMock,
                    statistics: {
                        ...idDataMock.statistics!,
                        summary_stats: [
                            ["Mode", "some_id"]
                        ] as unknown as Statistics["summary_stats"]
                    }
                } as ColumnArgs
            });

            const stats = screen.getByTestId("stats");
            expect(within(stats).queryByText("Mode")).not.toBeInTheDocument();
        });

        it("Should not display stats with invalid values", async () => {
            await renderColumnStatsChart({
                ...defaultProps,
                data: {
                    ...numericalDataMock,
                    statistics: {
                        ...numericalDataMock.statistics!,
                        summary_stats: [
                            ["Valid", 50],
                            ["Invalid", null],
                            ["Another", undefined]
                        ] as unknown as Statistics["summary_stats"]
                    },
                    sample_rows: ["10"]
                } as ColumnArgs
            });

            const statsContainer = verifyStatsContainer();

            expect(
                within(statsContainer).getByText("Valid")
            ).toBeInTheDocument();

            expect(
                within(statsContainer).queryByText("Invalid")
            ).not.toBeInTheDocument();
            expect(
                within(statsContainer).queryByText("Another")
            ).not.toBeInTheDocument();
        });
    });

    describe("Sample section", () => {
        it.each([
            {
                description:
                    "Should render sample section with sample rows when isSnapShot is false",
                isSnapShot: false,
                shouldRender: true
            },
            {
                description:
                    "Should not render sample section when isSnapShot is true",
                isSnapShot: true,
                shouldRender: false
            }
        ])("$description", async ({ isSnapShot, shouldRender }) => {
            await renderColumnStatsChart({
                ...defaultProps,
                data: categoricalDataMock,
                isSnapShot
            });

            if (shouldRender) {
                const sampleSection = screen.getByTestId("sample");
                expect(sampleSection).toBeInTheDocument();
                expect(
                    within(sampleSection).getByText("Sample")
                ).toBeInTheDocument();

                const sampleRows = categoricalDataMock.sample_rows as string[];
                sampleRows.forEach((value) => {
                    expect(
                        within(sampleSection).getByText(value)
                    ).toBeInTheDocument();
                });
            } else {
                expect(screen.queryByTestId("sample")).not.toBeInTheDocument();
            }
        });

        it("Should display N/A for empty sample rows", async () => {
            await renderColumnStatsChart({
                ...defaultProps,
                data: {
                    ...categoricalDataMock,
                    sample_rows: null
                } as ColumnArgs
            });

            const sampleSection = screen.getByTestId("sample");
            expect(sampleSection).toBeInTheDocument();

            expect(within(sampleSection).getAllByText("N/A")).toHaveLength(1);
        });

        it("Should not pad short sample rows with N/A", async () => {
            await renderColumnStatsChart({
                ...defaultProps,
                data: {
                    ...categoricalDataMock,
                    sample_rows: ["Category A", "Category B"]
                } as ColumnArgs
            });

            const sampleSection = screen.getByTestId("sample");
            expect(
                within(sampleSection).getByText("Category A")
            ).toBeInTheDocument();
            expect(
                within(sampleSection).getByText("Category B")
            ).toBeInTheDocument();
            expect(
                within(sampleSection).queryByText("N/A")
            ).not.toBeInTheDocument();
        });

        it("Should render image samples when URLs have image extensions", async () => {
            await renderColumnStatsChart({
                ...defaultProps,
                data: {
                    ...categoricalDataMock,
                    sample_rows: [
                        "image1.jpg",
                        "image2.png",
                        "image3.jpeg",
                        "image4.gif"
                    ]
                } as ColumnArgs
            });

            const sampleSection = screen.getByTestId("sample");
            const images = within(sampleSection).getAllByRole("img");
            expect(images.length).toBeGreaterThan(0);
            expect(images[0]).toHaveAttribute("src", "image1.jpg");
        });
    });

    describe("Distribution rendering", () => {
        it("Should not render distribution when no distribution data available", async () => {
            await renderColumnStatsChart({
                ...defaultProps,
                data: {
                    ...numericalDataMock,
                    statistics: {
                        ...numericalDataMock.statistics!,
                        distribution_points: null
                    }
                } as ColumnArgs
            });

            expect(
                screen.queryByTestId("distribution")
            ).not.toBeInTheDocument();
        });

        it("Should limit categorical distribution to first 10 items", async () => {
            const largeDistribution = Array(15)
                .fill(0)
                .map((_, i) => [`Category${i}`, 100 - i]);

            await renderColumnStatsChart({
                ...defaultProps,
                data: {
                    ...categoricalDataMock,
                    statistics: {
                        ...categoricalDataMock.statistics!,
                        distribution_category: largeDistribution
                    }
                } as ColumnArgs
            });

            verifyDistribution("bar-chart");
        });

        it("Should limit numerical distribution to first 20 items", async () => {
            const largeDistribution = Array(25)
                .fill(0)
                .map((_, i) => [i, i * 10]);

            await renderColumnStatsChart({
                ...defaultProps,
                data: {
                    ...numericalDataMock,
                    statistics: {
                        ...numericalDataMock.statistics!,
                        distribution_points: largeDistribution
                    }
                } as ColumnArgs
            });

            verifyDistribution("area-chart");
        });
    });
});
