// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
    ClassificationTargetDistribution,
    RegressionTargetDistribution,
    TimeFrameStats
} from "types/stats";

import StatsChart, {
    isClassificationTargetDistribution,
    StatsChartProps
} from "../StatsChart";

const defaultProps: StatsChartProps = {
    data: {} as TimeFrameStats,
    title: "Test Chart"
};

const mockBaseTimeFrameStats: Omit<TimeFrameStats, "target_distribution"> = {
    max_date: "2024-01-01",
    min_date: "2024-01-01",
    anchor_date: "2024-01-01",
    num_entities: 100,
    num_targets: 50,
    num_train_rows: 1000
};

const mockClassificationTargetDistribution: ClassificationTargetDistribution = {
    count: {
        Class1: 100,
        Class2: 200,
        Class3: 150
    },
    percentage: {
        Class1: 22.2,
        Class2: 44.4,
        Class3: 33.3
    }
};

const mockRegressionTargetDistribution: RegressionTargetDistribution = {
    target_mean: 50,
    target_min: 0,
    target_percentile_25: 25,
    target_median: 50,
    target_percentile_75: 75,
    target_max: 100,
    target_mode: 50,
    target_mode_count: 10,
    target_mode_percentage: 5,
    histogram: {
        bins: [0, 10, 20, 30, 40, 50],
        counts: [5, 10, 15, 20, 25, 30],
        bin_width: 10,
        num_bins: 5
    }
};

describe("StatsChart", () => {
    const renderStatsChart = async (props: StatsChartProps) => {
        await act(async () => {
            render(<StatsChart {...props} />);
        });
    };

    const expectBarChartToBePresent = () => {
        expect(document.querySelector("#bar-chart")).toBeInTheDocument();
    };

    const expectNoChartsRendered = () => {
        expect(document.querySelector("#bar-chart")).not.toBeInTheDocument();
        expect(document.querySelector("#area-chart")).not.toBeInTheDocument();
    };

    describe("isClassificationTargetDistribution helper", () => {
        it("Should identify classification distribution", () => {
            const result = isClassificationTargetDistribution(
                mockClassificationTargetDistribution
            );
            expect(result).toBe(true);
        });

        it("Should identify regression distribution", () => {
            const result = isClassificationTargetDistribution(
                mockRegressionTargetDistribution
            );
            expect(result).toBe(false);
        });

        it("Should return false when count is undefined", () => {
            const dist = {
                count: undefined,
                percentage: {}
            } as unknown as ClassificationTargetDistribution;
            expect(isClassificationTargetDistribution(dist)).toBe(false);
        });
    });

    describe("Empty states", () => {
        it("Should not render any chart when target_distribution is undefined", async () => {
            await renderStatsChart({
                ...defaultProps,
                data: {
                    ...mockBaseTimeFrameStats,
                    target_distribution: undefined
                } as unknown as TimeFrameStats
            });

            expectNoChartsRendered();
        });

        it.each([
            { description: "null", data: null },
            { description: "undefined", data: undefined }
        ])(
            "Should not render any chart when data is $description",
            async ({ data }) => {
                await renderStatsChart({
                    ...defaultProps,
                    data: data as unknown as TimeFrameStats
                });

                expectNoChartsRendered();
            }
        );
    });

    describe("Classification target distribution", () => {
        it("Should render BarChart for classification data", async () => {
            await renderStatsChart({
                ...defaultProps,
                data: {
                    ...mockBaseTimeFrameStats,
                    target_distribution: mockClassificationTargetDistribution
                } as TimeFrameStats,
                title: "Classification Distribution"
            });

            expectBarChartToBePresent();
        });

        it.each([
            {
                description: "single class",
                distribution: {
                    count: { OnlyClass: 1000 },
                    percentage: { OnlyClass: 100 }
                }
            },
            {
                description: "many classes",
                distribution: (() => {
                    const count: { [k: string]: number } = {};
                    const percentage: { [k: string]: number } = {};
                    for (let i = 0; i < 20; i++) {
                        count[`Class${i}`] = i * 10;
                        percentage[`Class${i}`] = (i * 10) / 2;
                    }
                    return { count, percentage };
                })()
            },
            {
                description: "empty counts",
                distribution: {
                    count: {},
                    percentage: {}
                }
            }
        ])(
            "Should handle classification with $description",
            async ({ distribution }) => {
                await renderStatsChart({
                    ...defaultProps,
                    data: {
                        ...mockBaseTimeFrameStats,
                        target_distribution:
                            distribution as ClassificationTargetDistribution
                    } as TimeFrameStats
                });

                expectBarChartToBePresent();
            }
        );
    });

    describe("Regression target distribution", () => {
        it("Should render BarChart for regression histogram data", async () => {
            await renderStatsChart({
                ...defaultProps,
                data: {
                    ...mockBaseTimeFrameStats,
                    target_distribution: mockRegressionTargetDistribution
                } as TimeFrameStats,
                title: "Regression Distribution"
            });

            expectBarChartToBePresent();
        });

        it("Should handle regression with decimal bin values", async () => {
            await renderStatsChart({
                ...defaultProps,
                data: {
                    ...mockBaseTimeFrameStats,
                    target_distribution: {
                        ...mockRegressionTargetDistribution,
                        target_mean: 0.5,
                        target_min: 0.1,
                        target_percentile_25: 0.3,
                        target_median: 0.5,
                        target_percentile_75: 0.7,
                        target_max: 0.9,
                        target_mode: 0.5,
                        target_mode_count: 100,
                        target_mode_percentage: 20,
                        histogram: {
                            bins: [0.123456, 0.234567, 0.345678, 0.456789],
                            counts: [25, 50, 75, 100],
                            bin_width: 0.111111,
                            num_bins: 3
                        }
                    } as RegressionTargetDistribution
                } as TimeFrameStats
            });

            expectBarChartToBePresent();
        });

        it.each([
            {
                description: "histogram is missing",
                distribution: {
                    ...mockRegressionTargetDistribution,
                    histogram: undefined
                }
            },
            {
                description: "histogram bins are missing",
                distribution: {
                    ...mockRegressionTargetDistribution,
                    histogram: {
                        counts: [10, 20, 30],
                        bin_width: 10,
                        num_bins: 3
                    }
                }
            },
            {
                description: "histogram counts are missing",
                distribution: {
                    ...mockRegressionTargetDistribution,
                    histogram: {
                        bins: [0, 10, 20, 30],
                        bin_width: 10,
                        num_bins: 3
                    }
                }
            },
            {
                description: "histogram bins and counts are empty",
                distribution: {
                    ...mockRegressionTargetDistribution,
                    histogram: {
                        bins: [],
                        counts: [],
                        bin_width: 0,
                        num_bins: 0
                    }
                }
            },
            {
                description: "histogram bins is null",
                distribution: {
                    ...mockRegressionTargetDistribution,
                    histogram: {
                        bins: null,
                        counts: [10, 20, 30],
                        bin_width: 10,
                        num_bins: 3
                    }
                }
            },
            {
                description: "histogram counts is null",
                distribution: {
                    ...mockRegressionTargetDistribution,
                    histogram: {
                        bins: [0, 10, 20, 30],
                        counts: null,
                        bin_width: 10,
                        num_bins: 3
                    }
                }
            }
        ])(
            "Should render BarChart with empty data when $description",
            async ({ distribution }) => {
                await renderStatsChart({
                    ...defaultProps,
                    data: {
                        ...mockBaseTimeFrameStats,
                        target_distribution:
                            distribution as unknown as RegressionTargetDistribution
                    } as TimeFrameStats
                });

                expectBarChartToBePresent();
            }
        );
    });

    describe("Chart dimensions", () => {
        it.each([
            {
                description: "numeric width and height",
                width: 500,
                height: 300
            },
            {
                description: "string width and height",
                width: "100%",
                height: "400px"
            }
        ])("Should apply custom $description", async ({ width, height }) => {
            await renderStatsChart({
                ...defaultProps,
                data: {
                    ...mockBaseTimeFrameStats,
                    target_distribution: {
                        count: { A: 100 },
                        percentage: { A: 100 }
                    } as ClassificationTargetDistribution
                } as TimeFrameStats,
                width,
                height
            });

            expectBarChartToBePresent();
        });
    });

    it("Should handle empty title", async () => {
        await renderStatsChart({
            ...defaultProps,
            data: {
                ...mockBaseTimeFrameStats,
                target_distribution: {
                    count: { A: 100 },
                    percentage: { A: 100 }
                } as ClassificationTargetDistribution
            } as TimeFrameStats,
            title: ""
        });

        expectBarChartToBePresent();
    });
});
