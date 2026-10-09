// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, within } from "@testing-library/react";

import { TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";

import TimeframeChart, {
    TimeframeChartProps
} from "../TimeframeChart/TimeframeChart";

import { timeFrameStatsRes } from "./MockResponses/TimeFrameStatsResponse";

const defaultProps: TimeframeChartProps = {
    queryStatus: jobStatus.Done,
    taskType: TaskType.REGRESSION,
    timeFrameStats: timeFrameStatsRes,
    noData: false,
    show: true
};

describe("TimeframeChart", () => {
    const renderTimeframeChart = async (props: TimeframeChartProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <TimeframeChart {...props} />
                </MemoryRouter>
            );
        });
    };

    it("Should not render the chart elements when show is false", async () => {
        await renderTimeframeChart({ ...defaultProps, show: false });

        const trainTableChart = document.getElementById(
            "train-table-stat-chart"
        );
        expect(trainTableChart).not.toBeInTheDocument();

        const trainTableChartEntities = document.getElementById(
            "train-table-stat-chart-entities"
        );
        expect(trainTableChartEntities).not.toBeInTheDocument();
    });

    it("Should render train chart and entities chart with correct axis titles", async () => {
        await renderTimeframeChart(defaultProps);

        const trainTableChart = document.getElementById(
            "train-table-stat-chart"
        );
        const trainTableChartEntities = document.getElementById(
            "train-table-stat-chart-entities"
        );

        expect(trainTableChart).toBeInTheDocument();
        expect(trainTableChartEntities).toBeInTheDocument();

        expect(
            await within(trainTableChart!).findByText(
                "Distribution of Regression Values"
            )
        ).toBeInTheDocument();

        expect(
            within(trainTableChart!).queryAllByText("Time Frames").length
        ).toBeGreaterThan(0);

        expect(
            within(trainTableChartEntities!).queryAllByText(
                "Number of Entities"
            ).length
        ).toBeGreaterThan(0);

        expect(
            within(trainTableChartEntities!).queryAllByText("Time Frames")
                .length
        ).toBeGreaterThan(0);
    });

    it("Should render empty state when no data is set as true and queryStatus is Running", async () => {
        await renderTimeframeChart({
            ...defaultProps,
            noData: true,
            timeFrameStats: [],
            queryStatus: jobStatus.Running
        });

        expect(screen.getByText("No data found!")).toBeInTheDocument();
    });

    it("Should show loading skeleton when queryStatus is Done and noData is true", async () => {
        await renderTimeframeChart({
            ...defaultProps,
            noData: true,
            timeFrameStats: []
        });

        expect(
            document.querySelector('[data-slot="skeleton"]')
        ).toBeInTheDocument();
    });

    it.each([
        {
            name: "binary classification",
            taskType: TaskType.BINARY_CLASSIFICATION,
            expectedTexts: [
                "Time Frames",
                "Count of Entities with a Given Value"
            ]
        },
        {
            name: "multilabel classification",
            taskType: TaskType.MULTILABEL_CLASSIFICATION,
            expectedTexts: ["Distribution of Labels", "Number of Entities"]
        },
        {
            name: "link prediction",
            taskType: TaskType.LINK_PREDICTION,
            expectedTexts: ["Number of Entities"]
        },
        {
            name: "forecasting",
            taskType: TaskType.FORECASTING,
            expectedTexts: ["Distribution of Forecasting Values", "Time Frames"]
        }
    ])(
        "Should render correct chart labels for $name task",
        async ({ taskType, expectedTexts }) => {
            await renderTimeframeChart({
                ...defaultProps,
                taskType
            });

            for (const text of expectedTexts) {
                expect(
                    (await screen.findAllByText(text)).length
                ).toBeGreaterThan(0);
            }
        }
    );
});
