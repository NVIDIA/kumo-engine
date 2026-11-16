// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScalarChart } from "@/components/blocks/chart/scalar-chart";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { useEffect, useState } from "react";
import { userEvent, waitFor } from "storybook/test";
import { createSeededRandom } from "@/stories/utils/seeded-random";

const meta: Meta<typeof ScalarChart> = {
    title: "Blocks/ScalarChart",
    component: ScalarChart,
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof ScalarChart>;

const seededRandom = createSeededRandom();

function generateExperimentData(numExperiments: number) {
    const experiments: { [key: string]: { xData: number[]; yData: number[] } } =
        {};

    for (let i = 0; i < numExperiments; i++) {
        const numPoints = Math.floor(seededRandom() * 5) + 3;
        const startEpoch = Math.floor(seededRandom() * 3);

        const xData: number[] = [];
        const yData: number[] = [];

        let currentValue = 0.7 + seededRandom() * 0.2;

        for (let j = 0; j < numPoints; j++) {
            xData.push(startEpoch + j);
            currentValue = currentValue + (seededRandom() - 0.3) * 0.1;
            currentValue = Math.max(0.7, Math.min(0.99, currentValue));
            yData.push(currentValue);
        }

        experiments[i.toString()] = { xData, yData };
    }

    return experiments;
}

const defaultChartData =
    // Data for more charts
    {
        "abc": {
            "0": {
                "xData": [0, 2],
                "yData": [0.8710181713104248, 0.99]
            },
            "1": {
                "xData": [2, 3, 4, 5, 6],
                "yData": [
                    0.8285988569259644, 0.9140286445617676, 0.9475704431533813,
                    0.9706845283508301, 0.9793838262557983
                ]
            }
        },
        "xyz": {
            "0": {
                "xData": [2, 3, 4, 5],
                "yData": [
                    0.8710181713104248, 0.9449172616004944, 0.9815854430198669,
                    0.9867959022521973
                ]
            },
            "1": {
                "xData": [2, 3, 4, 5, 6],
                "yData": [
                    0.8285988569259644, 0.9140286445617676, 0.9475704431533813,
                    0.9706845283508301, 0.9793838262557983
                ]
            }
        }
    };

const largeExperimentChartData = {
    "train": generateExperimentData(20),
    "val": generateExperimentData(20)
};

export const DefaultLinechart: Story = {
    render: function DefaultFunc() {
        return (
            <>
                <ScalarChart
                    id={"linechart"}
                    tag={"Loss"}
                    xAxisLabel={"Epoch"}
                    yAxisLabel={"Loss"}
                    dataDict={defaultChartData}
                />
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const svg = await waitFor(() => {
            const el = canvasElement.querySelector("svg");
            if (!el) throw new Error("svg not found");
            return el;
        });
        await userEvent.pointer({ target: svg, coords: { x: 150, y: 150 } });
        await userEvent.pointer({ target: svg, coords: { x: 200, y: 100 } });
        await userEvent.pointer({ target: svg, coords: { x: 0, y: 0 } });

        const selectAllCheckbox = await waitFor(() => {
            const el = canvasElement.querySelector(
                "[id$='-select-all']"
            ) as HTMLElement;
            if (!el) throw new Error("select-all checkbox not found");
            return el;
        });
        await userEvent.click(selectAllCheckbox);
        await userEvent.click(selectAllCheckbox);

        const firstLegendCheckbox = await waitFor(() => {
            const el = canvasElement.querySelector(
                "[id$='-legend-0']"
            ) as HTMLElement;
            if (!el) throw new Error("legend-0 checkbox not found");
            return el;
        });
        await userEvent.click(firstLegendCheckbox);
        await userEvent.click(firstLegendCheckbox);
    }
};

export const LargeExperimentChart: Story = {
    render: function LargeExperimentFunc() {
        return (
            <>
                <ScalarChart
                    id={"large-experiment-chart"}
                    tag={"Loss"}
                    xAxisLabel={"Epoch"}
                    yAxisLabel={"Loss"}
                    dataDict={largeExperimentChartData}
                />
            </>
        );
    }
};

const chartData: {
    "train": { [key: string]: { "xData": number[]; "yData": number[] } };
    "val": { [key: string]: { "xData": number[]; "yData": number[] } };
} = {
    "train": {
        "0": {
            "xData": [2, 3, 4, 5],
            "yData": [
                0.8710181713104248, 0.9449172616004944, 0.9815854430198669,
                0.9867959022521973
            ]
        },
        "1": {
            "xData": [2, 3, 4, 5, 6],
            "yData": [
                0.8285988569259644, 0.9140286445617676, 0.9475704431533813,
                0.9706845283508301, 0.9793838262557983
            ]
        },
        "2": {
            "xData": [1, 2, 3, 4, 5, 6],
            "yData": [
                0.8359747529029846, 0.8523208713531494, 0.850383152961731,
                0.9962378001213074, 0.9661967921257019, 0.8761967921257019
            ]
        }
    },
    "val": {
        "0": {
            "xData": [3, 4, 5, 6],
            "yData": [
                0.9128879904747009, 0.8960269689559937, 0.8654555678367615,
                0.8423555493354797
            ]
        },
        "1": {
            "xData": [3, 4, 5, 6, 7],
            "yData": [
                0.8359747529029846, 0.8923208713531494, 0.860383152961731,
                0.8862378001213074, 0.8861967921257019
            ]
        },
        "2": {
            "xData": [1, 2, 3, 4, 5, 6],
            "yData": [
                0.8359747529029846, 0.8523208713531494, 0.850383152961731,
                0.9962378001213074, 0.9661967921257019, 0.8761967921257019
            ]
        }
    }
};

export const LineChart: Story = {
    render: function DefaultFunc() {
        const [data, setData] = useState(chartData);

        useEffect(() => {
            setTimeout(() => {
                const nData = { ...chartData };
                nData.train["3"] = { ...nData.train["2"] };
                nData.val["3"] = { ...nData.val["2"] };
                nData.train["3"].yData = [
                    0.7359747529029846, 0.7523208713531494, 0.850383152961731,
                    0.7962378001213074, 0.9661967921257019, 0.8861967921257019
                ];
                nData.val["3"].yData = [
                    0.7359747529029846, 0.7523208713531494, 0.850383152961731,
                    0.7962378001213074, 0.9661967921257019, 0.8861967921257019
                ];
                setData({ ...nData });
            }, 5000);
        }, []);

        return (
            <>
                <ScalarChart
                    id={"linechart"}
                    tag={"Loss"}
                    xAxisLabel={"Epoch"}
                    yAxisLabel={"Loss"}
                    dataDict={data}
                />
            </>
        );
    }
};
