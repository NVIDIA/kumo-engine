// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DistributionChart } from "@/components/blocks/chart/distribution-chart";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { userEvent } from "storybook/test";

const meta: Meta<typeof DistributionChart> = {
    title: "Blocks/DistributionChart",
    component: DistributionChart,
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof DistributionChart>;

const xAxisData: string[] = [
    "Fri Nov 2, 2018 - Sun Dec 2, 2018",
    "Sun Dec 2, 2018 - Tue Jan 1, 2019",
    "Tue Jan 1, 2019 - Thu Jan 31, 2019",
    "Thu Jan 31, 2019 - Sat Mar 2, 2019",
    "Sat Mar 2, 2019 - Mon Apr 1, 2019",
    "Mon Apr 1, 2019 - Wed May 1, 2019",
    "Wed May 1, 2019 - Fri May 31, 2019",
    "Fri May 31, 2019 - Sun Jun 30, 2019",
    "Sun Jun 30, 2019 - Tue Jul 30, 2019",
    "Tue Jul 30, 2019 - Thu Aug 29, 2019",
    "Thu Aug 29, 2019 - Sat Sep 28, 2019",
    "Sat Sep 28, 2019 - Mon Oct 28, 2019",
    "Mon Oct 28, 2019 - Wed Nov 27, 2019",
    "Wed Nov 27, 2019 - Fri Dec 27, 2019",
    "Fri Dec 27, 2019 - Sun Jan 26, 2020",
    "Sun Jan 26, 2020 - Tue Feb 25, 2020",
    "Tue Feb 25, 2020 - Thu Mar 26, 2020",
    "Thu Mar 26, 2020 - Sat Apr 25, 2020",
    "Sat Apr 25, 2020 - Mon May 25, 2020",
    "Mon May 25, 2020 - Wed Jun 24, 2020",
    "Wed Jun 24, 2020 - Fri Jul 24, 2020",
    "Fri Jul 24, 2020 - Sun Aug 23, 2020",
    "Sun Aug 23, 2020 - Tue Sep 22, 2020"
];
const input: Record<string, (number | null)[]> = {
    "1": [
        965,
        1080,
        857,
        1149,
        884,
        1143,
        1164,
        1236,
        1242,
        991,
        1053,
        983,
        891,
        1232,
        952,
        892,
        723,
        null,
        571,
        1113,
        1252,
        1105,
        1008
    ],
    "2": [
        1572, 1221, 1341, 1244, 1246, 1389, 1368, 1610, 1531, 1133, 1290, 1336,
        1172, 1508, 1131, 1199, 1236, 1926, 1504, 1640, 1417, 1271, 1285
    ],
    "3": [
        1572, 1221, 1341, 1244, 1246, 1389, 1368, 1610, 1531, 1133, 1290, 1336,
        1172, 1508, 1131, 1199, 1236, 1926, 1504, 1640, 1417, 1271, 1285
    ],
    "4": [
        1572, 1221, 1341, 1244, 1246, 1389, 1368, 1610, 1531, 1133, 1290, 1336,
        1172, 1508, 1131, 1199, 1236, 1926, 1504, 1640, 1417, 1271, 1285
    ],
    "5": [
        1572, 1221, 1341, 1244, 1246, 1389, 1368, 1610, 1531, 1133, 1290, 1336,
        1172, 1508, 1131, 1199, 1236, 1926, 1504, 1640, 1417, 1271, 1285
    ],
    "6": [
        1572, 1221, 1341, 1244, 1246, 1389, 1368, 1610, 1531, 1133, 1290, 1336,
        1172, 1508, 1131, 1199, 1236, 1926, 1504, 1640, 1417, 1271, 1285
    ],
    "7": [
        1572, 1221, 1341, 1244, 1246, 1389, 1368, 1610, 1531, 1133, 1290, 1336,
        1172, 1508, 1131, 1199, 1236, 1926, 1504, 1640, 1417, 1271, 1285
    ],
    "8": [
        1572, 1221, 1341, 1244, 1246, 1389, 1368, 1610, 1531, 1133, 1290, 1336,
        1172, 1508, 1131, 1199, 1236, 1926, 1504, 1640, 1417, 1271, 1285
    ],
    "9": [
        1572, 1221, 1341, 1244, 1246, 1389, 1368, 1610, 1531, 1133, 1290, 1336,
        1172, 1508, 1131, 1199, 1236, 1926, 1504, 1640, 1417, 1271, 1285
    ],
    "10": [
        1572, 1221, 1341, 1244, 1246, 1389, 1368, 1610, 1531, 1133, 1290, 1336,
        1172, 1508, 1131, 1199, 1236, 1926, 1504, 1640, 1417, 1271, 1285
    ],
    "11": [
        1572, 1221, 1341, 1244, 1246, 1389, 1368, 1610, 1531, 1133, 1290, 1336,
        1172, 1508, 1131, 1199, 1236, 1926, 1504, 1640, 1417, 1271, 1285
    ],
    "": [
        9262, 9468, 9526, 9403, 9630, 9325, 9308, 9042, 9110, 9657, 9457, 9458,
        9687, 9143, 9666, 9657, 9777, 9651, 9654, 9098, 9192, 9432, 9494
    ]
};

/* const input1: Record<string, (number | null)[]> = {
    "Number of Entities": [
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33
    ]
}; */

export const BarChart: Story = {
    render: function DefaultFunc() {
        return (
            <>
                <DistributionChart
                    id={"barchart"}
                    title={"Training Table Statistics over Time"}
                    xAxisLabel={"Time Frames"}
                    yAxisLabel={"Number of Entities"}
                    chartData={input}
                    //lineChart={true}
                    xAxisData={xAxisData}
                />
            </>
        );
    },
    play: async ({ canvasElement }) => {
        await userEvent.pointer({
            target: canvasElement,
            coords: { x: 150, y: 150 }
        });
        await userEvent.pointer({
            target: canvasElement,
            coords: { x: 200, y: 100 }
        });
        await userEvent.pointer({
            target: canvasElement,
            coords: { x: 0, y: 0 }
        });
    }
};

const lineData: Record<string, (number | null)[]> = {
    "mean": [
        0.06060606060606061, 0, 0, 0.030303030303030304, 0, 0,
        0.030303030303030304, 0, 0, 0.12121212121212122, 0,
        0.030303030303030304, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    "min": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    "max": [
        2, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    "25%": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    "75%": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    "median": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    "mode": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
};

export const LineChart: Story = {
    render: function DefaultFunc() {
        return (
            <>
                <DistributionChart
                    id={"linechart"}
                    title={"Training Table Statistics over Time"}
                    xAxisLabel={"Time Frames"}
                    yAxisLabel={"Distribution of Regression Values"}
                    chartData={lineData}
                    lineChart={true}
                    xAxisData={xAxisData}
                />
            </>
        );
    },
    play: async ({ canvasElement }) => {
        await userEvent.pointer({
            target: canvasElement,
            coords: { x: 150, y: 150 }
        });
        await userEvent.pointer({
            target: canvasElement,
            coords: { x: 200, y: 100 }
        });
        await userEvent.pointer({
            target: canvasElement,
            coords: { x: 0, y: 0 }
        });
    }
};
