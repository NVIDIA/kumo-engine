// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";
import { DivergingBarChart } from "@/components/blocks/chart/diverging-bar-chart";

const meta: Meta<typeof DivergingBarChart> = {
    title: "Blocks/DivergingBarChart",
    component: DivergingBarChart,
    parameters: {
        layout: "centered"
    },
    tags: ["autodocs"],
    argTypes: {
        id: { control: "text" },
        xAxisLabels: { control: "object" },
        dataSetTruth: { control: "object" },
        dataSetPred: { control: "object" },
        populationRatio: { control: "object" },
        globalAvgScore: { control: "number" },
        totalRecords: { control: "number" },
        pageSize: { control: "number" },
        pageIndex: { control: "number" },
        onGotoPage: { action: "pageChanged" },
        dropdown_menus: { control: false }
    }
};

export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        id: "bar-chart",
        xAxisLabels: ["[20 - 25.5]", "(25.5 - 47.5]", "(47.5 - 61]"],
        dataSetTruth: ["1", "1", "0"],
        dataSetPred: ["1", "1", "0"],
        populationRatio: ["33.333", "33.333", "33.333"],
        globalAvgScore: 0.6666666666666666,
        totalRecords: 5,
        pageSize: 12,
        pageIndex: 1,
        onGotoPage: (page: number) => console.log("Go to page:", page)
    }
};

export const WithExtendedData: Story = {
    args: {
        id: "bar-chart-extended",
        xAxisLabels: [
            "[0 - 0]",
            "(0 - 1]",
            "(1 - 2]",
            "(2 - 3]",
            "(3 - 5]",
            "(5 - 8]",
            "(8 - 12]",
            "(12 - 18]",
            "(18 - 29]",
            "(29 - 53]",
            "(53 - 64+]"
        ],
        dataSetTruth: [
            "3.733",
            "0.26",
            "0.42",
            "0.63",
            "0.80",
            "1.022",
            "1.495",
            "2.289",
            "3.332",
            "4.875",
            "11.538"
        ],
        dataSetPred: [
            "0.28",
            "0.13",
            "0.15",
            "0.16",
            "0.20",
            "0.26",
            "0.37",
            "0.55",
            "0.99",
            "2.426",
            "8.573"
        ],
        populationRatio: [
            "6.211",
            "9.173",
            "9.432",
            "6.211",
            "10.374",
            "9.977",
            "9.614",
            "9.355",
            "9.674",
            "10.072",
            "9.908"
        ],
        globalAvgScore: 2.833776702232977,
        totalRecords: 100,
        pageSize: 12,
        pageIndex: 1,
        onGotoPage: (page: number) => console.log("Go to page:", page)
    }
};

export const WithNewData: Story = {
    args: {
        id: "bar-chart-new-data",
        xAxisLabels: [
            "[0.002695 - 0.01619]",
            "(0.01619 - 0.01961]",
            "(0.01961 - 0.02209]",
            "(0.02209 - 0.02439]",
            "(0.02439 - 0.02635]",
            "(0.02635 - 0.0288]",
            "(0.0288 - 0.0316]",
            "(0.0316 - 0.03493]",
            "(0.03493 - 0.04236]",
            "(0.04236 - 0.422]"
        ],
        dataSetTruth: [
            "1.306",
            "2.517",
            "2.765",
            "3.508",
            "3.492",
            "3.148",
            "3.558",
            "3.145",
            "2.918",
            "1.401"
        ],
        dataSetPred: [
            "0.52",
            "1.00",
            "1.242",
            "1.760",
            "1.730",
            "1.885",
            "2.030",
            "1.866",
            "1.758",
            "0.71"
        ],
        populationRatio: [
            "10.002",
            "10.002",
            "10.002",
            "10.020",
            "9.974",
            "10.306",
            "9.689",
            "10.002",
            "10.002",
            "10.002"
        ],
        globalAvgScore: 2.774414475855242,
        totalRecords: 100,
        pageSize: 12,
        pageIndex: 1,
        onGotoPage: (page: number) => console.log("Go to page:", page)
    }
};

export const WithNegativeData: Story = {
    args: {
        id: "bar-chart-new-data",
        xAxisLabels: [
            "[0.002695 - 0.01619]",
            "(0.01619 - 0.01961]",
            "(0.01961 - 0.02209]",
            "(0.02209 - 0.02439]",
            "(0.04236 - 0.422]"
        ],
        dataSetTruth: ["-1.306", "-2.517", "-2.765", "-3.508", "-3.492"],
        dataSetPred: ["-0.52", "-1.00", "-1.242", "-1.760", "-1.730"],
        populationRatio: ["10.002", "10.002", "10.002", "10.020", "9.974"],
        globalAvgScore: 2.4,
        totalRecords: 100,
        pageSize: 12,
        pageIndex: 1,
        onGotoPage: (page: number) => console.log("Go to page:", page)
    }
};

export const WithPositiveData: Story = {
    args: {
        id: "bar-chart-new-data",
        xAxisLabels: [
            "[0.002695 - 0.01619]",
            "(0.01619 - 0.01961]",
            "(0.01961 - 0.02209]",
            "(0.02209 - 0.02439]",
            "(0.04236 - 0.422]"
        ],
        dataSetTruth: ["1.306", "2.517", "2.765", "3.508", "3.492"],
        dataSetPred: ["0.52", "1.00", "1.242", "1.760", "1.730"],
        populationRatio: ["10.002", "10.002", "10.002", "10.020", "9.974"],
        globalAvgScore: 0,
        totalRecords: 100,
        pageSize: 12,
        pageIndex: 1,
        onGotoPage: (page: number) => console.log("Go to page:", page)
    }
};

export const WithBothData: Story = {
    args: {
        id: "bar-chart-new-data",
        xAxisLabels: [
            "[0.002695 - 0.01619]",
            "(0.01619 - 0.01961]",
            "(0.01961 - 0.02209]",
            "(0.02209 - 0.02439]",
            "(0.04236 - 0.422]"
        ],
        dataSetTruth: ["0.306", "-2.517", "-2.765", "-3.508", "-3.492"],
        dataSetPred: ["0.52", "-1.00", "-1.242", "-1.760", "-1.730"],
        populationRatio: ["10.002", "10.002", "10.002", "10.020", "9.974"],
        globalAvgScore: 1,
        totalRecords: 100,
        pageSize: 12,
        pageIndex: 1,
        onGotoPage: (page: number) => console.log("Go to page:", page)
    }
};
