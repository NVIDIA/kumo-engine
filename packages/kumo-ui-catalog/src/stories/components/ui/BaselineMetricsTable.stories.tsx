// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Meta, StoryObj } from "@storybook/react-vite";
import { BaseLineMetricsTable } from "@/components/ui/baseline-metrics-table";
import { Metric } from "@/types/baseline-metrics-table";

const meta: Meta<typeof BaseLineMetricsTable> = {
    title: "UI/BaseLineMetricsTable",
    component: BaseLineMetricsTable
};

export default meta;
type Story = StoryObj<typeof BaseLineMetricsTable>;

const mockData: Metric[] = [
    {
        best_val: 1,
        color_global_frequency: "rgba(18, 158, 30, 1)",
        color_kumo_model: "rgba(18, 158, 30, 1)",
        color_per_entity_frequency: "rgba(18, 158, 30, 1)",
        color_per_entity_last_1_label: "transparent",
        global_frequency: "0.50",
        kumo_model: "1",
        name: "MACRO",
        per_entity_frequency: "0.74",
        per_entity_last_1_label: "1",
        rowSpan: 2,
        spanIndex: 0,
        type: "Average Prediction"
    },
    {
        best_val: 1,
        color_global_frequency: "rgba(18, 158, 30, 1)",
        color_kumo_model: "rgba(18, 158, 30, 0.24)",
        color_per_entity_frequency: "rgba(18, 158, 30, 0.89)",
        color_per_entity_last_1_label: "transparent",
        global_frequency: "0.50",
        kumo_model: "0.81",
        name: "MICRO",
        per_entity_frequency: "0.71",
        per_entity_last_1_label: "1",
        rowSpan: 2,
        spanIndex: 0,
        type: "Average Prediction"
    },
    {
        best_val: 1,
        color_global_frequency: "rgba(18, 158, 30, 1)",
        color_kumo_model: "rgba(18, 158, 30, 1)",
        color_per_entity_frequency: "rgba(18, 158, 30, 0.61)",
        color_per_entity_last_1_label: "transparent",
        global_frequency: "0.50",
        kumo_model: "1",
        name: "MACRO",
        per_entity_frequency: "0.63",
        per_entity_last_1_label: "1",
        rowSpan: 2,
        spanIndex: 0,
        type: "AUPRC"
    },
    {
        best_val: 1,
        color_global_frequency: "rgba(255, 255, 255, 0)",
        color_kumo_model: "rgba(18, 158, 30, 0.2)",
        color_per_entity_frequency: "rgba(255, 255, 255, 0)",
        color_per_entity_last_1_label: "transparent",
        global_frequency: "0.25",
        kumo_model: "0.80",
        name: "MICRO",
        per_entity_frequency: "0.46",
        per_entity_last_1_label: "1",
        rowSpan: 2,
        spanIndex: 0,
        type: "AUPRC"
    },
    {
        best_val: 1,
        color_global_frequency: "rgba(18, 158, 30, 0.32)",
        color_kumo_model: "rgba(18, 158, 30, 1)",
        color_per_entity_frequency: "rgba(18, 158, 30, 0.75)",
        color_per_entity_last_1_label: "transparent",
        global_frequency: "0.33",
        kumo_model: "1",
        name: "MACRO",
        per_entity_frequency: "0.67",
        per_entity_last_1_label: "1",
        rowSpan: 2,
        spanIndex: 0,
        type: "AUROC"
    },
    {
        best_val: 1,
        color_global_frequency: "rgba(18, 158, 30, 0.32)",
        color_kumo_model: "rgba(255, 255, 255, 0)",
        color_per_entity_frequency: "rgba(18, 158, 30, 0.75)",
        color_per_entity_last_1_label: "transparent",
        global_frequency: "0.33",
        kumo_model: "0.75",
        name: "MICRO",
        per_entity_frequency: "0.67",
        per_entity_last_1_label: "1",
        rowSpan: 2,
        spanIndex: 0,
        type: "AUROC"
    }
];

const columns = [
    {
        Header: "Metrics",
        accessor: "type" as keyof Metric,
        colSpan: 2
    },
    {
        Header: "Metrics",
        accessor: "name" as keyof Metric,
        colSpan: 0
    },
    {
        Header: "Kumo Model",
        accessor: "kumo_model" as keyof Metric
    },
    {
        Header: "Global Frequency",
        accessor: "global_frequency" as keyof Metric
    },
    {
        Header: "Per Entity Frequency",
        accessor: "per_entity_frequency" as keyof Metric
    },
    {
        Header: "Per Entity Last 1 Label",
        accessor: "per_entity_last_1_label" as keyof Metric,
        tipText:
            "Predicting each entity with the target value in the previous time frame"
    }
];

export const Default: Story = {
    args: {
        data: mockData,
        columns
    }
};

const f1MapData = [
    {
        name: "Kumo Model",
        ["f1@100"]: "0.004",
        ["map@100"]: "0.025",
        ["precision@100"]: "0.002",
        ["recall@100"]: "0.086",
        ["color_f1@100"]: "rgba(18, 158, 30, 1)",
        ["color_map@100"]: "rgba(18, 158, 30, 1)",
        ["color_precision@100"]: "rgba(18, 158, 30, 1)",
        ["color_recall@100"]: "rgba(18, 158, 30, 1)"
    },
    {
        name: "Global Frequency",
        ["f1@100"]: "0",
        ["map@100"]: "0",
        ["precision@100"]: "0",
        ["recall@100"]: "0.007",
        ["color_f1@100"]: "rgba(255, 255, 255, 0)",
        ["color_map@100"]: "rgba(255, 255, 255, 0)",
        ["color_precision@100"]: "rgba(255, 255, 255, 0)",
        ["color_recall@100"]: "rgba(255, 255, 255, 0)",
        tipText:
            "Predicting all entities with the most popular ranked list of target values in all previous splits"
    },
    {
        name: "Per Entity Frequency",
        ["f1@100"]: "0.002",
        ["map@100"]: "0.009",
        ["precision@100"]: "0.001",
        ["recall@100"]: "0.043",
        ["color_f1@100"]: "rgba(18, 158, 30, 0.5)",
        ["color_map@100"]: "rgba(18, 158, 30, 0.36)",
        ["color_precision@100"]: "rgba(18, 158, 30, 0.5)",
        ["color_recall@100"]: "rgba(18, 158, 30, 0.46)",
        tipText:
            "Predicting each entity with the most popular ranked list of target values for this entity in all previous splits"
    }
];

const f1MapColumns = [
    {
        Header: "Metrics",
        accessor: "name"
    },
    {
        Header: "F1@100",
        accessor: "f1@100"
    },
    {
        Header: "MAP@100",
        accessor: "map@100"
    },
    {
        Header: "PRECISION@100",
        accessor: "precision@100"
    },
    {
        Header: "RECALL@100",
        accessor: "recall@100"
    }
];

export const WithF1MapMetrics: Story = {
    args: {
        data: f1MapData,
        columns: f1MapColumns
    }
};
