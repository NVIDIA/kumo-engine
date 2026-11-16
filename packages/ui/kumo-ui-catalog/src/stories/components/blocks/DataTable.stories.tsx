// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DataTable } from "@/components/blocks/data-table/data-table";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import { ColumnDef } from "@tanstack/react-table";
import React, { useState } from "react";

type SplitStatsRow = {
    metric: string;
    training: string | number | React.ReactNode;
    validation: string | number | React.ReactNode;
    holdout: string | number | React.ReactNode;
};

const SplitStatsColumns = (): ColumnDef<SplitStatsRow>[] => {
    return [
        {
            accessorKey: "metric",
            header: () => (
                <span className="text-title-secondary text-base">
                    Data Statistics
                </span>
            ),
            cell: ({ row }: { row: { original: SplitStatsRow } }) => (
                <span className="text-title-primary text-base">
                    {row.original.metric}
                </span>
            )
        },
        {
            accessorKey: "holdout",
            header: () => (
                <span className="text-base text-pink-700">Holdout</span>
            ),
            cell: ({ row }: { row: { original: SplitStatsRow } }) => (
                <div className="text-title-primary text-base">
                    {row.original.holdout}
                </div>
            )
        },
        {
            accessorKey: "validation",
            header: () => (
                <span className="text-title-secondary text-base">
                    Validation
                </span>
            ),
            cell: ({ row }: { row: { original: SplitStatsRow } }) => (
                <div className="text-title-primary text-base">
                    {row.original.validation}
                </div>
            )
        },
        {
            accessorKey: "training",
            header: () => (
                <span className="text-title-secondary text-base">Training</span>
            ),
            cell: ({ row }: { row: { original: SplitStatsRow } }) => (
                <div className="text-title-primary text-base">
                    {row.original.training}
                </div>
            )
        }
    ];
};

type MetricRowData = {
    metric_name: string;
    splits: string;
    kumo_model: string;
    max: number;
    cellBgColor?: string;
    rowClassName?: string;
    [key: string]: string | number | null | undefined;
};

const MetricColumns = (): ColumnDef<MetricRowData>[] => [
    {
        header: "Metric_name",
        accessorKey: "metric_name",
        enableSorting: false,
        id: "metric_name"
    },
    {
        header: "Splits",
        accessorKey: "splits",
        enableSorting: false,
        id: "splits"
    },
    {
        header: "Kumo Model",
        accessorKey: "kumo_model",
        enableSorting: false,
        id: "kumo_model"
    },
    {
        header: "Global Frequency",
        accessorKey: "global_frequency",
        enableSorting: false,
        id: "global_frequency"
    },
    {
        header: "Per Entity Frequency",
        accessorKey: "per_entity_frequency",
        enableSorting: false,
        id: "per_entity_frequency"
    },
    {
        header: "Per Entity Last 1 Label",
        accessorKey: "per_entity_last_1_label",
        enableSorting: false,
        id: "per_entity_last_1_label",
        meta: { className: "font-bold" }
    }
];

const metricTableData: MetricRowData[] = [
    {
        metric_name: "AP",
        splits: "Holdout",
        kumo_model: "0.90",
        global_frequency: "0.22",
        per_entity_frequency: "0.56",
        per_entity_last_1_label: "0.42",
        max: 0.9
    },
    {
        metric_name: "AP",
        splits: "Validation",
        kumo_model: "0.91",
        global_frequency: "0.22",
        per_entity_frequency: "0.55",
        per_entity_last_1_label: "0.38",
        max: 0.91
    },
    {
        metric_name: "AP",
        splits: "Training",
        kumo_model: "0.97",
        global_frequency: "-",
        per_entity_frequency: "-",
        per_entity_last_1_label: "-",
        max: 0.97
    },
    {
        metric_name: "AUPRC",
        splits: "Holdout",
        kumo_model: "0.90",
        global_frequency: "0.21",
        per_entity_frequency: "0.60",
        per_entity_last_1_label: "0.62",
        max: 0.9
    },
    {
        metric_name: "AUPRC",
        splits: "Validation",
        kumo_model: "0.91",
        global_frequency: "0.26",
        per_entity_frequency: "0.61",
        per_entity_last_1_label: "0.57",
        max: 0.91
    },
    {
        metric_name: "AUPRC",
        splits: "Training",
        kumo_model: "0.97",
        global_frequency: "-",
        per_entity_frequency: "-",
        per_entity_last_1_label: "-",
        max: 0.97
    },
    {
        metric_name: "AUROC",
        splits: "Holdout",
        kumo_model: "0.75",
        global_frequency: "0.45",
        per_entity_frequency: "0.79",
        per_entity_last_1_label: "0.72",
        max: 0.79
    },
    {
        metric_name: "AUROC",
        splits: "Validation",
        kumo_model: "0.78",
        global_frequency: "0.48",
        per_entity_frequency: "0.76",
        per_entity_last_1_label: "0.68",
        max: 0.78
    },
    {
        metric_name: "AUROC",
        splits: "Training",
        kumo_model: "0.90",
        global_frequency: "-",
        per_entity_frequency: "-",
        per_entity_last_1_label: "-",
        max: 0.9
    }
];

const tableData: SplitStatsRow[] = [
    {
        metric: "Prediction Window",
        training: "2018-11-02 00:00:00 - 2020-04-25 00:00:00",
        validation: "2020-04-25 00:00:00 - 2020-06-24 00:00:00",
        holdout: "2020-06-24 00:00:00 - 2020-09-22 00:00:00"
    },
    {
        metric: "Split Distribution",
        training: "78.26%",
        validation: "8.70%",
        holdout: "13.04%"
    },
    {
        metric: "Total Rows",
        training: 208386,
        validation: 23154,
        holdout: 34731
    },
    {
        metric: "Unique Entities",
        training: 11577,
        validation: 11577,
        holdout: 11577
    }
];

const meta: Meta<typeof DataTable> = {
    title: "Blocks/DataTable",
    component: DataTable,
    tags: ["autodocs"],
    parameters: {
        layout: "padded"
    }
};

export default meta;
type Story = StoryObj<typeof DataTable<SplitStatsRow, unknown>>;

export const DataTable_SplitsStats: Story = {
    name: "Splits Stats",
    render: () => {
        return (
            <DataTable
                id="split_stats_visibility"
                columns={SplitStatsColumns()}
                data={tableData}
                pageCount={1}
                defaultPageSize={50}
                enableMultiSort={false}
                enableGlobalFilter={false}
                columnsVisibility={{
                    holdout: true
                }}
            />
        );
    }
};

export const DataTable_Empty: Story = {
    name: "Empty",
    render: () => {
        return (
            <DataTable
                id="split_stats_empty"
                columns={SplitStatsColumns()}
                data={[]}
                pageCount={1}
                defaultPageSize={50}
                enableMultiSort={false}
                enableGlobalFilter={false}
            />
        );
    }
};

export const DataTable_EvaluationMetrics: Story = {
    name: "Evaluation Metrics",
    render: () => {
        return (
            <DataTable
                id="evaluation_metrics"
                columns={MetricColumns()}
                data={metricTableData}
                pageCount={1}
                defaultPageSize={50}
                enableGlobalFilter={false}
            />
        );
    }
};

export const DataTable_Pagination: Story = {
    name: "Pagination",
    render: () => {
        return (
            <DataTable
                id="evaluation_metrics_paginated"
                columns={MetricColumns()}
                data={metricTableData}
                pageCount={Math.ceil(metricTableData.length / 5)}
                defaultPageSize={5}
                enableGlobalFilter={false}
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const nextBtn = await waitFor(() =>
            canvas.getByTestId("pagination-next")
        );
        await userEvent.click(nextBtn);
        await waitFor(() => {
            expect(canvas.getByTestId("pagination-prev")).toBeInTheDocument();
        });
        await userEvent.click(canvas.getByTestId("pagination-prev"));
    }
};

export const DataTable_Loading: Story = {
    name: "Loading",
    render: () => {
        return (
            <DataTable
                id="loading_table"
                columns={SplitStatsColumns()}
                data={[]}
                pageCount={1}
                defaultPageSize={50}
                enableMultiSort={false}
                enableGlobalFilter={false}
                isLoading={true}
                numSkeletonRow={4}
            />
        );
    }
};

export const DataTable_RowClick: Story = {
    name: "Row Click",
    render: function DefaultFunc() {
        return (
            <DataTable
                id="row_click_table"
                columns={MetricColumns()}
                data={metricTableData}
                pageCount={1}
                defaultPageSize={50}
                enableGlobalFilter={false}
                onRowClick={(row) => {
                    console.log("Row clicked:", row);
                }}
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const rows = await waitFor(() => {
            const r = canvas.getAllByRole("row");
            expect(r.length).toBeGreaterThan(1);
            return r;
        });
        await userEvent.click(rows[1]);
    }
};

export const DataTable_RowExpand: Story = {
    name: "Row Expand",
    render: () => {
        return (
            <DataTable
                id="row_expand_table"
                columns={MetricColumns()}
                data={metricTableData}
                pageCount={1}
                defaultPageSize={50}
                enableGlobalFilter={false}
                rowCanExpand={true}
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const rows = await waitFor(() => {
            const r = canvas.getAllByRole("row");
            expect(r.length).toBeGreaterThan(1);
            return r;
        });
        await userEvent.click(rows[1]);
    }
};

const SplitStatsColumnsWithMaxSize = (): ColumnDef<SplitStatsRow>[] => {
    const cols = SplitStatsColumns();
    return [{ ...cols[0], maxSize: 200 }, ...cols.slice(1)];
};

export const DataTable_FirstColMaxSize: Story = {
    name: "First Col Max Size",
    render: () => {
        return (
            <DataTable
                id="first_col_maxsize_table"
                columns={SplitStatsColumnsWithMaxSize()}
                data={tableData}
                pageCount={1}
                defaultPageSize={50}
                enableMultiSort={false}
                enableGlobalFilter={false}
            />
        );
    }
};

export const DataTable_ManualPagination: Story = {
    name: "Manual Pagination",
    render: function DefaultFunc() {
        const [pageIndex, setPageIndex] = useState(0);
        const pageSize = 3;
        const pageCount = Math.ceil(metricTableData.length / pageSize);
        const pagedData = metricTableData.slice(
            pageIndex * pageSize,
            (pageIndex + 1) * pageSize
        );

        return (
            <DataTable
                id="manual_pagination_table"
                columns={MetricColumns()}
                data={pagedData}
                pageCount={pageCount}
                defaultPageSize={pageSize}
                enableGlobalFilter={false}
                manualPagination={true}
                manualPageIndex={pageIndex}
                handleManualPageChange={(page) => setPageIndex(page)}
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const nextBtn = await waitFor(() =>
            canvas.getByTestId("pagination-next")
        );
        await userEvent.click(nextBtn);
        await waitFor(() => {
            expect(canvas.getByTestId("pagination-prev")).toBeInTheDocument();
        });
    }
};

export const DataTable_GlobalFilter: Story = {
    name: "Global Filter",
    render: () => {
        return (
            <DataTable
                id="global_filter_table"
                columns={MetricColumns()}
                data={metricTableData}
                pageCount={1}
                defaultPageSize={50}
                enableGlobalFilter={true}
                globalFilterPlaceholder="Search metrics..."
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const search = await waitFor(() =>
            canvas.getByPlaceholderText("Search metrics...")
        );
        await userEvent.type(search, "AP");
        await waitFor(() => {
            expect(canvas.getAllByRole("row").length).toBeGreaterThan(1);
        });
    }
};

const StickyColumns = (): ColumnDef<MetricRowData>[] => [
    {
        header: "Metric",
        accessorKey: "metric_name",
        id: "metric_name",
        meta: { sticky: { left: 0 } }
    },
    {
        header: "Splits",
        accessorKey: "splits",
        id: "splits"
    },
    {
        header: "Kumo Model",
        accessorKey: "kumo_model",
        id: "kumo_model"
    },
    {
        header: "Max",
        accessorKey: "max",
        id: "max",
        meta: { sticky: { right: 0 } }
    }
];

export const DataTable_StickyColumns: Story = {
    name: "Sticky Columns",
    render: () => {
        return (
            <DataTable
                id="sticky_columns_table"
                columns={StickyColumns()}
                data={metricTableData}
                pageCount={1}
                defaultPageSize={50}
                enableGlobalFilter={false}
            />
        );
    }
};
