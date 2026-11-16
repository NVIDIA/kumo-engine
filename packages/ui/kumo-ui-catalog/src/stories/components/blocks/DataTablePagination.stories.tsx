// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DataTablePagination } from "@/components/blocks/data-table/data-table-pagination";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import { useState } from "react";

const meta: Meta<typeof DataTablePagination> = {
    title: "Blocks/DataTablePagination",
    component: DataTablePagination,
    tags: ["autodocs"],
    parameters: {
        layout: "padded"
    }
};

export default meta;
type Story = StoryObj<typeof DataTablePagination>;

export const SmallPageCount: Story = {
    render: function DefaultFunc() {
        const [pageIndex, setPageIndex] = useState(0);
        const pageCount = 5;
        return (
            <DataTablePagination
                canPreviousPage={pageIndex > 0}
                canNextPage={pageIndex < pageCount - 1}
                previousPage={() => setPageIndex((p) => Math.max(0, p - 1))}
                nextPage={() =>
                    setPageIndex((p) => Math.min(pageCount - 1, p + 1))
                }
                pageIndex={pageIndex}
                pageCount={pageCount}
                handlePageChange={(page) => setPageIndex(page - 1)}
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        const page3 = await waitFor(() => canvas.getByTestId("pagination-4"));
        await userEvent.click(page3);
        await userEvent.click(canvas.getByTestId("pagination-next"));
    }
};

export const ManyPagesNearStart: Story = {
    render: function DefaultFunc() {
        const [pageIndex, setPageIndex] = useState(0);
        const pageCount = 15;
        return (
            <DataTablePagination
                canPreviousPage={pageIndex > 0}
                canNextPage={pageIndex < pageCount - 1}
                previousPage={() => setPageIndex((p) => Math.max(0, p - 1))}
                nextPage={() =>
                    setPageIndex((p) => Math.min(pageCount - 1, p + 1))
                }
                pageIndex={pageIndex}
                pageCount={pageCount}
                handlePageChange={(page) => setPageIndex(page - 1)}
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await waitFor(() => {
            expect(canvas.getByRole("navigation")).toBeInTheDocument();
        });
        await userEvent.click(canvas.getByTestId("pagination-6"));
        await userEvent.click(canvas.getByTestId("pagination-next"));
    }
};

export const ManyPagesNearEnd: Story = {
    render: function DefaultFunc() {
        const pageCount = 15;
        const [pageIndex, setPageIndex] = useState(pageCount - 2);
        return (
            <DataTablePagination
                canPreviousPage={pageIndex > 0}
                canNextPage={pageIndex < pageCount - 1}
                previousPage={() => setPageIndex((p) => Math.max(0, p - 1))}
                nextPage={() =>
                    setPageIndex((p) => Math.min(pageCount - 1, p + 1))
                }
                pageIndex={pageIndex}
                pageCount={pageCount}
                handlePageChange={(page) => setPageIndex(page - 1)}
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await waitFor(() => {
            expect(canvas.getByRole("navigation")).toBeInTheDocument();
        });
        await userEvent.click(canvas.getByTestId("pagination-prev"));
        await userEvent.click(canvas.getByTestId("pagination-2"));
    }
};

export const ManyPagesMiddle: Story = {
    render: function DefaultFunc() {
        const pageCount = 20;
        const [pageIndex, setPageIndex] = useState(9);
        return (
            <DataTablePagination
                canPreviousPage={pageIndex > 0}
                canNextPage={pageIndex < pageCount - 1}
                previousPage={() => setPageIndex((p) => Math.max(0, p - 1))}
                nextPage={() =>
                    setPageIndex((p) => Math.min(pageCount - 1, p + 1))
                }
                pageIndex={pageIndex}
                pageCount={pageCount}
                handlePageChange={(page) => setPageIndex(page - 1)}
            />
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await waitFor(() => {
            expect(canvas.getByRole("navigation")).toBeInTheDocument();
        });
        await userEvent.click(canvas.getByTestId("pagination-prev"));
        await userEvent.click(canvas.getByTestId("pagination-next"));
    }
};
