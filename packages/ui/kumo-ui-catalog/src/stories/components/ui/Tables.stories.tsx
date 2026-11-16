// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// src/stories/Tables.stories.tsx
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import {
    ColumnDef,
    ColumnFiltersState,
    SortingState,
    VisibilityState,
    flexRender,
    getCoreRowModel,
    getFilteredRowModel,
    getPaginationRowModel,
    getSortedRowModel,
    useReactTable
} from "@tanstack/react-table";
import { ArrowUpDown, ChevronDown, MoreHorizontal } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
    DropdownMenu,
    DropdownMenuCheckboxItem,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow
} from "@/components/ui/table";

import { useState } from "react";

// Define the payment data type
type Payment = {
    id: string;
    amount: number;
    status: "pending" | "processing" | "success" | "failed";
    email: string;
    rowClassName?: string;
};

// Sample data for the table
const data: Payment[] = [
    {
        id: "p_1",
        amount: 100,
        status: "pending",
        email: "user1@example.com"
    },
    {
        id: "p_2",
        amount: 125,
        status: "processing",
        email: "user2@example.com"
    },
    {
        id: "p_3",
        amount: 150,
        status: "success",
        email: "user3@example.com"
    },
    {
        id: "p_4",
        amount: 175,
        status: "failed",
        email: "user4@example.com"
    },
    {
        id: "p_5",
        amount: 200,
        status: "success",
        email: "user5@example.com"
    },
    {
        id: "p_6",
        amount: 225,
        status: "processing",
        email: "user6@example.com"
    },
    {
        id: "p_7",
        amount: 250,
        status: "pending",
        email: "user7@example.com"
    },
    {
        id: "p_8",
        amount: 275,
        status: "failed",
        email: "user8@example.com"
    },
    {
        id: "p_9",
        amount: 300,
        status: "success",
        email: "user9@example.com",
        rowClassName: "text-red"
    },
    {
        id: "p_10",
        amount: 325,
        status: "pending",
        email: "user10@example.com"
    }
];

// Create the TableDef component
interface TableDefProps<TData, TValue> {
    columns: ColumnDef<TData, TValue>[];
    data: TData[];
}

function TableDef<TData, TValue>({
    columns,
    data
}: TableDefProps<TData, TValue>) {
    const [sorting, setSorting] = useState<SortingState>([]);
    const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
    const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(
        {}
    );
    const [rowSelection, setRowSelection] = useState({});

    const table = useReactTable({
        data,
        columns,
        getCoreRowModel: getCoreRowModel(),
        getPaginationRowModel: getPaginationRowModel(),
        onSortingChange: setSorting,
        getSortedRowModel: getSortedRowModel(),
        onColumnFiltersChange: setColumnFilters,
        getFilteredRowModel: getFilteredRowModel(),
        onColumnVisibilityChange: setColumnVisibility,
        onRowSelectionChange: setRowSelection,
        state: {
            sorting,
            columnFilters,
            columnVisibility,
            rowSelection
        }
    });

    return (
        <div className="w-full">
            <div className="flex items-center py-4">
                <Input
                    placeholder="Filter emails..."
                    aria-label="Filter emails"
                    value={
                        (table
                            .getColumn("email")
                            ?.getFilterValue() as string) ?? ""
                    }
                    onChange={(event) =>
                        table
                            .getColumn("email")
                            ?.setFilterValue(event.target.value)
                    }
                    className="max-w-sm"
                />
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button variant="light-outline" className="ml-auto">
                            Columns <ChevronDown className="ml-2 h-4 w-4" />
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                        {table
                            .getAllColumns()
                            .filter((column) => column.getCanHide())
                            .map((column) => {
                                return (
                                    <DropdownMenuCheckboxItem
                                        key={column.id}
                                        className="capitalize"
                                        checked={column.getIsVisible()}
                                        onCheckedChange={(value) =>
                                            column.toggleVisibility(!!value)
                                        }>
                                        {column.id}
                                    </DropdownMenuCheckboxItem>
                                );
                            })}
                    </DropdownMenuContent>
                </DropdownMenu>
            </div>
            <div className="rounded-md border">
                <Table>
                    <TableHeader>
                        {table.getHeaderGroups().map((headerGroup) => (
                            <TableRow key={headerGroup.id}>
                                {headerGroup.headers.map((header) => {
                                    return (
                                        <TableHead key={header.id}>
                                            {header.isPlaceholder
                                                ? null
                                                : flexRender(
                                                      header.column.columnDef
                                                          .header,
                                                      header.getContext()
                                                  )}
                                        </TableHead>
                                    );
                                })}
                            </TableRow>
                        ))}
                    </TableHeader>
                    <TableBody>
                        {table.getRowModel().rows?.length ? (
                            table.getRowModel().rows.map((row) => (
                                <TableRow
                                    key={row.id}
                                    data-state={
                                        row.getIsSelected() && "selected"
                                    }>
                                    {row.getVisibleCells().map((cell) => (
                                        <TableCell key={cell.id}>
                                            {flexRender(
                                                cell.column.columnDef.cell,
                                                cell.getContext()
                                            )}
                                        </TableCell>
                                    ))}
                                </TableRow>
                            ))
                        ) : (
                            <TableRow>
                                <TableCell
                                    colSpan={columns.length}
                                    className="h-24 text-center">
                                    No results.
                                </TableCell>
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
            </div>
            <div className="flex items-center justify-end space-x-2 py-4">
                <div className="text-muted-foreground text-sm">
                    {table.getFilteredSelectedRowModel().rows.length} of{" "}
                    {table.getFilteredRowModel().rows.length} row(s) selected.
                </div>
                <Button
                    variant="light-outline"
                    size="sm"
                    onClick={() => table.previousPage()}
                    disabled={!table.getCanPreviousPage()}>
                    Previous
                </Button>
                <Button
                    variant="light-outline"
                    size="sm"
                    onClick={() => table.nextPage()}
                    disabled={!table.getCanNextPage()}>
                    Next
                </Button>
            </div>
        </div>
    );
}

// Define the columns for the payment data
export const columns: ColumnDef<Payment>[] = [
    {
        id: "select",
        header: ({ table }) => (
            <div className="px-1">
                <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                    checked={table.getIsAllPageRowsSelected()}
                    onChange={(e) =>
                        table.toggleAllPageRowsSelected(!!e.target.checked)
                    }
                    aria-label="Select all"
                />
            </div>
        ),
        cell: ({ row }) => (
            <div className="px-1">
                <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                    checked={row.getIsSelected()}
                    onChange={(e) => row.toggleSelected(!!e.target.checked)}
                    aria-label={`Select row ${row.id}`}
                />
            </div>
        ),
        enableSorting: false,
        enableHiding: false
    },
    {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }) => (
            <div className="capitalize">{row.getValue("status")}</div>
        )
    },
    {
        accessorKey: "email",
        header: ({ column }) => {
            return (
                <Button
                    variant="light-outline"
                    onClick={() =>
                        column.toggleSorting(column.getIsSorted() === "asc")
                    }>
                    Email
                    <ArrowUpDown className="ml-2 h-4 w-4" />
                </Button>
            );
        },
        cell: ({ row }) => <div>{row.getValue("email")}</div>
    },
    {
        accessorKey: "amount",
        header: () => <div className="text-right">Amount</div>,
        cell: ({ row }) => {
            const amount = parseFloat(row.getValue("amount"));
            const formatted = new Intl.NumberFormat("en-US", {
                style: "currency",
                currency: "USD"
            }).format(amount);

            return <div className="text-right font-medium">{formatted}</div>;
        }
    },
    {
        id: "actions",
        header: "Actions",
        enableHiding: false,
        cell: ({ row }) => {
            const payment = row.original;

            return (
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button variant="light-outline" className="h-8 w-8 p-0">
                            <span className="sr-only">Open menu</span>
                            <MoreHorizontal className="h-4 w-4" />
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                        <DropdownMenuLabel>Actions</DropdownMenuLabel>
                        <DropdownMenuItem
                            onClick={() =>
                                navigator.clipboard.writeText(payment.id)
                            }>
                            Copy payment ID
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem>View customer</DropdownMenuItem>
                        <DropdownMenuItem>
                            View payment details
                        </DropdownMenuItem>
                    </DropdownMenuContent>
                </DropdownMenu>
            );
        }
    }
];

// Create the story component that uses TableDef
function TableStory() {
    return <TableDef columns={columns} data={data} />;
}

// Meta information for the Storybook story
const meta: Meta<typeof TableStory> = {
    title: "UI/Tables",
    component: TableStory,
    parameters: {
        layout: "padding"
    },
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof TableStory>;

// Define multiple stories for different states/configurations

export const Paginated: Story = {
    render: () => {
        const paginatedData = Array(100)
            .fill(null)
            .map((_, i) => ({
                id: `p_${i + 1}`,
                amount: 100 + i * 25,
                status: ["pending", "processing", "success", "failed"][
                    i % 4
                ] as Payment["status"],
                email: `user${i + 1}@example.com`
            }));

        return <TableDef columns={columns} data={paginatedData} />;
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const filterInput = canvas.getByPlaceholderText("Filter emails...");
        await userEvent.type(filterInput, "user1@");
        await expect(canvas.getByText("user1@example.com")).toBeVisible();
        expect(canvas.queryByText("user2@example.com")).toBeNull();

        await userEvent.clear(filterInput);

        const rowCheckbox = canvas.getByRole("checkbox", {
            name: /select row 0/i
        });
        await userEvent.click(rowCheckbox);
        await expect(canvas.getByText(/1 of/)).toBeVisible();

        const nextBtn = canvas.getByRole("button", { name: /next/i });
        await expect(nextBtn).toBeEnabled();
        await userEvent.click(nextBtn);

        await expect(canvas.getByText("user11@example.com")).toBeVisible();

        const prevBtn = canvas.getByRole("button", { name: /previous/i });
        await expect(prevBtn).toBeEnabled();
        await userEvent.click(prevBtn);

        await expect(canvas.getByText("user1@example.com")).toBeVisible();

        const emailSortBtn = canvas.getByRole("button", { name: /email/i });
        await userEvent.click(emailSortBtn);

        await waitFor(() => {
            expect(canvas.getByText("user1@example.com")).toBeVisible();
        });
    }
};

export const Filtered: Story = {
    render: () => {
        // Pre-filtered to only show success payments
        const filteredData = data.filter((item) => item.status === "success");
        return <TableDef columns={columns} data={filteredData} />;
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("user3@example.com")).toBeVisible();
        await expect(canvas.getByText("user5@example.com")).toBeVisible();
        await expect(
            canvas.queryByText("user1@example.com")
        ).not.toBeInTheDocument();
    }
};

export const Empty: Story = {
    render: () => {
        return <TableDef columns={columns} data={[]} />;
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("No results.")).toBeVisible();

        await expect(
            canvas.getByRole("button", { name: /previous/i })
        ).toBeDisabled();
        await expect(
            canvas.getByRole("button", { name: /next/i })
        ).toBeDisabled();
    }
};

// Additional story with custom columns
export const CustomColumns: Story = {
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(
            canvas.getByRole("columnheader", { name: /payment id/i })
        ).toBeVisible();
        await expect(
            canvas.getByRole("columnheader", { name: /amount/i })
        ).toBeVisible();
        await expect(
            canvas.getByRole("columnheader", { name: /status/i })
        ).toBeVisible();
    },
    render: () => {
        const customColumns: ColumnDef<Payment>[] = [
            {
                accessorKey: "id",
                header: "Payment ID"
            },
            {
                accessorKey: "amount",
                header: "Amount",
                cell: ({ row }) => {
                    const amount = parseFloat(row.getValue("amount"));
                    return <div>${amount.toFixed(2)}</div>;
                }
            },
            {
                accessorKey: "status",
                header: "Status",
                cell: ({ row }) => {
                    const status = row.getValue("status") as string;
                    return (
                        <div
                            className={`rounded-full px-2 py-1 text-xs font-medium ${
                                status === "success"
                                    ? "bg-green-100 text-green-800"
                                    : status === "processing"
                                      ? "bg-blue-100 text-blue-800"
                                      : status === "pending"
                                        ? "bg-yellow-100 text-yellow-800"
                                        : "bg-red-100 text-red-800"
                            }`}>
                            {status.charAt(0).toUpperCase() + status.slice(1)}
                        </div>
                    );
                }
            }
        ];

        return <TableDef columns={customColumns} data={data} />;
    }
};
