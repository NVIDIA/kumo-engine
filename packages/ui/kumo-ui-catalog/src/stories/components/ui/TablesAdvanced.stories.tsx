// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable @typescript-eslint/no-explicit-any */
// src/stories/AdvancedDataTable.stories.tsx
import React, { useState } from "react";
import { createSeededRandom } from "@/stories/utils/seeded-random";
import { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import {
    ColumnDef,
    ExpandedState,
    SortingState,
    VisibilityState,
    flexRender,
    getCoreRowModel,
    getExpandedRowModel,
    getPaginationRowModel,
    getSortedRowModel,
    useReactTable
} from "@tanstack/react-table";
import { ChevronDown, ChevronRight, Edit, Save, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow
} from "@/components/ui/table";
/*import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue
} from "@/components/ui/singleselect";*/
import { Card, CardContent } from "@/components/ui/card";

// Define the data types
type Order = {
    id: string;
    customer: string;
    date: string;
    status: "pending" | "processing" | "shipped" | "delivered" | "cancelled";
    total: number;
    items: OrderItem[];
};

type OrderItem = {
    id: string;
    product: string;
    quantity: number;
    price: number;
};

// Sample data
const orders: Order[] = [
    {
        id: "ord_001",
        customer: "John Doe",
        date: "2023-09-15",
        status: "delivered",
        total: 156.99,
        items: [
            {
                id: "item_001",
                product: "Mechanical Keyboard",
                quantity: 1,
                price: 89.99
            },
            {
                id: "item_002",
                product: "Mouse Pad",
                quantity: 1,
                price: 12.0
            },
            {
                id: "item_003",
                product: "USB Cable",
                quantity: 2,
                price: 27.5
            }
        ]
    },
    {
        id: "ord_002",
        customer: "Jane Smith",
        date: "2023-09-16",
        status: "shipped",
        total: 349.95,
        items: [
            {
                id: "item_004",
                product: 'Monitor 24"',
                quantity: 1,
                price: 249.99
            },
            {
                id: "item_005",
                product: "HDMI Cable",
                quantity: 1,
                price: 14.99
            },
            {
                id: "item_006",
                product: "Webcam HD",
                quantity: 1,
                price: 84.97
            }
        ]
    },
    {
        id: "ord_003",
        customer: "Robert Johnson",
        date: "2023-09-18",
        status: "processing",
        total: 1299.99,
        items: [
            {
                id: "item_007",
                product: "Laptop Pro",
                quantity: 1,
                price: 1199.99
            },
            {
                id: "item_008",
                product: "Laptop Case",
                quantity: 1,
                price: 49.99
            },
            {
                id: "item_009",
                product: "Screen Protector",
                quantity: 1,
                price: 24.99
            },
            {
                id: "item_010",
                product: "Mouse",
                quantity: 1,
                price: 25.02
            }
        ]
    },
    {
        id: "ord_004",
        customer: "Emily Davis",
        date: "2023-09-20",
        status: "pending",
        total: 199.95,
        items: [
            {
                id: "item_011",
                product: "Bluetooth Speakers",
                quantity: 1,
                price: 179.95
            },
            {
                id: "item_012",
                product: "Audio Cable",
                quantity: 1,
                price: 20.0
            }
        ]
    },
    {
        id: "ord_005",
        customer: "Michael Brown",
        date: "2023-09-21",
        status: "cancelled",
        total: 59.99,
        items: [
            {
                id: "item_013",
                product: "Phone Case",
                quantity: 1,
                price: 29.99
            },
            {
                id: "item_014",
                product: "Screen Cleaner",
                quantity: 1,
                price: 9.99
            },
            {
                id: "item_015",
                product: "USB Adapter",
                quantity: 1,
                price: 20.01
            }
        ]
    }
];

// Expandable row component
const OrderItemsTable = ({
    items,
    orderId
}: {
    items: OrderItem[];
    orderId: string;
}) => {
    return (
        <Card className="mx-auto w-full">
            <CardContent className="p-4">
                <h3 className="mb-2 font-medium">Order Items</h3>
                <Table containerLabel={`Order items for ${orderId}`}>
                    <TableHeader>
                        <TableRow>
                            <TableHead>Product</TableHead>
                            <TableHead className="text-right">
                                Quantity
                            </TableHead>
                            <TableHead className="text-right">
                                Unit Price
                            </TableHead>
                            <TableHead className="text-right">
                                Subtotal
                            </TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {items.map((item) => (
                            <TableRow key={item.id}>
                                <TableCell>{item.product}</TableCell>
                                <TableCell className="text-right">
                                    {item.quantity}
                                </TableCell>
                                <TableCell className="text-right">
                                    ${item.price.toFixed(2)}
                                </TableCell>
                                <TableCell className="text-right font-medium">
                                    ${(item.quantity * item.price).toFixed(2)}
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </CardContent>
        </Card>
    );
};

// Editable cell component
interface EditableCellProps {
    getValue: () => any;
    row: any;
    column: any;
    table: any;
}

const EditableCell = ({ getValue, row, column, table }: EditableCellProps) => {
    const initialValue = getValue();
    const [value, setValue] = useState(initialValue);
    const [isEditing, setIsEditing] = useState(false);

    const onSave = () => {
        table.options.meta?.updateData(row.index, column.id, value);
        setIsEditing(false);
    };

    const onCancel = () => {
        setValue(initialValue);
        setIsEditing(false);
    };

    if (!isEditing) {
        return (
            <div className="flex items-center">
                <span>{getValue()}</span>
                <Button
                    variant="light-outline"
                    size="sm"
                    className="ml-2 h-6 w-6 p-0"
                    aria-label="Edit"
                    onClick={() => setIsEditing(true)}>
                    <Edit className="h-3 w-3" />
                </Button>
            </div>
        );
    }

    return (
        <div className="flex items-center space-x-2">
            <Input
                value={value}
                onChange={(e) => setValue(e.target.value)}
                className="h-8"
                aria-label="Edit value"
            />
            <Button
                variant="light-outline"
                size="sm"
                className="h-6 w-6 p-0"
                aria-label="Save"
                onClick={onSave}>
                <Save className="h-3 w-3" />
            </Button>
            <Button
                variant="light-outline"
                size="sm"
                className="h-6 w-6 p-0"
                aria-label="Cancel"
                onClick={onCancel}>
                <X className="h-3 w-3" />
            </Button>
        </div>
    );
};

// Status cell component
/*interface StatusCellProps {
    getValue: () => string;
    row: any;
    column: any;
    table: any;
}

const StatusCell = ({ getValue, row, column, table }: StatusCellProps) => {
    const status = getValue() as Order["status"];
    const [value, setValue] = useState(status);
    const [isEditing, setIsEditing] = useState(false);

    const onSave = () => {
        table.options.meta?.updateData(row.index, column.id, value);
        setIsEditing(false);
    };

    const onCancel = () => {
        setValue(status);
        setIsEditing(false);
    };

    const getStatusColor = (status: Order["status"]) => {
        switch (status) {
            case "delivered":
                return "bg-green-100 text-green-800";
            case "shipped":
                return "bg-blue-100 text-blue-800";
            case "processing":
                return "bg-yellow-100 text-yellow-800";
            case "pending":
                return "bg-purple-100 text-purple-800";
            case "cancelled":
                return "bg-red-100 text-red-800";
            default:
                return "bg-gray-100 text-gray-800";
        }
    };

    if (!isEditing) {
        return (
            <div className="flex items-center">
                <span
                    className={`rounded-full px-2 py-1 text-xs font-medium ${getStatusColor(
                        status
                    )}`}>
                    {status.charAt(0).toUpperCase() + status.slice(1)}
                </span>
                <Button
                    variant="outline"
                    size="sm"
                    className="ml-2 h-6 w-6 p-0"
                    onClick={() => setIsEditing(true)}>
                    <Edit className="h-3 w-3" />
                </Button>
            </div>
        );
    }

    return (
        <div className="flex items-center space-x-2">
            <Select
                value={value}
                onValueChange={(val) => setValue(val as Order["status"])}>
                <SelectTrigger className="h-8 w-32">
                    <SelectValue placeholder="Select status" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="pending">Pending</SelectItem>
                    <SelectItem value="processing">Processing</SelectItem>
                    <SelectItem value="shipped">Shipped</SelectItem>
                    <SelectItem value="delivered">Delivered</SelectItem>
                    <SelectItem value="cancelled">Cancelled</SelectItem>
                </SelectContent>
            </Select>
            <Button
                variant="outline"
                size="sm"
                className="h-6 w-6 p-0"
                onClick={onSave}>
                <Save className="h-3 w-3" />
            </Button>
            <Button
                variant="outline"
                size="sm"
                className="h-6 w-6 p-0"
                onClick={onCancel}>
                <X className="h-3 w-3" />
            </Button>
        </div>
    );
};*/

// Define the columns
const columns: ColumnDef<Order>[] = [
    {
        id: "expander",
        header: "Expand",
        cell: ({ row }) => {
            return (
                <Button
                    variant="light-outline"
                    onClick={() => row.toggleExpanded()}
                    aria-label={
                        row.getIsExpanded() ? "Collapse row" : "Expand row"
                    }
                    className="h-6 w-6 p-0">
                    {row.getIsExpanded() ? (
                        <ChevronDown className="h-4 w-4" />
                    ) : (
                        <ChevronRight className="h-4 w-4" />
                    )}
                </Button>
            );
        }
    },
    {
        accessorKey: "id",
        header: "Order ID"
    },
    {
        accessorKey: "customer",
        header: "Customer",
        cell: EditableCell
    },
    {
        accessorKey: "date",
        header: "Date"
    },
    {
        accessorKey: "status",
        header: "Status"
        //cell: StatusCell
    },
    {
        accessorKey: "total",
        header: "Total",
        cell: ({ row }) => {
            const amount = parseFloat(row.getValue("total"));
            const formatted = new Intl.NumberFormat("en-US", {
                style: "currency",
                currency: "USD"
            }).format(amount);

            return <div className="text-right font-medium">{formatted}</div>;
        }
    }
];

// Define the AdvancedDataTable component
interface AdvancedDataTableProps {
    data: Order[];
    columns: ColumnDef<Order>[];
}

function AdvancedDataTable({ data, columns }: AdvancedDataTableProps) {
    const [tableData, setTableData] = useState<Order[]>(() => [...data]);
    const [expanded, setExpanded] = useState<ExpandedState>({});
    const [sorting, setSorting] = useState<SortingState>([]);
    const [columnVisibility, setColumnVisibility] = useState<VisibilityState>(
        {}
    );

    const table = useReactTable({
        data: tableData,
        columns,
        state: {
            expanded,
            sorting,
            columnVisibility
        },
        onExpandedChange: setExpanded,
        onSortingChange: setSorting,
        onColumnVisibilityChange: setColumnVisibility,
        getCoreRowModel: getCoreRowModel(),
        getExpandedRowModel: getExpandedRowModel(),
        getSortedRowModel: getSortedRowModel(),
        getPaginationRowModel: getPaginationRowModel(),
        meta: {
            updateData: (rowIndex: number, columnId: string, value: any) => {
                setTableData((old) =>
                    old.map((row, index) => {
                        if (index === rowIndex) {
                            return {
                                ...old[rowIndex],
                                [columnId]: value
                            };
                        }
                        return row;
                    })
                );
            }
        }
    });

    return (
        <div className="w-full">
            <div className="rounded-md border">
                <Table containerLabel="Orders table">
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
                                <React.Fragment key={row.id}>
                                    <TableRow>
                                        {row.getVisibleCells().map((cell) => (
                                            <TableCell key={cell.id}>
                                                {flexRender(
                                                    cell.column.columnDef.cell,
                                                    cell.getContext()
                                                )}
                                            </TableCell>
                                        ))}
                                    </TableRow>
                                    {row.getIsExpanded() && (
                                        <TableRow>
                                            <TableCell
                                                colSpan={columns.length}
                                                className="p-0">
                                                <div className="p-4">
                                                    <OrderItemsTable
                                                        items={
                                                            row.original.items
                                                        }
                                                        orderId={
                                                            row.original.id
                                                        }
                                                    />
                                                </div>
                                            </TableCell>
                                        </TableRow>
                                    )}
                                </React.Fragment>
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

// Create the story component
function AdvancedDataTableStory() {
    return <AdvancedDataTable columns={columns} data={orders} />;
}

// Meta information for the Storybook story
const meta: Meta<typeof AdvancedDataTableStory> = {
    title: "UI/AdvancedDataTable",
    component: AdvancedDataTableStory,
    parameters: {
        layout: "padded"
    },
    tags: ["autodocs"]
};

export default meta;
type Story = StoryObj<typeof AdvancedDataTableStory>;

// Define stories
export const Default: Story = {
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);

        const expandBtn = canvas.getAllByRole("button", {
            name: /expand row/i
        })[0];
        await userEvent.click(expandBtn);

        const expectedItems = ["Mechanical Keyboard", "Mouse Pad", "USB Cable"];
        for (const item of expectedItems) {
            await expect(canvas.getByText(item)).toBeVisible();
        }

        const collapseBtn = canvas.getByRole("button", {
            name: /collapse row/i
        });
        await userEvent.click(collapseBtn);
        for (const item of expectedItems) {
            expect(canvas.queryByText(item)).toBeNull();
        }

        const editBtns = canvas.getAllByRole("button", { name: /edit/i });
        await userEvent.click(editBtns[0]);

        const editInput = canvas.getByRole("textbox", {
            name: /edit value/i
        });
        await userEvent.clear(editInput);
        await userEvent.type(editInput, "John Updated");

        const saveBtn = canvas.getByRole("button", { name: /save/i });
        await userEvent.click(saveBtn);
        await expect(canvas.getByText("John Updated")).toBeVisible();

        const editBtnsAfterSave = canvas.getAllByRole("button", {
            name: /edit/i
        });
        await userEvent.click(editBtnsAfterSave[0]);

        const editInputAgain = canvas.getByRole("textbox", {
            name: /edit value/i
        });
        await userEvent.clear(editInputAgain);
        await userEvent.type(editInputAgain, "Should Not Save");

        const cancelBtn = canvas.getByRole("button", { name: /cancel/i });
        await userEvent.click(cancelBtn);
        await expect(canvas.getByText("John Updated")).toBeVisible();
        expect(canvas.queryByText("Should Not Save")).toBeNull();
    }
};

export const AllExpanded: Story = {
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("John Doe")).toBeVisible();
        await expect(
            canvas.getAllByRole("button", { name: /collapse row/i }).length
        ).toBeGreaterThan(0);
        await expect(canvas.getByText("Mechanical Keyboard")).toBeVisible();
    },
    render: () => {
        // Create a component with all rows expanded by default
        const ExpandedTable = () => {
            const [expanded, setExpanded] = useState<ExpandedState>({
                0: true,
                1: true,
                2: true,
                3: true,
                4: true
            });

            const table = useReactTable({
                data: orders,
                columns,
                state: {
                    expanded
                },
                onExpandedChange: setExpanded,
                getCoreRowModel: getCoreRowModel(),
                getExpandedRowModel: getExpandedRowModel()
            });

            return (
                <div className="w-full">
                    <div className="rounded-md border">
                        <Table containerLabel="All expanded orders table">
                            <TableHeader>
                                {table.getHeaderGroups().map((headerGroup) => (
                                    <TableRow key={headerGroup.id}>
                                        {headerGroup.headers.map((header) => (
                                            <TableHead key={header.id}>
                                                {header.isPlaceholder
                                                    ? null
                                                    : flexRender(
                                                          header.column
                                                              .columnDef.header,
                                                          header.getContext()
                                                      )}
                                            </TableHead>
                                        ))}
                                    </TableRow>
                                ))}
                            </TableHeader>
                            <TableBody>
                                {table.getRowModel().rows.map((row) => (
                                    <React.Fragment key={row.id}>
                                        <TableRow>
                                            {row
                                                .getVisibleCells()
                                                .map((cell) => (
                                                    <TableCell key={cell.id}>
                                                        {flexRender(
                                                            cell.column
                                                                .columnDef.cell,
                                                            cell.getContext()
                                                        )}
                                                    </TableCell>
                                                ))}
                                        </TableRow>
                                        {row.getIsExpanded() && (
                                            <TableRow>
                                                <TableCell
                                                    colSpan={columns.length}
                                                    className="p-0">
                                                    <div className="p-4">
                                                        <OrderItemsTable
                                                            items={
                                                                row.original
                                                                    .items
                                                            }
                                                            orderId={
                                                                row.original.id
                                                            }
                                                        />
                                                    </div>
                                                </TableCell>
                                            </TableRow>
                                        )}
                                    </React.Fragment>
                                ))}
                            </TableBody>
                        </Table>
                    </div>
                </div>
            );
        };

        return <ExpandedTable />;
    }
};

export const FilteredByStatus: Story = {
    render: () => {
        // Create a story with only delivered and shipped orders
        const filteredOrders = orders.filter(
            (order) =>
                order.status === "delivered" || order.status === "shipped"
        );
        return <AdvancedDataTable columns={columns} data={filteredOrders} />;
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("John Doe")).toBeVisible();
        await expect(canvas.getByText("Jane Smith")).toBeVisible();
        await expect(
            canvas.queryByText("Robert Johnson")
        ).not.toBeInTheDocument();
    }
};

export const LargeDataset: Story = {
    render: () => {
        // Generate a larger dataset
        const statuses: Order["status"][] = [
            "pending",
            "processing",
            "shipped",
            "delivered",
            "cancelled"
        ];

        const products = [
            "Laptop",
            "Monitor",
            "Keyboard",
            "Mouse",
            "Headphones",
            "Webcam",
            "Speakers",
            "USB Drive",
            "External HDD",
            "Phone Case"
        ];

        const rng = createSeededRandom();
        const largeDataset: Order[] = Array.from({ length: 100 }, (_, i) => {
            const itemCount = Math.floor(rng() * 5) + 1;
            const items: OrderItem[] = Array.from(
                { length: itemCount },
                (_, j) => {
                    const price = Math.floor(rng() * 200) + 10;
                    const quantity = Math.floor(rng() * 3) + 1;
                    return {
                        id: `item_${i}_${j}`,
                        product: `${
                            products[Math.floor(rng() * products.length)]
                        } (${i}-${j})`,
                        quantity,
                        price
                    };
                }
            );

            const total = items.reduce(
                (sum, item) => sum + item.price * item.quantity,
                0
            );

            // new Date() is frozen to 2024-06-15 via vi.setSystemTime in the test
            const date = new Date();
            date.setDate(date.getDate() - Math.floor(rng() * 30));
            const dateStr = date.toISOString().split("T")[0];

            return {
                id: `ord_${String(i + 1).padStart(3, "0")}`,
                customer: `Customer ${i + 1}`,
                date: dateStr,
                status: statuses[Math.floor(rng() * statuses.length)],
                total,
                items
            };
        });

        return <AdvancedDataTable columns={columns} data={largeDataset} />;
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await expect(canvas.getByText("Customer 1")).toBeVisible();
        const expandBtn = canvas.getAllByRole("button", {
            name: /expand row/i
        })[0];
        await userEvent.click(expandBtn);
        await waitFor(() => {
            expect(
                canvas.getByRole("button", { name: /collapse row/i })
            ).toBeVisible();
        });
    }
};
