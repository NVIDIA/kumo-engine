// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColumnDef } from "@kumo-ai/kumo-ui-catalog";

export interface ColumnRow {
    name: string;
    colType: string;
    samples?: Array<number | string>;
}

export const OutputTableColumns = (
    areSamplesAvailable: boolean
): ColumnDef<ColumnRow>[] => {
    const columns: ColumnDef<ColumnRow>[] = [
        {
            id: "name",
            header: "Column name",
            accessorKey: "name",
            cell: ({
                row
            }: {
                row: {
                    original: {
                        name: string;
                        colType: string;
                    };
                };
            }) => (
                <div className="text-title-primary text-sm">
                    {row?.original?.name}
                </div>
            ),
            sortingFn: (
                a: {
                    original: { name: string; colType: string };
                },
                b: {
                    original: { name: string; colType: string };
                }
            ): number => {
                try {
                    const valueA: string = a?.original?.name;
                    const valueB: string = b?.original?.name;
                    if (valueA === valueB) {
                        return 0;
                    }
                    return valueA < valueB ? -1 : 1;
                } catch (error) {
                    console.warn("custom sort error", error);
                    return 0;
                }
            }
        },
        {
            id: "colType",
            header: "Data type",
            accessorKey: "",
            cell: ({
                row
            }: {
                row: {
                    original: { name: string; colType: string };
                };
            }) => (
                <div className="whitespace-nowrap">
                    {row?.original?.colType}
                </div>
            )
        }
    ];

    if (areSamplesAvailable) {
        columns.push({
            id: "samples",
            header: "Column samples",
            accessorKey: "samples",
            cell: ({
                row
            }: {
                row: {
                    original: {
                        name: string;
                        colType: string;
                        samples: Array<string | number>;
                    };
                };
            }) => (
                <div className="text-title-primary text-sm">
                    {row?.original?.samples?.join(", ")}
                </div>
            ),
            sortingFn: (
                a: {
                    original: {
                        name: string;
                        colType: string;
                        samples: Array<string | number>;
                    };
                },
                b: {
                    original: {
                        name: string;
                        colType: string;
                        samples: Array<string | number>;
                    };
                }
            ): number => {
                try {
                    const valueA: string = a?.original?.name;
                    const valueB: string = b?.original?.name;
                    if (valueA === valueB) {
                        return 0;
                    }
                    return valueA > valueB ? -1 : 1;
                } catch (error) {
                    console.warn("custom sort error", error);
                    return 0;
                }
            }
        } as ColumnDef<ColumnRow>);
    }

    return columns;
};
