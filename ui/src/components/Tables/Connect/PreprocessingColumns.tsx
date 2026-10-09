// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    CellContext,
    Checkbox,
    ColumnDef,
    GridCol,
    Row,
    Select,
    Table,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { DropdownOption } from "types/inputs";
import { ColumnArgs } from "types/tables";

import { ChangeColumnType } from "./ColumnSelectionTable";
import { SampleCell } from "./SampleCell";

export interface PreprocessingColumnsProps {
    columnDTypes: DropdownOption[];
    columnSTypes: DropdownOption[];
    dTypeOptions: DropdownOption[];
    showPreview: boolean;
    onCreateTimeColSelect: (
        rowIndex: number,
        cellIndex: number,
        value: ColumnArgs
    ) => void;
    onEndTimeColSelect: (
        rowIndex: number,
        cellIndex: number,
        value: ColumnArgs
    ) => void;
    onSelectPrimaryKey: (index: number, rowData: ColumnArgs) => void;
    onSetColumnDataType: (e: DropdownOption, rowIndex: number) => void;
    onSetColumnSType: (e: DropdownOption, rowIndex: number) => void;
    preservedPageIndex: number;
    defaultPageSize: number;
    selected: Array<string>;
    sTypeOptions: DropdownOption[][];
}

export const PreprocessingColumns = ({
    columnDTypes,
    columnSTypes,
    dTypeOptions,
    showPreview,
    onCreateTimeColSelect,
    onEndTimeColSelect,
    onSelectPrimaryKey,
    onSetColumnDataType,
    onSetColumnSType,
    preservedPageIndex,
    defaultPageSize,
    selected,
    sTypeOptions
}: PreprocessingColumnsProps): ColumnDef<ColumnArgs, string>[] => {
    const previewColumn: ColumnDef<ColumnArgs, string> = {
        id: "preview",
        header: () => (
            <Tooltip id={"Preview"}>
                <TooltipTrigger>
                    <span>Preview</span>
                </TooltipTrigger>
                <TooltipContent
                    arrow={false}
                    sideOffset={5}
                    className="max-w-112.5 text-justify">
                    {"This shows you an example of raw data from this column " +
                        "to better inform type decisions"}
                </TooltipContent>
            </Tooltip>
        ),
        size:
            window.innerWidth > 1900
                ? Math.floor((window.innerWidth * 37) / 100)
                : Math.floor((window.innerWidth * 30) / 100),
        cell: ({ row }: { row: Row<ColumnArgs> }) => <SampleCell row={row} />
    };

    const baseColumns: ColumnDef<ColumnArgs, string>[] = [
        {
            id: "select",
            header: ({ table }: { table: Table<ColumnArgs> }) => (
                <Checkbox
                    checked={
                        table.getIsAllRowsSelected() ||
                        (table.getIsSomeRowsSelected() && "indeterminate")
                    }
                    onCheckedChange={(value: boolean) =>
                        table.toggleAllRowsSelected(!!value)
                    }
                    aria-label="Select all"
                    className="cursor-pointer"
                />
            ),
            cell: ({ row }: { row: Row<ColumnArgs> }) => (
                <Checkbox
                    checked={row.getIsSelected()}
                    onCheckedChange={(value: boolean) =>
                        row.toggleSelected(!!value)
                    }
                    aria-label="Select row"
                    className="cursor-pointer"
                />
            ),
            size: 30
        },
        {
            id: "name",
            header: () => (
                <Tooltip id={"Column-Name"}>
                    <TooltipTrigger>
                        <span>Column Name</span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} sideOffset={5}>
                        {
                            "This is the name of each column from your data source"
                        }
                    </TooltipContent>
                </Tooltip>
            ),
            size: Math.floor((window.innerWidth * 16) / 100),
            accessorKey: "name",
            cell: ({ row, getValue }: CellContext<ColumnArgs, string>) => (
                <div
                    data-testid={`column-name-${row.index}`}
                    className="text-black">
                    {getValue()}
                </div>
            )
        },
        {
            id: "data_type",
            header: () => (
                <Tooltip id={"Data-Type"}>
                    <TooltipTrigger>
                        <span>Data Type</span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} sideOffset={5}>
                        {
                            "The Data Type represents how the data is physically stored"
                        }
                    </TooltipContent>
                </Tooltip>
            ),
            size: Math.floor((window.innerWidth * 8) / 100),

            cell: ({ row }: { row: Row<ColumnArgs> }) => {
                const originalIndex =
                    preservedPageIndex * defaultPageSize +
                    (row.index as number);

                return (
                    <div className="text-black" id={`data-type-${row?.index}`}>
                        <ChangeColumnType
                            selected={columnDTypes[originalIndex]}
                            isDisabled={
                                selected[originalIndex] !== "true" ||
                                row?.original?.disableSelection === true
                            }
                            onChange={(e) => {
                                onSetColumnDataType(e, originalIndex);
                            }}
                            dType={true}
                            options={dTypeOptions}
                        />
                    </div>
                );
            },
            sortingFn: (a: Row<ColumnArgs>, b: Row<ColumnArgs>) => {
                return (a?.original?.dtype as string).localeCompare(
                    b?.original?.dtype as string
                );
            }
        },
        {
            id: "semantic_type",
            header: () => (
                <Tooltip id={"Semantic-Type"}>
                    <TooltipTrigger>
                        <span>Semantic Type</span>
                    </TooltipTrigger>
                    <TooltipContent
                        arrow={false}
                        sideOffset={5}
                        className="max-w-112.5 text-justify">
                        {
                            "The Semantic Type determines how data is processed and encoded"
                        }
                    </TooltipContent>
                </Tooltip>
            ),
            size: Math.floor((window.innerWidth * 8) / 100),
            cell: ({ row }: { row: Row<ColumnArgs> }) => {
                const originalIndex =
                    preservedPageIndex * defaultPageSize +
                    (row.index as number);

                return (
                    <div id={`type-${row?.index}`}>
                        <ChangeColumnType
                            selected={
                                columnSTypes[originalIndex]?.label ===
                                "Sequence"
                                    ? {
                                          label: "Embedding",
                                          value: columnSTypes[originalIndex]
                                              ?.value
                                      }
                                    : columnSTypes[originalIndex]
                            }
                            isDisabled={
                                selected[originalIndex] !== "true" ||
                                row?.original?.disableSelection === true
                            }
                            onChange={(e) => {
                                onSetColumnSType(e, originalIndex);
                            }}
                            options={sTypeOptions[originalIndex]}
                        />
                    </div>
                );
            },
            sortingFn: (a: Row<ColumnArgs>, b: Row<ColumnArgs>) => {
                return (a?.original?.stype as string).localeCompare(
                    b?.original?.stype as string
                );
            }
        },
        {
            id: "properties",
            header: () => (
                <Tooltip id={"Properties"}>
                    <TooltipTrigger>
                        <span>Properties</span>
                    </TooltipTrigger>
                    <TooltipContent arrow={false} sideOffset={5}>
                        {
                            "Select a property for this column: Primary Key, Create Date, or End Date."
                        }
                    </TooltipContent>
                </Tooltip>
            ),
            size: Math.floor((window.innerWidth * 8) / 100),

            cell: ({ row }: { row: Row<ColumnArgs> }) => {
                const originalIndex =
                    preservedPageIndex * defaultPageSize +
                    (row.index as number);
                const defaultvalue = { value: "Select", label: "Select" };

                return (
                    <GridCol xs={12} sm={12}>
                        <Select
                            id={`properties-select-${row?.index}`}
                            placeholder="Select"
                            search={false}
                            mode="single"
                            variant="nooutline"
                            sortOptions={false}
                            popoverClassName="w-38"
                            selectedValues={
                                row?.original?.isPrimary ||
                                row?.original?.isTimeColumn ||
                                row?.original?.isEndTimeColumn
                                    ? Array.of(
                                          row?.original?.isPrimary
                                              ? "primaryKey"
                                              : row?.original?.isTimeColumn
                                                ? "createDate"
                                                : row?.original?.isEndTimeColumn
                                                  ? "endDate"
                                                  : defaultvalue?.value
                                      )
                                    : undefined
                            }
                            options={[
                                {
                                    value: "primaryKey",
                                    label: "Primary Key"
                                },
                                {
                                    value: "createDate",
                                    label: "Create date"
                                },
                                { value: "endDate", label: "End date" }
                            ]}
                            onValuesChange={(newValue) => {
                                const currSel = [
                                    {
                                        value: "primaryKey",
                                        label: "Primary Key"
                                    },
                                    {
                                        value: "createDate",
                                        label: "Create date"
                                    },
                                    { value: "endDate", label: "End date" }
                                ]?.find((item) =>
                                    newValue.includes(item?.value as string)
                                );
                                const selectedValue = currSel as DropdownOption;
                                if (selectedValue.value === "primaryKey") {
                                    onSelectPrimaryKey(
                                        originalIndex,
                                        row?.original
                                    );
                                } else if (
                                    selectedValue.value === "createDate"
                                ) {
                                    onCreateTimeColSelect(
                                        originalIndex,
                                        0,
                                        row?.original
                                    );
                                } else if (selectedValue.value === "endDate") {
                                    onEndTimeColSelect(
                                        originalIndex,

                                        0,
                                        row?.original
                                    );
                                }
                            }}
                            disabled={
                                selected[originalIndex] !== "true" ||
                                row?.original?.disableSelection === true
                            }
                        />
                    </GridCol>
                );
            }
        }
    ];

    return showPreview ? [...baseColumns, previewColumn] : baseColumns;
};
